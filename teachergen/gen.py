"""Teacher rollouts for chosen corpus-D turns, sampled the way a live duel
samples its teacher references.

Duel path being mirrored: dueling.score_side -> RefCache.ensure_raw ->
terms.sample_teacher_rollouts -> VllmModel.sample. Per turn:
  prompt      evalsrv.chat.gen_prompt: the teacher's own chat template over
              the materialized prefix, ending inside <think>
  request     /v1/completions, temperature = [duel].temperature,
              max_tokens = max_thought + max_action for the turn's dialect
              (dueling.token_caps), add_special_tokens = False; k =
              [duel].n_teacher_samples independent requests, sticky-routed
              to one replica by turn_id (ModelPool._pick)
  (z, y)      evalsrv.chat.split_rollout with teacher-side semantics
              (require_think_close=False)

Right after sampling, each parsed rollout is echo-scored exactly as the duel
scores its references (RefCache.ensure_scored -> terms.score_teacher_rollouts,
thought_echo on under min_rg): lp_own = lpC(y|z_C), lp_empty = lpC(y|∅),
lp_thought = lpC(z_C|x), all mean logprob per byte from echo+logprobs
teacher forcing, sticky-routed to the replica that sampled the turn.

Differences from the duel, on purpose: unparsable rollouts are kept (y == "",
no echoes, lp_* = null) instead of dropped, and the raw completion is saved
before any parsing.

Output: one JSON line per turn in --out; failures go to <out>.errors.jsonl
and are retried on the next run (turns already in --out are skipped).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import httpx
import orjson

from affine import dialects
from evalsrv.chat import THINK_CLOSE, gen_prompt, split_rollout, think_closed
from evalsrv.corpus import CorpusSync
from evalsrv.dueling import token_caps
from evalsrv.terms import score_teacher_rollouts
from evalsrv.vllm_client import ModelPool, Served, VllmModel

log = logging.getLogger("teachergen")

TG_ROOT = Path(os.environ.get("TG_ROOT", "/dev/shm/affine-teachergen"))


def read_turn_ids(path: Path) -> list[str]:
    text = path.read_text()
    if path.suffix == ".json":
        ids = [str(t) for t in json.loads(text)]
    else:
        ids = [line.strip() for line in text.splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    return list(dict.fromkeys(ids))


def repair_and_load_done(out: Path) -> set[str]:
    """turn_ids already written; drops a torn last line left by a kill."""
    if not out.exists():
        return set()
    data = out.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]
        out.write_bytes(data)
        log.warning("truncated a partial last line in %s", out)
    return {orjson.loads(line)["turn_id"] for line in data.splitlines() if line}


async def sample_raw(teacher: ModelPool, sticky_key: str, prompt: str,
                     temperature: float, max_tokens: int) -> dict:
    """VllmModel.sample's request, but returning the whole response so the
    raw text survives. Same replica pick, semaphore and retries as the duel
    (ModelPool._pick / VllmModel._post)."""
    return await teacher._pick(sticky_key)._post({
        "model": teacher.cfg.request_model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "add_special_tokens": False,
    })


def rollout_record(choice: dict, usage: dict, action_kind: str) -> dict:
    raw = choice["text"]
    closed = think_closed(raw)
    if closed:
        reasoning, _, content = raw.partition(THINK_CLOSE)
    else:
        reasoning, content = raw, ""
    z, y = split_rollout(raw, action_kind)
    return {
        # Completion text exactly as sampled. The prompt ended inside an
        # open <think> (see prompt_tail on the turn), so the reasoning block
        # opens before this text and closes at the first </think> in it.
        "raw": raw,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "think_closed": closed,
        "reasoning": reasoning,
        "content": content,
        "z": z,
        "y": y,
        "parsed": bool(y),
        "lp_own": None,
        "lp_empty": None,
        "lp_thought": None,
    }


async def run(args: argparse.Namespace) -> None:
    contract = tomllib.loads(dialects.CONTRACT_TOML.read_text())
    duel = contract["duel"]
    caps = token_caps(duel)
    temperature = float(duel["temperature"])
    k = args.n_samples or int(duel["n_teacher_samples"])
    repo = os.environ.get("TEACHER_REPO", contract["teacher"]["repo"])
    revision = os.environ.get("TEACHER_REVISION") or None
    thought_echo = str(duel.get("score_mode", "reason")) in ("min_rg", "min_rga")

    corpus = CorpusSync(contract["dataset"]["corpus_base_url"],
                        contract["dataset"]["manifest_key"], args.data_dir)
    if not corpus.ready:
        sys.exit(f"no verified corpus in {args.data_dir}; run sync_corpus.py")
    cinfo = corpus.info()
    by_tid = {r["turn_id"]: r for r in corpus.load_index_rows()}

    wanted = read_turn_ids(args.turn_ids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    err_path = args.out.with_suffix(".errors.jsonl")
    meta_path = args.out.with_suffix(".meta.json")
    meta = {
        "teacher": {"repo": repo, "revision": revision},
        "sampling": {
            "temperature": temperature,
            "n_samples": k,
            "max_tokens_default": sum(caps(None)),
            "max_tokens_by_kind": {kind: list(caps(kind))
                                   for kind in dialects.DIALECTS},
            "add_special_tokens": False,
            "prompt": "evalsrv.chat.gen_prompt",
            "split": "evalsrv.chat.split_rollout(require_think_close=False)",
        },
        "echo": {
            "fn": "evalsrv.terms.score_teacher_rollouts",
            "thought_echo": thought_echo,
            "fields": "lp_own=lpC(y|z_C) lp_empty=lpC(y|empty) "
                      "lp_thought=lpC(z_C|x), mean logprob per byte",
        },
        "corpus": {key: cinfo[key] for key in (
            "corpus_epoch", "manifest_sha256", "schema_version", "view_spec",
            "corpus_base_url")},
        "weight_version_key": contract["subnet"].get("weight_version_key"),
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        for key in ("teacher", "sampling", "echo", "corpus"):
            if old.get(key) != meta[key] and not args.force:
                sys.exit(f"{meta_path} has different {key!r} than this run "
                         f"(old={old.get(key)} new={meta[key]}); use another "
                         "--out or pass --force")
    else:
        meta["created_at"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(meta, indent=2))

    done = repair_and_load_done(args.out)
    missing = [t for t in wanted if t not in by_tid]
    todo = [t for t in wanted if t in by_tid and t not in done]
    log.info("turn_ids=%d already_done=%d not_in_corpus=%d todo=%d | epoch=%s "
             "manifest=%s k=%d T=%s replicas=%d", len(wanted),
             len(done & set(wanted)), len(missing), len(todo),
             cinfo["corpus_epoch"], cinfo["manifest_sha256"][:12], k,
             temperature, len(args.urls))

    out_f = open(args.out, "ab")
    err_f = open(err_path, "ab")

    def write_error(tid: str, error: str) -> None:
        err_f.write(orjson.dumps({
            "turn_id": tid, "error": error,
            "at": datetime.now(timezone.utc).isoformat()}) + b"\n")
        err_f.flush()

    for tid in missing:
        write_error(tid, "turn_id_not_in_corpus")

    queue: asyncio.Queue[str] = asyncio.Queue()
    for tid in todo:
        queue.put_nowait(tid)
    mat_lock = asyncio.Lock()
    stats = {"turns": 0, "errors": 0, "rollouts": 0, "parsed": 0}
    started = time.monotonic()

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=None)) as http:
        # Teacher refs keep require_think_close off, as in run_duel.
        teacher = ModelPool([
            VllmModel(Served(name=f"teacher{i}", repo=repo, revision=revision,
                             port=0, base_url=url),
                      http, asyncio.Semaphore(args.per_replica))
            for i, url in enumerate(args.urls)])

        async def one(tid: str) -> None:
            async with mat_lock:  # chunk parsing is CPU-bound and cached
                rec = (await asyncio.to_thread(
                    corpus.materialize_turns, [by_tid[tid]]))[0]
            kind = rec.get("action_kind") or dialects.DEFAULT_KIND
            max_thought, max_action = caps(kind)
            prompt = await asyncio.to_thread(
                gen_prompt, repo, revision, rec["prefix"])
            ds = await asyncio.gather(*[
                sample_raw(teacher, tid, prompt, temperature,
                           max_thought + max_action)
                for _ in range(k)])
            rollouts = [rollout_record(d["choices"][0], d.get("usage") or {},
                                       kind) for d in ds]
            parsed = [r for r in rollouts if r["parsed"]]
            scored = await score_teacher_rollouts(
                teacher, rec["prefix"], [(r["z"], r["y"]) for r in parsed],
                thought_echo=thought_echo, sticky_key=tid)
            for r, s in zip(parsed, scored):
                r["lp_own"] = s["lp_own"]
                r["lp_empty"] = s["lp_empty"]
                r["lp_thought"] = s.get("lp_thought")
            out_f.write(orjson.dumps({
                "turn_id": tid,
                "action_kind": kind,
                "source": rec.get("source"),
                "stratum": rec.get("stratum"),
                "corpus_epoch": cinfo["corpus_epoch"],
                "prompt_tail": prompt[-64:],
                "max_tokens": max_thought + max_action,
                "temperature": temperature,
                "n_parsed": sum(r["parsed"] for r in rollouts),
                "rollouts": rollouts,
            }) + b"\n")
            out_f.flush()
            stats["rollouts"] += len(rollouts)
            stats["parsed"] += sum(r["parsed"] for r in rollouts)

        async def worker() -> None:
            while True:
                try:
                    tid = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await one(tid)
                except Exception as e:  # noqa: BLE001 — logged, retried next run
                    stats["errors"] += 1
                    write_error(tid, f"{type(e).__name__}: {e}"[:1000])
                stats["turns"] += 1
                if (stats["turns"] % args.log_every == 0
                        or stats["turns"] == len(todo)):
                    el = time.monotonic() - started
                    log.info("%d/%d turns (%.2f turns/s) errors=%d parse=%.3f",
                             stats["turns"], len(todo), stats["turns"] / el,
                             stats["errors"],
                             stats["parsed"] / max(stats["rollouts"], 1))

        await asyncio.gather(*[worker() for _ in range(args.concurrency)])

    out_f.close()
    err_f.close()
    log.info("done: %d turns written, %d errors (see %s)",
             stats["turns"] - stats["errors"], stats["errors"], err_path)


def main() -> None:
    ports = range(8100, 8108)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turn-ids", type=Path, required=True,
                    help="text file (one turn_id per line) or .json list")
    ap.add_argument("--out", type=Path, default=None,
                    help="output jsonl (default teachergen/out/<ids stem>.jsonl)")
    ap.add_argument("--urls", nargs="+",
                    default=os.environ.get("TEACHER_URLS", " ".join(
                        f"http://127.0.0.1:{p}/v1" for p in ports)).split())
    ap.add_argument("--n-samples", type=int, default=0,
                    help="rollouts per turn (default [duel].n_teacher_samples)")
    ap.add_argument("--concurrency", type=int, default=192,
                    help="turns in flight (duel default [duel].concurrency)")
    ap.add_argument("--per-replica", type=int, default=128,
                    help="max in-flight requests per replica")
    ap.add_argument("--data-dir", type=Path, default=TG_ROOT / "corpus")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--force", action="store_true",
                    help="append even if the existing meta differs")
    args = ap.parse_args()
    if args.out is None:
        args.out = Path(__file__).resolve().parent / "out" / f"{args.turn_ids.stem}.jsonl"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
