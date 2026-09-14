# gendata_AFF

Teacher datagen for Affine (SN120) corpus-D turns, sampled and echo-scored
exactly like a live duel scores its teacher references.

For each requested `turn_id` of the latest corpus epoch, the frozen teacher
(`Qwen/Qwen3.8-27B`) samples `[duel].n_teacher_samples` rollouts at
`[duel].temperature` with `max_thought_tokens + max_action_tokens`, through the
eval's own prompt rendering (`evalsrv.chat.gen_prompt`) and z/y split
(`evalsrv.chat.split_rollout`). Right after sampling, every parsed rollout is
echo-scored with `evalsrv.terms.score_teacher_rollouts`:
`lp_own = lpC(y|z)`, `lp_empty = lpC(y|∅)`, `lp_thought = lpC(z|x)` (mean
logprob per byte).

## Layout

```
teachergen/            setup + serving + generation (the part written for this repo)
affine/                vendored subset of github.com/AffineFoundation/affine
                       (affine/ + evalsrv/ modules gen.py imports, affine.toml)
ops/teacher-swarm/echo_cache_plugin/   production vLLM echo prefix-cache plugin
```

The vendored files are unmodified copies; `affine/affine.toml` is the contract
the sampling knobs are read from.

## Setup (8x GPU box)

```bash
cp .env.example .env          # put your HF_TOKEN in it
./teachergen/setup.sh         # gcc toolchain, uv venv (vLLM 0.28.0), weights, corpus
./teachergen/serve_teacher.sh start
./teachergen/serve_teacher.sh status
```

Everything bulky goes to `/dev/shm/affine-teachergen` (`TG_ROOT` in
`teachergen/env.sh`). It is RAM: rerun `setup.sh` after a reboot.

Host notes (Rocky 8, glibc 2.28, no compiler, no sudo):
- `teachergen/overrides.txt` pins `llguidance==1.6.1` (1.7.x wheels need glibc 2.31;
  only guided decoding uses it).
- gcc comes from conda-forge via micromamba; `CC`/`CXX` are exported for Triton.
- On 141 GB H200s at util 0.70 the engine only has 866 Mamba blocks, so replicas
  run with `--max-num-seqs 512`.
- Keep `GPU_UTIL=0.70`. At 0.85 replicas die with CUDA OOM under echo load: the
  echo plugin's prompt-logprobs step (`sampler.compute_logprobs`) allocates a
  ~5 GiB fp32 spike that 0.85 leaves no room for (5/8 H200 replicas crashed in
  ~7 min on 2026-09-14).

Host notes (Ubuntu 22.04, NVIDIA driver 570 / CUDA 12.8):
- vLLM 0.28.0 ships torch cu130, which refuses an R570 driver ("NVIDIA driver on
  your system is too old"). `env.sh` prepends `/usr/local/cuda-13.0/compat`
  (CUDA 13 forward-compat libcuda) to `LD_LIBRARY_PATH` when the driver is
  older than 580.

## Generate

```bash
source teachergen/env.sh
python teachergen/gen.py --turn-ids my_ids.txt     # one turn_id per line, or a .json list
```

Writes `teachergen/out/my_ids.jsonl` (one line per turn), `my_ids.meta.json`
(teacher revision, sampling, echo, corpus epoch + manifest sha) and
`my_ids.errors.jsonl`. Reruns skip finished turns and retry failed ones.

`gen.py` options beyond the duel defaults:
- `--batch-n`: sample the k rollouts as one `n=k` request (one prefill, the k
  decodes share its KV) instead of k requests; `return_token_ids` keeps
  per-rollout `completion_tokens`.
- `--echo thought`: compute only `lp_thought = lpC(z|x)`; `lp_own` / `lp_empty`
  stay null and the meta's `echo` block records it (a dir never mixes modes).
- `--limit N`, `--shard i --num-shards N`: first N ids, then round-robin
  `ids[i::N]`.

### Data-parallel run (one gen.py per replica)

```bash
# shard i -> replica port 8100+i, so a turn's sample + echoes share its prefix KV
teachergen/run_shards.sh start ids.txt teachergen/out/run1 [--echo thought]
teachergen/run_shards.sh status teachergen/out/run1
teachergen/run_shards.sh stop teachergen/out/run1   # start again to resume
```

Writes `shard_{0..7}.{jsonl,meta.json,errors.jsonl,log}`; merge by `turn_id`.
Env: `NUM_SHARDS` (8), `BASE_PORT` (8100), `LIMIT`, `CONCURRENCY` (192),
`PER_REPLICA` (128). Shards run with `HF_HUB_OFFLINE=1`: concurrent first
`gen_prompt` calls race past `get_tokenizer`'s cache and would each hit the hub.

`teachergen/bench_echo.sh ids.txt out_root thought all` measures steady turns/s
per echo mode on the same ids with a cold prefix cache (replicas started with
`VLLM_SERVER_DEV_MODE=1` for `POST /reset_prefix_cache`).

Measured on 8x H200, TP=1, k=3, `--batch-n`, corpus epoch 35 (turn mix: ~65%
bash, ~32% tool_call; mean prompt ~3.5k tokens; mean rollout ~310 tokens):

| echo | GPU_UTIL | turns/s (8 replicas) |
|---|---|---|
| all (lp_own + lp_empty + lp_thought) | 0.85 (OOM-prone) | ~5.1 |
| thought | 0.85 (OOM-prone) | ~10.8 |
| thought | 0.70 | ~5.3–6.2 (different id block; bursty) |

Per turn: `turn_id, action_kind, source, stratum, corpus_epoch, prompt_tail,
max_tokens, temperature, n_parsed, rollouts[]`. Per rollout: `raw` (completion
exactly as sampled; the prompt ends inside `<think>`), `reasoning` / `content`
(before / after `</think>`), `z`, `y`, `parsed`, `finish_reason`,
`prompt_tokens`, `completion_tokens`, `think_closed`, `lp_own`, `lp_empty`,
`lp_thought` (null when the rollout has no parseable action). The prefix is not
saved.
