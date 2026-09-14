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

## Generate

```bash
source teachergen/env.sh
python teachergen/gen.py --turn-ids my_ids.txt     # one turn_id per line, or a .json list
```

Writes `teachergen/out/my_ids.jsonl` (one line per turn), `my_ids.meta.json`
(teacher revision, sampling, echo, corpus epoch + manifest sha) and
`my_ids.errors.jsonl`. Reruns skip finished turns and retry failed ones.

Per turn: `turn_id, action_kind, source, stratum, corpus_epoch, prompt_tail,
max_tokens, temperature, n_parsed, rollouts[]`. Per rollout: `raw` (completion
exactly as sampled; the prompt ends inside `<think>`), `reasoning` / `content`
(before / after `</think>`), `z`, `y`, `parsed`, `finish_reason`,
`prompt_tokens`, `completion_tokens`, `think_closed`, `lp_own`, `lp_empty`,
`lp_thought` (null when the rollout has no parseable action). The prefix is not
saved.
