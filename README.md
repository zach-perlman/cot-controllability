# cot-controllability

Can open reasoning models follow rules about what goes in their chain of thought (e.g. "don't use the word X in
your reasoning")? And if they comply, is anything actually hidden from a monitor? The starting point is measuring
it properly on Qwen3-8B, Qwen3-32B and Qwen3.6-27B with the [CoT-Control](https://arxiv.org/abs/2603.05706) QA
suite (`third_party/CoTControl`, pinned) and the zero-shot prompts from
[Arun Jose's post](https://www.lesswrong.com/posts/BbP2wCyDGdPWJ7PwP/cot-controllability-evals-seem-very-under-elicited).

Results are drafts unless listed in `VERIFIED.md` (see `AGENTS.md` for the evidence norms). Numbers labeled
"judge" come from an LLM judge (Qwen3.8-27B) and are weaker evidence than the programmatic compliance grader.

## Experiments

| id | question | report |
|---|---|---|
| `exp01_paper_gate` | positive control: at the paper's settings (temperature 0, 25k reasoning cap, baseline prompt), does our vLLM setup reproduce the paper's Qwen3-32B CoT controllability (2.3%)? | `results/exp01_paper_gate/` |
| `exp02_prompt_grid` | baseline vs Arun's repetition / self-monitor / stacked prompts on 3 models x 9 modes: compliance, compliance without meta-discussion, compliance at matched CoT length, accuracy, and where the first violation happens | `results/exp02_prompt_grid/` |

Each experiment has a `manifest.json` in its results folder, written before the run.

## Running

Environments: `/venv/main` (analysis; CoT-Control's grader) and `/workspace/.venv-vllm` (generation and judge,
always through `scripts/vllm_python.sh`). Rebuild with `scripts/setup_envs.sh`; check with
`/venv/main/bin/python scripts/check_env.py`.

```bash
cd scripts
/venv/main/bin/python cc_items.py                              # fixed-seed 50 items -> cache/exp02/items.jsonl
/venv/main/bin/python cc_prompts.py requests --exp exp01       # request files (content-addressed ids)
/venv/main/bin/python cc_prompts.py requests --exp exp02
/venv/main/bin/python cc_prompts.py examples                   # one rendered prompt per (condition, mode)
/venv/main/bin/python check_locators.py                        # first-violation locator vs grader (fuzz)
./exp02_smoke_chain.sh                                         # Qwen3-8B, 3 items, all conditions
/venv/main/bin/python cc_manifest.py --exp exp01 && ./exp01_run.sh     # gate; exits 1 if it fails
/venv/main/bin/python cc_manifest.py --exp exp02 && ./exp02_chain.sh   # starts only after a passed gate
```

Pipeline per experiment: `cc_generate.py` (two-phase vLLM: reasoning to `</think>` or the 25k cap, then the answer)
-> `cc_grade.py` (CoT-Control's grader, first-violation locator, regex meta flag, accuracy) -> `cc_judge.py`
(Qwen3.8-27B with CoT-Control's judge prompts: ignore_question triple-check, meta-discussion on compliant traces)
-> `cc_analysis.py` (bootstrap CIs, figures, `REPORT_auto.md`) -> `cc_verification.py` (blinded human sheet, kappa).
Caches live in `cache/<exp>/` under content-hash names; analysis outputs in `results/<exp>/analysis/<run>/` and
`figures/<exp>/<run>/` are never overwritten.

### Faster generation (for the next experiments)

exp01/exp02 ran on one H100 with engine profile `v1`, at 400-600 output tokens/s: only ~4-6 long traces fit in the
KV cache at once, and the separate answer phase re-prefilled every prompt + reasoning (35-38 min per 1,000 Qwen3-32B
requests). The opt-in pieces below change scheduling or hardware use, not what is sampled:

- `cc_generate.py --engine stream`: the same requests (prompts, sampling parameters, seeds) in one priority queue.
  Each answer phase starts as soon as its reasoning ends, while the reasoning is still in the prefix cache. There is
  no per-chunk wait on the longest trace, rows are appended as they finish (resumable), and GPU memory use is 0.95.
  `--engine stream_mtp` also uses Qwen3.6's own multi-token-prediction draft head (same output distribution, but
  different samples; benchmark it first, since the gain shrinks at long context).
- `Qwen3.6-27B-FP8` (pinned in `cc_config.EXTRA_SUBJECTS`): 31 GB of weights instead of 56 GB, so ~2.5x the KV cache.
  It needs a bf16 precision check, as Qwen3-32B has.
- `generate_models.sh --exp <exp> [flags] -- <models>`: one model per GPU in parallel on a multi-GPU instance.
- `cc_download.py <models>`: pinned snapshots, refused if the disk would drop below 20 GB free.

Before a pre-registered run uses a non-default profile, compare it with `v1` and list it in the manifest deviations:

```bash
/venv/main/bin/python check_streaming.py        # CPU: stream issues v1's exact requests; crash + resume
/venv/main/bin/python check_engine_equivalence.py prepare --exp exp02     # smoke requests -> cache/exp02/engine_check/<key>/
./vllm_python.sh cc_generate.py --exp exp02 --model Qwen3-8B --requests <printed path> --sampling greedy
./vllm_python.sh cc_generate.py --exp exp02 --model Qwen3-8B --requests <printed path> --sampling greedy --engine stream
/venv/main/bin/python check_engine_equivalence.py compare --exp exp02 <v1 file> <stream file>
```

Greedy runs should mostly agree trace for trace; where they diverge, it should be late in the trace (floating-point
drift from a different batch composition), not at the start.
