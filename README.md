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
