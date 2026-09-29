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
