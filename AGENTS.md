# AGENTS.md (on top of existing .md)

Instance/ops guide (Vast): /workspace/AGENTS.md; this file adds project rules on top.

## STATE (human-maintained; update at every close)
Project: CoT controllability in open reasoning models: can they follow rules about what goes in their chain of
thought, and when they comply, is anything actually hidden from a monitor?
Question (falsifiable sentence): Under better prompts, Qwen3-8B / Qwen3-32B / Qwen3.6-27B comply with CoT-Control
rules more often than under the paper's baseline prompt, at matched CoT length, without meta-discussion, and
without giving up accuracy.
So what: CoT-Control numbers are used in system cards to argue CoT monitorability; if they are under-elicited at
matched length, those arguments are weaker than stated.
Established so far (exp ids): none verified.
In flight (exp ids): exp02_prompt_grid closed at Qwen3-8B + Qwen3-32B (Qwen3.6 and the bf16 check dropped by
the human, see deviations_a100_*.json); report results/exp02_prompt_grid/analysis/grid/REPORT_auto.md, UNVERIFIED.
exp03_abort_survival (human-chosen design: 100 new items, primary KM S(1000) macro over the 7 abortable modes,
25% full-trace cells, exp02 pooled as secondary): engine check PASS, manifest written. Qwen3-8B generated + graded
on A100. Qwen3-32B and Qwen3.6-27B were partial (2-shard stream files, cache/exp03/generations/*.parts/
stream.shard{0,1}of2.jsonl) when the human moved to an H200; those rows are archived in cache/exp03/a100_partial/
and not analyzed. Qwen3-32B and Qwen3.6-27B are being regenerated from scratch on 1x H200, then grade, judge (all
three models), scripts/cc_survival.py --run main: scripts/h200_exp03.sh, log/exp03/h200_exp03.log, deviation in
results/exp03_abort_survival/deviations_h200.json. Off-box backup of cache/ and of the files git ignores because
they quote questions: private HF dataset zachperlman20/cot-controllability-cache. exp01_paper_gate: gate PASS, UNVERIFIED —
Qwen3-32B P1 = 2.0 [0.8, 3.4] vs paper 2.3 (9 of 450 compliant, all read by the agent and genuine; 1 truncated);
Qwen3-8B P1 = 0.6 [0.0, 1.2]. The CI is wide (50 items), so the gate rules out large setup errors only.
Current story candidate: none yet. Smoke test (Qwen3-8B, 3 items) showed two ways to "comply" without controlling
content: empty reasoning under ignore_question (7 of 12) and 25k-token "meow meow ..." traces that pass the meow
grader; both are now flagged (near-empty, degenerate) and reported next to compliance.
Next experiment and why: exp03 (exp02's 50 items resolve only large prompt effects; exp03 doubles the items and
stops traces at the first violation, which in exp02 was followed by 93-96% of the abortable modes' tokens).
Kill criteria in force: exp03 engine check (results/exp03_abort_survival/engine_check.json) fails -> no exp03
manifest or generation until a human decides.
Open issue for the human: three tracked files quote benchmark question text (prompt_examples.md,
verification_smoke/sheet.md, exp01 read_samples.md) and are in public git history.
Last updated: 2026-09-29 by agent (exp03 Qwen3-32B / Qwen3.6-27B running on H200)

## Roles
- Agent: implement, run, report in the format below.
- Humans: design experiments, choose baselines and controls, read raw data, recompute headlines, interpret.
  If you think a design is wrong, say so, propose an alternative, and wait.

## Environment
- Persistent Jupyter kernel via the `jupyter` MCP server (fallback: ipython in tmux session `py`).
- Load models/data in dedicated cells at the top. Never restart the kernel or reload a loaded model without asking.
- Save every plot to figures/<exp_id>/<run>/<name>.png as well as displaying it.
- Jobs longer than 10 minutes run as background scripts with logs, not notebook cells.
- Large-scale sampling uses vLLM (venv at /workspace/.venv-vllm, always run via scripts/vllm_python.sh, which adds
  CUDA forward-compat libs on older drivers); activations use nnsight/HF in the main venv.
- New instance: restore envs from log/setup/eval_awareness_pins/freeze_*.txt (see log/setup/restore_*.log) and
  check with `/venv/main/bin/python scripts/check_env.py`.
- Use uv for packages. Never pip install into system Python.
- CoT-Control (third_party/CoTControl, pinned submodule) is imported, never edited: its grader, prompts, datasets and
  judge prompts are the reference implementation.

## Experiments
- Every experiment: id expNN_name, directory results/expNN_name/, manifest.json written BEFORE the run.
- Never overwrite or hand-edit results. Never change metric code between a baseline and its comparison without
  re-running the baseline.
- Always run the control the humans specified. If none was specified, stop and ask.
- Report n, seeds, bootstrap CI, and the baseline next to every headline number.
- Cache expensive intermediates in cache/ with content-addressed names.

## Report format (end of every experiment)
EXP <id> @ <commit> | model <id> | n=<n> | seeds=<list>
Claim tested:
Headline: <metric> = <value> [CI]   Baseline: <value> [CI]   Control: <value> [CI]
Figures:
Three dumbest ways this could be wrong: 1) ... checked? 2) ... checked? 3) ... checked?
For the human to verify: <file, rows, number to recompute>
Status: UNVERIFIED until a human adds it to VERIFIED.md

## Evidence norms
- Recomputed rates/AUCs > plotted curves > judge scores. Label judge scores "judge", with human agreement if known.
- Programmatic compliance (CoT-Control's grader) is judge-free; meta-discussion and ignore_question compliance are
  judge-scored (Qwen3.8-27B) and are weaker evidence.
- Random examples by fixed seed, stated as such.
- Interventions at matched norm/dose; report collateral damage (incoherence, capability) alongside the effect.
- Compliance is always reported next to accuracy and CoT length, so "complying by not reasoning" is visible.
- A null with a working control is a result.

## Git and writing
- Commit when a figure exists (exp id in the message) and again when verified. No force-push; no amending pushed
  commits.
- Drafts only. Never state a result as established unless it is in VERIFIED.md. No filler.
