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
In flight (exp ids): exp02_prompt_grid (launched after the gate). exp01_paper_gate: gate PASS, UNVERIFIED —
Qwen3-32B P1 = 2.0 [0.8, 3.4] vs paper 2.3 (9 of 450 compliant, all read by the agent and genuine; 1 truncated);
Qwen3-8B P1 = 0.6 [0.0, 1.2]. The CI is wide (50 items), so the gate rules out large setup errors only.
Current story candidate: none yet. Smoke test (Qwen3-8B, 3 items) showed two ways to "comply" without controlling
content: empty reasoning under ignore_question (7 of 12) and 25k-token "meow meow ..." traces that pass the meow
grader; both are now flagged (near-empty, degenerate) and reported next to compliance.
Next experiment and why: exp01 gate (does the paper's Qwen3-32B 2.3% reproduce under our vLLM setup?), then the
exp02 prompt grid.
Kill criteria in force: exp01 gate fails -> exp02 does not start until a human decides.
Last updated: 2026-09-29 by agent (exp01 gate passed, exp02 launched)

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
