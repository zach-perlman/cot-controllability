"""exp03_abort_survival setup: item draw, request file, manifest (each written once, never overwritten).

items:    EXP03_ITEMS_PER_SOURCE new questions per source (seed EXP03_ITEM_SEED), drawn like exp02's (cc_items: rows
          with at least one keyword) from the rows exp02 did not use.
requests: exp02's request rows (cc_prompts.build_prompt, same prompt texts and seed function) for 4 prompts x 9 modes
          x 1 rollout per item, plus 1 no-constraint trace per item. Each row adds
            abort_on_violation: mode in ABORT_MODES and the (item, mode) cell is not in the full-trace subsample;
            full_trace_cell: the cell is in the subsample (FULL_TRACE_SHARE of the items per abortable mode, drawn
                             per mode with FULL_TRACE_SEED; the same cells for every prompt and model).
manifest: results/exp03_abort_survival/manifest.json.

Run: /venv/main/bin/python scripts/cc_exp03.py items|requests|manifest
"""

from __future__ import annotations

import argparse
import datetime
import json
import random

import _cotcontrol
import cc_config as cfg
import cc_items
import cc_manifest
import cc_prompts


def draw_items() -> None:
    if cfg.EXP03_ITEMS_PATH.exists():
        raise SystemExit(f"{cfg.EXP03_ITEMS_PATH} exists; the item set is fixed once drawn")
    used = {json.loads(line)["item_id"] for line in cfg.ITEMS_PATH.open()}
    run_cceval = _cotcontrol.module("run_cceval")
    rng = random.Random(cfg.EXP03_ITEM_SEED)
    items = []
    for source, n in cfg.EXP03_ITEMS_PER_SOURCE.items():
        rows = [r for r in cc_items.load_source(run_cceval, source) if r["item_id"] not in used]
        chosen = sorted(rng.sample(rows, n), key=lambda r: r["row_id"])
        if any(not it["options"] for it in chosen):
            raise ValueError(f"{source}: an item has no options; the answer format needs a letter")
        items.extend(chosen)
        print(f"{source}: {n} of {len(rows)} eligible rows not used by exp02")
    cfg.EXP03_ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.EXP03_ITEMS_PATH.write_text("".join(json.dumps(it) + "\n" for it in items))
    print(f"wrote {len(items)} items to {cfg.EXP03_ITEMS_PATH}")


def load_items() -> list[dict]:
    return [json.loads(line) for line in cfg.EXP03_ITEMS_PATH.open()]


def full_trace_cells(items: list[dict]) -> set[tuple[str, str]]:
    """(item_id, mode) cells of the abortable modes that run to the end: a fixed share of the items per mode."""
    rng = random.Random(cfg.FULL_TRACE_SEED)
    ids = [it["item_id"] for it in items]
    n = round(cfg.FULL_TRACE_SHARE * len(ids))
    return {(iid, mode) for mode in cfg.ABORT_MODES for iid in rng.sample(ids, n)}


def request_rows(items: list[dict]) -> list[dict]:
    full = full_trace_cells(items)
    rows = []
    for item in items:
        for mode in cfg.MODES:
            grading_prompt = cc_prompts.cotcontrol_prompt(item, mode)[1]
            for prompt in cfg.EXP03_DESIGN["prompts"]:
                system, user = cc_prompts.build_prompt(item, mode, prompt)
                rows.append({"item_id": item["item_id"], "source": item["source"], "mode": mode, "prompt": prompt,
                             "rollout": 0, "system": system, "user": user, "grading_prompt": grading_prompt,
                             "full_trace_cell": (item["item_id"], mode) in full,
                             "abort_on_violation": mode in cfg.ABORT_MODES and (item["item_id"], mode) not in full})
        system, user = cc_prompts.cotcontrol_prompt(item, "baseline")
        for r in range(cfg.EXP03_DESIGN["no_constraint_rollouts"]):
            rows.append({"item_id": item["item_id"], "source": item["source"], "mode": cfg.NO_CONSTRAINT,
                         "prompt": cfg.NO_CONSTRAINT, "rollout": r, "system": system, "user": user,
                         "grading_prompt": None, "full_trace_cell": False, "abort_on_violation": False})
    for row in rows:
        row["seed"] = cfg.rollout_seed(row["item_id"], row["mode"], row["rollout"])
        row["request_id"] = cfg.content_key({k: row[k] for k in ("item_id", "mode", "prompt", "rollout", "system",
                                                                   "user")})
    return rows


def write_requests() -> None:
    path = cfg.EXP03.requests
    if path.exists():
        raise SystemExit(f"{path} exists; the request set is fixed once written")
    rows = request_rows(load_items())
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    n_abort = sum(r["abort_on_violation"] for r in rows)
    print(f"wrote {len(rows)} requests to {path} ({n_abort} may abort at the first violation)")


def write_manifest() -> None:
    exp = cfg.EXP03
    if exp.manifest.exists():
        raise SystemExit(f"{exp.manifest} exists; manifests are written once, before the run")
    rows = [json.loads(line) for line in exp.requests.open()]
    manifest = {
        "exp_id": exp.exp_id,
        "question": "exp02's question with enough items to resolve ~3-point prompt differences: do Arun Jose's "
                    "zero-shot prompts raise CoT-Control compliance over the paper's baseline prompt at matched "
                    "reasoning length (survival to a fixed token count), without meta-discussion and without "
                    "costing accuracy?",
        "design": {**cfg.EXP03_DESIGN, "abort_modes": cfg.ABORT_MODES, "full_trace_share": cfg.FULL_TRACE_SHARE,
                   "full_trace_seed": cfg.FULL_TRACE_SEED, "abort_margin_chars": cfg.ABORT_MARGIN_CHARS,
                   "abort_check_min_tokens": cfg.ABORT_CHECK_MIN_TOKENS,
                   "abort_rule": "scripts/cc_abort.py; replayed on every exp02 trace by scripts/check_abort_rule.py "
                                 "(results/exp03_abort_survival/abort_rule_check.json) before this manifest"},
        "sampling": {m: {"name": "card", "params": cfg.FAMILIES[cfg.SUBJECTS[m]["family"]]["sampling"]}
                     for m in cfg.GRID_MODELS},
        "primary_outcome": cfg.EXP03_PRIMARY,
        "primary_t_star_tokens": cfg.SURVIVAL_T_STAR,
        "robustness_of_primary": cfg.EXP03_ROBUSTNESS,
        "secondary_outcomes": [
            "P1 compliance (exp02's definition; fully observed, since compliant traces are never aborted), P2 "
            "(judge meta-discussion) and P2_regex, macro over (dataset, mode)",
            "survival curves S(t) for t in 0..%d tokens per (model, prompt), and S(t*) per mode"
            % cfg.SURVIVAL_CURVE_MAX_TOKENS,
            "accuracy on full traces only (ignore_question, repeat_sentences and the full-trace cells), per prompt, "
            "paired within the same cells; no-constraint accuracy",
            "near-empty, truncation and censoring-before-t* rates per (model, prompt)",
            "pooled with exp02's 50 items (same models, prompts, sampling; exp02 traces are full): S(t*) and P1",
        ],
        "not_estimable": "P3 (exp02's length-quintile reweighting) needs the final length of every trace; aborted "
                         "traces have none. The survival outcome replaces it.",
        "controls": {
            "no_constraint": "CoT-Control's unconstrained prompt, 1 trace per item: natural compliance per rule, "
                             "accuracy and length without a rule",
            "abort_audit": "the full-trace cells: the abort rule is replayed on them after the run (would it have "
                           "fired, and at the same violation?)",
            "judge_agreement": "as exp02 (blinded sheet; judge numbers are labelled 'judge')",
        },
        "power_note": "100 items x 9 modes = 900 traces per (model, prompt); 700 in the abortable modes for the "
                      "primary outcome. At 3-6% survival the standard error of a prompt - baseline difference is "
                      "about 1.1-1.3 points (independent traces; clustering by item widens it). With Holm over 9 "
                      "contrasts, a 4-5 point difference has about 80% power; 3 points is borderline.",
        "hardware": "8x A100-SXM4-40GB (PCIe, no NVLink); Qwen3-8B 8 shards x TP 1, Qwen3-32B (FP8 checkpoint; "
                    "A100 has no FP8 tensor cores, so vLLM runs it with bf16 activations) and Qwen3.6-27B 4 shards "
                    "x TP 2; judge Qwen3.8-27B TP 4",
        "engine": "streaming queue (one engine loop; a trace's answer phase is enqueued when its reasoning ends); "
                  "aborted traces get no answer phase. Compared with the batch engine on the smoke requests before "
                  "this manifest (results/exp03_abort_survival/engine_check.json).",
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "commit": cc_manifest.git_commit(),
        "cotcontrol_commit": cfg.COTCONTROL_COMMIT,
        "models": {m: cfg.ALL_MODELS[m] for m in cfg.GRID_MODELS},
        "judge": {**cfg.JUDGE_MODEL, "thinking_cap": cfg.JUDGE_THINKING_CAP_TOKENS,
                  "answer_cap": cfg.JUDGE_ANSWER_CAP_TOKENS, "ignore_question_checks": cfg.JUDGE_IGNORE_QUESTION_CHECKS,
                  "seed": cfg.JUDGE_SEED},
        "caps": {"reasoning": cfg.REASONING_CAP_TOKENS, "answer": cfg.ANSWER_CAP_TOKENS,
                 "max_model_len": cfg.VLLM_MAX_MODEL_LEN, "forced_close": cfg.FORCED_THINK_CLOSE},
        "items": {"path": str(cfg.EXP03_ITEMS_PATH.relative_to(cfg.REPO_ROOT)),
                  "sha256": cc_manifest.sha256_file(cfg.EXP03_ITEMS_PATH),
                  "per_source": cfg.EXP03_ITEMS_PER_SOURCE, "seed": cfg.EXP03_ITEM_SEED,
                  "disjoint_from": str(cfg.ITEMS_PATH.relative_to(cfg.REPO_ROOT))},
        "requests": {**cc_manifest.request_summary(exp.requests),
                     "abort_on_violation": sum(r["abort_on_violation"] for r in rows),
                     "full_trace_cells": sum(r["full_trace_cell"] for r in rows)},
        "modes": cfg.MODES,
        "seeds": {"item_draw": cfg.EXP03_ITEM_SEED, "full_trace_cells": cfg.FULL_TRACE_SEED,
                  "engine": cfg.VLLM_ENGINE_SEED, "per_request": "sha256(item|mode|rollout)[:8] (as exp02)",
                  "judge": cfg.JUDGE_SEED, "bootstrap": cfg.BOOTSTRAP_SEED},
        "bootstrap": {"iters": cfg.BOOTSTRAP_ITERS, "unit": "question (item), stratified by source",
                      "ci": cfg.CI_LEVEL},
        "deviations_from_paper": cc_manifest.DEVIATIONS[:2] + [
            "100 new fixed items (37 GPQA / 38 HLE / 25 MMLU-Pro), none of exp02's.",
            "Generation stops at the first violation in the abortable modes (except the full-trace cells).",
        ],
    }
    exp.manifest.parent.mkdir(parents=True, exist_ok=True)
    exp.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {exp.manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["items", "requests", "manifest"])
    args = parser.parse_args()
    {"items": draw_items, "requests": write_requests, "manifest": write_manifest}[args.cmd]()
