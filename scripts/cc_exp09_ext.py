"""exp09 extension (results/exp09_final_test/manifest_extension_exp08_parts.json, written before its generation):
exp08's openings and thinking on/off parts on exp09's 120 never-used questions and 6 of exp09's 7 models (MODEL_ORDER;
not GLM-4.7-Flash). It is not part
of exp09's pre-registered test (P1-P3); its contrasts are their own family.

Same builders as exp08 (cc_exp08: the opening sentences, CoT-Control's external-CoT prompt, A thinking off, the row
format and seeds), so rows render with cc_exp06's renderer as exp08's did. A's examples and their number per rule are
exactly exp09's short rows' (exp09's requests record), so the thinking-on cells are exp09's own short rows:
  openings  rules cc_exp08.OPENING_RULES; arms A|compliant, A|noncompliant, A|commitment (the reasoning starts with
            the opening, only the continuation is graded) and no_rule|compliant (copying control); A|none is exp09's
            short A row of the same question, rule and seed. Thinking on, stop at the first violation or 1200 tokens.
  channels  rules cc_exp08.RULES; baseline|none|off on all four, A|none|off where A's exp09 examples fit beside the
            16384-token response cap (exp08: not multiple_word_suppression); the thinking-on cells are exp09's short
            rows. Thinking off: the reasoning goes in <output_reasoning> tags in the response (not stopped early).

Steps (/venv/main/bin/python scripts/cc_exp09_ext.py ...):
  requests --model M   the model's rows (cache/exp09/requests_ext_<M>.jsonl) and record
  plan                 the gpu_lanes.py plan (results/exp09_final_test/plan_ext_1xB200.json)
  grade --model M      grade the model's rows (cache/exp09/grades/)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cc_config as cfg
import cc_exp07 as e7
import cc_exp07b as e7b
import cc_exp08 as e8
import cc_exp09 as e9
import cc_prompts
import exp07_candidates as cands
import exp07_rules as rules
import exp07b_candidates as b

OFF_PROMPT_LIMIT = cfg.VLLM_MAX_MODEL_LEN - cfg.EXP04_EXTERNAL_CAP_TOKENS - e8.CONTEXT_MARGIN
# Longest requests first (the run ends when its slowest requests do), rule by rule within a group.
REQUEST_ORDER = [("off", "A", "none"), ("off", "baseline", "none"), ("on", "A", "compliant"),
                 ("on", "A", "noncompliant"), ("on", "A", "commitment"), ("on", "no_rule", "compliant")]
# Every exp09 model but GLM-4.7-Flash: cc_exp06's renderer has no history format for its family
# (cfg.EXP06_HISTORY_REASONING), and adding one would change the cache key of every exp06/exp08 generation.
MODEL_ORDER = ["Qwen3.6-27B-FP8", "Gemma-4-31B-FP8", "Qwen3.8-27B-FP8", "Qwen3-32B", "Gemma-4-12B-FP8",
               "Qwen3.6-35B-A3B-FP8"]


def check_openings() -> None:
    """Compliant and commitment openings pass the grader on every exp09 question; non-compliant ones fail it."""
    for mode, texts in e8.OPENINGS.items():
        for it in e9.items():
            for kind, text in texts.items():
                if rules.compliant(mode, text, it) != (kind != "noncompliant"):
                    raise ValueError(f"{mode} {kind} opening: wrong compliance on {it['item_id']}")


def exp09_example_counts(model: str) -> dict[str, int]:
    record = json.loads((e9.EXP.results / f"requests_record_{model}.json").read_text())
    return {mode: r["A_examples"] for mode, r in record["rules"].items()}


def exp09_short_rows(model: str, arm: str) -> dict[tuple[str, str], dict]:
    """exp09's frozen short requests of one arm, by (question, rule)."""
    return {(r["item_id"], r["mode"]): r for r in map(json.loads, e9.requests_path(model).open())
            if r["channel"] == "short" and r["prompt"] == arm}


def check_same_as_exp09(row: dict, short: dict) -> None:
    """A thinking-on row of the extension has exp09's short row's prompt and seed (so exp09's row is its |none cell)."""
    for field in ("system", "user", "history", "seed"):
        if row[field] != short[field]:
            raise RuntimeError(f"{row['item_id']} {row['mode']} {row['exp06_cell']}: {field} differs from exp09")


def model_rows(model: str) -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer
    check_openings()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir(model))
    family = cfg.ALL_MODELS[model]["family"]
    source = e9.EXAMPLE_SOURCE.get(model, model)
    by_block = {cc_prompts.question_block(t["item"]): t["item"] for t in e7b.own_traces(source)}
    n_exp09 = exp09_example_counts(model)
    exp09_a, exp09_baseline = exp09_short_rows(model, "A"), exp09_short_rows(model, "baseline")
    out, record = [], {"model": model, "example_source": source, "rules": {}}
    for mode in e8.RULES:
        examples = e9.usable_examples(model, mode)
        n_on = n_exp09[mode]
        example_items = [by_block[ex.context.question_block] for ex in examples]

        def a_off_with(it: dict, n: int) -> cands.Prompt:
            return e8.a_off(it, mode, examples, example_items, n)

        n_off = e8.n_examples(tokenizer, family, a_off_with, examples[:n_on], e9.items(), OFF_PROMPT_LIMIT)
        a_runs_off = n_off == n_on
        record["rules"][mode] = {"A_examples_exp09": n_on, "A_examples_off": n_off, "A_runs_off": a_runs_off}
        for it in e9.items():
            a_on = b.many_examples(e9.context(it, mode, examples), mode, n_on)
            check_same_as_exp09(e8.row(it, mode, "A", "none", "on", a_on, "channels"), exp09_a[it["item_id"], mode])
            baseline_on = cands.baseline(e9.context(it, mode, examples))
            check_same_as_exp09(e8.row(it, mode, "baseline", "none", "on", baseline_on, "channels"),
                                exp09_baseline[it["item_id"], mode])
            if a_runs_off:
                out.append(e8.row(it, mode, "A", "none", "off", a_off_with(it, n_on), "channels"))
            out.append(e8.row(it, mode, "baseline", "none", "off", e8.baseline_off(it, mode), "channels"))
            if mode in e8.OPENING_RULES:
                texts = e8.OPENINGS[mode]
                for kind in ("compliant", "noncompliant", "commitment"):
                    out.append(e8.row(it, mode, "A", kind, "on", a_on, "openings", prefill=texts[kind]))
                no_rule = cands.Prompt(*cc_prompts.cotcontrol_prompt(it, "baseline"), [])
                out.append(e8.row(it, mode, "no_rule", "compliant", "on", no_rule, "openings",
                                  prefill=texts["compliant"]))
    if len({r["request_id"] for r in out}) != len(out):
        raise RuntimeError(f"{model}: duplicate request ids")
    rule_order = {m: k for k, m in enumerate(e8.RULES)}
    item_order = {it["item_id"]: k for k, it in enumerate(e9.items())}

    def cell_key(r: dict) -> tuple[str, str, str]:
        arm, opening, channel = r["exp06_cell"].split("|")
        return channel, arm, opening

    out.sort(key=lambda r: (REQUEST_ORDER.index(cell_key(r)), rule_order[r["mode"]], item_order[r["item_id"]]))
    record["cells"] = {c: sum(r["exp06_cell"] == c for r in out) for c in sorted({r["exp06_cell"] for r in out})}
    return out, record


# --- Paths, plan and grading ----------------------------------------------------------------------------------------
def requests_path(model: str) -> Path:
    return e9.EXP.cache / f"requests_ext_{model}.jsonl"


def generation_path(model: str) -> Path:
    paths = glob.glob(str(e9.EXP.generations / f"{model}__card__stream_abort_ext_{model}__*.jsonl"))
    if len(paths) != 1:
        raise SystemExit(f"{model}: expected one extension generation file, found {len(paths)}")
    return Path(paths[0])


def grades_path(model: str) -> Path:
    return e9.EXP.grades / generation_path(model).name


def grade(model: str) -> Path:
    """exp08's grading (cc_exp08.grade) with exp09's questions."""
    out = grades_path(model)
    if out.exists():
        return out
    requests = {r["request_id"]: r for r in map(json.loads, requests_path(model).open())}
    gens = [json.loads(line) for line in generation_path(model).open()]
    by_id = {it["item_id"]: it for it in e9.items()}
    rows = e7.grade_generations(gens, requests, by_id, model)
    blocks = {g["request_id"]: g.get("external_blocks") for g in gens}  # thinking off: closed tag blocks
    for r in rows:
        req = requests[r["request_id"]]
        r.update(cell=req["exp06_cell"], condition=req["condition"], part=req["exp08_part"],
                 channel=req["exp06_cell"].rsplit("|", 1)[1], external_blocks=blocks[r["request_id"]],
                 model_group="design" if model in e9.DESIGN_MODELS else "fresh")
    e9.EXP.grades.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out


def write_plan() -> Path:
    def job(model: str) -> dict:
        return {"exp": "exp09", "model": model, "requests": str(requests_path(model).relative_to(cfg.REPO_ROOT)),
                "items": str(e9.ITEMS_PATH.relative_to(cfg.REPO_ROOT)), "memory": e9.MEMORY[model],
                "after": ["/venv/main/bin/python", "cc_exp09_ext.py", "grade", "--model", model]}
    plan = {"name": "exp09_ext_1xB200", "mps": False, "gpus": {"0": [[job(m) for m in MODEL_ORDER]]}}
    path = e9.EXP.results / "plan_ext_1xB200.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "requests", "plan", "grade"],
                        help="check: build a model's rows and print its record, writing nothing")
    parser.add_argument("--model", choices=MODEL_ORDER)
    args = parser.parse_args()
    if args.command == "plan":
        print(write_plan())
        return
    if args.command == "grade":
        print(grade(args.model))
        return
    rows, record = model_rows(args.model)
    if args.command == "check":
        print(json.dumps(record, indent=1))
        print(f"{len(rows)} rows; first {rows[0]['exp06_cell']} {rows[0]['mode']}; "
              f"last {rows[-1]['exp06_cell']} {rows[-1]['mode']}")
        return
    e7.write_requests(requests_path(args.model), rows)
    (e9.EXP.results / f"requests_record_ext_{args.model}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {len(rows)} requests to {requests_path(args.model)}")


if __name__ == "__main__":
    main()
