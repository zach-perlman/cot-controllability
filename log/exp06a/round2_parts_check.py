"""Before writing the new round-2 request files (deviations_round2_decision.json):
1. the refactored round2_rows(model, "_round2") rebuilds requests_<model>_round2.jsonl exactly (same rows, same ids);
2. each new thinking-off row is built from round 1's thinking-off pieces: the same system prompt and user-message
   prefix as round 1's thinking-off base arm (stacked or baseline), V3's thinking-off example answers, and the same
   seed / cap / grading fields."""
import json
import sys

sys.path.insert(0, "scripts")
import cc_config as cfg
import cc_exp04
import cc_exp06a as x

for model in cfg.EXP06A_MODELS:
    old = cc_exp04.load_requests(x.requests_path(model, "_round2"))
    new = x.round2_rows(model, "_round2")
    same = [json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) for a, b in zip(old, new)]
    print(f"{model}: _round2 rebuilt {len(new)} rows, identical to file: {len(old) == len(new) and all(same)}")

    off = x.round2_rows(model, "_round2_thinking_off_half")
    added = x.round2_rows(model, "_round2_added")
    r1_off = [r for r in x.main_rows(model) if not r["thinking"]]
    r1 = {(r["item_id"], r["mode"], r["prompt"]): r for r in r1_off}
    print(f"  _round2_added {len(added)} rows, _round2_thinking_off_half {len(off)} rows "
          f"({len({r['item_id'] for r in off})} items)")
    for r in off:
        ref_stacked = r1[(r["item_id"], r["mode"], "stacked")]
        ref_fewshot = r1[(r["item_id"], r["mode"], "fewshot")]
        base, with_examples, blocks = cfg.EXP06A_ROUND2_ARMS[r["prompt"]]
        ref_base = ref_stacked if base == "stacked" else r1[(r["item_id"], r["mode"], "baseline_rerun")]
        checks = {
            "system == round-1 base": r["system"] == ref_base["system"],
            "user starts with round-1 base": r["user"].startswith(ref_base["user"]),
            "fields like round 1": all(r[k] == ref_base[k] for k in ("seed", "response_cap_tokens", "answer_phase",
                                                                     "abort_on_violation", "reasoning_stop_tokens",
                                                                     "grading_prompt", "thinking", "condition")),
            "example answers == V3 thinking-off": (not with_examples
                                                   or [h["answer"] for h in r["history"]]
                                                   == [h["answer"] for h in ref_fewshot["history"]]),
        }
        if not all(checks.values()):
            print("  FAIL", r["prompt"], r["mode"], checks)
    print(f"  thinking-off rows checked against round 1: {len(off)}")
