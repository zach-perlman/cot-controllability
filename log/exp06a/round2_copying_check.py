"""Falsification check on round 2's large gains (thinking on): is the compliant text a copy of the few-shot examples
or degenerate repetition rather than reasoning about the question? Per (model, arm), over all rows:
  example_8gram_%   share of the trace's word 8-grams that occur in the row's own example turns (copying)
  question_overlap  share of the question's content words (>= 5 letters) that the trace uses (engages the question)
  distinct_8gram_%  distinct / total word 8-grams within the trace (low = looping)
Prints only aggregates; no trace or question text."""
import glob
import json
import re
import sys

sys.path.insert(0, "scripts")
import pandas as pd

import cc_config as cfg
import cc_exp04
import cc_exp06a

ARMS = ["baseline_rerun", "stacked_rerun", "fewshot", "stacked_fewshot", "stacked_start_with", "stacked_all",
        "stacked_fewshot_failure_guide", "fewshot_failure_guide"]
WORD = re.compile(r"[a-z]+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower().replace("meow", " "))


def grams(ws: list[str], n: int = 8) -> list[tuple]:
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


items = {it["item_id"]: it for it in cc_exp06a.load_items()[0]}
recs = []
for model in cfg.EXP06A_MODELS:
    for part in cc_exp06a.request_parts(model)[:1] + ["_round2", "_round2_added"]:
        requests = {r["request_id"]: r for r in cc_exp04.load_requests(cc_exp06a.requests_path(model, part))}
        (path,) = glob.glob(str(cfg.EXP06A.generations / f"{model}__card__stream_abort_{model}{part}__*.jsonl"))
        for line in open(path):
            row = json.loads(line)
            req = requests[row["request_id"]]
            if req["prompt"] not in ARMS or not req["thinking"] or not row.get("reasoning"):
                continue
            trace = grams(words(row["reasoning"]))
            if not trace:
                continue
            example = {g for turn in req["history"] for g in grams(words(turn["reasoning"] or turn["answer"]))}
            q_words = {w for w in words(items[req["item_id"]]["question"]) if len(w) >= 5}
            t_words = set(words(row["reasoning"]))
            recs.append({"model": model, "arm": req["prompt"],
                         "example_8gram_%": 100 * sum(g in example for g in trace) / len(trace),
                         "question_overlap_%": 100 * len(q_words & t_words) / len(q_words) if q_words else None,
                         "distinct_8gram_%": 100 * len(set(trace)) / len(trace)})

table = pd.DataFrame(recs).groupby(["model", "arm"]).agg(["mean"]).round(1)
table.columns = [c[0] for c in table.columns]
table["n"] = pd.DataFrame(recs).groupby(["model", "arm"]).size()
print(table.to_string())
