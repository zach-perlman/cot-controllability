"""Observational upper bound on what a compliant prefill could add to stacked_all (thinking on, 100 questions).
The start-with sentence is exp04's compliant prefill sentence, so prefilling it only changes rows whose trace did
not already open with it. Splits stacked_all rows by whether the trace starts with the sentence and reports the share
clean through 1000 tokens (a trace that stops clean before 1000 counts as not clean). Not causal: rows that comply at
the start may be the easier ones; the prefill arm is the randomized version."""
import sys

sys.path.insert(0, "scripts")
import cc_exp06a
import cc_exp06a_analysis as an

MODELS = ["Qwen3.8-27B-FP8", "Gemma-4-31B-FP8", "Qwen3-32B"]
T = 1000

raw, _ = an.exp06a_rows(MODELS, skip_missing=True)
df = an.score(raw[[c for c in an.COLUMNS if c in raw.columns]], True)
df = df[(df["arm"] == "stacked_all") & df["thinking"].astype(bool)].copy()
df["starts"] = [str(r).lstrip().startswith(cc_exp06a.start_sentence(m)) for r, m in zip(df["reasoning"], df["mode"])]
df["clean_1000"] = (~df["event"] | (df["time"] >= T)) & (df["reasoning_tokens"] >= T)

for m in MODELS:
    d = df[df["model"] == m]
    g = d.groupby("starts")["clean_1000"].agg(["mean", "size"])
    share = d["starts"].mean()
    print(f"\n{m}: starts with sentence {share:.1%}; clean through {T} by group")
    print((g.assign(mean=g["mean"] * 100)).round(1).to_string())
    print("by rule (share starting / clean_1000 if starts / if not):")
    for mode, dm in d.groupby("mode"):
        s, n = dm[dm["starts"]], dm[~dm["starts"]]
        print(f"  {mode:22s} {dm['starts'].mean():5.0%}  {s['clean_1000'].mean() * 100 if len(s) else float('nan'):5.1f}"
              f"  {n['clean_1000'].mean() * 100 if len(n) else float('nan'):5.1f}")
