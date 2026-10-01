"""Length-robust check on the CoT-specificity table: share of ALL rows (51 thinking-off questions, both thinking
settings) whose graded text reaches 200 tokens with no violation. A short clean output counts as a failure here, so
brevity cannot raise this number. Gain = arm minus the in-environment baseline rerun (5-rule mean, percentage
points), per model."""
import sys

sys.path.insert(0, "scripts")
import cc_exp06a
import cc_exp06a_analysis as an

ARMS = ["stacked_rerun", "start_with", "warmup", "fewshot", "style_guide", "failure_guide", "monitor_stealth", "meta_allowed"]
MODELS = ["Qwen3.8-27B-FP8", "Gemma-4-31B-FP8", "Qwen3-32B"]

raw, _ = an.exp06a_rows(MODELS, skip_missing=True)
df = an.score(raw[[c for c in an.COLUMNS if c in raw.columns]], True)
df = df[df["item_id"].isin([it["item_id"] for it in cc_exp06a.thinking_off_items()])]
df["clean_200"] = (~df["event"] | (df["time"] >= 200)) & (df["reasoning_tokens"] >= 200)

rate = df.groupby(["model", "thinking", "arm", "mode"])["clean_200"].mean().groupby(["model", "thinking", "arm"]).mean() * 100
table = rate.unstack("thinking").rename(columns={True: "on", False: "off"})
for m in MODELS:
    t = table.loc[m]
    t["gain_on"] = t["on"] - t.loc["baseline_rerun", "on"]
    t["gain_off"] = t["off"] - t.loc["baseline_rerun", "off"]
    print(f"\n{m} (% of rows clean through 200 tokens; gain vs baseline_rerun)")
    print(t.round(1).to_string())
