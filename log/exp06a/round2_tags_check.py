"""Tags-only sensitivity for round 2's CoT specificity (v3 item 33). A thinking-off row with no <output_reasoning>
content counts as a violation at token 0, and the no-tag share falls under the round-2 arms (Qwen3.8: 32% at
baseline, 5% under stacked_all), so part of each thinking-off gain may be "used the tags". Recomputes the
clean-through-200 thinking-off gain among tagged rows only, next to the thinking-on gain (all rows, same 51
questions). Gains are vs the in-environment baseline rerun, 5-rule mean, percentage points."""
import sys

sys.path.insert(0, "scripts")
import cc_config as cfg
import cc_exp06a
import cc_exp06a_analysis as an
import pandas as pd

ARMS = ["baseline_rerun", "stacked_rerun", "stacked", *cfg.EXP06A_ROUND2_ARMS]
MODELS = ["Qwen3.8-27B-FP8", "Gemma-4-31B-FP8", "Qwen3-32B"]

raw, _ = an.exp06a_rows(MODELS, skip_missing=True)
df = an.score(raw[[c for c in an.COLUMNS if c in raw.columns]], True)
df = df[df["item_id"].isin([it["item_id"] for it in cc_exp06a.thinking_off_items()]) & df["arm"].isin(ARMS)]
df["clean_200"] = (~df["event"] | (df["time"] >= 200)) & (df["reasoning_tokens"] >= 200)
# thinking-on stacked is named stacked_rerun; thinking-off stacked is named stacked
df.loc[df["arm"] == "stacked_rerun", "arm"] = "stacked"


def rate(rows):
    return rows.groupby(["model", "arm", "mode"])["clean_200"].mean().groupby(["model", "arm"]).mean() * 100


on = df[df["thinking"].astype(bool)]
off = df[~df["thinking"].astype(bool)]
table = pd.DataFrame({
    "on": rate(on),
    "off_all": rate(off),
    "off_tagged": rate(off[~off["empty"]]),
    "no_tag_%": off.groupby(["model", "arm"])["empty"].mean() * 100,
})
for m in MODELS:
    t = table.loc[m].copy()
    base = t.loc["baseline_rerun"]
    t["gain_on"] = t["on"] - base["on"]
    t["gain_off_all"] = t["off_all"] - base["off_all"]
    t["gain_off_tagged"] = t["off_tagged"] - base["off_tagged"]
    t["on_minus_off_tagged"] = t["gain_on"] - t["gain_off_tagged"]
    print(f"\n{m}: clean through 200 tokens, 51 questions")
    print(t.round(1).to_string())
