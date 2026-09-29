"""Draw the fixed-seed item set shared by exp01 and exp02: cache/exp02/items.jsonl.

50 questions (GPQA 18, HLE 19, MMLU-Pro 13, proportional to the suite's 445 / 469 / 300) among rows that CoT-Control's
own loader keeps for word suppression (at least one keyword), so all nine modes are defined on every item. Each item
stores what every mode needs: CoT-Control's single-keyword and multi-keyword control data, the options, and the
correct letter (CoT-Control's convert_answer_to_letter). Refuses to overwrite an existing item file.

Run: /venv/main/bin/python scripts/cc_items.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import _cotcontrol
import cc_config as cfg


def load_source(run_cceval, source: str) -> list[dict]:
    """Rows of one QA source, merged across CoT-Control's per-mode loaders (keyed by its row id)."""
    path = Path(cfg.COTCONTROL_QA_DIR) / "datasets" / cfg.DATASETS[source]
    single = {s["id"]: s for s in run_cceval.load_evaluation_data(path, mode="word_suppression")}
    multi = {s["id"]: s for s in run_cceval.load_evaluation_data(path, mode="multiple_word_suppression")}
    rows = []
    for row_id in sorted(single, key=int):
        s, m = single[row_id], multi[row_id]
        rows.append({
            "item_id": f"{source}:{row_id}",
            "source": source,
            "row_id": int(row_id),
            "domain": s["domain"],
            "question": s["question"],
            "options": s["options"],
            "correct_answer": s["correct_answer"],
            "correct_letter": run_cceval.convert_answer_to_letter(s["correct_answer"], s["options"], s["question"]),
            "keyword": s["keyword"],
            "synonyms": s["synonyms"],
            "multi_keyword": m["keyword"],
            "multi_synonyms": m["synonyms"],
        })
    return rows


def main() -> None:
    if cfg.ITEMS_PATH.exists():
        raise FileExistsError(f"{cfg.ITEMS_PATH} exists; the item set is fixed once drawn")
    run_cceval = _cotcontrol.module("run_cceval")
    rng = random.Random(cfg.ITEM_SEED)
    items = []
    for source, n in cfg.ITEMS_PER_SOURCE.items():
        rows = load_source(run_cceval, source)
        chosen = rng.sample(rows, n)
        items.extend(sorted(chosen, key=lambda r: r["row_id"]))
        print(f"{source}: {n} of {len(rows)} eligible rows")
    for item in items:
        if not item["options"]:
            raise ValueError(f"{item['item_id']} has no options; the answer format needs a letter")
    cfg.ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with cfg.ITEMS_PATH.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"wrote {len(items)} items to {cfg.ITEMS_PATH}")


if __name__ == "__main__":
    main()
