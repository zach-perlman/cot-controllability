"""exp12 phase 4 (results/exp12_gemma_lenses/manifest.json, key "phase4"): does the fit text matter?
The stages scripts/run_exp12_phase4.sh adds after phase 3.

  aya-prompts      the 1000-prompt Aya list: rows rendered with Gemma's chat template, at least 128 tokens,
                   16 languages interleaved round-robin. Written once with a stamp; a rerun must reproduce it.
  fit-router       the K=8 router on those 1000 prompts (jpp_cli.fit_router with router_K8's settings).
  fit              the J++ LRP fit on the first 64, writing only the final experts and pooled R-lens: no
                   raw-sum checkpoint, so the disk needs 4.2 GB instead of 13 (the manifest's deviation_disk).
  per-eval-report  per eval: recall@10 and mean log10 rank of Aya minus WikiText and of the fit-to-fit
                   replicate, paired item-bootstrap CIs, and the manifest's rules A1-A4.

Run with /venv/main/bin/python from the repo root; HF_HOME must point at the model cache.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t

from exp12_lenses import (
    BOOTSTRAP_RESAMPLES,
    EVAL_DATA_DIR,
    MACRO_EVALS,
    MIN_MACRO_ITEMS,
    MODEL,
    jpp_cli,
    load_model,
    parse_ints,
    per_item_values,
)
from workspace_lens.config import LensConfig
from workspace_lens.fitting.condense_experts import ExpertJacobians
from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
from workspace_lens.routing.router import ActivationRouterCollection

logger = logging.getLogger("exp12.phase4")

AYA = "CohereLabs/aya_dataset"
# Fixed order: the round-robin cycles through these, so the first 64 prompts hold 4 per language.
AYA_LANGUAGES = (
    "English", "Spanish", "French", "German", "Portuguese", "Italian", "Dutch", "Polish", "Russian",
    "Simplified Chinese", "Japanese", "Korean", "Standard Arabic", "Hindi", "Turkish", "Vietnamese",
)
# Every prompt fills the recipe's 128-token window, as the WikiText records (>= 600 characters) do.
MIN_TOKENS = jpp_cli.RECIPE_MAX_SEQ_LEN
SEED = 0


### aya-prompts


def chat_messages(row: dict) -> list[dict[str, str]]:
    return [{"role": "user", "content": row["inputs"]}, {"role": "assistant", "content": row["targets"]}]


def render(tokenizer, row: dict) -> str:
    """The row as a user turn and a model turn in Gemma's chat template, without the leading <bos>:
    the fit tokenizer adds it (jlens.from_hf sets add_bos_token)."""
    text = tokenizer.apply_chat_template(chat_messages(row), tokenize=False)
    assert text.startswith(tokenizer.bos_token), text[:20]
    return text[len(tokenizer.bos_token):]


def template_ids(tokenizer, row: dict) -> list[int]:
    ids = tokenizer.apply_chat_template(chat_messages(row), tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def aya_prompts(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from huggingface_hub import dataset_info
    from transformers import AutoTokenizer

    revision = dataset_info(AYA).sha
    rows = load_dataset(AYA, split="train", revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.add_bos_token = True
    eval_items = json.loads(Path(EVAL_DATA_DIR, "lens-eval-multilingual.json").read_text())["items"]
    eval_prompts = [item["prompt"] for item in eval_items]

    # Per language, (row index, rendered text) in dataset order, so the seeded shuffle is reproducible.
    candidates: dict[str, list[tuple[int, str]]] = {language: [] for language in AYA_LANGUAGES}
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if row["language"] not in candidates:
            continue
        text = render(tokenizer, row)
        if text in seen or any(prompt in text for prompt in eval_prompts):
            continue
        seen.add(text)
        if len(tokenizer(text)["input_ids"]) >= MIN_TOKENS:
            candidates[row["language"]].append((index, text))

    per_language = -(-args.num_prompts // len(AYA_LANGUAGES))
    short = {language: len(rows_) for language, rows_ in candidates.items() if len(rows_) < per_language}
    if short:
        raise SystemExit(f"fewer than {per_language} usable rows: {short}")
    for language in AYA_LANGUAGES:
        random.Random(SEED).shuffle(candidates[language])
    interleaved = [candidates[language][i] for i in range(per_language) for language in AYA_LANGUAGES]
    chosen = interleaved[: args.num_prompts]
    prompts = [text for _, text in chosen]

    for index, text in chosen:  # what the fit tokenizes is exactly the chat template's token sequence
        assert tokenizer(text)["input_ids"] == template_ids(tokenizer, rows[index]), index

    path = Path(args.out)
    if path.exists():
        if json.loads(path.read_text()) != prompts:
            raise SystemExit(f"{path} holds a different prompt list")
        logger.info("%s already holds this list", path)
        return
    path.write_text(json.dumps(prompts, ensure_ascii=False))
    lengths = sorted(len(tokenizer(text)["input_ids"]) for text in prompts)
    jpp_cli.write_stamp(jpp_cli.stamp_path_beside(str(path)), {
        "stage": "exp12 aya-prompts", "dataset": AYA, "revision": revision, "languages": list(AYA_LANGUAGES),
        "seed": SEED, "min_tokens": MIN_TOKENS, "num_prompts": len(prompts),
        "usable_rows_per_language": {language: len(rows_) for language, rows_ in candidates.items()},
        "row_indices": [index for index, _ in chosen],
        "tokens_before_truncation": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]},
    })
    logger.info("%d prompts -> %s (%d-%d tokens)", len(prompts), path, lengths[0], lengths[-1])


def load_prompts(path: str, num_prompts: int) -> list[str]:
    prompts = json.loads(Path(path).read_text())
    if len(prompts) < num_prompts:
        raise SystemExit(f"{path} has {len(prompts)} prompts, {num_prompts} asked")
    return prompts[:num_prompts]


### fit-router


def fit_router(args: argparse.Namespace) -> None:
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists")
    router = jpp_cli.fit_router(
        load_model(),
        load_prompts(args.prompts_json, args.num_prompts),
        layers=args.layers,
        num_clusters=jpp_cli.RECIPE_NUM_CLUSTERS,
        projection_dim=64,
        skip_first_n_positions=jpp_cli.RECIPE_SKIP_FIRST_N_POSITIONS,
        max_seq_len=jpp_cli.RECIPE_MAX_SEQ_LEN,
        seed=SEED,
    )
    jpp_cli.write_stamp(jpp_cli.stamp_path_beside(args.out), {"stage": "exp12 fit-router", "args": vars(args)})
    router.save(args.out)
    logger.info("router -> %s", args.out)


### fit


class FinalExpertsTrainer(ExpertJacobianRelPTrainer):
    """Writes no raw-sum checkpoint. At the end of the fit it builds the experts and the pooled R-lens
    from the running sums with ExpertJacobians.from_sums, as exp12_lenses.SnapshotMixin does, and saves
    them under .partial names that are renamed once both are complete."""

    experts_out: str
    pooled_out: str
    pooled_name: str

    def write_checkpoint(self, *, final: bool = False) -> None:
        if not final:
            return
        self.config.num_prompts_trained_on = self.completed_prompt_count
        (num_clusters, router), = self.router_collections_K_dict.items()
        sums = self.fit_sums_K_dict[num_clusters]
        config = dataclasses.replace(
            self.config.with_router(num_clusters=num_clusters, projection_dim=router.projection_dim),
            checkpoint_name=Path(self.experts_out).stem,
        )
        experts = ExpertJacobians.from_sums(sums, config, min_kept_positions=jpp_cli.RECIPE_MIN_KEPT_POSITIONS)
        experts.save(self.experts_out + ".partial")
        experts.pooled_lens(checkpoint_name=self.pooled_name).save(self.pooled_out + ".partial", dtype=t.float32)
        os.replace(self.experts_out + ".partial", self.experts_out)
        os.replace(self.pooled_out + ".partial", self.pooled_out)
        jpp_cli.write_stamp(jpp_cli.stamp_path_beside(self.pooled_out), {
            "stage": "exp12 aya fit",
            "experts": self.experts_out,
            "num_prompts_trained_on": self.completed_prompt_count,
            "position_counts": {str(layer): s.position_count_E.tolist() for layer, s in sums.layer_sums_L_dict.items()},
            "fallback_experts": {str(layer): fallback.nonzero().flatten().tolist()
                                 for layer, fallback in experts.fallback_L_dict_Bool_E.items()},
        })
        logger.info("experts -> %s, pooled lens -> %s", self.experts_out, self.pooled_out)


def fit(args: argparse.Namespace) -> None:
    if os.path.exists(args.pooled_out):
        raise SystemExit(f"{args.pooled_out} exists")
    router = ActivationRouterCollection.load(args.router_path)
    config = LensConfig(
        hf_model_name=MODEL,
        checkpoint_name=f"{Path(args.pooled_out).stem}/shard0of1",
        artifacts_base_dir=args.artifacts_dir,
        source_layers=args.layers,
        relative_end_transport_layer=-1,
        jacobian_rows_per_pass=16,
        max_seq_len=jpp_cli.RECIPE_MAX_SEQ_LEN,
        skip_first_n_positions=jpp_cli.RECIPE_SKIP_FIRST_N_POSITIONS,
        checkpoint_every_n_prompts=None,
        lrp_mode="rlens",
    )
    trainer = FinalExpertsTrainer(
        config, load_model(), load_prompts(args.prompts_json, args.num_prompts),
        router_collections_K_dict={router.num_clusters: router},
    )
    trainer.experts_out = args.experts_out
    trainer.pooled_out = args.pooled_out
    trainer.pooled_name = Path(args.pooled_out).stem
    trainer.fit()


### per-eval-report

# (lens, reference): the two fit-text contrasts, then the fit-to-fit replicates that serve as their yardstick.
CONTRASTS = [("jpp_aya_64p", "jpp_64p"), ("rlens_aya_64p", "rlens_64p")]
REPLICATES = {"jpp_aya_64p": ("jpp_64p_b", "jpp_64p"), "rlens_aya_64p": ("rlens_64p_b", "rlens_64p")}


def paired_differences(values: pd.DataFrame, lens: str, reference: str, evals: list[str],
                       rng: np.random.Generator) -> dict[str, tuple[float, float, float]]:
    """Per eval, the macro over ``evals`` and the macro over the non-multilingual evals: the mean of the
    item differences ``lens - reference`` and its 95% percentile CI, items resampled within each eval
    (the same draws for every row)."""
    diff = (values.loc[lens] - values.loc[reference]).dropna()
    per_eval = {e: diff[e].to_numpy() for e in evals}
    english = [e for e in evals if e != "multilingual"]

    def summarise(means: dict[str, float]) -> dict[str, float]:
        return {**means, "macro": np.mean([means[e] for e in evals]),
                "macro_without_multilingual": np.mean([means[e] for e in english])}

    point = summarise({e: d.mean() for e, d in per_eval.items()})
    draws = [summarise({e: rng.choice(d, size=len(d)).mean() for e, d in per_eval.items()})
             for _ in range(BOOTSTRAP_RESAMPLES)]
    return {row: (float(point[row]), *map(float, np.percentile([d[row] for d in draws], [2.5, 97.5])))
            for row in point}


def per_eval_report(args: argparse.Namespace) -> None:
    ranks = pd.read_csv(args.ranks)
    graded = pd.read_csv(args.correctness_csv)[["eval", MODEL]]
    kept = graded[graded[MODEL].isna() | (graded[MODEL] == True)].groupby("eval").size()  # noqa: E712
    evals = [e for e in MACRO_EVALS if kept.get(e, 0) >= MIN_MACRO_ITEMS]
    assert "multilingual" in evals, evals
    lenses = sorted(set(ranks["lens"]))
    rng = np.random.default_rng(0)
    metrics = {"recall@10": per_item_values(ranks, 10, None), "mean log10 rank (lower is better)":
               per_item_values(ranks, None, None)}
    lines = [f"evals {evals}; items resampled within eval, {BOOTSTRAP_RESAMPLES} resamples, 95% percentile", ""]
    out: dict = {"evals": evals}

    for metric, values in metrics.items():
        lines.append(f"## {metric} per eval")
        lines.append("  " + " " * 15 + "".join(f"{e:>14s}" for e in evals))
        for lens in lenses:
            lines.append(f"  {lens:15s}" + "".join(f"{values.loc[lens][e].mean():14.3f}" for e in evals))
        lines.append("")
        pairs = [p for p in CONTRASTS + list(REPLICATES.values()) if set(p) <= set(lenses)]
        out[metric] = {}
        for lens, reference in pairs:
            result = paired_differences(values, lens, reference, evals, rng)
            out[metric][f"{lens} - {reference}"] = result
            lines.append(f"  {lens} - {reference}:")
            for row, (d, lo, hi) in result.items():
                lines.append(f"    {row:28s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append("")

    recall = out["recall@10"]
    lines.append("## rules (manifest phase4; recall@10)")
    for rule, (lens, reference) in zip(("A1 (J++)", "A2 (R-lens)"), CONTRASTS):
        key = f"{lens} - {reference}"
        if key not in recall:
            lines.append(f"  {rule}: {lens} missing")
            continue
        d, lo, hi = recall[key]["multilingual"]
        replicate_key = "{} - {}".format(*REPLICATES[lens])
        if replicate_key in recall:
            yardstick = abs(recall[replicate_key]["multilingual"][0])
            helps = lo > 0 and d > yardstick
            basis = f"needs CI low end > 0 (is {lo:+.3f}) and the difference > |replicate| (is {yardstick:.3f})"
        else:
            helps = lo > 0
            basis = f"needs CI low end > 0 (is {lo:+.3f}); replicate missing, so CI only (flagged)"
        lines.append(f"  {rule} multilingual {d:+.3f} [{lo:+.3f}, {hi:+.3f}]: {basis} -> "
                     f"{'Aya helps' if helps else 'no detectable help'}")
        e_d, e_lo, e_hi = recall[key]["macro_without_multilingual"]
        lines.append(f"     A3 other evals {e_d:+.3f} [{e_lo:+.3f}, {e_hi:+.3f}] -> "
                     f"{'costs English' if e_hi < 0 else 'no detectable English cost'}")
        m_d, m_lo, m_hi = recall[key]["macro"]
        lines.append(f"     A4 macro {m_d:+.3f} [{m_lo:+.3f}, {m_hi:+.3f}]")
    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(text + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(out, indent=1))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("aya-prompts")
    p.add_argument("--num-prompts", type=int, default=jpp_cli.NUM_WIKITEXT_FIT_PROMPTS)
    p.add_argument("--out", required=True)

    p = sub.add_parser("fit-router")
    p.add_argument("--prompts-json", required=True)
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--layers", type=parse_ints, required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("fit")
    p.add_argument("--prompts-json", required=True)
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--router-path", required=True)
    p.add_argument("--layers", type=parse_ints, required=True)
    p.add_argument("--experts-out", required=True)
    p.add_argument("--pooled-out", required=True)
    p.add_argument("--artifacts-dir", required=True)

    p = sub.add_parser("per-eval-report")
    p.add_argument("--ranks", required=True)
    p.add_argument("--correctness-csv", required=True)
    p.add_argument("--out", required=True)

    args = parser.parse_args()
    {"aya-prompts": aya_prompts, "fit-router": fit_router, "fit": fit,
     "per-eval-report": per_eval_report}[args.stage](args)


if __name__ == "__main__":
    main()
