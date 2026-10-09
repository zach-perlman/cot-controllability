"""exp12: fit lenses for google/gemma-4-31B-it with the public jpp_lens code (third_party/jpp_lens).

The stages jpp_lens's CLI already covers (fit-router, merge-experts, fit-weights, evaluate)
are called through it by scripts/run_exp12_lenses.sh. This file adds what the CLI lacks:

  correctness  grade the model on every readout-eval item (one greedy step), giving the CSV
               the correctness filter reads; jpp_lens ships a Qwen column only.
  fit          the one-shard fit, resumable, which also saves the sums at the readout layers
               after 8, 16 and 32 prompts: the data-efficiency check of the manifest.
  report       recall@k per lens, with paired item-bootstrap CIs against a reference lens.
  choose-n     the plain J-lens prompt count, by the manifest's rule.

Run with /venv/main/bin/python from the repo root; HF_HOME must point at the model cache.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t

REPO = Path(__file__).resolve().parents[1]
JPP_DIR = REPO / "third_party" / "jpp_lens"
sys.path[:0] = [str(JPP_DIR / "src"), str(JPP_DIR / "scripts")]

import jpp_cli  # noqa: E402
from lens_evals.readout_evals.readout_eval_items import (  # noqa: E402
    MACRO_EVALS,
    load_readout_eval_items,
)
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.config import LensConfig  # noqa: E402
from workspace_lens.fitting.condense_experts import ExpertJacobians  # noqa: E402
from workspace_lens.fitting.expert_fitting import ExpertJacobianTrainer  # noqa: E402
from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer  # noqa: E402
from workspace_lens.fitting.types import ExpertFitSums  # noqa: E402
from workspace_lens.fitting.utils import expert_checkpoint_filename  # noqa: E402
from workspace_lens.routing.router import ActivationRouterCollection  # noqa: E402

logger = logging.getLogger("exp12")

MODEL = "google/gemma-4-31B-it"
EVAL_DATA_DIR = str(JPP_DIR / "data" / "jlens" / "evaluations")
# The paper's Gemma 4 31B readout layers (depths k/8; koayon/jpp-lenses README).
READOUT_LAYERS = [8, 15, 22, 30, 38, 45, 52]
# An eval enters the macro only with at least this many items passing the correctness
# filter (the paper drops poetry for base Gemma, where 1 of 98 passes).
MIN_MACRO_ITEMS = 20
# choose-n: a snapshot is "enough" when its headline recall is within this of the full fit.
ENOUGH_TOLERANCE = 0.01
BOOTSTRAP_RESAMPLES = 2000


def load_model():
    return get_hf_model(MODEL, attn_implementation="sdpa")


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


### correctness


def correctness(args: argparse.Namespace) -> None:
    """One greedy step per item with jpp_lens's own grader; the CSV is jpp_lens's
    (every eval, the Qwen column) plus a column for this model."""
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; correctness is written once")
    items = load_readout_eval_items(EVAL_DATA_DIR)
    runner = jpp_cli.readout_runner(
        load_model(), layers=READOUT_LAYERS, max_seq_len=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN
    )
    graded = runner.grade_model_correctness(items, hf_model_name=MODEL)
    shipped = pd.read_csv(f"{EVAL_DATA_DIR}/model_correctness.csv")
    merged = shipped.merge(graded, on=["eval", "item"], how="outer", validate="one_to_one")
    assert len(merged) == len(shipped) == len(graded), "item keys differ from the shipped CSV"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    summary = merged.groupby("eval")[MODEL].agg(
        passed=lambda s: int((s == True).sum()),  # noqa: E712
        graded=lambda s: int(s.notna().sum()),
        total="size",
    )
    print(summary.to_string())


### fit with snapshots


class SnapshotMixin:
    """After the trainer's own checkpoint write, when the prompt count is one of
    ``snapshot_at``: build the experts and the pooled lens (the R-Lens of an LRP fit, the
    J-Lens of a gradient fit) from the running sums at ``snapshot_layers`` only, with the
    library's ExpertJacobians.from_sums, and save them as ``experts_<n>p.pt`` and
    ``<pooled_name>_<n>p.pt`` in ``snapshot_dir``. A snapshot already on disk is left
    alone (resume re-reaches the same counts)."""

    snapshot_dir: str
    snapshot_at: set[int]
    snapshot_layers: list[int]
    pooled_name: str
    min_kept_positions: int

    def write_checkpoint(self, *, final: bool = False) -> None:
        super().write_checkpoint(final=final)  # type: ignore[misc]
        n = self.completed_prompt_count  # type: ignore[attr-defined]
        if n not in self.snapshot_at:
            return
        experts_path = os.path.join(self.snapshot_dir, f"experts_{n:02d}p.pt")
        pooled_path = os.path.join(self.snapshot_dir, f"{self.pooled_name}_{n:02d}p.pt")
        if os.path.exists(pooled_path):
            return
        (num_clusters, router), = self.router_collections_K_dict.items()  # type: ignore[attr-defined]
        all_sums = self.fit_sums_K_dict[num_clusters].layer_sums_L_dict  # type: ignore[attr-defined]
        snapshot_sums = ExpertFitSums({layer: all_sums[layer] for layer in self.snapshot_layers})
        config = dataclasses.replace(
            self.config.with_router(  # type: ignore[attr-defined]
                num_clusters=num_clusters, projection_dim=router.projection_dim
            ),
            source_layers=list(self.snapshot_layers),
            checkpoint_name=f"snapshot_{n:02d}p",
        )
        experts = ExpertJacobians.from_sums(
            snapshot_sums, config, min_kept_positions=self.min_kept_positions
        )
        os.makedirs(self.snapshot_dir, exist_ok=True)
        experts.save(experts_path)
        experts.pooled_lens(checkpoint_name=f"{self.pooled_name}_{n:02d}p").save(
            pooled_path, dtype=t.float32
        )
        jpp_cli.write_stamp(
            jpp_cli.stamp_path_beside(pooled_path),
            {
                "stage": "fit snapshot",
                "num_prompts_trained_on": n,
                "next_prompt_idx": self.next_prompt_idx,  # type: ignore[attr-defined]
                "layers": self.snapshot_layers,
                "position_counts": {
                    str(layer): sums.position_count_E.tolist()
                    for layer, sums in snapshot_sums.layer_sums_L_dict.items()
                },
                "fallback_experts": {
                    str(layer): fallback.nonzero().flatten().tolist()
                    for layer, fallback in experts.fallback_L_dict_Bool_E.items()
                },
            },
        )
        logger.info("snapshot after %d prompts: %s, %s", n, experts_path, pooled_path)


class SnapshotRelPTrainer(SnapshotMixin, ExpertJacobianRelPTrainer):
    pass


class SnapshotGradientTrainer(SnapshotMixin, ExpertJacobianTrainer):
    pass


def fit(args: argparse.Namespace) -> None:
    """jpp_cli fit-shard's single-shard fit (same config fields), resuming from the run's
    checkpoint when one exists. Writes ``fit_done.json`` in the checkpoint directory when
    finished; the run script skips a fit that has it."""
    router = ActivationRouterCollection.load(args.router_path)
    routers = {router.num_clusters: router}
    prompts = jpp_cli.load_fit_prompts(args.num_prompts)
    trainer_class = SnapshotRelPTrainer if args.lrp_mode != "none" else SnapshotGradientTrainer
    checkpoint_name = f"{args.name}/shard0of1"
    checkpoint_file = expert_checkpoint_filename(router.num_clusters)
    existing = sorted(glob.glob(os.path.join(args.artifacts_dir, "*", checkpoint_name)))
    existing = [d for d in existing if os.path.exists(os.path.join(d, checkpoint_file))]
    if len(existing) > 1:
        raise SystemExit(f"several checkpoints for {args.name}: {existing}")
    model = load_model()
    if existing:
        if os.path.exists(os.path.join(existing[0], "fit_done.json")):
            raise SystemExit(f"{existing[0]} is already finished")
        trainer = trainer_class.from_checkpoint_dir(
            existing[0], prompts, router_collections_K_dict=routers, model=model
        )
    else:
        config = LensConfig(
            hf_model_name=MODEL,
            checkpoint_name=checkpoint_name,
            artifacts_base_dir=args.artifacts_dir,
            source_layers=args.layers,
            relative_end_transport_layer=-1,
            jacobian_rows_per_pass=16,
            max_seq_len=jpp_cli.RECIPE_MAX_SEQ_LEN,
            skip_first_n_positions=jpp_cli.RECIPE_SKIP_FIRST_N_POSITIONS,
            checkpoint_every_n_prompts=args.checkpoint_every,
            lrp_mode=args.lrp_mode,
        )
        trainer = trainer_class(config, model, prompts, router_collections_K_dict=routers)
    trainer.snapshot_dir = args.snapshot_dir
    trainer.snapshot_at = set(args.snapshot_at)
    trainer.snapshot_layers = args.snapshot_layers
    trainer.pooled_name = args.pooled_name
    trainer.min_kept_positions = jpp_cli.RECIPE_MIN_KEPT_POSITIONS
    missing = sorted(set(args.snapshot_layers) - set(trainer.source_layers))
    if missing:
        raise SystemExit(f"snapshot layers {missing} are not fit layers")
    trainer.fit()
    checkpoint_dir = trainer.config.checkpoint_path
    jpp_cli.write_stamp(
        os.path.join(checkpoint_dir, "fit_done.json"),
        {"stage": "exp12 fit", "args": vars(args), "checkpoint_dir": checkpoint_dir,
         "num_prompts_trained_on": trainer.completed_prompt_count},
    )
    Path(args.checkpoint_dir_out).write_text(checkpoint_dir + "\n")
    print(checkpoint_dir)


### report


def per_item_values(ranks: pd.DataFrame, k: int | None, layers: list[int] | None) -> pd.DataFrame:
    """[lens x (eval, item)] from each intermediate's min rank across ``layers`` (all, when
    None), averaged over the item's intermediates: with ``k``, the fraction read within
    top k (the README's recall@k); with ``k=None``, the mean log10 rank (continuous, so
    it moves before a rank crosses k)."""
    if layers is not None:
        ranks = ranks[ranks["layer"].isin(layers)]
    best = ranks.groupby(["lens", "eval", "item", "intermediate"])["rank"].min()
    value = (best <= k) if k is not None else np.log10(best)
    return value.groupby(["lens", "eval", "item"]).mean().unstack(["eval", "item"])


def macro_bootstrap(
    hits: pd.DataFrame, evals: list[str], reference: str, rng: np.random.Generator
) -> dict[str, dict[str, float]]:
    """Per lens: the macro (mean over ``evals`` of the item mean) and, against
    ``reference``, the paired difference with a 95% percentile CI from resampling items
    within each eval."""
    values = hits.to_numpy()  # [lens, item]
    eval_of_item = hits.columns.get_level_values("eval")
    columns = {e: np.flatnonzero(eval_of_item == e) for e in evals}
    ref_row = list(hits.index).index(reference)

    def macro(index_by_eval: dict[str, np.ndarray]) -> np.ndarray:
        return np.mean([values[:, idx].mean(axis=1) for idx in index_by_eval.values()], axis=0)

    point = macro(columns)
    draws = np.stack([
        macro({e: rng.choice(idx, size=len(idx)) for e, idx in columns.items()})
        for _ in range(BOOTSTRAP_RESAMPLES)
    ])  # [resample, lens]
    diffs = draws - draws[:, [ref_row]]
    lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
    return {
        lens: {"value": float(point[i]), "diff": float(point[i] - point[ref_row]),
               "diff_lo": float(lo[i]), "diff_hi": float(hi[i])}
        for i, lens in enumerate(hits.index)
    }


def report(args: argparse.Namespace) -> None:
    ranks = pd.concat([pd.read_csv(p) for p in args.ranks], ignore_index=True)
    ranks = ranks.drop_duplicates(["lens", "eval", "item", "intermediate", "layer"])
    if args.reference not in set(ranks["lens"]):
        raise SystemExit(f"reference {args.reference} not among {sorted(set(ranks['lens']))}")
    item_counts = ranks.groupby("eval")["item"].nunique()
    graded = pd.read_csv(args.correctness_csv)[["eval", MODEL]]
    kept = graded[graded[MODEL].isna() | (graded[MODEL] == True)].groupby("eval").size()  # noqa: E712
    evals = [e for e in MACRO_EVALS if kept.get(e, 0) >= MIN_MACRO_ITEMS]
    dropped = [e for e in MACRO_EVALS if e not in evals]
    layers = sorted(ranks["layer"].unique().tolist())
    rng = np.random.default_rng(0)

    lines = [f"items scored per eval: {item_counts.to_dict()}",
             f"items kept by the correctness filter: {kept.to_dict()}",
             f"macro over {evals}" + (f" (dropped, fewer than {MIN_MACRO_ITEMS} kept: {dropped})"
                                      if dropped else ""),
             f"reference lens: {args.reference}; CIs: paired item bootstrap within eval, "
             f"{BOOTSTRAP_RESAMPLES} resamples, 95% percentile", ""]
    out: dict = {"evals": evals, "dropped_evals": dropped, "reference": args.reference,
                 "k": args.k, "layers": layers}

    def table(title: str, result: dict[str, dict[str, float]]) -> None:
        lines.append(title)
        for lens, r in result.items():
            lines.append(f"  {lens:12s} {r['value']:7.3f}   vs ref {r['diff']:+.3f} "
                         f"[{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}]")
        lines.append("")

    out["headline"] = macro_bootstrap(per_item_values(ranks, args.k, None), evals, args.reference, rng)
    table(f"headline recall@{args.k} (min rank over layers {layers}):", out["headline"])
    out["log10_rank"] = macro_bootstrap(per_item_values(ranks, None, None), evals, args.reference, rng)
    table("mean log10 rank (min over layers; lower is better):", out["log10_rank"])

    out["per_layer"] = {
        str(layer): macro_bootstrap(per_item_values(ranks, args.k, [layer]), evals, args.reference, rng)
        for layer in layers
    }
    lines.append(f"per-layer recall@{args.k} (macro), diff vs ref in brackets:")
    lines.append("  " + " " * 12 + "".join(f"{layer:>15d}" for layer in layers))
    for lens in out["headline"]:
        lines.append(f"  {lens:12s}" + "".join(
            f"{out['per_layer'][str(layer)][lens]['value']:7.3f} ({out['per_layer'][str(layer)][lens]['diff']:+.3f})"
            for layer in layers))
    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(text + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(out, indent=1))


def choose_n(args: argparse.Namespace) -> None:
    """The smallest snapshot count whose pooled lens's headline recall is within
    ENOUGH_TOLERANCE of the full fit's pooled lens; the full count when none is."""
    headline = json.loads(Path(args.report_json).read_text())["headline"]
    full = headline[f"rlens_{args.full:02d}p"]["value"]
    chosen = args.full
    for n in sorted(args.candidates):
        if abs(headline[f"rlens_{n:02d}p"]["value"] - full) <= ENOUGH_TOLERANCE:
            chosen = n
            break
    print(chosen)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("correctness")
    p.add_argument("--out", required=True)

    p = sub.add_parser("fit")
    p.add_argument("--name", required=True, help="checkpoint name, e.g. jpp or jlens")
    p.add_argument("--router-path", required=True)
    p.add_argument("--lrp-mode", default="rlens")
    p.add_argument("--layers", type=parse_ints, required=True)
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--checkpoint-every", type=int, default=4)
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--snapshot-at", type=parse_ints, default=[])
    p.add_argument("--snapshot-layers", type=parse_ints, default=READOUT_LAYERS)
    p.add_argument("--pooled-name", required=True, help="rlens (LRP fit) or jlens (gradient fit)")
    p.add_argument("--checkpoint-dir-out", required=True, help="file that receives the checkpoint dir")

    p = sub.add_parser("report")
    p.add_argument("--ranks", nargs="+", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--correctness-csv", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("choose-n")
    p.add_argument("--report-json", required=True)
    p.add_argument("--candidates", type=parse_ints, default=[8, 16, 32])
    p.add_argument("--full", type=int, default=64)

    args = parser.parse_args()
    {"correctness": correctness, "fit": fit, "report": report, "choose-n": choose_n}[args.stage](args)


if __name__ == "__main__":
    main()
