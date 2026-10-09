"""exp12 readouts for the lens notebook (results/exp12_gemma_lenses/manifest.json, key "readouts").

  fit-jlens  the plain J-lens for google/gemma-4-31B-it: the gradient (no-LRP) fit on the first N
             recipe WikiText prompts at the readout layers, the recipe otherwise (128 tokens, first 16
             positions skipped, 16 rows per pass). Writes only the final pooled lens: no raw-sum
             checkpoint and no experts file (the disk is nearly full).
  readouts   per held-out item, layer and lens: each intermediate's rank (jpp_cli evaluate's rule,
             Readout Filtering on) and the lens's top tokens, plus the model's own top tokens at the
             readout position. The ranks of every lens also in --reference-ranks must equal them.

Run with /venv/main/bin/python from the repo root; HF_HOME must point at the model cache.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path

import pandas as pd
import torch as t

from exp12_lenses import EVAL_DATA_DIR, MODEL, READOUT_LAYERS, jpp_cli, load_model, parse_ints
from lens_evals.readout_evals.readout_evals import (
    ReadoutEvalRunner,
    _item_readout_and_candidates,
    min_rank_over_candidates,
)
from workspace_lens import get_hf_model
from workspace_lens.config import LensConfig
from workspace_lens.fitting.condense_experts import ExpertJacobians
from workspace_lens.fitting.expert_fitting import ExpertJacobianTrainer
from workspace_lens.routing.router import ActivationRouterCollection
from workspace_lens.utils import record_activations

logger = logging.getLogger("exp12.readouts")

TOP_K = 5


### fit-jlens


class PooledOnlyGradientTrainer(ExpertJacobianTrainer):
    """The gradient fit, writing at the end only the pooled lens built from the running sums (as
    exp12_lenses.SnapshotMixin builds its pooled snapshots)."""

    pooled_out: str

    def write_checkpoint(self, *, final: bool = False) -> None:
        if not final:
            return
        self.config.num_prompts_trained_on = self.completed_prompt_count
        (num_clusters, router), = self.router_collections_K_dict.items()
        config = dataclasses.replace(
            self.config.with_router(num_clusters=num_clusters, projection_dim=router.projection_dim),
            checkpoint_name=Path(self.pooled_out).stem,
        )
        experts = ExpertJacobians.from_sums(self.fit_sums_K_dict[num_clusters], config,
                                            min_kept_positions=jpp_cli.RECIPE_MIN_KEPT_POSITIONS)
        experts.pooled_lens(checkpoint_name=Path(self.pooled_out).stem).save(self.pooled_out + ".partial",
                                                                              dtype=t.float32)
        os.replace(self.pooled_out + ".partial", self.pooled_out)
        jpp_cli.write_stamp(jpp_cli.stamp_path_beside(self.pooled_out), {
            "stage": "exp12 fit-jlens", "lrp_mode": "none",
            "num_prompts_trained_on": self.completed_prompt_count, "layers": self.config.source_layers,
        })
        logger.info("J-lens -> %s", self.pooled_out)


def fit_jlens(args: argparse.Namespace) -> None:
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists")
    router = ActivationRouterCollection.load(args.router_path)
    config = LensConfig(
        hf_model_name=MODEL,
        checkpoint_name=f"{Path(args.out).stem}/shard0of1",
        artifacts_base_dir=args.artifacts_dir,
        source_layers=args.layers,
        relative_end_transport_layer=-1,
        jacobian_rows_per_pass=16,
        max_seq_len=jpp_cli.RECIPE_MAX_SEQ_LEN,
        skip_first_n_positions=jpp_cli.RECIPE_SKIP_FIRST_N_POSITIONS,
        checkpoint_every_n_prompts=None,
        lrp_mode="none",
    )
    trainer = PooledOnlyGradientTrainer(config, load_model(), jpp_cli.load_fit_prompts(args.num_prompts),
                                        router_collections_K_dict={router.num_clusters: router})
    trainer.pooled_out = args.out
    trainer.fit()


### readouts


class TopTokenRunner(ReadoutEvalRunner):
    """ReadoutEvalRunner.run's readout (same position, candidates, exclusions and rank rule), also
    keeping each lens's top tokens and the model's own."""

    @t.no_grad()
    def readouts(self, items) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        tokenizer = self.model.tokenizer
        decode = lambda ids: [tokenizer.decode([i]) for i in ids]  # noqa: E731
        final_layer = self.model.n_layers - 1
        rank_rows, top_rows, item_rows = [], [], []
        for item in items:
            readout = _item_readout_and_candidates(self.model, item, self.max_seq_len)
            if readout is None:
                continue
            position = readout.readout_position
            _, activations = record_activations(self.model, item.prompt, self.max_seq_len,
                                                sorted({*self.layers, final_layer}))
            model_logits = self.model.unembed(activations[final_layer][[position]].float())[0]
            item_rows.append({
                "eval": item.eval_slug, "item": item.name, "prompt": item.prompt,
                "intermediates": json.dumps(list(readout.candidate_ids_by_intermediate)),
                "readout_token": tokenizer.decode([int(readout.input_ids_Int_1S[0, position])]),
                "model_top_tokens": json.dumps(decode(model_logits.topk(TOP_K).indices.tolist())),
            })
            for lens_name, lens in self.lenses.items():
                for layer in self.layers:
                    logits = self._lens_logits(lens, activations[layer][[position]].float(), layer)
                    top_rows.append({"eval": item.eval_slug, "lens": lens_name, "item": item.name, "layer": layer,
                                     "top_tokens": json.dumps(decode(logits.topk(TOP_K).indices.tolist()))})
                    for intermediate, candidate_ids in readout.candidate_ids_by_intermediate.items():
                        rank_rows.append({"eval": item.eval_slug, "lens": lens_name, "item": item.name,
                                          "intermediate": intermediate, "layer": layer,
                                          "rank": min_rank_over_candidates(logits, candidate_ids)})
            if len(item_rows) % 25 == 0:
                logger.info("item %d/%d", len(item_rows), len(items))
        return pd.DataFrame(rank_rows), pd.DataFrame(top_rows), pd.DataFrame(item_rows)


def check_against_reference(ranks: pd.DataFrame, reference_path: str) -> list[str]:
    """The lenses shared with the reference ranks file, after asserting their ranks are identical."""
    reference = pd.read_csv(reference_path)
    shared = sorted(set(ranks["lens"]) & set(reference["lens"]))
    key = ["eval", "lens", "item", "intermediate", "layer"]
    ours = ranks[ranks["lens"].isin(shared)].set_index(key)["rank"].sort_index()
    theirs = reference[reference["lens"].isin(shared)].set_index(key)["rank"].sort_index()
    assert ours.index.equals(theirs.index), "item/intermediate/layer rows differ from the reference"
    mismatches = int((ours != theirs).sum())
    assert mismatches == 0, f"{mismatches} ranks differ from {reference_path}"
    return shared


def readouts(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    if (out / "ranks.csv").exists():
        raise SystemExit(f"{out}/ranks.csv exists")
    model = get_hf_model(args.hf_model_name, attn_implementation="sdpa")
    lenses = jpp_cli.load_lenses(args.lens, model, hf_model_name=args.hf_model_name, layers=args.layers)
    _, items, excluded_token_ids = jpp_cli.load_items_and_exclusions(
        model, eval_data_dir=EVAL_DATA_DIR, correctness_csv=args.correctness_csv,
        hf_model_name=args.hf_model_name, layers=args.layers,
        readout_max_seq_len=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN, items_spec="held-out:0")
    runner = TopTokenRunner(model, lenses, layers=args.layers, max_seq_len=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN,
                            excluded_token_ids=excluded_token_ids)
    ranks, top_tokens, item_table = runner.readouts(items)
    shared = check_against_reference(ranks, args.reference_ranks)
    logger.info("ranks identical to %s for %s", args.reference_ranks, shared)
    out.mkdir(parents=True, exist_ok=True)
    ranks.to_csv(out / "ranks.csv", index=False)
    top_tokens.to_csv(out / "top_tokens.csv", index=False)
    item_table.to_csv(out / "items.csv", index=False)
    jpp_cli.write_stamp(jpp_cli.stamp_path_beside(str(out / "ranks.csv")), {
        "stage": "exp12 readouts", "args": vars(args), "lenses": dict(zip(lenses, args.lens, strict=True)),
        "items": "held-out:0", "num_items": len(item_table), "num_excluded_token_ids": len(excluded_token_ids),
        "ranks_identical_to_reference_for": shared, "top_k": TOP_K,
    })


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("fit-jlens")
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--router-path", required=True)
    p.add_argument("--layers", type=parse_ints, default=READOUT_LAYERS)
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("readouts")
    p.add_argument("--hf-model-name", required=True)
    p.add_argument("--lens", nargs="+", required=True)
    p.add_argument("--layers", type=parse_ints, required=True)
    p.add_argument("--correctness-csv", required=True)
    p.add_argument("--reference-ranks", required=True)
    p.add_argument("--out-dir", required=True)

    args = parser.parse_args()
    {"fit-jlens": fit_jlens, "readouts": readouts}[args.stage](args)


if __name__ == "__main__":
    main()
