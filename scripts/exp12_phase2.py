"""exp12 phase 2 (results/exp12_gemma_lenses/manifest.json, key "phase2"): the stages
scripts/run_exp12_phase2.sh adds to phase 1 (scripts/exp12_lenses.py, left as it ran).

  evaluate-transfer  jpp_cli evaluate, plus lenses fitted on the base google/gemma-4-31B
                     (the released koayon/jpp-lenses J++ lens) scored on gemma-4-31B-it. Their
                     config is relabelled -it in memory so the library's model-match check
                     passes; items, Readout Filtering and scoring are jpp_cli's.
  fit-long           the J++ LRP fit on long WikiText records (512 tokens), several routers
                     sharing one set of backward passes, resumable, with one snapshot part way.
  offload            upload files to the private HF dataset, check the remote sha256, and with
                     --delete remove the local copy.

Run with /venv/main/bin/python from the repo root; HF_HOME must point at the model cache.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import logging
import os
from pathlib import Path

import torch as t

from exp12_lenses import (
    EVAL_DATA_DIR,
    MODEL,
    READOUT_LAYERS,
    REPO,
    jpp_cli,
    load_model,
    parse_ints,
)
from workspace_lens.config import LensConfig
from workspace_lens.fitting.condense_experts import ExpertJacobians
from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
from workspace_lens.fitting.types import ExpertFitSums
from workspace_lens.fitting.utils import expert_checkpoint_filename
from workspace_lens.routing.router import ActivationRouterCollection
from workspace_lens.utils import load_lens_file

logger = logging.getLogger("exp12.phase2")

BASE_MODEL = "google/gemma-4-31B"
HF_DATASET = "zachperlman20/cot-controllability-cache"


### evaluate-transfer


def evaluate_transfer(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; evaluations are written once")
    model = load_model()
    lenses = jpp_cli.load_lenses(args.lens, model, hf_model_name=MODEL, layers=READOUT_LAYERS)
    transferred = {}
    for spec in args.transferred:
        name, _, path = spec.partition("=")
        if not path or name in lenses:
            raise SystemExit(f"--transferred wants a new NAME=PATH, got {spec!r}")
        lens = load_lens_file(path, hf_model_name=BASE_MODEL)
        if lens.config.hf_model_name != BASE_MODEL:
            raise SystemExit(f"{path} was fitted on {lens.config.hf_model_name!r}, not {BASE_MODEL!r}")
        lens.config = dataclasses.replace(lens.config, hf_model_name=MODEL)
        lenses[name] = lens
        transferred[name] = {"path": path, "fitted_on": BASE_MODEL,
                             "source_layers": sorted(lens.source_layers)}
    _, items, excluded_token_ids = jpp_cli.load_items_and_exclusions(
        model,
        eval_data_dir=EVAL_DATA_DIR,
        correctness_csv=args.correctness_csv,
        hf_model_name=MODEL,
        layers=READOUT_LAYERS,
        readout_max_seq_len=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN,
        items_spec=args.items,
    )
    jpp_cli.warn_about_in_sample_items(args.lens, items)
    ranks_df, pass_df = jpp_cli.evaluate(
        model, lenses, items, layers=READOUT_LAYERS, excluded_token_ids=excluded_token_ids
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pass_path = jpp_cli.sibling_path(str(out), "_pass_at_k.csv")
    jpp_cli.write_stamp(jpp_cli.stamp_path_beside(str(out)), {
        "stage": "exp12 evaluate-transfer", "args": vars(args), "transferred": transferred,
        "num_excluded_token_ids": len(excluded_token_ids),
        "scored_item_names": [item.name for item in items],
    })
    ranks_df.to_csv(out, index=False)
    pass_df.to_csv(pass_path, index=False)
    print(pass_df[pass_df["eval"] == "macro"].to_string(index=False))


### fit-long


def load_long_prompts(path: Path, num_prompts: int, min_chars: int, min_tokens: int, tokenizer) -> list[str]:
    """The first ``num_prompts`` WikiText-103 records of at least ``min_chars`` characters
    (jlens's loader and order), each required to reach ``min_tokens`` tokens so every record
    fills the fit's ``max_seq_len``. Saved to ``path`` on first use; later runs (a resume)
    must load the same list."""
    from jlens.examples import load_wikitext_prompts

    prompts = load_wikitext_prompts(num_prompts, min_chars=min_chars)
    if len(prompts) < num_prompts:
        raise SystemExit(f"only {len(prompts)} records of >= {min_chars} chars")
    lengths = [len(tokenizer(p)["input_ids"]) for p in prompts]
    if min(lengths) < min_tokens:
        raise SystemExit(f"records shorter than {min_tokens} tokens: {lengths}")
    if path.exists():
        if json.loads(path.read_text()) != prompts:
            raise SystemExit(f"{path} holds a different prompt list")
    else:
        path.write_text(json.dumps(prompts))
    logger.info("%d long prompts, %d-%d tokens before truncation", len(prompts), min(lengths), max(lengths))
    return prompts


class LongSnapshotTrainer(ExpertJacobianRelPTrainer):
    """After the checkpoint write at ``snapshot_at`` prompts: the experts of router
    K=``snapshot_k`` and the pooled R-lens (the sum over experts, the same for every router)
    from the running sums, saved as ``experts_k<K>_<n>p.pt`` and ``<pooled_name>_<n>p.pt`` in
    ``snapshot_dir``. A snapshot already on disk is left alone (resume re-reaches the count)."""

    snapshot_dir: str
    snapshot_at: int
    snapshot_k: int
    pooled_name: str

    def write_checkpoint(self, *, final: bool = False) -> None:
        super().write_checkpoint(final=final)
        n = self.completed_prompt_count
        if n != self.snapshot_at:
            return
        experts_path = os.path.join(self.snapshot_dir, f"experts_k{self.snapshot_k}_{n:02d}p.pt")
        pooled_path = os.path.join(self.snapshot_dir, f"{self.pooled_name}_{n:02d}p.pt")
        if os.path.exists(pooled_path):
            return
        router = self.router_collections_K_dict[self.snapshot_k]
        sums = ExpertFitSums(dict(self.fit_sums_K_dict[self.snapshot_k].layer_sums_L_dict))
        config = dataclasses.replace(
            self.config.with_router(num_clusters=self.snapshot_k, projection_dim=router.projection_dim),
            checkpoint_name=f"snapshot_{n:02d}p",
        )
        experts = ExpertJacobians.from_sums(sums, config, min_kept_positions=jpp_cli.RECIPE_MIN_KEPT_POSITIONS)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        experts.save(experts_path)
        experts.pooled_lens(checkpoint_name=f"{self.pooled_name}_{n:02d}p").save(pooled_path, dtype=t.float32)
        jpp_cli.write_stamp(jpp_cli.stamp_path_beside(pooled_path), {
            "stage": "fit-long snapshot",
            "num_prompts_trained_on": n,
            "num_clusters": self.snapshot_k,
            "position_counts": {str(layer): s.position_count_E.tolist()
                                for layer, s in sums.layer_sums_L_dict.items()},
            "fallback_experts": {str(layer): fallback.nonzero().flatten().tolist()
                                 for layer, fallback in experts.fallback_L_dict_Bool_E.items()},
        })
        logger.info("snapshot after %d prompts: %s, %s", n, experts_path, pooled_path)


def fit_long(args: argparse.Namespace) -> None:
    """exp12_lenses.py fit (jpp_cli fit-shard's config fields, one shard, resumable), with
    long prompts, ``--max-seq-len``/``--rows-per-pass`` exposed, and one checkpoint file per
    router. Writes ``fit_done.json`` in the checkpoint directory when finished."""
    routers = {r.num_clusters: r for r in map(ActivationRouterCollection.load, args.router_paths)}
    jpp_cli.check_distinct_router_ks(list(routers.values()))
    if args.snapshot_k not in routers:
        raise SystemExit(f"--snapshot-k {args.snapshot_k} is not among the routers' Ks {sorted(routers)}")
    checkpoint_name = f"{args.name}/shard0of1"
    first_file = expert_checkpoint_filename(min(routers))
    existing = sorted(glob.glob(os.path.join(args.artifacts_dir, "*", checkpoint_name)))
    existing = [d for d in existing if os.path.exists(os.path.join(d, first_file))]
    if len(existing) > 1:
        raise SystemExit(f"several checkpoints for {args.name}: {existing}")
    if existing and os.path.exists(os.path.join(existing[0], "fit_done.json")):
        raise SystemExit(f"{existing[0]} is already finished")
    model = load_model()
    prompts = load_long_prompts(Path(args.prompts_json), args.num_prompts, args.min_chars,
                                args.max_seq_len, model.tokenizer)
    if existing:
        trainer = LongSnapshotTrainer.from_checkpoint_dir(
            existing[0], prompts, router_collections_K_dict=routers, model=model
        )
    else:
        config = LensConfig(
            hf_model_name=MODEL,
            checkpoint_name=checkpoint_name,
            artifacts_base_dir=args.artifacts_dir,
            source_layers=args.layers,
            relative_end_transport_layer=-1,
            jacobian_rows_per_pass=args.rows_per_pass,
            max_seq_len=args.max_seq_len,
            skip_first_n_positions=jpp_cli.RECIPE_SKIP_FIRST_N_POSITIONS,
            checkpoint_every_n_prompts=args.checkpoint_every,
            lrp_mode="rlens",
        )
        trainer = LongSnapshotTrainer(config, model, prompts, router_collections_K_dict=routers)
    trainer.snapshot_dir = args.snapshot_dir
    trainer.snapshot_at = args.snapshot_at
    trainer.snapshot_k = args.snapshot_k
    trainer.pooled_name = args.pooled_name
    trainer.fit()
    checkpoint_dir = trainer.config.checkpoint_path
    jpp_cli.write_stamp(os.path.join(checkpoint_dir, "fit_done.json"), {
        "stage": "exp12 fit-long", "args": vars(args), "checkpoint_dir": checkpoint_dir,
        "num_prompts_trained_on": trainer.completed_prompt_count,
    })
    Path(args.checkpoint_dir_out).write_text(checkpoint_dir + "\n")
    print(checkpoint_dir)


### offload


def file_digest(path: Path, digest) -> str:
    with path.open("rb") as f:
        while chunk := f.read(64 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def offload(args: argparse.Namespace) -> None:
    """Each file goes to the dataset under its repo-relative path. The local copy is
    deleted (--delete) only after the remote matches it: size plus the LFS sha256 for
    large files, the git blob id (sha1 of "blob <size>\\0" + content) for small ones."""
    from huggingface_hub import HfApi

    api = HfApi()
    if not api.dataset_info(HF_DATASET).private:
        raise SystemExit(f"{HF_DATASET} is not private; nothing uploaded")
    for path in map(Path, args.paths):
        rel = path.resolve().relative_to(REPO).as_posix()
        if not rel.startswith("cache/exp12/"):
            raise SystemExit(f"{rel}: only cache/exp12 files are offloaded")
        size = path.stat().st_size
        api.upload_file(path_or_fileobj=str(path), path_in_repo=rel, repo_id=HF_DATASET,
                        repo_type="dataset", commit_message=f"exp12: {rel}")
        (remote,) = api.get_paths_info(HF_DATASET, [rel], repo_type="dataset", expand=True)
        if remote.lfs is not None:
            kind, local, remote_id = "sha256", file_digest(path, hashlib.sha256()), remote.lfs.sha256
        else:
            blob = hashlib.sha1(f"blob {size}\0".encode())
            kind, local, remote_id = "git blob id", file_digest(path, blob), remote.blob_id
        if remote.size != size or remote_id != local:
            raise SystemExit(f"{rel}: remote size {remote.size} {kind} {remote_id} != local "
                             f"{size} {local}; local copy kept")
        print(f"uploaded {rel}: {size / 1e9:.2f} GB, {kind} {local} matches the remote")
        if args.delete:
            path.unlink()
            print(f"deleted local {rel}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("evaluate-transfer")
    p.add_argument("--lens", nargs="+", required=True, help="lenses fitted on gemma-4-31B-it (or 'logit')")
    p.add_argument("--transferred", nargs="+", required=True, help="NAME=PATH of a base-model lens")
    p.add_argument("--items", default="held-out:0")
    p.add_argument("--correctness-csv", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("fit-long")
    p.add_argument("--name", required=True)
    p.add_argument("--router-paths", nargs="+", required=True)
    p.add_argument("--layers", type=parse_ints, required=True)
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--min-chars", type=int, required=True)
    p.add_argument("--max-seq-len", type=int, required=True)
    p.add_argument("--rows-per-pass", type=int, required=True)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--prompts-json", required=True)
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--snapshot-at", type=int, required=True)
    p.add_argument("--snapshot-k", type=int, required=True)
    p.add_argument("--pooled-name", required=True)
    p.add_argument("--checkpoint-dir-out", required=True)

    p = sub.add_parser("offload")
    p.add_argument("paths", nargs="+")
    p.add_argument("--delete", action="store_true")

    args = parser.parse_args()
    {"evaluate-transfer": evaluate_transfer, "fit-long": fit_long, "offload": offload}[args.stage](args)


if __name__ == "__main__":
    main()
