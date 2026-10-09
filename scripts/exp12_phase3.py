"""exp12 phase 3 (results/exp12_gemma_lenses/manifest.json, key "phase3"): the stages
scripts/run_exp12_phase3.sh adds after phase 2.

  qwen-summary       the macro recall@10 of the Qwen pipeline check (P1) beside the paper's values.
  brew-gate          one forward pass of google/gemma-4-31B-it per row of an exp11 brew bank: the
                     10 colour logits at the answer position (G1). Rows go to cache (item ids only).
  brew-gate-report   accuracy, exp11's shortcut families against their permutation null, and the
                     single-table stir tally, for both banks; aggregates only.
  fit-replicate      the J++ LRP fit on a slice of the WikiText list (prompts skip..skip+num),
                     disjoint from the prompts of the fits it replicates (R).

Run with /venv/main/bin/python from the repo root; HF_HOME must point at the model cache.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch as t

from exp12_lenses import MODEL, jpp_cli, load_model, parse_ints
from workspace_lens.config import LensConfig
from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
from workspace_lens.fitting.utils import expert_checkpoint_filename
from workspace_lens.routing.router import ActivationRouterCollection

import exp11 as X
import exp11_brew as B

logger = logging.getLogger("exp12.phase3")

PAPER_QWEN = {"logit": 31.1, "jpp_released": 55.2, "jlens_camila": 35.7, "rlens_camila": 37.7}
SHORTCUT_FAMILIES = ("stop_early", "skip_one_stir", "reverse_order", "last_stir_only", "wrong_column", "inverse_lookup")
N_PERM = 200
N_BOOT = 2000


### qwen-summary


def qwen_summary(args: argparse.Namespace) -> None:
    table = pd.read_csv(args.pass_csv)
    macro = table[(table["eval"] == "macro") & (table["k"] == 10) & (table["weighting"] == "item")]
    lines = ["P1: macro recall@10 (item weighting) on Qwen/Qwen3.6-27B, held-out:0, vs the paper's Table 1",
             "(paper J/R rows may be the authors' own fits; jlens_camila/rlens_camila are descriptive)", ""]
    ours = {}
    for lens, value in zip(macro["lens"], macro["pass_at_k"]):
        ours[lens] = 100 * value
        paper = PAPER_QWEN.get(lens)
        lines.append(f"  {lens:14s} ours {ours[lens]:5.1f}   paper {paper if paper is not None else '-':>5}"
                     + (f"   diff {ours[lens] - paper:+.1f}" if paper is not None else ""))
    validated = (abs(ours.get("logit", float("nan")) - 31.1) <= 1.0
                 and abs(ours.get("jpp_released", float("nan")) - 55.2) <= 1.5)
    lines += ["", f"rule (|logit - 31.1| <= 1.0 and |jpp_released - 55.2| <= 1.5): "
                  f"{'VALIDATED' if validated else 'NOT VALIDATED'}"]
    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(text + "\n")


### brew gate


def gate_rows_path(bank: str) -> Path:
    return Path(f"cache/exp12/brew_gate/{bank}.jsonl")


def brew_gate(args: argparse.Namespace) -> None:
    import cc_exp10_render as R
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    out = gate_rows_path(args.bank)
    if out.exists():
        raise SystemExit(f"{out} exists; captures are written once")
    items = {r["item_id"]: r for r in X.load_bank(args.bank)}
    rows = [json.loads(line) for line in X.requests_path(args.bank).open()]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    colours = sorted(B.parse(items[rows[0]["item_id"]]["problem"]).table)
    colour_ids = []
    for colour in colours:
        ids = tokenizer.encode(" " + colour, add_special_tokens=False)
        assert len(ids) == 1, f"{colour!r} is {len(ids)} tokens"
        colour_ids.append(ids[0])
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=t.bfloat16, device_map="cuda").eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for n, row in enumerate(rows):
        ids = R.prompt_ids(tokenizer, "gemma4", row)
        if n == 0:
            logger.info("first prompt ends with %r", tokenizer.decode(ids[-6:]))
        with t.no_grad():
            logits = model(input_ids=t.tensor([ids], device="cuda"), use_cache=False).logits[0, -1].float().cpu()
        colour_logits = logits[colour_ids]
        records.append({
            "item_id": row["item_id"], "h": row["h"], "format": row.get("format"), "n_tokens": len(ids),
            "colours": colours, "colour_logits": [round(float(x), 4) for x in colour_logits],
            "predicted": colours[int(colour_logits.argmax())],
            "colour_mass": round(float(t.softmax(logits, -1)[colour_ids].sum()), 6),
            "top_token_is_colour": int(logits.argmax()) in colour_ids,
        })
        if n % 100 == 0:
            logger.info("%s %d/%d", args.bank, n, len(rows))
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    logger.info("%d rows -> %s", len(records), out)


def family_shares(wrong: list[dict], items: dict, same_h: list[str], rng: random.Random) -> dict[str, tuple]:
    """Per shortcut family: (observed share of wrong answers in the family's set, its bootstrap 95% CI, the
    permutation null: the share when each wrong answer is checked against another same-h item's set)."""
    sets = {iid: {f: s - {items[iid]["answer"]} for f, s in B.hypotheses(B.parse(items[iid]["problem"])).items()}
            for iid in same_h}
    out = {}
    for family in SHORTCUT_FAMILIES:
        hits = [r["predicted"] in sets[r["item_id"]][family] for r in wrong]
        boot = sorted(sum(rng.choices(hits, k=len(hits))) / len(hits) for _ in range(N_BOOT))
        perm = []
        for _ in range(N_PERM):
            permuted = []
            for r in wrong:
                other = rng.choice([i for i in same_h if i != r["item_id"]])
                if r["predicted"] != items[other]["answer"]:
                    permuted.append(r["predicted"] in sets[other][family])
            perm.append(sum(permuted) / max(len(permuted), 1))
        out[family] = (sum(hits) / len(hits), boot[int(0.025 * N_BOOT)], boot[int(0.975 * N_BOOT) - 1],
                       sum(perm) / N_PERM)
    return out


def brew_gate_report(args: argparse.Namespace) -> None:
    rng = random.Random(0)
    lines = ["G1: google/gemma-4-31B-it, no-CoT brew (exp11 banks), answer = argmax over the 10 colour tokens",
             "Qwen3.6-27B reference (exp11): three columns 1.00/0.12/0.05, last_stir_only 0.75 of h=2 errors "
             "(null 0.11); single table 1.00/0.76/0.35", ""]
    gate = {}
    for bank in ("test", "single_test"):
        rows = [json.loads(line) for line in gate_rows_path(bank).open()]
        items = {r["item_id"]: r for r in X.load_bank(bank)}
        by_h = defaultdict(list)
        for r in rows:
            r["correct"] = r["predicted"] == items[r["item_id"]]["answer"]
            by_h[r["h"]].append(r)
        lines.append(f"## bank {bank}")
        for h, group in sorted(by_h.items()):
            n = len(group)
            acc = sum(r["correct"] for r in group) / n
            mass = sum(r["colour_mass"] for r in group) / n
            top = sum(r["top_token_is_colour"] for r in group) / n
            lines.append(f"  h={h} n={n} accuracy {acc:.3f}  colour mass {mass:.3f}  top token a colour {top:.3f}")
            wrong = [r for r in group if not r["correct"]]
            if bank == "test" and h >= 2 and wrong:
                same_h = [iid for iid, it in items.items() if it["h"] == h and it["split"] == "eval"]
                shares = family_shares(wrong, items, same_h, rng)
                for family, (obs, lo, hi, null) in shares.items():
                    lines.append(f"    {family:15s} {obs:.3f} [{lo:.3f}, {hi:.3f}]  null {null:.3f}  "
                                 f"excess {obs - null:+.3f}")
                if h == 2:
                    gate = {"accuracy": acc, "n_wrong": len(wrong), "shares": shares}
            if bank == "single_test" and wrong:
                tally = defaultdict(int)
                for r in wrong:
                    b = B.parse(items[r["item_id"]]["problem"])
                    hops = [k for k in range(h + 4) if b.apply(b.start, [B.SINGLE] * k) == r["predicted"]]
                    tally[f"{min(hops, key=lambda k: abs(k - h)) - h:+d} stirs" if hops else "off_cycle"] += 1
                lines.append("    wrong by stirs applied: " + " ".join(f"{k}={v}" for k, v in sorted(tally.items())))
        lines.append("")
    a = gate["accuracy"] <= 0.5 and gate["n_wrong"] >= 100
    best_family, (obs, lo, _, null) = max(gate["shares"].items(), key=lambda kv: kv[1][0] - kv[1][3])
    b = obs - null >= 0.20 and lo > null
    verdict = ("PASS: structured errors, an exp11 replication is worth a manifest" if a and b else
               "errors but unstructured: exp11's contrasts do not transfer" if a else
               "Gemma composes: no error to read")
    lines.append(f"gate (three columns h=2): (a) accuracy {gate['accuracy']:.3f} <= 0.5 with {gate['n_wrong']} >= 100 "
                 f"wrong: {a}; (b) best family {best_family} {obs:.3f} vs null {null:.3f} "
                 f"(excess >= 0.20 and CI low end {lo:.3f} > null): {b} -> {verdict}")
    text = "\n".join(lines)
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")


### fit-replicate


def load_prompt_slice(args: argparse.Namespace, tokenizer) -> list[str]:
    """Prompts skip..skip+num of the WikiText records of >= min_chars characters (jlens's loader and order), checked
    disjoint from every list in --disjoint-from, each at least --min-tokens long; saved to --prompts-json on first
    use, and later runs (a resume) must load the same list."""
    from jlens.examples import load_wikitext_prompts

    prompts = load_wikitext_prompts(args.skip + args.num_prompts, min_chars=args.min_chars)[args.skip:]
    if len(prompts) < args.num_prompts:
        raise SystemExit(f"only {len(prompts)} records after skipping {args.skip}")
    for path in args.disjoint_from:
        other = json.loads(Path(path).read_text())
        if set(other) & set(prompts):
            raise SystemExit(f"replicate prompts overlap {path}")
    lengths = [len(tokenizer(p)["input_ids"]) for p in prompts]
    if min(lengths) < args.min_tokens:
        raise SystemExit(f"records shorter than {args.min_tokens} tokens: {sorted(lengths)[:5]}")
    path = Path(args.prompts_json)
    if path.exists():
        if json.loads(path.read_text()) != prompts:
            raise SystemExit(f"{path} holds a different prompt list")
    else:
        path.write_text(json.dumps(prompts))
    logger.info("%d prompts (skip %d), %d-%d tokens before truncation", len(prompts), args.skip, min(lengths), max(lengths))
    return prompts


def save_prompts(args: argparse.Namespace) -> None:
    """The prompt list of an existing fit, for --disjoint-from: the first num records of >= min_chars."""
    from jlens.examples import load_wikitext_prompts

    Path(args.out).write_text(json.dumps(load_wikitext_prompts(args.num_prompts, min_chars=args.min_chars)))


def fit_replicate(args: argparse.Namespace) -> None:
    """exp12_phase2.py fit-long with a prompt slice and no snapshot (library trainer, one checkpoint per router,
    resumable). Writes fit_done.json in the checkpoint directory when finished."""
    routers = {r.num_clusters: r for r in map(ActivationRouterCollection.load, args.router_paths)}
    jpp_cli.check_distinct_router_ks(list(routers.values()))
    checkpoint_name = f"{args.name}/shard0of1"
    first_file = expert_checkpoint_filename(min(routers))
    existing = sorted(glob.glob(os.path.join(args.artifacts_dir, "*", checkpoint_name)))
    existing = [d for d in existing if os.path.exists(os.path.join(d, first_file))]
    if len(existing) > 1:
        raise SystemExit(f"several checkpoints for {args.name}: {existing}")
    if existing and os.path.exists(os.path.join(existing[0], "fit_done.json")):
        raise SystemExit(f"{existing[0]} is already finished")
    model = load_model()
    prompts = load_prompt_slice(args, model.tokenizer)
    if existing:
        trainer = ExpertJacobianRelPTrainer.from_checkpoint_dir(
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
        trainer = ExpertJacobianRelPTrainer(config, model, prompts, router_collections_K_dict=routers)
    trainer.fit()
    checkpoint_dir = trainer.config.checkpoint_path
    jpp_cli.write_stamp(os.path.join(checkpoint_dir, "fit_done.json"), {
        "stage": "exp12 fit-replicate", "args": vars(args), "checkpoint_dir": checkpoint_dir,
        "num_prompts_trained_on": trainer.completed_prompt_count,
    })
    Path(args.checkpoint_dir_out).write_text(checkpoint_dir + "\n")
    print(checkpoint_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("qwen-summary")
    p.add_argument("--pass-csv", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("brew-gate")
    p.add_argument("--bank", choices=["test", "single_test"], required=True)

    p = sub.add_parser("brew-gate-report")
    p.add_argument("--out", required=True)

    p = sub.add_parser("save-prompts")
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--min-chars", type=int, required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("fit-replicate")
    p.add_argument("--name", required=True)
    p.add_argument("--router-paths", nargs="+", required=True)
    p.add_argument("--layers", type=parse_ints, required=True)
    p.add_argument("--skip", type=int, required=True)
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--min-chars", type=int, required=True)
    p.add_argument("--min-tokens", type=int, required=True)
    p.add_argument("--max-seq-len", type=int, required=True)
    p.add_argument("--rows-per-pass", type=int, required=True)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--disjoint-from", nargs="+", required=True)
    p.add_argument("--prompts-json", required=True)
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--checkpoint-dir-out", required=True)

    args = parser.parse_args()
    {"qwen-summary": qwen_summary, "brew-gate": brew_gate, "brew-gate-report": brew_gate_report,
     "save-prompts": save_prompts, "fit-replicate": fit_replicate}[args.stage](args)


if __name__ == "__main__":
    main()
