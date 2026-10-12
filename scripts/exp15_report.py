"""exp15 analysis: the manifest, the read report (P1, P2), the matcher audit and the external check.

Kept apart from exp15.py so that this file can change while run_exp15.sh runs the pipeline stages (which import
exp15, exp15_readers, exp15_edits and exp15_score only).

  manifest draft|locked    results/exp15_wrong_keys/manifest_draft.json | manifest.json (written once each)
  read_report --bank B --tag T    P1 and P2 for every available reader -> results/exp15_wrong_keys/T/
  select_writes --tag T           each method's write setting from the dev grid -> write_lock.json (once)
  write_report --bank B --tag T   P3 and P4 at the locked settings
  audit_sample | check_matcher    day-0 check 8 (50 readouts labelled blind)
  check_workspacebench            day-0 check 9
  check_zero_edit                 the edit harness at alpha = 0 reproduces the captured answers
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import exp15 as E
from exp15 import (AO_EXPLORATORY_LAYERS, AO_PRIMARY_LAYERS, DEV_BANK, EXP, MODEL, PRIMARY_CELLS, PRIMARY_LAYERS,
                   TEST_SPEC, TEXT_READ_SPEC, WHITENER_SPEC, atomic_write_text, check_dir, item_table, load_capture,
                   load_lens_reads, load_probe_reads, load_text_reads, probe_reads_path, require_absent, text_jobs,
                   text_reads_path)


# --- manifest -------------------------------------------------------------------------------------------------------
DEVIATIONS = [
    "Consistency gate: p(a') >= 0.3 is primary (plan: 0.5, sensitivity 0.8). At 0.5 dev kept 3 of 52 h=3 off-cycle "
    "errors; 0.0, 0.5 and 0.8 are reported as sensitivity (0.8 is expected to be empty: no dev off-cycle error "
    "reaches it).",
    "Test bank enlarged to 100 / 600 / 2000 items at h = 1 / 2 / 3 (plan: 100 / 300 / 1000), per K2: dev projects "
    "26 + 136 consistent off-cycle errors at the 0.3 gate.",
    "Primary read cells chosen on dev: question_end, nl_asst (the newline after 'assistant'), final. think_close was "
    "a candidate; its reads were at chance on dev.",
    "transformers stores hidden_states[64] after the final norm, so lenses read layers 0..62 (as planned) and no "
    "reader sees the raw block-63 output.",
    "J++ (koayon/jpp-lenses) ships one Jacobian per layer, so the plan's 'expert mixture evaluated at the clean h' "
    "does not apply; its single matrix is used for reads and writes.",
    "Lens write directions include the final norm's gain g: v = J_l^T (g * W_U[c]) (logit lens: g * W_U[c]); exp14 "
    "used W_U[c] J without g.",
    "agu18dec's pooled whitener is gated. Our own (natural-text passages, WHITENER_SPEC) stands in: ptag AR "
    "retrieval is scored in our whitened space, and ptag AR write directions are mapped back with our S^1/2.",
    "AO prompt: both renderings of the README's concepts_raw prompt were scored in day-0 check 6 and the README's "
    "line-break rendering was kept.",
    "Day-0 check 6 used the next 20 tokens (the template-lens passages are about 45 tokens long), not 32.",
    "Day-0 check 4 (lens sanity) fails its literal bar (>= 0.90 top-1 at L60; observed 0.76 to 0.86). On items "
    "where the model's colour softmax is confident (p >= 0.8) every lens is >= 0.99; misses concentrate on h=3 "
    "items with p < 0.8, where the last block often flips the top colour. The lenses are kept as readers; their "
    "reads on uncertain items are noisier, which is reported.",
    "Text readers on test run on all off-cycle errors plus seeded subsamples of 300 correct items per h (P1) and "
    "300 miscounts (secondary), to bound generation cost.",
    "The probe and diffmeans are trained on correct h=2 and h=3 items pooled (target: the key, s_{h-1}); other rows "
    "are scored by the mean of the five fold models.",
    "P2 pools h=2 and h=3 off-cycle errors (strata by h are reported); P4 uses h=3 only, as planned.",
    "Reader scores: rank-average within (cell, layer) pairs for continuous readers; mean naming fraction for text "
    "readers (the plan's 'fraction of samples naming c').",
    "The prompt-only analytic baseline gives AUC(w) = 0.5 by construction (w is never on the true path of an "
    "off-cycle error); the thinking-on baseline runs with the self-report secondary.",
    "Write settings are frozen in a separate write_lock.json (after the dev grid, before any test write trial), so "
    "the locked manifest is never edited.",
    "Matcher extended to Chinese colour words before the lock: about a quarter of dev AO samples answer in Chinese "
    "(often a bare character, e.g. '绿 ('), which the English-only matcher missed (recall 0.946 on a 30-readout "
    "audit of such samples; 4.3% of AO samples had a missed colour). A second hand-labelled audit (matcher_cjk) "
    "checks the extended matcher; the first audit is re-scored with it. The external check (day-0 check 9) ran "
    "with the English-only matcher.",
    "Day-0 check 7 (ptag AR) failed: retrieval@1 at L44 is 0.48 in our whitened space (bar 0.8, published 0.93). "
    "As pre-registered, ptag_ar stays out of the write tests unless debugging (at most 3 hours, logged in "
    "checks_day0/ptag_debug*.json) brings it over the bar.",
    "Added pipeline check: the edit harness at alpha = 0 (right-padded batches, hook installed) must reproduce the "
    "captured answers (checks_day0/zero_edit.json).",
]


def manifest_record(status: str) -> dict:
    """The pre-registration: question, items, readers, tests, write grid and selection rules, statistics, kill
    rules, day-0 check results and every deviation from the plan."""
    import exp15_edits as ED
    import exp15_readers as RD
    import exp15_score as S
    checks = {p.stem: json.loads(p.read_text()) for p in sorted(check_dir().glob("*.json"))}

    def brief(name, keys):
        return {k: checks[name][k] for k in keys if name in checks and k in checks[name]}
    return {
        "experiment": EXP.exp_id, "status": status, "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan": "/root/.cursor/plans/exp15_wrong-key_read_write_abef3741.plan.md",
        "question": ("On Qwen3.6-27B's no-CoT single-table brew errors that are off the true cycle, is the wrong "
                     "intermediate w = T^-1(answer) readable before the answer exists (read), and can additive "
                     "edits of matched size swap a key on correct items or repair an error (write)?"),
        "model": {"name": MODEL, "rendering": "exp10 C0: one h=2 shot, thinking off, 'Answer:' prefilled",
                  "layer_convention": "layer l = output of decoder block l = hidden_states[l + 1]",
                  "kernels": "flash-linear-attention 0.5.2; cuDNN SDPA disabled; causal_conv1d absent (torch path)"},
        "items": {"dev_bank": f"exp11 {DEV_BANK} (250 per h, seen in exp11)", "test_bank": TEST_SPEC,
                  "groups": "correct | offcycle (answer != T^k(s0) for k in 0..h+3) | miscount (unique k) | "
                            "miscount_ambiguous",
                  "consistent_p": S.CONSISTENT_P, "sensitivity_p": S.SENSITIVITY_P,
                  "controls": "all colours minus (true path, implied path, T(answer))",
                  "text_read_rows": TEXT_READ_SPEC},
        "readers": {
            "token_lenses": {"kinds": TOKEN_READERS, "score": "readout logits (template: cosine) over the 10 colours"},
            "probe": "exp11_probe features + fit (PCA-64, multinomial LR, wd 1e-2), 5-fold cross-fit",
            "diffmeans": "mean(x | key=c) - mean(x | key!=c), same folds",
            "text": {name: {**RD.READERS[name], "primary_layers": TEXT_PRIMARY_LAYERS[name]} for name in TEXT_READERS},
            "nla_adapter": RD.NLA_ADAPTER, "ao_prompt": RD.AO_PROMPT, "matcher": S.COLOUR_PATTERNS,
            "primary_cells": PRIMARY_CELLS, "primary_layers": PRIMARY_LAYERS,
            "ao_exploratory_layers": AO_EXPLORATORY_LAYERS},
        "tests": {
            "P1": "correct items, per h in {2, 3}: AUC of S(key = s_{h-1}) vs controls",
            "P2": {"population": f"off-cycle errors, h in {{2, 3}}, p(a') >= {S.CONSISTENT_P}",
                   "primary": ["AUC of S(w) vs controls", "direction test: S(w) vs S(T(answer))"],
                   "secondary": ["answer-leak filtered AUC (pairs where J++ ranks the answer outside its top 3)",
                                 "hypothesis decomposition: top among {w, true s_{h-1}, controls}",
                                 "strata: gold is / is not the runner-up colour; h = 2 / 3; gates 0.0 / 0.5 / 0.8",
                                 "AUC of the true s_{h-1} vs controls"]},
            "P3": {"population": "correct h=2 items", "edit": "c = s1 -> c' = s1' (s1', T(s1'), T(T(s1')) off the "
                                                              "true path and distinct)",
                   "outcomes": "success T(s1') | start_swap T(T(s1')) | echo s1' | unchanged | other",
                   "variants": {"answer_swap": "c = s2 -> c' = a' (ceiling and gate: success >= 0.30 on dev)",
                                "random_null": "seeded random unit direction, same alpha and layer",
                                "other_item": "c, c' = two colours unrelated to the item"},
                   "primary_metric": "per-item key_swap success - random_null success at the locked setting",
                   "head_to_head": "every method at layer 42, at its best dev alpha there"},
            "P4": {"population": f"off-cycle errors, h = 3, p(a') >= {S.CONSISTENT_P}",
                   "edit": "c = w -> c' = true s2; success = gold",
                   "controls": ["redirect: c' = r off both paths, success = T(r)", "random-direction null"],
                   "eligibility": "methods with dev key_swap success - null >= 0.10 that pass the gate"}},
        "secondary_exploratory": {
            "path_specificity": "S(true s_{h-1}) vs S(s_0..s_{h-2}) on correct items and errors, and S(w) vs the same "
                                "on errors (does a reader favour the held key, or path colours in general?)",
            "count_target": "miscount items with a unique implied k: S(k) vs other counts; S(h) vs other counts",
            "count_repair": "consistent miscounts: count word k -> h (logit, J, J++, template); dev grid, test at the "
                            "dev-best setting; random null",
            "self_report": "no-CoT 'colour just before the last stir' after the model's own answer; thinking-on "
                           "prompt-only baseline on 100 test off-cycle errors",
            "nla_iter300": "AV iter_000300 vs iter_000800 at the primary cells, layer 42",
            "best_of_16": "16 AV samples at the final cell ranked by AR cosine: top-1 vs random-1 vs all",
            "ar_edit": "per-item key-swap direction AR(explanation, s1 -> s1') - AR(explanation) at layer 42, all alphas,"
                       " vs the template nla_ar direction on the same items",
            "read_then_write": "P4 with c = each reader's decoded key (top colour other than the answer) instead of w",
            "ptag_rule": "ptag_ar enters the write tests only if day-0 check 7 passes (retrieval@1 >= 0.8 at L44)"},
        "writes": {"edit": "h_p += alpha ||h_p|| unit(v_l(c') - v_l(c)) at block-l output, positions start colour "
                           "token .. final token",
                   "methods": ED.METHODS, "layers": ED.METHOD_LAYERS, "alphas": ED.ALPHAS,
                   "directions": {"logit": "g * W_U[c]", "jlens/rlens/jpp": "J_l^T (g * W_U[c])",
                                  "template": "template-lens row (layer l, c)",
                                  "diffmeans": "dev correct h=2/3 items, final cell, key classes",
                                  "nla_ar": {"layer": 42, "templates": ED.NLA_TEMPLATES},
                                  "ptag_ar": {"templates": ED.PTAG_TEMPLATES, "whitener": WHITENER_SPEC}},
                   "selection_rule": ("per method, on dev: the (layer, alpha) maximising key_swap success - "
                                      "random_null success; ties -> smaller alpha, then earlier layer. Gate: the "
                                      "method's best answer_swap success over its grid >= 0.30, else 'cannot write "
                                      "at this edit size' (K3)."),
                   "lock_file": "write_lock.json"},
        "statistics": {"bootstrap": "items, 10k resamples, percentile 95% CI", "paired": "reader comparisons on "
                       "shared items", "multiplicity": "Holm over readers x {P1 h=2, P1 h=3, P2 AUC, P2 direction}",
                       "positive": "CI lower bound > 0.5 (AUC) or > 0 (null-adjusted rate)"},
        "kill_rules": {"K1": "no reader reads the on-path key on dev -> exploratory sweep, then count target",
                       "K2": "< 100 projected consistent off-cycle test errors -> enlarge h=3 bank (applied)",
                       "K3": "answer_swap gate failed -> 'cannot write at this edit size'",
                       "K4": "no method reaches dev key_swap - null >= 0.10 -> repair reported untestable"},
        "day0_checks": {
            "behaviour": brief("behaviour", ["pass", "max_abs_diff"]),
            "lens_sanity": brief("lens_sanity", ["pass_by_lens"]),
            "nla": brief("nla", ["cos_hidden43", "fve_mean", "pass_convention", "pass_fve"]),
            "oracles": {k: {kk: checks["oracles"][k][kk] for kk in ("overlap_true", "overlap_shuffled", "pass")}
                        for k in ("olens", "ao_one_line", "ao_readme_breaks")} if "oracles" in checks else {},
            "ptag": {"pass": checks["ptag"]["pass"], "L44": checks["ptag"]["layers"]["44"]} if "ptag" in checks else {},
            "matcher": brief("matcher", ["precision", "recall", "pass"]),
            "matcher_cjk": brief("matcher_cjk", ["precision", "recall", "pass"]),
            "zero_edit": brief("zero_edit", ["same_answer", "max_abs_dp", "pass"]),
            "external_workspacebench": brief("workspacebench", ["gold_rate", "offpath_rate", "pass"])},
        "deviations_from_plan": DEVIATIONS,
        "confounds": ["answer leak: a reader naming w because the answer's rule line is active (direction test, "
                      "J++ leak filter)", "path co-activation: the probe at h=3 mixes s1 and s2 on correct items",
                      "edit size: alpha * ||h|| matched across methods; random and other-item nulls",
                      "text readers: colour words in readouts can come from the prompt's rule list (controls are "
                      "colours in the same rule list)"],
    }


def write_manifest(status: str) -> None:
    path = EXP.results / ("manifest.json" if status == "locked" else "manifest_draft.json")
    require_absent(path)
    atomic_write_text(json.dumps(manifest_record(status), indent=1) + "\n", path)
    print(f"-> {path}")


# --- read report (P1, P2) -------------------------------------------------------------------------------------------
TOKEN_READERS = ("logit", "jlens", "rlens", "jpp", "template")
TRAINED_READERS = ("probe", "diffmeans")
TEXT_READERS = ("nla", "olens", "ao")
TEXT_PRIMARY_LAYERS = {"nla": (42,), "olens": (42,), "ao": AO_PRIMARY_LAYERS}


def reader_scores(bank: str) -> dict:
    """Every available reader's colour scores at its primary (cell, layer) pairs, as exp15_score.ReaderScores."""
    import exp15_readers as RD
    import exp15_score as S
    lr = load_lens_reads(bank)
    colours = list(lr["labels"][:10])
    pc = [list(lr["cells"]).index(c) for c in PRIMARY_CELLS]
    pairs = [(c, l) for c in PRIMARY_CELLS for l in PRIMARY_LAYERS]
    out = {}
    for kind in TOKEN_READERS:
        x = lr["scores"][kind][:, pc][:, :, list(PRIMARY_LAYERS), :10].float()
        out[kind] = S.ReaderScores(kind, "rank", pairs, {i: x[n].reshape(-1, 10) for n, i in enumerate(lr["item_ids"])})
    if probe_reads_path(bank).exists():
        pr = load_probe_reads(bank)
        assert list(pr["labels"]) == colours and tuple(pr["layers"]) == PRIMARY_LAYERS
        for kind in TRAINED_READERS:
            x = pr["scores"][kind]
            out[kind] = S.ReaderScores(kind, "rank", pairs,
                                       {i: x[n].reshape(-1, 10) for n, i in enumerate(pr["item_ids"])})
    for name in TEXT_READERS:
        if not text_reads_path(bank, name).exists():
            continue
        tpairs = [(c, l) for c, l in text_jobs(name) if l in TEXT_PRIMARY_LAYERS[name]]
        scores = {}
        for item_id, r in load_text_reads(bank, name).items():
            scores[item_id] = torch.stack([
                S.naming_fraction([RD.explanation_text(name, s) for s in r[f"{c}|{l}"]], colours, S.colours_named)
                for c, l in tpairs])
        out[name] = S.ReaderScores(name, "fraction", tpairs, scores)
    return out


def leak_keep(bank: str, table, pairs: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
    """Answer-leak filter: per item, the (cell, layer) pairs where J++ does not yet rank the model's answer in its
    top 3 colours."""
    lr = load_lens_reads(bank)
    colours = list(lr["labels"][:10])
    cells = list(lr["cells"])
    jpp = lr["scores"]["jpp"][..., :10].float()
    ci = [cells.index(c) for c, _ in pairs]
    li = [l for _, l in pairs]
    out = {}
    for n, it in enumerate(table):
        s = jpp[n, ci, li]                                         # [n_pairs, 10]
        above = (s > s[:, colours.index(it.pred)][:, None]).sum(-1)
        out[it.item_id] = above >= 3
    return out


def read_groups(table, metas) -> dict[str, list]:
    """Item sets of the read tests. P2 pools h=2 and h=3 off-cycle errors at p(a') >= CONSISTENT_P."""
    import exp15_score as S
    runner_up = {}
    for m in metas:
        order = sorted(range(10), key=lambda k: -m["colour_logits"][k])
        runner_up[m["item_id"]] = m["colours"][order[1]]
    off = [it for it in table if it.h in (2, 3) and it.group == "offcycle"]
    groups = {"P1_correct_h2": [it for it in table if it.h == 2 and it.group == "correct"],
              "P1_correct_h3": [it for it in table if it.h == 3 and it.group == "correct"],
              "P2_offcycle": [it for it in off if it.p_pred >= S.CONSISTENT_P]}
    for p in S.SENSITIVITY_P:
        groups[f"P2_offcycle_p>={p}"] = [it for it in off if it.p_pred >= p]
    for h in (2, 3):
        groups[f"P2_offcycle_h{h}"] = [it for it in groups["P2_offcycle"] if it.h == h]
    groups["P2_offcycle_gold_runner_up"] = [it for it in groups["P2_offcycle"] if runner_up[it.item_id] == it.gold]
    groups["P2_offcycle_gold_not_runner_up"] = [it for it in groups["P2_offcycle"] if runner_up[it.item_id] != it.gold]
    return groups


def path_metrics(s: torch.Tensor, it, index: dict[str, int]) -> dict:
    """Specificity within the true path: S(true s_{h-1}) vs S(s_0..s_{h-2}) ('prev_vs_earlier'), and on errors S(w)
    vs the same earlier colours. A reader of 'colours on the path' in general, rather than of the held key, would
    favour s_{h-1} over w on errors without favouring it over the earlier path colours."""
    import exp15_score as S
    earlier = [index[c] for c in it.path[:-2] if c not in (it.true_prev, it.key)]
    if not earlier:
        return {}
    out = {"prev_vs_earlier": S.item_auc(s, index[it.true_prev], earlier)}
    if it.group != "correct":
        out["key_vs_earlier"] = S.item_auc(s, index[it.key], earlier)
    return out


def read_report(bank: str, tag: str) -> None:
    """P1 (key vs controls on correct items) and P2 (on off-cycle errors: key vs controls, direction test, answer-leak
    filtered key AUC, hypothesis decomposition, runner-up strata) for every available reader. Writes
    results/exp15_wrong_keys/<tag>/read_report.{json,md}; the folder must be new."""
    import exp15_score as S
    out_dir = EXP.results / tag
    out_dir.mkdir(parents=True, exist_ok=False)
    table = item_table(bank)
    metas, _ = load_capture(bank, with_hidden=False)
    colours = metas[0]["colours"]
    index = {c: k for k, c in enumerate(colours)}
    groups = read_groups(table, metas)
    readers = reader_scores(bank)
    record = {"bank": bank, "consistent_p": S.CONSISTENT_P, "primary_cells": PRIMARY_CELLS,
              "group_sizes": {g: len(v) for g, v in groups.items()}, "readers": {}}
    pvals = {}
    for name, rs in readers.items():
        keep = leak_keep(bank, table, rs.pairs)
        entry = {"pairs": [f"{c}|{l}" for c, l in rs.pairs]}
        for g, members in groups.items():
            rows = [S.read_metrics(s, it, index) | path_metrics(s, it, index) for it in members
                    if (s := rs.score(it.item_id)) is not None]
            summary = {m: S.summarise(rows, m, 0.5) for m in ("key_auc", "direction", "prev_vs_earlier")}
            if g.startswith("P2"):
                summary["key_vs_earlier"] = S.summarise(rows, "key_vs_earlier", 0.5)
                summary["true_auc"] = S.summarise(rows, "true_auc", 0.5)
                for m in ("top_key", "top_true", "top_neither"):
                    summary[m] = S.summarise(rows, m, None)
                leak_rows = [S.read_metrics(s, it, index) for it in members
                             if (s := rs.score(it.item_id, keep[it.item_id])) is not None]
                summary["key_auc_leak_filtered"] = S.summarise(leak_rows, "key_auc", 0.5)
                summary["direction_leak_filtered"] = S.summarise(leak_rows, "direction", 0.5)
            entry[g] = summary
        for g, m in (("P1_correct_h2", "key_auc"), ("P1_correct_h3", "key_auc"), ("P2_offcycle", "key_auc"),
                     ("P2_offcycle", "direction")):
            pvals[f"{name}|{g}|{m}"] = entry[g][m]["p"]
        record["readers"][name] = entry
    record["holm"] = {k: round(v, 5) for k, v in S.holm({k: v for k, v in pvals.items() if v == v}).items()}
    atomic_write_text(json.dumps(record, indent=1) + "\n", out_dir / "read_report.json")
    atomic_write_text(read_report_md(record), out_dir / "read_report.md")
    print(read_report_md(record))


def read_report_md(record: dict) -> str:
    def cell(x):
        return f"{x['mean']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] n={x['n']}" if x["n"] else "-"
    lines = [f"# exp15 read report: {record['bank']} (UNVERIFIED)", "",
             f"Consistent off-cycle gate p(a') >= {record['consistent_p']}. Group sizes: {record['group_sizes']}", "",
             "| reader | P1 h=2 key AUC | P1 h=3 key AUC | P2 key AUC | P2 direction (key vs T(a')) | "
             "P2 key AUC, leak-filtered | P2 top: key / true / neither |",
             "|---|---|---|---|---|---|---|"]
    for name, e in record["readers"].items():
        p2 = e["P2_offcycle"]
        tops = " / ".join(f"{p2[m]['mean']:.2f}" for m in ("top_key", "top_true", "top_neither"))
        lines.append(f"| {name} | {cell(e['P1_correct_h2']['key_auc'])} | {cell(e['P1_correct_h3']['key_auc'])} | "
                     f"{cell(p2['key_auc'])} | {cell(p2['direction'])} | {cell(p2['key_auc_leak_filtered'])} | {tops} |")
    lines += ["", "Sensitivity (P2 key AUC by gate):", "", "| reader | " +
              " | ".join(g for g in record["group_sizes"] if g.startswith("P2_offcycle_")) + " |",
              "|---|" + "---|" * sum(g.startswith("P2_offcycle_") for g in record["group_sizes"])]
    for name, e in record["readers"].items():
        lines.append(f"| {name} | " + " | ".join(cell(e[g]["key_auc"]) for g in record["group_sizes"]
                                                  if g.startswith("P2_offcycle_")) + " |")
    lines += ["", "Path specificity: true s_{h-1} vs the earlier path colours s_0..s_{h-2} (and on errors w vs the "
              "same):", "", "| reader | correct h=2 | correct h=3 | off-cycle: s_{h-1} vs earlier | "
              "off-cycle: w vs earlier |", "|---|---|---|---|---|"]
    for name, e in record["readers"].items():
        p2 = e["P2_offcycle"]
        lines.append(f"| {name} | {cell(e['P1_correct_h2']['prev_vs_earlier'])} | "
                     f"{cell(e['P1_correct_h3']['prev_vs_earlier'])} | {cell(p2['prev_vs_earlier'])} | "
                     f"{cell(p2['key_vs_earlier'])} |")
    lines += ["", "Holm-adjusted p (family: P1 h=2, P1 h=3, P2 key AUC, P2 direction, all readers):", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(record["holm"].items(), key=lambda kv: kv[1])]
    return "\n".join(lines) + "\n"


# --- write report (P3, P4) and the dev selection --------------------------------------------------------------------
GATE_ANSWER_SWAP = 0.30
REPAIR_THRESHOLD = 0.10


def by_setting(trials: list[dict]) -> dict:
    """{(variant, method, layer, alpha): {item_id: record}}."""
    import collections
    out = collections.defaultdict(dict)
    for t in trials:
        out[(t["variant"], t["method"], t["layer"], t["alpha"])][t["item_id"]] = t
    return out


def success_minus_null(cells: dict, method: str, layer: int, alpha: float, variant: str = "key_swap",
                       null_variant: str = "random_null") -> list[float]:
    """Per item: 1[edit succeeded] - 1[the random direction at the same layer and alpha hit the same target]."""
    edit, null = cells.get((variant, method, layer, alpha), {}), cells.get((null_variant, "random", layer, alpha), {})
    return [float(r["outcome"] == "success") - float(null[i]["outcome"] == "success") for i, r in edit.items()
            if i in null]


def rate(cells: dict, key: tuple, outcome: str = "success") -> list[float]:
    return [float(r["outcome"] == outcome) for r in cells.get(key, {}).values()]


def select_write_settings(tag: str) -> None:
    """From the dev grid (trials_<dev>_select): per method, the (layer, alpha) maximising mean key_swap success -
    random_null success (ties: smaller alpha, then earlier layer); the answer-swap gate (best answer_swap success
    over the method's grid >= 0.30); each method's best alpha at layer 42; repair eligibility (gate passed and dev
    key_swap - null >= 0.10). Writes write_lock.json (once) and a dev write report in results/<tag>/."""
    import exp15_edits as ED
    out_dir = EXP.results / tag
    out_dir.mkdir(parents=True, exist_ok=False)
    cells = by_setting(E.load_trials(DEV_BANK, "select"))
    # Pre-registered: ptag AR writes only if day-0 check 7 passes, or if the (at most 3 h) debugging fixes it
    ptag_passed = json.loads((check_dir() / "ptag.json").read_text())["pass"]
    debug_path = check_dir() / "ptag_debug_read_site.json"
    ptag_v2_passed = debug_path.exists() and json.loads(debug_path.read_text())["pass"]
    methods = [m for m in ED.METHODS if m != "ptag_ar" or ptag_passed] + (["ptag_ar_v2"] if ptag_v2_passed else [])
    method_layers = ED.METHOD_LAYERS | {"ptag_ar_v2": ED.METHOD_LAYERS["ptag_ar"]}
    grid, primary, layer42, gates = {}, {}, {}, {}
    for m in methods:
        settings = [(l, a) for l in method_layers[m] for a in ED.ALPHAS if ("key_swap", m, l, a) in cells]
        if not settings:
            continue
        grid[m] = {f"L{l}|a{a}": {"key_minus_null": round(sum(d) / len(d), 4) if (d := success_minus_null(
                       cells, m, l, a)) else None,
                   "key_success": round(sum(r) / len(r), 4) if (r := rate(cells, ("key_swap", m, l, a))) else None,
                   "answer_swap_success": round(sum(r) / len(r), 4) if (r := rate(cells, ("answer_swap", m, l, a)))
                   else None} for l, a in settings}

        def score(setting):
            l, a = setting
            d = success_minus_null(cells, m, l, a)
            return (sum(d) / len(d), -a, -l)
        best = max(settings, key=score)
        gate = max(sum(r) / len(r) for l, a in settings if (r := rate(cells, ("answer_swap", m, l, a))))
        gates[m] = {"best_answer_swap": round(gate, 4), "pass": gate >= GATE_ANSWER_SWAP}
        primary[m] = {"layer": best[0], "alpha": best[1], "dev_key_minus_null": round(score(best)[0], 4)}
        at42 = [(42, a) for a in ED.ALPHAS if ("key_swap", m, 42, a) in cells]
        if at42:
            b42 = max(at42, key=score)
            layer42[m] = {"alpha": b42[1], "dev_key_minus_null": round(score(b42)[0], 4)}
    repair_methods = [m for m in primary if gates[m]["pass"] and primary[m]["dev_key_minus_null"] >= REPAIR_THRESHOLD]
    lock = {"written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rule": "see manifest writes.selection_rule",
            "ptag_check_passed": ptag_passed, "ptag_debug_read_site_passed": ptag_v2_passed,
            "primary": primary, "layer42": layer42, "gates": gates, "repair_methods": repair_methods,
            "k4_repair_testable": bool(repair_methods), "dev_items": len({i for c in cells.values() for i in c})}
    path = E.write_lock_path()
    require_absent(path)
    atomic_write_text(json.dumps(lock, indent=1) + "\n", path)
    atomic_write_text(json.dumps({"lock": lock, "grid": grid}, indent=1) + "\n", out_dir / "dev_write_grid.json")
    print(json.dumps(lock, indent=1))


def write_report(bank: str, tag: str) -> None:
    """P3 at the locked settings (key_swap - random_null with paired bootstrap CI, other_item, answer_swap, outcome
    shares, the layer-42 head-to-head) and P4 (repair, redirect, null) from trials_<bank>_{locked,repair}."""
    import collections
    import exp15_score as S
    out_dir = EXP.results / tag
    out_dir.mkdir(parents=True, exist_ok=False)
    lock = json.loads(E.write_lock_path().read_text())
    record = {"bank": bank, "lock": lock, "P3": {}, "P3_layer42": {}, "P4": {}}
    phase = "select" if bank == DEV_BANK else "locked"
    cells = by_setting(E.load_trials(bank, phase))

    def summary(m, l, a):
        out = {"layer": l, "alpha": a,
               "key_minus_null": S.summarise([{"x": v} for v in success_minus_null(cells, m, l, a)], "x", 0.0)}
        for variant in ("key_swap", "answer_swap", "other_item"):
            recs = cells.get((variant, m, l, a), {})
            if recs:
                shares = collections.Counter(r["outcome"] for r in recs.values())
                out[variant] = {k: round(v / len(recs), 4) for k, v in sorted(shares.items())} | {"n": len(recs)}
        nulls = cells.get(("random_null", "random", l, a), {})
        if nulls:
            shares = collections.Counter(r["outcome"] for r in nulls.values())
            out["random_null"] = {k: round(v / len(nulls), 4) for k, v in sorted(shares.items())} | {"n": len(nulls)}
        return out
    pvals = {}
    for m, s in lock["primary"].items():
        record["P3"][m] = summary(m, s["layer"], s["alpha"]) | {"gate": lock["gates"][m]}
        pvals[m] = record["P3"][m]["key_minus_null"]["p"]
    record["P3_holm"] = S.holm({k: v for k, v in pvals.items() if v == v})
    for m, s in lock["layer42"].items():
        record["P3_layer42"][m] = summary(m, 42, s["alpha"])
    if E.trials_path(bank, "repair").exists():
        rc = by_setting(E.load_trials(bank, "repair"))
        for m in lock["repair_methods"]:
            l, a = lock["primary"][m]["layer"], lock["primary"][m]["alpha"]
            entry = {"layer": l, "alpha": a}
            for variant in ("repair", "redirect"):
                d = success_minus_null(rc, m, l, a, variant, "repair_null") if variant == "repair" else None
                entry[variant] = S.summarise([{"x": v} for v in rate(rc, (variant, m, l, a))], "x", None)
                if d is not None:
                    entry["repair_minus_null"] = S.summarise([{"x": v} for v in d], "x", 0.0)
            entry["null_gold_rate"] = S.summarise([{"x": v} for v in rate(rc, ("repair_null", "random", l, a))], "x",
                                                  None)
            record["P4"][m] = entry
    atomic_write_text(json.dumps(record, indent=1) + "\n", out_dir / "write_report.json")
    atomic_write_text(write_report_md(record), out_dir / "write_report.md")
    print(write_report_md(record))


def write_report_md(record: dict) -> str:
    def ci(x):
        return f"{x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] n={x['n']}"
    lines = [f"# exp15 write report: {record['bank']} (UNVERIFIED)", "",
             "| method | layer | alpha | gate (best answer-swap) | key swap - null | key swap outcomes | "
             "other-item success | answer-swap success |", "|---|---|---|---|---|---|---|---|"]
    for m, e in record["P3"].items():
        ks = e.get("key_swap", {})
        lines.append(f"| {m} | {e['layer']} | {e['alpha']} | {e['gate']['best_answer_swap']:.2f} "
                     f"({'pass' if e['gate']['pass'] else 'fail'}) | {ci(e['key_minus_null'])} | "
                     + ", ".join(f"{k} {v:.2f}" for k, v in ks.items() if k != "n")
                     + f" | {e.get('other_item', {}).get('success', 0):.2f} | "
                       f"{e.get('answer_swap', {}).get('success', 0):.2f} |")
    lines += ["", "Head-to-head at layer 42 (each method at its best dev alpha there):", "",
              "| method | alpha | key swap - null | key swap success |", "|---|---|---|---|"]
    for m, e in record["P3_layer42"].items():
        lines.append(f"| {m} | {e['alpha']} | {ci(e['key_minus_null'])} | "
                     f"{e.get('key_swap', {}).get('success', 0):.2f} |")
    if record["P4"]:
        lines += ["", "P4 repair (consistent off-cycle h=3 errors):", "",
                  "| method | repair (gold) | repair - null | redirect (T(r)) | null gold rate |", "|---|---|---|---|---|"]
        for m, e in record["P4"].items():
            lines.append(f"| {m} | {ci(e['repair'])} | {ci(e['repair_minus_null'])} | {ci(e['redirect'])} | "
                         f"{ci(e['null_gold_rate'])} |")
    else:
        lines += ["", f"P4: no method eligible (K4: repair untestable). Repair methods: {record['lock']['repair_methods']}"]
    return "\n".join(lines) + "\n"


# --- day-0 check 8: matcher audit -----------------------------------------------------------------------------------
# 'matcher': 50 random dev readouts. 'matcher_cjk' (added on dev before the lock, after AO turned out to answer in
# Chinese in about a quarter of its samples): 30 AO readouts containing a Chinese colour character.
AUDITS = {"matcher": {"seed": "exp15_matcher_audit", "per_reader": {"nla": 20, "olens": 15, "ao": 15},
                      "only_cjk": False},
          "matcher_cjk": {"seed": "exp15_matcher_audit_cjk", "per_reader": {"ao": 30}, "only_cjk": True}}
CJK_COLOUR_CHARS = "红蓝绿金粉灰棕褐黑白紫"


def audit_readouts_path(name: str = "matcher"):
    return check_dir() / f"{name}_audit_readouts.json"


def audit_labels_path(name: str = "matcher"):
    return check_dir() / f"{name}_audit_labels.json"


def write_audit_sample(name: str = "matcher") -> None:
    """Dev readouts per AUDITS[name], written without the matcher's output so they can be labelled blind."""
    import random
    import exp15_readers as RD
    spec = AUDITS[name]
    rng = random.Random(spec["seed"])
    rows = []
    for reader, n in spec["per_reader"].items():
        reads = load_text_reads(DEV_BANK, reader)
        pool = sorted((item_id, key, k) for item_id, r in reads.items() for key, samples in r.items()
                      for k in range(len(samples))
                      if not spec["only_cjk"] or any(ch in samples[k] for ch in CJK_COLOUR_CHARS))
        for item_id, key, k in rng.sample(pool, n):
            rows.append({"audit_id": len(rows), "reader": reader, "item_id": item_id, "read": key, "sample": k,
                         "text": RD.explanation_text(reader, reads[item_id][key][k])})
    path = audit_readouts_path(name)
    require_absent(path)
    atomic_write_text(json.dumps({"spec": spec, "rows": rows}, indent=1) + "\n", path)
    print(f"-> {path}")


def matcher_agreement(name: str) -> dict:
    """Precision and recall of exp15_score.colours_named over (readout, colour) pairs against the hand labels."""
    import exp15_score as S
    rows = json.loads(audit_readouts_path(name).read_text())["rows"]
    labels = {int(k): set(v) for k, v in json.loads(audit_labels_path(name).read_text())["labels"].items()}
    assert set(labels) == {r["audit_id"] for r in rows}
    tp = fp = fn = 0
    disagreements = []
    for r in rows:
        found, truth = S.colours_named(r["text"]), labels[r["audit_id"]]
        tp, fp, fn = tp + len(found & truth), fp + len(found - truth), fn + len(truth - found)
        if found != truth:
            disagreements.append({"audit_id": r["audit_id"], "matcher_only": sorted(found - truth),
                                  "label_only": sorted(truth - found)})
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"n_readouts": len(rows), "n_true_mentions": tp + fn, "precision": round(precision, 4),
            "recall": round(recall, 4), "pass": precision >= 0.95 and recall >= 0.95,
            "disagreements": disagreements}


def check_matcher(name: str = "matcher") -> None:
    """Day-0 check 8 for one audit (pass: precision and recall >= 0.95). The CJK audit also re-scores the first
    audit's 50 readouts with the current matcher."""
    record = matcher_agreement(name)
    if name == "matcher_cjk":
        record["first_audit_with_current_matcher"] = matcher_agreement("matcher")
    E.write_check(name, record)


# --- day-0 check 9: WorkspaceBench external check -------------------------------------------------------------------
WSBENCH_ITEMS = "/tmp/wsbench_probe/evals/brew_intermediates/items.json"      # WorkspaceBench at commit 92d763e
WSBENCH_CELLS = {"emit_asst": -7, "emit_think": -3, "emit_colon": -1}          # from the end of the prefilled render


def colour_regex(colour: str):
    import re
    import exp15_score as S
    return re.compile(S.COLOUR_PATTERNS.get(colour, rf"\b{colour}\b"), re.IGNORECASE)


@torch.no_grad()
def check_workspacebench() -> None:
    """Day-0 check 9: our AO reimplementation on WorkspaceBench's 100 brew_intermediates items (depth 2: the gold is
    the colour after the first stir). Emission cells (newline after 'assistant', the \\n\\n after </think>, the
    prefilled colon) x layers 20..60 step 4, 4 samples each, scored by regex (their published 38% gold vs 20% off
    used an LLM judge and k = 1). Cell rate = share of (cell, layer) whose first sample names the colour; the item
    rule is theirs: pass iff K * G > sum_j O_j over the K off-trajectory candidates. Pass: gold cell rate > off cell
    rate, paired bootstrap CI above 0."""
    import exp15_readers as RD
    import exp15_score as S
    items = json.loads(open(WSBENCH_ITEMS).read())["items"]
    tokenizer, model = E.load_base()
    layers = RD.READERS["ao"]["layers"]
    hidden = {L: [] for L in layers}
    for it in items:
        text = tokenizer.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) + "Answer:"
        ids = tokenizer.encode(text, add_special_tokens=False)
        tail = [tokenizer.decode(ids[k]) for k in WSBENCH_CELLS.values()]
        assert tail == ["\n", "\n\n", ":"], tail
        hs = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states
        for L in layers:
            hidden[L].append(torch.stack([hs[L + 1][0, len(ids) + k] for k in WSBENCH_CELLS.values()]).float().cpu())
    del model, hs
    E.release()
    reader = RD.TextReader("ao")
    rows = []
    per_item = [{"gold": 0, "off": None, "cells": 0, "gold_any4": 0, "off_any4": None} for _ in items]
    for L in layers:
        vectors = torch.stack(hidden[L]).reshape(-1, hidden[L][0].shape[-1])            # [items * 3, d]
        samples = reader.read(vectors, L, seed=L)
        for n, it in enumerate(items):
            gold = it["intermediates"][0]
            options = it["options_adjacent"][0]
            off = [c for c in options if c not in {gold, it["start"], it["answer"]}]
            rec = per_item[n]
            rec["off"] = rec["off"] or [0] * len(off)
            rec["off_any4"] = rec["off_any4"] or [0] * len(off)
            for k in range(len(WSBENCH_CELLS)):
                s = samples[n * len(WSBENCH_CELLS) + k]
                rec["cells"] += 1
                rec["gold"] += bool(colour_regex(gold).search(s[0]))
                rec["gold_any4"] += any(colour_regex(gold).search(x) for x in s)
                for j, c in enumerate(off):
                    rec["off"][j] += bool(colour_regex(c).search(s[0]))
                    rec["off_any4"][j] += any(colour_regex(c).search(x) for x in s)
            if L == layers[len(layers) // 2] and n < 3:
                rows.append({"item": it["name"], "layer": L, "sample": samples[n * len(WSBENCH_CELLS)][0][:300]})
    del reader
    E.release()
    gold_rate = [r["gold"] / r["cells"] for r in per_item]
    off_rate = [sum(r["off"]) / len(r["off"]) / r["cells"] for r in per_item]
    diff = [g - o for g, o in zip(gold_rate, off_rate)]
    passes = [len(r["off"]) * r["gold"] > sum(r["off"]) for r in per_item]
    null = [sum(len(r["off"]) * r["off"][j] > (sum(r["off"]) - r["off"][j] + r["gold"])
                for j in range(len(r["off"]))) / len(r["off"]) for r in per_item]
    ci = S.boot_ci(diff)
    record = {"n_items": len(items), "layers": list(layers), "cells": list(WSBENCH_CELLS),
              "gold_rate": round(sum(gold_rate) / len(items), 4), "offpath_rate": round(sum(off_rate) / len(items), 4),
              "gold_minus_off": [round(x, 4) for x in ci[:3]],
              "gold_rate_any_of_4": round(sum(r["gold_any4"] / r["cells"] for r in per_item) / len(items), 4),
              "offpath_rate_any_of_4": round(sum(sum(r["off_any4"]) / len(r["off_any4"]) / r["cells"]
                                                 for r in per_item) / len(items), 4),
              "item_pass_rate": round(sum(passes) / len(items), 4), "item_null_rate": round(sum(null) / len(items), 4),
              "published_llm_judged": {"gold_cell_rate": 0.38, "off_cell_rate": 0.20},
              "pass": ci[1] > 0, "examples": rows}
    E.write_check("workspacebench", record)


# --- pipeline check: the edit harness with a zero edit reproduces the captured answers ------------------------------
ZERO_EDIT_SPEC = {"seed": "exp15_zero_edit", "n_items": 128, "layers": (28, 42)}


def check_zero_edit() -> None:
    """E.run_trials (right-padded batches of 32, the AdditiveEdit hook installed, alpha = 0) on seeded dev items must
    give the same argmax colour as the capture's single-item forward, and nearly the same p(answer)."""
    import random

    import exp15_edits as ED
    table = item_table(DEV_BANK)
    metas = {m["item_id"]: m for m in load_capture(DEV_BANK, with_hidden=False)[0]}
    chosen = random.Random(ZERO_EDIT_SPEC["seed"]).sample(table, ZERO_EDIT_SPEC["n_items"])
    trials = [ED.Trial(it.item_id, "zero_edit", "random", layer, 0.0, None, None, it.pred, {"success": it.pred},
                       seed=0) for it in chosen for layer in ZERO_EDIT_SPEC["layers"]]
    E.run_trials(DEV_BANK, "zero_edit", trials)
    records = E.load_trials(DEV_BANK, "zero_edit")
    same = [r["answer"] == r["success"] for r in records]
    dp = [abs(r["p_answer"] - float(torch.softmax(torch.tensor(metas[r["item_id"]]["colour_logits"]), -1).max()))
          for r in records if r["answer"] == r["success"]]
    record = {**ZERO_EDIT_SPEC, "n_trials": len(records), "same_answer": round(sum(same) / len(same), 4),
              "max_abs_dp": round(max(dp), 4), "mean_abs_dp": round(sum(dp) / len(dp), 5),
              "pass": sum(same) / len(same) >= 0.99}
    E.write_check("zero_edit", record)
    print(record)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["manifest", "read_report", "audit_sample", "check_matcher", "check_workspacebench",
                                   "select_writes", "write_report", "check_zero_edit"])
    p.add_argument("name", nargs="?", help="manifest status (draft | locked) or audit name (matcher | matcher_cjk)")
    p.add_argument("--bank", choices=E.BANKS, default=DEV_BANK)
    p.add_argument("--tag", help="analysis folder name under results/exp15_wrong_keys (must be new)")
    a = p.parse_args()
    if a.cmd == "manifest":
        write_manifest(a.name)
    elif a.cmd == "read_report":
        read_report(a.bank, a.tag)
    elif a.cmd == "audit_sample":
        write_audit_sample(a.name or "matcher")
    elif a.cmd == "check_matcher":
        check_matcher(a.name or "matcher")
    elif a.cmd == "check_workspacebench":
        check_workspacebench()
    elif a.cmd == "select_writes":
        select_write_settings(a.tag)
    elif a.cmd == "write_report":
        write_report(a.bank, a.tag)
    elif a.cmd == "check_zero_edit":
        check_zero_edit()


if __name__ == "__main__":
    main()
