"""exp14: causal swaps of the brew intermediate in Gemma-4-31B-it (results/exp14_brew_swaps/manifest.json).

On a three-ingredient brew item, s_0 is the start colour and s_j = table[s_{j-1}][stir j]. A clamp swap exchanges
the residual's coordinates on two colour directions (jpp_lens's clamp swap, written out here so that only the
positions after the table are edited) at four layers of a window. If swapping s1 for s1' makes the model answer
table[s1'][stir 2], the model computed its second lookup from that direction.

  capture   one clean forward per exp11 test item (three-ingredient, h=1..3): the 10 colour logits and the block
            outputs at each stir token and the answer position (fp16; cache/exp14/capture.pt, .jsonl)
  readouts  Gate 0 (descriptive): per lens, layer and captured position, how well each colour state is read
            (rank of the state among the 10 colours against the colours off the item's path), and the exp11
            colour probe's held-out accuracy (cache/exp14/readouts.pt)
  swaps     Gate 1 and Stage 2: on the first N_SWAP_ITEMS h=2 items the model answers correctly, swap s1 -> s1'
            (main), the answer s2 -> a' = table[s1'][stir 2] (answer_swap, the ceiling) and s1 -> an unrelated
            direction (random_null), per direction, window and scale (cache/exp14/swaps_h2.pt)
  repair    Stage 3: on h=3 items answered table[s1][stir 3] (stir 2 skipped), swap s1 -> s2 (repair) and
            s1 -> c (redirect), per direction and window, at each direction's scale chosen on the swaps stage
            (cache/exp14/swaps_h3.pt)
  report    the manifest's numbers and reading rules (results/exp14_brew_swaps/<run>/report.txt)

Layer l is the output of decoder block l (hidden_states[l + 1]), the lenses' numbering.
Run with /venv/main/bin/python from scripts/; HF_HOME must point at the model cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

import cc_config as cfg
import cc_exp10_render as R
import exp11 as X
import exp11_brew as B
import exp11_probe as P

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "third_party" / "jpp_lens" / "src")]
from workspace_lens.interventions.interventions import NEAR_PARALLEL_ABS_COSINE  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

EXP = cfg.Exp("exp14_brew_swaps", "exp14")
GEMMA = "google/gemma-4-31B-it"
FAMILY = "gemma4"
N_LAYERS = 60
LENS_FILES = {"jpp": REPO / "cache/exp12/lenses/jpp_64p.pt", "rlens": REPO / "cache/exp12/lenses/rlens_64p.pt"}
DIRECTIONS = ("jpp", "rlens", "logit", "diffmeans")
LENS_LAYERS = tuple(range(0, 59, 2))                     # the even layers both 64-prompt lenses are fitted at
WINDOWS = tuple(tuple(range(s, s + 8, 2)) for s in range(0, 53, 4))   # 14 windows of 4 even layers
SCALES = (1.0, 2.0, 4.0)
NULL_SCALE = 4.0
N_SWAP_ITEMS = 100
NULL_TOKEN = " table"
PROBE_K_PCA, PROBE_WD = 64, 1e-3                          # exp11's test-addendum probe settings
N_BOOT = 2000


def capture_paths() -> tuple[Path, Path]:
    return EXP.cache / "capture.pt", EXP.cache / "capture.jsonl"


def require_absent(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists; written once")
    path.parent.mkdir(parents=True, exist_ok=True)


# --- model ----------------------------------------------------------------------------------------------------------
def load_gemma():
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(GEMMA)
    model = AutoModelForImageTextToText.from_pretrained(GEMMA, dtype=torch.bfloat16, device_map="cuda").eval()
    return tokenizer, model


def text_parts(model):
    """(decoder layers, final norm, unembedding [V, d])."""
    lm = model.model.language_model
    return lm.layers, lm.norm, model.lm_head.weight


def colour_ids(tokenizer, colours) -> list[int]:
    ids = []
    for colour in colours:
        toks = tokenizer.encode(" " + colour, add_special_tokens=False)
        assert len(toks) == 1, f"{colour!r} is {len(toks)} tokens"
        ids.append(toks[0])
    return ids


def prompt_and_positions(tokenizer, row: dict, brew: B.Brew) -> tuple[list[int], dict[str, int]]:
    """The prompt ids, and the token index of each stir, the start sentence's first token (the first position
    after the table) and the answer position."""
    text = R.prompt_text(tokenizer, FAMILY, row)
    ids = R.prompt_ids(tokenizer, FAMILY, row)
    if tokenizer.encode(text, add_special_tokens=False) != ids[:len(tokenizer.encode(text, add_special_tokens=False))]:
        raise RuntimeError(f"{row['item_id']}: the rendered text does not tokenize to the prompt's prefix")
    named = {k: v for k, v in X.token_positions(tokenizer, text, brew).items() if k.startswith("stir")}
    offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    sentence = text.index("The potion starts out ", text.rindex("Problem: "))
    named["post_table"] = next(i for i, (a, b) in enumerate(offsets) if b > sentence)
    named["final"] = len(ids) - 1
    return ids, named


@torch.no_grad()
def forward(model, ids: list[int]):
    out = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False)
    return out.logits[0, -1].float(), out.hidden_states


def load_items() -> tuple[dict, list[dict]]:
    items = {r["item_id"]: r for r in X.load_bank("test")}
    rows = [json.loads(line) for line in X.requests_path("test").open()]
    return items, rows


# --- capture --------------------------------------------------------------------------------------------------------
def capture() -> None:
    pt, meta = capture_paths()
    require_absent(pt)
    tokenizer, model = load_gemma()
    items, rows = load_items()
    colours = sorted(B.parse(items[rows[0]["item_id"]]["problem"]).table)
    cids = colour_ids(tokenizer, colours)
    hidden, metas = [], []
    for n, row in enumerate(rows):
        brew = B.parse(items[row["item_id"]]["problem"])
        ids, named = prompt_and_positions(tokenizer, row, brew)
        logits, hs = forward(model, ids)
        stored = [k for k in named if k.startswith("stir")] + ["final"]
        hidden.append(torch.stack([hs[l + 1][0, [named[k] for k in stored]] for l in range(N_LAYERS)])
                      .to(torch.float16).cpu())                 # [layers, positions, d]
        colour_logits = logits[cids].cpu()
        metas.append({"item_id": row["item_id"], "h": row["h"], "n_tokens": len(ids), "positions": named,
                      "stored_positions": stored, "colours": colours,
                      "colour_logits": [round(float(x), 4) for x in colour_logits],
                      "predicted": colours[int(colour_logits.argmax())], "answer": items[row["item_id"]]["answer"],
                      "colour_mass": round(float(torch.softmax(logits, -1)[cids].sum()), 6)})
        if n % 100 == 0:
            print(f"capture {n}/{len(rows)}", flush=True)
    torch.save({"item_ids": [m["item_id"] for m in metas], "hidden": hidden}, pt)
    meta.write_text("".join(json.dumps(m) + "\n" for m in metas))
    print(f"{len(metas)} items -> {pt}")


def load_capture() -> list[dict]:
    pt, meta = capture_paths()
    items = {r["item_id"]: r for r in X.load_bank("test")}
    blob = torch.load(pt, weights_only=False)
    metas = [json.loads(line) for line in meta.open()]
    for m, h in zip(metas, blob["hidden"], strict=True):
        m["hidden"] = h
        m["brew"] = B.parse(items[m["item_id"]]["problem"])
    return metas


# --- colour roles ---------------------------------------------------------------------------------------------------
def off_path(brew: B.Brew) -> list[str]:
    """Colours not on the item's path s_0..s_h."""
    path = set(brew.path())
    return [c for c in sorted(brew.table) if c not in path]


def state_decoy(brew: B.Brew, j: int) -> str | None:
    """A row-mate decoy for state s_j (j >= 1): the entry of s_{j-1}'s rule line under an ingredient other than
    stir j's, chosen as the first such entry (in ingredient order) that is not on the path."""
    prev, used = brew.path()[j - 1], brew.stirs[j - 1]
    for ingredient in brew.ingredients:
        value = brew.table[prev][ingredient]
        if ingredient != used and value not in brew.path():
            return value
    return None


# --- readouts (Gate 0) ----------------------------------------------------------------------------------------------
@torch.no_grad()
def readouts() -> None:
    out = EXP.cache / "readouts.pt"
    require_absent(out)
    metas = load_capture()
    tokenizer, model = load_gemma()
    _, final_norm, w_u = text_parts(model)
    colours = metas[0]["colours"]
    w_colours = w_u[colour_ids(tokenizer, colours)].float()                       # [10, d]

    def colour_scores(x: torch.Tensor) -> torch.Tensor:
        """x [..., d] residuals in the final block's basis -> the model's colour logits [..., 10] (the final
        norm, then the unembedding; the monotone logit soft-cap is omitted)."""
        return final_norm(x.to(torch.bfloat16)).float() @ w_colours.T

    lenses = {}
    for name, path in LENS_FILES.items():
        jac = load_lens_file(str(path), hf_model_name=GEMMA).jacobians_L_dict_FN
        lenses[name] = {l: jac[l].cuda() for l in LENS_LAYERS}
    result = {"colours": colours, "lens_layers": LENS_LAYERS, "lens_scores": {}, "probe": {}}
    for m in metas:
        x = m["hidden"][list(LENS_LAYERS)].float().cuda()                       # [L, P, d]
        scores = {"logit": colour_scores(x)}
        for name, jac in lenses.items():
            scores[name] = torch.stack([colour_scores(x[k] @ jac[l].T) for k, l in enumerate(LENS_LAYERS)])
        result["lens_scores"][m["item_id"]] = {k: v.cpu() for k, v in scores.items()}   # [L, P, 10]
    # probes: per h and stored position, labels = each state s_1..s_h and its row-mate decoy (exp11 settings)
    for h in (1, 2, 3):
        ms = [m for m in metas if m["h"] == h]
        for pos_index, pos in enumerate(ms[0]["stored_positions"]):
            x = torch.stack([m["hidden"][:, pos_index] for m in ms])           # [n, 60, d]
            folds = P.stratified_folds(["all"] * len(ms))
            for j in range(1, h + 1):
                for label in ("state", "decoy"):
                    ys = [m["brew"].path()[j] if label == "state" else state_decoy(m["brew"], j) for m in ms]
                    keep = [i for i, y in enumerate(ys) if y is not None]
                    correct = P.cv_correct(x[keep], [colours.index(ys[i]) for i in keep], [folds[i] for i in keep],
                                           k_pca=PROBE_K_PCA, weight_decay=PROBE_WD)
                    result["probe"][(h, pos, j, label)] = {"item_ids": [ms[i]["item_id"] for i in keep],
                                                           "correct": correct}
            print(f"probe h={h} {pos}", flush=True)
    torch.save(result, out)
    print(f"-> {out}")


# --- swap machinery -------------------------------------------------------------------------------------------------
class ClampSwap:
    """At each layer of a window, at the edited positions: h <- h + V (c* - pinv(V) h), with V = [u_s, u_t] the unit
    source and target directions and c* the clean pass's coordinates on V with source and target exchanged, times
    the scale. A layer whose two directions are near-parallel is skipped."""

    def __init__(self, layers_mod, positions: list[int]):
        self.layers_mod = layers_mod
        self.positions = torch.tensor(positions, device="cuda")
        self.edits: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.handles = []

    def set(self, clean: dict[int, torch.Tensor], dirs: dict[int, tuple[torch.Tensor, torch.Tensor]], scale: float):
        """clean[l]: clean residuals at the edited positions [P, d] (fp32); dirs[l]: (u_source, u_target)."""
        self.edits = {}
        for l, (u_s, u_t) in dirs.items():
            if float((u_s @ u_t).abs()) > NEAR_PARALLEL_ABS_COSINE:
                continue
            basis = torch.stack([u_s, u_t], 1)                                   # [d, 2]
            pinv = torch.linalg.pinv(basis)                                      # [2, d]
            coords = clean[l] @ pinv.T                                           # [P, 2]
            target = scale * coords.flip(1)
            self.edits[l] = (basis, pinv, target)

    def __enter__(self):
        for l in self.edits:
            self.handles.append(self.layers_mod[l].register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _hook(self, layer: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            basis, pinv, target = self.edits[layer]
            h = hidden[0, self.positions].float()
            h = h + (target - h @ pinv.T) @ basis.T
            hidden = hidden.clone()
            hidden[0, self.positions] = h.to(hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        return hook


class Directions:
    """Unit swap directions per (kind, layer, colour): lens rows W_U[c] @ J_l (jpp, rlens), W_U[c] (logit), and
    difference-of-means directions fit on the other half of the capture (diffmeans)."""

    def __init__(self, w_u: torch.Tensor, tokenizer, colours: list[str], metas: list[dict]):
        self.colours = colours
        cids = colour_ids(tokenizer, colours)
        null_id = tokenizer.encode(NULL_TOKEN, add_special_tokens=False)
        assert len(null_id) == 1
        rows = w_u[cids + null_id].float()                                        # [11, d]
        self.rows = {c: rows[i] for i, c in enumerate(colours)} | {"<null>": rows[-1]}
        self.jac = {name: load_lens_file(str(path), hf_model_name=GEMMA).jacobians_L_dict_FN
                    for name, path in LENS_FILES.items()}
        for jac in self.jac.values():
            for l in LENS_LAYERS:
                jac[l] = jac[l].cuda()
        self.means = self._diffmeans(metas)

    @staticmethod
    def fold_of(item_id: str) -> int:
        return int(item_id.rsplit(":", 1)[1]) % 2

    def _diffmeans(self, metas: list[dict]) -> dict:
        """means[(fold, role, colour)] [L_lens, d]: mean residual of the items in `fold` whose role colour is c,
        minus the mean of those whose role colour is not c. role 's1': the stir-1 token, label s1 (h = 1..3);
        role 'answer': the answer position, label the model's clean answer."""
        out = {}
        for fold in (0, 1):
            for role in ("s1", "answer"):
                xs, ys = [], []
                for m in metas:
                    if self.fold_of(m["item_id"]) != fold:
                        continue
                    pos = 0 if role == "s1" else len(m["stored_positions"]) - 1
                    xs.append(m["hidden"][list(LENS_LAYERS), pos].float())
                    ys.append(m["brew"].path()[1] if role == "s1" else m["predicted"])
                x = torch.stack(xs)                                              # [n, L, d]
                for c in self.colours:
                    mask = torch.tensor([y == c for y in ys])
                    out[(fold, role, c)] = (x[mask].mean(0) - x[~mask].mean(0)).cuda()
        return out

    def unit(self, kind: str, layer: int, colour: str, *, fold: int, role: str, seed: str) -> torch.Tensor:
        """The unit direction of `colour` ('<null>' = the unrelated direction) at `layer`. For diffmeans, `fold`
        is the half the edited item is NOT in and `role` names the means; its null is a seeded random direction."""
        if kind == "diffmeans":
            if colour == "<null>":
                g = torch.Generator().manual_seed(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))
                v = torch.randn(self.rows["red"].shape[0], generator=g).cuda()
            else:
                v = self.means[(fold, role, colour)][LENS_LAYERS.index(layer)]
        elif kind == "logit":
            v = self.rows[colour]
        else:
            v = self.rows[colour] @ self.jac[kind][layer]
        return v / v.norm()


def swap_target_h2(brew: B.Brew, item_id: str) -> str | None:
    """s1' for a main h=2 swap: a colour off the path whose stir-2 entry a' is off the path and differs from its
    two other entries, which are also neither a' nor the clean answer. Seeded per item."""
    s0, s1, s2 = brew.path()
    ing2 = brew.stirs[1]
    eligible = []
    for c in off_path(brew):
        a = brew.table[c][ing2]
        others = [brew.table[c][x] for x in brew.ingredients if x != ing2]
        if a in (s0, s1, s2, c) or len(set(others)) < 2 or set(others) & {a, s2}:
            continue
        eligible.append(c)
    return random.Random(f"exp14 {item_id}").choice(eligible) if eligible else None


def redirect_target_h3(brew: B.Brew, item_id: str, predicted: str) -> str | None:
    """c for a Stage 3 redirect: a colour off the path whose stir-3 entry is neither gold, the clean answer, nor
    on the path. Seeded per item."""
    ing3 = brew.stirs[2]
    eligible = [c for c in off_path(brew)
                if brew.table[c][ing3] not in (*brew.path(), predicted, c)]
    return random.Random(f"exp14 redirect {item_id}").choice(eligible) if eligible else None


def run_trials(model, tokenizer, dirs: Directions, jobs: list[dict], out: Path, scales_by_kind: dict) -> None:
    """jobs: one per item, {item_id, row, brew, variants: {name: (source, target)}}. For each item: a clean
    forward, then one edited forward per (direction, window, scale, variant)."""
    layers_mod, _, _ = text_parts(model)
    cids = colour_ids(tokenizer, dirs.colours)
    results = []
    for n, job in enumerate(jobs):
        ids, named = prompt_and_positions(tokenizer, job["row"], job["brew"])
        positions = list(range(named["post_table"], named["final"] + 1))
        clean_logits, hs = forward(model, ids)
        clean = {l: hs[l + 1][0, positions].float() for l in LENS_LAYERS}
        swap = ClampSwap(layers_mod, positions)
        fold = 1 - Directions.fold_of(job["item_id"])
        record = {"item_id": job["item_id"], "clean": clean_logits[cids].cpu(), "trials": []}
        for kind in DIRECTIONS:
            for variant, (source, target, role) in job["variants"].items():
                scales = (NULL_SCALE,) if variant.endswith("null") else scales_by_kind[kind]
                for w, window in enumerate(WINDOWS):
                    pair = {l: (dirs.unit(kind, l, source, fold=fold, role=role, seed=job["item_id"]),
                                dirs.unit(kind, l, target, fold=fold, role=role, seed=f"{job['item_id']}:{l}"))
                            for l in window}
                    for scale in scales:
                        swap.set(clean, pair, scale)
                        with swap:
                            logits = model(input_ids=torch.tensor([ids], device="cuda"),
                                           use_cache=False).logits[0, -1].float()
                        record["trials"].append({"kind": kind, "variant": variant, "window": w, "scale": scale,
                                                 "n_layers_edited": len(swap.edits),
                                                 "logits": logits[cids].cpu()})
        results.append(record)
        if n % 5 == 0:
            print(f"{out.name} {n}/{len(jobs)}", flush=True)
    torch.save({"colours": dirs.colours, "windows": WINDOWS, "jobs": [{k: v for k, v in j.items() if k != "row"
                                                                         and k != "brew"} for j in jobs],
                "results": results}, out)
    print(f"-> {out}")


def swaps() -> None:
    out = EXP.cache / "swaps_h2.pt"
    require_absent(out)
    metas = load_capture()
    tokenizer, model = load_gemma()
    _, _, w_u = text_parts(model)
    dirs = Directions(w_u, tokenizer, metas[0]["colours"], metas)
    _, rows = load_items()
    rows = {r["item_id"]: r for r in rows}
    jobs = []
    for m in metas:
        if m["h"] != 2 or m["predicted"] != m["answer"]:
            continue
        brew = m["brew"]
        s1p = swap_target_h2(brew, m["item_id"])
        if s1p is None:
            continue
        s0, s1, s2 = brew.path()
        a_prime = brew.table[s1p][brew.stirs[1]]
        jobs.append({"item_id": m["item_id"], "row": rows[m["item_id"]], "brew": brew, "s1": s1, "s1_prime": s1p,
                     "s2": s2, "a_prime": a_prime,
                     "off_column": [brew.table[s1p][x] for x in brew.ingredients if x != brew.stirs[1]],
                     "variants": {"main": (s1, s1p, "s1"), "answer_swap": (s2, a_prime, "answer"),
                                  "random_null": (s1, "<null>", "s1")}})
        if len(jobs) == N_SWAP_ITEMS:
            break
    run_trials(model, tokenizer, dirs, jobs, out, {k: SCALES for k in DIRECTIONS})


def repair() -> None:
    out = EXP.cache / "swaps_h3.pt"
    require_absent(out)
    h2 = summarise_h2(torch.load(EXP.cache / "swaps_h2.pt", weights_only=False))
    scales_by_kind = {k: (h2["chosen_scale"][k],) for k in DIRECTIONS}
    metas = load_capture()
    tokenizer, model = load_gemma()
    _, _, w_u = text_parts(model)
    dirs = Directions(w_u, tokenizer, metas[0]["colours"], metas)
    _, rows = load_items()
    rows = {r["item_id"]: r for r in rows}
    jobs = []
    for m in metas:
        brew = m["brew"]
        if m["h"] != 3 or m["predicted"] == m["answer"]:
            continue
        s0, s1, s2, s3 = brew.path()
        if m["predicted"] != brew.table[s1][brew.stirs[2]]:              # stir 2 skipped
            continue
        c = redirect_target_h3(brew, m["item_id"], m["predicted"])
        if c is None:
            continue
        jobs.append({"item_id": m["item_id"], "row": rows[m["item_id"]], "brew": brew, "s1": s1, "s2": s2,
                     "gold": s3, "predicted": m["predicted"], "c": c, "c_answer": brew.table[c][brew.stirs[2]],
                     "variants": {"repair": (s1, s2, "s1"), "redirect": (s1, c, "s1")}})
    run_trials(model, tokenizer, dirs, jobs, out, scales_by_kind)


# --- analysis -------------------------------------------------------------------------------------------------------
def boot_ci(x: torch.Tensor, g: torch.Generator) -> tuple[float, float, float]:
    boot = x[torch.randint(len(x), (N_BOOT, len(x)), generator=g)].mean(1)
    return float(x.mean()), float(boot.quantile(0.025)), float(boot.quantile(0.975))


def fmt(ci) -> str:
    return f"{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]"


def h2_outcomes(blob: dict) -> dict:
    """Per (kind, variant, window, scale): per-item tensors success (argmax = a'), off-column rate (argmax is one
    of s1''s two other entries, halved: the per-column rate), changed (argmax != clean argmax), and the
    column-specific flip = success - off-column rate."""
    colours = blob["colours"]
    jobs = {j["item_id"]: j for j in blob["jobs"]}
    cells = defaultdict(lambda: defaultdict(list))
    for rec in blob["results"]:
        job = jobs[rec["item_id"]]
        clean = int(rec["clean"].argmax())
        for t in rec["trials"]:
            pred = colours[int(t["logits"].argmax())]
            key = (t["kind"], t["variant"], t["window"], t["scale"])
            success = float(pred == job["a_prime"])
            off = 0.5 * float(pred in job["off_column"])
            cells[key]["success"].append(success)
            cells[key]["off_column"].append(off)
            cells[key]["flip"].append(success - off)
            cells[key]["changed"].append(float(colours.index(pred) != clean))
            cells[key]["s1_prime"].append(float(pred == job["s1_prime"]))
    return {k: {m: torch.tensor(v) for m, v in d.items()} for k, d in cells.items()}


def held_out_best(cells: dict, kind: str, variant: str, metric: str, n: int) -> tuple[torch.Tensor, list]:
    """Cross-validated best cell: items split into halves by index parity; each half is scored at the (window,
    scale) that maximises the mean metric on the other half. Returns per-item scores and the two chosen cells."""
    keys = [k for k in cells if k[0] == kind and k[1] == variant]
    halves = [torch.arange(n) % 2 == 0, torch.arange(n) % 2 == 1]
    scores, chosen = torch.zeros(n), []
    for this, other in ((0, 1), (1, 0)):
        best = max(keys, key=lambda k: (float(cells[k][metric][halves[other]].mean()), -k[3], -k[2]))
        chosen.append(best[2:])
        scores[halves[this]] = cells[best][metric][halves[this]]
    return scores, chosen


def summarise_h2(blob: dict) -> dict:
    """The quantities the repair stage needs: per direction, the scale with the highest mean column-specific flip
    over all items and windows (ties to the smaller scale)."""
    cells = h2_outcomes(blob)
    chosen = {}
    for kind in DIRECTIONS:
        by_scale = {s: torch.stack([cells[(kind, "main", w, s)]["flip"] for w in range(len(WINDOWS))]).mean()
                    for s in SCALES}
        chosen[kind] = max(SCALES, key=lambda s: (float(by_scale[s]), -s))
    return {"cells": cells, "chosen_scale": chosen}


def deadline(curve: torch.Tensor) -> tuple[int, int]:
    """(first, last) window index whose value is >= half the curve's maximum (the curve's maximum must be > 0)."""
    above = (curve >= 0.5 * curve.max()).nonzero().flatten()
    return int(above[0]), int(above[-1])


def report_readouts(lines: list[str]) -> None:
    res = torch.load(EXP.cache / "readouts.pt", weights_only=False)
    metas = {m["item_id"]: m for m in load_capture()}
    colours = res["colours"]
    lines.append("## Gate 0 (descriptive): readability of each state")
    lines.append("lens AUC = P(score(state) > score(c)) for c off the item's path, averaged over items (0.5 = "
                 "chance); probe = held-out accuracy of the exp11 colour probe, state minus row-mate decoy")
    for h in (1, 2, 3):
        ms = [m for m in metas.values() if m["h"] == h]
        for pos_index, pos in enumerate(ms[0]["stored_positions"]):
            for j in range(1, h + 1):
                lines.append(f"\n### h={h} position={pos} state s{j}")
                for kind in ("logit", "jpp", "rlens"):
                    aucs = []
                    for m in ms:
                        sc = res["lens_scores"][m["item_id"]][kind][:, pos_index]            # [L, 10]
                        state = colours.index(m["brew"].path()[j])
                        others = [colours.index(c) for c in off_path(m["brew"])]
                        aucs.append((sc[:, [state]] > sc[:, others]).float().mean(1))
                    auc = torch.stack(aucs).mean(0)
                    best = int(auc.argmax())
                    row = " ".join(f"{LENS_LAYERS[k]}:{auc[k]:.2f}" for k in range(0, len(LENS_LAYERS), 3))
                    lines.append(f"  {kind:6s} best layer {LENS_LAYERS[best]} AUC {auc[best]:.2f} | {row}")
                st, dc = res["probe"][(h, pos, j, "state")], res["probe"][(h, pos, j, "decoy")]
                acc_s, acc_d = st["correct"].float().mean(0), dc["correct"].float().mean(0)
                best = int((acc_s - acc_d).argmax())
                row = " ".join(f"{l}:{acc_s[l] - acc_d[l]:+.2f}" for l in range(0, N_LAYERS, 6))
                lines.append(f"  probe  best layer {best} state {acc_s[best]:.2f} decoy {acc_d[best]:.2f} | {row}")


def report_h2(lines: list[str], g: torch.Generator) -> dict:
    blob = torch.load(EXP.cache / "swaps_h2.pt", weights_only=False)
    summary = summarise_h2(blob)
    cells = summary["cells"]
    n = len(blob["results"])
    lines.append(f"\n## Gate 1: s1 swaps on {n} correct h=2 items (windows of 4 even layers; window w starts at "
                 f"layer 4w)")
    lines.append("flip = P(answer = table[s1'][stir 2]) - P(answer = one given other entry of s1''s line); "
                 "held-out = each half of the items scored at the other half's best (window, scale)")
    verdicts = {}
    for kind in DIRECTIONS:
        flip, chosen = held_out_best(cells, kind, "main", "flip", n)
        ceiling, chosen_c = held_out_best(cells, kind, "answer_swap", "success", n)
        null = torch.stack([cells[(kind, "random_null", w, NULL_SCALE)]["success"] for w in range(len(WINDOWS))])
        f_ci, c_ci = boot_ci(flip, g), boot_ci(ceiling, g)
        passed = f_ci[0] >= 0.10 and f_ci[1] > 0 and c_ci[0] >= 0.30
        verdicts[kind] = passed
        lines.append(f"  {kind:9s} held-out flip {fmt(f_ci)} at {chosen}; answer-swap ceiling {fmt(c_ci)} at "
                     f"{chosen_c}; random-null success max over windows {float(null.mean(1).max()):.3f}; "
                     f"chosen scale for the repair stage {summary['chosen_scale'][kind]} -> "
                     f"{'PASS' if passed else 'fail'}")
    lines.append("\n### per window (scale = the direction's chosen scale): main flip / main success / answer-swap "
                 "success / changed (main)")
    for kind in DIRECTIONS:
        s = summary["chosen_scale"][kind]
        lines.append(f"  {kind}, scale {s}:")
        for w, window in enumerate(WINDOWS):
            mc, ac = cells[(kind, "main", w, s)], cells[(kind, "answer_swap", w, s)]
            lines.append(f"    layers {window[0]:2d}-{window[-1]:2d}: flip {mc['flip'].mean():+.3f}  success "
                         f"{mc['success'].mean():.3f}  answer-swap {ac['success'].mean():.3f}  changed "
                         f"{mc['changed'].mean():.3f}")
    lines.append("\n## Stage 2: timing (directions that passed Gate 1; first/last window with value >= half the "
                 "maximum; bootstrap over items)")
    for kind in DIRECTIONS:
        if not verdicts[kind]:
            continue
        s = summary["chosen_scale"][kind]
        flips = torch.stack([cells[(kind, "main", w, s)]["flip"] for w in range(len(WINDOWS))], 1)      # [n, W]
        answers = torch.stack([cells[(kind, "answer_swap", w, s)]["success"] for w in range(len(WINDOWS))], 1)
        d_s1, d_ans = deadline(flips.mean(0))[1], deadline(answers.mean(0))[1]
        boots = []
        for _ in range(N_BOOT):
            idx = torch.randint(n, (n,), generator=g)
            fm, am = flips[idx].mean(0), answers[idx].mean(0)
            if fm.max() > 0 and am.max() > 0:
                boots.append(deadline(am)[1] - deadline(fm)[1])
        boots = torch.tensor(boots, dtype=torch.float)
        lines.append(f"  {kind}: last window, s1 swap layers {WINDOWS[d_s1]}, answer swap layers {WINDOWS[d_ans]}; "
                     f"(answer - s1) in windows {d_ans - d_s1:+d} [{boots.quantile(0.025):+.0f}, "
                     f"{boots.quantile(0.975):+.0f}]")
    return {"verdicts": verdicts, "summary": summary}


def report_h3(lines: list[str], g: torch.Generator, h2: dict) -> None:
    blob = torch.load(EXP.cache / "swaps_h3.pt", weights_only=False)
    colours = blob["colours"]
    jobs = {j["item_id"]: j for j in blob["jobs"]}
    lines.append(f"\n## Stage 3: h=3 items answered table[s1][stir 3] (stir 2 skipped), n={len(blob['results'])}")
    lines.append("repair = swap s1 -> s2, scored as answer = gold; redirect = swap s1 -> c, scored as answer = "
                 "table[c][stir 3] (and its gold rate)")
    for kind in DIRECTIONS:
        per = defaultdict(lambda: defaultdict(list))
        for rec in blob["results"]:
            job = jobs[rec["item_id"]]
            for t in rec["trials"]:
                if t["kind"] != kind:
                    continue
                pred = colours[int(t["logits"].argmax())]
                per[(t["variant"], t["window"])]["gold"].append(float(pred == job["gold"]))
                per[(t["variant"], t["window"])]["c_answer"].append(float(pred == job["c_answer"]))
        if not per:
            continue
        tag = "passed Gate 1" if h2["verdicts"][kind] else "did not pass Gate 1: not read"
        scale = h2["summary"]["chosen_scale"][kind]
        lines.append(f"  {kind} ({tag}), scale {scale}:")
        for w, window in enumerate(WINDOWS):
            rep, red = per[("repair", w)], per[("redirect", w)]
            diff = torch.tensor(rep["gold"]) - torch.tensor(red["gold"])
            lines.append(f"    layers {window[0]:2d}-{window[-1]:2d}: repair gold {sum(rep['gold']) / len(rep['gold']):.3f} "
                         f"(minus redirect gold {fmt(boot_ci(diff, g))})  redirect to table[c][stir 3] "
                         f"{sum(red['c_answer']) / len(red['c_answer']):.3f}")
        # the pre-registered reading: at the window where this direction's h=2 s1 swap flips most (other items)
        cells = h2["summary"]["cells"]
        w_star = int(torch.stack([cells[(kind, "main", w, scale)]["flip"].mean() for w in range(len(WINDOWS))])
                     .argmax())
        rep, red = per[("repair", w_star)], per[("redirect", w_star)]
        repairs = boot_ci(torch.tensor(rep["gold"]) - torch.tensor(red["gold"]), g)
        redirects = boot_ci(torch.tensor(red["c_answer"]) - torch.tensor(rep["c_answer"]), g)
        lines.append(f"    reading at layers {WINDOWS[w_star]}: repairable (repair gold - redirect gold >= 0.10, CI "
                     f"low > 0) {fmt(repairs)} -> {repairs[0] >= 0.10 and repairs[1] > 0}; slot_read (redirect "
                     f"to table[c][stir 3] - the same under repair >= 0.10, CI low > 0) {fmt(redirects)} -> "
                     f"{redirects[0] >= 0.10 and redirects[1] > 0}")


def report(run: str) -> None:
    out = EXP.results / run / "report.txt"
    require_absent(out)
    g = torch.Generator().manual_seed(0)
    lines = [f"exp14 report ({run}); manifest results/exp14_brew_swaps/manifest.json", ""]
    metas = [json.loads(line) for line in capture_paths()[1].open()]
    for h in (1, 2, 3):
        ms = [m for m in metas if m["h"] == h]
        lines.append(f"clean accuracy h={h}: {sum(m['predicted'] == m['answer'] for m in ms) / len(ms):.3f} "
                     f"(n={len(ms)}; exp12 G1: {({1: 1.000, 2: 0.924, 3: 0.048})[h]:.3f})")
    lines.append("")
    if (EXP.cache / "readouts.pt").exists():
        report_readouts(lines)
    h2 = report_h2(lines, g) if (EXP.cache / "swaps_h2.pt").exists() else None
    if h2 and (EXP.cache / "swaps_h3.pt").exists():
        report_h3(lines, g, h2)
    text = "\n".join(lines)
    print(text)
    out.write_text(text + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["capture", "readouts", "swaps", "repair", "report"])
    p.add_argument("--run")
    a = p.parse_args()
    {"capture": capture, "readouts": readouts, "swaps": swaps, "repair": repair,
     "report": lambda: report(a.run)}[a.cmd]()


if __name__ == "__main__":
    main()
