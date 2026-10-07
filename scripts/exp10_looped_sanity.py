"""Sanity check for exp10's looped-model arm: does a candidate model work outside code, and on chain?

Not part of any pre-registered result: it decides which models enter exp10's manifest. IQuest-Coder is a code
specialist with no published general benchmarks, so it is checked against a general model (Qwen3-32B-FP8) on:
  gsm8k     60 GSM8K test problems (thinking on), the number after the last "Answer:"
  mmlu_pro  60 MMLU-Pro questions from CoT-Control's mini set (thinking on), the letter after the last "Answer:"
  ifeval    30 verifiable format instructions (thinking on), graded on the response after the reasoning:
              lowercase  no capital letter at all
              bullets    exactly 3 non-empty lines, each starting with "- "
              ending     fewer than 50 words, ending with "That is all."
  chain_c0  chain's dev eval items under C0 (exp10_task_conditions.fields; no-CoT, "Answer:" prefill)
  chain_c2  the same items under C2 (CoT), graded with cc_exp10_tasks.check
Every grader is deterministic (no LLM judge). Outputs go to cache/exp10/looped_sanity/ (gitignored).

  /venv/main/bin/python scripts/exp10_looped_sanity.py requests
  scripts/vllm_python.sh scripts/exp10_looped_sanity.py generate --model IQuest-40B-Loop-Thinking [--loop-num 1]
  /venv/main/bin/python scripts/exp10_looped_sanity.py grade
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import random
import re
from pathlib import Path

import cc_config as cfg

OUT = cfg.REPO_ROOT / "cache" / "exp10" / "looped_sanity"
REQUESTS = OUT / "requests.jsonl"
SEED = 20261006
N_GSM8K = 60
N_MMLU = 60
LETTERS = "ABCDEFGHIJ"
ANSWER_FORMAT = {"number": "Give your final answer in the format 'Answer: [ANSWER]' where [ANSWER] is just the "
                           "final number.",
                 "letter": "Give your final answer in the format 'Answer: [LETTER]' where [LETTER] is the letter "
                           "of the correct option."}
IF_TOPICS = ["the water cycle", "how bees make honey", "the history of the bicycle", "why the sky is blue",
             "how vaccines work", "the rules of chess", "volcanoes", "the printing press", "photosynthesis",
             "the phases of the moon"]
IF_PROMPTS = {
    "lowercase": "Write a short paragraph about {topic}. Your entire response must be in lowercase letters: no "
                 "capital letters at all.",
    "bullets": "Give exactly 3 bullet points about {topic}. Each line must start with '- ', and write nothing "
               "else: no title, no introduction, no closing line.",
    "ending": "Describe {topic} in fewer than 50 words, and end your response with the exact phrase 'That is all.'",
}
SAMPLING = {  # the card's thinking-mode settings; C0 rows are greedy (nocot-bench's calibration setting)
    "iquest": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    "qwen3": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "nanbeige4.2": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},  # card: reasoning and chat tasks
    "ouro": {"temperature": 1.0, "top_p": 0.7, "top_k": 0},  # card's generate() example (top_k 0: off)
}
MAX_TOKENS = {"think": 16000, "direct": 100}
MAX_MODEL_LEN = 20480
# The config field that sets the loop count, per family (vLLM: hf_overrides; HF: set on the loaded config).
LOOP_FIELD = {"iquest": "loop_num", "nanbeige4.2": "num_loops", "ouro": "total_ut_steps"}
HF_BATCH = 32
# Ouro under SDPA fails on left-padded batches ("(*bias): last dimension must be contiguous"); under eager attention
# padded batches run but give garbage (Ouro-1.4B chain C0: 4/105 greedy outputs match the unbatched run, the rest
# " 1." and blank lines), so Ouro runs unpadded, one sequence at a time.
HF_ATTENTION = {"ouro": "eager"}
HF_BATCH_BY_FAMILY = {"ouro": 1}


# --- Requests (/venv/main) ------------------------------------------------------------------------------------------
def gsm8k_rows() -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    picks = random.Random(SEED).sample(range(len(df)), N_GSM8K)
    return [{"id": f"gsm8k:{i}", "suite": "gsm8k", "mode": "think",
             "messages": [{"role": "user", "content": f"{df.question[i]}\n\n{ANSWER_FORMAT['number']}"}],
             "gold": df.answer[i].split("####")[-1].strip().replace(",", "")} for i in picks]


def mmlu_rows() -> list[dict]:
    import pandas as pd
    df = pd.read_csv(cfg.COTCONTROL_QA_DIR / "datasets" / cfg.DATASETS["MMLU-Pro"])
    picks = random.Random(SEED).sample(range(len(df)), N_MMLU)
    rows = []
    for i in picks:
        options = ast.literal_eval(df.options[i])
        listing = "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(options))
        rows.append({"id": f"mmlu_pro:{i}", "suite": "mmlu_pro", "mode": "think",
                     "messages": [{"role": "user", "content": f"{df.question[i]}\n\nOptions:\n{listing}\n\n"
                                                              f"{ANSWER_FORMAT['letter']}"}],
                     "gold": LETTERS[options.index(df.answer[i])]})
    return rows


def ifeval_rows() -> list[dict]:
    return [{"id": f"ifeval:{kind}:{t}", "suite": "ifeval", "mode": "think", "constraint": kind,
             "messages": [{"role": "user", "content": template.format(topic=topic)}], "gold": None}
            for kind, template in IF_PROMPTS.items() for t, topic in enumerate(IF_TOPICS)]


def chain_rows() -> list[dict]:
    import cc_exp10_tasks as X
    import exp10_task_conditions as T
    shots = X.shots("chain", "dev")
    rows = []
    for item in X.evals("chain", "dev"):
        for condition in ("C0", "C2"):
            f = T.fields("chain", condition, item, shots)
            messages = [{"role": "system", "content": f["system"]}] if f["system"] else []
            for turn in f["history"]:
                messages += [{"role": "user", "content": turn["user"]},
                             {"role": "assistant", "content": turn["answer"]}]
            messages.append({"role": "user", "content": f["user"]})
            rows.append({"id": f"chain_{condition.lower()}:{item['item_id']}", "suite": f"chain_{condition.lower()}",
                         "mode": "direct" if condition == "C0" else "think", "messages": messages,
                         "response_prefix": f["response_prefix"], "gold": item["answer"], "item": item})
    return rows


def write_requests() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if REQUESTS.exists():
        raise SystemExit(f"{REQUESTS} exists; request files are fixed once written")
    rows = gsm8k_rows() + mmlu_rows() + ifeval_rows() + chain_rows()
    with REQUESTS.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(rows)} requests -> {REQUESTS}")


# --- Generation (vLLM venv) -----------------------------------------------------------------------------------------
def run_name(model: str, loop_num: int | None) -> str:
    return model if loop_num is None else f"{model}_loop{loop_num}"


def render(model: str, suites: list[str] | None) -> tuple[list[dict], list[str]]:
    from transformers import AutoTokenizer
    rows = [r for r in map(json.loads, REQUESTS.open()) if suites is None or r["suite"] in suites]
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model_dir(model)), trust_remote_code=True)
    prompts = []
    for row in rows:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=row["mode"] == "think")
        prompts.append(text + (row.get("response_prefix") or ""))
    return rows, prompts


def write_generations(out_path: Path, model: str, loop_num: int | None, rows, prompts, texts, finishes, counts):
    with out_path.open("w") as f:
        for row, prompt, text, finish, n in zip(rows, prompts, texts, finishes, counts):
            f.write(json.dumps({"id": row["id"], "model": model, "loop_num": loop_num, "prompt": prompt,
                                "text": text, "finish": finish, "tokens": n}) + "\n")
    print(f"{len(rows)} rows -> {out_path}")


def generate(model: str, loop_num: int | None, suites: list[str] | None, think_tokens: int) -> None:
    from vllm import LLM, SamplingParams
    spec = cfg.ALL_MODELS[model]
    out_path = OUT / f"generations_{run_name(model, loop_num)}.jsonl"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists")
    rows, prompts = render(model, suites)
    overrides = {LOOP_FIELD[spec["family"]]: loop_num} if loop_num is not None else None
    llm = LLM(str(cfg.model_dir(model)), trust_remote_code=True, max_model_len=MAX_MODEL_LEN,
              seed=cfg.VLLM_ENGINE_SEED, gpu_memory_utilization=spec.get("gpu_memory_utilization", 0.9),
              hf_overrides=overrides, **spec.get("vllm_args", {}))
    card = SAMPLING[spec["family"]]
    # Ouro's generation_config lists only <|endoftext|> as EOS, so vLLM runs past the end of the turn without this.
    im_end = llm.get_tokenizer().convert_tokens_to_ids("<|im_end|>")
    params = [SamplingParams(max_tokens=think_tokens if r["mode"] == "think" else MAX_TOKENS["direct"],
                             seed=SEED + k, stop_token_ids=[im_end],
                             skip_special_tokens=False,  # Ouro's </think> is a special token (id 4)
                             **(card if r["mode"] == "think" else {"temperature": 0.0}))
              for k, r in enumerate(rows)]
    outputs = [out.outputs[0] for out in llm.generate(prompts, params)]
    write_generations(out_path, model, loop_num, rows, prompts, [o.text for o in outputs],
                      [o.finish_reason for o in outputs], [len(o.token_ids) for o in outputs])


def generate_hf(model: str, loop_num: int | None, suites: list[str] | None, think_tokens: int,
                batch_size: int | None = None, shard: tuple[int, int] = (0, 1)) -> None:
    """The same requests under HF transformers (Ouro, which vLLM dropped; and a reference for vLLM's generic
    backend). Batches of similar-length prompts, left-padded (Ouro: one unpadded sequence at a time); one torch
    seed per batch, so rows are not seed-matched to the vLLM run. shard (i, n): every n-th request from the i-th,
    so n processes can share the GPU (batch-1 decoding leaves it mostly idle); grade() merges the shards."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    spec = cfg.ALL_MODELS[model]
    path = str(cfg.model_dir(model))
    default_batch = HF_BATCH_BY_FAMILY.get(spec["family"], HF_BATCH)
    batch_size = batch_size or default_batch
    suffix = "_hf" if batch_size == default_batch else f"_hf_b{batch_size}"
    shard_i, shard_n = shard
    if shard_n > 1:
        suffix += f"_s{shard_i}of{shard_n}"
    out_path = OUT / f"generations_{run_name(model, loop_num)}{suffix}.jsonl"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists")
    rows, prompts = render(model, suites)
    rows, prompts = rows[shard_i::shard_n], prompts[shard_i::shard_n]
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    if loop_num is not None:
        setattr(config, LOOP_FIELD[spec["family"]], loop_num)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    net = AutoModelForCausalLM.from_pretrained(path, config=config, torch_dtype=torch.bfloat16, device_map="cuda",
                                               trust_remote_code=True,
                                               attn_implementation=HF_ATTENTION.get(spec["family"], "sdpa")).eval()
    card = SAMPLING[spec["family"]]
    eos = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id]
    texts, finishes, counts = [None] * len(rows), [None] * len(rows), [None] * len(rows)
    for mode in ("direct", "think"):
        order = sorted((k for k, r in enumerate(rows) if r["mode"] == mode), key=lambda k: len(prompts[k]))
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            torch.manual_seed(SEED + start)
            enc = tokenizer([prompts[k] for k in batch], return_tensors="pt", padding=True,
                            add_special_tokens=False).to("cuda")
            sampling = ({"do_sample": True, **card} if mode == "think" else {"do_sample": False})
            cap = think_tokens if mode == "think" else MAX_TOKENS["direct"]
            with torch.no_grad():
                out = net.generate(**enc, max_new_tokens=cap, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
                                   **sampling)
            for k, seq in zip(batch, out[:, enc.input_ids.shape[1]:]):
                ids = seq.tolist()
                stop = next((i for i, t in enumerate(ids) if t in eos), None)
                ids = ids if stop is None else ids[:stop]
                texts[k] = tokenizer.decode(ids, skip_special_tokens=False)
                finishes[k], counts[k] = ("length" if stop is None else "stop"), len(ids)
            print(f"{mode}: {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    write_generations(out_path, model, loop_num, rows, prompts, texts, finishes, counts)


# --- Grading (/venv/main) -------------------------------------------------------------------------------------------
def split_think(text: str) -> tuple[str | None, str | None]:
    """(reasoning, response). The template ends at "<|im_start|>assistant\\n", so the model opens <think> itself;
    no "</think>" means the reasoning ran out of tokens (response None)."""
    if "</think>" not in text:
        return text, None
    reasoning, response = text.split("</think>", 1)
    return reasoning.replace("<think>", "", 1).strip(), response.strip()


def last_answer(response: str) -> str | None:
    matches = list(re.finditer(r"answer\s*[:：]\s*(.+)", response, re.IGNORECASE))
    return matches[-1].group(1).strip() if matches else None


def last_boxed(response: str) -> str | None:
    """The lenient reading's fallback: Qwen3-32B often ignores the 'Answer:' format for \\boxed{...}."""
    matches = re.findall(r"\\boxed\{([^{}]*)\}", response)
    return matches[-1].strip() if matches else None


def grade_ifeval(constraint: str, response: str) -> bool:
    if constraint == "lowercase":
        return not re.search(r"[A-Z]", response)
    if constraint == "bullets":
        lines = [l for l in response.splitlines() if l.strip()]
        return len(lines) == 3 and all(l.startswith("- ") for l in lines)
    return len(response.split()) < 50 and response.endswith("That is all.")


def grade_row(request: dict, gen: dict) -> dict:
    import cc_exp10_tasks as X
    out = {"id": gen["id"], "suite": request["suite"], "tokens": gen["tokens"], "finish": gen["finish"]}
    if request["mode"] == "direct":
        witnesses = X.nocot_module("nocot.witnesses")
        predicted, ok = X.check("chain", gen["text"], request["item"], direct=True)
        reasoned = witnesses.is_content_cot(gen["text"])
        return {**out, "truncated": False, "predicted": predicted, "correct": bool(ok) and not reasoned,
                "h": request["item"]["h"]}
    reasoning, response = split_think(gen["text"])
    out["truncated"] = response is None
    if response is None:
        return {**out, "predicted": None, "correct": False, "h": request.get("item", {}).get("h")}
    if request["suite"] == "ifeval":
        return {**out, "predicted": None, "correct": grade_ifeval(request["constraint"], response),
                "constraint": request["constraint"]}
    # correct: the requested "Answer:" format only; correct_lenient: else the last \boxed{} (capability, not format)
    strict_line = last_answer(response)
    readings = {"correct": strict_line, "correct_lenient": strict_line or last_boxed(response)}
    for key, line in readings.items():
        if request["suite"] == "chain_c2":
            predicted, ok = X.check("chain", line, request["item"], direct=False)
            out["h"] = request["item"]["h"]
        elif request["suite"] == "gsm8k":
            m = re.search(r"-?\d+(?:\.\d+)?", (line or "").replace(",", "").replace("$", ""))
            predicted = m.group(0) if m else None
            ok = predicted is not None and float(predicted) == float(request["gold"])
        else:
            m = re.search(r"\b([A-J])\b", line or "")
            predicted = m.group(1) if m else None
            ok = predicted == request["gold"]
        out[key] = bool(ok)
        if key == "correct":
            out["predicted"] = predicted
    return out


def grade() -> None:
    import pandas as pd
    requests = {r["id"]: r for r in map(json.loads, REQUESTS.open())}
    rows = []
    for path in sorted(glob.glob(str(OUT / "generations_*.jsonl"))):
        name = re.sub(r"_s\d+of\d+$", "", Path(path).stem.removeprefix("generations_"))  # shards: one run
        for gen in map(json.loads, open(path)):
            rows.append({"run": name, **grade_row(requests[gen["id"]], gen)})
    df = pd.DataFrame(rows)
    df["correct_lenient"] = df["correct_lenient"].fillna(df["correct"]).astype(bool)  # rows with one reading
    df.to_json(OUT / "grades.jsonl", orient="records", lines=True)
    summary = df.groupby(["run", "suite"]).agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                               lenient=("correct_lenient", "mean"),
                                               truncated=("truncated", "mean"), mean_tokens=("tokens", "mean"))
    print(summary.round(3).to_string())
    ife = df[df.suite == "ifeval"]
    if len(ife):
        print("\nifeval by constraint\n" + ife.groupby(["run", "constraint"]).correct.mean().unstack().round(2)
              .to_string())
    chain = df[df.suite.str.startswith("chain")]
    if len(chain):
        print("\nchain accuracy by h\n" + chain.groupby(["run", "suite", "h"]).correct.mean().unstack().round(2)
              .to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["requests", "generate", "generate_hf", "grade"])
    parser.add_argument("--model", choices=list(cfg.ALL_MODELS))
    parser.add_argument("--loop-num", type=int, help="override a looped model's loop count (LOOP_FIELD)")
    parser.add_argument("--suites", nargs="+", help="only these suites (default: all)")
    parser.add_argument("--think-tokens", type=int, default=MAX_TOKENS["think"])
    parser.add_argument("--hf-batch", type=int, help="default: HF_BATCH_BY_FAMILY, else HF_BATCH; 1 is unpadded")
    parser.add_argument("--shard", default="0/1", help="i/n: generate_hf's every n-th request from the i-th")
    args = parser.parse_args()
    if args.command == "requests":
        write_requests()
    elif args.command == "generate":
        generate(args.model, args.loop_num, args.suites, args.think_tokens)
    elif args.command == "generate_hf":
        shard_i, shard_n = map(int, args.shard.split("/"))
        generate_hf(args.model, args.loop_num, args.suites, args.think_tokens, args.hf_batch, (shard_i, shard_n))
    else:
        grade()


if __name__ == "__main__":
    main()
