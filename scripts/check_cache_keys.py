"""Print the cache keys the exp01/exp02 outputs depend on (grader code key, judge key, generation file names).

Run it before and after a code change (e.g. on a `git worktree` of the previous commit and on the working tree);
identical output means every cached generation, grade and judge file is still found under its old name.
Run: /venv/main/bin/python scripts/check_cache_keys.py
"""

from __future__ import annotations

import json

import cc_config as cfg
import cc_generate as gen
import cc_grade
import cc_judge


def generation_name(exp: str, model: str, prompts=None, modes=None) -> str:
    requests_path = cfg.EXPERIMENTS[exp].requests
    requests = gen.select_requests(requests_path, prompts, modes)
    name, sampling = gen.sampling_for(exp, cfg.ALL_MODELS[model]["family"], None)
    return gen.output_path(requests_path, model, name, sampling, requests).name


def main() -> None:
    keys = {"grade_code_key": cc_grade.code_key(), "judge_key": cc_judge.judge_key()}
    for model in cfg.GATE_MODELS:
        keys[f"exp01/{model}"] = generation_name("exp01", model)
    for model in cfg.GRID_MODELS:
        keys[f"exp02/{model}"] = generation_name("exp02", model)
    check = cfg.PRECISION_CHECK
    keys[f"exp02/{check['model']}"] = generation_name("exp02", check["model"], [check["prompt"]], check["modes"])
    print(json.dumps(keys, indent=1))


if __name__ == "__main__":
    main()
