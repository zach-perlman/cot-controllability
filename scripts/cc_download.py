"""Download the pinned snapshot of each named model (the revision in cc_config) into $HF_HOME/hub.

Refuses to start a download that would leave less than MIN_FREE_GB on the disk (a full disk mid-run kills vLLM and
the partial cache files). Snapshots that are already complete are skipped.

Run: /venv/main/bin/python scripts/cc_download.py Qwen3.6-27B-FP8 [more models]
"""

from __future__ import annotations

import argparse
import shutil

import cc_config as cfg

MIN_FREE_GB = 20


def spec_of(model: str) -> dict:
    return cfg.JUDGE_MODEL if model == cfg.JUDGE_MODEL["name"] else cfg.ALL_MODELS[model]


def main() -> None:
    choices = list(cfg.ALL_MODELS) + [cfg.JUDGE_MODEL["name"]]
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=choices)
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for model in args.models:
        spec = spec_of(model)
        info = api.model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
        missing = [s for s in info.siblings if not (cfg.model_dir(model) / s.rfilename).exists()]
        if not missing:
            print(f"{model}: {spec['repo']}@{spec['revision'][:8]} already complete")
            continue
        need_gb = sum(s.size or 0 for s in missing) / 1e9
        free_gb = shutil.disk_usage(cfg.HF_HUB_DIR).free / 1e9
        if free_gb - need_gb < MIN_FREE_GB:
            raise SystemExit(f"{model}: needs {need_gb:.1f} GB, {free_gb:.1f} GB free; would leave less than "
                             f"{MIN_FREE_GB} GB")
        print(f"{model}: downloading {need_gb:.1f} GB ({len(missing)} files) of {spec['repo']}@{spec['revision']}")
        snapshot_download(spec["repo"], revision=spec["revision"], cache_dir=cfg.HF_HUB_DIR)
        print(f"{model}: done -> {cfg.model_dir(model)}")


if __name__ == "__main__":
    main()
