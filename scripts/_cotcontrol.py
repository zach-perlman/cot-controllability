"""Import helpers for the pinned CoT-Control repo (third_party/CoTControl/CoT-Control-QA).

Its modules are flat (prompts.py, grading.py, run_cceval.py, grade_compliance_csv.py, llm.py) and import each other
by bare name, so the QA directory goes on sys.path and they are imported as-is. Nothing in the repo is edited.
Needs pandas, python-dotenv and tqdm (the /venv/main research env).
"""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType

import cc_config as cfg


def _use_cotcontrol() -> None:
    if not (cfg.COTCONTROL_QA_DIR / "grading.py").is_file():
        raise FileNotFoundError(f"CoT-Control missing at {cfg.COTCONTROL_QA_DIR}; run `git submodule update --init`.")
    root = str(cfg.COTCONTROL_QA_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


def module(name: str) -> ModuleType:
    """One of CoT-Control-QA's modules: prompts, grading, run_cceval, grade_compliance_csv, llm."""
    _use_cotcontrol()
    return __import__(name)


def cotcontrol_commit() -> str:
    commit = subprocess.run(["git", "-C", str(cfg.COTCONTROL_QA_DIR), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    if commit != cfg.COTCONTROL_COMMIT:
        raise RuntimeError(f"CoT-Control is at {commit}, expected {cfg.COTCONTROL_COMMIT}")
    return commit
