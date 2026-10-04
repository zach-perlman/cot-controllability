"""Run generation jobs as lanes on GPUs: up to two vLLM engines share a GPU under CUDA MPS (so their kernels run at
the same time instead of time-slicing), each with its own fraction of the GPU's memory.

A plan (JSON) lists, per GPU, its lanes; a lane is a list of jobs run one after another:
  {"name": "...", "mps": true,
   "gpus": {"0": [[job, job], [job]], "1": [[job], [job, job]]}}
  job = {"exp": "exp09", "model": "Qwen3.8-27B-FP8", "requests": "cache/exp09/requests_Qwen3.8-27B-FP8.jsonl",
         "items": "cache/exp09/items.jsonl", "memory": 0.45, "after": ["/venv/main/bin/python", "cc_exp09.py", ...]}
Paths are relative to the repo root. A job starts when (1) the jobs before it in its lane are done, (2) no other
engine on its GPU is still loading (two engines profiling memory at once would each count the other's), and
(3) its memory fraction plus those of the engines running on the GPU is at most MAX_GPU_MEMORY. A job is its
cc_generate_abort.py shard (one shard, resumable), the merge, then its optional "after" command (grading), which runs
on the CPU in the background while the lane moves on.

Every POLL_SECONDS the progress of each running job (finished rows and generated tokens, from its shard file) is
appended to log/<exp>/lanes_<plan name>/progress.tsv; per-job logs are in the same directory.
  /venv/main/bin/python scripts/gpu_lanes.py plans/exp09_1xH200.json [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
MAX_GPU_MEMORY = 0.92  # total memory fraction of the engines on one GPU
POLL_SECONDS = 30
LOAD_TIMEOUT_SECONDS = 1800
MPS_PIPE = "/tmp/nvidia-mps"
MPS_LOG = "/tmp/nvidia-mps-log"


@dataclass
class Job:
    gpu: str
    lane: int
    position: int
    spec: dict
    log: Path
    proc: subprocess.Popen | None = None
    started: float | None = None
    loaded: bool = False
    finished: float | None = None
    failed: bool = False
    after: subprocess.Popen | None = None
    stream_lines: int = field(default=0)

    @property
    def name(self) -> str:
        return f"gpu{self.gpu}.lane{self.lane}.{self.position}_{self.spec['model']}"

    def stream_file(self) -> Path | None:
        requests = REPO / self.spec["requests"]
        stem = requests.stem.removeprefix("requests")
        found = glob.glob(str(requests.parent / "generations" /
                              f"{self.spec['model']}__card__stream_abort{stem}__*.parts" / "stream.shard0of1.jsonl"))
        return Path(found[0]) if len(found) == 1 else None


def stamp(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", flush=True)


# --- MPS ------------------------------------------------------------------------------------------------------------
def mps_env() -> dict:
    return {"CUDA_MPS_PIPE_DIRECTORY": MPS_PIPE, "CUDA_MPS_LOG_DIRECTORY": MPS_LOG}


def mps_running() -> bool:
    """The control socket outlives the daemon, so look for the process (its name truncated to 15 characters)."""
    return subprocess.run(["pgrep", "-x", "nvidia-cuda-mps"], capture_output=True).returncode == 0


def start_mps(gpus: list[str]) -> None:
    for d in (MPS_PIPE, MPS_LOG):
        Path(d).mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **mps_env(), "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
    if subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env).returncode != 0:
        raise SystemExit("could not start the MPS control daemon")
    stamp(f"MPS control daemon started for GPUs {','.join(gpus)} (pipe {MPS_PIPE})")


def stop_mps() -> None:
    subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env={**os.environ, **mps_env()})
    stamp("MPS control daemon stopped")


# --- Jobs -----------------------------------------------------------------------------------------------------------
def generate_command(spec: dict, merge: bool) -> list[str]:
    command = [str(SCRIPTS / "vllm_python.sh"), str(SCRIPTS / "cc_generate_abort.py"), "--exp", spec["exp"],
               "--model", spec["model"], "--requests", str(REPO / spec["requests"]),
               "--items", str(REPO / spec["items"])]
    return command + (["--merge-shards", "1"] if merge else ["--shard", "0/1"])


def start(job: Job, use_mps: bool) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": job.gpu, "CC_TENSOR_PARALLEL": "1",
           "CC_GPU_MEMORY_UTILIZATION": str(job.spec["memory"])}
    if use_mps:
        env.update(mps_env())
    job.log.parent.mkdir(parents=True, exist_ok=True)
    with job.log.open("a") as log:
        job.proc = subprocess.Popen(generate_command(job.spec, merge=False), cwd=SCRIPTS, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
    job.started = time.time()
    stamp(f"start {job.name} (memory {job.spec['memory']}) -> {job.log}")


def finish(job: Job) -> None:
    """After the shard exits: merge, then the job's "after" command in the background."""
    ok = job.proc.returncode == 0
    if ok:
        with job.log.open("a") as log:
            ok = subprocess.run(generate_command(job.spec, merge=True), cwd=SCRIPTS, stdout=log,
                                stderr=subprocess.STDOUT).returncode == 0
    job.finished, job.failed = time.time(), not ok
    minutes = (job.finished - job.started) / 60
    stamp(f"{'FAILED' if job.failed else 'done'} {job.name} after {minutes:.1f} min")
    if ok and job.spec.get("after"):
        with job.log.with_suffix(".after.log").open("a") as log:
            job.after = subprocess.Popen(job.spec["after"], cwd=SCRIPTS, stdout=log, stderr=subprocess.STDOUT)


def loaded(job: Job) -> bool:
    return "engine loaded" in job.log.read_text(errors="replace") if job.log.exists() else False


def progress_row(job: Job) -> str | None:
    path = job.stream_file()
    if path is None:
        return None
    rows = tokens = 0
    for line in path.open():
        r = json.loads(line)
        rows += 1
        tokens += r["reasoning_tokens"] + r["answer_tokens"]
    return f"{int(time.time())}\t{job.name}\t{round(time.time() - job.started)}\t{rows}\t{tokens}"


# --- Plan -----------------------------------------------------------------------------------------------------------
def load_jobs(plan: dict, log_dir: Path) -> list[Job]:
    jobs = []
    for gpu, lanes in plan["gpus"].items():
        if len(lanes) > 2:
            raise SystemExit(f"GPU {gpu}: {len(lanes)} lanes; at most 2 engines share a GPU")
        for lane, specs in enumerate(lanes):
            for position, spec in enumerate(specs):
                job = Job(gpu, lane, position, spec, log_dir / f"gpu{gpu}.lane{lane}.{position}_{spec['model']}.log")
                jobs.append(job)
    for gpu in plan["gpus"]:
        biggest = sorted((j.spec["memory"] for j in jobs if j.gpu == gpu), reverse=True)[:2]
        if len(plan["gpus"][gpu]) == 2 and sum(biggest) > MAX_GPU_MEMORY:
            stamp(f"note: GPU {gpu}'s two largest jobs ({biggest}) cannot run at the same time; they will queue")
    return jobs


def can_start(job: Job, jobs: list[Job]) -> bool:
    earlier = [j for j in jobs if j.gpu == job.gpu and j.lane == job.lane and j.position < job.position]
    if any(j.finished is None or j.failed for j in earlier):
        return False
    on_gpu = [j for j in jobs if j.gpu == job.gpu and j.proc is not None and j.finished is None]
    if any(not j.loaded for j in on_gpu):
        return False
    return sum(j.spec["memory"] for j in on_gpu) + job.spec["memory"] <= MAX_GPU_MEMORY + 1e-9


def run(plan: dict, plan_path: Path) -> None:
    exp = plan.get("log_group") or plan["gpus"][next(iter(plan["gpus"]))][0][0]["exp"]
    log_dir = REPO / "log" / exp / f"lanes_{plan['name']}"
    jobs = load_jobs(plan, log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "plan.json").write_text(plan_path.read_text())
    progress = (log_dir / "progress.tsv").open("a")
    progress.write("unix_time\tjob\tseconds_since_start\trows_done\ttokens_generated\n")
    use_mps = plan.get("mps", True)
    if use_mps:
        if mps_running():
            raise SystemExit(f"an MPS daemon is already running ({MPS_PIPE}); stop it first")
        start_mps(list(plan["gpus"]))
    started = time.time()
    try:
        while any(j.finished is None for j in jobs):
            for job in jobs:
                if job.proc is None and job.finished is None and can_start(job, jobs):
                    start(job, use_mps)
                elif job.proc is not None and job.finished is None:
                    if not job.loaded and loaded(job):
                        job.loaded = True
                        stamp(f"loaded {job.name} after {(time.time() - job.started) / 60:.1f} min")
                    if job.proc.poll() is not None:
                        job.loaded = True
                        finish(job)
                    elif not job.loaded and time.time() - job.started > LOAD_TIMEOUT_SECONDS:
                        raise SystemExit(f"{job.name} did not load within {LOAD_TIMEOUT_SECONDS} s")
            blocked = [j for j in jobs if j.proc is None and any(
                e.failed for e in jobs if e.gpu == j.gpu and e.lane == j.lane and e.position < j.position)]
            for job in blocked:  # a lane stops at its first failed job (rerun the plan: finished rows are kept)
                job.finished, job.failed = time.time(), True
                stamp(f"skipped {job.name}: an earlier job in its lane failed")
            for job in jobs:
                if job.proc is not None and job.finished is None and job.loaded:
                    row = progress_row(job)
                    if row:
                        progress.write(row + "\n")
            progress.flush()
            time.sleep(POLL_SECONDS)
        for job in jobs:
            if job.after is not None:
                job.after.wait()
                stamp(f"after-command of {job.name}: exit {job.after.returncode}")
    finally:
        for job in jobs:
            if job.proc is not None and job.proc.poll() is None:
                os.killpg(job.proc.pid, 15)
        if use_mps:
            stop_mps()
    failed = [j.name for j in jobs if j.failed]
    stamp(f"plan {plan['name']} finished in {(time.time() - started) / 3600:.2f} h; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the jobs and their order, start nothing")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.dry_run:
        for job in load_jobs(plan, Path("/tmp")):
            print(f"GPU {job.gpu} lane {job.lane} #{job.position}: {job.spec['model']} memory {job.spec['memory']} "
                  f"requests {job.spec['requests']}")
        return
    run(plan, args.plan)


if __name__ == "__main__":
    main()
