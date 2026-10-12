#!/bin/bash
# exp15 stages, in order. Usage: run_exp15.sh <stage> [<stage> ...]
#   external    day-0 check 9: our AO on WorkspaceBench's brew items (runs exp15_report.py)
#   dev_reads   dev capture, lens reads, probe/diffmeans, text reads (NLA, Oracle lens, AO)
#   ptag_check  day-0 check 7 (estimates the stand-in whitener on first run)
#   vectors     write directions (lens/template/diffmeans on CPU, then NLA AR and ptag AR on GPU)
#   dev_swaps   P3 selection grid on dev
#   test_reads  test bank (needs the locked manifest), capture, lens reads, probe/diffmeans, text reads
#   test_writes P3 on test at the locked settings, P4 on test (and on dev as a check)
# Every step skips work already on disk; logs go to log/exp15/<step>.log.
set -euo pipefail
cd "$(dirname "$0")"
PY=/venv/main/bin/python
LOG=../log/exp15
CACHE=../cache/exp15
RESULTS=../results/exp15_wrong_keys
mkdir -p "$LOG"

step() {            # step <log name> <python args...>
  local name=$1; shift
  echo "[$(date -u +%H:%M:%S)] $name"
  $PY exp15.py "$@" >> "$LOG/$name.log" 2>&1
}

unless() {          # unless <path> <log name> <python args...>: run the step only if <path> is absent
  local path=$1; shift
  if [ -e "$path" ]; then echo "skip $2 ($path exists)"; else step "$@"; fi
}

reads() {           # reads <bank> <tag>
  local bank=$1 tag=$2
  step "capture_$tag" capture --bank "$bank"
  unless "$CACHE/lens_reads_$bank.pt" "lens_reads_$tag" lens_reads --bank "$bank"
  unless "$CACHE/probe_reads_$bank.pt" "trained_reads_$tag" trained_reads --bank "$bank"
  for reader in nla olens ao; do
    step "text_reads_${tag}_$reader" text_reads "$reader" --bank "$bank"
  done
}

for stage in "$@"; do
  case $stage in
    external)
      if [ -e "$RESULTS/checks_day0/workspacebench.json" ]; then echo "skip external"
      else echo "[$(date -u +%H:%M:%S)] check_workspacebench"
           $PY exp15_report.py check_workspacebench >> "$LOG/check_workspacebench.log" 2>&1; fi ;;
    dev_reads) reads single_test dev ;;
    ptag_check) unless "$RESULTS/checks_day0/ptag.json" check_ptag check ptag ;;
    vectors)
      CUDA_VISIBLE_DEVICES="" step vectors_cpu vectors --parts lens diffmeans
      step vectors_gpu vectors --parts nla_ar ptag_ar ;;
    dev_swaps) step swaps_dev swaps --bank single_test ;;
    test_reads)
      unless "$CACHE/items_test.jsonl" bank_test bank
      unless "$CACHE/requests_test.jsonl" requests_test requests
      reads test test ;;
    test_writes)
      step swaps_test swaps --bank test
      step repair_test repair --bank test
      step repair_dev repair --bank single_test ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
echo "[$(date -u +%H:%M:%S)] done: $*"
