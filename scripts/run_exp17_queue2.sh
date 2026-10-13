#!/bin/bash
# exp17, after the Qwen locked test: dev-only follow-ups (nothing here touches test items).
#   qwen:            J++ answer swap at alpha 1-8 in the early dev bands; interchange patching on dev
#   gemma_occupied:  the occupied-subspace J++ variant: prep, chat-template positive control, dev grid, selection,
#                    preview, in-task control (late and early bands), interchange patching on dev
# Usage: bash run_exp17_queue2.sh <pid to wait for>
cd "$(dirname "$0")"
PY=/venv/main/bin/python
LOG=../log/exp17
wait_pid=$1
while [ -n "$wait_pid" ] && kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done

step() {   # step <log name> <env assignments...> -- <command...>
    local name=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    echo "$(date -u +%FT%TZ) start $name: ${envs[*]} $*" >> $LOG/queue2.log
    env "${envs[@]}" "$@" >> "$LOG/$name.log" 2>&1
    echo "$(date -u +%FT%TZ) end $name: exit $?" >> $LOG/queue2.log
}

step qwen_answer_early EXP17_MODEL=qwen -- $PY exp17_late.py early
step qwen_patch EXP17_MODEL=qwen -- $PY exp17_patch.py dev
step gemma_occupied EXP17_MODEL=gemma EXP17_LENS=occupied -- $PY exp17.py prep posctl dev select preview
step gemma_occupied_late EXP17_MODEL=gemma EXP17_LENS=occupied -- $PY exp17_late.py late
step gemma_occupied_early EXP17_MODEL=gemma EXP17_LENS=occupied -- $PY exp17_late.py early
step gemma_occupied_patch EXP17_MODEL=gemma EXP17_LENS=occupied -- $PY exp17_patch.py dev
echo "$(date -u +%FT%TZ) queue2 done" >> $LOG/queue2.log
