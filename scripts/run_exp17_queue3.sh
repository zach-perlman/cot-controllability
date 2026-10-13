#!/bin/bash
# exp17 test-bank runs locked by manifest_addendum_patch_and_gemma.json:
#   qwen patching, gemma_occupied patching (P1-P3 primary), then gemma_occupied's swap test (descriptive only, K1)
cd "$(dirname "$0")"
PY=/venv/main/bin/python
LOG=../log/exp17

step() {   # step <log name> <env assignments...> -- <command...>
    local name=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    echo "$(date -u +%FT%TZ) start $name: ${envs[*]} $*" >> $LOG/queue3.log
    env "${envs[@]}" "$@" >> "$LOG/$name.log" 2>&1
    echo "$(date -u +%FT%TZ) end $name: exit $?" >> $LOG/queue3.log
}

step qwen_patch_test EXP17_MODEL=qwen -- $PY exp17_patch.py test
step gemma_occupied_patch_test EXP17_MODEL=gemma EXP17_LENS=occupied -- $PY exp17_patch.py test
step gemma_occupied_test EXP17_MODEL=gemma EXP17_LENS=occupied -- $PY exp17.py test
echo "$(date -u +%FT%TZ) queue3 done" >> $LOG/queue3.log
