#!/bin/bash
# Usage: run_round.sh <config names...>  (runs sequentially; logs to logs/<name>.log)
cd "$(dirname "$0")/../../../.."
E=numerai/agents/experiments/nn_arch_ender20
for name in "$@"; do
  echo "=== $name $(date +%H:%M:%S)"
  PYTHONPATH=numerai .venv/bin/python -W ignore -m agents.code.modeling \
    --config $E/configs/$name.py --output-dir agents/experiments/nn_arch_ender20 \
    > $E/logs/$name.log 2>&1 || echo "FAILED $name"
done
echo "=== round done $(date +%H:%M:%S)"
