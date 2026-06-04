#!/usr/bin/env bash
set -u

cd "$HOME/vdaf-batched"

shards=20
for idx in $(seq 0 $((shards - 1))); do
  idx_padded=$(printf "%02d" "$idx")
  shards_padded=$(printf "%02d" "$shards")
  ts="cloudlab-shard${idx_padded}-of${shards_padded}-t1-safechunks"
  log="bench/results/full-${ts}.log"
  pid="bench/results/full-${ts}.pid"
  rm -f "$log" "$pid"

  nohup bash -lc '
set -u
cd "$HOME/vdaf-batched"
export PYTHONUNBUFFERED=1
idx="$1"
shards="$2"
ts="$3"
cmd=(
  "$HOME/vdaf-batched-venv/bin/python"
  "bench/run_bench.py"
  "--preset" "full"
  "--workers" "1"
  "--trials" "1"
  "--warmups" "0"
  "--shard-count" "$shards"
  "--shard-index" "$idx"
  "--timestamp" "$ts"
)
echo START_UTC "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo HOST "$(hostname)"
echo SHARD "$idx/$shards"
echo COMMAND "${cmd[*]}"
start=$(date +%s)
/usr/bin/time -f "WALL_CLOCK_SECONDS %e" "${cmd[@]}"
status=$?
end=$(date +%s)
echo END_UTC "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo EXIT_STATUS "$status"
echo WALL_CLOCK_SHELL_SECONDS "$((end - start))"
exit "$status"
' shard-runner "$idx" "$shards" "$ts" > "$log" 2>&1 &

  echo $! > "$pid"
  echo "$idx $! $log"
done
