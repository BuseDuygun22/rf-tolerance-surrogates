#!/bin/bash
# Resumes the full-curve route after the 2026-09-21 out-of-memory crash.
# train and lwo0-4 already finished (their json files exist) and are not rerun.
# stair and all 4 grid shards died with nothing written, so they start fresh.
# Fix: BATCH lowered 20,000 -> 4,000 in run_fullcurve.py, and grid shards now
# run 2 at a time (not 4) with a smaller thread count, so peak memory stays low.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
run() { name=$1; th=$2; shift 2; OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th python -W ignore -u run_fullcurve.py "$@" > fc_$name.log 2>&1; }

rm -f fc_stair.json fc_grid_*.json fc_stair.log fc_grid0.log fc_grid1.log fc_grid2.log fc_grid3.log

run stair 4 stair &
wait

run grid0 4 grid 0 &
run grid1 4 grid 1 &
wait
run grid2 4 grid 2 &
run grid3 4 grid 3 &
wait

for k in 0 1 2 3 4; do run oof$k 3 oof $k & done
wait

echo FC_ALL_DONE > fc_done.flag
