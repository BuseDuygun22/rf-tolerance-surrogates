#!/bin/bash
# Full-curve route, all stages. Threads are limited per process so the jobs share the 20 cores.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
run() { name=$1; th=$2; shift 2; OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th python -W ignore -u run_fullcurve.py "$@" > fc_$name.log 2>&1; }

rm -f fc_model.pkl fc_lwo_*.json fc_oof_*.json fc_grid_*.json fc_stair.json
run train 6 train &
for k in 0 1 2 3 4; do run lwo$k 3 lwo $k & done
# wait for the production model, then the studies that need it
while [ ! -f fc_model.pkl ]; do sleep 10; done
sleep 5
run stair 3 stair &
for s in 0 1 2 3; do run grid$s 3 grid $s & done
wait
for k in 0 1 2 3 4; do run oof$k 3 oof $k & done
wait
echo FC_ALL_DONE > fc_done.flag
