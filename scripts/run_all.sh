#!/usr/bin/env bash
# Full experiment matrix. Usage:  bash scripts/run_all.sh [THREADS] [REPS]
#   THREADS : comma list, e.g. "1,2,4,8" (default 1,2,4,8 - edit to match YOUR machine's cores)
#   REPS    : repetitions per configuration (default 5)
# Close other programs, plug in the laptop, and do not touch the machine while this runs.
set -euo pipefail
cd "$(dirname "$0")/.."
THREADS=${1:-1,2,4,8}
REPS=${2:-5}
GRAINS=1,2,4,8,16,32,64,128,256,1024,4096
SCHEDS=static,static1,dynamic,guided,task
OUT=data/results.csv
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false

mkdir -p data plots
g++ -O2 -std=c++17 -fopenmp src/bench.cpp -o bench
lscpu > data/machine_info.txt 2>&1 || true
g++ --version | head -1 >> data/machine_info.txt
[ -f "$OUT" ] && mv "$OUT" "data/results_old_$(date +%s).csv"

# --- 1. Synthetic: 3 problem sizes x variance levels (low/medium/high) + order sensitivity
for SIZE in 10000 40000 160000; do
  for CV in 0 0.25 1.0 2.0; do          # 0 & 0.25 = low, 1.0 = medium, 2.0 = high
    ./bench --workload synthetic --size $SIZE --cv $CV --mean 10000 --order random \
            --threads $THREADS --grains $GRAINS --scheds $SCHEDS --reps $REPS --out $OUT
  done
done
# adversarial ordering (heavy tasks clustered at the end) at the largest size
for CV in 1.0 2.0; do
  ./bench --workload synthetic --size 40000 --cv $CV --mean 10000 --order sorted \
          --threads $THREADS --grains $GRAINS --scheds $SCHEDS --reps $REPS --out $OUT
done

# --- 2. Real irregular kernels, 3 sizes each
for N in 20000 80000 320000; do
  ./bench --workload prime --size $N --base 1000000000 \
          --threads $THREADS --grains $GRAINS --scheds $SCHEDS --reps $REPS --out $OUT
done
for W in 300 600 1200; do
  ./bench --workload mandel --size $W --maxiter 2000 \
          --threads $THREADS --grains $GRAINS --scheds $SCHEDS --reps $REPS --out $OUT
done
echo "Done -> $OUT   (next: python3 scripts/plots.py)"
