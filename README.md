# Task granularity vs workload irregularity (OpenMP) - code package

## Requirements
Linux / WSL2 / macOS with g++ >= 9 and OpenMP (macOS: `brew install gcc`, use g++-14 instead of g++),
Python 3 with `pandas matplotlib numpy tabulate` (`pip install pandas matplotlib numpy tabulate`).

## Files
| Path | Purpose |
|---|---|
| src/bench.cpp | Benchmark: 3 workloads x 5 schedules x granularity sweep; writes raw CSV |
| scripts/run_all.sh | Full experiment matrix (3 sizes x variance levels, prime, Mandelbrot) |
| scripts/plots.py | Raw CSV -> 7 figures (PNG/SVG/PDF) + per-figure source CSVs + summary table |
| scripts/sim_model.py | Scheduling simulator (model prediction, NOT measured data) |
| data/results.csv | Raw data (created by you by running run_all.sh) |
| data/machine_info.txt | lscpu + compiler version (created by run_all.sh) |

## Quick start
```bash
nproc                                                  # how many cores do you have?
g++ -O2 -std=c++17 -fopenmp src/bench.cpp -o bench
# 1-minute smoke test (all checksum_ok must be 1):
OMP_PROC_BIND=close OMP_PLACES=cores ./bench --workload synthetic --size 4000 --cv 1.0 --mean 2000 \
     --threads 1,2 --grains 1,16,256 --scheds static,dynamic,guided,task --reps 2 --out data/smoke.csv
# full study (edit the thread list to your core count; ~1.5-3 h; run overnight):
bash scripts/run_all.sh 1,2,4,8 5
python3 scripts/plots.py                               # -> plots/*.png|svg|pdf + plots/data/*.csv
python3 scripts/sim_model.py --order sorted --outdir plots/model_sorted
```
Delete data/smoke.csv before submitting. Do not submit smoke-test numbers as results.

## bench.cpp options
`--workload synthetic|prime|mandel  --size N  --cv X  --mean units  --order random|sorted|reverse
 --threads 1,2,4  --grains 1,4,16  --scheds static,static1,dynamic,guided,task  --reps R  --out file.csv
 --base 1000000000 (prime)  --maxiter 2000 (mandel)  --seed S`

## CSV columns
workload,size,order,cv_target,cv_measured,max_over_mean,sched,grain,nchunks,threads,rep,t_serial_med,t_par,
speedup,efficiency,imbalance_pct,idle_pct,barrier_wait_pct,sched_overhead_pct,checksum_ok
- imbalance_pct = (max busy / mean busy - 1) x 100 over threads
- idle_pct = (1 - mean busy / T_par) x 100 = barrier_wait_pct + sched_overhead_pct
- barrier_wait_pct: time a thread waits for the slowest one (imbalance tail); sched_overhead_pct: time between
  chunks inside the runtime (dispatch, task creation/stealing, fork cost)
- speedup uses the median serial time of the same workload/size/CV/order.

## Design notes / known limitations
- busy time is measured with omp_get_wtime() around each chunk; for g=1 and tiny tasks the two timer calls add
  ~20-50 ns per chunk (counted as busy time). Our default task mean is ~10 us, so the distortion is <1%.
- `static` distributes contiguous blocks of chunks, so g only changes timer granularity there; `static1` is cyclic.
- `task` mode uses one producer thread creating one omp task per chunk - a deliberate (common) pattern whose
  creation bottleneck is part of what is measured.
- Checksums (XOR of per-task results) are compared with the serial run to catch wrong parallel results.
