// bench.cpp - Granularity vs workload-irregularity benchmark (C++17 + OpenMP)
//
// Workloads : synthetic (lognormal task costs, controllable CV), prime (trial-division
//             prime search), mandel (Mandelbrot pixels)
// Schedules : static (block), static1 (cyclic), dynamic, guided, task (omp task per chunk)
// Granularity g = number of consecutive tasks grouped into one schedulable chunk.
//
// Build  : g++ -O2 -std=c++17 -fopenmp src/bench.cpp -o bench
// Example: OMP_PROC_BIND=close OMP_PLACES=cores ./bench --workload synthetic --size 40000
//          --cv 1.0 --threads 1,2,4,8 --grains 1,4,16,64,256 --scheds static,dynamic,guided,task
//          --reps 5 --out data/results.csv
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
typedef unsigned long long u64;
typedef unsigned int u32;

struct Workload {
  string name;
  size_t n = 0;             // number of tasks
  vector<u32> units;        // synthetic: cost of each task (LCG iterations)
  u64 base = 1000000000ULL; // prime: first candidate
  int W = 0;                // mandel: image width = height
  int maxiter = 2000;       // mandel: iteration cap
};

static inline u64 lcg_work(u64 units, u64 seed) {
  u64 x = seed;
  for (u64 k = 0; k < units; k++) x = x * 6364136223846793005ULL + 1442695040888963407ULL;
  return x;
}

// Executes task i; returns a checksum contribution; if wk != nullptr stores the work done.
static inline u64 run_task(const Workload& w, size_t i, u32* wk) {
  if (w.name == "synthetic") {
    u32 u = w.units[i];
    if (wk) *wk = u;
    return lcg_work(u, i + 1);
  } else if (w.name == "prime") {
    u64 n = w.base + i;
    if ((n & 1ULL) == 0) { if (wk) *wk = 1; return 0; }
    u32 cnt = 1;
    for (u64 d = 3; d * d <= n; d += 2) {
      cnt++;
      if (n % d == 0) { if (wk) *wk = cnt; return 0; }
    }
    if (wk) *wk = cnt;
    return 1;  // prime
  } else {  // mandel
    int px = (int)(i % w.W), py = (int)(i / w.W);
    double cr = -2.0 + 3.0 * px / w.W, ci = -1.5 + 3.0 * py / w.W;
    double zr = 0, zi = 0;
    int it = 0;
    while (it < w.maxiter && zr * zr + zi * zi <= 4.0) {
      double t = zr * zr - zi * zi + cr;
      zi = 2.0 * zr * zi + ci;
      zr = t;
      it++;
    }
    if (wk) *wk = (u32)it;
    return (u64)it;
  }
}

struct RunResult {
  double tpar = 0, imbalance = 0, idle = 0, barrier = 0, sched = 0;
  u64 chk = 0;
};

// One timed parallel execution. Per-thread busy time = time spent inside chunk bodies.
// fin[t] = time (since region start) at which thread t ran out of work.
//   idle          = Tpar - busy           (all non-compute time)
//   barrier_wait  = Tpar - fin            (waiting for the slowest thread = imbalance tail)
//   sched_overhead= fin  - busy           (time inside the runtime fetching/creating chunks)
static RunResult run_parallel(const Workload& w, int P, const string& sched, size_t g) {
  const size_t n = w.n, nch = (n + g - 1) / g;
  vector<double> busy(P * 8, 0.0), fin(P * 8, 0.0);  // stride 8 avoids false sharing
  vector<u64> chk(P * 8, 0);
  double t0 = 0, t1 = 0;

  if (sched == "task") {
#pragma omp parallel num_threads(P)
    {
#pragma omp barrier
#pragma omp single
      {
        t0 = omp_get_wtime();
        for (size_t c = 0; c < nch; c++) {
#pragma omp task firstprivate(c)
          {
            int tid = omp_get_thread_num();
            double a = omp_get_wtime();
            size_t lo = c * g, hi = min(n, lo + g);
            u64 s = 0;
            for (size_t i = lo; i < hi; i++) s ^= run_task(w, i, nullptr);
            double b = omp_get_wtime();
            busy[tid * 8] += b - a;
            fin[tid * 8] = b - t0;
            chk[tid * 8] ^= s;
          }
        }
      }  // implicit barrier: all tasks complete
#pragma omp barrier
#pragma omp master
      t1 = omp_get_wtime();
    }
  } else {
    omp_sched_t kind = omp_sched_static;
    int chunk = 0;
    if (sched == "static") { kind = omp_sched_static; chunk = 0; }
    else if (sched == "static1") { kind = omp_sched_static; chunk = 1; }
    else if (sched == "dynamic") { kind = omp_sched_dynamic; chunk = 1; }
    else if (sched == "guided") { kind = omp_sched_guided; chunk = 1; }
    else { fprintf(stderr, "unknown schedule %s\n", sched.c_str()); exit(1); }
    omp_set_schedule(kind, chunk);
#pragma omp parallel num_threads(P)
    {
      int tid = omp_get_thread_num();
#pragma omp barrier
#pragma omp single
      { t0 = omp_get_wtime(); }
      double b = 0;
      u64 s = 0;
#pragma omp for schedule(runtime) nowait
      for (long c = 0; c < (long)nch; c++) {
        double a = omp_get_wtime();
        size_t lo = (size_t)c * g, hi = min(n, lo + g);
        for (size_t i = lo; i < hi; i++) s ^= run_task(w, i, nullptr);
        b += omp_get_wtime() - a;
      }
      fin[tid * 8] = omp_get_wtime() - t0;
      busy[tid * 8] = b;
      chk[tid * 8] = s;
#pragma omp barrier
#pragma omp master
      t1 = omp_get_wtime();
    }
  }

  RunResult r;
  r.tpar = t1 - t0;
  double sb = 0, mb = 0, sbar = 0, ssch = 0;
  for (int p = 0; p < P; p++) {
    double b = busy[p * 8], f = fin[p * 8];
    sb += b; mb = max(mb, b);
    sbar += r.tpar - f;
    ssch += max(0.0, f - b);
    r.chk ^= chk[p * 8];
  }
  double meanb = sb / P;
  r.imbalance = meanb > 0 ? (mb / meanb - 1.0) * 100.0 : 0.0;
  r.idle = (1.0 - meanb / r.tpar) * 100.0;
  r.barrier = (sbar / P) / r.tpar * 100.0;
  r.sched = (ssch / P) / r.tpar * 100.0;
  return r;
}

static double run_serial(const Workload& w, u64& chk) {
  double a = omp_get_wtime();
  u64 s = 0;
  for (size_t i = 0; i < w.n; i++) s ^= run_task(w, i, nullptr);
  double b = omp_get_wtime();
  chk = s;
  return b - a;
}

static void warmup(int P) {
  volatile u64 sink = 0;
#pragma omp parallel num_threads(P)
  { u64 x = lcg_work(2000000, omp_get_thread_num() + 1); if (x == 42) sink = x; }
  (void)sink;
}

static vector<string> split(const string& s) {
  vector<string> out; stringstream ss(s); string t;
  while (getline(ss, t, ',')) if (!t.empty()) out.push_back(t);
  return out;
}
static double median(vector<double> v) {
  sort(v.begin(), v.end());
  size_t m = v.size();
  return m % 2 ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
}

int main(int argc, char** argv) {
  Workload w; w.name = "synthetic";
  size_t size = 40000; double cv = 1.0, mean_units = 10000; string order = "random";
  string threads = "1,2,4", grains = "1,4,16,64,256", scheds = "static,dynamic,guided,task";
  string out = "results.csv"; int reps = 5; unsigned seed = 12345;
  for (int i = 1; i < argc; i++) {
    string a = argv[i];
    auto nx = [&]() -> string {
      if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); }
      return argv[++i];
    };
    if (a == "--workload") w.name = nx();
    else if (a == "--size") size = stoull(nx());
    else if (a == "--cv") cv = stod(nx());
    else if (a == "--mean") mean_units = stod(nx());
    else if (a == "--order") order = nx();
    else if (a == "--threads") threads = nx();
    else if (a == "--grains") grains = nx();
    else if (a == "--scheds") scheds = nx();
    else if (a == "--reps") reps = stoi(nx());
    else if (a == "--out") out = nx();
    else if (a == "--seed") seed = (unsigned)stoul(nx());
    else if (a == "--base") w.base = stoull(nx());
    else if (a == "--maxiter") w.maxiter = stoi(nx());
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
  }

  // ---- build workload ----
  if (w.name == "synthetic") {
    w.n = size; w.units.resize(size);
    mt19937_64 rng(seed);
    if (cv <= 0) fill(w.units.begin(), w.units.end(), (u32)mean_units);
    else {
      double s2 = log(1.0 + cv * cv), mu = log(mean_units) - s2 / 2.0;
      lognormal_distribution<double> d(mu, sqrt(s2));
      for (auto& u : w.units) u = (u32)max(1.0, round(d(rng)));
    }
    if (order == "sorted") sort(w.units.begin(), w.units.end());
    else if (order == "reverse") sort(w.units.begin(), w.units.end(), greater<u32>());
  } else if (w.name == "prime") { w.n = size; order = "natural"; cv = -1; }
  else if (w.name == "mandel") { w.W = (int)size; w.n = (size_t)size * size; order = "natural"; cv = -1; }
  else { fprintf(stderr, "unknown workload\n"); return 1; }

  // ---- cost statistics (serial pass, untimed) ----
  double mean = 0, m2 = 0, mx = 0;
  for (size_t i = 0; i < w.n; i++) {
    u32 wk = 0; run_task(w, i, &wk);
    double x = wk, d = x - mean; mean += d / (i + 1); m2 += d * (x - mean); mx = max(mx, x);
  }
  double cv_meas = mean > 0 ? sqrt(m2 / w.n) / mean : 0, max_over_mean = mean > 0 ? mx / mean : 0;

  bool newfile = !ifstream(out).good();
  FILE* f = fopen(out.c_str(), "a");
  if (!f) { perror("open output"); return 1; }
  if (newfile)
    fprintf(f, "workload,size,order,cv_target,cv_measured,max_over_mean,sched,grain,nchunks,threads,rep,"
               "t_serial_med,t_par,speedup,efficiency,imbalance_pct,idle_pct,barrier_wait_pct,sched_overhead_pct,checksum_ok\n");

  // ---- serial baseline ----
  vector<double> ts; u64 ref = 0;
  for (int r = 0; r < reps; r++) { u64 c; ts.push_back(run_serial(w, c)); ref = c; }
  double tser = median(ts);
  for (int r = 0; r < reps; r++)
    fprintf(f, "%s,%zu,%s,%g,%.4f,%.3f,serial,0,0,1,%d,%.6f,%.6f,1,1,0,0,0,0,1\n", w.name.c_str(), size,
            order.c_str(), cv, cv_meas, max_over_mean, r, tser, ts[r]);
  fprintf(stderr, "[%s n=%zu cv_target=%g cv_meas=%.3f max/mean=%.1f] serial median = %.4f s\n", w.name.c_str(),
          size, cv, cv_meas, max_over_mean, tser);

  // ---- parallel sweep ----
  for (auto& pt : split(threads)) {
    int P = stoi(pt); warmup(P);
    for (auto& sc : split(scheds))
      for (auto& gt : split(grains)) {
        size_t g = stoull(gt);
        for (int r = 0; r < reps; r++) {
          RunResult x = run_parallel(w, P, sc, g);
          double sp = tser / x.tpar;
          fprintf(f, "%s,%zu,%s,%g,%.4f,%.3f,%s,%zu,%zu,%d,%d,%.6f,%.6f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%d\n",
                  w.name.c_str(), size, order.c_str(), cv, cv_meas, max_over_mean, sc.c_str(), g,
                  (w.n + g - 1) / g, P, r, tser, x.tpar, sp, sp / P, x.imbalance, x.idle, x.barrier, x.sched,
                  x.chk == ref ? 1 : 0);
        }
        fflush(f);
      }
  }
  fclose(f);
  return 0;
}
