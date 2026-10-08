# PAGE

**PAGE: Performance-Guided Assignment and Row Virtualization for CPU–DSP Collaborative SpMV**

## Abstract

Sparse matrix–vector multiplication (SpMV) is a fundamental kernel in sparse scientific computing, yet its irregular memory access and uneven workload distribution make efficient execution challenging. Heterogeneous many-core digital signal processors (DSPs) combine single-instruction, multiple data (SIMD) execution with high-bandwidth, software-managed memories, making them attractive for data-intensive scientific computing workloads. Fully exploiting such platforms clearly requires efficient CPU–DSP collaborative execution. However, compared with CPU–GPU systems, general CPU–DSP collaborative SpMV remains relatively underexplored, with prior DSP-oriented work largely focusing on DSP-side execution or structured matrices. We find that HALAV, a recent state-of-the-art method for general CPU–DSP collaborative SpMV, still exposes coarse execution granularity at two levels: memory-constrained blocking quantizes CPU–DSP ownership, while oversized rows remain coarse scheduler-visible work units within the DSP. We therefore propose PAGE, which combines plan-aware, runtime-calibrated CPU–DSP assignment with selective Row Virtualization for oversized DSP rows, while preserving matrix semantics and the existing cross-row SIMD path. On the MT-3000 platform, PAGE achieves 2.98× and 2.18× speedups over CSR and HALAV, respectively, on the 15 canonical matrices, and 2.27× over HALAV on the broader 337-matrix evaluation. These results demonstrate the effectiveness of jointly refining inter-device and intra-DSP execution granularity for CPU–DSP collaborative SpMV.

**Index Terms—** SpMV, Heterogeneous many-core DSP, CPU–DSP assignment, Task granularity, Load balance

## Overview and requirements

PAGE performs CPU–DSP collaborative SpMV on MT-3000, with matrix preprocessing, runtime calibration, CPU computation, DSP kernels, and numerical validation. Defaults: **16 CPU threads, 24 DSP arrays, cluster 0**. Hybrid execution requires a 512-row-aligned split with nonzero work on both devices; the bundled 1024-by-1024 example satisfies this requirement.

Build on an SDK-equipped Linux ARM host with GCC/OpenMP, GNU Make, and Bash. Use the login node if your cluster supports compilation there. Running requires MT-3000 DSP compute nodes, SDK runtime libraries, and `yhbatch`/`yhrun`. The source, SDK, and matrix directories must be accessible from compute nodes.

## 1. Download

```bash
git clone https://github.com/paper-618/PAGE.git
cd PAGE
```

Alternatively, select **Code → Download ZIP** on the [repository](https://github.com/paper-618/PAGE) and extract it. Run all commands below from the source root containing `source/` and `scripts/`.

## 2. Configure and build

Change these values for your environment:

| Setting | Value to use |
| --- | --- |
| `PACKAGE_ROOT` | Source root; `$PWD` sets it automatically when run from that directory |
| `DSP_ROOT` | Installed MT-3000 SDK root containing `dsp_compiler/` and `hthreads/` |
| `MATRIX` | Absolute path to your Matrix Market `.mtx` file |
| `PARTITION` | Cluster partition providing MT-3000 DSP nodes |

```bash
export PACKAGE_ROOT="$PWD"
export DSP_ROOT=/path/to/mt3000_sdk
export DEV_PRO_PATH="$DSP_ROOT"
export MATRIX=/path/to/matrix.mtx
export PARTITION=YOUR_PARTITION

command -v make gcc yhbatch yhrun
mkdir -p results
set -o pipefail
bash scripts/build.sh 2>&1 | tee results/build.log
```

Continue only if the checks and build succeed. The build produces `source/page_mv/bin/page` and `source/page_mv/bin/kernel.dat`. If Make or GCC is missing, load the cluster's compiler environment before building.

## 3. Submit the run job

```bash
chmod u+x scripts/smoke_job.sh

yhbatch \
  --partition="$PARTITION" \
  --nodes=1 --ntasks=1 --exclusive \
  --time=01:00:00 \
  --job-name=PAGE-check \
  --output="$PACKAGE_ROOT/results/smoke-job.out" \
  --error="$PACKAGE_ROOT/results/smoke-job.err" \
  --export=ALL \
  "$PACKAGE_ROOT/scripts/smoke_job.sh" \
  "$PACKAGE_ROOT" "$DSP_ROOT" "$MATRIX"
```

The job runs `examples/tiny.mtx` first, then your matrix, using the compiled binaries. To test only the bundled example, omit the final `"$MATRIX"` argument; no external matrix is needed. Compute nodes do not need Make.

Wait for the job to finish. Adjust `--time` for matrix size and cluster limits. Do not rebuild while jobs using the same binaries are queued or running. Repeated jobs overwrite the same logs; save results before rerunning if needed.

## 4. Check results

```bash
tail -n 80 results/smoke-job.out
tail -n 80 results/smoke-job.err
grep -E '^\[PAGE\] (correct|spmv_time|speedup)|^Right!$' \
  results/tiny.log results/matrix.log
```

A successful job ends with `PAGE smoke passed`. Each matrix log must contain:

```text
[PAGE] correct = 1, max_rel_err = ...
Right!
```

When running only the bundled example, check `results/tiny.log` and omit `results/matrix.log` from the command above. Build output is in `results/build.log`. If the job fails, inspect that file and the job and matrix logs.

`spmv_time` is in milliseconds; `speedup` is relative to the program's CPU CSR reference path. Timing uses 5 warm-up iterations and 10 measurements, discards the minimum and maximum, and averages the remaining 8. Use real matrices for performance measurements.

## Source layout

```text
config/                     Runtime configuration
examples/                   Example matrix
scripts/                    Build, run, and job scripts
source/header/              Matrix I/O, CSR, formats, and preprocessing
source/page_mv/host_code/    CPU–DSP host program
source/page_mv/device_code/  DSP kernels and calibration
```

Matrix Market I/O references [NIST Matrix Market](http://math.nist.gov/MatrixMarket). CPU reference computation uses the HaLAV CSR path. The MT-3000 SDK supplies the DSP toolchain, hthreads, vector libraries, and math libraries.
