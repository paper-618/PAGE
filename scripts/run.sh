#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/run.sh <matrix.mtx> [DSP_arrays=24] [cluster_id=0]" >&2
  exit 2
fi
if [[ ! -f "$1" ]]; then
  echo "Matrix file does not exist: $1" >&2
  exit 2
fi
MATRIX=$(cd -- "$(dirname -- "$1")" && pwd)/$(basename -- "$1")
CORE=${2:-24}
CLUSTER=${3:-0}
if [[ ! "$CORE" =~ ^[1-9][0-9]*$ || ! "$CLUSTER" =~ ^[0-9]+$ ]]; then
  echo "DSP_arrays must be a positive integer; cluster_id must be nonnegative." >&2
  exit 2
fi
BIN="$ROOT/source/page_mv/bin"
if [[ ! -x "$BIN/page" || ! -f "$BIN/kernel.dat" ]]; then
  echo "Build PAGE first: DSP_ROOT=/path/to/sdk bash scripts/build.sh" >&2
  exit 2
fi
while IFS= read -r name; do
  case "$name" in PAGE_*) unset "$name" ;; esac
done < <(compgen -e)
source "$ROOT/config/final_env.sh"
export OMP_NUM_THREADS=16
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close OMP_PLACES=cores SPMV_BW_PROBE=0
if [[ -n "${DSP_ROOT:-${DEV_PRO_PATH:-}}" ]]; then
  SDK=${DSP_ROOT:-${DEV_PRO_PATH}}
  export LD_LIBRARY_PATH="$SDK/hthreads/lib:$SDK/dsp_compiler/lib:$SDK/third-party-lib:${LD_LIBRARY_PATH:-}"
fi
cd -- "$BIN"
exec ./page "$MATRIX" "$CORE" "$CLUSTER"
