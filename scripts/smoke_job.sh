#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: smoke_job.sh <package_root> <SDK_root> [matrix.mtx]" >&2
  exit 2
fi
ROOT=$(cd -- "$1" && pwd)
DSP_ROOT=$(cd -- "$2" && pwd)
export DSP_ROOT
MATRIX=${3:-}
if [[ ! -f "$ROOT/scripts/build.sh" || ! -f "$ROOT/scripts/run.sh" ]]; then
  echo "Invalid PAGE package root: $ROOT" >&2
  exit 2
fi
if [[ -n "$MATRIX" ]]; then
  if [[ ! -f "$MATRIX" ]]; then
    echo "Matrix file does not exist: $MATRIX" >&2
    exit 2
  fi
  MATRIX=$(cd -- "$(dirname -- "$MATRIX")" && pwd)/$(basename -- "$MATRIX")
fi
command -v yhrun >/dev/null || { echo "yhrun is unavailable in this job." >&2; exit 2; }
HOST=$(hostname -s 2>/dev/null || hostname)
cd -- "$ROOT"
mkdir -p results
echo "PAGE smoke node: $HOST"
echo "PAGE package: $ROOT"
echo "MT-3000 SDK: $DSP_ROOT"
echo "Resources: 16 CPU threads / 24 DSP arrays / cluster 0"
while IFS= read -r name; do
  case "$name" in PAGE_*) unset "$name" ;; esac
done < <(compgen -e)
export OMP_PROC_BIND=close OMP_PLACES=cores SPMV_BW_PROBE=0
if [[ ! -x "$ROOT/source/page_mv/bin/page" || ! -s "$ROOT/source/page_mv/bin/kernel.dat" ]]; then
  echo "Build PAGE on an SDK-equipped Linux ARM host before submitting this job:" >&2
  echo "  DSP_ROOT=\"$DSP_ROOT\" bash \"$ROOT/scripts/build.sh\"" >&2
  exit 2
fi
run_case() {
  local matrix="$1" log="$2"
  yhrun -N 1 -n 1 -w "$HOST" --unbuffered \
    bash "$ROOT/scripts/run.sh" "$matrix" 24 0 2>&1 | tee "$log"
  if ! grep -Eq '^\[PAGE\] correct = 1,' "$log" || ! grep -q '^Right!$' "$log"; then
    echo "PAGE correctness check did not pass; see $log" >&2
    return 1
  fi
}
run_case "$ROOT/examples/tiny.mtx" "$ROOT/results/tiny.log"
if [[ -n "$MATRIX" ]]; then
  run_case "$MATRIX" "$ROOT/results/matrix.log"
fi
echo "PAGE smoke passed"
