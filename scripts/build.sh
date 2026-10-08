#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MAKE=${MAKE:-make}
case "${1:-mt3000}" in
  clean)
    exec "$MAKE" -C "$ROOT/source/page_mv" veryclean
    ;;
  mt3000)
    ;;
  *)
    echo "Usage: bash scripts/build.sh [mt3000|clean]" >&2
    exit 2
    ;;
esac
DSP_ROOT=${DSP_ROOT:-${DEV_PRO_PATH:-}}
if [[ -z "$DSP_ROOT" ]]; then
  echo "Set DSP_ROOT to your MT-3000 SDK directory." >&2
  exit 2
fi
if [[ ! -d "$DSP_ROOT" ]]; then
  echo "SDK directory does not exist: $DSP_ROOT" >&2
  exit 2
fi
DSP_ROOT=$(cd -- "$DSP_ROOT" && pwd)
command -v "$MAKE" >/dev/null || { echo "GNU Make is unavailable." >&2; exit 2; }
command -v gcc >/dev/null || { echo "Host GCC is unavailable." >&2; exit 2; }
for item in dsp_compiler/bin/MT-3000-gcc dsp_compiler/bin/MT-3000-ld \
            dsp_compiler/bin/MT-3000-makedat hthreads/include/hthread_host.h \
            hthreads/include/hthread_device.h \
            hthreads/lib/libhthread_host.a hthreads/lib/libhthread_device.a \
            dsp_compiler/lib/vlib3000.a dsp_compiler/lib/slib3000.a; do
  if [[ ! -f "$DSP_ROOT/$item" ]]; then
    echo "Missing SDK component: $DSP_ROOT/$item" >&2
    exit 2
  fi
done
export DEV_PRO_PATH="$DSP_ROOT"
export PATH="$DSP_ROOT/dsp_compiler/bin:$PATH"
export LD_LIBRARY_PATH="$DSP_ROOT/hthreads/lib:$DSP_ROOT/dsp_compiler/lib:$DSP_ROOT/third-party-lib:${LD_LIBRARY_PATH:-}"
"$MAKE" -C "$ROOT/source/page_mv" clean DSP_ROOT="$DSP_ROOT"
"$MAKE" -C "$ROOT/source/page_mv" all DSP_ROOT="$DSP_ROOT" PAGE_FLAGS="${PAGE_FLAGS:-}"
if [[ ! -x "$ROOT/source/page_mv/bin/page" || ! -s "$ROOT/source/page_mv/bin/kernel.dat" ]]; then
  echo "Build did not produce page and kernel.dat." >&2
  exit 1
fi
echo "PAGE binaries: $ROOT/source/page_mv/bin"
