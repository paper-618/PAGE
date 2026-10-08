#ifndef _PAGE_COMMON_H_
#define _PAGE_COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>

#ifndef MAT_VAL_TYPE
#define MAT_VAL_TYPE double
#endif
#ifndef MAT_PTR_TYPE
#define MAT_PTR_TYPE int
#endif
#ifndef MAT_IDX_TYPE
#define MAT_IDX_TYPE int
#endif

#define SROW              16
#define SIGMA             512
#define SLICE_PER_SIGMA   (SIGMA / SROW)

#define GSM_X_CAP         780000

typedef unsigned short page_idx16_t;
typedef unsigned int   page_idx32_t;
#define PAGE_IDX16_BYTES 2
#define PAGE_IDX32_BYTES 4
#define CHUNK16            4096
#define CHUNK32            (12 * 1024)
#define CHUNK_MAX          CHUNK32
#define CHUNK              CHUNK32

#define PAGE_TILE_CAP    GSM_X_CAP

/* Packed x blocks are 64-byte aligned in both DDR and GSM. */
#define PAGE_XBLOCK_BYTES 64
#define PAGE_XBLOCK_ELEMS (PAGE_XBLOCK_BYTES / 8)
#define PAGE_XBLOCK_CAP   (GSM_X_CAP / PAGE_XBLOCK_ELEMS)

#define NBUF              2
#define PAGE_TILE_PACKED  1

#define PANEL_SLICE       512
#define PANEL_ROWS        (PANEL_SLICE * SROW)

#define Z_RATIO_CPU       1.0e9

#define PAGE_ROWCAP_MULT 4.0
#define PAGE_ROWCAP_MIN  16

#define PAGE_ROWCAP_MAX  4096

#define PAGE_OV_RATIO_MAX 1.0
#define PAGE_ROWCAP_RETRY 4

/* Virtualize rows when G_pre = W_max_slice / (W_unsplit_stored / coreNum) > 1. */
#define PAGE_VROW_G_THRESHOLD 1.0

#define PAGE_XRELOAD_BUDGET 0.30
#define PAGE_PANEL_MIN_SLICE 128

#define PAGE_XIN_BUDGET  0.35

#define PAGE_CW_BLK       4096

#define PAGE_FINAL_COLUMN_WINDOW 0

#define PAGE_BW_DDR      30.0e9
#define PAGE_BW_GSM      26.0e9
#define PAGE_BW_ARM      12.0e9

#define PAGE_BW_PIPE     (PAGE_BW_DDR + PAGE_BW_GSM)

#define PAGE_BW_ARM_MIN  1.0e9
#define PAGE_BW_ARM_MAX  1.0e11
#define PAGE_KERNEL_FIX_MS 0.15
#define PAGE_PANEL_FIX_MS  0.005
#define PAGE_BARRIER_FIX_MS 0.0

#define PAGE_BW_MIX      PAGE_BW_DDR

#define C_FMA             1.0
#define C_SG              1.6
#define C_SLICE           14.0

enum page_mode {
    PAGE_MODE_A   = 0,
    PAGE_MODE_B   = 1,
    PAGE_MODE_C   = 2,
    PAGE_MODE_CPU = 3
};

static const char *page_mode_name(int mode)
{
    switch (mode) {
        case PAGE_MODE_A:   return "A-gsm-resident";
        case PAGE_MODE_B:   return "B-sliding-window";
        case PAGE_MODE_C:   return "C-tiled";
        case PAGE_MODE_CPU: return "CPU-overt";
        default:             return "unknown";
    }
}

enum page_value_mode {
    PAGE_VALUE_GENERAL  = 0,
    PAGE_VALUE_UNIT     = 1,
    PAGE_VALUE_CONSTANT = 2
};

typedef struct {
    MAT_VAL_TYPE *values;
    int          *colIndices;
    int          *rowPointers;
    int           numRows;
    int           numCols;
    int           numNonzeros;

    int           value_mode;
    MAT_VAL_TYPE  constant_value;
} CSRMatrix;

#ifdef PAGE_USE_HTHREAD
#include "hthread_host.h"
#define PAGE_DEV_MALLOC(cid, sz)  hthread_malloc((cid), (sz), HT_MEM_RW)
#define PAGE_DEV_FREE(p)          do { if (p) hthread_free(p); } while (0)
#else
#define PAGE_DEV_MALLOC(cid, sz)  malloc((size_t)(sz))
#define PAGE_DEV_FREE(p)          do { if (p) free(p); } while (0)
#endif

#define PAGE_REL_TOL     1e-12
#define PAGE_ABS_EPS     1e-300

#endif
