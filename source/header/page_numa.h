#ifndef _PAGE_NUMA_H_
#define _PAGE_NUMA_H_

#include <stdlib.h>
#include <omp.h>
#include "common.h"

/* First-touch CSR pages with the consuming OpenMP team; return 0 on failure. */
static inline int page_parallel_first_touch_csr(CSRMatrix *csr)
{
    if (!csr || !csr->rowPointers || !csr->colIndices || !csr->values ||
        csr->numRows < 0 || csr->numNonzeros < 0) return 0;

    const int m = csr->numRows;
    const int nnz = csr->numNonzeros;
    int *rp2 = (int *)malloc((size_t)(m + 1) * sizeof(int));
    int *ci2 = (int *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
    MAT_VAL_TYPE *va2 = (MAT_VAL_TYPE *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(MAT_VAL_TYPE));
    if (!rp2 || !ci2 || !va2) {
        free(rp2); free(ci2); free(va2);
        return 0;
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i <= m; ++i) rp2[i] = csr->rowPointers[i];

#pragma omp parallel for schedule(static)
    for (int j = 0; j < nnz; ++j) {
        ci2[j] = csr->colIndices[j];
        va2[j] = csr->values[j];
    }

    free(csr->rowPointers);
    free(csr->colIndices);
    free(csr->values);
    csr->rowPointers = rp2;
    csr->colIndices = ci2;
    csr->values = va2;
    return 1;
}

static inline void page_parallel_first_touch_x(MAT_VAL_TYPE *dst,
                                                const MAT_VAL_TYPE *src,
                                                int n)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) dst[i] = src[i];
}

#endif
