#ifndef _PAGE_CPU_H_
#define _PAGE_CPU_H_

#include <omp.h>
#include "common.h"
#include "page_format.h"

static inline void page_row_chunk(const int *rp, int r0, int r1,
                                   int nth, int tid, int *lo_out, int *hi_out)
{
    if (nth <= 1) { *lo_out = r0; *hi_out = r1; return; }
    const long long base = rp[r0];
    const long long tot  = (long long)rp[r1] - base;
    const long long a = base + tot * (long long)tid / nth;
    const long long b = base + tot * (long long)(tid + 1) / nth;
    int l = r0, h = r1;
    while (l < h) { const int mid = (l + h) >> 1; if ((long long)rp[mid] < a) l = mid + 1; else h = mid; }
    const int lo = (tid == 0) ? r0 : l;
    l = lo; h = r1;
    while (l < h) { const int mid = (l + h) >> 1; if ((long long)rp[mid] < b) l = mid + 1; else h = mid; }
    *lo_out = lo;
    *hi_out = (tid == nth - 1) ? r1 : l;
}

static inline void page_csr_rows_kernel(const int *rp, const int *ci,
                                         const MAT_VAL_TYPE *va,
                                         const MAT_VAL_TYPE *x, MAT_VAL_TYPE *y,
                                         int lo, int hi, int y_off)
{
    for (int i = lo; i < hi; i++) {
        const int b = rp[i], e = rp[i + 1];
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int j = b;
        for (; j + 3 < e; j += 4) {
            s0 += va[j]     * x[ci[j]];
            s1 += va[j + 1] * x[ci[j + 1]];
            s2 += va[j + 2] * x[ci[j + 2]];
            s3 += va[j + 3] * x[ci[j + 3]];
        }
        for (; j < e; j++) s0 += va[j] * x[ci[j]];
        y[i - y_off] = (s0 + s1) + (s2 + s3);
    }
}

static inline void page_csr_rows_kernel_unit(const int *rp, const int *ci,
                                              const MAT_VAL_TYPE *x, MAT_VAL_TYPE *y,
                                              int lo, int hi, int y_off)
{
    for (int i = lo; i < hi; i++) {
        const int b = rp[i], e = rp[i + 1];
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int j = b;
        for (; j + 3 < e; j += 4) {
            s0 += x[ci[j]];
            s1 += x[ci[j + 1]];
            s2 += x[ci[j + 2]];
            s3 += x[ci[j + 3]];
        }
        for (; j < e; j++) s0 += x[ci[j]];
        y[i - y_off] = (s0 + s1) + (s2 + s3);
    }
}

static inline void page_csr_rows_kernel_constant(const int *rp, const int *ci,
                                                  MAT_VAL_TYPE c,
                                                  const MAT_VAL_TYPE *x, MAT_VAL_TYPE *y,
                                                  int lo, int hi, int y_off)
{
    for (int i = lo; i < hi; i++) {
        const int b = rp[i], e = rp[i + 1];
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int j = b;
        for (; j + 3 < e; j += 4) {
            s0 += c * x[ci[j]];
            s1 += c * x[ci[j + 1]];
            s2 += c * x[ci[j + 2]];
            s3 += c * x[ci[j + 3]];
        }
        for (; j < e; j++) s0 += c * x[ci[j]];
        y[i - y_off] = (s0 + s1) + (s2 + s3);
    }
}

static inline void page_csr_rows_dispatch(const int *rp, const int *ci,
                                           const MAT_VAL_TYPE *va,
                                           int value_mode, MAT_VAL_TYPE constant_value,
                                           const MAT_VAL_TYPE *x, MAT_VAL_TYPE *y,
                                           int lo, int hi, int y_off)
{
    if (value_mode == PAGE_VALUE_UNIT) {
        page_csr_rows_kernel_unit(rp, ci, x, y, lo, hi, y_off);
    } else if (value_mode == PAGE_VALUE_CONSTANT) {
        page_csr_rows_kernel_constant(rp, ci, constant_value, x, y, lo, hi, y_off);
    } else {
        page_csr_rows_kernel(rp, ci, va, x, y, lo, hi, y_off);
    }
}

static void page_cpu_spmv_overt_general(const CSRMatrix *csr,
                                         const MAT_VAL_TYPE *x, MAT_VAL_TYPE *y)
{
#pragma omp parallel
    {
        int lo, hi;
        page_row_chunk(csr->rowPointers, 0, csr->numRows,
                        omp_get_num_threads(), omp_get_thread_num(), &lo, &hi);
        page_csr_rows_kernel(csr->rowPointers, csr->colIndices, csr->values,
                              x, y, lo, hi, 0);
    }
}

static void page_cpu_spmv_overt(const CSRMatrix *csr,
                                 const MAT_VAL_TYPE *x, MAT_VAL_TYPE *y)
{
#pragma omp parallel
    {
        int lo, hi;
        page_row_chunk(csr->rowPointers, 0, csr->numRows,
                        omp_get_num_threads(), omp_get_thread_num(), &lo, &hi);
        page_csr_rows_dispatch(csr->rowPointers, csr->colIndices, csr->values,
                                csr->value_mode, csr->constant_value,
                                x, y, lo, hi, 0);
    }
}

static void page_cpu_spmv_rows_general(const CSRMatrix *csr, const MAT_VAL_TYPE *x,
                                        MAT_VAL_TYPE *y, int r0, int r1)
{
    if (r1 <= r0) return;
#pragma omp parallel
    {
        int lo, hi;
        page_row_chunk(csr->rowPointers, r0, r1,
                        omp_get_num_threads(), omp_get_thread_num(), &lo, &hi);
        page_csr_rows_kernel(csr->rowPointers, csr->colIndices, csr->values,
                              x, y, lo, hi, 0);
    }
}

static void page_cpu_spmv_rows(const CSRMatrix *csr, const MAT_VAL_TYPE *x,
                                MAT_VAL_TYPE *y, int r0, int r1)
{
    if (r1 <= r0) return;
#pragma omp parallel
    {
        int lo, hi;
        page_row_chunk(csr->rowPointers, r0, r1,
                        omp_get_num_threads(), omp_get_thread_num(), &lo, &hi);
        page_csr_rows_dispatch(csr->rowPointers, csr->colIndices, csr->values,
                                csr->value_mode, csr->constant_value,
                                x, y, lo, hi, 0);
    }
}

static void page_cpu_spmv_residual(const page_matrix *A,
                                    const MAT_VAL_TYPE *x, MAT_VAL_TYPE *y_res)
{
    if (!A->has_window || A->res_nnz <= 0) return;
#pragma omp parallel
    {
        int lo, hi;
        page_row_chunk(A->res_rp, 0, A->m_dsp,
                        omp_get_num_threads(), omp_get_thread_num(), &lo, &hi);
        page_csr_rows_dispatch(A->res_rp, A->res_ci, A->res_val,
                                A->value_mode, A->constant_value,
                                x, y_res, lo, hi, 0);
    }
}

static void page_add_residual(const page_matrix *A,
                               const MAT_VAL_TYPE *y_res, MAT_VAL_TYPE *y)
{
    if (!A->has_window || A->res_nnz <= 0) return;
    const int off = A->row_off, md = A->m_dsp;
    const int *rp = A->res_rp;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < md; i++)
        if (rp[i + 1] > rp[i]) y[off + i] += y_res[i];
}

static void page_finalize_y(const page_matrix *A,
                             MAT_VAL_TYPE *yd,
                             const MAT_VAL_TYPE *y_res,
                             MAT_VAL_TYPE *y)
{
    const int off = A->row_off;
    const int has_res = A->has_window && A->res_nnz > 0 && y_res != NULL;

    if (A->has_perm) {
        const int nblk = (A->nvrow + SIGMA - 1) / SIGMA;
#pragma omp parallel
        {
            MAT_VAL_TYPE tmp[SIGMA];
#pragma omp for schedule(static)
            for (int b = 0; b < nblk; b++) {
                const int base = b * SIGMA;
                int e = base + SIGMA;
                if (e > A->nvrow) e = A->nvrow;
                if (e <= base) continue;

                const int is_perm = (A->perm_blk && A->perm_blk[b]) ? 1 : 0;
                if (is_perm) {
                    for (int u = base; u < e; u++) {
                        MAT_VAL_TYPE v = yd[base + (int)A->rinv[u]];
                        if (has_res && u < A->m_dsp) v += y_res[u];
                        tmp[u - base] = v;
                    }
                    for (int u = base; u < e; u++) yd[u] = tmp[u - base];
                } else if (has_res) {
                    int lim = e;
                    if (lim > A->m_dsp) lim = A->m_dsp;
                    for (int u = base; u < lim; u++)
                        if (A->res_rp[u + 1] > A->res_rp[u]) yd[u] += y_res[u];
                }
            }
        }

        for (int j = 0; j < A->n_ov; j++)
            y[off + A->ov_row[j]] += yd[A->m_dsp + j];
        return;
    }

    if (has_res) {
        const int *rp = A->res_rp;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < A->m_dsp; i++)
            if (rp[i + 1] > rp[i]) y[off + i] += y_res[i];
    }
    for (int j = 0; j < A->n_ov; j++)
        y[off + A->ov_row[j]] += yd[A->m_dsp + j];
}

static inline int page_needs_sep_y(const page_matrix *A)
{
    (void)A;
    return 0;
}

static void page_csr_ref(const CSRMatrix *csr,
                          const MAT_VAL_TYPE *x, MAT_VAL_TYPE *y)
{
    const int m = csr->numRows;
#pragma omp parallel for
    for (int i = 0; i < m; i++) {
        MAT_VAL_TYPE sum = 0.0;
        for (int j = csr->rowPointers[i]; j < csr->rowPointers[i + 1]; j++)
            sum += csr->values[j] * x[csr->colIndices[j]];
        y[i] = sum;
    }
}

static void page_csr_ref_rows(const CSRMatrix *csr,
                               const MAT_VAL_TYPE *x, MAT_VAL_TYPE *y,
                               int r0, int r1)
{
    if (!csr || !x || !y || r1 <= r0) return;
    if (r0 < 0) r0 = 0;
    if (r1 > csr->numRows) r1 = csr->numRows;
#pragma omp parallel for
    for (int i = r0; i < r1; i++) {
        MAT_VAL_TYPE sum = 0.0;
        for (int j = csr->rowPointers[i]; j < csr->rowPointers[i + 1]; j++)
            sum += csr->values[j] * x[csr->colIndices[j]];
        y[i] = sum;
    }
}

static void page_csr_ref_residual(const page_matrix *A,
                                   const MAT_VAL_TYPE *x, MAT_VAL_TYPE *y_res)
{
    if (!A || !x || !y_res || !A->has_window || A->res_nnz <= 0) return;
#pragma omp parallel for
    for (int i = 0; i < A->m_dsp; i++) {
        MAT_VAL_TYPE sum = 0.0;
        for (int j = A->res_rp[i]; j < A->res_rp[i + 1]; j++)
            sum += A->res_val[j] * x[A->res_ci[j]];
        y_res[i] = sum;
    }
}

static void page_row_scale(const CSRMatrix *csr, const MAT_VAL_TYPE *x,
                            double *scale)
{
    const int m = csr->numRows;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        double s = 0.0;
        for (int j = csr->rowPointers[i]; j < csr->rowPointers[i + 1]; j++)
            s += fabs(csr->values[j]) * fabs(x[csr->colIndices[j]]);
        scale[i] = s;
    }
}

static int page_validate(const MAT_VAL_TYPE *y, const MAT_VAL_TYPE *ref,
                          const double *scale, int m, double *max_rel)
{
    int bad = 0, shown = 0;
    double worst = 0.0;
    for (int i = 0; i < m; i++) {
        const double d = fabs(y[i] - ref[i]);
        const double den = scale ? (scale[i] + PAGE_ABS_EPS)
                                 : (fabs(ref[i]) + PAGE_ABS_EPS);
        const double r = d / den;
        if (r > worst) worst = r;
        if (r > PAGE_REL_TOL) {
            bad++;
            if (shown < 8) {
                printf("    [MISMATCH] row %d: got %.17g expect %.17g rel=%.3e\n",
                       i, y[i], ref[i], r);
                shown++;
            }
        }
    }
    if (max_rel) *max_rel = worst;
    return bad;
}

#endif
