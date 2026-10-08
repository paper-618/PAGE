#ifndef _PAGE_VALUES_H_
#define _PAGE_VALUES_H_

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "common.h"

#ifndef PAGE_REPR_CPU_SEQ_BYTES
#define PAGE_REPR_CPU_SEQ_BYTES 12.0
#endif

typedef struct {
    int *numeric_row_nnz;
    long long input_nnz;
    long long numeric_nnz;
    long long explicit_zeros;

    int original_value_mode;
    MAT_VAL_TYPE original_constant_value;
    int numeric_value_mode;
    MAT_VAL_TYPE numeric_constant_value;

    long long sell_stored_original;
    long long sell_stored_numeric;
    double sell_amp_original;
    double sell_amp_numeric;
    int geometry_checked;
} page_value_analysis;

typedef struct {
    double bw_ddr;
    double bw_arm;
    double bw_arm_mix;
    double bw_pipe;
    double bw_pipe_mix;
    double bw_xload;
    double launch_ms;
} page_shadow_params;

typedef struct {
    double original_ms;
    double numeric_ms;
    double gain;
    int original_row_off;
    int numeric_row_off;
    double original_cpu_ms;
    double original_dsp_ms;
    double numeric_cpu_ms;
    double numeric_dsp_ms;
    long long original_suffix_stored;
    long long numeric_suffix_stored;
    int prune_selected;
} page_shadow_decision;

static inline const char *page_value_mode_name(int mode)
{
    switch (mode) {
        case PAGE_VALUE_UNIT:     return "unit";
        case PAGE_VALUE_CONSTANT: return "constant";
        default:                   return "general";
    }
}

static inline void page_value_analysis_init(page_value_analysis *a)
{
    if (!a) return;
    memset(a, 0, sizeof(*a));
    a->original_value_mode = PAGE_VALUE_GENERAL;
    a->numeric_value_mode = PAGE_VALUE_GENERAL;
    a->sell_amp_original = 1.0;
    a->sell_amp_numeric = 1.0;
}

static inline void page_value_analysis_free(page_value_analysis *a)
{
    if (!a) return;
    free(a->numeric_row_nnz);
    a->numeric_row_nnz = NULL;
}

static int page_int_desc_cmp(const void *pa, const void *pb)
{
    const int a = *(const int *)pa;
    const int b = *(const int *)pb;
    return (a < b) - (a > b);
}

static long long page_est_sigma_sell_stored(const int *row_nnz, int m)
{
    if (!row_nnz || m <= 0) return 0;
    int tmp[SIGMA];
    long long stored = 0;
    for (int base = 0; base < m; base += SIGMA) {
        int cnt = m - base;
        if (cnt > SIGMA) cnt = SIGMA;
        memcpy(tmp, row_nnz + base, (size_t)cnt * sizeof(int));
        qsort(tmp, (size_t)cnt, sizeof(int), page_int_desc_cmp);
        for (int k = 0; k < cnt; k += SROW) {
            const int mx = tmp[k];
            if (mx > 0) stored += (long long)SROW * (long long)mx;
        }
    }
    return stored;
}

static int page_classify_original_values_fast(const CSRMatrix *csr,
                                                int *mode_out,
                                                MAT_VAL_TYPE *constant_out)
{
    if (mode_out) *mode_out = PAGE_VALUE_GENERAL;
    if (constant_out) *constant_out = (MAT_VAL_TYPE)0.0;
    if (!csr || !csr->values || csr->numNonzeros <= 0) return 1;

    const int nnz = csr->numNonzeros;
    const MAT_VAL_TYPE first = csr->values[0];
    int all_same = 1;
#pragma omp parallel for schedule(static) reduction(&:all_same)
    for (int j = 0; j < nnz; ++j) {
        if (csr->values[j] != first) all_same = 0;
    }
    if (all_same) {
        if (mode_out) *mode_out = (first == (MAT_VAL_TYPE)1.0)
                                ? PAGE_VALUE_UNIT : PAGE_VALUE_CONSTANT;
        if (constant_out) *constant_out = first;
    }
    return 1;
}

static int page_analyze_numeric_structure(const CSRMatrix *csr,
                                           page_value_analysis *a)
{
    if (!csr || !a || !csr->rowPointers || !csr->colIndices || !csr->values ||
        csr->numRows < 0 || csr->numNonzeros < 0)
        return 0;

    page_value_analysis_init(a);
    const int m = csr->numRows;
    const int nnz = csr->numNonzeros;
    a->input_nnz = nnz;
    a->numeric_row_nnz = (int *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    if (!a->numeric_row_nnz) return 0;

    if (nnz == 0) {
        for (int i = 0; i < m; i++) a->numeric_row_nnz[i] = 0;
        return 1;
    }

    const MAT_VAL_TYPE first_all = csr->values[0];
    MAT_VAL_TYPE first_numeric = (MAT_VAL_TYPE)0.0;
    int have_numeric = 0;
    for (int j = 0; j < nnz; j++) {
        if (csr->values[j] != (MAT_VAL_TYPE)0.0) {
            first_numeric = csr->values[j];
            have_numeric = 1;
            break;
        }
    }

    long long zeros = 0;
    long long numeric = 0;
    int all_same_original = 1;
    int all_same_numeric = 1;

#pragma omp parallel for schedule(static) reduction(+:zeros,numeric) reduction(&:all_same_original,all_same_numeric)
    for (int i = 0; i < m; i++) {
        const int b = csr->rowPointers[i];
        const int e = csr->rowPointers[i + 1];
        int rn = 0;
        for (int j = b; j < e; j++) {
            const MAT_VAL_TYPE v = csr->values[j];
            if (v != first_all) all_same_original = 0;
            if (v == (MAT_VAL_TYPE)0.0) {
                zeros++;
            } else {
                rn++;
                numeric++;
                if (have_numeric && v != first_numeric) all_same_numeric = 0;
            }
        }
        a->numeric_row_nnz[i] = rn;
    }

    a->explicit_zeros = zeros;
    a->numeric_nnz = numeric;

    if (all_same_original) {
        a->original_constant_value = first_all;
        a->original_value_mode = (first_all == (MAT_VAL_TYPE)1.0)
                               ? PAGE_VALUE_UNIT : PAGE_VALUE_CONSTANT;
    }
    if (have_numeric && all_same_numeric) {
        a->numeric_constant_value = first_numeric;
        a->numeric_value_mode = (first_numeric == (MAT_VAL_TYPE)1.0)
                              ? PAGE_VALUE_UNIT : PAGE_VALUE_CONSTANT;
    }

    if (zeros <= 0) {
        a->numeric_nnz = nnz;
        a->numeric_value_mode = a->original_value_mode;
        a->numeric_constant_value = a->original_constant_value;
        return 1;
    }

    int *orig_row_nnz = (int *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    if (!orig_row_nnz) return 1;
    for (int i = 0; i < m; i++)
        orig_row_nnz[i] = csr->rowPointers[i + 1] - csr->rowPointers[i];

    a->sell_stored_original = page_est_sigma_sell_stored(orig_row_nnz, m);
    a->sell_stored_numeric = page_est_sigma_sell_stored(a->numeric_row_nnz, m);
    a->geometry_checked = 1;
    free(orig_row_nnz);

    if (a->input_nnz > 0)
        a->sell_amp_original = (double)a->sell_stored_original / (double)a->input_nnz;
    if (a->numeric_nnz > 0)
        a->sell_amp_numeric = (double)a->sell_stored_numeric / (double)a->numeric_nnz;
    else
        a->sell_amp_numeric = INFINITY;
    return 1;
}

static int page_make_numeric_shadow_csr(const CSRMatrix *src,
                                         const page_value_analysis *a,
                                         CSRMatrix *dst,
                                         int **owned_rp, int **owned_ci)
{
    if (owned_rp) *owned_rp = NULL;
    if (owned_ci) *owned_ci = NULL;
    if (!src || !a || !dst || !a->numeric_row_nnz || a->numeric_nnz <= 0 ||
        a->numeric_nnz > 2147483647LL) return 0;
    const int m = src->numRows;
    int *rp = (int *)malloc((size_t)(m + 1) * sizeof(int));
    int *ci = (int *)malloc((size_t)a->numeric_nnz * sizeof(int));
    if (!rp || !ci) { free(rp); free(ci); return 0; }
    rp[0] = 0;
    for (int i = 0; i < m; i++) rp[i + 1] = rp[i] + a->numeric_row_nnz[i];

#pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < m; i++) {
        int w = rp[i];
        const int b = src->rowPointers[i], e = src->rowPointers[i + 1];
        for (int j = b; j < e; j++)
            if (src->values[j] != (MAT_VAL_TYPE)0.0)
                ci[w++] = src->colIndices[j];
    }

    memset(dst, 0, sizeof(*dst));
    dst->numRows = src->numRows;
    dst->numCols = src->numCols;
    dst->numNonzeros = (int)a->numeric_nnz;
    dst->rowPointers = rp;
    dst->colIndices = ci;
    dst->values = NULL;
    dst->value_mode = PAGE_VALUE_GENERAL;
    dst->constant_value = 0.0;
    if (owned_rp) *owned_rp = rp;
    if (owned_ci) *owned_ci = ci;
    return 1;
}

static double page_shadow_unique_cols(long long nnz, int n)
{
    if (nnz <= 0 || n <= 0) return 0.0;
    const double x = (double)nnz / (double)n;
    double u = (double)n * (1.0 - exp(-x));
    if (u > (double)n) u = (double)n;
    if (u > (double)nnz) u = (double)nnz;
    return u;
}

static int page_shadow_tables(const CSRMatrix *csr, const int *row_nnz,
                               long long **pre_out, long long **sell_suf_out,
                               int *nblk_out)
{
    const int m = csr->numRows;
    const int nblk = (m + SIGMA - 1) / SIGMA;
    long long *pre = (long long *)malloc((size_t)(m + 1) * sizeof(long long));
    long long *suf = (long long *)malloc((size_t)(nblk + 1) * sizeof(long long));
    long long *blk = (long long *)malloc((size_t)(nblk > 0 ? nblk : 1) * sizeof(long long));
    if (!pre || !suf || !blk) { free(pre); free(suf); free(blk); return 0; }

    pre[0] = 0;
    for (int i = 0; i < m; i++) {
        const int len = row_nnz ? row_nnz[i]
                                : csr->rowPointers[i + 1] - csr->rowPointers[i];
        pre[i + 1] = pre[i] + (long long)len;
    }

    int tmp[SIGMA];
    for (int b = 0; b < nblk; b++) {
        const int base = b * SIGMA;
        int cnt = m - base;
        if (cnt > SIGMA) cnt = SIGMA;
        for (int i = 0; i < cnt; i++)
            tmp[i] = row_nnz ? row_nnz[base + i]
                             : csr->rowPointers[base + i + 1] - csr->rowPointers[base + i];
        qsort(tmp, (size_t)cnt, sizeof(int), page_int_desc_cmp);
        long long stored = 0;
        for (int k = 0; k < cnt; k += SROW) {
            const int mx = tmp[k];
            if (mx > 0) stored += (long long)SROW * (long long)mx;
        }
        blk[b] = stored;
    }
    suf[nblk] = 0;
    for (int b = nblk - 1; b >= 0; b--) suf[b] = suf[b + 1] + blk[b];
    free(blk);
    *pre_out = pre; *sell_suf_out = suf; *nblk_out = nblk;
    return 1;
}

static double page_shadow_eval_repr(const CSRMatrix *csr, const int *row_nnz,
                                     long long total_nnz,
                                     const page_shadow_params *p,
                                     int *best_row_off,
                                     double *best_cpu_ms, double *best_dsp_ms,
                                     long long *best_suffix_stored)
{
    if (!csr || !p || total_nnz <= 0) return 1.0e300;
    long long *pre = NULL, *suf = NULL;
    int nblk = 0;
    if (!page_shadow_tables(csr, row_nnz, &pre, &suf, &nblk)) return 1.0e300;

    const int m = csr->numRows, n = csr->numCols;
    const double bw_ddr = (p->bw_ddr > 0.0) ? p->bw_ddr : 1.0;
    const double bw_arm = (p->bw_arm > 0.0) ? p->bw_arm : bw_ddr;
    const double bw_arm_mix = (p->bw_arm_mix > 0.0) ? p->bw_arm_mix : bw_arm;
    const double bw_pipe = (p->bw_pipe > 0.0) ? p->bw_pipe : bw_ddr;
    const double bw_pipe_mix = (p->bw_pipe_mix > 0.0) ? p->bw_pipe_mix : bw_pipe;
    const double bw_xload = (p->bw_xload > 0.0) ? p->bw_xload : bw_ddr;
    const double launch_ms = (p->launch_ms >= 0.0) ? p->launch_ms : 0.0;

    double best_cpu = ((double)total_nnz * PAGE_REPR_CPU_SEQ_BYTES
                     + (double)m * 8.0 + (double)n * 8.0) / bw_arm * 1e3;
    double best = best_cpu;
    double best_dsp = 0.0;
    int best_ro = m;
    long long best_stored = 0;

    for (int b = 0; b < nblk; b++) {
        const int row_off = b * SIGMA;
        const long long prefix_nnz = pre[row_off];
        const long long suffix_nnz = total_nnz - prefix_nnz;
        if (suffix_nnz <= 0) continue;
        const int suffix_rows = m - row_off;
        const long long stored = suf[b];

        const int concurrent = prefix_nnz > 0;
        const double arm_bw_use = concurrent ? bw_arm_mix : bw_arm;
        const double pipe_bw_use = concurrent ? bw_pipe_mix : bw_pipe;

        const double arm_x = page_shadow_unique_cols(prefix_nnz, n);
        const double arm_bytes = (double)prefix_nnz * PAGE_REPR_CPU_SEQ_BYTES
                               + (double)row_off * 8.0 + arm_x * 8.0;
        const double ta = (arm_bytes > 0.0) ? arm_bytes / arm_bw_use * 1e3 : 0.0;

        const double pair_bytes = (double)stored * 16.0;
        const double meta_bytes = (double)stored * 4.0 + (double)suffix_rows * 8.0;
        const double x_elems = page_shadow_unique_cols(suffix_nnz, n);
        const double td = (pair_bytes / pipe_bw_use
                         + meta_bytes / bw_ddr
                         + x_elems * 8.0 / bw_xload) * 1e3 + launch_ms;
        const double t = (ta > td) ? ta : td;
        if (t < best) {
            best = t; best_ro = row_off; best_cpu = ta; best_dsp = td;
            best_stored = stored;
        }
    }

    free(pre); free(suf);
    if (best_row_off) *best_row_off = best_ro;
    if (best_cpu_ms) *best_cpu_ms = best_cpu;
    if (best_dsp_ms) *best_dsp_ms = best_dsp;
    if (best_suffix_stored) *best_suffix_stored = best_stored;
    return best;
}

static int page_shadow_select_prune(const CSRMatrix *csr,
                                     const page_value_analysis *a,
                                     const page_shadow_params *p,
                                     page_shadow_decision *d)
{
    if (!d) return 0;
    memset(d, 0, sizeof(*d));
    d->original_ms = d->numeric_ms = 1.0e300;
    d->gain = 1.0;
    if (!csr || !a || !p || a->explicit_zeros <= 0 || a->numeric_nnz <= 0 ||
        !a->numeric_row_nnz)
        return 0;

    d->original_ms = page_shadow_eval_repr(csr, NULL, a->input_nnz, p,
                                             &d->original_row_off,
                                             &d->original_cpu_ms, &d->original_dsp_ms,
                                             &d->original_suffix_stored);
    d->numeric_ms = page_shadow_eval_repr(csr, a->numeric_row_nnz, a->numeric_nnz, p,
                                            &d->numeric_row_off,
                                            &d->numeric_cpu_ms, &d->numeric_dsp_ms,
                                            &d->numeric_suffix_stored);
    if (d->numeric_ms > 0.0 && isfinite(d->numeric_ms) && isfinite(d->original_ms))
        d->gain = d->original_ms / d->numeric_ms;
    d->prune_selected = (isfinite(d->numeric_ms) && isfinite(d->original_ms) &&
                         d->numeric_ms < d->original_ms);
    return d->prune_selected;
}

static long long page_apply_zero_prune(CSRMatrix *csr)
{
    if (!csr || !csr->rowPointers || !csr->colIndices || !csr->values ||
        csr->numRows < 0 || csr->numNonzeros <= 0)
        return 0;

    const int old_nnz = csr->numNonzeros;
    int read_b = 0;
    int write_pos = 0;
    for (int i = 0; i < csr->numRows; i++) {
        const int read_e = csr->rowPointers[i + 1];
        csr->rowPointers[i] = write_pos;
        for (int j = read_b; j < read_e; j++) {
            const MAT_VAL_TYPE v = csr->values[j];
            if (v != (MAT_VAL_TYPE)0.0) {
                if (write_pos != j) {
                    csr->colIndices[write_pos] = csr->colIndices[j];
                    csr->values[write_pos] = v;
                }
                write_pos++;
            }
        }
        read_b = read_e;
    }
    csr->rowPointers[csr->numRows] = write_pos;
    csr->numNonzeros = write_pos;
    return (long long)old_nnz - (long long)write_pos;
}

static inline double page_cpu_actual_seq_bytes_per_nnz(const CSRMatrix *csr)
{
    if (csr && csr->value_mode != PAGE_VALUE_GENERAL)
        return (double)sizeof(int);
    return (double)sizeof(int) + (double)sizeof(MAT_VAL_TYPE);
}

#endif
