#ifndef _PAGE_PREP_H_
#define _PAGE_PREP_H_

#include <omp.h>
#include "common.h"
#include "page_format.h"

typedef struct {
    int    *row_nnz;
    int    *row_cmin;
    int    *row_cmax;
    unsigned char *row_sorted;
    long long *row_pre;

    int    *v_len;
    int    *v_cmin;
    int    *v_cmax;

    double  cv_global;
    double  nnz_per_row;
    double  bandwidth_ratio;
    int     max_row_nnz;
    int     dof_r;
    double  dof_hit;
    int     mode;
} page_stat;

typedef struct {
    double bw_ddr;
    double bw_gsm;
    double bw_arm;
    double bw_mix;
    double bw_pipe;
    double bw_arm_mix;
    double bw_pipe_mix;
    double launch_ms;
    double barrier_ms;
    double bw_xload;
    double xload_fixed_ms;
    double cpu_share;
    int row_off_override;
    int plan_policy;
    int force_cpu;
    int force_hybrid;
    int force_idx32;
    int fast_shadow_precheck;
    int dsp_value_mode;
    MAT_VAL_TYPE dsp_constant_value;
} page_opts;

static void page_opts_default(page_opts *o)
{
    o->bw_ddr      = PAGE_BW_DDR;
    o->bw_gsm      = PAGE_BW_GSM;
    o->bw_arm      = PAGE_BW_ARM;
    o->bw_mix      = PAGE_BW_MIX;
    o->bw_pipe     = PAGE_BW_PIPE;
    o->bw_arm_mix  = PAGE_BW_ARM;
    o->bw_pipe_mix = PAGE_BW_PIPE;
    o->launch_ms   = PAGE_KERNEL_FIX_MS;
    o->barrier_ms  = PAGE_BARRIER_FIX_MS;
    o->bw_xload    = PAGE_BW_DDR;
    o->xload_fixed_ms = 0.0;
    o->cpu_share   = -1.0;
    o->row_off_override = -1;
    o->plan_policy = -1;
    o->force_cpu = 0;
    o->force_hybrid = 0;
    o->force_idx32 = 0;
    o->fast_shadow_precheck = 0;
    o->dsp_value_mode = PAGE_VALUE_GENERAL;
    o->dsp_constant_value = (MAT_VAL_TYPE)0.0;
}

static void page_pass_a(const CSRMatrix *csr, page_stat *st)
{
    const int m = csr->numRows;
    st->row_nnz    = (int *)malloc((size_t)m * sizeof(int));
    st->row_cmin   = (int *)malloc((size_t)m * sizeof(int));
    st->row_cmax   = (int *)malloc((size_t)m * sizeof(int));
    st->row_sorted = (unsigned char *)malloc((size_t)m * sizeof(unsigned char));
    st->row_pre    = (long long *)malloc((size_t)(m + 1) * sizeof(long long));
    st->v_len = NULL; st->v_cmin = NULL; st->v_cmax = NULL;

    double sum = 0.0, sum2 = 0.0, sumw = 0.0;

#pragma omp parallel for schedule(static) reduction(+:sum,sum2,sumw)
    for (int i = 0; i < m; i++) {
        const int b = csr->rowPointers[i];
        const int e = csr->rowPointers[i + 1];
        int cmin = INT_MAX, cmax = -1, prev = -1, sorted = 1;
        for (int k = b; k < e; k++) {
            const int c = csr->colIndices[k];
            if (c < cmin) cmin = c;
            if (c > cmax) cmax = c;
            if (k > b && c < prev) sorted = 0;
            prev = c;
        }
        const int len = e - b;
        st->row_nnz[i]    = len;
        st->row_cmin[i]   = (cmax < 0) ? 0 : cmin;
        st->row_cmax[i]   = (cmax < 0) ? -1 : cmax;
        st->row_sorted[i] = (unsigned char)sorted;
        sum  += (double)len;
        sum2 += (double)len * (double)len;
        sumw += (cmax < 0) ? 0.0 : (double)(cmax - cmin + 1);
    }

    int maxlen = 0;
    st->row_pre[0] = 0;
    for (int i = 0; i < m; i++) {
        st->row_pre[i + 1] = st->row_pre[i] + st->row_nnz[i];
        if (st->row_nnz[i] > maxlen) maxlen = st->row_nnz[i];
    }

    const double mean = sum / (double)m;
    double var = sum2 / (double)m - mean * mean;
    if (var < 0.0) var = 0.0;
    st->nnz_per_row     = mean;
    st->max_row_nnz     = maxlen;
    st->cv_global       = (mean > 0.0) ? sqrt(var) / mean : 0.0;
    st->bandwidth_ratio = (csr->numCols > 0)
                        ? (sumw / (double)m) / (double)csr->numCols : 0.0;

    st->dof_r = 1; st->dof_hit = 0.0;
    if (m > 64) {
        const int nsample = (m - 1 < 4096) ? (m - 1) : 4096;
        const int stride  = (m - 1) / nsample;
        int hit = 0, tot = 0;
        for (int s = 0; s < nsample; s++) {
            const int i = s * stride;
            if (i + 1 >= m) break;
            const int b0 = csr->rowPointers[i], e0 = csr->rowPointers[i + 1];
            const int b1 = csr->rowPointers[i + 1];
            const int e1 = csr->rowPointers[i + 2 > m ? m : i + 2];
            tot++;
            if ((e0 - b0) != (e1 - b1) || (e0 - b0) == 0) continue;
            int same = 1;
            for (int k = 0; k < e0 - b0; k++)
                if (csr->colIndices[b0 + k] != csr->colIndices[b1 + k]) { same = 0; break; }
            hit += same;
        }
        if (tot > 0) st->dof_hit = (double)hit / (double)tot;
        if (st->dof_hit > 0.95) st->dof_r = 2;
    }

    st->mode = (csr->numCols <= GSM_X_CAP) ? PAGE_MODE_A : PAGE_MODE_B;
}

static void page_free_stat(page_stat *st)
{
    free(st->row_nnz);    st->row_nnz    = NULL;
    free(st->row_cmin);   st->row_cmin   = NULL;
    free(st->row_cmax);   st->row_cmax   = NULL;
    free(st->row_sorted); st->row_sorted = NULL;
    free(st->row_pre);    st->row_pre    = NULL;
    free(st->v_len);      st->v_len      = NULL;
    free(st->v_cmin);     st->v_cmin     = NULL;
    free(st->v_cmax);     st->v_cmax     = NULL;
}

static inline int page_lower_bound(const int *col, int lo, int hi, int key)
{
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (col[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static inline int page_vid(const page_matrix *A, int pos)
{
    if (!A->has_perm) return pos;
    return (pos / SIGMA) * SIGMA + (int)A->rperm[pos];
}

static inline int page_vpos(const page_matrix *A, int u)
{
    if (!A->has_perm) return u;
    return (u / SIGMA) * SIGMA + (int)A->rinv[u];
}

static inline int page_vparent(const page_matrix *A, int u)
{
    if (u < A->m_dsp) return u;
    if (u < A->m_dsp + A->n_ov) return A->ov_row[u - A->m_dsp];
    return -1;
}

static inline int page_vrow_global(const page_matrix *A, int u)
{
    const int p = page_vparent(A, u);
    return (p < 0) ? -1 : A->row_off + p;
}

static inline void page_vrange(const CSRMatrix *csr, const page_matrix *A,
                                int u, int *kb, int *ke)
{
    if (A->vr_kb) { *kb = A->vr_kb[u]; *ke = A->vr_ke[u]; return; }
    if (u < A->m_dsp) {
        const int r = A->row_off + u;
        *kb = csr->rowPointers[r]; *ke = csr->rowPointers[r + 1]; return;
    }
    *kb = 0; *ke = 0;
}

static int page_cmp_int_asc(const void *a, const void *b)
{
    const int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

#define PAGE_PACK_HASH_SIZE (1 << 19)

static int page_panel_unique_blocks(const CSRMatrix *csr, const page_matrix *A,
                                     int s0, int s1, int *out,
                                     int *keys, unsigned int *tags,
                                     unsigned int generation)
{
    const int HSIZE = PAGE_PACK_HASH_SIZE;
    if (!keys || !tags || generation == 0) return PAGE_XBLOCK_CAP + 1;

    int nblk = 0;
    const int p0 = s0 * SROW;
    int p1 = s1 * SROW;
    if (p1 > A->nvrow) p1 = A->nvrow;
    for (int pos = p0; pos < p1 && nblk <= PAGE_XBLOCK_CAP; pos++) {
        const int u = page_vid(A, pos);
        int kb, ke;
        page_vrange(csr, A, u, &kb, &ke);
        for (int k = kb; k < ke; k++) {
            const int c = csr->colIndices[k];
            if (A->has_window && (c < A->cw_lo || c >= A->cw_hi)) continue;
            if (c < 0) return PAGE_XBLOCK_CAP + 1;
            const int blk = c / PAGE_XBLOCK_ELEMS;
            if (blk > INT_MAX / PAGE_XBLOCK_BYTES) return PAGE_XBLOCK_CAP + 1;
            unsigned h = (unsigned)blk * 2654435761u;
            int slot = (int)(h & (HSIZE - 1));
            while (tags[slot] == generation && keys[slot] != blk)
                slot = (slot + 1) & (HSIZE - 1);
            if (tags[slot] == generation) continue;
            tags[slot] = generation;
            keys[slot] = blk;
            if (nblk <= PAGE_XBLOCK_CAP) out[nblk] = blk;
            nblk++;
            if (nblk > PAGE_XBLOCK_CAP) break;
        }
    }
    if (nblk <= PAGE_XBLOCK_CAP)
        qsort(out, (size_t)nblk, sizeof(int), page_cmp_int_asc);
    return nblk;
}

static int page_count_block_runs(const int *blk, int n)
{
    if (!blk || n <= 0) return 0;
    int runs = 1;
    for (int i = 1; i < n; i++)
        if (blk[i] != blk[i - 1] + 1) runs++;
    return runs;
}

static int page_pack_profitable(int span_elems, int nblk, int nruns,
                                 int coreNum, const page_opts *opt)
{
    if (nblk <= 0 || nblk > PAGE_XBLOCK_CAP || span_elems <= 0) return 0;
    const double bw = (opt && opt->bw_xload > 0.0) ? opt->bw_xload
                                                   : ((opt && opt->bw_ddr > 0.0) ? opt->bw_ddr : PAGE_BW_DDR);
    const double fix = (opt && opt->xload_fixed_ms > 0.0) ? opt->xload_fixed_ms : 0.0;
    const int ntile = (span_elems + PAGE_TILE_CAP - 1) / PAGE_TILE_CAP;

    const int packed_rounds = (nruns + coreNum - 1 + coreNum - 1) / coreNum;
    const double t_pack = ((double)nblk * PAGE_XBLOCK_BYTES) / bw * 1e3
                        + (double)packed_rounds * fix;
    const double t_tile = ((double)span_elems * sizeof(MAT_VAL_TYPE)) / bw * 1e3
                        + (double)ntile * fix;
    return t_pack < t_tile;
}

static inline int page_lower_bound_byteoff(const int *a, int lo, int hi, int byteoff)
{
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (a[mid] < byteoff) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static void page_build_vrows(const CSRMatrix *csr, page_stat *st,
                              page_matrix *A, int row_off, int m_dsp)
{
    const int *rp = csr->rowPointers + row_off;
    const int *rnnz   = st->row_nnz   + row_off;
    const int *rcmin  = st->row_cmin  + row_off;
    const int *rcmax  = st->row_cmax  + row_off;
    const unsigned char *rsort = st->row_sorted + row_off;

    int cap = (int)(PAGE_ROWCAP_MULT * st->nnz_per_row + 0.5);
    if (cap < PAGE_ROWCAP_MIN) cap = PAGE_ROWCAP_MIN;
    if (cap > PAGE_ROWCAP_MAX) cap = PAGE_ROWCAP_MAX;

    /* Estimate unsplit slice pressure from row lengths, without a timed SpMV pilot. */
    const long long nnz_dsp_pre = st->row_pre[row_off + m_dsp] - st->row_pre[row_off];
    int max_suffix_row_nnz = 0;
    long long unsplit_stored_pre = 0;
    long long max_unsplit_slice_work = 0;
    for (int base = 0; base < m_dsp; base += SROW) {
        int width = 0;
        const int end = (base + SROW < m_dsp) ? base + SROW : m_dsp;
        for (int i = base; i < end; i++) {
            if (rnnz[i] > width) width = rnnz[i];
            if (rnnz[i] > max_suffix_row_nnz) max_suffix_row_nnz = rnnz[i];
        }
        const long long unit = (long long)width * (long long)SROW;
        unsplit_stored_pre += unit;
        if (unit > max_unsplit_slice_work) max_unsplit_slice_work = unit;
    }

    const int gate_cores = (A->coreNum > 0) ? A->coreNum : 1;
    const double ideal_per_dsp = (unsplit_stored_pre > 0)
        ? (double)unsplit_stored_pre / (double)gate_cores : 0.0;
    const double g_pre = (ideal_per_dsp > 0.0)
        ? (double)max_unsplit_slice_work / ideal_per_dsp : 0.0;

    int g_gate_enabled = 1;
    {
        const char *e = getenv("PAGE_VROW_G_GATE");
        if (e && strcmp(e, "0") == 0) g_gate_enabled = 0;
    }

    int ablate_vrow = 0;
    {
        const char *e = getenv("PAGE_ABLATE_VROW");
        if (e && strcmp(e, "1") == 0) ablate_vrow = 1;
    }

    const int candidate_cap = cap;
    const int g_gate_pass = (!g_gate_enabled || g_pre > PAGE_VROW_G_THRESHOLD);

    if (ablate_vrow) {
        cap = (st->max_row_nnz > 0) ? st->max_row_nnz : PAGE_ROWCAP_MAX;
    } else if (!g_gate_pass) {
        cap = (max_suffix_row_nnz > 0) ? max_suffix_row_nnz : candidate_cap;
    }

    long long n_ov = 0;
    for (int retry = 0; ; retry++) {
        n_ov = 0;
        if (st->max_row_nnz > cap) {
#pragma omp parallel for schedule(static) reduction(+:n_ov)
            for (int i = 0; i < m_dsp; i++) {
                const int len = rnnz[i];
                if (len > cap) n_ov += (long long)((len + cap - 1) / cap) - 1;
            }
        }
        if (n_ov <= (long long)(PAGE_OV_RATIO_MAX * (double)m_dsp) ||
            cap >= PAGE_ROWCAP_MAX || retry >= PAGE_ROWCAP_RETRY) break;
        cap *= 2;
        if (cap > PAGE_ROWCAP_MAX) cap = PAGE_ROWCAP_MAX;
    }
    A->row_cap = cap;

    {
        const char *diag = getenv("PAGE_VROW_GATE_DIAG");
        if (diag && strcmp(diag, "0") != 0) {
            printf("[VROW_GATE] row_off=%d m_dsp=%d nnz_dsp=%lld max_row_nnz=%d "
                   "unsplit_stored=%lld max_sched_unit=%lld cores=%d "
                   "ideal_per_dsp=%.6f G_pre=%.9f threshold=%.6f "
                   "gate_enabled=%d gate_pass=%d candidate_cap=%d final_cap=%d "
                   "n_ov=%lld split=%d\n",
                   row_off, m_dsp, nnz_dsp_pre, max_suffix_row_nnz,
                   unsplit_stored_pre, max_unsplit_slice_work, gate_cores,
                   ideal_per_dsp, g_pre, (double)PAGE_VROW_G_THRESHOLD,
                   g_gate_enabled, g_gate_pass, candidate_cap, cap, n_ov,
                   (n_ov > 0));
        }
    }

    const long long nv_real = (long long)m_dsp + n_ov;
    const int nvrow = (int)(((nv_real + SROW - 1) / SROW) * SROW);

    A->row_off   = row_off;
    A->m_dsp     = m_dsp;
    A->n_ov      = (int)n_ov;
    A->nvrow     = nvrow;
    A->has_split = (n_ov > 0);
    A->ov_row    = (n_ov > 0) ? (int *)malloc((size_t)n_ov * sizeof(int)) : NULL;
    A->nnz_dsp   = st->row_pre[row_off + m_dsp] - st->row_pre[row_off];

    st->v_len  = (int *)malloc((size_t)nvrow * sizeof(int));
    st->v_cmin = (int *)malloc((size_t)nvrow * sizeof(int));
    st->v_cmax = (int *)malloc((size_t)nvrow * sizeof(int));

    if (!A->has_split) {

        A->vr_kb = A->vr_ke = NULL;
#pragma omp parallel for schedule(static)
        for (int u = 0; u < nvrow; u++) {
            if (u < m_dsp) {
                st->v_len[u]  = rnnz[u];
                st->v_cmin[u] = rcmin[u];
                st->v_cmax[u] = rcmax[u];
            } else {
                st->v_len[u] = 0; st->v_cmin[u] = 0; st->v_cmax[u] = -1;
            }
        }
        return;
    }

    A->vr_kb = (int *)malloc((size_t)nvrow * sizeof(int));
    A->vr_ke = (int *)malloc((size_t)nvrow * sizeof(int));

    long long *ov_pre = (long long *)malloc((size_t)(m_dsp + 1) * sizeof(long long));
    ov_pre[0] = 0;
    for (int i = 0; i < m_dsp; i++) {
        const int len = rnnz[i];
        const int nc  = (len > cap) ? (len + cap - 1) / cap : 1;
        ov_pre[i + 1] = ov_pre[i] + (nc - 1);
    }

#pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < m_dsp; i++) {
        const int len = rnnz[i];
        const int nc  = (len > cap) ? (len + cap - 1) / cap : 1;

        const int base = len / nc, rem = len % nc;
        int k = rp[i];

        const int c0 = base + (rem > 0 ? 1 : 0);
        A->vr_kb[i] = k; A->vr_ke[i] = k + c0;
        k += c0;

        for (int c = 1; c < nc; c++) {
            const int cw = base + (c < rem ? 1 : 0);
            const int u  = m_dsp + (int)(ov_pre[i] + (c - 1));
            A->vr_kb[u] = k; A->vr_ke[u] = k + cw;
            A->ov_row[u - m_dsp] = i;
            k += cw;
        }
    }
    free(ov_pre);

    for (int u = (int)nv_real; u < nvrow; u++) { A->vr_kb[u] = 0; A->vr_ke[u] = 0; }

#pragma omp parallel for schedule(static)
    for (int u = 0; u < nvrow; u++) {
        const int kb = A->vr_kb[u], ke = A->vr_ke[u];
        st->v_len[u] = ke - kb;
        if (ke <= kb) { st->v_cmin[u] = 0; st->v_cmax[u] = -1; continue; }
        const int par = (u < m_dsp) ? u : A->ov_row[u - m_dsp];
        if (rsort[par]) {
            st->v_cmin[u] = csr->colIndices[kb];
            st->v_cmax[u] = csr->colIndices[ke - 1];
        } else {
            int lo = INT_MAX, hi = -1;
            for (int k = kb; k < ke; k++) {
                const int c = csr->colIndices[k];
                if (c < lo) lo = c;
                if (c > hi) hi = c;
            }
            st->v_cmin[u] = lo; st->v_cmax[u] = hi;
        }
    }
}

static void page_sigma_sort(const int *vlen, int nvrow, int m_dsp,
                             int idx_bytes, const page_opts *opt,
                             unsigned short *rperm,
                             unsigned char *perm_blk, int *nperm_blk_out,
                             long long *st_id, long long *st_chosen,
                             int *ell_min_out)
{
    (void)m_dsp;
    const int nblk = (nvrow + SIGMA - 1) / SIGMA;
    long long sid = 0, sch = 0;
    int emin = INT_MAX, nperm = 0;
    const double bw_pipe = (opt && opt->bw_pipe > 0.0) ? opt->bw_pipe : PAGE_BW_PIPE;
    const double bw_arm = (opt && opt->bw_arm > 0.0) ? opt->bw_arm : PAGE_BW_ARM;

    {
        const char *e = getenv("PAGE_ABLATE_SIGMA");
        if (e && strcmp(e, "1") == 0) {
            for (int b = 0; b < nblk; b++) {
                const int base = b * SIGMA;
                const int len = (base + SIGMA <= nvrow) ? SIGMA : (nvrow - base);
                perm_blk[b] = 0;
                for (int i = 0; i < len; i++) rperm[base + i] = (unsigned short)i;
                for (int sl = 0; sl < len; sl += SROW) {
                    const int ee = (sl + SROW <= len) ? sl + SROW : len;
                    int w = 0;
                    for (int i = sl; i < ee; i++) if (vlen[base+i] > w) w = vlen[base+i];
                    sid += (long long)w * SROW;
                    if (w < emin) emin = w;
                }
            }
            sch = sid;
            *st_id = sid; *st_chosen = sch; *nperm_blk_out = 0;
            *ell_min_out = (emin == INT_MAX) ? 0 : emin;
            return;
        }
    }

#pragma omp parallel for schedule(dynamic, 8) reduction(+:sid,sch,nperm) reduction(min:emin)
    for (int b = 0; b < nblk; b++) {
        const int base = b * SIGMA;
        const int len  = (base + SIGMA <= nvrow) ? SIGMA : (nvrow - base);

        int maxlen = 0;
        for (int i = 0; i < len; i++)
            if (vlen[base + i] > maxlen) maxlen = vlen[base + i];

        if (maxlen > 0) {
            int *cnt = (int *)calloc((size_t)maxlen + 2, sizeof(int));
            for (int i = 0; i < len; i++) cnt[vlen[base + i]]++;
            int acc = 0;
            for (int L = maxlen; L >= 0; L--) {
                const int c = cnt[L]; cnt[L] = acc; acc += c;
            }
            for (int i = 0; i < len; i++)
                rperm[base + cnt[vlen[base + i]]++] = (unsigned short)i;
            free(cnt);
        } else {
            for (int i = 0; i < len; i++) rperm[base + i] = (unsigned short)i;
        }

        long long sid_b = 0, ssort_b = 0;
        int emin_id = INT_MAX, emin_sort = INT_MAX;
        for (int sl = 0; sl < len; sl += SROW) {
            const int e = (sl + SROW <= len) ? sl + SROW : len;
            int wi = 0, ws = 0;
            for (int i = sl; i < e; i++) {
                const int a = vlen[base + i];
                const int c = vlen[base + (int)rperm[base + i]];
                if (a > wi) wi = a;
                if (c > ws) ws = c;
            }
            sid_b   += (long long)wi * SROW;
            ssort_b += (long long)ws * SROW;
            if (wi < emin_id)   emin_id = wi;
            if (ws < emin_sort) emin_sort = ws;
        }

        (void)idx_bytes;
        const double t_id   = (double)sid_b   * 16.0 / bw_pipe;
        const double t_sort = (double)ssort_b * 16.0 / bw_pipe;

        const double t_finalize = 16.0 * (double)len / bw_arm;
        const int use_perm = (ssort_b < sid_b) && (t_sort + t_finalize < t_id);

        if (!use_perm) {
            for (int i = 0; i < len; i++) rperm[base + i] = (unsigned short)i;
            perm_blk[b] = 0;
            sch += sid_b;
            if (emin_id < emin) emin = emin_id;
        } else {
            perm_blk[b] = 1;
            nperm++;
            sch += ssort_b;
            if (emin_sort < emin) emin = emin_sort;
        }
        sid += sid_b;
    }
    *st_id = sid;
    *st_chosen = sch;
    *nperm_blk_out = nperm;
    *ell_min_out = (emin == INT_MAX) ? 0 : emin;
}

static int page_slice_width(const CSRMatrix *csr, const page_stat *st,
                             const page_matrix *A, int rs,
                             int xlo, int xhi, int whole_row)
{
    const int nvrow = A->nvrow;
    const int p0 = rs * SROW;
    int w = 0;
    for (int i = 0; i < SROW; i++) {
        const int pos = p0 + i;
        if (pos >= nvrow) break;
        const int u = page_vid(A, pos);
        int cnt;
        if (whole_row) {
            cnt = st->v_len[u];
        } else if (st->v_cmax[u] < 0) {
            cnt = 0;
        } else if (st->v_cmin[u] >= xlo && st->v_cmax[u] < xhi) {
            cnt = st->v_len[u];
        } else {
            int kb, ke;
            page_vrange(csr, A, u, &kb, &ke);
            const int gr = page_vrow_global(A, u);
            if (gr >= 0 && st->row_sorted[gr]) {
                const int b = page_lower_bound(csr->colIndices, kb, ke, xlo);
                const int e = page_lower_bound(csr->colIndices, b,  ke, xhi);
                cnt = e - b;
            } else {
                cnt = 0;
                for (int k = kb; k < ke; k++) {
                    const int c = csr->colIndices[k];
                    if (c >= xlo && c < xhi) cnt++;
                }
            }
        }
        if (cnt > w) w = cnt;
    }
    return w;
}

static int page_panels_try(const int *blk_lo, const int *blk_hi, int nsblk,
                            int nslice, int panel_target, int tile_cap,
                            int *ptmp_lo, int *pbb_lo, int *pbb_hi,
                            double *sum_width, long long *max_width)
{
    int npanel = 0, cur_slices = 0, cur_lo = INT_MAX, cur_hi = -1;
    ptmp_lo[0] = 0;
    for (int b = 0; b < nsblk; b++) {
        int bs = SLICE_PER_SIGMA;
        if (b == nsblk - 1) bs = nslice - b * SLICE_PER_SIGMA;

        int nlo = (blk_lo[b] < cur_lo) ? blk_lo[b] : cur_lo;
        int nhi = (blk_hi[b] > cur_hi) ? blk_hi[b] : cur_hi;
        const int width = (nhi > nlo) ? nhi - nlo : 0;
        const int cur_w = (cur_hi > cur_lo) ? cur_hi - cur_lo : 0;

        const int need_cut = (cur_slices > 0) &&
                             ((cur_slices + bs > panel_target) ||
                              (width > tile_cap && cur_w <= tile_cap));
        if (need_cut) {
            pbb_lo[npanel] = (cur_lo == INT_MAX) ? 0 : cur_lo;
            pbb_hi[npanel] = (cur_hi < 0) ? 0 : cur_hi;
            npanel++;
            ptmp_lo[npanel] = b * SLICE_PER_SIGMA;
            cur_slices = 0;
            nlo = blk_lo[b]; nhi = blk_hi[b];
        }
        cur_slices += bs; cur_lo = nlo; cur_hi = nhi;
    }
    pbb_lo[npanel] = (cur_lo == INT_MAX) ? 0 : cur_lo;
    pbb_hi[npanel] = (cur_hi < 0) ? 0 : cur_hi;
    npanel++;
    ptmp_lo[npanel] = nslice;

    double sw = 0.0; long long mw = 0;
    for (int p = 0; p < npanel; p++) {
        const long long w = (long long)pbb_hi[p] - pbb_lo[p];
        if (w > 0) sw += (double)w;
        if (w > mw) mw = w;
    }
    *sum_width = sw; *max_width = mw;
    return npanel;
}

static double page_x_bytes(int mode, double sum_width, int n)
{
    if (mode == PAGE_MODE_A && sum_width >= (double)n) return (double)n * 8.0;
    return sum_width * 8.0;
}

static double page_choose_window(const CSRMatrix *csr, const page_matrix *A,
                                  int *cw_lo_out, int *cw_hi_out)
{
    const int n  = csr->numCols;
    const int r0 = A->row_off, r1 = A->row_off + A->m_dsp;
    const int nb = (n + PAGE_CW_BLK - 1) / PAGE_CW_BLK;
    const int wb = PAGE_TILE_CAP / PAGE_CW_BLK;

    if (nb <= wb) { *cw_lo_out = 0; *cw_hi_out = n; return 1.0; }

    long long *h = (long long *)calloc((size_t)nb, sizeof(long long));

#pragma omp parallel
    {
        long long *hp = (long long *)calloc((size_t)nb, sizeof(long long));
#pragma omp for schedule(static)
        for (int i = r0; i < r1; i++)
            for (int k = csr->rowPointers[i]; k < csr->rowPointers[i + 1]; k++)
                hp[csr->colIndices[k] / PAGE_CW_BLK]++;
#pragma omp critical
        { for (int b = 0; b < nb; b++) h[b] += hp[b]; }
        free(hp);
    }

    long long tot = 0;
    for (int b = 0; b < nb; b++) tot += h[b];

    long long cur = 0;
    for (int b = 0; b < wb; b++) cur += h[b];
    long long best = cur; int best_b = 0;
    for (int b = wb; b < nb; b++) {
        cur += h[b] - h[b - wb];
        if (cur > best) { best = cur; best_b = b - wb + 1; }
    }
    free(h);

    int lo = best_b * PAGE_CW_BLK;
    int hi = lo + wb * PAGE_CW_BLK;
    if (hi > n) { hi = n; lo = (hi - PAGE_TILE_CAP > 0) ? hi - PAGE_TILE_CAP : 0; }
    *cw_lo_out = lo; *cw_hi_out = hi;
    return (tot > 0) ? (double)best / (double)tot : 1.0;
}

static void page_build_residual(const CSRMatrix *csr, page_matrix *A)
{
    const int r0 = A->row_off, md = A->m_dsp;
    const int lo = A->cw_lo, hi = A->cw_hi;
    A->res_rp = (int *)malloc((size_t)(md + 1) * sizeof(int));
    A->res_rp[0] = 0;

    int *cnt = (int *)malloc((size_t)md * sizeof(int));
#pragma omp parallel for schedule(dynamic, 4096)
    for (int i = 0; i < md; i++) {
        const int b = csr->rowPointers[r0 + i], e = csr->rowPointers[r0 + i + 1];
        int c = 0;
        for (int k = b; k < e; k++) {
            const int col = csr->colIndices[k];
            if (col < lo || col >= hi) c++;
        }
        cnt[i] = c;
    }
    for (int i = 0; i < md; i++) A->res_rp[i + 1] = A->res_rp[i] + cnt[i];
    free(cnt);

    const long long nz = A->res_rp[md];
    A->res_nnz = nz;
    A->res_ci  = (int *)malloc((size_t)(nz > 0 ? nz : 1) * sizeof(int));
    A->res_val = (MAT_VAL_TYPE *)malloc((size_t)(nz > 0 ? nz : 1) * sizeof(MAT_VAL_TYPE));

#pragma omp parallel for schedule(dynamic, 4096)
    for (int i = 0; i < md; i++) {
        const int b = csr->rowPointers[r0 + i], e = csr->rowPointers[r0 + i + 1];
        int w = A->res_rp[i];
        for (int k = b; k < e; k++) {
            const int col = csr->colIndices[k];
            if (col < lo || col >= hi) {
                A->res_ci[w] = col; A->res_val[w] = csr->values[k]; w++;
            }
        }
    }
}

static void page_plan(const CSRMatrix *csr, page_stat *st,
                       page_matrix *A, int coreNum, int cluster_id,
                       const page_opts *opt, int window_policy)
{
    const int nvrow  = A->nvrow;
    const int nslice = nvrow / SROW;
    const int nsblk  = (nvrow + SIGMA - 1) / SIGMA;

    A->coreNum = coreNum;
    A->dof_r = 1;
    A->idx_bytes = PAGE_IDX32_BYTES;

    int *blk_lo = (int *)malloc((size_t)nsblk * sizeof(int));
    int *blk_hi = (int *)malloc((size_t)nsblk * sizeof(int));
#pragma omp parallel for schedule(static)
    for (int b = 0; b < nsblk; b++) {
        const int u0 = b * SIGMA;
        const int u1 = (u0 + SIGMA < nvrow) ? u0 + SIGMA : nvrow;
        int lo = INT_MAX, hi = -1;
        for (int u = u0; u < u1; u++) {
            if (st->v_cmax[u] < 0) continue;
            if (st->v_cmin[u] < lo) lo = st->v_cmin[u];
            if (st->v_cmax[u] > hi) hi = st->v_cmax[u];
        }
        blk_lo[b] = (lo == INT_MAX) ? 0 : lo;
        blk_hi[b] = (hi < 0) ? 0 : hi + 1;
    }

    int blk_max_w = 0;
    for (int b = 0; b < nsblk; b++) {
        const int w = blk_hi[b] - blk_lo[b];
        if (w > blk_max_w) blk_max_w = w;
    }

    A->has_window = 0; A->cw_lo = 0; A->cw_hi = csr->numCols;
    if (window_policy && blk_max_w > PAGE_TILE_CAP) {
        int wlo, whi;
        (void)page_choose_window(csr, A, &wlo, &whi);
        {
            A->has_window = 1; A->cw_lo = wlo; A->cw_hi = whi;
            for (int b = 0; b < nsblk; b++) {
                int lo = blk_lo[b] > wlo ? blk_lo[b] : wlo;
                int hi = blk_hi[b] < whi ? blk_hi[b] : whi;
                if (hi <= lo) { lo = wlo; hi = wlo + 1; }
                blk_lo[b] = lo; blk_hi[b] = hi;
            }

#pragma omp parallel for schedule(dynamic, 256)
            for (int u = 0; u < nvrow; u++) {
                int kb, ke;
                page_vrange(csr, A, u, &kb, &ke);
                int cnt = 0, cl = INT_MAX, ch = -1;
                for (int k = kb; k < ke; k++) {
                    const int c = csr->colIndices[k];
                    if (c < wlo || c >= whi) continue;
                    cnt++;
                    if (c < cl) cl = c;
                    if (c > ch) ch = c;
                }
                st->v_len[u]  = cnt;
                st->v_cmin[u] = (ch < 0) ? 0 : cl;
                st->v_cmax[u] = ch;
            }
            for (int b = 0; b < nsblk; b++) {
                const int u0 = b * SIGMA;
                const int u1 = (u0 + SIGMA < nvrow) ? u0 + SIGMA : nvrow;
                int lo = INT_MAX, hi = -1;
                for (int u = u0; u < u1; u++) {
                    if (st->v_cmax[u] < 0) continue;
                    if (st->v_cmin[u] < lo) lo = st->v_cmin[u];
                    if (st->v_cmax[u] > hi) hi = st->v_cmax[u];
                }
                blk_lo[b] = (lo == INT_MAX) ? wlo : lo;
                blk_hi[b] = (hi < 0) ? wlo + 1 : hi + 1;
            }
            blk_max_w = 0;
            for (int b = 0; b < nsblk; b++) {
                const int w = blk_hi[b] - blk_lo[b];
                if (w > blk_max_w) blk_max_w = w;
            }
        }
    }

    A->rperm = (unsigned short *)malloc((size_t)nvrow * sizeof(unsigned short));
    const int n_sigma_blk = (nvrow + SIGMA - 1) / SIGMA;
    A->perm_blk = (unsigned char *)calloc((size_t)n_sigma_blk, sizeof(unsigned char));
    long long stored_id = 0, stored_sorted = 0;
    int ell_min = 0;
    page_sigma_sort(st->v_len, nvrow, A->m_dsp, A->idx_bytes, opt,
                     A->rperm, A->perm_blk, &A->nperm_blk,
                     &stored_id, &stored_sorted, &ell_min);

    A->perm_saved_elems = stored_id - stored_sorted;
    if (A->perm_saved_elems < 0) A->perm_saved_elems = 0;
    A->perm_est_save_ms = 0.0;
    A->perm_est_finalize_ms = 0.0;
    A->perm_guard_drop = 0;
    if (A->nperm_blk > 0 && opt) {
        const double pipe_bw = (opt->bw_pipe > 0.0) ? opt->bw_pipe : PAGE_BW_PIPE;
        A->perm_est_save_ms = ((double)A->perm_saved_elems * 16.0 / pipe_bw) * 1e3;

        long long perm_rows = (long long)A->nperm_blk * (long long)SIGMA;
        if (perm_rows > (long long)nvrow) perm_rows = (long long)nvrow;

        A->perm_est_finalize_ms = (16.0 * (double)perm_rows / opt->bw_arm) * 1e3;

        if (A->perm_est_save_ms < A->perm_est_finalize_ms) {
            for (int pos = 0; pos < nvrow; pos++)
                A->rperm[pos] = (unsigned short)(pos % SIGMA);
            memset(A->perm_blk, 0, (size_t)n_sigma_blk * sizeof(unsigned char));
            A->nperm_blk = 0;
            A->perm_guard_drop = 1;
        }
    }
    A->has_perm = (A->nperm_blk > 0) ? 1 : 0;

    if (A->has_perm) {
        A->rinv = (unsigned short *)malloc((size_t)nvrow * sizeof(unsigned short));
#pragma omp parallel for schedule(static)
        for (int pos = 0; pos < nvrow; pos++) {
            const int base = (pos / SIGMA) * SIGMA;
            A->rinv[base + (int)A->rperm[pos]] = (unsigned short)(pos - base);
        }
    } else {
        free(A->rperm); A->rperm = NULL;
        free(A->perm_blk); A->perm_blk = NULL;
        A->rinv = NULL;
    }

    const int slice_cap = coreNum * PANEL_SLICE;
    int panel_target = slice_cap;
    {
        const int np_est = (nslice + slice_cap - 1) / slice_cap;
        if (np_est > 1) {
            int t = (nslice + np_est - 1) / np_est;
            t = ((t + SLICE_PER_SIGMA - 1) / SLICE_PER_SIGMA) * SLICE_PER_SIGMA;
            if (t > slice_cap) t = slice_cap;
            if (t < SLICE_PER_SIGMA) t = SLICE_PER_SIGMA;
            panel_target = t;
        }
    }

    int *ptmp_lo = (int *)malloc((size_t)(nsblk + 2) * sizeof(int));
    int *pbb_lo  = (int *)malloc((size_t)(nsblk + 2) * sizeof(int));
    int *pbb_hi  = (int *)malloc((size_t)(nsblk + 2) * sizeof(int));

    double sum_w = 0.0; long long max_w = 0;
    int npanel = page_panels_try(blk_lo, blk_hi, nsblk, nslice, panel_target,
                                  PAGE_TILE_CAP, ptmp_lo, pbb_lo, pbb_hi,
                                  &sum_w, &max_w);

    if (coreNum > 1) {
        const double kfma = C_FMA + C_SG / (double)A->dof_r;
        const double c_mean = ((double)stored_sorted / (double)(nslice > 0 ? nslice : 1)
                               / (double)SROW) * kfma + C_SLICE;
        const double c_min  = (double)ell_min * kfma + C_SLICE;
        long long cap_target = (long long)((double)coreNum * PANEL_SLICE * c_min / c_mean);

        long long floor_t = (long long)coreNum * PANEL_SLICE / 8;
        if (floor_t < (long long)coreNum * SLICE_PER_SIGMA)
            floor_t = (long long)coreNum * SLICE_PER_SIGMA;
        if (cap_target < floor_t) cap_target = floor_t;
        if (cap_target < panel_target) {
            int t2 = (int)cap_target;
            t2 = (t2 / SLICE_PER_SIGMA) * SLICE_PER_SIGMA;
            if (t2 < SLICE_PER_SIGMA) t2 = SLICE_PER_SIGMA;
            double sw2; long long mw2;
            const int np2 = page_panels_try(blk_lo, blk_hi, nsblk, nslice, t2,
                                             PAGE_TILE_CAP, ptmp_lo, pbb_lo, pbb_hi,
                                             &sw2, &mw2);

            const double val_bytes = (double)A->nnz_dsp * (8.0 + (double)A->idx_bytes);
            const double dx = page_x_bytes(st->mode, sw2, csr->numCols)
                            - page_x_bytes(st->mode, sum_w, csr->numCols);
            if (dx <= PAGE_XRELOAD_BUDGET * val_bytes) {
                npanel = np2; sum_w = sw2; max_w = mw2; panel_target = t2;
            } else {
                page_panels_try(blk_lo, blk_hi, nsblk, nslice, panel_target,
                                 PAGE_TILE_CAP, ptmp_lo, pbb_lo, pbb_hi,
                                 &sum_w, &max_w);
            }
        }
    }

    if (max_w > PAGE_TILE_CAP && blk_max_w <= PAGE_TILE_CAP) {
        const double val_bytes = (double)A->nnz_dsp * (8.0 + (double)A->idx_bytes);
        const double budget    = PAGE_XRELOAD_BUDGET * val_bytes;
        int t = panel_target;
        while (t > PAGE_PANEL_MIN_SLICE) {
            int t2 = t / 2;
            t2 = ((t2 + SLICE_PER_SIGMA - 1) / SLICE_PER_SIGMA) * SLICE_PER_SIGMA;
            if (t2 < SLICE_PER_SIGMA) t2 = SLICE_PER_SIGMA;
            if (t2 >= t) break;
            double sw2; long long mw2;
            const int np2 = page_panels_try(blk_lo, blk_hi, nsblk, nslice, t2,
                                             PAGE_TILE_CAP, ptmp_lo, pbb_lo, pbb_hi,
                                             &sw2, &mw2);
            if (page_x_bytes(st->mode, sw2, csr->numCols) > budget) {
                npanel = page_panels_try(blk_lo, blk_hi, nsblk, nslice, panel_target,
                                          PAGE_TILE_CAP, ptmp_lo, pbb_lo, pbb_hi,
                                          &sum_w, &max_w);
                break;
            }
            npanel = np2; sum_w = sw2; max_w = mw2; t = t2;
            panel_target = t2;
            if (mw2 <= PAGE_TILE_CAP) break;
        }
    }

    {
        const double val_bytes = (double)A->nnz_dsp * (8.0 + (double)A->idx_bytes);
        const double budget    = PAGE_XIN_BUDGET * val_bytes;
        int guard = 0;
        while (npanel > 1 && panel_target < slice_cap && guard++ < 32 &&
               page_x_bytes(st->mode, sum_w, csr->numCols) > budget) {
            int t2 = panel_target * 2;
            if (t2 > slice_cap || t2 <= panel_target) t2 = slice_cap;
            t2 = (t2 / SLICE_PER_SIGMA) * SLICE_PER_SIGMA;
            if (t2 < SLICE_PER_SIGMA) t2 = SLICE_PER_SIGMA;
            if (t2 <= panel_target) break;
            panel_target = t2;
            npanel = page_panels_try(blk_lo, blk_hi, nsblk, nslice, panel_target,
                                      PAGE_TILE_CAP, ptmp_lo, pbb_lo, pbb_hi,
                                      &sum_w, &max_w);
        }
    }

    if (st->mode == PAGE_MODE_A && npanel > 1) {
        long long local_x = 0;
        for (int p = 0; p < npanel; p++) local_x += (long long)pbb_hi[p] - pbb_lo[p];
        if (local_x >= (long long)csr->numCols) {
            for (int p = 0; p < npanel; p++) { pbb_lo[p] = 0; pbb_hi[p] = csr->numCols; }
        }
    }

    if (A->has_window) {
        for (int p = 0; p < npanel; p++) { pbb_lo[p] = A->cw_lo; pbb_hi[p] = A->cw_hi; }
    }

    int *panel_pack_n = (int *)calloc((size_t)npanel, sizeof(int));
    int **panel_pack_cols = (int **)calloc((size_t)npanel, sizeof(int *));
    int *uniq_tmp = (int *)malloc((size_t)(PAGE_XBLOCK_CAP + 1) * sizeof(int));
    int *pack_keys = NULL;
    unsigned int *pack_tags = NULL;
    long long total_pack_cols = 0;
    const char *pack_env = getenv("PAGE_PACKED_X");
    const int allow_packed_x = !(pack_env && strcmp(pack_env, "0") == 0);
    int need_pack_probe = 0;
    if (allow_packed_x && !A->has_window) {
        for (int p = 0; p < npanel; p++)
            if (pbb_hi[p] - pbb_lo[p] > PAGE_TILE_CAP) { need_pack_probe = 1; break; }
    }
    if (need_pack_probe && uniq_tmp) {
        pack_keys = (int *)malloc((size_t)PAGE_PACK_HASH_SIZE * sizeof(int));
        pack_tags = (unsigned int *)calloc((size_t)PAGE_PACK_HASH_SIZE, sizeof(unsigned int));
    }
    if (need_pack_probe && uniq_tmp && pack_keys && pack_tags) {
        unsigned int generation = 1;
        for (int p = 0; p < npanel; p++) {
            const int span = pbb_hi[p] - pbb_lo[p];
            if (span <= PAGE_TILE_CAP) continue;
            const int nu = page_panel_unique_blocks(csr, A, ptmp_lo[p], ptmp_lo[p + 1], uniq_tmp,
                                                   pack_keys, pack_tags, generation++);
            const int nr = (nu > 0 && nu <= PAGE_XBLOCK_CAP) ? page_count_block_runs(uniq_tmp, nu) : 0;
            if (nu > 0 && nu <= PAGE_XBLOCK_CAP &&
                page_pack_profitable(span, nu, nr, coreNum, opt) &&
                total_pack_cols + (long long)nu <= (long long)INT_MAX) {
                panel_pack_n[p] = nu;
                panel_pack_cols[p] = (int *)malloc((size_t)nu * sizeof(int));
                if (panel_pack_cols[p]) {
                    memcpy(panel_pack_cols[p], uniq_tmp, (size_t)nu * sizeof(int));
                    total_pack_cols += nu;
                } else {
                    panel_pack_n[p] = 0;
                }
            }
        }
    }
    free(pack_tags); free(pack_keys); free(uniq_tmp);

    A->npanel = npanel;
    A->panel_slice_lo = (int *)PAGE_DEV_MALLOC(cluster_id, (size_t)(npanel + 1) * sizeof(int));
    A->panel_tile_lo  = (int *)PAGE_DEV_MALLOC(cluster_id, (size_t)(npanel + 1) * sizeof(int));
    memcpy(A->panel_slice_lo, ptmp_lo, (size_t)(npanel + 1) * sizeof(int));

    int ntile = 0;
    for (int p = 0; p < npanel; p++) {
        const int w = pbb_hi[p] - pbb_lo[p];
        if (panel_pack_n[p] > 0) ntile += 1;
        else ntile += (w <= PAGE_TILE_CAP) ? 1 : (w + PAGE_TILE_CAP - 1) / PAGE_TILE_CAP;
    }
    A->ntile = ntile;
    A->tile_xlo       = (int *)PAGE_DEV_MALLOC(cluster_id, (size_t)ntile * sizeof(int));
    A->tile_xhi       = (int *)PAGE_DEV_MALLOC(cluster_id, (size_t)ntile * sizeof(int));
    A->tile_inst_lo   = (int *)PAGE_DEV_MALLOC(cluster_id, (size_t)(ntile + 1) * sizeof(int));
    A->tile_packed    = (unsigned char *)PAGE_DEV_MALLOC(cluster_id, (size_t)ntile * sizeof(unsigned char));
    A->tile_xmap_lo   = (int *)PAGE_DEV_MALLOC(cluster_id, (size_t)(ntile + 1) * sizeof(int));
    A->tile_xmap      = total_pack_cols > 0
                      ? (int *)PAGE_DEV_MALLOC(cluster_id, (size_t)total_pack_cols * sizeof(int))
                      : NULL;
    A->packed_tiles = 0; A->packed_x_elems = 0; A->packed_runs = 0; A->packed_dma_rounds = 0;

    int t = 0, mapw = 0, mode = st->mode;
    for (int p = 0; p < npanel; p++) {
        A->panel_tile_lo[p] = t;
        const int lo = pbb_lo[p], hi = pbb_hi[p], w = hi - lo;
        if (panel_pack_n[p] > 0) {
            const int nu = panel_pack_n[p];
            A->tile_xlo[t] = lo; A->tile_xhi[t] = hi;
            A->tile_packed[t] = PAGE_TILE_PACKED;
            A->tile_xmap_lo[t] = mapw;
            for (int j = 0; j < nu; j++)
                A->tile_xmap[mapw + j] = panel_pack_cols[p][j] * PAGE_XBLOCK_BYTES;
            A->packed_runs += page_count_block_runs(panel_pack_cols[p], nu);
            mapw += nu;
            A->packed_tiles++; A->packed_x_elems += (long long)nu * PAGE_XBLOCK_ELEMS;
            t++;
        } else if (w <= PAGE_TILE_CAP) {
            A->tile_xlo[t] = lo;
            A->tile_xhi[t] = (hi > lo) ? hi : lo + 1;
            A->tile_packed[t] = 0;
            A->tile_xmap_lo[t] = mapw;
            t++;
        } else {
            mode = PAGE_MODE_C;
            for (int c0 = lo; c0 < hi; c0 += PAGE_TILE_CAP) {
                A->tile_xlo[t] = c0;
                A->tile_xhi[t] = (c0 + PAGE_TILE_CAP < hi) ? c0 + PAGE_TILE_CAP : hi;
                A->tile_packed[t] = 0;
                A->tile_xmap_lo[t] = mapw;
                t++;
            }
        }
    }
    A->panel_tile_lo[npanel] = t;
    A->tile_xmap_lo[ntile] = mapw;
    A->mode = mode;
    st->mode = mode;

    for (int p = 0; p < npanel; p++) free(panel_pack_cols[p]);
    free(panel_pack_cols); free(panel_pack_n);

    int ninst = 0;
    for (int p = 0; p < npanel; p++) {
        const int ns = A->panel_slice_lo[p + 1] - A->panel_slice_lo[p];
        ninst += ns * (A->panel_tile_lo[p + 1] - A->panel_tile_lo[p]);
    }
    A->ninst = ninst;
    /* Reserve cl[ie] for the DSP prefetch at the final instruction. */
    A->cl = (int *)PAGE_DEV_MALLOC(cluster_id, (size_t)(ninst + 1) * sizeof(int));
    A->cs = (int *)PAGE_DEV_MALLOC(cluster_id, (size_t)(ninst + 1) * sizeof(int));
    A->cl[ninst] = 0;

    int inst = 0;
    for (int p = 0; p < npanel; p++) {
        const int s0 = A->panel_slice_lo[p], s1 = A->panel_slice_lo[p + 1];
        const int t0 = A->panel_tile_lo[p],  t1 = A->panel_tile_lo[p + 1];

        const int single = (t1 - t0 == 1) && !A->has_window;
        for (int tt = t0; tt < t1; tt++) {
            A->tile_inst_lo[tt] = inst;
            const int xlo = A->tile_xlo[tt], xhi = A->tile_xhi[tt];
            const int base = inst;
#pragma omp parallel for schedule(dynamic, 64)
            for (int rs = s0; rs < s1; rs++)
                A->cl[base + (rs - s0)] =
                    page_slice_width(csr, st, A, rs, xlo, xhi, single);
            inst += (s1 - s0);
        }
    }
    A->tile_inst_lo[ntile] = inst;

    A->cs[0] = 0;
    for (int i = 0; i < ninst; i++) A->cs[i + 1] = A->cs[i] + A->cl[i] * SROW;

    A->x_elems = 0;
    int max_local_x = 0;
    for (int tt = 0; tt < ntile; tt++) {
        int width;
        if (A->tile_packed[tt]) {
            width = (A->tile_xmap_lo[tt + 1] - A->tile_xmap_lo[tt]) * PAGE_XBLOCK_ELEMS;
        } else {
            width = A->tile_xhi[tt] - A->tile_xlo[tt];
            if (tt > 0 && !A->tile_packed[tt - 1] &&
                A->tile_xlo[tt] == A->tile_xlo[tt - 1] &&
                A->tile_xhi[tt] == A->tile_xhi[tt - 1]) {
                if (width > max_local_x) max_local_x = width;
                continue;
            }
        }
        A->x_elems += (long long)width;
        if (width > max_local_x) max_local_x = width;
    }
    /* Implicit values reserve a zero slot; 16-bit indices allow 65535 real entries. */
    const int idx16_limit = (A->dsp_value_mode == PAGE_VALUE_GENERAL) ? 65536 : 65535;
    A->idx_bytes = opt->force_idx32
                 ? PAGE_IDX32_BYTES
                 : ((max_local_x <= idx16_limit) ? PAGE_IDX16_BYTES : PAGE_IDX32_BYTES);

    if (A->has_window) {
        long long inwin = 0;
        const int r0 = A->row_off, r1 = A->row_off + A->m_dsp;
        const int wlo = A->cw_lo, whi = A->cw_hi;
#pragma omp parallel for schedule(dynamic, 4096) reduction(+:inwin)
        for (int i = r0; i < r1; i++)
            for (int k = csr->rowPointers[i]; k < csr->rowPointers[i + 1]; k++) {
                const int c = csr->colIndices[k];
                if (c >= wlo && c < whi) inwin++;
            }
        A->nnz_dsp = inwin;
    }

    A->stored    = (long long)A->cs[ninst];
    A->pad_zeros = A->stored - A->nnz_dsp;
    A->z_ratio   = (A->nnz_dsp > 0) ? (double)A->pad_zeros / (double)A->nnz_dsp : 0.0;

    A->thread_ptr = (int *)PAGE_DEV_MALLOC(cluster_id,
                        (size_t)npanel * (coreNum + 1) * sizeof(int));
    {
        const double kfma = C_FMA + C_SG / (double)A->dof_r;

        double worst = 0.0;

        for (int p = 0; p < npanel; p++) {
            const int s0 = A->panel_slice_lo[p], s1 = A->panel_slice_lo[p + 1];
            const int ns = s1 - s0;
            const int t0 = A->panel_tile_lo[p], t1 = A->panel_tile_lo[p + 1];
            int *bnd = &A->thread_ptr[p * (coreNum + 1)];

            double *W = (double *)malloc((size_t)(ns + 1) * sizeof(double));
            W[0] = 0.0;
            for (int rs = 0; rs < ns; rs++) {
                double c = 0.0;
                for (int tt = t0; tt < t1; tt++)
                    c += (double)A->cl[A->tile_inst_lo[tt] + rs] * kfma + C_SLICE;
                W[rs + 1] = W[rs] + c;
            }
            const double per = W[ns] / (double)coreNum;

            bnd[0] = 0;
            for (int tid = 1; tid < coreNum; tid++) {
                const double bound = per * (double)tid;
                int lo = 0, hi = ns;
                while (lo < hi) {
                    const int mid = (lo + hi) >> 1;
                    if (W[mid] < bound) lo = mid + 1; else hi = mid;
                }
                if (lo > 0 && (W[lo] - bound) >= (bound - W[lo - 1])) lo--;
                if (lo < bnd[tid - 1]) lo = bnd[tid - 1];
                bnd[tid] = lo;
            }
            bnd[coreNum] = ns;

            for (int tid = 0; tid < coreNum; tid++)
                if (bnd[tid + 1] - bnd[tid] > PANEL_SLICE)
                    bnd[tid + 1] = bnd[tid] + PANEL_SLICE;
            bnd[coreNum] = ns;
            for (int tid = coreNum; tid >= 1; tid--)
                if (bnd[tid] - bnd[tid - 1] > PANEL_SLICE)
                    bnd[tid - 1] = bnd[tid] - PANEL_SLICE;
            bnd[0] = 0;
            for (int tid = 1; tid <= coreNum; tid++)
                if (bnd[tid] < bnd[tid - 1]) bnd[tid] = bnd[tid - 1];

            {
                double pmax = 0.0, psum = 0.0;
                for (int tid = 0; tid < coreNum; tid++) {
                    double c = 0.0;
                    for (int rs = bnd[tid]; rs < bnd[tid + 1]; rs++)
                        for (int tt = t0; tt < t1; tt++)
                            c += (double)A->cl[A->tile_inst_lo[tt] + rs] * kfma + C_SLICE;
                    if (c > pmax) pmax = c;
                    psum += c;
                }
                const double pavg = psum / (double)coreNum;
                if (pavg > 0.0) {
                    const double r = pmax / pavg - 1.0;
                    if (r > worst) worst = r;
                    if (getenv("PAGE_DEBUG_LB"))
                        fprintf(stderr, "[LB] panel %d ns=%d pmax=%.0f pavg=%.0f r=%.3f slices/core=%d..%d\n",
                                p, ns, pmax, pavg, r, bnd[1]-bnd[0], bnd[coreNum]-bnd[coreNum-1]);
                }
            }
            for (int tid = 0; tid <= coreNum; tid++) bnd[tid] += s0;
            free(W);
        }
        A->lb_imbalance = worst;
    }

    A->gsmxwidth = (int *)PAGE_DEV_MALLOC(cluster_id,
                        (size_t)ntile * (coreNum + 1) * sizeof(int));
    for (int tt = 0; tt < ntile; tt++) {
        const int width = A->tile_packed[tt]
                        ? (A->tile_xmap_lo[tt + 1] - A->tile_xmap_lo[tt])
                        : (A->tile_xhi[tt] - A->tile_xlo[tt]);
        const int part  = (width + coreNum - 1) / coreNum;
        int *g = &A->gsmxwidth[tt * (coreNum + 1)];
        g[0] = 0;
        for (int j = 0; j < coreNum; j++) {
            const int done = j * part;
            int pw = 0;
            if (done < width) pw = (done + part > width) ? (width - done) : part;
            g[j + 1] = g[j] + pw;
        }
        if (A->tile_packed[tt]) {
            const int ml = A->tile_xmap_lo[tt];
            int tile_crit = 0;
            for (int tid = 0; tid < coreNum; tid++) {
                const int b0 = g[tid], b1 = g[tid + 1];
                int calls = 0;
                if (b1 > b0) {
                    calls = 1;
                    for (int b = b0 + 1; b < b1; b++)
                        if (A->tile_xmap[ml + b] != A->tile_xmap[ml + b - 1] + PAGE_XBLOCK_BYTES)
                            calls++;
                }
                if (calls > tile_crit) tile_crit = calls;
            }
            A->packed_dma_rounds += tile_crit;
        }
    }

    free(blk_lo); free(blk_hi); free(ptmp_lo); free(pbb_lo); free(pbb_hi);
}

static void page_pass_b(const CSRMatrix *csr, const page_stat *st,
                         page_matrix *A, int cluster_id)
{
    const long long stored = A->stored;
    A->val  = (MAT_VAL_TYPE *)PAGE_DEV_MALLOC(cluster_id,
                  (size_t)stored * sizeof(MAT_VAL_TYPE));
    A->cidx = PAGE_DEV_MALLOC(cluster_id,
                  (size_t)stored * (size_t)A->idx_bytes);
    const int nvrow = A->nvrow;

    for (int p = 0; p < A->npanel; p++) {
        const int s0 = A->panel_slice_lo[p], s1 = A->panel_slice_lo[p + 1];
        const int t0 = A->panel_tile_lo[p],  t1 = A->panel_tile_lo[p + 1];
        const int single = (t1 - t0 == 1) && !A->has_window;
        for (int tt = t0; tt < t1; tt++) {
            const int xlo = A->tile_xlo[tt], xhi = A->tile_xhi[tt];
            const int ibase = A->tile_inst_lo[tt];
            const int packed = A->tile_packed ? (int)A->tile_packed[tt] : 0;
            const int map_lo = A->tile_xmap_lo ? A->tile_xmap_lo[tt] : 0;
            const int map_hi = A->tile_xmap_lo ? A->tile_xmap_lo[tt + 1] : 0;
            const int zero_local = packed
                ? (map_hi - map_lo) * PAGE_XBLOCK_ELEMS
                : (xhi - xlo);

#pragma omp parallel for schedule(dynamic, 32)
            for (int rs = s0; rs < s1; rs++) {
                const int inst = ibase + (rs - s0);
                const int w = A->cl[inst];
                if (w == 0) continue;
                const long long off = A->cs[inst];

                for (int i = 0; i < SROW; i++) {
                    const int pos = rs * SROW + i;
                    int j = 0;
                    if (pos < nvrow) {
                        const int u = page_vid(A, pos);
                        int kb, ke;
                        page_vrange(csr, A, u, &kb, &ke);

                        if (!single && !packed && ke > kb) {
                            const int gr = page_vrow_global(A, u);
                            if (gr >= 0 && st->row_sorted[gr]) {
                                const int b = page_lower_bound(csr->colIndices, kb, ke, xlo);
                                const int e = page_lower_bound(csr->colIndices, b, ke, xhi);
                                kb = b; ke = e;
                            }
                        }
                        for (int k = kb; k < ke; k++) {
                            const int c = csr->colIndices[k];
                            int local = -1;
                            if (packed) {
                                const int boff = (c / PAGE_XBLOCK_ELEMS) * PAGE_XBLOCK_BYTES;
                                const int q = page_lower_bound_byteoff(A->tile_xmap, map_lo, map_hi, boff);
                                if (q >= map_hi || A->tile_xmap[q] != boff) continue;
                                local = (q - map_lo) * PAGE_XBLOCK_ELEMS
                                      + (c % PAGE_XBLOCK_ELEMS);
                            } else {
                                if (!single && (c < xlo || c >= xhi)) continue;
                                local = c - xlo;
                            }
                            const long long dst = off + (long long)j * SROW + i;
                            A->val[dst] = csr->values[k];
                            if (A->idx_bytes == PAGE_IDX16_BYTES) {
                                if ((unsigned)local > 65535u) {
                                    fprintf(stderr,
                                        "[PAGE FATAL] runtime IDX16 overflow: local=%d tile=%d\n",
                                        local, tt);
                                    exit(2);
                                }
                                ((page_idx16_t *)A->cidx)[dst] = (page_idx16_t)local;
                            } else {
                                ((page_idx32_t *)A->cidx)[dst] =
                                    (page_idx32_t)(local * (int)sizeof(MAT_VAL_TYPE));
                            }
                            j++;
                        }
                    }
                    for (; j < w; j++) {
                        const long long dst = off + (long long)j * SROW + i;
                        A->val[dst] = 0.0;
                        /* Implicit-value padding gathers from the reserved zero slot. */
                        const int pad_local = (A->dsp_value_mode == PAGE_VALUE_GENERAL)
                                            ? 0 : zero_local;
                        if (A->idx_bytes == PAGE_IDX16_BYTES) {
                            if ((unsigned)pad_local > 65535u) {
                                fprintf(stderr,
                                    "[PAGE FATAL] implicit IDX16 zero-sentinel overflow: local=%d tile=%d\n",
                                    pad_local, tt);
                                exit(2);
                            }
                            ((page_idx16_t *)A->cidx)[dst] = (page_idx16_t)pad_local;
                        } else {
                            ((page_idx32_t *)A->cidx)[dst] =
                                (page_idx32_t)(pad_local * (int)sizeof(MAT_VAL_TYPE));
                        }
                    }
                }
            }
        }
    }
}

static double page_est_dsp_ms_bw(const page_matrix *A, const page_opts *o,
                                  double bw_pipe)
{
    if (!(bw_pipe > 0.0)) bw_pipe = (o->bw_pipe > 0.0) ? o->bw_pipe : PAGE_BW_PIPE;
    const double value_bytes = (A->dsp_value_mode == PAGE_VALUE_GENERAL) ? 8.0 : 0.0;
    const double pair_bytes = (double)A->stored * (value_bytes + 8.0);
    const double meta_ddr = (double)A->stored * (double)A->idx_bytes
                          + (double)(A->packed_x_elems / PAGE_XBLOCK_ELEMS) * 4.0
                          + (double)A->nvrow * 8.0;
    const double xbw = (o->bw_xload > 0.0) ? o->bw_xload : o->bw_ddr;
    const double tp = pair_bytes / bw_pipe;
    const double tm = (o->bw_ddr > 0.0) ? meta_ddr / o->bw_ddr : 0.0;
    const double tx = (xbw > 0.0) ? ((double)A->x_elems * 8.0) / xbw : 0.0;
    const double te = tm + tx + (double)A->packed_dma_rounds * o->xload_fixed_ms * 1e-3;
    const double launch = (o->launch_ms >= 0.0) ? o->launch_ms : PAGE_KERNEL_FIX_MS;
    const double barr   = (o->barrier_ms >= 0.0) ? o->barrier_ms : PAGE_BARRIER_FIX_MS;
    return (tp + te) * 1e3 + launch + 2.0 * (double)A->ntile * barr;
}

static double page_est_dsp_ms(const page_matrix *A, const page_opts *o)
{
    return page_est_dsp_ms_bw(A, o, o->bw_pipe);
}

static double page_est_arm_ms(long long nnz, int m, int n, const page_opts *o)
{
    const double bytes = (double)nnz * 12.0 + (double)m * 8.0 + (double)n * 8.0;
    return bytes / o->bw_arm * 1e3;
}

static double page_est_unique_cols(long long nnz, int n)
{
    if (nnz <= 0 || n <= 0) return 0.0;
    const double x = (double)nnz / (double)n;
    double u = (double)n * (1.0 - exp(-x));
    if (u > (double)n) u = (double)n;
    if (u > (double)nnz) u = (double)nnz;
    return u;
}

static double page_est_active_rows(long long nnz, int m)
{
    if (nnz <= 0 || m <= 0) return 0.0;
    const double x = (double)nnz / (double)m;
    double r = (double)m * (1.0 - exp(-x));
    if (r > (double)m) r = (double)m;
    if (r > (double)nnz) r = (double)nnz;
    return r;
}

static void page_free_plan_fields(page_matrix *A)
{
    PAGE_DEV_FREE(A->val);            A->val = NULL;
    PAGE_DEV_FREE(A->cidx);           A->cidx = NULL;
    PAGE_DEV_FREE(A->cl);             A->cl = NULL;
    PAGE_DEV_FREE(A->cs);             A->cs = NULL;
    PAGE_DEV_FREE(A->thread_ptr);     A->thread_ptr = NULL;
    PAGE_DEV_FREE(A->gsmxwidth);      A->gsmxwidth = NULL;
    PAGE_DEV_FREE(A->tile_xlo);       A->tile_xlo = NULL;
    PAGE_DEV_FREE(A->tile_xhi);       A->tile_xhi = NULL;
    PAGE_DEV_FREE(A->tile_inst_lo);   A->tile_inst_lo = NULL;
    PAGE_DEV_FREE(A->tile_packed);    A->tile_packed = NULL;
    PAGE_DEV_FREE(A->tile_xmap_lo);   A->tile_xmap_lo = NULL;
    PAGE_DEV_FREE(A->tile_xmap);      A->tile_xmap = NULL;
    PAGE_DEV_FREE(A->panel_slice_lo); A->panel_slice_lo = NULL;
    PAGE_DEV_FREE(A->panel_tile_lo);  A->panel_tile_lo = NULL;
    free(A->rperm);   A->rperm = NULL;
    free(A->rinv);    A->rinv = NULL;
    free(A->perm_blk); A->perm_blk = NULL;
    free(A->res_rp);  A->res_rp = NULL;
    free(A->res_ci);  A->res_ci = NULL;
    free(A->res_val); A->res_val = NULL;

    A->npanel = A->ntile = A->ninst = 0;
    A->stored = A->pad_zeros = A->x_elems = 0;
    A->packed_tiles = 0; A->packed_x_elems = 0; A->packed_runs = 0; A->packed_dma_rounds = 0;
    A->z_ratio = A->lb_imbalance = A->bytes_per_nnz = 0.0;
    A->has_perm = A->nperm_blk = 0;
    A->perm_saved_elems = 0;
    A->perm_est_save_ms = A->perm_est_finalize_ms = 0.0;
    A->perm_guard_drop = 0;
    A->has_window = 0; A->cw_lo = 0; A->cw_hi = A->n;
    A->res_nnz = 0;
}

static void page_restore_vstats(page_stat *st, const int *vlen,
                                 const int *vcmin, const int *vcmax, int nvrow)
{
    memcpy(st->v_len,  vlen,  (size_t)nvrow * sizeof(int));
    memcpy(st->v_cmin, vcmin, (size_t)nvrow * sizeof(int));
    memcpy(st->v_cmax, vcmax, (size_t)nvrow * sizeof(int));
}

static int page_window_structurally_possible(const page_stat *st,
                                               const page_matrix *A)
{
    if (!st || !A || A->nvrow <= 0 || !st->v_cmin || !st->v_cmax) return 0;
    const int nsblk = (A->nvrow + SIGMA - 1) / SIGMA;
    int blk_max_w = 0;
#pragma omp parallel for schedule(static) reduction(max:blk_max_w)
    for (int b = 0; b < nsblk; b++) {
        const int u0 = b * SIGMA;
        const int u1 = (u0 + SIGMA < A->nvrow) ? u0 + SIGMA : A->nvrow;
        int lo = INT_MAX, hi = -1;
        for (int u = u0; u < u1; u++) {
            if (st->v_cmax[u] < 0) continue;
            if (st->v_cmin[u] < lo) lo = st->v_cmin[u];
            if (st->v_cmax[u] > hi) hi = st->v_cmax[u];
        }
        const int w = (hi < 0 || lo == INT_MAX) ? 0 : (hi + 1 - lo);
        if (w > blk_max_w) blk_max_w = w;
    }
    return blk_max_w > PAGE_TILE_CAP;
}

static double page_est_plan_total_ms(const CSRMatrix *csr, const page_stat *st,
                                      const page_matrix *A, const page_opts *o)
{
    const int r0 = A->row_off;
    const int r1 = A->row_off + A->m_dsp;
    const long long prefix_nnz = st->row_pre[r0];
    const long long row_nnz = st->row_pre[r1] - st->row_pre[r0];
    long long residual_nnz = A->has_window ? (row_nnz - A->nnz_dsp) : 0;
    if (residual_nnz < 0) residual_nnz = 0;

    const long long arm_nnz = prefix_nnz + residual_nnz;
    const double res_rows = page_est_active_rows(residual_nnz, A->m_dsp);
    const double arm_rows = (double)r0 + res_rows;
    const double arm_x = page_est_unique_cols(arm_nnz, csr->numCols);
    const double arm_bytes = (double)arm_nnz * 12.0
                           + arm_rows * 8.0 + arm_x * 8.0;

    const int concurrent = (arm_bytes > 0.0);
    const double pipe_bw = concurrent && o->bw_pipe_mix > 0.0
                         ? o->bw_pipe_mix : o->bw_pipe;
    const double arm_bw = concurrent && o->bw_arm_mix > 0.0
                        ? o->bw_arm_mix : o->bw_arm;
    const double td = page_est_dsp_ms_bw(A, o, pipe_bw);
    const double ta = (arm_bytes > 0.0) ? arm_bytes / arm_bw * 1e3 : 0.0;

    const double tf_perm = A->perm_est_finalize_ms;
    const double tf_ov = (A->n_ov > 0)
        ? 24.0 * (double)A->n_ov / o->bw_arm * 1e3 : 0.0;
    const double tf_res = (res_rows > 0.0)
        ? 24.0 * res_rows / o->bw_arm * 1e3 : 0.0;
    const double tf = tf_perm + tf_ov + tf_res;

    double t = td;
    if (ta > t) t = ta;
    return t + tf;
}

static void page_plan_adaptive(const CSRMatrix *csr, page_stat *st,
                                page_matrix *A, int coreNum, int cluster_id,
                                const page_opts *o)
{
    const int nvrow = A->nvrow;
    const long long row_nnz = st->row_pre[A->row_off + A->m_dsp] - st->row_pre[A->row_off];
    const int base_mode = (csr->numCols <= GSM_X_CAP) ? PAGE_MODE_A : PAGE_MODE_B;

#if !PAGE_FINAL_COLUMN_WINDOW
    st->mode = base_mode;
    A->nnz_dsp = row_nnz;
    page_plan(csr, st, A, coreNum, cluster_id, o, 0);
    A->model_tiled_ms = page_est_plan_total_ms(csr, st, A, o);
    A->model_window_ms = -1.0;
    A->model_total_ms = A->model_tiled_ms;
    A->model_bw_mix = o->bw_mix;
    return;
#endif

    if (o->fast_shadow_precheck && o->plan_policy != 0 &&
        !page_window_structurally_possible(st, A)) {
        printf("[V19_SHADOW_GATE] row_off=%d m_dsp=%d nvrow=%d plan_policy=%d window_possible=0 action=single_tiled_exact\n",
               A->row_off, A->m_dsp, A->nvrow, o->plan_policy);
        st->mode = base_mode;
        A->nnz_dsp = row_nnz;
        page_plan(csr, st, A, coreNum, cluster_id, o, 0);
        A->model_tiled_ms = page_est_plan_total_ms(csr, st, A, o);
        A->model_window_ms = -1.0;
        A->model_total_ms = A->model_tiled_ms;
        A->model_bw_mix = o->bw_mix;
        return;
    }

    if (o->fast_shadow_precheck && o->plan_policy == 0) {
        st->mode = base_mode;
        A->nnz_dsp = row_nnz;
        page_plan(csr, st, A, coreNum, cluster_id, o, 0);
        A->model_tiled_ms = page_est_plan_total_ms(csr, st, A, o);
        A->model_window_ms = -1.0;
        A->model_total_ms = A->model_tiled_ms;
        A->model_bw_mix = o->bw_mix;
        return;
    }

    int *vlen0 = (int *)malloc((size_t)nvrow * sizeof(int));
    int *vcmin0 = (int *)malloc((size_t)nvrow * sizeof(int));
    int *vcmax0 = (int *)malloc((size_t)nvrow * sizeof(int));
    memcpy(vlen0, st->v_len, (size_t)nvrow * sizeof(int));
    memcpy(vcmin0, st->v_cmin, (size_t)nvrow * sizeof(int));
    memcpy(vcmax0, st->v_cmax, (size_t)nvrow * sizeof(int));

    if (o->plan_policy == 1) {
        st->mode = base_mode;
        A->nnz_dsp = row_nnz;
        page_plan(csr, st, A, coreNum, cluster_id, o, 1);
        if (A->has_window) {
            A->model_window_ms = page_est_plan_total_ms(csr, st, A, o);
            A->model_tiled_ms = -1.0;
            A->model_total_ms = A->model_window_ms;
            A->model_bw_mix = o->bw_mix;
            free(vlen0); free(vcmin0); free(vcmax0);
            return;
        }

        page_free_plan_fields(A);
        page_restore_vstats(st, vlen0, vcmin0, vcmax0, nvrow);
        st->mode = base_mode;
        A->nnz_dsp = row_nnz;
        page_plan(csr, st, A, coreNum, cluster_id, o, 0);
        A->model_tiled_ms = page_est_plan_total_ms(csr, st, A, o);
        A->model_window_ms = -1.0;
        A->model_total_ms = A->model_tiled_ms;
        A->model_bw_mix = o->bw_mix;
        free(vlen0); free(vcmin0); free(vcmax0);
        return;
    }

    st->mode = base_mode;
    A->nnz_dsp = row_nnz;
    page_plan(csr, st, A, coreNum, cluster_id, o, 0);
    const double tt = page_est_plan_total_ms(csr, st, A, o);

    page_free_plan_fields(A);
    page_restore_vstats(st, vlen0, vcmin0, vcmax0, nvrow);
    st->mode = base_mode;
    A->nnz_dsp = row_nnz;
    page_plan(csr, st, A, coreNum, cluster_id, o, 1);
    const int window_valid = A->has_window;
    const double tw = window_valid ? page_est_plan_total_ms(csr, st, A, o) : 1.0e300;

    if (!window_valid || !(tw < tt)) {
        page_free_plan_fields(A);
        page_restore_vstats(st, vlen0, vcmin0, vcmax0, nvrow);
        st->mode = base_mode;
        A->nnz_dsp = row_nnz;
        page_plan(csr, st, A, coreNum, cluster_id, o, 0);
    }

    A->model_tiled_ms = tt;
    A->model_window_ms = window_valid ? tw : -1.0;
    A->model_total_ms = page_est_plan_total_ms(csr, st, A, o);
    A->model_bw_mix = o->bw_mix;

    free(vlen0); free(vcmin0); free(vcmax0);
}

static double page_predict_rowoff_ms(const CSRMatrix *csr, const page_stat *st,
                                      const page_matrix *base, const page_opts *o,
                                      int row_off)
{
    const int m = csr->numRows;
    const int n = csr->numCols;
    const long long total_nnz = csr->numNonzeros;
    const long long prefix_nnz = st->row_pre[row_off];
    const long long suffix_nnz = total_nnz - prefix_nnz;
    const int suffix_rows = m - row_off;
    if (suffix_rows <= 0 || suffix_nnz <= 0 || base->nnz_dsp <= 0) return 1.0e300;

    const long long base_rows_nnz = st->row_pre[base->row_off + base->m_dsp]
                                  - st->row_pre[base->row_off];
    double keep = (base_rows_nnz > 0)
                ? (double)base->nnz_dsp / (double)base_rows_nnz : 1.0;
    if (keep < 0.0) keep = 0.0;
    if (keep > 1.0) keep = 1.0;

    const double dsp_nnz = (double)suffix_nnz * keep;
    const double residual_nnz = (double)suffix_nnz - dsp_nnz;
    const double stored_per = (base->nnz_dsp > 0)
                            ? (double)base->stored / (double)base->nnz_dsp : 1.0;
    const double stored = stored_per * dsp_nnz;

    const double ov_per_row = (base->m_dsp > 0)
                            ? (double)base->n_ov / (double)base->m_dsp : 0.0;
    double nvrow = (double)suffix_rows * (1.0 + ov_per_row);
    nvrow = ceil(nvrow / (double)SROW) * (double)SROW;

    double npanel = 1.0;
    if (base->nvrow > 0 && base->npanel > 0) {
        npanel = ceil((double)base->npanel * nvrow / (double)base->nvrow);
        if (npanel < 1.0) npanel = 1.0;
    }
    double x_elems = (double)base->x_elems;
    if (!base->has_window && base->npanel > 0)
        x_elems *= npanel / (double)base->npanel;

    const double extra_ddr = x_elems * 8.0 + nvrow * 8.0;

    const double res_rows = page_est_active_rows((long long)(residual_nnz + 0.5), suffix_rows);
    const double arm_nnz = (double)prefix_nnz + residual_nnz;
    const double arm_rows = (double)row_off + res_rows;
    const double arm_x = page_est_unique_cols((long long)(arm_nnz + 0.5), n);
    const double arm_bytes = arm_nnz * 12.0
                           + arm_rows * 8.0 + arm_x * 8.0;
    const int concurrent = (arm_bytes > 0.0);
    const double pipe_bw = concurrent && o->bw_pipe_mix > 0.0
                         ? o->bw_pipe_mix : o->bw_pipe;
    const double arm_bw = concurrent && o->bw_arm_mix > 0.0
                        ? o->bw_arm_mix : o->bw_arm;
    const double ntile = (base->npanel > 0 && base->ntile > 0)
                       ? ((double)base->ntile / (double)base->npanel) * npanel
                       : npanel;
    double td = ((stored * 16.0) / pipe_bw + extra_ddr / o->bw_ddr) * 1e3
              + o->launch_ms + 2.0 * ntile * o->barrier_ms;
    const double ta = arm_bytes > 0.0 ? arm_bytes / arm_bw * 1e3 : 0.0;
    const double row_frac = (base->m_dsp > 0)
                          ? (double)suffix_rows / (double)base->m_dsp : 1.0;
    const double tf_perm = base->perm_est_finalize_ms * row_frac;
    const double ov_pred = ov_per_row * (double)suffix_rows;
    const double tf = tf_perm
                    + 24.0 * (ov_pred + res_rows) / o->bw_arm * 1e3;

    double t = td;
    if (ta > t) t = ta;
    return t + tf;
}

static int page_choose_rowoff_model(const CSRMatrix *csr, const page_stat *st,
                                     const page_matrix *base, const page_opts *o,
                                     double *best_ms)
{
    const int m = csr->numRows;
    int best_r = o->force_hybrid ? -1 : 0;
    double best = o->force_hybrid ? 1.0e300
                                  : page_predict_rowoff_ms(csr, st, base, o, 0);

    if (m > SIGMA) {
        const int last = ((m - SIGMA) / SIGMA) * SIGMA;
        for (int r = SIGMA; r <= last; r += SIGMA) {
            if (o->force_hybrid) {
                const long long pn = st->row_pre[r];
                if (pn <= 0 || pn >= csr->numNonzeros) continue;
            }
            const double t = page_predict_rowoff_ms(csr, st, base, o, r);
            if (t < best) { best = t; best_r = r; }
        }
    }
    if (best_r < 0) {
        best_r = 0;
        best = page_predict_rowoff_ms(csr, st, base, o, 0);
    }
    if (best_ms) *best_ms = best;
    return best_r;
}

static int page_share_to_rowoff(const page_stat *st, int m, long long nnz, double s)
{
    if (!(s > 0.0)) return 0;
    if (s > 1.0) s = 1.0;
    const long long target = (long long)(s * (double)nnz);
    int lo = 0, hi = m;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (st->row_pre[mid] < target) lo = mid + 1; else hi = mid;
    }
    int row_off = (lo / SIGMA) * SIGMA;
    if (row_off > m - SIGMA) row_off = ((m - SIGMA) / SIGMA) * SIGMA;
    if (row_off < SIGMA) row_off = 0;
    return row_off;
}

static void page_discard_plan(page_matrix *A, page_stat *st)
{
    const int m = A->m, n = A->n, nnz = A->nnz, cn = A->coreNum;
    const int dsp_value_mode = A->dsp_value_mode;
    const MAT_VAL_TYPE dsp_constant_value = A->dsp_constant_value;
    page_free_matrix(A);
    free(st->v_len);  st->v_len  = NULL;
    free(st->v_cmin); st->v_cmin = NULL;
    free(st->v_cmax); st->v_cmax = NULL;
    A->m = m; A->n = n; A->nnz = nnz; A->coreNum = cn;
    A->dof_r = 1; A->idx_bytes = PAGE_IDX32_BYTES;
    A->dsp_value_mode = dsp_value_mode;
    A->dsp_constant_value = dsp_constant_value;
}

static void page_rebuild_at_rowoff(const CSRMatrix *csr, page_stat *st,
                                     page_matrix *A, int row_off,
                                     int coreNum, int cluster_id,
                                     const page_opts *o)
{
    page_discard_plan(A, st);
    st->mode = (csr->numCols <= GSM_X_CAP) ? PAGE_MODE_A : PAGE_MODE_B;
    page_build_vrows(csr, st, A, row_off, csr->numRows - row_off);
    page_plan_adaptive(csr, st, A, coreNum, cluster_id, o);
    A->model_total_ms = page_est_plan_total_ms(csr, st, A, o);
}

static void page_build_impl(const CSRMatrix *csr, page_stat *st, page_matrix *A,
                             int coreNum, int cluster_id, double tpre[3],
                             const page_opts *opt, int materialize_values)
{
    page_opts o;
    if (opt) o = *opt; else page_opts_default(&o);
    if (!(o.bw_mix > 0.0)) o.bw_mix = o.bw_ddr;

    memset(A, 0, sizeof(*A));
    const double t0 = omp_get_wtime();
    page_pass_a(csr, st);
    const double t1 = omp_get_wtime();

    const int m = csr->numRows, n = csr->numCols;
    A->m = m; A->n = n; A->nnz = csr->numNonzeros;
    A->value_mode = csr->value_mode;
    A->constant_value = csr->constant_value;
    A->dsp_value_mode = o.dsp_value_mode;
    A->dsp_constant_value = o.dsp_constant_value;
    A->dof_r = 1; A->idx_bytes = PAGE_IDX32_BYTES; A->coreNum = coreNum;
    const double t_arm_full = page_est_arm_ms(A->nnz, m, n, &o);

    if (o.force_cpu) {
        A->mode = PAGE_MODE_CPU; st->mode = PAGE_MODE_CPU;
        A->row_off = 0; A->m_dsp = 0; A->cpu_share = 1.0;
        A->est_dsp_ms = t_arm_full; A->est_arm_ms = t_arm_full;
        A->model_total_ms = t_arm_full; A->model_share_ms = t_arm_full;
        A->model_tiled_ms = A->model_window_ms = -1.0;
        A->model_bw_mix = o.bw_mix;
        tpre[0] = t1 - t0; tpre[1] = 0.0; tpre[2] = 0.0;
        return;
    }

    int row_off = 0;
    page_build_vrows(csr, st, A, 0, m);
    page_plan_adaptive(csr, st, A, coreNum, cluster_id, &o);

    double share_model_ms = A->model_total_ms;
    int chosen_row_off = 0;
    int predict_row = 0, correct_row = 0, replan_rounds = 0;
    double predict_ms = A->model_total_ms, correct_ms = A->model_total_ms;

    if (o.row_off_override >= 0) {
        chosen_row_off = o.row_off_override;
        if (chosen_row_off < 0) chosen_row_off = 0;
        if (chosen_row_off > m - SIGMA && m > SIGMA)
            chosen_row_off = ((m - SIGMA) / SIGMA) * SIGMA;
        if (chosen_row_off < SIGMA) chosen_row_off = 0;
        share_model_ms = page_predict_rowoff_ms(csr, st, A, &o, chosen_row_off);
        predict_row = correct_row = chosen_row_off;
        predict_ms = correct_ms = share_model_ms;
        if (chosen_row_off != 0) {
            page_rebuild_at_rowoff(csr, st, A, chosen_row_off, coreNum, cluster_id, &o);
            replan_rounds = 1;
        }
    } else if (o.cpu_share >= 0.0) {
        chosen_row_off = page_share_to_rowoff(st, m, A->nnz, o.cpu_share);
        share_model_ms = page_predict_rowoff_ms(csr, st, A, &o, chosen_row_off);
        predict_row = correct_row = chosen_row_off;
        predict_ms = correct_ms = share_model_ms;
        if (chosen_row_off != 0) {
            page_rebuild_at_rowoff(csr, st, A, chosen_row_off, coreNum, cluster_id, &o);
            replan_rounds = 1;
        }
    } else {

        const double full_dsp_exact = A->model_total_ms;
        int best_row = o.force_hybrid ? -1 : 0;
        int best_policy = -1;
        double best_exact = o.force_hybrid ? 1.0e300 : full_dsp_exact;
        int current_row = 0;

        if (o.force_hybrid) {
            const long long row0_nnz = st->row_pre[m] - st->row_pre[0];
            long long row0_res = A->has_window ? row0_nnz - A->nnz_dsp : 0;
            if (row0_res < 0) row0_res = 0;
            if (A->has_window && row0_res > 0 && A->nnz_dsp > 0) {
                best_row = 0;
                best_policy = -1;
                best_exact = A->model_total_ms;
            } else {
                page_opts wopt = o;
                wopt.plan_policy = 1;
                page_rebuild_at_rowoff(csr, st, A, 0, coreNum, cluster_id, &wopt);
                row0_res = A->has_window ? row0_nnz - A->nnz_dsp : 0;
                if (row0_res < 0) row0_res = 0;
                if (A->has_window && row0_res > 0 && A->nnz_dsp > 0) {
                    best_row = 0;
                    best_policy = 1;
                    best_exact = A->model_total_ms;
                }

                page_rebuild_at_rowoff(csr, st, A, 0, coreNum, cluster_id, &o);
            }
        }

        predict_row = page_choose_rowoff_model(csr, st, A, &o, &predict_ms);
        share_model_ms = predict_ms;
        chosen_row_off = predict_row;
        correct_row = predict_row;
        correct_ms = predict_ms;

        if (predict_row != 0) {
            page_rebuild_at_rowoff(csr, st, A, predict_row, coreNum, cluster_id, &o);
            replan_rounds = 1;
            current_row = predict_row;
            if (A->model_total_ms < best_exact) {
                best_exact = A->model_total_ms;
                best_row = predict_row;
                best_policy = -1;
            }

            correct_row = page_choose_rowoff_model(csr, st, A, &o, &correct_ms);
            if (correct_row != predict_row) {
                if (correct_row == 0) {

                } else {
                    page_rebuild_at_rowoff(csr, st, A, correct_row, coreNum, cluster_id, &o);
                    replan_rounds = 2;
                    current_row = correct_row;
                    if (A->model_total_ms < best_exact) {
                        best_exact = A->model_total_ms;
                        best_row = correct_row;
                        best_policy = -1;
                    }
                }
            }
        }

        if (!o.force_hybrid && t_arm_full <= best_exact) {
            page_discard_plan(A, st);
            A->mode = PAGE_MODE_CPU; st->mode = PAGE_MODE_CPU;
            A->m = m; A->n = n; A->nnz = csr->numNonzeros; A->coreNum = coreNum;
            A->dof_r = 1; A->idx_bytes = PAGE_IDX32_BYTES;
            A->row_off = 0; A->m_dsp = 0; A->cpu_share = 1.0;
            A->est_dsp_ms = best_exact; A->est_arm_ms = t_arm_full;
            A->model_total_ms = t_arm_full; A->model_share_ms = share_model_ms;
            A->model_bw_mix = o.bw_mix;
            A->replan_predict_row = predict_row;
            A->replan_correct_row = correct_row;
            A->replan_rounds = replan_rounds;
            A->replan_predict_ms = predict_ms;
            A->replan_correct_ms = correct_ms;
            tpre[0] = t1 - t0; tpre[1] = omp_get_wtime() - t1; tpre[2] = 0.0;
            return;
        }

        if (best_row < 0) {

            best_row = 0;
            best_exact = full_dsp_exact;
            printf("[FORCE_HYBRID] legal_hybrid_cut=0 fallback=full_dsp\n");
        }
        if (current_row != best_row || (best_row == 0 && best_policy == 1)) {
            if (best_row == 0 && best_policy == 1) {
                page_opts wopt = o;
                wopt.plan_policy = 1;
                page_rebuild_at_rowoff(csr, st, A, 0, coreNum, cluster_id, &wopt);
            } else {
                page_rebuild_at_rowoff(csr, st, A, best_row, coreNum, cluster_id, &o);
            }
            if (best_row != 0) replan_rounds++;
            current_row = best_row;
        }
        chosen_row_off = best_row;
    }

    row_off = chosen_row_off;
    A->cpu_share = (row_off > 0 && A->nnz > 0)
                 ? (double)st->row_pre[row_off] / (double)A->nnz : 0.0;
    A->model_share_ms = share_model_ms;
    A->model_bw_mix = o.bw_mix;
    A->replan_predict_row = predict_row;
    A->replan_correct_row = correct_row;
    A->replan_rounds = replan_rounds;
    A->replan_predict_ms = predict_ms;
    A->replan_correct_ms = correct_ms;

    A->model_total_ms = (A->mode == PAGE_MODE_CPU)
                      ? t_arm_full : page_est_plan_total_ms(csr, st, A, &o);
    if (!o.force_hybrid && o.row_off_override < 0 && o.cpu_share < 0.0 && t_arm_full <= A->model_total_ms) {
        const double model = A->model_total_ms;
        const int pr = A->replan_predict_row, cr = A->replan_correct_row;
        const int rr = A->replan_rounds;
        const double pm = A->replan_predict_ms, cm = A->replan_correct_ms;
        page_discard_plan(A, st);
        A->mode = PAGE_MODE_CPU; st->mode = PAGE_MODE_CPU;
        A->m = m; A->n = n; A->nnz = csr->numNonzeros; A->coreNum = coreNum;
        A->dof_r = 1; A->idx_bytes = PAGE_IDX32_BYTES;
        A->row_off = 0; A->m_dsp = 0; A->cpu_share = 1.0;
        A->est_dsp_ms = model; A->est_arm_ms = t_arm_full;
        A->model_total_ms = t_arm_full; A->model_share_ms = share_model_ms;
        A->model_bw_mix = o.bw_mix;
        A->replan_predict_row = pr; A->replan_correct_row = cr;
        A->replan_rounds = rr; A->replan_predict_ms = pm; A->replan_correct_ms = cm;
        tpre[0] = t1 - t0; tpre[1] = omp_get_wtime() - t1; tpre[2] = 0.0;
        return;
    }

    if (materialize_values && A->has_window) page_build_residual(csr, A);
    const double t2 = omp_get_wtime();

    A->bytes_per_nnz = (A->nnz_dsp > 0) ? page_bytes(A) / (double)A->nnz_dsp : 0.0;
    A->est_dsp_ms = page_est_dsp_ms(A, &o);
    A->est_arm_ms = t_arm_full;

    if (materialize_values) page_pass_b(csr, st, A, cluster_id);
    const double t3 = omp_get_wtime();

    tpre[0] = t1 - t0; tpre[1] = t2 - t1; tpre[2] = t3 - t2;
}

static void page_build(const CSRMatrix *csr, page_stat *st, page_matrix *A,
                        int coreNum, int cluster_id, double tpre[3],
                        const page_opts *opt)
{
    page_build_impl(csr, st, A, coreNum, cluster_id, tpre, opt, 1);
}

static void page_build_shadow(const CSRMatrix *csr, page_stat *st, page_matrix *A,
                               int coreNum, int cluster_id, double tpre[3],
                               const page_opts *opt)
{
    page_opts so;
    if (opt) so = *opt; else page_opts_default(&so);
    const char *e = getenv("PAGE_FAST_SHADOW_PLAN");
    so.fast_shadow_precheck = (e && strcmp(e, "1") == 0) ? 1 : 0;
    page_build_impl(csr, st, A, coreNum, cluster_id, tpre, &so, 0);
}

#endif
