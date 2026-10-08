#ifndef _PAGE_FORMAT_H_
#define _PAGE_FORMAT_H_

#include "common.h"

typedef struct {

    int m, n, nnz;
    int coreNum;
    int mode;
    int dof_r;
    int idx_bytes;

    int value_mode;
    MAT_VAL_TYPE constant_value;

    int dsp_value_mode;
    MAT_VAL_TYPE dsp_constant_value;

    int  npanel;
    int *panel_slice_lo;
    int *panel_tile_lo;

    int  ntile;
    int *tile_xlo;
    int *tile_xhi;
    int *tile_inst_lo;
    unsigned char *tile_packed;
    int *tile_xmap_lo;
    int *tile_xmap;
    int packed_tiles;
    long long packed_x_elems;
    int packed_runs;
    int packed_dma_rounds;

    int  ninst;
    int *cl;
    int *cs;

    MAT_VAL_TYPE *val;
    void         *cidx;

    unsigned short *rperm;
    unsigned short *rinv;
    int             has_perm;
    unsigned char  *perm_blk;
    int             nperm_blk;
    long long       perm_saved_elems;
    double          perm_est_save_ms;
    double          perm_est_finalize_ms;
    int             perm_guard_drop;

    int   nvrow;
    int   row_off;
    int   m_dsp;
    int   n_ov;
    int  *ov_row;
    int   row_cap;
    int   has_split;
    int  *vr_kb, *vr_ke;
    long long nnz_dsp;
    long long x_elems;

    int   has_window;
    int   cw_lo, cw_hi;
    int  *res_rp;
    int  *res_ci;
    MAT_VAL_TYPE *res_val;
    long long res_nnz;
    double cpu_share;
    double est_dsp_ms, est_arm_ms;

    double model_total_ms;
    double model_tiled_ms;
    double model_window_ms;
    double model_share_ms;
    double model_bw_mix;

    int    replan_predict_row;
    int    replan_correct_row;
    int    replan_rounds;
    double replan_predict_ms;
    double replan_correct_ms;

    int *thread_ptr;
    int *gsmxwidth;

    long long stored;
    long long pad_zeros;
    double    z_ratio;
    double    lb_imbalance;
    double    bytes_per_nnz;
} page_matrix;

static inline double page_bytes_ddr(const page_matrix *A)
{
    const double nz = (double)A->stored;
    const double r  = (double)(A->dof_r > 0 ? A->dof_r : 1);
    double xin = 0.0, xdesc = 0.0;
    for (int t = 0; t < A->ntile; t++) {
        const int packed = A->tile_packed ? (int)A->tile_packed[t] : 0;
        if (!packed && t > 0 && A->tile_packed && !A->tile_packed[t - 1] &&
            A->tile_xlo[t] == A->tile_xlo[t - 1] &&
            A->tile_xhi[t] == A->tile_xhi[t - 1])
            continue;
        if (packed && A->tile_xmap_lo) {
            const double nb = (double)(A->tile_xmap_lo[t + 1] - A->tile_xmap_lo[t]);
            xin += nb * (double)PAGE_XBLOCK_ELEMS;
            xdesc += nb * 4.0;
        } else {
            xin += (double)(A->tile_xhi[t] - A->tile_xlo[t]);
        }
    }
    const double value_bytes = (A->dsp_value_mode == PAGE_VALUE_GENERAL) ? 8.0 : 0.0;
    return nz * value_bytes + nz / r * (double)A->idx_bytes + xin * 8.0 + xdesc
         + (double)A->nvrow * 8.0;
}

static inline double page_bytes_gsm(const page_matrix *A)
{
    const double r = (double)(A->dof_r > 0 ? A->dof_r : 1);
    return (double)A->stored / r * 8.0;
}

static inline double page_bytes(const page_matrix *A)
{
    return page_bytes_ddr(A) + page_bytes_gsm(A);
}

static inline void page_free_matrix(page_matrix *A)
{
    PAGE_DEV_FREE(A->val);
    PAGE_DEV_FREE(A->cidx);
    PAGE_DEV_FREE(A->cl);
    PAGE_DEV_FREE(A->cs);
    PAGE_DEV_FREE(A->thread_ptr);
    PAGE_DEV_FREE(A->gsmxwidth);
    PAGE_DEV_FREE(A->tile_xlo);
    PAGE_DEV_FREE(A->tile_xhi);
    PAGE_DEV_FREE(A->tile_inst_lo);
    PAGE_DEV_FREE(A->tile_packed);
    PAGE_DEV_FREE(A->tile_xmap_lo);
    PAGE_DEV_FREE(A->tile_xmap);
    PAGE_DEV_FREE(A->panel_slice_lo);
    PAGE_DEV_FREE(A->panel_tile_lo);
    free(A->rperm);
    free(A->rinv);
    free(A->perm_blk);
    free(A->ov_row);
    free(A->vr_kb);
    free(A->vr_ke);
    free(A->res_rp);
    free(A->res_ci);
    free(A->res_val);
    memset(A, 0, sizeof(*A));
}

#endif
