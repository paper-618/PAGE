#include <compiler/m3000.h>
#include "hthread_device.h"
#include "../../header/common.h"

__gsm__ MAT_VAL_TYPE gx[GSM_X_CAP + 1];

static inline void page_issue(int buf, int off, int len,
                               MAT_VAL_TYPE *val, void *cidx, int idx_bytes,
                               int value_mode,
                               lvector MAT_VAL_TYPE **val_am,
                               lvector MAT_VAL_TYPE **xam,
                               unsigned int *ch_val, unsigned int *ch_x,
                               page_idx16_t *idx16,
                               page_idx32_t idx32[NBUF][CHUNK16])
{
    const int bytes = len * (int)sizeof(MAT_VAL_TYPE);
    if (value_mode == PAGE_VALUE_GENERAL) {
        ch_val[buf] = dma_p2p(&val[off], 1, bytes, 0,
                              val_am[buf], 1, bytes, 0, false, 0);
    } else {
        ch_val[buf] = 0;
    }

    if (idx_bytes == PAGE_IDX16_BYTES) {
        unsigned int chi = dma_p2p(&((page_idx16_t *)cidx)[off], 1,
                                   len * (int)sizeof(page_idx16_t), 0,
                                   idx16, 1,
                                   len * (int)sizeof(page_idx16_t), 0,
                                   false, 0);
        dma_wait(chi);
        /* Each ping-pong buffer owns descriptors until its gather completes. */
        for (int i = 0; i < len; i++) idx32[buf][i] = ((unsigned int)idx16[i]) << 3;
        ch_x[buf] = dma_sg(gx, (int *)idx32[buf], 1, bytes, 0,
                           xam[buf], 1, bytes, 0);
    } else {
        ch_x[buf] = dma_sg(gx, (int *)&((page_idx32_t *)cidx)[off],
                           1, bytes, 0, xam[buf], 1, bytes, 0);
    }
}

__global__ void page_spmv(int nvrow,
                           int n,
                           int coreNum,
                           int npanel,
                           int barrier_id,
                           int ntile,
                           int idx_bytes,
                           int value_mode,
                           int *panel_slice_lo,
                           int *panel_tile_lo,
                           int *tile_xlo,
                           int *tile_xhi,
                           int *tile_inst_lo,
                           unsigned char *tile_packed,
                           int *tile_xmap_lo,
                           int *tile_xmap,
                           int *cl,
                           int *cs,
                           MAT_VAL_TYPE *val,
                           void *cidx,
                           int *thread_ptr,
                           int *gsmxwidth,
                           MAT_VAL_TYPE *x,
                           MAT_VAL_TYPE *y,
                           MAT_VAL_TYPE *constant_ptr)
{
    const int tid = get_thread_id();
    const int chunk = (idx_bytes == PAGE_IDX16_BYTES) ? CHUNK16 : CHUNK32;
    (void)n; (void)ntile;

    lvector MAT_VAL_TYPE *val_am[NBUF], *xam[NBUF];
    for (int b = 0; b < NBUF; b++) {

        val_am[b] = vector_malloc(chunk * sizeof(MAT_VAL_TYPE));
        xam[b]    = vector_malloc(chunk * sizeof(MAT_VAL_TYPE));
    }
    lvector MAT_VAL_TYPE *yam =
        vector_malloc(PANEL_SLICE * SROW * sizeof(MAT_VAL_TYPE));

    /* The 16-bit staging buffer and two descriptor banks use 40 KB of SM. */
    page_idx16_t idx16[CHUNK16];
    page_idx32_t idx32[NBUF][CHUNK16];

    unsigned int ch_val[NBUF], ch_x[NBUF];
    const lvector MAT_VAL_TYPE vzero = vec_svbcast(0.0);
    MAT_VAL_TYPE scalar_const = 1.0;
    if (value_mode == PAGE_VALUE_CONSTANT && constant_ptr) scalar_const = constant_ptr[0];
    const lvector MAT_VAL_TYPE vconst = vec_svbcast(scalar_const);

    for (int p = 0; p < npanel; p++) {
        const int s0 = thread_ptr[p * (coreNum + 1) + tid];
        const int s1 = thread_ptr[p * (coreNum + 1) + tid + 1];
        const int myslice = s1 - s0;

        for (int i = 0; i < myslice; i++) yam[i] = vzero;

        for (int t = panel_tile_lo[p]; t < panel_tile_lo[p + 1]; t++) {
            /* Reuse contiguous windows only; packed maps need separate identity checks. */
            const int packed = tile_packed ? (int)tile_packed[t] : 0;
            const int reuse_x = (!packed && t > 0 && tile_packed && !tile_packed[t - 1] &&
                                 tile_xlo[t] == tile_xlo[t - 1] &&
                                 tile_xhi[t] == tile_xhi[t - 1]);
            const int xseg  = gsmxwidth[t * (coreNum + 1) + tid];
            const int xsize = gsmxwidth[t * (coreNum + 1) + tid + 1] - xseg;
            if (!reuse_x && xsize > 0) {
                unsigned int cx;
                if (packed) {

                    const int map0 = tile_xmap_lo[t];
                    const int bend = xseg + xsize;
                    int j = xseg;
                    while (j < bend) {
                        const int src_boff = tile_xmap[map0 + j];
                        int nr = 1;
                        while (j + nr < bend &&
                               tile_xmap[map0 + j + nr] == src_boff + nr * PAGE_XBLOCK_BYTES)
                            nr++;
                        const int ne = nr * PAGE_XBLOCK_ELEMS;
                        cx = dma_p2p(&x[src_boff / (int)sizeof(MAT_VAL_TYPE)], 1,
                                     ne * (int)sizeof(MAT_VAL_TYPE), 0,
                                     &gx[j * PAGE_XBLOCK_ELEMS], 1,
                                     ne * (int)sizeof(MAT_VAL_TYPE), 0,
                                     false, 0);
                        dma_wait(cx);
                        j += nr;
                    }
                } else {
                    cx = dma_p2p(&x[tile_xlo[t] + xseg], 1,
                                 xsize * sizeof(MAT_VAL_TYPE), 0,
                                 &gx[xseg], 1,
                                 xsize * sizeof(MAT_VAL_TYPE), 0,
                                 false, 0);
                    dma_wait(cx);
                }
            }
            /* Implicit-value padding must gather a real zero. */
            if (value_mode != PAGE_VALUE_GENERAL && tid == 0) {
                const int zero_local = packed
                    ? (tile_xmap_lo[t + 1] - tile_xmap_lo[t]) * PAGE_XBLOCK_ELEMS
                    : (tile_xhi[t] - tile_xlo[t]);
                gx[zero_local] = (MAT_VAL_TYPE)0.0;
            }
            core_barrier(barrier_id, coreNum);

            const int ib = tile_inst_lo[t] + (s0 - panel_slice_lo[p]);
            const int ie = tile_inst_lo[t] + (s1 - panel_slice_lo[p]);
            const int total = cs[ie] - cs[ib];

            if (total > 0) {
                const int nchunk = (total + chunk - 1) / chunk;
                int block_id = ib;
                int nexti    = cl[ib];

                const int len0 = (total > chunk) ? chunk : total;
                page_issue(0, cs[ib], len0, val, cidx, idx_bytes, value_mode,
                            val_am, xam, ch_val, ch_x, idx16, idx32);

                for (int k = 0; k < nchunk; k++) {
                    const int off = cs[ib] + k * chunk;
                    const int len = (off + chunk > cs[ie]) ? (cs[ie] - off) : chunk;

                    if (k + 1 < nchunk) {
                        const int off2 = off + len;
                        const int len2 = (off2 + chunk > cs[ie]) ? (cs[ie] - off2) : chunk;
                        page_issue((k + 1) & 1, off2, len2,
                                    val, cidx, idx_bytes, value_mode,
                                    val_am, xam, ch_val, ch_x, idx16, idx32);
                    }

                    if (value_mode == PAGE_VALUE_GENERAL) dma_wait(ch_val[k & 1]);
                    dma_wait(ch_x[k & 1]);

                    lvector MAT_VAL_TYPE *va = val_am[k & 1];
                    lvector MAT_VAL_TYPE *xa = xam[k & 1];
                    const int nvec = len / SROW;
                    int prei = 0;

                    while (block_id < ie) {
                        const int ub = (nexti <= nvec) ? nexti : nvec;
                        if (ub > prei) {
                            lvector MAT_VAL_TYPE a0 = vzero, a1 = vzero,
                                                 a2 = vzero, a3 = vzero;
                            int j = prei;
                            if (value_mode == PAGE_VALUE_UNIT) {
                                for (; j + 3 < ub; j += 4) {
                                    a0 = a0 + xa[j];
                                    a1 = a1 + xa[j + 1];
                                    a2 = a2 + xa[j + 2];
                                    a3 = a3 + xa[j + 3];
                                }
                                for (; j < ub; j++) a0 = a0 + xa[j];
                            } else if (value_mode == PAGE_VALUE_CONSTANT) {
                                for (; j + 3 < ub; j += 4) {
                                    a0 = vec_mula(vconst, xa[j],     a0);
                                    a1 = vec_mula(vconst, xa[j + 1], a1);
                                    a2 = vec_mula(vconst, xa[j + 2], a2);
                                    a3 = vec_mula(vconst, xa[j + 3], a3);
                                }
                                for (; j < ub; j++) a0 = vec_mula(vconst, xa[j], a0);
                            } else {
                                for (; j + 3 < ub; j += 4) {
                                    a0 = vec_mula(va[j],     xa[j],     a0);
                                    a1 = vec_mula(va[j + 1], xa[j + 1], a1);
                                    a2 = vec_mula(va[j + 2], xa[j + 2], a2);
                                    a3 = vec_mula(va[j + 3], xa[j + 3], a3);
                                }
                                for (; j < ub; j++) a0 = vec_mula(va[j], xa[j], a0);
                            }
                            yam[block_id - ib] += (a0 + a1) + (a2 + a3);
                        }
                        if (nexti >= nvec) break;
                        prei  = nexti;
                        nexti += cl[++block_id];
                    }
                    nexti -= nvec;
                }
            }
            core_barrier(barrier_id, coreNum);
        }

        int r0 = s0 * SROW;
        int rw = myslice * SROW;
        if (r0 + rw > nvrow) rw = nvrow - r0;
        if (rw > 0) {
            unsigned int cy = dma_p2p(yam, 1, rw * sizeof(MAT_VAL_TYPE), 0,
                                      &y[r0], 1, rw * sizeof(MAT_VAL_TYPE), 0,
                                      false, 0);
            dma_wait(cy);
        }
    }

    for (int b = 0; b < NBUF; b++) { vector_free(val_am[b]); vector_free(xam[b]); }
    vector_free(yam);
}

__global__ void page_null_kernel(int dummy)
{
    (void)dummy;
}

__global__ void page_bench_dma(int mode, int nrep, int nbytes,
                                MAT_VAL_TYPE *ddr_src, int *sg_idx)
{
    const int nelem = nbytes / (int)sizeof(MAT_VAL_TYPE);
    lvector MAT_VAL_TYPE *a = vector_malloc(nbytes);
    lvector MAT_VAL_TYPE *b = vector_malloc(nbytes);

    for (int r = 0; r < nrep; r++) {
        if (mode == 0) {
            unsigned int c1 = dma_p2p(ddr_src, 1, nbytes, 0, a, 1, nbytes, 0, false, 0);
            dma_wait(c1);
            unsigned int c2 = dma_sg(gx, sg_idx, 1, nbytes, 0, b, 1, nbytes, 0);
            dma_wait(c2);
        } else {
            unsigned int c1 = dma_p2p(ddr_src, 1, nbytes, 0, a, 1, nbytes, 0, false, 0);
            unsigned int c2 = dma_sg(gx, sg_idx, 1, nbytes, 0, b, 1, nbytes, 0);
            dma_wait(c1);
            dma_wait(c2);
        }
    }
    (void)nelem;
    vector_free(a); vector_free(b);
}

__global__ void page_bench_stream(int nrep, int nbytes, MAT_VAL_TYPE *ddr_src)
{
    lvector MAT_VAL_TYPE *a = vector_malloc(nbytes);
    for (int r = 0; r < nrep; r++) {
        unsigned int c = dma_p2p(ddr_src, 1, nbytes, 0, a, 1, nbytes, 0, false, 0);
        dma_wait(c);
    }
    vector_free(a);
}

__global__ void page_bench_xload(int nrep, int nbytes, MAT_VAL_TYPE *ddr_src)
{
    const int tid = get_thread_id();
    const int nelem = nbytes / (int)sizeof(MAT_VAL_TYPE);
    const int off = tid * nelem;
    for (int r = 0; r < nrep; r++) {
        unsigned int c = dma_p2p(&ddr_src[off], 1, nbytes, 0,
                                 &gx[off], 1, nbytes, 0, false, 0);
        dma_wait(c);
    }
}

__global__ void page_bench_pipeline(int nrep, int nbytes,
                                     MAT_VAL_TYPE *ddr_src, int *sg_idx,
                                     MAT_VAL_TYPE *out)
{
    const int tid = get_thread_id();
    const int nelem = nbytes / (int)sizeof(MAT_VAL_TYPE);
    const int nvec = nelem / SROW;
    lvector MAT_VAL_TYPE *va[NBUF], *xa[NBUF];
    for (int b = 0; b < NBUF; b++) {
        va[b] = vector_malloc(nbytes);
        xa[b] = vector_malloc(nbytes);
    }
    unsigned int cv[NBUF], cx[NBUF];
    const lvector MAT_VAL_TYPE z = vec_svbcast(0.0);
    lvector MAT_VAL_TYPE sink = z;

    if (nrep > 0) {
        cv[0] = dma_p2p(ddr_src, 1, nbytes, 0, va[0], 1, nbytes, 0, false, 0);
        cx[0] = dma_sg(gx, sg_idx, 1, nbytes, 0, xa[0], 1, nbytes, 0);
    }
    for (int r = 0; r < nrep; r++) {
        const int b = r & 1;
        const int nb = (r + 1) & 1;
        if (r + 1 < nrep) {
            cv[nb] = dma_p2p(ddr_src, 1, nbytes, 0, va[nb], 1, nbytes, 0, false, 0);
            cx[nb] = dma_sg(gx, sg_idx, 1, nbytes, 0, xa[nb], 1, nbytes, 0);
        }
        dma_wait(cv[b]);
        dma_wait(cx[b]);
        lvector MAT_VAL_TYPE a0 = z, a1 = z, a2 = z, a3 = z;
        int j = 0;
        for (; j + 3 < nvec; j += 4) {
            a0 = vec_mula(va[b][j],     xa[b][j],     a0);
            a1 = vec_mula(va[b][j + 1], xa[b][j + 1], a1);
            a2 = vec_mula(va[b][j + 2], xa[b][j + 2], a2);
            a3 = vec_mula(va[b][j + 3], xa[b][j + 3], a3);
        }
        for (; j < nvec; j++) a0 = vec_mula(va[b][j], xa[b][j], a0);
        sink += (a0 + a1) + (a2 + a3);
    }

    lvector MAT_VAL_TYPE *tmp = vector_malloc(SROW * sizeof(MAT_VAL_TYPE));
    tmp[0] = sink;
    unsigned int co = dma_p2p(tmp, 1, SROW * sizeof(MAT_VAL_TYPE), 0,
                              &out[tid * SROW], 1, SROW * sizeof(MAT_VAL_TYPE), 0,
                              false, 0);
    dma_wait(co);
    vector_free(tmp);
    for (int b = 0; b < NBUF; b++) { vector_free(va[b]); vector_free(xa[b]); }
}

__global__ void page_bench_barrier(int nrep, int barrier_id, int coreNum)
{
    for (int r = 0; r < nrep; r++) core_barrier(barrier_id, coreNum);
}

__global__ void page_diag_ddr_am(int nrep, int nbytes,
                                  MAT_VAL_TYPE *ddr_src, MAT_VAL_TYPE *out)
{
    const int tid = get_thread_id();
    const int nelem = nbytes / (int)sizeof(MAT_VAL_TYPE);
    MAT_VAL_TYPE *src = &ddr_src[tid * nelem];
    lvector MAT_VAL_TYPE *a = vector_malloc(nbytes);
    for (int r = 0; r < nrep; r++) {
        unsigned int c = dma_p2p(src, 1, nbytes, 0,
                                 a, 1, nbytes, 0, false, 0);
        dma_wait(c);
    }

    if (nbytes >= SROW * (int)sizeof(MAT_VAL_TYPE)) {
        unsigned int co = dma_p2p(a, 1, SROW * sizeof(MAT_VAL_TYPE), 0,
                                  &out[tid * SROW], 1,
                                  SROW * sizeof(MAT_VAL_TYPE), 0, false, 0);
        dma_wait(co);
    }
    vector_free(a);
}

__global__ void page_diag_gsm_am_gather(int nrep, int nbytes,
                                         int *sg_idx, MAT_VAL_TYPE *out)
{
    const int tid = get_thread_id();
    const int nelem = nbytes / (int)sizeof(MAT_VAL_TYPE);
    int *idx = &sg_idx[tid * nelem];
    lvector MAT_VAL_TYPE *a = vector_malloc(nbytes);
    for (int r = 0; r < nrep; r++) {
        unsigned int c = dma_sg(gx, idx, 1, nbytes, 0,
                                a, 1, nbytes, 0);
        dma_wait(c);
    }
    if (nbytes >= SROW * (int)sizeof(MAT_VAL_TYPE)) {
        unsigned int co = dma_p2p(a, 1, SROW * sizeof(MAT_VAL_TYPE), 0,
                                  &out[tid * SROW], 1,
                                  SROW * sizeof(MAT_VAL_TYPE), 0, false, 0);
        dma_wait(co);
    }
    vector_free(a);
}

__global__ void page_diag_pipeline(int nrep, int nbytes,
                                    MAT_VAL_TYPE *ddr_src, int *sg_idx,
                                    MAT_VAL_TYPE *out)
{
    const int tid = get_thread_id();
    const int nelem = nbytes / (int)sizeof(MAT_VAL_TYPE);
    const int nvec = nelem / SROW;
    MAT_VAL_TYPE *src = &ddr_src[tid * nelem];
    int *idx = &sg_idx[tid * nelem];
    lvector MAT_VAL_TYPE *va[NBUF], *xa[NBUF];
    for (int b = 0; b < NBUF; b++) {
        va[b] = vector_malloc(nbytes);
        xa[b] = vector_malloc(nbytes);
    }
    unsigned int cv[NBUF], cx[NBUF];
    const lvector MAT_VAL_TYPE z = vec_svbcast(0.0);
    lvector MAT_VAL_TYPE sink = z;

    if (nrep > 0) {
        cv[0] = dma_p2p(src, 1, nbytes, 0,
                        va[0], 1, nbytes, 0, false, 0);
        cx[0] = dma_sg(gx, idx, 1, nbytes, 0,
                       xa[0], 1, nbytes, 0);
    }
    for (int r = 0; r < nrep; r++) {
        const int b = r & 1;
        const int nb = (r + 1) & 1;
        if (r + 1 < nrep) {
            cv[nb] = dma_p2p(src, 1, nbytes, 0,
                             va[nb], 1, nbytes, 0, false, 0);
            cx[nb] = dma_sg(gx, idx, 1, nbytes, 0,
                            xa[nb], 1, nbytes, 0);
        }
        dma_wait(cv[b]);
        dma_wait(cx[b]);
        lvector MAT_VAL_TYPE a0 = z, a1 = z, a2 = z, a3 = z;
        int j = 0;
        for (; j + 3 < nvec; j += 4) {
            a0 = vec_mula(va[b][j],     xa[b][j],     a0);
            a1 = vec_mula(va[b][j + 1], xa[b][j + 1], a1);
            a2 = vec_mula(va[b][j + 2], xa[b][j + 2], a2);
            a3 = vec_mula(va[b][j + 3], xa[b][j + 3], a3);
        }
        for (; j < nvec; j++) a0 = vec_mula(va[b][j], xa[b][j], a0);
        sink += (a0 + a1) + (a2 + a3);
    }

    lvector MAT_VAL_TYPE *tmp = vector_malloc(SROW * sizeof(MAT_VAL_TYPE));
    tmp[0] = sink;
    unsigned int co = dma_p2p(tmp, 1, SROW * sizeof(MAT_VAL_TYPE), 0,
                              &out[tid * SROW], 1,
                              SROW * sizeof(MAT_VAL_TYPE), 0, false, 0);
    dma_wait(co);
    vector_free(tmp);
    for (int b = 0; b < NBUF; b++) {
        vector_free(va[b]);
        vector_free(xa[b]);
    }
}

__global__ void page_diag_compute_only(int nrep, int nbytes, MAT_VAL_TYPE *out)
{
    const int tid = get_thread_id();
    const int nvec = nbytes / (SROW * (int)sizeof(MAT_VAL_TYPE));
    lvector MAT_VAL_TYPE *a = vector_malloc(nbytes);
    lvector MAT_VAL_TYPE *b = vector_malloc(nbytes);
    const lvector MAT_VAL_TYPE va = vec_svbcast(1.0009765625 + (double)(tid & 7));
    const lvector MAT_VAL_TYPE vb = vec_svbcast(0.9990234375);
    const lvector MAT_VAL_TYPE z = vec_svbcast(0.0);
    for (int j = 0; j < nvec; j++) {
        a[j] = va;
        b[j] = vb;
    }
    lvector MAT_VAL_TYPE sink = z;
    for (int r = 0; r < nrep; r++) {
        lvector MAT_VAL_TYPE s0 = z, s1 = z, s2 = z, s3 = z;
        int j = 0;
        for (; j + 3 < nvec; j += 4) {
            s0 = vec_mula(a[j],     b[j],     s0);
            s1 = vec_mula(a[j + 1], b[j + 1], s1);
            s2 = vec_mula(a[j + 2], b[j + 2], s2);
            s3 = vec_mula(a[j + 3], b[j + 3], s3);
        }
        for (; j < nvec; j++) s0 = vec_mula(a[j], b[j], s0);
        sink += (s0 + s1) + (s2 + s3);
    }
    lvector MAT_VAL_TYPE *tmp = vector_malloc(SROW * sizeof(MAT_VAL_TYPE));
    tmp[0] = sink;
    unsigned int co = dma_p2p(tmp, 1, SROW * sizeof(MAT_VAL_TYPE), 0,
                              &out[tid * SROW], 1,
                              SROW * sizeof(MAT_VAL_TYPE), 0, false, 0);
    dma_wait(co);
    vector_free(tmp);
    vector_free(a);
    vector_free(b);
}
