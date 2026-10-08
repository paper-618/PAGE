#define PAGE_USE_HTHREAD 1
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>

#ifndef MADV_NOHUGEPAGE
#define MADV_NOHUGEPAGE 15
#endif

#include "hthread_host.h"
#include "../../header/common.h"
#include "../../header/mmio_highlevel.h"
#include "../../header/csr_mv.h"
#include "../../header/page_values.h"
#include "../../header/page_format.h"
#include "../../header/page_prep.h"
#include "../../header/page_cpu.h"
#include "../../header/page_numa.h"

#ifndef SPMV_WARMUP
#define SPMV_WARMUP 5
#endif
#ifndef SPMV_ITER
#define SPMV_ITER   10
#endif
#ifndef SPMV_BW_ELEMS
#define SPMV_BW_ELEMS (30 * 1024)
#endif
#ifndef SPMV_BW_REP
#define SPMV_BW_REP 200
#endif

#ifndef SPMV_ARM_CALIB_ITER
#define SPMV_ARM_CALIB_ITER 3
#endif
#ifndef PAGE_MIX_CALIB_SAMPLES
#define PAGE_MIX_CALIB_SAMPLES 3
#endif

enum {
    PAGE_CPU_BASELINE_OPT = 0,
    PAGE_CPU_BASELINE_HALAV_CSR = 1
};

static int g_page_cpu_baseline = PAGE_CPU_BASELINE_OPT;

static const char *page_cpu_baseline_name(void)
{
    return (g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR)
         ? "halav_csr" : "opt";
}

static int page_select_cpu_baseline_from_env(void)
{
    const char *env = getenv("PAGE_CPU_BASELINE");
    const char *v = (env && *env) ? env : "opt";
    if (strcmp(v, "halav_csr") == 0 || strcmp(v, "csr_ref") == 0 ||
        strcmp(v, "ref") == 0) {
        g_page_cpu_baseline = PAGE_CPU_BASELINE_HALAV_CSR;
        return 0;
    }
    if (strcmp(v, "opt") == 0 || strcmp(v, "optimized") == 0) {
        g_page_cpu_baseline = PAGE_CPU_BASELINE_OPT;
        return 0;
    }
    fprintf(stderr, "[CPU_BASELINE] ERROR: PAGE_CPU_BASELINE must be opt|halav_csr\n");
    return -1;
}

static inline void page_cpu_exec_full(const CSRMatrix *csr,
                                       const MAT_VAL_TYPE *x,
                                       MAT_VAL_TYPE *y)
{
    if (g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR)
        csr_mv(csr->numRows, csr->numCols, csr->rowPointers, csr->colIndices,
               csr->values, (MAT_VAL_TYPE *)x, y);
    else
        page_cpu_spmv_overt(csr, x, y);
}

static inline void page_cpu_exec_rows(const CSRMatrix *csr,
                                       const MAT_VAL_TYPE *x,
                                       MAT_VAL_TYPE *y,
                                       int r0, int r1)
{
    if (g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR) {
        if (r0 < 0) r0 = 0;
        if (r1 > csr->numRows) r1 = csr->numRows;
        if (r1 > r0)
            csrSpa_mv(r1 - r0, csr->numCols, r0, csr->rowPointers + r0,
                      csr->colIndices, csr->values, (MAT_VAL_TYPE *)x, y);
    } else {
        page_cpu_spmv_rows(csr, x, y, r0, r1);
    }
}

static inline void page_cpu_exec_residual(const page_matrix *A,
                                           const MAT_VAL_TYPE *x,
                                           MAT_VAL_TYPE *y_res)
{
    if (g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR) {
        if (A && x && y_res && A->has_window && A->res_nnz > 0)
            csrSpa_mv(A->m_dsp, A->n, 0, A->res_rp, A->res_ci, A->res_val,
                      (MAT_VAL_TYPE *)x, y_res);
    } else {
        page_cpu_spmv_residual(A, x, y_res);
    }
}

static inline void page_cpu_calib_full(const CSRMatrix *csr,
                                        const MAT_VAL_TYPE *x,
                                        MAT_VAL_TYPE *y)
{
    if (g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR)
        csr_mv(csr->numRows, csr->numCols, csr->rowPointers, csr->colIndices,
               csr->values, (MAT_VAL_TYPE *)x, y);
    else
        page_cpu_spmv_overt_general(csr, x, y);
}

static inline void page_cpu_calib_rows(const CSRMatrix *csr,
                                        const MAT_VAL_TYPE *x,
                                        MAT_VAL_TYPE *y,
                                        int r0, int r1)
{
    if (g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR) {
        if (r0 < 0) r0 = 0;
        if (r1 > csr->numRows) r1 = csr->numRows;
        if (r1 > r0)
            csrSpa_mv(r1 - r0, csr->numCols, r0, csr->rowPointers + r0,
                      csr->colIndices, csr->values, (MAT_VAL_TYPE *)x, y);
    } else {
        page_cpu_spmv_rows_general(csr, x, y, r0, r1);
    }
}

static double now_ms(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000.0 + t.tv_usec / 1000.0;
}

static int cmp_double(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_small(double *v, int n)
{
    if (n <= 0) return 0.0;
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

typedef struct {
    double median;
    double mad;
    double rel_sigma;
    int n;
} page_robust_stat;

static page_robust_stat page_robust_stats(const double *src, int n)
{
    page_robust_stat r; memset(&r, 0, sizeof(r));
    if (!src || n <= 0) return r;
    if (n > 16) n = 16;
    double a[16], d[16];
    for (int i = 0; i < n; i++) a[i] = src[i];
    r.median = median_small(a, n);
    for (int i = 0; i < n; i++) d[i] = fabs(src[i] - r.median);
    r.mad = median_small(d, n);
    r.rel_sigma = (r.median > 0.0) ? 1.4826 * r.mad / r.median : 0.0;
    r.n = n;
    return r;
}

static double page_fuse_bw(double bw0, double rel0, double bw1, double rel1)
{
    if (!(bw0 > 0.0) || !isfinite(bw0)) return bw1;
    if (!(bw1 > 0.0) || !isfinite(bw1)) return bw0;

    const double between = 0.5 * fabs(log(bw1 / bw0));
    const double r0 = sqrt(rel0 * rel0 + between * between);
    const double r1 = sqrt(rel1 * rel1 + between * between);
    const double v0 = bw0 * bw0 * r0 * r0 + 1e-12 * bw0 * bw0;
    const double v1 = bw1 * bw1 * r1 * r1 + 1e-12 * bw1 * bw1;
    const double w0 = 1.0 / v0, w1 = 1.0 / v1;
    return (w0 * bw0 + w1 * bw1) / (w0 + w1);
}

static double trimmed_mean_1each(const double *v, int n)
{
    if (n <= 0) return 0.0;
    double sum = 0.0, mn = v[0], mx = v[0];
    for (int i = 0; i < n; i++) {
        const double x = v[i];
        sum += x;
        if (x < mn) mn = x;
        if (x > mx) mx = x;
    }
    return (n > 2) ? (sum - mn - mx) / (double)(n - 2)
                   : sum / (double)n;
}

static void time_stats(double *v, int n, double *trimmed, double *raw_mean,
                       double *medi, double *mn, double *mx, double *cvpct)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v[i];
    *raw_mean = s / (double)n;
    *trimmed = trimmed_mean_1each(v, n);
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    *medi = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    *mn = v[0];
    *mx = v[n - 1];
    double var = 0.0;
    for (int i = 0; i < n; i++) var += (v[i] - *raw_mean) * (v[i] - *raw_mean);
    var = (n > 1) ? var / (double)(n - 1) : 0.0;
    *cvpct = (*raw_mean > 0.0) ? sqrt(var) / *raw_mean * 100.0 : 0.0;
}

typedef struct {
    double bw_probe_gbs;
    double bw_xload;
    double xload_fixed_ms;
    double bw_pipe;
    double launch_ms;
    double barrier_ms;
} page_platform_calib;

static void page_runtime_host(char *buf, size_t cap)
{
    const char *h = getenv("SLURMD_NODENAME");
    if (!h || !*h) h = getenv("HOSTNAME");
    if (!h || !*h) h = getenv("SLURM_NODELIST");
    if (!h || !*h) h = "unknown";
    if (cap > 0) {
        snprintf(buf, cap, "%s", h);
        buf[cap - 1] = '\0';
    }
}

static int load_platform_cache(const char *path, int coreNum, int cluster_id,
                               page_platform_calib *c)
{
    if (!path || !*path || !c) return 0;
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char tag[32], host_file[128], host_now[128] = {0};
    int core = 0, cluster = 0;
    page_runtime_host(host_now, sizeof(host_now));
    const int n = fscanf(f, "%31s %127s %d %d %lf %lf %lf %lf %lf %lf",
                         tag, host_file, &core, &cluster,
                         &c->bw_probe_gbs, &c->bw_xload, &c->xload_fixed_ms,
                         &c->bw_pipe, &c->launch_ms, &c->barrier_ms);
    fclose(f);
    if (n != 10 || strcmp(tag, "PAGE_PLATFORM_V1") != 0) return 0;
    if (strcmp(host_file, host_now) != 0 || core != coreNum || cluster != cluster_id) return 0;
    if (!(c->bw_probe_gbs > 0.0) || !(c->bw_xload > 0.0) || !(c->bw_pipe > 0.0)) return 0;
    return 1;
}

static void save_platform_cache(const char *path, int coreNum, int cluster_id,
                                const page_platform_calib *c)
{
    if (!path || !*path || !c) return;
    FILE *f = fopen(path, "w");
    if (!f) return;
    char host[128] = {0};
    page_runtime_host(host, sizeof(host));
    fprintf(f, "PAGE_PLATFORM_V1 %s %d %d %.17g %.17g %.17g %.17g %.17g %.17g\n",
            host, coreNum, cluster_id, c->bw_probe_gbs, c->bw_xload,
            c->xload_fixed_ms, c->bw_pipe, c->launch_ms, c->barrier_ms);
    fclose(f);
}

#ifndef PAGE_PREFIX_CALIB_ITER
#define PAGE_PREFIX_CALIB_ITER 3
#endif

static double calib_arm_prefix_ms(const CSRMatrix *csr, const MAT_VAL_TYPE *x,
                                  MAT_VAL_TYPE *tmp, int row_off,
                                  double *mad_out, double *rel_sigma_out)
{
    if (mad_out) *mad_out = 0.0;
    if (rel_sigma_out) *rel_sigma_out = 0.0;
    if (!csr || !x || !tmp || row_off <= 0) return 0.0;
    if (row_off > csr->numRows) row_off = csr->numRows;
    page_cpu_calib_rows(csr, x, tmp, 0, row_off);
    double samples[PAGE_PREFIX_CALIB_ITER];
    for (int i = 0; i < PAGE_PREFIX_CALIB_ITER; i++) {
        const double ta = now_ms();
        page_cpu_calib_rows(csr, x, tmp, 0, row_off);
        samples[i] = now_ms() - ta;
    }
    const page_robust_stat rs = page_robust_stats(samples, PAGE_PREFIX_CALIB_ITER);
    if (mad_out) *mad_out = rs.mad;
    if (rel_sigma_out) *rel_sigma_out = rs.rel_sigma;
    return rs.median;
}

static double page_prefix_model_bytes(const CSRMatrix *csr, int row_off)
{
    if (!csr || row_off <= 0) return 0.0;
    if (row_off > csr->numRows) row_off = csr->numRows;
    const long long nnz = (long long)csr->rowPointers[row_off] - csr->rowPointers[0];
    if (nnz <= 0) return 0.0;
    const double x = (csr->numCols > 0) ? (double)nnz / (double)csr->numCols : 0.0;
    double uniq = (csr->numCols > 0) ? (double)csr->numCols * (1.0 - exp(-x)) : 0.0;
    if (uniq > (double)csr->numCols) uniq = (double)csr->numCols;
    if (uniq > (double)nnz) uniq = (double)nnz;
    return (double)nnz * 12.0 + (double)row_off * 8.0 + uniq * 8.0;
}

static double run_stream_probe(int coreNum, int cluster_id)
{
    const char *enabled = getenv("SPMV_BW_PROBE");
    if (enabled && strcmp(enabled, "0") == 0) {
        printf("[BW] stream_bw: 0.00 (GB/s) disabled\n");
        return 0.0;
    }
    const int nelem = SPMV_BW_ELEMS;
    const int nbytes = nelem * (int)sizeof(MAT_VAL_TYPE);
    MAT_VAL_TYPE *src = (MAT_VAL_TYPE *)hthread_malloc(cluster_id,
                            (size_t)nbytes, HT_MEM_RW);
    if (!src) {
        fprintf(stderr, "[BW] stream probe allocation failed\n");
        return 0.0;
    }
    for (int i = 0; i < nelem; i++) src[i] = (MAT_VAL_TYPE)(i & 255);
    const int gid = hthread_group_create(cluster_id, coreNum);
    unsigned long args[3];
    args[0] = SPMV_BW_REP; args[1] = nbytes; args[2] = (unsigned long)src;
    hthread_group_exec(gid, "page_bench_stream", 2, 1, args);
    hthread_group_wait(gid);
    const double ta = now_ms();
    hthread_group_exec(gid, "page_bench_stream", 2, 1, args);
    hthread_group_wait(gid);
    const double elapsed = now_ms() - ta;
    const double bw = (elapsed > 0.0)
        ? (double)SPMV_BW_REP * nbytes * coreNum / (elapsed * 1e6) : 0.0;
    printf("[BW] stream_time: %.3f (ms)\n", elapsed);
    printf("[BW] stream_bw: %.2f (GB/s)\n", bw);
    printf("[BW] stream_bytes: %.0f\n", (double)SPMV_BW_REP * nbytes * coreNum);
    hthread_group_destroy(gid);
    hthread_free(src);
    return bw;
}

static double calib_xload_path(int coreNum, int cluster_id, double *fixed_ms_out)
{
    const int small_bytes = PAGE_XBLOCK_BYTES;
    const int large_bytes = 128 * 1024;
    const int large_elems = large_bytes / (int)sizeof(MAT_VAL_TYPE);
    const size_t total_elems = (size_t)coreNum * (size_t)large_elems;
    MAT_VAL_TYPE *src = (MAT_VAL_TYPE *)hthread_malloc(cluster_id,
                            total_elems * sizeof(MAT_VAL_TYPE), HT_MEM_RW);
    if (!src) {
        if (fixed_ms_out) *fixed_ms_out = 0.0;
        return PAGE_BW_DDR;
    }
    for (size_t i = 0; i < total_elems; i++) src[i] = (MAT_VAL_TYPE)(i & 255);
    const int gid = hthread_group_create(cluster_id, coreNum);
    unsigned long args[4];

    const int nlarge = 128;
    args[0] = (unsigned long)nlarge;
    args[1] = (unsigned long)large_bytes;
    args[2] = (unsigned long)src;
    hthread_group_exec(gid, "page_bench_xload", 2, 1, args);
    hthread_group_wait(gid);
    double ta = now_ms();
    hthread_group_exec(gid, "page_bench_xload", 2, 1, args);
    hthread_group_wait(gid);
    const double el_large = now_ms() - ta;

    const double bw = (el_large > 0.0)
        ? (double)nlarge * large_bytes * coreNum * 1e3 / el_large
        : 0.0;

    const int nsmall = 4096;
    args[0] = (unsigned long)nsmall;
    args[1] = (unsigned long)small_bytes;
    hthread_group_exec(gid, "page_bench_xload", 2, 1, args);
    hthread_group_wait(gid);
    ta = now_ms();
    hthread_group_exec(gid, "page_bench_xload", 2, 1, args);
    hthread_group_wait(gid);
    const double el_small = now_ms() - ta;
    const double per_small = (el_small > 0.0) ? el_small / (double)nsmall : 0.0;
    const double transfer_small = (bw > 0.0)
        ? ((double)small_bytes * coreNum) / bw * 1e3 : 0.0;
    double fixed = per_small - transfer_small;
    if (fixed < 0.0) fixed = 0.0;
    if (fixed_ms_out) *fixed_ms_out = fixed;
    hthread_group_destroy(gid);
    hthread_free(src);
    printf("[BW] xload_bw: %.2f (GB/s), xload64: %.6f ms/round, fixed: %.6f ms\n",
           bw / 1e9, per_small, fixed);
    printf("[BW] xload_bytes_per_sec: %.6e\n", bw);
    return (bw > 0.0) ? bw : PAGE_BW_DDR;
}

static double calib_dsp_pipeline(int coreNum, int cluster_id,
                                 double *launch_ms_out, double *barrier_ms_out)
{
    const int nelem = CHUNK;
    const int nbytes = nelem * (int)sizeof(MAT_VAL_TYPE);
    MAT_VAL_TYPE *src = (MAT_VAL_TYPE *)hthread_malloc(cluster_id,
                            (size_t)nbytes, HT_MEM_RW);
    int *idx = (int *)hthread_malloc(cluster_id,
                            (size_t)nelem * sizeof(int), HT_MEM_RW);
    MAT_VAL_TYPE *out = (MAT_VAL_TYPE *)hthread_malloc(cluster_id,
                            (size_t)coreNum * SROW * sizeof(MAT_VAL_TYPE), HT_MEM_RW);
    if (!src || !idx || !out) {
        if (src) hthread_free(src);
        if (idx) hthread_free(idx);
        if (out) hthread_free(out);
        if (launch_ms_out) *launch_ms_out = PAGE_KERNEL_FIX_MS;
        if (barrier_ms_out) *barrier_ms_out = PAGE_BARRIER_FIX_MS;
        return PAGE_BW_PIPE;
    }
    for (int i = 0; i < nelem; i++) {
        src[i] = (MAT_VAL_TYPE)(i & 255);
        idx[i] = ((i * 7919) % GSM_X_CAP) * (int)sizeof(MAT_VAL_TYPE);
    }

    const int gid = hthread_group_create(cluster_id, coreNum);
    unsigned long args[8];

    const int nlaunch = 64;
    args[0] = 0;
    for (int i = 0; i < 8; i++) {
        hthread_group_exec(gid, "page_null_kernel", 1, 0, args);
        hthread_group_wait(gid);
    }
    double ta = now_ms();
    for (int i = 0; i < nlaunch; i++) {
        hthread_group_exec(gid, "page_null_kernel", 1, 0, args);
        hthread_group_wait(gid);
    }
    const double launch_ms = (now_ms() - ta) / (double)nlaunch;

    const int nrep = SPMV_BW_REP;
    args[0] = (unsigned long)nrep;
    args[1] = (unsigned long)nbytes;
    args[2] = (unsigned long)src;
    args[3] = (unsigned long)idx;
    args[4] = (unsigned long)out;
    hthread_group_exec(gid, "page_bench_pipeline", 2, 3, args);
    hthread_group_wait(gid);
    ta = now_ms();
    hthread_group_exec(gid, "page_bench_pipeline", 2, 3, args);
    hthread_group_wait(gid);
    const double pipe_ms = now_ms() - ta;
    const double logical_bytes = 2.0 * (double)nrep * (double)nbytes * (double)coreNum;
    double bw_pipe = (pipe_ms > 0.0) ? logical_bytes / (pipe_ms * 1e-3) : PAGE_BW_PIPE;
    if (!(bw_pipe > 0.0) || !isfinite(bw_pipe)) bw_pipe = PAGE_BW_PIPE;

    const int bid = hthread_barrier_create(cluster_id);
    const int nbar = 512;
    args[0] = (unsigned long)nbar;
    args[1] = (unsigned long)bid;
    args[2] = (unsigned long)coreNum;
    hthread_group_exec(gid, "page_bench_barrier", 3, 0, args);
    hthread_group_wait(gid);
    ta = now_ms();
    hthread_group_exec(gid, "page_bench_barrier", 3, 0, args);
    hthread_group_wait(gid);
    const double barrier_ms = (now_ms() - ta) / (double)nbar;
    hthread_barrier_destroy(bid);

    printf("[BW] pipeline_bw: %.2f (GB/s)\n", bw_pipe / 1e9);
    printf("[BW] pipeline_time: %.3f (ms)\n", pipe_ms);
    printf("[BW] launch_ms: %.6f (ms)\n", launch_ms);
    printf("[BW] barrier_ms: %.6f (ms)\n", barrier_ms);

    if (launch_ms_out) *launch_ms_out = launch_ms;
    if (barrier_ms_out) *barrier_ms_out = barrier_ms;
    hthread_group_destroy(gid);
    hthread_free(src); hthread_free(idx); hthread_free(out);
    return bw_pipe;
}

static double calib_arm_bw(const CSRMatrix *csr, const MAT_VAL_TYPE *x,
                           MAT_VAL_TYPE *tmp, double *t_out,
                           double *mad_out, double *rel_sigma_out)
{
    const double bytes = (double)csr->numNonzeros * 12.0
                       + (double)csr->numRows * 8.0
                       + (double)csr->numCols * 8.0;
    page_cpu_calib_full(csr, x, tmp);
    double samples[SPMV_ARM_CALIB_ITER];
    for (int i = 0; i < SPMV_ARM_CALIB_ITER; i++) {
        const double ta = now_ms();
        page_cpu_calib_full(csr, x, tmp);
        samples[i] = now_ms() - ta;
    }
    const page_robust_stat rs = page_robust_stats(samples, SPMV_ARM_CALIB_ITER);
    const double med = rs.median;
    *t_out = med;
    if (mad_out) *mad_out = rs.mad;
    if (rel_sigma_out) *rel_sigma_out = rs.rel_sigma;
    if (!(med > 0.0)) return PAGE_BW_ARM;
    const double bw = bytes / (med * 1e-3);
    if (bw < PAGE_BW_ARM_MIN || bw > PAGE_BW_ARM_MAX) return PAGE_BW_ARM;
    return bw;
}

static double calib_mix_bw(const CSRMatrix *csr, const MAT_VAL_TYPE *x,
                           MAT_VAL_TYPE *tmp, int coreNum, int cluster_id,
                           double bw_pipe_solo, double arm_solo_ms,
                           double *elapsed_out, double *arm_mix_bw_out,
                           double *pipe_mix_bw_out, double *arm_mix_ms_out,
                           double *arm_mix_mad_out, double *arm_mix_rel_sigma_out,
                           double *elapsed_mad_out)
{
    const int nelem = CHUNK;
    const int nbytes = nelem * (int)sizeof(MAT_VAL_TYPE);
    const double arm_bytes = (double)csr->numNonzeros * 12.0
                           + (double)csr->numRows * 8.0
                           + (double)csr->numCols * 8.0;
    const double pipe_ref = (bw_pipe_solo > 0.0) ? bw_pipe_solo : PAGE_BW_PIPE;
    const double target_s = (arm_solo_ms > 0.0) ? arm_solo_ms * 1e-3
                                                : arm_bytes / PAGE_BW_ARM;
    const double bytes_per_rep = 2.0 * (double)nbytes * (double)coreNum;
    long long nr64 = (long long)ceil((pipe_ref * target_s) / bytes_per_rep);
    if (nr64 < 1) nr64 = 1;
    if (nr64 > INT_MAX) nr64 = INT_MAX;
    const int nrep = (int)nr64;

    MAT_VAL_TYPE *src = (MAT_VAL_TYPE *)hthread_malloc(cluster_id,
                            (size_t)nbytes, HT_MEM_RW);
    int *idx = (int *)hthread_malloc(cluster_id,
                            (size_t)nelem * sizeof(int), HT_MEM_RW);
    MAT_VAL_TYPE *out = (MAT_VAL_TYPE *)hthread_malloc(cluster_id,
                            (size_t)coreNum * SROW * sizeof(MAT_VAL_TYPE), HT_MEM_RW);
    if (!src || !idx || !out) {
        if (src) hthread_free(src);
        if (idx) hthread_free(idx);
        if (out) hthread_free(out);
        if (elapsed_out) *elapsed_out = 0.0;
        if (arm_mix_bw_out) *arm_mix_bw_out = (arm_solo_ms > 0.0)
            ? arm_bytes / (arm_solo_ms * 1e-3) : PAGE_BW_ARM;
        if (pipe_mix_bw_out) *pipe_mix_bw_out = pipe_ref;
        if (arm_mix_ms_out) *arm_mix_ms_out = arm_solo_ms;
        if (arm_mix_mad_out) *arm_mix_mad_out = 0.0;
        if (arm_mix_rel_sigma_out) *arm_mix_rel_sigma_out = 0.0;
        if (elapsed_mad_out) *elapsed_mad_out = 0.0;
        return pipe_ref;
    }
    for (int i = 0; i < nelem; i++) {
        src[i] = (MAT_VAL_TYPE)(i & 255);
        idx[i] = ((i * 7919) % GSM_X_CAP) * (int)sizeof(MAT_VAL_TYPE);
    }

    const int gid = hthread_group_create(cluster_id, coreNum);
    unsigned long args[5];
    args[0] = (unsigned long)nrep;
    args[1] = (unsigned long)nbytes;
    args[2] = (unsigned long)src;
    args[3] = (unsigned long)idx;
    args[4] = (unsigned long)out;

    double elapsed_samples[PAGE_MIX_CALIB_SAMPLES];
    double arm_samples[PAGE_MIX_CALIB_SAMPLES];
    for (int si = 0; si < PAGE_MIX_CALIB_SAMPLES; si++) {
        const double ta = now_ms();
        hthread_group_exec(gid, "page_bench_pipeline", 2, 3, args);
        const double ar0 = now_ms();
        page_cpu_calib_full(csr, x, tmp);
        const double ar1 = now_ms();
        hthread_group_wait(gid);
        elapsed_samples[si] = now_ms() - ta;
        arm_samples[si] = ar1 - ar0;
    }
    const page_robust_stat ers = page_robust_stats(elapsed_samples, PAGE_MIX_CALIB_SAMPLES);
    const page_robust_stat ars = page_robust_stats(arm_samples, PAGE_MIX_CALIB_SAMPLES);
    const double elapsed = ers.median;
    const double arm_mix_ms = ars.median;

    const double dsp_bytes = (double)nrep * bytes_per_rep;
    double bw_mix = (elapsed > 0.0)
                  ? (dsp_bytes + arm_bytes) / (elapsed * 1e-3) : pipe_ref;
    double bw_arm_mix = (arm_mix_ms > 0.0)
                      ? arm_bytes / (arm_mix_ms * 1e-3) : PAGE_BW_ARM;

    double bw_pipe_mix = (elapsed > 0.0)
                       ? dsp_bytes / (elapsed * 1e-3) : pipe_ref;
    if (!(bw_mix > 0.0) || !isfinite(bw_mix)) bw_mix = pipe_ref;
    if (!(bw_arm_mix > 0.0) || !isfinite(bw_arm_mix)) bw_arm_mix = PAGE_BW_ARM;
    if (!(bw_pipe_mix > 0.0) || !isfinite(bw_pipe_mix)) bw_pipe_mix = pipe_ref;

    if (elapsed_out) *elapsed_out = elapsed;
    if (arm_mix_bw_out) *arm_mix_bw_out = bw_arm_mix;
    if (pipe_mix_bw_out) *pipe_mix_bw_out = bw_pipe_mix;
    if (arm_mix_ms_out) *arm_mix_ms_out = arm_mix_ms;
    if (arm_mix_mad_out) *arm_mix_mad_out = ars.mad;
    if (arm_mix_rel_sigma_out) *arm_mix_rel_sigma_out = ars.rel_sigma;
    if (elapsed_mad_out) *elapsed_mad_out = ers.mad;
    printf("[BW] mix_bw: %.2f (GB/s)\n", bw_mix / 1e9);
    printf("[BW] arm_mix_bw: %.2f (GB/s)\n", bw_arm_mix / 1e9);
    printf("[BW] pipeline_mix_bw: %.2f (GB/s)\n", bw_pipe_mix / 1e9);
    printf("[BW] mix_time: %.3f (ms)\n", elapsed);
    printf("[BW] mix_arm_time: %.3f (ms)\n", arm_mix_ms);
    printf("[BW] mix_dsp_reps: %d\n", nrep);
    printf("[BW] mix_bytes: %.0f\n", dsp_bytes + arm_bytes);
    printf("[STABLE_CAL] mix_samples=%d elapsed_median_ms=%.6f elapsed_mad_ms=%.6f "
           "arm_mix_median_ms=%.6f arm_mix_mad_ms=%.6f arm_mix_rel_sigma=%.8f\n",
           PAGE_MIX_CALIB_SAMPLES, elapsed, ers.mad, arm_mix_ms, ars.mad, ars.rel_sigma);

    hthread_group_destroy(gid);
    hthread_free(src); hthread_free(idx); hthread_free(out);
    return bw_mix;
}

#ifndef PAGE_VALUE_SELECT_TRIALS
#define PAGE_VALUE_SELECT_TRIALS 3
#endif

static int page_select_cpu_value_mode(CSRMatrix *csr,
                                       int candidate_mode,
                                       MAT_VAL_TYPE candidate_constant,
                                       const MAT_VAL_TYPE *x,
                                       MAT_VAL_TYPE *tmp,
                                       double *general_ms_out,
                                       double *candidate_ms_out)
{
    if (general_ms_out) *general_ms_out = 0.0;
    if (candidate_ms_out) *candidate_ms_out = 0.0;
    if (g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR) {
        if (csr) {
            csr->value_mode = PAGE_VALUE_GENERAL;
            csr->constant_value = 0.0;
        }
        return PAGE_VALUE_GENERAL;
    }
    if (!csr || !x || !tmp || candidate_mode == PAGE_VALUE_GENERAL) {
        if (csr) { csr->value_mode = PAGE_VALUE_GENERAL; csr->constant_value = 0.0; }
        return PAGE_VALUE_GENERAL;
    }

    const int saved_mode = csr->value_mode;
    const MAT_VAL_TYPE saved_const = csr->constant_value;
    double g[PAGE_VALUE_SELECT_TRIALS], c[PAGE_VALUE_SELECT_TRIALS];

    csr->value_mode = PAGE_VALUE_GENERAL;
    csr->constant_value = 0.0;
    page_cpu_spmv_overt(csr, x, tmp);

    csr->value_mode = candidate_mode;
    csr->constant_value = candidate_constant;
    page_cpu_spmv_overt(csr, x, tmp);

    for (int it = 0; it < PAGE_VALUE_SELECT_TRIALS; it++) {
        if ((it & 1) == 0) {
            csr->value_mode = PAGE_VALUE_GENERAL;
            csr->constant_value = 0.0;
            double t0 = now_ms();
            page_cpu_spmv_overt(csr, x, tmp);
            g[it] = now_ms() - t0;

            csr->value_mode = candidate_mode;
            csr->constant_value = candidate_constant;
            t0 = now_ms();
            page_cpu_spmv_overt(csr, x, tmp);
            c[it] = now_ms() - t0;
        } else {
            csr->value_mode = candidate_mode;
            csr->constant_value = candidate_constant;
            double t0 = now_ms();
            page_cpu_spmv_overt(csr, x, tmp);
            c[it] = now_ms() - t0;

            csr->value_mode = PAGE_VALUE_GENERAL;
            csr->constant_value = 0.0;
            t0 = now_ms();
            page_cpu_spmv_overt(csr, x, tmp);
            g[it] = now_ms() - t0;
        }
    }

    const double gmed = median_small(g, PAGE_VALUE_SELECT_TRIALS);
    const double cmed = median_small(c, PAGE_VALUE_SELECT_TRIALS);
    if (general_ms_out) *general_ms_out = gmed;
    if (candidate_ms_out) *candidate_ms_out = cmed;

    if (cmed > 0.0 && cmed < gmed) {
        csr->value_mode = candidate_mode;
        csr->constant_value = candidate_constant;
        return candidate_mode;
    }

    csr->value_mode = PAGE_VALUE_GENERAL;
    csr->constant_value = 0.0;
    (void)saved_mode;
    (void)saved_const;
    return PAGE_VALUE_GENERAL;
}

#ifndef PAGE_PILOT_WARMUP
#define PAGE_PILOT_WARMUP 1
#endif
#ifndef PAGE_PILOT_TRIALS
#define PAGE_PILOT_TRIALS 3
#endif

static double page_pilot_plan(const CSRMatrix *csr, const page_matrix *A,
                               const MAT_VAL_TYPE *cpu_x,
                               const MAT_VAL_TYPE *dsp_x,
                               int coreNum, int cluster_id)
{
    const int m = csr->numRows;
    double samples[PAGE_PILOT_TRIALS];

    if (A->mode == PAGE_MODE_CPU) {
        MAT_VAL_TYPE *y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
        if (!y) return 1.0e300;
        for (int i = 0; i < PAGE_PILOT_WARMUP; i++)
            page_cpu_exec_full(csr, cpu_x, y);
        for (int i = 0; i < PAGE_PILOT_TRIALS; i++) {
            const double ta = now_ms();
            page_cpu_exec_full(csr, cpu_x, y);
            samples[i] = now_ms() - ta;
        }
        free(y);
        return median_small(samples, PAGE_PILOT_TRIALS);
    }

    const size_t yd_elems = (size_t)A->nvrow + SROW;
    const size_t cap = (size_t)A->row_off + yd_elems;
    MAT_VAL_TYPE *ybuf = (MAT_VAL_TYPE *)hthread_malloc(cluster_id,
                              sizeof(MAT_VAL_TYPE) * cap, HT_MEM_RW);
    MAT_VAL_TYPE *y_res = NULL;
    if (!ybuf) return 1.0e300;
    MAT_VAL_TYPE *yd = ybuf + A->row_off;
    MAT_VAL_TYPE *y = ybuf;
    memset(ybuf, 0, sizeof(MAT_VAL_TYPE) * cap);
    if (A->has_window && A->res_nnz > 0) {
        y_res = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)A->m_dsp);
        if (!y_res) { hthread_free(ybuf); return 1.0e300; }
    }

    const int gid = hthread_group_create(cluster_id, coreNum);
    const int bid = hthread_barrier_create(cluster_id);
    MAT_VAL_TYPE *dsp_const = NULL;
    if (A->dsp_value_mode == PAGE_VALUE_CONSTANT) {
        dsp_const = (MAT_VAL_TYPE *)hthread_malloc(cluster_id, sizeof(MAT_VAL_TYPE), HT_MEM_RW);
        if (!dsp_const) {
            hthread_barrier_destroy(bid); hthread_group_destroy(gid);
            free(y_res); hthread_free(ybuf); return 1.0e300;
        }
        dsp_const[0] = A->dsp_constant_value;
    }
    unsigned long args[32];
    args[0]  = (unsigned long)A->nvrow;
    args[1]  = (unsigned long)A->n;
    args[2]  = (unsigned long)coreNum;
    args[3]  = (unsigned long)A->npanel;
    args[4]  = (unsigned long)bid;
    args[5]  = (unsigned long)A->ntile;
    args[6]  = (unsigned long)A->idx_bytes;
    args[7]  = (unsigned long)A->dsp_value_mode;
    args[8]  = (unsigned long)A->panel_slice_lo;
    args[9]  = (unsigned long)A->panel_tile_lo;
    args[10] = (unsigned long)A->tile_xlo;
    args[11] = (unsigned long)A->tile_xhi;
    args[12] = (unsigned long)A->tile_inst_lo;
    args[13] = (unsigned long)A->tile_packed;
    args[14] = (unsigned long)A->tile_xmap_lo;
    args[15] = (unsigned long)A->tile_xmap;
    args[16] = (unsigned long)A->cl;
    args[17] = (unsigned long)A->cs;
    args[18] = (unsigned long)A->val;
    args[19] = (unsigned long)A->cidx;
    args[20] = (unsigned long)A->thread_ptr;
    args[21] = (unsigned long)A->gsmxwidth;
    args[22] = (unsigned long)dsp_x;
    args[23] = (unsigned long)yd;
    args[24] = (unsigned long)dsp_const;
    const int scalArgs = 8, vecArgs = 17;

    for (int it = 0; it < PAGE_PILOT_WARMUP + PAGE_PILOT_TRIALS; it++) {
        const double ta = now_ms();
        hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, args);
        page_cpu_exec_rows(csr, cpu_x, y, 0, A->row_off);
        page_cpu_exec_residual(A, cpu_x, y_res);
        hthread_group_wait(gid);
        page_finalize_y(A, yd, y_res, y);
        const double ms = now_ms() - ta;
        if (it >= PAGE_PILOT_WARMUP) samples[it - PAGE_PILOT_WARMUP] = ms;
    }

    hthread_barrier_destroy(bid);
    hthread_group_destroy(gid);
    free(y_res);
    if (dsp_const) hthread_free(dsp_const);
    hthread_free(ybuf);
    return median_small(samples, PAGE_PILOT_TRIALS);
}

static int page_hybrid_has_cpu_work(const page_matrix *A)
{
    if (!A || A->mode == PAGE_MODE_CPU || A->nnz_dsp <= 0) return 0;
    return (A->row_off > 0) || (A->has_window && A->res_nnz > 0);
}

static int page_hybrid_add_thread_candidate(int *cand, int n, int cap, int t)
{
    if (t < 1) t = 1;
    for (int i = 0; i < n; i++) if (cand[i] == t) return n;
    if (n < cap) cand[n++] = t;
    return n;
}

static double page_median3(double a, double b, double c)
{
    double v[3] = {a, b, c};
    return median_small(v, 3);
}

static int page_select_hybrid_threads(const CSRMatrix *csr, page_matrix *A,
                                       const MAT_VAL_TYPE *cpu_x,
                                       const MAT_VAL_TYPE *dsp_x, int coreNum,
                                       int cluster_id, int base_threads,
                                       double *selected_ms_out,
                                       double *base_ms_out,
                                       double *calib_ms_out)
{
    if (selected_ms_out) *selected_ms_out = -1.0;
    if (base_ms_out) *base_ms_out = -1.0;
    if (calib_ms_out) *calib_ms_out = 0.0;
    if (!page_hybrid_has_cpu_work(A)) return base_threads;
    if (base_threads < 1) base_threads = 1;

    int cand[4], nc = 0;
    nc = page_hybrid_add_thread_candidate(cand, nc, 4, (base_threads + 3) / 4);
    nc = page_hybrid_add_thread_candidate(cand, nc, 4, (base_threads + 1) / 2);
    nc = page_hybrid_add_thread_candidate(cand, nc, 4, (3 * base_threads + 3) / 4);
    nc = page_hybrid_add_thread_candidate(cand, nc, 4, base_threads);

    const unsigned long long feature_key =
        (unsigned long long)(unsigned)csr->numRows * 11400714819323198485ull ^
        (unsigned long long)(unsigned)csr->numCols * 14029467366897019727ull ^
        (unsigned long long)csr->numNonzeros;
    if (nc > 1) {
        int rotated[4];
        const int shift = (int)(feature_key % (unsigned long long)nc);
        for (int i = 0; i < nc; i++) rotated[i] = cand[(i + shift) % nc];
        for (int i = 0; i < nc; i++) cand[i] = rotated[i];
    }

    const double tc0 = now_ms();
    double first_ms[4];
    int best_i = -1, base_i = -1;
    double best_ms = 1.0e300;
    omp_set_dynamic(0);
    for (int i = 0; i < nc; i++) {
        omp_set_num_threads(cand[i]);
        first_ms[i] = page_pilot_plan(csr, A, cpu_x, dsp_x, coreNum, cluster_id);
        printf("[HYBRID_V1] stage=scan threads=%d pilot_ms=%.6f\n",
               cand[i], first_ms[i]);
        if (cand[i] == base_threads) base_i = i;
        if (first_ms[i] < best_ms) { best_ms = first_ms[i]; best_i = i; }
    }
    if (base_i < 0) {

        base_i = nc - 1;
    }

    int selected = cand[best_i];
    double selected_ms = first_ms[best_i];
    double base_ms = first_ms[base_i];

    if (selected != base_threads) {
        double b2, b3, w2, w3;

        const int winner_first = (int)(feature_key & 1ull);
        if (winner_first) {
            omp_set_num_threads(selected);     w2 = page_pilot_plan(csr, A, cpu_x, dsp_x, coreNum, cluster_id);
            omp_set_num_threads(base_threads); b2 = page_pilot_plan(csr, A, cpu_x, dsp_x, coreNum, cluster_id);
            omp_set_num_threads(base_threads); b3 = page_pilot_plan(csr, A, cpu_x, dsp_x, coreNum, cluster_id);
            omp_set_num_threads(selected);     w3 = page_pilot_plan(csr, A, cpu_x, dsp_x, coreNum, cluster_id);
        } else {
            omp_set_num_threads(base_threads); b2 = page_pilot_plan(csr, A, cpu_x, dsp_x, coreNum, cluster_id);
            omp_set_num_threads(selected);     w2 = page_pilot_plan(csr, A, cpu_x, dsp_x, coreNum, cluster_id);
            omp_set_num_threads(selected);     w3 = page_pilot_plan(csr, A, cpu_x, dsp_x, coreNum, cluster_id);
            omp_set_num_threads(base_threads); b3 = page_pilot_plan(csr, A, cpu_x, dsp_x, coreNum, cluster_id);
        }
        const double bmed = page_median3(base_ms, b2, b3);
        const double wmed = page_median3(selected_ms, w2, w3);
        printf("[HYBRID_V1] stage=confirm base_threads=%d base_samples_ms=%.6f,%.6f,%.6f "
               "candidate_threads=%d candidate_samples_ms=%.6f,%.6f,%.6f "
               "base_median_ms=%.6f candidate_median_ms=%.6f\n",
               base_threads, base_ms, b2, b3,
               selected, selected_ms, w2, w3, bmed, wmed);
        base_ms = bmed;
        selected_ms = wmed;
        if (!(wmed < bmed)) {
            selected = base_threads;
            selected_ms = bmed;
        }
    }

    omp_set_num_threads(selected);
    const double calib_ms = now_ms() - tc0;
    if (selected_ms_out) *selected_ms_out = selected_ms;
    if (base_ms_out) *base_ms_out = base_ms;
    if (calib_ms_out) *calib_ms_out = calib_ms;
    printf("[HYBRID_V1] stage=chosen base_threads=%d selected_threads=%d "
           "base_pilot_ms=%.6f selected_pilot_ms=%.6f predicted_gain=%.8f "
           "calib_ms=%.3f candidates=%d\n",
           base_threads, selected, base_ms, selected_ms,
           (selected_ms > 0.0 ? base_ms / selected_ms : 0.0), calib_ms, nc);
    return selected;
}

static int page_apply_hybrid_thread_policy(const CSRMatrix *csr, page_matrix *A,
                                            const MAT_VAL_TYPE *cpu_x,
                                            const MAT_VAL_TYPE *dsp_x, int coreNum,
                                            int cluster_id, int base_threads,
                                            int *selected_threads_out,
                                            double *selected_ms_out,
                                            double *base_ms_out,
                                            double *calib_ms_out)
{
    if (selected_threads_out) *selected_threads_out = base_threads;
    if (selected_ms_out) *selected_ms_out = -1.0;
    if (base_ms_out) *base_ms_out = -1.0;
    if (calib_ms_out) *calib_ms_out = 0.0;

    const char *e = getenv("PAGE_HYBRID_THREADS");
    const char *policy = (e && *e) ? e : "base";
    if (!page_hybrid_has_cpu_work(A)) {
        omp_set_num_threads(base_threads);
        printf("[HYBRID_V1] policy=%s active=0 reason=no_hybrid_cpu_work base_threads=%d\n",
               policy, base_threads);
        return 0;
    }

    int selected = base_threads;
    if (strcmp(policy, "auto") == 0) {
        selected = page_select_hybrid_threads(csr, A, cpu_x, dsp_x, coreNum, cluster_id,
                                               base_threads, selected_ms_out,
                                               base_ms_out, calib_ms_out);
    } else if (strcmp(policy, "base") == 0 || strcmp(policy, "off") == 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(base_threads);
        printf("[HYBRID_V1] policy=%s active=1 selected_threads=%d calibration=0\n",
               policy, base_threads);
    } else {
        char *end = NULL;
        long v = strtol(policy, &end, 10);
        if (!end || *end != '\0' || v < 1 || v > base_threads) {
            fprintf(stderr, "[HYBRID_V1] ERROR: PAGE_HYBRID_THREADS must be auto|base|off|1..%d\n",
                    base_threads);
            return -1;
        }
        selected = (int)v;
        omp_set_dynamic(0);
        omp_set_num_threads(selected);
        printf("[HYBRID_V1] policy=fixed active=1 selected_threads=%d base_threads=%d calibration=0\n",
               selected, base_threads);
    }
    if (selected_threads_out) *selected_threads_out = selected;
    return 0;
}

static void page_release_candidate(page_matrix *A, page_stat *st)
{
    if (A->mode != PAGE_MODE_CPU) page_free_matrix(A);
    page_free_stat(st);
}

typedef struct {
    int valid;
    double total_ms, total_mad_ms;
    double submit_ms, cpu_part_ms, wait_ms, dsp_elapsed_ms, finalize_ms;
} page_v15_pilot_stat;

static int page_v15_pilot_tails(const CSRMatrix *csr, const page_matrix *A,
                                 const MAT_VAL_TYPE *cpu_x,
                                 const MAT_VAL_TYPE *dsp_x,
                                 int coreNum, int cluster_id,
                                 page_v15_pilot_stat *out)
{
    if (out) memset(out, 0, sizeof(*out));
    if (!csr || !A || !out || A->mode == PAGE_MODE_CPU || A->nnz_dsp <= 0)
        return -1;

    const size_t yd_elems = (size_t)A->nvrow + SROW;
    const size_t cap = (size_t)A->row_off + yd_elems;
    MAT_VAL_TYPE *ybuf = (MAT_VAL_TYPE *)hthread_malloc(
        cluster_id, sizeof(MAT_VAL_TYPE) * cap, HT_MEM_RW);
    MAT_VAL_TYPE *y_res = NULL;
    if (!ybuf) return -1;
    MAT_VAL_TYPE *yd = ybuf + A->row_off;
    MAT_VAL_TYPE *y = ybuf;
    memset(ybuf, 0, sizeof(MAT_VAL_TYPE) * cap);
    if (A->has_window && A->res_nnz > 0) {
        y_res = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)A->m_dsp);
        if (!y_res) { hthread_free(ybuf); return -1; }
    }

    const int gid = hthread_group_create(cluster_id, coreNum);
    const int bid = hthread_barrier_create(cluster_id);
    MAT_VAL_TYPE *dsp_const = NULL;
    if (A->dsp_value_mode == PAGE_VALUE_CONSTANT) {
        dsp_const = (MAT_VAL_TYPE *)hthread_malloc(
            cluster_id, sizeof(MAT_VAL_TYPE), HT_MEM_RW);
        if (!dsp_const) {
            hthread_barrier_destroy(bid); hthread_group_destroy(gid);
            free(y_res); hthread_free(ybuf); return -1;
        }
        dsp_const[0] = A->dsp_constant_value;
    }

    unsigned long args[32];
    args[0]=(unsigned long)A->nvrow; args[1]=(unsigned long)A->n;
    args[2]=(unsigned long)coreNum; args[3]=(unsigned long)A->npanel;
    args[4]=(unsigned long)bid; args[5]=(unsigned long)A->ntile;
    args[6]=(unsigned long)A->idx_bytes; args[7]=(unsigned long)A->dsp_value_mode;
    args[8]=(unsigned long)A->panel_slice_lo; args[9]=(unsigned long)A->panel_tile_lo;
    args[10]=(unsigned long)A->tile_xlo; args[11]=(unsigned long)A->tile_xhi;
    args[12]=(unsigned long)A->tile_inst_lo; args[13]=(unsigned long)A->tile_packed;
    args[14]=(unsigned long)A->tile_xmap_lo; args[15]=(unsigned long)A->tile_xmap;
    args[16]=(unsigned long)A->cl; args[17]=(unsigned long)A->cs;
    args[18]=(unsigned long)A->val; args[19]=(unsigned long)A->cidx;
    args[20]=(unsigned long)A->thread_ptr; args[21]=(unsigned long)A->gsmxwidth;
    args[22]=(unsigned long)dsp_x; args[23]=(unsigned long)yd;
    args[24]=(unsigned long)dsp_const;
    const int scalArgs=8, vecArgs=17;

    double total[PAGE_PILOT_TRIALS], submit[PAGE_PILOT_TRIALS];
    double cpu[PAGE_PILOT_TRIALS], wait[PAGE_PILOT_TRIALS];
    double dsp[PAGE_PILOT_TRIALS], fin[PAGE_PILOT_TRIALS];
    for (int it=0; it<PAGE_PILOT_WARMUP+PAGE_PILOT_TRIALS; ++it) {
        const double ta=now_ms();
        hthread_group_exec(gid,"page_spmv",scalArgs,vecArgs,args);
        const double ar0=now_ms();
        page_cpu_exec_rows(csr,cpu_x,y,0,A->row_off);
        page_cpu_exec_residual(A,cpu_x,y_res);
        const double ar2=now_ms();
        hthread_group_wait(gid);
        const double tb=now_ms();
        page_finalize_y(A,yd,y_res,y);
        const double tc=now_ms();
        if (it>=PAGE_PILOT_WARMUP) {
            const int j=it-PAGE_PILOT_WARMUP;
            total[j]=tc-ta; submit[j]=ar0-ta; cpu[j]=ar2-ar0;
            wait[j]=tb-ar2; dsp[j]=tb-ta; fin[j]=tc-tb;
        }
    }
    page_robust_stat rt=page_robust_stats(total,PAGE_PILOT_TRIALS);
    page_robust_stat rs=page_robust_stats(submit,PAGE_PILOT_TRIALS);
    page_robust_stat rc=page_robust_stats(cpu,PAGE_PILOT_TRIALS);
    page_robust_stat rw=page_robust_stats(wait,PAGE_PILOT_TRIALS);
    page_robust_stat rd=page_robust_stats(dsp,PAGE_PILOT_TRIALS);
    page_robust_stat rf=page_robust_stats(fin,PAGE_PILOT_TRIALS);
    out->valid=1; out->total_ms=rt.median; out->total_mad_ms=rt.mad;
    out->submit_ms=rs.median; out->cpu_part_ms=rc.median;
    out->wait_ms=rw.median; out->dsp_elapsed_ms=rd.median;
    out->finalize_ms=rf.median;

    hthread_barrier_destroy(bid); hthread_group_destroy(gid);
    if (dsp_const) hthread_free(dsp_const);
    free(y_res); hthread_free(ybuf);
    return 0;
}

static int page_v15_feedback_rebalance(const CSRMatrix *csr,
                                        page_stat *st, page_matrix *A,
                                        const page_opts *base_opt,
                                        const MAT_VAL_TYPE *cpu_x,
                                        const MAT_VAL_TYPE *dsp_x,
                                        int coreNum, int cluster_id,
                                        double tpre[3], double *elapsed_ms_out)
{
    const double t0=now_ms();
    if (elapsed_ms_out) *elapsed_ms_out=0.0;
    if (!csr || !st || !A || !base_opt) return -1;

    if (A->mode==PAGE_MODE_CPU || A->row_off<=0 || A->nnz_dsp<=0 ||
        A->has_window || A->res_nnz!=0) {
        printf("[V15_REBAL] active=1 applicable=0 reason=non_row_split row_off=%d has_window=%d res_nnz=%lld dsp_nnz=%lld\n",
               A->row_off,A->has_window,A->res_nnz,A->nnz_dsp);
        if (elapsed_ms_out) *elapsed_ms_out=now_ms()-t0;
        return 0;
    }

    const int old_row=A->row_off;
    const long long cpu_nnz=st->row_pre[old_row];
    const long long dsp_nnz=A->nnz_dsp;
    page_v15_pilot_stat base;
    if (cpu_nnz<=0 || dsp_nnz<=0 ||
        page_v15_pilot_tails(csr,A,cpu_x,dsp_x,coreNum,cluster_id,&base)!=0 ||
        !base.valid || !(base.cpu_part_ms>0.0) || !(base.total_ms>0.0)) {
        printf("[V15_REBAL] active=1 applicable=0 reason=pilot_failed\n");
        if (elapsed_ms_out) *elapsed_ms_out=now_ms()-t0;
        return 0;
    }
    if (!(base.wait_ms>0.0)) {
        printf("[V15_REBAL] active=1 applicable=0 reason=dsp_endpoint_censored pilot_total_ms=%.6f cpu_ms=%.6f wait_ms=%.6f\n",
               base.total_ms,base.cpu_part_ms,base.wait_ms);
        if (elapsed_ms_out) *elapsed_ms_out=now_ms()-t0;
        return 0;
    }

    const double dsp_post_ms=base.cpu_part_ms+base.wait_ms;
    const double cpu_rate=base.cpu_part_ms/(double)cpu_nnz;
    const double dsp_rate=dsp_post_ms/(double)dsp_nnz;
    if (!(cpu_rate>0.0) || !(dsp_rate>0.0) || !isfinite(cpu_rate) || !isfinite(dsp_rate)) {
        printf("[V15_REBAL] active=1 applicable=0 reason=invalid_rates\n");
        if (elapsed_ms_out) *elapsed_ms_out=now_ms()-t0;
        return 0;
    }
    const double target_share=dsp_rate/(cpu_rate+dsp_rate);
    const int target_row=page_share_to_rowoff(st,csr->numRows,A->nnz,target_share);
    printf("[V15_REBAL] pilot_old row_off=%d cpu_nnz=%lld dsp_nnz=%lld total_ms=%.6f mad_ms=%.6f submit_ms=%.6f cpu_ms=%.6f wait_ms=%.6f dsp_post_ms=%.6f finalize_ms=%.6f cpu_rate_ns_per_nnz=%.6f dsp_rate_ns_per_nnz=%.6f target_cpu_share=%.8f target_row_off=%d\n",
           old_row,cpu_nnz,dsp_nnz,base.total_ms,base.total_mad_ms,base.submit_ms,
           base.cpu_part_ms,base.wait_ms,dsp_post_ms,base.finalize_ms,
           cpu_rate*1.0e6,dsp_rate*1.0e6,target_share,target_row);
    if (target_row<=0 || target_row==old_row || target_row>=csr->numRows) {
        printf("[V15_REBAL] decision=keep reason=no_distinct_legal_cut old_row=%d target_row=%d\n",old_row,target_row);
        if (elapsed_ms_out) *elapsed_ms_out=now_ms()-t0;
        return 0;
    }

    page_release_candidate(A,st); memset(st,0,sizeof(*st));
    page_opts copt=*base_opt; copt.force_cpu=0; copt.force_hybrid=1;
    copt.cpu_share=-1.0; copt.row_off_override=target_row;
    double ctp[3]={0,0,0};
    page_build(csr,st,A,coreNum,cluster_id,ctp,&copt);
    if (tpre) { tpre[0]=ctp[0]; tpre[1]=ctp[1]; tpre[2]=ctp[2]; }
    const long long cand_cpu_nnz=(A->row_off>0 && st->row_pre)?st->row_pre[A->row_off]:0;
    const int cand_valid=(A->mode!=PAGE_MODE_CPU && A->nnz_dsp>0 && cand_cpu_nnz>0 &&
                          !A->has_window && A->res_nnz==0);
    page_v15_pilot_stat cand; memset(&cand,0,sizeof(cand));
    if (cand_valid)
        page_v15_pilot_tails(csr,A,cpu_x,dsp_x,coreNum,cluster_id,&cand);
    const double sigma_old=1.4826*base.total_mad_ms;
    const double sigma_new=1.4826*cand.total_mad_ms;
    const double combined_sigma=hypot(sigma_old,sigma_new);
    const double gain_ms=base.total_ms-cand.total_ms;
    const int accept=cand_valid && cand.valid && cand.total_ms>0.0 &&
                     cand.total_ms<base.total_ms && gain_ms>combined_sigma;
    printf("[V15_REBAL] pilot_candidate row_off=%d cpu_nnz=%lld dsp_nnz=%lld valid=%d total_ms=%.6f mad_ms=%.6f cpu_ms=%.6f wait_ms=%.6f gain_ms=%.6f combined_sigma_ms=%.6f accept=%d\n",
           A->row_off,cand_cpu_nnz,A->nnz_dsp,cand_valid&&cand.valid,cand.total_ms,
           cand.total_mad_ms,cand.cpu_part_ms,cand.wait_ms,gain_ms,combined_sigma,accept);
    if (!accept) {
        page_release_candidate(A,st); memset(st,0,sizeof(*st));
        page_opts ropt=*base_opt; ropt.force_cpu=0; ropt.force_hybrid=1;
        ropt.cpu_share=-1.0; ropt.row_off_override=old_row;
        double rtp[3]={0,0,0}; page_build(csr,st,A,coreNum,cluster_id,rtp,&ropt);
        if (tpre) { tpre[0]=rtp[0]; tpre[1]=rtp[1]; tpre[2]=rtp[2]; }
        printf("[V15_REBAL] decision=keep old_row=%d restored_row=%d\n",old_row,A->row_off);
    } else {
        printf("[V15_REBAL] decision=accept old_row=%d final_row=%d pilot_speedup=%.8f\n",
               old_row,A->row_off,base.total_ms/cand.total_ms);
    }
    if (elapsed_ms_out) *elapsed_ms_out=now_ms()-t0;
    return 0;
}

#ifndef PAGE_V17_AB_TRIALS
#define PAGE_V17_AB_TRIALS 6
#endif

static int page_v17_cpu_csr_ab(const CSRMatrix *orig_csr,
                                const CSRMatrix *private_csr,
                                const MAT_VAL_TYPE *cpu_x,
                                int m, int cluster_id)
{
    if (!orig_csr || !private_csr || !cpu_x || m <= 0) return -1;
    MAT_VAL_TYPE *yo = (MAT_VAL_TYPE *)hthread_malloc(
        cluster_id, (size_t)m * sizeof(MAT_VAL_TYPE), HT_MEM_RW);
    MAT_VAL_TYPE *yp = (MAT_VAL_TYPE *)hthread_malloc(
        cluster_id, (size_t)m * sizeof(MAT_VAL_TYPE), HT_MEM_RW);
    if (!yo || !yp) {
        if (yo) hthread_free(yo);
        if (yp) hthread_free(yp);
        return -1;
    }
    memset(yo, 0, (size_t)m * sizeof(MAT_VAL_TYPE));
    memset(yp, 0, (size_t)m * sizeof(MAT_VAL_TYPE));
    page_cpu_exec_full(orig_csr, cpu_x, yo);
    page_cpu_exec_full(private_csr, cpu_x, yp);

    double to[PAGE_V17_AB_TRIALS], tp[PAGE_V17_AB_TRIALS];
    for (int i = 0; i < PAGE_V17_AB_TRIALS; ++i) {
        if ((i & 1) == 0) {
            double a = now_ms(); page_cpu_exec_full(orig_csr, cpu_x, yo); to[i] = now_ms() - a;
            a = now_ms(); page_cpu_exec_full(private_csr, cpu_x, yp); tp[i] = now_ms() - a;
        } else {
            double a = now_ms(); page_cpu_exec_full(private_csr, cpu_x, yp); tp[i] = now_ms() - a;
            a = now_ms(); page_cpu_exec_full(orig_csr, cpu_x, yo); to[i] = now_ms() - a;
        }
    }
    const page_robust_stat ro = page_robust_stats(to, PAGE_V17_AB_TRIALS);
    const page_robust_stat rp = page_robust_stats(tp, PAGE_V17_AB_TRIALS);
    const int equal = memcmp(yo, yp, (size_t)m * sizeof(MAT_VAL_TYPE)) == 0;
    printf("[V17_CSR_AB] mode=cpu trials=%d original_median_ms=%.6f original_mad_ms=%.6f "
           "private_median_ms=%.6f private_mad_ms=%.6f speedup_orig_over_private=%.8f output_equal=%d\n",
           PAGE_V17_AB_TRIALS, ro.median, ro.mad, rp.median, rp.mad,
           rp.median > 0.0 ? ro.median / rp.median : 0.0, equal);
    hthread_free(yo);
    hthread_free(yp);
    return equal ? 0 : -1;
}

static int page_v17_hybrid_csr_ab(const CSRMatrix *orig_csr,
                                   const CSRMatrix *private_csr,
                                   const page_matrix *A,
                                   const MAT_VAL_TYPE *cpu_x,
                                   const MAT_VAL_TYPE *dsp_x,
                                   int coreNum, int cluster_id)
{
    page_v15_pilot_stat o1, p1, p2, o2;
    memset(&o1, 0, sizeof(o1)); memset(&p1, 0, sizeof(p1));
    memset(&p2, 0, sizeof(p2)); memset(&o2, 0, sizeof(o2));
    if (page_v15_pilot_tails(orig_csr, A, cpu_x, dsp_x, coreNum, cluster_id, &o1) != 0 ||
        page_v15_pilot_tails(private_csr, A, cpu_x, dsp_x, coreNum, cluster_id, &p1) != 0 ||
        page_v15_pilot_tails(private_csr, A, cpu_x, dsp_x, coreNum, cluster_id, &p2) != 0 ||
        page_v15_pilot_tails(orig_csr, A, cpu_x, dsp_x, coreNum, cluster_id, &o2) != 0 ||
        !o1.valid || !o2.valid || !p1.valid || !p2.valid) {
        printf("[V17_CSR_AB] mode=hybrid valid=0\n");
        return -1;
    }
    const double om = 0.5 * (o1.total_ms + o2.total_ms);
    const double pm = 0.5 * (p1.total_ms + p2.total_ms);
    const double omad = 0.5 * (o1.total_mad_ms + o2.total_mad_ms);
    const double pmad = 0.5 * (p1.total_mad_ms + p2.total_mad_ms);
    printf("[V17_CSR_AB] mode=hybrid valid=1 sequence=original_private_private_original "
           "original_a_ms=%.6f private_a_ms=%.6f private_b_ms=%.6f original_b_ms=%.6f "
           "original_mean_median_ms=%.6f original_mean_mad_ms=%.6f "
           "private_mean_median_ms=%.6f private_mean_mad_ms=%.6f speedup_orig_over_private=%.8f "
           "row_off=%d dsp_nnz=%lld\n",
           o1.total_ms, p1.total_ms, p2.total_ms, o2.total_ms,
           om, omad, pm, pmad, pm > 0.0 ? om / pm : 0.0,
           A ? A->row_off : 0, A ? A->nnz_dsp : 0LL);
    return 0;
}

static const char *page_policy_name(int policy, int has_window)
{
    if (has_window) return "window";
    if (policy == 0) return "tiled";
    return "auto";
}

static const char *basename_noext(char *path)
{
    int len = (int)strlen(path);
    char *p = path;
    for (int i = len; i >= 0; --i)
        if (path[i] == '/') { p = &path[i] + 1; break; }
    int l = (int)strlen(p);
    if (l > 4 && strcmp(p + l - 4, ".mtx") == 0) p[l - 4] = '\0';
    return p;
}

#ifndef PAGE_CONT_DIAG_WARMUP
#define PAGE_CONT_DIAG_WARMUP 2
#endif
#ifndef PAGE_CONT_DIAG_ITER
#define PAGE_CONT_DIAG_ITER 7
#endif
#ifndef PAGE_CONT_DIAG_NBYTES
#define PAGE_CONT_DIAG_NBYTES (CHUNK * (int)sizeof(MAT_VAL_TYPE))
#endif
#ifndef PAGE_CONT_DIAG_COVER_FACTOR
#define PAGE_CONT_DIAG_COVER_FACTOR 2.0
#endif
#ifndef PAGE_DUAL_X_DIAG_WARMUP
#define PAGE_DUAL_X_DIAG_WARMUP 3
#endif
#ifndef PAGE_DUAL_X_DIAG_ITER
#define PAGE_DUAL_X_DIAG_ITER 10
#endif
#ifndef PAGE_XPLACE_DIAG_WARMUP
#define PAGE_XPLACE_DIAG_WARMUP 3
#endif
#ifndef PAGE_XPLACE_DIAG_ITER
#define PAGE_XPLACE_DIAG_ITER 10
#endif

static volatile double page_cont_diag_sink = 0.0;

typedef enum {
    PAGE_CONT_DSP_COMPUTE = 0,
    PAGE_CONT_DSP_DDR_AM,
    PAGE_CONT_DSP_XLOAD,
    PAGE_CONT_DSP_GATHER,
    PAGE_CONT_DSP_PIPELINE,
    PAGE_CONT_DSP_COUNT
} page_cont_dsp_mode;

typedef struct {
    int coreNum;
    int nbytes;
    int nelem;
    MAT_VAL_TYPE *src;
    int *idx;
    MAT_VAL_TYPE *out;
} page_cont_stress_ctx;

typedef struct {
    int nrep;
    double target_ms;
    double solo_submit_ms;
    double solo_post_ms;
    double concurrent_submit_ms;
    double concurrent_post_ms;
    double cpu_part_ms;
    double wait_after_cpu_ms;
    double cpu_slowdown;
    double dsp_slowdown;
    int dsp_completion_after_cpu;
} page_cont_stress_result;

static const char *page_cont_dsp_mode_name(page_cont_dsp_mode mode)
{
    switch (mode) {
        case PAGE_CONT_DSP_COMPUTE:  return "compute_only";
        case PAGE_CONT_DSP_DDR_AM:   return "ddr_to_am";
        case PAGE_CONT_DSP_XLOAD:    return "ddr_to_gsm_xload";
        case PAGE_CONT_DSP_GATHER:   return "gsm_to_am_gather";
        case PAGE_CONT_DSP_PIPELINE: return "p2p_gather_compute";
        default: return "unknown";
    }
}

static int page_cont_stress_init(page_cont_stress_ctx *c, int coreNum,
                                  int cluster_id)
{
    if (!c || coreNum <= 0) return 0;
    memset(c, 0, sizeof(*c));
    c->coreNum = coreNum;
    c->nbytes = PAGE_CONT_DIAG_NBYTES;
    c->nelem = c->nbytes / (int)sizeof(MAT_VAL_TYPE);
    const size_t total0 = (size_t)coreNum * (size_t)c->nelem;
    if (total0 > (size_t)GSM_X_CAP) {

        c->nelem = GSM_X_CAP / coreNum;
        c->nelem = (c->nelem / SROW) * SROW;
        c->nbytes = c->nelem * (int)sizeof(MAT_VAL_TYPE);
    }
    if (c->nelem <= 0 || c->nbytes <= 0) return 0;

    const size_t elems = (size_t)coreNum * (size_t)c->nelem;
    c->src = (MAT_VAL_TYPE *)hthread_malloc(cluster_id,
                    elems * sizeof(MAT_VAL_TYPE), HT_MEM_RW);
    c->idx = (int *)hthread_malloc(cluster_id, elems * sizeof(int), HT_MEM_RW);
    c->out = (MAT_VAL_TYPE *)hthread_malloc(cluster_id,
                    (size_t)coreNum * SROW * sizeof(MAT_VAL_TYPE), HT_MEM_RW);
    if (!c->src || !c->idx || !c->out) {
        if (c->src) hthread_free(c->src);
        if (c->idx) hthread_free(c->idx);
        if (c->out) hthread_free(c->out);
        memset(c, 0, sizeof(*c));
        return 0;
    }

#pragma omp parallel for schedule(static)
    for (size_t k = 0; k < elems; k++) {
        const int tid = (int)(k / (size_t)c->nelem);
        const int i = (int)(k - (size_t)tid * (size_t)c->nelem);
        const int base = tid * c->nelem;
        c->src[k] = (MAT_VAL_TYPE)((base + i) & 1023) / 17.0;
        const int local = (i * 7919) % c->nelem;

        c->idx[k] = (base + local) * (int)sizeof(MAT_VAL_TYPE);
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < coreNum * SROW; i++) c->out[i] = 0.0;
    return 1;
}

static void page_cont_stress_free(page_cont_stress_ctx *c)
{
    if (!c) return;
    if (c->src) hthread_free(c->src);
    if (c->idx) hthread_free(c->idx);
    if (c->out) hthread_free(c->out);
    memset(c, 0, sizeof(*c));
}

static void page_cont_exec_stress(int gid, page_cont_stress_ctx *c,
                                   page_cont_dsp_mode mode, int nrep)
{
    unsigned long a[5];
    if (nrep < 1) nrep = 1;
    a[0] = (unsigned long)nrep;
    a[1] = (unsigned long)c->nbytes;
    switch (mode) {
        case PAGE_CONT_DSP_COMPUTE:
            a[2] = (unsigned long)c->out;
            hthread_group_exec(gid, "page_diag_compute_only", 2, 1, a);
            break;
        case PAGE_CONT_DSP_DDR_AM:
            a[2] = (unsigned long)c->src;
            a[3] = (unsigned long)c->out;
            hthread_group_exec(gid, "page_diag_ddr_am", 2, 2, a);
            break;
        case PAGE_CONT_DSP_XLOAD:
            a[2] = (unsigned long)c->src;
            hthread_group_exec(gid, "page_bench_xload", 2, 1, a);
            break;
        case PAGE_CONT_DSP_GATHER:
            a[2] = (unsigned long)c->idx;
            a[3] = (unsigned long)c->out;
            hthread_group_exec(gid, "page_diag_gsm_am_gather", 2, 2, a);
            break;
        case PAGE_CONT_DSP_PIPELINE:
            a[2] = (unsigned long)c->src;
            a[3] = (unsigned long)c->idx;
            a[4] = (unsigned long)c->out;
            hthread_group_exec(gid, "page_diag_pipeline", 2, 3, a);
            break;
        default:
            a[2] = (unsigned long)c->out;
            hthread_group_exec(gid, "page_diag_compute_only", 2, 1, a);
            break;
    }
}

static void page_cont_fill_gx(int gid, page_cont_stress_ctx *c)
{
    unsigned long a[3];
    a[0] = 1;
    a[1] = (unsigned long)c->nbytes;
    a[2] = (unsigned long)c->src;
    hthread_group_exec(gid, "page_bench_xload", 2, 1, a);
    hthread_group_wait(gid);
}

static double page_cont_stress_once(int gid, page_cont_stress_ctx *c,
                                     page_cont_dsp_mode mode, int nrep,
                                     double *submit_ms_out)
{
    if (mode == PAGE_CONT_DSP_GATHER || mode == PAGE_CONT_DSP_PIPELINE)
        page_cont_fill_gx(gid, c);
    const double t0 = now_ms();
    page_cont_exec_stress(gid, c, mode, nrep);
    const double t1 = now_ms();
    hthread_group_wait(gid);
    const double t2 = now_ms();
    if (submit_ms_out) *submit_ms_out = t1 - t0;
    return t2 - t1;
}

static int page_cont_tune_nrep(int gid, page_cont_stress_ctx *c,
                                page_cont_dsp_mode mode, double target_ms)
{
    if (!(target_ms > 0.0)) target_ms = 1.0;
    int rep = 4;
    double submit = 0.0;
    (void)page_cont_stress_once(gid, c, mode, rep, &submit);
    double post = page_cont_stress_once(gid, c, mode, rep, &submit);
    if (!(post > 0.001) || !isfinite(post)) post = 0.001;
    long long nr = (long long)ceil((double)rep * target_ms / post);
    if (nr < 1) nr = 1;
    if (nr > 1000000) nr = 1000000;
    rep = (int)nr;

    post = page_cont_stress_once(gid, c, mode, rep, &submit);
    if (post > 0.001 && post < target_ms) {
        nr = (long long)ceil((double)rep * target_ms / post);
        if (nr < 1) nr = 1;
        if (nr > 1000000) nr = 1000000;
        rep = (int)nr;
    }
    return rep;
}

static inline void page_cont_cpu_part(const CSRMatrix *csr, const page_matrix *A,
                                       const MAT_VAL_TYPE *x,
                                       MAT_VAL_TYPE *y, MAT_VAL_TYPE *y_res)
{
    page_cpu_spmv_rows(csr, x, y, 0, A->row_off);
    page_cpu_spmv_residual(A, x, y_res);
}

static double page_cont_matrix_stream_range(const int *rp, const int *ci,
                                             const MAT_VAL_TYPE *va,
                                             int value_mode, int r0, int r1)
{
    if (!rp || !ci || r1 <= r0) return 0.0;
    double total = 0.0;
#pragma omp parallel reduction(+:total)
    {
        int lo, hi;
        page_row_chunk(rp, r0, r1, omp_get_num_threads(), omp_get_thread_num(), &lo, &hi);
        const int b = rp[lo], e = rp[hi];
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int j = b;
        if (value_mode == PAGE_VALUE_GENERAL && va) {
            for (; j + 3 < e; j += 4) {
                s0 += (double)ci[j]     * 1.0e-9 + (double)va[j]     * 1.0e-6;
                s1 += (double)ci[j + 1] * 1.0e-9 + (double)va[j + 1] * 1.0e-6;
                s2 += (double)ci[j + 2] * 1.0e-9 + (double)va[j + 2] * 1.0e-6;
                s3 += (double)ci[j + 3] * 1.0e-9 + (double)va[j + 3] * 1.0e-6;
            }
            for (; j < e; j++)
                s0 += (double)ci[j] * 1.0e-9 + (double)va[j] * 1.0e-6;
        } else {
            for (; j + 3 < e; j += 4) {
                s0 += (double)ci[j]     * 1.0e-9;
                s1 += (double)ci[j + 1] * 1.0e-9;
                s2 += (double)ci[j + 2] * 1.0e-9;
                s3 += (double)ci[j + 3] * 1.0e-9;
            }
            for (; j < e; j++) s0 += (double)ci[j] * 1.0e-9;
        }
        total += (s0 + s1) + (s2 + s3);
    }
    return total;
}

static double page_cont_x_gather_range(const int *rp, const int *ci,
                                        const MAT_VAL_TYPE *x, int r0, int r1)
{
    if (!rp || !ci || !x || r1 <= r0) return 0.0;
    double total = 0.0;
#pragma omp parallel reduction(+:total)
    {
        int lo, hi;
        page_row_chunk(rp, r0, r1, omp_get_num_threads(), omp_get_thread_num(), &lo, &hi);
        const int b = rp[lo], e = rp[hi];
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int j = b;
        for (; j + 3 < e; j += 4) {
            s0 += x[ci[j]];
            s1 += x[ci[j + 1]];
            s2 += x[ci[j + 2]];
            s3 += x[ci[j + 3]];
        }
        for (; j < e; j++) s0 += x[ci[j]];
        total += (s0 + s1) + (s2 + s3);
    }
    return total;
}

static double page_cont_cpu_component(const CSRMatrix *csr, const page_matrix *A,
                                       const MAT_VAL_TYPE *x, int kind)
{
    double v = 0.0;
    if (kind == 0) {
        v += page_cont_matrix_stream_range(csr->rowPointers, csr->colIndices,
                                             csr->values, csr->value_mode,
                                             0, A->row_off);
        if (A->has_window && A->res_nnz > 0)
            v += page_cont_matrix_stream_range(A->res_rp, A->res_ci, A->res_val,
                                                 A->value_mode, 0, A->m_dsp);
    } else {
        v += page_cont_x_gather_range(csr->rowPointers, csr->colIndices, x,
                                        0, A->row_off);
        if (A->has_window && A->res_nnz > 0)
            v += page_cont_x_gather_range(A->res_rp, A->res_ci, x,
                                            0, A->m_dsp);
    }
    page_cont_diag_sink += v * 1.0e-30;
    return v;
}

static int page_cont_tune_cpu_component(const CSRMatrix *csr, const page_matrix *A,
                                         const MAT_VAL_TYPE *x, int kind,
                                         double target_ms)
{
    const double t0 = now_ms();
    (void)page_cont_cpu_component(csr, A, x, kind);
    double one = now_ms() - t0;
    if (!(one > 0.001) || !isfinite(one)) one = 0.001;
    long long reps = (long long)ceil(target_ms / one);
    if (reps < 1) reps = 1;
    if (reps > 1024) reps = 1024;
    return (int)reps;
}

static void page_cont_run_cpu_component_diag(const char *name, int kind, int reps,
                                              const CSRMatrix *csr, const page_matrix *A,
                                              const MAT_VAL_TYPE *x, int gid,
                                              int scalArgs, int vecArgs,
                                              unsigned long *spmv_args)
{
    double solo[PAGE_CONT_DIAG_ITER], conc[PAGE_CONT_DIAG_ITER];
    double dsp_post[PAGE_CONT_DIAG_ITER], wait_after[PAGE_CONT_DIAG_ITER];
    for (int w = 0; w < PAGE_CONT_DIAG_WARMUP; w++)
        for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x, kind);
    for (int i = 0; i < PAGE_CONT_DIAG_ITER; i++) {
        const double a = now_ms();
        for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x, kind);
        solo[i] = now_ms() - a;
    }

    for (int w = 0; w < PAGE_CONT_DIAG_WARMUP; w++) {
        hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
        for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x, kind);
        hthread_group_wait(gid);
    }
    for (int i = 0; i < PAGE_CONT_DIAG_ITER; i++) {
        hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
        const double a = now_ms();
        for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x, kind);
        const double b = now_ms();
        hthread_group_wait(gid);
        const double c = now_ms();
        conc[i] = b - a;
        dsp_post[i] = c - a;
        wait_after[i] = c - b;
    }
    const double s = trimmed_mean_1each(solo, PAGE_CONT_DIAG_ITER);
    const double cc = trimmed_mean_1each(conc, PAGE_CONT_DIAG_ITER);
    const double d = trimmed_mean_1each(dsp_post, PAGE_CONT_DIAG_ITER);
    const double w = trimmed_mean_1each(wait_after, PAGE_CONT_DIAG_ITER);
    printf("[CONT_CPU_PROBE] kind=%s reps=%d solo_ms=%.6f with_real_dsp_ms=%.6f "
           "slowdown=%.8f real_dsp_post_endpoint_ms=%.6f wait_after_cpu_ms=%.6f\n",
           name, reps, s, cc, (s > 0.0 ? cc / s : 0.0), d, w);
}

static page_cont_stress_result page_cont_run_stress_diag(
        page_cont_dsp_mode mode, int gid, page_cont_stress_ctx *ctx,
        const CSRMatrix *csr, const page_matrix *A, const MAT_VAL_TYPE *x,
        MAT_VAL_TYPE *y, MAT_VAL_TYPE *y_res, double cpu_part_solo_ms)
{
    page_cont_stress_result r;
    memset(&r, 0, sizeof(r));
    r.target_ms = (cpu_part_solo_ms > 0.0)
                ? cpu_part_solo_ms * PAGE_CONT_DIAG_COVER_FACTOR : 1.0;
    if (r.target_ms < 1.0) r.target_ms = 1.0;
    r.nrep = page_cont_tune_nrep(gid, ctx, mode, r.target_ms);

    double solo_sub[PAGE_CONT_DIAG_ITER], solo_post[PAGE_CONT_DIAG_ITER];
    double con_sub[PAGE_CONT_DIAG_ITER], con_post[PAGE_CONT_DIAG_ITER];
    double cpu[PAGE_CONT_DIAG_ITER], wait[PAGE_CONT_DIAG_ITER];

    for (int w = 0; w < PAGE_CONT_DIAG_WARMUP; w++) {
        if (mode == PAGE_CONT_DSP_GATHER || mode == PAGE_CONT_DSP_PIPELINE)
            page_cont_fill_gx(gid, ctx);
        page_cont_exec_stress(gid, ctx, mode, r.nrep);
        hthread_group_wait(gid);
    }
    for (int i = 0; i < PAGE_CONT_DIAG_ITER; i++) {
        if (mode == PAGE_CONT_DSP_GATHER || mode == PAGE_CONT_DSP_PIPELINE)
            page_cont_fill_gx(gid, ctx);
        const double a = now_ms();
        page_cont_exec_stress(gid, ctx, mode, r.nrep);
        const double b = now_ms();
        hthread_group_wait(gid);
        const double c = now_ms();
        solo_sub[i] = b - a;
        solo_post[i] = c - b;
    }

    for (int w = 0; w < PAGE_CONT_DIAG_WARMUP; w++) {
        if (mode == PAGE_CONT_DSP_GATHER || mode == PAGE_CONT_DSP_PIPELINE)
            page_cont_fill_gx(gid, ctx);
        page_cont_exec_stress(gid, ctx, mode, r.nrep);
        page_cont_cpu_part(csr, A, x, y, y_res);
        hthread_group_wait(gid);
    }
    for (int i = 0; i < PAGE_CONT_DIAG_ITER; i++) {
        if (mode == PAGE_CONT_DSP_GATHER || mode == PAGE_CONT_DSP_PIPELINE)
            page_cont_fill_gx(gid, ctx);
        const double a = now_ms();
        page_cont_exec_stress(gid, ctx, mode, r.nrep);
        const double b = now_ms();
        page_cont_cpu_part(csr, A, x, y, y_res);
        const double c = now_ms();
        hthread_group_wait(gid);
        const double d = now_ms();
        con_sub[i] = b - a;
        cpu[i] = c - b;
        wait[i] = d - c;
        con_post[i] = d - b;
    }

    r.solo_submit_ms = trimmed_mean_1each(solo_sub, PAGE_CONT_DIAG_ITER);
    r.solo_post_ms = trimmed_mean_1each(solo_post, PAGE_CONT_DIAG_ITER);
    r.concurrent_submit_ms = trimmed_mean_1each(con_sub, PAGE_CONT_DIAG_ITER);
    r.concurrent_post_ms = trimmed_mean_1each(con_post, PAGE_CONT_DIAG_ITER);
    r.cpu_part_ms = trimmed_mean_1each(cpu, PAGE_CONT_DIAG_ITER);
    r.wait_after_cpu_ms = trimmed_mean_1each(wait, PAGE_CONT_DIAG_ITER);
    r.cpu_slowdown = (cpu_part_solo_ms > 0.0)
                   ? r.cpu_part_ms / cpu_part_solo_ms : 0.0;
    r.dsp_completion_after_cpu = (r.wait_after_cpu_ms > 0.0) ? 1 : 0;

    r.dsp_slowdown = (r.solo_post_ms > 0.0)
                   ? r.concurrent_post_ms / r.solo_post_ms : 0.0;
    return r;
}

static void page_cont_spin_delay(double ms)
{
    if (!(ms > 0.0)) return;
    const double t0 = now_ms();
    while (now_ms() - t0 < ms) { }
}

static int page_cont_packed_rounds_for_tile(const page_matrix *A, int t,
                                             int coreNum)
{
    if (!A || t < 0 || t >= A->ntile || !A->tile_packed ||
        !A->tile_packed[t] || !A->gsmxwidth || !A->tile_xmap_lo ||
        !A->tile_xmap) return 0;
    const int ml = A->tile_xmap_lo[t];
    const int *g = &A->gsmxwidth[t * (coreNum + 1)];
    int crit = 0;
    for (int tid = 0; tid < coreNum; tid++) {
        const int b0 = g[tid], b1 = g[tid + 1];
        int calls = 0;
        if (b1 > b0) {
            calls = 1;
            for (int b = b0 + 1; b < b1; b++)
                if (A->tile_xmap[ml + b] !=
                    A->tile_xmap[ml + b - 1] + PAGE_XBLOCK_BYTES)
                    calls++;
        }
        if (calls > crit) crit = calls;
    }
    return crit;
}

static double page_cont_tile_xload_model_ms(const page_matrix *A, int t,
                                              int coreNum, double bw_xload,
                                              double xload_fixed_ms)
{
    if (!A || t < 0 || t >= A->ntile || !(bw_xload > 0.0)) return 0.0;
    long long elems = 0;
    int rounds = 0;
    if (A->tile_packed && A->tile_packed[t] && A->tile_xmap_lo) {
        elems = (long long)(A->tile_xmap_lo[t + 1] - A->tile_xmap_lo[t])
              * PAGE_XBLOCK_ELEMS;
        rounds = page_cont_packed_rounds_for_tile(A, t, coreNum);
    } else if (A->tile_xlo && A->tile_xhi) {
        elems = (long long)A->tile_xhi[t] - (long long)A->tile_xlo[t];
    }
    if (elems < 0) elems = 0;
    return ((double)elems * sizeof(MAT_VAL_TYPE) / bw_xload) * 1e3
         + (double)rounds * xload_fixed_ms;
}

static void page_cont_numa_object(const char *name, const void *ptr, size_t bytes)
{
    if (!ptr || bytes == 0) {
        printf("[CONT_NUMA] object=%s available=0 bytes=%zu\n",
               name ? name : "unknown", bytes);
        return;
    }
    const unsigned long addr = (unsigned long)ptr;
    unsigned long vma_lo = 0, vma_hi = 0;
    char line[4096];
    FILE *fp = fopen("/proc/self/maps", "r");
    if (fp) {
        while (fgets(line, sizeof(line), fp)) {
            unsigned long lo = 0, hi = 0;
            if (sscanf(line, "%lx-%lx", &lo, &hi) == 2 &&
                addr >= lo && addr < hi) {
                vma_lo = lo; vma_hi = hi; break;
            }
        }
        fclose(fp);
    }
    if (!vma_lo) {
        printf("[CONT_NUMA] object=%s available=0 bytes=%zu ptr=%p\n",
               name ? name : "unknown", bytes, ptr);
        return;
    }

    long total_pages = 0, best_pages = 0;
    int best_node = -1, found = 0;
    fp = fopen("/proc/self/numa_maps", "r");
    if (fp) {
        while (fgets(line, sizeof(line), fp)) {
            unsigned long lo = 0;
            if (sscanf(line, "%lx", &lo) != 1 || lo != vma_lo) continue;
            found = 1;
            char copy[4096];
            strncpy(copy, line, sizeof(copy) - 1);
            copy[sizeof(copy) - 1] = '\0';
            char *tok = strtok(copy, " \t\n");
            while (tok) {
                int node = -1;
                long pages = 0;
                if (sscanf(tok, "N%d=%ld", &node, &pages) == 2 && pages >= 0) {
                    total_pages += pages;
                    if (pages > best_pages) {
                        best_pages = pages;
                        best_node = node;
                    }
                }
                tok = strtok(NULL, " \t\n");
            }
            break;
        }
        fclose(fp);
    }
    const double frac = (total_pages > 0)
                      ? (double)best_pages / (double)total_pages : 0.0;
    printf("[CONT_NUMA] object=%s available=%d bytes=%zu ptr=%p "
           "vma_start=0x%lx vma_end=0x%lx mapped_pages=%ld "
           "dominant_node=%d dominant_pages=%ld dominant_fraction=%.8f\n",
           name ? name : "unknown", found, bytes, ptr,
           vma_lo, vma_hi, total_pages, best_node, best_pages, frac);
}

static void page_cont_numa_object_raw(const char *name, const void *ptr)
{
    if (!ptr) {
        printf("[NUMA_RAW] object=%s available=0\n", name ? name : "unknown");
        return;
    }
    const unsigned long addr = (unsigned long)ptr;
    unsigned long vma_lo = 0;
    char line[8192];
    FILE *fp = fopen("/proc/self/maps", "r");
    if (fp) {
        while (fgets(line, sizeof(line), fp)) {
            unsigned long lo = 0, hi = 0;
            if (sscanf(line, "%lx-%lx", &lo, &hi) == 2 &&
                addr >= lo && addr < hi) {
                vma_lo = lo;
                break;
            }
        }
        fclose(fp);
    }
    if (!vma_lo) {
        printf("[NUMA_RAW] object=%s available=0 ptr=%p\n",
               name ? name : "unknown", ptr);
        return;
    }
    fp = fopen("/proc/self/numa_maps", "r");
    if (fp) {
        while (fgets(line, sizeof(line), fp)) {
            unsigned long lo = 0;
            if (sscanf(line, "%lx", &lo) != 1 || lo != vma_lo) continue;
            line[strcspn(line, "\r\n")] = '\0';
            printf("[NUMA_RAW] object=%s available=1 line=%s\n",
                   name ? name : "unknown", line);
            fclose(fp);
            return;
        }
        fclose(fp);
    }
    printf("[NUMA_RAW] object=%s available=0 vma_start=0x%lx\n",
           name ? name : "unknown", vma_lo);
}

static void page_cont_smaps_object(const char *name, const void *ptr)
{
    if (!ptr) {
        printf("[SMAPS] object=%s available=0\n", name ? name : "unknown");
        return;
    }
    const unsigned long addr = (unsigned long)ptr;
    FILE *fp = fopen("/proc/self/smaps", "r");
    if (!fp) {
        printf("[SMAPS] object=%s available=0 ptr=%p errno=%d\n",
               name ? name : "unknown", ptr, errno);
        return;
    }
    char line[8192];
    int in_vma = 0, found = 0;
    unsigned long vma_lo = 0, vma_hi = 0;
    long kernel_kb = -1, mmu_kb = -1, anon_huge_kb = -1;
    long shmem_pmd_kb = -1, file_pmd_kb = -1;
    int thpeligible = -1;
    char vmflags[2048] = {0};
    while (fgets(line, sizeof(line), fp)) {
        unsigned long lo = 0, hi = 0;
        if (sscanf(line, "%lx-%lx", &lo, &hi) == 2) {
            if (in_vma) break;
            if (addr >= lo && addr < hi) {
                in_vma = 1; found = 1; vma_lo = lo; vma_hi = hi;
            }
            continue;
        }
        if (!in_vma) continue;
        (void)sscanf(line, "KernelPageSize: %ld kB", &kernel_kb);
        (void)sscanf(line, "MMUPageSize: %ld kB", &mmu_kb);
        (void)sscanf(line, "AnonHugePages: %ld kB", &anon_huge_kb);
        (void)sscanf(line, "ShmemPmdMapped: %ld kB", &shmem_pmd_kb);
        (void)sscanf(line, "FilePmdMapped: %ld kB", &file_pmd_kb);
        (void)sscanf(line, "THPeligible: %d", &thpeligible);
        if (strncmp(line, "VmFlags:", 8) == 0) {
            const char *q = line + 8;
            while (*q == ' ' || *q == '\t') q++;
            size_t n = strcspn(q, "\r\n");
            if (n >= sizeof(vmflags)) n = sizeof(vmflags) - 1;
            memcpy(vmflags, q, n); vmflags[n] = '\0';
        }
    }
    fclose(fp);
    printf("[SMAPS] object=%s available=%d ptr=%p vma_start=0x%lx vma_end=0x%lx "
           "kernel_page_kb=%ld mmu_page_kb=%ld anon_huge_kb=%ld "
           "shmem_pmd_kb=%ld file_pmd_kb=%ld thpeligible=%d vmflags=%s\n",
           name ? name : "unknown", found, ptr, vma_lo, vma_hi,
           kernel_kb, mmu_kb, anon_huge_kb, shmem_pmd_kb, file_pmd_kb,
           thpeligible, vmflags[0] ? vmflags : "NA");
}

typedef struct {
    double submit_ms;
    double cpu_ms;
    double wait_ms;
    double finalize_ms;
    double total_ms;
} page_dual_x_hybrid_timing;

static page_dual_x_hybrid_timing page_dual_x_hybrid_once(
        int gid, int scalArgs, int vecArgs, unsigned long *spmv_args,
        const CSRMatrix *csr, const page_matrix *A, const MAT_VAL_TYPE *cpu_x,
        MAT_VAL_TYPE *y, MAT_VAL_TYPE *y_res, MAT_VAL_TYPE *yd)
{
    page_dual_x_hybrid_timing r;
    memset(&r, 0, sizeof(r));
    const double a = now_ms();
    hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
    const double b = now_ms();
    page_cont_cpu_part(csr, A, cpu_x, y, y_res);
    const double c = now_ms();
    hthread_group_wait(gid);
    const double d = now_ms();
    page_finalize_y(A, yd, y_res, y);
    const double e = now_ms();
    r.submit_ms = b - a;
    r.cpu_ms = c - b;
    r.wait_ms = d - c;
    r.finalize_ms = e - d;
    r.total_ms = e - a;
    return r;
}

static void page_dual_x_gather_pair(const CSRMatrix *csr, const page_matrix *A,
                                     const MAT_VAL_TYPE *x_shared,
                                     const MAT_VAL_TYPE *x_private,
                                     int gid, int scalArgs, int vecArgs,
                                     unsigned long *spmv_args,
                                     double target_ms)
{
    int reps_s = page_cont_tune_cpu_component(csr, A, x_shared, 1, target_ms);
    int reps_p = page_cont_tune_cpu_component(csr, A, x_private, 1, target_ms);
    int reps = reps_s > reps_p ? reps_s : reps_p;
    if (reps < 1) reps = 1;

    double ss[PAGE_DUAL_X_DIAG_ITER], ps[PAGE_DUAL_X_DIAG_ITER];
    double sc[PAGE_DUAL_X_DIAG_ITER], pc[PAGE_DUAL_X_DIAG_ITER];

    for (int w = 0; w < PAGE_DUAL_X_DIAG_WARMUP; w++) {
        for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x_shared, 1);
        for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x_private, 1);
    }
    for (int i = 0; i < PAGE_DUAL_X_DIAG_ITER; i++) {
        if ((i & 1) == 0) {
            double a = now_ms();
            for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x_shared, 1);
            ss[i] = now_ms() - a;
            a = now_ms();
            for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x_private, 1);
            ps[i] = now_ms() - a;
        } else {
            double a = now_ms();
            for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x_private, 1);
            ps[i] = now_ms() - a;
            a = now_ms();
            for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x_shared, 1);
            ss[i] = now_ms() - a;
        }
    }

    for (int w = 0; w < PAGE_DUAL_X_DIAG_WARMUP; w++) {
        hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
        for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x_shared, 1);
        hthread_group_wait(gid);
        hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
        for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x_private, 1);
        hthread_group_wait(gid);
    }
    for (int i = 0; i < PAGE_DUAL_X_DIAG_ITER; i++) {
        if ((i & 1) == 0) {
            hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
            double a = now_ms();
            for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x_shared, 1);
            sc[i] = now_ms() - a;
            hthread_group_wait(gid);

            hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
            a = now_ms();
            for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x_private, 1);
            pc[i] = now_ms() - a;
            hthread_group_wait(gid);
        } else {
            hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
            double a = now_ms();
            for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x_private, 1);
            pc[i] = now_ms() - a;
            hthread_group_wait(gid);

            hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
            a = now_ms();
            for (int r = 0; r < reps; r++) (void)page_cont_cpu_component(csr, A, x_shared, 1);
            sc[i] = now_ms() - a;
            hthread_group_wait(gid);
        }
    }

    const double ss_m = trimmed_mean_1each(ss, PAGE_DUAL_X_DIAG_ITER);
    const double ps_m = trimmed_mean_1each(ps, PAGE_DUAL_X_DIAG_ITER);
    const double sc_m = trimmed_mean_1each(sc, PAGE_DUAL_X_DIAG_ITER);
    const double pc_m = trimmed_mean_1each(pc, PAGE_DUAL_X_DIAG_ITER);
    printf("[DUAL_X_GATHER] reps=%d shared_solo_ms=%.6f private_solo_ms=%.6f "
           "shared_with_dsp_ms=%.6f private_with_dsp_ms=%.6f "
           "shared_slowdown=%.8f private_slowdown=%.8f "
           "private_over_shared_concurrent=%.8f private_over_shared_solo=%.8f\n",
           reps, ss_m, ps_m, sc_m, pc_m,
           ss_m > 0.0 ? sc_m / ss_m : 0.0,
           ps_m > 0.0 ? pc_m / ps_m : 0.0,
           sc_m > 0.0 ? pc_m / sc_m : 0.0,
           ss_m > 0.0 ? ps_m / ss_m : 0.0);
}

typedef struct {
    MAT_VAL_TYPE *ptr;
    size_t bytes;
    int mbind_ok;
    int madvise_ok;
    int nohuge_requested;
    int node_count;
    unsigned long maxnode;
    int err;
    int madvise_err;
    char allowed_list[512];
} page_xplace_mapping;

static int page_xplace_parse_allowed_nodes(unsigned long **mask_out,
                                             unsigned long *maxnode_out,
                                             int *count_out,
                                             char *list_out,
                                             size_t list_cap)
{
    if (mask_out) *mask_out = NULL;
    if (maxnode_out) *maxnode_out = 0;
    if (count_out) *count_out = 0;
    if (list_out && list_cap) list_out[0] = '\0';

    FILE *fp = fopen("/proc/self/status", "r");
    if (!fp) return 0;
    char line[4096];
    char list[2048] = {0};
    while (fgets(line, sizeof(line), fp)) {
        const char *tag = "Mems_allowed_list:";
        const size_t ntag = strlen(tag);
        if (strncmp(line, tag, ntag) != 0) continue;
        const char *q = line + ntag;
        while (*q == ' ' || *q == '\t') q++;
        size_t n = strcspn(q, "\r\n");
        if (n >= sizeof(list)) n = sizeof(list) - 1;
        memcpy(list, q, n);
        list[n] = '\0';
        break;
    }
    fclose(fp);
    if (!list[0]) return 0;

    long highest = -1;
    int count = 0;
    const char *q = list;
    while (*q) {
        char *end = NULL;
        long lo = strtol(q, &end, 10);
        if (end == q || lo < 0) return 0;
        long hi = lo;
        q = end;
        if (*q == '-') {
            q++;
            hi = strtol(q, &end, 10);
            if (end == q || hi < lo) return 0;
            q = end;
        }
        if (hi > highest) highest = hi;
        if (hi - lo + 1 > 1000000L) return 0;
        count += (int)(hi - lo + 1);
        while (*q == ' ' || *q == '\t') q++;
        if (*q == ',') q++;
        else if (*q != '\0') return 0;
    }
    if (highest < 0 || count <= 0) return 0;

    const unsigned long required_bits = (unsigned long)highest + 1UL;
    const unsigned long bits = 8UL * sizeof(unsigned long);
    const size_t words = (size_t)((required_bits + bits - 1UL) / bits);
    /* NUMA syscall masks must be rounded to machine-word bit counts. */
    const unsigned long maxnode = (unsigned long)words * bits;
    unsigned long *mask = (unsigned long *)calloc(words, sizeof(unsigned long));
    if (!mask) return 0;

    q = list;
    while (*q) {
        char *end = NULL;
        long lo = strtol(q, &end, 10);
        long hi = lo;
        q = end;
        if (*q == '-') {
            q++;
            hi = strtol(q, &end, 10);
            q = end;
        }
        for (long node = lo; node <= hi; node++) {
            const unsigned long u = (unsigned long)node;
            mask[u / bits] |= 1UL << (u % bits);
        }
        while (*q == ' ' || *q == '\t') q++;
        if (*q == ',') q++;
    }

    if (mask_out) *mask_out = mask; else free(mask);
    if (maxnode_out) *maxnode_out = maxnode;
    if (count_out) *count_out = count;
    if (list_out && list_cap) {
        strncpy(list_out, list, list_cap - 1);
        list_out[list_cap - 1] = '\0';
    }
    return 1;
}

static page_xplace_mapping page_xplace_alloc_interleaved_mode(
        const MAT_VAL_TYPE *src, int n_pad, int nohuge)
{
    page_xplace_mapping r;
    memset(&r, 0, sizeof(r));
    r.nohuge_requested = nohuge ? 1 : 0;
    if (!src || n_pad <= 0) { r.err = EINVAL; return r; }
    r.bytes = (size_t)n_pad * sizeof(MAT_VAL_TYPE);

    unsigned long *mask = NULL;
    if (!page_xplace_parse_allowed_nodes(&mask, &r.maxnode, &r.node_count,
                                          r.allowed_list, sizeof(r.allowed_list))) {
        r.err = EINVAL;
        return r;
    }

    void *mem = mmap(NULL, r.bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        r.err = errno;
        free(mask);
        return r;
    }

    /* Apply page-size policy before first touch and NUMA population. */
    if (nohuge) {
        errno = 0;
        if (madvise(mem, r.bytes, MADV_NOHUGEPAGE) != 0) {
            r.madvise_err = errno ? errno : EINVAL;
            r.err = r.madvise_err;
            munmap(mem, r.bytes);
            free(mask);
            return r;
        }
        r.madvise_ok = 1;
    } else {
        r.madvise_ok = 1;
    }

#ifdef SYS_mbind
    errno = 0;
    const long rc = syscall(SYS_mbind, mem, r.bytes, MPOL_INTERLEAVE,
                            mask, r.maxnode, 0UL);
    if (rc != 0) {
        r.err = errno ? errno : EINVAL;
        munmap(mem, r.bytes);
        free(mask);
        return r;
    }
    r.mbind_ok = 1;
#else
    r.err = ENOSYS;
    munmap(mem, r.bytes);
    free(mask);
    return r;
#endif
    free(mask);

    r.ptr = (MAT_VAL_TYPE *)mem;
    page_parallel_first_touch_x(r.ptr, src, n_pad);
    return r;
}

static page_xplace_mapping page_xplace_alloc_interleaved(
        const MAT_VAL_TYPE *src, int n_pad)
{
    return page_xplace_alloc_interleaved_mode(src, n_pad, 0);
}

static page_xplace_mapping page_xplace_alloc_interleaved_nohuge(
        const MAT_VAL_TYPE *src, int n_pad)
{
    return page_xplace_alloc_interleaved_mode(src, n_pad, 1);
}

static void page_xplace_free_interleaved(page_xplace_mapping *m)
{
    if (!m) return;
    if (m->ptr && m->bytes) munmap(m->ptr, m->bytes);
    m->ptr = NULL;
    m->bytes = 0;
}

typedef struct {
    void *ptr;
    size_t bytes;
    int mbind_ok;
    int madvise_ok;
    int err;
    int madvise_err;
} page_v17_raw_mapping;

typedef struct {
    CSRMatrix csr;
    page_v17_raw_mapping rp;
    page_v17_raw_mapping ci;
    page_v17_raw_mapping va;
    int valid;
    int node_count;
    unsigned long maxnode;
    char allowed_list[512];
    double setup_ms;
} page_v17_csr_mapping;

static page_v17_raw_mapping page_v17_alloc_raw_interleaved_nohuge(
        size_t bytes, const unsigned long *mask, unsigned long maxnode)
{
    page_v17_raw_mapping r;
    memset(&r, 0, sizeof(r));
    if (bytes == 0 || !mask || maxnode == 0) {
        r.err = EINVAL;
        return r;
    }
    r.bytes = bytes;
    void *mem = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        r.err = errno ? errno : ENOMEM;
        r.ptr = NULL;
        return r;
    }
    errno = 0;
    if (madvise(mem, bytes, MADV_NOHUGEPAGE) != 0) {
        r.madvise_err = errno ? errno : EINVAL;
        r.err = r.madvise_err;
        munmap(mem, bytes);
        return r;
    }
    r.madvise_ok = 1;
#ifdef SYS_mbind
    errno = 0;
    if (syscall(SYS_mbind, mem, bytes, MPOL_INTERLEAVE,
                mask, maxnode, 0UL) != 0) {
        r.err = errno ? errno : EINVAL;
        munmap(mem, bytes);
        r.ptr = NULL;
        return r;
    }
    r.mbind_ok = 1;
#else
    r.err = ENOSYS;
    munmap(mem, bytes);
    r.ptr = NULL;
    return r;
#endif
    r.ptr = mem;
    return r;
}

static void page_v17_free_raw(page_v17_raw_mapping *r)
{
    if (!r) return;
    if (r->ptr && r->bytes) munmap(r->ptr, r->bytes);
    memset(r, 0, sizeof(*r));
}

static void page_v17_free_csr_mapping(page_v17_csr_mapping *m)
{
    if (!m) return;
    page_v17_free_raw(&m->rp);
    page_v17_free_raw(&m->ci);
    page_v17_free_raw(&m->va);
    memset(m, 0, sizeof(*m));
}

static page_v17_csr_mapping page_v17_alloc_csr_interleaved_nohuge(
        const CSRMatrix *src)
{
    page_v17_csr_mapping out;
    memset(&out, 0, sizeof(out));
    if (!src || !src->rowPointers || !src->colIndices || !src->values ||
        src->numRows < 0 || src->numNonzeros < 0) return out;

    const double t0 = now_ms();
    unsigned long *mask = NULL;
    if (!page_xplace_parse_allowed_nodes(&mask, &out.maxnode, &out.node_count,
                                          out.allowed_list, sizeof(out.allowed_list))) {
        return out;
    }

    const size_t rp_bytes = (size_t)(src->numRows + 1) * sizeof(int);
    const size_t nz_elems = (size_t)(src->numNonzeros > 0 ? src->numNonzeros : 1);
    const size_t ci_bytes = nz_elems * sizeof(int);
    const size_t va_bytes = nz_elems * sizeof(MAT_VAL_TYPE);
    out.rp = page_v17_alloc_raw_interleaved_nohuge(rp_bytes, mask, out.maxnode);
    out.ci = page_v17_alloc_raw_interleaved_nohuge(ci_bytes, mask, out.maxnode);
    out.va = page_v17_alloc_raw_interleaved_nohuge(va_bytes, mask, out.maxnode);
    free(mask);

    if (!out.rp.ptr || !out.ci.ptr || !out.va.ptr ||
        !out.rp.mbind_ok || !out.ci.mbind_ok || !out.va.mbind_ok ||
        !out.rp.madvise_ok || !out.ci.madvise_ok || !out.va.madvise_ok) {
        page_v17_free_csr_mapping(&out);
        return out;
    }

    int *rp = (int *)out.rp.ptr;
    int *ci = (int *)out.ci.ptr;
    MAT_VAL_TYPE *va = (MAT_VAL_TYPE *)out.va.ptr;
#pragma omp parallel for schedule(static)
    for (int i = 0; i <= src->numRows; ++i) rp[i] = src->rowPointers[i];
#pragma omp parallel for schedule(static)
    for (int j = 0; j < src->numNonzeros; ++j) {
        ci[j] = src->colIndices[j];
        va[j] = src->values[j];
    }

    out.csr = *src;
    out.csr.rowPointers = rp;
    out.csr.colIndices = ci;
    out.csr.values = va;
    const int equal_rp = memcmp(rp, src->rowPointers, rp_bytes) == 0;
    const int equal_ci = (src->numNonzeros == 0) ||
        memcmp(ci, src->colIndices, (size_t)src->numNonzeros * sizeof(int)) == 0;
    const int equal_va = (src->numNonzeros == 0) ||
        memcmp(va, src->values, (size_t)src->numNonzeros * sizeof(MAT_VAL_TYPE)) == 0;
    const int distinct = (rp != src->rowPointers && ci != src->colIndices && va != src->values);
    out.valid = equal_rp && equal_ci && equal_va && distinct;
    out.setup_ms = now_ms() - t0;
    if (!out.valid) page_v17_free_csr_mapping(&out);
    return out;
}

static double page_v18_materialize_shadow_plan(const CSRMatrix *csr,
                                                 page_stat *st,
                                                 page_matrix *A,
                                                 int cluster_id,
                                                 const page_opts *opt,
                                                 double *pass_b_s_out)
{
    if (pass_b_s_out) *pass_b_s_out = 0.0;
    if (!csr || !st || !A || A->mode == PAGE_MODE_CPU || A->nnz_dsp <= 0)
        return 0.0;
    const double t0 = now_ms();
    if (A->has_window) page_build_residual(csr, A);
    A->bytes_per_nnz = (A->nnz_dsp > 0) ? page_bytes(A) / (double)A->nnz_dsp : 0.0;
    if (opt) A->est_dsp_ms = page_est_dsp_ms(A, opt);
    const double p0 = omp_get_wtime();
    page_pass_b(csr, st, A, cluster_id);
    if (pass_b_s_out) *pass_b_s_out = omp_get_wtime() - p0;
    return now_ms() - t0;
}

static void page_xplace_gather_triple(const CSRMatrix *csr,
                                       const page_matrix *A,
                                       const MAT_VAL_TYPE *x_shared,
                                       const MAT_VAL_TYPE *x_firsttouch,
                                       const MAT_VAL_TYPE *x_interleave,
                                       int gid, int scalArgs, int vecArgs,
                                       unsigned long *spmv_args,
                                       double target_ms)
{
    const MAT_VAL_TYPE *xs[3] = {x_shared, x_firsttouch, x_interleave};
    int reps = 1;
    for (int k = 0; k < 3; k++) {
        int r = page_cont_tune_cpu_component(csr, A, xs[k], 1, target_ms);
        if (r > reps) reps = r;
    }
    if (reps < 1) reps = 1;

    double solo[3][PAGE_XPLACE_DIAG_ITER];
    double conc[3][PAGE_XPLACE_DIAG_ITER];
    memset(solo, 0, sizeof(solo));
    memset(conc, 0, sizeof(conc));

    for (int w = 0; w < PAGE_XPLACE_DIAG_WARMUP; w++) {
        for (int k = 0; k < 3; k++)
            for (int r = 0; r < reps; r++)
                (void)page_cont_cpu_component(csr, A, xs[k], 1);
    }
    for (int i = 0; i < PAGE_XPLACE_DIAG_ITER; i++) {
        for (int pos = 0; pos < 3; pos++) {
            const int k = (i + pos) % 3;
            const double a = now_ms();
            for (int r = 0; r < reps; r++)
                (void)page_cont_cpu_component(csr, A, xs[k], 1);
            solo[k][i] = now_ms() - a;
        }
    }

    for (int w = 0; w < PAGE_XPLACE_DIAG_WARMUP; w++) {
        for (int k = 0; k < 3; k++) {
            hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
            for (int r = 0; r < reps; r++)
                (void)page_cont_cpu_component(csr, A, xs[k], 1);
            hthread_group_wait(gid);
        }
    }
    for (int i = 0; i < PAGE_XPLACE_DIAG_ITER; i++) {
        for (int pos = 0; pos < 3; pos++) {
            const int k = (i + pos) % 3;
            hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
            const double a = now_ms();
            for (int r = 0; r < reps; r++)
                (void)page_cont_cpu_component(csr, A, xs[k], 1);
            conc[k][i] = now_ms() - a;
            hthread_group_wait(gid);
        }
    }

    double sm[3], cm[3];
    for (int k = 0; k < 3; k++) {
        sm[k] = trimmed_mean_1each(solo[k], PAGE_XPLACE_DIAG_ITER);
        cm[k] = trimmed_mean_1each(conc[k], PAGE_XPLACE_DIAG_ITER);
    }
    printf("[XPLACE_GATHER] reps=%d shared_solo_ms=%.6f firsttouch_solo_ms=%.6f "
           "interleave_solo_ms=%.6f shared_with_dsp_ms=%.6f "
           "firsttouch_with_dsp_ms=%.6f interleave_with_dsp_ms=%.6f "
           "shared_slowdown=%.8f firsttouch_slowdown=%.8f interleave_slowdown=%.8f "
           "firsttouch_over_shared_solo=%.8f interleave_over_shared_solo=%.8f "
           "interleave_over_firsttouch_solo=%.8f firsttouch_over_shared_concurrent=%.8f "
           "interleave_over_shared_concurrent=%.8f interleave_over_firsttouch_concurrent=%.8f\n",
           reps, sm[0], sm[1], sm[2], cm[0], cm[1], cm[2],
           sm[0] > 0.0 ? cm[0] / sm[0] : 0.0,
           sm[1] > 0.0 ? cm[1] / sm[1] : 0.0,
           sm[2] > 0.0 ? cm[2] / sm[2] : 0.0,
           sm[0] > 0.0 ? sm[1] / sm[0] : 0.0,
           sm[0] > 0.0 ? sm[2] / sm[0] : 0.0,
           sm[1] > 0.0 ? sm[2] / sm[1] : 0.0,
           cm[0] > 0.0 ? cm[1] / cm[0] : 0.0,
           cm[0] > 0.0 ? cm[2] / cm[0] : 0.0,
           cm[1] > 0.0 ? cm[2] / cm[1] : 0.0);
}

static void page_htx_gather_triple(const CSRMatrix *csr,
                                    const page_matrix *A,
                                    const MAT_VAL_TYPE *x_shared,
                                    const MAT_VAL_TYPE *x_hthread_private,
                                    const MAT_VAL_TYPE *x_interleave,
                                    int gid, int scalArgs, int vecArgs,
                                    unsigned long *spmv_args,
                                    double target_ms)
{
    const MAT_VAL_TYPE *xs[3] = {x_shared, x_hthread_private, x_interleave};
    int reps = 1;
    for (int k = 0; k < 3; k++) {
        const int r = page_cont_tune_cpu_component(csr, A, xs[k], 1, target_ms);
        if (r > reps) reps = r;
    }
    if (reps < 1) reps = 1;

    double solo[3][PAGE_XPLACE_DIAG_ITER];
    double conc[3][PAGE_XPLACE_DIAG_ITER];
    memset(solo, 0, sizeof(solo));
    memset(conc, 0, sizeof(conc));

    for (int w = 0; w < PAGE_XPLACE_DIAG_WARMUP; w++) {
        for (int k = 0; k < 3; k++)
            for (int r = 0; r < reps; r++)
                (void)page_cont_cpu_component(csr, A, xs[k], 1);
    }
    for (int it = 0; it < PAGE_XPLACE_DIAG_ITER; it++) {
        for (int pos = 0; pos < 3; pos++) {
            const int k = (it + pos) % 3;
            const double a = now_ms();
            for (int r = 0; r < reps; r++)
                (void)page_cont_cpu_component(csr, A, xs[k], 1);
            solo[k][it] = now_ms() - a;
        }
    }

    for (int w = 0; w < PAGE_XPLACE_DIAG_WARMUP; w++) {
        for (int k = 0; k < 3; k++) {
            hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
            for (int r = 0; r < reps; r++)
                (void)page_cont_cpu_component(csr, A, xs[k], 1);
            hthread_group_wait(gid);
        }
    }
    for (int it = 0; it < PAGE_XPLACE_DIAG_ITER; it++) {
        for (int pos = 0; pos < 3; pos++) {
            const int k = (it + pos) % 3;
            hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
            const double a = now_ms();
            for (int r = 0; r < reps; r++)
                (void)page_cont_cpu_component(csr, A, xs[k], 1);
            conc[k][it] = now_ms() - a;
            hthread_group_wait(gid);
        }
    }

    double sm[3], cm[3];
    for (int k = 0; k < 3; k++) {
        sm[k] = trimmed_mean_1each(solo[k], PAGE_XPLACE_DIAG_ITER);
        cm[k] = trimmed_mean_1each(conc[k], PAGE_XPLACE_DIAG_ITER);
    }
    printf("[HTX_GATHER] reps=%d shared_solo_ms=%.6f hthread_private_solo_ms=%.6f "
           "interleave_solo_ms=%.6f shared_with_dsp_ms=%.6f "
           "hthread_private_with_dsp_ms=%.6f interleave_with_dsp_ms=%.6f "
           "shared_slowdown=%.8f hthread_private_slowdown=%.8f "
           "interleave_slowdown=%.8f hthread_private_over_shared_solo=%.8f "
           "interleave_over_shared_solo=%.8f hthread_private_over_interleave_solo=%.8f "
           "hthread_private_over_shared_concurrent=%.8f "
           "interleave_over_shared_concurrent=%.8f "
           "hthread_private_over_interleave_concurrent=%.8f\n",
           reps, sm[0], sm[1], sm[2], cm[0], cm[1], cm[2],
           sm[0] > 0.0 ? cm[0] / sm[0] : 0.0,
           sm[1] > 0.0 ? cm[1] / sm[1] : 0.0,
           sm[2] > 0.0 ? cm[2] / sm[2] : 0.0,
           sm[0] > 0.0 ? sm[1] / sm[0] : 0.0,
           sm[0] > 0.0 ? sm[2] / sm[0] : 0.0,
           sm[2] > 0.0 ? sm[1] / sm[2] : 0.0,
           cm[0] > 0.0 ? cm[1] / cm[0] : 0.0,
           cm[0] > 0.0 ? cm[2] / cm[0] : 0.0,
           cm[2] > 0.0 ? cm[1] / cm[2] : 0.0);
}

static void page_thp_gather_triple(const CSRMatrix *csr,
                                    const page_matrix *A,
                                    const MAT_VAL_TYPE *x_shared,
                                    const MAT_VAL_TYPE *x_interleave,
                                    const MAT_VAL_TYPE *x_nohuge,
                                    int gid, int scalArgs, int vecArgs,
                                    unsigned long *spmv_args,
                                    double target_ms)
{
    const MAT_VAL_TYPE *xs[3] = {x_shared, x_interleave, x_nohuge};
    int reps = 1;
    for (int k = 0; k < 3; k++) {
        const int r = page_cont_tune_cpu_component(csr, A, xs[k], 1, target_ms);
        if (r > reps) reps = r;
    }
    if (reps < 1) reps = 1;

    double solo[3][PAGE_XPLACE_DIAG_ITER];
    double conc[3][PAGE_XPLACE_DIAG_ITER];
    memset(solo, 0, sizeof(solo));
    memset(conc, 0, sizeof(conc));

    for (int w = 0; w < PAGE_XPLACE_DIAG_WARMUP; w++) {
        for (int k = 0; k < 3; k++)
            for (int r = 0; r < reps; r++)
                (void)page_cont_cpu_component(csr, A, xs[k], 1);
    }
    for (int it = 0; it < PAGE_XPLACE_DIAG_ITER; it++) {
        for (int pos = 0; pos < 3; pos++) {
            const int k = (it + pos) % 3;
            const double a = now_ms();
            for (int r = 0; r < reps; r++)
                (void)page_cont_cpu_component(csr, A, xs[k], 1);
            solo[k][it] = now_ms() - a;
        }
    }

    for (int w = 0; w < PAGE_XPLACE_DIAG_WARMUP; w++) {
        for (int k = 0; k < 3; k++) {
            hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
            for (int r = 0; r < reps; r++)
                (void)page_cont_cpu_component(csr, A, xs[k], 1);
            hthread_group_wait(gid);
        }
    }
    for (int it = 0; it < PAGE_XPLACE_DIAG_ITER; it++) {
        for (int pos = 0; pos < 3; pos++) {
            const int k = (it + pos) % 3;
            hthread_group_exec(gid, "page_spmv", scalArgs, vecArgs, spmv_args);
            const double a = now_ms();
            for (int r = 0; r < reps; r++)
                (void)page_cont_cpu_component(csr, A, xs[k], 1);
            conc[k][it] = now_ms() - a;
            hthread_group_wait(gid);
        }
    }

    double sm[3], cm[3];
    for (int k = 0; k < 3; k++) {
        sm[k] = trimmed_mean_1each(solo[k], PAGE_XPLACE_DIAG_ITER);
        cm[k] = trimmed_mean_1each(conc[k], PAGE_XPLACE_DIAG_ITER);
    }
    printf("[THP_GATHER] reps=%d shared_solo_ms=%.6f interleave_solo_ms=%.6f "
           "nohuge_solo_ms=%.6f shared_with_dsp_ms=%.6f "
           "interleave_with_dsp_ms=%.6f nohuge_with_dsp_ms=%.6f "
           "shared_slowdown=%.8f interleave_slowdown=%.8f nohuge_slowdown=%.8f "
           "interleave_over_shared_solo=%.8f nohuge_over_shared_solo=%.8f "
           "nohuge_over_interleave_solo=%.8f interleave_over_shared_concurrent=%.8f "
           "nohuge_over_shared_concurrent=%.8f nohuge_over_interleave_concurrent=%.8f\n",
           reps, sm[0], sm[1], sm[2], cm[0], cm[1], cm[2],
           sm[0] > 0.0 ? cm[0] / sm[0] : 0.0,
           sm[1] > 0.0 ? cm[1] / sm[1] : 0.0,
           sm[2] > 0.0 ? cm[2] / sm[2] : 0.0,
           sm[0] > 0.0 ? sm[1] / sm[0] : 0.0,
           sm[0] > 0.0 ? sm[2] / sm[0] : 0.0,
           sm[1] > 0.0 ? sm[2] / sm[1] : 0.0,
           cm[0] > 0.0 ? cm[1] / cm[0] : 0.0,
           cm[0] > 0.0 ? cm[2] / cm[0] : 0.0,
           cm[1] > 0.0 ? cm[2] / cm[1] : 0.0);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "用法: %s <matrix.mtx> <coreNum> <cluster_id>\n", argv[0]);
        return 1;
    }
    char *filename       = argv[1];
    const int coreNum    = atoi(argv[2]);
    const int cluster_id = atoi(argv[3]);
    const int omp_threads_base = omp_get_max_threads();

    int m, n, nnz, isSymmetric;
    int *csrRowPtr, *csrColIdx;
    MAT_VAL_TYPE *csrVal;

    double t0 = now_ms();
    if (mmio_allinone(&m, &n, &nnz, &isSymmetric, &csrRowPtr, &csrColIdx,
                      &csrVal, filename) != 0) {
        fprintf(stderr, "[PAGE] 读取矩阵失败: %s\n", filename);
        return 1;
    }
    const double t_read = now_ms() - t0;
    const char *name = basename_noext(filename);
    printf("[T] read_time: %.3f (ms)\n", t_read);

    CSRMatrix csr;
    csr.numRows = m; csr.numCols = n; csr.numNonzeros = nnz;
    csr.rowPointers = csrRowPtr; csr.colIndices = csrColIdx; csr.values = csrVal;
    csr.value_mode = PAGE_VALUE_GENERAL;
    csr.constant_value = 0.0;

    const char *cpu_baseline_env = getenv("PAGE_CPU_BASELINE");
    const char *cpu_baseline_req = (cpu_baseline_env && *cpu_baseline_env)
                                 ? cpu_baseline_env : "opt";
    if (page_select_cpu_baseline_from_env() != 0) return 5;
    printf("[CPU_BASELINE] requested=%s selected=%s threads=%d exact_halav_csr_calls=%d "
           "hybrid_cpu_uses_same_profile=1\n",
           cpu_baseline_req, page_cpu_baseline_name(), omp_get_max_threads(),
           g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR);

    const char *v18_fast_prep_env = getenv("PAGE_FAST_PREP");
    const int v18_fast_prep = (v18_fast_prep_env && strcmp(v18_fast_prep_env, "1") == 0);

    const char *v19_fast_shadow_env = getenv("PAGE_FAST_SHADOW_PLAN");
    const int v19_fast_shadow = (v19_fast_shadow_env && strcmp(v19_fast_shadow_env, "1") == 0);

    int numa_first_touch_applied = 0;
    double t_numa_csr_first_touch = 0.0, t_numa_x_first_touch = 0.0;
    if (g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR) {
        printf("[NUMA_PLACEMENT] strategy=halav_original_mmio_serial applied=0 "
               "csr_first_touch_ms=0.000 omp_threads=%d omp_proc_bind=%s omp_places=%s\n",
               omp_get_max_threads(),
               getenv("OMP_PROC_BIND") ? getenv("OMP_PROC_BIND") : "",
               getenv("OMP_PLACES") ? getenv("OMP_PLACES") : "");
    } else {
        const double nft0 = now_ms();
        numa_first_touch_applied = page_parallel_first_touch_csr(&csr);
        t_numa_csr_first_touch = now_ms() - nft0;
        if (!numa_first_touch_applied) {
            fprintf(stderr, "[NUMA_PLACEMENT] ERROR: parallel CSR first-touch failed\n");
            return 3;
        }
        csrRowPtr = csr.rowPointers; csrColIdx = csr.colIndices; csrVal = csr.values;
        printf("[NUMA_PLACEMENT] strategy=parallel_first_touch applied=%d "
               "csr_first_touch_ms=%.3f omp_threads=%d omp_proc_bind=%s omp_places=%s\n",
               numa_first_touch_applied, t_numa_csr_first_touch, omp_get_max_threads(),
               getenv("OMP_PROC_BIND") ? getenv("OMP_PROC_BIND") : "",
               getenv("OMP_PLACES") ? getenv("OMP_PLACES") : "");
    }

    const int input_nnz = csr.numNonzeros;
    page_value_analysis va;
    page_value_analysis_init(&va);
    const double tscan0 = now_ms();
    int analysis_ok = 0;
    int v18_fast_value_mode = PAGE_VALUE_GENERAL;
    MAT_VAL_TYPE v18_fast_constant_value = (MAT_VAL_TYPE)0.0;
    const int v18_fast_value_only = v18_fast_prep &&
        g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR;
    if (v18_fast_value_only) {
        analysis_ok = page_classify_original_values_fast(
            &csr, &v18_fast_value_mode, &v18_fast_constant_value);
        va.input_nnz = input_nnz;
        va.numeric_nnz = input_nnz;
        va.explicit_zeros = 0;
        va.original_value_mode = v18_fast_value_mode;
        va.original_constant_value = v18_fast_constant_value;
        va.numeric_value_mode = v18_fast_value_mode;
        va.numeric_constant_value = v18_fast_constant_value;
    } else {
        analysis_ok = page_analyze_numeric_structure(&csr, &va);
    }
    const double t_numeric_scan = now_ms() - tscan0;

    const long long explicit_zeros_found = v18_fast_value_only ? -1LL : (analysis_ok ? va.explicit_zeros : 0);
    const long long numeric_nnz_available = v18_fast_value_only ? -1LL : (analysis_ok ? va.numeric_nnz : input_nnz);
    const long long sell_stored_original = analysis_ok ? va.sell_stored_original : 0;
    const long long sell_stored_numeric = analysis_ok ? va.sell_stored_numeric : 0;
    const double sell_amp_original = analysis_ok ? va.sell_amp_original : 1.0;
    const double sell_amp_numeric = analysis_ok ? va.sell_amp_numeric : 1.0;
    const int geometry_checked = analysis_ok ? va.geometry_checked : 0;

    int prune_selected = 0;
    long long pruned_zeros = 0;
    double t_shadow_select = 0.0;
    double t_prune_compact = 0.0;
    page_shadow_decision shadow; memset(&shadow, 0, sizeof(shadow));
    shadow.original_ms = shadow.numeric_ms = 1.0e300; shadow.gain = 1.0;

    int candidate_value_mode = PAGE_VALUE_GENERAL;
    MAT_VAL_TYPE candidate_constant_value = (MAT_VAL_TYPE)0.0;
    double value_general_ms = 0.0, value_candidate_ms = 0.0;
    double t_value_select = 0.0;
    int selected_value_mode = PAGE_VALUE_GENERAL;

    const double available_prune_ratio = (!v18_fast_value_only && input_nnz > 0)
        ? (double)explicit_zeros_found / (double)input_nnz : 0.0;
    double applied_prune_ratio = 0.0;

    printf("[V628R2] input_nnz=%d, numeric_nnz_available=%lld, explicit_zeros_found=%lld, "
           "available_prune_ratio=%.8f, scan_ms=%.3f, geometry_checked=%d, "
           "sell_stored_original=%lld, sell_stored_numeric=%lld, "
           "sell_amp_original=%.8f, sell_amp_numeric=%.8f, "
           "analysis_mode=%s zero_count_measured=%d\n",
           input_nnz, numeric_nnz_available, explicit_zeros_found,
           available_prune_ratio, t_numeric_scan, geometry_checked,
           sell_stored_original, sell_stored_numeric,
           sell_amp_original, sell_amp_numeric,
           v18_fast_value_only ? "halav_fast_value_only" : "full_numeric_geometry",
           v18_fast_value_only ? 0 : 1);

    const double td0 = now_ms();
    hthread_dev_open(cluster_id);
    hthread_dat_load(cluster_id, "kernel.dat");
    printf("[T] device_init_time: %.3f (ms)\n", now_ms() - td0);

    double t_platform_calib = 0.0;
    int platform_cache_hit = 0;
    const char *platform_cache_path = getenv("PAGE_PLATFORM_CACHE");
    page_platform_calib pc; memset(&pc, 0, sizeof(pc));
    if (load_platform_cache(platform_cache_path, coreNum, cluster_id, &pc)) {
        platform_cache_hit = 1;
        printf("[CACHE] platform_calib_reused = 1, path = %s\n", platform_cache_path);
        printf("[BW] stream_bw: %.2f (GB/s) cached\n", pc.bw_probe_gbs);
        printf("[BW] xload_bw: %.2f (GB/s), xload64: 0.000000 ms/round, fixed: %.6f ms cached\n",
               pc.bw_xload / 1e9, pc.xload_fixed_ms);
        printf("[BW] xload_bytes_per_sec: %.6e\n", pc.bw_xload);
        printf("[BW] pipeline_bw: %.2f (GB/s) cached\n", pc.bw_pipe / 1e9);
        printf("[BW] launch_ms: %.6f (ms) cached\n", pc.launch_ms);
        printf("[BW] barrier_ms: %.6f (ms) cached\n", pc.barrier_ms);
    } else {
        double tpc0 = now_ms();
        pc.bw_probe_gbs = run_stream_probe(coreNum, cluster_id);
        t_platform_calib += now_ms() - tpc0;
        tpc0 = now_ms();
        pc.bw_xload = calib_xload_path(coreNum, cluster_id, &pc.xload_fixed_ms);
        t_platform_calib += now_ms() - tpc0;
        tpc0 = now_ms();
        pc.bw_pipe = calib_dsp_pipeline(coreNum, cluster_id,
                                        &pc.launch_ms, &pc.barrier_ms);
        t_platform_calib += now_ms() - tpc0;
        save_platform_cache(platform_cache_path, coreNum, cluster_id, &pc);
        printf("[CACHE] platform_calib_reused = 0, path = %s\n",
               platform_cache_path ? platform_cache_path : "");
    }
    const double bw_probe_gbs = pc.bw_probe_gbs;
    const double bw_xload = pc.bw_xload;
    const double xload_fixed_ms = pc.xload_fixed_ms;

    const char *preflight_only = getenv("PAGE_PLATFORM_PREFLIGHT_ONLY");
    if (preflight_only && strcmp(preflight_only, "1") == 0) {
        printf("[STABLE_CAL] platform_preflight_only=1 platform_cache_hit=%d\n", platform_cache_hit);
        page_value_analysis_free(&va);
        free(csrRowPtr); free(csrColIdx); free(csrVal);
        hthread_dev_close(cluster_id);
        return 0;
    }

    const int n_pad = ((n + PAGE_XBLOCK_ELEMS - 1) / PAGE_XBLOCK_ELEMS)
                    * PAGE_XBLOCK_ELEMS;
    MAT_VAL_TYPE *x = NULL;
    MAT_VAL_TYPE *ref = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
    double *scale     = (double *)malloc(sizeof(double) * (size_t)m);

    MAT_VAL_TYPE *x_seed = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)n_pad);
    if (!x_seed) { fprintf(stderr, "[NUMA_PLACEMENT] x_seed allocation failed\n"); return 3; }
    srand(1);
    for (int j = 0; j < n; ++j) x_seed[j] = (MAT_VAL_TYPE)((rand() % 2000) - 1000) / 13.0;
    for (int j = n; j < n_pad; ++j) x_seed[j] = 0.0;
    x = (MAT_VAL_TYPE *)hthread_malloc(cluster_id,
                sizeof(MAT_VAL_TYPE) * (size_t)n_pad, HT_MEM_RW);
    if (!x) { free(x_seed); fprintf(stderr, "[NUMA_PLACEMENT] x allocation failed\n"); return 3; }
    const double nftx0 = now_ms();
    if (g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR) {
        for (int j = 0; j < n_pad; ++j) x[j] = x_seed[j];
        printf("[HALAV_CSR_CPU] x_init=serial_hthread original_csr_parallel_first_touch=0\n");
    } else {
        page_parallel_first_touch_x(x, x_seed, n_pad);
    }
    t_numa_x_first_touch = now_ms() - nftx0;
    free(x_seed);
    printf("[NUMA_PLACEMENT] x_first_touch_ms=%.3f strategy=%s\n",
           t_numa_x_first_touch,
           g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR
               ? "halav_serial_hthread" : "parallel_first_touch");

    const MAT_VAL_TYPE *cpu_x = x;

    page_xplace_mapping x_cpu_prod_map;
    memset(&x_cpu_prod_map, 0, sizeof(x_cpu_prod_map));
    page_v17_csr_mapping v17_csr_map;
    memset(&v17_csr_map, 0, sizeof(v17_csr_map));
    CSRMatrix v17_original_csr;
    memset(&v17_original_csr, 0, sizeof(v17_original_csr));
    double t_dual_x_prod_setup = 0.0;
    double t_v17_csr_numa_setup = 0.0;
    double t_v15_rebalance = 0.0;
    const char *dual_x_prod_env = getenv("PAGE_DUAL_X_PROD");
    const int dual_x_prod = (dual_x_prod_env && strcmp(dual_x_prod_env, "1") == 0);
    const char *cpu_hybrid_side_env = getenv("PAGE_CPU_HYBRID_SIDE");
    const int cpu_hybrid_side = (cpu_hybrid_side_env && strcmp(cpu_hybrid_side_env, "1") == 0);
    const char *v15_rebalance_env = getenv("PAGE_CONT_REBALANCE");
    const int v15_rebalance = (v15_rebalance_env && strcmp(v15_rebalance_env, "1") == 0);
    const char *v17_csr_numa_env = getenv("PAGE_CSR_NUMA_PROD");
    const int v17_csr_numa = (v17_csr_numa_env && strcmp(v17_csr_numa_env, "1") == 0);
    const char *ab_mem_env = getenv("PAGE_ABLATE_MEMORY_PLACEMENT");
    const int ablate_memory_placement = (ab_mem_env && strcmp(ab_mem_env, "1") == 0);

    page_opts opt; page_opts_default(&opt);
    const char *exec_mode_env = getenv("PAGE_EXEC_MODE");
    const char *exec_mode = (exec_mode_env && *exec_mode_env) ? exec_mode_env : "auto";
    const char *same_part_diag_env = getenv("PAGE_SAME_PART_DIAG");
    const int same_part_diag = (same_part_diag_env && strcmp(same_part_diag_env, "1") == 0);
    const char *contention_diag_env = getenv("PAGE_CONTENTION_DIAG");
    const int contention_diag = (contention_diag_env && strcmp(contention_diag_env, "1") == 0);
    const char *dual_x_diag_env = getenv("PAGE_DUAL_X_DIAG");
    const int dual_x_diag = (dual_x_diag_env && strcmp(dual_x_diag_env, "1") == 0);
    const char *xplace_diag_env = getenv("PAGE_XPLACEMENT_DIAG");
    const int xplace_diag = (xplace_diag_env && strcmp(xplace_diag_env, "1") == 0);
    const char *htx_diag_env = getenv("PAGE_HTX_DIAG");
    const int htx_diag = (htx_diag_env && strcmp(htx_diag_env, "1") == 0);
    const char *thp_diag_env = getenv("PAGE_THP_DIAG");
    const int thp_diag = (thp_diag_env && strcmp(thp_diag_env, "1") == 0);
    if (contention_diag && !same_part_diag) {
        fprintf(stderr, "[CONT] ERROR: PAGE_CONTENTION_DIAG=1 requires PAGE_SAME_PART_DIAG=1\n");
        return 5;
    }
    if (dual_x_diag && !same_part_diag) {
        fprintf(stderr, "[DUAL_X] ERROR: PAGE_DUAL_X_DIAG=1 requires PAGE_SAME_PART_DIAG=1\n");
        return 5;
    }
    if (xplace_diag && !same_part_diag) {
        fprintf(stderr, "[XPLACE] ERROR: PAGE_XPLACEMENT_DIAG=1 requires PAGE_SAME_PART_DIAG=1\n");
        return 5;
    }
    if (htx_diag && !same_part_diag) {
        fprintf(stderr, "[HTX] ERROR: PAGE_HTX_DIAG=1 requires PAGE_SAME_PART_DIAG=1\n");
        return 5;
    }
    if (thp_diag && !same_part_diag) {
        fprintf(stderr, "[THP] ERROR: PAGE_THP_DIAG=1 requires PAGE_SAME_PART_DIAG=1\n");
        return 5;
    }
    const int xdiag_count = (dual_x_diag ? 1 : 0) + (xplace_diag ? 1 : 0) +
                            (htx_diag ? 1 : 0) + (thp_diag ? 1 : 0);
    if (xdiag_count > 1) {
        fprintf(stderr, "[THP] ERROR: enable only one of PAGE_DUAL_X_DIAG, PAGE_XPLACEMENT_DIAG, PAGE_HTX_DIAG, PAGE_THP_DIAG\n");
        return 5;
    }
    if (dual_x_prod && xdiag_count != 0) {
        fprintf(stderr, "[DUAL_X_PROD] ERROR: production dual-x cannot be combined with V5-V8 x diagnostics\n");
        return 5;
    }
    if (cpu_hybrid_side && !dual_x_prod) {
        fprintf(stderr, "[V14_CPU_HYBRIDSIDE] ERROR: PAGE_CPU_HYBRID_SIDE=1 requires PAGE_DUAL_X_PROD=1\n");
        return 5;
    }
    if (v15_rebalance && (!dual_x_prod || cpu_hybrid_side || strcmp(exec_mode,"hybrid")!=0 ||
                          g_page_cpu_baseline!=PAGE_CPU_BASELINE_HALAV_CSR)) {
        fprintf(stderr, "[V15_REBAL] ERROR: PAGE_CONT_REBALANCE=1 requires forced Hybrid, halav_csr CPU, production dual-x, and no CPU-HybridSide mode\n");
        return 5;
    }
    if (v17_csr_numa) {
        const int v17_hybrid_ok = (strcmp(exec_mode, "hybrid") == 0 && !cpu_hybrid_side);
        const int v17_cpu_side_ok = (strcmp(exec_mode, "cpu") == 0 && cpu_hybrid_side);
        if (!dual_x_prod || g_page_cpu_baseline != PAGE_CPU_BASELINE_HALAV_CSR ||
            (!v17_hybrid_ok && !v17_cpu_side_ok)) {
            fprintf(stderr, "[V17_CSR_NUMA] ERROR: PAGE_CSR_NUMA_PROD=1 requires halav_csr + production dual-x and either Page-Hybrid or CPU-HybridSide\n");
            return 5;
        }
    }
    if (v18_fast_prep) {
        const int v18_hybrid_ok = (strcmp(exec_mode, "hybrid") == 0 && !cpu_hybrid_side);
        const int v18_cpu_side_ok = (strcmp(exec_mode, "cpu") == 0 && cpu_hybrid_side);
        if (!v17_csr_numa || !dual_x_prod ||
            g_page_cpu_baseline != PAGE_CPU_BASELINE_HALAV_CSR ||
            (!v18_hybrid_ok && !v18_cpu_side_ok)) {
            fprintf(stderr, "[V18_FAST_PREP] ERROR: PAGE_FAST_PREP=1 requires the V17 strict HaLAV-CSR/private-x/private-CSR CPU-HybridSide or Page-Hybrid path\n");
            return 5;
        }
    }
    if (v19_fast_shadow) {
        if (!v18_fast_prep || strcmp(exec_mode, "hybrid") != 0 || cpu_hybrid_side ||
            !v17_csr_numa || !dual_x_prod ||
            g_page_cpu_baseline != PAGE_CPU_BASELINE_HALAV_CSR) {
            fprintf(stderr, "[V19_FAST_SHADOW] ERROR: PAGE_FAST_SHADOW_PLAN=1 requires V18 fast preprocessing on forced Page-Hybrid with V17 CSR-NUMA and production dual-x\n");
            return 5;
        }
    }
    printf("[DUAL_X] enabled=%d policy=same_partition_cpu_private_x_dsp_visible_x resources=%dCPU+%dDSP\n",
           dual_x_diag, omp_get_max_threads(), coreNum);
    printf("[XPLACE] enabled=%d policy=shared_vs_firsttouch_vs_runtime_allowed_interleave resources=%dCPU+%dDSP\n",
           xplace_diag, omp_get_max_threads(), coreNum);
    printf("[HTX] enabled=%d policy=shared_hthread_vs_separate_hthread_vs_runtime_interleave resources=%dCPU+%dDSP\n",
           htx_diag, omp_get_max_threads(), coreNum);
    printf("[THP] enabled=%d policy=shared_hthread_vs_interleave_default_vs_interleave_nohuge resources=%dCPU+%dDSP\n",
           thp_diag, omp_get_max_threads(), coreNum);
    printf("[DUAL_X_PROD] enabled=%d policy=cpu_nohuge_runtime_interleave_dsp_hthread_recalibrate_replan\n",
           dual_x_prod);
    printf("[V14_CPU_HYBRIDSIDE] enabled=%d policy=isolated_full_cpu_same_halav_csr_private_x_as_hybrid\n",
           cpu_hybrid_side);
    printf("[V15_REBAL] enabled=%d policy=one_shot_runtime_concurrent_tail_balance_no_matrix_names\n",v15_rebalance);
    printf("[V17_CSR_NUMA] enabled=%d policy=cpu_private_csr_mmap_nohuge_runtime_interleave original_mmio_retained_for_postformal_ab=1\n",
           v17_csr_numa);
    printf("[V18_FAST_PREP] enabled=%d policy=halav_value_only_plus_shadow_then_single_materialize formal_spmv_unchanged=1\n",
           v18_fast_prep);
    printf("[V19_FAST_SHADOW] enabled=%d policy=exact_sigma_block_span_precheck_skip_impossible_window_rebuild formal_spmv_unchanged=1 v15_repartition_unchanged=1\n",
           v19_fast_shadow);
    printf("[FINAL_PAGE] column_window=%d policy=compile_time_excluded final_config=1\n",
           PAGE_FINAL_COLUMN_WINDOW);
    printf("[ABLATION] vrow=%d sigma=%d packed_x=%d memory_placement=%d prefix=%s rebalance=%d fast_prep=%d fast_shadow=%d\n",
           getenv("PAGE_ABLATE_VROW") && strcmp(getenv("PAGE_ABLATE_VROW"),"1")==0,
           getenv("PAGE_ABLATE_SIGMA") && strcmp(getenv("PAGE_ABLATE_SIGMA"),"1")==0,
           getenv("PAGE_PACKED_X") && strcmp(getenv("PAGE_PACKED_X"),"0")==0,
           ablate_memory_placement,
           getenv("PAGE_PREFIX_CALIB") ? getenv("PAGE_PREFIX_CALIB") : "0",
           v15_rebalance, v18_fast_prep, v19_fast_shadow);
    int force_exec_mode = 0;
    if (strcmp(exec_mode, "cpu") == 0) {
        opt.force_cpu = 1;
        force_exec_mode = 1;
    } else if (strcmp(exec_mode, "hybrid") == 0) {
        opt.force_hybrid = 1;
        force_exec_mode = 1;
    } else if (strcmp(exec_mode, "dsp") == 0) {
        opt.row_off_override = 0;
        opt.plan_policy = 0;
        force_exec_mode = 1;
    } else if (strcmp(exec_mode, "auto") != 0) {
        fprintf(stderr, "[FORCE_MODE] ERROR: PAGE_EXEC_MODE must be auto|cpu|hybrid|dsp\n");
        return 5;
    }
    const char *dsp_index_env = getenv("PAGE_DSP_INDEX");
    if (dsp_index_env && *dsp_index_env) {
        if (strcmp(dsp_index_env, "32") == 0) opt.force_idx32 = 1;
        else if (strcmp(dsp_index_env, "auto") != 0) {
            fprintf(stderr, "[DSP_V1] ERROR: PAGE_DSP_INDEX must be auto|32\n");
            return 5;
        }
    }

    int hybrid_v2_planner_threads = omp_threads_base;
    if (strcmp(exec_mode, "hybrid") == 0) {
        const char *hte = getenv("PAGE_HYBRID_THREADS");
        if (hte && *hte && strcmp(hte, "auto") != 0 &&
            strcmp(hte, "base") != 0 && strcmp(hte, "off") != 0) {
            char *end = NULL;
            long v = strtol(hte, &end, 10);
            if (!end || *end != '\0' || v < 1 || v > omp_threads_base) {
                fprintf(stderr, "[HYBRID_V2] ERROR: fixed planner threads must be 1..%d\n",
                        omp_threads_base);
                return 5;
            }
            hybrid_v2_planner_threads = (int)v;
            omp_set_dynamic(0);
            omp_set_num_threads(hybrid_v2_planner_threads);
        }
    }
    printf("[HYBRID_V2] first_touch_threads=%d planner_threads=%d policy=%s coreNum=%d\n",
           omp_threads_base, hybrid_v2_planner_threads,
           getenv("PAGE_HYBRID_THREADS") ? getenv("PAGE_HYBRID_THREADS") : "base",
           coreNum);
    if (same_part_diag) {
        if (strcmp(exec_mode, "hybrid") != 0 || omp_threads_base != 16 ||
            hybrid_v2_planner_threads != 16 || coreNum != 24) {
            fprintf(stderr,
                    "[DIAG] ERROR: same-partition diagnostic requires "
                    "PAGE_EXEC_MODE=hybrid, base/planner CPU threads=16, DSP cores=24\n");
            return 5;
        }
        printf("[DIAG] enabled=1 design=same_partition cpu_threads=16 dsp_cores=24 "
               "cpu_kernel_changed=0 dsp_kernel_changed=0\n");
    }
    if (contention_diag) {
        printf("[CONT] enabled=1 design=controlled_interference cpu_threads=16 dsp_cores=24 "
               "resource_grid=0 production_cpu_kernel_changed=0 production_dsp_kernel_changed=0\n");
    }

    if (dual_x_prod && !ablate_memory_placement) {
        const int prod_hybrid_ok = (strcmp(exec_mode, "hybrid") == 0 && !cpu_hybrid_side);
        const int prod_cpu_side_ok = (strcmp(exec_mode, "cpu") == 0 && cpu_hybrid_side &&
                                      g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR);
        if ((!prod_hybrid_ok && !prod_cpu_side_ok) || omp_threads_base != 16 ||
            hybrid_v2_planner_threads != 16 || coreNum != 24) {
            fprintf(stderr,
                    "[DUAL_X_PROD] ERROR: production private-x requires either forced Hybrid, "
                    "or forced CPU with PAGE_CPU_HYBRID_SIDE=1 and halav_csr; "
                    "resources must be 16 CPU threads and 24 DSP cores\n");
            return 5;
        }
        const size_t xbytes = (size_t)n_pad * sizeof(MAT_VAL_TYPE);
        const double tx0 = now_ms();
        x_cpu_prod_map = page_xplace_alloc_interleaved_nohuge(x, n_pad);
        t_dual_x_prod_setup = now_ms() - tx0;
        if (!x_cpu_prod_map.ptr || !x_cpu_prod_map.mbind_ok ||
            !x_cpu_prod_map.madvise_ok) {
            fprintf(stderr,
                    "[DUAL_X_PROD] ERROR: x_cpu allocation failed errno=%d (%s) "
                    "madvise_errno=%d allowed=%s\n",
                    x_cpu_prod_map.err, strerror(x_cpu_prod_map.err),
                    x_cpu_prod_map.madvise_err,
                    x_cpu_prod_map.allowed_list[0] ? x_cpu_prod_map.allowed_list : "unknown");
            return 5;
        }
        const int equal = (memcmp(x_cpu_prod_map.ptr, x, xbytes) == 0);
        const int distinct = (x_cpu_prod_map.ptr != x);
        if (!equal || !distinct) {
            fprintf(stderr, "[DUAL_X_PROD] ERROR: x_cpu copy mismatch or aliased pointer\n");
            return 5;
        }
        cpu_x = x_cpu_prod_map.ptr;
        printf("[DUAL_X_PROD] allocation bytes=%zu setup_ms=%.6f equal=%d distinct=%d "
               "mbind_ok=%d madvise_ok=%d madvise_errno=%d allowed_nodes=%s "
               "allowed_count=%d maxnode=%lu x_dsp_ptr=%p x_cpu_ptr=%p\n",
               xbytes, t_dual_x_prod_setup, equal, distinct,
               x_cpu_prod_map.mbind_ok, x_cpu_prod_map.madvise_ok,
               x_cpu_prod_map.madvise_err, x_cpu_prod_map.allowed_list,
               x_cpu_prod_map.node_count, x_cpu_prod_map.maxnode,
               (void *)x, (void *)cpu_x);
        page_cont_numa_object("x_cpu_prod", cpu_x, xbytes);
        page_cont_numa_object_raw("x_cpu_prod", cpu_x);
        page_cont_smaps_object("x_cpu_prod", cpu_x);
    }
    if (v17_csr_numa && !ablate_memory_placement) {
        v17_original_csr = csr;
        v17_csr_map = page_v17_alloc_csr_interleaved_nohuge(&v17_original_csr);
        t_v17_csr_numa_setup = v17_csr_map.setup_ms;
        if (!v17_csr_map.valid) {
            fprintf(stderr, "[V17_CSR_NUMA] ERROR: private CSR allocation/copy failed allowed_nodes=%s\n",
                    v17_csr_map.allowed_list[0] ? v17_csr_map.allowed_list : "unknown");
            return 5;
        }
        csr = v17_csr_map.csr;
        const size_t rp_bytes = (size_t)(m + 1) * sizeof(int);
        const size_t ci_bytes = (size_t)(nnz > 0 ? nnz : 1) * sizeof(int);
        const size_t va_bytes = (size_t)(nnz > 0 ? nnz : 1) * sizeof(MAT_VAL_TYPE);
        printf("[V17_CSR_NUMA] allocation valid=1 setup_ms=%.6f rowptr_bytes=%zu colidx_bytes=%zu values_bytes=%zu "
               "mbind_ok=1 madvise_ok=1 allowed_nodes=%s allowed_count=%d maxnode=%lu "
               "original_rp=%p private_rp=%p original_ci=%p private_ci=%p original_va=%p private_va=%p\n",
               t_v17_csr_numa_setup, rp_bytes, ci_bytes, va_bytes,
               v17_csr_map.allowed_list, v17_csr_map.node_count, v17_csr_map.maxnode,
               (void *)v17_original_csr.rowPointers, (void *)csr.rowPointers,
               (void *)v17_original_csr.colIndices, (void *)csr.colIndices,
               (void *)v17_original_csr.values, (void *)csr.values);
        page_cont_numa_object("v17_csr_rowptr", csr.rowPointers, rp_bytes);
        page_cont_numa_object("v17_csr_colidx", csr.colIndices,
                               (size_t)nnz * sizeof(int));
        page_cont_numa_object("v17_csr_values", csr.values,
                               (size_t)nnz * sizeof(MAT_VAL_TYPE));
        page_cont_numa_object_raw("v17_csr_rowptr", csr.rowPointers);
        page_cont_numa_object_raw("v17_csr_colidx", csr.colIndices);
        page_cont_numa_object_raw("v17_csr_values", csr.values);
        page_cont_smaps_object("v17_csr_rowptr", csr.rowPointers);
        page_cont_smaps_object("v17_csr_colidx", csr.colIndices);
        page_cont_smaps_object("v17_csr_values", csr.values);
    }
    if (dual_x_prod && g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR) {
        printf("[V13] halav_csr_dual_x_prod=1 cpu_kernel=halav_csr_mv_csrSpa_mv "
               "csr_parallel_first_touch=0 zero_prune=0 "
               "cpu_x=nohuge_runtime_interleave dsp_x=serial_hthread cpu_csr=%s\n",
               v17_csr_numa ? "private_nohuge_runtime_interleave" : "halav_original_mmio_serial");
    }
    if (cpu_hybrid_side) {
        printf("[V14_CPU_HYBRIDSIDE] active=1 exec=cpu cpu_kernel=halav_csr_mv "
               "cpu_x=nohuge_runtime_interleave csr_placement=%s dsp_launch=0\n",
               v17_csr_numa ? "private_nohuge_runtime_interleave" : "halav_original_mmio_serial");
    }

    printf("[FORCE_MODE] requested=%s forced=%d force_idx32=%d\n", exec_mode, force_exec_mode, opt.force_idx32);
    opt.bw_xload = bw_xload;
    opt.xload_fixed_ms = xload_fixed_ms;

    if (bw_probe_gbs > 1.0 && bw_probe_gbs < 200.0) {
        opt.bw_ddr = bw_probe_gbs * 1e9;
        opt.bw_gsm = bw_probe_gbs * 1e9 * (PAGE_BW_GSM / PAGE_BW_DDR);
    }

    opt.bw_pipe = pc.bw_pipe;
    opt.launch_ms = pc.launch_ms;
    opt.barrier_ms = pc.barrier_ms;

    double t_arm_calib = 0.0, arm_calib_mad = 0.0, arm_calib_rel_sigma = 0.0;
    const char *no_calib = getenv("SPMV_ARM_CALIB");
    const double tcal0 = now_ms();
    if (!(no_calib && strcmp(no_calib, "0") == 0))
        opt.bw_arm = calib_arm_bw(&csr, cpu_x, ref, &t_arm_calib,
                                  &arm_calib_mad, &arm_calib_rel_sigma);
    const double t_calib = now_ms() - tcal0;
    printf("[BW] arm_bw: %.2f (GB/s)\n", opt.bw_arm / 1e9);
    printf("[BW] arm_calib_ms: %.4f (ms)\n", t_arm_calib);
    printf("[T] arm_calib_time: %.3f (ms)\n", t_calib);
    printf("[STABLE_CAL] arm_samples=%d arm_median_ms=%.6f arm_mad_ms=%.6f arm_rel_sigma=%.8f\n",
           SPMV_ARM_CALIB_ITER, t_arm_calib, arm_calib_mad, arm_calib_rel_sigma);

    double t_mix_calib = 0.0;
    double t_arm_mix_full = t_arm_calib;
    double mix_arm_mad = 0.0, mix_arm_rel_sigma = 0.0, mix_elapsed_mad = 0.0;
    const char *no_mix = getenv("SPMV_MIX_CALIB");
    const double tmix0 = now_ms();
    if (!(no_mix && strcmp(no_mix, "0") == 0))
        opt.bw_mix = calib_mix_bw(&csr, cpu_x, ref, coreNum, cluster_id,
                                  opt.bw_pipe, t_arm_calib, &t_mix_calib,
                                  &opt.bw_arm_mix, &opt.bw_pipe_mix, &t_arm_mix_full,
                                  &mix_arm_mad, &mix_arm_rel_sigma, &mix_elapsed_mad);
    else {
        opt.bw_mix = opt.bw_ddr;
        opt.bw_arm_mix = opt.bw_arm;
        opt.bw_pipe_mix = opt.bw_pipe;
        t_arm_mix_full = t_arm_calib;
    }
    const double t_mix_total = now_ms() - tmix0;
    const double t_matrix_calib = t_calib + t_mix_total;
    printf("[T] mix_calib_time: %.3f (ms)\n", t_mix_calib);
    printf("[T] platform_calib_time: %.3f (ms)\n", t_platform_calib);
    printf("[T] matrix_calib_time: %.3f (ms)\n", t_matrix_calib);
    printf("[PAGE] platform_cache_hit = %d\n", platform_cache_hit);

    if (!v18_fast_value_only && analysis_ok && explicit_zeros_found > 0 && va.numeric_nnz > 0) {
        CSRMatrix ncsr; int *nrp = NULL, *nci = NULL;
        const double tsh0 = now_ms();
        if (page_make_numeric_shadow_csr(&csr, &va, &ncsr, &nrp, &nci)) {
            page_stat so, sn; memset(&so, 0, sizeof(so)); memset(&sn, 0, sizeof(sn));
            page_matrix Ao, An; memset(&Ao, 0, sizeof(Ao)); memset(&An, 0, sizeof(An));
            double tpo[3] = {0,0,0}, tpn[3] = {0,0,0};
            page_build_shadow(&csr,  &so, &Ao, coreNum, cluster_id, tpo, &opt);
            page_build_shadow(&ncsr, &sn, &An, coreNum, cluster_id, tpn, &opt);

            shadow.original_ms = Ao.model_total_ms;
            shadow.numeric_ms = An.model_total_ms;
            shadow.gain = (An.model_total_ms > 0.0)
                        ? Ao.model_total_ms / An.model_total_ms : 1.0;
            shadow.original_row_off = (Ao.mode == PAGE_MODE_CPU) ? m : Ao.row_off;
            shadow.numeric_row_off = (An.mode == PAGE_MODE_CPU) ? m : An.row_off;
            shadow.original_cpu_ms = Ao.est_arm_ms;
            shadow.original_dsp_ms = Ao.est_dsp_ms;
            shadow.numeric_cpu_ms = An.est_arm_ms;
            shadow.numeric_dsp_ms = An.est_dsp_ms;
            shadow.original_suffix_stored = (Ao.mode == PAGE_MODE_CPU) ? 0 : Ao.stored;
            shadow.numeric_suffix_stored = (An.mode == PAGE_MODE_CPU) ? 0 : An.stored;
            shadow.prune_selected = (isfinite(An.model_total_ms) && isfinite(Ao.model_total_ms)
                                  && An.model_total_ms < Ao.model_total_ms);
            prune_selected = shadow.prune_selected;

            page_release_candidate(&Ao, &so);
            page_release_candidate(&An, &sn);
        }
        free(nrp); free(nci);
        t_shadow_select = now_ms() - tsh0;
    }

    if (g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR) {
        prune_selected = 0;
        shadow.prune_selected = 0;
        printf("[CPU_BASELINE] halav_csr_preserve_original_csr=1 zero_prune_forced_off=1 "
               "applies_to_cpu_and_hybrid=1\n");
    }

    const double tcompact0 = now_ms();
    if (prune_selected) pruned_zeros = page_apply_zero_prune(&csr);
    t_prune_compact = now_ms() - tcompact0;
    nnz = csr.numNonzeros;
    applied_prune_ratio = (input_nnz > 0)
        ? (double)pruned_zeros / (double)input_nnz : 0.0;

    candidate_value_mode = analysis_ok
        ? (prune_selected ? va.numeric_value_mode : va.original_value_mode)
        : PAGE_VALUE_GENERAL;
    candidate_constant_value = analysis_ok
        ? (prune_selected ? va.numeric_constant_value : va.original_constant_value)
        : (MAT_VAL_TYPE)0.0;

    printf("[V628R2] shadow_original_ms=%.6f, shadow_numeric_ms=%.6f, shadow_gain=%.8f, "
           "shadow_original_row_off=%d, shadow_numeric_row_off=%d, "
           "shadow_original_cpu_ms=%.6f, shadow_original_dsp_ms=%.6f, "
           "shadow_numeric_cpu_ms=%.6f, shadow_numeric_dsp_ms=%.6f, "
           "shadow_original_suffix_stored=%lld, shadow_numeric_suffix_stored=%lld, "
           "shadow_ms=%.3f\n",
           shadow.original_ms, shadow.numeric_ms, shadow.gain,
           shadow.original_row_off, shadow.numeric_row_off,
           shadow.original_cpu_ms, shadow.original_dsp_ms,
           shadow.numeric_cpu_ms, shadow.numeric_dsp_ms,
           shadow.original_suffix_stored, shadow.numeric_suffix_stored,
           t_shadow_select);
    printf("[V628R2] prune_selected=%d, pruned_zeros=%lld, applied_prune_ratio=%.8f, "
           "compact_ms=%.3f, candidate_value_mode=%s, candidate_value_mode_id=%d, "
           "candidate_constant_value=%.17g\n",
           prune_selected, pruned_zeros, applied_prune_ratio, t_prune_compact,
           page_value_mode_name(candidate_value_mode), candidate_value_mode,
           (double)candidate_constant_value);
    page_value_analysis_free(&va);

    const double tvsel0 = now_ms();
    selected_value_mode = page_select_cpu_value_mode(
        &csr, candidate_value_mode, candidate_constant_value, cpu_x, ref,
        &value_general_ms, &value_candidate_ms);
    t_value_select = now_ms() - tvsel0;
    printf("[V628R2] selected_value_mode=%s, selected_value_mode_id=%d, "
           "selected_constant_value=%.17g, general_ms=%.6f, candidate_ms=%.6f, "
           "value_select_ms=%.3f, actual_cpu_seq_nz_bytes=%.1f, planner_cpu_seq_nz_bytes=12.0\n",
           page_value_mode_name(selected_value_mode), selected_value_mode,
           (double)csr.constant_value, value_general_ms, value_candidate_ms,
           t_value_select, page_cpu_actual_seq_bytes_per_nnz(&csr));

    const char *dsp_values_env = getenv("PAGE_DSP_VALUES");
    const char *dsp_values = (dsp_values_env && *dsp_values_env) ? dsp_values_env : "general";
    if (strcmp(dsp_values, "auto") == 0) {
        opt.dsp_value_mode = candidate_value_mode;
        opt.dsp_constant_value = candidate_constant_value;
    } else if (strcmp(dsp_values, "general") == 0) {
        opt.dsp_value_mode = PAGE_VALUE_GENERAL;
        opt.dsp_constant_value = (MAT_VAL_TYPE)0.0;
    } else {
        fprintf(stderr, "[DSP_V1] ERROR: PAGE_DSP_VALUES must be general|auto\n");
        return 5;
    }
    printf("[DSP_V1] dsp_values=%s dsp_value_mode=%s dsp_value_mode_id=%d "
           "dsp_constant_value=%.17g dsp_index=%s\n",
           dsp_values, page_value_mode_name(opt.dsp_value_mode), opt.dsp_value_mode,
           (double)opt.dsp_constant_value, opt.force_idx32 ? "32" : "auto");

    int explicit_share = 0;
    {
        const char *e = getenv("PAGE_CPU_SHARE");
        if (e && *e) { opt.cpu_share = atof(e); explicit_share = 1; }
    }

    int force_rowoff_active = 0;
    int force_rowoff_requested = -1;
    {
        const char *e = getenv("PAGE_FORCE_ROW_OFF");
        if (e && *e) {
            char *end = NULL;
            errno = 0;
            long v = strtol(e, &end, 10);
            const int max_legal = (m > SIGMA) ? ((m - SIGMA) / SIGMA) * SIGMA : 0;
            if (errno != 0 || !end || *end != '\0' || v <= 0 || v >= m ||
                v > max_legal || (v % SIGMA) != 0) {
                fprintf(stderr,
                        "[FIXED_CUT] ERROR: PAGE_FORCE_ROW_OFF must be a legal "
                        "nonzero SIGMA-aligned hybrid cut (SIGMA=%d, m=%d, max=%d); got '%s'\n",
                        SIGMA, m, max_legal, e);
                return 5;
            }
            if (explicit_share) {
                fprintf(stderr, "[FIXED_CUT] ERROR: PAGE_FORCE_ROW_OFF cannot be combined with PAGE_CPU_SHARE\n");
                return 5;
            }
            if (v15_rebalance) {
                fprintf(stderr, "[FIXED_CUT] ERROR: PAGE_FORCE_ROW_OFF requires PAGE_CONT_REBALANCE=0\n");
                return 5;
            }
            force_rowoff_active = 1;
            force_rowoff_requested = (int)v;
            opt.cpu_share = -1.0;
            opt.row_off_override = force_rowoff_requested;
            printf("[FIXED_CUT] requested=%d active=1 sigma=%d policy=exact_original_row_boundary\n",
                   force_rowoff_requested, SIGMA);
        }
    }

    double t_tune = 0.0;
    double tune_balance_share = -1.0;
    double tune_best_ms = -1.0;
    int tune_candidates = 0;
    int tune_best_rowoff = -1;
    int tune_best_policy = -1;
    int tune_force_cpu = 0;
    const char *tune_best_route = "model";

    const char *et = getenv("PAGE_AUTOTUNE");

    const int do_tune = !force_exec_mode && !explicit_share && et && strcmp(et, "1") == 0;
    if (force_exec_mode && et && strcmp(et, "1") == 0)
        printf("[FORCE_MODE] autotune_requested=1 ignored=1 reason=fixed_execution_class\n");

    if (do_tune) {
        const double tt0 = now_ms();
        const int wide = (n > GSM_X_CAP);

        page_opts copt = opt;
        copt.cpu_share = -1.0;
        copt.row_off_override = 0;
        copt.plan_policy = 0;
        copt.force_cpu = 0;

        page_stat st0; memset(&st0, 0, sizeof(st0));
        page_matrix A0; double tp0[3] = {0, 0, 0};
        page_build(&csr, &st0, &A0, coreNum, cluster_id, tp0, &copt);
        double p0 = page_pilot_plan(&csr, &A0, cpu_x, x, coreNum, cluster_id);
        const double full_dsp_anchor_ms = p0;
        tune_candidates++;
        tune_best_ms = p0; tune_best_rowoff = 0; tune_best_policy = 0;
        tune_best_route = "tiled";
        printf("[TUNE] candidate row_off=%d share=%.6f route=%s pilot_ms=%.4f\n",
               A0.row_off, A0.cpu_share, page_policy_name(0, A0.has_window), p0);

        if (wide) {
            page_opts wopt = copt; wopt.plan_policy = 1;
            page_stat sw; memset(&sw, 0, sizeof(sw));
            page_matrix Aw; double tpw[3] = {0, 0, 0};
            page_build(&csr, &sw, &Aw, coreNum, cluster_id, tpw, &wopt);
            if (Aw.has_window) {
                const double pw = page_pilot_plan(&csr, &Aw, cpu_x, x, coreNum, cluster_id);
                tune_candidates++;
                printf("[TUNE] candidate row_off=%d share=%.6f route=window pilot_ms=%.4f\n",
                       Aw.row_off, Aw.cpu_share, pw);
                if (pw < tune_best_ms) {
                    tune_best_ms = pw; tune_best_rowoff = 0;
                    tune_best_policy = 1; tune_best_route = "window";
                }
            }
            page_release_candidate(&Aw, &sw);
        }

        double dslow = 1.0, aslow = 1.0;
        if (opt.bw_pipe_mix > 0.0 && opt.bw_pipe > 0.0)
            dslow = opt.bw_pipe / opt.bw_pipe_mix;
        if (opt.bw_arm_mix > 0.0 && opt.bw_arm > 0.0)
            aslow = opt.bw_arm / opt.bw_arm_mix;
        if (dslow < 1.0) dslow = 1.0;
        if (aslow < 1.0) aslow = 1.0;
        const double td_bal = full_dsp_anchor_ms * dslow;
        const double ta_bal = (t_arm_calib > 0.0 ? t_arm_calib : 1.0e300) * aslow;
        if (isfinite(td_bal) && isfinite(ta_bal) && td_bal > 0.0 && ta_bal > 0.0)
            tune_balance_share = td_bal / (td_bal + ta_bal);
        else
            tune_balance_share = 0.0;
        if (tune_balance_share < 0.0) tune_balance_share = 0.0;
        if (tune_balance_share > 1.0) tune_balance_share = 1.0;
        printf("[TUNE] balance_share=%.6f full_dsp_ms=%.4f arm_full_ms=%.4f dslow=%.4f aslow=%.4f\n",
               tune_balance_share, full_dsp_anchor_ms, t_arm_calib, dslow, aslow);

        int rows[5]; int nrows = 0;
        rows[nrows++] = 0;
        for (int q = 1; q <= 4; q++) {
            const double share = tune_balance_share * (double)q / 4.0;
            const int ro = page_share_to_rowoff(&st0, m, nnz, share);
            int dup = 0;
            for (int j = 0; j < nrows; j++) if (rows[j] == ro) { dup = 1; break; }
            if (!dup) rows[nrows++] = ro;
        }
        page_release_candidate(&A0, &st0);

        for (int ir = 1; ir < nrows; ir++) {
            const int ro = rows[ir];
            for (int policy = 0; policy <= (wide ? 1 : 0); policy++) {
                page_opts po = opt;
                po.cpu_share = -1.0;
                po.row_off_override = ro;
                po.plan_policy = policy;
                po.force_cpu = 0;
                page_stat ps; memset(&ps, 0, sizeof(ps));
                page_matrix PA; double tpp[3] = {0, 0, 0};
                page_build(&csr, &ps, &PA, coreNum, cluster_id, tpp, &po);
                if (policy == 1 && !PA.has_window) {
                    page_release_candidate(&PA, &ps);
                    continue;
                }
                const double pm = page_pilot_plan(&csr, &PA, cpu_x, x, coreNum, cluster_id);
                tune_candidates++;
                printf("[TUNE] candidate row_off=%d share=%.6f route=%s pilot_ms=%.4f\n",
                       PA.row_off, PA.cpu_share,
                       page_policy_name(policy, PA.has_window), pm);
                if (pm < tune_best_ms) {
                    tune_best_ms = pm; tune_best_rowoff = ro;
                    tune_best_policy = policy;
                    tune_best_route = PA.has_window ? "window" : "tiled";
                }
                page_release_candidate(&PA, &ps);
            }
        }

        if (t_arm_calib > 0.0) {
            tune_candidates++;
            printf("[TUNE] candidate row_off=%d share=1.000000 route=cpu pilot_ms=%.4f\n",
                   m, t_arm_calib);
            if (t_arm_calib < tune_best_ms) {
                tune_best_ms = t_arm_calib;
                tune_best_rowoff = 0; tune_best_policy = -1;
                tune_best_route = "cpu"; tune_force_cpu = 1;
            }
        }
        t_tune = now_ms() - tt0;
        printf("[TUNE] chosen row_off=%d route=%s pilot_ms=%.4f candidates=%d\n",
               tune_best_rowoff, tune_best_route, tune_best_ms, tune_candidates);
        printf("[T] autotune_time: %.3f (ms)\n", t_tune);

        opt.cpu_share = -1.0;
        opt.row_off_override = tune_best_rowoff;
        opt.plan_policy = tune_best_policy;
        opt.force_cpu = tune_force_cpu;
    }

    page_stat st; memset(&st, 0, sizeof(st));
    page_matrix A;
    double tpre[3] = {0, 0, 0};

    double t_final_build = 0.0;
    double t_v18_shadow_plan = 0.0;
    double t_v18_materialize = 0.0;
    double t_prefix_calib = 0.0;
    int prefix_calib_row = 0;
    double prefix_arm_solo_ms = 0.0, prefix_arm_mix_ms = 0.0;
    double prefix_arm_mix_bw = 0.0, prefix_mad = 0.0, prefix_rel_sigma = 0.0;
    double prefix_fused_bw = 0.0, prefix_fused_rel_sigma = 0.0;
    double t_mode_verify = 0.0;
    int prefix_candidate_mode = -1, mode_switch_verified = 0, mode_switch_accept = 0;
    double verify_current_ms = 0.0, verify_cpu_ms = 0.0;
    const char *prefix_env = getenv("PAGE_PREFIX_CALIB");
    const int do_prefix_calib = !explicit_share && !do_tune && !(prefix_env && strcmp(prefix_env, "0") == 0);
    const int v18_shadow_fastpath = v18_fast_prep && force_exec_mode &&
                                    strcmp(exec_mode, "hybrid") == 0 && do_prefix_calib;
    double tb0 = now_ms();
    if (v18_shadow_fastpath) {
        page_build_shadow(&csr, &st, &A, coreNum, cluster_id, tpre, &opt);
        t_v18_shadow_plan += now_ms() - tb0;
    } else {
        page_build(&csr, &st, &A, coreNum, cluster_id, tpre, &opt);
        t_final_build += now_ms() - tb0;
    }

    if (do_prefix_calib && A.mode != PAGE_MODE_CPU && A.row_off > 0) {
        prefix_calib_row = A.row_off;
        const double tp0 = now_ms();
        prefix_arm_solo_ms = calib_arm_prefix_ms(&csr, cpu_x, ref, prefix_calib_row,
                                                  &prefix_mad, &prefix_rel_sigma);
        t_prefix_calib = now_ms() - tp0;
        double contention = 1.0;
        if (t_arm_calib > 0.0 && t_arm_mix_full > 0.0) contention = t_arm_mix_full / t_arm_calib;
        if (!(contention > 0.0) || !isfinite(contention)) contention = 1.0;
        const double contention_rel_sigma = sqrt(arm_calib_rel_sigma * arm_calib_rel_sigma
                                                + mix_arm_rel_sigma * mix_arm_rel_sigma);
        prefix_fused_rel_sigma = sqrt(prefix_rel_sigma * prefix_rel_sigma
                                     + contention_rel_sigma * contention_rel_sigma);
        prefix_arm_mix_ms = prefix_arm_solo_ms * contention;
        const double arm_bytes = page_prefix_model_bytes(&csr, prefix_calib_row);
        if (arm_bytes > 0.0 && prefix_arm_mix_ms > 0.0)
            prefix_arm_mix_bw = arm_bytes / (prefix_arm_mix_ms * 1e-3);
        prefix_fused_bw = page_fuse_bw(opt.bw_arm_mix, mix_arm_rel_sigma,
                                        prefix_arm_mix_bw, prefix_fused_rel_sigma);
        printf("[CAL] prefix_row = %d, arm_solo_ms = %.4f, contention = %.4f, arm_mix_ms = %.4f, arm_mix_bw = %.2f (GB/s)\n",
               prefix_calib_row, prefix_arm_solo_ms, contention, prefix_arm_mix_ms, prefix_arm_mix_bw / 1e9);
        printf("[STABLE_CAL] prefix_samples=%d prefix_median_ms=%.6f prefix_mad_ms=%.6f "
               "prefix_rel_sigma=%.8f base_arm_mix_bw=%.6f prefix_arm_mix_bw=%.6f "
               "fused_arm_mix_bw=%.6f fused_rel_sigma=%.8f\n",
               PAGE_PREFIX_CALIB_ITER, prefix_arm_solo_ms, prefix_mad, prefix_rel_sigma,
               opt.bw_arm_mix / 1e9, prefix_arm_mix_bw / 1e9, prefix_fused_bw / 1e9,
               prefix_fused_rel_sigma);

        if (prefix_fused_bw > PAGE_BW_ARM_MIN && prefix_fused_bw < PAGE_BW_ARM_MAX) {
            page_opts copt = opt; copt.bw_arm_mix = prefix_fused_bw;
            page_stat cst; memset(&cst, 0, sizeof(cst));
            page_matrix CA; memset(&CA, 0, sizeof(CA));
            double ctp[3] = {0,0,0};
            const double v18cs0 = now_ms();
            page_build_shadow(&csr, &cst, &CA, coreNum, cluster_id, ctp, &copt);
            if (v18_shadow_fastpath) t_v18_shadow_plan += now_ms() - v18cs0;
            prefix_candidate_mode = CA.mode;
            int accept_correction = 1;
            if ((A.mode == PAGE_MODE_CPU) != (CA.mode == PAGE_MODE_CPU)) {
                const double tv0 = now_ms();
                mode_switch_verified = 1;
                verify_current_ms = page_pilot_plan(&csr, &A, cpu_x, x, coreNum, cluster_id);
                page_matrix cpu_probe; memset(&cpu_probe, 0, sizeof(cpu_probe));
                cpu_probe.mode = PAGE_MODE_CPU;
                verify_cpu_ms = page_pilot_plan(&csr, &cpu_probe, cpu_x, x, coreNum, cluster_id);
                if (CA.mode == PAGE_MODE_CPU)
                    accept_correction = verify_cpu_ms < verify_current_ms;
                else
                    accept_correction = verify_current_ms >= verify_cpu_ms ? 0 : 1;
                mode_switch_accept = accept_correction;
                t_mode_verify += now_ms() - tv0;
            }
            if (accept_correction) {
                opt.bw_arm_mix = prefix_fused_bw;
                if (v18_shadow_fastpath) {
                    page_release_candidate(&A, &st);
                    A = CA; st = cst;
                    memcpy(tpre, ctp, sizeof(ctp));
                    memset(&CA, 0, sizeof(CA));
                    memset(&cst, 0, sizeof(cst));
                } else {
                    page_release_candidate(&CA, &cst);
                    page_release_candidate(&A, &st);
                    memset(&st, 0, sizeof(st));
                    tb0 = now_ms();
                    page_build(&csr, &st, &A, coreNum, cluster_id, tpre, &opt);
                    t_final_build += now_ms() - tb0;
                }
            } else {
                page_release_candidate(&CA, &cst);
            }
        }
    }

    if (v18_shadow_fastpath) {
        t_v18_materialize = page_v18_materialize_shadow_plan(
            &csr, &st, &A, cluster_id, &opt, &tpre[2]);
        t_final_build += t_v18_materialize;
        printf("[V18_FAST_PREP] shadow_then_materialize=1 shadow_plan_ms=%.6f materialize_ms=%.6f row_off=%d\n",
               t_v18_shadow_plan, t_v18_materialize, A.row_off);
    }

    if (!force_exec_mode && !do_tune && !mode_switch_verified && A.mode != PAGE_MODE_CPU) {
        const double cpu_exec_est = (selected_value_mode != PAGE_VALUE_GENERAL && value_candidate_ms > 0.0)
                                  ? value_candidate_ms : t_arm_calib;
        const double rel_u = sqrt(arm_calib_rel_sigma * arm_calib_rel_sigma
                                + mix_arm_rel_sigma * mix_arm_rel_sigma
                                + prefix_fused_rel_sigma * prefix_fused_rel_sigma);
        const double scale_u = fmax(cpu_exec_est, A.model_total_ms);
        const double diff = fabs(cpu_exec_est - A.model_total_ms);
        if (cpu_exec_est > 0.0 && A.model_total_ms > 0.0 && diff <= scale_u * rel_u) {
            const double tv0 = now_ms();
            page_matrix cpu_probe; memset(&cpu_probe, 0, sizeof(cpu_probe));
            cpu_probe.mode = PAGE_MODE_CPU;
            const double hybrid_ms = page_pilot_plan(&csr, &A, cpu_x, x, coreNum, cluster_id);
            const double cpu_ms = page_pilot_plan(&csr, &cpu_probe, cpu_x, x, coreNum, cluster_id);
            t_mode_verify += now_ms() - tv0;
            printf("[STABLE_CAL] uncertainty_verify=1 cpu_est_ms=%.6f hybrid_model_ms=%.6f "
                   "rel_uncertainty=%.8f hybrid_pilot_ms=%.6f cpu_pilot_ms=%.6f\n",
                   cpu_exec_est, A.model_total_ms, rel_u, hybrid_ms, cpu_ms);
            if (cpu_ms < hybrid_ms) {
                page_release_candidate(&A, &st);
                memset(&st, 0, sizeof(st));
                page_opts cpuopt = opt; cpuopt.force_cpu = 1;
                tb0 = now_ms();
                page_build(&csr, &st, &A, coreNum, cluster_id, tpre, &cpuopt);
                t_final_build += now_ms() - tb0;
            }
        }
    }
    if (v15_rebalance) {
        if (page_v15_feedback_rebalance(&csr,&st,&A,&opt,cpu_x,x,coreNum,cluster_id,tpre,&t_v15_rebalance)!=0) {
            fprintf(stderr,"[V15_REBAL] ERROR: feedback repartition failed\n");
            return 5;
        }
    }

    if (force_rowoff_active) {
        const long long fixed_cpu_nnz = (A.row_off > 0 && st.row_pre) ? st.row_pre[A.row_off] : 0;
        const int exact_ok = (A.row_off == force_rowoff_requested);
        printf("[FIXED_CUT] final requested=%d actual=%d cpu_prefix_nnz=%lld dsp_nnz=%lld "
               "has_window=%d res_nnz=%lld exact=%d\n",
               force_rowoff_requested, A.row_off, fixed_cpu_nnz, A.nnz_dsp,
               A.has_window, A.res_nnz, exact_ok);
        if (!exact_ok) {
            fprintf(stderr, "[FIXED_CUT] ERROR: exact cut was not realized\n");
            return 4;
        }
    }

    if (force_exec_mode) {
        const long long cpu_prefix_nnz = (A.row_off > 0 && st.row_pre) ? st.row_pre[A.row_off] : 0;
        int mode_valid = 0;
        if (strcmp(exec_mode, "cpu") == 0) {
            mode_valid = (A.mode == PAGE_MODE_CPU && A.m_dsp == 0);
        } else if (strcmp(exec_mode, "hybrid") == 0) {
            mode_valid = (A.mode != PAGE_MODE_CPU && A.nnz_dsp > 0
                       && (cpu_prefix_nnz > 0 || A.res_nnz > 0));
        } else if (strcmp(exec_mode, "dsp") == 0) {
            mode_valid = (A.mode != PAGE_MODE_CPU && A.row_off == 0
                       && A.nnz_dsp > 0 && !A.has_window && A.res_nnz == 0);
        }
        printf("[FORCE_MODE] final requested=%s actual=%s row_off=%d m_dsp=%d cpu_prefix_nnz=%lld dsp_nnz=%lld has_window=%d res_nnz=%lld cpu_share=%.8f valid=%d\n",
               exec_mode, page_mode_name(A.mode), A.row_off, A.m_dsp,
               cpu_prefix_nnz, A.nnz_dsp, A.has_window, A.res_nnz,
               A.cpu_share, mode_valid);
        if (!mode_valid) {
            fprintf(stderr, "[FORCE_MODE] ERROR: requested execution class was not realized\n");
            return 4;
        }
    }
    int hybrid_v1_threads = omp_threads_base;
    double hybrid_v1_selected_ms = -1.0, hybrid_v1_base_ms = -1.0, hybrid_v1_calib_ms = 0.0;
    if (page_apply_hybrid_thread_policy(&csr, &A, cpu_x, x, coreNum, cluster_id,
                                         omp_threads_base, &hybrid_v1_threads,
                                         &hybrid_v1_selected_ms, &hybrid_v1_base_ms,
                                         &hybrid_v1_calib_ms) != 0) {
        return 5;
    }

    printf("[STABLE_CAL] prefix_candidate_mode=%d mode_switch_verified=%d mode_switch_accept=%d "
           "verify_current_ms=%.6f verify_cpu_ms=%.6f mode_verify_ms=%.3f final_mode=%d\n",
           prefix_candidate_mode, mode_switch_verified, mode_switch_accept,
           verify_current_ms, verify_cpu_ms, t_mode_verify, A.mode);
    printf("[HYBRID_V1] formal_threads=%d base_threads=%d selected_pilot_ms=%.6f "
           "base_pilot_ms=%.6f calibration_ms=%.3f\n",
           hybrid_v1_threads, omp_threads_base, hybrid_v1_selected_ms,
           hybrid_v1_base_ms, hybrid_v1_calib_ms);
    printf("[T] hybrid_thread_calib_time: %.3f (ms)\n", hybrid_v1_calib_ms);

    /* Setup time excludes matrix I/O and device initialization. */
    const double t_numa_first_touch = t_numa_csr_first_touch + t_numa_x_first_touch;
    const double t_pre = t_numeric_scan + t_shadow_select + t_prune_compact + t_value_select
                       + t_platform_calib + t_matrix_calib + t_tune
                       + t_prefix_calib + t_mode_verify + t_final_build + t_v18_shadow_plan
                       + t_numa_first_touch + t_dual_x_prod_setup + t_v17_csr_numa_setup
                       + t_v15_rebalance + hybrid_v1_calib_ms;
    printf("[T] numa_placement_time: %.3f (ms)\n", t_numa_first_touch);
    printf("[T] dual_x_prod_setup_time: %.3f (ms)\n", t_dual_x_prod_setup);
    printf("[T] v17_csr_numa_setup_time: %.3f (ms)\n", t_v17_csr_numa_setup);
    printf("[T] v15_rebalance_time: %.3f (ms)\n", t_v15_rebalance);
    printf("[T] numa_first_touch_time: %.3f (ms)\n", t_numa_first_touch);
    printf("[T] prefix_calib_time: %.3f (ms)\n", t_prefix_calib);
    printf("[T] stable_cal_extra_time: %.3f (ms)\n", t_mode_verify);
    printf("[PAGE] prefix_calib_row = %d, prefix_arm_solo_ms = %.4f, prefix_arm_mix_ms = %.4f, prefix_arm_mix_bw = %.2f (GB/s)\n",
           prefix_calib_row, prefix_arm_solo_ms, prefix_arm_mix_ms, prefix_arm_mix_bw / 1e9);
    printf("[T] final_build_time: %.3f (ms)\n", t_final_build);
    printf("[T] v18_shadow_plan_time: %.3f (ms)\n", t_v18_shadow_plan);
    printf("[T] v18_materialize_time: %.3f (ms)\n", t_v18_materialize);
    printf("[DUAL_X_PROD] planner_input enabled=%d cpu_x=%s dsp_x=hthread cpu_csr=%s "
           "arm_bw_gbs=%.6f arm_mix_bw_gbs=%.6f pipe_mix_bw_gbs=%.6f "
           "arm_calib_ms=%.6f arm_mix_ms=%.6f\n",
           dual_x_prod, dual_x_prod ? "nohuge_interleave" : "shared_hthread",
           v17_csr_numa ? "private_nohuge_interleave" : "default",
           opt.bw_arm / 1e9, opt.bw_arm_mix / 1e9, opt.bw_pipe_mix / 1e9,
           t_arm_calib, t_arm_mix_full);
    printf("[PAGE] filename = %s, m = %d, n = %d, nnz = %d, cv = %.6f\n",
           name, m, n, nnz, st.cv_global);
    printf("[PAGE] v628r2_input_nnz = %d, v628r2_numeric_nnz_available = %lld, "
           "v628r2_final_nnz = %d, v628r2_explicit_zeros_found = %lld, "
           "v628r2_prune_selected = %d, v628r2_pruned_zeros = %lld, "
           "v628r2_applied_prune_ratio = %.8f, v628r2_shadow_gain = %.8f, "
           "v628r2_value_mode = %d, v628r2_constant_value = %.17g\n",
           input_nnz, numeric_nnz_available, nnz, explicit_zeros_found,
           prune_selected, pruned_zeros, applied_prune_ratio, shadow.gain,
           csr.value_mode, (double)csr.constant_value);
    printf("[PAGE] mode = %s, nnz_per_row = %.2f, bandw_ratio = %.5f, "
           "dof_hit = %.2f, has_perm = %d\n",
           page_mode_name(A.mode), st.nnz_per_row, st.bandwidth_ratio,
           st.dof_hit, A.has_perm);
    if (A.mode != PAGE_MODE_CPU) {
        const int nsb = (A.nvrow + SIGMA - 1) / SIGMA;
        printf("[PAGE] perm_blocks = %d, sigma_blocks = %d, perm_block_ratio = %.4f\n",
               A.nperm_blk, nsb, nsb > 0 ? (double)A.nperm_blk / (double)nsb : 0.0);
        printf("[PAGE] perm_saved_elems = %lld, perm_est_save_ms = %.4f, "
               "perm_est_finalize_ms = %.4f, perm_guard_drop = %d\n",
               A.perm_saved_elems, A.perm_est_save_ms,
               A.perm_est_finalize_ms, A.perm_guard_drop);
    }
    printf("[T] pass_a_time: %.3f (ms)\n", tpre[0] * 1e3);
    printf("[T] plan_time: %.3f (ms)\n",   tpre[1] * 1e3);
    printf("[T] pass_b_time: %.3f (ms)\n", tpre[2] * 1e3);
    printf("[T] pre_total_time: %.3f (ms)\n", t_pre);
    printf("[PAGE] est_dsp_ms = %.4f, est_arm_ms = %.4f, "
           "bw_model_ddr = %.2f (GB/s), bw_model_gsm = %.2f (GB/s), "
           "bw_model_arm = %.2f (GB/s), bw_model_mix = %.2f (GB/s)\n",
           A.est_dsp_ms, A.est_arm_ms,
           opt.bw_ddr / 1e9, opt.bw_gsm / 1e9, opt.bw_arm / 1e9, opt.bw_mix / 1e9);
    printf("[PAGE] bw_model_pipeline = %.2f (GB/s), bw_model_arm_mix = %.2f (GB/s), "
           "bw_model_pipeline_mix = %.2f (GB/s), launch_ms = %.6f, barrier_ms = %.6f\n",
           opt.bw_pipe / 1e9, opt.bw_arm_mix / 1e9, opt.bw_pipe_mix / 1e9,
           opt.launch_ms, opt.barrier_ms);
    printf("[PAGE] model_total_ms = %.4f, model_share_ms = %.4f, "
           "model_tiled_ms = %.4f, model_window_ms = %.4f\n",
           A.model_total_ms, A.model_share_ms, A.model_tiled_ms, A.model_window_ms);
    printf("[PAGE] replan_predict_row = %d, replan_correct_row = %d, "
           "replan_rounds = %d, replan_predict_ms = %.4f, replan_correct_ms = %.4f\n",
           A.replan_predict_row, A.replan_correct_row, A.replan_rounds,
           A.replan_predict_ms, A.replan_correct_ms);

    if (A.mode != PAGE_MODE_CPU) {
        printf("[PAGE] npanel = %d, ntile = %d, ninst = %d, stored = %lld\n",
               A.npanel, A.ntile, A.ninst, A.stored);
        printf("[PAGE] Z2NZ = %.2f%%, zeroPadding = %lld, lb_imbalance = %.4f, "
               "bytes_per_nnz = %.2f\n",
               A.z_ratio * 100.0, A.pad_zeros, A.lb_imbalance, A.bytes_per_nnz);
        printf("[PAGE] nvrow = %d, row_off = %d, m_dsp = %d, n_ov = %d, "
               "row_cap = %d, has_split = %d, cpu_share = %.4f\n",
               A.nvrow, A.row_off, A.m_dsp, A.n_ov, A.row_cap,
               A.has_split, A.cpu_share);
        printf("[PAGE] x_elems = %lld, x_bytes_per_iter = %.0f, "
               "x_over_val = %.4f\n",
               A.x_elems, (double)A.x_elems * 8.0,
               (A.stored > 0)
                 ? (double)A.x_elems * 8.0
                   / ((double)A.stored * (8.0 + (double)A.idx_bytes)) : 0.0);
        printf("[DSP_V1] final_dsp_value_mode=%s final_dsp_value_mode_id=%d final_dsp_constant=%.17g idx_bytes=%d\n",
               page_value_mode_name(A.dsp_value_mode), A.dsp_value_mode,
               (double)A.dsp_constant_value, A.idx_bytes);
        printf("[PAGE] packed_tiles = %d, packed_x_elems = %lld, idx_bytes = %d, "
               "xmap_bytes = %.0f, packed_runs = %d, packed_dma_rounds = %d\n",
               A.packed_tiles, A.packed_x_elems, A.idx_bytes,
               (double)(A.packed_x_elems / PAGE_XBLOCK_ELEMS) * 4.0,
               A.packed_runs, A.packed_dma_rounds);
        printf("[PAGE] has_window = %d, cw_lo = %d, cw_hi = %d, cw_width = %d, "
               "res_nnz = %lld, res_ratio = %.4f\n",
               A.has_window, A.cw_lo, A.cw_hi, A.cw_hi - A.cw_lo, A.res_nnz,
               (A.nnz > 0) ? (double)A.res_nnz / (double)A.nnz : 0.0);
    }

    MAT_VAL_TYPE *ybuf = NULL, *yd = NULL, *y = NULL, *y_res = NULL;
    MAT_VAL_TYPE *diag_cpu_full_y = NULL;
    MAT_VAL_TYPE *diag_cpu_private_y = NULL;
    MAT_VAL_TYPE *diag_dual_y = NULL;
    MAT_VAL_TYPE *diag_cpu_interleave_y = NULL;
    MAT_VAL_TYPE *diag_interleave_y = NULL;
    MAT_VAL_TYPE *diag_cpu_hthread_y = NULL;
    MAT_VAL_TYPE *diag_hthread_y = NULL;
    MAT_VAL_TYPE *diag_cpu_nohuge_y = NULL;
    MAT_VAL_TYPE *diag_nohuge_y = NULL;
    MAT_VAL_TYPE *x_cpu_diag = NULL;
    MAT_VAL_TYPE *x_hthread_private_diag = NULL;
    page_xplace_mapping x_interleave_diag;
    page_xplace_mapping x_nohuge_diag;
    memset(&x_interleave_diag, 0, sizeof(x_interleave_diag));
    memset(&x_nohuge_diag, 0, sizeof(x_nohuge_diag));
    int dual_diag_bad = 0;
    int xplace_diag_bad = 0;
    int htx_diag_bad = 0;
    int thp_diag_bad = 0;
    size_t yd_elems = 0;

    if (A.mode == PAGE_MODE_CPU) {
        if (cpu_hybrid_side) {
            ybuf = (MAT_VAL_TYPE *)hthread_malloc(cluster_id,
                      sizeof(MAT_VAL_TYPE) * (size_t)m, HT_MEM_RW);
            if (!ybuf) {
                fprintf(stderr, "[V14_CPU_HYBRIDSIDE] ERROR: hthread y allocation failed\n");
                return 3;
            }
            y = ybuf;
            memset(ybuf, 0, sizeof(MAT_VAL_TYPE) * (size_t)m);
            printf("[V14_CPU_HYBRIDSIDE] y_init=hthread_malloc_plus_serial_memset dsp_launch=0\n");
        } else {
            y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            if (g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR && y) {
                memset(y, 0, sizeof(MAT_VAL_TYPE) * (size_t)m);
                printf("[HALAV_CSR_CPU] y_init=malloc_plus_serial_memset\n");
            }
        }
    } else {
        yd_elems = (size_t)A.nvrow + SROW;
        const size_t cap = (size_t)A.row_off + yd_elems;
        ybuf = (MAT_VAL_TYPE *)hthread_malloc(cluster_id,
                  sizeof(MAT_VAL_TYPE) * cap, HT_MEM_RW);
        yd = ybuf + A.row_off;
        y  = ybuf;
        memset(ybuf, 0, sizeof(MAT_VAL_TYPE) * cap);
    }

    if (A.mode != PAGE_MODE_CPU && A.has_window && A.res_nnz > 0)
        y_res = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)A.m_dsp);

    if (same_part_diag) {
        diag_cpu_full_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
        if (!diag_cpu_full_y) {
            fprintf(stderr, "[DIAG] ERROR: CPU-full diagnostic buffer allocation failed\n");
            return 3;
        }
    }

    double iter_ms[SPMV_ITER];
    double dsp_ms[SPMV_ITER] = {0.0}, perm_ms[SPMV_ITER] = {0.0};
    double arm_rows_ms[SPMV_ITER] = {0.0}, arm_res_ms[SPMV_ITER] = {0.0};

    double submit_ms[SPMV_ITER] = {0.0}, wait_after_arm_ms[SPMV_ITER] = {0.0};
    double cpu_iter_ms[SPMV_ITER] = {0.0};
    double arm_part_ms[SPMV_ITER] = {0.0};

    double diag_cpu_full_iter[SPMV_ITER] = {0.0};
    double diag_cpu_part_solo_iter[SPMV_ITER] = {0.0};
    double diag_cpu_rows_solo_iter[SPMV_ITER] = {0.0};
    double diag_cpu_res_solo_iter[SPMV_ITER] = {0.0};
    double diag_dsp_part_solo_iter[SPMV_ITER] = {0.0};
    double diag_dsp_submit_solo_iter[SPMV_ITER] = {0.0};
    double diag_dsp_wait_solo_iter[SPMV_ITER] = {0.0};

    double t_dsp = 0.0, t_perm = 0.0;
    double t_arm_rows = 0.0, t_arm_res = 0.0, t_arm_part = 0.0;
    double t_submit = 0.0, t_wait_after_arm = 0.0;
    double diag_cpu_full = 0.0, diag_cpu_part_solo = 0.0;
    double diag_cpu_rows_solo = 0.0, diag_cpu_res_solo = 0.0;
    double diag_dsp_part_solo = 0.0, diag_dsp_submit_solo = 0.0, diag_dsp_wait_solo = 0.0;
    printf("[PAGE] warmup_iterations = %d, timed_iterations = %d\n",
           SPMV_WARMUP, SPMV_ITER);

    if (A.mode == PAGE_MODE_CPU) {

        for (int i = 0; i < SPMV_WARMUP; i++) page_cpu_exec_full(&csr, cpu_x, y);
        for (int i = 0; i < SPMV_ITER; i++) {
            const double ta = now_ms();
            page_cpu_exec_full(&csr, cpu_x, y);
            iter_ms[i] = now_ms() - ta;
            cpu_iter_ms[i] = iter_ms[i];
        }
        printf("[PAGE] kernel = %s\n",
               g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR
                   ? "halav_csr_mv" : "page_cpu_spmv_overt");
        if (cpu_hybrid_side)
            printf("[V14_CPU_HYBRIDSIDE] formal_full_cpu=1 private_x=1 hthread_y=1 dsp_launch=0\n");
    } else {

        const int thread_id  = hthread_group_create(cluster_id, coreNum);
        const int barrier_id = hthread_barrier_create(cluster_id);

        MAT_VAL_TYPE *dsp_const = NULL;
        if (A.dsp_value_mode == PAGE_VALUE_CONSTANT) {
            dsp_const = (MAT_VAL_TYPE *)hthread_malloc(cluster_id, sizeof(MAT_VAL_TYPE), HT_MEM_RW);
            if (!dsp_const) {
                fprintf(stderr, "[DSP_V1] constant buffer allocation failed\n");
                return 3;
            }
            dsp_const[0] = A.dsp_constant_value;
        }

        unsigned long args[32];
        args[0]  = (unsigned long)A.nvrow;
        args[1]  = (unsigned long)A.n;
        args[2]  = (unsigned long)coreNum;
        args[3]  = (unsigned long)A.npanel;
        args[4]  = (unsigned long)barrier_id;
        args[5]  = (unsigned long)A.ntile;
        args[6]  = (unsigned long)A.idx_bytes;
        args[7]  = (unsigned long)A.dsp_value_mode;
        args[8]  = (unsigned long)A.panel_slice_lo;
        args[9]  = (unsigned long)A.panel_tile_lo;
        args[10] = (unsigned long)A.tile_xlo;
        args[11] = (unsigned long)A.tile_xhi;
        args[12] = (unsigned long)A.tile_inst_lo;
        args[13] = (unsigned long)A.tile_packed;
        args[14] = (unsigned long)A.tile_xmap_lo;
        args[15] = (unsigned long)A.tile_xmap;
        args[16] = (unsigned long)A.cl;
        args[17] = (unsigned long)A.cs;
        args[18] = (unsigned long)A.val;
        args[19] = (unsigned long)A.cidx;
        args[20] = (unsigned long)A.thread_ptr;
        args[21] = (unsigned long)A.gsmxwidth;
        args[22] = (unsigned long)x;
        args[23] = (unsigned long)yd;
        args[24] = (unsigned long)dsp_const;
        const int scalArgs = 8, vecArgs = 17;

        memset(yd, 0, sizeof(MAT_VAL_TYPE) * yd_elems);

        if (same_part_diag) {
            for (int i = 0; i < SPMV_WARMUP; i++)
                page_cpu_exec_full(&csr, cpu_x, diag_cpu_full_y);
            for (int i = 0; i < SPMV_ITER; i++) {
                const double da = now_ms();
                page_cpu_exec_full(&csr, cpu_x, diag_cpu_full_y);
                diag_cpu_full_iter[i] = now_ms() - da;
            }

            for (int i = 0; i < SPMV_WARMUP; i++) {
                page_cpu_exec_rows(&csr, cpu_x, y, 0, A.row_off);
                page_cpu_exec_residual(&A, cpu_x, y_res);
            }
            for (int i = 0; i < SPMV_ITER; i++) {
                const double da = now_ms();
                page_cpu_exec_rows(&csr, cpu_x, y, 0, A.row_off);
                const double db = now_ms();
                page_cpu_exec_residual(&A, cpu_x, y_res);
                const double dc = now_ms();
                diag_cpu_rows_solo_iter[i] = db - da;
                diag_cpu_res_solo_iter[i] = dc - db;
                diag_cpu_part_solo_iter[i] = dc - da;
            }

            for (int i = 0; i < SPMV_WARMUP; i++) {
                hthread_group_exec(thread_id, "page_spmv", scalArgs, vecArgs, args);
                hthread_group_wait(thread_id);
            }
            for (int i = 0; i < SPMV_ITER; i++) {
                const double da = now_ms();
                hthread_group_exec(thread_id, "page_spmv", scalArgs, vecArgs, args);
                const double db = now_ms();
                hthread_group_wait(thread_id);
                const double dc = now_ms();
                diag_dsp_submit_solo_iter[i] = db - da;
                diag_dsp_wait_solo_iter[i] = dc - db;
                diag_dsp_part_solo_iter[i] = dc - da;
            }

            diag_cpu_full = trimmed_mean_1each(diag_cpu_full_iter, SPMV_ITER);
            diag_cpu_part_solo = trimmed_mean_1each(diag_cpu_part_solo_iter, SPMV_ITER);
            diag_cpu_rows_solo = trimmed_mean_1each(diag_cpu_rows_solo_iter, SPMV_ITER);
            diag_cpu_res_solo = trimmed_mean_1each(diag_cpu_res_solo_iter, SPMV_ITER);
            diag_dsp_part_solo = trimmed_mean_1each(diag_dsp_part_solo_iter, SPMV_ITER);
            diag_dsp_submit_solo = trimmed_mean_1each(diag_dsp_submit_solo_iter, SPMV_ITER);
            diag_dsp_wait_solo = trimmed_mean_1each(diag_dsp_wait_solo_iter, SPMV_ITER);

            printf("[DIAG] phase=cpu_full ms=%.6f\n", diag_cpu_full);
            printf("[DIAG] phase=cpu_part_solo ms=%.6f rows_ms=%.6f residual_ms=%.6f\n",
                   diag_cpu_part_solo, diag_cpu_rows_solo, diag_cpu_res_solo);
            printf("[DIAG] phase=dsp_part_solo ms=%.6f submit_ms=%.6f wait_ms=%.6f\n",
                   diag_dsp_part_solo, diag_dsp_submit_solo, diag_dsp_wait_solo);
        }

        for (int i = 0; i < SPMV_WARMUP; i++) {
            hthread_group_exec(thread_id, "page_spmv", scalArgs, vecArgs, args);
            page_cpu_exec_rows(&csr, cpu_x, y, 0, A.row_off);
            page_cpu_exec_residual(&A, cpu_x, y_res);
            hthread_group_wait(thread_id);
            page_finalize_y(&A, yd, y_res, y);
        }
        for (int i = 0; i < SPMV_ITER; i++) {
            const double ta = now_ms();

            hthread_group_exec(thread_id, "page_spmv", scalArgs, vecArgs, args);
            const double ar0 = now_ms();
            page_cpu_exec_rows(&csr, cpu_x, y, 0, A.row_off);
            const double ar1 = now_ms();
            page_cpu_exec_residual(&A, cpu_x, y_res);
            const double ar2 = now_ms();
            hthread_group_wait(thread_id);
            const double tb = now_ms();
            page_finalize_y(&A, yd, y_res, y);
            const double tc = now_ms();
            submit_ms[i]         = ar0 - ta;
            arm_rows_ms[i]       = ar1 - ar0;
            arm_res_ms[i]        = ar2 - ar1;
            arm_part_ms[i]       = ar2 - ar0;
            wait_after_arm_ms[i] = tb - ar2;
            dsp_ms[i]            = tb - ta;
            perm_ms[i]           = tc - tb;
            iter_ms[i]           = tc - ta;
        }
        t_dsp      = trimmed_mean_1each(dsp_ms, SPMV_ITER);
        t_perm     = trimmed_mean_1each(perm_ms, SPMV_ITER);
        t_arm_rows = trimmed_mean_1each(arm_rows_ms, SPMV_ITER);
        t_arm_res  = trimmed_mean_1each(arm_res_ms, SPMV_ITER);
        t_arm_part = trimmed_mean_1each(arm_part_ms, SPMV_ITER);
        t_submit   = trimmed_mean_1each(submit_ms, SPMV_ITER);
        t_wait_after_arm = trimmed_mean_1each(wait_after_arm_ms, SPMV_ITER);

        if (dual_x_diag) {
            if (A.mode == PAGE_MODE_CPU) {
                fprintf(stderr, "[DUAL_X] ERROR: dual-x diagnostic requires a Hybrid partition\n");
                return 6;
            }
            const size_t xbytes = (size_t)n_pad * sizeof(MAT_VAL_TYPE);
            x_cpu_diag = (MAT_VAL_TYPE *)malloc(xbytes);
            diag_cpu_private_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            diag_dual_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            if (!x_cpu_diag || !diag_cpu_private_y || !diag_dual_y) {
                fprintf(stderr, "[DUAL_X] ERROR: CPU-private diagnostic allocation failed\n");
                return 6;
            }
            const double dxft0 = now_ms();
            page_parallel_first_touch_x(x_cpu_diag, x, n_pad);
            const double dxft_ms = now_ms() - dxft0;
            const int x_equal = (memcmp(x_cpu_diag, x, xbytes) == 0);
            printf("[DUAL_X] allocation bytes=%zu first_touch_ms=%.6f copy_equal=%d "
                   "x_dsp_ptr=%p x_cpu_ptr=%p\n",
                   xbytes, dxft_ms, x_equal, (void *)x, (void *)x_cpu_diag);
            if (!x_equal) {
                fprintf(stderr, "[DUAL_X] ERROR: x_cpu and x_dsp differ after copy\n");
                return 6;
            }
            page_cont_numa_object("x_dsp", x, xbytes);
            page_cont_numa_object("x_cpu", x_cpu_diag, xbytes);

            double cf_s[PAGE_DUAL_X_DIAG_ITER], cf_p[PAGE_DUAL_X_DIAG_ITER];
            double cp_s[PAGE_DUAL_X_DIAG_ITER], cp_p[PAGE_DUAL_X_DIAG_ITER];
            double hs_total[PAGE_DUAL_X_DIAG_ITER], hp_total[PAGE_DUAL_X_DIAG_ITER];
            double hs_cpu[PAGE_DUAL_X_DIAG_ITER], hp_cpu[PAGE_DUAL_X_DIAG_ITER];
            double hs_sub[PAGE_DUAL_X_DIAG_ITER], hp_sub[PAGE_DUAL_X_DIAG_ITER];
            double hs_wait[PAGE_DUAL_X_DIAG_ITER], hp_wait[PAGE_DUAL_X_DIAG_ITER];
            double hs_fin[PAGE_DUAL_X_DIAG_ITER], hp_fin[PAGE_DUAL_X_DIAG_ITER];

            for (int w = 0; w < PAGE_DUAL_X_DIAG_WARMUP; w++) {
                page_cpu_spmv_overt(&csr, x, diag_cpu_full_y);
                page_cpu_spmv_overt(&csr, x_cpu_diag, diag_cpu_private_y);
                page_cont_cpu_part(&csr, &A, x, y, y_res);
                page_cont_cpu_part(&csr, &A, x_cpu_diag, y, y_res);
            }
            for (int i = 0; i < PAGE_DUAL_X_DIAG_ITER; i++) {
                if ((i & 1) == 0) {
                    double a = now_ms();
                    page_cpu_spmv_overt(&csr, x, diag_cpu_full_y);
                    cf_s[i] = now_ms() - a;
                    a = now_ms();
                    page_cpu_spmv_overt(&csr, x_cpu_diag, diag_cpu_private_y);
                    cf_p[i] = now_ms() - a;
                    a = now_ms();
                    page_cont_cpu_part(&csr, &A, x, y, y_res);
                    cp_s[i] = now_ms() - a;
                    a = now_ms();
                    page_cont_cpu_part(&csr, &A, x_cpu_diag, y, y_res);
                    cp_p[i] = now_ms() - a;
                } else {
                    double a = now_ms();
                    page_cpu_spmv_overt(&csr, x_cpu_diag, diag_cpu_private_y);
                    cf_p[i] = now_ms() - a;
                    a = now_ms();
                    page_cpu_spmv_overt(&csr, x, diag_cpu_full_y);
                    cf_s[i] = now_ms() - a;
                    a = now_ms();
                    page_cont_cpu_part(&csr, &A, x_cpu_diag, y, y_res);
                    cp_p[i] = now_ms() - a;
                    a = now_ms();
                    page_cont_cpu_part(&csr, &A, x, y, y_res);
                    cp_s[i] = now_ms() - a;
                }
            }

            for (int w = 0; w < PAGE_DUAL_X_DIAG_WARMUP; w++) {
                (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                               &csr, &A, x, y, y_res, yd);
                (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                               &csr, &A, x_cpu_diag, y, y_res, yd);
            }
            for (int i = 0; i < PAGE_DUAL_X_DIAG_ITER; i++) {
                page_dual_x_hybrid_timing a, b;
                if ((i & 1) == 0) {
                    a = page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                                 &csr, &A, x, y, y_res, yd);
                    b = page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                                 &csr, &A, x_cpu_diag, y, y_res, yd);
                } else {
                    b = page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                                 &csr, &A, x_cpu_diag, y, y_res, yd);
                    a = page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                                 &csr, &A, x, y, y_res, yd);
                }
                hs_sub[i] = a.submit_ms; hs_cpu[i] = a.cpu_ms;
                hs_wait[i] = a.wait_ms; hs_fin[i] = a.finalize_ms; hs_total[i] = a.total_ms;
                hp_sub[i] = b.submit_ms; hp_cpu[i] = b.cpu_ms;
                hp_wait[i] = b.wait_ms; hp_fin[i] = b.finalize_ms; hp_total[i] = b.total_ms;
            }

            const double cf_s_m = trimmed_mean_1each(cf_s, PAGE_DUAL_X_DIAG_ITER);
            const double cf_p_m = trimmed_mean_1each(cf_p, PAGE_DUAL_X_DIAG_ITER);
            const double cp_s_m = trimmed_mean_1each(cp_s, PAGE_DUAL_X_DIAG_ITER);
            const double cp_p_m = trimmed_mean_1each(cp_p, PAGE_DUAL_X_DIAG_ITER);
            const double hs_sub_m = trimmed_mean_1each(hs_sub, PAGE_DUAL_X_DIAG_ITER);
            const double hs_cpu_m = trimmed_mean_1each(hs_cpu, PAGE_DUAL_X_DIAG_ITER);
            const double hs_wait_m = trimmed_mean_1each(hs_wait, PAGE_DUAL_X_DIAG_ITER);
            const double hs_fin_m = trimmed_mean_1each(hs_fin, PAGE_DUAL_X_DIAG_ITER);
            const double hs_total_m = trimmed_mean_1each(hs_total, PAGE_DUAL_X_DIAG_ITER);
            const double hp_sub_m = trimmed_mean_1each(hp_sub, PAGE_DUAL_X_DIAG_ITER);
            const double hp_cpu_m = trimmed_mean_1each(hp_cpu, PAGE_DUAL_X_DIAG_ITER);
            const double hp_wait_m = trimmed_mean_1each(hp_wait, PAGE_DUAL_X_DIAG_ITER);
            const double hp_fin_m = trimmed_mean_1each(hp_fin, PAGE_DUAL_X_DIAG_ITER);
            const double hp_total_m = trimmed_mean_1each(hp_total, PAGE_DUAL_X_DIAG_ITER);
            const double formal_shared_m = trimmed_mean_1each(iter_ms, SPMV_ITER);

            printf("[DUAL_X] cpu_full shared_ms=%.6f private_ms=%.6f private_over_shared=%.8f\n",
                   cf_s_m, cf_p_m, cf_s_m > 0.0 ? cf_p_m / cf_s_m : 0.0);
            printf("[DUAL_X] cpu_part_solo shared_ms=%.6f private_ms=%.6f private_over_shared=%.8f\n",
                   cp_s_m, cp_p_m, cp_s_m > 0.0 ? cp_p_m / cp_s_m : 0.0);
            printf("[DUAL_X] hybrid_shared submit_ms=%.6f cpu_part_ms=%.6f wait_ms=%.6f "
                   "finalize_ms=%.6f total_ms=%.6f formal_total_ms=%.6f\n",
                   hs_sub_m, hs_cpu_m, hs_wait_m, hs_fin_m, hs_total_m, formal_shared_m);
            printf("[DUAL_X] hybrid_dual submit_ms=%.6f cpu_part_ms=%.6f wait_ms=%.6f "
                   "finalize_ms=%.6f total_ms=%.6f\n",
                   hp_sub_m, hp_cpu_m, hp_wait_m, hp_fin_m, hp_total_m);
            printf("[DUAL_X] derived total_dual_over_shared=%.8f cpu_part_dual_over_shared=%.8f "
                   "shared_contention_ratio=%.8f dual_contention_ratio=%.8f "
                   "contention_relief_ratio=%.8f cpu_full_private_speedup_vs_dual=%.8f "
                   "cpu_full_shared_speedup_vs_shared=%.8f paired_shared_over_formal=%.8f\n",
                   hs_total_m > 0.0 ? hp_total_m / hs_total_m : 0.0,
                   hs_cpu_m > 0.0 ? hp_cpu_m / hs_cpu_m : 0.0,
                   cp_s_m > 0.0 ? hs_cpu_m / cp_s_m : 0.0,
                   cp_p_m > 0.0 ? hp_cpu_m / cp_p_m : 0.0,
                   (cp_p_m > 0.0 && hp_cpu_m > 0.0 && cp_s_m > 0.0)
                       ? (hs_cpu_m / cp_s_m) / (hp_cpu_m / cp_p_m) : 0.0,
                   hp_total_m > 0.0 ? cf_p_m / hp_total_m : 0.0,
                   hs_total_m > 0.0 ? cf_s_m / hs_total_m : 0.0,
                   formal_shared_m > 0.0 ? hs_total_m / formal_shared_m : 0.0);

            page_dual_x_gather_pair(&csr, &A, x, x_cpu_diag, thread_id,
                                     scalArgs, vecArgs, args,
                                     cp_p_m > cp_s_m ? cp_p_m : cp_s_m);

            (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                           &csr, &A, x_cpu_diag, y, y_res, yd);
            memcpy(diag_dual_y, y, sizeof(MAT_VAL_TYPE) * (size_t)m);
            (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                           &csr, &A, x, y, y_res, yd);
        }

        if (xplace_diag) {
            if (A.mode == PAGE_MODE_CPU) {
                fprintf(stderr, "[XPLACE] ERROR: x-placement diagnostic requires a Hybrid partition\n");
                return 7;
            }
            const size_t xbytes = (size_t)n_pad * sizeof(MAT_VAL_TYPE);

            x_cpu_diag = (MAT_VAL_TYPE *)malloc(xbytes);
            diag_cpu_private_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            diag_dual_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            if (!x_cpu_diag || !diag_cpu_private_y || !diag_dual_y) {
                fprintf(stderr, "[XPLACE] ERROR: V5 first-touch control allocation failed\n");
                return 7;
            }
            const double ft0 = now_ms();
            page_parallel_first_touch_x(x_cpu_diag, x, n_pad);
            const double ft_ms = now_ms() - ft0;

            diag_cpu_interleave_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            diag_interleave_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            if (!diag_cpu_interleave_y || !diag_interleave_y) {
                fprintf(stderr, "[XPLACE] ERROR: interleave diagnostic output allocation failed\n");
                return 7;
            }

            const double il0 = now_ms();
            x_interleave_diag = page_xplace_alloc_interleaved(x, n_pad);
            const double il_ms = now_ms() - il0;
            if (!x_interleave_diag.ptr || !x_interleave_diag.mbind_ok) {
                fprintf(stderr,
                        "[XPLACE] ERROR: runtime interleaved x allocation failed errno=%d (%s) allowed=%s\n",
                        x_interleave_diag.err, strerror(x_interleave_diag.err),
                        x_interleave_diag.allowed_list[0] ? x_interleave_diag.allowed_list : "unknown");
                return 7;
            }

            const int eq_ft = (memcmp(x_cpu_diag, x, xbytes) == 0);
            const int eq_il = (memcmp(x_interleave_diag.ptr, x, xbytes) == 0);
            printf("[XPLACE] allocation bytes=%zu firsttouch_ms=%.6f interleave_ms=%.6f "
                   "firsttouch_equal=%d interleave_equal=%d mbind_ok=%d "
                   "allowed_nodes=%s allowed_count=%d maxnode=%lu "
                   "x_shared_ptr=%p x_firsttouch_ptr=%p x_interleave_ptr=%p\n",
                   xbytes, ft_ms, il_ms, eq_ft, eq_il, x_interleave_diag.mbind_ok,
                   x_interleave_diag.allowed_list, x_interleave_diag.node_count,
                   x_interleave_diag.maxnode, (void *)x, (void *)x_cpu_diag,
                   (void *)x_interleave_diag.ptr);
            if (!eq_ft || !eq_il) {
                fprintf(stderr, "[XPLACE] ERROR: CPU-private x copy mismatch\n");
                return 7;
            }
            page_cont_numa_object("x_shared", x, xbytes);
            page_cont_numa_object("x_firsttouch", x_cpu_diag, xbytes);
            page_cont_numa_object("x_interleave", x_interleave_diag.ptr, xbytes);

            const MAT_VAL_TYPE *xs[3] = {x, x_cpu_diag, x_interleave_diag.ptr};
            MAT_VAL_TYPE *full_y[3] = {diag_cpu_full_y, diag_cpu_private_y,
                                       diag_cpu_interleave_y};
            double cf[3][PAGE_XPLACE_DIAG_ITER];
            double cp[3][PAGE_XPLACE_DIAG_ITER];
            double hsub[3][PAGE_XPLACE_DIAG_ITER];
            double hcpu[3][PAGE_XPLACE_DIAG_ITER];
            double hwait[3][PAGE_XPLACE_DIAG_ITER];
            double hfin[3][PAGE_XPLACE_DIAG_ITER];
            double htotal[3][PAGE_XPLACE_DIAG_ITER];
            memset(cf, 0, sizeof(cf)); memset(cp, 0, sizeof(cp));
            memset(hsub, 0, sizeof(hsub)); memset(hcpu, 0, sizeof(hcpu));
            memset(hwait, 0, sizeof(hwait)); memset(hfin, 0, sizeof(hfin));
            memset(htotal, 0, sizeof(htotal));

            for (int w = 0; w < PAGE_XPLACE_DIAG_WARMUP; w++) {
                for (int k = 0; k < 3; k++) {
                    page_cpu_spmv_overt(&csr, xs[k], full_y[k]);
                    page_cont_cpu_part(&csr, &A, xs[k], y, y_res);
                }
            }
            for (int i = 0; i < PAGE_XPLACE_DIAG_ITER; i++) {
                for (int pos = 0; pos < 3; pos++) {
                    const int k = (i + pos) % 3;
                    double a = now_ms();
                    page_cpu_spmv_overt(&csr, xs[k], full_y[k]);
                    cf[k][i] = now_ms() - a;
                    a = now_ms();
                    page_cont_cpu_part(&csr, &A, xs[k], y, y_res);
                    cp[k][i] = now_ms() - a;
                }
            }

            for (int w = 0; w < PAGE_XPLACE_DIAG_WARMUP; w++)
                for (int k = 0; k < 3; k++)
                    (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                                   &csr, &A, xs[k], y, y_res, yd);
            for (int i = 0; i < PAGE_XPLACE_DIAG_ITER; i++) {
                for (int pos = 0; pos < 3; pos++) {
                    const int k = (i + pos) % 3;
                    page_dual_x_hybrid_timing z = page_dual_x_hybrid_once(
                        thread_id, scalArgs, vecArgs, args, &csr, &A,
                        xs[k], y, y_res, yd);
                    hsub[k][i] = z.submit_ms; hcpu[k][i] = z.cpu_ms;
                    hwait[k][i] = z.wait_ms; hfin[k][i] = z.finalize_ms;
                    htotal[k][i] = z.total_ms;
                }
            }

            double cfm[3], cpm[3], hsubm[3], hcpum[3], hwaitm[3], hfinm[3], htotalm[3];
            for (int k = 0; k < 3; k++) {
                cfm[k] = trimmed_mean_1each(cf[k], PAGE_XPLACE_DIAG_ITER);
                cpm[k] = trimmed_mean_1each(cp[k], PAGE_XPLACE_DIAG_ITER);
                hsubm[k] = trimmed_mean_1each(hsub[k], PAGE_XPLACE_DIAG_ITER);
                hcpum[k] = trimmed_mean_1each(hcpu[k], PAGE_XPLACE_DIAG_ITER);
                hwaitm[k] = trimmed_mean_1each(hwait[k], PAGE_XPLACE_DIAG_ITER);
                hfinm[k] = trimmed_mean_1each(hfin[k], PAGE_XPLACE_DIAG_ITER);
                htotalm[k] = trimmed_mean_1each(htotal[k], PAGE_XPLACE_DIAG_ITER);
            }
            const double formal_shared_m = trimmed_mean_1each(iter_ms, SPMV_ITER);

            printf("[XPLACE] cpu_full shared_ms=%.6f firsttouch_ms=%.6f interleave_ms=%.6f "
                   "firsttouch_over_shared=%.8f interleave_over_shared=%.8f "
                   "interleave_over_firsttouch=%.8f\n",
                   cfm[0], cfm[1], cfm[2],
                   cfm[0] > 0.0 ? cfm[1] / cfm[0] : 0.0,
                   cfm[0] > 0.0 ? cfm[2] / cfm[0] : 0.0,
                   cfm[1] > 0.0 ? cfm[2] / cfm[1] : 0.0);
            printf("[XPLACE] cpu_part_solo shared_ms=%.6f firsttouch_ms=%.6f interleave_ms=%.6f "
                   "firsttouch_over_shared=%.8f interleave_over_shared=%.8f "
                   "interleave_over_firsttouch=%.8f\n",
                   cpm[0], cpm[1], cpm[2],
                   cpm[0] > 0.0 ? cpm[1] / cpm[0] : 0.0,
                   cpm[0] > 0.0 ? cpm[2] / cpm[0] : 0.0,
                   cpm[1] > 0.0 ? cpm[2] / cpm[1] : 0.0);
            const char *labels[3] = {"shared", "firsttouch", "interleave"};
            for (int k = 0; k < 3; k++) {
                printf("[XPLACE] hybrid_%s submit_ms=%.6f cpu_part_ms=%.6f wait_ms=%.6f "
                       "finalize_ms=%.6f total_ms=%.6f%s",
                       labels[k], hsubm[k], hcpum[k], hwaitm[k], hfinm[k], htotalm[k],
                       k == 0 ? " formal_total_ms=" : "\n");
                if (k == 0) printf("%.6f\n", formal_shared_m);
            }
            printf("[XPLACE] derived firsttouch_total_over_shared=%.8f "
                   "interleave_total_over_shared=%.8f interleave_total_over_firsttouch=%.8f "
                   "shared_contention_ratio=%.8f firsttouch_contention_ratio=%.8f "
                   "interleave_contention_ratio=%.8f "
                   "interleave_relief_vs_shared=%.8f interleave_relief_vs_firsttouch=%.8f "
                   "strong_cpu_speedup_vs_firsttouch_hybrid=%.8f "
                   "strong_cpu_speedup_vs_interleave_hybrid=%.8f "
                   "prealloc_cpu_full_ms=%.6f prealloc_cpu_speedup_vs_firsttouch_hybrid=%.8f "
                   "prealloc_cpu_speedup_vs_interleave_hybrid=%.8f "
                   "postalloc_shared_cpu_over_prealloc=%.8f "
                   "same_layout_firsttouch_cpu_speedup=%.8f "
                   "same_layout_interleave_cpu_speedup=%.8f paired_shared_over_formal=%.8f\n",
                   htotalm[0] > 0.0 ? htotalm[1] / htotalm[0] : 0.0,
                   htotalm[0] > 0.0 ? htotalm[2] / htotalm[0] : 0.0,
                   htotalm[1] > 0.0 ? htotalm[2] / htotalm[1] : 0.0,
                   cpm[0] > 0.0 ? hcpum[0] / cpm[0] : 0.0,
                   cpm[1] > 0.0 ? hcpum[1] / cpm[1] : 0.0,
                   cpm[2] > 0.0 ? hcpum[2] / cpm[2] : 0.0,
                   (cpm[2] > 0.0 && hcpum[2] > 0.0 && cpm[0] > 0.0)
                       ? (hcpum[0] / cpm[0]) / (hcpum[2] / cpm[2]) : 0.0,
                   (cpm[2] > 0.0 && hcpum[2] > 0.0 && cpm[1] > 0.0)
                       ? (hcpum[1] / cpm[1]) / (hcpum[2] / cpm[2]) : 0.0,
                   htotalm[1] > 0.0 ? cfm[0] / htotalm[1] : 0.0,
                   htotalm[2] > 0.0 ? cfm[0] / htotalm[2] : 0.0,
                   diag_cpu_full,
                   htotalm[1] > 0.0 ? diag_cpu_full / htotalm[1] : 0.0,
                   htotalm[2] > 0.0 ? diag_cpu_full / htotalm[2] : 0.0,
                   diag_cpu_full > 0.0 ? cfm[0] / diag_cpu_full : 0.0,
                   htotalm[1] > 0.0 ? cfm[1] / htotalm[1] : 0.0,
                   htotalm[2] > 0.0 ? cfm[2] / htotalm[2] : 0.0,
                   formal_shared_m > 0.0 ? htotalm[0] / formal_shared_m : 0.0);

            double target = cpm[0];
            if (cpm[1] > target) target = cpm[1];
            if (cpm[2] > target) target = cpm[2];
            page_xplace_gather_triple(&csr, &A, x, x_cpu_diag,
                                       x_interleave_diag.ptr, thread_id,
                                       scalArgs, vecArgs, args, target);

            (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                           &csr, &A, x_cpu_diag, y, y_res, yd);
            memcpy(diag_dual_y, y, sizeof(MAT_VAL_TYPE) * (size_t)m);
            (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                           &csr, &A, x_interleave_diag.ptr, y, y_res, yd);
            memcpy(diag_interleave_y, y, sizeof(MAT_VAL_TYPE) * (size_t)m);
            (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                           &csr, &A, x, y, y_res, yd);
        }

        if (htx_diag) {
            if (A.mode == PAGE_MODE_CPU) {
                fprintf(stderr, "[HTX] ERROR: hthread-x diagnostic requires a Hybrid partition\n");
                return 8;
            }
            const size_t xbytes = (size_t)n_pad * sizeof(MAT_VAL_TYPE);
            const double ht0 = now_ms();
            x_hthread_private_diag = (MAT_VAL_TYPE *)hthread_malloc(
                cluster_id, xbytes, HT_MEM_RW);
            if (!x_hthread_private_diag) {
                fprintf(stderr, "[HTX] ERROR: second hthread x allocation failed\n");
                return 8;
            }
            page_parallel_first_touch_x(x_hthread_private_diag, x, n_pad);
            const double ht_ms = now_ms() - ht0;

            diag_cpu_hthread_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            diag_hthread_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            diag_cpu_interleave_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            diag_interleave_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            if (!diag_cpu_hthread_y || !diag_hthread_y ||
                !diag_cpu_interleave_y || !diag_interleave_y) {
                fprintf(stderr, "[HTX] ERROR: diagnostic output allocation failed\n");
                return 8;
            }

            const double il0 = now_ms();
            x_interleave_diag = page_xplace_alloc_interleaved(x, n_pad);
            const double il_ms = now_ms() - il0;
            if (!x_interleave_diag.ptr || !x_interleave_diag.mbind_ok) {
                fprintf(stderr,
                        "[HTX] ERROR: runtime interleaved x allocation failed errno=%d (%s) allowed=%s\n",
                        x_interleave_diag.err, strerror(x_interleave_diag.err),
                        x_interleave_diag.allowed_list[0] ? x_interleave_diag.allowed_list : "unknown");
                return 8;
            }

            const int eq_ht = (memcmp(x_hthread_private_diag, x, xbytes) == 0);
            const int eq_il = (memcmp(x_interleave_diag.ptr, x, xbytes) == 0);
            const int distinct_ht = (x_hthread_private_diag != x);
            const int distinct_il = (x_interleave_diag.ptr != x &&
                                     x_interleave_diag.ptr != x_hthread_private_diag);
            printf("[HTX] allocation bytes=%zu hthread_private_ms=%.6f interleave_ms=%.6f "
                   "hthread_private_equal=%d interleave_equal=%d "
                   "hthread_private_distinct=%d interleave_distinct=%d mbind_ok=%d "
                   "allowed_nodes=%s allowed_count=%d maxnode=%lu "
                   "x_shared_ptr=%p x_hthread_private_ptr=%p x_interleave_ptr=%p\n",
                   xbytes, ht_ms, il_ms, eq_ht, eq_il, distinct_ht, distinct_il,
                   x_interleave_diag.mbind_ok, x_interleave_diag.allowed_list,
                   x_interleave_diag.node_count, x_interleave_diag.maxnode,
                   (void *)x, (void *)x_hthread_private_diag,
                   (void *)x_interleave_diag.ptr);
            if (!eq_ht || !eq_il || !distinct_ht || !distinct_il) {
                fprintf(stderr, "[HTX] ERROR: private x copy mismatch or non-distinct allocation\n");
                return 8;
            }

            page_cont_numa_object("x_shared", x, xbytes);
            page_cont_numa_object("x_hthread_private", x_hthread_private_diag, xbytes);
            page_cont_numa_object("x_interleave", x_interleave_diag.ptr, xbytes);
            page_cont_numa_object_raw("x_shared", x);
            page_cont_numa_object_raw("x_hthread_private", x_hthread_private_diag);
            page_cont_numa_object_raw("x_interleave", x_interleave_diag.ptr);

            const MAT_VAL_TYPE *xs[3] = {x, x_hthread_private_diag,
                                         x_interleave_diag.ptr};
            MAT_VAL_TYPE *full_y[3] = {diag_cpu_full_y, diag_cpu_hthread_y,
                                       diag_cpu_interleave_y};
            double cf[3][PAGE_XPLACE_DIAG_ITER];
            double cp[3][PAGE_XPLACE_DIAG_ITER];
            double hsub[3][PAGE_XPLACE_DIAG_ITER];
            double hcpu[3][PAGE_XPLACE_DIAG_ITER];
            double hwait[3][PAGE_XPLACE_DIAG_ITER];
            double hfin[3][PAGE_XPLACE_DIAG_ITER];
            double htotal[3][PAGE_XPLACE_DIAG_ITER];
            memset(cf, 0, sizeof(cf)); memset(cp, 0, sizeof(cp));
            memset(hsub, 0, sizeof(hsub)); memset(hcpu, 0, sizeof(hcpu));
            memset(hwait, 0, sizeof(hwait)); memset(hfin, 0, sizeof(hfin));
            memset(htotal, 0, sizeof(htotal));

            for (int w = 0; w < PAGE_XPLACE_DIAG_WARMUP; w++) {
                for (int k = 0; k < 3; k++) {
                    page_cpu_spmv_overt(&csr, xs[k], full_y[k]);
                    page_cont_cpu_part(&csr, &A, xs[k], y, y_res);
                }
            }
            for (int it = 0; it < PAGE_XPLACE_DIAG_ITER; it++) {
                for (int pos = 0; pos < 3; pos++) {
                    const int k = (it + pos) % 3;
                    double a = now_ms();
                    page_cpu_spmv_overt(&csr, xs[k], full_y[k]);
                    cf[k][it] = now_ms() - a;
                    a = now_ms();
                    page_cont_cpu_part(&csr, &A, xs[k], y, y_res);
                    cp[k][it] = now_ms() - a;
                }
            }

            for (int w = 0; w < PAGE_XPLACE_DIAG_WARMUP; w++)
                for (int k = 0; k < 3; k++)
                    (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                                   &csr, &A, xs[k], y, y_res, yd);
            for (int it = 0; it < PAGE_XPLACE_DIAG_ITER; it++) {
                for (int pos = 0; pos < 3; pos++) {
                    const int k = (it + pos) % 3;
                    page_dual_x_hybrid_timing z = page_dual_x_hybrid_once(
                        thread_id, scalArgs, vecArgs, args, &csr, &A,
                        xs[k], y, y_res, yd);
                    hsub[k][it] = z.submit_ms; hcpu[k][it] = z.cpu_ms;
                    hwait[k][it] = z.wait_ms; hfin[k][it] = z.finalize_ms;
                    htotal[k][it] = z.total_ms;
                }
            }

            double cfm[3], cpm[3], hsubm[3], hcpum[3], hwaitm[3], hfinm[3], htotalm[3];
            for (int k = 0; k < 3; k++) {
                cfm[k] = trimmed_mean_1each(cf[k], PAGE_XPLACE_DIAG_ITER);
                cpm[k] = trimmed_mean_1each(cp[k], PAGE_XPLACE_DIAG_ITER);
                hsubm[k] = trimmed_mean_1each(hsub[k], PAGE_XPLACE_DIAG_ITER);
                hcpum[k] = trimmed_mean_1each(hcpu[k], PAGE_XPLACE_DIAG_ITER);
                hwaitm[k] = trimmed_mean_1each(hwait[k], PAGE_XPLACE_DIAG_ITER);
                hfinm[k] = trimmed_mean_1each(hfin[k], PAGE_XPLACE_DIAG_ITER);
                htotalm[k] = trimmed_mean_1each(htotal[k], PAGE_XPLACE_DIAG_ITER);
            }
            const double formal_shared_m = trimmed_mean_1each(iter_ms, SPMV_ITER);

            printf("[HTX] cpu_full shared_ms=%.6f hthread_private_ms=%.6f interleave_ms=%.6f "
                   "hthread_private_over_shared=%.8f interleave_over_shared=%.8f "
                   "hthread_private_over_interleave=%.8f\n",
                   cfm[0], cfm[1], cfm[2],
                   cfm[0] > 0.0 ? cfm[1] / cfm[0] : 0.0,
                   cfm[0] > 0.0 ? cfm[2] / cfm[0] : 0.0,
                   cfm[2] > 0.0 ? cfm[1] / cfm[2] : 0.0);
            printf("[HTX] cpu_part_solo shared_ms=%.6f hthread_private_ms=%.6f interleave_ms=%.6f "
                   "hthread_private_over_shared=%.8f interleave_over_shared=%.8f "
                   "hthread_private_over_interleave=%.8f\n",
                   cpm[0], cpm[1], cpm[2],
                   cpm[0] > 0.0 ? cpm[1] / cpm[0] : 0.0,
                   cpm[0] > 0.0 ? cpm[2] / cpm[0] : 0.0,
                   cpm[2] > 0.0 ? cpm[1] / cpm[2] : 0.0);
            const char *labels[3] = {"shared", "hthread_private", "interleave"};
            for (int k = 0; k < 3; k++) {
                printf("[HTX] hybrid_%s submit_ms=%.6f cpu_part_ms=%.6f wait_ms=%.6f "
                       "finalize_ms=%.6f total_ms=%.6f%s",
                       labels[k], hsubm[k], hcpum[k], hwaitm[k], hfinm[k], htotalm[k],
                       k == 0 ? " formal_total_ms=" : "\n");
                if (k == 0) printf("%.6f\n", formal_shared_m);
            }
            printf("[HTX] derived hthread_private_total_over_shared=%.8f "
                   "interleave_total_over_shared=%.8f hthread_private_total_over_interleave=%.8f "
                   "shared_contention_ratio=%.8f hthread_private_contention_ratio=%.8f "
                   "interleave_contention_ratio=%.8f hthread_private_relief_vs_shared=%.8f "
                   "interleave_relief_vs_shared=%.8f "
                   "prealloc_cpu_full_ms=%.6f prealloc_cpu_speedup_vs_hthread_private_hybrid=%.8f "
                   "prealloc_cpu_speedup_vs_interleave_hybrid=%.8f "
                   "postalloc_shared_cpu_over_prealloc=%.8f "
                   "same_layout_hthread_private_cpu_speedup=%.8f "
                   "same_layout_interleave_cpu_speedup=%.8f paired_shared_over_formal=%.8f\n",
                   htotalm[0] > 0.0 ? htotalm[1] / htotalm[0] : 0.0,
                   htotalm[0] > 0.0 ? htotalm[2] / htotalm[0] : 0.0,
                   htotalm[2] > 0.0 ? htotalm[1] / htotalm[2] : 0.0,
                   cpm[0] > 0.0 ? hcpum[0] / cpm[0] : 0.0,
                   cpm[1] > 0.0 ? hcpum[1] / cpm[1] : 0.0,
                   cpm[2] > 0.0 ? hcpum[2] / cpm[2] : 0.0,
                   (cpm[1] > 0.0 && hcpum[1] > 0.0 && cpm[0] > 0.0)
                       ? (hcpum[0] / cpm[0]) / (hcpum[1] / cpm[1]) : 0.0,
                   (cpm[2] > 0.0 && hcpum[2] > 0.0 && cpm[0] > 0.0)
                       ? (hcpum[0] / cpm[0]) / (hcpum[2] / cpm[2]) : 0.0,
                   diag_cpu_full,
                   htotalm[1] > 0.0 ? diag_cpu_full / htotalm[1] : 0.0,
                   htotalm[2] > 0.0 ? diag_cpu_full / htotalm[2] : 0.0,
                   diag_cpu_full > 0.0 ? cfm[0] / diag_cpu_full : 0.0,
                   htotalm[1] > 0.0 ? cfm[1] / htotalm[1] : 0.0,
                   htotalm[2] > 0.0 ? cfm[2] / htotalm[2] : 0.0,
                   formal_shared_m > 0.0 ? htotalm[0] / formal_shared_m : 0.0);

            double target = cpm[0];
            if (cpm[1] > target) target = cpm[1];
            if (cpm[2] > target) target = cpm[2];
            page_htx_gather_triple(&csr, &A, x, x_hthread_private_diag,
                                    x_interleave_diag.ptr, thread_id,
                                    scalArgs, vecArgs, args, target);

            (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                           &csr, &A, x_hthread_private_diag, y, y_res, yd);
            memcpy(diag_hthread_y, y, sizeof(MAT_VAL_TYPE) * (size_t)m);
            (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                           &csr, &A, x_interleave_diag.ptr, y, y_res, yd);
            memcpy(diag_interleave_y, y, sizeof(MAT_VAL_TYPE) * (size_t)m);
            (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                           &csr, &A, x, y, y_res, yd);
        }

        if (thp_diag) {
            if (A.mode == PAGE_MODE_CPU) {
                fprintf(stderr, "[THP] ERROR: nohuge diagnostic requires a Hybrid partition\n");
                return 9;
            }
            const size_t xbytes = (size_t)n_pad * sizeof(MAT_VAL_TYPE);

            diag_cpu_interleave_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            diag_interleave_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            diag_cpu_nohuge_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            diag_nohuge_y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (size_t)m);
            if (!diag_cpu_interleave_y || !diag_interleave_y ||
                !diag_cpu_nohuge_y || !diag_nohuge_y) {
                fprintf(stderr, "[THP] ERROR: diagnostic output allocation failed\n");
                return 9;
            }

            const double il0 = now_ms();
            x_interleave_diag = page_xplace_alloc_interleaved(x, n_pad);
            const double il_ms = now_ms() - il0;
            const double nh0 = now_ms();
            x_nohuge_diag = page_xplace_alloc_interleaved_nohuge(x, n_pad);
            const double nh_ms = now_ms() - nh0;

            if (!x_interleave_diag.ptr || !x_interleave_diag.mbind_ok) {
                fprintf(stderr,
                        "[THP] ERROR: default interleave allocation failed errno=%d (%s) allowed=%s\n",
                        x_interleave_diag.err, strerror(x_interleave_diag.err),
                        x_interleave_diag.allowed_list[0] ? x_interleave_diag.allowed_list : "unknown");
                return 9;
            }
            if (!x_nohuge_diag.ptr || !x_nohuge_diag.mbind_ok || !x_nohuge_diag.madvise_ok) {
                fprintf(stderr,
                        "[THP] ERROR: nohuge interleave allocation failed errno=%d (%s) madvise_errno=%d allowed=%s\n",
                        x_nohuge_diag.err, strerror(x_nohuge_diag.err), x_nohuge_diag.madvise_err,
                        x_nohuge_diag.allowed_list[0] ? x_nohuge_diag.allowed_list : "unknown");
                return 9;
            }

            const int eq_il = (memcmp(x_interleave_diag.ptr, x, xbytes) == 0);
            const int eq_nh = (memcmp(x_nohuge_diag.ptr, x, xbytes) == 0);
            const int distinct_il = (x_interleave_diag.ptr != x);
            const int distinct_nh = (x_nohuge_diag.ptr != x &&
                                     x_nohuge_diag.ptr != x_interleave_diag.ptr);
            const int same_allowed = (strcmp(x_interleave_diag.allowed_list,
                                             x_nohuge_diag.allowed_list) == 0);
            printf("[THP] allocation bytes=%zu interleave_ms=%.6f nohuge_ms=%.6f "
                   "interleave_equal=%d nohuge_equal=%d interleave_distinct=%d "
                   "nohuge_distinct=%d interleave_mbind_ok=%d nohuge_mbind_ok=%d "
                   "nohuge_madvise_ok=%d nohuge_madvise_errno=%d "
                   "interleave_allowed_nodes=%s nohuge_allowed_nodes=%s same_allowed=%d "
                   "allowed_count=%d maxnode=%lu x_shared_ptr=%p x_interleave_ptr=%p "
                   "x_nohuge_ptr=%p\n",
                   xbytes, il_ms, nh_ms, eq_il, eq_nh, distinct_il, distinct_nh,
                   x_interleave_diag.mbind_ok, x_nohuge_diag.mbind_ok,
                   x_nohuge_diag.madvise_ok, x_nohuge_diag.madvise_err,
                   x_interleave_diag.allowed_list, x_nohuge_diag.allowed_list,
                   same_allowed, x_interleave_diag.node_count, x_interleave_diag.maxnode,
                   (void *)x, (void *)x_interleave_diag.ptr, (void *)x_nohuge_diag.ptr);
            if (!eq_il || !eq_nh || !distinct_il || !distinct_nh || !same_allowed) {
                fprintf(stderr, "[THP] ERROR: mapping equality/distinctness/allowed-mask check failed\n");
                return 9;
            }

            page_cont_numa_object("x_shared", x, xbytes);
            page_cont_numa_object("x_interleave", x_interleave_diag.ptr, xbytes);
            page_cont_numa_object("x_nohuge", x_nohuge_diag.ptr, xbytes);
            page_cont_numa_object_raw("x_shared", x);
            page_cont_numa_object_raw("x_interleave", x_interleave_diag.ptr);
            page_cont_numa_object_raw("x_nohuge", x_nohuge_diag.ptr);
            page_cont_smaps_object("x_interleave", x_interleave_diag.ptr);
            page_cont_smaps_object("x_nohuge", x_nohuge_diag.ptr);

            const MAT_VAL_TYPE *xs[3] = {x, x_interleave_diag.ptr, x_nohuge_diag.ptr};
            MAT_VAL_TYPE *full_y[3] = {diag_cpu_full_y, diag_cpu_interleave_y,
                                       diag_cpu_nohuge_y};
            double cf[3][PAGE_XPLACE_DIAG_ITER];
            double cp[3][PAGE_XPLACE_DIAG_ITER];
            double hsub[3][PAGE_XPLACE_DIAG_ITER];
            double hcpu[3][PAGE_XPLACE_DIAG_ITER];
            double hwait[3][PAGE_XPLACE_DIAG_ITER];
            double hfin[3][PAGE_XPLACE_DIAG_ITER];
            double htotal[3][PAGE_XPLACE_DIAG_ITER];
            memset(cf, 0, sizeof(cf)); memset(cp, 0, sizeof(cp));
            memset(hsub, 0, sizeof(hsub)); memset(hcpu, 0, sizeof(hcpu));
            memset(hwait, 0, sizeof(hwait)); memset(hfin, 0, sizeof(hfin));
            memset(htotal, 0, sizeof(htotal));

            for (int w = 0; w < PAGE_XPLACE_DIAG_WARMUP; w++) {
                for (int k = 0; k < 3; k++) {
                    page_cpu_spmv_overt(&csr, xs[k], full_y[k]);
                    page_cont_cpu_part(&csr, &A, xs[k], y, y_res);
                }
            }
            for (int it = 0; it < PAGE_XPLACE_DIAG_ITER; it++) {
                for (int pos = 0; pos < 3; pos++) {
                    const int k = (it + pos) % 3;
                    double a = now_ms();
                    page_cpu_spmv_overt(&csr, xs[k], full_y[k]);
                    cf[k][it] = now_ms() - a;
                    a = now_ms();
                    page_cont_cpu_part(&csr, &A, xs[k], y, y_res);
                    cp[k][it] = now_ms() - a;
                }
            }

            for (int w = 0; w < PAGE_XPLACE_DIAG_WARMUP; w++)
                for (int k = 0; k < 3; k++)
                    (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                                   &csr, &A, xs[k], y, y_res, yd);
            for (int it = 0; it < PAGE_XPLACE_DIAG_ITER; it++) {
                for (int pos = 0; pos < 3; pos++) {
                    const int k = (it + pos) % 3;
                    page_dual_x_hybrid_timing z = page_dual_x_hybrid_once(
                        thread_id, scalArgs, vecArgs, args, &csr, &A,
                        xs[k], y, y_res, yd);
                    hsub[k][it] = z.submit_ms; hcpu[k][it] = z.cpu_ms;
                    hwait[k][it] = z.wait_ms; hfin[k][it] = z.finalize_ms;
                    htotal[k][it] = z.total_ms;
                }
            }

            double cfm[3], cpm[3], hsubm[3], hcpum[3], hwaitm[3], hfinm[3], htotalm[3];
            for (int k = 0; k < 3; k++) {
                cfm[k] = trimmed_mean_1each(cf[k], PAGE_XPLACE_DIAG_ITER);
                cpm[k] = trimmed_mean_1each(cp[k], PAGE_XPLACE_DIAG_ITER);
                hsubm[k] = trimmed_mean_1each(hsub[k], PAGE_XPLACE_DIAG_ITER);
                hcpum[k] = trimmed_mean_1each(hcpu[k], PAGE_XPLACE_DIAG_ITER);
                hwaitm[k] = trimmed_mean_1each(hwait[k], PAGE_XPLACE_DIAG_ITER);
                hfinm[k] = trimmed_mean_1each(hfin[k], PAGE_XPLACE_DIAG_ITER);
                htotalm[k] = trimmed_mean_1each(htotal[k], PAGE_XPLACE_DIAG_ITER);
            }
            const double formal_shared_m = trimmed_mean_1each(iter_ms, SPMV_ITER);

            printf("[THP] cpu_full shared_ms=%.6f interleave_ms=%.6f nohuge_ms=%.6f "
                   "interleave_over_shared=%.8f nohuge_over_shared=%.8f "
                   "nohuge_over_interleave=%.8f\n",
                   cfm[0], cfm[1], cfm[2],
                   cfm[0] > 0.0 ? cfm[1] / cfm[0] : 0.0,
                   cfm[0] > 0.0 ? cfm[2] / cfm[0] : 0.0,
                   cfm[1] > 0.0 ? cfm[2] / cfm[1] : 0.0);
            printf("[THP] cpu_part_solo shared_ms=%.6f interleave_ms=%.6f nohuge_ms=%.6f "
                   "interleave_over_shared=%.8f nohuge_over_shared=%.8f "
                   "nohuge_over_interleave=%.8f\n",
                   cpm[0], cpm[1], cpm[2],
                   cpm[0] > 0.0 ? cpm[1] / cpm[0] : 0.0,
                   cpm[0] > 0.0 ? cpm[2] / cpm[0] : 0.0,
                   cpm[1] > 0.0 ? cpm[2] / cpm[1] : 0.0);
            const char *labels[3] = {"shared", "interleave", "nohuge"};
            for (int k = 0; k < 3; k++) {
                printf("[THP] hybrid_%s submit_ms=%.6f cpu_part_ms=%.6f wait_ms=%.6f "
                       "finalize_ms=%.6f total_ms=%.6f%s",
                       labels[k], hsubm[k], hcpum[k], hwaitm[k], hfinm[k], htotalm[k],
                       k == 0 ? " formal_total_ms=" : "\n");
                if (k == 0) printf("%.6f\n", formal_shared_m);
            }
            printf("[THP] derived interleave_total_over_shared=%.8f "
                   "nohuge_total_over_shared=%.8f nohuge_total_over_interleave=%.8f "
                   "shared_contention_ratio=%.8f interleave_contention_ratio=%.8f "
                   "nohuge_contention_ratio=%.8f interleave_relief_vs_shared=%.8f "
                   "nohuge_relief_vs_shared=%.8f nohuge_relief_vs_interleave=%.8f "
                   "prealloc_cpu_full_ms=%.6f prealloc_cpu_speedup_vs_interleave_hybrid=%.8f "
                   "prealloc_cpu_speedup_vs_nohuge_hybrid=%.8f "
                   "postalloc_shared_cpu_over_prealloc=%.8f "
                   "same_layout_interleave_cpu_speedup=%.8f "
                   "same_layout_nohuge_cpu_speedup=%.8f paired_shared_over_formal=%.8f\n",
                   htotalm[0] > 0.0 ? htotalm[1] / htotalm[0] : 0.0,
                   htotalm[0] > 0.0 ? htotalm[2] / htotalm[0] : 0.0,
                   htotalm[1] > 0.0 ? htotalm[2] / htotalm[1] : 0.0,
                   cpm[0] > 0.0 ? hcpum[0] / cpm[0] : 0.0,
                   cpm[1] > 0.0 ? hcpum[1] / cpm[1] : 0.0,
                   cpm[2] > 0.0 ? hcpum[2] / cpm[2] : 0.0,
                   (cpm[1] > 0.0 && hcpum[1] > 0.0 && cpm[0] > 0.0)
                       ? (hcpum[0] / cpm[0]) / (hcpum[1] / cpm[1]) : 0.0,
                   (cpm[2] > 0.0 && hcpum[2] > 0.0 && cpm[0] > 0.0)
                       ? (hcpum[0] / cpm[0]) / (hcpum[2] / cpm[2]) : 0.0,
                   (cpm[1] > 0.0 && hcpum[1] > 0.0 && cpm[2] > 0.0 && hcpum[2] > 0.0)
                       ? (hcpum[1] / cpm[1]) / (hcpum[2] / cpm[2]) : 0.0,
                   diag_cpu_full,
                   htotalm[1] > 0.0 ? diag_cpu_full / htotalm[1] : 0.0,
                   htotalm[2] > 0.0 ? diag_cpu_full / htotalm[2] : 0.0,
                   diag_cpu_full > 0.0 ? cfm[0] / diag_cpu_full : 0.0,
                   htotalm[1] > 0.0 ? cfm[1] / htotalm[1] : 0.0,
                   htotalm[2] > 0.0 ? cfm[2] / htotalm[2] : 0.0,
                   formal_shared_m > 0.0 ? htotalm[0] / formal_shared_m : 0.0);

            double target = cpm[0];
            if (cpm[1] > target) target = cpm[1];
            if (cpm[2] > target) target = cpm[2];
            page_thp_gather_triple(&csr, &A, x, x_interleave_diag.ptr,
                                    x_nohuge_diag.ptr, thread_id,
                                    scalArgs, vecArgs, args, target);

            (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                           &csr, &A, x_interleave_diag.ptr, y, y_res, yd);
            memcpy(diag_interleave_y, y, sizeof(MAT_VAL_TYPE) * (size_t)m);
            (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                           &csr, &A, x_nohuge_diag.ptr, y, y_res, yd);
            memcpy(diag_nohuge_y, y, sizeof(MAT_VAL_TYPE) * (size_t)m);
            (void)page_dual_x_hybrid_once(thread_id, scalArgs, vecArgs, args,
                                           &csr, &A, x, y, y_res, yd);
        }

        if (contention_diag) {
            const long long cont_prefix_nnz = (A.row_off > 0 && st.row_pre)
                                            ? st.row_pre[A.row_off] : 0;
            const long long cont_cpu_nnz = cont_prefix_nnz + A.res_nnz;
            const double cpu_val_bytes = (csr.value_mode == PAGE_VALUE_GENERAL) ? 8.0 : 0.0;
            const double dsp_val_bytes = (A.dsp_value_mode == PAGE_VALUE_GENERAL) ? 8.0 : 0.0;
            const double cpu_matrix_bytes = (double)cont_cpu_nnz * (4.0 + cpu_val_bytes)
                                          + (double)(A.row_off + A.m_dsp + 2) * 4.0;
            const double cpu_x_logical_bytes = (double)cont_cpu_nnz * 8.0;
            const double dsp_pair_bytes = (double)A.stored * (dsp_val_bytes + 8.0);
            const double dsp_meta_bytes = (double)A.stored * (double)A.idx_bytes
                                        + (double)(A.packed_x_elems / PAGE_XBLOCK_ELEMS) * 4.0;
            const double dsp_xload_bytes = (double)A.x_elems * 8.0;
            const double dsp_ywrite_bytes = (double)A.nvrow * 8.0;
            const double model_xload_ms = (opt.bw_xload > 0.0)
                ? dsp_xload_bytes / opt.bw_xload * 1e3
                  + (double)A.packed_dma_rounds * opt.xload_fixed_ms
                : 0.0;
            const double model_pair_ms = (opt.bw_pipe > 0.0)
                ? dsp_pair_bytes / opt.bw_pipe * 1e3 : 0.0;
            const double model_meta_ms = (opt.bw_ddr > 0.0)
                ? (dsp_meta_bytes + dsp_ywrite_bytes) / opt.bw_ddr * 1e3 : 0.0;
            int first_tile = -1;
            if (A.npanel > 0 && A.panel_tile_lo) first_tile = A.panel_tile_lo[0];
            const double first_xload_ms = page_cont_tile_xload_model_ms(
                &A, first_tile, coreNum, opt.bw_xload, opt.xload_fixed_ms);
            const double formal_hybrid_ms = trimmed_mean_1each(iter_ms, SPMV_ITER);

            printf("[CONT_TRAFFIC] cpu_work_nnz=%lld cpu_matrix_bytes=%.0f cpu_x_logical_bytes=%.0f "
                   "dsp_stored=%lld dsp_pair_bytes=%.0f dsp_meta_bytes=%.0f dsp_xload_bytes=%.0f "
                   "dsp_ywrite_bytes=%.0f ntile=%d packed_tiles=%d packed_dma_rounds=%d "
                   "model_pair_ms=%.6f model_xload_ms=%.6f model_meta_y_ms=%.6f "
                   "first_tile=%d first_tile_xload_model_ms=%.6f\n",
                   cont_cpu_nnz, cpu_matrix_bytes, cpu_x_logical_bytes,
                   A.stored, dsp_pair_bytes, dsp_meta_bytes, dsp_xload_bytes,
                   dsp_ywrite_bytes, A.ntile, A.packed_tiles, A.packed_dma_rounds,
                   model_pair_ms, model_xload_ms, model_meta_ms,
                   first_tile, first_xload_ms);

            page_cont_numa_object("csr_rowptr", csr.rowPointers,
                                   (size_t)(csr.numRows + 1) * sizeof(int));
            page_cont_numa_object("csr_colidx", csr.colIndices,
                                   (size_t)csr.numNonzeros * sizeof(int));
            if (csr.values)
                page_cont_numa_object("csr_values", csr.values,
                                       (size_t)csr.numNonzeros * sizeof(MAT_VAL_TYPE));
            page_cont_numa_object("x", x, (size_t)n_pad * sizeof(MAT_VAL_TYPE));
            page_cont_numa_object("hybrid_ybuf", ybuf,
                                   ((size_t)A.row_off + yd_elems) * sizeof(MAT_VAL_TYPE));
            if (A.has_window && A.res_nnz > 0) {
                page_cont_numa_object("res_colidx", A.res_ci,
                                       (size_t)A.res_nnz * sizeof(int));
                if (A.res_val)
                    page_cont_numa_object("res_values", A.res_val,
                                           (size_t)A.res_nnz * sizeof(MAT_VAL_TYPE));
            }

            page_cont_stress_ctx sc;
            if (!page_cont_stress_init(&sc, coreNum, cluster_id)) {
                fprintf(stderr, "[CONT] ERROR: synthetic DSP stress allocation failed\n");
                return 6;
            }
            printf("[CONT] synthetic_stress nbytes_per_core=%d total_bytes_per_round=%zu "
                   "warmup=%d timed=%d target_cover_factor=%.2f\n",
                   sc.nbytes, (size_t)sc.nbytes * (size_t)coreNum,
                   PAGE_CONT_DIAG_WARMUP, PAGE_CONT_DIAG_ITER,
                   (double)PAGE_CONT_DIAG_COVER_FACTOR);

            for (int md = 0; md < PAGE_CONT_DSP_COUNT; md++) {
                const page_cont_dsp_mode mode = (page_cont_dsp_mode)md;
                const page_cont_stress_result sr = page_cont_run_stress_diag(
                    mode, thread_id, &sc, &csr, &A, x, y, y_res, diag_cpu_part_solo);
                printf("[CONT_DSP_STRESS] mode=%s nrep=%d nbytes=%d target_ms=%.6f "
                       "solo_submit_ms=%.6f solo_post_ms=%.6f concurrent_submit_ms=%.6f "
                       "concurrent_post_endpoint_ms=%.6f cpu_part_ms=%.6f "
                       "wait_after_cpu_ms=%.6f cpu_slowdown=%.8f dsp_slowdown=%.8f "
                       "dsp_completion_after_cpu=%d\n",
                       page_cont_dsp_mode_name(mode), sr.nrep, sc.nbytes, sr.target_ms,
                       sr.solo_submit_ms, sr.solo_post_ms, sr.concurrent_submit_ms,
                       sr.concurrent_post_ms, sr.cpu_part_ms, sr.wait_after_cpu_ms,
                       sr.cpu_slowdown, sr.dsp_slowdown, sr.dsp_completion_after_cpu);
            }

            const int matrix_reps = page_cont_tune_cpu_component(
                &csr, &A, x, 0, diag_cpu_part_solo);
            const int gather_reps = page_cont_tune_cpu_component(
                &csr, &A, x, 1, diag_cpu_part_solo);
            page_cont_run_cpu_component_diag("matrix_stream", 0, matrix_reps,
                                               &csr, &A, x, thread_id,
                                               scalArgs, vecArgs, args);
            page_cont_run_cpu_component_diag("x_gather", 1, gather_reps,
                                               &csr, &A, x, thread_id,
                                               scalArgs, vecArgs, args);

            double ov_cpu[PAGE_CONT_DIAG_ITER], ov_dsp[PAGE_CONT_DIAG_ITER];
            double ov_wait[PAGE_CONT_DIAG_ITER], ov_fin[PAGE_CONT_DIAG_ITER];
            double ov_total[PAGE_CONT_DIAG_ITER], ov_submit[PAGE_CONT_DIAG_ITER];
            for (int w = 0; w < PAGE_CONT_DIAG_WARMUP; w++) {
                hthread_group_exec(thread_id, "page_spmv", scalArgs, vecArgs, args);
                page_cont_spin_delay(first_xload_ms);
                page_cont_cpu_part(&csr, &A, x, y, y_res);
                hthread_group_wait(thread_id);
                page_finalize_y(&A, yd, y_res, y);
            }
            for (int i = 0; i < PAGE_CONT_DIAG_ITER; i++) {
                const double a = now_ms();
                hthread_group_exec(thread_id, "page_spmv", scalArgs, vecArgs, args);
                const double b = now_ms();
                page_cont_spin_delay(first_xload_ms);
                const double c = now_ms();
                page_cont_cpu_part(&csr, &A, x, y, y_res);
                const double d = now_ms();
                hthread_group_wait(thread_id);
                const double e = now_ms();
                page_finalize_y(&A, yd, y_res, y);
                const double f = now_ms();
                ov_submit[i] = b - a;
                ov_cpu[i] = d - c;
                ov_dsp[i] = e - b;
                ov_wait[i] = e - d;
                ov_fin[i] = f - e;
                ov_total[i] = f - a;
            }
            const double ov_submit_m = trimmed_mean_1each(ov_submit, PAGE_CONT_DIAG_ITER);
            const double ov_cpu_m = trimmed_mean_1each(ov_cpu, PAGE_CONT_DIAG_ITER);
            const double ov_dsp_m = trimmed_mean_1each(ov_dsp, PAGE_CONT_DIAG_ITER);
            const double ov_wait_m = trimmed_mean_1each(ov_wait, PAGE_CONT_DIAG_ITER);
            const double ov_fin_m = trimmed_mean_1each(ov_fin, PAGE_CONT_DIAG_ITER);
            const double ov_total_m = trimmed_mean_1each(ov_total, PAGE_CONT_DIAG_ITER);
            printf("[CONT_OVERLAP] mode=delay_first_tile_xload delay_ms=%.6f "
                   "submit_ms=%.6f cpu_part_ms=%.6f dsp_post_submit_endpoint_ms=%.6f "
                   "wait_after_cpu_ms=%.6f finalize_ms=%.6f total_ms=%.6f "
                   "cpu_slowdown_vs_solo=%.8f total_ratio_vs_normal=%.8f\n",
                   first_xload_ms, ov_submit_m, ov_cpu_m, ov_dsp_m, ov_wait_m,
                   ov_fin_m, ov_total_m,
                   diag_cpu_part_solo > 0.0 ? ov_cpu_m / diag_cpu_part_solo : 0.0,
                   formal_hybrid_ms > 0.0 ? ov_total_m / formal_hybrid_ms : 0.0);

            page_cont_stress_free(&sc);
            printf("[CONT] sink=%.17g\n", (double)page_cont_diag_sink);
        }

        hthread_barrier_destroy(barrier_id);
        hthread_group_destroy(thread_id);
        if (dsp_const) hthread_free(dsp_const);
        printf("[PAGE] kernel = page_spmv\n");
    }

    for (int i = 0; i < SPMV_ITER; i++) {
        printf("[ITER] impl=page iter=%d mode=%d total_ms=%.6f submit_ms=%.6f "
               "arm_rows_ms=%.6f arm_residual_ms=%.6f wait_after_arm_ms=%.6f "
               "dsp_elapsed_ms=%.6f finalize_ms=%.6f cpu_only_ms=%.6f\n",
               i + 1, (int)A.mode, iter_ms[i], submit_ms[i], arm_rows_ms[i],
               arm_res_ms[i], wait_after_arm_ms[i], dsp_ms[i], perm_ms[i],
               cpu_iter_ms[i]);
    }

    double t_spmv, t_raw, t_med, t_min, t_max, t_cv;
    time_stats(iter_ms, SPMV_ITER, &t_spmv, &t_raw, &t_med, &t_min, &t_max, &t_cv);

    if (same_part_diag && A.mode != PAGE_MODE_CPU) {
        const long long cpu_prefix_nnz = (A.row_off > 0 && st.row_pre) ? st.row_pre[A.row_off] : 0;
        const long long cpu_work_nnz = cpu_prefix_nnz + A.res_nnz;
        const double cpu_work_frac = (nnz > 0) ? (double)cpu_work_nnz / (double)nnz : 0.0;

        const double ideal_overlap_phase = fmax(diag_cpu_part_solo, diag_dsp_wait_solo);
        const double ideal_total_est = t_submit + ideal_overlap_phase + t_perm;
        const double cpu_contention = (diag_cpu_part_solo > 0.0)
                                    ? t_arm_part / diag_cpu_part_solo : 0.0;
        const double dsp_post_concurrent = t_dsp - t_submit;
        const double dsp_contention = (diag_dsp_wait_solo > 0.0)
                                    ? dsp_post_concurrent / diag_dsp_wait_solo : 0.0;
        const double parallel_tax = (ideal_overlap_phase > 0.0)
                                  ? dsp_post_concurrent / ideal_overlap_phase : 0.0;
        const double total_tax = (ideal_total_est > 0.0)
                               ? t_spmv / ideal_total_est : 0.0;
        const double ideal_speedup = (ideal_total_est > 0.0)
                                   ? diag_cpu_full / ideal_total_est : 0.0;
        const double actual_speedup = (t_spmv > 0.0)
                                    ? diag_cpu_full / t_spmv : 0.0;
        const double recoverable_gap = t_spmv - ideal_total_est;

        printf("[DIAG] partition row_off=%d cpu_prefix_nnz=%lld cpu_residual_nnz=%lld "
               "cpu_work_nnz=%lld dsp_nnz=%lld cpu_work_frac=%.8f cpu_share=%.8f\n",
               A.row_off, cpu_prefix_nnz, A.res_nnz, cpu_work_nnz, A.nnz_dsp,
               cpu_work_frac, A.cpu_share);
        printf("[DIAG] phase=hybrid cpu_part_concurrent_ms=%.6f parallel_phase_ms=%.6f "
               "submit_ms=%.6f wait_after_cpu_ms=%.6f finalize_ms=%.6f total_ms=%.6f\n",
               t_arm_part, t_dsp, t_submit, t_wait_after_arm, t_perm, t_spmv);
        printf("[DIAG] derived corrected=1 ideal_overlap_phase_ms=%.6f ideal_total_est_ms=%.6f "
               "cpu_contention_ratio=%.8f dsp_contention_endpoint_ratio=%.8f "
               "dsp_wait_after_cpu_ms=%.8f parallel_tax_ratio=%.8f total_tax_ratio=%.8f "
               "ideal_speedup_vs_cpu_full=%.8f actual_speedup_vs_cpu_full=%.8f "
               "recoverable_gap_ms=%.6f\n",
               ideal_overlap_phase, ideal_total_est, cpu_contention, dsp_contention,
               t_wait_after_arm, parallel_tax, total_tax, ideal_speedup,
               actual_speedup, recoverable_gap);
    }

    if (A.mode != PAGE_MODE_CPU) {
        const long long v9_cpu_prefix_nnz = (A.row_off > 0 && st.row_pre) ? st.row_pre[A.row_off] : 0;
        const long long v9_cpu_work_nnz = v9_cpu_prefix_nnz + A.res_nnz;
        const double v9_cpu_work_frac = (A.nnz > 0) ? (double)v9_cpu_work_nnz / (double)A.nnz : 0.0;
        printf("[DUAL_X_PROD] formal enabled=%d row_off=%d cpu_share=%.8f "
               "cpu_prefix_nnz=%lld cpu_residual_nnz=%lld cpu_work_frac=%.8f "
               "dsp_nnz=%lld submit_ms=%.6f cpu_part_ms=%.6f wait_ms=%.6f "
               "finalize_ms=%.6f total_ms=%.6f\n",
               dual_x_prod, A.row_off, A.cpu_share, v9_cpu_prefix_nnz, A.res_nnz,
               v9_cpu_work_frac, A.nnz_dsp, t_submit, t_arm_part,
               t_wait_after_arm, t_perm, t_spmv);
    }

    if (v17_csr_numa) {
        printf("[V17_CSR_NUMA] formal enabled=1 mode=%s cpu_csr=private_nohuge_runtime_interleave "
               "cpu_x=nohuge_runtime_interleave exact_halav_csr_calls=%d row_off=%d total_ms=%.6f\n",
               cpu_hybrid_side ? "cpu_hybrid_side" : "page_hybrid",
               g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR,
               A.row_off, t_spmv);
    }
    if (v18_fast_prep) {
        printf("[V18_FAST_PREP] formal unchanged=1 mode=%s exact_halav_csr_calls=%d row_off=%d total_ms=%.6f\n",
               cpu_hybrid_side ? "cpu_hybrid_side" : "page_hybrid",
               g_page_cpu_baseline == PAGE_CPU_BASELINE_HALAV_CSR,
               A.row_off, t_spmv);
    }
    if (v19_fast_shadow) {
        printf("[V19_FAST_SHADOW] formal unchanged=1 mode=page_hybrid exact_structural_gate=1 row_off=%d total_ms=%.6f shadow_ms=%.6f\n",
               A.row_off, t_spmv, t_v18_shadow_plan);
    }

    printf("[T] parallel_phase_time: %.4f (ms)\n", t_dsp);
    printf("[T] dsp_only_time: %.4f (ms)\n", t_dsp);
    printf("[T] unpermute_time: %.4f (ms)\n", t_perm);
    printf("[T] arm_rows_time: %.4f (ms)\n", t_arm_rows);
    printf("[T] arm_residual_time: %.4f (ms)\n", t_arm_res);
    printf("[T] finalize_time: %.4f (ms)\n", t_perm);
    printf("[T] cpu_only_time: %.4f (ms)\n",
           (A.mode == PAGE_MODE_CPU) ? t_spmv : t_perm);

    for (int i = 0; i < SPMV_WARMUP; i++) csr_mv(m, n, csr.rowPointers, csr.colIndices, csr.values, x, ref);
    double citer[SPMV_ITER];
    for (int i = 0; i < SPMV_ITER; i++) {
        const double ta = now_ms();
        csr_mv(m, n, csr.rowPointers, csr.colIndices, csr.values, x, ref);
        citer[i] = now_ms() - ta;
    }
    double t_csr, c_raw, c_med, c_min, c_max, c_cv;
    time_stats(citer, SPMV_ITER, &t_csr, &c_raw, &c_med, &c_min, &c_max, &c_cv);

    page_row_scale(&csr, x, scale);
    double maxrel = 0.0;
    const int bad = page_validate(y, ref, scale, m, &maxrel);
    if (same_part_diag && diag_cpu_full_y) {
        double diag_cpu_maxrel = 0.0;
        const int diag_cpu_bad = page_validate(diag_cpu_full_y, ref, scale, m, &diag_cpu_maxrel);
        printf("[DIAG] cpu_full_correct=%d cpu_full_max_rel_err=%.3e\n",
               diag_cpu_bad == 0, diag_cpu_maxrel);
    }
    if (dual_x_diag && diag_cpu_private_y && diag_dual_y) {
        double priv_rel = 0.0, dual_rel = 0.0;
        const int priv_bad = page_validate(diag_cpu_private_y, ref, scale, m, &priv_rel);
        const int dual_bad = page_validate(diag_dual_y, ref, scale, m, &dual_rel);
        printf("[DUAL_X] correctness cpu_full_private=%d cpu_full_private_max_rel_err=%.3e "
               "hybrid_dual=%d hybrid_dual_max_rel_err=%.3e\n",
               priv_bad == 0, priv_rel, dual_bad == 0, dual_rel);
        if (priv_bad != 0 || dual_bad != 0) dual_diag_bad = 1;
    }
    if (xplace_diag && diag_cpu_private_y && diag_dual_y &&
        diag_cpu_interleave_y && diag_interleave_y) {
        double ft_cpu_rel = 0.0, ft_h_rel = 0.0, il_cpu_rel = 0.0, il_h_rel = 0.0;
        const int ft_cpu_bad = page_validate(diag_cpu_private_y, ref, scale, m, &ft_cpu_rel);
        const int ft_h_bad = page_validate(diag_dual_y, ref, scale, m, &ft_h_rel);
        const int il_cpu_bad = page_validate(diag_cpu_interleave_y, ref, scale, m, &il_cpu_rel);
        const int il_h_bad = page_validate(diag_interleave_y, ref, scale, m, &il_h_rel);
        printf("[XPLACE] correctness cpu_firsttouch=%d cpu_firsttouch_max_rel_err=%.3e "
               "hybrid_firsttouch=%d hybrid_firsttouch_max_rel_err=%.3e "
               "cpu_interleave=%d cpu_interleave_max_rel_err=%.3e "
               "hybrid_interleave=%d hybrid_interleave_max_rel_err=%.3e\n",
               ft_cpu_bad == 0, ft_cpu_rel, ft_h_bad == 0, ft_h_rel,
               il_cpu_bad == 0, il_cpu_rel, il_h_bad == 0, il_h_rel);
        if (ft_cpu_bad || ft_h_bad || il_cpu_bad || il_h_bad) xplace_diag_bad = 1;
    }
    if (htx_diag && diag_cpu_hthread_y && diag_hthread_y &&
        diag_cpu_interleave_y && diag_interleave_y) {
        double ht_cpu_rel = 0.0, ht_h_rel = 0.0, il_cpu_rel = 0.0, il_h_rel = 0.0;
        const int ht_cpu_bad = page_validate(diag_cpu_hthread_y, ref, scale, m, &ht_cpu_rel);
        const int ht_h_bad = page_validate(diag_hthread_y, ref, scale, m, &ht_h_rel);
        const int il_cpu_bad = page_validate(diag_cpu_interleave_y, ref, scale, m, &il_cpu_rel);
        const int il_h_bad = page_validate(diag_interleave_y, ref, scale, m, &il_h_rel);
        printf("[HTX] correctness cpu_hthread_private=%d cpu_hthread_private_max_rel_err=%.3e "
               "hybrid_hthread_private=%d hybrid_hthread_private_max_rel_err=%.3e "
               "cpu_interleave=%d cpu_interleave_max_rel_err=%.3e "
               "hybrid_interleave=%d hybrid_interleave_max_rel_err=%.3e\n",
               ht_cpu_bad == 0, ht_cpu_rel, ht_h_bad == 0, ht_h_rel,
               il_cpu_bad == 0, il_cpu_rel, il_h_bad == 0, il_h_rel);
        if (ht_cpu_bad || ht_h_bad || il_cpu_bad || il_h_bad) htx_diag_bad = 1;
    }
    if (thp_diag && diag_cpu_interleave_y && diag_interleave_y &&
        diag_cpu_nohuge_y && diag_nohuge_y) {
        double il_cpu_rel = 0.0, il_h_rel = 0.0, nh_cpu_rel = 0.0, nh_h_rel = 0.0;
        const int il_cpu_bad = page_validate(diag_cpu_interleave_y, ref, scale, m, &il_cpu_rel);
        const int il_h_bad = page_validate(diag_interleave_y, ref, scale, m, &il_h_rel);
        const int nh_cpu_bad = page_validate(diag_cpu_nohuge_y, ref, scale, m, &nh_cpu_rel);
        const int nh_h_bad = page_validate(diag_nohuge_y, ref, scale, m, &nh_h_rel);
        printf("[THP] correctness cpu_interleave=%d cpu_interleave_max_rel_err=%.3e "
               "hybrid_interleave=%d hybrid_interleave_max_rel_err=%.3e "
               "cpu_nohuge=%d cpu_nohuge_max_rel_err=%.3e "
               "hybrid_nohuge=%d hybrid_nohuge_max_rel_err=%.3e\n",
               il_cpu_bad == 0, il_cpu_rel, il_h_bad == 0, il_h_rel,
               nh_cpu_bad == 0, nh_cpu_rel, nh_h_bad == 0, nh_h_rel);
        if (il_cpu_bad || il_h_bad || nh_cpu_bad || nh_h_bad) thp_diag_bad = 1;
    }
    printf("[PAGE] correct = %d, max_rel_err = %.3e\n", bad == 0, maxrel);
    if (bad == 0) printf("Right!\n"); else printf("%d Wrong!\n", bad);

    const double gflops     = 2.0 * (double)nnz / (1e6 * t_spmv);
    const double gflops_csr = 2.0 * (double)nnz / (1e6 * t_csr);
    double b_ddr, b_gsm;
    if (A.mode == PAGE_MODE_CPU) {
        b_ddr = (double)nnz * page_cpu_actual_seq_bytes_per_nnz(&csr)
              + (double)m * 8.0 + (double)n * 8.0;
        b_gsm = 0.0;
    } else {
        b_ddr = page_bytes_ddr(&A);
        b_gsm = page_bytes_gsm(&A);
    }
    const double bw_ddr = b_ddr / (1e6 * t_spmv);
    const double bw_gsm = b_gsm / (1e6 * t_spmv);
    const double peak   = (bw_probe_gbs > 1.0) ? bw_probe_gbs : 34.0;

    printf("[PAGE] timing_policy: warmup=%d timed=%d trim_low=1 trim_high=1 average_remaining=%d\n",
           SPMV_WARMUP, SPMV_ITER, SPMV_ITER - 2);
    printf("[PAGE] spmv_time: %.4f (ms)\n", t_spmv);
    printf("[PAGE] spmv_raw_mean: %.4f (ms)\n", t_raw);
    printf("[PAGE] spmv_median: %.4f (ms)\n", t_med);
    printf("[PAGE] spmv_min: %.4f (ms)\n", t_min);
    printf("[PAGE] spmv_max: %.4f (ms)\n", t_max);
    printf("[PAGE] spmv_cv: %.2f%%\n", t_cv);
    printf("[PAGE] csr_mv_time: %.4f (ms)\n", t_csr);
    printf("[PAGE] csr_mv_raw_mean: %.4f (ms)\n", c_raw);
    printf("[PAGE] csr_mv_median: %.4f (ms)\n", c_med);
    printf("[PAGE] csr_mv_min: %.4f (ms)\n", c_min);
    printf("[PAGE] csr_mv_max: %.4f (ms)\n", c_max);
    printf("[PAGE] csr_mv_cv: %.2f%%\n", c_cv);
    printf("[PAGE] gflops: %.4f\n", gflops);
    printf("[PAGE] gflops_csr: %.4f\n", gflops_csr);
    printf("[PAGE] speedup: %.4f\n", t_csr / t_spmv);
    printf("[PAGE] bw_ddr: %.2f (GB/s)\n", bw_ddr);
    printf("[PAGE] bw_gsm: %.2f (GB/s)\n", bw_gsm);
    printf("[PAGE] bytes_ddr: %.0f\n", b_ddr);
    printf("[PAGE] bytes_gsm: %.0f\n", b_gsm);

    printf("[PAGE] bwu: %.2f%%\n", bw_ddr / peak * 100.0);
    printf("[PAGE] bwu_gsm: %.2f%%\n",
           bw_gsm / (peak * (PAGE_BW_GSM / PAGE_BW_DDR)) * 100.0);
    printf("[PAGE] pre_ratio: %.2f\n", t_pre / t_spmv);

    if (v17_csr_numa) {
        if (cpu_hybrid_side) {
            (void)page_v17_cpu_csr_ab(&v17_original_csr, &csr, cpu_x, m, cluster_id);
        } else if (A.mode != PAGE_MODE_CPU) {
            (void)page_v17_hybrid_csr_ab(&v17_original_csr, &csr, &A, cpu_x, x,
                                          coreNum, cluster_id);
        }
    }

    if (A.mode != PAGE_MODE_CPU) page_free_matrix(&A);
    page_free_stat(&st);
    hthread_free(x);
    page_xplace_free_interleaved(&x_cpu_prod_map);
    page_v17_free_csr_mapping(&v17_csr_map);
    if (x_hthread_private_diag) hthread_free(x_hthread_private_diag);
    free(x_cpu_diag);
    page_xplace_free_interleaved(&x_interleave_diag);
    page_xplace_free_interleaved(&x_nohuge_diag);
    if (ybuf) { hthread_free(ybuf); }
    else { if (yd) hthread_free(yd); if (y) free(y); }
    free(y_res);
    free(diag_cpu_full_y);
    free(diag_cpu_private_y);
    free(diag_dual_y);
    free(diag_cpu_interleave_y);
    free(diag_interleave_y);
    free(diag_cpu_hthread_y);
    free(diag_hthread_y);
    free(diag_cpu_nohuge_y);
    free(diag_nohuge_y);
    free(ref); free(scale);
    free(csrRowPtr); free(csrColIdx); free(csrVal);
    hthread_dev_close(cluster_id);
    return (bad == 0 && dual_diag_bad == 0 && xplace_diag_bad == 0 &&
            htx_diag_bad == 0 && thp_diag_bad == 0) ? 0 : 3;
}
