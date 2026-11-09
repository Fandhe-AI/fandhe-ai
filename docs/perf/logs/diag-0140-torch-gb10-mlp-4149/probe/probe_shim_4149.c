/* イシュー #4149: GB10 の PyTorch が CPU で sgemm_／cblas_sgemm を、CUDA で cuBLAS の GEMM 系を呼ぶ回数と転送先を数える観測専用 shim。
 * 役割: LD_PRELOAD で BLAS／cuBLAS の GEMM 入口を横取りし、呼び出し回数・先頭の (m,n,k)・転送先ライブラリを記録する。
 *       計算は元の関数へ引数そのままで転送する（時間計測には使わない。計時腕では LD_PRELOAD を設定しない）。
 * 呼び出し元: probe/probe_driver_4149.py（回数カウンタ fandhe_probe_count を ctypes で読む）と probe/run_probe.sh。
 * 派生元: docs/perf/logs/cpu-gemm-resident-reuse-3646/gb10-gcpuinf/blas_probe/sgemm_probe.c（#3646）。
 * 依存を増やさない（gcc と libdl のみ。リポジトリのビルド対象外・計測値の生成には関与しない）。
 * cuBLAS は宣言なしの void* 引数で ABI を合わせる（ヘッダ非依存。soname は固定の &'static 相当の文字列だけ）。 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>

enum { C_SGEMM_, C_CBLAS, C_CUBLAS_SGEMM, C_CUBLAS_GEMMEX, C_CUBLASLT, C_N };
static unsigned long counts[C_N];
static int first_m[C_N][4], first_n[C_N][4], first_k[C_N][4];
static const char *next_lib[C_N] = {"unresolved", "unresolved", "unresolved", "unresolved", "unresolved"};

typedef void (*sgemm_fn)(const char *, const char *, const int *, const int *, const int *, const float *,
                         const float *, const int *, const float *, const int *, const float *, float *, const int *);
typedef void (*cblas_sgemm_fn)(int, int, int, int, int, int, float, const float *, int, const float *, int, float, float *, int);
typedef int (*cublas_sgemm_fn)(void *, int, int, int, int, int, const float *, const float *, int, const float *, int,
                               const float *, float *, int);
typedef int (*cublas_gemmex_fn)(void *, int, int, int, int, int, const void *, const void *, int, int, const void *, int, int,
                                const void *, void *, int, int, int, int);
typedef int (*cublaslt_matmul_fn)(void *, void *, const void *, const void *, void *, const void *, void *, const void *,
                                  const void *, void *, void *, void *, const void *, void *, size_t, void *);

/* torch._C は RTLD_LOCAL で読まれるため、その依存の NVPL／cuBLAS は RTLD_NEXT から見えないことがある。
 * ロード済みの固定 soname を RTLD_NOLOAD で引く（新規ロードはしない）。見つからなければ RTLD_NEXT。 */
static void *find_next(const char *const *sonames, const char *sym) {
  for (; *sonames; sonames++) {
    void *h = dlopen(*sonames, RTLD_NOW | RTLD_NOLOAD);
    void *p = h ? dlsym(h, sym) : 0;
    if (p) return p;
  }
  return dlsym(RTLD_NEXT, sym);
}
static const char *const NVPL[] = {"libnvpl_blas_lp64_gomp.so.0", 0};
static const char *const CUBLAS[] = {"libcublas.so.13", "libcublas.so.12", 0};
static const char *const CUBLASLT[] = {"libcublasLt.so.13", "libcublasLt.so.12", 0};

static const char *lib_of(void *sym) {
  Dl_info info;
  if (sym && dladdr(sym, &info) && info.dli_fname) return info.dli_fname;
  return "unresolved";
}
static void note(int id, int m, int n, int k) {
  unsigned long c = counts[id]++;
  if (c < 4) { first_m[id][c] = m; first_n[id][c] = n; first_k[id][c] = k; }
}

void sgemm_(const char *ta, const char *tb, const int *m, const int *n, const int *k, const float *alpha, const float *a,
            const int *lda, const float *b, const int *ldb, const float *beta, float *c, const int *ldc) {
  static sgemm_fn next;
  if (!next) { next = (sgemm_fn)find_next(NVPL, "sgemm_"); next_lib[C_SGEMM_] = lib_of((void *)next); }
  note(C_SGEMM_, *m, *n, *k);
  next(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cblas_sgemm(int layout, int ta, int tb, int m, int n, int k, float alpha, const float *a, int lda, const float *b,
                 int ldb, float beta, float *c, int ldc) {
  static cblas_sgemm_fn next;
  if (!next) { next = (cblas_sgemm_fn)find_next(NVPL, "cblas_sgemm"); next_lib[C_CBLAS] = lib_of((void *)next); }
  note(C_CBLAS, m, n, k);
  next(layout, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

int cublasSgemm_v2(void *h, int ta, int tb, int m, int n, int k, const float *alpha, const float *a, int lda, const float *b,
                   int ldb, const float *beta, float *c, int ldc) {
  static cublas_sgemm_fn next;
  if (!next) { next = (cublas_sgemm_fn)find_next(CUBLAS, "cublasSgemm_v2"); next_lib[C_CUBLAS_SGEMM] = lib_of((void *)next); }
  note(C_CUBLAS_SGEMM, m, n, k);
  return next(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

int cublasGemmEx(void *h, int ta, int tb, int m, int n, int k, const void *alpha, const void *a, int at, int lda, const void *b,
                 int bt, int ldb, const void *beta, void *c, int ct, int ldc, int compute, int algo) {
  static cublas_gemmex_fn next;
  if (!next) { next = (cublas_gemmex_fn)find_next(CUBLAS, "cublasGemmEx"); next_lib[C_CUBLAS_GEMMEX] = lib_of((void *)next); }
  note(C_CUBLAS_GEMMEX, m, n, k);
  return next(h, ta, tb, m, n, k, alpha, a, at, lda, b, bt, ldb, beta, c, ct, ldc, compute, algo);
}

int cublasLtMatmul(void *lt, void *desc, const void *alpha, const void *a, void *adesc, const void *b, void *bdesc,
                   const void *beta, const void *c, void *cdesc, void *d, void *ddesc, const void *algo, void *ws,
                   size_t wssize, void *stream) {
  static cublaslt_matmul_fn next;
  if (!next) { next = (cublaslt_matmul_fn)find_next(CUBLASLT, "cublasLtMatmul"); next_lib[C_CUBLASLT] = lib_of((void *)next); }
  note(C_CUBLASLT, 0, 0, 0);
  return next(lt, desc, alpha, a, adesc, b, bdesc, beta, c, cdesc, d, ddesc, algo, ws, wssize, stream);
}

/* Python（ctypes）から step 前後の差分を読むための入口。 */
unsigned long fandhe_probe_count(int id) { return (id >= 0 && id < C_N) ? counts[id] : 0; }
const char *fandhe_probe_next_lib(int id) { return (id >= 0 && id < C_N) ? next_lib[id] : "invalid"; }
int fandhe_probe_first(int id, int idx, int which) {
  if (id < 0 || id >= C_N || idx < 0 || idx >= 4) return -1;
  return which == 0 ? first_m[id][idx] : which == 1 ? first_n[id][idx] : first_k[id][idx];
}
