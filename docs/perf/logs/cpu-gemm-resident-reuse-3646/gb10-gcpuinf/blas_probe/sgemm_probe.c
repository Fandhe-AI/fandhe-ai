/* イシュー #3646 G-CPU-INF (iii): PyTorch の CPU mm が実行時に sgemm_ を呼ぶか・解決先がどのライブラリかを確認する観測専用 shim。
 * 役割: LD_PRELOAD で sgemm_／cblas_sgemm を横取りし、呼び出し回数・先頭の (m,n,k)・転送先（RTLD_NEXT）の
 *       ライブラリパスをプロセス終了時に stderr へ出す。計算は元の関数へそのまま転送する（時間計測には使わない）。
 * 呼び出し元: gb10-gcpuinf/blas_probe/run_probe.sh（perf が perf_event_paranoid=4 で使えないための代替確認）。
 * 依存を増やさない（gcc と libdl のみ。リポジトリのビルド対象外・計測値の生成には関与しない）。 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

typedef void (*sgemm_fn)(const char *, const char *, const int *, const int *, const int *, const float *,
                         const float *, const int *, const float *, const int *, const float *, float *, const int *);
typedef void (*cblas_sgemm_fn)(int, int, int, int, int, int, float, const float *, int, const float *, int, float, float *, int);

static unsigned long n_sgemm, n_cblas;
static int first_m[8], first_n[8], first_k[8];
static const char *next_lib_sgemm = "unresolved";
static const char *next_lib_cblas = "unresolved";

/* torch._C は RTLD_LOCAL で読まれるため、その依存の NVPL は RTLD_NEXT から見えない。ロード済みの固定 soname を
 * RTLD_NOLOAD で引く（新規ロードはしない）。見つからなければ RTLD_NEXT。 */
static void *find_next(const char *sym) {
  void *h = dlopen("libnvpl_blas_lp64_gomp.so.0", RTLD_NOW | RTLD_NOLOAD);
  void *p = h ? dlsym(h, sym) : 0;
  return p ? p : dlsym(RTLD_NEXT, sym);
}

static const char *lib_of(void *sym) {
  Dl_info info;
  if (sym && dladdr(sym, &info) && info.dli_fname) return info.dli_fname;
  return "unresolved";
}

void sgemm_(const char *ta, const char *tb, const int *m, const int *n, const int *k, const float *alpha, const float *a,
            const int *lda, const float *b, const int *ldb, const float *beta, float *c, const int *ldc) {
  static sgemm_fn next;
  if (!next) {
    next = (sgemm_fn)find_next("sgemm_");
    next_lib_sgemm = lib_of((void *)next);
  }
  if (n_sgemm == 0) { static const char msg[] = "PROBE sgemm_ entered\n"; (void)!write(2, msg, sizeof msg - 1); }
  if (n_sgemm < 8) { first_m[n_sgemm] = *m; first_n[n_sgemm] = *n; first_k[n_sgemm] = *k; }
  n_sgemm++;
  next(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cblas_sgemm(int layout, int ta, int tb, int m, int n, int k, float alpha, const float *a, int lda, const float *b,
                 int ldb, float beta, float *c, int ldc) {
  static cblas_sgemm_fn next;
  if (!next) {
    next = (cblas_sgemm_fn)find_next("cblas_sgemm");
    next_lib_cblas = lib_of((void *)next);
  }
  if (n_cblas == 0) { static const char msg[] = "PROBE cblas_sgemm entered\n"; (void)!write(2, msg, sizeof msg - 1); }
  n_cblas++;
  next(layout, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

__attribute__((destructor)) static void report(void) {
  fprintf(stderr, "PROBE sgemm_ calls=%lu next_lib=%s\n", n_sgemm, next_lib_sgemm);
  fprintf(stderr, "PROBE cblas_sgemm calls=%lu next_lib=%s\n", n_cblas, next_lib_cblas);
  for (unsigned long i = 0; i < n_sgemm && i < 8; i++)
    fprintf(stderr, "PROBE sgemm_ first[%lu] m=%d n=%d k=%d\n", i, first_m[i], first_n[i], first_k[i]);
}
