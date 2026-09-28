# TensorFlow 2.16.2／2.21.0・SciPy 1.18.1 CPU GEMM ディスパッチ経路解析

読み取り解析のみ（実測・A/B・性能断定は行わない）。イシュー #2095・親 #2089「他ライブラリのコード取得・詳細解析（負けセルの原因帰属）」Phase 2。

## §1 判断サマリ

- 対象は framework-compare スコアボードの CPU セルで比較対象になっている TensorFlow（M4 Max: 2.16.2 CPU 実行・GB10: 2.21.0）と SciPy 1.18.1（`scipy.linalg.blas.sgemm`）。計測ハーネスは `docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py`。
- **本解析で判明した最重要事実**（§6・確定）: TensorFlow の oneDNN（`_MklMatMul` への eager op-rewrite）有効・無効は `TF_ENABLE_ONEDNN_OPTS` 環境変数と `DefaultOneDnnPolicy()` の既定値で決まるが、後者の aarch64 分岐は **`ARM_NEOVERSE_V1`（MIDR `implementer=0x41`・`part_num=0xd40`）専用のハードコード判定**であり、他の Arm コア（Apple Silicon・Nvidia Grace 等）は一致しない。GB10（Grace CPU）が Neoverse V1 と同一 MIDR を返すかどうかは実機の `/proc/cpuinfo` 読み取りでしか確認できないため、oneDNN 既定 OFF の結論は「ロジックは確定・実機一致は推定」とする。
- macOS arm64（M4 Max）向け公式 wheel は `.bazelrc` の `release_macos_arm64` config に `mkl_aarch64` 系 `--define` が一切現れず、`INTEL_MKL` マクロが未定義になるため `IsMklEnabled()` は常に `false`（確定。§4）。M4 Max の TF CPU GEMM は oneDNN op-rewrite を経由しない。
- Linux aarch64（GB10）向け公式 wheel は `release_arm64_linux` config が `mkl_aarch64_threadpool` を経由し `build_with_mkl_aarch64=true` → `INTEL_MKL` 定義（確定。§5）。ただし上記の Neoverse V1 専用判定により既定は OFF 寄りと推測される（推定）。
- **GB10 実測では CPU GEMM（N=256〜2048）・train・infer の全マッチセルで fandhe-ai 0.8.0 が TensorFlow 2.21.0・SciPy 1.18.1 を上回っている**（§10 表 1）。N=4096 は fandhe-ai 側の CPU 実測データが存在せず比較不能。
- **M4 Max では N=256 の CPU GEMM で fandhe-ai がわずかに負ける**（107.4 GFLOPS 対 TF 129 GFLOPS・SciPy 121 GFLOPS）。train・infer では SciPy に負ける（TF には両方とも勝つ）。負けセルは N=256 GEMM・train・infer の 3 種（§10 表 2）。**表 2 の train・infer は fandhe-ai 側を `fresh` モード値で統一している**（GEMM 行・表 1〈GB10〉と計測条件をそろえるため。fandhe-ai の `reuse` モード値を使う旧版の判定〈train は TF に対しても僅差で負け〉は §10 表 2 直後の注記を参照）。
- SciPy の BLAS リンク先は wheels.yml のビルドマトリクスから macOS arm64 で openblas 変種と accelerate 変種の**両方**がビルドされることまでは確定できたが、PyPI へ実際に公開される変種の断定はワークフロー読み取りだけでは不可能（§8・§11）。
- コードの持ち込みはなし。結論と `path:line`・タグ固定 URL のみを記録する。

## §2 対象・版・出典

| フレームワーク | 版 | 計測ホスト | 上流リポジトリ（タグ固定） | タグのコミット SHA | ライセンス |
|---|---|---|---|---|---|
| TensorFlow | 2.16.2 | M4 Max（macOS arm64） | https://github.com/tensorflow/tensorflow/tree/v2.16.2 | `810f233968cec850915324948bbbc338c97cf57f`（軽量タグ） | Apache-2.0 |
| TensorFlow | 2.21.0 | DGX Spark GB10（Linux aarch64） | https://github.com/tensorflow/tensorflow/tree/v2.21.0 | `a481b10260dfdf833a1b16007eead49c1d7febf3`（軽量タグ） | Apache-2.0 |
| SciPy | 1.18.1 | 両ホスト共通 | https://github.com/scipy/scipy/tree/v1.18.1 | タグオブジェクト `c2df8ace0d9859f090e79058b2bd2ffc584f19e6` → コミット `e4e854eaa8f18d807cd3496028e257e36caa93cc`（注釈付きタグ） | BSD-3-Clause |

- SHA は `gh api repos/<org>/<repo>/git/ref/tags/<tag>`（TF）・`gh api repos/scipy/scipy/git/tags/<tag_sha>`（SciPy の注釈付きタグを commit へ解決）で取得した。
- SciPy を計測版と同じ **1.18.1** に固定した理由: 比較対象という目的上、計測に使われた版と解析対象を一致させることを優先した（イシュー本文の「最新」を計測版に読み替え）。2026-09-28 時点で `git ls-remote --tags` により 1.18.1 が SciPy の最新安定タグであることを確認済み（1.19 系は未リリース）。
- 取得方法: `git clone` ではなく `gh api .../git/ref/tags/<tag>`・`gh api search/code`・`curl -sL https://raw.githubusercontent.com/<org>/<repo>/<tag>/<path>` の組み合わせ（リポジトリ取得・ビルド実行は一切行っていない。scratchpad へ個別ファイルのみ取得し作業後に破棄）。

## §3 TensorFlow CPU GEMM ディスパッチ（版共通の連鎖）

Python 呼び出しから CPU カーネルまでの主経路（2.16.2／2.21.0 とも同一ファイル配置。確定）:

1. `tf.matmul(a, b)`（bench_py.py の eager 呼び出し）→ `MatMul` op（rank-2 の場合。`@` 演算子〈`__matmul__`〉も同じ経路）
2. eager 実行時、`tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite.cc` が `MatMul` → `_MklMatMul` への書き換えを判定する（`IsMklEnabled()` が true の場合のみ。§5・§6）
3. 書き換えが起きない場合（既定 OFF 時）: `tensorflow/core/kernels/matmul_op_impl.h`（`LaunchMatMul` の CPU 特殊化）が `Eigen::ThreadPoolDevice` 上のテンソル contraction を呼ぶ（`Eigen/Core`・`unsupported/Eigen/CXX11/Tensor` 使用。[`matmul_op_impl.h#L30-L31`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/kernels/matmul_op_impl.h#L30-L31)）
4. Eigen contraction 自体も、ビルド時に `TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL` が定義されていれば `third_party/xla/third_party/tsl/tsl/framework/contraction/eigen_contraction_kernel.h` を経由し、さらに `TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL` が定義されていれば内部で **`dnnl_sgemm` を直接呼ぶ**（[`eigen_contraction_kernel.h#L171`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/xla/third_party/tsl/tsl/framework/contraction/eigen_contraction_kernel.h#L171)）。これは手順 2 の eager op-rewrite（`_MklMatMul` への差し替え）とは**別の、ビルド時にのみ決まる**置換経路であり、`TF_ENABLE_ONEDNN_OPTS` の実行時値とは独立である（確定・重要）。`tensorflow/core/kernels/BUILD` の `no_mkldnn_contraction_kernel` config_setting コメントは「既定でこの mkldnn 経由 sgemm が有効、明示的に `--define=tensorflow_mkldnn_contraction_kernel=0` か bazel flag で無効化する」という設計意図を記す（[`kernels/BUILD#L100-L110`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/kernels/BUILD#L100-L110)）。ただし `TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL`／`TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL` マクロを実際に定義している copts 宣言箇所（`tensorflow.bzl` の `tf_copts()` 本体には現れない）は本解析の時間内で特定できなかった。§11 へ申し送る（推定の境界）。
5. 書き換えが起きる場合: `tensorflow/core/kernels/mkl/mkl_matmul_op.cc` が oneDNN primitive を構築・実行する（`_MklMatMul` カーネル本体。ファイル内容までは深追いしていない＝スコープを「委譲境界まで」に限定）。
6. graph モード専用の `tensorflow/core/common_runtime/mkl_layout_pass.cc`・Grappler の remapper 最適化はスコープ外（bench_py.py は eager 実行のため経路に入らない。存在の指摘のみ）。

## §4 M4 Max（macOS arm64・TF 2.16.2）: oneDNN は build 時点で無効

`.bazelrc` の `build:release_macos_arm64` は `--cpu=darwin_arm64`・`--define=tensorflow_mkldnn_contraction_kernel=0`・`--macos_minimum_os=12.0` のみを設定し、`mkl`／`mkl_aarch64`／`mkl_aarch64_threadpool` のいずれの config も継承しない（[`.bazelrc#L694-L699`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/.bazelrc#L694-L699)）。

`if_mkl()`（`third_party/xla/third_party/tsl/tsl/mkl/build_defs.bzl`）は `select({"@local_tsl//tsl/mkl:build_with_mkl_aarch64": if_true, "@local_tsl//tsl:linux_x86_64": if_true, "@local_tsl//tsl:windows": if_true, "//conditions:default": if_false})`（[`build_defs.bzl#L34-L38`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/xla/third_party/tsl/tsl/mkl/build_defs.bzl#L34-L38)）であり、macOS はいずれの条件にも一致しないため `-DINTEL_MKL`（`tensorflow.bzl` の `if_mkl(["-DINTEL_MKL"])`。[`tensorflow.bzl#L460`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/tensorflow.bzl#L460)）は定義されない。

結果として `tensorflow/core/util/port.cc::IsMklEnabled()` は `#ifndef INTEL_MKL return false;` の分岐に入り、`TF_ENABLE_ONEDNN_OPTS` を設定しても意味を持たない（コンパイル時に無効化された経路のため。[`port.cc#L89-L91`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/util/port.cc#L89-L91)）。**確定**: M4 Max の TF CPU GEMM は常に Eigen `ThreadPoolDevice` 経路（§3 手順 3〜4）であり `_MklMatMul` op-rewrite（§3 手順 5）には到達しない。手順 4 の Eigen-custom-contraction-kernel（`dnnl_sgemm` 直呼び）が macOS 向けにも既定有効かは §3 の未特定事項に依存するため推定に留める。

## §5 GB10（Linux aarch64・TF 2.21.0）: oneDNN はビルドに含まれるが既定値は実機依存

`.bazelrc`（2.21.0）の `common:release_arm64_linux` は `--config=mkl_aarch64_threadpool` を経由し（[`.bazelrc#L775-L778`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/.bazelrc#L775-L778)）、`mkl_aarch64_threadpool` config は `--define=build_with_mkl_aarch64=true`（OpenMP は `--@compute_library//:openmp=false` で明示的に無効。[`.bazelrc#L290-L292`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/.bazelrc#L290-L292)）を設定する。これは §4 の `if_mkl()` selector の `build_with_mkl_aarch64` 条件に一致するため `-DINTEL_MKL` が定義される（確定）。

`tensorflow/core/util/port.cc::DefaultOneDnnPolicy()`（2.21.0 でも同一ファイル・ほぼ同一実装。§7 差分表参照）の分岐条件を要約する（[`port.cc#L108-L124`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/core/util/port.cc#L108-L124)）: `INTEL_MKL` 未定義なら無効。Google 内部ビルドなら常に有効。Windows かつ x86 なら常に有効。Linux では x86 の AVX512_VNNI／AVX512_BF16／AVX_VNNI／AMX_TILE／AMX_INT8／AMX_BF16 のいずれかを満たすか、または `TestAarch64CPU(ARM_NEOVERSE_V1)` が真の場合にのみ有効。それ以外（上記いずれにも該当しない Linux aarch64 を含む）は無効。

`TestAarch64CPU(ARM_NEOVERSE_V1)` の実体（`third_party/xla/third_party/tsl/tsl/platform/cpu_info.cc`）は `/proc/cpuinfo` 経由で `MIDR_EL1` レジスタを読み、`implementer == 0x41`（Arm Ltd.）かつ `part_num == 0xd40`（Neoverse V1 固有の part number）の場合のみ `true` を返す（[`cpu_info.cc#L441-L456`・`#L465-L473`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/xla/third_party/tsl/tsl/platform/cpu_info.cc#L441-L473)。2.21.0 でも同一実装、パスは未変更）。`ARM_NEOVERSE_N1` は enum にはあるが `TestAarch64CPU` の switch 文では判定対象外（`default: return 0`）。

GB10 は Nvidia Grace CPU（Arm ベースのカスタム SoC。plan 記載は Cortex-X925／A725 系だが Grace 系サーバー CPU は一般に Neoverse V2 ベースの Nvidia 独自実装であり、いずれにせよ Neoverse **V1** の part number `0xd40` と一致する保証はない）。よって「GB10 では oneDNN 既定 OFF」という結論は、**判定ロジック自体は確定・GB10 実機の MIDR が Neoverse V1 と一致しないという結論は推定**（実機 `cat /proc/cpuinfo` または `python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"` 相当の確認が必要。§11 へ申し送り）。`TF_ENABLE_ONEDNN_OPTS=1` を明示設定すれば既定値に関わらず oneDNN が有効化される（[`port.cc#L143-L160`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/core/util/port.cc#L143-L160)）が、bench_py.py（§7.1）はこの環境変数を設定していないため計測時は既定値に従う。

## §6 Eigen／oneDNN 選択条件のまとめ

| 条件 | 内容 | 確定／推定 |
|---|---|---|
| ビルド時: `INTEL_MKL` 定義有無 | `if_mkl()` の select 対象（`build_with_mkl_aarch64` define／`linux_x86_64`／`windows`）。macOS は非該当 | 確定（§4・§5） |
| 実行時: `IsMklEnabled()` の既定値 | `INTEL_MKL` 未定義なら常に false。定義時は `DefaultOneDnnPolicy()`（x86 CPU 機能検査 or aarch64 は Neoverse V1 のみ）→ `TF_ENABLE_ONEDNN_OPTS` で上書き可 | 確定（ロジック）／推定（GB10 実機一致） |
| Eigen contraction 自体の oneDNN sgemm 直呼び | `TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL`／`_MKLDNN_CONTRACTION_KERNEL` マクロ定義時、`INTEL_MKL`（op-rewrite 用フラグ）とは独立に発生しうる。マクロの定義元 copts は本解析で未特定 | 推定（境界。§11） |
| CPU 機能検査（x86） | `AVX512_VNNI`／`AVX512_BF16`／`AVX_VNNI`／`AMX_TILE`／`AMX_INT8`／`AMX_BF16` のいずれか | 確定（GB10／M4 Max はいずれも aarch64 のため非該当） |
| dtype・サイズによる分岐 | `mkl_matmul_op.cc` 内部の分岐は未深追い（委譲境界外） | 対象外（スコープ外） |
| graph モード（`mkl_layout_pass.cc`・Grappler remapper） | eager 実行（bench_py.py）の経路には入らない | スコープ外（存在の指摘のみ） |

## §7 TF スレッド方針

### §7.1 bench_py.py の計測条件（確定）

`docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py` の `TF.__init__`（[`bench_py.py#L125-L136`](../perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py)）は `TF_CPP_MIN_LOG_LEVEL=2` と `tf.config.set_visible_devices([], "GPU")`（CPU セル計測時）のみを設定し、`TF_NUM_INTRAOP_THREADS`／`TF_NUM_INTEROP_THREADS`／`OMP_NUM_THREADS`／`tf.config.threading.set_*_parallelism_threads` のいずれも呼んでいない。**すなわち計測は TF のスレッド既定値をそのまま使う**（環境変数によるスレッド数固定は行われていない）。

### §7.2 既定値の決め方（確定・`process_util.cc`）

| 決定要素 | 環境変数 | config API | 既定値 |
|---|---|---|---|
| inter-op 並列度 | `TF_NUM_INTEROP_THREADS`（[`process_util.cc#L94-L96`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/common_runtime/process_util.cc#L94-L96)） | `tf.config.threading.set_inter_op_parallelism_threads` | `port::MaxParallelism()`（[`process_util.cc#L43-L51`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/common_runtime/process_util.cc#L43-L51)） |
| intra-op 並列度（Eigen `ThreadPoolDevice`） | `TF_NUM_INTRAOP_THREADS`（[`process_util.cc#L100-L102`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/common_runtime/process_util.cc#L100-L102)） | `tf.config.threading.set_intra_op_parallelism_threads` | `port::MaxParallelism()`（ENABLE_ONEDNN_OPENMP かつ ENABLE_MKL の場合のみ `DefaultNumIntraOpThreads()` 経由。[`process_util.cc#L108-L121`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/common_runtime/process_util.cc#L108-L121)） |
| oneDNN（OpenMP ビルド時のみ） | `OMP_NUM_THREADS`（`ENABLE_ONEDNN_OPENMP && ENABLE_MKL` ガード） | なし | 未設定なら 0（未反映） |

`OMPThreadsFromEnvironment()`／`DefaultNumIntraOpThreads()` は `#if defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL)` でガードされている（[`process_util.cc#L104-L121`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/common_runtime/process_util.cc#L104-L121)）。GB10 の公式 wheel は `mkl_aarch64_threadpool`（`--@compute_library//:openmp=false`。§5）を使うため OpenMP 経路ではなく、`OMP_NUM_THREADS` は GB10 の計測に無関係と判断できる（確定）。

eager 実行（`tf.matmul` 直接呼び出し）のスレッドプールが `InitComputePool`（Session 用）と同一かどうかは `EagerContext` 側の配線を追い切れておらず、上記表の「既定値決定ロジック」は Session/Eager 双方が依拠する共通ユーティリティ関数であることのみ確定している（正確な eager 側呼び出し箇所は §11 へ申し送り）。

### §7.3 2.16.2 と 2.21.0 の差分

| 項目 | 2.16.2 | 2.21.0 |
|---|---|---|
| `port.cc`（`DefaultOneDnnPolicy`／`IsMklEnabled`）のファイル位置・実装 | 同一（[`port.cc`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/util/port.cc)） | 同一（`IsAArch64Available()` 関数が追加され行番号が全体的に +18 前後ずれる以外は不変。[`port.cc`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/core/util/port.cc)） |
| `process_util.cc` のスレッド既定値決定 | 同一 | 未再確認（コード全文は 2.16.2 側のみ精読。ファイル自体は同一パスに存在することのみ `search/code` で確認） |
| `.bazelrc` の `mkl_aarch64`（非 threadpool）config | `--define=build_with_acl=true` を含む（Arm Compute Library 利用。[`.bazelrc#L217-L220`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/.bazelrc#L217-L220)） | `mkl_aarch64` は `mkl_aarch64_threadpool` の単純なエイリアスとなり ACL 関連 define が `.bazelrc` から消えている（[`.bazelrc#L294-L295`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/.bazelrc#L294-L295)）。公式 Linux aarch64 wheel は両版とも `mkl_aarch64_threadpool` を直接使うため、この差分はデフォルト公開 wheel の挙動には影響しない（確定） |

## §8 SciPy BLAS 選択とスレッド

### §8.1 実行時の型選択（確定）

`scipy.linalg.blas.get_blas_funcs` → `find_best_blas_type`（[`blas.py#L289-L353`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/linalg/blas.py#L289-L353)）は渡された配列の dtype から `s`／`d`／`c`／`z` の BLAS/LAPACK プレフィックスと `prefer_fortran` フラグのみを決める。bench_py.py は `f32` 配列を渡すため `sgemm`（`_fblas` モジュールの f2py 生成関数）が選ばれる。

### §8.2 f2py ラッパーでの入力順序（確定は宣言まで・実挙動は未実測）

`scipy/linalg/fblas_l3.pyf.src` の `<prefix>gemm` 宣言（[`fblas_l3.pyf.src#L16-L48`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/linalg/fblas_l3.pyf.src#L16-L48)）は `a`・`b` を `<ftype> dimension(lda,ka), intent(in)` として宣言し、`intent(c)` を付けていない。f2py の既定挙動は「Fortran（列優先）配列を期待し、C 連続配列が渡された場合は暗黙のコピーまたは転置解釈を行う」ため、bench_py.py が `np.ascontiguousarray`（C 順）で渡す `A`・`B` に対して f2py 層でコピーが発生する可能性がある。**ただし `<prefix>gemm` の低レベル f2py ラッパーは `trans_a`／`trans_b` を明示引数として持ち、SciPy の高レベル `gemm` ドキュメントが謳う「転置フラグの入れ替えで C 順序でもコピーを避ける」最適化が `blas.py` 側のどの層で行われるか（あるいは行われないか）は、この `.pyf` 宣言だけからは断定できない**。実装が存在するとすれば `scipy/linalg/blas.py` の `get_blas_funcs` 呼び出し元（呼び出し側コード。SciPy 内部ではなく `bench_py.py` 自身がどう呼ぶかに依存）かレイヤーの外側になる。本計測（`bench_py.py#L177`）は `self.blas.sgemm(1.0, a, b)` という最も単純な呼び出しのみを行っており、`trans_a`／`trans_b` を明示していないため、`intent(c)` 非宣言の帰結として**コピーが発生している可能性が高いが未実測**（推定。GEMM セルの計測境界に直接関係するため §10 の差分候補へ計上）。

### §8.3 ビルド時の BLAS 選択

- Linux（`manylinux`）: `pyproject.toml` の `[tool.cibuildwheel.linux]`・`[tool.cibuildwheel.linux.environment]` は `PKG_CONFIG_PATH = "/project/.openblas"`（[`pyproject.toml#L157-L167`](https://github.com/scipy/scipy/blob/v1.18.1/pyproject.toml#L157-L167)）を設定しており、`scipy-openblas32`（ILP64 ではなく `openblas32` = 32-bit int の LP64 系列名。厳密な ILP64/LP64 の切替設定箇所までは未確認）へリンクすることが読み取れる。GB10（`manylinux_aarch64`）も同じ Linux ビルド経路のため OpenBLAS 系（確定寄りだが、実際のホイール metadata 未確認のため推定）。
- macOS arm64: `.github/workflows/wheels.yml` の CI マトリクスは `[macos-14, macosx, arm64, openblas, "12.3"]` と `[macos-14, macosx, arm64, accelerate, "14.0"]` の**両方**を含み（[`wheels.yml#L57-L60`](https://github.com/scipy/scipy/blob/v1.18.1/.github/workflows/wheels.yml#L57-L60)）、`accelerate` 変種は `CIBW_CONFIG_SETTINGS="setup-args=-Dblas=accelerate"` でビルドされる（[`wheels.yml#L140-L144`](https://github.com/scipy/scipy/blob/v1.18.1/.github/workflows/wheels.yml#L140-L144)）。この workflow 自体には PyPI へのアップロード（`twine`／`upload_pypi` 相当）ステップが含まれておらず（`upload-artifact` のみ確認）、**どちらの変種が実際に `pip install scipy==1.18.1` で配布されるかはこのファイルからは確定できない**（未確定。§11 へ申し送り）。
- ILP64／LP64 の判定: `pyproject.toml`・`wheels.yml` の読み取り範囲では明示的な記述を確認できず未確定（§11）。

### §8.4 スレッド（スコープ外）

SciPy 自身はスレッド数を決めない。リンク先 BLAS が決める（OpenBLAS: `OPENBLAS_NUM_THREADS`／`OMP_NUM_THREADS`、Accelerate: `VECLIB_MAXIMUM_THREADS`）。BLAS ライブラリ内部の既定スレッド数導出ロジックはスコープ外（Eigen／oneDNN／ACL／OpenBLAS／Accelerate の内部実装に踏み込まない方針。冒頭「読み取り解析のみ」の方針および §11 の非目標記載を参照）。`threadpoolctl` は SciPy がスレッドプール検出に対応していることの指摘に留める。

### §8.5 train／infer セルの注記

`bench_py.py` の SciPy 経路（[`bench_py.py#L166`](../perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py) 付近のコメント「GEMM は `scipy.linalg.blas.sgemm`、MLP は NumPy で手書き backprop」）が示すとおり、train／infer タスクで SciPy が関与するのは行列積（`sgemm`）呼び出しのみで、要素演算・加算・reduce は NumPy 側。NumPy 内部の実装はスコープ外。

## §9 fandhe との対照表

| 項目 | fandhe-ai（`crates/backend-cpu`） |
|---|---|
| バックエンド選択 | 自作 BLIS 型単一経路（`crates/backend-cpu/src/gemm_blis/`）。ISA（NEON／SME）は実行時検出（`crates/backend-cpu/src/sme_detect.rs`） |
| スレッド数の既定値・上書き | `rayon::current_num_threads()` を基準に `RAYON_NUM_THREADS` 明示設定時はそれを優先（`crates/backend-cpu/src/thread_limit.rs::effective_num_threads`。`RAYON_NUM_THREADS` 未指定時はコア種別〈`cpu_capacity`〉に基づく大コア数への丸めにフォールバック） |
| 小形状でのスレッド上限 | `crates/backend-cpu/src/small_shape_thread_cap.rs::should_cap` が仕事量 `m * n.max(N_CLAMP) * k` の閾値未満で専用の小規模スレッドプールへ切り替える（`docs/perf/cpu-gemm-small-shape-thread-cap.md`） |
| スレッドのアフィニティ | `crates/backend-cpu/src/gb10_affinity.rs`（GB10 の `cpu_capacity` 誤検出〈#1364〉対策。`docs/backend-cpu-gb10-affinity-design.md`） |
| 出力確保・パッキング | `crates/backend-cpu/src/gemm_blis/partition.rs`（パーティショニング）・`cache_params.rs`（タイルサイズ）・`microkernel.rs`／`microkernel/neon.rs`／`microkernel/sme.rs`（マイクロカーネル） |

TF（Eigen `ThreadPoolDevice` またはビルド構成次第で oneDNN `dnnl_sgemm`）・SciPy（薄い f2py ラッパー越しの OpenBLAS／Accelerate 直呼び）はいずれも汎用スレッドプール・汎用 BLAS 実装に依存するのに対し、fandhe-ai は GB10／M4 Max の実測（`docs/perf/cpu-gemm-*.md` 群）に基づく専用チューニング（小形状キャップ・アフィニティ・固定コスト補正）を持つ。この構造差が §10 の負けセル・勝ちセルの原因候補になりうるが、本 doc では断定しない。

## §10 負けセルとの紐付け

### 表 1: GB10（DGX Spark。TF 2.21.0・SciPy 1.18.1・fandhe-ai 0.8.0）

出典: `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/results-dgx-py-0.8.0.jsonl`（TF／SciPy）・`docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/results-dgx-0.8.0.jsonl`（fandhe-ai。`"device":"cpu"` 行）。

| タスク | N | fandhe-ai (GFLOPS / ms) | TensorFlow 2.21.0 | SciPy 1.18.1 | 判定 |
|---|---|---|---|---|---|
| gemm | 256 | 145.9 GF | 59.6 GF | 61.9 GF | fandhe-ai 勝ち |
| gemm | 512 | 180.2 GF | 86.7 GF | 107.4 GF | fandhe-ai 勝ち |
| gemm | 1024 | 525.7 GF | 235.2 GF | 187.0 GF | fandhe-ai 勝ち |
| gemm | 2048 | 1020.9 GF | 603.8 GF | 338.8 GF | fandhe-ai 勝ち |
| gemm | 4096 | データなし | 944.4 GF | 457.4 GF | 比較不能（fandhe-ai 側 CPU N=4096 未計測） |
| train | 64 | 0.892 ms | 2.720 ms | 1.198 ms | fandhe-ai 勝ち |
| infer | 64 | 0.178 ms | 0.913 ms | 0.600 ms | fandhe-ai 勝ち |

GB10 では本解析範囲のマッチセルすべてで fandhe-ai が両 FW を上回る（**負けセルなし**）。`gemm` 行は全行 `parity_fail_count: 0`（判定可能。出典 JSONL の `gemm` 行のみが同フィールドを持つ）。`train`／`infer` 行の出典 JSONL（`results-dgx-py-0.8.0.jsonl`）には `parity_fail_count` フィールド自体が存在せず（fandhe-ai 側 `train`／`infer` 出力にも同様に存在しない設計）、判定不能（parity 検査の対象外）である——ゼロ fail の記述は `gemm` 行に限る。

### 表 2: M4 Max（TF 2.16.2・SciPy 1.18.1 は `gen_1988.py::M4_PY` 転記値・fandhe-ai は `results-m4max-0.8.0.jsonl`）

出典: `docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.py::M4_PY`（TF／SciPy。2026-09-12 ページからの転記と明記されている。`mode: fresh` で記録。[`gen_1988.py#L101-L107`](../perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.py)）・`scripts/bench/framework-compare/results/raw/results-m4max-0.8.0.jsonl`（fandhe-ai。`fresh` モード。表 1〈GB10〉・本表の GEMM 行と計測条件をそろえるため train・infer も `fresh` で統一する）。

| タスク | N | fandhe-ai (GFLOPS / ms) | TensorFlow 2.16.2 | SciPy 1.18.1 | 判定 |
|---|---|---|---|---|---|
| gemm | 256 | 107.4 GF | 129 GF | 121 GF | **fandhe-ai 負け（対 TF・対 SciPy 双方）** |
| gemm | 512 | 420.8 GF | 282 GF | 227 GF | fandhe-ai 勝ち |
| gemm | 1024 | 739.2 GF | 471 GF | 372 GF | fandhe-ai 勝ち |
| gemm | 2048 | 999.4 GF | 699 GF | 470 GF | fandhe-ai 勝ち |
| train | 64 | 0.835 ms（fresh） | 0.96 ms | 0.31 ms | **fandhe-ai 負け（対 SciPy のみ。対 TF は勝ち）** |
| infer | 64 | 0.178 ms（fresh） | 0.266 ms | 0.164 ms | **fandhe-ai 負け（対 SciPy のみ。対 TF は勝ち）** |

M4 Max は N=256 GEMM・train・infer の 3 セルで負け（いずれも対 SciPy のみで、対 TF はすべて勝ち）。0.9.0 版の M4 実測 JSONL は本リポジトリに未収録（Mac 側の実測申し送り事項。MEMORY.md 記載どおり）のため、0.8.0 データを参照値として使用している。

**train・infer セルのモード選択について（注記）**: 本表の train・infer は fandhe-ai 側を `fresh` モード値（train 0.835ms・infer 0.178ms）で統一している。理由は GEMM 行（本表・表 1 とも `fresh`）・表 1（GB10。train・infer を含め fandhe-ai・TF・SciPy とも `fresh` 同士）と計測条件をそろえるため。参考として、同じ JSONL には fandhe-ai の `reuse` モード値（train 1.000ms・infer 0.195ms）も存在し、`docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.py` のスコアボード本体は `reuse` を主表示列に採用している（`me = data.get((..., 'reuse'))`。[`gen_1988.py#L146-L147`](../perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.py)）が、TF／SciPy 側の値（`M4_PY`）は `fresh` としてのみ記録されており `reuse` 相当の値がそもそも存在しない（transcribed 転記データのため実測が 1 モードのみ）。`reuse` 値を採用すると train は TF 2.16.2（0.96ms）に対しても僅差で負けに転じる（1.000ms > 0.96ms）が、本 doc は fandhe-ai・TF・SciPy 間で計測条件（`fresh` 同士）をそろえることを優先し、上表を主判定として採用する。

## §11 限界・申し送り

- 到達範囲はディスパッチ層まで。Eigen の contraction 実装本体・oneDNN（`dnnl_sgemm`）内部・Arm Compute Library・OpenBLAS・Accelerate の内部カーネル実装には踏み込んでいない。
- graph モード（`mkl_layout_pass.cc`・Grappler remapper）は対象外（bench_py.py は eager のみ）。NumPy・Eigen・oneDNN・ACL・OpenBLAS・Accelerate 内部の実装詳細は非目標。
- 実機構成の確認が必要な項目（Phase 3 への申し送り）:
  - GB10 実機で `python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"` 相当・`cat /proc/cpuinfo`（MIDR 由来の `implementer`／`part_num`）を確認し、§5 の Neoverse V1 判定結果を確定させる
  - GB10・M4 Max 実機で `python -c "import scipy; scipy.show_config()"` を実行し、§8.3 の BLAS 変種（OpenBLAS／Accelerate・ILP64／LP64）を確定させる
  - §3 手順 4 の `TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL`／`TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL` マクロの定義元 copts 箇所の特定
  - §7.2 の eager 実行時スレッドプール配線（`EagerContext` 側）の追跡
  - §8.2 の f2py `sgemm` 呼び出しにおける実際のコピー発生有無（`np.ascontiguousarray` 入力に対する f2py 層の挙動）の実測確認
- Phase 3（負けセルの原因確定・A/B・性能実測）は本 doc の対象外。§10 の表 1・表 2 は既存 JSONL・転記データの突合結果であり、新規計測は一切行っていない。

## §12 ライセンス記録

- **TensorFlow**: Apache License 2.0。タグ `v2.16.2`／`v2.21.0` 時点の `LICENSE` ファイルで確認（リポジトリルート）。
- **SciPy**: BSD 3-Clause License。タグ `v1.18.1` 時点の `LICENSE.txt` で確認（リポジトリルート）。
- 読み取りで言及したが本リポジトリへは何も取り込んでいないもの: Eigen（MPL-2.0）・oneDNN（Apache-2.0）・Arm Compute Library（MIT）・OpenBLAS（BSD-3-Clause）。
- 本 doc は上流コードの逐語引用を含まず、結論・識別子名・`path:line`・出典 URL のみを記録している。本リポジトリへ何も取り込んでいないため、`docs/license-matrix.md`・`.claude/rules/deps-policy.md` の対象外。

## §13 出典

### 上流ファイル一覧（タグ固定 URL）

- TensorFlow 2.16.2: [`tensorflow/core/util/port.cc`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/util/port.cc)・[`tensorflow/core/util/util.cc`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/util/util.cc)・[`tensorflow/core/common_runtime/process_util.cc`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/common_runtime/process_util.cc)・[`tensorflow/core/kernels/matmul_op_impl.h`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/kernels/matmul_op_impl.h)・[`tensorflow/core/kernels/BUILD`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/kernels/BUILD)・[`third_party/xla/third_party/tsl/tsl/framework/contraction/eigen_contraction_kernel.h`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/xla/third_party/tsl/tsl/framework/contraction/eigen_contraction_kernel.h)・[`third_party/xla/third_party/tsl/tsl/platform/cpu_info.h`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/xla/third_party/tsl/tsl/platform/cpu_info.h)・[`third_party/xla/third_party/tsl/tsl/platform/cpu_info.cc`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/xla/third_party/tsl/tsl/platform/cpu_info.cc)・[`third_party/xla/third_party/tsl/tsl/mkl/build_defs.bzl`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/xla/third_party/tsl/tsl/mkl/build_defs.bzl)・[`third_party/mkl/build_defs.bzl`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/mkl/build_defs.bzl)・[`tensorflow/tensorflow.bzl`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/tensorflow.bzl)・[`.bazelrc`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/.bazelrc)
- TensorFlow 2.21.0: [`tensorflow/core/util/port.cc`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/core/util/port.cc)・[`.bazelrc`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/.bazelrc)
- SciPy 1.18.1: [`scipy/linalg/blas.py`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/linalg/blas.py)・[`scipy/linalg/fblas_l3.pyf.src`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/linalg/fblas_l3.pyf.src)・[`pyproject.toml`](https://github.com/scipy/scipy/blob/v1.18.1/pyproject.toml)・[`.github/workflows/wheels.yml`](https://github.com/scipy/scipy/blob/v1.18.1/.github/workflows/wheels.yml)

### 本リポジトリ側の参照 doc・データ

- `docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py`（計測ハーネス）
- `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/results-dgx-py-0.8.0.jsonl`・`results-dgx-0.8.0.jsonl`（GB10 実測 JSONL）
- `docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.py`（M4 Max 転記データ `M4_PY`）
- `scripts/bench/framework-compare/results/raw/results-m4max-0.8.0.jsonl`（M4 Max fandhe-ai 実測 JSONL）
- `crates/backend-cpu/src/gemm_blis/`・`sme_detect.rs`・`thread_limit.rs`・`small_shape_thread_cap.rs`・`gb10_affinity.rs`（fandhe 側対照）
- `docs/backend-cpu-gb10-affinity-design.md`・`docs/cpu-matmul-fixed-cost-design.md`・`docs/perf/cpu-gemm-small-shape-thread-cap.md`（既存判定記録との突合対象。今回の差分候補〈§8.2 の f2py コピー有無・§5 の Neoverse V1 MIDR 一致確認〉は上記いずれの記録にも未収録の新規項目であり、既存 REJECT／undetermined 記録との重複はない）

### 既存記録との突合結果（Step 6）

`docs/perf/cpu-gemm-default-thread-limit.md`・`docs/perf/cpu-gemm-small-shape-thread-cap.md`・`docs/perf/cpu-gemm-gb10-affinity-ab.md`・`docs/perf/cpu-gemm-blocking-sweep.md`・`docs/perf/cpu-gemm-2d-dynamic-variant.md`・`docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`・`docs/perf/cpu-gemm-ic-dynamic-variant.md`・`docs/perf/cpu-gemm-sme-fmopa-microkernel.md`・`docs/perf/cpu-gemm-neon-b-laneq-fma.md`・`docs/perf/cpu-matmul-fixed-cost-impl.md`・`docs/perf/cpu-mse-backward-sequential-threshold.md`・`docs/perf/cpu-gemm-candle-cpu-retune.md`・`docs/perf/cpu-gemm-candle-gate-remeasurement.md`（以上 `docs/perf/` 配下）・`docs/cpu-gemm-prefetch-decision.md`・`docs/cpu-gemm-b-packing-sharing-decision.md`・`docs/cpu-gemm-2d-dynamic-partition-design.md`（以上 `docs/perf/` ではなく `docs/` 直下）はいずれも fandhe-ai 自身の CPU GEMM 最適化記録であり、TensorFlow／SciPy のディスパッチ経路そのものを扱ったものではないため、本 doc の差分候補（§5 の Neoverse V1 判定・§8.2 の f2py コピー有無）と重複する既存判定は確認できなかった（`git grep -l -E "REJECT|undetermined|判定不能" -- docs` によるスコープ全体の突合を含む）。Phase 3（#2098）への入力として §11 の各項目をそのまま引き継ぐ。
