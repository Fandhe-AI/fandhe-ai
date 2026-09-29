# TensorFlow 2.16.2／2.21.0・SciPy 1.18.1 CPU GEMM ディスパッチ経路解析

読み取り解析のみ（実測・A/B・性能断定は行わない）。イシュー #2095・親 #2089「他ライブラリのコード取得・詳細解析（負けセルの原因帰属）」Phase 2。

## §1 判断サマリ

- 対象は framework-compare スコアボードの CPU セルで比較対象になっている TensorFlow（M4 Max: 2.16.2 CPU 実行・GB10: 2.21.0）と SciPy 1.18.1（`scipy.linalg.blas.sgemm`）。計測ハーネスは `docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py`（GB10 は同ディレクトリの実測 JSONL で確認できる。M4 Max の TF／SciPy は元 JSONL が未収録の転記値で、同一ハーネスという根拠はスコアボードの記載〈`docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/body_1988.html:9`〉に留まる。§10）。
- **本解析で判明した最重要事実**（§6・確定）: TensorFlow の oneDNN（`_MklMatMul` への eager op-rewrite）有効・無効は `TF_ENABLE_ONEDNN_OPTS` 環境変数と `DefaultOneDnnPolicy()` の既定値で決まるが、後者の aarch64 分岐は **`ARM_NEOVERSE_V1`（MIDR `implementer=0x41`・`part_num=0xd40`）専用のハードコード判定**であり、他の Arm コア（Apple Silicon・GB10 の Cortex-X925／Cortex-A725 等）は一致しない。判定に使う MIDR は `/sys/devices/system/cpu/cpu<N>/regs/identification/midr_el1`（`N` は `/sys/devices/system/cpu/present` の先頭）から読まれる（§5）。GB10 の `midr_el1` は既存実測ログ `docs/perf/logs/cpu-gemm-rayon-sweep-1305/env_info_raw-dgx.txt:36-44` に記録済みで（cpu0: part `0xd87`〈Cortex-A725〉・cpu5: part `0xd85`〈Cortex-X925〉）、いずれも `0xd40` と一致しない。よって **GB10 の oneDNN op-rewrite は `TF_ENABLE_ONEDNN_OPTS` 未設定時は既定 OFF（確定。判定ロジックは上流コード、MIDR 値は既存実測ログ）**。
- macOS arm64（M4 Max）向け公式 wheel は `.bazelrc` の `release_macos_arm64` config に `mkl_aarch64` 系 `--define` が一切現れず、`INTEL_MKL` マクロが未定義になるため `IsMklEnabled()` は常に `false`（確定。§4）。M4 Max の TF CPU GEMM は oneDNN op-rewrite を経由しない。
- Linux aarch64（GB10）向け公式 wheel は `release_arm64_linux` config が `mkl_aarch64_threadpool` を経由し `build_with_mkl_aarch64=true` → `INTEL_MKL` 定義（確定。§5）。ただし上記の Neoverse V1 専用判定と GB10 の実測 MIDR により、`TF_ENABLE_ONEDNN_OPTS` を設定しない本計測（bench_py.py。§5）では oneDNN op-rewrite は既定 OFF（確定。§5）。
- **比較可能な範囲**（§10「比較可能範囲」で両ハーネスの計測コードを突き合わせた結果）: GEMM は同一入力・同一 checksum 定義で、GB10 は両側 `parity_fail_count: 0` かつ checksum も相互に一致するため比較可能。infer は演算列・GEMM 回数・checksum 定義が一致し、違いは初期パラメータの値だけなので**条件付きで比較可能**（checksum はフレームワーク間の数値一致の確認に使えない）。train は fandhe-ai 0.8.0 の backward だけが第 1 層の入力勾配 dX を計算し、backward の GEMM が 4 回対 3 回で処理が同等といえないため、**勝ち負けの判定から外し参考値（同等性は未確認）とする**。
- **GB10 実測では判定対象セル（CPU GEMM N=256〜4096・infer）すべてで fandhe-ai 0.8.0 が TensorFlow 2.21.0・SciPy 1.18.1 を上回っている**（§10 表 1。infer は条件付き。比較値はすべて `fresh` モード同士）。train は参考値で判定しない。N=4096 の fandhe-ai 値は本計測 JSONL ではなく同キャンペーンの追加計測 `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/results-dgx-0.8.0-extra.jsonl:6`（`fresh`・1128.2 GFLOPS）で、TF 944.4 GFLOPS・SciPy 457.4 GFLOPS（`results-dgx-py-0.8.0.jsonl:12`・`:19`。いずれも `fresh`）を上回る。
- **M4 Max では N=256 の CPU GEMM で fandhe-ai がわずかに負ける**（107.4 GFLOPS 対 TF 129 GFLOPS・SciPy 121 GFLOPS）。infer は SciPy に負ける（条件付き。TF には勝つ）。負けセルは N=256 GEMM と infer（対 SciPy）の 2 セルで、train は参考値のため負けセルに数えない（§10 表 2）。M4 Max の TF／SciPy は元 JSONL がリポジトリに無い転記値で、GEMM の parity と train・infer の checksum を本リポジトリ内で再確認できない（§10）。**表 2 の fandhe-ai 側は `fresh` モード値で統一している**（GEMM 行・表 1〈GB10〉と計測条件をそろえるため。`reuse` モード値の扱いは §10 表 2 直後の注記を参照）。
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

1. `tf.matmul(a, b)`（bench_py.py の GEMM タスクの eager 呼び出し。[`bench_py.py#L142-L144`](../perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py)）→ `MatMul` op（rank-2 の場合。train／infer タスクが使う `@` 演算子〈`__matmul__`。`bench_py.py#L153`・`#L163`〉も同じ経路）
2. eager 実行時、`tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite.cc` が `MatMul` → `_MklMatMul` への書き換えを判定する（`IsMklEnabled()` が true の場合のみ。§5・§6）
3. 書き換えが起きない場合（既定 OFF 時）: `tensorflow/core/kernels/matmul_op_impl.h`（`LaunchMatMul` の CPU 特殊化）が `Eigen::ThreadPoolDevice` 上のテンソル contraction を呼ぶ（`Eigen/Core`・`unsupported/Eigen/CXX11/Tensor` 使用。[`matmul_op_impl.h#L30-L31`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/kernels/matmul_op_impl.h#L30-L31)）
4. Eigen contraction 自体も、ビルド時に `TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL` が定義されていれば `third_party/xla/third_party/tsl/tsl/framework/contraction/eigen_contraction_kernel.h` を経由し、さらに `TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL` が定義されていれば内部で **`dnnl_sgemm` を直接呼ぶ**（[`eigen_contraction_kernel.h#L171`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/xla/third_party/tsl/tsl/framework/contraction/eigen_contraction_kernel.h#L171)）。これは手順 2 の eager op-rewrite（`_MklMatMul` への差し替え）とは**別の、ビルド時にのみ決まる**置換経路であり、`TF_ENABLE_ONEDNN_OPTS` の実行時値とは独立である（確定・重要）。`tensorflow/core/kernels/BUILD` の `no_mkldnn_contraction_kernel` config_setting コメントは「既定でこの mkldnn 経由 sgemm が有効、明示的に `--define=tensorflow_mkldnn_contraction_kernel=0` か bazel flag で無効化する」という設計意図を記す（[`kernels/BUILD#L100-L110`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/kernels/BUILD#L100-L110)）。ただし `TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL`／`TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL` マクロを実際に定義している copts 宣言箇所（`tensorflow.bzl` の `tf_copts()` 本体には現れない）は本解析の時間内で特定できなかった。§11 へ申し送る（推定の境界）。
5. 書き換えが起きる場合: `tensorflow/core/kernels/mkl/mkl_matmul_op.cc` が oneDNN primitive を構築・実行する（`_MklMatMul` カーネル本体。ファイル内容までは深追いしていない＝スコープを「委譲境界まで」に限定）。
6. graph モード専用の `tensorflow/core/common_runtime/mkl_layout_pass.cc`・Grappler の remapper 最適化はスコープ外（bench_py.py は eager 実行のため経路に入らない。存在の指摘のみ）。

## §4 M4 Max（macOS arm64・TF 2.16.2）: oneDNN は build 時点で無効

`.bazelrc` の `build:release_macos_arm64` は `--cpu=darwin_arm64`・`--define=tensorflow_mkldnn_contraction_kernel=0`・`--macos_minimum_os=12.0` のみを設定し、`mkl`／`mkl_aarch64`／`mkl_aarch64_threadpool` のいずれの config も継承しない（[`.bazelrc#L694-L699`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/.bazelrc#L694-L699)）。

`if_mkl()`（`third_party/xla/third_party/tsl/tsl/mkl/build_defs.bzl`）は `select({"@local_tsl//tsl/mkl:build_with_mkl_aarch64": if_true, "@local_tsl//tsl:linux_x86_64": if_true, "@local_tsl//tsl:windows": if_true, "//conditions:default": if_false})`（[`build_defs.bzl#L34-L38`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/xla/third_party/tsl/tsl/mkl/build_defs.bzl#L34-L38)）であり、macOS はいずれの条件にも一致しないため `-DINTEL_MKL`（`tensorflow.bzl` の `if_mkl(["-DINTEL_MKL"])`。[`tensorflow.bzl#L460`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/tensorflow.bzl#L460)）は定義されない。

結果として `tensorflow/core/util/port.cc::IsMklEnabled()` は `#ifndef INTEL_MKL return false;` の分岐に入り、`TF_ENABLE_ONEDNN_OPTS` を設定しても意味を持たない（コンパイル時に無効化された経路のため。[`port.cc#L89-L91`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/util/port.cc#L89-L91)）。**確定**: M4 Max の TF CPU GEMM は常に Eigen `ThreadPoolDevice` 経路（§3 手順 3〜4）であり `_MklMatMul` op-rewrite（§3 手順 5）には到達しない。手順 4 の Eigen-custom-contraction-kernel（`dnnl_sgemm` 直呼び）についても、`release_macos_arm64` config が `--define=tensorflow_mkldnn_contraction_kernel=0` を明示指定している（本節冒頭・[`.bazelrc#L694-L699`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/.bazelrc#L694-L699)）ため、この define を使う公式 macOS arm64 wheel では **無効（確定）**。未特定なのは §3 手順 4 で申し送った、`TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL`／`TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL` マクロを実際に定義している copts 宣言箇所のみであり、macOS wheel での有効・無効自体は本節の `--define=tensorflow_mkldnn_contraction_kernel=0` により確定している。

## §5 GB10（Linux aarch64・TF 2.21.0）: oneDNN はビルドに含まれるが既定値は OFF（MIDR 実測記録で確定）

`.bazelrc`（2.21.0）の `common:release_arm64_linux` は `--config=mkl_aarch64_threadpool` を経由し（[`.bazelrc#L775-L778`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/.bazelrc#L775-L778)）、`mkl_aarch64_threadpool` config は `--define=build_with_mkl_aarch64=true`（OpenMP は `--@compute_library//:openmp=false` で明示的に無効。[`.bazelrc#L290-L292`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/.bazelrc#L290-L292)）を設定する。これは §4 の `if_mkl()` selector の `build_with_mkl_aarch64` 条件に一致するため `-DINTEL_MKL` が定義される（確定）。

`tensorflow/core/util/port.cc::DefaultOneDnnPolicy()`（2.21.0 でも同一ファイル・ほぼ同一実装。§7 差分表参照）の分岐条件を要約する（[`port.cc#L108-L124`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/core/util/port.cc#L108-L124)）: `INTEL_MKL` 未定義なら無効。Google 内部ビルドなら常に有効。Windows かつ x86 なら常に有効。Linux では x86 の AVX512_VNNI／AVX512_BF16／AVX_VNNI／AMX_TILE／AMX_INT8／AMX_BF16 のいずれかを満たすか、または `TestAarch64CPU(ARM_NEOVERSE_V1)` が真の場合にのみ有効。それ以外（上記いずれにも該当しない Linux aarch64 を含む）は無効。

`TestAarch64CPU(ARM_NEOVERSE_V1)` の実体（`third_party/xla/third_party/tsl/tsl/platform/cpu_info.cc`）は、`getauxval(AT_HWCAP) & HWCAP_CPUID` を確認したうえで `/sys/devices/system/cpu/present` の先頭の CPU 番号 `N` を取り、`/sys/devices/system/cpu/cpu<N>/regs/identification/midr_el1` を 1 回だけ読む。`implementer == 0x41`（Arm Ltd.）かつ `part_num == 0xd40`（Neoverse V1 固有の part number）の場合のみ `is_arm_neoverse_v1_` を立て、`TestAarch64CPU(ARM_NEOVERSE_V1)` はその値を返す（[2.16.2 `cpu_info.cc#L397-L459`・`#L465-L473`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/xla/third_party/tsl/tsl/platform/cpu_info.cc#L397-L473)・[2.21.0 `cpu_info.cc#L407-L469`・`#L484-L494`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/third_party/xla/third_party/tsl/tsl/platform/cpu_info.cc#L407-L494)。両版で同一ロジック・同一パス）。`/proc/cpuinfo` は読まない。`ARM_NEOVERSE_N1`（part `0xd0c`）は enum と MIDR 判定にはあるが `TestAarch64CPU` の switch 文では判定対象外（`default` 分岐。2.16.2 は `return 0`・2.21.0 は `return false`）。

GB10 の CPU は Cortex-X925 ×10 と Cortex-A725 ×10 の構成で、`lscpu` にはこの 2 機種だけが現れる（`docs/perf/logs/cpu-gemm-kc-sweep-1315/lscpu-dgx.txt:7`・`:19`）。`midr_el1` の実測値は既存ログ `docs/perf/logs/cpu-gemm-rayon-sweep-1305/env_info_raw-dgx.txt:36-44` に記録されている: cpu0 は `0x00000000410fd871`（implementer `0x41`・part `0xd87`・Cortex-A725）、cpu5 は `0x00000000410fd851`（part `0xd85`・Cortex-X925）。`present` の中身そのものは記録されていないが、2 機種の part はどちらも `0xd40` と一致しないため、TF がどの CPU 番号を読んでも `TestAarch64CPU(ARM_NEOVERSE_V1)` は false になる。x86 の CPU 機能検査も aarch64 では該当しない（§6）。よって **「GB10 では `TF_ENABLE_ONEDNN_OPTS` 未設定時に oneDNN op-rewrite が既定 OFF」は確定**（判定ロジックは上流コード、MIDR 値は既存実測ログ。どちらも本 doc の新規計測ではない）。なお `INTEL_MKL` 定義（本節冒頭）は公式 wheel のビルド設定から読み取ったもので、計測環境の wheel が公式ビルドかどうかの実機確認（`python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"` 相当）は §11 へ申し送る。未定義の場合も `IsMklEnabled()` は false なので、既定 OFF の結論は変わらない。`TF_ENABLE_ONEDNN_OPTS=1` を明示設定すれば既定値に関わらず oneDNN が有効化される（[`port.cc#L143-L160`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/core/util/port.cc#L143-L160)）が、bench_py.py（§7.1）はこの環境変数を設定していないため計測時は既定値に従う。

## §6 Eigen／oneDNN 選択条件のまとめ

| 条件 | 内容 | 確定／推定 |
|---|---|---|
| ビルド時: `INTEL_MKL` 定義有無 | `if_mkl()` の select 対象（`build_with_mkl_aarch64` define／`linux_x86_64`／`windows`）。macOS は非該当 | 確定（§4・§5） |
| 実行時: `IsMklEnabled()` の既定値 | `INTEL_MKL` 未定義なら常に false。定義時は `DefaultOneDnnPolicy()`（x86 CPU 機能検査 or aarch64 は Neoverse V1 のみ）→ `TF_ENABLE_ONEDNN_OPTS` で上書き可 | 確定（ロジック。GB10 は MIDR 実測記録の part `0xd87`／`0xd85` が `0xd40` と一致せず既定 false。§5） |
| Eigen contraction 自体の oneDNN sgemm 直呼び | `TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL`／`_MKLDNN_CONTRACTION_KERNEL` マクロ定義時、`INTEL_MKL`（op-rewrite 用フラグ）とは独立に発生しうる。マクロの定義元 copts は本解析で未特定 | 推定（境界。§11） |
| CPU 機能検査（x86） | `AVX512_VNNI`／`AVX512_BF16`／`AVX_VNNI`／`AMX_TILE`／`AMX_INT8`／`AMX_BF16` のいずれか | 確定（GB10／M4 Max はいずれも aarch64 のため非該当） |
| dtype・サイズによる分岐 | `mkl_matmul_op.cc` 内部の分岐は未深追い（委譲境界外） | 対象外（スコープ外） |
| graph モード（`mkl_layout_pass.cc`・Grappler remapper） | eager 実行（bench_py.py）の経路には入らない | スコープ外（存在の指摘のみ） |

## §7 TF スレッド方針

### §7.1 bench_py.py の計測条件（確定）

`docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py` の `TF.__init__`（[`bench_py.py#L124-L138`](../perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py)）は `TF_CPP_MIN_LOG_LEVEL=2` と `tf.config.set_visible_devices([], "GPU")`（CPU セル計測時）のみを設定し、`TF_NUM_INTRAOP_THREADS`／`TF_NUM_INTEROP_THREADS`／`OMP_NUM_THREADS`／`tf.config.threading.set_*_parallelism_threads` のいずれも呼んでいない。**すなわち計測は TF のスレッド既定値をそのまま使う**（環境変数によるスレッド数固定は行われていない）。

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

### §8.1 実行時の関数解決（確定）

bench_py.py の `SciPy.__init__` は `import scipy.linalg.blas as blas` で得たモジュールをそのまま `self.blas` に保持し（[`bench_py.py#L171-L172`](../perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py)）、`mm` は `self.blas.sgemm(1.0, a, b)` を直接呼ぶ（`bench_py.py#L176-L177`）。GEMM タスクは `matmul_to_host` → `mm`（`#L178-L179`）、train／infer タスクは `train_step`／`forward_to_host` 内の `self.mm`（`#L182-L198`）を通って同じ `sgemm` に到達する。

`scipy.linalg.blas.sgemm` というモジュール属性は、`blas.py` が読み込み時に `scipy.__config__` の BLAS ビルド情報から `HAS_LP64`／`HAS_ILP64` を決め（[`blas.py#L239-L254`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/linalg/blas.py#L239-L254)）、`HAS_LP64` なら `from scipy.linalg._fblas import *`、そうでなければ `from scipy.linalg._fblas_64 import *` で取り込んだ f2py 生成関数そのものである（[`blas.py#L256-L259`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/linalg/blas.py#L256-L259)）。**計測経路に `get_blas_funcs`／`find_best_blas_type`（[`blas.py#L289-L353`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/linalg/blas.py#L289-L353)・`#L431-L531`）は現れない**。この 2 つは配列の dtype から `s`／`d`／`c`／`z` のプレフィックスを選ぶ API だが、bench_py.py は関数名で `s`（単精度）を固定しており、入力も `upload` が `np.ascontiguousarray(x, dtype=np.float32)` で f32・C 連続にそろえる（`bench_py.py#L174-L175`）。上流 `blas.py` の注記は、このように直接 import した関数がビルド構成次第で LP64・ILP64 のどちらにもなりうると明記している（[`blas.py#L33-L37`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/linalg/blas.py#L33-L37)）。本計測でどちらが使われたかは、§8.3 の BLAS 変種・ILP64／LP64 と同じく実機の `scipy.show_config()` でしか確定できない（§11）。

### §8.2 f2py ラッパーでの入力順序（宣言と上流注記までは確定・コストは未実測）

`scipy/linalg/fblas_l3.pyf.src` の `<prefix>gemm` 宣言（[`fblas_l3.pyf.src#L16-L48`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/linalg/fblas_l3.pyf.src#L16-L48)）は `a`・`b` を `<ftype> dimension(lda,ka), intent(in)` として宣言し、`intent(c)` を付けていない（f2py は Fortran〈列優先〉配列として扱う）。上流 `blas.py` のモジュール注記も「C 連続配列を渡すと、f2py は内部で Fortran 連続配列を渡す」と明記している（[`blas.py#L22-L25`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/linalg/blas.py#L22-L25)）。§8.1 のとおり bench_py.py は Python 層を挟まずこの f2py ラッパーを直接呼び（`bench_py.py#L177`）、`trans_a`／`trans_b` も指定しない。GEMM タスクの `A`・`B` は `upload` で C 連続にそろえた配列なので（`bench_py.py#L175`・`#L205`）、上流注記どおりなら f2py 層で Fortran 連続への変換（コピー）が起き、その変換は計測窓（`run_gemm` の `t0`〜`dt`。`bench_py.py#L208-L211`）の内側で毎回起きる。変換の実挙動とコスト（GEMM セルの時間に占める割合）は未実測（GEMM セルの計測境界に直接関係するため §10 の差分候補へ計上・§11 へ申し送り）。

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

### 比較可能範囲（ハーネス間の突き合わせ。表 1・表 2 の判定の前提）

表 1・表 2 の各セルが同じ処理を比べているかを、計測コード（fandhe-ai 側 `scripts/bench/framework-compare/bench-fandhe/src/main.rs`〈以下 `main.rs`〉・TF／SciPy 側 `docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py`〈以下 `bench_py.py`〉）と、計測に使われた fandhe-ai 0.8.0 のソース（タグ `v0.8.0`。`crates/autodiff`・`crates/facade`）で突き合わせた。`main.rs` の該当定義（`build_model`・`mlp_data`・`run_train`・`run_infer`・`gemm_inputs`）と `fandhe-ai =0.8.0` ピンは、計測直後の main（`b7a266a0`・2026-09-13）でも同じであることを確認したうえで、HEAD の行番号を引く。

| 項目 | fandhe-ai 0.8.0（`main.rs`） | TensorFlow／SciPy（`bench_py.py`） | 一致 |
|---|---|---|---|
| GEMM の入力 | `gemm_inputs`: `Xorshift64Star::new(SEED_A／SEED_B).fill_vec(n*n)`（`main.rs:413-423`。生成式は `bench-common/src/lib.rs:182-184`・シードは `:188-193`） | `fill_vec(SEED_A／SEED_B, n*n)`（`bench_py.py:24-42`・`:204`） | 一致 |
| GEMM の checksum・parity | 全要素の f64 逐次和（`main.rs:471`）。parity は FMA 参照 GEMM と複合判定（`bench-common::parity`） | f64 累積和（`bench_py.py:51-54`）。parity は同じ契約の移植（`bench_py.py:57-87`） | 一致 |
| MLP の構成 | 784→256→10・ReLU・bias あり（`main.rs:141-144`・`:990-995`） | 同じ（`bench_py.py:18`・`:153`・`:184-186`） | 一致 |
| 入力 `x`・目標 `y` | `SEED_X`／`SEED_Y` の `fill_vec`（`main.rs:981-988`） | 同じ（`bench_py.py:225`） | 一致 |
| バッチサイズ | 64（`main.rs:141`） | 64（`bench_py.py:18`） | 一致 |
| 初期パラメータ | `Sequential::add_linear(.., SEED_L1／SEED_L2)`（`main.rs:990-995`）→ `Linear::new`。W・b とも `U(-1/√in, 1/√in)` で、シードは `derive_seed(seed, salt)` で導出し、bias は非ゼロ（v0.8.0 `crates/autodiff/src/nn/linear.rs:66-73`・`nn/init.rs:55-59`・`:73-76`・`:90-94`・`crates/facade/src/compat/sequential.rs:116-124`） | W = `fill_vec(SEED_L1／SEED_L2)`（`[-0.5, 0.5)`）・b = 0（`bench_py.py:223-224`）。candle・burn のハーネスも同じ初期化（`bench-candle/src/main.rs:435-450`・`bench-burn/src/main.rs:189-196`） | **不一致** |
| 損失と正規化 | `mse_loss` = `Reduction::Mean`（全 64×10 要素の平均。v0.8.0 `crates/autodiff/src/var.rs:425-427`。`main.rs:1012`） | TF `tf.reduce_mean(tf.square(pred - y))`（`bench_py.py:154`）・SciPy `np.mean(d * d)` と勾配 `(2.0 / n) * d`（`n = d.size`。`bench_py.py:188-190`） | 一致 |
| optimizer | ホスト側 SGD `p - LR * g`（LR 0.01。`main.rs:147`・`:1021-1036`） | TF `assign_sub(g * LR)`（`bench_py.py:156-157`）・SciPy の in-place 減算（`bench_py.py:194`） | 更新式は一致（実装は異なる） |
| forward の GEMM 回数 | 2 回。CPU infer は `Sequential::predict` の tape 不要経路で Linear→ReLU を `gemm_bias_act` へ融合する（v0.8.0 `sequential.rs:221-228`・`:239-246`。`main.rs:1521`） | 2 回（TF は `@` 2 回・`bench_py.py:163`、SciPy は `sgemm` 2 回・`bench_py.py:198`） | 一致（融合の有無は実装差） |
| backward の GEMM 回数（train） | **4 回**（dX・dW1・dH・dW2）。v0.8.0 の `Op::MatMul`／`Op::LinearAct` の VJP（`grad.rs:84-88`・`:365-397`）は `matmul_vjp`（`:505-520`。`da`・`db` を無条件に計算）を通して入力側の勾配を常に返し、`Tape::backward`（`backward.rs:175-202`）は入力側で枝刈りしない。第 1 層の入力 `x` は勾配不要の葉だが dX = g[64,256] × W1ᵀ[256,784] が計算される | **3 回**（dW1・dH・dW2）。TF は eager tape が watch 外かつ `sources` 外の入力を unneeded とし（[`tape.h#L743-L750`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/c/eager/tape.h#L743-L750)・[2.21.0 `#L744-L751`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/c/eager/tape.h#L744-L751)）、`_MatMulGrad` は `skip_input_indices` に 0 があれば `_MatMulGradAgainstSecondOnly` へ分岐する（[`math_grad.py#L1697-L1709`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/python/ops/math_grad.py#L1697-L1709)・[2.21.0 `#L1696-L1708`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/python/ops/math_grad.py#L1696-L1708)。`skip_input_indices` を勾配関数へ渡す口は [`backprop.py#L92-L136`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/python/eager/backprop.py#L92-L136)。確定〈Python 側の skip 分岐と C++ tape の unneeded 判定。両者の間の pywrap での受け渡しは未追跡〉）。`x` は `tf.identity(tf.constant(..))`（`bench_py.py:139-141`）で watch されない。SciPy は `gw2`・`gh`・`gw1` だけを計算する（`bench_py.py:191-193`） | **不一致** |
| 1 計測の単位（train） | 1 step。100 step のうち先頭 20 を warmup とし、残り 80 step の中央値（`main.rs:145-146`・`:1003`・`:1045`） | 同じ（`bench_py.py:19`・`:233-238`） | 一致 |
| 1 計測の単位（infer） | forward 1 回 + ホスト実体化 + checksum（warmup 20・計測 20。`main.rs:1520-1523`・`:1537-1542`） | 同じ（`bench_py.py:248-254`） | 一致 |
| checksum（train） | 100 step 目の更新前の loss（`main.rs:1014-1017`・`:1056`） | 同じ（`bench_py.py:235`・`:241`） | 定義は一致 |
| checksum（infer） | 出力 64×10 の全要素の f64 逐次和（`checksum_tensor`。`main.rs:395-411`・`:1522`） | 同じ（`bench_py.py:51-54`・`:251`） | 定義は一致 |
| `mode: fresh` の中身 | train は step ごとに新しい tape を作り、パラメータと勾配をホストへ読み出して `Tensor::from_slice` で作り直す（`main.rs:1004-1038`）。CPU infer は `predict`（tape 不要） | プロトコルは 1 種類だけで、`mode` は `"fresh"` を固定で書き出す（`bench_py.py:277`）。TF は step ごとに `GradientTape`、SciPy は NumPy の手書き backprop | 名前だけが共通（中身は各ハーネスの定義） |

**checksum が違う理由**: 定義（train は 100 step 目の loss、infer は全出力の f64 逐次和）は両ハーネスで同じで、違うのは初期パラメータの値だけである。これは実測 JSONL でも裏付けられる。共有初期化（`fill_vec` の W・bias 0）を使う実装は、フレームワークとデバイスを問わず同じ値になる。

- GB10: TF train 0.1175494（`docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/results-dgx-py-0.8.0.jsonl:13`）・infer 1381.70878（`:14`）、SciPy 0.1175495（`:20`）・1381.70877（`:21`）、PyTorch CPU 0.1175496（`:6`）・1381.70880（`:7`）、candle CPU 0.117550・1381.708783（`docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/results-dgx-0.8.0.jsonl:25`・`:26`）、burn CPU 0.117549・1381.708825（同 `:38`・`:39`）
- M4 Max: candle CPU 0.117550・1381.708783、burn CPU 0.117549・1381.708825（`scripts/bench/framework-compare/results/raw/results-m4max-0.8.0.jsonl:23`・`:24`・`:32`・`:33`）。TF／SciPy は転記値で checksum が残っていない
- fandhe-ai 0.8.0 は GB10 CPU・CUDA、M4 Max CPU・Metal、`fresh`・`reuse` のすべての行で train 0.080541・infer 13.976574 と自分自身では一致する（`results-dgx-0.8.0.jsonl:10-13`・`:46`・`:84`、`results-m4max-0.8.0.jsonl:10-13`・`:41`・`:79`）

framework-compare の README も「重みの値は異なるが実行時間には影響しない」（`scripts/bench/framework-compare/README.md:43-44`）とし、train の最終 loss と train／infer の要素単位検証をフレームワーク間では突合しないと明記している（同 `:131-134`・`:650-651`）。dX の追加計算は loss の値に影響しないため、checksum の差の原因ではない。

**セルごとの判定**:

| セル | 判定 | 根拠と条件 |
|---|---|---|
| GEMM・GB10（N=256〜4096） | 比較可能（処理が同等） | 入力・checksum 定義・parity 契約が一致し、両側とも `parity_fail_count: 0`。checksum も全形状で本体の複合判定（相対誤差 1e-3 未満）の範囲で一致する（例: N=256 は fandhe-ai 237.54666〈`results-dgx-0.8.0.jsonl:6`〉・TF 237.546618〈`results-dgx-py-0.8.0.jsonl:8`〉・SciPy 237.546660〈`:15`〉、N=4096 は fandhe-ai −25768.747284〈`results-dgx-0.8.0-extra.jsonl:6`〉・TF −25768.737869〈`results-dgx-py-0.8.0.jsonl:12`〉・SciPy −25768.729558〈`:19`〉） |
| GEMM・M4 Max（N=256〜2048） | 比較可能（TF／SciPy 側は転記値） | 処理は同じ（C = A×B・同一形状）。fandhe-ai 側は実測 JSONL で `parity_fail_count: 0`（`results-m4max-0.8.0.jsonl:1-4`）。TF／SciPy 側は元 JSONL が未収録で（`docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.py:82-86`）、転記スクリプトが転記行へ `parity_fail_count: 0` を一律に書き込む（`gen_1988.py:101-102`）ため実測値ではない。同じハーネス・同じシードで計測したという根拠はスコアボードの記載（`docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/body_1988.html:9`）に留まる。よって M4 Max の TF／SciPy GEMM を「ゼロ fail を確認済み」とは扱わない |
| infer（両ホスト） | **条件付きで比較可能** | forward の演算列・GEMM 回数・形状・checksum 定義が一致し、違いは初期パラメータの値だけで演算量は同じ。条件: (1) 値の違いが実行時間に影響しないことは README の設計前提（`README.md:43-44`）で、本リポジトリに A/B 実測はない。(2) 入力が違うため checksum が一致せず、GEMM の parity に当たるフレームワーク間の数値一致確認がない。(3) M4 Max の TF／SciPy は転記値で checksum が無く、同じハーネスで計測したことは `body_1988.html:9` の記載と、candle・burn の M4 Max checksum が共有初期化の値と一致することによる間接確認に留まる |
| train（両ホスト） | **参考値（同等性は未確認）。勝ち負けの判定から外す** | backward の GEMM 回数が 4 回対 3 回で演算量が違う（fandhe-ai だけが dX を計算する。上表）。初期パラメータの値も違う。`reuse` モードの `Op::LinearResident` の VJP も d_input を計算する（v0.8.0 `grad.rs:280-288`）ため、モードを替えてもこの差は残る。GB10 で fandhe-ai が追加の GEMM を抱えたまま上回っている点は向きとしては保守的だが、判定には使わない。M4 Max の対 SciPy の負けは、この差とライブラリ実装差を切り分けられない |

### 表 1: GB10（DGX Spark。TF 2.21.0・SciPy 1.18.1・fandhe-ai 0.8.0）

出典（比較値はすべて `mode: fresh` 同士）:

- TF／SciPy: `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/results-dgx-py-0.8.0.jsonl:8-21`（TF は L8〜L14・SciPy は L15〜L21）。bench_py.py は計測プロトコルが 1 種類のみで、`mode` を `"fresh"` 固定で書き出す（[`bench_py.py#L277`](../perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py)）。
- fandhe-ai（gemm N=256〜2048・train・infer）: `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/results-dgx-0.8.0.jsonl` の `"device":"cpu"`・`"mode":"fresh"` 行（gemm: L6〜L9、train: L12、infer: L13）。
- fandhe-ai（gemm N=4096）: 同キャンペーンの追加計測 `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/results-dgx-0.8.0-extra.jsonl:6`（`"mode":"fresh"`・1128.210 GFLOPS・`parity_fail_count: 0`）。本計測 JSONL には CPU gemm N=4096 行がなく、追加計測スクリプトがこのセルを埋める目的で同じ `bench-fandhe` を `--mode fresh` で起動した（`docs/perf/logs/lowlayer-diagnosis-2026-09-12/scripts/dgx-run2.sh:2`・`:21`。同スクリプト L13 の RAYON スイープと違い `taskset`・`RAYON_NUM_THREADS` の指定はない）。同ファイル L5 には `reuse` 値（1142.795 GFLOPS）もあるが、TF／SciPy 側に `reuse` 相当の値がないため判定には使わない（どちらを使っても判定は変わらない）。

| タスク | N | fandhe-ai (GFLOPS / ms) | TensorFlow 2.21.0 | SciPy 1.18.1 | 判定 |
|---|---|---|---|---|---|
| gemm | 256 | 145.9 GF | 59.6 GF | 61.9 GF | fandhe-ai 勝ち |
| gemm | 512 | 180.2 GF | 86.7 GF | 107.4 GF | fandhe-ai 勝ち |
| gemm | 1024 | 525.7 GF | 235.2 GF | 187.0 GF | fandhe-ai 勝ち |
| gemm | 2048 | 1020.9 GF | 603.8 GF | 338.8 GF | fandhe-ai 勝ち |
| gemm | 4096 | 1128.2 GF（追加計測 `results-dgx-0.8.0-extra.jsonl:6`） | 944.4 GF | 457.4 GF | fandhe-ai 勝ち |
| train | 64 | 0.892 ms | 2.720 ms | 1.198 ms | 参考値（同等性は未確認。backward の GEMM 回数が 4 対 3。§10「比較可能範囲」） |
| infer | 64 | 0.178 ms | 0.913 ms | 0.600 ms | fandhe-ai 勝ち（条件付き。初期パラメータの値だけが違う。§10「比較可能範囲」） |

train・infer の値の出典は TF `results-dgx-py-0.8.0.jsonl:13`・`:14`、SciPy `:20`・`:21`、fandhe-ai `results-dgx-0.8.0.jsonl:12`・`:13`。

GB10 では判定対象セル（GEMM 5 形状〈N=4096 を含む〉と infer）すべてで fandhe-ai が両 FW を上回る（**負けセルなし**。infer は条件付き）。train は参考値で、勝ち負けには数えない。`gemm` 行は全行 `parity_fail_count: 0`（判定可能。fandhe-ai の N=4096 行〈`results-dgx-0.8.0-extra.jsonl:6`〉を含む。出典 JSONL の `gemm` 行のみが同フィールドを持つ）。ただし TF／SciPy の N=2048・4096 行は `parity_scaled_abs_rescued` がそれぞれ TF 4・547、SciPy 2・611（`results-dgx-py-0.8.0.jsonl:11`・`:12`・`:18`・`:19`）で、framework-compare ハーネス限定のスケール付き絶対誤差項（`bench_py.py#L80-L82`）で救済された要素を含む。fandhe-ai 側の `gemm` 行は全行 0。`train`／`infer` 行の出典 JSONL（`results-dgx-py-0.8.0.jsonl`）には `parity_fail_count` フィールド自体が存在せず（fandhe-ai 側 `train`／`infer` 出力にも同様に存在しない設計）、判定不能（parity 検査の対象外）である——ゼロ fail の記述は `gemm` 行に限る。train／infer の同等性の根拠は parity ではなく、上の「比較可能範囲」のハーネス突き合わせである。

### 表 2: M4 Max（TF 2.16.2・SciPy 1.18.1 は `gen_1988.py::M4_PY` 転記値・fandhe-ai は `results-m4max-0.8.0.jsonl`）

出典: `docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.py::M4_PY`（TF／SciPy。2026-09-12 ページからの転記と明記されている。`mode: fresh` で記録。[`gen_1988.py#L101-L107`](../perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.py)）・`scripts/bench/framework-compare/results/raw/results-m4max-0.8.0.jsonl`（fandhe-ai。`fresh` モード〈gemm: L1〜L4、train: L10、infer: L11〉。表 1〈GB10〉・本表の GEMM 行と計測条件をそろえるため train・infer も `fresh` で統一する）。CPU gemm N=4096 は `M4_PY` に TF／SciPy の値がなく（`gen_1988.py#L92`・`#L95`）、fandhe-ai 側の同 JSONL にも CPU N=4096 行がないため、本表には含めない。TF／SciPy は元 JSONL がリポジトリに無い転記値で（`gen_1988.py#L82-L86`）、`gemm` 行の `parity_fail_count: 0` は転記スクリプトが一律に書き込んだ値（`gen_1988.py#L101-L102`）であり実測ではない。train・infer は checksum も残っていない。このため本表の TF／SciPy 側については、表 1 と違い parity・checksum による同等性の確認ができない（§10「比較可能範囲」）。fandhe-ai 側の `gemm` 行は実測 JSONL で全行 `parity_fail_count: 0`（`results-m4max-0.8.0.jsonl:1-4`）。

| タスク | N | fandhe-ai (GFLOPS / ms) | TensorFlow 2.16.2 | SciPy 1.18.1 | 判定 |
|---|---|---|---|---|---|
| gemm | 256 | 107.4 GF | 129 GF | 121 GF | **fandhe-ai 負け（対 TF・対 SciPy 双方）** |
| gemm | 512 | 420.8 GF | 282 GF | 227 GF | fandhe-ai 勝ち |
| gemm | 1024 | 739.2 GF | 471 GF | 372 GF | fandhe-ai 勝ち |
| gemm | 2048 | 999.4 GF | 699 GF | 470 GF | fandhe-ai 勝ち |
| train | 64 | 0.835 ms（fresh） | 0.96 ms | 0.31 ms | 参考値（同等性は未確認。backward の GEMM 回数が 4 対 3。§10「比較可能範囲」） |
| infer | 64 | 0.178 ms（fresh） | 0.266 ms | 0.164 ms | **fandhe-ai 負け（条件付き。対 SciPy のみ。対 TF は勝ち）** |

M4 Max の負けは N=256 GEMM（対 TF・対 SciPy 双方）と infer（対 SciPy のみ。条件付き）の 2 セル。train は参考値で、勝ち負けには数えない（値の上では TF より速く SciPy より遅いが、演算量が違うため判定しない）。

**比較に使う fandhe-ai の版について（表 1・表 2 共通）**: fandhe-ai 0.9.0 の実測 JSONL も本リポジトリに収録されている（M4 Max: `docs/perf/logs/framework-compare-0.9.0-remeasure/m4max-series-a/results-m4max-0.9.0-median5.jsonl`・`m4max-series-b/results-m4max-0.9.0-median5.jsonl`、GB10: `docs/perf/logs/framework-compare-0.9.0-remeasure/gb10/results-dgx-0.9.0.jsonl`・`results-dgx-0.9.0-extra.jsonl`）。しかし TensorFlow／SciPy は 0.9.0 の再計測では計測されておらず、2026-09-12 の値がそのまま流用されている（`docs/perf/logs/framework-compare-0.9.0-remeasure/README.md:21`・`scoreboard/gen_090.py:7`・`:77`）。本 doc は比較対象 FW と同じ時期（2026-09-12。GB10 は同一キャンペーン `lowlayer-diagnosis-2026-09-12`）に計測された fandhe-ai 0.8.0 を比較値として使う。0.9.0 との対比は `gen_090.py` のスコアボードが扱っており、本 doc では行わない。

**train・infer セルのモード選択について（注記）**: 本表の train・infer は fandhe-ai 側を `fresh` モード値（train 0.835ms・infer 0.178ms。`results-m4max-0.8.0.jsonl:10`・`:11`）で統一している。理由は GEMM 行（本表・表 1 とも `fresh`）・表 1（GB10。train・infer を含め fandhe-ai・TF・SciPy とも `fresh` 同士）と計測条件をそろえるため。ただし `fresh` は名前が共通なだけで、中身は各ハーネスの定義である（§10「比較可能範囲」の `mode: fresh` の行）。参考として、同じ JSONL には fandhe-ai の `reuse` モード値（train 1.000ms・infer 0.195ms。`results-m4max-0.8.0.jsonl:41`・`:79`）も存在し、`docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.py` のスコアボード本体は `reuse` を主表示列に採用している（`me = data.get((..., 'reuse'))`。[`gen_1988.py#L146-L147`](../perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.py)）が、TF／SciPy 側の値（`M4_PY`）は `fresh` としてのみ記録されており `reuse` 相当の値がそもそも存在しない（transcribed 転記データのため実測が 1 モードのみ）。infer は `reuse` 値（0.195ms）でも対 TF 勝ち・対 SciPy 負けで結論は変わらない。train は `fresh`・`reuse` のどちらでも参考値である（`reuse` でも dX を計算するため。§10「比較可能範囲」）。`reuse` の train 値（1.000ms）は TF 2.16.2（0.96ms）より遅いが、これも判定には使わない。

## §11 限界・申し送り

- 到達範囲はディスパッチ層まで。Eigen の contraction 実装本体・oneDNN（`dnnl_sgemm`）内部・Arm Compute Library・OpenBLAS・Accelerate の内部カーネル実装には踏み込んでいない。
- graph モード（`mkl_layout_pass.cc`・Grappler remapper）は対象外（bench_py.py は eager のみ）。NumPy・Eigen・oneDNN・ACL・OpenBLAS・Accelerate 内部の実装詳細は非目標。
- 実機構成の確認が必要な項目（Phase 3 への申し送り）:
  - GB10 実機で `python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"` 相当を実行し、計測環境の TF wheel が公式ビルド（`INTEL_MKL` 定義）であることを確認する（§5。Neoverse V1 判定は既存実測ログ `docs/perf/logs/cpu-gemm-rayon-sweep-1305/env_info_raw-dgx.txt:36-44` の MIDR で確定済みのため申し送り対象外。wheel が公式ビルドでなくても既定 OFF の結論は変わらない）
  - GB10・M4 Max 実機で `python -c "import scipy; scipy.show_config()"` を実行し、§8.3 の BLAS 変種（OpenBLAS／Accelerate・ILP64／LP64）を確定させる
  - §3 手順 4 の `TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL`／`TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL` マクロの定義元 copts 箇所の特定
  - §7.2 の eager 実行時スレッドプール配線（`EagerContext` 側）の追跡
  - §8.2 の f2py `sgemm` 呼び出しにおけるコピー（上流注記上は C 連続入力で発生）の実挙動とコスト（`np.ascontiguousarray` 入力に対する f2py 層の挙動）の実測確認
  - §8.1 の直接 import した `sgemm` が LP64（`_fblas`）・ILP64（`_fblas_64`）のどちらだったかの確認（§8.3 の `scipy.show_config()` と同じ手順で確定できる）
- train・infer の比較可能性を上げるための申し送り（本 doc はハーネスを変更しない）:
  - infer を「条件付き」から「比較可能」にするには、fandhe-ai 側を共有初期化（`fill_vec` の W・bias 0）で計測し、checksum がほかの実装（1381.70878 前後）と一致することを確かめる必要がある
  - train を判定対象に戻すには、backward の GEMM 回数をそろえる必要がある。fandhe-ai 0.8.0 の autodiff が勾配不要の葉（入力 `x`）についても入力側勾配を計算する挙動（`grad.rs:84-88`・`:365-397`・`:505-520`、`backward.rs:175-202`）はライブラリ側の性質であり、本 doc のスコープ外
  - M4 Max の TF／SciPy の元 JSONL（checksum・parity 付き）が見つかれば、表 2 を表 1 と同じ根拠で再判定できる
- Phase 3（負けセルの原因確定・A/B・性能実測）は本 doc の対象外。§10 の表 1・表 2 は既存 JSONL・転記データの突合結果であり、新規計測は一切行っていない。

## §12 ライセンス記録

- **TensorFlow**: Apache License 2.0。タグ `v2.16.2`／`v2.21.0` 時点の `LICENSE` ファイルで確認（リポジトリルート）。
- **SciPy**: BSD 3-Clause License。タグ `v1.18.1` 時点の `LICENSE.txt` で確認（リポジトリルート）。
- 読み取りで言及したが本リポジトリへは何も取り込んでいないもの: Eigen（MPL-2.0）・oneDNN（Apache-2.0）・Arm Compute Library（MIT）・OpenBLAS（BSD-3-Clause）。
- 本 doc は上流コードの逐語引用を含まず、結論・識別子名・`path:line`・出典 URL のみを記録している。本リポジトリへ何も取り込んでいないため、`docs/license-matrix.md`・`.claude/rules/deps-policy.md` の対象外。

## §13 出典

### 上流ファイル一覧（タグ固定 URL）

- TensorFlow 2.16.2: [`tensorflow/core/util/port.cc`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/util/port.cc)・[`tensorflow/core/util/util.cc`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/util/util.cc)・[`tensorflow/core/common_runtime/process_util.cc`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/common_runtime/process_util.cc)・[`tensorflow/core/kernels/matmul_op_impl.h`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/kernels/matmul_op_impl.h)・[`tensorflow/core/kernels/BUILD`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/core/kernels/BUILD)・[`third_party/xla/third_party/tsl/tsl/framework/contraction/eigen_contraction_kernel.h`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/xla/third_party/tsl/tsl/framework/contraction/eigen_contraction_kernel.h)・[`third_party/xla/third_party/tsl/tsl/platform/cpu_info.h`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/xla/third_party/tsl/tsl/platform/cpu_info.h)・[`third_party/xla/third_party/tsl/tsl/platform/cpu_info.cc`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/xla/third_party/tsl/tsl/platform/cpu_info.cc)・[`third_party/xla/third_party/tsl/tsl/mkl/build_defs.bzl`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/xla/third_party/tsl/tsl/mkl/build_defs.bzl)・[`third_party/mkl/build_defs.bzl`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/third_party/mkl/build_defs.bzl)・[`tensorflow/tensorflow.bzl`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/tensorflow.bzl)・[`.bazelrc`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/.bazelrc)
- TensorFlow 2.21.0: [`tensorflow/core/util/port.cc`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/core/util/port.cc)・[`third_party/xla/third_party/tsl/tsl/platform/cpu_info.cc`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/third_party/xla/third_party/tsl/tsl/platform/cpu_info.cc)・[`.bazelrc`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/.bazelrc)
- TensorFlow eager backward（§10「比較可能範囲」の backward GEMM 回数の根拠。両タグ）: [`tensorflow/c/eager/tape.h`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/c/eager/tape.h)・[`tensorflow/python/ops/math_grad.py`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/python/ops/math_grad.py)・[`tensorflow/python/eager/backprop.py`](https://github.com/tensorflow/tensorflow/blob/v2.16.2/tensorflow/python/eager/backprop.py)（2.21.0 は同名パスの [`tape.h`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/c/eager/tape.h)・[`math_grad.py`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/python/ops/math_grad.py)・[`backprop.py`](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/python/eager/backprop.py)）
- SciPy 1.18.1: [`scipy/linalg/blas.py`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/linalg/blas.py)・[`scipy/linalg/fblas_l3.pyf.src`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/linalg/fblas_l3.pyf.src)・[`pyproject.toml`](https://github.com/scipy/scipy/blob/v1.18.1/pyproject.toml)・[`.github/workflows/wheels.yml`](https://github.com/scipy/scipy/blob/v1.18.1/.github/workflows/wheels.yml)

### 本リポジトリ側の参照 doc・データ

- `docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py`（計測ハーネス）
- `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/results-dgx-py-0.8.0.jsonl`・`results-dgx-0.8.0.jsonl`・`results-dgx-0.8.0-extra.jsonl`（GB10 実測 JSONL。`-extra` は CPU gemm N=4096 `fresh` 行〈L6〉の出典）・`docs/perf/logs/lowlayer-diagnosis-2026-09-12/scripts/dgx-run2.sh`（追加計測の起動条件）
- `docs/perf/logs/cpu-gemm-rayon-sweep-1305/env_info_raw-dgx.txt`（GB10 の `midr_el1` 実測記録。§5 の Neoverse V1 判定の根拠）・`docs/perf/logs/cpu-gemm-kc-sweep-1315/lscpu-dgx.txt`（GB10 の CPU 機種構成）
- `docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.py`（M4 Max 転記データ `M4_PY`）
- `scripts/bench/framework-compare/results/raw/results-m4max-0.8.0.jsonl`（M4 Max fandhe-ai 実測 JSONL）
- `docs/perf/logs/framework-compare-0.9.0-remeasure/`（fandhe-ai 0.9.0 の実測 JSONL。TF／SciPy は未再計測のため本 doc の比較には使わない。§10 の版に関する注記参照）
- `crates/backend-cpu/src/gemm_blis/`・`sme_detect.rs`・`thread_limit.rs`・`small_shape_thread_cap.rs`・`gb10_affinity.rs`（fandhe 側対照）
- §10「比較可能範囲」のハーネス突き合わせ: `scripts/bench/framework-compare/bench-fandhe/src/main.rs`・`bench-common/src/lib.rs`・`bench-candle/src/main.rs`・`bench-burn/src/main.rs`・`scripts/bench/framework-compare/README.md`（`:43-44`・`:131-134`・`:650-651`）、タグ `v0.8.0` の `crates/autodiff/src/nn/linear.rs`・`nn/init.rs`・`var.rs`・`grad.rs`・`backward.rs`・`crates/facade/src/compat/sequential.rs`、`docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/body_1988.html`（M4 Max の TF／SciPy が同一ハーネス・同一シードで計測されたという記載）
- `docs/backend-cpu-gb10-affinity-design.md`・`docs/cpu-matmul-fixed-cost-design.md`・`docs/perf/cpu-gemm-small-shape-thread-cap.md`（既存判定記録との突合対象。今回の差分候補〈§8.2 の f2py コピーの実挙動・コスト〉は上記いずれの記録にも未収録の新規項目であり、既存 REJECT／undetermined 記録との重複はない。§5 の Neoverse V1 判定は既存実測ログ `docs/perf/logs/cpu-gemm-rayon-sweep-1305/env_info_raw-dgx.txt:36-44` の MIDR で確定済みのため差分候補から外した）

### 既存記録との突合結果（Step 6）

`docs/perf/cpu-gemm-default-thread-limit.md`・`docs/perf/cpu-gemm-small-shape-thread-cap.md`・`docs/perf/cpu-gemm-gb10-affinity-ab.md`・`docs/perf/cpu-gemm-blocking-sweep.md`・`docs/perf/cpu-gemm-2d-dynamic-variant.md`・`docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`・`docs/perf/cpu-gemm-ic-dynamic-variant.md`・`docs/perf/cpu-gemm-sme-fmopa-microkernel.md`・`docs/perf/cpu-gemm-neon-b-laneq-fma.md`・`docs/perf/cpu-matmul-fixed-cost-impl.md`・`docs/perf/cpu-mse-backward-sequential-threshold.md`・`docs/perf/cpu-gemm-candle-cpu-retune.md`・`docs/perf/cpu-gemm-candle-gate-remeasurement.md`（以上 `docs/perf/` 配下）・`docs/cpu-gemm-prefetch-decision.md`・`docs/cpu-gemm-b-packing-sharing-decision.md`・`docs/cpu-gemm-2d-dynamic-partition-design.md`（以上 `docs/perf/` ではなく `docs/` 直下）はいずれも fandhe-ai 自身の CPU GEMM 最適化記録であり、TensorFlow／SciPy のディスパッチ経路そのものを扱ったものではないため、本 doc の差分候補（§8.2 の f2py コピーの実挙動・コスト）と重複する既存判定は確認できなかった（§5 の Neoverse V1 判定は、判定記録ではなく環境記録 `docs/perf/logs/cpu-gemm-rayon-sweep-1305/env_info_raw-dgx.txt` の MIDR 実測値で確定させた）（`git grep -l -E "REJECT|undetermined|判定不能" -- docs` によるスコープ全体の突合を含む）。Phase 3（#2098）への入力として §11 の各項目をそのまま引き継ぐ。
