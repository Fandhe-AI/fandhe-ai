# PyTorch 2.14 CPU 経路解析（Accelerate/BLAS・スレッド方針・小形状）

イシュー #2094（親 #2089・Phase 2「他ライブラリのコード取得・詳細解析」。
後続は #2098）。**docs 専用・コード変更なし・実測なし**の解析記録。PyTorch
v2.14.0 の CPU 経路をソースコード（GitHub API 経由の読み取り専用取得。
リポジトリへの持ち込みなし）から特定し、スコアボードの CPU 負けセルの
原因候補を fandhe 側の実装事実と対照する。実機での確認・A/B は Phase 3
（#2098）へ申し送る。

## 1. 判断サマリ

- PyTorch の `mm`／`addmm`（fp32・CPU）は同一関数 `addmm_impl_cpu_`
  （`aten/src/ATen/native/LinearAlgebra.cpp:1392`）に集約される。`mm` も
  `beta=0` で同関数を呼ぶ（`:1628` `mm_out_cpu`）ため、**mm と addmm は
  bias 融合の有無以外はディスパッチ経路が同一**である
- CPU GEMM の既定精度（fp32・reduced precision 未 opt-in）は
  **oneDNN の bf32/tf32 高速化パスを経由せず**、`at::native::cpublas::gemm`
  経由で BLAS の `sgemm_`（macOS 非 iOS はこの Fortran 呼出し）または
  `cblas_sgemm`（iOS 限定）へ直結する（`CPUBlas.cpp:216-247`）
- **M4（macOS arm64）**: BLAS 選択は Accelerate framework の Fortran
  `sgemm` シンボル探索（`cmake/Modules/FindBLAS.cmake:123-135`。vecLib
  より優先）。呼び出しは Fortran インターフェース `sgemm_`（`cblas_sgemm`
  ではない。`CPUBlas.cpp:236-246`。`C10_IOS` 分岐の外）
- **GB10（aarch64 Linux・cu130 ホイール）は推定**: 公式ビルドスクリプトが
  aarch64 では `USE_MKLDNN=1`／`USE_MKLDNN_ACL=1`（`.ci/pytorch/build.sh:90-93`）、
  かつ cu130 aarch64 manywheel の Docker イメージが NVPL BLAS
  （`nvpl_blas`・linux-sbsa 版）と Arm Compute Library（ACL）を組み込む
  （`.ci/docker/manywheel/Dockerfile_cuda_aarch64:89-122`。
  `install_nvpl.sh`／`install_acl.sh`）ことをビルドスクリプト上で確認した。
  通常の非転置 `x @ w` 形状では addmm の ACL ヒューリスティク条件
  （`transpose_c && transpose_a && !transpose_b`）を満たさない見込みのため、
  **NVPL の `sgemm_` が主経路と推定**する（実行時分岐の実機確認はしていない）
- **スレッド方針は M4 と GB10 で構造が異なる**: intra-op スレッド数の既定値
  導出（`ATen/ParallelCommon.cpp:103-130`）は macOS + aarch64 限定で
  `sysctlbyname("hw.perflevel0.physicalcpu", ...)` により P コア数のみを
  使う（M4 Max で 12。`docs/perf/cpu-gemm-baseline-remeasurement.md:59`
  の実測と整合）。GB10（Linux aarch64）にはこの分岐がなく
  `TaskThreadPoolBase::defaultNumThreads()` に落ちる（GB10 実測
  `torch.get_num_threads()=20`＝`nproc`。
  `docs/perf/logs/parity-torch-cpu-truth-1985/env_info.txt`）。
  `torch.set_num_threads` は OpenMP／MKL のスレッド数のみ制御し
  （`ParallelNative.cpp:207` の `mkl_set_num_threads(1)` が唯一の BLAS 側
  呼び出し）、Accelerate 内部スレッド数への明示的な連動コードはソース上
  存在しない
- **小形状の固定費**: 要素ごと演算（add／relu／sub／pow 等）の
  `TensorIterator::for_each` 既定 grain size は 32768（`TensorIterator.h:76`）。
  `parallel_for` は `numiter > grain_size` を満たさない限り fork-join を
  経由せず呼び出しスレッドでそのまま実行する（`Parallel-inl.h:9-30`）。
  ベンチ形状（BATCH=64・784→256→10。要素数は最大 64×784=50176、多くは
  64×256=16384 以下）ではほぼ全ての要素ごと演算が直列実行される見込み
  （`mean` 等の縮約演算は要素ごとの `for_each` とは異なるディスパッチ
  経路を経由するため本条件を直接適用できず、この判断サマリの対象外。
  §7.1 参照。縮約演算側の並列化条件はソース未確認のため Phase 3 へ
  申し送る）。
  CPU アロケータ（`c10/core/impl/alloc_cpu.cpp`）はキャッシュを持たず
  `posix_memalign` 等を呼び出しごとに直接行う（64B アライメント。
  `c10/core/alignment.h:15`。分岐条件は `#ifdef C10_MOBILE`〈モバイル
  ビルドのみ 16B〉であり、M4（macOS デスクトップ）・GB10（Linux
  サーバー）はいずれも非モバイルビルドのため 64B。詳細は §7.2 参照）
- ベンチで測定した PyTorch train 0.20 ms（M4）・367 µs（GB10）は
  `nn.Linear`（addmm 融合）ではなく **`x @ w1 + b1` の分離 mm＋add**
  （`docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py:105-114`）を
  経由した値である
- fandhe 側（CPU バックエンド本番経路）はこれらと 3 点で構造が異なる:
  (a) `nn::Linear::forward` も非融合の matmul→add だが epilogue 融合版
  （`forward_with_activation`）を別途持つ、(b) PanelBuffers（GEMM の
  A/B パネルバッファ）を rayon タスク・呼び出しごとに `vec!` で毎回確保、
  (c) MSE backward 等の小要素数演算でも PyTorch の GRAIN_SIZE 相当の
  逐次フォールバックを持たず常に rayon を使う（§7 参照）

## 2. 対象・版・出典

- 対象: PyTorch タグ `v2.14.0`（GitHub `pytorch/pytorch`）
- 解決済みコミット SHA: `2b3ec34829036a65cd9d1398ea72a0167dc37470`
  （`gh api repos/pytorch/pytorch/git/ref/tags/v2.14.0` で解決。軽量タグ
  のため `.object.type == "commit"` で直接得られた。注釈付きタグの
  二段解決は不要だった）
- 取得方法: `gh api -H 'Accept: application/vnd.github.raw+json'
  'repos/pytorch/pytorch/contents/<path>?ref=v2.14.0'` によるファイル単位
  取得（読み取り専用。ローカル一時ディレクトリへ保存し読了後に削除。
  リポジトリへの持ち込みなし）
- 参照ファイル一覧（`path:line` は本 doc 内の各所に記載。ファイル自体は
  取得・削除済みでリポジトリに残らない）:
  - `aten/src/ATen/native/LinearAlgebra.cpp`
  - `aten/src/ATen/native/CPUBlas.cpp`
  - `aten/src/ATen/native/mkldnn/Matmul.cpp`
  - `aten/src/ATen/Parallel.h`／`Parallel-inl.h`
  - `aten/src/ATen/ParallelNative.cpp`／`ParallelCommon.cpp`
  - `aten/src/ATen/TensorIterator.h`
  - `c10/core/impl/alloc_cpu.cpp`／`c10/core/alignment.h`
  - `cmake/Modules/FindBLAS.cmake`
  - `.ci/pytorch/build.sh`
  - `.ci/docker/common/install_nvpl.sh`
  - `.ci/docker/manywheel/Dockerfile_cuda_aarch64`
  - `LICENSE`
- 上流 URL: `https://github.com/pytorch/pytorch/tree/v2.14.0`
- ライセンス: `LICENSE`（BSD 系。「From PyTorch」「From Caffe2」の複数
  Copyright 表記＋標準 3 条項 BSD 文言「Redistribution and use in source
  and binary forms, with or without modification, are permitted provided
  that the following conditions are met」を確認）。§11 に記録を集約する

## 3. PyTorch CPU GEMM ディスパッチ（共通の連鎖）

`torch.matmul`／`@`（2 次元）は `aten::mm` へ落ち、`mm_out_cpu`
（`LinearAlgebra.cpp:1628`）が `addmm_impl_cpu_`（`:1392`。`self=result`・
`beta=0`・`alpha=1` で呼ぶ）に委譲する。`torch.addmm`／`nn.Linear` の
`F.linear` も同じ `addmm_impl_cpu_` を通る（`addmm_out_cpu`・`:1609`）。

`addmm_impl_cpu_` 内部の分岐（`:1392-1550`）:

1. 転置ストライドから `transpose_a`／`transpose_b`／`transpose_c` と
   `lda`／`ldb`／`ldc` を導出する（BLAS 呼び出し用の列優先パラメータへの
   変換。コピーを避けるための `resolve_conj`／`clone` 判定を含む）
2. **aarch64 限定の ACL ヒューリスティク分岐**（`:1505`
   `#if defined(__aarch64__) && AT_MKLDNN_ACL_ENABLED()`）: `transpose_c
   && transpose_a && !transpose_b` かつ
   `apply_mkldnn_matmul_heur(...)`（`:1512`。形状ベースのサイズ
   ヒューリスティク）が真で、dtype が f32／bf16／f16 のいずれかのときのみ
   `mkldnn_matmul`（`:1518`。oneDNN 経由で ACL GEMM カーネルへ）へ
   ディスパッチする。**それ以外の形状・dtype・OS では素通り**
3. 上記に該当しなければ `_AT_DISPATCH_ADDMM_TYPES` マクロ経由で
   `at::native::cpublas::gemm`（`:1532`）を呼ぶ

`cpublas::gemm`（float 版・`CPUBlas.cpp:207-247`）は:

1. `#if AT_MKLDNN_ENABLED()` ブロックで `mkldnn_reduced_f32_gemm`
   （`CPUBlas.cpp:217`。実体は `Matmul.cpp:302` の
   `mkldnn_gemm<float>`）を試す。この関数は bf32／tf32 の**いずれかが
   有効化されている**ときのみ true を返す（`Matmul.cpp:161-166` の
   `bf32_usable`／`tf32_usable` 判定。既定は無効——`use_mkldnn_bf32_matmul`
   は `at::globalContext().float32Precision(...)` が
   `Float32Precision::BF16` のときのみ真、`use_mkldnn_tf32_matmul` は
   x86_64 限定＋AMX FP16 命令必須＋`Float32Precision::TF32` 明示設定が
   条件。デフォルトの `torch.matmul`／`nn.Linear` 呼び出しでは両方とも
   偽）
2. 上記が false（既定挙動）なら `#if AT_BUILD_WITH_BLAS()` ブロックの
   `use_blas_gemm`（`:110-119`。整数オーバーフロー・leading dimension
   の妥当性検査のみで dtype・OS 非依存）が真の場合 BLAS を直接呼ぶ
   （`:221-247`）。**非 iOS（`C10_IOS` 未定義。macOS デスクトップ・
   Linux 双方を含む）は Fortran インターフェース `sgemm_`（`:236-246`）
   を使い、iOS のみ `cblas_sgemm`（CBLAS。`:228-235`）を使う**
3. どちらも成立しなければ `gemm_stub`（ベクトル化された素朴実装。
   dispatcher 経由）へフォールバックする

## 4. M4（macOS arm64）: Accelerate ディスパッチ（確定）

- BLAS ライブラリ選択は CMake の `FindBLAS.cmake` が Apple 優先で探索する。
  `WITH_BLAS` 未指定時、`vecLib` より先に「Accelerate」フレームワークの
  Fortran `sgemm` シンボルを `check_fortran_libraries` で探す
  （`FindBLAS.cmake:123-135`。見つかれば `BLAS_INFO=accelerate`・
  `BLAS_IS_ACCELERATE=1`）。既存実測記録
  `docs/perf/cpu-gemm-baseline-remeasurement.md:60`（`torch.__config__.show()`
  実測。**ただし torch 2.13.0 でのプローブであり、本解析対象の 2.14.0
  ではない**。§10 限界節参照）の `BLAS_INFO=accelerate` と整合する
- 実際に呼ばれる Accelerate の API は **Fortran BLAS インターフェース
  `sgemm_`**（`CPUBlas.cpp:236-246`。`C10_IOS` 分岐の外側＝macOS
  デスクトップはこちら）であり、CBLAS（`cblas_sgemm`）でも vDSP／vForce
  系（`vDSP_mmul` 等）でもない。ソースコード上、vDSP／vForce の呼び出しは
  `CPUBlas.cpp`・`Matmul.cpp` のいずれにも存在しない
- Accelerate 内部で AMX と SME のどちらのマイクロカーネルへ実際に
  ディスパッチされるかは **閉源であり本解析では到達不能**。推測は書かず
  `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` を参照するにとどめる
- `torch.set_num_threads` が Accelerate 内部のスレッド数に効くかは、
  ソース上 Accelerate 向けの明示的なスレッド数設定呼び出しが存在しない
  ため（§6 参照）、**効かない可能性が高いが、Accelerate 内部の暗黙の
  スレッド管理までは追えないため断定しない**

## 5. GB10（Grace・aarch64 Linux・cu130 ホイール）: ディスパッチ推定

本節は**推定**である。実行時の分岐先（`DNNL_VERBOSE` 等での確認）は
Phase 3 へ申し送る（§10）。

- 根拠 1（ビルドフラグ）: 公式 CI ビルドスクリプト `.ci/pytorch/build.sh`
  の `90-93` 行が `BUILD_ENVIRONMENT` に `aarch64` を含む場合
  `USE_MKLDNN=1`・`USE_MKLDNN_ACL=1`・`ACL_ROOT_DIR=/acl` を設定する
- 根拠 2（BLAS ライブラリ）: cu130 aarch64 manywheel の Docker イメージ
  定義（`.ci/docker/manywheel/Dockerfile_cuda_aarch64:89-122`）が
  `install_nvpl.sh`（NVIDIA Performance Libraries。`nvpl_blas`
  `linux-sbsa`〈NVIDIA Grace 系サーバ CPU 向け〉版を導入）と
  `install_acl.sh`（Arm Compute Library）の双方を組み込む
- 根拠 3（既存実測記録）:
  `docs/perf/logs/parity-torch-cpu-truth-1985/env_info.txt` が GB10 実機で
  `oneDNN v3.12.0`・`ATen parallel backend OpenMP`・`torch.get_num_threads()
  =20`（`OMP_NUM_THREADS` 未設定）を記録している（oneDNN 有効＝
  `USE_MKLDNN=1` のビルドと整合。NVPL がリンクされているかは
  `env_info.txt` からは確認できない）
- 総合推定: 通常形状の `mm`（`x @ w`。転置なし）は §3 の ACL ヒューリス
  ティク条件（`transpose_c && transpose_a && !transpose_b`）を満たさない
  見込みのため、`cpublas::gemm` の BLAS 経路（`sgemm_`。リンク先は
  **NVPL と推定**）が主経路になると考えられる。`h.T @ gp`
  （backward の重み勾配。`bench_py.py:191/193` 相当のパターン）のような
  転置入力を伴う形状では ACL ヒューリスティクが成立しうるため、
  forward と backward でディスパッチ先が異なる可能性がある（未検証）

## 6. スレッド方針

### 6.1 ATen intra-op スレッド数の既定値

`at::internal::intraop_default_num_threads()`（`ParallelCommon.cpp:103-130`）:

1. `OMP_NUM_THREADS`／`MKL_NUM_THREADS` 環境変数があればそれを使う
2. 未設定の場合、**macOS + aarch64 限定**で
   `sysctlbyname("hw.perflevel0.physicalcpu", ...)`（`:121`）により
   P コア（performance core）数のみを取得し、1 より大きければそれを
   既定スレッド数として即 return する（「Apple Silicon では P コアのみに
   制限する」という明示コメント付き）
3. それ以外（Linux aarch64 を含む）は `TaskThreadPoolBase::defaultNumThreads()`
   （`caffe2/utils/threadpool` 系。実体は概ね
   `std::thread::hardware_concurrency()` 相当）に委ねる

M4 Max で `torch.get_num_threads()=12`（既存実測
`docs/perf/cpu-gemm-baseline-remeasurement.md:59`。P コア数と一致）、
GB10 で `torch.get_num_threads()=20`（`nproc=20` と一致。
`env_info.txt`）という既存記録は、この分岐と整合する。

### 6.2 `set_num_threads` と BLAS 側スレッドの連動

`ParallelNative.cpp` の `init_num_threads()`（`:201-207`）が明示的に
制御しているのは **OpenMP**（`omp_set_num_threads(1)`）と **MKL**
（`mkl_set_num_threads(1)`）のみで、Accelerate（vecLib）・NVPL・ACL 向け
のスレッド数設定呼び出しはソース上見当たらない。したがって
`torch.set_num_threads(n)` は ATen 側の `parallel_for` fork-join
スレッド数を変えるが、BLAS ライブラリ内部（`sgemm_` 呼び出し自体の
並列化）のスレッド数を変えるかは各 BLAS 実装依存であり、PyTorch 側
ソースからは断定できない。

### 6.3 fandhe との対照

fandhe の CPU 並列は rayon のグローバルスレッドプール 1 つのみ
（GEMM も要素ごと演算も同じプールを共有）。PyTorch は「ATen intra-op
プール（要素ごと演算・`parallel_for`）」と「BLAS ライブラリ内部の
スレッド管理（GEMM 本体）」が分離しており、両者のスレッド数は
独立に決まりうる（GEMM は BLAS ライブラリのデフォルトスレッド数、
要素ごと演算は ATen の intra-op スレッド数）。

## 7. 小形状の固定費

### 7.1 要素ごと演算の逐次フォールバック

`TensorIterator.h:76` が `at::internal::GRAIN_SIZE = 32768` を定義し、
`TensorIterator::for_each` の既定 grain size として使われる
（`TensorIterator.h:444`）。`at::parallel_for`（`Parallel-inl.h:9-41`）は

```
use_parallel = numiter > grain_size && numiter > 1
               && !in_parallel_region() && get_num_threads() > 1
```

が偽なら `internal::invoke_parallel`（rayon 相当のタスク分割・
`ParallelNative.cpp:144-193`）を経由せず、呼び出しスレッドで
そのままループ本体を実行する（fork-join 自体が発生しない）。

ベンチ形状（`bench_py.py:18` `BATCH, D_IN, D_HID, D_OUT = 64, 784,
256, 10`）での要素数:

| op | 出力要素数 | 32768 との関係 |
|---|---:|---|
| `x @ w1`（784→256 の mm 本体は BLAS 側で別途並列） | 64×256=16384 | 未満 |
| `+ b1`（broadcast add） | 64×256=16384 | 未満 |
| `relu` | 64×256=16384 | 未満 |
| `h @ w2`（同上） | 64×10=640 | 未満 |
| `+ b2` | 64×10=640 | 未満 |
| `(pred-y)**2` | 64×10=640 | 未満 |
| backward 側の要素ごと演算（relu マスク・sub 等） | 同程度 | 未満 |

いずれも 32768 未満のため、**PyTorch 側の要素ごと演算（add・relu・
MSE 系）は GRAIN_SIZE の逐次フォールバックにより fork-join オーバー
ヘッドを払わない見込み**（実測ではなくソース上の条件からの推定。
実測確認は Phase 3）。GEMM 本体（`x @ w1` 等）は BLAS ライブラリ内部の
並列化に委ねられるため本節の対象外。

`.mean()`（640→1 の縮約）は要素ごと演算ではなく縮約演算であり、
`TensorIterator::for_each` の grain size のみからは逐次／並列の
判定ができない（縮約は一般に専用の縮約カーネル・グレインサイズ計算
〈部分和の木構造・SIMD 縮約等〉を持ちうるため）。本解析では `mean`
の縮約ディスパッチ経路・並列化条件をソース上確認できていないため、
上記の逐次実行という結論から明示的に除外する。確認は Phase 3
（#2098）へ申し送る。

### 7.2 メモリ確保

`c10/core/impl/alloc_cpu.cpp` の `alloc_cpu`（`:105/112/119/126` の
プラットフォーム別分岐。Linux は `posix_memalign`〈`:126`〉、アライメント
は `c10/core/alignment.h` の `gAlignment`。分岐は CPU アーキテクチャ
（x86_64 系／その他）ではなく **`#ifdef C10_MOBILE`**〈モバイルビルド
のみ 16B・それ以外（デスクトップ・サーバー向けビルド）は 64B〉で、
`gAlignment = 64`（`:15`）が非モバイルビルドの値。解析対象の
M4（macOS デスクトップビルド）・GB10（Linux サーバー向け cu130
ホイール）はいずれも非モバイルビルドのため **64B**）
は呼び出しごとに直接確保する薄い wrapper で、専用キャッシュは見当たら
ない（PyTorch 全体としては別途 `CachingHostAllocator` 等が CUDA pinned
memory 向けに存在するが、通常の CPU テンソル確保はこの直接経路）。

### 7.3 fandhe との対照

`docs/perf/lowlayer-diagnosis-2026-09-12.md:104` は fandhe の MSE
backward（要素数 640。上記 PyTorch と同形状）が rayon の fork-join
固定費として約 70 µs かかると記録している。fandhe には PyTorch の
GRAIN_SIZE 相当の「小要素数は逐次実行する」閾値が本番経路に結線されて
いない（`MSE_BACKWARD_PARALLEL_MIN_ELEMS=0`。§8 参照）ため、この
640 要素規模でも常に rayon タスクを起こす。**これが CPU train の
小形状固定費における主要な構造差の候補**である（ただし #1578 の
既存判定は「REJECT」——後述 §8 の対照表参照）。

## 8. fandhe との対照表

| PyTorch の機構（file:line） | fandhe の現状（file:line・const 値） | 既存判定（イシュー・doc） | 新規差分候補か |
|---|---|---|---|
| `addmm_impl_cpu_` 単一入口（`LinearAlgebra.cpp:1392`）で mm/addmm 共通化 | `nn::Linear::forward`（`crates/autodiff/src/nn/linear.rs:287`）は非融合 matmul→add。`forward_with_activation`（`:311`）が epilogue 融合版として別途存在 | なし（設計方針の相違） | 対象外（§9） |
| GEMM は BLAS ライブラリのグローバルスレッド管理に委ねる（ATen intra-op プールと分離） | rayon 単一グローバルプール（GEMM も要素ごと演算も共有）。`gemm_blis_parallel`（`crates/backend-cpu/src/gemm_blis/mod.rs:476`）・`gemm_blis_bias_act_parallel`（`:713`） | #1350（CUDA 側 Graph 化の文脈。CPU 側スレッド分離は未検討） | 候補（§9） |
| P コア限定スレッド数（macOS+aarch64 限定。`ParallelCommon.cpp:121`） | `BIG_CORE_LIMIT_ENABLED=false`（`crates/backend-cpu/src/thread_limit.rs:112`。既定 OFF） | #1363／#1364: REJECT（`docs/perf/cpu-gemm-default-thread-limit.md` §6） | 既存判定どおり対象外 |
| GRAIN_SIZE=32768 の要素ごと演算逐次フォールバック（`TensorIterator.h:76`） | `MSE_BACKWARD_PARALLEL_MIN_ELEMS=0`（`crates/backend-cpu/src/mse.rs:68`。常に並列） | #1578: REJECT（`docs/perf/cpu-mse-backward-sequential-threshold.md`） | 既存判定どおり対象外（ただし §7.3 のとおり構造差自体は実在） |
| BLAS 呼び出しは 1 回の `sgemm_`（`CPUBlas.cpp:236`）。GEMM 内部の分割は BLAS 実装依存で不可視 | `PanelBuffers::new`（`gemm_blis/mod.rs:411`）が呼び出し・rayon タスクごとに `vec![0.0f32; ..]` で A/B パネルを確保 | #1481／#1482: REJECT（`docs/perf/cpu-matmul-fixed-cost-impl.md`。alloc 固定費は棄却済み） | 既存判定どおり対象外 |
| aarch64 ACL ヒューリスティク（`transpose_c && transpose_a && !transpose_b`。`LinearAlgebra.cpp:1505`）で転置形状を別カーネルへ | `_nt`／`_tn` 専用入口（`gemm_blis_parallel_nt`〈`:651`〉／`gemm_blis_parallel_tn`〈`:670`〉） | #1577（ReLU マスク stride 対応。`docs/perf/train-reuse-relu-mask-stride.md`） | 既存判定どおり対象外 |
| GB10（Linux aarch64）は `TaskThreadPoolBase::defaultNumThreads()`＝論理コア数全部（`ParallelCommon.cpp:126`） | `GB10_AFFINITY_ENABLED=false`（`crates/backend-cpu/src/gb10_affinity.rs:126`。既定 OFF） | #1576: undetermined（`docs/perf/cpu-gemm-gb10-affinity-ab.md`） | 既存判定どおり対象外 |
| `set_num_threads` は OpenMP／MKL のみ制御（`ParallelNative.cpp:207`）。Accelerate/NVPL/ACL への連動は未確認 | rayon プール 1 本（バックエンド分離なし） | なし | 候補（§9。ATen intra-op と BLAS スレッドの分離という設計差） |
| `GEMM_THREADING_THRESHOLD`（`gemm_blis/mod.rs:225`）は `#[cfg(test)]` 限定で本番未結線 | 同上（fandhe 側の事実として記録） | #811／#1027: 本番未結線（`docs/perf/cpu-gemm-small-shape-serial-fallback.md`） | 既存判定どおり対象外 |
| `SMALL_SHAPE_CAP_ENABLED=false`（fandhe 側） | 同左（`crates/backend-cpu/src/small_shape_thread_cap.rs:87`） | #1575: REJECT（`docs/perf/cpu-gemm-small-shape-thread-cap.md`） | 既存判定どおり対象外 |

## 9. 差分候補（既存判定に対応しない行のみ。起票はしていない）

以下は #2098（Phase 3）の入口として列挙するのみで、Issue 化は
ユーザー承認が必要（`.claude/rules/out-of-scope-tracking.md`）。

1. **GEMM 用スレッドプールと要素ごと演算用スレッドプールの分離**:
   PyTorch は BLAS ライブラリ（Accelerate／NVPL 等）の内部スレッド管理と
   ATen intra-op プールが構造的に分離しているのに対し、fandhe は
   rayon 単一グローバルプールで両方を賄う。この構造差が train の
   backward（GEMM 系と要素ごと演算が交互に発生する区間）でスレッド
   起床・synchronization コストにどう影響するかは未検証
2. **`torch.set_num_threads` と BLAS 内部スレッド数の非連動**の裏を返すと、
   fandhe の rayon プールはスレッド数変更が GEMM・要素ごと演算の両方に
   即座に効く一方、意図しない競合（GEMM 実行中に要素ごと演算タスクが
   同じプールへ enqueue される等）が起きうるかは未検証

## 10. 限界・申し送り

- **GB10 のディスパッチ先は実行時確認していない**（本解析はビルド
  スクリプト・既存実測ログからの推定に留まる）。Phase 3 で
  `torch.__config__.show()`（`BLAS_INFO`／`USE_MKLDNN_ACL` フィールド）と
  `DNNL_VERBOSE=1` 実行時ログにより、実際の `mm`／`addmm` 呼び出しが
  oneDNN（ACL）と BLAS（NVPL）のどちらを通ったかを確認する必要がある
- **`docs/perf/cpu-gemm-baseline-remeasurement.md` の `BLAS_INFO=accelerate`
  実測は torch 2.13.0 でのプローブ**であり、本解析対象の 2.14.0 では
  ない。§3〜§4 のソースコード解析（BLAS 選択ロジック・`sgemm_` 呼び出し）
  は 2.14.0 タグ時点のものであり、2.13.0→2.14.0 間で当該コードパスに
  変更が入っていないかまでは確認していない（変更履歴の diff は本解析の
  スコープ外）
- **Accelerate 内部（AMX／SME どちらのマイクロカーネルが選ばれるか）は
  閉源のため到達不能**。`docs/perf/cpu-gemm-sme-fmopa-microkernel.md` の
  既存記録を参照するのみで、本解析では新たな推測を加えない
- **数値は実測として提示していない**。既存の実測値（785.9 µs・
  1130.5 µs・0.20 ms・33 µs・367 µs・140 µs・70 µs 等）は出典を付けて
  「参考」として引用したのみで、本解析自体は新規計測を行っていない
- 差分候補（§9）の実装・A/B は Phase 3（#2098）へ引き継ぐ。新規 Issue
  起票にはユーザー承認が必要

## 11. ライセンス記録

- **PyTorch**: タグ `v2.14.0` の `LICENSE` ファイルは BSD 系（「From
  PyTorch」「From Caffe2」を含む複数の Copyright 表記＋標準 3 条項 BSD
  文言）。本解析は上流コードを逐語転記せず、関数名・マクロ名・
  `path:line`・要約のみを記録しているため、依存追加ではなく比較対象
  としての調査利用にとどまる。`docs/license-matrix.md` の区分表には
  追加しない（依存の追加を伴わないため）
- **Accelerate**: Apple 独自（クローズドソース）。本解析ではライブラリ
  境界（`sgemm_` 呼び出し）までしか到達しておらず、内部実装には
  触れていない

## 12. 出典

- PyTorch v2.14.0: `https://github.com/pytorch/pytorch/tree/v2.14.0`
  （解決済みコミット `2b3ec34829036a65cd9d1398ea72a0167dc37470`）
- `docs/perf/cpu-gemm-baseline-remeasurement.md`
- `docs/perf/lowlayer-diagnosis-2026-09-12.md`
- `docs/perf/train-step-phase-breakdown.md` §17.2／§17.6
- `docs/perf/logs/framework-compare-0.9.0-remeasure/scoreboard/gen_090.py`
- `scripts/bench/framework-compare/results/raw/results-dgx-py-0.8.0.jsonl`
- `docs/perf/logs/parity-torch-cpu-truth-1985/env_info.txt`
- `docs/perf/cpu-gemm-default-thread-limit.md`
- `docs/perf/cpu-gemm-small-shape-thread-cap.md`
- `docs/perf/cpu-gemm-gb10-affinity-ab.md`
- `docs/perf/cpu-mse-backward-sequential-threshold.md`
- `docs/perf/cpu-matmul-fixed-cost-impl.md`
- `docs/perf/cpu-gemm-small-shape-serial-fallback.md`
- `docs/perf/train-reuse-relu-mask-stride.md`
- `docs/perf/cpu-gemm-sme-fmopa-microkernel.md`
- `crates/backend-cpu/src/gemm_blis/mod.rs`・`thread_limit.rs`・
  `small_shape_thread_cap.rs`・`gb10_affinity.rs`・`mse.rs`・`ops.rs`
- `crates/autodiff/src/nn/linear.rs`
