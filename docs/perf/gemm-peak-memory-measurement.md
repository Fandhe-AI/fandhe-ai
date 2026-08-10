# GEMM 4096³ ピークメモリ実測記録（#178・TASK-14.2a）

イシュー #178「test(backend): TASK-14.2a GEMM 4096³ ピークメモリの実測」の実測記録。
受け入れ条件「バックエンド別ピーク値の実測記録が残されている」に対応する。計測ハーネス自体は
本イシューで新規整備した（`crates/bench-harness/src/peak_memory.rs`・`Makefile` の
`peak-memory-bench` ターゲット）。内部計測 API 自体は TASK-14.1（#173〜#176）で実装済みの
`tensor_core::memory_stats::{MemoryStats, AllocationTracker, TrackedAllocation}`
（`crates/tensor-core/src/memory_stats.rs`）を用いる。

係数の確定・再調整（REQ-14 の「384MiB 以内」判定）は兄弟イシュー #179（TASK-14.2b）が
本記録を入力として実施する。計測手段の環境差文書化は #180（TASK-14.3）のスコープであり、
本ドキュメントは実測データの記録に留める。

## 状態: CPU 実測済み・Metal 実機実測済み（#385）・CUDA 実機実測済み（#392）

本実装セッションは Linux x86_64（QEMU/KVM 仮想化環境、NVIDIA RTX 3060 passthrough）
worktree で行っており、`libnvrtc`（CUDA toolkit）が導入されていない
（`ldconfig -p | grep nvrtc` で未検出）。「Metal（Apple Silicon）実機はこのセッションから
利用できない」という当時の前提は、イシュー #385（本記録内「Metal 実機実測結果」節）で
Apple Silicon 実機（Apple M4 Max・macOS 26.6）から直接実行し解消済みである。CUDA
（DGX Spark GB10）についても、イシュー #392（本記録内「CUDA 実機実測結果」節）で
実機（NVIDIA GB10・CUDA 13.0）から SSH リモート実行し解消済みである。

既存の先例（`docs/perf/startup-cost-measurement.md`〈#171〉・
`docs/perf/dispatch-boundary-measurement.md`〈#69〉・
`docs/perf/cuda-tensor-core-measurement.md`〈#64〉）と同じ運用を採った:
**CPU は本セッションで実測を完了**し、**CUDA（DGX Spark GB10）・Metal（Apple Silicon）は
当初、再現可能な計測手順と結果転記テンプレートのみを整備**して「実機未実施」と明記して
いたが、Metal は #385・CUDA は #392 でそれぞれ実機実測を完了し解消済みである。
CUDA については、当初の本環境（libnvrtc 未導入）でも動的ロード契約により実行時に panic
せず型付きエラーへ fail-closed に倒れることを実地確認していた（「CUDA 実行時エラーの
実地確認」節。当時の記録として維持する）。

## 環境

| 項目 | 値 |
|------|-----|
| OS | Linux 7.0.0-28-generic（Ubuntu、`uname -a` 実測） |
| CPU | 12 vCPU（QEMU/KVM 仮想化、`lscpu` 実測） |
| GPU | NVIDIA GeForce RTX 3060（`nvidia-smi -L` で検出。CUDA toolkit 非搭載のため実行時未使用） |
| rustc | 1.96.0（`rustc --version` 実測） |
| ビルドプロファイル | `--release`（`cargo build -p bench-harness --release --bins`） |

`cpu-run1.json`／`cpu-run2.json` は PR #370 codex-review 指摘 P1（`gemm_alloc_peak_bytes`
未実装時点の旧スキーマ実測データが残存していた）を受け、`gemm_alloc_peak_bytes`
実装後の同一環境・同一手順（上表）で再実測した値に更新済み（本節以下の数値も同時に更新）。

### 環境（Metal。イシュー #385）

| 項目 | 値 |
|------|-----|
| チップ | Apple M4 Max（`sysctl -n machdep.cpu.brand_string` 実測。依存イシュー #380 の
  `docs/backend-metal-real-device-testing.md`「実行環境（#380）」節と同一実機） |
| OS | macOS 26.6（build 25G72。`sw_vers` 実測） |
| rustc | 1.96.0 (ac68faa20 2026-05-25)（`rustc --version` 実測） |
| toolchain | `stable-aarch64-apple-darwin`（`rust-toolchain.toml` 単一真実源） |
| ビルドプロファイル | `--release`（`cargo build -p bench-harness --release --bins`） |
| 計測日 | 2026-08-10 |
| 実行方式 | ローカル直接実行（SSH・転送不要。`docs/real-hardware-verification-env.md` 7.1 節
  「Mac 上でそのまま実行」） |

### 環境（CUDA。イシュー #392）

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA GB10（sm_121。`nvidia-smi -L` 実測。driver 580.159.03） |
| OS | Linux 6.17.0-1026-nvidia aarch64（`uname -srm` 実測） |
| CUDA | 13.0.88（`nvcc --version` 実測。`V13.0.88`） |
| toolchain / rustc | `stable`・rustc 1.97.0 (2d8144b78 2026-07-07)（`rustc --version` 実測） |
| ビルドプロファイル | `--release`（`cargo build -p bench-harness --release --bins`） |
| 計測リビジョン | `4ba5365a0d7e68fd54b50412f268278465503a40`（`.rev-stamp` 実測。転送後に
  `ssh … cat .rev-stamp` で worktree の `git rev-parse HEAD` と一致確認済み） |
| 計測日 | 2026-08-10 |
| 実行方式 | SSH リモート実行・rsync 転送（`local.fandhe.spark-dbd9`。
  `docs/real-hardware-verification-env.md` 2〜3 節） |
| 実行コマンド | `$CARGO_TARGET_DIR/release/peak_memory_bench --backend cuda --trials 5
  --out docs/perf/peak-memory/cuda-run{1,2}.json`（ビルド済みバイナリを直接実行。
  `cargo run` 経由にすると孫プロセスの計測混入を避けにくいため回避。PR #445 と同型） |
| GPU 占有状況（計測前後） | 計測前後とも常駐 2 プロセスのみ（ComfyUI 約 170MiB・
  Kokoro TTS 約 870MiB）・`utilization.gpu` 0%（`nvidia-smi --query-compute-apps` /
  `--query-gpu=utilization.gpu` 実測。他プロセスの介入なし） |

## 計測方法

`cargo run -p bench-harness --release --bin peak_memory_bench -- --backend <cpu|cuda|metal>
--trials 5 --out <path.json>`（`make peak-memory-bench BACKEND=<...> TRIALS=5` が内部で
同一のバイナリ呼び出しを行う）を用いた。GEMM サイズは既定値（`--size` 省略時 4096、
REQ-14 の代表ワークロード M=N=K=4096, f32）。

### 計測手順（1 trial。`crates/bench-harness/src/peak_memory.rs` モジュールコメント参照）

1. バックエンド入口（`CpuMemory::new()`／`CudaMemory::new(&CudaDevice)`／
   `MetalMemory::new(MetalContext)`）を単一インスタンスで構築し `reset_peak()`
2. `MemoryOps` 経由で代表ワーキングセットを確保: `upload(A)`（4096×4096 f32 = 64MiB）→
   `upload(B)`（64MiB）→ `alloc_zeroed(C)`（64MiB）。期待計上値 =
   201,326,592 バイト（192MiB）
3. ワーキングセット保持中に `BackendOps::gemm(A, B)` を実行（所要秒数を記録）。CPU
   バックエンドはこの区間を `bench_harness::alloc_tracker::measure()`（`peak_memory_bench`
   バイナリが `TrackingAllocator` を `#[global_allocator]` として宣言するため有効になる）
   で包み、GEMM 実行区間中の実ヒープ確保量の純増分ピークを `gemm_alloc_peak_bytes` として
   採取する（PR #370 codex-review 指摘 P1 対応。「計測境界」節参照）
4. `peak_allocated_bytes()`（`peak_bytes`）を採取
5. A・B・C の 3 バッファを drop し `allocated_bytes()`（`allocated_after_drop_bytes`）が
   0 に戻ることをリーク検査として記録
6. 補助参考値として Linux では `/proc/self/status` の `VmHWM` を `vm_hwm_bytes` に記録する
   （macOS は `libc` クレートの新規追加が必要になるため本イシューでは未実装。#180 へ申し送り）

試行は 5 回・中央値を採用（`.claude/rules/coding-rust.md` ベンチ規約）。行列データは
`bench_harness::rng::Xorshift64Star`（決定的シード。trial ごとに系列を変えつつ再現可能）で
生成した。

### 計測境界（重要）

[`MemoryStats`] が計上するのは `MemoryOps`（`alloc_zeroed`／`upload`）経由のデバイスバッファ
確保**のみ**である。`BackendOps::gemm` 演算内部の一時確保（CPU: `CpuBackendOps::gemm` の
出力 `Vec<f32>` と BLIS パッキングバッファ、CUDA: `CudaGemm::run_tiled_f32` の stream 直接
確保、Metal: `MetalGemm` の直接確保）は計測対象外である（`tensor_core::memory_stats`
モジュールコメント「計測対象の粒度」）。`BackendOps::gemm` は `&Tensor<f32>` を直接受け取る
API であり `DeviceBuffer`（`MemoryOps` の確保結果）を経由しないため（各バックエンドの
`ops.rs` 参照）、本ハーネスの手順は「GEMM 実行に必要な最小限のデバイス常駐量を模した計測」
であり、GEMM カーネル内部の一時確保量そのものを計測するものではない。この分離自体が
本イシューの計測境界であり、下記「内部 API 値と外部参考値の乖離」節がその影響範囲を示す。

CPU バックエンドに限り、この計測境界の外側（GEMM 実装内部の実ヒープ確保）を
`gemm_alloc_peak_bytes`（`bench_harness::alloc_tracker`。PR #370 codex-review 指摘 P1 対応）
が埋める。`std::alloc::GlobalAlloc` フックによる実測のため、CUDA（`cudarc` driver 確保）・
Metal（`objc2-metal` `MTLBuffer`）は Rust の `GlobalAlloc` を経由せず引き続き対象外（`None`）
である。

## CPU 実測結果

`peak_memory_bench --backend cpu --trials 5` を `--out` 付きで 2 セット実行した。生 JSON は
`docs/perf/peak-memory/cpu-run1.json`・`cpu-run2.json`（全試行の `samples` 込み）。

| セット | peak_bytes（中央値 / Q1 / Q3） | 理論最小ワーキングセット | 対理論比 | gemm_secs（中央値 / Q1 / Q3, 秒） | vm_hwm_bytes（参考値） | gemm_alloc_peak_bytes（中央値 / Q1 / Q3） |
|--------|-------------------------------|------------------------|---------|-----------------------------------|------------------------|---------------------------------------------|
| run1 | 201,326,592 / 201,326,592 / 201,326,592 | 201,326,592 | 1.000 | 0.2165 / 0.2144 / 0.2191 | 346,324,992（全 trial 同値） | 75,065,424 / 75,065,424 / 75,071,568 |
| run2 | 201,326,592 / 201,326,592 / 201,326,592 | 201,326,592 | 1.000 | 0.2064 / 0.2059 / 0.2179 | 346,501,120（全 trial 同値） | 75,065,424 / 75,059,280 / 75,065,424 |

**内部計測 API のピーク値は 5 試行すべて・2 セットとも決定的に 201,326,592 バイト（192MiB）
ちょうど**であり、理論最小ワーキングセット（A+B+C 各 64MiB の合計）と完全一致した（対理論比
1.000）。これは `MemoryOps` 経由の確保が A・B・C の 3 バッファのみであり、他の中間確保を
一切含まないという実装契約どおりの結果である。drop 後の `allocated_after_drop_bytes` は
全 10 trial（2 セット × 5 試行）で 0 を記録し、リークは検出されなかった。

**`gemm_alloc_peak_bytes`（GEMM 実行区間中の実ヒープ確保量の純増分ピーク）は両セットとも
中央値 約 71.6MiB（75,065,424 バイト）**であり、`peak_bytes`（`MemoryOps` 経由・192MiB
ちょうどで決定的）とは独立に変動する実測値であることを確認した（`crates/backend-cpu/src/
ops.rs` の出力バッファ `vec![0.0f32; m * n]`〈64MiB〉のみでは説明できない超過分 約
7.6MiB〈7,956,560 バイト〉が観測されており、これが `gemm_blis_parallel`
〈`crates/backend-cpu/src/gemm_blis.rs`〉の BLIS パッキングバッファ等に相当すると考えられる）。

### 内部 API 値と外部参考値（`VmHWM`）の乖離

`VmHWM`（プロセス全体のピーク常駐セットサイズ）は run1 約 330.3MiB（346,324,992 バイト）・
run2 約 330.4MiB（346,501,120 バイト）であり、内部 API 値（192MiB）に対し **約 1.72 倍**
大きい。両セットの `VmHWM` は全 trial で同一値（プロセス全体の生涯ピークのため、5 trial 目まで
に到達した最大値が記録される）。

差分（約 138MiB）の主な要因は、計測境界の外側にある GEMM 演算内部の一時確保である。
本イシューで新規実装した `gemm_alloc_peak_bytes`（PR #370 codex-review 指摘 P1 対応）に
より、このうち **約 71.6MiB（75,065,424 バイト・中央値）は GEMM 実行区間中の実ヒープ確保
として実測できた**（上表「CPU 実測結果」参照）:

- `CpuBackendOps::gemm`（`crates/backend-cpu/src/ops.rs`）の出力バッファ
  `let mut out = vec![0.0f32; m * n];`（4096×4096 f32 = 64MiB）
- `gemm_blis_parallel`（`crates/backend-cpu/src/gemm_blis.rs`）内部の BLIS 型ブロッキング・
  パッキングバッファ等（出力バッファ超過分 約 7.6MiB に相当。rayon 並列ワーカー数分の内訳
  までは本イシューでは分解しない）
- `Tensor::contiguous()` による A・B の実体化コピー（`gemm` 呼び出し内で `a.contiguous()`・
  `b.contiguous()` を呼ぶため、`upload` 前の host 側 Tensor とは別に、GEMM 実行時点でさらに
  A・B 相当のコピーが一時的に存在しうる。今回の入力は生成直後で既に contiguous のため
  ノーコピーだった可能性が高く、`gemm_alloc_peak_bytes` の超過分〈約 7.6MiB〉の主因は
  BLIS パッキングバッファ側と考えられる）

理論最小ワーキングセット（192MiB）+ `gemm_alloc_peak_bytes` 中央値（約 71.6MiB）=
約 263.6MiB に対し `VmHWM`（約 330.3〜330.4MiB）はなお **約 66.7〜66.8MiB 上回る**。
この残差は `gemm_alloc_peak_bytes` の計測境界（`BackendOps::gemm` 実行区間中の純増分の
み）の外側（プロセス起動時の初期ヒープ・スレッドプール〈rayon ワーカー〉のスタック確保・
`upload`/`alloc_zeroed` 自体のホスト側一時確保等）に起因すると考えられ、本イシューでは
個別の内訳計測までは行わない。

**判断材料としての申し送り**: 内部 API 値（192MiB・対理論比 1.000）に対し `VmHWM` 参考値
（約 330MiB・対理論比 約 1.72）には無視できない乖離がある。`gemm_alloc_peak_bytes` の実測
（約 71.6MiB）によりこの乖離の大部分（約半分）は GEMM 実装内部の実ヒープ確保で説明できる
ことが分かったが、残り約 66.7〜66.8MiB は依然未分解である。REQ-14 の係数上限
（384MiB 以内 = 対理論比 2.0 以内）と比較すると、内部 API 値だけを見れば余裕があるが、
プロセス全体のピーク常駐（GEMM 内部確保を含む実態）は対理論比 1.72 まで到達しており、
係数の余裕は内部 API 値が示すほど大きくない可能性がある。係数の確定・再調整は、この
`gemm_alloc_peak_bytes` 実測データ（分解済みの約半分）を踏まえて #179（TASK-14.2b）で
判断することを推奨する。

### 再現性確認

run1・run2 の 2 セットとも内部 API ピーク値は完全に同一（201,326,592 バイト）であり、
`gemm_secs` の中央値も 0.2165 秒・0.2064 秒と近い値（差 約 5%）で安定していた。
`vm_hwm_bytes` も両セットでほぼ同一（差 約 172KiB、プロセス起動オーバーヘッドの揺らぎ程度）。
`gemm_alloc_peak_bytes` の中央値も両セットとも 75,065,424 バイトで完全一致した。

## CUDA 実行時エラーの実地確認

`cargo run -p bench-harness --release --bin peak_memory_bench -- --backend cuda --trials 1`
を実行し、以下の結果を得た（本環境は `libcuda` は passthrough 経由で存在するが
`libnvrtc`〈CUDA toolkit〉が未導入）:

```
計測失敗: バックエンド呼び出し失敗: CUDA unavailable: CUDA NVRTC library unavailable:
libnvrtc dynamic library not found (dlopen failed); CUDA toolkit is not installed or
not on the library search path
```

`CudaDevice::is_available()`（`libcuda` の dlopen プローブ）は本環境では `true` を返すため
（passthrough により `libcuda` 自体は存在）、`CudaDevice::new(0)` は成功し `CudaMemory`
構築・`upload`/`alloc_zeroed` も成功する。失敗は `BackendOps::gemm`
（`CudaBackendOps::gemm` → `CudaGemm::new` → NVRTC ライブラリのロード）の時点で発生する。
プロセスは panic せず非ゼロ終了コード（`ExitCode::FAILURE`）で終了し、
`PeakMemoryError::Backend` として型付きエラーが呼び出し元へ返る（fail-closed 契約の実地確認。
`crates/backend-cuda/src/device.rs` の動的ロードゲート方針と整合）。

## Metal 実機実測結果（イシュー #385）

`peak_memory_bench --backend metal --trials 5` を `--out` 付きで 2 セット実行した（実行
環境は上記「環境（Metal。イシュー #385）」節）。生 JSON は
`docs/perf/peak-memory/metal-run1.json`・`metal-run2.json`（全試行の `samples` 込み）。

| セット | peak_bytes（中央値 / Q1 / Q3） | 理論最小 | 対理論比 | gemm_secs（中央値 / Q1 / Q3, 秒） | vm_hwm_bytes | gemm_alloc_peak_bytes |
|---|---|---|---|---|---|---|
| metal-run1 | 201,326,592 / 201,326,592 / 201,326,592 | 201,326,592 | 1.000 | 0.050730 / 0.049709 / 0.050758 | -（macOS 未実装） | -（`GlobalAlloc` 非経由） |
| metal-run2 | 201,326,592 / 201,326,592 / 201,326,592 | 201,326,592 | 1.000 | 0.048812 / 0.048437 / 0.049373 | - | - |

**内部計測 API のピーク値は run1・run2 とも 5 試行すべてで決定的に 201,326,592 バイト
（192MiB）ちょうど**であり、理論最小ワーキングセットと完全一致した（対理論比 1.000）。
`allocated_after_drop_bytes` は全 10 trial（2 セット × 5 試行）で 0 を記録し、リークは
検出されなかった。run1／run2 間で `peak_bytes` は完全一致（再現性確認）、`gemm_secs`
中央値の差は約 4%（0.050730 秒 対 0.048812 秒）に収まった。

### AC2: 係数上限（384MiB 以内 = 対理論比 2.0 以内）の充足可否

**内部 API 値（対理論比 1.000）は REQ-14 初期リリース係数上限 2.0 を余裕をもって満たす**
（超過なし。上限値 2.0 は本イシューでは変更しない。変更はユーザー承認必須。
`.claude/rules/coding-rust.md`・`docs/real-hardware-verification-env.md`）。

### 計測境界の限界（Metal 固有。「計測境界（重要）」節の適用）

Metal の `peak_bytes` も CPU 同様「計測境界（重要）」節のとおり `MemoryOps` 経由
（`upload(A)`・`upload(B)`・`alloc_zeroed(C)`）の確保のみを計上する。`MetalGemm`
（`crates/backend-metal/src/ops.rs`）がカーネル実行のために直接確保する `MTLBuffer`
（`objc2-metal` 経由）は計測対象外である。したがって **対理論比が 1.000 に張り付くのは
「Metal のワーキングセットが理論最小どおりだった」ことの証明ではなく、この計測手法の
設計上の性質**（`MemoryOps` 経由の 3 バッファ以外を計上しない契約）であることを踏まえて
数値を読む必要がある。

CPU 実測では `VmHWM`（約 1.72 倍）という外部対照があったが、**macOS では
`vm_hwm_bytes` が構造的に `None`**（`read_vm_hwm_bytes` は `#[cfg(target_os = "linux")]`
実装のみ。`libc` クレートの新規追加は許容依存 8 区分外のため実装しない既存判断。
「スコープ外・申し送り」節参照）であり、`gemm_alloc_peak_bytes` も Metal は契約上常に
`None`（`GlobalAlloc` を経由しないため）。したがって内部 API 値に対するプロセス／デバイス
レベルの相互検証手段が macOS には構造的に存在しない。

参考として、ビルド済みバイナリを `/usr/bin/time -l` で直接計測した（`cargo run` 経由に
すると孫プロセスのピーク RSS が確実に合算されないため回避）:

```
/usr/bin/time -l ./target/release/peak_memory_bench --backend metal --trials 5
```

- `maximum resident set size`: 552,386,560 バイト（約 526.8MiB。理論最小の約 2.74 倍）
- `peak memory footprint`: 711,345,040 バイト（約 678.4MiB。理論最小の約 3.53 倍）

いずれも理論最小ワーキングセット（192MiB）を大きく上回る妥当な値であり（サニティゲート
「約 200MiB を下回れば誤計測」は満たさない＝正常値）、Apple Silicon の統合メモリでは
RSS が `MTLBuffer` 相当の確保も含みうることと整合する。**この値は「ハーネス外の粗い
参考値」に留め、内部 API 値との厳密比較（対理論比の算出等）には用いない**（プロセス
起動オーバーヘッド・Metal デバイス／ライブラリの内部確保・シェーダコンパイル用バッファ等
の内訳分解は行わない）。実 `MTLBuffer` 確保量そのものの計測フックは既存申し送りどおり
別イシューのスコープである。

### `gemm_secs` の解釈（Metal 固有）

`MetalBackendOps::gemm`（`crates/backend-metal/src/ops.rs:74-76`）は呼び出しごとに
`MetalContext::new()` と `MetalGemm::new(&ctx)`（`gemm.metal` のランタイム MSL コンパイル
とパイプライン構築）を実行する。`run_metal_trial`
（`crates/bench-harness/src/peak_memory.rs:788` 付近）の計測窓はこの `ops.gemm()`
呼び出し全体を含むため、**Metal の `gemm_secs` は「デバイス／ライブラリ構築＋シェーダ
コンパイル＋カーネル実行」の合計**であり、CPU の `gemm_secs`（カーネル実行のみ）とは
直接比較できない。両セットの `samples`（生 JSON 参照）を確認したところ、run1 の 1 trial 目
（0.064219 秒）が他 4 trial（0.048〜0.051 秒）より明確に大きく、初回のシェーダコンパイル
コストが乗った外れ値と考えられる。run2 では同様の外れ値は観測されなかった（1 trial 目
0.052226 秒は他との差が小さい）。計測時に他プロセスの GPU 負荷がないことを事前確認して
おり（`ps ax` 実測。PR #437／イシュー #381 のベンチとの競合なし）、`gemm_secs` は
他プロセス由来の変動を含まない。

## CUDA 実機実測結果（イシュー #392）

`peak_memory_bench --backend cuda --trials 5` を `--out` 付きで 2 セット実行した（実行
環境は上記「環境（CUDA。イシュー #392）」節）。生 JSON は
`docs/perf/peak-memory/cuda-run1.json`・`cuda-run2.json`（全試行の `samples` 込み）。
再現性確認として `make peak-memory-bench BACKEND=cuda TRIALS=5`（`--out` なし）も
1 回実行し、`#[ignore]` 分離済みスモークテスト
`cuda_peak_memory_matches_theoretical_minimum`（256³）も実機で実行・pass 済み。

| セット | peak_bytes（中央値 / Q1 / Q3） | 理論最小 | 対理論比 | gemm_secs（中央値 / Q1 / Q3, 秒） | vm_hwm_bytes（参考値） | gemm_alloc_peak_bytes |
|---|---|---|---|---|---|---|
| cuda-run1 | 201,326,592 / 201,326,592 / 201,326,592 | 201,326,592 | 1.000 | 0.210158 / 0.209847 / 0.214925 | 371,097,600〜386,506,752（trial 毎に単調増加） | -（`GlobalAlloc` 非経由。CUDA 契約） |
| cuda-run2 | 201,326,592 / 201,326,592 / 201,326,592 | 201,326,592 | 1.000 | 0.204331 / 0.204243 / 0.206938 | 371,138,560〜382,996,480（trial 毎に単調増加） | - |

**内部計測 API のピーク値は run1・run2 とも 5 試行すべてで決定的に 201,326,592 バイト
（192MiB）ちょうど**であり、理論最小ワーキングセットと完全一致した（対理論比 1.000）。
`allocated_after_drop_bytes` は全 10 trial（2 セット × 5 試行）で 0 を記録し、リークは
検出されなかった。`gemm_alloc_peak_bytes` は全 10 trial で `null`（CUDA は `cudarc` の
driver 確保が Rust の `GlobalAlloc` を経由しないため契約どおり）。run1／run2 間で
`peak_bytes` は完全一致（再現性確認）。

### AC2: 係数上限（384MiB 以内 = 対理論比 2.0 以内）の充足可否

**内部 API 値（対理論比 1.000）は REQ-14 初期リリース係数上限 2.0 を余裕をもって満たす**
（超過なし。上限値 2.0 は本イシューでは変更しない。変更はユーザー承認必須。
`.claude/rules/coding-rust.md`・`docs/real-hardware-verification-env.md`）。

### 計測境界の限界（CUDA 固有。「計測境界（重要）」節の適用）

CUDA の `peak_bytes` も CPU・Metal 同様「計測境界（重要）」節のとおり `MemoryOps` 経由
（`upload(A)`・`upload(B)`・`alloc_zeroed(C)`）の確保のみを計上する。
`CudaGemm::run_tiled_f32`（`crates/backend-cuda/src/ops.rs`）がカーネル実行のために
stream 上へ直接確保するバッファは計測対象外である。したがって **対理論比が 1.000 に
張り付くのは「CUDA のワーキングセットが理論最小どおりだった」ことの証明ではなく、
この計測手法の設計上の性質**（`MemoryOps` 経由の 3 バッファ以外を計上しない契約）で
あることを踏まえて数値を読む必要がある（Metal 節と同じ注意喚起）。

**Metal と異なり、実機ノードが Linux aarch64 のため `vm_hwm_bytes`
（`/proc/self/status` の `VmHWM`。`read_vm_hwm_bytes` は `#[cfg(target_os = "linux")]`
実装）は全 10 trial で non-null が実測できた**（上表参照。CPU と同様の外部対照が存在
する）。ただし GB10 は統合メモリアーキテクチャであり、`cudarc` のデバイス確保が
プロセスの RSS（`VmHWM`）に計上されるか否かは本イシューでは断定できない。このため
`vm_hwm_bytes` は CPU 節と同じ「参考値」に留め、**対理論比の算出には用いない**
（判定対象は内部計測 API 値のみ。`docs/peak-memory-coefficient-decision.md`
「2. 判定対象の定義」）。`vm_hwm_bytes` は trial が進むごとに単調増加しているが、これは
`VmHWM` がプロセス全体の生涯ピーク値であり 5 trial 目までに到達した最大値が記録される
という CPU 節と同じ性質（`gemm-peak-memory-measurement.md`「内部 API 値と外部参考値の
乖離」節）による。

### gemm_secs の解釈（CUDA 固有）

`CudaBackendOps::gemm`（`crates/backend-cuda/src/ops.rs:79`）は呼び出しごとに
`CudaGemm::new`（NVRTC ランタイムコンパイル）を実行するため、`gemm_secs` は
「NVRTC ソース→PTX コンパイル＋H2D/D2H 転送＋カーネル実行」の合計であり、CPU の
`gemm_secs`（カーネル実行のみ）とは直接比較できない。イシュー #391（TASK-14.1 系
CUDA 起動コスト実測）の知見「NVRTC source→PTX コンパイルは `CUDA_CACHE_PATH` の
設定に関係なく毎プロセス発生し、キャッシュが効くのはドライバ側 PTX→SASS 変換のみ」
（`docs/perf/startup-cost-measurement.md`〈#391〉CUDA 節参照）がこの現象の根拠である。
両セットの `samples`（生 JSON 参照）を確認したが、run1・run2 とも 5 試行間の差は
小さく（run1: 0.2097〜0.2227 秒、run2: 0.2023〜0.2159 秒）、Metal 節で観測されたような
明確な初回外れ値は見られなかった。外れ値の取捨選択は行わず全 5 試行で中央値を算出した。

### 再現性確認

run1・run2 の 2 セットとも内部 API ピーク値は完全に同一（201,326,592 バイト）であり、
`gemm_secs` の中央値も 0.210158 秒・0.204331 秒と近い値（差 約 3%）で安定していた。
`make peak-memory-bench BACKEND=cuda TRIALS=5`（`--out` なし）経路の中央値
（0.217090 秒）は run1 の Q1〜Q3（0.209847〜0.214925 秒）よりわずかに高いが、NVRTC
コンパイルコストの試行間変動（同経路の 2 trial 目が 0.829344 秒の外れ値を含む）を
踏まえると同オーダーの値であり、実行方式（ビルド済みバイナリ直接実行 対
`cargo run` 経由）による系統的な乖離ではないと判断する。

## CUDA/Metal 実機実測の再現手順（CPU・Metal・CUDA いずれも実測済み）

以下のコマンド・同一のバイナリで再現できる（`--backend cpu|cuda|metal` を切り替える）:

```bash
cargo build -p bench-harness --release --bins
cargo run -p bench-harness --release --bin peak_memory_bench -- \
  --backend cuda --trials 5 --out docs/perf/peak-memory/cuda-run1.json
cargo run -p bench-harness --release --bin peak_memory_bench -- \
  --backend cuda --trials 5 --out docs/perf/peak-memory/cuda-run2.json
```

CUDA 実機実測時は `crates/bench-harness/tests/peak_memory_smoke.rs` の
`cuda_peak_memory_matches_theoretical_minimum`（`#[ignore]` 分離済み）を
`cargo test -p bench-harness --release --test peak_memory_smoke -- --ignored --exact
cuda_peak_memory_matches_theoretical_minimum` で実行し、内部 API 値が理論最小
ワーキングセットと一致することも合わせて確認することを推奨する（本イシュー #392・
Metal 側の同種テスト `metal_peak_memory_matches_theoretical_minimum` はいずれも
実行・pass 済み）。

## スコープ外・申し送り（`.claude/rules/out-of-scope-tracking.md` 準拠）

- **係数の確定・再調整（384MiB 判定）**: 兄弟イシュー #179（TASK-14.2b）が本記録
  （特に「内部 API 値と外部参考値の乖離」節のデータ）を入力として実施する。確定結果は
  `docs/peak-memory-coefficient-decision.md` に記録済み（係数 2.0 を維持・超過なし）
- **計測手段の環境差文書化**（`docs/peak-memory-measurement-methods.md` 等）: #180
  （TASK-14.3）のスコープ
- **`BackendOps::gemm` 演算内部の一時確保への計測フック**: CPU バックエンドは
  `gemm_alloc_peak_bytes`（`bench_harness::alloc_tracker`。PR #370 codex-review 指摘 P1
  対応）として実装済み（本記録「内部 API 値と外部参考値の乖離」節参照）。CUDA・Metal は
  デバイス確保が Rust の `GlobalAlloc` を経由しないため引き続き対象外（`None`）であり、
  実 GPU デバイス確保量を測るバックエンド固有の代替手段（`cudarc`／`objc2-metal` 側の
  確保量フック等）は別イシューのスコープとする。自動運転下では Issue 起票せず、本記録と
  PR 本文への申し送りに留める（起票はユーザー承認必須。`.claude/rules/out-of-scope-tracking.md`）
- **`VmHWM` とのなお約 66.7〜66.8MiB の残差の内訳分解**: `gemm_alloc_peak_bytes` の計測境界
  （`BackendOps::gemm` 実行区間中の純増分のみ）の外側にある要因（プロセス起動時の初期ヒープ・
  rayon ワーカースレッドのスタック確保等）の内訳分解は本イシューでは行わない
- **CUDA（DGX Spark GB10）の実機実測**: イシュー #392 で実施済み（「CUDA 実機実測結果」
  節参照。Metal〈Apple Silicon〉はイシュー #385 で実施済み）
- **macOS の `getrusage`（`ru_maxrss`）相当の実装**: `libc` クレートの新規追加が必要になるため
  （許容依存 8 区分外。`.claude/rules/deps-policy.md`）、本イシューでは実装しない
  （`vm_hwm_bytes` は macOS では常に `None`）。必要であれば #180 または新規 Issue で
  ユーザー承認を得たうえで検討する
- **実 `MTLBuffer` 確保量の計測フック（`objc2-metal` 側）**: 「Metal 実機実測結果」節の
  計測境界の限界に記載のとおり、`MetalGemm` が直接確保するデバイスバッファは計測対象外の
  ままであり、実装は別イシューのスコープとする（#385 では実施しない）

## 生データ

- `docs/perf/peak-memory/cpu-run1.json`
- `docs/perf/peak-memory/cpu-run2.json`
- `docs/perf/peak-memory/metal-run1.json`（イシュー #385）
- `docs/perf/peak-memory/metal-run2.json`（イシュー #385）
- `docs/perf/peak-memory/cuda-run1.json`（イシュー #392）
- `docs/perf/peak-memory/cuda-run2.json`（イシュー #392）
