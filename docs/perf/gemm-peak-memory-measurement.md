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

## 状態: CPU 実測済み・CUDA/Metal 実機未実施

本実装セッションは Linux x86_64（QEMU/KVM 仮想化環境、NVIDIA RTX 3060 passthrough）
worktree で行っており、`libnvrtc`（CUDA toolkit）が導入されていない
（`ldconfig -p | grep nvrtc` で未検出）。Metal（Apple Silicon）実機はこのセッションから
利用できない。

既存の先例（`docs/perf/startup-cost-measurement.md`〈#171〉・
`docs/perf/dispatch-boundary-measurement.md`〈#69〉・
`docs/perf/cuda-tensor-core-measurement.md`〈#64〉）と同じ運用を採る:
**CPU は本セッションで実測を完了**し、**CUDA（DGX Spark GB10）・Metal（Apple Silicon）は
再現可能な計測手順と結果転記テンプレートのみを整備**して「実機未実施」と明記する。
CUDA については、動的ロード契約により本環境でも実行時に panic せず型付きエラーへ
fail-closed に倒れることを実地確認した（「CUDA 実行時エラーの実地確認」節）。

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

## CUDA/Metal 実機実測の再現手順（未実施。転記テンプレート）

DGX Spark GB10（CUDA 実機）・Apple Silicon（Metal 実機）で実測する場合は、以下と同一の
コマンド・同一のバイナリで再現できる:

```bash
cargo build -p bench-harness --release --bins
cargo run -p bench-harness --release --bin peak_memory_bench -- \
  --backend cuda --trials 5 --out docs/perf/peak-memory/cuda-run1.json
cargo run -p bench-harness --release --bin peak_memory_bench -- \
  --backend cuda --trials 5 --out docs/perf/peak-memory/cuda-run2.json
# Metal は --backend metal に置き換え、cuda-run*.json → metal-run*.json
```

転記テンプレート（実機実測後、上記 CPU 実測結果と同型の表を追記する）:

| セット | peak_bytes（中央値 / Q1 / Q3） | 理論最小ワーキングセット | 対理論比 | gemm_secs（中央値 / Q1 / Q3, 秒） | vm_hwm_bytes（参考値。Metal は取得不能につき「-」） | gemm_alloc_peak_bytes（参考値。CUDA/Metal は `GlobalAlloc` 非経由につき常に「-」） |
|--------|-------------------------------|------------------------|---------|-----------------------------------|------------------------------------------------------|---------------------------------------------------------------------------------------|
| run1（CUDA/Metal） | (未実施) | 201,326,592 | (未実施) | (未実施) | (未実施) | - |
| run2（CUDA/Metal） | (未実施) | 201,326,592 | (未実施) | (未実施) | (未実施) | - |

実機実測時は `crates/bench-harness/tests/peak_memory_smoke.rs` の
`cuda_peak_memory_matches_theoretical_minimum`／`metal_peak_memory_matches_theoretical_minimum`
（`#[ignore]` 分離済み）を `cargo test -p bench-harness -- --ignored` で実行し、内部 API 値が
理論最小ワーキングセットと一致することも合わせて確認することを推奨する。

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
- **CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機実測**: 転記テンプレート運用。
  実機実施時は同一バイナリ・同一コマンドで再現可能
- **macOS の `getrusage`（`ru_maxrss`）相当の実装**: `libc` クレートの新規追加が必要になるため
  （許容依存 8 区分外。`.claude/rules/deps-policy.md`）、本イシューでは実装しない
  （`vm_hwm_bytes` は macOS では常に `None`）。必要であれば #180 または新規 Issue で
  ユーザー承認を得たうえで検討する

## 生データ

- `docs/perf/peak-memory/cpu-run1.json`
- `docs/perf/peak-memory/cpu-run2.json`
