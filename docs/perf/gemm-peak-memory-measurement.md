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
3. ワーキングセット保持中に `BackendOps::gemm(A, B)` を実行（所要秒数を記録）
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

## CPU 実測結果

`peak_memory_bench --backend cpu --trials 5` を `--out` 付きで 2 セット実行した。生 JSON は
`docs/perf/peak-memory/cpu-run1.json`・`cpu-run2.json`（全試行の `samples` 込み）。

| セット | peak_bytes（中央値 / Q1 / Q3） | 理論最小ワーキングセット | 対理論比 | gemm_secs（中央値 / Q1 / Q3, 秒） | vm_hwm_bytes（参考値） |
|--------|-------------------------------|------------------------|---------|-----------------------------------|------------------------|
| run1 | 201,326,592 / 201,326,592 / 201,326,592 | 201,326,592 | 1.000 | 0.2236 / 0.2177 / 0.2277 | 346,431,488（全 trial 同値） |
| run2 | 201,326,592 / 201,326,592 / 201,326,592 | 201,326,592 | 1.000 | 0.2153 / 0.2088 / 0.2189 | 346,402,816（全 trial 同値） |

**内部計測 API のピーク値は 5 試行すべて・2 セットとも決定的に 201,326,592 バイト（192MiB）
ちょうど**であり、理論最小ワーキングセット（A+B+C 各 64MiB の合計）と完全一致した（対理論比
1.000）。これは `MemoryOps` 経由の確保が A・B・C の 3 バッファのみであり、他の中間確保を
一切含まないという実装契約どおりの結果である。drop 後の `allocated_after_drop_bytes` は
全 10 trial（2 セット × 5 試行）で 0 を記録し、リークは検出されなかった。

### 内部 API 値と外部参考値（`VmHWM`）の乖離

`VmHWM`（プロセス全体のピーク常駐セットサイズ）は run1/run2 とも約 330.4MiB
（346,431,488 / 346,402,816 バイト）であり、内部 API 値（192MiB）に対し **約 1.72 倍**
大きい。両セットの `VmHWM` は全 trial で同一値（プロセス全体の生涯ピークのため、5 trial 目まで
に到達した最大値が記録される）。

差分（約 138MiB）の主な要因は、計測境界の外側にある GEMM 演算内部の一時確保と考えられる:

- `CpuBackendOps::gemm`（`crates/backend-cpu/src/ops.rs`）の出力バッファ
  `let mut out = vec![0.0f32; m * n];`（4096×4096 f32 = 64MiB）
- `gemm_blis_parallel`（`crates/backend-cpu/src/gemm_blis.rs`）内部の BLIS 型ブロッキング・
  パッキングバッファ（rayon 並列ワーカー数分。本イシューでは内訳の実測までは行わない）
- `Tensor::contiguous()` による A・B の実体化コピー（`gemm` 呼び出し内で `a.contiguous()`・
  `b.contiguous()` を呼ぶため、`upload` 前の host 側 Tensor とは別に、GEMM 実行時点でさらに
  A・B 相当のコピーが一時的に存在しうる）

出力バッファ（64MiB）だけでも 192MiB + 64MiB = 256MiB となり、残り約 74MiB が BLIS
パッキングバッファ・contiguous コピー等の内訳と推定されるが、本イシューでは個別の内訳計測
までは行わない（`VmHWM` はプロセス全体の参考値であり、GEMM 演算専用の内訳分解はできない）。

**判断材料としての申し送り**: 内部 API 値（192MiB・対理論比 1.000）に対し `VmHWM` 参考値
（約 330MiB・対理論比 約 1.72）には無視できない乖離がある。REQ-14 の係数上限
（384MiB 以内 = 対理論比 2.0 以内）と比較すると、内部 API 値だけを見れば余裕があるが、
プロセス全体のピーク常駐（GEMM 内部確保を含む実態）は対理論比 1.72 まで到達しており、
係数の余裕は内部 API 値が示すほど大きくない可能性がある。`BackendOps::gemm` 演算内部の
一時確保への計測フック組み込み要否は、この乖離データを踏まえて #179（TASK-14.2b。係数確定・
再調整）で判断することを推奨する。

### 再現性確認

run1・run2 の 2 セットとも内部 API ピーク値は完全に同一（201,326,592 バイト）であり、
`gemm_secs` の中央値も 0.2236 秒・0.2153 秒と近い値（差 約 4%）で安定していた。
`vm_hwm_bytes` も両セットでほぼ同一（差 約 28KiB、プロセス起動オーバーヘッドの揺らぎ程度）。

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

| セット | peak_bytes（中央値 / Q1 / Q3） | 理論最小ワーキングセット | 対理論比 | gemm_secs（中央値 / Q1 / Q3, 秒） | vm_hwm_bytes（参考値。Metal は取得不能につき「-」） |
|--------|-------------------------------|------------------------|---------|-----------------------------------|------------------------------------------------------|
| run1（CUDA/Metal） | (未実施) | 201,326,592 | (未実施) | (未実施) | (未実施) |
| run2（CUDA/Metal） | (未実施) | 201,326,592 | (未実施) | (未実施) | (未実施) |

実機実測時は `crates/bench-harness/tests/peak_memory_smoke.rs` の
`cuda_peak_memory_matches_theoretical_minimum`／`metal_peak_memory_matches_theoretical_minimum`
（`#[ignore]` 分離済み）を `cargo test -p bench-harness -- --ignored` で実行し、内部 API 値が
理論最小ワーキングセットと一致することも合わせて確認することを推奨する。

## スコープ外・申し送り（`.claude/rules/out-of-scope-tracking.md` 準拠）

- **係数の確定・再調整（384MiB 判定）**: 兄弟イシュー #179（TASK-14.2b）が本記録
  （特に「内部 API 値と外部参考値の乖離」節のデータ）を入力として実施する
- **計測手段の環境差文書化**（`docs/peak-memory-measurement-methods.md` 等）: #180
  （TASK-14.3）のスコープ
- **`BackendOps::gemm` 演算内部の一時確保への計測フック組み込み要否**: 本実測の内部 API 値
  （対理論比 1.000）と `VmHWM` 参考値（対理論比 約 1.72）の乖離データを判断材料として本記録に
  残した。自動運転下では Issue 起票せず、本記録と PR 本文への申し送りに留める
  （起票はユーザー承認必須。`.claude/rules/out-of-scope-tracking.md`）
- **CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機実測**: 転記テンプレート運用。
  実機実施時は同一バイナリ・同一コマンドで再現可能
- **macOS の `getrusage`（`ru_maxrss`）相当の実装**: `libc` クレートの新規追加が必要になるため
  （許容依存 8 区分外。`.claude/rules/deps-policy.md`）、本イシューでは実装しない
  （`vm_hwm_bytes` は macOS では常に `None`）。必要であれば #180 または新規 Issue で
  ユーザー承認を得たうえで検討する

## 生データ

- `docs/perf/peak-memory/cpu-run1.json`
- `docs/perf/peak-memory/cpu-run2.json`
