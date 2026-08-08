# CPU GEMM epilogue 融合（bias・activation）計測記録（#203・TASK-12.1f）

イシュー #203「feat(fusion): TASK-12.1f GEMM epilogue 融合（bias・activation）の実装」の実測記録。
受け入れ条件「Linear+bias+ReLU 相当で非融合比の性能向上を実測（5 回中央値）・数値一致維持」に対応する。

## 計測環境

| 項目 | 値 |
|------|-----|
| CPU | QEMU Virtual CPU version 2.5+（`/proc/cpuinfo` 実測。物理ハードウェアではなく仮想化環境） |
| 論理コア数 | 12（`nproc`） |
| OS | Linux 7.0.0-28-generic |
| rustc | 1.96.0 (ac68faa20 2026-05-25) |
| ビルド条件 | `RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p backend-cpu --release`（AVX2+FMA を実行時 ISA ディスパッチが選ぶよう明示。`gemm_blis` の dispatch は実行時検出のためこのフラグなしでも動作するが、コンパイル時の AVX2 コード生成を確実にするため付与） |
| 計測プロトコル | `bench-harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3 記録。TASK-8.1 準拠。coding-rust.md の「5 回計測の中央値」下限を包含） |
| 計測バイナリ | `crates/backend-cpu/tests/gemm_epilogue_perf.rs`（`#[ignore]` 分離） |
| 比較対象（非融合 baseline） | `tensor_core::BackendOps::gemm_bias_act` の**デフォルト実装そのもの**（`CpuBackendOps` 経由で `ops.gemm(...)` → `ops.add(...)` → `ops.relu(...)` を明示的に呼ぶ。両ステップとも `elementwise` の実カーネル〈`PARALLEL_THRESHOLD=1<<15` 要素以上で rayon 並列化。本ハーネスの全形状はこの閾値を超える〉・`Tensor` 出力割当を経由する。利用者が現在 `gemm_bias_act` から実際に得る経路と完全に同一コードパス） |
| 比較対象（融合） | `CpuBackendOps::gemm_bias_act`（本イシューで追加。オーバーライド経由で `gemm_blis_bias_act_parallel` を呼ぶ。行パネル並列の GEMM 完了直後・同一 `rayon` タスク内で bias 加算・activation を適用。中間 `Tensor` 割当は出力 1 個のみ） |

**計測方法についての注記**: 初回実装時は `gemm_blis_parallel` を直接呼び逐次 `for` ループで bias 加算・`relu` を模した baseline を用いていたが、実際の非融合経路（`elementwise::add`／`elementwise::relu`）は本ハーネスの全形状（最小 262144 要素）で rayon 並列化されるため、逐次 baseline は実際に利用者が得る経路より大幅に遅く改善比を過大評価していた（レビュー指摘・修正済み）。本表は `BackendOps` トレイト経由（両側とも実カーネル・実 `Tensor` 割当）に統一した計測結果である。

## 再現コマンド

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p backend-cpu --release \
  -- --ignored gemm_epilogue_perf --nocapture
```

## 実測結果

Linear+bias+ReLU 相当形状（M=バッチ、K=in_features、N=out_features）と正方形状の計 5 形状で計測した。

| 形状（M, N, K） | 非融合 median (s) | 非融合 Q1〜Q3 (s) | 融合 median (s) | 融合 Q1〜Q3 (s) | 改善比 |
|---|---|---|---|---|---|
| 256, 1024, 1024 | 0.003873 | 0.003661〜0.003980 | 0.001969 | 0.001936〜0.002067 | **1.967x** |
| 1024, 1024, 1024（1 回目） | 0.013545 | 0.013334〜0.014450 | 0.005347 | 0.005251〜0.005396 | **2.533x** |
| 512, 512, 512 | 0.001869 | 0.001833〜0.002058 | 0.000814 | 0.000798〜0.000819 | **2.296x** |
| 1024, 1024, 1024（2 回目） | 0.013448 | 0.013105〜0.013655 | 0.005263 | 0.005160〜0.005295 | **2.555x** |
| 2048, 2048, 2048 | 0.069515 | 0.065964〜0.071567 | 0.047581 | 0.040232〜0.050477 | **1.461x** |

全形状で融合カーネルが非融合（デフォルト実装そのもの）を上回り、改善比は 1.46〜2.56 倍（本環境実測）。
CUTLASS 系実測の動機（平均 1.38〜1.45 倍。イシュー #203 本文）以上の改善が全形状で確認でき、受け入れ条件「非融合比の性能向上を実測（5 回中央値）」を満たす。

CUTLASS 系実測より本環境の改善比が大きく出ている理由は、非融合側が「rayon 並列 GEMM → rayon 並列 add → rayon 並列 relu」の 3 回の並列リージョン起動（タスク分割・スレッド同期コストを 3 回払う）と中間 `Tensor` 2 個の割当・C の再読み出し 2 回を伴うのに対し、融合側は 1 回の並列リージョン（GEMM の行パネル並列）内で epilogue まで完結させるためと考えられる。M=N=K=2048（改善比 1.46 倍、他形状より相対的に小さい）は Q1〜Q3 幅がやや広く（融合側 0.040232〜0.050477s）、本環境（QEMU Virtual CPU・複数エージェント並列実行中の共有ホスト）のノイズの影響を受けている可能性がある。

## 数値一致

融合版（`gemm_blis_bias_act_parallel`）と非融合合成（逐次参照実装）は、epilogue（bias 加算・activation）が要素ごとに独立な演算で演算順序に依存しないため **bit 完全一致**する。MR/NR/MC/KC/NC 境界を跨ぐ形状グリッド（`SHAPE_GRID_M/N/K`）・REQ-2 統一複合判定（相対誤差 1e-3 未満 or 絶対誤差 1e-5 未満）の両方で検証済み（`crates/backend-cpu/tests/gemm_epilogue_parity.rs`）。GEMM 本体の FMA 契約（`f32::mul_add`）・累積順序は `gemm_blis_parallel` から変更していない。

## スコープ外（本イシューで対応しない事項）

- **GPU（CUDA NVRTC・Metal MSL）のカーネル内 epilogue 融合**: `BackendOps::gemm_bias_act` はデフォルト実装（非融合合成）にフォールバックする。CUDA／Metal は本イシュー時点で GEMM カーネルのみ実装済みで elementwise（`add`/`relu`）が未実装（`Unsupported` を返す）ため、`bias`/`act` 指定時は透過的に `Unsupported` となる。GPU カーネル内 epilogue の実装自体は実機（DGX Spark GB10・Metal 実機）での検証が前提であり、`out-of-scope-tracking.md` に従いユーザー承認を得たうえで別 Issue として追跡する。
- **融合グラフ機構（TASK-12.1a〜e・#161〜#165）からの自動検出接続**: 本イシューは GEMM カーネル入口レベルの epilogue 融合 API として自己完結させ、演算グラフ上の `matmul→add(bias)→relu` パターン自動検出からの接続は #164（バックエンド抽象層統合）以降の統合点に委ねる。
- **matmul 複合ワークロードでの融合効果の性能目標への組み込み**: 本計測は epilogue 融合単体の相対改善比の実測であり、TASK-12.2（性能目標達成の判定）の前提とはしない。
