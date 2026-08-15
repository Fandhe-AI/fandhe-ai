# CPU GEMM packing バッファの gemm 呼び出し単位確保・再利用 計測記録（#556）

イシュー #556「perf(backend-cpu): packing バッファを gemm 呼び出し全体で 1 回確保して再利用する」の実測記録。

## 背景

`#554`（PR #644）で `pack_a`／`pack_b` は呼び出し元確保の panel サブスライスへ直接書き込む形へ改善済みだったが、panel バッファ自体（`b_panel`／`a_panel`）の確保は依然 `gemm_blis_region` の jc×pc×ic ループ内にあった（`vec![0.0f32; ...]` を反復ごとに実行）。本イシューはこの確保を [`PanelBuffers`]（`crates/backend-cpu/src/gemm_blis/mod.rs`）として `dispatch_region`（カーネル型確定直後）で 1 回に切り出し、`gemm_blis_region` はそのサブスライスを再借用するのみに変更した。

## ヒープ確保回数の削減

MC=128・KC=256・NC=512（`src/gemm_blis/mod.rs` の定数）・M=N=K=4096 の場合の概算（単一スレッド換算。jc ブロック数 = 4096/512 = 8、pc ブロック数 = 4096/256 = 16、ic ブロック数 = 4096/128 = 32）:

| バッファ | 変更前（呼び出しあたり） | 変更後（呼び出しあたり） |
|---|---|---|
| `b_panel` | jc×pc = 8×16 = 128 回確保 | 1 回（直列） |
| `a_panel` | jc×pc×ic = 8×16×32 = 4,096 回確保 | 1 回（直列） |

並列経路（`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`）では、`dispatch_region` が rayon の行パネルタスクごとに呼ばれるため、A/B 各 1 組を**タスクごとに 1 回**確保する（タスク数 = `num_threads` 相当。以前はタスクごとに上記回数のヒープ確保が発生していた）。

## 数値一致

`PanelBuffers` の再利用は各反復のサブスライスを `pack_a`／`pack_b` が完全に上書きする（端タイルは `dst.fill(0.0)` してから書く。`pack.rs`）ため、前反復の残留値には依存しない。累積順序・FMA 契約（`f32::mul_add` 連鎖）は変更していないため、`gemm_naive` との bit 完全一致契約（REQ-2）は維持される。既存 parity テスト（`tests/gemm_blis_parity.rs`・`tests/gemm_epilogue_parity.rs`）が変更なしで全て green（`cargo test -p backend-cpu` 実測）。加えて、並列タスク間のバッファ分離を検証する回帰テスト `gemm_blis_parallel_panel_buffers_are_task_local_across_repeated_runs`（固定 4 スレッドプール・複数行パネル／端タイルを跨ぐ形状 m=257,n=193,k=131 を 8 回反復）を新設し green を確認した。

## 計測環境

| 項目 | 値 |
|------|-----|
| CPU | QEMU Virtual CPU version 2.5+（`/proc/cpuinfo` 実測。物理ハードウェアではなく仮想化環境。複数エージェント並列実行中の共有ホスト） |
| 論理コア数 | 12（`nproc`） |
| OS | Linux 7.0.0-29-generic |
| rustc | 1.96.0 (ac68faa20 2026-05-25) |
| ビルド条件 | `RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p backend-cpu --release`（`cpu-gemm-epilogue-fusion.md` と同一条件） |
| 計測プロトコル | `bench-harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3 記録。`crates/backend-cpu/tests/gemm_blis_perf.rs`） |

## 再現コマンド

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p backend-cpu --release \
  -- --ignored gemm_blis_perf --nocapture
```

## 実測結果（変更前 origin/main `76d89fe` vs 変更後、同一環境・同一コマンド）

| 形状（M=N=K） | 変更前 median (s) | 変更前 Q1〜Q3 (s) | 変更後 median (s) | 変更後 Q1〜Q3 (s) | 差分 |
|---|---|---|---|---|---|
| 512 | 0.000774 | 0.000764〜0.000798 | 0.000762 | 0.000756〜0.000765 | -1.6%（改善） |
| 1024 | 0.004185 | 0.003829〜0.004617 | 0.004969 | 0.004790〜0.005010 | +18.7%（悪化） |
| 2048 | 0.029755 | 0.026128〜0.031552 | 0.028714 | 0.027640〜0.031417 | -3.5%（改善） |

M=512・M=2048 は僅かに改善、M=1024 は悪化しており、いずれも Q1〜Q3 幅（本環境のノイズ幅）と同程度かそれを超える変動に留まる。本環境（QEMU 仮想 CPU・複数エージェント並列実行中の共有ホスト）のノイズが単一計測（各形状 1 回の 20-run 中央値）の差を覆い隠すため、本結果からは有意な性能改善を主張しない。受け入れ基準は「確保回数削減の事実＋実測値の記録」であり、性能下限判定（REQ-8）は本イシューのスコープ外（計画 §5 ステップ 5）。ヒープ確保回数そのものの削減（上表「ヒープ確保回数の削減」節。呼び出しあたり最大 4,096 回 → 1〜num_threads 回）は計測ではなくコード変更から直接導かれる確定した事実である。

## スコープ外（本イシューで対応しない事項）

- **B packing のスレッド間重複計算の再構成**: `gemm_blis_parallel` は行パネル分割で各タスクが 5-loop 全体を独立実行する構成のため、同じ B 列ブロックを複数タスクが個別に packing し直す（matrixmultiply が採る「loop4/loop5 を直列化し B バッファ 1 本を全スレッドで共有」方式とは異なる）。本イシューはこの並列分割の再構成を範囲に含めない（計画 §1）。将来最適化候補として PR 本文に記載する。
- **より安定した計測環境（専有ハードウェア等）での再計測**: 本環境（共有 QEMU ホスト）はノイズが大きく、僅かな改善比較には不向き。実機検証は `docs/real-hardware-verification-env.md` の対象（CUDA/Metal）とは別軸のため、本イシューでは追加の Issue 化は行わない。
