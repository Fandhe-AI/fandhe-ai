# CUDA `mse_loss_backward` ストリーム順序契約の確認・GB10 A/B（イシュー #1692）

親 #1582（MSE backward を Metal encode-only／CUDA ストリーム化）の分解
sub-issue。「CUDA 側 `mse_loss_backward` が本設計文書の非同期投入契約に
既に準拠していることを確認し、必要な場合のみ軽微な調整を行う」を目的とする。

## 1. コード読解で確定した事実（実測ではない）

- `crates/backend-cuda/src/ops.rs::CudaBackendOps::mse_loss_backward` は
  明示的な `stream.synchronize()` を持たず、
  `crates/backend-cuda/src/mse.rs::CudaMse::run_mse_backward_f32` へ
  委譲するのみ。
- `run_mse_backward_f32`（forward の `run_mse_loss_f32` も同様）は
  `clone_htod`（非同期 H2D）→ カーネル起動（非同期）→
  `memory::readback`（`clone_dtoh` + `stream.synchronize()` を 1 箇所へ
  集約）という構成で、`grep -rn synchronize
  crates/backend-cuda/src/{mse,kernels_mse}.rs` は 0 件（`docs/backend-
  cuda-async-execution-design.md` §2.2 棚卸し表に追記済み）。
- `docs/backend-cuda-async-execution-design.md` §2.3 が定める「ホスト
  `Tensor` を返す `BackendOps` 演算は戻り値の D2H が構造的な同期点で
  あり、readback 境界 1 箇所へ集約する」契約に**既に準拠している**。
- 唯一の潜在リスクだった I4（`CudaSlice::drop` が `has_async_alloc()`
  偽の環境ではホスト側 `stream.synchronize()` へフォールバックする）は、
  `docs/perf/lowlayer-diagnosis-2026-09-12.md` §2（出典
  `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/
  async_ordering_real_device.log`）が GB10 実機 `has_async_alloc() =
  true` を実測済みであることで解消済みと確認した。

**結論**: `crates/backend-cuda/src/{ops,mse}.rs` への機能変更は不要。
本 PR の変更は `docs/backend-cuda-async-execution-design.md`（§2.2 棚卸し
表・§16 実装記録の追記）・`crates/backend-cuda/src/mse.rs`（doc comment
追記のみ）・`crates/facade/tests/mse_backward_bench.rs`（新設マイクロ
ベンチ）・本ドキュメントに限る。

## 2. 事前登録判定規則（実装着手前に固定。issue コメント原文）

イシュー #1692 のコメント
<https://github.com/Fandhe-AI/fandhe-ai/issues/1692#issuecomment-5654656690>
に固定済み。要約:

- **対象**: `crates/facade/tests/mse_backward_bench.rs`（新設・`#[ignore]`）
  の `mse_backward_cases`。デバイス `FANDHE_BENCH_DEVICE=cuda`。
- **形状**: 主 `train_shape`（`[64, 10]`。訓練ハーネス実形状・640 要素・
  固定費支配）、副 `general_shape`（`SIZES = [16384, 65536, 1048576]`。
  一般形状スイープ）
- **計測区間**: forward は `.to_tensor()` で計測前に実体化、計測区間は
  `tape.backward(&loss)` のみ（`mse_loss` の VJP 呼び出しに限定）
- **手順**: 5 round・run 単位で before/after の起動順を反転・warmup 5 →
  計測 20
- **判定**: `median_s` の ratio(after/before) <= 1.00 を非後退の目安と
  する。ただし本 issue は `crates/*/src` への機能変更を伴わない見込み
  のため、**`crates/*/src` の差分がコメントのみ（または皆無）の場合、
  before/after は同一ソースの再測定であり ADOPT／REJECT の判定対象では
  なく、ノイズ床・再現性の記録として扱う**（この解釈自体を規則として
  明記し、事後に持ち出さない）
- **checksum**: `grad[...].fold_bits` が before/after で完全一致する
  こと（数値契約は不変のため必須）
- **正しさ（REQ-2）**: `crates/backend-cuda/tests/mse_parity.rs::
  mse_matches_cpu_across_shapes`（`#[ignore]`）を GB10 で実行し複合判定
  （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）pass を確認する
- **秘密情報**: 内部ホスト名を出力・報告に含めない（`docs/real-
  hardware-verification-env.md` の規約どおり `$CUDA_NODE` 変数越しに
  扱う）

## 3. 実測記録

本エージェント実行環境（Linux コンテナ／worktree）に DGX Spark GB10
実機への到達手段（`docs/real-hardware-verification-env.local.md`・
`CUDA_NODE` 環境変数・`~/.ssh/config` のいずれも確認できず）がないため、
**実測値は一切含まれていない**。本ドキュメントはスキャフォールド
（実行スクリプト・事前登録判定規則・記入欄）のみを提供し、実測は GB10
実機を持つセッションへ申し送る（`docs/perf/logs/train-resident-grad-
cuda-1560/` と同型の運用）。

実行スクリプト・記入欄は
`docs/perf/logs/cuda-mse-backward-1692/`（`README.md`・
`orchestrate.sh`・`env_info.txt`）を参照。

### 記入欄（GB10 実機実測後に埋める）

| 項目 | 値 |
|------|-----|
| 実測日 | (未実測) |
| before_sha | |
| after_sha | |
| `mse_parity.rs::mse_matches_cpu_across_shapes` | |
| `train_shape` median_s ratio（after/before・5 round 中央値） | |
| `general_shape[16384/65536/1048576]` median_s ratio | |
| `grad[...].fold_bits` 完全一致 | |
| 結論（ADOPT／REJECT／ノイズ床記録のみ） | |

## 4. スコープ外事項

- MSE VJP の `dPred` はホストへ readback された後、`Op::LinearResident`
  等の次段 VJP へ再度 H2D される可能性がある（backward チェーン全体で
  のホスト往復の残存）。これは `BackendOps::linear_forward_device` 系の
  デバイス常駐チェーン拡張（#1216／#1673 の設計）と関係する別範囲で
  あり、本 issue のスコープ外とする（`.claude/rules/out-of-scope-
  tracking.md` に従いユーザー承認なしに issue 化はしない）。

## 5. 出典

- `crates/backend-cuda/src/{ops,mse}.rs`
- `docs/backend-cuda-async-execution-design.md` §2.2・§2.3・§16
- `docs/perf/lowlayer-diagnosis-2026-09-12.md` §2
- `crates/facade/tests/elementwise_vjp_bench.rs`（イシュー #1583。本ベンチの参照実装）
- `crates/backend-cuda/tests/mse_parity.rs`
