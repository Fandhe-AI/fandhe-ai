# Metal `mse_loss_backward` encode-only 化の M4 Max A/B（イシュー #1691）

親 #1582（MSE backward の Metal encode-only／CUDA ストリーム化）の
測定系 sub-issue。実装系 #1690（PR #1785。squash `38b72b1f`・親
`84490ad1`）で導入した `crates/backend-metal/src/mse.rs` の 3 箇所の
`ctx.dispatch_sync`（forward `mse_partial_f32`／`mse_finalize_f32` 2
段リダクション・backward `run_mse_backward_f32` 1 段）の `ctx.encode` +
`DispatchFailureCell` 登録化を、M4 Max 実機で REQ-2 複合判定・
非後退（5 run 中央値・checksum 完全一致・ratio<=1.00）の事前登録規則で
確認する。

## 1. コード読解で確定した事実（実測ではない）

- `git diff --stat 84490ad1 38b72b1f -- crates/` は `mse.rs` のみを
  差分として含む（他はテスト doc comment・設計文書）。
- **同期回数の内訳**（`docs/backend-metal-command-batching-design.md`
  §7.5.1・§7.5.2）:
  - forward `run_mse_loss_f32`: 旧実装は 2 段リダクション
    （`mse_partial_f32`／`mse_finalize_f32`）がそれぞれ独立に
    `dispatch_sync`（commit + wait）していたが、#1690 で両方を
    `ctx.encode`（待たない）へ変更し最後に 1 回だけ
    `ctx.synchronize()` するようになった（**2 wait → 1 wait**）。
  - backward `run_mse_backward_f32`: 呼び出し元
    `crates/autodiff/src/grad.rs`（`Op::MseLoss` の VJP）が戻り値
    `dpred` に対して直後に `dense_vec(&dpred)`（ホスト即時アクセス）
    する契約のため、本関数は encode-only 化後も関数内で**必ず自ら**
    `ctx.synchronize()` する（**1 wait → 1 wait。不変**）。
  - したがって 1 step 全体のカウンタ見積り（`mnist_scale_train_reuse_
    metal_batch_counters`）は #1566 適用後の 11/8/8 から #1690 適用後
    **11/7/7** へ変わる見積り（forward の −1 command_buffer・−1
    wait）。backward 限定窓（`mnist_scale_train_reuse_metal_backward_
    dinput_phase`）の仮説（encode_delta=5／command_buffer_delta=3／
    wait_delta=3）は**変わらない**。
- **他 issue との二重主張回避**（§7.5.3）: `Op::LinearResident` の
  d_input（#1561/#1562/#1563）・bias 勾配（#1564/#1566）側の同期境界
  は本 issue では変更しておらず、それらの待ちについて本 issue は何も
  主張しない。
- 動機となった数値（backward の loss 項が M4 Max 共有負荷下 67〜124
  µs。`docs/perf/lowlayer-diagnosis-2026-09-12.md` §4）は
  `FANDHE_DIAG_BACKWARD=1` の**診断専用計装パッチ**由来で本番コードで
  は取れない値であり、本 issue の判定には用いない（動機の記録のみ）。

## 2. 事前登録判定規則

issue コメント
<https://github.com/Fandhe-AI/fandhe-ai/issues/1691#issuecomment-5656028993>
に固定済み。要約:

- **対象**: `crates/facade/tests/mse_backward_bench.rs::
  mse_backward_cases`（`FANDHE_BENCH_DEVICE=metal`）・framework-compare
  train Metal（`size=64 / reuse`）・カウンタテスト・parity テスト・
  既存 `#[ignore]` 群。
- **比較腕**: before=`84490ad1`・after=`38b72b1f` 以降の main。
- **計測層**（すべて record_only・専有ゲートなし）:
  - (a) REQ-2 正しさ（`mse_parity.rs::mse_matches_cpu_across_shapes`）
  - (b) bit 同一（`metal_reuse_step_grad_bit_dump`。件数・ラベル集合
    検証込み fail-closed）
  - (c) カウンタ（`mnist_scale_train_reuse_metal_batch_counters` after
    11/7/7 hard assert・before 11/8/8 再現。`backward_dinput_phase` は
    両腕とも 5/3/3 をおおむね再現・record-only）
  - (d) backward マイクロベンチ 5 round・起動順反転・非後退確認
    （事前見通し ≈1.00。改善は狙わない）
  - (e) framework-compare train Metal A/B（**主判定・Tier 1**。reuse
    `step_total` 5 run 中央値比 ≤ 1.00・checksum 完全一致。fresh は
    対照・`--phases` は診断）
  - (f) 既存 `#[ignore]` 群非後退（after 腕）
- **総合判定**: (a)(b)(f) pass かつ (c) 一致かつ (d)(e) すべて ≤ 1.00
  → ADOPT。(e) または (d) に > 1.00 → REJECT（原因帰属は判定と分けて
  記録）。実機到達不能・件数検証失敗 → undetermined。

## 3. 実測記録

本エージェント実行環境（Linux コンテナ／worktree）に Apple Silicon
実機への到達手段がないため、**実測値は一切含まれていない**。本文書は
スキャフォールド（実行スクリプト・事前登録判定規則・記入欄）のみを
提供し、実測は Mac セッションへ申し送る（`docs/perf/logs/train-
resident-grad-cuda-1560/`・`docs/perf/logs/cuda-mse-backward-1692/`
と同型の運用）。

実行スクリプト・記入欄は
`docs/perf/logs/metal-mse-backward-1691/`（`README.md`・
`orchestrate.sh`・`aggregate.py`・`env_info.txt`）を参照。

### 記入欄（M4 Max 実機実測後に埋める）

| 項目 | 値 |
|------|-----|
| 実測日 | (未実測) |
| before_sha | |
| after_sha | |
| (a) `mse_parity.rs::mse_matches_cpu_across_shapes` | |
| (b) bit dump 件数・ラベル集合・diff | |
| (c) `batch_counters` before/after (encode/cb/wait) | |
| (c) `backward_dinput_phase` before/after (encode/cb/wait) | |
| (d) `train_shape/640` median_s ratio (5 run) | |
| (d) `general_shape[16384/65536/1048576]` median_s ratio | |
| (d) `grad[...].fold_bits` 完全一致 | |
| (e) `run_ab_mse_encode_metal.sh` reuse step_total 5 run 中央値比 | |
| (e) checksum 完全一致 | |
| (f) 既存 `#[ignore]` 群非後退 | |
| 結論（ADOPT／REJECT／undetermined） | |

## 4. スコープ外事項

- CUDA 側は #1692 で分解済み・別スコープ（コード変更なし）。
- `dpred` のホスト往復残存は `BackendOps::linear_forward_device` 系の
  デバイス常駐チェーン拡張（#1216／#1673 の設計）と関係する別範囲で
  あり、本 issue のスコープ外とする
  （`.claude/rules/out-of-scope-tracking.md` に従いユーザー承認なしに
  issue 化はしない）。
- tolerance／baseline・ガードレール閾値の変更は対象外（不変）。

## 5. 出典

- `docs/backend-metal-command-batching-design.md` §7.5（#1690 実装記録・
  §7.5.4 実測記入欄）
- `docs/perf/cuda-mse-backward-stream-contract.md`（#1692・CUDA 側の
  同型記録）
- `docs/perf/lowlayer-diagnosis-2026-09-12.md` §4（動機となった診断
  専用計装の実測値。本判定には用いない）
- 親 #1582 → 実装 #1690（PR #1785）→ 本 issue #1691（測定）
