# イシュー #1691 実測スキャフォールド（Metal `mse_loss_backward` encode-only 化）

## 2026-09-16 実測済み（M4 Max）

本ディレクトリには 2026-09-16 の M4 Max 実機実測の生成物を保存済み
（`orchestrate.sh` の (a)〜(d)(f) 生成物・`orchestrate_stdout.log`・
`progress_1691_excerpt.txt`・`ignored_store_parity_serial.log`・(e) 手動
実行分 `ab/`）。`orchestrate.sh` は (f) の `device_param_store_backend_
parity`（`grad_readout_contract_on_metal`。main 既存 FAIL・本 issue 対象外）
で `set -eu` 中断したため `uptime_after.txt` は未生成。verdict は
**REJECT**（(d) `general_shape 16384` ratio 1.0109 > 1.00・(f) FAIL 1 件）。
実値・判定・原因分析は `docs/perf/metal-mse-backward-encode-only-ab.md`
§3 と `env_info.txt` を正とする。以下はスキャフォールド作成時の記述。

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には Apple Silicon 実機への
到達手段がないため、**実測値は一切含まれていない**。本ディレクトリは
スキャフォールド（実行スクリプト・事前登録判定規則・記入欄）のみを
提供し、実測は M4 Max 実機を持つ Mac セッションへ申し送る
（`docs/perf/logs/metal-dinput-sync-1563/`・`docs/perf/logs/cuda-mse-
backward-1692/` と同型の運用）。

## 目的

イシュー #1691「`mse_loss_backward` encode-only 化（#1690。PR #1785）を
M4 Max で A/B し REQ-2 複合判定を確認する」の実測記録先。
`crates/backend-metal/src/mse.rs` の 3 箇所の `ctx.dispatch_sync`
（forward 2 段リダクション・backward 1 段）を `ctx.encode` +
`DispatchFailureCell` 登録へ切替えた変更（`docs/backend-metal-command-
batching-design.md` §7.5）を対象に、REQ-2 複合判定・非後退（5 run
中央値・checksum 完全一致・ratio<=1.00）を確認する。

## 比較対象 2 腕（事前登録・固定）

- **before 腕**: `84490ad1`（PR #1785 マージ直前 main。#1690 適用前）
- **after 腕**: 本イシューのブランチ（または `38b72b1f` 以降の main。
  `crates/*/src` は評価対象コミットと同一）

両腕とも workspace `version = "0.8.0"`。`crates/facade` への
`[patch.crates-io.fandhe-ai]` path patch（CLI `--config` 引数のみ。
deps-policy.md 第 9 区分）は framework-compare A/B（下記 (e)）のみで
使う（(a)〜(d)(f) は `cargo test` を各腕のツリーで直接実行する）。

## 事前登録判定規則（計測前に固定・結果を見て変更しない）

issue コメント
<https://github.com/Fandhe-AI/fandhe-ai/issues/1691#issuecomment-5656028993>
に固定済み（`docs/perf/metal-mse-backward-encode-only-ab.md` §2 に要約
転記済み）。要約:

- **(a) REQ-2 正しさ**: `cargo test -p fandhe-ai-backend-metal --release
  --test mse_parity -- --ignored --nocapture`
  （`mse_matches_cpu_across_shapes`。after 腕のみ）が pass。
- **(b) bit 同一**: `metal_reuse_step_grad_bit_dump` を両腕で実行し
  `^(step\[|final\.param\[)` 行の diff が 0 行。件数（
  `EXPECTED_BITDUMP_LINES=4462`）・ラベル集合一致を先に検証し、崩れたら
  UNDETERMINED（fail-closed）。
- **(c) カウンタ**: `mnist_scale_train_reuse_metal_batch_counters` が
  after で 11/7/7（hard assert）・before で 11/8/8 再現。
  `backward_dinput_phase` は両腕とも 5/3/3 をおおむね再現
  （record-only）。
- **(d) backward マイクロベンチ**: `mse_backward_bench.rs::
  mse_backward_cases`（`FANDHE_BENCH_DEVICE=metal`）を 5 round・
  起動順反転・別プロセス。`train_shape`（主）・`general_shape` 3 形状
  （副）で 5 run 中央値比 after/before ≤ 1.00 かつ `grad[...].fold_bits`
  全 run・両腕で完全一致。**事前見通し ≈1.00**（backward wait 不変。
  改善は狙わない・非後退確認）。
- **(e) framework-compare train Metal A/B（主判定・Tier 1）**:
  `size=64 / reuse` の `step_total` 5 run 中央値比 ≤ 1.00 かつ
  checksum 完全一致。fresh は対照（非判定）・`--phases` は診断
  （非判定）。
- **(f) 既存 `#[ignore]` 群非後退（after 腕）**: `mse_parity`・
  `mnist_scale_train_reuse_bench`・`command_batching`・
  `command_batching_bench`・`gemm_resident_parity`・
  `device_param_store_backend_parity`。
- **総合判定**: (a)(b)(f) pass かつ (c) 一致かつ (d)(e) すべて ≤ 1.00
  → ADOPT。(e) または (d) に > 1.00 → REJECT。実機到達不能・件数検証
  失敗 → undetermined。
- **秘密情報**: 内部ホスト名を出力・報告に含めない。

## 使い方（M4 Max 実機。ユーザー承認・別セッション）

1. `BEFORE_TREE` は `84490ad1` を checkout した独立ツリー、
   `AFTER_TREE` は本イシューのブランチ（またはマージ後の HEAD）を
   checkout した独立ツリーとして用意する。

```sh
git clone git@github.com:Fandhe-AI/fandhe-ai.git /path/to/mse-encode-1691-before
(cd /path/to/mse-encode-1691-before && git checkout 84490ad1)

git clone git@github.com:Fandhe-AI/fandhe-ai.git /path/to/mse-encode-1691-after
(cd /path/to/mse-encode-1691-after && git fetch origin test/1691-metal-mse-encode-only-ab && git checkout test/1691-metal-mse-encode-only-ab)
```

2. `./orchestrate.sh` を実行する（下記コマンド）。

```sh
cd docs/perf/logs/metal-mse-backward-1691
MSE_AB_BEFORE_REPO=/path/to/mse-encode-1691-before \
MSE_AB_AFTER_REPO=/path/to/mse-encode-1691-after \
  sh orchestrate.sh
# ドライラン（コマンド列の確認のみ・実行しない。実機不要）:
sh orchestrate.sh --dry-run
```

## 生成物

- `mse_parity_after.log`: (a) REQ-2 正しさ確認
- `bitdump_before.log`／`bitdump_after.log`／`bitdump_*_filtered.txt`／
  `bitdump_*_labels.txt`／`bitdump_label_diff.txt`／`bitdump_diff.txt`:
  (b) bit 同一確認一式
- `batch_counters_{before,after}.log`／`backward_phase_{before,after}.log`:
  (c) カウンタ実測
- `before_round{1..5}.log`／`after_round{1..5}.log`／`rounds.log`:
  (d) backward マイクロベンチ生ログ
- `aggregate.md`: `aggregate.py` の集計結果（5 run 中央値比・
  fold_bits 一致判定）
- `ignored_after_*.log`: (f) 既存 `#[ignore]` 群
- `ab/`: (e) `run_ab_mse_encode_metal.sh` の実行ログ（生成物本体は
  `$AFTER_REPO/scripts/bench/framework-compare/` 配下に残るため実測後
  手動でコピーする運用）
- `env_info_{before,after}.txt`／`uptime_{before,after}.txt`: 実行環境
  記入欄（内部ホスト名は含めない）

## 関連

- `docs/perf/metal-mse-backward-encode-only-ab.md`（記録文書本体）
- `docs/backend-metal-command-batching-design.md` §7.5（#1690 実装記録）
- `docs/perf/logs/metal-dinput-sync-1563/`（同型構成の踏襲元）
- `docs/perf/logs/cuda-mse-backward-1692/`（CUDA 側の対応 issue）
- 親 #1582 → 実装 #1690（PR #1785）→ 本イシュー #1691（測定）
