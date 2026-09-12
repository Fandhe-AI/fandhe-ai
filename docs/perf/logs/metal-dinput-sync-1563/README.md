# `d_input` 経路の同期境界回収の実装（イシュー #1563）実測ログ

`Op::LinearResident` の VJP で encode-only の d_weight
（`fill_resident_weight_grad`）を、同期点を持つ d_input
（`gemm_resident_lhs`）より前へ移す変更（層内合流。
`docs/backend-metal-command-batching-design.md` §7.4）の
bit 同一確認・`#[ignore]` 群非後退確認・カウンタ実測・A/B を行う
ログ置き場。

**現状（本 PR 時点）**: `orchestrate.sh`・本 README・
`env_info.txt.example` は用意済みだが、実測ログ（`bitdump_before.log`
等の生成物）自体は未生成。本 PR を書いた実行環境に Apple Silicon 実機へ
のアクセス経路がないため、下記の実測は未実施のまま記入欄として残す
（`docs/backend-metal-command-batching-design.md` §7.4 の記入欄と対応）。
Mac セッションで `orchestrate.sh` を実行し、生成物をこのディレクトリへ
収める。

## 事前準備（Mac セッション側）

before 腕（origin/main。#1563 適用前の HEAD）・after 腕（本ブランチ）を
それぞれ独立した worktree／clone に展開する。例:

```sh
git clone git@github.com:Fandhe-AI/fandhe-ai.git /path/to/dinput-sync-1563-before
(cd /path/to/dinput-sync-1563-before && git checkout origin/main)

git clone git@github.com:Fandhe-AI/fandhe-ai.git /path/to/dinput-sync-1563-after
(cd /path/to/dinput-sync-1563-after && git fetch origin perf/1563-metal-dinput-sync-reorder && git checkout perf/1563-metal-dinput-sync-reorder)
```

## 実行手順

```sh
cd docs/perf/logs/metal-dinput-sync-1563
DINPUT_SYNC_BEFORE_REPO=/path/to/dinput-sync-1563-before \
DINPUT_SYNC_AFTER_REPO=/path/to/dinput-sync-1563-after \
  sh orchestrate.sh
# ドライラン（コマンド列の確認のみ・実行しない）:
sh orchestrate.sh --dry-run
```

## 生成物

- `bitdump_before.log`／`bitdump_after.log`: 両腕の
  `metal_reuse_step_grad_bit_dump`（`--ignored --nocapture`）全出力
- `bitdump_before_filtered.txt`／`bitdump_after_filtered.txt`：
  `^(step\[|final\.param\[)` 行のみを抽出したもの
- `bitdump_before_labels.txt`／`bitdump_after_labels.txt`：上記 2
  ファイルの各行から ` = ` 右辺（値）を除いた項目ラベルのみを sort した
  もの（件数・項目集合検証用。PR #1665 codex-review 指摘）
- `bitdump_label_diff.txt`: 上記 2 ラベルファイルの `diff -u`。空（0
  行）であることを期待（両腕の抽出項目集合が一致することの確認。
  非空の場合は grep 抽出漏れ・出力形式変化等を疑う）
- `bitdump_diff.txt`: `bitdump_before_filtered.txt`／
  `bitdump_after_filtered.txt` の `diff -u`。**空である（0 行）ことを
  期待**（bit 完全一致。`EXPECTED_BITDUMP_LINES=4462` 行前後）。ただし
  この diff だけでは「両腕とも grep 抽出が 0 件（または同じ部分集合）」
  で偶然 0 行になるケースを検出できないため、`orchestrate.sh` は比較前に
  (i) 両腕の抽出件数が 0 より大きいこと、(ii) 両腕とも
  `EXPECTED_BITDUMP_LINES` と一致すること、(iii) `bitdump_label_diff.txt`
  が空であることを独立に検証し、いずれかが崩れていれば標準出力へ
  「bit dump: UNDETERMINED」と記録する（「一致」と誤報告しない。
  fail-closed）
- `ignored_after_mnist.log`／`ignored_after_gemm_parity.log`／
  `ignored_after_store_parity.log`／`ignored_after_command_batching.log`:
  after 腕の既存 `#[ignore]` テスト群（非後退確認）
- `batch_counters_after.log`／`batch_counters_before.log`:
  `mnist_scale_train_reuse_metal_batch_counters`（hard assert）の両腕
  実行結果。before は 11/9/9、after は 11/8/8 を期待
- `backward_phase_after.log`／`backward_phase_before.log`:
  `mnist_scale_train_reuse_metal_backward_dinput_phase`（record-only）の
  両腕実行結果。before は 5/4/3、after は 5/3/3 を期待仮説とする
- `issue_1562_backfill.log`: before 腕で #1562 の `orchestrate.sh` を
  実行したログ（#1562 §7.3.4 の記入欄を埋めるための副産物。追加コスト
  なし）
- `ab/run_ab_1563.log`: `scripts/bench/framework-compare/
  run_ab_dinput_sync_metal.sh` の実行ログ。A/B 本体の生成物
  （`compare-train-1563.md`・`results/raw/*` 等）は
  `$AFTER_REPO/scripts/bench/framework-compare/` 配下に残るため、実測後
  それらを手動でこのディレクトリの `ab/` へコピーする
- `env_info_before.txt`／`env_info_after.txt`: `uname -srm`・`sw_vers`・
  `rustc -V`・`sysctl machdep.cpu.brand_string`・各腕の
  `git rev-parse HEAD`（内部ホスト名は含めない）
- `uptime_before.txt`／`uptime_after.txt`: 実行前後の 1 回計測
  （record_only 運用。専有ゲートは課さない）

## 事前登録判定規則（record only・non-gating。計測後に緩和しない）

1. bit 同一: 両腕の抽出件数が `EXPECTED_BITDUMP_LINES`（4462）と一致し
   `bitdump_label_diff.txt` が空（項目集合一致）であることをまず確認し
   たうえで、`bitdump_diff.txt` が空（差分 0 行）であること。件数・項目
   集合検証に失敗した場合は「一致」と判定せず「判定不能
   （UNDETERMINED）」として記録する（grep 抽出漏れ等による空ファイル
   同士の見かけ上の一致を bit-identical と誤報告しないため）。
   `bitdump_diff.txt` に実差分がある場合は「回収しない」判断の対象
   （d_input・d_weight は独立計算という前提が崩れている可能性）とし、
   コード側の assert 値（11/8/8・5/3/3）を実測へ合わせるのではなく原因
   調査を優先する（fail-closed）
2. `#[ignore]` 群非後退: `ignored_after_*.log` の各テストが pass する
   こと（既知の pre-existing FAIL がある場合はそのテスト個別実行で
   ベースラインを確認したうえで区別する）
3. カウンタ:
   - before 腕: `batch_counters_before.log` が 11/9/9 を再現すること
     （#1555 時点の値。再現しない場合は同一セッション内の環境要因を
     疑う）
   - after 腕: `batch_counters_after.log` が 11/8/8 であること（hard
     assert のためテスト自体が pass/fail で判定される）
   - before 腕: `backward_phase_before.log` の 5 trial が 5/4/3 を
     おおむね再現すること（#1562 時点の仮説。record-only）
   - after 腕: `backward_phase_after.log` の 5 trial が 5/3/3 を
     おおむね再現すること（#1563 の更新仮説。record-only）
4. A/B（Tier 1・必須）: `size=64 / reuse` の `step_total` 5 run 中央値比
   after/before ≤ 1.00 かつ checksum 完全一致（`run_ab_dinput_sync_metal.sh`
   の `--require-checksum-exact` が機械判定する）
5. 対照（非判定）: fresh セル（`Op::LinearAct` 経路・本変更非到達）
6. 診断（非判定）: `--phases` の `backward`／`device_update`／`step_total`
   内訳
7. 専有ゲートなし・record_only（イシュー #1519 系のユーザー指示に従う）。
   uptime／pmset を記録する
8. 未達（>1.00・符号不一致等）でも undetermined／REJECT を正式結果として
   記録し、結線を差し戻す判断は記録のうえ別途行う（本イシューはコード
   変更を伴う実装イシューのため、A/B が未達だった場合の対応方針は PR
   本文・`docs/backend-metal-command-batching-design.md` §7.4 へ記録する）

## 関連

- `docs/backend-metal-command-batching-design.md` §7.3（#1562 の調査結果）・
  §7.4（本イシューの実装・実測記入欄）
- `docs/perf/logs/metal-dinput-sync-1562/`（前段の測定タスクの記入欄。
  本オーケストレーションのステップ 4 が埋める）
- `docs/autodiff-nograd-leaf-dinput-skip-decision.md`（回収案 (c)。本
  イシューでは実装しない）
- 親イシュー #1557 → #1561 → #1562（測定・完了）→ 本イシュー #1563
  （実装・前後比較）
