# Metal split-K 公開入口の数値契約ゲート解除（イシュー #1513）実測ログ

`SPLIT_K_NUMERIC_CONTRACT_APPROVED` を `false` → `true` へ切り替え、自動判定
入口 `MetalGemm::dispatch_split_k_strided_prepared` が承認済み形状で実際に
split-K 経路を実行するようになったこと（承認記録:
`docs/backend-metal-splitk-parity-judgment-decision.md` §7・2026-09-10
ユーザー承認）を Apple M4 Max 実機で確認するログ置き場。

**現状（2026-09-16 実測済み・#1904）**: `run_auto_entry.sh` を Apple M4 Max
実機で実行し、`auto_entry.log`・`bit_match.log`・`parity.log`・`env_info.txt`・
`uptime_before.txt`／`uptime_after.txt`／`uptime_during.log` をこのディレクトリへ
収めた。判定結果は下記「事前登録判定規則」直後の「実測結果（#1904）」節を参照:
(1) は `ok`、(2) は `(64,64,63)` fixture が `Err(MetalError::
StridedTiledIneligible)` を返し `FAILED`（`Ok(Classic{NotEligible})` を期待する
旧 fixture との食い違い。契約確定・是正は #1899）、(3) は `ok`。是正後（#1899）の
fixture 2 分割・新設テストに基づく再実測は**未実施のまま**このディレクトリへ
申し送る（再実行時は下記「再実行時の注意（#1899）」に従い既存ログを退避する）。

## 実行手順

```sh
cd docs/perf/logs/metal-gemm-splitk-auto-entry-1513
sh run_auto_entry.sh
```

## 生成物

- `auto_entry.log`: `cargo test -p fandhe-ai-backend-metal --release --test
  gemm_splitk_auto_entry_parity -- --ignored --nocapture` の全出力（本
  イシューの主対象。2 テスト: `auto_entry_dispatches_split_k_for_eligible_
  shapes_and_matches_baseline`〈`[auto-entry] m=.. n=.. k=.. trans_a=..
  trans_b=.. fail_count=x/y ...` 形式の 44 行 + テスト結果〉・
  `auto_entry_falls_back_to_classic_not_eligible_for_non_split_k_shapes`）
- `bit_match.log`: 既存 AC-1（`gemm_splitk_bit_match.rs`）の非後退確認出力
- `parity.log`: 既存 AC-2（`gemm_splitk_parity.rs`。`_with_plan` 版）の
  非後退確認出力
- `env_info.txt`: `uname -srm`・`sw_vers`・`rustc -V`・
  `sysctl machdep.cpu.brand_string`・実行コミットの `git rev-parse HEAD`
  （内部ホスト名は含めない）
- `uptime_before.txt`／`uptime_after.txt`: 実行前後の 1 回計測
- `uptime_during.log`: 実行中 10 秒間隔でサンプリングした load average
  推移（専有ゲートは不要だが、判定への影響有無を事後確認するため記録する）

## 事前登録判定規則（計測後に緩和しない。`docs/perf/metal-gemm-splitk-two-pass.md` §5.9 と同一）

1. 実行コマンドは `run_auto_entry.sh` のもの
2. pass 条件:
   - (1) `auto_entry_dispatches_split_k_for_eligible_shapes_and_matches_
     baseline` が `ok`（= 承認済み 11 形状 × 4 パターン = 44 組合せすべてで
     `SplitKRoute::Split` へ到達し、`partitions` が `should_split_k` の
     算出値と一致し、4 指標が `common::splitk_parity_baseline::BASELINES`
     の ceiling 以下）
   - (2) `auto_entry_falls_back_to_classic_not_eligible_for_non_split_k_
     shapes` が `ok`（= `(512,512,512)`・`(64,64,63)` で
     `SplitKFallbackReason::NotEligible` を返し
     `NumericContractPendingApproval` ではない・CPU 参照実装と bit 完全
     一致）
   - (3) `bit_match.log`・`parity.log` の既存 `#[ignore]` テストが引き続き
     全て `ok`（本切替による classic 経路・split-K 経路双方の非後退確認）
3. ceiling 超過・`NumericContractPendingApproval` の再出現・classic 経路
   への意図しないフォールバックは FAIL として記録する（baseline の上方
   更新・ゲートの再無効化はユーザー承認事項。`gemm.rs::
   SPLIT_K_NUMERIC_CONTRACT_APPROVED` の doc コメント参照）
4. 共有負荷下で可（専有ゲート不要。parity・到達確認は負荷非依存）。
   `uptime_during.log` の load average 推移を `docs/perf/metal-gemm-splitk-
   two-pass.md` §5.9 へ転記する

**改訂（#1899・ユーザー承認 2026-09-16）**: 規則 2(2) は `(512,512,512)` と
`(64,64,63)` を単一の `NON_ELIGIBLE_SHAPES` fixture として一律
`Classic{NotEligible}` を期待する原文のまま**残す**（歴史記録）。実際には
`(64,64,63)`（m/n/k が 8 の倍数でない）は `strided_tiled_eligibility` の
事前条件に違反するため `Err(MetalError::StridedTiledIneligible)` を返す
のが正しい契約（(A)。`docs/backend-metal-splitk-decision.md` §5「自動判定
入口の事前条件契約（#1899）」）であることが 2026-09-16 の実測（下記
「実測結果（#1904）」）で判明し確定した。是正後は fixture を 2 分割し、
`(512,512,512)` のみを本規則 2(2) の対象（`Classic{NotEligible}` 期待）
として残し、`(64,64,63)` は別テスト
`auto_entry_rejects_precondition_violating_shapes_with_typed_err`
（`Err(MetalError::StridedTiledIneligible)`＋`c_buf` 無変更を期待）へ
移した。これは入口契約の是正であり、tolerance・`BASELINES`・
`dispatch_auto` の判定は不変（閾値緩和ではない）。

## 実測結果（#1904・2026-09-16・origin/main `3e43bbd0`・共有負荷下）

- (1): **ok**（11 形状 × 4 パターン = 44 組合せすべて `SplitKRoute::Split`
  へ到達し baseline ceiling 以下）
- (2): **FAILED**（`(512,512,512)` は `Classic{NotEligible}`＋bit 一致を
  通過したが、`(64,64,63)` で `dispatch_split_k_strided_prepared` が
  `strided tiled GEMM route ineligible: m/n/k must all be multiples of 8`
  の `Err` を返し panic した。`run_auto_entry.sh` は `set -e` により本
  FAIL で中断。是正・契約確定は #1899）
- (3): **ok**（`gemm_splitk_bit_match.rs` 2 件・`gemm_splitk_parity.rs`
  〈#1512 baseline 方式再切替後、実機で初めて全形状 pass〉。手動実行で
  補完）
- 採否判定: なし（本イシュー #1513 のスコープは数値契約ゲート解除の
  確認のみ・入口契約自体の判断は #1899 へ切り出し）

詳細は `docs/perf/metal-gemm-splitk-two-pass.md` §5.9「M4 Max 実機実測
（2026-09-16・origin/main `3e43bbd0`・共有負荷下）」を正とする。

## 再実行時の注意（#1899）

`run_auto_entry.sh` は固定ファイル名（`auto_entry.log` 等）へ**上書き**する
ため、#1899 是正後の fixture（2 分割・新規テスト 1 件追加）で再実行すると
上記 #1904 の FAIL 証跡が失われる。`docs/perf/logs/metal-realdevice-
phase2-2026-09-16/README.md` §3.2 が #1904 の結果を引用しているため、
再実行前に既存ログを日付・イシュー番号付きで退避すること（例:
`for f in auto_entry bit_match parity; do cp "$f.log" "$f.2026-09-16-1904.log"; done`）。
`run_auto_entry.sh` 自体（事前登録コマンド）は変更しない。

## 関連

- `docs/backend-metal-splitk-parity-judgment-decision.md`（承認記録 §7）
- `docs/perf/metal-gemm-splitk-two-pass.md` §5.9（本イシューのゲート解除
  記録）
- `crates/backend-metal/src/gemm.rs::SPLIT_K_NUMERIC_CONTRACT_APPROVED`
  （本イシューで `true` へ切替した定数）
- `crates/backend-metal/tests/gemm_splitk_auto_entry_parity.rs`（本イシュー
  で新設した受け入れテスト。#1899 で fixture 2 分割・
  `auto_entry_rejects_precondition_violating_shapes_with_typed_err` を追加）
- `docs/backend-metal-splitk-decision.md` §5「自動判定入口の事前条件契約
  （#1899）」（入口の事前条件契約 (A) の確定記録）
- `crates/backend-metal/tests/splitk_parity_baseline_contract.rs::
  approved_baseline_shapes_are_split_k_eligible`（Linux 実行可能な前提条件
  の集合検査。本イシューで新設）
- `docs/perf/logs/metal-gemm-splitk-parity-baseline-1512/`（#1512 の同型
  ログ置き場。本ディレクトリの雛形）
