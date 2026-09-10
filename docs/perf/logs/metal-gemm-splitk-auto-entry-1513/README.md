# Metal split-K 公開入口の数値契約ゲート解除（イシュー #1513）実測ログ

`SPLIT_K_NUMERIC_CONTRACT_APPROVED` を `false` → `true` へ切り替え、自動判定
入口 `MetalGemm::dispatch_split_k_strided_prepared` が承認済み形状で実際に
split-K 経路を実行するようになったこと（承認記録:
`docs/backend-metal-splitk-parity-judgment-decision.md` §7・2026-09-10
ユーザー承認）を Apple M4 Max 実機で確認するログ置き場。

**現状（本 PR 時点）**: このディレクトリはまだ空。本 PR を書いた実行環境に
Apple Silicon 実機へのアクセス経路がないため、下記の実測は未実施のまま
記入欄として残す。Mac セッションで `run_auto_entry.sh` を実行し、生成物を
このディレクトリへ収める。

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

## 関連

- `docs/backend-metal-splitk-parity-judgment-decision.md`（承認記録 §7）
- `docs/perf/metal-gemm-splitk-two-pass.md` §5.9（本イシューのゲート解除
  記録）
- `crates/backend-metal/src/gemm.rs::SPLIT_K_NUMERIC_CONTRACT_APPROVED`
  （本イシューで `true` へ切替した定数）
- `crates/backend-metal/tests/gemm_splitk_auto_entry_parity.rs`（本イシュー
  で新設した受け入れテスト）
- `crates/backend-metal/tests/splitk_parity_baseline_contract.rs::
  approved_baseline_shapes_are_split_k_eligible`（Linux 実行可能な前提条件
  の集合検査。本イシューで新設）
- `docs/perf/logs/metal-gemm-splitk-parity-baseline-1512/`（#1512 の同型
  ログ置き場。本ディレクトリの雛形）
