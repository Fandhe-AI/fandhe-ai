# Metal f32 split-K parity baseline 再切替（イシュー #1512）実測ログ

`crates/backend-metal/tests/gemm_splitk_parity.rs` を厳密ゼロ fail 判定
（`assert_parity`）から実測ベースライン非後退方式（`assert_no_split_k_
parity_regression`）へ再切替したこと（承認記録:
`docs/backend-metal-splitk-parity-judgment-decision.md` §7・2026-09-10
ユーザー承認）を Apple M4 Max 実機で確認するログ置き場。

**現状（本 PR 時点）**: このディレクトリはまだ空。本 PR を書いた実行環境に
Apple Silicon 実機へのアクセス経路がないため、下記の実測は未実施のまま
記入欄として残す。Mac セッションで `run_parity.sh` を実行し、生成物を
このディレクトリへ収める。

## 実行手順

```sh
cd docs/perf/logs/metal-gemm-splitk-parity-baseline-1512
sh run_parity.sh
```

## 生成物

- `parity.log`: `cargo test -p fandhe-ai-backend-metal --release --features
  internal-diagnostics --test gemm_splitk_parity -- --ignored --nocapture`
  の全出力（`m=.. n=.. k=.. trans_a=.. trans_b=.. fail_count=x/y
  max_abs_diff=.. mean_abs_diff=.. max_rel_err=..` 形式の 44 行 +
  テスト結果サマリを含む）
- `env_info.txt`: `uname -srm`・`sw_vers`・`rustc -V`・
  `sysctl machdep.cpu.brand_string`・実行コミットの `git rev-parse HEAD`
  （内部ホスト名は含めない）
- `uptime_before.txt`／`uptime_after.txt`: 実行前後の 1 回計測
- `uptime_during.log`: 実行中 10 秒間隔でサンプリングした load average
  推移（専有ゲートは不要だが、判定への影響有無を事後確認するため記録する）

## 事前登録判定規則（計測後に緩和しない。`docs/perf/metal-gemm-splitk-two-pass.md` §5.8 と同一）

1. 実行コマンドは `run_parity.sh` のもの（`--features internal-diagnostics`
   必須。feature 未指定はテストバイナリがビルドされず暗黙 pass になるため、
   `parity.log` に `running 1 test` と `test ... ok` の両方があることを
   確認する）
2. pass 条件:
   - (1) `#[ignore]` テスト 1 件が `ok`（= 44 組合せすべてで
     `SplitKRoute::Split` 到達かつ 4 指標が ceiling 以下）
   - (2) `parity.log` の 11 形状 × 4 パターンの出力値がすべて
     `crates/backend-metal/tests/common/splitk_parity_baseline.rs::
     BASELINES` の ceiling 以下
   - (3) 同一 `(m, n, k)` の 4 パターン（NN/NT/TN/TT）が同一集計値
3. 実測値が `docs/perf/metal-gemm-splitk-two-pass.md` §5.3 の表と bit
   同一でなくとも ceiling 以下なら pass（差分があれば §5.8 に「実測値差分
   あり・非後退」と記録する）。ceiling 超過は FAIL として記録し、baseline
   の上方更新はしない（ユーザー承認事項。`common/splitk_parity_baseline.rs`
   の doc コメント参照）
4. 共有負荷下で可（専有ゲート不要。parity は負荷非依存）。`uptime_during.log`
   の load average 推移を `docs/perf/metal-gemm-splitk-two-pass.md` §5.8 へ
   転記する

## 関連

- `docs/backend-metal-splitk-parity-judgment-decision.md`（承認記録 §7）
- `docs/perf/metal-gemm-splitk-two-pass.md` §5.5／§5.8（判定方式の経緯・
  再切替の記録）
- `crates/backend-metal/tests/gemm_splitk_parity.rs`（判定の実利用側）
- `crates/backend-metal/tests/splitk_parity_baseline_contract.rs`（Linux
  実行可能な型検査・falsification テスト。本 PR で新設）
