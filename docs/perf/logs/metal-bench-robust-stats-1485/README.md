# M4 Max 実機非後退確認（イシュー #1485）

ロバスト統計の採用決定（併記のみ・置換なし・閾値不変・案 E 不採用。
`docs/perf/metal-bench-noise-protocol.md` §8.6 の承認記録）を受け、
`StabilityResult::aux`（#1483）・`phase1_round_stats` 行への補助キー出力
（#1484）が実機で機能し、かつ `within_gate` 判定が補助統計の影響を
一切受けないことを M4 Max 実機で機械確認した記録（コード変更なし。
本イシューでは性能ゲート達成の確認は目的としない）。

## 目的

1. `trimmed_spread_k1`／`iqr_spread`／`mad_spread` の 3 キーが
   `phase1_round_stats` 行へこの順で出力されること
2. `within_gate` が同一行の `spread <= gate` の数値比較とすべて一致
   すること（補助値が判定に影響していない機械的証拠）
3. 既存キー順（`size rounds spread gate within_gate median_secs
   min_secs min_round_idx max_secs max_round_idx round_medians_secs`）
   が既存ログと一致すること
4. 既存 `#[ignore]` bit 一致・parity テスト群・example 単体テスト・
   `bench-harness` テスト・集計スクリプトの `--self-test` が非後退
   であること

## 再現コマンド

```sh
cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench \
  --release --features internal-diagnostics -- --phase1-only
```

`--max-load-avg` を付けないため環境ガードは `record_only`（判定なし・
待機なし）。共有負荷下のため多くのサイズで `within_gate=false` に
なることは想定内（ゲート成立自体は本イシューの目的ではない）。

## ファイル一覧

- `phase1_only_run1.log`: 上記コマンドの標準出力全文
- `env_info.txt`: `uname -m`／`rustc -V`／`git rev-parse HEAD`／`uptime`
  （内部ホスト名・ユーザーパスは含めない）
- `uptime_before.txt`／`uptime_after.txt`: 実行前後の `uptime`
- `pmset_therm_before.txt`／`pmset_therm_after.txt`: 実行前後の
  `pmset -g therm`（サーマル状態）
- `ignored_tests.log`: `#[ignore]` bit 一致・parity テスト群・example
  単体テスト・`bench-harness` テスト・集計スクリプト `--self-test` の
  実行結果（既存動作の非後退確認）
- `aggregate.md`: `phase1_round_stats` 行の集計表（size 別
  `spread`／`gate`／`within_gate`／3 補助キー）と `verify_1485.py` の
  実行結果全文
- `verify_1485.py`: 判定不変・キー順を機械検査するスクリプト（標準
  ライブラリのみ。`docs/perf/logs/metal-gemm-transpose-route-ab-1242/
  aggregate.py` 等と同じ Python3 標準ライブラリのみ方針）

## 判定不変の確認方法

`phase1_only_run1.log` から `phase1_round_stats` 行を抽出し、以下を
`verify_1485.py`（標準ライブラリのみの Python3 スクリプト）で機械検査
した（実行例: `python3 verify_1485.py phase1_only_run1.log`。詳細な
結果は `aggregate.md` を参照）:

- 行数が 5（size=256/512/1024/2048/4096）
- 各行末尾に `trimmed_spread_k1=`・`iqr_spread=`・`mad_spread=` の
  3 キーがこの順で存在
- 各行で `within_gate` の値が同一行の `spread <= gate` の数値比較と
  一致
- `trimmed_spread_k1=` より前のキー列が既存ログ
  （`docs/perf/logs/metal-gemm-transpose-route-ab-1242/1267-attempt1-run.log`
  L12 等）のキー順と一致
