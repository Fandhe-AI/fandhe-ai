# train／infer の phase 内訳と同期固定費（Mac mini・イシュー #3180）

数値の正式記録は `docs/perf/train-step-phase-breakdown.md` §18・`docs/perf/infer-reuse-phase-breakdown.md` §11。
本ディレクトリはその生ログ一式。**診断のみ**（tolerance・判定規則・相手側ベンチ・ライブラリの `src/` と既定値は不変）。
規則の正は `RULE.txt`（計測前に固定）。

## 構成

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（固定 2026-10-20T02:05:00Z。計測前コミットに含む） |
| `orchestrate_macmini.sh` | Mac mini 側の計測本体（ビルド → 5 run × 8 セル → 各 run 後に同期固定費 → カウンタ 2 本 → クロスチェック） |
| `aggregate.py` | 集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`／`python3 -I aggregate.py . > aggregate.md`）。欠損・非数値は拒否 |
| `run{1..5}.jsonl`・`run{1..5}.err` | 腕 A の生ログ（`bench-fandhe --phases`。`.err` は 5 本とも空） |
| `load_gate.log` | 負荷ゲート記録（run 開始前 load1 < 3.0。5/5 `gate=pass`） |
| `rev-stamp.txt`・`tree.txt`・`env_info.txt` | 腕の同定（フル SHA・registry ピン 1 行・機体情報。ホスト名は含めない） |
| `counters/{batch_counters,backward_dinput_phase}.log` | 腕 B の同期回数（`<WORK>` は作業ディレクトリのマスク） |
| `sync-fixed-cost/run{1..5}.log`・`infer_fixed_cost_diag.log` | 同期 1 回の固定費（空 dispatch）5 run と、wall − GPU busy のクロスチェック |
| `cpu-diag/device_update_phases_run2.log` | 参考（単発）。既存の #2106 診断テストによる CPU `device_update` の内訳 |
| `aggregate.md` | 上記の機械集計 |

## 腕と条件

- **ツリー**: 計測前コミット `b2267d46458eccdb671c1cf477276fef0e714537`（main `5f75d633f` ＋本 PR の新規ファイル）を
  `git archive` して `.rev-stamp` を置き rsync で転送した。ツリー指紋（`find crates scripts/bench/framework-compare -type f
  ! -path '*/target*'` を sort した各 sha256 の sha256）は送信側・Mac mini 側とも `ddb733c94118…09b22` で一致。
- **腕 A（phase 内訳）**: `bench-fandhe` を registry `fandhe-ai =0.11.0` のままビルド（`tree.txt` は `fandhe-ai v0.11.0` の 1 行）。
- **腕 B（同期回数・固定費）**: 同ツリーの main workspace（facade・backend-metal は path 依存＝HEAD ソース。v0.11.0 から 95 commit 先）。
  **腕 A と腕 B のソース版が違う**。bench-fandhe は registry 版のため同期カウンタ（`#[doc(hidden)]`・facade 非再公開）を読めず、
  backend-metal の直接依存追加は依存 allowlist の変更で所有者承認が要るため行っていない。このため「同一バイナリで回数も記録」はできていない。
- **機体**: Mac mini（Apple M4・4P＋6E・16 GB・macOS 27.0.1・rustc 1.99.0）。8 セル（cpu／metal × train／infer × fresh／reuse）× 5 run。
- **負荷**: 5/5 `gate=pass`。run1 前は直前のビルド直後で 3 分待った（`load_gate.log`）。計測中に他のビルド・ベンチは並走させていない
  （`env_info.txt` の開始時 load は直前ビルドの残り）。ただし同じ Mac mini は他の作業にも使われる共有機であり、起動直前に cargo／bench のプロセスが無いことは確認した。
- **target**: main workspace は `target-3180`、framework-compare は `target-3180-fc`（既存の共有 target は触っていない）。

## 注意・限界

- `aggregate.py` は計測後に 1 点だけ直した（`SYNC_FIXED_COST_EMPTY_DISPATCH` 行を `--nocapture` 出力の接頭辞つきでも拾うよう正規表現を `search` へ）。計測値・判定には影響しない。
- `phases` の `step_total` は通常ベンチの測定窓より狭く、系列 B（`framework-compare-0.11.0-macmini-remeasure`）の値と一致しない
  （例: train Metal reuse は `step_total` 1885.9 µs・系列 B 2132.9 µs）。差の原因は切り分けていない。
- 空 dispatch の固定費は実カーネルを伴う同期を代表しない（§18.5）。
- **参考系列（cpu-diag）の扱い**: `diag-instrumentation.patch` による backward 内部の内訳は**未取得**（時間の制約で適用を試みていない）。
  代替として #2106 の診断テストを 2 回回し（1 回目は §18 に数値のみ転記、2 回目のログを `cpu-diag/` に保存）。なお 2 回目の実行時、Mac mini 上の
  展開ツリーの `Cargo.lock` を cargo が再解決して更新した（コミット物の `Cargo.lock` は不変。正式系列の計測はすべてこの更新前に完了）。
  このため cpu-diag は依存版が正式系列と異なる可能性がある参考値で、正式値ではない。
- 副系列 B'（v0.11.0 タグのツリーでのカウンタ確認）は未実施。
- checksum が一致しても bit 同一の証拠ではない。
