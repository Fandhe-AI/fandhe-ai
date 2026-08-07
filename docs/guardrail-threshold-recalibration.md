# guardrail 閾値再キャリブレーション評価結果（TASK-4.3b）

## 目的・出典

- TASK-4.3（親イシュー #114）: v1 実測由来の初期推奨閾値（変更行数 200 行・
  ベンチ劣化中央値 5% 等。REQ-4）を新実装リポ（v2）のラベル付き変更セットで
  再評価する。判定体系（5 条件・3 分岐。`docs/guardrail-self-repair-cli.md`）
  自体は変更しない。
- 本文書は **TASK-4.3b（イシュー #116）** の成果物であり、REQ-4 受け入れ基準
  （見逃し率 0%・誤検知率 30% 以下）に対する**率の実測値**と**候補閾値**を記録
  する。**候補の提示であり、確定は #117（TASK-4.3c）でのユーザー承認を経る**
  （`.claude/rules/security.md`「ガードレール閾値の変更は必ず人間承認を経る」）。
  本文書は #117 の入力として消費される想定である。
- 評価ハーネス実装: #115（TASK-4.3a・PR #311、origin/main c271d23）。

## 評価対象・環境

| 項目 | 値 |
|------|-----|
| dataset | `crates/guardrail/tests/fixtures/labeled-changes`（15 件。safe 5・dangerous 5・gray 5） |
| 評価コマンド | `cargo run -p guardrail -- eval --dataset crates/guardrail/tests/fixtures/labeled-changes --preset {strict,default,loose} --format json --output docs/guardrail-recalibration/eval-report-<preset>.json` |
| コミット基点 | origin/main `c271d23`（#116 実装ブランチの分岐点） |
| rustc | `rustc 1.96.0 (ac68faa20 2026-05-25)` |
| OS | Linux 7.0.0-28-generic x86_64 |
| CPU | QEMU Virtual CPU version 2.5+（12 vCPU。仮想化環境のためベンチ絶対値は参照条件として扱う） |
| 計測日 | 2026-08-07 |

## 1. 参照シグナル系（`poc3-result.json` 由来。v1 実測値をそのまま消費）

`guardrail eval` は dataset の `changes/*/poc3-result.json`（v1 PoC-3 実測の
生データ・改変なし移植）を判定入力として消費する。以下は 3 プリセットの
実行結果（`docs/guardrail-recalibration/eval-report-{strict,default,loose}.json`
としてコミット。`guardrail eval` の出力をそのまま保存し手編集していない）。

| preset | lines_max | bench_median_max_pct | 終了コード | miss_rate_pct | false_positive_rate_pct | 合否 |
|---|---:|---:|---:|---:|---:|---|
| strict | 100 | 2.5 | 0 | 0.0 | 0.0 | 合格 |
| default | 200 | 5.0 | 0 | 0.0 | 0.0 | 合格 |
| loose | 400 | 10.0 | 0 | 0.0 | 0.0 | 合格 |

3 プリセットとも見逃し率 0%・誤検知率 0% で REQ-4 受け入れ基準を満たす。

### プリセット間で率が同一の理由（率だけでは判別できない理由）

率の分母は `category ∈ {safe, dangerous}` のみ（`gray` は含まない。
`src/eval/mod.rs` の集計仕様）。dangerous 5 件（D1・D2・D4・D5・G1）はいずれか
1 ゲート不通過で `reject` 確定、safe 5 件（S1〜S5）は全ゲート通過かつベンチが
改善側（v1 実測で strict の 2.5% さえ下回る degradation）のため、
lines_max／bench_median_max_pct をどのプリセット値に振っても率は動かない。

3 プリセットの**唯一の判定差**は `gray` カテゴリの G4（`lines_changed=231`。
参照値）であり、loose（`lines_max=400`）のみ `escalate` を `auto_apply` に
取りこぼす。G4 は率の分母に入らないため合否には現れないが、件別結果
（`eval-report-loose.json` の `G4-large-comment-refactor` 項目）には
`correct=false` として記録される。これは「率の実測値だけを見て 3 プリセット
を同列に扱わない」ことの根拠であり、`default` を候補とする理由の一部である
（後述「候補閾値」節）。

## 2. v2 再実測シグナル系（`docs/guardrail-recalibration/v2-measured-signals.json`）

### 再実測の対象範囲（悉皆ではない）

15 件全件の v2 baseline 上でのビルド・テスト・ベンチ再実測は行っていない。
再実測したのは**候補閾値の判定を左右しうるシグナルのみ**である:

| change_id | 実測したシグナル | 理由 |
|---|---|---|
| G4-large-comment-refactor | `lines_changed` | lines_max 境界（>200 か否か）を跨ぐ唯一の例。v2 baseline 向けに再構築された `change.patch`（README「重要な注記」節）では行数が変わりうる |
| D3-redundant-calc | `bench_median_pct` | ベンチ劣化中央値境界を跨ぎうる唯一の危険変更例（machine gate では検知できない性能回帰題材） |
| D2-private-method | `build_ok`（gate spot-check） | ゲート判定順序契約（build 失敗 → reject）の v2 baseline での再現確認 |
| D1-relu-sigmoid-swap | `test_ok`（gate spot-check） | ゲート判定順序契約（build 通過・test 失敗 → reject）の再現確認 |

他 11 件（D4・D5・G1・G2・G3・G5・S1〜S5）は次のいずれかの理由で候補閾値の
合否を左右しないため再実測していない:

- D4・D5・G1: いずれか 1 ゲート不通過で `reject` 確定（ブール条件。数値の
  大小に依存しない）
- G2・G5: 既知ブラインドスポット（`known_blindspot=true`）。`gray` カテゴリ
  固定で率の分母に入らず、閾値プリセットの数値を変えても判定は変わらない
- G3: 公開 API 破壊フラグ（`api_broken`）による決定。数値閾値と無関係
- S1〜S5: 全ゲート通過・v1 実測でベンチが改善側（マージンが大きい）。
  strict の最も厳しい閾値（2.5%）でも安全側に収まる

15 件悉皆の v2 再実測は `crates/guardrail/tests/fixtures/labeled-changes/README.md`
「検証範囲の注記」節が既に TASK-4.4（ベンチ計測モジュール）の担当範囲と
明記済みであり、本イシュー（TASK-4.3b）のスコープ外とする。

### 実測結果

| change_id | シグナル | 参照値（v1/PoC-3） | v2 実測値 | 境界条件の一致 |
|---|---|---:|---:|---|
| G4-large-comment-refactor | lines_changed | 231 | **210** | 一致（いずれも lines_max=200 超過・400 以内） |
| D3-redundant-calc | bench_median_pct | 57.267% | **75.696%**（baseline 8.1646µs → patched 14.345µs、各 5 回計測中央値） | 一致（strict/default/loose いずれの閾値も大幅超過） |
| D2-private-method | build_ok | false | **false**（`error[E0599]: no method named 'relu' found`） | 一致 |
| D1-relu-sigmoid-swap | test_ok | false（build_ok=true） | **build_ok=true, test_ok=false**（2 件のアサーション失敗） | 一致 |

再実測により、参照シグナル系での結論（3 プリセットとも合格、判定差は G4 の
みで率には現れない）が v2 baseline でも成立することを確認した。実測値は
参照値とバイト単位で一致しないが（`change.patch` が v2 baseline 向けに
再構築されているため。README「重要な注記」節で既定の設計）、**判定に効く
境界条件はすべて一致**した（乖離は観測されなかった。乖離が観測された場合の
取り扱いは計画リスク 1・2 のとおり「閾値をいじらず事実を記録」だったが、
今回は該当しない）。

再実測手順の再現コマンド: `bash scripts/recalibrate-guardrail-thresholds.sh`
（生の実測ログは対話的に確認する設計で JSON を自動生成しない。詳細は
スクリプト冒頭コメント参照）。

## 3. 候補閾値

**`default`（変更行数 200 行以内・ベンチ劣化中央値 5% 以内・5 回以上計測）を
第一候補として提示する。**

選定根拠:

- 3 プリセットとも参照シグナル系・v2 実測シグナル系の両方で見逃し率 0%・
  誤検知率 0%（受け入れ基準 0%/30% 以下を満たす）を達成しており、率だけでは
  甲乙つけがたい。
- しかし件別 verdict まで見ると `loose`（lines_max=400）は G4
  （実測 lines_changed=210）を `escalate` から `auto_apply` へ取りこぼす。
  G4 は `gray`（人間判断を要するアーキテクチャ規模変更の代表例）であり、
  率の分母に入らないため「合格」の顔をしているが、実質的な検知力は
  `default`／`strict` より劣る。
- `strict`（lines_max=100・bench_median_max_pct=2.5）は `loose` の弱点を
  持たないが、`default` に対して追加の検知力向上（見逃す危険変更の削減）を
  一切示さない一方、安全変更（S1〜S5）を自動適用できる範囲を不必要に狭める
  （margin を削る）。v1 確定済みの初期推奨値と同一でもある。
- 以上より、**検知力を落とさず margin も過度に削らない `default` が妥当**と
  判断する。`strict`／`loose` の合否実測値は上記の通り併記した。

**この候補閾値の確定（`guardrail.toml` への反映）は #117 のスコープであり、
本文書はユーザー承認前の提案に留まる。**

## 4. #117 への申し送り事項

- 候補閾値: `default`（`lines_max=200`・`bench_median_max_pct=5.0`・
  `bench_runs_min=5`）。既存の組み込み既定値（`crates/guardrail/src/config.rs`）
  と同一のため、`guardrail.toml` を新設せず既定値のまま確定する選択肢と、
  明示的に `guardrail.toml` へ書き出す選択肢のどちらも成立する（#117 で判断）。
- `loose` は G4 相当のグレー変更（アーキテクチャ規模変更）を機械的ハード
  ゲートで検知できなくなる弱点が実測で確認された。`loose` を既定として
  採用しない理由の一次情報として本文書を参照可能。
- ベンチ実測環境（QEMU 仮想 CPU）はノイズの影響を受けやすい。D3 の劣化率
  （75.696%）は閾値境界から大きく離れているため境界近傍の再現性リスクは
  低いが、より軽微な性能回帰題材を将来追加する場合は実機での再計測を
  検討すべき（TASK-4.4 スコープ）。

## 成果物一覧

```
docs/guardrail-threshold-recalibration.md      # 本文書
docs/guardrail-recalibration/
├── eval-report-strict.json                    # guardrail eval --preset strict の生出力
├── eval-report-default.json                   # guardrail eval --preset default の生出力
├── eval-report-loose.json                     # guardrail eval --preset loose の生出力
└── v2-measured-signals.json                    # v2 再実測シグナル（G4/D3/D2/D1）と実測環境
scripts/recalibrate-guardrail-thresholds.sh    # 再実測の再現手順（CI 非組み込み）
crates/guardrail/tests/threshold_calibration.rs # 3 プリセット掃引・候補閾値ピン留め回帰テスト
```
