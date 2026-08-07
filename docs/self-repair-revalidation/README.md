# 自己修復ループ再実証結果の記録（TASK-3.3d・イシュー #143）

TASK-3.3（`docs/spec/05-tasks.md:144`）「自作コア上での自己修復ループ完走の
再実証」の成果物本体である。REQ-3 の v2 追加受け入れ基準
（`docs/spec/04-requirements.md:96`）「自作コアに対する自己修復ループの
人間介在なし完走を新実装リポで再実証すること」に対し、バグ修正種別
（[`bug-fix/`](./bug-fix/)・TASK-3.3b・#141）と機能追加種別
（[`feature-addition/`](./feature-addition/)・TASK-3.3c・#142）の 2 種別
それぞれで完走実証を実施し、その記録を本ディレクトリへ整備した
（TASK-3.3a・#140 の実証計画・完走判定基準に基づく）。

本 README は 2 種別の記録を統合し、TASK-3.4（#145）実装後に適用可能となった
改竄検知ログ（JSON Lines ハッシュチェーン）を両ハーネスへ結線したうえで
（TASK-3.3d・本イシュー）、#144（TASK-3.3e・人間評価）の入力となる統合結果を
提供する。

## 1. サマリ

| 種別 | 記録先 | 最終結論 | 試行回数 | 合計所要時間 | 判断根拠（要約） |
|------|--------|---------|---------|-------------|-----------------|
| バグ修正（TASK-3.3b・#141） | [`bug-fix/`](./bug-fix/) | `LoopOutcome::Adopted` | 2 | 17,078 ms（`loop-report.json` `total_duration_ms`） | attempt 1: `cargo test --release` の既知正解値テスト（`mlp_grad_*_matches_numeric`）が analytic/numeric 勾配不一致で失敗し却下。attempt 2: build/test/clippy 全通過＋ベンチゲート機構完走（合成ワークロード・劣化率中央値 0.234%）＋diff 由来シグナルが `guardrail::decide` の自動適用条件を満たし採用 |
| 機能追加（TASK-3.3c・#142） | [`feature-addition/`](./feature-addition/) | `LoopOutcome::Adopted` | 2 | 9,484 ms（`loop-report.json` `total_duration_ms`。ハーネス全体の壁時計時間は `harness_wall_time_ms` 9,557 ms） | attempt 1: 符号分岐を欠く誤実装で受け入れ基準テスト（`leaky_relu_matches_known_values`。`got=0.05, want=0.5`）不合格・却下。attempt 2: 既存組み込み演算（`relu`・四則）合成による正実装で build/test/clippy 全通過＋ベンチ実測（劣化率中央値 -0.0052%）＋diff 由来シグナル（`lines_changed=28`・`api_broken=false`・`gaming_suspect=false`）が自動適用条件を満たし採用 |

両種別とも `guardrail.toml`（TASK-4.3c・#117 確定値。`bench_median_max_pct=5.0`・
`bench_runs_min=5`・`lines_max=200`）を一切変更せず、`guardrail::decide` を
唯一の取り込み判断経路として使用した（迂回経路なし）。数値の出典は各
`loop-report.json`（`docs/self-repair-revalidation/bug-fix/loop-report.json`・
`docs/self-repair-revalidation/feature-addition/loop-report.json`）。

## 2. 完走判定基準の充足マトリクス

実証計画（`docs/self-repair-revalidation-plan.md` 5 節）が定める判定基準
6 項目に対する、2 種別の充足状況を統合する。バグ修正種別は
`bug-fix/README.md` §4 の判定表からそのまま転記し、機能追加種別は
（同種の判定表を個別 README に持たないため）本 README が
`feature-addition/loop-report.json`・`feature-addition/README.md` の実測値・
記述から導出した（過大表現を避けるため、両種別とも実際の実測範囲を超えて
「充足」と判定しない）。

| # | 判定基準 | バグ修正（#141） | 機能追加（#142） |
|---|---------|------------------|------------------|
| 1 | `self-repair run` の 1 回起動・追加の人間入力なしで `AutoApply` へ到達 | 部分充足（CLI 未実装・lib 直接呼び出しで代替） | 部分充足（同左） |
| 2 | 検証 4 ゲート全通過・`guardrail` 3 分岐判定を迂回なく経由 | 部分充足（3 ゲートは実測、ベンチは機構完走確認〈`bench` フィールドは `NotRun` のまま `guardrail::decide` へ渡る〉） | 充足（`FeatureAdditionCompositeGate` が build/test/clippy＋ベンチを全ゲート通過後に実測し、`bench` フィールドが `Measured` として `guardrail::decide` へ渡る。`gate_report: "build=pass test=pass clippy=pass bench=measured"`。§3 参照） |
| 3 | `--max-attempts` 上限内で完走 | 充足（`max_attempts=2` で attempt 2 採用） | 充足（`max_attempts=5` で attempt 2 採用） |
| 4 | JSON Lines ログのハッシュチェーン検証（`self-repair verify-log`）を通過 | **部分充足（本イシューで新規充足）**。`loop-log.jsonl` を出力・lib `verify_chain` で検証済み。外部コマンド CLI は未実装 | **部分充足（本イシューで新規充足）**。同左 |
| 5 | ベンチ劣化中央値が承認済み閾値内（5 回計測中央値） | 部分充足（機構完走確認は合成ワークロード。閾値内 0.234% < 5.0%） | 部分充足（判定基準 2 とは異なり計測経路自体は実測だが、baseline・candidate 双方に同一の合成ワークロード〈`leaky_relu_like_workload`〉を用いており、`feature-addition/README.md`「シグナルは実測のみ」節が明記するとおり「真の劣化率は構造的に 0% 近傍になる」——候補実装固有の性能劣化を検出するものではない。閾値内 -0.0052% < 5.0% は経路実測の結果であり、性能特性そのものの実測結果ではない） |
| 6 | 判定レポート JSON の `signal_source` が `"measured"` | 未充足（スコープ外。`signal_source` は CLI 出力仕様） | 充足（`loop-report.json.signal_source == "measured"`） |

判定基準 4 が両種別とも「未充足」から「部分充足」へ移行したことが本イシュー
（#143）の主な成果である。「部分充足」に留まる理由は次節で明確化する。

## 3. 判定基準 4 の充足範囲（過大表現をしない）

- **充足していること**: TASK-3.4（#145）が実装した
  `self_repair::LogWriter::append_report`/`append_failure` を両ハーネスへ
  結線し、ループ実行のたび `loop-log.jsonl`（JSON Lines・SHA-256 ハッシュ
  チェーン。`docs/self-repair-log-format.md` 準拠）を出力した。書き出し
  直後にハーネス自身が `self_repair::verify_chain` を呼び、`Ok`（チェーン
  整合）であることを assert している（テスト成功がその実証）。
- **充足していないこと**: `self-repair verify-log`
  （`docs/guardrail-self-repair-cli.md` §3.2 が境界を定める外部コマンド）は
  CLI バイナリ本体が未実装のため呼び出していない。本実証における「検証を
  通過した」は **lib 直接呼び出し（`verify_chain` 関数呼び出し）による検証**
  であり、独立プロセスとしての CLI 経由検証ではない。
- 監査手順・改竄検知の実効性（フィールド改変・レコード削除・順序入れ替え・
  未知フィールド注入の検知）は
  [`bug-fix/README.md` §8](./bug-fix/README.md#8-改竄検知ログloop-logjsonlの監査手順) /
  [`feature-addition/README.md`「改竄検知ログ」節](./feature-addition/README.md#改竄検知ログloop-logjsonl)
  に詳細を記載し、`crates/self-repair/src/logging.rs` の既存単体テストへの
  参照で実証する（本イシューで負検査ハーネスを新規に作らない。既存テストが
  fail-closed 挙動を個別に実証済みのため）。

## 4. 監査手順（要約）

1. `docs/self-repair-log-format.md` 6 節に従い、対象 `loop-log.jsonl` に対し
   `self_repair::verify_chain(path)` を呼ぶ（Rust コードから、または
   `cargo test -p self-repair --test logging_chain` 等で間接的に）。
   `Err(LogError::ChainViolation { .. })` が返れば改竄・破損の疑いとして
   扱い、以降のログ内容を信頼しない（fail-closed）。
2. `verify_chain` を通過したログについて、`loop_start → detection →
   attempt ×n → loop_outcome`（正常終了）の順にレコードを読み、`attempt`
   ごとの `outcome.kind`/`outcome.reason` から判断根拠を時系列に復元する
   （`docs/self-repair-log-format.md` 4 節の対応表）。
3. `loop-report.json`（手動構築の JSON。試行回数・所要時間・ベンチ計測値・
   diff 由来シグナルを含む、より詳細な記録）と突き合わせ、`attempt_count`・
   `outcome` が一致することを確認する（本イシューでの再生成時に
   両ファイルの整合を確認済み。§1 のサマリ数値の出典）。
4. **検知できない改竄（末尾切り詰め）**: `verify_chain` 単体では検知できない
   （`docs/self-repair-log-format.md` 6 節 3）。外部アンカー運用（同仕様書
   7 節）は運用指針の文書化のみで自動化実装は行っていない。

## 5. 未充足・スコープ外事項の追跡先

| 事項 | 追跡先 |
|------|--------|
| `self-repair run`/`verify-log` CLI バイナリの実装 | `docs/guardrail-self-repair-cli.md` §3 記載のタスク（後続イシュー） |
| 候補 diff（バグ修正のワークツリー差分）に対するベンチ劣化率の直接実測・真の 4 ゲート合成の `src/` 本体への昇格 | #136 系 |
| 外部アンカー運用（末尾切り詰め対策）の自動化実装 | 未起票（`docs/self-repair-log-format.md` 7 節が運用指針のみを文書化。自動化が必要と判断された場合は別途 Issue 化） |
| 完走可否の人間評価 | #144（TASK-3.3e）。評価・判定案は [`completion-judgment.md`](./completion-judgment.md) に記録した |

## 6. #144（人間評価）向けの評価導線

1. 本 README §1 のサマリ・§2 の充足マトリクスで全体像を把握する。
2. 種別ごとの詳細判断根拠は各ディレクトリの README（
   [`bug-fix/README.md`](./bug-fix/README.md)・
   [`feature-addition/README.md`](./feature-addition/README.md)）の
   「実行結果サマリ」節・完走判定基準表を参照する。
3. 生データは `loop-report.json`（試行ごとの理由・診断値の全文）と
   `loop-log.jsonl`（ハッシュチェーン付き段階レコード。§4 の手順で
   `verify_chain` により整合性を確認可能）の両方を参照できる。
4. §2 の「部分充足」「未充足」区分（バグ修正: 判定基準 1・2・4・5・6、
   機能追加: 判定基準 1・4・5）がいずれもスコープ外事項（§5。CLI 未実装・
   4 ゲート合成の `src/` 本体未昇格・合成ワークロード限界等）に起因する
   ものであり、ガードレール閾値・テスト許容誤差の緩和によるものではない
   ことを、各 README の該当節（「3. 4 ゲート合成についての設計上の制約」・
   「シグナルは実測のみ」等）で確認する。
5. ログ整合性の事前確認（`verify_chain` 実行結果）・判定基準 6 項目の
   個別評価・REQ-3 受け入れ基準との突合・判定案は
   [`completion-judgment.md`](./completion-judgment.md) に記録した。
   人間判定は同文書 8 節の記録欄に行う。
