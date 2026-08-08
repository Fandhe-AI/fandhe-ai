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

> **注記（TASK-3.3d・本イシューでの更新）**: 両種別とも `self-repair run`
> CLI 経由・候補 diff 直接実測（`RepairCompositeGate`・TASK-3.2a・#137）へ
> 移行が完了している（バグ修正: PR #372・#141 再実証。機能追加: PR #361・
> #142 再調査）。以下 §1〜§6 は移行後の現行証跡（各 `loop-report.json`・
> [`bug-fix/README.md`](./bug-fix/README.md)・
> [`feature-addition/README.md`](./feature-addition/README.md)）を正として
> 記載する。旧版（lib 直接呼び出し・合成ワークロード時点）の記述は本更新で
> 置き換えた。§3〜§8 の全面再評価（all-or-nothing 規則の再適用を含む）は
> 引き続き TASK-3.3e（#144）側で行う。

## 1. サマリ

| 種別 | 記録先 | 最終結論 | 試行回数 | 合計所要時間 | 判断根拠（要約） |
|------|--------|---------|---------|-------------|-----------------|
| バグ修正（TASK-3.3b・#141） | [`bug-fix/`](./bug-fix/) | `Adopted`（exit 0） | 2 | 40,323 ms（`loop-report.json` `total_duration_ms`。ハーネス全体の壁時計時間は `harness_wall_time_ms` 40,534 ms） | attempt 1: `cargo test --release` の既知正解値テスト（`mlp_grad_*_matches_numeric`）が analytic/numeric 勾配不一致で失敗し却下。attempt 2: build/test/clippy 全通過＋候補 diff（`var.rs` の relu 実装復元）直接ベンチ実測（劣化率中央値 **-74.86%**。誤って混入させた sigmoid すり替えバグ比での高速化であり、性能改善そのものが目的ではない）＋diff 由来シグナル（`lines_changed=2`・`api_broken=false`・`gaming_suspect=false`）が `guardrail::decide` の自動適用条件を満たし採用 |
| 機能追加（TASK-3.3c・#142） | [`feature-addition/`](./feature-addition/) | `Adopted`（exit 0） | 2 | 45,290 ms（`loop-report.json` `total_duration_ms`。ハーネス全体の壁時計時間は `harness_wall_time_ms` 45,322 ms） | attempt 1: 符号分岐を欠く誤実装で受け入れ基準テスト（`leaky_relu_matches_known_values`。`got=0.05, want=0.5`）不合格・却下。attempt 2: 既存組み込み演算（`relu`・四則）合成による正実装で build/test/clippy 全通過＋候補 diff 直接ベンチ実測（劣化率中央値 0.49%）＋diff 由来シグナル（`lines_changed=28`・`api_broken=false`・`gaming_suspect=false`）が自動適用条件を満たし採用 |

両種別とも `guardrail.toml`（TASK-4.3c・#117 確定値。`bench_median_max_pct=5.0`・
`bench_runs_min=5`・`lines_max=200`）を一切変更せず、`guardrail::decide` を
唯一の取り込み判断経路として使用した（迂回経路なし）。数値の出典は各
`loop-report.json`（`docs/self-repair-revalidation/bug-fix/loop-report.json`・
`docs/self-repair-revalidation/feature-addition/loop-report.json`）。

## 2. 完走判定基準の充足マトリクス

実証計画（`docs/self-repair-revalidation-plan.md` 5 節）が定める判定基準
6 項目に対する、2 種別の充足状況を統合する。両種別とも各個別 README の
判定表からそのまま転記した（バグ修正: `bug-fix/README.md` §4。機能追加:
`feature-addition/README.md`「完走判定基準の充足状況」節。過大表現を避ける
ため、両種別とも実際の実測範囲を超えて「充足」と判定しない）。

| # | 判定基準 | バグ修正（#141） | 機能追加（#142） |
|---|---------|------------------|------------------|
| 1 | `self-repair run` の 1 回起動・追加の人間入力なしで `AutoApply` へ到達 | **充足**（CLI バイナリ〈PR #361〉を 1 回起動し `outcome=Adopted`・exit 0 へ到達。本ハーネス自体の CLI 移行は PR #372） | **充足**（CLI バイナリ〈PR #361〉を 1 回起動し `outcome=Adopted`・exit 0 へ到達） |
| 2 | 検証 4 ゲート全通過・`guardrail` 3 分岐判定を迂回なく経由 | **充足**（`RepairCompositeGate` が候補 diff〈`var.rs`〉に対し build/test/clippy＋直接ベンチを実測し `gate_report="build=pass test=pass clippy=pass bench=measured-direct"`。`guardrail::decide` を唯一の判定経路として使用） | **充足**（同一機構〈`RepairCompositeGate`〉で build/test/clippy/bench 全ゲート通過。判定は `GuardrailAdoptionJudge` → `guardrail::decide` の単一経路のみ） |
| 3 | `--max-attempts` 上限内で完走 | 充足（`max_attempts=5` のうち attempt 2 で採用。`attempt_count=2`） | 充足（`max_attempts=5` のうち attempt 2 で採用。`attempt_count=2`） |
| 4 | JSON Lines ログのハッシュチェーン検証（`self-repair verify-log`）を通過 | **充足**（`self-repair verify-log` を外部コマンドとして別プロセス起動し exit 0・`records=5, last_seq=4` を確認） | **充足**（同じく外部コマンド経由〈子プロセス起動〉で exit 0 を確認） |
| 5 | ベンチ劣化中央値が承認済み閾値内（5 回計測中央値・候補 diff 直接実測） | **充足**（`DirectBenchRunner` が候補 diff そのものを直接計測。中央値 -74.86%〈`bench_median_pct=-74.85729651466627`。負値は改善方向。§1 参照〉< `guardrail.toml` の `bench_median_max_pct=5.0`） | **充足**（同一機構で直接実測。中央値 0.49%〈`bench_median_pct=0.4876067129737649`〉< `bench_median_max_pct=5.0`） |
| 6 | 判定レポート JSON の `signal_source` が `"measured"` | 充足（`loop-report.json.signal_source == "measured"`） | 充足（`loop-report.json.signal_source == "measured"`） |

両種別とも全 6 項目「充足」に到達している。§3 では判定基準 4 の充足範囲
（過大表現をしない限界の明記）を、§5 では検証スコープ・末尾切り詰め等の
残存限界を正直に記載する。

## 3. 判定基準 4 の充足範囲（過大表現をしない）

- **充足していること**: TASK-3.4（#145）が実装した
  `self_repair::LogWriter::append_report`/`append_failure` を両ハーネスへ
  結線し、ループ実行のたび `loop-log.jsonl`（JSON Lines・SHA-256 ハッシュ
  チェーン。`docs/self-repair-log-format.md` 準拠）を出力した。両種別とも
  `self-repair verify-log`（`docs/guardrail-self-repair-cli.md` §3.2 が
  境界を定める外部コマンド）を独立した別プロセスとして起動し、exit 0 を
  確認済みである（バグ修正: `bug-fix/README.md` §8。機能追加:
  `feature-addition/README.md`「改竄検知ログ」節）。
- **なお残る限界**: `verify_chain` 単体では末尾切り詰め（ログ末尾のレコード
  削除で正常終了に見せかける改竄）を検知できない
  （`docs/self-repair-log-format.md` 6 節 3）。外部アンカー運用（同仕様書
  7 節）は運用指針の文書化のみで自動化実装は行っていない（§5 参照）。
- 監査手順・改竄検知の実効性（フィールド改変・レコード削除・順序入れ替え・
  未知フィールド注入の検知）は
  [`bug-fix/README.md` §8](./bug-fix/README.md#8-改竄検知ログloop-logjsonlの監査手順) /
  [`feature-addition/README.md`「改竄検知ログ」節](./feature-addition/README.md#改竄検知ログloop-logjsonl)
  に詳細を記載し、`crates/self-repair/src/logging.rs` の既存単体テストへの
  参照で実証する（本イシューで負検査ハーネスを新規に作らない。既存テストが
  fail-closed 挙動を個別に実証済みのため）。

## 4. 監査手順（要約）

1. **`self-repair verify-log` CLI を主手段とする**（第三者再検証手順として
   `bug-fix/README.md` §8 と整合）。監査担当者は
   `cargo run -p self-repair -- verify-log --log docs/self-repair-revalidation/<種別>/loop-log.jsonl`
   を独立プロセスとして起動し、`OK: ... records=N, last_seq=M` を確認する。
   非 0 終了コードは改竄・破損の疑いとして扱い、以降のログ内容を信頼しない
   （fail-closed）。lib `self_repair::verify_chain(path)` の直接呼び出しは
   代替手段として残す（`Err(LogError::ChainViolation { .. })` を検知条件と
   する点は CLI と同一）。
2. 検証を通過したログについて、`loop_start → detection → attempt ×n →
   loop_outcome`（正常終了）の順にレコードを読み、`attempt` ごとの
   `outcome.kind`/`outcome.reason` から判断根拠を時系列に復元する
   （`docs/self-repair-log-format.md` 4 節の対応表）。
3. `loop-report.json`（手動構築の JSON。試行回数・所要時間・ベンチ計測値・
   diff 由来シグナルを含む、より詳細な記録）と突き合わせ、`attempt_count`・
   `outcome` が一致することを確認する。本イシューでの再検証時に確認済みの
   値: バグ修正 5 レコード・`attempt_count=2`・`last_seq=4`、機能追加
   5 レコード・`attempt_count=2`・`last_seq=4`（§1 のサマリ数値の出典）。
4. **検知できない改竄（末尾切り詰め）**: `verify_chain` 単体では検知できない
   （`docs/self-repair-log-format.md` 6 節 3）。外部アンカー運用（同仕様書
   7 節）は運用指針の文書化のみで自動化実装は行っていない。

## 5. スコープ外事項の追跡先

| 事項 | 追跡先・現状 |
|------|-------------|
| `self-repair run`/`verify-log` CLI バイナリの実装 | **解消済み**（PR #356・#361）。`docs/guardrail-self-repair-cli.md` §3 が正式仕様 |
| 候補 diff に対するベンチ劣化率の直接実測・真の 4 ゲート合成の `src/` 本体への昇格 | **解消済み**（#137・PR #355。`RepairCompositeGate` が両種別で候補 diff 直接ベンチを実施） |
| 外部アンカー運用（末尾切り詰め対策）の自動化実装 | 未起票のまま（`docs/self-repair-log-format.md` 7 節が運用指針のみを文書化。自動化が必要と判断された場合は別途 Issue 化） |
| 実 workspace 全体を対象にした検証 | #139（TASK-3.3 本体）の別スコープ。検証対象は `crates/autodiff` 単体クレート（実行時間の理由。`bug-fix/README.md` §3 と同一スコーピング判断） |
| 完走可否の人間評価 | #144（TASK-3.3e）。全面再評価済み（判定基準 6 項目・両種別とも「充足」）。評価・判定案は [`completion-judgment.md`](./completion-judgment.md) に記録し、最終判定は同文書 8.2 節に人間が記入する |

## 6. #144（人間評価）向けの評価導線

1. 本 README §1 のサマリ・§2 の充足マトリクスで全体像を把握する。
2. 種別ごとの詳細判断根拠は各ディレクトリの README（
   [`bug-fix/README.md`](./bug-fix/README.md)・
   [`feature-addition/README.md`](./feature-addition/README.md)）の
   「実行結果サマリ」節・完走判定基準表を参照する。
3. 生データは `loop-report.json`（試行ごとの理由・診断値の全文）と
   `loop-log.jsonl`（ハッシュチェーン付き段階レコード。§4 の手順で
   `verify-log` により整合性を確認可能）の両方を参照できる。
4. §2 のとおり両種別とも判定基準 6 項目すべて「充足」に到達している。
   ただし検証スコープの限定（`crates/autodiff` 単体クレート。実 workspace
   全体は対象外）・`verify_chain` の末尾切り詰め検知限界・外部アンカー
   運用の未自動化という残存限界がある（§3・§5）。これらはガードレール
   閾値・テスト許容誤差の緩和によるものではないことを、各 README の該当節
   （`bug-fix/README.md` §3・`feature-addition/README.md`「シグナルは実測
   のみ」等）で確認する。
5. ログ整合性の事前確認（`verify-log` 実行結果）・判定基準 6 項目の
   個別評価・REQ-3 受け入れ基準との突合・AI 起草判定案は
   [`completion-judgment.md`](./completion-judgment.md) に記録した。
   同文書 §3〜§8 の全面再評価（all-or-nothing 規則の再適用を含む）は
   #144 側で完了しており、一次結果は「充足」である（同文書 5 節）。
   人間による最終判定は同文書 8.2 節の記録欄に行う。
