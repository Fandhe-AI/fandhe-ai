# 機能追加種別のループ完走実証（TASK-3.3c・イシュー #142）

`loop-report.json`・`loop-log.jsonl` はいずれも
`crates/self-repair/tests/feature_addition_loop_completion_task_3_3c.rs` の
実行によって**実測**された結果である（捏造・手組みではない。§「再現手順」の
環境変数 `SELF_REPAIR_TASK_3_3C_WRITE_DOCS=1` を明示指定して実行した場合のみ
本ディレクトリへ上書きされる）。

## 題材・完走判定基準

REQ-3 の v2 追加受け入れ基準（`docs/spec/04-requirements.md:96`）「自作コアに
対する自己修復ループの人間介在なし完走を新実装リポで再実証する」を、PoC-2
検証題材 (c)（`leaky_relu` 新規実装）で 1 ループ実測したものである。題材選定・
完走判定基準は TASK-3.3a（#140・PR #322）で人間承認済み
（`docs/self-repair-revalidation-plan.md` §4.2）。完走判定基準 6 項目は同文書
§5 参照。

## 完走判定基準の充足状況（実測。all-or-nothing 規則。#139 判断 (c) 準拠）

| 基準 | 内容 | 充足 | 備考 |
|---|---|---|---|
| 1 | `self-repair run` 1 回起動・追加の人間入力なしで exit 0（`Verdict::AutoApply`） | **未充足** | 実測 outcome=`Escalated`・exit=10。原因は下記「基準 1 が未充足の原因」参照 |
| 2 | 検証 4 ゲート全通過・guardrail 3 分岐判定を lib 直接呼び出しで経由 | 充足 | 試行 2 は build/test/clippy/bench 全ゲート通過（`gate_report`）。判定は `GuardrailAdoptionJudge` → `guardrail::decide` の単一経路のみ |
| 3 | `--max-attempts` 上限内で完走 | 充足 | 2 試行（上限 5 以内） |
| 4 | `self-repair verify-log` のハッシュチェーン検証を通過 | 充足 | 外部コマンド経由（子プロセス起動）で exit 0 を確認 |
| 5 | ベンチ劣化中央値が承認済み閾値内・5 回計測中央値・**候補 diff 直接実測** | 部分充足 | `RepairCompositeGate`（#137）による直接実測は充足。ただし試行 2 が最終的に Escalated のため「完走」の一部としては成立していない |
| 6 | `signal_source` が `"measured"` | 充足 | `--signals` 契約検証パスは未使用 |

**基準 1 が未充足のため、all-or-nothing 規則（#140 承認済み）上「人間介在なし
完走」とは認められない。** #144（人間評価・completion-judgment 再判定）へ
この結果をそのまま引き継ぐ。

### 基準 1 が未充足の原因（実装・実測で判明。ガードレール判定は変更していない）

`crates/self-repair/src/diff_signals.rs::api_signature_touched` は追加・削除
いずれの `pub fn` 行も「API 破壊」として検出するヒューリスティックであり
（`tests/verify_direct_composite_integration.rs::
case_a_harmless_candidate_diff_completes_with_measured_bench` の doc に
「新規 pub fn 追加はヒューリスティック上検出される想定」と既に明記されている
既存仕様。TASK-3.2a・#137 由来）、`crates/self-repair/src/judge.rs` は
`api_broken=true` を無条件で `Escalated` へ写像する
（`judge::tests::api_broken_yields_escalate`）。PoC-2 題材 (c) の受け入れ
基準（`tests/leaky_relu_acceptance.rs`）は `pub fn leaky_relu` の追加を
要求するため、この構成では機能追加種別の候補が自動適用（`Adopted`・exit 0）
へ到達する経路が存在しない。

ヒューリスティックの精緻化（新規追加と既存シグネチャの削除を区別する等）・
`policy-exclusion.toml` への例外追加はいずれもガードレール判定・除外リストの
変更でありユーザー承認必須（`.claude/rules/security.md`）のため、本イシュー
（#142）では変更していない。対応方針（ヒューリスティックの見直しを提案する／
題材を見直す／基準 1 の解釈を見直す 等）はユーザー判断事項として #142・#144
側へ引き継ぐ。

## CLI 経由への移行（#139 判断 (c) 差し戻し対応）

旧版（PR #338）は **lib API 直接呼び出しの実証ハーネス**として
`self_repair::SelfRepairLoop` を直接構築し、diff 由来シグナルは
`FeatureAdditionCompositeGate`（構築時固定・合成ベンチワークロード）を使って
いた。#139 reopen コメントの判断 (c) 差し戻しにより、以下へ移行した
（実装計画 #142 §3）:

- `self-repair run` CLI（3.1 節。本イシューで実装。`crates/self-repair/src/
  cli.rs`・`main.rs`）を **1 回だけ子プロセス起動**する（基準 1 の「1 回
  起動」要件）
- `self-repair verify-log` CLI（3.2 節。#145 実装済み）を別プロセスとして
  起動し、ハッシュチェーンを検証する（基準 4）
- 検証ゲートは `RepairCompositeGate`（TASK-3.2a・#137）を使い、diff 由来
  4 シグナルを試行ごとに実測し直し、ベンチは baseline commit と候補適用済み
  sandbox 双方の release ビルドを比較する「候補 diff 直接実測」
  （`DirectBenchRunner`）を用いる（基準 5）

## 保守対象・ループ構成

- 保守対象: `crates/self-repair/tests/fixtures/feature-addition-leaky-relu/baseline/`
  （実 `autodiff`/`tensor-core` への path 依存を持つ隔離サンドボックス crate。
  `crates/guardrail/tests/fixtures/labeled-changes/baseline/` と同じ空
  `[workspace]` テーブルによる隔離方針）。`leaky_relu` が未実装の baseline
  状態で固定されている。
- `--kind feature-addition`: `FeatureAdditionDetector`（`cargo test --release`
  の失敗を検出）・`FeatureAdditionFixGenerator`（候補 2 件。試行 1 = 符号
  分岐を欠く誤実装、試行 2 = 既存組み込み演算〈`relu`・四則〉の合成による
  正実装）。
- VerificationGate: `RepairCompositeGate`（`--bench-bin bench_workload`・
  `--workload-source src/bin/bench_workload.rs`）。build/test/clippy 3 ゲート
  〈`CargoVerificationGate`〉と候補 diff 直接ベンチ実測〈`DirectBenchRunner`〉
  を合成し、全ゲート通過後に限りベンチを実測する。
- AdoptionJudge: `GuardrailAdoptionJudge`。閾値はリポジトリルート
  `guardrail.toml`（TASK-4.3c 承認済み）を `guardrail::config::resolve` で
  読み込み、数値のハードコード・緩和は行わない。

## シグナルは実測のみ（`signal_source: "measured"`）

いずれも CLI バイナリ内部（`RepairCompositeGate::verify` → `crate::
diff_signals::measure_diff_signals`／`crate::verify_bench_direct::
DirectBenchRunner`）が試行ごとに sandbox 内の使い捨て git リポジトリを対象に
実測する。ハーネス自身は事前計算しない（旧版との差分）。

- `lines_changed`: `git diff --numstat` 実測値。
- `exclusion_rule_ids`: `guardrail::policy_exclusion::ExclusionEvaluation::evaluate`
  （リポジトリルート `policy-exclusion.toml`）の実行結果。
- `api_broken`: `git diff -U0` 中の追加・削除行に `pub fn` を含むかの実測
  （「基準 1 が未充足の原因」参照。既存シグネチャの削除だけでなく新規追加も
  検出する設計）。
- `gaming_suspect`: 変更ファイルパスに `tests/` 配下が含まれるかの実測
  （`gaming_suspect_from_files`）。
- ベンチ: `DirectBenchRunner::measure`（baseline commit の
  `git worktree add --detach` と候補適用済み sandbox の双方を release ビルド
  し外部タイミングで比較。5 回以上計測）→
  `guardrail::median_gate::bench_signal_from_measurements` で
  `guardrail::BenchSignal::Measured` へ変換。旧版（`FeatureAdditionCompositeGate`。
  baseline/candidate 双方に同一の合成ワークロードを使用）と異なり、実際の
  実装差分に固有の性能特性を計測する。

未計測値を fail-open な既定値で埋めていない（`.claude/rules/security.md` A08）。

## 再現手順

```bash
cargo test -p self-repair --test feature_addition_loop_completion_task_3_3c -- --ignored --nocapture
```

本テストは基準 1 が未充足であること自体を固定する実証のため `#[ignore]` で
分離している（通常 CI では実行しない。理由は `#[ignore = "..."]` の属性
文字列・テスト関数のドキュメンテーションコメント参照）。実行のたび sandbox
は一意な一時ディレクトリ（`std::env::temp_dir()`／
`self-repair-feature-addition-task-3-3c-sandbox-<pid>`）に作られ、テスト
終了時に削除される。

結果の書き出し先は既定で `target/self-repair-revalidation/feature-addition/`
（git 管理対象外）であり、通常の実行では本ディレクトリの commit 済み
`loop-report.json`/`loop-log.jsonl` を上書きしない（sandbox の PID・一時パス
が実行のたび変わり、無関係な tracked diff が毎回発生するのを避けるため）。
commit 済みの記録を更新する場合のみ、環境変数を明示指定して実行する:

```bash
SELF_REPAIR_TASK_3_3C_WRITE_DOCS=1 \
  cargo test -p self-repair --test feature_addition_loop_completion_task_3_3c -- --ignored --nocapture
```

`self-repair run` は 1 回しか起動しない（基準 1 の「1 回起動」を docs 反映の
ためだけに破らない）。環境変数指定時は `target/` へ書き出した実行結果の
`loop-report.json`/`loop-log.jsonl` をそのまま `docs/` へ複製する（再実行
しない）。`--log`/`--output`/`--policy-exclusion` はリポジトリルート相対
パスへ変換して記録するが、`--repo`（sandbox）・`--candidates`（一時ファイル）
は実行のたび変わる絶対パスのままである（既知の制約）。

## 改竄検知ログ（`loop-log.jsonl`）

`loop-log.jsonl` は `docs/self-repair-log-format.md`（TASK-3.4・#145 の正式
仕様書）が定める JSON Lines・SHA-256 ハッシュチェーン形式である。監査手順・
フォーマット詳細は同仕様書 6 節を参照。本実証固有の要点のみ以下に示す。

- **段階列**: `loop_start`（`kind: "feature_addition"`）→ `detection` →
  `attempt`（試行 1: `verification_failed`）→ `attempt`（試行 2:
  `escalated`）→ `loop_outcome`（`outcome.kind: "escalated"`）の 5 レコード
  （`loop-report.json` の `attempt_count: 2` と対応）。
- **検証**: `self-repair run` バイナリ自身が書き込み直後に
  `self_repair::verify_chain` で自己検証したうえで、本ハーネスが
  `self-repair verify-log` を**外部コマンドとして別プロセス起動**し exit 0
  であることを assert している（基準 4。旧版は lib 呼び出しのみだった）。
- **改竄検知の実効性**: `verify_chain` がフィールド改変・レコード削除・
  順序入れ替え・未知フィールド注入をいずれも検知することは
  `crates/self-repair/src/logging.rs` の単体テスト
  （`verify_chain_detects_stage_field_tampering` 等。詳細は
  `docs/self-repair-revalidation/bug-fix/README.md` §8 の一覧を参照）で
  個別に実証済みであり、本ハーネスで同種の負検査を作り直すことはしない。
- **末尾切り詰めの限界**: `verify_chain` 単体では検知できない
  （`docs/self-repair-log-format.md` 6 節 3）。外部アンカー運用（同仕様書
  7 節）は運用指針の文書化のみで自動化実装は行っていない
  （out-of-scope-tracking.md 準拠）。

## 対象外（既存イシューで追跡）

- 完走ログの最終的な記録様式・配置場所の確定 → イシュー #143
- 基準 1 未充足への対応方針の確定（ヒューリスティック見直し・題材見直し・
  基準解釈の見直しのいずれか） → #144（人間評価）・ユーザー判断待ち。
  `.claude/rules/out-of-scope-tracking.md` 準拠で必要なら別途 Issue 起票する
- `--kind perf-regression` は `self-repair run` 未対応（`PerfRegressionDetector`/
  `PerfRegressionFixGenerator` が `CommandRunner` ベースの検出器と非対称な
  構築契約〈`BenchMeasurer`・戦略リスト〉を持つため。#141／#142 いずれも
  本種別を必要としない）
- `crates/self-repair/tests/fixtures/feature-addition-leaky-relu/baseline/
  Cargo.lock`（フィクスチャ隔離用の空 `[workspace]` テーブルを持つ独立
  crate。`crates/guardrail/tests/fixtures/labeled-changes/baseline/` と同じ
  隔離方針）はリポジトリルート外の `Cargo.lock` であり、`scripts/
  check-forbidden-deps.sh` の依存禁止検査（ルート `Cargo.lock` のみが対象）
  の走査範囲外にある。現状フィクスチャ側の依存は `tensor-core`／`autodiff`
  への path 依存のみで禁止リスト該当なし。
