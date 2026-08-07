# 機能追加種別のループ完走実証（TASK-3.3c・イシュー #142）

`loop-report.json` は
`crates/self-repair/tests/feature_addition_loop_completion_task_3_3c.rs` の
実行によって**実測**された完走ログである（捏造・手組みではない。試行のたび
に本ファイルへ上書きされる）。

## 題材・完走判定基準

REQ-3 の v2 追加受け入れ基準（`docs/spec/04-requirements.md:96`）「自作コアに
対する自己修復ループの人間介在なし完走を新実装リポで再実証する」を、PoC-2
検証題材 (c)（`leaky_relu` 新規実装）で 1 ループ実測したものである。題材選定・
完走判定基準は TASK-3.3a（#140・PR #322）で人間承認済み
（`docs/self-repair-revalidation-plan.md` §4.2）。

## CLI 経由完走との差分（重要・明示）

`docs/guardrail-self-repair-cli.md` §5.1 が定める「`self-repair run` CLI 経由の
完走」・§5.4「JSON Lines ログのハッシュチェーン検証」は、CLI バイナリ
（`self-repair run`。未実装）・ログ形式移植（TASK-3.4・イシュー #145。ハッシュ
チェーン等の改竄検知形式）の両方に依存し、本イシュー（#142）のスコープでは
ない。

本実証は **lib API 直接呼び出しの実証ハーネス**（統合テスト）として
[`self_repair::SelfRepairLoop`] を 1 回起動し、`LoopReport` を JSON 化した
完走ログをここへ記録する。CLI 経由・ハッシュチェーン検証の充足は #145 実装後
に #144（人間評価）側で判断する。本ディレクトリへの記録配置自体も #143 で
最終様式が確定するまでの暫定配置である。

## 保守対象・ループ構成

- 保守対象: `crates/self-repair/tests/fixtures/feature-addition-leaky-relu/baseline/`
  （実 `autodiff`/`tensor-core` への path 依存を持つ隔離サンドボックス crate。
  `crates/guardrail/tests/fixtures/labeled-changes/baseline/` と同じ空
  `[workspace]` テーブルによる隔離方針）。`leaky_relu` が未実装の baseline
  状態で固定されている。
- Detector: `FeatureAdditionDetector`（`cargo test --release` の失敗を検出）。
- FixGenerator: `FeatureAdditionFixGenerator`。候補 2 件（試行 1 = 符号分岐を
  欠く誤実装、試行 2 = 既存組み込み演算〈`relu`・四則〉の合成による正実装）。
- VerificationGate: `self_repair::verify_composite::FeatureAdditionCompositeGate`
  （TASK-3.3c で新設。build/test/clippy 3 ゲート
  〈`CargoVerificationGate`〉とベンチゲート〈`SelfRepairBenchGate`。
  bench-harness 経由・5 回計測中央値〉を合成し、全ゲート通過後に限りベンチを
  実測する）。
- AdoptionJudge: `GuardrailAdoptionJudge`。閾値はリポジトリルート
  `guardrail.toml`（TASK-4.3c 承認済み）を `guardrail::config::resolve` で
  読み込み、数値のハードコード・緩和は行わない。

## シグナルは実測のみ（`signal_source: "measured"`）

- `lines_changed`: sandbox 内の使い捨て git リポジトリ（`git init` した
  baseline コミット）に対する候補 2（採用された実装）の `git diff --numstat`
  実測値。
- `exclusion_rule_ids`: `guardrail::policy_exclusion::ExclusionEvaluation::evaluate`
  （組み込み既定ルール）の実行結果。
- `api_broken`: baseline の既存公開関数シグネチャがすべて候補内に保存されて
  いるかの実測比較。
- `gaming_suspect`: 候補がテストファイル（`tests/leaky_relu_acceptance.rs`・
  baseline の `mod tests`）を一切変更しないことに基づく（本ハーネスでは
  固定 `false`。候補生成そのものが「テストファイルへ触れない」よう構成
  されているため、この判断はハーネス設計上の不変条件であり diff 解析結果
  ではない）。
- ベンチ: `SelfRepairBenchGate::run`（5 回計測）→
  `guardrail::median_gate::bench_signal_from_measurements` で
  `guardrail::BenchSignal::Measured` へ変換。

未計測値を fail-open な既定値で埋めていない（`.claude/rules/security.md` A08）。

## 再現手順

```bash
cargo test -p self-repair --test feature_addition_loop_completion_task_3_3c -- --nocapture
```

実行のたび sandbox は一意な一時ディレクトリ
（`std::env::temp_dir()`／`self-repair-feature-addition-task-3-3c-sandbox-<pid>`）
に作られ、テスト終了時に削除される。

完走ログの書き出し先は既定で `target/self-repair-revalidation/feature-addition/
loop-report.json`（git 管理対象外）であり、通常の `cargo test` 実行では本
ディレクトリの `loop-report.json`（commit 済み）を上書きしない（sandbox の
PID・一時パスを含む `log_tail` が実行のたび変わり、無関係な tracked diff が
毎回発生するのを避けるため）。commit 済みの記録を更新する場合のみ、環境変数
を明示指定して実行する:

```bash
SELF_REPAIR_TASK_3_3C_WRITE_DOCS=1 \
  cargo test -p self-repair --test feature_addition_loop_completion_task_3_3c
```

## 対象外（既存イシューで追跡）

- CLI（`self-repair run`）経由の完走 → 後続タスク（`docs/guardrail-self-repair-cli.md`
  記載のタスクで追跡済み）
- JSON Lines ログのハッシュチェーン検証（改竄検知形式） → イシュー #145
  （TASK-3.4）
- 完走ログの最終的な記録様式・配置場所の確定 → イシュー #143
- `crates/self-repair/Cargo.toml` は本イシューでは変更していない（並行実装
  中のイシュー #141〈TASK-3.3b〉との編集衝突回避。本テストは
  `verify_bench`／`guardrail` 経由の既存依存のみで完結する）。
