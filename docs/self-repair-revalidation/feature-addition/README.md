# 機能追加種別のループ完走実証（TASK-3.3c・イシュー #142）

`loop-report.json`・`loop-log.jsonl` はいずれも
`crates/self-repair/tests/feature_addition_loop_completion_task_3_3c.rs` の
実行によって**実測**された完走ログである（捏造・手組みではない。§「再現手順」
の環境変数 `SELF_REPAIR_TASK_3_3C_WRITE_DOCS=1` を明示指定して実行した場合の
み本ディレクトリへ上書きされる）。

## 題材・完走判定基準

REQ-3 の v2 追加受け入れ基準（`docs/spec/04-requirements.md:96`）「自作コアに
対する自己修復ループの人間介在なし完走を新実装リポで再実証する」を、PoC-2
検証題材 (c)（`leaky_relu` 新規実装）で 1 ループ実測したものである。題材選定・
完走判定基準は TASK-3.3a（#140・PR #322）で人間承認済み
（`docs/self-repair-revalidation-plan.md` §4.2）。

## CLI 経由完走との差分（重要・明示）

`docs/guardrail-self-repair-cli.md` §5.1 が定める「`self-repair run` CLI 経由の
完走」は、CLI バイナリ（`self-repair run`。未実装）に依存し、本イシュー
（#142）のスコープではない。

本実証は **lib API 直接呼び出しの実証ハーネス**（統合テスト）として
[`self_repair::SelfRepairLoop`] を 1 回起動し、`LoopReport` を JSON 化した
完走ログ（`loop-report.json`）をここへ記録する。§5.4「JSON Lines ログの
ハッシュチェーン検証」は TASK-3.4（#145）実装済みの `self_repair::LogWriter`/
`verify_chain` を TASK-3.3d（#143）で本ハーネスへ結線し、`loop-log.jsonl`
として同じディレクトリへ出力・検証している（外部コマンド `self-repair
verify-log` CLI は未実装のため、検証は lib 呼び出し経由。詳細は本 README
末尾「改竄検知ログ」節）。CLI 経由の再実施の充足は #144（人間評価）側で
判断する。

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
- `gaming_suspect`: (a) 変更ファイルパスに `tests/` 配下が含まれるか、
  (b) `TARGET_FILE`（`src/activations.rs`）の `git diff --unified=0` ハンクが
  baseline の `#[cfg(test)] mod tests` 境界行以降と重なるか、のいずれかを
  `changed_files`・diff ハンク行番号から実測して判定する（本ハーネスの候補
  生成は「テストファイル・テストモジュールへ触れない」よう構成されているが、
  `signal_source: "measured"` を名乗る以上ハードコードの `false` 固定では
  なく、実行のたび diff から導出する。`measure_signals_for_candidate2`・
  `diff_touches_boundary` 参照）。**既知の限界**: `mod tests` marker 直前
  への隣接挿入（本ハーネスが実際に候補を挿入する位置そのもの）は境界内と
  判定しない。既存テスト内容を一切変えず `mod tests` の直前へ新規
  `#[cfg(test)]` ブロックを丸ごと追加するような候補は、本判定では検知
  できない（`diff_touches_boundary` doc 参照）。
- ベンチ: `SelfRepairBenchGate::run`（5 回計測）→
  `guardrail::median_gate::bench_signal_from_measurements` で
  `guardrail::BenchSignal::Measured` へ変換。**既知の制約**: baseline・
  candidate 双方に同一の合成ワークロード（`leaky_relu_like_workload`。
  `verify_composite.rs` 参照）を渡しており、実際の `leaky_relu` 実装差分の
  性能特性そのものは計測していない（真の劣化率は構造的に 0% 近傍になる）。
  `bench_median_pct` は「ベンチゲートの実行経路が実測で機能すること」の
  実証であり、候補実装固有の性能劣化検出ではない点に留意する。

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

`loop-report.json` と同じ出力先へ改竄検知ログ `loop-log.jsonl` も併せて
書き出す（次節参照）。`target/` フォールバック・`docs/` 明示書き込みの
いずれでも、固定ファイル名 `loop-log.jsonl` を実行のたび削除してから
`LogWriter::open` するため、`loop-report.json` の `fs::write` 上書きと
同じく「このディレクトリはこの 1 回の実行を記述する」契約を保つ
（複数回の実行が同一チェーンへ継ぎ足されることはない）。

## 改竄検知ログ（`loop-log.jsonl`）

`loop-log.jsonl` は `docs/self-repair-log-format.md`（TASK-3.4・#145 の正式
仕様書）が定める JSON Lines・SHA-256 ハッシュチェーン形式である。監査手順・
フォーマット詳細は同仕様書 6 節を参照。本実証固有の要点のみ以下に示す。

- **段階列**: `loop_start`（`kind: "feature_addition"`）→ `detection` →
  `attempt`（試行 1: `verification_failed`）→ `attempt`（試行 2:
  `adopted`）→ `loop_outcome`（`outcome.kind: "adopted"`）の 5 レコード
  （`loop-report.json` の `attempt_count: 2` と対応）。
- **検証**: ハーネス自身が書き込み直後に `self_repair::verify_chain` を
  呼び、`Ok` であることを assert している
  （`crates/self-repair/tests/feature_addition_loop_completion_task_3_3c.rs`
  の `write_loop_report`）。第三者が独立に再検証する場合は同じ
  `verify_chain(path)` を呼べばよい（`self-repair verify-log` 外部コマンドは
  未実装のため、lib 呼び出しが現時点で唯一の検証手段）。
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

- CLI（`self-repair run`/`verify-log`）経由の完走・検証 → 後続タスク
  （`docs/guardrail-self-repair-cli.md` 記載のタスクで追跡済み）。ログ機構
  自体（`LogWriter`/`verify_chain`）は TASK-3.4（#145）で実装済み・
  TASK-3.3d（#143）で本ハーネスへ結線済み（下記「改竄検知ログ」節参照）
- 完走ログの最終的な記録様式・配置場所の確定 → イシュー #143（本更新で反映）
- `crates/self-repair/Cargo.toml` は本イシューでは変更していない（並行実装
  中のイシュー #141〈TASK-3.3b〉との編集衝突回避。本テストは
  `verify_bench`／`guardrail` 経由の既存依存のみで完結する）。
- `crates/self-repair/tests/fixtures/feature-addition-leaky-relu/baseline/
  Cargo.lock`（フィクスチャ隔離用の空 `[workspace]` テーブルを持つ独立
  crate。`crates/guardrail/tests/fixtures/labeled-changes/baseline/` と同じ
  隔離方針）はリポジトリルート外の `Cargo.lock` であり、`scripts/
  check-forbidden-deps.sh` の依存禁止検査（ルート `Cargo.lock` のみが対象）
  の走査範囲外にある。現状フィクスチャ側の依存は `tensor-core`／`autodiff`
  への path 依存のみで禁止リスト該当なし。
