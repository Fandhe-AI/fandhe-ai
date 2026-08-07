# guardrail 判定器変更時フロー（TASK-6.2）

`crates/guardrail` の判定ロジック（閾値・検知ルール・除外リスト評価等）を変更する際に必ず通す運用ルールを定める。TASK-6.1（#146〜#148・#199）で検証機構は整備済みであり、本ドキュメントはその**運用フロー**を明文化する（検証機構自体の再実装・再定義は行わない）。

v1 リポジトリ（`Fandhe-AI/rust-ai-library-v1`）の `docs/guardrail-change-policy.md` を移植したものだが、v2 は検証機構が全面的に刷新されているため、参照先はすべて v2 の実体に更新している。

## 1. 目的・根拠

- `docs/spec/04-requirements.md` REQ-6 の受け入れ基準 2 項目め（L144）: 「判定器の実装変更（閾値ロジック・検知ルールの追加変更）は、REQ-4/REQ-5 の受け入れ基準（見逃し率 0%・誤検知率 30% 以下）を満たすことを確認してから反映すること」
- v1 PoC-3 での発見事項（baseline タグの取り違えにより誤った合格判定が出ていたバグ）が示す通り、**判定器自体も検証対象である**。判定器を変更した PR がその判定器自身の検証をすり抜けて反映される事態を防ぐ
- `.claude/rules/security.md` A08（ソフトウェア・データ整合性）: 「判定の迂回経路を作らない」。本ドキュメントはこの原則の運用実装にあたる
- v2 では REQ-4（1 層目・判定器単体）と REQ-5（2 層目・除外リスト適用後）の**2 層検証**が求められる（`docs/spec/04-requirements.md` REQ-6 の 2026-08-05 追加基準）。1 層のみの確認では REQ-6 の受け入れ基準を満たさない

## 2. 前提: TASK-6.1 で整備済みの検証機構

以下はいずれも TASK-6.1（#146〜#148・#199）で整備済み。本ドキュメントはこれらを**参照**し、値を二重管理しない。

| 項目 | 実体 |
|------|------|
| ラベル付き回帰テストセット | `crates/guardrail/tests/fixtures/labeled-changes/`（15 件固定・safe 5／dangerous 5／gray 5。件数の根拠は同ディレクトリの `README.md`） |
| 2 層検証テスト（1 層目・REQ-4） | `scripts/run-guardrail-regression.sh` の `LAYER1_TESTS`（`eval_harness`・`labeled_changes_labels`） |
| 2 層検証テスト（2 層目・REQ-5） | 同スクリプトの `LAYER2_TESTS`（`label_invariant_empty_exclusions`・`label_invariant_safe_side_monotonicity`・`blindspot_g2_regression`・`blindspot_g5_regression`） |
| `guardrail eval` CLI | 1 層目の検証を担う。終了コード契約は `docs/guardrail-self-repair-cli.md` 1.3 節・2.3 節を参照 |
| 合格基準の正本 | `crates/guardrail/src/eval/mod.rs` の `MISS_RATE_MAX_PCT`（見逃し率上限）・`FALSE_POSITIVE_RATE_MAX_PCT`（誤検知率上限）。**数値は本ドキュメントに転記しない**（正本を直接参照する） |
| ローカル再現 | `make guardrail-regression`（`Makefile` L228-234。self-test → layer1 → layer2 の順で実行） |
| PR/push 契機の CI | `.github/workflows/ci.yml` の `guardrail-regression` ジョブ（`scripts/run-guardrail-regression.sh self-test` 等を実行し、`ci-complete` 集約ジョブの必須項目に含まれる） |
| 定期実行・失敗の可視化 | `.github/workflows/guardrail-regression-schedule.yml`（毎日 20:30 UTC 実行。report ジョブが失敗時に Issue を起票し、監視経路を欠落させない） |
| 検証ゲート | `ci.yml` の `verification-gates` ジョブ（build/test/clippy）＋ `.github/workflows/verification-gate-bench.yml`（bench。`scripts/run-verification-gates.sh`） |

## 3. 適用範囲（何が「判定器の実装変更」か）

以下のいずれかを変更する PR は本ドキュメントのフロー（4 節）を必ず通す。

- `crates/guardrail/src/` 配下（判定ロジック・閾値定数・検知ルール・除外リスト評価処理）
- `guardrail.toml`・`policy-exclusion.toml`（判定器の設定ファイル。形式・運用フローは `docs/policy-exclusion-design.md` を参照）
- 回帰データセット（`crates/guardrail/tests/fixtures/labeled-changes/` 配下。件数・ラベル・patch の追加変更）
- 2 層検証テスト自体（`LAYER1_TESTS`／`LAYER2_TESTS` が指すテストファイルの内容変更）
- `scripts/run-guardrail-regression.sh`（`LAYER1_TESTS`／`LAYER2_TESTS` リストを含む）
- 関連 workflow（`ci.yml` の `guardrail-regression` ジョブ・`guardrail-regression-schedule.yml`）

対象外の例: `docs/guardrail-self-repair-cli.md` 等のドキュメントのみの変更（判定ロジックに影響しない）、`crates/self-repair` のうち guardrail 判定結果を消費するだけで判定ロジックを含まない箇所。判定ロジックへの影響有無の判断に迷う場合は安全側（対象内）で扱う。

## 4. 変更時の必須確認フロー

### (1) 2 層検証の通過確認

`ci.yml` の `guardrail-regression` ジョブは PR/push 契機で自動実行され、`ci-complete` 集約ジョブが fail-closed で判定する（v1 は schedule のみで PR では自動実行されなかったが、v2 は運用が変わっている点に注意）。CI green を必須とし、ローカルでは `make guardrail-regression` で事前再現することを推奨する。

### (2) テスト追加時の LAYER リスト追記運用

`scripts/run-guardrail-regression.sh` の `self-test` サブコマンドは、`LAYER1_TESTS`／`LAYER2_TESTS` が指すテストファイル・fixture README の存在を fail-closed で検証する（テストの無言削除・リネームによる検証の空洞化を検知する設計）。この仕組み上、リストへの追記漏れはテスト自体が実行されないまま気づかれない状態を生む。

- 1 層目（REQ-4・判定器単体）のテストを追加する場合は `LAYER1_TESTS` へ追記する
- 2 層目（REQ-5・除外リスト適用後・本番相当経路）のテストを追加する場合は `LAYER2_TESTS` へ追記する
- 追記は対象テスト追加と**同一 PR**で行う（`scripts/run-guardrail-regression.sh` 冒頭コメントが本節へ委任している運用ルールの受け皿）

### (3) レビュー体制

`reviewer` によるレビューに加え、`.claude/rules/security.md`「レビュー体制」節・`.claude/rules/delegation-impl.md` の方針に従い、guardrail 変更を含む PR は `security-auditor` による並列監査を実施する。

### (4) 承認が必要な変更

以下は `.claude/rules/security.md`「自己修復ループ固有のガードレール」・`.claude/rules/delegation-impl.md`「禁止事項」に定める通り、必ず人間（ユーザー）の承認を経る。

- ガードレール閾値・テスト許容誤差・ポリシー除外リストの変更
- 閾値を変更する場合は、`docs/guardrail-threshold-recalibration.md` に定める再評価手順（ラベル付き変更セットでの見逃し率・誤検知率の実測）を経ること（REQ-4 の 2026-08-05 追加基準）

## 5. 禁止事項（fail-closed）

- eval ゲート・2 層検証の self-test を迂回する経路を新設しない
- ラベル付きデータセットの縮小（15 件未満、または各カテゴリ 5 件未満）や blindspot ケース（`blindspot_g2_regression`・`blindspot_g5_regression`）の削除により見かけ上の合格を作らない。データセットの縮小・カテゴリ変更はユーザー承認必須（4 節 (4)）
- 判定器の実装変更と、判定基準を緩和する方向のデータセット変更を同一 PR に混在させない（変更の影響を切り分け可能にするため）
- `MISS_RATE_MAX_PCT`・`FALSE_POSITIVE_RATE_MAX_PCT`（`crates/guardrail/src/eval/mod.rs`）等の合格基準定数の変更は、ガードレール閾値の変更としてユーザー承認必須（4 節 (4)）

## 6. 関連ドキュメント

- `docs/guardrail-self-repair-cli.md`（CLI 仕様・終了コード契約）
- `docs/policy-exclusion-design.md`（除外リスト設定形式・運用フロー）
- `docs/guardrail-threshold-recalibration.md`（閾値再キャリブレーション手順）
- `crates/guardrail/tests/fixtures/labeled-changes/README.md`（2 層ラベルモデル・データセット定義）
- `CLAUDE.md`（委譲マッピング・Conventions）
- `.claude/rules/security.md`（OWASP Top 10・自己修復ループのガードレール）
- `.claude/rules/delegation-impl.md`（実装 Agent への禁止事項）

## スコープ外（out-of-scope-tracking.md 対応）

- `guardrail check` 自身が 4 ゲートを起動して baseline 比較を行う結線（#103 残スコープ・TASK-8.2）は本ドキュメントの対象外（既存 Issue で追跡済み）
