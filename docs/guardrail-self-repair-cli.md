# guardrail／self-repair CLI コマンド仕様

イシュー #183（親 #181）の成果物。`docs/spec/04-requirements.md`
「画面・インターフェース要件」（L294-298、rust-ai-library-spec リポジトリ）が
「具体的な CLI コマンド仕様は Phase 5（タスク分解後の実装リポ側）で定める」と
明示委譲している事項を、TASK-4.1（`guardrail` 移植・#101 系）・TASK-3.1
（`self-repair` 骨格移植・#131）・TASK-6.1（CI 常設化・#146）の実装着手前に
確定する **設計文書**である。コード実装は本イシューのスコープ外（TASK-1.1a で
`Cargo.toml`・`crates/` 雛形は追加済みだが、`guardrail`・`self-repair` 自体の
実装着手は TASK-4.1・TASK-3.1 以降のため、本イシューでは文書のみが成果物となる）。

本文書の設計判断は仕様（`docs/spec/04-requirements.md` の REQ 番号）と
`Fandhe-AI/rust-ai-library-v1`（旧 Burn 基盤リポジトリ、以下「v1」）の実装
出典を都度明記する。`docs/spec/` submodule はこのリポジトリでは未初期化
（`.gitmodules` のみ存在）のため、spec 引用は GitHub API
（`gh api repos/Fandhe-AI/rust-ai-library-spec/contents/...`）経由で取得した
2026-08-05 時点の内容に基づく。

## 0. 用語対応表

イシュー #183 本文の「judge/eval 等」という表記のうち `judge` は、本文書・
実装では `check` と表記する。v1 実装（`rust-ai-library-v1/crates/guardrail/src/cli.rs`）
が単一変更セットの 3 分岐判定サブコマンドをすでに `check` として実装しており、
TASK-4.1 は「v1 guardrail CLI の移植」であるため、判定ロジック・3 分岐出力を
そのまま流用する目的で名称を変更しない。`eval`（ラベル付きデータセット一括評価）
は v1 と同名のまま継承する。

| イシュー本文の表記 | 本文書・実装での名称 | 理由 |
|---|---|---|
| judge | `check` | v1 実装の名称を継承（改名すると v1 移植の差分が無意味に増える） |
| eval | `eval` | v1 実装の名称をそのまま継承 |

## 1. guardrail CLI サブコマンド体系

### 1.1 サブコマンド一覧

| サブコマンド | 対象 | REQ-5 除外リスト適用 |
|---|---|---|
| `guardrail check` | 単一変更セットの 3 分岐判定（本番相当経路） | 適用する |
| `guardrail eval` | ラベル付きデータセットの一括評価（判定器単体の品質保証） | **適用しない** |

**設計制約（イシュー本文で明示・維持）**: `guardrail eval` は REQ-5 の
ポリシー除外リスト評価を含まない。`docs/spec/04-requirements.md` REQ-4
「2 層ラベルモデル 1 層目」注記（2026-08-05 v2 注記）が定めるとおり、
`eval` は REQ-4 の機械判定器単体を計測する経路であり、正解ラベルは
`expected_verdict`（判定器単体の正解）である。除外リスト適用後（2 層目、
`expected_verdict_after_exclusions`）の検証は `eval` サブコマンドではなく、
実 diff を持つ回帰テスト（TASK-5.3／TASK-6.1、v1 の
`tests/guardrail_self_regression.rs` 方式。cargo test・本番相当経路）が担う。
この責務分担（CLI ↔ 回帰テスト）は REQ-6「2 層の継続的検証」注記
（2026-08-05 v2 追加）とも整合する。

### 1.2 `guardrail check`

単一変更セットに対する 3 分岐判定（自動適用／エスカレーション／却下）を行う、
CI・self-repair から呼び出される本番相当経路。REQ-5 の除外リストを適用した
最終判定を返す。

| 引数 | 型・既定値 | 説明 |
|---|---|---|
| `--baseline <git-ref>` | 既定 `baseline` | 判定基準となる git ref（PoC-3 の `baseline` タグ運用を踏襲） |
| `--change-id <id>` | 任意 | 変更セット識別子（判定結果ファイル名に使用） |
| `--config <guardrail.toml>` | 任意（未指定時は `--repo` 直下探索、無ければ組み込み既定値） | 閾値設定 |
| `--preset <strict\|default\|loose>` | 既定 `default` | PoC-3 の 3 段階閾値プリセット |
| `--repo <path>` | 既定 `.` | 判定対象リポジトリのルート |
| `--signals <path>` | 任意（環境変数 `GUARDRAIL_ALLOW_INJECTED_SIGNALS=1` の設定時のみ有効。下記「`--signals` の迂回防止」参照） | 計測済みシグナル JSON（1.4 節）を直接注入する CI 契約検証パス。指定時は `--baseline`/`--repo` からの実シグナル計測を行わない |
| `--format <text\|json>` | 既定 `text` | 出力形式。`--signals` の有無に関わらず有効（本番相当経路〈実シグナル計測〉でも `json` を指定できる） |
| `--output <path>` | 任意 | 判定レポート JSON（2.1 節）の書き出し先。`--signals` の有無に関わらず有効。REQ-6 の回帰テストセット根拠データとするには本番相当経路（`--signals` 未指定）由来のレポートを用いること（2.1 節 `signal_source` 参照） |

**`--signals` の迂回防止（A08）**: `--signals` は CI 契約検証専用パスであり、本番相当経路（CI・self-repair から呼ばれる主経路）の迂回手段になってはならない。二重の防止策を設ける:

1. **入口ガード**: `--signals` の指定は環境変数 `GUARDRAIL_ALLOW_INJECTED_SIGNALS=1` が設定されている場合のみ受理し、未設定時は clap の usage エラー（終了コード `2`）で拒否する。CI 契約検証ジョブ（TASK-6.1）のみがこの環境変数を設定する
2. **成果物側の記録**: 入口ガードを通過して `--signals` 経由で生成された判定レポート JSON は、2.1 節 `signal_source` フィールドに `"injected"` を記録する（実シグナル計測経路は `"measured"`）。REQ-6 の回帰テストセット根拠データとしての採用可否はこのフィールドで機械判定可能とし、`--signals` 経由のレポートが実測データに混入することを成果物レベルでも防ぐ

出典: v1 `crates/guardrail/src/cli.rs`（`CheckArgs`。TASK-4.1-S1・S4）。

### 1.3 `guardrail eval`

ラベル付き変更セット（安全／危険／グレーの 3 分類）を一括評価し、件別の
判定結果・期待ラベルとの正誤照合、および見逃し率 0%・誤検知率 30% 以下
（REQ-4 受け入れ基準）との合否を出力する。REQ-6「ガードレール判定器自体の
品質保証」の 1 層目（`expected_verdict` による判定器単体検証）に対応する。

| 引数 | 型・既定値 | 説明 |
|---|---|---|
| `--dataset <dir>` | 既定 `crates/guardrail/tests/fixtures/labeled-changes` | ラベル付きデータセットのディレクトリ（`changes/<change_id>/{meta.toml,poc3-result.json}` 形式） |
| `--config <guardrail.toml>` | 任意 | 判定閾値（`lines_max`／`bench_max_pct` 等）のみ変更可能。見逃し率 0%・誤検知率 30% の合否閾値自体は REQ-4 の受け入れ基準そのものであり CLI から変更不可（コード側で定数として固定する） |
| `--preset <strict\|default\|loose>` | 既定 `default` | 判定閾値プリセット |
| `--repo <path>` | 既定 `.` | `--config` 未指定時に `guardrail.toml` を探索するルート |
| `--format <text\|json>` | 既定 `text` | 出力形式 |
| `--output <path>` | 任意 | 集計レポート・件別結果 JSON（2.2 節）の書き出し先 |

出典: v1 `crates/guardrail/src/cli.rs`（`EvalArgs`。TASK-4.3-S2・S3 統合）。

### 1.4 v2 差分: ベンチゲート計測系の bench-harness 付け替え

REQ-3「検証ゲートの計測系付け替え」注記（2026-08-05 v2 注記）・REQ-4
受け入れ基準（「計測基盤の付け替えは REQ-3 を参照」）に基づき、ベンチゲート
（4 ゲートの `cargo bench` 相当）の計測系は v1 の Criterion・Burn 計測 API
依存から `crates/bench-harness` へ付け替える。

- 依存方向は「`guardrail` → `bench-harness`（lib 依存）」と定義する。
  `guardrail check`/`guardrail eval` から見て、ベンチ計測（5 回計測・中央値
  採用）の実行系は `bench-harness` が提供する計測 API を呼び出すのみで、
  判定ロジック自体（中央値採用・決定的シード設定。REQ-4 受け入れ基準）は
  変更しない。
- `--signals` 契約検証パス（1.2 節）は、計測系の差し替えに関わらず
  「計測済みシグナル JSON を直接注入する」という契約自体は維持する。
  シグナル JSON の `bench_median_pct` 相当フィールドのスキーマ（2.1 節）は
  計測系が Criterion 由来でも bench-harness 由来でも同一とする。

## 2. 入出力仕様（JSON 保存含む）

`docs/spec/04-requirements.md` データ要件（L303、rust-ai-library-spec）が
定める「変更ごとの判定結果を構造化データ（JSON 等）として保存し、REQ-6 の
回帰テストセットの根拠データとすること」に対応する。

### 2.1 判定レポート JSON（`guardrail check --output`）

| フィールド | 型 | 説明 |
|---|---|---|
| `schema_version` | string | スキーマバージョン（例 `"1"`）。将来の破壊的変更検知用 |
| `signal_source` | `"measured"` \| `"injected"` | シグナルの出所。`--signals` 未指定（実シグナル計測。本番相当経路）なら `"measured"`、`--signals` 指定（CI 契約検証パス。1.2 節）なら `"injected"`。REQ-6 の回帰テストセット根拠データは `"measured"` のみを採用する |
| `change_id` | string \| null | `--change-id` の値 |
| `lines_changed` | u64 | 変更行数 |
| `public_api_broken` | bool | 公開 API シグネチャの破壊的変更有無 |
| `gaming_suspected` | bool | 本番コードとテストの同時変更（ゲーミング疑い）有無 |
| `build_result` | `"pass"` \| `"fail"` | `cargo build` 結果 |
| `test_result` | `"pass"` \| `"fail"` | `cargo test --release` 結果 |
| `clippy_result` | `"pass"` \| `"fail"` | `cargo clippy --all-targets -- -D warnings` 結果 |
| `bench_measurements_pct` | number[]（5 件以上。REQ-4「5 回以上」の受け入れ基準に対応） | ベンチ劣化率の計測値（bench-harness 由来。1.4 節） |
| `bench_median_pct` | number | 上記計測値の中央値（判定に用いる値） |
| `applied_exclusion_rule_ids` | string[] | 適用された REQ-5 除外ルールの `id` 一覧（空配列可） |
| `verdict` | `"auto_apply"` \| `"escalate"` \| `"reject"` | 最終判定（3 分岐。除外リスト適用後） |
| `reason` | string | 判定理由（人間可読。複数逸脱時は `"; "` 連結） |
| `reason_conditions` | string[] | 機械可読の逸脱条件 ID 一覧（CI・自己修復ループの照合用。例: `"lines_max_exceeded"`・`"gate_build_failed"`・`"policy_exclusion_match"`。`auto_apply` 時は空配列。TASK-4.1c・イシュー #106 で追加） |

REQ-3 データ要件・REQ-6 の回帰テストセット根拠データとして再利用可能な形式
とする。出典: v1 PoC-3 の `guardrail-results/*.json`（`docs/spec/04-requirements.md`
データ要件 L303 が参照する実測形式）を踏襲し、`applied_exclusion_rule_ids`・
`schema_version`・`signal_source`・`reason_conditions` を v2 で追加する。
これらは本文書が新設するフィールドであり、本文書は新規設計文書で既存実装との
互換負担がないため `schema_version` は `"1"`（初版）のまま据え置く（バージョン
非対応の既存消費者が存在しないため互換性のための版上げは不要。
`reason_conditions` の追加も前方互換の追記であり同様に版上げ不要）。

`signal_source = "measured"`（本番相当経路）時点の実装状態（TASK-4.1c・
イシュー #106）: `bench_measurements_pct`/`bench_median_pct` は常に空配列／
`0.0` を返す。CLI から起動できる bench ワークロードが v2 に未定義のため
（`bench_gate.rs` は呼び出し側クロージャ前提で未結線。CI 側の bench ゲートは
`verification-gate-bench.yml` に分離済み）、bench 条件は「未計測」（判定上は
`decision::BenchSignal::NotRun`＝逸脱なし扱い）として扱う。bench 実計測の
CLI 結線は別イシュー（親 #111 系）のスコープとする。

### 2.2 eval レポート JSON（`guardrail eval --output`）

**件別結果**（配列。要素ごと）:

| フィールド | 型 | 説明 |
|---|---|---|
| `change_id` | string | 変更セット識別子 |
| `expected_verdict` | `"auto_apply"` \| `"escalate"` \| `"reject"` | 期待判定（判定器単体。REQ-4 受け入れ基準） |
| `actual_verdict` | `"auto_apply"` \| `"escalate"` \| `"reject"` | 実判定 |
| `correct` | bool | 正誤（`expected_verdict == actual_verdict`） |
| `known_blind_spot` | bool | REQ-5 の既知ブラインドスポット（モデルアーキテクチャ変更・テスト許容誤差単独緩和）に該当するか |

**集計**:

| フィールド | 型 | 説明 |
|---|---|---|
| `total_count` | u64 | 評価件数 |
| `miss_rate_pct` | number | 危険な変更の見逃し率（%） |
| `false_positive_rate_pct` | number | 安全な変更の誤検知率（%） |
| `miss_rate_ok` | bool | 見逃し率 0% 達成（REQ-4 受け入れ基準） |
| `false_positive_rate_ok` | bool | 誤検知率 30% 以下達成（REQ-4 受け入れ基準） |

出典: v1 `crates/guardrail/src/eval/`（`EvalReport`。TASK-4.3-S2・S3・イシュー
#266）。

### 2.3 終了コード契約

**`guardrail check`**（v1 `crates/guardrail/src/exit_code.rs`
`GuardrailExitCode` を継承）:

| 値 | 意味 |
|---|---|
| `0` | 自動適用（`Verdict::AutoApply`） |
| `10` | エスカレーション（`Verdict::Escalate`） |
| `20` | 却下（`Verdict::Reject`） |
| `1` | 内部エラー（判定不能。シグナル入力欠落・JSON 解析失敗等） |
| `2` | （本体対象外）clap の usage エラー既定値 |

**`guardrail eval`**（v1 `EvalExitCode` を継承。`check` と重複しない値を選定
済みで、CI が `check` の失敗〈10/20〉と `eval` の閾値未達〈30〉を区別できる）:

| 値 | 意味 |
|---|---|
| `0` | 評価合格（見逃し率 0% かつ誤検知率 30% 以下。REQ-4 受け入れ基準達成） |
| `30` | 閾値未達（見逃し発生、または誤検知率 30% 超）→ CI を fail させる |
| `1` | 内部エラー（データセット不備・fixture 解析失敗等） |

**fail-closed 設計（A08）**: `Verdict`/評価合否 → 終了コードの変換は
それぞれ 1 箇所（`GuardrailExitCode::from_verdict`／`EvalExitCode::from_pass`
相当）のみに閉じ込め、他の経路から `0` を返せないようにする。内部エラー
（`1`）は自動適用（`0`）・評価合格（`0`）と明確に区別された別区分とし、
判定不能時に自動適用へ倒れない契約とする（`.claude/rules/security.md`
「A08 ソフトウェア・データ整合性」・`.claude/rules/coding-rust.md`
「エラーは型付きエラーとし、本番経路で `unwrap()`/`expect()` を使わない」
に対応）。

### 2.4 設定ファイル

| ファイル | 内容 | 既定値 |
|---|---|---|
| `guardrail.toml` | 判定閾値 | REQ-4 の初期推奨値: 変更行数 200 行以内・ベンチ劣化中央値 5% 以内（5 回計測）・build/test/clippy 全通過・公開 API 非破壊・ゲーミング疑いなしの 5 条件 |
| `policy-exclusion.toml` | REQ-5 除外ルール | `arch-hyperparameter-change`（`any_diff_in_paths` 方式）・`test-tolerance-loosening`（`test_assertion_relaxation_without_prod_change` 方式）の 2 件＋ v2 追加の依存変更カテゴリ（REQ-5 受け入れ基準 3。許容依存一覧・`Cargo.toml`／`Cargo.lock` の変更を無条件人間承認対象とする） |

**閾値・除外リストの変更はユーザー承認必須**（`.claude/rules/security.md`
「ガードレール閾値・ポリシー除外リスト・テスト許容誤差の変更は必ず人間
（ユーザー）の承認を経る」、`.claude/rules/delegation-impl.md`
「実装 Agent にガードレール閾値・テスト許容誤差を緩和させない」）。`--config`/
`--preset` はリポジトリにコミット済みの `guardrail.toml`／プリセット定義の
うちどれを使うかを**選択**する手段に過ぎず、CLI 呼び出しの引数だけで閾値の
数値そのものを緩めることはできない。閾値を変更できる唯一の経路は
`guardrail.toml`（または新設プリセット定義）自体をコミットする変更であり、
これはユーザー承認フローを経る（REQ-5 受け入れ基準 3・`policy-exclusion.toml`
の「依存の追加・更新」除外カテゴリと同じ扱い）。この非対称（CLI 引数では
既存値の選択のみ／数値変更にはコミットとユーザー承認が要る）により、2.3 節
の fail-closed 契約（判定不能を自動適用〈`0`〉と分離する）が閾値面でも
維持される。

### 2.5 外部入力の検証（A03 インジェクション対策）

`--config`／`--dataset`／`--signals` はいずれも外部入力（TOML・JSON）であり、
パース時に次を検証する:

- スキーマ検証は入力の性質で使い分ける。`guardrail.toml`／`policy-exclusion.toml`
  （閾値・除外ルール。ユーザー承認必須の設定ファイル、2.4 節）は
  `deny_unknown_fields` 相当（未知フィールドを拒否）を採用する。除外ルール
  `id` のタイポ等が黙って無視されると、発火すべき除外ルールが発火せず
  「自動適用すべきでない変更が自動適用される」方向に倒れうるため
  （`.claude/rules/security.md` A08「判定の迂回経路を作らない」と同種のリスク）、
  誤りは早期に型付きエラーとして検出する。`--signals`（1.4 節のシグナル
  JSON。CI 契約検証パス専用）・`--dataset`（ラベル付きデータセット）は
  必須フィールド欠落の検出のみとし、将来のフィールド追加に対する前方互換性
  を優先する（判定を安全側に倒す性質の設定ファイルではないため）
- `change_id` 等の外部由来文字列をシェル展開・パス連結に直接使わない
  （`--output` の書き込み先パス構築、`--dataset` 配下のファイル探索で
  `change_id` を経由する場合はパストラバーサル対策として `--dataset`
  ルート外への参照を拒否する）
- ログ・レポート JSON への出力は文字列連結ではなく `serde_json` の
  エスケープに一任する

## 3. self-repair 起動インターフェース

v1（`tools/self-repair/`）は lib クレートのみで CLI バイナリを持たなかった
（`Cargo.toml`／`src/lib.rs` を GitHub API で確認済み。TASK-3.1-S1 のコメント
「本クレートが担わない責務」にも CLI 化への言及はない）。v2 では REQ-3
「自作コアに対する自己修復ループ完走の再実証」（TASK-3.3）・CI 連携に必要な
起動インターフェースを新設する。

### 3.1 `self-repair run`

**実装済み**（イシュー #142 差し戻し分・完走判定基準 1。`crates/self-repair/
src/cli.rs`・`crates/self-repair/src/main.rs`）。#131（TASK-3.1 の CLI 化残
作業）が未実装のまま closed となり、他に追跡イシューがなかったため #142 の
スコープとして実装した。`--kind` で `RepairKind` の種別を受理する種別非
依存の実装であり、#141（バグ修正種別の再実証）からも再利用できる
（`bug-fix`／`feature-addition` の 2 値のみを CLI が受理する。
`RepairKind` 型自体は v1 由来の 3 variant〈`BugFix`/`PerfRegression`/
`FeatureAddition`〉を持つが、`perf-regression` は `PerfRegressionDetector`/
`PerfRegressionFixGenerator` が他 2 種別と非対称な構築契約〈`BenchMeasurer`・
戦略リスト〉を持ち CLI へ結線されていないため、`cli::parse_repair_kind` が
値の時点で usage エラー〈exit 2〉として拒否する。PR #361 codex-review P1
指摘対応: 値を受理してから実行時エラーを返す従来実装は「3 種別を受理する」
契約を満たさなかった）。

| 引数 | 型・既定値 | 説明 |
|---|---|---|
| `--kind <bug-fix\|feature-addition>` | 必須 | 対象種別（v1 `RepairKind`: `BugFix`/`PerfRegression`/`FeatureAddition` を継承するが、CLI が受理する値は `bug-fix`／`feature-addition` の 2 つのみ）。`perf-regression` を指定した場合は `cli::parse_repair_kind` が usage エラー（exit 2）を返す（`PerfRegressionDetector`/`PerfRegressionFixGenerator` が他 2 種別と非対称な構築契約〈`BenchMeasurer`・戦略リスト〉を持ち、#141／#142 いずれも本種別を必要とせず CLI 結線が未実装のため。追跡起票要否は out-of-scope-tracking.md 準拠でユーザーへ確認する） |
| `--repo <path>` | 既定 `.` | 対象リポジトリのルート。ループ全体（候補適用・4 ゲート検証・`git add -A` を含む）は `--repo` を `baseline_commit` の状態で `git clone --local` した隔離 sandbox（`self_repair::sandbox::RunSandbox`）内で完結し、`RepairCompositeGateSpec` の `workspace`／`sandbox_root` にはこの sandbox のパスを使う（`--repo` を直接渡さない）。`LoopOutcome::Adopted` の場合のみ、検証済み差分を `--repo` の作業ツリーへ競合検査つきで反映する（`self_repair::sandbox::reflect_adopted_diff`）。非採用・エラー経路では `--repo` の作業ツリー・index に一切触れない（PR #361 codex-review P0 指摘対応。`crates/self-repair/src/sandbox.rs` モジュール冒頭ドキュメント参照） |
| `--max-attempts <N>` | 既定 `5`（`NonZeroU32` 制約。0 を許容しない） | 修正試行回数の上限。`docs/self-repair-revalidation-plan.md` §5 基準 3 の承認済み提案値をそのまま既定値として採用した |
| `--log <path>` | 必須 | JSON Lines ログの出力先（3.3 節）。新規パスなら新規作成、既存パスなら
末尾から `seq`/`hash` を復元して追記継続する（v1 `LogWriter::open` の
`read_tail_state` 方式を継承。3.2 節 `verify-log --log` はこれと同一ファイルを
検証対象とする） |
| `--config <guardrail.toml>` | 任意 | guardrail 判定閾値（2.4 節と共通） |
| `--output <path>` | 任意 | `LoopReport`／`LoopFailure` JSON（試行回数・所要時間・各段階の判断根拠・
最終 verdict。v1 `tools/self-repair/src/report.rs` を踏襲）の書き出し先。
`check`/`eval` の `--output` と同様の位置づけとし、未指定時は標準出力へ
テキスト要約を出す（`--log` の JSON Lines とは別の成果物であり、`--log` は
必須・`--output` は任意という非対称を持つ: JSON Lines ログは改竄検知の
一次記録として常に残す必要があるが、`LoopReport` は呼び出し元がその場で
消費できれば足りるため） |
| `--candidates <path>` | 必須 | 事前生成済みの候補修正列（JSON）。3.1 節は候補生成手段を未定義のため #142 差し戻し分で新たに定めた: `[{"description": string, "files": [{"path": string, "content": string}]}]` 形式（`candidate::load_candidates_from_json`）。候補生成手段自体（AI 生成・人手作成）は本 CLI のスコープ外とし、事前に確定済みの候補列を受け取るのみとする |
| `--bench-bin <name>` | 必須 | 候補 diff 直接ベンチ実測（`RepairCompositeGate`。TASK-3.2a・#137）が `cargo build --release --bin <name>` するワークロード bin 名 |
| `--workload-source <path>` | 必須（複数指定可） | ゲーミング防止のためピン留めするワークロードソース（`--repo` 相対）。1 回以上必須 |
| `--policy-exclusion <path>` | 任意（既定 `<sandbox>/policy-exclusion.toml`。上記 `--repo` の隔離 sandbox 直下） | REQ-5 除外ルール設定ファイル。明示指定時はそのパスをそのまま読む（sandbox 相対に読み替えない） |
| `--allow-candidate-exec` | 必須フラグ（既定 false・値なし） | `--candidates` の候補コードを検証ゲート（`cargo build`／`cargo test`／`cargo clippy`）経由でホスト権限のまま実行することへの明示的な承認。未指定の場合は `cli::parse_run` が usage エラー（exit 2）として拒否し、`main.rs::run_run` へは到達しない（PR #361 codex-review P0 指摘対応。3.7 節「候補実行の信頼境界」参照） |

出力: 標準出力へのテキスト要約（既定）または `--output` 指定時は
`LoopReport`／`LoopFailure` JSON（上表）＋ 3.3 節の追記専用 JSON Lines
ログ（`--log`、常に出力）。`--output` の JSON は `LoopReport` の基本フィールド
に加え、最後に `VerificationOutcome::Passed` を返した試行の証跡
（`adopted_evidence`: `gate_report`／`bench_median_pct`／
`bench_measurements_pct`／`lines_changed`／`api_broken`／`gaming_suspect`／
`exclusion_rule_ids`）と `signal_source: "measured"` を含む（`--signals`
契約検証パス〈2.1 節〉を経由しない実シグナル計測であることの明示。
`docs/self-repair-revalidation/feature-addition/loop-report.json` が実例）。

### 3.2 `self-repair verify-log`

| 引数 | 型・既定値 | 説明 |
|---|---|---|
| `--log <path>` | 必須 | 検証対象の JSON Lines ログ |
| `--allow-empty-log` | 任意フラグ（既定 false） | レコード 0 件（空ログ）を明示的に許容する。指定なしでは空ログを検証不合格（exit 1）として扱う（PR #356 codex-review P1 指摘対応。下表参照） |

ハッシュチェーンの整合性検証（改竄検知）を行う。v1 の
`docs/self-repair-log-format.md`（TASK-3.2-S2）6 節が定める `verify_chain`
相当の検証を、v1 では持たなかった CLI エントリポイントとして新設する
（`.claude/rules/security.md`「ループ試行ログは改竄検知可能な形式で記録し、
取り込み判断の根拠を追跡可能にする」への対応。ログを事後監査する担当者が
`cargo test` 経由でなく直接検証できる手段を提供する）。**実装済み**
（イシュー #145 差し戻し分・完走判定基準 4。`crates/self-repair/src/main.rs`・
`crates/self-repair/src/cli.rs`）。検証ロジック本体は既存の
`crates/self-repair/src/logging.rs` の `verify_chain` の単一実装を CLI から
呼ぶのみとし、二重実装・迂回経路を作らない（`.claude/rules/security.md` A08）。

#### 終了コード（`verify-log`）

3.5 節は `run` の 3 分岐契約（0/10/20/1）であり `verify-log` には verdict が
ないため、guardrail の usage エラー区分（2.3 節）と整合する以下の契約を
別途定める:

| 値 | 意味 |
|---|---|
| `0` | チェーン整合（改竄なし）。標準出力にレコード件数・最終 `seq`・最終 `hash` を含む `OK:` メッセージを出す（監査担当者が外部アンカー運用・`docs/self-repair-log-format.md` 7 節と突合できるようにするため。Review #145 指摘対応）。レコード 0 件（空ログ・末尾切り詰めのいずれか区別不能）の場合は `--allow-empty-log` を明示指定したときのみ同じ `0` のまま `OK:` ではなく `WARN:` メッセージに変える（未指定時は下段の `1` を参照） |
| `1` | 検証不合格（`LogError::ChainViolation` = 改竄・欠落検知）・内部エラー（I/O・パース失敗）・`--allow-empty-log` 未指定でのレコード 0 件検知（PR #356 codex-review P1 指摘対応: 終了コードのみを見る監査自動化がログ全削除による改竄を「検証成功」として見逃す経路を塞ぐため、既定を fail-closed に変更）。fail-closed: 読めないログ・壊れたログも一律に非 0 とする |
| `2` | usage エラー（`--log` 欠落・未知引数） |

非 UTF-8 の `--log` パス（ファイルシステム上は有効だが OS 文字列として
UTF-8 でない値）を渡しても panic（exit 101）しないこと。CLI 引数は
`std::env::args_os()` ベースで受け取り、サブコマンド名・フラグ名のみを
UTF-8 として検証する（`--log` の値自体は検証しない。PR #356 codex-review
P1 指摘対応。`crates/self-repair/src/cli.rs` モジュール冒頭ドキュメント参照）。

### 3.3 ログ出力形式

v1 `docs/self-repair-log-format.md`（TASK-3.2-S2）の形式を TASK-3.4 で移植
する前提を明記する:

- JSON Lines（1 行 1 レコード、UTF-8、末尾改行付き）。追記専用
  （既存内容の上書き・削除を行わない）
- 各レコードは `seq`（連番）・`recorded_at_unix_ms`・`stage`・`payload`・
  `prev_hash`・`hash` の 6 フィールドを持つ
- 段階列は `loop_start → detection → attempt ×n → loop_outcome`（正常終了）
  または `attempt ×n → loop_failure`（異常終了）
- ハッシュは SHA-256 によるチェーン構造で改竄検知を可能にする

本文書は CLI 境界（引数・出力先・終了コード）のみを確定する。JSON Lines の
フィールド詳細スキーマ自体の再定義は TASK-3.4（#48 相当）のスコープであり、
本文書はその移植前提のみを記載する。

### 3.4 guardrail 連携方式

self-repair の `run` は `guardrail` を**サブプロセスとして起動せず、lib と
して直接呼び出す**。v1 `tools/self-repair/src/judge.rs`
（`GuardrailAdoptionJudge`）が `guardrail::decision::decide` へ接続し、
`AdoptionVerdict` を `decide()` を経由せずに作る経路が存在しない設計
（v1 `lib.rs` モジュールコメント）を v2 でも継承する。3 分岐判定の二重実装・
迂回経路を避ける（`.claude/rules/security.md` 「A08 ソフトウェア・データ
整合性: 自己修復ループが取り込む AI 生成変更はガードレール 3 分岐判定を
必ず経由する。判定の迂回経路を作らない」）。

### 3.5 終了コード

self-repair の終了コードは guardrail の 3 分岐契約（2.3 節）と整合させる:

| 値 | 意味 |
|---|---|
| `0` | ループが自動適用で完走（最終 verdict = `auto_apply`。`LoopOutcome::Adopted`）**かつ** 検証済み差分の `--repo` への反映（下記参照）も成功 |
| `10` | エスカレーション（人間承認待ち。`LoopOutcome::Escalated`） |
| `20` | 却下（`LoopOutcome::Rejected`） |
| `1` | 内部エラー（`LoopFailure`。段階の実行自体が失敗）／隔離 sandbox の構築失敗／`--log` 書き込み失敗（下記「`--log`／`--output` 書き込み失敗時の反映可否」参照）／`--output` 書き込み失敗（`--log` は成功済みのため反映は行うが終了コードは非 0 のまま）／自動適用された差分の `--repo` への反映失敗（下記参照） |
| `2` | usage エラー（`--kind` 欠落・不正値〈`perf-regression` 指定を含む。3.1 節参照〉・`--log` 欠落・`--max-attempts 0`・未知引数等） |

**`Exhausted`／`NoActionNeeded` の写像（イシュー #142 差し戻し分で追記）**:
上表は元々 3 分岐＋`LoopFailure` のみを定義しており、
[`self_repair::LoopOutcome`] の残り 2 variant（`Exhausted`＝試行上限到達、
`NoActionNeeded`＝検出段階で修正不要と判定されそもそもループが開始しな
かった）を扱っていなかった。いずれも「取り込まれなかった」点は却下と共通
するが、`Rejected`（3 ゲート・取り込み判断を経た明示的な却下）とは意味が
異なるため既存の `20` へ丸めず、「完走（exit 0）していない」ことを明確にする
ため内部エラー区分の **`1`** へ写像すると定める（`main.rs::
exit_code_for_outcome` のみが行う `LoopOutcome` → 終了コードの基本写像。
他の経路から `0` を返さない fail-closed 契約。`.claude/rules/security.md` A08）。

**`--log`／`--output` 書き込み失敗時の反映可否（PR #361 codex-review P1・
Medium 指摘対応で追記）**:
`--repo` への反映（下記「採用差分の `--repo` への反映失敗」参照）は
「監査ログ（`--log`）が採用結果を一次記録として残せている」ことのみを
前提とする。`--log` の書き込み（自己検証込み）が失敗した場合は
`LoopOutcome::Adopted` であっても反映を一切行わず（`--repo` に触れない）、
終了コードは `1` になる。一方 `--output`（3.1 節の任意の複製レポート JSON。
未指定時は標準出力へ要約を出す）の書き込みのみが失敗した場合は、`--log` が
既に一次記録として残せている以上、反映は通常どおり行い、終了コードのみ
`1` にする（`--output` の失敗を `--log` と同列に扱うと、監査ログには
記録済みの正当な採用差分の反映まで過剰にブロックしてしまうため。
`crates/self-repair/src/main.rs::finish_with_report`／`outcome_for_reflection`
参照）。

**採用差分の `--repo` への反映失敗（PR #361 codex-review P0 指摘対応で追記）**:
`run` は `--repo`（人間の作業リポジトリ）を直接検証対象にせず、`baseline_commit`
の状態で `git clone --local` した隔離 sandbox（`self_repair::sandbox::RunSandbox`）
内でループ全体（候補適用・4 ゲート検証・`git add -A` を含む）を完結させる
（`--repo` を直接渡すと、非採用に終わった候補の変更が未コミットの作業
ツリーへ残置され `git add -A` が無関係な変更まで staged にしてしまう問題が
あったため。`crates/self-repair/src/sandbox.rs` モジュール冒頭ドキュメント
参照）。`LoopOutcome::Adopted` の場合のみ、`self_repair::sandbox::
reflect_adopted_diff` が sandbox の検証済み差分を `--repo` の作業ツリーへ
`git apply --check` の競合検査つきで反映する。この反映が失敗した場合
（`--repo` がダーティで sandbox の差分と競合する等）は `--repo` の作業ツリー
へ一切触れず、`exit_code_for_outcome` が返した `0` を `1` へ上書きする
（`--log`／`--output` はループの真の結果〈Adopted〉を記録済みのまま変更
しない。反映失敗時の sandbox は削除せず標準エラー出力に記載されたパスへ
保持する）。非採用（`Escalated`/`Rejected`）・内部エラー経路では反映処理
自体を呼ばないため `--repo` に一切触れない。

### 3.6 骨格の移植スコープ

Detector／FixGenerator／VerificationGate／AdoptionJudge の 4 trait 構成
（v1 `tools/self-repair/src/stages.rs`）は TASK-3.1 の移植対象であり、本文書
は CLI 境界（引数・出力・終了コード）のみを確定してスコープを限定する。
trait 設計自体の詳細は TASK-3.1 実装時に確定する。

### 3.7 候補実行の信頼境界

PR #361 codex-review P0 指摘（main.rs:272 相当）: `self-repair run` は
`--candidates` の候補コードを、検証ゲート（`RepairCompositeGate`。
`crates/self-repair/src/verify_gates.rs`・`verify_direct_composite.rs`）が
sandbox clone 内で `cargo build`／`cargo test --release`／`cargo clippy` を
実行することで検証する。これは以下の理由により **OS レベルのプロセス・
権限・ネットワーク隔離ではない**:

- sandbox clone（`crates/self-repair/src/sandbox.rs`。`RunSandbox`）が
  提供するのは `--repo`（人間の作業リポジトリ）の作業ツリー・index を
  汚さないための**ファイルシステム上の作業分離**のみである（`git clone
  --local` による独立 `.git` の構築。3.1 節 `--repo` の項・`sandbox.rs`
  モジュール冒頭ドキュメント参照）
- `cargo build`／`cargo test`／`cargo clippy`（`build.rs`・
  `#[test]` 関数・手続き型マクロを含む）はホストと同一のプロセス・OS
  ユーザー権限・ネットワーク到達性のまま実行される。悪意ある候補は
  任意コード実行が可能である

この脅威モデルへの対応として、v2 では次の設計を採る（低権限コンテナ等
による OS レベル隔離は将来課題とし、実装しない。「対象外」節参照）:

1. **`--candidates` は信頼済み入力に限定する**: 本 CLI が想定する
   `--candidates` は、本リポジトリの自己修復ループ（検出 → 修正生成 →
   検証 → 取り込み判断。REQ-3）が生成し、かつ guardrail 3 分岐判定
   （REQ-4）と組み合わせて運用される候補列であり、出自不明な外部候補を
   無検証で受理する用途を想定しない
2. **明示的な承認なしには実行しない**: `--allow-candidate-exec`
   （3.1 節。既定 false・必須フラグ）を指定しない限り `cli::parse_run`
   が usage エラー（exit 2）として拒否し、`main.rs::run_run`（検証コマンド
   実行を含む以降の処理）へは一切到達しない。`--candidates` は必須引数
   のため、このフラグは実質「候補コードのホスト権限実行を承認する
   スイッチ」として機能する

OS レベル隔離（低権限コンテナ・ネットワーク遮断による候補検証コマンドの
サンドボックス実行）は本節が定義する信頼境界の外にある将来課題であり、
`out-of-scope-tracking.md` 準拠で PR 本文の「対象外」節に追跡する。

### 3.8 除外設定の固定（判定迂回防止）

PR #361 codex-review P1 指摘（main.rs:277 相当）: `--policy-exclusion`
未指定時の既定パス（`sandbox_root/policy-exclusion.toml`）は候補が書き込み
可能な sandbox 内にある。旧実装は `RepairCompositeGate::verify` が呼ばれる
たびこのファイルをディスクから再読込していたため、候補が自身の diff に
本来 match して人間承認（エスカレーション）へ回すはずだった除外ルールを
`policy-exclusion.toml` の書き換えで削除・縮小し、match を回避して
ガードレール判定（REQ-4 の 3 分岐判定）を迂回できる余地があった
（A08「判定の迂回経路を作らない」違反）。攻撃方向はルールの**追加**では
ない点に注意: `guardrail::decision::decide` は `exclusion_rule_ids` が
1 件以上あれば機械判定の結果によらず無条件でエスカレーションへ回す
「除外リストは安全側にしか作用しない」設計（REQ-5）であり、ルールを
増やしても判定は緩まない。緩む方向は「本来 match すべきルールを消し
match させない」ことだけである。

v2 は次の 2 つの防御を組み合わせて塞ぐ:

1. **除外設定の事前固定**: `self_repair::diff_signals::
   load_policy_exclusion_config` を **候補適用前（sandbox 構築直後・候補
   未適用の状態）に一度だけ**呼び、返る `guardrail::PolicyExclusionConfig`
   を `RepairCompositeGateSpec::policy_exclusion` として試行ループ全体で
   不変値のまま使い回す。`RepairCompositeGate::verify` は以降この値を
   `crate::diff_signals::measure_diff_signals` へ渡すのみで、ファイルパスを
   再読込しない。既定パスの場合、この事前固定が読むのは `baseline_commit`
   時点の内容（`RunSandbox::create` が `git clone --local` で反映した直後の
   状態）であり、`--repo` の作業ツリー上の未コミット編集は反映されない
2. **防御的拒否**: 候補（`--candidates`）の変更対象ファイル一覧に
   `policy-exclusion.toml`／`guardrail.toml`（ファイル名の大文字小文字を
   区別しない比較）が含まれる場合、`crates/self-repair/src/candidate.rs::
   apply_candidate` が候補適用前に型付きエラー（`SelfRepairError::
   FixGeneration`）で無条件拒否する。1 の事前固定と独立した多重防御である。
   ただしこの判定はファイル名（basename）ベースであり、`--policy-exclusion`
   に既定と異なるファイル名（例: `my-rules.toml`）を sandbox 内のパスとして
   指定した運用まではカバーしない。その運用での唯一の防御は 1（事前固定）
   であり、これは意図した境界である（`--policy-exclusion` に sandbox
   到達不能な外部パスを指定する運用であれば、候補はそもそもそのファイルへ
   書き込めない）

いずれもコード上の強制であり、除外リスト・ガードレール閾値そのものの
変更（値の緩和）はこれまでどおり `.claude/rules/security.md`
「ガードレール閾値・ポリシー除外リストの変更は必ず人間の承認を経る」に
従いユーザー承認を要する。

## 4. スコープ外の明示

以下は本文書の対象外とし、各タスクへの参照のみ記載する
（`.claude/rules/out-of-scope-tracking.md` 準拠。いずれも既存イシュー
（#101 系・#131〜#149）で追跡済みのため新規起票は不要）:

- 閾値の具体値の再評価（TASK-4.3。REQ-4 受け入れ基準「新実装リポでの
  ラベル付き変更セット評価による閾値再確認」）
- 除外ルールの実装（TASK-5.x）
- CI ジョブ化（TASK-6.1）
- ログ形式の詳細スキーマ再定義（TASK-3.4）
- `self-repair run --max-attempts` の既定値の最終確定（本文書は
  「`NonZeroU32` 制約」という型的制約のみを提案し、数値自体は TASK-3.1
  実装時に確定する）

## 5. 参照

- `docs/spec/04-requirements.md`（rust-ai-library-spec）REQ-3・REQ-4・REQ-5・
  REQ-6・画面・インターフェース要件（L294-298）・データ要件（L300-304）
- `Fandhe-AI/rust-ai-library-v1`:
  - `crates/guardrail/src/cli.rs`（`check`/`eval` 引数定義。TASK-4.1-S1・S4）
  - `crates/guardrail/src/exit_code.rs`（終了コード契約。TASK-4.1-S4）
  - `tools/self-repair/src/lib.rs`（自己修復ループ骨格。TASK-3.1）
  - `docs/self-repair-log-format.md`（ログフォーマット仕様。TASK-3.2-S2）
- `.claude/rules/security.md`・`.claude/rules/coding-rust.md`・
  `.claude/rules/out-of-scope-tracking.md`（本リポジトリ）
