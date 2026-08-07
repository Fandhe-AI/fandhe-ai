# ポリシー除外リストの設定形式仕様・統合点・運用フロー（TASK-5.1）

TASK-5.1（`docs/spec/05-tasks.md:188-193`）は「`policy-exclusion.toml` の
移植・依存追加カテゴリの新設」を 1 タスクとして定義しているが、本リポジトリ
では同一ファイルへの並行編集を避けるため複数イシューに分割している
（[`.claude/rules/delegation-impl.md`](../.claude/rules/delegation-impl.md)
禁止事項: 複数 Agent に同一ファイルを並行編集させない）。

- **本イシュー（#119・TASK-5.1a）**: v1 資産 `policy-exclusion.toml`・
  `docs/policy-exclusion-design.md` の**忠実移植のみ**を担当する。ルール値
  （`id`／`category`／`description`／`rationale`／`paths`／`match`／`action`）
  はバイト単位で不変とし、値の追加・変更を一切含まない（自動運転の安全側
  判断）。
- **後続イシュー（#120・TASK-5.1b・human-required）**: REQ-1 の受け皿
  として「許容依存一覧の追加・更新、`Cargo.toml`／`Cargo.lock` の変更」
  カテゴリを新設する。プロダクト判断を伴うため担当は人間（実装補助は
  Claude Code 可。`docs/spec/05-tasks.md:193`）。
- **後続イシュー（#121・TASK-5.2）**: 除外ルールの match 方式実装・
  ガードレール統合（`crates/guardrail/src/policy_exclusion.rs` の新設・
  組み込み既定値との一致回帰テスト）。
- **後続イシュー（#125・TASK-5.3）**: 2 層ラベルモデルの不変条件回帰
  テスト。
- **後続イシュー（#128〜#130・TASK-5.4）**: ブラインドスポット（G2/G5）
  回帰テストの移植。

想定読者は、TASK-5.1b（#120）・TASK-5.2（#121・設定ファイル読み込み実装・
無条件人間承認判定の統合・初期設定ファイルの依存カテゴリ追加）の実装者、
および TASK-5.3（ブラインドスポット回帰テスト・#125・#128〜#130）の
実装者である。これらが迷わない粒度（スキーマのフィールド名・型・検証規則）
まで具体化することを目的とし、プロダクト判断を伴うカテゴリの新規追加・
意味論の拡張は本イシュー（#119）のスコープ外とする（`docs/spec/05-tasks.md`
TASK-5.1 の担当欄は「人間（実装補助は Claude Code 可）」であり、既存 2
カテゴリの具体化・忠実移植に限定する）。

## 1. 目的・根拠

- **REQ-5**（`docs/spec/04-requirements.md:121-136`）: ガードレールの機械判定
  だけでは検知できない変更カテゴリ（モデルアーキテクチャ変更・テスト許容
  誤差の単独緩和）を、ポリシーとして無条件に人間承認必須と定義する仕組みを
  持つこと。受け入れ基準は次の 5 点（`docs/spec/04-requirements.md:130-134`。
  2026-08-05 v2 基盤非依存化で 2 項目追加）。
  1. モデルアーキテクチャ・ハイパーパラメータ変更は無条件人間承認（PoC-3 G2）
  2. テスト許容誤差・アサーション条件の単独緩和は無条件人間承認（PoC-3 G5。
     対象にはバックエンド間数値一致テストの許容誤差〈REQ-2〉・PyTorch
     参照値比較の許容誤差〈REQ-7〉を含み、これらの変更は該当要件の改定と
     セットでのみ人間承認を経て可能とする）
  3. **（v2 追加）** 依存（許容依存一覧・`Cargo.toml`／`Cargo.lock`）の
     追加・更新を除外カテゴリに含めること（REQ-1 の受け皿。TASK-5.1b・
     #120 のスコープ）
  4. 除外リストの対象ファイル・対象変更種別を**設定として明示的に定義し、
     更新可能な形で管理**すること。除外ルールは「パス／変更内容パターンに
     マッチした事実のみで発火する」保守的な match 方式とし、変更内容の
     意図解釈を伴う判定は行わない
  5. **（v2 追加）2 層ラベルモデルの 2 層目**: 除外リスト適用後の正解判定
     （`expected_exclusion_rule_ids`・`expected_verdict_after_exclusions`）
     を、ラベル付き変更セットの各項目に対して定義し、次の不変条件を回帰
     テストで機械検証すること。(1) `expected_exclusion_rule_ids` が空の
     場合、`expected_verdict_after_exclusions == expected_verdict`。
     (2) 非空の場合、`severity(expected_verdict_after_exclusions) >=
     max(severity(expected_verdict), Escalate)`（**除外リストは安全側に
     しか作用しない＝見逃し方向に緩まない**）。TASK-5.3（#125）のスコープ。
  - **（v2 追加）保守的設計のトレードオフ**: `arch-hyperparameter-change`
    はパスのみで match するため、`src/model*.rs` へのドキュメントコメント
    追加・定数抽出等の挙動不変の変更（ラベル付きデータセットの S1・S3
    相当）も除外リスト適用後は `escalate` となる。これは判定器のバグでは
    なく、除外リストが「変更内容の意図（ドキュメントのみか実質的な
    アーキテクチャ変更か）を区別しない意図的に保守的な設計」であることに
    由来する、意図されたトレードオフである（見逃しより過剰エスカレー
    ションを許容する）。
- **PoC-3 発見事項 5**（`docs/spec/03-poc/poc-3-guardrail-validity/README.md:154`）:
  G2（隠れ層次元数変更）・G5（テスト許容誤差単独緩和）が REQ-4 の機械的
  シグナル（build/test/clippy/bench/行数/API 破壊/ゲーミング同時変更検知）
  を全て回避し自動適用されてしまうことを実測で確認済み。本リポジトリでは
  `crates/guardrail/tests/fixtures/labeled-changes/changes/G2-hidden-dim-increase/`・
  `G5-test-only-loosen/` にラベル付きフィクスチャ（`change.patch`・
  `meta.toml`、`known_blindspot = true`）として移植済みであり（TASK-4.2a/b
  マージ済み）、`meta.toml` の `expected_exclusion_rule_ids` は既に
  `["arch-hyperparameter-change"]`・`["test-tolerance-loosening"]` を参照
  している（4.1 節参照）。
- **[`.claude/rules/security.md`](../.claude/rules/security.md)**: 「ガードレール
  閾値・ポリシー除外リスト・テスト許容誤差の変更は必ず人間（ユーザー）の
  承認を経る」「自己修復ループが取り込む AI 生成変更はガードレール 3 分岐
  判定を必ず経由する。判定の迂回経路を作らない」。除外リストは判定を迂回
  させないための最後の砦であり、除外リスト自体の変更にも人間承認を必須と
  する。

## 2. 設定ファイルの配置・形式

- リポジトリルートに配置する `policy-exclusion.toml`（TOML）。
  `guardrail.toml`（閾値設定、`crates/guardrail/src/config.rs`）と同格の
  配置とするが、**別ファイルに分離**する。理由:
  - REQ-5 受け入れ基準「除外リストは設定として明示的に定義し更新可能な
    形で管理する」に対応し、除外リストを閾値設定と独立に更新できるように
    する。
  - ガードレール閾値の変更（数値の緩和方向）と、除外リストの変更（対象
    カテゴリ・パターンの変更）は、承認単位として性質が異なる（前者は
    「自動化の感度」の調整、後者は「何を完全自律の対象外にするか」という
    ポリシー判断）。同一ファイルに混在させると変更差分の意図が読み取り
    にくくなる。
- 冒頭コメントの形式は `guardrail.toml`・`policy-exclusion.toml`（本イシュー
  #119 で作成）の前例に倣い、目的・対応する REQ 番号・数値/ルール変更には
  ユーザー承認が必須である旨を明記する。
- トップレベルに `schema_version`（整数）フィールドを持たせ、将来の
  カテゴリ追加（`docs/spec/06-roadmap.md` の「新たなブラインドスポットが
  発見された場合は REQ-5 のポリシー除外リストへ追加する運用」）に備える。
  現時点でサポートするのは `schema_version = 1` のみとし、未知のバージョン
  値は 4 節のとおりパースエラーとする（サイレントな後方互換フォール
  バックはしない）。
- **v2 実体**: `crates/guardrail/src/config.rs` が `guardrail.toml` の
  読み込みで `toml_lite::MAX_INPUT_BYTES`（64 KiB。
  `crates/guardrail/src/toml_lite.rs:38`）の上限・`deny_unknown_fields`
  相当の未知キー拒否（`config.rs:155-184` 付近）・`repo_root/guardrail.toml`
  への配置慣行（`config.rs:117-141` 付近）の前例を既に持つ。TASK-5.2
  （#121）で新設する `policy_exclusion.rs` はこの `config.rs` の流儀を
  そのまま踏襲する（v1 の `MAX_CONFIG_FILE_BYTES` 参照は v2 では
  `toml_lite::MAX_INPUT_BYTES` に読み替える）。**なお v2 の許容依存 8
  区分（[`deps-policy.md`](../.claude/rules/deps-policy.md)）には `toml`
  クレートは含まれない**。TOML パースは `crates/guardrail/src/toml_lite.rs`
  の自作パーサ（`toml_lite`）を用いる。同パーサの array-of-tables・
  インラインテーブル（サブテーブル `[exclusion.match]`）対応の要否は
  TASK-5.2（#121）の実装事項として確認する。

## 3. スキーマ仕様（TASK-5.2〈#121〉が実装する契約）

### 3.1 トップレベル

```toml
schema_version = 1

[[exclusion]]
# ... (3.2 参照)
```

- `schema_version`: `u32`。必須。`1` 以外はパースエラー。
- `exclusion`: array-of-tables。必須。**空配列は許可しない**（4 節参照）。

### 3.2 `[[exclusion]]` 要素

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `id` | 文字列 | 必須 | ルールの一意識別子（例 `arch-hyperparameter-change`）。同一設定ファイル内で重複不可 |
| `category` | 文字列（列挙） | 必須 | 変更種別。初期値は `"architecture_change"` / `"test_tolerance_loosening"` の 2 値のみ許可（依存変更カテゴリは TASK-5.1b・#120 が追加） |
| `description` | 文字列 | 必須 | 日本語での説明（何を対象とするルールか） |
| `rationale` | 文字列 | 必須 | 根拠。PoC-3 発見事項 5（G2/G5）・REQ-5 受け入れ基準への参照を含めること |
| `paths` | 文字列配列（glob） | 必須 | 対象ファイルパターン。空配列は不可（4 節参照） |
| `match` | インラインテーブル | 必須 | 対象変更種別の判定方法（3.3 節） |
| `action` | 文字列（列挙） | 必須 | 現状 `"human_approval"` のみ許可。fail-closed のため他の値は不可 |

`paths` の glob 記法は `crates/guardrail` の他の設定・シグナル実装
（`git diff --name-only` 由来のファイルパスに対するマッチ）と同様、
シェル展開を経由しない実装（`glob` クレート等によるパターン評価。導入
する場合は依存追加としてユーザー承認必須）を前提とする。パスはリポジトリ
ルート相対とし、`**` によるディレクトリ階層をまたぐ一致をサポートする
こと（`crates/<name>/src/...` という本リポジトリの workspace 構成と、
PoC-3 フィクスチャの標準化されていない単一クレート構成 `src/...` の
双方に同一パターンで一致させるため。4.1 節の対応表を参照）。

### 3.3 `match` フィールド（判定方法）

`match` はカテゴリごとに判定意味論が異なるため、`type` フィールドで
判定方式を切り替えるインラインテーブルとする。未知の `type` はパース
エラーとする（4 節）。TASK-5.1 では以下の 2 種類のみを定義する
（新しい `type` の追加はカテゴリ追加と同様、プロダクト判断を伴うため
本イシューのスコープ外）。

#### 3.3.1 `type = "any_diff_in_paths"`（カテゴリ 1: `architecture_change` が使用）

- **入力**: `paths` に一致する変更ファイルの一覧（`git diff --name-only
  <baseline> -- . ':!Cargo.lock'` 相当。v1 出典
  `rust-ai-library-v1/crates/guardrail/src/gaming.rs` の `changed_files`
  と同一の変更ファイル列挙を再利用する想定。v2 への移植は TASK-5.2
  〈#121〉）。
- **判定**: 変更ファイル一覧のいずれかが `paths` のいずれかの glob に
  一致すれば match（差分の内容・削除行の有無は問わない）。
- **追加フィールドなし**（`type` のみで判定が完結する）。

#### 3.3.2 `type = "test_assertion_relaxation_without_prod_change"`（カテゴリ 2: `test_tolerance_loosening` が使用）

- **入力**:
  - `assertion_patterns`（文字列配列）: 削除行に対する許容誤差
    リテラル・アサーションのパターン。初期値は
    `["assert!", "abs() <", "1e-[0-9]"]` を固定で使う（v1 出典
    `rust-ai-library-v1/crates/guardrail/src/gaming.rs:6-10, 133-142` の
    `contains_assertion_pattern`（(a) 条件）と**同一のリテラル集合**。
    パースエラー防止のためフィールドとしては受け付けるが、TASK-5.2
    実装では初期値からの変更を想定しない。変更する場合はゲーミング検知
    ロジック〈v1 `gaming.rs`。v2 未移植〉とのパターン乖離が生じないよう、
    両モジュールを同時にレビューすること）。
  - `touches_test_assertion`: `git diff -U0 <baseline> -- . ':!Cargo.lock'`
    の削除行（`^-` かつ `^--` でない行。v1 出典 `gaming.rs:124-131` の
    `is_removed_content_line` と同一条件）のいずれかが
    `assertion_patterns` のいずれかに一致するか（v1 出典
    `gaming.rs:110-122` の `touches_test_assertion` をそのまま再利用する
    想定）。
  - `touches_prod_logic`: `src/*.rs`（`tests/*` を除く）の変更ファイルに
    ついて、変更 hunk の baseline 側開始行が現在の作業ツリーファイル中の
    `mod tests` 行より前かどうか（v1 出典 `gaming.rs:144-169` の
    `touches_prod_logic` をそのまま再利用する想定）。
- **判定**: `touches_test_assertion == true` かつ
  `touches_prod_logic == false` の場合に match。
- **v1 `gaming.rs` の疑い検知（`suspect = touches_test_assertion &&
  touches_prod_logic`）との関係**: 本カテゴリは意図的に
  `touches_prod_logic` の**否定**を要求する。REQ-4 のゲーミング検知は
  「本番コードとテストの同時変更」（両方 true）のみを対象とし、
  「テストファイル単独の変更」（`touches_test_assertion` のみ true）は
  現行ルールでは検知できないブラインドスポットである
  （REQ-5 受け入れ基準 2 項目め、PoC-3 発見事項 5）。両者は排他的な
  入力領域をカバーする補完関係にあり、本カテゴリの `match` はその
  ギャップを埋めるために存在する。`touches_test_assertion` と
  `touches_prod_logic` が両方 true の場合（G1 相当。v2 フィクスチャ
  `crates/guardrail/tests/fixtures/labeled-changes/changes/G1-gaming/`）は
  REQ-4 のゲーミング検知が既に却下・エスカレーションを担うため、本
  カテゴリでは match させない設計とする（二重の除外リスト適用を避け、
  判定根拠を一意に保つ）。
- **`paths` との関係**: `type = "test_assertion_relaxation_without_prod_
  change"` の判定自体はリポジトリ全体の diff を対象とする
  （v1 `gaming.rs` の実装がファイル単位ではなく差分全体を走査するため）。
  `paths` フィールドは「本ルールが対象とみなすファイル種別の宣言」
  （4.1 節の記述例では `["**/*.rs"]` = 全 Rust ソースファイル）として
  保持し、将来 `paths` を絞り込みたくなった場合（例: 特定クレートの
  テストのみ対象外にする）の拡張余地として空にしない。

## 4. バリデーション・fail-closed 規則（security.md A03/A08 対応）

- **`deny_unknown_fields`**: `crates/guardrail/src/config.rs` の設定
  パース（`ThresholdsRaw` 相当の未知キー拒否、`config.rs:155-184` 付近）
  と同一方針で、トップレベル・`[[exclusion]]`・`match` インラインテーブル
  のすべてに適用する。タイポ由来の未知キーはパースエラーとする。
- **ファイルサイズ上限**: `crates/guardrail/src/toml_lite.rs::MAX_INPUT_BYTES`
  （64 KiB。`toml_lite.rs:38`）と同値の上限を設ける。異常に巨大な設定
  ファイルの読み込み・パースを未然に拒否する（A03: 外部入力の検証・DoS
  的入力の早期棄却）。
- **`schema_version`**: `1` 以外はパースエラー（2 節）。
- **`exclusion` 配列の空・カテゴリ欠落**: 空配列はエラーとする。さらに
  安全側の規則として、初期 2 カテゴリ（`architecture_change` /
  `test_tolerance_loosening`）の**両方が最低 1 件ずつ存在しない設定ファイルは
  エラー**とする（受け入れ基準 1・2 は「無条件で人間承認必須」であり、
  一方のカテゴリが欠落した設定ファイルを読み込ませて自動適用させてしまう
  ことは REQ-5 の趣旨に反するため）。**TASK-5.1b（#120）でカテゴリが
  3 種類に拡張された場合、本規則は「初期 2 カテゴリ＋新設カテゴリの計 3
  種類がすべて最低 1 件ずつ存在すること」へ更新する想定**（本イシューでは
  変更しない）。
- **未知の `category` / `match.type` / `action`**: パースエラーとする。
  `crates/guardrail/src/decision.rs` の「未知値はワイルドカードで許容せず
  拒否する」方針（`decision.rs:55-56`「`match` は網羅列挙とし `_ =>`
  ワイルドカードを使わない」）と同一の思想（黙って「除外対象外」に
  落とさない）。
- **`paths` の空配列**: 不可。`id` ごとに最低 1 パターンを要求する。
- **glob パターンの検証**: パースエラーとして拒否する不正パターンの例:
  - 絶対パス（`/` 始まり）: リポジトリルート外を指せてしまうため禁止
  - `..` を含むパス要素: リポジトリルート外への脱出を禁止（A03）
  - シェル呼び出しを経由しない実装（`glob` クレート等の直接評価）を前提とし、
    パターン文字列をシェルコマンドに展開しない
- **設定ファイル欠落時の既定動作**: `crates/guardrail/src/config.rs` が
  `--config` 未指定時に組み込み既定値（安全側）へフォールバックする方針と
  **同じ形にする**。`policy-exclusion.toml` がリポジトリルートに存在しない
  場合、**空の除外リストにフォールバックしてはならない**（除外リストが
  「存在しない」＝「無条件人間承認の対象が 0 件」に読み替えられると、
  security.md A08 が禁じる判定迂回経路になる）。組み込み既定値として、
  4.1 節の記述例と同一内容（初期 2 カテゴリ）をコードに埋め込み、ファイル
  欠落時もこれを使う。本イシュー（#119）が作成する `policy-exclusion.toml`
  の初期内容は、この組み込み既定値（TASK-5.2・#121 で実装）と数値・文字列
  が一致することを回帰テストで保証する想定（`config.rs` の
  `repo_root_guardrail_toml_matches_builtin_defaults` と同種のテストを
  TASK-5.2 側で追加する。本イシューでは設計・設定ファイル作成のみ）。
- **除外リスト自体の変更**: 除外リスト（本ファイル・組み込み既定値の
  いずれも）の変更はユーザー承認必須（security.md）。運用フロー（誰が
  どう承認するか）の詳細は 6 節で規定する。

## 4.1 初期 2 カテゴリの TOML 記述例

**v1 設計文書からの唯一の記述差分（値不変・フィールド順序のみ修正）**:
v1 `docs/policy-exclusion-design.md` の本節記述例は `action` を
`[exclusion.match]` より後に置いていたが、これは TOML 仕様上
`exclusion.match.action` として `match` テーブルへ帰属してしまう誤りで
あった（2 節冒頭コメントの「フィールド順序についての注意」と矛盾する）。
v1 の実ファイル `policy-exclusion.toml`（本イシューが移植した値の出典）は
`action` を `[exclusion.match]` より前に置く正しい順序で書かれており、
本節の記述例もそれに合わせて `action` の位置のみ修正した（キー・値・
テーブル名はいずれも不変）。

### カテゴリ 1: モデルアーキテクチャ・ハイパーパラメータ変更（G2 根拠）

```toml
schema_version = 1

[[exclusion]]
id = "arch-hyperparameter-change"
category = "architecture_change"
description = "モデルアーキテクチャ定義ファイル（隠れ層次元数・レイヤー構成等のハイパーパラメータを含む）への変更"
rationale = "PoC-3 発見事項5（G2: 隠れ層次元数 8→10 拡大）が REQ-4 の機械的シグナル（build/test/clippy/bench/行数/API破壊/ゲーミング検知）を全て回避し自動適用された（docs/spec/03-poc/poc-3-guardrail-validity/README.md）。REQ-5 受け入れ基準1に対応。"
paths = ["**/src/model*.rs", "**/src/nn/**", "**/src/*model*/**"]
action = "human_approval"

[exclusion.match]
type = "any_diff_in_paths"
```

- **G2 フィクスチャとの対応**: `crates/guardrail/tests/fixtures/labeled-changes/
  changes/G2-hidden-dim-increase/change.patch` は `src/model.rs` を変更する
  （PoC-3 の標準化されていない単一クレート構成。本リポジトリの workspace
  構成では `crates/<name>/src/model.rs` に相当する）。`paths` の
  `**/src/model*.rs` は `**` によって深さを問わず一致するため、フィクスチャの
  `src/model.rs`・本リポジトリの `crates/<name>/src/model.rs` の両方に
  一致する。`match.type = "any_diff_in_paths"` により、差分内容（隠れ層次元数の
  具体的な変更幅等）を問わずファイルパスの一致のみで match する。同フィクスチャ
  の `meta.toml`（`crates/guardrail/tests/fixtures/labeled-changes/changes/
  G2-hidden-dim-increase/meta.toml`。TASK-4.2a/b マージ済み）は既に
  `expected_exclusion_rule_ids = ["arch-hyperparameter-change"]`・
  `expected_verdict_after_exclusions = "escalate"` を宣言しており、本ルール
  `id` と一致する。

### カテゴリ 2: テスト許容誤差・アサーション条件の単独緩和（G5 根拠）

```toml
[[exclusion]]
id = "test-tolerance-loosening"
category = "test_tolerance_loosening"
description = "本番コード変更を伴わない、テストの許容誤差・アサーション条件の単独緩和"
rationale = "PoC-3 発見事項5（G5: leaky_relu 許容誤差 1e-6→1e-2 の単独緩和）が REQ-4 のゲーミング検知（本番コードとテストの同時変更のみ対象）をすり抜けて自動適用された。REQ-5 受け入れ基準2に対応。"
paths = ["**/*.rs"]
action = "human_approval"

[exclusion.match]
type = "test_assertion_relaxation_without_prod_change"
assertion_patterns = ["assert!", "abs() <", "1e-[0-9]"]
```

- **G5 フィクスチャとの対応**: `crates/guardrail/tests/fixtures/labeled-changes/
  changes/G5-test-only-loosen/change.patch` は `src/activations.rs` の
  `mod tests` 内（`leaky_relu` の既知値テスト）で `assert!((got - want).abs() <
  1e-6, ...)` を `1e-2` に緩める。この行は削除行として `assert!` と
  `abs() <` の両方に一致するため `touches_test_assertion = true`。同じ
  `change.patch` 内に `src/model.rs`（または他の非テスト領域）への変更は
  無く、変更 hunk は `activations.rs` 内 `mod tests` 行以降にのみ存在するため
  `touches_prod_logic = false`。よって `match` は成立する。同フィクスチャの
  `meta.toml`（`.../G5-test-only-loosen/meta.toml`。TASK-4.2a/b マージ済み）
  は既に `expected_exclusion_rule_ids = ["test-tolerance-loosening"]`・
  `expected_verdict_after_exclusions = "escalate"` を宣言しており、本ルール
  `id` と一致する。
- **`paths = ["**/*.rs"]` である理由**: G5 の変更箇所は `tests/` ディレクトリ
  ではなく `src/activations.rs` の**ファイル内 `mod tests` ブロック**である
  （v1 `gaming.rs` の `touches_prod_logic` がファイル単位ではなく `mod tests`
  行番号を基準にテスト領域・本番領域を判定する設計に合わせたもの）。
  そのため「テストファイルの glob」に限定した `paths`（例:
  `["**/tests/**"]`）では G5 を検知できない。`match.type` 側の判定
  （`touches_test_assertion && !touches_prod_logic`）が実質的な絞り込みを
  担うため、`paths` は全 Rust ソースを対象とする広いパターンにしている。

## 5. 後続タスクへの引き継ぎ事項

- **TASK-5.1b（#120・依存追加カテゴリの新設・human-required）**:
  REQ-1 の受け皿として「許容依存一覧の追加・更新、`Cargo.toml`／
  `Cargo.lock` の変更」カテゴリを新設する。プロダクト判断（カテゴリ名・
  対象 `paths`・match 方式の設計）を伴うため担当は人間（実装補助は
  Claude Code 可）。本イシュー（#119）では新設しない。
- **TASK-5.2（#121・除外ルールの match 方式実装・ガードレール統合）**:
  `crates/guardrail/src/policy_exclusion.rs`（想定パス。`crates/guardrail/
  src/config.rs` と同様の構成: `deny_unknown_fields`・
  `toml_lite::MAX_INPUT_BYTES` 上限・値域検証・fail-closed）を実装する。
  `match.type` ごとの判定ロジックは v1 `gaming.rs` の
  `touches_test_assertion` / `touches_prod_logic` / `changed_files` 相当を
  v2 へ移植・再利用する（関数を `pub(crate)` に昇格するか、共通ヘルパーへ
  切り出すかは実装時に判断する）。組み込み既定値との数値一致を回帰テスト
  で保証する（4 節）。
- **TASK-5.3（#125・2 層ラベルモデルの不変条件回帰テスト）**: 1 節に
  定める不変条件 (1)(2) を機械検証する回帰テストスイートを実装する。
- **TASK-5.4（#128・#129・#130・ブラインドスポット（G2/G5）回帰テストの
  移植）**: 本設計の `match` 判定が `G2-hidden-dim-increase`・
  `G5-test-only-loosen` の `change.patch` に対して実際に match することを
  確認する回帰テストを実装・移植・更新する。
- **依存クレート追加時の注意**: `glob` クレート等、パターンマッチングに
  外部クレートが必要になる場合はライセンス確認
  （[`docs/license-matrix.md`](./license-matrix.md) 更新）とセットで行い、
  [`.claude/rules/deps-policy.md`](../.claude/rules/deps-policy.md) の
  許容依存 8 区分外の追加としてユーザー承認必須とする
  （[`.claude/rules/coding-rust.md`](../.claude/rules/coding-rust.md)）。
  本イシューでは依存追加の要否を決定しない（実装時の判断）。

## 6. ガードレール判定フローとの統合点・運用フロー

想定読者は 5 節と同じ（TASK-5.1b・TASK-5.2〜5.4 の実装者）に加え、
TASK-5.2（無条件人間承認判定の統合・#121）の実装者を主対象とする。
本節は判定順序・実装配置・記録項目の**契約**を定めるものであり、
`policy_exclusion.rs` の関数シグネチャ等の実装詳細は TASK-5.2 側の
判断に委ねる（契約レベルに留める）。

### 6.1 判定フローとの統合点（判定順序契約）

- **評価位置**: `guardrail check` フローで changeset 収集後、REQ-4 の
  5 シグナル評価と**独立に**除外リスト評価（TASK-5.2・#121 実装予定の
  `crates/guardrail/src/policy_exclusion.rs`）を必ず実行する。除外リスト
  評価は他シグナルの収集成否によらずスキップしない（fail-closed。
  security.md A05）。
- **v2 での実装済み契約**: `crates/guardrail/src/decision.rs`
  （`decide` 関数）は既に `DecisionInput` に `exclusion_rule_ids`
  （match したルール `id` の一覧。空なら match なし）を持ち、**判定は
  `decide()` 1 箇所に閉じる**設計を満たしている（`decision.rs:20-56`
  モジュールコメント）。`decide()` の呼び出し元（`cli.rs`/`main.rs` の
  `check` フロー）が `decide()` の返す `Verdict` を外側でラップして
  差し替える方式は**採らない**。これは `crates/guardrail/src/exit_code.rs`
  の `GuardrailExitCode::from_verdict`（`exit_code.rs:46-51`）が「`Verdict`
  → 終了コード変換をこの 1 関数のみに閉じ込め、他経路から 0 を返せない
  ようにする」のと同じ設計思想であり、判定の迂回経路を作らないという
  security.md A08 の要請に対応する。**本イシュー（#119）時点では
  `exclusion_rule_ids` を実際に埋める `policy_exclusion.rs` 自体が未実装
  （TASK-5.2・#121 のスコープ）であり、`decide()` は空の
  `exclusion_rule_ids` を前提に呼び出される状態のまま**である。自己修復
  ループ（`crates/self-repair`。TASK-3.1 未着手）と `guardrail::decide`
  の統合は、`docs/guardrail-self-repair-cli.md` 3.4 節が規定する「lib
  直接呼び出し」設計に従い、`decide()` を必ず経由する契約となる予定
  （v1 `tools/self-repair/src/judge.rs` 相当。v2 では `crates/self-repair`
  側の実装として移植する）。
- **判定順序契約**（v2 実装済み。`decision.rs:28-56` モジュールコメントの
  現行契約）:
  1. **却下（Reject・最優先）**: build/test/clippy のいずれかが
     `GateSignal::Failed`。
  2. **除外リスト・エスカレーション**: 除外ルールのいずれかに match
     （`Reason::ExclusionMatch`。機械可読 ID `"policy_exclusion_match"`）
     → REQ-4 のエスカレーション条件（3.）・自動適用条件（5.）を
     **参照せず**無条件で `Verdict::Escalate` とする。
  3. **機械エスカレーション**: 現行の REQ-4 条件（変更行数上限・公開 API
     破壊・ゲーミング疑い・ベンチ劣化）。
  4. **ゲート未全通過エスカレーション**: `Reject`（1.）に該当せず、かつ
     `GateSignals::all_passed()` が `false`（`Skipped` を含み `Failed` は
     含まない状態）の場合も `Verdict::Escalate` とする（「Skipped gates
     allow auto-apply」への回帰防止として既に実装済みの規則であり、本節
     はこれを判定順序契約の一部として明文化するのみで意味を変更しない）。
  5. **自動適用**: 上記 1〜4 いずれにも該当しない場合のみ。
- **Reject との優先関係**: REQ-5 受け入れ基準（本ファイル 1 節）の
  「機械判定の結果によらず無条件で人間承認必須」の趣旨は「人間承認なしの
  取り込み（自動適用）を絶対に許さない」ことであり、`Reject`（取り込み
  拒否）はこの趣旨に抵触しない。したがって除外 match は `Reject` を
  `Escalate` へ格下げしない（判定順序 1. を最優先のまま維持する。安全側
  ＝機械ゲートの厳しさを弱めない設計）。ただし **`Reject` 確定時も除外
  match の評価・記録は必ず行い**、`Report`（6.2 節）に match 事実を残す
  （「結果によらず評価される」ことの担保・判断根拠の追跡可能性。
  security.md「取り込み判断の根拠を追跡可能にする」）。v2 の
  `decision.rs` は `Decision::exclusion_rule_ids`（`decision.rs:48-53`）
  として非空の `exclusion_rule_ids` を却下時も含め常に記録する契約を
  既に満たしている。なお G2/G5 は「REQ-4 の機械的シグナルを全て回避し
  自動適用される」ブラインドスポットの定義上、通常は全ゲート通過
  （`Reject` にならない）ケースであるため、`Reject` と除外 match が競合
  するのは本来カテゴリ外のコーナーケースである。
- **fail-closed の適用範囲**: `decision.rs` 現行方針（`match` は網羅列挙
  とし `_ =>` ワイルドカードを使わない。`decision.rs:55-56`）を、除外
  ルール `category` / `match.type` の分岐にも同様に適用する（4 節「未知の
  `category` / `match.type` / `action` はパースエラー」と合わせ、判定
  ロジック側でも未知 variant を黙って「match なし」に落とさない）。

### 6.2 出力契約への反映

- **v2 での実装済み契約**: `Report`（`crates/guardrail/src/report.rs`。
  `Report::new` が `Decision` からフィールドを構築する現行実装）は既に
  `applied_exclusion_rule_ids: Vec<String>` フィールドを持つ
  （`report.rs:58, 104, 129`）。`decision.exclusion_rule_ids()` の値を
  そのまま転記する（`report.rs:129`）。本イシュー（#119）時点では
  match 結果自体が常に空（`policy_exclusion.rs` 未実装）のため、この
  フィールドが実際に非空となるのは TASK-5.2（#121）以降である。
- 終了コードは既存の **10（Escalate）をそのまま使い、新しい終了コードは
  追加しない**（`crates/guardrail/src/exit_code.rs:30-61` の
  `GuardrailExitCode` 契約〈0=AutoApply/10=Escalate/20=Reject/1=
  InternalError/2=UsageError〉を不変に保つ。CI・自己修復ループ側の分岐
  変更が不要になる）。
- 自己修復ループ試行ログ（TASK-3.x・未着手で実装予定の
  `docs/self-repair-log-format.md` 相当）の `attempt` レコードは、除外
  match によるエスカレーションも同じ `escalated` variant として記録
  される想定（6.1 節のとおり `decide()` の出力を経由するため、ループ側
  から見て機械エスカレーションと区別する追加フィールドは必須ではない）。
  除外 match の詳細（ルール `id` 等）をログ本文にも残すかどうか・
  追加フィールドの要否は自己修復ループ実装（TASK-3.x）・TASK-5.2 実装時
  に確定する。

### 6.3 自己修復ループとの統合

- `docs/guardrail-self-repair-cli.md` 3.4 節が規定する「lib 直接呼び出し」
  設計に従い、`crates/self-repair`（TASK-3.1 未着手）は
  `guardrail::DecisionInput::new` → `guardrail::decide` を必ず経由し、
  返る `Verdict` の 3 variant を網羅列挙して自己修復側の判定へ変換する
  契約を持つ想定（v1 `tools/self-repair/src/judge.rs` の
  `GuardrailAdoptionJudge::judge` 相当）。6.1 節の統合方式（`decide()`
  内への除外リスト統合）が既に v2 で成立しているため、自己修復ループ側は
  追加改修なしで除外リストの無条件エスカレーションが効く構造となる
  （`Verdict::Escalate` → 人間承認待ちとしてループが停止する動作）。

### 6.4 除外リスト更新の運用フロー（人間承認必須）

- **変更対象の定義**: `policy-exclusion.toml`（本イシュー #119 で移植・
  TASK-5.1b・#120 でカテゴリ追加）と組み込み既定値（コード内。TASK-5.2・
  #121 で実装。4 節参照）の両方。両者は回帰テストで一致を保証するため、
  **必ず同一 PR で同時に変更**する。
- **標準フロー**:
  1. **提案**（Issue 起票または PR）: 変更理由・PoC/実測根拠（新規
     ブラインドスポットの再現手順等）を本文に明記する。Issue 起票自体も
     ユーザー承認が必須（CLAUDE.md「ユーザー承認フロー」）。
  2. **security-auditor 監査**を並列実施する（`.claude/rules/security.md`
     「依存追加・ガードレール・CI/hooks 変更を含む PR は security-auditor
     の監査を並列で実施する」）。
  3. **人間（ユーザー）承認の記録**（PR 上の明示的 approve、または
     承認内容が特定できるコメント）。承認記録のない除外リスト変更 PR は
     マージしない。
  4. マージ後、TASK-5.3〜5.4 系回帰テスト（5 節参照）が green であることを
     確認する。
- **ルール追加（新ブラインドスポット発見時）**: REQ-5 の設計思想（機械判定
  では技術的に検知不能な変更カテゴリをポリシーで補う。
  `docs/spec/04-requirements.md:123`）に基づき、新たなブラインドスポットが
  発見された場合は本ファイル（除外リスト）側へルールを追加する運用とし、
  ラベル付きフィクスチャ（`crates/guardrail/tests/fixtures/labeled-changes/`、
  `known_blindspot = true`）と回帰テストの追加を**セットで必須**とする
  （フィクスチャなしのルール追加は「match することを検証できないルール」
  であり受け入れない）。
- **ルール緩和・削除**: 追加より厳格に扱う（回帰テストの期待値変更を
  伴うため）。上記標準フローに加え、緩和によって再露出するブラインド
  スポット（対応する G2/G5 等のフィクスチャが再び自動適用されないか）の
  評価を PR 本文に必須記載する。
- **禁止事項の再掲**: AI（Claude Code・自己修復ループ）単独での除外
  リスト変更を禁止する（`.claude/rules/delegation-impl.md` 禁止事項・
  security.md）。自己修復ループの修正候補差分が `policy-exclusion.toml`
  自体を変更する場合、変更カテゴリ（`architecture_change` /
  `test_tolerance_loosening` / 依存変更〈TASK-5.1b・#120 で追加予定〉の
  いずれか、あるいはスキーマ自体の変更）を問わず人間承認が必須である
  （除外リストの自己書き換えによる判定迂回を防止するため。REQ-5・
  security.md A08 の核心）。この禁止は 6.1 節の `decide()` 統合設計とは
  独立の運用ルールであり、`policy_exclusion.rs` 自体の変更検知は現時点
  でコードによる強制を持たない（次項参照）。
- **スコープ外事項**: 「`policy-exclusion.toml`（および組み込み既定値の
  変更）を CI で機械検出し、承認記録の有無を検査するガード」は、本設計
  では将来課題としての言及に留める。実装する場合は
  [`out-of-scope-tracking.md`](../.claude/rules/out-of-scope-tracking.md)
  の手順に従い、ユーザー承認のうえ別 Issue として起票する。

---

## 出典

- `docs/spec/04-requirements.md:121-136`（REQ-5）
- `docs/spec/05-tasks.md:188-213`（TASK-5.1〜5.4）
- `docs/spec/03-poc/poc-3-guardrail-validity/README.md:154`（発見事項5）
- `docs/spec/06-roadmap.md:104-124`（ガードレール移植・ポリシー除外リスト
  移植のマイルストーン記述、新ブラインドスポット発見時の除外リスト追加運用）
- `crates/guardrail/src/config.rs`（設定ファイルの流儀・fail-closed 前例）
- `crates/guardrail/src/toml_lite.rs:38`（`MAX_INPUT_BYTES` 自作パーサ上限）
- `crates/guardrail/src/decision.rs:1-56, 61-64`（判定順序契約・`decide` 本体）
- `crates/guardrail/src/exit_code.rs:30-61`（`from_verdict` への変換一元化・終了コード契約）
- `crates/guardrail/src/report.rs:46-129`（`Report` 構造体・`Report::new`・`applied_exclusion_rule_ids`）
- `docs/guardrail-self-repair-cli.md` 2.4 節・3.4 節（設定ファイル位置づけ・lib 直接呼び出し設計）
- `crates/guardrail/tests/fixtures/labeled-changes/changes/G2-hidden-dim-increase/`
- `crates/guardrail/tests/fixtures/labeled-changes/changes/G5-test-only-loosen/`
- v1 出典（アーカイブ済み・改修して再利用元）: `rust-ai-library-v1/policy-exclusion.toml`・
  `rust-ai-library-v1/docs/policy-exclusion-design.md`・
  `rust-ai-library-v1/crates/guardrail/src/gaming.rs`・
  `rust-ai-library-v1/tools/self-repair/src/judge.rs`
- `guardrail.toml`（文書スタイル前例）
- `.claude/rules/security.md`・`.claude/rules/delegation-impl.md`（人間承認必須・AI 単独変更禁止の運用根拠）
- `.claude/rules/deps-policy.md`（許容依存 8 区分・依存追加時のユーザー承認）
- `CLAUDE.md`（ユーザー承認フロー: ガードレール閾値・ポリシー除外リストの変更、Issue 起票）
