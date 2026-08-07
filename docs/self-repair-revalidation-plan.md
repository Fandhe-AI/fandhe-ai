# 自己修復ループ再実証 実証計画・題材選定（TASK-3.3a）

本文書は TASK-3.3（自作コア上での自己修復ループ完走の再実証）の第 1 サブタスク
TASK-3.3a の成果物である。**コード変更は含まない**。バグ修正・機能追加の 2 種別の
題材と、実証完走の判定基準を定義し、人間承認（本文書を含む PR のレビュー・マージ）
を経て確定する。

## 1. 目的・spec 根拠

- REQ-3 の v2 追加受け入れ基準（`docs/spec/04-requirements.md:96`）は「自作コア
  （REQ-1 の自作対象 7 項目）に対する自己修復ループの人間介在なし完走を新実装リポで
  再実証すること」を要求する。v2 PoC（PoC-v2-1〜v2-6）はこの再実証を対象にしていない
  ため、本タスクが必要になった。
- TASK-3.3（`docs/spec/05-tasks.md:142-147`）は上記受け入れ基準に対応する実測タスク
  であり、担当区分は「共同（実施は Claude Code、題材選定・可否判断は人間）」。本文書
  （TASK-3.3a）はそのうち題材選定・完走判定基準の合意形成を担う。
- ロードマップのリスク表（`docs/spec/06-roadmap.md:206`）は「PoC-2 実証済みのループ
  構造を踏襲し新規設計要素を最小化する」ことと「題材選定の判断待ちがクリティカルパスに
  影響しうる」ことを明記しており、本イシューを先行して確定させることが後続作業
  （#141〜#144）の着手条件になる。
- PoC-2（v1 Burn 基盤、`docs/spec/03-poc/poc-2-ai-self-maintenance/README.md`）は
  バグ修正・性能回帰・機能追加の 3 題材すべてで完走し、危険な変更 6 件を全却下（見逃し
  率 0%）した実績を持つ。本計画はそのループ構造を v2（自作コア）向けに移植する。

### イシューツリーとの対応

| イシュー | 内容 | 本文書との関係 |
|---------|------|---------------|
| #139（TASK-3.3 親） | 自作コア上での自己修復ループ完走の再実証 | 本文書はそのサブタスク #140 の成果物 |
| #140（TASK-3.3a・本イシュー） | 実証計画・題材選定（人間承認） | 本文書そのもの |
| #141 | バグ修正題材の実証実行 | 本文書 4 節の題材・5 節の完走判定基準を用いる |
| #142 | 機能追加題材の実証実行 | 同上 |
| #143 | 記録整備 | 本文書 6 節の記録様式を用いる |
| #144 | 人間評価 | 本文書 5 節の完走判定基準で判定する |

## 2. 保守対象と実証範囲

REQ-1 の自作対象 7 項目（`docs/spec/01-brainstorm.md` 自作対象一覧表）は次のとおり。

1. テンソル型（`tensor-core`）
2. autodiff（`autodiff`）
3. 演算グラフ・カーネル融合機構
4. 計算カーネル（`backend-cpu`／`backend-cuda`／`backend-metal`）
5. バックエンド抽象層
6. ONNX 演算マッピング（`onnx-interop`）
7. guardrail・self-repair 自体

2 種別の題材（4 節）が実際にカバーする項目は次のとおりで、REQ-3 の受け入れ基準
（「少なくとも 2 種別で完走」）が要求する範囲を満たす。7 項目すべてを 1 サイクルで
網羅することは要求されていないため、範囲を意図的に絞る。

| 題材 | 保守対象（自作対象 7 項目のうち） |
|------|-----------------------------------|
| バグ修正（4 節） | autodiff（活性化関数の演算グラフ構築） |
| 機能追加（4 節） | autodiff（新規演算の追加）・compat API 層（互換 API 経由の呼び出し） |
| 両題材共通 | guardrail・self-repair 自体（3 分岐判定・検証 4 ゲートを両題材が経由する） |

計算カーネル（backend-cpu 等）・onnx-interop・演算グラフ融合機構は本 2 題材の直接の
修正対象には含まれないが、活性化関数の評価は `backend-cpu` の要素毎演算カーネル
（`crates/backend-cpu/src/elementwise.rs`）を経由するため、検証 4 ゲート（build／
test／clippy／bench）を通じて間接的に検証される。

## 3. 題材選定基準

以下の 5 条件をすべて満たす題材のみを採用する。

- **(a) 機械検出・機械検証が可能**: 検証 4 ゲート（`cargo build`／
  `cargo test --release`／`cargo clippy --all-targets -- -D warnings`／
  `cargo bench`）のいずれかで検出可能であり、既知正解値テストまたはベンチ対象に
  含まれること。
- **(b) 修正差分が承認済み auto_apply 閾値内に収まりうる規模**（変更行数 200 行以内
  等）。**閾値自体は本イシューでも実証実行でも変更しない**（`guardrail.toml`
  の変更はユーザー承認必須。TASK-4.3c で確定済みの値をそのまま用いる）。
- **(c) policy-exclusion ルールの対象パス・変更種別に該当しないこと**:
  `arch-hyperparameter-change`（`crates/guardrail/src/policy_exclusion/any_diff_in_paths.rs`）
  ・`test-tolerance-loosening`（`crates/guardrail/src/policy_exclusion/mod.rs:140`）
  ・依存変更カテゴリのいずれにも該当しない。除外発火はエスカレーション終了となり
  「人間介在なし完走」（REQ-3 受け入れ基準）が成立しなくなるため（REQ-5 整合）。
- **(d) PoC-2 実証済みループ構造の踏襲**: 新規設計要素を最小化する
  （`docs/spec/06-roadmap.md:206` リスク表の対策方針）。
- **(e) 決定的シードで再現可能**: `guardrail::determinism`（`crates/self-repair/src/lib.rs`
  が再輸出）経由で同一シード→同一系列が成立すること。

## 4. 題材（推奨案＋代替案）

いずれも人間が本 PR のレビューで選択・承認する対象であり、推奨案を第一候補とする。

### 4.1 バグ修正題材（#141 用）

| | 推奨案 | 代替案 |
|---|--------|--------|
| 題材 | autodiff 活性化関数の取り違えバグ注入 | backend-cpu elementwise カーネルの符号バグ |
| 内容 | `Var::relu`（`crates/autodiff/src/var.rs:257`）の実装本体を sigmoid 相当の演算グラフにすり替える | `crates/backend-cpu/src/elementwise.rs` の符号反転バグ（labeled-changes `D4-leaky-relu-sign-bug` 類型。`crates/guardrail/tests/fixtures/labeled-changes/changes/D4-leaky-relu-sign-bug/`） |
| PoC-2 対応題材 | 題材(a)（活性化関数取り違え）の v2 移植 | D4 系の派生（PoC-3 系の労働成果を流用） |
| 選定基準との適合 | (a) 既知正解値テストで検出可能・(b) 差分小・(c) 除外リスト非該当・(d) PoC-2 構造そのまま・(e) 決定的シード不要な純粋関数 | 同左（ただし backend-cpu 側は数値経路がバックエンド固有の FMA 契約に触れるため、要修正差分の見積もりに追加調査が要る） |
| 推奨理由 | 修正対象が 1 ファイル・1 関数に閉じ、build ゲートによるハルシネーション却下の再現条件（PoC-2 private メソッド誤用検出の再現）も揃えやすい | — |

### 4.2 機能追加題材（#142 用）

| | 推奨案 | 代替案 |
|---|--------|--------|
| 題材 | `autodiff::Var::leaky_relu` の新規実装 | ELU 等、他の要素毎演算の追加 |
| 内容 | `Var::leaky_relu` の追加＋`compat::Sequential`（`crates/guardrail/tests/fixtures/labeled-changes/baseline/src/compat.rs:68` の `add_leaky_relu` 相当）への組み込み＋既知値テスト | ELU（`f(x) = x if x > 0 else alpha * (exp(x) - 1)`）の新規実装。数式が leaky_relu よりやや複雑 |
| PoC-2 対応題材 | 題材(c)（LeakyReLU 新規実装）の v2 移植 | 新規（PoC-2 に対応題材なし） |
| v2 コアでの現状 | `crates/autodiff/src/var.rs` に `leaky_relu` 未実装であることを確認済み（`grep -rn leaky_relu crates/` は `crates/guardrail/tests/fixtures/labeled-changes/` 配下のテストフィクスチャのみヒットし、`autodiff`／`backend-cpu` 本体には存在しない） | 同様に未実装 |
| 選定基準との適合 | (a)〜(e) すべて満たす。PoC-2 実測（数値精度回帰の既知正解値検出。REQ-3 受け入れ基準）の再現性が高い | 同左だが数式複雑度がやや高く (d) の「新規設計要素の最小化」に反する |
| 推奨理由 | PoC-2 題材(c) と同一演算のため実測比較が可能。guardrail の labeled-changes フィクスチャ（`S2-gelu-add`・`D4-leaky-relu-sign-bug`・`G5-test-only-loosen`）が同一演算を既に扱っており、既知値・許容誤差の参照実装が手元にある | — |

## 5. 完走判定基準

#144 の人間評価は以下の基準で判定する。すべて満たした場合のみ「人間介在なし完走」
（REQ-3 受け入れ基準）と認める。

1. `self-repair run --kind <bug-fix|feature-addition>`（`docs/guardrail-self-repair-cli.md`
   §3.1 の値をそのまま使用）の **1 回起動・追加の人間入力なし**で終了コード `0`
   （`Verdict::AutoApply`。§3.5）に到達すること。
2. 検証 4 ゲート（build／test --release／clippy -D warnings／bench）全通過。
   `guardrail` の 3 分岐判定を **lib 直接呼び出しで経由し、迂回経路がない**こと
   （`crates/self-repair/src/lib.rs` のコメント方針・`.claude/rules/security.md`
   A08 整合）。
3. `--max-attempts` 上限内で完走すること（提案値 5。§3.1 の `NonZeroU32` 制約に整合。
   数値自体は TASK-3.1 側で最終確定するため、本文書の提案値は暫定とする）。
4. JSON Lines ログ（試行回数・所要時間・判断根拠）が
   `self-repair verify-log`（§3.2）のハッシュチェーン検証を通過すること。
5. ベンチ劣化中央値が承認済み閾値内（TASK-4.3c で確定済みの `guardrail.toml` の値。
   5 回計測の中央値採用・単発計測禁止。閾値自体は変更しない）。
6. 判定レポート JSON の `signal_source` フィールド（§2.1）が `"measured"` であること
   （`--signals` は CI 契約検証専用であり実証には使用しない。§1.2・7 節参照）。

## 6. 記録様式（#143 との接続）

- LoopReport JSON・JSON Lines ログの保存先は `docs/self-repair-revalidation/`
  配下を提案する（例: `docs/self-repair-revalidation/bug-fix/`・
  `docs/self-repair-revalidation/feature-addition/`）。実際のディレクトリ構成は
  #143 側で確定する。
- `signal_source == "measured"` のレポートのみを実証結果として採用する
  （`"injected"` は CI 契約検証パス専用。§1.2・2.1 節）。

## 7. 前提条件

実証実行（#141・#142）は以下のタスク完了後に着手する。

- TASK-3.1（#131）: `self-repair` クレートの骨格実装。本文書執筆時点で
  `crates/self-repair/src/lib.rs` は `guardrail::determinism` の再輸出のみで、
  クレート本体（検出・修正生成・ループ制御）は未実装。
- TASK-3.2（#136）: 検証 4 ゲートのベンチ計測系の `bench-harness` への付け替え。

いずれも本文書執筆時点で open。TASK-3.3a（本イシュー）は文書のみの成果物のため
先行着手できるが、#141・#142 の着手は上記 2 タスクの完了を前提とする。

## 8. 承認記録

| 項目 | 内容 |
|------|------|
| 承認者 | （PR マージ時に記入） |
| 承認日 | （PR マージ時に記入） |
| 承認対象 | バグ修正題材（4.1 節）・機能追加題材（4.2 節）・完走判定基準（5 節） |

本文書を含む PR のレビュー・マージが、TASK-3.3a の受け入れ条件
「2 種別の題材と完走判定基準が承認済み」の人間承認に当たる。
