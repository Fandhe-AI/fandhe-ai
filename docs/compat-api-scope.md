# compat API 層の対象範囲（TASK-9.2b）

イシュー #96（親: #94 = TASK-9.2、ルート: Phase 4 #71）の成果物。
REQ-9「Python 慣習寄りの互換 API 層」（`docs/spec/04-requirements.md:200-209`）の
受け入れ基準のうち「互換 API 層の対象範囲は本フェーズでは初期方針（活性化関数・
基本レイヤーの薄いラッパー）に限定し、全方位の API 網羅は対象外とすること」
（`04-requirements.md:206`）を明文化する。TASK-9.2 の成果物として
`docs/compat-api-scope.md` が明示されている（`docs/spec/05-tasks.md:307-311`）。

`docs/public-api-design.md` は「compat 層は自作コアの素の公開 API とは別
レイヤーであること」という境界のみを 1 章で明記し（同ファイル 6・13・556 行目）、
詳細範囲は REQ-9 系後続タスクへ委譲している。本文書はその受け皿である。

## 1. 対象範囲（in scope）

REQ-9 の受け入れ基準（`docs/spec/04-requirements.md:203-206`）・
TASK-9.1／TASK-9.2（`docs/spec/05-tasks.md:299-311`）に基づき、以下に限定する。

- **`compat::array()`**: numpy 慣習のテンソル生成関数（`np.array()` 相当）。
  自作テンソル（`tensor-core`）の上に構築する（TASK-9.2a・#95）
- **`compat::Sequential`**: Keras 慣習のレイヤー積み上げビルダー
  （`.add_linear()`／`.add_relu()` 等のメソッドチェーン、
  `docs/spec/04-requirements.md:205`）。自作 NN モジュール（`autodiff::nn`）の
  上に構築する（TASK-9.2a・#95）
- **基本レイヤー・基本活性化関数の薄いラッパー**（TASK-9.1）:
  - Linear 層（TASK-9.1a・#91）
  - ReLU・Sigmoid・Tanh の 3 活性化関数（TASK-9.1b・#92 で実装済み。
    `crates/autodiff/src/nn/activation.rs`）

対象範囲はこの 3 種（配列生成関数・Sequential ビルダー・基本レイヤー／活性化の
薄いラッパー）に限定する。

## 2. 対象外（out of scope）

- 全方位の Python API 網羅を明示的に対象外とする（`04-requirements.md:206`・
  `05-tasks.md:309`）。具体的には以下を含むが、これらに限らない。
  - numpy の ufunc 群・ファンシーインデックス・ブロードキャスト以外の
    高度な配列操作の網羅
  - Keras の全レイヤー種別（Conv 系・RNN 系・Embedding 等）・callbacks・
    `fit()`／`compile()` 等の学習ループ API の網羅
  - Softmax（損失関数 CrossEntropy・#189 と密結合のため対象外。
    `crates/autodiff/src/nn/activation.rs` 冒頭コメント）・GELU 等の
    追加活性化関数（必要になった時点で後続イシューに切り出す）
  - pandas 等、numpy／Keras 以外の Python ライブラリとの互換
- 対象外要望が生じた場合の受け皿は 2 通り。
  - 実装リポ側で追跡が完結する事項: `.claude/rules/out-of-scope-tracking.md`
    の規約に沿って Issue で追跡する
  - 受け入れ基準・REQ-9 自体の改定が必要な事項: 正本 spec リポジトリ
    （`Fandhe-AI/rust-ai-library-spec`）側での対応をユーザーに提案する
    （`docs/spec/` は本リポでは編集しない）

## 3. 設計原則

- **薄いラッパーに徹する**: compat 層は数値計算ロジックを自ら持たず、
  自作コア（`tensor-core`／`autodiff`）の API への委譲のみを行う
  （REQ-9・`.claude/rules/coding-rust.md`「互換 API 層は自作コアの上の
  薄いラッパーに徹する」）。`crates/autodiff/src/nn/activation.rs` の
  `Relu`／`Sigmoid`／`Tanh` は各 `forward` が対応する `Var` メソッドを
  呼ぶだけの実装であり、この原則の実例である
- **v1 試作は参考実績にとどめる**: v1 の `compat::array()`／
  `compat::Sequential`（PoC-1）・`add_leaky_relu`（PoC-2）は Burn 前提の
  試作であり、「薄いラッパー層が構造的に成立すること」の参考実績として
  残すが、v2（完全自作コア）の受け入れ基準達成の直接的な実測根拠には
  用いない（`04-requirements.md:207`）。自作コア確定後の再実装・再検証を
  要する
- **PoC-v2-6 を v2 実例として参照する**: `Mlp::from_safetensors`
  （`docs/spec/03-poc/poc-v2-6-interop/code/rust/src/mlp.rs`）は自作テンソル
  上に構築された薄い互換層の v2 実例である。numpy／Keras 慣習の本格的な
  互換 API 層そのものではないが、自作コア上でも薄いラッパーが成立し
  得ることを示す傍証として位置づける（`04-requirements.md:200-209`）

## 4. 実装配置（本文書執筆時点）

- `compat::array()`／`compat::Sequential` 自体の配置クレート・公開名は
  TASK-9.2a（#95、本文書執筆時点で未マージ）で確定する。9 クレート構成
  （`tensor-core`・`autodiff`・`backend-cpu`・`backend-cuda`・`backend-metal`・
  `onnx-interop`・`guardrail`・`self-repair`・`bench-harness`）に compat 専用
  クレートは存在しないため、既存クレート（`autodiff` 等）内のモジュールと
  なる見込みだが、本文書では確定済みの事実として記述しない
- Linear 層（TASK-9.1a・#91）も本文書執筆時点で未マージであり、同様に
  実装詳細を先取りしない
- 活性化関数（ReLU・Sigmoid・Tanh）は TASK-9.1b（#92）で実装済み
  （`crates/autodiff/src/nn/activation.rs`）。共通 `Module` trait は
  同イシュー時点では未定義で、`compat::Sequential` 設計時（TASK-9.2a・#91／
  #95）に確定する予定（`crates/autodiff/src/nn/mod.rs`）

## 5. 範囲拡張の手続き

本文書が定める対象範囲（1 節）の拡張は、以下いずれかの手続きを経ることを
必須とする。AI 自律メンテナンス（self-repair ループ）による無断拡大は
行わない（REQ-5／`.claude/rules/security.md` の自己修復ガードレールと整合）。

1. 正本 spec リポジトリ（`Fandhe-AI/rust-ai-library-spec`）側での REQ-9
   受け入れ基準の改定
2. 本リポジトリのユーザー承認を得たうえでの Issue 起票・本文書の更新

## 6. 出典一覧

| 出典 | 内容 |
|------|------|
| `docs/spec/04-requirements.md:200-209` | REQ-9 概要・受け入れ基準・関連 PoC |
| `docs/spec/05-tasks.md:299-311` | TASK-9.1（基本 NN モジュール）・TASK-9.2（compat 再実装・対象範囲明文化） |
| `docs/spec/03-poc/poc-v2-6-interop/code/rust/src/mlp.rs` | `Mlp::from_safetensors`（自作コア上の薄い互換層の v2 実例） |
| `docs/public-api-design.md:6,13,556` | compat 層と自作コア素の公開 API の境界記述 |
| `crates/autodiff/src/nn/activation.rs` | TASK-9.1b（#92）実装済みの ReLU／Sigmoid／Tanh |
| `crates/autodiff/src/nn/mod.rs` | compat 層が積む「レイヤー」モジュール群の入口コメント |
| `.claude/rules/coding-rust.md` | 「互換 API 層は自作コアの上の薄いラッパーに徹する（REQ-9）」 |
| `.claude/rules/out-of-scope-tracking.md` | 対象外事項の Issue 追跡規約 |
| イシュー #91（TASK-9.1a・Linear 層） | 本文書執筆時点で未マージ |
| イシュー #92（TASK-9.1b・基本活性化関数群） | クローズ済み |
| イシュー #94（TASK-9.2・親） | 本文書執筆時点で未マージ |
| イシュー #95（TASK-9.2a・compat::array／Sequential 実装） | 本文書執筆時点で未マージ |
| イシュー #96（TASK-9.2b・本文書） | 本イシュー |
