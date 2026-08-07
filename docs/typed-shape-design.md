# 型レベル shape safety（const generics）設計 — TASK-10.1a

## 0. 位置づけ

本文書は `docs/spec/05-tasks.md` TASK-10.1（REQ-10・Could・M3）を 3 分割した最初のサブタスク（イシュー #98）の成果物であり、**型設計（適用・非適用の境界を含む）の確定のみ**を対象とする。`crates/` 配下の Rust コードは本イシューでは変更しない（並行イシューとのファイル競合防止。`.claude/rules/delegation-impl.md`）。

> **実装後追補（#101・TASK-10.2）**: 後続イシュー #99（実装）では、本節以降の §4 が示す型名（`TypedMatrix`／`TypedVector`／`Linear`）とは異なる型名（`FixedVec`／`FixedMat`／`BatchedFeatures`）・API 形状（GEMM 等の実計算を `kernel` クロージャへ委譲する方式）が採用された。`tensor-core` クレートが計算カーネルを持たずバックエンド抽象層に実計算を委ねる設計（REQ-1/REQ-2）が、本文書執筆時点ではまだ `typed` モジュールの API 形状に反映されていなかったための変更である。設計名と実装名の対応表・変更理由は `docs/typed-shape-limitations.md` 第 1 節を参照。以下の §4 は設計時点の記録としてそのまま残す。

- 実装（型レベル API の追加）: 後続イシュー #99
- コンパイル成功／失敗テストの整備: 後続イシュー #100
- 限界の正式ドキュメント化（`docs/typed-shape-limitations.md`、TASK-10.2）: 後続イシュー #101

## 1. 目的・spec 根拠

- **REQ-10**（`docs/spec/04-requirements.md:211-224`）: アーキテクチャ上固定される次元（レイヤーの in/out features 等）についてのみ、コンパイル時 shape 検証を自作テンソル `Tensor<T>` の上の限定レイヤーとして提供する。
- **TASK-10.1**（`docs/spec/05-tasks.md:317-322`）: PoC-7 の試作方針（const generics）を自作テンソル型（`tensor-core`）の上で実装する。アーキテクチャ上固定される次元に限定し、バッチ次元・シーケンス長など実行時変動次元には適用しない。
- **PoC-v2-1「テンソル型の設計判断」**（`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md` 実施内容 1 節の表）: 自作テンソル基盤の shape 検査方式は**実行時検査**（`Tensor::new` が `ShapeError` を返す）に確定済み。理由は safetensors／ONNX からロードする重みの shape は実行時にしか決まらない（モデル構造が動的）ため。型レベル検査は「コンパイル時に shape が既知な層（固定サイズの Linear 層等）」に限定して後続 PoC・実装フェーズで追加検討する、と明記されている。
- **v1 PoC-7**（`docs/spec/03-poc/poc-7-type-safety/README.md`、v1 の Burn 互換 API 上の実測。参考実績として位置づけ）: 誤りパターン P1〜P6 を列挙し、動的 shape 方式（方式X）はコンパイル時検出 1/6（16.7%）、型レベル方式（方式Y）は 6/6（100%、うち P6 連結軸は部分保証）を達成した。バッチ次元を型に載せると可変バッチ推論と衝突する点、値の偶然一致（P4'/P5'）は型でも実行時でも検出できない構造的な残存リスクである点が、基盤非依存の設計方針として v2 に引き継がれている。
- **`docs/public-api-design.md:178-188`（§2.5）**: 基盤 `Tensor<T>` は rank を型パラメータに含めない（実行時 rank）と決定済み。同 `:547`「rank 型載せの最終確定」が「TASK-10.x 実装時に本方針で問題がないか再確認すること」と申し送っており、本文書の第 2 節・第 3 節でその再確認結果を記録する。

## 2. 適用境界（受け入れ条件の中核）

### 適用する

- **アーキテクチャ上コンパイル時に固定される次元**: 全結合層の in/out features、固定サイズの重み行列・バイアスの各次元。モデル構造（レイヤー定義）を書く時点で値が確定しており、safetensors／ONNX ロード後も変わらない次元に限る。

### 適用しない

| 対象 | 理由 |
|------|------|
| バッチサイズ | v1 PoC-7 で可変バッチ推論との衝突が実証済み（`docs/spec/03-poc/poc-7-type-safety/README.md`）。REQ-10 受け入れ基準（`04-requirements.md:218`）が明示的に除外を要求 |
| シーケンス長 | バッチサイズと同様、実行時に変動する次元であり型に載せない |
| safetensors／ONNX から実行時ロードする動的 shape 経路 | PoC-v2-1 の確定事項（実行時検査採用の理由そのもの）。モデル構造が動的なため、ロード直後の shape はコンパイル時に既知ではない |
| 基盤 `Tensor<T>` の rank | `docs/public-api-design.md` §2.5 の決定（実行時 rank）を本文書で再確認し維持する（第 3 節参照）。rank は基盤テンソル型自体の設計で担保し、型レベル shape レイヤー側では扱わない |
| デバイス整合性の型タグ | REQ-10 受け入れ基準（`04-requirements.md:221`）により概念実証止まりと位置づけ、実運用のマルチバックエンド環境での妥当性は本要件では未検証・将来検討事項とする。v1 PoC-7 の `Dev` ゼロサイズマーカー型（`Cpu0`/`Cpu1`）は概念実証としては有効だったが、複数バックエンド・複数物理デバイスでの検証はスコープ外のまま持ち越す |

## 3. §2.5「rank を型に載せるか」の再確認結果

`docs/public-api-design.md:547` の申し送りに対する再確認結果: **基盤 `Tensor<T>` を実行時 rank のまま据え置く決定を維持する**。

理由:
- 型レベル shape レイヤーの適用対象は「in/out features 等の固定次元」に限定されており、rank そのものを型パラメータ化する必要は生じない。第 4 節で定義する `TypedMatrix`／`TypedVector` は rank を固定 2／固定 1 として型自体に埋め込む（後述）ため、基盤 `Tensor<T>` 側の rank 総称化は不要。
- safetensors／ONNX ロード直後の `Tensor<T>` は依然として実行時 rank・実行時 shape のままであり、型レベル世界への持ち込みは第 4 節の境界 API（`try_from_tensor`）を経由する。基盤層の rank を型に載せると、この境界 API の入力型自体が変わってしまい PoC-v2-1 の確定事項と矛盾する。

## 4. 型設計

### 4.1 固定次元ラッパー型

```rust
/// 固定次元 2 階テンソル（行列）のラッパー。内部に `Tensor<T>` を保持する
/// newtype とし、生成時に一度だけ shape を検証してから以降の演算では
/// 型検査のみで shape 一致を保証する（PoC-7 方式Y の継承。TASK-10.1a）。
///
/// ROWS/COLS はアーキテクチャ上固定される次元（全結合層の in/out features
/// 等）を表す。バッチ次元・シーケンス長はここに含めない（第 2 節）。
pub struct TypedMatrix<T: Element, const ROWS: usize, const COLS: usize> {
    inner: Tensor<T>,
}

/// 固定次元 1 階テンソル（バイアス等）のラッパー。
pub struct TypedVector<T: Element, const N: usize> {
    inner: Tensor<T>,
}
```

- `Element` は `crates/tensor-core/src/element.rs` の既存トレイトをそのまま使う。
- `inner` は private とし、`TypedMatrix`／`TypedVector` の外から直接 shape を書き換えられないようにする（invariant: `inner.shape() == [ROWS, COLS]` を型の生存期間中つねに満たす）。

### 4.2 動的↔型付き世界の境界 API

```rust
impl<T: Element, const ROWS: usize, const COLS: usize> TypedMatrix<T, ROWS, COLS> {
    /// 実行時 shape を型引数 `ROWS`/`COLS` と突合してから型付き世界へ
    /// 持ち込む。safetensors/ONNX ロード直後の `Tensor<T>`（実行時
    /// shape、PoC-v2-1 確定）を、コンパイル時に既知の固定次元層
    /// （Linear 層の重み等）へ橋渡しする唯一の入口とする。
    ///
    /// 失敗時は既存の `ShapeError` をそのまま返す（型レベル専用の
    /// エラー型を新設しない。`ops_shape.rs` の実行時検査と同じ語彙を
    /// エラー処理側が共有できるようにするため）。
    pub fn try_from_tensor(tensor: Tensor<T>) -> Result<Self, ShapeError> { .. }

    /// 型付き世界から動的 `Tensor<T>` へ脱出する。バッチ次元を持つ
    /// 実際の演算（forward 等）は動的 `Tensor<T>` 側で行うため、
    /// 型付きレイヤーの外に出る経路を必ず用意する。
    pub fn into_inner(self) -> Tensor<T> { .. }
    pub fn as_tensor(&self) -> &Tensor<T> { .. }
}
```

- `try_from_tensor` が `security.md` の「外部フォーマットパースは長さ・形状の検証を先に行う」方針と整合する接続点になる。safetensors／ONNX ロード経路は、動的 `Tensor<T>` を経由したあと `try_from_tensor` で一度だけ型検査済みの世界に入る。
- `TypedVector` にも同名の `try_from_tensor`／`into_inner`／`as_tensor` を用意する。

### 4.3 演算シグネチャ

```rust
impl<T: Element, const M: usize, const K: usize> TypedMatrix<T, M, K> {
    /// 内側次元 `K` の一致を型検査に委ねる（v1 PoC-7 方式Y の継承。
    /// `docs/spec/03-poc/poc-7-type-safety/README.md` P1 相当の
    /// 「形状不一致の行列積」をコンパイル時に排除する）。
    pub fn matmul<const N: usize>(&self, rhs: &TypedMatrix<T, K, N>) -> TypedMatrix<T, M, N> { .. }
}
```

- 内部実装は既存 `ops_shape.rs` の `matmul_out_shape` 等の実行時検査関数、または `Tensor<T>::matmul` 相当をそのまま呼び出す（`inner` は既に shape 保証済みのため、実行時検査は理論上冗長になるが、二重防御として維持するか #99 の実装時に判断する。本文書ではどちらでもよいとし、性能への影響が無視できる場合は維持を推奨する）。

### 4.4 バッチ次元ハイブリッド構成

固定次元レイヤー（例: `Linear<T, const IN: usize, const OUT: usize>` 相当。#99 のスコープ）の forward は、入力・出力を**動的 `Tensor<T>`（batch × IN → batch × OUT）で受ける**。重み（`TypedMatrix<T, IN, OUT>`）・バイアス（`TypedVector<T, OUT>`）は型付きだが、forward の引数・戻り値のバッチ次元は動的のままとする。

```rust
// 概念例（#99 で確定させる。ここでは境界の設計のみを示す）
pub struct Linear<T: Element, const IN: usize, const OUT: usize> {
    weight: TypedMatrix<T, IN, OUT>,
    bias: TypedVector<T, OUT>,
}

impl<T: Element, const IN: usize, const OUT: usize> Linear<T, IN, OUT> {
    /// `input` は動的 `Tensor<T>`（shape `[batch, IN]`）。features 次元
    /// （`IN`）のみ実行時に型パラメータと突合し、バッチ次元は型に
    /// 載せない（第 2 節。v1 PoC-7 可変バッチ推論との衝突を回避）。
    pub fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, ShapeError> { .. }
}
```

- `forward` の内部では `input` の features 次元（末尾軸）が `IN` と一致するかを実行時 `ShapeError` で検査する。これは「型で守れるのは重み・バイアスの固定次元同士の整合であり、実行時に来る入力のバッチ次元・features 次元適合は実行時検査に委ねる」という REQ-10 の限定適用方針そのものである。

### 4.5 stable Rust 制約

- `generic_const_exprs`（nightly 限定機能）は使用しない。したがって `ROWS + ROWS2` のような型レベル算術を要求する演算（例: 連結 `cat`）は本レイヤーの設計対象外とする。
- 連結等、型レベル算術が必要な演算は動的 `Tensor<T>` へフォールバックする（v1 PoC-7 の P6 が「連結軸サイズ `M1+M2` は stable Rust の const generics では型に表現できず動的 `Tensor<B,2>` にフォールバックする」と結論づけたのと同じ制約。`docs/spec/03-poc/poc-7-type-safety/README.md` P6 行）。非連結軸のみ型で保証する部分保証の余地は #99 の実装時に判断する。

## 5. 記述コスト方針（REQ-10 受け入れ基準）

型レベル API はオプトインの追加レイヤーであり、既存 `Tensor<T>` API・compat 層（REQ-9）の既定経路は動的 shape のまま維持する（公開 API 非破壊。`docs/public-api-design.md` §2.5 の決定と整合）。`TypedMatrix`／`TypedVector`／`Linear<T, IN, OUT>` 等は使いたい場合にのみ選択する構成とし、型注釈の増加を全面強制しない（REQ-10 `04-requirements.md:220`）。

## 6. モジュール配置案

- `crates/tensor-core/src/typed.rs`（または要素が増えた場合は `typed/` ディレクトリへ分割）として #99 で追加する。
- `crates/tensor-core/src/lib.rs` から `pub mod typed;` として re-export する。既存モジュール（`tensor`・`ops_shape`・`broadcast`・`element`・`error`）とは独立したオプトインレイヤーであることが `lib.rs` のクレートドキュメント（`//!`）から分かるようにコメントを追加する。

## 7. テスト戦略の設計指針（#100 への引き継ぎ）

- **第一候補**: rustdoc の ` ```compile_fail ` doctest。追加依存ゼロで、公開 API ドキュメントの一部としてコンパイル失敗例を残せる。
- **`trybuild`**（v1 PoC-7 で採用実績あり、`.stderr` ゴールデンファイルで実際のコンパイルエラーを固定化できる）は許容依存 8 区分（`.claude/rules/deps-policy.md`）に含まれず、**追加にはユーザー承認が必須**。本イシュー（自動運転・依存追加不可）では採否を確定させない。採用可否の判断は #100 側でユーザー承認フローに乗せること。
- コンパイル成功ケース（正しい型付けが実際に通ること）の検証は通常の `#[test]` で行う（型レベル検査そのものは実行時に何も観測できないため、コンパイルが通ること自体がテストの主目的になる）。

## 8. 既知の限界（#101 への引き継ぎ、見出しレベルのみ）

詳細は TASK-10.2（#101）の `docs/typed-shape-limitations.md` に委譲する。本文書では次の見出しのみを列挙する。

- 値の偶然一致（P4'／P5' 相当）は型でも実行時でも検出不能（構造的な残存リスク。v1 PoC-7 実測）。
- P6（連結）は非連結軸のみ型で保証する部分保証であり、連結軸サイズの静的検証は stable Rust では不可（第 4.5 節）。
- rank の保証は基盤 `Tensor<T>` の実行時検査に委ね、型レベル shape レイヤーでは扱わない（第 3 節の再確認結果）。
- デバイス整合性の型タグは概念実証止まりであり、実運用のマルチバックエンド環境での妥当性は将来検討事項（第 2 節）。

## 9. セキュリティ・ガードレール接続（OWASP Top 10 観点）

- **A03 インジェクション**: `try_from_tensor` は safetensors／ONNX 由来の実行時 shape を型付き世界へ持ち込む唯一の入口であり、失敗時は既存 `ShapeError` を返す。外部フォーマットパースの長さ・形状検証を先に行う方針（`.claude/rules/security.md`）と整合する。
- **A08 ソフトウェア・データ整合性**: 型レベル shape 検証は AI 自律メンテナンスの静的検査ゲート（`cargo build` の型検査フェーズ）を強化する設計であり、ガードレール（REQ-3〜5）の検査対象を増やす方向にのみ働く。ガードレール閾値・テスト許容誤差の変更はスコープ外（本イシューでは触れない）。
- 依存追加なし。トークン・実環境パス等の秘密情報は本文書に含まない。
