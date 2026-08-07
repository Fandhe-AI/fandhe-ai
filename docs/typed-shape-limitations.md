# 型レベル shape safety の限界 — TASK-10.2

## 0. 位置づけ・spec 根拠

本文書は `docs/spec/05-tasks.md` TASK-10.2（REQ-10・Could・M3、前提タスク TASK-10.1）の成果物であり、`docs/typed-shape-design.md`（TASK-10.1a・#98）§8「既知の限界」が見出しレベルで引き継いだ 4 項目を詳細化する。TASK-10.1 は #99（実装・`crates/tensor-core/src/typed.rs`）・#100（テスト整備・`crates/tensor-core/tests/typed_shape.rs`）を経て完了済み（#97 クローズ）であり、本文書はその実装済み API を対象に限界を記録する。

- **REQ-10**（`docs/spec/04-requirements.md:211-224`）: 型レベル shape safety の限定適用に関する要件。受け入れ基準のうち本文書が対応するのは次の 2 点。
  - `:222` 「値が偶然一致するケースは型でも実行時でも検出できない」という型安全性の限界をドキュメントに明記すること
  - `:219` rank（次元数）の検証は自作テンソル型自体の設計で担保し、rank を型に載せるかを自作 API 設計時に決定してドキュメントに記録すること
- **TASK-10.2**（`docs/spec/05-tasks.md:324-329`）: 上記 2 点に加え、rank 検証を自作テンソル型がどう保証するか（v1 は Burn 標準 API 前提だったが v2 では設計自体で保証する必要がある点）の明記を成果物 `docs/typed-shape-limitations.md` に求める。
- **関連文書**: `docs/typed-shape-design.md`（TASK-10.1a 設計文書）§2（適用境界）・§3（rank を型に載せるかの再確認）・§4.5（stable Rust 制約）・§8（本文書への引き継ぎ）。`docs/public-api-design.md` §2.5「REQ-10 との関係: rank を型に載せるか」（基盤 `Tensor<T>` の rank 実行時化を決定した文書）。
- **PoC 根拠**: PoC-v2-1（`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md`「テンソル型の設計判断」表、実施内容 1 節）が自作テンソル基盤の shape 検査方式を実行時検査に確定。v1 PoC-7（`docs/spec/03-poc/poc-7-type-safety/README.md`）は Burn 互換 API 上（v1）の実測であり、本文書では**参考実績**として位置づける（`04-requirements.md:213` の指定通り、v2 の実測ではない）。

## 1. 実装 API の全体像と設計名→実装名の対応

`crates/tensor-core/src/typed.rs` は基盤 `Tensor<T>`（実行時 shape・rank、PoC-v2-1 確定）の上に積む、アーキテクチャ上コンパイル時に固定される次元（全結合層の in/out features 等）に限定したオプトインレイヤーである。

| 型 | 役割 | 境界コンストラクタ |
|---|---|---|
| `FixedVec<T, const N: usize>` | 1 次元固定長テンソル（bias 等）。内部 shape は常に `[N]` | `from_tensor` |
| `FixedMat<T, const IN: usize, const OUT: usize>` | 2 次元固定次元テンソル（全結合層の重み等）。内部 shape は常に `[IN, OUT]` | `from_tensor` |
| `BatchedFeatures<T, const F: usize>` | バッチ入り特徴量テンソル `[batch, F]`。batch は意図的に型パラメータへ含めない実行時次元（第 3 節） | `from_tensor` |

いずれの `from_tensor` も実行時 `Tensor<T>` の rank・shape を const パラメータと突合してから受け入れる fail-closed な境界コンストラクタであり、rank 不一致は `ShapeError::RankMismatch`、shape 不一致は `ShapeError::ShapeMismatch` で拒否する（`crates/tensor-core/src/typed.rs:36-44`、`crates/tensor-core/src/error.rs:19-82`）。unchecked に構築する経路は設けていない。

`BatchedFeatures` は `matmul_with`/`add_bias_with` を提供する。いずれも実計算を呼び出し元が渡す `kernel: impl FnOnce(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>, ShapeError>` クロージャへ委譲し、`kernel` 実行後の出力 shape（特徴次元・batch 次元とも）を再検査してから型付きに包む（`crates/tensor-core/src/typed.rs:178-305`）。`tensor-core` は計算カーネルを持たず GEMM 等の実装は `backend-cpu` 等が担う（`docs/public-api-design.md` §4）ため、型付き演算は「基盤 `Tensor<T>` を計算する関数を型付きで包む」ジェネリック合成として設計されている。

### 設計名→実装名の対応表（#101 コメント記載の残課題への対応）

`docs/typed-shape-design.md` §4（TASK-10.1a、#98／PR #283）が示した設計時の型名・API 形状は、実装フェーズ（#99／PR #286）で以下のとおり変更された。イシュー #101 のコメントが「TASK-10.2 の限界文書化、または `typed-shape-design.md` の実装後追補で解消する」と提案していた命名・型構成の不一致を、本表で解消する。

| 設計文書（§4） | 実装（`typed.rs`） | 変更点 |
|---|---|---|
| `TypedMatrix<T, ROWS, COLS>` | `FixedMat<T, IN, OUT>` | 型名変更。型パラメータの意味論も「行数・列数」から「全結合層の入力次元・出力次元」へ具体化 |
| `TypedVector<T, N>` | `FixedVec<T, N>` | 型名変更のみ、意味論は同一（1 次元固定長） |
| `Linear<T, IN, OUT>`（weight・bias を保持する構造体） | `BatchedFeatures<T, F>` + `FixedMat`/`FixedVec` を呼び出し側が個別に保持する構成 | `Linear` 構造体は実装されず、`BatchedFeatures::matmul_with`/`add_bias_with` が入力側（バッチ付き特徴量）を主語とする API に置き換わった |
| `try_from_tensor` | `from_tensor` | メソッド名変更のみ、境界検査の意味論は同一 |
| 演算メソッドが内部で GEMM 相当を直接実行する想定（§4.3 `matmul`） | `matmul_with`/`add_bias_with` が `kernel` クロージャを受け取り実計算を委譲 | `tensor-core` がカーネルを持たない設計（`docs/public-api-design.md` §4）に合わせた変更。呼び出し元がバックエンド固有の GEMM/加算実装を注入する |

上記変更は、`tensor-core` クレートが計算カーネルを持たずバックエンド抽象層（`backend-cpu`/`backend-cuda`/`backend-metal`）に実計算を委ねるという REQ-1/REQ-2 のクレート分割方針が、設計文書執筆時点（#98）ではまだ `typed` モジュールの API 形状に反映されていなかったために生じた。機能面（shape 不一致のコンパイルエラー実証・検証テスト green）は #99・#100 とも受け入れ条件を満たしており、本表により文書間の整合を確保する。

## 2. 限界 1: 値の偶然一致は型でも実行時でも検出できない

REQ-10 受け入れ基準（`04-requirements.md:222`）が明記を要求する構造的な限界。

v1 PoC-7（`docs/spec/03-poc/poc-7-type-safety/README.md`「誤りパターン別の検出分類表」）の実測で、以下 2 パターンは動的形状方式（方式X）・型レベル方式（方式Y）のいずれでも検出できないことが確認されている。

- **P4'**（バッチ数と特徴量数がたまたま同じ、例: batch=8=features）: `bias_add_batch_feature_confusion_with_equal_sizes_is_not_caught` 相当。バッチ次元用のテンソルを特徴量次元用の引数に渡しても、サイズが数値的に一致していれば shape 検証を通過してしまう。
- **P5'**（正方行列で転置忘れ）: `linear_forward_missing_transpose_square_case_still_compiles_but_is_semantically_wrong` 相当。重み行列が正方行列の場合、転置を忘れても shape 上は matmul が成立してしまう。

PoC-7 の静的検出可能割合の実測は、方式X（Burn 動的形状）が 6 パターン中 1 パターン（P3、rank 取り違え）＝ **16.7%**、方式Y（型レベル shape 試作）が 6 パターン中 6 パターン（P6 は連結軸を除く部分保証）＝ **100%** だった（同 README「静的検出可能な割合」節）。これは「サイズが実際に異なる」典型的なバグに対する評価であり、P4'・P5' の偶然一致サブケースは別枠として除外されている。

この限界は本モジュール（`FixedVec`/`FixedMat`/`BatchedFeatures`）にもそのまま当てはまる: `from_tensor` の境界検査、`matmul_with`/`add_bias_with` の内側次元・出力 shape 検査は、いずれも shape（次元のサイズ）の数値的な一致しか見ておらず、渡されたテンソルが意味的に正しい役割（重みか bias か、転置済みか否か）を持つかどうかまでは検証しない。**「型安全 = バグゼロ」ではなく、「サイズが異なる場合の発覚タイミングを実行時からコンパイル時（もしくは fail-closed な実行時検査）へ前倒しできる」という限定的な効果**である点を、本モジュールの利用者は前提として理解する必要がある。

## 3. 限界 2: 型レベル算術は stable Rust で表現不可

`generic_const_exprs`（`M1 + M2` のような型レベル算術を可能にする機能）は 2026-08 時点でも nightly 限定の unstable 機能であり、本モジュールは stable Rust のみで構成する方針を採る（`typed.rs:21-23`、`docs/typed-shape-design.md` §4.5）。

このため、出力 shape が入力の算術結果になる演算（例: 連結 `cat`）は型で完全には表現できない。v1 PoC-7 の P6（連結時の非連結軸の不一致）は、非連結軸のサイズ（`N`）のみ型で保証し、連結軸サイズ（`M1+M2`）は動的 `Tensor<T>` へフォールバックするハイブリッド構成で妥協している（同 README「誤りパターン別の検出分類表」P6 行）。本モジュールは連結演算自体を実装しておらず、必要になった場合も同様の部分保証（非連結軸のみ型検査、連結軸は動的 shape）に留まる設計上の制約として引き継ぐ。

## 4. rank 検証の保証方法（v1 → v2 の変更点）

REQ-10 受け入れ基準（`04-requirements.md:219`）・TASK-10.2 の内容（`05-tasks.md:325`）が明記を求める、rank 検証の担保方法の記録。

- **v1**: Burn `Tensor<B, D>` の型パラメータ `D`（rank）が既にコンパイル時に rank を保証しており、追加実装は不要という前提だった（v1 PoC-7「発見事項」節: 「rank（次元数）は既に Burn 標準 API で型安全」）。イシュー #40 で REQ-10 の受け入れ基準文言はこの前提が v2 では成立しないことを反映して更新済み（`05-tasks.md:537`）。
- **v2（本リポジトリの決定）**: Burn を使用しないため、Burn 標準 API による rank 保証は存在しない。代わりに rank は次の 2 段構えの**実行時検査**で保証する。
  1. 基盤 `Tensor<T>` 自体: `Tensor::new` 等のコンストラクタが `ShapeError` を返す（PoC-v2-1 確定事項、`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md`「テンソル型の設計判断」表）。
  2. `typed` モジュールの境界コンストラクタ: `FixedVec::from_tensor`／`FixedMat::from_tensor`／`BatchedFeatures::from_tensor` が、受け取った `Tensor<T>` の `rank()` を期待値（それぞれ 1／2／2）と突合し、不一致は `ShapeError::RankMismatch { expected, actual }` で拒否する（`typed.rs:64-79,106-122,146-161`）。

**rank を型パラメータとして基盤 `Tensor<T>` に載せない決定**は `docs/public-api-design.md` §2.5「REQ-10 との関係: rank を型に載せるか」で確定し、`docs/typed-shape-design.md` §3「§2.5「rank を型に載せるか」の再確認結果」で再確認・維持されている。理由は、safetensors／ONNX からロードする重みの shape・rank は実行時にしか決まらない（モデル構造が動的）ため、rank を基盤テンソル型自体の型パラメータにすると動的 shape のロード経路と両立しないことにある（PoC-v2-1 と同一の理由）。型レベル shape レイヤー（本モジュール）が対象とする `FixedVec`/`FixedMat`/`BatchedFeatures` はいずれも rank を型自体に固定 1 または固定 2 として埋め込んでいる（`FixedVec` は常に rank 1、`FixedMat`/`BatchedFeatures` は常に rank 2）ため、rank の型パラメータ化を要求せずに rank の型安全性相当の効果（誤った rank のテンソルを渡すと `from_tensor` が拒否する）を実行時 fail-closed 検査で得ている。

## 5. 適用除外（限界と区別して記載）

以下は「検出できない限界」ではなく、REQ-10 が意図的に型レベル shape の対象外と定めている項目である（`docs/typed-shape-design.md` §2「適用境界」）。

| 対象 | 適用しない理由 |
|---|---|
| バッチサイズ | v1 PoC-7 で可変バッチ推論との衝突が実証済み。const generics は値がコンパイル時に確定している必要があるため、バッチサイズごとに別の型（モノモーフィック化）が発生し可変バッチに対応できない。REQ-10 受け入れ基準（`04-requirements.md:218`）が明示的に除外を要求。`BatchedFeatures<T, F>` の第 1 軸（batch）は常に実行時次元のまま（`typed.rs:18-20`） |
| シーケンス長 | バッチサイズと同様、実行時に変動する次元であり型に載せない |
| safetensors／ONNX から実行時ロードする動的 shape 経路 | PoC-v2-1 の確定事項（実行時検査採用の理由そのもの）。モデル構造が動的なため、ロード直後の shape はコンパイル時に既知ではない。`from_tensor` 境界コンストラクタが動的世界から型付き世界への唯一の入口となる |
| デバイス整合性の型パラメータ化（デバイスタグ） | REQ-10 受け入れ基準（`04-requirements.md:221`）により概念実証止まりと位置づけ。v1 PoC-7 の `Dev` ゼロサイズマーカー型（`Cpu0`/`Cpu1`）は概念実証としては有効だったが、複数バックエンド・複数物理デバイスでの検証は本要件では未検証・将来検討事項。本モジュール（`typed.rs`）はデバイスタグを実装していない |

## 6. 二重防御との関係

`BatchedFeatures::matmul_with`/`add_bias_with` は `kernel` クロージャの出力 shape（特徴次元・batch 次元とも）を再検査してから型付きに包む（`typed.rs:178-193,253-268,290-305`。テスト: `typed.rs` 内 `matmul_with_rejects_kernel_output_shape_mismatch`／`matmul_with_rejects_kernel_output_batch_mismatch` 等、`crates/tensor-core/tests/typed_shape.rs` の外部消費者視点テスト）。

この出力再検査は、「型検査を通っている＝呼び出し元の `kernel` が正しい shape を返す」ことを型システムだけでは保証できない（`kernel` は任意のクロージャであり、誤った shape の `Tensor<T>` を返しうる）という事情に対する二重防御である。`.claude/rules/coding-rust.md`「カーネル実装の境界検査（REQ-8）」が要求する「性能下限・最適化の達成を理由に、カーネル側の手動境界チェックを省略しない」方針と同趣旨であり、**型レベル shape 検査（コンパイル時）を導入したことを理由に、実行時の出力 shape 再検査を省略しない**という設計判断を本モジュールも踏襲している。第 2 節の限界（値の偶然一致は検出不能）と合わせ、型検査は実行時検査を代替するものではなく積み重ねる防御層であることを明記する。
