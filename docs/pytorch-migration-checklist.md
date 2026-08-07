# PyTorch 資産移行チェックリスト

<!--
本書の役割: PyTorch で学習・保存した資産（重み・モデル）を本ライブラリへ
移行する利用者向けの手順書。正本は REQ-7（`docs/spec/04-requirements.md`）
であり、本書は TASK-7.5（`docs/spec/05-tasks.md`）の成果物として、
実装済み API（`crates/onnx-interop/`・`crates/tensor-core/`）と
突き合わせた具体的チェックリストを提供する。
-->

## 本書の位置づけ

本ライブラリは `burn-store`／`burn-onnx` に依存せず、safetensors（フォーマットパーサのみ許容）・
ONNX（`prost` による protobuf デコードのみ許容）を自作テンソル・自作 ONNX インタープリタへ
自前で取り込む（REQ-7、`docs/spec/04-requirements.md`）。要件・受け入れ基準の正本は REQ-7 であり、
本書は移行作業者がその場で参照できるよう実装 API 名・ファイルパスを突き合わせた手順書に徹する。
本書と REQ-7 の記載が食い違う場合は REQ-7 を正とする。

## 経路の選択

| 経路 | 用途 | 移行対象 |
|------|------|---------|
| safetensors 経路 | 重みのみ移行し、モデル構造は Rust 側で再実装する | `state_dict()` 等でエクスポートした `.safetensors` ファイル |
| ONNX 経路 | グラフ構造ごと移行する（構造再実装が不要） | `torch.onnx.export()` 等でエクスポートした `.onnx` ファイル |

両経路とも `burn-import`／`burn-store`／`burn-onnx` は使用しない（`.claude/rules/deps-policy.md` 依存禁止リスト・CI 機械検査対象）。

## safetensors 経路チェックリスト

- [ ] `onnx_interop::st_load::load_safetensors_f32()`（または
      `load_safetensors_f32_from_bytes()`）でロードする（`crates/onnx-interop/src/st_load.rs`）。
      dtype は F32 のみを受け付け、それ以外は即エラーとなる。データ変換前に
      ヘッダ・レイアウト整合検査 → dtype 検査 → データ長と shape 要素数積の
      一致検査 → `Tensor::new` の shape 検査、の順で行われる（OWASP A03 対策、
      `.claude/rules/security.md`）
- [ ] **`nn.Linear.weight` の `[out_features, in_features]` 転置は呼び出し側が
      `tensor_core::Tensor::transpose_2d()`（`crates/tensor-core/src/tensor.rs:275`）を
      明示的に呼ぶ**。`st_load` は暗黙アダプタを持たない設計であり、`burn-store` の
      `PyTorchToBurnAdapter` 相当の暗黙変換は一切行わない（REQ-7 受け入れ基準、
      `st_load.rs` モジュール冒頭ドキュメント）
- [ ] **`onnx_interop::require_keys()`（クレートルートへ再エクスポート、
      `crates/onnx-interop/src/require_keys.rs`）で期待キー集合の充足を検査する**。
      不足キーは全件収集して `LoadError` で返される（無言 skip 禁止。PoC-v2-6 の
      詰まりポイント対策）
- [ ] 逆方向（保存）は `onnx_interop::st_save`（`crates/onnx-interop/src/st_save.rs`）を用いる。
      `st_load` と対称の契約であり、キー・形状・値をそのまま書き出す（転置・キーリネームは行わない）。
      dtype は `Tensor<f32>` のみ（型レベルで保証）
- [ ] ラウンドトリップ（save → load → save）の bit 一致は `tests/st_roundtrip.rs`
      （特殊浮動小数点値・各 rank）で検証済み。チェックポイント保存・再開の等価性は
      `tests/st_checkpoint.rs` を参照

> 注意: `st_load::LoadError` と `require_keys::LoadError` は #73／#74 それぞれの独立実装として
> 別名前空間で並存する（`crates/onnx-interop/src/lib.rs` 冒頭ドキュメント参照。統合整理は
> #74 側でスコープ外事項として追跡中）。

## ONNX 経路チェックリスト

- [ ] `onnx::proto::ModelProto`（`crates/onnx-interop/src/onnx/proto.rs`。`prost` 手書き
      derive によるデコードで `protoc` に依存しない） → `onnx::graph::build_graph()`
      （`crates/onnx-interop/src/onnx/graph.rs`） → `onnx::interp::run()`
      （`crates/onnx-interop/src/onnx/interp.rs:793`）の 3 段でグラフを実行する
- [ ] **動的差し替え可能性**: `interp::run()` はランタイムインタープリタ方式であり、
      モデルファイルをランタイム読込する。v1 の「ビルド時 codegen のため再ビルド必須」
      という制約は v2 では解消されており、実行時のモデルファイル差し替えが可能である
      （REQ-7 受け入れ基準「動的差し替え」）
- [ ] 対応オペ範囲: 初期 8 オペ（`Gemm`/`Relu`/`Sigmoid`/`Shape`/`Gather`/`Unsqueeze`/
      `Concat`/`Slice`）に加え、Transformer 対応に必要な残 14 種別（`Add`/`Cast`/`Constant`/
      `Div`/`Erf`/`LayerNormalization`/`MatMul`/`Mod`/`Mul`/`Reshape`/`Softmax`/`Sqrt`/`Squeeze`/
      `Transpose`）が実装済み（合計 22 種別、`crates/onnx-interop/src/ops/`）。`transformer.onnx`
      の end-to-end 推論は `tests/onnx_transformer_e2e.rs`（TASK-7.4a・#301）で実測済み
- [ ] 動的境界 Slice パターン（v1 `burn-onnx` の失敗パターン、`tracel-ai/burn#5295`）は
      自前インタープリタで対応済み（`tests/onnx_slice_dynamic_bounds.rs`）
- [ ] 未対応の `op_type` を含むグラフを渡すと `InterpError::UnsupportedOp` で拒否される
      （黙って無視されない）

## 数値一致の確認

PyTorch 参照値との比較には REQ-7 の事前固定判定式を用いる。

```
abs_err / (|ref| + 1e-6) ≤ 1e-3
```

実測根拠: safetensors 直読み・ONNX インタープリタとも最大相対誤差 0.000000
（PoC-v2-6、`docs/spec/03-poc/poc-v2-6-interop/README.md`「計測結果」節）。

> **この判定式は REQ-2 のバックエンド間数値一致 OR 複合判定
> （「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」）とは別指標であり、混同・単独緩和は禁止**。
> いずれの許容誤差も変更にはユーザー承認が必須である（`.claude/rules/coding-rust.md`・
> `.claude/rules/security.md`）。

## codegen 方式（将来オプション）のセキュリティ要件

ONNX 経路はランタイムインタープリタ方式を主経路とし、ビルド時コード生成（codegen）は
「モデル確定後のビルド時最適化」の将来オプションと位置づける（REQ-7）。**現状 codegen 方式は
未実装**である。将来実装する場合は、以下を必須要件とする（緩和不可。PoC-v2-6 の
`sanitize_ident` 設計準拠、OWASP A03 対策）。

- [ ] 生成コードに絶対パスを埋め込まない
- [ ] ONNX ノード識別子は英数字・`_` 以外の文字を置換する `sanitize_ident` によるサニタイズを
      経てからでなければ、Rust 識別子として使用しない

## 禁止事項

- `burn-import`／`burn-store`／`burn-onnx` は使用しない（`.claude/rules/deps-policy.md`
  依存禁止リスト、CI で機械検査）
- `docs/spec/`（正本 submodule）の受け入れ基準・判定式を、本書を含むいかなる文書側でも
  弱める記述に置き換えない。変更が必要な場合は正本リポジトリ
  （`Fandhe-AI/rust-ai-library-spec`）側で対応する
