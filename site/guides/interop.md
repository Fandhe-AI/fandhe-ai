# ONNX・safetensors 相互運用

## サポート境界（現状の重要な制約）

**ONNX import は `fandhe_ai::interop::onnx`（`OnnxModel`／`OnnxValue`／
`OnnxError`）として、safetensors save／load は `fandhe_ai::interop::
safetensors`（`LoadError`／`SaveError`／`load_safetensors_f32`／
`load_safetensors_f32_from_bytes`／`require_keys`／`save_safetensors_f32`／
`save_safetensors_f32_to_bytes`）として、いずれも `fandhe-ai` から
公開されています。** ONNX export は現時点で公開されていません
（`onnx-interop` クレート。公開名 `fandhe-ai-onnx-interop`。依存解決の
ための公開であり直接利用はサポート対象外）。`fandhe-ai` が唯一の
サポートされる公開 API 面であるという原則（[API Reference](/api/)参照）
に従うと、`onnx-interop` を直接 `use` する経路はサポート対象外の内部
利用にあたります。ONNX export を利用者向けに公開する入口の新設は
本ページのスコープ外です。

### ONNX import の最小コード例

```rust
use std::collections::HashMap;

use fandhe_ai::interop::onnx::{OnnxModel, OnnxValue};
use fandhe_ai::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = OnnxModel::from_path("model.onnx")?;

    let mut feeds = HashMap::new();
    feeds.insert(
        "input".to_string(),
        OnnxValue::F32(Tensor::<f32>::new(vec![0.0, 1.0], &[1, 2])?),
    );

    let outputs = model.run(feeds)?;
    if let OnnxValue::F32(y) = &outputs["output"] {
        println!("{:?}", y.as_slice());
    }
    Ok(())
}
```

**推論専用・ホスト CPU 実行のみ**（`OnnxModel::run` は `BackendOps`／
`Device` を経由しないため GPU 実行にはなりません）で**autograd 未接続**
（入出力は [`Tensor`](/api/)であり `Var` ではないため勾配は取れません）。
`OnnxValue::F16` は `half::f16` を素通しするため、扱うには利用者側が
`half` クレートへ直接依存する必要があります。数値契約は
「`abs_err/(|ref|+1e-6) <= 1e-3`」という ONNX 固有の判定式であり、
[数値一致契約](/guides/numerical-parity/)のバックエンド間統一複合判定とは
別指標です。以下は残りの設計解説です（ONNX export 部分は動くコード例を
用意していません）。

### safetensors の最小コード例

`compat::Sequential::state_dict`／`load_state_dict`（`docs/compat-
callbacks-design.md`）と組み合わせて、モデルの重みをファイルへ保存・
復元できます。

```rust
use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::safetensors::{load_safetensors_f32, save_safetensors_f32};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Sequential::new().add_linear(4, 8, /* seed = */ 1)?;
    let path = Path::new("weights.safetensors");
    save_safetensors_f32(path, &model.state_dict())?;

    let mut restored = Sequential::new().add_linear(4, 8, /* seed = */ 2)?;
    restored.load_state_dict(load_safetensors_f32(path)?)?;
    Ok(())
}
```

保存 → 読込は bit 完全一致を保証し、`load_state_dict` は不足キー・
形状不一致を型付き `Err` で fail-closed に拒否します（strict・
two-pass アトミック契約。既存 `Sequential` の shape が保存前と異なる
場合はパラメータが一切変更されません）。

## safetensors: ワイヤフォーマット処理のみ

`safetensors` クレートは**ワイヤフォーマットの読み書きのみ**に使い、
テンソルへのマッピング（`fandhe_ai_tensor_core::Tensor` への変換）は自作
しています。外部クレートに委ねる範囲を「バイト列の構造化」だけに
絞り込むことで、テンソル抽象自体は完全自作コアの方針
（`.claude/rules/coding-rust.md`）と両立させています。

## dtype は F32 限定・キー充足検査は fail-closed

保存・読込の双方で dtype は `F32` のみをサポートします。`Tensor<f32>`
にのみ型付けされた関数群のため型レベルでも保証されますが、読込側は
さらに実行時にも dtype を検査し、F32 以外は `UnsupportedDtype` として
明示的にエラーにします（無言でスキップして後続処理を進めることは
しません）。同様に、期待するキー集合に対する充足検査も無言 skip せず
fail-closed に倒す設計です（`.claude/rules/security.md` A03
「外部フォーマットパースは長さ・形状の検証を先に行う」）。

保存 → 読込のラウンドトリップは bit 一致を保証します。これは
[数値一致契約](/guides/numerical-parity/)の「同一実装同士の比較には
許容誤差を持ち込まない」という考え方と同じ設計思想です。

## ONNX: prost 手書き derive

ONNX の protobuf デコードには `prost` を使いますが、`prost-build`
（ビルド時に `protoc` を要求する）は使わず、**手書き derive** で
対応しています。ビルド時の外部ツール依存（`protoc` のインストール
要求）を避け、CI・開発環境のセットアップを単純に保つための判断です。

## 外部フォーマットの入力検証を先に行う

safetensors・ONNX（prost）いずれの外部フォーマットパースも、長さ・
形状の検証をデータ変換より先に行います（OWASP Top 10 の A03
インジェクション対策と同じ考え方: 外部入力を信頼せず、想定外の
サイズ・形状のデータで内部状態が壊れる前に検査で弾きます。
`.claude/rules/security.md`）。
