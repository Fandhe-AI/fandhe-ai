# rust-ai-library

Rust 製 AI/ML ライブラリです。Burn 等の既存フレームワークに依存せず、テンソル・
autodiff・演算グラフ／カーネル融合機構・計算カーネル・バックエンド抽象層を
**完全自作コア**として実装しています。

## このライブラリの構成

内部は 10 個のクレートに分かれていますが、利用者が直接触れるのは `facade`
クレートだけです。

| クレート | 役割 |
|---|---|
| `facade` | **唯一のサポートされる公開 API 面**。composition root（`Device` → バックエンドの結線）と compat 公開面（`compat::array`／`compat::Sequential`）を提供します |
| `tensor-core`・`autodiff`・`backend-cpu`・`backend-cuda`・`backend-metal` | 内部クレート。直接利用はサポート対象外です |
| `onnx-interop`・`guardrail`・`self-repair`・`bench-harness` | 相互運用・自己修復ループ・ベンチ計測を担う内部クレート |

`tensor-core`／`autodiff`／`backend-*` の型・関数は Rust の可視性としては
`pub` な箇所がありますが、サポート境界上は内部 API です。利用者が使うことを
想定する入口は `facade::tape()`／`facade::tape_for(Device)` と
`facade::compat::{array, Sequential}` のみです。

## バックエンド

バックエンド切替は feature フラグを使わない **cfg ベース**です。CPU は常に
利用可能な既定バックエンドで、CUDA・Metal は実行時にデバイスの存在を検証し、
利用できない場合はエラーを返します（自動フォールバックはしません）。詳細は
[Getting Started](/getting-started/) を参照してください。

## 次に読むもの

- [Getting Started](/getting-started/): インストール・最小
  コード例・バックエンド切替
- [API Reference](/api/): `compat` API・guardrail／self-repair
  CLI の要点
