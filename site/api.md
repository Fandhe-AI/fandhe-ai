# API Reference

## サポート境界

`facade` クレートが**唯一のサポートされる公開 API 面**です。
`tensor-core`・`autodiff`・`backend-cpu`・`backend-cuda`・`backend-metal` は
内部クレートであり、これらを `facade` を経由せず直接利用することは
サポート対象外です。

`autodiff` の `Tape::new_with_ops` や各バックエンドの実装型は、Rust の
可視性としては `pub`（クレート単体のドキュメント上は到達可能）な箇所が
ありますが、サポート境界上は内部 API です。技術的に `pub` であることと、
利用者向けにサポートされる公開面であることは区別しています。

利用者が使うことを想定する入口は次の 2 つです。

- `facade::tape()` / `facade::tape_for(Device)`: composition root。
  `Device` 識別子を受け取り、対応するバックエンドへ結線した `Tape` を
  構築します（詳細は [Getting Started](/getting-started/)
  のバックエンド切替節）
- `facade::compat::{array, Sequential}`: numpy/Keras 慣習の互換 API 層

`facade` は任意の `BackendOps` 実装を注入できる公開 API をあえて設けて
いません。`Tape` は `facade` 側の newtype でラップされており、
`var`／`backward` の 2 メソッドのみを公開しています。

## ページ一覧

- [compat API](/api/compat/): `compat::array`・
  `compat::Sequential` の要点
- [guardrail / self-repair CLI](/api/cli/): ガードレール
  判定・自己修復ループの CLI コマンド要点
