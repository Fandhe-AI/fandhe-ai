# API Reference

## サポート境界

`fandhe-ai` クレートが**唯一のサポートされる公開 API 面**です。
`fandhe-ai-tensor-core`・`fandhe-ai-autodiff`・`fandhe-ai-backend-cpu`・
`fandhe-ai-backend-cuda`・`fandhe-ai-backend-metal` は内部クレートであり、
これらを `fandhe-ai` を経由せず直接利用することはサポート対象外です。

`fandhe-ai-autodiff` の `Tape::new_with_ops` や各バックエンドの実装型は、Rust の
可視性としては `pub`（クレート単体のドキュメント上は到達可能）な箇所が
ありますが、サポート境界上は内部 API です。技術的に `pub` であることと、
利用者向けにサポートされる公開面であることは区別しています。

利用者が使うことを想定する入口は次の 4 つです。

- `fandhe_ai::tape()` / `fandhe_ai::tape_for(Device)`: composition root。
  `Device` 識別子を受け取り、対応するバックエンドへ結線した `Tape` を
  構築します（詳細は [Getting Started](/getting-started/)
  のバックエンド切替節）
- `fandhe_ai::compat::{array, Sequential}`: numpy/Keras 慣習の互換 API 層
- `fandhe_ai::optim`: `Sgd`／`AdamW`／`clip_grad_norm`／`LrScheduler` 等の
  optimizer 群の再エクスポート
- `fandhe_ai::DeviceParamStore` / `Tape::step_device_param_store`:
  学習ループのパラメータ更新をデバイス上に常駐させる経路

`fandhe-ai` は任意の `BackendOps` 実装を注入できる公開 API をあえて設けて
いません。`Tape` は `fandhe-ai` 側の newtype でラップされており、
`var`／`backward` の 2 メソッドのみを公開しています。

## ページ一覧

- [compat API](/api/compat/): `compat::array`・
  `compat::Sequential` の要点
- [guardrail / self-repair CLI](/api/cli/): ガードレール
  判定・自己修復ループの CLI コマンド要点
