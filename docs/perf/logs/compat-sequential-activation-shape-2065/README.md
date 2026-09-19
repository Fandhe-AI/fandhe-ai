# イシュー #2065 実測記録先（`compat::Sequential` の活性化・形状層追加）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には CUDA（DGX Spark
GB10）・Metal（Apple Silicon）実機への到達手段がないため、**実測値は
一切含まれていない**。本ディレクトリは実行コマンド・保存すべきログ
一覧・事前登録判定規則のみを提供し、実測は GB10／Mac 実機を持つ
セッションへ申し送る。

## 目的

イシュー #2065「`compat::Sequential` の活性化・形状層追加」で新設した
6 メソッド（`add_softmax`／`add_log_softmax`／`add_gelu`／
`add_gelu_tanh`／`add_softplus`／`add_flatten`）のうち、CPU 上の正しさ
は `crates/facade/tests/compat_sequential_activation_shape.rs`（Linux
実行可能）で既に検証済み。本記録先は CUDA／Metal 実機上での forward
数値一致（`Softmax`／`LogSoftmax`／`Gelu`／`GeluTanh`／`Softplus` の
既存カーネル。#1594／#1713 で実装済み）・`Flatten`（算術を含まない
view 演算のため bit 完全一致契約）の実機確認を対象とする。新規カーネル
は追加していない（`Flatten` は `Var::reshape` への委譲のみ）。

## 実行コマンド

```bash
# CUDA（DGX Spark GB10 等）
cargo test -p fandhe-ai --release --test compat_sequential_activation_shape_backend_parity -- --ignored --nocapture cuda_

# Metal（Apple Silicon。macOS 限定でコンパイルされる）
cargo test -p fandhe-ai --release --test compat_sequential_activation_shape_backend_parity -- --ignored --nocapture metal_
```

## 保存すべきログ

- `compat_sequential_activation_shape_backend_parity-cuda.log`: 上記
  CUDA 実行の生出力（3 テスト: `cuda_flatten_softmax_matches_cpu`・
  `cuda_log_softmax_gelu_softplus_matches_cpu`・
  `cuda_flatten_only_bit_exact`）
- `compat_sequential_activation_shape_backend_parity-metal.log`: 上記
  Metal 実行の生出力（同型 3 テスト）
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・macOS／Xcode バージョンの要約程度に留める）

## 事前登録判定規則

- **`Flatten` 単独（`*_flatten_only_bit_exact`）**: CPU 参照実装
  （`Sequential::predict`）と対象デバイスの `forward` 出力が **bit
  完全一致**すること（算術を含まない純粋な view 演算のため。
  `assert_eq!` による厳密比較）。
- **`Softmax`／`LogSoftmax`／`Gelu`／`GeluTanh`／`Softplus` を含む
  混在モデル（`*_flatten_softmax_matches_cpu`・
  `*_log_softmax_gelu_softplus_matches_cpu`）**: `fandhe_ai_backend_
  cpu::parity::assert_parity`（REQ-2 統一複合判定。相対誤差 1e-3 未満
  または絶対誤差 1e-5 未満）を満たすこと。既存カーネル（#1594／#1713）
  自体の parity は各実装済み issue で別途実測済みのため、本記録は
  `compat::Sequential` 経由の合成（`Flatten` 併用）が数値を崩さない
  ことの確認に限る。
- tolerance・baseline は不変。既存 `#[ignore]` 群（`compat_sequential_
  layers_backend_parity.rs` 等）の非後退確認は本イシューの必須事項
  ではない（新規カーネルを追加していないため）。
