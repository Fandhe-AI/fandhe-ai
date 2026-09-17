# イシュー #1955 実測記録先（facade `nn::rnn` の CUDA／Metal 実機 parity）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10・
Apple Silicon いずれの実機への到達手段もないため、**実測値は一切
含まれていない**。本ディレクトリは実行コマンド・保存すべきログ
一覧・事前登録判定規則のみを提供し、実測は該当実機を持つセッション
（Mac／GB10 セッション）へ申し送る。

## 目的

イシュー #1955「`nn::rnn`（`Rnn`／`Lstm`／`Gru`）を
`fandhe_ai::nn::rnn` として素の再エクスポートで公開する」の facade
経由 CUDA／Metal 実機 parity 記録先。CPU 上の facade 経由・内部
クレート直接呼び出しの bit 完全一致は本 PR に含まれる
`crates/facade/tests/nn_rnn_facade_bit_identity.rs` の通常（非
`#[ignore]`）テストで既に検証済み（Linux CI で実行される）。本
ディレクトリが対象とするのは同ファイル末尾の `#[ignore]` テスト
（`metal_rnn_forward_matches_cpu`／`metal_rnn_backward_dweight_ih_
matches_cpu`／`cuda_rnn_forward_matches_cpu`／`cuda_rnn_backward_
dweight_ih_matches_cpu`）のみ。

## 実行コマンド

```bash
# Metal 実機（Apple Silicon 上でのみコンパイル・実行される）
cargo test -p fandhe-ai --release --test nn_rnn_facade_bit_identity -- --ignored --nocapture metal_

# CUDA 実機（DGX Spark GB10 等）
cargo test -p fandhe-ai --release --test nn_rnn_facade_bit_identity -- --ignored --nocapture cuda_
```

`make test-ignored-metal-facade`／`make test-ignored-cuda` の対象
ターゲットに `nn_rnn_facade_bit_identity` が含まれる場合はそれ
経由でもよい（各 Makefile ターゲットの実対象一覧を確認すること）。

## 保存すべきログ

- `metal-rnn-parity-ignored.log`: 上記 Metal コマンドの生出力
- `cuda-rnn-parity-ignored.log`: 上記 CUDA コマンドの生出力
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA・Metal バージョンの要約程度に留める）

## 事前登録判定規則

- **REQ-2 統一複合判定**（相対誤差 1e-3 未満 または 絶対誤差 1e-5
  未満。`fandhe_ai_backend_cpu::parity::assert_parity` を使用）で
  forward（`h_n`）・backward（`weight_ih` の勾配）とも fail 0 件で
  あること。CPU（`fandhe_ai::tape()`）を基準に Metal／CUDA
  （`fandhe_ai::tape_for(Device::Metal)`／`tape_for(Device::Cuda(0))`）
  を比較する。GEMM カーネル自体が CPU／Metal／CUDA で異なるため
  bit 完全一致は主張しない（CPU 上の facade↔内部クレート直接呼び出し
  の bit 完全一致とは別軸の検証）
- run-to-run 決定性（同一入力での複数回実行が同一結果になること）
  は本テスト自体では明示検証しないが、FAIL の再現性確認に用いる
- 上記が FAIL の場合は `Tape::rnn_forward_seq` 等の委譲経路自体は
  変更していない（`&self.0` を渡すだけの薄い委譲。イシュー #1955）
  ため、原因は内部クレート `fandhe_ai_autodiff::nn::rnn` 側（#1647）
  または各バックエンドのゲート pointwise カーネル側にある可能性が
  高く、本イシューの再オープンではなく該当バックエンド実装側の
  issue へ切り出すことを検討する
