# イシュー #1766 実測記録先（CUDA Conv2d im2col／col2im）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10 実機
への到達手段（`docs/real-hardware-verification-env.local.md`・
`CUDA_NODE` 環境変数・`~/.ssh/config` のいずれも確認できず）がないため、
**実測値は一切含まれていない**。本ディレクトリは実行コマンド・保存
すべきログ一覧のみを提供し、実測は GB10 実機を持つセッションへ申し送る
（`docs/perf/logs/cuda-gemm-batched-1716/README.md` と同型の運用）。

> **実測の正式な受け皿はイシュー #1771**（`docs/perf/logs/conv-
> realdevice-1771/`）。#1766〜#1769 4 イシューの実行手順・判定規則を
> 統合したランブックを参照すること。

## 目的

イシュー #1766「CUDA Conv2d forward／backward（im2col＋GEMM）」の実機
正しさ検証記録先。本イシューは正しさ・数値契約のみを対象とし（性能
最適化・デバイス常駐化はスコープ外・`docs/conv-ops-design.md` §16
「引き継ぎ」）、以下の `#[ignore]` テストを GB10 実機で実行して green
を確認することが目的。

## 実行コマンド

```bash
# 1) crates/backend-cuda の im2col／col2im bit 同一・parity（形状網羅。
#    重なり窓・dilation・groups／depthwise・padding のみの窓・座標
#    アンダーフロー形状・N=0・256 ブロック境界をまたぐ numel・NaN／
#    ±inf／−0.0 通過・非 contiguous view 入力・run-to-run bit 同一）
cargo test -p fandhe-ai-backend-cuda --release --test im2col_col2im_parity -- --ignored --nocapture

# 2) crates/facade の conv2d forward／backward REQ-2 複合判定
#    （d_input／d_weight／d_bias）
cargo test -p fandhe-ai --release --test conv2d_backend_parity -- --ignored --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）にも上記が
自動的に含まれる。

## 保存すべきログ

- `im2col_col2im_parity-ignored.log`: 上記 1) の生出力（形状ごとの
  bit 一致・run-to-run 決定性の判定内訳）
- `conv2d_backend_parity-ignored.log`: 上記 2) の生出力
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **im2col bit 同一**: CPU 参照実装（`backend-cpu::im2col::im2col`）と
  byte 単位完全一致すること（算術を含まない純粋コピー演算のため
  `.claude/rules/coding-rust.md` 数値契約節の bit 完全一致契約）
- **col2im bit 同一**: CPU 参照実装（`backend-cpu::im2col::col2im`。
  `f64` 逐次和・1 回 `f32` downcast）と byte 単位完全一致すること
  （CUDA は `double` ネイティブアキュムレータ。NaN のみクラス一致で
  比較）
- **conv2d forward／backward REQ-2 parity**: CPU 参照実装
  （`fandhe_ai::tape()`）との複合判定（相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満）が全 fail 0 件であること（GEMM 段由来の差の
  みを許容する設計。im2col／col2im 単体は上記のとおり bit 一致）
- **run-to-run 決定性**: 同一入力で 2 回起動しても bit 同一であること
- 上記いずれも FAIL の場合は本番結線（`ops.rs::CudaBackendOps::
  im2col`／`col2im`）を見直す（イシュー再オープン）
