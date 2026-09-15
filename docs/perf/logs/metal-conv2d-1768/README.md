# イシュー #1768 実測記録先（Metal Conv2d im2col／col2im）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には Apple Silicon 実機
への到達手段（`docs/real-hardware-verification-env.local.md`）が
ないため、**実測値は一切含まれていない**。本ディレクトリは実行
コマンド・保存すべきログ一覧のみを提供し、実測は Mac 実機を持つ
セッションへ申し送る（`docs/perf/logs/cuda-conv2d-1766/README.md` と
同型の運用）。

> **実測の正式な受け皿はイシュー #1771**（`docs/perf/logs/conv-
> realdevice-1771/`）。統合ランブックを参照すること。

## 目的

イシュー #1768「Metal Conv2d forward／backward（im2col＋GEMM）」の
実機正しさ検証記録先。本イシューは正しさ・数値契約のみを対象とし
（性能最適化・デバイス常駐化はスコープ外・`docs/conv-ops-design.md`
§15「#1768」節「引き継ぎ」）、以下の `#[ignore]` テストを M4 Max
実機で実行して green を確認することが目的。

## 実行コマンド

```bash
# 1) crates/backend-metal の im2col／col2im bit 同一・parity（形状網羅。
#    重なり窓・dilation・groups／depthwise・padding のみの窓・座標
#    アンダーフロー形状・N=0・256 threadgroup 境界をまたぐ numel・
#    NaN／±inf／−0.0・非 contiguous view 入力・run-to-run bit 同一）
cargo test -p fandhe-ai-backend-metal --release --test im2col_col2im_parity -- --ignored --nocapture

# 2) crates/facade の conv2d forward／backward REQ-2 複合判定
#    （d_input／d_weight／d_bias）
cargo test -p fandhe-ai --release --test conv2d_backend_parity -- --ignored --nocapture
```

`make test-ignored-metal`（該当する場合）にも上記が自動的に含まれる。

## 保存すべきログ

- `im2col_col2im_parity-ignored.log`: 上記 1) の生出力（形状ごとの
  bit 一致・run-to-run 決定性の判定内訳）
- `conv2d_backend_parity-ignored.log`: 上記 2) の生出力
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  macOS／Metal バージョンの要約程度に留める）

## 事前登録判定規則

- **im2col bit 同一**: CPU 参照実装（`backend-cpu::im2col::im2col`）と
  byte 単位完全一致すること（算術を含まない純粋コピー演算のため
  `.claude/rules/coding-rust.md` 数値契約節の bit 完全一致契約）
- **col2im bit 同一**: CPU 参照実装（`backend-cpu::im2col::col2im`。
  `f64` 逐次和・1 回 `f32` downcast）と byte 単位完全一致すること
  （Metal は binary64 ソフトウェアエミュレーションアキュムレータ。
  NaN のみクラス一致で比較）
- **conv2d forward／backward REQ-2 parity**: CPU 参照実装
  （`fandhe_ai::tape()`）との複合判定（相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満）が全 fail 0 件であること（GEMM 段由来の差の
  みを許容する設計。im2col／col2im 単体は上記のとおり bit 一致）
- **run-to-run 決定性**: 同一入力で 2 回起動しても bit 同一であること
- 上記いずれも FAIL の場合は本番結線（`ops.rs::MetalBackendOps::
  im2col`／`col2im`）を見直す（イシュー再オープン）
