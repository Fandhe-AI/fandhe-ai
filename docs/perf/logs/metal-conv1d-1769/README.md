# イシュー #1769 実測記録先（Metal Conv1d を Conv2d の特化として実装する）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree・x86_64）には Apple Silicon
実機への到達手段（`docs/real-hardware-verification-env.local.md`・
`METAL_NODE` 環境変数・`~/.ssh/config` のいずれも確認できず）がないため、
**実測値は一切含まれていない**。本ディレクトリは実行コマンド・保存
すべきログ一覧のみを提供し、実測は Apple Silicon 実機を持つセッション
（後続 #1771「CUDA／Metal 実機 parity・実測」が正式な受け皿）へ申し送る
（`docs/perf/logs/cuda-conv1d-1767/README.md` と同型の運用）。

> **実測の正式な受け皿はイシュー #1771**（`docs/perf/logs/conv-
> realdevice-1771/`）。統合ランブックを参照すること。

## 目的

イシュー #1769「Conv1d を Conv2d の特化として実装する」の実機正しさ
検証記録先。origin/main 時点で `Var::conv1d`（#1765）は新規カーネルを
持たず `Var::conv2d`（#1764）への reshape 併合のみで、Metal
`BackendOps::im2col`／`col2im`（#1768）は形状汎用カーネルのため
`H=1`・`kh=1` の 1d 形状もそのまま処理する。本イシューはこの「特化」
契約（1d 形状が 2d と同一カーネル経路を通り、手動 reshape 版と
forward／backward とも bit 完全一致すること）を実機で確認する
（新規カーネル・facade 新規公開面なし。性能最適化は対象外）。

本ディレクトリは #1768（Metal Conv2d）・#1769（Metal Conv1d）両方の
M4 Max 実機実測の受け皿を兼ねる（#1768 の実装記録が実測を本イシューへ
引き継いだため）。

## 実行コマンド

```bash
# 1) crates/backend-metal の im2col／col2im bit 同一・parity
#    （2d 形状網羅 + 1d 形状 6 件を含む CASES 全体。#1768 分含む）
cargo test -p fandhe-ai-backend-metal --release --test im2col_col2im_parity -- --ignored --nocapture

# 2) crates/facade の conv2d forward／backward REQ-2 複合判定
#    （#1768 分。1d の前提となる 2d 経路の非後退確認）
cargo test -p fandhe-ai --release --test conv2d_backend_parity -- --ignored --nocapture

# 3) crates/facade の conv1d forward／backward REQ-2 複合判定・
#    conv1d↔手動 reshape conv2d の Metal bit 完全一致（本イシュー分）
cargo test -p fandhe-ai --release --test conv1d_backend_parity -- --ignored --nocapture
```

`make test-ignored-metal` にも 1) が自動的に含まれる。

## 保存すべきログ

- `im2col_col2im_parity-ignored.log`: 上記 1) の生出力（1d 形状 6 件
  を含む形状ごとの bit 一致・run-to-run 決定性の判定内訳）
- `conv2d_backend_parity-ignored.log`: 上記 2) の生出力
- `conv1d_backend_parity-ignored.log`: 上記 3) の生出力
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。チップ型番・
  macOS／Metal バージョン程度に留める）

## 事前登録判定規則

- **im2col bit 同一（1d 形状含む）**: CPU 参照実装
  （`backend-cpu::im2col::im2col`）と byte 単位完全一致すること
  （算術を含まない純粋コピー演算のため bit 完全一致契約）
- **col2im bit 同一（1d 形状含む）**: CPU 参照実装
  （`backend-cpu::im2col::col2im`。`f64` 逐次和・1 回 `f32` downcast）
  と byte 単位完全一致すること（Metal は binary64 逐次加算の
  64bit 整数ソフトウェアエミュレーション〈`shaders/im2col.metal::
  im2col_f64_*`〉。NaN のみクラス一致で比較）
- **conv1d↔手動 reshape conv2d の Metal bit 完全一致**:
  `metal_conv1d_matches_manual_reshape_conv2d_bit_exact` が forward・
  d_input・d_weight・d_bias いずれも bit 完全一致であること（「特化」
  契約の直接検証。同一カーネル・同一形状を通るため機構的に成立する
  はず）
- **conv1d／conv2d forward／backward REQ-2 parity**: CPU 参照実装
  （`fandhe_ai::tape()`）との複合判定（相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満）が全 fail 0 件であること（GEMM 段由来の差の
  みを許容する設計）
- **run-to-run 決定性**: 同一入力で 2 回起動しても bit 同一であること
- 上記いずれも FAIL の場合は本番結線（`ops.rs::MetalBackendOps::
  im2col`／`col2im`・`Var::conv1d`／`conv2d`）を見直す
  （イシュー再オープン）
