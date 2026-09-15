# イシュー #1767 実測記録先（CUDA Conv1d を Conv2d の特化として実装する）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10 実機
への到達手段（`docs/real-hardware-verification-env.local.md`・
`CUDA_NODE` 環境変数・`~/.ssh/config` のいずれも確認できず。ローカル
`nvidia-smi` も NVML driver／library 不一致で初期化不能）がないため、
**実測値は一切含まれていない**。本ディレクトリは実行コマンド・保存
すべきログ一覧のみを提供し、実測は GB10 実機を持つセッション（後続
#1771「CUDA／Metal 実機 parity・実測」が正式な受け皿）へ申し送る
（`docs/perf/logs/cuda-conv2d-1766/README.md` と同型の運用）。

## 目的

イシュー #1767「Conv1d を Conv2d の特化として実装する」の実機正しさ
検証記録先。origin/main 時点で `Var::conv1d`（#1765）は新規カーネルを
持たず `Var::conv2d`（#1764）への reshape 併合のみで、CUDA
`BackendOps::im2col`／`col2im`（#1766）は形状汎用カーネルのため
`H=1`・`kh=1` の 1d 形状もそのまま処理する。本イシューはこの「特化」
契約（1d 形状が 2d と同一カーネル経路を通り、手動 reshape 版と
forward／backward とも bit 完全一致すること）を実機で確認する
（新規カーネル・facade 新規公開面なし。性能最適化は対象外）。

## 実行コマンド

```bash
# 1) crates/backend-cuda の im2col／col2im bit 同一・parity
#    （2d 形状網羅 + 1d 形状 6 件を含む CASES 全体）
cargo test -p fandhe-ai-backend-cuda --release --test im2col_col2im_parity -- --ignored --nocapture

# 2) crates/facade の conv2d forward／backward REQ-2 複合判定
#    （#1766 分。1d の前提となる 2d 経路の非後退確認）
cargo test -p fandhe-ai --release --test conv2d_backend_parity -- --ignored --nocapture

# 3) crates/facade の conv1d forward／backward REQ-2 複合判定・
#    conv1d↔手動 reshape conv2d の CUDA bit 完全一致（本イシュー分）
cargo test -p fandhe-ai --release --test conv1d_backend_parity -- --ignored --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）にも 1) が
自動的に含まれる。

## 保存すべきログ

- `im2col_col2im_parity-ignored.log`: 上記 1) の生出力（1d 形状 6 件
  を含む形状ごとの bit 一致・run-to-run 決定性の判定内訳）
- `conv2d_backend_parity-ignored.log`: 上記 2) の生出力
- `conv1d_backend_parity-ignored.log`: 上記 3) の生出力
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **im2col bit 同一（1d 形状含む）**: CPU 参照実装
  （`backend-cpu::im2col::im2col`）と byte 単位完全一致すること
  （算術を含まない純粋コピー演算のため bit 完全一致契約）
- **col2im bit 同一（1d 形状含む）**: CPU 参照実装
  （`backend-cpu::im2col::col2im`。`f64` 逐次和・1 回 `f32` downcast）
  と byte 単位完全一致すること（CUDA は `double` ネイティブ
  アキュムレータ。NaN のみクラス一致で比較）
- **conv1d↔手動 reshape conv2d の CUDA bit 完全一致**:
  `cuda_conv1d_matches_manual_reshape_conv2d_bit_exact` が forward・
  d_input・d_weight・d_bias いずれも bit 完全一致であること（「特化」
  契約の直接検証。同一カーネル・同一形状を通るため機構的に成立する
  はず）
- **conv1d／conv2d forward／backward REQ-2 parity**: CPU 参照実装
  （`fandhe_ai::tape()`）との複合判定（相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満）が全 fail 0 件であること（GEMM 段由来の差の
  みを許容する設計）
- **run-to-run 決定性**: 同一入力で 2 回起動しても bit 同一であること
- 上記いずれも FAIL の場合は本番結線（`ops.rs::CudaBackendOps::
  im2col`／`col2im`・`Var::conv1d`／`conv2d`）を見直す
  （イシュー再オープン）
