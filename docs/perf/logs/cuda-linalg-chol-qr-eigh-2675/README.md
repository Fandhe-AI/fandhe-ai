# CUDA 分解系線形代数（cholesky・QR・eigh。#2675）実機未実測・新規 unsafe 承認依頼の申し送り

> **更新（#3136）**: Stage B（起動の実装）は実装済み。D-C 承認（親 #3133）と `docs/gpu-linalg-kernel-launch-unsafe-audit.md`（CU-3〜CU-5）に基づく。実機（GB10）での parity 実測は **#3138 へ申し送り・未実施**（本実行環境は GPU なし）。以下は Stage A 時点の記録で、「承認待ち」の記述は履歴として残す。

`docs/backend-gpu-linalg-decision.md` §3〜§6・§10 参照。本実装エージェント実行環境は CUDA 実機（DGX Spark GB10）に
到達できず、また **カーネル起動に必要な新規 `unsafe` のユーザー承認がイシュー #2675 に確認できなかった**ため、
実装は #2674（LU 系）と同じ「Stage A」（`unsafe` なし）までで止めている。

## 実装済み（Stage A）

- `crates/backend-cuda/src/kernels_linalg_factor.rs`: Cholesky・reduced Householder QR・巡回 Jacobi eigh の `double`
  カーネルソース（1 行列 = 1 block・`__d*_rn` intrinsic による非縮約 FMA 契約・手動境界チェック維持・縮約と収束判定は
  逐次で CPU の算術順序と同一）。ソース走査テスト付き。
- `crates/backend-cuda/src/linalg.rs`: サイズ検証 `validate_square_factor_dims`／`validate_qr_dims`、ホスト後処理
  `normalize_max_abs_sign`／`eigh_sort_and_normalize`／`qr_finalize`（`backend-cpu` の同名処理の意図的複製）、
  NVRTC コンパイル・ロード `CudaLinalgFactor::new`。ゲートは #2674 の `set_gpu_linalg_enabled` を共有（既定 OFF・facade 非公開）。
- `ops.rs`: `linalg_cholesky`／`linalg_qr` を置換し、`linalg_eigh` のオーバーライドを新設（CUDA ではこれまで trait 既定に
  落ちていた）。ゲート OFF は従来どおり `Unsupported`。ゲート ON は rank・正方性・サイズ検証（デバイス非接触）→
  空入力はホストで CPU と同じ形 → それ以外は fail-closed の `Unsupported`（起動は承認待ち）。`Var` はホスト `f64` へフォールバックする。

## 承認依頼（Stage B: 新規 `unsafe` 3 ブロック）

cudarc の `launch_builder(..).launch(cfg)` は `unsafe fn`。次の 3 ブロックのみを追加する承認を求める。

- ブロック 1: `linalg_cholesky_f64` の `launch`（grid `(1,1,1)`・block `FACTOR_BLOCK_DIM`）。
- ブロック 2: `linalg_qr_f64` の `launch`（同上）。
- ブロック 3: `linalg_eigh_f64` の `launch`（同上）。
- SAFETY 根拠: 各バッファ長は検証済みの次元から確保、`i32` 範囲は検証済みで切り詰めなし、カーネル内に手動境界チェック
  （REQ-8）、単一 block のため block 間同期は不要、`status` は必ず読み戻して判定する。

承認後の作業: 3 ブロックと `run_*` 関数・`context_cache` への `cached_linalg_factor` 追加、`factor_launch_pending` を
実起動へ置換（H2D は f32→f64 変換、D2H 後に `linalg.rs` のホスト後処理と 1 回の `as f32`）、`status[0] != 0` を
cholesky は「対称正定値でない」・eigh は「巡回 Jacobi 非収束」の `InvalidArgument` へ写像。

## 測定コマンド案（実機）

```sh
# NVRTC コンパイル・ロード確認（Stage A でも検証可能）と parity（承認待ちの間は parity が fail する想定）
cargo test -p fandhe-ai-backend-cuda --release --test linalg_factor_parity -- --ignored --nocapture
```

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 | 未実測 | NVRTC コンパイル確認のみ。parity は Stage B（unsafe 承認）後 |
