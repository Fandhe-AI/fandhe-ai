# CUDA LU 系線形代数（#2674・#3135）実機未実測の申し送り

> **状態（#3135）: Stage B 実装済み・実機未実測。** カーネル起動（`CudaLinalg::launch_lu`＝CU-1・`run_lu_solve_f32`＝CU-2）と
> 実機 `#[ignore]` テストを追加済み。GB10 での NVRTC・parity・特異写像の実測は #3138 で行う（下の実測記入欄は未記入のまま）。
> 以下の「実装済み（Stage A）」「承認依頼」は #2674 時点の記録（履歴）。

`docs/backend-gpu-linalg-decision.md` §3〜§6・§10 参照。本実装エージェント実行環境は CUDA 実機（DGX Spark GB10）に
到達できず、また **カーネル起動に必要な新規 `unsafe` のユーザー承認がイシュー #2674 に確認できなかった**ため、
実装は「Stage A」（`unsafe` なし）までで止めている。

## 実装済み（Stage A）

- `crates/backend-cuda/src/linalg.rs`: 内部 opt-in ゲート `set_gpu_linalg_enabled`／`gpu_linalg_enabled`（既定 OFF・
  facade 非公開）、サイズ検証 `validate_lu_dims`、ホスト後処理 `lu_diag_product`／`permutation_sign`、
  NVRTC コンパイル・ロード `CudaLinalg::new`。
- `crates/backend-cuda/src/kernels_linalg.rs`: 部分ピボット LU（単一 block）と前進・後退代入の `double` カーネルソース
  （`__dmul_rn`／`__dsub_rn`／`__ddiv_rn` による非縮約 FMA 契約・手動境界チェック維持）。
- `ops.rs`: `linalg_inv`／`linalg_solve`／`linalg_det` はゲート OFF なら従来どおり `Unsupported`。ゲート ON では
  形状・サイズ検証（デバイス非接触）→ `n == 0` はホストで CPU と同じ結果 → それ以外は fail-closed の
  `Unsupported`（「起動の新規 unsafe はユーザー承認待ち」）。`Var` 側はホスト `f64` へフォールバックする。

## 承認依頼（Stage B: 新規 `unsafe` 2 ブロック）

cudarc の `launch_builder(..).launch(cfg)` は `unsafe fn`。次の 2 ブロックのみを `CudaLinalg::run_lu_solve_f32`／
`run_lu_det_f32`（`context_cache::with_driver_call` で包む）に追加する承認を求める。

- ブロック 1: `linalg_lu_f64` の `launch`（grid `(1,1,1)`・block `LU_BLOCK_DIM`）。
- ブロック 2: `linalg_lu_solve_f64` の `launch`（grid `ceil(k / LU_SOLVE_BLOCK_DIM)`）。
- SAFETY 根拠: 各バッファ長は検証済み `n`／`k` から確保、`n`・`k` は `i32` 範囲検証済みで切り詰めなし、カーネル内
  手動境界チェック（REQ-8）、LU は単一 block で block 間同期不要、代入カーネルは `status[0]` を読んで特異時は何も書かない。
  状態フラグは必ず読み戻して判定する。

承認後の作業: 上記 2 ブロックと `context_cache::cached_linalg` の追加、`ops.rs` の `lu_launch_pending` を実起動へ置換、
`crates/backend-cuda/tests/linalg_lu_parity.rs` への実機 `#[ignore]` parity テスト追加（`CpuBackendOps` と
`assert_parity`、設計記録 §5 の入力集合）。

## 測定コマンド案（実機）

```sh
# NVRTC コンパイル・ロード確認（Stage A でも検証可能）
cargo test -p fandhe-ai-backend-cuda --release --test linalg_lu_parity -- --ignored --nocapture
```

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 | 未実測 | NVRTC コンパイル確認のみ。parity は Stage B（unsafe 承認）後 |
