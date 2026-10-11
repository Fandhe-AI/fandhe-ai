# Metal 線形代数（LU 系・cholesky・QR。#2676）実機 parity 未実測・新規 unsafe 承認依頼の申し送り

`docs/backend-gpu-linalg-decision.md` §3〜§6・§10 参照。**カーネル起動に必要な新規 `unsafe` のユーザー承認がイシュー #2676
に確認できなかった**ため、実装は CUDA 側（#2674・#2675）と同じ「Stage A」（`unsafe` なし）までで止めている。
Metal の eigh は決定記録 §2 のとおり対象外（trait 既定のまま）。

## 実装済み（Stage A）

- `crates/backend-metal/src/shaders/linalg.metal`: `linalg_lu_f32`・`linalg_lu_solve_f32`・`linalg_cholesky_f32`・
  `linalg_qr_f32`（1 行列 = 1 threadgroup・`#pragma clang fp contract(off)`・`precise::sqrt`／`precise::divide`・
  `fma(` 不使用・QR は scale/ssq ノルム・手動境界チェック維持・早期 return でバリアを不一致にしない）。
- `crates/backend-metal/src/linalg.rs`: ゲート（既定 OFF・facade 非公開）、サイズ検証（`u32` 添字・確保上限）、ホスト後処理
  （`lu_diag_product`／`permutation_sign`／`qr_finalize`。`backend-cpu` の複製）、`scale_ssq_norm_f32`。
- `crates/backend-metal/src/linalg_model.rs`: 上記 4 カーネルと演算順序が同一の f32 ホスト逐語モデル。
- `crates/backend-metal/src/linalg_kernel.rs`: MSL ソース保持と `MetalLinalg::new`（パイプライン構築。safe API のみ）。
- `ops.rs`: `linalg_inv`／`solve`／`det`／`cholesky`／`qr` を置換。ゲート OFF は従来どおり `Unsupported`。ゲート ON は
  rank・正方性・サイズ検証（デバイス非接触）→ 空入力はホストで CPU と同じ形 → それ以外は fail-closed の `Unsupported`。
- テスト: `tests/linalg_model_parity.rs`（f32 モデル対 CPU 参照を `assert_parity` で突合。Linux 実行可）、
  `tests/linalg_source_evidence.rs`（MSL 契約トークンの固定）、`tests/linalg_parity.rs`（ゲート・形状・空入力。実機は `#[ignore]`）。

## 承認依頼（Stage B: 新規 `unsafe`）

objc2-metal の `setBuffer_offset_atIndex`／`setBytes_length_atIndex` は `unsafe fn`。次のいずれかの承認を求める。

- 案 A（推奨）: `rnn_cell.rs` の `set_buffer_arg`／`set_u32_arg` 相当と `sort.rs::set_scalar_args` を共有の `pub(crate)`
  モジュールへ移設して再利用する。既存の `unsafe` ブロックの移設のみで正味の新規ブロックはないが、`unsafe` に触れるため
  承認と security-auditor の監査を要する。
- 案 B: `linalg_kernel.rs` に `scan.rs` と同型の `unsafe` ブロックを新設する。
- 不採用: `elementwise::encode_nary_dispatch` の流用（引数レイアウトと起動形状が固定で、n・k の 2 スカラーを渡せない）。
- SAFETY 根拠: バッファ長は検証済みの次元から確保、`u32` 範囲は検証済みで切り詰めなし、カーネル内に手動境界チェック（REQ-8）、
  単一 threadgroup 起動（`threadgroup_position_in_grid != 0` は何もしない）、`status` は必ず読み戻して判定する。

承認後の作業: `context_cache::cached_linalg` の追加、`ops.rs` の `linalg_launch_pending` を実起動へ置換（status の読み戻し、
threadgroup 幅は `maxTotalThreadsPerThreadgroup` で clamp）、エラー写像（solve／inv のピボット 0 は `InvalidArgument`、
det のピボット 0 は `0.0`、非正定値は `InvalidArgument`）、実機 parity の実行。

## 測定コマンド（実機）

```sh
# MSL コンパイルとパイプライン構築（Stage A でも検証可能）
cargo test -p fandhe-ai-backend-metal --release --test linalg_parity -- --ignored --nocapture pipeline_compiles_on_device
# parity（承認待ちの間は fail する想定）
cargo test -p fandhe-ai-backend-metal --release --test linalg_parity -- --ignored --nocapture
```

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-11 | Apple M4 Max | `pipeline_compiles_on_device` | pass | MSL 4 カーネルのコンパイルとパイプライン構築のみ（起動なし） |
| 未実測 | Apple M4 Max | parity（`--ignored` 全件） | 未実測 | Stage B（unsafe 承認）後 |
