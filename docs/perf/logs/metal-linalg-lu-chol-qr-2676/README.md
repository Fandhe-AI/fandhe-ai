# Metal 線形代数（LU 系・cholesky・QR。#2676・#3137）実機 parity 未実測の申し送り

`docs/backend-gpu-linalg-decision.md` §3〜§6・§10 参照。Stage A（#2676）は `unsafe` なしまでで止め、Stage B（#3137）で
案 A（既存ヘルパーの共有モジュールへの移設。正味の新規 `unsafe` ブロックなし。`docs/gpu-linalg-kernel-launch-unsafe-audit.md` §3）により起動を結線した。
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

## 実装済み（Stage B。#3137）

- `crates/backend-metal/src/encoder_args.rs`: `setBuffer`／`setBytes` の共有ヘルパー（案 A。`rnn_cell.rs`・`sort.rs` から移設）。
- `linalg_kernel.rs`: `run_lu_solve_f32`・`run_lu_det_f32`・`run_cholesky_f32`・`run_qr_f32`（`dispatch_sync` 固定・
  `status` は 0 初期化・因数分解は単一 threadgroup・幅は `maxTotalThreadsPerThreadgroup` で clamp・`k == 0` は solve を encode しない）。
- `linalg.rs`: `decode_lu_status`・`decode_cholesky_status`・`det_from_lu`・`clamp_tg_width`・`validate_launch_stride_headroom`（Linux で単体テスト）。
- `context_cache::cached_linalg`、`ops.rs` の 5 演算を実起動へ置換。エラー写像: solve／inv のピボット 0 は `InvalidArgument`、
  det のピボット 0 は `0.0`、非正定値は `InvalidArgument`、デバイス失敗は `map_metal_error`。
- 実機の数値一致（`--ignored` 全件）は **未実測**。計測の事前登録・入力集合・記録は `docs/perf/logs/metal-linalg-parity-3139/`（#3139）。
  #3139 で入力集合を是正した（QR 17×17 を κ₂ ≤ 1e3 の入力へ差し替え・det は n ≤ 17・QR に 1e±20 倍と `1e20 × I₂` を追加。tolerance は不変）。

## 測定コマンド（実機）

```sh
# MSL コンパイルとパイプライン構築（Stage A でも検証可能）
cargo test -p fandhe-ai-backend-metal --release --test linalg_parity -- --ignored --nocapture pipeline_compiles_on_device
# parity（Stage B 実装済み。実測は #3139）
cargo test -p fandhe-ai-backend-metal --release --test linalg_parity -- --ignored --nocapture
```

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-11 | Apple M4 Max | `pipeline_compiles_on_device` | pass | MSL 4 カーネルのコンパイルとパイプライン構築のみ（起動なし） |
| 未実測 | Apple M4 Max | parity（`--ignored` 全件） | 未実測 | #3139 で Mac mini に置き換え |
| 未実測（2026-10-16 時点） | Mac mini（Apple M4） | parity（`--ignored` 全件） | 未実測 | 実機に到達できず未実行。事前登録 `metal-linalg-parity-3139/RULE.txt`。Mac 側での実行を申し送り |
