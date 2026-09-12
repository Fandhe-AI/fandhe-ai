# 線形代数（inv／solve／det／qr／cholesky／svd）・matrix_norm の設計

イシュー #1621（親イシュー #1573「Tier 2: 線形代数」・`docs/spec/04-requirements.md`
REQ-9 2026-09-12 追記・fandhe-ai-spec#65/#66・PR #69）。`torch.linalg.{inv, solve,
det, qr, cholesky, svd, matrix_norm}` 相当を facade（`fandhe_ai::Var`）へ追加した設計
記録。

## 1. 背景

`docs/compat-feature-gap.md` §2.6「線形代数」は本イシュー着手時点で `matmul`（rank-2）・
`bmm`（なし）・`einsum`（なし）・`transpose` の 4 行のみで、inv/solve/det/qr/cholesky/svd/
matrix_norm はいずれも欠落していた。REQ-9 の Tier 2 列挙・`docs/compat-api-scope.md`
§1.3 が実装リポ側の対応 issue として本 #1621 を割り当てている。

## 2. スコープ

- 対象: rank-2（2 次元行列）限定。バッチ次元（`[..., n, n]`）は対象外（#1600 bmm と
  同時期に検討する別イシューへ引き継ぐ）
- CPU: 実装先行（`crates/backend-cpu/src/linalg.rs`）。CUDA／Metal: GPU カーネル未実装
  のため `BackendOps::linalg_*` の既定 `Unsupported` を明示オーバーライドする（GPU
  カーネル実装は別イシューのスコープ。out-of-scope-tracking.md 対象）
- `det` の特異行列における勾配（SVD 経由）・`svd` の重複特異値での勾配・`qr` の
  `m<n` backward・`svd(full_matrices=true)`・`cholesky(upper=true)`・`matrix_norm` の
  一般 `p`／`dim` 指定・`eigh`／`lstsq`／`pinv`／`matrix_rank`／`slogdet` は対象外

## 3. 全体構成

既存 Op 追加のパターン（`docs/compat-feature-gap.md` §4）を踏襲する：

```
tensor-core   BackendOps に既定 Unsupported の 7 メソッド + 値型（QrFactors／SvdFactors／MatrixNormOrd）
autodiff      eval::linalg（ホスト参照実装。f64 内部計算）／Op variant／Var メソッド／grad.rs VJP
backend-cpu   linalg.rs（CPU 実装。eval::linalg と同一アルゴリズム・同一符号規約の意図的複製）＋ ops.rs 配線
backend-cuda／backend-metal  ops.rs で明示的 Unsupported（sum／max の先例と同型）
facade        新規値型の 1 行 pub use（tests/api_surface.rs の行単位走査に合わせる）
```

### 3.1 値型（`tensor-core::backend_ops`）

- `QrFactors { q: Tensor<f32>, r: Tensor<f32> }`（`GemmChecksum` と同型の pub フィールド struct）
- `SvdFactors { u: Tensor<f32>, s: Tensor<f32>, vh: Tensor<f32> }`
- `MatrixNormOrd { Fro, One, Inf, Nuc, Spectral }`（`#[non_exhaustive]`。`Activation`／
  `MseReduction` と同方針。将来 `ord=p`／`dim` 指定版を追加しうる）

### 3.2 `BackendOps` の 7 メソッド

`linalg_inv`／`linalg_solve`／`linalg_det`／`linalg_cholesky`／`linalg_qr`／`linalg_svd`／
`linalg_matrix_norm`。既定実装は `gemm_checksum` と同じ非破壊拡張パターン（`Err(
BackendError::Unsupported(...))`）。`matrix_norm` は 5 ord すべてを 1 メソッドで扱い、
`Nuc`／`Spectral` は実装内部で特異値分解を用いる（`Var` 側で `svd` ノードを別途合成しない）。

### 3.3 `Op` variant（`crates/autodiff/src/tape.rs::Op`）

```
Inv { input }
Solve { a, b }
Det { input }
Cholesky { input }
QrQ { input, r: Tensor<f32> }       // 兄弟因子 R を payload に保持（Arc 共有・cheap clone）
QrR { input, q: Tensor<f32> }
SvdU { input, s: Tensor<f32>, vh: Tensor<f32> }
SvdS { input, u: Tensor<f32>, vh: Tensor<f32> }
SvdVh { input, u: Tensor<f32>, s: Tensor<f32> }
MatrixNorm { input, ord: MatrixNormOrd }
```

いずれも `push_eager`（非 elementwise。融合対象外）。

**多出力の扱い**: テープは 1 ノード 1 出力のため、QR／SVD は出力ごとにノードを積む。
VJP はコタンジェントに線形なので、各出力ノードが「自分の upstream だけを非ゼロ、他を
ゼロ」とした部分寄与を返し、`Tape::backward` が入力ノードへ合算すれば全体の VJP に
一致する（`view_node_fan_out_accumulates_gradient` と同じ蓄積機構。
`crates/autodiff/tests/linalg_backward.rs::qr_multi_output_gradient_accumulates_to_
single_input`／`svd_multi_output_gradient_accumulates_to_single_input` で検証）。

### 3.4 `Var` メソッドの二段フォールバック

`Var::mse_loss_with`（`crates/autodiff/src/var.rs`）と同じ規律: `self.tape.ops().X()` を
先に試み、`Err(BackendError::Unsupported(_))` のときのみ `eval::linalg::X` へフォール
バックし、それ以外のエラー（`InvalidArgument`＝特異行列等）は伝播する（判定迂回経路を
作らない。`.claude/rules/security.md` A08）。バックエンド戻り値の shape はフォールバック
前に検証してから `push_eager` する。

### 3.5 数値契約

- **内部精度**: 分解・解法は `f64` で逐次固定順序に計算し、出力時に 1 回だけ `f32` へ
  downcast する（正規化統計の `f64` アキュムレータ契約と同じ思想。matmul 系 FMA 契約は
  変更しない。`.claude/rules/coding-rust.md`）。同一入力に対し run-to-run で bit
  決定的（`svd_is_bit_deterministic_across_repeated_calls` テストで検証）
- **符号・ゲージ規約**（`eval::linalg` と `backend-cpu::linalg` の parity 成立に必須）:
  QR は `R` の対角を非負に正規化（Householder の符号を吸収）。SVD は特異値を降順（同値
  は安定ソート）に並べ、各 `V` 列は最大絶対値成分（同値は最小添字）が正になるよう符号を
  正規化し `U = A V / σ` で導出する（`σ==0` の列は Gram–Schmidt で補完）。Cholesky は
  下三角のみ返す（上三角は 0）
- **eval と CPU の関係**: `autodiff::eval::linalg`（`pub(crate)`）と `backend-cpu::linalg`
  は同一アルゴリズム・同一規約で実装するが、依存方向の制約（`autodiff` → `backend-cpu`
  の依存は作れる一方、逆に `backend-cpu` が `autodiff` の非公開実装へ依存することはでき
  ない。`crates/autodiff/tests/architecture_boundaries.rs` が機械検査する不変条件）に
  より、コードは意図的に複製する（数式の実体を 2 か所に持つ）。受け入れ判定は REQ-2
  複合判定（`assert_parity`）とし bit 同一を受け入れ条件にしない
- **エラー分類**: 特異／非正定値／非収束は `BackendError::InvalidArgument`（既存
  variant）。`Unsupported` は「バックエンドが未実装」の意味に限定し、`Var` 側のフォール
  バック条件から `InvalidArgument` を除外する
- 非有限入力（NaN／inf）は事前検査で拒否しない（現状の CPU 実装は LU／Cholesky／QR／
  SVD いずれも計算過程で非有限を自然に検出し `InvalidArgument`（Cholesky の対角チェッ
  ク）または非収束（SVD）として拒否する設計。明示的な事前 NaN スキャンは追加していない）
- **空行列（`n=0`）の挙動**: `det([0,0]) = 1.0`（空積）。`inv`／`cholesky`／`qr`／`svd`
  は対応 shape の空テンソルを返す。`solve` は `[0,k]`。`matrix_norm` は未検証（スコープ
  外。空行列は主要ユースケースでないため）

### 3.6 各演算の VJP（`crates/autodiff/src/grad.rs`・`eval::linalg` の `*_vjp` 関数）

| Op | 式 | 実装方式 |
|---|---|---|
| `Inv` | `dA = -(Xᵀ g Xᵀ)`（`X = A⁻¹` = forward 記録値） | `Mat`（内部 `f64` 稠密行列）の `matmul` |
| `Solve` | `dB = A^{-T} g`・`dA = -dB Xᵀ` | `solve_transposed`（`Aᵀ` の LU 分解） |
| `Det` | `dA = g · det(A) · A^{-T}` | `inv(a)` を再利用（特異なら fail-closed で伝播） |
| `Cholesky` | `Φ = tril(Lᵀ dL)`（対角 1/2）・`S = L^{-T} Φ L^{-1}`・`dA = (S+Sᵀ)/2` | `Lᵀ` の LU 分解を 2 回の三角解法に再利用 |
| `QrQ`／`QrR` | `M = R dRᵀ − dQᵀQ`・`dA = (dQ + Q copyltu(M)) R^{-T}` | `m ≥ n` 限定（`m<n` は `InvalidArgument`）。`copyltu` は下三角を上三角へ複製して対称化 |
| `SvdU`／`SvdS`／`SvdVh` | Townsend (2016) の標準式（`F_ij = 1/(s_j²−s_i²)`, `i≠j`）+ `m≠n` 補正項 `(I−UUᵀ)dU S⁻¹Vᵀ`・`US⁻¹dVᵀ(I−VVᵀ)` | 特異値が近接／重複（`|s_j²-s_i²| < 1e-9`）の場合 `InvalidArgument` |
| `MatrixNorm` | `Fro`: `g·A/‖A‖`（`‖A‖==0` は 0）／`One`／`Inf`: 最大列（行）の `sign(A)`（同値タイは最初の添字）／`Nuc`: `g·UVᵀ`／`Spectral`: `g·u₀v₀ᵀ` | `Nuc`／`Spectral` は VJP 内で `svd(a)` を再計算 |

行列積は `tensor-core::BackendOps::gemm_fp32_strict` を経由せず、`eval::linalg`
内部の `Mat`（`f64` 稠密行列）型で完結させる（分解サイズが小さい前提の参照実装として、
三角解法・特異値スケーリングと同じ型で精度を統一するため）。

## 4. テスト構成

| 層 | ファイル | 内容 |
|---|---|---|
| `eval::linalg`（自己完結） | `crates/autodiff/src/eval/linalg.rs` 内 `#[cfg(test)]` | forward の既知値・不変量（27 件）・VJP 数値微分（grad-check。#223 承認済み定数 `H=1e-3`／`TAU=1e-4`／`REL_TOL=1e-2`／`ABS_TOL=1e-3` をそのまま再利用） |
| `Var`／`Tape` end-to-end | `crates/autodiff/tests/linalg_backward.rs` | `common::naive_ops()`（`linalg_*` 未実装）経由で `eval::linalg` フォールバック・多出力蓄積・エラー契約（17 件） |
| `backend-cpu::linalg`（自己完結） | `crates/backend-cpu/src/linalg.rs` 内 `#[cfg(test)]` | forward の既知値・決定性（13 件） |
| `CpuBackendOps` trait 経由 | `crates/backend-cpu/tests/linalg_parity.rs` | `assert_parity`（REQ-2 複合判定）・決定性・エラー契約（14 件） |
| CUDA／Metal 契約 | `crates/backend-cpu/tests/backend_ops_dispatch.rs` | 実機なしで `Unsupported`・panic しないことを確認（2 件追加） |
| facade 到達性 | `crates/facade/tests/linalg_facade.rs` | `fandhe_ai::tape()`（CPU 実装経路）と `fandhe_ai_autodiff::Tape::new()`（`eval::linalg` フォールバック経路）の一致を `assert_parity` で突合（8 件） |

## 5. スコープ外（`.claude/rules/out-of-scope-tracking.md` 対象）

- バッチ次元（`[..., n, n]`）対応（#1600 bmm と同時期に検討）
- GPU（CUDA／Metal）カーネル実装（本イシューは `Unsupported` フォールバック止まり）
- `det` の特異行列における勾配（SVD 経路）・`svd` の重複特異値での勾配・`qr` の `m<n`
  backward・`svd(full_matrices=true)`・`cholesky(upper=true)`・`matrix_norm` の一般
  `p`／`dim` 指定・`torch.linalg.eigh`／`lstsq`／`pinv`／`matrix_rank`／`slogdet`
