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
  正規化する（`σ==0` の列は Gram–Schmidt で補完し、補完後にも同じ符号規約を適用する。
  codex-review 指摘・2026-09-13 是正）。Cholesky は下三角のみ返す（上三角は 0）
- **QR のメモリ方式**: reduced `Q`（`[m,k]`）は正規化済み Householder ベクトル
  （列インデックス付き）のみを蓄積し、`E_k`（`I_m` の先頭 `k` 列）から出発して反射を
  逆順（`col` の大きい順）に適用する O(mk) メモリの構築方式とする。以前は
  `Q = H_0 H_1 ... H_{k-1}` を `m×m` の `f64` 単位行列上で明示構築していたため、
  `[100000,1]` のような縦長入力（データ自体は約 400 KB）でも中間 `Q` だけで約 80 GB を
  要求しメモリ枯渇を招いていた（codex-review 指摘・2026-09-13 是正）
- **Jacobi SVD の収束判定**: `jacobi_svd_tall` の列直交収束判定は `gamma.abs() <=
  JACOBI_EPS * sqrt(alpha*beta)` の相対しきい値のみで行い、絶対下限（`.max(EPS)`）を
  持たない。絶対下限があると `JACOBI_EPS² = 1e-28` という入力スケール非依存の閾値が
  生じ、列ノルムが約 1e-15 スケールの小さい入力で非直交な列を誤って収束扱いし、誤った
  特異値・特異ベクトルを返していた（codex-review 指摘・2026-09-13 是正）
- **`det` の LU 対角積オーバーフロー対策**: `det(A)`（および `det_vjp` 内部で再計算する
  `det(A)`）は LU 対角成分の総積を単純な `f64` 逐次積では計算しない。正負に極端な
  スケールが混在する対角（例 `diag([1e30; 11], [1e-30; 11])`。真の行列式は約 `1.0`）
  では、11 個の `1e30` を掛けた時点で `f64` の表現範囲（約 `1.8e308`）を超えて `Inf`
  になり、続く `1e-30` を掛けても `Inf` のまま戻らない（対角順序を逆にすると `0.0` に
  なる）——`det_vjp` の勾配 `dA = g · det(A) · A^{-T}` もこの中間値に依存するため、
  本来有限な勾配が `Inf`／`NaN`／全ゼロになる（codex-review 指摘 PRRT_kwDOTuUCJc6hxiys）。
  `eval::linalg::lu_diag_product`／`backend-cpu::linalg::lu_diag_product`（`frexp`／
  `ldexp` 相当。`f64::to_bits`／`from_bits` によるビット操作のみで `unsafe` を使わない）
  が、各対角要素を `m·2^e`（`0.5<=|m|<1`）へ分解し、仮数の積を毎回 `[0.5,1)` 近傍へ
  正規化しつつ指数を整数（`i64`）で加算する方式で計算する。log-abs 方式（`ln` の和を
  経由する方式）は不要な丸め誤差を追加するため採用しない。最終合成（`ldexp`）が `f64`
  の表現範囲を超える場合にのみ `Inf`／`0.0` を返す（真値が範囲外のときの正しい挙動）。
  `det`・`det_vjp` は同じ `lu_diag_product` を経由し、両クレートは意図的な複製として
  同一の演算列を保つ（下記「eval と CPU の関係」）
- **eval と CPU の関係**: `autodiff::eval::linalg`（`pub(crate)`）と `backend-cpu::linalg`
  は同一アルゴリズム・同一規約で実装するが、依存方向の制約（`autodiff` → `backend-cpu`
  の依存は作れる一方、逆に `backend-cpu` が `autodiff` の非公開実装へ依存することはでき
  ない。`crates/autodiff/tests/architecture_boundaries.rs` が機械検査する不変条件）に
  より、コードは意図的に複製する（数式の実体を 2 か所に持つ）。受け入れ判定は REQ-2
  複合判定（`assert_parity`）とし bit 同一を受け入れ条件にしない
- **エラー分類**: 特異／非正定値／非収束は、`Var`（`crates/autodiff/src/var.rs`）が
  返す `AutodiffError` としては **`AutodiffError::InvalidArgument(_)`** に統一する
  （公開ドキュメント契約。CPU 本番経路〈`BackendOps::linalg_*` が返す
  `BackendError::InvalidArgument(_)`〉・フォールバック経路（`eval::linalg` が返す
  `AutodiffError::InvalidArgument(_)`）のどちらを通ったかで呼び出し元から見える
  variant が変わらないよう、`var.rs::unify_backend_error` が CPU 本番経路側を
  `AutodiffError::InvalidArgument` へ写像する。以前は逆方向〈フォールバック側を
  `AutodiffError::Backend(BackendError::InvalidArgument(_))` へ包む〉へ統一しており、
  本番経路の数値エラーが公開ドキュメント記載の variant と一致しない不整合があった
  （codex-review 指摘・2026-09-13 是正）。`Unsupported` は「バックエンドが未実装」の
  意味に限定し、`Var` 側のフォールバック条件から `InvalidArgument` を除外する
- 非有限入力（NaN／inf）は事前検査で拒否しない。実際に拒否できるかは演算ごとに異なり
  一様ではない（codex-review 指摘・2026-09-12 是正。当初の「LU／Cholesky／QR／SVD いず
  れも自然に検出し拒否する」という記述は不正確だった）: **Cholesky** は対角チェック
  （非有限または非正）で `InvalidArgument` を返す。**SVD** は `jacobi_svd_tall` の収束
  判定（`gamma.abs() <= JACOBI_EPS * ...`）が非有限入力で常に偽となり 60 スイープ
  非収束の `InvalidArgument` として拒否される——ただし `min(m,n) == 1` の場合は
  列ペア走査（`q in (p+1)..n`）自体が一度も実行されないため、この判定を経由せず
  `converged` が初期値 `true` のまま成功してしまい、非有限入力を拒否しない
  （codex-review 指摘・2026-09-12 追補）。一方 **LU 分解ベース（`inv`／`solve`／`det`）**
  はピボット判定が `pivot == 0.0` の厳密等値比較のみのため、`inf`／`NaN` はこの判定を
  素通りし「分解成功」として扱われ、後続の前進・後退代入で `inf`／`NaN`／`0.0`（例:
  `1/inf = 0.0`）を含む値が **エラーにならず** 返る（例: `inv([[inf]])` は
  `Err` にならず `[[0.0]]` を返す）。**QR** はそもそも `Result` を返さない設計
  （`fn qr(...) -> (Tensor<f32>, Tensor<f32>)`）のため非有限入力を拒否する経路自体が
  存在せず、非有限値が計算結果へそのまま伝播しうる。明示的な事前 NaN／inf スキャンは
  いずれの演算にも追加していない（拒否経路の統一・追加は数値契約の変更にあたり別
  Issue でユーザー承認を得て検討する）
- **空行列（`n=0`）の挙動**: `det([0,0]) = 1.0`（空積）。`inv`／`cholesky`／`qr`／`svd`
  は対応 shape の空テンソルを返す。`solve` は `[0,k]`。`matrix_norm` は未検証（スコープ
  外。空行列は主要ユースケースでないため）

### 3.6 各演算の VJP（`crates/autodiff/src/grad.rs`・`eval::linalg` の `*_vjp` 関数）

| Op | 式 | 実装方式 |
|---|---|---|
| `Inv` | `dA = -(Xᵀ g Xᵀ)`（`X = A⁻¹`） | `X` は forward の `f32` 記録値を再利用せず、`a`（入力）から `f64` の `LuDecomp` で改めて計算する（極端なスケールで forward の `f32` 丸め済み `X` が `Inf` になり有限勾配が壊れるのを防ぐ。codex-review 指摘・2026-09-13 是正）。`Mat`（内部 `f64` 稠密行列）の `matmul` |
| `Solve` | `dB = A^{-T} g`・`dA = -dB Xᵀ`（`X = A^{-1} B`） | `X` は forward の `f32` 記録値を再利用せず、`a`／`b`（両入力）から `f64` の `LuDecomp` で改めて計算する（`Inv` と同じ理由）。`dB` は `solve_transposed`（`Aᵀ` の LU 分解）で計算し `f32` へ downcast する前の `f64` 中間値を `dA` の計算にも使う |
| `Det` | `dA = g · det(A) · A^{-T}` | `inv(a)` は呼ばず、`a` から改めて `LuDecomp` を作り行列式・逆行列を 1 回の分解から導出する（特異なら fail-closed で伝播）。`det(A)` は `lu_diag_product`（仮数・指数分離方式）で計算し、単純な `f64` 逐次積の中間オーバーフローを避ける（上記「`det` の LU 対角積オーバーフロー対策」） |
| `Cholesky` | `Φ = tril(Lᵀ dL)`（対角 1/2）・`S = L^{-T} Φ L^{-1}`・`dA = (S+Sᵀ)/2` | `Lᵀ` の LU 分解を 2 回の三角解法に再利用 |
| `QrQ`／`QrR` | `M = R dRᵀ − dQᵀQ`・`dA = (dQ + Q copyltu(M)) R^{-T}` | `m ≥ n` 限定（`m<n` は `InvalidArgument`）。`copyltu` は下三角を上三角へ複製して対称化 |
| `SvdU`／`SvdS`／`SvdVh` | Townsend (2016) の標準式（`F_ij = 1/(s_j²−s_i²)`, `i≠j`）+ `m≠n` 補正項 `(I−UUᵀ)dU S⁻¹Vᵀ`・`US⁻¹dVᵀ(I−VVᵀ)` | 特異値が近接／重複（`|s_j²-s_i²| < 1e-9`）の場合 `InvalidArgument`。`U Uᵀ`（`m×m`）・`V Vᵀ`（`n×n`）は明示構築せず `X − U(UᵀX)` 型の積順序（結合則）で `k×k` 以下の中間行列のみを経由する（`m×m`／`n×n` の確保は `[100000,1]` のような入力で約 80 GB を要求しメモリ枯渇を招く。codex-review 指摘・2026-09-13 是正）。`du`／`dvh` が `None`（多出力ノードのうち他ノードのみが損失へ到達するケース）の項（`term2`／`term3`）はゼロ寄与であることが自明なため計算自体を省略する |
| `MatrixNorm` | `Fro`: `g·A/‖A‖`（`‖A‖==0` は 0）／`One`／`Inf`: 最大列（行）の `sign(A)`（同値タイは最初の添字・ゼロ要素は `sign(0)=0`）／`Nuc`: `g·UVᵀ`／`Spectral`: `g·u₀v₀ᵀ` | `Nuc`／`Spectral` は VJP 内で `svd(a)` を再計算。入力に NaN を含む場合（`Fro` の `‖A‖`・`One`／`Inf` の列／行和のいずれかが NaN）は「ノルム 0」分岐で勾配を 0 にすり替えず、勾配全体へ NaN を伝播する（codex-review 指摘・2026-09-13 是正） |

行列積は `tensor-core::BackendOps::gemm_fp32_strict` を経由せず、`eval::linalg`
内部の `Mat`（`f64` 稠密行列）型で完結させる（分解サイズが小さい前提の参照実装として、
三角解法・特異値スケーリングと同じ型で精度を統一するため）。

## 4. テスト構成

| 層 | ファイル | 内容 |
|---|---|---|
| `eval::linalg`（自己完結） | `crates/autodiff/src/eval/linalg.rs` 内 `#[cfg(test)]` | forward の既知値・不変量・VJP 数値微分（grad-check。#223 承認済み定数 `H=1e-3`／`TAU=1e-4`／`REL_TOL=1e-2`／`ABS_TOL=1e-3` をそのまま再利用）に加え、codex-review 指摘の回帰（Jacobi 相対収束判定・QR の O(mk) メモリ・NaN 伝播・零特異値の符号規約・`Inv`／`Solve` VJP の f64 再計算）を含む（40 件） |
| `Var`／`Tape` end-to-end | `crates/autodiff/tests/linalg_backward.rs` | `common::naive_ops()`（`linalg_*` 未実装）経由で `eval::linalg` フォールバック・多出力蓄積・エラー契約（18 件） |
| `backend-cpu::linalg`（自己完結） | `crates/backend-cpu/src/linalg.rs` 内 `#[cfg(test)]` | forward の既知値・決定性に加え `eval::linalg` と同型の codex-review 回帰（22 件） |
| `CpuBackendOps` trait 経由 | `crates/backend-cpu/tests/linalg_parity.rs` | `assert_parity`（REQ-2 複合判定）・決定性・エラー契約（14 件） |
| CUDA／Metal 契約 | `crates/backend-cpu/tests/backend_ops_dispatch.rs` | 実機なしで `Unsupported`・panic しないことを確認（2 件追加） |
| facade 到達性 | `crates/facade/tests/linalg_facade.rs` | `fandhe_ai::tape()`（CPU 実装経路）と `fandhe_ai_autodiff::Tape::new()`（`eval::linalg` フォールバック経路）の一致を `assert_parity` で突合。CPU 本番経路のエラー variant が公開ドキュメントどおり `AutodiffError::InvalidArgument` になることの直接確認を含む（9 件） |

## 5. スコープ外（`.claude/rules/out-of-scope-tracking.md` 対象）

- バッチ次元（`[..., n, n]`）対応（#1600 bmm と同時期に検討）
- GPU（CUDA／Metal）カーネル実装（本イシューは `Unsupported` フォールバック止まり）
- `det` の特異行列における勾配（SVD 経路）・`svd` の重複特異値での勾配・`qr` の `m<n`
  backward・`svd(full_matrices=true)`・`cholesky(upper=true)`・`matrix_norm` の一般
  `p`／`dim` 指定・`torch.linalg.eigh`／`lstsq`／`pinv`／`matrix_rank`／`slogdet`
