# ScalarUnaryOp／ScalarBinaryOp dispatch 機構の設計記録

イシュー #1634（親 #1592）。承認記録: イシュー #1634 コメント（2026-09-12・
ユーザー承認）で公開クレート `fandhe-ai-tensor-core` の `BackendOps` trait
拡張の実装着手が承認済み（前提の spec 改定 fandhe-ai-spec#66・
`docs/spec` 追従〈#1656〉・#1591 はマージ済み）。

## 1. 背景・目的

`docs/compat-feature-gap.md` §2.4 が指摘するとおり、`BackendOps` の
elementwise 面は `add`／`mul`／`relu`／`exp`／`tanh` の 5 演算固定で、
演算を 1 つ足すごとに trait メソッド＋3 バックエンド実装＋VJP が必要に
なる構造だった（`crates/tensor-core/src/backend_ops.rs` の該当 5 メソッ
ド）。本イシューは「演算種別を `#[non_exhaustive]` enum で表し、1 対の
dispatch メソッド（既定 `Unsupported`）へ集約する」機構を `tensor-core`
に定義し、CPU 参照実装と `autodiff` の汎用 VJP を接続する。後続の
#1593/#1595 は `Var`／facade 配線、#1635/#1636 は enum から式テンプレー
トを生成する GPU カーネルの追加のみで済む前提を作る。

既存 5 演算（`add`／`mul`／`relu`／`exp`／`tanh`）の公開シグネチャ・
挙動（遅延融合契約・bit 値・性能）は変更しない。

## 2. 親 #1592 の分担

| 区分 | 内容 | 担当 |
|---|---|---|
| (a) core | enum・dispatch 定義・autodiff VJP 接続・CPU 参照実装 | **本イシュー（#1634）** |
| (b) CUDA | enum から式テンプレートを生成する CUDA カーネル | #1635 |
| (c) Metal | 同・Metal カーネル | #1636 |
| `Var` 公開メソッド | `sub`／`div`／`pow`／比較等の facade 到達経路 | #1593 |
| 活性化 | GELU／SiLU／…の facade 到達経路 | #1595 |

## 3. enum variant 表

### 3.1 `ScalarUnaryOp`（`crates/tensor-core/src/scalar_op.rs`）

payload なし: `Neg`・`Abs`・`Sqrt`・`Log`・`Log2`・`Log10`・`Sin`・`Cos`・
`Tan`・`Relu`・`Exp`・`Tanh`・`Sigmoid`・`Gelu`（erf 版）・`GeluTanh`・
`Silu`・`Hardswish`。

payload あり: `LeakyRelu { negative_slope: f32 }`・`Elu { alpha: f32 }`・
`Softplus { beta: f32, threshold: f32 }`・`Clamp { min: f32, max: f32 }`・
`PowScalar { exponent: f32 }`。

### 3.2 `ScalarBinaryOp`

`Add`・`Sub`・`Mul`・`Div`・`Pow`・`Maximum`・`Minimum`・`Gt`・`Ge`・`Lt`・
`Le`・`Eq`・`Ne`（比較は f32 の `0.0`／`1.0` 出力。bool テンソル型は
#1613 の対象でスコープ外）。

`derive` は両 enum とも `Debug, Clone, Copy, PartialEq`（`f32` payload を
持つため `Eq` は実装しない）。

### 3.3 NVRTC キャッシュキー（#1635 への申し送り）

`ScalarOpKind::kind_name(&self) -> &'static str`（`Debug` 出力とは独立の
安定文字列）を用意した。CUDA 側 NVRTC キャッシュキーは `kind_name()` の
みを使い、`f32` ペイロード値はカーネル引数として渡すこと（キャッシュ
キーに値を混ぜない）。

## 4. 既存 `BinaryElementwiseOp`／`UnaryElementwiseOp`（#1584）との違い

既存 2 enum は `binary_elementwise_device`／`unary_elementwise_device`
（`DeviceBuffer` 常駐 dispatch）専用で、「ホスト版と同一カーネルにより
bit 同一」「shape 完全一致限定（broadcast 非対応）」という狭い契約を
持つ。本 enum はホスト `Tensor` 入出力・`add`／`mul` と同じ broadcast
意味論・超越関数（REQ-2 複合判定対象）を含み契約が異なるため、既存
enum への variant 追加ではなく別 enum として新設した。

`DeviceBuffer` 常駐版の ScalarOp dispatch は本イシューのスコープ外
（§8）。

## 5. forward 数式の単一情報源

`ScalarUnaryOp::apply`／`ScalarBinaryOp::apply`（`tensor-core::scalar_op`）
が forward 数式の単一情報源（`.claude/rules/code-comment-style.md`
「数式の実体を二重管理しない」）。CPU 参照実装
（`backend-cpu::scalar_elementwise`）・`autodiff` のホストフォールバック
（`autodiff::eval::scalar`）・単体テストがいずれもここへ委譲する。

`erf`（`Gelu` の誤差関数版が使う）は依存追加不可（deps-policy.md）の
ため `f64` 精度の自作近似（Abramowitz–Stegun 7.1.26。最大絶対誤差
1.5e-7）を実装した（`scalar_op::erf_f64`）。GPU 側 `erff` との差は
REQ-2 複合判定（絶対誤差 1e-5 未満）に収まる想定（GPU カーネル実装は
#1635／#1636 のスコープ）。

## 6. CPU 参照実装の bit 同一性（§3.6 委譲の向き）

`ScalarUnaryOp::Add`（該当なし。二項）／`ScalarBinaryOp::Add`／`Mul`・
`ScalarUnaryOp::Exp`／`Tanh` は既存 `add_slice`／`mul_slice`／
`exp_slice`／`tanh_slice` と**同一の 1 回の IEEE 754 演算**を行う
map 演算であり、演算順序（逐次／`rayon` 並列のどちらで処理するか）が
結果に影響しない（`elementwise.rs` モジュール doc「並列化」と同じ理
由）。このため CPU 側は既存カーネルへの明示的な委譲を行わず、汎用
ループ（`scalar_elementwise::scalar_unary`／`scalar_binary`）が
`ScalarUnaryOp::apply`／`ScalarBinaryOp::apply` を直接呼ぶだけで、
既存 5 演算と自動的に bit 同一になる（`tests/scalar_op_parity.rs` で
機械確認）。

**`ScalarUnaryOp::Relu` のみ例外**: 既存 `BackendOps::relu`
（`elementwise::relu`。`f32::max` で `NaN` を伝播しない契約）とは異な
り、本 variant は PyTorch の `torch.relu` に合わせ `NaN` を明示的に伝
播する（§7 数値規約）。非 `NaN` 入力では両者とも `x.max(0.0)` と同値
のため bit 同一だが、`NaN` 入力でのみ結果が異なる（意図的な差異。
`ScalarUnaryOp::Relu` の doc comment 参照）。

## 7. 数値規約（テストで固定）

| 項目 | 規約 |
|---|---|
| `Relu`／`Maximum`／`Minimum`／`Clamp` の `NaN` | 伝播する（明示分岐） |
| `Relu` の劣勾配 | `x == 0` で `0`（既存 `Op::Relu` と同じ `v > 0.0` 規約） |
| `Abs` の劣勾配 | `x == 0` で `0`（`sign(0) = 0`） |
| `Clamp` | 範囲外で勾配 `0`・境界上は `1`（PyTorch 準拠）。`min > max` は常に `max`（panic しない） |
| `Maximum`／`Minimum` の tie | `0.5`／`0.5` に分配（`Var::max`〈縮約〉の先勝ち規約とは別演算） |
| 比較演算 | 出力 `0.0`／`1.0`。VJP は両入力ともゼロ勾配（寄与を省略しない） |
| `Div` | IEEE のまま（`inf`／`NaN`。panic しない）。`da = 1/b`, `db = -a/b²` |
| `Pow`（binary） | `da = b·a^(b−1)`, `db = y·ln(a)`。`a == 0` は `db = 0`（PyTorch のマスク規約） |
| `PowScalar` | `da = p·x^(p−1)` |
| `Sqrt`／`Log*` | 定義域外は IEEE。導関数 `0.5/y`・`1/(x·ln b)` |
| `Gelu`（erf） | `0.5·x·(1+erf(x/√2))`。導関数 `Φ(x) + x·φ(x)` |
| `Sigmoid` | `autodiff::eval::sigmoid_scalar` と同じ数値安定形（独立実装。§9 参照） |

## 8. autodiff 側の統合方針

- `Op::ScalarUnary { op, input }`／`Op::ScalarBinary { op, a, b }` は
  **eager**（`push_eager`。`Op::Sigmoid`／`Op::Where` と同型）。遅延融
  合経路（`push_lazy`・`is_lazy_elementwise`）には入れない。
- `is_checkpoint_eligible` は **`false`**（`recompute_value` 分岐を持
  たないため。最小・安全側の判断。`Op::Where` と同じ整理）。
- **#1583 の REJECT を尊重**: `ELEMENTWISE_VJP_VIA_BACKEND_OPS = false`
  は確定判定のまま変更しない。`upstream ⊙ factor` は既存のゲート付き
  `vjp_elementwise_mul`（`grad.rs`）を通す。新たな backend-ops 経由
  VJP 経路は作らない。
- forward の dispatch は `where_cond_with_fallback` と同型
  （`scalar_unary_with_fallback`／`scalar_binary_with_fallback`。
  `Unsupported` のときのみホスト `apply` へフォールバック・他エラー
  は fail-closed 伝播・戻り shape を検査）。
- binary VJP の broadcast 縮約は `Op::Add` と同じ `reduce_bias_grad`
  （bias パターンは f64 相当・それ以外は `reduce_to_shape` へ委譲）
  を `da`／`db` 双方に使う。
- `Var` の入口は **`pub(crate)`**（`Var::scalar_unary`／
  `scalar_binary`）。`Var` は facade が素で再エクスポートしているため
  `pub fn` 追加は `docs/compat-api-scope.md` §5 の範囲拡張＝#1593/#1595
  の承認事項。

## 9. `tensor-core` → `autodiff` の依存方向による意図的複製

`sigmoid_stable`（数値安定形シグモイド）・`nan_propagating_max`／`min`
は `autodiff::eval` に同名の実装が既に存在するが、`tensor-core` は
`autodiff` に依存できない（依存方向は `autodiff` → `tensor-core` の
一方向）ため、`tensor-core::scalar_op` 側に独立実装として複製した
（数式は同一。crate 境界による意図的な複製であり、`.claude/rules/
code-comment-style.md` が禁じる「同一クレート内の陳腐化しやすい重複」
には当たらない）。

## 10. 未実装・スコープ外

- CUDA／Metal の `ScalarOp` カーネル（#1635／#1636）。
- `Var` 公開メソッド（`sub`／`div`／`pow`／活性化等）・facade 範囲拡張
  （#1593／#1595）。
- `DeviceBuffer` 常駐版 `ScalarOp` dispatch（`binary_elementwise_device`
  ／`unary_elementwise_device` と同型の常駐版）。
- checkpoint 再計算適格化（`is_checkpoint_eligible == true`）。
- bool dtype 出力（比較演算の出力を `Tensor<bool>` にする案。#1613 の
  対象）。
- 既存 5 演算の遅延融合経路への `ScalarOp` 統合・`PARALLEL_THRESHOLD`
  の見直し（性能パラメータ変更はユーザー承認事項）。

## 11. 検証

- `crates/tensor-core/src/scalar_op.rs` 内 `#[cfg(test)]`: 全 variant
  の既知値・NaN／inf 伝播・`erf` 精度・`Clamp(min>max)`。
- `crates/backend-cpu/tests/scalar_op_parity.rs`: 全 variant × 逐次ホス
  ト参照との bit 同一（contiguous・`PARALLEL_THRESHOLD` 境界・非
  contiguous view・broadcast）・既存 5 演算との bit 同一（非 NaN 入
  力）・`Relu` の意図的な NaN 分岐差異・shape 不一致の fail-closed。
- `crates/autodiff/src/eval/scalar.rs` 内 `#[cfg(test)]`: forward・
  VJP 係数の単体テスト。
- `crates/autodiff/src/grad.rs` 内 `#[cfg(test)]`: 全 variant の解析
  勾配と中央差分の突合（kink を避けた固定値）・kink 点の劣勾配直接検
  証・`Var::scalar_unary`／`scalar_binary` の `Tape` 経由エンドツーエ
  ンド（backward・poison 伝播・`is_checkpoint_eligible == false`）・
  shape 不一致の fail-closed。
