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
  の承認事項。**#1711 で超越関数系 8 演算（`log`／`log2`／`log10`／
  `sin`／`cos`／`tan`／`abs`／`neg`）の `pub fn` を追加済み**（`Var::
  scalar_unary` 自体は `pub(crate)` のまま不変。facade 新規公開面なし・
  既存 `Var` 再エクスポート経由で到達）。`sub`／`div`／`pow`／`sqrt`
  は #1710、`clamp`／比較演算は #1712 が対象。

## 9. `tensor-core` → `autodiff` の依存方向による意図的複製

`sigmoid_stable`（数値安定形シグモイド）・`nan_propagating_max`／`min`
は `autodiff::eval` に同名の実装が既に存在するが、`tensor-core` は
`autodiff` に依存できない（依存方向は `autodiff` → `tensor-core` の
一方向）ため、`tensor-core::scalar_op` 側に独立実装として複製した
（数式は同一。crate 境界による意図的な複製であり、`.claude/rules/
code-comment-style.md` が禁じる「同一クレート内の陳腐化しやすい重複」
には当たらない）。

## 10. 未実装・スコープ外

- CUDA／Metal の `ScalarOp` カーネル（#1635／#1636）。CUDA は #1700〜
  #1702（算術・超越関数・比較演算＋`Clamp`）で実装済み。Metal は #1707
  （算術系 `Sub`／`Div`／`Pow`／`Sqrt`）＋#1708（超越関数系 `Neg`／
  `Abs`／`Log`／`Log2`／`Log10`／`Sin`／`Cos`／`Tan`）＋#1709（比較演算
  6 種〈`Gt`／`Ge`／`Lt`／`Le`／`Eq`／`Ne`〉＋`Clamp`。ペイロードあり
  unary kind 向け起動引数配線〈`crates/backend-metal/src/
  scalar_op_source.rs::UnaryPayload`〉を新設）で実装済み。両バックエンド
  とも残 kind（`Add`／`Mul`／`Maximum`／`Minimum`・活性化系・
  `LeakyRelu`／`Elu`／`Softplus`／`PowScalar`）はいずれの sub issue にも
  含まれず対象外（`.claude/rules/out-of-scope-tracking.md` 対象）。
- `Var` 公開メソッド（`sub`／`div`／`pow`／活性化等）・facade 範囲拡張
  （#1593／#1595）。**超越関数系 8 演算（`log`／`log2`／`log10`／
  `sin`／`cos`／`tan`／`abs`／`neg`）は #1711 で実装済み**（`pub(crate)`
  入口自体は不変。facade 新規公開面なし）。`sub`／`div`／`pow`／`sqrt`
  は #1710、`clamp`／比較演算は #1712 が対象。
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

## 12. PR #1686 codex-review／Bugbot 指摘の是正

PR #1686（本ドキュメント §1〜11 の実装）に対する codex-review 7 件・
Cursor Bugbot 1 件の指摘を是正した記録。tolerance 定数・既存テストの
許容誤差は変更していない（`.claude/rules/coding-rust.md` 「バックエン
ド間数値一致テストの許容誤差を単独で緩和しない」）。

- **P1（`crates/backend-cpu/tests/scalar_op_parity.rs`）**:
  `elementwise::PARALLEL_THRESHOLD`（`pub(crate)`）の値を統合テスト側
  でリテラル複製していた。逐次／`rayon` 並列の境界検証テストを
  `crates/backend-cpu/src/scalar_elementwise.rs` 内 `#[cfg(test)]`
  単体テストへ移し、`PARALLEL_THRESHOLD` を直接参照する形にして複製
  を解消した。
- **P2（unary `PowScalar` の勾配。`crates/autodiff/src/eval/scalar.rs`
  `unary_grad_factor`）**: `x == 0.0` かつ `exponent == 0.0` のとき
  `exponent * x.powf(exponent - 1.0)` が `0.0 * inf` = `NaN` になって
  いた。forward は `x^0 = 1`（IEEE 754 の `0^0` 規約どおり有限）の
  定数関数のため、`exponent == 0.0` を先に判定し勾配 `0.0` を返すよう
  にした。
- **P2（binary `Pow` の `db`。`crates/autodiff/src/eval/scalar.rs`
  `binary_partials`）**: `a == 0.0` かつ `b == 0.0` のとき
  `da = b * a.powf(b - 1.0)` が同型の `NaN` を生んでいた
  （`db` は既存の `a == 0.0` マスクで元々 `0.0`）。`da` も
  `b == 0.0` を先に判定し `0.0` を返すようマスクした（forward が
  `b` に関して定数関数 `a^0 = 1` になるため）。
- **P2（binary `Div` の `db`。同ファイル）**: `db = -a / (b * b)` は
  `b * b` を先に評価するため `a == b == 1e-30` で `b*b` が `0.0` へ
  underflow して `-inf`、`a == b == 1e20` で `b*b` が `inf` へ
  overflow して `-0.0` になり、いずれも数学的に有限な値
  （`-1/b`）から乖離していた。`db = -(a / b) / b` へ変形し、非 NaN
  入力では既存式と数学的に同値のまま overflow/underflow 耐性を上げた。
- **P2（Softplus forward。`crates/tensor-core/src/scalar_op.rs`）**:
  `(1.0 + (beta * x).exp()).ln()` は `(beta * x)` が大きく負のとき
  `(beta * x).exp()` が `1.0` の ulp 近傍まで小さくなり `1.0 + …` の
  加算で桁落ちしていた（`beta=1e-6, threshold=20, x=-15000000` で
  正しい `0.305902` の代わりに `0.357628` を返す）。`f32::ln_1p` を
  使う `(beta * x).exp().ln_1p() / beta` へ置き換えた。
- **P2（ELU forward。同ファイル）**: `alpha * (x.exp() - 1.0)` は
  `x` が `0` に近いとき `x.exp()` が `1.0` へ丸まり桁落ちしていた
  （`x=-1e-8, alpha=1e8` でほぼ `-1.0` になるべきが `0.0` になる）。
  `f32::exp_m1` を使う `alpha * x.exp_m1()` へ置き換えた。
- **P2（Hardswish forward。同ファイル）**: `x * relu6(x + 3.0) / 6.0`
  は乗算 `x * relu6(...)` を先に評価するため、恒等領域（`x >= 3.0`）
  の巨大な `x`（例 `1e38`）で `x * 6.0` が overflow して `inf` に
  なっていた（正しい値は `x` 自身）。`x * (relu6(x + 3.0) / 6.0)` へ
  括弧を入れ替え、先に `[0.0, 1.0]` へ正規化してから `x` を掛ける
  順序にした（非恒等領域では数学的に同値。ULP 単位の丸め差はあり
  うるが bit 同一を主張する既存テストは add/mul/relu/exp/tanh の
  5 演算限定であり Hardswish には無関係。§11 の許容誤差ベース検証
  はそのまま通る）。
- **P2（GELU tanh 近似版の勾配。`gelu_tanh_grad`。同ファイル）**:
  `|x|` が極端に大きい（例 `1e20`）と `x*x*x`／`x*x` が overflow して
  `du_dx` が `inf` になる一方、`u` も同時に飽和し `tanh_u` が厳密に
  `±1.0` になって `sech2 = 1 - tanh_u^2` が `0.0` になり、
  `0.0 * inf` で `NaN` が生じていた。`sech2 == 0.0`（飽和済み）の
  場合は第 2 項を解析的な極限値である `0.0` として扱い、`du_dx` の
  評価自体を経由しないガードを追加した。
- **Bugbot Low（binary `Maximum`／`Minimum` の VJP。`binary_partials`）**:
  `a`／`b` のいずれかが `NaN` のとき IEEE 754 比較（`>`／`<`）が
  すべて `false` になりタイ分割（`0.5`/`0.5`）へ落ちていた。forward
  （`nan_propagating_max`/`_min`）は `NaN` を明示伝播しており、
  `Clamp` が `NaN` 入力で勾配をゼロにする規約（§3.4）とも整合しない
  ため、**`Maximum`／`Minimum` は `NaN` 入力（いずれか一方でも）では
  タイ分割ではなく両入力の勾配を `(0.0, 0.0)` にする**規約へ統一し、
  §3.4 数値規約に明記した（本モジュール `binary_partials` 冒頭コメント
  参照）。
- **P2（ELU の VJP 係数。`unary_grad_factor`。2 回目の codex-review
  指摘）**: 負側の導関数 `alpha * exp(x)` を丸め済みの forward 出力
  から `y + alpha` で復元していたため、`alpha` が大きく `x` が負の
  とき（例 `x=-20, alpha=1e8`）`y` が `-alpha` へ丸まり係数が `0.0`
  になって勾配が消失していた（解析値は約 `0.2061`）。入力 `x` から
  直接 `alpha * x.exp()` を計算するよう変更した（§8 の ELU 導関数
  契約どおり。f64 昇格は不要）。回帰テスト
  `scalar_unary_elu_grad_is_computed_from_input_not_rounded_forward_output`
  を `crates/autodiff/src/grad.rs` に追加。

新規 `pub` 公開面は追加していない（`PARALLEL_THRESHOLD` の直接参照は
`pub(crate)` のままクレート内単体テストから使う形。ホスト参照実装
`ScalarUnaryOp::apply`／`ScalarBinaryOp::apply`・VJP 係数関数
（`unary_grad_factor`／`binary_partials`）のシグネチャは不変）。
回帰テストは `crates/tensor-core/src/scalar_op.rs`（Softplus／ELU／
Hardswish／`gelu_tanh_grad` の極端入力）・`crates/autodiff/src/grad.rs`
（PowScalar／Pow da／Div db／Maximum・Minimum の NaN）・
`crates/backend-cpu/src/scalar_elementwise.rs`（`PARALLEL_THRESHOLD`
境界の移設）に追加した。
