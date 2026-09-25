//! `mish`・`hardtanh`・`relu6`・`prelu`・`glu` の 5 活性化演算
//! （イシュー #2146・親 #2131「5-B 演算」）。
//!
//! **新規 `Op` はゼロ（受け入れ条件）**: いずれも既存の `Var::mul`
//! （`Op::Mul`）・`Var::tanh`（`Op::Tanh`）・`Var::softplus`
//! （`Op::ScalarUnary`）・`Var::clamp`（`Op::ScalarUnary`）・
//! `Var::where_cond`（`Op::Where`）・`Var::detach`・`Var::narrow`・
//! `Var::sigmoid`（`Op::Sigmoid`）・`Var::reshape` の合成のみで構成
//! する。いずれも CPU・CUDA・Metal の全バックエンドに経路があり
//! （超越関数を含む演算は既定 `Unsupported` でホスト参照実装へ
//! フォールバック）、専用カーネルなしで到達可能。`crates/backend-*`・
//! `crates/tensor-core` は変更しない。
//!
//! **facade 非公開（意図的）**: `crates/autodiff/src/matrix_ops.rs`・
//! `rearrange_ops.rs` モジュール doc と同じ理由・同じ判断枠組みによる。
//! `Var` は facade（`fandhe_ai` クレート）から直接再エクスポートされる
//! ため、`Var` への inherent メソッド追加は即座に facade 公開面へ
//! 出てしまう。イシュー #2146 本文は facade 公開面（`Var` への委譲
//! メソッド・`compat::Sequential::add_*` 5 種）を承認事項として明示し、
//! 親 #2131 はこのツリーに限り「設計判断記録 → 承認 → 実装」の 2 段階
//! を定めるため、承認が取れるまでは自由関数として `Var` の外に置き
//! 到達不能にする（`docs/autodiff-activation-ops-decision.md` §2.1）。
//! 承認後は `Var::mish` 等の薄い委譲メソッドを追加し、facade 側の
//! 保留ガード（`crates/facade/src/lib.rs::
//! VarActivationOpsHoldDoctestGuard`）を撤去する。
//!
//! **数値契約**（詳細は `docs/autodiff-activation-ops-decision.md` §3
//! の表を参照）:
//! - `mish`: `x * tanh(softplus(x, 1.0, 20.0))`。`softplus`／`tanh` は
//!   いずれも超越関数を含むため forward／backward とも REQ-2 統一
//!   複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で比較
//!   する。PyTorch CPU 参照実装は softplus に閾値を持たないが、本実装
//!   は `threshold=20.0`（`nn.Softplus` 既定）を使う。`x > 20` では
//!   `exp(-20) ≈ 2e-9` が f32 の丸め誤差より小さいため両者の差は
//!   REQ-2 判定内に収まる（`docs/autodiff-activation-ops-decision.md`
//!   §4）。
//! - `hardtanh`／`relu6`: forward は `Var::clamp(min, max)` と bit
//!   完全一致する（`where_cond` の分岐がいずれの経路でも `clamp` と
//!   同じ値を選ぶ構造上の帰結。詳細は [`hardtanh`] doc）。backward は
//!   開区間 `min < x < max` でのみ勾配 1・境界と区間外は 0（PyTorch
//!   `hardtanh_backward` と同じ。`Var::clamp` の境界含む勾配とは異なる
//!   契約）。
//! - `prelu`: forward は選択と乗算 1 回のみのため bit 完全一致する。
//!   入力勾配も同様に bit 完全一致する。`weight` の勾配は
//!   `reduce_to_shape` の縮約を経由するため REQ-2 統一複合判定で
//!   比較する（`outer` backward と同じ扱い）。
//! - `glu`: forward は `narrow`（コピー）→ `sigmoid`（超越関数）→
//!   `mul` の合成のため、`sigmoid` を経由する側の丸めにより REQ-2
//!   統一複合判定で比較する。
//!
//! **PyTorch との既知の差分**（`docs/autodiff-activation-ops-decision.md`
//! §4 に集約）:
//! - `mish` の softplus 閾値（上記）。
//! - `glu` は負の `dim` を受け付けない（本クレートの慣例に従い
//!   `usize`。PyTorch の既定 `dim=-1` は呼び出し側が明示的な軸番号で
//!   指定する）。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: 外部から
//! 渡る `min`／`max`（有限・`min < max`）・`num_parameters`（`C ≥ 1`）・
//! `weight` の shape（rank 1・`C > 1` なら入力 rank ≥ 2 かつ
//! `shape[1] == C`）・`dim`（`glu` の `shape[dim]` が偶数）は計算・
//! メモリ確保の前にすべて検査し、違反は型付きエラー（
//! `AutodiffError::InvalidArgument`／`AutodiffError::Shape`）で
//! fail-closed に拒否する。本番経路で `unwrap()`／`expect()` は
//! 使わない。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::var::Var;

/// `hardtanh`／`prelu` が使う値依存マスク構築の共通ヘルパー。
///
/// `x.host_view()`（`RefCell` 借用を関数内へ閉じ込めた contiguous な
/// ホスト読み出し。`var.rs::VarHostView` doc 参照）で読み出した値へ
/// `predicate` を適用し、真の位置を `true` とする `Tensor<bool>` を
/// 組み立てる。`matrix_ops::build_tril_triu_mask` と異なり、マスクは
/// 添字ではなく `x` の**値**に依存するため、tril/triu のように
/// 添字だけから決定的に構築することはできない
/// （`Var::gt`／`bool_ops::gt_bool` は比較対象がスカラーではなく
/// `Var` を要求するため、定数スカラーとの比較にはホスト経由が
/// 最も単純で、`x` は既に実体化済みの値を読むだけで確保量も
/// `x` の要素数に比例する〈`broadcast_to` の stride-0 view 経由で
/// 無関係に巨大化する余地がない〉。REQ-8 の確保前検査対象外）。
fn build_value_mask<'t>(
    x: &Var<'t>,
    predicate: impl Fn(f32) -> bool,
) -> Result<Tensor<bool>, AutodiffError> {
    let shape = x.shape();
    let view = x.host_view();
    let data: Vec<bool> = view.iter().map(|&v| predicate(v)).collect();
    drop(view);
    Tensor::new(data, &shape).map_err(AutodiffError::Shape)
}

/// Mish（`x * tanh(softplus(x))`。PyTorch `F.mish`／`nn.Mish` 相当）。
///
/// `Var::softplus(1.0, 20.0)`（`nn.Softplus` 既定パラメータ）→
/// `Var::tanh`（無条件成功）→ `Var::mul` の 3 演算合成。`softplus`
/// が `beta`／`threshold` の検査（有限・`beta > 0`）を担うため本関数
/// 側の追加検査は不要。数値契約はモジュール doc 参照。
pub fn mish<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    let sp = x.softplus(1.0, 20.0)?;
    let activated = sp.tanh();
    x.mul(&activated)
}

/// Hardtanh（PyTorch `F.hardtanh`／`nn.Hardtanh` 相当）: `min < x < max`
/// の開区間では `x` をそのまま通し、それ以外（境界・区間外・`NaN`）は
/// `clamp(x, min, max)` を返す。
///
/// **forward が `Var::clamp` と bit 完全一致する理由**: 開区間内では
/// `a = x` を選び、`clamp` も範囲内の `x` を素通しするため一致する。
/// 境界・区間外（`b` 側）では `b = clamp(x)` をそのまま返すため定義
/// から一致する。`NaN` は開区間判定（`min < x` かつ `x < max`）が
/// 常に偽になるため `b` 側（`clamp(NaN) == NaN`。`ScalarUnaryOp::Clamp`
/// の数値契約）を通り、`NaN` の bit パターンも保存される。
///
/// **backward の契約**: `Var::where_cond` の VJP は選択された枝
/// （`a` か `b`）へのみ勾配を流す。`b = clamp(x).detach()` は
/// `detach()` によって勾配経路を切っているため、境界・区間外では
/// 入力勾配が構造的に 0 になる（PyTorch `hardtanh_backward` の開区間
/// 契約と一致）。開区間内では `a = x` の恒等勾配（1）がそのまま流れる。
///
/// **検査**: `min`／`max` は有限・`min < max` を要求する（PyTorch
/// `nn.Hardtanh` の `max_val > min_val` 契約と同じ）。`±inf` はどちらも
/// 受け付ける（有限性検査は `NaN` のみを弾く実質的な効果になる）。
pub fn hardtanh<'t>(x: &Var<'t>, min: f32, max: f32) -> Result<Var<'t>, AutodiffError> {
    if min.is_nan() || max.is_nan() {
        return Err(AutodiffError::InvalidArgument(format!(
            "activation_ops::hardtanh: min/max must not be NaN, got min={min}, max={max}"
        )));
    }
    if min >= max {
        return Err(AutodiffError::InvalidArgument(format!(
            "activation_ops::hardtanh: min must be less than max, got min={min}, max={max}"
        )));
    }
    let open_mask = build_value_mask(x, |v| v > min && v < max)?;
    let clamped = x.clamp(min, max)?.detach()?;
    Var::where_cond(&open_mask, x, &clamped)
}

/// ReLU6（PyTorch `F.relu6`／`nn.ReLU6` 相当）: `hardtanh(x, 0.0, 6.0)`
/// への薄い委譲。数値・境界勾配の契約は [`hardtanh`] を参照。
pub fn relu6<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    hardtanh(x, 0.0, 6.0)
}

/// PReLU（PyTorch `F.prelu`／`nn.PReLU` 相当）:
/// `x > 0 ? x : weight * x`（chennel-wise の傾き）。
///
/// `weight` は rank 1・`C = weight.shape()[0] ≥ 1` を要求する。
/// `C == 1` は全チャネル共有（`x` の shape によらずスカラー的に
/// broadcast）、`C > 1` は `x` の rank が 2 以上かつ `x.shape()[1] == C`
/// を要求する（PyTorch のチャネル軸＝軸 1 規約）。`weight` を
/// `[1, C, 1, …]`（`x` の rank に合わせた形。`C == 1` は全軸 1）へ
/// `Var::reshape` してから乗算するため、`Var::mul` の NumPy 互換
/// broadcast がチャネル軸だけに `weight` を効かせる。
///
/// **`relu(x) + w * (x - relu(x))` を採らない理由**: `x = +inf` で
/// `inf - inf = NaN` を生むため（モジュール doc 参照）。
///
/// **数値契約**: forward は選択と乗算 1 回のみのため bit 完全一致
/// する。入力勾配（`where_cond` の選択・`mul` の局所勾配）も bit
/// 完全一致する。`weight` の勾配は `mul` の broadcast 縮約
/// （`reduce_to_shape`）を経由するため REQ-2 統一複合判定で比較する。
pub fn prelu<'t>(x: &Var<'t>, weight: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.check_same_tape(weight)?;
    let w_shape = weight.shape();
    if w_shape.len() != 1 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 1,
            actual: w_shape.len(),
        }));
    }
    let c = w_shape[0];
    if c == 0 {
        return Err(AutodiffError::InvalidArgument(
            "activation_ops::prelu: weight must have at least 1 element (num_parameters >= 1)"
                .into(),
        ));
    }
    let x_shape = x.shape();
    let x_rank = x_shape.len();
    let w_broadcast_shape: Vec<usize> = if c == 1 {
        // `x_rank == 0`（スカラー入力）では空 shape（`[]`）へ reshape
        // する。`x_rank.max(1)` で底上げすると rank-0 入力でも `[1]` へ
        // broadcast され、活性化演算の shape 保存契約（rank-0 入力 →
        // rank-0 出力。他の 4 活性化演算・`prelu_scalar_weight_rank0_
        // preserves_shape` テスト参照）を破る（イシュー #2146 レビュー
        // 指摘・codex #2262 スレッド・Bugbot 同旨）。
        vec![1; x_rank]
    } else {
        if x_rank < 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: x_rank,
            }));
        }
        if x_shape[1] != c {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![c],
                rhs: vec![x_shape[1]],
            }));
        }
        let mut shape = vec![1usize; x_rank];
        shape[1] = c;
        shape
    };
    let w_b = weight.reshape(&w_broadcast_shape)?;
    let pos_mask = build_value_mask(x, |v| v > 0.0)?;
    let scaled = w_b.mul(x)?;
    Var::where_cond(&pos_mask, x, &scaled)
}

/// GLU（Gated Linear Unit。PyTorch `F.glu`／`nn.GLU` 相当）:
/// `x` を軸 `dim` に沿って前後半へ 2 分割し `a * sigmoid(b)` を返す
/// （`a` が前半、`b` が後半）。
///
/// **検査**: `x` が rank 0 なら `ShapeError::AxisOutOfRange`（`dim` は
/// 必然的に範囲外）。`dim >= rank` も同様。`x.shape()[dim]` が奇数なら
/// `AutodiffError::InvalidArgument`（`chunk` は奇数長を不均等に分割
/// するため、分割前に弾く必要がある）。負の `dim` は受け付けない
/// （本クレートの慣例に従い `usize`。モジュール doc「PyTorch との
/// 既知の差分」参照）。
///
/// **数値契約**: `narrow` はコピー系演算のため `a`／`b` の抽出自体は
/// bit 完全一致する。`sigmoid` が超越関数のため forward 全体としては
/// REQ-2 統一複合判定で比較する。
pub fn glu<'t>(x: &Var<'t>, dim: usize) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let rank = shape.len();
    if rank == 0 || dim >= rank {
        return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            axis: dim,
            rank,
        }));
    }
    let axis_len = shape[dim];
    if !axis_len.is_multiple_of(2) {
        return Err(AutodiffError::InvalidArgument(format!(
            "activation_ops::glu: shape[{dim}] must be even, got {axis_len}"
        )));
    }
    let half = axis_len / 2;
    let a = x.narrow(dim, 0, half)?;
    let b = x.narrow(dim, half, half)?;
    a.mul(&b.sigmoid())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn approx_eq(a: f32, b: f32) {
        let diff = (a - b).abs();
        let rel_ok = diff / b.abs().max(1e-12) < 1e-3;
        let abs_ok = diff < 1e-5;
        assert!(
            rel_ok || abs_ok,
            "expected approx {b}, got {a} (diff={diff})"
        );
    }

    #[test]
    fn mish_forward_matches_reference() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![-1.0f32, 0.0, 1.0, 2.0], &[4]).unwrap());
        let y = mish(&x).unwrap();
        let out = y.to_tensor();
        let data = out.as_slice().unwrap();
        // 手計算参照値（softplus(x) = ln(1+exp(x))、mish = x*tanh(softplus(x))）。
        for (&xi, &yi) in [-1.0f32, 0.0, 1.0, 2.0].iter().zip(data.iter()) {
            let sp = (1.0 + xi.exp()).ln();
            let expected = xi * sp.tanh();
            approx_eq(yi, expected);
        }
    }

    #[test]
    fn mish_handles_large_positive_without_nan() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![50.0f32], &[1]).unwrap());
        let y = mish(&x).unwrap();
        let out = y.to_tensor();
        let v = out.as_slice().unwrap()[0];
        assert!(v.is_finite());
        approx_eq(v, 50.0);
    }

    #[test]
    fn hardtanh_forward_matches_clamp_bit_exact() {
        let tape = Tape::new();
        let data = vec![-5.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, f32::NAN];
        let x = tape.var(&Tensor::new(data.clone(), &[8]).unwrap());
        let y = hardtanh(&x, -1.0, 1.0).unwrap();
        let clamped = x.clamp(-1.0, 1.0).unwrap();
        let y_data = y.to_tensor();
        let c_data = clamped.to_tensor();
        let ys = y_data.as_slice().unwrap();
        let cs = c_data.as_slice().unwrap();
        for (yv, cv) in ys.iter().zip(cs.iter()) {
            if yv.is_nan() {
                assert!(cv.is_nan());
            } else {
                assert_eq!(yv.to_bits(), cv.to_bits());
            }
        }
    }

    #[test]
    fn hardtanh_backward_boundary_is_zero() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![-1.0f32, 0.0, 1.0], &[3]).unwrap());
        let y = hardtanh(&x, -1.0, 1.0).unwrap();
        let sum = y.sum(None).unwrap();
        let grads = tape.backward(&sum).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        let g = dx.host_slice();
        // -1.0・1.0 は境界（open interval 外）なので勾配 0、0.0 は区間内で勾配 1。
        assert_eq!(g[0], 0.0);
        assert_eq!(g[1], 1.0);
        assert_eq!(g[2], 0.0);
    }

    #[test]
    fn hardtanh_rejects_min_ge_max() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![0.0f32], &[1]).unwrap());
        assert!(hardtanh(&x, 1.0, 1.0).is_err());
        assert!(hardtanh(&x, 2.0, 1.0).is_err());
        assert!(hardtanh(&x, f32::NAN, 1.0).is_err());
    }

    #[test]
    fn relu6_matches_hardtanh_0_6() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![-1.0f32, 0.0, 3.0, 6.0, 7.0], &[5]).unwrap());
        let y = relu6(&x).unwrap();
        let expected = hardtanh(&x, 0.0, 6.0).unwrap();
        let yd = y.to_tensor();
        let ed = expected.to_tensor();
        assert_eq!(
            yd.as_slice()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            ed.as_slice()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn relu6_backward_zero_grad_at_zero() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![0.0f32, 3.0], &[2]).unwrap());
        let y = relu6(&x).unwrap();
        let sum = y.sum(None).unwrap();
        let grads = tape.backward(&sum).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        let g = dx.host_slice();
        assert_eq!(g[0], 0.0);
        assert_eq!(g[1], 1.0);
    }

    #[test]
    fn prelu_scalar_weight_forward_and_grad() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![-2.0f32, 3.0], &[2]).unwrap());
        let w = tape.var(&Tensor::new(vec![0.25f32], &[1]).unwrap());
        let y = prelu(&x, &w).unwrap();
        let out = y.to_tensor();
        let d = out.as_slice().unwrap();
        assert_eq!(d[0], -0.5);
        assert_eq!(d[1], 3.0);

        let sum = y.sum(None).unwrap();
        let grads = tape.backward(&sum).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        let dw = grads.get(&w).unwrap().unwrap();
        let dxs = dx.host_slice();
        assert_eq!(dxs[0], 0.25);
        assert_eq!(dxs[1], 1.0);
        approx_eq(dw.host_slice()[0], -2.0);
    }

    #[test]
    fn prelu_channelwise_weight() {
        let tape = Tape::new();
        // shape [1, 2, 2]: channel 0 と channel 1 で異なる傾きを持つ。
        let x = tape.var(&Tensor::new(vec![-1.0f32, -2.0, -3.0, -4.0], &[1, 2, 2]).unwrap());
        let w = tape.var(&Tensor::new(vec![0.1f32, 0.2], &[2]).unwrap());
        let y = prelu(&x, &w).unwrap();
        let out = y.to_tensor();
        let d = out.as_slice().unwrap();
        assert_eq!(d[0], -0.1);
        assert_eq!(d[1], -0.2);
        assert_eq!(d[2], -0.6);
        assert_eq!(d[3], -0.8);
    }

    #[test]
    fn prelu_rejects_bad_weight_shape() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![1.0f32], &[1]).unwrap());
        let w_rank2 = tape.var(&Tensor::new(vec![0.1f32], &[1, 1]).unwrap());
        assert!(prelu(&x, &w_rank2).is_err());

        let x2 = tape.var(&Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap());
        let w_c2 = tape.var(&Tensor::new(vec![0.1f32, 0.2], &[2]).unwrap());
        // rank 1 の x に C=2 の weight を適用しようとするとエラー（rank < 2 要求違反）。
        assert!(prelu(&x2, &w_c2).is_err());
    }

    /// PReLU のスカラー入力（`x` が rank 0・`weight` が `C == 1`）で
    /// 出力の rank が保存されることを固定する（イシュー #2146 レビュー
    /// 指摘・codex/#2262 スレッド・Bugbot 同旨。`w_broadcast_shape` を
    /// `x_rank.max(1)` で底上げすると `x_rank == 0` でも `[1]` へ
    /// reshape され `[] ⊕ [1] -> [1]` の rank-0 → rank-1 leak が起きる。
    /// `vec![1; x_rank]` は `x_rank == 0` で空 shape（`[]`）になり、他の
    /// 4 活性化演算と同じ rank 保存契約を満たす）。
    #[test]
    fn prelu_scalar_weight_rank0_preserves_shape() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![-4.0f32], &[]).unwrap());
        let w = tape.var(&Tensor::new(vec![0.25f32], &[1]).unwrap());
        let y = prelu(&x, &w).unwrap();
        assert_eq!(
            y.shape(),
            Vec::<usize>::new(),
            "rank-0 入力は rank-0 出力を保つこと"
        );
        let out = y.to_tensor();
        assert_eq!(out.as_slice().unwrap()[0], -1.0);

        let sum = y.sum(None).unwrap();
        let grads = tape.backward(&sum).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        let dw = grads.get(&w).unwrap().unwrap();
        assert_eq!(
            dx.shape(),
            &[] as &[usize],
            "x の勾配 shape も rank-0 を保つこと"
        );
        assert_eq!(
            dw.shape(),
            &[1usize] as &[usize],
            "weight の勾配は weight 自身の shape [1] へ縮約されること"
        );
        assert_eq!(dx.host_slice()[0], 0.25);
        approx_eq(dw.host_slice()[0], -4.0);
    }

    #[test]
    fn glu_forward_matches_reference() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![1.0f32, 2.0, 0.0, 0.0], &[4]).unwrap());
        let y = glu(&x, 0).unwrap();
        let out = y.to_tensor();
        let d = out.as_slice().unwrap();
        // a=[1,2], b=[0,0] -> sigmoid(0)=0.5 -> [0.5, 1.0]
        approx_eq(d[0], 0.5);
        approx_eq(d[1], 1.0);
    }

    #[test]
    fn glu_rejects_odd_axis_len_and_bad_dim() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap());
        assert!(glu(&x, 0).is_err());

        let x2 = tape.var(&Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap());
        assert!(glu(&x2, 1).is_err());

        let scalar = tape.var(&Tensor::new(vec![1.0f32], &[]).unwrap());
        assert!(glu(&scalar, 0).is_err());
    }

    #[test]
    fn glu_dim1_forward_matches_reference() {
        // shape [2, 4]・dim=1: 各行を前半 2 列・後半 2 列へ分割する
        // （`glu_forward_matches_reference` は dim=0 のみを検証していた
        // ため、非既定軸のカバレッジを補う）。
        let tape = Tape::new();
        let x = tape
            .var(&Tensor::new(vec![1.0f32, 2.0, 0.0, 0.0, -1.0, 3.0, 0.0, 0.0], &[2, 4]).unwrap());
        let y = glu(&x, 1).unwrap();
        let out = y.to_tensor();
        let d = out.as_slice().unwrap();
        // 行 0: a=[1,2], b=[0,0] -> sigmoid(0)=0.5 -> [0.5, 1.0]
        // 行 1: a=[-1,3], b=[0,0] -> sigmoid(0)=0.5 -> [-0.5, 1.5]
        approx_eq(d[0], 0.5);
        approx_eq(d[1], 1.0);
        approx_eq(d[2], -0.5);
        approx_eq(d[3], 1.5);
    }

    /// 中心差分（`h = 1e-3`）で `mish`／`glu` の backward を検算する
    /// （実装計画 §5「backward: 有限差分で検算する」。境界値ちょうど
    /// を避けた代表値を使う）。tape 経由の backward が独立に計算した
    /// 数値微分と一致することを確認するため、合成のどこかで符号や
    /// 係数を取り違えていないかを bit 完全一致テストとは別の角度で
    /// 検証する。
    fn finite_diff_grad(f: impl Fn(f32) -> f32, x: f32, h: f32) -> f32 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    #[test]
    fn mish_backward_matches_finite_difference() {
        fn mish_scalar(x: f32) -> f32 {
            let sp = (1.0 + x.exp()).ln();
            x * sp.tanh()
        }

        let xs = [-2.3f32, -0.7, 0.4, 1.9];
        for &xi in &xs {
            let tape = Tape::new();
            let x = tape.var(&Tensor::new(vec![xi], &[1]).unwrap());
            let y = mish(&x).unwrap();
            let grads = tape.backward(&y).unwrap();
            let dx = grads.get(&x).unwrap().unwrap();
            let analytic = dx.host_slice()[0];
            let numeric = finite_diff_grad(mish_scalar, xi, 1e-3);
            let diff = (analytic - numeric).abs();
            assert!(
                diff < 5e-2,
                "mish backward mismatch at x={xi}: analytic={analytic}, numeric={numeric}"
            );
        }
    }

    #[test]
    fn glu_backward_matches_finite_difference() {
        // glu([a, b], dim=0) = a * sigmoid(b) の a・b それぞれについて
        // 偏微分を中心差分と突合する（2 要素 fixture・境界値を避けた
        // 代表値）。
        fn glu_scalar(a: f32, b: f32) -> f32 {
            a * (1.0 / (1.0 + (-b).exp()))
        }

        let cases = [(-1.3f32, 0.6f32), (2.1, -0.9)];
        for &(a, b) in &cases {
            let tape = Tape::new();
            let x = tape.var(&Tensor::new(vec![a, b], &[2]).unwrap());
            let y = glu(&x, 0).unwrap();
            let grads = tape.backward(&y).unwrap();
            let dx = grads.get(&x).unwrap().unwrap();
            let analytic = dx.host_slice();

            let numeric_da = finite_diff_grad(|v| glu_scalar(v, b), a, 1e-3);
            let numeric_db = finite_diff_grad(|v| glu_scalar(a, v), b, 1e-3);

            assert!(
                (analytic[0] - numeric_da).abs() < 5e-2,
                "glu d/da mismatch at (a={a}, b={b}): analytic={}, numeric={numeric_da}",
                analytic[0]
            );
            assert!(
                (analytic[1] - numeric_db).abs() < 5e-2,
                "glu d/db mismatch at (a={a}, b={b}): analytic={}, numeric={numeric_db}",
                analytic[1]
            );
        }
    }
}
