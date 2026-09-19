//! GroupNorm／InstanceNorm 層（イシュー #2066・親 #2058）。
//!
//! いずれも既存の最終軸限定 `Var::layer_norm`（`nn/norm.rs`。イシュー
//! #1596）を「軸削減（reshape）→ `layer_norm`（affine なし）→ 逆
//! reshape」で呼ぶだけの合成として実装する（`Var::einsum`
//! （`crate::einsum`）と同型の「既存カーネルの再利用のみ・新規
//! `Op`／`BackendOps` を追加しない」方針）。GroupNorm 専用の
//! `Op`／VJP・カーネルは持たない。
//!
//! ## 軸削減公式
//!
//! 入力 `x` は常に `[N, C, *S]`（チャネル軸は dim 1。`*S` は 0 個以上の
//! 空間軸。`batch_norm_layout`〈`nn/batch_norm.rs`〉と同じレイアウト
//! 契約）。`c = g*cg + j`（`g` はグループ番号 `0..groups`・`j` は
//! グループ内チャネルオフセット `0..cg`・`cg = C/groups`）とすると、
//! row-major の要素順序で
//!
//! ```text
//! index(n, c, s) = (n*C + c) * spatial_numel + s
//!                = (n*groups + g) * hidden + (j*spatial_numel + s)
//! ```
//!
//! が `C = groups*cg` を代入するだけで恒等的に成立する（転置を一切
//! 伴わない）。したがって `x.reshape([rows, hidden])`
//! （`rows = N*groups`・`hidden = cg*spatial_numel`）は元の `[N,C,*S]`
//! の「グループ `g`・グループ内オフセット `j`・空間位置 `s`」を
//! 「行 = `(n,g)`・列 = `(j,s)`」へ単純に relabel するだけであり、
//! `Var::layer_norm` を最終軸（列）方向に適用すれば PyTorch
//! `nn.GroupNorm` の定義（各 `(n,g)` について `cg*spatial_numel` 要素の
//! 平均・分散で正規化）と一致する。
//!
//! `InstanceNorm` は `groups = C`（各チャネルを独立した 1 グループと
//! みなす）とした `GroupNorm` に等しい（PyTorch の既知の等価関係
//! `GroupNorm(num_groups=C) == InstanceNorm*d(C, affine=False)`）。
//!
//! ## affine 非対応（本イシューのスコープ）
//!
//! PyTorch の `nn.GroupNorm`／`nn.InstanceNorm*d` はいずれも学習可能な
//! per-channel `weight`／`bias`（`affine=True` が既定〈GroupNorm〉／
//! `affine=False` が既定〈InstanceNorm1d/2d〉）を持ちうるが、本実装は
//! **affine を持たない**（`InstanceNorm1d/2d` の既定 `affine=False` と
//! 一致・`GroupNorm` は PyTorch 既定〈`affine=True`〉と異なる）。
//!
//! 理由: affine を持たせると per-channel パラメータへの勾配縮約は
//! `[N,C,*S]` → `[1,C,1,...]` への broadcast reduce になるが、これは
//! rank-2 `[m,n]→[n]/[1,n]` 限定の `f64` 縮約経路
//! （`crate::grad::reduce_bias_grad`）に乗らず、汎用 `reduce_to_shape`
//! （`crate::grad`。純 `f32` 逐次和）にしか乗らない。
//! `.claude/rules/coding-rust.md` の「勾配の長軸縮約は `f64` 相当」
//! 契約に抵触しうるため、本イシューでは affine 対応を見送り、別
//! イシューのスコープとする（`docs/norm-ops-design.md` §11 参照）。

use fandhe_ai_tensor_core::{BackendError, BackendOps, ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval;
use crate::var::Var;

/// GroupNorm 層の既定 `eps`（PyTorch `nn.GroupNorm` の既定値）。
pub const GROUP_NORM_DEFAULT_EPS: f32 = 1e-5;
/// InstanceNorm 層の既定 `eps`（PyTorch `nn.InstanceNorm1d`／
/// `InstanceNorm2d` の既定値）。
pub const INSTANCE_NORM_DEFAULT_EPS: f32 = 1e-5;

/// `eps` の fail-closed 検査（`nn/norm.rs::validate_eps` と同型の複製。
/// 各 `nn` モジュールファイルが独自に `validate_eps` を持つ既存慣習
/// 〈`norm.rs`／`batch_norm.rs`〉に倣う）。
fn validate_eps(eps: f32, who: &str) -> Result<(), AutodiffError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{who}: eps must be finite and non-negative, got {eps}"
        )));
    }
    Ok(())
}

/// `dims` の要素数の積を `usize` オーバーフロー検査つきで計算する
/// （`tensor-core::checked_numel` は `pub(crate)` で `autodiff` から
/// 到達できないため、`Var::reshape`／`crate::einsum` と同型の
/// `checked_mul`／`try_fold` を本クレート側で自前実装する）。
fn checked_product(dims: &[usize]) -> Result<usize, AutodiffError> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))
}

/// GroupNorm／InstanceNorm（`groups = C`）が起動前に `(rows, hidden)`
/// を導出するための shape 検査。`nn::normalization` 内でのみ使う
/// private ヘルパー（`tensor-core::row_norm_layout`／`batch_norm_layout`
/// と同じ位置付けだが、GroupNorm 固有のためこのモジュールに閉じる）。
///
/// - `shape` が rank < 2 の場合 `ShapeError::RankMismatch`
///   （`expected: 2`）を返す（チャネル軸〈dim 1〉自体が存在しないため）。
/// - `groups == 0` は `AutodiffError::InvalidArgument` で拒否する
///   （呼び出し元のコンストラクタ引数検査。`Linear::new` の
///   `in_features == 0` 拒否と同型の「構築不可能な引数を計算前に弾く」
///   規律）。
/// - `c % groups != 0` は `AutodiffError::InvalidArgument` で拒否する。
/// - `n == 0 || cg == 0 || spatial にゼロを含む`（テンソル全体が空
///   要素数になる場合。`cg == 0` は `c == 0` の場合のみ成立する—— `c %
///   groups == 0` かつ `groups > 0` を満たしつつ `c/groups == 0` と
///   なるのは `c == 0` のときのみ）は積の計算自体を避け `hidden = 0`
///   を返す（`batch_norm_layout` の「4 軸のいずれかが 0 なら
///   spatial=0」規約と同型。巨大な軸長を伴う正当な空テンソルを誤って
///   overflow 扱いしないため）。この場合 `rows` は `n` を返す
///   （`hidden = 0` のとき `reshape([rows, hidden])` は `rows` の値に
///   依らず要素数 0 で一致するため、どの `rows` を選んでも後続の
///   `reshape` は成功する）。
pub(crate) fn group_norm_layout(
    shape: &[usize],
    groups: usize,
    who: &str,
) -> Result<(usize, usize), AutodiffError> {
    let rank = shape.len();
    if rank < 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: rank,
        }));
    }
    if groups == 0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{who}: groups must be non-zero"
        )));
    }
    let n = shape[0];
    let c = shape[1];
    let spatial = &shape[2..];
    if !c.is_multiple_of(groups) {
        return Err(AutodiffError::InvalidArgument(format!(
            "{who}: num_channels ({c}) must be divisible by groups ({groups})"
        )));
    }
    let cg = c / groups;
    if n == 0 || cg == 0 || spatial.contains(&0) {
        return Ok((n, 0));
    }
    let spatial_numel = checked_product(spatial)?;
    let hidden = cg
        .checked_mul(spatial_numel)
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
    let rows = n
        .checked_mul(groups)
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
    Ok((rows, hidden))
}

/// `tape` を経由せず `ops` を直接呼んで GroupNorm／InstanceNorm の
/// forward をホスト常駐 `Tensor` で計算する（`Module::forward_host`
/// の本体。`nn::norm::RmsNorm`／`LayerNorm` の `forward_host` と同じ
/// 判定規律——`ops.layer_norm` を試み `Unsupported` のときのみ
/// `eval::layer_norm_rows` へフォールバックする。`.claude/rules/
/// security.md` A08「判定迂回経路を作らない」）。`crate::nn::module`
/// の `impl Module for GroupNorm`／`InstanceNorm` から呼ばれる
/// （`pub(crate)`）。
pub(crate) fn group_norm_forward_host(
    ops: &dyn BackendOps,
    input: &Tensor<f32>,
    groups: usize,
    eps: f32,
    who: &str,
) -> Result<Tensor<f32>, AutodiffError> {
    let x_shape = input.shape().to_vec();
    let (rows, hidden) = group_norm_layout(&x_shape, groups, who)?;
    let reshaped = input.contiguous().reshape(&[rows, hidden])?;
    let value = match ops.layer_norm(&reshaped, None, None, eps) {
        Ok(v) => v,
        Err(BackendError::Unsupported(_)) => {
            eval::layer_norm_rows(&reshaped, None, None, eps, rows, hidden)
        }
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    if value.shape() != reshaped.shape() {
        return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: value.shape().to_vec(),
                rhs: reshaped.shape().to_vec(),
            },
        )));
    }
    Ok(value.reshape(&x_shape)?)
}

/// GroupNorm 層（PyTorch `nn.GroupNorm` 相当。**affine なし**
/// 〈モジュール doc「affine 非対応」節参照〉）。学習可能パラメータを
/// 持たない値型のため `Copy` を導出する（`nn::Dropout` 等の無状態層と
/// 同型）。
#[derive(Debug, Clone, Copy)]
pub struct GroupNorm {
    groups: usize,
    eps: f32,
}

impl GroupNorm {
    /// `groups == 0` は構築不可能な引数として計算前に拒否する
    /// （`Linear::new` の `in_features == 0` 拒否と同型）。`eps` は
    /// `validate_eps` で有限かつ非負であることを検証する。
    /// `num_channels` との整合性（`num_channels % groups == 0`）は
    /// 入力 shape が定まる forward 時（[`Self::forward`]）に検査する
    /// （層自体は `num_channels` を保持しないため）。
    pub fn new(groups: usize, eps: f32) -> Result<Self, AutodiffError> {
        validate_eps(eps, "GroupNorm::new")?;
        if groups == 0 {
            return Err(AutodiffError::InvalidArgument(
                "GroupNorm::new: groups must be non-zero".to_string(),
            ));
        }
        Ok(Self { groups, eps })
    }

    /// グループ数（[`Self::new`] の引数）。
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// `eps`（`var(x) + eps` の加算項）。構築時に `validate_eps` で
    /// 有限かつ非負であることを検証済み。
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// `input: [N, C, *S]` を `[N*groups, (C/groups)*|S|]` へ reshape
    /// し（モジュール doc「軸削減公式」参照）、`Var::layer_norm`
    /// （affine なし）を適用してから元の shape へ戻す。`tape` 引数を
    /// 取らない理由: 学習可能パラメータを持たないため
    /// `bind(&tape)` 手順が不要（`RmsNorm`／`LayerNorm::forward` との
    /// 違い）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let x_shape = input.shape();
        let (rows, hidden) = group_norm_layout(&x_shape, self.groups, "GroupNorm::forward")?;
        let reshaped = input.contiguous()?.reshape(&[rows, hidden])?;
        let normalized = reshaped.layer_norm(None, None, self.eps)?;
        normalized.reshape(&x_shape)
    }
}

/// InstanceNorm 層（PyTorch `nn.InstanceNorm1d`／`InstanceNorm2d`
/// 相当。**affine なし**〈PyTorch 既定 `affine=False` と一致〉）。
/// `groups = num_channels`（入力 shape から forward 時に導出）とした
/// [`GroupNorm`] に等しい（モジュール doc「軸削減公式」節末尾）。
#[derive(Debug, Clone, Copy)]
pub struct InstanceNorm {
    eps: f32,
}

impl InstanceNorm {
    /// `eps` は `validate_eps` で有限かつ非負であることを検証する。
    /// `GroupNorm::new` と異なり `groups` を構築時に持たない——入力の
    /// `shape[1]`（チャネル数）から forward 時に導出するため。
    pub fn new(eps: f32) -> Result<Self, AutodiffError> {
        validate_eps(eps, "InstanceNorm::new")?;
        Ok(Self { eps })
    }

    /// `eps`（`var(x) + eps` の加算項）。
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// `input: [N, C, *S]`（`*S` は 1 個以上の空間軸。rank < 3 は
    /// `ShapeError::RankMismatch` で拒否する——PyTorch
    /// `InstanceNorm1d`／`InstanceNorm2d` が最低 1 個の空間軸
    /// 〈`[N,C,L]`／`[N,C,H,W]`〉を要求する契約と揃える。[`GroupNorm`]
    /// の rank 制約〈rank >= 2〉より厳しい制約を個別に課す理由は
    /// InstanceNorm が「各チャネルを独立に正規化する」という定義上、
    /// 空間軸 0 個〈`[N,C]`〉では GroupNorm との違いが意味をなさない
    /// ため）。`num_channels == 0` の場合は入力を恒等（`contiguous`
    /// コピー）で返す（`groups = 0` を `group_norm_layout` へ渡すと
    /// 構築時 `groups == 0` 検査〈ユーザー入力起因のエラー〉と混同する
    /// ため、入力 shape 由来の空テンソルはここで個別に短絡させる）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let x_shape = input.shape();
        let rank = x_shape.len();
        if rank < 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: rank,
            }));
        }
        let c = x_shape[1];
        if c == 0 {
            return input.contiguous();
        }
        let (rows, hidden) = group_norm_layout(&x_shape, c, "InstanceNorm::forward")?;
        let reshaped = input.contiguous()?.reshape(&[rows, hidden])?;
        let normalized = reshaped.layer_norm(None, None, self.eps)?;
        normalized.reshape(&x_shape)
    }
}

// `crate::nn::module::Module` への統合（`as_group_norm`／
// `as_instance_norm` フック・`impl Module for GroupNorm`／
// `InstanceNorm`）は `nn/module.rs` に配置する（`RmsNorm`／`LayerNorm`
// と同じ配置規約——`Module` trait 自身とその実装は `module.rs` に
// 集約し、層本体を持つファイルは trait を知らない。`module.rs` 側の
// `impl Module for GroupNorm`／`InstanceNorm` が本ファイルの
// `pub(crate) fn group_norm_forward_host` を直接呼ぶ）。

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data.to_vec(), shape).expect("test fixture: shape とデータ長は一致させている")
    }

    fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
        t.as_slice()
            .expect("test: expected contiguous tensor")
            .to_vec()
    }

    // --- 構築時検査 ---

    #[test]
    fn group_norm_new_rejects_zero_groups() {
        let err = GroupNorm::new(0, GROUP_NORM_DEFAULT_EPS).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn group_norm_new_rejects_non_finite_eps() {
        let err = GroupNorm::new(2, f32::NAN).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        let err = GroupNorm::new(2, -1e-5).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn instance_norm_new_rejects_non_finite_eps() {
        let err = InstanceNorm::new(f32::NAN).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        let err = InstanceNorm::new(-1e-5).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    // --- forward の shape 検査 ---

    #[test]
    fn group_norm_forward_rejects_rank_below_2() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[1.0], &[1]));
        let layer = GroupNorm::new(1, GROUP_NORM_DEFAULT_EPS).unwrap();
        let err = layer.forward(&x).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn group_norm_forward_rejects_non_divisible_channels() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[0.0; 6], &[1, 3, 2]));
        let layer = GroupNorm::new(2, GROUP_NORM_DEFAULT_EPS).unwrap();
        let err = layer.forward(&x).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn instance_norm_forward_rejects_rank_below_3() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let layer = InstanceNorm::new(INSTANCE_NORM_DEFAULT_EPS).unwrap();
        let err = layer.forward(&x).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    // --- 数値検証 ---

    /// biased 分散（÷N）で各グループを正規化した期待値を `f64` で
    /// 計算するテスト専用ヘルパー（`Var::layer_norm` doc「分散は
    /// biased（÷N）」と同じ定義）。
    fn hand_computed_group_norm(groups: &[Vec<f32>], eps: f32) -> Vec<f32> {
        groups
            .iter()
            .flat_map(|g| {
                let n = g.len() as f64;
                let mean = g.iter().map(|&v| v as f64).sum::<f64>() / n;
                let var = g.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
                let rstd = 1.0 / (var + eps as f64).sqrt();
                g.iter()
                    .map(move |&v| ((v as f64 - mean) * rstd) as f32)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// `N=1, C=4, groups=2, spatial=[2]` の手計算突合。各グループ
    /// （チャネル対 (0,1)・(2,3)）それぞれ 4 要素（`cg=2 *
    /// spatial_numel=2`）の平均・分散（biased・÷N）で正規化される。
    #[test]
    fn group_norm_forward_matches_hand_computed_values() {
        // x[n=0, c, s]。groups=2 → group0 = channels {0,1} = [1,2,3,4]
        // （4 要素）・group1 = channels {2,3} = [5,6,7,8]（4 要素）。
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&data, &[1, 4, 2]));
        let layer = GroupNorm::new(2, 1e-5).unwrap();
        let out = dense_vec(&layer.forward(&x).unwrap().to_tensor());

        let expected = hand_computed_group_norm(&[data[0..4].to_vec(), data[4..8].to_vec()], 1e-5);
        assert_eq!(out.len(), expected.len());
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "o={o} e={e}");
        }
    }

    /// `groups=1` は「全非バッチ軸を最終軸へ畳んだ `layer_norm`」と
    /// bit 一致する（軸削減公式の境界ケース検証）。
    #[test]
    fn group_norm_groups_eq_1_matches_layer_norm_over_all_non_batch_dims() {
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.37 - 3.0).collect();
        let shape = [2usize, 3, 4];

        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&data, &shape));
        let via_group_norm = dense_vec(
            &GroupNorm::new(1, 1e-5)
                .unwrap()
                .forward(&x)
                .unwrap()
                .to_tensor(),
        );

        let x2 = tape.var(&t(&data, &shape));
        let via_manual = dense_vec(
            &x2.reshape(&[2, 12])
                .unwrap()
                .layer_norm(None, None, 1e-5)
                .unwrap()
                .reshape(&shape)
                .unwrap()
                .to_tensor(),
        );

        assert_eq!(via_group_norm, via_manual);
    }

    /// `groups = channels` の `GroupNorm` と `InstanceNorm` は同一入力
    /// で bit 一致する（PyTorch の既知の等価関係の直接検証）。
    #[test]
    fn group_norm_groups_eq_channels_matches_instance_norm() {
        let data: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.5).collect();
        let shape = [2usize, 4, 3];

        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x1 = tape.var(&t(&data, &shape));
        let via_group_norm = dense_vec(
            &GroupNorm::new(4, 1e-5)
                .unwrap()
                .forward(&x1)
                .unwrap()
                .to_tensor(),
        );

        let x2 = tape.var(&t(&data, &shape));
        let via_instance_norm = dense_vec(
            &InstanceNorm::new(1e-5)
                .unwrap()
                .forward(&x2)
                .unwrap()
                .to_tensor(),
        );

        assert_eq!(via_group_norm, via_instance_norm);
    }

    #[test]
    fn group_norm_zero_channels_returns_empty_output_with_matching_shape() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[], &[2, 0, 3]));
        let layer = GroupNorm::new(1, GROUP_NORM_DEFAULT_EPS).unwrap();
        let out = layer.forward(&x).unwrap().to_tensor();
        assert_eq!(out.shape(), &[2, 0, 3]);
    }

    #[test]
    fn instance_norm_zero_channels_returns_input_unchanged() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[], &[2, 0, 3]));
        let layer = InstanceNorm::new(INSTANCE_NORM_DEFAULT_EPS).unwrap();
        let out = layer.forward(&x).unwrap().to_tensor();
        assert_eq!(out.shape(), &[2, 0, 3]);
    }

    // --- 勾配（中央差分突合） ---
    //
    // `autodiff` は `backend-cpu` 等の具体バックエンドクレートに依存
    // しない（`test_support.rs` モジュール doc 参照）ため、実カーネル
    // ではなく `test_support::test_ops()`（`eval.rs` の naive 参照実装
    // へ委譲する `TestOps`。`layer_norm` はオーバーライドしないため
    // 常に `eval::layer_norm_rows` フォールバックを経由する）を解析
    // 勾配・数値勾配の両方で共有し、同一の forward 経路であることを
    // 保証する。

    const GRAD_H: f64 = 1e-3;
    const GRAD_TAU: f32 = 1e-4;
    const GRAD_REL_TOL: f32 = 1e-2;
    const GRAD_ABS_TOL: f32 = 1e-3;

    fn assert_grad_close(label: &str, analytic: &[f32], numeric: &[f32]) {
        assert_eq!(analytic.len(), numeric.len(), "{label}: 要素数不一致");
        for (i, (&a, &n)) in analytic.iter().zip(numeric.iter()).enumerate() {
            let diff = (a - n).abs();
            let rel = diff / a.abs().max(n.abs()).max(GRAD_TAU);
            assert!(
                rel <= GRAD_REL_TOL || diff <= GRAD_ABS_TOL,
                "{label}[{i}]: analytic={a} numeric={n} diff={diff} rel={rel}"
            );
        }
    }

    /// `L(x) = Σ (forward(x) ⊙ s)` の中央差分勾配（要素ごと・`f64`
    /// 集計。`grad.rs::numeric_grad_unary` と同じ方式だが、GroupNorm／
    /// InstanceNorm はテンソル全体を縮約する演算のため `Tensor ->
    /// Tensor` の `group_norm_forward_host`（tape 不要）を直接叩いて
    /// 評価する——毎回フレッシュな `Tape` を張り直す必要がない分
    /// シンプルになる）。
    fn numeric_grad(
        x: &Tensor<f32>,
        s: &[f32],
        forward: impl Fn(&Tensor<f32>) -> Tensor<f32>,
    ) -> Vec<f32> {
        let shape = x.shape().to_vec();
        let mut data = dense_vec(x);
        let mut grad = vec![0f32; data.len()];
        for i in 0..data.len() {
            let orig = data[i] as f64;
            data[i] = (orig + GRAD_H) as f32;
            let lp = scalar_dot(&forward(&t(&data, &shape)), s);
            data[i] = (orig - GRAD_H) as f32;
            let lm = scalar_dot(&forward(&t(&data, &shape)), s);
            data[i] = orig as f32;
            grad[i] = ((lp - lm) / (2.0 * GRAD_H)) as f32;
        }
        grad
    }

    fn scalar_dot(a: &Tensor<f32>, s: &[f32]) -> f64 {
        dense_vec(a)
            .iter()
            .zip(s.iter())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum()
    }

    #[test]
    fn group_norm_grad_matches_numeric() {
        let shape = [2usize, 4, 2];
        let data: Vec<f32> = (0..16).map(|i| (i as f32 - 7.5) * 0.31).collect();
        let s: Vec<f32> = (0..16)
            .map(|i| ((i * 3 + 1) % 5) as f32 * 0.2 - 0.4)
            .collect();
        let groups = 2usize;
        let eps = 1e-5;

        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&data, &shape));
        let s_var = tape.var(&t(&s, &shape));
        let out = GroupNorm::new(groups, eps).unwrap().forward(&x).unwrap();
        let proj = out.mul(&s_var).unwrap().sum(None).unwrap();
        let grads = tape.backward(&proj).unwrap();
        let dx_analytic = dense_vec(grads.get(&x).unwrap().expect("到達する"));

        let host_ops = crate::test_support::test_ops();
        let dx_numeric = numeric_grad(&t(&data, &shape), &s, |x_tensor| {
            group_norm_forward_host(host_ops.as_ref(), x_tensor, groups, eps, "test").unwrap()
        });

        assert_grad_close("group_norm dx", &dx_analytic, &dx_numeric);
    }

    #[test]
    fn instance_norm_grad_matches_numeric() {
        let shape = [1usize, 3, 3];
        let data: Vec<f32> = (0..9).map(|i| (i as f32 - 4.0) * 0.42).collect();
        let s: Vec<f32> = (0..9)
            .map(|i| ((i * 2 + 1) % 4) as f32 * 0.25 - 0.3)
            .collect();
        let eps = 1e-5;

        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&data, &shape));
        let s_var = tape.var(&t(&s, &shape));
        let out = InstanceNorm::new(eps).unwrap().forward(&x).unwrap();
        let proj = out.mul(&s_var).unwrap().sum(None).unwrap();
        let grads = tape.backward(&proj).unwrap();
        let dx_analytic = dense_vec(grads.get(&x).unwrap().expect("到達する"));

        let host_ops = crate::test_support::test_ops();
        let dx_numeric = numeric_grad(&t(&data, &shape), &s, |x_tensor| {
            group_norm_forward_host(host_ops.as_ref(), x_tensor, 3, eps, "test").unwrap()
        });

        assert_grad_close("instance_norm dx", &dx_analytic, &dx_numeric);
    }

    // --- Module 統合（module.rs の `impl Module for GroupNorm`／
    // `InstanceNorm` を経由。`crate::nn::module::Module` を明示 import
    // して直接呼び出しと突合する） ---

    #[test]
    fn group_norm_module_forward_matches_direct_call() {
        use crate::nn::module::Module;

        let shape = [1usize, 4, 2];
        let data: Vec<f32> = (0..8).map(|i| i as f32 * 0.5 - 1.0).collect();
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let layer = GroupNorm::new(2, GROUP_NORM_DEFAULT_EPS).unwrap();

        let x1 = tape.var(&t(&data, &shape));
        let via_module = dense_vec(&Module::forward(&layer, &tape, &x1).unwrap().to_tensor());

        let x2 = tape.var(&t(&data, &shape));
        let via_direct = dense_vec(&layer.forward(&x2).unwrap().to_tensor());

        assert_eq!(via_module, via_direct);
    }

    #[test]
    fn instance_norm_module_forward_matches_direct_call() {
        use crate::nn::module::Module;

        let shape = [1usize, 3, 2];
        let data: Vec<f32> = (0..6).map(|i| i as f32 * 0.7 - 1.5).collect();
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let layer = InstanceNorm::new(INSTANCE_NORM_DEFAULT_EPS).unwrap();

        let x1 = tape.var(&t(&data, &shape));
        let via_module = dense_vec(&Module::forward(&layer, &tape, &x1).unwrap().to_tensor());

        let x2 = tape.var(&t(&data, &shape));
        let via_direct = dense_vec(&layer.forward(&x2).unwrap().to_tensor());

        assert_eq!(via_module, via_direct);
    }

    #[test]
    fn group_norm_named_parameters_is_empty() {
        use crate::nn::module::Module;
        let layer = GroupNorm::new(2, GROUP_NORM_DEFAULT_EPS).unwrap();
        assert!(Module::named_parameters(&layer).is_empty());
    }

    #[test]
    fn instance_norm_named_parameters_is_empty() {
        use crate::nn::module::Module;
        let layer = InstanceNorm::new(INSTANCE_NORM_DEFAULT_EPS).unwrap();
        assert!(Module::named_parameters(&layer).is_empty());
    }

    #[test]
    fn group_norm_forward_host_matches_var_forward() {
        use crate::nn::module::Module;

        let shape = [2usize, 4, 3];
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.13 - 1.5).collect();
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let layer = GroupNorm::new(2, GROUP_NORM_DEFAULT_EPS).unwrap();

        let x = tape.var(&t(&data, &shape));
        let via_tape = dense_vec(&layer.forward(&x).unwrap().to_tensor());

        let host_ops = crate::test_support::test_ops();
        let via_host =
            dense_vec(&Module::forward_host(&layer, host_ops.as_ref(), &t(&data, &shape)).unwrap());

        assert_eq!(via_tape, via_host);
    }

    #[test]
    fn instance_norm_forward_host_matches_var_forward() {
        use crate::nn::module::Module;

        let shape = [2usize, 3, 3];
        let data: Vec<f32> = (0..18).map(|i| i as f32 * 0.21 - 2.0).collect();
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let layer = InstanceNorm::new(INSTANCE_NORM_DEFAULT_EPS).unwrap();

        let x = tape.var(&t(&data, &shape));
        let via_tape = dense_vec(&layer.forward(&x).unwrap().to_tensor());

        let host_ops = crate::test_support::test_ops();
        let via_host =
            dense_vec(&Module::forward_host(&layer, host_ops.as_ref(), &t(&data, &shape)).unwrap());

        assert_eq!(via_tape, via_host);
    }
}
