//! Embedding 層（イシュー #1604。`docs/compat-api-scope.md` §1.2 Tier
//! 1「Embedding」）。
//!
//! `nn::Linear`（`linear.rs`）・`nn::RmsNorm`／`LayerNorm`（`norm.rs`）と
//! 同じ「本体（`Embedding`。`Tensor<f32>` を永続保持する層パラメータ）
//! → `bind(&tape)` で `Var` 化した `EmbeddingVars`（1 ステップ分の
//! テープ登録済みパラメータ）」の分離パターンを踏襲する（`tape.rs` の
//! 「学習ループでの運用」節参照。`Tape` はステップごとに生成・破棄
//! される前提のため）。
//!
//! forward の実体は `Var::embedding`（`var.rs`）——`Op::Embedding`
//! （`tape.rs`）として tape に記録され、`BackendOps::gather`／
//! `scatter`（イシュー #1776 で 3 バックエンドとも実装済み）の上に
//! 直接乗るため、本モジュール自体は新規カーネルを持たない。
//!
//! **`Module` trait の実装（イシュー #1760 で解消）**: `Module::forward`
//! は f32 `Var` 入力契約だが、embedding の入力は本来整数クラス id
//! （`Tensor<i32>`）であり型が一致しない。本イシューでは
//! [`EmbeddingVars::forward_from_var`] を新設し、入力の `Var<f32>` を
//! `Var::to_tensor()` で実体化したうえで [`ids_from_f32`]（非有限・
//! 非整数・負・`i32::MAX` 超過を fail-closed に拒否する厳格変換。
//! 黙示の飽和・切り捨て変換はしない。`.claude/rules/security.md`
//! A03）を通して整数 id へ変換することでこの型不一致を橋渡しする。
//! `compat::Sequential` 側は [`crate::nn::module::Module::as_embedding`]
//! フックで `Embedding` 層を認識する（`as_linear` と同型。旧版がここで
//! 挙げていた「`Sequential` に積んだ `Embedding` が黙って学習されない
//! 罠」はこのフックで解消済み）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::init::{WEIGHT_SEED_SALT, derive_seed, try_normal_init};
use crate::tape::Tape;
use crate::var::Var;

/// Embedding 層のパラメータ本体。`weight` は
/// `[num_embeddings, embedding_dim]`。
pub struct Embedding {
    weight: Tensor<f32>,
    padding_idx: Option<usize>,
    /// 層別 `requires_grad` 凍結フラグ（イシュー #2137。`nn::Linear`
    /// と同型）。既定 `true`。
    requires_grad: bool,
}

impl Embedding {
    /// 決定的シードで `N(0, 1)`（標準正規分布。PyTorch `nn.Embedding`
    /// 既定初期化）で初期化する（`nn/init.rs::try_normal_init` 参照。
    /// `Linear::new` と同じくグローバル `manual_seed` 状態から独立）。
    /// `padding_idx` を渡すと当該行を全ゼロで初期化する（PyTorch
    /// `nn.Embedding(padding_idx=..)` の既定挙動）。
    ///
    /// `num_embeddings == 0` は `AutodiffError::InvalidArgument` で
    /// 拒否する（`Linear::new` の `in_features == 0` 拒否と同型の
    /// 判断: 0 行テーブルは全 id が構造的に範囲外になり、`Var::
    /// embedding` の呼び出し不能な層を静かに構築してしまうため）。
    /// `embedding_dim == 0` は妥当な shape（`tensor-core` はサイズ 0
    /// 軸を許容する）としてそのまま受理する（`Linear::new` の
    /// `out_features == 0` 許容と対称。この非対称は本 doc の契約と
    /// して固定する）。`padding_idx >= num_embeddings` は
    /// `AutodiffError::InvalidArgument` で拒否する。
    ///
    /// `num_embeddings * embedding_dim`（重み要素数）は `checked_mul`
    /// で検証してから初期化する。未検査で乗算すると大きな
    /// `num_embeddings`／`embedding_dim`（例:
    /// `Embedding::new(usize::MAX / 2 + 1, 2, ..)`）で overflow し、
    /// `Tensor::new` の検証に到達する前に panic しうる（本番経路
    /// panic 禁止。`.claude/rules/coding-rust.md`。`nn::rnn::
    /// build_gate_params` と同型の対応。codex-review P1 指摘）。続く
    /// 重み確保自体も `try_normal_init`（`Vec::try_reserve_exact` 使用）
    /// で行い、要素数乗算が overflow しなくても確保バイト数が
    /// `isize::MAX` を超える場合（例:
    /// `Embedding::new(1, isize::MAX as usize, ..)`）に panic せず
    /// `AutodiffError::InvalidArgument` を返す（codex-review P1
    /// 指摘）。
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        seed: u64,
    ) -> Result<Embedding, AutodiffError> {
        if num_embeddings == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Embedding::new: num_embeddings must be > 0".to_string(),
            ));
        }
        if let Some(p) = padding_idx
            && p >= num_embeddings
        {
            return Err(AutodiffError::InvalidArgument(format!(
                "Embedding::new: padding_idx {p} が範囲 [0, {num_embeddings}) を外れている"
            )));
        }
        // `num_embeddings * embedding_dim` の要素数乗算を `checked_mul`
        // で検証する（`nn::rnn::build_gate_params` と同型。イシュー
        // #1604 codex-review P1 指摘: 未検査乗算は大きな入力で
        // overflow し `Tensor::new` の検証に到達する前に panic する）。
        let weight_len = num_embeddings.checked_mul(embedding_dim).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "Embedding::new: num_embeddings (={num_embeddings}) * embedding_dim (={embedding_dim}) \
                 overflowed usize"
            ))
        })?;
        let weight_seed = derive_seed(seed, WEIGHT_SEED_SALT);
        // `try_normal_init` は `Vec::try_reserve_exact` で確保可否を
        // 先に検証するため、`weight_len` の乗算自体は overflow しなく
        // ても確保バイト数が `isize::MAX` を超える場合に panic せず
        // `Err` を返す（イシュー #1604 codex-review P1 指摘）。
        let mut weight_data = try_normal_init(weight_len, weight_seed).map_err(|err| {
            AutodiffError::InvalidArgument(format!(
                "Embedding::new: weight (num_embeddings={num_embeddings}, embedding_dim={embedding_dim}, \
                 len={weight_len}) 分のバッファを確保できません: {err}"
            ))
        })?;
        if let Some(p) = padding_idx {
            let start = p * embedding_dim;
            let end = start + embedding_dim;
            for v in &mut weight_data[start..end] {
                *v = 0.0;
            }
        }
        let weight = Tensor::new(weight_data, &[num_embeddings, embedding_dim])?;
        Ok(Embedding {
            weight,
            padding_idx,
            requires_grad: true,
        })
    }

    /// 明示的な重みから構築する（safetensors ロード相当の
    /// `from_pretrained` 入口。テスト・将来の永続化経路向け。イシュー
    /// #1604）。`weight` は rank 2 かつ `shape()[0] > 0` を要求する
    /// （`Linear::from_parameters` の「壊れた checkpoint を静かに
    /// 受理しない」契約と同型。A03: 外部由来パラメータを計算前に
    /// 検証する。`.claude/rules/security.md`）。与えられた `weight` の
    /// 値はそのまま保持する（`padding_idx` 行を上書きしない——
    /// `from_pretrained` は既存の学習済み値を尊重する）。
    pub fn from_parameters(
        weight: Tensor<f32>,
        padding_idx: Option<usize>,
    ) -> Result<Embedding, AutodiffError> {
        if weight.rank() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: weight.rank(),
            }));
        }
        let num_embeddings = weight.shape()[0];
        if num_embeddings == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Embedding::from_parameters: weight.shape()[0] (num_embeddings) must be > 0"
                    .to_string(),
            ));
        }
        if let Some(p) = padding_idx
            && p >= num_embeddings
        {
            return Err(AutodiffError::InvalidArgument(format!(
                "Embedding::from_parameters: padding_idx {p} が範囲 [0, {num_embeddings}) を\
                 外れている"
            )));
        }
        Ok(Embedding {
            weight,
            padding_idx,
            requires_grad: true,
        })
    }

    /// このステップの `tape` へ `weight` を葉ノードとして登録し、
    /// `forward` を呼べる `EmbeddingVars` を返す（`Linear::bind` と
    /// 同型）。
    pub fn bind<'t>(&self, tape: &'t Tape) -> EmbeddingVars<'t> {
        let weight = tape.var_with_requires_grad(&self.weight, self.requires_grad);
        EmbeddingVars {
            weight,
            padding_idx: self.padding_idx,
        }
    }

    /// 現在の重み（`[num_embeddings, embedding_dim]`）への参照を返す。
    pub fn weight(&self) -> &Tensor<f32> {
        &self.weight
    }

    /// [`crate::nn::module::Module::set_requires_grad`]（`Embedding`
    /// 実装。`module.rs` 参照）の本体（イシュー #2137）。
    pub(crate) fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// [`crate::nn::module::Module::requires_grad`]（`Embedding` 実装）
    /// の本体。
    pub(crate) fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// `Embedding::new`／`from_parameters` に渡した `padding_idx` を
    /// そのまま返す。
    pub fn padding_idx(&self) -> Option<usize> {
        self.padding_idx
    }

    /// 埋め込みテーブルの行数（`weight.shape()[0]`）を返す。
    pub fn num_embeddings(&self) -> usize {
        self.weight.shape()[0]
    }

    /// 埋め込みベクトルの次元数（`weight.shape()[1]`）を返す。
    pub fn embedding_dim(&self) -> usize {
        self.weight.shape()[1]
    }

    /// [`crate::nn::module::Module::set_parameter`]（`impl Module for
    /// Embedding`。`module.rs` 参照）の本体（イシュー #1760）。
    /// `nn::norm::RmsNorm::set_parameter` と同型（shape 保存置換のみ）
    /// だが、`Embedding` は affine なし構成を持たないため対象は
    /// `"weight"` 1 件のみ。
    pub(crate) fn set_parameter(
        &mut self,
        name: &str,
        value: Tensor<f32>,
    ) -> Result<(), AutodiffError> {
        match name {
            "weight" => {
                if value.shape() != self.weight.shape() {
                    return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                        lhs: value.shape().to_vec(),
                        rhs: self.weight.shape().to_vec(),
                    }));
                }
                self.weight = value;
                Ok(())
            }
            _ => Err(AutodiffError::InvalidArgument(format!(
                "Embedding::set_parameter: no parameter named `{name}`"
            ))),
        }
    }
}

/// [`EmbeddingVars::forward_from_var`] が使う、f32 テンソルの各要素を
/// 厳格に i32 id へ変換する補助関数（イシュー #1760）。`Module::forward`
/// の f32 `Var` 契約と [`Var::embedding`] の整数 id 契約を橋渡しする。
///
/// 非有限（NaN／inf）・非整数（`fract() != 0.0`）・負・`i32::MAX` 超過の
/// いずれも `AutodiffError::InvalidArgument` で拒否する（黙示の飽和・
/// 切り捨て変換はしない。`.claude/rules/security.md` A03。
/// `tensor-core::cast` の NaN→0 飽和変換とは異なる方針を意図的に取る）。
/// 範囲 `>= num_embeddings` の検査は行わず [`Var::embedding`] 側の
/// 既存検査へ委譲する（重複実装しない。REQ-9）。
///
/// `v as f64` へ一度昇格してから `i32::MAX as f64` と比較する理由:
/// `i32::MAX`（`2^31 - 1`）は f32 で正確に表現できず、`i32::MAX as f32`
/// は `2^31` へ丸め上がる（`as` キャストは float→int で飽和するため
/// 境界値の判定を静かに緩めてしまう）。`f32 → f64` の昇格は無損失
/// なので、`f64` 側で `i32::MAX` と比較すれば境界を厳密に判定できる。
fn ids_from_f32(input: &Tensor<f32>) -> Result<Tensor<i32>, AutodiffError> {
    let dense = input.contiguous();
    let values = dense.as_slice().ok_or_else(|| {
        AutodiffError::InvalidArgument(
            "Embedding::forward_from_var: contiguous() 直後の as_slice() が None（内部不変条件 \
             違反）"
                .to_string(),
        )
    })?;
    let mut ids = Vec::with_capacity(values.len());
    for &v in values {
        if !v.is_finite() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Embedding::forward_from_var: id は有限値のみ許容する（got {v}）"
            )));
        }
        if v.fract() != 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Embedding::forward_from_var: id は整数値のみ許容する（got {v}）"
            )));
        }
        if v < 0.0 || (v as f64) > i32::MAX as f64 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Embedding::forward_from_var: id は範囲 [0, {}] を外れている（got {v}）",
                i32::MAX
            )));
        }
        ids.push(v as i32);
    }
    Tensor::new(ids, input.shape()).map_err(AutodiffError::Shape)
}

/// `Embedding::bind` が返す、1 ステップ分のテープに登録済み
/// パラメータ（`LinearVars` と同型）。`weight` を公開する理由:
/// `Tape::backward` 後に `Gradients::get(&vars.weight)` で勾配を
/// 取り出すのは呼び出し側（optimizer）の責務であり、`EmbeddingVars`
/// 自身は勾配更新 API を持たない。
pub struct EmbeddingVars<'t> {
    pub weight: Var<'t>,
    padding_idx: Option<usize>,
}

impl<'t> EmbeddingVars<'t> {
    /// `ids` が指す行を抽出する（`Var::embedding` への薄い委譲。
    /// `nn::Linear::forward` と同じ「本体は `var.rs` 側」方針）。
    pub fn forward(&self, ids: &Tensor<i32>) -> Result<Var<'t>, AutodiffError> {
        self.weight.embedding(ids, self.padding_idx)
    }

    /// `Module::forward`（f32 `Var` 契約）から embedding を呼べるように
    /// する橋渡し（イシュー #1760。モジュール doc「`Module` trait の
    /// 実装」節参照）。`input` を層 1（fallible）実体化境界
    /// `crate::tape::materialize_fallible` で実体化し
    /// `ids_from_f32`（非公開のためコードスパン表記で参照しリンク化
    /// しない。`nn/attention.rs` の `sdpa_compose` 等と同じ規約）で
    /// 厳格に整数 id へ変換してから [`Self::forward`] へ委譲する。
    ///
    /// **`Var::to_tensor()`（層 2・非 fallible 境界）を使わない理由**
    /// （codex-review P0 指摘・イシュー #1760 PR #1884 レビュー）:
    /// `to_tensor()` は checkpoint 解放済みノードの再計算失敗
    /// （poison）をゼロテンソルへ黙って吸収する契約
    /// （`materialize_non_fallible` doc 「poison 契約」節参照）。
    /// この吸収されたゼロが `ids_from_f32` の「非有限・非整数・負を
    /// 拒否」検査は素通りし（`0.0` は有限な整数値）「有効な id 0」
    /// として本関数がそのまま `forward` へ渡してしまい、かつ
    /// `Tensor<i32>` へ変換した時点でテープ追跡から切り離されるため
    /// （doc 下記「勾配経路を持たない」参照）、後続の
    /// `Tape::backward`（層 1・`materialize_fallible` 経由で poison を
    /// `AutodiffError` として検出する fail-closed 契約）からもこの
    /// 汚染を検知する経路がなくなる。`materialize_fallible` を直接
    /// 使えば、poison 発生時にここで即座に `Err` として呼び出し元へ
    /// 伝播し、汚染されたゼロを「有効な id」として取り込む事故を防ぐ。
    ///
    /// **`input` 自身は勾配経路を持たない**: [`Var::embedding`] の
    /// `index` 引数が常に非 `Var`（生 `Tensor<i32>`）である契約と同じ
    /// で、embedding の添字に有意味な勾配は存在しないため（他クラスへ
    /// 動かした場合の劣化を表す連続な勾配が定義できない）、`input` を
    /// 一度実体化してテープ追跡を切り離すのは意図的な設計。
    pub fn forward_from_var(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let input_tensor = {
            let nodes = input.tape().nodes.borrow();
            crate::tape::materialize_fallible(&nodes, input.tape().ops(), input.node_id())?.clone()
        };
        let ids = ids_from_f32(&input_tensor)?;
        self.forward(&ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_padding_row_is_all_zero() {
        let emb = Embedding::new(4, 3, Some(1), 42).unwrap();
        let w = emb.weight();
        let row: Vec<f32> = (0..3).map(|c| w.get(&[1, c]).unwrap()).collect();
        assert_eq!(row, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn new_non_padding_rows_are_non_zero() {
        let emb = Embedding::new(4, 3, Some(1), 42).unwrap();
        let w = emb.weight();
        for r in [0usize, 2, 3] {
            let row: Vec<f32> = (0..3).map(|c| w.get(&[r, c]).unwrap()).collect();
            assert!(
                row.iter().any(|&v| v != 0.0),
                "row {r} は非ゼロ値を含むはず: {row:?}"
            );
        }
    }

    #[test]
    fn new_rejects_zero_num_embeddings() {
        let Err(err) = Embedding::new(0, 3, None, 1) else {
            panic!("Embedding::new(0, ..) は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn new_accepts_zero_embedding_dim() {
        let emb = Embedding::new(4, 0, None, 1).unwrap();
        assert_eq!(emb.weight().shape(), &[4, 0]);
    }

    #[test]
    fn new_rejects_padding_idx_out_of_range() {
        let Err(err) = Embedding::new(4, 3, Some(4), 1) else {
            panic!("padding_idx 範囲外は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    // codex-review P1 指摘（イシュー #1604）の回帰テスト: 要素数乗算
    // オーバーフローを `checked_mul` で検査せずに `Tensor::new` へ
    // 到達すると panic していた（本番経路 panic 禁止）。
    #[test]
    fn new_rejects_element_count_overflow_without_panicking() {
        let Err(err) = Embedding::new(usize::MAX / 2 + 1, 2, Some(0), 42) else {
            panic!("num_embeddings * embedding_dim の overflow は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    // codex-review P1 指摘（イシュー #1604）の回帰テスト: 要素数乗算が
    // overflow しなくても `Vec::with_capacity` 相当の確保が
    // `isize::MAX` バイトを超えると capacity overflow で panic して
    // いた。`try_reserve_exact` で確保失敗を型付きエラーへ変換する。
    #[test]
    fn new_rejects_allocation_too_large_without_panicking() {
        let Err(err) = Embedding::new(1, isize::MAX as usize, None, 42) else {
            panic!("確保不能なほど大きい weight は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn from_parameters_rejects_rank_mismatch() {
        let w = Tensor::<f32>::zeros(&[4, 3, 2]).unwrap();
        let Err(err) = Embedding::from_parameters(w, None) else {
            panic!("rank 不一致は Err を返すはず")
        };
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::RankMismatch { .. })
        ));
    }

    #[test]
    fn from_parameters_rejects_zero_num_embeddings() {
        let w = Tensor::<f32>::zeros(&[0, 3]).unwrap();
        let Err(err) = Embedding::from_parameters(w, None) else {
            panic!("num_embeddings == 0 は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn from_parameters_preserves_given_values() {
        let w = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let emb = Embedding::from_parameters(w.clone(), None).unwrap();
        assert_eq!(
            emb.weight().host_slice().into_owned(),
            w.host_slice().into_owned()
        );
    }

    #[test]
    fn same_seed_produces_same_weights() {
        let a = Embedding::new(4, 3, None, 42).unwrap();
        let b = Embedding::new(4, 3, None, 42).unwrap();
        assert_eq!(
            a.weight().host_slice().into_owned(),
            b.weight().host_slice().into_owned()
        );
    }

    #[test]
    fn linear_new_is_unaffected_by_global_manual_seed_state() {
        // `nn::Linear` と同じ「グローバル manual_seed 状態から独立」
        // 契約（モジュール冒頭コメント参照）。
        fandhe_ai_tensor_core::rng::manual_seed(1);
        let a = Embedding::new(4, 3, None, 42).unwrap();
        fandhe_ai_tensor_core::rng::manual_seed(999);
        let b = Embedding::new(4, 3, None, 42).unwrap();
        assert_eq!(
            a.weight().host_slice().into_owned(),
            b.weight().host_slice().into_owned()
        );
    }

    // イシュー #1760: `ids_from_f32`（f32 → 厳格な i32 変換）の単体
    // テスト。

    #[test]
    fn ids_from_f32_accepts_integer_values() {
        let input = Tensor::new(vec![0.0, 1.0, 3.0, 2.0], &[2, 2]).unwrap();
        let ids = ids_from_f32(&input).unwrap();
        assert_eq!(ids.shape(), &[2, 2]);
        let values: Vec<i32> = (0..4).map(|i| ids.get(&[i / 2, i % 2]).unwrap()).collect();
        assert_eq!(values, vec![0, 1, 3, 2]);
    }

    #[test]
    fn ids_from_f32_rejects_non_integer() {
        let input = Tensor::new(vec![1.5_f32], &[1]).unwrap();
        let Err(err) = ids_from_f32(&input) else {
            panic!("非整数値は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn ids_from_f32_rejects_negative() {
        let input = Tensor::new(vec![-1.0_f32], &[1]).unwrap();
        let Err(err) = ids_from_f32(&input) else {
            panic!("負値は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn ids_from_f32_rejects_nan_and_inf() {
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let input = Tensor::new(vec![v], &[1]).unwrap();
            let Err(err) = ids_from_f32(&input) else {
                panic!("非有限値 {v} は Err を返すはず")
            };
            assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        }
    }

    #[test]
    fn ids_from_f32_rejects_out_of_i32_range() {
        // `i32::MAX` は f32 で正確に表現できないため、代表的に大きい
        // 有限整数値（f32 で厳密に表現できる `2^31` ちょうど）で検証
        // する（doc comment「`v as f64` へ一度昇格」節参照）。
        let input = Tensor::new(vec![2_147_483_648.0_f32], &[1]).unwrap();
        let Err(err) = ids_from_f32(&input) else {
            panic!("i32::MAX 超過は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn ids_from_f32_accepts_zero() {
        let input = Tensor::new(vec![0.0_f32], &[1]).unwrap();
        let ids = ids_from_f32(&input).unwrap();
        assert_eq!(ids.get(&[0]).unwrap(), 0);
    }

    // `EmbeddingVars::forward_from_var` / `impl Module for Embedding`
    // の単体テスト（イシュー #1760）。

    #[test]
    fn forward_from_var_matches_forward_with_raw_ids() {
        let emb = Embedding::new(4, 3, None, 42).unwrap();
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let raw_ids = Tensor::<i32>::new(vec![0, 2, 1], &[3]).unwrap();
        let expected = emb.bind(&tape).forward(&raw_ids).unwrap().to_tensor();

        let ids_as_f32 = Tensor::new(vec![0.0_f32, 2.0, 1.0], &[3]).unwrap();
        let input_var = tape.var(&ids_as_f32);
        let actual = emb
            .bind(&tape)
            .forward_from_var(&input_var)
            .unwrap()
            .to_tensor();

        assert_eq!(
            expected.host_slice().into_owned(),
            actual.host_slice().into_owned()
        );
    }

    #[test]
    fn module_forward_rejects_non_integer_input() {
        use crate::nn::module::Module;

        let emb = Embedding::new(4, 3, None, 42).unwrap();
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let bad_ids = Tensor::new(vec![1.5_f32], &[1]).unwrap();
        let input_var = tape.var(&bad_ids);
        let Err(err) = Module::forward(&emb, &tape, &input_var) else {
            panic!("非整数 id は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn set_parameter_replaces_weight_preserving_shape() {
        let mut emb = Embedding::new(4, 3, None, 42).unwrap();
        let new_weight = Tensor::new(vec![9.0_f32; 12], &[4, 3]).unwrap();
        emb.set_parameter("weight", new_weight.clone()).unwrap();
        assert_eq!(
            emb.weight().host_slice().into_owned(),
            new_weight.host_slice().into_owned()
        );
    }

    #[test]
    fn set_parameter_rejects_shape_mismatch() {
        let mut emb = Embedding::new(4, 3, None, 42).unwrap();
        let wrong_shape = Tensor::new(vec![1.0_f32; 6], &[2, 3]).unwrap();
        let Err(err) = emb.set_parameter("weight", wrong_shape) else {
            panic!("shape 不一致は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::Shape(_)));
    }

    #[test]
    fn set_parameter_rejects_unknown_name() {
        let mut emb = Embedding::new(4, 3, None, 42).unwrap();
        let value = Tensor::new(vec![1.0_f32; 12], &[4, 3]).unwrap();
        let Err(err) = emb.set_parameter("bogus", value) else {
            panic!("未知名は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }
}
