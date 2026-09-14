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
//! **`Module` trait は実装しない（確定判断）**: `Module::forward` は
//! f32 `Var` 入力契約だが、embedding の入力は整数クラス id
//! （`Tensor<i32>`）であり型が一致しない。さらに `compat::Sequential`
//! の `bind`／`trainable_parameters`／`apply_parameters` は `as_linear`
//! フックにしか反応しないため、`Module` を実装すると `Sequential` に
//! 積んだ `Embedding` が**黙って学習されない**罠になる（重みが
//! optimizer の対象から漏れる）。`Module`／`Sequential` 統合
//! （`as_embedding` フック・f32 id 入力の受理）は本イシューのスコープ
//! 外として別イシューへ引き継ぐ（`.claude/rules/out-of-scope-tracking.md`）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::init::{WEIGHT_SEED_SALT, derive_seed, normal_init};
use crate::tape::Tape;
use crate::var::Var;

/// Embedding 層のパラメータ本体。`weight` は
/// `[num_embeddings, embedding_dim]`。
pub struct Embedding {
    weight: Tensor<f32>,
    padding_idx: Option<usize>,
}

impl Embedding {
    /// 決定的シードで `N(0, 1)`（標準正規分布。PyTorch `nn.Embedding`
    /// 既定初期化）で初期化する（`nn/init.rs::normal_init` 参照。
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
        let weight_seed = derive_seed(seed, WEIGHT_SEED_SALT);
        let mut weight_data = normal_init(num_embeddings * embedding_dim, weight_seed);
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
        })
    }

    /// このステップの `tape` へ `weight` を葉ノードとして登録し、
    /// `forward` を呼べる `EmbeddingVars` を返す（`Linear::bind` と
    /// 同型）。
    pub fn bind<'t>(&self, tape: &'t Tape) -> EmbeddingVars<'t> {
        let weight = tape.var(&self.weight);
        EmbeddingVars {
            weight,
            padding_idx: self.padding_idx,
        }
    }

    pub fn weight(&self) -> &Tensor<f32> {
        &self.weight
    }

    pub fn padding_idx(&self) -> Option<usize> {
        self.padding_idx
    }

    pub fn num_embeddings(&self) -> usize {
        self.weight.shape()[0]
    }

    pub fn embedding_dim(&self) -> usize {
        self.weight.shape()[1]
    }
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
}
