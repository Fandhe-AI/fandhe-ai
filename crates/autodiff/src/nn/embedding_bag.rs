//! EmbeddingBag 層（イシュー #2161・親 #2131。`docs/compat-api-scope.md`
//! §1.2 Tier 1「Embedding」の深掘り）。
//!
//! `nn::Embedding`（`embedding.rs`）が行う gather（`Var::embedding`）に
//! 続けて、bag（行の集合）単位で `sum`／`mean`／`max` へ縮約する
//! （`torch.nn.EmbeddingBag` 相当）。新規 `Op`／`BackendOps`／VJP／
//! カーネルは追加しない合成のみで構成する（REQ-9）。
//!
//! # id 範囲検査を `Var::embedding` へ一本化する設計（orphan node 回避）
//!
//! [`EmbeddingBagVars::forward_with_offsets`] は bag ごとに個別へ
//! `Var::embedding` を呼ばない。全 bag の（`padding_idx` を除外した）
//! id を host 側で 1 本の配列へ連結し、**単一の** [`crate::var::Var::
//! embedding`] 呼び出しでまとめて gather する。`Var::embedding` は
//! バックエンド呼び出し（≒ tape へのノード追加）より前に id 範囲を
//! 検査してから `push_eager` する契約（`var.rs::embedding` doc
//! 「検査順序」節）を持つため、この 1 回の呼び出しは「全 bag ぶんの
//! id が妥当なら成功しノードを 1 つ積む／範囲外 id が 1 件でもあれば
//! 何もノードを積まず即座に `Err`」という**原子的**な単位になる。
//! bag ごとに `embedding` を呼ぶ設計だと、後方の bag で範囲外 id が
//! 見つかった時点で前方の bag が既にノードを積んでいる（失敗時に
//! tape へ孤児ノードが残る）ため、この設計で回避する。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_i32;
use crate::nn::embedding::ids_from_f32;
use crate::nn::init::{WEIGHT_SEED_SALT, derive_seed, try_normal_init};
use crate::tape::Tape;
use crate::var::Var;

/// bag 単位の縮約方法（`torch.nn.EmbeddingBag(mode=..)` 相当）。
/// `#[non_exhaustive]`: 将来 `Sum`／`Mean`／`Max` 以外の縮約
/// （PyTorch には存在しないが本クレート独自の拡張余地）を非破壊で
/// 追加できるようにする（`fandhe-ai-autodiff` は crates.io 公開クレート。
/// `docs/crates-io-naming-decision.md`）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingBagMode {
    /// bag 内の埋め込みベクトルの和。
    Sum,
    /// bag 内の埋め込みベクトルの平均（PyTorch 既定）。
    Mean,
    /// bag 内の埋め込みベクトルの要素ごと最大値。同値タイの勾配配分は
    /// [`crate::var::Var::max`]（`grad.rs::max_vjp`）の「先勝ち決定的」
    /// 規約に従う（`docs/autodiff-amax-grad-distribution-decision.md`
    /// 参照）。
    Max,
}

impl Default for EmbeddingBagMode {
    /// PyTorch `torch.nn.EmbeddingBag` の既定値（`mode='mean'`）。
    fn default() -> Self {
        Self::Mean
    }
}

/// EmbeddingBag 層のパラメータ本体。`weight` は
/// `[num_embeddings, embedding_dim]`（[`crate::nn::embedding::
/// Embedding`] と同一 shape 契約）。
pub struct EmbeddingBag {
    weight: Tensor<f32>,
    mode: EmbeddingBagMode,
    padding_idx: Option<usize>,
    /// 層別 `requires_grad` 凍結フラグ（`Embedding` と同型。イシュー
    /// #2137）。既定 `true`。
    requires_grad: bool,
}

impl EmbeddingBag {
    /// [`crate::nn::embedding::Embedding::new`] と同じ構築規律（決定的
    /// シードで `N(0, 1)` 初期化・`padding_idx` 行はゼロ初期化・要素数
    /// 乗算 overflow／確保失敗を `checked_mul`／`try_normal_init` で
    /// 型付きエラー化）に `mode` を加えたもの。
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        mode: EmbeddingBagMode,
        padding_idx: Option<usize>,
        seed: u64,
    ) -> Result<EmbeddingBag, AutodiffError> {
        if num_embeddings == 0 {
            return Err(AutodiffError::InvalidArgument(
                "EmbeddingBag::new: num_embeddings must be > 0".to_string(),
            ));
        }
        if let Some(p) = padding_idx
            && p >= num_embeddings
        {
            return Err(AutodiffError::InvalidArgument(format!(
                "EmbeddingBag::new: padding_idx {p} が範囲 [0, {num_embeddings}) を外れている"
            )));
        }
        let weight_len = num_embeddings.checked_mul(embedding_dim).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "EmbeddingBag::new: num_embeddings (={num_embeddings}) * embedding_dim \
                 (={embedding_dim}) overflowed usize"
            ))
        })?;
        let weight_seed = derive_seed(seed, WEIGHT_SEED_SALT);
        let mut weight_data = try_normal_init(weight_len, weight_seed).map_err(|err| {
            AutodiffError::InvalidArgument(format!(
                "EmbeddingBag::new: weight (num_embeddings={num_embeddings}, \
                 embedding_dim={embedding_dim}, len={weight_len}) 分のバッファを確保できません: \
                 {err}"
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
        Ok(EmbeddingBag {
            weight,
            mode,
            padding_idx,
            requires_grad: true,
        })
    }

    /// [`crate::nn::embedding::Embedding::from_parameters`] と同じ
    /// 検査規律（rank 2・`shape()[0] > 0`・`padding_idx` 範囲）。
    pub fn from_parameters(
        weight: Tensor<f32>,
        mode: EmbeddingBagMode,
        padding_idx: Option<usize>,
    ) -> Result<EmbeddingBag, AutodiffError> {
        if weight.rank() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: weight.rank(),
            }));
        }
        let num_embeddings = weight.shape()[0];
        if num_embeddings == 0 {
            return Err(AutodiffError::InvalidArgument(
                "EmbeddingBag::from_parameters: weight.shape()[0] (num_embeddings) must be > 0"
                    .to_string(),
            ));
        }
        if let Some(p) = padding_idx
            && p >= num_embeddings
        {
            return Err(AutodiffError::InvalidArgument(format!(
                "EmbeddingBag::from_parameters: padding_idx {p} が範囲 [0, {num_embeddings}) を\
                 外れている"
            )));
        }
        Ok(EmbeddingBag {
            weight,
            mode,
            padding_idx,
            requires_grad: true,
        })
    }

    /// このステップの `tape` へ `weight` を葉ノードとして登録し、
    /// `forward` を呼べる `EmbeddingBagVars` を返す（`Embedding::bind`
    /// と同型）。
    pub fn bind<'t>(&self, tape: &'t Tape) -> EmbeddingBagVars<'t> {
        let weight = tape.var_with_requires_grad(&self.weight, self.requires_grad);
        EmbeddingBagVars {
            weight,
            mode: self.mode,
            padding_idx: self.padding_idx,
        }
    }

    /// 現在の重み（`[num_embeddings, embedding_dim]`）への参照を返す。
    pub fn weight(&self) -> &Tensor<f32> {
        &self.weight
    }

    /// 構築済みの縮約方法。
    pub fn mode(&self) -> EmbeddingBagMode {
        self.mode
    }

    /// `EmbeddingBag::new`／`from_parameters` に渡した `padding_idx`。
    pub fn padding_idx(&self) -> Option<usize> {
        self.padding_idx
    }

    /// 埋め込みテーブルの行数（`weight.shape()[0]`）。
    pub fn num_embeddings(&self) -> usize {
        self.weight.shape()[0]
    }

    /// 埋め込みベクトルの次元数（`weight.shape()[1]`）。
    pub fn embedding_dim(&self) -> usize {
        self.weight.shape()[1]
    }

    /// [`crate::nn::module::Module::set_requires_grad`]（`EmbeddingBag`
    /// 実装）の本体。
    pub(crate) fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// [`crate::nn::module::Module::requires_grad`]（`EmbeddingBag`
    /// 実装）の本体。
    pub(crate) fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// [`crate::nn::module::Module::set_parameter`]（`EmbeddingBag`
    /// 実装）の本体。`Embedding::set_parameter` と同型（shape 保存
    /// 置換のみ・対象は `"weight"` 1 件のみ）。
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
                "EmbeddingBag::set_parameter: no parameter named `{name}`"
            ))),
        }
    }
}

/// `EmbeddingBag::bind` が返す、1 ステップ分のテープに登録済み
/// パラメータ（`EmbeddingVars` と同型）。
pub struct EmbeddingBagVars<'t> {
    pub weight: Var<'t>,
    mode: EmbeddingBagMode,
    padding_idx: Option<usize>,
}

impl<'t> EmbeddingBagVars<'t> {
    /// 長さの揃った bag（`ids`: `[B, L]`）をまとめて処理する。
    ///
    /// - `padding_idx` が `None` かつ `L > 0` の**高速経路**:
    ///   `weight.embedding(ids, None)` で `[B, L, D]` を得てから
    ///   `dim=1` を縮約する（`Var::embedding` 呼び出しが 1 回のみで
    ///   済み、[`Self::forward_with_offsets`] の bag ごと連結処理を
    ///   経由しない）
    /// - それ以外（`padding_idx` が `Some`、または `L == 0`）は
    ///   `offsets = [0, L, 2L, .., B*L]`（`include_last_offset = true`）
    ///   を組み立てて [`Self::forward_with_offsets`] へ委譲する
    ///
    /// `ids` の rank が 2 でない場合は
    /// [`AutodiffError::Shape`]（[`ShapeError::RankMismatch`]）。
    pub fn forward(&self, ids: &Tensor<i32>) -> Result<Var<'t>, AutodiffError> {
        if ids.rank() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: ids.rank(),
            }));
        }
        let shape = ids.shape();
        let (b, l) = (shape[0], shape[1]);

        if self.padding_idx.is_none() && l > 0 {
            let embedded = self.weight.embedding(ids, None)?;
            return match self.mode {
                EmbeddingBagMode::Sum => embedded.sum(Some(1)),
                EmbeddingBagMode::Mean => embedded.mean(Some(1)),
                EmbeddingBagMode::Max => embedded.max(Some(1)),
            };
        }

        let flat_data = dense_vec_i32(ids);
        let flat_len = b.checked_mul(l).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "EmbeddingBagVars::forward: B (={b}) * L (={l}) overflowed usize"
            ))
        })?;
        let flat = Tensor::new(flat_data, &[flat_len]).map_err(AutodiffError::Shape)?;
        let offsets: Vec<usize> = (0..=b).map(|i| i * l).collect();
        self.forward_with_offsets(&flat, &offsets, true)
    }

    /// 可変長 bag（`ids`: rank 1 `[N]`・`offsets`: bag の開始位置列）を
    /// 処理する（`torch.nn.functional.embedding_bag(..., offsets=..)`
    /// 相当）。
    ///
    /// `include_last_offset == false` のとき `offsets.len()` が
    /// bag 数（最後の bag の終端は暗黙に `N`）、`true` のとき
    /// `offsets.len() - 1` が bag 数（`offsets` 末尾が明示的に `N` と
    /// 一致する必要がある）。
    ///
    /// 検査順序（モジュール doc「id 範囲検査を `Var::embedding` へ
    /// 一本化する設計」節参照。いずれも tape を操作する前に完了する）:
    /// ①`ids` が rank 1 → ②`offsets` が空でない → ③`offsets[0] == 0`
    /// → ④`offsets` が単調非減少 → ⑤全 `offsets` が `<= N` →
    /// ⑥`include_last_offset` のとき末尾が `== N` → ⑦bag 数が `> 0`
    /// （`0` は [`crate::var::Var::cat`] が空リストを拒否するため事前に
    /// 拒否する）。
    pub fn forward_with_offsets(
        &self,
        ids: &Tensor<i32>,
        offsets: &[usize],
        include_last_offset: bool,
    ) -> Result<Var<'t>, AutodiffError> {
        if ids.rank() != 1 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 1,
                actual: ids.rank(),
            }));
        }
        let n = ids.shape()[0];

        if offsets.is_empty() {
            return Err(AutodiffError::InvalidArgument(
                "EmbeddingBagVars::forward_with_offsets: offsets must not be empty".to_string(),
            ));
        }
        if offsets[0] != 0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "EmbeddingBagVars::forward_with_offsets: offsets[0] must be 0, got {}",
                offsets[0]
            )));
        }
        for w in offsets.windows(2) {
            if w[1] < w[0] {
                return Err(AutodiffError::InvalidArgument(format!(
                    "EmbeddingBagVars::forward_with_offsets: offsets must be non-decreasing \
                     (got {} then {})",
                    w[0], w[1]
                )));
            }
        }
        for &o in offsets {
            if o > n {
                return Err(AutodiffError::InvalidArgument(format!(
                    "EmbeddingBagVars::forward_with_offsets: offset {o} exceeds ids length \
                     ({n})"
                )));
            }
        }
        let last = match offsets.last() {
            Some(&last) => last,
            None => {
                return Err(AutodiffError::InvalidArgument(
                    "EmbeddingBagVars::forward_with_offsets: offsets must not be empty".to_string(),
                ));
            }
        };
        if include_last_offset && last != n {
            return Err(AutodiffError::InvalidArgument(format!(
                "EmbeddingBagVars::forward_with_offsets: include_last_offset requires \
                 offsets.last() == ids.len() ({n}), got {last}"
            )));
        }
        let bag_count = if include_last_offset {
            offsets.len() - 1
        } else {
            offsets.len()
        };
        let embedding_dim = self.weight.shape()[1];
        if bag_count == 0 {
            // 空バッチ（`B == 0`。呼び出し元の bag 数がそもそも 0 件）:
            // `Var::cat` は空リストを拒否するため、通常の bag 構築経路
            // （後段の `Var::cat(&bags, 0)`）を経由せず shape
            // `[0, embedding_dim]` の Var を直接返す（cursor[bot] Low
            // 指摘「Empty batch rejected with padding」。PR #2281 是正）。
            // 要素数 0 のため実データを持たず、`weight` への勾配寄与も
            // 構造的に発生しない（backward 対象要素が存在しない）ので
            // `var_no_grad` で十分（[`Self::forward`] の高速経路も
            // 同様に B==0 では embedded/縮約の結果が空 shape になる）。
            let empty = Tensor::<f32>::zeros(&[0, embedding_dim]).map_err(AutodiffError::Shape)?;
            return Ok(self.weight.tape().var_no_grad(&empty));
        }

        let bag_bounds: Vec<(usize, usize)> = (0..bag_count)
            .map(|i| {
                let start = offsets[i];
                let end = if i + 1 < offsets.len() {
                    offsets[i + 1]
                } else {
                    n
                };
                (start, end)
            })
            .collect();

        // padding_idx と一致する id を除外しつつ全 bag ぶんの id を
        // 1 本へ連結する（モジュール doc 参照。除外は id の値が
        // `padding_idx` と厳密一致する場合のみ——範囲外・負の id は
        // ここでは弾かず、後続の単一 `Var::embedding` 呼び出しの検査に
        // 委ねる）。
        let ids_flat = dense_vec_i32(ids);
        let padding_idx = self.padding_idx;
        let mut filtered_ids: Vec<i32> = Vec::with_capacity(n);
        let mut bag_filtered_bounds: Vec<(usize, usize)> = Vec::with_capacity(bag_count);
        for &(start, end) in &bag_bounds {
            let filtered_start = filtered_ids.len();
            for &id in &ids_flat[start..end] {
                let is_padding = padding_idx.is_some_and(|p| id >= 0 && id as usize == p);
                if !is_padding {
                    filtered_ids.push(id);
                }
            }
            bag_filtered_bounds.push((filtered_start, filtered_ids.len()));
        }

        let filtered_len = filtered_ids.len();
        let filtered_tensor =
            Tensor::new(filtered_ids, &[filtered_len]).map_err(AutodiffError::Shape)?;
        // ここが本モジュール唯一の `Var::embedding` 呼び出し
        // （モジュール doc「id 範囲検査を `Var::embedding` へ一本化する
        // 設計」節）。
        let embedded_all = self.weight.embedding(&filtered_tensor, None)?;

        // 空 bag（`bag_len == 0` になる可能性のある bag が 1 件でも
        // あれば構築する。バッチ内の全 bag が空になるケース（codex
        // P1／cursor[bot] Medium 指摘「Empty bags detach bag output」・
        // PR #2281 是正）に備え、`weight` の行 0 への微分可能な経路を
        // 保ったままゼロ値を出力する必要がある（`weight.narrow(0, 0,
        // 1)` は `num_embeddings > 0` により常に有効）。単純に
        // `tape.var_no_grad` のゼロ定数を使うと、その bag が
        // `requires_grad == false` のまま `Var::cat` へ渡り、バッチ内
        // 全 bag が空の場合は `cat` 出力全体が `requires_grad == false`
        // に落ちて `Tape::backward` が失敗する（`weight` への期待される
        // ゼロ勾配が得られない）。
        //
        // **`weight_row0.mul(&zero_scale)`（旧実装）の NaN／inf 汚染
        // （codex P1 指摘・PR #2281 是正）**: `mul` の forward 値は
        // 両オペランドの実際の積（IEEE 754）であるため、`weight` の
        // 行 0 に `NaN`／`±inf` が含まれていると `NaN * 0.0 == NaN`・
        // `inf * 0.0 == NaN` となり、空 bag の出力が「常にゼロを返す」
        // という公開契約（PyTorch `nn.EmbeddingBag` の空 bag 契約）を
        // 破る。`from_parameters`／`set_parameter` は重みの有限性を
        // 検証しないため、この経路は重みの値次第で壊れていた。
        //
        // **`masked_fill` による修正**: `Var::masked_fill` は
        // 選択操作（`out[i] = if mask[i] { value } else { input[i] }`。
        // `eval::masked_fill`／各バックエンド `masked_fill` 実装。
        // `crates/backend-cpu/src/elementwise.rs::masked_fill_slice`
        // 参照）であり、`mul` と異なり value と input を算術的に
        // 組み合わせない。全要素を mask する
        // （`weight_row0.masked_fill(&all_true, 0.0)`）ことで:
        // - forward 値は `input`（`weight_row0`）の実際の値に一切
        //   依存せず常に定数 `0.0`（有限）になる（`weight_row0` が
        //   `NaN`／`inf` を含んでいても出力へ伝播しない）。
        // - backward（`masked_fill_vjp`）は mask された位置の勾配を
        //   常に `0.0` にする（`upstream` と `mask` のみから計算し
        //   `input` の値を読まないため、こちらも `weight` の実際の
        //   値に依存せず安全）。
        // - `self`（`weight_row0`）がグラフ中に存在するため出力の
        //   `requires_grad` は `weight` へ正しくリンクされたまま
        //   （旧実装と同じく全 bag 空バッチでも `Tape::backward` が
        //   成立する）。
        let weight_row0 = self.weight.narrow(0, 0, 1)?;
        let all_masked = Tensor::<bool>::new(vec![true; embedding_dim], &[1, embedding_dim])
            .map_err(AutodiffError::Shape)?;

        let mut bags: Vec<Var<'t>> = Vec::with_capacity(bag_count);
        for &(fstart, fend) in &bag_filtered_bounds {
            let bag_len = fend - fstart;
            if bag_len == 0 {
                // 空 bag（元々長さ 0、または全要素が padding_idx で
                // 除外された）はゼロ行（PyTorch `nn.EmbeddingBag` の
                // 空 bag 出力契約）。値は常に `0.0`（`weight` の値に
                // 非依存）のまま `weight` への微分可能な経路を保つ
                // （上記コメント参照）。
                bags.push(weight_row0.masked_fill(&all_masked, 0.0)?);
                continue;
            }
            let slice = embedded_all.narrow(0, fstart, bag_len)?;
            let reduced = match self.mode {
                EmbeddingBagMode::Sum => slice.sum(Some(0))?,
                EmbeddingBagMode::Mean => slice.mean(Some(0))?,
                EmbeddingBagMode::Max => slice.max(Some(0))?,
            };
            bags.push(reduced.reshape(&[1, embedding_dim])?);
        }
        Var::cat(&bags, 0)
    }

    /// `Module::forward`（f32 `Var` 契約）から呼べるようにする橋渡し
    /// （[`crate::nn::embedding::EmbeddingVars::forward_from_var`] と
    /// 同型。`materialize_fallible`〈層 1・fail-closed〉を使う理由も
    /// 同一——`to_tensor()`〈層 2〉の poison 吸収を避ける）。`input`
    /// は rank 2（`[B, L]`）を要求する（[`Self::forward`] の rank
    /// 検査へ委譲）。
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
    use crate::eval::dense_vec;

    fn tape() -> Tape {
        Tape::new_with_ops(crate::test_support::test_ops())
    }

    #[test]
    fn new_rejects_zero_num_embeddings() {
        let Err(err) = EmbeddingBag::new(0, 3, EmbeddingBagMode::Mean, None, 1) else {
            panic!("num_embeddings == 0 は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn new_rejects_padding_idx_out_of_range() {
        let Err(err) = EmbeddingBag::new(4, 3, EmbeddingBagMode::Mean, Some(4), 1) else {
            panic!("padding_idx 範囲外は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn new_padding_row_is_all_zero() {
        let bag = EmbeddingBag::new(4, 3, EmbeddingBagMode::Sum, Some(1), 42).unwrap();
        let w = bag.weight();
        let row: Vec<f32> = (0..3).map(|c| w.get(&[1, c]).unwrap()).collect();
        assert_eq!(row, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn from_parameters_rejects_rank_mismatch() {
        let w = Tensor::<f32>::zeros(&[4, 3, 2]).unwrap();
        let Err(err) = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Mean, None) else {
            panic!("rank 不一致は Err を返すはず")
        };
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::RankMismatch { .. })
        ));
    }

    #[test]
    fn default_mode_is_mean() {
        assert_eq!(EmbeddingBagMode::default(), EmbeddingBagMode::Mean);
    }

    /// `sum`／`mean`／`max` が「`Embedding` + 手動縮約」と一致すること
    /// （rank 2・`padding_idx` なしの高速経路）。
    #[test]
    fn forward_matches_manual_reduction_rank2() {
        let w = Tensor::<f32>::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[4, 3],
        )
        .unwrap();
        let ids = Tensor::<i32>::new(vec![0, 1, 2, 3], &[2, 2]).unwrap();

        for mode in [
            EmbeddingBagMode::Sum,
            EmbeddingBagMode::Mean,
            EmbeddingBagMode::Max,
        ] {
            let t = tape();
            let bag = EmbeddingBag::from_parameters(w.clone(), mode, None).unwrap();
            let out = bag.bind(&t).forward(&ids).unwrap();
            let got = dense_vec(&out.to_tensor());

            // 手動縮約: bag0 = rows [0,1]、bag1 = rows [2,3]。
            let expected: Vec<f32> = match mode {
                EmbeddingBagMode::Sum => vec![
                    1.0 + 4.0,
                    2.0 + 5.0,
                    3.0 + 6.0,
                    7.0 + 10.0,
                    8.0 + 11.0,
                    9.0 + 12.0,
                ],
                EmbeddingBagMode::Mean => vec![
                    (1.0 + 4.0) / 2.0,
                    (2.0 + 5.0) / 2.0,
                    (3.0 + 6.0) / 2.0,
                    (7.0 + 10.0) / 2.0,
                    (8.0 + 11.0) / 2.0,
                    (9.0 + 12.0) / 2.0,
                ],
                EmbeddingBagMode::Max => vec![4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
            };
            assert_eq!(got, expected, "mode={mode:?}");
        }
    }

    /// rank 1 + offsets（`include_last_offset` あり）が rank 2 と一致
    /// すること。
    #[test]
    fn forward_with_offsets_matches_rank2_equivalent() {
        let w = Tensor::<f32>::new((0..12).map(|v| v as f32).collect(), &[4, 3]).unwrap();
        let ids2d = Tensor::<i32>::new(vec![0, 1, 2, 3], &[2, 2]).unwrap();
        let ids1d = Tensor::<i32>::new(vec![0, 1, 2, 3], &[4]).unwrap();

        let t1 = tape();
        let bag1 = EmbeddingBag::from_parameters(w.clone(), EmbeddingBagMode::Sum, None).unwrap();
        let out_rank2 = bag1.bind(&t1).forward(&ids2d).unwrap().to_tensor();

        let t2 = tape();
        let bag2 = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).unwrap();
        let out_offsets = bag2
            .bind(&t2)
            .forward_with_offsets(&ids1d, &[0, 2, 4], true)
            .unwrap()
            .to_tensor();

        assert_eq!(dense_vec(&out_rank2), dense_vec(&out_offsets));
    }

    #[test]
    fn forward_with_offsets_without_include_last_offset() {
        let w = Tensor::<f32>::new((0..8).map(|v| v as f32).collect(), &[4, 2]).unwrap();
        let ids = Tensor::<i32>::new(vec![0, 1, 2, 3], &[4]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).unwrap();
        // offsets の長さ = bag 数（2）。最後の bag は暗黙に N=4 まで。
        let out = bag
            .bind(&t)
            .forward_with_offsets(&ids, &[0, 2], false)
            .unwrap()
            .to_tensor();
        // bag0 = rows[0,1] = [0,1]+[2,3] = [2,4]、bag1 = rows[2,3] = [4,5]+[6,7] = [10,12]
        assert_eq!(dense_vec(&out), vec![2.0, 4.0, 10.0, 12.0]);
    }

    #[test]
    fn padding_idx_excluded_from_sum_and_mean() {
        let w = Tensor::<f32>::new(vec![1.0, 1.0, 2.0, 2.0, 9.0, 9.0], &[3, 2]).unwrap();
        // row 0 は padding_idx（本来 0 初期化だが from_parameters は上書きしないため
        // 明示的に [1,1] を入れて「除外されていること」を検出できるようにする）。
        let ids = Tensor::<i32>::new(vec![0, 1], &[1, 2]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Mean, Some(0)).unwrap();
        let out = bag.bind(&t).forward(&ids).unwrap().to_tensor();
        // padding_idx=0 が除外されるため、bag は row 1 [2,2] のみの平均 = [2,2]。
        assert_eq!(dense_vec(&out), vec![2.0, 2.0]);
    }

    #[test]
    fn empty_bag_is_zero() {
        let w = Tensor::<f32>::new(vec![1.0, 1.0, 2.0, 2.0], &[2, 2]).unwrap();
        let ids = Tensor::<i32>::new(Vec::<i32>::new(), &[0]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).unwrap();
        // offsets=[0, 0]（include_last_offset）は「開始 0・終了 0」の
        // bag 1 個（要素 0 個）を表す。
        let out = bag
            .bind(&t)
            .forward_with_offsets(&ids, &[0, 0], true)
            .unwrap()
            .to_tensor();
        assert_eq!(dense_vec(&out), vec![0.0, 0.0]);
    }

    /// PR #2281 是正（codex P1／cursor[bot] Medium「Empty bags detach bag
    /// output」）: バッチ内の全 bag が空（全 id が `padding_idx`）でも
    /// `weight` への微分可能な経路を維持し、`Tape::backward` が
    /// 「勾配追跡なし」で失敗しないこと。値は空 bag 契約どおり
    /// `0.0` のまま。
    #[test]
    fn all_bags_empty_still_backward_reaches_weight() {
        let w = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        // 2 bag とも id が全て padding_idx=0 のため、どちらも空 bag。
        let ids = Tensor::<i32>::new(vec![0, 0, 0, 0], &[2, 2]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, Some(0)).unwrap();
        let vars = bag.bind(&t);
        let out = vars.forward(&ids).unwrap();
        assert_eq!(dense_vec(&out.to_tensor()), vec![0.0, 0.0, 0.0, 0.0]);

        let loss = out.sum(None).unwrap();
        let grads = t.backward(&loss).unwrap();
        let weight_grad = grads
            .get(&vars.weight)
            .unwrap()
            .expect("全 bag が空でも weight への勾配経路は維持されるはず");
        // 値としては空 bag からの寄与は 0（`0.0` 倍したため）。
        assert_eq!(dense_vec(weight_grad), vec![0.0, 0.0, 0.0, 0.0]);
    }

    /// codex P1 指摘（PR #2281。`all_bags_empty_still_backward_reaches_
    /// weight` 是正後の再指摘）: `weight` の行 0 に `NaN`／`±inf` が
    /// 含まれていても、空 bag の出力は PyTorch `nn.EmbeddingBag` の
    /// 空 bag 契約どおり常に `0.0` であること（`weight_row0.mul(&zero)`
    /// による旧実装は `NaN * 0.0 == NaN`／`inf * 0.0 == NaN` で汚染
    /// されていた。`masked_fill` ベースの現行実装は `weight` の値を
    /// 出力へ一切伝播しない）。あわせて `weight` への勾配経路（値は
    /// `0.0`）も維持されること。
    #[test]
    fn empty_bag_output_is_zero_even_when_weight_row0_is_non_finite() {
        let w = Tensor::<f32>::new(vec![f32::NAN, f32::INFINITY, 3.0, 4.0], &[2, 2]).unwrap();
        // bag 0: 全 id が padding_idx=0 のため空。bag 1: 通常の非空 bag。
        let ids = Tensor::<i32>::new(vec![0, 0, 1, 1], &[2, 2]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, Some(0)).unwrap();
        let vars = bag.bind(&t);
        let out = vars.forward(&ids).unwrap();
        let got = dense_vec(&out.to_tensor());
        // bag 0（空。weight 行 0 は [NaN, inf] だが出力は常に [0, 0]）、
        // bag 1（row 1 の和 = [3,4]+[3,4] = [6,8]）。
        assert_eq!(&got[..2], &[0.0, 0.0]);
        assert_eq!(&got[2..], &[6.0, 8.0]);

        let loss = out.sum(None).unwrap();
        let grads = t.backward(&loss).unwrap();
        let weight_grad = grads
            .get(&vars.weight)
            .unwrap()
            .expect("weight への勾配経路は非有限値でも維持されるはず");
        let dw = dense_vec(weight_grad);
        // 空 bag（行 0）からの寄与は常に 0（NaN/inf 汚染なし）。
        // 行 1 は非空 bag に 2 回参照されるため勾配は 2 倍。
        assert_eq!(dw, vec![0.0, 0.0, 2.0, 2.0]);
    }

    /// PR #2281 是正（cursor[bot] Low「Empty batch rejected with
    /// padding」）: `padding_idx` 設定時または `L == 0` 時、`B == 0` の
    /// ランク 2 バッチ（bag 0 個）が拒否されず shape `[0, D]` を返す
    /// こと（no-padding fast path 以外でも成功する）。
    #[test]
    fn forward_rank2_empty_batch_with_padding_idx_succeeds() {
        let w = Tensor::<f32>::new(vec![1.0, 1.0, 2.0, 2.0], &[2, 2]).unwrap();
        let ids = Tensor::<i32>::new(Vec::<i32>::new(), &[0, 2]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, Some(0)).unwrap();
        let out = bag.bind(&t).forward(&ids).unwrap().to_tensor();
        assert_eq!(out.shape(), &[0, 2]);
    }

    /// 同上。`L == 0`（padding_idx なし）でも `forward_with_offsets` が
    /// 経由され `B == 0` が拒否されないこと。
    #[test]
    fn forward_rank2_empty_batch_with_zero_length_succeeds() {
        let w = Tensor::<f32>::new(vec![1.0, 1.0, 2.0, 2.0], &[2, 2]).unwrap();
        let ids = Tensor::<i32>::new(Vec::<i32>::new(), &[0, 0]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).unwrap();
        let out = bag.bind(&t).forward(&ids).unwrap().to_tensor();
        assert_eq!(out.shape(), &[0, 2]);
    }

    #[test]
    fn forward_with_offsets_rejects_empty_offsets() {
        let w = Tensor::<f32>::new(vec![1.0, 1.0], &[1, 2]).unwrap();
        let ids = Tensor::<i32>::new(vec![0], &[1]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).unwrap();
        let Err(err) = bag.bind(&t).forward_with_offsets(&ids, &[], false) else {
            panic!("空 offsets は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn forward_with_offsets_rejects_non_zero_start() {
        let w = Tensor::<f32>::new(vec![1.0, 1.0], &[1, 2]).unwrap();
        let ids = Tensor::<i32>::new(vec![0], &[1]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).unwrap();
        let Err(err) = bag.bind(&t).forward_with_offsets(&ids, &[1], false) else {
            panic!("offsets[0] != 0 は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn forward_with_offsets_rejects_non_monotonic() {
        let w = Tensor::<f32>::new(vec![1.0, 1.0, 2.0, 2.0], &[2, 2]).unwrap();
        let ids = Tensor::<i32>::new(vec![0, 1], &[2]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).unwrap();
        let Err(err) = bag.bind(&t).forward_with_offsets(&ids, &[0, 2, 1], false) else {
            panic!("非単調 offsets は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn forward_with_offsets_rejects_offset_exceeding_n() {
        let w = Tensor::<f32>::new(vec![1.0, 1.0], &[1, 2]).unwrap();
        let ids = Tensor::<i32>::new(vec![0], &[1]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).unwrap();
        let Err(err) = bag.bind(&t).forward_with_offsets(&ids, &[0, 5], false) else {
            panic!("N を超える offset は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn forward_with_offsets_rejects_include_last_offset_mismatch() {
        let w = Tensor::<f32>::new(vec![1.0, 1.0], &[1, 2]).unwrap();
        let ids = Tensor::<i32>::new(vec![0], &[1]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).unwrap();
        let Err(err) = bag.bind(&t).forward_with_offsets(&ids, &[0, 0], true) else {
            panic!("include_last_offset の末尾不一致は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn forward_rejects_out_of_range_id_without_leaving_orphan_nodes() {
        let w = Tensor::<f32>::new(vec![1.0, 1.0, 2.0, 2.0], &[2, 2]).unwrap();
        // bag0 は正常、bag1 に範囲外 id（2）を混入。
        let ids = Tensor::<i32>::new(vec![0, 1, 2, 0], &[2, 2]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).unwrap();
        let vars = bag.bind(&t);
        let node_count_before = t.nodes.borrow().len();
        let Err(err) = vars.forward(&ids) else {
            panic!("範囲外 id は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        // `weight` の bind（既に完了済み）以降、`forward` 自体は何も
        // ノードを積まずに失敗している（モジュール doc「id 範囲検査を
        // `Var::embedding` へ一本化する設計」節）。
        assert_eq!(t.nodes.borrow().len(), node_count_before);
    }

    #[test]
    fn forward_rejects_rank_mismatch() {
        let w = Tensor::<f32>::new(vec![1.0, 1.0], &[1, 2]).unwrap();
        let ids = Tensor::<i32>::new(vec![0], &[1]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).unwrap();
        let Err(err) = bag.bind(&t).forward(&ids) else {
            panic!("rank != 2 は Err を返すはず")
        };
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::RankMismatch { .. })
        ));
    }

    #[test]
    fn forward_from_var_matches_forward_with_raw_ids() {
        let w = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let ids = Tensor::<i32>::new(vec![0, 1, 2, 0], &[2, 2]).unwrap();
        let t = tape();
        let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).unwrap();
        let expected = bag.bind(&t).forward(&ids).unwrap().to_tensor();

        let ids_f32 = Tensor::new(vec![0.0_f32, 1.0, 2.0, 0.0], &[2, 2]).unwrap();
        let input_var = t.var(&ids_f32);
        let actual = bag
            .bind(&t)
            .forward_from_var(&input_var)
            .unwrap()
            .to_tensor();
        assert_eq!(dense_vec(&expected), dense_vec(&actual));
    }

    #[test]
    fn set_parameter_replaces_weight_preserving_shape() {
        let mut bag = EmbeddingBag::from_parameters(
            Tensor::zeros(&[3, 2]).unwrap(),
            EmbeddingBagMode::Sum,
            None,
        )
        .unwrap();
        let new_weight = Tensor::new(vec![9.0_f32; 6], &[3, 2]).unwrap();
        bag.set_parameter("weight", new_weight.clone()).unwrap();
        assert_eq!(
            bag.weight().host_slice().into_owned(),
            new_weight.host_slice().into_owned()
        );
    }

    #[test]
    fn set_parameter_rejects_shape_mismatch() {
        let mut bag = EmbeddingBag::from_parameters(
            Tensor::zeros(&[3, 2]).unwrap(),
            EmbeddingBagMode::Sum,
            None,
        )
        .unwrap();
        let wrong_shape = Tensor::new(vec![1.0_f32; 4], &[2, 2]).unwrap();
        let Err(err) = bag.set_parameter("weight", wrong_shape) else {
            panic!("shape 不一致は Err を返すはず")
        };
        assert!(matches!(err, AutodiffError::Shape(_)));
    }
}
