//! scaled dot product attention（PyTorch `torch.nn.functional.
//! scaled_dot_product_attention` 相当）を既存の `Var` 演算の合成として
//! 実装する（イシュー #1639。親 #1605「MultiheadAttention」の sub-issue
//! (a)。`docs/spec/04-requirements.md` REQ-9 2026-09-12 追記 Tier 1・
//! `docs/compat-api-scope.md` §1.2）。
//!
//! **設計方針（新規 `Op`／`BackendOps` メソッド／カーネルを追加しない。
//! [`crate::einsum`] と同型）**: `QK^T`（バッチ行列積）→ scale →
//! （任意の）causal／padding mask → softmax → `V` とのバッチ行列積、
//! という計算を [`Var::matmul`]（rank≥2 受理・NumPy 互換バッチ
//! ブロードキャスト。イシュー #1715）・[`Var::transpose`]（zero-copy
//! view）・[`Var::mul`]（scale。`nn::optim::amp::scale_loss` と同じ
//! 「スカラー Leaf を 1 つ登録して掛ける」先例）・[`Var::masked_fill`]
//! （イシュー #1637）・[`Var::softmax`]（イシュー #1594）の合成として
//! 組み立てる。VJP は各構成演算の VJP 合成として自動的に成立し
//! （`grad.rs` へ専用 VJP を追加しない）、バックエンド間数値一致も
//! これら既存演算の parity 契約（CUDA TF32 opt-in の追従を含む）に
//! そのまま帰着する。「対応する Op／`BackendOps` メソッドを追加する」
//! という受入要件は、分解先の演算がすでに CPU／CUDA／Metal 全てに
//! 実装済みであるため合成によって自動的に充足される。
//!
//! **内部 Leaf（scale 用）について**: `scale` を掛けるために内部で
//! `Tape::var(&Tensor::scalar(scale))` を呼び 1 個の Leaf ノードを
//! 登録する（`amp::scale_loss` と同一パターン）。`Tape::leaf_count`／
//! `Tape::leaf` の列挙対象になるが、optimizer・`SequentialVars::
//! trainable_vars` は明示的な `Var` 集合を扱い全 Leaf を走査しないため
//! 学習ループへの影響はない。
//!
//! **対象外（本 issue のスコープ外。`out-of-scope-tracking.md`）**:
//! `dropout_p`（SDPA への結線は対象外。`Var::dropout` 自体は #1603 で
//! 実装済み）・`enable_gqa`・attention
//! weights の返却・f16／bf16 経路（#1626）・CUDA／Metal 専用の融合
//! attention カーネル（`docs/kernel-fusion.md` の「複合ワークロードで
//! 融合を性能目標の前提にしない」方針と整合）・`MultiheadAttention`
//! Module 自体（in/out projection・head 分割。#1640）。

use fandhe_ai_tensor_core::{ShapeError, Tensor, broadcast_shape};

use crate::error::AutodiffError;
use crate::var::Var;

/// `scale`（明示指定・既定値のいずれも）の共通検証。`amp::
/// validate_scale` と同じ「非有限・非正は 0 除算・非有限伝播を未然に
/// 防ぐため演算グラフへ記録する前に弾く」fail-closed 方針。
fn validate_scale(scale: f32) -> Result<(), AutodiffError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "scaled_dot_product_attention: scale must be finite and > 0.0, got {scale}"
        )));
    }
    Ok(())
}

/// causal（top-left aligned。`j <= i` のみ attend を許可）ブロックマスク
/// `[l, s]` を構築する（PyTorch `_scaled_dot_product_attention_math` 参照
/// 実装の `torch.ones(L, S, dtype=bool).tril(diagonal=0)` の否定と同一
/// 規約。`l != s` の非正方形状も対応）。`blocked[i][j] = j > i` であり、
/// 任意の `i >= 0` に対し `j = 0` は常に許可される（`0 > i` は `i >= 0`
/// では常に偽）ため、`l > 0 && s > 0` の下では全 masked 行は構造的に
/// 生じない。
pub(crate) fn causal_blocked_mask(l: usize, s: usize) -> Result<Tensor<bool>, ShapeError> {
    let mut data = Vec::with_capacity(l * s);
    for i in 0..l {
        for j in 0..s {
            data.push(j > i);
        }
    }
    // `data.len() == l * s` は上記ループから自明に成立するが、本番経路
    // （`is_causal=true` の公開 API から呼ばれる）で panic させないため
    // `Tensor::new` の `Result` をそのまま呼び出し元へ伝播する
    // （AGENTS.md「本番経路の panic 禁止」・coding-rust の明示禁止事項）。
    Tensor::new(data, &[l, s])
}

/// `blocked`（`[..., l, s]` へ broadcast 済み・`.contiguous()` 済みの
/// bool テンソル）の最終軸（`s` 列）を 1 行ずつ走査し、全要素が `true`
/// （全 key が masked）の行がないか検査する。全 masked 行は
/// `exp(-inf - max) / sum(...)` がバックエンド依存の不定値（`0/0`）を
/// 生みうるため、演算グラフへ `masked_fill` を記録する前に拒否する
/// （§2.4。`ShapeError` ではなく `InvalidArgument`——値の組合せの問題
/// であり shape 自体は妥当なため）。
fn reject_fully_masked_rows(blocked: &Tensor<bool>) -> Result<(), AutodiffError> {
    let shape = blocked.shape();
    let rank = shape.len();
    if rank == 0 {
        return Ok(());
    }
    let s = shape[rank - 1];
    if s == 0 {
        // 列が 0 本の「行」に「全要素が masked」という主張は空虚
        // （vacuous truth で誤って fail させない）。S == 0 自体は
        // 呼び出し元が `l > 0 && s > 0` の場合のみ本関数を呼ぶため
        // 通常到達しないが、防御的に安全側で早期 return する。
        return Ok(());
    }
    let contiguous = blocked.contiguous();
    let data = contiguous.as_slice().unwrap_or_default();
    if data.chunks_exact(s).any(|row| row.iter().all(|&b| b)) {
        return Err(AutodiffError::InvalidArgument(
            "scaled_dot_product_attention: a row is fully masked by attn_mask/is_causal \
             (no attendable key), which would make softmax produce backend-dependent \
             indeterminate values"
                .to_string(),
        ));
    }
    Ok(())
}

/// `attn_mask`（`true` = attend。PyTorch bool mask 規約）を `scores`
/// の shape（`[..., l, s]`）へ broadcast したうえで否定し、
/// [`Var::masked_fill`] へ渡す「block」テンソル（`true` = fill 対象）を
/// 作る。`Tensor<bool>` に要素ごとの否定演算がないため、`as_slice` で
/// 読み出してから組み立てる（`Var::where_cond`／`masked_fill` の
/// broadcast → `as_slice` パターンと同型）。
fn negate_broadcast_mask(
    mask: &Tensor<bool>,
    scores_shape: &[usize],
) -> Result<Tensor<bool>, AutodiffError> {
    let allowed = mask
        .broadcast_to(scores_shape)
        .map_err(AutodiffError::Shape)?
        .contiguous();
    let allowed_data: Vec<bool> = allowed.as_slice().map(|s| s.to_vec()).unwrap_or_default();
    let blocked_data: Vec<bool> = allowed_data.iter().map(|&a| !a).collect();
    Tensor::new(blocked_data, scores_shape).map_err(AutodiffError::Shape)
}

/// `scaled_dot_product_attention` 本体（[`Var::scaled_dot_product_attention`]
/// が薄く委譲する。モジュール doc §設計方針参照）。
pub(crate) fn scaled_dot_product_attention<'t>(
    query: &Var<'t>,
    key: &Var<'t>,
    value: &Var<'t>,
    attn_mask: Option<&Tensor<bool>>,
    is_causal: bool,
    scale: Option<f32>,
) -> Result<Var<'t>, AutodiffError> {
    query.check_same_tape(key)?;
    query.check_same_tape(value)?;

    if attn_mask.is_some() && is_causal {
        return Err(AutodiffError::InvalidArgument(
            "scaled_dot_product_attention: attn_mask and is_causal are mutually exclusive"
                .to_string(),
        ));
    }

    let q_shape = query.shape();
    if q_shape.len() < 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: q_shape.len(),
        }));
    }
    let e = q_shape[q_shape.len() - 1];

    let scale_eff = match scale {
        Some(s) => s,
        None => {
            if e == 0 {
                return Err(AutodiffError::InvalidArgument(
                    "scaled_dot_product_attention: scale must be given explicitly when \
                     query's last dim (E) == 0 (1/sqrt(0) is undefined)"
                        .to_string(),
                ));
            }
            1.0 / (e as f32).sqrt()
        }
    };
    validate_scale(scale_eff)?;

    // scale を q 側（scores より要素数が少ない）へ先に掛ける（丸め順序の
    // 選択であり REQ-2 統一複合判定の範囲内。`Op::Mul` は lazy ノードで
    // 後続 `matmul` の入力実体化境界で実行される）。
    let scale_var = query.tape().var(&Tensor::scalar(scale_eff));
    let q_scaled = query.mul(&scale_var)?;

    let k_rank = key.shape().len();
    if k_rank < 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: k_rank,
        }));
    }
    // zero-copy view（`Var::transpose`）。`matmul` の rhs としてそのまま
    // 渡せる（`normalize_batched_operand` が必要なら contiguous 化する）。
    let k_t = key.transpose(k_rank - 2, k_rank - 1)?;

    // `[..., L, E] @ [..., E, S] -> [..., L, S]`。E 不一致・バッチ
    // broadcast 不能は `Var::matmul`（`matmul_out_shape`）が検査済み。
    let scores = q_scaled.matmul(&k_t)?;
    let scores_shape = scores.shape();
    let rank = scores_shape.len();
    let l = scores_shape[rank - 2];
    let s = scores_shape[rank - 1];

    // L == 0 または S == 0 は空テンソル契約（§2.4「0 サイズ」）。
    // 「全 masked 行」の値検査は行自体が存在しないため意味をなさず
    // 省略するが、`attn_mask` の broadcast 形状検証（不能形状の
    // `ShapeError` 化）は 0 サイズでも常に実施する（PR #1845
    // codex-review 指摘: L==0/S==0 で検証を丸ごとスキップすると
    // broadcast 不能な `attn_mask` を渡しても受理されてしまう）。
    let scores = if is_causal {
        if l > 0 && s > 0 {
            let blocked = causal_blocked_mask(l, s).map_err(AutodiffError::Shape)?;
            reject_fully_masked_rows(&blocked)?;
            // `masked_fill` 自身が `[l, s]` を `scores_shape` へ
            // broadcast する（`Var::masked_fill` doc 参照）。
            scores.masked_fill(&blocked, f32::NEG_INFINITY)?
        } else {
            // causal の block パターンは `l`／`s` のみから決定的に
            // 導出され外部形状を持たないため、0 サイズでは検証すべき
            // 追加形状が存在しない（そのまま softmax／後続 matmul の
            // 0 サイズ契約へ委ねる）。
            scores
        }
    } else if let Some(mask) = attn_mask {
        // mask の broadcast 先は scores 自身のバッチ形状（query／key
        // 由来）だけでなく、最終的に `weights.matmul(value)` で
        // 合成される value 側のバッチ形状とも共通化する必要がある。
        // mask の broadcast 先を scores 自身のバッチ形状のみに限定
        // すると、value 側のみバッチが大きいケース（例: query/key
        // バッチ=1・value バッチ=5・mask がそのバッチ 5 を明示する
        // 場合）で本来妥当な mask を誤って拒否してしまう（PR #1845
        // codex-review 指摘）。value の rank<2 は後続の
        // `weights.matmul(value)` が検査する契約（§2.4）を崩さない
        // よう、value の rank が妥当な場合に限りバッチ共通化を試みる
        // （rank<2 の場合は scores 自身の形状のまま従来どおり進み、
        // 最終 matmul が適切な `ShapeError` を返す）。
        let value_shape = value.shape();
        let mask_target_shape: Vec<usize> = if value_shape.len() >= 2 {
            let value_batch_shape = &value_shape[..value_shape.len() - 2];
            let scores_batch_shape = &scores_shape[..rank - 2];
            let combined_batch_shape = broadcast_shape(scores_batch_shape, value_batch_shape)
                .map_err(AutodiffError::Shape)?;
            let mut full_shape = combined_batch_shape;
            full_shape.push(l);
            full_shape.push(s);
            full_shape
        } else {
            scores_shape.clone()
        };
        // scores 自身のバッチ形状が共通形状より小さい場合のみ実体化を
        // 伴う broadcast（`Var::broadcast_to`）を挟む。等しい場合
        // （典型的には value 側のバッチが scores 以下の通常経路）は
        // 恒等 view のため余分なノードを増やさない。
        let scores = if mask_target_shape != scores_shape {
            scores.broadcast_to(&mask_target_shape)?
        } else {
            scores
        };
        let blocked = negate_broadcast_mask(mask, &mask_target_shape)?;
        if l > 0 && s > 0 {
            reject_fully_masked_rows(&blocked)?;
        }
        scores.masked_fill(&blocked, f32::NEG_INFINITY)?
    } else {
        scores
    };

    // 最終軸（S）方向の softmax（CPU／CUDA／Metal いずれも行カーネルへ
    // 到達する契約。イシュー #1594）。`scores` は `attn_mask` 分岐で
    // value 側バッチを含む形状へ broadcast されている場合があるため、
    // 軸番号は元の `rank` ではなく broadcast 後の実際の rank から
    // 算出する（`rank - 1` を使うと value 側バッチ拡張時に軸番号が
    // ずれる。PR #1845 codex-review 指摘の是正に伴う整合修正）。
    let softmax_axis = scores.shape().len() - 1;
    let weights = scores.softmax(softmax_axis)?;

    // `[..., L, S] @ [..., S, Ev] -> [..., L, Ev]`。S 不一致
    // （`key[-2] != value[-2]`）はここで `Var::matmul` が検査する
    // （§2.4「既存検査への委譲」）。
    weights.matmul(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_blocked_mask_square_is_strict_upper_triangle() {
        let m = causal_blocked_mask(3, 3).unwrap();
        assert_eq!(m.shape(), &[3, 3]);
        let data = m.as_slice().unwrap();
        // 行 i・列 j: blocked = j > i。
        let expected = [
            false, true, true, // i=0: j=0 のみ許可
            false, false, true, // i=1: j=0,1 許可
            false, false, false, // i=2: 全許可
        ];
        assert_eq!(data, &expected);
    }

    #[test]
    fn causal_blocked_mask_rectangular_l_gt_s_last_rows_fully_open() {
        // L=4, S=2: i>=1 の行はすべて j<=i を満たす（j は最大でも 1）。
        let m = causal_blocked_mask(4, 2).unwrap();
        let data = m.as_slice().unwrap();
        assert_eq!(
            data,
            &[false, true, false, false, false, false, false, false]
        );
    }

    #[test]
    fn causal_blocked_mask_rectangular_l_lt_s_later_columns_blocked() {
        // L=2, S=4: i=0 は j=0 のみ許可、i=1 は j=0,1 のみ許可。
        let m = causal_blocked_mask(2, 4).unwrap();
        let data = m.as_slice().unwrap();
        assert_eq!(data, &[false, true, true, true, false, false, true, true]);
    }

    #[test]
    fn causal_blocked_mask_never_produces_fully_masked_row() {
        for l in 1..6 {
            for s in 1..6 {
                let m = causal_blocked_mask(l, s).unwrap();
                assert!(reject_fully_masked_rows(&m).is_ok(), "l={l} s={s}");
            }
        }
    }

    #[test]
    fn reject_fully_masked_rows_detects_all_true_row() {
        // 行 0 = [false, true]（一部 masked）・行 1 = [false, true]（同）
        // はいずれも「全 masked」ではない。
        let m = Tensor::new(vec![false, true, false, true], &[2, 2]).unwrap();
        assert!(reject_fully_masked_rows(&m).is_ok());
        // 行 0 は [true, true]（全 masked）を含む。
        let m2 = Tensor::new(vec![true, true, false, true], &[2, 2]).unwrap();
        assert!(reject_fully_masked_rows(&m2).is_err());
    }

    #[test]
    fn reject_fully_masked_rows_zero_cols_is_vacuously_ok() {
        let m = Tensor::new(Vec::<bool>::new(), &[3, 0]).unwrap();
        assert!(reject_fully_masked_rows(&m).is_ok());
    }

    #[test]
    fn validate_scale_rejects_non_finite_and_non_positive() {
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(validate_scale(bad).is_err(), "scale={bad}");
        }
        assert!(validate_scale(0.125).is_ok());
    }

    #[test]
    fn negate_broadcast_mask_broadcasts_and_negates() {
        // mask: [1, 2] （`true` = attend）を [2, 2] へ broadcast してから
        // 否定する。
        let mask = Tensor::new(vec![true, false], &[1, 2]).unwrap();
        let blocked = negate_broadcast_mask(&mask, &[2, 2]).unwrap();
        assert_eq!(blocked.shape(), &[2, 2]);
        assert_eq!(blocked.as_slice().unwrap(), &[false, true, false, true]);
    }
}
