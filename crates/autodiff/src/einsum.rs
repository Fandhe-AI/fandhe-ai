//! einsum 記法（PyTorch `torch.einsum`／TensorFlow `tf.einsum` 相当）の
//! 解釈器と、既存の `Var` 演算（[`crate::var::Var::matmul`]／`sum`／
//! `permute`／`reshape`／`mul`／`contiguous`）への分解ドライバ
//! （イシュー #1620・`docs/compat-feature-gap.md` §2.6）。
//!
//! **設計方針（新規カーネルを追加しない）**: 汎用縮約記法を「既存 GEMM
//! （`Var::matmul`）」「既存縮約（`Var::sum`）」「既存 view（`Var::
//! permute`／`reshape`）」「既存 broadcast 乗算（`Var::mul`）」への
//! 分解として実装する。これにより VJP は分解先の各演算の VJP 合成
//! として自動的に成立し（`einsum` 専用の VJP を `grad.rs` に追加しない）、
//! バックエンド間数値一致も既存の parity 契約（`Var::matmul` と同一の
//! FMA 契約・CUDA TF32 opt-in 挙動）に帰着する。`BackendOps` への
//! メソッド追加も行わない——分解先の演算がすでに 3 バックエンドで
//! 実装済みのため「該当バックエンドすべてに実装」は分解によって
//! 自動的に充足される。
//!
//! **受理範囲（v1・安全側）**:
//! - 添字は ASCII 英字のみ（空白は無視）。
//! - ellipsis（`...`）は未対応（`AutodiffError::InvalidArgument`）。
//! - 同一オペランド内の添字重複（対角／trace。例 `"ii->i"`）は未対応。
//! - 出力添字の重複は未対応。
//! - `->` 省略時は NumPy `einsum` 既定と同じ「入力に 1 回だけ現れる
//!   添字を ASCII 昇順に並べたもの」を出力とみなす。
//! - オペランドは 1〜2 個限定。3 個以上は本イシューのスコープ外として
//!   拒否する（N 項の左畳み込みは将来拡張。`out-of-scope-tracking.md`
//!   に従いユーザー承認後に別途対応する）。
//! - 2 項の縮約で、両オペランドと出力に共通する「batch 添字」
//!   （例 `"bij,bjk->bik"`）を伴うものは rank≥3 `matmul`（#1600）が
//!   未実装のため拒否する（`compute_binary_plan` が判定。ガード撤去
//!   だけでは対応できず、rank≥3 `matmul` 実装後に `einsum_matmul_path`
//!   を `[batch..., L, K]×[batch..., K, R]` 形状へ再設計する必要がある。
//!   `docs/compat-feature-gap.md` #1620 追補参照）。batch 添字を伴わない
//!   縮約（GEMM 相当の単一 contract 軸群・要素ごと乗算相当）はすべて
//!   対応する。
//!
//! **数値契約**: GEMM 経路（`contract` が非空・`batch` が空）は
//! `Var::matmul` をそのまま呼ぶため、CUDA TF32 opt-in の挙動を含めて
//! `matmul` と同一。追加の丸め経路は作らない
//! （`.claude/rules/coding-rust.md` FMA 契約統一）。
//!
//! **検証と Var 操作の分離**: `einsum_binary` は `compute_binary_plan`
//! （純関数。添字集合のみから分類・拒否判定を行い `Var` を一切
//! 操作しない）が全検証（presum 計画・batch/contract/left/right 分類・
//! 内部整合性検査・batch∧contract 非空の拒否）を終えてから、初めて
//! `apply_presum`（`Var::sum` を実行）を呼ぶ。旧実装は presum の
//! `Var::sum` を tape へ push した**後**に batch 拒否を行っており、
//! 拒否時に迷子ノードが残る欠陥があった（分類は添字集合のみで決まり
//! presum の実行結果には依存しないため、順序を入れ替えられる）。

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::error::AutodiffError;
use crate::var::Var;
use fandhe_ai_tensor_core::ShapeError;

/// spec 文字列の長さ上限（バイト。REQ-8 趣旨の境界検査・A03 対策）。
/// 外部入力になりうる einsum 記法に対し、パース前に無制限な文字列を
/// 受け付けない安全側の上限。
const MAX_SPEC_LEN: usize = 256;

/// パース済み einsum 記法（`crate::einsum::parse` の出力）。
#[derive(Debug, Clone, PartialEq, Eq)]
struct EinsumSpec {
    inputs: Vec<Vec<char>>,
    output: Vec<char>,
}

/// `spec` 文字列を [`EinsumSpec`] へ解釈する（純関数。`Var` に依存しない）。
///
/// 受理範囲はモジュール doc「受理範囲」節を参照。拒否条件はすべて
/// `AutodiffError::InvalidArgument`（型付きエラー。本番経路で panic
/// させない）で返す。
fn parse(spec: &str) -> Result<EinsumSpec, AutodiffError> {
    if spec.len() > MAX_SPEC_LEN {
        return Err(AutodiffError::InvalidArgument(format!(
            "einsum: spec の長さ（{} バイト）が上限（{MAX_SPEC_LEN} バイト）を超えている",
            spec.len()
        )));
    }
    if spec.contains("...") {
        return Err(AutodiffError::InvalidArgument(
            "einsum: ellipsis（\"...\"）記法は未対応（イシュー #1620 のスコープ外）".to_string(),
        ));
    }

    let (lhs, rhs) = match spec.split_once("->") {
        Some((l, r)) => (l, Some(r)),
        None => (spec, None),
    };

    let mut inputs: Vec<Vec<char>> = Vec::new();
    for part in lhs.split(',') {
        let chars: Vec<char> = part.chars().filter(|c| !c.is_whitespace()).collect();
        if chars.is_empty() {
            return Err(AutodiffError::InvalidArgument(
                "einsum: 空のオペランド添字グループが指定された".to_string(),
            ));
        }
        for &c in &chars {
            if !c.is_ascii_alphabetic() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "einsum: ASCII 英字以外の添字文字 '{c}' は未対応"
                )));
            }
        }
        let mut seen: HashSet<char> = HashSet::new();
        for &c in &chars {
            if !seen.insert(c) {
                return Err(AutodiffError::InvalidArgument(format!(
                    "einsum: 同一オペランド内の添字重複（対角／trace）'{c}' は未対応"
                )));
            }
        }
        inputs.push(chars);
    }
    if inputs.is_empty() {
        return Err(AutodiffError::InvalidArgument(
            "einsum: オペランドが指定されていない".to_string(),
        ));
    }

    let all_input_chars: HashSet<char> = inputs.iter().flatten().copied().collect();

    let output: Vec<char> = match rhs {
        Some(r) => {
            let chars: Vec<char> = r.chars().filter(|c| !c.is_whitespace()).collect();
            for &c in &chars {
                if !c.is_ascii_alphabetic() {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "einsum: ASCII 英字以外の出力添字文字 '{c}' は未対応"
                    )));
                }
            }
            let mut seen: HashSet<char> = HashSet::new();
            for &c in &chars {
                if !seen.insert(c) {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "einsum: 出力添字の重複 '{c}' は未対応"
                    )));
                }
            }
            for &c in &chars {
                if !all_input_chars.contains(&c) {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "einsum: 出力添字 '{c}' がどのオペランドの添字にも現れない"
                    )));
                }
            }
            chars
        }
        None => {
            // 省略時は NumPy `einsum` 既定と同じ「入力に 1 回だけ現れる
            // 添字を ASCII 昇順」（`BTreeMap` で昇順反復を保証する）。
            let mut counts: BTreeMap<char, usize> = BTreeMap::new();
            for &c in &all_input_chars {
                counts.insert(c, 0);
            }
            for c in inputs.iter().flatten().copied() {
                *counts.entry(c).or_insert(0) += 1;
            }
            counts
                .into_iter()
                .filter(|&(_, n)| n == 1)
                .map(|(c, _)| c)
                .collect()
        }
    };

    Ok(EinsumSpec { inputs, output })
}

/// [`Var::einsum`] の実装本体（`pub(crate)`。`Var::einsum` から呼ばれる
/// 唯一の呼び出し元）。クロステープ検査 → パース → オペランド数・
/// rank 一致検査 → 次元サイズ一致検査 → 分解ドライバの順で処理する
/// （tape へノードを push するのは分解ドライバ内部の検証完了後のみ。
/// モジュール doc「検証と Var 操作の分離」参照）。
pub(crate) fn einsum<'t>(spec: &str, operands: &[&Var<'t>]) -> Result<Var<'t>, AutodiffError> {
    if operands.is_empty() {
        return Err(AutodiffError::InvalidArgument(
            "einsum: オペランドが指定されていない".to_string(),
        ));
    }
    for pair in operands.windows(2) {
        pair[0].check_same_tape(pair[1])?;
    }

    let parsed = parse(spec)?;
    if parsed.inputs.len() != operands.len() {
        return Err(AutodiffError::InvalidArgument(format!(
            "einsum: spec のオペランド数（{}）と渡された Var の数（{}）が一致しない",
            parsed.inputs.len(),
            operands.len()
        )));
    }
    for (labels, operand) in parsed.inputs.iter().zip(operands.iter()) {
        let rank = operand.shape().len();
        if rank != labels.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "einsum: オペランドの rank（{rank}）と添字数（{}）が一致しない",
                labels.len()
            )));
        }
    }

    // 添字 → 次元サイズの対応表を構築し、複数オペランド間で共有される
    // 添字の次元サイズが一致することを検査する（forward 実行前の
    // fail-closed な shape 検査。`.claude/rules/security.md` A03）。
    let mut dim_of: HashMap<char, usize> = HashMap::new();
    for (labels, operand) in parsed.inputs.iter().zip(operands.iter()) {
        let shape = operand.shape();
        for (&c, &size) in labels.iter().zip(shape.iter()) {
            match dim_of.get(&c) {
                Some(&existing) if existing != size => {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "einsum: 添字 '{c}' の次元サイズが不一致（{existing} と {size}）"
                    )));
                }
                _ => {
                    dim_of.insert(c, size);
                }
            }
        }
    }

    match operands.len() {
        1 => einsum_unary(*operands[0], &parsed.inputs[0], &parsed.output),
        2 => einsum_binary(
            *operands[0],
            *operands[1],
            &parsed.inputs[0],
            &parsed.inputs[1],
            &parsed.output,
            &dim_of,
        ),
        n => Err(AutodiffError::InvalidArgument(format!(
            "einsum: オペランド数 {n} 個（3 個以上）は本イシュー（#1620）のスコープ外のため未対応"
        ))),
    }
}

/// `current` の並びを `target` の並びへ揃えるための `Var::permute` 引数
/// （純関数・`Var` を一切操作しない）。`current`／`target` は同一の
/// 添字集合の順列であることを呼び出し元が保証する契約
/// （不一致は内部エラーとして拒否する——到達すれば本モジュール自身の
/// ロジックバグ）。
fn perm_indices(current: &[char], target: &[char]) -> Result<Vec<usize>, AutodiffError> {
    if current.len() != target.len() {
        return Err(AutodiffError::InvalidArgument(
            "einsum: 内部エラー（permute の添字数不一致。分解ロジックの契約違反）".to_string(),
        ));
    }
    let mut perm = Vec::with_capacity(target.len());
    for &t in target {
        let idx = current.iter().position(|&c| c == t).ok_or_else(|| {
            AutodiffError::InvalidArgument(
                "einsum: 内部エラー（permute で添字が見つからない。分解ロジックの契約違反）"
                    .to_string(),
            )
        })?;
        perm.push(idx);
    }
    Ok(perm)
}

/// `v`（添字順 `current`）を `target` の順へ並べ替える（`Var::permute`
/// 1 回に帰着。イシュー #1679 で追加された `Var::permute` を使う——
/// 旧実装の `transpose` チェーンから置き換え）。恒等順序（`current ==
/// target`）の場合は `Var::permute` を呼ばずノードを積まない
/// （`"ij,jk->ik"` が `MatMul` ノード 1 個だけを記録する契約。
/// モジュール doc「検証と Var 操作の分離」節の前段に相当する最適化）。
fn apply_permute<'t>(
    v: Var<'t>,
    current: &[char],
    target: &[char],
) -> Result<Var<'t>, AutodiffError> {
    if current == target {
        return Ok(v);
    }
    let perm = perm_indices(current, target)?;
    v.permute(&perm)
}

/// `v` の shape が `shape` と既に一致する場合は `Var::reshape` を呼ばず
/// ノードを積まない（`apply_permute` の恒等スキップと同じ最適化。
/// `"ij,jk->ik"` の GEMM 経路で入力側の reshape が恒等になるケースを
/// 吸収する）。
fn apply_reshape<'t>(v: Var<'t>, shape: &[usize]) -> Result<Var<'t>, AutodiffError> {
    if v.shape() == shape {
        return Ok(v);
    }
    v.reshape(shape)
}

/// 出力（`keep`）に現れない添字の軸位置を、位置の降順（`Var::sum` が
/// 縮約後に軸をひとつ詰めるため、降順に適用すれば未処理の軸の位置が
/// ずれない）で算出する（純関数・`Var::sum` は呼ばない。`apply_presum`
/// が本関数の出力をそのまま消費する）。
fn plan_presum(labels: &[char], keep: &HashSet<char>) -> (Vec<usize>, Vec<char>) {
    let mut cur_labels = labels.to_vec();
    let positions: Vec<usize> = {
        let mut p: Vec<usize> = cur_labels
            .iter()
            .enumerate()
            .filter(|(_, c)| !keep.contains(c))
            .map(|(i, _)| i)
            .collect();
        p.sort_unstable_by(|a, b| b.cmp(a));
        p
    };
    for &pos in &positions {
        cur_labels.remove(pos);
    }
    (positions, cur_labels)
}

/// [`plan_presum`] が算出した軸位置（降順）を実際に
/// `Var::sum(Some(pos))` の連鎖として適用する（副作用あり。tape へ
/// ノードを push する）。呼び出し元（`einsum_unary`／`einsum_binary`）
/// は、この関数を呼ぶ**前**にすべての検証（分類・拒否判定）を完了
/// させている契約（モジュール doc「検証と Var 操作の分離」参照）。
fn apply_presum<'t>(v: Var<'t>, positions: &[usize]) -> Result<Var<'t>, AutodiffError> {
    let mut cur = v;
    for &pos in positions {
        cur = cur.sum(Some(pos))?;
    }
    Ok(cur)
}

/// 単項 einsum（オペランド 1 個）。出力に現れない添字を `sum` で縮約
/// してから、残った添字を出力順へ `permute` で並べ替える。
fn einsum_unary<'t>(
    a: Var<'t>,
    in_labels: &[char],
    out_labels: &[char],
) -> Result<Var<'t>, AutodiffError> {
    let keep: HashSet<char> = out_labels.iter().copied().collect();
    let (positions, cur_labels) = plan_presum(in_labels, &keep);
    let v = apply_presum(a, &positions)?;
    apply_permute(v, &cur_labels, out_labels)
}

/// `compute_binary_plan` が純粋に算出する 2 項 einsum の分類結果
/// （添字集合のみから決まり `Var` を一切操作しない）。`einsum_binary`
/// はこの構造体が（拒否判定を含め）確定してから初めて `Var` 操作
/// （`apply_presum`／`Var::mul`／`Var::matmul` 等）を開始する。
struct BinaryPlan {
    /// a 側 presum の対象軸位置（降順。`apply_presum` へそのまま渡す）。
    a_presum: Vec<usize>,
    /// a 側 presum 後の添字順。
    a_labels: Vec<char>,
    b_presum: Vec<usize>,
    b_labels: Vec<char>,
    /// 両オペランドと出力すべてに現れる（要素ごとに独立な軸）。
    batch: Vec<char>,
    /// 両オペランドに現れるが出力には現れない（縮約対象）。
    contract: Vec<char>,
    /// a' のみに現れる（presum 済みのため出力に現れる保証あり）。
    left: Vec<char>,
    /// b' のみに現れる（同上）。
    right: Vec<char>,
}

/// 2 項 einsum の添字を batch／contract／left／right へ分類し、
/// presum 計画・内部整合性検査・batch∧contract 非空の拒否判定まで
/// すべて添字集合のみで（`Var` に一切触れず）完了させる。
///
/// presum（相手にも出力にも現れない添字を先に和で除去。PyTorch
/// `sumproduct_pair` と同じ最適化）で除去される添字は、定義上
/// batch／contract（＝両オペランドの共有添字）には含まれえない
/// （共有添字は必ず `keep_a`／`keep_b` に含まれ presum で除去されない）
/// ため、batch／contract の分類自体は presum の実行結果に依存しない。
/// この性質により、実際に `Var::sum` を呼ぶ（`apply_presum`）前に
/// 全検証を完了できる。
fn compute_binary_plan(
    a_labels_in: &[char],
    b_labels_in: &[char],
    out_labels: &[char],
) -> Result<BinaryPlan, AutodiffError> {
    let set_a: HashSet<char> = a_labels_in.iter().copied().collect();
    let set_b: HashSet<char> = b_labels_in.iter().copied().collect();
    let set_out: HashSet<char> = out_labels.iter().copied().collect();

    let keep_a: HashSet<char> = set_b.union(&set_out).copied().collect();
    let keep_b: HashSet<char> = set_a.union(&set_out).copied().collect();
    let (a_presum, a_labels) = plan_presum(a_labels_in, &keep_a);
    let (b_presum, b_labels) = plan_presum(b_labels_in, &keep_b);

    let set_a2: HashSet<char> = a_labels.iter().copied().collect();
    let set_b2: HashSet<char> = b_labels.iter().copied().collect();

    let mut batch: Vec<char> = set_a2
        .intersection(&set_b2)
        .filter(|c| set_out.contains(c))
        .copied()
        .collect();
    let mut contract: Vec<char> = set_a2
        .intersection(&set_b2)
        .filter(|c| !set_out.contains(c))
        .copied()
        .collect();
    let mut left: Vec<char> = a_labels
        .iter()
        .filter(|c| !set_b2.contains(c))
        .copied()
        .collect();
    let mut right: Vec<char> = b_labels
        .iter()
        .filter(|c| !set_a2.contains(c))
        .copied()
        .collect();
    // 決定的な順序（ASCII 昇順）にしておく——分解の中間 shape・tape
    // 構造が run ごとに変わらないようにするため（`docs/perf/
    // train-step-phase-breakdown.md` 系の記録が前提とする決定性と同じ
    // 方針）。
    batch.sort_unstable();
    contract.sort_unstable();
    left.sort_unstable();
    right.sort_unstable();

    // 内部整合性の防御的検査: batch ∪ left ∪ right（集合として）が
    // 出力添字集合と一致すること。presum・分類ロジックが正しければ
    // 常に成立するはずだが、本番経路 panic 禁止方針のため、万一の
    // ロジック不備は誤った出力を静かに返さず型付きエラーで拒否する
    // （`.claude/rules/security.md` A08）。
    let reconstructed: HashSet<char> = batch
        .iter()
        .chain(left.iter())
        .chain(right.iter())
        .copied()
        .collect();
    if reconstructed != set_out {
        return Err(AutodiffError::InvalidArgument(
            "einsum: 内部エラー（batch/left/right の再構成が出力添字と一致しない。分解ロジックの契約違反）"
                .to_string(),
        ));
    }

    if !contract.is_empty() && !batch.is_empty() {
        return Err(AutodiffError::InvalidArgument(
            "einsum: batch 添字を伴う縮約は rank>=3 の matmul（イシュー #1600）が未実装のため非対応"
                .to_string(),
        ));
    }

    Ok(BinaryPlan {
        a_presum,
        a_labels,
        b_presum,
        b_labels,
        batch,
        contract,
        left,
        right,
    })
}

/// 二項 einsum（オペランド 2 個）。`compute_binary_plan` が全検証を
/// 終えたのちに `apply_presum`（`Var::sum` 実行）を呼び、`contract` の
/// 有無で `Var::mul`（broadcast）経路と `Var::matmul`（GEMM）経路を
/// 切り替える（モジュール doc「受理範囲」節参照）。
fn einsum_binary<'t>(
    a: Var<'t>,
    b: Var<'t>,
    a_labels: &[char],
    b_labels: &[char],
    out_labels: &[char],
    dim_of: &HashMap<char, usize>,
) -> Result<Var<'t>, AutodiffError> {
    let plan = compute_binary_plan(a_labels, b_labels, out_labels)?;

    let a = apply_presum(a, &plan.a_presum)?;
    let b = apply_presum(b, &plan.b_presum)?;

    if plan.contract.is_empty() {
        einsum_mul_path(EinsumMulPlan {
            a,
            b,
            a_labels: plan.a_labels,
            b_labels: plan.b_labels,
            batch: &plan.batch,
            left: &plan.left,
            right: &plan.right,
            out_labels,
            dim_of,
        })
    } else {
        einsum_matmul_path(EinsumMatmulPlan {
            a,
            b,
            a_labels: plan.a_labels,
            b_labels: plan.b_labels,
            contract: &plan.contract,
            left: &plan.left,
            right: &plan.right,
            out_labels,
            dim_of,
        })
    }
}

/// `dim_of` を引いて `labels` の要素数の積を `checked_mul` で算出する
/// （REQ-8 趣旨の境界検査・A03 対策。`Var::reshape` は最終形状の要素数
/// 一致を検査するが、その手前で `l`／`k`／`r` 自体をオーバーフローさせ
/// ないための自前チェック。`labels` は `dim_of` を構築した検査済みの
/// 添字集合の部分集合であることを呼び出し元が保証するため、キー欠落は
/// ロジックバグとして扱う——`Var::reshape`／`Var::flatten` と同じ
/// `checked_mul` 規律）。
fn checked_numel(labels: &[char], dim_of: &HashMap<char, usize>) -> Result<usize, AutodiffError> {
    labels
        .iter()
        .try_fold(1usize, |acc, c| acc.checked_mul(dim_of[c]))
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))
}

/// [`einsum_mul_path`] へ渡す引数群（`#[allow(clippy::too_many_
/// arguments)]` を避けるための構造体化。個々のフィールドの意味は
/// [`BinaryPlan`] と同一）。
struct EinsumMulPlan<'a, 't> {
    a: Var<'t>,
    b: Var<'t>,
    a_labels: Vec<char>,
    b_labels: Vec<char>,
    batch: &'a [char],
    left: &'a [char],
    right: &'a [char],
    out_labels: &'a [char],
    dim_of: &'a HashMap<char, usize>,
}

/// `contract` が空の二項 einsum を `Var::mul`（NumPy 互換 broadcast）で
/// 計算する。両オペランドを `[batch..., left..., 1×|right|]`／
/// `[batch..., 1×|left|, right...]` へ揃えてから乗算することで、
/// `left`／`right` 軸を broadcast 経由で外積的に展開する
/// （`docs/spec` に無い一般化だが、PyTorch の `sumproduct_pair` が
/// `contract` 空の場合に行う分解と同型）。
fn einsum_mul_path<'t>(plan: EinsumMulPlan<'_, 't>) -> Result<Var<'t>, AutodiffError> {
    let EinsumMulPlan {
        a,
        b,
        a_labels,
        b_labels,
        batch,
        left,
        right,
        out_labels,
        dim_of,
    } = plan;

    let sizes = |labels: &[char]| -> Vec<usize> { labels.iter().map(|c| dim_of[c]).collect() };

    // A: [batch..., left...] へ並べ替えたのち、right 分の size-1 軸を
    // 末尾へ追加する（broadcast で b 側の right 軸に合わせるため）。
    let mut a_target: Vec<char> = batch.to_vec();
    a_target.extend_from_slice(left);
    let a_perm = apply_permute(a, &a_labels, &a_target)?;
    let mut a_shape = sizes(batch);
    a_shape.extend(sizes(left));
    a_shape.extend(std::iter::repeat_n(1usize, right.len()));
    let a_reshaped = apply_reshape(a_perm.contiguous()?, &a_shape)?;

    // B: [batch..., right...] へ並べ替えたのち、left 分の size-1 軸を
    // batch と right の間へ挿入する。
    let mut b_target: Vec<char> = batch.to_vec();
    b_target.extend_from_slice(right);
    let b_perm = apply_permute(b, &b_labels, &b_target)?;
    let mut b_shape = sizes(batch);
    b_shape.extend(std::iter::repeat_n(1usize, left.len()));
    b_shape.extend(sizes(right));
    let b_reshaped = apply_reshape(b_perm.contiguous()?, &b_shape)?;

    let product = a_reshaped.mul(&b_reshaped)?;

    let mut cur_labels: Vec<char> = batch.to_vec();
    cur_labels.extend_from_slice(left);
    cur_labels.extend_from_slice(right);
    apply_permute(product, &cur_labels, out_labels)
}

/// [`einsum_matmul_path`] へ渡す引数群（`EinsumMulPlan` と同じ理由で
/// 構造体化）。
struct EinsumMatmulPlan<'a, 't> {
    a: Var<'t>,
    b: Var<'t>,
    a_labels: Vec<char>,
    b_labels: Vec<char>,
    contract: &'a [char],
    left: &'a [char],
    right: &'a [char],
    out_labels: &'a [char],
    dim_of: &'a HashMap<char, usize>,
}

/// `contract` が非空（かつ `batch` が空。呼び出し元 `compute_binary_
/// plan` で検査済み）の二項 einsum を `Var::matmul`（GEMM）で計算する。
/// `contract` に複数添字が含まれる場合は、それらをまとめて 1 本の
/// `K` 軸へ `reshape` してから 2 次元 `matmul` を 1 回呼ぶ（多軸縮約を
/// 単一 GEMM 呼び出しへ帰着させる標準的な手法）。恒等 permute・shape
/// 不変の reshape はいずれもスキップされる（`apply_permute`／
/// `apply_reshape`）ため、`"ij,jk->ik"` は `MatMul` ノード 1 個だけを
/// tape へ記録する（`Var::matmul` 直接呼び出しと bit 同一になる根拠）。
fn einsum_matmul_path<'t>(plan: EinsumMatmulPlan<'_, 't>) -> Result<Var<'t>, AutodiffError> {
    let EinsumMatmulPlan {
        a,
        b,
        a_labels,
        b_labels,
        contract,
        left,
        right,
        out_labels,
        dim_of,
    } = plan;

    // A: [left..., contract...] → [L, K]。
    let mut a_target: Vec<char> = left.to_vec();
    a_target.extend_from_slice(contract);
    let a_perm = apply_permute(a, &a_labels, &a_target)?;
    let l = checked_numel(left, dim_of)?;
    let k = checked_numel(contract, dim_of)?;
    let a_2d = apply_reshape(a_perm.contiguous()?, &[l, k])?;

    // B: [contract..., right...] → [K, R]。
    let mut b_target: Vec<char> = contract.to_vec();
    b_target.extend_from_slice(right);
    let b_perm = apply_permute(b, &b_labels, &b_target)?;
    let r = checked_numel(right, dim_of)?;
    let b_2d = apply_reshape(b_perm.contiguous()?, &[k, r])?;

    // GEMM 本体（`Var::matmul`。FMA 契約・TF32 opt-in 挙動は `matmul`
    // と完全に同一——本モジュールが新規カーネルを追加しない中核）。
    let out_2d = a_2d.matmul(&b_2d)?;

    // [L, R] → [left dims..., right dims...]（L=R=1 の空次元も含め、
    // matmul 出力は常に contiguous のため reshape は非 contiguous
    // エラーになりえない）。
    let mut out_shape: Vec<usize> = left.iter().map(|c| dim_of[c]).collect();
    out_shape.extend(right.iter().map(|c| dim_of[c]));
    let out_reshaped = apply_reshape(out_2d, &out_shape)?;

    let mut cur_labels: Vec<char> = left.to_vec();
    cur_labels.extend_from_slice(right);
    apply_permute(out_reshaped, &cur_labels, out_labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;
    use fandhe_ai_tensor_core::Tensor;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
    }

    #[test]
    fn parse_accepts_explicit_output() {
        let spec = parse("ij,jk->ik").unwrap();
        assert_eq!(spec.inputs, vec![vec!['i', 'j'], vec!['j', 'k']]);
        assert_eq!(spec.output, vec!['i', 'k']);
    }

    #[test]
    fn parse_ignores_whitespace() {
        let spec = parse(" i j , j k -> i k ").unwrap();
        assert_eq!(spec.inputs, vec![vec!['i', 'j'], vec!['j', 'k']]);
        assert_eq!(spec.output, vec!['i', 'k']);
    }

    #[test]
    fn parse_infers_output_when_omitted() {
        // "ij,jk" は j が 2 回・i と k が 1 回ずつ現れるため、既定出力は
        // 「1 回だけ現れる添字を昇順」= "ik"。
        let spec = parse("ij,jk").unwrap();
        assert_eq!(spec.output, vec!['i', 'k']);
    }

    #[test]
    fn parse_infers_output_sums_all_when_all_repeated() {
        // "ij,ij" は i・j とも 2 回ずつ現れるため既定出力は空（全縮約）。
        let spec = parse("ij,ij").unwrap();
        assert_eq!(spec.output, Vec::<char>::new());
    }

    #[test]
    fn parse_rejects_ellipsis() {
        assert!(matches!(
            parse("...ij->...ji"),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn parse_rejects_duplicate_axis_within_operand() {
        assert!(matches!(
            parse("ii->i"),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn parse_rejects_duplicate_output_axis() {
        assert!(matches!(
            parse("ij->ii"),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn parse_rejects_unknown_output_axis() {
        assert!(matches!(
            parse("ij->k"),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn parse_rejects_non_ascii_alphabetic() {
        assert!(matches!(
            parse("i1->i"),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn parse_rejects_empty_operand_group() {
        assert!(matches!(
            parse(",jk->k"),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    /// `Var::contiguous` は既に contiguous な入力に対して新規ノードを
    /// 積まず自身をそのまま返す（`permute` チェーンの有無に関わらない
    /// passthrough 契約。イシュー #1620）。
    #[test]
    fn contiguous_is_passthrough_for_already_contiguous_value() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let nodes_before = tape.nodes.borrow().len();
        let c = a.contiguous().unwrap();
        let nodes_after = tape.nodes.borrow().len();
        assert_eq!(
            nodes_before, nodes_after,
            "contiguous な入力は新規ノードを積まない"
        );
        assert_eq!(c.node_id(), a.node_id());
    }

    /// `Var::contiguous` は permute で non-contiguous になった値に対し
    /// `Op::Contiguous` ノードを 1 個だけ積む（イシュー #1620）。
    #[test]
    fn contiguous_materializes_non_contiguous_permute_result() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let permuted = a.permute(&[1, 0]).unwrap();
        let nodes_before = tape.nodes.borrow().len();
        let c = permuted.contiguous().unwrap();
        let nodes_after = tape.nodes.borrow().len();
        assert_eq!(
            nodes_after,
            nodes_before + 1,
            "非 contiguous な入力は Op::Contiguous を 1 個積む"
        );
        assert_ne!(c.node_id(), permuted.node_id());
        assert_eq!(c.shape(), vec![3, 2]);
    }

    /// `Op::Contiguous` の VJP は恒等パススルー（`grad.rs::
    /// vjp_dispatch_contiguous_returns_single_input` の end-to-end 版）。
    /// `permute` を挟んだ `contiguous()` の backward が期待どおりの
    /// shape・一様勾配になることを確認する（イシュー #1620）。
    #[test]
    fn contiguous_backward_passes_upstream_through_inverse_permute() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let permuted = a.permute(&[1, 0]).unwrap();
        let c = permuted.contiguous().unwrap();
        // sum() を 2 回適用してスカラー損失を作る（`.claude/rules/
        // coding-rust.md` のテスト規約に沿い決定的な合成のみを使う）。
        let loss = c.sum(None).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let da = grads.get(&a).unwrap().expect("a は loss に到達する");
        // 全要素 1 の一様勾配は permute／contiguous のいずれを経ても
        // 形状（[2,3]）どおり全 1 になる（縮約が sum のみのため）。
        assert_eq!(da.shape(), &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(da.get(&[i, j]), Some(1.0));
            }
        }
    }

    /// `Err` を返す入口（batch 添字を伴う縮約の拒否）の後、tape に
    /// ノードが 1 つも push されていないこと（迷子ノードが残らない
    /// こと）を確認する。`crate::einsum::einsum`（`Var::einsum` の
    /// 実装本体）を直接呼び、presum の `Var::sum` が実行される**前**に
    /// `compute_binary_plan` の拒否判定へ到達する設計（モジュール doc
    /// 「検証と Var 操作の分離」）を検証する回帰テスト（イシュー
    /// #1620）。
    #[test]
    fn einsum_rejects_batch_contraction_without_pushing_nodes() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![0.0; 2 * 3 * 4], &[2, 3, 4]));
        let b = tape.var(&t(vec![0.0; 2 * 4 * 5], &[2, 4, 5]));
        let nodes_before = tape.nodes.borrow().len();
        let result = einsum("bij,bjk->bik", &[&a, &b]);
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
        let nodes_after = tape.nodes.borrow().len();
        assert_eq!(
            nodes_before, nodes_after,
            "batch 添字を伴う縮約の拒否は Var::sum（presum）を一切 push しない"
        );
    }
}
