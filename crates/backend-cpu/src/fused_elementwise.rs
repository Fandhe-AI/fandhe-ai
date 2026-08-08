//! CPU 融合カーネル（単一パス実行器。TASK-12.1c・#163）。
//!
//! `tensor_core::fusion`（TASK-12.1a〜c・#161〜#163）が検出・生成した
//! elementwise 連鎖を、`elementwise.rs` の per-op カーネル呼び出し
//! （葉の読み・出力の書きがノード数分発生する）ではなく、**出力要素
//! ごとに 1 回のレジスタ内評価で完結する単一パス**として実行する
//! （PoC-9 `ElemwiseFuse` 方式。`docs/fusion-graph-design.md` §2.4）。
//! `tensor_core::FusionPlan` の公開 DTO アクセサ（`ops`／`output_shape`／
//! `dtype`／`leaf_count`）のみを読み、`tensor_core` 内部の `pub(crate)`
//! 融合 IR（`FusionGraph` 等）には一切依存しない（設計書 §3.4
//! 「privacy 制約」）。
//!
//! # 呼び出し元・呼び出し先の文脈
//!
//! 本モジュールの [`run_fused_elementwise`] は、`BackendOps::run_fused`
//! （trait への追加・`CpuBackendOps` での override 実装はいずれも #164
//! のスコープ）から薄く委譲される想定の関数カーネルである
//! （`elementwise.rs`・`gemm.rs` と同じ「trait 定義なし・関数ベース」
//! 構成。`lib.rs` 冒頭コメント参照）。#163 時点では `CpuBackendOps` から
//! は呼ばれず、本クレートの統合テスト（`tests/fused_elementwise_parity.rs`）
//! が直接呼んで受け入れ条件（融合カーネルの数値が非融合実行と一致
//! すること）を検証する。
//!
//! # 出力ノードの契約
//!
//! `tensor_core::fusion::plan` モジュール冒頭のドキュメンテーション
//! コメントが確定する契約（「発生順で最後の `FusedOpKind` エントリが
//! 出力ノード」）をそのまま前提とする。
//!
//! # 数値契約（REQ-2・`.claude/rules/coding-rust.md`）
//!
//! スカラー演算の定義は `elementwise.rs` の per-op カーネルと完全に
//! 揃える（融合の有無で許容誤差・演算定義を変えない。設計書 §4
//! 「融合の有無で許容誤差を変えない」の実装面の裏付け）: `add`/`mul` は
//! `+`/`*`、`relu` は `x.max(0.0)`（NaN → 0.0。`elementwise.rs` の既存
//! 挙動と一致させる）、`exp`/`tanh` は `f32::exp`/`f32::tanh`。elementwise
//! 連鎖に積和畳み込みは現れないため FMA 契約（`f32::mul_add`。GEMM 専用）
//! は適用外（`elementwise.rs` 冒頭コメントの既存整理を踏襲）。
//!
//! # 境界検査（REQ-8）
//!
//! 添字アクセスはすべて Rust の境界検査付きインデックス（`Vec`／
//! スライスの `[]`）のみを用い、`get_unchecked` 等の unchecked アクセス
//! や `unsafe` は使わない。レジスタ配列の添字（`FusedNodeIndex`）は
//! `FusionPlan` 構築時（`from_segment`／`from_ops`）に「自ノードより
//! 手前のみを参照する」ことを検証済みだが、**性能下限・最適化の達成を
//! 理由に本モジュール側の境界検査を省略しない**（`.claude/rules/
//! coding-rust.md`「カーネル実装の境界検査」）——構築時検証はあくまで
//! `FusionPlan` の不変条件であり、本モジュールはそれに加えて安全な
//! （境界検査付きの）アクセス経路のみを用いることで二重に保護する。

use rayon::prelude::*;
use tensor_core::device::BackendError;
use tensor_core::{DType, FusedOpKind, FusionPlan, ShapeError, Tensor};

use crate::elementwise::PARALLEL_THRESHOLD;

/// `plan` が表す融合済み elementwise 連鎖を `leaves` を入力として実行し、
/// 単一の出力 `Tensor<f32>` を返す。
///
/// # 実行前検証（fail-closed。設計書 §4・§5、REQ-8）
///
/// 1. `leaves.len() == plan.leaf_count()`（不一致は
///    [`BackendError::ShapeMismatch`]。設計書 §3.4「`run_fused` の
///    `leaves` の長さはこの値と一致する契約」）
/// 2. `plan.dtype() == DType::F32`（それ以外は [`BackendError::Unsupported`]。
///    現状 `FusionOp` は F32 のみを対象とするため、実運用では到達しない
///    防御的検査）
/// 3. 各 leaf が `plan.output_shape()` と同一 shape かつ contiguous
///    であること（elementwise 恒等 shape 契約。`graph.rs::FusionGraph::push`
///    が構築時に検証する契約と同じ shape を実行時にも要求する）。
///    **非 contiguous な leaf は `contiguous()` で実体化せず拒否する**
///    ——非 contiguous 連鎖は `detect_fusion` が非融合へ倒す設計
///    （`docs/fusion-graph-design.md` §1・§2.3）であり、ここで暗黙
///    実体化すると設計判断を迂回することになるため（呼び出し元
///    〈#164〉のバグを隠さない）。
pub fn run_fused_elementwise(
    plan: &FusionPlan,
    leaves: &[&Tensor<f32>],
) -> Result<Tensor<f32>, BackendError> {
    if leaves.len() != plan.leaf_count() {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: plan.leaf_count(),
                actual: leaves.len(),
            },
        ));
    }
    if plan.dtype() != DType::F32 {
        return Err(BackendError::Unsupported(format!(
            "run_fused_elementwise: unsupported dtype {:?} (F32 elementwise only)",
            plan.dtype()
        )));
    }

    let output_shape = plan.output_shape();
    let mut leaf_slices: Vec<&[f32]> = Vec::with_capacity(leaves.len());
    for (i, leaf) in leaves.iter().enumerate() {
        if leaf.shape() != output_shape {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: output_shape.to_vec(),
                rhs: leaf.shape().to_vec(),
            }));
        }
        let slice = leaf.as_slice().ok_or_else(|| {
            BackendError::Unsupported(format!(
                "run_fused_elementwise: leaf {i} is non-contiguous (broadcast/transpose \
                 view); detect_fusion already routes such chains to the non-fused \
                 fallback (docs/fusion-graph-design.md §2.3), so a validly constructed \
                 FusionPlan should never reach this rejection in practice"
            ))
        })?;
        leaf_slices.push(slice);
    }

    let ops: Vec<FusedOpKind> = plan.ops().collect();
    // `tensor_core::fusion::plan` モジュール冒頭「出力ノードの契約」:
    // 発生順で最後のエントリが出力ノード。`FusionPlan::from_ops`／
    // `from_segment` はいずれも空 `ops`（少なくとも 1 個の elementwise
    // ノードを要求）を拒否済みのため、`ops` は必ず 1 要素以上を持つ
    // （空 `FusionPlan` はここに到達しない）。
    let output_index = ops.len().saturating_sub(1);

    let numel: usize = output_shape.iter().product();
    let mut out = vec![0.0f32; numel];

    if numel >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .enumerate()
            .for_each(|(i, o)| *o = eval_one(&ops, &leaf_slices, output_index, i));
    } else {
        for (i, o) in out.iter_mut().enumerate() {
            *o = eval_one(&ops, &leaf_slices, output_index, i);
        }
    }

    Tensor::new(out, output_shape).map_err(BackendError::ShapeMismatch)
}

/// 出力要素 `i`（フラット添字）における `ops` 全体の評価結果を返す。
///
/// レジスタ配列（`ops.len()` 長の `Vec<f32>`）へ発生順（トポロジカル順）
/// に書き込む単一パスのインタープリタ。fan-out（同一ノードが複数ノード
/// から参照される）はレジスタの再読で解決し、同じ中間値を再計算しない
/// （`docs/fusion-graph-design.md` §2.4「レジスタ内解決」の実体。
/// これにより本カーネルのメモリアクセスは葉の読み 1 回・出力の書き
/// 1 回に閉じる——融合によるカーネル呼び出し削減効果の実体）。
///
/// 添字はすべて境界検査付き（`[]`）。`ops`（`FusionPlan` 構築時検証済み）
/// が保証する「自ノードより手前のみを参照する」不変条件により、
/// `regs[lhs]`／`regs[rhs]`／`regs[input]` は常にこの時点で計算済みの
/// 要素を指す。
fn eval_one(ops: &[FusedOpKind], leaf_slices: &[&[f32]], output_index: usize, i: usize) -> f32 {
    let mut regs = vec![0.0f32; ops.len()];
    for (idx, op) in ops.iter().enumerate() {
        regs[idx] = match *op {
            FusedOpKind::Input { leaf_index } => leaf_slices[leaf_index][i],
            // 加減算のみで libm 非経由（`elementwise.rs::add_slice` と同一定義）。
            FusedOpKind::Add { lhs, rhs } => regs[lhs] + regs[rhs],
            // 乗算のみで libm 非経由（`elementwise.rs::mul_slice` と同一定義）。
            FusedOpKind::Mul { lhs, rhs } => regs[lhs] * regs[rhs],
            // `x.max(0.0)`。NaN 入力で 0.0 を返す（`elementwise.rs::relu_slice`
            // と同一定義。NaN 伝播しない Rust `f32::max` 仕様どおり）。
            FusedOpKind::Relu { input } => regs[input].max(0.0),
            // `f32::exp`（libm 経由。`elementwise.rs::exp_slice` と同一定義）。
            FusedOpKind::Exp { input } => regs[input].exp(),
            // `f32::tanh`（libm 経由。`elementwise.rs::tanh_slice` と同一定義）。
            FusedOpKind::Tanh { input } => regs[input].tanh(),
        };
    }
    regs[output_index]
}
