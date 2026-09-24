//! GPU 融合 elementwise カーネル（`crate::fused_elementwise`）の
//! ホスト逐語モデル（区分 B-1・イシュー #2085）。
//!
//! `crate::kernels_fused_elementwise::generate_source` が生成する
//! CUDA C カーネルと**同じ発生順・同じ演算定義**で
//! [`fandhe_ai_tensor_core::FusedOpKind`] 列を純 Rust で評価する
//! オラクル。`Add`／`Mul` は Rust の `+`／`*`（Rust は暗黙の FMA 縮約を
//! 行わないため CUDA 側の非縮約 intrinsic `__fadd_rn`／`__fmul_rn` と
//! 同じ丸めになる）、`Relu` は生成カーネルと同じ三項式
//! `if x > 0.0 { x } else { 0.0 }`（`backend-cpu::fused_elementwise::
//! eval_one` の `x.max(0.0)` とは **符号付きゼロの扱いが異なりうる**。
//! `-0.0` 入力時 `x > 0.0` は false のため `0.0`〈正〉を返すが
//! `(-0.0).max(0.0)` の符号ビットは実装依存。CPU 融合カーネルとの
//! bit 突合〈実装計画 §5.1 (e)〉ではこの差異を踏まえ `-0.0` 入力を
//! 除外する）、`Exp`／`Tanh` は `f32::exp`／`f32::tanh`（プロセス内の
//! libm 呼び出しであり CUDA デバイス側 `expf`／`tanhf` と bit 一致する
//! 保証はない——本モデルは CUDA デバイス出力とは REQ-2 複合判定でのみ
//! 突合する契約であり、本モデル自身の bit 完全一致契約は
//! `crate::fused_elementwise::compile_program`／`launch_program_f32`
//! （実機経由）の**プロセス内 CPU 参照実装**である
//! `backend-cpu::fused_elementwise::run_fused_elementwise`
//! との突合に限る。実装計画 §2.7）。
//!
//! # 公開範囲
//!
//! `#[cfg(test)]` にしない（`crates/facade` の統合テスト
//! 〈勾配 bit 完全一致検証。実装計画 §5.1 (d)〉から `Tape::new_with_ops`
//! のフィクスチャ経由で使うため。`pooling_model.rs` 等の CPU クレート
//! 内 `#[cfg(test)]` オラクルとは異なる公開方針）。`#[doc(hidden)]`
//! （`lib.rs`）により `docs.rs` 上の公開 API 一覧には現れない
//! （`docs/compat-api-scope.md` §0 の「内部クレートの跨クレート限定
//! 公開」と同じ整理: `fandhe_ai::compat` からは再エクスポートしない）。
//! デバイス非依存の純 Rust 関数のみで構成する。

use fandhe_ai_tensor_core::FusedOpKind;

/// `crate::fused_elementwise::ElementwiseProgram::ops` と同じ `ops`
/// 列・`leaves` を受け取り、出力ベクタを返す（純関数）。
///
/// `leaves` は各要素が同一長（呼び出し元が `plan.output_shape()` との
/// 一致を検証済みである契約。`backend-cpu::fused_elementwise::
/// run_fused_elementwise` と同じ呼び出し規約）。`ops` は
/// `crate::fused_elementwise::match_elementwise_plan` が受理した
/// allowlist 済みの列のみを渡す契約（未知 variant は安全な既定値
/// `0.0` を返す防御的分岐。`kernels_fused_elementwise.rs::
/// generate_source` と同じ多層防御）。
// `i` は `leaves` の添字だけでなく、各要素位置ごとに `regs` バッファを
// 使い回すループカウンタとしても使うため（`i` を介した `leaves[..][i]`
// アクセスは `eval_program_host` 内側の演算ループで発生する）、
// clippy の「イテレータへ書き換えよ」提案（`needless_range_loop`）は
// 適用しない（提案どおり書き換えると `leaves` 全体のイテレータと
// `regs` の共有可変状態が絡み、かえって可読性が落ちる）。
#[allow(clippy::needless_range_loop)]
pub fn eval_program_host(ops: &[FusedOpKind], leaves: &[&[f32]]) -> Vec<f32> {
    if ops.is_empty() {
        return Vec::new();
    }
    let numel = leaves.first().map_or(0, |s| s.len());
    let output_index = ops.len() - 1;
    let mut out = Vec::with_capacity(numel);
    let mut regs = vec![0.0f32; ops.len()];
    for i in 0..numel {
        for (idx, op) in ops.iter().enumerate() {
            regs[idx] = match *op {
                FusedOpKind::Input { leaf_index } => leaves[leaf_index][i],
                // Rust の `+`／`*` は暗黙の FMA 縮約を行わないため
                // CUDA 側の `__fadd_rn`／`__fmul_rn`（非縮約 intrinsic）
                // と同じ丸めになる（モジュール冒頭コメント参照）。
                FusedOpKind::Add { lhs, rhs } => regs[lhs] + regs[rhs],
                FusedOpKind::Mul { lhs, rhs } => regs[lhs] * regs[rhs],
                // 生成カーネルと同一の三項式（`x.max(0.0)` ではない。
                // モジュール冒頭コメント「`-0.0` の扱い」参照）。
                FusedOpKind::Relu { input } => {
                    if regs[input] > 0.0 {
                        regs[input]
                    } else {
                        0.0
                    }
                }
                FusedOpKind::Exp { input } => regs[input].exp(),
                FusedOpKind::Tanh { input } => regs[input].tanh(),
                // allowlist 検証済みの呼び出し契約のため到達しない
                // 防御的分岐（`FusedOpKind` は `#[non_exhaustive]`）。
                _ => 0.0,
            };
        }
        out.push(regs[output_index]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_program_host_empty_ops_returns_empty_vec() {
        assert_eq!(eval_program_host(&[], &[]), Vec::<f32>::new());
    }

    #[test]
    fn eval_program_host_matches_manual_computation_for_chain() {
        // i0, i1, add(0,1), relu(2), exp(3), tanh(4)
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Input { leaf_index: 1 },
            FusedOpKind::Add { lhs: 0, rhs: 1 },
            FusedOpKind::Relu { input: 2 },
            FusedOpKind::Exp { input: 3 },
            FusedOpKind::Tanh { input: 4 },
        ];
        let a = [1.0f32, -2.0, 3.0];
        let b = [0.5f32, 1.5, -4.0];
        let out = eval_program_host(&ops, &[&a, &b]);
        let expected: Vec<f32> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let s = x + y;
                let r = if s > 0.0 { s } else { 0.0 };
                r.exp().tanh()
            })
            .collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn eval_program_host_relu_ternary_form_treats_negative_zero_as_nonpositive() {
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Relu { input: 0 },
        ];
        let x = [-0.0f32];
        let out = eval_program_host(&ops, &[&x]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], 0.0);
        // 符号ビットは `+0.0`（`x > 0.0` が false の分岐は常に定数
        // `0.0` を返すため）。
        assert!(out[0].is_sign_positive());
    }
}
