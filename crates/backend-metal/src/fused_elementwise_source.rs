//! GPU `run_fused` の elementwise allowlist 融合カーネル向け MSL ソース
//! 生成（実行時コンパイル用の文字列組み立て。区分 B-1・イシュー
//! #2085・`docs/autodiff-graph-optimization-scope-decision.md` §5。
//! CUDA 側 `backend-cuda::kernels_fused_elementwise` の Metal 対応版）。
//!
//! `crate::scalar_op_source`（`ScalarUnaryOp`／`ScalarBinaryOp` の kind
//! ごとの実行時ソース生成）と同型の方式で、[`fandhe_ai_tensor_core::
//! FusedOpKind`] の列（`FusionPlan::ops()` の発生順写し）から都度
//! ソースを組み立てる純関数（[`generate_source`]）を提供する。呼び出し元
//! （`fused_elementwise.rs::match_elementwise_plan` が受理した
//! `ElementwiseProgram`）は既に allowlist 検証済みの op 列のみを渡す
//! 契約であり、本モジュール自体は検証を行わない。
//!
//! # 数値契約（REQ-2・`.claude/rules/coding-rust.md`）
//!
//! [`crate::pipeline::compile_source`] は必ず `MathMode::Safe` +
//! `MathFloatingPointFunctions::Precise`（`pipeline.rs::compile_options`）
//! を適用する（迂回禁止契約。`pipeline.rs` モジュール冒頭コメント「2
//! 経路で確実に同一適用」）。この既定下で `+`／`*` は correctly rounded
//! かつ**暗黙の FMA 縮約を行わない**（`scalar_op_source.rs` モジュール
//! 冒頭「コンパイルオプションと数値契約」で `+ - * /` が correctly
//! rounded・bit 同一と既に確立済みの契約と同一根拠であり、本モジュールは
//! 新たな pragma を追加しない——**`#pragma METAL fp contract(off)` は
//! 使わない**: リポジトリ内に既存使用例・検証済み綴りがなく〈実装計画
//! §2.4 注記〉、`MathMode::Safe` 単独で既に `Add`／`Mul` に対して同一の
//! 縮約遮断契約を達成している既存 kind〈`scalar_op_source.rs` の
//! `Sub`／`Div`〉があるため、未検証の pragma を追加するより既存契約への
//! 依拠を選ぶ。`Relu` は `elementwise.metal::ew_relu_f32` と同一の三項式
//! （`x > 0.0f ? x : 0.0f`）、`Exp`／`Tanh` は同一の `metal::precise::exp`／
//! `metal::precise::tanh` を用いる。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `elementwise.metal` と同じ `if (idx < numel)` 手動境界チェックを
//! 維持する（性能下限達成を理由に省略しない）。
//!
//! 実際の呼び出し元（`ops.rs::MetalBackendOps::
//! run_fused_elementwise_allowlist`）は `cfg(target_os = "macos")`
//! 限定のため、Linux 単体ビルドでは本モジュールの項目が「クレート内
//! から到達不能」と判定され dead_code lint が誤検知する
//! （`row_kernel.rs`・`fused_elementwise.rs` と同じ理由・同じ対処）。

#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use fandhe_ai_tensor_core::FusedOpKind;

/// 生成カーネルの関数名（固定リテラル）。CUDA 側
/// `kernels_fused_elementwise::FUSED_EW_FUNCTION_NAME` と同じ理由
/// （プランごとに個別コンパイル・ロードするため一意性はキャッシュキー側
/// のみで担保すればよい）で固定し、`pipeline::make_pipeline` が要求する
/// `&'static str` をリテラルのまま満たす（`Box::leak` 不要。
/// `context_cache::cached_fused_elementwise_pipeline` doc 参照）。
pub(crate) const FUSED_EW_FUNCTION_NAME: &str = "fused_ew_f32";

/// [`crate::kernels_fused_elementwise::cache_key`]（CUDA 側）と同一の
/// 正準文字列規約でキャッシュキーを生成する（両バックエンドでキー
/// 生成規約を揃えても実害はないが、独立した実装であり相互運用は
/// 前提としない）。
pub(crate) fn cache_key(ops: &[FusedOpKind], leaf_count: usize) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(ops.len());
    for op in ops {
        parts.push(match *op {
            FusedOpKind::Input { leaf_index } => format!("i{leaf_index}"),
            FusedOpKind::Add { lhs, rhs } => format!("add({lhs},{rhs})"),
            FusedOpKind::Mul { lhs, rhs } => format!("mul({lhs},{rhs})"),
            FusedOpKind::Relu { input } => format!("relu({input})"),
            FusedOpKind::Exp { input } => format!("exp({input})"),
            FusedOpKind::Tanh { input } => format!("tanh({input})"),
            // allowlist 検証済みの呼び出し契約のため到達しない防御的分岐
            // （`FusedOpKind` は `#[non_exhaustive]`）。
            _ => "?".to_string(),
        });
    }
    format!("{}|leaves={leaf_count}", parts.join(";"))
}

/// `ops` を単一パスの MSL カーネルソースへ変換する。
///
/// 生成手順: 葉引数 `l0..l{leaf_count-1}`（`device const float*
/// [[buffer(i)]]`）を宣言順に並べ、続けて `out`（`device float*
/// [[buffer(leaf_count)]]`）・`numel`（`constant uint&
/// [[buffer(leaf_count+1)]]`）を宣言する。カーネル本体は
/// `if (idx < numel)` ガード内で `ops` を発生順に評価し、各ノードを
/// ローカル変数 `r{i}` へ書き込む。出力ノードは発生順で最後のエントリ
/// （`fusion::plan` モジュール冒頭「出力ノードの契約」）。
///
/// `ops` が空の場合は空文字列を返す（`match_elementwise_plan` は空
/// `ops` を持つ `FusionPlan` を構築しない契約のため実運用では到達しない
/// 防御的空処理。CUDA 側 `generate_source` と同一方針）。
pub(crate) fn generate_source(ops: &[FusedOpKind], leaf_count: usize) -> String {
    if ops.is_empty() {
        return String::new();
    }

    let mut params = String::new();
    for i in 0..leaf_count {
        params.push_str(&format!("    device const float* l{i} [[buffer({i})]],\n"));
    }
    let out_index = leaf_count;
    let numel_index = leaf_count + 1;
    params.push_str(&format!(
        "    device float* out [[buffer({out_index})]],\n    constant uint& numel \
         [[buffer({numel_index})]],\n    uint idx [[thread_position_in_grid]]"
    ));

    let mut body = String::new();
    for (i, op) in ops.iter().enumerate() {
        let line = match *op {
            FusedOpKind::Input { leaf_index } => format!("float r{i} = l{leaf_index}[idx];"),
            // `MathMode::Safe` 下で correctly rounded・非縮約
            // （モジュール冒頭「数値契約」参照）。
            FusedOpKind::Add { lhs, rhs } => format!("float r{i} = r{lhs} + r{rhs};"),
            FusedOpKind::Mul { lhs, rhs } => format!("float r{i} = r{lhs} * r{rhs};"),
            // `elementwise.metal::ew_relu_f32` と同一の三項式。
            FusedOpKind::Relu { input } => {
                format!("float r{i} = r{input} > 0.0f ? r{input} : 0.0f;")
            }
            FusedOpKind::Exp { input } => format!("float r{i} = metal::precise::exp(r{input});"),
            FusedOpKind::Tanh { input } => format!("float r{i} = metal::precise::tanh(r{input});"),
            // allowlist 検証済みの呼び出し契約のため到達しない防御的分岐。
            _ => format!("float r{i} = 0.0f;"),
        };
        body.push_str("        ");
        body.push_str(&line);
        body.push('\n');
    }
    let output_index = ops.len() - 1;

    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\nkernel void \
         {FUSED_EW_FUNCTION_NAME}(\n{params}\n) {{\n    if (idx < numel) {{\n{body}        \
         out[idx] = r{output_index};\n    }}\n}}\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ops() -> Vec<FusedOpKind> {
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Input { leaf_index: 1 },
            FusedOpKind::Add { lhs: 0, rhs: 1 },
            FusedOpKind::Relu { input: 2 },
            FusedOpKind::Exp { input: 3 },
            FusedOpKind::Tanh { input: 4 },
        ]
    }

    #[test]
    fn cache_key_matches_cuda_side_canonical_form() {
        assert_eq!(
            cache_key(&sample_ops(), 2),
            "i0;i1;add(0,1);relu(2);exp(3);tanh(4)|leaves=2"
        );
    }

    #[test]
    fn generated_source_uses_precise_exp_and_tanh() {
        let src = generate_source(&sample_ops(), 2);
        assert!(src.contains("metal::precise::exp(r3)"));
        assert!(src.contains("metal::precise::tanh(r4)"));
    }

    #[test]
    fn generated_source_uses_ternary_relu() {
        let src = generate_source(&sample_ops(), 2);
        assert!(src.contains("float r3 = r2 > 0.0f ? r2 : 0.0f;"));
    }

    #[test]
    fn generated_source_has_boundary_guard() {
        let src = generate_source(&sample_ops(), 2);
        assert!(src.contains("if (idx < numel)"));
    }

    #[test]
    fn generated_source_output_assigns_last_register() {
        let src = generate_source(&sample_ops(), 2);
        assert!(src.contains("out[idx] = r5;"));
    }

    #[test]
    fn generated_source_declares_buffer_indices_in_order() {
        let src = generate_source(&sample_ops(), 2);
        assert!(src.contains("l0 [[buffer(0)]]"));
        assert!(src.contains("l1 [[buffer(1)]]"));
        assert!(src.contains("out [[buffer(2)]]"));
        assert!(src.contains("numel [[buffer(3)]]"));
    }

    #[test]
    fn generated_source_empty_ops_returns_empty_string() {
        assert_eq!(generate_source(&[], 0), "");
    }
}
