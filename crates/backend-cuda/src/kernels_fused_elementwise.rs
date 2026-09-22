//! GPU `run_fused` の elementwise allowlist 融合カーネル向け CUDA C
//! ソース生成（実行時 NVRTC コンパイル用の文字列組み立て。区分 B-1・
//! イシュー #2085・`docs/autodiff-graph-optimization-scope-decision.md`
//! §5）。
//!
//! `kernels_elementwise.rs`（固定 5 カーネルの静的文字列）とは異なり、
//! 本モジュールは [`fandhe_ai_tensor_core::FusedOpKind`] の列（`FusionPlan::ops()`
//! の発生順写し）から**都度**ソースを組み立てる純関数
//! （[`generate_source`]）を提供する。呼び出し元
//! （`fused_elementwise.rs::match_elementwise_plan` が受理した
//! `ElementwiseProgram`）は既に allowlist 検証済みの op 列のみを渡す
//! 契約であり、本モジュール自体は検証を行わない（fail-closed 判定は
//! 呼び出し元の責務。設計判断は `docs/autodiff-graph-optimization-scope-
//! decision.md` §5 B-1 を正とする）。
//!
//! # 数値契約（REQ-2・`.claude/rules/coding-rust.md`）
//!
//! `Add`／`Mul` は非縮約 intrinsic（`__fadd_rn`／`__fmul_rn`）で生成する
//! （`kernels_adam.rs` の先例と同一方針）。融合カーネル内で `Mul → Add`
//! が同一関数内に連続すると、NVRTC 既定の `--fmad=true`
//! （`nvrtc.rs::compile_ptx`）により FMA へ縮約され、per-op 経路
//! （`kernels_elementwise.rs::EW_ADD_F32`／`EW_MUL_F32`。生の `+`／`*`）・
//! CPU `backend-cpu::fused_elementwise::eval_one` と bit 不一致になる
//! （PR #2085 実装計画 §2.4「FMA 縮約の遮断」）。`Relu` は
//! `kernels_elementwise.rs::EW_RELU_F32` と同一の三項式
//! （`x > 0.0f ? x : 0.0f`）、`Exp`／`Tanh` は同一の単精度組み込み
//! （`expf`／`tanhf`）を用いる。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `kernels_elementwise.rs` と同じ `if (idx < numel)` 手動境界チェックを
//! 維持する（性能下限達成を理由に省略しない）。

use fandhe_ai_tensor_core::FusedOpKind;

/// 生成カーネルの関数名（固定リテラル）。PTX／MSL ライブラリはプラン
/// 単位で個別にコンパイル・ロードするため、関数名の一意性はキャッシュ
/// キー側（[`cache_key`]）のみで担保すればよい。固定名により
/// `context_cache::cached_fused_elementwise_kernel` の `load_function`
/// 呼び出しへ `&'static str` をそのまま渡せる（`Box::leak` 不要）。
pub(crate) const FUSED_EW_FUNCTION_NAME: &str = "fused_ew_f32";

/// `ops`（[`FusedOpKind`] 列。`FusionPlan::ops()` の発生順写し）と
/// `leaf_count` から、プロセス内キャッシュキーとして使う正準文字列を
/// 生成する（実装計画 §2.5 の例: `i0;i1;add(0,1);relu(2);exp(3);tanh(4)|
/// leaves=2`）。
///
/// `ops` は呼び出し元（[`crate::fused_elementwise::match_elementwise_plan`]）
/// が allowlist 検証済みであることを前提とする。未知 variant
/// （`FusedOpKind` は `#[non_exhaustive]`）が紛れ込んだ場合は `"?"` を
/// 埋め込みキーとして扱う（allowlist 検証済みの呼び出し契約のため実運用
/// では到達しない防御的分岐。到達した場合も [`generate_source`] 側の
/// 対応する防御的分岐が `float r{i} = 0.0f;` を生成しコンパイル自体は
/// 成立するため、コンパイル失敗ではなく静かな 0.0 フォールバックにな
/// る。ただし allowlist 検証済み契約のため実運用では到達せず実害はな
/// い。二重防御）。
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
            // （`FusedOpKind` は `#[non_exhaustive]`。CPU `eval_one` と
            // 同じ多層防御の考え方）。
            _ => "?".to_string(),
        });
    }
    format!("{}|leaves={leaf_count}", parts.join(";"))
}

/// `ops` を単一パスの CUDA C カーネルソースへ変換する。
///
/// 生成手順: 葉引数 `l0..l{leaf_count-1}`（`const float* __restrict__`）
/// を宣言順に並べ、続けて `out`（`float* __restrict__`）・`numel`（`int`）
/// を宣言する。カーネル本体は `if (idx < numel)` ガード内で `ops` を
/// 発生順に評価し、各ノードをローカル変数 `r{i}` へ書き込む
/// （`FusedNodeIndex` は「自ノードより手前のみ参照する」契約
/// 〈`fandhe_ai_tensor_core::fusion::plan` モジュール冒頭〉のため、`r{i}` は
/// 参照時点で必ず計算済み）。出力ノードは発生順で最後のエントリ
/// （`fusion::plan` モジュール冒頭「出力ノードの契約」）。
///
/// `ops` が空の場合は空文字列を返す（呼び出し元 `match_elementwise_plan`
/// は空 `ops` を持つ `FusionPlan` を構築しない契約〈`FusionPlanError::
/// NoElementwiseNode`〉のため実運用では到達しない。防御的空処理）。
pub(crate) fn generate_source(ops: &[FusedOpKind], leaf_count: usize) -> String {
    if ops.is_empty() {
        return String::new();
    }

    let mut params = String::new();
    for i in 0..leaf_count {
        params.push_str(&format!("const float* __restrict__ l{i}, "));
    }
    params.push_str("float* __restrict__ out, int numel");

    let mut body = String::new();
    for (i, op) in ops.iter().enumerate() {
        let line = match *op {
            FusedOpKind::Input { leaf_index } => format!("float r{i} = l{leaf_index}[idx];"),
            // 非縮約 intrinsic（モジュール冒頭「数値契約」参照）。
            FusedOpKind::Add { lhs, rhs } => format!("float r{i} = __fadd_rn(r{lhs}, r{rhs});"),
            FusedOpKind::Mul { lhs, rhs } => format!("float r{i} = __fmul_rn(r{lhs}, r{rhs});"),
            // `kernels_elementwise.rs::EW_RELU_F32` と同一の三項式。
            FusedOpKind::Relu { input } => {
                format!("float r{i} = r{input} > 0.0f ? r{input} : 0.0f;")
            }
            FusedOpKind::Exp { input } => format!("float r{i} = expf(r{input});"),
            FusedOpKind::Tanh { input } => format!("float r{i} = tanhf(r{input});"),
            // allowlist 検証済みの呼び出し契約のため到達しない防御的分岐。
            // 本番経路 panic 禁止方針（`.claude/rules/coding-rust.md`）に
            // 従い `unreachable!()` ではなく安全な既定値を書く（CPU
            // `eval_one` の `_ => 0.0` 分岐と同じ多層防御）。
            _ => format!("float r{i} = 0.0f;"),
        };
        body.push_str("        ");
        body.push_str(&line);
        body.push('\n');
    }
    let output_index = ops.len() - 1;

    format!(
        "extern \"C\" __global__ void {FUSED_EW_FUNCTION_NAME}({params})\n{{\n    int idx = \
         blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < numel) {{\n{body}        \
         out[idx] = r{output_index};\n    }}\n}}\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ops() -> Vec<FusedOpKind> {
        // i0, i1, add(0,1), relu(2), exp(3), tanh(4) — 4 段連鎖（実装計画
        // §5.1 (b) のソース証跡テスト対象パターン）。
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
    fn cache_key_matches_expected_canonical_form() {
        assert_eq!(
            cache_key(&sample_ops(), 2),
            "i0;i1;add(0,1);relu(2);exp(3);tanh(4)|leaves=2"
        );
    }

    #[test]
    fn cache_key_distinguishes_different_leaf_counts() {
        let ops = sample_ops();
        assert_ne!(cache_key(&ops, 2), cache_key(&ops, 3));
    }

    /// ソース証跡（実装計画 §5.1 (b)）: `Add`／`Mul` は非縮約 intrinsic
    /// を使い、生の `+`／`*` を演算本体に含まない。`if (idx < numel)`
    /// ガード内の本体行のみを対象とする（index 計算
    /// `blockIdx.x * blockDim.x + threadIdx.x` は対象外）。
    #[test]
    fn generated_source_uses_non_contracting_intrinsics_for_add_and_mul() {
        let src = generate_source(&sample_ops(), 2);
        assert!(src.contains("float r2 = __fadd_rn(r0, r1);"));
        assert!(!src.contains("r0 + r1"));
        // index 計算行（`blockIdx.x * blockDim.x + threadIdx.x`）は対象外。
        // 本体行（`float r<i> = ...;`）に限定して `*` の非使用を検査する。
        for line in src
            .lines()
            .filter(|l| l.trim_start().starts_with("float r"))
        {
            assert!(!line.contains(" * "), "本体行に生の乗算が含まれる: {line}");
        }
    }

    #[test]
    fn generated_source_uses_single_precision_expf_and_tanhf() {
        let src = generate_source(&sample_ops(), 2);
        assert!(src.contains("expf(r3)"));
        assert!(src.contains("tanhf(r4)"));
        // double 版（`exp(`／`tanh(`）を含まないこと。
        assert!(!src.contains(" exp(r3)"));
        assert!(!src.contains(" tanh(r4)"));
    }

    #[test]
    fn generated_source_uses_ternary_relu() {
        let src = generate_source(&sample_ops(), 2);
        assert!(src.contains("float r3 = r2 > 0.0f ? r2 : 0.0f;"));
    }

    #[test]
    fn generated_source_param_count_matches_leaf_count_plus_two() {
        let ops = sample_ops();
        let src = generate_source(&ops, 2);
        // シグネチャ行のみを対象に `,` の個数を数える（本体側の
        // 括弧付き引数と誤って混同しないよう最初の `{` 手前で切る）。
        let sig = src.split('{').next().unwrap();
        let comma_count = sig.matches(',').count();
        // leaf_count(=2) 個の葉引数 + out + numel = 4 引数
        // なので区切り `,` は 3 個。
        assert_eq!(comma_count, 3);
        let _ = ops;
    }

    #[test]
    fn generated_source_has_boundary_guard() {
        let src = generate_source(&sample_ops(), 2);
        assert!(src.contains("if (idx < numel)"));
    }

    #[test]
    fn generated_source_output_assigns_last_register() {
        let ops = sample_ops();
        let src = generate_source(&ops, 2);
        assert!(src.contains("out[idx] = r5;"));
    }

    #[test]
    fn generated_source_empty_ops_returns_empty_string() {
        assert_eq!(generate_source(&[], 0), "");
    }
}
