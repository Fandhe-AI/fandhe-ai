//! `MetalBackendOps::gemm`（`BackendOps` トレイト経由。`gemm_fp32_strict`
//! の既定実装が委譲する）の VJP 専用 NT/TN strided 入口（イシュー #1215）
//! の受け入れ基準対応テスト。`backend-cuda::tests::gemm_transposed_parity`
//! （#1214）と同じ構成方針を Metal 実機向けに移植するが、**数値契約が
//! 異なる**点に注意する: CPU（#1213）・CUDA（#1214）は転置元 storage を
//! 転置してから既存 NN カーネルへ渡す方式のため bit 完全一致契約だった
//! のに対し、Metal の NT/TN 入口は `dispatch_auto`
//! （`gemm_simdgroup_tiled`）とは異なるカーネル（`gemm_tiled_bias_act`。
//! classic strided）を通るため累積順序が変わりうる。したがって本
//! テストは `assert_eq!` によるビット一致を要求せず、REQ-2 統一複合
//! 判定（`fandhe_ai_backend_cpu::assert_parity`。相対誤差 1e-3 未満
//! または絶対誤差 1e-5 未満）のみで受け入れる（`docs/matmul-vjp-zero-
//! copy-decision.md` §4.4・`gemm_resident_lhs`〈#1040〉と同じ契約）。
//!
//! 型検査（Linux／CI 相当）:
//! ```sh
//! cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
//! ```
//! 実機実行（Apple Silicon 必須）:
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test gemm_transposed_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    // 実機依存テストのため決定的な軽量疑似乱数生成（外部依存追加を避ける。
    // `.claude/rules/deps-policy.md`）。xorshift 系の最小実装
    // （`backend-cuda::tests::gemm_transposed_parity` と同一実装）。
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state % 2000) as f32 - 1000.0) / 1000.0
        })
        .collect()
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().unwrap().to_vec()
}

/// `gemm(g, w.transpose_2d())`（NT: 元の B `[n,k]` を転置した `[k,n]`
/// view。`matmul_vjp` の d_input `g @ Wᵀ` が該当）が、`contiguous()` 経路
/// および CPU 参照実装（`matmul_reference_fma`）のいずれとも REQ-2
/// 複合判定で一致することを、8 の倍数でない形状を含む複数形状で確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn gemm_nt_transposed_view_matches_contiguous_and_cpu_reference_on_real_device() {
    let ops = MetalBackendOps::new();

    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (4, 8, 4),
        (33, 17, 9),
        (64, 256, 784),
        (64, 256, 10),
    ] {
        let g = tensor(random_matrix(0x1000 + m as u64, m * k), &[m, k]);
        // w: 論理形状 [n,k]（Linear.weight 相当）。transpose_2d() で [k,n] view。
        let w = tensor(random_matrix(0x2000 + n as u64, n * k), &[n, k]);
        let w_t = w.transpose_2d().unwrap();
        // 1x1 形状の転置は `as_slice()` が `Some` を返す退行ケース（行優先／
        // 列優先の区別が意味を持たない）ため、view 判定のアサーションは
        // 付けない（`m*k==1 || n*k==1` を含む形状群でも `gemm` 自体の
        // NT／従来経路いずれもここでは検証対象外——数値結果の一致のみ確認）。
        let actual = <MetalBackendOps as BackendOps>::gemm_fp32_strict(&ops, &g, &w_t).unwrap();
        let expected = <MetalBackendOps as BackendOps>::gemm_fp32_strict(
            &ops,
            &g,
            &tensor(contiguous_slice(&w_t), &[k, n]),
        )
        .unwrap();

        let mut cpu_expected = vec![0.0f32; m * n];
        fandhe_ai_backend_cpu::matmul_reference_fma(
            &contiguous_slice(&g),
            &contiguous_slice(&w_t),
            &mut cpu_expected,
            m,
            n,
            k,
        )
        .unwrap();

        assert_eq!(actual.shape(), expected.shape(), "m={m} k={k} n={n}");
        fandhe_ai_backend_cpu::assert_parity(
            &format!("gemm NT vs contiguous() m={m} k={k} n={n}"),
            &contiguous_slice(&actual),
            &contiguous_slice(&expected),
        );
        fandhe_ai_backend_cpu::assert_parity(
            &format!("gemm NT vs CPU reference m={m} k={k} n={n}"),
            &contiguous_slice(&actual),
            &cpu_expected,
        );
    }
}

/// [`gemm_nt_transposed_view_matches_contiguous_and_cpu_reference_on_real_device`]
/// の TN 版（`x.transpose_2d()`。`matmul_vjp` の d_weight `Aᵀ @ g`・
/// `Op::LinearResident.d_weight` `xᵀ @ g` が該当）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn gemm_tn_transposed_view_matches_contiguous_and_cpu_reference_on_real_device() {
    let ops = MetalBackendOps::new();

    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (4, 8, 4),
        (33, 17, 9),
        (256, 64, 784),
        (256, 64, 10),
    ] {
        let x = tensor(random_matrix(0x3000 + m as u64, m * k), &[m, k]);
        let x_t = x.transpose_2d().unwrap();
        // NT 側と同じ理由（1x1 退行ケース）で view 判定のアサーションは
        // 付けない。
        let g = tensor(random_matrix(0x4000 + n as u64, m * n), &[m, n]);

        let actual = <MetalBackendOps as BackendOps>::gemm_fp32_strict(&ops, &x_t, &g).unwrap();
        let expected = <MetalBackendOps as BackendOps>::gemm_fp32_strict(
            &ops,
            &tensor(contiguous_slice(&x_t), &[k, m]),
            &g,
        )
        .unwrap();

        let mut cpu_expected = vec![0.0f32; k * n];
        fandhe_ai_backend_cpu::matmul_reference_fma(
            &contiguous_slice(&x_t),
            &contiguous_slice(&g),
            &mut cpu_expected,
            k,
            n,
            m,
        )
        .unwrap();

        assert_eq!(actual.shape(), expected.shape(), "m={m} k={k} n={n}");
        fandhe_ai_backend_cpu::assert_parity(
            &format!("gemm TN vs contiguous() m={m} k={k} n={n}"),
            &contiguous_slice(&actual),
            &contiguous_slice(&expected),
        );
        fandhe_ai_backend_cpu::assert_parity(
            &format!("gemm TN vs CPU reference m={m} k={k} n={n}"),
            &contiguous_slice(&actual),
            &cpu_expected,
        );
    }
}

/// 両方転置（TT）は本イシューのスコープ外（従来どおり `contiguous()`
/// フォールバック）だが、計算結果自体が正しいことを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn gemm_tt_both_transposed_falls_back_but_still_correct_on_real_device() {
    let ops = MetalBackendOps::new();
    let (big_m, big_k, big_n) = (17usize, 23usize, 19usize);
    let orig_a = tensor(random_matrix(0x5000, big_k * big_m), &[big_k, big_m]);
    let a_t = orig_a.transpose_2d().unwrap(); // [big_m, big_k]
    let orig_b = tensor(random_matrix(0x6000, big_n * big_k), &[big_n, big_k]);
    let b_t = orig_b.transpose_2d().unwrap(); // [big_k, big_n]

    let actual = <MetalBackendOps as BackendOps>::gemm_fp32_strict(&ops, &a_t, &b_t).unwrap();
    let expected = <MetalBackendOps as BackendOps>::gemm_fp32_strict(
        &ops,
        &tensor(contiguous_slice(&a_t), &[big_m, big_k]),
        &tensor(contiguous_slice(&b_t), &[big_k, big_n]),
    )
    .unwrap();

    // TT は従来経路（`dispatch_auto`。両オペランドとも `contiguous()`
    // 済み）のまま不変のため bit 完全一致のはず。
    assert_eq!(contiguous_slice(&actual), contiguous_slice(&expected));
}

/// `narrow` 後の転置は、`layout::classify_2d` が `ld > rows`（非対応
/// 次元より大きい leading dimension）の列優先 view として分類できる
/// ため、**`contiguous()` フォールバックではなく NT strided 入口を
/// 経由する**（rank-2 の `narrow`＋`transpose_2d` は常に行優先／列優先の
/// いずれかに分類可能——`classify_2d` doc「`ld >= rows`／`ld >= cols`」
/// 参照。真にフォールバックする「分類不能」ケースは
/// [`gemm_tt_both_transposed_falls_back_but_still_correct_on_real_device`]
/// が担う）。本テストは strided 入口が `ld` が論理次元より大きい
/// パディング済み view を正しく扱うことを確認する。カーネルが異なる
/// ため §2 の理由により `assert_eq!` ではなく複合判定を使う。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn gemm_narrow_then_transpose_uses_strided_entry_with_padded_ld_and_is_correct_on_real_device() {
    let ops = MetalBackendOps::new();
    let (m, k, n) = (6usize, 5usize, 4usize);
    let w0 = tensor(random_matrix(0x7000, n * (k + 2)), &[n, k + 2]);
    let w_narrowed = w0.narrow(1, 1, k).unwrap();
    let w_t = w_narrowed.transpose_2d().unwrap();
    assert!(!w_t.is_contiguous());

    let g = tensor(random_matrix(0x8000, m * k), &[m, k]);
    let actual = <MetalBackendOps as BackendOps>::gemm_fp32_strict(&ops, &g, &w_t).unwrap();
    let expected = <MetalBackendOps as BackendOps>::gemm_fp32_strict(
        &ops,
        &g,
        &tensor(contiguous_slice(&w_t), &[k, n]),
    )
    .unwrap();

    fandhe_ai_backend_cpu::assert_parity(
        "gemm narrow-then-transpose (padded ld) vs contiguous()",
        &contiguous_slice(&actual),
        &contiguous_slice(&expected),
    );
}

/// `&dyn BackendOps` トレイトオブジェクト経由でも NT 入口が同結果を返す
/// ことを確認する（`autodiff::grad::matmul_vjp` は `dyn BackendOps` 経由
/// で `gemm_fp32_strict` を呼ぶため、この呼び出し経路自体を検証する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn gemm_nt_via_trait_object_matches_concrete_call_on_real_device() {
    let ops = MetalBackendOps::new();
    let dyn_ops: &dyn BackendOps = &ops;

    let (m, k, n) = (16usize, 32usize, 24usize);
    let g = tensor(random_matrix(0x9500, m * k), &[m, k]);
    let w = tensor(random_matrix(0x9600, n * k), &[n, k]);
    let w_t = w.transpose_2d().unwrap();

    let via_trait_object = dyn_ops.gemm_fp32_strict(&g, &w_t).unwrap();
    let via_concrete = <MetalBackendOps as BackendOps>::gemm_fp32_strict(&ops, &g, &w_t).unwrap();

    assert_eq!(
        contiguous_slice(&via_trait_object),
        contiguous_slice(&via_concrete)
    );
}
