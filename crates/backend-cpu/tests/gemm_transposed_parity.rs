//! `CpuBackendOps::gemm`／`gemm_resident_lhs` の VJP 専用 NT/TN 2 パターン
//! 入口（イシュー #1213）の受け入れ基準対応テスト。
//!
//! `Tensor::transpose_2d()` の zero-copy view を `gemm`／`gemm_resident_lhs`
//! へ渡した結果が、同じ view を `contiguous()`（明示的な転置コピー）して
//! から渡した結果と bit 完全一致することを検証する（本イシューの設計上
//! 「NT/TN 入口は contiguous() 後に NN で pack した panel と同一バイト列
//! を書く」ため、REQ-2 統一複合判定ではなく bit 完全一致で比較する）。
//! `narrow` 後の転置（一般 stride）・TT（両方転置）は本イシューのスコープ
//! 外で `contiguous()` フォールバックのまま結果が正しいことのみを確認する。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::{CpuBackendOps, CpuMemory};
use fandhe_ai_tensor_core::BackendOps;
use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::buffer::{DeviceBufferView, MemoryOps};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().unwrap().to_vec()
}

/// `gemm(g, w.transpose_2d())`（NT: 元の B `[n,k]` を転置した `[k,n]`
/// view）が `gemm(g, w.transpose_2d().contiguous())` と bit 完全一致する
/// ことを、MR/NR/MC/KC/NC 境界を跨ぐ複数形状で確認する（`matmul_vjp` の
/// d_input `g @ Wᵀ` が該当）。`(1,1,1)` は転置しても contiguous のまま
/// （単一要素）だが、判定条件（`strides == [1, shape[0]]`）自体は成立し
/// NT 入口を経由するため境界ケースとして含める。
#[test]
fn gemm_nt_transposed_view_matches_contiguous_across_shapes() {
    let ops = CpuBackendOps::new();
    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (4, 8, 4),
        (37, 65, 33),
        (128, 129, 96),
    ] {
        let g = tensor(random_matrix(0x1000 + m as u64, m * k), &[m, k]);
        // w: 論理形状 [n,k]（Linear.weight 相当）。transpose_2d() で [k,n] view。
        let w = tensor(random_matrix(0x2000 + n as u64, n * k), &[n, k]);
        let w_t = w.transpose_2d().unwrap();

        let actual = ops.gemm(&g, &w_t).unwrap();
        let expected = ops
            .gemm(&g, &tensor(contiguous_slice(&w_t), &[k, n]))
            .unwrap();

        assert_eq!(actual.shape(), expected.shape(), "m={m} k={k} n={n}");
        assert_eq!(
            contiguous_slice(&actual),
            contiguous_slice(&expected),
            "gemm の NT 入口は contiguous() 経路と bit 完全一致するはず（m={m} k={k} n={n}）"
        );
    }
}

/// `gemm(x.transpose_2d(), g)`（TN: 元の A `[m,k]` を転置した `[k,m]`
/// view）版（`matmul_vjp` の d_weight `Aᵀ @ g` が該当）。
#[test]
fn gemm_tn_transposed_view_matches_contiguous_across_shapes() {
    let ops = CpuBackendOps::new();
    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (4, 8, 4),
        (37, 65, 33),
        (128, 129, 96),
    ] {
        let x = tensor(random_matrix(0x3000 + m as u64, m * k), &[m, k]);
        let x_t = x.transpose_2d().unwrap();
        let g = tensor(random_matrix(0x4000 + n as u64, m * n), &[m, n]);

        let actual = ops.gemm(&x_t, &g).unwrap();
        let expected = ops
            .gemm(&tensor(contiguous_slice(&x_t), &[k, m]), &g)
            .unwrap();

        assert_eq!(actual.shape(), expected.shape(), "m={m} k={k} n={n}");
        assert_eq!(
            contiguous_slice(&actual),
            contiguous_slice(&expected),
            "gemm の TN 入口は contiguous() 経路と bit 完全一致するはず（m={m} k={k} n={n}）"
        );
    }
}

/// 両方転置（TT）は本イシューのスコープ外（従来どおり `contiguous()`
/// フォールバック）だが、計算結果自体は正しいことを確認する。
///
/// `gemm` の `a`（形状 `[big_m, big_k]`）を「元の `[big_k, big_m]` を
/// 転置した view」として、`b`（形状 `[big_k, big_n]`）を「元の
/// `[big_n, big_k]` を転置した view」として与える（NT／TN 双方の
/// 転置元の作り方を組み合わせる）。
#[test]
fn gemm_tt_both_transposed_falls_back_but_still_correct() {
    let ops = CpuBackendOps::new();
    let (big_m, big_k, big_n) = (17usize, 23usize, 19usize);
    let orig_a = tensor(random_matrix(0x5000, big_k * big_m), &[big_k, big_m]);
    let a_t = orig_a.transpose_2d().unwrap(); // [big_m, big_k]
    let orig_b = tensor(random_matrix(0x6000, big_n * big_k), &[big_n, big_k]);
    let b_t = orig_b.transpose_2d().unwrap(); // [big_k, big_n]

    let actual = ops.gemm(&a_t, &b_t).unwrap();
    let expected = ops
        .gemm(
            &tensor(contiguous_slice(&a_t), &[big_m, big_k]),
            &tensor(contiguous_slice(&b_t), &[big_k, big_n]),
        )
        .unwrap();

    assert_eq!(contiguous_slice(&actual), contiguous_slice(&expected));
}

/// `narrow` 後の転置（一般 stride。`ld > rows` となり
/// `dense_transposed_view` の判定に落ちる）は `contiguous()` フォール
/// バック経由でも結果自体は正しいことを確認する（一般 stride 化は本
/// イシューのスコープ外。`docs/matmul-vjp-zero-copy-decision.md` §3.2）。
#[test]
fn gemm_narrow_then_transpose_general_stride_falls_back_but_still_correct() {
    let ops = CpuBackendOps::new();
    let (m, k, n) = (6usize, 5usize, 4usize);
    // w0: [n, k+2] から**列方向**に narrow して [n, k] を切り出し、
    // transpose_2d する。行方向（先頭次元）の narrow は offset のみで
    // ストライドが row_major のまま変わらず（transpose 後も
    // `dense_transposed_view` の判定に合致してしまい NT 経路を正しく
    // 通せる）、本テストが検証したい「一般 stride で判定に落ちる」
    // ケースにならないため、列方向 narrow で行ストライド（元の列数
    // k+2）が narrow 後の列数 k と食い違う真の一般 stride を作る。
    let w0 = tensor(random_matrix(0x7000, n * (k + 2)), &[n, k + 2]);
    let w_narrowed = w0.narrow(1, 1, k).unwrap();
    let w_t = w_narrowed.transpose_2d().unwrap();
    assert!(!w_t.is_contiguous());

    let g = tensor(random_matrix(0x8000, m * k), &[m, k]);
    let actual = ops.gemm(&g, &w_t).unwrap();
    let expected = ops
        .gemm(&g, &tensor(contiguous_slice(&w_t), &[k, n]))
        .unwrap();

    assert_eq!(contiguous_slice(&actual), contiguous_slice(&expected));
}

/// `gemm_resident_lhs(w_dev, b.transpose_2d())`（NT: `b` が転置格納）が、
/// `gemm_resident_lhs(w_dev, b.transpose_2d().contiguous())` と bit 完全
/// 一致することを確認する（`Op::LinearResident` の d_input `W @ gᵀ` が
/// 該当。イシュー #1213）。
#[test]
fn gemm_resident_lhs_nt_transposed_view_matches_contiguous() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();

    for &(p, q, r) in &[(1usize, 1usize, 1usize), (4, 8, 4), (37, 65, 33)] {
        let w = tensor(random_matrix(0x9000 + p as u64, p * q), &[p, q]);
        // b0: 論理形状 [r,q]。transpose_2d() で [q,r] view（gemm_resident_lhs
        // の第 2 引数 `b` の期待形状）。
        let b0 = tensor(random_matrix(0xa000 + r as u64, r * q), &[r, q]);
        let b_t = b0.transpose_2d().unwrap();

        let w_dev = mem.upload(&w).unwrap();
        let w_shape = [p, q];
        let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
        let actual = ops.gemm_resident_lhs(w_view, &b_t).unwrap();

        let w_view2 = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
        let expected = ops
            .gemm_resident_lhs(w_view2, &tensor(contiguous_slice(&b_t), &[q, r]))
            .unwrap();

        assert_eq!(actual.shape(), expected.shape(), "p={p} q={q} r={r}");
        assert_eq!(
            contiguous_slice(&actual),
            contiguous_slice(&expected),
            "gemm_resident_lhs の NT 入口は contiguous() 経路と bit 完全一致するはず（p={p} q={q} r={r}）"
        );
    }
}
