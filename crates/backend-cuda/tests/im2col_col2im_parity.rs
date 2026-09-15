//! イシュー #1766: `BackendOps::im2col`／`col2im`（Conv2d の im2col+GEMM
//! 展開・畳み戻し。設計 `docs/conv-ops-design.md` §1766）の CPU-CUDA
//! 数値一致検証。
//!
//! `constant_pad_parity.rs`（#1756）・`scan_parity.rs`（#1740）と同じ
//! 構成方針を踏襲する: 環境適応スモーク（属性なし。通常 CI で実行し、
//! CUDA 非搭載環境では `BackendError::CudaUnavailable` を確認して
//! panic しないことのみ検証。デバイス初期化より前に返る shape 検査
//! 経路は GPU 有無に依らず検証する）と、実機必須の形状網羅
//! （`#[ignore]`。DGX Spark GB10 等）を分離する。
//!
//! **契約**: `im2col` は算術を含まない純粋なコピー演算のため
//! **bit 完全一致**（`constant_pad_parity.rs::assert_bits_eq` と同じ
//! 判定ヘルパーを用いる）。`col2im` は `double`（CUDA ネイティブ）
//! アキュムレータの逐次加算・1 回 `f32` downcast で CPU 参照実装
//! （`backend-cpu::im2col::col2im` の `f64` 逐次和・1 回 downcast）と
//! **bit 完全一致**（`.claude/rules/coding-rust.md` 数値契約節・
//! `scan_parity.rs` と同型の契約）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test im2col_col2im_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Conv2dParams, Tensor, im2col_out_shape};

/// bit 完全一致（`value` が NaN の場合のみクラス一致）の判定ヘルパー
/// （`constant_pad_parity.rs::assert_bits_eq` と同型の複製。クレート内
/// で `pub(crate)` 共有できないテストバイナリ間の理由は同ファイルの
/// doc を参照）。
fn assert_bits_eq(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: 要素数が一致しない");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.is_nan() || e.is_nan() {
            assert!(
                a.is_nan() && e.is_nan(),
                "{label}: 要素 {i} が NaN クラス一致しない（actual={a}, expected={e}）"
            );
        } else {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}: 要素 {i} が bit 一致しない（actual={a:?}, expected={e:?}）"
            );
        }
    }
}

fn params(
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Conv2dParams {
    Conv2dParams::new(kernel_size, stride, padding, dilation, groups).expect("valid Conv2dParams")
}

fn random_tensor(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    Tensor::new(Xorshift64Star::new(seed).fill_vec(numel), shape).expect("valid tensor")
}

/// `im2col` の CPU-CUDA parity（bit 完全一致）を確認する共通ヘルパー。
fn assert_im2col_parity(
    cpu: &CpuBackendOps,
    cuda: &CudaBackendOps,
    input: &Tensor<f32>,
    p: &Conv2dParams,
    label: &str,
) {
    let cpu_out = cpu
        .im2col(input, p)
        .expect("cpu im2col always succeeds for valid input");
    let cuda_out = cuda
        .im2col(input, p)
        .expect("cuda im2col must succeed on real device");

    assert_eq!(
        cpu_out.shape(),
        cuda_out.shape(),
        "im2col({label}): shape 不一致"
    );
    assert_bits_eq(
        &format!("im2col({label}): CPU と CUDA"),
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );

    // run-to-run で bit 同一（決定的カーネル）。
    let cuda_out2 = cuda.im2col(input, p).expect("cuda im2col rerun");
    assert_bits_eq(
        &format!("im2col({label}): run-to-run"),
        cuda_out2.as_slice().expect("contiguous"),
        cuda_out.as_slice().expect("contiguous"),
    );
}

/// `col2im` の CPU-CUDA parity（bit 完全一致）を確認する共通ヘルパー。
/// `d_col` は `im2col_out_shape` が定める形状の任意のテンソル（乱数で
/// 埋めた「勾配」相当）でよい——col2im 自身の入力位置定常な走査・
/// 縮約順序の正しさを検証する目的であり、im2col との往復整合性は
/// 別観点（`facade/tests/conv2d_backend_parity.rs` の backward テスト
/// が担う）。
fn assert_col2im_parity(
    cpu: &CpuBackendOps,
    cuda: &CudaBackendOps,
    d_col: &Tensor<f32>,
    input_shape: &[usize],
    p: &Conv2dParams,
    label: &str,
) {
    let cpu_out = cpu
        .col2im(d_col, input_shape, p)
        .expect("cpu col2im always succeeds for valid input");
    let cuda_out = cuda
        .col2im(d_col, input_shape, p)
        .expect("cuda col2im must succeed on real device");

    assert_eq!(
        cpu_out.shape(),
        cuda_out.shape(),
        "col2im({label}): shape 不一致"
    );
    assert_bits_eq(
        &format!("col2im({label}): CPU と CUDA"),
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );

    let cuda_out2 = cuda
        .col2im(d_col, input_shape, p)
        .expect("cuda col2im rerun");
    assert_bits_eq(
        &format!("col2im({label}): run-to-run"),
        cuda_out2.as_slice().expect("contiguous"),
        cuda_out.as_slice().expect("contiguous"),
    );
}

/// 1 ケース分の `(in_shape, kernel, stride, padding, dilation, groups)`
/// をまとめる（`constant_pad_parity.rs::ShapePads` と同じ、テスト
/// 本体の可読性のための局所型）。
struct Case {
    label: &'static str,
    in_shape: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
}

/// README（`docs/perf/logs/cuda-conv2d-1766/README.md`）・設計 doc
/// （`docs/conv-ops-design.md` §1766）が明記する形状網羅の本体:
/// 重なり窓・dilation・groups／depthwise・padding のみの窓・座標
/// アンダーフロー形状を含む。
const CASES: &[Case] = &[
    Case {
        label: "basic no pad",
        in_shape: [1, 2, 5, 5],
        kernel: [3, 3],
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
    },
    Case {
        label: "overlapping windows (stride < kernel extent)",
        in_shape: [1, 1, 6, 6],
        kernel: [3, 3],
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
        groups: 1,
    },
    Case {
        label: "dilation",
        in_shape: [1, 1, 9, 9],
        kernel: [3, 3],
        stride: [1, 1],
        padding: [2, 2],
        dilation: [2, 2],
        groups: 1,
    },
    Case {
        label: "groups depthwise",
        in_shape: [1, 4, 6, 6],
        kernel: [3, 3],
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
        groups: 4,
    },
    Case {
        label: "groups (non-depthwise, 2 groups, batch>1)",
        in_shape: [2, 6, 5, 5],
        kernel: [3, 3],
        stride: [2, 2],
        padding: [1, 1],
        dilation: [1, 1],
        groups: 2,
    },
    Case {
        label: "padding-only window (window fully out of range)",
        in_shape: [1, 1, 2, 2],
        kernel: [3, 3],
        stride: [1, 1],
        padding: [5, 5],
        dilation: [1, 1],
        groups: 1,
    },
    Case {
        label: "coordinate underflow (large padding, kernel=1)",
        in_shape: [1, 1, 3, 3],
        kernel: [1, 1],
        stride: [1, 1],
        padding: [4, 4],
        dilation: [1, 1],
        groups: 1,
    },
    Case {
        label: "stride > kernel extent (gaps between windows)",
        in_shape: [1, 1, 7, 7],
        kernel: [2, 2],
        stride: [3, 3],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
    },
    Case {
        label: "asymmetric H/W padding+dilation+groups combined",
        in_shape: [1, 4, 5, 7],
        kernel: [3, 2],
        stride: [2, 1],
        padding: [3, 1],
        dilation: [2, 3],
        groups: 2,
    },
    // --- 1d 形状（イシュー #1767。`H=1`・`kh=1`・`sh=1`・`ph=0`・
    // `dh=1` に固定した「Conv1d を Conv2d の特化として実装する」形状。
    // `Var::conv1d`〈#1765〉が内部で `[N, Cin, 1, L]`／
    // `[Cout, Cin_g, 1, k]` へ reshape してからこの CUDA `im2col`／
    // `col2im` を呼ぶため、`H` 軸を通常の 2d 形状と同じ形状パラメータ
    // 検査経路（`LaunchShape::derive`・`conv_out_len` 等）へそのまま
    // 通すことを確認する）。
    Case {
        label: "1d basic no pad",
        in_shape: [1, 2, 1, 9],
        kernel: [1, 3],
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
    },
    Case {
        label: "1d overlapping windows (pad)",
        in_shape: [1, 1, 1, 6],
        kernel: [1, 3],
        stride: [1, 1],
        padding: [0, 1],
        dilation: [1, 1],
        groups: 1,
    },
    Case {
        label: "1d dilation",
        in_shape: [1, 1, 1, 9],
        kernel: [1, 3],
        stride: [1, 1],
        padding: [0, 2],
        dilation: [1, 2],
        groups: 1,
    },
    Case {
        label: "1d groups depthwise",
        in_shape: [1, 4, 1, 6],
        kernel: [1, 3],
        stride: [1, 1],
        padding: [0, 1],
        dilation: [1, 1],
        groups: 4,
    },
    Case {
        label: "1d groups (2 groups, batch>1)",
        in_shape: [2, 6, 1, 5],
        kernel: [1, 3],
        stride: [1, 2],
        padding: [0, 1],
        dilation: [1, 1],
        groups: 2,
    },
    Case {
        label: "1d stride > kernel extent",
        in_shape: [1, 1, 1, 7],
        kernel: [1, 2],
        stride: [1, 3],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
    },
];

fn run_case(cpu: &CpuBackendOps, cuda: &CudaBackendOps, seed: u64, case: &Case) {
    let p = params(
        case.kernel,
        case.stride,
        case.padding,
        case.dilation,
        case.groups,
    );
    let input = random_tensor(seed, &case.in_shape);
    assert_im2col_parity(cpu, cuda, &input, &p, case.label);

    let col_shape =
        im2col_out_shape(&case.in_shape, &p).expect("im2col_out_shape: case は有効な形状のはず");
    let d_col = random_tensor(seed.wrapping_add(1), &col_shape);
    assert_col2im_parity(cpu, cuda, &d_col, &case.in_shape, &p, case.label);
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA 不在なら
/// `BackendError::CudaUnavailable` を確認して早期 return する
/// （`constant_pad_parity.rs::constant_pad_parity_smoke_env_adaptive`
/// と同じ分岐パターン）。デバイス初期化より前に返る shape 検査経路
/// （`im2col_out_shape` の再検査。`col2im` の `d_col` 形状不一致）は
/// GPU 有無に依らず検証する。
#[test]
fn im2col_col2im_parity_smoke_env_adaptive() {
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();

    let p = params([3, 3], [1, 1], [1, 1], [1, 1], 1);
    let input = random_tensor(1001, &[1, 2, 5, 5]);

    match cuda.im2col(&input, &p) {
        Ok(_) => {
            // 代表ケースをいくつか（形状網羅の全体は #[ignore] 側）。
            let mut seed = 2001u64;
            for case in &CASES[..4] {
                seed += 11;
                run_case(&cpu, &cuda, seed, case);
            }

            // 1d 形状（`H=1`・`kh=1`）代表 1 件も通常 CI で確認する
            // （イシュー #1767。`Var::conv1d` が reshape する
            // `[N, Cin, 1, L]`／`[Cout, Cin_g, 1, k]` 形状が im2col／
            // col2im の shape 検査経路〈`LaunchShape::derive` 等〉を
            // 通ることの最小確認。網羅は `#[ignore]` 側 `CASES` 全体）。
            let case_1d = CASES
                .iter()
                .find(|c| c.label == "1d basic no pad")
                .expect("1d basic no pad case must exist");
            run_case(&cpu, &cuda, 2101, case_1d);

            // N=0（空入力・空出力の早期リターン経路。driver 非接触の
            // ops.rs 分岐が実際に到達することを実機で確認する）。
            let p_n0 = params([3, 3], [1, 1], [1, 1], [1, 1], 1);
            let input_n0 =
                Tensor::new(Vec::<f32>::new(), &[0usize, 2, 5, 5]).expect("valid tensor");
            let cpu_n0 = cpu
                .im2col(&input_n0, &p_n0)
                .expect("cpu im2col(N=0) always succeeds");
            let cuda_n0 = cuda
                .im2col(&input_n0, &p_n0)
                .expect("cuda im2col(N=0) must succeed on real device");
            assert_eq!(cpu_n0.shape(), cuda_n0.shape());
            assert!(cpu_n0.shape().contains(&0));

            let col_shape_n0 =
                im2col_out_shape(&[0usize, 2, 5, 5], &p_n0).expect("valid im2col_out_shape");
            let d_col_n0 =
                Tensor::new(Vec::<f32>::new(), &col_shape_n0).expect("valid empty d_col");
            let cpu_back_n0 = cpu
                .col2im(&d_col_n0, &[0usize, 2, 5, 5], &p_n0)
                .expect("cpu col2im(N=0) always succeeds");
            let cuda_back_n0 = cuda
                .col2im(&d_col_n0, &[0usize, 2, 5, 5], &p_n0)
                .expect("cuda col2im(N=0) must succeed on real device");
            assert_eq!(cpu_back_n0.shape(), cuda_back_n0.shape());

            // shape 不一致（rank mismatch）は `BackendError::
            // ShapeMismatch` を返す（実装側の再検査。`.claude/rules/
            // security.md` A08）。
            let err = cuda
                .im2col(&random_tensor(1, &[1, 2, 5]), &p)
                .expect_err("rank mismatch must be rejected");
            assert!(matches!(err, BackendError::ShapeMismatch(_)));

            // col2im: d_col の shape が `im2col_out_shape` の想定と
            // 不一致なら `ShapeMismatch`。
            let wrong_d_col = random_tensor(2, &[1, 1, 1, 1]);
            let err = cuda
                .col2im(&wrong_d_col, &[1, 2, 5, 5], &p)
                .expect_err("d_col shape mismatch must be rejected");
            assert!(matches!(err, BackendError::ShapeMismatch(_)));
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");

            // shape 検査はデバイス初期化より前に走るため CUDA 非搭載
            // 環境でも検証できる（CPU の返す値と一致することも確認）。
            let bad_input = random_tensor(1, &[1, 2, 5]); // rank mismatch
            let cpu_err = cpu
                .im2col(&bad_input, &p)
                .expect_err("cpu must reject rank mismatch");
            let cuda_err = cuda
                .im2col(&bad_input, &p)
                .expect_err("rank mismatch must be rejected even without CUDA");
            assert!(matches!(cpu_err, BackendError::ShapeMismatch(_)));
            assert!(matches!(cuda_err, BackendError::ShapeMismatch(_)));
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::im2col: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体。README・設計 doc §1766 の
/// 事前登録判定規則に対応する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn im2col_col2im_matches_cpu_across_shapes() {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let mut seed = 30_000u64;
    for case in CASES {
        seed += 13;
        run_case(&cpu, &cuda, seed, case);
    }
}

/// `IM2COL_BLOCK_DIM=256` の grid 境界をまたぐ numel（`out_shape`／
/// `input_shape` 双方の numel を 256 の非倍数にし、最終ブロックの
/// 一部スレッドのみが有効という境界検査〈REQ-8〉を強く踏む形状）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn im2col_col2im_crosses_256_block_boundary() {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    // kernel=1×1・stride=1・padding=0・dilation=1・groups=1 なら
    // im2col の `out_shape=[N, 1, Cin, H*W]`・col2im の `input_shape`
    // の numel がどちらも `N*Cin*H*W` に一致するため、`W` を選ぶだけで
    // 256 の非倍数（257・300 いずれもブロック境界をまたぐ）に調整
    // できる。
    for &w in &[257usize, 300, 513] {
        let p = params([1, 1], [1, 1], [0, 0], [1, 1], 1);
        let in_shape = [1usize, 1, 1, w];
        let input = random_tensor(40_000 + w as u64, &in_shape);
        assert_im2col_parity(&cpu, &cuda, &input, &p, &format!("boundary W={w}"));

        let col_shape = im2col_out_shape(&in_shape, &p).expect("valid im2col_out_shape");
        assert_eq!(
            col_shape.iter().product::<usize>(),
            w,
            "numel が想定と不一致"
        );
        let d_col = random_tensor(50_000 + w as u64, &col_shape);
        assert_col2im_parity(
            &cpu,
            &cuda,
            &d_col,
            &in_shape,
            &p,
            &format!("boundary W={w}"),
        );
    }
}

/// NaN／±inf／−0.0 の通過（im2col は純粋コピーのためそのまま
/// 出力へ現れる。col2im は `f64`／`double` アキュムレータへの加算に
/// よりクラスが変化しうるため NaN のみクラス一致で判定する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn im2col_col2im_special_values_pass_through() {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let p = params([2, 2], [1, 1], [1, 1], [1, 1], 1);
    let input = Tensor::new(
        vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            1.5,
            -2.5,
            f32::NAN,
            3.0,
        ],
        &[1, 1, 3, 3],
    )
    .expect("valid tensor");
    assert_im2col_parity(&cpu, &cuda, &input, &p, "special values");

    let col_shape = im2col_out_shape(&[1, 1, 3, 3], &p).expect("valid im2col_out_shape");
    let numel: usize = col_shape.iter().product();
    // d_col 側にも NaN／±inf／−0.0 を混在させ、col2im の逐次加算が
    // それらを含む重なり窓（padding=1・kernel=2）でも CPU と bit
    // 一致することを確認する。
    let mut vals = Vec::with_capacity(numel);
    for i in 0..numel {
        vals.push(match i % 6 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => -0.0,
            4 => 0.0,
            _ => (i as f32) * 0.5 - 3.0,
        });
    }
    let d_col = Tensor::new(vals, &col_shape).expect("valid tensor");
    assert_col2im_parity(&cpu, &cuda, &d_col, &[1, 1, 3, 3], &p, "special values");
}

/// 非 contiguous な input（narrow 後の view）を渡しても contiguous 化
/// 後の結果と一致する（`gemm_transposed_parity.rs::w_narrowed` と同じ
/// narrow パターン。`im2col.rs::CudaIm2col::run_im2col_f32` は呼び出し
/// 元 `ops.rs` で `input.contiguous()` してから渡すため、非
/// contiguous でも正しい結果になることを実機で確認する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn im2col_matches_cpu_for_non_contiguous_input() {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    // [N, Cin, H, W+2] を確保し最後の軸（W）を narrow して
    // [N, Cin, H, W] の非 contiguous view を作る。
    let base = random_tensor(60_001, &[1, 2, 5, 7]);
    let narrowed = base.narrow(3, 1, 5).expect("valid narrow");
    assert!(!narrowed.is_contiguous(), "narrow 後は非 contiguous のはず");
    assert_eq!(narrowed.shape(), &[1, 2, 5, 5]);

    let p = params([3, 3], [1, 1], [1, 1], [1, 1], 1);
    assert_im2col_parity(&cpu, &cuda, &narrowed, &p, "narrowed non-contiguous input");
}
