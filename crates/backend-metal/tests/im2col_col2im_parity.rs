//! イシュー #1768: `BackendOps::im2col`／`col2im`（Conv2d の im2col+GEMM
//! 展開・畳み戻し。設計 `docs/conv-ops-design.md`）の CPU-Metal 数値
//! 一致検証。`crates/backend-cuda/tests/im2col_col2im_parity.rs`
//! （イシュー #1766）の Metal 対応版。
//!
//! `scan_parity.rs`（#1740）と同じ構成方針: 本ファイル全体を macOS 限定
//! （`objc2` 系 FFI に触れる `MetalBackendOps` を使うため）とし、
//! 全テストに Apple Silicon 実機必須の `#[ignore]` を付ける。
//!
//! **契約**: `im2col` は算術を含まない純粋なコピー演算のため
//! **bit 完全一致**。`col2im` は binary64 ソフトウェアエミュレーション
//! （`shaders/im2col.metal::im2col_f64_*`）アキュムレータの逐次加算・
//! 1 回 `f32` downcast で CPU 参照実装（`backend-cpu::im2col::col2im`
//! の `f64` 逐次和・1 回 downcast）と **bit 完全一致**
//! （`.claude/rules/coding-rust.md` 数値契約節）。
//!
//! 実行コマンド（Apple Silicon 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test im2col_col2im_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::{BackendOps, Conv2dParams, Tensor, im2col_out_shape};

/// bit 完全一致（`value` が NaN の場合のみクラス一致）の判定ヘルパー
/// （CUDA 版 `im2col_col2im_parity.rs::assert_bits_eq` と同型）。
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

fn assert_im2col_parity(
    cpu: &CpuBackendOps,
    metal: &MetalBackendOps,
    input: &Tensor<f32>,
    p: &Conv2dParams,
    label: &str,
) {
    let cpu_out = cpu
        .im2col(input, p)
        .expect("cpu im2col always succeeds for valid input");
    let metal_out = metal
        .im2col(input, p)
        .expect("metal im2col must succeed on real device");

    assert_eq!(
        cpu_out.shape(),
        metal_out.shape(),
        "im2col({label}): shape 不一致"
    );
    assert_bits_eq(
        &format!("im2col({label}): CPU と Metal"),
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );

    // run-to-run で bit 同一（決定的カーネル）。
    let metal_out2 = metal.im2col(input, p).expect("metal im2col rerun");
    assert_bits_eq(
        &format!("im2col({label}): run-to-run"),
        metal_out2.as_slice().expect("contiguous"),
        metal_out.as_slice().expect("contiguous"),
    );
}

fn assert_col2im_parity(
    cpu: &CpuBackendOps,
    metal: &MetalBackendOps,
    d_col: &Tensor<f32>,
    input_shape: &[usize],
    p: &Conv2dParams,
    label: &str,
) {
    let cpu_out = cpu
        .col2im(d_col, input_shape, p)
        .expect("cpu col2im always succeeds for valid input");
    let metal_out = metal
        .col2im(d_col, input_shape, p)
        .expect("metal col2im must succeed on real device");

    assert_eq!(
        cpu_out.shape(),
        metal_out.shape(),
        "col2im({label}): shape 不一致"
    );
    assert_bits_eq(
        &format!("col2im({label}): CPU と Metal"),
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );

    let metal_out2 = metal
        .col2im(d_col, input_shape, p)
        .expect("metal col2im rerun");
    assert_bits_eq(
        &format!("col2im({label}): run-to-run"),
        metal_out2.as_slice().expect("contiguous"),
        metal_out.as_slice().expect("contiguous"),
    );
}

struct Case {
    label: &'static str,
    in_shape: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
}

/// CUDA `im2col_col2im_parity.rs::CASES`（イシュー #1766）と同一の
/// 形状網羅: 重なり窓・dilation・groups／depthwise・padding のみの窓・
/// 座標アンダーフロー形状・1d 形状を含む。
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

fn run_case(cpu: &CpuBackendOps, metal: &MetalBackendOps, seed: u64, case: &Case) {
    let p = params(
        case.kernel,
        case.stride,
        case.padding,
        case.dilation,
        case.groups,
    );
    let input = random_tensor(seed, &case.in_shape);
    assert_im2col_parity(cpu, metal, &input, &p, case.label);

    let col_shape =
        im2col_out_shape(&case.in_shape, &p).expect("im2col_out_shape: case は有効な形状のはず");
    let d_col = random_tensor(seed.wrapping_add(1), &col_shape);
    assert_col2im_parity(cpu, metal, &d_col, &case.in_shape, &p, case.label);
}

/// 実機必須の形状網羅（受け入れ条件の本体）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn im2col_col2im_matches_cpu_across_shapes() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let mut seed = 30_000u64;
    for case in CASES {
        seed += 13;
        run_case(&cpu, &metal, seed, case);
    }
}

/// `IM2COL_THREADGROUP_WIDTH=256` の grid 境界をまたぐ numel（CUDA
/// 版 `im2col_col2im_crosses_256_block_boundary` と同型）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn im2col_col2im_crosses_256_threadgroup_boundary() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    for &w in &[257usize, 300, 513] {
        let p = params([1, 1], [1, 1], [0, 0], [1, 1], 1);
        let in_shape = [1usize, 1, 1, w];
        let input = random_tensor(40_000 + w as u64, &in_shape);
        assert_im2col_parity(&cpu, &metal, &input, &p, &format!("boundary W={w}"));

        let col_shape = im2col_out_shape(&in_shape, &p).expect("valid im2col_out_shape");
        assert_eq!(
            col_shape.iter().product::<usize>(),
            w,
            "numel が想定と不一致"
        );
        let d_col = random_tensor(50_000 + w as u64, &col_shape);
        assert_col2im_parity(
            &cpu,
            &metal,
            &d_col,
            &in_shape,
            &p,
            &format!("boundary W={w}"),
        );
    }
}

/// NaN／±inf／−0.0 の通過（im2col は純粋コピーのためそのまま出力へ
/// 現れる。col2im は binary64 ソフトウェアエミュレーションアキュムレータ
/// への加算によりクラスが変化しうるため NaN のみクラス一致で判定する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn im2col_col2im_special_values_pass_through() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

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
    assert_im2col_parity(&cpu, &metal, &input, &p, "special values");

    let col_shape = im2col_out_shape(&[1, 1, 3, 3], &p).expect("valid im2col_out_shape");
    let numel: usize = col_shape.iter().product();
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
    assert_col2im_parity(&cpu, &metal, &d_col, &[1, 1, 3, 3], &p, "special values");
}

/// 非 contiguous な input（narrow 後の view）を渡しても contiguous 化
/// 後の結果と一致する（CUDA 版 `im2col_matches_cpu_for_non_contiguous_
/// input` と同型。`im2col.rs::MetalIm2col::run_im2col_f32` は呼び出し
/// 元 `ops.rs` で `input.contiguous()` してから渡すため、非
/// contiguous でも正しい結果になることを実機で確認する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn im2col_matches_cpu_for_non_contiguous_input() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let base = random_tensor(60_001, &[1, 2, 5, 7]);
    let narrowed = base.narrow(3, 1, 5).expect("valid narrow");
    assert!(!narrowed.is_contiguous(), "narrow 後は非 contiguous のはず");
    assert_eq!(narrowed.shape(), &[1, 2, 5, 5]);

    let p = params([3, 3], [1, 1], [1, 1], [1, 1], 1);
    assert_im2col_parity(&cpu, &metal, &narrowed, &p, "narrowed non-contiguous input");
}

/// N=0（空入力・空出力の早期リターン経路）の実機確認。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn im2col_col2im_zero_batch() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let p = params([3, 3], [1, 1], [1, 1], [1, 1], 1);
    let input = Tensor::new(Vec::<f32>::new(), &[0usize, 2, 5, 5]).expect("valid tensor");
    let cpu_out = cpu.im2col(&input, &p).expect("cpu im2col(N=0) succeeds");
    let metal_out = metal
        .im2col(&input, &p)
        .expect("metal im2col(N=0) must succeed on real device");
    assert_eq!(cpu_out.shape(), metal_out.shape());
    assert!(cpu_out.shape().contains(&0));

    let col_shape = im2col_out_shape(&[0usize, 2, 5, 5], &p).expect("valid im2col_out_shape");
    let d_col = Tensor::new(Vec::<f32>::new(), &col_shape).expect("valid empty d_col");
    let cpu_back = cpu
        .col2im(&d_col, &[0usize, 2, 5, 5], &p)
        .expect("cpu col2im(N=0) succeeds");
    let metal_back = metal
        .col2im(&d_col, &[0usize, 2, 5, 5], &p)
        .expect("metal col2im(N=0) must succeed on real device");
    assert_eq!(cpu_back.shape(), metal_back.shape());
}
