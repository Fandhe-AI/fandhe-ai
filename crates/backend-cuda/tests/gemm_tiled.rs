//! tiled GEMM（`CudaGemm::run_tiled_*`）の環境適応型テスト＋実機必須テスト。
//!
//! `tests/gemm_naive.rs` の構成（CUDA 搭載・非搭載どちらの環境でも green に
//! なる環境適応型テスト＋受け入れ条件そのものである数値一致・性能比較
//! テストを `#[ignore]` で分離）をそのまま踏襲する
//! （`.claude/rules/coding-rust.md` の実機依存テスト分離規約）。

use backend_cuda::{CudaDevice, CudaError, CudaGemm};

/// `CudaGemm::new` は tiled カーネル追加後も CUDA 非搭載環境で panic せず
/// 型付きエラーを返す契約を維持している（`tests/gemm_naive.rs` の同名
/// テストと同じ主張だが、`new` が今回コンパイルするカーネル数が 2→4 に
/// 増えたことによる回帰がないことを tiled 側の入口としても確認する）。
#[test]
fn new_compiles_tiled_kernels_or_returns_typed_error_without_panicking() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => {
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    match CudaGemm::new(&device) {
        Ok(_gemm) => {
            // CUDA 搭載環境: naive/tiled 計 4 カーネルのコンパイルが成功した。
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
        }
        Err(other) => panic!("unexpected CudaError variant from CudaGemm::new: {other}"),
    }
}

/// 複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。
/// `tests/gemm_naive.rs` と同一の判定式（許容誤差は単独緩和しない。
/// `.claude/rules/coding-rust.md`）。
const RELATIVE_TOLERANCE: f64 = 1e-3;
const ABSOLUTE_RESCUE_THRESHOLD: f64 = 1e-5;

/// `a`（GPU 出力）と `b`（CPU 参照）を要素ごとに複合判定し、
/// 不一致セル数を返す（0 なら PASS）。`tests/gemm_naive.rs` の同名関数と
/// 同一実装（テストファイル間で共有クレートを新設するほどの重複ではない
/// ため、意図的にそのまま複製する）。
fn count_composite_mismatches(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len(), "compare: length mismatch");
    a.iter()
        .zip(b.iter())
        .filter(|&(&x, &y)| {
            let xf = x as f64;
            let yf = y as f64;
            let diff = (xf - yf).abs();
            let scale = xf.abs().max(yf.abs()).max(1e-12);
            let rel = diff / scale;
            rel >= RELATIVE_TOLERANCE && diff >= ABSOLUTE_RESCUE_THRESHOLD
        })
        .count()
}

/// 決定的シードで A・B（f32）を生成し、CPU 参照実装（`gemm_naive`。
/// `mul_add` FMA 契約）と GPU tiled カーネルの出力を複合判定で照合する。
fn assert_tiled_f32_matches_cpu_reference(gemm: &CudaGemm, seed: u64, m: u32, n: u32, k: u32) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
    backend_cpu::gemm_naive(&a, &b, &mut c_ref, m as usize, n as usize, k as usize)
        .expect("backend-cpu gemm_naive shape validation must pass for well-formed test input");

    let c_gpu = gemm
        .run_tiled_f32(&a, &b, m, n, k)
        .expect("CudaGemm::run_tiled_f32 must succeed on CUDA-equipped test runner");

    let mismatches = count_composite_mismatches(&c_gpu, &c_ref);
    assert_eq!(
        mismatches,
        0,
        "tiled f32 GEMM CPU/GPU mismatch: {mismatches}/{} cells failed composite tolerance \
         (rel<{RELATIVE_TOLERANCE} or abs<{ABSOLUTE_RESCUE_THRESHOLD}), shape m={m} n={n} k={k}",
        c_ref.len()
    );
}

/// 実機（DGX Spark GB10 等）必須の数値一致テスト。受け入れ条件の一部。
///
/// 形状ケースは `kernels::TILE`（32）の境界を踏むケースを含める:
/// 128^3・512^3（TILE の整数倍）・64x96x128（非正方・TILE の整数倍）・
/// 1x1x1（num_tiles=1 の最小ケース）・17x23x19・33x31x65（いずれも TILE の
/// 非整数倍で、末尾タイルのゼロパディング分岐（`kernels.rs` の三項ガード）
/// を実際に踏む）。CI self-hosted runner は CUDA toolkit 非搭載のため通常
/// 実行ではスキップされる（実行導線の整備は #36 のスコープ）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn tiled_f32_matches_cpu_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive/tiled kernel compilation must succeed");

    let cases: &[(u32, u32, u32)] = &[
        (128, 128, 128),
        (512, 512, 512),
        (64, 96, 128),
        (1, 1, 1),
        (17, 23, 19),
        (33, 31, 65),
    ];
    for (idx, &(m, n, k)) in cases.iter().enumerate() {
        assert_tiled_f32_matches_cpu_reference(&gemm, 2000 + idx as u64, m, n, k);
    }
}

/// K 大のストレスケース（M=N=256, K=4096）。PoC-v2-3 の数値一致確認節・
/// PoC-v2-5 の FMA 契約統一ケースに対応する。tiled カーネルはタイル単位で
/// 部分和を積み上げる（`kernels.rs` の `for t in 0..num_tiles` ループ）ため、
/// naive カーネル（K 方向を 1 要素ずつ逐次加算）と加算順序が異なりうる。
/// この形状は加算順序の違いに由来する丸め誤差が複合判定を破らないことを
/// 確認する回帰ケースである。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn tiled_f32_matches_cpu_reference_k_stress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive/tiled kernel compilation must succeed");

    assert_tiled_f32_matches_cpu_reference(&gemm, 8888, 256, 256, 4096);
}

/// `k == 0`（`num_tiles == 0` 経路。`kernels.rs` の
/// `(k > 0) ? (k - 1) / TILE + 1 : 0` 参照）で C が全 0 になることを確認する。
/// `validate_gemm_dims` は `k == 0` を形状不整合として拒否しない
/// （`a_len == m*k == 0` かつ `b_len == k*n == 0` であれば長さは一致する）
/// ため、空スライスを渡して起動する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn tiled_f32_zero_k_returns_all_zero() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive/tiled kernel compilation must succeed");

    let (m, n, k) = (4u32, 4u32, 0u32);
    let c = gemm
        .run_tiled_f32(&[], &[], m, n, k)
        .expect("k==0 must be a valid no-accumulation shape, not a launch error");
    assert_eq!(c.len(), (m as usize) * (n as usize));
    assert!(c.iter().all(|&v| v == 0.0), "k==0 output must be all zero");
}

/// m==0／n==0（`backend-cpu::gemm_naive` と同じ no-op 形状）で
/// `run_tiled_f32` を呼んでも CUDA 起動自体が発生せず（`gemm.rs` の
/// `run_f32_kernel` 早期 return。naive/tiled 共通ヘルパー）、空の結果を
/// 返すことを実機で確認する（`tests/gemm_naive.rs` の同名テストの tiled 版）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn tiled_f32_zero_dim_shape_returns_empty_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive/tiled kernel compilation must succeed");

    let c = gemm
        .run_tiled_f32(&[], &[1.0, 2.0, 3.0, 4.0], 0, 4, 1)
        .expect("m==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());

    let c = gemm
        .run_tiled_f32(&[1.0, 2.0], &[], 2, 0, 1)
        .expect("n==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());
}

/// tiled f16 GEMM（実機必須）。`tests/gemm_naive.rs` の f16 テストと同じ
/// 判断（f16 向け tolerance の妥当性確認は #36 へ委ね、本テストは
/// panic せず妥当な形状の出力を返すことまでを確認する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn tiled_f16_runs_and_returns_expected_shape() {
    use half::f16;

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive/tiled kernel compilation must succeed");

    let mut rng = bench_harness::rng::Xorshift64Star::new(7777);
    let (m, n, k) = (64u32, 64u32, 64u32);
    let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
    let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));

    let c_gpu = gemm
        .run_tiled_f16(&a, &b, m, n, k)
        .expect("CudaGemm::run_tiled_f16 must succeed on CUDA-equipped test runner");

    assert_eq!(c_gpu.len(), (m as usize) * (n as usize));
}

/// 性能比較テスト（受け入れ条件の本体）: 「tiled GEMM が naive 比で
/// PoC-v2-3 相当の性能改善を示す」ことを実機で検証する。
///
/// 同一入力（M=N=K=4096、決定的シード）で naive と tiled の `run_*_f32` を
/// それぞれ warmup 2 回のあと 5 回計測し、`bench_harness::median_q1_q3`
/// で中央値を求めて比較する
/// （`.claude/rules/coding-rust.md` の「ベンチは 5 回計測の中央値を採用」）。
///
/// 判定閾値: PoC-v2-3 実測（README 計測結果節）の 4096 比 tiled/naive =
/// 1.8316/1.2576 TFLOPS ≒ 1.46 倍に対し、ホスト側転送込み計測（本テストは
/// `run_*` の呼び出し全体を計測しており、PoC の GPU 実行時間のみの計測とは
/// 異なる）による希釈と実行環境揺らぎを見込んだ保守値 **1.1 倍以上**とする
/// （4096 ではカーネル時間がホスト-デバイス転送時間を支配するため希釈は
/// 限定的と見積もる）。この閾値は受け入れ条件の判定基準であり、
/// 変更にはユーザー承認が必要（`.claude/rules/coding-rust.md`
/// 「バックエンド間数値一致テストの許容誤差を単独で緩和しない」と同じ
/// 精神をベンチ閾値にも適用する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn tiled_f32_outperforms_naive_at_4096() {
    const WARMUP: usize = 2;
    const SAMPLES: usize = 5;
    const MIN_SPEEDUP: f64 = 1.1;

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive/tiled kernel compilation must succeed");

    let (m, n, k) = (4096u32, 4096u32, 4096u32);
    let mut rng = bench_harness::rng::Xorshift64Star::new(31415);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    // clippy::type_complexity 対策の型エイリアス（naive/tiled 双方の
    // `run_*_f32` シグネチャを同一クロージャ引数として受けるための共通型）。
    type GemmRunFn<'a> = dyn Fn(&[f32], &[f32], u32, u32, u32) -> Result<Vec<f32>, CudaError> + 'a;

    let measure = |run: &GemmRunFn<'_>| -> f64 {
        for _ in 0..WARMUP {
            run(&a, &b, m, n, k).expect("warmup run must succeed on CUDA-equipped test runner");
        }
        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let start = std::time::Instant::now();
            run(&a, &b, m, n, k).expect("measured run must succeed on CUDA-equipped test runner");
            samples.push(start.elapsed().as_secs_f64());
        }
        bench_harness::median_q1_q3(&samples)
            .expect("5 non-NaN samples must yield quartiles")
            .median
    };

    let naive_median = measure(&|a, b, m, n, k| gemm.run_naive_f32(a, b, m, n, k));
    let tiled_median = measure(&|a, b, m, n, k| gemm.run_tiled_f32(a, b, m, n, k));

    let speedup = naive_median / tiled_median;
    assert!(
        speedup >= MIN_SPEEDUP,
        "tiled GEMM must outperform naive by at least {MIN_SPEEDUP}x at M=N=K=4096 \
         (PoC-v2-3 realized ~1.46x); measured speedup={speedup:.3}x \
         (naive_median={naive_median:.6}s, tiled_median={tiled_median:.6}s)"
    );
}
