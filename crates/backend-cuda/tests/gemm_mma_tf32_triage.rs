//! イシュー #1122 診断専用テスト: `CudaMmaTf32Gemm::run_tf32`（生
//! `mma.sync`(m16n8k8) 経路。`tests/gemm_mma_tf32.rs`）・
//! `CudaGemm::run_wmma_tf32`（staged 経路。`tests/mma_tf32_vs_wmma_tf32_staged.rs`）
//! の両方で GB10 実機の `assert_parity`（厳密ゼロ fail）が FAIL する原因を、
//! 「TF32 丸め誤差」と「機能欠陥（フラグメントマッピング等）」のどちらに
//! 起因するかで切り分けるための診断テスト。
//!
//! **本ファイルは受け入れ判定には使わない**（`.claude/rules/coding-rust.md`
//! 「TF32/f16 Tensor Core 経路の parity テスト判定方式」の対象外）。
//! `#[ignore]` 実機依存の `--nocapture` 前提のダンプ出力のみを提供し、
//! 既存の tolerance 定数・カーネル・既存テストは一切変更しない。
//!
//! ## 切り分けロジック
//!
//! 両カーネルとも入力 A/B を `wmma::__float_to_tf32`（PTX `cvt.rna.tf32.f32`
//! 相当。仮数部下位 13 bit を round-to-nearest-away で切り捨てる）で丸めて
//! から積和する（`crates/backend-cuda/src/kernels_mma_tf32.rs:442,453`
//! `CONVERT_A_STAGE_GROUP`/`CONVERT_B_STAGE_GROUP` マクロ参照）。
//!
//! よって、ホスト側で A/B を同じ TF32 RNA 丸めへ事前に丸めてから CPU 参照
//! （`matmul_reference_fma`）を計算すれば、GPU 側が「TF32 丸め済みの入力に
//! 対して機能的に正しい積和をしている」場合、GPU 出力と
//! 「TF32 丸め済み入力の CPU 参照」との差は f32 累算順序差レベル
//! （相対誤差 1e-6 程度以下）に縮小するはずである。
//!
//! - (a) と比べて (b) の乖離が大幅に小さければ「TF32 丸め由来」
//!   （既存の複合判定 tolerance が TF32 丸め誤差を吸収しきれていないだけ）
//! - (b) でも大きな乖離が残るなら「機能欠陥」（丸め以外の要因）
//!
//! 加えて mma.sync 経路と staged 経路の直接比較 (d) と、fail 要素の
//! 座標・`row%16`/`col%8` ヒストグラム（mma.sync の m16n8k8 フラグメント
//! 境界への偏り検出用）を出力し、機能欠陥だった場合に原因箇所（フラグメント
//! マッピング等）を絞り込む手がかりにする。

use fandhe_ai_backend_cpu::{CompareReport, compare, matmul_reference_fma};
use fandhe_ai_backend_cuda::{CudaDevice, CudaGemm, CudaMmaTf32Gemm};

/// PTX `cvt.rna.tf32.f32` 相当のホスト側 TF32 丸め（round-to-nearest-away）。
///
/// TF32 は符号 1 bit・指数 8 bit・仮数 10 bit（f32 の仮数 23 bit のうち
/// 上位 10 bit のみ保持）。下位 13 bit を丸めて捨てる。away 方向丸め
/// （0 から遠い側への丸め。tie は繰り上げ）は「丸め位置に `0x1000`
/// （13 bit 目の 1）を加算してから下位 13 bit をマスクする」操作と等価
/// （通常丸め対象の桁上げが指数部へ伝播する場合も含めて整数加算のみで
/// 表現できる。`kernels_mma_tf32.rs` の `wmma::__float_to_tf32` が
/// CUDA 側で行う丸めと同じ結果になる）。
///
/// **前提（本テストの入力範囲に限定）**: 入力は `[-1, 1)` の一様乱数
/// （`bench_harness::rng::Xorshift64Star::fill_vec`）のみを想定し、
/// NaN・Inf・非正規化数は生成されないためこの関数では扱わない。
fn round_to_tf32_rna(x: f32) -> f32 {
    let bits = x.to_bits();
    // 仮数下位 13 bit の丸め: 加算によるオーバーフローが指数部へ桁上げ
    // されるのは符号なし整数加算の通常の挙動であり、IEEE 754 の指数
    // インクリメントと整合する（`wrapping_add` で明示: この範囲の入力
    // （有限・非 Inf 化）では実際にオーバーフローしない）。
    let rounded = bits.wrapping_add(0x1000) & !0x1FFFu32;
    f32::from_bits(rounded)
}

fn round_slice_to_tf32_rna(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&x| round_to_tf32_rna(x)).collect()
}

#[cfg(test)]
mod tf32_rounding_unit_tests {
    use super::round_to_tf32_rna;

    #[test]
    fn value_already_within_10bit_mantissa_is_unchanged() {
        // 1.5 = 1.1(b) * 2^0: 仮数部は上位 1 bit のみ使用（下位 22 bit は
        // すでに 0）ため、TF32（仮数 10 bit）へ丸めても値は変わらない。
        let x = 1.5f32;
        assert_eq!(round_to_tf32_rna(x), x);
    }

    #[test]
    fn zero_is_unchanged() {
        assert_eq!(round_to_tf32_rna(0.0f32), 0.0f32);
    }

    #[test]
    fn midpoint_rounds_away_from_zero() {
        // 仮数下位 13 bit がちょうど中間点（0x1000）の値は away 方向
        // （繰り上げ）へ丸まる。1.0 のビットパターンに 0x1000 だけ立てた
        // 値を作り、丸め後は次に大きい TF32 表現可能値（+0x2000）に
        // 一致することを確認する。
        let base = 1.0f32.to_bits();
        let x = f32::from_bits(base | 0x1000);
        let rounded = round_to_tf32_rna(x);
        assert_eq!(rounded.to_bits(), base.wrapping_add(0x2000) & !0x1FFF);
        assert!(rounded > x || rounded.to_bits() == (base.wrapping_add(0x2000) & !0x1FFF));
    }

    #[test]
    fn negative_value_rounds_symmetrically() {
        let base = (-1.5f32).to_bits();
        let perturbed = f32::from_bits(base | 0x1000);
        let rounded = round_to_tf32_rna(perturbed);
        // 丸め後は仮数下位 13 bit が必ず 0（TF32 表現可能値）。
        assert_eq!(rounded.to_bits() & 0x1FFF, 0);
    }
}

/// fail 要素の座標・`row%16`/`col%8` ヒストグラムを stdout へダンプする
/// （m16n8k8 mma フラグメント境界への偏り検出用。`limit` 件まで詳細座標を
/// 出す）。
fn dump_fail_coordinates(label: &str, actual: &[f32], expected: &[f32], n: usize, limit: usize) {
    let mut fail_coords: Vec<(usize, usize)> = Vec::new();
    let mut hist_row16 = [0usize; 16];
    let mut hist_col8 = [0usize; 8];
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let xf = a as f64;
        let yf = e as f64;
        let diff = (xf - yf).abs();
        let scale = xf.abs().max(yf.abs()).max(1e-12);
        let rel = diff / scale;
        let pass = rel < fandhe_ai_backend_cpu::RELATIVE_TOLERANCE
            || diff < fandhe_ai_backend_cpu::ABSOLUTE_RESCUE_THRESHOLD;
        if !pass {
            let row = idx / n;
            let col = idx % n;
            hist_row16[row % 16] += 1;
            hist_col8[col % 8] += 1;
            if fail_coords.len() < limit {
                fail_coords.push((row, col));
            }
        }
    }
    println!("  [{label}] fail coords (up to {limit}): {:?}", fail_coords);
    println!("  [{label}] row%16 histogram: {:?}", hist_row16);
    println!("  [{label}] col%8  histogram: {:?}", hist_col8);
}

fn print_report(label: &str, report: Result<CompareReport, fandhe_ai_backend_cpu::ParityError>) {
    match report {
        Ok(r) => println!(
            "  [{label}] fail_count={}/{} max_abs_diff={:.3e} max_rel_err={:.3e} \
             mean_abs_diff={:.3e} mean_rel_err={:.3e}",
            r.fail_count, r.total, r.max_abs_diff, r.max_rel_err, r.mean_abs_diff, r.mean_rel_err,
        ),
        Err(err) => println!("  [{label}] compare error: {err}"),
    }
}

/// 診断専用ダンプ本体（1 形状分）。既存テストの入力生成規則（`across_shapes`
/// は `5000+idx`、K 支配的ストレスケースは `9001`/`9002` 相当）を踏襲した
/// シードを渡す想定。
#[allow(clippy::too_many_arguments)]
fn triage_shape(
    mma: &CudaMmaTf32Gemm,
    wmma: &CudaGemm,
    label: &str,
    seed: u64,
    m: u32,
    n: u32,
    k: u32,
    dump_coords: bool,
) {
    let (m_us, n_us, k_us) = (m as usize, n as usize, k as usize);
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec(m_us * k_us);
    let b = rng.fill_vec(k_us * n_us);

    // (a) mma.sync 出力 vs 「丸めなし」CPU f32 参照（既存テストと同じ基準）。
    let mut c_ref_f32 = vec![0.0f32; m_us * n_us];
    matmul_reference_fma(&a, &b, &mut c_ref_f32, m_us, n_us, k_us)
        .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    let c_mma = mma.run_tf32(&a, &b, m, n, k).expect(
        "CudaMmaTf32Gemm::run_tf32 must succeed on a compute capability >= 8.0 test runner",
    );

    // (b) mma.sync 出力 vs 「TF32 事前丸め入力」の CPU 参照。GPU 側と同じ
    // 丸め済み入力を使うことで、TF32 丸め誤差成分を参照値側にも反映させる。
    let a_tf32 = round_slice_to_tf32_rna(&a);
    let b_tf32 = round_slice_to_tf32_rna(&b);
    let mut c_ref_tf32 = vec![0.0f32; m_us * n_us];
    matmul_reference_fma(&a_tf32, &b_tf32, &mut c_ref_tf32, m_us, n_us, k_us)
        .expect("matmul_reference_fma shape validation must pass for tf32-rounded test input");

    // (c) staged（run_wmma_tf32）出力 vs (b) と同じ TF32 事前丸め参照。
    let c_staged = wmma
        .run_wmma_tf32(&a, &b, m, n, k)
        .expect("CudaGemm::run_wmma_tf32 must succeed on a compute capability >= 8.0 test runner");

    println!("--- {label} (m={m} n={n} k={k} seed={seed}) ---");
    print_report(
        "(a) mma vs CPU f32 ref (no pre-rounding)",
        compare(&c_mma, &c_ref_f32),
    );
    print_report(
        "(b) mma vs CPU ref w/ TF32-rounded inputs",
        compare(&c_mma, &c_ref_tf32),
    );
    print_report(
        "(c) staged vs CPU ref w/ TF32-rounded inputs",
        compare(&c_staged, &c_ref_tf32),
    );
    print_report("(d) mma vs staged (direct)", compare(&c_mma, &c_staged));

    if dump_coords {
        dump_fail_coordinates("(a) mma vs f32 ref", &c_mma, &c_ref_f32, n_us, 32);
        dump_fail_coordinates("(d) mma vs staged", &c_mma, &c_staged, n_us, 32);
    }
}

/// 診断ダンプ本体。`--ignored --nocapture` で実行し、stdout の表を目視で
/// 確認する用途（受け入れ判定には使わない。ファイル冒頭コメント参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須・診断専用（受け入れ判定対象外）"]
fn mma_tf32_triage_dump() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let mma = CudaMmaTf32Gemm::new(&device).expect("TF32 mma.sync kernel compilation must succeed");
    let wmma = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");
    assert!(
        wmma.wmma_tf32_staged_available(),
        "staged kernel must be available on this ignored test runner (reason: {:?})",
        wmma.wmma_tf32_staged_unavailable_reason()
    );

    // 既存テストの生成規則を踏襲: 小規模・網羅形状は across_shapes 系の
    // `5000+idx`、K 支配的ストレス形状は K4096 系の `9001` を流用する
    // （`tests/gemm_mma_tf32.rs`・`tests/mma_tf32_vs_wmma_tf32_staged.rs`
    // 参照）。
    triage_shape(
        &mma,
        &wmma,
        "16x8x8 (1 mma.sync call)",
        5000,
        16,
        8,
        8,
        true,
    );
    triage_shape(
        &mma,
        &wmma,
        "64x64x64 (1 block tile)",
        5001,
        64,
        64,
        64,
        false,
    );
    triage_shape(&mma, &wmma, "512x512x512", 5002, 512, 512, 512, true);
    triage_shape(
        &mma,
        &wmma,
        "256x256x4096 (K-dominant)",
        9001,
        256,
        256,
        4096,
        false,
    );
}

/// `TRIAGE_ROW` 機械可読プレフィックス付きで 1 行出力する
/// （`test=<元テスト名> shape=MxNxK seed=<seed>` の後に `compare` 統計を
/// 続ける。baseline 非後退方式〈`docs/spec/04-requirements.md` REQ-2
/// 2026-09-02 追記〉への再割り当て提案に必要な全形状の実測値を、
/// grep 可能な単一行フォーマットで残す）。
fn print_triage_row(test_name: &str, m: u32, n: u32, k: u32, seed: u64, report: &CompareReport) {
    println!(
        "TRIAGE_ROW test={test_name} shape={m}x{n}x{k} seed={seed} \
         fail_count={}/{} max_abs_diff={:.6e} max_rel_err={:.6e} mean_abs_diff={:.6e}",
        report.fail_count,
        report.total,
        report.max_abs_diff,
        report.max_rel_err,
        report.mean_abs_diff,
    );
}

/// mma.sync 出力 vs CPU f32 参照（丸めなし。`matmul_reference_fma`）の
/// `compare` レポートを返す（`assert_parity` は使わず統計のみ取得する）。
fn mma_vs_cpu_ref_report(
    mma: &CudaMmaTf32Gemm,
    seed: u64,
    m: u32,
    n: u32,
    k: u32,
) -> CompareReport {
    let (m_us, n_us, k_us) = (m as usize, n as usize, k as usize);
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec(m_us * k_us);
    let b = rng.fill_vec(k_us * n_us);

    let mut c_ref = vec![0.0f32; m_us * n_us];
    matmul_reference_fma(&a, &b, &mut c_ref, m_us, n_us, k_us)
        .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    let c_mma = mma.run_tf32(&a, &b, m, n, k).expect(
        "CudaMmaTf32Gemm::run_tf32 must succeed on a compute capability >= 8.0 test runner",
    );

    compare(&c_mma, &c_ref).expect("compare: mma/c_ref lengths must match by construction")
}

/// mma.sync 出力 vs staged（`CudaGemm::run_wmma_tf32`）出力の `compare`
/// レポートを返す（`assert_parity` は使わず統計のみ取得する）。
fn mma_vs_staged_report(
    mma: &CudaMmaTf32Gemm,
    wmma: &CudaGemm,
    seed: u64,
    m: u32,
    n: u32,
    k: u32,
) -> CompareReport {
    let (m_us, n_us, k_us) = (m as usize, n as usize, k as usize);
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec(m_us * k_us);
    let b = rng.fill_vec(k_us * n_us);

    let c_staged = wmma
        .run_wmma_tf32(&a, &b, m, n, k)
        .expect("CudaGemm::run_wmma_tf32 must succeed on a compute capability >= 8.0 test runner");
    let c_mma = mma.run_tf32(&a, &b, m, n, k).expect(
        "CudaMmaTf32Gemm::run_tf32 must succeed on a compute capability >= 8.0 test runner",
    );

    compare(&c_mma, &c_staged).expect("compare: mma/staged lengths must match by construction")
}

/// baseline 非後退方式（`docs/spec/04-requirements.md` REQ-2 2026-09-02
/// 追記）への再割り当て提案に必要な、全形状・全シードの実測値を
/// `TRIAGE_ROW` 形式で出力する診断テスト（`assert_parity` の厳密ゼロ fail
/// 判定は使わず、全ケースを最後まで走らせて統計のみ収集する）。
///
/// 1・2 は既存の 2 テストと**形状リスト・シード規則を完全一致**させる
/// （`tests/gemm_mma_tf32.rs::mma_tf32_matches_reference_across_shapes`・
/// `tests/gemm_mma_tf32.rs::mma_tf32_k4096_stress`・
/// `tests/mma_tf32_vs_wmma_tf32_staged.rs::
/// mma_tf32_matches_wmma_tf32_staged_across_shapes`・同ファイルの
/// `..._k4096_stress`）。3 は厳密ゼロ fail が成立する最小形状・シードの
/// 探索用に追加した独自ケース。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須・診断専用（受け入れ判定対象外）"]
fn mma_tf32_triage_all_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let mma = CudaMmaTf32Gemm::new(&device).expect("TF32 mma.sync kernel compilation must succeed");
    let wmma = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");
    assert!(
        wmma.wmma_tf32_staged_available(),
        "staged kernel must be available on this ignored test runner (reason: {:?})",
        wmma.wmma_tf32_staged_unavailable_reason()
    );

    // --- 1. mma_tf32_matches_reference_across_shapes と完全一致する形状・
    // シード（seed = 5000 + idx）。`gemm_mma_tf32.rs` の cases 配列を
    // そのまま複製する。
    let across_shapes_cases: &[(u32, u32, u32)] = &[
        (16, 8, 8),
        (64, 64, 64),
        (128, 128, 128),
        (512, 512, 512),
        (60, 68, 36),
        (68, 60, 20),
        (96, 68, 72),
        (64, 96, 256),
        (4, 4, 4),
    ];
    for (idx, &(m, n, k)) in across_shapes_cases.iter().enumerate() {
        let seed = 5000 + idx as u64;
        let report = mma_vs_cpu_ref_report(&mma, seed, m, n, k);
        print_triage_row(
            "mma_tf32_matches_reference_across_shapes",
            m,
            n,
            k,
            seed,
            &report,
        );
    }
    // mma_tf32_k4096_stress（seed=9001）。
    {
        let (m, n, k) = (4096, 4096, 4096);
        let seed = 9001;
        let report = mma_vs_cpu_ref_report(&mma, seed, m, n, k);
        print_triage_row("mma_tf32_k4096_stress", m, n, k, seed, &report);
    }

    // --- 2. mma_tf32_matches_wmma_tf32_staged_across_shapes と完全一致する
    // 形状・シード（seed = 6000 + idx）。`mma_tf32_vs_wmma_tf32_staged.rs`
    // の cases 配列をそのまま複製する（`(16,8,8)` は含まれない点に注意）。
    let staged_cases: &[(u32, u32, u32)] = &[
        (64, 64, 64),
        (128, 128, 128),
        (512, 512, 512),
        (60, 68, 36),
        (68, 60, 20),
        (96, 68, 72),
        (64, 96, 256),
        (4, 4, 4),
    ];
    for (idx, &(m, n, k)) in staged_cases.iter().enumerate() {
        let seed = 6000 + idx as u64;
        let report = mma_vs_staged_report(&mma, &wmma, seed, m, n, k);
        print_triage_row(
            "mma_tf32_matches_wmma_tf32_staged_across_shapes",
            m,
            n,
            k,
            seed,
            &report,
        );
    }
    // mma_tf32_matches_wmma_tf32_staged_k4096_stress（seed=9002）。
    {
        let (m, n, k) = (4096, 4096, 4096);
        let seed = 9002;
        let report = mma_vs_staged_report(&mma, &wmma, seed, m, n, k);
        print_triage_row(
            "mma_tf32_matches_wmma_tf32_staged_k4096_stress",
            m,
            n,
            k,
            seed,
            &report,
        );
    }

    // --- 3. 厳密ゼロ fail が成立する最小形状・シードの探索用の独自ケース
    // （既存テストには存在しない組合せ）。
    let minimal_shapes: &[(u32, u32, u32)] = &[(1, 1, 1), (1, 1, 8), (16, 8, 8)];
    for &(m, n, k) in minimal_shapes {
        for seed in 7001..=7005u64 {
            let report = mma_vs_cpu_ref_report(&mma, seed, m, n, k);
            print_triage_row("minimal_shape_search", m, n, k, seed, &report);
        }
    }
}
