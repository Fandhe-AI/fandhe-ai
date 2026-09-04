//! イシュー #1155 診断専用テスト: `specialized_mma_parity.rs::
//! specialized_mma_f16_matches_default_and_reference_across_shapes` の
//! `(m,n,k)=(256,512,1024)`・`CompiledDims::DYNAMIC_ALL`・seed=4003
//! （`4000 + idx 3`）で GB10 実機の CPU 参照複合判定が FAIL する原因を、
//! 「f16 丸め誤差（累算順序差含む）」と「機能欠陥（フラグメント
//! マッピング・境界・store 誤り等）」のどちらに起因するかで切り分ける。
//!
//! `gemm_mma_tf32_triage.rs`（#1122）と同型の位置づけであり、**本ファイルは
//! 受け入れ判定には使わない**（`.claude/rules/coding-rust.md`「TF32/f16
//! Tensor Core 経路の parity テスト判定方式」の対象外）。`#[ignore]`
//! 実機依存の `--nocapture` 前提のダンプ出力のみを提供し、既存の
//! tolerance 定数（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）・
//! カーネル・既存テスト（`specialized_mma_parity.rs`）は一切変更しない。
//!
//! ## 切り分けロジック
//!
//! `SpecializedMmaKernelHandle`（`run_specialized_mma_f16`）は
//! `#define` によるコンパイル時定数の焼き込みのみで演算命令列・
//! アキュムレート順序を変えない契約（`kernels_mma.rs` 冒頭コメント）
//! のため、CPU 参照との乖離は Tensor Core 側の f32 アキュムレート順序差
//! （逐次 k 昇順の CPU 参照とは異なる順序で部分和を蓄積する）と f16 出力
//! 丸めのいずれか、または機能欠陥に起因する。判別には 2 段の証拠を使う:
//!
//! 1. **統計的証拠**（主ダンプ）: fail セルの座標・|厳密値|（f64 逐次
//!    計算）・CPU 参照自身の厳密値からの乖離・GPU 出力の厳密値からの
//!    乖離を比較する。GPU 側の乖離が CPU 参照側の乖離と同程度の桁
//!    （f32 ULP レベル。~1e-5〜1e-6）に収まり、fail セルが |厳密値| が
//!    小さい打ち消し合いセルに集中し、フラグメント境界
//!    （`row%16`/`col%8`）・ブロックタイル境界（`row%BM`/`col%BN`）への
//!    偏りがなければ「順序・丸め由来」を示唆する。
//! 2. **決定的証拠**（整数厳密入力テスト）: A・B の要素を
//!    `{-1, 0, +1}`（f16 で厳密表現・積も厳密・|部分和| ≤ K=1024 は
//!    f16・f32 とも厳密表現可）にすると、累算順序・Tensor Core 内部
//!    精度に依らず GPU 出力は CPU 参照と **bit 一致しなければならない**。
//!    1 要素でも不一致があれば機能欠陥（順序・丸めでは説明できない）、
//!    全一致なら演算経路（インデックス・境界・store）は正しいと言える
//!    （`gemm_mma_tf32_triage.rs` の TF32 事前丸め手法の f16 版に相当し、
//!    統計比較より強い判別になる）。

use fandhe_ai_backend_cpu::{
    ABSOLUTE_RESCUE_THRESHOLD, CompareReport, RELATIVE_TOLERANCE, compare, matmul_reference_fma,
};
use fandhe_ai_backend_cuda::{
    CompiledDims, CudaDevice, CudaMmaGemm, diagnostics, run_specialized_mma_f16,
};
use half::f16;

// --- ホスト側ヘルパー（GPU 不要。以下は非 ignore ユニットテストで
// 単体検証する） ---

/// `specialized_mma_parity.rs::gen_ab` と同一の決定的生成規則（一様乱数
/// `[-1, 1)` の f16 化）。GPU 側・CPU 参照の双方が同一入力を使うための
/// ヘルパー。
fn gen_ab(seed: u64, m: u32, n: u32, k: u32) -> (Vec<f16>, Vec<f16>) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
    let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));
    (a, b)
}

/// 整数厳密入力（{-1, 0, +1} のみ）を決定的に生成する（判別軸 2 の
/// 決め手）。`next_f32()` は `[-1, 1)` の一様分布を返すため、3 分位
/// （`< -1/3` / `< 1/3` / それ以外）で 3 値へ写像する。f16 で厳密表現
/// 可能な値のみを使うため丸め誤差は一切発生しない。
fn gen_ab_exact_int(seed: u64, m: u32, n: u32, k: u32) -> (Vec<f16>, Vec<f16>) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let map = |v: f32| -> f16 {
        if v < -1.0 / 3.0 {
            f16::from_f32(-1.0)
        } else if v < 1.0 / 3.0 {
            f16::from_f32(0.0)
        } else {
            f16::from_f32(1.0)
        }
    };
    let a: Vec<f16> = (0..(m as usize) * (k as usize))
        .map(|_| map(rng.next_f32()))
        .collect();
    let b: Vec<f16> = (0..(k as usize) * (n as usize))
        .map(|_| map(rng.next_f32()))
        .collect();
    (a, b)
}

/// f16→f32→`matmul_reference_fma`（丸め前 f32。既存テストと同一手順・
/// 同一関数を再利用し判定ロジックを複製しない）で CPU 参照値を得る。
fn cpu_reference_f32(a: &[f16], b: &[f16], m: u32, n: u32, k: u32) -> Vec<f32> {
    let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32 = vec![0.0f32; (m as usize) * (n as usize)];
    matmul_reference_fma(
        &a_f32,
        &b_f32,
        &mut c_ref_f32,
        m as usize,
        n as usize,
        k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    c_ref_f32
}

/// `values` を f16 へ丸めてから f32 へ戻す（GPU 側の出力丸め・CPU 参照の
/// 「f16 丸め後」表現を揃えて比較するためのヘルパー）。
fn round_f16(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
}

/// f64 逐次アキュムレート（k 昇順）による「厳密値」参照。K=1024 程度の
/// 和では f64 の丸め誤差（~1e-13 オーダー）は f32/f16 の判別対象より
/// 十分小さく「厳密値」として扱える（累算順序非依存の前提の確認は
/// 非 ignore ユニットテストで検証する）。`matmul_reference_fma` の
/// イテレーション順序（i 外側・k 中間・j 内側。`parity.rs` 参照）を
/// 踏襲する。
fn exact_reference_f64(a: &[f16], b: &[f16], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        let a_row = &a[i * k..i * k + k];
        let c_row = &mut c[i * n..i * n + n];
        for (p, &a_ip) in a_row.iter().enumerate() {
            let a_ip = a_ip.to_f32() as f64;
            let b_row = &b[p * n..p * n + n];
            for j in 0..n {
                c_row[j] += a_ip * (b_row[j].to_f32() as f64);
            }
        }
    }
    c
}

/// 16 要素チャンクを f64 で厳密に足してから f32 化し、チャンク単位で
/// f32 加算する参考モデル（Tensor Core 内部の部分和グルーピングを
/// 粗く模したもの。ホストで厳密再現できないため**参考情報**に留め、
/// 切り分けの決め手にはしない。本ファイル冒頭コメント参照）。
fn chunk16_model_f32(a: &[f16], b: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        let a_row = &a[i * k..i * k + k];
        let c_row = &mut c[i * n..i * n + n];
        for j in 0..n {
            let mut acc = 0.0f32;
            let mut p = 0usize;
            while p < k {
                let end = (p + 16).min(k);
                let mut chunk_sum = 0.0f64;
                for pp in p..end {
                    let a_ip = a_row[pp].to_f32() as f64;
                    let b_pj = b[pp * n + j].to_f32() as f64;
                    chunk_sum += a_ip * b_pj;
                }
                acc += chunk_sum as f32;
                p = end;
            }
            c_row[j] = acc;
        }
    }
    c
}

/// f16 の bit パターン差（signed magnitude → 2 の補数相当の単調写像に
/// 変換してから差を取る。ULP 距離の標準的な求め方）。
fn f16_ulp_distance(x: f16, y: f16) -> u32 {
    // `to_bits()` は符号なし 16bit（`u16`）を返すため、符号ビット
    // （bit 15）で正負を判定してから単調写像へ変換する（`bits as i32`
    // は常に非負になるため、符号ビットの有無で分岐する必要がある）。
    fn key(v: f16) -> i32 {
        let bits = v.to_bits();
        let magnitude = (bits & 0x7FFF) as i32;
        if bits & 0x8000 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }
    (key(x) - key(y)).unsigned_abs()
}

fn print_report(label: &str, report: &Result<CompareReport, fandhe_ai_backend_cpu::ParityError>) {
    match report {
        Ok(r) => println!(
            "  [{label}] fail_count={}/{} max_abs_diff={:.3e} max_fail_abs_diff={:.3e} \
             max_rel_err={:.3e} mean_abs_diff={:.3e}",
            r.fail_count,
            r.total,
            r.max_abs_diff,
            r.max_fail_abs_diff,
            r.max_rel_err,
            r.mean_abs_diff,
        ),
        Err(err) => println!("  [{label}] compare error: {err}"),
    }
}

/// `TRIAGE_ROW` 機械可読プレフィックス付きで 1 行出力する（`test=<label>
/// shape=MxNxK seed=<seed>` の後に `compare` 統計を続ける。
/// `gemm_mma_tf32_triage.rs::print_triage_row` と同じフォーマット。
/// 後続 issue（#1159/#1161）が grep 可能な形式で残す）。
fn print_triage_row(label: &str, m: u32, n: u32, k: u32, seed: u64, report: &CompareReport) {
    println!(
        "TRIAGE_ROW test={label} shape={m}x{n}x{k} seed={seed} \
         fail_count={}/{} max_abs_diff={:.6e} max_rel_err={:.6e} mean_abs_diff={:.6e}",
        report.fail_count,
        report.total,
        report.max_abs_diff,
        report.max_rel_err,
        report.mean_abs_diff,
    );
}

/// 複合判定（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD` を読むだけ。
/// 定数は複製しない）で fail したセルの座標・値・厳密値からの乖離・
/// ULP 距離・フラグメント/ブロックタイル境界ヒストグラムをダンプする。
/// fail 判定は [`compare`] と同一の複合判定式を用いる（`scale` の分母を
/// `gpu.abs().max(cpu.abs())` とする点を含め、実装を分岐させない。
/// `host_helper_unit_tests::dump_fail_cells_fail_count_matches_compare`
/// が両者の fail 件数一致を回帰検査する）。fail 件数を返り値としても
/// 返す（同ユニットテストが `compare` の `fail_count` と突き合わせる
/// ため）。
fn dump_fail_cells(
    label: &str,
    gpu_f16: &[f16],
    cpu_ref_f16: &[f16],
    exact_f64: &[f64],
    n: usize,
    limit: usize,
) -> usize {
    let (bm, bn) = diagnostics::mma_f16_block_tile();
    let mut printed = 0usize;
    let mut fail_count = 0usize;
    let mut small_exact_count = 0usize; // |厳密| < 1e-2 の fail セル数
    let mut gpu_err_le_2x_cpu_err_or_tiny = 0usize;
    let mut ulp_le_1_count = 0usize;
    let mut hist_row16 = [0usize; 16];
    let mut hist_col8 = [0usize; 8];
    let mut hist_row_bm = vec![0usize; bm as usize];
    let mut hist_col_bn = vec![0usize; bn as usize];

    for idx in 0..gpu_f16.len() {
        let gpu = gpu_f16[idx].to_f32() as f64;
        let cpu = cpu_ref_f16[idx].to_f32() as f64;
        let exact = exact_f64[idx];
        let diff = (gpu - cpu).abs();
        let scale = gpu.abs().max(cpu.abs()).max(1e-12);
        let rel = diff / scale;
        let pass = rel < RELATIVE_TOLERANCE || diff < ABSOLUTE_RESCUE_THRESHOLD;
        if pass {
            continue;
        }
        fail_count += 1;

        let row = idx / n;
        let col = idx % n;
        hist_row16[row % 16] += 1;
        hist_col8[col % 8] += 1;
        hist_row_bm[row % (bm as usize)] += 1;
        hist_col_bn[col % (bn as usize)] += 1;

        let gpu_err = (gpu - exact).abs();
        let cpu_err = (cpu - exact).abs();
        if exact.abs() < 1e-2 {
            small_exact_count += 1;
        }
        if gpu_err <= 2.0 * cpu_err || gpu_err <= 1e-4 {
            gpu_err_le_2x_cpu_err_or_tiny += 1;
        }
        let ulp = f16_ulp_distance(gpu_f16[idx], f16::from_f64(exact));
        if ulp <= 1 {
            ulp_le_1_count += 1;
        }

        if printed < limit {
            println!(
                "  [{label}] cell (row={row}, col={col}): gpu={gpu:.6e} cpu_ref={cpu:.6e} \
                 exact={exact:.6e} |gpu-exact|={gpu_err:.3e} |cpu-exact|={cpu_err:.3e} \
                 ulp(gpu,exact)={ulp}"
            );
            printed += 1;
        }
    }

    println!(
        "  [{label}] fail_count={fail_count} small_exact(|exact|<1e-2)={small_exact_count} \
         gpu_err<=2x_cpu_err_or_tiny={gpu_err_le_2x_cpu_err_or_tiny} ulp<=1={ulp_le_1_count}"
    );
    println!("  [{label}] row%16 histogram: {:?}", hist_row16);
    println!("  [{label}] col%8  histogram: {:?}", hist_col8);
    println!("  [{label}] row%BM({bm}) histogram: {:?}", hist_row_bm);
    println!("  [{label}] col%BN({bn}) histogram: {:?}", hist_col_bn);
    fail_count
}

// --- 非 ignore ユニットテスト（GPU 不要。CI の `cargo test --all-features`
// で実行される。ホスト側ヘルパーの前提条件を検証する） ---

#[cfg(test)]
mod host_helper_unit_tests {
    use super::*;

    #[test]
    fn gen_ab_exact_int_only_produces_negative_one_zero_one() {
        let (a, b) = gen_ab_exact_int(4003, 8, 8, 16);
        for &v in a.iter().chain(b.iter()) {
            let f = v.to_f32();
            assert!(
                f == -1.0 || f == 0.0 || f == 1.0,
                "gen_ab_exact_int must only emit {{-1,0,1}}, got {f}"
            );
        }
    }

    #[test]
    fn gen_ab_exact_int_is_deterministic() {
        let (a1, b1) = gen_ab_exact_int(4003, 8, 8, 16);
        let (a2, b2) = gen_ab_exact_int(4003, 8, 8, 16);
        assert_eq!(a1, a2);
        assert_eq!(b1, b2);
    }

    #[test]
    fn f16_ulp_distance_basic_properties() {
        assert_eq!(f16_ulp_distance(f16::from_f32(1.0), f16::from_f32(1.0)), 0);
        let one = f16::from_f32(1.0);
        let next = f16::from_bits(one.to_bits() + 1);
        assert_eq!(f16_ulp_distance(one, next), 1);
        // 符号跨ぎ: 最小正の非ゼロ値と最小負の非ゼロ値は 2 ULP 離れている
        // （+0 と -0 を挟むため）。
        let smallest_pos = f16::from_bits(1);
        let smallest_neg = f16::from_bits(0x8000 | 1);
        assert_eq!(f16_ulp_distance(smallest_pos, smallest_neg), 2);
    }

    #[test]
    fn exact_reference_f64_matches_matmul_reference_fma_within_f32_ulp() {
        // 小形状・通常の乱数入力（整数厳密でなくてよい）で f64 厳密計算
        // と f32 逐次 FMA 参照の差が微小であることを確認する（本テストの
        // 前提「K が小さければ f64 参照は f32 参照とほぼ一致する」の確認。
        // 大きな乖離は打ち消し合いセルの丸め誤差であり本ヘルパーの
        // バグではない）。
        let (m, n, k) = (4usize, 4usize, 8usize);
        let (a, b) = gen_ab(1, m as u32, n as u32, k as u32);
        let exact = exact_reference_f64(&a, &b, m, n, k);
        let cpu_ref = cpu_reference_f32(&a, &b, m as u32, n as u32, k as u32);
        for (idx, (&e, &c)) in exact.iter().zip(cpu_ref.iter()).enumerate() {
            let diff = (e - c as f64).abs();
            assert!(
                diff < 1e-4,
                "index {idx}: exact={e} cpu_ref={c} diff={diff} exceeds small-K tolerance"
            );
        }
    }

    #[test]
    fn dump_fail_cells_fail_count_matches_compare() {
        // `dump_fail_cells` の fail 判定は `compare` と同一の複合判定式を
        // 使わなければならない（本ファイル冒頭の `dump_fail_cells` ドキュ
        // メンテーションコメント参照）。手組みの gpu/cpu 値には
        // |cpu| > |gpu| かつ cpu が負のセルを含める
        // （`scale` の分母を `cpu`〈符号あり〉のまま使う実装ミスがあると
        // ここで `compare` と食い違う。回帰対象）。
        let gpu_f16 = [
            f16::from_f32(1.0),
            f16::from_f32(0.5),
            f16::from_f32(-3.0),
            f16::from_f32(0.0),
        ];
        let cpu_ref_f16 = [
            f16::from_f32(1.0),   // 一致（pass）
            f16::from_f32(0.505), // 微小差（rel ~1e-2 < 相対閾値ではないが絶対閾値未満、判定は compare に委ねる）
            f16::from_f32(-3.5),  // |cpu|>|gpu| かつ cpu が負（本テストの核）
            f16::from_f32(0.02),  // 絶対誤差のみで判定されるセル
        ];
        let exact_f64 = [1.0f64, 0.5, -3.2, 0.01];

        let gpu_f32: Vec<f32> = gpu_f16.iter().map(|x| x.to_f32()).collect();
        let cpu_f32: Vec<f32> = cpu_ref_f16.iter().map(|x| x.to_f32()).collect();
        let expected_fail_count = compare(&gpu_f32, &cpu_f32)
            .expect("compare: equal-length hand-built vectors must not error")
            .fail_count;

        let dumped_fail_count =
            dump_fail_cells("unit-test", &gpu_f16, &cpu_ref_f16, &exact_f64, 4, 0);

        assert_eq!(
            dumped_fail_count, expected_fail_count,
            "dump_fail_cells の fail 判定は compare() の複合判定と一致しなければならない"
        );
    }

    #[test]
    fn integer_exact_inputs_make_reference_fma_and_f64_exact_agree_bitwise() {
        // 整数厳密入力（{-1,0,1}）では累算順序に依らず和が f32/f64 とも
        // 厳密表現可能なため、`matmul_reference_fma`（f32 FMA 逐次）と
        // `exact_reference_f64`（f64 逐次）は完全一致する（判別軸 2 の
        // 前提そのものをホスト側で確認する）。
        let (m, n, k) = (8usize, 8usize, 64usize);
        let (a, b) = gen_ab_exact_int(4003, m as u32, n as u32, k as u32);
        let exact = exact_reference_f64(&a, &b, m, n, k);
        let cpu_ref = cpu_reference_f32(&a, &b, m as u32, n as u32, k as u32);
        for (idx, (&e, &c)) in exact.iter().zip(cpu_ref.iter()).enumerate() {
            assert_eq!(
                e, c as f64,
                "index {idx}: integer-exact inputs must agree bitwise (f64 exact={e} vs f32 ref={c})"
            );
        }
    }
}

// --- `#[ignore]` 診断テスト（GB10 実機必須。受け入れ判定には使わない） ---

/// 主ダンプ: `(256,512,1024)` seed=4003（FAIL 再現）・追加 seed 3 個
/// （同形状での fail 件数の seed 依存性の確認）・コントロール
/// `(128,256,128)` seed=4002（既存テストで pass 済みの形状）で
/// GPU（`DYNAMIC_ALL` 特化・既定 `run_f16`）・CPU 参照・f64 厳密値を
/// 比較し、fail セルの詳細・ヒストグラムをダンプする。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須・診断専用（受け入れ判定対象外）"]
fn specialized_mma_f16_triage_dump() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let default_gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    let cases: &[(&str, u64, u32, u32, u32, bool)] = &[
        (
            "(256,512,1024) seed=4003 [FAIL repro]",
            4003,
            256,
            512,
            1024,
            true,
        ),
        ("(256,512,1024) seed=7101", 7101, 256, 512, 1024, false),
        ("(256,512,1024) seed=7102", 7102, 256, 512, 1024, false),
        ("(256,512,1024) seed=7103", 7103, 256, 512, 1024, false),
        (
            "(128,256,128) seed=4002 [control, known pass]",
            4002,
            128,
            256,
            128,
            false,
        ),
    ];

    for &(label, seed, m, n, k, dump_coords) in cases {
        let (a, b) = gen_ab(seed, m, n, k);
        let (m_us, n_us, k_us) = (m as usize, n as usize, k as usize);

        let default_c = default_gemm
            .run_f16(&a, &b, m, n, k)
            .unwrap_or_else(|err| panic!("{label}: default run_f16 failed: {err}"));
        let specialized_c =
            run_specialized_mma_f16(&device, CompiledDims::DYNAMIC_ALL, &a, &b, m, n, k)
                .unwrap_or_else(|err| panic!("{label}: run_specialized_mma_f16 failed: {err}"));
        let bit_match = default_c == specialized_c;

        println!("--- {label} ---");
        println!("  DYNAMIC_ALL vs default run_f16 bit-match: {bit_match}");

        // CPU 参照（丸めなし f32・f16 丸め後）と f64 厳密値。
        let cpu_ref_f32 = cpu_reference_f32(&a, &b, m, n, k);
        let cpu_ref_f16: Vec<f16> = cpu_ref_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let exact = exact_reference_f64(&a, &b, m_us, n_us, k_us);
        let exact_f32: Vec<f32> = exact.iter().map(|&x| x as f32).collect();

        let gpu_f32: Vec<f32> = specialized_c.iter().map(|x| x.to_f32()).collect();
        let cpu_ref_rounded: Vec<f32> = cpu_ref_f16.iter().map(|x| x.to_f32()).collect();

        // (a) GPU f16 vs CPU 参照 f16（既存 specialized_mma_parity.rs と
        // 同じ判定基準）。
        print_report(
            "(a) gpu vs cpu_ref(f16)",
            &compare(&gpu_f32, &cpu_ref_rounded),
        );
        // (b) GPU f16 vs f16(厳密)。
        let exact_f16_as_f32 = round_f16(&exact_f32);
        print_report(
            "(b) gpu vs f16(exact)",
            &compare(&gpu_f32, &exact_f16_as_f32),
        );
        // (b') CPU 参照 f16 vs f16(厳密)。参照自身の誤差予算。
        print_report(
            "(b') cpu_ref(f16) vs f16(exact)",
            &compare(&cpu_ref_rounded, &exact_f16_as_f32),
        );
        // (p) 丸め前: GPU f16→f32 vs 厳密 f32・CPU 参照 f32 vs 厳密 f32。
        print_report(
            "(p) gpu(->f32) vs exact(f32, pre-round)",
            &compare(&gpu_f32, &exact_f32),
        );
        print_report(
            "(p) cpu_ref(f32, pre-round) vs exact(f32, pre-round)",
            &compare(&cpu_ref_f32, &exact_f32),
        );
        // (c) チャンク 16 モデル f16 vs GPU f16（参考情報。決め手にしない）。
        let chunk_model = chunk16_model_f32(&a, &b, m_us, n_us, k_us);
        let chunk_model_f16 = round_f16(&chunk_model);
        print_report(
            "(c, reference only) chunk16-model(f16) vs gpu",
            &compare(&chunk_model_f16, &gpu_f32),
        );

        if let Ok(report) = compare(&gpu_f32, &cpu_ref_rounded) {
            // seed=4003 の (256,512,1024) は
            // `specialized_mma_f16_matches_default_and_reference_across_shapes`
            // の該当ケース（idx=3・`assert_parity` 呼び出し）と同一の
            // 比較（GPU f16 vs CPU 参照 f16）のため、#1161 が grep で
            // 突き合わせられるよう元テスト名をラベルに使う。他 seed・
            // コントロール形状は本テスト固有のラベルにする。
            let triage_label = if label.starts_with("(256,512,1024) seed=4003") {
                "specialized_mma_f16_matches_default_and_reference_across_shapes"
            } else {
                "specialized_mma_f16_triage_dump"
            };
            print_triage_row(triage_label, m, n, k, seed, &report);
        }

        if dump_coords {
            dump_fail_cells(
                "gpu vs cpu_ref, exact-referenced",
                &specialized_c,
                &cpu_ref_f16,
                &exact,
                n_us,
                40,
            );
        }
    }
}

/// 決め手: 整数厳密入力（{-1,0,+1}）では累算順序・Tensor Core 内部精度に
/// 依らず GPU 出力は CPU 参照と bit 一致しなければならない。1 要素でも
/// 不一致があれば機能欠陥、全一致なら演算経路は正しい（本ファイル冒頭
/// コメント「判別軸 2」）。`(256,512,1024)` に加え、端タイル形状の
/// コントロール `(128,256,128)`・`(200,264,104)` も検査する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須・診断専用（受け入れ判定対象外）"]
fn specialized_mma_f16_triage_exact_integer_inputs() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let default_gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    let cases: &[(u32, u32, u32)] = &[(256, 512, 1024), (128, 256, 128), (200, 264, 104)];

    for &(m, n, k) in cases {
        let (a, b) = gen_ab_exact_int(4003, m, n, k);

        let cpu_ref_f32 = cpu_reference_f32(&a, &b, m, n, k);
        let cpu_ref_f16: Vec<f16> = cpu_ref_f32.iter().map(|&x| f16::from_f32(x)).collect();

        let default_c = default_gemm
            .run_f16(&a, &b, m, n, k)
            .unwrap_or_else(|err| panic!("shape ({m},{n},{k}): default run_f16 failed: {err}"));
        let specialized_c =
            run_specialized_mma_f16(&device, CompiledDims::DYNAMIC_ALL, &a, &b, m, n, k)
                .unwrap_or_else(|err| {
                    panic!("shape ({m},{n},{k}): run_specialized_mma_f16 failed: {err}")
                });

        for (label, gpu_c) in [
            ("default run_f16", &default_c),
            ("DYNAMIC_ALL specialized", &specialized_c),
        ] {
            let mismatches: Vec<(usize, f32, f32)> = gpu_c
                .iter()
                .zip(cpu_ref_f16.iter())
                .enumerate()
                .filter(|(_, (g, c))| g != c)
                .take(40)
                .map(|(idx, (g, c))| (idx, g.to_f32(), c.to_f32()))
                .collect();

            println!(
                "shape ({m},{n},{k}) [{label}]: mismatches (up to 40) vs integer-exact CPU \
                 reference: {mismatches:?}"
            );

            assert_eq!(
                gpu_c, &cpu_ref_f16,
                "shape ({m},{n},{k}) [{label}]: 整数厳密入力（{{-1,0,+1}}）は累算順序・\
                 Tensor Core 内部精度に依らず CPU 参照と bit 一致するはずです。不一致は \
                 順序・丸め由来では説明できない機能欠陥の証跡です（先頭 40 件は上記に \
                 ダンプ済み）"
            );
        }
    }
}
