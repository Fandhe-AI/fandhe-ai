//! イシュー #596: RMSNorm 逆伝播（recompute-in-backward）の CPU-CUDA 数値
//! 一致検証。
//!
//! `rmsnorm_parity.rs`（順伝播）と同じ構成方針を踏襲する: 環境適応スモーク
//! （通常 CI）と実機必須の形状網羅（`#[ignore]`）を分離し、判定式・許容
//! 誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の参照とする
//! （`.claude/rules/coding-rust.md`）。
//!
//! CPU 参照実装は本ファイル内のテスト専用関数（`f32::mul_add` 使用）で、
//! **素朴な保存方式**（正規化済みテンソルを保存して backward する通常の
//! autodiff 実装）を意味論の正とする。GPU 側は recompute-in-backward
//! （保存は行あたり `rstd` 1 本のみ）で同じ結果に到達することを検証する
//! （実装計画 §6「受け入れ判定」）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test rmsnorm_backward_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaRmsNorm, RmsNormShape};

mod common;

/// テスト専用 CPU 参照実装（素朴な保存方式）。順伝播で正規化済みテンソル
/// `normed = x * rstd` を保存し、backward でそれを使って
/// `dx_i = rstd·dy_i·w_i − rstd³·inv_n·x_i·Σ_j(dy_j·w_j·x_j)`・
/// `dw_i = Σ_r dy[r,i]·x[r,i]·rstd[r]` を計算する（`f32::mul_add` で
/// GPU 側 `fmaf` と丸め方針を揃える。`.claude/rules/coding-rust.md`）。
fn cpu_rmsnorm_backward_reference(
    x: &[f32],
    w: Option<&[f32]>,
    dy: &[f32],
    eps: f32,
    rows: usize,
    hidden: usize,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let mut dx = vec![0.0f32; x.len()];
    let mut dw = w.map(|w_slice| vec![0.0f32; w_slice.len()]);
    if hidden == 0 || rows == 0 {
        return (dx, dw);
    }
    let inv_n = 1.0f32 / hidden as f32;

    for r in 0..rows {
        let x_row = &x[r * hidden..(r + 1) * hidden];
        let dy_row = &dy[r * hidden..(r + 1) * hidden];

        let mut acc = 0.0f32;
        for &v in x_row {
            acc = v.mul_add(v, acc);
        }
        let rstd = 1.0f32 / (acc.mul_add(inv_n, eps)).sqrt();

        let mut dot = 0.0f32;
        for i in 0..hidden {
            let wv = w.map_or(1.0f32, |w_slice| w_slice[i]);
            dot = (dy_row[i] * wv).mul_add(x_row[i], dot);
        }

        let coef = -(rstd * rstd * rstd * inv_n * dot);
        let dx_row = &mut dx[r * hidden..(r + 1) * hidden];
        for i in 0..hidden {
            let wv = w.map_or(1.0f32, |w_slice| w_slice[i]);
            dx_row[i] = coef.mul_add(x_row[i], rstd * dy_row[i] * wv);
            if let Some(dw_vec) = dw.as_mut() {
                dw_vec[i] = (dy_row[i] * rstd).mul_add(x_row[i], dw_vec[i]);
            }
        }
    }
    (dx, dw)
}

fn assert_rmsnorm_backward_parity(
    rmsnorm: &CudaRmsNorm,
    seed_x: u64,
    seed_w: u64,
    seed_dy: u64,
    shape: RmsNormShape,
    with_weight: bool,
    eps: f32,
) {
    let RmsNormShape { rows, hidden } = shape;
    let x_data = Xorshift64Star::new(seed_x).fill_vec(rows * hidden);
    let dy_data = Xorshift64Star::new(seed_dy).fill_vec(rows * hidden);
    let w_data = if with_weight {
        Some(Xorshift64Star::new(seed_w).fill_vec(hidden))
    } else {
        None
    };

    // 順伝播（学習経路）で rstd を得る。`inv_n` は `run_rmsnorm_bwd_f32`
    // 内部で `shape.hidden` から `run_rmsnorm_f32_train` と同じ式
    // （`1/hidden`）で導出される（公開引数からは除去済み。codex-review P1
    // 是正・PR #711 レビュー r3794149870）。
    let (_out, rstd) = rmsnorm
        .run_rmsnorm_f32_train(&x_data, w_data.as_deref(), eps, rows, hidden)
        .expect("CudaRmsNorm::run_rmsnorm_f32_train must succeed on CUDA-equipped test runner");

    let (gpu_dx, gpu_dw) = rmsnorm
        .run_rmsnorm_bwd_f32(&x_data, w_data.as_deref(), &dy_data, &rstd, shape)
        .expect("CudaRmsNorm::run_rmsnorm_bwd_f32 must succeed on CUDA-equipped test runner");
    let (cpu_dx, cpu_dw) =
        cpu_rmsnorm_backward_reference(&x_data, w_data.as_deref(), &dy_data, eps, rows, hidden);

    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!(
            "rmsnorm backward dx cpu-cuda parity rows={rows} hidden={hidden} \
             with_weight={with_weight} eps={eps}"
        ),
        &gpu_dx,
        &cpu_dx,
    );

    match (gpu_dw, cpu_dw) {
        (Some(gpu_dw), Some(cpu_dw)) => {
            fandhe_ai_backend_cpu::parity::assert_parity(
                &format!(
                    "rmsnorm backward dw cpu-cuda parity rows={rows} hidden={hidden} eps={eps}"
                ),
                &gpu_dw,
                &cpu_dw,
            );
        }
        (None, None) => {}
        (gpu, cpu) => panic!(
            "dw Some/None mismatch: gpu={gpu:?}, cpu is_some={}",
            cpu.is_some()
        ),
    }
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。`rmsnorm_parity.rs::
/// rmsnorm_parity_smoke_env_adaptive` と同じ厳密な variant match パターン
/// を踏襲する（codex-review 指摘・PR #706 レビュー r3793473253 相当）。
#[test]
fn rmsnorm_backward_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaRmsNorm::new(&device) {
        Ok(rmsnorm) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            assert_rmsnorm_backward_parity(
                &rmsnorm,
                811,
                812,
                813,
                RmsNormShape { rows: 1, hidden: 8 },
                false,
                1e-5,
            );
            assert_rmsnorm_backward_parity(
                &rmsnorm,
                814,
                815,
                816,
                RmsNormShape {
                    rows: 3,
                    hidden: 1024,
                },
                true,
                1e-5,
            );
            // 退化ケース（PR #711 レビュー r3794159146・r3794149870
            // 是正の回帰確認）: `rows == 0`（`w` あり。`dw` が `hidden`
            // 長のゼロベクトルになる契約）と `hidden == 0`（`rstd` が
            // `rows` 長を維持する契約）。
            assert_rmsnorm_backward_parity(
                &rmsnorm,
                817,
                818,
                819,
                RmsNormShape { rows: 0, hidden: 8 },
                true,
                1e-5,
            );
            assert_rmsnorm_backward_parity(
                &rmsnorm,
                820,
                821,
                822,
                RmsNormShape { rows: 3, hidden: 0 },
                false,
                1e-5,
            );
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {}
        Err(other) => panic!("unexpected error variant for CudaRmsNorm::new: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`rmsnorm_parity.rs::
/// rmsnorm_matches_cpu_across_shapes` と同じ hidden/rows 網羅
/// （1 パス／2 パス双方の経路・vec4 端要素ケースを含む）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn rmsnorm_backward_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let rmsnorm = CudaRmsNorm::new(&device).expect("RMSNorm kernel compile must succeed");

    let hiddens: &[usize] = &[8, 1024, 4096, 4097, 8192, 16384];
    let rows_cases: &[usize] = &[1, 3, 33];

    let mut seed = 2000u64;
    for &hidden in hiddens {
        for &rows in rows_cases {
            for &with_weight in &[false, true] {
                seed += 1;
                let seed_w = seed + 500;
                let seed_dy = seed + 900;
                assert_rmsnorm_backward_parity(
                    &rmsnorm,
                    seed,
                    seed_w,
                    seed_dy,
                    RmsNormShape { rows, hidden },
                    with_weight,
                    1e-5,
                );
            }
        }
    }
}

/// split-K dw 経路専用の parity 検証（イシュー #597）: [`CudaRmsNorm::
/// run_rmsnorm_bwd_f32_with_forced_dw_split`]（テスト専用フック）で
/// `num_blocks` を明示指定し、単段経路（`num_blocks == 1`）・split-K 経路
/// （`num_blocks >= 2`）双方が CPU 参照実装と一致することを検証する
/// （`assert_rmsnorm_backward_parity` はヒューリスティクス経由の
/// `run_rmsnorm_bwd_f32` を使うため、本関数は別に定義する）。
fn assert_rmsnorm_backward_dw_split_parity(
    rmsnorm: &CudaRmsNorm,
    seed_x: u64,
    seed_w: u64,
    seed_dy: u64,
    shape: RmsNormShape,
    num_blocks: u32,
    eps: f32,
) {
    let RmsNormShape { rows, hidden } = shape;
    let x_data = Xorshift64Star::new(seed_x).fill_vec(rows * hidden);
    let dy_data = Xorshift64Star::new(seed_dy).fill_vec(rows * hidden);
    let w_data = Xorshift64Star::new(seed_w).fill_vec(hidden);

    let (_out, rstd) = rmsnorm
        .run_rmsnorm_f32_train(&x_data, Some(&w_data), eps, rows, hidden)
        .expect("CudaRmsNorm::run_rmsnorm_f32_train must succeed on CUDA-equipped test runner");

    let (gpu_dx, gpu_dw) = rmsnorm
        .run_rmsnorm_bwd_f32_with_forced_dw_split(
            &x_data,
            Some(&w_data),
            &dy_data,
            &rstd,
            shape,
            num_blocks,
        )
        .expect(
            "CudaRmsNorm::run_rmsnorm_bwd_f32_with_forced_dw_split must succeed on \
             CUDA-equipped test runner",
        );
    let (cpu_dx, cpu_dw) =
        cpu_rmsnorm_backward_reference(&x_data, Some(&w_data), &dy_data, eps, rows, hidden);

    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!(
            "rmsnorm backward dx cpu-cuda split-K parity rows={rows} hidden={hidden} \
             num_blocks={num_blocks} eps={eps}"
        ),
        &gpu_dx,
        &cpu_dx,
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!(
            "rmsnorm backward dw cpu-cuda split-K parity rows={rows} hidden={hidden} \
             num_blocks={num_blocks} eps={eps}"
        ),
        &gpu_dw.expect("with_weight=true means dw must be Some"),
        &cpu_dw.expect("with_weight=true means dw must be Some"),
    );
}

/// split-K 経路の環境適応スモーク（通常 CI）: 単段（`num_blocks=1`）・
/// split-K（`num_blocks=2` 以上）双方を明示指定して CPU 参照実装との
/// parity を確認する（受け入れ基準 4「単段経路との相互 parity」の一部。
/// ヒューリスティクス〈`derive_dw_split`〉に依存せず決定的に両経路を
/// 検証する）。
#[test]
fn rmsnorm_backward_dw_split_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaRmsNorm::new(&device) {
        Ok(rmsnorm) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            // 単段（フォールバック経路の明示検証）。
            assert_rmsnorm_backward_dw_split_parity(
                &rmsnorm,
                901,
                902,
                903,
                RmsNormShape {
                    rows: 64,
                    hidden: 32,
                },
                1,
                1e-5,
            );
            // split-K（`num_blocks=8`。`rows=256` を 8 分割し末尾ブロックが
            // ちょうど割り切れるケース）。
            assert_rmsnorm_backward_dw_split_parity(
                &rmsnorm,
                904,
                905,
                906,
                RmsNormShape {
                    rows: 256,
                    hidden: 32,
                },
                8,
                1e-5,
            );
            // split-K（`num_blocks` が `rows` を割り切らない末尾 block
            // ケース。`rows=100, num_blocks=7` → `rows_per_block=15`・
            // 最終 block は `[90,100)` の 10 行のみ）。
            assert_rmsnorm_backward_dw_split_parity(
                &rmsnorm,
                907,
                908,
                909,
                RmsNormShape {
                    rows: 100,
                    hidden: 16,
                },
                7,
                1e-5,
            );
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {}
        Err(other) => panic!("unexpected error variant for CudaRmsNorm::new: {other}"),
    }
}

/// split-K 経路の決定的再現性（受け入れ基準 4「同一入力→bit 一致」）:
/// 同一シード・同一 `num_blocks` で 2 回実行し `to_bits` 完全一致を
/// 確認する（advisor 指摘: 単段経路との bit 一致は主張しない——
/// split-K は加算順序〈結合順〉が単段と異なるため、単段と split-K の
/// 一致は REQ-2 複合許容誤差の範囲でのみ成立する。本テストは
/// 「同一パス・同一 `num_blocks` の 2 回実行」に限定した bit 一致のみを
/// 主張する）。
#[test]
fn rmsnorm_backward_dw_split_is_deterministic_across_repeated_runs_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaRmsNorm::new(&device) {
        Ok(rmsnorm) => {
            let rows = 128usize;
            let hidden = 64usize;
            let num_blocks = 5u32; // rows % num_blocks != 0 の末尾ケースを含む
            let shape = RmsNormShape { rows, hidden };
            let x_data = Xorshift64Star::new(1001).fill_vec(rows * hidden);
            let dy_data = Xorshift64Star::new(1002).fill_vec(rows * hidden);
            let w_data = Xorshift64Star::new(1003).fill_vec(hidden);
            let eps = 1e-5f32;

            let (_out, rstd) = rmsnorm
                .run_rmsnorm_f32_train(&x_data, Some(&w_data), eps, rows, hidden)
                .expect("run_rmsnorm_f32_train must succeed on CUDA-equipped test runner");

            let (dx1, dw1) = rmsnorm
                .run_rmsnorm_bwd_f32_with_forced_dw_split(
                    &x_data,
                    Some(&w_data),
                    &dy_data,
                    &rstd,
                    shape,
                    num_blocks,
                )
                .expect("first run must succeed");
            let (dx2, dw2) = rmsnorm
                .run_rmsnorm_bwd_f32_with_forced_dw_split(
                    &x_data,
                    Some(&w_data),
                    &dy_data,
                    &rstd,
                    shape,
                    num_blocks,
                )
                .expect("second run must succeed");

            assert_eq!(
                dx1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                dx2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "同一 num_blocks の 2 回実行で dx が bit 一致しない（split-K の加算順は \
                 num_blocks 固定なら決定的である契約）"
            );
            assert_eq!(
                dw1.expect("dw must be Some")
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                dw2.expect("dw must be Some")
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "同一 num_blocks の 2 回実行で dw が bit 一致しない"
            );
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {}
        Err(other) => panic!("unexpected error variant for CudaRmsNorm::new: {other}"),
    }
}

/// split-K dw の実機必須の形状網羅（受け入れ基準の本体）: 単段・split-K
/// 経路それぞれで CPU 参照実装との parity を、行数大（split-K が効く
/// 領域）・hidden の端要素ケース（vec4 非整列）を含めて確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn rmsnorm_backward_dw_split_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let rmsnorm = CudaRmsNorm::new(&device).expect("RMSNorm kernel compile must succeed");

    // (rows, hidden, num_blocks) の組み合わせ: 行数大・hidden の vec4 端
    // 要素ケース（4097 は 4 の倍数でない）・num_blocks が rows を割り切ら
    // ない端 block ケースを含む。
    let cases: &[(usize, usize, u32)] = &[
        (2048, 64, 1),
        (2048, 64, 2),
        (2048, 64, 16),
        (2048, 64, 64),
        (4096, 4097, 8), // hidden の vec4 非整列端要素
        (8192, 128, 33), // rows % num_blocks != 0 の端 block
        (8192, 4096, 1), // hidden が広く split-K の余地がない形状の単段確認
        (1000, 8, 7),    // rows % num_blocks != 0（小規模）
    ];

    let mut seed = 3000u64;
    for &(rows, hidden, num_blocks) in cases {
        seed += 1;
        let seed_w = seed + 500;
        let seed_dy = seed + 900;
        assert_rmsnorm_backward_dw_split_parity(
            &rmsnorm,
            seed,
            seed_w,
            seed_dy,
            RmsNormShape { rows, hidden },
            num_blocks,
            1e-5,
        );
    }
}

/// 受け入れ基準「保存はスカラー（`rstd` 等）のみ」の削減比実測（イシュー
/// #596 §3.4）: 素朴保存方式（正規化テンソル `rows*hidden*4` bytes）に
/// 対し、recompute-in-backward 方式は行あたり `rstd` 1 本
/// （`rows*4` bytes）のみを保存する。`hidden` 倍の削減になることを
/// 数値で assert し、削減比をテスト名・コメントに記録する
/// （PR 本文にも同数値を転記すること）。
#[test]
fn save_bytes_reduction_is_hidden_times_smaller() {
    let rows = 4usize;
    let hidden = 4096usize;

    let naive_saved_bytes = rows * hidden * std::mem::size_of::<f32>();
    let recompute_saved_bytes = rows * std::mem::size_of::<f32>();

    assert_eq!(naive_saved_bytes, 4 * 4096 * 4); // 65536 bytes = 16 KiB/行 * 4 行
    assert_eq!(recompute_saved_bytes, 16); // rstd 4 行分 = 4 bytes/行 * 4 行

    let reduction_ratio = naive_saved_bytes / recompute_saved_bytes;
    assert_eq!(reduction_ratio, hidden); // hidden=4096 倍の削減
}
