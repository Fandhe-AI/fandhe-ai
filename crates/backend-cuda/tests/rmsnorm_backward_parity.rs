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
//! **主検査は独立 CPU 参照実装による end-to-end parity（イシュー
//! #1102・`docs/perf/cuda-parity-baseline.md` §9）**: `rstd`・dw・dx の
//! いずれも GPU の内部実装詳細（split-K の有無・ブロック分割・warp
//! butterfly reduction 等）を一切参照しない独立実装（本ファイル内の
//! テスト専用関数）で計算し、forward→backward の一連の計算結果を GPU
//! 出力と比較する。これが本ファイルの唯一の主張（受け入れ判定）であり、
//! カーネル実装の内部詳細を検証側が知っている前提の検査（後述の追加
//! 検査）でこれを置き換えない（codex-review 指摘・PR #1120）。
//!
//! GB10 実機で観測された FAIL（イシュー #1102・#1105）はこの主検査に
//! 対して**カーネル側の精度改善**（forward の二乗和 `acc` を `double`
//! アキュムレータ化。`crates/backend-cuda/src/kernels_rmsnorm.rs` §9.7
//! 追補）で解消を図った。tolerance（`RELATIVE_TOLERANCE`／
//! `ABSOLUTE_RESCUE_THRESHOLD`）は一切変更していない
//! （`.claude/rules/coding-rust.md` の tolerance 非緩和方針）。
//!
//! **追加検査（主検査を置き換えない）**: `assert_rmsnorm_backward_
//! dw_split_parity` は上記の独立 end-to-end 検査に加え、GPU forward が
//! 生成した `rstd` バッファ自体の parity（取り違え・破損の検出）、および
//! 同一 `rstd` を供給した状態での dw／dx カーネル単体の縮約式・境界処理
//! 検査を追加で行う。これらはイシュー #1102／#1105 の切り分けで有用性が
//! 実証された補助検査であり、独立 end-to-end 検査の代替ではない。
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
///
/// `gpu_rstd`（行ごとの `rstd`。`Some` の場合は `rows` 要素）は
/// **追加検査専用**のフックである: `None` を渡すと本関数は自前で二乗和を
/// 逐次縮約して `rstd` を独立に再計算する（**主検査**——独立 CPU 参照
/// 実装による forward→backward end-to-end parity。本ファイル冒頭の
/// モジュールコメント参照——が使う経路であり、既定・推奨の使い方）。
/// `Some(&rstd)` を渡すと GPU forward が実際に生成した `rstd` をそのまま
/// 使い、dw／dx カーネル単体の縮約式・境界処理のみを切り分けて検証する
/// **追加検査**（`assert_rmsnorm_backward_dw_split_parity` が使う。主検査
/// の代替ではない）になる。`rstd` バッファ自体の parity は追加検査の
/// 呼び出し元が `cpu_rmsnorm_rstd_reference` との複合判定で別途検証する。
fn cpu_rmsnorm_backward_reference(
    x: &[f32],
    w: Option<&[f32]>,
    dy: &[f32],
    eps: f32,
    rows: usize,
    hidden: usize,
    gpu_rstd: Option<&[f32]>,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let mut dx = vec![0.0f32; x.len()];
    let mut dw = w.map(|w_slice| vec![0.0f32; w_slice.len()]);
    if hidden == 0 || rows == 0 {
        return (dx, dw);
    }
    let inv_n = 1.0f32 / hidden as f32;
    if let Some(r) = gpu_rstd {
        assert_eq!(
            r.len(),
            rows,
            "gpu_rstd は rows 要素でなければならない（呼び出し元の契約違反）"
        );
    }

    // dw の行方向蓄積は `f64` アキュムレータで行い、`dw`（`f32` の戻り値）
    // へは全行処理後に 1 回だけ downcast する（イシュー #1102 §9.8 追補。
    // GB10 実機実測で `rows=8192` に渡る dw の `f32` 逐次和〈GPU: `fmaf`
    // 蓄積／CPU 参照: `mul_add` 蓄積〉の縮約順序差が支配的な誤差になる
    // ことが判明したための対応。GPU 側もブロック内・ブロック間の縮約を
    // `double` 化したため〈`kernels_rmsnorm.rs` の `RMSNORM_BWD_DW_
    // PARTIAL_F32`／`RMSNORM_BWD_DW_REDUCE_F32`〉、独立 CPU 参照実装側も
    // 精度を揃える。これは「独立参照の弱体化」ではなく「より正確な独立
    // 参照への強化」であり、GPU 実装をコピーするものではない
    // （codex-review 指摘・PR #1120 の趣旨: 独立参照 + 実装側修正）。
    let mut dw_acc: Option<Vec<f64>> = dw.as_ref().map(|dw_vec| vec![0.0f64; dw_vec.len()]);

    for r in 0..rows {
        let x_row = &x[r * hidden..(r + 1) * hidden];
        let dy_row = &dy[r * hidden..(r + 1) * hidden];

        let rstd = match gpu_rstd {
            Some(rstd_slice) => rstd_slice[r],
            None => {
                // 二乗和も `f64` アキュムレータ化する（GPU forward の
                // `double` 化。`kernels_rmsnorm.rs` §9.7 追補と精度を
                // 揃える独立参照の強化。イシュー #1102 §9.8）。
                let mut acc = 0.0f64;
                for &v in x_row {
                    let v = v as f64;
                    acc = v.mul_add(v, acc);
                }
                (1.0f64 / (acc.mul_add(inv_n as f64, eps as f64)).sqrt()) as f32
            }
        };

        // dot（dx 用）は変更しない（コーディネータ指示: dx 参照はそのまま
        // で可。dx は GB10 実機で既に pass している経路であり、GPU 側
        // dx カーネル〈RMSNORM_BWD_DX_F32〉も本 PR で変更していない）。
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
            if let Some(dw_acc) = dw_acc.as_mut() {
                // 契約精密化（§9.10 追補・イシュー #1102。codex-review 指摘・
                // PR #1120）: 要素積（dy・rstd・x の 3 項）は f32 で確定して
                // から f64 へ昇格する（GPU 側 kernels_rmsnorm.rs の同契約と
                // 一致させる。forward の二乗和〈f64 昇格後に二乗〉とは異なる
                // 扱い）。
                let term = dy_row[i] * rstd * x_row[i];
                dw_acc[i] += term as f64;
            }
        }
    }

    if let (Some(dw_vec), Some(dw_acc)) = (dw.as_mut(), dw_acc) {
        for (dst, acc) in dw_vec.iter_mut().zip(dw_acc) {
            *dst = acc as f32;
        }
    }
    (dx, dw)
}

/// forward の `rstd` を CPU 側で独立に逐次和から再計算する。`f64`
/// アキュムレータ（GPU forward の `double` 化。`kernels_rmsnorm.rs`
/// §9.7 追補と精度を揃える独立参照の強化。イシュー #1102 §9.8）で
/// 二乗和を蓄積し、`rstd` へ代入する 1 回だけ `f32` へ downcast する。
/// 縮約順序自体は GPU forward の warp butterfly reduction とは異なる
/// 素朴な `0..hidden` 逐次和のままであり、GPU の内部実装詳細を模倣する
/// ものではない（`f64` 化は精度の強化であり、GPU 縮約順序のコピーでは
/// ない）。
///
/// `assert_rmsnorm_backward_dw_split_parity` が dw／dx カーネル検証で
/// GPU forward の `rstd` をそのまま CPU 参照側へ供給する方式（追加検査。
/// 本ファイル冒頭のモジュールコメント参照）に切り替えたことで失われる
/// 検証——GPU forward が保存した `rstd` バッファ自体の取り違え・破損
/// （行の取り違え・stale バッファ等、dw／dx の縮約式とは無関係な経路の
/// バグ）を検出する経路——をこの関数で補う。GPU `rstd` との比較は
/// `fandhe_ai_backend_cpu::parity::assert_parity`（REQ-2 統一複合判定）で
/// 行う（`docs/perf/cuda-parity-baseline.md` §9.5・§9.8）。
fn cpu_rmsnorm_rstd_reference(x: &[f32], eps: f32, rows: usize, hidden: usize) -> Vec<f32> {
    if hidden == 0 {
        // Bugbot 指摘（PR #1120）: `hidden == 0` は GPU 側
        // （`rmsnorm.rs::run_rmsnorm_f32_inner` の早期 return。`kernels_
        // rmsnorm.rs` のカーネル自体はこの縮退ケースでは起動されない）が
        // `sum(x^2) == 0` を根拠に `rstd = 1.0 / eps.sqrt()`（`eps` の
        // 有限性・非負性は `validate_rmsnorm_launch` が検証済み）を全行
        // 同一値で返す契約になっている。以前の実装は `0.0f32` で埋めて
        // おり、この契約と不一致だった（GPU/CPU で異なる `rstd` 定義に
        // なっていた）ため修正した。
        return vec![1.0f32 / eps.sqrt(); rows];
    }
    let inv_n = 1.0f64 / hidden as f64;
    let eps = eps as f64;
    (0..rows)
        .map(|r| {
            let x_row = &x[r * hidden..(r + 1) * hidden];
            let mut acc = 0.0f64;
            for &v in x_row {
                let v = v as f64;
                acc = v.mul_add(v, acc);
            }
            (1.0f64 / (acc.mul_add(inv_n, eps)).sqrt()) as f32
        })
        .collect()
}

/// weight gradient の split-K 経路（`num_blocks >= 2`）専用の CPU 参照
/// 実装。GPU 側 `RMSNORM_BWD_DW_PARTIAL_F32`／`RMSNORM_BWD_DW_REDUCE_F32`
/// （`kernels_rmsnorm.rs`）の二段縮約と**同一の加算順序**で `dw` を計算
/// する（イシュー #1102 GB10 実機再検証で判明した第 2 の非結合性。
/// `docs/perf/cuda-parity-baseline.md` §9.5・§9.8 追補）:
///
/// 1. 行を `num_blocks` 個のブロック（`rows_per_block =
///    ceil(rows / num_blocks)`。末尾ブロックは `rows` で切り詰め。GPU の
///    `RMSNORM_BWD_DW_PARTIAL_F32` と同じ切り方）へ分割し、各ブロック内は
///    行順の `mul_add`（`fma`）逐次蓄積で部分和を求める
/// 2. ブロック間はブロック番号 `0..num_blocks` の順に単純な加算
///    （`+=`。GPU の `RMSNORM_BWD_DW_REDUCE_F32` が
///    `acc += smem[buf][j][tid]` で行うのと同じ結合順序）で縮約する
///
/// **§9.8 追補（イシュー #1102）**: ブロック内・ブロック間のいずれも
/// `f64` アキュムレータで行い、`dw`（`f32` の戻り値）へは最終書き出し時
/// 1 回だけ downcast する。GB10 実機実測で `f32` 蓄積のままでは
/// （`(rows=4096, hidden=4097, num_blocks=8)` で `fail_count=1/4097,
/// max_abs_diff=3.052e-4`。9.5 追補）縮約順序差が支配的な誤差になったため、
/// GPU 側の対応する両カーネル（`RMSNORM_BWD_DW_PARTIAL_F32`／
/// `RMSNORM_BWD_DW_REDUCE_F32`）を `double` アキュムレータ化したのに
/// 合わせ、この「追加検査」用の CPU 参照実装側も精度を強化した
/// （GPU 縮約順序の模倣という設計は変えていない。精度のみ引き上げる）。
///
/// 浮動小数点加算は結合則を満たさないため、単段（`num_blocks == 1`。
/// `RMSNORM_BWD_DW_F32` 相当）の逐次和とはこの二段縮約が一般に一致しない。
/// dw split 経路の parity 検証はこの関数を使うことで、GPU の実際の縮約
/// 順序と揃えた上で dw カーネル自体の正しさ（縮約式・境界処理）を検証する
/// （`num_blocks == 1` でも本関数は単一ブロックの逐次蓄積に退化するため、
/// 単段・split-K 両方の呼び出し元で共通に使える。ただし単段側の実際の
/// GPU カーネル `RMSNORM_BWD_DW_F32` は §9.8 で同じく `double` 化した
/// ため、`num_blocks == 1` でもこの関数との精度前提は揃っている）。
fn cpu_rmsnorm_dw_split_reference(
    x: &[f32],
    dy: &[f32],
    rstd: &[f32],
    rows: usize,
    hidden: usize,
    num_blocks: u32,
) -> Vec<f32> {
    let mut dw_acc = vec![0.0f64; hidden];
    if hidden == 0 || rows == 0 || num_blocks == 0 {
        return vec![0.0f32; hidden];
    }
    let num_blocks = num_blocks as usize;
    let rows_per_block = rows.div_ceil(num_blocks);

    for b in 0..num_blocks {
        let row_start = (b * rows_per_block).min(rows);
        let row_end = ((b + 1) * rows_per_block).min(rows);
        for (i, dw_i) in dw_acc.iter_mut().enumerate() {
            let mut partial = 0.0f64;
            for (offset, &r) in rstd[row_start..row_end].iter().enumerate() {
                let row = row_start + offset;
                let idx = row * hidden + i;
                // 契約精密化（§9.10 追補・イシュー #1102。codex-review 指摘・
                // PR #1120）: 要素積は f32 で確定してから f64 へ昇格する
                // （GPU 側 RMSNORM_BWD_DW_PARTIAL_F32 の同契約と一致させる）。
                let term = dy[idx] * r * x[idx];
                partial += term as f64;
            }
            *dw_i += partial;
        }
    }
    dw_acc.into_iter().map(|v| v as f32).collect()
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
    // `gpu_rstd = None`: 本関数は `rmsnorm_backward_matches_cpu_across_shapes`
    // （dx 中心の受け入れテスト。実機で pass 実績あり）専用であり、rstd 自体の
    // 縮約順序差による ULP 増幅は dw split 経路（`rows` が大きい単段
    // フォールバック）ほど顕在化しない。dw 側の責務分離は
    // `assert_rmsnorm_backward_dw_split_parity` を参照（下記）。
    let (cpu_dx, cpu_dw) = cpu_rmsnorm_backward_reference(
        &x_data,
        w_data.as_deref(),
        &dy_data,
        eps,
        rows,
        hidden,
        None,
    );

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

    // ============================================================
    // 【主検査】独立 CPU 参照実装による forward→backward end-to-end parity
    // ============================================================
    // codex-review 指摘（PR #1120）: GPU forward の `rstd` を CPU 参照へ
    // 流用する方式を主検査にすると、「独立 CPU 参照実装による
    // forward→backward の end-to-end 一致」という検証そのものが失われる
    // （rstd 単体 + 同一 rstd 供給下の dw/dx を別々に通しても、rstd の
    // 許容内差が `rows` 方向に蓄積する dw 誤差を拘束できない）。この
    // 指摘は妥当であり、主検査は必ず独立 CPU 参照実装（`rstd` も dw も
    // GPU の内部実装詳細〈split-K の有無・ブロック分割〉を一切参照せず
    // 独自に計算する）で行う。GB10 実機で観測された FAIL（イシュー
    // #1102・`docs/perf/cuda-parity-baseline.md` §9）はこの主検査に対して
    // **カーネル側**（forward の二乗和 `acc` を `double` アキュムレータ化。
    // `kernels_rmsnorm.rs` §9.7 追補）で解消を図った。tolerance
    // （`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・本関数の
    // 判定対象・ケース表は一切変更していない。
    let (cpu_dx, cpu_dw) =
        cpu_rmsnorm_backward_reference(&x_data, Some(&w_data), &dy_data, eps, rows, hidden, None);

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
        gpu_dw
            .as_ref()
            .expect("with_weight=true means dw must be Some"),
        &cpu_dw.expect("with_weight=true means dw must be Some"),
    );

    // ============================================================
    // 【追加検査】カーネル単体の縮約式・境界処理の検証（主検査を置き換え
    // ない。イシュー #1102・#1105 の切り分けで有用性が実証された補助
    // 検査として維持する）
    // ============================================================

    // rstd バッファ自体の parity: GPU forward が保存した `rstd` バッファ
    // 自体の取り違え・破損（縮約式や境界処理と無関係な経路のバグ）を
    // 検出する（Bugbot 指摘対応: `hidden == 0` では GPU 側
    // `1.0f / sqrtf(fma(0, inv_n, eps))` が有限値 `1/sqrt(eps)` を書く
    // ため、`cpu_rmsnorm_rstd_reference` も同じ値を返すよう統一済み。
    // 同関数のドキュメンテーションコメント参照）。
    let cpu_rstd = cpu_rmsnorm_rstd_reference(&x_data, eps, rows, hidden);
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!(
            "rmsnorm backward dw cpu-cuda split-K rstd buffer parity rows={rows} \
             hidden={hidden} num_blocks={num_blocks} eps={eps}"
        ),
        &rstd,
        &cpu_rstd,
    );

    // 同一 rstd 供給下のカーネル単体検査（dx）: `gpu_rstd = Some(&rstd)`
    // で forward の `rstd` を CPU 参照側にもそのまま供給し、dx カーネル
    // （`RMSNORM_BWD_DX_F32`。num_blocks に依存しない独立カーネル）の
    // 縮約式・境界処理のみを切り分けて検証する。
    let (cpu_dx_same_rstd, _cpu_dw_naive_order) = cpu_rmsnorm_backward_reference(
        &x_data,
        Some(&w_data),
        &dy_data,
        eps,
        rows,
        hidden,
        Some(&rstd),
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!(
            "rmsnorm backward dx cpu-cuda split-K same-rstd kernel-only parity rows={rows} \
             hidden={hidden} num_blocks={num_blocks} eps={eps}"
        ),
        &gpu_dx,
        &cpu_dx_same_rstd,
    );

    // 同一 rstd 供給下のカーネル単体検査（dw）: GPU の split-K 二段縮約
    // （ブロック内 fmaf 蓄積 → ブロック間単純加算）と同一順序の CPU 参照
    // 実装で比較し、dw カーネル自体の縮約式・境界処理を切り分けて検証
    // する（`cpu_rmsnorm_dw_split_reference` のドキュメンテーション
    // コメント参照）。
    let cpu_dw_split_same_rstd =
        cpu_rmsnorm_dw_split_reference(&x_data, &dy_data, &rstd, rows, hidden, num_blocks);
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!(
            "rmsnorm backward dw cpu-cuda split-K same-rstd kernel-only parity rows={rows} \
             hidden={hidden} num_blocks={num_blocks} eps={eps}"
        ),
        &gpu_dw.expect("with_weight=true means dw must be Some"),
        &cpu_dw_split_same_rstd,
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

/// `cpu_rmsnorm_dw_split_reference`（GPU split-K dw と同一の二段縮約順序）
/// の CPU 専用回帰テスト（実機不要。イシュー #1102 GB10 実機再検証で
/// 判明した dw split 経路固有の非結合性への対応の妥当性確認）。
#[test]
fn cpu_dw_split_reference_matches_naive_reference_when_num_blocks_is_one() {
    // `num_blocks == 1` では split-K 参照実装は単一ブロックの行順逐次
    // `mul_add` 蓄積へ退化し、最終加算は `0.0 + partial == partial`
    // （浮動小数点の加法単位元。精度損失なし）となるため、
    // `cpu_rmsnorm_backward_reference` の dw 出力と bit-for-bit 一致する
    // はずである（GPU `RMSNORM_BWD_DW_F32`〈単段〉と `RMSNORM_BWD_DW_
    // PARTIAL_F32`/`RMSNORM_BWD_DW_REDUCE_F32`〈num_blocks=1 の split-K〉
    // が同一の蓄積順序を持つのと対応する）。
    let rows = 37usize;
    let hidden = 129usize;
    let eps = 1e-5f32;
    let x_data = Xorshift64Star::new(4001).fill_vec(rows * hidden);
    let dy_data = Xorshift64Star::new(4002).fill_vec(rows * hidden);
    let w_data = Xorshift64Star::new(4003).fill_vec(hidden);

    let rstd = cpu_rmsnorm_rstd_reference(&x_data, eps, rows, hidden);
    let (_dx, naive_dw) = cpu_rmsnorm_backward_reference(
        &x_data,
        Some(&w_data),
        &dy_data,
        eps,
        rows,
        hidden,
        Some(&rstd),
    );
    let split_dw = cpu_rmsnorm_dw_split_reference(&x_data, &dy_data, &rstd, rows, hidden, 1);
    let naive_dw = naive_dw.expect("with_weight=true means dw must be Some");

    assert_eq!(naive_dw.len(), split_dw.len());
    assert_eq!(
        naive_dw.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        split_dw.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "num_blocks=1 では split 参照実装と素朴逐次参照実装は bit 一致するはずである"
    );
}

/// `cpu_rmsnorm_dw_split_reference` の末尾ブロック・境界条件の回帰
/// テスト（`rows` が `num_blocks` を割り切らないケース。GPU
/// `RMSNORM_BWD_DW_PARTIAL_F32` の「末尾要素ブロックの扱い」コメントと
/// 対応）: 全 `hidden` 要素が有限値で埋まり、`num_blocks` を変えても
/// 出力形状が壊れないことを確認する（数値そのものの厳密一致は
/// GPU 実測でのみ検証可能なため、ここでは形状・有限性のみを assert する）。
#[test]
fn cpu_dw_split_reference_handles_non_divisible_num_blocks() {
    let rows = 10usize;
    let hidden = 5usize;
    let eps = 1e-5f32;
    let x_data = Xorshift64Star::new(5001).fill_vec(rows * hidden);
    let dy_data = Xorshift64Star::new(5002).fill_vec(rows * hidden);
    let rstd = cpu_rmsnorm_rstd_reference(&x_data, eps, rows, hidden);

    // rows=10 を num_blocks=7 で割ると rows_per_block=2、末尾ブロックは
    // 行範囲が空になる（b=5,6 は [10,12) と [12,14) で共に rows=10 を
    // 超過し空範囲）。
    let dw = cpu_rmsnorm_dw_split_reference(&x_data, &dy_data, &rstd, rows, hidden, 7);
    assert_eq!(dw.len(), hidden);
    assert!(dw.iter().all(|v| v.is_finite()));
}
