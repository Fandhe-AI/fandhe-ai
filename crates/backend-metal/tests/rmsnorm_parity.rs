//! イシュー #604: 融合 RMSNorm 順伝播カーネル（MSL・warp 内 reduction・
//! persistent threadgroup）の CPU-Metal 数値一致検証。
//!
//! `tests/cpu_metal_parity.rs`（GEMM）と同じ構成方針を踏襲する: Metal 実機
//! （Apple Silicon）依存のため `#![cfg(target_os = "macos")]` でファイル
//! 全体を macOS 限定にし、各テストに `#[ignore]` を付けて通常 CI では
//! 実行しない。判定式・許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の
//! 参照とする（`.claude/rules/coding-rust.md`）。
//!
//! CPU 参照実装は本ファイル内のテスト専用関数（`f32::mul_add` 使用。CUDA
//! 側 `tests/rmsnorm_parity.rs::cpu_rmsnorm_reference` と同一意味論）で
//! ある。
//!
//! 実行コマンド（Mac 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test rmsnorm_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_metal::{MetalContext, MetalRmsNorm};

/// テスト専用 CPU 参照実装（`f32::mul_add` を使用し、GPU 側 `fma()` と
/// 丸め方針を揃える）。`out = x * rsqrt(mean(x^2, axis=-1) + eps) * w`
/// （`w` が `None` の場合は乗算をスキップ）。CUDA 側
/// `backend-cuda::tests::rmsnorm_parity::cpu_rmsnorm_reference` と同一
/// 意味論（実装計画 §6.2「CPU 参照は CUDA 側 `cpu_rmsnorm_reference` と
/// 同一意味論」）。
fn cpu_rmsnorm_reference(
    x: &[f32],
    w: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    if hidden == 0 {
        return out;
    }
    let inv_n = 1.0f32 / hidden as f32;
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mut acc = 0.0f32;
        for &v in row {
            acc = v.mul_add(v, acc);
        }
        let rstd = 1.0f32 / (acc.mul_add(inv_n, eps)).sqrt();
        let out_row = &mut out[r * hidden..(r + 1) * hidden];
        for i in 0..hidden {
            let mut normed = row[i] * rstd;
            if let Some(w) = w {
                normed *= w[i];
            }
            out_row[i] = normed;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn assert_rmsnorm_parity(
    ctx: &MetalContext,
    rmsnorm: &MetalRmsNorm,
    seed_x: u64,
    seed_w: u64,
    rows: usize,
    hidden: usize,
    with_weight: bool,
    eps: f32,
) {
    let x_data = Xorshift64Star::new(seed_x).fill_vec(rows * hidden);
    let w_data = if with_weight {
        Some(Xorshift64Star::new(seed_w).fill_vec(hidden))
    } else {
        None
    };

    let gpu_out = rmsnorm
        .run_rmsnorm_f32(ctx, &x_data, w_data.as_deref(), eps, rows, hidden)
        .expect("MetalRmsNorm::run_rmsnorm_f32 must succeed on Metal-equipped test runner");
    let cpu_out = cpu_rmsnorm_reference(&x_data, w_data.as_deref(), eps, rows, hidden);

    assert_eq!(gpu_out.len(), cpu_out.len());
    assert_parity(
        &format!(
            "rmsnorm cpu-metal parity rows={rows} hidden={hidden} with_weight={with_weight} \
             eps={eps}"
        ),
        &gpu_out,
        &cpu_out,
    );
}

/// 実機必須の形状網羅（受け入れ条件の本体。実装計画 §6.2「形状網羅」）。
///
/// hidden の網羅: 8（極小・4 の倍数）・9（極小・非倍数）・1024（1 パス
/// 中位）・4096（1 パス上限〈`ONEPASS_MAX_HIDDEN`〉ちょうど）・4097（2 パス
/// 強制・非倍数）・8192（2 パス）。rows の網羅: 1・3・33（persistent grid
/// を超えて行ループを複周回させる）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn rmsnorm_matches_cpu_across_shapes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let rmsnorm = MetalRmsNorm::new(&ctx).expect("RMSNorm パイプラインの構築に失敗した");

    let hiddens: &[usize] = &[8, 9, 1024, 4096, 4097, 8192];
    let rows_cases: &[usize] = &[1, 3, 33];

    let mut seed = 1000u64;
    for &hidden in hiddens {
        for &rows in rows_cases {
            for &with_weight in &[false, true] {
                seed += 1;
                let seed_w = seed + 500;
                assert_rmsnorm_parity(
                    &ctx,
                    &rmsnorm,
                    seed,
                    seed_w,
                    rows,
                    hidden,
                    with_weight,
                    1e-5,
                );
            }
        }
    }

    // eps=0.0（`run_fused` 経由の canonical プランと同じ eps。有限値の
    // 境界ケース）。
    assert_rmsnorm_parity(&ctx, &rmsnorm, 9001, 9002, 4, 256, false, 0.0);

    // hidden=1（行長 1。ベクトル化経路〈hidden % 4 == 0〉に入らない
    // 最小ケース）。
    assert_rmsnorm_parity(&ctx, &rmsnorm, 9101, 9102, 5, 1, false, 1e-5);
}

/// `run_fused`（`ops.rs::MetalBackendOps::run_fused`）経由の canonical
/// RMSNorm プラン実行を CPU per-op 合成（`mul → sum → rsqrt → broadcast →
/// mul`）と突き合わせる（実装計画 §6.2「run_fused 経路の実機テスト」）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn rmsnorm_run_fused_matches_cpu_composed() {
    use fandhe_ai_tensor_core::{BackendOps, DType, FusedOpKind, FusionPlan, Tensor};

    let hidden = 16usize;
    let x_data = Xorshift64Star::new(9101).fill_vec(hidden);
    let x = Tensor::new(x_data.clone(), &[hidden]).expect("valid tensor");

    let ops = vec![
        FusedOpKind::Input { leaf_index: 0 },
        FusedOpKind::Mul { lhs: 0, rhs: 0 },
        FusedOpKind::Sum {
            input: 1,
            axis: None,
        },
        FusedOpKind::Rsqrt { input: 2 },
        FusedOpKind::Broadcast {
            input: 3,
            axis: None,
        },
        FusedOpKind::Mul { lhs: 4, rhs: 0 },
    ];
    let plan = FusionPlan::from_ops(ops, vec![hidden], DType::F32, 1)
        .expect("canonical RMSNorm plan must construct");

    let metal = fandhe_ai_backend_metal::MetalBackendOps::new();
    let fused_out = metal
        .run_fused(&plan, &[&x])
        .expect("MetalBackendOps::run_fused must succeed on Metal-equipped test runner");

    let sq: Vec<f32> = x_data.iter().map(|v| v * v).collect();
    let sum: f32 = sq.iter().sum();
    let rstd = 1.0f32 / sum.sqrt();
    let composed: Vec<f32> = x_data.iter().map(|v| v * rstd).collect();

    assert_eq!(fused_out.shape(), &[hidden]);
    assert_parity(
        "rmsnorm run_fused vs cpu composed (canonical plan)",
        fused_out.as_slice().expect("contiguous"),
        &composed,
    );
}

/// CPU-Metal 直接突合（イシュー #607）: `fandhe_ai_backend_cpu::rmsnorm::
/// run_rmsnorm_f32`（NEON/rayon 参照実装）を GPU 出力と直接比較する。
/// 実機必須（`#[ignore]`。CI ではコンパイルのみ）。既存の
/// `cpu_rmsnorm_reference`（テスト専用ローカル参照実装）と数学的に同一
/// だが、本テストは実クレート API の呼び出し経路自体を検証する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn rmsnorm_matches_backend_cpu_directly() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let rmsnorm = MetalRmsNorm::new(&ctx).expect("RMSNorm パイプラインの構築に失敗した");

    let rows = 3usize;
    let hidden = 4097usize; // NEON 端要素を含む。
    let eps = 1e-5f32;
    let x_data = Xorshift64Star::new(41_001).fill_vec(rows * hidden);
    let w_data = Xorshift64Star::new(41_002).fill_vec(hidden);

    let gpu_out = rmsnorm
        .run_rmsnorm_f32(&ctx, &x_data, Some(&w_data), eps, rows, hidden)
        .expect("MetalRmsNorm::run_rmsnorm_f32 must succeed on Metal-equipped test runner");
    let cpu_out =
        fandhe_ai_backend_cpu::rmsnorm::run_rmsnorm_f32(&x_data, Some(&w_data), eps, rows, hidden)
            .expect("fandhe_ai_backend_cpu::rmsnorm::run_rmsnorm_f32 must succeed");

    assert_parity(
        "rmsnorm cpu(backend_cpu)-metal direct parity",
        &gpu_out,
        &cpu_out,
    );
}

/// 極値入力（有限だが `f32` の二乗が overflow する大きさ・非正規化数級の
/// 極小値）での CPU-Metal parity（イシュー #1102。codex-review 指摘・
/// PR #1120: 単純な Kahan 補償和は `v.x * v.x` を `f32` のまま先に計算
/// するため、有限入力〈例 `2e20f`〉でも二乗が overflow して `inf` になり、
/// Kahan 補償計算が `inf - inf` で `NaN` を生んでいた。scale/ssq 方式
/// 〈`rmsnorm.metal::rmsnorm_ssq_add` 等。overflow-safe な二乗和
/// アルゴリズム〉でこれを解消したことを実機で確認する）。CPU 参照は
/// `fandhe_ai_backend_cpu::rmsnorm::run_rmsnorm_f32`（本 PR で `f64`
/// アキュムレータ化済み。同じ極値でも二乗和が overflow しない）を使う。
/// tolerance は変更していない。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn rmsnorm_matches_backend_cpu_with_extreme_magnitude_values() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let rmsnorm = MetalRmsNorm::new(&ctx).expect("RMSNorm パイプラインの構築に失敗した");

    // hidden=17（非 4 の倍数。スカラー経路を含む）で、要素の一部を
    // 2e20f（f32 の二乗〈4e40〉が f32 表現範囲〈最大約 3.4e38〉を超えて
    // overflow する大きさ）・1e-38f（非正規化数級の極小値）・通常範囲の
    // 値の混在にする。
    let rows = 2usize;
    let hidden = 17usize;
    let eps = 1e-5f32;
    let mut x_data = Xorshift64Star::new(51_001).fill_vec(rows * hidden);
    // 行 0: 先頭 2 要素を極大値へ差し替える。
    x_data[0] = 2e20f32;
    x_data[1] = -1.5e20f32;
    // 行 1: 非正規化数級の極小値を先頭 2 要素へ混在させる。
    x_data[hidden] = 1e-38f32;
    x_data[hidden + 1] = -3e-39f32;

    let gpu_out = rmsnorm
        .run_rmsnorm_f32(&ctx, &x_data, None, eps, rows, hidden)
        .expect("MetalRmsNorm::run_rmsnorm_f32 must succeed on Metal-equipped test runner");
    assert!(
        gpu_out.iter().all(|v| v.is_finite()),
        "GPU 出力に非有限値（NaN/inf）が含まれている（scale/ssq 方式への \
         回帰の可能性）: {gpu_out:?}"
    );

    let cpu_out = fandhe_ai_backend_cpu::rmsnorm::run_rmsnorm_f32(&x_data, None, eps, rows, hidden)
        .expect("fandhe_ai_backend_cpu::rmsnorm::run_rmsnorm_f32 must succeed");

    assert_parity(
        "rmsnorm cpu(backend_cpu)-metal extreme magnitude parity",
        &gpu_out,
        &cpu_out,
    );
}

/// `eps` が極端に大きい場合（`f32::MAX` 級）の CPU-Metal parity（イシュー
/// #1102。codex-review 指摘・PR #1120 1 件目: `eps` を疑似要素として
/// `scale/ssq` へ折り込む際に `sqrt(eps * n)` を素朴に計算すると、`eps`
/// が `f32::MAX` 級・`hidden` がある程度大きい場合に `eps * n` 自体が
/// `f32` の表現範囲を超えて `inf` になり、`rstd` が誤って `0` になって
/// いた〈CPU/CUDA は `f64` で `sum_sq*inv_n + eps` を計算するため有限〉。
/// `sqrt(eps) * sqrt(n)` への変形〈`rmsnorm.metal::rmsnorm_finalize_rstd`〉
/// でこの中間 overflow を解消したことを実機で確認する）。tolerance は
/// 変更していない。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn rmsnorm_matches_backend_cpu_with_extreme_eps() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let rmsnorm = MetalRmsNorm::new(&ctx).expect("RMSNorm パイプラインの構築に失敗した");

    // hidden=17（非 4 の倍数。スカラー経路を含む）・eps=f32::MAX。通常
    // 範囲の x に対し eps が支配的になり、真の rstd は極めて小さい有限値
    // （1/sqrt(eps) 程度）になる。
    let rows = 2usize;
    let hidden = 17usize;
    let eps = f32::MAX;
    let x_data = Xorshift64Star::new(52_001).fill_vec(rows * hidden);

    let gpu_out = rmsnorm
        .run_rmsnorm_f32(&ctx, &x_data, None, eps, rows, hidden)
        .expect("MetalRmsNorm::run_rmsnorm_f32 must succeed on Metal-equipped test runner");
    assert!(
        gpu_out.iter().all(|v| v.is_finite()),
        "GPU 出力に非有限値（NaN/inf）が含まれている（eps*n の中間 overflow への \
         回帰の可能性）: {gpu_out:?}"
    );
    assert!(
        gpu_out.iter().all(|v| *v != 0.0f32),
        "GPU 出力が全てゼロになっている（rstd が誤って 0 になる回帰の可能性）: {gpu_out:?}"
    );

    let cpu_out = fandhe_ai_backend_cpu::rmsnorm::run_rmsnorm_f32(&x_data, None, eps, rows, hidden)
        .expect("fandhe_ai_backend_cpu::rmsnorm::run_rmsnorm_f32 must succeed");

    assert_parity(
        "rmsnorm cpu(backend_cpu)-metal extreme eps parity",
        &gpu_out,
        &cpu_out,
    );
}

/// `NaN` 要素を含む行の CPU-Metal parity（イシュー #1102。codex-review
/// 指摘・PR #1120 2 件目: `rmsnorm_ssq_add` の `a > scale` 比較は `NaN`
/// で常に偽になり、`scale` が未だ `0.0f`（このレーンで最初の要素が
/// `NaN` だった場合）だと寄与が黙って捨てられ非有限値が伝播しなかった。
/// CPU/CUDA は `f64` 逐次和のため `NaN` を含む行全体が `NaN` になる意味論
/// であり、Metal も同じ意味論へ揃えたことを実機で確認する）。`NaN` を
/// 含む行と含まない行を混在させ、`NaN` 要素の位置（先頭・非先頭）も
/// 変えて検証する。tolerance は変更していない（`NaN` 同士は
/// `assert_parity` の複合判定では比較できないため、本テストは
/// `is_nan()`／`is_finite()` の直接検査で判定する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn rmsnorm_propagates_nan_matching_backend_cpu() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let rmsnorm = MetalRmsNorm::new(&ctx).expect("RMSNorm パイプラインの構築に失敗した");

    let rows = 3usize;
    let hidden = 17usize; // 非 4 の倍数。スカラー経路を含む。
    let eps = 1e-5f32;
    let mut x_data = Xorshift64Star::new(53_001).fill_vec(rows * hidden);
    // 行 0: 先頭要素（このレーンで最初に処理される要素）が NaN
    // （codex-review が指摘した scale==0 のまま捨てられるケースの直撃）。
    x_data[0] = f32::NAN;
    // 行 1: 非先頭要素が NaN（scale が既に非ゼロになった後の NaN 遭遇）。
    x_data[hidden + 3] = f32::NAN;
    // 行 2: NaN なし（対照。他行への非伝播を確認する）。

    let gpu_out = rmsnorm
        .run_rmsnorm_f32(&ctx, &x_data, None, eps, rows, hidden)
        .expect("MetalRmsNorm::run_rmsnorm_f32 must succeed on Metal-equipped test runner");
    let cpu_out = fandhe_ai_backend_cpu::rmsnorm::run_rmsnorm_f32(&x_data, None, eps, rows, hidden)
        .expect("fandhe_ai_backend_cpu::rmsnorm::run_rmsnorm_f32 must succeed");

    for row in 0..2 {
        let gpu_row = &gpu_out[row * hidden..(row + 1) * hidden];
        let cpu_row = &cpu_out[row * hidden..(row + 1) * hidden];
        assert!(
            gpu_row.iter().all(|v| v.is_nan()),
            "行 {row}: NaN 要素を含む行の GPU 出力が NaN へ伝播していない: {gpu_row:?}"
        );
        assert!(
            cpu_row.iter().all(|v| v.is_nan()),
            "行 {row}: NaN 要素を含む行の CPU 出力が NaN へ伝播していない（テスト前提 \
             崩れ）: {cpu_row:?}"
        );
    }
    let row2_start = 2 * hidden;
    let gpu_row2 = &gpu_out[row2_start..row2_start + hidden];
    assert!(
        gpu_row2.iter().all(|v| v.is_finite()),
        "NaN を含まない行（行 2）の GPU 出力に非有限値が混入している（他行への非伝播が \
         破れている）: {gpu_row2:?}"
    );
    assert_parity(
        "rmsnorm cpu(backend_cpu)-metal nan-free row parity",
        gpu_row2,
        &cpu_out[row2_start..row2_start + hidden],
    );
}
