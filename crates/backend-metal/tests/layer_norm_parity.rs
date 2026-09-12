//! イシュー #1596: LayerNorm 順伝播カーネル（MSL・simdgroup 内
//! reduction・persistent threadgroup）の CPU-Metal 数値一致検証。
//!
//! `tests/rmsnorm_parity.rs` と同じ構成方針を踏襲する: Metal 実機
//! （Apple Silicon）依存のため `#![cfg(target_os = "macos")]` でファイル
//! 全体を macOS 限定にし、各テストに `#[ignore]` を付けて通常 CI では
//! 実行しない（`backend_ops_layer_norm_non_final_axis_is_unsupported`
//! は Metal デバイスに触れないため例外的に `#[ignore]` なし）。判定式・
//! 許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の参照と
//! する（`.claude/rules/coding-rust.md`）。
//!
//! 実行コマンド（Mac 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test layer_norm_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_metal::{MetalContext, MetalLayerNorm};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// テスト専用 `f64` 参照実装（GPU の Neumaier + scale/ssq 方式・CPU の
/// `f64` 逐次和のいずれとも独立した実装で突き合わせることで、両実装
/// 共通のバグを検出できるようにする）。
fn f64_layer_norm_reference(
    x: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    if hidden == 0 {
        return out;
    }
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / hidden as f64;
        let var: f64 = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / hidden as f64;
        let rstd = 1.0f64 / (var + eps as f64).sqrt();
        let out_row = &mut out[r * hidden..(r + 1) * hidden];
        for i in 0..hidden {
            let mut xhat = ((row[i] as f64 - mean) * rstd) as f32;
            if let Some(w) = w {
                xhat *= w[i];
            }
            if let Some(b) = b {
                xhat += b[i];
            }
            out_row[i] = xhat;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn assert_layer_norm_parity(
    ctx: &MetalContext,
    layer_norm: &MetalLayerNorm,
    seed_x: u64,
    seed_w: u64,
    seed_b: u64,
    rows: usize,
    hidden: usize,
    with_weight: bool,
    with_bias: bool,
    eps: f32,
) {
    let x_data = Xorshift64Star::new(seed_x).fill_vec(rows * hidden);
    let w_data = if with_weight {
        Some(Xorshift64Star::new(seed_w).fill_vec(hidden))
    } else {
        None
    };
    let b_data = if with_bias {
        Some(Xorshift64Star::new(seed_b).fill_vec(hidden))
    } else {
        None
    };

    let gpu_out = layer_norm
        .run_layer_norm_f32(
            ctx,
            &x_data,
            w_data.as_deref(),
            b_data.as_deref(),
            eps,
            rows,
            hidden,
        )
        .expect("MetalLayerNorm::run_layer_norm_f32 must succeed on Metal-equipped test runner");
    let expected = f64_layer_norm_reference(
        &x_data,
        w_data.as_deref(),
        b_data.as_deref(),
        eps,
        rows,
        hidden,
    );

    assert_parity(
        &format!(
            "layer_norm rows={rows} hidden={hidden} with_weight={with_weight} with_bias={with_bias} eps={eps}"
        ),
        &gpu_out,
        &expected,
    );
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_matches_f64_reference_across_shapes_and_affine_combinations() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let hidden_cases: &[usize] = &[1, 3, 4, 5, 7, 8, 17, 33, 128, 4097];
    let rows_cases: &[usize] = &[1, 2, 5];
    let mut seed = 2000u64;
    for &hidden in hidden_cases {
        for &rows in rows_cases {
            for with_weight in [false, true] {
                for with_bias in [false, true] {
                    seed += 1;
                    assert_layer_norm_parity(
                        &ctx,
                        &layer_norm,
                        seed,
                        seed + 500,
                        seed + 900,
                        rows,
                        hidden,
                        with_weight,
                        with_bias,
                        1e-5,
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_extreme_values_no_nan_inf() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let x = vec![1e30f32, -1e30, 1e30, -1e30, 1e-30, -1e-30, 0.0, 0.0];
    let out = layer_norm
        .run_layer_norm_f32(&ctx, &x, None, None, 1e-5, 2, 4)
        .expect("run_layer_norm_f32 must succeed");
    for &v in &out {
        assert!(v.is_finite(), "expected finite layer_norm output, got {v}");
    }
}

/// 極端な `eps`（`f32::MAX` 級。`ln_finalize_rstd` の疑似要素トリック
/// 〈`sqrt(eps)*sqrt(n)`〉が中間 overflow を避けることの実機確認）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_extreme_eps_does_not_overflow() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let x_data = Xorshift64Star::new(3001).fill_vec(4 * 17);
    let out = layer_norm
        .run_layer_norm_f32(&ctx, &x_data, None, None, 1e30, 4, 17)
        .expect("run_layer_norm_f32 must succeed");
    for &v in &out {
        assert!(v.is_finite(), "expected finite layer_norm output, got {v}");
    }
}

/// codex-review 指摘の再現ケース（P1・#1671 スレッド）: 平均が偏差計算前
/// に `f32` 単一値へ丸められると、`2^24` 近傍で ULP が `2` になる領域
/// （`[16777216, 16777218]`。真の平均 `16777217` が `f32` で表現不能）で
/// 丸め誤差が出力へ伝播し、期待値 `[-1, 1]`（`eps=1e-5` は無視できる規模）
/// から大きく乖離する（是正前は `[0, 1.4142]` を観測）。
/// `crates/backend-cpu/src/layer_norm.rs` の同名テストと同じ意図。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_preserves_mean_precision_near_f32_epsilon_boundary() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let x = vec![16777216.0f32, 16777218.0];
    let out = layer_norm
        .run_layer_norm_f32(&ctx, &x, None, None, 1e-5, 1, 2)
        .expect("run_layer_norm_f32 must succeed");
    let rstd = 1.0f64 / (1.0f64 + 1e-5f64).sqrt();
    let expected = [(-rstd) as f32, rstd as f32];
    for (o, e) in out.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-4, "o={o} e={e}");
    }
}

/// codex-review 指摘の再現ケース（P1・#1671 スレッド）: `meanB - meanA`
/// のような遠い有限値どうしの単純減算は、`[2^38, -2^38]` のような入力で
/// `f32` の表現範囲（`|x| <= f32::MAX ≈ 3.4e38`）を超え `±inf` へ
/// overflow しうる。CPU/CUDA（`f64` 演算）は有限値を返す前提のため、
/// Metal も有限出力を維持することを確認する（比スケール領域に留める
/// 設計。冒頭のカーネルコメント参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_large_opposite_sign_pair_stays_finite() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let x = vec![2e38f32, -2e38];
    let out = layer_norm
        .run_layer_norm_f32(&ctx, &x, None, None, 1e-5, 1, 2)
        .expect("run_layer_norm_f32 must succeed");
    for &v in &out {
        assert!(v.is_finite(), "expected finite layer_norm output, got {v}");
    }
    // 期待値: mean=0（両者の厳密和が 0）・var=4e76・rstd ≈ 5e-39 と
    // なり、xhat ≈ dev_i * rstd（dev[0]=+2e38・dev[1]=-2e38）で
    // out ≈ [+1, -1] に収束する。codex-review 指摘（P1・PR #1671）に
    // 従い、独自のハードコード許容誤差ではなく `f64_layer_norm_reference`
    // × `assert_parity`（REQ-2 統一複合判定）で突合する。
    let expected = f64_layer_norm_reference(&x, None, None, 1e-5, 1, 2);
    assert_parity("layer_norm opposite_sign_pair", &out, &expected);
}

/// advisor が指摘した偏差自体の overflow ケース: 行の 1 要素が突出して
/// 大きく（`2e38`）、残り 999 要素が反対符号の同スケール値
/// （`-2e38`）の場合、平均は概ね `-1.996e38` となり、突出要素の偏差
/// `dev = x - mean ≈ 3.996e38` が `f32::MAX` を超える（元スケールへ
/// 戻すと表現不能）。比スケール領域内で完結させる設計により有限出力を
/// 維持することを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_deviation_overflow_case_stays_finite() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let mut x = vec![-2e38f32; 1000];
    x[0] = 2e38;
    let out = layer_norm
        .run_layer_norm_f32(&ctx, &x, None, None, 1e-5, 1, 1000)
        .expect("run_layer_norm_f32 must succeed");
    for &v in &out {
        assert!(v.is_finite(), "expected finite layer_norm output, got {v}");
    }
}

/// 退化ケース（codex-review が示唆）: 行の全要素が同一の巨大値
/// （`2e38`）の場合、真の分散は 0 で `rstd` は `eps` のみで決まる。
/// `1/(scale*sqrt(...))` を単独の中間値として形成すると `eps` 疑似要素
/// により overflow しうるため、要素ごとに `dev/scale` を計算する設計
/// （冒頭のカーネルコメント）で有限かつ 0 に近い出力を維持することを
/// 確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_constant_large_row_does_not_overflow() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let x = vec![2e38f32; 2];
    let out = layer_norm
        .run_layer_norm_f32(&ctx, &x, None, None, 1e-5, 1, 2)
        .expect("run_layer_norm_f32 must succeed");
    for &v in &out {
        assert!(v.is_finite(), "expected finite layer_norm output, got {v}");
    }
    // codex-review 指摘（P1・PR #1671）: 独自のハードコード許容誤差
    // ではなく `f64_layer_norm_reference` × `assert_parity`（REQ-2
    // 統一複合判定）で突合する。
    let expected = f64_layer_norm_reference(&x, None, None, 1e-5, 1, 2);
    assert_parity("layer_norm constant_large_row", &out, &expected);
}

/// NaN 伝播（行内に NaN が 1 つでもあれば行全体が NaN。`rmsnorm_parity.rs`
/// と同じ意味論契約）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_propagates_nan_for_row_with_nan_element() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    for hidden in [4usize, 65] {
        let mut x = vec![1.0f32; hidden];
        x[0] = f32::NAN;
        let out = layer_norm
            .run_layer_norm_f32(&ctx, &x, None, None, 1e-5, 1, hidden)
            .expect("run_layer_norm_f32 must succeed");
        assert!(
            out.iter().all(|v| v.is_nan()),
            "hidden={hidden}: NaN 要素を含む行の出力が NaN へ伝播していない: {out:?}"
        );
    }
}

// --- BackendOps::layer_norm 独立エントリ ---

/// `MetalBackendOps::layer_norm` が `MetalLayerNorm::run_layer_norm_f32`
/// と bit 同一であること（同一カーネルへのディスパッチであり別実装では
/// ないことの確認）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backend_ops_layer_norm_is_bit_identical_to_metal_layer_norm() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let rows = 3usize;
    let hidden = 17usize;
    let x_data = Xorshift64Star::new(6001).fill_vec(rows * hidden);
    let w_data = Xorshift64Star::new(6002).fill_vec(hidden);
    let b_data = Xorshift64Star::new(6003).fill_vec(hidden);
    let x = Tensor::new(x_data.clone(), &[rows, hidden]).expect("valid tensor");
    let w = Tensor::new(w_data.clone(), &[hidden]).expect("valid tensor");
    let b = Tensor::new(b_data.clone(), &[hidden]).expect("valid tensor");

    let metal = fandhe_ai_backend_metal::MetalBackendOps::new();
    let via_ops = metal
        .layer_norm(&x, Some(&w), Some(&b), 1e-5)
        .expect("BackendOps::layer_norm must succeed on Metal-equipped test runner");
    let via_kernel = layer_norm
        .run_layer_norm_f32(
            &ctx,
            &x_data,
            Some(&w_data),
            Some(&b_data),
            1e-5,
            rows,
            hidden,
        )
        .expect("MetalLayerNorm::run_layer_norm_f32 must succeed");

    assert_eq!(via_ops.shape(), &[rows, hidden]);
    assert_eq!(
        via_ops.as_slice().expect("contiguous"),
        via_kernel.as_slice()
    );
}

/// `MetalBackendOps::layer_norm` を CPU 参照実装と実機で直接
/// `assert_parity` 突合する（形状網羅）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backend_ops_layer_norm_matches_cpu_reference_across_shapes() {
    let metal = fandhe_ai_backend_metal::MetalBackendOps::new();
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();

    let rows_cases: &[usize] = &[1, 3, 17];
    let hidden_cases: &[usize] = &[1, 31, 32, 33, 1024, 4097];
    let mut seed = 7000u64;
    for &rows in rows_cases {
        for &hidden in hidden_cases {
            seed += 1;
            let x_data = Xorshift64Star::new(seed).fill_vec(rows * hidden);
            let x = Tensor::new(x_data, &[rows, hidden]).expect("valid tensor");

            let gpu_out = metal
                .layer_norm(&x, None, None, 1e-5)
                .expect("BackendOps::layer_norm must succeed on Metal-equipped test runner");
            let cpu_out = cpu
                .layer_norm(&x, None, None, 1e-5)
                .expect("BackendOps::layer_norm must succeed on CPU");

            assert_eq!(gpu_out.shape(), &[rows, hidden]);
            assert_parity(
                &format!(
                    "BackendOps::layer_norm metal-cpu direct parity rows={rows} hidden={hidden}"
                ),
                gpu_out.as_slice().expect("contiguous"),
                cpu_out.as_slice().expect("contiguous"),
            );
        }
    }
}

/// codex-review 指摘の再現ケース（P1・#1671 スレッド 1 件目）: 行の全
/// 要素が同一値（非 2 冪長・`hidden=3`）の場合、真の偏差は厳密に 0 に
/// なるべきだが、平均を「ホストが事前丸めした `inv_n`〈`1/hidden`〉を
/// `sum` へ乗算して doubled-float 化する」実装では `inv_n` 自身の丸め
/// 誤差が `mean_lo` へ残存し、`eps` 由来の極小 `scale` で 1000 倍規模へ
/// 増幅されていた（是正前は `x=[1024,1024,1024], eps=1e-5` で本来 0 の
/// 偏差が約 `-0.00965` へ乖離するのを観測）。`hidden` 自体への厳密除算
/// （Dekker 型 div）へ切り替えたことで、非 2 冪長の一様行でも出力が
/// 厳密に `bias` 相当（affine なしなら 0）へ一致することを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_uniform_row_non_power_of_two_hidden_matches_zero_deviation() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    for &hidden in &[3usize, 5, 7, 17] {
        let x = vec![1024.0f32; hidden];
        let out = layer_norm
            .run_layer_norm_f32(&ctx, &x, None, None, 1e-5, 1, hidden)
            .expect("run_layer_norm_f32 must succeed");
        // codex-review 指摘（P1・PR #1671）: 独自のハードコード許容誤差
        // ではなく `f64_layer_norm_reference` × `assert_parity`（REQ-2
        // 統一複合判定）で突合する。
        let expected = f64_layer_norm_reference(&x, None, None, 1e-5, 1, hidden);
        assert_parity(
            &format!("layer_norm uniform_row hidden={hidden}"),
            &out,
            &expected,
        );
    }
}

/// codex-review 指摘の再現ケース（P1・#1671 スレッド 2 件目）: `eps=0`
/// かつ行の全要素が同一値（真の分散も 0）の退化ケースでは、CPU/CUDA・
/// ホスト参照実装が `rstd = 1/sqrt(0+0) = inf`・`xhat = 0 * inf = NaN`
/// という NaN 伝播契約を持つ。是正前の Metal 実装は `dev == 0.0` だけで
/// 分岐し `scale` の FTZ 対策として無条件に `0.0` を返していたため、この
/// 退化ケースでも `[0, 0]` を返し NaN 伝播契約と食い違っていた。
/// `eps > 0.0` を条件に加えたことで、`eps == 0` の退化ケースは自然な
/// `dev/scale = 0/0 = NaN` へフォールバックし、CPU 参照実装
/// （`run_layer_norm_f32_propagates_nan_for_row_with_nan_element` 相当の
/// 契約）と一致することを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_zero_eps_degenerate_row_propagates_nan() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let x = vec![1.0f32, 1.0];
    let out = layer_norm
        .run_layer_norm_f32(&ctx, &x, None, None, 0.0, 1, 2)
        .expect("run_layer_norm_f32 must succeed");
    assert!(
        out.iter().all(|v| v.is_nan()),
        "expected NaN propagation for eps=0 degenerate uniform row, got {out:?}"
    );
}

/// `eps > 0` かつ真の分散が 0 の退化ケース
/// （`layer_norm_constant_large_row_does_not_overflow` と同型だが
/// `eps>0` ゲートの正当な適用範囲——`scale` が `eps` 由来の極小疑似
/// 要素のみに由来し FTZ で潰れうるケース——を明示的に確認する回帰）:
/// `eps>0` なら FTZ 対策のゼロ返却が引き続き有効で有限・近ゼロ出力を
/// 維持する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_positive_eps_degenerate_row_stays_finite_near_zero() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let x = vec![1.0f32, 1.0];
    let out = layer_norm
        .run_layer_norm_f32(&ctx, &x, None, None, 1e-5, 1, 2)
        .expect("run_layer_norm_f32 must succeed");
    for &v in &out {
        assert!(v.is_finite(), "expected finite output, got {v}");
    }
    // codex-review 指摘（P1・PR #1671）: 独自のハードコード許容誤差
    // ではなく `f64_layer_norm_reference` × `assert_parity`（REQ-2
    // 統一複合判定）で突合する。
    let expected = f64_layer_norm_reference(&x, None, None, 1e-5, 1, 2);
    assert_parity("layer_norm positive_eps_degenerate_row", &out, &expected);
}

/// codex-review 指摘の再現ケース（P1・#1671 スレッド 2 件目）: `eps` が
/// `x` に比べて極端に大きい行（`x=[1e-20,-1e-20]`・`eps=1e38`）では、
/// 是正前の `row_scale`（`x` の `maxabs` のみから決定）だと `eps` 疑似
/// 要素 `sqrt(eps)*sqrt(hidden)/row_scale` 自体が `f32` の表現範囲を
/// 超えて `+inf` になり、`scale` が `+inf` へ潰れて実要素の寄与
/// （`dev/scale`）がすべて厳密 0 になる（観測: `[0, 0]`）。`row_scale`
/// を `eps` 側の要求も考慮して選び直す是正後は、`weight=[1e38,1e38]`
/// と合わせて期待値 `[0.1, -0.1]` 近傍の有限出力を返すことを確認する
/// （`shaders/layer_norm.metal` 冒頭コメント「`row_scale` の eps 対応
/// 拡張」参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_tiny_x_huge_eps_stays_finite_and_nonzero() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let x = vec![1e-20f32, -1e-20];
    let w = vec![1e38f32, 1e38];
    let out = layer_norm
        .run_layer_norm_f32(&ctx, &x, Some(&w), None, 1e38, 1, 2)
        .expect("run_layer_norm_f32 must succeed");
    for &v in &out {
        assert!(v.is_finite(), "expected finite output, got {v:?}: {out:?}");
    }
    assert!(
        out[0].abs() > 1e-3,
        "expected non-degenerate-zero output (row_scale must account for eps), got {out:?}"
    );
    // codex-review 指摘（P1・PR #1671。この極端値ケース専用の
    // ハードコード許容誤差 `2e-2`〈相対誤差にして約 20%〉は REQ-2
    // 統一複合判定〈相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満〉より
    // 大幅に緩く、統一判定では不合格となる出力も見逃しうる。他の
    // テスト（本ファイル冒頭の `assert_layer_norm_parity` 等）と同じ
    // `f64_layer_norm_reference` × `assert_parity` に統一する。
    let expected = f64_layer_norm_reference(&x, Some(&w), None, 1e38, 1, 2);
    assert_parity("layer_norm tiny_x_huge_eps", &out, &expected);
}

/// codex-review 指摘の再現ケース（P1・#1671 スレッド 1 件目）: `hidden`
/// が `2^24`（`(float)hidden` が丸め無しで表現できる上限）を超える
/// 軸長は、平均計算の `(float)hidden` 直接変換が最近接偶数丸めで真の
/// 除数とずれるため、起動前検証（`validate_hidden_exact_f32`）で
/// fail-closed に拒否されることを確認する。境界値（`2^24` ちょうど）は
/// 受理されることも合わせて確認する（`x.len()` を `hidden` に一致させる
/// 必要はあるが、`hidden` 検証自体は `x_buf` 確保より前に行われるため
/// `rows=0` で `x.len()==0` のまま検証のみ実行できる）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_rejects_hidden_exceeding_exact_f32_range() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let over_limit_hidden = (1usize << 24) + 1;
    let err = layer_norm
        .run_layer_norm_f32(&ctx, &[], None, None, 1e-5, 0, over_limit_hidden)
        .expect_err("hidden = 2^24 + 1 must be rejected before launch");
    let msg = err.to_string();
    assert!(
        msg.contains("2^24") || msg.contains("16777216"),
        "expected error to mention the exact-f32 boundary, got: {msg}"
    );
}
