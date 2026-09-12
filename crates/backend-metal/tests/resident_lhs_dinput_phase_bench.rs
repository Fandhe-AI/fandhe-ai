//! `Op::LinearResident` の VJP（`crates/autodiff/src/grad.rs`）が
//! `d_input` を求める際に呼ぶ `MetalBackendOps::gemm_resident_lhs`
//! （`crates/backend-metal/src/ops.rs`）の GPU dispatch 部分
//! （`MetalGemm::dispatch_strided_bias_act_prepared` = encode +
//! `ctx.synchronize()`〈同期・GPU 完了待ち〉+ `read_to_vec`〈readback〉）
//! を隔離環境で phase 分解する診断ベンチ（親イシュー #1557・子イシュー
//! #1561 の最初の子イシュー #1562）。
//!
//! # 背景・位置づけ
//!
//! `docs/backend-metal-command-batching-design.md` §7.2（#1555）は
//! 「`d_input` の GEMM（`BackendOps::gemm` → `dispatch_strided_bias_act_prepared`
//! → 内部 `ctx.synchronize()` → `download`）は resident 化の対象外
//! （回収は部分的）」と明記して次イシューへ引き継いでいた。本ファイルは
//! その `d_input` 経路の同期・readback コストを、本番のプロセスワイド
//! singleton `MetalContext`（診断カウンタを持つ）を経由せず、**自前の
//! `MetalContext::new()`** で隔離計測する（`gemm_naive_parity.rs`・
//! `gemm_splitk_auto_wiring.rs` と同型のパターン。singleton の
//! バッチング機構〈forward・SGD と共有バッチ〉の影響を受けない代わりに、
//! 本番 in-situ での「他 dispatch との重なりを除いた真の増分」は
//! 測れない——本ファイルは上限目安を出すに留まる。詳細は
//! `docs/backend-metal-command-batching-design.md` §7.3「隔離計測の
//! 限界の明記」参照）。
//!
//! # 対象形状（2 層 MLP MNIST 規模。`mnist_scale_train_reuse_bench.rs` と同一）
//!
//! `grad.rs::vjp` の `Op::LinearResident` 分岐: `w: [k, n]`（resident
//! weight）・`g: [m, n]`（upstream 勾配）・`g_t = transpose2d(g): [n, m]`・
//! `tmp = gemm_resident_lhs(w, g_t): [k, m]`（内部では `w` を
//! `(p, q) = (k, n)` として扱い `gemm(w[p,q], g_t[q,r]) -> [p,r]`。
//! `r = m`）・`d_input = transpose2d(tmp): [m, k]`。
//!
//! - L1: `w=[784,256]` → `(p,q)=(784,256)`、`g_t=[256,64]` → `r=64`
//!   （`tmp=[784,64]` → `d_input=[64,784]`）
//! - L2: `w=[256,10]` → `(p,q)=(256,10)`、`g_t=[10,64]` → `r=64`
//!   （`tmp=[256,64]` → `d_input=[64,256]`）
//!
//! 本ファイルは production の `upload_operand_for_resident_gemm`
//! （転置 view のゼロコピーアップロード。`ops.rs` 内 private 関数）を
//! 経由せず、`MetalBuffer::new_with_data` で `g_t` 相当の行優先データを
//! 直接アップロードする**簡略レプリカ**である（値そのものは無関係な
//! 乱数。目的は同一形状・同一 dispatch 呼び出し列でのコスト分解であり、
//! 数値正しさの検証ではない。`assert_parity` 等の複合判定は行わない）。
//!
//! # Variant A（現状相当）・Variant B（回収余地の上限測定）
//!
//! - Variant A: `encode_strided_bias_act_prepared`（encode）→
//!   `ctx.synchronize()`（同期＝GPU 完了待ち）→ `read_to_vec`
//!   （readback）→ ホスト側転置（`transpose_row_major`）の 4 区間を
//!   `Instant` で分解する（現状の `dispatch_strided_bias_act_prepared`
//!   全体に相当）。
//! - Variant B: `encode_strided_bias_act_prepared` のみを計測し、
//!   同期・readback を計測窓の外に出す（次 trial への影響を避けるため
//!   計測外で `ctx.synchronize()` する）。`(Variant A の
//!   encode+sync+readback) - (Variant B の encode)` が「同期・readback
//!   を排除できた場合に理論上回収できる時間」の上限目安になる。
//!
//! 5 回計測中央値（`.claude/rules/coding-rust.md`）。本番コード
//! （`crates/autodiff`・`crates/backend-metal/src/{ops,grad}.rs` 等）は
//! 変更しない（記録専用・non-gating）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --test resident_lhs_dinput_phase_bench -- --ignored --nocapture
//! ```
//!
//! `internal-diagnostics` feature を有効にすると、Variant A の同期区間を
//! `MetalContext::synchronize_with_gpu_timestamps`（イシュー #1259 で
//! 公開化済み）へ差し替え、GPU カーネル専有時間（`kernel_gpu_secs`）と
//! host 側純粋待ち時間を追加で分離出力する（`docs/perf/metal-gemm-reuse-
//! phase-breakdown.md` と同じ手法の横展開）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --features internal-diagnostics \
//!   --test resident_lhs_dinput_phase_bench -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_metal::layout::MatrixLayout;
use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm};

/// パイプライン初回コンパイル・プール未使用（本ファイルは
/// `alloc_uninit_pooled` を経由しない生成のためプール自体は関与しないが、
/// MSL パイプラインの初回コンパイルコストを計測対象から除く）ための
/// 捨て試行数。
const WARMUP: usize = 5;

/// 5 回計測中央値方針（`.claude/rules/coding-rust.md`）。
const TRIALS: usize = 5;

/// `data`（`rows * cols` の行優先データ）を転置した `cols * rows` の
/// 行優先データを返す。`grad::vjp` の `transpose2d`（`Tensor<f32>` 専用・
/// autodiff クレート内 private）の簡略ホスト側レプリカ
/// （`backend-metal` は `autodiff` に依存しないため独自実装する。
/// 数値定義は同一: `out[c * rows + r] = data[r * cols + c]`）。
fn transpose_row_major(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Variant A（4 区間分解）・Variant B（encode-only）双方の 1 形状分の
/// 5 回計測中央値を求めて出力する。
#[allow(clippy::too_many_arguments)]
fn run_case(label: &str, p: usize, q: usize, r: usize, seed_a: u64, seed_b: u64) {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    let a_data = Xorshift64Star::new(seed_a).fill_vec(p * q);
    let b_data = Xorshift64Star::new(seed_b).fill_vec(q * r);
    let a_buf =
        MetalBuffer::new_with_data(&ctx, &a_data).expect("A バッファのアップロードに失敗した");
    let b_buf =
        MetalBuffer::new_with_data(&ctx, &b_data).expect("B バッファのアップロードに失敗した");
    let zero_c = vec![0.0f32; p * r];

    let a_layout = MatrixLayout {
        rows: p,
        cols: q,
        ld: q,
        transposed: false,
    };
    let b_layout = MatrixLayout {
        rows: q,
        cols: r,
        ld: r,
        transposed: false,
    };

    // steady-state 到達（MSL パイプライン初回コンパイルコストの除去）。
    for _ in 0..WARMUP {
        let c_buf = MetalBuffer::new_with_data(&ctx, &zero_c).expect("C バッファの確保に失敗した");
        gemm.dispatch_strided_bias_act_prepared(
            &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, None, false, &c_buf, p, r, q,
        )
        .expect("warmup dispatch に失敗した");
    }

    // Variant A: encode / synchronize / readback / transpose の 4 区間。
    let mut encode_secs = Vec::with_capacity(TRIALS);
    let mut sync_secs = Vec::with_capacity(TRIALS);
    let mut readback_secs = Vec::with_capacity(TRIALS);
    let mut transpose_secs = Vec::with_capacity(TRIALS);
    #[cfg(feature = "internal-diagnostics")]
    let mut kernel_gpu_secs: Vec<f64> = Vec::with_capacity(TRIALS);

    for _ in 0..TRIALS {
        let c_buf = MetalBuffer::new_with_data(&ctx, &zero_c).expect("C バッファの確保に失敗した");

        let t_encode = Instant::now();
        gemm.encode_strided_bias_act_prepared(
            &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, None, false, &c_buf, p, r, q,
        )
        .expect("encode_strided_bias_act_prepared に失敗した");
        encode_secs.push(t_encode.elapsed().as_secs_f64());

        let t_sync = Instant::now();
        #[cfg(not(feature = "internal-diagnostics"))]
        {
            ctx.synchronize().expect("synchronize に失敗した");
        }
        #[cfg(feature = "internal-diagnostics")]
        {
            let batches = ctx
                .synchronize_with_gpu_timestamps()
                .expect("synchronize_with_gpu_timestamps に失敗した");
            // 本隔離コンテキストは他 dispatch を挟まないため、ちょうど
            // 1 バッチ・1 ラベルのはず（混入検知）。
            if let Some(secs) = batches.iter().find_map(|b| b.kernel_gpu_secs()) {
                kernel_gpu_secs.push(secs);
            }
        }
        sync_secs.push(t_sync.elapsed().as_secs_f64());

        let t_readback = Instant::now();
        let raw = c_buf.read_to_vec();
        readback_secs.push(t_readback.elapsed().as_secs_f64());

        let t_transpose = Instant::now();
        let transposed = transpose_row_major(&raw, p, r);
        transpose_secs.push(t_transpose.elapsed().as_secs_f64());
        assert_eq!(
            transposed.len(),
            p * r,
            "transpose_row_major の出力長は入力と一致するはず"
        );
    }

    // Variant B: encode-only（同期・readback を計測窓の外に出す）。
    // 次 trial への影響回避のため、計測直後（窓の外）で drain する。
    let mut encode_only_secs = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let c_buf = MetalBuffer::new_with_data(&ctx, &zero_c).expect("C バッファの確保に失敗した");
        let t0 = Instant::now();
        gemm.encode_strided_bias_act_prepared(
            &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, None, false, &c_buf, p, r, q,
        )
        .expect("encode_strided_bias_act_prepared（Variant B）に失敗した");
        encode_only_secs.push(t0.elapsed().as_secs_f64());
        // 計測窓の外での drain（次 trial のバッチ肥大化防止。計測対象は
        // encode 呼び出し自体のみ）。
        ctx.synchronize().expect("drain synchronize に失敗した");
    }

    let q_encode = median_q1_q3(&encode_secs).expect("encode_secs の分位点計算に失敗した");
    let q_sync = median_q1_q3(&sync_secs).expect("sync_secs の分位点計算に失敗した");
    let q_readback = median_q1_q3(&readback_secs).expect("readback_secs の分位点計算に失敗した");
    let q_transpose = median_q1_q3(&transpose_secs).expect("transpose_secs の分位点計算に失敗した");
    let q_encode_only =
        median_q1_q3(&encode_only_secs).expect("encode_only_secs の分位点計算に失敗した");

    let variant_a_total_median = q_encode.median + q_sync.median + q_readback.median;
    let recoverable_upper_bound = variant_a_total_median - q_encode_only.median;

    println!(
        "[resident_lhs_dinput_phase_bench::{label}] p={p} q={q} r={r} (n={TRIALS}) \
         variant_a: encode={:.6}ms sync={:.6}ms readback={:.6}ms transpose={:.6}ms \
         total(encode+sync+readback)={:.6}ms | variant_b: encode_only={:.6}ms | \
         recoverable_upper_bound(A_total - B_encode)={:.6}ms — record only, \
         non-gating（`docs/backend-metal-command-batching-design.md` §7.3）",
        q_encode.median * 1e3,
        q_sync.median * 1e3,
        q_readback.median * 1e3,
        q_transpose.median * 1e3,
        variant_a_total_median * 1e3,
        q_encode_only.median * 1e3,
        recoverable_upper_bound * 1e3,
    );

    #[cfg(feature = "internal-diagnostics")]
    if let Ok(q_kernel) = median_q1_q3(&kernel_gpu_secs) {
        println!(
            "[resident_lhs_dinput_phase_bench::{label}] kernel_gpu (GPUEndTime-GPUStartTime) \
             median={:.6}ms — sync 区間のうち純粋な GPU カーネル専有時間の内訳 \
             （残りは host 側の waitUntilCompleted 待ち・ドライバオーバーヘッド）",
            q_kernel.median * 1e3,
        );
    }
}

/// L1（`w=[784,256]`・`g_t=[256,64]`）の `d_input` GEMM 単体フェーズ分解。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dinput_phase_l1_784x256x64() {
    run_case("l1_784x256x64", 784, 256, 64, 0x1562_0001, 0x1562_0002);
}

/// L2（`w=[256,10]`・`g_t=[10,64]`）の `d_input` GEMM 単体フェーズ分解。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dinput_phase_l2_256x10x64() {
    run_case("l2_256x10x64", 256, 10, 64, 0x1562_0003, 0x1562_0004);
}
