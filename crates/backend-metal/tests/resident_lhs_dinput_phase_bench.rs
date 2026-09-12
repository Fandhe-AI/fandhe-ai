//! `Op::LinearResident` の VJP（`crates/autodiff/src/grad.rs`）が
//! `d_input` を求める際に呼ぶ `MetalBackendOps::gemm_resident_lhs`
//! （`crates/backend-metal/src/ops.rs`）の GPU dispatch 部分
//! （`MetalGemm::dispatch_strided_bias_act_prepared` = encode +
//! `ctx.synchronize()`〈同期・GPU 完了待ち〉のみ。readback〈`read_to_vec`
//! 等〉は含まず、呼び出し元〈`gemm_resident_lhs`〉が同期後に別途行う）
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
//! （転置 view のゼロコピーアップロード。`ops.rs` 内 private 関数）
//! そのものは経由しないが、**その結果として実際に GEMM カーネルへ渡る
//! `MatrixLayout` は本番と同一にする**（イシュー #1562 codex-review
//! 是正）。production では `b = g_t = transpose2d(g)`（`g: [r, q]`
//! 行優先・`g_t: [q, r]`）を `layout::classify_2d` に通すと
//! `strides == [1, q]`・`sc(=q) >= rows(=q)` により
//! `MatrixLayout { rows: q, cols: r, ld: q, transposed: true }`
//! （転置＝strided ロード経路）を返す（`layout.rs:103-110`）。本ファイルは
//! `g` の物理行優先ストレージ（`[r, q]`）そのものを `MetalBuffer::
//! new_with_data` で直接アップロードし、`b_layout` を上記と同じ
//! `{ rows: q, cols: r, ld: q, transposed: true }` に設定することで、
//! `upload_operand_for_resident_gemm` を経由しない**簡略レプリカ**
//! ながら、GEMM カーネルが実際に読むロード経路（NT strided）を本番と
//! 一致させる（値そのものは無関係な乱数。目的は同一形状・同一
//! ロードパターン・同一 dispatch 呼び出し列でのコスト分解であり、
//! 数値正しさの検証ではない。`assert_parity` 等の複合判定は行わない）。
//!
//! # Variant A（現状相当）・Variant B（回収余地の上限測定）
//!
//! - Variant A: `encode_strided_bias_act_prepared`（encode）→
//!   `ctx.synchronize()`（同期＝GPU 完了待ち）→ `read_to_vec`
//!   （readback）→ 転置 view 作成（`transpose` 区間）の 4 区間を
//!   `Instant` で分解する（現状の `dispatch_strided_bias_act_prepared`
//!   全体に相当）。**転置 view 作成区間は production の
//!   `d_input = transpose2d(tmp)`（`grad.rs`）と同じ `Tensor::transpose`
//!   （`tensor-core::tensor.rs:388-406`。`storage` を `Arc::clone` する
//!   だけの zero-copy stride view）を実際に呼び、戻り値を
//!   `std::hint::black_box` で消費するだけの O(1) 操作である（旧版は
//!   ここで `p*r` 要素すべてをコピーする独自ホスト転置
//!   `transpose_row_major` を計測しており、production のコストを
//!   過大に見積もっていた。イシュー #1562 codex-review 是正。回収余地の
//!   本体は GEMM 本体＋同期＋readback であり、転置 view 作成自体は
//!   計測誤差レベルの寄与しかない）。
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
//! **同一 GPU 資源競合の回避（イシュー #1562 codex-review 是正）**:
//! 本ファイルの 2 テスト（L1・L2）はそれぞれ独立の `MetalContext::new()`
//! を持つが、同一物理 GPU を奪い合うと互いの `synchronize()` 計測へ
//! 資源競合が混入しうる。`cargo test` の既定（マルチスレッド並列実行）
//! ではなく **`--test-threads=1` で直列化して実行する**（下記コマンド例・
//! `orchestrate.sh`・README とも統一）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --test resident_lhs_dinput_phase_bench -- --ignored --nocapture --test-threads=1
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
//!   --test resident_lhs_dinput_phase_bench -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use std::hint::black_box;
use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_metal::layout::MatrixLayout;
use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm};
use fandhe_ai_tensor_core::Tensor;

/// パイプライン初回コンパイル・プール未使用（本ファイルは
/// `alloc_uninit_pooled` を経由しない生成のためプール自体は関与しないが、
/// MSL パイプラインの初回コンパイルコストを計測対象から除く）ための
/// 捨て試行数。
const WARMUP: usize = 5;

/// 5 回計測中央値方針（`.claude/rules/coding-rust.md`）。
const TRIALS: usize = 5;

/// Variant A（4 区間分解）・Variant B（encode-only）双方の 1 形状分の
/// 5 回計測中央値を求めて出力する。
#[allow(clippy::too_many_arguments)]
fn run_case(label: &str, p: usize, q: usize, r: usize, seed_a: u64, seed_b: u64) {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    let a_data = Xorshift64Star::new(seed_a).fill_vec(p * q);
    // `g_data` は production の `g: [r, q]`（行優先。VJP upstream 勾配）の
    // 物理ストレージそのものに相当する（`grad.rs` の `g_t = transpose2d(g)`
    // は zero-copy view のため、物理バッファの内容・要素数は `g` と同一）。
    // 要素数は `q * r`（`= r * q`）で変わらないが、値の意味づけが
    // 「NN 配置の `g_t`」から「`g` の物理行優先データ」へ変わる点が
    // イシュー #1562 codex-review 是正の要点（下記 `b_layout` 参照）。
    let g_data = Xorshift64Star::new(seed_b).fill_vec(r * q);
    let a_buf =
        MetalBuffer::new_with_data(&ctx, &a_data).expect("A バッファのアップロードに失敗した");
    let b_buf =
        MetalBuffer::new_with_data(&ctx, &g_data).expect("B バッファのアップロードに失敗した");
    let zero_c = vec![0.0f32; p * r];

    let a_layout = MatrixLayout {
        rows: p,
        cols: q,
        ld: q,
        transposed: false,
    };
    // production が実際に `dispatch_strided_bias_act_prepared` へ渡す
    // `b_layout` と同一（`layout::classify_2d` が `g_t: [q, r]`・
    // `strides == [1, q]` を分類した結果。`docs/matmul-vjp-zero-copy-
    // decision.md` §4.4）。`transposed: false`・`ld: r`（NN 配置）ではなく
    // `transposed: true`・`ld: q`（strided／転置ロード経路）にすることで、
    // 本ベンチが実際に本番と同じカーネル内ロードパターンを計測する
    // （イシュー #1562 codex-review 是正。旧版は NN 配置で計測しており
    // 本番の NT 配置と異なるロード経路を測っていた）。
    let b_layout = MatrixLayout {
        rows: q,
        cols: r,
        ld: q,
        transposed: true,
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
    // 各試行の encode+sync+readback 合算値（イシュー #1562 codex-review
    // 是正）。区間ごとの中央値の和 `q_encode.median + q_sync.median +
    // q_readback.median` は、各区間の最遅試行が揃って同一試行で起きるとは
    // 限らないため、5 試行の合計時間の中央値と一般に一致しない
    // （区間中央値の和は試行合計の中央値の上界にも下界にもならない）。
    // ここでは試行ごとに 3 区間を合算した配列を作り、その配列へ
    // `median_q1_q3` を適用することで「5 試行の合計時間の中央値」を
    // 直接求める。
    let mut trial_total_secs = Vec::with_capacity(TRIALS);
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
        let readback_elapsed = t_readback.elapsed().as_secs_f64();
        readback_secs.push(readback_elapsed);

        // この試行の encode+sync+readback 合算（試行内で対応する 3 区間の
        // 実測値のみを足し合わせる。区間別配列の中央値同士を足す旧実装は
        // 試行間の対応関係を失っていた）。
        trial_total_secs.push(
            *encode_secs.last().expect("この試行の encode_secs")
                + *sync_secs.last().expect("この試行の sync_secs")
                + readback_elapsed,
        );

        // production の `d_input = transpose2d(tmp)`（`grad.rs`）と同じ
        // `Tensor::transpose`（zero-copy stride view。`storage` を
        // `Arc::clone` するだけで要素コピーを行わない）を実際に呼び、
        // その O(1) 操作自体のコストを計測する（イシュー #1562
        // codex-review 是正。旧版は `raw` の全要素をコピーする独自の
        // ホスト転置を計測しており、view 作成のコストを過大評価していた）。
        // `tmp: [p, r]` 行優先データから `Tensor::new` で構築し
        // `.transpose(0, 1)` する（`transpose2d` は `Tensor::transpose(0,1)`
        // への薄い委譲。`grad.rs::transpose2d` 参照）。
        let t_transpose = Instant::now();
        let tmp_tensor =
            Tensor::new(raw, &[p, r]).expect("tmp Tensor の構築（p*r 要素）に失敗した");
        let transposed = tmp_tensor
            .transpose(0, 1)
            .expect("transpose2d と同じ 2 軸転置 view 作成に失敗した");
        black_box(&transposed);
        transpose_secs.push(t_transpose.elapsed().as_secs_f64());
        assert_eq!(
            transposed.shape(),
            &[r, p],
            "transpose view の shape は [r, p] のはず（transpose2d と同一定義）"
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
    // 区間別中央値の和ではなく、試行ごとの合算値配列に `median_q1_q3` を
    // 適用した「5 試行の合計時間の中央値」（イシュー #1562 codex-review
    // 是正）。
    let q_trial_total =
        median_q1_q3(&trial_total_secs).expect("trial_total_secs の分位点計算に失敗した");

    let variant_a_total_median = q_trial_total.median;
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
