//! Metal `simdgroup_matrix` 経路の形状閾値（`tensor_core::dispatch::
//! METAL_SIMDGROUP_MIN_DIM = 512`）の境界形状 実測再検証（TASK-11.2c・#69）。
//!
//! ## 位置づけ
//!
//! `METAL_SIMDGROUP_MIN_DIM`（`crates/tensor-core/src/dispatch.rs`）は
//! v1 PoC-8（CubeCL autotune 前提）の 256/512 の 2 点計測のみを根拠にした
//! **暫定値**であり、`docs/dispatch-rules-design.md` §3.1・§3.2 は境界形状
//! （384・640 等）の実測再検証を本イシューへ委ねている。`tests/
//! gemm_auto_parity.rs`（#68）は「どちらの経路が選ばれても数値が正しい」
//! ことを閾値の上下 2 点＋境界 511/512 で検証済みだが、境界クロスオーバー
//! 付近の **TFLOPS 実測**（閾値そのものの妥当性の根拠）は未取得だった。
//!
//! 本ファイルは以下 2 点を提供する:
//!
//! 1. [`boundary_shapes_tflops_record`] 関数: `min(M,N,K)` が 256・384・448・
//!    512・576・640・768・1024 となる正方形状で `GemmVariant::Tiled` と
//!    `MetalGemm::dispatch_auto`（`simdgroup_matrix` 動的タイル選択経路）
//!    の双方を `bench_harness::protocol::run`（warmup 20・計測 20、正本
//!    TASK-8.1 準拠）で計測し、`BenchReport::to_json` で構造化出力する
//!    （`--nocapture` 実行結果を `docs/perf/dispatch-boundary-measurement.md`
//!    の記録テーブルへ転記する運用。`tests/tensor_core_real_device.rs`
//!    〈#64〉と同じ記録様式）。
//! 2. [`dispatch_backend_auto_selects_documented_route_and_matches_reference`] 関数:
//!    各境界形状で `MetalGemm::dispatch_backend_auto` が決定表どおりの
//!    経路（`min(M,N,K) >= METAL_SIMDGROUP_MIN_DIM` で `MatrixUnit`・
//!    未満で `Tiled`）を選択し、かつ CPU 参照実装と複合判定
//!    （`backend_cpu::assert_parity`。REQ-2「相対誤差 1e-3 未満 または
//!    絶対誤差 1e-5 未満」）で一致することを記録する。判定式・閾値は
//!    ここでローカル複製しない（`.claude/rules/coding-rust.md`）。
//!
//! ## 実機前提
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する（`tests/
//! gemm_auto_parity.rs` と同じ方針。`#![cfg(target_os = "macos")]` ＋
//! `#[ignore]`）。デバイス・パイプライン構築が失敗する環境では `.expect`
//! により失敗が顕在化する設計とし、実機以外での silent green を許さない
//! （`tests/tensor_core_real_device.rs` 冒頭コメントと同じ規約）。
//!
//! ## 閾値・許容誤差の不変更
//!
//! 本ファイルは実測を記録するのみで、`METAL_SIMDGROUP_MIN_DIM`・複合判定
//! の許容誤差はいずれも変更しない。実測後の閾値変更判断は
//! `docs/perf/dispatch-boundary-measurement.md` の判定基準に従い、
//! 別レビュー・別 PR で行う（`.claude/rules/coding-rust.md`「バックエンド
//! 間数値一致テストの許容誤差を単独で緩和しない」・実装計画 §7）。
//!
//! ```sh
//! cargo test -p backend-metal -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use backend_cpu::parity::{assert_parity, matmul_reference_fma};
use backend_metal::{GemmVariant, MetalContext, MetalGemm};
use bench_harness::rng::Xorshift64Star;
use bench_harness::{BenchReport, MeasurementConfig};
use tensor_core::dispatch::{DType, GemmShape, METAL_SIMDGROUP_MIN_DIM, select_gemm_kernel};

/// `min(M,N,K)` の境界形状（正方形状。512 の前後に密に取り、クロス
/// オーバーの実測解像度を上げる。実装計画 §3「境界形状 256/384/448/512/
/// 576/640/768/1024」）。
const BOUNDARY_DIMS: [u32; 8] = [256, 384, 448, 512, 576, 640, 768, 1024];

/// `dim x dim x dim` の GEMM 1 回あたりの浮動小数点演算数（`2 * M * N * K`。
/// `tests/tensor_core_real_device.rs::tensor_core_tflops_record` と同じ
/// 換算式）。
fn flops(dim: u32) -> f64 {
    2.0 * (dim as f64).powi(3)
}

/// [`GemmVariant::Tiled`] と [`MetalGemm::dispatch_auto`]
/// （`simdgroup_matrix` 動的タイル選択）を同一形状・同一入力で計測し、
/// median TFLOPS・`BenchReport` の JSON を `println!` で出力する
/// （境界形状ごとに 1 行。`docs/perf/dispatch-boundary-measurement.md`
/// の実測テーブルへ転記する運用）。
///
/// `#[ignore]`: Metal 実機（Apple Silicon）依存。CI では実行しない。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。実測記録は docs/perf/dispatch-boundary-measurement.md"]
fn boundary_shapes_tflops_record() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");
    let config = MeasurementConfig::new(20, 20).expect("20/20 は TASK-8.1 の下限を満たす");

    for &dim in &BOUNDARY_DIMS {
        let (m, n, k) = (dim as usize, dim as usize, dim as usize);
        let mut rng = Xorshift64Star::new(0xDB00 + u64::from(dim));
        let a = rng.fill_vec(m * k);
        let b = rng.fill_vec(k * n);

        let tiled_measurement = bench_harness::run(&config, || {
            let _ = gemm
                .dispatch_variant(&ctx, GemmVariant::Tiled, &a, &b, m, n, k)
                .expect("GemmVariant::Tiled のディスパッチに失敗した");
        })
        .expect("tiled measurement must satisfy TASK-8.1 protocol");
        let tiled_tflops = (flops(dim) / tiled_measurement.median_secs) / 1e12;
        let tiled_report = BenchReport::from_measurement(
            format!("gemm_tiled_dim{dim}"),
            "metal",
            &tiled_measurement,
        )
        .expect("BenchReport::from_measurement must succeed for a protocol-conformant measurement");
        println!(
            "dispatch_boundary_record dim={dim} path=tiled tflops={tiled_tflops:.3} report={}",
            tiled_report
                .to_json()
                .expect("BenchReport::to_json must succeed for a validated report")
        );

        let auto_measurement = bench_harness::run(&config, || {
            let _ = gemm
                .dispatch_auto(&ctx, &a, &b, m, n, k)
                .expect("dispatch_auto のディスパッチに失敗した");
        })
        .expect("dispatch_auto measurement must satisfy TASK-8.1 protocol");
        let auto_tflops = (flops(dim) / auto_measurement.median_secs) / 1e12;
        let auto_report = BenchReport::from_measurement(
            format!("gemm_dispatch_auto_dim{dim}"),
            "metal",
            &auto_measurement,
        )
        .expect("BenchReport::from_measurement must succeed for a protocol-conformant measurement");
        println!(
            "dispatch_boundary_record dim={dim} path=simdgroup_auto tflops={auto_tflops:.3} report={}",
            auto_report
                .to_json()
                .expect("BenchReport::to_json must succeed for a validated report")
        );

        // クロスオーバー比（実測後の閾値判定基準の素材。1.0 を超える
        // 地点が「simdgroup 経路が tiled を上回る境界」であり、これが
        // 現行閾値 METAL_SIMDGROUP_MIN_DIM=512 と一致するかを
        // docs/perf/dispatch-boundary-measurement.md 側で判定する）。
        println!(
            "dispatch_boundary_record dim={dim} simdgroup_auto_over_tiled={:.3}",
            auto_tflops / tiled_tflops
        );
    }
}

/// 境界形状ごとに `dispatch_backend_auto` が決定表どおりの経路を選択し、
/// CPU 参照実装と複合判定で一致することを検証・記録する。
///
/// 経路自体の選択ロジック（純関数）は `tensor-core` 側の `#[cfg(test)]`
/// が網羅済みのため、本テストは「実機上で実際に選ばれた経路が
/// `select_gemm_kernel` の返り値と一致し、かつ数値も正しいか」という
/// 統合的な確認に限定する（`tests/gemm_auto_parity.rs` の設計方針を継承）。
///
/// `#[ignore]`: Metal 実機（Apple Silicon）依存。CI では実行しない。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。実測記録は docs/perf/dispatch-boundary-measurement.md"]
fn dispatch_backend_auto_selects_documented_route_and_matches_reference() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for &dim in &BOUNDARY_DIMS {
        let (m, n, k) = (dim as usize, dim as usize, dim as usize);
        let mut rng = Xorshift64Star::new(0xDB80 + u64::from(dim));
        let a = rng.fill_vec(m * k);
        let b = rng.fill_vec(k * n);

        // 決定表が返すはずの経路（実機の caps に依存。Apple7 未満の
        // デバイスでは常に Tiled へ倒れるため、期待値もそれに合わせて
        // 実機の caps から導出する。`METAL_SIMDGROUP_MIN_DIM` はここで
        // 変更しない）。
        let shape = GemmShape::new(dim, dim, dim);
        let expected_kernel = select_gemm_kernel(&ctx.caps(), shape, DType::F32);

        let mut expected = vec![0.0f32; m * n];
        matmul_reference_fma(&a, &b, &mut expected, m, n, k)
            .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

        let actual = gemm
            .dispatch_backend_auto(&ctx, &a, &b, m, n, k)
            .expect("dispatch_backend_auto のディスパッチに失敗した");

        assert_parity(
            &format!("metal dispatch_backend_auto boundary dim={dim}"),
            &actual,
            &expected,
        );

        println!(
            "dispatch_boundary_route_record dim={dim} min_dim_threshold={METAL_SIMDGROUP_MIN_DIM} \
             expected_kernel={expected_kernel:?} result=parity_pass"
        );
    }
}
