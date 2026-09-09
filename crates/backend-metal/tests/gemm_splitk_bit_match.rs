//! split-K 2 パス GEMM（イシュー #1474。opt-in・`dispatch_auto` へ未結線）
//! の AC-1: 同一入力に対する run-to-run の bit 同一を、split 数
//! 2/4/8/16/32 × `docs/backend-metal-splitk-decision.md` §3 の対象 9 形状
//! （`(32,32,*)`/`(64,64,*)`/`(128,128,*)`。K=2048/4096/8192）で実機確認
//! する受け入れテスト。加えて `MetalGemm::new()`（classic 経路）の
//! 決定性が split-K 実装追加後も非後退のままであることを確認する
//! （`SPLIT_K_ENABLED=false` の既存経路が定数畳み込みで一切変わらない
//! 契約。`shaders/gemm.metal` の該当コメント参照）。
//!
//! いずれのケースも `dispatch_split_k_strided_prepared_with_plan` の
//! 戻り値が [`fandhe_ai_backend_metal::SplitKRoute::Split`] であることを
//! assert し、フォールバック（classic 経路）による自明合格を排除する。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。CI（GitHub
//! ホステッド・ubuntu-latest）では `#![cfg(target_os = "macos")]` により
//! コンパイル対象外になり、`#[ignore]` により通常の `cargo test` からも
//! 除外される（`tests/gemm_fine_barrier_bit_match.rs` と同じ方針）。
//!
//! 実機実行（Apple Silicon 必須）:
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --features internal-diagnostics --test gemm_splitk_bit_match -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_metal::layout::classify_2d;
use fandhe_ai_backend_metal::tile;
use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm, SplitKRoute};

/// AC-1 のビット単位一致検証を `to_bits()` 経由で行う
/// （`tests/gemm_fine_barrier_bit_match.rs::assert_bit_exact` と同じ理由:
/// `assert_eq!` の `f32` 数値比較は `+0.0 == -0.0` を区別できず符号ビットの
/// 差異を見逃しうるため）。
fn assert_bit_exact(run1: &[f32], run2: &[f32], context: &str) {
    let bits1: Vec<u32> = run1.iter().map(|v| v.to_bits()).collect();
    let bits2: Vec<u32> = run2.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        bits1, bits2,
        "{context}: split-K の run-to-run 出力がビット単位で一致しなかった\
         （パーティション昇順の固定順序縮約が破れている疑いがある。\
         shaders/gemm.metal::gemm_splitk_reduce を確認すること）。"
    );
}

/// 対象 9 形状（`docs/backend-metal-splitk-decision.md` §3。
/// `should_split_k` が Some を返す組の一部を用いる）。
const TARGET_SHAPES: &[(usize, usize, usize)] = &[
    (32, 32, 2048),
    (32, 32, 4096),
    (32, 32, 8192),
    (64, 64, 2048),
    (64, 64, 4096),
    (64, 64, 8192),
    (128, 128, 2048),
    (128, 128, 4096),
    (128, 128, 8192),
];

/// `should_split_k` の自動判定した `partitions` に加え、明示指定した
/// 分割数 2/4/8/16/32（`_with_plan`）でも run-to-run bit 一致を確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn split_k_run_to_run_bit_match_for_target_shapes_and_partition_counts() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in TARGET_SHAPES {
        let a = Xorshift64Star::new(0x1000 + m as u64 * 31 + k as u64).fill_vec(m * k);
        let b = Xorshift64Star::new(0x2000 + n as u64 * 37 + k as u64).fill_vec(k * n);
        let a_layout = classify_2d(&[m, k], &[k as isize, 1]).unwrap();
        let b_layout = classify_2d(&[k, n], &[n as isize, 1]).unwrap();

        for &partitions in &[2u32, 4, 8, 16, 32] {
            // K タイル数を `partitions` で割り切れる範囲に丸めた明示計画
            // （`should_split_k` の手順 3〜5 と同じ導出式。AC-1 は
            // `partitions` を横断的に固定検証するための直接指定であり、
            // 自動判定〈`should_split_k`〉のパラメータ探索とは独立）。
            let tile_cfg = tile::split_k_tile(m, n);
            let k_tiles = (k as u32).div_ceil(tile_cfg.bk);
            if partitions < 2 || partitions as u64 > k_tiles as u64 {
                continue;
            }
            let k_per_partition = (k_tiles / partitions).max(1) * tile_cfg.bk;
            let plan = tile::SplitKPlan {
                tile: tile_cfg,
                partitions,
                k_per_partition,
            };

            let mut runs: Vec<Vec<f32>> = Vec::with_capacity(2);
            for run_idx in 0..2 {
                let a_buf = MetalBuffer::new_with_data(&ctx, &a)
                    .expect("A バッファのアップロードに失敗した");
                let b_buf = MetalBuffer::new_with_data(&ctx, &b)
                    .expect("B バッファのアップロードに失敗した");
                let c_buf =
                    MetalBuffer::new_zeroed(&ctx, m * n).expect("C バッファの確保に失敗した");

                let route = gemm
                    .dispatch_split_k_strided_prepared_with_plan(
                        &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf, m, n, k, plan,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "dispatch_split_k_strided_prepared_with_plan failed \
                             (m={m}, n={n}, k={k}, partitions={partitions}, run={run_idx}): {e}"
                        )
                    });
                assert!(
                    matches!(route, SplitKRoute::Split { .. }),
                    "m={m}, n={n}, k={k}, partitions={partitions}: classic 経路へ\
                     フォールバックした（route={route:?}）。フォールバックによる\
                     自明合格を排除するため split-K 経路への到達を要求する。"
                );

                runs.push(c_buf.read_to_vec());
            }

            assert_bit_exact(
                &runs[0],
                &runs[1],
                &format!("m={m}, n={n}, k={k}, partitions={partitions}"),
            );
        }
    }
}

/// split-K 実装（`SPLIT_K_ENABLED` function constant・`SplitKParams`
/// buffer(6) の追加）が classic 経路（`MetalGemm::new()`・`dispatch_auto`）
/// の決定性を後退させていないことを確認する（AC-5 の run-to-run 版。
/// `tests/gemm_fine_barrier_bit_match.rs` と同じ検証観点）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn classic_dispatch_auto_remains_run_to_run_bit_exact_after_split_k_addition() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for size in [512usize, 1024, 2048] {
        let mut rng = Xorshift64Star::new(0xC0FFEE);
        let a = rng.fill_vec(size * size);
        let b = rng.fill_vec(size * size);

        let run1 = gemm
            .dispatch_auto(&ctx, &a, &b, size, size, size)
            .expect("run1 dispatch_auto に失敗した");
        let run2 = gemm
            .dispatch_auto(&ctx, &a, &b, size, size, size)
            .expect("run2 dispatch_auto に失敗した");

        assert_bit_exact(&run1, &run2, &format!("classic dispatch_auto size={size}"));
    }
}
