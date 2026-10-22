//! m が小さい GEMM（MLP 784→256→10・batch 64 級）の job 統合ゲート（イシュー #3198）。
//!
//! 役割: `TwoDDynamic`（(mc, nc) 2D 動的分配）の `jobs_per_worker` を、仕事量の小さい形状に
//! 限って [`SMALL_M_JOBS_PER_WORKER`] へ下げ、job 数（= A の再 pack 回数・job ごとの確保回数・
//! C のコピーイン／アウト回数）を減らす。ゲート [`SMALL_M_JOB_MERGE_ENABLED`] は既定 OFF で、
//! OFF のとき [`jobs_per_worker_for`] は常に `TWO_D_JOBS_PER_WORKER` を返し、現行と同一の挙動になる。
//!
//! 呼び出し元: `gemm_blis_parallel_with_transpose`・`gemm_blis_bias_act_parallel`（いずれも
//! `TWO_D_DYNAMIC_PRODUCTION_ENABLED` 側の `dispatch_two_d_dynamic` 呼び出し）。
//!
//! bit 一致の根拠: `jobs_per_worker` は job の大きさを決める性能パラメータにすぎず、各 C 要素は
//! 常に 1 job に属し、pc 昇順・p 昇順の同じ FMA 連鎖で計算される（2D 設計
//! `docs/cpu-gemm-2d-dynamic-partition-design.md` §3 条件 6）。FMA 契約・tolerance は不変。
//!
//! #3199（同じゲートの下）: 適格形状（[`select_shared_a`]）では、行帯ごとに A を 1 回だけ
//! pack して job 間で共有し、job ごとの確保（`c_local`・A／B パネル）を worker ごとの
//! スクラッチ使い回しへ寄せる。`col_bands == 1` では C の staging も省く。結果は現行経路と
//! bit 完全一致する。設計・採否・A/B の扱いは
//! `docs/cpu-gemm-small-m-job-merge-prepack-decision.md` を正とする。

use super::{NR_CLAMP, TWO_D_JOBS_PER_WORKER};

/// job 統合の本番ゲート（既定 OFF。#3201 の A/B 後に #3227 が反転を判断する）。
pub(super) const SMALL_M_JOB_MERGE_ENABLED: bool = false;

/// job 統合の対象にする仕事量の上限（`m * max(n, NR_CLAMP) * k` がこれ未満）。
/// `small_shape_thread_cap::SMALL_SHAPE_CAP_MAX_WORK`（#1575）と同じ境界で、学習の最大仕事量
/// 12,845,056 を含む最小の 2 のべき乗。
pub(super) const SMALL_M_MAX_WORK: usize = 1 << 24;

/// ゲート ON かつ対象形状のときの `jobs_per_worker`（根拠は #3177 §4 の jpw=1 A/B）。
pub(super) const SMALL_M_JOBS_PER_WORKER: usize = 1;

/// 純関数版。`enabled` を引数に取り、ゲート状態に依らずテストできる。
/// `m < 2`（gemv 経路）・仕事量が上限以上・乗算が飽和する形状は対象外（既定値を返す）。
pub(super) fn select_jobs_per_worker(
    enabled: bool,
    m: usize,
    n: usize,
    k: usize,
    default_jpw: usize,
) -> usize {
    if !enabled || m < 2 {
        return default_jpw;
    }
    let work = m.saturating_mul(n.max(NR_CLAMP)).saturating_mul(k);
    if work < SMALL_M_MAX_WORK {
        SMALL_M_JOBS_PER_WORKER
    } else {
        default_jpw
    }
}

/// 本番入口が `dispatch_two_d_dynamic` へ渡す `jobs_per_worker` を決める。
pub(super) fn jobs_per_worker_for(m: usize, n: usize, k: usize) -> usize {
    select_jobs_per_worker(SMALL_M_JOB_MERGE_ENABLED, m, n, k, TWO_D_JOBS_PER_WORKER)
}

/// A 共有バッファの上限語数（1 MiB。`align_up(m, mr) * k <= これ` を適格とする。仕事量の境界が
/// `<` なのと違い `<=`）。dW 形状（m=784 級）や k が極端に長い形状では共有をやめる。
pub(super) const SMALL_M_SHARE_A_MAX_WORDS: usize = 1 << 18;

/// A 共有経路の適格判定（純関数）。次をすべて満たすとき true:
/// `enabled`・`m >= 2`・仕事量 `< SMALL_M_MAX_WORK`・`m <= blocks_mc`（ic ループが 1 周）・
/// `align_up(m, mr) * k <= SMALL_M_SHARE_A_MAX_WORDS`（乗算が溢れる形状は false）。
pub(super) fn select_shared_a(
    enabled: bool,
    m: usize,
    n: usize,
    k: usize,
    mr: usize,
    blocks_mc: usize,
) -> bool {
    if !enabled || m < 2 || mr == 0 || m > blocks_mc {
        return false;
    }
    let work = m.saturating_mul(n.max(NR_CLAMP)).saturating_mul(k);
    if work >= SMALL_M_MAX_WORK {
        return false;
    }
    m.div_ceil(mr)
        .checked_mul(mr)
        .and_then(|rows| rows.checked_mul(k))
        .is_some_and(|words| words <= SMALL_M_SHARE_A_MAX_WORDS)
}

#[cfg(test)]
mod tests {
    use super::super::{
        GemmTranspose, default_blocks, gemm_blis_bias_act_parallel,
        gemm_blis_parallel_two_d_dynamic_shared_a_with_params,
        gemm_blis_parallel_two_d_dynamic_with_params, gemm_blis_parallel_with_transpose,
        install_in_pool_for_test, partition,
    };
    use super::*;
    use fandhe_ai_tensor_core::Activation;

    /// (m, n, k, 転置種別) の対象 6 形状（MLP 784→256→10・batch 64。#3177）。
    const SHAPES: [(usize, usize, usize, GemmTranspose); 6] = [
        (64, 256, 784, GemmTranspose::Nn),
        (784, 256, 64, GemmTranspose::Tn),
        (64, 784, 256, GemmTranspose::Nt),
        (64, 10, 256, GemmTranspose::Nn),
        (256, 10, 64, GemmTranspose::Tn),
        (64, 256, 10, GemmTranspose::Nt),
    ];

    #[test]
    fn gate_is_off_by_default() {
        // 実行時 assert にする理由（#3201）: `const { assert! }` だとゲートを反転した計測ツリーで
        // lib テスト自体がコンパイル不能（E0080）になり report テストを実行できない。
        // main（OFF）での固定という意味は変えない。
        assert!(!std::hint::black_box(SMALL_M_JOB_MERGE_ENABLED));
    }

    #[test]
    fn off_keeps_default_jobs_per_worker() {
        for &(m, n, k, _) in &SHAPES {
            assert_eq!(jobs_per_worker_for(m, n, k), TWO_D_JOBS_PER_WORKER);
        }
        assert_eq!(jobs_per_worker_for(512, 512, 512), TWO_D_JOBS_PER_WORKER);
    }

    #[test]
    fn on_selects_merge_only_for_small_work() {
        let d = TWO_D_JOBS_PER_WORKER;
        for &(m, n, k, _) in &SHAPES {
            assert_eq!(select_jobs_per_worker(true, m, n, k, d), 1, "{m}x{n}x{k}");
        }
        assert_eq!(select_jobs_per_worker(true, 512, 512, 512, d), d);
        // 境界: work == 1 << 24 は対象外、直下は対象
        assert_eq!(select_jobs_per_worker(true, 2, 8, 1 << 20, d), d);
        assert_eq!(select_jobs_per_worker(true, 2, 8, (1 << 20) - 1, d), 1);
        // 飽和: オーバーフロー形状は対象外
        assert_eq!(
            select_jobs_per_worker(true, usize::MAX, usize::MAX, 2, d),
            d
        );
        // m = 1（gemv 経路）は対象外
        assert_eq!(select_jobs_per_worker(true, 1, 256, 784, d), d);
        // OFF は常に既定
        assert_eq!(select_jobs_per_worker(false, 64, 256, 784, d), d);
    }

    fn rand_vec(seed: u32, len: usize, special: bool) -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                if special && i % 97 == 5 {
                    [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][(i / 97) % 3]
                } else {
                    (s as f32 / u32::MAX as f32) * 2.0 - 1.0
                }
            })
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn assert_bit_equal(m: usize, n: usize, k: usize, t: GemmTranspose, special: bool) {
        let a = rand_vec(0x1234_5678 ^ m as u32, m * k, special);
        let b = rand_vec(0x9abc_def1 ^ n as u32, k * n, special);
        for threads in [1usize, 10, 20] {
            let (c_prod, c_merge) = install_in_pool_for_test(threads, || {
                let mut c_prod = vec![0.0f32; m * n];
                gemm_blis_parallel_with_transpose(&a, &b, &mut c_prod, m, n, k, t).unwrap();
                let mut c_merge = vec![0.0f32; m * n];
                gemm_blis_parallel_two_d_dynamic_with_params(
                    &a,
                    &b,
                    &mut c_merge,
                    m,
                    n,
                    k,
                    default_blocks(),
                    SMALL_M_JOBS_PER_WORKER,
                    t,
                )
                .unwrap();
                (c_prod, c_merge)
            });
            assert_eq!(
                bits(&c_prod),
                bits(&c_merge),
                "{m}x{n}x{k} {t:?} T={threads} special={special}"
            );
        }
    }

    #[test]
    fn jpw1_is_bit_identical_to_production_on_target_shapes() {
        for &(m, n, k, t) in &SHAPES {
            assert_bit_equal(m, n, k, t, false);
            assert_bit_equal(m, n, k, t, true);
        }
    }

    #[test]
    fn jpw1_is_bit_identical_on_fractional_shapes() {
        for &(m, n, k) in &[(63usize, 257usize, 785usize), (65, 10, 9)] {
            for t in [GemmTranspose::Nn, GemmTranspose::Nt, GemmTranspose::Tn] {
                assert_bit_equal(m, n, k, t, false);
                assert_bit_equal(m, n, k, t, true);
            }
        }
    }

    #[test]
    fn bias_act_entry_matches_jpw1_plus_epilogue_on_l1_forward() {
        let (m, n, k) = (64usize, 256usize, 784usize);
        let a = rand_vec(7, m * k, false);
        let b = rand_vec(11, k * n, false);
        let bias = rand_vec(13, n, false);
        let mut c_prod = vec![0.0f32; m * n];
        gemm_blis_bias_act_parallel(&a, &b, &mut c_prod, m, n, k, Some(&bias), Activation::Relu)
            .unwrap();
        let mut c_merge = vec![0.0f32; m * n];
        gemm_blis_parallel_two_d_dynamic_with_params(
            &a,
            &b,
            &mut c_merge,
            m,
            n,
            k,
            default_blocks(),
            SMALL_M_JOBS_PER_WORKER,
            GemmTranspose::Nn,
        )
        .unwrap();
        for (i, v) in c_merge.iter_mut().enumerate() {
            *v = (*v + bias[i % n]).max(0.0);
        }
        assert_eq!(bits(&c_prod), bits(&c_merge));
    }

    // ---- #3199: A 共有経路 ----

    #[test]
    fn select_shared_a_off_is_always_false() {
        for &(m, n, k, _) in &SHAPES {
            assert!(!select_shared_a(false, m, n, k, 8, 128));
        }
    }

    #[test]
    fn select_shared_a_boundaries() {
        // m = MC は true、MC + 1 は false。
        assert!(select_shared_a(true, 128, 16, 64, 8, 128));
        assert!(!select_shared_a(true, 129, 16, 64, 8, 128));
        // align_up(m, mr) * k == 1 << 18 は true、超過は false（mr=8・mr=6）。
        assert!(select_shared_a(true, 64, 8, (1 << 18) / 64, 8, 128));
        assert!(!select_shared_a(true, 64, 8, (1 << 18) / 64 + 1, 8, 128));
        // mr=6: align_up(64,6)=66 → k=3971 で 66*3971=262086 <= 262144、k=3972 で超過。
        assert!(select_shared_a(true, 64, 8, 3971, 6, 128));
        assert!(!select_shared_a(true, 64, 8, 3972, 6, 128));
        // 仕事量 == 1 << 24 は false。
        assert!(!select_shared_a(true, 2, 8, 1 << 20, 8, 128));
        // m = 1 は false。
        assert!(!select_shared_a(true, 1, 256, 784, 8, 128));
        // 飽和・オーバーフロー形状は false。
        assert!(!select_shared_a(true, 64, usize::MAX, usize::MAX, 8, 128));
        assert!(!select_shared_a(true, 64, 8, usize::MAX, 8, 128));
    }

    #[test]
    fn select_shared_a_on_six_shapes() {
        for (i, &(m, n, k, _)) in SHAPES.iter().enumerate() {
            for mr in [6usize, 8] {
                // 0・2・3・5 番（m=64）は適格、dW の 1・4 番（m>=256）は不適格。
                let expected = !matches!(i, 1 | 4);
                assert_eq!(
                    select_shared_a(true, m, n, k, mr, 128),
                    expected,
                    "{m}x{n}x{k} mr={mr}"
                );
            }
        }
    }

    fn init_c(len: usize) -> Vec<f32> {
        rand_vec(0x0bad_cafe ^ len as u32, len, false)
    }

    /// 共有 A 強制入口（jpw 指定）を、本番入口（OFF）と #3198 腕（staging あり）の両方と bit 比較する。
    fn assert_shared_bit_equal(
        m: usize,
        n: usize,
        k: usize,
        t: GemmTranspose,
        special: bool,
        blocks: crate::gemm::BlockSizes,
        threads_list: &[usize],
    ) {
        let a = rand_vec(0x1234_5678 ^ m as u32, m * k, special);
        let b = rand_vec(0x9abc_def1 ^ n as u32, k * n, special);
        let c0 = init_c(m * n);
        for &threads in threads_list {
            install_in_pool_for_test(threads, || {
                let mut c_arm = c0.clone();
                gemm_blis_parallel_two_d_dynamic_with_params(
                    &a,
                    &b,
                    &mut c_arm,
                    m,
                    n,
                    k,
                    blocks,
                    SMALL_M_JOBS_PER_WORKER,
                    t,
                )
                .unwrap();
                for jpw in [1usize, 2] {
                    let mut c_shared = c0.clone();
                    gemm_blis_parallel_two_d_dynamic_shared_a_with_params(
                        &a,
                        &b,
                        &mut c_shared,
                        m,
                        n,
                        k,
                        blocks,
                        jpw,
                        t,
                    )
                    .unwrap();
                    let mut c_ref = c0.clone();
                    gemm_blis_parallel_two_d_dynamic_with_params(
                        &a, &b, &mut c_ref, m, n, k, blocks, jpw, t,
                    )
                    .unwrap();
                    assert_eq!(
                        bits(&c_shared),
                        bits(&c_ref),
                        "{m}x{n}x{k} {t:?} T={threads} jpw={jpw} special={special}"
                    );
                    if jpw == SMALL_M_JOBS_PER_WORKER {
                        assert_eq!(bits(&c_shared), bits(&c_arm));
                    }
                }
                if blocks == default_blocks() {
                    let mut c_prod = c0.clone();
                    gemm_blis_parallel_with_transpose(&a, &b, &mut c_prod, m, n, k, t).unwrap();
                    let mut c_shared = c0.clone();
                    gemm_blis_parallel_two_d_dynamic_shared_a_with_params(
                        &a,
                        &b,
                        &mut c_shared,
                        m,
                        n,
                        k,
                        blocks,
                        TWO_D_JOBS_PER_WORKER,
                        t,
                    )
                    .unwrap();
                    assert_eq!(bits(&c_prod), bits(&c_shared), "prod {m}x{n}x{k} {t:?}");
                }
            });
        }
    }

    #[test]
    fn shared_a_is_bit_identical_on_target_shapes() {
        for &(m, n, k, t) in &SHAPES {
            for special in [false, true] {
                assert_shared_bit_equal(m, n, k, t, special, default_blocks(), &[1, 10, 20]);
            }
        }
    }

    #[test]
    fn shared_a_is_bit_identical_on_fractional_and_tn_shapes() {
        for &(m, n, k) in &[(63usize, 257usize, 785usize), (65, 10, 9)] {
            for t in [GemmTranspose::Nn, GemmTranspose::Nt, GemmTranspose::Tn] {
                for special in [false, true] {
                    assert_shared_bit_equal(m, n, k, t, special, default_blocks(), &[1, 10, 20]);
                }
            }
        }
        // m <= 128 の Tn（pack_a_from_transposed の共有 pack 分岐）。
        assert_shared_bit_equal(
            64,
            256,
            784,
            GemmTranspose::Tn,
            false,
            default_blocks(),
            &[1, 10],
        );
        for m in [7usize, 9, 13, 127, 128, 129] {
            for t in [GemmTranspose::Nn, GemmTranspose::Nt, GemmTranspose::Tn] {
                assert_shared_bit_equal(m, 70, 33, t, true, default_blocks(), &[1, 10]);
            }
        }
        for k in [255usize, 257, 513] {
            for t in [GemmTranspose::Nn, GemmTranspose::Nt, GemmTranspose::Tn] {
                assert_shared_bit_equal(64, 50, k, t, false, default_blocks(), &[1, 10]);
            }
        }
    }

    #[test]
    fn shared_a_is_bit_identical_with_non_default_blocks() {
        let mut kc64 = default_blocks();
        kc64.kc = 64;
        let mut nc16 = default_blocks();
        nc16.nc = 16;
        let mut mc16 = default_blocks();
        mc16.mc = 16;
        for t in [GemmTranspose::Nn, GemmTranspose::Nt, GemmTranspose::Tn] {
            // pc ブロックが 64,64,64,8 になる。
            assert_shared_bit_equal(64, 40, 200, t, false, kc64, &[1, 10]);
            // 1 job の中で jc が複数回まわり、同じ共有 A を使い回す。
            assert_shared_bit_equal(64, 100, 50, t, false, nc16, &[1, 10]);
            // m > blocks.mc → 従来経路へフォールバック。
            assert_shared_bit_equal(64, 40, 50, t, false, mc16, &[1, 10]);
        }
    }

    #[test]
    fn shared_a_paths_cover_direct_write_and_staging() {
        let blocks = default_blocks();
        // T=1 は job_grid の特例で 1x1 → col_bands == 1（直接書き込み）。
        let g1 = partition::job_grid(64, 256, 8, 12, &blocks, 1, 1).unwrap();
        assert_eq!(g1.col_bands, 1);
        let g2 = partition::job_grid(64, 10, 8, 12, &blocks, 10, 1).unwrap();
        assert_eq!(g2.col_bands, 1);
        // 64x256x784 の T=10 は col_bands > 1（staging 使い回し）。
        let g3 = partition::job_grid(64, 256, 8, 12, &blocks, 10, 1).unwrap();
        assert!(g3.col_bands > 1);
    }

    #[test]
    fn shared_a_bias_act_matches_entry_plus_epilogue() {
        let (m, n, k) = (64usize, 256usize, 784usize);
        let a = rand_vec(7, m * k, false);
        let b = rand_vec(11, k * n, false);
        let bias = rand_vec(13, n, false);
        let mut c_prod = vec![0.0f32; m * n];
        gemm_blis_bias_act_parallel(&a, &b, &mut c_prod, m, n, k, Some(&bias), Activation::Relu)
            .unwrap();
        let mut c_shared = vec![0.0f32; m * n];
        gemm_blis_parallel_two_d_dynamic_shared_a_with_params(
            &a,
            &b,
            &mut c_shared,
            m,
            n,
            k,
            default_blocks(),
            SMALL_M_JOBS_PER_WORKER,
            GemmTranspose::Nn,
        )
        .unwrap();
        for (i, v) in c_shared.iter_mut().enumerate() {
            *v = (*v + bias[i % n]).max(0.0);
        }
        assert_eq!(bits(&c_prod), bits(&c_shared));
    }
}
