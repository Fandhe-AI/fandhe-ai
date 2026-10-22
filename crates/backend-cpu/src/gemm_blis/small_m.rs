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
//! 後続: #3199 が同じゲートの下で A の共有 pack と確保削減を足す。設計・採否・A/B の扱いは
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

#[cfg(test)]
mod tests {
    use super::super::{
        GemmTranspose, default_blocks, gemm_blis_bias_act_parallel,
        gemm_blis_parallel_two_d_dynamic_with_params, gemm_blis_parallel_with_transpose,
        install_in_pool_for_test,
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
        const { assert!(!SMALL_M_JOB_MERGE_ENABLED) };
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
}
