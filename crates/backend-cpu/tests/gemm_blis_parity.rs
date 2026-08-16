//! `backend-cpu::gemm_blis` の受け入れ基準対応テスト（TASK-1.6f・#184）。
//!
//! 受け入れ条件は「`gemm_blis`／`gemm_blis_parallel` が `gemm_naive` と
//! bit 完全一致する（FMA 契約統一・累積順序保持）こと」。`tests/gemm_parity.rs`
//! （TASK-1.6a）と同じ契約テスト方針を、MR/NR/MC/KC/NC の境界を跨ぐ形状
//! グリッドに対して適用する。
//!
//! 1. 手計算・小規模ケースで正しさの基準を確認し、
//! 2. MR/NR/MC/KC/NC の各境界を跨ぐ形状グリッドで `gemm_naive` と bit 完全
//!    一致することを確認し（`assert_eq!`）、
//! 3. K を大きく取ったストレス形状で FMA 契約の退行を検出し、
//! 4. 境界条件（m=0／n=0／k=0）・エラー経路を検証する。

use backend_cpu::{GemmError, gemm_blis, gemm_blis_parallel, gemm_naive};
use bench_harness::rng::Xorshift64Star;

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

// --- 1. 既知値 ---

#[test]
fn gemm_blis_matches_hand_computed_2x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];
    gemm_blis(&a, &b, &mut c, 2, 2, 2).unwrap();
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn gemm_blis_identity_is_noop() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut c = vec![0.0; 6];
    gemm_blis(&a, &identity, &mut c, 2, 3, 3).unwrap();
    assert_eq!(c, a);
}

// --- 2. gemm_blis / gemm_blis_parallel が naive と bit 完全一致 ---
//
// MR/NR は ISA ごとに異なる（scalar 4x4・neon 8x12（既定。#559 で 8x8 から
// 拡張）・avx2 6x16・avx512 8x32。`src/gemm_blis/microkernel.rs`）。実行時
// ISA ディスパッチ（#185・`microkernel::Isa::detect`）によりどの ISA が
// 選ばれても bit 完全一致するため、本テストは特定 ISA を固定せず実行環境で
// 検出された ISA 経路をそのまま検証する。MC=128・KC=256・NC=512 は
// `src/gemm_blis/mod.rs` の定数と同じ値。グリッドは各境界（MR/NR の
// 最小値・MC/KC/NC）を跨ぐよう m・n・k を選ぶ。n には NEON NR=12（既定。
// #559）の境界（11/12/13）近傍も含める。

const SHAPE_GRID_M: [usize; 7] = [1, 5, 11, 12, 13, 51, 200];
const SHAPE_GRID_N: [usize; 8] = [1, 5, 11, 12, 13, 19, 512, 600];
const SHAPE_GRID_K: [usize; 5] = [1, 3, 255, 257, 700];

#[test]
fn gemm_blis_matches_naive_bit_exact_shape_grid() {
    let mut seed = 100u64;
    for &m in &SHAPE_GRID_M {
        for &n in &SHAPE_GRID_N {
            for &k in &SHAPE_GRID_K {
                seed += 1;
                let a = random_matrix(seed, m * k);
                let b = random_matrix(seed + 1000, k * n);

                let mut c_naive = vec![0.0f32; m * n];
                gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

                let mut c_blis = vec![0.0f32; m * n];
                gemm_blis(&a, &b, &mut c_blis, m, n, k).unwrap();

                assert_eq!(
                    c_naive, c_blis,
                    "gemm_blis が gemm_naive と bit 一致しない（m={m}, n={n}, k={k}）"
                );
            }
        }
    }
}

/// MC/KC/NC いずれも複数ブロックを跨ぐ形状（`tests/gemm_parity.rs` の
/// `gemm_blocked_matches_naive_bit_exact_multi_block` と同一形状）で
/// `gemm_blis`・`gemm_blis_parallel` を検証する。
#[test]
fn gemm_blis_and_parallel_match_naive_bit_exact_multi_block() {
    let (m, n, k) = (200, 600, 700);
    let a = random_matrix(20, m * k);
    let b = random_matrix(21, k * n);

    let mut c_naive = vec![0.0; m * n];
    gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

    let mut c_blis = vec![0.0; m * n];
    gemm_blis(&a, &b, &mut c_blis, m, n, k).unwrap();
    assert_eq!(c_naive, c_blis);

    let mut c_parallel = vec![0.0; m * n];
    gemm_blis_parallel(&a, &b, &mut c_parallel, m, n, k).unwrap();
    assert_eq!(c_naive, c_parallel);
}

#[test]
fn gemm_blis_parallel_matches_naive_bit_exact() {
    let (m, n, k) = (129, 130, 131);
    let a = random_matrix(3, m * k);
    let b = random_matrix(4, k * n);

    let mut c_naive = vec![0.0; m * n];
    gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

    let mut c_parallel = vec![0.0; m * n];
    gemm_blis_parallel(&a, &b, &mut c_parallel, m, n, k).unwrap();

    assert_eq!(c_naive, c_parallel);
}

/// `gemm_blis_parallel` はパネル分割を rayon の稼働スレッド数から決める
/// ため、スレッド数によらず結果が一致することを固定する（
/// `tests/gemm_parity.rs::gemm_parallel_matches_naive_bit_exact_across_thread_pools`
/// と同一パターン）。
#[test]
fn gemm_blis_parallel_matches_naive_bit_exact_across_thread_pools() {
    let (m, n, k) = (523, 600, 700);
    let a = random_matrix(30, m * k);
    let b = random_matrix(31, k * n);

    let mut c_naive = vec![0.0; m * n];
    gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

    for num_threads in [1usize, 3, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap_or_else(|e| panic!("{num_threads} スレッドの rayon プール構築に失敗: {e}"));

        let mut c_parallel = vec![0.0; m * n];
        pool.install(|| gemm_blis_parallel(&a, &b, &mut c_parallel, m, n, k).unwrap());

        assert_eq!(
            c_naive, c_parallel,
            "gemm_blis_parallel（num_threads={num_threads}）が gemm_naive と bit 一致しない"
        );
    }
}

/// panel packing バッファを gemm 呼び出し単位で 1 回確保・再利用する
/// 変更（#556）の回帰テスト。各 rayon 行パネルタスクが `PanelBuffers`
/// を所有する設計（`src/gemm_blis/mod.rs` の [`PanelBuffers`] ドキュメント
/// コメント参照）のため、タスク間でバッファが誤って共有・再利用されると
/// 別パネルの packing 結果が混入し数値が壊れる。固定 4 スレッドプール・
/// 複数行パネル／端タイルを跨ぐ形状で複数回実行し、毎回
/// `gemm_blis`（直列）と bit 完全一致することを確認する
/// （`gemm_blis_parallel_matches_naive_bit_exact_across_thread_pools` と
/// 同一パターンだが、こちらは同一プールでの反復実行に焦点を当てる）。
#[test]
fn gemm_blis_parallel_panel_buffers_are_task_local_across_repeated_runs() {
    let (m, n, k) = (257, 193, 131);
    let a = random_matrix(40, m * k);
    let b = random_matrix(41, k * n);

    let mut c_serial = vec![0.0; m * n];
    gemm_blis(&a, &b, &mut c_serial, m, n, k).unwrap();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap_or_else(|e| panic!("4 スレッドの rayon プール構築に失敗: {e}"));

    for run in 0..8 {
        let mut c_parallel = vec![0.0; m * n];
        pool.install(|| gemm_blis_parallel(&a, &b, &mut c_parallel, m, n, k).unwrap());

        assert_eq!(
            c_serial, c_parallel,
            "gemm_blis_parallel（run={run}）が gemm_blis（直列）と bit 一致しない \
             （panel バッファのタスク間分離が壊れている可能性）"
        );
    }
}

// --- 3. FMA 契約の固定（K が大きいストレス形状） ---

/// K=4096 は PoC-v2-5 の GPU 数値一致ストレスケースと同一規模
/// （`.claude/rules/coding-rust.md`）。`gemm_blis` が `acc += a*b`
/// （非 FMA）へ退行すると失敗する契約テスト。
#[test]
fn gemm_blis_uses_mul_add_fma_contract() {
    let (m, n, k) = (8, 8, 4096);
    let a = random_matrix(5, m * k);
    let b = random_matrix(6, k * n);

    let mut c_actual = vec![0.0; m * n];
    gemm_blis(&a, &b, &mut c_actual, m, n, k).unwrap();

    let mut c_expected = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                let idx = i * n + j;
                c_expected[idx] = a_ip.mul_add(b[p * n + j], c_expected[idx]);
            }
        }
    }

    assert_eq!(c_actual, c_expected);
}

// --- 4. 境界条件 ---

#[test]
fn gemm_blis_handles_1x1x1() {
    let a = vec![2.0f32];
    let b = vec![3.0f32];
    let mut c = vec![0.0f32];
    gemm_blis(&a, &b, &mut c, 1, 1, 1).unwrap();
    assert_eq!(c, vec![6.0]);
}

#[test]
fn gemm_blis_handles_zero_m() {
    let a: Vec<f32> = vec![];
    let b = vec![1.0f32, 2.0];
    let mut c: Vec<f32> = vec![];
    gemm_blis(&a, &b, &mut c, 0, 2, 1).unwrap();
    assert!(c.is_empty());
}

#[test]
fn gemm_blis_handles_zero_k() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let mut c = vec![0.0f32; 6];
    gemm_blis(&a, &b, &mut c, 2, 3, 0).unwrap();
    assert_eq!(c, vec![0.0; 6]);
}

#[test]
fn gemm_zero_n_is_noop_across_blis_and_parallel() {
    let a = vec![1.0f32, 2.0];
    let b: Vec<f32> = vec![];

    let mut c_blis: Vec<f32> = vec![];
    gemm_blis(&a, &b, &mut c_blis, 1, 0, 2).unwrap();
    assert!(c_blis.is_empty());

    let mut c_parallel: Vec<f32> = vec![];
    gemm_blis_parallel(&a, &b, &mut c_parallel, 1, 0, 2).unwrap();
    assert!(c_parallel.is_empty());
}

#[test]
fn gemm_zero_m_and_n_is_noop_across_blis_and_parallel() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];

    let mut c_blis: Vec<f32> = vec![];
    gemm_blis(&a, &b, &mut c_blis, 0, 0, 0).unwrap();
    assert!(c_blis.is_empty());

    let mut c_parallel: Vec<f32> = vec![];
    gemm_blis_parallel(&a, &b, &mut c_parallel, 0, 0, 0).unwrap();
    assert!(c_parallel.is_empty());
}

// --- エラー経路 ---

#[test]
fn gemm_blis_rejects_a_len_mismatch() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c = vec![0.0; 4];
    let err = gemm_blis(&a, &b, &mut c, 2, 2, 2).unwrap_err();
    assert!(matches!(
        err,
        GemmError::ALenMismatch {
            expected: 4,
            actual: 3
        }
    ));
}

#[test]
fn gemm_blis_rejects_dim_product_overflow() {
    let a = vec![0.0f32; 1];
    let b = vec![0.0f32; 1];
    let mut c = vec![0.0f32; 1];
    let err = gemm_blis(&a, &b, &mut c, usize::MAX, 2, 2).unwrap_err();
    assert!(matches!(err, GemmError::DimProductOverflow));
}

#[test]
fn gemm_blis_and_parallel_reject_same_shape_errors() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c1 = vec![0.0; 4];
    assert!(gemm_blis(&a, &b, &mut c1, 2, 2, 2).is_err());
    let mut c2 = vec![0.0; 4];
    assert!(gemm_blis_parallel(&a, &b, &mut c2, 2, 2, 2).is_err());
}
