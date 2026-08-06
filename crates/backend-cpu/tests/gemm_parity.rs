//! `backend-cpu::gemm` の受け入れ基準対応テスト（TASK-1.6a・#21）。
//!
//! 受け入れ条件は「GEMM の数値が参照実装と一致し FMA 契約が守られている
//! こと」（イシュー #21）。本ファイルは
//!
//! 1. `gemm_naive` を既知値・単位行列で正しさを確認する基準とし、
//! 2. `gemm_blocked`／`gemm_parallel` が同じ演算順序（p 昇順の
//!    `mul_add` 累積）のため `gemm_naive` と **bit 完全一致**することを
//!    確認し（`assert_eq!`。累積順序が変わらない限り丸め差は生じない）、
//! 3. K を大きく取ったストレス形状でスカラー `mul_add` 参照実装と
//!    bit 完全一致することで FMA 契約（REQ-2）の退行を検出し、
//! 4. 境界条件（極小・空行列）・エラー経路を検証する。
//!
//! を CI（self-hosted）で実行する非 `#[ignore]` テストとして提供する。
//! PoC evidence（`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/evidence/
//! dump_case_512.bin`）に対する数値突合は `#[ignore]` で分離し、submodule
//! checkout 済みのローカル環境でのみ実行する
//! （`crates/tensor-core/tests/poc_v2_1_parity.rs` の確立パターンを踏襲。
//! CI は `docs/spec` submodule を checkout しない）。

use backend_cpu::{GemmError, gemm_blocked, gemm_naive, gemm_parallel};
use bench_harness::rng::Xorshift64Star;

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

// --- 1. 既知値 ---

#[test]
fn gemm_naive_matches_hand_computed_2x2() {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // A@B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];
    gemm_naive(&a, &b, &mut c, 2, 2, 2).unwrap();
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn gemm_naive_identity_is_noop() {
    // A @ I = A
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut c = vec![0.0; 6];
    gemm_naive(&a, &identity, &mut c, 2, 3, 3).unwrap();
    assert_eq!(c, a);
}

// --- 2. blocked / parallel が naive と bit 完全一致 ---

/// blocked 版・parallel 版とも「p 昇順に `mul_add` で累積する」演算順序を
/// naive 版から変えていない（`kernel_block` は naive と同じ ikj 順を
/// ブロック単位に限定して適用するのみ。`src/gemm.rs` の設計変更の経緯
/// コメント参照）ため、累積順序に起因する丸め差は生じず bit 完全一致する。
/// この一致性は FMA 契約（`mul_add` 統一）が崩れると失われる契約テストで
/// もある。
#[test]
fn gemm_blocked_matches_naive_bit_exact() {
    // MC=128・KC=256・NC=512 のいずれも 1 ブロックに収まる小規模形状
    // （jc・pc・ic いずれのブロッキングループも 1 反復のみ）。ブロック
    // 境界をまたぐ経路の検証は
    // `gemm_blocked_matches_naive_bit_exact_multi_block` を参照
    // （issue #21 レビュー指摘: 本ケースは「境界をまたぐ」形状ではない）。
    let (m, n, k) = (37, 41, 53);
    let a = random_matrix(1, m * k);
    let b = random_matrix(2, k * n);

    let mut c_naive = vec![0.0; m * n];
    gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

    let mut c_blocked = vec![0.0; m * n];
    gemm_blocked(&a, &b, &mut c_blocked, m, n, k).unwrap();

    assert_eq!(c_naive, c_blocked);
}

/// NC（512）・KC（256）・MC（128）のすべてで複数ブロックを跨ぐ形状
/// （jc: 512,1088 の 2 反復・pc: 256,512,640 の 3 反復・ic/gemm_parallel の
/// 並列パネル内 MC ブロッキング: 128 刻みで複数反復）で `gemm_blocked`・
/// `gemm_parallel` を `gemm_naive` と bit 完全一致比較する（issue #21
/// レビュー指摘の Medium 項目: 既存テストはすべて jc・pc いずれも 1 反復
/// のみで、`kernel_block` のオフセット計算〈`(pc+p)*n+jc` 等〉や
/// `nc_len`/`kc_len` のクランプが複数ブロックにまたがる形状でのみ
/// 顕在化するバグを検出できていなかった）。
#[test]
fn gemm_blocked_matches_naive_bit_exact_multi_block() {
    let (m, n, k) = (200, 600, 700);
    let a = random_matrix(20, m * k);
    let b = random_matrix(21, k * n);

    let mut c_naive = vec![0.0; m * n];
    gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

    let mut c_blocked = vec![0.0; m * n];
    gemm_blocked(&a, &b, &mut c_blocked, m, n, k).unwrap();
    assert_eq!(c_naive, c_blocked);

    let mut c_parallel = vec![0.0; m * n];
    gemm_parallel(&a, &b, &mut c_parallel, m, n, k).unwrap();
    assert_eq!(c_naive, c_parallel);
}

#[test]
fn gemm_parallel_matches_naive_bit_exact() {
    // パネル数 > 一般的なスレッド数となるよう大きめの M を取る。
    let (m, n, k) = (129, 130, 131);
    let a = random_matrix(3, m * k);
    let b = random_matrix(4, k * n);

    let mut c_naive = vec![0.0; m * n];
    gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

    let mut c_parallel = vec![0.0; m * n];
    gemm_parallel(&a, &b, &mut c_parallel, m, n, k).unwrap();

    assert_eq!(c_naive, c_parallel);
}

/// `gemm_parallel` はパネル分割・タスク数を rayon の稼働スレッド数に応じて
/// 決めるため（`src/gemm.rs` の並列パネル分割ロジック）、スレッド数が
/// 結果に影響しないことは自明ではない。1・3・16 スレッドの `ThreadPool` を
/// 明示構築し `install` 内で `gemm_parallel` を実行して、`gemm_naive` との
/// bit 完全一致（許容誤差なし）をスレッド数によらず確認する（issue #21
/// 方針コメント
/// https://github.com/Fandhe-AI/rust-ai-library/issues/21#issuecomment-5200554933
/// で指定されたレビュー Medium 指摘対応。`tests/reduction.rs` の
/// `full_reduction_deterministic_across_thread_pools` と同一パターン）。
/// 形状は NC/KC/MC いずれも複数ブロックを跨ぐ ragged panel
/// （`gemm_blocked_matches_naive_bit_exact_multi_block` と同種の意図で
/// M=523 は MC=128 の倍数でない端数パネルを含む）。
#[test]
fn gemm_parallel_matches_naive_bit_exact_across_thread_pools() {
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
        pool.install(|| gemm_parallel(&a, &b, &mut c_parallel, m, n, k).unwrap());

        assert_eq!(
            c_naive, c_parallel,
            "gemm_parallel（num_threads={num_threads}）が gemm_naive と bit 一致しない"
        );
    }
}

// --- 3. FMA 契約の固定（K が大きいストレス形状） ---

/// `acc += a*b`（乗算・加算を別々に丸める）実装への退行で失敗する契約
/// テスト。K=4096 は PoC-v2-5 の GPU 数値一致ストレスケースと同一規模
/// （`.claude/rules/coding-rust.md`「PoC-v2-5 の K=4096 ストレスケースで
/// 実測確認済み」）。
#[test]
fn gemm_naive_uses_mul_add_fma_contract() {
    let (m, n, k) = (8, 8, 4096);
    let a = random_matrix(5, m * k);
    let b = random_matrix(6, k * n);

    let mut c_actual = vec![0.0; m * n];
    gemm_naive(&a, &b, &mut c_actual, m, n, k).unwrap();

    // スカラー参照実装（`f32::mul_add` を明示的に用いる、テスト内独立実装）。
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
fn gemm_blocked_handles_1x1x1() {
    let a = vec![2.0f32];
    let b = vec![3.0f32];
    let mut c = vec![0.0f32];
    gemm_blocked(&a, &b, &mut c, 1, 1, 1).unwrap();
    assert_eq!(c, vec![6.0]);
}

#[test]
fn gemm_blocked_handles_shape_smaller_than_block_size() {
    // MC=128・KC=256・NC=512 未満の形状でも壊れないこと。
    let (m, n, k) = (3, 5, 7);
    let a = random_matrix(10, m * k);
    let b = random_matrix(11, k * n);
    let mut c_naive = vec![0.0; m * n];
    gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();
    let mut c_blocked = vec![0.0; m * n];
    gemm_blocked(&a, &b, &mut c_blocked, m, n, k).unwrap();
    assert_eq!(c_naive, c_blocked);
}

#[test]
fn gemm_naive_handles_zero_m() {
    let a: Vec<f32> = vec![];
    let b = vec![1.0f32, 2.0];
    let mut c: Vec<f32> = vec![];
    gemm_naive(&a, &b, &mut c, 0, 2, 1).unwrap();
    assert!(c.is_empty());
}

#[test]
fn gemm_naive_handles_zero_k() {
    // K=0: 縮約次元が空なら C は変化しない（呼び出し前ゼロ初期化のまま）。
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let mut c = vec![0.0f32; 6];
    gemm_naive(&a, &b, &mut c, 2, 3, 0).unwrap();
    assert_eq!(c, vec![0.0; 6]);
}

#[test]
fn gemm_naive_handles_zero_n() {
    let a = vec![1.0f32, 2.0];
    let b: Vec<f32> = vec![];
    let mut c: Vec<f32> = vec![];
    gemm_naive(&a, &b, &mut c, 1, 0, 2).unwrap();
    assert!(c.is_empty());
}

/// N=0 は validate_dims が合法とみなす形状（b・c は空スライス）だが、
/// `gemm_parallel` は `par_chunks_mut(panel_rows * n)` のチャンクサイズが
/// 0 になりうる経路を持つため、naive／blocked と挙動が食い違う
/// リグレッションが生じやすい（`gemm_parallel` は本テスト追加前は
/// この形状でパニックしていた）。3 実装が同じ no-op として振る舞う
/// ことを固定する（parity 契約。issue #21 レビュー指摘）。
#[test]
fn gemm_zero_n_is_noop_across_all_three_kernels() {
    let a = vec![1.0f32, 2.0];
    let b: Vec<f32> = vec![];

    let mut c_naive: Vec<f32> = vec![];
    gemm_naive(&a, &b, &mut c_naive, 1, 0, 2).unwrap();
    assert!(c_naive.is_empty());

    let mut c_blocked: Vec<f32> = vec![];
    gemm_blocked(&a, &b, &mut c_blocked, 1, 0, 2).unwrap();
    assert!(c_blocked.is_empty());

    let mut c_parallel: Vec<f32> = vec![];
    gemm_parallel(&a, &b, &mut c_parallel, 1, 0, 2).unwrap();
    assert!(c_parallel.is_empty());
}

/// M=0・N=0 の組合せでも `gemm_parallel` がパニックしないことを確認する
/// （`par_chunks_mut` のチャンクサイズ 0 が M=0 単独では顕在化しないため、
/// N=0 の経路と分けて固定する）。
#[test]
fn gemm_zero_m_and_n_is_noop_across_all_three_kernels() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];

    let mut c_naive: Vec<f32> = vec![];
    gemm_naive(&a, &b, &mut c_naive, 0, 0, 0).unwrap();
    assert!(c_naive.is_empty());

    let mut c_blocked: Vec<f32> = vec![];
    gemm_blocked(&a, &b, &mut c_blocked, 0, 0, 0).unwrap();
    assert!(c_blocked.is_empty());

    let mut c_parallel: Vec<f32> = vec![];
    gemm_parallel(&a, &b, &mut c_parallel, 0, 0, 0).unwrap();
    assert!(c_parallel.is_empty());
}

// --- エラー経路 ---

#[test]
fn gemm_naive_rejects_a_len_mismatch() {
    let a = vec![1.0, 2.0, 3.0]; // m*k=4 を期待
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c = vec![0.0; 4];
    let err = gemm_naive(&a, &b, &mut c, 2, 2, 2).unwrap_err();
    assert!(matches!(
        err,
        GemmError::ALenMismatch {
            expected: 4,
            actual: 3
        }
    ));
}

#[test]
fn gemm_naive_rejects_b_len_mismatch() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0, 3.0]; // k*n=4 を期待
    let mut c = vec![0.0; 4];
    let err = gemm_naive(&a, &b, &mut c, 2, 2, 2).unwrap_err();
    assert!(matches!(
        err,
        GemmError::BLenMismatch {
            expected: 4,
            actual: 3
        }
    ));
}

#[test]
fn gemm_naive_rejects_c_len_mismatch() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c = vec![0.0; 3]; // m*n=4 を期待
    let err = gemm_naive(&a, &b, &mut c, 2, 2, 2).unwrap_err();
    assert!(matches!(
        err,
        GemmError::CLenMismatch {
            expected: 4,
            actual: 3
        }
    ));
}

#[test]
fn gemm_naive_rejects_dim_product_overflow() {
    let a = vec![0.0f32; 1];
    let b = vec![0.0f32; 1];
    let mut c = vec![0.0f32; 1];
    let err = gemm_naive(&a, &b, &mut c, usize::MAX, 2, 2).unwrap_err();
    assert!(matches!(err, GemmError::DimProductOverflow));
}

#[test]
fn gemm_blocked_and_parallel_reject_same_shape_errors() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c1 = vec![0.0; 4];
    assert!(gemm_blocked(&a, &b, &mut c1, 2, 2, 2).is_err());
    let mut c2 = vec![0.0; 4];
    assert!(gemm_parallel(&a, &b, &mut c2, 2, 2, 2).is_err());
}

// --- PoC evidence 数値突合（#[ignore]。docs/spec submodule 必須） ---

/// `dump_case_512.bin` のヘッダ（m・n・k）を読み、ファイル長との整合を
/// 検証してから本体（A・B・C の f32 LE 配列）を返す。パース時の境界検証を
/// 本体読み込みより先に行う（OWASP A03 観点。`.claude/rules/security.md`
/// 「外部フォーマットパースは長さ・形状の検証を先に行う」。
/// `crates/tensor-core/tests/poc_v2_1_parity.rs::parse_dump_case` と
/// 同一のパース方針）。
struct DumpCase {
    m: usize,
    n: usize,
    k: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
}

fn parse_dump_case(bytes: &[u8]) -> DumpCase {
    const HEADER_LEN: usize = 12;
    assert!(
        bytes.len() >= HEADER_LEN,
        "dump_case_512.bin: ヘッダ長（12 バイト）に満たない不正なファイル"
    );
    let m = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

    let a_len = m
        .checked_mul(k)
        .expect("dump_case_512.bin: m*k がオーバーフローする不正なヘッダ");
    let b_len = k
        .checked_mul(n)
        .expect("dump_case_512.bin: k*n がオーバーフローする不正なヘッダ");
    let c_len = m
        .checked_mul(n)
        .expect("dump_case_512.bin: m*n がオーバーフローする不正なヘッダ");
    let total_f32 = a_len
        .checked_add(b_len)
        .and_then(|v| v.checked_add(c_len))
        .expect("dump_case_512.bin: 本体要素数がオーバーフローする不正なヘッダ");
    let body_bytes = total_f32
        .checked_mul(4)
        .expect("dump_case_512.bin: 本体バイト数がオーバーフローする不正なヘッダ");
    let expected_len = HEADER_LEN
        .checked_add(body_bytes)
        .expect("dump_case_512.bin: ファイル総バイト数がオーバーフローする不正なヘッダ");
    assert_eq!(
        bytes.len(),
        expected_len,
        "dump_case_512.bin: ヘッダ（m={m}, n={n}, k={k}）から期待される長さ \
         {expected_len} バイトと実ファイル長 {} バイトが一致しない",
        bytes.len()
    );

    let read_f32_le = |start: usize, len: usize| -> Vec<f32> {
        bytes[start..start + len * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let a_start = HEADER_LEN;
    let b_start = a_start + a_len * 4;
    let c_start = b_start + b_len * 4;
    let a = read_f32_le(a_start, a_len);
    let b = read_f32_le(b_start, b_len);
    let c = read_f32_le(c_start, c_len);

    DumpCase { m, n, k, a, b, c }
}

/// PoC-v2-1 の複合判定（相対誤差 1e-4 以内 または 絶対誤差 1e-5 以内。
/// `evidence/numeric_check.log`）をそのまま用いる。この許容誤差は
/// 単独緩和しない（`.claude/rules/coding-rust.md`）。
fn numeric_close(actual: f32, expected: f32) -> bool {
    const REL_TOL: f32 = 1e-4;
    const ABS_RESCUE: f32 = 1e-5;
    let abs_diff = (actual - expected).abs();
    if abs_diff <= ABS_RESCUE {
        return true;
    }
    let rel_err = abs_diff / expected.abs().max(f32::EPSILON);
    rel_err <= REL_TOL
}

/// `gemm_parallel` の出力を PoC evidence の C（PyTorch `torch.matmul` と
/// PoC-v2-1 で複合判定 PASS 済み）と突合する。PoC の内側ループは
/// `c += a * b`（非 FMA）だったのに対し本実装は `mul_add` を用いるため
/// PoC 出力とは bit 一致しない。したがって bit 完全一致ではなく PoC の
/// 複合判定（相対誤差 1e-4 以内 または 絶対誤差 1e-5 以内）による許容誤差
/// 比較とする（イシュー #21 実装計画のテスト方針 7 に対応）。
#[test]
#[ignore = "docs/spec submodule の checkout が必要（CI は submodule 非取得）"]
fn gemm_parallel_matches_poc_evidence_within_tolerance() {
    let path = format!(
        "{}/../../docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/evidence/dump_case_512.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "dump_case_512.bin を読めない（{path}）: {e}。\
             `git submodule update --init docs/spec` で submodule を \
             checkout してから再実行すること。"
        )
    });
    let case = parse_dump_case(&bytes);
    assert_eq!((case.m, case.n, case.k), (512, 512, 512));

    let mut c_actual = vec![0.0f32; case.m * case.n];
    gemm_parallel(&case.a, &case.b, &mut c_actual, case.m, case.n, case.k).unwrap();

    let mut fail_cells = 0usize;
    for (actual, expected) in c_actual.iter().zip(case.c.iter()) {
        if !numeric_close(*actual, *expected) {
            fail_cells += 1;
        }
    }
    assert_eq!(
        fail_cells,
        0,
        "gemm_parallel（512x512x512）が dump 内 C と複合判定（相対 1e-4 以内 \
         または絶対 1e-5 以内）で不一致（fail_cells={fail_cells}/{}）",
        c_actual.len()
    );
}
