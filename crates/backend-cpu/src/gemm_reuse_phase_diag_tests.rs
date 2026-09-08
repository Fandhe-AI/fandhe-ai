//! CPU GEMM reuse 計測境界（`scripts/bench/framework-compare` の
//! `bench-fandhe --task gemm --mode reuse --phases`。イシュー #1182・
//! CPU 向け追加はイシュー #1290）の `matmul` 区間内訳を実測分解する
//! 診断テスト。
//!
//! # 背景
//!
//! `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §8.1・§8.5 は
//! 「framework-compare reuse 境界とカーネル単体の GFLOP/s 差
//! （M4 Max 16〜24%・GB10 31〜48%）」を candle 比未達の寄与最大要因と
//! **推定**したまま、CPU では reuse 1 反復のどこに固定費が乗るかを分解
//! していなかった。`gemm --mode reuse --phases`（イシュー #1182）は
//! `matmul`／`to_tensor`／`host_copy`／`checksum`／`iter_total` の
//! 5 区間まで公開 API 呼び出し境界で分解できるが、`matmul` 区間の内側
//! （`Var::matmul` → `CpuBackendOps::gemm` → `gemm_blis_parallel`。
//! `crate::ops::CpuBackendOps::gemm` 参照）は fandhe-ai 0.7.0 の公開
//! API では観測不能である。本ファイルはその内側を `crates/backend-cpu`
//! の非公開 API・`fandhe_ai_autodiff::Tape`（dev-dependency）へ直接
//! アクセスして分解する（CUDA `gemm_reuse_phase_diag_tests.rs`〈#1182〉・
//! Metal 同名ファイル〈#1189〉と同じ設計・同じ配置理由）。
//!
//! # `Var::matmul` 1 反復との対応
//!
//! `crates/autodiff/src/var.rs::Var::matmul` の実体は
//! `materialize_fallible(..).clone()`（A・B 各 1 回。`Tensor` は
//! `Arc` 共有のため clone は安価）→ `self.tape.ops().gemm(&lhs, &rhs)`
//! （`CpuBackendOps::gemm`。`crates/backend-cpu/src/ops.rs`）→
//! `push_eager`（`freeze_leaf_prefix` + ノード push。
//! `crates/autodiff/src/tape.rs`）。`CpuBackendOps::gemm` 自体は
//! `vec![0.0f32; m*n]`（C 確保）→ `gemm_into_slice`（本番 NN 経路は
//! `gemm_blis_parallel`。イシュー #1213 の転置専用入口は本診断の対象外）
//! → `Tensor::new(out, &out_shape)` の 3 段に分解できる。本ファイルは
//! この内側を次の区間として計測する:
//!
//! | phase | 実体 | 対応する Layer A 区間 |
//! | --- | --- | --- |
//! | `alloc_c` | `zeroed_output(n*n)`（本番 `CpuBackendOps::gemm` と同じ確保。イシュー #1299 でしきい値以上の rayon 並列ゼロ書き込み分岐を追加したが、本番既定は `usize::MAX` で無効化のため実質は従来どおり逐次確保。M4 Max スモークで N=2048 の後退を確認したため。#1301 が DGX 実機実測で有効化可否を判断する） | `matmul` 内側 |
//! | `kernel` | `gemm_blis_parallel(a, b, &mut c, n, n, n)`（本番 RowPanel・既定スレッド数） | `matmul` 内側 |
//! | `tensor_wrap` | `Tensor::new(c, &[n, n])` | `matmul` 内側 |
//! | `ops_gemm` | `CpuBackendOps::gemm` 呼び出し 1 回（alloc_c+kernel+tensor_wrap の本番合成。別試行として計測） | `matmul` − `ops_gemm` ≈ autodiff 残差 |
//! | `tape_matmul` | `Tape` 上で `a.matmul(&b)`（`tests/fusion_effect_perf.rs` と同じ `Tape::new_with_ops` 経由。reuse＝tape は 1 回だけ構築し葉 Var を使い回す） | Layer A `matmul` の HEAD レプリカ（突合の基準） |
//! | `to_tensor` | `c_var.to_tensor()` | Layer A `to_tensor` 対応 |
//! | `host_copy` | `.contiguous().as_slice().to_vec()` | Layer A `host_copy` 対応 |
//! | `checksum` | f64 全要素和 | Layer A `checksum` 対応 |
//!
//! CPU には CUDA／Metal の H2D／D2H・ストリーム同期に相当する区間が
//! 存在しない（ホスト常駐のまま演算する）。
//!
//! # 忠実性の注意点（README「CPU での区間定義と Layer B」節にも記す）
//!
//! - **keep-alive**: 本番 reuse（`run_gemm_reuse`）は matmul の出力
//!   `Tensor` をアロケータへ返却せず毎反復新規ページに書く一方、
//!   readout（ホストコピー `Vec<f32>`）は反復ごとに破棄する。本診断は
//!   `kernel`／`ops_gemm` 各パスの出力 `Tensor`（`wrapped`／`ops_out`）
//!   のみを `keep_alive` に保持しアロケータのページ再利用でコストが
//!   消える乖離を避け、readout コピー（`out`）は本番と同じく保持しない
//!   （`tape_matmul` パスの出力は tape 自身が内部で保持するため
//!   `keep_alive` への追加は不要。N=2048 で `wrapped`＋`ops_out` 計
//!   2 個 × (20 warmup + 20 測定) × 16 MiB ≈ 1.3 GiB＋tape 側 40 個 ×
//!   16 MiB ≈ 640 MiB。合計約 2 GiB オーダー）。
//! - **calloc／first-touch の帰属**: `zeroed_output` がしきい値未満の
//!   `vec![0.0; n*n]` へ倒れる場合、大サイズでは OS の遅延ゼロページ
//!   （mmap）に倒れうるため、初回書き込みの page-fault コストは
//!   `alloc_c` ではなく `kernel`（実際に書き込む側）に計上されうる。
//!   `alloc_c` を「確保コストの上限」と読まない。イシュー #1299 は
//!   しきい値（`GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS`）以上で `alloc_c`
//!   区間内に並列ゼロ書き込み（first-touch を複数スレッドへ分散）まで
//!   前倒しする分岐を追加したが、**M4 Max スモーク実測で N=2048 の
//!   `alloc_c` が逐次経路比 約 3〜22 倍・`ops_gemm` 合成が中央値約 29%
//!   後退することを確認したため、本番既定は `usize::MAX`（無効化）**
//!   （`docs/perf/cpu-matmul-fixed-cost-impl.md`）。macOS の
//!   `vec![0.0f32; n*n]` は遅延ゼロページをそのまま返しフォールトが
//!   `kernel` 側へ遅延される一方、並列書き込みは全ページのフォールトを
//!   `alloc_c` 側へ前倒しするため、この機構では帰属の曖昧さは解消され
//!   ず単にコストの計上区間が移動しただけだった。Linux（DGX Spark
//!   GB10）の glibc heap 経路では §3.C の当初仮説（calloc の memset）が
//!   依然成立しうるため、#1301 が DGX 実機実測で有効化可否を判断する。
//! - `RAYON_NUM_THREADS` は既定のまま（`bench-fandhe` の gate プロトコル
//!   と同一環境）。
//!
//! # gating しない方針（CUDA/Metal の診断テストと同じ理由）
//!
//! 本ファイルの `#[test]` は各フェーズが例外なく完了し出力が有限・
//! 要素数一致・checksum 非ゼロであることのみを検証し、フェーズ間の
//! 大小関係・絶対値への `assert!` は行わない（環境揺らぎによる
//! flaky 化防止）。数値は `println!` に残し、後続イシュー（#1292）が
//! `docs/perf/cpu-gemm-candle-gate-remeasurement.md` へ転記する一次
//! 情報とする。
//!
//! `--test-threads=1` 必須ではない（CUDA/Metal と異なり単一デバイス
//! の排他ハンドルを共有しないため）が、`rayon` グローバルスレッド
//! プールをプロセス全体で共有するため、他テストと同時実行すると
//! カーネル計測がスレッド競合でぶれる。実機実測（`#[ignore]`）は
//! README の実行コマンド（`--test-threads=1` 推奨）に従う。

use std::time::Instant;

use bench_harness::rng::Xorshift64Star;
use bench_harness::{Quartiles, median_q1_q3};
use fandhe_ai_autodiff::Tape;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

use crate::gemm_blis::gemm_blis_parallel;
use crate::ops::{CpuBackendOps, zeroed_output};

const WARMUP_TRIALS: usize = 20;
const MEASURED_TRIALS: usize = 20;

/// `bench-fandhe --task gemm --mode reuse --phases`（イシュー #1182）が
/// 対象とする CPU gate 形状（README「GEMM ゲート 5 回計測」節。
/// cpu={512,1024,2048}。GPU の {1024,2048,4096} とは異なる）。
const SIZES: [usize; 3] = [512, 1024, 2048];

fn gen_square_ab(seed: u64, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let a = rng.fill_vec(n * n);
    let b = rng.fill_vec(n * n);
    (a, b)
}

fn median_of(samples: &[f64]) -> Quartiles {
    median_q1_q3(samples)
        .expect("samples collected from successful trials must be non-empty and NaN-free")
}

fn print_quartiles_ms(label: &str, q: Quartiles) {
    println!(
        "    {label}: median={:.4} ms  q1={:.4} ms  q3={:.4} ms",
        q.median * 1e3,
        q.q1 * 1e3,
        q.q3 * 1e3
    );
}

/// 1 反復分のフェーズ計測結果（ファイル冒頭「対応」表参照）。`to_tensor`
/// と `host_copy` を分離しているのは Layer A（`measure_gemm_reuse_phases`）
/// が両者を独立区間として計時しているため（1:1 対応）。
struct PhaseSample {
    alloc_c_secs: f64,
    kernel_secs: f64,
    tensor_wrap_secs: f64,
    ops_gemm_secs: f64,
    tape_matmul_secs: f64,
    to_tensor_secs: f64,
    host_copy_secs: f64,
    checksum_secs: f64,
}

/// 1 サイズの 1 反復を計測する。`a_slice`／`b_slice` は呼び出し元が
/// ループ外で 1 回だけ `contiguous().as_slice()` した結果（`a_tensor`／
/// `b_tensor` はループ内で不変のため。`kernel` 区間の計時窓から
/// `contiguous()`/`as_slice()` の呼び出しコストを除外し、README の
/// 「H2D 相当なし（`Arc` clone は計測対象外）」対応表と整合させる）。
/// `ops` は毎反復新規構築する（`CpuBackendOps::new()` は ZST でホット
/// スポットではないと `docs/perf/cpu-infer-predict-profile.md` で確認
/// 済みのため `ops_gemm` の計測に影響しない）。
// 各引数は 1 反復の各フェーズ計測に必要な独立した入力（呼び出し元で
// 1 回だけ準備済みの Var／Tensor／スライス・サイズ・keep_alive バッ
// ファ）であり、構造体へ束ねると呼び出し元での使い分け（フェーズ毎に
// 異なる組み合わせで再利用）が読みにくくなるため、テスト専用ヘルパー
// として引数個数の lint を明示的に許容する。
#[allow(clippy::too_many_arguments)]
fn measure_one_phase_trial(
    a_var: &fandhe_ai_autodiff::Var<'_>,
    b_var: &fandhe_ai_autodiff::Var<'_>,
    a_tensor: &Tensor<f32>,
    b_tensor: &Tensor<f32>,
    a_slice: &[f32],
    b_slice: &[f32],
    n: usize,
    keep_alive: &mut Vec<Tensor<f32>>,
) -> PhaseSample {
    let ops = CpuBackendOps::new();

    // (1) alloc_c: 本番 `CpuBackendOps::gemm` と同じ出力確保
    // （イシュー #1299 以降は `zeroed_output`。しきい値未満は従来どおり
    // `vec![0.0f32; n*n]`、以上は rayon 並列ゼロ書き込みへ分岐する。
    // この区間内で first-touch も含めて計時するのは変更前と同じ）。
    let t = Instant::now();
    let mut c = zeroed_output(n * n);
    let alloc_c_secs = t.elapsed().as_secs_f64();

    // (2) kernel: 本番 NN 経路のマイクロカーネル本体
    // （`gemm_into_slice` の NN 分岐と同一関数）。`a_slice`／`b_slice`
    // は呼び出し元で事前取得済みのため、この計時窓には
    // `gemm_blis_parallel` 自体のコストのみが乗る。
    let t = Instant::now();
    gemm_blis_parallel(a_slice, b_slice, &mut c, n, n, n).expect("gemm_blis_parallel must succeed");
    let kernel_secs = t.elapsed().as_secs_f64();

    // (3) tensor_wrap: `CpuBackendOps::gemm` 末尾の `Tensor::new` 相当。
    let t = Instant::now();
    let wrapped = Tensor::new(c, &[n, n]).expect("Tensor::new must succeed");
    let tensor_wrap_secs = t.elapsed().as_secs_f64();

    // (4) ops_gemm: `CpuBackendOps::gemm` を 1 回（alloc_c+kernel+
    // tensor_wrap の本番合成。別試行として計測するため上記 3 区間の
    // 合計とは独立にオーバーヘッドを比較できる）。
    let t = Instant::now();
    let ops_out = ops
        .gemm(a_tensor, b_tensor)
        .expect("CpuBackendOps::gemm must succeed");
    let ops_gemm_secs = t.elapsed().as_secs_f64();

    // (5) tape_matmul: Layer A `matmul` 区間の HEAD 側レプリカ
    // （`Var::matmul` そのもの。`tests/fusion_effect_perf.rs:193` と
    // 同じ `Tape::new_with_ops` 経由の呼び出し方）。
    let t = Instant::now();
    let c_var = a_var.matmul(b_var).expect("Var::matmul must succeed");
    let tape_matmul_secs = t.elapsed().as_secs_f64();

    // (6) to_tensor + (7) host_copy: Layer A の非 feature 経路
    // （`measure_gemm_reuse_phases` の `#[cfg(not(feature =
    // "host-view-readout"))]` 分岐）と同一の 2 区間分割
    // （`c.to_tensor()` → `.contiguous().as_slice().to_vec()`）。
    let t = Instant::now();
    let tensor = c_var.to_tensor();
    let to_tensor_secs = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let out = tensor
        .contiguous()
        .as_slice()
        .expect("contiguous slice")
        .to_vec();
    let host_copy_secs = t.elapsed().as_secs_f64();

    // (8) checksum: f64 全要素和（Layer A `checksum` 区間対応）。
    let t = Instant::now();
    let checksum: f64 = out.iter().map(|&x| x as f64).sum();
    let checksum_secs = t.elapsed().as_secs_f64();

    // sanity: 有限・要素数一致・非ゼロ checksum（§冒頭「gating しない
    // 方針」参照。大小関係への assert は行わない）。
    assert_eq!(out.len(), n * n);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "output must be finite (n={n})"
    );
    assert!(
        checksum.is_finite() && checksum != 0.0,
        "checksum must be finite and non-zero (n={n})"
    );
    assert_eq!(ops_out.numel(), n * n);

    // keep-alive: 本番 reuse（`run_gemm_reuse`）は matmul の出力 `Tensor`
    // をアロケータへ返却せず毎反復新規ページに書く一方、readout
    // （`to_tensor`/`host_copy` が生む `Vec<f32>` コピー）はそのつど
    // 破棄する（`one` クロージャのスコープを出ると `out` は drop）。
    // 本診断もこれに合わせ、`wrapped`（kernel パス C）・`ops_out`
    // （ops_gemm パス C）の 2 つの `Tensor` のみをアロケータのページ
    // 再利用を防ぐため保持し、`out`（readout コピー）は保持しない
    // （§冒頭「keep-alive」注記参照。`tape_matmul` パスの `c_var` は
    // tape 自身が内部で保持するため keep_alive への追加不要）。
    keep_alive.push(wrapped);
    keep_alive.push(ops_out);

    PhaseSample {
        alloc_c_secs,
        kernel_secs,
        tensor_wrap_secs,
        ops_gemm_secs,
        tape_matmul_secs,
        to_tensor_secs,
        host_copy_secs,
        checksum_secs,
    }
}

fn run_size(n: usize) {
    let (a_data, b_data) = gen_square_ab(0x1290_a000 ^ (n as u64), n);
    let a_tensor = Tensor::new(a_data, &[n, n]).expect("a tensor construction");
    let b_tensor = Tensor::new(b_data, &[n, n]).expect("b tensor construction");
    // `a_tensor`／`b_tensor` はループ内で不変のため、`contiguous().
    // as_slice()` は 1 回だけ呼ぶ（`kernel` 計時窓の外。§`measure_one_
    // phase_trial` doc 参照）。`Tensor::new` 直後は既に contiguous
    // なため `contiguous()` は `Arc` clone のみ（`Tensor::contiguous`
    // doc 参照）。
    let a_slice = a_tensor
        .contiguous()
        .as_slice()
        .expect("a contiguous")
        .to_vec();
    let b_slice = b_tensor
        .contiguous()
        .as_slice()
        .expect("b contiguous")
        .to_vec();

    // reuse 形: tape・葉 Var は 1 回だけ構築し全反復で使い回す
    // （`run_gemm_reuse`／`measure_gemm_reuse_phases` と同じ規約）。
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let a_var = tape.var(&a_tensor);
    let b_var = tape.var(&b_tensor);

    // keep-alive: `wrapped`／`ops_out`（各反復の kernel パス／ops_gemm
    // パス C）のみを保持する（§`measure_one_phase_trial` の keep-alive
    // 注記参照）。
    let mut keep_alive: Vec<Tensor<f32>> =
        Vec::with_capacity(2 * (WARMUP_TRIALS + MEASURED_TRIALS));

    for _ in 0..WARMUP_TRIALS {
        let _ = measure_one_phase_trial(
            &a_var,
            &b_var,
            &a_tensor,
            &b_tensor,
            &a_slice,
            &b_slice,
            n,
            &mut keep_alive,
        );
    }

    let mut alloc_c = Vec::with_capacity(MEASURED_TRIALS);
    let mut kernel = Vec::with_capacity(MEASURED_TRIALS);
    let mut tensor_wrap = Vec::with_capacity(MEASURED_TRIALS);
    let mut ops_gemm = Vec::with_capacity(MEASURED_TRIALS);
    let mut tape_matmul = Vec::with_capacity(MEASURED_TRIALS);
    let mut to_tensor = Vec::with_capacity(MEASURED_TRIALS);
    let mut host_copy = Vec::with_capacity(MEASURED_TRIALS);
    let mut checksum = Vec::with_capacity(MEASURED_TRIALS);

    for _ in 0..MEASURED_TRIALS {
        let s = measure_one_phase_trial(
            &a_var,
            &b_var,
            &a_tensor,
            &b_tensor,
            &a_slice,
            &b_slice,
            n,
            &mut keep_alive,
        );
        alloc_c.push(s.alloc_c_secs);
        kernel.push(s.kernel_secs);
        tensor_wrap.push(s.tensor_wrap_secs);
        ops_gemm.push(s.ops_gemm_secs);
        tape_matmul.push(s.tape_matmul_secs);
        to_tensor.push(s.to_tensor_secs);
        host_copy.push(s.host_copy_secs);
        checksum.push(s.checksum_secs);
    }

    let total: f64 = [
        &alloc_c,
        &kernel,
        &tensor_wrap,
        &to_tensor,
        &host_copy,
        &checksum,
    ]
    .iter()
    .map(|v| median_of(v).median)
    .sum();

    println!("  N={n} (median over {MEASURED_TRIALS} trials, {WARMUP_TRIALS} warmup):");
    print_quartiles_ms("alloc_c", median_of(&alloc_c));
    print_quartiles_ms("kernel", median_of(&kernel));
    print_quartiles_ms("tensor_wrap", median_of(&tensor_wrap));
    print_quartiles_ms(
        "ops_gemm (alloc_c+kernel+tensor_wrap 本番合成)",
        median_of(&ops_gemm),
    );
    print_quartiles_ms(
        "tape_matmul (Layer A matmul レプリカ)",
        median_of(&tape_matmul),
    );
    print_quartiles_ms("to_tensor", median_of(&to_tensor));
    print_quartiles_ms("host_copy", median_of(&host_copy));
    print_quartiles_ms("checksum", median_of(&checksum));
    println!(
        "    sum of medians (alloc_c+kernel+tensor_wrap+to_tensor+host_copy+checksum): {:.4} ms",
        total * 1e3
    );
}

/// 実機依存の診断テスト（`docs/perf/cpu-gemm-candle-gate-remeasurement.md`
/// への転記元。イシュー #1292 で実測記録する）。他テストとの rayon
/// スレッドプール競合を避けるため `--test-threads=1` 推奨（README
/// 「CPU での区間定義と Layer B」節参照）。
#[test]
#[ignore = "実機（DGX Spark GB10 / Apple M4 Max）で cargo test -p \
            fandhe-ai-backend-cpu --release --lib -- --ignored \
            gemm_reuse_phase_diag_cpu --nocapture --test-threads=1 として \
            5 回独立プロセス実行し中央値を docs/perf/ へ記録する（イシュー #1292）"]
fn gemm_reuse_phase_diag_cpu() {
    for n in SIZES {
        run_size(n);
    }
}

/// 非 `#[ignore]` スモーク: モジュールがコンパイル・完走し、各フェーズが
/// 有限値・要素数一致・checksum 非ゼロで終わることを CI で固定する
/// （数値の判定は行わない。#1292 で数値記録）。
#[test]
fn gemm_reuse_phase_diag_cpu_smoke_small() {
    let n = 64;
    let (a_data, b_data) = gen_square_ab(0x1290_5ee0, n);
    let a_tensor = Tensor::new(a_data, &[n, n]).expect("a tensor construction");
    let b_tensor = Tensor::new(b_data, &[n, n]).expect("b tensor construction");

    let a_slice = a_tensor
        .contiguous()
        .as_slice()
        .expect("a contiguous")
        .to_vec();
    let b_slice = b_tensor
        .contiguous()
        .as_slice()
        .expect("b contiguous")
        .to_vec();

    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let a_var = tape.var(&a_tensor);
    let b_var = tape.var(&b_tensor);
    let mut keep_alive: Vec<Tensor<f32>> = Vec::with_capacity(2 * 4);

    for _ in 0..2 {
        let _ = measure_one_phase_trial(
            &a_var,
            &b_var,
            &a_tensor,
            &b_tensor,
            &a_slice,
            &b_slice,
            n,
            &mut keep_alive,
        );
    }
    let last = measure_one_phase_trial(
        &a_var,
        &b_var,
        &a_tensor,
        &b_tensor,
        &a_slice,
        &b_slice,
        n,
        &mut keep_alive,
    );
    assert!(last.alloc_c_secs.is_finite());
    assert!(last.kernel_secs.is_finite());
    assert!(last.tensor_wrap_secs.is_finite());
    assert!(last.ops_gemm_secs.is_finite());
    assert!(last.tape_matmul_secs.is_finite());
    assert!(last.to_tensor_secs.is_finite());
    assert!(last.host_copy_secs.is_finite());
    assert!(last.checksum_secs.is_finite());
}
