//! `docs/cpu-gemm-prefetch-decision.md`（#489・#751 格下げ判断）を覆す
//! 帯域律速根拠の有無を実測するための診断テスト（イシュー #1319）。
//!
//! # 背景・目的
//!
//! `docs/perf/cpu-gemm-candle-cpu-retune.md` §2 は「1024/2048 で劣位・
//! 4096 で拮抗」という観測を「A packing 重複コスト（メモリ帯域）が
//! 演算量 2N³ に対して相対的に重い」仮説と結び付けた（候補 2:
//! `vld1q_f32_x3` 経路 prefetch）。`docs/cpu-gemm-prefetch-decision.md`
//! の 2026-08-19 追補は BLIS armv8a・matrixmultiply の一次ソース照合
//! から packed A/B への k ループ内 PRFM 省略（HW ストリーム
//! プリフェッチャー任せ）を確認し「原則不要」へ格下げしたが、本リポ
//! 自身の実機での帯域律速有無は未計測のままだった。本ファイルはこの
//! 空白を埋め、格下げ判断を覆すか維持するかを決める一次情報を採取する。
//!
//! `unsafe asm!`（PRFM 発行）は本ファイルにも本番経路にも一切追加
//! しない（イシュー #1319 の受入条件。着手にはユーザー承認が必要）。
//!
//! # 「帯域律速」の 2 系統（判定を誤らないための分離）
//!
//! | 系統 | 意味 | PRFM で解消できるか | 位置づけ |
//! |---|---|---|---|
//! | (a) GEMM 全体の DRAM トラフィック vs 実測到達帯域 | retune §2 の
//!   「A packing 重複コスト」仮説そのもの | できない（PRFM は移動
//!   バイト数を減らさない） | 文脈（[`row_panel_traffic_bytes`]・
//!   [`achievable_bandwidth_diag`]）。単独では承認依頼の根拠にしない |
//! | (b) マイクロカーネル k ループのロードレイテンシ露出 | packed
//!   パネルが L1 に無いとき HW プリフェッチャーが隠蔽しきれず FMA が
//!   ストールするか | できる可能性がある（PRFM 本来の用途） | 主判定
//!   （[`microkernel_residency_diag`]） |
//!
//! # 事前宣言ゲート（計測前に確定。以後変更しない）
//!
//! **G-b（主判定）**: `R_dram = streamed_dram の中央値 GFLOP/s
//! / l1_resident の中央値 GFLOP/s` を単スレッド・全コア同時の両条件で
//! 算出する。
//!
//! - 根拠あり: M4 Max で `R_dram <= 0.90` かつ 5 run 中 3 run 以上で
//!   `R_dram <= 0.95`、かつ DGX が矛盾しない（`R_dram <= 0.95`）。
//!   または DGX 単独で `R_dram <= 0.90`（5 run 中 3 run 以上 `<= 0.95`）
//! - 根拠なし（格下げ維持）: 両実機とも `R_dram >= 0.95`
//! - それ以外: 判定不可（undetermined。根拠ありへ格上げしない）
//!
//! **G-a（文脈）**: `Q = required_bw / achievable_bw`。`required_bw` は
//! [`row_panel_traffic_bytes`] の総トラフィック上界 ÷ RowPanel 実測時間
//! （`gemm_blis_variant_ab_1024_2048` の中央値 GFLOP/s から換算）。
//! `achievable_bw` は [`achievable_bandwidth_diag`] の実測値。
//! `Q >= 0.5` なら「DRAM 帯域圧迫あり」フラグを立てるが、単独では
//! asm 承認根拠にしない（G-b が主判定）。
//!
//! 集計・ゲート適用は `docs/perf/logs/cpu-gemm-prefetch-bandwidth-1319/
//! aggregate.py` が行う（本ファイルの `#[test]`／`#[ignore]` は
//! `println!` で生データを出すのみで assert による大小判定は行わない。
//! `gemm_reuse_phase_diag_tests.rs` と同じ「gating しない」方針）。
//!
//! # 実行方法
//!
//! ```text
//! cargo test -p fandhe-ai-backend-cpu --release -- --ignored microkernel_residency_diag --nocapture
//! cargo test -p fandhe-ai-backend-cpu --release -- --ignored achievable_bandwidth_diag --nocapture
//! ```
//!
//! RowPanel 実測時間（G-a 用）は既存 `gemm_blis::tests::
//! gemm_blis_variant_ab_1024_2048`（コード変更なし）を別途 5 回起動して
//! 採取する。

use crate::gemm_blis::microkernel::{Microkernel, NeonKernel};

/// マイクロカーネルタイル形状（`crate::gemm_blis::microkernel::neon`
/// の `MR`/`NR` と同値。診断専用の複製定数であり本番カーネル選択には
/// 一切関与しない）。
const MR: usize = NeonKernel::MR;
const NR: usize = NeonKernel::NR;

/// `gemm_blis/mod.rs` の本番既定ブロックサイズ（`MC`/`KC`/`NC`。
/// イシュー #1315 で KC=256 維持が確定済み）の診断用複製値。
/// [`row_panel_traffic_bytes`] のトラフィック上界モデル専用であり、
/// 本番の `default_blocks()` を変更・参照するものではない（値が
/// ドリフトした場合はモデルの参考値がずれるのみで安全側。ドリフト
/// 検知は本イシューのスコープ外）。
const MODEL_MC: usize = 128;
const MODEL_KC: usize = 256;
const MODEL_NC: usize = 512;

/// panel-pair 列を生成する関数ポインタの型（`microkernel_residency_diag`
/// のシナリオ表で使う。clippy `type_complexity` 回避のための別名）。
type MakePanelPairsFn = fn(u32) -> Vec<(Vec<f32>, Vec<f32>)>;

/// [`row_panel_traffic_bytes`] の返り値（各トラフィック要素とその合計。
/// バイト単位）。
#[derive(Debug, Clone, Copy)]
struct TrafficModel {
    b_read_bytes: u64,
    a_read_bytes: u64,
    packed_write_bytes: u64,
    c_rmw_bytes: u64,
    total_bytes: u64,
}

/// `gemm_blis_parallel`（RowPanel。行パネル 1 次元分割・
/// `panel_rows = m.div_ceil(num_threads)`）が 1 回の GEMM で動かす
/// DRAM トラフィックの**上界モデル**（純関数。系統 (a) の文脈情報）。
///
/// `docs/perf/cpu-gemm-candle-cpu-retune.md` §2 の重複コスト診断
/// （「B は各タスクが (jc,pc) ごとに重複 pack」「A は jc 反復回数ぶん
/// 重複 pack」）をバイト数へ翻訳したもので、SLC/L2 でのヒットは
/// 無視した上界（実際のトラフィックはこれ以下）。
///
/// - B 読み出し: 各スレッド（`num_threads` 本）が独立に B 全体
///   （`k * n` 要素）を pack し直すため `num_threads * k * n * 4`
/// - A 読み出し: jc 反復数（`n.div_ceil(nc)`）ぶん m×k 全体を重複 pack
///   するため `n.div_ceil(nc) * m * k * 4`
/// - packed 書き込み: 上記 A・B 読み出しと同じ要素数を packed バッファ
///   へ書き込むため合計は読み出しと同量
/// - C RMW: kc ブロック数（`k.div_ceil(kc)`）ぶん C 全体（`m * n` 要素）
///   を読み書き（read-modify-write）するため
///   `k.div_ceil(kc) * m * n * 4 * 2`
fn row_panel_traffic_bytes(
    m: usize,
    n: usize,
    k: usize,
    mc: usize,
    kc: usize,
    nc: usize,
    num_threads: usize,
) -> TrafficModel {
    let _ = mc; // MC は本モデルの他項に現れないが引数として残し将来の精緻化に備える。
    let elem = 4u64;
    let b_read_bytes = (num_threads as u64) * (k as u64) * (n as u64) * elem;
    let jc_reps = n.div_ceil(nc) as u64;
    let a_read_bytes = jc_reps * (m as u64) * (k as u64) * elem;
    let packed_write_bytes = a_read_bytes + b_read_bytes;
    let kc_reps = k.div_ceil(kc) as u64;
    let c_rmw_bytes = kc_reps * (m as u64) * (n as u64) * elem * 2;
    let total_bytes = b_read_bytes + a_read_bytes + packed_write_bytes + c_rmw_bytes;
    TrafficModel {
        b_read_bytes,
        a_read_bytes,
        packed_write_bytes,
        c_rmw_bytes,
        total_bytes,
    }
}

/// xorshift32 による決定的疑似乱数生成（テスト専用。`gemm_blis::tests`
/// の同名ヘルパーと同じアルゴリズムだが可視性の都合で複製する）。
/// `[-1.0, 1.0)` の範囲で有限値のみを返すため NaN/Inf 混入なし。
fn xorshift32_vec(seed: u32, len: usize) -> Vec<f32> {
    let mut state = seed.max(1);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // 上位ビットを使い [0, 1) へ正規化してから [-1, 1) へ写す。
        let normalized = (state >> 8) as f32 / (1u32 << 24) as f32;
        out.push(normalized * 2.0 - 1.0);
    }
    out
}

/// [`NeonKernel::run`] が期待する packed panel レイアウト
/// （`crate::gemm_blis::pack::pack_a`/`pack_b` と同一の
/// `p * MR + i` / `p * NR + j` 行優先。パッキング関数自体は
/// `pub(super)` で本ファイルから到達できないため、本診断は同じ
/// レイアウトを直接構築する。実際の値は xorshift32 の疑似乱数で
/// あり RowPanel 本番経路とは無関係〈カーネル自体の正しさは
/// `gemm_blis::microkernel` 側の既存テストが担保する〉）。
fn make_panel_pair(seed: u32, kc_len: usize) -> (Vec<f32>, Vec<f32>) {
    let ap = xorshift32_vec(seed, MR * kc_len);
    let bp = xorshift32_vec(seed.wrapping_add(1), kc_len * NR);
    (ap, bp)
}

/// 1 回の [`NeonKernel::run`] 呼び出しを行い `c_tile` をゼロから
/// 積算する（呼び出し間の値の蓄積による非有限化を防ぐため毎回
/// リセットする。全モード共通のオーバーヘッドのため比較に影響しない）。
fn run_once(kernel: NeonKernel, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
    c_tile.iter_mut().for_each(|v| *v = 0.0);
    kernel.run(ap, bp, c_tile, kc_len);
}

/// 残差感度計測（G-b）の 1 系列（`l1_resident`/`streamed_l2`/
/// `streamed_dram`）を実行し中央値 GFLOP/s を返す。
///
/// - `panel_pairs`: 走査対象の (A panel, B panel) 列（1 要素なら
///   `l1_resident`〈同一パネルを繰り返し使用〉、複数要素なら順次
///   走査してストリーミングを模す）
/// - `iters`: 呼び出し総数（`panel_pairs.len()` の倍数である必要は
///   ない。`iters % panel_pairs.len()` 番目のパネルを毎回選ぶ）
fn measure_kernel_throughput(
    kernel: NeonKernel,
    panel_pairs: &[(Vec<f32>, Vec<f32>)],
    iters: usize,
) -> f64 {
    assert!(
        !panel_pairs.is_empty(),
        "panel_pairs は空であってはならない"
    );
    let kc_len = panel_pairs[0].0.len() / MR;
    let mut c_tile = vec![0.0f32; MR * NR];
    // warmup（キャッシュ・分岐予測を安定させる。計測対象外）
    let warmup = (iters / 10).max(4);
    for i in 0..warmup {
        let (ap, bp) = &panel_pairs[i % panel_pairs.len()];
        run_once(kernel, ap, bp, &mut c_tile, kc_len);
    }
    let flops_per_call = 2.0 * (MR as f64) * (NR as f64) * (kc_len as f64);
    let start = std::time::Instant::now();
    for i in 0..iters {
        let (ap, bp) = &panel_pairs[i % panel_pairs.len()];
        run_once(kernel, ap, bp, &mut c_tile, kc_len);
    }
    let elapsed = start.elapsed().as_secs_f64();
    // c_tile を読み出しコンパイラによる呼び出し全体の最適化除去を防ぐ
    // （黒箱化。計測対象の作業が消し去られないようにする）。
    std::hint::black_box(&c_tile);
    let total_flops = flops_per_call * (iters as f64);
    total_flops / elapsed / 1e9
}

/// 単スレッド・全コア同時の 2 条件で `measure_kernel_throughput` を
/// 実行する。全コア条件は各スレッドが**独立**な panel 列を持つ
/// （キャッシュライン共有によるスキューを避けるため。§2 事前宣言
/// ゲートの「各スレッドが独立バッファを走査」に対応）。
fn measure_kernel_throughput_both_modes(
    kernel: NeonKernel,
    make_panel_pairs: impl Fn(u32) -> Vec<(Vec<f32>, Vec<f32>)> + Sync,
    iters: usize,
) -> (f64, f64) {
    let single_pairs = make_panel_pairs(1);
    let single = measure_kernel_throughput(kernel, &single_pairs, iters);

    // `crate::thread_limit::effective_num_threads` を使う（本番
    // `gemm_blis_parallel` と同じ既定並列度算出。`ThreadLimitReport::
    // detected_big_cores` を直接使わない理由: `BIG_CORE_LIMIT_ENABLED`
    // が `false`〈#1364 REJECT 確定〉のため本番では未使用の値であり、
    // DGX Spark GB10 では `cpu_capacity` sysfs 誤検出により
    // `Some(1)`（実質シングルスレッド化）を返しうる
    // 〈`docs/perf/cpu-gemm-default-thread-limit.md` §6〉。全コア条件が
    // 誤って単スレッド相当になるのを避けるため `effective_num_threads`
    // 経由で本番と同じフォールバック（`BIG_CORE_LIMIT_ENABLED=false`
    // 時は `rayon::current_num_threads()` をそのまま使う）を踏襲する。
    let num_threads = crate::thread_limit::effective_num_threads(rayon::current_num_threads());
    let per_thread: Vec<f64> = (0..num_threads).collect::<Vec<_>>().par_iter_map(|&t| {
        let pairs = make_panel_pairs(1000 + t as u32);
        measure_kernel_throughput(kernel, &pairs, iters)
    });
    let multi = per_thread.iter().sum::<f64>();
    (single, multi)
}

/// `.par_iter().map(...).collect()` の薄いラッパー（rayon 依存を
/// 局所化する目的の最小限のヘルパー。トレイト名は診断専用）。
trait ParIterMap<T> {
    fn par_iter_map<F, R>(&self, f: F) -> Vec<R>
    where
        F: Fn(&T) -> R + Sync + Send,
        R: Send;
}

impl<T: Sync> ParIterMap<T> for Vec<T> {
    fn par_iter_map<F, R>(&self, f: F) -> Vec<R>
    where
        F: Fn(&T) -> R + Sync + Send,
        R: Send,
    {
        use rayon::prelude::*;
        self.par_iter().map(f).collect()
    }
}

/// 総トラフィック量から生成する panel 列の要素数を決める。
/// `target_bytes` を 1 panel-pair あたりのバイト数
/// （`(MR*kc_len + kc_len*NR) * 4`）で割って個数を求める。
fn panel_pairs_for_target_bytes(
    seed_base: u32,
    kc_len: usize,
    target_bytes: usize,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    let bytes_per_pair = (MR * kc_len + kc_len * NR) * std::mem::size_of::<f32>();
    let count = (target_bytes / bytes_per_pair).max(1);
    (0..count)
        .map(|i| make_panel_pair(seed_base.wrapping_add(i as u32 * 2), kc_len))
        .collect()
}

const RESIDENCY_ITERS: usize = 4096;
/// `streamed_l2` の目標総量（L1〈本カーネルの packed panel 対〉より
/// 大きく LLC より小さい範囲。Apple M4 Max・DGX Spark GB10 とも
/// L2/SLC は 512 KiB を大きく上回るためこの値は「L1 に収まらない」
/// ことのみを保証する下限的な設計）。
const STREAMED_L2_TARGET_BYTES: usize = 512 * 1024;
/// `streamed_dram` の目標総量（両実機の L2/SLC を確実に上回り DRAM
/// トラフィックを強制する。プロセスあたり ~256 MiB は M4 Max 64 GB・
/// DGX Spark GB10 110 GB の空きに対し十分小さい）。
const STREAMED_DRAM_TARGET_BYTES: usize = 256 * 1024 * 1024;

/// (b) マイクロカーネル k ループの残差感度計測（G-b・主判定）。
///
/// packed パネルの常駐先を `l1_resident`（同一パネル反復再利用）→
/// `streamed_l2`（~512 KiB 循環）→ `streamed_dram`（~256 MiB 循環）
/// の順に変え、単スレッド・全コア同時それぞれで `NeonKernel::run`
/// のスループット（GFLOP/s）を計測する。HW ストリームプリフェッチャー
/// が DRAM ストリーミングを完全に隠蔽するなら 3 モードの GFLOP/s は
/// ほぼ同一（`R_dram = streamed_dram/l1_resident ≈ 1.0`）になるはず
/// で、有意に低下する（`R_dram` が事前宣言ゲート閾値を下回る）場合に
/// のみ「ロードレイテンシがマイクロカーネルへ露出している」＝PRFM
/// 導入の帯域律速根拠として扱う。
///
/// 出力は `mode=<name> threads=<single|multi> median_gflops=<f64>`
/// 形式の行（`docs/perf/logs/cpu-gemm-prefetch-bandwidth-1319/
/// aggregate.py` がこの形式を正規表現で読む）。
#[test]
#[ignore = "実機（Apple M4 Max・DGX Spark GB10）専用の残差感度計測。--release 実行推奨。イシュー #1319"]
fn microkernel_residency_diag() {
    let kernel = NeonKernel;
    let kc_len = MODEL_KC;

    let scenarios: [(&str, MakePanelPairsFn); 3] = [
        ("l1_resident", |seed| vec![make_panel_pair(seed, MODEL_KC)]),
        ("streamed_l2", |seed| {
            panel_pairs_for_target_bytes(seed, MODEL_KC, STREAMED_L2_TARGET_BYTES)
        }),
        ("streamed_dram", |seed| {
            panel_pairs_for_target_bytes(seed, MODEL_KC, STREAMED_DRAM_TARGET_BYTES)
        }),
    ];

    for (name, make_pairs) in scenarios {
        let (single, multi) =
            measure_kernel_throughput_both_modes(kernel, make_pairs, RESIDENCY_ITERS);
        println!("mode={name} threads=single median_gflops={single:.6}");
        println!("mode={name} threads=multi median_gflops={multi:.6}");
    }
    let _ = kc_len;
}

/// (a) 実測到達可能帯域（G-a・文脈情報）。512 MiB バッファに対する
/// 単スレッド／全コア同時（rayon `par_chunks`）の f64 read-sum
/// スループットを GB/s で出力する。`std::hint::black_box` で
/// 最適化除去を防ぐ。
///
/// 出力は `mode=<achievable_read> threads=<single|multi>
/// median_gbps=<f64>` 形式（`microkernel_residency_diag` と同じ
/// aggregate.py が読む）。
#[test]
#[ignore = "実機専用の到達帯域計測。--release 実行推奨。イシュー #1319"]
fn achievable_bandwidth_diag() {
    const BUF_LEN: usize = 512 * 1024 * 1024 / std::mem::size_of::<f32>();
    const WARMUP: usize = 3;
    const ROUNDS: usize = 20;

    let buf = xorshift32_vec(42, BUF_LEN);

    // 単スレッド read-sum。
    for _ in 0..WARMUP {
        let s: f64 = buf.iter().map(|&v| v as f64).sum();
        std::hint::black_box(s);
    }
    let start = std::time::Instant::now();
    for _ in 0..ROUNDS {
        let s: f64 = buf.iter().map(|&v| v as f64).sum();
        std::hint::black_box(s);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let bytes_total = (BUF_LEN * std::mem::size_of::<f32>() * ROUNDS) as f64;
    let gbps_single = bytes_total / elapsed / 1e9;
    println!("mode=achievable_read threads=single median_gbps={gbps_single:.6}");

    // 全コア同時 read-sum（rayon par_chunks）。
    use rayon::prelude::*;
    // `crate::thread_limit::effective_num_threads` を使う（本番
    // `gemm_blis_parallel` と同じ既定並列度算出。`ThreadLimitReport::
    // detected_big_cores` を直接使わない理由: `BIG_CORE_LIMIT_ENABLED`
    // が `false`〈#1364 REJECT 確定〉のため本番では未使用の値であり、
    // DGX Spark GB10 では `cpu_capacity` sysfs 誤検出により
    // `Some(1)`（実質シングルスレッド化）を返しうる
    // 〈`docs/perf/cpu-gemm-default-thread-limit.md` §6〉。全コア条件が
    // 誤って単スレッド相当になるのを避けるため `effective_num_threads`
    // 経由で本番と同じフォールバック（`BIG_CORE_LIMIT_ENABLED=false`
    // 時は `rayon::current_num_threads()` をそのまま使う）を踏襲する。
    let num_threads = crate::thread_limit::effective_num_threads(rayon::current_num_threads());
    let chunk_len = BUF_LEN.div_ceil(num_threads).max(1);
    for _ in 0..WARMUP {
        let s: f64 = buf
            .par_chunks(chunk_len)
            .map(|c| c.iter().map(|&v| v as f64).sum::<f64>())
            .sum();
        std::hint::black_box(s);
    }
    let start = std::time::Instant::now();
    for _ in 0..ROUNDS {
        let s: f64 = buf
            .par_chunks(chunk_len)
            .map(|c| c.iter().map(|&v| v as f64).sum::<f64>())
            .sum();
        std::hint::black_box(s);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let gbps_multi = bytes_total / elapsed / 1e9;
    println!("mode=achievable_read threads=multi median_gbps={gbps_multi:.6}");
}

#[cfg(test)]
mod model_tests {
    use super::*;

    /// N=2048・T=16 代表値でモデルが doc 記載のオーダー（約 900 MiB
    /// 総トラフィック・約 48 GB/s@914 GFLOP/s 相当）になることを固定
    /// する（CI 実行可・実機不要の純関数テスト）。
    #[test]
    fn row_panel_traffic_bytes_n2048_order_of_magnitude() {
        let model = row_panel_traffic_bytes(2048, 2048, 2048, MODEL_MC, MODEL_KC, MODEL_NC, 16);
        let mib = model.total_bytes as f64 / (1024.0 * 1024.0);
        // 上界モデルであり実測値そのものではないため、doc に記す約 900 MiB
        // 近傍（600〜1400 MiB）に収まることのみを固定する。
        assert!(
            (600.0..1400.0).contains(&mib),
            "total_bytes={mib:.1} MiB がモデル想定レンジ外"
        );
    }

    /// 総トラフィックは各要素の単純和と一致する（内訳の取り違えを防ぐ
    /// 回帰）。
    #[test]
    fn row_panel_traffic_bytes_total_matches_sum_of_parts() {
        let model = row_panel_traffic_bytes(1024, 1024, 1024, MODEL_MC, MODEL_KC, MODEL_NC, 8);
        let sum =
            model.b_read_bytes + model.a_read_bytes + model.packed_write_bytes + model.c_rmw_bytes;
        assert_eq!(model.total_bytes, sum);
    }

    /// スレッド数が増えるほど B 読み出し（重複 pack）が線形に増える
    /// （RowPanel の重複コスト構造そのものを固定する回帰）。
    #[test]
    fn row_panel_traffic_bytes_b_read_scales_with_threads() {
        let m1 = row_panel_traffic_bytes(1024, 1024, 1024, MODEL_MC, MODEL_KC, MODEL_NC, 1);
        let m2 = row_panel_traffic_bytes(1024, 1024, 1024, MODEL_MC, MODEL_KC, MODEL_NC, 4);
        assert_eq!(m2.b_read_bytes, m1.b_read_bytes * 4);
    }

    /// xorshift32_vec は有限値のみを返し `[-1.0, 1.0)` に収まる
    /// （NaN/Inf 混入なしの自己検証）。
    #[test]
    fn xorshift32_vec_produces_finite_bounded_values() {
        let v = xorshift32_vec(7, 1000);
        assert_eq!(v.len(), 1000);
        assert!(v.iter().all(|x| x.is_finite() && *x >= -1.0 && *x < 1.0));
    }

    /// panel_pairs_for_target_bytes は要素数 1 以上を返し、総バイト数が
    /// おおむね target に近いことを確認する（下限保証のみ）。
    #[test]
    fn panel_pairs_for_target_bytes_produces_nonempty() {
        let pairs = panel_pairs_for_target_bytes(1, MODEL_KC, 1024 * 1024);
        assert!(!pairs.is_empty());
    }

    /// `make_panel_pair` が [`NeonKernel::run`] の契約
    /// （`ap.len() == MR*kc_len`・`bp.len() == kc_len*NR`）を満たし、
    /// 呼び出しが panic せず有限の結果を返すことを確認する（実機不要）。
    #[test]
    fn neon_kernel_accepts_synthetic_panel_pair() {
        let (ap, bp) = make_panel_pair(3, MODEL_KC);
        assert_eq!(ap.len(), MR * MODEL_KC);
        assert_eq!(bp.len(), MODEL_KC * NR);
        let mut c_tile = vec![0.0f32; MR * NR];
        NeonKernel.run(&ap, &bp, &mut c_tile, MODEL_KC);
        assert!(c_tile.iter().all(|v| v.is_finite()));
    }
}
