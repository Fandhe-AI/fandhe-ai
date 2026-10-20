//! Mac mini の Metal GEMM タイル選択の追跡と、全 `CANDIDATES` の純カーネル時間計測
//! （イシュー #3181。診断専用・本番コード不変）。
//! すべて Metal 実機（Apple Silicon）依存の `#[ignore]` テストで、CI では実行しない。
//! 型検査だけは `make check-cross-metal-tests` が担う。
//!
//! # 位置づけ
//!
//! - 規則の正（計測前に固定）: `docs/perf/logs/metal-tile-macmini-3181/RULE.txt`
//! - 結論: `docs/perf/metal-gemm-tile-select-macmini.md`
//! - 雛形: `gemm_steel_candidate_diag_tests.rs`（候補の強制実行・hash・CPU 参照 parity）・
//!   `gemm_tile_class_diag_tests.rs`。`gemm_reuse_phase_diag_tests` の `pub(crate)` 面
//!   （`measure_one_phase_trial`・`gen_square_ab`・`median_of`）へ到達するため
//!   `lib.rs` のクレートルート兄弟モジュールとして置く。
//! - 呼び出し元: `docs/perf/logs/metal-tile-macmini-3181/orchestrate_macmini.sh`。
//!   出力行は同ディレクトリの `aggregate.py` の正規表現と 1 対 1 で対応する
//!   （変更時は両方を更新する）。
//!
//! # テスト構成
//!
//! - 記録のみ: `tile_select_trace_square_sizes`（本番 `dispatch_auto` がどのタイルを
//!   選ぶかの追跡）・`tile_candidates_kernel_gpu_square_sizes`（全候補の計測）
//! - assert あり（前提ゲート）: `tile_candidates_match_cpu_reference`
//!   （REQ-2 `assert_parity`。tolerance は不変）
//!
//! # 前提
//!
//! `--test-threads=1` で実行する。N=4096 の出力（64 MiB）は試行ごとに解放し、
//! 全候補ぶんを保持しない（16 GB 機向け）。
//!
//! # プロダクションコード不変
//!
//! `tile.rs`・`gemm.rs`・`shaders/` は無変更。選択ロジックと tolerance には触れない。

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::device::{probe_gpu_core_count, probe_soc_brand_string};
use crate::gemm::MetalGemm;
use crate::gemm_reuse_phase_diag_tests::{
    MEASURED_TRIALS, WARMUP_TRIALS, gen_square_ab, measure_one_phase_trial, median_of,
};
use crate::split_k_runtime::split_k_enabled;
use crate::tile::{self, GemmRoute, TileConfig};

/// 事前登録サイズ（RULE.txt 項 3）。
const SIZES: [usize; 5] = [256, 512, 1024, 2048, 4096];

/// f32 出力の f64 逐次和（参考値。判定には使わない）。
fn checksum_f64(values: &[f32]) -> f64 {
    values.iter().fold(0.0f64, |acc, &v| acc + v as f64)
}

/// f32 出力の全要素ビット列に対する FNV-1a 64bit（bit 一致判定用。
/// `gemm_steel_candidate_diag_tests::bits_hash_fnv1a64` と同型）。
fn bits_hash_fnv1a64(values: &[f32]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    values.iter().fold(OFFSET, |h, v| {
        v.to_bits()
            .to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(PRIME))
    })
}

/// `cfg` を `"<CANDIDATES の番号|none>:<構成>"`（空白なし）で表す。
fn label_of(cfg: TileConfig) -> String {
    let idx = tile::CANDIDATES
        .iter()
        .position(|c| *c == cfg)
        .map_or_else(|| "none".to_string(), |i| i.to_string());
    format!(
        "{idx}:bm{}xbn{}xbk{}_wm{}xwn{}_staged{}",
        cfg.bm, cfg.bn, cfg.bk, cfg.wm, cfg.wn, cfg.staged
    )
}

/// 経路を `Classic/<label>` または `SplitK/<plan>`（空白なし）の文字列にする。
fn route_label(route: &GemmRoute) -> String {
    match route {
        GemmRoute::Classic(cfg) => format!("Classic/{}", label_of(*cfg)),
        GemmRoute::SplitK(plan) => format!("SplitK/{plan:?}").replace(' ', ""),
    }
}

/// 1 回の `dispatch_tiled_prepared` 実行結果（C 行列）。
fn run_tiled(
    ctx: &MetalContext,
    gemm: &MetalGemm,
    a: &[f32],
    b: &[f32],
    n: usize,
    cfg: TileConfig,
) -> Vec<f32> {
    let a_buf = MetalBuffer::new_with_data(ctx, a).expect("A アップロードに失敗した");
    let b_buf = MetalBuffer::new_with_data(ctx, b).expect("B アップロードに失敗した");
    let c_buf = MetalBuffer::new_zeroed(ctx, n * n).expect("C 確保に失敗した");
    gemm.dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, n, n, n, cfg)
        .expect("dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");
    c_buf.read_to_vec()
}

/// 本番 `dispatch_auto` が各 N でどの経路・タイルを選ぶかを記録する（記録のみ）。
///
/// `dispatch_auto_with_route` が返す `Classic(cfg)` は「要求した cfg」で、fallback 後の
/// 解決構成ではないため、`resolve_tile_config` の結果を `resolved=` に別記する。
///
/// 出力行（`aggregate.py` と一致させる）:
/// - `DEVICE gpu_cores=<u32|none> soc="<s|none>" verified_m4_max=<bool> split_k_enabled=<bool>`
/// - `TRACE N=<n> select=<l> select_for_device=<l> should_split_k=<none|plan> route=<r> executed_route=<r> resolved=<l>`
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tile_select_trace_square_sizes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("MetalGemm::new に失敗した");
    let verified = ctx.verified_m4_max_gpu_core_count();
    println!(
        "DEVICE gpu_cores={} soc=\"{}\" verified_m4_max={} split_k_enabled={}",
        probe_gpu_core_count().map_or_else(|| "none".to_string(), |v| v.to_string()),
        probe_soc_brand_string().unwrap_or_else(|| "none".to_string()),
        verified.is_some(),
        split_k_enabled()
    );
    for n in SIZES {
        let sel = tile::select(n, n, n);
        let sel_dev = tile::select_for_device(n, n, n, verified);
        let split = tile::should_split_k(n, n, n)
            .map_or_else(|| "none".to_string(), |p| format!("{p:?}").replace(' ', ""));
        let route = tile::select_route_for_device(n, n, n, verified);
        let (a, b) = gen_square_ab(0x3181_a000 ^ (n as u64), n);
        let (_out, executed) = gemm
            .dispatch_auto_with_route(&ctx, &a, &b, n, n, n)
            .expect("dispatch_auto_with_route に失敗した");
        // 実際に dispatch される classic 構成（SplitK のときは sel_dev を参考に解決する）。
        let requested = match &executed {
            GemmRoute::Classic(cfg) => *cfg,
            GemmRoute::SplitK(_) => sel_dev,
        };
        let resolved = gemm
            .resolve_tile_config(&ctx, requested)
            .expect("構成の解決に失敗した");
        println!(
            "TRACE N={n} select={} select_for_device={} should_split_k={split} route={} executed_route={} resolved={}",
            label_of(sel),
            label_of(sel_dev),
            route_label(&route),
            route_label(&executed),
            label_of(resolved)
        );
    }
}

/// 全 `CANDIDATES` × SIZES が CPU 参照（FMA 契約）と REQ-2 複合判定で一致すること
/// （前提ゲート。tolerance は不変）。fallback する候補はスキップして記録する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tile_candidates_match_cpu_reference() {
    use fandhe_ai_backend_cpu::parity::{assert_parity, matmul_reference_fma};

    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("MetalGemm::new に失敗した");
    for n in SIZES {
        let (a, b) = gen_square_ab(0x3181_a000 ^ (n as u64), n);
        let mut expected = vec![0.0f32; n * n];
        matmul_reference_fma(&a, &b, &mut expected, n, n, n)
            .expect("CPU 参照実装の形状検証に失敗した");
        for (idx, cfg) in tile::CANDIDATES.iter().copied().enumerate() {
            let resolved = gemm
                .resolve_tile_config(&ctx, cfg)
                .expect("構成の解決に失敗した");
            if resolved != cfg {
                println!(
                    "PARITY N={n} cand={idx} fallback=true resolved={}",
                    label_of(resolved)
                );
                continue;
            }
            let out = run_tiled(&ctx, &gemm, &a, &b, n, cfg);
            assert_parity(
                &format!("metal tile-select N={n} cand={idx}"),
                &out,
                &expected,
            );
            println!("PARITY N={n} cand={idx} fallback=false ok=true");
        }
    }
}

/// 全 `CANDIDATES` を同一形状で強制実行し、純カーネル時間（GPU タイムスタンプ）などを
/// trial ごとに開始オフセットを回して計測する（記録のみ）。fallback は abort せず記録する。
///
/// 出力行（`aggregate.py` と一致させる）:
/// - `CAND N=<n> idx=<i> requested=<l> resolved=<l> fallback=<bool> selected=<bool> checksum=<f> hash=<16hex> bit_identical_vs_selected=<bool>`
/// - `CAND N=<n> idx=<i> kernel_gpu_median_ms=<v> q1=<v> q3=<v> encode_median_ms=<v> commit_wait_median_ms=<v> alloc_c_median_ms=<v> wall_sum_median_ms=<v>`
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tile_candidates_kernel_gpu_square_sizes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("MetalGemm::new に失敗した");
    let verified = ctx.verified_m4_max_gpu_core_count();
    let num = tile::CANDIDATES.len();

    for n in SIZES {
        let cfgs: Vec<TileConfig> = tile::CANDIDATES.to_vec();
        let resolved: Vec<TileConfig> = cfgs
            .iter()
            .map(|c| {
                gemm.resolve_tile_config(&ctx, *c)
                    .expect("構成の解決に失敗した")
            })
            .collect();
        let selected_cfg = tile::select_for_device(n, n, n, verified);
        let selected_idx = cfgs.iter().position(|c| *c == selected_cfg);
        let (a, b) = gen_square_ab(0x3181_a000 ^ (n as u64), n);

        // trial 0: checksum・hash・選択タイルとの bit 一致（参考。タイル違いは契約外）。
        let selected_hash = selected_idx.map(|i| {
            let out = run_tiled(&ctx, &gemm, &a, &b, n, cfgs[i]);
            bits_hash_fnv1a64(&out)
        });
        for idx in 0..num {
            let out = run_tiled(&ctx, &gemm, &a, &b, n, cfgs[idx]);
            let hash = bits_hash_fnv1a64(&out);
            println!(
                "CAND N={n} idx={idx} requested={} resolved={} fallback={} selected={} checksum={:.6} hash={hash:016x} bit_identical_vs_selected={}",
                label_of(cfgs[idx]),
                label_of(resolved[idx]),
                resolved[idx] != cfgs[idx],
                Some(idx) == selected_idx,
                checksum_f64(&out),
                selected_hash == Some(hash),
            );
        }

        // 出力を全保持するとメモリが膨らむため、試行ごとに keep_alive を clear する。
        let mut keep: Vec<Vec<Vec<f32>>> = (0..num).map(|_| Vec::with_capacity(1)).collect();
        for _ in 0..WARMUP_TRIALS {
            for idx in 0..num {
                let _ = measure_one_phase_trial(&ctx, &gemm, &a, &b, n, cfgs[idx], &mut keep[idx]);
                keep[idx].clear();
            }
        }
        let mk = || -> Vec<Vec<f64>> {
            (0..num)
                .map(|_| Vec::with_capacity(MEASURED_TRIALS))
                .collect()
        };
        let (mut kg, mut enc, mut cw, mut al, mut wall) = (mk(), mk(), mk(), mk(), mk());
        for trial in 0..MEASURED_TRIALS {
            let offset = trial % num;
            for step in 0..num {
                let idx = (offset + step) % num;
                let s = measure_one_phase_trial(&ctx, &gemm, &a, &b, n, cfgs[idx], &mut keep[idx]);
                keep[idx].clear();
                kg[idx].push(s.kernel_gpu_secs);
                enc[idx].push(s.encode_secs);
                cw[idx].push(s.commit_wait_secs);
                al[idx].push(s.alloc_c_secs);
                wall[idx].push(s.alloc_c_secs + s.encode_secs + s.commit_wait_secs);
            }
        }
        for idx in 0..num {
            let q = median_of(&kg[idx]);
            println!(
                "CAND N={n} idx={idx} kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4} encode_median_ms={:.4} commit_wait_median_ms={:.4} alloc_c_median_ms={:.4} wall_sum_median_ms={:.4}",
                q.median * 1e3,
                q.q1 * 1e3,
                q.q3 * 1e3,
                median_of(&enc[idx]).median * 1e3,
                median_of(&cw[idx]).median * 1e3,
                median_of(&al[idx]).median * 1e3,
                median_of(&wall[idx]).median * 1e3,
            );
        }
    }
}
