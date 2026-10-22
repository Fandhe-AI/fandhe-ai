//! simdgroup 交互配置候補（`FRAG_INTERLEAVE_ENABLED`・function constant index 19。
//! 実装はイシュー #3211）の Mac mini 純カーネル A/B 計測（イシュー #3212。診断専用・
//! 本番コード不変）。すべて Metal 実機（Apple Silicon）依存の `#[ignore]` テストで、
//! CI では実行しない。型検査だけは `make check-cross-metal-tests` が担う。
//!
//! # 位置づけ
//!
//! - 規則の正（計測前に固定）: `docs/perf/logs/metal-frag-interleave-ab-macmini-3212/RULE.txt`
//! - 結論: `docs/perf/metal-gemm-steel-candidates.md` §10.8・§10.9
//! - 雛形: `gemm_m64_thin_tile_ab_diag_tests.rs`（#3209。巡回計測・hash・CPU 参照 parity）と
//!   `gemm_tile_select_macmini_diag_tests.rs`（#3181）。`gemm_reuse_phase_diag_tests` の
//!   `pub(crate)` 面（`median_of`・`WARMUP_TRIALS`・`MEASURED_TRIALS`）へ届くため
//!   `lib.rs` のクレートルート兄弟として置く。
//! - 呼び出し元: `docs/perf/logs/metal-frag-interleave-ab-macmini-3212/orchestrate_macmini.sh`。
//!   出力行は同ディレクトリの `aggregate.py` の正規表現と 1 対 1 で対応する
//!   （変更時は両方を更新する）。
//!
//! # 腕
//!
//! 2 インスタンス（ゲート OFF の `MetalGemm::new`／ゲート ON の
//! `MetalGemm::new_with_frag_interleave(ctx, true)`）を同一プロセスで構築する。
//! 主腕（判定に使う）: `cur`（OFF）と `fi`（ON）。どちらも
//! `select_for_verified_device` が返す同一構成。補助腕（記録のみ）:
//! `tile::CANDIDATES[1..=3]` それぞれの OFF／ON（`c1_off`・`c1_on` …）。
//! 純カーネル時間は encode だけを記録して `synchronize_with_gpu_timestamps` で取る
//! （`diag_encode_tiled_nn`）。
//!
//! # 出力行（`aggregate.py` と 1 対 1）
//!
//! - `DEVICE gpu_cores=<u32|none> soc="<s|none>" verified_tile_table=<s> split_k_enabled=<bool>`
//! - `TRACE n=<N> should_split_k=<none|plan> route=<r> cur=<l> resolved_cur=<l> unroll_acc_eff=<bool> classic=<bool> active_on=<bool>`
//! - `PARITY n=<N> arm=<cur|fi> fallback=<bool> ok=<true|false|skipped>`
//! - `BITIDENT n=<N> pair=<fi_vs_cur|cK_on_vs_cK_off> fallback=<bool> ok=<true|false|skipped>`
//! - `ARM n=<N> arm=<a> role=<judge|record_only> gate=<on|off> requested=<l> resolved=<l> fallback=<bool> active=<bool> checksum=<f> hash=<16hex>`
//! - `MEAS n=<N> arm=<a> kernel_gpu_median_ms=<v> q1=<v> q3=<v> encode_median_ms=<v> commit_wait_median_ms=<v> alloc_c_median_ms=<v> wall_sum_median_ms=<v>`
//!
//! # 前提
//!
//! `--test-threads=1` で実行する（`cached_context()` はプロセスワイド singleton で、
//! 1 バッチ 1 ラベルを assert するため）。入力は決定的シードの有限値のみ
//! （NaN・inf を含めない）。N=4096 の出力（64 MiB）は試行ごとに解放し全腕ぶんを保持
//! しない（16 GB 機向け）。`expect` は診断テストだけで使う。

use std::time::Instant;

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::device::{probe_gpu_core_count, probe_soc_brand_string};
use crate::gemm::MetalGemm;
use crate::gemm_reuse_phase_diag_tests::{MEASURED_TRIALS, WARMUP_TRIALS, median_of};
use crate::split_k_runtime::split_k_enabled;
use crate::tile::{self, GemmRoute, TileConfig};

/// 事前登録サイズ（RULE.txt 項 3。正方 NN）。
const SIZES: [usize; 5] = [256, 512, 1024, 2048, 4096];

/// 補助腕（記録のみ）に使う `tile::CANDIDATES` の添字（RULE.txt 項 4）。
const AUX_INDICES: [usize; 3] = [1, 2, 3];

/// 測る腕 1 つ。`on` は使うインスタンス（ゲート ON か）。
#[derive(Clone)]
struct Arm {
    name: String,
    on: bool,
    cfg: TileConfig,
    judge: bool,
}

/// `cur`・`fi`（判定対象）＋補助腕（記録のみ）を返す。`cur`／`fi` は先頭 2 要素。
fn arms_for(ctx: &MetalContext, n: usize) -> Vec<Arm> {
    let cur_cfg = tile::select_for_verified_device(n, n, n, ctx.verified_tile_table_device());
    let mut v = vec![
        Arm {
            name: "cur".into(),
            on: false,
            cfg: cur_cfg,
            judge: true,
        },
        Arm {
            name: "fi".into(),
            on: true,
            cfg: cur_cfg,
            judge: true,
        },
    ];
    for &i in &AUX_INDICES {
        for on in [false, true] {
            v.push(Arm {
                name: format!("c{i}_{}", if on { "on" } else { "off" }),
                on,
                cfg: tile::CANDIDATES[i],
                judge: false,
            });
        }
    }
    v
}

/// 決定的な有限値（xorshift64*。`[-1, 1)`。NaN・inf を含めない）。
fn fill_finite(seed: u64, len: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let r = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((r >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn square_ab(n: usize) -> (Vec<f32>, Vec<f32>) {
    (
        fill_finite(0x3212_a000 ^ (n as u64), n * n),
        fill_finite(0x3212_b000 ^ (n as u64), n * n),
    )
}

/// f32 出力の f64 逐次和（参考値。判定には使わない）。
fn checksum_f64(values: &[f32]) -> f64 {
    values.iter().fold(0.0f64, |acc, &v| acc + v as f64)
}

/// f32 出力の全要素ビット列に対する FNV-1a 64bit（#3181・#3209 と同型）。
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

/// 全要素 `to_bits` 一致（NaN 同士は payload を含め `to_bits` で比べる契約だが、
/// 入力は有限値のみのため NaN は出ない想定。出た場合は NaN クラス一致で扱う）。
fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            if x.is_nan() || y.is_nan() {
                x.is_nan() && y.is_nan()
            } else {
                x.to_bits() == y.to_bits()
            }
        })
}

/// `cfg` を `"<C番号|none>:<構成>"`（空白なし）で表す。
fn label_of(cfg: TileConfig) -> String {
    let tag = tile::CANDIDATES
        .iter()
        .position(|c| *c == cfg)
        .map_or_else(|| "none".to_string(), |i| format!("C{i}"));
    format!(
        "{tag}:bm{}xbn{}xbk{}_wm{}xwn{}_staged{}",
        cfg.bm, cfg.bn, cfg.bk, cfg.wm, cfg.wn, cfg.staged
    )
}

fn route_label(route: &GemmRoute) -> String {
    match route {
        GemmRoute::Classic(cfg) => format!("Classic/{}", label_of(*cfg)),
        GemmRoute::SplitK(plan) => format!("SplitK/{plan:?}").replace(' ', ""),
    }
}

/// そのインスタンス・解決後構成で交互配置が実際に有効になるか（シェーダ
/// `FRAG_INTERLEAVE_ACTIVE` のモデル。RULE.txt 項 5。split-K なし・Legacy・K=1 段）。
fn active_of(gemm: &MetalGemm, resolved: TileConfig) -> bool {
    tile::frag_interleave_active_model(
        resolved,
        gemm.frag_interleave_enabled(),
        gemm.unroll_acc_effective(resolved),
        false,
        true,
        1,
    )
}

/// 1 trial の区間計測値（秒）。
struct Sample {
    kernel_gpu: f64,
    encode: f64,
    commit_wait: f64,
    alloc_c: f64,
    resolved: TileConfig,
    output: Option<Vec<f32>>,
}

/// 腕 1 回ぶんを encode-only で記録し、commit + GPU 完了待ちで `kernel_gpu` を取る。
/// 1 バッチ・1 ラベル・`kernel_gpu <= commit_wait` を assert する（RULE.txt 項 7）。
#[allow(clippy::too_many_arguments)]
fn trial(
    ctx: &MetalContext,
    gemms: (&MetalGemm, &MetalGemm),
    arm: &Arm,
    a_buf: &MetalBuffer,
    b_buf: &MetalBuffer,
    n: usize,
    capture: bool,
) -> Sample {
    let gemm = if arm.on { gemms.1 } else { gemms.0 };
    let t = Instant::now();
    let c_buf = MetalBuffer::alloc_uninit_pooled(ctx, n * n).expect("C 確保に失敗した");
    let alloc_c = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let resolved = gemm
        .diag_encode_tiled_nn(ctx, a_buf, b_buf, &c_buf, n, n, n, arm.cfg)
        .expect("diag_encode_tiled_nn に失敗した");
    let encode = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let batches = ctx
        .synchronize_with_gpu_timestamps()
        .expect("synchronize に失敗した");
    let commit_wait = t.elapsed().as_secs_f64();
    assert_eq!(
        batches.len(),
        1,
        "1 trial = 1 バッチのはず（--test-threads=1 で実行すること）"
    );
    assert_eq!(
        batches[0].labels(),
        ["diag_encode_tiled_nn"],
        "想定外のラベル"
    );
    let kernel_gpu = batches[0]
        .kernel_gpu_secs()
        .expect("GPUStart／GPUEnd タイムスタンプが取れない");
    assert!(
        kernel_gpu >= 0.0 && kernel_gpu <= commit_wait,
        "kernel_gpu が不正"
    );
    let output = capture.then(|| c_buf.read_to_vec());
    Sample {
        kernel_gpu,
        encode,
        commit_wait,
        alloc_c,
        resolved,
        output,
    }
}

/// 2 インスタンス（OFF／ON）を構築し、ゲート値を確認する。
fn pair(ctx: &MetalContext) -> (MetalGemm, MetalGemm) {
    let off = MetalGemm::new(ctx).expect("OFF GEMM の構築に失敗した");
    let on = MetalGemm::new_with_frag_interleave(ctx, true).expect("ON GEMM の構築に失敗した");
    assert!(!off.frag_interleave_enabled());
    assert!(on.frag_interleave_enabled());
    (off, on)
}

/// 到達追跡（記録のみ）。N ごとの経路・解決構成・交互配置の有効判定を決定的に残す
/// （RULE.txt 項 5）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn frag_interleave_ab_reach_trace() {
    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    let (off, on) = pair(&ctx);
    let device = ctx.verified_tile_table_device();
    println!(
        "DEVICE gpu_cores={} soc=\"{}\" verified_tile_table={} split_k_enabled={}",
        probe_gpu_core_count().map_or_else(|| "none".to_string(), |v| v.to_string()),
        probe_soc_brand_string().unwrap_or_else(|| "none".to_string()),
        format!("{device:?}").replace(' ', ""),
        split_k_enabled()
    );
    for &n in &SIZES {
        let split = tile::should_split_k(n, n, n)
            .map_or_else(|| "none".to_string(), |p| format!("{p:?}").replace(' ', ""));
        let route = tile::select_route_for_verified_device(n, n, n, device);
        let classic = matches!(route, GemmRoute::Classic(_));
        let cur = tile::select_for_verified_device(n, n, n, device);
        let resolved = off
            .resolve_tile_config(&ctx, cur)
            .expect("構成の解決に失敗した");
        let resolved_on = on
            .resolve_tile_config(&ctx, cur)
            .expect("構成の解決に失敗した");
        assert_eq!(resolved, resolved_on, "OFF／ON で解決構成が異なる");
        let active = classic && split == "none" && active_of(&on, resolved);
        println!(
            "TRACE n={n} should_split_k={split} route={} cur={} resolved_cur={} unroll_acc_eff={} classic={classic} active_on={active}",
            route_label(&route),
            label_of(cur),
            label_of(resolved),
            on.unroll_acc_effective(resolved)
        );
    }
}

/// 主腕（`cur`・`fi`）が CPU 参照（FMA 契約）と REQ-2 複合判定で一致すること（前提ゲート。
/// tolerance は不変）。fallback する腕は記録してスキップ。`cur` の不合格のみ最後に assert。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn frag_interleave_ab_prereq_parity() {
    use fandhe_ai_backend_cpu::parity::{compare, matmul_reference_fma};

    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    let (off, on) = pair(&ctx);
    let mut cur_failed = false;
    for &n in &SIZES {
        let (a, b) = square_ab(n);
        let mut expected = vec![0.0f32; n * n];
        matmul_reference_fma(&a, &b, &mut expected, n, n, n).expect("CPU 参照の形状検証に失敗した");
        let a_buf = MetalBuffer::new_with_data(&ctx, &a).expect("A アップロード");
        let b_buf = MetalBuffer::new_with_data(&ctx, &b).expect("B アップロード");
        for arm in arms_for(&ctx, n).into_iter().filter(|a| a.judge) {
            let s = trial(&ctx, (&off, &on), &arm, &a_buf, &b_buf, n, true);
            if s.resolved != arm.cfg {
                println!("PARITY n={n} arm={} fallback=true ok=skipped", arm.name);
                continue;
            }
            let out = s.output.expect("capture=true");
            let report = compare(&out, &expected).expect("出力長が参照と一致しない");
            let pass = report.passes();
            println!("PARITY n={n} arm={} fallback=false ok={pass}", arm.name);
            if !pass {
                println!(
                    "PARITY-DETAIL n={n} arm={} fail_count={}/{} max_abs_diff={:.3e} max_rel_err={:.3e}",
                    arm.name,
                    report.fail_count,
                    report.total,
                    report.max_abs_diff,
                    report.max_rel_err
                );
                if arm.name == "cur" {
                    cur_failed = true;
                }
            }
        }
    }
    assert!(
        !cur_failed,
        "cur 腕が CPU 参照との複合判定に失敗した（系列を成立させない）"
    );
}

/// `fi` と `cur`、補助腕の ON と OFF の出力 bit 一致を記録する前提ゲート（RULE.txt 項 8）。
/// 不一致は `ok=false` で記録して継続する（集計側が NOT_ADOPTABLE にする）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn frag_interleave_ab_bitident_report() {
    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    let (off, on) = pair(&ctx);
    for &n in &SIZES {
        let (a, b) = square_ab(n);
        let a_buf = MetalBuffer::new_with_data(&ctx, &a).expect("A アップロード");
        let b_buf = MetalBuffer::new_with_data(&ctx, &b).expect("B アップロード");
        let arms = arms_for(&ctx, n);
        let run = |arm: &Arm| {
            let s = trial(&ctx, (&off, &on), arm, &a_buf, &b_buf, n, true);
            (s.resolved == arm.cfg, s.output.expect("capture=true"))
        };
        let report = |label: &str, base: &Arm, other: &Arm| {
            let (base_ok, base_out) = run(base);
            let (other_ok, other_out) = run(other);
            let ok = if !base_ok || !other_ok {
                "skipped".to_string()
            } else {
                bits_equal(&base_out, &other_out).to_string()
            };
            println!(
                "BITIDENT n={n} pair={label} fallback={} ok={ok}",
                !(base_ok && other_ok)
            );
        };
        report("fi_vs_cur", &arms[0], &arms[1]);
        for &i in &AUX_INDICES {
            let off_arm = arms
                .iter()
                .find(|a| a.name == format!("c{i}_off"))
                .expect("補助腕 OFF");
            let on_arm = arms
                .iter()
                .find(|a| a.name == format!("c{i}_on"))
                .expect("補助腕 ON");
            report(&format!("c{i}_on_vs_c{i}_off"), off_arm, on_arm);
        }
    }
}

/// 全腕を同一 N で巡回順に計測する（記録のみ。fallback は abort せず記録する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn frag_interleave_ab_kernel_gpu() {
    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    let (off, on) = pair(&ctx);
    for &n in &SIZES {
        let (a, b) = square_ab(n);
        let a_buf = MetalBuffer::new_with_data(&ctx, &a).expect("A アップロード");
        let b_buf = MetalBuffer::new_with_data(&ctx, &b).expect("B アップロード");
        drop((a, b));
        let arms = arms_for(&ctx, n);
        let num = arms.len();

        // 解決構成・hash（計測前に全腕を 1 回ずつ記録する）。
        for arm in &arms {
            let s = trial(&ctx, (&off, &on), arm, &a_buf, &b_buf, n, true);
            let out = s.output.as_ref().expect("capture=true");
            let gemm = if arm.on { &on } else { &off };
            println!(
                "ARM n={n} arm={} role={} gate={} requested={} resolved={} fallback={} active={} checksum={:.6} hash={:016x}",
                arm.name,
                if arm.judge { "judge" } else { "record_only" },
                if arm.on { "on" } else { "off" },
                label_of(arm.cfg),
                label_of(s.resolved),
                s.resolved != arm.cfg,
                active_of(gemm, s.resolved),
                checksum_f64(out),
                bits_hash_fnv1a64(out),
            );
        }

        for _ in 0..WARMUP_TRIALS {
            for arm in &arms {
                let _ = trial(&ctx, (&off, &on), arm, &a_buf, &b_buf, n, false);
            }
        }
        let mk = || -> Vec<Vec<f64>> {
            (0..num)
                .map(|_| Vec::with_capacity(MEASURED_TRIALS))
                .collect()
        };
        let (mut kg, mut enc, mut cw, mut al, mut wall) = (mk(), mk(), mk(), mk(), mk());
        for trial_i in 0..MEASURED_TRIALS {
            let offset = trial_i % num;
            for step in 0..num {
                let idx = (offset + step) % num;
                let s = trial(&ctx, (&off, &on), &arms[idx], &a_buf, &b_buf, n, false);
                kg[idx].push(s.kernel_gpu);
                enc[idx].push(s.encode);
                cw[idx].push(s.commit_wait);
                al[idx].push(s.alloc_c);
                wall[idx].push(s.alloc_c + s.encode + s.commit_wait);
            }
        }
        for (idx, arm) in arms.iter().enumerate() {
            let q = median_of(&kg[idx]);
            println!(
                "MEAS n={n} arm={} kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4} encode_median_ms={:.4} commit_wait_median_ms={:.4} alloc_c_median_ms={:.4} wall_sum_median_ms={:.4}",
                arm.name,
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
