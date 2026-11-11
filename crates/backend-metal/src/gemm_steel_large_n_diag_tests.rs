//! N≥1024 限定の steel 候補 T0U-LU-FB（`tile::STEEL_LARGE_N_ENABLED`・
//! `MetalGemm::new_with_steel_large_n`。イシュー #4160。#4166 で本番既定 ON へ結線）の
//! Mac mini 純カーネル A/B 計測
//! （診断専用・本番コード不変）。すべて Metal 実機（Apple Silicon）依存の `#[ignore]`
//! テストで CI では実行しない。型検査だけは `make check-cross-metal-tests` が担う。
//!
//! # 位置づけ
//!
//! - 規則の正（計測前に固定）: `docs/perf/logs/metal-steel-large-n-ab-macmini-4160/RULE.txt`
//!   （雛形は同ディレクトリの `RULE.template.txt`）
//! - 結論の記入先: `docs/perf/metal-gemm-steel-candidates.md` §11
//! - 雛形: `gemm_frag_interleave_ab_diag_tests.rs`（#3212）。`gemm_reuse_phase_diag_tests` の
//!   `pub(crate)` 面（`median_of`・`WARMUP_TRIALS`・`MEASURED_TRIALS`）へ届くため
//!   `lib.rs` のクレートルート兄弟として置く。
//! - 呼び出し元: 同ディレクトリの `orchestrate_macmini.sh`。出力行は `aggregate.py` の
//!   正規表現と 1 対 1 で対応する（変更時は両方を更新する）。
//!
//! # 腕
//!
//! - `cur`: `MetalGemm::new_with_steel_large_n(ctx, false)`（結線前の表経路 = 現行 M4Base 表。
//!   判定の基準。結線後は `MetalGemm::new` が `sl` と同じになるため明示的に OFF で作る）
//! - `sl`: `MetalGemm::new_with_steel_large_n(ctx, true)`（判定の対象。結線後の本番既定と同じ）
//! - `ref`: `new_with_steel_candidate(ctx, true, true, true)` で `CANDIDATES[0]` を強制
//!   （bit 一致の照合専用。時間は記録のみ）
//! - `cur_aa`: `cur` と同構成（OFF）の 2 本目（A/A 対照。記録のみ。判定に使わない）
//!
//! `cur`／`sl`／`cur_aa` は本番 reuse 経路（`dispatch_auto_resident`）と同じ classic 計画を
//! `diag_encode_classic_auto_nn` で encode する。
//!
//! # 出力行（`aggregate.py` と 1 対 1）
//!
//! - `DEVICE gpu_cores=<u32|none> soc="<s|none>" verified_tile_table=<s> split_k_enabled=<bool>`
//! - `TRACE n=<N> arm=<cur|sl> cfg=<l> profile=<table|steel_large_n> route=<Classic|SplitK> fallback=<bool>`
//! - `PARITY n=<N> arm=<cur|sl> ok=<true|false>`
//! - `BITIDENT n=<N> pair=<sl_vs_ref|sl_vs_cur> ok=<true|false>`
//! - `MEAS n=<N> arm=<a> role=<judge|record_only> kernel_gpu_ms=<v> encode_ms=<v> commit_wait_ms=<v> alloc_c_ms=<v> wall_sum_ms=<v> hash=<16hex>`
//!
//! # 前提
//!
//! `--test-threads=1` で実行する（1 バッチ 1 ラベルを assert するため）。入力は決定的
//! シードの有限値のみ。N=4096 の出力は試行ごとに解放する（16 GB 機向け）。
//! `expect` は診断テストだけで使う。

use std::time::Instant;

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::device::{probe_gpu_core_count, probe_soc_brand_string};
use crate::gemm::{GateProfile, MetalGemm};
use crate::gemm_reuse_phase_diag_tests::{MEASURED_TRIALS, WARMUP_TRIALS, median_of};
use crate::split_k_runtime::split_k_enabled;
use crate::tile::{self, GemmRoute, TileConfig};

/// 事前登録サイズ（正方 NN）。
const SIZES: [usize; 5] = [256, 512, 1024, 2048, 4096];

/// 実測対象（opt-in が効く）サイズ。これ未満は no-op であることを確認する。
fn in_scope(n: usize) -> bool {
    n >= 1024
}

/// 腕の識別。`cur`／`sl`／`cur_aa` は auto 計画、`ref` は `CANDIDATES[0]` 強制。
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArmKind {
    Cur,
    Sl,
    Ref,
    CurAa,
}

impl ArmKind {
    fn name(self) -> &'static str {
        match self {
            ArmKind::Cur => "cur",
            ArmKind::Sl => "sl",
            ArmKind::Ref => "ref",
            ArmKind::CurAa => "cur_aa",
        }
    }
    fn judge(self) -> bool {
        matches!(self, ArmKind::Cur | ArmKind::Sl)
    }
}

const ALL_ARMS: [ArmKind; 4] = [ArmKind::Cur, ArmKind::Sl, ArmKind::Ref, ArmKind::CurAa];

struct Gemms {
    cur: MetalGemm,
    sl: MetalGemm,
    r#ref: MetalGemm,
    cur_aa: MetalGemm,
}

fn build(ctx: &MetalContext) -> Gemms {
    let cur = MetalGemm::new_with_steel_large_n(ctx, false).expect("cur の構築に失敗した");
    let sl = MetalGemm::new_with_steel_large_n(ctx, true).expect("sl の構築に失敗した");
    let r =
        MetalGemm::new_with_steel_candidate(ctx, true, true, true).expect("ref の構築に失敗した");
    let cur_aa = MetalGemm::new_with_steel_large_n(ctx, false).expect("cur_aa の構築に失敗した");
    assert!(!cur.steel_large_n_enabled() && !cur_aa.steel_large_n_enabled());
    assert!(sl.steel_large_n_enabled());
    Gemms {
        cur,
        sl,
        r#ref: r,
        cur_aa,
    }
}

/// 決定的な有限値（xorshift64*。`[-1, 1)`）。
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
        fill_finite(0x4160_a000 ^ (n as u64), n * n),
        fill_finite(0x4160_b000 ^ (n as u64), n * n),
    )
}

/// f32 出力の全要素ビット列に対する FNV-1a 64bit。
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

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
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

fn profile_label(p: GateProfile) -> &'static str {
    match p {
        GateProfile::Instance => "table",
        GateProfile::SteelLargeN => "steel_large_n",
    }
}

struct Sample {
    kernel_gpu: f64,
    encode: f64,
    commit_wait: f64,
    alloc_c: f64,
    resolved: TileConfig,
    profile: GateProfile,
    output: Option<Vec<f32>>,
}

/// 腕 1 回ぶんを encode-only で記録し、commit + GPU 完了待ちで `kernel_gpu` を取る。
fn trial(
    ctx: &MetalContext,
    g: &Gemms,
    arm: ArmKind,
    a_buf: &MetalBuffer,
    b_buf: &MetalBuffer,
    n: usize,
    capture: bool,
) -> Sample {
    let t = Instant::now();
    let c_buf = MetalBuffer::alloc_uninit_pooled(ctx, n * n).expect("C 確保に失敗した");
    let alloc_c = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let (resolved, profile) = match arm {
        ArmKind::Cur => g
            .cur
            .diag_encode_classic_auto_nn(ctx, a_buf, b_buf, &c_buf, n),
        ArmKind::Sl => {
            g.sl.diag_encode_classic_auto_nn(ctx, a_buf, b_buf, &c_buf, n)
        }
        ArmKind::CurAa => g
            .cur_aa
            .diag_encode_classic_auto_nn(ctx, a_buf, b_buf, &c_buf, n),
        ArmKind::Ref => g
            .r#ref
            .diag_encode_tiled_nn(ctx, a_buf, b_buf, &c_buf, n, n, n, tile::CANDIDATES[0])
            .map(|r| (r, GateProfile::Instance)),
    }
    .expect("encode に失敗した");
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
        profile,
        output,
    }
}

fn print_device(ctx: &MetalContext) {
    let device = ctx.verified_tile_table_device();
    println!(
        "DEVICE gpu_cores={} soc=\"{}\" verified_tile_table={} split_k_enabled={}",
        probe_gpu_core_count().map_or_else(|| "none".to_string(), |v| v.to_string()),
        probe_soc_brand_string().unwrap_or_else(|| "none".to_string()),
        format!("{device:?}").replace(' ', ""),
        split_k_enabled()
    );
}

/// 本番既定が ON であること（結線後の実機側ドリフト検査。Linux の `tile::` 単体テストと対）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn steel_large_n_default_is_on() {
    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    let g = MetalGemm::new(&ctx).expect("構築に失敗した");
    assert!(g.steel_large_n_enabled());
}

/// 到達追跡。N ごとの解決構成・ゲート組・経路・フォールバック有無を決定的に残す。
/// 期待: N≥1024 の `sl` は `profile=steel_large_n cfg=C0 route=Classic fallback=false`、
/// N≤512 は `profile=table`（`cur` と同一）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn steel_large_n_reach_trace() {
    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    let g = build(&ctx);
    print_device(&ctx);
    let device = ctx.verified_tile_table_device();
    for &n in &SIZES {
        let route = match tile::select_route_for_verified_device(n, n, n, device) {
            GemmRoute::Classic(_) => "Classic",
            GemmRoute::SplitK(_) => "SplitK",
        };
        for (name, gemm) in [("cur", &g.cur), ("sl", &g.sl)] {
            let (requested, profile, resolved) = gemm
                .diag_classic_plan_resolved(&ctx, n, n, n)
                .expect("計画の解決に失敗した");
            println!(
                "TRACE n={n} arm={name} cfg={} profile={} route={route} fallback={}",
                label_of(resolved),
                profile_label(profile),
                requested != resolved
            );
            if name == "sl" && in_scope(n) {
                assert_eq!(profile, GateProfile::SteelLargeN, "n={n}");
                assert_eq!(resolved, tile::CANDIDATES[0], "n={n}");
                assert_eq!(requested, resolved, "n={n}: フォールバックが発生した");
                assert_eq!(route, "Classic", "n={n}");
            } else {
                assert_eq!(profile, GateProfile::Instance, "n={n} arm={name}");
            }
        }
    }
}

/// bit 一致。N≥1024 は `sl` と `ref`（同じカーネル）、N≤512 は `sl` と `cur`（no-op）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn steel_large_n_bit_identity() {
    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    let g = build(&ctx);
    for &n in &SIZES {
        let (a, b) = square_ab(n);
        let a_buf = MetalBuffer::new_with_data(&ctx, &a).expect("A アップロード");
        let b_buf = MetalBuffer::new_with_data(&ctx, &b).expect("B アップロード");
        drop((a, b));
        let sl = trial(&ctx, &g, ArmKind::Sl, &a_buf, &b_buf, n, true);
        let sl_out = sl.output.expect("capture=true");
        if in_scope(n) {
            let r = trial(&ctx, &g, ArmKind::Ref, &a_buf, &b_buf, n, true);
            let ok = bits_equal(&sl_out, &r.output.expect("capture=true"));
            println!("BITIDENT n={n} pair=sl_vs_ref ok={ok}");
            assert!(ok, "n={n}: sl と ref が bit 不一致");
        } else {
            let c = trial(&ctx, &g, ArmKind::Cur, &a_buf, &b_buf, n, true);
            assert_eq!(sl.resolved, c.resolved, "n={n}: no-op のはずが構成が異なる");
            let ok = bits_equal(&sl_out, &c.output.expect("capture=true"));
            println!("BITIDENT n={n} pair=sl_vs_cur ok={ok}");
            assert!(ok, "n={n}: sl と cur が bit 不一致（no-op 違反）");
        }
    }
}

/// `cur`・`sl` が CPU 参照（FMA 契約）と REQ-2 統一複合判定で一致すること（前提ゲート。
/// tolerance は不変）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn steel_large_n_prereq_parity() {
    use fandhe_ai_backend_cpu::parity::{compare, matmul_reference_fma};

    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    let g = build(&ctx);
    for &n in &SIZES {
        let (a, b) = square_ab(n);
        let mut expected = vec![0.0f32; n * n];
        matmul_reference_fma(&a, &b, &mut expected, n, n, n).expect("CPU 参照の形状検証に失敗した");
        let a_buf = MetalBuffer::new_with_data(&ctx, &a).expect("A アップロード");
        let b_buf = MetalBuffer::new_with_data(&ctx, &b).expect("B アップロード");
        drop((a, b));
        for arm in [ArmKind::Cur, ArmKind::Sl] {
            let s = trial(&ctx, &g, arm, &a_buf, &b_buf, n, true);
            let out = s.output.expect("capture=true");
            let report = compare(&out, &expected).expect("出力長が参照と一致しない");
            let pass = report.passes();
            println!("PARITY n={n} arm={} ok={pass}", arm.name());
            assert!(
                pass,
                "n={n} arm={}: fail_count={}/{} max_abs_diff={:.3e} max_rel_err={:.3e}",
                arm.name(),
                report.fail_count,
                report.total,
                report.max_abs_diff,
                report.max_rel_err
            );
        }
    }
}

/// 全腕を同一 N で巡回順に計測する（記録のみ。判定は `aggregate.py`）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn steel_large_n_kernel_gpu() {
    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    let g = build(&ctx);
    let num = ALL_ARMS.len();
    for &n in &SIZES {
        let (a, b) = square_ab(n);
        let a_buf = MetalBuffer::new_with_data(&ctx, &a).expect("A アップロード");
        let b_buf = MetalBuffer::new_with_data(&ctx, &b).expect("B アップロード");
        drop((a, b));

        let mut hashes = Vec::with_capacity(num);
        for arm in ALL_ARMS {
            let s = trial(&ctx, &g, arm, &a_buf, &b_buf, n, true);
            hashes.push(bits_hash_fnv1a64(s.output.as_ref().expect("capture=true")));
            let _ = (s.resolved, s.profile);
        }
        for _ in 0..WARMUP_TRIALS {
            for arm in ALL_ARMS {
                let _ = trial(&ctx, &g, arm, &a_buf, &b_buf, n, false);
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
                let s = trial(&ctx, &g, ALL_ARMS[idx], &a_buf, &b_buf, n, false);
                kg[idx].push(s.kernel_gpu);
                enc[idx].push(s.encode);
                cw[idx].push(s.commit_wait);
                al[idx].push(s.alloc_c);
                wall[idx].push(s.alloc_c + s.encode + s.commit_wait);
            }
        }
        for (idx, arm) in ALL_ARMS.iter().enumerate() {
            println!(
                "MEAS n={n} arm={} role={} kernel_gpu_ms={:.4} encode_ms={:.4} commit_wait_ms={:.4} alloc_c_ms={:.4} wall_sum_ms={:.4} hash={:016x}",
                arm.name(),
                if arm.judge() { "judge" } else { "record_only" },
                median_of(&kg[idx]).median * 1e3,
                median_of(&enc[idx]).median * 1e3,
                median_of(&cw[idx]).median * 1e3,
                median_of(&al[idx]).median * 1e3,
                median_of(&wall[idx]).median * 1e3,
                hashes[idx],
            );
        }
    }
}
