//! m=64 級 GEMM 向け細タイル候補（`tile::M64_THIN_CANDIDATES`・イシュー #3208。ゲート
//! `M64_THIN_TILE_ENABLED` は既定 OFF）の Mac mini 純カーネル A/B 計測（イシュー #3209。
//! 診断専用・本番コード不変）。すべて Metal 実機（Apple Silicon）依存の `#[ignore]` テストで、
//! CI では実行しない。型検査だけは `make check-cross-metal-tests` が担う。
//!
//! # 位置づけ
//!
//! - 規則の正（計測前に固定）: `docs/perf/logs/metal-m64-thin-tile-ab-macmini-3209/RULE.txt`
//! - 結論: `docs/perf/metal-gemm-m64-thin-tile-candidates.md` §7
//! - 雛形: `gemm_tile_select_macmini_diag_tests.rs`（#3181。到達追跡・hash・CPU 参照 parity・
//!   巡回計測）。`gemm_reuse_phase_diag_tests` の `pub(crate)` 面（`median_of`・
//!   `WARMUP_TRIALS`・`MEASURED_TRIALS`）へ到達するため `lib.rs` のクレートルート兄弟として置く。
//! - 呼び出し元: `docs/perf/logs/metal-m64-thin-tile-ab-macmini-3209/orchestrate_macmini.sh`。
//!   出力行は同ディレクトリの `aggregate.py` の正規表現と 1 対 1 で対応する
//!   （変更時は両方を更新する）。
//!
//! # 計測の組み立て
//!
//! 純カーネル時間は encode だけを記録して `synchronize_with_gpu_timestamps` で取る。
//! NN は `diag_encode_tiled_nn`、実向き strided（NT／TN）は #3209 で新設した
//! `diag_encode_strided_tiled`、`prod`（record_only）は本番スカラー経路
//! `encode_strided_bias_act_prepared`（ラベル `gemm_bias_act_strided`）を使う。
//! `dispatch_strided_tiled_prepared` は `dispatch_sync` で即 synchronize するため使えない。
//!
//! # テスト構成
//!
//! - 記録のみ: `m64_thin_reach_trace`（到達追跡）・`m64_thin_candidates_kernel_gpu`（計測）
//! - assert あり（前提ゲート）: `m64_thin_candidates_match_cpu_reference`
//!   （REQ-2 `assert_parity`。tolerance は不変）
//!
//! # 前提
//!
//! `--test-threads=1` で実行する（`cached_context()` はプロセスワイド singleton）。
//! 入力は決定的シードの有限値のみ（NaN・inf を含めない）。`expect` は診断テストだけで使う。

use std::time::Instant;

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::device::{probe_gpu_core_count, probe_soc_brand_string};
use crate::gemm::MetalGemm;
use crate::gemm_reuse_phase_diag_tests::{MEASURED_TRIALS, WARMUP_TRIALS, median_of};
use crate::layout::{MatrixLayout, classify_2d};
use crate::split_k_runtime::split_k_enabled;
use crate::tile::{self, GemmRoute, M64_THIN_CANDIDATES, TileConfig};

/// 事前登録の NN 6 形状（RULE.txt 項 3(a)。MLP 784→256→10・batch 64）。
const NN_SHAPES: [(usize, usize, usize); 6] = [
    (64, 256, 784),
    (64, 10, 256),
    (784, 64, 256),
    (256, 64, 10),
    (784, 256, 64),
    (256, 10, 64),
];

/// 事前登録の実向き strided 2 形状（RULE.txt 項 3(b)。適格形状のみ）。
const STRIDED_CASES: [(usize, usize, usize, Orient); 2] =
    [(784, 64, 256, Orient::Nt), (784, 256, 64, Orient::Tn)];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Orient {
    Nn,
    Nt,
    Tn,
}

impl Orient {
    fn name(self) -> &'static str {
        match self {
            Orient::Nn => "NN",
            Orient::Nt => "NT",
            Orient::Tn => "TN",
        }
    }
}

/// 測る腕。`Cur` は ゲート OFF の本番 classic 構成、`Thin(i)` は `M64_THIN_CANDIDATES[i]`、
/// `Prod` は本番スカラー経路（record_only）。
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Cur,
    Thin(usize),
    Prod,
}

impl Arm {
    fn name(self) -> String {
        match self {
            Arm::Cur => "cur".to_string(),
            Arm::Thin(i) => format!("T{}", i + 1),
            Arm::Prod => "prod".to_string(),
        }
    }
}

fn all_arms() -> Vec<Arm> {
    let mut v = vec![Arm::Cur];
    v.extend((0..M64_THIN_CANDIDATES.len()).map(Arm::Thin));
    v.push(Arm::Prod);
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

/// 論理 rows×cols（行優先）を転置 view の物理配置へ並べ替える。
fn transpose_dense(logical: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = logical[r * cols + c];
        }
    }
    out
}

/// 向きごとの物理オペランドとレイアウト。`a_log`／`b_log` は論理行列（CPU 参照用）。
struct Operands {
    /// 実効次元（pad8。生の形状が 8 の倍数ならそのまま）。
    eff: (usize, usize, usize),
    a_log: Vec<f32>,
    b_log: Vec<f32>,
    a_phys: Vec<f32>,
    b_phys: Vec<f32>,
    a_layout: MatrixLayout,
    b_layout: MatrixLayout,
}

fn operands(orient: Orient, m: usize, n: usize, k: usize) -> Operands {
    // 本番 `dispatch_auto` は 8 の倍数へゼロ詰めした実効次元で選択・dispatch する
    // （`diag_encode_tiled_nn` は 8 非整列を `NotEightAligned` で拒否する）。そのため
    // 生の形状（n=10 等）は実効次元（pad8）へゼロ詰めして計測する。ゼロ詰め領域は
    // FMA で累積値を変えないため、CPU 参照との比較も実効次元で行える。
    let (me, ne, ke) = (pad8(m), pad8(n), pad8(k));
    let a_raw = fill_finite(0x3209_a000 ^ ((m * 31 + k) as u64), m * k);
    let b_raw = fill_finite(0x3209_b000 ^ ((n * 37 + k) as u64), k * n);
    let a_log = zero_pad(&a_raw, m, k, me, ke);
    let b_log = zero_pad(&b_raw, k, n, ke, ne);
    let (m, n, k) = (me, ne, ke);
    let (a_phys, a_layout) = if orient == Orient::Tn {
        (
            transpose_dense(&a_log, m, k),
            classify_2d(&[m, k], &[1, m as isize]).expect("A 転置 view"),
        )
    } else {
        (
            a_log.clone(),
            classify_2d(&[m, k], &[k as isize, 1]).expect("A view"),
        )
    };
    let (b_phys, b_layout) = if orient == Orient::Nt {
        (
            transpose_dense(&b_log, k, n),
            classify_2d(&[k, n], &[1, k as isize]).expect("B 転置 view"),
        )
    } else {
        (
            b_log.clone(),
            classify_2d(&[k, n], &[n as isize, 1]).expect("B view"),
        )
    };
    Operands {
        eff: (m, n, k),
        a_log,
        b_log,
        a_phys,
        b_phys,
        a_layout,
        b_layout,
    }
}

fn pad8(x: usize) -> usize {
    x.div_ceil(8) * 8
}

/// rows×cols（行優先）を rows_p×cols_p へゼロ詰めする。
fn zero_pad(src: &[f32], rows: usize, cols: usize, rows_p: usize, cols_p: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows_p * cols_p];
    for r in 0..rows {
        out[r * cols_p..r * cols_p + cols].copy_from_slice(&src[r * cols..(r + 1) * cols]);
    }
    out
}

/// `prod` 腕の bias・act（RULE.txt 項 4。L1 forward は bias+relu、L2 forward は bias、
/// ほかはなし）。
fn prod_epilogue(m: usize, n: usize, k: usize) -> (bool, bool) {
    match (m, n, k) {
        (64, 256, 784) => (true, true),
        (64, 10, 256) => (true, false),
        _ => (false, false),
    }
}

/// 腕が要求するタイル構成。`Prod` は構成を持たない。
fn arm_cfg(arm: Arm, ctx: &MetalContext, m: usize, n: usize, k: usize) -> Option<TileConfig> {
    match arm {
        Arm::Cur => Some(tile::select_for_verified_device(
            m,
            n,
            k,
            ctx.verified_tile_table_device(),
        )),
        Arm::Thin(i) => Some(M64_THIN_CANDIDATES[i]),
        Arm::Prod => None,
    }
}

/// f32 出力の f64 逐次和（参考値。判定には使わない）。
fn checksum_f64(values: &[f32]) -> f64 {
    values.iter().fold(0.0f64, |acc, &v| acc + v as f64)
}

/// f32 出力の全要素ビット列に対する FNV-1a 64bit（#3181 と同型）。
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

/// `cfg` を `"<M64_THIN 番号 T1..|cur系|none>:<構成>"`（空白なし）で表す。
fn label_of(cfg: TileConfig) -> String {
    let tag = M64_THIN_CANDIDATES
        .iter()
        .position(|c| *c == cfg)
        .map(|i| format!("T{}", i + 1))
        .or_else(|| {
            tile::CANDIDATES
                .iter()
                .position(|c| *c == cfg)
                .map(|i| format!("C{i}"))
        })
        .unwrap_or_else(|| "none".to_string());
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

/// 1 trial の区間計測値（秒）。
struct Sample {
    kernel_gpu: f64,
    encode: f64,
    commit_wait: f64,
    alloc_c: f64,
    resolved: Option<TileConfig>,
    output: Option<Vec<f32>>,
}

/// 腕 1 回ぶんを encode-only で記録し、commit + GPU 完了待ちで `kernel_gpu` を取る。
/// 1 バッチ・1 ラベル・`kernel_gpu <= commit_wait` を assert する（RULE.txt 項 7）。
#[allow(clippy::too_many_arguments)]
fn trial(
    ctx: &MetalContext,
    gemm: &MetalGemm,
    arm: Arm,
    orient: Orient,
    ops_bufs: (&MetalBuffer, &MetalBuffer, &MetalBuffer),
    layouts: (MatrixLayout, MatrixLayout),
    dims: (usize, usize, usize),
    capture: bool,
) -> Sample {
    let (a_buf, b_buf, bias_buf) = ops_bufs;
    let (a_layout, b_layout) = layouts;
    let (m, n, k) = dims;
    let t = Instant::now();
    let c_buf = MetalBuffer::alloc_uninit_pooled(ctx, m * n).expect("C 確保に失敗した");
    let alloc_c = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let (resolved, expected_label) = match arm {
        Arm::Prod => {
            let (use_bias, relu) = prod_epilogue(m, n, k);
            let bias = if use_bias { Some((bias_buf, 0)) } else { None };
            gemm.encode_strided_bias_act_prepared(
                ctx, a_buf, 0, a_layout, b_buf, 0, b_layout, bias, relu, &c_buf, m, n, k,
            )
            .expect("encode_strided_bias_act_prepared に失敗した");
            (None, "gemm_bias_act_strided")
        }
        Arm::Cur | Arm::Thin(_) => {
            let cfg = arm_cfg(arm, ctx, m, n, k).expect("タイル腕は構成を持つ");
            if orient == Orient::Nn {
                let r = gemm
                    .diag_encode_tiled_nn(ctx, a_buf, b_buf, &c_buf, m, n, k, cfg)
                    .expect("diag_encode_tiled_nn に失敗した");
                (Some(r), "diag_encode_tiled_nn")
            } else {
                let r = gemm
                    .diag_encode_strided_tiled(
                        ctx, a_buf, 0, a_layout, b_buf, 0, b_layout, &c_buf, m, n, k, cfg,
                    )
                    .expect("diag_encode_strided_tiled に失敗した");
                (Some(r), "diag_encode_strided_tiled")
            }
        }
    };
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
    assert_eq!(batches[0].labels(), [expected_label], "想定外のラベル");
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

/// (向き, 形状) の測定セル一覧（NN 6 形状 + strided 2 形状）。
fn cells() -> Vec<(Orient, usize, usize, usize)> {
    let mut v: Vec<_> = NN_SHAPES
        .iter()
        .map(|&(m, n, k)| (Orient::Nn, m, n, k))
        .collect();
    v.extend(STRIDED_CASES.iter().map(|&(m, n, k, o)| (o, m, n, k)));
    v
}

/// 到達追跡（記録のみ）。机上の前提「ゲート ON で変わるのは (64,256,784)→T1 と
/// (784,64,256)→T3 の 2 件」を決定的に残す（RULE.txt 項 5）。
///
/// 出力行（`aggregate.py` と一致させる）:
/// - `DEVICE gpu_cores=<u32|none> soc="<s|none>" verified_tile_table=<s> split_k_enabled=<bool>`
/// - `TRACE shape=<m>x<n>x<k> eff=<m>x<n>x<k> should_split_k=<none|plan> route_off=<r> thin_on=<l|none> cur=<l> resolved_cur=<l>`
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn m64_thin_reach_trace() {
    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("MetalGemm::new に失敗した");
    let device = ctx.verified_tile_table_device();
    println!(
        "DEVICE gpu_cores={} soc=\"{}\" verified_tile_table={} split_k_enabled={}",
        probe_gpu_core_count().map_or_else(|| "none".to_string(), |v| v.to_string()),
        probe_soc_brand_string().unwrap_or_else(|| "none".to_string()),
        format!("{device:?}").replace(' ', ""),
        split_k_enabled()
    );
    for &(m, n, k) in &NN_SHAPES {
        let shape = format!("{m}x{n}x{k}");
        let (m, n, k) = (pad8(m), pad8(n), pad8(k));
        let split = tile::should_split_k(m, n, k)
            .map_or_else(|| "none".to_string(), |p| format!("{p:?}").replace(' ', ""));
        let route = tile::select_route_for_verified_device(m, n, k, device);
        let thin =
            tile::select_m64_thin_with(m, n, k, true).map_or_else(|| "none".to_string(), label_of);
        let cur = tile::select_for_verified_device(m, n, k, device);
        let resolved = gemm
            .resolve_tile_config(&ctx, cur)
            .expect("構成の解決に失敗した");
        println!(
            "TRACE shape={shape} eff={m}x{n}x{k} should_split_k={split} route_off={} thin_on={thin} cur={} resolved_cur={}",
            route_label(&route),
            label_of(cur),
            label_of(resolved)
        );
    }
}

/// 候補と `cur` が CPU 参照（FMA 契約）と REQ-2 複合判定で一致すること（前提ゲート。
/// tolerance は不変）。fallback する腕は記録してスキップする（abort しない）。
///
/// 出力行: `PARITY orient=<o> shape=<m>x<n>x<k> eff=<m>x<n>x<k> arm=<a> fallback=<bool> ok=<bool|skipped>`
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn m64_thin_candidates_match_cpu_reference() {
    use fandhe_ai_backend_cpu::parity::{assert_parity, matmul_reference_fma};

    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("MetalGemm::new に失敗した");
    for (orient, m, n, k) in cells() {
        let shape = format!("{m}x{n}x{k}");
        let ops = operands(orient, m, n, k);
        let (m, n, k) = ops.eff;
        let mut expected = vec![0.0f32; m * n];
        matmul_reference_fma(&ops.a_log, &ops.b_log, &mut expected, m, n, k)
            .expect("CPU 参照の形状検証に失敗した");
        let a_buf = MetalBuffer::new_with_data(&ctx, &ops.a_phys).expect("A アップロード");
        let b_buf = MetalBuffer::new_with_data(&ctx, &ops.b_phys).expect("B アップロード");
        let bias = MetalBuffer::new_zeroed(&ctx, n).expect("bias 確保");
        for arm in all_arms().into_iter().filter(|a| *a != Arm::Prod) {
            let cfg = arm_cfg(arm, &ctx, m, n, k).expect("構成あり");
            let s = trial(
                &ctx,
                &gemm,
                arm,
                orient,
                (&a_buf, &b_buf, &bias),
                (ops.a_layout, ops.b_layout),
                (m, n, k),
                true,
            );
            // 実際に実行した構成が要求と違う腕（fallback）は判定から除く（abort しない）。
            if s.resolved != Some(cfg) {
                println!(
                    "PARITY orient={} shape={shape} eff={m}x{n}x{k} arm={} fallback=true ok=skipped",
                    orient.name(),
                    arm.name()
                );
                continue;
            }
            let out = s.output.expect("capture=true");
            assert_parity(
                &format!("m64-thin {} {m}x{n}x{k} {}", orient.name(), arm.name()),
                &out,
                &expected,
            );
            println!(
                "PARITY orient={} shape={shape} eff={m}x{n}x{k} arm={} fallback=false ok=true",
                orient.name(),
                arm.name()
            );
        }
    }
}

/// 全腕を同一セルで巡回順に計測する（記録のみ。fallback は abort せず記録する）。
///
/// 出力行（`aggregate.py` と一致させる）:
/// - `CAND orient=<o> shape=<m>x<n>x<k> eff=<m>x<n>x<k> arm=<a> role=<judge|record_only> requested=<l|prod> resolved=<l|prod> fallback=<bool> checksum=<f> hash=<16hex> bit_identical_vs_cur=<bool|na>`
/// - `CAND orient=<o> shape=<m>x<n>x<k> eff=<m>x<n>x<k> arm=<a> kernel_gpu_median_ms=<v> q1=<v> q3=<v> encode_median_ms=<v> commit_wait_median_ms=<v> alloc_c_median_ms=<v>`
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn m64_thin_candidates_kernel_gpu() {
    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("MetalGemm::new に失敗した");
    let arms = all_arms();
    let num = arms.len();

    for (orient, m, n, k) in cells() {
        let shape = format!("{m}x{n}x{k}");
        let ops = operands(orient, m, n, k);
        let (m, n, k) = ops.eff;
        let a_buf = MetalBuffer::new_with_data(&ctx, &ops.a_phys).expect("A アップロード");
        let b_buf = MetalBuffer::new_with_data(&ctx, &ops.b_phys).expect("B アップロード");
        let bias_host = fill_finite(0x3209_c000 ^ (n as u64), n);
        let bias = MetalBuffer::new_with_data(&ctx, &bias_host).expect("bias アップロード");
        let run = |arm: Arm, capture: bool| {
            trial(
                &ctx,
                &gemm,
                arm,
                orient,
                (&a_buf, &b_buf, &bias),
                (ops.a_layout, ops.b_layout),
                (m, n, k),
                capture,
            )
        };

        // 解決構成（計測前に全腕を記録する。要求と解決が違う腕は fallback=true）。
        let mut fallback = vec![false; num];
        let mut cur_hash: Option<u64> = None;
        for (i, arm) in arms.iter().enumerate() {
            let s = run(*arm, true);
            let out = s.output.as_ref().expect("capture=true");
            let hash = bits_hash_fnv1a64(out);
            if *arm == Arm::Cur {
                cur_hash = Some(hash);
            }
            let (req, res, role) = match arm_cfg(*arm, &ctx, m, n, k) {
                Some(cfg) => {
                    let resolved = s.resolved.expect("タイル腕は解決構成を返す");
                    fallback[i] = resolved != cfg;
                    (label_of(cfg), label_of(resolved), "judge")
                }
                None => ("prod".to_string(), "prod".to_string(), "record_only"),
            };
            // `cur` は先頭の腕なので、後続の腕は cur_hash と比較できる。
            let bit = match (arm, cur_hash) {
                (Arm::Prod, _) | (_, None) => "na".to_string(),
                (_, Some(h)) => (h == hash).to_string(),
            };
            println!(
                "CAND orient={} shape={shape} eff={m}x{n}x{k} arm={} role={role} requested={req} resolved={res} fallback={} checksum={:.6} hash={hash:016x} bit_identical_vs_cur={bit}",
                orient.name(),
                arm.name(),
                fallback[i],
                checksum_f64(out),
            );
        }

        for _ in 0..WARMUP_TRIALS {
            for arm in &arms {
                let _ = run(*arm, false);
            }
        }
        let mk = || -> Vec<Vec<f64>> {
            (0..num)
                .map(|_| Vec::with_capacity(MEASURED_TRIALS))
                .collect()
        };
        let (mut kg, mut enc, mut cw, mut al) = (mk(), mk(), mk(), mk());
        for trial_i in 0..MEASURED_TRIALS {
            let offset = trial_i % num;
            for step in 0..num {
                let idx = (offset + step) % num;
                let s = run(arms[idx], false);
                kg[idx].push(s.kernel_gpu);
                enc[idx].push(s.encode);
                cw[idx].push(s.commit_wait);
                al[idx].push(s.alloc_c);
            }
        }
        for (idx, arm) in arms.iter().enumerate() {
            let q = median_of(&kg[idx]);
            println!(
                "CAND orient={} shape={shape} eff={m}x{n}x{k} arm={} kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4} encode_median_ms={:.4} commit_wait_median_ms={:.4} alloc_c_median_ms={:.4}",
                orient.name(),
                arm.name(),
                q.median * 1e3,
                q.q1 * 1e3,
                q.q3 * 1e3,
                median_of(&enc[idx]).median * 1e3,
                median_of(&cw[idx]).median * 1e3,
                median_of(&al[idx]).median * 1e3,
            );
        }
    }
}
