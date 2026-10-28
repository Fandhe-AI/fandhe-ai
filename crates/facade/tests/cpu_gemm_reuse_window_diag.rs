//! イシュー #3671（`docs/perf/cpu-gemm-reuse-fresh-inversion.md` §10）: GB10 の CPU GEMM で
//! N=256〜2048 の bench reuse 窓が #3646 診断 `resident_public` より遅い原因の切り分け診断。
//!
//! 背景: #3646 の専有正式系列で N=4096 の逆転は `download` に帰属できたが、N=256〜2048 は
//! ブリッジ（bench の `--phases` matmul 区間 vs 診断）が 1.63〜2.04 倍で fail した。コード読解では
//! 窓の中身は同じで、差は窓の外（bench は反復ごとに `verify_strict` で参照 N² と出力 N² を全要素比較）と
//! プロセス構成（常駐データ・参照 GEMM の前処理・腕の交互実行）にある。本ファイルは
//! #3646 の `resident_public`（窓の中は不変）へ bench 由来の要因を 1 つずつ「足す」腕を置く。
//!
//! - `rp_verify_touch`: 窓の外で N² の参照 Vec と出力を全要素比較（`verify_strict` のメモリ
//!   アクセスパターンの写し。bench-common へは path 依存できないため**アクセスパターンだけの複製**で、
//!   parity 契約ではない）
//! - `rp_verify_gap`: 同じ比較回数を L1 に収まる 4 KiB バッファ上で行う（メモリトラフィックなしの空き時間）
//! - `rp_hold_footprint`: bench と同じ 5 本の N² Vec を保持するだけ（触らない）
//! - `rp_ref_compute`: ブロック開始前に単一スレッド O(N³) の参照 GEMM を 1 回計算して保持する
//! - `rp_solo`（別テスト）: `GEMM_WINDOW_ONLY_ARM=resident_public` で 1 プロセス 1 腕 1 ラウンドを回す
//!
//! 窓の外の `Tape::reset` はコード読解で bench・診断とも呼ばないため腕を作らない
//! （bench `measure_gemm_reuse_resident` は `release_gemm_reuse_iteration` を呼ばず、
//! 診断の `one_iter` も呼ばない。テスト 2 で葉数が `matmul_resident` 前後で不変であることを確認する）。
//!
//! - テスト 1・2（CI 実行）: 全腕の出力が公開経路と bit 一致することと、tape の葉数不変を assert する
//! - テスト 3・4（`#[ignore]`・実機用 record-only）: 判定規則は
//!   `docs/perf/logs/cpu-gemm-reuse-window-3671/RULE.txt`（実測前に固定）に従い、
//!   `orchestrate.sh`／`aggregate.py` が独立 5 プロセスの出力を集計する
//!
//! 本番コード・tolerance・baseline・bench の窓は変更しない。新しい `unsafe` は使わない。

use std::hint::black_box;
use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tape, Tensor, tape_for};

/// bench-common `SEED_A`／`SEED_B` と同値（出典: framework-compare/bench-common/src/lib.rs）。
const SEED_A: u64 = 0xA11CE;
const SEED_B: u64 = 0xB0B;
const ROUNDS: usize = 4;
const SIZES: [usize; 5] = [256, 512, 1024, 2048, 4096];
/// 標準出力の 1 行 1 区間 JSON の接頭辞（`orchestrate.sh` が抽出）。
const PREFIX: &str = "DIAG_JSON ";
/// `rp_verify_gap` の L1 常駐バッファ長（f32 要素数。4 KiB 相当）。
const GAP_BUF_LEN: usize = 1024;
/// `rp_solo` で許可する腕名（allowlist。許可外は panic で拒否する）。
const SOLO_ALLOWED: [&str; 1] = ["resident_public"];

/// N ごとの (warmup, 計測) 反復数。N=4096 は 1 反復が長いので RULE.txt で別に固定している。
fn iters_for(n: usize) -> (usize, usize) {
    if n >= 4096 { (5, 10) } else { (20, 20) }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Fresh,
    ResidentPublic,
    VerifyTouch,
    VerifyGap,
    HoldFootprint,
    RefCompute,
}

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    kind: Kind,
}

const ARMS: [Arm; 6] = [
    Arm {
        name: "fresh",
        kind: Kind::Fresh,
    },
    Arm {
        name: "resident_public",
        kind: Kind::ResidentPublic,
    },
    Arm {
        name: "rp_verify_touch",
        kind: Kind::VerifyTouch,
    },
    Arm {
        name: "rp_verify_gap",
        kind: Kind::VerifyGap,
    },
    Arm {
        name: "rp_hold_footprint",
        kind: Kind::HoldFootprint,
    },
    Arm {
        name: "rp_ref_compute",
        kind: Kind::RefCompute,
    },
];

fn inputs(n: usize) -> (Tensor<f32>, Tensor<f32>) {
    let a = Xorshift64Star::new(SEED_A).fill_vec(n * n);
    let b = Xorshift64Star::new(SEED_B).fill_vec(n * n);
    (
        Tensor::new(a, &[n, n]).expect("形状とデータ長は一致させている"),
        Tensor::new(b, &[n, n]).expect("形状とデータ長は一致させている"),
    )
}

fn checksum(v: &[f32]) -> f64 {
    v.iter().map(|&x| f64::from(x)).sum()
}

/// (arm, n, phase) のセル識別子。
type CellKey = (&'static str, usize, &'static str);

/// (arm, n, phase) ごとのサンプル蓄積器。
#[derive(Default)]
struct Rec {
    cells: Vec<(CellKey, Vec<f64>)>,
    /// 記録するか（warmup 中は false）。
    on: bool,
    arm: &'static str,
    n: usize,
    /// (arm, n) → 全計測反復で共通だった checksum の bit。
    checksums: Vec<((&'static str, usize), u64)>,
}

impl Rec {
    fn push(&mut self, phase: &'static str, dt: f64) {
        if !self.on {
            return;
        }
        let key = (self.arm, self.n, phase);
        let i = match self.cells.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                self.cells.push((key, Vec::new()));
                self.cells.len() - 1
            }
        };
        self.cells[i].1.push(dt);
    }

    fn timed<T>(&mut self, phase: &'static str, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        let dt = t.elapsed().as_secs_f64();
        self.push(phase, dt);
        out
    }

    /// 同一 (arm, n) の全計測反復で bit 一致を要求し、不一致なら panic する。
    fn record_checksum(&mut self, s: f64) {
        if !self.on {
            return;
        }
        let bits = s.to_bits();
        let key = (self.arm, self.n);
        match self.checksums.iter().find(|(k, _)| *k == key) {
            None => self.checksums.push((key, bits)),
            Some((_, prev)) => assert_eq!(
                *prev, bits,
                "arm {} n={}: 反復間で checksum が不一致",
                key.0, key.1
            ),
        }
    }
}

/// 常駐腕の共有状態。upload・参照値の確保はすべてブロック開始時（計測窓の外）。
struct Resident {
    tape: Tape,
    /// 束縛用 store。`ResidentTensor` の寿命中は保持する（#3646 診断と同じ）。
    _store: fandhe_ai::DeviceParamStore,
    ra: fandhe_ai::ResidentTensor,
    rb: fandhe_ai::ResidentTensor,
    /// bench の常駐データ相当（`a_host`・`b_host`・`GemmReference.c` 等）を保持する Vec 群。
    held: Vec<Vec<f32>>,
    /// `rp_verify_touch` が比較する参照出力（init で 1 回計算した値の写し）。
    reference: Vec<f32>,
    /// `rp_verify_gap` 用の L1 常駐バッファ対。
    gap_a: Vec<f32>,
    gap_b: Vec<f32>,
}

/// bench `GemmReference::compute` の写し: `available_parallelism()` 個（`n` で頭打ち）の scoped worker が
/// 行を分担する ikj 順の素朴な O(N³) GEMM。前処理で bench と同じスレッドスタックの確保・解放と
/// CPU 履歴を再現するためだけの計算で、数値契約には使わない（H-A2）。
fn naive_reference(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    let workers = std::thread::available_parallelism()
        .map(|w| w.get())
        .unwrap_or(1)
        .min(n)
        .max(1);
    let rows_per_chunk = n.div_ceil(workers);
    std::thread::scope(|scope| {
        for (chunk_idx, c_chunk) in c.chunks_mut(rows_per_chunk * n).enumerate() {
            let row_start = chunk_idx * rows_per_chunk;
            scope.spawn(move || {
                for local_i in 0..c_chunk.len() / n {
                    let i = row_start + local_i;
                    for k in 0..n {
                        let aik = a[i * n + k];
                        let row = &b[k * n..(k + 1) * n];
                        let out = &mut c_chunk[local_i * n..(local_i + 1) * n];
                        for (o, &bv) in out.iter_mut().zip(row) {
                            *o = aik.mul_add(bv, *o);
                        }
                    }
                }
            });
        }
    });
    c
}

impl Resident {
    fn new(kind: Kind, a: &Tensor<f32>, b: &Tensor<f32>, n: usize) -> Self {
        let tape = tape_for(Device::Cpu).expect("CPU tape");
        // bench `gemm_binding_store` と同じ束縛用 store（`matmul_resident` は世代を見ない）。
        let store = Sequential::new()
            .add_linear(1, 1, 1)
            .expect("linear")
            .init_device_param_store(&tape)
            .expect("store");
        let ra = tape.upload_resident(&store, a).expect("upload a");
        let rb = tape.upload_resident(&store, b).expect("upload b");
        let a_s = a.as_slice().expect("contiguous");
        let b_s = b.as_slice().expect("contiguous");
        let mut held = Vec::new();
        let mut reference = Vec::new();
        match kind {
            Kind::VerifyTouch => {
                let out = tape.matmul_resident(&ra, &rb).expect("matmul_resident");
                reference = out.contiguous().as_slice().expect("contiguous").to_vec();
            }
            Kind::HoldFootprint => {
                // a_data・b_data・a_host・b_host・参照 c の 5 本（bench と同数）。内容は入力の写しで
                // 触れずに保持するだけ（確保時の書き込みのみ）。
                for _ in 0..5 {
                    held.push(vec![1.0f32; n * n]);
                }
            }
            Kind::RefCompute => {
                held.push(a_s.to_vec());
                held.push(b_s.to_vec());
                held.push(a_s.to_vec());
                held.push(b_s.to_vec());
                held.push(naive_reference(a_s, b_s, n));
            }
            _ => {}
        }
        let gap_a: Vec<f32> = (0..GAP_BUF_LEN).map(|i| i as f32).collect();
        let gap_b = gap_a.clone();
        Self {
            tape,
            _store: store,
            ra,
            rb,
            held,
            reference,
            gap_a,
            gap_b,
        }
    }
}

/// 参照と出力の全要素 bit 比較（`verify_strict` のアクセスパターンの写し）。不一致数を返す。
fn verify_touch(reference: &[f32], out: &[f32]) -> usize {
    reference
        .iter()
        .zip(out)
        .filter(|(r, o)| r.to_bits() != o.to_bits())
        .count()
}

/// `verify_touch` と同じ比較回数 `N²` を L1 常駐バッファ上で行う。メモリトラフィックなしの空き時間。
fn verify_gap(a: &[f32], b: &[f32], total: usize) -> usize {
    let mut mism = 0usize;
    let mut left = total;
    while left > 0 {
        let take = left.min(a.len());
        mism += verify_touch(&a[..take], &b[..take]);
        left -= take;
    }
    mism
}

fn new_fresh_tape() -> Tape {
    tape_for(Device::Cpu).expect("CPU tape")
}

/// 1 反復。窓は `iter_total`（窓の中は全腕で `resident_public` と同一）。
fn one_iter(
    rec: &mut Rec,
    arm: Arm,
    res: &Resident,
    (a, b): (&Tensor<f32>, &Tensor<f32>),
    n: usize,
    capture: Option<&mut Vec<f32>>,
) {
    let mut cap = capture;
    let mut take = |s: &[f32]| {
        if let Some(c) = cap.as_deref_mut() {
            *c = s.to_vec();
        }
    };
    if arm.kind == Kind::Fresh {
        let t_all = Instant::now();
        let tape = rec.timed("tape_build", new_fresh_tape);
        let (va, vb) = rec.timed("var_upload", || (tape.var(a), tape.var(b)));
        let c = rec.timed("matmul", || va.matmul(&vb).expect("matmul"));
        let view = rec.timed("readout", || c.host_view());
        let s = rec.timed("checksum", || checksum(&view));
        take(&view);
        rec.push("iter_total", t_all.elapsed().as_secs_f64());
        rec.record_checksum(s);
        drop(view);
        // Tape の drop は窓の外（bench の `elapsed` 取得後と同じ位置）。
        rec.timed("tape_drop", move || drop(tape));
        return;
    }
    let t_all = Instant::now();
    let c = rec.timed("matmul", || {
        res.tape
            .matmul_resident(&res.ra, &res.rb)
            .expect("matmul_resident")
    });
    let out = rec.timed("readout", || c.contiguous());
    let sl = out.as_slice().expect("contiguous");
    let s = rec.timed("checksum", || checksum(sl));
    take(sl);
    rec.push("iter_total", t_all.elapsed().as_secs_f64());
    rec.record_checksum(s);
    // 窓の外: 腕ごとの「足す要因」。区間 `gap` で計時し、両 verify 腕の空き時間が揃うことを記録する。
    match arm.kind {
        Kind::VerifyTouch => {
            let m = rec.timed("gap", || verify_touch(&res.reference, sl));
            assert_eq!(
                black_box(m),
                0,
                "参照と出力が不一致（検算の写しが壊れている）"
            );
        }
        Kind::VerifyGap => {
            let m = rec.timed("gap", || verify_gap(&res.gap_a, &res.gap_b, n * n));
            assert_eq!(black_box(m), 0);
        }
        _ => {}
    }
}

/// 1 ブロック（同一 arm・同一 n の warmup + 計測）。入力の upload・前処理は窓の外。
fn drive(
    rec: &mut Rec,
    arm: Arm,
    n: usize,
    ab: (&Tensor<f32>, &Tensor<f32>),
    (warmup, iters): (usize, usize),
    mut captured: Option<&mut Vec<Vec<f32>>>,
) {
    rec.arm = arm.name;
    rec.n = n;
    let res = Resident::new(arm.kind, ab.0, ab.1, n);
    for i in 0..warmup + iters {
        rec.on = i >= warmup;
        let mut out = Vec::new();
        let want = rec.on && captured.is_some();
        one_iter(rec, arm, &res, ab, n, want.then_some(&mut out));
        if want
            && !out.is_empty()
            && let Some(c) = captured.as_deref_mut()
        {
            c.push(out);
        }
    }
    rec.on = false;
    black_box(&res.held);
}

/// 公開経路（`upload_resident → matmul_resident`）の出力。
fn public_output(a: &Tensor<f32>, b: &Tensor<f32>, n: usize) -> Vec<f32> {
    let res = Resident::new(Kind::ResidentPublic, a, b, n);
    let out = res
        .tape
        .matmul_resident(&res.ra, &res.rb)
        .expect("matmul_resident");
    out.contiguous().as_slice().expect("contiguous").to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn cpu_gemm_window_arms_match_public_path_bit_exact() {
    for n in [64usize, 128] {
        let (a, b) = inputs(n);
        let expect = bits(&public_output(&a, &b, n));
        for arm in ARMS {
            let mut rec = Rec::default();
            let mut outs = Vec::new();
            drive(&mut rec, arm, n, (&a, &b), (0, 3), Some(&mut outs));
            assert_eq!(outs.len(), 3);
            for out in &outs {
                assert_eq!(
                    bits(out),
                    expect,
                    "arm {} n={n}: 出力が公開経路と bit 不一致",
                    arm.name
                );
            }
        }
    }
}

#[test]
fn matmul_resident_does_not_grow_tape_leaves() {
    // 「`Tape::reset` の有無」仮説の読解根拠を CI で補強する: 公開 API で数えられるのは葉数のみで、
    // 窓の前後で葉数が変わらない（resident 経路が葉を積まない）ことだけを確認する。
    // ノード数は公開 API で数えられないため、読解（reset 非呼び出し）のみを根拠とする。
    let (a, b) = inputs(64);
    let res = Resident::new(Kind::ResidentPublic, &a, &b, 64);
    let before = res.tape.leaf_count();
    for _ in 0..3 {
        res.tape
            .matmul_resident(&res.ra, &res.rb)
            .expect("matmul_resident");
    }
    assert_eq!(res.tape.leaf_count(), before);
}

fn emit(rec: &Rec, series: &str) {
    let threads = std::env::var("RAYON_NUM_THREADS").ok();
    for ((arm, n, phase), secs) in &rec.cells {
        let q = median_q1_q3(secs).expect("サンプルは非空・非 NaN");
        let mut sorted = secs.clone();
        sorted.sort_by(f64::total_cmp);
        let checksum_bits = (*phase == "iter_total")
            .then(|| {
                rec.checksums
                    .iter()
                    .find(|(k, _)| *k == (*arm, *n))
                    .map(|(_, b)| format!("{b:016x}"))
            })
            .flatten();
        let line = serde_json::json!({
            "issue": 3671,
            "series": series,
            "arm": arm,
            "n": n,
            "phase": phase,
            "median_s": q.median,
            "q1_s": q.q1,
            "q3_s": q.q3,
            "min_s": sorted.first(),
            "max_s": sorted.last(),
            "n_samples": secs.len(),
            "checksum_bits": checksum_bits,
            "rayon_num_threads_env": threads,
            "target_arch": std::env::consts::ARCH,
            "target_os": std::env::consts::OS,
        });
        println!("{PREFIX}{line}");
    }
}

/// 実機用 record-only 診断（`orchestrate.sh` が独立プロセスで 5 回起動）。
#[test]
#[ignore = "実機計測用の record-only 診断（RULE.txt・orchestrate.sh 参照）"]
fn cpu_gemm_reuse_window_phases() {
    let mut rec = Rec::default();
    for n in SIZES {
        let (a, b) = inputs(n);
        for round in 0..ROUNDS {
            // 順序効果を打ち消すためラウンドごとに arm 順を反転する。
            let order: Vec<Arm> = if round % 2 == 0 {
                ARMS.to_vec()
            } else {
                ARMS.iter().rev().copied().collect()
            };
            for arm in order {
                drive(&mut rec, arm, n, (&a, &b), iters_for(n), None);
            }
        }
    }
    emit(&rec, "diag");
}

/// `rp_solo` 系列: 1 プロセスで `resident_public` だけを bench と同じ反復構造
/// （init 1 回＋warmup＋計測。腕の交互実行なし・1 ラウンド）で回す。
/// `GEMM_WINDOW_ONLY_ARM` は allowlist で検証し、許可外・未設定は panic で拒否する。
#[test]
#[ignore = "実機計測用の record-only 診断（RULE.txt・orchestrate.sh --only-arm 参照）"]
fn cpu_gemm_reuse_window_solo() {
    let want = std::env::var("GEMM_WINDOW_ONLY_ARM").expect("GEMM_WINDOW_ONLY_ARM が未設定");
    assert!(
        SOLO_ALLOWED.contains(&want.as_str()),
        "GEMM_WINDOW_ONLY_ARM={want:?} は allowlist 外"
    );
    let arm = ARMS
        .iter()
        .copied()
        .find(|a| a.name == want)
        .expect("allowlist の腕は ARMS に存在する");
    let mut rec = Rec::default();
    for n in SIZES {
        let (a, b) = inputs(n);
        drive(&mut rec, arm, n, (&a, &b), iters_for(n), None);
    }
    // arm 名は "rp_solo" として出す（同一プロセス内の resident_public と区別する）。
    let mut solo = Rec::default();
    for ((_, n, phase), secs) in rec.cells {
        solo.cells.push((("rp_solo", n, phase), secs));
    }
    for ((_, n), bits) in rec.checksums {
        solo.checksums.push((("rp_solo", n), bits));
    }
    emit(&solo, "solo");
}
