//! fandhe-ai benchmark binary.
//!
//! Protocol: warmup 20 -> measure 20 (train: 100 SGD steps total, first 20
//! treated as warmup, stats over the remaining 80). Each measured iteration
//! builds a fresh tape. The measured region ends with host materialization
//! (`to_tensor()` + element readout) so asynchronous device execution cannot
//! leak out of the timing window.
//!
//! `--mode reuse`（イシュー #925。`gemm` タスクのみ）: 上記の毎回新規 tape
//! プロトコルは fandhe-ai の CUDA/Metal でタイル初期化コスト（CUDA コンテキ
//! スト作成・NVRTC カーネルコンパイル等）を毎計測に含めてしまい、デバイス・
//! グラフを使い回す candle / Burn との比較でフレームワーク間の不公平が生じる
//! （`results/summary.md` 環境 2 の備考）。reuse モードは tape を 1 回だけ
//! 構築し、その初期化コスト（init_s）を「カーネル実行時間」（中央値・Q1/Q3）
//! と分離して記録することで、初期化コストとカーネル実行を切り分けて比較可能
//! にする。
//!
//! `train --mode reuse`（イシュー #958）: gemm の reuse（イシュー #925）と
//! 同じ「初期化コストとカーネル実行の分離」の考えを学習ループへ適用する。
//! fresh の `run_train` は各 step でホスト経由 SGD（勾配を download →
//! ホストで `p - lr*g` → `apply_parameters` で書き戻し）を行っており、
//! candle（`Var::set`）や Burn（デバイス上更新）と非対称なプロトコルに
//! なっている（#957 背景）。reuse は #954 で追加されたデバイス常駐パラ
//! メータ更新 API（`fandhe_ai::DeviceParamStore`）を使い、`p - lr*g` の
//! 更新自体をデバイス上で完結させる。
//!
//! 参照実装は `crates/facade/tests/device_param_store_train.rs` の
//! `train_with_device_param_store`（`init_device_param_store` で 1 回だけ
//! 全パラメータを H2D upload → 以後は同一 `DeviceParamStore` を使い回す）
//! であり、本関数（`run_train_reuse`）はその構造に揃える。tape 自体は
//! （gemm reuse と異なり）**step ごとに新規生成**する: `fandhe_ai_autodiff::
//! Tape` はノード列クリア API を持たず学習ループはステップごとに tape を
//! 生成・破棄する設計契約（`crates/autodiff/src/tape.rs`）であり、単一
//! tape を 100 step 使い回すと `Tape::backward` の逆順走査コストが step
//! 数に比例して増加し 1 step の計測時間が非定常になる。reuse で使い回す
//! のは tape ではなく `DeviceParamStore`（デバイス常駐バッファ・デバイス
//! を固定する側）であり、fresh/reuse の計時差は「ホスト経由 SGD vs デバ
//! イス常駐更新」に限定される。
//!
//! 既知の前提（改善量の解釈範囲・codex-review PR #1104 P2 是正）: 0.5.0 の
//! `Sequential::forward_resident` が呼ぶのは `DeviceParamStore::
//! register_resident_params`（D2H を伴わない。#1059 で D2H を伴う旧
//! `register_resident_leaves` から分離。`crates/autodiff/src/optim/
//! device_store.rs` doc 参照）であり、forward 自体に毎 step の D2H は
//! 発生しない。各 step 中の唯一のホスト同期点は `loss_readout`
//! （`loss.to_tensor().get()`）であり、これが（1 step ずれた形で）前
//! step の backward・デバイス上 SGD 更新の完了を保証する（ストリーム
//! 順序保証。`docs/backend-cuda-async-execution-design.md` §3 I1/I2・
//! 本関数下部のループ冒頭コメント参照）。reuse が排除するのは「毎 step
//! のホスト経由 `p - lr*g` 計算 + 再アップロード（H2D）」であり、
//! パラメータの D2H（forward 用）は 0.5.0 では構造的に発生しない。
//!
//! `train --phases`（イシュー #1009）: `run_train`/`run_train_reuse` が
//! 1 step の合計時間しか記録しない点を補い、公開 API の呼び出し境界で
//! 区間分解した median/Q1/Q3 を `task:"train_phases"` の JSONL 行として
//! 出力する（元は `--task train` 限定だったが #1182 で `--task gemm
//! --mode reuse` にも拡張。`--task gemm --mode fresh`／`--task infer`
//! との組合せは引き続き MEASURE_ERROR）。区間は「公開 API のどの呼び出し
//! に時間が乗るか」を表し、GPU 内部（カーネル／転送）の内訳ではない
//! （`fandhe-ai` のホスト常駐 `Tensor<f32>` は CUDA/Metal で演算ごとに
//! H2D→カーネル→D2H を行うため）。詳細な区間定義・「同期待ち」を独立
//! 区間にできない理由は README「`train --phases`」節を参照。実装は
//! `measure_train_phases`/`measure_train_reuse_phases`（計測本体）と
//! `run_train_phases`/`run_train_reuse_phases`（JSONL emit）に分離する。
//!
//! `gemm --mode reuse --phases`（イシュー #1182）: #1142（`docs/perf/
//! cuda-gemm-candle-gate-remeasurement.md` §4.3・§8）が「reuse の計測
//! 境界に残る H2D／D2H／同期の固定費が candle 比を押し下げている」と
//! **推定**したまま未確定だった内容を、`train --phases` と同じ方法論で
//! 実測確定する。`gemm --mode reuse`（`run_gemm_reuse`）1 反復の内側
//! （`matmul` 区間）は `readout_var`（`to_tensor()` +
//! `contiguous().as_slice().to_vec()`）をここでは展開し、`matmul`／
//! `to_tensor`／`host_copy`／`checksum`／`iter_total` の 5 区間として
//! `task:"gemm_phases"` の JSONL 行に出力する（`run_gemm_reuse` 本体は
//! 変更しない）。`matmul` 区間の内側にホスト→デバイス転送・カーネル
//! 実行・デバイス→ホスト転送・ストリーム同期が全て閉じており、公開
//! API ではこれ以上分離できない。内訳（H2D／カーネル専有時間／D2H の
//! 実測分解）は `crates/backend-cuda` 側の診断テスト
//! （`gemm_reuse_phase_diag_tests`）が別途取り、突合結果を
//! `docs/perf/cuda-gemm-reuse-phase-breakdown.md` に記録する。詳細は
//! README「`gemm --mode reuse --phases`」節を参照。

use bench_common::*;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, SgdConfig, Tape, Tensor};
use std::time::{Duration, Instant};

const FRAMEWORK: &str = "fandhe-ai";
const VERSION: &str = "0.6.0";

const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const TRAIN_STEPS: usize = 100;
const TRAIN_WARMUP: usize = 20;
const LR: f32 = 0.01;

// `train --phases`（イシュー #1009）の区間名。JSONL の `phase` フィールド値
// になる（`PhaseRecord` 側で `[a-z0-9_]+` allowlist 検証される定数のみ渡す
// ため、定数自体もその制約を満たす）。README「train --phases」節の区間定義
// 表と対応する。
const PHASE_TAPE_BUILD: &str = "tape_build";
const PHASE_LEAF_REGISTER: &str = "leaf_register";
const PHASE_FORWARD: &str = "forward";
const PHASE_FORWARD_RESIDENT: &str = "forward_resident";
const PHASE_LOSS_READOUT: &str = "loss_readout";
const PHASE_BACKWARD: &str = "backward";
const PHASE_PARAM_READOUT: &str = "param_readout";
const PHASE_HOST_SGD: &str = "host_sgd";
const PHASE_APPLY_PARAMS: &str = "apply_params";
const PHASE_DEVICE_UPDATE: &str = "device_update";
const PHASE_TAPE_DROP: &str = "tape_drop";
const PHASE_STEP_TOTAL: &str = "step_total";

// `gemm --mode reuse --phases`（イシュー #1182）の区間定数。#1142
// §4.3 が「candle 比を押し下げている固定費は reuse 計測境界に残る
// H2D／D2H／同期」と推定した内容を、公開 API 呼び出し境界で分解して
// 実測確定するための計装。`matmul` 区間の内側に H2D（A/B のアップロード）
// ・カーネル実行・D2H（結果ダウンロード）・ストリーム同期が全て閉じて
// おり、fandhe-ai 0.6.0 の公開 API（`Var::matmul`）ではこれ以上分離
// できない（内訳は `crates/backend-cuda` の診断テストが別途取る。
// `docs/perf/cuda-gemm-reuse-phase-breakdown.md` 参照）。
const PHASE_GEMM_MATMUL: &str = "matmul";
const PHASE_GEMM_TO_TENSOR: &str = "to_tensor";
const PHASE_GEMM_HOST_COPY: &str = "host_copy";
const PHASE_GEMM_CHECKSUM: &str = "checksum";
const PHASE_GEMM_ITER_TOTAL: &str = "iter_total";

/// `train --phases` の 1 step 分の区間計測を保持する順序付きサンプル集合
/// （イシュー #1009）。phase の初出順が `phase_index`（README「train
/// --phases」節・summarize.py (b'') 節の表示順と一致させる）。
/// `measure_train_phases`/`measure_train_reuse_phases` は全 phase を
/// `TRAIN_STEPS` 回ずつ push する構造（ループ本体が phase を毎回同じ順序で
/// 通過する）ため、同一 phase の `durations()` は要素数 `TRAIN_STEPS`・
/// インデックス i が「同じ step」を指す前提が成り立つ
/// （`tests::train_phases_each_step_phase_sum_does_not_exceed_total` が
/// この前提を固定する）。
struct PhaseSamples {
    order: Vec<&'static str>,
    samples: std::collections::HashMap<&'static str, Vec<Duration>>,
}

impl PhaseSamples {
    fn new() -> Self {
        Self {
            order: Vec::new(),
            samples: std::collections::HashMap::new(),
        }
    }

    fn push(&mut self, phase: &'static str, dur: Duration) {
        match self.samples.get_mut(phase) {
            Some(v) => v.push(dur),
            None => {
                self.order.push(phase);
                self.samples.insert(phase, vec![dur]);
            }
        }
    }

    fn durations(&self, phase: &str) -> &[Duration] {
        self.samples.get(phase).map(Vec::as_slice).unwrap_or(&[])
    }
}

fn make_tape(device: &str) -> Result<Tape, Box<dyn std::error::Error>> {
    match device {
        "cpu" => Ok(fandhe_ai::tape()),
        // Device::Metal exists only on macOS (cfg-gated in fandhe-ai).
        #[cfg(target_os = "macos")]
        "metal" => fandhe_ai::tape_for(Device::Metal).map_err(|e| {
            format!("MEASURE_ERROR: fandhe-ai tape_for(Device::Metal) failed: {e}").into()
        }),
        #[cfg(not(target_os = "macos"))]
        "metal" => Err("MEASURE_ERROR: Device::Metal is macOS-only".into()),
        // fandhe-ai selects backends via cfg + runtime probing (cudarc dynamic
        // load), not cargo features: fail-fast with BackendError when absent.
        "cuda" => fandhe_ai::tape_for(Device::Cuda(0)).map_err(|e| {
            format!("MEASURE_ERROR: fandhe-ai tape_for(Device::Cuda(0)) failed: {e}").into()
        }),
        other => Err(format!("MEASURE_ERROR: unknown device '{other}'").into()),
    }
}

/// Host-materialize a Var result and return a checksum (forces sync).
fn checksum_var(v: &fandhe_ai::Var) -> Result<f64, Box<dyn std::error::Error>> {
    let t = v.to_tensor();
    let slice = t
        .contiguous()
        .as_slice()
        .ok_or("as_slice() returned None after contiguous()")?
        .to_vec();
    Ok(slice.iter().map(|&x| x as f64).sum())
}

/// Host-materialize a Var result and return the raw elements (forces sync).
/// `run_gemm`/`run_gemm_reuse`（イシュー #970）は checksum（全要素和）に
/// 加え要素単位の参照比較（`GemmReference::verify`）が必要なため、
/// `checksum_var` とは別に生の `Vec<f32>` を返す readout を用意する
/// （`checksum_var`/`checksum_tensor` は `run_infer` が引き続き使うため
/// シグネチャを変更しない）。
fn readout_var(v: &fandhe_ai::Var) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let t = v.to_tensor();
    Ok(t.contiguous()
        .as_slice()
        .ok_or("as_slice() returned None after contiguous()")?
        .to_vec())
}

fn checksum_tensor(t: &Tensor<f32>) -> Result<f64, Box<dyn std::error::Error>> {
    let slice = t
        .contiguous()
        .as_slice()
        .ok_or("as_slice() returned None after contiguous()")?
        .to_vec();
    Ok(slice.iter().map(|&x| x as f64).sum())
}

fn gemm_inputs(n: usize) -> Result<(Tensor<f32>, Tensor<f32>), Box<dyn std::error::Error>> {
    // イシュー #970 codex-review 指摘（PR #978・P1）: `n * n`（要素数）を
    // 入力ベクタ生成前に `gemm_element_count` で検証する（`--size` は
    // 未検証な CLI 入力であり、無検証の乗算は debug で panic・release では
    // wrap した長さで確保・使用してしまう）。`run_gemm`/`run_gemm_reuse`
    // 双方がこの関数を経由するため、両経路とも生成前検証が及ぶ。
    let len = gemm_element_count(n)?;
    let a = Xorshift64Star::new(SEED_A).fill_vec(len);
    let b = Xorshift64Star::new(SEED_B).fill_vec(len);
    Ok((Tensor::new(a, &[n, n])?, Tensor::new(b, &[n, n])?))
}

fn run_gemm(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let n = cli.size;
    let (a_data, b_data) = gemm_inputs(n)?;
    // 要素単位検証（イシュー #970）の参照値。本体 `backend-cpu::parity::
    // matmul_reference_fma` と同じ FMA 契約（f32 mul_add・逐次 k 昇順）の
    // 参照 GEMM を計測窓の外（warmup 前）で 1 回だけ計算する。
    let a_host = a_data
        .contiguous()
        .as_slice()
        .ok_or("a_data as_slice() returned None")?
        .to_vec();
    let b_host = b_data
        .contiguous()
        .as_slice()
        .ok_or("b_data as_slice() returned None")?
        .to_vec();
    let reference = GemmReference::compute(n, &a_host, &b_host)?;

    let mut checksum = 0.0;
    let mut parity: Option<ParityStats> = None;

    let one = |sync_checksum: &mut f64,
               parity: &mut Option<ParityStats>|
     -> Result<Duration, Box<dyn std::error::Error>> {
        // fresh tape per measurement (no accumulated graph)。計時開始は
        // tape 構築より前に置く（イシュー #925 レビュー指摘）。「fresh は
        // tape/デバイス初期化コスト（CUDA コンテキスト作成・NVRTC カーネル
        // コンパイル等）を毎計測に含む」という上記モジュールコメント・reuse
        // モードとの対比説明が実際の計測範囲と一致するようにするため。
        let start = Instant::now();
        let tape = make_tape(&cli.device)?;
        let a = tape.var(&a_data);
        let b = tape.var(&b_data);
        let c = a.matmul(&b)?;
        // sync: materialize result on host and read elements（従来どおり
        // 計測窓内。checksum の計算コストも従来と変えない）。
        let out = readout_var(&c)?;
        *sync_checksum = out.iter().map(|&x| x as f64).sum();
        let elapsed = start.elapsed();
        // イシュー #965 codex-review 指摘: sync_checksum は毎反復上書きされる
        // ため、ループ後に最後の値だけを検査すると途中反復の縮退を見逃す。
        // burn 側の縮退 checksum 遮断（bench-burn/src/main.rs）と対称に、
        // fandhe-ai 側でも将来同種の不具合が出た場合に壊れた計算の実行時間を
        // 性能値として記録しないよう、checksum 計算直後・warmup を含む
        // 全反復で検査する。
        validate_gemm_checksum(*sync_checksum)?;
        // イシュー #970: 要素単位の複合判定（O(n^2)）は計測窓の外で行う
        // （GEMM 自体は O(n^3) だが、比較コストが計測時間へ混入するのを
        // 避けるため elapsed 取得後に実行する）。反復間の worst-case を
        // 保持し、途中反復の破損（要素の入れ替わり等、checksum では
        // 見逃しうる破損）も見逃さない。
        let stats = reference.verify(&out)?;
        *parity = Some(match parity.take() {
            Some(prev) => prev.worst(stats),
            None => stats,
        });
        Ok(elapsed)
    };

    for _ in 0..WARMUP_ITERS {
        one(&mut checksum, &mut parity)?;
    }
    let mut durations = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        durations.push(one(&mut checksum, &mut parity)?);
    }
    let st = stats(&durations)?;
    Record {
        framework: FRAMEWORK,
        framework_version: VERSION,
        task: "gemm",
        device: &cli.device,
        size: n,
        stats: st,
        gflops: Some(gemm_gflops(n, st.median_s)),
        throughput_per_s: None,
        checksum,
        warmup: WARMUP_ITERS,
        iters: MEASURE_ITERS,
        mode: "fresh",
        init_s: None,
        parity,
        tf32: false,
    }
    .emit(&cli.out)?;
    Ok(())
}

/// `--mode reuse` の gemm 計測（イシュー #925）。tape/デバイスを 1 回だけ
/// 構築し、その構築 + 葉 Var 登録 + 初回 matmul + ホスト実体化までの経過を
/// init_s として分離記録したうえで、同一 tape 上で warmup 残り + 計測を回す。
/// 葉 Var（A・B）は 1 回だけ登録して使い回すが、matmul の結果ノードは呼ぶ
/// たびに tape へ蓄積される（N=2048 で約 16 MiB/回 × 40 回 ≒ 640 MiB。
/// N=4096 でも約 2.6 GiB で対象 GPU メモリ内に収まる。README 計測プロトコル
/// 節に明記）。
fn run_gemm_reuse(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let n = cli.size;
    let (a_data, b_data) = gemm_inputs(n)?;
    // イシュー #970: 参照 GEMM は init_s 計測（tape/デバイス初期化コスト）
    // を汚さないよう、init_start より前に計算する。
    let a_host = a_data
        .contiguous()
        .as_slice()
        .ok_or("a_data as_slice() returned None")?
        .to_vec();
    let b_host = b_data
        .contiguous()
        .as_slice()
        .ok_or("b_data as_slice() returned None")?
        .to_vec();
    let reference = GemmReference::compute(n, &a_host, &b_host)?;

    // init_s: tape 構築 + 葉 Var 登録 + 初回 matmul + ホスト実体化までの
    // 経過（CUDA コンテキスト作成・NVRTC コンパイル等の一度きりのコストを
    // すべて含む）。
    let init_start = Instant::now();
    let tape = make_tape(&cli.device)?;
    let a = tape.var(&a_data);
    let b = tape.var(&b_data);
    let c0 = a.matmul(&b)?;
    let out0 = readout_var(&c0)?;
    let mut checksum: f64 = out0.iter().map(|&x| x as f64).sum();
    let init_s = init_start.elapsed().as_secs_f64();
    // イシュー #965 codex-review 指摘: checksum は毎反復上書きされるため、
    // ループ後に最後の値だけを検査すると途中反復の縮退を見逃す。init 計測分
    // を含め reuse 経路（同一 tape を使い回す）でも fresh 経路と同様に
    // checksum 計算直後・全反復で検証する。
    validate_gemm_checksum(checksum)?;
    // イシュー #970: init 計測分の要素単位検証は init_s の外（elapsed 取得後）
    // で行う。以後の反復と worst-case で集約する。
    let mut parity = reference.verify(&out0)?;

    // 残り warmup（1 回は init 計測内で消費済み）+ 計測本体。同一 tape・同一
    // 葉 Var を使い回し、matmul のみを繰り返す。
    let mut one = || -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let c = a.matmul(&b)?;
        let out = readout_var(&c)?;
        checksum = out.iter().map(|&x| x as f64).sum();
        let elapsed = start.elapsed();
        validate_gemm_checksum(checksum)?;
        parity = parity.worst(reference.verify(&out)?);
        Ok(elapsed)
    };
    for _ in 0..WARMUP_ITERS.saturating_sub(1) {
        one()?;
    }
    let mut durations = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        durations.push(one()?);
    }
    let st = stats(&durations)?;
    Record {
        framework: FRAMEWORK,
        framework_version: VERSION,
        task: "gemm",
        device: &cli.device,
        size: n,
        stats: st,
        gflops: Some(gemm_gflops(n, st.median_s)),
        throughput_per_s: None,
        checksum,
        warmup: WARMUP_ITERS,
        iters: MEASURE_ITERS,
        mode: "reuse",
        init_s: Some(init_s),
        parity: Some(parity),
        tf32: false,
    }
    .emit(&cli.out)?;
    Ok(())
}

/// `gemm --mode reuse --phases`（イシュー #1182）の計測本体。`run_gemm_reuse`
/// と**同一の処理順・同一 API 呼び出し**（参照 GEMM を init 前に計算・
/// init_s の定義・`validate_gemm_checksum` を全反復で実施・
/// `GemmReference::verify` は計測窓外で worst 集約）を保ちながら、
/// `readout_var`（`to_tensor` + `contiguous().as_slice().to_vec()` を
/// まとめて呼ぶ）をここでは展開し、`to_tensor`／`host_copy`（ホスト
/// コピー）／`checksum`（全要素和）を個別区間として `Instant` で計時する。
/// `run_gemm_reuse` 本体は本関数の追加によって変更しない（AC-2）。
fn measure_gemm_reuse_phases(
    cli: &Cli,
) -> Result<(PhaseSamples, f64, f64, ParityStats), Box<dyn std::error::Error>> {
    let n = cli.size;
    let (a_data, b_data) = gemm_inputs(n)?;
    let a_host = a_data
        .contiguous()
        .as_slice()
        .ok_or("a_data as_slice() returned None")?
        .to_vec();
    let b_host = b_data
        .contiguous()
        .as_slice()
        .ok_or("b_data as_slice() returned None")?
        .to_vec();
    let reference = GemmReference::compute(n, &a_host, &b_host)?;

    // init_s: `run_gemm_reuse` と同一定義（tape 構築 + 葉 Var 登録 +
    // 初回 matmul + ホスト実体化までの経過）。区間分解の対象は warmup
    // 残り + 計測本体のみとし、init 自体は phase を push しない
    // （`measure_train_reuse_phases` の init と同型）。
    let init_start = Instant::now();
    let tape = make_tape(&cli.device)?;
    let a = tape.var(&a_data);
    let b = tape.var(&b_data);
    let c0 = a.matmul(&b)?;
    let out0 = readout_var(&c0)?;
    let mut checksum: f64 = out0.iter().map(|&x| x as f64).sum();
    let init_s = init_start.elapsed().as_secs_f64();
    validate_gemm_checksum(checksum)?;
    let mut parity = reference.verify(&out0)?;

    let mut phases = PhaseSamples::new();
    // 1 回は init 計測内で消費済み（`run_gemm_reuse` と同じ warmup 消費
    // 規約）。
    let total_iters = WARMUP_ITERS.saturating_sub(1) + MEASURE_ITERS;
    for _ in 0..total_iters {
        let iter_start = Instant::now();

        let t0 = Instant::now();
        let c = a.matmul(&b)?;
        phases.push(PHASE_GEMM_MATMUL, t0.elapsed());

        let t0 = Instant::now();
        let t = c.to_tensor();
        phases.push(PHASE_GEMM_TO_TENSOR, t0.elapsed());

        let t0 = Instant::now();
        let out = t
            .contiguous()
            .as_slice()
            .ok_or("as_slice() returned None after contiguous()")?
            .to_vec();
        phases.push(PHASE_GEMM_HOST_COPY, t0.elapsed());

        let t0 = Instant::now();
        checksum = out.iter().map(|&x| x as f64).sum();
        phases.push(PHASE_GEMM_CHECKSUM, t0.elapsed());

        // `run_gemm_reuse` と同じく、elapsed の計測窓はここで閉じる
        // （checksum 計算までを計時対象とし、以降の検証コストは含めない）。
        phases.push(PHASE_GEMM_ITER_TOTAL, iter_start.elapsed());

        validate_gemm_checksum(checksum)?;
        parity = parity.worst(reference.verify(&out)?);
    }

    Ok((phases, checksum, init_s, parity))
}

/// [`measure_gemm_reuse_phases`] の結果を phase ごとに 1 行の JSONL
/// （`task:"gemm_phases"`）として出力する。train 側 `emit_phase_records`
/// （`task:"train_phases"` 固定・`TRAIN_WARMUP`／`BATCH` 決め打ち）とは
/// task・size・warmup・iters・parity の扱いが異なるため合流させず、
/// `gemm_phases` 専用の emit 関数として並置する（`summarize.py` 側も
/// `gemm_phases` 専用の集計節を持つ。README「`gemm --mode reuse --phases`」
/// 節参照）。
fn emit_gemm_phase_records(
    cli: &Cli,
    phases: &PhaseSamples,
    mode: &'static str,
    checksum: f64,
    init_s: Option<f64>,
    parity: ParityStats,
) -> Result<(), Box<dyn std::error::Error>> {
    let warmup = WARMUP_ITERS.saturating_sub(1);
    for (phase_index, &phase) in phases.order.iter().enumerate() {
        let measured = &phases.durations(phase)[warmup..];
        let st = stats(measured)?;
        PhaseRecord {
            base: Record {
                framework: FRAMEWORK,
                framework_version: VERSION,
                task: "gemm_phases",
                device: &cli.device,
                size: cli.size,
                stats: st,
                gflops: None,
                throughput_per_s: None,
                checksum,
                warmup,
                iters: measured.len(),
                mode,
                init_s,
                parity: Some(parity),
                tf32: false,
            },
            phase,
            phase_index,
        }
        .emit(&cli.out)?;
    }
    Ok(())
}

/// `--task gemm --mode reuse --phases`（イシュー #1182）。
fn run_gemm_reuse_phases(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let (phases, checksum, init_s, parity) = measure_gemm_reuse_phases(cli)?;
    emit_gemm_phase_records(cli, &phases, "reuse", checksum, Some(init_s), parity)
}

fn mlp_data() -> Result<(Tensor<f32>, Tensor<f32>), Box<dyn std::error::Error>> {
    let x = Xorshift64Star::new(SEED_X).fill_vec(BATCH * D_IN);
    let y = Xorshift64Star::new(SEED_Y).fill_vec(BATCH * D_OUT);
    Ok((
        Tensor::new(x, &[BATCH, D_IN])?,
        Tensor::new(y, &[BATCH, D_OUT])?,
    ))
}

fn build_model() -> Result<Sequential, Box<dyn std::error::Error>> {
    Ok(Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)?
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)?)
}

fn run_train(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let mut model = build_model()?;
    let (x_data, y_data) = mlp_data()?;
    let mut durations = Vec::with_capacity(TRAIN_STEPS);
    let mut last_loss = 0.0f32;

    for _ in 0..TRAIN_STEPS {
        let start = Instant::now();
        let updated: Vec<Tensor<f32>> = {
            // fresh tape per step
            let tape = make_tape(&cli.device)?;
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);
            let pred = bound.forward(&tape, &x)?;
            let loss = pred.mse_loss(&y)?;
            // host readout of the loss (sync point inside the step)
            last_loss = loss
                .to_tensor()
                .get(&[])
                .ok_or("loss should be a scalar with shape []")?;
            let grads = tape.backward(&loss)?;
            let grad_refs = bound.trainable_grads(&grads)?;
            let param_refs = model.trainable_parameters();
            let mut next = Vec::with_capacity(param_refs.len());
            for (param, grad) in param_refs.iter().zip(grad_refs.iter()) {
                let p = param
                    .contiguous()
                    .as_slice()
                    .ok_or("param as_slice None")?
                    .to_vec();
                let g = grad
                    .contiguous()
                    .as_slice()
                    .ok_or("grad as_slice None")?
                    .to_vec();
                let upd: Vec<f32> = p.iter().zip(g.iter()).map(|(p, g)| p - LR * g).collect();
                next.push(Tensor::from_slice(&upd, param.shape())?);
            }
            next
        };
        model.apply_parameters(updated)?;
        durations.push(start.elapsed());
    }

    if !last_loss.is_finite() {
        return Err(format!("MEASURE_ERROR: final loss not finite: {last_loss}").into());
    }
    let measured = &durations[TRAIN_WARMUP..];
    let st = stats(measured)?;
    Record {
        framework: FRAMEWORK,
        framework_version: VERSION,
        task: "train",
        device: &cli.device,
        size: BATCH,
        stats: st,
        gflops: None,
        throughput_per_s: None,
        checksum: last_loss as f64,
        warmup: TRAIN_WARMUP,
        iters: measured.len(),
        mode: "fresh",
        init_s: None,
        parity: None,
        tf32: false,
    }
    .emit(&cli.out)?;
    Ok(())
}

/// `--mode reuse` の train 計測（イシュー #958）。`DeviceParamStore` を
/// 1 回だけ構築し（`init_s` として初期化コストを分離記録）、以後の各 step
/// は新規 tape 上で `forward_resident` → `backward_device_param_store` →
/// `step_device_param_store`（デバイス上 SGD 更新）を行う。ホスト経由の
/// download/upload（fresh の `p - lr*g` 相当）はループ内で一切行わない。
/// モジュール doc「train --mode reuse」節に設計判断の詳細を記す。
///
/// fandhe-ai 0.5.0 から `forward_resident`（イシュー #1059）が積むグラフは
/// `Op::LinearResident` を含み、素の `Tape::backward` はこれを解決できず
/// 型付きエラーで拒否する。`store` の DeviceParamStore を渡す
/// `Tape::backward_device_param_store` が必須（`docs/device-resident-
/// update-design.md` §3.3e・`docs/compat-api-scope.md`「backward」節）。
fn run_train_reuse(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let model = build_model()?;
    let (x_data, y_data) = mlp_data()?;

    // init_s: 初回 tape 構築 + 全パラメータの 1 回限りの H2D upload
    // （`init_device_param_store`）+ その完了保証のための明示同期点まで
    // の経過（README「train --mode reuse」節「init_s の定義」参照）。
    // 以後の DeviceParamStore はこのデバイス・バッファを固定して使い回す。
    //
    // `init_device_param_store` の内部実装（`MemoryOps::upload`）は CUDA
    // では `clone_htod` 等の非同期 H2D コピーを発行するのみで、発行元
    // ストリーム上の完了を待たない（`DeviceParamStore::new` doc・
    // `CudaMemory::upload_inner` 参照）。`elapsed()` を非同期発行直後に
    // 取得すると、転送完了待ちのコストが `init_s` から漏れ、代わりに
    // 最初の `forward_resident` 内の download 同期（暗黙のストリーム
    // 同期点）へ計上されてしまう（codex-review PR #998 P2 指摘 1）。
    //
    // 完了を保証する同期点として `sync_device_param_store_to_host` を
    // 使うのは、`bench-fandhe` が依存できる公開 API 面（`fandhe-ai
    // =0.4.0`。第 9 区分の適用範囲拡張・`.claude/rules/deps-policy.md`）
    // に「ホスト転送を伴わない完了待ち」（`bench-harness::sync::
    // SyncPoint::wait_idle` 相当）が公開されていないためであり、これは
    // `bench-fandhe`（`fandhe-ai` crate 経由のみで実装する制約）からは
    // 解決できないギャップである。そのため `init_s` は「純粋な H2D
    // upload 時間」ではなく、そのアップロードを確定させる D2H 実体化
    // （`sync_to_host` が返す `Vec<Tensor<f32>>` の構築コスト）も含む
    // （codex-review PR #998 P2 指摘 2。ダウンロードした内容自体は計測
    // 目的ではなく破棄する）。この扱いは `run_gemm_reuse` の `init_s`
    // が「初回 matmul + ホスト実体化」を明示的に含めている前例（本ファイル
    // `run_gemm_reuse` doc 参照）と整合する。公開 API 面へホスト転送を
    // 伴わない完了待ちを追加する対応は本 PR のスコープ外（`facade` の
    // 公開面変更・crates.io 再公開を要するため）とし、必要であれば別途
    // 追跡する。
    let init_start = Instant::now();
    let init_tape = make_tape(&cli.device)?;
    let mut store = model.init_device_param_store(&init_tape)?;
    let _ = init_tape.sync_device_param_store_to_host(&store)?;
    drop(init_tape);
    let init_s = init_start.elapsed().as_secs_f64();

    let config = SgdConfig::new(LR);
    let mut durations = Vec::with_capacity(TRAIN_STEPS);
    let mut last_loss = 0.0f32;

    // ループ全体の計時に関する注意（codex-review PR #998 P2 指摘 3）:
    // `step_device_param_store`（デバイス上 SGD 更新）が CUDA で更新
    // カーネルを非同期発行する場合、直後の `start.elapsed()` はその
    // 更新の完了を待たない。ここで `sync_to_host` 相当の明示同期を
    // 追加すると（init_s と同じ理由で）D2H 実体化コストが毎 step の
    // 計測へ混入し、reuse が候補とする「ホスト転送を伴わない完了待ち」
    // API（上記 init_s のコメント参照）が公開 API 面に無いという同じ
    // ギャップに阻まれる。
    //
    // 代わりに、この step 自身の `loss_readout`（`loss.to_tensor().get()`
    // の D2H 実体化）を同期点として利用する（codex-review PR #1104 P2
    // 是正: 0.5.0 の `forward_resident` は #1059 で D2H を伴わない
    // `register_resident_params` に切り替わっており、forward 側の D2H
    // には依存しない。モジュール doc「既知の前提」節参照）。
    // `docs/backend-cuda-async-execution-design.md` §3 のストリーム順序
    // 保証（I1）・同期点での完了保証（I2）により、`loss_readout` の
    // D2H は先行する全投入済み作業（forward_i 自身に加え、ループ内で
    // その手前に投入済みの前 step の backward_{i-1}・update_{i-1} を
    // 含む）の完了を保証してから復帰する。よって計測窓 i は
    // 実際には「step i-1 の更新完了待ち + forward_i + backward_i +
    // step i の更新発行」を計測しており、定常状態では
    // `forward + backward + update` の総和に等しい（境界がひとつずれる
    // だけで、欠落する項はない）。ずれの影響を受けるのは先頭の計測 step
    // （warmup 20 step 側に含まれ捨てられる）と最終 step の更新完了
    // （ループ後の `sync_device_param_store_to_host` による終端同期が
    // 保証する）のみであり、`median_s`/`q1_s`/`q3_s` の対象となる残り
    // 80 step の統計には影響しない。
    for _ in 0..TRAIN_STEPS {
        let start = Instant::now();
        // fresh tape per step（モジュール doc 参照: DeviceParamStore は
        // 使い回すが tape は使い回さない）。
        let tape = make_tape(&cli.device)?;
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = model.forward_resident(&tape, &x, &mut store)?;
        let loss = pred.mse_loss(&y)?;
        // host readout of the loss（fresh と同じくループ内の同期点）。
        last_loss = loss
            .to_tensor()
            .get(&[])
            .ok_or("loss should be a scalar with shape []")?;
        let grads = tape.backward_device_param_store(&loss, &store)?;
        // デバイス上 SGD 更新（ホストへの download/upload を経由しない）。
        tape.step_device_param_store(&mut store, &grads, &config)?;
        durations.push(start.elapsed());
    }

    if !last_loss.is_finite() {
        return Err(format!("MEASURE_ERROR: final loss not finite: {last_loss}").into());
    }

    // 終端同期: 新規 tape から DeviceParamStore の内容をホストへ実体化する
    // （計測窓の外。gemm reuse の checksum 実体化と同じ位置づけ）。件数が
    // trainable_parameters() と不一致・非有限要素があれば MEASURE_ERROR
    // として記録を拒否する（A08: 破損した学習結果を性能値として残さない）。
    let final_tape = make_tape(&cli.device)?;
    let synced = final_tape.sync_device_param_store_to_host(&store)?;
    let expected_len = model.trainable_parameters().len();
    if synced.len() != expected_len {
        return Err(format!(
            "MEASURE_ERROR: sync_device_param_store_to_host returned {} tensors, expected {expected_len}",
            synced.len()
        )
        .into());
    }
    for t in &synced {
        let slice = t
            .contiguous()
            .as_slice()
            .ok_or("synced param as_slice() returned None")?
            .to_vec();
        if slice.iter().any(|v| !v.is_finite()) {
            return Err("MEASURE_ERROR: synced parameter contains non-finite element".into());
        }
    }

    let measured = &durations[TRAIN_WARMUP..];
    let st = stats(measured)?;
    Record {
        framework: FRAMEWORK,
        framework_version: VERSION,
        task: "train",
        device: &cli.device,
        size: BATCH,
        stats: st,
        gflops: None,
        throughput_per_s: None,
        checksum: last_loss as f64,
        warmup: TRAIN_WARMUP,
        iters: measured.len(),
        mode: "reuse",
        init_s: Some(init_s),
        parity: None,
        tf32: false,
    }
    .emit(&cli.out)?;
    Ok(())
}

/// `train --phases`（fresh。イシュー #1009）の計測本体。`run_train` と
/// 同一の処理順・同一の API 呼び出し（fresh/phases 間の最終 loss 一致は
/// `tests::train_phases_fresh_final_loss_matches_run_train` が固定する）を
/// 公開 API 呼び出し境界で区間分解する。`bound`/`param_refs`（いずれも
/// `model` への不変借用）は `model.apply_parameters`（`&mut self`）の前に
/// 明示的に手放す（`run_train` の `{}` ブロックによるスコープ終端と同じ
/// 借用構造。ここでは明示 `drop` を「テンソル解放」区間の計測点として使う）。
fn measure_train_phases(cli: &Cli) -> Result<(PhaseSamples, f32), Box<dyn std::error::Error>> {
    let mut model = build_model()?;
    let (x_data, y_data) = mlp_data()?;
    let mut phases = PhaseSamples::new();
    let mut last_loss = 0.0f32;

    for _ in 0..TRAIN_STEPS {
        let step_start = Instant::now();

        let t0 = Instant::now();
        let tape = make_tape(&cli.device)?;
        phases.push(PHASE_TAPE_BUILD, t0.elapsed());

        let t0 = Instant::now();
        let bound = model.bind(&tape);
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        phases.push(PHASE_LEAF_REGISTER, t0.elapsed());

        let t0 = Instant::now();
        let pred = bound.forward(&tape, &x)?;
        let loss = pred.mse_loss(&y)?;
        phases.push(PHASE_FORWARD, t0.elapsed());

        let t0 = Instant::now();
        last_loss = loss
            .to_tensor()
            .get(&[])
            .ok_or("loss should be a scalar with shape []")?;
        phases.push(PHASE_LOSS_READOUT, t0.elapsed());

        let t0 = Instant::now();
        let grads = tape.backward(&loss)?;
        let grad_refs = bound.trainable_grads(&grads)?;
        phases.push(PHASE_BACKWARD, t0.elapsed());

        // param_refs は model への不変借用（bound と共存可）。
        let param_refs = model.trainable_parameters();
        let t0 = Instant::now();
        let mut host_params = Vec::with_capacity(param_refs.len());
        let mut host_grads = Vec::with_capacity(param_refs.len());
        let mut shapes = Vec::with_capacity(param_refs.len());
        for (param, grad) in param_refs.iter().zip(grad_refs.iter()) {
            let p = param
                .contiguous()
                .as_slice()
                .ok_or("param as_slice None")?
                .to_vec();
            let g = grad
                .contiguous()
                .as_slice()
                .ok_or("grad as_slice None")?
                .to_vec();
            shapes.push(param.shape().to_vec());
            host_params.push(p);
            host_grads.push(g);
        }
        phases.push(PHASE_PARAM_READOUT, t0.elapsed());

        let t0 = Instant::now();
        let mut next = Vec::with_capacity(host_params.len());
        for ((p, g), shape) in host_params.iter().zip(host_grads.iter()).zip(shapes.iter()) {
            let upd: Vec<f32> = p.iter().zip(g.iter()).map(|(p, g)| p - LR * g).collect();
            next.push(Tensor::from_slice(&upd, shape)?);
        }
        phases.push(PHASE_HOST_SGD, t0.elapsed());

        // `apply_parameters` は `&mut model` を要求するため、model への
        // 不変借用（bound・param_refs）をここで明示的に手放す。この解放
        // コストは「テンソル解放」区間（tape_drop）の一部として、後段の
        // tape 解放コストと合算して記録する（プロトコル節参照）。
        let t0 = Instant::now();
        drop(param_refs);
        drop(bound);
        let borrow_release = t0.elapsed();

        let t0 = Instant::now();
        model.apply_parameters(next)?;
        phases.push(PHASE_APPLY_PARAMS, t0.elapsed());

        let t0 = Instant::now();
        drop(tape);
        phases.push(PHASE_TAPE_DROP, borrow_release + t0.elapsed());

        phases.push(PHASE_STEP_TOTAL, step_start.elapsed());
    }

    if !last_loss.is_finite() {
        return Err(format!("MEASURE_ERROR: final loss not finite: {last_loss}").into());
    }
    Ok((phases, last_loss))
}

/// `train --phases`（reuse。イシュー #1009）の計測本体。`run_train_reuse`
/// と同一の処理順・同一 API 呼び出しを区間分解する（`init_s` の定義・
/// 終端同期の検証は `run_train_reuse` と同一。モジュール doc「train
/// --mode reuse」節参照）。`backward` ではなく
/// `backward_device_param_store` を使う理由は `run_train_reuse` doc 参照
/// （0.5.0 の `Op::LinearResident` 契約。イシュー #1059）。
fn measure_train_reuse_phases(
    cli: &Cli,
) -> Result<(PhaseSamples, f32, f64), Box<dyn std::error::Error>> {
    let model = build_model()?;
    let (x_data, y_data) = mlp_data()?;

    let init_start = Instant::now();
    let init_tape = make_tape(&cli.device)?;
    let mut store = model.init_device_param_store(&init_tape)?;
    let _ = init_tape.sync_device_param_store_to_host(&store)?;
    drop(init_tape);
    let init_s = init_start.elapsed().as_secs_f64();

    let config = SgdConfig::new(LR);
    let mut phases = PhaseSamples::new();
    let mut last_loss = 0.0f32;

    for _ in 0..TRAIN_STEPS {
        let step_start = Instant::now();

        let t0 = Instant::now();
        let tape = make_tape(&cli.device)?;
        phases.push(PHASE_TAPE_BUILD, t0.elapsed());

        let t0 = Instant::now();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        phases.push(PHASE_LEAF_REGISTER, t0.elapsed());

        let t0 = Instant::now();
        let pred = model.forward_resident(&tape, &x, &mut store)?;
        let loss = pred.mse_loss(&y)?;
        phases.push(PHASE_FORWARD_RESIDENT, t0.elapsed());

        let t0 = Instant::now();
        last_loss = loss
            .to_tensor()
            .get(&[])
            .ok_or("loss should be a scalar with shape []")?;
        phases.push(PHASE_LOSS_READOUT, t0.elapsed());

        let t0 = Instant::now();
        let grads = tape.backward_device_param_store(&loss, &store)?;
        phases.push(PHASE_BACKWARD, t0.elapsed());

        let t0 = Instant::now();
        tape.step_device_param_store(&mut store, &grads, &config)?;
        phases.push(PHASE_DEVICE_UPDATE, t0.elapsed());

        let t0 = Instant::now();
        drop(tape);
        phases.push(PHASE_TAPE_DROP, t0.elapsed());

        phases.push(PHASE_STEP_TOTAL, step_start.elapsed());
    }

    if !last_loss.is_finite() {
        return Err(format!("MEASURE_ERROR: final loss not finite: {last_loss}").into());
    }

    // 終端同期: run_train_reuse と同一の検証（A08: 破損した学習結果を
    // 性能値として残さない）。
    let final_tape = make_tape(&cli.device)?;
    let synced = final_tape.sync_device_param_store_to_host(&store)?;
    let expected_len = model.trainable_parameters().len();
    if synced.len() != expected_len {
        return Err(format!(
            "MEASURE_ERROR: sync_device_param_store_to_host returned {} tensors, expected {expected_len}",
            synced.len()
        )
        .into());
    }
    for t in &synced {
        let slice = t
            .contiguous()
            .as_slice()
            .ok_or("synced param as_slice() returned None")?
            .to_vec();
        if slice.iter().any(|v| !v.is_finite()) {
            return Err("MEASURE_ERROR: synced parameter contains non-finite element".into());
        }
    }

    Ok((phases, last_loss, init_s))
}

/// `measure_train_phases`/`measure_train_reuse_phases` の結果を phase ごと
/// に 1 行の JSONL（`task:"train_phases"`）として出力する（§3.3）。
/// `--phases` 実行時は既存の `task:"train"` 行は出さない（`step_total` 行が
/// 代替する。計時分割つきの step 合計を通常プロトコルの値と混同させない
/// ため）。
fn emit_phase_records(
    cli: &Cli,
    phases: &PhaseSamples,
    mode: &'static str,
    checksum: f64,
    init_s: Option<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    for (phase_index, &phase) in phases.order.iter().enumerate() {
        let measured = &phases.durations(phase)[TRAIN_WARMUP..];
        let st = stats(measured)?;
        PhaseRecord {
            base: Record {
                framework: FRAMEWORK,
                framework_version: VERSION,
                task: "train_phases",
                device: &cli.device,
                size: BATCH,
                stats: st,
                gflops: None,
                throughput_per_s: None,
                checksum,
                warmup: TRAIN_WARMUP,
                iters: measured.len(),
                mode,
                init_s,
                parity: None,
                tf32: false,
            },
            phase,
            phase_index,
        }
        .emit(&cli.out)?;
    }
    Ok(())
}

/// `--task train --mode fresh --phases`（イシュー #1009）。
fn run_train_phases(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let (phases, last_loss) = measure_train_phases(cli)?;
    emit_phase_records(cli, &phases, "fresh", last_loss as f64, None)
}

/// `--task train --mode reuse --phases`（イシュー #1009）。
fn run_train_reuse_phases(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let (phases, last_loss, init_s) = measure_train_reuse_phases(cli)?;
    emit_phase_records(cli, &phases, "reuse", last_loss as f64, Some(init_s))
}

fn run_infer(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let model = build_model()?;
    let (x_data, _) = mlp_data()?;
    let mut checksum = 0.0;

    let one = |sync_checksum: &mut f64| -> Result<Duration, Box<dyn std::error::Error>> {
        match cli.device.as_str() {
            "cpu" => {
                // predict() builds an internal default (CPU) tape
                let start = Instant::now();
                let out = model.predict(&x_data)?;
                *sync_checksum = checksum_tensor(&out)?;
                Ok(start.elapsed())
            }
            _ => {
                // explicit tape on the requested device + forward + host sync
                let tape = make_tape(&cli.device)?;
                let start = Instant::now();
                let x = tape.var(&x_data);
                let out = model.forward(&tape, &x)?;
                *sync_checksum = checksum_var(&out)?;
                Ok(start.elapsed())
            }
        }
    };

    for _ in 0..WARMUP_ITERS {
        one(&mut checksum)?;
    }
    let mut durations = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        durations.push(one(&mut checksum)?);
    }
    let st = stats(&durations)?;
    Record {
        framework: FRAMEWORK,
        framework_version: VERSION,
        task: "infer",
        device: &cli.device,
        size: BATCH,
        stats: st,
        gflops: None,
        throughput_per_s: Some(1.0 / st.median_s),
        checksum,
        warmup: WARMUP_ITERS,
        iters: MEASURE_ITERS,
        mode: "fresh",
        init_s: None,
        parity: None,
        tf32: false,
    }
    .emit(&cli.out)?;
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_cli()?;
    dispatch(&cli)
}

/// task × mode × `--phases` の分岐。reuse モードは受け入れ条件の範囲
/// （gemm・train）に限定し、infer × reuse は MEASURE_ERROR で fail-fast
/// する（gemm はイシュー #925 §2.1・§8、train はイシュー #958。infer への
/// 拡張は将来対応が必要な場合は別イシューで追跡）。`--phases` は
/// `--task train`（fresh/reuse 双方）にのみ対応し、`gemm`/`infer` との
/// 組合せは MEASURE_ERROR とする（イシュー #1009）。`run()` から分離して
/// あるのは `parse_cli()`（`std::env::args()` 依存）を経由せず
/// `tests::phases_with_gemm_or_infer_is_measure_error` から直接分岐を
/// 検証できるようにするため。
///
/// **`--tf32`（イシュー #1042）は本バイナリでは常に MEASURE_ERROR で
/// fail-fast する**: `bench-fandhe` は crates.io 公開版 `fandhe-ai
/// =0.6.0` に完全固定されており（deps-policy 第 9 区分。
/// `check_framework_compare` が registry 取得元を fail-closed 検査する
/// ため path 依存への差し替えは不可）。`fandhe_ai::set_cuda_tf32_gemm_enabled`
/// 自体は crates.io 公開版から呼び出し可能になったが（承認ピンは v0.5.0 公開
/// 時点で `>= 0.5.0` を満たしている）、`bench-fandhe`（`main.rs`）側の
/// 呼び出し結線・`run_all` の tf32 スイープ追加（C-2。
/// `docs/cuda-tf32-optin-api-decision.md` 参照）は依然未実施のため
/// fail-fast する。`--phases` の対象外組合せ拒否と同型の allowlist 方式で、
/// `cli.phases`（`match` の第 3 要素）より先に検査する。
fn dispatch(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    if cli.tf32 {
        return Err(
            "MEASURE_ERROR: --tf32 requires fandhe-ai >= 0.5.0 (wiring not implemented in \
             bench-fandhe yet; see docs/cuda-tf32-optin-api-decision.md C-2; issue #1042)"
                .into(),
        );
    }
    match (cli.task.as_str(), cli.mode.as_str(), cli.phases) {
        ("train", "fresh", true) => run_train_phases(cli),
        ("train", "reuse", true) => run_train_reuse_phases(cli),
        // イシュー #1182: `gemm --mode reuse --phases` を追加。fresh の
        // gemm・infer との組合せは引き続き MEASURE_ERROR（下の catch-all）。
        ("gemm", "reuse", true) => run_gemm_reuse_phases(cli),
        (task, mode, true) => Err(format!(
            "MEASURE_ERROR: --phases is only implemented for task 'train' or task 'gemm' with \
             --mode reuse (got task='{task}' mode='{mode}'; issue #1009 / #1182)"
        )
        .into()),
        ("gemm", "fresh", false) => run_gemm(cli),
        ("gemm", "reuse", false) => run_gemm_reuse(cli),
        ("train", "fresh", false) => run_train(cli),
        ("train", "reuse", false) => run_train_reuse(cli),
        ("infer", "fresh", false) => run_infer(cli),
        (task, "reuse", false) => Err(format!(
            "MEASURE_ERROR: --mode reuse is not implemented for task '{task}' (gemm / train only; issue #925 / #958)"
        )
        .into()),
        (other, _, false) => Err(format!("MEASURE_ERROR: unknown task '{other}'").into()),
    }
}

/// `run_train`（fresh）と `run_train_reuse`（reuse）の最終 loss（checksum）
/// が統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
/// `.claude/rules/coding-rust.md`）の範囲内で一致することを検証する
/// （受け入れ条件 5。イシュー #958）。cpu 経由・実機非依存のため `#[ignore]`
/// は付けない。`--release` 推奨（README 参照。debug では GEMM が遅い）。
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// テスト間で衝突しない一時 JSONL パスを作る（pid + カウンタで一意化。
    /// 並行テスト実行時の読み取り／削除の混入を防ぐ）。
    fn temp_out_path(tag: &str) -> std::path::PathBuf {
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "bench-fandhe-test-{tag}-{}-{n}.jsonl",
            std::process::id()
        ))
    }

    fn make_cli(task: &str, mode: &str, out: &std::path::Path) -> Cli {
        Cli {
            task: task.to_string(),
            device: "cpu".to_string(),
            size: 64,
            out: out.to_string_lossy().into_owned(),
            mode: mode.to_string(),
            phases: false,
            tf32: false,
        }
    }

    /// JSONL の最終行から `"checksum":<value>` を取り出す最小パーサー
    /// （フル JSON デコーダを新規依存させないための最小実装。`Record::emit`
    /// の出力形式に依存する）。
    fn last_line_checksum(path: &std::path::Path) -> f64 {
        let content = std::fs::read_to_string(path).expect("test: JSONL 読み取り失敗");
        let last = content.lines().next_back().expect("test: JSONL に行がない");
        let key = "\"checksum\":";
        let start = last
            .find(key)
            .expect("test: checksum フィールドが見つからない")
            + key.len();
        let rest = &last[start..];
        let end = rest
            .find([',', '}'])
            .expect("test: checksum フィールドの終端が見つからない");
        rest[..end]
            .trim()
            .parse::<f64>()
            .expect("test: checksum を f64 としてパースできない")
    }

    #[test]
    fn train_reuse_matches_fresh_final_loss_within_composite_tolerance() {
        let fresh_path = temp_out_path("fresh");
        let reuse_path = temp_out_path("reuse");

        run_train(&make_cli("train", "fresh", &fresh_path)).expect("run_train (fresh) failed");
        run_train_reuse(&make_cli("train", "reuse", &reuse_path))
            .expect("run_train_reuse (reuse) failed");

        let fresh_checksum = last_line_checksum(&fresh_path);
        let reuse_checksum = last_line_checksum(&reuse_path);

        let _ = std::fs::remove_file(&fresh_path);
        let _ = std::fs::remove_file(&reuse_path);

        assert!(
            fresh_checksum.is_finite() && reuse_checksum.is_finite(),
            "checksum must be finite: fresh={fresh_checksum} reuse={reuse_checksum}"
        );
        let abs_diff = (fresh_checksum - reuse_checksum).abs();
        let rel_diff = abs_diff / fresh_checksum.abs().max(1e-12);
        assert!(
            abs_diff < 1e-5 || rel_diff < 1e-3,
            "fresh/reuse final loss mismatch: fresh={fresh_checksum} reuse={reuse_checksum} \
             abs_diff={abs_diff} rel_diff={rel_diff}"
        );
    }

    #[test]
    fn train_reuse_produces_expected_record_fields() {
        let out = temp_out_path("reuse-fields");
        run_train_reuse(&make_cli("train", "reuse", &out)).expect("run_train_reuse failed");
        let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&out);
        let last = content.lines().next_back().expect("test: JSONL に行がない");
        assert!(last.contains("\"task\":\"train\""), "line={last}");
        assert!(last.contains("\"mode\":\"reuse\""), "line={last}");
        assert!(last.contains("\"init_s\":"), "line={last}");
        assert!(!last.contains("\"init_s\":null"), "line={last}");
    }

    // イシュー #1009: `train --phases` の受け入れ条件（区間別 median/Q1/Q3
    // を JSONL に出力する・cpu で動作する・fresh/reuse の最終 loss が
    // run_train/run_train_reuse と一致する）を固定する回帰テスト群。

    fn make_phases_cli(mode: &str, out: &std::path::Path) -> Cli {
        Cli {
            task: "train".to_string(),
            device: "cpu".to_string(),
            size: 64,
            out: out.to_string_lossy().into_owned(),
            mode: mode.to_string(),
            phases: true,
            tf32: false,
        }
    }

    #[test]
    fn train_phases_fresh_emits_one_row_per_phase_in_order() {
        let out = temp_out_path("phases-fresh-order");
        run_train_phases(&make_phases_cli("fresh", &out)).expect("run_train_phases failed");
        let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&out);
        let lines: Vec<&str> = content.lines().collect();
        // fresh の区間数（モジュール doc の PHASE_* 定数のうち fresh 経路で
        // 使うもの）: tape_build/leaf_register/forward/loss_readout/
        // backward/param_readout/host_sgd/apply_params/tape_drop/step_total
        // = 10。
        assert_eq!(lines.len(), 10, "lines={lines:?}");
        for (i, line) in lines.iter().enumerate() {
            assert!(line.contains("\"task\":\"train_phases\""), "line={line}");
            assert!(
                line.contains(&format!("\"phase_index\":{i}")),
                "line={line}"
            );
        }
        assert!(lines.last().unwrap().contains("\"phase\":\"step_total\""));
        assert!(!lines.iter().any(|l| l.contains("\"init_s\":")));
    }

    #[test]
    fn train_phases_each_step_phase_sum_does_not_exceed_total() {
        let cli = make_phases_cli("fresh", &temp_out_path("phases-fresh-sum"));
        let (phases, _last_loss) = measure_train_phases(&cli).expect("measure_train_phases failed");
        let totals = phases.durations(PHASE_STEP_TOTAL);
        assert_eq!(totals.len(), TRAIN_STEPS);
        let component_phases: Vec<&str> = phases
            .order
            .iter()
            .copied()
            .filter(|&p| p != PHASE_STEP_TOTAL)
            .collect();
        for (step, &total) in totals.iter().enumerate() {
            let sum: Duration = component_phases
                .iter()
                .map(|&p| phases.durations(p)[step])
                .sum();
            assert!(
                sum <= total,
                "step={step}: phase sum {sum:?} exceeds step_total {total:?}"
            );
            // 計時オーバーヘッド（Instant::now() 呼び出し自体のコスト）の
            // 上限を固定する回帰テスト。数値一致許容誤差ではない
            // （coding-rust.md のバックエンド間許容誤差とは無関係）。
            assert!(
                sum.as_secs_f64() >= 0.9 * total.as_secs_f64(),
                "step={step}: phase sum {sum:?} is less than 90% of step_total {total:?}"
            );
        }
    }

    #[test]
    fn train_phases_fresh_final_loss_matches_run_train() {
        let fresh_path = temp_out_path("fresh-vs-phases");
        run_train(&make_cli("train", "fresh", &fresh_path)).expect("run_train failed");
        let fresh_checksum = last_line_checksum(&fresh_path);
        let _ = std::fs::remove_file(&fresh_path);

        let phases_cli = make_phases_cli("fresh", &temp_out_path("phases-loss-fresh"));
        let (_phases, last_loss) =
            measure_train_phases(&phases_cli).expect("measure_train_phases failed");
        let phases_checksum = last_loss as f64;

        let abs_diff = (fresh_checksum - phases_checksum).abs();
        let rel_diff = abs_diff / fresh_checksum.abs().max(1e-12);
        assert!(
            abs_diff < 1e-5 || rel_diff < 1e-3,
            "run_train/measure_train_phases final loss mismatch: \
             fresh={fresh_checksum} phases={phases_checksum} abs_diff={abs_diff} rel_diff={rel_diff}"
        );
    }

    #[test]
    fn train_reuse_phases_final_loss_matches_run_train_reuse() {
        let reuse_path = temp_out_path("reuse-vs-phases");
        run_train_reuse(&make_cli("train", "reuse", &reuse_path)).expect("run_train_reuse failed");
        let reuse_checksum = last_line_checksum(&reuse_path);
        let _ = std::fs::remove_file(&reuse_path);

        let phases_cli = make_phases_cli("reuse", &temp_out_path("phases-loss-reuse"));
        let (_phases, last_loss, _init_s) =
            measure_train_reuse_phases(&phases_cli).expect("measure_train_reuse_phases failed");
        let phases_checksum = last_loss as f64;

        let abs_diff = (reuse_checksum - phases_checksum).abs();
        let rel_diff = abs_diff / reuse_checksum.abs().max(1e-12);
        assert!(
            abs_diff < 1e-5 || rel_diff < 1e-3,
            "run_train_reuse/measure_train_reuse_phases final loss mismatch: \
             reuse={reuse_checksum} phases={phases_checksum} abs_diff={abs_diff} rel_diff={rel_diff}"
        );
    }

    #[test]
    fn train_reuse_phases_includes_init_s() {
        let out = temp_out_path("phases-reuse-init");
        run_train_reuse_phases(&make_phases_cli("reuse", &out))
            .expect("run_train_reuse_phases failed");
        let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&out);
        for line in content.lines() {
            assert!(line.contains("\"task\":\"train_phases\""), "line={line}");
            assert!(line.contains("\"mode\":\"reuse\""), "line={line}");
            assert!(line.contains("\"init_s\":"), "line={line}");
            assert!(!line.contains("\"init_s\":null"), "line={line}");
        }
    }

    #[test]
    fn phases_with_gemm_fresh_or_infer_is_measure_error() {
        // `dispatch()`（`run()` の分岐本体。`parse_cli()` を経由せず直接
        // 呼べるよう分離してある）を通して、`--phases` が `--task train`
        // （fresh/reuse）と `--task gemm --mode reuse`（イシュー #1182）
        // 限定であり、`gemm --mode fresh`・`infer` は引き続き拒否される
        // ことを固定する。
        for (task, mode) in [("gemm", "fresh"), ("infer", "fresh")] {
            let out = temp_out_path(&format!("phases-unsupported-{task}-{mode}"));
            let cli = Cli {
                task: task.to_string(),
                device: "cpu".to_string(),
                size: 64,
                out: out.to_string_lossy().into_owned(),
                mode: mode.to_string(),
                phases: true,
                tf32: false,
            };
            let err = dispatch(&cli).expect_err("task/--phases combination must be rejected");
            let msg = err.to_string();
            assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
            assert!(msg.contains("--phases"), "msg={msg}");
            assert!(msg.contains(task), "msg={msg}");
        }
    }

    // イシュー #1182: `gemm --mode reuse --phases` の受け入れ条件
    // （区間別 median/Q1/Q3 を JSONL に出力する・cpu で動作する・
    // checksum が `run_gemm_reuse` と一致する・phase 合計が iter_total を
    // 超えない）を固定する回帰テスト群。`train --phases` の (a)〜(d) と
    // 同型。

    fn make_gemm_phases_cli(out: &std::path::Path) -> Cli {
        Cli {
            task: "gemm".to_string(),
            device: "cpu".to_string(),
            size: 64,
            out: out.to_string_lossy().into_owned(),
            mode: "reuse".to_string(),
            phases: true,
            tf32: false,
        }
    }

    /// (a) `gemm --mode reuse --phases` が 5 区間（matmul/to_tensor/
    /// host_copy/checksum/iter_total）を `phase_index` 連番・
    /// `task:"gemm_phases"`・`init_s` あり・`parity_*` キーあり・末尾
    /// `iter_total` で出力することを固定する。
    #[test]
    fn gemm_reuse_phases_emits_one_row_per_phase_in_order() {
        let out = temp_out_path("gemm-phases-order");
        run_gemm_reuse_phases(&make_gemm_phases_cli(&out)).expect("run_gemm_reuse_phases failed");
        let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&out);
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 5, "lines={lines:?}");
        for (i, line) in lines.iter().enumerate() {
            assert!(line.contains("\"task\":\"gemm_phases\""), "line={line}");
            assert!(line.contains("\"mode\":\"reuse\""), "line={line}");
            assert!(
                line.contains(&format!("\"phase_index\":{i}")),
                "line={line}"
            );
            assert!(line.contains("\"init_s\":"), "line={line}");
            assert!(!line.contains("\"init_s\":null"), "line={line}");
            assert!(line.contains("\"parity_total\":"), "line={line}");
        }
        assert!(lines.last().unwrap().contains("\"phase\":\"iter_total\""));
    }

    /// (c) `gemm --mode reuse --phases` の checksum（各行の `checksum`
    /// フィールド）が `run_gemm_reuse` の JSONL 出力の checksum と
    /// **JSONL-vs-JSONL** で一致することを固定する。同一入力・同一 cpu
    /// 経路・同一 `Record::to_json_line` の `{:.6}` 整形を経るため完全
    /// 一致する（in-memory の f64 同士は丸め誤差で比較できないため、
    /// 複合判定ではなく文字列一致を使う。§3.1 の設計メモ参照）。
    #[test]
    fn gemm_reuse_phases_checksum_matches_run_gemm_reuse() {
        let reuse_path = temp_out_path("gemm-reuse-vs-phases");
        run_gemm_reuse(&make_cli("gemm", "reuse", &reuse_path)).expect("run_gemm_reuse failed");
        let reuse_checksum_line = last_line_checksum(&reuse_path);
        let _ = std::fs::remove_file(&reuse_path);

        let phases_path = temp_out_path("gemm-phases-checksum");
        run_gemm_reuse_phases(&make_gemm_phases_cli(&phases_path))
            .expect("run_gemm_reuse_phases failed");
        let content = std::fs::read_to_string(&phases_path).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&phases_path);

        // `last_line_checksum` は文字列としてではなく f64 へパースして
        // 返すため、`{:.6}` 整形後の値同士を比較する形になる（同一入力・
        // 同一経路であれば bit 単位で同じ丸めを経るため完全一致する）。
        for line in content.lines() {
            let key = "\"checksum\":";
            let start = line.find(key).expect("checksum field missing") + key.len();
            let rest = &line[start..];
            let end = rest.find([',', '}']).expect("checksum field end missing");
            let phase_checksum: f64 = rest[..end].trim().parse().expect("checksum not f64");
            assert_eq!(
                phase_checksum, reuse_checksum_line,
                "phase checksum diverges from run_gemm_reuse: line={line}"
            );
        }
    }

    /// (d) `gemm --mode reuse --phases` の各反復について、構成区間
    /// （matmul/to_tensor/host_copy/checksum）の合計が `iter_total` を
    /// 超えず、かつ計時オーバーヘッドの上限（90%）を満たすことを固定する
    /// （`train_phases_each_step_phase_sum_does_not_exceed_total` と同型）。
    #[test]
    fn gemm_reuse_phases_each_iter_phase_sum_does_not_exceed_total() {
        let cli = make_gemm_phases_cli(&temp_out_path("gemm-phases-sum"));
        let (phases, _checksum, _init_s, _parity) =
            measure_gemm_reuse_phases(&cli).expect("measure_gemm_reuse_phases failed");
        let totals = phases.durations(PHASE_GEMM_ITER_TOTAL);
        assert_eq!(totals.len(), WARMUP_ITERS.saturating_sub(1) + MEASURE_ITERS);
        let component_phases: Vec<&str> = phases
            .order
            .iter()
            .copied()
            .filter(|&p| p != PHASE_GEMM_ITER_TOTAL)
            .collect();
        for (iter, &total) in totals.iter().enumerate() {
            let sum: Duration = component_phases
                .iter()
                .map(|&p| phases.durations(p)[iter])
                .sum();
            assert!(
                sum <= total,
                "iter={iter}: phase sum {sum:?} exceeds iter_total {total:?}"
            );
            assert!(
                sum.as_secs_f64() >= 0.9 * total.as_secs_f64(),
                "iter={iter}: phase sum {sum:?} is less than 90% of iter_total {total:?}"
            );
        }
    }

    /// 実機（CUDA）依存の smoke テスト（coding-rust.md「実機依存テストは
    /// `#[ignore]` で分離」）。
    #[test]
    #[ignore]
    fn gemm_reuse_phases_cuda_smoke() {
        let out = temp_out_path("gemm-phases-cuda-smoke");
        let cli = Cli {
            task: "gemm".to_string(),
            device: "cuda".to_string(),
            size: 1024,
            out: out.to_string_lossy().into_owned(),
            mode: "reuse".to_string(),
            phases: true,
            tf32: false,
        };
        dispatch(&cli).expect("cuda gemm --mode reuse --phases smoke failed");
        let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&out);
        assert_eq!(content.lines().count(), 5, "content={content}");
        assert!(
            content.contains("\"phase\":\"iter_total\""),
            "content={content}"
        );
    }

    /// 実機（Metal）依存の smoke テスト。macOS のみコンパイル対象。
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore]
    fn gemm_reuse_phases_metal_smoke() {
        let out = temp_out_path("gemm-phases-metal-smoke");
        let cli = Cli {
            task: "gemm".to_string(),
            device: "metal".to_string(),
            size: 1024,
            out: out.to_string_lossy().into_owned(),
            mode: "reuse".to_string(),
            phases: true,
            tf32: false,
        };
        dispatch(&cli).expect("metal gemm --mode reuse --phases smoke failed");
        let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&out);
        assert_eq!(content.lines().count(), 5, "content={content}");
        assert!(
            content.contains("\"phase\":\"iter_total\""),
            "content={content}"
        );
    }

    /// イシュー #1042: `bench-fandhe` は `fandhe-ai =0.4.0` に完全固定
    /// されており本イシューの新 API を呼べないため、`--tf32` は task/mode
    /// の組合せに関わらず常に MEASURE_ERROR で fail-fast する
    /// （`docs/cuda-tf32-optin-api-decision.md` C-1）。
    #[test]
    fn tf32_flag_is_always_measure_error() {
        for (task, mode) in [("gemm", "fresh"), ("gemm", "reuse"), ("train", "fresh")] {
            let out = temp_out_path(&format!("tf32-unsupported-{task}-{mode}"));
            let cli = Cli {
                task: task.to_string(),
                device: "cuda".to_string(),
                size: 64,
                out: out.to_string_lossy().into_owned(),
                mode: mode.to_string(),
                phases: false,
                tf32: true,
            };
            let err = dispatch(&cli).expect_err("--tf32 must be rejected on bench-fandhe");
            let msg = err.to_string();
            assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
            assert!(msg.contains("--tf32"), "msg={msg}");
            assert!(msg.contains("0.5.0"), "msg={msg}");
        }
    }

    /// 実機（CUDA）依存の smoke テスト（coding-rust.md「実機依存テストは
    /// `#[ignore]` で分離」）。fresh/reuse とも行数・`step_total` の存在の
    /// みを確認する（数値そのものの妥当性は cpu 側の回帰テストが担う）。
    #[test]
    #[ignore]
    fn train_phases_cuda_smoke() {
        for mode in ["fresh", "reuse"] {
            let out = temp_out_path(&format!("phases-cuda-smoke-{mode}"));
            let cli = Cli {
                task: "train".to_string(),
                device: "cuda".to_string(),
                size: 64,
                out: out.to_string_lossy().into_owned(),
                mode: mode.to_string(),
                phases: true,
                tf32: false,
            };
            dispatch(&cli).expect("cuda train --phases smoke failed");
            let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
            let _ = std::fs::remove_file(&out);
            assert!(content.lines().count() > 1, "content={content}");
            assert!(
                content.contains("\"phase\":\"step_total\""),
                "content={content}"
            );
        }
    }

    /// 実機（Metal）依存の smoke テスト。macOS のみコンパイル対象
    /// （coding-rust.md「実機依存テストは `#[ignore]` で分離」）。
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore]
    fn train_phases_metal_smoke() {
        for mode in ["fresh", "reuse"] {
            let out = temp_out_path(&format!("phases-metal-smoke-{mode}"));
            let cli = Cli {
                task: "train".to_string(),
                device: "metal".to_string(),
                size: 64,
                out: out.to_string_lossy().into_owned(),
                mode: mode.to_string(),
                phases: true,
                tf32: false,
            };
            dispatch(&cli).expect("metal train --phases smoke failed");
            let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
            let _ = std::fs::remove_file(&out);
            assert!(content.lines().count() > 1, "content={content}");
            assert!(
                content.contains("\"phase\":\"step_total\""),
                "content={content}"
            );
        }
    }
}
