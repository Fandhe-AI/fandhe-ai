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
//!
//! 借用ビュー readout（CPU/CUDA 既定経路・Metal は runtime legacy 維持。
//! イシュー #1337/#1437/#1438・codex-review 指摘 PR #1452 P2）: 上記
//! `readout_var` の展開（`to_tensor` 区間 + `host_copy` 区間）は、
//! #1182 §6/§9 が確定した「`host_copy`（`.to_vec()` の memcpy）が
//! `iter_total` の 25.6〜53.0%（CUDA）を占める」の対策として、`readout_var`
//! 自体を借用ビュー（`fandhe_ai::VarHostView`／`Tensor::host_slice`。
//! #1335・#1336）へ切り替えたものを CPU/CUDA の既定経路とする（#1438 で
//! 旧計測専用 cargo feature（#1337 導入）を撤去し常時有効化。#1436/#1437
//! が CUDA D2H 宛先未タッチによる N=1024/2048 の後退を診断・是正し、
//! #1438 で全 N 非後退を確認したうえで既定化した）。`to_tensor` 区間は
//! `Var::host_view()`（contiguous なら `Arc` 複製のみ）、`host_copy` 区間
//! は借用スライス取得（追加コピーなし・想定 ≈0）。
//!
//! **Metal は既定経路に含めない**: `docs/perf/metal-gemm-candle-gate-
//! remeasurement.md` §13.5 が「負荷変動と readout 切替の効果が分離
//! できておらず、ADOPT 判定は暫定の参考結果に留める」と明記しており、
//! `readout_uses_borrowed_view`（本ファイル。`device == "metal"` で
//! `false`）が runtime 判定で legacy 経路（`to_tensor()` + `.to_vec()`）
//! を維持する。cargo feature（#1337 導入・#1438 で撤去）を再導入する
//! ものではなく、`device` 文字列 1 個の比較に閉じた分岐である。
//!
//! checksum（全要素 `f64` 逐次和）・parity（`GemmReference::verify`）の
//! 契約・計算式はいずれの分岐でも不変（呼び出し側は両型とも
//! `Deref<Target=[f32]>` のため無変更）。crates.io 公開版
//! `fandhe-ai =0.7.0` には該当 API が未収録のため、ピン更新までは
//! `crates/facade` への path patch が CLI `--config` 経由で必須
//! （`bench_fandhe_pin_guard.sh` が未指定時に early-exit で検知する。
//! README「借用ビュー readout（既定経路）」節参照）。

use bench_common::*;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, SgdConfig, Tape, Tensor};
use std::time::{Duration, Instant};

const FRAMEWORK: &str = "fandhe-ai";
const VERSION: &str = "0.7.0";

/// イシュー #1350: `--graph` が渡された `--task train` 計測の record
/// 生成時に 1 回だけ呼び、launch 固定費の診断カウンタ（`fandhe_ai::
/// cuda_graph_step_stats()`）を `bench_common::GraphStepStatsRecord`
/// （facade 型を直接持たない bench-common 側の薄いコピー。`Record::
/// graph_stats` 参照）へ詰め替える。呼び出し自体は計測ループの外
/// （各 `run_train*` 関数の末尾、`Record` 組み立て直前）でのみ行うため
/// 計測時間には計上されない。`cli.graph` が `None`（off 計測）のときは
/// 呼び出し元が呼ばない契約（`graph-step` feature 無効ビルドでは
/// `fandhe_ai::cuda_graph_step_stats` 自体が存在しないため、この関数
/// 自体を `cfg(feature = "graph-step")` の外からは呼べない）。
#[cfg(feature = "graph-step")]
fn current_graph_stats() -> GraphStepStatsRecord {
    let s = fandhe_ai::cuda_graph_step_stats();
    GraphStepStatsRecord {
        captured: s.captured,
        replayed: s.replayed,
        graph_launches: s.graph_launches,
        sgd_kernel_launches: s.sgd_kernel_launches,
    }
}

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
// おり、fandhe-ai 0.7.0 の公開 API（`Var::matmul`）ではこれ以上分離
// できない（内訳は CUDA／Metal／CPU 各バックエンドの
// `gemm_reuse_phase_diag_tests` が別途取る。`docs/perf/
// cuda-gemm-reuse-phase-breakdown.md`・`docs/perf/
// metal-gemm-reuse-phase-breakdown.md` 参照。CPU はイシュー #1290
// で診断テストを追加し実測は #1292 へ引き継ぐ）。
const PHASE_GEMM_MATMUL: &str = "matmul";
const PHASE_GEMM_TO_TENSOR: &str = "to_tensor";
const PHASE_GEMM_HOST_COPY: &str = "host_copy";
const PHASE_GEMM_CHECKSUM: &str = "checksum";
const PHASE_GEMM_ITER_TOTAL: &str = "iter_total";

// `infer --phases`（イシュー #1217）の区間定数。CPU fresh（`predict`）・
// GPU fresh（`leaf_register`/`forward`。上の `PHASE_LEAF_REGISTER`/
// `PHASE_FORWARD` を再利用）・reuse（`predict_resident`）で公開 API 呼び
// 出しの粒度が異なるため、それぞれ 1 回の呼び出しに対応する区間を持つ。
// `to_tensor`/`host_copy`/`checksum`/`iter_total` は `gemm --mode reuse
// --phases`（イシュー #1182）の `readout_var` 展開と同一の意味（`Var`/
// `Tensor` のホスト実体化 → コピー → 総和）であり、上の `PHASE_GEMM_*`
// 定数をそのまま再利用する（task が異なる JSONL 行に同じ文字列値が乗る
// だけで、`phase` は task ごとの名前空間ではなく区間の意味を表す値の
// ため問題ない）。README「`infer --mode reuse` / `infer --phases`」節
// 参照。
const PHASE_INFER_PREDICT: &str = "predict";
const PHASE_INFER_PREDICT_RESIDENT: &str = "predict_resident";

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

/// Metal 限定の runtime legacy フォールバック判定（codex-review 指摘・
/// PR #1452 P2）。
///
/// `docs/perf/metal-gemm-candle-gate-remeasurement.md` §13.5 は「負荷
/// 変動と借用ビュー readout 切替の効果が分離できておらず、Metal の
/// ADOPT 判定は暫定の参考結果に留め、`runtime Device::Metal` 限定の
/// legacy フォールバックの要否は再計測後の判断とする」と明記している。
/// 本関数はその「要否判断」自体を、再計測が揃うまでの間 fail-closed に
/// 「Metal は legacy を維持する」へ倒したものであり、CPU/CUDA は #1337/
/// #1438 で確定した ADOPT（借用ビュー既定）のまま変えない。
///
/// #1438 が撤去したのはコンパイル時 cargo feature（`host-view-readout`）
/// のみであり、本関数はその cargo feature を再導入しない runtime 分岐
/// （`device` 文字列 1 個の比較）である——`readout_var`／`checksum_var`／
/// `checksum_tensor`／`measure_gemm_reuse_phases`／`measure_infer_phases`
/// （codex-review 指摘・PR #1452 P2 でインライン展開を 3 箇所追加）の
/// インライン展開すべてがこの 1 箇所を経由するため、Metal の既定経路
/// （gemm・infer 本体計測と `--phases` 診断）が内部で矛盾しない。Metal
/// の ADOPT が確定したら、この関数を `true` 固定に変更する（または
/// 呼び出し側から分岐ごと削除する）1 箇所の変更で足りるよう集約して
/// ある。
fn readout_uses_borrowed_view(device: &str) -> bool {
    device != "metal"
}

/// `readout_var`（`gemm --mode reuse` の `host_copy` 区間。#1182 §6/§9）
/// の戻り型。CPU/CUDA（[`readout_uses_borrowed_view`] が `true`）では
/// `fandhe_ai::VarHostView`（`Deref<Target=[f32]>`。#1335）を返し、
/// `contiguous()` が `Arc` 複製のみで済む形状では memcpy を伴わない借用
/// 読み出しになる（`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §6/§9
/// の (ii) を「checksum／parity 契約を一切変えず host_copy 側のみを
/// 実現する」形で満たす。イシュー #1337・#1438 で既定経路化）。Metal
/// では旧 legacy 経路（`to_tensor()` + `.to_vec()`。所有 `Vec<f32>`）を
/// 維持する（`readout_uses_borrowed_view` doc 参照）。呼び出し側
/// （`out.iter()`・`GemmReference::verify(&out)` の deref 強制）はいずれ
/// の分岐でも無変更で成立する。
///
/// crates.io 公開版 `fandhe-ai =0.7.0` には `VarHostView`／
/// `Var::host_view` が未収録（#1335 は未リリースの HEAD で追加）のため、
/// ピン未更新の間は `crates/facade` への path patch（CLI `--config`）が
/// 必須（`bench_fandhe_pin_guard.sh` が未指定時に registry ビルドを
/// fail-closed で拒否する。#1438 で既定経路化・旧計測専用 cargo
/// feature（#1337 導入）は撤去済み。Metal の legacy 分岐は runtime 判定
/// のためこの制約と無関係に成立する）。
enum HostReadout {
    Borrowed(fandhe_ai::VarHostView),
    Owned(Vec<f32>),
}

impl std::ops::Deref for HostReadout {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        match self {
            HostReadout::Borrowed(view) => view,
            HostReadout::Owned(vec) => vec.as_slice(),
        }
    }
}

/// Host-materialize a Var result and return a checksum (forces sync).
///
/// CPU/CUDA は借用ビュー（`Var::host_view()`）に対して `f64` 逐次和を
/// 計算する（#1337・#1438 で既定経路化）。Metal は
/// [`readout_uses_borrowed_view`] により legacy 経路（`to_tensor()` +
/// `.to_vec()`）を維持する。要素順序・型・和の計算式はいずれの分岐でも
/// 不変（`readout_var_matches_legacy_to_vec_bit_exact` で bit 同一を
/// 確認）。
fn checksum_var(v: &fandhe_ai::Var, device: &str) -> Result<f64, Box<dyn std::error::Error>> {
    if readout_uses_borrowed_view(device) {
        let view = v.host_view();
        Ok(view.iter().map(|&x| x as f64).sum())
    } else {
        let t = v.to_tensor();
        let slice = t
            .contiguous()
            .as_slice()
            .ok_or("as_slice() returned None after contiguous()")?
            .to_vec();
        Ok(slice.iter().map(|&x| x as f64).sum())
    }
}

/// Host-materialize a Var result and return the raw elements (forces sync).
/// `run_gemm`/`run_gemm_reuse`（イシュー #970）は checksum（全要素和）に
/// 加え要素単位の参照比較（`GemmReference::verify`）が必要なため、
/// `checksum_var` とは別に [`HostReadout`] を返す readout を用意する
/// （`checksum_var`/`checksum_tensor` は `run_infer` が引き続き使うため
/// シグネチャを変更しない）。
///
/// イシュー #1337・#1438: CPU/CUDA は `Var::to_tensor()` + `.to_vec()`
/// （旧 legacy 経路の `host_copy` memcpy）の代わりに `Var::host_view()`
/// （`contiguous()` が `Arc` 複製のみで済む場合は memcpy を伴わない）を
/// 返す。Metal は [`readout_uses_borrowed_view`]（codex-review 指摘・
/// PR #1452 P2）により legacy 経路を維持する。戻り値
/// （[`HostReadout`]）はいずれの分岐も `Deref<Target=[f32]>` のため、
/// 呼び出し側（`run_gemm`/`run_gemm_reuse`/`measure_gemm_reuse_phases`
/// の `out.iter()`・`reference.verify(&out)`）は無変更で成立する。
fn readout_var(
    v: &fandhe_ai::Var,
    device: &str,
) -> Result<HostReadout, Box<dyn std::error::Error>> {
    if readout_uses_borrowed_view(device) {
        Ok(HostReadout::Borrowed(v.host_view()))
    } else {
        let t = v.to_tensor();
        let owned = t
            .contiguous()
            .as_slice()
            .ok_or("as_slice() returned None after contiguous()")?
            .to_vec();
        Ok(HostReadout::Owned(owned))
    }
}

/// CPU/CUDA は [`fandhe_ai_tensor_core::Tensor::host_slice`]（contiguous
/// なら借用 `Cow::Borrowed`・非 contiguous のみ実体化して `Cow::Owned`。
/// #1335）に対して `f64` 逐次和を計算する（#1337・#1438 で既定経路化）。
/// Metal は [`readout_uses_borrowed_view`] により legacy 経路
/// （`contiguous().as_slice().to_vec()`）を維持する。
fn checksum_tensor(t: &Tensor<f32>, device: &str) -> Result<f64, Box<dyn std::error::Error>> {
    if readout_uses_borrowed_view(device) {
        let slice = t.host_slice();
        Ok(slice.iter().map(|&x| x as f64).sum())
    } else {
        let slice = t
            .contiguous()
            .as_slice()
            .ok_or("as_slice() returned None after contiguous()")?
            .to_vec();
        Ok(slice.iter().map(|&x| x as f64).sum())
    }
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

    // イシュー #1339: `--device-checksum` は fresh モードでも計測窓を
    // 「tape 構築 → 葉登録 → `matmul_checksum(ChecksumOnly)`」へ置換する
    // （`device-checksum` feature 有効時のみ到達。`dispatch` が feature
    // 無効ビルドでは事前に MEASURE_ERROR にする）。
    #[cfg(feature = "device-checksum")]
    if cli.device_checksum {
        return run_gemm_device_checksum(cli, &a_data, &b_data, &reference);
    }

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
        let out = readout_var(&c, &cli.device)?;
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
        let stats = reference.verify_strict(&out)?;
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
        managed: cli.managed,
        device_checksum: cli.device_checksum,
        // イシュー #1350: `--graph` は `--task train` 限定（dispatch の
        // ゲート参照）のため gemm 計測では常に None。
        graph: None,
        graph_stats: None,
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

    // イシュー #1339: reuse モードも `--device-checksum` で init_s の定義を
    // 「tape 構築 + 葉登録 + 初回 `matmul_checksum` + 8 バイト読み戻し」へ
    // 置換する（`device-checksum` feature 有効時のみ到達）。
    #[cfg(feature = "device-checksum")]
    if cli.device_checksum {
        return run_gemm_reuse_device_checksum(cli, &a_data, &b_data, &reference);
    }

    // init_s: tape 構築 + 葉 Var 登録 + 初回 matmul + ホスト実体化までの
    // 経過（CUDA コンテキスト作成・NVRTC コンパイル等の一度きりのコストを
    // すべて含む）。
    let init_start = Instant::now();
    let tape = make_tape(&cli.device)?;
    let a = tape.var(&a_data);
    let b = tape.var(&b_data);
    let c0 = a.matmul(&b)?;
    let out0 = readout_var(&c0, &cli.device)?;
    let mut checksum: f64 = out0.iter().map(|&x| x as f64).sum();
    let init_s = init_start.elapsed().as_secs_f64();
    // イシュー #965 codex-review 指摘: checksum は毎反復上書きされるため、
    // ループ後に最後の値だけを検査すると途中反復の縮退を見逃す。init 計測分
    // を含め reuse 経路（同一 tape を使い回す）でも fresh 経路と同様に
    // checksum 計算直後・全反復で検証する。
    validate_gemm_checksum(checksum)?;
    // イシュー #970: init 計測分の要素単位検証は init_s の外（elapsed 取得後）
    // で行う。以後の反復と worst-case で集約する。
    let mut parity = reference.verify_strict(&out0)?;

    // 残り warmup（1 回は init 計測内で消費済み）+ 計測本体。同一 tape・同一
    // 葉 Var を使い回し、matmul のみを繰り返す。
    let mut one = || -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let c = a.matmul(&b)?;
        let out = readout_var(&c, &cli.device)?;
        checksum = out.iter().map(|&x| x as f64).sum();
        let elapsed = start.elapsed();
        validate_gemm_checksum(checksum)?;
        parity = parity.worst(reference.verify_strict(&out)?);
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
        managed: cli.managed,
        device_checksum: cli.device_checksum,
        // イシュー #1350: gemm は `--graph` 対象外（dispatch のゲート
        // 参照）。
        graph: None,
        graph_stats: None,
    }
    .emit(&cli.out)?;
    Ok(())
}

/// `run_gemm` の `--device-checksum` 分岐（イシュー #1339。`device-checksum`
/// feature 有効時のみコンパイルされる）。計測窓を「fresh tape 構築 → 葉
/// 登録 → `matmul_checksum(ChecksumOnly)`」へ置換し、GPU バックエンドでは
/// `C` のホスト download を毎反復回避する契約（`BackendOps::gemm_checksum`
/// doc 参照。CUDA／Metal は本イシュー時点で未実装のため
/// `BackendError::Unsupported` がそのまま伝播し MEASURE_ERROR になる）。
/// ループ後の未計時 1 反復（`ChecksumReadout::WithOutput`）で要素単位
/// parity（`GemmReference::verify`）と、決定的カーネルであれば成り立つ
/// はずの「末尾 checksum と最終計時反復の checksum の bit 一致」を検証
/// する（AC-2）。
#[cfg(feature = "device-checksum")]
fn run_gemm_device_checksum(
    cli: &Cli,
    a_data: &Tensor<f32>,
    b_data: &Tensor<f32>,
    reference: &GemmReference,
) -> Result<(), Box<dyn std::error::Error>> {
    use fandhe_ai::ChecksumReadout;
    let n = cli.size;
    let mut checksum = 0.0;
    let one = |sync_checksum: &mut f64| -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let tape = make_tape(&cli.device)?;
        let a = tape.var(a_data);
        let b = tape.var(b_data);
        let result = a.matmul_checksum(&b, ChecksumReadout::ChecksumOnly)?;
        *sync_checksum = result.checksum;
        let elapsed = start.elapsed();
        validate_gemm_checksum(*sync_checksum)?;
        Ok(elapsed)
    };
    for _ in 0..WARMUP_ITERS {
        one(&mut checksum)?;
    }
    let mut durations = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        durations.push(one(&mut checksum)?);
    }
    let tape = make_tape(&cli.device)?;
    let a = tape.var(a_data);
    let b = tape.var(b_data);
    let tail = a.matmul_checksum(&b, ChecksumReadout::WithOutput)?;
    let output = tail
        .output
        .ok_or("MEASURE_ERROR: matmul_checksum(WithOutput) returned output=None (issue #1339)")?;
    let out = output.as_slice().ok_or(
        "MEASURE_ERROR: matmul_checksum(WithOutput).output.as_slice() returned None \
         (issue #1339)",
    )?;
    let parity = reference.verify_strict(out)?;
    if tail.checksum.to_bits() != checksum.to_bits() {
        return Err(format!(
            "MEASURE_ERROR: tail matmul_checksum ({}) is not bit-identical to the last timed \
             iteration's checksum ({}); GEMM kernel selection may be non-deterministic \
             (issue #1339)",
            tail.checksum, checksum
        )
        .into());
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
        parity: Some(parity),
        tf32: false,
        managed: cli.managed,
        device_checksum: true,
        graph: None,
        graph_stats: None,
    }
    .emit(&cli.out)?;
    Ok(())
}

/// `run_gemm_reuse` の `--device-checksum` 分岐（イシュー #1339。
/// [`run_gemm_device_checksum`] の reuse 版）。`init_s` は「tape 構築 +
/// 葉登録 + 初回 `matmul_checksum` + 8 バイト読み戻し」までの経過時間
/// と再定義する（README「計測プロトコル」節に明記する契約差分）。
#[cfg(feature = "device-checksum")]
fn run_gemm_reuse_device_checksum(
    cli: &Cli,
    a_data: &Tensor<f32>,
    b_data: &Tensor<f32>,
    reference: &GemmReference,
) -> Result<(), Box<dyn std::error::Error>> {
    use fandhe_ai::ChecksumReadout;
    let n = cli.size;
    let init_start = Instant::now();
    let tape = make_tape(&cli.device)?;
    let a = tape.var(a_data);
    let b = tape.var(b_data);
    let r0 = a.matmul_checksum(&b, ChecksumReadout::ChecksumOnly)?;
    let mut checksum = r0.checksum;
    let init_s = init_start.elapsed().as_secs_f64();
    validate_gemm_checksum(checksum)?;

    let mut one = || -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let result = a.matmul_checksum(&b, ChecksumReadout::ChecksumOnly)?;
        checksum = result.checksum;
        let elapsed = start.elapsed();
        validate_gemm_checksum(checksum)?;
        Ok(elapsed)
    };
    for _ in 0..WARMUP_ITERS.saturating_sub(1) {
        one()?;
    }
    let mut durations = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        durations.push(one()?);
    }
    let tail = a.matmul_checksum(&b, ChecksumReadout::WithOutput)?;
    let output = tail
        .output
        .ok_or("MEASURE_ERROR: matmul_checksum(WithOutput) returned output=None (issue #1339)")?;
    let out = output.as_slice().ok_or(
        "MEASURE_ERROR: matmul_checksum(WithOutput).output.as_slice() returned None \
         (issue #1339)",
    )?;
    let parity = reference.verify_strict(out)?;
    if tail.checksum.to_bits() != checksum.to_bits() {
        return Err(format!(
            "MEASURE_ERROR: tail matmul_checksum ({}) is not bit-identical to the last timed \
             iteration's checksum ({}); GEMM kernel selection may be non-deterministic \
             (issue #1339)",
            tail.checksum, checksum
        )
        .into());
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
        managed: cli.managed,
        device_checksum: true,
        graph: None,
        graph_stats: None,
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
    let out0 = readout_var(&c0, &cli.device)?;
    let mut checksum: f64 = out0.iter().map(|&x| x as f64).sum();
    let init_s = init_start.elapsed().as_secs_f64();
    validate_gemm_checksum(checksum)?;
    let mut parity = reference.verify_strict(&out0)?;

    let mut phases = PhaseSamples::new();
    // 1 回は init 計測内で消費済み（`run_gemm_reuse` と同じ warmup 消費
    // 規約）。
    let total_iters = WARMUP_ITERS.saturating_sub(1) + MEASURE_ITERS;
    for _ in 0..total_iters {
        let iter_start = Instant::now();

        let t0 = Instant::now();
        let c = a.matmul(&b)?;
        phases.push(PHASE_GEMM_MATMUL, t0.elapsed());

        // イシュー #1337・#1438: `to_tensor`／`host_copy` の 2 区間は
        // `readout_var` の実装（`readout_uses_borrowed_view` 分岐）を
        // ここで展開したもの（module doc・README「`gemm --mode reuse
        // --phases`」節参照）。区間名・順序・件数（5 区間）はいずれの
        // 分岐でも不変。Metal は `readout_var` と同じく
        // [`readout_uses_borrowed_view`]（codex-review 指摘・PR #1452
        // P2）により legacy 経路を維持し、`--phases` 診断と gemm 本体
        // 計測が同じ分岐判定を共有する（内部矛盾を避ける）。
        let out = if readout_uses_borrowed_view(&cli.device) {
            // `to_tensor` 区間: `Var::host_view()` の構築
            // （`materialize_non_fallible(..).contiguous()`。contiguous
            // な場合は `Tensor` 内部 `Arc` の複製のみで memcpy を伴わない
            // ——旧 `to_tensor` 区間が担っていた「`Tape` 借用からの
            // 実体化」に対応する区間として維持する）。
            let t0 = Instant::now();
            let view = c.host_view();
            phases.push(PHASE_GEMM_TO_TENSOR, t0.elapsed());

            // `host_copy` 区間: 借用スライスの取得（`Deref::deref`。
            // 追加コピーなし・旧区間の memcpy が消える想定値 ≈0。
            // #1182 §6/§9 が確定した「host_copy が主因」を打ち消す本
            // 変更の中心区間）。
            let t0 = Instant::now();
            let out: &[f32] = &view;
            phases.push(PHASE_GEMM_HOST_COPY, t0.elapsed());
            let _ = out;
            HostReadout::Borrowed(view)
        } else {
            // `to_tensor` 区間: `Var::to_tensor()`（`Tape` の `RefCell`
            // 借用から `Tensor<f32>` へ複製）。
            let t0 = Instant::now();
            let t = c.to_tensor();
            phases.push(PHASE_GEMM_TO_TENSOR, t0.elapsed());

            // `host_copy` 区間: `contiguous().as_slice().to_vec()`
            // （旧 legacy 経路の memcpy 本体）。
            let t0 = Instant::now();
            let owned = t
                .contiguous()
                .as_slice()
                .ok_or("as_slice() returned None after contiguous()")?
                .to_vec();
            phases.push(PHASE_GEMM_HOST_COPY, t0.elapsed());
            HostReadout::Owned(owned)
        };

        let t0 = Instant::now();
        checksum = out.iter().map(|&x| x as f64).sum();
        phases.push(PHASE_GEMM_CHECKSUM, t0.elapsed());

        // `run_gemm_reuse` と同じく、elapsed の計測窓はここで閉じる
        // （checksum 計算までを計時対象とし、以降の検証コストは含めない）。
        phases.push(PHASE_GEMM_ITER_TOTAL, iter_start.elapsed());

        validate_gemm_checksum(checksum)?;
        parity = parity.worst(reference.verify_strict(&out)?);
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
                managed: cli.managed,
                device_checksum: false,
                graph: None,
                graph_stats: None,
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
        managed: cli.managed,
        device_checksum: false,
        // イシュー #1350: `run_train`（fresh）は毎 step ホスト側で
        // `p - lr*g` を計算する経路で `DeviceParamStore::step`（capture
        // 対象の update 区間）に到達しないが、`--graph` 値自体は対照
        // 計測（design doc §9・実装計画 (c)）として記録する。
        // `graph_stats` はプロセスワイド累積カウンタのため、この経路で
        // 実際に capture／replay されていなければ 0 のまま記録される
        // （実際に到達するかどうかも本計測の観測対象）。
        graph: cli.graph.as_deref(),
        #[cfg(feature = "graph-step")]
        graph_stats: cli.graph.as_ref().map(|_| current_graph_stats()),
        #[cfg(not(feature = "graph-step"))]
        graph_stats: None,
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
        managed: cli.managed,
        device_checksum: false,
        // イシュー #1350: `run_train_reuse` は `DeviceParamStore::step`
        // （capture 対象の update 区間。`sgd_step_device_tracked`）へ
        // 到達する主対象（実装計画 (a)(b)）。
        graph: cli.graph.as_deref(),
        #[cfg(feature = "graph-step")]
        graph_stats: cli.graph.as_ref().map(|_| current_graph_stats()),
        #[cfg(not(feature = "graph-step"))]
        graph_stats: None,
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
                managed: cli.managed,
                device_checksum: false,
                // イシュー #1350: `mode="reuse"` の `device_update` 行が
                // 主対象（実装計画 (b)）。`fresh` は `DeviceParamStore::
                // step` 非到達の対照として `graph` 値のみ記録する。
                graph: cli.graph.as_deref(),
                #[cfg(feature = "graph-step")]
                graph_stats: cli.graph.as_ref().map(|_| current_graph_stats()),
                #[cfg(not(feature = "graph-step"))]
                graph_stats: None,
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
                *sync_checksum = checksum_tensor(&out, &cli.device)?;
                Ok(start.elapsed())
            }
            _ => {
                // explicit tape on the requested device + forward + host sync
                let tape = make_tape(&cli.device)?;
                let start = Instant::now();
                let x = tape.var(&x_data);
                let out = model.forward(&tape, &x)?;
                *sync_checksum = checksum_var(&out, &cli.device)?;
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
        managed: cli.managed,
        device_checksum: false,
        // イシュー #1350: infer は `--graph` 対象外（dispatch のゲート
        // 参照。`DeviceParamStore::step` 自体を呼ばないタスク）。
        graph: None,
        graph_stats: None,
    }
    .emit(&cli.out)?;
    Ok(())
}

/// `infer --mode reuse`（イシュー #1217）。`train --mode reuse`（イシュー
/// #958）の「デバイス常駐パラメータで初期化コストとカーネル実行を分離
/// する」考えを推論へ適用する。`Sequential::predict_resident`（0.6.0 で
/// 公開済み。facade 経由の `DeviceParamStore` 常駐重み forward）を使い、
/// `init_s` を「1 回限りの H2D upload + その完了保証の同期」として
/// `run_train_reuse` と同一定義で分離記録する。
///
/// `run_train_reuse` の `init_s` コメント（本ファイル該当箇所）が示す
/// とおり、`sync_device_param_store_to_host` を使うのは「ホスト転送を
/// 伴わない完了待ち」が公開 API 面に無いためであり、同じギャップが
/// ここにも及ぶ（重複を避けるためコメントは反復しない）。
///
/// tape は `predict_resident` 内部で毎呼び出し生成・破棄される
/// （`Sequential::predict_resident` doc「`Tape` はこの呼び出しのスコープ
/// 内で破棄される」参照）ため、`run_train_reuse` と異なり reuse できる
/// のは `DeviceParamStore` のみで warmup を init 側で消費する必要はなく、
/// `run_infer`（fresh）と同じ `WARMUP_ITERS`/`MEASURE_ITERS` の反復数を
/// そのまま使う。
fn run_infer_reuse(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let model = build_model()?;
    let (x_data, _) = mlp_data()?;

    let init_start = Instant::now();
    let init_tape = make_tape(&cli.device)?;
    let store = model.init_device_param_store(&init_tape)?;
    let _ = init_tape.sync_device_param_store_to_host(&store)?;
    drop(init_tape);
    let init_s = init_start.elapsed().as_secs_f64();

    let mut checksum = 0.0;
    let one = |sync_checksum: &mut f64| -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let out = model.predict_resident(&store, &x_data)?;
        *sync_checksum = checksum_tensor(&out, &cli.device)?;
        Ok(start.elapsed())
    };

    for _ in 0..WARMUP_ITERS {
        one(&mut checksum)?;
    }
    let mut durations = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        durations.push(one(&mut checksum)?);
    }

    if !checksum.is_finite() {
        return Err(format!("MEASURE_ERROR: final checksum not finite: {checksum}").into());
    }

    // 終端同期: `run_train_reuse`/`measure_train_reuse_phases` と同一の
    // 検証（A08: 破損した推論結果を性能値として残さない）。
    // `predict_resident` は `store` を `&DeviceParamStore` で読むのみで
    // 更新しないため、init 直後との差分は生じないはずだが、reuse する
    // store がここまでの呼び出しで壊れていないことを終端でも確認する。
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
        mode: "reuse",
        init_s: Some(init_s),
        parity: None,
        tf32: false,
        managed: cli.managed,
        device_checksum: false,
        // イシュー #1350: infer は `--graph` 対象外（上記 `run_infer`
        // 参照）。
        graph: None,
        graph_stats: None,
    }
    .emit(&cli.out)?;
    Ok(())
}

/// `infer --phases`（イシュー #1217）の計測本体。`mode`（`"fresh"`／
/// `"reuse"`）と `cli.device`（`"cpu"` か否か）の組合せで区間集合が
/// 異なる（README「`infer --mode reuse` / `infer --phases`」節の表）:
///
/// - fresh・cpu: `predict`／`host_copy`／`checksum`／`iter_total`
///   （`Sequential::predict` は公開 API 上単一呼び出しでこれ以上分解
///   できない）
/// - fresh・metal/cuda: `leaf_register`／`forward`／`to_tensor`／
///   `host_copy`／`checksum`／`iter_total`（`run_infer` の非 cpu 分岐と
///   同じく `make_tape` は計測窓の外。`readout_var` 相当の展開は
///   `measure_gemm_reuse_phases` と同型）
/// - reuse: `predict_resident`／`host_copy`／`checksum`／`iter_total`
///   （`predict_resident` 内部は private ヘルパー
///   `forward_from_flat_leaves` を経由し公開 API からは分解不能。
///   README に明記）
///
/// `run_infer`/`run_infer_reuse` 本体は変更しない（AC-2 と同じ方針。
/// `measure_gemm_reuse_phases` が `run_gemm_reuse` を変更しないのと同型）。
fn measure_infer_phases(
    cli: &Cli,
    mode: &str,
) -> Result<(PhaseSamples, f64, Option<f64>), Box<dyn std::error::Error>> {
    let model = build_model()?;
    let (x_data, _) = mlp_data()?;
    let mut phases = PhaseSamples::new();
    let mut checksum = 0.0f64;
    let iters = WARMUP_ITERS + MEASURE_ITERS;

    let init_s = if mode == "reuse" {
        let init_start = Instant::now();
        let init_tape = make_tape(&cli.device)?;
        let store = model.init_device_param_store(&init_tape)?;
        let _ = init_tape.sync_device_param_store_to_host(&store)?;
        drop(init_tape);
        let init_s = init_start.elapsed().as_secs_f64();

        for _ in 0..iters {
            let iter_start = Instant::now();

            let t0 = Instant::now();
            let out = model.predict_resident(&store, &x_data)?;
            phases.push(PHASE_INFER_PREDICT_RESIDENT, t0.elapsed());

            // イシュー #1438 codex-review 指摘（PR #1452 P2）:
            // 通常推論（`run_infer_reuse`）が既定化した借用ビュー
            // 読み出し（[`readout_uses_borrowed_view`]）を `--phases`
            // 診断側にも揃える。CPU/CUDA は `Tensor::host_slice()`
            // （contiguous なら memcpy なしの `Cow::Borrowed`）・Metal は
            // legacy 経路（`contiguous().as_slice().to_vec()`）を維持し、
            // gemm 側の分岐判定と矛盾しないようにする。
            let t0 = Instant::now();
            let host_slice: std::borrow::Cow<'_, [f32]> = if readout_uses_borrowed_view(&cli.device)
            {
                out.host_slice()
            } else {
                std::borrow::Cow::Owned(
                    out.contiguous()
                        .as_slice()
                        .ok_or("predict_resident output as_slice() returned None")?
                        .to_vec(),
                )
            };
            phases.push(PHASE_GEMM_HOST_COPY, t0.elapsed());

            let t0 = Instant::now();
            checksum = host_slice.iter().map(|&x| x as f64).sum();
            phases.push(PHASE_GEMM_CHECKSUM, t0.elapsed());

            phases.push(PHASE_GEMM_ITER_TOTAL, iter_start.elapsed());
        }

        // 終端同期（`run_infer_reuse` と同一の検証。A08）。
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
            let s = t
                .contiguous()
                .as_slice()
                .ok_or("synced param as_slice() returned None")?
                .to_vec();
            if s.iter().any(|v| !v.is_finite()) {
                return Err("MEASURE_ERROR: synced parameter contains non-finite element".into());
            }
        }
        Some(init_s)
    } else if cli.device == "cpu" {
        for _ in 0..iters {
            let iter_start = Instant::now();

            let t0 = Instant::now();
            let out = model.predict(&x_data)?;
            phases.push(PHASE_INFER_PREDICT, t0.elapsed());

            // イシュー #1438 codex-review 指摘（PR #1452 P2）: 上の
            // reuse 分岐と同じく [`readout_uses_borrowed_view`] を適用
            // する（module doc 参照）。
            let t0 = Instant::now();
            let host_slice: std::borrow::Cow<'_, [f32]> = if readout_uses_borrowed_view(&cli.device)
            {
                out.host_slice()
            } else {
                std::borrow::Cow::Owned(
                    out.contiguous()
                        .as_slice()
                        .ok_or("predict output as_slice() returned None")?
                        .to_vec(),
                )
            };
            phases.push(PHASE_GEMM_HOST_COPY, t0.elapsed());

            let t0 = Instant::now();
            checksum = host_slice.iter().map(|&x| x as f64).sum();
            phases.push(PHASE_GEMM_CHECKSUM, t0.elapsed());

            phases.push(PHASE_GEMM_ITER_TOTAL, iter_start.elapsed());
        }
        None
    } else {
        for _ in 0..iters {
            // `run_infer` の非 cpu 分岐と同じく tape 構築は計測窓の外
            // （D4。`make_tape` は fresh tape のホスト側構築で GPU
            // カーネル実行を伴わないが、`run_infer` の既存プロトコル・
            // 既存 (c) 数値の前提を変えないため踏襲する）。
            let tape = make_tape(&cli.device)?;
            let iter_start = Instant::now();

            let t0 = Instant::now();
            let x = tape.var(&x_data);
            phases.push(PHASE_LEAF_REGISTER, t0.elapsed());

            let t0 = Instant::now();
            let out = model.forward(&tape, &x)?;
            phases.push(PHASE_FORWARD, t0.elapsed());

            let t0 = Instant::now();
            let t = out.to_tensor();
            phases.push(PHASE_GEMM_TO_TENSOR, t0.elapsed());

            // イシュー #1438 codex-review 指摘（PR #1452 P2）: 上の 2
            // 分岐と同じく [`readout_uses_borrowed_view`] を適用する
            // （module doc 参照。GPU fresh 経路も `to_tensor` 後の
            // `host_copy` 区間は同一判定を共有する）。
            let t0 = Instant::now();
            let host_slice: std::borrow::Cow<'_, [f32]> = if readout_uses_borrowed_view(&cli.device)
            {
                t.host_slice()
            } else {
                std::borrow::Cow::Owned(
                    t.contiguous()
                        .as_slice()
                        .ok_or("as_slice() returned None after contiguous()")?
                        .to_vec(),
                )
            };
            phases.push(PHASE_GEMM_HOST_COPY, t0.elapsed());

            let t0 = Instant::now();
            checksum = host_slice.iter().map(|&x| x as f64).sum();
            phases.push(PHASE_GEMM_CHECKSUM, t0.elapsed());

            phases.push(PHASE_GEMM_ITER_TOTAL, iter_start.elapsed());
        }
        None
    };

    if !checksum.is_finite() {
        return Err(format!("MEASURE_ERROR: final checksum not finite: {checksum}").into());
    }

    Ok((phases, checksum, init_s))
}

/// [`measure_infer_phases`] の結果を phase ごとに 1 行の JSONL
/// （`task:"infer_phases"`）として出力する。`emit_gemm_phase_records`
/// と同型（`--phases` 実行時は既存 `task:"infer"` 行を出さない）。
fn emit_infer_phase_records(
    cli: &Cli,
    phases: &PhaseSamples,
    mode: &'static str,
    checksum: f64,
    init_s: Option<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    for (phase_index, &phase) in phases.order.iter().enumerate() {
        let measured = &phases.durations(phase)[WARMUP_ITERS..];
        let st = stats(measured)?;
        PhaseRecord {
            base: Record {
                framework: FRAMEWORK,
                framework_version: VERSION,
                task: "infer_phases",
                device: &cli.device,
                size: BATCH,
                stats: st,
                gflops: None,
                throughput_per_s: None,
                checksum,
                warmup: WARMUP_ITERS,
                iters: measured.len(),
                mode,
                init_s,
                parity: None,
                tf32: false,
                managed: cli.managed,
                device_checksum: false,
                // イシュー #1350: infer は `--graph` 対象外（`run_infer`
                // 参照）。
                graph: None,
                graph_stats: None,
            },
            phase,
            phase_index,
        }
        .emit(&cli.out)?;
    }
    Ok(())
}

/// `--task infer --phases`（fresh/reuse。イシュー #1217）。
fn run_infer_phases(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let mode: &'static str = if cli.mode == "reuse" {
        "reuse"
    } else {
        "fresh"
    };
    let (phases, checksum, init_s) = measure_infer_phases(cli, mode)?;
    emit_infer_phase_records(cli, &phases, mode, checksum, init_s)
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

/// task × mode × `--phases` の分岐。reuse モードは gemm（イシュー #925
/// §2.1・§8）・train（イシュー #958）・infer（イシュー #1217）に対応する。
/// `--phases` は `--task train`（fresh/reuse 双方。イシュー #1009）・
/// `--task infer`（fresh/reuse 双方。イシュー #1217）・`--task gemm --mode
/// reuse`（イシュー #1182）に対応し、`gemm --mode fresh` との組合せのみ
/// MEASURE_ERROR とする。`run()` から分離してあるのは `parse_cli()`
/// （`std::env::args()` 依存）を経由せず `tests::
/// phases_with_gemm_fresh_is_measure_error` から直接分岐を検証できる
/// ようにするため。
///
/// **`--tf32`（イシュー #1042）は本バイナリでは常に MEASURE_ERROR で
/// fail-fast する**: `bench-fandhe` は crates.io 公開版 `fandhe-ai
/// =0.7.0` に完全固定されており（deps-policy 第 9 区分。
/// `check_framework_compare` が registry 取得元を fail-closed 検査する
/// ため path 依存への差し替えは不可）。`fandhe_ai::set_cuda_tf32_gemm_enabled`
/// 自体は crates.io 公開版から呼び出し可能になったが（承認ピンは v0.5.0 公開
/// 時点で `>= 0.5.0` を満たしている）、`bench-fandhe`（`main.rs`）側の
/// 呼び出し結線・`run_all` の tf32 スイープ追加（C-2。
/// `docs/cuda-tf32-optin-api-decision.md` 参照）は依然未実施のため
/// fail-fast する。`--phases` の対象外組合せ拒否と同型の allowlist 方式で、
/// `cli.phases`（`match` の第 3 要素）より先に検査する。
///
/// **`--managed`（イシュー #1353）**: CUDA managed memory 配置
/// （`fandhe_ai::set_cuda_managed_memory_enabled`。#1352）を有効化して
/// 計測する。`--tf32` と異なり crates.io 公開版 `fandhe-ai =0.7.0` には
/// 当該 API 自体が未収録（#1352 は本イシュー時点で未リリースの HEAD）
/// のため、`managed-placement` feature（既定無効）で呼び出しをコンパイル
/// 時に分離する: feature 無効時は API 呼び出しコード自体が存在せず
/// `=0.7.0` ピンのままビルド成立する契約を保つ。`device != "cuda"` は
/// プロセスワイドフラグが cpu 計測で無音 no-op になるのを防ぐため
/// fail-fast する（README「`--managed`」節参照）。
fn dispatch(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    if cli.tf32 {
        return Err(
            "MEASURE_ERROR: --tf32 requires fandhe-ai >= 0.5.0 (wiring not implemented in \
             bench-fandhe yet; see docs/cuda-tf32-optin-api-decision.md C-2; issue #1042)"
                .into(),
        );
    }
    if cli.managed {
        if cli.device != "cuda" {
            return Err(format!(
                "MEASURE_ERROR: --managed is only meaningful for --device cuda (got \
                 device='{}'; managed placement affects only the CUDA backend. issue #1353)",
                cli.device
            )
            .into());
        }
        #[cfg(feature = "managed-placement")]
        {
            fandhe_ai::set_cuda_managed_memory_enabled(true);
            if !fandhe_ai::cuda_managed_memory_enabled() {
                return Err(
                    "MEASURE_ERROR: set_cuda_managed_memory_enabled(true) did not take effect \
                     (cuda_managed_memory_enabled() returned false after enabling; issue #1353)"
                        .into(),
                );
            }
        }
        #[cfg(not(feature = "managed-placement"))]
        {
            return Err(
                "MEASURE_ERROR: --managed requires fandhe-ai >= 0.8.0 or a path-patched facade \
                 built with --features managed-placement (issue #1353; see \
                 scripts/bench/framework-compare/README.md \"--managed\" section)"
                    .into(),
            );
        }
    }
    // イシュー #1350: `--graph <on|stream-only>` は `--device cuda` かつ
    // `--task train` 限定（`DeviceParamStore::step` の update 区間のみが
    // capture 対象のため。gemm／infer は `sgd_step_device_tracked` に
    // 到達しない）。`bench-fandhe` は crates.io 公開版 `fandhe-ai =0.7.0`
    // ピンには `cuda_graph_step_mode`/`cuda_graph_step_stats` API 自体が
    // 未収録のため、`graph-step` feature（既定無効）でコンパイル時に
    // 分離する（`--managed`／`--device-checksum` と同型の allowlist
    // 方式）。
    //
    // **`on` と `stream-only` でゲートの形が異なる**（`fandhe_ai_backend_
    // cuda::graph` モジュール冒頭コメントの契約）:
    // - `on`: `fandhe_ai::set_cuda_graph_step_enabled(true)` を呼ぶ（API
    //   明示設定は環境変数より優先される）。**この呼び出しは本関数の
    //   これより前の分岐（`--tf32`／`--managed`／`--device-checksum`）が
    //   いずれも CUDA デバイスを初期化しないことに依存しており、かつ
    //   `dispatch` から呼ばれる `run_train*`（`make_tape`/`CudaDevice::
    //   new` を呼ぶ最初の箇所）より確実に前で実行される契約を、この
    //   関数自体がその境界であることで満たす**。
    // - `stream-only`: API setter を呼ぶと `EXPLICIT` フラグが立ち
    //   環境変数 `FANDHE_AI_CUDA_GRAPH_STEP` が以後無視される契約
    //   （`graph.rs` の `step_graph_mode` 優先順位）があるため、API は
    //   呼ばない。代わりに `cuda_graph_step_mode() ==
    //   CudaGraphStepMode::StreamOnly` を確認し、起動側が環境変数を
    //   export し忘れている場合は fail-closed で拒否する（「off 行が
    //   実は stream-only のまま計測される」silent failure を防ぐ）。
    // - `--graph` 未指定でも `cuda_graph_step_mode() != Off` なら
    //   拒否する（環境変数が誤って漏れ「off 行」が実は on/stream-only
    //   になる fail-open を遮断する。`run_ab_graph_cuda.sh` の交互起動
    //   ループが環境変数を意図せず引き継ぐ事故を機械的に検出する）。
    if let Some(graph_mode) = cli.graph.as_deref() {
        if cli.device != "cuda" || cli.task != "train" {
            return Err(format!(
                "MEASURE_ERROR: --graph is only meaningful for --device cuda --task train (got \
                 device='{}' task='{}'; the CUDA Graph step capture path only wraps \
                 DeviceParamStore::step's update segment, unreachable from gemm/infer tasks. \
                 issue #1350)",
                cli.device, cli.task
            )
            .into());
        }
        #[cfg(feature = "graph-step")]
        {
            match graph_mode {
                "on" => {
                    fandhe_ai::set_cuda_graph_step_enabled(true);
                    if fandhe_ai::cuda_graph_step_mode() != fandhe_ai::CudaGraphStepMode::On {
                        return Err(
                            "MEASURE_ERROR: set_cuda_graph_step_enabled(true) did not take \
                             effect (cuda_graph_step_mode() != On after enabling; issue #1350)"
                                .into(),
                        );
                    }
                }
                "stream-only" => {
                    if fandhe_ai::cuda_graph_step_mode() != fandhe_ai::CudaGraphStepMode::StreamOnly
                    {
                        return Err(
                            "MEASURE_ERROR: --graph stream-only requires the launcher to export \
                             FANDHE_AI_CUDA_GRAPH_STEP=stream-only before this process starts \
                             (cuda_graph_step_mode() != StreamOnly; the API setter cannot select \
                             this diagnostic mode. issue #1350)"
                                .into(),
                        );
                    }
                }
                // `parse_cli_from` の allowlist（"on"／"stream-only" のみ
                // 受理）を既に通過済みのため到達しない分岐。
                other => {
                    return Err(format!(
                        "MEASURE_ERROR: unreachable --graph value '{other}' (should have been \
                         rejected by bench-common's CLI parser; issue #1350)"
                    )
                    .into());
                }
            }
        }
        #[cfg(not(feature = "graph-step"))]
        {
            return Err(format!(
                "MEASURE_ERROR: --graph {graph_mode} requires fandhe-ai >= 0.8.0 or a \
                 path-patched facade built with --features graph-step (issue #1350; see \
                 scripts/bench/framework-compare/README.md \"--graph\" section)"
            )
            .into());
        }
    } else {
        // イシュー #1350: 環境変数漏れの fail-open 防止（上記コメント
        // 参照）。`graph-step` feature が無効なビルドでは
        // `cuda_graph_step_mode` 自体を呼べないため、この検査は feature
        // 有効時のみ行う（無効時は facade の新 API 自体が存在しない）。
        #[cfg(feature = "graph-step")]
        if fandhe_ai::cuda_graph_step_mode() != fandhe_ai::CudaGraphStepMode::Off {
            return Err(
                "MEASURE_ERROR: FANDHE_AI_CUDA_GRAPH_STEP is set in the environment but \
                 --graph was not passed (this would silently measure an on/stream-only row as \
                 if it were off; issue #1350)"
                    .into(),
            );
        }
    }
    // イシュー #1339: `--device-checksum` は `--task gemm`（`--phases` なし。
    // `run_gemm`／`run_gemm_reuse` のみが device reduction 分岐を持つ）
    // 限定の allowlist 方式。`device-checksum` feature（crates.io 公開版
    // `fandhe-ai =0.7.0` には `Var::matmul_checksum` API が未収録のため
    // path patch 前提）が無効なビルドでは常に MEASURE_ERROR とする
    // （`--managed` と同型）。
    if cli.device_checksum {
        if cli.task != "gemm" || cli.phases {
            return Err(format!(
                "MEASURE_ERROR: --device-checksum is only supported for --task gemm without \
                 --phases (got task='{}' phases={}; issue #1339)",
                cli.task, cli.phases
            )
            .into());
        }
        #[cfg(not(feature = "device-checksum"))]
        {
            return Err(
                "MEASURE_ERROR: --device-checksum requires fandhe-ai >= 0.8.0 or a \
                 path-patched facade built with --features device-checksum (issue #1339; see \
                 scripts/bench/framework-compare/README.md \"--device-checksum\" section)"
                    .into(),
            );
        }
    }
    match (cli.task.as_str(), cli.mode.as_str(), cli.phases) {
        ("train", "fresh", true) => run_train_phases(cli),
        ("train", "reuse", true) => run_train_reuse_phases(cli),
        // イシュー #1182: `gemm --mode reuse --phases` を追加。fresh の
        // gemm との組合せは引き続き MEASURE_ERROR（下の catch-all）。
        ("gemm", "reuse", true) => run_gemm_reuse_phases(cli),
        // イシュー #1217: `infer --phases` を fresh/reuse 双方に追加
        // （train と異なり、fresh も対応する。README「`infer --phases`」
        // 節参照）。
        ("infer", "fresh", true) | ("infer", "reuse", true) => run_infer_phases(cli),
        (task, mode, true) => Err(format!(
            "MEASURE_ERROR: --phases is only implemented for task 'train', task 'infer', or \
             task 'gemm' with --mode reuse (got task='{task}' mode='{mode}'; issue #1009 / \
             #1182 / #1217)"
        )
        .into()),
        ("gemm", "fresh", false) => run_gemm(cli),
        ("gemm", "reuse", false) => run_gemm_reuse(cli),
        ("train", "fresh", false) => run_train(cli),
        ("train", "reuse", false) => run_train_reuse(cli),
        ("infer", "fresh", false) => run_infer(cli),
        // イシュー #1217: `infer --mode reuse` を追加。
        ("infer", "reuse", false) => run_infer_reuse(cli),
        (task, "reuse", false) => Err(format!(
            "MEASURE_ERROR: --mode reuse is not implemented for task '{task}' (gemm / train / \
             infer only; issue #925 / #958 / #1217)"
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
            managed: false,
            device_checksum: false,
            graph: None,
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
            managed: false,
            device_checksum: false,
            graph: None,
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
    fn phases_with_gemm_fresh_is_measure_error() {
        // `dispatch()`（`run()` の分岐本体。`parse_cli()` を経由せず直接
        // 呼べるよう分離してある）を通して、`--phases` が `--task train`
        // （fresh/reuse）・`--task infer`（fresh/reuse。イシュー #1217）・
        // `--task gemm --mode reuse`（イシュー #1182）限定であり、
        // `gemm --mode fresh` のみ引き続き拒否されることを固定する。
        let (task, mode) = ("gemm", "fresh");
        let out = temp_out_path(&format!("phases-unsupported-{task}-{mode}"));
        let cli = Cli {
            task: task.to_string(),
            device: "cpu".to_string(),
            size: 64,
            out: out.to_string_lossy().into_owned(),
            mode: mode.to_string(),
            phases: true,
            tf32: false,
            managed: false,
            device_checksum: false,
            graph: None,
        };
        let err = dispatch(&cli).expect_err("task/--phases combination must be rejected");
        let msg = err.to_string();
        assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
        assert!(msg.contains("--phases"), "msg={msg}");
        assert!(msg.contains(task), "msg={msg}");
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
            managed: false,
            device_checksum: false,
            graph: None,
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

    /// イシュー #1290: `make_gemm_phases_cli`（size 64 固定）の size 可変版。
    /// CPU gate 対象形状（README「GEMM ゲート 5 回計測」節。cpu={512,1024,2048}）
    /// の下限に近い N=512 でのスモーク・checksum 一致検証に使う（AC-1/AC-2）。
    fn make_gemm_phases_cli_sized(out: &std::path::Path, size: usize) -> Cli {
        Cli {
            size,
            ..make_gemm_phases_cli(out)
        }
    }

    /// イシュー #1290 受け入れ条件 3: `gemm --mode reuse --phases` が
    /// CPU gate 対象形状の下限（N=512）でも size=64 の既存スモーク
    /// （`gemm_reuse_phases_emits_one_row_per_phase_in_order`）と同じ
    /// 5 区間・スキーマで完走することを固定する。実機非依存（cpu）の
    /// ため `#[ignore]` は付けない。
    #[test]
    fn gemm_reuse_phases_cpu_smoke_n512() {
        let out = temp_out_path("gemm-phases-cpu-smoke-n512");
        let cli = make_gemm_phases_cli_sized(&out, 512);
        run_gemm_reuse_phases(&cli).expect("run_gemm_reuse_phases (N=512) failed");
        let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&out);
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 5, "lines={lines:?}");
        for (i, line) in lines.iter().enumerate() {
            assert!(line.contains("\"task\":\"gemm_phases\""), "line={line}");
            assert!(line.contains("\"device\":\"cpu\""), "line={line}");
            assert!(line.contains("\"size\":512"), "line={line}");
            assert!(line.contains("\"mode\":\"reuse\""), "line={line}");
            assert!(
                line.contains(&format!("\"phase_index\":{i}")),
                "line={line}"
            );
            assert!(line.contains("\"init_s\":"), "line={line}");
            assert!(!line.contains("\"init_s\":null"), "line={line}");
            assert!(!line.contains("\"checksum\":null"), "line={line}");
        }
        assert!(
            lines.last().unwrap().contains("\"phase\":\"iter_total\""),
            "content={content}"
        );
    }

    /// イシュー #1290 受け入れ条件 3: `gemm_reuse_phases_checksum_matches_
    /// run_gemm_reuse`（size 64）と同じ検証を N=512 でも固定する
    /// （AC-2 の bit 一致契約が CPU gate 対象形状の下限でも崩れないこと）。
    #[test]
    fn gemm_reuse_phases_checksum_matches_run_gemm_reuse_n512() {
        let reuse_path = temp_out_path("gemm-reuse-vs-phases-n512");
        let reuse_cli = Cli {
            size: 512,
            ..make_cli("gemm", "reuse", &reuse_path)
        };
        run_gemm_reuse(&reuse_cli).expect("run_gemm_reuse (N=512) failed");
        let reuse_checksum_line = last_line_checksum(&reuse_path);
        let _ = std::fs::remove_file(&reuse_path);

        let phases_path = temp_out_path("gemm-phases-checksum-n512");
        let phases_cli = make_gemm_phases_cli_sized(&phases_path, 512);
        run_gemm_reuse_phases(&phases_cli).expect("run_gemm_reuse_phases (N=512) failed");
        let content = std::fs::read_to_string(&phases_path).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&phases_path);

        for line in content.lines() {
            let key = "\"checksum\":";
            let start = line.find(key).expect("checksum field missing") + key.len();
            let rest = &line[start..];
            let end = rest.find([',', '}']).expect("checksum field end missing");
            let phase_checksum: f64 = rest[..end].trim().parse().expect("checksum not f64");
            assert_eq!(
                phase_checksum, reuse_checksum_line,
                "phase checksum diverges from run_gemm_reuse (N=512): line={line}"
            );
        }
    }

    /// イシュー #1337・#1438: `readout_var`（[`HostReadout`] 経由・
    /// `Var::host_view()`）の内容が「`to_tensor()` +
    /// `contiguous().as_slice().to_vec()`」（旧 legacy 経路。テスト内に
    /// インライン展開して保持する固定点）と bit 同一であることを固定する
    /// 回帰テスト（cpu・実機非依存）。
    #[test]
    fn readout_var_matches_legacy_to_vec_bit_exact() {
        let tape = make_tape("cpu").expect("test: make_tape 失敗");
        let a_data = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("test: a_data 構築失敗");
        let b_data = Tensor::new(vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2])
            .expect("test: b_data 構築失敗");
        let a = tape.var(&a_data);
        let b = tape.var(&b_data);
        let c = a.matmul(&b).expect("test: matmul 失敗");

        let legacy: Vec<f32> = c
            .to_tensor()
            .contiguous()
            .as_slice()
            .expect("test: as_slice() が None")
            .to_vec();

        let via_readout = readout_var(&c, "cpu").expect("test: readout_var 失敗");
        assert_eq!(
            &via_readout[..],
            &legacy[..],
            "readout_var の値が legacy 経路（to_tensor + to_vec）と bit 一致しない"
        );

        // checksum も同一（`f64` 逐次和の計算式・順序が feature 有無で
        // 不変であることの確認）。
        let legacy_checksum: f64 = legacy.iter().map(|&x| x as f64).sum();
        let via_checksum = checksum_var(&a.matmul(&b).expect("test: matmul(2) 失敗"), "cpu")
            .expect("test: checksum_var 失敗");
        assert_eq!(via_checksum.to_bits(), legacy_checksum.to_bits());
    }

    /// codex-review 指摘（PR #1452 P2）: `readout_uses_borrowed_view` が
    /// `device == "metal"` で選ぶ legacy 分岐（[`HostReadout::Owned`]）
    /// が、CPU/CUDA の借用ビュー分岐（[`HostReadout::Borrowed`]）と
    /// bit 同一の値・checksum を返すことを固定する回帰テスト（tape 自体
    /// は cpu。`readout_var`/`checksum_var` の `device` 引数はコピー方式
    /// のみを選ぶため、`Device::Metal` 実機を要さず両分岐を同一プロセス
    /// 内で直接比較できる）。
    #[test]
    fn readout_var_metal_legacy_branch_matches_borrowed_view_bit_exact() {
        assert!(readout_uses_borrowed_view("cpu"));
        assert!(readout_uses_borrowed_view("cuda"));
        assert!(!readout_uses_borrowed_view("metal"));

        let tape = make_tape("cpu").expect("test: make_tape 失敗");
        let a_data = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("test: a_data 構築失敗");
        let b_data = Tensor::new(vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2])
            .expect("test: b_data 構築失敗");
        let a = tape.var(&a_data);
        let b = tape.var(&b_data);

        let c_borrowed = a.matmul(&b).expect("test: matmul(borrowed) 失敗");
        let via_borrowed = readout_var(&c_borrowed, "cpu").expect("test: readout_var(cpu) 失敗");

        let c_owned = a.matmul(&b).expect("test: matmul(owned) 失敗");
        let via_owned = readout_var(&c_owned, "metal").expect("test: readout_var(metal) 失敗");

        assert_eq!(
            &via_borrowed[..],
            &via_owned[..],
            "device=\"metal\" の legacy 分岐が借用ビュー分岐と bit 一致しない"
        );

        let checksum_borrowed = checksum_var(&a.matmul(&b).expect("test: matmul(3) 失敗"), "cpu")
            .expect("test: checksum_var(cpu) 失敗");
        let checksum_owned = checksum_var(&a.matmul(&b).expect("test: matmul(4) 失敗"), "metal")
            .expect("test: checksum_var(metal) 失敗");
        assert_eq!(
            checksum_borrowed.to_bits(),
            checksum_owned.to_bits(),
            "device=\"metal\" の checksum_var が借用ビュー分岐と bit 一致しない"
        );
    }

    /// イシュー #1337・#1335・#1438: `Var::host_view()`（`readout_var`
    /// が内部で使う）が返す `VarHostView` の寿命契約——`Tape` の借用を
    /// 保持しないため、生存中に同じ `Tape` へノード追加演算（`matmul`）
    /// を呼んでも panic しない——を、`readout_var` 経由の呼び出し
    /// パターン（`measure_gemm_reuse_phases` の反復内で `out`
    /// （`HostReadout`）を生存させたまま次反復の `a.matmul(&b)` を呼ぶ形）
    /// で確認する回帰点（`Var::host_view` doc の「P1 是正」参照。#1438 で
    /// 常時実行化・旧計測専用 feature（#1337 導入）の cfg は撤去）。
    #[test]
    fn host_view_readout_keeps_tape_usable() {
        let tape = make_tape("cpu").expect("test: make_tape 失敗");
        let a_data = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).expect("test: a_data");
        let b_data = Tensor::new(vec![1.0f32, 0.0, 0.0, 1.0], &[2, 2]).expect("test: b_data");
        let a = tape.var(&a_data);
        let b = tape.var(&b_data);

        let c1 = a.matmul(&b).expect("test: matmul(1) 失敗");
        let out1 = readout_var(&c1, "cpu").expect("test: readout_var(1) 失敗");
        // `out1`（`VarHostView`）を生存させたまま同じ `Tape` へ演算を
        // 追加で呼ぶ。旧実装（`Tape` の `RefCell` 借用を持ち越す版）は
        // ここで `borrow_mut()` の実行時 panic を起こしていた。
        let c2 = a
            .matmul(&b)
            .expect("test: matmul(2) failed while out1 alive (panic bug?)");
        let out2 = readout_var(&c2, "cpu").expect("test: readout_var(2) 失敗");

        assert_eq!(
            &out1[..],
            &out2[..],
            "同一入力の 2 回の matmul で結果が食い違う"
        );
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
            managed: false,
            device_checksum: false,
            graph: None,
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
            managed: false,
            device_checksum: false,
            graph: None,
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

    // イシュー #1217: `infer --mode reuse`／`infer --phases`（fresh/reuse）
    // の受け入れ条件を固定する回帰テスト群。`train --mode reuse`／
    // `gemm --mode reuse --phases` と同型の検証（checksum 複合判定・
    // JSONL フィールド・phase 順序・Σphase ≤ iter_total）を infer へ適用
    // する。

    fn make_infer_phases_cli(mode: &str, out: &std::path::Path) -> Cli {
        Cli {
            task: "infer".to_string(),
            device: "cpu".to_string(),
            size: 64,
            out: out.to_string_lossy().into_owned(),
            mode: mode.to_string(),
            phases: true,
            tf32: false,
            managed: false,
            device_checksum: false,
            graph: None,
        }
    }

    /// `infer --mode reuse`（`predict_resident`）の checksum が `infer
    /// --mode fresh`（`predict`）と統一複合判定（相対誤差 1e-3 未満また
    /// は絶対誤差 1e-5 未満。coding-rust.md）の範囲内で一致することを
    /// 固定する（`train_reuse_matches_fresh_final_loss_within_composite_tolerance`
    /// と同型）。
    #[test]
    fn infer_reuse_matches_fresh_checksum_within_composite_tolerance() {
        let fresh_path = temp_out_path("infer-fresh");
        let reuse_path = temp_out_path("infer-reuse");

        run_infer(&make_cli("infer", "fresh", &fresh_path)).expect("run_infer (fresh) failed");
        run_infer_reuse(&make_cli("infer", "reuse", &reuse_path))
            .expect("run_infer_reuse (reuse) failed");

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
            "fresh/reuse infer checksum mismatch: fresh={fresh_checksum} reuse={reuse_checksum} \
             abs_diff={abs_diff} rel_diff={rel_diff}"
        );
    }

    #[test]
    fn infer_reuse_produces_expected_record_fields() {
        let out = temp_out_path("infer-reuse-fields");
        run_infer_reuse(&make_cli("infer", "reuse", &out)).expect("run_infer_reuse failed");
        let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&out);
        let last = content.lines().next_back().expect("test: JSONL に行がない");
        assert!(last.contains("\"task\":\"infer\""), "line={last}");
        assert!(last.contains("\"mode\":\"reuse\""), "line={last}");
        assert!(last.contains("\"init_s\":"), "line={last}");
        assert!(!last.contains("\"init_s\":null"), "line={last}");
        assert!(last.contains("\"throughput_per_s\":"), "line={last}");
        assert!(!last.contains("\"throughput_per_s\":null"), "line={last}");
    }

    /// (a) fresh・cpu の `infer --phases` が 4 区間（predict/host_copy/
    /// checksum/iter_total）を `phase_index` 連番・`task:"infer_phases"`・
    /// `init_s` なし・末尾 `iter_total` で出力することを固定する。
    #[test]
    fn infer_phases_fresh_cpu_emits_one_row_per_phase_in_order() {
        let out = temp_out_path("infer-phases-fresh-cpu-order");
        run_infer_phases(&make_infer_phases_cli("fresh", &out)).expect("run_infer_phases failed");
        let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&out);
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 4, "lines={lines:?}");
        for (i, line) in lines.iter().enumerate() {
            assert!(line.contains("\"task\":\"infer_phases\""), "line={line}");
            assert!(line.contains("\"mode\":\"fresh\""), "line={line}");
            assert!(
                line.contains(&format!("\"phase_index\":{i}")),
                "line={line}"
            );
            // fresh は `init_s: None` のため `to_json_line` がキー自体を
            // 省略する（`bench-common::Record::to_json_line`。reuse 側の
            // `infer_phases_reuse_emits_one_row_per_phase_in_order_with_init_s`
            // と対称）。
            assert!(!line.contains("\"init_s\""), "line={line}");
        }
        assert!(lines.last().unwrap().contains("\"phase\":\"iter_total\""));
        assert!(
            lines[0].contains("\"phase\":\"predict\""),
            "lines={lines:?}"
        );
    }

    /// reuse の `infer --phases` が 4 区間（predict_resident/host_copy/
    /// checksum/iter_total）を `init_s` ありで出力することを固定する
    /// （cpu で動作する。`device_class` 区分による GPU 専用区間との違い
    /// は README「`infer --phases`」節の表を参照）。
    #[test]
    fn infer_phases_reuse_emits_one_row_per_phase_in_order_with_init_s() {
        let out = temp_out_path("infer-phases-reuse-order");
        run_infer_phases(&make_infer_phases_cli("reuse", &out)).expect("run_infer_phases failed");
        let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&out);
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 4, "lines={lines:?}");
        for (i, line) in lines.iter().enumerate() {
            assert!(line.contains("\"task\":\"infer_phases\""), "line={line}");
            assert!(line.contains("\"mode\":\"reuse\""), "line={line}");
            assert!(
                line.contains(&format!("\"phase_index\":{i}")),
                "line={line}"
            );
            assert!(line.contains("\"init_s\":"), "line={line}");
            assert!(!line.contains("\"init_s\":null"), "line={line}");
        }
        assert!(lines.last().unwrap().contains("\"phase\":\"iter_total\""));
        assert!(
            lines[0].contains("\"phase\":\"predict_resident\""),
            "lines={lines:?}"
        );
    }

    /// (c) fresh の `infer --phases` の checksum（各行の `checksum`）が
    /// `run_infer` の JSONL 出力の checksum と一致することを固定する
    /// （`gemm_reuse_phases_checksum_matches_run_gemm_reuse` と同型）。
    #[test]
    fn infer_phases_fresh_checksum_matches_run_infer() {
        let fresh_path = temp_out_path("infer-fresh-vs-phases");
        run_infer(&make_cli("infer", "fresh", &fresh_path)).expect("run_infer failed");
        let fresh_checksum = last_line_checksum(&fresh_path);
        let _ = std::fs::remove_file(&fresh_path);

        let phases_path = temp_out_path("infer-phases-fresh-checksum");
        run_infer_phases(&make_infer_phases_cli("fresh", &phases_path))
            .expect("run_infer_phases failed");
        let content = std::fs::read_to_string(&phases_path).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&phases_path);

        for line in content.lines() {
            let key = "\"checksum\":";
            let start = line.find(key).expect("checksum field missing") + key.len();
            let rest = &line[start..];
            let end = rest.find([',', '}']).expect("checksum field end missing");
            let phase_checksum: f64 = rest[..end].trim().parse().expect("checksum not f64");
            assert_eq!(
                phase_checksum, fresh_checksum,
                "phase checksum diverges from run_infer: line={line}"
            );
        }
    }

    /// (c) reuse の `infer --phases` の checksum が `run_infer_reuse` の
    /// JSONL 出力の checksum と一致することを固定する。
    #[test]
    fn infer_phases_reuse_checksum_matches_run_infer_reuse() {
        let reuse_path = temp_out_path("infer-reuse-vs-phases");
        run_infer_reuse(&make_cli("infer", "reuse", &reuse_path)).expect("run_infer_reuse failed");
        let reuse_checksum = last_line_checksum(&reuse_path);
        let _ = std::fs::remove_file(&reuse_path);

        let phases_path = temp_out_path("infer-phases-reuse-checksum");
        run_infer_phases(&make_infer_phases_cli("reuse", &phases_path))
            .expect("run_infer_phases failed");
        let content = std::fs::read_to_string(&phases_path).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&phases_path);

        for line in content.lines() {
            let key = "\"checksum\":";
            let start = line.find(key).expect("checksum field missing") + key.len();
            let rest = &line[start..];
            let end = rest.find([',', '}']).expect("checksum field end missing");
            let phase_checksum: f64 = rest[..end].trim().parse().expect("checksum not f64");
            assert_eq!(
                phase_checksum, reuse_checksum,
                "phase checksum diverges from run_infer_reuse: line={line}"
            );
        }
    }

    /// (d) fresh/reuse それぞれについて、各反復の構成区間合計が
    /// `iter_total` を超えず、計時オーバーヘッドの上限（90%）を満たす
    /// ことを固定する（`gemm_reuse_phases_each_iter_phase_sum_does_not_exceed_total`
    /// と同型。標本数は `WARMUP_ITERS + MEASURE_ITERS`）。
    #[test]
    fn infer_phases_each_iter_phase_sum_does_not_exceed_total() {
        for mode in ["fresh", "reuse"] {
            let cli =
                make_infer_phases_cli(mode, &temp_out_path(&format!("infer-phases-sum-{mode}")));
            let (phases, _checksum, _init_s) =
                measure_infer_phases(&cli, mode).expect("measure_infer_phases failed");
            let totals = phases.durations(PHASE_GEMM_ITER_TOTAL);
            assert_eq!(totals.len(), WARMUP_ITERS + MEASURE_ITERS, "mode={mode}");
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
                    "mode={mode} iter={iter}: phase sum {sum:?} exceeds iter_total {total:?}"
                );
                assert!(
                    sum.as_secs_f64() >= 0.9 * total.as_secs_f64(),
                    "mode={mode} iter={iter}: phase sum {sum:?} is less than 90% of iter_total \
                     {total:?}"
                );
            }
        }
    }

    /// 実機（CUDA）依存の smoke テスト。fresh は 6 区間（leaf_register/
    /// forward/to_tensor/host_copy/checksum/iter_total）・reuse は 4 区間
    /// （predict_resident/host_copy/checksum/iter_total）。
    #[test]
    #[ignore]
    fn infer_reuse_and_phases_cuda_smoke() {
        let reuse_out = temp_out_path("infer-reuse-cuda-smoke");
        dispatch(&Cli {
            task: "infer".to_string(),
            device: "cuda".to_string(),
            size: 64,
            out: reuse_out.to_string_lossy().into_owned(),
            mode: "reuse".to_string(),
            phases: false,
            tf32: false,
            managed: false,
            device_checksum: false,
            graph: None,
        })
        .expect("cuda infer --mode reuse smoke failed");
        let reuse_content = std::fs::read_to_string(&reuse_out).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&reuse_out);
        assert!(
            reuse_content.contains("\"task\":\"infer\"")
                && reuse_content.contains("\"mode\":\"reuse\""),
            "content={reuse_content}"
        );

        for (mode, expected_rows) in [("fresh", 6usize), ("reuse", 4usize)] {
            let out = temp_out_path(&format!("infer-phases-cuda-smoke-{mode}"));
            dispatch(&Cli {
                task: "infer".to_string(),
                device: "cuda".to_string(),
                size: 64,
                out: out.to_string_lossy().into_owned(),
                mode: mode.to_string(),
                phases: true,
                tf32: false,
                managed: false,
                device_checksum: false,
                graph: None,
            })
            .expect("cuda infer --phases smoke failed");
            let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
            let _ = std::fs::remove_file(&out);
            assert_eq!(
                content.lines().count(),
                expected_rows,
                "mode={mode} content={content}"
            );
            assert!(
                content.contains("\"phase\":\"iter_total\""),
                "mode={mode} content={content}"
            );
        }
    }

    /// 実機（Metal）依存の smoke テスト。macOS のみコンパイル対象。
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore]
    fn infer_reuse_and_phases_metal_smoke() {
        let reuse_out = temp_out_path("infer-reuse-metal-smoke");
        dispatch(&Cli {
            task: "infer".to_string(),
            device: "metal".to_string(),
            size: 64,
            out: reuse_out.to_string_lossy().into_owned(),
            mode: "reuse".to_string(),
            phases: false,
            tf32: false,
            managed: false,
            device_checksum: false,
            graph: None,
        })
        .expect("metal infer --mode reuse smoke failed");
        let reuse_content = std::fs::read_to_string(&reuse_out).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&reuse_out);
        assert!(
            reuse_content.contains("\"task\":\"infer\"")
                && reuse_content.contains("\"mode\":\"reuse\""),
            "content={reuse_content}"
        );

        for (mode, expected_rows) in [("fresh", 6usize), ("reuse", 4usize)] {
            let out = temp_out_path(&format!("infer-phases-metal-smoke-{mode}"));
            dispatch(&Cli {
                task: "infer".to_string(),
                device: "metal".to_string(),
                size: 64,
                out: out.to_string_lossy().into_owned(),
                mode: mode.to_string(),
                phases: true,
                tf32: false,
                managed: false,
                device_checksum: false,
                graph: None,
            })
            .expect("metal infer --phases smoke failed");
            let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
            let _ = std::fs::remove_file(&out);
            assert_eq!(
                content.lines().count(),
                expected_rows,
                "mode={mode} content={content}"
            );
            assert!(
                content.contains("\"phase\":\"iter_total\""),
                "mode={mode} content={content}"
            );
        }
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
                managed: false,
                device_checksum: false,
                graph: None,
            };
            let err = dispatch(&cli).expect_err("--tf32 must be rejected on bench-fandhe");
            let msg = err.to_string();
            assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
            assert!(msg.contains("--tf32"), "msg={msg}");
            assert!(msg.contains("0.5.0"), "msg={msg}");
        }
    }

    /// イシュー #1353: `--managed` は `--device cuda` 以外では常に
    /// MEASURE_ERROR で fail-fast する（プロセスワイドフラグが cpu 計測で
    /// 無音 no-op になるのを防ぐため）。
    #[test]
    fn managed_flag_on_non_cuda_device_is_measure_error() {
        for device in ["cpu", "metal"] {
            let out = temp_out_path(&format!("managed-non-cuda-{device}"));
            let cli = Cli {
                task: "gemm".to_string(),
                device: device.to_string(),
                size: 64,
                out: out.to_string_lossy().into_owned(),
                mode: "fresh".to_string(),
                phases: false,
                tf32: false,
                managed: true,
                device_checksum: false,
                graph: None,
            };
            let err = dispatch(&cli).expect_err("--managed must be rejected on non-cuda device");
            let msg = err.to_string();
            assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
            assert!(msg.contains("--managed"), "msg={msg}");
        }
    }

    /// イシュー #1353: `managed-placement` feature が無効な既定ビルド
    /// （`fandhe-ai =0.7.0` ピンには `set_cuda_managed_memory_enabled` API
    /// 自体が未収録）では、`--device cuda` でも `--managed` は常に
    /// MEASURE_ERROR で fail-fast する。本テストはこのビルド構成（既定
    /// feature）でのみ意味を持つ（`managed-placement` feature 有効ビルド
    /// では実際に path patch 済み facade を呼び出す経路が走るため、本
    /// テストとは別に GB10 実機実測で検証する。README「`--managed`」節）。
    #[test]
    #[cfg(not(feature = "managed-placement"))]
    fn managed_flag_without_feature_is_measure_error() {
        let out = temp_out_path("managed-no-feature-cuda");
        let cli = Cli {
            task: "gemm".to_string(),
            device: "cuda".to_string(),
            size: 64,
            out: out.to_string_lossy().into_owned(),
            mode: "fresh".to_string(),
            phases: false,
            tf32: false,
            managed: true,
            device_checksum: false,
            graph: None,
        };
        let err = dispatch(&cli)
            .expect_err("--managed must be rejected without managed-placement feature");
        let msg = err.to_string();
        assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
        assert!(msg.contains("--managed"), "msg={msg}");
        assert!(msg.contains("0.8.0"), "msg={msg}");
    }

    /// イシュー #1350: `--graph` は `--device cuda --task train` 以外では
    /// 常に MEASURE_ERROR で fail-fast する（`DeviceParamStore::step` の
    /// update 区間へ到達しない task／device の組合せで無音 no-op になる
    /// のを防ぐ。`--managed` と同型）。
    #[test]
    fn graph_flag_on_non_cuda_or_non_train_is_measure_error() {
        let cases = [
            ("cpu", "train"),
            ("metal", "train"),
            ("cuda", "gemm"),
            ("cuda", "infer"),
        ];
        for (device, task) in cases {
            let out = temp_out_path(&format!("graph-non-target-{device}-{task}"));
            let cli = Cli {
                task: task.to_string(),
                device: device.to_string(),
                size: 64,
                out: out.to_string_lossy().into_owned(),
                mode: "fresh".to_string(),
                phases: false,
                tf32: false,
                managed: false,
                device_checksum: false,
                graph: Some("on".to_string()),
            };
            let err = dispatch(&cli)
                .expect_err("--graph must be rejected outside --device cuda --task train");
            let msg = err.to_string();
            assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
            assert!(msg.contains("--graph"), "msg={msg}");
        }
    }

    /// イシュー #1350: `graph-step` feature が無効な既定ビルド（`fandhe-ai
    /// =0.7.0` ピンには `cuda_graph_step_mode`/`cuda_graph_step_stats` API
    /// 自体が未収録）では、`--device cuda --task train` でも `--graph` は
    /// 常に MEASURE_ERROR で fail-fast する。本テストはこのビルド構成
    /// （既定 feature）でのみ意味を持つ（`graph-step` feature 有効ビルド
    /// では実際に path patch 済み facade を呼び出す経路が走るため、本
    /// テストとは別に GB10 実機実測で検証する。README「`--graph`」節）。
    #[test]
    #[cfg(not(feature = "graph-step"))]
    fn graph_flag_without_feature_is_measure_error() {
        for mode in ["on", "stream-only"] {
            let out = temp_out_path(&format!("graph-no-feature-{mode}"));
            let cli = Cli {
                task: "train".to_string(),
                device: "cuda".to_string(),
                size: 64,
                out: out.to_string_lossy().into_owned(),
                mode: "reuse".to_string(),
                phases: false,
                tf32: false,
                managed: false,
                device_checksum: false,
                graph: Some(mode.to_string()),
            };
            let err =
                dispatch(&cli).expect_err("--graph must be rejected without graph-step feature");
            let msg = err.to_string();
            assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
            assert!(msg.contains("--graph"), "msg={msg}");
            assert!(msg.contains("0.8.0"), "msg={msg}");
        }
    }

    /// イシュー #1339: `--device-checksum` は `--task gemm`（`--phases`
    /// なし）限定の allowlist 方式で、それ以外の task／`--phases` 併用は
    /// 常に MEASURE_ERROR で fail-fast する。
    #[test]
    fn device_checksum_flag_on_non_gemm_task_is_measure_error() {
        for task in ["train", "infer"] {
            let out = temp_out_path(&format!("device-checksum-non-gemm-{task}"));
            let cli = Cli {
                task: task.to_string(),
                device: "cpu".to_string(),
                size: 64,
                out: out.to_string_lossy().into_owned(),
                mode: "fresh".to_string(),
                phases: false,
                tf32: false,
                managed: false,
                device_checksum: true,
                graph: None,
            };
            let err =
                dispatch(&cli).expect_err("--device-checksum must be rejected for non-gemm tasks");
            let msg = err.to_string();
            assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
            assert!(msg.contains("--device-checksum"), "msg={msg}");
        }
    }

    /// イシュー #1339: `--device-checksum --phases` の併用は常に
    /// MEASURE_ERROR（`run_gemm`／`run_gemm_reuse` のみが対応し、
    /// `gemm --mode reuse --phases` は対象外）。
    #[test]
    fn device_checksum_flag_with_phases_is_measure_error() {
        let out = temp_out_path("device-checksum-phases");
        let cli = Cli {
            task: "gemm".to_string(),
            device: "cpu".to_string(),
            size: 64,
            out: out.to_string_lossy().into_owned(),
            mode: "reuse".to_string(),
            phases: true,
            tf32: false,
            managed: false,
            device_checksum: true,
            graph: None,
        };
        let err = dispatch(&cli).expect_err("--device-checksum --phases must be rejected");
        let msg = err.to_string();
        assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
        assert!(msg.contains("--device-checksum"), "msg={msg}");
    }

    /// イシュー #1339: `device-checksum` feature が無効な既定ビルド
    /// （`fandhe-ai =0.7.0` ピンには `Var::matmul_checksum` API 自体が
    /// 未収録）では、`--task gemm` でも `--device-checksum` は常に
    /// MEASURE_ERROR で fail-fast する。本テストはこのビルド構成（既定
    /// feature）でのみ意味を持つ（`device-checksum` feature 有効ビルドは
    /// 本ファイル冒頭の手動確認手順・README「`--device-checksum`」節を
    /// 参照。CPU 経由の実測は同 feature 有効ビルドで別途確認済み）。
    #[test]
    #[cfg(not(feature = "device-checksum"))]
    fn device_checksum_flag_without_feature_is_measure_error() {
        let out = temp_out_path("device-checksum-no-feature");
        let cli = Cli {
            task: "gemm".to_string(),
            device: "cpu".to_string(),
            size: 64,
            out: out.to_string_lossy().into_owned(),
            mode: "fresh".to_string(),
            phases: false,
            tf32: false,
            managed: false,
            device_checksum: true,
            graph: None,
        };
        let err = dispatch(&cli)
            .expect_err("--device-checksum must be rejected without device-checksum feature");
        let msg = err.to_string();
        assert!(msg.starts_with("MEASURE_ERROR:"), "msg={msg}");
        assert!(msg.contains("--device-checksum"), "msg={msg}");
        assert!(msg.contains("0.8.0"), "msg={msg}");
    }

    /// イシュー #1339: `device-checksum` feature 有効ビルド（`--config
    /// patch.crates-io.fandhe-ai.path=...` 併用。README「`--device-checksum`」
    /// 節参照）でのみコンパイル・実行される cpu 経由の end-to-end 検証。
    /// `--device-checksum` あり／なしで同一入力の checksum が一致すること
    /// （`gemm_checksum` は `gemm` と同一カーネル・同一縮約順序の契約）を
    /// fresh／reuse 双方で確認する。実機非依存（cpu）のため `#[ignore]` は
    /// 付けない。
    #[test]
    #[cfg(feature = "device-checksum")]
    fn device_checksum_matches_legacy_checksum_fresh_and_reuse() {
        for mode in ["fresh", "reuse"] {
            let out_off = temp_out_path(&format!("device-checksum-off-{mode}"));
            let mut cli_off = make_cli("gemm", mode, &out_off);
            cli_off.device_checksum = false;
            dispatch(&cli_off).expect("--device-checksum off は成功するはず");
            let checksum_off = last_line_checksum(&out_off);

            let out_on = temp_out_path(&format!("device-checksum-on-{mode}"));
            let mut cli_on = make_cli("gemm", mode, &out_on);
            cli_on.device_checksum = true;
            dispatch(&cli_on).expect("--device-checksum on は cpu バックエンドで成功するはず");
            let checksum_on = last_line_checksum(&out_on);

            assert_eq!(
                checksum_off, checksum_on,
                "mode={mode}: --device-checksum の有無で checksum が一致するはず \
                 （gemm_checksum は gemm と同一カーネル・同一縮約順序の契約）"
            );
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
                managed: false,
                device_checksum: false,
                graph: None,
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
                managed: false,
                device_checksum: false,
                graph: None,
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
    // --- ハーネス限定のスケール付き絶対誤差救済項（イシュー #1247）:
    // fandhe-ai 側 GEMM は本救済項なしに 0 fail のままであることを固定
    // し、`bench-common::parity` 側の第 3 項が本体の回帰を隠す経路を
    // 遮断する（`docs/candle-parity-tolerance-contract-decision.md` §6
    // 「fandhe-ai 側 0 fail 不変」の根拠を実装で固定）。 ---------------

    /// N ∈ {64, 256, 512, 2048} の CPU GEMM は、`GemmReference::verify`
    /// が使う `ScaledAbsTolerance`（実ベンチ入力由来）による救済に依存
    /// せず、既存 2 条件のみで 0 fail であることを固定する（受け入れ条件
    /// (b)）。N=2048 は candle 比較で判定不能になる形状（#1184）だが、
    /// fandhe-ai 自身の CPU 参照比較では引き続き 0 fail のままである
    /// ことがここでの主張。
    #[test]
    fn gemm_cpu_parity_zero_fail_without_scaled_rescue() {
        for n in [64usize, 256, 512, 2048] {
            let (a_data, b_data) = gemm_inputs(n).expect("gemm_inputs");
            let a_host = a_data
                .contiguous()
                .as_slice()
                .expect("a_data as_slice")
                .to_vec();
            let b_host = b_data
                .contiguous()
                .as_slice()
                .expect("b_data as_slice")
                .to_vec();
            let reference = GemmReference::compute(n, &a_host, &b_host).expect("compute");

            let tape = make_tape("cpu").expect("make_tape(cpu)");
            let a = tape.var(&a_data);
            let b = tape.var(&b_data);
            let c = a.matmul(&b).expect("matmul");
            let out = readout_var(&c, "cpu").expect("readout_var");

            let stats = reference.verify(&out).expect("verify");
            assert_eq!(stats.fail_count, 0, "n={n}: fail_count must be 0");
            assert_eq!(
                stats.scaled_abs_rescued, 0,
                "n={n}: fandhe-ai 側 CPU GEMM はスケール付き絶対誤差救済に依存せず 0 fail のままである契約"
            );

            // `ScaledAbsTolerance::NONE`（既存 2 条件のみ）でも明示的に
            // 0 fail であることを確認する（`GemmReference::verify` が
            // 内部で使う `tol` に依存しない、独立した確認）。
            let legacy = compare_elementwise(&out, reference.as_slice(), &ScaledAbsTolerance::NONE)
                .expect("compare_elementwise (legacy)");
            assert_eq!(legacy.fail_count, 0, "n={n}: legacy fail_count must be 0");
        }
    }

    /// `run_gemm` が emit する JSONL に新設 2 キー（イシュー #1247）が
    /// 含まれ、fandhe-ai 側 CPU GEMM では `parity_scaled_abs_rescued:0`
    /// のまま（救済に依存しない）であることを、実際の CLI 経路
    /// （`Record::to_json_line` 結線）で固定する。
    #[test]
    fn run_gemm_jsonl_contains_scaled_abs_keys_with_zero_rescue() {
        let out = temp_out_path("gemm-scaled-abs-keys");
        let cli = make_cli("gemm", "fresh", &out);
        run_gemm(&cli).expect("run_gemm (cpu fresh, size=64) failed");
        let content = std::fs::read_to_string(&out).expect("test: JSONL 読み取り失敗");
        let _ = std::fs::remove_file(&out);
        let last = content.lines().next_back().expect("JSONL に行がない");

        assert!(last.contains("\"parity_fail_count\":0"), "line={last}");
        assert!(
            last.contains("\"parity_scaled_abs_rescued\":0"),
            "line={last}"
        );
        assert!(last.contains("\"parity_scaled_abs_bound\":"), "line={last}");
    }
}
