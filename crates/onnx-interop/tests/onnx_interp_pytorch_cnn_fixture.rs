//! ONNX `Conv`・`MaxPool`・`AveragePool`・`GlobalAveragePool`・
//! `BatchNormalization`・`Flatten` import の PyTorch 実生成 fixture 突合
//! （イシュー #2329・親 #2185 の残作業）。
//!
//! `tests/onnx_interp_cnn_ops.rs`（イシュー #2200）・`tests/onnx_interp_
//! conv_pool.rs`（イシュー #2199）は `fandhe_ai_autodiff::nn::*::
//! forward_host` との**内部突合**（自作 export → 自作 import の自己整合）
//! に留まっていた（`docs/perf/logs/onnx-conv-pool-import-2199/README.md`
//! 「torch 実生成 fixture による突合の未実施」節）。本ファイルは
//! `tests/fixtures/pytorch-onnx-cnn-ops/`（`torch.onnx.export` が実生成した
//! `.onnx`・`reference.json`）を decode → `build_graph` → `run` の全経路で
//! 実行し、PyTorch の実行結果・`state_dict` と突合する（外部で生成された
//! グラフ表現に対する import の到達性の実証。CI はコミット済み fixture の
//! みを読み、torch には依存しない）。
//!
//! ## 判定方式についての注記（REQ-2・REQ-7 との混同禁止。2026-09-28
//! ユーザー承認で正式方式へ移行）
//!
//! 縮約系（`Conv`・`AveragePool`・`GlobalAveragePool`・`BatchNormalization`。
//! dynamo 分解経路の `ReduceMean` を含む）の結合順序（PyTorch CPU 実行系
//! vs 本クレートの直接ループ）は異なるため、bit 完全一致は原理的に目標に
//! できない——PyTorch 参照値との **bit 同一**は縮約系の受け入れ条件から
//! 外す（元は #2185 の受け入れ条件チェックボックスが要求していたが、
//! `docs/spec/04-requirements.md` REQ-7 自体は「相対誤差 1e-3 以内」を
//! 求めるのみで bit 同一は要求していない。すなわち本改定は spec の緩和では
//! なく、実装リポ側 issue の受け入れ条件を spec の定めに整合させる変更
//! ——`docs/onnx-pytorch-fixture-reduction-parity-judgment-decision.md`
//! 参照）。
//!
//! 縮約系の判定は次の **併用方式**（[`Expectation::Req7BaselineNonRegression`]）
//! で行う:
//! 1. `tests/model_zoo_parity.rs` と同じ **REQ-7 事前固定式**
//!    `abs_err / (|ref| + 1e-6) <= 1e-3` の `fail_count == 0`
//!    を必須条件として維持する（tolerance 自体は変更しない）
//! 2. その上で、ケースごとの実測上限 baseline（[`ReductionBaseline`]・
//!    `REDUCTION_BASELINES`）に対する fail-closed 非後退判定
//!    （`total` 完全一致・`fail_count`／`max_abs_diff`／`max_rel_err`／
//!    `mean_abs_diff` のいずれも記録済み ceiling 以下）を行う。ceiling は
//!    実測値そのもの（余裕係数なし）であり、baseline の追加・更新は実測値
//!    のみ・人間承認必須（`crates/backend-cuda/tests/common/
//!    parity_baseline.rs::ParityBaseline` と同じ fail-closed 設計）
//!
//! 純粋な選択・形状操作（`MaxPool`・`Flatten`）は従来どおり
//! [`Expectation::BitExact`]（フォールバックなしの bit 一致のみ）を維持
//! する。縮約系 op（[`REDUCTION_OP_TYPES`]）を含む演算列と含まない演算列の
//! どちらへ各 [`Expectation`] を適用するかは
//! [`assert_expectation_matches_op_types`] が構造的に検査し、表の書き換え
//! による無断緩和（縮約系ケースを `BitExact` 側へ、あるいはその逆へ誤って
//! 動かす類の変更）を機械的に検知する。
//!
//! `.claude/rules/coding-rust.md` の REQ-2 バックエンド間数値一致 OR 複合
//! 判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）とは別指標であり、
//! 混同してどちらかを緩和しない。ONNX インタープリタはホスト CPU 実行のみ
//! （`onnx::interp::run`。`interp.rs` のディスパッチ表で対象 6 op はすべて
//! `(compute_*(..)?, false)` = 常にホスト実行）のため REQ-2 の対象外
//! （構造的 N/A。`docs/onnx-model-zoo-parity.md` §4 と同じ理由）。
//!
//! ## 計算経路の決定性（CPU feature 検出との無関係性）
//!
//! 本ファイルが経由する計算（`ops::conv`／`ops::pool`／`ops::batch_norm`／
//! `ops::global_average_pool`・dynamo 分解経路の `ReduceMean`
//! 〈`onnx::interp_ext::compute_reduce_mean` → `fandhe_ai_autodiff::Var::
//! mean` → `default_ops::NaiveOps::sum` → `autodiff::eval::sum` の
//! `Iterator::sum`〉）はいずれも `rayon`・SIMD intrinsics・
//! `is_x86_feature_detected!` 等のランタイム分岐を含まない単純な逐次
//! ループ（`f32::mul_add`／`f64` 蓄積。走査順は入力の row-major 順に固定）
//! である。`fandhe-ai-onnx-interop` クレート自体も `rayon` に依存しない
//! （`backend-cpu` の AVX2/AVX-512/NEON 系 SIMD カーネル・ランタイム CPU
//! feature 検出はこのクレートから到達しない）。したがって本ファイルの
//! baseline は CI（GitHub ホステッド `ubuntu-latest`・x86_64）・ローカル
//! （x86_64）を問わず単一の値で成立し、実行環境の CPU feature 差・
//! スレッド数差による揺れは生じない（経路ごとに baseline を分ける必要は
//! ない）。
//!
//! ## 純粋な選択・形状操作（bit 一致のみを許容）
//!
//! `MaxPool`・`Flatten` はフォールバックなしの bit 一致のみを要求する
//! （選択・形状操作であり縮約を伴わないため）。
//!
//! ## exporter による経路の違い
//!
//! `torch.onnx.export` は `dynamo=False`（TorchScript exporter。opset 13）と
//! `dynamo=True`（既定 exporter。opset 20）の双方を fixture 化している。
//! dynamo exporter は `GlobalAveragePool` を `ReduceMean`（+ `Unsqueeze`／
//! `Squeeze`）へ、`Flatten` を `Reshape` へ分解する場合があり、その場合は
//! **別の演算列を経由して同じ計算を行う**（`reference.json` の
//! `exporters.<exporter>.op_types` が実際の演算列を記録する。
//! `tests/fixtures/pytorch-onnx-cnn-ops/README.md` の op 列表を参照）。
//! 本クレートはいずれの分解後 op（`ReduceMean`／`Unsqueeze`／`Squeeze`／
//! `Reshape`／`Shape`／`Slice`／`Concat`／`Constant`）も実装済みのため、
//! 実測ではすべてのケース・両 exporter で `run` が成功した
//! （`ImportError` を期待するケースは無かった。R5 の実測結果）。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use fandhe_ai_onnx_interop::onnx::graph::{RawTensor, build_graph};
use fandhe_ai_onnx_interop::onnx::interp::{Value, run};
use fandhe_ai_onnx_interop::onnx::proto;
use fandhe_ai_tensor_core::Tensor;
use serde::Deserialize;

/// 読み込みを許容するファイルサイズ上限（16 MiB。fixture 全体〈272 KB
/// 実測〉に対して十分な余裕を持たせつつ、細工・破損した巨大ファイルの
/// 丸ごと読み込みによるメモリ枯渇（A03）を防ぐ。
/// `tests/model_zoo_parity.rs::MAX_READ_BYTES`〈1 GiB〉より厳しい値を
/// 使うのは、本 fixture が小さいテンソルのみを扱う設計（実装計画 §2）で
/// あり、上限を実際のサイズへ近づけるほど異常な入力を早期に弾けるため。
const MAX_READ_BYTES: u64 = 16 * 1024 * 1024;

/// 上限付きでファイル全体を読み込む（`model_zoo_parity.rs::
/// read_file_bounded_with_limit` と同型。TOCTOU 回避のため `metadata`
/// 取得と読み込みを同一 `File` ハンドルに対して行い、事前の `len` 検査に
/// 加えて `take(max_bytes + 1)` による実読込量検査も行う fail-closed
/// 二重防御）。
fn read_file_bounded(path: &Path) -> Vec<u8> {
    use std::io::Read;
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("ファイルオープン失敗: {} ({e})", path.display()));
    let len = file
        .metadata()
        .unwrap_or_else(|e| panic!("メタデータ取得失敗: {} ({e})", path.display()))
        .len();
    assert!(
        len <= MAX_READ_BYTES,
        "ファイルサイズ上限超過: {} ({len} bytes > {MAX_READ_BYTES} bytes)",
        path.display()
    );
    let mut buf = Vec::new();
    let read_len = file
        .take(MAX_READ_BYTES + 1)
        .read_to_end(&mut buf)
        .unwrap_or_else(|e| panic!("読み込み失敗: {} ({e})", path.display()));
    assert!(
        read_len as u64 <= MAX_READ_BYTES,
        "ファイルサイズ上限超過（読み込み時検査）: {}",
        path.display()
    );
    buf
}

// ---- `reference.json` のデシリアライズ用構造体 ----
// （生成側 `gen_reference.py` が書き出す構造を鏡写しにする。未知フィールドは
// `serde` の既定〈無視〉に任せる。外部データのため個々のフィールドは全て
// 明示的に検証してから使う）。

#[derive(Deserialize)]
struct TensorRecord {
    shape: Vec<i64>,
    bits: Vec<u32>,
}

#[derive(Deserialize)]
struct ExporterRecord {
    #[serde(default)]
    onnx_file: Option<String>,
    #[serde(default)]
    op_types: Option<Vec<String>>,
    #[serde(default)]
    name_map: Option<HashMap<String, String>>,
    /// `name_map` に対応が見つからなかった initializer 名（`gen_reference.
    /// py::export_one` が「隠さず unmapped として記録する」方針で書き出す
    /// フィールド。R2 検査で `graph.initializers` の全件が `name_map` と
    /// 本フィールドのいずれかに過不足なく属することを検証するために使う
    /// レビュー指摘対応。イシュー #2329 PR #2343）。
    #[serde(default)]
    unmapped_initializers: Option<Vec<String>>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
struct CaseRecord {
    input: TensorRecord,
    output: TensorRecord,
    state_dict: HashMap<String, TensorRecord>,
    exporters: HashMap<String, ExporterRecord>,
}

#[derive(Deserialize)]
struct ReferenceDoc {
    cases: HashMap<String, CaseRecord>,
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pytorch-onnx-cnn-ops")
}

fn load_reference() -> ReferenceDoc {
    let root = fixture_root();
    let bytes = read_file_bounded(&root.join("reference.json"));
    serde_json::from_slice(&bytes).expect("reference.json の parse に失敗した")
}

/// `TensorRecord`（u32 bit パターン列）を `Tensor<f32>` へ復元する。
/// 長さと shape の要素数一致を先に検査してから `f32::from_bits` で復元する
/// （`model_zoo_parity.rs::load_tensor_pb` と同じ「復号より前に整合を
/// 検査する」順序。A03）。
fn tensor_from_record(rec: &TensorRecord) -> Tensor<f32> {
    let mut expected: usize = 1;
    for &d in &rec.shape {
        assert!(d >= 0, "負の shape 次元: {d}");
        expected = expected
            .checked_mul(d as usize)
            .unwrap_or_else(|| panic!("shape 要素数オーバーフロー: {:?}", rec.shape));
    }
    assert_eq!(
        rec.bits.len(),
        expected,
        "bits 長と shape 要素数が不一致（shape={:?}）",
        rec.shape
    );
    let data: Vec<f32> = rec.bits.iter().map(|&b| f32::from_bits(b)).collect();
    let shape: Vec<usize> = rec.shape.iter().map(|&d| d as usize).collect();
    Tensor::new(data, &shape).expect("Tensor::new 失敗（reference.json の shape 不整合）")
}

/// `RawTensor::F32` を `(&[f32], &[i64])` へ落とす。initializer が
/// FLOAT 以外（`Reshape`/`ReduceMean` の shape 定数は INT64）の場合は
/// `None` を返す（R2 は重み系 initializer のみが対象）。
fn raw_f32(t: &RawTensor) -> Option<(&[f32], &[i64])> {
    match t {
        RawTensor::F32 { data, shape } => Some((data, shape)),
        _ => None,
    }
}

/// R2: `graph.initializers` のうち `name_map`（onnx 初期化子名 -> PyTorch
/// `state_dict` キー）に載っているものが、対応する `state_dict` テンソルと
/// bit 完全一致することを検査する。`unmapped_initializers`（Reshape の
/// shape 定数等、モデルパラメータではない initializer）は対象外
/// （`gen_reference.py::export_one` の値ベース対応付けの生成時コメント
/// 参照）。
///
/// **網羅性検査（codex-review 指摘対応。イシュー #2329 PR #2343）**:
/// `name_map` のみを検査する素朴な実装は、対応表が空・一部欠落していても
/// 何も検査せず素通りしてしまう（R2 の「重みの bit 一致」を実質検証しない
/// blind spot）。これを防ぐため、値の突合に先立って次の 2 点を fail-closed
/// に検証する。
/// 1. `name_map` の値（`state_dict` キー側）が `case.state_dict` の全キーと
///    重複なく一対一対応すること（`state_dict` の重みが 1 つも漏れず・
///    2 重に数えられず検査対象になることの保証）。ただし `*.num_batches_
///    tracked`（`nn.BatchNorm*d` が eval 時には参照しない整数バッファで、
///    ONNX グラフへ export されない PyTorch 既知の仕様。実測で
///    `bn*_eval*` ケース全件が該当）は本検査の対象から除外する
///    （`gen_reference.py::tensor_record` が dtype を問わず `f32` へ
///    キャストして記録するため、除外しないと ONNX 側に対応物が存在しない
///    バッファまで bit 一致検査対象に含めることを要求してしまい誤検知に
///    なる）
/// 2. `graph.initializers` の全キーが `name_map`（検査対象）と
///    `unmapped_initializers`（意図的に対象外と記録済み）のいずれか一方に
///    過不足なく属すること（初期化子が静かに検査対象からも記録からも
///    漏れるのを防ぐ）
fn assert_r2_initializers_match_state_dict(
    graph: &fandhe_ai_onnx_interop::onnx::graph::Graph,
    case: &CaseRecord,
    exporter: &ExporterRecord,
    case_name: &str,
    exporter_name: &str,
) {
    let name_map = exporter
        .name_map
        .as_ref()
        .unwrap_or_else(|| panic!("{case_name} [{exporter_name}]: name_map が無い"));

    // 検査 1: name_map の値（state_dict キー）が state_dict の全キーと
    // 重複なく一対一対応することを検査する（bijection）。これにより
    // name_map が空・一部欠落しているケースは state_dict が非空である限り
    // ここで fail する（素通りを防ぐ）。
    let mut mapped_sd_keys: Vec<&String> = name_map.values().collect();
    mapped_sd_keys.sort();
    let mapped_sd_keys_unique_count = {
        let mut dedup = mapped_sd_keys.clone();
        dedup.dedup();
        dedup.len()
    };
    assert_eq!(
        mapped_sd_keys.len(),
        mapped_sd_keys_unique_count,
        "{case_name} [{exporter_name}]: name_map の値（state_dict キー）に重複がある: \
         {mapped_sd_keys:?}"
    );
    // `num_batches_tracked` は nn.BatchNorm*d の eval 時未使用バッファで
    // ONNX へ export されない（PyTorch 既知の仕様）ため、必須対応の対象
    // から除外する（このコメント直上のドキュメンテーションコメント参照）。
    let mut state_dict_keys: Vec<&String> = case
        .state_dict
        .keys()
        .filter(|k| !k.ends_with(".num_batches_tracked") && *k != "num_batches_tracked")
        .collect();
    state_dict_keys.sort();
    assert_eq!(
        mapped_sd_keys, state_dict_keys,
        "{case_name} [{exporter_name}]: name_map の対応先（state_dict キー集合。\
         num_batches_tracked を除く）が case.state_dict の全キーと一致しない \
         （漏れ・過剰のいずれか）。mapped={mapped_sd_keys:?} state_dict={state_dict_keys:?}"
    );

    // 検査 2: graph.initializers の全キーが name_map（検査対象）と
    // unmapped_initializers（意図的に対象外と記録済み）のいずれか一方に
    // 過不足なく属することを検査する（初期化子の静かな取りこぼしを防ぐ）。
    let unmapped = exporter
        .unmapped_initializers
        .as_ref()
        .unwrap_or_else(|| panic!("{case_name} [{exporter_name}]: unmapped_initializers が無い"));
    let mut accounted: Vec<&String> = name_map.keys().chain(unmapped.iter()).collect();
    accounted.sort();
    let accounted_unique_count = {
        let mut dedup = accounted.clone();
        dedup.dedup();
        dedup.len()
    };
    assert_eq!(
        accounted.len(),
        accounted_unique_count,
        "{case_name} [{exporter_name}]: name_map と unmapped_initializers の間で \
         initializer 名が重複している: {accounted:?}"
    );
    let mut graph_init_names: Vec<&String> = graph.initializers.keys().collect();
    graph_init_names.sort();
    assert_eq!(
        accounted, graph_init_names,
        "{case_name} [{exporter_name}]: graph.initializers の全件が name_map・\
         unmapped_initializers のいずれにも過不足なく属さない（取りこぼし検出）。\
         accounted={accounted:?} graph={graph_init_names:?}"
    );

    for (onnx_name, sd_key) in name_map {
        let raw = graph.initializers.get(onnx_name).unwrap_or_else(|| {
            panic!(
                "{case_name} [{exporter_name}]: initializer '{onnx_name}' が \
                 build_graph 後に存在しない"
            )
        });
        let (data, shape) = raw_f32(raw).unwrap_or_else(|| {
            panic!("{case_name} [{exporter_name}]: initializer '{onnx_name}' は F32 以外")
        });
        let sd_rec = case.state_dict.get(sd_key).unwrap_or_else(|| {
            panic!(
                "{case_name} [{exporter_name}]: state_dict に '{sd_key}' が無い \
                 （reference.json の不整合）"
            )
        });
        assert_eq!(
            shape, &sd_rec.shape,
            "{case_name} [{exporter_name}]: '{onnx_name}'（state_dict '{sd_key}'）の shape 不一致"
        );
        assert_eq!(
            data.len(),
            sd_rec.bits.len(),
            "{case_name} [{exporter_name}]: '{onnx_name}' の要素数不一致"
        );
        for (i, (&d, &want_bits)) in data.iter().zip(sd_rec.bits.iter()).enumerate() {
            assert_eq!(
                d.to_bits(),
                want_bits,
                "{case_name} [{exporter_name}]: R2 bit 不一致 '{onnx_name}'[{i}] \
                 (actual={d}, expected={})",
                f32::from_bits(want_bits)
            );
        }
    }
}

/// 出力の突合結果（bit 不一致要素数・max_abs_diff・max_rel_err・
/// mean_abs_diff）。`mean_abs_diff` は `f64` で蓄積してから要素数で割る
/// （`crates/backend-cuda/tests/common/parity_baseline.rs::ParityBaseline`
/// の `baseline_mean_abs_diff_ceiling` と同じ「集計値は `f64` で確定して
/// から比較する」設計）。
struct DiffStats {
    /// 比較対象の総要素数（`ReductionBaseline::total` との完全一致検査に使う）。
    total: usize,
    fail_count: usize,
    bit_mismatch_count: usize,
    max_abs_diff: f32,
    max_rel_err: f32,
    mean_abs_diff: f64,
}

fn diff_stats(actual: &[f32], expected: &[f32]) -> DiffStats {
    assert_eq!(actual.len(), expected.len(), "要素数不一致");
    let total = actual.len();
    let mut fail_count = 0usize;
    let mut bit_mismatch_count = 0usize;
    let mut max_abs_diff = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut abs_diff_sum = 0.0f64;
    for (&a, &e) in actual.iter().zip(expected.iter()) {
        if a.to_bits() != e.to_bits() {
            bit_mismatch_count += 1;
        }
        let abs_diff = (a - e).abs();
        if abs_diff.is_finite() && abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
        abs_diff_sum += f64::from(abs_diff);
        let rel_err = abs_diff / (e.abs() + 1e-6);
        if rel_err.is_finite() {
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }
        } else {
            max_rel_err = f32::INFINITY;
        }
        // fail-closed: 非有限 rel_err・1e-3 超過のいずれも fail 扱い
        // （`model_zoo_parity.rs::assert_req7` と同じ規律）。
        if !rel_err.is_finite() || rel_err > 1e-3 {
            fail_count += 1;
        }
    }
    let mean_abs_diff = abs_diff_sum / total as f64;
    DiffStats {
        total,
        fail_count,
        bit_mismatch_count,
        max_abs_diff,
        max_rel_err,
        mean_abs_diff,
    }
}

/// 縮約を伴う演算（PyTorch 実行系との結合順序差により bit 完全一致を
/// 目標にできない op 種別）。[`Expectation`] の割当が [`EXPECTATIONS`] の
/// 表書き換えで無断に緩和・厳格化されていないかを
/// [`assert_expectation_matches_op_types`] が機械検査する際の判定基準
/// として使う（モジュール doc「判定方式についての注記」参照）。
const REDUCTION_OP_TYPES: &[&str] = &[
    "Conv",
    "AveragePool",
    "GlobalAveragePool",
    "BatchNormalization",
    "ReduceMean",
];

/// 出力の判定方式。モジュール doc「判定方式についての注記」参照
/// （2026-09-28 ユーザー承認で正式方式へ移行）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Expectation {
    /// 選択・形状操作（`MaxPool`・`Flatten`）。フォールバックなしの bit 一致のみ。
    /// 縮約系 op（[`REDUCTION_OP_TYPES`]）を一切含まない演算列にのみ適用する。
    BitExact,
    /// 縮約系（`Conv`・`AveragePool`・`GlobalAveragePool`・
    /// `BatchNormalization`。dynamo 分解経路の `ReduceMean` を含む）。
    /// PyTorch 実行系との結合順序差により bit 完全一致は原理的に目標に
    /// できないため、縮約系の受け入れ条件から bit 同一を外し、次の併用
    /// 方式で判定する（2026-09-28 ユーザー承認）:
    ///
    /// 1. REQ-7 事前固定式 `abs_err / (|ref| + 1e-6) <= 1e-3` の
    ///    `fail_count == 0`（必須条件。tolerance は変更しない）
    /// 2. ケースごとの実測上限 baseline（[`ReductionBaseline`]）に対する
    ///    fail-closed 非後退判定（`total`・`fail_count`・`max_abs_diff`・
    ///    `max_rel_err`・`mean_abs_diff` のいずれも記録済み値を超えないこと）
    ///
    /// 縮約系 op を 1 つ以上含む演算列にのみ適用する
    /// （[`assert_expectation_matches_op_types`] 参照）。
    Req7BaselineNonRegression,
}

/// `(case_name, exporter_name) -> Expectation` の期待値表。
/// 実測（本 PR 作成時に `--nocapture` で確認した結果。
/// `docs/perf/logs/onnx-cnn-ops-pytorch-fixture-2329/README.md` に転記済み）
/// に基づき固定する。表と `reference.json` のケース集合が完全一致することを
/// [`expectation_table_matches_reference_cases`] で検査し、取りこぼしを
/// fail-closed に止める。
const EXPECTATIONS: &[(&str, &str, Expectation)] = &[
    // `Conv` は縮約系（REDUCTION_OP_TYPES）のため Req7BaselineNonRegression
    // を適用する（bit 一致は縮約系の受け入れ条件から外れた。モジュール doc
    // 「判定方式についての注記」参照）。実測で bit 一致したケースも baseline
    // の ceiling を 0 として記録する（緩和ではなく、成立した厳しさをそのまま
    // 非後退契約に固定する）。
    ("conv2d_basic", "ts", Expectation::Req7BaselineNonRegression),
    (
        "conv2d_basic",
        "dynamo",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "conv2d_stride_dil_group",
        "ts",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "conv2d_stride_dil_group",
        "dynamo",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "conv2d_nobias",
        "ts",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "conv2d_nobias",
        "dynamo",
        Expectation::Req7BaselineNonRegression,
    ),
    ("conv1d_basic", "ts", Expectation::Req7BaselineNonRegression),
    (
        "conv1d_basic",
        "dynamo",
        Expectation::Req7BaselineNonRegression,
    ),
    ("maxpool2d_basic", "ts", Expectation::BitExact),
    ("maxpool2d_basic", "dynamo", Expectation::BitExact),
    ("maxpool2d_pad_dil_ceil", "ts", Expectation::BitExact),
    ("maxpool2d_pad_dil_ceil", "dynamo", Expectation::BitExact),
    ("maxpool1d_basic", "ts", Expectation::BitExact),
    ("maxpool1d_basic", "dynamo", Expectation::BitExact),
    (
        "avgpool2d_include_pad",
        "ts",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "avgpool2d_include_pad",
        "dynamo",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "avgpool2d_exclude_pad",
        "ts",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "avgpool2d_exclude_pad",
        "dynamo",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "avgpool2d_ceil_overhang_incl",
        "ts",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "avgpool2d_ceil_overhang_incl",
        "dynamo",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "avgpool2d_ceil_overhang_excl",
        "ts",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "avgpool2d_ceil_overhang_excl",
        "dynamo",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "avgpool1d_basic",
        "ts",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "avgpool1d_basic",
        "dynamo",
        Expectation::Req7BaselineNonRegression,
    ),
    ("gap2d", "ts", Expectation::Req7BaselineNonRegression),
    ("gap2d", "dynamo", Expectation::Req7BaselineNonRegression),
    ("gap1d", "ts", Expectation::Req7BaselineNonRegression),
    ("gap1d", "dynamo", Expectation::Req7BaselineNonRegression),
    ("bn2d_eval", "ts", Expectation::Req7BaselineNonRegression),
    (
        "bn2d_eval",
        "dynamo",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "bn2d_eval_eps",
        "ts",
        Expectation::Req7BaselineNonRegression,
    ),
    (
        "bn2d_eval_eps",
        "dynamo",
        Expectation::Req7BaselineNonRegression,
    ),
    ("bn1d_eval", "ts", Expectation::Req7BaselineNonRegression),
    (
        "bn1d_eval",
        "dynamo",
        Expectation::Req7BaselineNonRegression,
    ),
    ("flatten_default", "ts", Expectation::BitExact),
    ("flatten_default", "dynamo", Expectation::BitExact),
    ("flatten_start2", "ts", Expectation::BitExact),
    ("flatten_start2", "dynamo", Expectation::BitExact),
];

/// `(case_name, exporter_name)` に対応する [`Expectation`] を `EXPECTATIONS`
/// から引く単一の真実源（レビュー指摘対応。イシュー #2329）。
/// `fixture_test!` マクロはこの関数を経由するため、各テスト関数呼び出し側で
/// `Expectation` を個別に指定する必要がなくなり、表の更新とテストの判定方式が
/// 構造的にズレなくなる（表を更新してもマクロ呼び出し側の引数を直し忘れて
/// 検出されない、という取りこぼし経路を排除する）。
fn expectation_for(case_name: &str, exporter_name: &str) -> Expectation {
    EXPECTATIONS
        .iter()
        .find(|(c, e, _)| *c == case_name && *e == exporter_name)
        .unwrap_or_else(|| {
            panic!("EXPECTATIONS に '{case_name}' [{exporter_name}] のエントリが無い")
        })
        .2
}

/// 期待値表と `reference.json` のケース集合が過不足なく一致することを
/// 検査する（`model_zoo_parity.rs` の思想と同じ。ケースの取りこぼしを
/// fail-closed に止める）。
#[test]
fn expectation_table_matches_reference_cases() {
    let doc = load_reference();
    let mut table_keys: Vec<(String, String)> = EXPECTATIONS
        .iter()
        .map(|(c, e, _)| (c.to_string(), e.to_string()))
        .collect();
    table_keys.sort();
    let mut ref_keys: Vec<(String, String)> = Vec::new();
    for (case_name, case) in &doc.cases {
        for exporter_name in case.exporters.keys() {
            ref_keys.push((case_name.clone(), exporter_name.clone()));
        }
    }
    ref_keys.sort();
    assert_eq!(
        table_keys, ref_keys,
        "EXPECTATIONS と reference.json のケース集合が不一致（取りこぼし検出）"
    );
}

/// 1 ケース・1 exporter を decode → build_graph → run し、[`DiffStats`]
/// までを計算する（判定方式に依存しない共通部分）。[`run_case`] から呼ばれる
/// 単一の真実源（レビュー指摘対応。イシュー #2329 PR #2343。判定ロジックを
/// 重複実装すると decode／run 経路がズレるリスクがあるため一本化する）。
fn compute_case_stats(case_name: &str, exporter_name: &str, case: &CaseRecord) -> DiffStats {
    let exporter = case
        .exporters
        .get(exporter_name)
        .unwrap_or_else(|| panic!("{case_name} [{exporter_name}]: reference.json に無い"));
    assert!(
        exporter.error.is_none(),
        "{case_name} [{exporter_name}]: fixture 生成時に export が失敗している \
         （{:?}）。生成側の問題のため fixture を再生成すること",
        exporter.error
    );
    let onnx_file = exporter
        .onnx_file
        .as_deref()
        .unwrap_or_else(|| panic!("{case_name} [{exporter_name}]: onnx_file が無い"));
    // fixture ディレクトリ直下に閉じる（`/`・`..` を含む名前は拒否。A03）。
    assert!(
        !onnx_file.contains('/') && !onnx_file.contains('\\') && !onnx_file.contains(".."),
        "{case_name} [{exporter_name}]: onnx_file が不正なパスを含む: {onnx_file}"
    );
    let onnx_path = fixture_root().join(onnx_file);
    let bytes = read_file_bounded(&onnx_path);
    let model = proto::decode_model(&bytes)
        .unwrap_or_else(|e| panic!("{case_name} [{exporter_name}]: decode 失敗: {e}"));
    let graph = build_graph(&model)
        .unwrap_or_else(|e| panic!("{case_name} [{exporter_name}]: build_graph 失敗: {e}"));

    // `exporter.op_types` は生成側（`gen_reference.py`）が export 直後に記録した
    // 期待 op 列であり、これまでは `eprintln!` によるログ出力にしか使っていなかった
    // （レビュー指摘。イシュー #2329 PR #2343。対象 op〈Conv・Pool・BN・Flatten〉の
    // 実行が別演算列へ回帰しても、参照出力さえ一致すれば fixture テストが検出でき
    // ない状態だった）。ここで実際に decode した `graph.nodes` の `op_type` 列と
    // 突合し、fixture が意図した op 列のまま保たれていることを検証する。
    // `op_types` 欠落を無言で検査省略しない（既出の R2 initializer 対応表が空でも
    // 通過していた指摘と同型の「無言 skip」を避けるため、現行 39 レコード全件が
    // 値を持つ前提を `unwrap_or_else` で強制する。欠落は生成側の問題として fail）。
    let expected_op_types = exporter.op_types.as_ref().unwrap_or_else(|| {
        panic!(
            "{case_name} [{exporter_name}]: reference.json に op_types が無い \
             （生成側の問題のため fixture を再生成すること）"
        )
    });
    let actual_op_types: Vec<&str> = graph.nodes.iter().map(|n| n.op_type.as_str()).collect();
    let expected_op_types_ref: Vec<&str> = expected_op_types.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        actual_op_types, expected_op_types_ref,
        "{case_name} [{exporter_name}]: decode したグラフの op_type 列が \
         reference.json の記録と一致しない（対象 op の実行経路が別の演算列へ \
         置き換わっている可能性）。actual={actual_op_types:?} \
         expected={expected_op_types_ref:?}"
    );

    assert_r2_initializers_match_state_dict(&graph, case, exporter, case_name, exporter_name);

    let input_tensor = tensor_from_record(&case.input);
    let mut feeds: HashMap<String, Value> = HashMap::new();
    feeds.insert("x".to_string(), Value::F32(input_tensor));

    let result = run(&graph, feeds)
        .unwrap_or_else(|e| panic!("{case_name} [{exporter_name}]: run 失敗: {e}"));
    let actual = match result.get("y") {
        Some(Value::F32(t)) => t,
        Some(other) => panic!("{case_name} [{exporter_name}]: 出力が F32 以外: {other:?}"),
        None => panic!("{case_name} [{exporter_name}]: 出力 'y' が無い"),
    };

    let expected_tensor = tensor_from_record(&case.output);
    assert_eq!(
        actual.shape(),
        expected_tensor.shape(),
        "{case_name} [{exporter_name}]: 出力 shape 不一致"
    );
    let actual_slice = actual.as_slice().expect("actual as_slice 失敗");
    let expected_slice = expected_tensor.as_slice().expect("expected as_slice 失敗");
    let stats = diff_stats(actual_slice, expected_slice);

    eprintln!(
        "{case_name} [{exporter_name}]: op_types={:?} bit_mismatch={}/{} max_abs_diff={:?} \
         max_rel_err={:?} req7_fail_count={} mean_abs_diff={:?}",
        exporter.op_types,
        stats.bit_mismatch_count,
        actual_slice.len(),
        stats.max_abs_diff,
        stats.max_rel_err,
        stats.fail_count,
        stats.mean_abs_diff,
    );

    stats
}

/// [`Expectation`] の割当が [`EXPECTATIONS`] の表書き換えで無断に緩和・
/// 厳格化されていないかを機械検査する（`op_types` 列に縮約系 op
/// （[`REDUCTION_OP_TYPES`]）を 1 つでも含む場合は
/// [`Expectation::Req7BaselineNonRegression`]、含まない場合は
/// [`Expectation::BitExact`] でなければならない）。表の手動編集で縮約系
/// ケースを `BitExact` 側へ動かす・あるいは純粋な選択・形状操作ケースを
/// `Req7BaselineNonRegression` 側へ動かす、という双方向の書き換えミスを
/// fail-closed に検出する（レビュー指摘の再発防止。「R2 対応表が空でも
/// 通過する」「op_types が未検査」と同型の構造的な見逃しを防ぐ狙い）。
fn assert_expectation_matches_op_types(
    case_name: &str,
    exporter_name: &str,
    expectation: Expectation,
    op_types: &[String],
) {
    let has_reduction_op = op_types
        .iter()
        .any(|op| REDUCTION_OP_TYPES.contains(&op.as_str()));
    match expectation {
        Expectation::BitExact => assert!(
            !has_reduction_op,
            "{case_name} [{exporter_name}]: 縮約系 op（{op_types:?}）を含む演算列に \
             Expectation::BitExact が割り当てられている（EXPECTATIONS の誤り。縮約系は \
             Req7BaselineNonRegression を使うこと）"
        ),
        Expectation::Req7BaselineNonRegression => assert!(
            has_reduction_op,
            "{case_name} [{exporter_name}]: 縮約系 op を含まない演算列（{op_types:?}）に \
             Expectation::Req7BaselineNonRegression が割り当てられている（EXPECTATIONS の \
             誤り。純粋な選択・形状操作は BitExact を使うこと）"
        ),
    }
}

/// 1 ケース・1 exporter を decode → build_graph → run → 判定まで実行する。
/// 判定は [`compute_case_stats`] が返す [`DiffStats`] に対して
/// [`Expectation`] ごとの合否基準を適用する（モジュール doc「判定方式に
/// ついての注記」参照。2026-09-28 ユーザー承認で正式方式へ移行）。
fn run_case(case_name: &str, exporter_name: &str, case: &CaseRecord, expectation: Expectation) {
    let op_types = case
        .exporters
        .get(exporter_name)
        .and_then(|e| e.op_types.as_ref())
        .unwrap_or_else(|| panic!("{case_name} [{exporter_name}]: op_types が無い"));
    assert_expectation_matches_op_types(case_name, exporter_name, expectation, op_types);

    let stats = compute_case_stats(case_name, exporter_name, case);

    match expectation {
        Expectation::BitExact => {
            assert_eq!(
                stats.bit_mismatch_count, 0,
                "{case_name} [{exporter_name}]: bit 不一致（BitExact 期待。\
                 選択・形状操作でフォールバックは許容しない）"
            );
        }
        Expectation::Req7BaselineNonRegression => {
            // 必須条件 1: REQ-7 事前固定式（tolerance は緩めない
            // `.claude/rules/coding-rust.md`）。fail-closed な回帰ガードで
            // あり、これ単独では非後退契約にならない（baseline 判定を
            // 併用する）。
            assert_eq!(
                stats.fail_count, 0,
                "{case_name} [{exporter_name}]: REQ-7 事前固定式で fail \
                 （max_rel_err={}）。tolerance は緩めない（`.claude/rules/\
                 coding-rust.md`）。ユーザー判断が必要",
                stats.max_rel_err
            );
            // 必須条件 2: ケースごとの実測上限 baseline に対する fail-closed
            // 非後退判定（モジュール doc「判定方式についての注記」参照）。
            let baseline = find_reduction_baseline(case_name, exporter_name);
            assert_no_reduction_baseline_regression(case_name, exporter_name, &stats, baseline);
        }
    }
}

/// `(case_name, exporter_name)` の 2 引数のみを取る。判定方式（[`Expectation`]）
/// は呼び出し側で個別指定せず、[`expectation_for`] 経由で `EXPECTATIONS`
/// （単一の真実源）から引く（レビュー指摘対応。イシュー #2329。以前は
/// 呼び出し側にも `Expectation` を渡していたため、表を更新してもここを
/// 直し忘れると検出されないズレが生じ得た）。
macro_rules! fixture_test {
    ($fn_name:ident, $case_name:literal, $exporter_name:literal) => {
        #[test]
        fn $fn_name() {
            let doc = load_reference();
            let case = doc
                .cases
                .get($case_name)
                .unwrap_or_else(|| panic!("reference.json に '{}' が無い", $case_name));
            let expectation = expectation_for($case_name, $exporter_name);
            run_case($case_name, $exporter_name, case, expectation);
        }
    };
}

fixture_test!(conv2d_basic_ts, "conv2d_basic", "ts");
fixture_test!(conv2d_basic_dynamo, "conv2d_basic", "dynamo");
fixture_test!(conv2d_stride_dil_group_ts, "conv2d_stride_dil_group", "ts");
fixture_test!(
    conv2d_stride_dil_group_dynamo,
    "conv2d_stride_dil_group",
    "dynamo"
);
fixture_test!(conv2d_nobias_ts, "conv2d_nobias", "ts");
fixture_test!(conv2d_nobias_dynamo, "conv2d_nobias", "dynamo");
fixture_test!(conv1d_basic_ts, "conv1d_basic", "ts");
fixture_test!(conv1d_basic_dynamo, "conv1d_basic", "dynamo");
fixture_test!(maxpool2d_basic_ts, "maxpool2d_basic", "ts");
fixture_test!(maxpool2d_basic_dynamo, "maxpool2d_basic", "dynamo");
fixture_test!(maxpool2d_pad_dil_ceil_ts, "maxpool2d_pad_dil_ceil", "ts");
fixture_test!(
    maxpool2d_pad_dil_ceil_dynamo,
    "maxpool2d_pad_dil_ceil",
    "dynamo"
);
fixture_test!(maxpool1d_basic_ts, "maxpool1d_basic", "ts");
fixture_test!(maxpool1d_basic_dynamo, "maxpool1d_basic", "dynamo");
fixture_test!(avgpool2d_include_pad_ts, "avgpool2d_include_pad", "ts");
fixture_test!(
    avgpool2d_include_pad_dynamo,
    "avgpool2d_include_pad",
    "dynamo"
);
fixture_test!(avgpool2d_exclude_pad_ts, "avgpool2d_exclude_pad", "ts");
fixture_test!(
    avgpool2d_exclude_pad_dynamo,
    "avgpool2d_exclude_pad",
    "dynamo"
);
fixture_test!(
    avgpool2d_ceil_overhang_incl_ts,
    "avgpool2d_ceil_overhang_incl",
    "ts"
);
fixture_test!(
    avgpool2d_ceil_overhang_incl_dynamo,
    "avgpool2d_ceil_overhang_incl",
    "dynamo"
);
fixture_test!(
    avgpool2d_ceil_overhang_excl_ts,
    "avgpool2d_ceil_overhang_excl",
    "ts"
);
fixture_test!(
    avgpool2d_ceil_overhang_excl_dynamo,
    "avgpool2d_ceil_overhang_excl",
    "dynamo"
);
fixture_test!(avgpool1d_basic_ts, "avgpool1d_basic", "ts");
fixture_test!(avgpool1d_basic_dynamo, "avgpool1d_basic", "dynamo");
fixture_test!(gap2d_ts, "gap2d", "ts");
fixture_test!(gap2d_dynamo, "gap2d", "dynamo");
fixture_test!(gap1d_ts, "gap1d", "ts");
fixture_test!(gap1d_dynamo, "gap1d", "dynamo");
fixture_test!(bn2d_eval_ts, "bn2d_eval", "ts");
fixture_test!(bn2d_eval_dynamo, "bn2d_eval", "dynamo");
fixture_test!(bn2d_eval_eps_ts, "bn2d_eval_eps", "ts");
fixture_test!(bn2d_eval_eps_dynamo, "bn2d_eval_eps", "dynamo");
fixture_test!(bn1d_eval_ts, "bn1d_eval", "ts");
fixture_test!(bn1d_eval_dynamo, "bn1d_eval", "dynamo");
fixture_test!(flatten_default_ts, "flatten_default", "ts");
fixture_test!(flatten_default_dynamo, "flatten_default", "dynamo");
fixture_test!(flatten_start2_ts, "flatten_start2", "ts");
fixture_test!(flatten_start2_dynamo, "flatten_start2", "dynamo");

// ==== 縮約系（Expectation::Req7BaselineNonRegression）の実測上限 baseline ====
//
// `crates/backend-cuda/tests/common/parity_baseline.rs::ParityBaseline` と
// 同じ fail-closed 設計（実測値のみ・追加更新は人間承認必須・ケース集合の
// 完全一致検査・NaN／負値拒否）を、ONNX PyTorch fixture 突合向けに移植する。
// tolerance 定数の変更ではなく、REQ-7 事前固定式を通過した後の集計結果
// （fail_count・max_abs_diff・max_rel_err・mean_abs_diff）が既知の実測値
// から悪化していないかを見る非後退の上限である
// （`.claude/rules/coding-rust.md` の「勾配の長軸縮約」節とは無関係の
// 別契約）。

/// 縮約系 1 ケース・1 exporter の記録済み実測上限 baseline。
///
/// ceiling は実測値そのもの（余裕係数を掛けない。`docs/onnx-pytorch-fixture-
/// reduction-parity-judgment-decision.md` 2026-09-28 ユーザー承認）。
/// `baseline_fail_count` は REQ-7 事前固定式での fail 要素数であり、
/// `run_case` が別途 `stats.fail_count == 0` を必須条件として検査済みの
/// ため本 baseline でも常に `0` を要求する（`reduction_baselines_are_well_
/// formed` が機械検査する）。
#[derive(Debug, Clone, Copy)]
struct ReductionBaseline {
    case_name: &'static str,
    exporter_name: &'static str,
    total: usize,
    baseline_fail_count: usize,
    baseline_max_abs_diff_ceiling: f32,
    baseline_max_rel_err_ceiling: f32,
    baseline_mean_abs_diff_ceiling: f64,
}

/// 記録済み baseline 一覧（28 行 = 縮約系 14 ケース × 2 exporter）。
///
/// 出典: 本 PR 作成時に `cargo test -p fandhe-ai-onnx-interop --test \
/// onnx_interp_pytorch_cnn_fixture -- --nocapture --test-threads=1` で実測
/// した値（`docs/perf/logs/onnx-cnn-ops-pytorch-fixture-2329/README.md` に
/// 転記済み）。計算経路は CPU feature 検出・`rayon` 並列に依存しない決定的
/// 逐次ループのため、CI（ubuntu-latest x86_64）・ローカル（x86_64）を問わず
/// 単一の値で成立する（モジュール doc「計算経路の決定性」参照）。
static REDUCTION_BASELINES: &[ReductionBaseline] = &[
    ReductionBaseline {
        case_name: "conv2d_basic",
        exporter_name: "ts",
        total: 256,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 0.0,
        baseline_max_rel_err_ceiling: 0.0,
        baseline_mean_abs_diff_ceiling: 0.0,
    },
    ReductionBaseline {
        case_name: "conv2d_basic",
        exporter_name: "dynamo",
        total: 256,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 0.0,
        baseline_max_rel_err_ceiling: 0.0,
        baseline_mean_abs_diff_ceiling: 0.0,
    },
    ReductionBaseline {
        case_name: "conv2d_stride_dil_group",
        exporter_name: "ts",
        total: 100,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 1.192_092_9e-7,
        baseline_max_rel_err_ceiling: 2.799_909_5e-6,
        baseline_mean_abs_diff_ceiling: 3.3006072044372556e-8,
    },
    ReductionBaseline {
        case_name: "conv2d_stride_dil_group",
        exporter_name: "dynamo",
        total: 100,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 1.192_092_9e-7,
        baseline_max_rel_err_ceiling: 2.799_909_5e-6,
        baseline_mean_abs_diff_ceiling: 3.3006072044372556e-8,
    },
    ReductionBaseline {
        case_name: "conv2d_nobias",
        exporter_name: "ts",
        total: 256,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 0.0,
        baseline_max_rel_err_ceiling: 0.0,
        baseline_mean_abs_diff_ceiling: 0.0,
    },
    ReductionBaseline {
        case_name: "conv2d_nobias",
        exporter_name: "dynamo",
        total: 256,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 0.0,
        baseline_max_rel_err_ceiling: 0.0,
        baseline_mean_abs_diff_ceiling: 0.0,
    },
    ReductionBaseline {
        case_name: "conv1d_basic",
        exporter_name: "ts",
        total: 20,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 1.192_092_9e-7,
        baseline_max_rel_err_ceiling: 4.256_559e-6,
        baseline_mean_abs_diff_ceiling: 1.862645149230957e-8,
    },
    ReductionBaseline {
        case_name: "conv1d_basic",
        exporter_name: "dynamo",
        total: 20,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 1.192_092_9e-7,
        baseline_max_rel_err_ceiling: 4.256_559e-6,
        baseline_mean_abs_diff_ceiling: 1.862645149230957e-8,
    },
    ReductionBaseline {
        case_name: "avgpool2d_include_pad",
        exporter_name: "ts",
        total: 48,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 5.960_464_5e-8,
        baseline_max_rel_err_ceiling: 4.206_918e-6,
        baseline_mean_abs_diff_ceiling: 1.3812496035825461e-8,
    },
    ReductionBaseline {
        case_name: "avgpool2d_include_pad",
        exporter_name: "dynamo",
        total: 48,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 5.960_464_5e-8,
        baseline_max_rel_err_ceiling: 4.206_918e-6,
        baseline_mean_abs_diff_ceiling: 1.3812496035825461e-8,
    },
    ReductionBaseline {
        case_name: "avgpool2d_exclude_pad",
        exporter_name: "ts",
        total: 48,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 5.960_464_5e-8,
        baseline_max_rel_err_ceiling: 1.254_769_7e-6,
        baseline_mean_abs_diff_ceiling: 1.415416287879149e-8,
    },
    ReductionBaseline {
        case_name: "avgpool2d_exclude_pad",
        exporter_name: "dynamo",
        total: 48,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 5.960_464_5e-8,
        baseline_max_rel_err_ceiling: 1.254_769_7e-6,
        baseline_mean_abs_diff_ceiling: 1.415416287879149e-8,
    },
    ReductionBaseline {
        case_name: "avgpool2d_ceil_overhang_incl",
        exporter_name: "ts",
        total: 48,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 5.960_464_5e-8,
        baseline_max_rel_err_ceiling: 6.191_493e-7,
        baseline_mean_abs_diff_ceiling: 9.216212977965673e-9,
    },
    ReductionBaseline {
        case_name: "avgpool2d_ceil_overhang_incl",
        exporter_name: "dynamo",
        total: 48,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 5.960_464_5e-8,
        baseline_max_rel_err_ceiling: 6.191_493e-7,
        baseline_mean_abs_diff_ceiling: 9.216212977965673e-9,
    },
    ReductionBaseline {
        case_name: "avgpool2d_ceil_overhang_excl",
        exporter_name: "ts",
        total: 48,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 1.192_092_9e-7,
        baseline_max_rel_err_ceiling: 6.804_367_3e-6,
        baseline_mean_abs_diff_ceiling: 1.5056381622950237e-8,
    },
    ReductionBaseline {
        case_name: "avgpool2d_ceil_overhang_excl",
        exporter_name: "dynamo",
        total: 48,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 1.192_092_9e-7,
        baseline_max_rel_err_ceiling: 6.804_367_3e-6,
        baseline_mean_abs_diff_ceiling: 1.5056381622950237e-8,
    },
    ReductionBaseline {
        case_name: "avgpool1d_basic",
        exporter_name: "ts",
        total: 15,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 1.192_092_9e-7,
        baseline_max_rel_err_ceiling: 1.083_659_8e-7,
        baseline_mean_abs_diff_ceiling: 1.5397866566975913e-8,
    },
    ReductionBaseline {
        case_name: "avgpool1d_basic",
        exporter_name: "dynamo",
        total: 15,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 1.192_092_9e-7,
        baseline_max_rel_err_ceiling: 1.083_659_8e-7,
        baseline_mean_abs_diff_ceiling: 1.5397866566975913e-8,
    },
    ReductionBaseline {
        case_name: "gap2d",
        exporter_name: "ts",
        total: 3,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 3.725_290_3e-9,
        baseline_max_rel_err_ceiling: 6.665_004_5e-8,
        baseline_mean_abs_diff_ceiling: 1.241763432820638e-9,
    },
    ReductionBaseline {
        case_name: "gap2d",
        exporter_name: "dynamo",
        total: 3,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 1.490_116_1e-8,
        baseline_max_rel_err_ceiling: 1.999_501_5e-7,
        baseline_mean_abs_diff_ceiling: 1.1175870895385742e-8,
    },
    ReductionBaseline {
        case_name: "gap1d",
        exporter_name: "ts",
        total: 3,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 0.0,
        baseline_max_rel_err_ceiling: 0.0,
        baseline_mean_abs_diff_ceiling: 0.0,
    },
    ReductionBaseline {
        case_name: "gap1d",
        exporter_name: "dynamo",
        total: 3,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 5.960_464_5e-8,
        baseline_max_rel_err_ceiling: 9.228_162_4e-8,
        baseline_mean_abs_diff_ceiling: 2.9802322387695313e-8,
    },
    ReductionBaseline {
        case_name: "bn2d_eval",
        exporter_name: "ts",
        total: 150,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 4.768_371_6e-7,
        baseline_max_rel_err_ceiling: 9.008_323e-7,
        baseline_mean_abs_diff_ceiling: 2.966572841008504e-8,
    },
    ReductionBaseline {
        case_name: "bn2d_eval",
        exporter_name: "dynamo",
        total: 150,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 4.768_371_6e-7,
        baseline_max_rel_err_ceiling: 9.008_323e-7,
        baseline_mean_abs_diff_ceiling: 2.966572841008504e-8,
    },
    ReductionBaseline {
        case_name: "bn2d_eval_eps",
        exporter_name: "ts",
        total: 150,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 2.384_185_8e-7,
        baseline_max_rel_err_ceiling: 6.060_796_5e-7,
        baseline_mean_abs_diff_ceiling: 3.9380975067615506e-8,
    },
    ReductionBaseline {
        case_name: "bn2d_eval_eps",
        exporter_name: "dynamo",
        total: 150,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 2.384_185_8e-7,
        baseline_max_rel_err_ceiling: 6.060_796_5e-7,
        baseline_mean_abs_diff_ceiling: 3.9380975067615506e-8,
    },
    ReductionBaseline {
        case_name: "bn1d_eval",
        exporter_name: "ts",
        total: 42,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 1.192_092_9e-7,
        baseline_max_rel_err_ceiling: 1.286_728_8e-7,
        baseline_mean_abs_diff_ceiling: 2.9979717163812546e-8,
    },
    ReductionBaseline {
        case_name: "bn1d_eval",
        exporter_name: "dynamo",
        total: 42,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 1.192_092_9e-7,
        baseline_max_rel_err_ceiling: 1.286_728_8e-7,
        baseline_mean_abs_diff_ceiling: 2.9979717163812546e-8,
    },
];

/// `(case_name, exporter_name)` に対応する [`ReductionBaseline`] を引く。
/// 未登録ケースは fail-closed に panic する（黙って skip しない）。
#[track_caller]
fn find_reduction_baseline(case_name: &str, exporter_name: &str) -> &'static ReductionBaseline {
    REDUCTION_BASELINES
        .iter()
        .find(|b| b.case_name == case_name && b.exporter_name == exporter_name)
        .unwrap_or_else(|| {
            panic!(
                "{case_name} [{exporter_name}]: REDUCTION_BASELINES に行が無い \
                 （baseline の追加は実測値のみ・人間承認必須。未登録ケースを \
                 黙って通過させない）"
            )
        })
}

/// [`ReductionBaseline`] に対する fail-closed 非後退判定。
/// `crates/backend-cuda/tests/common/parity_baseline.rs::
/// assert_no_parity_regression` と同型（total 完全一致・fail_count／
/// max_abs_diff／max_rel_err／mean_abs_diff のいずれも記録済み値以下）。
#[track_caller]
fn assert_no_reduction_baseline_regression(
    case_name: &str,
    exporter_name: &str,
    stats: &DiffStats,
    baseline: &ReductionBaseline,
) {
    assert_eq!(
        stats.total, baseline.total,
        "{case_name} [{exporter_name}]: 比較対象の要素数が baseline({}) と \
         一致しない（形状・比較対象がずれている可能性）",
        baseline.total,
    );
    assert!(
        stats.fail_count <= baseline.baseline_fail_count,
        "{case_name} [{exporter_name}]: baseline 非後退契約 FAIL — fail_count が \
         後退しました (actual={}, baseline={})",
        stats.fail_count,
        baseline.baseline_fail_count,
    );
    assert!(
        stats.max_abs_diff <= baseline.baseline_max_abs_diff_ceiling,
        "{case_name} [{exporter_name}]: baseline 非後退契約 FAIL — max_abs_diff が \
         後退しました (actual={:?}, ceiling={:?})",
        stats.max_abs_diff,
        baseline.baseline_max_abs_diff_ceiling,
    );
    assert!(
        stats.max_rel_err <= baseline.baseline_max_rel_err_ceiling,
        "{case_name} [{exporter_name}]: baseline 非後退契約 FAIL — max_rel_err が \
         後退しました (actual={:?}, ceiling={:?})",
        stats.max_rel_err,
        baseline.baseline_max_rel_err_ceiling,
    );
    assert!(
        stats.mean_abs_diff <= baseline.baseline_mean_abs_diff_ceiling,
        "{case_name} [{exporter_name}]: baseline 非後退契約 FAIL — mean_abs_diff が \
         後退しました (actual={:?}, ceiling={:?})",
        stats.mean_abs_diff,
        baseline.baseline_mean_abs_diff_ceiling,
    );
}

/// `REDUCTION_BASELINES` の構造的整合性を検査する（`ParityBaseline` と同じ
/// fail-closed 設計。実機実測系の baseline とは異なり通常 CI で常時実行
/// できるため `#[ignore]` を付けない）。
///
/// 1. `(case_name, exporter_name)` の集合が `EXPECTATIONS` の
///    `Req7BaselineNonRegression` エントリと過不足なく一致する（重複行・
///    取りこぼしの検出）
/// 2. 各行の `baseline_fail_count == 0`（REQ-7 式は `run_case` 側で必須条件
///    として別途検査済みのため、baseline 側の記録も 0 以外を許さない）
/// 3. 各行の ceiling は有限・非負（NaN・負値の混入を拒否）
/// 4. 各行の `total > 0`
#[test]
fn reduction_baselines_are_well_formed() {
    let mut expectation_keys: Vec<(&str, &str)> = EXPECTATIONS
        .iter()
        .filter(|(_, _, e)| *e == Expectation::Req7BaselineNonRegression)
        .map(|(c, e, _)| (*c, *e))
        .collect();
    expectation_keys.sort_unstable();

    let mut baseline_keys: Vec<(&str, &str)> = REDUCTION_BASELINES
        .iter()
        .map(|b| (b.case_name, b.exporter_name))
        .collect();
    baseline_keys.sort_unstable();
    let baseline_keys_unique_count = {
        let mut dedup = baseline_keys.clone();
        dedup.dedup();
        dedup.len()
    };
    assert_eq!(
        baseline_keys.len(),
        baseline_keys_unique_count,
        "REDUCTION_BASELINES に重複行がある: {baseline_keys:?}"
    );
    assert_eq!(
        expectation_keys, baseline_keys,
        "REDUCTION_BASELINES と EXPECTATIONS の Req7BaselineNonRegression \
         エントリが不一致（取りこぼし・過剰のいずれか）"
    );

    for b in REDUCTION_BASELINES {
        assert!(
            b.total > 0,
            "{} [{}]: baseline.total が 0",
            b.case_name,
            b.exporter_name
        );
        assert_eq!(
            b.baseline_fail_count, 0,
            "{} [{}]: baseline_fail_count が 0 以外（REQ-7 式は必須条件のため \
             baseline 側も 0 のみを許容する）",
            b.case_name, b.exporter_name
        );
        assert!(
            b.baseline_max_abs_diff_ceiling.is_finite() && b.baseline_max_abs_diff_ceiling >= 0.0,
            "{} [{}]: baseline_max_abs_diff_ceiling が非有限または負値: {:?}",
            b.case_name,
            b.exporter_name,
            b.baseline_max_abs_diff_ceiling
        );
        assert!(
            b.baseline_max_rel_err_ceiling.is_finite() && b.baseline_max_rel_err_ceiling >= 0.0,
            "{} [{}]: baseline_max_rel_err_ceiling が非有限または負値: {:?}",
            b.case_name,
            b.exporter_name,
            b.baseline_max_rel_err_ceiling
        );
        assert!(
            b.baseline_mean_abs_diff_ceiling.is_finite() && b.baseline_mean_abs_diff_ceiling >= 0.0,
            "{} [{}]: baseline_mean_abs_diff_ceiling が非有限または負値: {:?}",
            b.case_name,
            b.exporter_name,
            b.baseline_mean_abs_diff_ceiling
        );
    }
}

// ============================================================================
// external data（外部 `.data` ファイル）fixture 突合（イシュー #2347）
// ============================================================================
//
// `tests/fixtures/pytorch-onnx-external-data/`（PyTorch dynamo exporter の
// external data を再 inline 化しない生出力。`README.md` 参照）を
// `onnx::external_data::build_graph_with_external_data` 経由で読み込み、
// #2329 の `Req7BaselineNonRegression`（`REDUCTION_BASELINES`）と**同じ
// 仕組み**（ケース集合の完全一致検査・`fail_count == 0` を必須条件とした
// うえで `total`／`max_abs_diff`／`max_rel_err`／`mean_abs_diff` の実測値
// そのものを ceiling とする fail-closed 非後退判定）を、本 fixture 専用の
// `EXTERNAL_DATA_BASELINES` で適用する（2026-09-28 ユーザー承認・レビュー
// 対応）。`REDUCTION_BASELINES` の既存行を再利用しない
// （`pytorch-onnx-cnn-ops` fixture と `pytorch-onnx-external-data` fixture
// は生成のたびに重みの実際の bit 列が変わりうる別個のコミット済み
// fixture〈[`ExternalManifestEntry`] のコメント参照〉のため、たとえ
// `case_name` が同じでも重み・参照出力は独立している。テーブルを分離する
// ことで、どちらの fixture を再生成しても互いの baseline を巻き込まずに
// 更新できる）。

use fandhe_ai_onnx_interop::onnx::external_data::{
    ExternalDataOptions, build_graph_with_external_data,
};

/// external data 経由の fixture がある 3 ケース（いずれも `Conv` を含み、
/// 432／288 バイトの重み initializer が external になる。`README.md`
/// 「実測結果」節参照）。
///
/// **`MaxPool`／`Flatten` 等のパラメータを持たない op のケース
/// （`pytorch-onnx-cnn-ops` fixture 側に存在する `maxpool2d_*`・
/// `flatten_*` 等）はここに含まれない**: これらの op は学習可能な重み
/// （`initializer`）を一切持たないため、PyTorch dynamo exporter が
/// external data として切り出す対象（`TensorProto.data_location =
/// EXTERNAL` を持ちうる initializer）自体が存在せず、external data 経由
/// で import する意味のあるケースを構成できない（`gen_external.py` が
/// `pytorch-onnx-cnn-ops` の 19 ケース全件に dynamo export を試みた際も、
/// 実際に initializer が external になったのは `Conv` 系の重みのみだった
/// という実測に基づく。`docs/onnx-external-data-decision.md` §8・
/// fixture 側 `README.md`「実測結果」節も参照）。
const EXTERNAL_DATA_CASE_NAMES: &[&str] =
    &["conv2d_basic", "conv2d_nobias", "conv2d_stride_dil_group"];

fn external_fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pytorch-onnx-external-data")
}

#[derive(Deserialize)]
struct ExternalManifestEntry {
    was_external_data: bool,
    external_initializers: Vec<ExternalManifestInitializer>,
    op_types: Vec<String>,
    /// このスクリプト自身が生成した重みに対する参照入出力（`README.md`
    /// 「実測結果」節参照。`../pytorch-onnx-cnn-ops/reference.json` は
    /// 再利用しない——同じ `deterministic_seed`／`CASES` でも torch の
    /// マイナーバージョン内パッチ差・生成環境差で重み初期化の実際の bit
    /// 列が再現しないことを実測で確認したため、常に自己完結ペアで判定
    /// する）。
    input: TensorRecord,
    output: TensorRecord,
}

#[derive(Deserialize)]
struct ExternalManifestInitializer {
    name: String,
    data_type: i32,
    #[serde(default)]
    location: Option<String>,
}

fn load_external_manifest() -> HashMap<String, ExternalManifestEntry> {
    let bytes = read_file_bounded(&external_fixture_root().join("manifest.json"));
    serde_json::from_slice(&bytes).expect("manifest.json の parse に失敗した")
}

/// 1 ケース分の external data fixture を実際に読み込み・実行し、
/// [`diff_stats`] と同じ判定材料（`DiffStats`）を得る。参照入出力は
/// `manifest.json`（このスクリプト自身が生成した重みに対する自己完結
/// 参照値。[`ExternalManifestEntry`] のコメント参照）を使う。
fn compute_external_case_stats(case_name: &str, entry: &ExternalManifestEntry) -> DiffStats {
    let model_path = external_fixture_root().join(format!("{case_name}_dynamo.onnx"));
    let bytes = read_file_bounded(&model_path);
    let model = proto::decode_model(&bytes)
        .unwrap_or_else(|e| panic!("{case_name}: external fixture decode 失敗: {e}"));
    let graph = build_graph_with_external_data(
        &model,
        &external_fixture_root(),
        &ExternalDataOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{case_name}: external data 解決に失敗: {e}"));

    let input_tensor = tensor_from_record(&entry.input);
    let mut feeds: HashMap<String, Value> = HashMap::new();
    feeds.insert("x".to_string(), Value::F32(input_tensor));
    let result =
        run(&graph, feeds).unwrap_or_else(|e| panic!("{case_name}: external data run 失敗: {e}"));
    let actual = match result.get("y") {
        Some(Value::F32(t)) => t,
        other => panic!("{case_name}: 出力が F32 以外／欠落: {other:?}"),
    };
    let expected_tensor = tensor_from_record(&entry.output);
    assert_eq!(
        actual.shape(),
        expected_tensor.shape(),
        "{case_name}: 出力 shape 不一致"
    );
    let stats = diff_stats(
        actual.as_slice().expect("as_slice 失敗"),
        expected_tensor.as_slice().expect("as_slice 失敗"),
    );
    // `run_case` と同型の実測値ログ（`EXTERNAL_DATA_BASELINES` の再測定時に
    // `--nocapture` で拾う。イシュー #2347・§4 レビュー対応）。
    eprintln!(
        "{case_name} [external-data]: total={} req7_fail_count={} max_abs_diff={:?} \
         max_rel_err={:?} mean_abs_diff={:?}",
        stats.total, stats.fail_count, stats.max_abs_diff, stats.max_rel_err, stats.mean_abs_diff,
    );
    stats
}

/// external data fixture（`EXTERNAL_DATA_CASE_NAMES`）1 ケース分の記録済み
/// 実測上限 baseline（`ReductionBaseline`／`REDUCTION_BASELINES` と同型・
/// 別テーブル。モジュール冒頭コメント参照）。`exporter_name` フィールドを
/// 持たない（本 fixture 群は dynamo exporter の生出力のみで `ts` 変種を
/// 持たないため。`ReductionBaseline` との構造差はこの 1 点のみ）。
#[derive(Debug, Clone, Copy)]
struct ExternalDataBaseline {
    case_name: &'static str,
    total: usize,
    baseline_fail_count: usize,
    baseline_max_abs_diff_ceiling: f32,
    baseline_max_rel_err_ceiling: f32,
    baseline_mean_abs_diff_ceiling: f64,
}

/// 記録済み baseline 一覧（3 行 = `EXTERNAL_DATA_CASE_NAMES` の 3 ケース）。
///
/// 出典: 本レビュー対応時に `cargo test -p fandhe-ai-onnx-interop --test \
/// onnx_interp_pytorch_cnn_fixture external_data_fixture_matches_self_
/// contained_reference -- --nocapture --test-threads=1` で実測した値
/// （`compute_external_case_stats` の `eprintln!` 出力。2026-09-28）。
/// fixture はリポジトリにコミット済みの固定バイト列（`.onnx`／`.onnx.data`
/// ・`manifest.json` とも再生成しない限り不変）のため、`REDUCTION_
/// BASELINES` と同じく CI・ローカルを問わず単一の値で成立する。
static EXTERNAL_DATA_BASELINES: &[ExternalDataBaseline] = &[
    ExternalDataBaseline {
        case_name: "conv2d_basic",
        total: 256,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 0.0,
        baseline_max_rel_err_ceiling: 0.0,
        baseline_mean_abs_diff_ceiling: 0.0,
    },
    ExternalDataBaseline {
        case_name: "conv2d_nobias",
        total: 256,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 0.0,
        baseline_max_rel_err_ceiling: 0.0,
        baseline_mean_abs_diff_ceiling: 0.0,
    },
    ExternalDataBaseline {
        case_name: "conv2d_stride_dil_group",
        total: 100,
        baseline_fail_count: 0,
        baseline_max_abs_diff_ceiling: 1.192_092_9e-7,
        baseline_max_rel_err_ceiling: 3.731_992_1e-6,
        baseline_mean_abs_diff_ceiling: 3.0193477869033815e-8,
    },
];

/// `case_name` に対応する [`ExternalDataBaseline`] を引く。未登録ケースは
/// fail-closed に panic する（`find_reduction_baseline` と同型。黙って
/// skip しない）。
#[track_caller]
fn find_external_data_baseline(case_name: &str) -> &'static ExternalDataBaseline {
    EXTERNAL_DATA_BASELINES
        .iter()
        .find(|b| b.case_name == case_name)
        .unwrap_or_else(|| {
            panic!(
                "{case_name}: EXTERNAL_DATA_BASELINES に行が無い（baseline の追加は \
                 実測値のみ・人間承認必須。未登録ケースを黙って通過させない）"
            )
        })
}

/// [`ExternalDataBaseline`] に対する fail-closed 非後退判定
/// （`assert_no_reduction_baseline_regression` と同型）。
#[track_caller]
fn assert_no_external_data_baseline_regression(
    case_name: &str,
    stats: &DiffStats,
    baseline: &ExternalDataBaseline,
) {
    assert_eq!(
        stats.total, baseline.total,
        "{case_name}: 比較対象の要素数が baseline({}) と一致しない（形状・\
         比較対象がずれている可能性）",
        baseline.total,
    );
    assert!(
        stats.fail_count <= baseline.baseline_fail_count,
        "{case_name}: baseline 非後退契約 FAIL — fail_count が後退しました \
         (actual={}, baseline={})",
        stats.fail_count,
        baseline.baseline_fail_count,
    );
    assert!(
        stats.max_abs_diff <= baseline.baseline_max_abs_diff_ceiling,
        "{case_name}: baseline 非後退契約 FAIL — max_abs_diff が後退しました \
         (actual={:?}, ceiling={:?})",
        stats.max_abs_diff,
        baseline.baseline_max_abs_diff_ceiling,
    );
    assert!(
        stats.max_rel_err <= baseline.baseline_max_rel_err_ceiling,
        "{case_name}: baseline 非後退契約 FAIL — max_rel_err が後退しました \
         (actual={:?}, ceiling={:?})",
        stats.max_rel_err,
        baseline.baseline_max_rel_err_ceiling,
    );
    assert!(
        stats.mean_abs_diff <= baseline.baseline_mean_abs_diff_ceiling,
        "{case_name}: baseline 非後退契約 FAIL — mean_abs_diff が後退しました \
         (actual={:?}, ceiling={:?})",
        stats.mean_abs_diff,
        baseline.baseline_mean_abs_diff_ceiling,
    );
}

/// `EXTERNAL_DATA_BASELINES` の構造的整合性を検査する
/// （`reduction_baselines_are_well_formed` と同型）。
#[test]
fn external_data_baselines_are_well_formed() {
    let mut case_keys: Vec<&str> = EXTERNAL_DATA_CASE_NAMES.to_vec();
    case_keys.sort_unstable();

    let mut baseline_keys: Vec<&str> = EXTERNAL_DATA_BASELINES
        .iter()
        .map(|b| b.case_name)
        .collect();
    baseline_keys.sort_unstable();
    let baseline_keys_unique_count = {
        let mut dedup = baseline_keys.clone();
        dedup.dedup();
        dedup.len()
    };
    assert_eq!(
        baseline_keys.len(),
        baseline_keys_unique_count,
        "EXTERNAL_DATA_BASELINES に重複行がある: {baseline_keys:?}"
    );
    assert_eq!(
        case_keys, baseline_keys,
        "EXTERNAL_DATA_BASELINES と EXTERNAL_DATA_CASE_NAMES が不一致（取りこぼし・過剰のいずれか）"
    );

    for b in EXTERNAL_DATA_BASELINES {
        assert!(b.total > 0, "{}: baseline.total が 0", b.case_name);
        assert_eq!(
            b.baseline_fail_count, 0,
            "{}: baseline_fail_count が 0 以外（REQ-7 式は必須条件のため baseline \
             側も 0 のみを許容する）",
            b.case_name
        );
        assert!(
            b.baseline_max_abs_diff_ceiling.is_finite() && b.baseline_max_abs_diff_ceiling >= 0.0,
            "{}: baseline_max_abs_diff_ceiling が非有限または負値: {:?}",
            b.case_name,
            b.baseline_max_abs_diff_ceiling
        );
        assert!(
            b.baseline_max_rel_err_ceiling.is_finite() && b.baseline_max_rel_err_ceiling >= 0.0,
            "{}: baseline_max_rel_err_ceiling が非有限または負値: {:?}",
            b.case_name,
            b.baseline_max_rel_err_ceiling
        );
        assert!(
            b.baseline_mean_abs_diff_ceiling.is_finite() && b.baseline_mean_abs_diff_ceiling >= 0.0,
            "{}: baseline_mean_abs_diff_ceiling が非有限または負値: {:?}",
            b.case_name,
            b.baseline_mean_abs_diff_ceiling
        );
    }
}

/// 1. external data 経由で計算した出力が、`manifest.json` に記録した
///    自己完結参照出力（PyTorch がこのスクリプトの重みに対して計算した
///    値）と一致すること（縮約系 op `Conv` を含むため REQ-7 事前固定式
///    `fail_count == 0` を要求したうえで、`EXTERNAL_DATA_BASELINES` による
///    fail-closed 非後退判定〈`total`／`max_abs_diff`／`max_rel_err`／
///    `mean_abs_diff` の実測値そのものを ceiling とする〉も適用する
///    （#2329 の `Req7BaselineNonRegression` と同じ判定方式。2026-09-28
///    ユーザー承認。モジュール冒頭コメント参照）。
/// 2. initializer の bit 完全一致（external data 経由での読み込み値 vs
///    `.onnx.data` ファイルの生バイト列を直接 f32 として解釈した値）。
///    実装計画 §4.5 の 1・3 をまとめたもの（2 は `.data` ファイルへの直接
///    突合に置き換え。時間制約により `conv2d_*` 3 ケースへ限定——
///    `README.md`「実測結果」節参照）。
#[test]
fn external_data_fixture_matches_self_contained_reference() {
    let manifest = load_external_manifest();
    for &case_name in EXTERNAL_DATA_CASE_NAMES {
        let entry = manifest
            .get(case_name)
            .unwrap_or_else(|| panic!("manifest.json に '{case_name}' が無い"));

        // 1. REQ-7 事前固定式（fail_count == 0）を必須条件として適用した
        //    うえで、baseline 非後退判定（EXTERNAL_DATA_BASELINES）を課す。
        let stats = compute_external_case_stats(case_name, entry);
        assert_eq!(
            stats.fail_count, 0,
            "{case_name}: REQ-7 事前固定式 fail_count が 0 でない（total={} \
             max_abs_diff={:?} max_rel_err={:?} mean_abs_diff={:?}）",
            stats.total, stats.max_abs_diff, stats.max_rel_err, stats.mean_abs_diff
        );
        let baseline = find_external_data_baseline(case_name);
        assert_no_external_data_baseline_regression(case_name, &stats, baseline);

        // 2. initializer の bit 完全一致（external data 経由 vs `.onnx.data`
        //    の生バイト列を直接解釈した値）。
        let ext_model_path = external_fixture_root().join(format!("{case_name}_dynamo.onnx"));
        let ext_bytes = read_file_bounded(&ext_model_path);
        let ext_model = proto::decode_model(&ext_bytes).expect("external decode 失敗");
        let ext_graph = build_graph_with_external_data(
            &ext_model,
            &external_fixture_root(),
            &ExternalDataOptions::default(),
        )
        .expect("external data 解決失敗");

        for init in &entry.external_initializers {
            let data_path = external_fixture_root().join(
                init.location
                    .as_deref()
                    .unwrap_or_else(|| panic!("{case_name}: location が無い")),
            );
            let raw = read_file_bounded(&data_path);
            let expected: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            match ext_graph.initializers.get(&init.name) {
                Some(RawTensor::F32 { data, .. }) => assert_eq!(
                    data, &expected,
                    "{case_name}: initializer '{}' が .data ファイルの生バイト列と bit 一致しない",
                    init.name
                ),
                other => panic!(
                    "{case_name}: initializer '{}' が RawTensor::F32 でない: {other:?}",
                    init.name
                ),
            }
        }
    }
}

/// 3. fixture が実際に external data 経路を通っていることの検査
///    （manifest とデコード後の proto の両方から数える。実装計画 §4.5-4）。
#[test]
fn external_data_fixture_actually_uses_external_path() {
    let manifest = load_external_manifest();
    let mut total_external_f32 = 0usize;
    for &case_name in EXTERNAL_DATA_CASE_NAMES {
        let entry = manifest
            .get(case_name)
            .unwrap_or_else(|| panic!("manifest.json に '{case_name}' が無い"));
        assert!(
            entry.was_external_data,
            "{case_name}: manifest 上 external data ではない"
        );
        assert!(
            !entry.external_initializers.is_empty(),
            "{case_name}: manifest 上 external な initializer が 0 件"
        );
        for init in &entry.external_initializers {
            assert_eq!(
                init.data_type,
                proto::data_type::FLOAT,
                "{case_name}: 期待は FLOAT"
            );
            assert!(init.location.is_some(), "{case_name}: location が無い");
            total_external_f32 += 1;
        }
        assert_eq!(
            entry.op_types,
            vec!["Conv".to_string()],
            "{case_name}: op_types 不一致"
        );

        // proto を実際に decode して data_location=EXTERNAL のテンソルが
        // 存在することも確認する（manifest 側の記録だけに依存しない）。
        let model_path = external_fixture_root().join(format!("{case_name}_dynamo.onnx"));
        let bytes = read_file_bounded(&model_path);
        let model = proto::decode_model(&bytes).expect("decode 失敗");
        let g = model.graph.as_ref().expect("graph が無い");
        let n_external = g
            .initializer
            .iter()
            .filter(|t| {
                t.data_location == fandhe_ai_onnx_interop::onnx::proto::data_location::EXTERNAL
            })
            .count();
        assert!(
            n_external >= 1,
            "{case_name}: decode した proto に external initializer が無い"
        );
    }
    assert!(
        total_external_f32 >= 1,
        "F32（Conv の weight）が external な fixture が 1 件も無い（A1 要件）"
    );
    // 本 fixture 群では INT64 の shape 定数は 16〜24 バイトと小さく external
    // にならなかった（`gen_external.py` コメント・`README.md` 参照。実測に
    // 基づき §4.5-4 の要件を調整済み）。
}

/// 4. A6 の回帰: external データを持つ `.onnx` をバイト列入口
///    （`decode_model` -> `build_graph`）へ渡した場合、従来どおり
///    `GraphError::RawDataByteLenMismatch` になることを確認する
///    （`onnx_external_data.rs::bytes_entry_point_still_rejects_external_data_model`
///    の実 fixture 版）。
#[test]
fn external_data_fixture_bytes_entry_point_still_rejects() {
    use fandhe_ai_onnx_interop::onnx::graph::GraphError;

    for &case_name in EXTERNAL_DATA_CASE_NAMES {
        let model_path = external_fixture_root().join(format!("{case_name}_dynamo.onnx"));
        let bytes = read_file_bounded(&model_path);
        let model = proto::decode_model(&bytes).expect("decode 失敗");
        let result = build_graph(&model);
        assert!(
            matches!(
                result,
                Err(GraphError::RawDataByteLenMismatch {
                    actual_bytes: 0,
                    ..
                })
            ),
            "{case_name}: バイト列入口が external data モデルを拒否しなかった（A6 回帰）: {result:?}"
        );
    }
}
