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
//! ## 判定方式についての注記（REQ-2・REQ-7 との混同禁止）
//!
//! 縮約系（`Conv`・`AveragePool`・`GlobalAveragePool`・`BatchNormalization`）
//! の結合順序（PyTorch CPU 実行系 vs 本クレートの直接ループ）は異なるため
//! bit 完全一致は主張しない。実測で bit 一致したケースは [`Expectation::
//! BitExact`] として固定する（厳しい側。`f32::mul_add`／`f64` 縮約は決定的
//! なため CI・実機で結果は揺れない）。bit 一致しなかったケースは
//! `tests/model_zoo_parity.rs` と同じ **REQ-7 事前固定式**
//! `abs_err / (|ref| + 1e-6) <= 1e-3` を暫定適用する
//! （[`Expectation::Req7Provisional`]）。**この暫定適用は最終判定方式として
//! ユーザー承認を得たものではない**——実測ログ（`docs/perf/logs/
//! onnx-cnn-ops-pytorch-fixture-2329/README.md`）と PR 本文に「最終判定
//! 方式は承認待ち」と明記する。`ParityBaseline` のような実測上限 baseline
//! は本ファイルでは新設しない（新設には人間承認が必要なため）。
//!
//! `.claude/rules/coding-rust.md` の REQ-2 バックエンド間数値一致 OR 複合
//! 判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）とは別指標であり、
//! 混同してどちらかを緩和しない。ONNX インタープリタはホスト CPU 実行のみ
//! （`onnx::interp::run`。`interp.rs` のディスパッチ表で対象 6 op はすべて
//! `(compute_*(..)?, false)` = 常にホスト実行）のため REQ-2 の対象外
//! （構造的 N/A。`docs/onnx-model-zoo-parity.md` §4 と同じ理由）。
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

/// 出力の突合結果（bit 不一致要素数・max_abs_diff・max_rel_err）。
struct DiffStats {
    fail_count: usize,
    bit_mismatch_count: usize,
    max_abs_diff: f32,
    max_rel_err: f32,
}

fn diff_stats(actual: &[f32], expected: &[f32]) -> DiffStats {
    assert_eq!(actual.len(), expected.len(), "要素数不一致");
    let mut fail_count = 0usize;
    let mut bit_mismatch_count = 0usize;
    let mut max_abs_diff = 0.0f32;
    let mut max_rel_err = 0.0f32;
    for (&a, &e) in actual.iter().zip(expected.iter()) {
        if a.to_bits() != e.to_bits() {
            bit_mismatch_count += 1;
        }
        let abs_diff = (a - e).abs();
        if abs_diff.is_finite() && abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
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
    DiffStats {
        fail_count,
        bit_mismatch_count,
        max_abs_diff,
        max_rel_err,
    }
}

/// 出力の判定方式。モジュール doc「判定方式についての注記」参照。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Expectation {
    /// 選択・形状操作（`MaxPool`・`Flatten`）。フォールバックなしの bit 一致のみ。
    BitExact,
    /// 縮約系（`Conv`・`AveragePool`・`GlobalAveragePool`・`BatchNormalization`）。
    /// 実測で bit 一致したケースは [`Self::BitExact`] へ固定し、しなかった
    /// ケースのみ本 variant で REQ-7 暫定判定にフォールバックする
    /// （最終判定方式はユーザー承認待ち。モジュール doc 参照）。
    Req7Provisional,
}

/// `(case_name, exporter_name) -> Expectation` の期待値表。
/// 実測（本 PR 作成時に `--nocapture` で確認した結果。
/// `docs/perf/logs/onnx-cnn-ops-pytorch-fixture-2329/README.md` に転記済み）
/// に基づき固定する。表と `reference.json` のケース集合が完全一致することを
/// [`expectation_table_matches_reference_cases`] で検査し、取りこぼしを
/// fail-closed に止める。
const EXPECTATIONS: &[(&str, &str, Expectation)] = &[
    // 実測で bit 一致（厳しい側へ固定。モジュール doc §5.4 の方針）。
    ("conv2d_basic", "ts", Expectation::BitExact),
    ("conv2d_basic", "dynamo", Expectation::BitExact),
    (
        "conv2d_stride_dil_group",
        "ts",
        Expectation::Req7Provisional,
    ),
    (
        "conv2d_stride_dil_group",
        "dynamo",
        Expectation::Req7Provisional,
    ),
    ("conv2d_nobias", "ts", Expectation::BitExact),
    ("conv2d_nobias", "dynamo", Expectation::BitExact),
    ("conv1d_basic", "ts", Expectation::Req7Provisional),
    ("conv1d_basic", "dynamo", Expectation::Req7Provisional),
    ("maxpool2d_basic", "ts", Expectation::BitExact),
    ("maxpool2d_basic", "dynamo", Expectation::BitExact),
    ("maxpool2d_pad_dil_ceil", "ts", Expectation::BitExact),
    ("maxpool2d_pad_dil_ceil", "dynamo", Expectation::BitExact),
    ("maxpool1d_basic", "ts", Expectation::BitExact),
    ("maxpool1d_basic", "dynamo", Expectation::BitExact),
    ("avgpool2d_include_pad", "ts", Expectation::Req7Provisional),
    (
        "avgpool2d_include_pad",
        "dynamo",
        Expectation::Req7Provisional,
    ),
    ("avgpool2d_exclude_pad", "ts", Expectation::Req7Provisional),
    (
        "avgpool2d_exclude_pad",
        "dynamo",
        Expectation::Req7Provisional,
    ),
    (
        "avgpool2d_ceil_overhang_incl",
        "ts",
        Expectation::Req7Provisional,
    ),
    (
        "avgpool2d_ceil_overhang_incl",
        "dynamo",
        Expectation::Req7Provisional,
    ),
    (
        "avgpool2d_ceil_overhang_excl",
        "ts",
        Expectation::Req7Provisional,
    ),
    (
        "avgpool2d_ceil_overhang_excl",
        "dynamo",
        Expectation::Req7Provisional,
    ),
    ("avgpool1d_basic", "ts", Expectation::Req7Provisional),
    ("avgpool1d_basic", "dynamo", Expectation::Req7Provisional),
    ("gap2d", "ts", Expectation::Req7Provisional),
    ("gap2d", "dynamo", Expectation::Req7Provisional),
    ("gap1d", "ts", Expectation::BitExact),
    ("gap1d", "dynamo", Expectation::Req7Provisional),
    ("bn2d_eval", "ts", Expectation::Req7Provisional),
    ("bn2d_eval", "dynamo", Expectation::Req7Provisional),
    ("bn2d_eval_eps", "ts", Expectation::Req7Provisional),
    ("bn2d_eval_eps", "dynamo", Expectation::Req7Provisional),
    ("bn1d_eval", "ts", Expectation::Req7Provisional),
    ("bn1d_eval", "dynamo", Expectation::Req7Provisional),
    ("flatten_default", "ts", Expectation::BitExact),
    ("flatten_default", "dynamo", Expectation::BitExact),
    ("flatten_start2", "ts", Expectation::BitExact),
    ("flatten_start2", "dynamo", Expectation::BitExact),
];

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

/// 1 ケース・1 exporter を decode → build_graph → run → 判定まで実行する。
fn run_case(case_name: &str, exporter_name: &str, case: &CaseRecord, expectation: Expectation) {
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
        "{case_name} [{exporter_name}]: op_types={:?} bit_mismatch={}/{} max_abs_diff={} \
         max_rel_err={} req7_fail_count={}",
        exporter.op_types,
        stats.bit_mismatch_count,
        actual_slice.len(),
        stats.max_abs_diff,
        stats.max_rel_err,
        stats.fail_count
    );

    match expectation {
        Expectation::BitExact => {
            assert_eq!(
                stats.bit_mismatch_count, 0,
                "{case_name} [{exporter_name}]: bit 不一致（BitExact 期待。\
                 選択・形状操作でフォールバックは許容しない）"
            );
        }
        Expectation::Req7Provisional => {
            assert_eq!(
                stats.fail_count, 0,
                "{case_name} [{exporter_name}]: 暫定 REQ-7 判定でも fail \
                 （max_rel_err={}）。tolerance は緩めない（`.claude/rules/\
                 coding-rust.md`）。ユーザー判断が必要",
                stats.max_rel_err
            );
        }
    }
}

macro_rules! fixture_test {
    ($fn_name:ident, $case_name:literal, $exporter_name:literal, $expectation:expr) => {
        #[test]
        fn $fn_name() {
            let doc = load_reference();
            let case = doc
                .cases
                .get($case_name)
                .unwrap_or_else(|| panic!("reference.json に '{}' が無い", $case_name));
            run_case($case_name, $exporter_name, case, $expectation);
        }
    };
}

fixture_test!(conv2d_basic_ts, "conv2d_basic", "ts", Expectation::BitExact);
fixture_test!(
    conv2d_basic_dynamo,
    "conv2d_basic",
    "dynamo",
    Expectation::BitExact
);
fixture_test!(
    conv2d_stride_dil_group_ts,
    "conv2d_stride_dil_group",
    "ts",
    Expectation::Req7Provisional
);
fixture_test!(
    conv2d_stride_dil_group_dynamo,
    "conv2d_stride_dil_group",
    "dynamo",
    Expectation::Req7Provisional
);
fixture_test!(
    conv2d_nobias_ts,
    "conv2d_nobias",
    "ts",
    Expectation::BitExact
);
fixture_test!(
    conv2d_nobias_dynamo,
    "conv2d_nobias",
    "dynamo",
    Expectation::BitExact
);
fixture_test!(
    conv1d_basic_ts,
    "conv1d_basic",
    "ts",
    Expectation::Req7Provisional
);
fixture_test!(
    conv1d_basic_dynamo,
    "conv1d_basic",
    "dynamo",
    Expectation::Req7Provisional
);
fixture_test!(
    maxpool2d_basic_ts,
    "maxpool2d_basic",
    "ts",
    Expectation::BitExact
);
fixture_test!(
    maxpool2d_basic_dynamo,
    "maxpool2d_basic",
    "dynamo",
    Expectation::BitExact
);
fixture_test!(
    maxpool2d_pad_dil_ceil_ts,
    "maxpool2d_pad_dil_ceil",
    "ts",
    Expectation::BitExact
);
fixture_test!(
    maxpool2d_pad_dil_ceil_dynamo,
    "maxpool2d_pad_dil_ceil",
    "dynamo",
    Expectation::BitExact
);
fixture_test!(
    maxpool1d_basic_ts,
    "maxpool1d_basic",
    "ts",
    Expectation::BitExact
);
fixture_test!(
    maxpool1d_basic_dynamo,
    "maxpool1d_basic",
    "dynamo",
    Expectation::BitExact
);
fixture_test!(
    avgpool2d_include_pad_ts,
    "avgpool2d_include_pad",
    "ts",
    Expectation::Req7Provisional
);
fixture_test!(
    avgpool2d_include_pad_dynamo,
    "avgpool2d_include_pad",
    "dynamo",
    Expectation::Req7Provisional
);
fixture_test!(
    avgpool2d_exclude_pad_ts,
    "avgpool2d_exclude_pad",
    "ts",
    Expectation::Req7Provisional
);
fixture_test!(
    avgpool2d_exclude_pad_dynamo,
    "avgpool2d_exclude_pad",
    "dynamo",
    Expectation::Req7Provisional
);
fixture_test!(
    avgpool2d_ceil_overhang_incl_ts,
    "avgpool2d_ceil_overhang_incl",
    "ts",
    Expectation::Req7Provisional
);
fixture_test!(
    avgpool2d_ceil_overhang_incl_dynamo,
    "avgpool2d_ceil_overhang_incl",
    "dynamo",
    Expectation::Req7Provisional
);
fixture_test!(
    avgpool2d_ceil_overhang_excl_ts,
    "avgpool2d_ceil_overhang_excl",
    "ts",
    Expectation::Req7Provisional
);
fixture_test!(
    avgpool2d_ceil_overhang_excl_dynamo,
    "avgpool2d_ceil_overhang_excl",
    "dynamo",
    Expectation::Req7Provisional
);
fixture_test!(
    avgpool1d_basic_ts,
    "avgpool1d_basic",
    "ts",
    Expectation::Req7Provisional
);
fixture_test!(
    avgpool1d_basic_dynamo,
    "avgpool1d_basic",
    "dynamo",
    Expectation::Req7Provisional
);
fixture_test!(gap2d_ts, "gap2d", "ts", Expectation::Req7Provisional);
fixture_test!(
    gap2d_dynamo,
    "gap2d",
    "dynamo",
    Expectation::Req7Provisional
);
fixture_test!(gap1d_ts, "gap1d", "ts", Expectation::BitExact);
fixture_test!(
    gap1d_dynamo,
    "gap1d",
    "dynamo",
    Expectation::Req7Provisional
);
fixture_test!(
    bn2d_eval_ts,
    "bn2d_eval",
    "ts",
    Expectation::Req7Provisional
);
fixture_test!(
    bn2d_eval_dynamo,
    "bn2d_eval",
    "dynamo",
    Expectation::Req7Provisional
);
fixture_test!(
    bn2d_eval_eps_ts,
    "bn2d_eval_eps",
    "ts",
    Expectation::Req7Provisional
);
fixture_test!(
    bn2d_eval_eps_dynamo,
    "bn2d_eval_eps",
    "dynamo",
    Expectation::Req7Provisional
);
fixture_test!(
    bn1d_eval_ts,
    "bn1d_eval",
    "ts",
    Expectation::Req7Provisional
);
fixture_test!(
    bn1d_eval_dynamo,
    "bn1d_eval",
    "dynamo",
    Expectation::Req7Provisional
);
fixture_test!(
    flatten_default_ts,
    "flatten_default",
    "ts",
    Expectation::BitExact
);
fixture_test!(
    flatten_default_dynamo,
    "flatten_default",
    "dynamo",
    Expectation::BitExact
);
fixture_test!(
    flatten_start2_ts,
    "flatten_start2",
    "ts",
    Expectation::BitExact
);
fixture_test!(
    flatten_start2_dynamo,
    "flatten_start2",
    "dynamo",
    Expectation::BitExact
);
