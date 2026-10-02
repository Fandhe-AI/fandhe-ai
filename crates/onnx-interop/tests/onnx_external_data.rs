//! ONNX external data（イシュー #2347）の合成入力テスト。
//!
//! `fandhe_ai_onnx_interop::onnx::external_data` の A2〜A5 検証（パス
//! トラバーサル・symlink・数値文法・範囲・重複・合計上限）を、実際に
//! ファイルシステム上へ合成した `.data` ファイルへ対して確認する。
//! PyTorch 実生成 fixture 側の判定は
//! `tests/onnx_interp_pytorch_cnn_fixture.rs` を参照（本ファイルは torch に
//! 依存しない）。
//!
//! 一時ディレクトリは `support::temp_dir::TempDirGuard`（一意名 + `create_dir`
//! による排他作成。作成前の削除なし）で作り、終了時に削除する（#2383）。

mod support;

use std::io::Write;
use std::path::{Path, PathBuf};

use fandhe_ai_onnx_interop::onnx::external_data::{
    ExternalDataError, ExternalDataOptions, LocationRejectReason, build_graph_with_external_data,
};
use fandhe_ai_onnx_interop::onnx::graph::{GraphError, build_graph};
// `RawTensor` は実解決に成功した際の initializer 値検査（external data の
// 実解決が成功する unix・Windows 限定のテストのみ）で使う。それ以外の
// ビルドでは未使用になるため揃えて cfg する（イシュー #2349）。
#[cfg(any(unix, windows))]
use fandhe_ai_onnx_interop::onnx::graph::RawTensor;
use fandhe_ai_onnx_interop::onnx::proto::{
    GraphProto, ModelProto, StringStringEntryProto, TensorProto, data_location, data_type,
};

/// テスト専用の一時ディレクトリを作る（テストごとに固有のサブディレクトリ名
/// を要求し、並行実行時の衝突を避ける）。
struct TempDir(support::temp_dir::TempDirGuard);

impl TempDir {
    /// `name` はディレクトリ名の可読性のためだけに使う。一意性は
    /// `TempDirGuard`（pid・ナノ秒・カウンタ + `create_dir` 排他作成）が担保する。
    fn new(name: &str) -> Self {
        TempDir(support::temp_dir::TempDirGuard::new(&format!(
            "external-data-{name}"
        )))
    }

    fn path(&self) -> &Path {
        self.0.path()
    }

    fn write_file(&self, rel: &str, bytes: &[u8]) -> PathBuf {
        let p = self.path().join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&p)
            .unwrap();
        f.write_all(bytes).unwrap();
        p
    }
}

fn ext_entry(key: &str, value: &str) -> StringStringEntryProto {
    StringStringEntryProto {
        key: key.to_string(),
        value: value.to_string(),
    }
}

/// `location`（必須）・`offset`（省略可）・`length`（省略可）から
/// external_data の Vec を組み立てる。
fn external_entries(
    location: &str,
    offset: Option<&str>,
    length: Option<&str>,
) -> Vec<StringStringEntryProto> {
    let mut v = vec![ext_entry("location", location)];
    if let Some(o) = offset {
        v.push(ext_entry("offset", o));
    }
    if let Some(l) = length {
        v.push(ext_entry("length", l));
    }
    v
}

fn external_tensor(
    name: &str,
    dims: Vec<i64>,
    dt: i32,
    location: &str,
    offset: Option<&str>,
    length: Option<&str>,
) -> TensorProto {
    TensorProto {
        dims,
        data_type: dt,
        float_data: Vec::new(),
        int64_data: Vec::new(),
        name: name.to_string(),
        raw_data: Vec::new(),
        external_data: external_entries(location, offset, length),
        data_location: data_location::EXTERNAL,
    }
}

fn model_with_initializer(t: TensorProto) -> ModelProto {
    ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: Vec::new(),
            name: "g".to_string(),
            initializer: vec![t],
            input: Vec::new(),
            output: Vec::new(),
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
        }),
        opset_import: Vec::new(),
    }
}

fn model_with_initializers(ts: Vec<TensorProto>) -> ModelProto {
    ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: Vec::new(),
            name: "g".to_string(),
            initializer: ts,
            input: Vec::new(),
            output: Vec::new(),
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
        }),
        opset_import: Vec::new(),
    }
}

fn assert_external_err(
    result: Result<fandhe_ai_onnx_interop::onnx::graph::Graph, GraphError>,
) -> ExternalDataError {
    match result {
        Err(GraphError::ExternalData(e)) => e,
        other => panic!("ExternalData エラーを期待したが: {other:?}"),
    }
}

// --- 正常系 ---

#[cfg(any(unix, windows))]
#[test]
fn f32_external_tensor_loads_and_matches_bit_exact_inline() {
    let dir = TempDir::new("f32-basic");
    let bytes: [f32; 4] = [1.0, -2.5, 3.5, 0.0];
    let mut raw = Vec::new();
    for v in bytes {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    dir.write_file("w.onnx.data", &raw);

    let t = external_tensor("w", vec![2, 2], data_type::FLOAT, "w.onnx.data", None, None);
    let model = model_with_initializer(t);
    let graph = build_graph_with_external_data(&model, dir.path(), &ExternalDataOptions::default())
        .expect("external data の読み込みは成功するはず");
    match graph.initializers.get("w") {
        Some(RawTensor::F32 { data, shape }) => {
            assert_eq!(shape.as_slice(), &[2, 2]);
            assert_eq!(data.as_slice(), &bytes);
        }
        other => panic!("RawTensor::F32 を期待: {other:?}"),
    }
}

#[cfg(any(unix, windows))]
#[test]
fn int64_external_tensor_loads() {
    let dir = TempDir::new("i64-basic");
    let values: [i64; 3] = [1, -2, 3];
    let mut raw = Vec::new();
    for v in values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    dir.write_file("shape.onnx.data", &raw);

    let t = external_tensor(
        "shape",
        vec![3],
        data_type::INT64,
        "shape.onnx.data",
        None,
        None,
    );
    let model = model_with_initializer(t);
    let graph = build_graph_with_external_data(&model, dir.path(), &ExternalDataOptions::default())
        .expect("external data の読み込みは成功するはず");
    match graph.initializers.get("shape") {
        Some(RawTensor::I64 { data, .. }) => assert_eq!(data.as_slice(), &values),
        other => panic!("RawTensor::I64 を期待: {other:?}"),
    }
}

#[cfg(any(unix, windows))]
#[test]
fn bool_and_f16_external_tensors_load() {
    let dir = TempDir::new("bool-f16");
    dir.write_file("mask.onnx.data", &[0u8, 1u8, 1u8]);
    let half = half::f16::from_f32(2.5).to_le_bytes();
    dir.write_file("h.onnx.data", &half);

    let bool_t = external_tensor(
        "mask",
        vec![3],
        data_type::BOOL,
        "mask.onnx.data",
        None,
        None,
    );
    let f16_t = external_tensor("h", vec![1], data_type::FLOAT16, "h.onnx.data", None, None);
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: Vec::new(),
            name: "g".to_string(),
            initializer: vec![bool_t, f16_t],
            input: Vec::new(),
            output: Vec::new(),
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
        }),
        opset_import: Vec::new(),
    };
    let graph = build_graph_with_external_data(&model, dir.path(), &ExternalDataOptions::default())
        .expect("external data の読み込みは成功するはず");
    match graph.initializers.get("mask") {
        Some(RawTensor::Bool { data, .. }) => assert_eq!(data.as_slice(), &[false, true, true]),
        other => panic!("RawTensor::Bool を期待: {other:?}"),
    }
    match graph.initializers.get("h") {
        Some(RawTensor::F16 { data, .. }) => assert_eq!(data[0].to_f32(), 2.5),
        other => panic!("RawTensor::F16 を期待: {other:?}"),
    }
}

#[cfg(any(unix, windows))]
#[test]
fn omitted_length_reads_to_eof() {
    let dir = TempDir::new("omit-length");
    let v: f32 = 7.5;
    dir.write_file("solo.onnx.data", &v.to_le_bytes());
    let t = external_tensor(
        "solo",
        vec![1],
        data_type::FLOAT,
        "solo.onnx.data",
        None,
        None,
    );
    let model = model_with_initializer(t);
    let graph = build_graph_with_external_data(&model, dir.path(), &ExternalDataOptions::default())
        .expect("length 省略（EOF まで）は成功するはず");
    match graph.initializers.get("solo") {
        Some(RawTensor::F32 { data, .. }) => assert_eq!(data[0], 7.5),
        other => panic!("RawTensor::F32 を期待: {other:?}"),
    }
}

#[cfg(any(unix, windows))]
#[test]
fn offset_selects_correct_region_in_shared_file() {
    let dir = TempDir::new("shared-offset");
    let a: f32 = 1.0;
    let b: f32 = 2.0;
    let mut raw = Vec::new();
    raw.extend_from_slice(&a.to_le_bytes());
    raw.extend_from_slice(&b.to_le_bytes());
    dir.write_file("shared.onnx.data", &raw);

    let t_a = external_tensor(
        "a",
        vec![1],
        data_type::FLOAT,
        "shared.onnx.data",
        Some("0"),
        Some("4"),
    );
    let t_b = external_tensor(
        "b",
        vec![1],
        data_type::FLOAT,
        "shared.onnx.data",
        Some("4"),
        Some("4"),
    );
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: Vec::new(),
            name: "g".to_string(),
            initializer: vec![t_a, t_b],
            input: Vec::new(),
            output: Vec::new(),
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
        }),
        opset_import: Vec::new(),
    };
    let graph = build_graph_with_external_data(&model, dir.path(), &ExternalDataOptions::default())
        .expect("隣接区間の読み込みは成功するはず");
    match graph.initializers.get("a") {
        Some(RawTensor::F32 { data, .. }) => assert_eq!(data[0], 1.0),
        other => panic!("{other:?}"),
    }
    match graph.initializers.get("b") {
        Some(RawTensor::F32 { data, .. }) => assert_eq!(data[0], 2.0),
        other => panic!("{other:?}"),
    }
}

// --- 異常系: location のパス検証（A2） ---

fn expect_location_reject(location: &str, dims: Vec<i64>) -> LocationRejectReason {
    let dir = TempDir::new("loc-reject");
    let t = external_tensor("x", dims, data_type::FLOAT, location, None, Some("4"));
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    match err {
        ExternalDataError::InvalidLocation { reason, .. } => reason,
        other => panic!("InvalidLocation を期待: {other:?}"),
    }
}

#[test]
fn absolute_location_is_rejected() {
    assert_eq!(
        expect_location_reject("/etc/passwd", vec![1]),
        LocationRejectReason::Absolute
    );
}

#[test]
fn parent_dir_traversal_is_rejected() {
    assert_eq!(
        expect_location_reject("../x.data", vec![1]),
        LocationRejectReason::ParentDir
    );
    assert_eq!(
        expect_location_reject("a/../../x.data", vec![1]),
        LocationRejectReason::ParentDir
    );
}

#[test]
fn backslash_in_location_is_rejected() {
    assert_eq!(
        expect_location_reject("a\\b.data", vec![1]),
        LocationRejectReason::Backslash
    );
}

#[test]
fn drive_prefix_location_is_rejected() {
    assert_eq!(
        expect_location_reject("C:x.data", vec![1]),
        LocationRejectReason::DrivePrefix
    );
}

#[test]
fn empty_location_is_rejected() {
    assert_eq!(
        expect_location_reject("", vec![1]),
        LocationRejectReason::Empty
    );
}

#[test]
fn too_long_location_is_rejected() {
    let long = "a".repeat(4097);
    assert_eq!(
        expect_location_reject(&long, vec![1]),
        LocationRejectReason::TooLong
    );
}

#[cfg(unix)]
#[test]
fn symlink_final_component_is_rejected() {
    let dir = TempDir::new("symlink-final");
    let real = dir.write_file("real.onnx.data", &[0u8; 4]);
    let link = dir.path().join("link.onnx.data");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    let t = external_tensor(
        "x",
        vec![1],
        data_type::FLOAT,
        "link.onnx.data",
        None,
        Some("4"),
    );
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(
        err,
        ExternalDataError::InvalidLocation {
            reason: LocationRejectReason::Symlink,
            ..
        }
    ));
}

#[cfg(unix)]
#[test]
fn symlink_intermediate_component_is_rejected() {
    let dir = TempDir::new("symlink-mid");
    let real_dir = dir.path().join("real_dir");
    std::fs::create_dir_all(&real_dir).unwrap();
    std::fs::write(real_dir.join("f.data"), [0u8; 4]).unwrap();
    let link_dir = dir.path().join("link_dir");
    std::os::unix::fs::symlink(&real_dir, &link_dir).unwrap();

    let t = external_tensor(
        "x",
        vec![1],
        data_type::FLOAT,
        "link_dir/f.data",
        None,
        Some("4"),
    );
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(
        err,
        ExternalDataError::InvalidLocation {
            reason: LocationRejectReason::Symlink,
            ..
        }
    ));
}

/// symlink が `base_dir` の**外**（別の一時ディレクトリ）を指していても、
/// ターゲットへ到達する前に「経路成分がシンボリックリンクである」こと
/// 自体で拒否されることを確認する（aarch64 Linux の `openat` フラグ定数
/// 取り違えバグ類型の回帰テスト。イシュー #2347。旧手書き定数実装では
/// aarch64 で `O_NOFOLLOW`/`O_DIRECTORY` の値が x86 の値のまま化けており
/// シンボリックリンク拒否自体が機能していなかった——本テストはターゲット
/// の中身ではなく拒否理由〈`LocationRejectReason::Symlink`〉と、
/// `base_dir` 外のファイル内容が読み込み結果に混入しないことの両方を
/// 検査することで、この類型のプラットフォーム定数バグを再発検知する）。
#[cfg(unix)]
#[test]
fn symlink_escaping_base_dir_via_absolute_target_is_rejected() {
    let dir = TempDir::new("symlink-escape");
    let outside = TempDir::new("symlink-escape-outside");
    // base_dir 外の秘密データ（読み込まれてはならない）。
    let secret = outside.write_file("secret.data", &[0xAAu8; 4]);
    let link = dir.path().join("link.onnx.data");
    std::os::unix::fs::symlink(&secret, &link).unwrap();

    let t = external_tensor(
        "x",
        vec![1],
        data_type::FLOAT,
        "link.onnx.data",
        None,
        Some("4"),
    );
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(
        matches!(
            err,
            ExternalDataError::InvalidLocation {
                reason: LocationRejectReason::Symlink,
                ..
            }
        ),
        "symlink 経由の base_dir 脱出が Symlink 以外の理由で扱われた: {err:?}"
    );
}

#[cfg(any(unix, windows))]
#[test]
fn missing_file_is_io_not_found() {
    let dir = TempDir::new("missing-file");
    let t = external_tensor("x", vec![1], data_type::FLOAT, "nope.data", None, Some("4"));
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(
        err,
        ExternalDataError::Io {
            kind: std::io::ErrorKind::NotFound,
            ..
        }
    ));
}

#[cfg(any(unix, windows))]
#[test]
fn directory_as_location_is_rejected() {
    let dir = TempDir::new("dir-as-loc");
    std::fs::create_dir_all(dir.path().join("subdir")).unwrap();
    let t = external_tensor("x", vec![1], data_type::FLOAT, "subdir", None, Some("4"));
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(
        err,
        ExternalDataError::InvalidLocation {
            reason: LocationRejectReason::NotRegularFile,
            ..
        }
    ));
}

// --- 異常系: offset/length の文法・範囲（A3） ---

fn expect_number_reject(offset: Option<&str>, length: Option<&str>) -> ExternalDataError {
    let dir = TempDir::new("num-reject");
    dir.write_file("f.data", &[0u8; 16]);
    let t = external_tensor("x", vec![1], data_type::FLOAT, "f.data", offset, length);
    let model = model_with_initializer(t);
    assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ))
}

#[test]
fn negative_offset_is_rejected() {
    assert!(matches!(
        expect_number_reject(Some("-1"), Some("4")),
        ExternalDataError::InvalidNumber {
            field: "offset",
            ..
        }
    ));
}

#[test]
fn plus_prefixed_offset_is_rejected() {
    assert!(matches!(
        expect_number_reject(Some("+4"), Some("4")),
        ExternalDataError::InvalidNumber {
            field: "offset",
            ..
        }
    ));
}

#[test]
fn whitespace_offset_is_rejected() {
    assert!(matches!(
        expect_number_reject(Some(" 4"), Some("4")),
        ExternalDataError::InvalidNumber {
            field: "offset",
            ..
        }
    ));
}

#[test]
fn non_numeric_offset_is_rejected() {
    assert!(matches!(
        expect_number_reject(Some("abc"), Some("4")),
        ExternalDataError::InvalidNumber {
            field: "offset",
            ..
        }
    ));
}

#[test]
fn empty_offset_is_rejected() {
    assert!(matches!(
        expect_number_reject(Some(""), Some("4")),
        ExternalDataError::InvalidNumber {
            field: "offset",
            ..
        }
    ));
}

#[cfg(any(unix, windows))]
#[test]
fn u64_overflow_length_is_rejected() {
    assert!(matches!(
        expect_number_reject(Some("0"), Some("99999999999999999999")),
        ExternalDataError::InvalidNumber {
            field: "length",
            ..
        }
    ));
}

#[cfg(any(unix, windows))]
#[test]
fn offset_plus_length_overflow_is_rejected() {
    let dir = TempDir::new("offset-overflow");
    dir.write_file("f.data", &[0u8; 16]);
    let t = external_tensor(
        "x",
        vec![1],
        data_type::FLOAT,
        "f.data",
        Some(&u64::MAX.to_string()),
        Some("4"),
    );
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(err, ExternalDataError::RangeOutOfFile { .. }));
}

#[cfg(any(unix, windows))]
#[test]
fn range_exceeding_file_length_is_rejected() {
    let dir = TempDir::new("range-exceed");
    dir.write_file("f.data", &[0u8; 4]);
    let t = external_tensor(
        "x",
        vec![2],
        data_type::FLOAT,
        "f.data",
        Some("0"),
        Some("8"),
    );
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(err, ExternalDataError::RangeOutOfFile { .. }));
}

#[cfg(any(unix, windows))]
#[test]
fn length_mismatch_with_expected_bytes_is_rejected() {
    let dir = TempDir::new("length-mismatch");
    dir.write_file("f.data", &[0u8; 16]);
    // dims=[1] FLOAT なので期待バイト長は 4 だが length=8 を指定する。
    let t = external_tensor(
        "x",
        vec![1],
        data_type::FLOAT,
        "f.data",
        Some("0"),
        Some("8"),
    );
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(
        err,
        ExternalDataError::LengthMismatch {
            expected_bytes: 4,
            actual_bytes: 8,
            ..
        }
    ));
}

#[cfg(any(unix, windows))]
#[test]
fn huge_length_is_rejected_before_allocation() {
    let dir = TempDir::new("huge-length");
    dir.write_file("f.data", &[0u8; 16]);
    let t = external_tensor(
        "x",
        vec![1],
        data_type::FLOAT,
        "f.data",
        Some("0"),
        Some(&(1u64 << 62).to_string()),
    );
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    // ファイル長（16 バイト）を大きく超えるため、合計上限に達する前に
    // RangeOutOfFile で拒否される（確保は一切発生しない）。
    assert!(matches!(err, ExternalDataError::RangeOutOfFile { .. }));
}

#[cfg(any(unix, windows))]
#[test]
fn total_size_limit_is_enforced() {
    let dir = TempDir::new("total-limit");
    let raw = vec![0u8; 16];
    dir.write_file("f.data", &raw);
    let t = external_tensor(
        "x",
        vec![4],
        data_type::FLOAT,
        "f.data",
        Some("0"),
        Some("16"),
    );
    let model = model_with_initializer(t);
    let options = ExternalDataOptions {
        max_total_bytes: 8,
        ..ExternalDataOptions::default()
    };
    let err = assert_external_err(build_graph_with_external_data(&model, dir.path(), &options));
    assert!(matches!(
        err,
        ExternalDataError::TotalSizeLimitExceeded { limit: 8, .. }
    ));
}

#[cfg(any(unix, windows))]
#[test]
fn external_file_count_limit_is_enforced() {
    // サイズ 0 のテンソルを別々の空ファイルへ分散させても
    // `max_total_bytes`（合計バイト数）はすり抜けるため、distinct ファイル
    // 数の上限（`max_external_files`）で拒否されることを確認する
    // （A04 資源枯渇対策。PR #2348 コードレビュー対応・
    // PRRT_kwDOTuUCJc6mlxhy）。
    let dir = TempDir::new("file-count-limit");
    dir.write_file("a.data", &[]);
    dir.write_file("b.data", &[]);
    dir.write_file("c.data", &[]);
    let t_a = external_tensor(
        "a",
        vec![0],
        data_type::FLOAT,
        "a.data",
        Some("0"),
        Some("0"),
    );
    let t_b = external_tensor(
        "b",
        vec![0],
        data_type::FLOAT,
        "b.data",
        Some("0"),
        Some("0"),
    );
    let t_c = external_tensor(
        "c",
        vec![0],
        data_type::FLOAT,
        "c.data",
        Some("0"),
        Some("0"),
    );
    let model = model_with_initializers(vec![t_a, t_b, t_c]);
    let options = ExternalDataOptions {
        max_external_files: 2,
        ..ExternalDataOptions::default()
    };
    let err = assert_external_err(build_graph_with_external_data(&model, dir.path(), &options));
    assert!(matches!(
        err,
        ExternalDataError::TooManyExternalFiles { limit: 2 }
    ));
}

#[cfg(any(unix, windows))]
#[test]
fn external_file_count_limit_allows_shared_file_reuse() {
    // 同一ファイルを複数テンソルが参照する通常の分割形式（1 ファイルを
    // initializer 群が共有する構成）は、distinct ファイル数としては 1 件
    // としてしか数えないため、上限に抵触しないことを確認する。
    let dir = TempDir::new("file-count-limit-shared");
    dir.write_file("shared.data", &[0u8; 8]);
    let t_a = external_tensor(
        "a",
        vec![1],
        data_type::FLOAT,
        "shared.data",
        Some("0"),
        Some("4"),
    );
    let t_b = external_tensor(
        "b",
        vec![1],
        data_type::FLOAT,
        "shared.data",
        Some("4"),
        Some("4"),
    );
    let model = model_with_initializers(vec![t_a, t_b]);
    let options = ExternalDataOptions {
        max_external_files: 1,
        ..ExternalDataOptions::default()
    };
    let graph = build_graph_with_external_data(&model, dir.path(), &options)
        .expect("同一ファイル共有は distinct ファイル数 1 件のため上限内");
    assert_eq!(graph.initializers.len(), 2);
}

// --- fd 予算（PR #2348 codex P1 是正の回帰テスト） ---

/// fd 予算テストで参照させる distinct な external data ファイル数。子
/// プロセスの fd soft limit（[`FD_BUDGET_SOFT_LIMIT`]）を大きく上回り、
/// かつ既定の `max_external_files`（4096）以内の本数にする: 旧構成（パス 1
/// で開いたハンドルをすべて保持）ならこの本数で確実に `EMFILE`
/// （`ExternalDataError::Io`）になり、`TooManyExternalFiles` には
/// 到達しない。
#[cfg(unix)]
const FD_BUDGET_FILE_COUNT: usize = 512;

/// 子プロセスへ課す fd の soft limit（`ulimit -S -n`）。テストハーネス
/// 自身・標準入出力・`base_dir` fd 等を含めても十分に動作し、かつ
/// [`FD_BUDGET_FILE_COUNT`] を大きく下回る値。
#[cfg(unix)]
const FD_BUDGET_SOFT_LIMIT: usize = 64;

/// 親テストが子プロセスへ「fd 制限下の子として実行中」であることを
/// 伝える環境変数。
#[cfg(unix)]
const FD_BUDGET_CHILD_ENV: &str = "FANDHE_ONNX_EXTERNAL_DATA_FD_BUDGET_CHILD";

/// Linux で現在プロセスが開いている fd 数を `/proc/self/fd` から数える
/// （`read_dir` 自身が開くディレクトリ fd を 1 件含むが、前後比較では
/// 相殺される）。
#[cfg(target_os = "linux")]
fn count_open_fds() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .expect("/proc/self/fd を列挙できない")
        .count()
}

/// 現在プロセスの fd soft limit（`RLIMIT_NOFILE` の `rlim_cur`）を
/// `getrlimit(2)` で読む（unix 共通。Linux 限定の `/proc/self/limits` に
/// 依存しないことで macOS 等でも子プロセスへの制限適用を確認できる。PR
/// #2348 security-auditor P2-3）。`libc` は本クレートの `cfg(unix)` 限定の
/// 通常依存（deps-policy.md 第 10 区分）で、統合テストからも参照できる。
#[cfg(unix)]
fn nofile_soft_limit() -> std::io::Result<libc::rlim_t> {
    let mut rl = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `&mut rl` はこの呼び出しの間生存する、初期化済みの
    // `libc::rlimit` への排他参照で、`getrlimit(2)` は成功時にのみこの
    // 出力バッファへ書き込む。`RLIMIT_NOFILE` は POSIX 定義の有効な資源
    // 種別。戻り値 0 を確認してから `rl` を読む。
    let ret = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut rl) };
    if ret != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(rl.rlim_cur)
}

/// 小さな external data ファイルを多数（[`FD_BUDGET_FILE_COUNT`] 本・
/// すべて別ファイル）参照するモデルが解決でき、全 initializer が正しい
/// 値になることを検査する。
///
/// 通常の `cargo test` では fd 制限を課さない機能テストとして動き、
/// [`many_small_external_files_do_not_exhaust_fd_limit_in_child_process`]
/// から fd soft limit を下げた子プロセスとして起動された場合は、(1) 制限が
/// 実際に効いていること（unix 共通。`getrlimit(RLIMIT_NOFILE)` で確認）、
/// (2) 制限を大きく超える本数のファイルを `EMFILE` なしで読めること、
/// (3) 解決の前後で開いている fd 数が増えていないこと（Linux のみ。
/// 子は `--test-threads=1` のため他テストの fd と混ざらない）も検査する。
/// `plan`／`load` が external data ファイルのハンドルを 1 つずつ開いては
/// 閉じ、同時保持数をファイル数に依存させないこと（`external_data.rs`
/// モジュール doc「ハンドル非保持の構成」節）の回帰テスト。
#[cfg(unix)]
#[test]
fn many_small_external_files_load_with_bounded_open_handles() {
    let in_child = std::env::var_os(FD_BUDGET_CHILD_ENV).is_some();
    if in_child {
        let soft = nofile_soft_limit().expect("getrlimit(RLIMIT_NOFILE) で soft limit を読めない");
        assert_eq!(
            soft, FD_BUDGET_SOFT_LIMIT as libc::rlim_t,
            "子プロセスに fd soft limit が適用されていない（テストが空振りになる）"
        );
    }

    let dir = TempDir::new("fd-budget-many-files");
    let mut tensors = Vec::with_capacity(FD_BUDGET_FILE_COUNT);
    for i in 0..FD_BUDGET_FILE_COUNT {
        let location = format!("w{i}.data");
        dir.write_file(&location, &(i as f32).to_le_bytes());
        tensors.push(external_tensor(
            &format!("w{i}"),
            vec![1],
            data_type::FLOAT,
            &location,
            None,
            Some("4"),
        ));
    }
    let model = model_with_initializers(tensors);

    #[cfg(target_os = "linux")]
    let fds_before = count_open_fds();
    let graph = build_graph_with_external_data(&model, dir.path(), &ExternalDataOptions::default())
        .unwrap_or_else(|e| {
            panic!(
                "{FD_BUDGET_FILE_COUNT} 本の小さな external data ファイルは fd を枯渇させずに \
                 解決できるはず（in_child={in_child}）: {e:?}"
            )
        });
    #[cfg(target_os = "linux")]
    if in_child {
        let fds_after = count_open_fds();
        assert!(
            fds_after <= fds_before,
            "解決後に fd が残っている（before={fds_before} after={fds_after}）"
        );
    }

    assert_eq!(graph.initializers.len(), FD_BUDGET_FILE_COUNT);
    for i in 0..FD_BUDGET_FILE_COUNT {
        match graph.initializers.get(&format!("w{i}")) {
            Some(RawTensor::F32 { data, .. }) => assert_eq!(data, &vec![i as f32]),
            other => panic!("w{i}: RawTensor::F32 を期待: {other:?}"),
        }
    }
}

/// [`many_small_external_files_load_with_bounded_open_handles`] を、fd の
/// soft limit を [`FD_BUDGET_SOFT_LIMIT`] へ下げた子プロセスで実行する
/// （PR #2348 codex P1 是正の回帰テスト）。`RLIMIT_NOFILE` はプロセス全体
/// に効くため、テストプロセス自身ではなく `sh -c 'ulimit -S -n …; exec …'`
/// 経由で起動した子プロセス（同じテストバイナリ）にだけ課す（`unsafe`・
/// 追加依存なし）。フィルタ不一致で 0 件実行のまま exit 0 になる空振りを
/// 防ぐため、子の出力に `1 passed` が含まれることも確認する。
#[cfg(unix)]
#[test]
fn many_small_external_files_do_not_exhaust_fd_limit_in_child_process() {
    if std::env::var_os(FD_BUDGET_CHILD_ENV).is_some() {
        // 子プロセス内では再帰起動しない（子は `--exact` で上のテストだけを
        // 実行するため通常は到達しないが、防御的に抜ける）。
        return;
    }
    let exe = std::env::current_exe().expect("テストバイナリのパスを取得できない");
    let output = std::process::Command::new("/bin/sh")
        .arg("-c")
        .arg(format!(
            "ulimit -S -n {FD_BUDGET_SOFT_LIMIT} && exec \"$0\" \"$@\""
        ))
        .arg(&exe)
        .arg("many_small_external_files_load_with_bounded_open_handles")
        .arg("--exact")
        .arg("--test-threads=1")
        .arg("--nocapture")
        .env(FD_BUDGET_CHILD_ENV, "1")
        .output()
        .expect("子プロセスを起動できない");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "fd soft limit={FD_BUDGET_SOFT_LIMIT} の子プロセスで {FD_BUDGET_FILE_COUNT} 本の \
         external data 解決が失敗した（status={:?}）\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    assert!(
        stdout.contains("1 passed"),
        "子プロセスで対象テストが実行されていない（空振り）\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}

// --- 異常系: 重複・重なり（A5） ---

#[cfg(any(unix, windows))]
#[test]
fn overlapping_regions_in_same_file_are_rejected() {
    let dir = TempDir::new("overlap");
    dir.write_file("f.data", &[0u8; 16]);
    let t_a = external_tensor(
        "a",
        vec![1],
        data_type::FLOAT,
        "f.data",
        Some("0"),
        Some("4"),
    );
    let t_b = external_tensor(
        "b",
        vec![1],
        data_type::FLOAT,
        "f.data",
        Some("2"),
        Some("4"),
    );
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: Vec::new(),
            name: "g".to_string(),
            initializer: vec![t_a, t_b],
            input: Vec::new(),
            output: Vec::new(),
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
        }),
        opset_import: Vec::new(),
    };
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(err, ExternalDataError::OverlappingRegion { .. }));
}

/// `length == 0` のテンソル（1 バイトも読まない）は区間としての幅を
/// 持たないため、既存の非ゼロ長区間の内側の `offset` を指していても
/// `OverlappingRegion` として拒否されないことを確認する（レビュー対応:
/// 0 バイト読み込みが誤って区間重複として扱われていた不具合の回帰
/// テスト。#2347）。
#[cfg(any(unix, windows))]
#[test]
fn zero_length_region_inside_existing_region_is_not_overlapping() {
    let dir = TempDir::new("zero-length-overlap");
    dir.write_file("f.data", &[0u8; 16]);
    let t_a = external_tensor(
        "a",
        vec![2],
        data_type::FLOAT,
        "f.data",
        Some("0"),
        Some("8"),
    );
    // offset=4 は t_a の区間 [0, 8) の内側だが、length=0 のため 1 バイトも
    // 読まない（dims=[0] で expected_bytes も 0 になる）。
    let t_b = external_tensor(
        "b",
        vec![0],
        data_type::FLOAT,
        "f.data",
        Some("4"),
        Some("0"),
    );
    let model = model_with_initializers(vec![t_a, t_b]);
    let graph = build_graph_with_external_data(&model, dir.path(), &ExternalDataOptions::default())
        .expect("length=0 のテンソルは既存区間の内側でも overlap 拒否されないはず");
    assert_eq!(graph.initializers.len(), 2);
}

/// 同一ファイルを指す `location` の表記ゆれ（`f.data`／`./f.data`）が
/// overlap 検出・ハンドル再利用を欺かないことの契約保証テスト（Cursor
/// Bugbot 指摘・PR #2348 review thread `PRRT_kwDOTuUCJc6mkYTr`）。
/// `Path` の `Eq`/`Hash` はコンポーネント単位のため、この表記ゆれ自体は
/// 旧実装（`base_dir.join(location)` の生文字列連結キー）でも実は畳み
/// 込まれていた（`Path` の実装詳細に依存した偶然の回避）。より厳密な
/// 回帰は [`overlapping_regions_via_hard_link_are_rejected`]（ハードリンク
/// はパス文字列としても `Path::components()` 正規化後も異なるが実体は
/// 同一のファイルであり、`file_key_for` の実体キー〈unix は dev/ino、
/// Windows はボリュームシリアル + ファイルインデックス〉でのみ検出できる）
/// を参照。
#[cfg(any(unix, windows))]
#[test]
fn overlapping_regions_via_equivalent_location_spelling_are_rejected() {
    let dir = TempDir::new("overlap-spelling");
    dir.write_file("f.data", &[0u8; 16]);
    let t_a = external_tensor(
        "a",
        vec![1],
        data_type::FLOAT,
        "f.data",
        Some("0"),
        Some("4"),
    );
    // `t_b` は `t_a` と同一ファイルへ表記だけを変えて（先頭に `./` を付与）
    // 重なる区間（offset=2, length=4 は offset=0, length=4 と重なる）を
    // 指す。正規化前の文字列結合キーだと "f.data" と "./f.data" が別ファイル
    // 扱いになり、この重複が検出できなかった。
    let t_b = external_tensor(
        "b",
        vec![1],
        data_type::FLOAT,
        "./f.data",
        Some("2"),
        Some("4"),
    );
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: Vec::new(),
            name: "g".to_string(),
            initializer: vec![t_a, t_b],
            input: Vec::new(),
            output: Vec::new(),
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
        }),
        opset_import: Vec::new(),
    };
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(err, ExternalDataError::OverlappingRegion { .. }));
}

/// 経路としては異なる（正規化後も異なる文字列の）が実体が同一のファイル
/// （ハードリンク）を 2 つの `location` から参照した場合でも overlap を
/// 検出することを確認する（Cursor Bugbot 指摘・PR #2348 review thread
/// `PRRT_kwDOTuUCJc6mkYTr` への厳密な回帰。旧実装
/// （`base_dir.join(location)` の生文字列連結キー、または `parts` から
/// 再構築した正規化済み相対パスをキーにする案）はどちらも「経路」だけ
/// を見るため、`f.data` と `alias.data`（`std::fs::hard_link` で作った
/// 同一 inode のハードリンク）を異なるファイルとして扱い、この重複を
/// 見落とす。`file_key_for` の dev/ino ベースキーはファイルの実体
/// そのもので同一性判定するため、この経路のみ検出できる）。
///
/// Windows でも `file_key_for` は `(dwVolumeSerialNumber, nFileIndex)` の
/// 実体キーで同一性を判定し、ハードリンクは同じキーに畳み込まれるため、
/// 本テストは `unix` / `windows` 共通で成立する（#2485。#2393 の Windows
/// 実機検証で cfg を一時的に広げて NTFS・ReFS の pass を確認済み。exFAT は
/// ハードリンク非対応で fixture 作成の段階で失敗する）。
#[cfg(any(unix, windows))]
#[test]
fn overlapping_regions_via_hard_link_are_rejected() {
    let dir = TempDir::new("overlap-hardlink");
    let real = dir.write_file("f.data", &[0u8; 16]);
    let alias = dir.path().join("alias.data");
    std::fs::hard_link(&real, &alias).expect(
        "hard_link の作成に失敗した（ハードリンクを作れるファイルシステムが前提。exFAT 等の非対応 FS では fixture 作成で失敗し、封じ込め判定の失敗ではない）",
    );

    let t_a = external_tensor(
        "a",
        vec![1],
        data_type::FLOAT,
        "f.data",
        Some("0"),
        Some("4"),
    );
    // `t_b` は `t_a` と同一 inode（ハードリンク）を異なる `location`
    // 文字列（`alias.data`）で参照し、重なる区間を指す。
    let t_b = external_tensor(
        "b",
        vec![1],
        data_type::FLOAT,
        "alias.data",
        Some("2"),
        Some("4"),
    );
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: Vec::new(),
            name: "g".to_string(),
            initializer: vec![t_a, t_b],
            input: Vec::new(),
            output: Vec::new(),
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
        }),
        opset_import: Vec::new(),
    };
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(err, ExternalDataError::OverlappingRegion { .. }));
}

#[cfg(any(unix, windows))]
#[test]
fn duplicate_identical_region_is_rejected() {
    let dir = TempDir::new("dup-region");
    dir.write_file("f.data", &[0u8; 16]);
    let t_a = external_tensor(
        "a",
        vec![1],
        data_type::FLOAT,
        "f.data",
        Some("0"),
        Some("4"),
    );
    let t_b = external_tensor(
        "b",
        vec![1],
        data_type::FLOAT,
        "f.data",
        Some("0"),
        Some("4"),
    );
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: Vec::new(),
            name: "g".to_string(),
            initializer: vec![t_a, t_b],
            input: Vec::new(),
            output: Vec::new(),
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
        }),
        opset_import: Vec::new(),
    };
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(err, ExternalDataError::OverlappingRegion { .. }));
}

/// 1 ファイルに重ならない長さ 1 の区間を指定するテンソルの本数（codex P0
/// 回帰テスト用）。旧実装（テンソルごとに同一ファイルの既存区間を全走査）
/// では約 N²/2 ≒ 12.5 億回の区間比較になる規模。
#[cfg(any(unix, windows))]
const MANY_REGIONS_COUNT: usize = 50_000;

/// `MANY_REGIONS_COUNT` 本の BOOL テンソル（dims=[1]・1 バイト）が 1 つの
/// `.data` ファイルの `[i, i+1)` を 1 件ずつ参照するモデルを作る（テンソル
/// は位置の逆順に並べ、入力順と位置順を一致させない）。
#[cfg(any(unix, windows))]
fn many_unit_regions_model(extra: Option<TensorProto>) -> ModelProto {
    let mut tensors: Vec<TensorProto> = (0..MANY_REGIONS_COUNT)
        .rev()
        .map(|i| {
            external_tensor(
                &format!("b{i}"),
                vec![1],
                data_type::BOOL,
                "many.data",
                Some(&i.to_string()),
                Some("1"),
            )
        })
        .collect();
    tensors.extend(extra);
    model_with_initializers(tensors)
}

/// 1 ファイルに重ならない短い区間を多数並べた入力を、区間重複検査が二次
/// 時間にならず現実的な時間で解決できることを検査する（codex P0 是正。
/// `max_external_files` は同一ファイルを 1 件としか数えず、
/// `max_total_bytes` も短い区間の合計しか制限しないため、上限では抑え
/// られない入力）。実時間の閾値では判定せず、件数を大きく取って既定の
/// テストタイムアウト内に通ること自体と、全値の正しさで確認する。
#[cfg(any(unix, windows))]
#[test]
fn many_non_overlapping_unit_regions_in_one_file_resolve() {
    let dir = TempDir::new("many-unit-regions");
    let bytes: Vec<u8> = (0..MANY_REGIONS_COUNT).map(|i| (i % 2) as u8).collect();
    dir.write_file("many.data", &bytes);
    let model = many_unit_regions_model(None);
    let graph = build_graph_with_external_data(&model, dir.path(), &ExternalDataOptions::default())
        .expect("重ならない区間だけなら解決できるはず");
    assert_eq!(graph.initializers.len(), MANY_REGIONS_COUNT);
    for i in [0, 1, 2, MANY_REGIONS_COUNT / 2, MANY_REGIONS_COUNT - 1] {
        match graph.initializers.get(&format!("b{i}")) {
            Some(RawTensor::Bool { data, .. }) => assert_eq!(data, &vec![i % 2 == 1]),
            other => panic!("b{i}: RawTensor::Bool を期待: {other:?}"),
        }
    }
}

/// 上と同じ多数区間の入力に、中ほどの区間と 1 バイトだけ重なるテンソルを
/// 1 件足すと `OverlappingRegion` で拒否し、重なる 2 テンソルの名前を
/// 報告する（ソート＋走査への置き換えで検出漏れが無いことの確認）。
#[cfg(any(unix, windows))]
#[test]
fn one_overlap_among_many_unit_regions_is_rejected_with_names() {
    let dir = TempDir::new("many-unit-regions-overlap");
    dir.write_file("many.data", &vec![0u8; MANY_REGIONS_COUNT]);
    let mid = MANY_REGIONS_COUNT / 2;
    let extra = external_tensor(
        "intruder",
        vec![2],
        data_type::BOOL,
        "many.data",
        Some(&mid.to_string()),
        Some("2"),
    );
    let model = many_unit_regions_model(Some(extra));
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    match err {
        ExternalDataError::OverlappingRegion {
            tensor_name,
            other_tensor_name,
        } => {
            // 同じ offset の区間は入力順で並ぶため、先に現れた `b{mid}` が
            // 既存側、後から現れた `intruder` が後側になる（決定的）。
            assert_eq!(tensor_name, "intruder");
            assert_eq!(other_tensor_name, format!("b{mid}"));
        }
        other => panic!("OverlappingRegion を期待: {other:?}"),
    }
}

#[test]
fn duplicate_initializer_name_is_rejected_before_io() {
    let dir = TempDir::new("dup-init-name");
    // ファイルを作らない: I/O の前に拒否されることを確認する。
    let t_a = external_tensor(
        "x",
        vec![1],
        data_type::FLOAT,
        "f.data",
        Some("0"),
        Some("4"),
    );
    let t_b = external_tensor(
        "x",
        vec![1],
        data_type::FLOAT,
        "f.data",
        Some("4"),
        Some("4"),
    );
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: Vec::new(),
            name: "g".to_string(),
            initializer: vec![t_a, t_b],
            input: Vec::new(),
            output: Vec::new(),
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
        }),
        opset_import: Vec::new(),
    };
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(
        err,
        ExternalDataError::DuplicateInitializerName { .. }
    ));
}

// --- 異常系: external_data のキー・data_location（A5・フィールド整合性） ---

#[test]
fn unknown_key_is_rejected() {
    let dir = TempDir::new("unknown-key");
    dir.write_file("f.data", &[0u8; 4]);
    let mut t = external_tensor("x", vec![1], data_type::FLOAT, "f.data", None, Some("4"));
    t.external_data.push(ext_entry("bogus", "1"));
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(err, ExternalDataError::UnknownKey { .. }));
}

#[test]
fn checksum_key_is_rejected_fail_closed() {
    let dir = TempDir::new("checksum-key");
    dir.write_file("f.data", &[0u8; 4]);
    let mut t = external_tensor("x", vec![1], data_type::FLOAT, "f.data", None, Some("4"));
    t.external_data.push(ext_entry("checksum", "deadbeef"));
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(err, ExternalDataError::ChecksumUnsupported { .. }));
}

#[test]
fn duplicate_key_is_rejected() {
    let dir = TempDir::new("dup-key");
    dir.write_file("f.data", &[0u8; 4]);
    let mut t = external_tensor("x", vec![1], data_type::FLOAT, "f.data", None, Some("4"));
    t.external_data.push(ext_entry("location", "f.data"));
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(err, ExternalDataError::DuplicateKey { .. }));
}

#[test]
fn missing_location_key_is_rejected() {
    let dir = TempDir::new("missing-location");
    let t = TensorProto {
        dims: vec![1],
        data_type: data_type::FLOAT,
        float_data: Vec::new(),
        int64_data: Vec::new(),
        name: "x".to_string(),
        raw_data: Vec::new(),
        external_data: vec![ext_entry("offset", "0")],
        data_location: data_location::EXTERNAL,
    };
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(err, ExternalDataError::MissingLocationKey { .. }));
}

#[test]
fn invalid_data_location_value_is_rejected() {
    let dir = TempDir::new("invalid-data-location");
    let t = TensorProto {
        dims: vec![1],
        data_type: data_type::FLOAT,
        float_data: Vec::new(),
        int64_data: Vec::new(),
        name: "x".to_string(),
        raw_data: Vec::new(),
        external_data: Vec::new(),
        data_location: 2,
    };
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(
        err,
        ExternalDataError::InvalidDataLocation { value: 2, .. }
    ));
}

#[test]
fn external_with_inline_raw_data_present_is_rejected() {
    let dir = TempDir::new("external-with-inline");
    dir.write_file("f.data", &[0u8; 4]);
    let mut t = external_tensor("x", vec![1], data_type::FLOAT, "f.data", None, Some("4"));
    t.raw_data = vec![0u8; 4];
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(
        err,
        ExternalDataError::InconsistentDataFields { .. }
    ));
}

#[test]
fn default_with_external_data_present_is_rejected() {
    let dir = TempDir::new("default-with-external");
    let t = TensorProto {
        dims: vec![1],
        data_type: data_type::FLOAT,
        float_data: vec![1.0],
        int64_data: Vec::new(),
        name: "x".to_string(),
        raw_data: Vec::new(),
        external_data: vec![ext_entry("location", "f.data")],
        data_location: data_location::DEFAULT,
    };
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(
        err,
        ExternalDataError::InconsistentDataFields { .. }
    ));
}

#[test]
fn nonexistent_base_dir_is_rejected() {
    let t = external_tensor("x", vec![1], data_type::FLOAT, "f.data", None, Some("4"));
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        Path::new("/nonexistent/base/dir/for/onnx/external/data/test"),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(err, ExternalDataError::InvalidBaseDir { .. }));
}

// --- Constant 属性テンソル（A1: run まで通す） ---

#[cfg(any(unix, windows))]
#[test]
fn constant_attribute_tensor_external_data_resolves_and_runs() {
    use fandhe_ai_onnx_interop::onnx::interp::run;
    use fandhe_ai_onnx_interop::onnx::proto::{AttributeProto, NodeProto, ValueInfoProto};

    let dir = TempDir::new("constant-attr");
    let bytes: [f32; 2] = [3.0, 4.0];
    let mut raw = Vec::new();
    for v in bytes {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    dir.write_file("const.onnx.data", &raw);

    let t = external_tensor("", vec![2], data_type::FLOAT, "const.onnx.data", None, None);
    let attr = AttributeProto {
        name: "value".to_string(),
        f: 0.0,
        i: 0,
        s: Vec::new(),
        t: Some(t),
        floats: Vec::new(),
        ints: Vec::new(),
        r#type: 4,
    };
    let node = NodeProto {
        input: Vec::new(),
        output: vec!["y".to_string()],
        name: "n_const".to_string(),
        op_type: "Constant".to_string(),
        attribute: vec![attr],
        domain: String::new(),
    };
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: vec![node],
            name: "g".to_string(),
            initializer: Vec::new(),
            input: Vec::new(),
            output: vec![ValueInfoProto {
                name: "y".to_string(),
            }],
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
        }),
        opset_import: Vec::new(),
    };

    let graph = build_graph_with_external_data(&model, dir.path(), &ExternalDataOptions::default())
        .expect("Constant 属性テンソルの external data 解決は成功するはず");
    let result = run(&graph, std::collections::HashMap::new()).expect("run は成功するはず");
    match &result["y"] {
        fandhe_ai_onnx_interop::onnx::interp::Value::F32(t) => {
            assert_eq!(t.contiguous().as_slice().unwrap(), &bytes)
        }
        other => panic!("Value::F32 を期待: {other:?}"),
    }
}

// --- A6: バイト列入口の回帰（external データを持つモデルは従来どおり拒否） ---

#[test]
fn bytes_entry_point_still_rejects_external_data_model() {
    use fandhe_ai_onnx_interop::onnx::proto::{decode_model, encode_model};

    let dir = TempDir::new("a6-bytes-entry");
    dir.write_file("w.onnx.data", &[0u8; 4]);
    let t = external_tensor(
        "w",
        vec![1],
        data_type::FLOAT,
        "w.onnx.data",
        None,
        Some("4"),
    );
    let model = model_with_initializer(t);

    // build_graph_with_external_data 経由なら成功することの対照確認。
    // 実解決の成功は unix・Windows 限定（それ以外では `resolve_and_open` が
    // 常に `UnsupportedPlatformForSecureResolve` で fail-closed 拒否する。
    // `external_data.rs` モジュール doc「それ以外（wasm32 等）」節）ため、
    // 対照確認部分だけを cfg で分け、それ以外ではその拒否契約を検査する
    // （Cursor Bugbot 指摘・PR #2348 対応・イシュー #2349 で Windows へ拡張。
    // 下の `from_bytes` 側の A6 回帰検査は external data 解決を経由しない
    // ため OS 非依存のまま実行する）。
    let ext_result =
        build_graph_with_external_data(&model, dir.path(), &ExternalDataOptions::default());
    #[cfg(any(unix, windows))]
    ext_result.expect("external data 経路は成功するはず");
    #[cfg(not(any(unix, windows)))]
    assert!(matches!(
        assert_external_err(ext_result),
        ExternalDataError::UnsupportedPlatformForSecureResolve { .. }
    ));

    // バイト列入口（decode_model -> build_graph）は data_location を一切
    // 参照しないため、external データを持つモデルは raw_data・float_data
    // とも空のまま扱われ従来どおり拒否される（A6）。
    let bytes = encode_model(&model);
    let decoded = decode_model(&bytes).expect("decode 自体は成功するはず");
    let result = build_graph(&decoded);
    assert!(matches!(
        result,
        Err(GraphError::RawDataByteLenMismatch {
            expected_bytes: 4,
            actual_bytes: 0,
            ..
        })
    ));
}

/// `base_dir` 内に FIFO（named pipe）が置かれていても `open`/`openat` が
/// ハングしないことを確認する（High・Cursor Bugbot 指摘
/// PRRT_kwDOTuUCJc6mk6-d の回帰テスト）。`O_NONBLOCK` 抜きの
/// `open`（`O_NOFOLLOW` のみ）は対向の reader/writer が現れるまで
/// 無期限にブロックし得るため、`build_graph_with_external_data` を
/// 別スレッドで呼び出し `recv_timeout` で有界に待つ（回帰が再発しても
/// テストプロセスごと無期限ハングさせないための防御。本 crate 本体には
/// この有界待ちは無く、修正そのものが `open` 呼び出し自体を
/// ノンブロッキングにする）。unix 全般（`no_follow_open` 経路。Linux は
/// `openat2`／逐次 `openat`、macOS・その他 unix は逐次 `openat`）が対象で、
/// 非 unix は `UnsupportedPlatformForSecureResolve` で即座に拒否されるため
/// 本シナリオ自体が発生しない。
#[cfg(unix)]
#[test]
fn fifo_location_does_not_hang_open() {
    // `mkfifo(2)` は production コードと同じく `libc` crate（本クレートの
    // `cfg(unix)` 限定依存。deps-policy.md 第 10 区分）経由で呼ぶ
    // （`mode_t` の幅は OS ごとに異なる〈Linux は u32・macOS は u16〉ため
    // 手書きの `extern "C"` 宣言は使わない）。

    let dir = TempDir::new("fifo-hang");
    let fifo_path = dir.path().join("pipe.data");
    let c_path = std::ffi::CString::new(fifo_path.as_os_str().as_encoded_bytes())
        .expect("パスに NUL は含まれない");
    // SAFETY: `c_path` はこの呼び出しの間生存する有効な NUL 終端 C 文字列。
    // `libc::mkfifo` は POSIX 標準関数で、失敗時は errno を設定し負値を返す
    // だけであり、他のメモリ安全性への影響はない。
    let rc = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
    assert_eq!(
        rc,
        0,
        "mkfifo に失敗した: {}",
        std::io::Error::last_os_error()
    );

    let t = external_tensor("x", vec![1], data_type::FLOAT, "pipe.data", None, Some("4"));
    let model = model_with_initializer(t);
    let base_dir = dir.path().to_path_buf();

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result =
            build_graph_with_external_data(&model, &base_dir, &ExternalDataOptions::default());
        // 受信側が既にタイムアウトして関数を抜けている場合は send が
        // 失敗し得るが、その場合はテスト側が既に fail 済みのため無視する。
        let _ = tx.send(result);
    });

    let result = rx.recv_timeout(std::time::Duration::from_secs(10)).expect(
        "build_graph_with_external_data が FIFO の open で無期限に \
             ハングした（O_NONBLOCK 欠落の回帰）",
    );
    let err = assert_external_err(result);
    // FIFO は通常ファイルではないため、open 自体は成功し得ても
    // （ノンブロッキングであれば）`is_file()` 検証で拒否される。
    assert!(matches!(
        err,
        ExternalDataError::InvalidLocation {
            reason: LocationRejectReason::NotRegularFile,
            ..
        }
    ));
}

/// 非 unix（Windows 等）契約テスト: external data を含むモデルは、
/// companion ファイルの実在有無やパス文法の正当性に関わらず、常に
/// `ExternalDataError::UnsupportedPlatformForSecureResolve` で fail-closed
/// に拒否されることを固定する（Cursor Bugbot 指摘・PRRT_kwDOTuUCJc6mrW…
/// 対応。`external_data.rs` モジュール doc「それ以外（wasm32 等）」節
/// 参照。Windows 対応はイシュー #2349 で実装済みのため対象から外れる）。
/// base_dir・companion ファイルはいずれも実在させない（unix・Windows 以外の
/// `resolve_and_open` は `location` の文字列検証のみ行い、ファイルの open を
/// 一切試みないため実在は不要）。
#[cfg(not(any(unix, windows)))]
#[test]
fn external_data_is_rejected_as_unsupported_platform_when_not_unix() {
    let dir = TempDir::new("non-unix-unsupported");
    let t = external_tensor(
        "x",
        vec![1],
        data_type::FLOAT,
        "f.data",
        Some("0"),
        Some("4"),
    );
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(
        err,
        ExternalDataError::UnsupportedPlatformForSecureResolve { .. }
    ));
}

/// unix・Windows 以外向け契約テスト（`location` の文法検証は OS 非依存で
/// resolve より前に行われるため、unix・Windows 以外でも
/// `UnsupportedPlatformForSecureResolve` ではなく具体的な
/// `InvalidLocation` が返ることを固定する。unix 版の
/// `symlink_escaping_base_dir_via_absolute_target_is_rejected` 等と対照的
/// に、こちらはファイルシステムへ一切触れない構文検証のみの契約）。
#[cfg(not(any(unix, windows)))]
#[test]
fn absolute_location_is_rejected_before_platform_check_on_non_unix() {
    assert_eq!(
        expect_location_reject("/abs/path.data", vec![1]),
        LocationRejectReason::Absolute
    );
}

// --- 確保の失敗可能化・所有権ベースのグラフ構築（PR #2348 codex P0 是正） ---

/// external 由来の initializer（FLOAT／INT64／BOOL／FLOAT16）・inline の
/// initializer（`raw_data` と `float_data` の両形式）・external な Constant
/// 属性テンソルを併せ持つモデルで、`build_graph_with_external_data`
/// （所有権ベースの `build_graph_owned` + 失敗可能確保の復号）が、旧経路
/// （複製 → `resolve_external_data` → バイト列入口と同じ `build_graph`）と
/// 完全に同じ `Graph` を返すことを固定する（`Graph: PartialEq`）。
#[cfg(any(unix, windows))]
#[test]
fn owned_build_matches_resolve_then_build_graph() {
    use fandhe_ai_onnx_interop::onnx::external_data::resolve_external_data;
    use fandhe_ai_onnx_interop::onnx::proto::{AttributeProto, NodeProto, ValueInfoProto};

    let dir = TempDir::new("owned-equivalence");
    let mut shared = Vec::new();
    for v in [1.5f32, -2.25, 3.0, f32::MIN_POSITIVE] {
        shared.extend_from_slice(&v.to_le_bytes());
    }
    let f32_len = shared.len();
    for v in [i64::MIN, -1, 0, i64::MAX] {
        shared.extend_from_slice(&v.to_le_bytes());
    }
    let i64_end = shared.len();
    // BOOL は非ゼロ→true（2・255 も true）。
    shared.extend_from_slice(&[0u8, 1, 2, 255]);
    let bool_end = shared.len();
    for v in [0.5f32, -65504.0, 1.0e-4] {
        shared.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    let f16_end = shared.len();
    for v in [9.0f32, 10.0] {
        shared.extend_from_slice(&v.to_le_bytes());
    }
    let const_end = shared.len();
    dir.write_file("all.onnx.data", &shared);

    let s = |n: usize| n.to_string();
    let f32_t = external_tensor(
        "wf",
        vec![2, 2],
        data_type::FLOAT,
        "all.onnx.data",
        Some("0"),
        Some(&s(f32_len)),
    );
    let i64_t = external_tensor(
        "wi",
        vec![4],
        data_type::INT64,
        "all.onnx.data",
        Some(&s(f32_len)),
        Some(&s(i64_end - f32_len)),
    );
    let bool_t = external_tensor(
        "wb",
        vec![4],
        data_type::BOOL,
        "./all.onnx.data",
        Some(&s(i64_end)),
        Some(&s(bool_end - i64_end)),
    );
    let f16_t = external_tensor(
        "wh",
        vec![3],
        data_type::FLOAT16,
        "all.onnx.data",
        Some(&s(bool_end)),
        Some(&s(f16_end - bool_end)),
    );
    let inline_raw = TensorProto {
        dims: vec![1],
        data_type: data_type::FLOAT,
        name: "inline_raw".to_string(),
        raw_data: 4.0f32.to_le_bytes().to_vec(),
        ..Default::default()
    };
    let inline_typed = TensorProto {
        dims: vec![2],
        data_type: data_type::FLOAT,
        name: "inline_typed".to_string(),
        float_data: vec![5.0, 6.0],
        ..Default::default()
    };
    let const_t = external_tensor(
        "",
        vec![2],
        data_type::FLOAT,
        "all.onnx.data",
        Some(&s(f16_end)),
        Some(&s(const_end - f16_end)),
    );
    let node = NodeProto {
        input: Vec::new(),
        output: vec!["c".to_string()],
        name: "n_const".to_string(),
        op_type: "Constant".to_string(),
        attribute: vec![AttributeProto {
            name: "value".to_string(),
            f: 0.0,
            i: 0,
            s: Vec::new(),
            t: Some(const_t),
            floats: Vec::new(),
            ints: Vec::new(),
            r#type: 4,
        }],
        domain: String::new(),
    };
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: vec![node],
            name: "g".to_string(),
            initializer: vec![f32_t, inline_raw, i64_t, bool_t, inline_typed, f16_t],
            input: Vec::new(),
            output: vec![ValueInfoProto {
                name: "c".to_string(),
            }],
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
        }),
        opset_import: Vec::new(),
    };

    let options = ExternalDataOptions::default();
    let owned = build_graph_with_external_data(&model, dir.path(), &options)
        .expect("所有権ベースの構築は成功するはず");
    let mut resolved = model.clone();
    resolve_external_data(&mut resolved, dir.path(), &options).expect("resolve は成功するはず");
    let reference = build_graph(&resolved).expect("旧経路の build_graph は成功するはず");
    assert_eq!(owned, reference);

    // 値そのものも確認する（両経路が同じ誤りを共有していないことの補強）。
    match owned.initializers.get("wb") {
        Some(RawTensor::Bool { data, .. }) => {
            assert_eq!(data.as_slice(), &[false, true, true, true])
        }
        other => panic!("RawTensor::Bool を期待: {other:?}"),
    }
    match owned.initializers.get("wi") {
        Some(RawTensor::I64 { data, shape }) => {
            assert_eq!(data.as_slice(), &[i64::MIN, -1, 0, i64::MAX]);
            assert_eq!(shape.as_slice(), &[4]);
        }
        other => panic!("RawTensor::I64 を期待: {other:?}"),
    }
}

/// 所有権ベースの構築でも、`build_graph` と同じ検証エラーを返す（ここでは
/// external initializer の復号後に行うトポロジ検証の失敗）。検証ロジックを
/// 共有していることの回帰。
#[cfg(any(unix, windows))]
#[test]
fn owned_build_keeps_topology_validation() {
    use fandhe_ai_onnx_interop::onnx::proto::ValueInfoProto;

    let dir = TempDir::new("owned-topology");
    dir.write_file("w.onnx.data", &1.0f32.to_le_bytes());
    let t = external_tensor("w", vec![1], data_type::FLOAT, "w.onnx.data", None, None);
    let mut model = model_with_initializer(t);
    if let Some(g) = model.graph.as_mut() {
        g.output.push(ValueInfoProto {
            name: "missing".to_string(),
        });
    }
    let err = build_graph_with_external_data(&model, dir.path(), &ExternalDataOptions::default())
        .expect_err("未生成のグラフ出力は拒否されるはず");
    assert_eq!(
        err,
        GraphError::UnknownGraphOutput {
            tensor_name: "missing".to_string()
        }
    );
}

/// 疎ファイル（`File::set_len` で 4 GiB へ伸ばした実データ 0 バイトの
/// ファイル）に対し、4 GiB の単一 FLOAT テンソルを宣言した小さなモデルは、
/// `max_total_bytes` を下げて渡せば読み込み・確保の前に
/// `TotalSizeLimitExceeded` で拒否される（低メモリ環境で呼び出し側が予算を
/// 下げる運用の回帰。実確保は一切しない）。64bit ターゲット限定（32bit
/// では dims の積が `usize` を超え `ElementCountOverflow` になるため）。
#[cfg(all(unix, target_pointer_width = "64"))]
#[test]
fn sparse_file_huge_tensor_is_rejected_by_lowered_budget_before_allocation() {
    let dir = TempDir::new("sparse-huge");
    let path = dir.write_file("big.onnx.data", &[]);
    let file_len: u64 = 1 << 32;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(file_len)
        .expect("疎ファイルの作成（set_len）に失敗した");
    let t = external_tensor(
        "big",
        vec![1 << 30],
        data_type::FLOAT,
        "big.onnx.data",
        Some("0"),
        Some(&file_len.to_string()),
    );
    let model = model_with_initializer(t);
    let options = ExternalDataOptions {
        max_total_bytes: 1 << 30,
        ..ExternalDataOptions::default()
    };
    let err = assert_external_err(build_graph_with_external_data(&model, dir.path(), &options));
    assert_eq!(
        err,
        ExternalDataError::TotalSizeLimitExceeded {
            limit: 1 << 30,
            requested: file_len,
        }
    );
}

/// codex P0 の再現経路そのもの: 小さな `.onnx` 相当のモデルと疎ファイル
/// （4 TiB へ `set_len` した実データ 0 バイトのファイル）で 4 TiB の単一
/// FLOAT テンソルを宣言し、`max_total_bytes` を事実上無制限にして上限検査を
/// 通過させると、`load` の区間バッファ確保が失敗し、プロセスを終了させずに
/// `AllocationFailed` を返す（是正前の `vec![0u8; n]` では同じ入力で
/// テストプロセスごと abort した）。
///
/// 確保要求がアロケータに拒否されることを前提にするため、Linux の
/// `vm.overcommit_memory` が 0（ヒューリスティック。物理メモリ＋スワップを
/// 明らかに超える要求を拒否）または 2（厳格）の環境でのみ実行する。1
/// （常に許可）では確保が成功し 4 TiB の読み込みへ進みうるため実行しない
/// （確保成功後のページ実コミット時の OOM は受容済みの残存リスク。
/// `docs/onnx-external-data-decision.md` 5 節）。
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
#[test]
fn sparse_file_unallocatable_tensor_returns_allocation_failed_instead_of_abort() {
    let mode = std::fs::read_to_string("/proc/sys/vm/overcommit_memory").unwrap_or_default();
    if !matches!(mode.trim(), "0" | "2") {
        eprintln!("vm.overcommit_memory={:?} のため実行しない", mode.trim());
        return;
    }
    let dir = TempDir::new("sparse-unallocatable");
    let path = dir.write_file("huge.onnx.data", &[]);
    let file_len: u64 = 1 << 42;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(file_len)
        .expect("疎ファイルの作成（set_len）に失敗した");
    let t = external_tensor(
        "huge",
        vec![1 << 40],
        data_type::FLOAT,
        "huge.onnx.data",
        Some("0"),
        Some(&file_len.to_string()),
    );
    let model = model_with_initializer(t);
    let options = ExternalDataOptions {
        max_total_bytes: u64::MAX,
        ..ExternalDataOptions::default()
    };
    let err = assert_external_err(build_graph_with_external_data(&model, dir.path(), &options));
    assert_eq!(
        err,
        ExternalDataError::AllocationFailed {
            tensor_name: "huge".to_string(),
            bytes: file_len,
        }
    );
}

// --- 実行経路・export 経路のメモリ確保（PR #2348 codex P0 是正 2 回目） ---

/// `name` を出力する `Constant` ノード（`value` 属性に `t`）。
#[cfg(any(unix, windows))]
fn constant_node(
    name: &str,
    output: &str,
    t: TensorProto,
) -> fandhe_ai_onnx_interop::onnx::proto::NodeProto {
    use fandhe_ai_onnx_interop::onnx::proto::{AttributeProto, NodeProto, attribute_type};
    NodeProto {
        input: Vec::new(),
        output: vec![output.to_string()],
        name: name.to_string(),
        op_type: "Constant".to_string(),
        attribute: vec![AttributeProto {
            name: "value".to_string(),
            t: Some(t),
            r#type: attribute_type::TENSOR,
            ..Default::default()
        }],
        domain: String::new(),
    }
}

/// ノード列・initializer・グラフ出力を指定したモデル。
#[cfg(any(unix, windows))]
fn model_with(
    nodes: Vec<fandhe_ai_onnx_interop::onnx::proto::NodeProto>,
    initializers: Vec<TensorProto>,
    outputs: &[&str],
) -> ModelProto {
    use fandhe_ai_onnx_interop::onnx::proto::ValueInfoProto;
    ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: nodes,
            name: "g".to_string(),
            initializer: initializers,
            input: Vec::new(),
            output: outputs
                .iter()
                .map(|n| ValueInfoProto {
                    name: n.to_string(),
                })
                .collect(),
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
        }),
        opset_import: Vec::new(),
    }
}

/// 実行時値を dtype・shape・要素の bit 列へ正規化する（NaN を含む f32／f16
/// も bit 単位で比較するため）。
#[cfg(any(unix, windows))]
fn value_bits(
    v: &fandhe_ai_onnx_interop::onnx::interp::Value,
) -> (&'static str, Vec<usize>, Vec<u64>) {
    use fandhe_ai_onnx_interop::onnx::interp::Value;
    match v {
        Value::F32(t) => {
            let c = t.contiguous();
            let bits = c
                .as_slice()
                .unwrap()
                .iter()
                .map(|x| u64::from(x.to_bits()))
                .collect();
            ("f32", t.shape().to_vec(), bits)
        }
        Value::I64(t) => {
            let c = t.contiguous();
            let bits = c.as_slice().unwrap().iter().map(|x| *x as u64).collect();
            ("i64", t.shape().to_vec(), bits)
        }
        Value::Bool(t) => {
            let c = t.contiguous();
            let bits = c
                .as_slice()
                .unwrap()
                .iter()
                .map(|x| u64::from(*x))
                .collect();
            ("bool", t.shape().to_vec(), bits)
        }
        Value::F16(t) => {
            let c = t.contiguous();
            let bits = c
                .as_slice()
                .unwrap()
                .iter()
                .map(|x| u64::from(x.to_bits()))
                .collect();
            ("f16", t.shape().to_vec(), bits)
        }
    }
}

/// external data 由来の `Constant` 属性テンソル（4 dtype）と initializer を
/// 持つモデルの推論結果が、同じ値を inline で持つモデル（バイト列入口と同じ
/// `build_graph`）と bit 単位で一致し、同じ `Graph` で複数回 `run` しても
/// 結果が変わらないこと（実行時の復号・initializer 複製を失敗可能確保へ
/// 置き換えた後も数値結果が不変であることの回帰テスト）。あわせて export
/// （`build_model_proto` → `try_encode_model`）が inline モデルの export と
/// 同一バイト列になることも確認する。
#[cfg(any(unix, windows))]
#[test]
fn external_constant_attributes_run_bit_identical_to_inline_across_runs() {
    use fandhe_ai_onnx_interop::onnx::export::{
        ExportOptions, build_model_proto, try_encode_model,
    };
    use fandhe_ai_onnx_interop::onnx::interp::run;
    use fandhe_ai_onnx_interop::onnx::proto::encode_model;

    let f32_raw: Vec<u8> = [1.5f32, -0.0, f32::NAN, f32::MIN_POSITIVE]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let i64_raw: Vec<u8> = [i64::MIN, -1, 0, i64::MAX]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let bool_raw: Vec<u8> = vec![0, 1, 2, 255];
    let f16_raw: Vec<u8> = [half::f16::from_f32(0.25), half::f16::NAN]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let w_raw: Vec<u8> = [3.0f32, -4.5]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    // 1 つの `.data` ファイルへ連結し、offset で各テンソルを指す。
    // （ノード名〈空なら initializer〉, 出力名, dtype, dims, 生バイト列）
    let parts = [
        ("n_f", "cf", data_type::FLOAT, vec![2, 2], &f32_raw),
        ("n_i", "ci", data_type::INT64, vec![4], &i64_raw),
        ("n_b", "cb", data_type::BOOL, vec![4], &bool_raw),
        ("n_h", "ch", data_type::FLOAT16, vec![2], &f16_raw),
        ("", "w", data_type::FLOAT, vec![2], &w_raw),
    ];
    let dir = TempDir::new("constant-bit-identical");
    let mut blob = Vec::new();
    let mut ext_nodes = Vec::new();
    let mut inline_nodes = Vec::new();
    let mut ext_inits = Vec::new();
    let mut inline_inits = Vec::new();
    for (node_name, out, dt, dims, raw) in parts {
        let offset = blob.len().to_string();
        let length = raw.len().to_string();
        blob.extend_from_slice(raw);
        let tensor_name = if node_name.is_empty() { out } else { "" };
        let ext = external_tensor(
            tensor_name,
            dims.clone(),
            dt,
            "all.onnx.data",
            Some(&offset),
            Some(&length),
        );
        let mut inline = ext.clone();
        inline.external_data.clear();
        inline.data_location = data_location::DEFAULT;
        inline.raw_data = raw.clone();
        if node_name.is_empty() {
            ext_inits.push(ext);
            inline_inits.push(inline);
        } else {
            ext_nodes.push(constant_node(node_name, out, ext));
            inline_nodes.push(constant_node(node_name, out, inline));
        }
    }
    dir.write_file("all.onnx.data", &blob);
    let outputs = ["cf", "ci", "cb", "ch", "w"];
    let ext_model = model_with(ext_nodes, ext_inits, &outputs);
    let inline_model = model_with(inline_nodes, inline_inits, &outputs);

    let ext_graph =
        build_graph_with_external_data(&ext_model, dir.path(), &ExternalDataOptions::default())
            .expect("external data の解決は成功するはず");
    let inline_graph = build_graph(&inline_model).expect("inline モデルの構築は成功するはず");

    let reference = run(&inline_graph, std::collections::HashMap::new()).unwrap();
    for round in 0..3 {
        let out = run(&ext_graph, std::collections::HashMap::new())
            .unwrap_or_else(|e| panic!("run {round} 回目が失敗した: {e:?}"));
        assert_eq!(out.len(), outputs.len());
        for name in outputs {
            assert_eq!(
                value_bits(&out[name]),
                value_bits(&reference[name]),
                "run {round} 回目の出力 {name} が inline モデルと bit 一致しない"
            );
        }
    }

    let options = ExportOptions::default();
    let ext_bytes = try_encode_model(&build_model_proto(&ext_graph, &options).unwrap()).unwrap();
    let inline_proto = build_model_proto(&inline_graph, &options).unwrap();
    assert_eq!(ext_bytes, encode_model(&inline_proto));
}

/// codex P0（2 回目）の指摘経路のうち読み込み段: 疎ファイル（4 TiB）で
/// 4 TiB の `Constant` 属性テンソルを宣言すると、属性テンソルの slot も
/// initializer と同じ `load` の区間バッファ（失敗可能確保）で
/// `AllocationFailed` になりプロセスは終了しない。**実行時の復号には到達
/// しない**（raw の読み込みで先に止まる）ため、実行時経路は
/// [`runtime_allocation_failures_are_typed_errors_under_address_space_limit`]
/// が別に検証する。ゲート（Linux・`vm.overcommit_memory` 0／2・64bit）は
/// [`sparse_file_unallocatable_tensor_returns_allocation_failed_instead_of_abort`]
/// と同じ。
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
#[test]
fn sparse_file_unallocatable_constant_attribute_returns_allocation_failed() {
    let mode = std::fs::read_to_string("/proc/sys/vm/overcommit_memory").unwrap_or_default();
    if !matches!(mode.trim(), "0" | "2") {
        eprintln!("vm.overcommit_memory={:?} のため実行しない", mode.trim());
        return;
    }
    let dir = TempDir::new("sparse-unallocatable-constant");
    let path = dir.write_file("huge.onnx.data", &[]);
    let file_len: u64 = 1 << 42;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(file_len)
        .expect("疎ファイルの作成（set_len）に失敗した");
    let t = external_tensor(
        "",
        vec![1 << 40],
        data_type::FLOAT,
        "huge.onnx.data",
        Some("0"),
        Some(&file_len.to_string()),
    );
    let model = model_with(vec![constant_node("n_const", "y", t)], Vec::new(), &["y"]);
    let options = ExternalDataOptions {
        max_total_bytes: u64::MAX,
        ..ExternalDataOptions::default()
    };
    let err = assert_external_err(build_graph_with_external_data(&model, dir.path(), &options));
    assert_eq!(
        err,
        ExternalDataError::AllocationFailed {
            tensor_name: "n_const:value".to_string(),
            bytes: file_len,
        }
    );
}

/// [`runtime_allocation_failure_child`] を子プロセスとして起動したことを
/// 示す環境変数（値はシナリオ名）。
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
const RUNTIME_ALLOC_CHILD_ENV: &str = "FANDHE_ONNX_RUNTIME_ALLOC_CHILD";

/// 子プロセスで読み込む external テンソルの大きさ（バイト）。読み込み済みの
/// 状態から「さらに同じ大きさを 1 回確保する」実行時処理だけを、アドレス
/// 空間上限（`RLIMIT_AS`）で失敗させる。
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
const RUNTIME_ALLOC_TENSOR_BYTES: u64 = 64 << 20;

/// 現在プロセスの仮想メモリ量（`/proc/self/status` の `VmSize`。バイト）。
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
fn vm_size_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    let line = status
        .lines()
        .find(|l| l.starts_with("VmSize:"))
        .expect("VmSize 行が無い");
    let kib: u64 = line
        .trim_start_matches("VmSize:")
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .expect("VmSize を数値として読めない");
    kib * 1024
}

/// 現在プロセスのアドレス空間 soft limit を `soft` へ下げる（hard limit は
/// 変えない）。hard limit がそれより小さい場合は hard limit を使う。
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
fn lower_address_space_soft_limit(soft: u64) {
    let mut rl = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `&mut rl` はこの呼び出しの間生存する初期化済み `libc::rlimit`
    // への排他参照で、`getrlimit(2)` は成功時にのみ書き込む。`RLIMIT_AS` は
    // 有効な資源種別。戻り値 0 を確認してから `rl` を読む。
    let ret = unsafe { libc::getrlimit(libc::RLIMIT_AS, &mut rl) };
    assert_eq!(ret, 0, "getrlimit(RLIMIT_AS) に失敗した");
    let new = libc::rlimit {
        rlim_cur: (soft as libc::rlim_t).min(rl.rlim_max),
        rlim_max: rl.rlim_max,
    };
    // SAFETY: `&new` はこの呼び出しの間生存する初期化済み `libc::rlimit`
    // への共有参照で、`setrlimit(2)` は読むだけ。soft ≤ hard を満たす値を
    // 渡すため hard limit の引き上げ（特権が要る）にはならない。
    let ret = unsafe { libc::setrlimit(libc::RLIMIT_AS, &new) };
    assert_eq!(ret, 0, "setrlimit(RLIMIT_AS) に失敗した");
}

/// 子プロセス側の本体（親テスト
/// [`runtime_allocation_failures_are_typed_errors_under_address_space_limit`]
/// から `RUNTIME_ALLOC_CHILD_ENV` 付きで起動された場合のみ動く。通常の
/// `cargo test` では何もせず pass する）。
///
/// [`RUNTIME_ALLOC_TENSOR_BYTES`] の external テンソル（疎ファイル。読み込み
/// で 0 が実コミットされる）を `build_graph_with_external_data` で読み込んだ
/// 後、アドレス空間の soft limit を「現在の `VmSize` ＋ テンソルの半分」へ
/// 下げ、テンソル 1 個分の追加確保を要する処理を実行する:
///
/// - `initializer`: `interp::run` が initializer を実行時値へ複製する
/// - `constant`: `interp::run` が `Constant` 属性テンソルを復号する
/// - `autograd`: `BoundGraph::bind` が initializer を実行時値へ複製する
/// - `export`: `export::build_model_proto` が initializer をバイト列化する
///
/// いずれも `AllocationFailed`（テンソル名・要求バイト数つき）になり、
/// プロセスは終了しないこと。是正前（無条件の `clone`／`collect`）は
/// 同じ入力で子プロセスが SIGABRT で落ちる。
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
#[test]
fn runtime_allocation_failure_child() {
    use fandhe_ai_onnx_interop::onnx::autograd::{AutogradError, BindOptions, BoundGraph};
    use fandhe_ai_onnx_interop::onnx::export::{ExportError, ExportOptions, build_model_proto};
    use fandhe_ai_onnx_interop::onnx::interp::{InterpError, run};

    let Some(scenario) = std::env::var_os(RUNTIME_ALLOC_CHILD_ENV) else {
        return;
    };
    let scenario = scenario.to_string_lossy().into_owned();
    let n = RUNTIME_ALLOC_TENSOR_BYTES;
    let dir = TempDir::new("runtime-alloc-child");
    let path = dir.write_file("big.onnx.data", &[]);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(n)
        .unwrap();
    let dims = vec![(n / 4) as i64];
    let length = n.to_string();
    let model = if scenario == "constant" {
        let t = external_tensor(
            "",
            dims,
            data_type::FLOAT,
            "big.onnx.data",
            None,
            Some(&length),
        );
        model_with(vec![constant_node("n_const", "y", t)], Vec::new(), &["y"])
    } else {
        let t = external_tensor(
            "w",
            dims,
            data_type::FLOAT,
            "big.onnx.data",
            None,
            Some(&length),
        );
        model_with(Vec::new(), vec![t], &["w"])
    };
    let graph = build_graph_with_external_data(&model, dir.path(), &ExternalDataOptions::default())
        .expect("読み込みは上限を下げる前に成功するはず");

    // 被検対象でない準備（`Tape`・export オプション）は上限を下げる前に
    // 済ませ、上限下では「テンソル 1 個分の追加確保を要する処理」だけを
    // 実行する（無関係な初期化の確保失敗で誤検知しないため）。
    let tape = fandhe_ai_autodiff::Tape::new();
    let export_options = ExportOptions::default();

    lower_address_space_soft_limit(vm_size_bytes() + n / 2);

    let (expected_name, got) = match scenario.as_str() {
        "initializer" => match run(&graph, std::collections::HashMap::new()) {
            Err(InterpError::AllocationFailed { tensor_name, bytes }) => {
                ("w", (tensor_name, bytes))
            }
            other => panic!("InterpError::AllocationFailed を期待したが {other:?}"),
        },
        "constant" => match run(&graph, std::collections::HashMap::new()) {
            Err(InterpError::AllocationFailed { tensor_name, bytes }) => {
                ("n_const:value", (tensor_name, bytes))
            }
            other => panic!("InterpError::AllocationFailed を期待したが {other:?}"),
        },
        "autograd" => match BoundGraph::bind(&graph, &tape, &BindOptions::default()) {
            Err(AutogradError::Interp(InterpError::AllocationFailed { tensor_name, bytes })) => {
                ("w", (tensor_name, bytes))
            }
            Err(other) => panic!("AllocationFailed を期待したが {other:?}"),
            Ok(_) => panic!("AllocationFailed を期待したが bind が成功した"),
        },
        "export" => match build_model_proto(&graph, &export_options) {
            Err(ExportError::AllocationFailed { tensor_name, bytes }) => {
                ("w", (tensor_name, bytes))
            }
            other => panic!("ExportError::AllocationFailed を期待したが {other:?}"),
        },
        other => panic!("未知のシナリオ: {other}"),
    };
    assert_eq!(got, (expected_name.to_string(), n));
}

/// 実行経路（`interp::run` の initializer 複製・`Constant` 属性テンソルの
/// 復号・`autograd` の bind）と export 経路の失敗可能確保を、実際の確保
/// 失敗で検証する（PR #2348 codex P0 是正 2 回目の回帰テスト）。
///
/// 確保失敗はアドレス空間上限（`RLIMIT_AS`）で起こすため
/// `vm.overcommit_memory` の設定に依存しない。`RLIMIT_AS` はプロセス全体に
/// 効くため、同じテストバイナリを子プロセスとして起動し
/// [`runtime_allocation_failure_child`] だけを `--exact` で実行する（テスト
/// プロセス自身の上限は変えない）。子が異常終了（是正前の abort）した場合・
/// 対象テストが実行されなかった場合（空振り）はいずれも失敗にする。
/// `RLIMIT_AS` が確保要求を拒否することを前提にするため Linux 限定（macOS は
/// `RLIMIT_AS` を強制しない）。
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
#[test]
fn runtime_allocation_failures_are_typed_errors_under_address_space_limit() {
    if std::env::var_os(RUNTIME_ALLOC_CHILD_ENV).is_some() {
        return;
    }
    let exe = std::env::current_exe().expect("テストバイナリのパスを取得できない");
    for scenario in ["initializer", "constant", "autograd", "export"] {
        let output = std::process::Command::new(&exe)
            .arg("runtime_allocation_failure_child")
            .arg("--exact")
            .arg("--test-threads=1")
            .arg("--nocapture")
            .env(RUNTIME_ALLOC_CHILD_ENV, scenario)
            .output()
            .expect("子プロセスを起動できない");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "シナリオ {scenario}: アドレス空間上限下の子プロセスが失敗した（status={:?}。\
             abort なら確保が無条件のまま）\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
        assert!(
            stdout.contains("1 passed"),
            "シナリオ {scenario}: 子プロセスで対象テストが実行されていない（空振り）\n\
             stdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }
}

// --- Windows 向け封じ込めオープン（イシュー #2349。`win_contained_open`） ---
//
// このブロックのテストは `#[cfg(windows)]` のため Windows 実機・Windows
// ターゲット CI（`ci.yml` の `cargo clippy（onnx-interop の external
// data・Windows ターゲット）` ステップ）でのみコンパイル・実行される。
// Linux CI ではコンパイル対象にすら入らない。実機での実行結果は
// `docs/onnx-external-data-decision.md` §8 の申し送りを参照。

/// junction（`mklink /J`）が最終成分（ディレクトリを指す）・途中成分の
/// いずれにあっても `ReparsePoint` で拒否されることを確認する（計画
/// §5.3。`win_contained_open` モジュール doc 2. の事後チェックの回帰）。
#[cfg(windows)]
fn make_junction(link: &Path, target: &Path) {
    let status = std::process::Command::new("cmd")
        .args([
            "/C",
            "mklink",
            "/J",
            &link.to_string_lossy(),
            &target.to_string_lossy(),
        ])
        .status()
        .expect("mklink の起動に失敗した");
    assert!(status.success(), "mklink /J が失敗した: {status:?}");
}

#[cfg(windows)]
#[test]
fn junction_intermediate_component_is_rejected() {
    let dir = TempDir::new("win-junction-mid");
    let real_dir = dir.path().join("real_dir");
    std::fs::create_dir_all(&real_dir).unwrap();
    std::fs::write(real_dir.join("f.data"), [0u8; 4]).unwrap();
    let link_dir = dir.path().join("link_dir");
    make_junction(&link_dir, &real_dir);

    let t = external_tensor(
        "x",
        vec![1],
        data_type::FLOAT,
        "link_dir/f.data",
        None,
        Some("4"),
    );
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(
        err,
        ExternalDataError::InvalidLocation {
            reason: LocationRejectReason::ReparsePoint,
            ..
        }
    ));
}

/// 最終成分自体が junction（ディレクトリを指す）の場合の拒否。junction は
/// ディレクトリを指すため `location` はそのディレクトリ内のファイルを指す
/// 形にし、junction 自体を途中成分として辿らせる（ファイルへの junction は
/// Windows に存在しないため、この形が最終成分＝junction の唯一の到達経路）。
#[cfg(windows)]
#[test]
fn junction_as_final_directory_component_is_rejected() {
    let dir = TempDir::new("win-junction-final-dir");
    let real_dir = dir.path().join("real_dir");
    std::fs::create_dir_all(&real_dir).unwrap();
    let link_dir = dir.path().join("link_dir");
    make_junction(&link_dir, &real_dir);

    // junction 自体をディレクトリとして参照する（内部のファイルではなく
    // junction 自体が最終成分になるよう、junction の中身は空のままにする）。
    let t = external_tensor("x", vec![1], data_type::FLOAT, "link_dir", None, Some("4"));
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    // junction はディレクトリ属性も reparse point 属性も持つため、
    // 実装の検査順序次第で `ReparsePoint` または `NotRegularFile` の
    // いずれかになりうる（`win_contained_open::resolve_and_open` は
    // reparse point 属性を最初に検査するため `ReparsePoint` を期待する）。
    assert!(matches!(
        err,
        ExternalDataError::InvalidLocation {
            reason: LocationRejectReason::ReparsePoint,
            ..
        }
    ));
}

/// シンボリックリンク（最終成分）の拒否。作成には
/// `SeCreateSymbolicLinkPrivilege` または Windows の開発者モードが要るため
/// 通常 CI では走らない。
#[cfg(windows)]
#[test]
#[ignore = "要 SeCreateSymbolicLinkPrivilege または開発者モード（Windows 実機限定）"]
fn symlink_final_component_is_rejected_on_windows() {
    let dir = TempDir::new("win-symlink-final");
    let real = dir.write_file("real.onnx.data", &[0u8; 4]);
    let link = dir.path().join("link.onnx.data");
    std::os::windows::fs::symlink_file(&real, &link).expect("symlink_file の作成に失敗した");

    let t = external_tensor(
        "x",
        vec![1],
        data_type::FLOAT,
        "link.onnx.data",
        None,
        Some("4"),
    );
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(
        err,
        ExternalDataError::InvalidLocation {
            reason: LocationRejectReason::ReparsePoint,
            ..
        }
    ));
}

/// シンボリックリンク（途中成分）の拒否。作成の権限要件は上記と同じ。
#[cfg(windows)]
#[test]
#[ignore = "要 SeCreateSymbolicLinkPrivilege または開発者モード（Windows 実機限定）"]
fn symlink_intermediate_component_is_rejected_on_windows() {
    let dir = TempDir::new("win-symlink-mid");
    let real_dir = dir.path().join("real_dir");
    std::fs::create_dir_all(&real_dir).unwrap();
    std::fs::write(real_dir.join("f.data"), [0u8; 4]).unwrap();
    let link_dir = dir.path().join("link_dir");
    std::os::windows::fs::symlink_dir(&real_dir, &link_dir).expect("symlink_dir の作成に失敗した");

    let t = external_tensor(
        "x",
        vec![1],
        data_type::FLOAT,
        "link_dir/f.data",
        None,
        Some("4"),
    );
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(
        err,
        ExternalDataError::InvalidLocation {
            reason: LocationRejectReason::ReparsePoint,
            ..
        }
    ));
}

/// Windows 固有の字句検査（代替データストリーム・予約デバイス名・禁止
/// 文字・末尾のドット/空白）が `build_graph_with_external_data` の pass 1
/// （ファイルシステムへ一切触れない構文検証）で拒否されることを確認する
/// （`windows_component_reject_reason` の単体テスト
/// 〈`external_data.rs::windows_lexical_tests`〉の統合版）。
#[cfg(windows)]
#[test]
fn windows_lexical_rejections_via_full_pipeline() {
    assert_eq!(
        expect_location_reject("a.data:stream", vec![1]),
        LocationRejectReason::AlternateDataStream
    );
    assert_eq!(
        expect_location_reject("CON", vec![1]),
        LocationRejectReason::ReservedDeviceName
    );
    assert_eq!(
        expect_location_reject("nul.data", vec![1]),
        LocationRejectReason::ReservedDeviceName
    );
    assert_eq!(
        expect_location_reject("COM1.data", vec![1]),
        LocationRejectReason::ReservedDeviceName
    );
    assert_eq!(
        expect_location_reject("x.", vec![1]),
        LocationRejectReason::InvalidComponentName
    );
    assert_eq!(
        expect_location_reject("a*b", vec![1]),
        LocationRejectReason::InvalidComponentName
    );
}

/// 他プロセス（相当。同一プロセス内の別ハンドル）が external data ファイル
/// を書き込みハンドルで開いている間、`build_graph_with_external_data` が
/// 共有違反で `Io` として fail-closed に拒否することを確認する
/// （`win_contained_open` の最終成分の共有モードが `FILE_SHARE_READ` のみ
/// であることの回帰。計画 §5.3）。
#[cfg(windows)]
#[test]
fn write_handle_causes_sharing_violation_rejection() {
    let dir = TempDir::new("win-share-violation");
    let path = dir.write_file("w.onnx.data", &[0u8; 4]);
    // 書き込みアクセス（`GENERIC_WRITE`）を持つハンドルを保持したまま
    // 読み込みを試みる。Rust std の既定共有モード（`FILE_SHARE_READ |
    // FILE_SHARE_WRITE | FILE_SHARE_DELETE`）はこのハンドル自体は寛容だが、
    // 競合判定は双方向で行われる: `resolve_and_open` の最終ファイル open
    // （`win_contained_open::open_component`）は `FILE_SHARE_READ` のみを
    // 要求するため（`FILE_SHARE_WRITE` を含まない）、このハンドルが持つ
    // `GENERIC_WRITE` アクセスと衝突し `ERROR_SHARING_VIOLATION` になる
    // （MS Learn "CreateFileA/W" の共有モード判定規則: 新規 open の
    // 共有モードが既存ハンドルのアクセス権を許容しない場合に失敗する）。
    let _writer = std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("書き込みハンドルの取得に失敗した");

    let t = external_tensor(
        "x",
        vec![1],
        data_type::FLOAT,
        "w.onnx.data",
        None,
        Some("4"),
    );
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        dir.path(),
        &ExternalDataOptions::default(),
    ));
    assert!(
        matches!(err, ExternalDataError::Io { .. }),
        "共有違反は Io として拒否されるはずだが: {err:?}"
    );
}

/// UNC パス（`\\localhost\C$\...`）を `base_dir` に指定すると
/// `InvalidBaseDir` で拒否されることを確認する（計画 §3.3・§5.3。管理
/// 共有の有効化状態に依存するため通常 CI では走らない）。
#[cfg(windows)]
#[test]
#[ignore = "管理共有（C$）の設定に依存する（Windows 実機限定）"]
fn unc_base_dir_is_rejected() {
    let unc = Path::new(r"\\localhost\C$\Windows\Temp");
    let t = external_tensor("x", vec![1], data_type::FLOAT, "f.data", None, Some("4"));
    let model = model_with_initializer(t);
    let err = assert_external_err(build_graph_with_external_data(
        &model,
        unc,
        &ExternalDataOptions::default(),
    ));
    assert!(matches!(err, ExternalDataError::InvalidBaseDir { .. }));
}
