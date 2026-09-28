//! ONNX external data（イシュー #2347）の合成入力テスト。
//!
//! `fandhe_ai_onnx_interop::onnx::external_data` の A2〜A5 検証（パス
//! トラバーサル・symlink・数値文法・範囲・重複・合計上限）を、実際に
//! ファイルシステム上へ合成した `.data` ファイルへ対して確認する。
//! PyTorch 実生成 fixture 側の判定は
//! `tests/onnx_interp_pytorch_cnn_fixture.rs` を参照（本ファイルは torch に
//! 依存しない）。
//!
//! 一時ディレクトリは `std::env::temp_dir()` 配下にプロセス ID + テスト名で
//! 一意にして作り、終了時に削除する（`tests/st_save.rs` と同型）。

use std::io::Write;
use std::path::{Path, PathBuf};

use fandhe_ai_onnx_interop::onnx::external_data::{
    ExternalDataError, ExternalDataOptions, LocationRejectReason, build_graph_with_external_data,
};
use fandhe_ai_onnx_interop::onnx::graph::{GraphError, RawTensor, build_graph};
use fandhe_ai_onnx_interop::onnx::proto::{
    GraphProto, ModelProto, StringStringEntryProto, TensorProto, data_location, data_type,
};

/// テスト専用の一時ディレクトリを作る（テストごとに固有のサブディレクトリ名
/// を要求し、並行実行時の衝突を避ける）。
struct TempDir(PathBuf);

impl TempDir {
    /// `name` はディレクトリ名の可読性のためだけに使う。一意性は
    /// プロセス ID + プロセス内グローバルカウンタで担保する（同名 `name`
    /// を渡すヘルパ関数〈`expect_location_reject`／`expect_number_reject`〉
    /// が並行実行される複数テストから呼ばれても、`Drop` による削除が
    /// 他テストのディレクトリを巻き込まないようにするため）。
    fn new(name: &str) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "onnx-interop-external-data-test-{}-{name}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("一時ディレクトリの作成に失敗した");
        TempDir(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn write_file(&self, rel: &str, bytes: &[u8]) -> PathBuf {
        let p = self.0.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        p
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
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

fn assert_external_err(
    result: Result<fandhe_ai_onnx_interop::onnx::graph::Graph, GraphError>,
) -> ExternalDataError {
    match result {
        Err(GraphError::ExternalData(e)) => e,
        other => panic!("ExternalData エラーを期待したが: {other:?}"),
    }
}

// --- 正常系 ---

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
    let options = ExternalDataOptions { max_total_bytes: 8 };
    let err = assert_external_err(build_graph_with_external_data(&model, dir.path(), &options));
    assert!(matches!(
        err,
        ExternalDataError::TotalSizeLimitExceeded { limit: 8, .. }
    ));
}

// --- 異常系: 重複・重なり（A5） ---

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

/// 同一ファイルを指す `location` の表記ゆれ（`f.data`／`./f.data`）が
/// overlap 検出・ハンドル再利用を欺かないことの契約保証テスト（Cursor
/// Bugbot 指摘・PR #2348 review thread `PRRT_kwDOTuUCJc6mkYTr`）。
/// `Path` の `Eq`/`Hash` はコンポーネント単位のため、この表記ゆれ自体は
/// 旧実装（`base_dir.join(location)` の生文字列連結キー）でも実は畳み
/// 込まれていた（`Path` の実装詳細に依存した偶然の回避）。より厳密な
/// 回帰は [`overlapping_regions_via_hard_link_are_rejected`]（ハードリンク
/// はパス文字列としても `Path::components()` 正規化後も異なるが実体は
/// 同一のファイルであり、`file_key_for` の dev/ino ベースキーでのみ
/// 検出できる）を参照。
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
#[cfg(unix)]
#[test]
fn overlapping_regions_via_hard_link_are_rejected() {
    let dir = TempDir::new("overlap-hardlink");
    let real = dir.write_file("f.data", &[0u8; 16]);
    let alias = dir.path().join("alias.data");
    std::fs::hard_link(&real, &alias).expect("hard_link の作成に失敗した");

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
    build_graph_with_external_data(&model, dir.path(), &ExternalDataOptions::default())
        .expect("external data 経路は成功するはず");

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
