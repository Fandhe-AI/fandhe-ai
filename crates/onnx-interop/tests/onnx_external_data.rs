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
use fandhe_ai_onnx_interop::onnx::graph::{GraphError, build_graph};
// `RawTensor` は実解決に成功した際の initializer 値検査（cfg(unix) 限定の
// テストのみ）で使う。非 unix ビルドでは未使用になるため揃えて cfg する。
#[cfg(unix)]
use fandhe_ai_onnx_interop::onnx::graph::RawTensor;
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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
#[cfg(unix)]
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
/// 同一のファイルであり、`file_key_for` の dev/ino ベースキーでのみ
/// 検出できる）を参照。
#[cfg(unix)]
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

#[cfg(unix)]
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
#[cfg(unix)]
const MANY_REGIONS_COUNT: usize = 50_000;

/// `MANY_REGIONS_COUNT` 本の BOOL テンソル（dims=[1]・1 バイト）が 1 つの
/// `.data` ファイルの `[i, i+1)` を 1 件ずつ参照するモデルを作る（テンソル
/// は位置の逆順に並べ、入力順と位置順を一致させない）。
#[cfg(unix)]
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
#[cfg(unix)]
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
#[cfg(unix)]
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

#[cfg(unix)]
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
    // 実解決の成功は unix 限定（非 unix では `resolve_and_open` が常に
    // `UnsupportedPlatformForSecureResolve` で fail-closed 拒否する。
    // `external_data.rs` モジュール doc「非 unix（Windows 等）」節）ため、
    // 対照確認部分だけを cfg で分け、非 unix ではその拒否契約を検査する
    // （Cursor Bugbot 指摘・PR #2348 対応。下の `from_bytes` 側の A6 回帰
    // 検査は external data 解決を経由しないため OS 非依存のまま実行する）。
    let ext_result =
        build_graph_with_external_data(&model, dir.path(), &ExternalDataOptions::default());
    #[cfg(unix)]
    ext_result.expect("external data 経路は成功するはず");
    #[cfg(not(unix))]
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
/// 対応。`external_data.rs` モジュール doc「非 unix（Windows 等）」節
/// 参照。Windows 対応はイシュー #2349 で追跡中）。base_dir・companion
/// ファイルはいずれも実在させない（非 unix の `resolve_and_open` は
/// `location` の文字列検証のみ行い、ファイルの open を一切試みないため
/// 実在は不要）。
#[cfg(not(unix))]
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

/// 非 unix 契約テスト（`location` の文法検証は OS 非依存で resolve より
/// 前に行われるため、非 unix でも `UnsupportedPlatformForSecureResolve`
/// ではなく具体的な `InvalidLocation` が返ることを固定する。unix 版の
/// `symlink_escaping_base_dir_via_absolute_target_is_rejected` 等と対照的
/// に、こちらはファイルシステムへ一切触れない構文検証のみの契約）。
#[cfg(not(unix))]
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
#[cfg(unix)]
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
#[cfg(unix)]
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
