//! `OnnxModel::from_path_with_limits`／`OnnxExternalDataLimits`（イシュー
//! #2360）の予算回帰テスト。facade 単独の入口から external data の
//! 合計バイト上限・ファイル数上限を指定でき、超過モデルが確保前に
//! `OnnxError::InvalidModel` で拒否されること、および `from_path` の
//! 既定挙動が不変であることを固定する。
//!
//! 合成モデルは `fandhe_ai_onnx_interop::onnx::proto` で組み立てる
//! （`interop_onnx_internal_parity.rs` と同じく統合テストからの内部クレート
//! 参照は `src/` 限定ガードの対象外）。
//!
//! 注意: 疎ファイルで巨大テンソルを宣言するモデルに対し、既定予算
//! （64 GiB）で `from_path` を呼んではならない（検査を通過し実確保に進む）。

#![cfg(all(unix, target_pointer_width = "64"))]

use std::io::Write;
use std::path::{Path, PathBuf};

use fandhe_ai::interop::onnx::{OnnxError, OnnxExternalDataLimits, OnnxModel};
use fandhe_ai_onnx_interop::onnx::proto::{
    GraphProto, ModelProto, StringStringEntryProto, TensorProto, data_location, data_type,
    encode_model,
};

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "facade-external-limits-{}-{name}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("一時ディレクトリの作成に失敗した");
        TempDir(dir)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn entry(k: &str, v: &str) -> StringStringEntryProto {
    StringStringEntryProto {
        key: k.to_string(),
        value: v.to_string(),
    }
}

fn ext_f32(name: &str, dims: Vec<i64>, location: &str, length: u64) -> TensorProto {
    TensorProto {
        dims,
        data_type: data_type::FLOAT,
        float_data: Vec::new(),
        int64_data: Vec::new(),
        name: name.to_string(),
        raw_data: Vec::new(),
        external_data: vec![
            entry("location", location),
            entry("offset", "0"),
            entry("length", &length.to_string()),
        ],
        data_location: data_location::EXTERNAL,
    }
}

/// `model.onnx` を書き出してそのパスを返す。
fn write_model(dir: &TempDir, initializers: Vec<TensorProto>) -> PathBuf {
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: Vec::new(),
            name: "g".to_string(),
            initializer: initializers,
            input: Vec::new(),
            output: Vec::new(),
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
        }),
        opset_import: Vec::new(),
    };
    let p = dir.path().join("model.onnx");
    std::fs::write(&p, encode_model(&model)).unwrap();
    p
}

fn write_data(dir: &TempDir, name: &str, bytes: &[u8]) {
    let mut f = std::fs::File::create(dir.path().join(name)).unwrap();
    f.write_all(bytes).unwrap();
}

fn invalid_model_message(r: Result<OnnxModel, OnnxError>) -> String {
    match r {
        Err(OnnxError::InvalidModel { message }) => message,
        Err(other) => panic!("InvalidModel を期待したが {other:?}"),
        Ok(_) => panic!("InvalidModel を期待したが成功した"),
    }
}

/// 4 GiB の疎ファイルで 4 GiB テンソルを宣言し、予算 1 GiB で確保前に拒否。
#[test]
fn from_path_with_limits_rejects_sparse_huge_tensor_before_allocation() {
    let dir = TempDir::new("sparse");
    let f = std::fs::File::create(dir.path().join("big.onnx.data")).unwrap();
    f.set_len(1 << 32).unwrap();
    let path = write_model(
        &dir,
        vec![ext_f32("w", vec![1 << 30], "big.onnx.data", 1 << 32)],
    );
    let mut limits = OnnxExternalDataLimits::default();
    limits.max_total_bytes = 1 << 30;
    let msg = invalid_model_message(OnnxModel::from_path_with_limits(&path, &limits));
    assert!(msg.contains("limit=1073741824"), "{msg}");
    assert!(msg.contains("requested=4294967296"), "{msg}");
}

#[test]
fn from_path_with_limits_rejects_too_many_external_files() {
    let dir = TempDir::new("files");
    write_data(&dir, "a.data", &1.0f32.to_le_bytes());
    write_data(&dir, "b.data", &2.0f32.to_le_bytes());
    let path = write_model(
        &dir,
        vec![
            ext_f32("a", vec![1], "a.data", 4),
            ext_f32("b", vec![1], "b.data", 4),
        ],
    );
    let mut limits = OnnxExternalDataLimits::default();
    limits.max_external_files = 1;
    let msg = invalid_model_message(OnnxModel::from_path_with_limits(&path, &limits));
    assert!(msg.contains("ファイル数"), "{msg}");
    assert!(msg.contains("limit=1"), "{msg}");
    // 上限 2 なら成功する。
    limits.max_external_files = 2;
    assert!(OnnxModel::from_path_with_limits(&path, &limits).is_ok());
}

#[test]
fn from_path_with_limits_accepts_model_at_exact_budget() {
    let dir = TempDir::new("exact");
    write_data(&dir, "w.data", &1.5f32.to_le_bytes());
    let path = write_model(&dir, vec![ext_f32("w", vec![1], "w.data", 4)]);
    let mut limits = OnnxExternalDataLimits::default();
    limits.max_total_bytes = 4;
    assert!(OnnxModel::from_path_with_limits(&path, &limits).is_ok());
    limits.max_total_bytes = 3;
    let msg = invalid_model_message(OnnxModel::from_path_with_limits(&path, &limits));
    assert!(msg.contains("limit=3"), "{msg}");
}

/// `from_path` は既定予算の `from_path_with_limits` と同じ結果になる
/// （成功可否・失敗時のエラー種別が一致。実 fixture での bit 一致は既存の
/// external data テストが担う）。
#[test]
fn from_path_matches_from_path_with_default_limits() {
    let dir = TempDir::new("default");
    write_data(&dir, "w.data", &1.5f32.to_le_bytes());
    let path = write_model(&dir, vec![ext_f32("w", vec![1], "w.data", 4)]);
    assert!(OnnxModel::from_path(&path).is_ok());
    assert!(OnnxModel::from_path_with_limits(&path, &OnnxExternalDataLimits::default()).is_ok());
    // 欠落 companion は両経路とも同じ Io エラー種別。
    std::fs::remove_file(dir.path().join("w.data")).unwrap();
    let a = OnnxModel::from_path(&path);
    let b = OnnxModel::from_path_with_limits(&path, &OnnxExternalDataLimits::default());
    match (a, b) {
        (Err(OnnxError::Io(x)), Err(OnnxError::Io(y))) => assert_eq!(x.kind(), y.kind()),
        _ => panic!("両方 Io エラーを期待"),
    }
}
