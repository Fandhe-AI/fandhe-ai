//! `onnx-interop::st_save`/`st_load` の save→load bit 一致検証の拡充
//! （イシュー #198・親イシュー #196・REQ-7）。
//!
//! `tests/st_save.rs`（#197）が既にカバーする項目（PyTorch fixture の
//! load→save→load bit 一致・ヘッダ構造検証・同一マップの決定性・
//! 非 contiguous view・空マップ・0 要素テンソル・ファイル書き出し経路）
//! とは重複させず、本ファイルは次の 3 点のみを扱う:
//!
//! 1. 特殊浮動小数点値（NaN・符号反転 NaN・±Infinity・-0.0・サブノーマル・
//!    `f32::MIN_POSITIVE`・`f32::MAX`）の bit 一致（値比較 `==` では NaN・
//!    -0.0 を検証できないため必ず `to_bits()` 比較を使う）
//! 2. save→load→save の 2 回目バイト列が 1 回目と完全一致すること
//!    （`st_save.rs` の「同一マップ 2 回 save」決定性テストに対し、
//!    **load を挟んだ**安定性を検証する。sha256 改竄検知運用の前提を
//!    固定する。`.claude/rules/security.md` A08）
//! 3. rank 1〜4 の shape・値の bit 一致
//!
//! 数値判定はすべて `to_bits()` の bit 一致とし、tolerance（許容誤差）は
//! 新設・緩和しない（`coding-rust.md`「許容誤差を単独で緩和しない」）。

use std::collections::HashMap;

use onnx_interop::st_load::load_safetensors_f32_from_bytes;
use onnx_interop::st_save::save_safetensors_f32_to_bytes;
use tensor_core::Tensor;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// 全要素の `to_bits()` が一致することを確認する（値比較 `==` は
/// NaN・-0.0 を検証できないため使わない）。
fn assert_bits_eq(key: &str, expected: &Tensor<f32>, actual: &Tensor<f32>) {
    assert_eq!(
        expected.shape(),
        actual.shape(),
        "shape mismatch for key {key}"
    );
    let expected_dense = expected.contiguous();
    let actual_dense = actual.contiguous();
    let expected_slice = expected_dense
        .as_slice()
        .expect("test fixture: contiguous() 後は as_slice() が Some のはず");
    let actual_slice = actual_dense
        .as_slice()
        .expect("test fixture: contiguous() 後は as_slice() が Some のはず");
    for (i, (a, b)) in expected_slice.iter().zip(actual_slice.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "bit mismatch for key {key} at index {i}: expected={a:?} (0x{:08x}) actual={b:?} (0x{:08x})",
            a.to_bits(),
            b.to_bits(),
        );
    }
}

// --- 1. 特殊浮動小数点値の bit 一致 ---

#[test]
fn special_float_values_round_trip_bit_exact() {
    // ペイロード付き NaN（0x7FC0_1234）・符号反転 NaN（0xFFC0_1234）は
    // 通常の NaN ビットパターンと異なる仮数部を持つ。safetensors の
    // ワイヤフォーマット（生バイト列コピー・`to_le_bytes`/`from_le_bytes`。
    // `st_save.rs`/`st_load.rs` モジュール冒頭ドキュメント参照）はペイロード
    // を保存対象として区別しないため、bit パターンごと保持されるはずである。
    let values = vec![
        f32::from_bits(0x7FC0_1234), // NaN（ペイロード付き）
        f32::from_bits(0xFFC0_1234), // 符号反転 NaN（ペイロード付き）
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0_f32,
        0.0_f32,
        f32::from_bits(1), // 最小の正のサブノーマル
        f32::MIN_POSITIVE, // 最小の正の正規化数
        f32::MAX,
        f32::MIN,
    ];
    let n = values.len();
    let original = tensor(values, &[n]);

    let mut tensors: HashMap<String, Tensor<f32>> = HashMap::new();
    tensors.insert("special".to_string(), original.clone());

    let bytes = save_safetensors_f32_to_bytes(&tensors, None).expect("書き出しに失敗した");
    let reloaded = load_safetensors_f32_from_bytes(&bytes).expect("再ロードに失敗した");

    assert_bits_eq("special", &original, &reloaded["special"]);
}

// --- 2. save→load→save の安定性（sha256 改竄検知運用の前提） ---

#[test]
fn save_load_save_produces_identical_bytes() {
    let mut tensors: HashMap<String, Tensor<f32>> = HashMap::new();
    tensors.insert(
        "a".to_string(),
        tensor(vec![1.0, -2.5, 3.25, f32::NAN, -0.0], &[5]),
    );
    tensors.insert("b".to_string(), tensor(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]));

    let bytes1 = save_safetensors_f32_to_bytes(&tensors, None).expect("1 回目の書き出しに失敗した");
    let reloaded = load_safetensors_f32_from_bytes(&bytes1).expect("再ロードに失敗した");
    let bytes2 =
        save_safetensors_f32_to_bytes(&reloaded, None).expect("2 回目の書き出しに失敗した");

    assert_eq!(
        bytes1, bytes2,
        "save→load→save の 2 回目バイト列が 1 回目と一致しない\
         （sha256 改竄検知運用の前提が崩れている疑い）"
    );
}

// --- 3. 各 rank の bit 一致 ---

#[test]
fn various_ranks_round_trip_bit_exact() {
    let mut tensors: HashMap<String, Tensor<f32>> = HashMap::new();
    tensors.insert(
        "rank1".to_string(),
        tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[7]),
    );
    tensors.insert(
        "rank2".to_string(),
        tensor((0..15).map(|i| i as f32 * 0.5 - 3.0).collect(), &[3, 5]),
    );
    tensors.insert(
        "rank3".to_string(),
        tensor((0..24).map(|i| i as f32 * 0.25).collect(), &[2, 3, 4]),
    );
    tensors.insert(
        "rank4".to_string(),
        tensor((0..120).map(|i| (i as f32).sin()).collect(), &[2, 3, 4, 5]),
    );

    let bytes = save_safetensors_f32_to_bytes(&tensors, None).expect("書き出しに失敗した");
    let reloaded = load_safetensors_f32_from_bytes(&bytes).expect("再ロードに失敗した");

    for key in ["rank1", "rank2", "rank3", "rank4"] {
        assert_bits_eq(key, &tensors[key], &reloaded[key]);
    }
}

/// rank 0（shape `[]`・スカラー）の受理可否確認。`safetensors` クレートの
/// `TensorView::new` は shape `[]` を拒否しない（データ長 1 要素 ×
/// dtype サイズのテンソルとして受理する）ことを実装時に確認したため、
/// bit 一致の対象に含める。拒否される実装へ変わった場合はこのテストが
/// 失敗して検知できる（新たに shape `[]` を排除する仕様変更・実装変更は
/// 本イシューでは行わない）。
#[test]
fn rank_zero_scalar_round_trips_bit_exact() {
    let mut tensors: HashMap<String, Tensor<f32>> = HashMap::new();
    tensors.insert("scalar".to_string(), tensor(vec![42.5], &[]));

    let bytes = save_safetensors_f32_to_bytes(&tensors, None).expect("書き出しに失敗した");
    let reloaded = load_safetensors_f32_from_bytes(&bytes).expect("再ロードに失敗した");

    assert_bits_eq("scalar", &tensors["scalar"], &reloaded["scalar"]);
}
