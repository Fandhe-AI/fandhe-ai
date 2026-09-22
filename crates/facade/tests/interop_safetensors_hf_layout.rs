//! HF（PyTorch）レイアウトの safetensors ↔ `compat::Sequential`
//! state_dict 変換（`convert.rs`。イシュー #2080・親 #2059）の契約
//! テスト。`fandhe_ai`（facade）・`std` のみを import する
//! （`interop_safetensors_roundtrip.rs`・`interop_onnx_import.rs` と
//! 同じ流儀。facade が唯一のサポートされる公開 API 面。
//! `docs/compat-api-scope.md` §0）。
//!
//! `crates/facade/examples/hf_safetensors_sequential/convert.rs` を
//! `#[path]` で直接取り込む（同ディレクトリの変換ロジック一次ソースを
//! 二重実装しない）。
//!
//! 対象契約（`convert.rs` モジュール doc・REQ-7 参照）:
//! 1. `to_pytorch_layout` → 一時ファイル保存 → `load_safetensors_f32` →
//!    `from_pytorch_layout` → `load_state_dict` の往復が推論出力の
//!    bit 完全一致を保つ
//! 2. 不足キーは `require_keys` が全件報告する（無言 skip 禁止）
//! 3. allowlist に無い余剰キーは黙って drop せず型付き `Err`
//! 4. `in_proj_weight`／`in_proj_bias` の shape 不一致は分割前に検査され
//!    panic しない
//! 5. `lm_head.weight` 等の余剰テンソルは allowlist 経由でのみ分離される
//! 6. 同一モデルから 2 回生成したチェックポイントのバイト列は完全一致
//!    （決定的出力）
//! 7. F32 以外の dtype は変換より前に型付き `Err`（`load_safetensors_f32`
//!    自身の fail-closed 検査に委譲。手書きヘッダで検証する——facade は
//!    `safetensors` クレートへ直接依存しないため `#[path]` 取り込みの
//!    範囲でバイト列を自作する）。

#[path = "../examples/hf_safetensors_sequential/convert.rs"]
mod convert;

use convert::{ConvertError, from_pytorch_layout, split_in_proj, to_pytorch_layout};
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::safetensors::{
    LoadError, load_safetensors_f32, require_keys, save_safetensors_f32,
};
use std::collections::HashMap;

const VOCAB: usize = 6;
const EMBED_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const FEED_FORWARD: usize = 16;

fn build_model(seed: u64) -> Sequential {
    Sequential::new()
        .add_embedding(VOCAB, EMBED_DIM, None, seed)
        .unwrap()
        .add_transformer_encoder(EMBED_DIM, NUM_HEADS, FEED_FORWARD, seed + 1)
        .unwrap()
}

fn probe_ids() -> Tensor<f32> {
    Tensor::new(vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).unwrap()
}

fn synthetic_pytorch_checkpoint() -> (HashMap<String, Tensor<f32>>, Sequential) {
    let model = build_model(/* seed = */ 42);
    let sd = model.state_dict();
    let mut pt = to_pytorch_layout(&sd).unwrap();
    let lm_head = Tensor::new(vec![0.1_f32; VOCAB * EMBED_DIM], &[VOCAB, EMBED_DIM]).unwrap();
    pt.insert("lm_head.weight".to_string(), lm_head);
    (pt, model)
}

fn temp_dir_for(test_name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "fandhe-ai-hf-safetensors-hf-layout-{}-{test_name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn assert_tensor_bit_exact(a: &Tensor<f32>, b: &Tensor<f32>, label: &str) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape 不一致");
    let a_bits: Vec<u32> = a
        .contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let b_bits: Vec<u32> = b
        .contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(a_bits, b_bits, "{label}: 要素 bit 不一致");
}

/// safetensors ワイヤフォーマット（8 バイト LE ヘッダ長 + JSON ヘッダ +
/// データ本体）を手書きで組み立てる。facade は `safetensors` クレートへ
/// 直接依存しないため（許容依存は `onnx-interop` 経由の間接依存のみ。
/// `.claude/rules/deps-policy.md`）、`unsupported_dtype_is_typed_error`
/// テスト専用に最小限のバイト列を自作する。
fn raw_safetensors_bytes(dtype: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let shape_json = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let header = format!(
        "{{\"w\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape_json}],\"data_offsets\":[0,{}]}}}}",
        data.len()
    );
    let header_bytes = header.into_bytes();
    let mut out = Vec::with_capacity(8 + header_bytes.len() + data.len());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(data);
    out
}

// ---- 正常系 ----

/// PyTorch レイアウトの合成チェックポイントをファイルへ保存 →
/// `load_safetensors_f32` → `from_pytorch_layout` → 別 seed の
/// `Sequential` へ `load_state_dict` → 元モデルとの推論出力が
/// bit 完全一致する（本イシューの主目的の往復契約）。
#[test]
fn hf_layout_roundtrip_restores_sequential_bit_exact() {
    let (pt, source_model) = synthetic_pytorch_checkpoint();
    let dir = temp_dir_for("roundtrip");
    let path = dir.join("model.safetensors");
    save_safetensors_f32(&path, &pt).unwrap();

    let loaded = load_safetensors_f32(&path).unwrap();
    let (restored_state, extra) = from_pytorch_layout(&loaded, &["lm_head.weight"]).unwrap();

    let mut restored_model = build_model(/* seed = */ 999);
    restored_model.load_state_dict(restored_state).unwrap();

    assert_eq!(extra.len(), 1);
    assert!(extra.contains_key("lm_head.weight"));

    let ids = probe_ids();
    let out_source = source_model.predict(&ids).unwrap();
    let out_restored = restored_model.predict(&ids).unwrap();
    assert_tensor_bit_exact(&out_source, &out_restored, "predict() 出力");

    std::fs::remove_file(&path).ok();
}

/// 必須キーを 2 件除去すると `require_keys` が両方を列挙する
/// （無言 skip 禁止の実体。`crates/onnx-interop/src/st_load.rs` の
/// 既存契約をそのまま使う）。
#[test]
fn missing_keys_are_reported_in_full() {
    let (mut pt, _model) = synthetic_pytorch_checkpoint();
    pt.remove("0.weight");
    pt.remove("1.linear1.weight");

    let mut expected_keys: Vec<&str> = pt.keys().map(String::as_str).collect();
    // 除去した 2 件も含め、本来チェックポイントが持つべきキー集合を検査
    // 対象にする（無言 skip 禁止の実効性を確認するため）。
    expected_keys.push("0.weight");
    expected_keys.push("1.linear1.weight");

    let err = require_keys(&pt, &expected_keys).unwrap_err();
    match err {
        LoadError::MissingKeys(missing) => {
            let mut missing_sorted = missing.clone();
            missing_sorted.sort();
            assert_eq!(missing_sorted, vec!["0.weight", "1.linear1.weight"]);
        }
        other => panic!("MissingKeys を期待したが {other:?} だった"),
    }
}

/// allowlist に無い余剰キー（`position_ids` 等）は黙って drop されず
/// 型付き `Err` になる（REQ-7「無言 skip 禁止」）。
#[test]
fn unexpected_key_is_rejected_not_dropped() {
    let (mut pt, _model) = synthetic_pytorch_checkpoint();
    pt.insert(
        "position_ids".to_string(),
        Tensor::new(vec![0.0_f32, 1.0, 2.0], &[3]).unwrap(),
    );

    // allowlist は `lm_head.weight` のみ。`position_ids` は allowlist にも
    // 変換規則（`{idx}.` 接頭辞）にも該当しないため拒否される。
    let err = from_pytorch_layout(&pt, &["lm_head.weight"]).unwrap_err();
    match err {
        ConvertError::UnexpectedKey(key) => assert_eq!(key, "position_ids"),
        other => panic!("UnexpectedKey を期待したが {other:?} だった"),
    }
}

/// `in_proj_weight` が `[3E, E]` でない場合、分割前の shape 検査で
/// panic せず型付き `Err` になる（`.claude/rules/coding-rust.md`
/// 「境界検査を省略しない」）。
#[test]
fn in_proj_shape_mismatch_is_typed_error() {
    let bad_weight = Tensor::new(
        vec![0.0_f32; EMBED_DIM * EMBED_DIM],
        &[EMBED_DIM, EMBED_DIM],
    )
    .unwrap(); // 本来は [3E, E] のはずが [E, E] のまま
    let bias = Tensor::new(vec![0.0_f32; 3 * EMBED_DIM], &[3 * EMBED_DIM]).unwrap();

    let err = split_in_proj(&bad_weight, &bias, EMBED_DIM).unwrap_err();
    match err {
        ConvertError::ShapeMismatch { key, expected, .. } => {
            assert_eq!(key, "self_attn.in_proj_weight");
            assert_eq!(expected, vec![3 * EMBED_DIM, EMBED_DIM]);
        }
        other => panic!("ShapeMismatch を期待したが {other:?} だった"),
    }
}

/// `{idx}.self_attn.` プレフィックスを持つ「未知の」余剰キー（例:
/// `1.self_attn.extra_buffer`）は `unexpected_key_is_rejected_not_dropped`
/// がカバーするプレフィックスなしキー（`position_ids`）とは別経路
/// （`split_index_prefix` で `Some((idx, rest))` になるため
/// `extra_allowlist` 判定を経由せず `{idx}.` グループへ入る）を通る。
/// 既知の 12 キーを消費した後、`sub` に本キーが未消費で残っていれば
/// 無言 drop されず `UnexpectedKey` で拒否される（REQ-7「無言 skip
/// 禁止」。コードレビュー指摘の回帰テスト）。
#[test]
fn unexpected_prefixed_key_within_layer_is_rejected_not_dropped() {
    let (mut pt, _model) = synthetic_pytorch_checkpoint();
    pt.insert(
        "1.self_attn.extra_buffer".to_string(),
        Tensor::new(vec![0.0_f32], &[1]).unwrap(),
    );

    let err = from_pytorch_layout(&pt, &["lm_head.weight"]).unwrap_err();
    match err {
        ConvertError::UnexpectedKey(key) => {
            assert_eq!(key, "1.self_attn.extra_buffer");
        }
        other => panic!("UnexpectedKey を期待したが {other:?} だった"),
    }
}

/// `in_proj_weight` が rank-1（例 shape `[24]`）等、`shape()[1]` の
/// index が範囲外になる不正 shape の場合でも `from_pytorch_layout` は
/// panic せず型付き `Err` を返す（`.claude/rules/security.md` A03
/// 「境界検査を省略しない」。コードレビュー指摘の回帰テスト:
/// 修正前は `embed_dim = in_proj_weight.shape()[1]` が shape 検証より
/// 先に実行され `index out of bounds` で panic していた）。
#[test]
fn in_proj_rank1_weight_is_typed_error_not_panic() {
    let (mut pt, _model) = synthetic_pytorch_checkpoint();
    pt.insert(
        "1.self_attn.in_proj_weight".to_string(),
        Tensor::new(vec![0.0_f32; 3 * EMBED_DIM], &[3 * EMBED_DIM]).unwrap(),
    );

    let err = from_pytorch_layout(&pt, &["lm_head.weight"]).unwrap_err();
    match err {
        ConvertError::ShapeMismatch { key, actual, .. } => {
            assert_eq!(key, "1.self_attn.in_proj_weight（rank）");
            assert_eq!(actual, vec![1]);
        }
        other => panic!("ShapeMismatch を期待したが {other:?} だった"),
    }
}

/// `lm_head.weight`（`Sequential` の位置 index 接頭辞を持たない余剰
/// テンソル）が `from_pytorch_layout` の 2 つ目の戻り値（`extra`）へ
/// 分離され、1 つ目の戻り値（`Sequential` 側キー集合）には混入しない。
#[test]
fn head_weight_is_separated_via_allowlist() {
    let (pt, _model) = synthetic_pytorch_checkpoint();
    let (seq_state, extra) = from_pytorch_layout(&pt, &["lm_head.weight"]).unwrap();

    assert!(!seq_state.contains_key("lm_head.weight"));
    assert!(extra.contains_key("lm_head.weight"));
    assert_eq!(extra.len(), 1);
}

/// 同一モデルから 2 回生成した合成チェックポイントのバイト列は完全一致
/// する（`save_safetensors_f32_to_bytes` のキー昇順ソートによる決定的
/// 出力契約。`crates/onnx-interop/src/st_save.rs` に委譲）。
#[test]
fn synthetic_checkpoint_bytes_are_deterministic() {
    use fandhe_ai::interop::safetensors::save_safetensors_f32_to_bytes;

    let (pt1, _model1) = synthetic_pytorch_checkpoint();
    let (pt2, _model2) = synthetic_pytorch_checkpoint();

    let bytes1 = save_safetensors_f32_to_bytes(&pt1, None).unwrap();
    let bytes2 = save_safetensors_f32_to_bytes(&pt2, None).unwrap();
    assert_eq!(bytes1, bytes2);
}

/// F32 以外の dtype（例: F16）を含む safetensors バイト列は、変換ロジック
/// より前に `load_safetensors_f32_from_bytes` 自体が型付き `Err` で拒否
/// する（`st_load.rs` の既存 fail-closed 契約。本テストは手書き
/// ヘッダで最小バイト列を構築する）。
#[test]
fn unsupported_dtype_is_typed_error() {
    use fandhe_ai::interop::safetensors::load_safetensors_f32_from_bytes;

    let bytes = raw_safetensors_bytes("F16", &[1], &[0u8, 0u8]);
    let err = load_safetensors_f32_from_bytes(&bytes).unwrap_err();
    match err {
        LoadError::UnsupportedDtype { key, dtype } => {
            assert_eq!(key, "w");
            assert_eq!(dtype, "F16");
        }
        other => panic!("UnsupportedDtype を期待したが {other:?} だった"),
    }
}
