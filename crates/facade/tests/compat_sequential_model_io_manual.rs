//! イシュー #2188（親 #2131）「`compat::Sequential` 層構成シリアライズ
//! （`save_model`・`load_model`）」の facade 公開面は未承認のまま保留
//! した（`docs/compat-model-io-decision.md` §0）。本ファイルは、承認後
//! の `save_model`／`load_model` が被せる土台部分——`state_dict`／
//! `load_state_dict`／`fandhe_ai::interop::safetensors` の**既存公開
//! API のみ**を使った層構成パラメータの roundtrip——が、深い異種
//! スタック・transformer encoder（内部 residual）・embedding+
//! attention の各構成で bit 完全一致することを先行して固定する
//! （`interop_safetensors_roundtrip.rs` と同じ流儀。EMA
//! `compat_sequential_ema_manual.rs`〈#2307〉と同じ位置づけ）。
//!
//! 受入基準の「skip connection を含む複雑な構成」は、
//! `compat::Sequential` が任意の分岐を表現できないため（層は
//! `Box<dyn Module>` の単純な直列リスト。`docs/compat-model-io-
//! decision.md` §2「受入基準からの逸脱」参照）、内部に residual を
//! 持つ [`fandhe_ai::compat::Sequential::add_transformer_encoder`] で
//! 代替する。同じ理由により `transformer_encoder_residual_stack_
//! roundtrip_bit_identical` は「先頭 Linear」を持たない構成にした
//! （`nn::Linear::forward` は rank≥3 入力を意図的に拒否するため
//! （`crates/autodiff/src/nn/linear.rs`「rank≥3 は対象外」節）、
//! transformer encoder の rank-3 出力へ直接 Linear を繋げず、
//! 手前に `add_flatten` を挟む）。
//!
//! いずれのテストも BatchNorm の running stats を「一度も train
//! モードで forward していない初期値」の状態で比較するため、
//! `predict` を呼ぶ**前**に必ず [`Sequential::eval`] を呼ぶ
//! （BatchNorm は `RefCell` 内部可変性で running stats を保持し、
//! train モードの forward は `&self` のままそれを更新するため。
//! `crates/autodiff/src/nn/batch_norm.rs` 参照）。承認後の
//! `save_model` は train 後の running stats も別キーで保存する設計
//! （同 decision doc §4・§5）だが、本ファイルはその前段の「初期値の
//! ままの roundtrip」のみを検証する。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::safetensors::{load_safetensors_f32, save_safetensors_f32};
use fandhe_ai::{AutodiffError, Tensor};

/// テストごとに衝突しない一時ディレクトリ（プロセス ID + テスト名）を
/// 作り、`Drop` で必ず削除する（`interop_safetensors_roundtrip.rs` の
/// `temp_dir_for` と同型だが、こちらは呼び出し側の `unwrap` パニックが
/// 途中で発生しても確実に片付くよう `Drop` ガードにした）。
struct TempDirGuard {
    path: std::path::PathBuf,
}

impl TempDirGuard {
    fn new(test_name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "fandhe-ai-model-io-manual-{}-{test_name}",
            std::process::id()
        ));
        // 前回異常終了の残骸があれば消してから作り直す。
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn join(&self, name: &str) -> std::path::PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
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

/// `state_dict()` を safetensors ファイル経由で往復し、`load_state_dict`
/// した先のモデルへ全キー・全要素を bit 完全一致で反映する（seed の
/// 異なる 2 モデル間で往復させることで「たまたま同じ初期値だった」を
/// 排除する）。あわせて `eval()` 後の `predict` 出力も bit 完全一致
/// することを確認する。
fn assert_state_dict_roundtrip_bit_identical(
    dir: &TempDirGuard,
    file_name: &str,
    mut model_a: Sequential,
    mut model_b: Sequential,
    sample_input: &Tensor<f32>,
) {
    // BatchNorm 等のモード依存層を持つ構成でも、running stats を初期値
    // のまま比較するため `predict` の前に必ず `eval()` する（モジュール
    // doc 冒頭「非復元のもの」参照）。
    model_a.eval();
    model_b.eval();

    let sd_a = model_a.state_dict();
    let path = dir.join(file_name);
    save_safetensors_f32(&path, &sd_a).unwrap();
    let loaded = load_safetensors_f32(&path).unwrap();

    model_b.load_state_dict(loaded).unwrap();
    let sd_b = model_b.state_dict();

    assert_eq!(
        sd_a.len(),
        sd_b.len(),
        "state_dict のキー数が一致しない（層構成がずれている疑い）"
    );
    for (key, tensor_a) in &sd_a {
        let tensor_b = sd_b
            .get(key.as_str())
            .unwrap_or_else(|| panic!("load 後の state_dict に `{key}` が存在しない"));
        assert_tensor_bit_exact(tensor_a, tensor_b, key);
    }

    let predicted_a = model_a.predict(sample_input).unwrap();
    let predicted_b = model_b.predict(sample_input).unwrap();
    assert_tensor_bit_exact(&predicted_a, &predicted_b, "predict 出力");
}

// ---- テスト 1: 深い異種スタック ----

const DEEP_IMG_N: usize = 2;
const DEEP_IMG_C: usize = 3;
const DEEP_IMG_H: usize = 8;
const DEEP_IMG_W: usize = 8;

/// Conv2d → BatchNorm2d → ReLU → MaxPool2d → Conv2d → GELU →
/// AdaptiveAvgPool2d → Flatten → Linear → LayerNorm → Dropout →
/// RmsNorm → Softplus → LeakyReLU → Linear の 12 層構成
/// （畳み込み・正規化・pooling・活性化・全結合を混在させた深い異種
/// スタック）を `seed_base` から決定的に構築する。
fn build_deep_heterogeneous_model(seed_base: u64) -> Result<Sequential, AutodiffError> {
    Sequential::new()
        .add_conv2d(
            DEEP_IMG_C,
            4,
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 1],
            /* groups = */ 1,
            seed_base,
        )?
        .add_batch_norm2d(4, 1e-5, 0.1)?
        .add_relu()
        .add_max_pool2d([2, 2], None, [0, 0], [1, 1])?
        .add_conv2d(4, 6, [3, 3], [1, 1], [1, 1], [1, 1], 1, seed_base + 1)?
        .add_gelu()
        .add_adaptive_avg_pool2d([1, 1])?
        .add_flatten(1, 3)
        .add_linear(6, 16, seed_base + 2)?
        .add_layer_norm(16, 1e-5)?
        .add_dropout(0.5)?
        .add_rms_norm(16, 1e-5)?
        .add_softplus(1.0, 20.0)?
        .add_leaky_relu(0.01)
        .add_linear(16, 3, seed_base + 3)
}

fn deep_heterogeneous_sample_input() -> Tensor<f32> {
    let mut rng = Xorshift64Star::new(0xDEEF_0001);
    let data = rng.fill_vec(DEEP_IMG_N * DEEP_IMG_C * DEEP_IMG_H * DEEP_IMG_W);
    Tensor::new(data, &[DEEP_IMG_N, DEEP_IMG_C, DEEP_IMG_H, DEEP_IMG_W]).unwrap()
}

#[test]
fn deep_heterogeneous_stack_params_roundtrip_bit_identical() {
    let dir = TempDirGuard::new("deep-heterogeneous");
    let model_a = build_deep_heterogeneous_model(1).unwrap();
    let model_b = build_deep_heterogeneous_model(101).unwrap();
    let input = deep_heterogeneous_sample_input();
    assert_state_dict_roundtrip_bit_identical(
        &dir,
        "deep_heterogeneous.safetensors",
        model_a,
        model_b,
        &input,
    );
}

// ---- テスト 2: transformer encoder（内部 residual）の積み重ね ----

const TENC_D_MODEL: usize = 6;
const TENC_HEADS: usize = 2;
const TENC_DIM_FF: usize = 12;
const TENC_SEQ: usize = 4;
const TENC_BATCH: usize = 2;
const TENC_OUT: usize = 3;

/// `TransformerEncoder` を 3 層積み重ね（各層が self-attention の
/// residual＋FFN の residual を内部に持つ。skip connection の代替
/// ケース）、`LayerNorm` → `Flatten` → `Linear` で締める。
fn build_transformer_stack_model(seed_base: u64) -> Result<Sequential, AutodiffError> {
    Sequential::new()
        .add_transformer_encoder(TENC_D_MODEL, TENC_HEADS, TENC_DIM_FF, seed_base)?
        .add_transformer_encoder(TENC_D_MODEL, TENC_HEADS, TENC_DIM_FF, seed_base + 1)?
        .add_transformer_encoder(TENC_D_MODEL, TENC_HEADS, TENC_DIM_FF, seed_base + 2)?
        .add_layer_norm(TENC_D_MODEL, 1e-5)?
        .add_flatten(1, 2)
        .add_linear(TENC_SEQ * TENC_D_MODEL, TENC_OUT, seed_base + 3)
}

fn transformer_stack_sample_input() -> Tensor<f32> {
    let mut rng = Xorshift64Star::new(0xDEEF_0002);
    let data = rng.fill_vec(TENC_BATCH * TENC_SEQ * TENC_D_MODEL);
    Tensor::new(data, &[TENC_BATCH, TENC_SEQ, TENC_D_MODEL]).unwrap()
}

#[test]
fn transformer_encoder_residual_stack_roundtrip_bit_identical() {
    let dir = TempDirGuard::new("transformer-stack");
    let model_a = build_transformer_stack_model(2).unwrap();
    let model_b = build_transformer_stack_model(202).unwrap();
    let input = transformer_stack_sample_input();
    assert_state_dict_roundtrip_bit_identical(
        &dir,
        "transformer_stack.safetensors",
        model_a,
        model_b,
        &input,
    );
}

// ---- テスト 3: Embedding → MultiheadAttention → Linear ----

const EMB_NUM_EMBEDDINGS: usize = 10;
const EMB_DIM: usize = 4;
const EMB_HEADS: usize = 2;
const EMB_BATCH: usize = 3;
const EMB_SEQ: usize = 5;
const EMB_OUT: usize = 3;

/// `Embedding`（`padding_idx = Some(0)`）→ `MultiheadAttention` →
/// `Flatten` → `Linear`。入力は整数値を f32 化した id 列（`add_embedding`
/// doc「入力契約」参照。非整数・範囲外は fail-closed で拒否されるため
/// 整数値ちょうどを渡す）。
fn build_embedding_mha_model(seed_base: u64) -> Result<Sequential, AutodiffError> {
    Sequential::new()
        .add_embedding(EMB_NUM_EMBEDDINGS, EMB_DIM, Some(0), seed_base)?
        .add_multihead_attention(EMB_DIM, EMB_HEADS, seed_base + 1)?
        .add_flatten(1, 2)
        .add_linear(EMB_SEQ * EMB_DIM, EMB_OUT, seed_base + 2)
}

fn embedding_mha_sample_input() -> Tensor<f32> {
    let mut rng = Xorshift64Star::new(0xDEEF_0003);
    let ids: Vec<f32> = (0..EMB_BATCH * EMB_SEQ)
        .map(|_| (rng.next_u64() % EMB_NUM_EMBEDDINGS as u64) as f32)
        .collect();
    Tensor::new(ids, &[EMB_BATCH, EMB_SEQ]).unwrap()
}

#[test]
fn embedding_mha_stack_roundtrip_bit_identical() {
    let dir = TempDirGuard::new("embedding-mha");
    let model_a = build_embedding_mha_model(3).unwrap();
    let model_b = build_embedding_mha_model(303).unwrap();
    let input = embedding_mha_sample_input();
    assert_state_dict_roundtrip_bit_identical(
        &dir,
        "embedding_mha.safetensors",
        model_a,
        model_b,
        &input,
    );
}
