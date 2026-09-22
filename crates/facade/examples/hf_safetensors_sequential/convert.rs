//! HF（PyTorch）レイアウトの safetensors state_dict と fandhe
//! `compat::Sequential::state_dict` レイアウトとの相互変換ロジック
//! （イシュー #2080・親 #2059）。
//!
//! ## 呼び出し文脈
//!
//! `main.rs`（本 example 本体。合成チェックポイントの生成に
//! [`to_pytorch_layout`] を使い、ロード側の復元に [`from_pytorch_layout`]
//! を使う）と `crates/facade/tests/interop_safetensors_hf_layout.rs`
//! （`#[path]` で本ファイルを直接取り込む統合テスト）の 2 箇所から
//! 使われる、本ディレクトリの変換ロジック一次ソースである。
//!
//! ## 対象アーキテクチャ（スコープ限定）
//!
//! `compat::Sequential::new().add_embedding(..).add_transformer_encoder(..)`
//! の 2 層構成**限定**。他のレイヤー種別（`Linear` 単体・`Conv2d` 等）や
//! 他アーキテクチャ（GPT-2 `Conv1D` の逆規約・HF BERT の
//! `attention.self.query.*` 形式等）は対象外（`docs/
//! huggingface-safetensors-interop-guide.md` §2「前提と非目標」参照）。
//! 全 HF アーキテクチャへの一般化は本イシューのスコープ外。
//!
//! ## REQ-7 契約（`.claude/rules/deps-policy.md`・
//! `crates/onnx-interop/src/st_load.rs` モジュール doc と同型）
//!
//! 1. **暗黙アダプタなし**: レイアウトの差（転置・pack/split）は本
//!    モジュールが明示的に行う操作としてのみ存在し、`load_safetensors_f32`
//!    自体は一切変換しない（`fandhe_ai::interop::safetensors` はそのまま
//!    使う）。
//! 2. **無言 skip 禁止**: 変換規則にも `extra_allowlist` にも無い
//!    キーは [`ConvertError::UnexpectedKey`] で拒否する（
//!    [`from_pytorch_layout`]）。欠落キーは [`ConvertError::MissingKey`]
//!    で拒否する。
//! 3. **shape 検証を分割・転置より先に行う**（index panic を起こさない。
//!    `.claude/rules/security.md` A03）。

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use fandhe_ai::{ShapeError, Tensor};

/// 変換の型付きエラー（fail-closed。無言 drop・panic をしない）。
#[derive(Debug)]
pub enum ConvertError {
    /// `from_pytorch_layout` で、変換規則にも `extra_allowlist` にも
    /// 該当しないキーが見つかった（REQ-7「無言 skip 禁止」）。
    UnexpectedKey(String),
    /// 変換規則が要求するキーが入力に存在しない。
    MissingKey(String),
    /// 変換規則が要求する shape と実際の shape が食い違う（分割・転置を
    /// 行う前に検査するため、ここで拒否されれば後続の index 操作は
    /// 一切実行されない）。
    ShapeMismatch {
        key: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// `Tensor::new`／`transpose_2d`／`narrow` が返す shape エラー。
    Tensor(ShapeError),
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConvertError::UnexpectedKey(k) => {
                write!(f, "未知のキー（allowlist にも変換規則にも無い）: {k}")
            }
            ConvertError::MissingKey(k) => write!(f, "必須キーが見つかりません: {k}"),
            ConvertError::ShapeMismatch {
                key,
                expected,
                actual,
            } => write!(
                f,
                "shape 不一致（key={key}）: expected={expected:?}, actual={actual:?}"
            ),
            ConvertError::Tensor(e) => write!(f, "tensor shape エラー: {e}"),
        }
    }
}

impl std::error::Error for ConvertError {}

impl From<ShapeError> for ConvertError {
    fn from(e: ShapeError) -> Self {
        ConvertError::Tensor(e)
    }
}

/// `state` から `key` を取り出し、無ければ [`ConvertError::MissingKey`]
/// を返す（無言 skip 禁止の実体。`get` の `Option` を握り潰さない）。
fn take<'a>(
    state: &'a HashMap<String, Tensor<f32>>,
    key: &str,
) -> Result<&'a Tensor<f32>, ConvertError> {
    state
        .get(key)
        .ok_or_else(|| ConvertError::MissingKey(key.to_string()))
}

/// `tensor` の shape が `expected` と一致することを検査する（分割・
/// 転置より前に呼ぶ。`.claude/rules/security.md` A03「境界検査を
/// 省略しない」）。
fn check_shape(tensor: &Tensor<f32>, expected: &[usize], key: &str) -> Result<(), ConvertError> {
    if tensor.shape() == expected {
        Ok(())
    } else {
        Err(ConvertError::ShapeMismatch {
            key: key.to_string(),
            expected: expected.to_vec(),
            actual: tensor.shape().to_vec(),
        })
    }
}

/// `"{idx}.{rest}"` キーを `(idx, rest)` へ分解する。`idx` が数字でない
/// キー（HF チェックポイントの LM head 等、`Sequential` の位置 index
/// 接頭辞を持たない余剰テンソル）は `None` を返す（呼び出し側が
/// allowlist 判定へ回す）。
fn split_index_prefix(key: &str) -> Option<(usize, &str)> {
    let (idx_str, rest) = key.split_once('.')?;
    let idx: usize = idx_str.parse().ok()?;
    Some((idx, rest))
}

/// [`split_in_proj`] の逆（3 つの射影を 1 本の PyTorch `in_proj_weight`／
/// `in_proj_bias` へ pack する）。`Tensor` に `cat` 相当が無いため、
/// 行優先の連続データを直接連結して組み立てる（3 テンソルとも
/// `contiguous()` を経由してから読む。非連続ビューの `as_slice()` は
/// `None` になりうるため）。
fn pack_in_proj(
    q_w: &Tensor<f32>,
    k_w: &Tensor<f32>,
    v_w: &Tensor<f32>,
    q_b: &Tensor<f32>,
    k_b: &Tensor<f32>,
    v_b: &Tensor<f32>,
    embed_dim: usize,
) -> Result<(Tensor<f32>, Tensor<f32>), ConvertError> {
    let e = embed_dim;
    for (t, key) in [
        (q_w, "q_proj.weight"),
        (k_w, "k_proj.weight"),
        (v_w, "v_proj.weight"),
    ] {
        check_shape(t, &[e, e], key)?;
    }
    for (t, key) in [
        (q_b, "q_proj.bias"),
        (k_b, "k_proj.bias"),
        (v_b, "v_proj.bias"),
    ] {
        check_shape(t, &[e], key)?;
    }

    let mut weight_data = Vec::with_capacity(3 * e * e);
    for w in [q_w, k_w, v_w] {
        // fandhe `Linear.weight` は `[in, out]`。PyTorch `in_proj_weight`
        // の各ブロックは `[out, in]` のため、pack 前に転置する
        // （REQ-7「暗黙アダプタなし」: 転置はここで明示的に行う）。
        let pt_block = w.transpose_2d()?.contiguous();
        let slice = pt_block
            .as_slice()
            .expect("contiguous() 直後は as_slice() が必ず Some");
        weight_data.extend_from_slice(slice);
    }
    let in_proj_weight = Tensor::new(weight_data, &[3 * e, e])?;

    let mut bias_data = Vec::with_capacity(3 * e);
    for b in [q_b, k_b, v_b] {
        let cont = b.contiguous();
        let slice = cont
            .as_slice()
            .expect("contiguous() 直後は as_slice() が必ず Some");
        bias_data.extend_from_slice(slice);
    }
    let in_proj_bias = Tensor::new(bias_data, &[3 * e])?;

    Ok((in_proj_weight, in_proj_bias))
}

/// PyTorch `nn.MultiheadAttention` の packed `in_proj_weight [3E, E]`／
/// `in_proj_bias [3E]` を、fandhe `MultiheadAttention` が要求する
/// 個別射影 `q_proj`／`k_proj`／`v_proj`（各 `weight: [E, E]`・
/// `bias: [E]`。fandhe レイアウトの `[in, out]`）へ分割する（イシュー
/// #2080）。
///
/// 分割前に shape を検証する（`.claude/rules/coding-rust.md`
/// 「カーネル実装の境界検査」と同じ fail-closed 方針を host 側の
/// スライスにも適用し、index panic を起こさない）。
/// [`split_in_proj`] の戻り値型（`(q_w, q_b, k_w, k_b, v_w, v_b)`）。
/// clippy `type_complexity` 回避のための型定義切り出し（`.claude/rules/
/// coding-rust.md`「コード品質」: `#[allow]` の安易な追加で黙らせない）。
pub type SplitInProj = (
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
    Tensor<f32>,
);

pub fn split_in_proj(
    in_proj_weight: &Tensor<f32>,
    in_proj_bias: &Tensor<f32>,
    embed_dim: usize,
) -> Result<SplitInProj, ConvertError> {
    let e = embed_dim;
    check_shape(in_proj_weight, &[3 * e, e], "self_attn.in_proj_weight")?;
    check_shape(in_proj_bias, &[3 * e], "self_attn.in_proj_bias")?;

    let q_w = in_proj_weight.narrow(0, 0, e)?.transpose_2d()?.contiguous();
    let k_w = in_proj_weight.narrow(0, e, e)?.transpose_2d()?.contiguous();
    let v_w = in_proj_weight
        .narrow(0, 2 * e, e)?
        .transpose_2d()?
        .contiguous();

    let q_b = in_proj_bias.narrow(0, 0, e)?.contiguous();
    let k_b = in_proj_bias.narrow(0, e, e)?.contiguous();
    let v_b = in_proj_bias.narrow(0, 2 * e, e)?.contiguous();

    Ok((q_w, q_b, k_w, k_b, v_w, v_b))
}

/// 1 層分（`Sequential` の位置 index 1 個分。`rest`-keyed サブマップ）の
/// 構造を判別する。本 example が対象とする 2 種類（`Embedding`・
/// `TransformerEncoderLayer`）のみを認識し、それ以外は
/// [`ConvertError::UnexpectedKey`]（未知の層構造は allowlist にも
/// 変換規則にも無いキーの一種として扱う）で拒否する。
#[derive(Clone, Copy, PartialEq, Eq)]
enum LayerKind {
    Embedding,
    TransformerEncoder,
}

fn detect_layer_kind(sub: &HashMap<String, Tensor<f32>>) -> Option<LayerKind> {
    if sub.contains_key("self_attn.q_proj.weight") {
        Some(LayerKind::TransformerEncoder)
    } else if sub.len() == 1 && sub.contains_key("weight") {
        Some(LayerKind::Embedding)
    } else {
        None
    }
}

fn detect_layer_kind_pytorch(sub: &HashMap<String, Tensor<f32>>) -> Option<LayerKind> {
    if sub.contains_key("self_attn.in_proj_weight") {
        Some(LayerKind::TransformerEncoder)
    } else if sub.len() == 1 && sub.contains_key("weight") {
        Some(LayerKind::Embedding)
    } else {
        None
    }
}

/// fandhe `compat::Sequential::state_dict()` の出力（本 example が組んだ
/// `Embedding → TransformerEncoderLayer` 構成限定）を、PyTorch
/// `nn.TransformerEncoderLayer` 慣習のキー・レイアウトへ変換する
/// （HF チェックポイントを**合成**するための逆方向変換。[`main`] が
/// 使う）。
pub fn to_pytorch_layout(
    state: &HashMap<String, Tensor<f32>>,
) -> Result<HashMap<String, Tensor<f32>>, ConvertError> {
    let grouped = group_by_index(state)?;
    let mut out = HashMap::new();

    for (idx, sub) in grouped {
        let kind = detect_layer_kind(&sub)
            .ok_or_else(|| ConvertError::UnexpectedKey(format!("{idx}.<unrecognized layer>")))?;
        match kind {
            LayerKind::Embedding => {
                let w = take(&sub, "weight")?;
                out.insert(format!("{idx}.weight"), w.clone());
            }
            LayerKind::TransformerEncoder => {
                let q_w = take(&sub, "self_attn.q_proj.weight")?;
                let q_b = take(&sub, "self_attn.q_proj.bias")?;
                let k_w = take(&sub, "self_attn.k_proj.weight")?;
                let k_b = take(&sub, "self_attn.k_proj.bias")?;
                let v_w = take(&sub, "self_attn.v_proj.weight")?;
                let v_b = take(&sub, "self_attn.v_proj.bias")?;
                let embed_dim = q_w.shape()[0];

                let (in_proj_weight, in_proj_bias) =
                    pack_in_proj(q_w, k_w, v_w, q_b, k_b, v_b, embed_dim)?;
                out.insert(format!("{idx}.self_attn.in_proj_weight"), in_proj_weight);
                out.insert(format!("{idx}.self_attn.in_proj_bias"), in_proj_bias);

                let out_proj_w = take(&sub, "self_attn.out_proj.weight")?;
                let out_proj_b = take(&sub, "self_attn.out_proj.bias")?;
                out.insert(
                    format!("{idx}.self_attn.out_proj.weight"),
                    out_proj_w.transpose_2d()?.contiguous(),
                );
                out.insert(format!("{idx}.self_attn.out_proj.bias"), out_proj_b.clone());

                for (rest, transpose) in [
                    ("linear1.weight", true),
                    ("linear1.bias", false),
                    ("linear2.weight", true),
                    ("linear2.bias", false),
                    ("norm1.weight", false),
                    ("norm1.bias", false),
                    ("norm2.weight", false),
                    ("norm2.bias", false),
                ] {
                    let t = take(&sub, rest)?;
                    let converted = if transpose {
                        t.transpose_2d()?.contiguous()
                    } else {
                        t.clone()
                    };
                    out.insert(format!("{idx}.{rest}"), converted);
                }
            }
        }
    }
    Ok(out)
}

/// [`to_pytorch_layout`] の逆（HF/PyTorch レイアウトの safetensors
/// state_dict を fandhe `compat::Sequential::load_state_dict` へ渡せる
/// キー・レイアウトへ復元する。本イシューの主目的）。
///
/// `extra_allowlist` に列挙されたキー（`Sequential` の位置 index 接頭辞
/// を持たない、例えば LM head 等の余剰テンソル）は **明示的に** 2 つ目の
/// 戻り値マップへ分離する（REQ-7「無言 skip 禁止」: allowlist に無い
/// キーは黙って drop せず [`ConvertError::UnexpectedKey`] で拒否する）。
///
/// # 戻り値
/// `(sequential_state_dict, extra_tensors)`。前者を
/// `Sequential::load_state_dict` へそのまま渡せる。
/// [`from_pytorch_layout`] の戻り値型（`(sequential_state_dict,
/// extra_tensors)`）。clippy `type_complexity` 回避のための型定義切り出し
/// （[`SplitInProj`] と同方針）。
pub type FromPytorchLayoutResult = (HashMap<String, Tensor<f32>>, HashMap<String, Tensor<f32>>);

pub fn from_pytorch_layout(
    pt: &HashMap<String, Tensor<f32>>,
    extra_allowlist: &[&str],
) -> Result<FromPytorchLayoutResult, ConvertError> {
    let mut grouped: BTreeMap<usize, HashMap<String, Tensor<f32>>> = BTreeMap::new();
    let mut extra = HashMap::new();

    for (key, tensor) in pt {
        match split_index_prefix(key) {
            Some((idx, rest)) => {
                grouped
                    .entry(idx)
                    .or_default()
                    .insert(rest.to_string(), tensor.clone());
            }
            None => {
                if extra_allowlist.contains(&key.as_str()) {
                    extra.insert(key.clone(), tensor.clone());
                } else {
                    return Err(ConvertError::UnexpectedKey(key.clone()));
                }
            }
        }
    }

    let mut out = HashMap::new();
    for (idx, sub) in grouped {
        let kind = detect_layer_kind_pytorch(&sub)
            .ok_or_else(|| ConvertError::UnexpectedKey(format!("{idx}.<unrecognized layer>")))?;
        match kind {
            LayerKind::Embedding => {
                let w = take(&sub, "weight")?;
                out.insert(format!("{idx}.weight"), w.clone());
            }
            LayerKind::TransformerEncoder => {
                let in_proj_weight = take(&sub, "self_attn.in_proj_weight")?;
                let in_proj_bias = take(&sub, "self_attn.in_proj_bias")?;
                // shape 検証（rank 検査）を `shape()[1]` の index より先に行う
                // （`.claude/rules/security.md` A03「境界検査を省略しない」・
                // 本モジュール doc §「REQ-7 契約」3.）。rank 不足の入力
                // （例 shape `[24]`）で `shape()[1]` が index out of bounds
                // panic するのを防ぐ。embed_dim 自体の妥当性（`3 * e` 行数と
                // 一致するか等）は後続の [`split_in_proj`] 内 `check_shape` が
                // 検査する。
                if in_proj_weight.shape().len() != 2 {
                    // embed_dim 未確定のため expected は rank（次元数）のみを
                    // 示す（`[3*E, E]` の `E` は shape 検査後にしか分から
                    // ない）。
                    return Err(ConvertError::ShapeMismatch {
                        key: format!("{idx}.self_attn.in_proj_weight（rank）"),
                        expected: vec![2],
                        actual: vec![in_proj_weight.shape().len()],
                    });
                }
                let embed_dim = in_proj_weight.shape()[1];

                let (q_w, q_b, k_w, k_b, v_w, v_b) =
                    split_in_proj(in_proj_weight, in_proj_bias, embed_dim)?;
                out.insert(format!("{idx}.self_attn.q_proj.weight"), q_w);
                out.insert(format!("{idx}.self_attn.q_proj.bias"), q_b);
                out.insert(format!("{idx}.self_attn.k_proj.weight"), k_w);
                out.insert(format!("{idx}.self_attn.k_proj.bias"), k_b);
                out.insert(format!("{idx}.self_attn.v_proj.weight"), v_w);
                out.insert(format!("{idx}.self_attn.v_proj.bias"), v_b);

                let out_proj_w = take(&sub, "self_attn.out_proj.weight")?;
                let out_proj_b = take(&sub, "self_attn.out_proj.bias")?;
                out.insert(
                    format!("{idx}.self_attn.out_proj.weight"),
                    out_proj_w.transpose_2d()?.contiguous(),
                );
                out.insert(format!("{idx}.self_attn.out_proj.bias"), out_proj_b.clone());

                let mut known_keys: std::collections::HashSet<&str> = [
                    "self_attn.in_proj_weight",
                    "self_attn.in_proj_bias",
                    "self_attn.out_proj.weight",
                    "self_attn.out_proj.bias",
                ]
                .into_iter()
                .collect();

                for (rest, transpose) in [
                    ("linear1.weight", true),
                    ("linear1.bias", false),
                    ("linear2.weight", true),
                    ("linear2.bias", false),
                    ("norm1.weight", false),
                    ("norm1.bias", false),
                    ("norm2.weight", false),
                    ("norm2.bias", false),
                ] {
                    let t = take(&sub, rest)?;
                    let converted = if transpose {
                        t.transpose_2d()?.contiguous()
                    } else {
                        t.clone()
                    };
                    out.insert(format!("{idx}.{rest}"), converted);
                    known_keys.insert(rest);
                }

                // REQ-7「無言 skip 禁止」: 上記の既知キー集合を take() で
                // 消費した後、`sub` に未消費キーが残っていれば
                // `{idx}.self_attn.` 等のプレフィックスを持つ「未知の」
                // 余剰キーが無言で drop されていたことになる。ここで明示
                // 検査し [`ConvertError::UnexpectedKey`] で拒否する。
                let mut unexpected: Vec<&str> = sub
                    .keys()
                    .map(String::as_str)
                    .filter(|k| !known_keys.contains(k))
                    .collect();
                unexpected.sort_unstable();
                if let Some(rest) = unexpected.first() {
                    return Err(ConvertError::UnexpectedKey(format!("{idx}.{rest}")));
                }
            }
        }
    }
    Ok((out, extra))
}

/// [`to_pytorch_layout`] 内部専用: fandhe state_dict を `{idx}.{rest}`
/// で index ごとにグループ化する。fandhe 側の出力は本 example が組んだ
/// モデルに限定されるため未知プレフィックスは無い前提だが、パース
/// できないキー（`.` を含まない等）は [`ConvertError::UnexpectedKey`]
/// にする（fail-closed。想定外の入力を無言で無視しない）。
fn group_by_index(
    state: &HashMap<String, Tensor<f32>>,
) -> Result<BTreeMap<usize, HashMap<String, Tensor<f32>>>, ConvertError> {
    let mut grouped: BTreeMap<usize, HashMap<String, Tensor<f32>>> = BTreeMap::new();
    for (key, tensor) in state {
        let (idx, rest) =
            split_index_prefix(key).ok_or_else(|| ConvertError::UnexpectedKey(key.clone()))?;
        grouped
            .entry(idx)
            .or_default()
            .insert(rest.to_string(), tensor.clone());
    }
    Ok(grouped)
}
