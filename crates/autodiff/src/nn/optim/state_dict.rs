//! optimizer 内部状態（moment／velocity 系バッファ・`step_count`・bias
//! correction 用の `beta^t` 逐次積・NAdam の `mu_product`）の
//! 取り出し／書き戻し機構（イシュー #2174・親 #2131）。
//!
//! PyTorch `torch.optim.Optimizer.state_dict()`／`load_state_dict()` に
//! 相当する契約を、既存の safetensors 入出力（`facade::interop::
//! safetensors`。f32 の LE バイトをそのまま読み書きする。`onnx-interop`
//! クレートの `st_load`／`st_save`）でロスレスに往復できる形で提供する。
//! `AdamW`・`Adam`・`RmsProp`・`Adagrad`・`Lamb`・`Adadelta`・`Adamax`・
//! `NAdam`・`RAdam`（9 optimizer）が [`OptimizerStateDict`] を実装する
//! （各ファイル末尾の `impl OptimizerStateDict for X` 参照）。
//!
//! # facade 公開の保留（イシュー #2173 と同型の判断）
//!
//! `AdamW`／`Adam`／`RmsProp`／`Adagrad`／`Lamb` は
//! `crates/facade/src/optim.rs` で `fandhe_ai::optim` へ再エクスポート
//! 済みの型だが、これらへ inherent `state_dict`／`load_state_dict` を
//! 追加すると、その時点で facade の公開面が広がる。イシュー本文には
//! 所有者の承認コメントがなく（`docs/autodiff-optimizer-state-dict-
//! decision.md` §1）、親 #2131 は「facade 公開面の拡張は設計判断記録 →
//! 承認 → 実装の 2 段」を定めているため、本イシューでは
//! [`OptimizerStateDict`] を内部クレート限定（`fandhe_ai_autodiff`）に
//! 留め、facade（`fandhe_ai::optim`）へは再エクスポートしない
//! （`crates/facade/src/lib.rs::OptimizerStateDictHoldDoctestGuard`・
//! `crates/facade/tests/api_surface.rs` の対応する否定ガードが多層防御
//! で固定する。`nn/optim/param_group.rs` モジュール doc・
//! `docs/autodiff-param-groups-decision.md` と同型の判断）。
//!
//! # キー配置（形式バージョン 1）
//!
//! - **種別マーカー** `__optimizer__.<kind>`（shape `[1]`・値 `1.0`）。
//!   `<kind>` は `adamw`／`adam`／`rmsprop`／`adagrad`／`lamb`／
//!   `adadelta`／`adamax`／`nadam`／`radam`。キー集合の完全一致検査
//!   （下記「検証」節）と組み合わせて、別種 optimizer の state（例:
//!   `Adam` の state を `AdamW` に読み込む）を fail-closed で拒否する
//!   （`Adam`／`AdamW` はバッファ名が同じ `m`／`v` のため、マーカーが
//!   ないと黙って受理されてしまう。PyTorch は種別の異なる state の
//!   読み込みを許すが、本実装は安全側に逸脱する）。
//! - **スカラー状態**（ロスレス符号化。下記「符号化」節）:
//!   - `step_count.u64_u16x4`（全 9 種）
//!   - `beta1_pow_t.f64_u16x4`（`AdamW`・`Adam`・`Lamb`・`RAdam`・
//!     `Adamax`）
//!   - `beta2_pow_t.f64_u16x4`（`AdamW`・`Adam`・`Lamb`・`RAdam`・
//!     `NAdam`）
//!   - `mu_product`（shape `[1]` の生 f32。`NAdam` のみ）
//! - **スロットバッファ** `state.<i>.<buffer>`（`i` は 0 始まりの
//!   呼び出し順スロット添字。バッファ名は各 optimizer の `SlotState`
//!   フィールド名をそのまま使う: `AdamW`／`Adam`／`Lamb` は `m`・`v`、
//!   `RmsProp` は `square_avg`・`grad_avg`・`momentum_buffer`、
//!   `Adagrad` は `state_sum`、`Adadelta` は `square_avg`・
//!   `acc_delta`、`Adamax` は `exp_avg`・`exp_inf`、`NAdam`・`RAdam`
//!   は `exp_avg`・`exp_avg_sq`）。`states` が空（初回 `step()` 前）の
//!   ときはスロットキーを一切出さない。
//!
//! ハイパーパラメータ（config）は保存しない。呼び出し側が同じ config
//! で `new` してから load する契約とする（PyTorch の `param_groups` の
//! 一部を保存する挙動との差）。
//!
//! # 符号化（ロスレス・NaN パターンを出さない）
//!
//! `u64`（`step_count`）・`f64`（`beta*_pow_t`。`to_bits()` 経由）は
//! 下位から 16bit ずつ 4 語に切り出し、各語を整数値の `f32`
//! （`0.0..=65535.0`。`f32` で厳密に表現できる）として shape `[4]` に
//! 格納する。`f32::from_bits` によるビットキャストと異なり NaN の bit
//! パターンを一切生まないため、NaN を正規化しうる外部ツールを経由
//! しても壊れない。復号時は各要素が有限・整数・`0.0..=65535.0` である
//! こと、shape が厳密に `[4]` であることを検証し、違反はすべて
//! `AutodiffError::InvalidArgument` とする。
//!
//! `beta*_pow_t` は復号後にさらに「有限かつ `[0.0, 1.0]`」であることを
//! 検証する（`beta ∈ [0,1)` なので正常な状態は必ずこの範囲に入る。
//! 範囲外は bias correction のゼロ除算・負値を招くため）。`mu_product`
//! は有限であることを検証する。バッファ値（`m`・`v` 等）自体は値域を
//! 検証しない（学習が生んだ値をそのまま bit 単位で復元することを
//! 優先する）。
//!
//! # `load_state_dict` の検証順（fail-closed・状態変更前に全件検証）
//!
//! 1. キー集合を照合する（期待キー = マーカー + スカラー +
//!    `state.<i>.<buf>`〈`i` はスロット添字集合から復元〉と、実際の
//!    キー集合の完全一致。欠落キー・余剰キーをそれぞれ昇順で全件
//!    列挙する。`nn::module::Module::load_state_dict` と同じ書式）。
//!    スロット添字は `state.<s>.<buf>` の `<s>` が
//!    `s.parse::<usize>()` に成功し `i.to_string() == s` であるものだけ
//!    から復元する（`01`・`+1` 等の非正規表記はこの時点で「期待キー」
//!    に含まれず、結果として余剰キーとして検出される）。
//! 2. マーカーの shape・値を検証する。
//! 3. スカラーを復号・検証する。
//! 4. スロットバッファの shape 一致（同一スロット内の全バッファ）を
//!    検証し、[`crate::eval::dense_vec`] で論理 row-major 順の値を
//!    読み出す（非 contiguous な view 入力にも対応するため）。
//!
//! ここまでをすべてローカル変数（[`DecodedState`]）に閉じ込めてから、
//! 呼び出し元（各 optimizer の `load_state_dict`）が最後に自身の
//! フィールドを一括で置き換える。途中でエラーが出た場合、呼び出し元の
//! 状態は一切変わらない。

use std::collections::{BTreeSet, HashMap};

use fandhe_ai_tensor_core::Tensor;

use crate::error::AutodiffError;
use crate::eval::dense_vec;

/// [`OptimizerStateDict::state_dict`]／[`OptimizerStateDict::
/// load_state_dict`] が固定する形式バージョン（種別マーカーの値）。
pub(crate) const FORMAT_VERSION: f32 = 1.0;

/// `step_count`（`u64`）のキー名。
pub(crate) const STEP_COUNT_KEY: &str = "step_count.u64_u16x4";
/// `beta1_pow_t`（`f64`）のキー名。
pub(crate) const BETA1_POW_T_KEY: &str = "beta1_pow_t.f64_u16x4";
/// `beta2_pow_t`（`f64`）のキー名。
pub(crate) const BETA2_POW_T_KEY: &str = "beta2_pow_t.f64_u16x4";
/// `mu_product`（`f32`。NAdam 限定）のキー名。
pub(crate) const MU_PRODUCT_KEY: &str = "mu_product";

/// optimizer の内部状態を `HashMap<String, Tensor<f32>>` として取り出し
/// ／書き戻す（モジュール冒頭 doc「キー配置」節）。`fandhe_ai_autodiff`
/// 内部クレート限定の公開（facade 非公開。モジュール冒頭 doc「facade
/// 公開の保留」節参照）。
pub trait OptimizerStateDict {
    /// 現在の内部状態を [`Tensor<f32>`] のマップとして返す。
    ///
    /// `states` が空（初回 `step()` 前）のときはスロットキーを含まない
    /// マップを返す（マーカー・`step_count`・該当する場合は
    /// `beta*_pow_t`／`mu_product` のみ）。
    ///
    /// # Errors
    ///
    /// バッファのデータ長と shape の要素数積は構造上つねに整合する
    /// ため、実運用で `Err` になることは想定していない
    /// （[`fandhe_ai_tensor_core::Tensor::new`] が返す `Result` を
    /// そのまま伝播するのみ。本番経路で `unwrap`/`expect` を使わない
    /// 方針〈`.claude/rules/coding-rust.md`〉に従い、シグネチャは
    /// infallible にせず `Result` のまま公開する）。
    fn state_dict(&self) -> Result<HashMap<String, Tensor<f32>>, AutodiffError>;

    /// [`OptimizerStateDict::state_dict`] が返したマップ（または
    /// safetensors 経由で往復させたマップ）から内部状態を復元する。
    ///
    /// # Errors
    ///
    /// キー集合の不一致・スロット添字の非正規表記・shape 不一致・
    /// スカラーの符号化違反・値域違反（`beta*_pow_t`／`mu_product`）・
    /// 種別マーカーの不一致は、すべて `AutodiffError::InvalidArgument`
    /// または `AutodiffError::Shape` を返す（モジュール冒頭 doc
    /// 「`load_state_dict` の検証順」節）。検証はすべて状態変更前に
    /// 完了させるため、`Err` を返した場合 `self` は一切変更されない。
    fn load_state_dict(&mut self, state: HashMap<String, Tensor<f32>>)
    -> Result<(), AutodiffError>;
}

/// [`OptimizerStateDict::state_dict`] のマーカーキー
/// （`__optimizer__.<kind>`）を組み立てる。
pub(crate) fn marker_key(kind: &str) -> String {
    format!("__optimizer__.{kind}")
}

/// [`OptimizerStateDict::state_dict`] のスロットバッファキー
/// （`state.<i>.<buffer>`）を組み立てる。
pub(crate) fn slot_key(index: usize, buffer: &str) -> String {
    format!("state.{index}.{buffer}")
}

/// `u64` の bit 列を下位から 16bit ずつ 4 語へ切り出し、各語を整数値の
/// `f32`（`0.0..=65535.0`）として並べる（モジュール冒頭 doc「符号化」
/// 節）。`f32::from_bits` によるビットキャストと異なり NaN の bit
/// パターンを生まない。
pub(crate) fn encode_bits_u16x4(bits: u64) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = ((bits >> (16 * i)) & 0xFFFF) as u32 as f32;
    }
    out
}

/// [`encode_bits_u16x4`] の逆変換。各要素が有限・整数・
/// `0.0..=65535.0` であること、`data.len() == 4` であることを検証する
/// （違反はすべて `AutodiffError::InvalidArgument`）。
pub(crate) fn decode_bits_u16x4(key: &str, data: &[f32]) -> Result<u64, AutodiffError> {
    if data.len() != 4 {
        return Err(AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict: `{key}` must have exactly 4 elements \
             (u16x4 encoding), got {}",
            data.len()
        )));
    }
    let mut bits: u64 = 0;
    for (i, &v) in data.iter().enumerate() {
        if !(v.is_finite() && v.fract() == 0.0 && (0.0..=65535.0).contains(&v)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "OptimizerStateDict::load_state_dict: `{key}` word {i} must be a finite \
                 integer in [0.0, 65535.0], got {v}"
            )));
        }
        bits |= (v as u64) << (16 * i);
    }
    Ok(bits)
}

/// [`Tensor<f32>`] を shape `[4]` の u16×4 符号化として構築する
/// （エンコード側。`Tensor::new` の `Result` をそのまま伝播する）。
pub(crate) fn encode_u16x4_tensor(bits: u64) -> Result<Tensor<f32>, AutodiffError> {
    Tensor::new(encode_bits_u16x4(bits).to_vec(), &[4]).map_err(AutodiffError::Shape)
}

/// キー `key` に対応する [`Tensor<f32>`] を shape `[4]` の u16×4 符号化
/// として復号する（[`decode_bits_u16x4`] のテンソル版）。
pub(crate) fn decode_u16x4_tensor(key: &str, tensor: &Tensor<f32>) -> Result<u64, AutodiffError> {
    if tensor.shape() != [4] {
        return Err(AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict: `{key}` must have shape [4], got {:?}",
            tensor.shape()
        )));
    }
    decode_bits_u16x4(key, &dense_vec(tensor))
}

/// `f64` を [`encode_u16x4_tensor`] 経由で符号化する（`to_bits()` の
/// bit 列をそのまま u16×4 へ切り出す）。
pub(crate) fn encode_f64_tensor(value: f64) -> Result<Tensor<f32>, AutodiffError> {
    encode_u16x4_tensor(value.to_bits())
}

/// キー `key` に対応する `f64` を [`decode_u16x4_tensor`] 経由で復号
/// する。
pub(crate) fn decode_f64_tensor(key: &str, tensor: &Tensor<f32>) -> Result<f64, AutodiffError> {
    Ok(f64::from_bits(decode_u16x4_tensor(key, tensor)?))
}

/// `beta*_pow_t` の値域検証（モジュール冒頭 doc「符号化」節）。
/// `beta ∈ [0,1)` の逐次積は必ず `[0.0, 1.0]` に収まるため、範囲外は
/// bias correction のゼロ除算・負値を招く壊れた状態として拒否する。
pub(crate) fn validate_pow_t_range(key: &str, value: f64) -> Result<(), AutodiffError> {
    if !(value.is_finite() && (0.0..=1.0).contains(&value)) {
        return Err(AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict: `{key}` must be finite and in \
             [0.0, 1.0], got {value}"
        )));
    }
    Ok(())
}

/// 種別マーカー（`__optimizer__.<kind>`）の shape `[1]`・値
/// [`FORMAT_VERSION`] を検証する（別種 optimizer の state を fail-closed
/// で拒否する。モジュール冒頭 doc「キー配置」節）。
pub(crate) fn validate_marker(kind: &str, tensor: &Tensor<f32>) -> Result<(), AutodiffError> {
    let key = marker_key(kind);
    if tensor.shape() != [1] {
        return Err(AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict: `{key}` must have shape [1], got {:?}",
            tensor.shape()
        )));
    }
    let value = dense_vec(tensor)[0];
    if value != FORMAT_VERSION {
        return Err(AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict: `{key}` must equal {FORMAT_VERSION} \
             (format version marker for optimizer kind `{kind}`; a mismatch usually means \
             a state_dict from a different optimizer kind was passed), got {value}"
        )));
    }
    Ok(())
}

/// 1 スロット分の復元済みバッファ（`shape` と、バッファ名 → 論理
/// row-major 順の値）。[`DecodedState::slots`] の要素型（`clippy::
/// type_complexity` 回避のための型エイリアス）。
pub(crate) type DecodedSlot = (Vec<usize>, HashMap<String, Vec<f32>>);

/// [`OptimizerStateDict::load_state_dict`] の検証を通過した後の
/// 復元済み状態（呼び出し元が最後に自身のフィールドへ一括代入する。
/// モジュール冒頭 doc「`load_state_dict` の検証順」節）。
#[derive(Debug)]
pub(crate) struct DecodedState {
    pub(crate) step_count: u64,
    pub(crate) beta1_pow_t: Option<f64>,
    pub(crate) beta2_pow_t: Option<f64>,
    pub(crate) mu_product: Option<f32>,
    /// スロット添字の昇順（`0..n`）で並んだ [`DecodedSlot`]。
    pub(crate) slots: Vec<DecodedSlot>,
}

/// `state.<idx>.<buf>` 形式のキーのうち、`<idx>` が
/// `idx.to_string() == <idx>`（非正規表記〈`01`・`+1` 等〉を除く）を
/// 満たすものだけからスロット添字集合を復元する（モジュール冒頭 doc
/// 「`load_state_dict` の検証順」節「1.」）。
fn canonical_slot_indices(state: &HashMap<String, Tensor<f32>>) -> BTreeSet<usize> {
    let mut indices = BTreeSet::new();
    for key in state.keys() {
        let Some(rest) = key.strip_prefix("state.") else {
            continue;
        };
        let Some((idx_str, _buf)) = rest.split_once('.') else {
            continue;
        };
        if let Ok(idx) = idx_str.parse::<usize>()
            && idx.to_string() == idx_str
        {
            indices.insert(idx);
        }
    }
    indices
}

/// [`OptimizerStateDict::load_state_dict`] の実装本体。各 optimizer
/// ファイルの `impl OptimizerStateDict` はハイパーパラメータ・状態
/// フィールド名が異なるだけの薄い shim（`kind`・バッファ名集合・
/// `beta1`/`beta2`/`mu_product` の有無を渡すだけ）として本関数へ委譲
/// する。
///
/// 状態変更前に全件検証を完了させる契約（モジュール冒頭 doc
/// 「`load_state_dict` の検証順」節）を守るため、`self` を一切
/// 受け取らず、検証済みの値のみを [`DecodedState`] として返す。
pub(crate) fn decode_state_dict(
    kind: &str,
    state: &HashMap<String, Tensor<f32>>,
    buffer_names: &[&str],
    has_beta1: bool,
    has_beta2: bool,
    has_mu_product: bool,
) -> Result<DecodedState, AutodiffError> {
    // 1. キー集合の完全一致検査（`Module::load_state_dict` と同じ
    //    「欠落 → 余剰」の順・昇順列挙の書式）。
    let slot_indices = canonical_slot_indices(state);
    let num_slots = slot_indices.iter().next_back().map_or(0, |&max| max + 1);
    // 添字が 0..num_slots で連続していない場合（欠番）は、下記の
    // 期待キー集合が欠番スロットの全バッファを「欠落キー」として
    // 自然に検出する（`slot_indices` に欠番があっても
    // `expected` 側は 0..num_slots を総なめするため）。

    let mut expected: BTreeSet<String> = BTreeSet::new();
    expected.insert(marker_key(kind));
    expected.insert(STEP_COUNT_KEY.to_string());
    if has_beta1 {
        expected.insert(BETA1_POW_T_KEY.to_string());
    }
    if has_beta2 {
        expected.insert(BETA2_POW_T_KEY.to_string());
    }
    if has_mu_product {
        expected.insert(MU_PRODUCT_KEY.to_string());
    }
    for i in 0..num_slots {
        for buf in buffer_names {
            expected.insert(slot_key(i, buf));
        }
    }

    let actual: BTreeSet<String> = state.keys().cloned().collect();

    let missing: Vec<&String> = expected.difference(&actual).collect();
    if !missing.is_empty() {
        return Err(AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict（kind=`{kind}`）: missing keys: {missing:?}"
        )));
    }
    let unexpected: Vec<&String> = actual.difference(&expected).collect();
    if !unexpected.is_empty() {
        return Err(AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict（kind=`{kind}`）: unexpected keys: \
             {unexpected:?}"
        )));
    }

    // 2. マーカー検証（種別違いの拒否）。
    // 直前のキー集合完全一致検査により必ず存在するため `if let` で
    // 安全に取り出す（`unwrap`/`expect` は使わない）。
    let Some(marker) = state.get(&marker_key(kind)) else {
        return Err(AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict（kind=`{kind}`）: internal error: marker key \
             missing after key-set validation"
        )));
    };
    validate_marker(kind, marker)?;

    // 3. スカラーの復号・検証。
    let Some(step_count_tensor) = state.get(STEP_COUNT_KEY) else {
        return Err(AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict（kind=`{kind}`）: internal error: \
             `{STEP_COUNT_KEY}` missing after key-set validation"
        )));
    };
    let step_count = decode_u16x4_tensor(STEP_COUNT_KEY, step_count_tensor)?;

    let beta1_pow_t = if has_beta1 {
        let Some(t) = state.get(BETA1_POW_T_KEY) else {
            return Err(AutodiffError::InvalidArgument(format!(
                "OptimizerStateDict::load_state_dict（kind=`{kind}`）: internal error: \
                 `{BETA1_POW_T_KEY}` missing after key-set validation"
            )));
        };
        let value = decode_f64_tensor(BETA1_POW_T_KEY, t)?;
        validate_pow_t_range(BETA1_POW_T_KEY, value)?;
        Some(value)
    } else {
        None
    };

    let beta2_pow_t = if has_beta2 {
        let Some(t) = state.get(BETA2_POW_T_KEY) else {
            return Err(AutodiffError::InvalidArgument(format!(
                "OptimizerStateDict::load_state_dict（kind=`{kind}`）: internal error: \
                 `{BETA2_POW_T_KEY}` missing after key-set validation"
            )));
        };
        let value = decode_f64_tensor(BETA2_POW_T_KEY, t)?;
        validate_pow_t_range(BETA2_POW_T_KEY, value)?;
        Some(value)
    } else {
        None
    };

    let mu_product = if has_mu_product {
        let Some(t) = state.get(MU_PRODUCT_KEY) else {
            return Err(AutodiffError::InvalidArgument(format!(
                "OptimizerStateDict::load_state_dict（kind=`{kind}`）: internal error: \
                 `{MU_PRODUCT_KEY}` missing after key-set validation"
            )));
        };
        if t.shape() != [1] {
            return Err(AutodiffError::InvalidArgument(format!(
                "OptimizerStateDict::load_state_dict（kind=`{kind}`）: `{MU_PRODUCT_KEY}` must \
                 have shape [1], got {:?}",
                t.shape()
            )));
        }
        let value = dense_vec(t)[0];
        if !value.is_finite() {
            return Err(AutodiffError::InvalidArgument(format!(
                "OptimizerStateDict::load_state_dict（kind=`{kind}`）: `{MU_PRODUCT_KEY}` must \
                 be finite, got {value}"
            )));
        }
        Some(value)
    } else {
        None
    };

    // 4. スロットバッファの shape 一致検査・論理 row-major 順の読み出し
    //    （非 contiguous な view 入力にも対応するため `dense_vec` を
    //    使う。`crate::eval::dense_vec` doc 参照）。
    let mut slots: Vec<DecodedSlot> = Vec::with_capacity(num_slots);
    for i in 0..num_slots {
        let mut buffers: HashMap<String, Vec<f32>> = HashMap::with_capacity(buffer_names.len());
        let mut slot_shape: Option<Vec<usize>> = None;
        for buf in buffer_names {
            let key = slot_key(i, buf);
            let Some(tensor) = state.get(&key) else {
                return Err(AutodiffError::InvalidArgument(format!(
                    "OptimizerStateDict::load_state_dict（kind=`{kind}`）: internal error: \
                     `{key}` missing after key-set validation"
                )));
            };
            match &slot_shape {
                None => slot_shape = Some(tensor.shape().to_vec()),
                Some(shape) if shape.as_slice() != tensor.shape() => {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "OptimizerStateDict::load_state_dict（kind=`{kind}`）: slot {i} has \
                         mismatched buffer shapes: `{key}` has shape {:?}, expected {:?} \
                         (from an earlier buffer in the same slot)",
                        tensor.shape(),
                        shape
                    )));
                }
                Some(_) => {}
            }
            buffers.insert((*buf).to_string(), dense_vec(tensor));
        }
        // `buffer_names` が空でない限り `slot_shape` は必ず `Some`
        // （空の場合はここへ到達しない optimizer は存在しない。
        // 全 9 種が 1 個以上のバッファ名を持つ）。
        let shape = slot_shape.unwrap_or_default();
        slots.push((shape, buffers));
    }

    Ok(DecodedState {
        step_count,
        beta1_pow_t,
        beta2_pow_t,
        mu_product,
        slots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u16x4_roundtrip_u64_boundary_values() {
        for v in [0u64, 1, u32::MAX as u64 + 1, u64::MAX] {
            let encoded = encode_bits_u16x4(v);
            let decoded = decode_bits_u16x4("k", &encoded).unwrap();
            assert_eq!(decoded, v, "u64 往復不一致: {v}");
        }
    }

    #[test]
    fn u16x4_roundtrip_f64_boundary_values() {
        for v in [
            1.0f64,
            0.0f64,
            f64::MIN_POSITIVE,
            5e-324f64, // 最小非正規数
            0.999f64.powi(10_000),
        ] {
            let encoded = encode_bits_u16x4(v.to_bits());
            let decoded = f64::from_bits(decode_bits_u16x4("k", &encoded).unwrap());
            assert_eq!(decoded.to_bits(), v.to_bits(), "f64 往復不一致: {v}");
        }
    }

    #[test]
    fn decode_rejects_non_integer_word() {
        let err = decode_bits_u16x4("k", &[0.5, 0.0, 0.0, 0.0]).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn decode_rejects_negative_word() {
        let err = decode_bits_u16x4("k", &[-1.0, 0.0, 0.0, 0.0]).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn decode_rejects_out_of_range_word() {
        let err = decode_bits_u16x4("k", &[65536.0, 0.0, 0.0, 0.0]).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn decode_rejects_nan_word() {
        let err = decode_bits_u16x4("k", &[f32::NAN, 0.0, 0.0, 0.0]).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn decode_rejects_inf_word() {
        let err = decode_bits_u16x4("k", &[f32::INFINITY, 0.0, 0.0, 0.0]).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn decode_rejects_wrong_length() {
        let err = decode_bits_u16x4("k", &[0.0, 0.0, 0.0]).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn pow_t_range_rejects_out_of_range() {
        assert!(validate_pow_t_range("k", 1.5).is_err());
        assert!(validate_pow_t_range("k", -0.1).is_err());
        assert!(validate_pow_t_range("k", f64::NAN).is_err());
        assert!(validate_pow_t_range("k", 0.0).is_ok());
        assert!(validate_pow_t_range("k", 1.0).is_ok());
    }

    #[test]
    fn canonical_slot_indices_rejects_non_canonical_forms() {
        let mut state = HashMap::new();
        state.insert(
            "state.01.m".to_string(),
            Tensor::new(vec![0.0], &[1]).unwrap(),
        );
        state.insert(
            "state.+1.m".to_string(),
            Tensor::new(vec![0.0], &[1]).unwrap(),
        );
        state.insert(
            "state.2.m".to_string(),
            Tensor::new(vec![0.0], &[1]).unwrap(),
        );
        let indices = canonical_slot_indices(&state);
        assert_eq!(indices, BTreeSet::from([2]));
    }

    fn base_state(kind: &str) -> HashMap<String, Tensor<f32>> {
        let mut state = HashMap::new();
        state.insert(marker_key(kind), Tensor::new(vec![1.0], &[1]).unwrap());
        state.insert(STEP_COUNT_KEY.to_string(), encode_u16x4_tensor(3).unwrap());
        state
    }

    #[test]
    fn decode_state_dict_empty_slots_roundtrip() {
        let state = base_state("adamw");
        let decoded = decode_state_dict("adamw", &state, &["m", "v"], false, false, false)
            .expect("空スロットは検証を通るはず");
        assert_eq!(decoded.step_count, 3);
        assert!(decoded.slots.is_empty());
    }

    #[test]
    fn decode_state_dict_detects_missing_and_unexpected_keys() {
        let mut state = base_state("adamw");
        state.insert(slot_key(0, "m"), Tensor::new(vec![1.0, 2.0], &[2]).unwrap());
        // `v` が欠落。
        let err = decode_state_dict("adamw", &state, &["m", "v"], false, false, false).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        let mut state2 = base_state("adamw");
        state2.insert(
            "unexpected.key".to_string(),
            Tensor::new(vec![0.0], &[1]).unwrap(),
        );
        let err2 =
            decode_state_dict("adamw", &state2, &["m", "v"], false, false, false).unwrap_err();
        assert!(matches!(err2, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn decode_state_dict_rejects_marker_kind_mismatch() {
        let state = base_state("adam");
        let err = decode_state_dict("adamw", &state, &["m", "v"], false, false, false).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn decode_state_dict_rejects_slot_shape_mismatch() {
        let mut state = base_state("adamw");
        state.insert(slot_key(0, "m"), Tensor::new(vec![1.0, 2.0], &[2]).unwrap());
        state.insert(
            slot_key(0, "v"),
            Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        );
        let err = decode_state_dict("adamw", &state, &["m", "v"], false, false, false).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }
}
