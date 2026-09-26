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
//!   - `num_slots.u64_u16x4`（全 9 種・必須）: スロット数を実在する
//!     バッファキーの最大添字から推測するのではなく、独立したメタ
//!     データとして保存・照合する（P0 レビュー指摘・イシュー #2174
//!     PR #2304: 単一バッファ optimizer で末尾スロットの全バッファが
//!     欠落しても `step_count` だけ進んだ状態で load が成功して
//!     しまう問題への対応。下記「`load_state_dict` の検証順」節）
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
//! （`NAdam` 限定）も同様に「有限かつ `[0.0, 1.0]`」であることを検証
//! する（初期値 `1.0` から `mu ∈ [0, beta1)` を逐次乗じる積のため。
//! 有限性のみの検証では正常な逐次積では生じない負値・`f32::MAX` 等を
//! 受理してしまう。P0 レビュー指摘・イシュー #2174 PR #2304）。
//! `step_count` は復号後にさらに「load 直後の 1 回の `step()` 呼び出し
//! （`NAdam` はさらに内部で `step + 1` を評価する）が確実に成功する」
//! ことを検証する（`decode_u16x4_tensor` は `u64::MAX` を含む任意の値を
//! ロスレスに受理できてしまうため。[`validate_step_count_headroom`]
//! 参照）。**恒久的な panic 対策は各 optimizer の `step()` 側にある**:
//! `self.step_count += 1` の素朴な加算は `checked_add` に置き換え済み
//! （2 回目以降の呼び出しで `step_count` が `u64::MAX` 近傍に達しても
//! panic せず `AutodiffError::InvalidArgument` を返す。`NAdam` の
//! `step + 1` も `u64` 加算を経由しない `f64` 式へ変更済み）ため、本検証
//! は「load 直後の 1 回」を超える呼び出し回数の安全性には**依存しない**
//! （P0 レビュー指摘・イシュー #2174 PR #2304 の是正コミットで追加）。
//! 本検証はあくまで load 時点の fail-early（型・値域の一次防御。復元
//! 直後に `step()` できないほど step_count が近すぎる状態を早期に
//! 拒否する）としての位置づけである。バッファ値（`m`・`v` 等）
//! 自体は値域を検証しない（学習が生んだ値をそのまま bit 単位で復元
//! することを優先する）。
//!
//! # `load_state_dict` の検証順（fail-closed・状態変更前に全件検証）
//!
//! 1. `num_slots.u64_u16x4` を復号する（欠落は即 `Err`）。ここで得た
//!    スロット数は、実在するバッファキーの最大添字から推測した値では
//!    なく、[`state_dict`](OptimizerStateDict::state_dict) が独立に
//!    書き出したメタデータそのものである（単一バッファ optimizer で
//!    末尾スロットの全バッファが欠落しても検出できてしまう問題への
//!    対応。P0 レビュー指摘・イシュー #2174 PR #2304）。続けて
//!    `num_slots * バッファ本数` を checked 乗算し、オーバーフロー
//!    または実際のキー総数（`state.len()`）を上回る場合は即 `Err` と
//!    する（1 スロットにつき最低 1 本のバッファキーが実在しなければ
//!    ならないため、正当な `state_dict` ではこの不等式は成立しない。
//!    巨大な `num_slots` 1 件で `0..num_slots` の全走査・大量の文字列
//!    生成を誘発する DoS を、`expected` 集合を構築する前に遮断する）。
//! 2. キー集合を照合する（期待キー = マーカー + スカラー +
//!    `num_slots` から復元した `0..num_slots` の `state.<i>.<buf>` と、
//!    実際のキー集合の完全一致。欠落キー・余剰キーをそれぞれ昇順で
//!    全件列挙する。`nn::module::Module::load_state_dict` と同じ
//!    書式）。非正規表記（`01`・`+1` 等）のスロット添字キーは
//!    `expected` 側に現れないため、結果として余剰キーとして検出
//!    される。
//! 3. マーカーの shape・値を検証する。
//! 4. スカラー（`step_count`／`beta*_pow_t`／`mu_product`）を復号・
//!    検証する。
//! 5. スロットバッファの shape 一致（同一スロット内の全バッファ）を
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
/// スロット数（`u64`）のキー名。実在するバッファキーの最大添字から
/// 推測するのではなく、独立した必須メタデータとして保存・照合する
/// （P0 レビュー指摘・イシュー #2174 PR #2304: 末尾スロットの全
/// バッファが欠落していても検出できない問題、および巨大添字 1 件で
/// `0..num_slots` の全走査を誘発できる問題への対応。モジュール冒頭 doc
/// 「キー配置」節）。
pub(crate) const NUM_SLOTS_KEY: &str = "num_slots.u64_u16x4";
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

/// `step_count` の値域検証（load 時点の fail-early 用。P0 レビュー
/// 指摘・イシュー #2174 PR #2304: `decode_u16x4_tensor` は `u64::MAX`
/// を含む任意のロスレス符号化値を正しく受理してしまうため、極端に
/// 大きい `step_count` を無検証で受理すると復元直後の 1 回目の
/// `step()` が直感的でない失敗をする可能性がある）。**panic 防止の
/// 恒久対策ではない**: 9 optimizer すべての `step()` は `self.
/// step_count` の加算を `checked_add` で行うため、本関数を経由しない
/// 経路（`step()` を直接繰り返し呼ぶ通常の学習ループ）でも
/// `step_count` が `u64::MAX` に達すれば panic せず型付きエラーを返す
/// （是正コミットで追加）。本関数は「load 直後の 1 回の `step()` が
/// 確実に成功する」という早期の利用者向けエラーメッセージを提供する
/// ための一次検証に留める。
pub(crate) fn validate_step_count_headroom(
    kind: &str,
    step_count: u64,
    has_mu_product: bool,
) -> Result<(), AutodiffError> {
    // `NAdam`（`has_mu_product`）は `nadam.rs::step` が `mu_next` 導出で
    // `step + 1` 相当の式を評価するため、他 8 optimizer より 1 回分
    // 保守的な余裕（`u64::MAX - 2`）を要求する。もっとも `nadam.rs` の
    // 当該式は `u64` 加算ではなく `(step as f64 + 1.0)` で評価するため
    // `u64` オーバーフローはそもそも起きない（是正コミットで変更済み）。
    // この余裕は panic 防止のためではなく、`load` 直後の 1 回の
    // `step()` が安全に実行できることを保証する fail-early 側の
    // 保守的な閾値として維持する。
    let max_safe = if has_mu_product {
        u64::MAX - 2
    } else {
        u64::MAX - 1
    };
    if step_count > max_safe {
        return Err(AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict（kind=`{kind}`）: `{STEP_COUNT_KEY}` value \
             {step_count} is too large; the next `step()` call would overflow the internal \
             `step_count` counter (maximum safely loadable value is {max_safe})"
        )));
    }
    Ok(())
}

/// `mu_product`（`NAdam` 限定）の値域検証（P0 レビュー指摘・イシュー
/// #2174 PR #2304）。初期値は `1.0` で、以降は `step()` のたびに
/// `mu ∈ [0, beta1)`（`beta1 ∈ [0,1)` の正常な config）を乗じる逐次積
/// のため、正常な状態は必ず `[0.0, 1.0]` に収まる。範囲外（負値・
/// `f32::MAX` 等）を無検証で受理すると、次の `step()` の bias
/// correction 分母（`1.0 - mu_product`）で符号反転・オーバーフローを
/// 招く壊れた状態になる。
pub(crate) fn validate_mu_product_range(key: &str, value: f32) -> Result<(), AutodiffError> {
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
    // 1. `num_slots` を独立したメタデータとして先に復号する（実在する
    //    バッファキーの最大添字からは推測しない）。欠落は他のキーの
    //    整合性に関わらず即 `Err`（P0 レビュー指摘・イシュー #2174
    //    PR #2304: 単一バッファ optimizer で末尾スロットの全バッファが
    //    欠落しても検出できない問題への対応。モジュール冒頭 doc
    //    「`load_state_dict` の検証順」節）。
    let Some(num_slots_tensor) = state.get(NUM_SLOTS_KEY) else {
        return Err(AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict（kind=`{kind}`）: missing key: `{NUM_SLOTS_KEY}`"
        )));
    };
    let num_slots_u64 = decode_u16x4_tensor(NUM_SLOTS_KEY, num_slots_tensor)?;
    let num_slots: usize = usize::try_from(num_slots_u64).map_err(|_| {
        AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict（kind=`{kind}`）: `{NUM_SLOTS_KEY}` value \
             {num_slots_u64} does not fit in `usize` on this platform"
        ))
    })?;

    // 添字だけで巨大ループ・巨大メモリ確保（さらには `usize::MAX` 付近
    // での加算オーバーフロー）を誘発できないよう、`expected` 集合を
    // 構築する前に checked 演算と入力規模の上限で弾く（P0 レビュー
    // 指摘: 少数キーでも巨大な `num_slots` 1 件で `0..num_slots` の
    // 全走査・大量の文字列生成を誘発できる問題への対応）。1 スロット
    // につき `buffer_names.len()` 本以上のキーが実際に存在しなければ
    // ならないため、`num_slots * buffer_names.len()` が実際のキー総数
    // （`state.len()`）を超えることは正当な `state_dict` では構造上
    // あり得ない。
    let actual: BTreeSet<String> = state.keys().cloned().collect();
    let expected_slot_key_count = num_slots.checked_mul(buffer_names.len()).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict（kind=`{kind}`）: `{NUM_SLOTS_KEY}` value \
             {num_slots} is too large (slot key count overflow)"
        ))
    })?;
    if expected_slot_key_count > actual.len() {
        return Err(AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict（kind=`{kind}`）: `{NUM_SLOTS_KEY}` value \
             {num_slots} is inconsistent with the number of provided keys ({}); a valid \
             state_dict must contain at least {expected_slot_key_count} slot buffer keys",
            actual.len()
        )));
    }

    // 2. キー集合の完全一致検査（`Module::load_state_dict` と同じ
    //    「欠落 → 余剰」の順・昇順列挙の書式）。`num_slots`（上記で
    //    メタデータから確定済み）を用いて `0..num_slots` の
    //    `state.<i>.<buf>` を機械的に列挙するため、末尾スロットの
    //    全バッファ欠落・非正規表記の添字（`01`・`+1` 等）はいずれも
    //    ここで「欠落キー」または「余剰キー」として検出される。
    let mut expected: BTreeSet<String> = BTreeSet::new();
    expected.insert(marker_key(kind));
    expected.insert(STEP_COUNT_KEY.to_string());
    expected.insert(NUM_SLOTS_KEY.to_string());
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

    // 3. マーカー検証（種別違いの拒否）。
    // 直前のキー集合完全一致検査により必ず存在するため `if let` で
    // 安全に取り出す（`unwrap`/`expect` は使わない）。
    let Some(marker) = state.get(&marker_key(kind)) else {
        return Err(AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict（kind=`{kind}`）: internal error: marker key \
             missing after key-set validation"
        )));
    };
    validate_marker(kind, marker)?;

    // 4. スカラーの復号・検証。
    let Some(step_count_tensor) = state.get(STEP_COUNT_KEY) else {
        return Err(AutodiffError::InvalidArgument(format!(
            "OptimizerStateDict::load_state_dict（kind=`{kind}`）: internal error: \
             `{STEP_COUNT_KEY}` missing after key-set validation"
        )));
    };
    let step_count = decode_u16x4_tensor(STEP_COUNT_KEY, step_count_tensor)?;
    validate_step_count_headroom(kind, step_count, has_mu_product)?;

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
        validate_mu_product_range(MU_PRODUCT_KEY, value)?;
        Some(value)
    } else {
        None
    };

    // 5. スロットバッファの shape 一致検査・論理 row-major 順の読み出し
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

    /// P0 レビュー指摘（イシュー #2174 PR #2304）: `mu_product` は
    /// 有限性のみの検証では正常な逐次積では生じない負値・`f32::MAX`
    /// 等を受理してしまう。値域 `[0.0, 1.0]` を外れる値・非有限値を
    /// fail-closed で拒否し、正常範囲（境界含む）は受理することを
    /// 固定する。
    #[test]
    fn mu_product_range_rejects_out_of_range() {
        assert!(validate_mu_product_range("k", -0.1).is_err());
        assert!(validate_mu_product_range("k", 1.5).is_err());
        assert!(validate_mu_product_range("k", f32::MAX).is_err());
        assert!(validate_mu_product_range("k", f32::NAN).is_err());
        assert!(validate_mu_product_range("k", f32::INFINITY).is_err());
        assert!(validate_mu_product_range("k", 0.0).is_ok());
        assert!(validate_mu_product_range("k", 1.0).is_ok());
        assert!(validate_mu_product_range("k", 0.5).is_ok());
    }

    /// P0 レビュー指摘（イシュー #2174 PR #2304）: `decode_u16x4_tensor`
    /// は `u64::MAX` を含む任意のロスレス符号化値を正しく受理するため、
    /// 無検証だと極端に大きい `step_count` を load 時点でそのまま
    /// 受理してしまう（panic 防止の恒久対策は `step()` 側の
    /// `checked_add`。`validate_step_count_headroom` の doc 参照）。
    /// `has_mu_product=false`（8 optimizer）は `u64::MAX - 1` まで、
    /// `has_mu_product=true`（`NAdam`）は `u64::MAX - 2` まで load 直後の
    /// 1 回の `step()` が安全に成功することを固定する。
    #[test]
    fn step_count_headroom_rejects_values_too_close_to_u64_max() {
        assert!(validate_step_count_headroom("adamw", u64::MAX, false).is_err());
        assert!(validate_step_count_headroom("adamw", u64::MAX - 1, false).is_ok());
        assert!(validate_step_count_headroom("adamw", u64::MAX - 2, false).is_ok());
        assert!(validate_step_count_headroom("adamw", 0, false).is_ok());

        assert!(validate_step_count_headroom("nadam", u64::MAX, true).is_err());
        assert!(validate_step_count_headroom("nadam", u64::MAX - 1, true).is_err());
        assert!(validate_step_count_headroom("nadam", u64::MAX - 2, true).is_ok());
        assert!(validate_step_count_headroom("nadam", 0, true).is_ok());
    }

    /// `num_slots` 個のスロット（キーはまだ挿入しない）を宣言した基礎
    /// state を組み立てる（マーカー・`step_count`・`num_slots` メタ
    /// データのみ）。呼び出し側がスロットバッファキーを追加・欠落
    /// させて各検証パスを試験する。
    fn base_state(kind: &str, num_slots: u64) -> HashMap<String, Tensor<f32>> {
        let mut state = HashMap::new();
        state.insert(marker_key(kind), Tensor::new(vec![1.0], &[1]).unwrap());
        state.insert(STEP_COUNT_KEY.to_string(), encode_u16x4_tensor(3).unwrap());
        state.insert(
            NUM_SLOTS_KEY.to_string(),
            encode_u16x4_tensor(num_slots).unwrap(),
        );
        state
    }

    #[test]
    fn decode_state_dict_empty_slots_roundtrip() {
        let state = base_state("adamw", 0);
        let decoded = decode_state_dict("adamw", &state, &["m", "v"], false, false, false)
            .expect("空スロットは検証を通るはず");
        assert_eq!(decoded.step_count, 3);
        assert!(decoded.slots.is_empty());
    }

    /// P0 レビュー指摘（イシュー #2174 PR #2304）: `step_count ==
    /// u64::MAX` を `decode_state_dict` レベルで fail-closed に拒否
    /// することを固定する（`decode_u16x4_tensor` 自体はロスレスに
    /// 受理するため、`validate_step_count_headroom` の呼び出し漏れが
    /// 無いことの回帰検査を兼ねる）。`has_mu_product=true`（NAdam
    /// 相当）は `u64::MAX - 1` も拒否する。
    #[test]
    fn decode_state_dict_rejects_step_count_at_u64_max() {
        let mut state = base_state("adamw", 0);
        state.insert(
            STEP_COUNT_KEY.to_string(),
            encode_u16x4_tensor(u64::MAX).unwrap(),
        );
        let err = decode_state_dict("adamw", &state, &["m", "v"], false, false, false).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        // has_mu_product=true（NAdam 相当）は `u64::MAX - 1` も拒否する
        // （load 直後の 1 回の `step()` に対する保守的な追加の余裕。
        // `validate_step_count_headroom` の doc 参照）。
        let mut state_nadam = base_state("nadam", 0);
        state_nadam.insert(
            STEP_COUNT_KEY.to_string(),
            encode_u16x4_tensor(u64::MAX - 1).unwrap(),
        );
        state_nadam.insert(
            BETA2_POW_T_KEY.to_string(),
            encode_f64_tensor(0.999).unwrap(),
        );
        state_nadam.insert(
            MU_PRODUCT_KEY.to_string(),
            Tensor::new(vec![1.0], &[1]).unwrap(),
        );
        let err_nadam = decode_state_dict(
            "nadam",
            &state_nadam,
            &["exp_avg", "exp_avg_sq"],
            false,
            true,
            true,
        )
        .unwrap_err();
        assert!(matches!(err_nadam, AutodiffError::InvalidArgument(_)));
    }

    /// P0 レビュー指摘（イシュー #2174 PR #2304）: `mu_product`
    /// （NAdam 限定）が有限性のみでなく値域 `[0.0, 1.0]` でも検証
    /// されることを `decode_state_dict` レベルで固定する。
    #[test]
    fn decode_state_dict_rejects_mu_product_out_of_range() {
        let mut state = base_state("nadam", 0);
        state.insert(
            BETA2_POW_T_KEY.to_string(),
            encode_f64_tensor(0.999).unwrap(),
        );
        state.insert(
            MU_PRODUCT_KEY.to_string(),
            Tensor::new(vec![f32::MAX], &[1]).unwrap(),
        );
        let err = decode_state_dict(
            "nadam",
            &state,
            &["exp_avg", "exp_avg_sq"],
            false,
            true,
            true,
        )
        .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        let mut state_neg = base_state("nadam", 0);
        state_neg.insert(
            BETA2_POW_T_KEY.to_string(),
            encode_f64_tensor(0.999).unwrap(),
        );
        state_neg.insert(
            MU_PRODUCT_KEY.to_string(),
            Tensor::new(vec![-1.0], &[1]).unwrap(),
        );
        let err_neg = decode_state_dict(
            "nadam",
            &state_neg,
            &["exp_avg", "exp_avg_sq"],
            false,
            true,
            true,
        )
        .unwrap_err();
        assert!(matches!(err_neg, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn decode_state_dict_rejects_missing_num_slots_key() {
        let mut state = base_state("adamw", 0);
        state.remove(NUM_SLOTS_KEY);
        let err = decode_state_dict("adamw", &state, &["m", "v"], false, false, false).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn decode_state_dict_rejects_num_slots_inconsistent_with_key_count() {
        // 少数キーしか無いのに `num_slots` だけ巨大な値を宣言する攻撃
        // 入力（P0 レビュー指摘・イシュー #2174 PR #2304）。乗算自体は
        // オーバーフローしない大きさ（`checked_mul` 自体は成功する）
        // でも、`expected` 集合を構築する巨大ループへ入る前に拒否
        // されることを固定する。
        let state = base_state("adamw", 1_000_000);
        let err = decode_state_dict("adamw", &state, &["m", "v"], false, false, false).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn decode_state_dict_rejects_num_slots_overflowing_slot_key_count() {
        // `num_slots * buffer_names.len()` が `usize` 乗算で
        // オーバーフローする場合も、`usize::MAX` に極めて近い実際の
        // キー総数を用意できない以上、上記の不整合検査で先に拒否
        // される（checked 演算自体の非パニックも併せて固定する）。
        let state = base_state("adamw", usize::MAX as u64);
        let err = decode_state_dict("adamw", &state, &["m", "v"], false, false, false).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn decode_state_dict_detects_missing_and_unexpected_keys() {
        // index 0 の `v` が欠落（`num_slots = 1` なので `state.0.v` は
        // 期待キーに含まれる）。
        let mut state = base_state("adamw", 1);
        state.insert(slot_key(0, "m"), Tensor::new(vec![1.0, 2.0], &[2]).unwrap());
        let err = decode_state_dict("adamw", &state, &["m", "v"], false, false, false).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        let mut state2 = base_state("adamw", 0);
        state2.insert(
            "unexpected.key".to_string(),
            Tensor::new(vec![0.0], &[1]).unwrap(),
        );
        let err2 =
            decode_state_dict("adamw", &state2, &["m", "v"], false, false, false).unwrap_err();
        assert!(matches!(err2, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn decode_state_dict_rejects_non_canonical_slot_index() {
        // `num_slots = 1` を宣言しつつ `state.0.*` の代わりに非正規
        // 表記 `state.01.*` を挿入する。`state.0.*` は欠落キー、
        // `state.01.*` は余剰キーとしてそれぞれ検出される（`num_slots`
        // 由来の `expected` 集合には canonical な `state.0.*` しか
        // 含まれないため）。
        let mut state = base_state("adamw", 1);
        state.insert(
            "state.01.m".to_string(),
            Tensor::new(vec![1.0, 2.0], &[2]).unwrap(),
        );
        state.insert(
            "state.01.v".to_string(),
            Tensor::new(vec![1.0, 2.0], &[2]).unwrap(),
        );
        let err = decode_state_dict("adamw", &state, &["m", "v"], false, false, false).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn decode_state_dict_rejects_marker_kind_mismatch() {
        let state = base_state("adam", 0);
        let err = decode_state_dict("adamw", &state, &["m", "v"], false, false, false).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn decode_state_dict_rejects_slot_shape_mismatch() {
        let mut state = base_state("adamw", 1);
        state.insert(slot_key(0, "m"), Tensor::new(vec![1.0, 2.0], &[2]).unwrap());
        state.insert(
            slot_key(0, "v"),
            Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        );
        let err = decode_state_dict("adamw", &state, &["m", "v"], false, false, false).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }
}
