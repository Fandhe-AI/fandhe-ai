//! ONNX `MaxPool`／`AveragePool`（opset-13 系。イシュー #2199・親 #2185）
//! オペ。ONNX import が Conv→Pool 型の CNN（Model Zoo の mnist／
//! squeezenet／resnet 等）を扱えるようにするための追加で、`ops::conv`
//! （イシュー #2076・`conv.rs`）の 1D 対応（`crates/onnx-interop/src/
//! ops/conv.rs`）と対になる。
//!
//! `X: [N, C, L]`（1D）または `[N, C, H, W]`（2D）を受け取り、同じ rank の
//! 出力を返す。`Indices` 出力（ONNX 仕様上 `MaxPool` の任意 2 番目の出力）
//! は非対応（呼び出し元 [`crate::onnx::interp`] の
//! `require_single_output` が単一出力を強制するため、複数出力を宣言した
//! ノードは fail-closed に拒否される）。`storage_order` 属性は検証のみ
//! 行い、単一出力しか返さないため実際の走査順へは影響しない。
//!
//! 本クレートは `fandhe_ai_tensor_core::Pool2dParams`（PyTorch 由来の
//! `padding <= kernel/2` 上限・`ceil_mode` 非対応）を使わない。ONNX
//! 仕様は非対称 `pads`・`ceil_mode` を許容し、`Pool2dParams` の制約が
//! 表現できないため、本モジュールは検証・出力長計算・直接ループを
//! 独自に実装する。
//!
//! 数値契約（ceil_mode=0・対称 pads・avg の dilation=1〈本モジュールが
//! 常に要求する〉の範囲では [`fandhe_ai_backend_cpu`] の
//! `max_pool2d`／`avg_pool2d`（`pooling.rs`）・
//! `fandhe_ai_autodiff::nn::{MaxPool1d,MaxPool2d,AvgPool1d,AvgPool2d}`
//! と bit 完全一致する走査順・タイ規則・NaN 規則・f64 縮約規約を踏襲する
//! （`.claude/rules/coding-rust.md` の勾配の長軸縮約 f64 縮約と同型の
//! 契約。ONNX `AveragePool` の `count_include_pad=1` かつ `ceil_mode=1`
//! ではみ出す窓が生じる場合の divisor は PyTorch／ONNX Runtime 準拠の
//! 「padded 座標でのクリップ」規則を採る。ONNX 仕様の文言はこの点で
//! 曖昧なため、本 doc をその判断の記録とする）。
//!
//! 境界検査（REQ-8。`.claude/rules/coding-rust.md`）: 入力 index は
//! [`window_input_pos`]（`oh*sh + kh*dh - ph` を `checked_mul`／
//! `checked_add`／`checked_sub` の連鎖で計算し、パディング領域相当の
//! 「表現不能」な座標は範囲外として扱う。`conv.rs` の同型パターンを
//! 踏襲する）経由でのみ参照し、`unsafe` な無検査アクセスは行わない。

use fandhe_ai_tensor_core::Tensor;

use super::error::OpError;

/// `MaxPool`／`AveragePool` の属性。ONNX Pool-13 系仕様の
/// `auto_pad`／`ceil_mode`／`dilations`／`kernel_shape`／`pads`／
/// `strides`（`MaxPool`）・`count_include_pad`（`AveragePool`）・
/// `storage_order`（`MaxPool`。検証のみ）を、`conv.rs::ConvAttrs` と
/// 同じ「decode 層に依存しないプレーンな構造体」設計で受け取る。
#[derive(Debug, Clone, Default)]
pub struct PoolAttrs {
    /// `kernel_shape`。ONNX 仕様上必須（`conv.rs::ConvAttrs::kernel_shape`
    /// と異なり省略時の重み shape 由来の推論元が無いため）。
    pub kernel_shape: Vec<i64>,
    /// `strides`。空なら全軸 1。
    pub strides: Vec<i64>,
    /// `pads`（ONNX 軸順 `[x1_begin, ..., xN_begin, x1_end, ..., xN_end]`）。
    /// 空なら全軸 0。`conv.rs::ConvAttrs::pads` と異なり非対称パディング
    /// （`xi_begin != xi_end`）も受理する（ONNX 仕様上合法。§3.1）。
    pub pads: Vec<i64>,
    /// `dilations`。空なら全軸 1。`AveragePool` は全要素 1 のみ受理する
    /// （PyTorch に dilated average pooling が無く、divisor の意味論を
    /// 曖昧にしないため。`MaxPool` は任意の dilation を受理する）。
    pub dilations: Vec<i64>,
    /// `ceil_mode`。`{0, 1}` のみ受理する。
    pub ceil_mode: i64,
    /// `count_include_pad`（`AveragePool` のみ）。`{0, 1}` のみ受理する。
    /// ONNX 仕様の既定は `0`（PyTorch の既定 `true` とは異なる。
    /// `interp::compute_average_pool` が呼び出し時に既定値を適用する）。
    pub count_include_pad: i64,
    /// `storage_order`（`MaxPool` のみ）。`{0, 1}` のみ受理する。本クレートは
    /// 単一出力（`values`）のみを返すため計算結果には影響しないが、値の
    /// 妥当性は検証する（未知の列挙値を無言で無視しない。OWASP A03）。
    pub storage_order: i64,
    /// `auto_pad`。[`super::conv::ConvAttrs::auto_pad`] と同じ受理条件
    /// （空文字列または `"NOTSET"` のみ）。
    pub auto_pad: String,
}

/// 窓添字 `k` から入力座標を符号安全に逆算する（`out_idx*stride +
/// k*dilation - pad_begin`。`fandhe_ai_backend_cpu::pooling::
/// window_input_pos`・`conv.rs` の座標計算と同型。`pad_begin` を超える
/// 減算はアンダーフローとして `None`（パディング領域）を返す。`strides`／
/// `dilations`／`pads` は非信頼な ONNX 属性から `usize::MAX` 近傍まで
/// 受理されうるため、`checked_mul`／`checked_add`／`checked_sub` の連鎖
/// のみを用い `as` キャストは一切行わない。OWASP A03。
/// `.claude/rules/security.md`）。
fn window_input_pos(
    out_idx: usize,
    stride: usize,
    k: usize,
    dilation: usize,
    pad_begin: usize,
) -> Option<usize> {
    let base = out_idx.checked_mul(stride)?;
    let offset = k.checked_mul(dilation)?;
    let unpadded = base.checked_add(offset)?;
    unpadded.checked_sub(pad_begin)
}

/// [`pool_out_axis_len`] の軸ごとのパラメータ（`kernel`／`stride`／
/// `dilation`／`pad_begin`／`pad_end`）をまとめた小構造体。
/// clippy::too_many_arguments 回避のための引数集約であり、呼び出し元
/// （[`max_pool`]／[`average_pool`]）の H／W 軸それぞれで組み立てる。
struct PoolAxisParams {
    k: usize,
    s: usize,
    d: usize,
    pb: usize,
    pe: usize,
}

/// 1 軸分の出力長を ONNX Pool 系仕様の式で計算する。
///
/// `eff = dilation·(kernel-1)+1`（実効カーネル幅）・`padded = in_len + pb
/// + pe`。`floor` 時は `(padded-eff)/stride + 1`、`ceil` 時は
/// `ceil((padded-eff)/stride) + 1` を求めたうえで、`(out-1)·stride >=
///   in_len + pb` なら `out -= 1`（最後の窓が入力内または左パディング内で
/// 始まらない場合は除外する。PyTorch・新しい ONNX 仕様の規則。§3.1）。
/// すべて `checked_*` 演算で行い、`in_len == 0`・`padded < eff`・
/// オーバーフロー・出力長 0 はいずれも [`OpError::InvalidPoolAttribute`]
///   で拒否する。
///
/// 引数は [`PoolAxisParams`]（軸ごとの `kernel`／`stride`／`dilation`／
/// `pad_begin`／`pad_end` をまとめた小構造体。clippy::too_many_arguments
/// 回避のため `conv.rs` と同様の設計を踏襲する）で受け取る。
fn pool_out_axis_len(
    op: &'static str,
    in_len: usize,
    axis: PoolAxisParams,
    ceil_mode: bool,
) -> Result<usize, OpError> {
    let PoolAxisParams { k, s, d, pb, pe } = axis;
    let overflow = |what: &str| OpError::InvalidPoolAttribute {
        reason: format!("{op}: 出力長計算が {what} でオーバーフローした"),
    };
    if in_len == 0 {
        return Err(OpError::InvalidPoolAttribute {
            reason: format!("{op}: 入力の空間長は 1 以上でなければならない（実際 0）"),
        });
    }
    let eff = k
        .checked_sub(1)
        .and_then(|km1| km1.checked_mul(d))
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| overflow("実効カーネル幅"))?;
    let padded = in_len
        .checked_add(pb)
        .and_then(|v| v.checked_add(pe))
        .ok_or_else(|| overflow("padded 入力長"))?;
    if padded < eff {
        return Err(OpError::InvalidPoolAttribute {
            reason: format!("{op}: 実効カーネル幅 {eff} が padded 入力長 {padded} を超える"),
        });
    }
    let numerator = padded - eff;
    let mut out = if ceil_mode {
        let q = numerator / s;
        let r = numerator % s;
        let ceil_q = if r == 0 {
            q
        } else {
            q.checked_add(1).ok_or_else(|| overflow("ceil 商"))?
        };
        ceil_q.checked_add(1).ok_or_else(|| overflow("出力長"))?
    } else {
        (numerator / s)
            .checked_add(1)
            .ok_or_else(|| overflow("出力長"))?
    };
    if ceil_mode {
        let bound = in_len.checked_add(pb).ok_or_else(|| overflow("境界"))?;
        if let Some(lhs) = out.checked_sub(1).and_then(|v| v.checked_mul(s))
            && lhs >= bound
        {
            out -= 1;
        }
    }
    if out == 0 {
        return Err(OpError::InvalidPoolAttribute {
            reason: format!("{op}: 属性の組み合わせにより出力長が 0 になる"),
        });
    }
    Ok(out)
}

/// 1 出力位置 `out_idx` について、有効タップ（パディング領域外の入力
/// 位置）を持つ窓添字 `ki` の連続範囲 `(ki_min, ki_max)`（両端含む）を
/// 算術的に求める。有効タップが 1 つも無ければ `None`。
///
/// `window_input_pos` は `ih = out_idx*s + ki*d - pb` を計算するが、
/// `0 <= ih < in_len` を満たす `ki` の集合は `d > 0` である限り
/// （`ki*d` が非減少なので）常に連続範囲になる。この事実を使い、
/// **`kernel_shape`（`k`）に依存しない `O(1)` の範囲計算**へ数式変形する
/// （`checked_*` 演算の連鎖のみで `as` キャストは行わない。REQ-8・
/// OWASP A03。`window_input_pos` と同型の安全側フォールバック方針）。
///
/// これにより 1 出力位置あたりの反復回数は `kernel_shape`／`pads` の値に
/// 関わらず `min(kernel_shape, in_len/dilation + 1)` 相当に収まり、外部
/// 属性由来の巨大な `kernel_shape`・`pads`（例: `kernel_shape=
/// 1_000_000_000`・`pad_begin=999_999_999`）を与えても [`axis_windows_nonempty`]
/// 自体や呼び出し元の直接ループ（[`max_pool`]／[`average_pool`]）が
/// `kernel_shape` に比例した反復を行わない（イシュー #2199 codex-review
/// 指摘: 事前検証だけで約 10 億回反復する DoS 相当の停止を修正）。
///
/// 返す範囲は `ki_min <= ki_max` の昇順であり、呼び出し元がこの範囲を
/// そのまま `ki_min..=ki_max` として昇順走査すれば、従来の `0..k`
/// 走査と同じ「`kh`／`kw` 外側・内側の row-major」順序・タイ規則・NaN
/// 規則を保つ（モジュール doc の bit 完全一致契約は不変）。
fn valid_tap_range(
    out_idx: usize,
    stride: usize,
    kernel: usize,
    dilation: usize,
    pad_begin: usize,
    in_len: usize,
) -> Option<(usize, usize)> {
    let base = out_idx.checked_mul(stride)?;
    // 上限: base + ki*dilation - pad_begin < in_len
    //   ⇔ ki*dilation < pad_begin + in_len - base
    // base が `pad_begin + in_len` を超える場合は全タップが入力より
    // 右側に外れるため有効タップなし（`checked_sub` の `None` で表現）。
    let hi_exclusive = pad_begin.checked_add(in_len)?.checked_sub(base)?;
    if hi_exclusive == 0 {
        return None;
    }
    let ki_max_by_hi = (hi_exclusive - 1) / dilation;
    // 下限: base + ki*dilation >= pad_begin ⇔ ki*dilation >= pad_begin - base
    // base >= pad_begin なら ki=0 から既に条件を満たす。
    let ki_min = match pad_begin.checked_sub(base) {
        None | Some(0) => 0,
        Some(lo) => lo.div_ceil(dilation),
    };
    if ki_min > ki_max_by_hi || ki_min >= kernel {
        return None;
    }
    let ki_max = ki_max_by_hi.min(kernel - 1);
    Some((ki_min, ki_max))
}

/// 軸 `out_len` の全出力位置が少なくとも 1 つの有効タップ（パディング
/// 領域外の入力位置）を持つことを検証する（空窓の `max` を未定義に
/// せず、`count_include_pad=0` の `AveragePool` の 0 除算を未然に防ぐ。
/// §3.1）。`count_include_pad=1` の `AveragePool` はこの検査の対象外
/// （呼び出し元 [`average_pool`] 参照。空窓でも divisor は padded 座標
/// 基準で計算され 0 除算にならない。Bugbot 指摘 #2199 対応: 全タップが
/// padding の窓を ONNX 上正当な `0` 平均として受理する）。
///
/// [`valid_tap_range`] の `O(1)` 判定を使うため、本関数 1 回あたりの
/// 反復回数は `kernel_shape` の大きさに影響されず `out_len` のみに
/// 依存する。ただし `out_len` 自体は `pads`／`kernel_shape` 属性から
/// 導出され、`Conv` の `kernel_shape`（重みテンソルの実データサイズで
/// 自然に上限される。`conv.rs` 参照）と異なり `MaxPool`／`AveragePool`
/// には対応する重みテンソルが無く攻撃者が任意の大きさを指定できるため、
/// 呼び出し元（[`max_pool`]／[`average_pool`]）は本関数を呼ぶ前に
/// [`ensure_spatial_out_bound`] で `out_len` の積を上限検査する（イシュー
/// #2199 codex-review 指摘: 本関数を `kernel_shape` 非依存の `O(1)`
/// 判定にしても `out_len` 自体が `pads` 経由で約 10 億に達し得るため
/// 事前検証・後続の直接計算ループの双方が長時間停止する。`out_len` の
/// 積を検査前に上限で拒否することで両方を同時に閉じる）。
fn axis_windows_nonempty(
    out_len: usize,
    k: usize,
    s: usize,
    d: usize,
    pb: usize,
    in_len: usize,
) -> bool {
    (0..out_len).all(|out_idx| valid_tap_range(out_idx, s, k, d, pb, in_len).is_some())
}

/// `MaxPool`／`AveragePool` の空間出力要素数（`h_out * w_out`）の上限
/// （DoS 対策の安全弁。イシュー #2199 codex-review 指摘）。
///
/// `Conv` の `kernel_shape` は重みテンソルの実データサイズで自然に上限
/// されるが（`conv.rs:502` 「上限は課さない」coment 参照）、`MaxPool`／
/// `AveragePool` の `kernel_shape`／`pads` はテンソルを伴わない整数属性
/// のため攻撃者が任意の大きさを与えられる（例:
/// `kernel_shape=[1_000_000_000]`・`pads=[999_999_999, 999_999_999]`・
/// `strides=[1]`・`in_len=1` → `out_len` ≈ 1_000_000_000）。この場合
/// [`axis_windows_nonempty`] の `O(out_len)` 走査だけでなく、後続の
/// 直接計算ループ・出力バッファ確保（`vec![0f32; out_len]`）も同じ
/// `out_len` に比例するため、`axis_windows_nonempty` 単体を `O(1)` に
/// しても DoS は閉じない。よって出力規模そのものを計算の入口で
/// 制限する（レビュー指摘の 2 案のうち「出力規模を計算前に制限する」
/// 案を採用。`2^26`〈約 6700 万〉は実在の CNN 空間出力（例:
/// 8192×8192 特徴マップ相当）を十分に超える一方、`O(1)` 判定を
/// この上限回数繰り返しても数十〜数百 ms 程度に収まる）。
const MAX_POOL_SPATIAL_OUT_ELEMENTS: usize = 1 << 26;

/// [`MAX_POOL_SPATIAL_OUT_ELEMENTS`] を超える空間出力（`h_out * w_out`）
/// を要求する属性の組み合わせを [`OpError::InvalidPoolAttribute`] で
/// 拒否する。[`max_pool`]／[`average_pool`] は [`pool_out_axis_len`] で
/// `h_out`／`w_out` を求めた直後・[`axis_windows_nonempty`] や
/// 直接計算ループへ進む前に必ず本関数を呼ぶ（呼び出し順序は両関数の
/// doc コメント「検証順序」を参照）。
fn ensure_spatial_out_bound(op: &'static str, h_out: usize, w_out: usize) -> Result<(), OpError> {
    match h_out.checked_mul(w_out) {
        Some(total) if total <= MAX_POOL_SPATIAL_OUT_ELEMENTS => Ok(()),
        _ => Err(OpError::InvalidPoolAttribute {
            reason: format!(
                "{op}: 空間出力要素数（{h_out} * {w_out}）が上限 \
                 {MAX_POOL_SPATIAL_OUT_ELEMENTS} を超える（`kernel_shape`／\
                 `pads` に起因する巨大な出力サイズは DoS 対策として拒否する）"
            ),
        }),
    }
}

/// `[usize; 2]` 属性（`kernel_shape`／`strides`／`dilations`）を軸数
/// `spatial_rank` 分読む。空なら `default` を全軸に適用し、長さ不一致・
/// 非負性違反・0 要素（`kernel_shape` は 0 を許さない。`strides`／
/// `dilations` も ONNX 仕様上 1 以上）は
/// [`OpError::InvalidPoolAttribute`] で拒否する。
fn parse_axis_list(
    op: &'static str,
    name: &str,
    values: &[i64],
    spatial_rank: usize,
    default: usize,
) -> Result<Vec<usize>, OpError> {
    if values.is_empty() {
        return Ok(vec![default; spatial_rank]);
    }
    if values.len() != spatial_rank {
        return Err(OpError::InvalidPoolAttribute {
            reason: format!(
                "{op}: `{name}` の長さは {spatial_rank} でなければならない（実際 {}）",
                values.len()
            ),
        });
    }
    values
        .iter()
        .map(|&v| {
            let u = usize::try_from(v).map_err(|_| OpError::InvalidPoolAttribute {
                reason: format!("{op}: `{name}` の要素は非負でなければならない（実際 {v}）"),
            })?;
            if u == 0 {
                return Err(OpError::InvalidPoolAttribute {
                    reason: format!("{op}: `{name}` の要素は 1 以上でなければならない（実際 0）"),
                });
            }
            Ok(u)
        })
        .collect()
}

/// `pads` 属性（ONNX 軸順 `[x1_begin, ..., xN_begin, x1_end, ..., xN_end]`）
/// を軸数 `spatial_rank` 分読む。空なら全軸 0。長さは `2 * spatial_rank`
/// でなければならない。`conv.rs::parse_pads` と異なり非対称（`begin !=
/// end`）を許容する（§3.1）。`(begins, ends)` を返す。
fn parse_pool_pads(
    op: &'static str,
    values: &[i64],
    spatial_rank: usize,
) -> Result<(Vec<usize>, Vec<usize>), OpError> {
    if values.is_empty() {
        return Ok((vec![0; spatial_rank], vec![0; spatial_rank]));
    }
    let expected_len = spatial_rank * 2;
    if values.len() != expected_len {
        return Err(OpError::InvalidPoolAttribute {
            reason: format!(
                "{op}: `pads` の長さは {expected_len} でなければならない（実際 {}）",
                values.len()
            ),
        });
    }
    let mut parsed = Vec::with_capacity(expected_len);
    for &v in values {
        let u = usize::try_from(v).map_err(|_| OpError::InvalidPoolAttribute {
            reason: format!("{op}: `pads` の要素は非負でなければならない（実際 {v}）"),
        })?;
        parsed.push(u);
    }
    let begins = parsed[..spatial_rank].to_vec();
    let ends = parsed[spatial_rank..].to_vec();
    Ok((begins, ends))
}

/// `{0, 1}` のみを許す整数フラグ（`ceil_mode`／`count_include_pad`／
/// `storage_order`）を検証する。
fn parse_bool_flag(op: &'static str, name: &str, value: i64) -> Result<bool, OpError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(OpError::InvalidPoolAttribute {
            reason: format!("{op}: `{name}` は 0 または 1 でなければならない（実際 {other}）"),
        }),
    }
}

/// 1D（`[N,C,L]`）または 2D（`[N,C,H,W]`）の入力を受け取り、常に 2D
/// 相当の `(h, w)` 軸パラメータへ正規化する。1D の場合は `H` 軸を
/// `k=1・s=1・pb=0・pe=0・d=1`（恒等写像）に固定する
/// （`conv.rs` モジュール doc の 1D 持ち上げ方針・`nn::MaxPool1d::
/// forward_host` の `[N,C,1,L]` reshape 併合と同型）。
///
/// 戻り値: `(x4, kh, kw, sh, sw, dh, dw, ph_b, ph_e, pw_b, pw_e,
/// is_1d)`。`x4` は rank 4 へ持ち上げた（2D の場合はそのまま）
/// contiguous テンソル。
#[allow(clippy::type_complexity)]
fn validate_and_lift(
    op: &'static str,
    x: &Tensor<f32>,
    attrs: &PoolAttrs,
    require_unit_dilation: bool,
) -> Result<
    (
        Tensor<f32>,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        bool,
    ),
    OpError,
> {
    if !attrs.auto_pad.is_empty() && attrs.auto_pad != "NOTSET" {
        return Err(OpError::InvalidPoolAttribute {
            reason: format!(
                "{op}: `auto_pad` は \"NOTSET\"（または省略）のみ対応する（実際 \"{}\"）",
                attrs.auto_pad
            ),
        });
    }
    let rank = x.rank();
    let is_1d = match rank {
        4 => false,
        3 => true,
        actual => {
            return Err(OpError::RankMismatch {
                op,
                expected: 4,
                actual,
            });
        }
    };
    let spatial_rank = if is_1d { 1 } else { 2 };

    if attrs.kernel_shape.is_empty() {
        return Err(OpError::InvalidPoolAttribute {
            reason: format!("{op}: `kernel_shape` は必須である"),
        });
    }
    let kernel = parse_axis_list(op, "kernel_shape", &attrs.kernel_shape, spatial_rank, 1)?;
    let strides = parse_axis_list(op, "strides", &attrs.strides, spatial_rank, 1)?;
    let dilations = parse_axis_list(op, "dilations", &attrs.dilations, spatial_rank, 1)?;
    if require_unit_dilation && dilations.iter().any(|&d| d != 1) {
        return Err(OpError::InvalidPoolAttribute {
            reason: format!(
                "{op}: `dilations` はすべて 1 でなければならない（dilated average pooling 非対応）"
            ),
        });
    }
    let (pad_begins, pad_ends) = parse_pool_pads(op, &attrs.pads, spatial_rank)?;
    let _ceil_mode = parse_bool_flag(op, "ceil_mode", attrs.ceil_mode)?;
    let _storage_order = parse_bool_flag(op, "storage_order", attrs.storage_order)?;

    let (kh, kw) = if is_1d {
        (1, kernel[0])
    } else {
        (kernel[0], kernel[1])
    };
    let (sh, sw) = if is_1d {
        (1, strides[0])
    } else {
        (strides[0], strides[1])
    };
    let (dh, dw) = if is_1d {
        (1, dilations[0])
    } else {
        (dilations[0], dilations[1])
    };
    let (ph_b, pw_b) = if is_1d {
        (0, pad_begins[0])
    } else {
        (pad_begins[0], pad_begins[1])
    };
    let (ph_e, pw_e) = if is_1d {
        (0, pad_ends[0])
    } else {
        (pad_ends[0], pad_ends[1])
    };

    let x4 = if is_1d {
        let s = x.shape();
        let (n, c, l) = (s[0], s[1], s[2]);
        x.contiguous()
            .reshape(&[n, c, 1, l])
            .map_err(OpError::from)?
    } else {
        x.contiguous()
    };

    Ok((x4, kh, kw, sh, sw, dh, dw, ph_b, ph_e, pw_b, pw_e, is_1d))
}

/// `MaxPool(X)` を計算する（イシュー #2199）。検証順序: `auto_pad` →
/// `X` の rank（3 または 4）→ `kernel_shape`（必須）／`strides`／
/// `dilations`／`pads` の属性検査 → `ceil_mode`／`storage_order` の
/// `{0,1}` 検査 → 出力長計算（`pool_out_axis_len`）→ 空間出力規模の
/// 上限検査（`ensure_spatial_out_bound`。DoS 対策）→ 空窓検査
/// （`axis_windows_nonempty`）→ 直接ループでの計算。1 つでも失敗すれば
/// 以降の計算を行わない（`.claude/rules/security.md` A08）。
///
/// 数値契約: `kh` 外側・`kw` 内側の row-major で走査し、`v > best ||
/// (v.is_nan() && !best.is_nan())` の場合のみ更新する（先勝ち決定的・
/// NaN 伝播。`fandhe_ai_backend_cpu::pooling::max_pool2d` と同一規則）。
pub fn max_pool(x: &Tensor<f32>, attrs: &PoolAttrs) -> Result<Tensor<f32>, OpError> {
    const OP: &str = "MaxPool";
    let (x4, kh, kw, sh, sw, dh, dw, ph_b, ph_e, pw_b, pw_e, is_1d) =
        validate_and_lift(OP, x, attrs, false)?;
    let ceil_mode = parse_bool_flag(OP, "ceil_mode", attrs.ceil_mode)?;

    let s4 = x4.shape().to_vec();
    let (n, c, h_in, w_in) = (s4[0], s4[1], s4[2], s4[3]);
    let h_out = pool_out_axis_len(
        OP,
        h_in,
        PoolAxisParams {
            k: kh,
            s: sh,
            d: dh,
            pb: ph_b,
            pe: ph_e,
        },
        ceil_mode,
    )?;
    let w_out = pool_out_axis_len(
        OP,
        w_in,
        PoolAxisParams {
            k: kw,
            s: sw,
            d: dw,
            pb: pw_b,
            pe: pw_e,
        },
        ceil_mode,
    )?;
    // `axis_windows_nonempty`／直接計算ループ・出力バッファ確保のいずれも
    // `h_out`／`w_out` に比例するため、それらへ進む前に空間出力規模を
    // 上限検査する（DoS 対策。関数群 doc の「検証順序」・
    // `ensure_spatial_out_bound` doc 参照）。
    ensure_spatial_out_bound(OP, h_out, w_out)?;
    if !axis_windows_nonempty(h_out, kh, sh, dh, ph_b, h_in) {
        return Err(OpError::InvalidPoolAttribute {
            reason: format!("{OP}: H 軸に有効タップを持たない出力窓が存在する"),
        });
    }
    if !axis_windows_nonempty(w_out, kw, sw, dw, pw_b, w_in) {
        return Err(OpError::InvalidPoolAttribute {
            reason: format!("{OP}: W 軸に有効タップを持たない出力窓が存在する"),
        });
    }

    let x_slice = x4
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("MaxPool(X)"))?;

    let out_len = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(h_out))
        .and_then(|v| v.checked_mul(w_out))
        .ok_or(OpError::Shape(
            fandhe_ai_tensor_core::ShapeError::ElementCountOverflow,
        ))?;
    let mut out = vec![0f32; out_len];
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                // H 軸の有効タップ範囲は `ow` に依存しないため `oh` ループの
                // 外で 1 度だけ求める（[`valid_tap_range`] は `O(1)`）。
                let h_range = valid_tap_range(oh, sh, kh, dh, ph_b, h_in);
                for ow in 0..w_out {
                    let mut best: Option<f32> = None;
                    if let Some((h_lo, h_hi)) = h_range
                        && let Some((w_lo, w_hi)) = valid_tap_range(ow, sw, kw, dw, pw_b, w_in)
                    {
                        // 反復回数は `h_hi-h_lo+1`／`w_hi-w_lo+1`（高々
                        // `in_len/dilation + 1`）に収まり、`kernel_shape`
                        // の値そのもの（外部 ONNX 属性）には依存しない
                        // （DoS 対策。イシュー #2199 codex-review 指摘）。
                        for khi in h_lo..=h_hi {
                            let Some(ih) =
                                window_input_pos(oh, sh, khi, dh, ph_b).filter(|&v| v < h_in)
                            else {
                                continue;
                            };
                            for kwi in w_lo..=w_hi {
                                let Some(iw) =
                                    window_input_pos(ow, sw, kwi, dw, pw_b).filter(|&v| v < w_in)
                                else {
                                    continue;
                                };
                                let x_idx = ((ni * c + ci) * h_in + ih) * w_in + iw;
                                let v = *x_slice
                                    .get(x_idx)
                                    .ok_or(OpError::NonContiguousInternal("MaxPool(X read)"))?;
                                best = Some(match best {
                                    None => v,
                                    Some(b) => {
                                        if v > b || (v.is_nan() && !b.is_nan()) {
                                            v
                                        } else {
                                            b
                                        }
                                    }
                                });
                            }
                        }
                    }
                    // `axis_windows_nonempty` の事前検査により到達しないはず
                    // だが、契約違反時に panic せず型付きエラーで拒否する
                    // （REQ-8・A08。`fandhe_ai_backend_cpu::pooling::max_pool2d`
                    // と同じ安全側フォールバック）。
                    let v = best.ok_or(OpError::Shape(
                        fandhe_ai_tensor_core::ShapeError::ElementCountOverflow,
                    ))?;
                    let out_idx = ((ni * c + ci) * h_out + oh) * w_out + ow;
                    if let Some(slot) = out.get_mut(out_idx) {
                        *slot = v;
                    } else {
                        return Err(OpError::NonContiguousInternal("MaxPool(out write)"));
                    }
                }
            }
        }
    }

    let out_shape: Vec<usize> = if is_1d {
        vec![n, c, w_out]
    } else {
        vec![n, c, h_out, w_out]
    };
    Tensor::new(out, &out_shape).map_err(OpError::from)
}

/// `AveragePool(X)` を計算する（イシュー #2199）。検証は [`max_pool`]
/// とほぼ同順序だが、`dilations` はすべて 1 のみ受理し（§3.1）、空窓
/// 検査（`axis_windows_nonempty`）は `count_include_pad=0` の場合のみ
/// 行う。`count_include_pad=1` では全タップが padding の窓（空窓）も
/// ONNX 上正当（divisor は padded 座標基準で計算され `acc=0` のまま
/// `0` を返す）であり、空窓を一律拒否すると当該窓を含む import 全体を
/// 誤って拒否してしまうため（Bugbot 指摘 #2199 対応。`count_include_pad=1`
/// のとき divisor が 0 にならないことは `pool_out_axis_len` の出力長
/// 調整規則（最後の窓が padded 範囲内で始まる）から保証される）。
///
/// 数値契約: 有効タップを `kh` 外側・`kw` 内側の row-major で `f64` へ
/// 逐次加算し `(acc / divisor as f64) as f32` で 1 回だけ downcast する
/// （`fandhe_ai_backend_cpu::pooling::avg_pool2d`・
/// `.claude/rules/coding-rust.md` の f64 縮約規約と同型）。divisor は
/// `count_include_pad=0` なら有効タップ数、`count_include_pad=1` なら
/// 軸ごとの `min(out_idx*stride + kernel, in_len+pad_begin+pad_end) -
/// out_idx*stride` の積（ceil モードのはみ出し分を含めない。§3.1）。
/// 有効タップの走査は `valid_tap_range` が返す `O(1)` 範囲に限るため、
/// `max_pool` と同様に `kernel_shape` の値そのものには反復回数が依存
/// しない。加えて `h_out * w_out`（空間出力規模）自体も
/// `ensure_spatial_out_bound` で上限検査する（`max_pool` と同型の DoS
/// 対策。イシュー #2199 codex-review 指摘。`ensure_spatial_out_bound`
/// doc 参照）。
pub fn average_pool(x: &Tensor<f32>, attrs: &PoolAttrs) -> Result<Tensor<f32>, OpError> {
    const OP: &str = "AveragePool";
    let (x4, kh, kw, sh, sw, dh, dw, ph_b, ph_e, pw_b, pw_e, is_1d) =
        validate_and_lift(OP, x, attrs, true)?;
    let ceil_mode = parse_bool_flag(OP, "ceil_mode", attrs.ceil_mode)?;
    let count_include_pad = parse_bool_flag(OP, "count_include_pad", attrs.count_include_pad)?;

    let s4 = x4.shape().to_vec();
    let (n, c, h_in, w_in) = (s4[0], s4[1], s4[2], s4[3]);
    let h_out = pool_out_axis_len(
        OP,
        h_in,
        PoolAxisParams {
            k: kh,
            s: sh,
            d: dh,
            pb: ph_b,
            pe: ph_e,
        },
        ceil_mode,
    )?;
    let w_out = pool_out_axis_len(
        OP,
        w_in,
        PoolAxisParams {
            k: kw,
            s: sw,
            d: dw,
            pb: pw_b,
            pe: pw_e,
        },
        ceil_mode,
    )?;
    // `count_include_pad` の値に関わらず、後続の直接計算ループ・出力
    // バッファ確保は `h_out`／`w_out` に比例するため、空窓検査の分岐
    // より前に空間出力規模を上限検査する（DoS 対策。
    // `ensure_spatial_out_bound` doc 参照）。
    ensure_spatial_out_bound(OP, h_out, w_out)?;
    // `count_include_pad=1` では divisor が padded 座標基準で計算され
    // 空窓でも 0 除算にならないため、空窓検査は `count_include_pad=0`
    // （divisor=valid_count のため空窓が 0 除算に直結する）の場合のみ
    // 行う（Bugbot 指摘 #2199 対応。関数 doc 参照）。
    if !count_include_pad {
        if !axis_windows_nonempty(h_out, kh, sh, dh, ph_b, h_in) {
            return Err(OpError::InvalidPoolAttribute {
                reason: format!("{OP}: H 軸に有効タップを持たない出力窓が存在する"),
            });
        }
        if !axis_windows_nonempty(w_out, kw, sw, dw, pw_b, w_in) {
            return Err(OpError::InvalidPoolAttribute {
                reason: format!("{OP}: W 軸に有効タップを持たない出力窓が存在する"),
            });
        }
    }

    let x_slice = x4
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("AveragePool(X)"))?;

    let padded_h = h_in
        .checked_add(ph_b)
        .and_then(|v| v.checked_add(ph_e))
        .ok_or(OpError::InvalidPoolAttribute {
            reason: format!("{OP}: H 軸の padded 長がオーバーフローした"),
        })?;
    let padded_w = w_in
        .checked_add(pw_b)
        .and_then(|v| v.checked_add(pw_e))
        .ok_or(OpError::InvalidPoolAttribute {
            reason: format!("{OP}: W 軸の padded 長がオーバーフローした"),
        })?;

    let out_len = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(h_out))
        .and_then(|v| v.checked_mul(w_out))
        .ok_or(OpError::Shape(
            fandhe_ai_tensor_core::ShapeError::ElementCountOverflow,
        ))?;
    let mut out = vec![0f32; out_len];
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                // `count_include_pad=1` の H 軸 divisor 寄与
                // （padded 座標でのクリップ。ceil モードのはみ出しのみ
                // 除外する。§3.1）。
                let h_start = oh.checked_mul(sh).ok_or(OpError::InvalidPoolAttribute {
                    reason: format!("{OP}: H 軸の窓開始位置計算がオーバーフローした"),
                })?;
                let h_divisor_incl = h_start
                    .checked_add(kh)
                    .map(|end| end.min(padded_h))
                    .and_then(|end| end.checked_sub(h_start))
                    .ok_or(OpError::InvalidPoolAttribute {
                        reason: format!("{OP}: H 軸の divisor 計算がオーバーフローした"),
                    })?;
                // H 軸の有効タップ範囲（`ow` に依存しないため `oh` ループの
                // 外で 1 度だけ求める。`count_include_pad=1` では空窓
                // （`None`）もありうる。[`valid_tap_range`] doc 参照）。
                let h_range = valid_tap_range(oh, sh, kh, dh, ph_b, h_in);
                for ow in 0..w_out {
                    let w_start = ow.checked_mul(sw).ok_or(OpError::InvalidPoolAttribute {
                        reason: format!("{OP}: W 軸の窓開始位置計算がオーバーフローした"),
                    })?;
                    let w_divisor_incl = w_start
                        .checked_add(kw)
                        .map(|end| end.min(padded_w))
                        .and_then(|end| end.checked_sub(w_start))
                        .ok_or(OpError::InvalidPoolAttribute {
                            reason: format!("{OP}: W 軸の divisor 計算がオーバーフローした"),
                        })?;

                    let mut acc: f64 = 0.0;
                    let mut valid_count: usize = 0;
                    // 反復回数は `kernel_shape` の値そのものに依存せず
                    // `h_range`／`w_range` の幅（高々 `in_len/dilation+1`）
                    // に収まる（DoS 対策。イシュー #2199 codex-review 指摘）。
                    if let Some((h_lo, h_hi)) = h_range
                        && let Some((w_lo, w_hi)) = valid_tap_range(ow, sw, kw, dw, pw_b, w_in)
                    {
                        for khi in h_lo..=h_hi {
                            let Some(ih) =
                                window_input_pos(oh, sh, khi, dh, ph_b).filter(|&v| v < h_in)
                            else {
                                continue;
                            };
                            for kwi in w_lo..=w_hi {
                                let Some(iw) =
                                    window_input_pos(ow, sw, kwi, dw, pw_b).filter(|&v| v < w_in)
                                else {
                                    continue;
                                };
                                let x_idx = ((ni * c + ci) * h_in + ih) * w_in + iw;
                                let v = *x_slice
                                    .get(x_idx)
                                    .ok_or(OpError::NonContiguousInternal("AveragePool(X read)"))?;
                                acc += f64::from(v);
                                valid_count += 1;
                            }
                        }
                    }
                    let divisor = if count_include_pad {
                        h_divisor_incl.checked_mul(w_divisor_incl).ok_or(
                            OpError::InvalidPoolAttribute {
                                reason: format!("{OP}: divisor 計算がオーバーフローした"),
                            },
                        )?
                    } else {
                        valid_count
                    };
                    // `count_include_pad=0` では `axis_windows_nonempty` の
                    // 事前検査により到達しないはずだが、契約違反時に panic
                    // せず型付きエラーで拒否する（REQ-8・A08）。
                    // `count_include_pad=1` では空窓自体は正当（`acc=0`）
                    // だが、`divisor` が 0 になるのは [`pool_out_axis_len`]
                    // の出力長調整規則違反時のみで理論上到達しない
                    // フォールバックである。
                    if divisor == 0 {
                        return Err(OpError::InvalidPoolAttribute {
                            reason: format!("{OP}: divisor が 0 になった（空窓）"),
                        });
                    }
                    let v = (acc / divisor as f64) as f32;
                    let out_idx = ((ni * c + ci) * h_out + oh) * w_out + ow;
                    if let Some(slot) = out.get_mut(out_idx) {
                        *slot = v;
                    } else {
                        return Err(OpError::NonContiguousInternal("AveragePool(out write)"));
                    }
                }
            }
        }
    }

    let out_shape: Vec<usize> = if is_1d {
        vec![n, c, w_out]
    } else {
        vec![n, c, h_out, w_out]
    };
    Tensor::new(out, &out_shape).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs(kernel: Vec<i64>) -> PoolAttrs {
        PoolAttrs {
            kernel_shape: kernel,
            ..PoolAttrs::default()
        }
    }

    #[test]
    fn max_pool_2x2_stride2_no_padding() {
        // X: [1,1,2,2] = [[1,2],[3,4]] -> single window -> max=4
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let y = max_pool(&x, &attrs(vec![2, 2])).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1, 1]);
        assert_eq!(y.get(&[0, 0, 0, 0]).unwrap(), 4.0);
    }

    #[test]
    fn max_pool_overlapping_windows() {
        // X: [1,1,1,4] = [1,3,2,4], kernel=2, stride=1 -> windows: max(1,3)=3, max(3,2)=3, max(2,4)=4
        let x = Tensor::<f32>::new(vec![1.0, 3.0, 2.0, 4.0], &[1, 1, 1, 4]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![1, 2],
            strides: vec![1, 1],
            ..PoolAttrs::default()
        };
        let y = max_pool(&x, &attrs).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1, 3]);
        assert_eq!(y.get(&[0, 0, 0, 0]).unwrap(), 3.0);
        assert_eq!(y.get(&[0, 0, 0, 1]).unwrap(), 3.0);
        assert_eq!(y.get(&[0, 0, 0, 2]).unwrap(), 4.0);
    }

    #[test]
    fn max_pool_tie_first_match_wins() {
        // 同値タイでは走査順で先に見つかった値が残る（NaN 以外は同値なので
        // 実質どちらでも同じだが、更新条件 `v > best` を明示的に固定する）。
        let x = Tensor::<f32>::new(vec![5.0, 5.0], &[1, 1, 1, 2]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![1, 2],
            ..PoolAttrs::default()
        };
        let y = max_pool(&x, &attrs).unwrap();
        assert_eq!(y.get(&[0, 0, 0, 0]).unwrap(), 5.0);
    }

    #[test]
    fn max_pool_nan_propagates() {
        let x = Tensor::<f32>::new(vec![1.0, f32::NAN, 2.0], &[1, 1, 1, 3]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![1, 3],
            ..PoolAttrs::default()
        };
        let y = max_pool(&x, &attrs).unwrap();
        assert!(y.get(&[0, 0, 0, 0]).unwrap().is_nan());
    }

    #[test]
    fn max_pool_1d_matches_2d_lift() {
        let x1d = Tensor::<f32>::new(vec![1.0, 3.0, 2.0, 4.0], &[1, 1, 4]).unwrap();
        let x2d = Tensor::<f32>::new(vec![1.0, 3.0, 2.0, 4.0], &[1, 1, 1, 4]).unwrap();
        let attrs1d = PoolAttrs {
            kernel_shape: vec![2],
            strides: vec![2],
            ..PoolAttrs::default()
        };
        let attrs2d = PoolAttrs {
            kernel_shape: vec![1, 2],
            strides: vec![1, 2],
            ..PoolAttrs::default()
        };
        let y1d = max_pool(&x1d, &attrs1d).unwrap();
        let y2d = max_pool(&x2d, &attrs2d).unwrap();
        assert_eq!(y1d.shape(), &[1, 1, 2]);
        assert_eq!(y2d.shape(), &[1, 1, 1, 2]);
        for i in 0..2 {
            assert_eq!(
                y1d.get(&[0, 0, i]).unwrap(),
                y2d.get(&[0, 0, 0, i]).unwrap()
            );
        }
    }

    #[test]
    fn max_pool_ceil_mode_adjusts_output_length() {
        // L=5, k=2, s=2, p=1: floor -> (5+2-2)/2+1 = 3。ceil (padded=7)
        // -> ceil(5/2)+1=4 だが調整規則で最後の窓が (4-1)*2=6 >= 5+1=6 の
        // ため 3 に戻る（調整規則が発火するケース。実装計画 §4 記載）。
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 1, 5]).unwrap();
        let floor_attrs = PoolAttrs {
            kernel_shape: vec![1, 2],
            strides: vec![1, 2],
            pads: vec![0, 1, 0, 1],
            ..PoolAttrs::default()
        };
        let ceil_attrs = PoolAttrs {
            ceil_mode: 1,
            ..floor_attrs.clone()
        };
        let y_floor = max_pool(&x, &floor_attrs).unwrap();
        let y_ceil = max_pool(&x, &ceil_attrs).unwrap();
        assert_eq!(y_floor.shape(), &[1, 1, 1, 3]);
        assert_eq!(y_ceil.shape(), &[1, 1, 1, 3]);
    }

    #[test]
    fn max_pool_asymmetric_pads_accepted() {
        // pads=[0,1] だが h/w にそれぞれ begin/end が異なるケース。
        // pads (ONNX順, spatial_rank=2): [h_b, w_b, h_e, w_e] = [0,0,0,1]
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[1, 1, 1, 3]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![1, 2],
            strides: vec![1, 2],
            pads: vec![0, 0, 0, 1],
            ..PoolAttrs::default()
        };
        let y = max_pool(&x, &attrs).unwrap();
        // padded (asym end only): [1,2,3,0] -> windows: max(1,2)=2, max(3,0)=3
        assert_eq!(y.shape(), &[1, 1, 1, 2]);
        assert_eq!(y.get(&[0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(y.get(&[0, 0, 0, 1]).unwrap(), 3.0);
    }

    #[test]
    fn average_pool_count_include_pad_false() {
        // X:[1,1,1,3]=[1,2,3], kernel=2, stride=1, pad=[0,1] (w only, symmetric via [0,0,0,1]? use begin=0,end=1)
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[1, 1, 1, 3]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![1, 2],
            strides: vec![1, 1],
            pads: vec![0, 0, 0, 1],
            count_include_pad: 0,
            ..PoolAttrs::default()
        };
        let y = average_pool(&x, &attrs).unwrap();
        // windows: (1,2)/2=1.5, (2,3)/2=2.5, (3,pad)/1=3.0 (count_include_pad=0)
        assert_eq!(y.shape(), &[1, 1, 1, 3]);
        assert_eq!(y.get(&[0, 0, 0, 0]).unwrap(), 1.5);
        assert_eq!(y.get(&[0, 0, 0, 1]).unwrap(), 2.5);
        assert_eq!(y.get(&[0, 0, 0, 2]).unwrap(), 3.0);
    }

    #[test]
    fn average_pool_count_include_pad_true() {
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[1, 1, 1, 3]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![1, 2],
            strides: vec![1, 1],
            pads: vec![0, 0, 0, 1],
            count_include_pad: 1,
            ..PoolAttrs::default()
        };
        let y = average_pool(&x, &attrs).unwrap();
        // count_include_pad=1: last window divisor=kernel(2) not clipped since
        // padded_w = 3+0+1=4, window [2,4) end=4 == padded_w -> divisor=2.
        assert_eq!(y.get(&[0, 0, 0, 2]).unwrap(), 1.5); // (3+0)/2
    }

    #[test]
    fn average_pool_f64_accumulation_order() {
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![2, 2],
            ..PoolAttrs::default()
        };
        let y = average_pool(&x, &attrs).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1, 1]);
        assert_eq!(y.get(&[0, 0, 0, 0]).unwrap(), 2.5);
    }

    #[test]
    fn average_pool_1d_matches_2d_lift() {
        let x1d = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]).unwrap();
        let x2d = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]).unwrap();
        let attrs1d = PoolAttrs {
            kernel_shape: vec![2],
            strides: vec![2],
            ..PoolAttrs::default()
        };
        let attrs2d = PoolAttrs {
            kernel_shape: vec![1, 2],
            strides: vec![1, 2],
            ..PoolAttrs::default()
        };
        let y1d = average_pool(&x1d, &attrs1d).unwrap();
        let y2d = average_pool(&x2d, &attrs2d).unwrap();
        assert_eq!(y1d.shape(), &[1, 1, 2]);
        for i in 0..2 {
            assert_eq!(
                y1d.get(&[0, 0, i]).unwrap(),
                y2d.get(&[0, 0, 0, i]).unwrap()
            );
        }
    }

    #[test]
    fn average_pool_dilation_not_one_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 4, 4]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![2, 2],
            dilations: vec![2, 2],
            ..PoolAttrs::default()
        };
        let err = average_pool(&x, &attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidPoolAttribute { .. }));
    }

    #[test]
    fn kernel_shape_missing_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 4, 4]).unwrap();
        let err = max_pool(&x, &PoolAttrs::default()).unwrap_err();
        assert!(matches!(err, OpError::InvalidPoolAttribute { .. }));
    }

    #[test]
    fn kernel_shape_length_mismatch_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 4, 4]).unwrap();
        let err = max_pool(&x, &attrs(vec![2, 2, 2])).unwrap_err();
        assert!(matches!(err, OpError::InvalidPoolAttribute { .. }));
    }

    #[test]
    fn negative_kernel_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 4, 4]).unwrap();
        let err = max_pool(&x, &attrs(vec![-1, 2])).unwrap_err();
        assert!(matches!(err, OpError::InvalidPoolAttribute { .. }));
    }

    #[test]
    fn zero_kernel_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 4, 4]).unwrap();
        let err = max_pool(&x, &attrs(vec![0, 2])).unwrap_err();
        assert!(matches!(err, OpError::InvalidPoolAttribute { .. }));
    }

    #[test]
    fn zero_stride_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 4, 4]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![2, 2],
            strides: vec![0, 1],
            ..PoolAttrs::default()
        };
        let err = max_pool(&x, &attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidPoolAttribute { .. }));
    }

    #[test]
    fn ceil_mode_out_of_range_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 4, 4]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![2, 2],
            ceil_mode: 2,
            ..PoolAttrs::default()
        };
        let err = max_pool(&x, &attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidPoolAttribute { .. }));
    }

    #[test]
    fn same_upper_auto_pad_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 4, 4]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![2, 2],
            auto_pad: "SAME_UPPER".to_string(),
            ..PoolAttrs::default()
        };
        let err = max_pool(&x, &attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidPoolAttribute { .. }));
    }

    #[test]
    fn rank5_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 1, 4, 4]).unwrap();
        let err = max_pool(&x, &attrs(vec![2, 2])).unwrap_err();
        assert!(matches!(
            err,
            OpError::RankMismatch {
                op: "MaxPool",
                expected: 4,
                actual: 5,
            }
        ));
    }

    #[test]
    fn empty_window_rejected() {
        // W 軸: kernel=1・pad_begin=1 のため out_idx=0 の窓が padding のみ
        // となり空になる（padded_w=1+1+0=2 -> out_w=2、out_idx=0 は
        // window_input_pos(0,1,0,1,1)= 0-1 でアンダーフロー -> 無効）。
        // Bugbot 指摘 #2199: 以前の同名テストは実際には空窓を生成せず
        // `is_ok()` を assert しており契約を検証していなかった。
        let x = Tensor::<f32>::zeros(&[1, 1, 1, 1]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![1, 1],
            pads: vec![0, 1, 0, 0],
            ..PoolAttrs::default()
        };
        let err = max_pool(&x, &attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidPoolAttribute { .. }));
    }

    #[test]
    fn average_pool_empty_window_count_include_pad_false_rejected() {
        // `empty_window_rejected` と同じ空窓（W 軸 out_idx=0）。
        // `count_include_pad=0` では divisor=valid_count のため 0 除算に
        // 直結する空窓を事前検査で拒否する。
        let x = Tensor::<f32>::zeros(&[1, 1, 1, 1]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![1, 1],
            pads: vec![0, 1, 0, 0],
            count_include_pad: 0,
            ..PoolAttrs::default()
        };
        let err = average_pool(&x, &attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidPoolAttribute { .. }));
    }

    #[test]
    fn average_pool_empty_window_count_include_pad_true_accepted() {
        // Bugbot 指摘 #2199: `count_include_pad=1` では全タップが padding
        // の窓も ONNX 上正当（divisor は padded 座標基準・acc=0 のため
        // 結果は 0）であり、一律拒否は誤り。
        let x = Tensor::<f32>::new(vec![5.0], &[1, 1, 1, 1]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![1, 1],
            pads: vec![0, 1, 0, 0],
            count_include_pad: 1,
            ..PoolAttrs::default()
        };
        let y = average_pool(&x, &attrs).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1, 2]);
        // out_idx=0: 完全 padding 窓 -> 0/divisor(1) = 0。
        assert_eq!(y.get(&[0, 0, 0, 0]).unwrap(), 0.0);
        // out_idx=1: window_input_pos(1,1,0,1,1)=0 -> 有効タップ 1 個 (5.0)。
        assert_eq!(y.get(&[0, 0, 0, 1]).unwrap(), 5.0);
    }

    #[test]
    fn max_pool_huge_kernel_shape_with_single_valid_tap_completes_fast() {
        // イシュー #2199 codex-review 指摘の再現ケース: kernel_shape が
        // 約 10 億でも、事前検査（`axis_windows_nonempty`）・直接ループの
        // 双方が `kernel_shape` に比例した反復を行わない（`O(1)`
        // `valid_tap_range` 経由）ため、テストが実用時間で完了する。
        // W: k=1_000_000_000, pad_begin=999_999_999, in_len=1
        // -> padded_w = 1_000_000_000 = eff_w のため出力長 1、
        //    有効タップは ki=999_999_999 のみ。
        let x = Tensor::<f32>::new(vec![7.0], &[1, 1, 1, 1]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![1, 1_000_000_000],
            pads: vec![0, 999_999_999, 0, 0],
            ..PoolAttrs::default()
        };
        let y = max_pool(&x, &attrs).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1, 1]);
        assert_eq!(y.get(&[0, 0, 0, 0]).unwrap(), 7.0);
    }

    #[test]
    fn max_pool_huge_pads_producing_huge_out_len_rejected_fast() {
        // イシュー #2199 codex-review 指摘（P0）の再現ケース: 前段の
        // `max_pool_huge_kernel_shape_with_single_valid_tap_completes_fast`
        // は `out_len == 1` になる組み合わせのため DoS を再現しない。
        // 本テストは指摘そのものの属性（in_len=1・kernel_shape=
        // 1_000_000_000・pad_begin=pad_end=999_999_999・stride=1）を使い、
        // `out_len` が約 10 億に達する組み合わせを再現する。
        // `ensure_spatial_out_bound` が `axis_windows_nonempty`／直接計算
        // ループへ進む前に拒否するため、`#[test]` の既定タイムアウト内で
        // 完了する（修正前は約 10 億回の走査で長時間停止した）。
        let x = Tensor::<f32>::zeros(&[1, 1, 1, 1]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![1, 1_000_000_000],
            pads: vec![0, 999_999_999, 0, 999_999_999],
            ..PoolAttrs::default()
        };
        let err = max_pool(&x, &attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidPoolAttribute { .. }));
    }

    #[test]
    fn average_pool_huge_pads_producing_huge_out_len_rejected_fast() {
        // 上記 `max_pool` 版と同型。`count_include_pad=1`（既定）は
        // `axis_windows_nonempty` を呼ばないが、直接計算ループ・出力
        // バッファ確保が `out_len` に比例するため、こちらも
        // `ensure_spatial_out_bound` で同じ属性を拒否できることを検証する。
        let x = Tensor::<f32>::zeros(&[1, 1, 1, 1]).unwrap();
        let attrs = PoolAttrs {
            kernel_shape: vec![1, 1_000_000_000],
            pads: vec![0, 999_999_999, 0, 999_999_999],
            ..PoolAttrs::default()
        };
        let err = average_pool(&x, &attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidPoolAttribute { .. }));
    }

    #[test]
    fn ensure_spatial_out_bound_accepts_at_cap_and_rejects_over_cap() {
        // `ensure_spatial_out_bound` の境界値検査（実際の pool 計算経路
        // からは独立にユニットテストする）。
        assert!(ensure_spatial_out_bound("Test", 1, MAX_POOL_SPATIAL_OUT_ELEMENTS).is_ok());
        assert!(ensure_spatial_out_bound("Test", 1, MAX_POOL_SPATIAL_OUT_ELEMENTS + 1).is_err());
        // オーバーフローも上限超過として拒否する。
        assert!(ensure_spatial_out_bound("Test", usize::MAX, 2).is_err());
    }

    #[test]
    fn huge_pads_and_strides_no_panic() {
        let x = Tensor::<f32>::zeros(&[1, 1, 3, 3]).unwrap();
        let huge_pad = 1i64 << 62;
        let attrs = PoolAttrs {
            kernel_shape: vec![2, 2],
            pads: vec![huge_pad, huge_pad, huge_pad, huge_pad],
            strides: vec![i64::MAX, i64::MAX],
            ..PoolAttrs::default()
        };
        // padded は 3 + 2*huge_pad ≈ 2^63 で checked_add がオーバーフロー
        // し得るため、fail-closed に拒否されることのみ確認する（panic しない）。
        let _ = max_pool(&x, &attrs);
    }
}
