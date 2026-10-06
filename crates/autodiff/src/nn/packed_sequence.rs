//! 可変長系列の pack／unpack（`PackedSequence`）と RNN 系への結線
//! （イシュー #2647・親 #2625「Phase 4」・ルート #2499）。
//!
//! PyTorch `torch.nn.utils.rnn` の `pack_padded_sequence`／`pad_packed_sequence`／
//! `PackedSequence` 相当を、`Rnn`／`Lstm`／`Gru`・`StackedRnn`／`StackedLstm`／
//! `StackedGru`（`rnn.rs`・`rnn_stacked.rs`）から処理できる形で提供する。既存の
//! `forward_seq` は全系列が同じ長さの `[T,B,D]` しか受けられず、
//! `docs/autodiff-rnn-cell-tape-design.md` が v1 スコープ外としていた
//! 「`pack_padded_sequence` 相当の可変長系列」を本モジュールが内部実装として埋める。
//!
//! **新規 `Op`・`BackendOps` メソッド・VJP はゼロ**: pack／unpack は値のコピーだけで
//! 算術を含まないため、既存の `Var::index_select`（`Op::Gather`）・`Var::cat`
//! （`Op::Concat`）・`Var::narrow`（`Op::Narrow`）・`Var::reshape` と、RNN 系は既存の
//! セル `Op`（`Var::rnn_cell`／`lstm_cell`／`gru_cell`）の合成だけで構成する
//! （#2639 `shape_view_ops`・#2164 `rnn_stacked` と同じ方式）。公開済みクレート
//! `fandhe-ai-tensor-core` の trait 面は広げず、`crates/tensor-core`・`crates/backend-*`
//! は変更しない。
//!
//! **バックエンド到達性**: `gather`／`scatter`／`concat` は 3 バックエンドが実カーネル
//! または `Unsupported`→ホスト参照実装のフォールバックを持つ既存経路で、RNN セルも
//! 既存の `BackendOps` メソッドを呼ぶ。本モジュールは GPU 専用カーネルを持たない。
//!
//! **公開形（未承認・保留）**: facade（`fandhe_ai`）への公開形は未承認で、承認依頼は
//! #2677・公開自体は承認後の #2678・#2679。推奨案は決定記録
//! `docs/autodiff-packed-sequence-decision.md` §7（推奨案の記録であり承認記録ではない）。
//! 保留中は `PackedSequenceHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
//! `crates/facade/tests/api_surface.rs` の否定ガードが facade への漏出を拒否する。
//! このため本モジュールの追加物は **自由関数と本モジュール専用の型だけ**とし、facade が
//! 再エクスポート済みの型（`Var`・`Tape`・`Rnn` など）へ inherent メソッドを足さない。
//!
//! **数値契約**: pack／unpack の forward はコピーのみで PyTorch と bit 完全一致する。
//! 勾配と RNN 出力は REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5
//! 未満）で比較し、tolerance は変更しない。FMA 契約・`f64` アキュムレータ契約の新たな
//! 対象はない。
//!
//! **PyTorch との差分**:
//! - 層間 dropout は packed の `data` へ層ごとに 1 回適用する（PyTorch も `data` へ適用
//!   する）。padded 版 `forward_seq`（step ごとに適用）とは RNG 消費順が異なる。
//! - `lengths`・添字は `usize`。長さ 0 の系列・負の添字・テンソル引数形は非対応。
//! - `h0`／`c0`／`h_n`／`c_n` は元のバッチ順の `[B,H]`（`sorted_indices` で内部的に
//!   並べ替える）。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: `lengths`・`total_length`・
//! `PackedSequence::new` の `batch_sizes`／`sorted_indices`・`h0`／`c0` は外部入力として
//! 扱い、tape へノードを積む前にすべて検査する（引数起因のエラーで孤児ノードを残さない）。
//! 加算・乗算は `checked_*`、`usize`→`i32` は `i32::try_from`、添字ベクタは
//! `checked_index_alloc_len`（1 GiB 上限）と `try_reserve_exact` で確保前に検査する。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::module::Module;
use crate::nn::rnn::{Gru, GruCellVars, Lstm, LstmCellVars, Rnn, RnnCellVars, reserve_outputs};
use crate::nn::rnn_stacked::{StackedGru, StackedLstm, StackedRnn};
use crate::rearrange_ops::{checked_axis_len_as_i32, checked_index_alloc_len};
use crate::var::Var;

// =====================================================================
// 内部ヘルパー
// =====================================================================

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::InvalidArgument(msg)
}

fn overflow() -> AutodiffError {
    AutodiffError::Shape(ShapeError::ElementCountOverflow)
}

/// `usize` の添字列を `i32` の 1 次元 `Tensor` にする（`Var::index_select` の入力）。
/// 確保前に `checked_index_alloc_len` を通し、各要素は `i32::try_from` で変換する。
fn index_tensor(idx: &[usize]) -> Result<Tensor<i32>, AutodiffError> {
    checked_index_alloc_len(idx.len())?;
    let mut data: Vec<i32> = Vec::new();
    data.try_reserve_exact(idx.len()).map_err(|_| overflow())?;
    for &v in idx {
        data.push(i32::try_from(v).map_err(|_| overflow())?);
    }
    Ok(Tensor::new(data, &[idx.len()])?)
}

/// 置換 `perm` の逆置換（`inv[perm[i]] = i`）。
fn invert_permutation(perm: &[usize]) -> Result<Vec<usize>, AutodiffError> {
    let mut inv: Vec<usize> = Vec::new();
    inv.try_reserve_exact(perm.len()).map_err(|_| overflow())?;
    inv.resize(perm.len(), 0);
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    Ok(inv)
}

/// `index_select(0, idx)` を `idx` が `None` のときは恒等として適用する。
fn reorder_rows<'t>(v: &Var<'t>, idx: Option<&[usize]>) -> Result<Var<'t>, AutodiffError> {
    match idx {
        Some(i) => v.index_select(0, &index_tensor(i)?),
        None => Ok(*v),
    }
}

// =====================================================================
// PackedSequence
// =====================================================================

/// `pack_padded_sequence` が返す、パディングを除いて詰めた可変長系列
/// （PyTorch `PackedSequence` 相当）。
///
/// - `data`: `[N, *]`（`N = Σ lengths`）。tape に束縛された `Var` で、pack を通って
///   入力へ勾配が流れる。並びは「step 0 の全系列 → step 1 の生存系列 → …」（長さ降順）。
/// - `batch_sizes`: 長さ `T_max = max(lengths)`。各 step の生存系列数（非増加・各要素
///   ≥ 1・総和 = `N`）。
/// - `sorted_indices`／`unsorted_indices`: `enforce_sorted = false` で pack した場合の
///   並べ替え（`enforce_sorted = true` では PyTorch と同じく `None`）。`Some` のとき
///   互いに逆置換。
///
/// フィールドは非公開で、[`PackedSequence::new`] か pack 関数でしか構築できない
/// （不変条件を満たさない値を作らせない）。RNN 系（[`rnn_forward_packed`] など）が
/// 入力として受け取り、出力としても同じ型を返す。
#[derive(Debug, Clone)]
pub struct PackedSequence<'t> {
    data: Var<'t>,
    batch_sizes: Vec<usize>,
    sorted_indices: Option<Vec<usize>>,
    unsorted_indices: Option<Vec<usize>>,
}

impl<'t> PackedSequence<'t> {
    /// 不変条件を検査して構築する。`unsorted_indices` は `sorted_indices` の逆置換として
    /// 内部で導出する。RNN 出力の組み立てや、葉の `data` から unpack するテストで使う。
    ///
    /// 検査: `data` が rank 1 以上／`batch_sizes` が非空・各要素 ≥ 1・非増加／総和が
    /// `data.shape()[0]` に一致／`sorted_indices` が `0..batch_sizes[0]` の置換。
    pub fn new(
        data: Var<'t>,
        batch_sizes: Vec<usize>,
        sorted_indices: Option<Vec<usize>>,
    ) -> Result<Self, AutodiffError> {
        let shape = data.shape();
        if shape.is_empty() {
            return Err(invalid(
                "PackedSequence::new: data must have rank >= 1".to_string(),
            ));
        }
        let first = match batch_sizes.first() {
            Some(&f) => f,
            None => {
                return Err(invalid(
                    "PackedSequence::new: batch_sizes must not be empty".to_string(),
                ));
            }
        };
        if batch_sizes.contains(&0) {
            return Err(invalid(
                "PackedSequence::new: batch_sizes must be >= 1".to_string(),
            ));
        }
        if batch_sizes.windows(2).any(|w| w[0] < w[1]) {
            return Err(invalid(
                "PackedSequence::new: batch_sizes must be non-increasing".to_string(),
            ));
        }
        let total = batch_sizes
            .iter()
            .try_fold(0usize, |a, &b| a.checked_add(b))
            .ok_or_else(overflow)?;
        if total != shape[0] {
            return Err(invalid(format!(
                "PackedSequence::new: sum(batch_sizes) (={total}) must equal data.shape()[0] \
                 (={})",
                shape[0]
            )));
        }
        let unsorted_indices = match &sorted_indices {
            None => None,
            Some(s) => {
                if s.len() != first {
                    return Err(invalid(format!(
                        "PackedSequence::new: sorted_indices has length {} but batch size is \
                         {first}",
                        s.len()
                    )));
                }
                let mut seen: Vec<bool> = Vec::new();
                seen.try_reserve_exact(first).map_err(|_| overflow())?;
                seen.resize(first, false);
                for &i in s {
                    if i >= first || seen[i] {
                        return Err(invalid(
                            "PackedSequence::new: sorted_indices must be a permutation of \
                             0..batch_size"
                                .to_string(),
                        ));
                    }
                    seen[i] = true;
                }
                Some(invert_permutation(s)?)
            }
        };
        Ok(Self {
            data,
            batch_sizes,
            sorted_indices,
            unsorted_indices,
        })
    }

    /// 詰めたデータ `[N, *]`。
    pub fn data(&self) -> &Var<'t> {
        &self.data
    }

    /// 各 step の生存系列数。
    pub fn batch_sizes(&self) -> &[usize] {
        &self.batch_sizes
    }

    /// 長さ降順への並べ替え（`enforce_sorted = true` で pack した場合は `None`）。
    pub fn sorted_indices(&self) -> Option<&[usize]> {
        self.sorted_indices.as_deref()
    }

    /// `sorted_indices` の逆置換（元のバッチ順へ戻す添字）。
    pub fn unsorted_indices(&self) -> Option<&[usize]> {
        self.unsorted_indices.as_deref()
    }
}

// =====================================================================
// pack / unpack
// =====================================================================

/// パディング済みの `input`（`[T,B,*]`、`batch_first` なら `[B,T,*]`）を `lengths` に従って
/// 詰める（PyTorch `pack_padded_sequence` 相当）。
///
/// `lengths[i]` は元のバッチ順の系列 `i` の長さ（`1..=T`）。`enforce_sorted = true` なら
/// `lengths` は非増加でなければならず並べ替えは行わない（`sorted_indices` は `None`）。
/// `false` なら長さ降順の安定ソート（同長は元の添字昇順）で並べ替える。
///
/// 実装は `Var::contiguous`→`reshape`→`index_select(0, idx)`（`Op::Gather` 1 個）。
/// 引数起因のエラーでは tape にノードを積まない。
pub fn pack_padded_sequence<'t>(
    input: &Var<'t>,
    lengths: &[usize],
    batch_first: bool,
    enforce_sorted: bool,
) -> Result<PackedSequence<'t>, AutodiffError> {
    let shape = input.shape();
    if shape.len() < 2 {
        return Err(invalid(format!(
            "pack_padded_sequence: input must have rank >= 2 (got rank {})",
            shape.len()
        )));
    }
    let (t_len, b_len) = if batch_first {
        (shape[1], shape[0])
    } else {
        (shape[0], shape[1])
    };
    if b_len == 0 || lengths.len() != b_len {
        return Err(invalid(format!(
            "pack_padded_sequence: lengths has {} entries but batch size is {b_len} (batch \
             size must be >= 1)",
            lengths.len()
        )));
    }
    for (i, &l) in lengths.iter().enumerate() {
        if l == 0 || l > t_len {
            return Err(invalid(format!(
                "pack_padded_sequence: lengths[{i}] (={l}) must be in 1..={t_len}"
            )));
        }
    }
    if enforce_sorted && lengths.windows(2).any(|w| w[0] < w[1]) {
        return Err(invalid(
            "pack_padded_sequence: lengths must be sorted in non-increasing order when \
             enforce_sorted is true"
                .to_string(),
        ));
    }
    let total = lengths
        .iter()
        .try_fold(0usize, |a, &l| a.checked_add(l))
        .ok_or_else(overflow)?;
    let tb = t_len.checked_mul(b_len).ok_or_else(overflow)?;
    checked_axis_len_as_i32(tb)?;
    checked_index_alloc_len(total)?;

    let mut sorted: Vec<usize> = Vec::new();
    sorted.try_reserve_exact(b_len).map_err(|_| overflow())?;
    sorted.extend(0..b_len);
    if !enforce_sorted {
        // 安定ソート: 同長は元の添字昇順のまま。
        sorted.sort_by(|&a, &b| lengths[b].cmp(&lengths[a]));
    }
    let t_max = lengths[sorted[0]];
    let mut batch_sizes: Vec<usize> = Vec::new();
    batch_sizes
        .try_reserve_exact(t_max)
        .map_err(|_| overflow())?;
    for t in 0..t_max {
        batch_sizes.push(lengths.iter().filter(|&&l| l > t).count());
    }

    let mut idx: Vec<usize> = Vec::new();
    idx.try_reserve_exact(total).map_err(|_| overflow())?;
    for (t, &b) in batch_sizes.iter().enumerate() {
        for &s in &sorted[..b] {
            idx.push(if batch_first {
                s * t_len + t
            } else {
                t * b_len + s
            });
        }
    }
    let index = index_tensor(&idx)?;

    let mut flat_shape: Vec<usize> = Vec::with_capacity(shape.len() - 1);
    flat_shape.push(tb);
    flat_shape.extend_from_slice(&shape[2..]);
    let data = input
        .contiguous()?
        .reshape(&flat_shape)?
        .index_select(0, &index)?;

    let (sorted_indices, unsorted_indices) = if enforce_sorted {
        (None, None)
    } else {
        let inv = invert_permutation(&sorted)?;
        (Some(sorted), Some(inv))
    };
    Ok(PackedSequence {
        data,
        batch_sizes,
        sorted_indices,
        unsorted_indices,
    })
}

/// 詰めた系列を `[T_out,B,*]`（`batch_first` なら `[B,T_out,*]`）へ戻す（PyTorch
/// `pad_packed_sequence` 相当）。戻り値は `(padded, lengths)` で、`lengths` は元の
/// バッチ順の各系列の長さ。
///
/// 範囲外はすべて `padding_value`（`NaN`・`±inf` も可）で埋める。`total_length` が
/// `Some(L)` なら `T_out = L`（`L >= T_max` が必要）、`None` なら `T_out = T_max`。
///
/// 実装は `Var::cat([data, pad_row], 0)`→`index_select`→`reshape`（パディング位置は
/// `data` の末尾に連結した 1 行を指す）。パディング行は勾配を追跡しない。
pub fn pad_packed_sequence<'t>(
    sequence: &PackedSequence<'t>,
    batch_first: bool,
    padding_value: f32,
    total_length: Option<usize>,
) -> Result<(Var<'t>, Vec<usize>), AutodiffError> {
    let bs = &sequence.batch_sizes;
    let t_max = bs.len();
    let b_len = bs[0];
    let t_out = match total_length {
        Some(l) => {
            if l < t_max {
                return Err(invalid(format!(
                    "pad_packed_sequence: total_length (={l}) must be >= the longest sequence \
                     ({t_max})"
                )));
            }
            l
        }
        None => t_max,
    };
    let shape = sequence.data.shape();
    let n = shape[0];
    let n_ext = n.checked_add(1).ok_or_else(overflow)?;
    let out_rows = t_out.checked_mul(b_len).ok_or_else(overflow)?;
    checked_axis_len_as_i32(n_ext)?;
    checked_index_alloc_len(out_rows)?;

    // 各 step の data 内先頭オフセット。
    let mut offsets: Vec<usize> = Vec::new();
    offsets.try_reserve_exact(t_max).map_err(|_| overflow())?;
    let mut acc = 0usize;
    for &b in bs {
        offsets.push(acc);
        acc = acc.checked_add(b).ok_or_else(overflow)?;
    }

    let mut idx: Vec<usize> = Vec::new();
    idx.try_reserve_exact(out_rows).map_err(|_| overflow())?;
    let pad_pos = n;
    let (outer, inner) = if batch_first {
        (b_len, t_out)
    } else {
        (t_out, b_len)
    };
    for o in 0..outer {
        for i in 0..inner {
            let (t, b) = if batch_first { (i, o) } else { (o, i) };
            let j = match &sequence.unsorted_indices {
                Some(u) => u[b],
                None => b,
            };
            idx.push(if t < t_max && j < bs[t] {
                offsets[t] + j
            } else {
                pad_pos
            });
        }
    }
    let index = index_tensor(&idx)?;

    let mut lengths: Vec<usize> = Vec::new();
    lengths.try_reserve_exact(b_len).map_err(|_| overflow())?;
    lengths.resize(b_len, 0);
    for &b in bs {
        for l in lengths.iter_mut().take(b) {
            *l += 1;
        }
    }
    let lengths_orig = match &sequence.unsorted_indices {
        Some(u) => u.iter().map(|&j| lengths[j]).collect(),
        None => lengths,
    };

    let mut row_shape: Vec<usize> = Vec::with_capacity(shape.len());
    row_shape.push(1);
    row_shape.extend_from_slice(&shape[1..]);
    let tape = sequence.data.tape();
    let pad_row = tape.var_no_grad(&Tensor::full(&row_shape, padding_value)?);
    let ext = Var::cat(&[sequence.data, pad_row], 0)?;
    let mut out_shape: Vec<usize> = Vec::with_capacity(shape.len() + 1);
    out_shape.push(outer);
    out_shape.push(inner);
    out_shape.extend_from_slice(&shape[1..]);
    let padded = ext.index_select(0, &index)?.reshape(&out_shape)?;
    Ok((padded, lengths_orig))
}

// =====================================================================
// RNN 系の packed 実行（共通エンジン）
// =====================================================================

/// 1 ステップのセル関数（状態 `Vec` の先頭が出力 `h`、LSTM は `[h, c]`）。
type StepFn<'a, 't> = dyn Fn(&Var<'t>, &[Var<'t>]) -> Result<Vec<Var<'t>>, AutodiffError> + 'a;

/// 1 方向ぶんの packed 再帰を回す。`init` は sorted 順の `[B,H]`（状態ごと）。
/// 戻り値は `(出力 [N,H], 最終状態（状態ごと [B,H]・sorted 順）)`。
///
/// 順方向は PyTorch の可変長再帰と同じく、系列が終了した行を `narrow` で切り出して最終状態
/// として保存する。逆方向は各系列が自分の最終 step から始まるよう、系列が合流する step で
/// 初期状態の該当行を `cat` で連結する（padded 実行と結果が変わる点）。
fn run_direction<'t>(
    step: &StepFn<'_, 't>,
    data: &Var<'t>,
    bs: &[usize],
    init: &[Var<'t>],
    reverse: bool,
) -> Result<(Var<'t>, Vec<Var<'t>>), AutodiffError> {
    let n_state = init.len();
    let total = bs
        .iter()
        .try_fold(0usize, |a, &b| a.checked_add(b))
        .ok_or_else(overflow)?;
    let mut outs: Vec<Var<'t>> = reserve_outputs(bs.len())?;

    if !reverse {
        let mut states: Vec<Var<'t>> = init.to_vec();
        let mut finished: Vec<Vec<Var<'t>>> = (0..n_state).map(|_| Vec::new()).collect();
        let mut offset = 0usize;
        let mut last_b = bs[0];
        for &b in bs {
            if b < last_b {
                for s in 0..n_state {
                    finished[s].push(states[s].narrow(0, b, last_b - b)?);
                    states[s] = states[s].narrow(0, 0, b)?;
                }
            }
            last_b = b;
            let x_t = data.narrow(0, offset, b)?;
            offset = offset.checked_add(b).ok_or_else(overflow)?;
            states = step(&x_t, &states)?;
            outs.push(states[0]);
        }
        let mut finals = Vec::with_capacity(n_state);
        for s in 0..n_state {
            finished[s].push(states[s]);
            finished[s].reverse();
            finals.push(Var::cat(&finished[s], 0)?);
        }
        Ok((Var::cat(&outs, 0)?, finals))
    } else {
        let t_max = bs.len();
        let mut last_b = bs[t_max - 1];
        let mut states: Vec<Var<'t>> = Vec::with_capacity(n_state);
        for v in init {
            states.push(v.narrow(0, 0, last_b)?);
        }
        let mut offset = total;
        for &b in bs.iter().rev() {
            if b > last_b {
                for s in 0..n_state {
                    let extra = init[s].narrow(0, last_b, b - last_b)?;
                    states[s] = Var::cat(&[states[s], extra], 0)?;
                }
            }
            last_b = b;
            offset = offset.checked_sub(b).ok_or_else(overflow)?;
            let x_t = data.narrow(0, offset, b)?;
            states = step(&x_t, &states)?;
            outs.push(states[0]);
        }
        outs.reverse();
        Ok((Var::cat(&outs, 0)?, states))
    }
}

/// セルのパラメータ束 `P` を取る 1 ステップ関数（[`run_packed`] へ渡す）。
type PStepFn<'a, 't, P> =
    dyn Fn(&P, &Var<'t>, &[Var<'t>]) -> Result<Vec<Var<'t>>, AutodiffError> + 'a;

/// 共通エンジンへ渡す層構成。
struct StackSpec {
    num_layers: usize,
    num_directions: usize,
    dropout: f32,
    training: bool,
    input_size: usize,
    hidden: usize,
}

/// エンジンの戻り値。`finals[s][k]` は状態 `s`・セル `k = layer*dirs + dir` の最終状態
/// （元のバッチ順 `[B,H]`）。
struct PackedRun<'t, P> {
    output: PackedSequence<'t>,
    finals: Vec<Vec<Var<'t>>>,
    params: Vec<P>,
}

/// 6 本の RNN 関数が共有する packed 実行エンジン。`bind(layer, dir)` はセルの
/// `bind(tape)`（BPTT の重み共有のため `(layer, dir)` ごとに 1 回）、`step` は 1 セル
/// ステップ。入口検査（入力幅・状態の shape・同一 tape）はすべて `bind` とゼロ状態の
/// 確保より前に行う。
fn run_packed<'t, P>(
    op_name: &str,
    input: &PackedSequence<'t>,
    spec: &StackSpec,
    init: &[Option<&[Var<'t>]>],
    bind: &dyn Fn(usize, usize) -> Result<P, AutodiffError>,
    step: &PStepFn<'_, 't, P>,
) -> Result<PackedRun<'t, P>, AutodiffError> {
    let data = &input.data;
    let shape = data.shape();
    if shape.len() != 2 || shape[1] != spec.input_size {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: shape,
            rhs: vec![input.batch_sizes.iter().sum(), spec.input_size],
        }));
    }
    let b_len = input.batch_sizes[0];
    let total_cells = spec
        .num_layers
        .checked_mul(spec.num_directions)
        .ok_or_else(overflow)?;
    for slot in init.iter().flatten() {
        if slot.len() != total_cells {
            return Err(invalid(format!(
                "{op_name}: initial state has length {} but expected {total_cells} \
                 (num_layers * num_directions)",
                slot.len()
            )));
        }
        for v in slot.iter() {
            data.check_same_tape(v)?;
            if v.shape() != [b_len, spec.hidden] {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: v.shape(),
                    rhs: vec![b_len, spec.hidden],
                }));
            }
        }
    }
    let tape = data.tape();
    let sorted = input.sorted_indices.as_deref();
    let mut init_sorted: Vec<Vec<Var<'t>>> = Vec::with_capacity(init.len());
    for slot in init {
        let mut per_cell: Vec<Var<'t>> = reserve_outputs(total_cells)?;
        for k in 0..total_cells {
            let v = match slot {
                Some(s) => reorder_rows(&s[k], sorted)?,
                None => tape.var(&Tensor::zeros(&[b_len, spec.hidden])?),
            };
            per_cell.push(v);
        }
        init_sorted.push(per_cell);
    }

    let mut layer_in = *data;
    let mut finals: Vec<Vec<Var<'t>>> = (0..init.len())
        .map(|_| Vec::with_capacity(total_cells))
        .collect();
    let mut params: Vec<P> = reserve_outputs(total_cells)?;
    for layer in 0..spec.num_layers {
        let mut dir_outs: Vec<Var<'t>> = Vec::with_capacity(spec.num_directions);
        for dir in 0..spec.num_directions {
            let k = layer * spec.num_directions + dir;
            let p = bind(layer, dir)?;
            let init_k: Vec<Var<'t>> = init_sorted.iter().map(|s| s[k]).collect();
            let (out, fin) = run_direction(
                &|x, st| step(&p, x, st),
                &layer_in,
                &input.batch_sizes,
                &init_k,
                dir == 1,
            )?;
            dir_outs.push(out);
            for (s, f) in fin.into_iter().enumerate() {
                finals[s].push(f);
            }
            params.push(p);
        }
        let combined = if spec.num_directions == 1 {
            dir_outs[0]
        } else {
            Var::cat(&[dir_outs[0], dir_outs[1]], 1)?
        };
        layer_in = if layer + 1 < spec.num_layers {
            combined.dropout(spec.dropout, spec.training)?
        } else {
            combined
        };
    }

    let unsorted = input.unsorted_indices.as_deref();
    for per_state in &mut finals {
        for v in per_state.iter_mut() {
            *v = reorder_rows(v, unsorted)?;
        }
    }
    let output = PackedSequence {
        data: layer_in,
        batch_sizes: input.batch_sizes.clone(),
        sorted_indices: input.sorted_indices.clone(),
        unsorted_indices: input.unsorted_indices.clone(),
    };
    Ok(PackedRun {
        output,
        finals,
        params,
    })
}

fn internal(op_name: &str) -> AutodiffError {
    invalid(format!("{op_name}: internal error: missing cell or state"))
}

fn single_spec(input_size: usize, hidden: usize) -> StackSpec {
    StackSpec {
        num_layers: 1,
        num_directions: 1,
        dropout: 0.0,
        training: false,
        input_size,
        hidden,
    }
}

// =====================================================================
// 出力型
// =====================================================================

/// [`rnn_forward_packed`]／[`gru_forward_packed`] の戻り値。`output` は入力と同じ
/// `batch_sizes`・並べ替えを持つ `[N,H]` の packed 出力、`h_n` は元のバッチ順の最終隠れ
/// 状態 `[B,H]`、`params` は呼び出しで実際に使われたテープ登録済みパラメータ。
#[derive(Debug)]
#[non_exhaustive]
pub struct PackedRnnSeqOutput<'t, P> {
    pub output: PackedSequence<'t>,
    pub h_n: Var<'t>,
    pub params: P,
}

/// [`lstm_forward_packed`] の戻り値（`c_n` を加えた LSTM 版）。
#[derive(Debug)]
#[non_exhaustive]
pub struct PackedLstmSeqOutput<'t> {
    pub output: PackedSequence<'t>,
    pub h_n: Var<'t>,
    pub c_n: Var<'t>,
    pub params: LstmCellVars<'t>,
}

/// [`stacked_rnn_forward_packed`]／[`stacked_gru_forward_packed`] の戻り値。`h_n`・
/// `params` の並びは `index = layer * num_directions + direction`。
#[derive(Debug)]
#[non_exhaustive]
pub struct StackedPackedRnnSeqOutput<'t, P> {
    pub output: PackedSequence<'t>,
    pub h_n: Vec<Var<'t>>,
    pub params: Vec<P>,
}

/// [`stacked_lstm_forward_packed`] の戻り値。
#[derive(Debug)]
#[non_exhaustive]
pub struct StackedPackedLstmSeqOutput<'t> {
    pub output: PackedSequence<'t>,
    pub h_n: Vec<Var<'t>>,
    pub c_n: Vec<Var<'t>>,
    pub params: Vec<LstmCellVars<'t>>,
}

// =====================================================================
// 単層・単方向
// =====================================================================

/// 単層 RNN（tanh）の packed 実行。`h0` は元のバッチ順の `[B,H]`（省略時はゼロ）。
pub fn rnn_forward_packed<'t>(
    rnn: &Rnn,
    input: &PackedSequence<'t>,
    h0: Option<&Var<'t>>,
) -> Result<PackedRnnSeqOutput<'t, RnnCellVars<'t>>, AutodiffError> {
    const OP: &str = "rnn_forward_packed";
    let cell = rnn.cell();
    let spec = single_spec(cell.input_size(), cell.hidden_size());
    let tape = input.data.tape();
    let h0_slice = h0.map(std::slice::from_ref);
    let run = run_packed(
        OP,
        input,
        &spec,
        &[h0_slice],
        &|_, _| Ok(cell.bind(tape)),
        &|p, x, st| Ok(vec![p.forward(x, &st[0])?]),
    )?;
    let h_n = run
        .finals
        .into_iter()
        .next()
        .and_then(|v| v.into_iter().next())
        .ok_or_else(|| internal(OP))?;
    let params = run.params.into_iter().next().ok_or_else(|| internal(OP))?;
    Ok(PackedRnnSeqOutput {
        output: run.output,
        h_n,
        params,
    })
}

/// 単層 GRU の packed 実行（[`rnn_forward_packed`] と同じ契約）。
pub fn gru_forward_packed<'t>(
    gru: &Gru,
    input: &PackedSequence<'t>,
    h0: Option<&Var<'t>>,
) -> Result<PackedRnnSeqOutput<'t, GruCellVars<'t>>, AutodiffError> {
    const OP: &str = "gru_forward_packed";
    let cell = gru.cell();
    let spec = single_spec(cell.input_size(), cell.hidden_size());
    let tape = input.data.tape();
    let h0_slice = h0.map(std::slice::from_ref);
    let run = run_packed(
        OP,
        input,
        &spec,
        &[h0_slice],
        &|_, _| Ok(cell.bind(tape)),
        &|p, x, st| Ok(vec![p.forward(x, &st[0])?]),
    )?;
    let h_n = run
        .finals
        .into_iter()
        .next()
        .and_then(|v| v.into_iter().next())
        .ok_or_else(|| internal(OP))?;
    let params = run.params.into_iter().next().ok_or_else(|| internal(OP))?;
    Ok(PackedRnnSeqOutput {
        output: run.output,
        h_n,
        params,
    })
}

/// 単層 LSTM の packed 実行。`h0`／`c0` は元のバッチ順の `[B,H]`（省略時はゼロ）。
pub fn lstm_forward_packed<'t>(
    lstm: &Lstm,
    input: &PackedSequence<'t>,
    h0: Option<&Var<'t>>,
    c0: Option<&Var<'t>>,
) -> Result<PackedLstmSeqOutput<'t>, AutodiffError> {
    const OP: &str = "lstm_forward_packed";
    let cell = lstm.cell();
    let spec = single_spec(cell.input_size(), cell.hidden_size());
    let tape = input.data.tape();
    let run = run_packed(
        OP,
        input,
        &spec,
        &[h0.map(std::slice::from_ref), c0.map(std::slice::from_ref)],
        &|_, _| Ok(cell.bind(tape)),
        &|p, x, st| {
            let (h, c) = p.forward(x, &st[0], &st[1])?;
            Ok(vec![h, c])
        },
    )?;
    let mut it = run.finals.into_iter();
    let h_n = it
        .next()
        .and_then(|v| v.into_iter().next())
        .ok_or_else(|| internal(OP))?;
    let c_n = it
        .next()
        .and_then(|v| v.into_iter().next())
        .ok_or_else(|| internal(OP))?;
    let params = run.params.into_iter().next().ok_or_else(|| internal(OP))?;
    Ok(PackedLstmSeqOutput {
        output: run.output,
        h_n,
        c_n,
        params,
    })
}

// =====================================================================
// 多層・双方向
// =====================================================================

macro_rules! stacked_spec {
    ($model:expr) => {{
        let cfg = $model.config();
        StackSpec {
            num_layers: cfg.num_layers(),
            num_directions: cfg.num_directions(),
            dropout: cfg.dropout(),
            training: Module::training($model),
            input_size: $model.input_size(),
            hidden: $model.hidden_size(),
        }
    }};
}

/// 多層・双方向 RNN の packed 実行。`h0` は長さ `num_layers * num_directions` の
/// 元のバッチ順 `[B,H]`（省略時はゼロ）。層間 dropout は packed の `data` へ層ごとに
/// 1 回適用する（padded 版と RNG 消費順が異なる。モジュール doc 参照）。
pub fn stacked_rnn_forward_packed<'t>(
    rnn: &StackedRnn,
    input: &PackedSequence<'t>,
    h0: Option<&[Var<'t>]>,
) -> Result<StackedPackedRnnSeqOutput<'t, RnnCellVars<'t>>, AutodiffError> {
    const OP: &str = "stacked_rnn_forward_packed";
    let spec = stacked_spec!(rnn);
    let tape = input.data.tape();
    let run = run_packed(
        OP,
        input,
        &spec,
        &[h0],
        &|l, d| {
            rnn.cell(l, d)
                .map(|c| c.bind(tape))
                .ok_or_else(|| internal(OP))
        },
        &|p, x, st| Ok(vec![p.forward(x, &st[0])?]),
    )?;
    let h_n = run.finals.into_iter().next().ok_or_else(|| internal(OP))?;
    Ok(StackedPackedRnnSeqOutput {
        output: run.output,
        h_n,
        params: run.params,
    })
}

/// 多層・双方向 GRU の packed 実行（[`stacked_rnn_forward_packed`] と同じ契約）。
pub fn stacked_gru_forward_packed<'t>(
    gru: &StackedGru,
    input: &PackedSequence<'t>,
    h0: Option<&[Var<'t>]>,
) -> Result<StackedPackedRnnSeqOutput<'t, GruCellVars<'t>>, AutodiffError> {
    const OP: &str = "stacked_gru_forward_packed";
    let spec = stacked_spec!(gru);
    let tape = input.data.tape();
    let run = run_packed(
        OP,
        input,
        &spec,
        &[h0],
        &|l, d| {
            gru.cell(l, d)
                .map(|c| c.bind(tape))
                .ok_or_else(|| internal(OP))
        },
        &|p, x, st| Ok(vec![p.forward(x, &st[0])?]),
    )?;
    let h_n = run.finals.into_iter().next().ok_or_else(|| internal(OP))?;
    Ok(StackedPackedRnnSeqOutput {
        output: run.output,
        h_n,
        params: run.params,
    })
}

/// 多層・双方向 LSTM の packed 実行。`h0`／`c0` は長さ `num_layers * num_directions`。
pub fn stacked_lstm_forward_packed<'t>(
    lstm: &StackedLstm,
    input: &PackedSequence<'t>,
    h0: Option<&[Var<'t>]>,
    c0: Option<&[Var<'t>]>,
) -> Result<StackedPackedLstmSeqOutput<'t>, AutodiffError> {
    const OP: &str = "stacked_lstm_forward_packed";
    let spec = stacked_spec!(lstm);
    let tape = input.data.tape();
    let run = run_packed(
        OP,
        input,
        &spec,
        &[h0, c0],
        &|l, d| {
            lstm.cell(l, d)
                .map(|c| c.bind(tape))
                .ok_or_else(|| internal(OP))
        },
        &|p, x, st| {
            let (h, c) = p.forward(x, &st[0], &st[1])?;
            Ok(vec![h, c])
        },
    )?;
    let mut it = run.finals.into_iter();
    let h_n = it.next().ok_or_else(|| internal(OP))?;
    let c_n = it.next().ok_or_else(|| internal(OP))?;
    Ok(StackedPackedLstmSeqOutput {
        output: run.output,
        h_n,
        c_n,
        params: run.params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn leaf<'t>(tape: &'t Tape, data: Vec<f32>, shape: &[usize]) -> Var<'t> {
        tape.var(&Tensor::new(data, shape).expect("test: shape/len"))
    }

    fn values(v: &Var<'_>) -> Vec<f32> {
        v.to_tensor().host_slice().into_owned()
    }

    /// `[T=3, B=2]` time-major。lengths = [2, 3]（未整列）。
    #[test]
    fn pack_sorts_stably_and_derives_batch_sizes() {
        let tape = Tape::new();
        // x[t][b] = 10*t + b
        let x = leaf(&tape, vec![0., 1., 10., 11., 20., 21.], &[3, 2]);
        let p = pack_padded_sequence(&x, &[2, 3], false, false).expect("pack");
        assert_eq!(p.batch_sizes(), &[2, 2, 1]);
        assert_eq!(p.sorted_indices(), Some(&[1usize, 0][..]));
        assert_eq!(p.unsorted_indices(), Some(&[1usize, 0][..]));
        // sorted: b=1 (len3), b=0 (len2)
        assert_eq!(values(p.data()), vec![1., 0., 11., 10., 21.]);
    }

    #[test]
    fn pack_ties_keep_original_order() {
        let tape = Tape::new();
        let x = leaf(&tape, vec![0., 1., 2., 10., 11., 12.], &[2, 3]);
        let p = pack_padded_sequence(&x, &[2, 2, 2], false, false).expect("pack");
        assert_eq!(p.sorted_indices(), Some(&[0usize, 1, 2][..]));
        assert_eq!(p.batch_sizes(), &[3, 3]);
    }

    #[test]
    fn pack_enforce_sorted_has_no_indices() {
        let tape = Tape::new();
        let x = leaf(&tape, vec![0., 1., 10., 11.], &[2, 2]);
        let p = pack_padded_sequence(&x, &[2, 1], false, true).expect("pack");
        assert!(p.sorted_indices().is_none() && p.unsorted_indices().is_none());
        assert_eq!(values(p.data()), vec![0., 1., 10.]);
    }

    #[test]
    fn pack_batch_first_matches_time_major() {
        let tape = Tape::new();
        let tm = leaf(&tape, vec![0., 1., 10., 11., 20., 21.], &[3, 2]);
        let bf = leaf(&tape, vec![0., 10., 20., 1., 11., 21.], &[2, 3]);
        let a = pack_padded_sequence(&tm, &[3, 2], false, true).expect("tm");
        let b = pack_padded_sequence(&bf, &[3, 2], true, true).expect("bf");
        assert_eq!(values(a.data()), values(b.data()));
        assert_eq!(a.batch_sizes(), b.batch_sizes());
    }

    #[test]
    fn round_trip_restores_input_with_padding() {
        let tape = Tape::new();
        let x = leaf(&tape, vec![0., 1., 10., 11., 20., 21.], &[3, 2]);
        let p = pack_padded_sequence(&x, &[2, 3], false, false).expect("pack");
        let (padded, lens) = pad_packed_sequence(&p, false, -1.0, None).expect("unpack");
        assert_eq!(lens, vec![2, 3]);
        assert_eq!(values(&padded), vec![0., 1., 10., 11., -1., 21.]);
        let (longer, _) = pad_packed_sequence(&p, false, 0.0, Some(4)).expect("total_length");
        assert_eq!(longer.to_tensor().shape(), &[4, 2]);
    }

    #[test]
    fn errors_do_not_push_nodes() {
        let tape = Tape::new();
        let x = leaf(&tape, vec![0.; 6], &[3, 2]);
        let before = tape.len();
        assert!(pack_padded_sequence(&x, &[0, 3], false, false).is_err());
        assert!(pack_padded_sequence(&x, &[4, 3], false, false).is_err());
        assert!(pack_padded_sequence(&x, &[3], false, false).is_err());
        assert!(pack_padded_sequence(&x, &[2, 3], false, true).is_err());
        assert_eq!(tape.len(), before);
        let p = pack_padded_sequence(&x, &[3, 2], false, true).expect("pack");
        let before = tape.len();
        assert!(pad_packed_sequence(&p, false, 0.0, Some(2)).is_err());
        assert_eq!(tape.len(), before);
    }

    #[test]
    fn new_rejects_invalid_invariants() {
        let tape = Tape::new();
        let d = leaf(&tape, vec![0.; 3], &[3]);
        assert!(PackedSequence::new(d, vec![], None).is_err());
        assert!(PackedSequence::new(d, vec![1, 2], None).is_err());
        assert!(PackedSequence::new(d, vec![2, 0], None).is_err());
        assert!(PackedSequence::new(d, vec![1, 1], None).is_err());
        assert!(PackedSequence::new(d, vec![2, 1], Some(vec![0, 0])).is_err());
        assert!(PackedSequence::new(d, vec![2, 1], Some(vec![0])).is_err());
        let ok = PackedSequence::new(d, vec![2, 1], Some(vec![1, 0])).expect("valid");
        assert_eq!(ok.unsorted_indices(), Some(&[1usize, 0][..]));
    }

    #[test]
    fn pack_gradient_routes_only_to_valid_positions() {
        let tape = Tape::new();
        let x = leaf(&tape, vec![0., 1., 10., 11.], &[2, 2]);
        let p = pack_padded_sequence(&x, &[2, 1], false, true).expect("pack");
        let loss = p.data().sum(None).expect("sum");
        let g = tape.backward(&loss).expect("backward");
        let gx = g.get(&x).expect("get").expect("some");
        assert_eq!(gx.host_slice().into_owned(), vec![1., 1., 1., 0.]);
    }
}
