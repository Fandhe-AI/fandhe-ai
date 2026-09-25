//! `repeat`・`tile`・`flip`・`roll` の 4 種形状演算（イシュー #2143・親
//! #2131「5-B 演算」）。
//!
//! **新規 `Op` はゼロ（受け入れ条件）**: いずれも既存の `Var::index_select`
//! （実体は `Var::gather` → `Op::Gather`）・`Var::broadcast_to`
//! （`Op::BroadcastTo`）・`Var::reshape`（`Op::Reshape`。`repeat` の空
//! テンソル最終化のみで使用。PR #2256 codex-review／cursor-review 指摘
//! 対応）の合成のみで構成する。いずれも CPU・CUDA・Metal の全バック
//! エンドに経路があり（`gather` は既定 `Unsupported` でホスト参照実装へ
//! フォールバック）、専用カーネルなしで到達可能。
//!
//! **facade 非公開（意図的）**: `crates/autodiff/src/bool_ops.rs`
//! モジュール doc と同じ理由・同じ判断枠組みによる。`Var` は facade
//! （`fandhe_ai` クレート）から直接再エクスポートされるため、`Var` への
//! inherent メソッド追加は即座に facade 公開面へ出てしまう。イシュー
//! #2143 本文は facade 公開面を承認事項として明示し、親 #2131 はこの
//! ツリーに限り「設計判断記録 → 承認 → 実装」の 2 段階を定めるため、
//! 承認が取れるまでは自由関数として `Var` の外に置き到達不能にする
//! （`docs/autodiff-rearrange-ops-decision.md` §2.1）。承認後は
//! `Var::repeat` 等の薄い委譲メソッドを追加し、facade 側の保留ガード
//! （`crates/facade/src/lib.rs::VarRearrangeOpsHoldDoctestGuard`）を
//! 撤去する。
//!
//! **数値契約**: forward は値のコピーのみ（算術を含まない）ため 3
//! バックエンド間で構造的に bit 完全一致する（`NaN` の payload も保存
//! される）。backward（`Op::Gather` の VJP。`grad.rs`）は scatter-add で
//! 行う。`flip`／`roll` は各入力要素への寄与が常に 1 つのため
//! `0 + g` の単純代入になり bit 一致する。`repeat`／`tile` は `r` 個の
//! コピーの勾配を合算するため、GPU の scatter 加算順序次第では REQ-2
//! の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で
//! 比較する（`.claude/rules/coding-rust.md`）。
//!
//! **PyTorch との差分（doc に明記。承認後の facade 版でも据え置く
//! 予定）**:
//! - `roll` の `dims: None`（flatten してから roll する形）は非対応。
//!   `Var::reshape` の contiguous 制約（`ShapeError::
//!   NonContiguousReshape`）を回避する設計上の理由による
//!   （`Var::expand` の「-1 非対応」注記と同じ扱い）。
//! - 軸番号は非負のみ（PyTorch の負軸表記は非対応。`Var::squeeze`／
//!   `unsqueeze` 等、本クレートの他の形状演算と同じ規約）。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: 外部から渡る
//! `repeats`／`reps`／`dims`／`shifts` はバックエンドを呼ぶ前にすべて
//! 検査する（長さ・軸範囲・重複・`checked_mul` によるオーバーフロー・
//! `i32` 範囲）。巨大な繰り返し数による添字ベクタの過大確保は、確保前に
//! 出力要素数を検査して拒否する（本番経路 panic 禁止規約）。この検査は
//! `usize` オーバーフローの有無だけでなく、`checked_index_alloc_len` の
//! 実用上の確保バイト数上限（`MAX_INDEX_ALLOC_BYTES`。既定 1 GiB）も
//! 含む——`isize::MAX` 検査のみでは技術的にオーバーフローしない範囲の
//! 巨大値（例: shape `[1]` に `repeats=[1_000_000_000]` で 4GB）を
//! 確保前に拒否できず、`collect()` の実確保失敗による abort を招き
//! うるため（codex-review 指摘・PR #2256）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::var::Var;

/// 軸長 `n` を添字 `Tensor<i32>` の要素値として使える範囲（`0..=
/// i32::MAX`）に収まるか検査する。`Var::gather`（`index_select` の
/// 委譲先）は添字値を `i32` として受け取るため、軸長がこれを超える
/// 場合は添字を作る前に拒否する（`.claude/rules/coding-rust.md` 本番
/// 経路 panic 禁止・REQ-8 境界検査の趣旨）。
///
/// `pub(crate)`: イシュー #2144（親 #2131）の `crate::matrix_ops`
/// （`diag`／`tril`／`triu` の添字生成）も同じ検査を必要とするため、
/// クレート内共有ヘルパーへ昇格した（可視性のみの変更・本モジュール内
/// の挙動は不変。実装計画「設計判断」§2.4 参照）。
pub(crate) fn checked_axis_len_as_i32(n: usize) -> Result<(), AutodiffError> {
    i32::try_from(n)
        .map(|_| ())
        .map_err(|_| AutodiffError::Shape(ShapeError::ElementCountOverflow))
}

/// 添字ベクタ 1 本あたりに許容する確保バイト数の実用上限（codex-review
/// 指摘・PR #2256。`scripts/bench/oss-gemm-compare/src/main.rs::
/// MAX_BUFFER_BYTES`〈コミット 6c653324〉と同じ考え方の踏襲——本クレート
/// 内・tensor-core 側にも「実用上確保不能なサイズを拒否する」既存の
/// 汎用定数はなく〈`checked_numel_for`／`bool_ops::checked_bytes_for`
/// はいずれも `isize::MAX` バイトのみを検査し、本関数の従来実装と同じ
/// 弱点を持つ〉、本モジュール独自に新設する。`isize::MAX`（64bit では
/// 約 8 EiB）は `Vec` の allocation 契約上の上限でしかなく、実際に確保
/// 可能かどうかとは無関係のため、`checked_mul` の overflow 検査だけでは
/// shape `[1]` に `repeats=[1_000_000_000]`（4GB）のような実用上確保
/// 不能なサイズを検査（本モジュール doc「境界検査」節）で拒否できない
/// （`collect()` が実際に確保を試み、失敗すれば `handle_alloc_error` で
/// プロセスが abort し得る）。1 GiB を暫定値として採用する——将来
/// 正当な理由があれば見直してよい（ユーザー承認不要。ガードレール閾値・
/// テスト許容誤差〈`.claude/rules/security.md`〉には該当しない）。
const MAX_INDEX_ALLOC_BYTES: usize = 1 << 30;

/// 添字ベクタ `Vec<i32>`（長さ `len`）を確保する前のサイズ検査
/// （`crate::bool_ops::checked_bytes_for` と同型の独立複製。同じ理由
/// による複製——モジュールをまたいで `pub(crate)` 化するほどの共有価値
/// がなく、検査対象の型が固定〈`i32`〉のため専用化した）。要素数積の
/// `usize` オーバーフローに加え、`MAX_INDEX_ALLOC_BYTES`（実用上の
/// 確保上限）に収まるかも検査する。`repeat`／`tile` の巨大 `repeats`
/// だけでなく `flip`／`roll` も `broadcast_to` の stride-0 view（`n` が
/// 実体を伴わず巨大になりうる）経由で同じ添字巨大化を起こせるため、
/// 呼び出し元を問わずこの関数を唯一の確保前チェックポイントとする。
///
/// `pub(crate)`: `crate::matrix_ops`（マスク・添字の確保前サイズ検査）
/// も同じ上限を共有する（`checked_axis_len_as_i32` と同じ昇格理由）。
pub(crate) fn checked_index_alloc_len(len: usize) -> Result<(), AutodiffError> {
    let bytes = len
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(ShapeError::ElementCountOverflow)
        .map_err(AutodiffError::Shape)?;
    if bytes > MAX_INDEX_ALLOC_BYTES {
        return Err(AutodiffError::Shape(ShapeError::ElementCountOverflow));
    }
    Ok(())
}

/// 軸 `d` が `rank` 範囲内かを検査する（`Var::squeeze`／`gather` 等
/// 既存の形状演算と同じ `ShapeError::AxisOutOfRange` 契約）。
fn checked_axis_in_range(d: usize, rank: usize) -> Result<(), AutodiffError> {
    if d >= rank {
        return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            axis: d,
            rank,
        }));
    }
    Ok(())
}

/// 指定軸を反転する（`torch.flip` 相当。イシュー #2143）。
///
/// `dims` は 0 個以上・重複不可（PyTorch と同じ。重複は
/// `ShapeError::DuplicateAxis`）。`dims` が空なら新しいノードを積まず
/// `x` をそのまま返す（`Var::dropout` の早期リターンと同型）。
///
/// 実装は各軸へ逆順添字（`[n-1, …, 0]`）の [`Var::index_select`] を
/// 順に適用する合成。`n == 0` の軸は添字長 0 になり自然に空テンソルを
/// 保つ（除算を伴わないため特別扱い不要）。
pub fn flip<'t>(x: &Var<'t>, dims: &[usize]) -> Result<Var<'t>, AutodiffError> {
    if dims.is_empty() {
        return Ok(*x);
    }
    let rank = x.shape().len();
    let mut seen = std::collections::HashSet::with_capacity(dims.len());
    for &d in dims {
        checked_axis_in_range(d, rank)?;
        if !seen.insert(d) {
            return Err(AutodiffError::Shape(ShapeError::DuplicateAxis { axis: d }));
        }
    }

    let mut cur = *x;
    for &d in dims {
        let n = cur.shape()[d];
        checked_axis_len_as_i32(n)?;
        checked_index_alloc_len(n)?;
        let idx_data: Vec<i32> = (0..n).rev().map(|i| i as i32).collect();
        let idx = Tensor::new(idx_data, &[n]).map_err(AutodiffError::Shape)?;
        cur = cur.index_select(d, &idx)?;
    }
    Ok(cur)
}

/// 指定軸を循環シフトする（`torch.roll` 相当。イシュー #2143）。
///
/// `shifts.len() == dims.len()` かつ 1 個以上を必須とする（`torch.roll`
/// の `dims=None`〈flatten してから roll する形〉は非対応。モジュール
/// doc「PyTorch との差分」参照）。`dims` の重複は PyTorch と同じく
/// 許容し、指定順に逐次適用する。
///
/// 軸ごとに `s = shift.rem_euclid(n)` を正規化し、`n == 0` または
/// `s == 0` の軸は添字を作らずスキップする（no-op）。それ以外は添字
/// `idx[j] = (j + n - s) % n` で [`Var::index_select`] する。
pub fn roll<'t>(x: &Var<'t>, shifts: &[isize], dims: &[usize]) -> Result<Var<'t>, AutodiffError> {
    if shifts.is_empty() || shifts.len() != dims.len() {
        return Err(AutodiffError::InvalidArgument(format!(
            "rearrange_ops::roll: shifts と dims は同じ長さ・1 個以上が\
             必要（shifts.len()={}, dims.len()={}）",
            shifts.len(),
            dims.len()
        )));
    }
    let rank = x.shape().len();
    for &d in dims {
        checked_axis_in_range(d, rank)?;
    }

    let mut cur = *x;
    for (&d, &shift) in dims.iter().zip(shifts) {
        let n = cur.shape()[d];
        if n == 0 {
            continue;
        }
        let n_isize = isize::try_from(n)
            .map_err(|_| AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
        let s = shift.rem_euclid(n_isize) as usize;
        if s == 0 {
            continue;
        }
        checked_axis_len_as_i32(n)?;
        checked_index_alloc_len(n)?;
        let idx_data: Vec<i32> = (0..n).map(|j| ((j + n - s) % n) as i32).collect();
        let idx = Tensor::new(idx_data, &[n]).map_err(AutodiffError::Shape)?;
        cur = cur.index_select(d, &idx)?;
    }
    Ok(cur)
}

/// 軸ごとに繰り返す（`torch.Tensor.repeat` 相当。イシュー #2143）。
///
/// `repeats.len() < rank` は `AutodiffError::InvalidArgument`（PyTorch と
/// 同じ）。`repeats.len() > rank` の場合は [`Var::broadcast_to`] で先頭に
/// 長さ 1 の軸を追加してから rank を揃える（view のため元が非 contiguous
/// でもよい）。
///
/// 軸ごとの最終要素数 `axis_total = n * r`（および全軸積
/// `total_out_elems`）をどの軸も確保する前に `checked_mul` で確定させ
/// てから分岐する（PR #2256 codex-review／cursor-review 指摘。詳細は
/// 実装内コメント）: 全軸積が 0（いずれかの軸で `r == 0` または
/// `n == 0`）なら最終出力は空テンソルであり、他の軸の `r` が大きくても
/// 実際に巨大な添字ベクタは確保せず [`Var::reshape`] で最終 shape へ
/// 一括変換する。それ以外は軸ごとに `r == 1` ならスキップし（確保上限
/// チェックも対象外）、それ以外は添字 `idx = (0..n*r).map(|j| j % n)`
/// で [`Var::index_select`] する。全軸が `r == 1` で rank も変わらない
/// 場合は新しいノードを積まず `x` をそのまま返す。
pub fn repeat<'t>(x: &Var<'t>, repeats: &[usize]) -> Result<Var<'t>, AutodiffError> {
    let in_shape = x.shape();
    let rank = in_shape.len();
    if repeats.len() < rank {
        return Err(AutodiffError::InvalidArgument(format!(
            "rearrange_ops::repeat: repeats の長さ {} が rank {} 未満で\
             PyTorch 仕様に反する",
            repeats.len(),
            rank
        )));
    }

    let mut cur = *x;
    if repeats.len() > rank {
        let extra = repeats.len() - rank;
        let mut padded_shape = vec![1usize; extra];
        padded_shape.extend_from_slice(&in_shape);
        cur = cur.broadcast_to(&padded_shape)?;
    }

    // Pass 1（確保なし・overflow 検査のみ）: 軸ごとの最終要素数
    // `axis_total = n_d * r_d` と、全軸を通した最終出力の総要素数
    // `total_out_elems` を `checked_mul` で確定させる。この時点では
    // まだ `MAX_INDEX_ALLOC_BYTES`（実用上の確保上限）を検査しない
    // （codex-review 指摘・PR #2256「ゼロ係数より先の大きな軸が空
    // テンソル契約をエラーに変える」）——ある軸の `r` が大きくても
    // 他の軸に `r == 0`（または `n == 0`）があれば最終出力は空テンソル
    // になり、その軸へ実際に `n*r` 長の添字ベクタを確保する必要はない
    // ため、全軸積が確定するまで上限判定を保留する。`cur.shape()` は
    // broadcast 由来の stride-0 view でも論理 shape を返す。
    let cur_shape = cur.shape();
    let mut axis_totals: Vec<usize> = Vec::with_capacity(repeats.len());
    let mut total_out_elems: usize = 1;
    for (d, &r) in repeats.iter().enumerate() {
        let n = cur_shape[d];
        let axis_total = n
            .checked_mul(r)
            .ok_or(ShapeError::ElementCountOverflow)
            .map_err(AutodiffError::Shape)?;
        axis_totals.push(axis_total);
        total_out_elems = total_out_elems
            .checked_mul(axis_total)
            .ok_or(ShapeError::ElementCountOverflow)
            .map_err(AutodiffError::Shape)?;
    }

    if total_out_elems == 0 {
        // 最終出力が空テンソルになるケース（`axis_totals` のいずれかが
        // 0）。他の軸の `r` がどれだけ大きくても、実際に `n*r` 長の
        // 添字ベクタを確保する必要はない（cursor(Medium)「Zero axis
        // bypasses allocation guard」・codex(P2)「ゼロ係数より先の大きな
        // 軸が空テンソル契約をエラーに変える」・PR #2256）。最初に見つ
        // かった `axis_total == 0` の軸だけ空添字（長さ 0）で
        // `index_select` し実体を空にしてから、`Var::reshape` で最終
        // shape（`axis_totals`）へ一括変換する。`Tensor::is_contiguous`
        // は `numel() == 0` を常に連続とみなすため（NumPy 方式。
        // `crates/tensor-core/src/tensor.rs::is_contiguous`）、
        // 個々の軸が target と異なる中間 shape のままでも
        // `ShapeError::NonContiguousReshape` にはならない。
        // `total_out_elems == 0` は `axis_totals` の `checked_mul` 連鎖
        // （usize 積）の結果であり、積が 0 になるのは因子のいずれかが
        // 0 の場合に限る（IEEE 754 ではなく usize 演算のため NaN 等の
        // 例外経路は存在しない）。よってこの `position` は必ず `Some`
        // を返すが、内部不変条件の変更がここへ影響しても本番経路の
        // panic に漏れないよう `.expect()` ではなく型付きエラーへ変換
        // する（codex-review 指摘・P1・PR #2256「ゼロ軸探索に .expect()
        // を使用しており panic 禁止規約違反」）。
        let z = axis_totals
            .iter()
            .position(|&t| t == 0)
            .ok_or(ShapeError::ElementCountOverflow)
            .map_err(AutodiffError::Shape)?;
        let empty_idx = Tensor::new(Vec::new(), &[0]).map_err(AutodiffError::Shape)?;
        cur = cur.index_select(z, &empty_idx)?;
        return cur.reshape(&axis_totals);
    }

    // 非ゼロケースのうち、少なくとも 1 軸が `r != 1`（実際に
    // `index_select` による実体化が発生する）場合は、個々の添字ベクタ
    // が上限未満でも最終出力の総要素数 `total_out_elems` が 1 GiB
    // （`i32`／`f32` 相当 4 バイト換算）を超えないかも検査する
    // （cursor-review(Medium)「Large repeat can abort process」・
    // PR #2256——各軸の `axis_totals[d]` 単体は `MAX_INDEX_ALLOC_BYTES`
    // 未満でも、複数軸の `r != 1` を掛け合わせた最終出力（または経路上
    // の中間実体化）が 1 GiB を大きく超えうる。逐次実行される
    // `index_select` は軸ごとに要素数を増やしていくため、実体化を伴う
    // 経路での中間確保サイズの上界は最終値 `total_out_elems` に一致する
    // ）。全軸 `r == 1`（`repeat_no_op_on_huge_broadcast_view_does_not_
    // reject` が検証する no-op 経路）は `index_select` を一切呼ばず
    // `broadcast_to` 由来の stride-0 view のまま返るため、この検査の
    // 対象外とする。
    if repeats.iter().any(|&r| r != 1) {
        checked_index_alloc_len(total_out_elems)?;
    }

    // `r == 1` の軸は添字ベクタを確保せず（後続ループで `continue` に
    // よりスキップ）そのまま no-op になるため、軸単位の確保上限チェック
    // の対象からも外す（cursor(Medium)「Allocation cap rejects no-op
    // repeat」・PR #2256——`broadcast_to` 由来の stride-0 view で論理長
    // が 1GiB 換算の上限を超える軸でも、`r == 1`（no-op）呼び出しでは
    // 実際の確保が発生しないため誤って拒否しない）。`r != 1` の軸に
    // 限り、実際に確保する添字ベクタの長さ `axis_totals[d]` が
    // `i32` 添字値として表現できる範囲か（`checked_axis_len_as_i32`）・
    // 実用上の確保上限に収まるか（`checked_index_alloc_len`）を検査する。
    for (d, &r) in repeats.iter().enumerate() {
        if r == 1 {
            continue;
        }
        checked_axis_len_as_i32(cur_shape[d])?;
        checked_index_alloc_len(axis_totals[d])?;
    }

    for (d, &r) in repeats.iter().enumerate() {
        if r == 1 {
            continue;
        }
        let n = cur.shape()[d];
        // `n * r` は上記の事前検証ループで既に overflow・確保上限の
        // 両方を確認済みのため、ここでは再検証せず素の乗算でよい
        // （`axis_totals[d]` と同じ値になる。軸 `d` の長さは他の軸への
        // `index_select` の影響を受けないため一致する）。
        let total = n * r;
        // `total == 0`（`n == 0` または `r == 0`）は `total_out_elems == 0`
        // 分岐で既に処理済みのため、ここへは到達しない
        // （`total_out_elems` はこの軸の `axis_total` を乗算因子に含む）。
        debug_assert_ne!(total, 0);
        let idx_data: Vec<i32> = (0..total).map(|j| (j % n) as i32).collect();
        let idx = Tensor::new(idx_data, &[total]).map_err(AutodiffError::Shape)?;
        cur = cur.index_select(d, &idx)?;
    }
    Ok(cur)
}

/// 繰り返す（`torch.tile` 相当。イシュー #2143）。
///
/// `reps.len() < rank` の場合は PyTorch と同じく先頭を `1` で埋めて
/// rank に合わせてから [`repeat`] へ委譲する。`reps.len() >= rank` の
/// 場合は [`repeat`] にそのまま委譲する（`reps.len() > rank` の rank
/// 拡張は `repeat` 側の先頭軸追加に任せる）。
pub fn tile<'t>(x: &Var<'t>, reps: &[usize]) -> Result<Var<'t>, AutodiffError> {
    let rank = x.shape().len();
    if reps.len() >= rank {
        return repeat(x, reps);
    }
    let pad = rank - reps.len();
    let mut padded = vec![1usize; pad];
    padded.extend_from_slice(reps);
    repeat(x, &padded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    // --- flip: 単一軸・複数軸・rank 0・軸長 0 ---

    #[test]
    fn flip_single_axis() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        let out = flip(&x, &[0]).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![4.0, 3.0, 2.0, 1.0]
        );
    }

    #[test]
    fn flip_multiple_axes_2d() {
        let tape = Tape::new();
        // [[1,2,3],[4,5,6]] を両軸反転すると [[6,5,4],[3,2,1]]。
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let out = flip(&x, &[0, 1]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[2, 3]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        );
    }

    #[test]
    fn flip_empty_dims_is_noop_no_new_node() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let before = tape.len();
        let out = flip(&x, &[]).unwrap();
        assert_eq!(tape.len(), before);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            x.to_tensor().host_slice().into_owned()
        );
    }

    #[test]
    fn flip_zero_length_axis() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0, 2]));
        let out = flip(&x, &[0]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[0, 2]);
    }

    #[test]
    fn flip_axis_out_of_range_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(
            flip(&x, &[1]),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: 1,
                rank: 1
            }))
        ));
    }

    #[test]
    fn flip_duplicate_axis_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        assert!(matches!(
            flip(&x, &[0, 0]),
            Err(AutodiffError::Shape(ShapeError::DuplicateAxis { axis: 0 }))
        ));
    }

    // --- flip: 非 contiguous 入力（transpose の view） ---

    #[test]
    fn flip_non_contiguous_input() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let xt = x.transpose(0, 1).unwrap();
        let out = flip(&xt, &[0]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[3, 2]);
        // transpose: [[1,4],[2,5],[3,6]]、軸 0 反転: [[3,6],[2,5],[1,4]]。
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![3.0, 6.0, 2.0, 5.0, 1.0, 4.0]
        );
    }

    // --- flip: 勾配（各入力要素への寄与は常に 1 つ・bit 一致） ---

    #[test]
    fn flip_gradient_matches_reference() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        let y = flip(&x, &[0]).unwrap();
        let w = tape.var(&t(vec![10.0, 20.0, 30.0, 40.0], &[4]));
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        // dloss/dx[i] = w[flip 後の対応位置] = w の逆順。
        assert_eq!(
            bits(dx.host_slice().as_ref()),
            bits(&[40.0, 30.0, 20.0, 10.0])
        );
    }

    // --- roll: 単一軸・正シフト・負シフト・軸長超過シフト ---

    #[test]
    fn roll_positive_shift() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]));
        let out = roll(&x, &[2], &[0]).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![4.0, 5.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn roll_negative_shift() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]));
        let out = roll(&x, &[-2], &[0]).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![3.0, 4.0, 5.0, 1.0, 2.0]
        );
    }

    #[test]
    fn roll_shift_exceeding_axis_len_wraps() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let out = roll(&x, &[7], &[0]).unwrap(); // 7 % 3 == 1
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![3.0, 1.0, 2.0]
        );
    }

    #[test]
    fn roll_zero_shift_is_noop_no_new_node() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let before = tape.len();
        let out = roll(&x, &[0], &[0]).unwrap();
        assert_eq!(tape.len(), before);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            x.to_tensor().host_slice().into_owned()
        );
    }

    #[test]
    fn roll_duplicate_axis_applies_sequentially() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        // 同じ軸へ shift=1 を 2 回適用 == shift=2 相当。
        let out = roll(&x, &[1, 1], &[0, 0]).unwrap();
        let expected = roll(&x, &[2], &[0]).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            expected.to_tensor().host_slice().into_owned()
        );
    }

    #[test]
    fn roll_length_mismatch_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(
            roll(&x, &[1, 2], &[0]),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn roll_empty_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(
            roll(&x, &[], &[]),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn roll_axis_out_of_range_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(matches!(
            roll(&x, &[1], &[1]),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: 1,
                rank: 1
            }))
        ));
    }

    #[test]
    fn roll_zero_length_axis_is_noop() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0, 2]));
        let out = roll(&x, &[3], &[0]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[0, 2]);
    }

    // --- roll: 勾配（bit 一致） ---

    #[test]
    fn roll_gradient_matches_reference() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        let y = roll(&x, &[1], &[0]).unwrap(); // [4,1,2,3]
        let w = tape.var(&t(vec![10.0, 20.0, 30.0, 40.0], &[4]));
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        // y[i] = x[(i-1) mod 4] なので dloss/dx[j] = w[(j+1) mod 4]。
        assert_eq!(
            bits(dx.host_slice().as_ref()),
            bits(&[20.0, 30.0, 40.0, 10.0])
        );
    }

    // --- repeat: 基本・r==0・rank 拡張 ---

    #[test]
    fn repeat_basic_1d() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let out = repeat(&x, &[3]).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn repeat_2d_per_axis() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = repeat(&x, &[2, 1]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[4, 2]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn repeat_zero_produces_zero_length_axis() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let out = repeat(&x, &[0]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[0]);
    }

    // --- ゼロ軸が後続にあるため最終出力が空テンソルになるケース
    // （cursor(Medium)「Zero axis bypasses allocation guard」・
    // codex(P2)「ゼロ係数より先の大きな軸が空テンソル契約をエラーに
    // 変える」・PR #2256）。先頭軸の `r` は `MAX_INDEX_ALLOC_BYTES`
    // （1 GiB）換算の添字ベクタを要求する規模だが、後続軸に `r == 0`
    // があるため最終出力は空テンソルであり、`Ok` で成功し実際には
    // 巨大な確保を行わないことを検証する（`n == 1` ケース）。

    #[test]
    fn repeat_large_axis_before_zero_axis_returns_empty_ok() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0], &[1, 1]));
        let out = repeat(&x, &[300_000_000, 0]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[300_000_000, 0]);
        assert_eq!(out.to_tensor().host_slice().into_owned(), Vec::<f32>::new());
    }

    // 同様のケースだが軸長 `n > 1`（`broadcast_to` 由来ではない実体軸）
    // でも `reshape` による最終化経路（`n*r` 分の添字ベクタを介さない）
    // を通ることを検証する。
    #[test]
    fn repeat_large_axis_before_zero_axis_with_n_gt_1() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2, 1]));
        let out = repeat(&x, &[200_000_000, 0]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[400_000_000, 0]);
    }

    // 空テンソル経由の backward: 勾配は入力 shape のゼロ埋めになる
    // （空テンソルへの寄与なので値そのものは検証対象外・shape 一致のみ
    // 確認する）。ゼロ軸を先に処理する経路が勾配計算でも小さい中間
    // テンソルに留まることを間接的に確認する。
    #[test]
    fn repeat_empty_output_gradient_matches_input_shape() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0], &[1, 1]));
        let y = repeat(&x, &[300_000_000, 0]).unwrap();
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(dx.shape(), &[1, 1]);
        assert_eq!(dx.host_slice().into_owned(), vec![0.0]);
    }

    // --- `r == 1`（no-op）専用軸は確保上限チェックの対象外（cursor
    // (Medium)「Allocation cap rejects no-op repeat」・PR #2256）。
    // `broadcast_to` の stride-0 view で論理長を `MAX_INDEX_ALLOC_BYTES`
    // 換算の上限超（256M 要素超）まで拡張した軸でも、`repeats` が
    // 全軸 `r == 1` なら実際の確保が発生せず誤って拒否しないことを
    // 検証する（`to_tensor()` は呼ばない——呼ぶと実体化のため意図的に
    // 避ける。`flip_on_broadcast_view_rejects_practically_unallocatable_
    // size` と同じ手法）。
    #[test]
    fn repeat_no_op_on_huge_broadcast_view_does_not_reject() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0], &[1]));
        let huge = x.broadcast_to(&[300_000_000]).unwrap();
        let before = tape.len();
        let out = repeat(&huge, &[1]).unwrap();
        assert_eq!(tape.len(), before);
        assert_eq!(out.shape(), &[300_000_000]);
    }

    #[test]
    fn repeat_rank_extension_adds_leading_axis() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let out = repeat(&x, &[2, 1]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[2, 2]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn repeat_all_ones_same_rank_is_noop_no_new_node() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let before = tape.len();
        let out = repeat(&x, &[1]).unwrap();
        assert_eq!(tape.len(), before);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            x.to_tensor().host_slice().into_owned()
        );
    }

    #[test]
    fn repeat_too_short_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        assert!(matches!(
            repeat(&x, &[2]),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn repeat_non_contiguous_input() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let xt = x.transpose(0, 1).unwrap(); // [[1,4],[2,5],[3,6]]
        let out = repeat(&xt, &[1, 2]).unwrap();
        assert_eq!(out.to_tensor().shape(), &[3, 4]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 4.0, 1.0, 4.0, 2.0, 5.0, 2.0, 5.0, 3.0, 6.0, 3.0, 6.0]
        );
    }

    // --- repeat: 勾配（r 個のコピーの合算。整数値なので合算順序に
    // 依らず bit 一致） ---

    #[test]
    fn repeat_gradient_sums_contributions() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let y = repeat(&x, &[3]).unwrap(); // [1,2,1,2,1,2]
        let w = tape.var(&t(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[6]));
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        // 各入力要素は 3 回コピーされ、重み全て 1 なので dx = [3, 3]。
        assert_eq!(bits(dx.host_slice().as_ref()), bits(&[3.0, 3.0]));
    }

    // --- tile: 基本・reps 短い／長い ---

    #[test]
    fn tile_reps_shorter_than_rank_pads_leading_ones() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = tile(&x, &[2]).unwrap(); // reps=[2] は [1,2] へパディング
        assert_eq!(out.to_tensor().shape(), &[2, 4]);
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
        );
    }

    #[test]
    fn tile_reps_longer_than_rank_delegates_to_repeat() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let out = tile(&x, &[2, 1]).unwrap();
        let expected = repeat(&x, &[2, 1]).unwrap();
        assert_eq!(out.to_tensor().shape(), expected.to_tensor().shape());
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            expected.to_tensor().host_slice().into_owned()
        );
    }

    #[test]
    fn tile_matches_repeat_when_same_rank() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = tile(&x, &[2, 1]).unwrap();
        let expected = repeat(&x, &[2, 1]).unwrap();
        assert_eq!(
            out.to_tensor().host_slice().into_owned(),
            expected.to_tensor().host_slice().into_owned()
        );
    }

    // --- 有限差分検算（1 ケース。repeat の合成勾配を独立に検証） ---

    #[test]
    fn repeat_gradient_matches_finite_difference() {
        let eps = 1e-3f32;
        let base = vec![0.7f32, -1.3, 2.1];
        let weights = vec![1.0f32, 0.5, -0.25, 2.0, 0.1, -1.0];

        let eval = |data: &[f32]| -> f32 {
            let tape = Tape::new();
            let x = tape.var(&t(data.to_vec(), &[3]));
            let w = tape.var(&t(weights.clone(), &[6]));
            let y = repeat(&x, &[2]).unwrap();
            let loss = y.mul(&w).unwrap().sum(None).unwrap();
            loss.to_tensor().host_slice()[0]
        };

        let tape = Tape::new();
        let x = tape.var(&t(base.clone(), &[3]));
        let w = tape.var(&t(weights.clone(), &[6]));
        let y = repeat(&x, &[2]).unwrap();
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        let dx = dx.host_slice().into_owned();

        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += eps;
            let mut minus = base.clone();
            minus[i] -= eps;
            let numeric = (eval(&plus) - eval(&minus)) / (2.0 * eps);
            assert!(
                (numeric - dx[i]).abs() < 1e-2,
                "有限差分検算が解析解と乖離: i={i}, numeric={numeric}, analytic={}",
                dx[i]
            );
        }
    }

    // --- エラー系: オーバーフロー（境界検査。REQ-8） ---

    #[test]
    fn repeat_rejects_element_count_overflow_without_panicking() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0; 2], &[2]));
        let err = repeat(&x, &[usize::MAX]).expect_err("巨大な repeats は確保前に拒否されるはず");
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    // --- エラー系: 実用上確保不能なサイズ（`MAX_INDEX_ALLOC_BYTES`。
    // codex-review 指摘・PR #2256）。isize::MAX には収まる（overflow
    // しない）が実運用では確保不能な規模を、accept 側の実確保を伴わず
    // 確保前に拒否できることを検証する。accept 側（上限未満での成功）
    // は数百 MB 規模の確保を CI で走らせることになるため意図的に
    // テストしない（`crates/backend-cpu/src/ops.rs:2659` の 2 GiB 級
    // 境界テスト省略と同じ判断）。
    #[test]
    fn repeat_rejects_practically_unallocatable_size_without_panicking() {
        let tape = Tape::new();
        // shape [1] に repeats=[1_000_000_000] は 4GB（i32 換算）の
        // 添字ベクタになり、`usize` 積としては overflow しないため
        // `isize::MAX` 検査のみでは通過してしまっていた（codex-review
        // 指摘の再現ケースそのもの）。
        let x = tape.var(&t(vec![1.0], &[1]));
        let err = repeat(&x, &[1_000_000_000])
            .expect_err("実用上確保不能な repeats は確保前に拒否されるはず");
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    // --- エラー系: 軸ごとの添字ベクタは上限未満でも、複数軸の `r != 1`
    // を掛け合わせた最終出力の総要素数が 1 GiB 換算の上限を超える
    // ケース（cursor-review(Medium)「Large repeat can abort process」・
    // PR #2256）。各軸単体の確保上限チェックだけでは見逃すことを確認
    // する。
    #[test]
    fn repeat_rejects_total_output_size_even_when_each_axis_is_small() {
        let tape = Tape::new();
        // 各軸の `axis_total` は 40,000（i32 換算 160,000 バイト。
        // `MAX_INDEX_ALLOC_BYTES` = 1 GiB を大きく下回る）だが、2 軸の
        // 積は 1.6e9 要素（4 バイト換算で約 6.4 GiB）となり上限を超える。
        let x = tape.var(&t(vec![1.0], &[1, 1]));
        let err = repeat(&x, &[40_000, 40_000])
            .expect_err("軸ごとの積が実用上確保不能な規模なら確保前に拒否されるはず");
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn flip_on_broadcast_view_rejects_practically_unallocatable_size() {
        let tape = Tape::new();
        // `broadcast_to` は stride-0 view のため、実体を伴わずに軸長を
        // 巨大化できる（レビュー指摘の一般化: `repeat`／`tile` の
        // `repeats` に限らず `flip`／`roll` も同じ `checked_index_
        // alloc_len` を通るため同じ上限で拒否される）。
        let x = tape.var(&t(vec![1.0], &[1]));
        let big = x.broadcast_to(&[1_000_000_000]).unwrap();
        let err = flip(&big, &[0]).expect_err("broadcast 由来の巨大軸長も確保前に拒否されるはず");
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ElementCountOverflow)
        ));
    }
}
