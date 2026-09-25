//! `prod`・`logsumexp`・`any`・`all`・`norm_p`（p-ノルム）の 5 縮約
//! （イシュー #2147・親 #2131「5-B 演算」）。
//!
//! **facade 非公開（意図的）**: `crates/autodiff/src/matrix_ops.rs`
//! モジュール doc と同じ理由・同じ判断枠組みによる。`Var` は facade
//! （`fandhe_ai` クレート）から直接再エクスポートされるため、`Var` への
//! inherent メソッド追加は即座に facade 公開面へ出てしまう。イシュー
//! #2147 本文は facade 公開面（`Var::prod` 等の委譲メソッド）を承認
//! 事項として明示し、親 #2131 はこのツリーに限り「設計判断記録 →
//! 承認 → 実装」の 2 段階を定めるため、承認が取れるまでは自由関数と
//! して `Var` の外に置き到達不能にする（`docs/autodiff-reduce-ops-
//! decision.md` §0）。承認後は `Var::prod` 等の薄い委譲メソッドを追加
//! し、facade 側の保留ガード（`crates/facade/src/lib.rs::
//! VarReduceOpsHoldDoctestGuard`）を撤去する。
//!
//! **PyTorch 相当・出力型**（詳細は `docs/autodiff-reduce-ops-
//! decision.md` §1 の表を参照）:
//!
//! | 演算 | PyTorch 相当 | 出力 | 微分 |
//! |---|---|---|---|
//! | [`prod`] | `torch.prod` | f32 | 可 |
//! | [`logsumexp`] | `torch.logsumexp` | f32 | 可 |
//! | [`any`] | `torch.any` | **f32 の 0.0／1.0 マスク** | 勾配ゼロ |
//! | [`all`] | `torch.all` | **f32 の 0.0／1.0 マスク** | 勾配ゼロ |
//! | [`norm_p`] | `torch.linalg.vector_norm(ord=p)` | f32 | 可 |
//!
//! `any`／`all` の **bool 出力版**は `crate::bool_ops`（イシュー #2141）
//! の対象であり本モジュールの対象外——本モジュールが返すのは既存の
//! `Var::gt` 等と同じ f32 の 0.0／1.0 マスク（勾配ゼロの tape ノード）。
//!
//! **新規 `Op` の有無**:
//! - [`prod`]・[`any`]・[`all`] は既存 `Op` の合成のみ（新規 `Op`
//!   なし）: `prod` は `cumprod（Op::Cumprod）→ narrow（Op::Narrow）→
//!   squeeze（reshape へ委譲）`、`any`／`all` は `ne（Op::ScalarBinary）
//!   → max／min（Op::Max／Op::Min）`。
//! - [`logsumexp`]・[`norm_p`] は専用 `Op`（`crate::tape::Op::
//!   LogSumExp`／`Op::PNorm`）を追加する。`sum`／`exp`／`log` や
//!   `max`／`pow`／`sum` の素朴な合成では、全要素が `-inf`（または
//!   `+inf`）の lane・overflow を起こす `p` で `NaN` が出るため
//!   （`docs/autodiff-reduce-ops-decision.md` §2.4・§2.5）。
//!
//! **数値契約**（詳細は `docs/autodiff-reduce-ops-decision.md` §3）:
//! - `prod`: `Op::Cumprod`（イシュー #1731）の forward は `f64`
//!   アキュムレータで計算し 1 回だけ `f32` へ落とすため、零要素を含む
//!   場合でも正確。VJP は除算を使わない厳密形（排他 prefix 積）。
//! - `logsumexp`：`m = max(x)`（`f64`。非有限なら安定化シフトを `0` に
//!   切り替える）→ `Σ exp(x_i − m)` を `f64` で蓄積 → `ln(acc) + m`。
//! - `norm_p`：`mx = max|x_i|` を括り出す overflow-safe なスケール形
//!   （`mx · (Σ (|x_i|/mx)^p)^(1/p)`）。
//! - `any`／`all`：出力値は厳密に `0.0` か `1.0` のみで縮約順序に依存
//!   しないため 3 バックエンドで **bit 完全一致**する。
//!
//! **PyTorch との差分**（`docs/autodiff-reduce-ops-decision.md` §4）:
//! - `logsumexp` の `y == -inf`（縮約対象が全て `-inf`）lane の勾配は
//!   `0`（PyTorch は `NaN`）。`NaN` 勾配で学習を汚染しない安全側の判断。
//! - `norm_p` の `p` は有限かつ正のみ許容（`0`／負／`±inf`／`NaN` は
//!   `AutodiffError::InvalidArgument`）。PyTorch は `inf`／`0`／負の
//!   `p` も受け付けるが、inf ノルムの勾配分配方式が本リポでは未定
//!   （`Var::max` の先勝ち VJP のみ・均等分配は別イシュー）のため見送る。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: `prod` の
//! 空縮約（`n == 0`）は単位元 `1.0` を `narrow` 呼び出し前に返す
//! （`narrow(n-1)` の underflow 回避）。`any`／`all` の空縮約は
//! `max`／`min` が単位元を持たずエラーになるため、`any(∅) = 0.0`・
//! `all(∅) = 1.0` を明示的に返す（PyTorch と同じ規約）。いずれも
//! `x`（`any`／`all` は `x.ne(&zero)`）への計算グラフ依存を保った
//! `empty_reduce_identity`（`x.sum(dim)` の単位元 `0.0` 契約 +
//! 定数バイアス加算）で構築し、`x` から独立した定数葉としては返さない
//! （codex-review P2 是正・PR #2263。対応前は `push_leaf` による独立葉
//! 登録で `x` への逆伝播経路が失われていた）。`norm_p` の `p` は
//! 有限性・正値を dispatch 前に検査する（`nn/norm.rs::validate_eps`
//! と同じ fail-closed 規律）。本番経路で `unwrap()`／`expect()` は
//! 使わない。
//!
//! **確保前のバイト数上限検査は全公開入口の冒頭で一律に行う契約
//! （codex-review P1 是正・累計 3 段階・イシュー #2147・PR #2263）**:
//! 唯一の共有ヘルパ `ensure_alloc_fits_f32`（本ファイル内 `fn`）が
//! `checked_bytes_for::<f32>`（`crate::bool_ops`）で入力 shape・
//! `out_shape`（既に求まっている場合）の要素数積の `usize`
//! オーバーフロー・`Vec` allocation 上限（`isize::MAX` バイト）超過を
//! 検査する。[`prod`]・[`logsumexp`]・[`any`]・[`all`]・[`norm_p`] の
//! 全 5 入口は、**あらゆる分岐（空縮約・`p` の特殊化・`dim` の有無・
//! `Unsupported` フォールバック）や合成演算・実体化（`contiguous`／
//! `cumprod`／`ne`／`materialize_one`／既存 API への委譲）よりも前**に
//! 本ヘルパを呼ぶ（各関数 doc「確保前のバイト数上限検査」参照）。
//!
//! 是正の経緯（3 段階。各段の詳細は各関数 doc・
//! `docs/autodiff-reduce-ops-decision.md` §2.2〜§2.5）:
//! 1. 初版は `out_shape.iter().product()` と `vec![...; numel]` を
//!    無検査で実行しており、例えば shape `[0, usize::MAX]` を
//!    `dim=Some(0)` で縮約すると capacity overflow で panic しえた
//!    （是正: 空縮約分岐に `checked_bytes_for` を追加）。
//! 2. `logsumexp`／`norm_p` は非空縮約（`n != 0`）でも `x` が小さな
//!    ストレージを巨大な shape へ broadcast した view であれば
//!    `outer`／`inner`（broadcast 側の次元）が巨大になりうるため、
//!    `materialize_one` の入力実体化（`gather_elements`／
//!    `dense_vec`）・`BackendOps::logsumexp`／`vector_norm_p`
//!    （`backend-cpu::reduction::axis_reduce_logsumexp`／
//!    `axis_reduce_vector_norm_p` の `.collect()`）・そのフォールバック
//!    （`eval::logsumexp_along`／`vector_norm_p_along` の
//!    `vec![0f32; outer * inner]`）のいずれも未検査だった（是正:
//!    dispatch 前に入力 shape・`out_shape` の両方を検査）。
//! 3. 棚卸しの結果、`prod`／`any`／`all` は空縮約分岐でしか検査して
//!    おらず非空縮約（`contiguous`／`cumprod`／`ne` の実体化）が未検査
//!    のまま残っていた。加えて `norm_p` は `p ∈ {1.0, 2.0}` の委譲判定
//!    （`Var::norm_l1`／`norm_l2` への委譲）が検査より前にあり、委譲先
//!    が独自に確保前検査を持たない場合は本関数の検査を迂回していた
//!    （是正: 共有ヘルパ `ensure_alloc_fits_f32` を導入し、全 5 入口
//!    の冒頭・全分岐より前に呼ぶ形へ統一。codex-review 新規指摘 2 件・
//!    PR #2263）。

use fandhe_ai_tensor_core::{BackendError, Tensor, reduce_out_shape};

use crate::bool_ops::checked_bytes_for;
use crate::error::AutodiffError;
use crate::eval;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

/// `dim` に沿った縮約対象の要素数（`Var::mean`／`Var::norm` と同じ
/// 「`dim=None` は全要素数・`dim=Some(axis)` は `shape[axis]`」規約）。
fn reduce_axis_len(shape: &[usize], dim: Option<usize>) -> usize {
    match dim {
        None => shape.iter().product(),
        Some(axis) => shape[axis],
    }
}

/// 本モジュールの全公開入口（[`prod`]・[`logsumexp`]・[`any`]・
/// [`all`]・[`norm_p`]）が冒頭で呼ぶ、唯一の確保前バイト数上限検査
/// ヘルパ（codex-review P1 是正 3・イシュー #2147・PR #2263）。
///
/// `input_shape`（`x` の shape。`materialize_one`／`contiguous`／`ne`
/// など、入力を実体化するあらゆる経路が確保しうる最大サイズ）を
/// `checked_bytes_for::<f32>` で検査し、`out_shape` が既に求まって
/// いる場合（`reduce_out_shape` は shape 演算のみで確保を行わないため
/// 常に先に呼べる）は出力側も併せて検査する。
///
/// **呼び出し規律**: 各 `pub fn` は、空縮約分岐・`p` の特殊化
/// （`norm_p` の `p ∈ {1.0, 2.0}` 委譲）・`dim` の有無・
/// `Unsupported` フォールバックのいずれの分岐にも入る前、かつ
/// `contiguous`／`cumprod`／`ne`／`materialize_one`／既存 API への
/// 委譲（`norm_l1`／`norm_l2`）のいずれの実体化よりも前に本関数を
/// 呼ぶ（モジュール doc「確保前のバイト数上限検査」参照。対応前は
/// `prod`／`any`／`all` が空縮約分岐でしか検査しておらず、非空縮約で
/// `contiguous`／`cumprod`／`ne` が無検査に確保していた。`norm_p` は
/// `p ∈ {1.0, 2.0}` の委譲が検査より前にあり、委譲先
/// `Var::norm_l1`／`norm_l2`〈本 PR の差分外の既存経路〉が
/// 確保前検査を欠いていた場合に迂回されていた。codex-review 新規
/// 指摘 2 件・PR #2263）。
///
/// `prod` の `dim=None` 経路（`reshape([n])` → `cumprod(0)`）・
/// `Some(axis)` 経路（`cumprod(axis)`）はいずれも入力と同じ要素数の
/// 中間 shape しか作らないため、`input_shape` の検査のみで
/// `contiguous`／`cumprod` の確保も守られる（中間 shape が入力より
/// 大きくなる経路は本モジュールに存在しない）。
fn ensure_alloc_fits_f32(
    input_shape: &[usize],
    out_shape: Option<&[usize]>,
) -> Result<(), AutodiffError> {
    checked_bytes_for::<f32>(input_shape)?;
    if let Some(out_shape) = out_shape {
        checked_bytes_for::<f32>(out_shape)?;
    }
    Ok(())
}

/// バックエンド実装の戻り値 shape を検証する（`var.rs::verify_shape`
/// と同型。`pub(crate)` 化を避け本モジュール内に複製する——
/// `var.rs` 側は `fn`〈非 `pub(crate)`〉のため呼べない）。
fn verify_shape(actual: &[usize], expected: &[usize]) -> Result<(), AutodiffError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                lhs: actual.to_vec(),
                rhs: expected.to_vec(),
            },
        )))
    }
}

/// CPU 本番経路の `BackendError` を `AutodiffError` へ写像する
/// （`var.rs::unify_backend_error` と同型の複製。理由は
/// [`verify_shape`] と同じ）。
fn unify_backend_error(err: BackendError) -> AutodiffError {
    match err {
        BackendError::InvalidArgument(msg) => AutodiffError::InvalidArgument(msg),
        other => AutodiffError::Backend(other),
    }
}

/// `x` を層 1 で実体化した `Tensor<f32>` を返す（`bool_ops::
/// materialize_pair` と同じ「`nodes` の `RefCell` 借用をこのブロック内に
/// 閉じ込め、返す前に解放する」パターン）。
fn materialize_one<'t>(x: &Var<'t>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = x.tape().nodes.borrow();
    let ops = x.tape().ops();
    Ok(materialize_fallible(&nodes, ops, x.node_id())?.clone())
}

/// [`prod`]／[`any`]／[`all`] の空縮約（`n == 0`）分岐が共通で使う、
/// `x`（または `x` から合成した中間 `Var`。`any`／`all` は `x.ne(&zero)`
/// を渡す）への計算グラフ依存を保ったまま単位元を返すヘルパー
/// （codex-review P2 是正・イシュー #2147・PR #2263）。
///
/// `Var::sum`（`Op::Sum`）は縮約対象要素数 0 のとき単位元 `0.0` を
/// 返す既存契約を持つ（`backend-cpu::reduction::sum` モジュール doc
/// 「空縮約の意味論」。NumPy 互換）ため、`base.sum(dim)` がそのまま
/// `0.0` の定数を `out_shape` で返す。`bias`（`prod`／`all` は
/// `1.0`・`any` は `0.0`）を加算して求める単位元へ揃える。
///
/// `Op::Sum` の VJP（`grad.rs::unreduce_broadcast`）が `input` へ
/// shape 相応（縮約対象軸が 0 長のため要素数 0）の勾配を記録するため、
/// 対応前の `push_leaf` による独立葉登録と異なり `base`（ひいては
/// `x`）への逆伝播経路が保たれる。`any`／`all` は `base` が
/// `x.ne(&zero)` であるため、`ne` の VJP（常にゼロを返す。
/// `ScalarBinaryOp::Ne` の契約）を経由して非空の `any`／`all` と同じ
/// 「勾配ゼロだが経路は保持」の形になる。
///
/// **事前条件（呼び出し元が満たす）**: `ensure_alloc_fits_f32`
/// （`out_shape` を含む）で確保前検証済みであること。`base.sum(dim)`
/// 内部の `Vec` 確保（`backend-cpu::reduction::axis_reduce_sum` の
/// `(0..total_out).into_par_iter().collect()`）はそれ自体は無検査の
/// ため、呼び出し元が事前に境界検査を通す規律に依存する。呼び出し元
/// （[`prod`]・[`any`]・[`all`]）は各関数冒頭で `ensure_alloc_fits_f32`
/// を呼ぶため、この事前条件は分岐に依らず常に満たされる。
fn empty_reduce_identity<'t>(
    base: &Var<'t>,
    dim: Option<usize>,
    bias: f32,
) -> Result<Var<'t>, AutodiffError> {
    let summed = base.sum(dim)?;
    if bias == 0.0 {
        return Ok(summed);
    }
    let bias_val = Tensor::scalar(bias);
    let bias_id = base.tape().push_leaf(bias_val, false);
    summed.add(&Var::from_raw(base.tape(), bias_id))
}

/// 縮約対象の要素数 `n` に沿った累積積（`torch.prod` 相当。イシュー
/// #2147）。`dim: None` は全軸縮約（先に `reshape([numel])` してから
/// `cumprod(0)` を取る）。
///
/// `Var::cumprod`（`Op::Cumprod`。イシュー #1731）の forward は `f64`
/// アキュムレータで計算し 1 回だけ `f32` へ落とすため、零要素を含む
/// 入力でも正確。VJP は除算を使わない厳密形（排他 prefix 積 ×
/// 後ろ向き Horner 型再帰）のため、零要素が 0 個・1 個・2 個以上の
/// いずれでも正しい。
///
/// **確保前のバイト数上限検査（`ensure_alloc_fits_f32`。codex-review
/// P1 是正・イシュー #2147・PR #2263）**: 空縮約（`n == 0`）に限らず
/// 非空縮約でも `x` が小さなストレージを巨大な shape へ broadcast した
/// view であれば `contiguous()`／`cumprod` が入力 shape 相応の巨大な
/// `Vec` を無検査に確保しうる（対応前は空縮約分岐でしか検査しておらず
/// 非空縮約が未検査のまま残っていた。codex-review 新規指摘・PR
/// #2263）。関数冒頭で入力 shape・`out_shape` の双方を検査し、要素数積
/// の `usize` オーバーフロー・`Vec` allocation 上限〈`isize::MAX`
/// バイト〉超過のいずれも型付きエラーで拒否する（本番経路 panic 禁止
/// 規約 `.claude/rules/coding-rust.md`）。
///
/// **空縮約（`n == 0`）は単位元 `1.0`**（PyTorch と同じ）を、`x` への
/// 計算グラフ依存を保ったまま返す（`empty_reduce_identity`。
/// `narrow(n-1)` の underflow を避けるため合成より前に分岐する）。
pub fn prod<'t>(x: &Var<'t>, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let out_shape = reduce_out_shape(&shape, dim)?;
    ensure_alloc_fits_f32(&shape, Some(&out_shape))?;
    let n = reduce_axis_len(&shape, dim);
    if n == 0 {
        return empty_reduce_identity(x, dim, 1.0);
    }
    match dim {
        None => {
            // `reshape` は非 contiguous view を `ShapeError::
            // NonContiguousReshape` で拒否する（`Var::reshape` の契約。
            // `var.rs`）。呼び出し元が転置・narrow 等の非 contiguous
            // view を渡しうるため、`dim: Some` 分岐と同様に `reshape`
            // 前に `contiguous()` で実体化する。
            let flat = x.contiguous()?.reshape(&[n])?;
            let cp = flat.cumprod(0)?;
            let last = cp.narrow(0, n - 1, 1)?;
            last.squeeze(Some(0))
        }
        Some(axis) => {
            let cp = x.cumprod(axis)?;
            let last = cp.narrow(axis, n - 1, 1)?;
            // `narrow` は `axis` が末尾軸でない限り非 contiguous な
            // stride view を返しうる。`squeeze`（`Var::reshape` への
            // 委譲）は contiguous 入力を要求する（`ShapeError::
            // NonContiguousReshape`）ため、`squeeze` 前に
            // `contiguous()` で実体化する（`matrix_ops::
            // diag_2d_to_1d` の `narrow → gather → squeeze` は
            // `gather` が暗黙に実体化するため同じ問題を踏まないが、
            // `prod` は `gather` を経由しないため明示的に挟む必要が
            // ある）。
            last.contiguous()?.squeeze(Some(axis))
        }
    }
}

/// log-sum-exp（`torch.logsumexp(dim)` 相当。イシュー #2147）。
/// `dim: None` は全軸縮約（スカラー）。専用 `Op::LogSumExp`
/// （`crate::tape::Op`）を直接構築する（`Var::norm`〈`var.rs`〉と同じ
/// フォールバック契約: `BackendOps::logsumexp` → `Unsupported` の
/// ときのみ `eval::logsumexp_along` へ切り替える）。
///
/// **空縮約（`n == 0`）は [`AutodiffError::InvalidArgument`]**
/// （`-inf` を黙って返さない安全側の判断。`Var::norm` と同じ方針）。
///
/// **確保前のバイト数上限検査（`ensure_alloc_fits_f32`。codex-review
/// P1 是正・イシュー #2147・PR #2263）**: `x` が小さなストレージを
/// 巨大な shape へ broadcast した view の場合、`materialize_one`
/// （`BackendOps::logsumexp` → `Unsupported` 時の
/// `eval::logsumexp_along` フォールバックを含む）や
/// `backend-cpu::reduction::logsumexp`（`axis_reduce_logsumexp`・
/// `gather_elements`）が要素数積・出力 shape から確保する `Vec` の
/// バイト数は、`n == 0` チェックだけでは検出できない（`n != 0` でも
/// `outer`／`inner` の broadcast 次元が巨大なら overflow しうる）。
/// `ensure_alloc_fits_f32` を関数冒頭（`n == 0` 判定より前）で呼び、
/// 入力 shape（`gather_elements`／`dense_vec` が実体化する側）・
/// `out_shape`（`axis_reduce_logsumexp`／`eval::logsumexp_along` の
/// `vec![0f32; outer * inner]` が確保する側）の両方を確保前に検査し、
/// `eval.rs` モジュール契約（「shape が既に整合していることを前提とし
/// `ShapeError` を返さない」）を保ったまま、本関数（呼び出し元）側で
/// `Result` として拒否する。
pub fn logsumexp<'t>(x: &Var<'t>, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let out_shape = reduce_out_shape(&shape, dim)?;
    ensure_alloc_fits_f32(&shape, Some(&out_shape))?;
    let n = reduce_axis_len(&shape, dim);
    if n == 0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "reduce_ops::logsumexp: 縮約対象の要素数が 0（dim={dim:?}）"
        )));
    }
    let input_val = materialize_one(x)?;
    let value = match x.tape().ops().logsumexp(&input_val, dim) {
        Ok(v) => {
            verify_shape(v.shape(), &out_shape)?;
            v
        }
        Err(BackendError::Unsupported(_)) => eval::logsumexp_along(&input_val, dim, &out_shape),
        Err(other) => return Err(unify_backend_error(other)),
    };
    let id = x.tape().push_eager(
        Op::LogSumExp {
            input: x.node_id(),
            dim,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}

/// `dim` 軸のいずれかが非ゼロなら `1.0`、それ以外は `0.0`（`torch.any`
/// 相当。イシュー #2147）。**出力は f32 の 0.0／1.0 マスク**（bool 版は
/// `crate::bool_ops` の対象。モジュール doc 参照）。`NaN != 0` は真の
/// ため `NaN` は真として扱う（PyTorch と一致）。`-0.0` は偽。
///
/// `x.ne(&zero) → max(dim)` の合成（新規 `Op` なし）。出力値は厳密に
/// `0.0`／`1.0` のみのため 3 バックエンドで bit 完全一致する。勾配は
/// `ne`（`ScalarBinaryOp::Ne`）の VJP がゼロを返すため、合成しただけで
/// 自動的に勾配ゼロの tape ノードになる。
///
/// **確保前のバイト数上限検査（`ensure_alloc_fits_f32`。`prod` と同じ
/// codex-review P1 是正・イシュー #2147・PR #2263）**: 空縮約
/// （`n == 0`）に限らず非空縮約でも、`x` が小さなストレージを巨大な
/// shape へ broadcast した view であれば `x.ne(&zero)` が入力 shape
/// 相応の `Vec` を無検査に確保しうる（対応前は空縮約分岐でしか検査
/// しておらず非空縮約が未検査のまま残っていた。codex-review 新規
/// 指摘・PR #2263）。関数冒頭で入力 shape・`out_shape` の双方を検査
/// する。
///
/// **空縮約（`n == 0`）は `0.0`**（PyTorch と同じ。`max` は単位元を
/// 持たずエラーになるため合成できず、`x.ne(&zero)` を `sum(dim)` へ
/// 通した単位元（`empty_reduce_identity`）で代替する。空縮約軸の
/// `sum` は `0.0` を返す契約〈モジュール doc「空縮約の意味論」〉の
/// ため、`max`/`min` の代わりに使っても値は変わらない）。
pub fn any<'t>(x: &Var<'t>, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let out_shape = reduce_out_shape(&shape, dim)?;
    ensure_alloc_fits_f32(&shape, Some(&out_shape))?;
    let n = reduce_axis_len(&shape, dim);
    let zero_val = Tensor::scalar(0.0f32);
    let zero_id = x.tape().push_leaf(zero_val, false);
    let zero = Var::from_raw(x.tape(), zero_id);
    if n == 0 {
        return empty_reduce_identity(&x.ne(&zero)?, dim, 0.0);
    }
    x.ne(&zero)?.max(dim)
}

/// `dim` 軸の全要素が非ゼロなら `1.0`、それ以外は `0.0`（`torch.all`
/// 相当。イシュー #2147）。[`any`] と対称（`x.ne(&zero) → min(dim)`）。
///
/// **確保前のバイト数上限検査**: [`any`] doc「確保前のバイト数上限
/// 検査」と同じ理由・同じタイミング（関数冒頭・`ensure_alloc_fits_f32`）。
///
/// **空縮約（`n == 0`）は `1.0`**（PyTorch と同じ。[`any`] と対称。
/// 空縮約軸の `sum` 単位元 `0.0` に `1.0` を加算する形で
/// `empty_reduce_identity` を使う）。
pub fn all<'t>(x: &Var<'t>, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let out_shape = reduce_out_shape(&shape, dim)?;
    ensure_alloc_fits_f32(&shape, Some(&out_shape))?;
    let n = reduce_axis_len(&shape, dim);
    let zero_val = Tensor::scalar(0.0f32);
    let zero_id = x.tape().push_leaf(zero_val, false);
    let zero = Var::from_raw(x.tape(), zero_id);
    if n == 0 {
        return empty_reduce_identity(&x.ne(&zero)?, dim, 1.0);
    }
    x.ne(&zero)?.min(dim)
}

/// p-ノルム（`torch.linalg.vector_norm(ord=p)` 相当。イシュー #2147）。
/// `dim: None` は全軸縮約（スカラー）。
///
/// `p` は有限かつ正のみ許容する（`NaN`・`±inf`・`0`・負の値は
/// [`AutodiffError::InvalidArgument`]。モジュール doc「PyTorch との
/// 差分」参照）。`p == 1.0`／`p == 2.0` は既存の
/// `Var::norm_l1`／`norm_l2`（`pub(crate)` の `Var::norm` 経由）へ
/// 委譲し、`norm_p(x, 2.0, d)` と `norm_l2(d)` が **bit 同一**になる。
///
/// それ以外の `p` は専用 `Op::PNorm` を直接構築する（`Var::norm` と
/// 同じフォールバック契約: `BackendOps::vector_norm_p` →
/// `Unsupported` のときのみ `eval::vector_norm_p_along` へ切り替え）。
///
/// **空縮約（`n == 0`）は [`AutodiffError::InvalidArgument`]**
/// （`Var::norm` と同じ方針）。
///
/// **確保前のバイト数上限検査（`ensure_alloc_fits_f32`。codex-review
/// P1 是正・イシュー #2147・PR #2263）**: [`logsumexp`] doc「確保前の
/// バイト数上限検査」と同じ理由で、関数冒頭・`p` の有限性／正値検査
/// よりも前に検査する（あらゆる分岐に先んじる、という本モジュールの
/// 統一契約〈モジュール doc「確保前のバイト数上限検査は全公開入口の
/// 冒頭で一律に行う契約」〉に従う）。とくに `p ∈ {1.0, 2.0}` の委譲
/// 判定より前に検査することが要点（対応前は委譲判定が検査より前に
/// あり、委譲先 `Var::norm_l1`／`norm_l2`〈`Var::norm` `pub(crate)`
/// 経由。本 PR の差分外・イシュー #1723 の既存経路〉が独自に確保前
/// 検査を持たない場合、本関数の検査を迂回してしまっていた。
/// codex-review 新規指摘・PR #2263。委譲そのもの〈`Var::norm_l1`／
/// `norm_l2`／`Var::norm` 本体〉の改修は既存 API のためスコープ外だが、
/// 委譲前に本関数が検査することで巨大 broadcast shape は委譲前に
/// 拒否される）。
pub fn norm_p<'t>(x: &Var<'t>, p: f32, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let out_shape = reduce_out_shape(&shape, dim)?;
    ensure_alloc_fits_f32(&shape, Some(&out_shape))?;
    if !p.is_finite() || p <= 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "reduce_ops::norm_p: p は有限かつ正である必要がある、got {p}"
        )));
    }
    if p == 1.0 {
        return x.norm_l1(dim);
    }
    if p == 2.0 {
        return x.norm_l2(dim);
    }
    let n = reduce_axis_len(&shape, dim);
    if n == 0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "reduce_ops::norm_p: 縮約対象の要素数が 0（dim={dim:?}）"
        )));
    }
    let input_val = materialize_one(x)?;
    let value = match x.tape().ops().vector_norm_p(&input_val, p, dim) {
        Ok(v) => {
            verify_shape(v.shape(), &out_shape)?;
            v
        }
        Err(BackendError::Unsupported(_)) => {
            eval::vector_norm_p_along(&input_val, p, dim, &out_shape)
        }
        Err(other) => return Err(unify_backend_error(other)),
    };
    let id = x.tape().push_eager(
        Op::PNorm {
            input: x.node_id(),
            p,
            dim,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    // --- prod ---

    #[test]
    fn prod_basic() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        let out = prod(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 24.0);
    }

    #[test]
    fn prod_with_one_zero_element() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 0.0, 3.0, 4.0], &[4]));
        let out = prod(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn prod_with_two_zero_elements() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 0.0, 0.0, 4.0], &[4]));
        let out = prod(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn prod_dim_axis() {
        let tape = Tape::new();
        // [[1,2],[3,4]] -> dim=0: [3, 8]
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = prod(&x, Some(0)).unwrap();
        assert_eq!(out.to_tensor().shape(), &[2]);
        assert_eq!(out.to_tensor().host_slice().into_owned(), vec![3.0, 8.0]);
    }

    #[test]
    fn prod_empty_reduction_is_one() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0]));
        let out = prod(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 1.0);
    }

    #[test]
    fn prod_dim_none_accepts_non_contiguous_input() {
        // `transpose` は非 contiguous な stride view を返す（イシュー
        // #2147 codex-review 指摘: `dim: None` 分岐が `reshape` 前の
        // `contiguous()` を欠き `ShapeError::NonContiguousReshape` で
        // 落ちていた）。転置後の `prod(None)` が全要素積を返せることを
        // 検証する。
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let xt = x.transpose(0, 1).unwrap();
        let out = prod(&xt, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 24.0);
    }

    #[test]
    fn prod_gradient_with_zero_matches_finite_difference() {
        let eps = 1e-3f32;
        let base = vec![2.0f32, 0.0, 3.0];
        let eval_fn = |data: &[f32]| -> f32 {
            let tape = Tape::new();
            let x = tape.var(&t(data.to_vec(), &[3]));
            let y = prod(&x, None).unwrap();
            y.to_tensor().host_slice()[0]
        };
        let tape = Tape::new();
        let x = tape.var(&t(base.clone(), &[3]));
        let y = prod(&x, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += eps;
            let mut minus = base.clone();
            minus[i] -= eps;
            let numeric = (eval_fn(&plus) - eval_fn(&minus)) / (2.0 * eps);
            assert!(
                (numeric - dx[i]).abs() < 1e-2,
                "prod 勾配の有限差分検算が乖離: i={i} numeric={numeric} analytic={}",
                dx[i]
            );
        }
    }

    // --- logsumexp ---

    #[test]
    fn logsumexp_basic_matches_naive() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let out = logsumexp(&x, None).unwrap();
        let expected = (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln() as f32;
        assert!((out.to_tensor().host_slice()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn logsumexp_large_values_does_not_overflow() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1e30, 1e30], &[2]));
        let out = logsumexp(&x, None).unwrap();
        let v = out.to_tensor().host_slice()[0];
        assert!(v.is_finite());
        // 2 要素とも 1e30 の logsumexp は 1e30 + ln(2) に極めて近い。
        assert!((v - 1e30).abs() < 1.0);
    }

    #[test]
    fn logsumexp_all_neg_inf_returns_neg_inf_and_zero_grad() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![f32::NEG_INFINITY, f32::NEG_INFINITY], &[2]));
        let out = logsumexp(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], f32::NEG_INFINITY);
        let grads = tape.backward(&out).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(dx.host_slice().into_owned(), vec![0.0, 0.0]);
    }

    #[test]
    fn logsumexp_with_pos_inf_is_pos_inf() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, f32::INFINITY], &[2]));
        let out = logsumexp(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], f32::INFINITY);
    }

    #[test]
    fn logsumexp_with_pos_inf_gradient_distributes_to_inf_elements() {
        // codex-review 指摘（イシュー #2147）: `+inf` を含む
        // `logsumexp` の勾配が `inf - inf = NaN` になっていた。修正後は
        // `+inf` 要素へ上流勾配を均等分配し、有限要素は 0 になる。
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, f32::INFINITY, 5.0, f32::INFINITY], &[4]));
        let out = logsumexp(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], f32::INFINITY);
        let grads = tape.backward(&out).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![0.0, 0.5, 0.0, 0.5]);
    }

    #[test]
    fn logsumexp_nan_propagates() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, f32::NAN], &[2]));
        let out = logsumexp(&x, None).unwrap();
        assert!(out.to_tensor().host_slice()[0].is_nan());
    }

    #[test]
    fn logsumexp_empty_reduction_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0]));
        assert!(matches!(
            logsumexp(&x, None),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn logsumexp_gradient_matches_finite_difference() {
        let eps = 1e-3f32;
        let base = vec![0.5f32, -1.2, 2.3];
        let eval_fn = |data: &[f32]| -> f32 {
            let tape = Tape::new();
            let x = tape.var(&t(data.to_vec(), &[3]));
            let y = logsumexp(&x, None).unwrap();
            y.to_tensor().host_slice()[0]
        };
        let tape = Tape::new();
        let x = tape.var(&t(base.clone(), &[3]));
        let y = logsumexp(&x, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += eps;
            let mut minus = base.clone();
            minus[i] -= eps;
            let numeric = (eval_fn(&plus) - eval_fn(&minus)) / (2.0 * eps);
            assert!(
                (numeric - dx[i]).abs() < 1e-2,
                "logsumexp 勾配の有限差分検算が乖離: i={i} numeric={numeric} analytic={}",
                dx[i]
            );
        }
    }

    // --- any / all ---

    #[test]
    fn any_true_when_one_nonzero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0, 0.0, 3.0], &[3]));
        let out = any(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 1.0);
    }

    #[test]
    fn any_false_when_all_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0, 0.0, 0.0], &[3]));
        let out = any(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn any_negative_zero_is_false() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![-0.0, -0.0], &[2]));
        let out = any(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn any_nan_is_true() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0, f32::NAN], &[2]));
        let out = any(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 1.0);
    }

    #[test]
    fn any_empty_reduction_is_false() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0]));
        let out = any(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn any_dim_axis() {
        let tape = Tape::new();
        // [[0,1],[0,0]] -> any(dim=1) = [1, 0]
        let x = tape.var(&t(vec![0.0, 1.0, 0.0, 0.0], &[2, 2]));
        let out = any(&x, Some(1)).unwrap();
        assert_eq!(out.to_tensor().host_slice().into_owned(), vec![1.0, 0.0]);
    }

    #[test]
    fn any_gradient_is_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let y = any(&x, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(dx.host_slice().into_owned(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn all_true_when_all_nonzero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let out = all(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 1.0);
    }

    #[test]
    fn all_false_when_one_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 0.0, 3.0], &[3]));
        let out = all(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn all_empty_reduction_is_true() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0]));
        let out = all(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 1.0);
    }

    #[test]
    fn all_gradient_is_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let y = all(&x, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(dx.host_slice().into_owned(), vec![0.0, 0.0, 0.0]);
    }

    // --- norm_p ---

    #[test]
    fn norm_p_matches_l1_for_p_one() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![-1.0, 2.0, -3.0], &[3]));
        let a = norm_p(&x, 1.0, None).unwrap();
        let b = x.norm_l1(None).unwrap();
        assert_eq!(
            a.to_tensor().host_slice()[0].to_bits(),
            b.to_tensor().host_slice()[0].to_bits()
        );
    }

    #[test]
    fn norm_p_matches_l2_for_p_two() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![3.0, 4.0], &[2]));
        let a = norm_p(&x, 2.0, None).unwrap();
        let b = x.norm_l2(None).unwrap();
        assert_eq!(
            a.to_tensor().host_slice()[0].to_bits(),
            b.to_tensor().host_slice()[0].to_bits()
        );
        assert_eq!(a.to_tensor().host_slice()[0], 5.0);
    }

    #[test]
    fn norm_p_three_matches_naive() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 2.0], &[3]));
        let out = norm_p(&x, 3.0, None).unwrap();
        let expected = (1.0f64 + 8.0 + 8.0).powf(1.0 / 3.0) as f32;
        assert!((out.to_tensor().host_slice()[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn norm_p_large_p_does_not_overflow() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1e38, 1.0], &[2]));
        let out = norm_p(&x, 50.0, None).unwrap();
        assert!(out.to_tensor().host_slice()[0].is_finite());
    }

    #[test]
    fn norm_p_zero_vector_is_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0, 0.0, 0.0], &[3]));
        let out = norm_p(&x, 3.0, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn norm_p_rejects_invalid_p() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        for p in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                matches!(norm_p(&x, p, None), Err(AutodiffError::InvalidArgument(_))),
                "p={p} は拒否されるはず"
            );
        }
    }

    #[test]
    fn norm_p_empty_reduction_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0]));
        assert!(matches!(
            norm_p(&x, 3.0, None),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn norm_p_zero_element_with_p_less_than_one_has_zero_grad_at_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0, 3.0], &[2]));
        let y = norm_p(&x, 0.5, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert!(dx.host_slice()[0].is_finite());
        assert_eq!(dx.host_slice()[0], 0.0);
    }

    #[test]
    fn norm_p_with_inf_gradient_distributes_to_inf_elements() {
        // codex-review 指摘（イシュー #2147）: `±inf` を含む p-norm の
        // VJP が `inf / inf = NaN` になっていた。修正後は `±inf` 要素へ
        // 符号付きで上流勾配を均等分配し、有限要素は 0 になる。
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, f32::INFINITY, 5.0, f32::NEG_INFINITY], &[4]));
        let out = norm_p(&x, 3.0, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], f32::INFINITY);
        let grads = tape.backward(&out).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![0.0, 0.5, 0.0, -0.5]);
    }

    #[test]
    fn norm_p_small_p_overflow_does_not_zero_finite_gradient() {
        // Cursor Bugbot（Medium）・Codex（P2）指摘（イシュー #2147）:
        // `pnorm_vjp` が `norm.is_infinite()` を「入力に ±inf を含む」
        // 判定にそのまま使っていたため、`±inf` を一切含まない有限入力
        // でも `p` が極小・有効要素数（ここでは同値の複数要素）が多いと
        // `mx * acc.powf(1/p)` が `f64` の範囲を超えてオーバーフローし、
        // release build では `inf_count == 0` の 0 除算相当で全要素の
        // 勾配が黙って `0.0` になっていた（debug build では
        // `debug_assert!` が panic）。修正後は `mx.is_infinite()` で
        // 「実際の ±inf 入力」と区別し、有限入力のオーバーフローは
        // `acc`（常に有限）を経由した log-domain で計算するため、有限
        // 入力の勾配が誤って 0 にならない。
        let tape = Tape::new();
        let x = tape.var(&t(vec![7.0, 7.0], &[2]));
        let p = 0.0005f32;
        let out = norm_p(&x, p, None).unwrap();
        // 極小 p × 複数の同値要素により forward 自体も f64 の範囲を
        // 超えてオーバーフローする（`docs/autodiff-reduce-ops-decision.md`
        // のスケール形計算でも避けられない、数学的に真に巨大な値）。
        assert!(out.to_tensor().host_slice()[0].is_infinite());
        let grads = tape.backward(&out).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        for &v in &dx {
            assert!(
                v > 0.0,
                "有限入力（±inf を含まない）の勾配が 0 のまま（旧欠陥の再発）: dx={dx:?}"
            );
            assert!(!v.is_nan(), "勾配が NaN になった: dx={dx:?}");
        }
    }

    #[test]
    fn norm_p_gradient_matches_finite_difference() {
        let eps = 1e-3f32;
        let base = vec![1.5f32, -2.5, 3.5];
        let eval_fn = |data: &[f32]| -> f32 {
            let tape = Tape::new();
            let x = tape.var(&t(data.to_vec(), &[3]));
            let y = norm_p(&x, 3.0, None).unwrap();
            y.to_tensor().host_slice()[0]
        };
        let tape = Tape::new();
        let x = tape.var(&t(base.clone(), &[3]));
        let y = norm_p(&x, 3.0, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += eps;
            let mut minus = base.clone();
            minus[i] -= eps;
            let numeric = (eval_fn(&plus) - eval_fn(&minus)) / (2.0 * eps);
            assert!(
                (numeric - dx[i]).abs() < 1e-2,
                "norm_p 勾配の有限差分検算が乖離: i={i} numeric={numeric} analytic={}",
                dx[i]
            );
        }
    }

    // --- 共通: dim 範囲外 ---

    #[test]
    fn out_of_range_dim_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(prod(&x, Some(5)).is_err());
        assert!(logsumexp(&x, Some(5)).is_err());
        assert!(any(&x, Some(5)).is_err());
        assert!(all(&x, Some(5)).is_err());
        assert!(norm_p(&x, 3.0, Some(5)).is_err());
    }
}
