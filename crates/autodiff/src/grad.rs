//! 演算ごとの勾配関数（VJP: vector-Jacobian product）と `Op` 単位の
//! ディスパッチ入口 `vjp()`。
//!
//! TASK-1.5a（#16）が記録したテープ構造（`tape::Op`/`TapeNode`）に対し、
//! 「出力側勾配（upstream）→ 各入力 `NodeId` への勾配」の変換層を提供
//! する（spec 根拠: `docs/spec/05-tasks.md` TASK-1.5、
//! `docs/spec/03-poc/poc-v2-2-autodiff/code/rust/src/tape.rs` の
//! backward 実装）。`Tape::backward`（`backward.rs`・TASK-1.5c・#18）は
//! ノード列を発生順とは逆順に走査しながら本モジュールの `vjp()` を
//! 呼び、返り値（入力 `NodeId` ごとの勾配寄与）を蓄積する。**勾配の
//! 蓄積そのものは本モジュールの責務ではない**（`backward.rs` 側で
//! 複数の出力先から同一入力ノードへ流入する勾配を合算する）。
//!
//! 値計算は `eval.rs`（クレート非公開の暫定 CPU 参照実装）のヘルパー
//! を再利用し、forward と勾配計算で数式の実体を 2 か所に別実装しない
//! （PoC-v2-2 の方針を踏襲。`eval::matmul` は `f32::mul_add` による
//! FMA 契約統一済みのため、`MatMul` の VJP もここを経由するだけで
//! 契約を引き継ぐ）。

use fandhe_ai_tensor_core::{BackendOps, Tensor};

use crate::error::AutodiffError;
use crate::eval::{self, build_tensor, dense_vec};
use crate::tape::{NodeId, Op, ResidentResolver, TapeNode, materialize_fallible};
use crate::var::Reduction;

/// ノード 1 個分の VJP。`upstream`（出力側勾配）と記録済みノード列
/// `nodes` から、各入力 `NodeId` への勾配寄与を返す。`out_value` は
/// 当該ノードの forward 記録値で、`Exp`/`Tanh`/`Sigmoid`/`Max` が
/// 再計算を避けて再利用する（`Sigmoid` は TASK-9.1b・#92 で追加）。
/// `Op::Leaf` は入力を持たないため空 `Vec` を返す。
///
/// **TASK-12.1d（#164）**: `Add`／`Sum` の入力 shape は `TapeNode.shape`
/// （実体化なしに算出済み。`tape.rs`）から直接読み、実体化を要求しない
/// （`docs/fusion-graph-design.md` §3.5.1）。`MatMul`／`Mul`／`Relu`／
/// `Max`／`MseLoss`／`CrossEntropyLoss` は入力の実際の値を要するため、
/// forward 記録済みの未実体化ノードを [`materialize_fallible`]（層 1。
/// `run_fused` の失敗のうち `Unsupported` 以外は `?` で伝播する）経由で
/// 読む（`Var::value`〈層 2〉は呼ばない。§3.5.2）。
///
/// `resident`（イシュー #1022）: `Op::LinearResident` の VJP が
/// `weight`／`bias` のデバイス常駐バッファを取得するための
/// [`ResidentResolver`]。素の [`crate::tape::Tape::backward`] からは
/// `None` が渡り、`DeviceParamStore::backward`（`optim::device_store`）
/// 経由の呼び出し（`Tape::backward_with_resident`）でのみ `Some` になる
/// （`tape::Op::LinearResident` doc「素の `Tape::backward`（resolver
/// なし）では型付きエラー」参照）。`Op::ResidentLeaf` 自身は `Op::Leaf`
/// と同じく入力を持たないため `resident` を参照しない。
pub(crate) fn vjp(
    op: &Op,
    out_value: &Tensor<f32>,
    upstream: &Tensor<f32>,
    nodes: &[TapeNode],
    ops: &dyn BackendOps,
    resident: Option<&dyn ResidentResolver>,
) -> Result<Vec<(NodeId, Tensor<f32>)>, AutodiffError> {
    // `Op` は `CrossEntropyLoss` の `targets: Tensor<i32>` payload
    // ゆえに `Copy` を持たない（`tape.rs::Op` doc 参照）。旧
    // `match *op`（`Copy` 前提の値コピー）を `op.clone()` に置き換え、
    // それ以外の分岐は変更しない。
    let contributions = match op.clone() {
        Op::Leaf => Vec::new(),
        Op::MatMul(a, b) => {
            let a_val = materialize_fallible(nodes, ops, a)?;
            let b_val = materialize_fallible(nodes, ops, b)?;
            let (da, db) = matmul_vjp(a_val, b_val, upstream);
            vec![(a, da), (b, db)]
        }
        Op::Add(a, b) => {
            let a_shape = &nodes[a.0].shape;
            let b_shape = &nodes[b.0].shape;
            let da = reduce_to_shape(upstream, a_shape);
            let db = reduce_to_shape(upstream, b_shape);
            vec![(a, da), (b, db)]
        }
        Op::Mul(a, b) => {
            let a_val = materialize_fallible(nodes, ops, a)?;
            let b_val = materialize_fallible(nodes, ops, b)?;
            let da = reduce_to_shape(&eval::mul(upstream, b_val), a_val.shape());
            let db = reduce_to_shape(&eval::mul(upstream, a_val), b_val.shape());
            vec![(a, da), (b, db)]
        }
        Op::Relu(a) => {
            // 劣勾配は x = 0 で 0 とする（PoC-v2-2 準拠）。NaN 入力は
            // マスク不成立（`v > 0.0` が false）となり勾配 0 を返す。
            let a_val = materialize_fallible(nodes, ops, a)?;
            let da = elementwise_mul_mask(upstream, a_val, |v| v > 0.0);
            vec![(a, da)]
        }
        Op::Exp(a) => {
            // d/dx exp(x) = exp(x)。forward 記録値 `out_value` を
            // 再利用し `exp` を再計算しない。
            let da = eval::mul(upstream, out_value);
            vec![(a, da)]
        }
        Op::Tanh(a) => {
            // d/dx tanh(x) = 1 - tanh(x)^2。同じく `out_value` を再利用。
            let factor = tanh_grad_factor(out_value);
            let da = eval::mul(upstream, &factor);
            vec![(a, da)]
        }
        Op::Sigmoid(a) => {
            // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))。
            // `Exp`/`Tanh` と同じく forward 記録値 `out_value`
            // （= sigmoid(x)）を再利用し再計算しない（TASK-9.1b・#92）。
            let factor = sigmoid_grad_factor(out_value);
            let da = eval::mul(upstream, &factor);
            vec![(a, da)]
        }
        Op::Sum { input, dim } => {
            let input_shape = &nodes[input.0].shape;
            let da = unreduce_broadcast(upstream, input_shape, dim);
            vec![(input, da)]
        }
        Op::Max { input, dim } => {
            let input_val = materialize_fallible(nodes, ops, input)?;
            let da = max_vjp(input_val, dim, out_value, upstream);
            vec![(input, da)]
        }
        Op::MseLoss {
            pred,
            target,
            reduction,
        } => {
            let pred_val = materialize_fallible(nodes, ops, pred)?;
            let target_val = materialize_fallible(nodes, ops, target)?;
            let (dpred, dtarget) = mse_loss_vjp(pred_val, target_val, upstream, reduction);
            vec![(pred, dpred), (target, dtarget)]
        }
        Op::CrossEntropyLoss {
            logits,
            targets,
            class_dim,
            reduction,
        } => {
            let logits_val = materialize_fallible(nodes, ops, logits)?;
            let dlogits =
                cross_entropy_loss_vjp(logits_val, &targets, class_dim, reduction, upstream);
            // `targets` は非追跡（`Var`/`NodeId` を持たない）ため勾配
            // 寄与を返すのは `logits` の 1 系統のみ（`tape::Op::
            // CrossEntropyLoss` doc 参照）。
            vec![(logits, dlogits)]
        }
        // デバイス常駐パラメータの葉（イシュー #1022）。`Op::Leaf` と同じく
        // 入力を持たないため寄与なし（`tape::Op::ResidentLeaf` doc 参照）。
        Op::ResidentLeaf { .. } => Vec::new(),
        // デバイス常駐 weight（・bias）で forward した Linear 相当ノード
        // （イシュー #1022）。`resident`（`ResidentResolver`）経由でしか
        // `weight` の `DeviceBuffer<f32>` を取得できないため、`None` の
        // 場合は型付きエラーで拒否する（`tape::Op::LinearResident` doc
        // 「素の `Tape::backward`（resolver なし）では型付きエラー」）。
        Op::LinearResident {
            input,
            weight,
            bias,
        } => {
            let Some(resident) = resident else {
                return Err(AutodiffError::InvalidArgument(
                    "grad::vjp: Op::LinearResident requires DeviceParamStore::backward (a plain \
                     Tape::backward cannot resolve the resident weight buffer)"
                        .to_string(),
                ));
            };
            // イシュー #1022 P1 是正（codex-review 指摘）: `weight`／
            // `bias` の `NodeId` は `DeviceParamStore::
            // register_resident_params`／`snapshot_resident_params` が発行した
            // `ResidentLeaf` から来るが、`ResidentLeaf` 自体はライフタイム
            // 引数のみで `Tape` の同一性を保証しない（`optim::device_store::
            // ResidentLeaf::tape_id` 検証は `linear_forward` 側の別途対応。
            // `optim/device_store.rs` モジュール冒頭参照）。ここでは
            // 縦深防御として `nodes[weight.0]` の直接添字アクセス（別
            // テープの葉が混入した場合に範囲外添字 panic・無関係ノード
            // 誤読の余地があった）を `nodes.get(...)` へ置き換え、
            // fail-closed に拒否する（`.claude/rules/security.md` A08）。
            let weight_node = nodes.get(weight.0).ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "grad::vjp: Op::LinearResident.weight node_id is out of range for this tape \
                     (contract violation: leaf registered on a different Tape?)"
                        .to_string(),
                )
            })?;
            let (store_id, slot) = match &weight_node.op {
                Op::ResidentLeaf { store_id, slot } => (*store_id, *slot),
                _ => {
                    return Err(AutodiffError::InvalidArgument(
                        "grad::vjp: Op::LinearResident.weight does not point to an \
                         Op::ResidentLeaf node (contract violation)"
                            .to_string(),
                    ));
                }
            };
            let w_dev = resident.resident_buffer(store_id, slot)?;
            let x_val = materialize_fallible(nodes, ops, input)?;

            // d_input^T = W @ g^T（`W: [k,n]`・`g: [m,n]` → `g^T: [n,m]`
            // → `tmp: [k,m]`）。`W` はデバイス常駐のまま
            // `ops.gemm_resident_lhs` へ渡し、ホストへ download しない
            // （本イシューの受け入れ条件の中核）。
            let g_t = transpose2d(upstream);
            let tmp = ops
                .gemm_resident_lhs(w_dev, &g_t)
                .map_err(AutodiffError::Backend)?;
            let d_input = transpose2d(&tmp);

            // d_weight = x^T @ g（既存 `matmul_vjp` の `dB` と同一式。
            // `x`・`g` はいずれもホスト常駐のため通常の `eval::matmul` で
            // 計算できる）。イシュー #1046: `x_t` は `transpose2d` の
            // zero-copy view であり、`eval::matmul` 側が
            // `layout::classify_2d` で分類して直接読み出すためホスト側
            // 転置コピーは発生しない（`eval::MATMUL_HOST_REPACK_COUNT`）。
            let x_t = transpose2d(x_val);
            let d_weight = eval::matmul(&x_t, upstream);

            let mut contributions = vec![(input, d_input), (weight, d_weight)];
            if let Some(bias_id) = bias {
                // bias の勾配は `Op::Add` の VJP と同じ縮約
                // （`reduce_to_shape`。行方向ブロードキャストの逆演算）。
                // `weight` と同じ理由で `nodes.get(...)` を経由し、
                // 範囲外添字 panic を防ぐ（fail-closed）。
                let bias_node = nodes.get(bias_id.0).ok_or_else(|| {
                    AutodiffError::InvalidArgument(
                        "grad::vjp: Op::LinearResident.bias node_id is out of range for this \
                         tape (contract violation: leaf registered on a different Tape?)"
                            .to_string(),
                    )
                })?;
                let d_bias = reduce_to_shape(upstream, &bias_node.shape);
                contributions.push((bias_id, d_bias));
            }
            contributions
        }
        // view ノード（イシュー #1047・親 #1043「カーネル融合・autodiff
        // 実行モデルの強化」）。`Reshape`/`Transpose` は逆写像も同じ演算
        // 族（reshape は「元の shape へ戻す」・transpose は対合）で
        // 表現でき、いずれも zero-copy（`Tensor::reshape`/`transpose`
        // が `storage: Arc<Storage>` を共有するのみ）。中間バッファを
        // 持たないという本イシューの受け入れ条件は、forward（`tape.rs::
        // resolve_view`）だけでなく backward（本 VJP）でも成立する。
        Op::Reshape { input } => {
            let input_shape = &nodes[input.0].shape;
            // `upstream` は out_shape（このノード自身の shape）を持つ。
            // 非 contiguous（例: 上流に `transpose` が挟まる）な場合は
            // zero-copy な `reshape` が `ShapeError::NonContiguousReshape`
            // を返しうるため、その場合に限り `contiguous()`（明示コピー）
            // を経由してから戻す（勾配バッファ側の話であり、view ノード
            // 自身が確保を持つわけではない。zero-copy を優先する順序を
            // 明記する）。
            let da = match upstream.reshape(input_shape) {
                Ok(t) => t,
                Err(_) => upstream.contiguous().reshape(input_shape).unwrap_or_else(|_| {
                    debug_assert!(
                        false,
                        "grad::vjp: Op::Reshape の逆伝播で reshape が失敗した（forward 側の契約違反）"
                    );
                    upstream.clone()
                }),
            };
            vec![(input, da)]
        }
        Op::Transpose { input, dim0, dim1 } => {
            // transpose は対合（同じ軸で 2 回適用すると恒等）のため、
            // 逆伝播も同じ `dim0`/`dim1` で `upstream` を transpose する
            // だけで閉じる（zero-copy。`tape::Op::Transpose` doc 参照）。
            let da = upstream.transpose(dim0, dim1).unwrap_or_else(|_| {
                debug_assert!(
                    false,
                    "grad::vjp: Op::Transpose の逆伝播で transpose が失敗した（forward 側の契約違反）"
                );
                upstream.clone()
            });
            vec![(input, da)]
        }
    };
    Ok(contributions)
}

/// 2 次元 `matmul` の転置。shape 検査は forward（`Var::matmul` →
/// `matmul_out_shape`）が済ませた 2 次元前提であり、`transpose(0, 1)`
/// は構造的に失敗しえない。それでも本番経路で `unwrap()`/`expect()`
/// を使わない方針（`.claude/rules/coding-rust.md`）のため、失敗時は
/// `debug_assert!` で契約違反を検知しつつ入力をそのまま返す
/// （到達すれば forward 側の shape 検査ロジックにバグがある）。
fn transpose2d(tensor: &Tensor<f32>) -> Tensor<f32> {
    match tensor.transpose(0, 1) {
        Ok(t) => t,
        Err(_) => {
            debug_assert!(
                false,
                "transpose2d: matmul VJP の rank-2 前提が崩れた（forward 側の契約違反）"
            );
            tensor.clone()
        }
    }
}

/// `MatMul(A, B)` の VJP: `dA = g @ Bᵀ`、`dB = Aᵀ @ g`
/// （`A: [m,k]`・`B: [k,n]`・`g: [m,n]`）。`eval::matmul` を再利用する
/// ため FMA 契約（`f32::mul_add`）は forward と自動的に統一される。
///
/// イシュー #1046: `transpose2d`（下記。`Tensor::transpose` の zero-copy
/// stride view）で作った転置オペランドは、`eval::matmul` 側
/// （`eval::matmul_operand`・`crate::layout::classify_2d`）が
/// `ld`／`transposed` フラグの添字式で直接読み出すため、本関数を含む
/// このホスト参照経路にホスト側転置コピー（`contiguous()` の repack）
/// は発生しない（`eval::MATMUL_HOST_REPACK_COUNT` で機械検証）。
fn matmul_vjp(a: &Tensor<f32>, b: &Tensor<f32>, g: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
    let b_t = transpose2d(b);
    let a_t = transpose2d(a);
    let da = eval::matmul(g, &b_t);
    let db = eval::matmul(&a_t, g);
    (da, db)
}

/// ブロードキャストの逆演算。`add`/`mul` の VJP が返す勾配は forward
/// 出力の shape（ブロードキャスト後）を持つため、元の入力 shape
/// （`target_shape`）へ縮約する必要がある。NumPy 風ブロードキャスト
/// は「先頭に新設された軸」と「入力側が size 1 だった軸」を複製する
/// ため、その逆演算は同じ軸集合を合計で潰せばよい（PoC-v2-2 の
/// `sum_rows_data` を任意 shape・任意軸へ一般化した実装）。
fn reduce_to_shape(g: &Tensor<f32>, target_shape: &[usize]) -> Tensor<f32> {
    let g_shape = g.shape().to_vec();
    if g_shape == target_shape {
        return g.clone();
    }
    debug_assert!(
        g_shape.len() >= target_shape.len(),
        "reduce_to_shape: broadcast 後 shape の rank は入力 rank 以上のはず（契約違反）"
    );
    let rank_diff = g_shape.len() - target_shape.len();
    let mut padded_target = vec![1usize; rank_diff];
    padded_target.extend_from_slice(target_shape);

    let mut data = dense_vec(g);
    let mut cur_shape = g_shape;
    for axis in 0..cur_shape.len() {
        if padded_target[axis] == 1 && cur_shape[axis] != 1 {
            let outer: usize = cur_shape[..axis].iter().product();
            let axis_len = cur_shape[axis];
            let inner: usize = cur_shape[axis + 1..].iter().product();
            let mut reduced = vec![0f32; outer * inner];
            for o in 0..outer {
                for a in 0..axis_len {
                    for i in 0..inner {
                        let src = (o * axis_len + a) * inner + i;
                        reduced[o * inner + i] += data[src];
                    }
                }
            }
            data = reduced;
            cur_shape[axis] = 1;
        }
    }
    build_tensor(data, target_shape)
}

/// 同 shape の 2 テンソルに対する要素ごとの条件付き選択
/// （`g` をそのまま通すか 0 にするかを `mask_src` の値で決める）。
/// `Relu` の VJP（`g ⊙ 1[x > 0]`）専用の最小実装。
fn elementwise_mul_mask(
    g: &Tensor<f32>,
    mask_src: &Tensor<f32>,
    keep: impl Fn(f32) -> bool,
) -> Tensor<f32> {
    let shape = g.shape().to_vec();
    let g_data = dense_vec(g);
    let mask_data = dense_vec(mask_src);
    let out: Vec<f32> = g_data
        .iter()
        .zip(mask_data.iter())
        .map(|(&gv, &mv)| if keep(mv) { gv } else { 0.0 })
        .collect();
    build_tensor(out, &shape)
}

/// `Tanh` の VJP 係数 `1 - tanh(x)^2` を forward 記録値 `out_value`
/// （= `tanh(x)`）から計算する（再計算を避ける）。
fn tanh_grad_factor(out_value: &Tensor<f32>) -> Tensor<f32> {
    let shape = out_value.shape().to_vec();
    let data = dense_vec(out_value);
    let out: Vec<f32> = data.iter().map(|&v| 1.0 - v * v).collect();
    build_tensor(out, &shape)
}

/// `Sigmoid` の VJP 係数 `sigmoid(x) * (1 - sigmoid(x))` を forward
/// 記録値 `out_value`（= `sigmoid(x)`）から計算する（TASK-9.1b・#92。
/// `tanh_grad_factor` と同型の out_value 再利用パターン）。
fn sigmoid_grad_factor(out_value: &Tensor<f32>) -> Tensor<f32> {
    let shape = out_value.shape().to_vec();
    let data = dense_vec(out_value);
    let out: Vec<f32> = data.iter().map(|&v| v * (1.0 - v)).collect();
    build_tensor(out, &shape)
}

/// `Sum` の VJP: 出力側勾配 `g` を入力 shape へブロードキャストして
/// 複製する（`sum` の逆演算は複製、`reduce_out_shape` が縮約軸を
/// 除去済み〈keepdim なし〉のため、`dim: Some(axis)` はいったん
/// size-1 軸を挿入してから `broadcast_to` する）。
fn unreduce_broadcast(g: &Tensor<f32>, input_shape: &[usize], dim: Option<usize>) -> Tensor<f32> {
    match dim {
        None => {
            let value = dense_vec(g).first().copied().unwrap_or(0.0);
            match Tensor::full(input_shape, value) {
                Ok(t) => t,
                Err(_) => {
                    debug_assert!(
                        false,
                        "unreduce_broadcast: dim=None の full() 構築が失敗した（契約違反）"
                    );
                    g.clone()
                }
            }
        }
        Some(axis) => {
            let mut inserted_shape = g.shape().to_vec();
            inserted_shape.insert(axis, 1);
            let reshaped = match g.contiguous().reshape(&inserted_shape) {
                Ok(t) => t,
                Err(_) => {
                    debug_assert!(
                        false,
                        "unreduce_broadcast: reduce_out_shape 逆算の reshape が失敗した（契約違反）"
                    );
                    return g.clone();
                }
            };
            match reshaped.broadcast_to(input_shape) {
                Ok(t) => t.contiguous(),
                Err(_) => {
                    debug_assert!(
                        false,
                        "unreduce_broadcast: 挿入軸からの broadcast_to が失敗した（契約違反）"
                    );
                    reshaped
                }
            }
        }
    }
}

/// `Max` の VJP: 出力側勾配 `g` を、縮約軸に沿った最大値の位置のみへ
/// 伝播する。**同値タイは「最初に現れる最大要素 1 箇所のみ」へ伝播
/// する**（PyTorch `amax` の均等分配とは異なる、決定的な選択。
/// PoC-v2-2 のビット一致決定性方針・`train_repro` と整合させるための
/// 設計判断）。`out_value` は forward 記録済みの縮約後最大値で、走査中
/// に現れる要素と exact 一致するかで argmax 位置を判定する（同一デー
/// タ・同一 reduction 経路のため bit 一致する）。
///
/// Issue #224（先勝ち挙動の再確認。compat 層〈REQ-9〉実装時に要再確認
/// としていた事項）の結論: **本挙動を維持する（変更なし）**。
/// compat 層（TASK-9.2a・#95 で実装。TASK-9.4・#411 で `fandhe_ai::compat`
/// へ移設済み）の公開面は `array()`／`Sequential`（Linear・ReLU・
/// Sigmoid・Tanh）に限定され（`docs/compat-api-scope.md` §1〜2）、
/// `max`/`amax` 相当 API が存在しないため PyTorch 互換を要求する利用者
/// 向け経路が現時点でない。均等分配へ変更すると勾配値そのものが変わり
/// 上記の決定性方針と衝突するため、先勝ちを維持する。再検討条件:
/// `fandhe_ai::compat`（REQ-9 追記・#52）の公開面に `amax` 相当の縮約 API を
/// 追加する段階になった場合にのみ PyTorch 互換の要否を改めて判断する
/// （`docs/compat-api-scope.md` にも記録）。
fn max_vjp(
    input: &Tensor<f32>,
    dim: Option<usize>,
    out_value: &Tensor<f32>,
    g: &Tensor<f32>,
) -> Tensor<f32> {
    let in_shape = input.shape().to_vec();
    let in_data = dense_vec(input);
    let g_data = dense_vec(g);
    let out_data = dense_vec(out_value);
    let mut grad = vec![0f32; in_data.len()];
    match dim {
        None => {
            if let (Some(target), Some(gv)) = (out_data.first(), g_data.first())
                && let Some(idx) = in_data.iter().position(|&v| v == *target)
            {
                grad[idx] = *gv;
            }
        }
        Some(axis) => {
            let outer: usize = in_shape[..axis].iter().product();
            let axis_len = in_shape[axis];
            let inner: usize = in_shape[axis + 1..].iter().product();
            for o in 0..outer {
                for i in 0..inner {
                    let out_idx = o * inner + i;
                    // `out_data`/`g_data` の要素数は `reduce_out_shape`
                    // の契約上 `outer * inner` と一致するはずだが、
                    // 本ファイルの他ヘルパー（`transpose2d`／
                    // `unreduce_broadcast` 等）と同様、契約違反時に
                    // release ビルドで境界外アクセス panic させず
                    // `debug_assert!` で検知しつつ安全側（当該要素の
                    // 勾配は 0 のまま）へフォールバックする
                    // （coding-rust.md「本番経路で unwrap/expect を
                    // 使わない」方針の趣旨に揃える）。
                    let (Some(&target), Some(&g_val)) =
                        (out_data.get(out_idx), g_data.get(out_idx))
                    else {
                        debug_assert!(
                            false,
                            "max_vjp: out_value/g の要素数が reduce_out_shape の想定と不一致（契約違反）"
                        );
                        continue;
                    };
                    for a in 0..axis_len {
                        let src = (o * axis_len + a) * inner + i;
                        match in_data.get(src) {
                            Some(&v) if v == target => {
                                grad[src] = g_val;
                                break;
                            }
                            Some(_) => {}
                            None => {
                                debug_assert!(
                                    false,
                                    "max_vjp: input の要素数が in_shape と不一致（契約違反）"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    build_tensor(grad, &in_shape)
}

/// `MseLoss{pred, target, reduction}` の VJP: `dPred = g · 2(pred −
/// target) / n`（mean）／`g · 2(pred − target)`（sum）、
/// `dTarget = −dPred`（`g` はスカラー上流勾配）。#190 で sum 縮約を
/// 追加（`reduction` 分岐は forward の `eval::mse_loss` と対称）。
/// `n == 0` は mean・sum ともゼロ除算を避け zeros を返す。PoC-v2-2 は
/// `target` 側の勾配計算をスキップしていたが、本実装は数学的に完全な
/// VJP を返す（`target` 側を使うか捨てるかは #18 の勾配蓄積側の責務で
/// あり、ここでは両方提供する）。
fn mse_loss_vjp(
    pred: &Tensor<f32>,
    target: &Tensor<f32>,
    g: &Tensor<f32>,
    reduction: Reduction,
) -> (Tensor<f32>, Tensor<f32>) {
    let shape = pred.shape().to_vec();
    let n = pred.numel();
    if n == 0 {
        let zeros = build_tensor(vec![0f32; 0], &shape);
        return (zeros.clone(), zeros);
    }
    let g_value = dense_vec(g).first().copied().unwrap_or(0.0);
    let pred_data = dense_vec(pred);
    let target_data = dense_vec(target);
    let scale = match reduction {
        Reduction::Mean => g_value * 2.0 / n as f32,
        Reduction::Sum => g_value * 2.0,
    };
    let dpred_data: Vec<f32> = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &t)| scale * (p - t))
        .collect();
    let dtarget_data: Vec<f32> = dpred_data.iter().map(|&v| -v).collect();
    let dpred = build_tensor(dpred_data, &shape);
    let dtarget = build_tensor(dtarget_data, &shape);
    (dpred, dtarget)
}

/// `CrossEntropyLoss(logits, targets)` の VJP:
/// `d loss / d logits[..., c, ...] = (softmax(logits)[..., c, ...] − 1{c == t}) × g`
/// （`g` はサンプルごとのスカラー係数。`Mean` は `g = upstream / N`、
/// `Sum` は `g = upstream`。`N` はサンプル数 `= targets.numel()`）。
/// `eval::softmax_along` を再利用し、forward（`eval::cross_entropy_loss`
/// の log-sum-exp）と数式の実体を分離しない（`grad.rs` 冒頭 doc）。
/// `targets` は非追跡のため戻り値は `logits` 側の勾配のみ（呼び出し元
/// `vjp()` の `CrossEntropyLoss` 分岐参照）。
fn cross_entropy_loss_vjp(
    logits: &Tensor<f32>,
    targets: &Tensor<i32>,
    class_dim: usize,
    reduction: Reduction,
    upstream: &Tensor<f32>,
) -> Tensor<f32> {
    let shape = logits.shape().to_vec();
    let outer: usize = shape[..class_dim].iter().product();
    let axis_len = shape[class_dim];
    let inner: usize = shape[class_dim + 1..].iter().product();
    let n = outer * inner;

    let softmax = eval::softmax_along(logits, class_dim);
    let mut grad = dense_vec(&softmax);
    let target_data = eval::dense_vec_i32(targets);

    let g_value = dense_vec(upstream).first().copied().unwrap_or(0.0);
    let scale = match reduction {
        Reduction::Mean if n > 0 => g_value / n as f32,
        Reduction::Mean => 0.0,
        Reduction::Sum => g_value,
    };

    for o in 0..outer {
        for i in 0..inner {
            let t = target_data[o * inner + i];
            // forward（`Var::cross_entropy_loss`）が事前検査済みの前提
            // （`0 <= t < axis_len`）。範囲外は契約違反であり
            // `debug_assert!` で検知しつつ onehot 減算をスキップする
            // 安全側フォールバック（`eval::cross_entropy_loss` と同型の
            // 契約違反対応）。
            if t >= 0 && (t as usize) < axis_len {
                let idx = (o * axis_len + t as usize) * inner + i;
                grad[idx] -= 1.0;
            } else {
                debug_assert!(
                    false,
                    "cross_entropy_loss_vjp: target 添字が範囲外（契約違反）"
                );
            }
        }
    }
    let scaled: Vec<f32> = grad.iter().map(|&v| v * scale).collect();
    build_tensor(scaled, &shape)
}

#[cfg(test)]
mod tests {
    //! 受け入れ条件「各演算の解析勾配が数値微分と一致する」の直接検証。
    //!
    //! 各演算について、固定の重みテンソル `s`（forward 出力と同じ
    //! shape）によるスカラー射影 `L(x) = Σ (op(x) ⊙ s)` を定義し、
    //! 解析側（`vjp` 経由）と数値側（中央差分。f64 で集計）を突合する。
    //! `Tape`/`Var` を経由せず `eval.rs` の値計算を直接叩くため、
    //! `Tape::backward`（#18）が未実装でも検証できる。
    //!
    //! **判定基準**: 要素ごとに「相対誤差
    //! `|ad − num| / max(|ad|, |num|, τ)` が 1e-2 以下」または
    //! 「絶対誤差が 1e-3 以下」（`h = 1e-3`・`τ = 1e-4`）。PoC-v2-2 の
    //! 1e-4 は f64 前提（`torch.autograd.gradcheck` と同じ理由で f64
    //! 必須と PoC 自身が明記）だが、本実装は f32 のため中央差分の
    //! 丸め誤差床 `≈ ε_f32 · |L| / h ≈ 1e-4` を踏まえた本イシュー
    //! 新規の grad-check 専用閾値とする（バックエンド間数値一致判定
    //! 〈相対 1e-3 / 絶対 1e-5〉とは別系統）。
    //!
    //! **承認記録（#223・承認済み）**: `CLAUDE.md` Conventions・
    //! `.claude/rules/delegation-impl.md` の「テスト許容誤差の変更は
    //! ユーザー承認必須」規定に基づく本閾値（新規 grad-check 専用
    //! `REL_TOL`/`ABS_TOL`）の承認は完了している。
    //! - 承認者: ユーザー／承認日: 2026-08-09
    //! - 承認記録: <https://github.com/Fandhe-AI/fandhe-ai/issues/223#issuecomment-5230026874>
    //! - 判断材料: 全 grad-check テストの実測誤差マージン採取で、
    //!   最も僅差のケースでも絶対誤差側に約 3.4 倍の余裕
    //!   （実測 `diff ≈ 2.9e-4` に対し `ABS_TOL = 1e-3`）を確認
    //!
    //! 値（`REL_TOL`/`ABS_TOL`/`TAU`/`H`）の変更が必要になった場合は
    //! 改めてユーザー承認が必須であり、#223 系譜の新規 Issue で追跡する。
    //!
    //! **キンク・タイ回避**: ReLU は `|x| >= 10h` の固定入力のみ、
    //! Max は同値タイのない固定入力のみを使う（固定値のため再生成
    //! ガードは不要。PoC-v2-2 `grad_check.rs` と同方針）。

    use super::*;

    const H: f64 = 1e-3;
    const TAU: f32 = 1e-4;
    const REL_TOL: f32 = 1e-2;
    const ABS_TOL: f32 = 1e-3;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data.to_vec(), shape)
            .expect("test fixture: shape とデータ長は事前に一致させている")
    }

    fn assert_grad_close(label: &str, analytic: &Tensor<f32>, numeric: &Tensor<f32>) {
        let a = dense_vec(analytic);
        let n = dense_vec(numeric);
        assert_eq!(
            a.len(),
            n.len(),
            "{label}: analytic/numeric の要素数が一致しない"
        );
        for (i, (&av, &nv)) in a.iter().zip(n.iter()).enumerate() {
            let diff = (av - nv).abs();
            let rel = diff / av.abs().max(nv.abs()).max(TAU);
            assert!(
                rel <= REL_TOL || diff <= ABS_TOL,
                "{label}[{i}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
            );
        }
    }

    /// `L(x) = Σ (forward(x) ⊙ s)` の f64 集計によるスカラー射影値。
    fn scalar_dot(a: &Tensor<f32>, s: &Tensor<f32>) -> f64 {
        dense_vec(a)
            .iter()
            .zip(dense_vec(s).iter())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum()
    }

    /// 単一入力 `x` に対する `L` の中央差分勾配（要素ごと・f64 集計）。
    fn numeric_grad_unary(
        x: &Tensor<f32>,
        s: &Tensor<f32>,
        forward: impl Fn(&Tensor<f32>) -> Tensor<f32>,
    ) -> Tensor<f32> {
        let shape = x.shape().to_vec();
        let mut data = dense_vec(x);
        let mut grad = vec![0f32; data.len()];
        for i in 0..data.len() {
            let orig = data[i] as f64;
            data[i] = (orig + H) as f32;
            let lp = scalar_dot(&forward(&build_tensor(data.clone(), &shape)), s);
            data[i] = (orig - H) as f32;
            let lm = scalar_dot(&forward(&build_tensor(data.clone(), &shape)), s);
            data[i] = orig as f32;
            grad[i] = ((lp - lm) / (2.0 * H)) as f32;
        }
        build_tensor(grad, &shape)
    }

    // --- MatMul ---

    #[test]
    fn matmul_grad_matches_numeric() {
        let a = t(&[1.0, 2.0, -1.0, 0.5, 3.0, -2.0], &[2, 3]);
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[3, 2]);
        let s = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);

        let g = s.clone();
        let (da, db) = matmul_vjp(&a, &b, &g);

        let num_da = numeric_grad_unary(&a, &s, |x| eval::matmul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::matmul(&a, x));

        assert_grad_close("matmul dA", &da, &num_da);
        assert_grad_close("matmul dB", &db, &num_db);
    }

    /// イシュー #1046 受け入れ条件 (a) の機械検証: `matmul_vjp` は
    /// `transpose2d`（zero-copy view）で作った転置オペランドを
    /// `eval::matmul` へ渡すが、`eval::matmul_operand` が
    /// `layout::classify_2d` で分類して直接読み出すため、ホスト側
    /// 転置コピー（`eval::MATMUL_HOST_REPACK_COUNT`）が発生しない
    /// ことを確認する。
    #[test]
    fn matmul_vjp_does_not_repack_transposed_operands() {
        let a = t(&[1.0, 2.0, -1.0, 0.5, 3.0, -2.0], &[2, 3]);
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[3, 2]);
        let g = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);

        let before = eval::MATMUL_HOST_REPACK_COUNT.with(|c| c.get());
        let _ = matmul_vjp(&a, &b, &g);
        let after = eval::MATMUL_HOST_REPACK_COUNT.with(|c| c.get());

        assert_eq!(
            before, after,
            "matmul_vjp: 転置オペランド（transpose2d の zero-copy view）が \
             eval::matmul でホスト側転置コピーへフォールバックした \
             （MATMUL_HOST_REPACK_COUNT が増加した）"
        );
    }

    // --- Add（同 shape・bias broadcast・スカラー broadcast） ---

    #[test]
    fn add_grad_same_shape_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&s, a.shape());
        let db = reduce_to_shape(&s, b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::add(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("add(same) dA", &da, &num_da);
        assert_grad_close("add(same) dB", &db, &num_db);
    }

    #[test]
    fn add_grad_bias_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let b = t(&[0.5, 1.5, -1.0], &[3]);
        let s = t(&[1.0, -1.0, 2.0, 0.5, 1.0, -0.5], &[2, 3]);

        let da = reduce_to_shape(&s, a.shape());
        let db = reduce_to_shape(&s, b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::add(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("add(bias) dA", &da, &num_da);
        assert_grad_close("add(bias) dB", &db, &num_db);
    }

    #[test]
    fn add_grad_scalar_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[2.0], &[]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&s, a.shape());
        let db = reduce_to_shape(&s, b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::add(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("add(scalar) dA", &da, &num_da);
        assert_grad_close("add(scalar) dB", &db, &num_db);
    }

    // --- Mul（同 shape・bias broadcast・スカラー broadcast） ---

    #[test]
    fn mul_grad_same_shape_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&eval::mul(&s, &b), a.shape());
        let db = reduce_to_shape(&eval::mul(&s, &a), b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::mul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::mul(&a, x));

        assert_grad_close("mul(same) dA", &da, &num_da);
        assert_grad_close("mul(same) dB", &db, &num_db);
    }

    #[test]
    fn mul_grad_bias_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let b = t(&[0.5, 1.5, -1.0], &[3]);
        let s = t(&[1.0, -1.0, 2.0, 0.5, 1.0, -0.5], &[2, 3]);

        let da = reduce_to_shape(&eval::mul(&s, &b), a.shape());
        let db = reduce_to_shape(&eval::mul(&s, &a), b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::mul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::mul(&a, x));

        assert_grad_close("mul(bias) dA", &da, &num_da);
        assert_grad_close("mul(bias) dB", &db, &num_db);
    }

    #[test]
    fn mul_grad_scalar_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[2.0], &[]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&eval::mul(&s, &b), a.shape());
        let db = reduce_to_shape(&eval::mul(&s, &a), b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::mul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::mul(&a, x));

        assert_grad_close("mul(scalar) dA", &da, &num_da);
        assert_grad_close("mul(scalar) dB", &db, &num_db);
    }

    // --- Relu（正負混在。|x| >= 10h でキンク回避） ---

    #[test]
    fn relu_grad_matches_numeric() {
        let a = t(&[2.0, -3.0, 0.5, -0.02, 1.5, -1.5], &[2, 3]);
        let s = t(&[1.0, -1.0, 2.0, 0.5, -0.5, 1.0], &[2, 3]);

        let g = s.clone();
        let da = elementwise_mul_mask(&g, &a, |v| v > 0.0);
        let num_da = numeric_grad_unary(&a, &s, eval::relu);

        assert_grad_close("relu dA", &da, &num_da);
    }

    #[test]
    fn relu_subgradient_at_zero_is_zero() {
        // x = 0 における劣勾配は 0 とする（PoC-v2-2 準拠。中央差分は
        // キンクで数値的に不安定なため、ここは解析式の直接検証のみ）。
        let a = t(&[0.0], &[1]);
        let g = t(&[3.0], &[1]);
        let da = elementwise_mul_mask(&g, &a, |v| v > 0.0);
        assert_eq!(dense_vec(&da), vec![0.0]);
    }

    // --- Exp ---

    #[test]
    fn exp_grad_matches_numeric() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::exp(&a);
        let g = s.clone();
        let da = eval::mul(&g, &out_value);
        let num_da = numeric_grad_unary(&a, &s, eval::exp);

        assert_grad_close("exp dA", &da, &num_da);
    }

    // --- Tanh ---

    #[test]
    fn tanh_grad_matches_numeric() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::tanh(&a);
        let g = s.clone();
        let factor = tanh_grad_factor(&out_value);
        let da = eval::mul(&g, &factor);
        let num_da = numeric_grad_unary(&a, &s, eval::tanh);

        assert_grad_close("tanh dA", &da, &num_da);
    }

    // --- Sigmoid（飽和域を含む） ---

    #[test]
    fn sigmoid_grad_matches_numeric() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::sigmoid(&a);
        let g = s.clone();
        let factor = sigmoid_grad_factor(&out_value);
        let da = eval::mul(&g, &factor);
        let num_da = numeric_grad_unary(&a, &s, eval::sigmoid);

        assert_grad_close("sigmoid dA", &da, &num_da);
    }

    #[test]
    fn sigmoid_grad_saturated_region_matches_numeric() {
        // |x| が大きい飽和域（勾配 ≈ 0）でも中央差分と一致することを
        // 確認する（`eval::sigmoid` の数値安定形が飽和域で NaN/Inf を
        // 出さないことの間接検証も兼ねる）。
        let a = t(&[8.0, -8.0, 15.0, -15.0], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::sigmoid(&a);
        let g = s.clone();
        let factor = sigmoid_grad_factor(&out_value);
        let da = eval::mul(&g, &factor);
        let num_da = numeric_grad_unary(&a, &s, eval::sigmoid);

        assert_grad_close("sigmoid(saturated) dA", &da, &num_da);
    }

    // --- Sum（dim: None / Some(0) / Some(1)） ---

    #[test]
    fn sum_grad_dim_none_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[2.0], &[]);

        let g = s.clone();
        let da = unreduce_broadcast(&g, a.shape(), None);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::sum(x, None, &[]));

        assert_grad_close("sum(None) dA", &da, &num_da);
    }

    #[test]
    fn sum_grad_dim_0_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0, 2.0], &[3]);

        let g = s.clone();
        let da = unreduce_broadcast(&g, a.shape(), Some(0));
        let num_da = numeric_grad_unary(&a, &s, |x| eval::sum(x, Some(0), &[3]));

        assert_grad_close("sum(dim=0) dA", &da, &num_da);
    }

    #[test]
    fn sum_grad_dim_1_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0], &[2]);

        let g = s.clone();
        let da = unreduce_broadcast(&g, a.shape(), Some(1));
        let num_da = numeric_grad_unary(&a, &s, |x| eval::sum(x, Some(1), &[2]));

        assert_grad_close("sum(dim=1) dA", &da, &num_da);
    }

    // --- Max（dim: None / Some(0) / Some(1)。同値タイなし） ---

    #[test]
    fn max_grad_dim_none_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[2.0], &[]);

        let out_value = eval::max(&a, None, &[]);
        let g = s.clone();
        let da = max_vjp(&a, None, &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::max(x, None, &[]));

        assert_grad_close("max(None) dA", &da, &num_da);
    }

    #[test]
    fn max_grad_dim_0_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0, 2.0], &[3]);

        let out_value = eval::max(&a, Some(0), &[3]);
        let g = s.clone();
        let da = max_vjp(&a, Some(0), &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::max(x, Some(0), &[3]));

        assert_grad_close("max(dim=0) dA", &da, &num_da);
    }

    #[test]
    fn max_grad_dim_1_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0], &[2]);

        let out_value = eval::max(&a, Some(1), &[2]);
        let g = s.clone();
        let da = max_vjp(&a, Some(1), &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::max(x, Some(1), &[2]));

        assert_grad_close("max(dim=1) dA", &da, &num_da);
    }

    // --- Max（同値タイ。#224: 先勝ち決定的挙動の回帰固定） ---
    //
    // タイ発生時は最大値位置が複数あり数値微分（中央差分）が定義でき
    // ないため、上記の同値タイなしケースとは異なり厳密値アサーション
    // （数値微分比較なし）で「最初に現れる最大要素 1 箇所のみに勾配が
    // 伝播し、他はゼロになる」先勝ち挙動そのものを固定する。

    #[test]
    fn max_grad_dim_none_tie_first_wins() {
        // 最大値 5.0 がインデックス 1・3 の 2 箇所に現れるタイケース。
        let a = t(&[1.0, 5.0, 3.0, 5.0], &[4]);
        let g = t(&[2.0], &[]);

        let out_value = eval::max(&a, None, &[]);
        let da = max_vjp(&a, None, &out_value, &g);
        let grad = dense_vec(&da);

        assert_eq!(
            grad,
            vec![0.0, 2.0, 0.0, 0.0],
            "max(None) タイ時は最初に現れる最大要素（idx=1）のみへ伝播するはず"
        );
        // 勾配総量が上流勾配 g と一致すること（先勝ちでも保存量は保たれる）。
        assert_eq!(grad.iter().sum::<f32>(), 2.0);
    }

    #[test]
    fn max_grad_dim_axis_tie_first_wins() {
        // shape [2, 3]。行 0 は列 0・2 が 5.0 でタイ、行 1 はタイなし。
        let a = t(&[5.0, 1.0, 5.0, 1.0, -2.0, 4.0], &[2, 3]);
        let g = t(&[3.0, 7.0], &[2]);

        let out_value = eval::max(&a, Some(1), &[2]);
        let da = max_vjp(&a, Some(1), &out_value, &g);
        let grad = dense_vec(&da);

        assert_eq!(
            grad,
            vec![3.0, 0.0, 0.0, 0.0, 0.0, 7.0],
            "max(dim=1) タイ行（行 0）は軸方向で最初の最大要素（列 0）のみへ伝播するはず"
        );
        // 各 (outer) スライスごとに勾配総量が上流勾配 g[outer] と一致すること。
        assert_eq!(grad[0..3].iter().sum::<f32>(), 3.0);
        assert_eq!(grad[3..6].iter().sum::<f32>(), 7.0);
    }

    // --- MseLoss（pred/target 両勾配） ---

    #[test]
    fn mse_loss_grad_mean_matches_numeric() {
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let s = t(&[3.0], &[]);

        let g = s.clone();
        let (dpred, dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Mean);
        let num_dpred =
            numeric_grad_unary(&pred, &s, |x| eval::mse_loss(x, &target, Reduction::Mean));
        let num_dtarget =
            numeric_grad_unary(&target, &s, |x| eval::mse_loss(&pred, x, Reduction::Mean));

        assert_grad_close("mse(mean) dPred", &dpred, &num_dpred);
        assert_grad_close("mse(mean) dTarget", &dtarget, &num_dtarget);
    }

    #[test]
    fn mse_loss_grad_sum_matches_numeric() {
        // sum 縮約（#190）。scale が `2/n` ではなく `2` になる分岐を
        // mean と同じ数値微分ハーネスで検証する。
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let s = t(&[3.0], &[]);

        let g = s.clone();
        let (dpred, dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Sum);
        let num_dpred =
            numeric_grad_unary(&pred, &s, |x| eval::mse_loss(x, &target, Reduction::Sum));
        let num_dtarget =
            numeric_grad_unary(&target, &s, |x| eval::mse_loss(&pred, x, Reduction::Sum));

        assert_grad_close("mse(sum) dPred", &dpred, &num_dpred);
        assert_grad_close("mse(sum) dTarget", &dtarget, &num_dtarget);
    }

    #[test]
    fn mse_loss_grad_n_zero_is_zero() {
        // numel() == 0 はゼロ除算を避け zeros を返す（ガード条件の
        // 直接検証。中央差分は空テンソルに対して定義できないため
        // 解析式のみで確認する）。mean/sum いずれも同じ早期 return
        // 経路（`n == 0` 分岐）を通るため mean のみ代表して検証する。
        let pred = build_tensor(Vec::new(), &[0]);
        let target = build_tensor(Vec::new(), &[0]);
        let g = t(&[1.0], &[]);
        let (dpred, dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Mean);
        assert!(dense_vec(&dpred).is_empty());
        assert!(dense_vec(&dtarget).is_empty());
    }

    // --- CrossEntropyLoss（#191。PyTorch 参照値との突合は
    //     `tests/nn_cross_entropy.rs`、ここでは既存の
    //     `numeric_grad_unary`/`assert_grad_close`〈中央差分〉基盤に
    //     揃えた eval レベルの grad check を行う） ---

    #[test]
    fn cross_entropy_loss_grad_matches_numeric() {
        let logits = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let targets = fandhe_ai_tensor_core::Tensor::new(vec![2i32, 0], &[2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        // forward 出力は既に scalar shape [] のため、`s` も scalar
        // （`mse_loss_grad_matches_numeric` と同じ「射影 s がスカラー」
        // パターン）。
        let s = t(&[3.0], &[]);

        let g = s.clone();
        let dlogits = cross_entropy_loss_vjp(&logits, &targets, 1, Reduction::Mean, &g);
        let num_dlogits = numeric_grad_unary(&logits, &s, |x| {
            eval::cross_entropy_loss(x, &targets, 1, Reduction::Mean)
        });

        assert_grad_close("cross_entropy_loss(mean) dLogits", &dlogits, &num_dlogits);
    }

    #[test]
    fn cross_entropy_loss_grad_sum_matches_numeric() {
        let logits = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let targets = fandhe_ai_tensor_core::Tensor::new(vec![2i32, 0], &[2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let s = t(&[3.0], &[]);

        let g = s.clone();
        let dlogits = cross_entropy_loss_vjp(&logits, &targets, 1, Reduction::Sum, &g);
        let num_dlogits = numeric_grad_unary(&logits, &s, |x| {
            eval::cross_entropy_loss(x, &targets, 1, Reduction::Sum)
        });

        assert_grad_close("cross_entropy_loss(sum) dLogits", &dlogits, &num_dlogits);
    }

    // --- reduce_to_shape（中間軸縮約。ランク同一で先頭・末尾以外の
    //     軸を broadcast 元へ潰す経路。add/mul の bias broadcast テスト
    //     は末尾軸・スカラーテストは rank 0 のみで、この経路は未カバー
    //     だった） ---

    #[test]
    fn reduce_to_shape_middle_axis_matches_numeric() {
        // g: [2,4,3] を target_shape [2,1,3]（中間軸 dim=1 が size 1）
        // へ縮約する。add(a, b) の b 側勾配として同じ経路を通す
        // （a: [2,4,3]、b: [2,1,3] からの broadcast）。
        let a = t(
            &[
                1.0, -2.0, 3.0, 0.5, -1.0, 2.0, 1.5, -0.5, 2.5, -1.5, 0.5, -2.5, 3.0, -1.0, 0.5,
                -0.5, 1.0, -1.5, 2.0, -2.0, 0.5, -0.5, 1.5, -1.0,
            ],
            &[2, 4, 3],
        );
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[2, 1, 3]);
        let s = t(
            &[
                1.0, -1.0, 2.0, 0.5, 1.0, -0.5, 2.0, -2.0, 0.5, -0.5, 1.5, -1.0, 0.2, -0.2, 0.4,
                0.1, 0.2, -0.1, 0.4, -0.4, 0.1, -0.1, 0.3, -0.2,
            ],
            &[2, 4, 3],
        );

        let db = reduce_to_shape(&s, b.shape());
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("reduce_to_shape(middle axis) dB", &db, &num_db);
    }

    // --- vjp() ディスパッチの疎通確認（#18 との継ぎ目契約） ---
    //
    // Low 指摘: MatMul のみが vjp() 経由で疎通確認されており、他の
    // 8 演算（Add/Mul/Relu/Exp/Tanh/Sum/Max/MseLoss）は内部ヘルパーを
    // 直接呼ぶ形でしか検証されていなかった。各 match アームの配線
    // （`nodes[a.0]`/`nodes[b.0]` の対応順序）を通しで検証するため、
    // 全 9 演算（Leaf を除く）を vjp() 経由でテストする。Sigmoid
    // （TASK-9.1b・#92）追加により対象は 10 演算に拡大。
    // CrossEntropyLoss（#191）追加により対象は 11 演算に拡大
    // （Add/Mul/Relu/Exp/Tanh/Sigmoid/Sum/Max/MseLoss/MatMul/
    // CrossEntropyLoss）。

    fn leaf_node(value: Tensor<f32>) -> TapeNode {
        // `TapeNode`（TASK-12.1d・#164）は `shape` を独立フィールドとして
        // 持ち、`value` は `OnceCell` になった。テスト用の葉ノードは
        // 常に実体化済み（`OnceCell::from`）として構築する。
        let shape = value.shape().to_vec();
        TapeNode {
            op: Op::Leaf,
            shape,
            value: std::cell::OnceCell::from(value),
            lazy_chain_size: 0,
        }
    }

    /// `vjp()` の第 5 引数（`ops: &dyn BackendOps`）用テストフィクスチャ。
    /// 本モジュールのテストはすべて `leaf_node` で葉ノード（常に実体化
    /// 済み）のみを組み立てるため `materialize_fallible` は早期リターン
    /// し、`ops` の実体は使われない（`crate::test_support::TestOps` を
    /// 形式的に渡すのみ）。
    fn test_ops() -> crate::test_support::TestOps {
        crate::test_support::TestOps
    }

    #[test]
    fn vjp_dispatch_matmul_returns_both_inputs() {
        let a = t(&[1.0, 2.0, -1.0, 0.5, 3.0, -2.0], &[2, 3]);
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[3, 2]);
        let out_value = eval::matmul(&a, &b);
        let (expected_da, expected_db) = matmul_vjp(&a, &b, &t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]));
        let nodes = vec![leaf_node(a), leaf_node(b)];
        let op = Op::MatMul(NodeId(0), NodeId(1));
        let g = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);

        let grads = vjp(&op, &out_value, &g, &nodes, &test_ops(), None).unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected_da));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&expected_db));
    }

    #[test]
    fn vjp_dispatch_add_returns_both_inputs_in_order() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let g = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);
        let out_value = eval::add(&a, &b);
        let nodes = vec![leaf_node(a), leaf_node(b)];
        let op = Op::Add(NodeId(0), NodeId(1));

        let grads = vjp(&op, &out_value, &g, &nodes, &test_ops(), None).unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&g));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&g));
    }

    #[test]
    fn vjp_dispatch_mul_returns_both_inputs_in_order() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let g = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);
        let out_value = eval::mul(&a, &b);
        let nodes = vec![leaf_node(a.clone()), leaf_node(b.clone())];
        let op = Op::Mul(NodeId(0), NodeId(1));

        let grads = vjp(&op, &out_value, &g, &nodes, &test_ops(), None).unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&eval::mul(&g, &b)));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&eval::mul(&g, &a)));
    }

    #[test]
    fn vjp_dispatch_relu_returns_single_input() {
        let a = t(&[2.0, -3.0, 0.5, -0.02], &[2, 2]);
        let g = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);
        let out_value = eval::relu(&a);
        let nodes = vec![leaf_node(a.clone())];
        let op = Op::Relu(NodeId(0));

        let grads = vjp(&op, &out_value, &g, &nodes, &test_ops(), None).unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(
            dense_vec(&grads[0].1),
            dense_vec(&elementwise_mul_mask(&g, &a, |v| v > 0.0))
        );
    }

    #[test]
    fn vjp_dispatch_exp_returns_single_input() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let g = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::exp(&a);
        let nodes = vec![leaf_node(a)];
        let op = Op::Exp(NodeId(0));

        let grads = vjp(&op, &out_value, &g, &nodes, &test_ops(), None).unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(
            dense_vec(&grads[0].1),
            dense_vec(&eval::mul(&g, &out_value))
        );
    }

    #[test]
    fn vjp_dispatch_tanh_returns_single_input() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let g = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::tanh(&a);
        let nodes = vec![leaf_node(a)];
        let op = Op::Tanh(NodeId(0));

        let grads = vjp(&op, &out_value, &g, &nodes, &test_ops(), None).unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = eval::mul(&g, &tanh_grad_factor(&out_value));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_sigmoid_returns_single_input() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let g = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::sigmoid(&a);
        let nodes = vec![leaf_node(a)];
        let op = Op::Sigmoid(NodeId(0));

        let grads = vjp(&op, &out_value, &g, &nodes, &test_ops(), None).unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = eval::mul(&g, &sigmoid_grad_factor(&out_value));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_sum_returns_single_input() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let g = t(&[1.0, -1.0, 2.0], &[3]);
        let out_value = eval::sum(&a, Some(0), &[3]);
        let nodes = vec![leaf_node(a)];
        let op = Op::Sum {
            input: NodeId(0),
            dim: Some(0),
        };

        let grads = vjp(&op, &out_value, &g, &nodes, &test_ops(), None).unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = unreduce_broadcast(&g, &[2, 3], Some(0));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_max_returns_single_input() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let g = t(&[1.0, -1.0, 2.0], &[3]);
        let out_value = eval::max(&a, Some(0), &[3]);
        let nodes = vec![leaf_node(a.clone())];
        let op = Op::Max {
            input: NodeId(0),
            dim: Some(0),
        };

        let grads = vjp(&op, &out_value, &g, &nodes, &test_ops(), None).unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = max_vjp(&a, Some(0), &out_value, &g);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_mse_loss_mean_returns_both_inputs_in_order() {
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let g = t(&[3.0], &[]);
        let out_value = eval::mse_loss(&pred, &target, Reduction::Mean);
        let nodes = vec![leaf_node(pred.clone()), leaf_node(target.clone())];
        let op = Op::MseLoss {
            pred: NodeId(0),
            target: NodeId(1),
            reduction: Reduction::Mean,
        };

        let grads = vjp(&op, &out_value, &g, &nodes, &test_ops(), None).unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        let (expected_dpred, expected_dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Mean);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected_dpred));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&expected_dtarget));
    }

    #[test]
    fn vjp_dispatch_mse_loss_sum_returns_both_inputs_in_order() {
        // sum 縮約（#190）でも `Op::MseLoss` ディスパッチが reduction を
        // 正しく `mse_loss_vjp` へ引き渡すことを確認する。
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let g = t(&[3.0], &[]);
        let out_value = eval::mse_loss(&pred, &target, Reduction::Sum);
        let nodes = vec![leaf_node(pred.clone()), leaf_node(target.clone())];
        let op = Op::MseLoss {
            pred: NodeId(0),
            target: NodeId(1),
            reduction: Reduction::Sum,
        };

        let grads = vjp(&op, &out_value, &g, &nodes, &test_ops(), None).unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        let (expected_dpred, expected_dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Sum);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected_dpred));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&expected_dtarget));
    }

    #[test]
    fn vjp_dispatch_cross_entropy_loss_returns_single_input() {
        // `targets` は非追跡（`NodeId` を持たない Op payload）のため、
        // `MseLoss`（pred/target 2 系統）とは異なり寄与は `logits` の
        // 1 系統のみ（`grads.len() == 1`）であることが配線検証の要点
        // （`tape::Op::CrossEntropyLoss` doc 参照）。
        let logits = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let targets = fandhe_ai_tensor_core::Tensor::new(vec![2i32, 0], &[2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let g = t(&[3.0], &[]);
        let out_value = eval::cross_entropy_loss(&logits, &targets, 1, Reduction::Mean);
        let nodes = vec![leaf_node(logits.clone())];
        let op = Op::CrossEntropyLoss {
            logits: NodeId(0),
            targets: targets.clone(),
            class_dim: 1,
            reduction: Reduction::Mean,
        };

        let grads = vjp(&op, &out_value, &g, &nodes, &test_ops(), None).unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = cross_entropy_loss_vjp(&logits, &targets, 1, Reduction::Mean, &g);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }
}
