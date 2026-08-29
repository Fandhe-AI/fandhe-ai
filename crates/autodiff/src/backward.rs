//! 勾配伝播 API（`Tape::backward`・`Gradients`）。
//!
//! TASK-1.5a（#16）が記録したテープ列（`tape::TapeNode`）を発生順とは
//! 逆順に走査し、TASK-1.5b（#17）の `grad::vjp()` が返す「ノードごとの
//! 入力側勾配寄与」を入力 `NodeId` へ蓄積する。`vjp()` 自身は寄与を返す
//! だけで蓄積しない、という #17 で確定した責務境界をここで引き継ぐ
//! （`grad.rs` モジュールコメント参照）。
//!
//! API 形状は `docs/public-api-design.md` §3.1 で確定済み。逆走査・
//! 蓄積アルゴリズムは PoC-v2-2 参照実装
//! （`docs/spec/03-poc/poc-v2-2-autodiff/code/rust/src/tape.rs` の
//! `backward()`/`accumulate()`）を f32・`vjp()` ディスパッチ経由の形へ
//! productize したもの。
//!
//! **学習ループでの運用**: `Tape` はステップごとに使い捨てる運用を
//! 前提とする（`tape.rs` モジュールコメント「学習ループでの運用」・
//! `docs/public-api-design.md` §3.1.1）。`backward()` はテープへノードを
//! 追加しないため、`nodes` の不変借用 1 回で走査を完結できる
//! （`RefCell` の二重可変借用 panic の余地を作らない）。

use fandhe_ai_tensor_core::Tensor;

use crate::error::AutodiffError;
use crate::grad;
use crate::tape::{NodeId, Op, ResidentResolver, Tape, TapeId};
use crate::var::Var;

/// `Tape::backward` が返す勾配の入れ物。`Var` 単位で `get()` から
/// 引ける（`docs/public-api-design.md` §3.1）。
///
/// ノードごとの勾配を `Option` で保持し、loss から到達しないノードは
/// `None` のまま残す（PoC-v2-2 の「None を zeros 埋め」はしない設計。
/// 未到達ノードの勾配を暗黙に 0 とみなすと、呼び出し側が「そもそも
/// loss に寄与していない」ことと「寄与はあるが偶然 0」を区別できなく
/// なるため、`get()` の戻り値を `Option` のまま伝える）。
#[derive(Debug)]
pub struct Gradients {
    tape_id: TapeId,
    grads: Vec<Option<Tensor<f32>>>,
}

impl Gradients {
    /// `var` に対応する勾配を取得する。`var` が別 `Tape` に属する場合は
    /// `Err(TapeMismatch)`（`Var::matmul` 等のクロステープ検査
    /// 〈`var.rs`〉と同じ契約）。backward 実行後に同一テープへ追加された
    /// `Var`（`node_id()` が `grads.len()` 以上）や、loss から到達しない
    /// `Var` は `Ok(None)`（境界外アクセスで panic しない。
    /// `.claude/rules/coding-rust.md` 本番経路 panic 禁止方針）。
    pub fn get(&self, var: &Var<'_>) -> Result<Option<&Tensor<f32>>, AutodiffError> {
        if var.tape_id() != self.tape_id {
            return Err(AutodiffError::TapeMismatch);
        }
        Ok(self.grads.get(var.node_id().0).and_then(|g| g.as_ref()))
    }
}

impl Tape {
    /// `loss` を起点に逆伝播し、各ノードへ流入した勾配を `Gradients` へ
    /// まとめて返す。
    ///
    /// 処理は「①クロステープ検査 → ②シード設定 → ③逆走査 → ④蓄積」の
    /// 順（`var.rs` の forward 演算メソッドと対になる構成）。
    ///
    /// **非スカラー loss のセマンティクス**: `loss` が非スカラー
    /// （shape が `[]` でない）の場合、シードは全要素 1 の同 shape
    /// テンソルとする。これは `sum(loss).backward()` と数学的に等価な
    /// 「暗黙の総和射影」であり、エラーにはしない安全側の全域定義
    /// （PyTorch の `Tensor.backward()` がスカラー以外に `gradient`
    /// 引数を要求してエラーにするのとは異なるが、本 API は
    /// `docs/public-api-design.md` §3.1 が `gradient` 引数を持たない
    /// 単純形として確定させているため、全域で定義可能なこの意味論を
    /// 採用する）。
    pub fn backward(&self, loss: &Var<'_>) -> Result<Gradients, AutodiffError> {
        self.backward_impl(loss, None)
    }

    /// [`Tape::backward`] のデバイス常駐対応版（イシュー #1022）。
    /// `resolver`（`fandhe_ai_autodiff::optim::device_store::
    /// DeviceParamStore` が実装する [`ResidentResolver`]）を
    /// `grad::vjp` へスレッドし、`Op::LinearResident` の VJP が
    /// weight のデバイスバッファを取得できるようにする。
    /// `DeviceParamStore::backward`（`optim::device_store`）からのみ
    /// 呼ばれる（`tape::Op::LinearResident` doc 参照）。
    pub(crate) fn backward_with_resident(
        &self,
        loss: &Var<'_>,
        resolver: &dyn ResidentResolver,
    ) -> Result<Gradients, AutodiffError> {
        self.backward_impl(loss, Some(resolver))
    }

    /// `backward`／`backward_with_resident` の共通実装（イシュー #1022 で
    /// `resolver` 引数を追加する形へ整理）。
    fn backward_impl(
        &self,
        loss: &Var<'_>,
        resolver: Option<&dyn ResidentResolver>,
    ) -> Result<Gradients, AutodiffError> {
        if loss.tape_id() != self.id {
            return Err(AutodiffError::TapeMismatch);
        }

        // `backward` はノードを追加しないため、走査全体を単一の不変
        // 借用で完結させる（`Var::value()`/`to_tensor()` のドキュメント
        // が警告する `RefCell` 二重可変借用 panic の経路をそもそも
        // 作らない）。
        let nodes = self.nodes.borrow();
        let n = nodes.len();

        let mut grads: Vec<Option<Tensor<f32>>> = vec![None; n];
        // `loss` 自身が forward 記録済みの未実体化ノード（elementwise の
        // 遅延グラフの末端）である場合に備え、`materialize_fallible`
        // （層 1）経由で shape を読む（TASK-12.1d・#164）。
        let loss_shape = crate::tape::materialize_fallible(&nodes, self.ops(), loss.node_id())?
            .shape()
            .to_vec();
        // `loss_shape` は既に構築済みの `loss.value`（同 shape のテンソル）から
        // 取得しているため、現行の `tensor-core` 実装では本分岐は到達不能
        // （同 shape での `full()` が失敗する経路が存在しない）。ただし
        // `backward()` は `Result<Gradients, AutodiffError>` を返す設計であり、
        // 契約違反時に「エラーを返さず loss 自身の値を誤って seed に使う」
        // フォールバックは値中立でない（`eval.rs` の値中立フォールバック方針に
        // 反する）。将来の `tensor-core` 実装変更で到達可能になった場合に備え、
        // `debug_assert!` + 暗黙の誤った値ではなく `Err` を返す安全側の実装とする
        // （#18 レビュー指摘）。
        let seed = Tensor::full(&loss_shape, 1.0f32).map_err(|err| {
            AutodiffError::Backward(format!(
                "loss 自身の shape でのシードテンソル構築に失敗した（契約違反）: {err}"
            ))
        })?;
        grads[loss.node_id().0] = Some(seed);

        // 発生順とは逆順に走査する（Wengert list の逆伝播。PoC-v2-2の
        // `backward()` と同じ順序）。各ノードの `grads[id]` が `None`
        // のままなら loss から到達しない部分グラフであり、その先の
        // 入力ノードへは勾配を流さない（伝播スキップ）。
        for id in (0..n).rev() {
            // `grads[id]` は `Gradients::get()` が返す最終値そのものの
            // ため `take()` せず複製する（各ノードは逆走査で高々 1 回
            // しか処理しないが、値自体は結果として保持し続ける必要が
            // ある。取り除くと非葉ノードの `get()` が常に `None` になる）。
            let Some(upstream) = grads[id].clone() else {
                continue;
            };
            // ノード自身の forward 値（`out_value`）を層 1（fallible）
            // 経由で実体化する。当該ノードが elementwise の遅延グラフ
            // 末端の場合に備える（TASK-12.1d・#164。`Var::value()`〈層
            // 2〉は呼ばず、`Unsupported` 以外の失敗は `?` で伝播する）。
            let node = &nodes[id];
            // `Op::ResidentLeaf`（イシュー #1022）はホスト値を持たない
            // （`TapeNode::value` が常に空。`Tape::push_resident_leaf`
            // 参照）ため、`materialize_fallible` を呼ぶと「実体化済みの
            // はずが未実体化だった」契約違反フォールバック（`tape.rs::
            // lazy_leaf_value` の `debug_assert!` + ゼロ埋め）に誤って
            // 到達してしまう。`grad::vjp` は本 variant を `Op::Leaf` と
            // 同じく `out_value` を参照しないため、プレースホルダで
            // 十分（`Op::Leaf` は元々実体化済みで実害がなかったのと同じ
            // 理由で、ここでも実際の値は使われない）。
            let node_value = if matches!(node.op, Op::ResidentLeaf { .. }) {
                Tensor::scalar(0.0)
            } else {
                crate::tape::materialize_fallible(&nodes, self.ops(), NodeId(id))?.clone()
            };
            let contributions = grad::vjp(
                &node.op,
                &node_value,
                &upstream,
                &nodes,
                self.ops(),
                resolver,
            )?;
            for (target, contribution) in contributions {
                accumulate(&mut grads, target, contribution);
            }
        }

        Ok(Gradients {
            tape_id: self.id,
            grads,
        })
    }
}

/// 同一入力ノードへ複数経路から流入した勾配を合算する（PoC-v2-2
/// `accumulate()` 相当）。初回流入は `Some` を差し込むだけ、2 回目以降
/// は既存値と `eval::add`（同 shape 加算）で合算する。
fn accumulate(grads: &mut [Option<Tensor<f32>>], target: NodeId, contribution: Tensor<f32>) {
    match grads[target.0].take() {
        Some(existing) => {
            grads[target.0] = Some(crate::eval::add(&existing, &contribution));
        }
        None => {
            grads[target.0] = Some(contribution);
        }
    }
}
