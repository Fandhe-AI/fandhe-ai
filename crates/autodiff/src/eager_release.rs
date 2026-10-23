//! reuse モード（同じ `Tape` で `matmul` を繰り返す）の結果バッファ蓄積を避ける、
//! consumer のない eager matmul 結果の自動解放（イシュー #3226・既定 OFF のゲート）。
//!
//! # 役割と帰属
//!
//! `docs/perf/cpu-gemm-reuse-fresh-inversion.md`（#3187）の確定原因: reuse では各反復の
//! 結果ノード（`TapeNode::value`）が `Tape` に残り続けるため、毎反復新しい出力バッファが
//! 要り、出力 1 枚分のページを新規に first-touch する（minflt/iter は fresh 0.00 に対し
//! reuse 1.37〜1.84）。host arena（#2104）は返却が起きないため hit=0 で効かない。
//! 本モジュールは「次の `Var::matmul` が出力を確保する**前**に、直前の matmul 結果が
//! 不要なら値を手放す」ことで、定常状態を 1 枠に収め、glibc の free list（arena OFF）
//! または arena の free list（arena ON）から再利用させる。
//!
//! # bit 同一の根拠（recompute 契約）
//!
//! 解放は `release_checkpoint_region`（activation checkpointing）と同じく
//! `value.take()` に加えて `recompute = true` を立てる。解放済みノードを後から読むと
//! `materialize_*` が `recompute_value` で同じ `ops.gemm` を再実行し、forward と bit 同一の
//! 値を返す（checkpoint の既存契約）。`docs/autodiff-retain-graph-accumulate-decision.md`
//! §2.1 が退けたのは `recompute == false` のまま解放して release ビルドで黙ってゼロを返す
//! 形であり、本機構はそれに当たらない（同 doc に追補）。再利用バッファの残値は
//! `take_zeroed_f32` が全要素ゼロ埋めするため結果へ混ざらない。
//!
//! # 呼び出し文脈
//!
//! `Var::matmul`（通常版のみ）が `release_before_matmul`（確保前）と `note_matmul`
//! （push 後）を呼ぶ。`matmul_fp32_strict`・`matmul_low_precision`・その他 op は候補を
//! 更新しない。`Tape::reset`・`rollback_to` が候補を消す。GPU（CPU 以外の device）は
//! 対象外。ゲート定数の反転は ADOPT 後の #3227 でのみ行う。

use std::cell::Cell;
use std::marker::PhantomData;

use fandhe_ai_tensor_core::Device;

use crate::tape::{NodeId, Tape};

/// 自動解放の既定有効フラグ。**既定 OFF**。A/B（`docs/perf/logs/cpu-gemm-reuse-release-3226/
/// RULE.txt`）が ADOPT となった後の #3227 でのみ `true` にする。after 腕はこの 1 行だけを反転する。
pub(crate) const EAGER_MATMUL_RELEASE_DEFAULT_ENABLED: bool = false;

thread_local! {
    static OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// スコープ限定の上書きを元に戻す RAII ガード（`!Send`。`tensor_core::alloc` と同型）。
#[must_use = "ガードを drop すると上書きが元に戻る"]
pub struct EagerReleaseOverrideGuard {
    prev: Option<bool>,
    _not_send: PhantomData<*const ()>,
}

impl Drop for EagerReleaseOverrideGuard {
    fn drop(&mut self) {
        let _ = OVERRIDE.try_with(|c| c.set(self.prev));
    }
}

/// 現スレッドの有効／無効をスコープ限定で上書きする（テスト・診断用。ネスト可）。
/// facade からは再公開しない内部クレートの結線面。
pub fn override_enabled_for_scope(enabled: bool) -> EagerReleaseOverrideGuard {
    let prev = OVERRIDE.with(|c| c.replace(Some(enabled)));
    EagerReleaseOverrideGuard {
        prev,
        _not_send: PhantomData,
    }
}

/// 現スレッドで自動解放が有効か（TLS 破棄後は既定値へ倒す）。
pub fn is_enabled() -> bool {
    OVERRIDE
        .try_with(Cell::get)
        .ok()
        .flatten()
        .unwrap_or(EAGER_MATMUL_RELEASE_DEFAULT_ENABLED)
}

/// `Var::matmul` が出力を確保する前に呼ぶ。直前の matmul 結果が不要なら値を手放す。
///
/// `lhs`／`rhs` は今回の matmul の入力ノード。借用に失敗した場合（`Var::value()` の `Ref` が
/// 生存中等）は何もしない（panic しない）。
pub(crate) fn release_before_matmul(tape: &Tape, lhs: NodeId, rhs: NodeId) {
    if !is_enabled() {
        return;
    }
    let Some(c) = tape.last_eager_matmul.take() else {
        return;
    };
    if c == lhs || c == rhs || tape.ops().device() != Device::Cpu {
        return;
    }
    let Ok(mut nodes) = tape.nodes.try_borrow_mut() else {
        return;
    };
    if c.0 >= nodes.len() {
        return;
    }
    {
        let node = &nodes[c.0];
        // `release_checkpoint_region` と同じ解放条件。未実体化なら解放するものがない。
        if node.value.get().is_none()
            || node.recompute
            || node.fp32_strict
            || node.low_precision
            || !node.op.is_checkpoint_eligible()
        {
            return;
        }
    }
    // c より後ろのノードが c を入力に取っていれば consumer あり。解放しない
    // （読まれるたびに再計算されて逆に遅くなるため）。
    let mut consumed = false;
    for node in &nodes[c.0 + 1..] {
        node.op.for_each_input(|i| {
            if i == c {
                consumed = true;
            }
        });
        if consumed {
            return;
        }
    }
    let node = &mut nodes[c.0];
    if let Some(v) = node.value.take() {
        node.recompute = true;
        // arena OFF なら drop（glibc が再利用）、ON なら次の `take_zeroed_f32` が hit する。
        // 利用者が Arc を共有中なら回収されず、Tape 側の参照を手放すだけになる。
        fandhe_ai_tensor_core::alloc::recycle_tensor_f32(v);
    }
}

/// `Var::matmul`（通常版）が `Op::MatMul` を push した直後に呼び、解放候補を更新する。
pub(crate) fn note_matmul(tape: &Tape, id: NodeId) {
    if is_enabled() {
        tape.last_eager_matmul.set(Some(id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_ops;
    use fandhe_ai_tensor_core::Tensor;

    fn mats() -> (Tensor<f32>, Tensor<f32>) {
        let a = Tensor::new((0..6).map(|i| i as f32 * 0.5 - 1.0).collect(), &[2, 3]).unwrap();
        let b = Tensor::new((0..12).map(|i| i as f32 * 0.25 + 0.1).collect(), &[3, 4]).unwrap();
        (a, b)
    }

    fn state(tape: &Tape, id: NodeId) -> (bool, bool) {
        let nodes = tape.nodes.borrow();
        (nodes[id.0].value.get().is_some(), nodes[id.0].recompute)
    }

    #[test]
    fn default_is_off_and_nodes_untouched() {
        assert!(!is_enabled(), "既定 OFF");
        let tape = Tape::new_with_ops(test_ops());
        let (a, b) = mats();
        let (va, vb) = (tape.var(&a), tape.var(&b));
        let c1 = va.matmul(&vb).unwrap();
        let _c2 = va.matmul(&vb).unwrap();
        assert_eq!(state(&tape, c1.node_id()), (true, false));
    }

    #[test]
    fn releases_previous_and_reads_back_bit_identical() {
        let (a, b) = mats();
        let expect: Vec<u32> = {
            let tape = Tape::new_with_ops(test_ops());
            let c = tape.var(&a).matmul(&tape.var(&b)).unwrap();
            c.to_tensor()
                .host_slice()
                .iter()
                .map(|x| x.to_bits())
                .collect()
        };
        let _g = override_enabled_for_scope(true);
        let tape = Tape::new_with_ops(test_ops());
        let (va, vb) = (tape.var(&a), tape.var(&b));
        let c1 = va.matmul(&vb).unwrap();
        let c2 = va.matmul(&vb).unwrap();
        assert_eq!(state(&tape, c1.node_id()), (false, true));
        assert_eq!(state(&tape, c2.node_id()), (true, false));
        let got: Vec<u32> = c1
            .to_tensor()
            .host_slice()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(got, expect);
        assert_eq!(c1.host_view().len(), 8);
    }

    #[test]
    fn keeps_node_with_consumer_or_used_as_input() {
        let _g = override_enabled_for_scope(true);
        let (a, b) = mats();
        let tape = Tape::new_with_ops(test_ops());
        let (va, vb) = (tape.var(&a), tape.var(&b));
        let c = va.matmul(&vb).unwrap();
        let _r = c.sigmoid();
        let _c2 = va.matmul(&vb).unwrap();
        assert_eq!(state(&tape, c.node_id()), (true, false), "consumer あり");

        let tape = Tape::new_with_ops(test_ops());
        let (va, vb) = (tape.var(&a), tape.var(&b));
        let (sq, _) = (va.matmul(&vb).unwrap(), ());
        let bt = tape.var(&Tensor::new(vec![1.0; 12], &[4, 3]).unwrap());
        let _next = sq.matmul(&bt).unwrap();
        assert_eq!(
            state(&tape, sq.node_id()),
            (true, false),
            "次の matmul の入力"
        );
    }

    #[test]
    fn strict_output_is_not_released_and_reset_clears_candidate() {
        let _g = override_enabled_for_scope(true);
        let (a, b) = mats();
        let tape = Tape::new_with_ops(test_ops());
        let (va, vb) = (tape.var(&a), tape.var(&b));
        let s = va.matmul_fp32_strict(&vb).unwrap();
        let _c = va.matmul(&vb).unwrap();
        assert_eq!(state(&tape, s.node_id()), (true, false));

        let mut tape = Tape::new_with_ops(test_ops());
        {
            let (va, vb) = (tape.var(&a), tape.var(&b));
            let _ = va.matmul(&vb).unwrap();
        }
        tape.reset();
        assert!(tape.last_eager_matmul.get().is_none());
    }
}
