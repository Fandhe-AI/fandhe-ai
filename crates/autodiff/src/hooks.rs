//! backward hook の登録簿（`Tape` の side table）と解除ハンドル。
//!
//! 役割: `Tape::register_backward_hook`（`tape.rs`）が登録した hook を
//! node ごとに登録順（FIFO）で保持し、`backward.rs::backward_impl` が逆走査の
//! 各ノードで確定勾配を渡して発火するための索引を提供する。`checkpoints`
//! （activation checkpointing の区間索引）と同型の `Tape` 直属 side table
//! であり、`Var`／`TapeNode` には何も足さない（`Var: Copy` と
//! `fandhe-ai =0.10.0` の公開面を保つ。設計記録
//! `docs/autodiff-forward-backward-hooks-design.md` §4.1 案 C・§5.2・§5.7・§14.2）。
//!
//! # 境界
//!
//! - hook は観察専用（`&Tensor<f32>` を受け `Result<(), AutodiffError>` を返す）で
//!   勾配を書き換えられない。`'static` により `&Tape`／`Var<'t>` を捕捉できず、
//!   hook 内からテープを再入操作する経路を型で塞ぐ。
//! - `Send + Sync` 境界は `Tape: Send`（`Box<dyn BackendOps + Send>` を持つ
//!   既存契約）を hook 保持後も維持するために必須（`Arc<dyn Fn + Send + Sync>`）。
//! - 発火は `backward_impl` の単一箇所のみ。登録簿の借用は
//!   [`HookRegistry::snapshot_for`] が `Arc` を複製して即座に解放するため、
//!   hook 呼び出し中は `RefCell` 借用を保持しない（hook の panic 後も `Tape` は
//!   一貫状態のまま）。
//! - CPU 参照実装であり、CUDA／Metal の数値経路は追加しない（ホスト側
//!   `Tensor<f32>` を渡すだけのため実機 parity の申し送りは発生しない。同 §6・§14.7）。

use std::collections::HashMap;
use std::sync::Arc;

use fandhe_ai_tensor_core::Tensor;

use crate::error::AutodiffError;
use crate::tape::TapeId;

/// backward hook 本体の型（`Arc` で保持する観察専用クロージャ）。
///
/// `Fn(&Tensor<f32>)` が受ける値は当該ノードの確定済み勾配（`Gradients::get`
/// が返す値と同一）。`Err` は最初のものが `backward` の戻り値としてそのまま伝播する。
pub(crate) type BackwardHookFn = dyn Fn(&Tensor<f32>) -> Result<(), AutodiffError> + Send + Sync;

/// `Tape::register_backward_hook` が返す解除ハンドル（`Tape::remove_hook` へ値で渡す）。
///
/// 不透明型: フィールドは非公開で accessor も持たない。`Clone`／`Copy` を持たないため
/// 同一ハンドルでの二重解除は型上起きない（設計記録 §14.4 P3。後付けは非破壊・
/// 削除は破壊的のため最小から始める）。別 `Tape` のハンドル・`Tape::reset` を
/// またいだ旧世代のハンドルは `Tape::remove_hook` が `TapeMismatch` で拒否する。
#[derive(Debug)]
pub struct HookHandle {
    pub(crate) tape_id: TapeId,
    pub(crate) epoch: u64,
    pub(crate) node: usize,
    pub(crate) seq: u64,
}

/// node → 登録順 `(seq, hook)` の索引（`Tape::hooks`）。
#[derive(Default)]
pub(crate) struct HookRegistry {
    by_node: HashMap<usize, Vec<(u64, Arc<BackwardHookFn>)>>,
    next_seq: u64,
}

impl HookRegistry {
    /// hook を `node` の末尾（FIFO）へ追加し、採番した `seq` を返す。
    /// `seq` はリセットされず単調増加する（`u64` 溢れは `Err`）。
    pub(crate) fn insert(
        &mut self,
        node: usize,
        hook: Arc<BackwardHookFn>,
    ) -> Result<u64, AutodiffError> {
        let seq = self.next_seq;
        let next = seq.checked_add(1).ok_or_else(|| {
            AutodiffError::InvalidArgument(
                "Tape::register_backward_hook: hook の通番が u64 を溢れた".into(),
            )
        })?;
        self.next_seq = next;
        self.by_node.entry(node).or_default().push((seq, hook));
        Ok(seq)
    }

    /// `(node, seq)` の hook を取り除く。空になった node キーは削除する。
    /// 該当が無ければ `false`（呼び出し元の `Tape::remove_hook` は epoch 一致の
    /// 前提で呼ぶため、通常は `true`）。
    pub(crate) fn remove(&mut self, node: usize, seq: u64) -> bool {
        let Some(list) = self.by_node.get_mut(&node) else {
            return false;
        };
        let before = list.len();
        list.retain(|(s, _)| *s != seq);
        let removed = list.len() != before;
        if list.is_empty() {
            self.by_node.remove(&node);
        }
        removed
    }

    /// `node` の hook を登録順に複製して返す（借用を返さない）。
    pub(crate) fn snapshot_for(&self, node: usize) -> Vec<Arc<BackwardHookFn>> {
        self.by_node
            .get(&node)
            .map(|list| list.iter().map(|(_, h)| Arc::clone(h)).collect())
            .unwrap_or_default()
    }

    /// 全 hook を消去する（`Tape::reset` から。`next_seq` は維持する）。
    pub(crate) fn clear(&mut self) {
        self.by_node.clear();
    }

    /// hook が 1 件も無いか（未登録時の backward 早期 return 用）。
    pub(crate) fn is_empty(&self) -> bool {
        self.by_node.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noop() -> Arc<BackwardHookFn> {
        Arc::new(|_g: &Tensor<f32>| Ok(()))
    }

    #[test]
    fn insert_remove_keeps_fifo_and_prunes_empty_nodes() {
        let mut r = HookRegistry::default();
        assert!(r.is_empty());
        let a = r.insert(3, noop()).expect("insert a");
        let b = r.insert(3, noop()).expect("insert b");
        assert!(b > a);
        assert_eq!(r.snapshot_for(3).len(), 2);
        assert!(r.remove(3, a));
        assert!(!r.remove(3, a));
        assert_eq!(r.snapshot_for(3).len(), 1);
        assert!(r.remove(3, b));
        assert!(r.is_empty());
        assert!(r.snapshot_for(3).is_empty());
    }

    #[test]
    fn clear_drops_all_but_keeps_seq_monotonic() {
        let mut r = HookRegistry::default();
        let a = r.insert(0, noop()).expect("insert");
        r.clear();
        assert!(r.is_empty());
        let b = r.insert(0, noop()).expect("insert");
        assert!(b > a);
    }

    #[test]
    fn seq_overflow_is_an_error() {
        let mut r = HookRegistry {
            next_seq: u64::MAX,
            ..HookRegistry::default()
        };
        assert!(matches!(
            r.insert(0, noop()),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(r.is_empty());
    }

    /// resident 葉（`Var` を外部へ露出しないためクレート内で検証）への登録は
    /// `InvalidArgument` で拒否し、登録簿を変更しない（設計記録 §14.4 P7）。
    #[test]
    fn resident_leaf_registration_is_rejected_without_touching_registry() {
        use crate::tape::Tape;
        use crate::var::Var;

        let tape = Tape::new();
        let id = tape.push_resident_leaf(vec![2], 0, 0);
        let v = Var::from_raw(&tape, id);
        assert!(matches!(
            tape.register_backward_hook(&v, |_| Ok(())),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(tape.backward_hooks_for(id.0).is_empty());
    }
}
