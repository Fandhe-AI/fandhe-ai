//! anomaly detection（`backward_detect_anomaly`。イシュー #2671・親 #2668。契約の正は
//! `docs/autodiff-jacobian-hessian-gradcheck-decision.md` §3.5）。
//!
//! PyTorch `torch.autograd.detect_anomaly` に相当する機能を、**プロセスワイド／テープ単位の
//! モードを持たない自由関数**として提供する。既存の [`Tape::backward`] を 1 回呼んだうえで、
//! テープ上の値と勾配を読み取り専用で走査し、最初に NaN／±inf を持つノードの
//! 段階（forward／gradient）・node id・Op 種別名・shape を型付きエラー
//! （[`AutodiffError::Backward`]）で報告する。`Tape::backward`（`backward_impl`）・`Op`・
//! VJP・`AutodiffError` は変更しない。
//!
//! **検出範囲（過剰な保証をしない）**:
//! - forward 段階は「backward 呼び出し後に**実体化済み**のノード値」だけを見る。未実体化の
//!   遅延ノード・デバイス常駐葉（`ResidentLeaf`）は新たに実体化せず自然にスキップする
//!   （検出のためにテープの状態を変えない）。
//! - gradient 段階は「勾配が到達したノード」だけを見る。勾配スロットが `None` のノードは対象外。
//! - 非有限値が複数ある場合の「最初」は、forward は node id 昇順（発生順）、gradient は
//!   node id 降順（逆伝播の走査順）で最初に見つかったもの。forward で見つかれば gradient は
//!   走査しない（原因に近い側を先に報告する）。
//! - GPU テープではホスト読み出しが起きる（デバッグ用途で、性能は非保証）。
//! - `compat::Sequential::fit` など学習ループへの結線は対象外。
//!
//! **情報露出の抑制**: メッセージへ載せるのは段階・node id・Op 種別名・shape のみ。
//! テンソルの値、`Op` の payload（`CrossEntropyLoss` の targets・スカラー演算の定数・
//! `Custom` の利用者定義名）は載せない（`op_kind_name` 参照）。
//!
//! **公開形は未承認（保留）**: facade（`fandhe_ai`）へは公開しない（承認依頼 #2677・公開
//! #2678）。保留は facade の `GradcheckAnomalyHoldDoctestGuard` と `tests/api_surface.rs` の
//! 否定ガードで機械固定している。
//!
//! **公開状況（イシュー #2678）**: 承認形どおり公開済み: facade `Tape::backward_detect_anomaly`。本モジュール自体は facade から再エクスポートしない。
//! 上の「未承認」「保留」「承認依頼は #2677」の記述は #2677 時点のもので、承認形の公開は #2678 で行った
//! （ルート #2499 の承認コメント issuecomment-6033824965・`docs/compat-api-scope.md` §5.1）。

use std::fmt::{self, Write as _};

use crate::backward::Gradients;
use crate::error::AutodiffError;
use crate::tape::{NodeId, Op, Tape};
use crate::var::Var;

/// 消費側ノード候補として併記する最大件数（メッセージ長の上限）。
const MAX_CONSUMER_CANDIDATES: usize = 4;

/// `{:?}` の出力から識別子文字（英数字・`_`）の先頭部分だけを受け取り、最初の非識別子文字
/// （`(`・`{`・空白等）で `fmt::Error` を返してフォーマットを中断する書き込み先。
struct IdentPrefix {
    buf: String,
}

impl fmt::Write for IdentPrefix {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for c in s.chars() {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.buf.push(c);
            } else {
                return Err(fmt::Error);
            }
        }
        Ok(())
    }
}

/// `Op` の variant 名だけを返す。`Op` は `derive(Debug)` で payload（targets・定数・
/// 利用者定義名）を含むため、`{op:?}` をそのままメッセージへ入れてはならない。
/// derive の Debug は variant 名を最初に書くので、識別子以外が現れた時点で打ち切る
/// （約 150 variant の網羅 match を持たない設計。`Op` の追加に追従不要）。
fn op_kind_name(op: &Op) -> String {
    let mut w = IdentPrefix { buf: String::new() };
    // 打ち切りで `Err` が返るのは意図どおり（payload を出力しないため）。
    let _ = write!(w, "{op:?}");
    if w.buf.is_empty() {
        "Op".to_string()
    } else {
        w.buf
    }
}

/// テンソルが NaN／±inf を 1 つでも含むか（論理順のホスト値で判定）。
fn has_non_finite(t: &fandhe_ai_tensor_core::Tensor<f32>) -> bool {
    t.host_slice().iter().any(|v| !v.is_finite())
}

/// `tape.backward(loss)` を実行し、非有限値（NaN／±inf）を最初に生んだノードを検出する。
///
/// 手順（契約。順序固定）:
/// 1. [`Tape::backward`] を呼ぶ。そのエラー（追跡なし loss・別テープ等）はそのまま伝播する。
/// 2. forward 走査（node id 昇順）: 実体化済みのノード値に非有限値があれば
///    `Err(Backward)`（段階 forward）。
/// 3. gradient 走査（node id 降順＝逆伝播順）: 勾配に非有限値があれば `Err(Backward)`
///    （段階 gradient。そのノードを入力に持つ消費側ノードを候補として併記する。
///    勾配の蓄積オーバーフローでも起こりうるため、候補は断定ではない）。
/// 4. どちらも無ければ手順 1 の [`Gradients`] をそのまま返す（`Tape::backward` と bit 一致）。
///
/// 検出は読み取りのみで、テープのノード・値・勾配を書き換えない。検出範囲の限定は
/// モジュール doc を参照。
pub fn backward_detect_anomaly(tape: &Tape, loss: &Var<'_>) -> Result<Gradients, AutodiffError> {
    let grads = tape.backward(loss)?;
    let nodes = tape.nodes.borrow();

    for (id, node) in nodes.iter().enumerate() {
        let Some(value) = node.value.get() else {
            continue;
        };
        if has_non_finite(value) {
            return Err(AutodiffError::Backward(format!(
                "anomaly detection: forward 値に非有限値（NaN/inf）を検出: \
                 node {id}・Op {}・shape {:?}",
                op_kind_name(&node.op),
                node.shape
            )));
        }
    }

    let slots = grads.grad_slots();
    for (id, slot) in slots.iter().enumerate().rev() {
        let Some(g) = slot else {
            continue;
        };
        if !has_non_finite(g) {
            continue;
        }
        let Some(node) = nodes.get(id) else {
            continue;
        };
        let mut consumers: Vec<String> = Vec::new();
        for (cid, cnode) in nodes.iter().enumerate() {
            let mut uses = false;
            cnode.op.for_each_input(|NodeId(i)| uses |= i == id);
            if uses && consumers.len() < MAX_CONSUMER_CANDIDATES {
                consumers.push(format!("node {cid}（{}）", op_kind_name(&cnode.op)));
            }
        }
        let hint = if consumers.is_empty() {
            String::new()
        } else {
            format!(
                "。このノードを入力に持つ消費側の候補: {}",
                consumers.join("・")
            )
        };
        return Err(AutodiffError::Backward(format!(
            "anomaly detection: 勾配に非有限値（NaN/inf）を検出: \
             node {id}・Op {}・shape {:?}{hint}",
            op_kind_name(&node.op),
            node.shape
        )));
    }

    drop(nodes);
    Ok(grads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::NodeId;
    use fandhe_ai_tensor_core::Tensor;

    #[test]
    fn op_kind_name_returns_variant_name_only() {
        assert_eq!(op_kind_name(&Op::Leaf), "Leaf");
        assert_eq!(op_kind_name(&Op::Add(NodeId(1), NodeId(2))), "Add");
    }

    #[test]
    fn op_kind_name_does_not_leak_scalar_payload() {
        let name = op_kind_name(&Op::Add(NodeId(123456), NodeId(654321)));
        assert!(
            !name.contains("123456") && !name.contains("654321"),
            "{name}"
        );
    }

    #[test]
    fn op_kind_name_does_not_leak_cross_entropy_targets() {
        let targets = Tensor::new(vec![7_i32, 9_i32], &[2]).unwrap();
        let op = Op::CrossEntropyLoss {
            logits: NodeId(3),
            targets,
            class_dim: 1,
            reduction: crate::var::Reduction::Mean,
        };
        // 前提: Debug 全体には payload が含まれる（打ち切りが効いていることの裏付け）。
        assert!(format!("{op:?}").contains("targets"));
        assert_eq!(op_kind_name(&op), "CrossEntropyLoss");
    }

    #[test]
    fn has_non_finite_detects_each_class() {
        let t = |v: Vec<f32>| {
            let n = v.len();
            Tensor::new(v, &[n]).unwrap()
        };
        assert!(!has_non_finite(&t(vec![0.0, 1.0, -2.5])));
        assert!(has_non_finite(&t(vec![0.0, f32::NAN])));
        assert!(has_non_finite(&t(vec![f32::INFINITY])));
        assert!(has_non_finite(&t(vec![f32::NEG_INFINITY, 1.0])));
    }
}
