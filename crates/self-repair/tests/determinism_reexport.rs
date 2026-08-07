//! TASK-4.4b「双方（guardrail・self-repair）で有効」の受け入れ条件を検証する。
//!
//! `self_repair::determinism`（`guardrail::determinism` の再輸出）経由でも、
//! `guardrail` 側と同一の決定性契約（同一シード → 同一系列）が成立することを
//! 確認する。

use self_repair::determinism::seeded_rng;

#[test]
fn reexported_seeded_rng_is_deterministic() {
    let mut a = seeded_rng("self-repair.trial-log");
    let mut b = seeded_rng("self-repair.trial-log");
    for _ in 0..50 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn reexported_seeded_rng_matches_guardrail_directly() {
    // self-repair 経由の再輸出と guardrail 直接呼び出しが完全に同一実体
    // （同じ関数・同じ定数）であることを、生成系列の一致で確認する。
    let mut via_self_repair = self_repair::determinism::seeded_rng("cross-crate-check");
    let mut via_guardrail = guardrail::determinism::seeded_rng("cross-crate-check");
    for _ in 0..50 {
        assert_eq!(via_self_repair.next_u64(), via_guardrail.next_u64());
    }
}
