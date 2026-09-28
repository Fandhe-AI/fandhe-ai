//! 統合テスト共通の決定的シード PRNG（xorshift64*。codex-review 指摘
//! `PRRT_kwDOTuUCJc6mzcKf`・PR #2351・イシュー #2349 是正）。
//!
//! `crates/onnx-interop/tests/*.rs` は `onnx-interop` の公開 API のみを
//! 経由する別クレート扱いのため（`crates/autodiff/tests/common/mod.rs`
//! と同型の構成理由）、`bench_harness::rng::Xorshift64Star` を直接
//! 再利用すると `bench-harness` dev-dependency を経由する。`bench-harness`
//! は `fandhe-ai-backend-cuda`（`crates/backend-cuda/src/nvrtc.rs`。openat・
//! /proc/self/fd 等 unix 系 API への TOCTOU 対策のため非 unix 向け
//! フォールバックを持たず `compile_error!` で拒否する設計。#509／PR #677）
//! へ無条件の通常依存を持つため、`bench-harness` dev-dependency 全体が
//! 非 unix ターゲットでビルド不能になる。external data の Windows 対応
//! （イシュー #2349）で `cargo check --tests --target
//! x86_64-pc-windows-msvc` を通す際、本クレートの CPU のみで完結する
//! 既存テスト（学習チェックポイント・ONNX export/import parity 等）が
//! `bench-harness` 経由の間接依存だけを理由に丸ごと Windows から除外
//! されるのはテスト弱体化（`.claude/rules/coding-rust.md`「テスト・
//! ベンチ」節の精神）に当たるため、`bench-harness`／`backend-cuda` の
//! クレート境界を変更せず（それらは `runtime-builder`／`backend-builder`
//! の管轄でありイシュー #2349 のスコープ外）、本クレートのテスト専用に
//! 同一アルゴリズムを複製して依存を切り離す。
//!
//! アルゴリズム自体は `crates/bench-harness/src/rng.rs`（移植元
//! `docs/spec/03-poc/poc-v2-5-backend-numeric-parity/code/rust/src/rng.rs`）
//! と**差分なしで複製**する。決定的シード PRNG は仕様として固定済みの
//! アルゴリズムであり、コード変更のたびに 2 重管理が必要になる実装詳細
//! （`.claude/rules/code-comment-style.md`「何を書かないか」）には
//! 該当しない。
//!
//! 各テストファイルは `mod support;` で読み込み、
//! `support::Xorshift64Star` を使う（本ファイルは全ファイルが同じ関数を
//! 使うわけではないため `#![allow(dead_code)]` を付ける。`crates/autodiff/
//! tests/common/mod.rs` と同型）。

#![allow(dead_code)]

/// xorshift64* 状態。呼び出し元はシード値のみを指定し、生成される数列は
/// 完全に決定的になる。暗号学的に安全な PRNG ではないため、テスト入力
/// 生成以外（鍵・トークン生成等）に使わないこと（OWASP A02。
/// `.claude/rules/security.md`）。
pub struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    /// シードが 0 だと xorshift の不動点（常に 0 を返す）に陥るため、
    /// 0 は非零値（黄金比由来の定数）に補正する。
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    /// 次の 64bit 乱数を返す。
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// `[-1.0, 1.0)` の範囲に収まる f32 を返す。
    pub fn next_f32(&mut self) -> f32 {
        // 上位 24bit を仮数部として使い、[0, 1) の一様分布を作ってから [-1, 1) に写す。
        let bits = (self.next_u64() >> 40) as u32; // 24bit
        let unit = bits as f32 / (1u32 << 24) as f32; // [0, 1)
        unit * 2.0 - 1.0
    }

    /// 長さ `len` の f32 ベクトルを決定的に生成する。
    pub fn fill_vec(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next_f32()).collect()
    }
}
