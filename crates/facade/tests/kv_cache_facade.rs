//! KV キャッシュ付き attention の facade 公開経路（`fandhe_ai::nn::kv_cache`・
//! `Tape::stateful_attention_forward`。イシュー #2579・親 #2499）の単体テスト。
//!
//! 公開形の正は `docs/kv-cache-design.md` §11.4。facade 経路（`StatefulAttention::from_config`
//! + `Tape::stateful_attention_forward`）の挙動を、
//! - autodiff 直経路（`StatefulAttention::new(MultiheadAttention::from_config(..))` +
//!   `forward`。`NaiveOps`）との出力一致（REQ-2 統一複合判定 `assert_parity`。tolerance 不変）
//! - 全系列 prefill との一致（prefill + decode × N・複数トークン追記〈mask 規則 (c)〉）
//! - キャッシュ状態の推移・reset・エラー経路（原子性・`TapeMismatch`）・`from_config` の拒否条件
//! - backward の成功
//!
//! で固定する。すべて CPU で動き `#[ignore]` は付けない（CUDA／Metal の decode 列 parity は
//! `kv_cache_backend_parity.rs` と `docs/perf/logs/kv-cache-2084/README.md` の申し送りを参照）。

use fandhe_ai::nn::kv_cache::{MultiheadAttentionConfig, StatefulAttention};
use fandhe_ai::{AutodiffError, Tensor};
use fandhe_ai_autodiff::nn::MultiheadAttention;
use fandhe_ai_backend_cpu::parity::assert_parity;

const B: usize = 2;
const E: usize = 4;
const H: usize = 2;
const TOTAL: usize = 5;
const SEED: u64 = 11;

fn cfg() -> MultiheadAttentionConfig {
    MultiheadAttentionConfig::new(E, H)
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture: shape とデータ長は一致させている")
}

fn full_input() -> Tensor<f32> {
    let data: Vec<f32> = (0..B * TOTAL * E)
        .map(|i| ((211 + i as i64) % 17 - 8) as f32 * 0.03)
        .collect();
    t(data, &[B, TOTAL, E])
}

/// `[B, TOTAL, E]` から時刻 `[from, to)` を切り出す（`[B, to-from, E]`）。
fn slice_time(x: &Tensor<f32>, from: usize, to: usize) -> Tensor<f32> {
    let src = x.as_slice().expect("fixture: contiguous");
    let mut out = Vec::with_capacity(B * (to - from) * E);
    for b in 0..B {
        let base = b * TOTAL * E;
        out.extend_from_slice(&src[base + from * E..base + to * E]);
    }
    t(out, &[B, to - from, E])
}

fn data_of(t: &Tensor<f32>) -> Vec<f32> {
    t.as_slice().expect("contiguous").to_vec()
}

/// facade 経路で `chunks`（各要素の時刻長）の順に forward し、各ステップ出力をまとめて返す。
fn run_facade(x: &Tensor<f32>, chunks: &[usize]) -> (StatefulAttention, Vec<Vec<f32>>) {
    let tape = fandhe_ai::tape();
    let mut sa = StatefulAttention::from_config(&cfg(), SEED).expect("from_config");
    let mut outs = Vec::new();
    let mut at = 0;
    for &len in chunks {
        let step = slice_time(x, at, at + len);
        let y = tape
            .stateful_attention_forward(&mut sa, &tape.var(&step))
            .expect("forward");
        assert_eq!(y.to_tensor().shape(), &[B, len, E]);
        outs.push(data_of(&y.to_tensor()));
        at += len;
    }
    (sa, outs)
}

/// 各ステップ出力を `[B, len, E]` の行列として時刻方向へ連結した `[B, TOTAL, E]` 相当の
/// フラット列にする（バッチごとに時刻順へ並べ直す）。
fn concat_time(chunks: &[usize], outs: &[Vec<f32>]) -> Vec<f32> {
    let mut full = Vec::with_capacity(B * TOTAL * E);
    for b in 0..B {
        for (len, o) in chunks.iter().zip(outs) {
            full.extend_from_slice(&o[b * len * E..(b + 1) * len * E]);
        }
    }
    full
}

#[test]
fn facade_path_matches_autodiff_direct_path_for_prefill_then_decode() {
    let x = full_input();
    let chunks = [2usize, 1, 1, 1];
    let (_sa, facade_outs) = run_facade(&x, &chunks);

    let naive = fandhe_ai_autodiff::Tape::new();
    let mut direct = fandhe_ai_autodiff::nn::StatefulAttention::new(
        MultiheadAttention::from_config(&cfg(), SEED).expect("direct from_config"),
    );
    let mut at = 0;
    for (len, facade_out) in chunks.iter().zip(&facade_outs) {
        let step = slice_time(&x, at, at + len);
        let y = direct.forward(&naive, &naive.var(&step)).expect("direct");
        assert_parity(
            "StatefulAttention facade vs autodiff direct",
            facade_out,
            &data_of(&y.to_tensor()),
        );
        at += len;
    }
}

#[test]
fn incremental_decode_matches_full_prefill() {
    let x = full_input();
    let (_s, full) = run_facade(&x, &[TOTAL]);
    for chunks in [
        vec![2usize, 1, 1, 1],
        vec![1, 1, 1, 1, 1],
        // 複数トークン追記（mask 規則 (c)）。
        vec![2, 3],
        vec![3, 2],
    ] {
        let (sa, outs) = run_facade(&x, &chunks);
        assert_eq!(sa.seq_len(), TOTAL, "{chunks:?}");
        assert_parity(
            "incremental vs full prefill",
            &concat_time(&chunks, &outs),
            &full[0],
        );
    }
}

#[test]
fn cache_state_transitions_and_reset() {
    let x = full_input();
    let tape = fandhe_ai::tape();
    let mut sa = StatefulAttention::from_config(&cfg(), SEED).expect("from_config");
    assert_eq!(sa.seq_len(), 0);
    assert!(sa.cache().is_empty());

    tape.stateful_attention_forward(&mut sa, &tape.var(&slice_time(&x, 0, 3)))
        .expect("prefill");
    assert_eq!(sa.seq_len(), 3);
    assert_eq!(sa.cache().batch(), Some(B));
    assert_eq!(sa.cache().embed_dim(), Some(E));

    tape.stateful_attention_forward(&mut sa, &tape.var(&slice_time(&x, 3, 4)))
        .expect("decode");
    assert_eq!(sa.seq_len(), 4);

    sa.reset_cache();
    assert_eq!(sa.seq_len(), 0);
    assert_eq!(sa.cache().batch(), None);
    // reset 後に別バッチ数でも再 prefill できる。
    let one = t(vec![0.1; 3 * E], &[1, 3, E]);
    tape.stateful_attention_forward(&mut sa, &tape.var(&one))
        .expect("re-prefill");
    assert_eq!(sa.seq_len(), 3);
    assert_eq!(sa.cache().batch(), Some(1));
}

#[test]
fn error_paths_leave_cache_unchanged() {
    let x = full_input();
    let tape = fandhe_ai::tape();
    let mut sa = StatefulAttention::from_config(&cfg(), SEED).expect("from_config");
    tape.stateful_attention_forward(&mut sa, &tape.var(&slice_time(&x, 0, 2)))
        .expect("prefill");
    let before = sa.seq_len();

    // rank 不正（2 次元）。
    let rank2 = t(vec![0.1; B * E], &[B, E]);
    assert!(
        tape.stateful_attention_forward(&mut sa, &tape.var(&rank2))
            .is_err()
    );
    // E 不一致。
    let bad_e = t(vec![0.1; B * 3], &[B, 1, 3]);
    assert!(
        tape.stateful_attention_forward(&mut sa, &tape.var(&bad_e))
            .is_err()
    );
    // キャッシュの B と不一致。
    let bad_b = t(vec![0.1; E], &[1, 1, E]);
    assert!(
        tape.stateful_attention_forward(&mut sa, &tape.var(&bad_b))
            .is_err()
    );
    assert_eq!(sa.seq_len(), before, "エラー時にキャッシュが変化した");

    // 別 Tape の Var は TapeMismatch。
    let other = fandhe_ai::tape();
    let foreign = other.var(&slice_time(&x, 2, 3));
    let err = tape
        .stateful_attention_forward(&mut sa, &foreign)
        .expect_err("別 Tape の Var は拒否される");
    assert!(matches!(err, AutodiffError::TapeMismatch), "{err:?}");
    assert_eq!(sa.seq_len(), before);
}

#[test]
fn from_config_rejects_unsupported_configs() {
    for c in [
        MultiheadAttentionConfig::new(E, H).with_batch_first(false),
        MultiheadAttentionConfig::new(E, H).with_kdim(E + 1),
        MultiheadAttentionConfig::new(E, H).with_vdim(E + 1),
        // embed_dim が num_heads で割り切れない。
        MultiheadAttentionConfig::new(E + 1, H),
    ] {
        assert!(
            matches!(
                StatefulAttention::from_config(&c, 0),
                Err(AutodiffError::InvalidArgument(_))
            ),
            "{c:?}"
        );
    }
}

#[test]
fn backward_from_current_step_output_succeeds() {
    let x = full_input();
    let tape = fandhe_ai::tape();
    let mut sa = StatefulAttention::from_config(&cfg(), SEED).expect("from_config");
    tape.stateful_attention_forward(&mut sa, &tape.var(&slice_time(&x, 0, 3)))
        .expect("prefill");
    let y = tape
        .stateful_attention_forward(&mut sa, &tape.var(&slice_time(&x, 3, 4)))
        .expect("decode");
    let loss = y.sum(None).expect("sum");
    tape.backward(&loss).expect("backward");
}
