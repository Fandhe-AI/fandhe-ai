//! speculative decoding 用の KV キャッシュ巻き戻しヘルパー（イシュー #2885。
//! 設計の正は `docs/facade-speculative-decoding-batching-design.md` §3・§4.1・§7）。
//!
//! # 役割
//!
//! speculative decoding（#2886 の `generate::speculative`）では、target が
//! draft の K トークンを 1 回の forward でまとめて検証する。この forward で
//! target／draft 双方の [`KvCache`] が K トークン分進むため、棄却位置より
//! 後ろをキャッシュから外す必要がある。本モジュールは設計 §4.1 の**案 (i)**
//! （検証 forward 前に `KvCache::clone()` を退避 → 棄却時に退避分へ復元 →
//! 受理分だけを `forward_step` で再 forward）を `generate` 配下の内部
//! ヘルパーとして提供する。`KvCache` は `Arc` 共有の `Tensor` を持つため
//! clone は O(1) で、ヘルパー自身はバッファを確保しない。
//!
//! # 不変条件との関係
//!
//! 復元は「過去に `forward_with_cache` が書いた `KvCache` 値を丸ごと書き戻す」
//! だけである。`k`／`v` を個別に操作したり任意の `Tensor` を注入したりしない
//! ため、「k／v は両方 Some か両方 None」「書き手は `forward_with_cache`
//! のみ」の不変条件（`nn/attention.rs` の `KvCache` doc）は緩めない。`KvCache`
//! への公開メソッド追加（案 (ii)）や `pub(crate)` 切り詰め（案 (iii)）は
//! 設計 §10 論点 4 の承認事項であり本モジュールでは採らない。
//!
//! # 可視性
//!
//! 項目は `pub(super)`（見える範囲は `crate::generate` 配下＝将来の兄弟
//! `speculative.rs` を含む）。`generate` 直下のヘルパーでは `pub(super)` が
//! `pub(crate)` と同じ範囲になるが、本モジュールは 1 段深いため `pub(super)`
//! は厳密に狭い（#2894 の申し送り）。関数名は `validate` で始めない
//! （facade `api_surface.rs` の `validate*` 非公開インベントリと紛れさせない）。
//!
//! # 対象外・失敗時の契約
//!
//! - `num_kv_layers() == 0` のモデルは巻き戻す対象がなく、黙って素通りさせず
//!   `AutodiffError::InvalidArgument` で拒否する（設計 §4.1・§10 論点 3）
//! - [`rewind_and_replay`] は失敗時に `Err` を返す前に退避時点の状態へ戻す
//!   （複数層の `forward_step` は層をまたいで原子的でないため）。呼び出し側は
//!   `Err` を伝播し、そのキャッシュで処理を続けないこと。[`KvSnapshot`] は値で
//!   消費され使い回せない
//! - B = 1 の制限は課さない（巻き戻し自体はバッチに依存しない。入口検査は
//!   設計 §7 の順序で #2886 が行う）
//!
//! # 暫定の `dead_code` 抑止
//!
//! #2886 が呼び出すまで非テストビルドでは未使用になるため、`mod.rs` の
//! `mod kv_rewind;` 宣言に `expect(dead_code)` を付けている。呼び出し開始後は
//! `unfulfilled_lint_expectations` が落ちるので、**#2886 でこの属性を外す**こと。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::KvCache;

use super::{AutoregressiveModel, validate_forward_step_output};

/// 検証 forward 前に退避した KV キャッシュ配列（案 (i)）。
///
/// [`KvSnapshot::capture`] が作り、[`rewind_and_replay`] が値で消費する。
#[derive(Debug)]
pub(super) struct KvSnapshot {
    caches: Vec<KvCache>,
    seq_len: usize,
    batch: Option<usize>,
    embed_dim: Option<usize>,
}

impl KvSnapshot {
    /// `caches` を退避する（`Arc` 共有の clone）。
    ///
    /// 層数ゼロ・層数不一致・層間で `seq_len`／`batch`／`embed_dim` が揃って
    /// いない場合は巻き戻し位置が定まらないため `Err`。
    pub(super) fn capture<M: AutoregressiveModel + ?Sized>(
        model: &M,
        caches: &[KvCache],
    ) -> Result<KvSnapshot, AutodiffError> {
        check_layer_count(model, caches.len())?;
        let (seq_len, batch, embed_dim) = common_state(caches)?;
        Ok(KvSnapshot {
            caches: caches.to_vec(),
            seq_len,
            batch,
            embed_dim,
        })
    }

    /// 退避時点の `S_cached`（全層共通）。
    pub(super) fn seq_len(&self) -> usize {
        self.seq_len
    }
}

fn check_layer_count<M: AutoregressiveModel + ?Sized>(
    model: &M,
    caches_len: usize,
) -> Result<(), AutodiffError> {
    let layers = model.num_kv_layers();
    if layers == 0 || caches_len == 0 {
        return Err(AutodiffError::InvalidArgument(
            "kv_rewind: num_kv_layers() == 0（状態なしモデル）は KV キャッシュ巻き戻しの対象外"
                .to_string(),
        ));
    }
    if caches_len != layers {
        return Err(AutodiffError::InvalidArgument(format!(
            "kv_rewind: caches の層数 {caches_len} が num_kv_layers() = {layers} と一致しない"
        )));
    }
    Ok(())
}

/// 全層で共通の `(seq_len, batch, embed_dim)` を返す。揃っていなければ `Err`。
type CommonState = (usize, Option<usize>, Option<usize>);

fn common_state(caches: &[KvCache]) -> Result<CommonState, AutodiffError> {
    let Some(first) = caches.first() else {
        return Err(AutodiffError::InvalidArgument(
            "kv_rewind: caches が空".to_string(),
        ));
    };
    let state = (first.seq_len(), first.batch(), first.embed_dim());
    for (i, c) in caches.iter().enumerate() {
        if (c.seq_len(), c.batch(), c.embed_dim()) != state {
            return Err(AutodiffError::InvalidArgument(format!(
                "kv_rewind: 層 {i} の KV キャッシュ状態が層 0 と揃っていない"
            )));
        }
    }
    Ok(state)
}

/// 退避時点の状態へ丸ごと書き戻す（`KvCache` 値の代入のみ）。
fn restore(caches: &mut [KvCache], snapshot: &KvSnapshot) -> Result<(), AutodiffError> {
    if caches.len() != snapshot.caches.len() {
        return Err(AutodiffError::InvalidArgument(format!(
            "kv_rewind: 復元先の層数 {} が退避時の {} と一致しない",
            caches.len(),
            snapshot.caches.len()
        )));
    }
    for (dst, src) in caches.iter_mut().zip(&snapshot.caches) {
        *dst = src.clone();
    }
    Ok(())
}

/// 退避分へ復元し、`accepted_ids`（`[B, L_acc]`）を再 forward する。
///
/// `L_acc == 0`（全棄却。speculative の正常結果）は復元のみで `Ok(None)`。
/// `L_acc >= 1` は再 forward の logits（`[B, L_acc, V]`）を `Ok(Some(..))` で返す。
/// 失敗時は退避時点へ戻してから `Err` を返す（モジュール doc 参照）。
pub(super) fn rewind_and_replay<M: AutoregressiveModel + ?Sized>(
    model: &M,
    caches: &mut [KvCache],
    snapshot: KvSnapshot,
    accepted_ids: &Tensor<i32>,
    expected_vocab: usize,
) -> Result<Option<Tensor<f32>>, AutodiffError> {
    // 層数検査（ここまで caches に触れない）
    check_layer_count(model, caches.len())?;
    if caches.len() != snapshot.caches.len() {
        return Err(AutodiffError::InvalidArgument(format!(
            "kv_rewind: caches の層数 {} が退避時の {} と一致しない",
            caches.len(),
            snapshot.caches.len()
        )));
    }
    // accepted_ids 検査（復元前に拒否）
    let shape = accepted_ids.shape();
    if shape.len() != 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: shape.len(),
        }));
    }
    let (b, l_acc) = (shape[0], shape[1]);
    if let Some(snap_b) = snapshot.batch
        && snap_b != b
    {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![snap_b, l_acc],
        }));
    }
    let expected_seq = snapshot.seq_len.checked_add(l_acc).ok_or_else(|| {
        AutodiffError::InvalidArgument(
            "kv_rewind: seq_len + L_acc が usize をオーバーフローする".to_string(),
        )
    })?;

    restore(caches, &snapshot)?;
    // 全棄却は speculative の正常な結果なのでエラーにしない
    if l_acc == 0 {
        return Ok(None);
    }
    match replay(
        model,
        caches,
        &snapshot,
        accepted_ids,
        (b, l_acc),
        expected_seq,
        expected_vocab,
    ) {
        Ok(logits) => Ok(Some(logits)),
        Err(e) => {
            // 層をまたいで原子的でない forward_step の途中状態を残さない。
            // 復元自体は長さ検査済みで失敗しないため、元のエラーを優先して返す。
            let _ = restore(caches, &snapshot);
            Err(e)
        }
    }
}

fn replay<M: AutoregressiveModel + ?Sized>(
    model: &M,
    caches: &mut [KvCache],
    snapshot: &KvSnapshot,
    accepted_ids: &Tensor<i32>,
    (b, l_acc): (usize, usize),
    expected_seq: usize,
    expected_vocab: usize,
) -> Result<Tensor<f32>, AutodiffError> {
    let logits = model.forward_step(accepted_ids, caches)?;
    let vocab = validate_forward_step_output(&logits, b, l_acc)?;
    if vocab != expected_vocab {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: logits.shape().to_vec(),
            rhs: vec![b, l_acc, expected_vocab],
        }));
    }
    let (seq_len, batch, embed_dim) = common_state(caches)?;
    if seq_len != expected_seq
        || batch != Some(b)
        || (snapshot.embed_dim.is_some() && embed_dim != snapshot.embed_dim)
    {
        return Err(AutodiffError::InvalidArgument(format!(
            "kv_rewind: 再 forward 後の KV キャッシュが期待と異なる \
             （seq_len {seq_len} / 期待 {expected_seq}、batch {batch:?} / 期待 {b}）"
        )));
    }
    Ok(logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;
    use crate::nn::{LinearVars, MultiheadAttentionVars};
    use std::cell::Cell;

    const V: usize = 4;
    const E: usize = 4;
    const HEADS: usize = 2;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("fixture: shape とデータ長は一致させている")
    }

    fn seq(seed: i64, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((seed + i as i64 * 7) % 13 - 6) as f32 * 0.05)
            .collect()
    }

    fn ids(v: Vec<i32>, b: usize) -> Tensor<i32> {
        let l = v.len() / b;
        Tensor::new(v, &[b, l]).expect("fixture")
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Fault {
        None,
        /// N 回目（1 始まり）の forward_step 呼び出しで Err
        FailAtCall(usize),
        /// キャッシュを一切進めない
        SkipCache,
        /// 第 0 層のみ進める
        OnlyFirstLayer,
        /// 語彙サイズを V + 1 にする
        BadVocab,
        /// L 軸を +1 にする
        BadLen,
    }

    /// `Embedding → MHA（KV キャッシュ付き）× layers → lm head` のテスト専用モデル。
    struct TestModel {
        layers: usize,
        fault: Fault,
        calls: Cell<usize>,
    }

    impl TestModel {
        fn new(layers: usize, fault: Fault) -> TestModel {
            TestModel {
                layers,
                fault,
                calls: Cell::new(0),
            }
        }
    }

    impl AutoregressiveModel for TestModel {
        fn num_kv_layers(&self) -> usize {
            self.layers
        }

        fn forward_step(
            &self,
            new_ids: &Tensor<i32>,
            caches: &mut [KvCache],
        ) -> Result<Tensor<f32>, AutodiffError> {
            let n = self.calls.get() + 1;
            self.calls.set(n);
            if self.fault == Fault::FailAtCall(n) {
                return Err(AutodiffError::InvalidArgument("fixture: 注入失敗".into()));
            }
            let tape = Tape::new();
            let emb = tape.var(&t(seq(1, V * E), &[V, E]));
            let mut x = emb.embedding(new_ids, None)?;
            let run = match self.fault {
                Fault::SkipCache => 0,
                Fault::OnlyFirstLayer => 1.min(self.layers),
                _ => self.layers,
            };
            for cache in caches.iter_mut().take(run) {
                let lin = |s: i64| LinearVars {
                    weight: tape.var(&t(seq(s, E * E), &[E, E])),
                    bias: Some(tape.var(&t(seq(s + 1, E), &[E]))),
                };
                let mha = MultiheadAttentionVars::new(HEADS, lin(2), lin(4), lin(6), lin(8))?;
                x = mha.forward_with_cache(&x, &x, &x, cache)?;
            }
            let (b, l) = (new_ids.shape()[0], new_ids.shape()[1]);
            let flat = x.reshape(&[b * l, E])?;
            let lm = LinearVars {
                weight: tape.var(&t(seq(10, E * V), &[E, V])),
                bias: Some(tape.var(&t(seq(9, V), &[V]))),
            };
            let logits = lm.forward(&flat)?.reshape(&[b, l, V])?.to_tensor();
            Ok(match self.fault {
                Fault::BadVocab => t(vec![0.0; b * l * (V + 1)], &[b, l, V + 1]),
                Fault::BadLen => t(vec![0.0; b * (l + 1) * V], &[b, l + 1, V]),
                _ => logits,
            })
        }
    }

    fn bits(t: &Tensor<f32>) -> Vec<u32> {
        t.contiguous()
            .host_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    /// `KvCache` は `PartialEq` を持たない（公開面を広げない）ため bit 一致で比較する。
    fn assert_kv_bit_eq(a: &[KvCache], b: &[KvCache]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert_eq!(x.is_empty(), y.is_empty());
            assert_eq!(x.seq_len(), y.seq_len());
            assert_eq!(x.batch(), y.batch());
            assert_eq!(x.embed_dim(), y.embed_dim());
            for (p, q) in [(x.k(), y.k()), (x.v(), y.v())] {
                match (p, q) {
                    (None, None) => {}
                    (Some(p), Some(q)) => {
                        assert_eq!(p.shape(), q.shape());
                        assert_eq!(bits(p), bits(q));
                    }
                    _ => panic!("k/v の有無が不一致"),
                }
            }
        }
    }

    /// prefill（3 トークン）済みのモデルとキャッシュを作る。
    fn prefilled(layers: usize, fault: Fault) -> (TestModel, Vec<KvCache>) {
        let model = TestModel::new(layers, fault);
        let mut caches = vec![KvCache::new(); layers];
        TestModel::new(layers, Fault::None)
            .forward_step(&ids(vec![0, 1, 2], 1), &mut caches)
            .expect("prefill");
        (model, caches)
    }

    // T1: 往復で検証 forward 前と bit 一致する
    #[test]
    fn restore_returns_to_pre_verify_state() {
        for layers in [1, 2] {
            let (model, mut caches) = prefilled(layers, Fault::None);
            let before = caches.clone();
            let snap = KvSnapshot::capture(&model, &caches).unwrap();
            assert_eq!(snap.seq_len(), 3);
            model
                .forward_step(&ids(vec![3, 0, 1, 2], 1), &mut caches)
                .unwrap();
            assert_eq!(caches[0].seq_len(), 7);
            // 検証 forward が退避側を書き換えていない
            assert_kv_bit_eq(&snap.caches, &before);
            restore(&mut caches, &snap).unwrap();
            assert_kv_bit_eq(&caches, &before);
        }
    }

    fn reference(layers: usize, accepted: Vec<i32>) -> (Vec<KvCache>, Tensor<f32>) {
        let (model, mut caches) = prefilled(layers, Fault::None);
        let logits = model.forward_step(&ids(accepted, 1), &mut caches).unwrap();
        (caches, logits)
    }

    // T2・T4: 部分受理・全受理の再 forward が新規キャッシュの同分割と bit 一致する
    #[test]
    fn replay_matches_fresh_split_forward() {
        for layers in [1, 2] {
            for accepted in [vec![3], vec![3, 0], vec![3, 0, 1, 2]] {
                let (model, mut caches) = prefilled(layers, Fault::None);
                let snap = KvSnapshot::capture(&model, &caches).unwrap();
                model
                    .forward_step(&ids(vec![3, 0, 1, 2], 1), &mut caches)
                    .unwrap();
                let got =
                    rewind_and_replay(&model, &mut caches, snap, &ids(accepted.clone(), 1), V)
                        .unwrap()
                        .expect("L_acc >= 1");
                let (want_caches, want_logits) = reference(layers, accepted.clone());
                assert_eq!(caches[0].seq_len(), 3 + accepted.len());
                assert_kv_bit_eq(&caches, &want_caches);
                assert_eq!(bits(&got), bits(&want_logits));
            }
        }
    }

    // T3: 全棄却は復元のみ・forward を呼ばない
    #[test]
    fn zero_accepted_restores_without_forward() {
        let (model, mut caches) = prefilled(2, Fault::None);
        let before = caches.clone();
        let snap = KvSnapshot::capture(&model, &caches).unwrap();
        model
            .forward_step(&ids(vec![3, 0], 1), &mut caches)
            .unwrap();
        let calls = model.calls.get();
        let empty = Tensor::new(Vec::<i32>::new(), &[1, 0]).unwrap();
        let out = rewind_and_replay(&model, &mut caches, snap, &empty, V).unwrap();
        assert!(out.is_none());
        assert_eq!(model.calls.get(), calls, "forward を呼ばない");
        assert_kv_bit_eq(&caches, &before);
    }

    // T5: 状態なしモデルは対象外
    #[test]
    fn stateless_model_is_rejected() {
        let model = TestModel::new(0, Fault::None);
        let mut caches: Vec<KvCache> = Vec::new();
        assert!(KvSnapshot::capture(&model, &caches).is_err());
        let snap = KvSnapshot {
            caches: Vec::new(),
            seq_len: 0,
            batch: None,
            embed_dim: None,
        };
        assert!(rewind_and_replay(&model, &mut caches, snap, &ids(vec![0], 1), V).is_err());
    }

    // T6: 層数不一致は caches に触れず拒否
    #[test]
    fn layer_count_mismatch_is_rejected_without_touching_caches() {
        let (model, caches) = prefilled(2, Fault::None);
        assert!(KvSnapshot::capture(&model, &caches[..1]).is_err());
        let snap = KvSnapshot::capture(&model, &caches).unwrap();
        let mut short = caches[..1].to_vec();
        let before = short.clone();
        let r = rewind_and_replay(&model, &mut short, snap, &ids(vec![0], 1), V);
        assert!(r.is_err());
        assert_kv_bit_eq(&short, &before);
    }

    fn assert_fault_restores(fault: Fault) {
        let (model, mut caches) = prefilled(2, fault);
        let before = caches.clone();
        let snap = KvSnapshot::capture(&model, &caches).unwrap();
        model.calls.set(0);
        let r = rewind_and_replay(&model, &mut caches, snap, &ids(vec![3, 0], 1), V);
        assert!(r.is_err());
        assert_kv_bit_eq(&caches, &before);
    }

    // T7: 再 forward の Err は伝播し退避時点へ戻る
    #[test]
    fn forward_error_propagates_and_restores() {
        assert_fault_restores(Fault::FailAtCall(1));
    }

    // T8: キャッシュが進まない／一部の層だけ進む
    #[test]
    fn cache_not_advanced_is_detected_and_restores() {
        assert_fault_restores(Fault::SkipCache);
        assert_fault_restores(Fault::OnlyFirstLayer);
    }

    // T9: logits の shape・vocab 不一致
    #[test]
    fn bad_logits_are_detected_and_restore() {
        assert_fault_restores(Fault::BadVocab);
        assert_fault_restores(Fault::BadLen);
        let (model, mut caches) = prefilled(1, Fault::None);
        let before = caches.clone();
        let snap = KvSnapshot::capture(&model, &caches).unwrap();
        let r = rewind_and_replay(&model, &mut caches, snap, &ids(vec![3], 1), V + 1);
        assert!(r.is_err());
        assert_kv_bit_eq(&caches, &before);
    }

    // T10: accepted_ids の rank・バッチ不一致は復元前に拒否（caches 不変）
    #[test]
    fn bad_accepted_ids_rejected_before_restore() {
        let (model, mut caches) = prefilled(1, Fault::None);
        let snap_a = KvSnapshot::capture(&model, &caches).unwrap();
        let snap_b = KvSnapshot::capture(&model, &caches).unwrap();
        // 検証 forward でキャッシュを進めた状態を「不変」の基準にする
        model.forward_step(&ids(vec![3], 1), &mut caches).unwrap();
        let advanced = caches.clone();
        let rank1 = Tensor::new(vec![0i32, 1], &[2]).unwrap();
        assert!(rewind_and_replay(&model, &mut caches, snap_a, &rank1, V).is_err());
        assert_kv_bit_eq(&caches, &advanced);
        let batch2 = ids(vec![0, 1], 2);
        assert!(rewind_and_replay(&model, &mut caches, snap_b, &batch2, V).is_err());
        assert_kv_bit_eq(&caches, &advanced);
    }

    // T11: 層間で seq_len が揃っていないキャッシュは capture で拒否
    #[test]
    fn misaligned_layers_are_rejected_at_capture() {
        let (model, mut caches) = prefilled(2, Fault::None);
        let only0 = TestModel::new(2, Fault::OnlyFirstLayer);
        only0.forward_step(&ids(vec![3], 1), &mut caches).unwrap();
        assert_ne!(caches[0].seq_len(), caches[1].seq_len());
        assert!(KvSnapshot::capture(&model, &caches).is_err());
    }
}
