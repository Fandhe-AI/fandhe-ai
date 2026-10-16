//! KV キャッシュ付き attention（`fandhe_ai_autodiff::nn` の `KvCache`・
//! `StatefulAttention`）を facade へ再エクスポートする**純再エクスポート
//! モジュール**（`crate::nn::rnn` と同型。facade 独自の型・関数は持ち込まない）。
//!
//! イシュー #2579（親 #2499 のルート一括承認〈2026-10-04〉・#2577／#2579 の承認記録）。
//! 公開形の正は `docs/kv-cache-design.md` §11.4（P1〜P4）。再エクスポートするのは
//! `KvCache`・`StatefulAttention`・`MultiheadAttentionConfig` の 3 名のみ。
//!
//! # `Sequential::add_*` を設けない理由
//!
//! `StatefulAttention` は `Module` を実装せず、`forward` が `&mut self` でキャッシュを
//! 更新する（`Module::forward` は `&self` で状態を持たない前提のため。
//! `docs/kv-cache-design.md` §9）。`compat::Sequential` の `Var → Var` 平坦鎖とは
//! 構造的に合わないため、独立モジュールとして公開する。
//!
//! # forward の入口
//!
//! `StatefulAttention::forward` は第 1 引数に生の `fandhe_ai_autodiff::Tape` を取るが、
//! facade の [`crate::Tape`] は newtype でこれを取り出せない（REQ-12）。入口は
//! [`crate::Tape::stateful_attention_forward`]（`&self.0` を渡すだけの薄い委譲）で、
//! `forward_with_cache` は公開しない。構築は `StatefulAttention::from_config`
//! （`MultiheadAttention` 型を facade が公開しないための入口）を使う。
//!
//! 外部の `KvCache` を進める版として [`crate::Tape::stateful_attention_forward_with_cache`]
//! も提供する（イシュー #3096。`docs/facade-speculative-decoding-batching-design.md` §19.4）。
//! 内部キャッシュ版は `sa` 自身のキャッシュを進め（`&mut sa`）、外部キャッシュ版は
//! `&sa` と `&mut KvCache` を受けて `sa` の内部キャッシュに触れない。
//! `AutoregressiveModel::forward_step` の中で `caches[i]` を進める用途は後者を使う。
//! `KvCache` の公開メソッド追加は `valid_len` の 1 件だけ（#3105・同 §21.5）で、
//! 巻き戻しフックは公開しない（同 §19.9）。
//!
//! 公開しないもの: `MultiheadAttention`・`MultiheadAttentionVars`・`forward_with_cache`。
//!
//! # 既知の制限
//!
//! - `StatefulAttention::new(mha)`／`mha()` は facade から名指しできない型を扱う
//!   メソッドとして残る（`docs/kv-cache-design.md` §11.4 P2 の残課題）
//! - self-attention 限定で、`batch_first=true`・`kdim = vdim = embed_dim` のみ対応する
//!   （それ以外の config は `from_config` が `InvalidArgument` で拒否する）
//! - padding 付き（可変長）キャッシュは crate 内部（#3106 の束ねる経路）だけが作る。公開面で
//!   使えるのは `valid_len` の参照だけで、任意の mask・`Tensor` を入れる経路はない（§21.4）
//! - 過去ステップへ勾配は流れない（キャッシュは勾配なしの葉。現ステップの射影
//!   パラメータへは流れる）。decode ループでは `Tape` の再作成か reset を推奨する
//! - デバイス常駐キャッシュ（K-3）は対象外
//!
//! # 利用例（prefill → decode）
//!
//! ```
//! use fandhe_ai::nn::kv_cache::{MultiheadAttentionConfig, StatefulAttention};
//! use fandhe_ai::Tensor;
//!
//! let tape = fandhe_ai::tape();
//! let mut sa = StatefulAttention::from_config(&MultiheadAttentionConfig::new(4, 2), 0).unwrap();
//!
//! // prefill: x は [B=1, L=3, E=4]
//! let x = Tensor::new((0..12).map(|i| i as f32 * 0.1).collect::<Vec<f32>>(), &[1, 3, 4]).unwrap();
//! let y = tape.stateful_attention_forward(&mut sa, &tape.var(&x)).unwrap();
//! assert_eq!(y.to_tensor().shape(), &[1usize, 3, 4]);
//! assert_eq!(sa.seq_len(), 3);
//!
//! // decode: 1 トークン
//! let t = Tensor::new(vec![0.1f32, 0.2, 0.3, 0.4], &[1, 1, 4]).unwrap();
//! let y = tape.stateful_attention_forward(&mut sa, &tape.var(&t)).unwrap();
//! assert_eq!(y.to_tensor().shape(), &[1usize, 1, 4]);
//! assert_eq!(sa.seq_len(), 4);
//! ```
//!
//! # 利用例（外部キャッシュ）
//!
//! ```
//! use fandhe_ai::nn::kv_cache::{KvCache, MultiheadAttentionConfig, StatefulAttention};
//! use fandhe_ai::Tensor;
//!
//! let tape = fandhe_ai::tape();
//! let sa = StatefulAttention::from_config(&MultiheadAttentionConfig::new(4, 2), 0).unwrap();
//! let mut cache = KvCache::new();
//! let other = KvCache::new();
//!
//! // prefill: x は [B=1, L=3, E=4]
//! let x = Tensor::new((0..12).map(|i| i as f32 * 0.1).collect::<Vec<f32>>(), &[1, 3, 4]).unwrap();
//! tape.stateful_attention_forward_with_cache(&sa, &tape.var(&x), &mut cache).unwrap();
//! assert_eq!(cache.seq_len(), 3);
//!
//! // decode: 1 トークン
//! let t = Tensor::new(vec![0.1f32, 0.2, 0.3, 0.4], &[1, 1, 4]).unwrap();
//! tape.stateful_attention_forward_with_cache(&sa, &tape.var(&t), &mut cache).unwrap();
//! assert_eq!(cache.seq_len(), 4);
//!
//! // sa 自身と別のキャッシュは進まない
//! assert_eq!(sa.seq_len(), 0);
//! assert_eq!(other.seq_len(), 0);
//! ```

pub use fandhe_ai_autodiff::nn::{KvCache, MultiheadAttentionConfig, StatefulAttention};
