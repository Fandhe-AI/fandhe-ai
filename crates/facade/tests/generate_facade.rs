//! `fandhe_ai::inference::{AutoregressiveModel, GenerateConfig, SamplingStrategy, generate}`
//! の facade 公開経路（イシュー #2575・親 #2499）の単体テスト。
//!
//! 公開形の正は `docs/facade-generate-decision.md` §13.2・§17。facade は autodiff の
//! 自己回帰ループを純再エクスポートするだけなので、ここでは
//! - autodiff 直経路と同一型・同一結果であること
//! - 3 戦略の出力 shape・prompt 保持・seed 決定性・グローバル RNG 非消費
//! - 入力・設定・モデル出力の fail-closed 検査（OWASP A03/A04）が公開経路でも効くこと
//! - `Tape::stateful_attention_forward` を使う KV キャッシュ付きモデルが最後まで動き、
//!   キャッシュなしの全系列再計算と REQ-2 統一複合判定（`assert_parity`。tolerance 不変）で一致すること
//!
//! を固定する。すべて CPU で動くため `#[ignore]` は付けない（CUDA／Metal 実機 parity は
//! `generate_backend_parity.rs` と `docs/perf/logs/generate-2191/README.md` の申し送りを参照）。
//! 比較用の autodiff 直経路・`assert_parity` だけが内部クレートを直接参照する。

use std::cell::RefCell;

use fandhe_ai::inference::{AutoregressiveModel, GenerateConfig, SamplingStrategy, generate};
use fandhe_ai::nn::kv_cache::{KvCache, MultiheadAttentionConfig, StatefulAttention};
use fandhe_ai::{AutodiffError, Tensor};
use fandhe_ai_backend_cpu::parity::assert_parity;

const V: usize = 5;

fn ids(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("fixture: shape とデータ長は一致させている")
}

/// 故障の注入方法（セキュリティ検査用）。
#[derive(Clone, Copy)]
enum Fault {
    None,
    /// rank 2 の logits を返す。
    BadRank,
    /// 語彙サイズがステップ間で変化する。
    VocabDrift,
    /// 末尾位置の logits に NaN を含める。
    Nan,
}

/// 直前トークンだけで logits が決まる状態なしモデル（表引き）。
struct Table {
    fault: Fault,
    calls: RefCell<usize>,
}

impl Table {
    fn new(fault: Fault) -> Self {
        Self {
            fault,
            calls: RefCell::new(0),
        }
    }
}

impl AutoregressiveModel for Table {
    fn num_kv_layers(&self) -> usize {
        0
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        assert!(caches.is_empty(), "num_kv_layers == 0 なので caches は空");
        let call = {
            let mut c = self.calls.borrow_mut();
            *c += 1;
            *c
        };
        let (b, l) = (new_ids.shape()[0], new_ids.shape()[1]);
        let v = match self.fault {
            Fault::VocabDrift if call > 1 => V + 1,
            _ => V,
        };
        let src = new_ids.as_slice().expect("contiguous");
        let mut out = vec![0.0f32; b * l * v];
        for (i, &id) in src.iter().enumerate() {
            for j in 0..v {
                // 緩やかな非一様分布（TopK／Temperature でも複数候補が残る）。
                out[i * v + j] = (((id as usize * 3 + j * 5) % 7) as f32) * 0.4;
            }
        }
        if matches!(self.fault, Fault::Nan) {
            let last = (b * l - 1) * v;
            out[last] = f32::NAN;
        }
        if matches!(self.fault, Fault::BadRank) {
            return Tensor::new(out, &[b * l, v]).map_err(AutodiffError::from);
        }
        Tensor::new(out, &[b, l, v]).map_err(AutodiffError::from)
    }
}

#[test]
fn greedy_topk_temperature_output_shape_and_prompt_preserved() {
    let model = Table::new(Fault::None);
    for strategy in [
        SamplingStrategy::Greedy,
        SamplingStrategy::TopK(3),
        SamplingStrategy::Temperature(0.8),
    ] {
        let cfg = GenerateConfig::new(7, strategy).with_seed(3);

        // rank 1 入力は rank 1 出力。
        let p1 = ids(vec![1, 2], &[2]);
        let o1 = generate(&model, &p1, &cfg).expect("rank1");
        assert_eq!(o1.shape(), &[7]);
        assert_eq!(&o1.as_slice().unwrap()[..2], &[1, 2]);

        // rank 2 入力は [B, max_length]。
        let p2 = ids(vec![1, 2, 3, 4], &[2, 2]);
        let o2 = generate(&model, &p2, &cfg).expect("rank2");
        assert_eq!(o2.shape(), &[2, 7]);
        let s = o2.as_slice().unwrap();
        assert_eq!(&s[..2], &[1, 2]);
        assert_eq!(&s[7..9], &[3, 4]);
        assert!(s.iter().all(|&t| (0..V as i32).contains(&t)));
    }
}

#[test]
fn same_seed_is_deterministic_and_global_rng_is_untouched() {
    let model = Table::new(Fault::None);
    let prompt = ids(vec![0, 1], &[2]);
    let cfg = GenerateConfig::new(12, SamplingStrategy::Temperature(1.5)).with_seed(99);

    // このテストだけがグローバル RNG を触る（同一バイナリ内の他テストは触らない）。
    fandhe_ai::manual_seed(1234);
    let before = fandhe_ai::rand(&[4]).unwrap();
    fandhe_ai::manual_seed(1234);
    let a = generate(&model, &prompt, &cfg).unwrap();
    let after = fandhe_ai::rand(&[4]).unwrap();
    assert_eq!(
        before.as_slice().unwrap(),
        after.as_slice().unwrap(),
        "generate はグローバル RNG を消費してはならない"
    );

    let b = generate(&model, &prompt, &cfg).unwrap();
    assert_eq!(a.as_slice().unwrap(), b.as_slice().unwrap());
}

#[test]
fn facade_reexport_is_the_same_item_as_autodiff() {
    let model = Table::new(Fault::None);
    let prompt = ids(vec![2, 3], &[1, 2]);
    // 参照側: 内部クレート直経路。facade の GenerateConfig／SamplingStrategy をそのまま渡せる
    // こと自体が同一型であることの証明になる。
    let cfg = GenerateConfig::new(8, SamplingStrategy::TopK(2)).with_seed(5);
    let via_facade = generate(&model, &prompt, &cfg).unwrap();
    let via_autodiff = fandhe_ai_autodiff::generate::generate(&model, &prompt, &cfg).unwrap();
    assert_eq!(
        via_facade.as_slice().unwrap(),
        via_autodiff.as_slice().unwrap()
    );
}

#[test]
fn config_constructors_and_public_fields() {
    let c = GenerateConfig::new(4, SamplingStrategy::TopK(2))
        .with_temperature(0.7)
        .with_seed(9);
    assert_eq!(c.max_length, 4);
    assert_eq!(c.temperature, 0.7);
    assert_eq!(c.top_k, Some(2));
    assert!(matches!(c.strategy, SamplingStrategy::TopK(2)));
    assert_eq!(c.seed, 9);
}

fn assert_rejected(r: Result<Tensor<i32>, AutodiffError>, what: &str) {
    match r {
        Err(AutodiffError::InvalidArgument(_)) | Err(AutodiffError::Shape(_)) => {}
        other => panic!("{what}: fail-closed の Err を期待したが {other:?}"),
    }
}

#[test]
fn invalid_inputs_and_configs_are_rejected() {
    let model = Table::new(Fault::None);
    let ok_prompt = ids(vec![1, 2], &[2]);

    assert_rejected(
        generate(
            &model,
            &ok_prompt,
            &GenerateConfig::new(1, SamplingStrategy::Greedy),
        ),
        "max_length < prompt 長",
    );
    assert_rejected(
        generate(
            &model,
            &ids(vec![], &[0]),
            &GenerateConfig::new(3, SamplingStrategy::Greedy),
        ),
        "空 prompt",
    );
    assert_rejected(
        generate(
            &model,
            &ids(vec![1, 2], &[1, 1, 2]),
            &GenerateConfig::new(3, SamplingStrategy::Greedy),
        ),
        "rank 3 入力",
    );
    assert_rejected(
        generate(
            &model,
            &ok_prompt,
            &GenerateConfig::new(4, SamplingStrategy::TopK(0)),
        ),
        "TopK(0)",
    );
    assert_rejected(
        generate(
            &model,
            &ok_prompt,
            &GenerateConfig::new(4, SamplingStrategy::TopK(V + 1)),
        ),
        "top_k > vocab",
    );

    // pub フィールドを書き換えて矛盾させた設定は validate が拒否する。
    let mut bad = GenerateConfig::new(4, SamplingStrategy::Greedy);
    bad.temperature = 2.0;
    assert_rejected(generate(&model, &ok_prompt, &bad), "Greedy + temperature");
    let mut bad = GenerateConfig::new(4, SamplingStrategy::Temperature(0.5));
    bad.top_k = Some(2);
    assert_rejected(generate(&model, &ok_prompt, &bad), "Temperature + top_k");
}

#[test]
fn untrusted_model_output_is_rejected() {
    let prompt = ids(vec![1, 2], &[2]);
    let cfg = GenerateConfig::new(6, SamplingStrategy::Greedy);
    for (fault, what) in [
        (Fault::BadRank, "rank 不正の logits"),
        (Fault::VocabDrift, "ステップ間の語彙サイズ変化"),
        (Fault::Nan, "末尾位置の NaN"),
    ] {
        assert_rejected(generate(&Table::new(fault), &prompt, &cfg), what);
    }
}

// ---------------------------------------------------------------------
// KV キャッシュ付きモデル（facade の公開 API のみ）
// ---------------------------------------------------------------------

const E: usize = 4;
const H: usize = 2;
const ATT_SEED: u64 = 17;

fn embed(id: i32) -> [f32; E] {
    let mut r = [0.0; E];
    for (k, x) in r.iter_mut().enumerate() {
        *x = (((id as usize * 5 + k * 3) % 11) as f32 - 5.0) * 0.07;
    }
    r
}

fn proj(y: &[f32]) -> [f32; V] {
    let mut r = [0.0; V];
    for (j, o) in r.iter_mut().enumerate() {
        for (k, &v) in y.iter().enumerate() {
            *o += v * (((k * 7 + j * 3) % 9) as f32 - 4.0) * 0.1;
        }
    }
    r
}

fn embed_batch(new_ids: &Tensor<i32>) -> Tensor<f32> {
    let (b, l) = (new_ids.shape()[0], new_ids.shape()[1]);
    let data: Vec<f32> = new_ids
        .as_slice()
        .unwrap()
        .iter()
        .flat_map(|&i| embed(i))
        .collect();
    Tensor::new(data, &[b, l, E]).unwrap()
}

fn project_all(y: &Tensor<f32>) -> Tensor<f32> {
    let (b, l) = (y.shape()[0], y.shape()[1]);
    let data: Vec<f32> = y.as_slice().unwrap().chunks(E).flat_map(proj).collect();
    Tensor::new(data, &[b, l, V]).unwrap()
}

/// `Tape::stateful_attention_forward`（KV キャッシュ）を使うモデル。facade からは
/// `forward_step` の `caches` へ書き込めないため、`StatefulAttention` を自前で持つ
/// （`num_kv_layers() == 0`。`inference` モジュール doc の既知の制限）。
/// `generate` 1 回につき新しいインスタンスを作ること（キャッシュ reset は利用者責務）。
struct CachedLm {
    sa: RefCell<StatefulAttention>,
    logits_log: RefCell<Vec<Vec<f32>>>,
}

impl CachedLm {
    fn fresh() -> Self {
        Self {
            sa: RefCell::new(
                StatefulAttention::from_config(&MultiheadAttentionConfig::new(E, H), ATT_SEED)
                    .expect("from_config"),
            ),
            logits_log: RefCell::new(Vec::new()),
        }
    }
}

impl AutoregressiveModel for CachedLm {
    fn num_kv_layers(&self) -> usize {
        0
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        _caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let tape = fandhe_ai::tape();
        let y = tape.stateful_attention_forward(
            &mut self.sa.borrow_mut(),
            &tape.var(&embed_batch(new_ids)),
        )?;
        let logits = project_all(&y.to_tensor());
        self.logits_log
            .borrow_mut()
            .push(logits.as_slice().unwrap().to_vec());
        Ok(logits)
    }
}

/// キャッシュを使わず、毎ステップ全履歴を新しい `StatefulAttention` で prefill し直す参照。
struct RecomputeLm {
    history: RefCell<Vec<Vec<i32>>>,
    logits_log: RefCell<Vec<Vec<f32>>>,
}

impl AutoregressiveModel for RecomputeLm {
    fn num_kv_layers(&self) -> usize {
        0
    }

    fn forward_step(
        &self,
        new_ids: &Tensor<i32>,
        _caches: &mut [KvCache],
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (b, l_new) = (new_ids.shape()[0], new_ids.shape()[1]);
        let mut hist = self.history.borrow_mut();
        if hist.is_empty() {
            hist.resize(b, Vec::new());
        }
        let src = new_ids.as_slice().unwrap();
        for (bi, h) in hist.iter_mut().enumerate() {
            h.extend_from_slice(&src[bi * l_new..(bi + 1) * l_new]);
        }
        let total = hist[0].len();
        let flat: Vec<i32> = hist.iter().flatten().copied().collect();
        let full_ids = Tensor::new(flat, &[b, total]).unwrap();

        let mut sa =
            StatefulAttention::from_config(&MultiheadAttentionConfig::new(E, H), ATT_SEED)?;
        let tape = fandhe_ai::tape();
        let y = tape.stateful_attention_forward(&mut sa, &tape.var(&embed_batch(&full_ids)))?;
        let logits = project_all(&y.to_tensor());

        // 末尾 l_new 位置だけを切り出す。
        let all = logits.as_slice().unwrap();
        let mut out = Vec::with_capacity(b * l_new * V);
        for bi in 0..b {
            let base = (bi * total + (total - l_new)) * V;
            out.extend_from_slice(&all[base..base + l_new * V]);
        }
        self.logits_log.borrow_mut().push(out.clone());
        Tensor::new(out, &[b, l_new, V]).map_err(AutodiffError::from)
    }
}

#[test]
fn kv_cached_model_runs_and_matches_full_recompute() {
    let prompt = ids(vec![1, 3, 0, 2, 4, 1], &[2, 3]);
    for strategy in [SamplingStrategy::Greedy, SamplingStrategy::TopK(3)] {
        let cfg = GenerateConfig::new(8, strategy).with_seed(21);

        let cached = CachedLm::fresh();
        let out_cached = generate(&cached, &prompt, &cfg).expect("cached");
        assert_eq!(out_cached.shape(), &[2, 8]);

        let reference = RecomputeLm {
            history: RefCell::new(Vec::new()),
            logits_log: RefCell::new(Vec::new()),
        };
        let out_ref = generate(&reference, &prompt, &cfg).expect("reference");

        // 各 forward_step の logits を REQ-2 統一複合判定で突合する（tolerance 不変）。
        let a = cached.logits_log.borrow();
        let b = reference.logits_log.borrow();
        assert_eq!(a.len(), b.len());
        assert_eq!(
            a.len(),
            5,
            "prefill 1 回 + decode (max_length - prompt - 1) 回"
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_parity(&format!("generate logits step {i}"), x, y);
        }
        assert_eq!(out_cached.as_slice().unwrap(), out_ref.as_slice().unwrap());
    }
}
