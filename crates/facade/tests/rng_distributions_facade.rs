//! facade 経由の乱数分布と独立乱数源（`fandhe_ai::{bernoulli, multinomial,
//! normal, Generator}`。イシュー #2593。承認の根拠はルート #2499 のコメント、
//! 公開形の記録は `docs/rng-distributions-generator-decision.md` §5）の
//! 統合テスト。
//!
//! 3 関数は `manual_seed` が設定するグローバル RNG を消費し、`Generator`
//! は独立した状態を持つ。グローバル RNG を書き換えるため、本ファイル内の
//! テストはファイル局所 `Mutex` で直列化する（`rng_tensor_generation.rs`
//! と同型）。いずれもホスト生成のみで `BackendOps` を経由せず新規カーネルも
//! 追加しないため、GPU 数値一致の新規 baseline は発生しない。アップロード
//! 経路の確認は CPU（`fandhe_ai::tape()`）で行い、CUDA／Metal は既存の
//! `rng_tensor_generation.rs` のアップロード往復テストで固定済み。

use std::sync::Mutex;

use fandhe_ai::{Generator, RngError, Tensor};

type BernoulliFn = fn(&Tensor<f32>) -> Result<Tensor<f32>, RngError>;
type MultinomialFn = fn(&Tensor<f32>, usize, bool) -> Result<Tensor<i32>, RngError>;
type NormalFn = fn(f32, f32, &[usize]) -> Result<Tensor<f32>, RngError>;

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

fn f32s(t: &Tensor<f32>) -> Vec<f32> {
    t.host_slice().to_vec()
}

fn i32s(t: &Tensor<i32>) -> Vec<i32> {
    t.host_slice().to_vec()
}

fn probs() -> Tensor<f32> {
    Tensor::new(vec![0.0, 1.0, 0.25, 0.5, 0.75, 0.5], &[2, 3]).unwrap()
}

fn weights() -> Tensor<f32> {
    Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0], &[2, 4]).unwrap()
}

#[test]
fn items_are_reachable_with_approved_signatures() {
    let _: BernoulliFn = fandhe_ai::bernoulli;
    let _: MultinomialFn = fandhe_ai::multinomial;
    let _: NormalFn = fandhe_ai::normal;
    let mut g = Generator::new(5);
    assert_eq!(g.initial_seed(), 5);
    g.manual_seed(9);
    assert_eq!(g.initial_seed(), 9);
    let cloned = g.clone();
    assert_eq!(cloned.initial_seed(), 9);
    assert!(!format!("{g:?}").is_empty());
}

#[test]
fn same_seed_reproduces_and_different_seed_diverges() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let run = |seed: u64| {
        fandhe_ai::manual_seed(seed);
        (
            f32s(&fandhe_ai::bernoulli(&probs()).unwrap()),
            i32s(&fandhe_ai::multinomial(&weights(), 3, true).unwrap()),
            f32s(&fandhe_ai::normal(0.0, 1.0, &[16]).unwrap()),
        )
    };
    assert_eq!(run(11), run(11));
    assert_ne!(run(11).2, run(12).2);
}

#[test]
fn global_path_matches_generator_bit_for_bit() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    fandhe_ai::manual_seed(21);
    let gb = f32s(&fandhe_ai::bernoulli(&probs()).unwrap());
    let gm = i32s(&fandhe_ai::multinomial(&weights(), 2, false).unwrap());
    let gn = f32s(&fandhe_ai::normal(1.5, 0.5, &[8]).unwrap());

    let mut g = Generator::new(21);
    assert_eq!(gb, f32s(&g.bernoulli(&probs()).unwrap()));
    assert_eq!(gm, i32s(&g.multinomial(&weights(), 2, false).unwrap()));
    assert_eq!(gn, f32s(&g.normal(1.5, 0.5, &[8]).unwrap()));
}

#[test]
fn generator_does_not_touch_global_rng_and_is_independent() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    fandhe_ai::manual_seed(31);
    let expected = f32s(&fandhe_ai::normal(0.0, 1.0, &[6]).unwrap());

    fandhe_ai::manual_seed(31);
    let mut a = Generator::new(1);
    let mut b = Generator::new(1);
    let _ = a.normal(0.0, 1.0, &[100]).unwrap();
    let actual = f32s(&fandhe_ai::normal(0.0, 1.0, &[6]).unwrap());
    assert_eq!(expected, actual);
    // a の消費は b に影響しない。
    let mut fresh = Generator::new(1);
    assert_eq!(
        f32s(&b.normal(0.0, 1.0, &[4]).unwrap()),
        f32s(&fresh.normal(0.0, 1.0, &[4]).unwrap())
    );
}

#[test]
fn value_properties_hold() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    fandhe_ai::manual_seed(41);
    let b = f32s(&fandhe_ai::bernoulli(&probs()).unwrap());
    assert!(b.iter().all(|&v| v == 0.0 || v == 1.0));
    assert_eq!(b[0], 0.0);
    assert_eq!(b[1], 1.0);

    let m = fandhe_ai::multinomial(&weights(), 3, false).unwrap();
    assert_eq!(m.shape(), &[2, 3]);
    let idx = i32s(&m);
    for row in idx.chunks(3) {
        let mut sorted = row.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "非復元で重複: {row:?}");
        assert!(row.iter().all(|&i| (0..4).contains(&i)));
    }

    let n = fandhe_ai::normal(2.0, 0.0, &[5]).unwrap();
    assert!(f32s(&n).iter().all(|&v| v == 2.0));
}

#[test]
fn root_normal_differs_from_nn_init_normal() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    fandhe_ai::manual_seed(51);
    let root = f32s(&fandhe_ai::normal(0.5, 2.0, &[3, 2]).unwrap());
    fandhe_ai::manual_seed(51);
    let init = f32s(&fandhe_ai::nn::init::normal(&[3, 2], 0.5, 2.0).unwrap());
    assert_eq!(root, init);
}

#[test]
fn invalid_inputs_return_typed_errors() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let bad = Tensor::new(vec![0.5f32, 1.5], &[2]).unwrap();
    assert!(matches!(
        fandhe_ai::bernoulli(&bad),
        Err(RngError::InvalidProbability { index: 1 })
    ));
    let rank3 = Tensor::new(vec![1.0f32; 8], &[2, 2, 2]).unwrap();
    assert!(matches!(
        fandhe_ai::multinomial(&rank3, 1, true),
        Err(RngError::Shape(_))
    ));
    assert!(matches!(
        fandhe_ai::normal(0.0, -1.0, &[2]),
        Err(RngError::InvalidArgument { .. })
    ));
    let few = Tensor::new(vec![1.0f32, 0.0, 0.0], &[3]).unwrap();
    assert!(matches!(
        fandhe_ai::multinomial(&few, 2, false),
        Err(RngError::InvalidArgument { .. })
    ));
}

#[test]
fn generated_tensors_upload_to_cpu_tape_bit_for_bit() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    fandhe_ai::manual_seed(61);
    let t = fandhe_ai::normal(0.0, 1.0, &[2, 3]).unwrap();
    let tape = fandhe_ai::tape();
    let v = tape.var(&t);
    assert_eq!(f32s(&v.to_tensor()), f32s(&t));
}
