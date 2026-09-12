//! PR #1681（イシュー #1624 activation checkpointing）の codex-review・
//! Cursor Bugbot 指摘 3 件（tape.rs P0/P1/P1）に対する回帰テスト。
//!
//! - `layer1_recompute_error_propagates_instead_of_zero_fallback`:
//!   checkpoint 解放済みノードの再計算が backend エラーを起こした場合、
//!   層 1（`materialize_fallible` 経由）がゼロテンソルへ静かに吸収せず
//!   `Err` をそのまま伝播することを確認する（P0 是正の回帰）。
//! - `backward_with_held_value_ref_returns_typed_error_not_panic`:
//!   checkpoint 登録済みテープに対し `Var::value()` の `Ref` を保持した
//!   まま `Tape::backward` を呼んでも panic せず型付きエラーを返す
//!   ことを確認する（P1 是正の回帰）。
//! - `checkpoint_recompute_of_shared_ancestor_is_memoized`: `h =
//!   h.matmul(&h)` を繰り返す DAG（`Op::MatMul` の左右入力が同一
//!   `NodeId` を指す fan-in）を checkpoint 区間に収めた場合、
//!   `gemm` 呼び出し回数が指数的に増加せず線形〜多項式に収まることを
//!   確認する（P1 是正の回帰。共有祖先の重複再計算対策）。

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::Tape;
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, ChecksumReadout, Device, GemmChecksum, Tensor,
};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// [`common::NaiveOps`] 相当の `gemm` に薄い計装をかける `BackendOps`
/// ラッパー。`gemm` 呼び出し回数をカウントし、`fail_on_call`（1 始まり）
/// に一致した呼び出しだけ意図的に `Err` を返す（`None` なら常に成功）。
/// `common::NaiveOps` は非公開（`autodiff` の統合テストからは
/// `mod common;` 経由でのみ見える）ため、同じ意味論を薄く再実装する
/// （`common::mod.rs` 冒頭コメントと同じ理由: 具体バックエンドクレート
/// へ依存しない）。
struct InstrumentedOps {
    inner: Box<dyn BackendOps + Send>,
    gemm_calls: Arc<AtomicUsize>,
    fail_on_call: Option<usize>,
}

impl BackendOps for InstrumentedOps {
    fn device(&self) -> Device {
        self.inner.device()
    }

    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let n = self.gemm_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_on_call == Some(n) {
            return Err(BackendError::KernelLaunchFailed(format!(
                "InstrumentedOps: 意図的な再計算失敗（呼び出し {n} 回目）"
            )));
        }
        self.inner.gemm(a, b)
    }

    fn gemm_checksum(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        readout: ChecksumReadout,
    ) -> Result<GemmChecksum, BackendError> {
        self.inner.gemm_checksum(a, b, readout)
    }

    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.add(a, b)
    }

    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.mul(a, b)
    }

    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.relu(a)
    }

    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.exp(a)
    }

    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.tanh(a)
    }

    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.sum(a, dim)
    }

    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.max(a, dim)
    }
}

/// **P0 是正の回帰**（codex-review・Cursor Bugbot 指摘。tape.rs
/// `lazy_leaf_value` → `build_lazy_plan` 経由の fail-open）。
///
/// `m = a.matmul(&b)` を checkpoint 区間に収め、区間の外で `r = m.relu()`
/// （lazy elementwise。`m` を未実体化のまま参照する）を作り、`r` を
/// そのまま `Tape::backward` の loss に渡す。`backward_impl` は開始直後
/// に `materialize_fallible(loss)` で shape を読むため、ここで `r` の
/// 実体化（`build_lazy_plan` の葉参照）が checkpoint 解放済みの `m` の
/// 再計算（2 回目の `gemm` 呼び出し）を要求する。この再計算を意図的に
/// 失敗させ、`Tape::backward` が `Err` を返すこと（ゼロテンソルへ
/// 静かに変換されて成功しないこと）を確認する。
#[test]
fn layer1_recompute_error_propagates_instead_of_zero_fallback() {
    let gemm_calls = Arc::new(AtomicUsize::new(0));
    let ops = InstrumentedOps {
        inner: common::naive_ops(),
        gemm_calls: Arc::clone(&gemm_calls),
        // 1 回目（checkpoint 内の forward）は成功させ、2 回目
        // （backward 側の再計算）だけを失敗させる。
        fail_on_call: Some(2),
    };
    let tape = Tape::new_with_ops(Box::new(ops));

    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));

    let m = tape
        .checkpoint(|| a.matmul(&b))
        .expect("checkpoint 内の forward（1 回目の gemm）は成功する");
    let r = m.relu();

    let result = tape.backward(&r);
    assert!(
        result.is_err(),
        "checkpoint 解放済みノードの再計算がバックエンドエラーを起こした場合、\
         `Tape::backward` は Err を返すべき（ゼロテンソルへ静かに変換して \
         成功してはならない）: {result:?}"
    );
    assert_eq!(
        gemm_calls.load(Ordering::SeqCst),
        2,
        "1 回目（forward）成功・2 回目（backward 再計算）失敗のシナリオである契約を\
         テスト自身が満たしているかの自己検証"
    );
}

/// **P1 是正の回帰**（codex-review 指摘。tape.rs
/// `release_checkpoints_ending_at` の `borrow_mut()` panic）。
///
/// checkpoint 登録済みのテープに対し、`Var::value()` が返す `Ref` を
/// 保持したまま `Tape::backward` を呼ぶと、backward の逆走査中に
/// `release_checkpoints_ending_at` が要求する `self.nodes.borrow_mut()`
/// が実行時 panic していた（本番経路 panic 禁止方針違反）。
/// `try_borrow_mut` 化により、panic ではなく型付きエラーを返すことを
/// 確認する。
#[test]
fn backward_with_held_value_ref_returns_typed_error_not_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));

    let m = tape
        .checkpoint(|| a.matmul(&b))
        .expect("checkpoint 内の forward は成功する");
    let loss = m
        .sum(None)
        .expect("sum は成功する（checkpoint 解放とは無関係の非破壊読み出し）");

    // `loss` 自身は checkpoint 区間の外（`sum` は checkpoint 呼び出し後）
    // なので値は既に実体化済みだが、`value()` が返す `Ref` は
    // `self.tape.nodes` への不変借用を保持したまま呼び出し元スコープに
    // 生存し続ける——この状態で `backward` を呼ぶのが本テストの要点。
    let held_ref = loss.value();

    // panic せずに戻ってくること自体が本テストの主張。`Err` であること
    // まで確認し、`held_ref` は明示的に drop してから結果を検証する
    // （借用が解放された後でも `Err` のまま変わらないことを示す）。
    let result = tape.backward(&loss);
    drop(held_ref);

    assert!(
        result.is_err(),
        "checkpoint 登録済みテープで外部 `Ref` を保持したまま backward を呼んだ場合、\
         panic ではなく型付きエラーを返すべき: {result:?}"
    );
}

/// **P1 是正の回帰**（codex-review 指摘。共有祖先の O(2^n) 再計算）。
///
/// `h_{i} = h_{i-1}.matmul(&h_{i-1})`（左右入力が同一 `NodeId`）を
/// `N` 回繰り返す chain を checkpoint 区間に収める。checkpoint は
/// `output`（＝ `h_N` 自身）を除く区間内の全ノードを解放するため、
/// backward が `h_N = MatMul(h_{N-1}, h_{N-1})` の VJP を計算する際、
/// `h_{N-1}` の再計算（`recompute_value` の単一呼び出し木）が
/// `Op::MatMul(a, b)` で `a == b` となる分岐を経て再帰する。
/// メモ化なしでは、この 1 回の `recompute_value` 呼び出しだけで
/// 深さ `N-1` の 2 分木を辿ることになり `gemm` 呼び出しが指数的に
/// 増加する。メモ化ありなら、backward 全体（複数回の再計算呼び出しの
/// 合計）でも高々多項式（本テストでは緩めに `N * N` 未満）に収まる
/// ことを確認する。
#[test]
fn checkpoint_recompute_of_shared_ancestor_is_memoized() {
    const N: usize = 16;

    let gemm_calls = Arc::new(AtomicUsize::new(0));
    let ops = InstrumentedOps {
        inner: common::naive_ops(),
        gemm_calls: Arc::clone(&gemm_calls),
        fail_on_call: None,
    };
    let tape = Tape::new_with_ops(Box::new(ops));

    let x0 = tape.var(&t(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]));
    let h_n = tape
        .checkpoint(|| {
            let mut cur = x0;
            for _ in 0..N {
                cur = cur.matmul(&cur)?;
            }
            Ok(cur)
        })
        .expect("checkpoint 内の forward（N 回の gemm）は成功する");

    let forward_calls = gemm_calls.load(Ordering::SeqCst);
    assert_eq!(
        forward_calls, N,
        "forward 側は N 回の逐次 squaring で N 回の gemm 呼び出しになるはず"
    );

    let loss = h_n
        .sum(None)
        .expect("checkpoint の output（h_n）は解放されないため sum は再計算なしで成功する");
    let grads = tape
        .backward(&loss)
        .expect("メモ化により再計算が破綻せず backward は成功するはず");
    assert!(
        grads
            .get(&x0)
            .expect("x0 は matmul 経由で loss へ到達する")
            .is_some(),
        "x0 への勾配が伝播しているはず"
    );

    let total_calls = gemm_calls.load(Ordering::SeqCst);
    let backward_calls = total_calls - forward_calls;
    // メモ化なし（旧実装）ではこの 1 回の checkpoint 区間の再計算だけで
    // `Op::MatMul(a, a)` の分岐が段数ぶん倍々に膨らみ、N=16 なら
    // 2^15 = 32768 回を優に超える gemm 呼び出しになる（実行自体が
    // 事実上停止する規模）。メモ化ありなら backward 全体でも高々
    // `N * N` 未満（本実装は各 backward ループ反復ごとに独立した
    // `recompute_value` 呼び出しを行うため厳密な線形〈O(N)〉ではなく
    // 多項式〈O(N^2)〉になりうるが、指数的増加は起きない）に収まる。
    assert!(
        backward_calls < N * N,
        "backward 側の再計算 gemm 呼び出し回数が指数的に増加していないことを確認\
         （実測 {backward_calls} 回、上限 {}）",
        N * N
    );
}
