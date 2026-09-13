//! PR #1681（イシュー #1624 activation checkpointing）の codex-review・
//! Cursor Bugbot 指摘に対する回帰テスト。
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
//! - `layer2_poisoned_recompute_is_detected_by_layer1_backward`:
//!   層 2（`to_tensor()`）が checkpoint 解放済みノードの再計算失敗を
//!   ゼロ値へ吸収して poison フラグを立てた場合、その後の層 1
//!   （`Tape::backward`）がキャッシュ済みゼロ値を信頼せず `Err` を
//!   返すことを確認する（P0 是正の回帰。祖先自身の poison 検出）。
//! - `layer2_poison_propagates_to_lazy_descendant`: `out = m.relu()`
//!   のような lazy elementwise 子孫を、祖先 `m`（checkpoint 解放済み）
//!   の再計算に一度も成功していない状態で層 2（`to_tensor()`）から
//!   直接読み出すと、旧実装は `materialize_non_fallible` の最終手段
//!   `eval_fallback`（poison 検査なし）を経由してゼロ値を「正常な値」
//!   として `out` 自身の `OnceCell` へキャッシュしてしまい、`out` の
//!   `recompute_failed` が立たないまま以後の層 1 が汚染を見逃していた。
//!   祖先の再計算失敗が子孫（lazy 出力）の poison にも伝播し、以後の
//!   `Tape::backward` が確実に `Err` を返すことを確認する（P0 是正の
//!   回帰。codex-review・Cursor Bugbot 指摘）。
//! - `deep_checkpoint_chain_recompute_does_not_overflow_stack`: 数万段
//!   の `sigmoid` 連鎖を checkpoint 区間に収め、解放後の単一の
//!   再計算呼び出し（`recompute_value`）がこの深さを反復的な作業
//!   スタックで評価し切り、Rust の呼び出しスタックオーバーフローで
//!   プロセスが abort しないことを確認する（P1 是正の回帰）。
//! - `eager_sigmoid_propagates_poison_from_checkpoint_freed_input`:
//!   `push_eager`（`Var::sigmoid` 等、`.value()` で入力を infallible に
//!   読んでから即座に計算する経路）が、checkpoint 解放済み入力の
//!   再計算失敗（poison）を出力ノードへ伝播し損ねていた P0 是正の
//!   回帰（codex-review 指摘。PR #1681 スレッド
//!   `crates/autodiff/src/tape.rs:1267`）。`m.sigmoid()` の出力が
//!   `m` の poison を継承し、以後の `sum(None)`（fallible）と
//!   `Tape::backward` がいずれも `Err` を返すことを確認する。

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
/// に一致した呼び出しだけ、または `fail_from_call`（1 始まり）**以降の
/// 全呼び出し**を意図的に `Err` を返す（いずれも `None` なら常に成功）。
/// `fail_from_call` は、`materialize_non_fallible`（層 2）の
/// `build_lazy_plan` → `fallback_per_op` → `eval_fallback` という
/// 3 段の再計算試行が**すべて**失敗することを要求するテスト
/// （`layer2_poison_propagates_to_lazy_descendant`）のために追加した
/// （`fail_on_call` の単発失敗では 2 段目・3 段目が成功してしまい
/// 目的のシナリオを再現できない）。`common::NaiveOps` は非公開
/// （`autodiff` の統合テストからは `mod common;` 経由でのみ見える）
/// ため、同じ意味論を薄く再実装する（`common::mod.rs` 冒頭コメントと
/// 同じ理由: 具体バックエンドクレートへ依存しない）。
struct InstrumentedOps {
    inner: Box<dyn BackendOps + Send>,
    gemm_calls: Arc<AtomicUsize>,
    fail_on_call: Option<usize>,
    fail_from_call: Option<usize>,
    // `fail_from_call` の失敗区間の終端（inclusive）。`None` は
    // 無期限（`fail_from_call` 以降ずっと失敗）を意味する。層 2 の
    // 3 段フォールバック試行だけを失敗させ、その後（`Tape::backward`
    // 自身の反復による独立した再計算試行）は成功させたいテスト
    // （`layer2_poison_propagates_to_lazy_descendant`）のために追加。
    fail_until_call: Option<usize>,
}

impl BackendOps for InstrumentedOps {
    fn device(&self) -> Device {
        self.inner.device()
    }

    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let n = self.gemm_calls.fetch_add(1, Ordering::SeqCst) + 1;
        let in_fail_range = self
            .fail_from_call
            .is_some_and(|from| n >= from && self.fail_until_call.is_none_or(|until| n <= until));
        let should_fail = self.fail_on_call == Some(n) || in_fail_range;
        if should_fail {
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
/// **Cursor Bugbot 指摘の是正（イシュー #1624 PR #1681 レビュー）**:
/// 旧実装は `m = a.matmul(&b)` を checkpoint の**戻り値そのもの**
/// （`output`）にしていたため、`release_checkpoint_region` の契約
/// （`[lo, output)`。`output` 自身は解放しない）により `m` が一度も
/// 解放されず、目的の再計算（2 回目の `gemm`）が発生しないまま
/// テストの意図が検証できていなかった。本テストでは `m` を checkpoint
/// 区間の**内部中間ノード**にし、区間の出力を `m.relu()`（checkpoint
/// 内で計算。`m` を入力に取る）にすることで `m` を確実に解放対象へする。
///
/// `out = relu(m)`（lazy・未実体化）をそのまま `Tape::backward` の
/// loss に渡す。`backward_impl` は開始直後に `materialize_fallible
/// (loss)` で `out` を実体化するため、`build_lazy_plan` の葉参照が
/// checkpoint 解放済みの `m` の再計算（2 回目の `gemm` 呼び出し）を
/// 要求する。この再計算を意図的に失敗させ、`Tape::backward` が `Err`
/// を返すこと（ゼロテンソルへ静かに変換されて成功しないこと）を
/// 確認する。
#[test]
fn layer1_recompute_error_propagates_instead_of_zero_fallback() {
    let gemm_calls = Arc::new(AtomicUsize::new(0));
    let ops = InstrumentedOps {
        inner: common::naive_ops(),
        gemm_calls: Arc::clone(&gemm_calls),
        // 1 回目（checkpoint 内の forward）は成功させ、2 回目
        // （backward 側の再計算）だけを失敗させる。
        fail_on_call: Some(2),
        fail_from_call: None,
        fail_until_call: None,
    };
    let tape = Tape::new_with_ops(Box::new(ops));

    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));

    let out = tape
        .checkpoint(|| {
            // `m` は checkpoint 区間の**内部**ノード（`output` ではない）
            // であるため `register_checkpoint` により確実に解放される。
            // `out = m.relu()` は lazy elementwise（`Op::is_lazy_
            // elementwise`）のため、ここではまだ実体化されない
            // （`m.relu()` 呼び出し自体が backend を叩かない）。
            let m = a.matmul(&b)?;
            Ok(m.relu())
        })
        .expect("checkpoint 内の forward（1 回目の gemm）は成功する");

    // `out` を直接 loss として `backward` に渡す（`sum` 等で先に
    // 実体化させない）。`backward_impl` 冒頭の `materialize_fallible
    // (loss)` が `out`（lazy）の実体化を要求し、`build_lazy_plan` の
    // 葉参照が checkpoint 解放済みの `m` の再計算（2 回目の `gemm`）を
    // 要求する。
    let result = tape.backward(&out);
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
         テスト自身が満たしているかの自己検証（`m` が実際に解放され、\
         backward が再計算を要求したことの証跡でもある）"
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
        fail_from_call: None,
        fail_until_call: None,
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

/// **P0 是正の回帰（layer 2 poison 契約。codex-review 指摘。イシュー
/// #1624 PR #1681 レビュー）**。
///
/// 層 2（`materialize_non_fallible`。`Var::value`／`Var::to_tensor` が
/// 使う非 fallible 境界）は checkpoint 解放済みノードの再計算が真の
/// バックエンド実行失敗を起こしても契約上 `Tensor<f32>`（ゼロ埋め
/// フォールバック）を返さざるを得ない。本テストは、この失敗が
/// `TapeNode::recompute_failed`（poison フラグ）として記録され、
/// 以後 `Tape::backward`（層 1・`materialize_fallible` 経由）が
/// このノードのキャッシュ済みゼロ値を「正しい実体化結果」として
/// 信頼せず `Err` を返すことを確認する——poison 契約がなければ
/// `to_tensor()` が握り潰したゼロテンソルをそのまま使って backward が
/// 成功し、ゼロ勾配が静かに返ってしまう。
#[test]
fn layer2_poisoned_recompute_is_detected_by_layer1_backward() {
    let gemm_calls = Arc::new(AtomicUsize::new(0));
    let ops = InstrumentedOps {
        inner: common::naive_ops(),
        gemm_calls: Arc::clone(&gemm_calls),
        // 1 回目（forward）は成功させ、2 回目（`m.to_tensor()` が
        // 誘発する層 2 経由の再計算）だけを失敗させる。
        fail_on_call: Some(2),
        fail_from_call: None,
        fail_until_call: None,
    };
    let tape = Tape::new_with_ops(Box::new(ops));

    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));

    // `checkpoint_from`（`Var::checkpoint_from`）を使い、`m`（matmul
    // 結果）を通常の局所変数として保持したまま `out = m.relu()` を
    // 区間の出力にする。`m` は区間内部ノードのため解放される。
    let m = a.matmul(&b).expect("forward の 1 回目の gemm は成功する");
    let out = m.relu();
    let checkpointed_out = out
        .checkpoint_from(&[&a, &b])
        .expect("checkpoint_from 自体は forward を再実行しないため成功する");

    // 層 2 を直接呼び、解放済み `m` の再計算失敗を誘発する。契約どおり
    // panic せずゼロテンソルが返る（shape のみ保たれる）。
    let zero_fallback = m.to_tensor();
    assert_eq!(
        zero_fallback.shape(),
        &[2usize, 2],
        "層 2 は失敗時も shape を保ったゼロテンソルを返す契約"
    );

    // 汚染済みのキャッシュ値を層 1（`Tape::backward`）が「正しい実体化
    // 結果」として信頼し、ゼロ勾配のまま静かに成功してはならない。
    let result = tape.backward(&checkpointed_out);
    assert!(
        result.is_err(),
        "poison フラグが立った checkpoint 解放済みノードを層 1 が\
         検出できず、ゼロ勾配のまま backward が成功してしまっている: {result:?}"
    );
    assert_eq!(
        gemm_calls.load(Ordering::SeqCst),
        2,
        "1 回目（forward）成功・2 回目（`m.to_tensor()` 経由の再計算）\
         失敗のシナリオである契約をテスト自身が満たしているかの自己検証\
         （backward 自体は poison 検出により追加の gemm 呼び出しを\
         要求しない）"
    );
}

/// **P0 是正の回帰（lazy descendant poison 伝播。codex-review・Cursor
/// Bugbot 指摘。イシュー #1624 PR #1681 レビュー）**。
///
/// 前テスト（`layer2_poisoned_recompute_is_detected_by_layer1_backward`）
/// は `m.to_tensor()` を**先に**呼んで `m` 自身を直接 poison するため、
/// 以後の `build_lazy_plan` は「既に実体化済み（かつ poison 済み）の
/// 葉」として `m` を検出でき、`lazy_leaf_value_fallible` の
/// `poisoned_err` チェックだけで層 1 が正しく `Err` を返す——この経路は
/// 旧実装でも正しく動いており、`materialize_non_fallible` の最終手段
/// `eval_fallback`（poison 検査を持たない旧 `lazy_leaf_value` 経由）は
/// 一度も通らない。
///
/// 本テストは `m` を**一度も単独で読み出さず**、`out = m.relu()` を
/// 層 2（`to_tensor()`）から直接実体化させる。`m` は未実体化のまま
/// なので、`materialize_non_fallible(out)` は `build_lazy_plan` →
/// `fallback_per_op` → `eval_fallback` の 3 段の再計算試行をすべて
/// 経由する（`fail_from_call`〜`fail_until_call` の区間 `[2, 4]` で
/// この 3 段だけを確実に失敗させる。実測値: `to_tensor()` 完了時点で
/// `gemm_calls == 4`〈1 回目の forward + 3 段の再計算試行〉）。
///
/// **区間指定にした理由**（単なる `fail_from_call`〈無期限〉ではなく
/// `[2, 4]` に限定する理由）: `Tape::backward` 自身の逆走査ループは
/// `id = out` を処理した**後**、`id = m` を処理する際にも
/// （`grad::vjp` が実際に値を使うかどうかに関わらず）
/// `materialize_fallible(m)` を無条件に呼び、`m` の再計算を独立に
/// もう一度試みる（`backward.rs` 該当行 doc 参照）。この 5 回目の
/// 呼び出しまで失敗させ続けると、`out` 自身の poison 検出が効いて
/// いなくても「`m` の再計算自体が失敗して `Err` になる」という
/// **別の理由**で本テストが `is_err()` を満たしてしまい、目的の
/// 伝播漏れを検出できなくなる（是正前のコードでも偶然 pass する）。
/// 5 回目以降を成功させることで、是正前のコードでは「`out` の
/// poison が検出されないまま `materialize_fallible(out)` がキャッシュ
/// 済みゼロ値を正常値として通し、`m` 自身は 5 回目の再計算で正しい
/// 値を得て `Tape::backward` 全体が **`Ok`** で完走してしまう
/// （ゼロ埋めされた `out_value` に由来する誤った——ゼロに潰れた——
/// 勾配を silently 返す）」という本来の不具合を明確に区別できる。
#[test]
fn layer2_poison_propagates_to_lazy_descendant() {
    let gemm_calls = Arc::new(AtomicUsize::new(0));
    let ops = InstrumentedOps {
        inner: common::naive_ops(),
        gemm_calls: Arc::clone(&gemm_calls),
        fail_on_call: None,
        // 1 回目（forward）は成功させ、2〜4 回目（`out.to_tensor()` が
        // 誘発する `m` の再計算試行。`build_lazy_plan`／
        // `fallback_per_op`／`eval_fallback` の 3 回）だけを失敗させる。
        // 5 回目以降（`Tape::backward` 自身の独立した再計算試行）は
        // 成功させる（doc 参照）。
        fail_from_call: Some(2),
        fail_until_call: Some(4),
    };
    let tape = Tape::new_with_ops(Box::new(ops));

    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));

    // `m` は checkpoint 区間の内部中間ノード（`checkpoint_from` の
    // `output` ではない）であるため確実に解放される。`out = m.relu()`
    // は lazy elementwise のため `m.relu()` 呼び出し自体は backend を
    // 叩かず、`out` は未実体化のまま区間の出力になる。
    let m = a.matmul(&b).expect("forward の 1 回目の gemm は成功する");
    let out = m.relu();
    let checkpointed_out = out
        .checkpoint_from(&[&a, &b])
        .expect("checkpoint_from 自体は forward を再実行しないため成功する");

    // `m` を一度も単独で読み出さず、`out` 自身を層 2 から直接実体化
    // する。契約どおり panic せずゼロテンソル（shape のみ保たれる）が
    // 返る。
    let zero_fallback = checkpointed_out.to_tensor();
    assert_eq!(
        zero_fallback.shape(),
        &[2usize, 2],
        "層 2 は失敗時も shape を保ったゼロテンソルを返す契約"
    );
    assert_eq!(
        gemm_calls.load(Ordering::SeqCst),
        4,
        "1 回目（forward）成功・2〜4 回目（層 2 の 3 段フォールバック \
         試行）失敗のシナリオである契約をテスト自身が満たしているかの \
         自己検証"
    );

    // `out` 自身は checkpoint の解放対象ではない（`Op::Relu` は
    // elementwise で非適格）が、祖先 `m` の再計算失敗が `out` の
    // poison フラグへ伝播しているはずであり、以後の層 1
    // （`Tape::backward`）が `out` を読む時点（`m` 自身の 5 回目の
    // 再計算を試みるより前）で検出して `Err` を返すべき。
    let result = tape.backward(&checkpointed_out);
    assert!(
        result.is_err(),
        "祖先（checkpoint 解放済みノード）の再計算失敗が lazy な子孫の \
         poison へ伝播しておらず、層 1 がゼロ勾配のまま backward を \
         成功させてしまっている: {result:?}"
    );
    // `out` の poison 検出（`materialize_fallible` 冒頭のキャッシュ済み
    // 値チェック）で即座に `Err` になるはずであり、`m` 自身の 5 回目の
    // 再計算試行（成功するよう仕込んである）には到達しない契約——
    // 到達してしまっている場合、`out` の poison 検出ではなく別の理由
    // （`m` 自身の再計算失敗等）で偶然 `Err` になっているだけであり、
    // 本テストが検証したい伝播漏れを実際には検出できていないことを
    // 意味する。
    assert_eq!(
        gemm_calls.load(Ordering::SeqCst),
        4,
        "`Tape::backward` が `out` の poison 検出より先に `m` 自身の \
         再計算（5 回目の gemm 呼び出し）へ到達してしまっている（伝播が \
         効いていれば `out` の時点で即座に Err になり `m` の再計算には \
         到達しないはず）"
    );
}

/// **P1 是正の回帰（`recompute_value` の反復化。codex-review 指摘。
/// イシュー #1624 PR #1681 レビュー）**。
///
/// `sigmoid` を数万段連鎖させた区間を checkpoint に収めると、区間内の
/// 全中間ノード（`output` 自身を除く）が解放される。旧実装の
/// `recompute_value` は Rust の呼び出しスタックを直接使う再帰関数
/// だったため、この深さの祖先鎖を単一の再計算呼び出しで辿ると
/// スタックオーバーフローでプロセスが abort していた。本テストは
/// カスタムのスタックサイズ指定（`std::thread::Builder::stack_size`）
/// を使わず、テストランナーの既定スタックのまま深い連鎖を単一の
/// `to_tensor()` 呼び出し（層 2 → `recompute_infallible` →
/// `recompute_value`）で解決できることを確認する。
#[test]
fn deep_checkpoint_chain_recompute_does_not_overflow_stack() {
    // 旧再帰実装であれば既定のスレッドスタックサイズを優に超える深さ
    // （1 段あたり複数のスタックフレーム・ローカル変数を消費する）。
    const CHAIN_LEN: usize = 50_000;

    let tape = Tape::new_with_ops(common::naive_ops());
    let x0 = tape.var(&t(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]));

    // ループ内で「区間出力の 1 つ手前」（深さ `CHAIN_LEN - 1` の祖先鎖
    // を持つ、確実に解放される中間ノード）を保持しておく。
    let mut deepest_freed: Option<fandhe_ai_autodiff::Var<'_>> = None;
    let output = tape
        .checkpoint(|| {
            let mut cur = x0;
            for i in 0..CHAIN_LEN {
                cur = cur.sigmoid();
                if i == CHAIN_LEN - 2 {
                    deepest_freed = Some(cur);
                }
            }
            Ok(cur)
        })
        .expect("checkpoint 内の forward（CHAIN_LEN 回の sigmoid）は成功する");
    // `output`（区間の戻り値。`checkpoint` の契約により解放されない）
    // 自体は本テストの検証対象ではないが、変数として保持しないと
    // 「未使用」警告になるため明示的に握りつぶす。
    let _ = output;

    let deepest = deepest_freed.expect("CHAIN_LEN >= 2 のためループ内で必ず設定される");

    // 単一の `to_tensor()` 呼び出しが、深さ `CHAIN_LEN - 1` の祖先鎖を
    // 反復的な作業スタック（ヒープ確保）で評価し切り、abort せずに
    // 戻ってくること自体が本テストの主張。
    let value = deepest.to_tensor();
    assert_eq!(value.shape(), &[2usize, 2]);
    let data = value
        .as_slice()
        .expect("test fixture: contiguous な出力のはず");
    for v in data {
        assert!(
            v.is_finite() && *v > 0.0 && *v < 1.0,
            "sigmoid の出力域 (0, 1) に収まっているはず（反復評価が forward と \
             同じ計算を再現していることの弱い間接検証）: {v}"
        );
    }
}

/// **P0 是正の回帰（eager 演算の poison 伝播漏れ。codex-review 指摘。
/// PR #1681 スレッド `crates/autodiff/src/tape.rs:1267`）**。
///
/// `Var::sigmoid`（`push_eager` 経由の eager 演算。`materialize_
/// fallible` を経由せず `.value()`〈層 2・非 fallible〉で入力を読んで
/// 即座に計算する）が、checkpoint 解放済み入力の再計算失敗（poison）を
/// 出力ノードへ伝播していなかった不具合の回帰。
///
/// `m = a.matmul(&b)` を checkpoint 区間の**内部**ノードとして解放した
/// 後（区間の出力は `m.relu()`。`layer2_poisoned_recompute_is_detected_
/// by_layer1_backward` と同じ `checkpoint_from` パターン）、`m.sigmoid()`
/// を呼ぶと `Var::sigmoid` 内部の `self.value()` が `m` の再計算
/// （意図的に失敗させた 2 回目の `gemm`）を誘発し、`m` の
/// `TapeNode::recompute_failed`（poison）を立てたうえで契約どおり
/// ゼロテンソルを返す（`eval::sigmoid(0.0) == 0.5`）。
///
/// 是正前は `push_eager` がこの poison を無視して `sigmoid` の出力を
/// 常に `recompute_failed: false` で登録していたため、`sig.sum(None)`
/// （fallible）が「0.5 の一様値」を正常な計算結果として `Ok` を返し、
/// `Tape::backward` も poison を検出できずに完走してしまっていた
/// （fail-closed 契約違反。`.claude/rules/security.md` A08）。是正後は
/// `push_eager` が `Op::for_each_input` で `m` の poison を検出し
/// `sig` へ継承するため、`sum(None)`・`backward` のいずれも `Err` を
/// 返す。
#[test]
fn eager_sigmoid_propagates_poison_from_checkpoint_freed_input() {
    let gemm_calls = Arc::new(AtomicUsize::new(0));
    let ops = InstrumentedOps {
        inner: common::naive_ops(),
        gemm_calls: Arc::clone(&gemm_calls),
        // 1 回目（forward）は成功させ、2 回目（`m.sigmoid()` が
        // `self.value()` 経由で誘発する `m` の再計算）だけを失敗させる。
        fail_on_call: Some(2),
        fail_from_call: None,
        fail_until_call: None,
    };
    let tape = Tape::new_with_ops(Box::new(ops));

    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));

    // `m` は checkpoint 区間の内部中間ノード（`checkpoint_from` の
    // `output` ではない）であるため確実に解放される。
    let m = a.matmul(&b).expect("forward の 1 回目の gemm は成功する");
    let out = m.relu();
    let checkpointed_out = out
        .checkpoint_from(&[&a, &b])
        .expect("checkpoint_from 自体は forward を再実行しないため成功する");
    // `checkpointed_out`（区間の出力）は本テストの検証対象ではないが、
    // `m` を解放するために `checkpoint_from` を呼ぶ必要があるため
    // 変数として保持する（未使用警告の抑止）。
    let _ = &checkpointed_out;

    // `m.sigmoid()` は eager 演算（`push_eager` 経由）。内部の
    // `self.value()` が `m` の再計算（2 回目の gemm・意図的に失敗）を
    // 誘発し、`m` の poison フラグを立てたうえでゼロテンソル
    // （`eval::sigmoid(0.0) == 0.5`）を返す。この時点で `Var::sigmoid`
    // 自身は非 fallible な契約のため panic せず `Var` を返す。
    let sig = m.sigmoid();
    assert_eq!(
        gemm_calls.load(Ordering::SeqCst),
        2,
        "1 回目（forward）成功・2 回目（`m.sigmoid()` 経由の再計算）\
         失敗のシナリオである契約をテスト自身が満たしているかの自己検証"
    );

    // 是正前は poison が `sig` へ伝播せず、`sum(None)` が「0.5 の一様値」
    // を正常な計算結果として `Ok` を返してしまっていた。
    let sum_result = sig.sum(None);
    assert!(
        sum_result.is_err(),
        "checkpoint 解放済み入力 `m` の再計算失敗（poison）が eager 演算 \
         `sigmoid` の出力へ伝播しておらず、後続の fallible 演算 `sum` が \
         汚染された値をそのまま `Ok` として返してしまっている: {sum_result:?}"
    );

    // `Tape::backward` も同じ poison 検出（`materialize_fallible` 冒頭の
    // `poisoned_err`）により `Err` を返すべき（ゼロ勾配のまま静かに
    // 成功してはならない）。
    let backward_result = tape.backward(&sig);
    assert!(
        backward_result.is_err(),
        "poison 済みの eager 演算出力 `sig` を `Tape::backward` が検出できず、\
         ゼロ勾配のまま backward が成功してしまっている: {backward_result:?}"
    );
}

/// [`elementwise_leaves_poisoned`] を「到達可能なノードのみ走査」する
/// よう書き換えた codex-review 是正（イシュー #1624 PR #1681 レビュー・
/// `crates/autodiff/src/tape.rs:2700` 付近）の回帰テスト。
///
/// checkpoint を一度も使わない `sigmoid` の N 段連鎖（`Var::sigmoid` は
/// `push_eager` 経由の eager 演算で、各段の出力は即座に実体化される）
/// は、旧実装（`id` から `0` まで全 `NodeId` を線形走査）だと 1 段ごとに
/// O(N) の検査が乗り forward 全体が O(N²) へ悪化する。是正後は各段の
/// 入力（直前の sigmoid 出力）が既に実体化済みであるため
/// `elementwise_leaves_poisoned` が即返しの高速経路（自身の
/// `recompute_failed` フラグ検査のみ）を通り、1 段あたり O(1) に戻る。
/// 本テストは実行時間を計測せず、数万段の連鎖でも forward が
/// （タイムアウトせずに）完走することのみを確認する。
#[test]
fn eager_sigmoid_chain_without_checkpoint_completes_forward() {
    const CHAIN_LEN: usize = 50_000;

    let tape = Tape::new_with_ops(common::naive_ops());
    let mut cur = tape.var(&t(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]));
    for _ in 0..CHAIN_LEN {
        cur = cur.sigmoid();
    }

    // checkpoint を経由しないため `to_tensor()`（層 2）はすべて既に
    // eager に実体化済みの値をそのまま返すはずで、`Err` にはならない。
    let out = cur.to_tensor();
    assert_eq!(
        out.shape(),
        &[2, 2],
        "eager sigmoid 連鎖の出力 shape が入力から変化してはならない"
    );
}
