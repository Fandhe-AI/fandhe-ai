//! `Var::device`／`Var::to`／`Var::to_tape`（イシュー #1614）の契約
//! テスト。`docs/facade-device-transfer-enumeration-design.md` §5「テスト」
//! に対応する (A) 節。
//!
//! 数値微分は使わない——`to`／`to_tape` は算術を含まないデバイス
//! 束縛・値転送の検査であり、`common::naive_ops()`（`f32::mul_add` の
//! FMA 契約で決定的）を使った解析解・bit 一致で検証する
//! （`no_grad_detach.rs` と同じ方針）。

mod common;

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, ChecksumReadout, Device, GemmChecksum, Tensor,
};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

// --- A1/A2: `Var::to` ---

/// A1: 同一デバイスへの `to` は恒等（tape へノードを追加しない）。
#[test]
fn to_same_device_is_identity_and_preserves_gradient() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let len_before = tape.leaf_count();

    let x_same = x
        .to(Device::Cpu)
        .expect("同一デバイスへの to は常に成功する");
    // 恒等（ノード追加なし）: 葉の個数は不変。
    assert_eq!(tape.leaf_count(), len_before);

    // `loss = sum(x.to(Cpu) * x)` の勾配が、`to` を省いた場合
    // （`sum(x * x)` の勾配 `2x`）と bit 一致することを確認する。
    let product = x_same.mul(&x).expect("同 shape の要素積は失敗しない");
    let loss = product.sum(None).expect("全軸縮約は失敗しない");
    let grads = tape
        .backward(&loss)
        .expect("追跡対象の祖先を持つため成功する");
    let dx = grads
        .get(&x)
        .expect("x は requires_grad=true の葉")
        .expect("x は loss に到達する");
    let expected: Vec<f32> = x
        .to_tensor()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| 2.0 * v)
        .collect();
    assert_eq!(dx.as_slice().unwrap(), expected.as_slice());
}

/// A2: 別デバイスへの `to`（同一 tape 上では表現できない）は
/// `AutodiffError::DeviceMismatch` を返し、tape の長さは変わらない。
#[test]
fn to_different_device_returns_device_mismatch_without_mutating_tape() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let len_before = tape.leaf_count();

    let err = x
        .to(Device::Cuda(0))
        .expect_err("naive_ops は Device::Cpu のため Cuda(0) は不一致");
    match err {
        AutodiffError::DeviceMismatch { requested, actual } => {
            assert_eq!(requested, Device::Cuda(0));
            assert_eq!(actual, Device::Cpu);
        }
        other => panic!("DeviceMismatch を期待したが {other:?} だった"),
    }
    assert_eq!(tape.leaf_count(), len_before, "失敗時は tape を変更しない");
}

// --- A3〜A6: `Var::to_tape` ---

/// A3: 同一 tape への `to_tape` は恒等（ノード追加なし）。
#[test]
fn to_tape_same_tape_is_identity() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let len_before = tape.leaf_count();

    let x2 = x.to_tape(&tape).expect("同一 tape への転送は常に成功する");
    assert_eq!(tape.leaf_count(), len_before);
    assert_eq!(
        x2.to_tensor().as_slice().unwrap(),
        x.to_tensor().as_slice().unwrap()
    );
}

/// A4: 異なる 2 つの naive tape 間の `to_tape` は、値が bit 一致で
/// 転送先の新しい葉として登録され、転送先の `backward` で勾配を
/// 受け取る。転送元 tape の `backward` は転送の影響を受けない
/// （勾配 bit 一致）。`requires_grad` の引き継ぎ（`var`→true）も確認する。
#[test]
fn to_tape_across_tapes_transfers_value_and_isolates_gradients() {
    let source = Tape::new_with_ops(common::naive_ops());
    let target = Tape::new_with_ops(common::naive_ops());

    let x = source.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let len_target_before = target.leaf_count();

    let x_on_target = x
        .to_tape(&target)
        .expect("別 tape への転送は実体化済みノードなら失敗しない");
    assert_eq!(
        target.leaf_count(),
        len_target_before + 1,
        "転送先の葉が +1 される"
    );
    assert_eq!(
        x_on_target.to_tensor().as_slice().unwrap(),
        x.to_tensor().as_slice().unwrap(),
        "転送された値は bit 完全一致（算術を含まないため）"
    );

    // 転送先で計算グラフを組み、backward すると `x_on_target` は
    // requires_grad=true の葉として勾配を受け取る（source.var と同じ
    // 引き継ぎ）。
    let product = x_on_target
        .mul(&x_on_target)
        .expect("同 shape の要素積は失敗しない");
    let loss = product.sum(None).expect("全軸縮約は失敗しない");
    let grads = target
        .backward(&loss)
        .expect("転送先での backward は成功する");
    let d_target = grads
        .get(&x_on_target)
        .expect("x_on_target は requires_grad=true の葉")
        .expect("loss に到達する");
    let expected: Vec<f32> = x
        .to_tensor()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| 2.0 * v)
        .collect();
    assert_eq!(d_target.as_slice().unwrap(), expected.as_slice());

    // 転送元 `source` 側の tape は転送の影響を一切受けない
    // （sum(x*x) を source 上で独立に組んでも同じ勾配になる）。
    let product_src = x.mul(&x).expect("同 shape の要素積は失敗しない");
    let loss_src = product_src.sum(None).expect("全軸縮約は失敗しない");
    let grads_src = source
        .backward(&loss_src)
        .expect("転送元での backward も独立して成功する");
    let d_src = grads_src
        .get(&x)
        .expect("x は requires_grad=true の葉")
        .expect("loss_src に到達する");
    assert_eq!(d_src.as_slice().unwrap(), expected.as_slice());
}

/// A4b: `Tape::var_no_grad` で登録した葉（requires_grad=false）を
/// `to_tape` すると、転送先でも requires_grad=false のまま引き継がれる
/// （`Gradients::get` が `GradientTrackingDisabled` を返す）。
#[test]
fn to_tape_preserves_requires_grad_false() {
    let source = Tape::new_with_ops(common::naive_ops());
    let target = Tape::new_with_ops(common::naive_ops());

    let x = source.var_no_grad(&t(vec![1.0, 2.0], &[2]));
    let x_on_target = x.to_tape(&target).expect("転送は失敗しない");

    let w = target.var(&t(vec![3.0, 4.0], &[2]));
    let product = x_on_target.mul(&w).expect("同 shape の要素積は失敗しない");
    let loss = product.sum(None).expect("全軸縮約は失敗しない");
    let grads = target.backward(&loss).expect("w が追跡対象のため成功する");

    let err = grads
        .get(&x_on_target)
        .expect_err("requires_grad=false のまま転送されているため型付きエラー");
    assert!(matches!(err, AutodiffError::GradientTrackingDisabled));

    // w 側は通常どおり勾配を受け取る（x_on_target の値そのもの）。
    let dw = grads.get(&w).expect("w は葉").expect("loss に到達する");
    assert_eq!(dw.as_slice().unwrap(), x.to_tensor().as_slice().unwrap());
}

/// A5: lazy elementwise 連鎖（`add`→`mul` 等）を `to_tape` すると
/// 転送前に実体化され、転送後の値は期待値と bit 一致する。
///
/// **是正（codex-review 指摘。PR #1864）**: 当初実装は期待値を
/// `chain.to_tensor()`（層 2・`materialize_non_fallible`）で取得して
/// いたが、これは呼び出し時点で `chain` の `OnceCell` を確定的に
/// 埋めてしまう（`materialize_non_fallible` は `OnceCell::get_or_init`
/// で結果をキャッシュする）。そのため後続の `to_tape`（層 1・
/// `materialize_fallible` 経由）は「まだ実体化されていない未実体化
/// ノードを `build_lazy_plan` から直接実体化する経路」を通らず、
/// 単に `nodes[id.0].value.get()` で既存のキャッシュ済み値を読むだけ
/// になり、本テストが検証したかった「未実体化ノードの転送」経路が
/// 一度も exercise されていなかった。ここでは期待値を `chain` に触れ
/// ずに入力データから独立に計算し、`chain`（未実体化のまま）を直接
/// `to_tape` へ渡すことで、この経路を実際に通す。
#[test]
fn to_tape_materializes_lazy_elementwise_chain_before_transfer() {
    let source = Tape::new_with_ops(common::naive_ops());
    let target = Tape::new_with_ops(common::naive_ops());

    let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let y_data = vec![5.0f32, 6.0, 7.0, 8.0];
    let x = source.var(&t(x_data.clone(), &[2, 2]));
    let y = source.var(&t(y_data.clone(), &[2, 2]));
    // add→mul は lazy elementwise 連鎖として構築されうる（`docs/
    // fusion-graph-design.md`）。`chain` はここまで一度も `to_tensor()`
    // 等で実体化しておらず、`to_tape` 呼び出し時点で未実体化のまま
    // であることを前提に検証する。
    let chain = x
        .add(&y)
        .expect("同 shape の加算")
        .mul(&x)
        .expect("同 shape の乗算");

    // 期待値は `chain` に触れず、入力データから独立に計算する
    // （`(x + y) * x` の naive ops と同じ FMA 契約は使わないが、
    // 加算・乗算のみの単純な式のため丸め誤差なく bit 一致する）。
    let expected: Vec<f32> = x_data
        .iter()
        .zip(y_data.iter())
        .map(|(xv, yv)| (xv + yv) * xv)
        .collect();

    let chain_on_target = chain
        .to_tape(&target)
        .expect("lazy チェーンでも materialize_fallible 経由で実体化されて転送される");
    assert_eq!(
        chain_on_target.to_tensor().as_slice().unwrap(),
        expected.as_slice()
    );
}

/// A6: `Tape::reset` は葉プレフィックスより後ろに積まれたノードを
/// truncate する（`Tape::var` と同じ契約）。`to_tape` で登録した葉も
/// 「非葉ノードの後に積まれた」場合はこの対象になる。
#[test]
fn to_tape_leaf_is_subject_to_reset_like_ordinary_leaf() {
    let source = Tape::new_with_ops(common::naive_ops());
    let mut target = Tape::new_with_ops(common::naive_ops());

    // target 上でまず演算を 1 つ積み、葉プレフィックスを確定させる
    // （`Tape::freeze_leaf_prefix`。最初の非葉ノードで固定）。
    let a = target.var(&t(vec![1.0, 2.0], &[2]));
    let b = target.var(&t(vec![3.0, 4.0], &[2]));
    let _ = a.add(&b).expect("葉プレフィックスを確定させるための演算");
    let leaf_count_before = target.leaf_count();

    let x = source.var(&t(vec![9.0, 9.0], &[2]));
    let _x_on_target = x
        .to_tape(&target)
        .expect("転送自体は葉プレフィックス確定後でも成功する");
    assert_eq!(
        target.leaf_count(),
        leaf_count_before,
        "reset で保持される葉プレフィックスは非葉ノード追加時点で固定済みのため、\
         転送された葉自体は reset 前提の葉数に含まれない"
    );

    target.reset();
    assert_eq!(
        target.leaf_count(),
        leaf_count_before,
        "reset 後も保持される葉の個数は不変（転送先の葉プレフィックスの外側に\
         積まれたノードのみ truncate される契約）"
    );
}

// --- A7: checkpoint 解放済み・再計算失敗（poison）ノードの to_tape ---

/// `checkpoint_review_1624.rs::InstrumentedOps` と同型の最小限フィクス
/// チャ。2 回目以降の `gemm` 呼び出しを意図的に失敗させ、checkpoint
/// 解放済みノードの再計算失敗を再現する。
struct FailingSecondGemmOps {
    inner: Box<dyn BackendOps + Send>,
    gemm_calls: Arc<AtomicUsize>,
}

impl BackendOps for FailingSecondGemmOps {
    fn device(&self) -> Device {
        self.inner.device()
    }

    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let n = self.gemm_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if n >= 2 {
            return Err(BackendError::KernelLaunchFailed(format!(
                "FailingSecondGemmOps: 意図的な再計算失敗（呼び出し {n} 回目）"
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

/// A7: checkpoint（イシュー #1624）で解放済みになったノードの再計算が
/// 失敗（poison）した場合、`to_tape` は stale／不正な値を新しい葉へ
/// 複製せず `Err` を返す（`Var::detach` と同じ `materialize_fallible`
/// 経由の fail-closed 方針）。
#[test]
fn to_tape_on_poisoned_checkpoint_node_returns_err() {
    let gemm_calls = Arc::new(AtomicUsize::new(0));
    let ops = FailingSecondGemmOps {
        inner: common::naive_ops(),
        gemm_calls: Arc::clone(&gemm_calls),
    };
    let source = Tape::new_with_ops(Box::new(ops));
    let target = Tape::new_with_ops(common::naive_ops());

    let a = source.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = source.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));

    let out = source
        .checkpoint(|| {
            // `m` は checkpoint 区間の内部ノードのため解放される。
            // `out = m.relu()` は lazy elementwise のためここではまだ
            // 実体化されない（1 回目の gemm のみ消費）。
            let m = a.matmul(&b)?;
            Ok(m.relu())
        })
        .expect("checkpoint 内の forward（1 回目の gemm）は成功する");

    // `to_tape` は層 1（`materialize_fallible`）経由で `out` を実体化
    // しようとするが、`m` の再計算（2 回目の gemm）が失敗するため
    // `Err` を返す（Cell 経由の Cursor Bugbot 指摘対応で `Cell` 型は
    // ここでは不要だが、テスト自体が checkpoint 解放済みノードの
    // 「一度も成功していない」状態を再現していることを示すため
    // `gemm_calls` を直接検証する）。
    let call_snapshot: Cell<usize> = Cell::new(gemm_calls.load(Ordering::SeqCst));
    assert_eq!(
        call_snapshot.get(),
        1,
        "checkpoint 内の forward で 1 回だけ gemm が呼ばれている"
    );

    let err = out
        .to_tape(&target)
        .expect_err("checkpoint 解放済みノードの再計算失敗は Err で伝播する（fail-closed）");
    assert!(
        matches!(err, AutodiffError::Backend(_)),
        "再計算失敗は BackendError 由来の AutodiffError::Backend を期待したが {err:?} だった"
    );
}
