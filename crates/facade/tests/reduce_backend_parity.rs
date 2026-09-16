//! `Var::sum`／`Var::max`（`fandhe_ai_autodiff::var`）の forward／backward
//! を facade 横断で検証する（イシュー #1584・親イシュー #1571）。
//! `Var::min`／`argmax`／`argmin`（イシュー #1720）も本ファイルへ追加
//! 済み（`argmax`／`argmin` は非微分演算のため forward 解析値の比較
//! のみ・CUDA 実機比較は #1720 スコープ外〈`CudaBackendOps::argmax`／
//! `argmin` は未実装のまま既定 `Unsupported`〉）。
//!
//! 属性なしのテストは CPU（`fandhe_ai::tape_for(Device::Cpu)`）のみを
//! 対象とし、既知の解析値との一致を確認する（CI で常時実行）。
//! `#[ignore]` テストは `tape_for(Device::Cuda(0))` の forward／backward
//! を CPU tape と REQ-2 統一複合判定（[`fandhe_ai_backend_cpu::
//! assert_parity`]）で突き合わせる（実機必須。`max`／`min` は Metal で
//! 未実装のため対象外のまま）。
//!
//! イシュー #1896 で `sum`（Metal）が `reduce::MetalReduce` へ結線
//! されたことを受け、`sum`／`mean`／`sum_dims`（axis なしの単純縮約）
//! を CPU tape と bit 完全一致で突き合わせる `#[cfg(target_os =
//! "macos")]` `#[ignore]` テストを追加した（`Var::sum`／`Op::Sum` の
//! VJP は算術を伴わないホスト側のため勾配も bit 一致する。`max`は
//! 引き続き対象外。`conv2d_backend_parity.rs::
//! metal_conv2d_backward_matches_cpu` と同型の cfg 構成）。
//!
//! ```sh
//! cargo test -p fandhe-ai --release --test reduce_backend_parity -- --ignored --nocapture
//! ```
//!
//! イシュー #1719（親 #1601「Phase 2（Tier 1）」）で `Var::mean`／
//! `sum_dims`／`max_dims`／`mean_dims`（複数軸・`keepdim` 対応の縮約）
//! を追加した際、本ファイルへ以下を追補した:
//! - (f) `Var::mean` が `fandhe_ai_backend_cpu::reduction::mean` と
//!   **bit 一致**すること（forward 側の「`sum` の結果を 1 回だけ除算
//!   する」丸め規律が合成実装〈`tape::Op::Mean`〉と参照実装の両方で
//!   同一であることの直接確認）。
//! - (g) `sum_dims`／`max_dims`／`mean_dims` の CPU 解析値突合
//!   （複数軸・`keepdim`）。
//! - (h) `#[ignore]` CUDA 実機 parity（`sum_dims`／`max_dims`／
//!   `mean_dims`。Metal は `max` 自体が `Unsupported`〈TASK-1.9c
//!   スコープ外〉のため `max_dims` は対象外のまま。`sum`（イシュー
//!   #1896 結線済み）を用いる `sum_dims`／`mean_dims` の Metal parity
//!   は下記 (l) が別途追加する）。

use fandhe_ai::{Device, Tensor, tape_for};

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().unwrap().to_vec()
}

/// (a) CPU tape 上で `Var::sum(None)`（全軸）の forward／backward が
/// 既知の解析値と一致することを確認する（`Σ` の勾配は全要素 1）。
#[test]
fn cpu_sum_all_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    let a = tape.var(&tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let loss = a.sum(None).unwrap();
    assert_eq!(dense_vec(&loss.to_tensor()), vec![21.0]);

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    assert_eq!(dense_vec(da), vec![1.0; 6]);
}

/// (b) CPU tape 上で `Var::sum(Some(axis))`（単一軸）の forward／backward
/// が既知の解析値と一致することを確認する。非スカラー `loss` は
/// 「暗黙の総和射影」（`backward.rs::non_scalar_loss_seed_is_implicit_
/// sum_projection` と同じ契約）でシード全要素 1 として逆伝播される
/// ため、`Σ_axis` の勾配もやはり全要素 1 になる。
#[test]
fn cpu_sum_axis_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    let a = tape.var(&tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let loss = a.sum(Some(0)).unwrap();
    assert_eq!(dense_vec(&loss.to_tensor()), vec![5.0, 7.0, 9.0]);

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    assert_eq!(dense_vec(da), vec![1.0; 6]);
}

/// (c) CPU tape 上で `Var::max(None)`（全軸）の forward／backward が
/// 既知の解析値と一致することを確認する（勾配は argmax 位置のみ 1、
/// それ以外は 0。`grad.rs::max_vjp` の先勝ち決定的規約）。
#[test]
fn cpu_max_all_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    // 最大値 6.0 は唯一（index 5）。
    let a = tape.var(&tensor(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]));

    let loss = a.max(None).unwrap();
    assert_eq!(dense_vec(&loss.to_tensor()), vec![6.0]);

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    assert_eq!(dense_vec(da), vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
}

/// (d) CPU tape 上で `Var::max(Some(axis))`（単一軸）の forward／backward
/// が既知の解析値と一致することを確認する。
#[test]
fn cpu_max_axis_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    // shape [2, 3]、axis=0 の各列で最大値の行を確認する。
    // col0: max(1,4)=4 (row1) col1: max(5,2)=5 (row0) col2: max(3,6)=6 (row1)
    let a = tape.var(&tensor(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]));

    let loss = a.max(Some(0)).unwrap();
    assert_eq!(dense_vec(&loss.to_tensor()), vec![4.0, 5.0, 6.0]);

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    // row-major [2,3]: [row0_col0, row0_col1, row0_col2, row1_col0, row1_col1, row1_col2]
    assert_eq!(dense_vec(da), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
}

/// (e) CPU tape 上で `Var::min(None)`（全軸）の forward／backward が
/// 既知の解析値と一致することを確認する（イシュー #1720。`Var::max`
/// と対称・`min` は `BackendOps::min`〈CPU 実装済み〉経由）。
#[test]
fn cpu_min_all_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    // 最小値 1.0 は唯一（index 0）。
    let a = tape.var(&tensor(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]));

    let loss = a.min(None).unwrap();
    assert_eq!(dense_vec(&loss.to_tensor()), vec![1.0]);

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    assert_eq!(dense_vec(da), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

/// (f) CPU tape 上で `Var::min(Some(axis))`（単一軸）の forward／backward
/// が既知の解析値と一致することを確認する（イシュー #1720）。
#[test]
fn cpu_min_axis_forward_and_backward_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    // shape [2, 3]、axis=0 の各列で最小値の行を確認する。
    // col0: min(1,4)=1 (row0) col1: min(5,2)=2 (row1) col2: min(3,6)=3 (row0)
    let a = tape.var(&tensor(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]));

    let loss = a.min(Some(0)).unwrap();
    assert_eq!(dense_vec(&loss.to_tensor()), vec![1.0, 2.0, 3.0]);

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    assert_eq!(dense_vec(da), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

/// (g) CPU tape 上で `Var::argmax`／`Var::argmin`（全軸・単一軸）が
/// 既知の解析値と一致することを確認する（イシュー #1720。非微分演算
/// のため `to_tensor()`／`backward` は関与しない）。
#[test]
fn cpu_argmax_and_argmin_match_analytic_values() {
    let tape = tape_for(Device::Cpu).unwrap();
    let a = tape.var(&tensor(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]));

    let argmax_all = a.argmax(None).unwrap();
    assert_eq!(argmax_all.contiguous().as_slice().unwrap(), &[5]);
    let argmin_all = a.argmin(None).unwrap();
    assert_eq!(argmin_all.contiguous().as_slice().unwrap(), &[0]);

    // axis=0: col0 argmax=row1(idx1) argmin=row0(idx0)
    //         col1 argmax=row0(idx0) argmin=row1(idx1)
    //         col2 argmax=row1(idx1) argmin=row0(idx0)
    let argmax_axis0 = a.argmax(Some(0)).unwrap();
    assert_eq!(argmax_axis0.contiguous().as_slice().unwrap(), &[1, 0, 1]);
    let argmin_axis0 = a.argmin(Some(0)).unwrap();
    assert_eq!(argmin_axis0.contiguous().as_slice().unwrap(), &[0, 1, 0]);
}

/// (h) `Var::mean`（全軸・単一軸）が `fandhe_ai_backend_cpu::
/// reduction::mean`（参照実装）と**bit 一致**することを確認する
/// （イシュー #1719）。forward（`tape::Op::Mean`）は「`sum` の結果を
/// ホスト側で 1 回だけ除算する」合成実装であり、`reduction::mean` も
/// 同じ丸め規律（`sum` の後に 1 回だけ除算）のため理論上 bit 一致する。
#[test]
fn cpu_mean_matches_reduction_mean_bit_exact() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = [2usize, 3];
    for dim in [None, Some(0usize), Some(1usize)] {
        let tape = tape_for(Device::Cpu).unwrap();
        let a = tape.var(&tensor(data.clone(), &shape));
        let got = a.mean(dim).unwrap().to_tensor();

        let reference =
            fandhe_ai_backend_cpu::reduction::mean(&tensor(data.clone(), &shape), dim).unwrap();
        assert_eq!(
            dense_vec(&got),
            dense_vec(&reference),
            "mean(dim={dim:?}) が reduction::mean と bit 一致しない"
        );
    }
}

/// (i) `sum_dims`／`max_dims`／`mean_dims`（複数軸・`keepdim`）の CPU
/// 解析値突合（イシュー #1719）。shape `[2, 3, 4]` の `dims=[0, 2]` は
/// kept 軸 `[1]` が縮約軸の間に挟まる非連続ケース（`permute` 必須）。
#[test]
fn cpu_sum_max_mean_dims_match_analytic_values() {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let shape = [2usize, 3, 4];
    let dims = [0usize, 2usize];

    // dims=[0,2] を手計算した解析値（kept 軸 1 の各要素につき、
    // 軸 0（サイズ 2）× 軸 2（サイズ 4）＝8 要素を縮約）。
    // 総和は `Σ_{i,k} data[i,j,k]`（j 固定）。
    let expected_sum = [
        // j=0: index (i,0,k) for i in 0..2, k in 0..4
        // i=0: 0,1,2,3 / i=1: 12,13,14,15 → sum=60
        60.0, // j=1: i=0: 4,5,6,7 / i=1: 16,17,18,19 → sum=92
        92.0, // j=2: i=0: 8,9,10,11 / i=1: 20,21,22,23 → sum=124
        124.0,
    ];

    let tape = tape_for(Device::Cpu).unwrap();
    let a = tape.var(&tensor(data.clone(), &shape));

    let sum_squeezed = a.sum_dims(&dims, false).unwrap();
    assert_eq!(sum_squeezed.to_tensor().shape(), &[3]);
    assert_eq!(dense_vec(&sum_squeezed.to_tensor()), expected_sum);

    let a2 = tape.var(&tensor(data.clone(), &shape));
    let sum_keepdim = a2.sum_dims(&dims, true).unwrap();
    assert_eq!(sum_keepdim.to_tensor().shape(), &[1, 3, 1]);
    assert_eq!(dense_vec(&sum_keepdim.to_tensor()), expected_sum);

    let a3 = tape.var(&tensor(data.clone(), &shape));
    let max_squeezed = a3.max_dims(&dims, false).unwrap();
    // j=0: max(0,1,2,3,12,13,14,15)=15 / j=1: max(...)=19 / j=2: max(...)=23
    assert_eq!(dense_vec(&max_squeezed.to_tensor()), vec![15.0, 19.0, 23.0]);

    let a4 = tape.var(&tensor(data, &shape));
    let mean_squeezed = a4.mean_dims(&dims, false).unwrap();
    let count = (2 * 4) as f32;
    let expected_mean: Vec<f32> = expected_sum.iter().map(|v| v / count).collect();
    assert_eq!(dense_vec(&mean_squeezed.to_tensor()), expected_mean);

    // 勾配経路も壊れていないことを確認する（合計 loss へ集約し backward）。
    let a5 = tape.var(&tensor((0..24).map(|v| v as f32).collect(), &shape));
    let loss = a5.sum_dims(&dims, false).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a5).unwrap().expect("a5 は loss に到達する");
    // sum_dims の勾配は縮約対象軸全体へ 1 を複製するだけ（sum の VJP と同じ）。
    assert_eq!(dense_vec(da), vec![1.0; 24]);
}

/// (j) `tape_for(Device::Cuda(0))` の `Var::sum`／`Var::max`（全軸・単一
/// 軸）forward／backward が CPU tape と REQ-2 統一複合判定で一致する
/// ことを確認する（実機必須）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_sum_and_max_forward_and_backward_match_cpu_tape_on_real_device() {
    let data = {
        // 決定的疑似乱数（Xorshift64Star。U[-0.5, 0.5)）。強い相殺を
        // 起こさない系列で `assert_parity`（REQ-2）の前提を満たす。
        let mut state = 0x1234_5678_9abc_def0u64;
        (0..24)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
            })
            .collect::<Vec<f32>>()
    };
    let shape = [4usize, 6];

    for dim in [None, Some(0usize), Some(1usize)] {
        let cpu_tape = tape_for(Device::Cpu).unwrap();
        let cpu_a = cpu_tape.var(&tensor(data.clone(), &shape));
        let cuda_tape = tape_for(Device::Cuda(0))
            .expect("CUDA device 0 must be available on ignored test runner");
        let cuda_a = cuda_tape.var(&tensor(data.clone(), &shape));

        // sum
        let cpu_sum = cpu_a.sum(dim).unwrap();
        let cuda_sum = cuda_a.sum(dim).unwrap();
        assert_parity_tensors(
            &cuda_sum.to_tensor(),
            &cpu_sum.to_tensor(),
            &format!("sum forward: dim={dim:?}"),
        );
        let cpu_grads = cpu_tape.backward(&cpu_sum).unwrap();
        let cuda_grads = cuda_tape.backward(&cuda_sum).unwrap();
        let cpu_da = cpu_grads
            .get(&cpu_a)
            .unwrap()
            .expect("a は loss に到達する");
        let cuda_da = cuda_grads
            .get(&cuda_a)
            .unwrap()
            .expect("a は loss に到達する");
        assert_parity_tensors(cuda_da, cpu_da, &format!("sum backward: dim={dim:?}"));

        // max
        let cpu_tape = tape_for(Device::Cpu).unwrap();
        let cpu_a = cpu_tape.var(&tensor(data.clone(), &shape));
        let cuda_tape = tape_for(Device::Cuda(0))
            .expect("CUDA device 0 must be available on ignored test runner");
        let cuda_a = cuda_tape.var(&tensor(data.clone(), &shape));

        let cpu_max = cpu_a.max(dim).unwrap();
        let cuda_max = cuda_a.max(dim).unwrap();
        assert_parity_tensors(
            &cuda_max.to_tensor(),
            &cpu_max.to_tensor(),
            &format!("max forward: dim={dim:?}"),
        );
        let cpu_grads = cpu_tape.backward(&cpu_max).unwrap();
        let cuda_grads = cuda_tape.backward(&cuda_max).unwrap();
        let cpu_da = cpu_grads
            .get(&cpu_a)
            .unwrap()
            .expect("a は loss に到達する");
        let cuda_da = cuda_grads
            .get(&cuda_a)
            .unwrap()
            .expect("a は loss に到達する");
        assert_parity_tensors(cuda_da, cpu_da, &format!("max backward: dim={dim:?}"));

        // min（イシュー #1720。CUDA は `CudaBackendOps::min` 実装済み）
        let cpu_tape = tape_for(Device::Cpu).unwrap();
        let cpu_a = cpu_tape.var(&tensor(data.clone(), &shape));
        let cuda_tape = tape_for(Device::Cuda(0))
            .expect("CUDA device 0 must be available on ignored test runner");
        let cuda_a = cuda_tape.var(&tensor(data.clone(), &shape));

        let cpu_min = cpu_a.min(dim).unwrap();
        let cuda_min = cuda_a.min(dim).unwrap();
        assert_parity_tensors(
            &cuda_min.to_tensor(),
            &cpu_min.to_tensor(),
            &format!("min forward: dim={dim:?}"),
        );
        let cpu_grads = cpu_tape.backward(&cpu_min).unwrap();
        let cuda_grads = cuda_tape.backward(&cuda_min).unwrap();
        let cpu_da = cpu_grads
            .get(&cpu_a)
            .unwrap()
            .expect("a は loss に到達する");
        let cuda_da = cuda_grads
            .get(&cuda_a)
            .unwrap()
            .expect("a は loss に到達する");
        assert_parity_tensors(cuda_da, cpu_da, &format!("min backward: dim={dim:?}"));
    }
}

/// (k) `tape_for(Device::Cuda(0))` の `sum_dims`／`max_dims`／
/// `mean_dims`（複数軸・非連続 `dims`）forward／backward が CPU tape
/// と REQ-2 統一複合判定で一致することを確認する（実機必須。イシュー
/// #1719）。`merge_for_reduction`（`crate::reduce_dims`）が経由する
/// `permute`／`contiguous`／`reshape` は 3 バックエンドとも既存実装
/// （イシュー #1597／#1598／#1620）のため、本テストは主に併合後の単一
/// 軸 `sum`／`max` 呼び出しが CUDA 実機でも一致することの確認になる。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_sum_max_mean_dims_forward_and_backward_match_cpu_tape_on_real_device() {
    let data = {
        // (e) と同じ決定的疑似乱数（Xorshift64Star。U[-0.5, 0.5)）。
        let mut state = 0x0fed_cba9_8765_4321u64;
        (0..24)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
            })
            .collect::<Vec<f32>>()
    };
    let shape = [2usize, 3, 4];
    let dims = [0usize, 2usize]; // kept=[1]（非連続。permute 必須）

    for keepdim in [false, true] {
        let cpu_tape = tape_for(Device::Cpu).unwrap();
        let cpu_a = cpu_tape.var(&tensor(data.clone(), &shape));
        let cuda_tape = tape_for(Device::Cuda(0))
            .expect("CUDA device 0 must be available on ignored test runner");
        let cuda_a = cuda_tape.var(&tensor(data.clone(), &shape));

        // sum_dims
        let cpu_sum = cpu_a.sum_dims(&dims, keepdim).unwrap();
        let cuda_sum = cuda_a.sum_dims(&dims, keepdim).unwrap();
        assert_parity_tensors(
            &cuda_sum.to_tensor(),
            &cpu_sum.to_tensor(),
            &format!("sum_dims forward: keepdim={keepdim}"),
        );
        let cpu_grads = cpu_tape.backward(&cpu_sum).unwrap();
        let cuda_grads = cuda_tape.backward(&cuda_sum).unwrap();
        let cpu_da = cpu_grads
            .get(&cpu_a)
            .unwrap()
            .expect("a は loss に到達する");
        let cuda_da = cuda_grads
            .get(&cuda_a)
            .unwrap()
            .expect("a は loss に到達する");
        assert_parity_tensors(
            cuda_da,
            cpu_da,
            &format!("sum_dims backward: keepdim={keepdim}"),
        );

        // max_dims
        let cpu_tape = tape_for(Device::Cpu).unwrap();
        let cpu_a = cpu_tape.var(&tensor(data.clone(), &shape));
        let cuda_tape = tape_for(Device::Cuda(0))
            .expect("CUDA device 0 must be available on ignored test runner");
        let cuda_a = cuda_tape.var(&tensor(data.clone(), &shape));

        let cpu_max = cpu_a.max_dims(&dims, keepdim).unwrap();
        let cuda_max = cuda_a.max_dims(&dims, keepdim).unwrap();
        assert_parity_tensors(
            &cuda_max.to_tensor(),
            &cpu_max.to_tensor(),
            &format!("max_dims forward: keepdim={keepdim}"),
        );
        let cpu_grads = cpu_tape.backward(&cpu_max).unwrap();
        let cuda_grads = cuda_tape.backward(&cuda_max).unwrap();
        let cpu_da = cpu_grads
            .get(&cpu_a)
            .unwrap()
            .expect("a は loss に到達する");
        let cuda_da = cuda_grads
            .get(&cuda_a)
            .unwrap()
            .expect("a は loss に到達する");
        assert_parity_tensors(
            cuda_da,
            cpu_da,
            &format!("max_dims backward: keepdim={keepdim}"),
        );

        // mean_dims
        let cpu_tape = tape_for(Device::Cpu).unwrap();
        let cpu_a = cpu_tape.var(&tensor(data.clone(), &shape));
        let cuda_tape = tape_for(Device::Cuda(0))
            .expect("CUDA device 0 must be available on ignored test runner");
        let cuda_a = cuda_tape.var(&tensor(data.clone(), &shape));

        let cpu_mean = cpu_a.mean_dims(&dims, keepdim).unwrap();
        let cuda_mean = cuda_a.mean_dims(&dims, keepdim).unwrap();
        assert_parity_tensors(
            &cuda_mean.to_tensor(),
            &cpu_mean.to_tensor(),
            &format!("mean_dims forward: keepdim={keepdim}"),
        );
        let cpu_grads = cpu_tape.backward(&cpu_mean).unwrap();
        let cuda_grads = cuda_tape.backward(&cuda_mean).unwrap();
        let cpu_da = cpu_grads
            .get(&cpu_a)
            .unwrap()
            .expect("a は loss に到達する");
        let cuda_da = cuda_grads
            .get(&cuda_a)
            .unwrap()
            .expect("a は loss に到達する");
        assert_parity_tensors(
            cuda_da,
            cpu_da,
            &format!("mean_dims backward: keepdim={keepdim}"),
        );
    }
}

/// テンソル同士の統一複合判定（`linear_forward_device_real_device.rs`
/// と同じ方式。REQ-2 の唯一の実体である
/// [`fandhe_ai_backend_cpu::assert_parity`] へ委譲する）。
fn assert_parity_tensors(actual: &Tensor<f32>, expected: &Tensor<f32>, ctx: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{ctx}: shape mismatch");
    let a = actual.contiguous();
    let e = expected.contiguous();
    fandhe_ai_backend_cpu::assert_parity(ctx, a.as_slice().unwrap(), e.as_slice().unwrap());
}

/// (l) `tape_for(Device::Metal)` の `Var::sum(None)`（全軸）
/// forward／backward が CPU tape と**bit 完全一致**することを確認する
/// （イシュー #1896。`sum`（Metal）は `ops::MetalBackendOps::sum` が
/// `reduce::MetalReduce` へ結線されており CPU 参照実装〈`fandhe_ai_
/// backend_cpu::reduction::sum`〉と bit 完全一致する契約〈`crate::
/// reduce_model` doc〉。`Op::Sum` の VJP は算術を伴わないホスト側の
/// ブロードキャストのみ〈`grad.rs`〉のため勾配も bit 一致する）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_sum_all_forward_and_backward_match_cpu_bit_exact() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = [2usize, 3];

    let metal_tape = tape_for(Device::Metal).unwrap();
    let metal_a = metal_tape.var(&tensor(data.clone(), &shape));
    let metal_loss = metal_a.sum(None).unwrap();
    let metal_grads = metal_tape.backward(&metal_loss).unwrap();
    let metal_da = metal_grads
        .get(&metal_a)
        .unwrap()
        .expect("a は loss に到達する");

    let cpu_tape = tape_for(Device::Cpu).unwrap();
    let cpu_a = cpu_tape.var(&tensor(data, &shape));
    let cpu_loss = cpu_a.sum(None).unwrap();
    let cpu_grads = cpu_tape.backward(&cpu_loss).unwrap();
    let cpu_da = cpu_grads
        .get(&cpu_a)
        .unwrap()
        .expect("a は loss に到達する");

    assert_eq!(
        dense_vec(&metal_loss.to_tensor()),
        dense_vec(&cpu_loss.to_tensor()),
        "sum(None) forward が Metal/CPU で bit 一致しない"
    );
    assert_eq!(
        dense_vec(metal_da),
        dense_vec(cpu_da),
        "sum(None) backward が Metal/CPU で bit 一致しない"
    );
}

/// (m) `tape_for(Device::Metal)` の `Var::sum(Some(axis))`（単一軸）
/// forward／backward が CPU tape と**bit 完全一致**することを確認する
/// （イシュー #1896。(l) と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_sum_axis_forward_and_backward_match_cpu_bit_exact() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = [2usize, 3];

    let metal_tape = tape_for(Device::Metal).unwrap();
    let metal_a = metal_tape.var(&tensor(data.clone(), &shape));
    let metal_loss = metal_a.sum(Some(0)).unwrap();
    let metal_grads = metal_tape.backward(&metal_loss).unwrap();
    let metal_da = metal_grads
        .get(&metal_a)
        .unwrap()
        .expect("a は loss に到達する");

    let cpu_tape = tape_for(Device::Cpu).unwrap();
    let cpu_a = cpu_tape.var(&tensor(data, &shape));
    let cpu_loss = cpu_a.sum(Some(0)).unwrap();
    let cpu_grads = cpu_tape.backward(&cpu_loss).unwrap();
    let cpu_da = cpu_grads
        .get(&cpu_a)
        .unwrap()
        .expect("a は loss に到達する");

    assert_eq!(
        dense_vec(&metal_loss.to_tensor()),
        dense_vec(&cpu_loss.to_tensor()),
        "sum(Some(0)) forward が Metal/CPU で bit 一致しない"
    );
    assert_eq!(
        dense_vec(metal_da),
        dense_vec(cpu_da),
        "sum(Some(0)) backward が Metal/CPU で bit 一致しない"
    );
}

/// (n) `tape_for(Device::Metal)` の `Var::mean`（全軸・単一軸）
/// forward／backward が CPU tape と**bit 完全一致**することを確認する
/// （イシュー #1896。`Var::mean` は `sum` の結果をホスト側で 1 回だけ
/// `n as f32` で除算する合成実装〈`var.rs::mean`〉であり、`sum` 自体が
/// Metal/CPU で bit 一致するため mean も bit 一致する）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_mean_forward_and_backward_match_cpu_bit_exact() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = [2usize, 3];

    for dim in [None, Some(0usize), Some(1usize)] {
        let metal_tape = tape_for(Device::Metal).unwrap();
        let metal_a = metal_tape.var(&tensor(data.clone(), &shape));
        let metal_loss = metal_a.mean(dim).unwrap();
        let metal_grads = metal_tape.backward(&metal_loss).unwrap();
        let metal_da = metal_grads
            .get(&metal_a)
            .unwrap()
            .expect("a は loss に到達する");

        let cpu_tape = tape_for(Device::Cpu).unwrap();
        let cpu_a = cpu_tape.var(&tensor(data.clone(), &shape));
        let cpu_loss = cpu_a.mean(dim).unwrap();
        let cpu_grads = cpu_tape.backward(&cpu_loss).unwrap();
        let cpu_da = cpu_grads
            .get(&cpu_a)
            .unwrap()
            .expect("a は loss に到達する");

        assert_eq!(
            dense_vec(&metal_loss.to_tensor()),
            dense_vec(&cpu_loss.to_tensor()),
            "mean(dim={dim:?}) forward が Metal/CPU で bit 一致しない"
        );
        assert_eq!(
            dense_vec(metal_da),
            dense_vec(cpu_da),
            "mean(dim={dim:?}) backward が Metal/CPU で bit 一致しない"
        );
    }
}
