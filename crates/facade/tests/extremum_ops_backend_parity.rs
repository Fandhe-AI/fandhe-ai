//! `fandhe_ai_autodiff::extremum_ops`（イシュー #2154・facade 非公開の
//! 内部入口。`crates/autodiff/src/extremum_ops.rs` モジュール doc 参照）
//! のバックエンド間 parity テスト（`reduce_ops_backend_parity.rs` と
//! 同型）。
//!
//! `extremum_ops` は facade から再エクスポートされないため、本テストは
//! `fandhe_ai_autodiff::extremum_ops::*` を直接 use する（facade の dev
//! 依存に `fandhe-ai-autodiff` が既に含まれている）。
//!
//! `amax`／`amin` の forward は `Var::max`／`min` と bit 同一（選択演算
//! であり縮約順序に依存しない）ため、forward・backward いずれも
//! **bit 完全一致**を第一の判定とする（`reduce_ops_backend_parity.rs`
//! の `any`／`all` と同じ扱い。`prod`／`logsumexp`／`norm_p` の REQ-2
//! 統一複合判定は本ファイルでは使わない）。
//!
//! - 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps`〉と
//!   `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`〉の突き合わせ）:
//!   forward・backward の bit 完全一致テストを 1 対ずつ。
//! - `#[ignore]`（`tape_for(Device::Metal)`〈`cfg(target_os =
//!   "macos")` 限定〉／`tape_for(Device::Cuda(0))` で同じ経路を CPU
//!   tape と比較）: 対称に置く（計 4 件）。
//!
//!   実機（DGX Spark GB10／Apple Silicon）への到達手段が本エージェント
//!   実行環境にないため未実施のまま Mac／GB10 セッションへ申し送る
//!   （`docs/perf/logs/amax-amin-2154/README.md`）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::extremum_ops::{amax, amin};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

trait VarSource {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_>;
}

impl VarSource for fandhe_ai::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

impl VarSource for fandhe_ai_autodiff::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

fn f32_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// `amax`／`amin` のタイを含む fixture（2x2。列 0 は `5.0` の 2 つ組
/// タイ、列 1 はタイなし）。
fn tie_fixture() -> Tensor<f32> {
    Tensor::new(vec![5.0, 1.0, 5.0, 9.0], &[2, 2]).expect("test fixture: shape 一致")
}

/// `amax`／`amin` forward が CPU（`fandhe_ai::tape()`）と NaiveOps
/// （`fandhe_ai_autodiff::Tape::new()`）で bit 完全一致することを
/// 確認する（選択演算であり縮約順序に依存しないため）。
#[test]
fn cpu_amax_amin_forward_bit_matches_naive_reference() {
    let data = tie_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    for dim in [None, Some(0), Some(1)] {
        assert_eq!(
            f32_bits(&amax(&x_cpu, dim).unwrap().to_tensor()),
            f32_bits(&amax(&x_naive, dim).unwrap().to_tensor()),
            "amax dim={dim:?}"
        );
        assert_eq!(
            f32_bits(&amin(&x_cpu, dim).unwrap().to_tensor()),
            f32_bits(&amin(&x_naive, dim).unwrap().to_tensor()),
            "amin dim={dim:?}"
        );
    }
}

/// `amax`／`amin` backward（均等分配 VJP。`extremum_even_split_vjp`）が
/// CPU と NaiveOps でいずれも bit 完全一致することを確認する。VJP は
/// `input`／`out_value` から決定論的に計算されるホスト側計算のため、
/// バックエンドに依らず同一結果になる。
#[test]
fn cpu_amax_amin_gradient_bit_matches_naive_reference() {
    let data = tie_fixture();

    // `dim=Some(0)`（列ごとに縮約）: 列 0 は行 0・1 とも `5.0` でタイ、
    // 列 1 はタイなし（`1.0` vs `9.0`）。均等分配（先勝ちとの違いの
    // 直接確認）が列 0 でのみ現れる。
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = amax(&x_cpu, Some(0)).unwrap().sum(None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let loss_naive = amax(&x_naive, Some(0)).unwrap().sum(None).unwrap();
    let dx_naive = naive_tape
        .backward(&loss_naive)
        .unwrap()
        .get(&x_naive)
        .unwrap()
        .unwrap()
        .clone();
    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_naive), "amax gradient");
    assert_eq!(dx_cpu.host_slice().into_owned(), vec![0.5, 0.0, 0.5, 1.0]);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = amin(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let loss_naive = amin(&x_naive, None).unwrap();
    let dx_naive = naive_tape
        .backward(&loss_naive)
        .unwrap()
        .get(&x_naive)
        .unwrap()
        .unwrap()
        .clone();
    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_naive), "amin gradient");
    assert_eq!(dx_cpu.host_slice().into_owned(), vec![0.0, 1.0, 0.0, 0.0]);
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: Mac／DGX Spark GB10 実機セッションへ
// 申し送る（`docs/perf/logs/amax-amin-2154/README.md`）。
// ---------------------------------------------------------------------

/// `amax`／`amin` forward の CPU／Metal 実機比較。CPU 側の同型カバレッジ
/// は `cpu_amax_amin_forward_bit_matches_naive_reference`。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/amax-amin-2154/README.md 参照"]
fn metal_amax_amin_forward_matches_cpu_reference() {
    let data = tie_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);

    for dim in [None, Some(0), Some(1)] {
        assert_eq!(
            f32_bits(&amax(&x_cpu, dim).unwrap().to_tensor()),
            f32_bits(&amax(&x_metal, dim).unwrap().to_tensor()),
            "amax dim={dim:?}"
        );
        assert_eq!(
            f32_bits(&amin(&x_cpu, dim).unwrap().to_tensor()),
            f32_bits(&amin(&x_metal, dim).unwrap().to_tensor()),
            "amin dim={dim:?}"
        );
    }
}

/// `amax`／`amin` forward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/amax-amin-2154/README.md 参照"]
fn cuda_amax_amin_forward_matches_cpu_reference() {
    let data = tie_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);

    for dim in [None, Some(0), Some(1)] {
        assert_eq!(
            f32_bits(&amax(&x_cpu, dim).unwrap().to_tensor()),
            f32_bits(&amax(&x_cuda, dim).unwrap().to_tensor()),
            "amax dim={dim:?}"
        );
        assert_eq!(
            f32_bits(&amin(&x_cpu, dim).unwrap().to_tensor()),
            f32_bits(&amin(&x_cuda, dim).unwrap().to_tensor()),
            "amin dim={dim:?}"
        );
    }
}

/// `amax`／`amin` backward の CPU／Metal 実機比較（VJP はホスト側計算
/// のため bit 完全一致するはず。`Var::min` の Metal 経路が既定の
/// `Unsupported` フォールバック〈ホスト計算〉へ到達することもここで
/// 併せて確認する）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/amax-amin-2154/README.md 参照"]
fn metal_amax_amin_gradient_matches_cpu_reference() {
    let data = tie_fixture();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = amax(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);
    let loss_metal = amax(&x_metal, None).unwrap();
    let dx_metal = metal_tape
        .backward(&loss_metal)
        .unwrap()
        .get(&x_metal)
        .unwrap()
        .unwrap()
        .clone();
    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_metal), "amax gradient");

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = amin(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);
    let loss_metal = amin(&x_metal, None).unwrap();
    let dx_metal = metal_tape
        .backward(&loss_metal)
        .unwrap()
        .get(&x_metal)
        .unwrap()
        .unwrap()
        .clone();
    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_metal), "amin gradient");
}

/// `amax`／`amin` backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/amax-amin-2154/README.md 参照"]
fn cuda_amax_amin_gradient_matches_cpu_reference() {
    let data = tie_fixture();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = amax(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);
    let loss_cuda = amax(&x_cuda, None).unwrap();
    let dx_cuda = cuda_tape
        .backward(&loss_cuda)
        .unwrap()
        .get(&x_cuda)
        .unwrap()
        .unwrap()
        .clone();
    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_cuda), "amax gradient");

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = amin(&x_cpu, None).unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);
    let loss_cuda = amin(&x_cuda, None).unwrap();
    let dx_cuda = cuda_tape
        .backward(&loss_cuda)
        .unwrap()
        .get(&x_cuda)
        .unwrap()
        .unwrap()
        .clone();
    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_cuda), "amin gradient");
}

// --- 確保前のバイト数上限検査（`reduce_ops_backend_parity.rs::
// cpu_norm_p_two_rejects_huge_broadcast_before_delegating` と同型の
// facade 経路代表テスト。イシュー #2154） ---

#[test]
fn cpu_amax_rejects_huge_broadcast_before_materializing() {
    let cpu_tape = fandhe_ai::tape();
    let base = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).expect("test fixture: shape 一致");
    let huge = base.broadcast_to(&[1usize << 61, 4]).unwrap();
    let x = cpu_tape.make_var(&huge);
    assert!(matches!(
        amax(&x, Some(1)),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
}
