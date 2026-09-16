//! TASK-1.9d（#47）: Metal 実機での `BackendOps` 経由数値一致検証。
//!
//! `cpu_metal_parity.rs`（TASK-2.2c・#55）は `MetalGemm::dispatch` を直接
//! 呼び出す形で CPU-Metal ペアの数値一致を検証しているが、本ファイルは
//! 抽象層 `fandhe_ai_tensor_core::backend_ops::MetalBackendOps`（TASK-1.9c・#46）が
//! 内部で使う `MetalGemm::dispatch_auto`（動的タイル選択。TASK-1.8c・#40）
//! を経由した場合にも同じ複合判定（REQ-2）が成立することを固定する。
//! 判定式・許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の参照とする
//! （`.claude/rules/coding-rust.md`）。
//!
//! `cpu_metal_parity.rs`（基準形状 512^3・K=4096 ストレス）とは異なる形状を
//! 選び、`tile.rs::select` の動的タイル選択境界（`SMALL=64`。`crate::tile`
//! 参照）近傍を 1〜2 ケース含める。`LARGE=512` 境界はイシュー #744 是正で
//! **真の正方立方形状（`m == n == k`）に限り、かつ実測範囲内（`m <= 4096`）
//! でのみ**撤去済み（この範囲の `m == n == k` は `CANDIDATES[3]` を返す）。
//! `m != n` の準正方長方形、`m == n` でも `k != m`（K 未実測の正方出力。
//! 例: (2048,2048,64)）、および `m == n == k` でも 4096 超（実測対象外）の
//! 場合は `select_with_occupancy` が引き続き `m >= LARGE && n >= LARGE` で
//! 境界前後の候補を切り替える（#744 是正前と同一挙動。PR #760 codex-review
//! 指摘対応で本コメントを実装へ整合。詳細は
//! `docs/perf/metal-tile-select-correction.md`）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`cpu_metal_parity.rs` と同方針。`#![cfg(target_os = "macos")]` により
//! Linux self-hosted CI ではコンパイル対象外になり、`#[ignore]` により
//! 通常の `cargo test` からも除外される）。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨。
//! `docs/backend-metal-real-device-testing.md` 参照）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// `MetalBackendOps::gemm`（`dispatch_auto` 委譲）を CPU `BackendOps::gemm`
/// と複合判定で突き合わせる。
fn assert_backend_ops_gemm_parity(seed_a: u64, seed_b: u64, m: usize, n: usize, k: usize) {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let a_data = Xorshift64Star::new(seed_a).fill_vec(m * k);
    let b_data = Xorshift64Star::new(seed_b).fill_vec(k * n);
    let a = Tensor::new(a_data, &[m, k]).expect("valid tensor");
    let b = Tensor::new(b_data, &[k, n]).expect("valid tensor");

    let cpu_result = cpu.gemm(&a, &b).expect("cpu gemm always succeeds");
    let metal_result = metal
        .gemm(&a, &b)
        .expect("MetalBackendOps::gemm must succeed on Metal-equipped test runner");

    assert_eq!(metal_result.shape(), cpu_result.shape());
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("BackendOps cpu-metal gemm parity m={m} n={n} k={k}"),
        metal_result.as_slice().expect("contiguous"),
        cpu_result.as_slice().expect("contiguous"),
    );
}

/// 基準ケース（`tile.rs::select` の中形状経路。本ケースは `m == n == k = 96`
/// の真の正方立方形状かつ実測上限〈4096〉内のため、#744 是正後は
/// `CANDIDATES[3]`〈32x32〉が選ばれる想定。この想定が成り立つのは
/// `m == n == k` かつ `m <= 4096` の範囲に限る（PR #760 レビュー対応で本
/// コメントをファイル冒頭の境界説明・`tile.rs::select_with_occupancy` の
/// 実装へ整合。全帯域一律ではない）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backend_ops_gemm_matches_cpu_mid_square_shape() {
    assert_backend_ops_gemm_parity(501, 502, 96, 96, 96);
}

/// `tile.rs::select` の動的タイル選択境界（`SMALL=64`）近傍ケース:
/// `m/n/k` のいずれかが 64 未満のとき `SINGLE_SIMDGROUP_8X8` へ分岐する
/// （`tile.rs` 参照）。63（境界未満）・65（境界以上）を隣接させて選択境界の
/// 両側を実機で確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backend_ops_gemm_matches_cpu_near_small_tile_threshold() {
    assert_backend_ops_gemm_parity(503, 504, 63, 63, 63);
    assert_backend_ops_gemm_parity(505, 506, 65, 65, 65);
}

/// `tile.rs::select` の縦長・横長分岐（`ASPECT_RATIO=2`）ケース。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backend_ops_gemm_matches_cpu_tall_and_wide_shapes() {
    assert_backend_ops_gemm_parity(507, 508, 128, 64, 96); // 縦長（m >= 2n）
    assert_backend_ops_gemm_parity(509, 510, 64, 128, 96); // 横長（n >= 2m）
}

/// 非正方・非 2 冪境界（`cpu_metal_parity.rs` の基準形状 512^3 とは異なる
/// 素数近傍形状）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backend_ops_gemm_matches_cpu_non_power_of_two_shape() {
    assert_backend_ops_gemm_parity(511, 512, 97, 71, 83);
}

/// `max`（reduction）の `Unsupported` 契約を検証する。
///
/// `MetalBackendOps::max`（`backend-metal/src/ops.rs`）は
/// `MetalContext::new` を呼ばず即座に `BackendError::Unsupported` を返す
/// 実装（スコープ外の未実装カーネル用プレースホルダ。
/// out-of-scope-tracking.md 対象）のため、本テストは Metal 実機・デバイス
/// 初期化を一切必要としない。他の実機依存テストと同じファイルに
/// 置かれてはいるが（`MetalBackendOps` 自体が `cfg(target_os = "macos")`
/// 限定のため非 macOS 環境ではコンパイル対象に入らない）、実機依存では
/// ないため `#[ignore]` を付けない。macOS 上での通常の
/// `cargo test -p fandhe-ai-backend-metal`（`--ignored` なし）で毎回実行され、
/// `max` が `Unsupported` を返し続けることを回帰的に固定する。
///
/// `sum` はイシュー #1896 で `reduce::MetalReduce` へ結線されデバイス
/// 初期化を要するようになったため本テストの対象外とし、実機での数値
/// 一致検証は下記 `#[ignore]` 付き `backend_ops_sum_matches_cpu_bit_exact`
/// が担う。
///
/// `add`／`mul`／`relu`／`exp`／`tanh` はイシュー #605 で実カーネル化
/// 済みのため（`elementwise::MetalElementwise` 経由。`MetalContext::new`
/// を呼びデバイス初期化を要する）本テストの対象外とし、実機での数値
/// 一致検証は下記 `#[ignore]` 付き
/// `backend_ops_elementwise_matches_cpu` が引き継ぐ（Cursor Bugbot
/// 指摘・PR #717 レビュースレッド。旧テストはこの分離前に 5 演算も
/// `Unsupported` と誤って assert しており、実カーネル化後は macOS の
/// 通常 `cargo test` が失敗していた）。
///
/// `backend_ops_dispatch.rs`（`backend-cpu/tests/`）は `ops_for` 経由の
/// GEMM ディスパッチのみを検証しており、Metal の reduction カバレッジは
/// 含まない（Cursor Bugbot 指摘・PR #264 レビュースレッド。旧コメントは
/// 誤って「テストと分離していない」と記述していたが、実際は「そもそも
/// カバーしていない」が正確な記述である）。
#[test]
fn max_remains_unsupported_without_device_init() {
    let metal = MetalBackendOps::new();
    let a = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]).expect("valid tensor");

    assert!(matches!(
        metal.max(&a, None),
        Err(BackendError::Unsupported(_))
    ));
}

/// `sum`（`reduce::MetalReduce` 経由。イシュー #1896）が
/// `fandhe_ai_backend_cpu::reduction::sum`（`CpuBackendOps::sum`）と
/// bit 完全一致することを実機で検証する（`crate::reduce_model` doc
/// 「CPU 参照実装との演算順序の対応」参照。`dim=None`／`Some(axis)`
/// 〈rank 1〜4・複数軸位置〉・非 contiguous 入力〈`transpose` view〉・
/// 0 サイズ契約〈空テンソル・`shape[axis]==0`・非縮約軸 0・巨大な非零軸
/// を含む 0 サイズ shape が `ShapeMismatch` にならないこと〉・範囲外
/// `dim`〈デバイス非接触で `ShapeMismatch`〉・NaN 混入〈クラス一致〉・
/// run-to-run 決定性を対象とする。`reduce_parity.rs`（起動 API 直叩き）
/// の `BackendOps` 経由版）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backend_ops_sum_matches_cpu_bit_exact() {
    let metal = MetalBackendOps::new();
    let cpu = CpuBackendOps::new();

    let assert_bit_exact = |metal_t: &Tensor<f32>, cpu_t: &Tensor<f32>, label: &str| {
        assert_eq!(metal_t.shape(), cpu_t.shape(), "{label}: shape 不一致");
        let m = metal_t.as_slice().expect("metal sum 出力は contiguous");
        let c = cpu_t.as_slice().expect("cpu sum 出力は contiguous");
        assert_eq!(m.len(), c.len(), "{label}: 要素数不一致");
        for (i, (&mv, &cv)) in m.iter().zip(c.iter()).enumerate() {
            if mv.is_nan() && cv.is_nan() {
                continue;
            }
            assert_eq!(
                mv.to_bits(),
                cv.to_bits(),
                "{label}[{i}]: bit 不一致（metal={mv:?}, cpu={cv:?}）"
            );
        }
    };

    let mut rng = Xorshift64Star::new(0x1896_0000);
    let gen_vec = |n: usize, rng: &mut Xorshift64Star| -> Vec<f32> {
        (0..n).map(|_| rng.next_f32() * 1024.0 - 512.0).collect()
    };

    // dim=None・複数 rank の dim=Some(axis)。
    let cases: &[(&[usize], Option<usize>)] = &[
        (&[7], None),
        (&[7], Some(0)),
        (&[3, 4], Some(0)),
        (&[3, 4], Some(1)),
        (&[2, 3, 5], Some(1)),
        (&[2, 3, 4, 2], Some(2)),
    ];
    for &(shape, dim) in cases {
        let numel: usize = shape.iter().product();
        let data = gen_vec(numel, &mut rng);
        let a = Tensor::new(data, shape).expect("tensor");
        let m = metal.sum(&a, dim).expect("metal sum must succeed");
        let c = cpu.sum(&a, dim).expect("cpu sum always succeeds");
        assert_bit_exact(&m, &c, &format!("shape={shape:?} dim={dim:?}"));
    }

    // 非 contiguous 入力（transpose view）。
    {
        let data = gen_vec(12, &mut rng);
        let a = Tensor::new(data, &[3, 4]).expect("tensor");
        let a_t = a.transpose(0, 1).expect("transpose");
        let m = metal
            .sum(&a_t, Some(1))
            .expect("metal sum on transposed view");
        let c = cpu.sum(&a_t, Some(1)).expect("cpu sum on transposed view");
        assert_bit_exact(&m, &c, "transpose view");
    }

    // 0 サイズ契約: 空テンソル（dim=None → 0.0）。
    {
        let a = Tensor::<f32>::new(Vec::new(), &[0]).expect("empty tensor");
        let m = metal.sum(&a, None).expect("metal sum on empty");
        let c = cpu.sum(&a, None).expect("cpu sum on empty");
        assert_bit_exact(&m, &c, "empty numel=0 dim=None");
    }

    // 0 サイズ契約: shape[axis]==0（空縮約 → 各出力 0.0）。
    {
        let a = Tensor::<f32>::new(Vec::new(), &[0, 3]).expect("tensor");
        let m = metal.sum(&a, Some(0)).expect("metal sum shape[axis]=0");
        let c = cpu.sum(&a, Some(0)).expect("cpu sum shape[axis]=0");
        assert_bit_exact(&m, &c, "shape[axis]=0");
    }

    // 0 サイズ契約: 非縮約軸 0（空出力）。
    {
        let a = Tensor::<f32>::new(Vec::new(), &[0, 3, 5]).expect("tensor");
        let m = metal.sum(&a, Some(1)).expect("metal sum empty output");
        let c = cpu.sum(&a, Some(1)).expect("cpu sum empty output");
        assert_bit_exact(&m, &c, "empty output (outer=0)");
    }

    // 0 サイズ契約: 巨大な非零軸を含む 0 サイズ shape が ShapeMismatch
    // にならず CPU と同じ結果になること（`checked_numel` より前に
    // 0 サイズ早期リターンを行う検査順序の直接検証。`ops.rs::sum`
    // doc「検査順序」参照）。
    {
        let huge = 1usize << 40;
        let a = Tensor::<f32>::new(Vec::new(), &[huge, 0, huge]).expect("huge zero-sized tensor");
        let m = metal
            .sum(&a, None)
            .expect("metal sum on huge zero-sized shape must not overflow-reject");
        let c = cpu.sum(&a, None).expect("cpu sum on huge zero-sized shape");
        assert_bit_exact(&m, &c, "huge zero-sized shape dim=None");

        let m_axis = metal
            .sum(&a, Some(1))
            .expect("metal sum(dim=1) on huge zero-sized shape");
        let c_axis = cpu
            .sum(&a, Some(1))
            .expect("cpu sum(dim=1) on huge zero-sized shape");
        assert_bit_exact(&m_axis, &c_axis, "huge zero-sized shape dim=Some(1)");
    }

    // 範囲外 dim: デバイス非接触で ShapeMismatch。
    {
        let a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor");
        assert!(matches!(
            metal.sum(&a, Some(5)),
            Err(BackendError::ShapeMismatch(_))
        ));
    }

    // NaN 混入（クラス一致）。
    {
        let a = Tensor::new(vec![1.0f32, f32::NAN, 3.0, 4.0], &[4]).expect("tensor");
        let m = metal.sum(&a, None).expect("metal sum with nan");
        let c = cpu.sum(&a, None).expect("cpu sum with nan");
        assert_bit_exact(&m, &c, "nan propagation");
    }

    // run-to-run 決定性。
    {
        let data = gen_vec(64, &mut rng);
        let a = Tensor::new(data, &[64]).expect("tensor");
        let m1 = metal.sum(&a, None).expect("metal sum run1");
        let m2 = metal.sum(&a, None).expect("metal sum run2");
        assert_bit_exact(&m1, &m2, "run-to-run determinism");
    }
}

/// `add`／`mul`／`relu`／`exp`／`tanh`（`elementwise::MetalElementwise`
/// 経由。イシュー #605）が CPU 参照実装と数値一致することを実機で固定
/// する（複合判定は REQ-2・`.claude/rules/coding-rust.md`。他の
/// `#[ignore]` 実機依存テストと同方針）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backend_ops_elementwise_matches_cpu() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let a = Tensor::new(vec![1.0, -2.0, 3.0, -4.0, 0.5, -0.25], &[2, 3]).expect("valid tensor");
    let b = Tensor::new(vec![2.0, 0.5, -1.0, 4.0, -3.0, 1.5], &[2, 3]).expect("valid tensor");

    let cpu_add = cpu.add(&a, &b).expect("cpu add");
    let metal_add = metal.add(&a, &b).expect("metal add");
    assert_tensor_parity("BackendOps cpu-metal add parity", &cpu_add, &metal_add);

    let cpu_mul = cpu.mul(&a, &b).expect("cpu mul");
    let metal_mul = metal.mul(&a, &b).expect("metal mul");
    assert_tensor_parity("BackendOps cpu-metal mul parity", &cpu_mul, &metal_mul);

    let cpu_relu = cpu.relu(&a).expect("cpu relu");
    let metal_relu = metal.relu(&a).expect("metal relu");
    assert_tensor_parity("BackendOps cpu-metal relu parity", &cpu_relu, &metal_relu);

    let cpu_exp = cpu.exp(&a).expect("cpu exp");
    let metal_exp = metal.exp(&a).expect("metal exp");
    assert_tensor_parity("BackendOps cpu-metal exp parity", &cpu_exp, &metal_exp);

    let cpu_tanh = cpu.tanh(&a).expect("cpu tanh");
    let metal_tanh = metal.tanh(&a).expect("metal tanh");
    assert_tensor_parity("BackendOps cpu-metal tanh parity", &cpu_tanh, &metal_tanh);
}

/// [`backend_ops_elementwise_matches_cpu`] 用の複合判定ヘルパー（REQ-2:
/// 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。`fandhe_ai_backend_cpu::parity`
/// を唯一の参照とする方針（`.claude/rules/coding-rust.md`）に合わせ、
/// 判定式は再定義せず [`assert_backend_ops_gemm_parity`] と同じ
/// `fandhe_ai_backend_cpu::parity::assert_parity` へ委譲する。
fn assert_tensor_parity(label: &str, cpu: &Tensor<f32>, metal: &Tensor<f32>) {
    assert_eq!(cpu.shape(), metal.shape());
    let cpu_owned = cpu.contiguous();
    let metal_owned = metal.contiguous();
    fandhe_ai_backend_cpu::parity::assert_parity(
        label,
        metal_owned.as_slice().expect("metal contiguous"),
        cpu_owned.as_slice().expect("cpu contiguous"),
    );
}

/// `ops_for`（`fandhe_ai_tensor_core::backend_ops`）を介したディスパッチでも同じ
/// 数値一致が成立することを固定する（`Device::Metal` 選択の回帰保護）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn ops_for_selects_metal_backend_and_matches_cpu() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();
    let ops: Vec<&dyn BackendOps> = vec![&cpu, &metal];

    let a_data = Xorshift64Star::new(513).fill_vec(80 * 80);
    let b_data = Xorshift64Star::new(514).fill_vec(80 * 80);
    let a = Tensor::new(a_data, &[80, 80]).expect("valid tensor");
    let b = Tensor::new(b_data, &[80, 80]).expect("valid tensor");

    let cpu_result = fandhe_ai_tensor_core::ops_for(&ops, Device::Cpu)
        .expect("cpu ops registered")
        .gemm(&a, &b)
        .expect("cpu gemm always succeeds");
    let metal_result = fandhe_ai_tensor_core::ops_for(&ops, Device::Metal)
        .expect("metal ops registered")
        .gemm(&a, &b)
        .expect("MetalBackendOps::gemm must succeed on Metal-equipped test runner");

    fandhe_ai_backend_cpu::parity::assert_parity(
        "ops_for-dispatched metal gemm vs cpu",
        metal_result.as_slice().expect("contiguous"),
        cpu_result.as_slice().expect("contiguous"),
    );
}
