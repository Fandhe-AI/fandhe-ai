//! イシュー #1326 調査: Metal 4 `tensor<>`＋Metal Performance Primitives
//! （MPP。`MetalPerformancePrimitives.framework` の `matmul2d`）の
//! 可用性・`objc2-metal =0.3.2` バインディングからの到達性・純カーネル
//! 時間（GPU タイムスタンプ）を実測し、「完全自作コア」（REQ-1）との
//! 整合をユーザーが判断するための材料を整備する診断テスト群。
//!
//! # 位置づけ（調査限定。本番結線なし）
//!
//! 本ファイルは `tile.rs`／`gemm.rs`／`shaders/gemm.metal`／
//! `Cargo.toml`／`Cargo.lock` への変更を一切含まない（新規診断テスト
//! ファイルの追加のみ）。MPP はシェーダ側 vendor primitive（Cargo 依存
//! ではなく Metal SDK フレームワークヘッダの `#include`）であり、これを
//! 自作コアの範囲内と見なすかは `docs/backend-metal-mpp-tensor-decision.md`
//! §6 でユーザー判断事項として整理する。結論・採否は本ファイルでは
//! 出さない。
//!
//! # 起動経路（計画の Route C。`docs/backend-metal-mpp-tensor-decision.md`
//! §2 参照）
//!
//! `MTLComputeCommandEncoder`（classic encoder。既存 `MetalContext::
//! encode` 経路）には tensor を直接バインドする API が存在しない
//! （SDK 26 系 `MTLComputeCommandEncoder.h` 実測）。そのためカーネルは
//! `device const float*`／`device float*`（通常のバッファ引数）を受け取り、
//! カーネル本体内で `tensor<device float, dextents<int32_t,2>,
//! tensor_inline>(ptr, extents)` を構築してから `mpp::tensor_ops::
//! matmul2d<Descriptor, execution_simdgroups<4>>::run` を呼ぶ。ホスト側は
//! 既存の `setBuffer_offset_atIndex`／`setBytes_length_atIndex`／
//! `dispatchThreadgroups_threadsPerThreadgroup` のみで完結し、
//! `MTLTensor`／`MTL4*`（Route A'／B。到達性のみ doc に記録し本ファイルは
//! 実装しない）は不要。
//!
//! # 正方行列限定（N=M=K）
//!
//! 本調査は純カーネル時間比較が目的のため、`gen_square_ab` が生成する
//! N×N 正方行列のみを対象とする。extents 順序（`extent(0)` が連続
//! 次元。ヘッダ注記「NN では K = A.extents().extent(0) = B.extents().
//! extent(1)」）は非正方形状で取り違えやすいが、正方形状では
//! `dextents<int32_t,2>(n, n)` のみで済み、この論点を回避できる
//! （非正方・転置パターンは本イシューのスコープ外。§8 参照）。
//!
//! # MSL ソースの配置（`src/shaders/` に置かない理由）
//!
//! `src/shaders/*.metal` は本番 `include_str!` 対象・`shader_source_
//! evidence` テストの走査対象のため、本調査専用のソースをそこへ
//! 置くと本番シェーダ集合に誤って混入したように見える。本モジュール
//! 内の `const &str` として保持し、Apple ヘッダ本文はリポジトリへ
//! 複製しない（システムパスの `#include` のみで参照する）。
//!
//! # 配置理由（既存診断テスト群と同じ判断）
//!
//! `crate::context::MetalContext`・`crate::buffer::MetalBuffer`・
//! `crate::pipeline::{compile_options, make_pipeline}`（いずれも
//! `pub(crate)`）・`crate::gemm_reuse_phase_diag_tests::{gen_square_ab,
//! median_of}`（`pub(crate)`）へ到達するため、integration test では
//! なく `lib.rs` の兄弟モジュールとして配置する。`objc2` 系 FFI 型に
//! 触れるため `cfg(all(test, target_os = "macos"))` を付ける。
//!
//! # 実行時は必ず `--test-threads=1`
//!
//! `MetalContext::synchronize_with_gpu_timestamps` はプロセスワイドの
//! 完了バッチ数を検証するため、GPU 上での複数テストスレッド競合を
//! 避ける必要がある（既存診断テストと同じ理由）。
//!
//! # unsafe の範囲
//!
//! 新規 `unsafe` は本ファイル内（`#[cfg(test)]` 限定）に閉じ、
//! `encode_dispatch_tiled`（`gemm.rs`）と同一形の FFI 呼び出し
//! （`setBuffer_offset_atIndex`／`setBytes_length_atIndex`）のみを
//! 追加する。本番コードへの unsafe 追加はない。

use objc2::rc::Retained;
use objc2_metal::{
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLGPUFamily, MTLLanguageVersion,
    MTLSize,
};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::gemm_reuse_phase_diag_tests::{gen_square_ab, median_of};
use crate::pipeline::{MtlLibrary, MtlPipeline};

/// MPP `matmul2d`（NN・f32・`relaxed_precision=false`）の Route C 実装。
/// `M`/`N`/`K` は正方形状前提のため単一の `constant uint&`（buffer(3)）
/// で渡す。`matmul2d_descriptor` のタイル（64×32）・`execution_
/// simdgroups<4>` は `MetalPerformancePrimitives.framework/Headers/
/// MPPTensorOpsMatMul2d.h` 冒頭の基本例をそのまま採用する（境界検査
/// 込みの `slice()` 版。`coding-rust.md`「境界検査を省略しない」を
/// 満たす — `matmul2d::run` 自体が呼び出し側 tensor の extents に
/// 対して端タイルの境界検査を行う契約。ヘッダ冒頭コメント参照）。
///
/// **`tgid.x`/`tgid.y` の割当はヘッダ冒頭のコメント例（`A.slice(0,
/// tgid.y*64)`／`B.slice(tgid.x*32, 0)`）をそのまま採用すると M4 Max
/// 実機で出力の約半数が不一致になることを実測で確認した（イシュー
/// #1326 実装時。`docs/backend-metal-mpp-tensor-decision.md` §2.3）。
/// 同ヘッダの `if (tgid.x*64 + 63 < M && tgid.y*32 + 31 < N)` 境界検査
/// コメント（`tgid.x` が M タイル・`tgid.y` が N タイルを指す）と
/// dispatch grid（`MTLSizeMake((M+63)/64, (N+31)/32, 1)`。width=M
/// タイル数）は整合するため、**冒頭のコメント例の `tgid.x`/`tgid.y` は
/// 誤記（非コンパイル対象のコメントであるため実測で検出されなかった
/// 可能性が高い）と判断し、境界検査コメント・dispatch grid と整合する
/// 向き（`tgid.x`→M タイル・`tgid.y`→N タイル）へ入れ替えて実装した**。
/// 本ファイルの `mpp_matches_cpu_reference`（正確性スモーク）で入れ替え
/// 後の実装が REQ-2 複合判定に pass することを確認済み。
const MPP_GEMM_NN_F32_SRC: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void mpp_gemm_nn_f32(
    device float* A_ptr [[buffer(0)]],
    device float* B_ptr [[buffer(1)]],
    device float* C_ptr [[buffer(2)]],
    constant uint& N_DIM [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const int32_t n = static_cast<int32_t>(N_DIM);
    tensor<device float, dextents<int32_t, 2>, tensor_inline> A(A_ptr, dextents<int32_t, 2>(n, n));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> B(B_ptr, dextents<int32_t, 2>(n, n));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> C(C_ptr, dextents<int32_t, 2>(n, n));

    constexpr auto matmulDescriptor =
        matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, false, false);
    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;

    auto mA = A.slice(0, tgid.x * 64);
    auto mB = B.slice(tgid.y * 32, 0);
    auto mC = C.slice(tgid.y * 32, tgid.x * 64);

    matmulOp.run(mA, mB, mC);
}
"#;

/// コンパイル可否プローブ専用の最小ソース（(1) の判定を `mpp_gemm_
/// nn_f32` 本体の他要因〈カーネル記述ミス〉から切り離すため、
/// ヘッダの `#include` とディスクリプタ構築のみを検査する）。
const MPP_COMPILE_PROBE_SRC: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void mpp_compile_probe(device float* dummy [[buffer(0)]]) {
    constexpr auto d = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, false, false);
    (void)d;
    dummy[0] = 0.0f;
}
"#;

/// [`crate::pipeline::compile_options`]（`MathMode::Safe` +
/// `MathFloatingPointFunctions::Precise`。既存本番と同一の丸め方針）に
/// `setLanguageVersion(Version4_0)` を追加した `MTLCompileOptions` で
/// `src` をコンパイルする（MPP ヘッダは `__HAVE_TENSOR__` ガード付きで
/// Metal 4.0 言語版を要求するため）。`pipeline::compile_source` は
/// private のためここでは複製せず、`compile_options()` を再利用しつつ
/// 言語版だけをオーバーライドする（本番 `compile_gemm_library` 経路は
/// 無変更）。
fn compile_mpp_source(
    device: &crate::context::MtlDevice,
    src: &str,
) -> Result<Retained<MtlLibrary>, MetalError> {
    use objc2_foundation::NSString;

    let options = crate::pipeline::compile_options();
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    let ns_src = NSString::from_str(src);
    device
        .newLibraryWithSource_options_error(&ns_src, Some(&options))
        .map_err(|err| MetalError::LibraryCompilation {
            message: err.localizedDescription().to_string(),
        })
}

/// (1) コンパイル可否スモーク: `supportsFamily(Metal4)`・
/// `architecture().name()` をログへ残したうえで、最小 MPP ソースを
/// ランタイムコンパイルできるかを記録する。**gating しない**（成否
/// どちらであっても panic させず結果を `println!` する。可否自体が
/// 調査対象のデータであり、失敗を異常系として扱わない — ファイル
/// 冒頭「位置づけ」参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn mpp_metal4_compile_probe() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let device = ctx.device();

    let supports_metal4 = device.supportsFamily(MTLGPUFamily::Metal4);
    let arch_name = device.architecture().name().to_string();
    println!("mpp_probe device_architecture={arch_name} supports_metal4={supports_metal4}");

    match compile_mpp_source(device, MPP_COMPILE_PROBE_SRC) {
        Ok(_lib) => {
            println!("mpp_probe compile_result=ok");
        }
        Err(e) => {
            println!("mpp_probe compile_result=error message={e:?}");
        }
    }
}

/// [`mpp_matches_cpu_reference`] が対象とする小形状（タイル端の非倍数
/// 形状含む。K ストレスは正方形状のため N=K で兼ねる）。
const PARITY_SIZES: [usize; 3] = [8, 64, 100];

/// (4) 正確性スモーク: MPP `matmul2d` 出力を CPU 参照実装
/// （`fandhe_ai_backend_cpu::parity::matmul_reference_fma`）と REQ-2
/// 複合判定で比較する。**extent 順序の取り違えは有限だが誤った値を
/// 生むため、本テストが pass することを (3) のベンチより先に確認する
/// 契約**（ファイル冒頭「配置理由」参照）。コンパイルが失敗する環境
/// （[`mpp_metal4_compile_probe`] が `compile_result=error` を報告する
/// 環境）では本テストも同じ理由で失敗しうる — その場合は「可用性＝
/// 不可」として doc に記録し、本テストの失敗を以て#1326 (4) は未実施
/// と扱う（実装計画 §6「不成立時の扱い」）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn mpp_matches_cpu_reference() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let device = ctx.device();
    let library = compile_mpp_source(device, MPP_GEMM_NN_F32_SRC)
        .expect("MPP GEMM ソースのコンパイルに失敗した（mpp_metal4_compile_probe の結果を先に確認すること）");
    let pipeline = crate::pipeline::make_pipeline(device, &library, "mpp_gemm_nn_f32")
        .expect("mpp_gemm_nn_f32 パイプラインの構築に失敗した");

    for n in PARITY_SIZES {
        let (a, b) = gen_square_ab(0x1326_5000 ^ (n as u64), n);
        let out = run_mpp_gemm_nn(&ctx, &pipeline, &a, &b, n);

        let mut expected = vec![0.0f32; n * n];
        fandhe_ai_backend_cpu::parity::matmul_reference_fma(&a, &b, &mut expected, n, n, n)
            .expect("CPU 参照実装 matmul_reference_fma への入力は形状整合済み");

        let report = fandhe_ai_backend_cpu::parity::compare(&expected, &out)
            .expect("要素数一致・NaN 非混入の入力に対して compare は常に Ok を返す");
        assert!(
            report.passes(),
            "N={n}: MPP matmul2d の出力が CPU 参照実装との複合判定（相対誤差 \
             1e-3 未満 または 絶対誤差 1e-5 未満）に pass しない（report={report:?}）。 \
             extent 順序の取り違え、または C 事前ゼロ化漏れの可能性がある"
        );
        println!(
            "mpp_parity N={n} pass=true mean_abs_diff={:.6e}",
            report.mean_abs_diff
        );
    }
}

/// [`run_mpp_gemm_nn`] の 1 回のディスパッチを実行し、出力を読み戻す
/// （タイミング計測なし版。[`mpp_matches_cpu_reference`] が使う）。
/// C は MPP の `run`（`C = A*B + C`）契約上、呼び出し前にゼロ初期化
/// されている必要がある（`MPPTensorOpsMatMul2d.h` 冒頭例「Assumes C is
/// initialized to zero」）ため [`MetalBuffer::new_zeroed`] を使う
/// （プールなし。本調査は正確性検証のみで性能非対象のためプール経由
/// 確保に揃える必要はない）。
fn run_mpp_gemm_nn(
    ctx: &MetalContext,
    pipeline: &MtlPipeline,
    a: &[f32],
    b: &[f32],
    n: usize,
) -> Vec<f32> {
    let a_buf = MetalBuffer::new_with_data(ctx, a).expect("A upload must succeed");
    let b_buf = MetalBuffer::new_with_data(ctx, b).expect("B upload must succeed");
    let c_buf = MetalBuffer::new_zeroed(ctx, n * n).expect("C zeroed allocation must succeed");

    encode_mpp_nn(ctx, pipeline, &a_buf, &b_buf, &c_buf, n);

    ctx.synchronize_with_gpu_timestamps()
        .expect("synchronize must succeed");
    c_buf.read_to_vec()
}

/// MPP GEMM の 1 回のディスパッチを記録する（`gemm.rs::encode_
/// dispatch_tiled` と同一形の FFI 呼び出し。本ファイル冒頭「unsafe の
/// 範囲」参照）。`ctx.encode` を介するため、他の診断テストと同じく
/// 記録のみ・待たない契約（呼び出し元が `synchronize` する）。
fn encode_mpp_nn(
    ctx: &MetalContext,
    pipeline: &MtlPipeline,
    a_buf: &MetalBuffer,
    b_buf: &MetalBuffer,
    c_buf: &MetalBuffer,
    n: usize,
) {
    let n_u32 = n as u32;
    let thread_execution_width = pipeline.threadExecutionWidth();
    ctx.encode(
        "mpp_gemm_nn_f32",
        &[a_buf.raw(), b_buf.raw(), c_buf.raw()],
        None,
        |encoder| {
            encoder.setComputePipelineState(pipeline);
            // SAFETY: `encode_dispatch_tiled`（`gemm.rs`）と同一の契約。
            // `a_buf`/`b_buf`/`c_buf` は `synchronize` 完了まで呼び出し元
            // スタックフレームで生存する（`ctx.encode` が `resources` を
            // 通じて `in_flight` へ retain する）。
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(a_buf.raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(b_buf.raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(c_buf.raw()), 0, 2);
            }
            // SAFETY: `n_u32` はローカル変数で `setBytes` 呼び出しの間
            // 生存し、渡す長さ (`size_of::<u32>()`) はポインタ先の型と
            // 一致する（`encode_dispatch_tiled` の `Dims`/`GemmStrides`
            // `setBytes` と同型の契約）。
            unsafe {
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&n_u32).cast(),
                    std::mem::size_of::<u32>(),
                    3,
                );
            }

            let threadgroups = MTLSize {
                width: n.div_ceil(64),
                height: n.div_ceil(32),
                depth: 1,
            };
            let threads_per_tg = MTLSize {
                width: thread_execution_width * 4,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        },
    )
    .expect("mpp_gemm_nn_f32 dispatch must succeed");
}

/// [`mpp_kernel_gpu_ab_vs_production_select`] が対象とするサイズ
/// （実装計画 §3.3。512 は非後退情報として追加してよいとされるが、本
/// 調査では主対象の 3 点に限定する）。
const KERNEL_GPU_AB_SIZES: [usize; 3] = [1024, 2048, 4096];

const WARMUP_TRIALS: usize = 20;
const MEASURED_TRIALS: usize = 20;

/// 1 回の MPP dispatch を計測し `kernel_gpu_secs` を返す（`resolved_
/// cfg` の概念が存在しないため `PhaseSample` は再利用せず薄い専用
/// ヘルパとする。fail-closed 検証は `gemm_reuse_phase_diag_tests::
/// measure_one_phase_trial` と同じ内容をここに複製する）。
fn measure_mpp_trial(
    ctx: &MetalContext,
    pipeline: &MtlPipeline,
    a: &[f32],
    b: &[f32],
    n: usize,
    keep_alive: &mut Vec<Vec<f32>>,
) -> f64 {
    let a_buf = MetalBuffer::new_with_data(ctx, a).expect("A upload must succeed");
    let b_buf = MetalBuffer::new_with_data(ctx, b).expect("B upload must succeed");
    let c_buf = MetalBuffer::new_zeroed(ctx, n * n).expect("C zeroed allocation must succeed");

    encode_mpp_nn(ctx, pipeline, &a_buf, &b_buf, &c_buf, n);

    let batches = ctx
        .synchronize_with_gpu_timestamps()
        .expect("synchronize (commit + waitUntilCompleted) must succeed");
    assert_eq!(
        batches.len(),
        1,
        "synchronize_with_gpu_timestamps must complete exactly one batch per encode_mpp_nn call \
         (got {}; run with --test-threads=1)",
        batches.len()
    );
    let batch = &batches[0];
    assert_eq!(
        batch.labels(),
        ["mpp_gemm_nn_f32"],
        "unexpected dispatch labels in the completed batch: {:?}",
        batch.labels()
    );
    let kernel_gpu_secs = batch.kernel_gpu_secs().unwrap_or_else(|| {
        panic!(
            "GPUStartTime/GPUEndTime must both be non-zero for a completed batch (labels={:?})",
            batch.labels()
        )
    });
    assert!(
        kernel_gpu_secs >= 0.0,
        "kernel_gpu_secs must be non-negative: {kernel_gpu_secs}"
    );

    let out = c_buf.read_to_vec();
    keep_alive.push(out);
    kernel_gpu_secs
}

/// 本番選択構成（`tile::select_for_device` が実際に選ぶ `gemm_
/// simdgroup_tiled` 構成。`gemm::MetalGemm::diag_encode_tiled_nn`
/// 経由）の 1 回の kernel_gpu を計測する（E7/E8 `run_ab_pair` の base
/// arm 相当を MPP 診断内で複製したもの。`gemm_bk32_diag_tests::
/// run_ab_pair` は `pair_label` 前提の出力形式が異なるため、MPP との
/// 比較に適した形式で本ファイル専用に実装する）。
fn measure_production_trial(
    ctx: &MetalContext,
    gemm: &crate::gemm::MetalGemm,
    a: &[f32],
    b: &[f32],
    n: usize,
    cfg: crate::tile::TileConfig,
    keep_alive: &mut Vec<Vec<f32>>,
) -> (f64, crate::tile::TileConfig) {
    let a_buf = MetalBuffer::new_with_data(ctx, a).expect("A upload must succeed");
    let b_buf = MetalBuffer::new_with_data(ctx, b).expect("B upload must succeed");
    let c_buf =
        MetalBuffer::alloc_uninit_pooled(ctx, n * n).expect("pooled C allocation must succeed");

    let resolved_cfg = gemm
        .diag_encode_tiled_nn(ctx, &a_buf, &b_buf, &c_buf, n, n, n, cfg)
        .expect("diag_encode_tiled_nn must succeed");

    let batches = ctx
        .synchronize_with_gpu_timestamps()
        .expect("synchronize must succeed");
    assert_eq!(
        batches.len(),
        1,
        "synchronize_with_gpu_timestamps must complete exactly one batch per diag_encode_tiled_nn call \
         (got {}; run with --test-threads=1)",
        batches.len()
    );
    let batch = &batches[0];
    let kernel_gpu_secs = batch.kernel_gpu_secs().unwrap_or_else(|| {
        panic!(
            "GPUStartTime/GPUEndTime must both be non-zero for a completed batch (labels={:?})",
            batch.labels()
        )
    });

    let out = c_buf.read_to_vec();
    keep_alive.push(out);
    (kernel_gpu_secs, resolved_cfg)
}

/// (3) 純カーネル時間 A/B（head=MPP `matmul2d` Route C・base=本番選択
/// 構成 `tile::select_for_device`）。E7/E8 と同じ交互測定・trial index
/// による開始 arm 回転で order-bias を相殺し、trial 0 の出力を REQ-2
/// 複合判定で比較する（**bit 完全一致は要求しない** — MPP は別カーネル
/// 実装のため加算順序が異なりうる。`coding-rust.md`「バックエンド間
/// 数値一致は統一複合判定」を単一バックエンド内の異実装比較にも適用
/// する）。`assert!` による優劣判定（gating）は行わない（有効性判断は
/// `docs/backend-metal-mpp-tensor-decision.md` §3 で人間が行う）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn mpp_kernel_gpu_ab_vs_production_select() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let device = ctx.device();
    let library = compile_mpp_source(device, MPP_GEMM_NN_F32_SRC)
        .expect("MPP GEMM ソースのコンパイルに失敗した（mpp_metal4_compile_probe の結果を先に確認すること）");
    let mpp_pipeline = crate::pipeline::make_pipeline(device, &library, "mpp_gemm_nn_f32")
        .expect("mpp_gemm_nn_f32 パイプラインの構築に失敗した");
    let gemm = crate::gemm::MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    for n in KERNEL_GPU_AB_SIZES {
        let (a, b) = gen_square_ab(0x1326_a000 ^ (n as u64), n);
        let base_cfg =
            crate::tile::select_for_device(n, n, n, ctx.verified_m4_max_gpu_core_count());
        println!("N={n} pair=mpp_vs_production_select production_select_resolved={base_cfg:?}");

        let mut keep_alive_base: Vec<Vec<f32>> =
            Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);
        let mut keep_alive_head: Vec<Vec<f32>> =
            Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);

        for _ in 0..WARMUP_TRIALS {
            let _ =
                measure_production_trial(&ctx, &gemm, &a, &b, n, base_cfg, &mut keep_alive_base);
            let _ = measure_mpp_trial(&ctx, &mpp_pipeline, &a, &b, n, &mut keep_alive_head);
        }

        let mut kernel_gpu_base: Vec<f64> = Vec::with_capacity(MEASURED_TRIALS);
        let mut kernel_gpu_head: Vec<f64> = Vec::with_capacity(MEASURED_TRIALS);
        let mut first_output_base: Option<Vec<f32>> = None;
        let mut first_output_head: Option<Vec<f32>> = None;

        for trial in 0..MEASURED_TRIALS {
            let base_first = trial % 2 == 0;
            let (sample_base, sample_head) = if base_first {
                let sb = measure_production_trial(
                    &ctx,
                    &gemm,
                    &a,
                    &b,
                    n,
                    base_cfg,
                    &mut keep_alive_base,
                );
                let sh = measure_mpp_trial(&ctx, &mpp_pipeline, &a, &b, n, &mut keep_alive_head);
                (sb, sh)
            } else {
                let sh = measure_mpp_trial(&ctx, &mpp_pipeline, &a, &b, n, &mut keep_alive_head);
                let sb = measure_production_trial(
                    &ctx,
                    &gemm,
                    &a,
                    &b,
                    n,
                    base_cfg,
                    &mut keep_alive_base,
                );
                (sb, sh)
            };

            assert_eq!(
                sample_base.1, base_cfg,
                "N={n} trial={trial}: production select がフォールバックした \
                 (requested={base_cfg:?}, resolved={:?})。性能比較の前提が \
                 崩れるため中断する",
                sample_base.1
            );

            kernel_gpu_base.push(sample_base.0);
            kernel_gpu_head.push(sample_head);

            if trial == 0 {
                first_output_base = keep_alive_base.last().cloned();
                first_output_head = keep_alive_head.last().cloned();
            }
        }

        let out_base = first_output_base.expect("trial 0 の base 出力は必ず Some");
        let out_head = first_output_head.expect("trial 0 の head 出力は必ず Some");
        let report = fandhe_ai_backend_cpu::parity::compare(&out_base, &out_head)
            .expect("parity::compare は要素数一致・NaN 非混入の入力に対して常に Ok を返す");
        assert!(
            report.passes(),
            "N={n}: production select / MPP matmul2d の出力が複合判定（相対誤差 \
             1e-3 未満 または 絶対誤差 1e-5 未満）に pass しない（report={report:?}）"
        );

        let q_base = median_of(&kernel_gpu_base);
        let q_head = median_of(&kernel_gpu_head);
        println!(
            "N={n} pair=mpp_vs_production_select mode=base resolved_tile={base_cfg:?} \
             kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
            q_base.median * 1e3,
            q_base.q1 * 1e3,
            q_base.q3 * 1e3
        );
        println!(
            "N={n} pair=mpp_vs_production_select mode=head kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
            q_head.median * 1e3,
            q_head.q1 * 1e3,
            q_head.q3 * 1e3
        );
        println!(
            "N={n} pair=mpp_vs_production_select head_over_base_kernel_gpu={:.4}",
            q_head.median / q_base.median
        );

        // N ごとのループスコープで両 arm の keep_alive を drop してから
        // 次の N へ進み、ホスト側メモリのピークを 1 N 分に抑える（E7/E8
        // と同じ設計）。
        drop(keep_alive_base);
        drop(keep_alive_head);
    }
}
