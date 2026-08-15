//! NVRTC で `cp.async.bulk.tensor`（TMA）が sm_121 で生成・実行可能かを
//! 実機で確認する spike プローブ（#483・親 #479 Phase A・A-3 #480）。
//!
//! ## 位置づけ
//!
//! GEMM 性能改善トラッキング（#479／#480）の Phase B（TMA 前提の最適化
//! タスク群 B-12〜B-14）を起票してよいかを、プロダクションコードに触れる
//! 前に確定するための調査専用テストである。本ファイルはカーネル実装
//! （`kernels_mma.rs` 等）を追加するものではなく、TMA 命令列が
//! **コンパイルできるか**（[`tma_nvrtc_compile_probe`]）・
//! **実行して転送が成立するか**（[`tma_execution_probe`]・
//! [`tma_execution_probe_cta`]）の 2 点のみを記録する。結果は
//! `docs/cuda-tensor-core-design.md` の「TMA（cp.async.bulk.tensor）
//! sm_121 プローブ」節へ転記し、B-12〜B-14 の起票要否を判断する材料に
//! する（起票自体は `out-of-scope-tracking.md` に従いユーザー承認のうえ
//! 別途行う）。
//!
//! ## CUTLASS 側の根拠・cluster / cta 両スコープを probe する理由
//!
//! CUTLASS では `CUTE_ARCH_TMA_SM120_ENABLED` が SM121（`"a"` サフィックス
//! 無し・`__CUDA_ARCH__ == 1210`）でも有効化される設計になっている
//! （`include/cute/arch/config.hpp:154-158`・`include/cutlass/arch/config.h:197-204`。
//! 2026-08 時点の CUTLASS ソース調査）。これは「sm_121（GB10）でも TMA が
//! 使える可能性がある」という状況証拠であり、本プローブが確認するのは
//! それを NVRTC 実行時コンパイル経由でも再現できるかという点である
//! （CUTLASS は nvcc オフラインコンパイルの CuTe C++ DSL を経由するため、
//! 本リポジトリの NVRTC 実行時コンパイル方式・生 PTX インラインアセンブリ
//! 方式とは経路が異なる）。
//!
//! ただし `CUTE_ARCH_TMA_SM120_ENABLED` パスは sm_120/sm_121 では
//! `shared::cluster`（Hopper 由来のクラスタスコープ TMA。
//! `cuLaunchKernelEx` + クラスタ次元指定が必須）ではなく
//! `shared::cta`（クラスタ launch 不要の CTA スコープ TMA）の opcode を
//! 発行する設計である。クラスタスコープの opcode のみを probe すると、
//! sm_121 で CTA スコープ TMA 自体は機能しているにもかかわらず
//! クラスタスコープ固有の要因（クラスタ launch のサポート状況等）だけで
//! 失敗し、「TMA は sm_121 で使えない」と誤って記録して B-12〜B-14 を
//! 不当にブロックしうる（イシュー #483 PR #634 cursor[bot] レビュー
//! 指摘）。そのため本ファイルは [`TMA_PROBE_KERNEL`]（`shared::cluster`）
//! と [`TMA_PROBE_KERNEL_CTA`]（`shared::cta`）の両方を独立に
//! コンパイル・実行 probe し、結果を variant 別に記録する。
//!
//! ## 実機前提・検証状態
//!
//! `tests/tensor_core_real_device.rs`・`kernels_mma.rs` と同じ規約に従う:
//! 両テストとも `#[ignore]` 分離（DGX Spark GB10 等 sm_121 実機必須）で、
//! CUDA デバイス・NVRTC が利用できない環境では `.expect` により失敗を
//! 顕在化させる（silent green を許さない）。本ファイルのカーネルソース・
//! `cuTensorMapEncodeTiled` 呼び出しパラメータは実機コンパイル・実行を
//! 一度も通過していない（`kernels_mma.rs` 冒頭コメント「検証状態」と同じ
//! 位置づけ。実機での最初の実行が構文・パラメータ検証を兼ねる）。
//!
//! ## 依存・A03 対応
//!
//! `cuTensorMapEncodeTiled`（`cudarc::driver::sys`）は cudarc 0.19.8 の
//! 既存 API であり新規依存の追加ではない（`.claude/rules/deps-policy.md`。
//! `cuda-13000` feature で既に有効）。カーネルソースはコンパイル時定数
//! （`&'static str`）のみとし外部入力を連結しない（`nvrtc.rs` の A03
//! 契約を踏襲。`.claude/rules/security.md`）。
//!
//! 接続情報・実ホスト名は本ファイルに一切記載しない
//! (`.claude/rules/security.md`・`docs/real-hardware-verification-env.md`)。

use std::ffi::c_void;

use cudarc::driver::sys::{
    self, CUlaunchAttribute, CUlaunchAttributeID, CUlaunchAttributeValue, CUlaunchConfig,
    CUtensorMap, CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapInterleave,
    CUtensorMapL2promotion, CUtensorMapSwizzle,
};
use cudarc::driver::{DevicePtr, DevicePtrMut};

use backend_cuda::CudaDevice;

/// [`tma_nvrtc_compile_probe`]・[`tma_execution_probe`]・
/// [`tma_execution_probe_cta`] が共通で試す arch の優先順位付きリスト
/// （#483 受け入れ基準の判定基準「`compute_121` または
/// `compute_121a`/`compute_121f` でコンパイル・実行成功」に対応）。
///
/// コンパイルプローブと実行プローブで別々に arch リストを持つと、
/// 一方だけ更新してもう一方が古いまま残るドリフト（PR #634 codex-review
/// 指摘: 実行プローブが `compute_121` 固定で代替 arch のみ成功する環境の
/// 実行可能性を検証できない）が再発しうるため、単一の const へ集約する。
const PROBE_ARCHS: [&str; 3] = ["compute_121", "compute_121a", "compute_121f"];

/// TMA プローブの最小 inline PTX カーネル（NVRTC 向け）。`shared::cluster`
/// スコープ版（[`TMA_PROBE_KERNEL_CTA`] は `shared::cta` スコープ版。
/// 本ファイル冒頭コメント「cluster / cta 両スコープを probe する理由」
/// 参照）。
///
/// `cuda_runtime.h`/`cuda.h` は NVRTC に同梱されないため（`nvrtc.rs`・
/// `kernels_mma.rs` と同じ制約）、`CUtensorMap` はホスト側
/// （`cudarc::driver::sys::CUtensorMap`。`opaque: [u64; 16]`・
/// `#[repr(align(128))]`。`cuda-13000` feature 下の実測値。cudarc-0.19.8
/// `src/driver/sys/mod.rs:7314-7320` で確認済み。`align(64)` 版は
/// `cuda-12xxx` feature 用の別定義であり本リポジトリでは使わない）と
/// バイトレイアウトが一致する手書き typedef で代替する。
///
/// 手順（1 スレッドが代表して実行し、残りは `__syncthreads()` で合流する
/// 一般的な TMA パターン）:
/// 1. `mbarrier.init.shared::cta.b64`（期待到着数 1）
/// 2. `fence.proxy.async.shared::cta`（init の可視化）
/// 3. `mbarrier.arrive.expect_tx.shared::cta.b64`（転送予定バイト数を予約）
/// 4. `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes`
///    （global の `(0,0)` タイル 1 枚を shared へ非同期転送）
/// 5. `mbarrier.try_wait.parity.shared::cta.b64` のポーリングループ
/// 6. `fence.proxy.async.shared::cta`（転送データの可視化）
/// 7. 全スレッドで `__syncthreads()` に合流後、shared→out の書き戻し
///    （REQ-8: `tid < TILE_ELEMS && tid < out_len` の手動境界チェック付き。
///    `.claude/rules/coding-rust.md`「カーネル実装の境界検査」）
///
/// `shared::cluster` 修飾子はクラスタ起動（`cuLaunchKernelEx` +
/// `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`）を要求するため、ホスト側
/// ([`tma_execution_probe`]) はクラスタ次元 `(1,1,1)` を明示指定する
/// （クラスタサイズ 1 でも `cuLaunchKernel`〈非 Ex 版〉ではなく
/// `cuLaunchKernelEx` 経由が必要という前提のもとでの選択。実機未検証）。
const TMA_PROBE_KERNEL: &str = r#"
// `cudarc::driver::sys::CUtensorMap`（`cuda-13000` feature）は
// `#[repr(align(128))]`（本ファイル冒頭コメント参照）。ここでのズレは
// `__grid_constant__ CUtensorMap tensor_map` パラメータ以降のカーネル
// 引数（`out`・`out_len`）の ABI オフセット計算を狂わせうるため、
// ホスト側の実際のアライメントと必ず一致させる。
typedef struct __align__(128) {
    unsigned long long opaque[16];
} CUtensorMap;

#define TILE_M 16
#define TILE_N 16
#define TILE_ELEMS (TILE_M * TILE_N)
#define TILE_BYTES (TILE_ELEMS * 4)

extern "C" __global__ void __launch_bounds__(256) tma_probe_2d(
    const __grid_constant__ CUtensorMap tensor_map,
    float* __restrict__ out,
    int out_len)
{
    __shared__ __align__(128) float smem_tile[TILE_ELEMS];
    __shared__ __align__(8) unsigned long long mbar;
    // mbarrier ポーリングが `TMA_POLL_LIMIT` 回で完了しなかった場合に
    // 全スレッドへ「転送未完了」を伝える合流フラグ（tid 0 が書き、
    // 全スレッドが __syncthreads() 後に読む）。TMA ディスクリプタ不正・
    // sm_121 での TMA 機能不全時にカーネルが無期限ハングするのを防ぐ
    // （プローブの目的は失敗の記録であり、無期限占有ではない）。
    __shared__ unsigned timed_out;

    int tid = threadIdx.x;

    if (tid == 0) {
        unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(mbar_addr));
        asm volatile("fence.proxy.async.shared::cta;\n");

        unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_tile);
        // `&tensor_map`（`__grid_constant__` パラメータ）は NVRTC/nvcc が
        // 生成する generic アドレス表現をそのまま `l` 制約へ渡す（公開
        // TMA サンプルの標準パターン。手動 `cvta.param` を挟むと二重変換に
        // なる可能性があるため挟まない。実機未検証 — 本ファイル冒頭
        // コメント「検証状態」参照）。
        unsigned long long map_addr = (unsigned long long)&tensor_map;

        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
            :: "r"(mbar_addr), "r"((int)TILE_BYTES));

        int coord_x = 0;
        int coord_y = 0;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];\n"
            :: "r"(smem_addr), "l"(map_addr), "r"(coord_x), "r"(coord_y), "r"(mbar_addr)
            : "memory");

        unsigned phase_parity = 0;
        unsigned complete = 0;
        // 上限回数（タイムアウト）付きポーリング。TMA ディスクリプタが
        // 不正、あるいは sm_121 で TMA が機能しない場合に mbarrier が
        // 永遠に complete を返さず thread 0 が無期限ハングするのを防ぐ
        // （本ファイル冒頭コメント参照）。回数は「実機で正常時は数十〜
        // 数百 iteration で完了する」想定に対し十分な安全マージンを
        // 取った値であり、実測に基づくチューニング値ではない
        // （TMA タイムアウトの確立された基準値は存在しないため）。
        #define TMA_POLL_LIMIT 1000000u
        unsigned poll_count = 0;
        while (!complete && poll_count < TMA_POLL_LIMIT) {
            asm volatile(
                "{\n"
                ".reg .pred p;\n"
                "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                "selp.u32 %0, 1, 0, p;\n"
                "}\n"
                : "=r"(complete)
                : "r"(mbar_addr), "r"(phase_parity));
            poll_count++;
        }
        timed_out = complete ? 0u : 1u;
        asm volatile("fence.proxy.async.shared::cta;\n");
    }

    __syncthreads();

    // REQ-8: shared→out の書き戻しは out_len を超えるインデックスへ書かない
    // （手動境界チェック省略禁止。`.claude/rules/coding-rust.md`）。
    //
    // `timed_out` != 0（mbarrier が TMA_POLL_LIMIT 回のポーリングで
    // 完了を確認できなかった）場合、shared 側の転送完了は未保証のため
    // `smem_tile` を読まず NaN センチネル（0x7fc00000）を書く。ホスト側
    // ([`tma_execution_probe`]) はビット等値比較で失敗するため
    // 「未完了」を「値が偶然一致しない一般的な不一致」と区別なく検出でき、
    // かつハングせずテストとして終了する。
    if (tid < TILE_ELEMS && tid < out_len) {
        if (timed_out != 0) {
            // `__int_as_float` 等の CUDA math intrinsics は NVRTC 既定の
            // 組み込みヘッダに含まれない可能性があるため（`nvrtc.rs` の
            // 制約と同じ）、bit pattern の再解釈は union で行う。
            union {
                unsigned u;
                float f;
            } nan_bits;
            nan_bits.u = 0x7fc00000u;
            out[tid] = nan_bits.f;
        } else {
            out[tid] = smem_tile[tid];
        }
    }
}
"#;

/// TMA プローブの最小 inline PTX カーネル（NVRTC 向け）。`shared::cta`
/// スコープ版。[`TMA_PROBE_KERNEL`]（`shared::cluster` スコープ版）との
/// 唯一の実質差分は `cp.async.bulk.tensor` 命令の第 4 修飾子
/// （`shared::cluster` → `shared::cta`）とカーネル関数名
/// （`tma_probe_2d` → `tma_probe_2d_cta`。同一モジュール内で 2 カーネルを
/// 区別するため）のみで、他の手順（mbarrier 初期化・タイムアウト付き
/// ポーリング・REQ-8 境界チェック等）は [`TMA_PROBE_KERNEL`] と同一
/// （本ファイル冒頭コメント「cluster / cta 両スコープを probe する理由」
/// 参照）。`shared::cta` スコープはクラスタ起動を要求しないため、ホスト側
/// ([`tma_execution_probe_cta`]) は `cuLaunchKernelEx` ではなく通常の
/// `cuLaunchKernel`（`cudarc::driver::result::launch_kernel`）で起動する。
const TMA_PROBE_KERNEL_CTA: &str = r#"
// `cudarc::driver::sys::CUtensorMap`（`cuda-13000` feature）は
// `#[repr(align(128))]`（本ファイル冒頭コメント参照）。ここでのズレは
// `__grid_constant__ CUtensorMap tensor_map` パラメータ以降のカーネル
// 引数（`out`・`out_len`）の ABI オフセット計算を狂わせうるため、
// ホスト側の実際のアライメントと必ず一致させる。
typedef struct __align__(128) {
    unsigned long long opaque[16];
} CUtensorMap;

#define TILE_M 16
#define TILE_N 16
#define TILE_ELEMS (TILE_M * TILE_N)
#define TILE_BYTES (TILE_ELEMS * 4)

extern "C" __global__ void __launch_bounds__(256) tma_probe_2d_cta(
    const __grid_constant__ CUtensorMap tensor_map,
    float* __restrict__ out,
    int out_len)
{
    __shared__ __align__(128) float smem_tile[TILE_ELEMS];
    __shared__ __align__(8) unsigned long long mbar;
    // mbarrier ポーリングが `TMA_POLL_LIMIT` 回で完了しなかった場合に
    // 全スレッドへ「転送未完了」を伝える合流フラグ（tid 0 が書き、
    // 全スレッドが __syncthreads() 後に読む）。TMA ディスクリプタ不正・
    // sm_121 での TMA 機能不全時にカーネルが無期限ハングするのを防ぐ
    // （プローブの目的は失敗の記録であり、無期限占有ではない）。
    __shared__ unsigned timed_out;

    int tid = threadIdx.x;

    if (tid == 0) {
        unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(mbar_addr));
        asm volatile("fence.proxy.async.shared::cta;\n");

        unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_tile);
        // `&tensor_map`（`__grid_constant__` パラメータ）は NVRTC/nvcc が
        // 生成する generic アドレス表現をそのまま `l` 制約へ渡す（公開
        // TMA サンプルの標準パターン。手動 `cvta.param` を挟むと二重変換に
        // なる可能性があるため挟まない。実機未検証 — 本ファイル冒頭
        // コメント「検証状態」参照）。
        unsigned long long map_addr = (unsigned long long)&tensor_map;

        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
            :: "r"(mbar_addr), "r"((int)TILE_BYTES));

        int coord_x = 0;
        int coord_y = 0;
        // `shared::cluster` ではなく `shared::cta`（本 const のドキュメント
        // コメント参照）。クラスタ起動を要求しない CTA スコープの TMA。
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];\n"
            :: "r"(smem_addr), "l"(map_addr), "r"(coord_x), "r"(coord_y), "r"(mbar_addr)
            : "memory");

        unsigned phase_parity = 0;
        unsigned complete = 0;
        // 上限回数（タイムアウト）付きポーリング。TMA ディスクリプタが
        // 不正、あるいは sm_121 で TMA が機能しない場合に mbarrier が
        // 永遠に complete を返さず thread 0 が無期限ハングするのを防ぐ
        // （本ファイル冒頭コメント参照）。回数は「実機で正常時は数十〜
        // 数百 iteration で完了する」想定に対し十分な安全マージンを
        // 取った値であり、実測に基づくチューニング値ではない
        // （TMA タイムアウトの確立された基準値は存在しないため）。
        #define TMA_POLL_LIMIT 1000000u
        unsigned poll_count = 0;
        while (!complete && poll_count < TMA_POLL_LIMIT) {
            asm volatile(
                "{\n"
                ".reg .pred p;\n"
                "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                "selp.u32 %0, 1, 0, p;\n"
                "}\n"
                : "=r"(complete)
                : "r"(mbar_addr), "r"(phase_parity));
            poll_count++;
        }
        timed_out = complete ? 0u : 1u;
        asm volatile("fence.proxy.async.shared::cta;\n");
    }

    __syncthreads();

    // REQ-8: shared→out の書き戻しは out_len を超えるインデックスへ書かない
    // （手動境界チェック省略禁止。`.claude/rules/coding-rust.md`）。
    //
    // `timed_out` != 0（mbarrier が TMA_POLL_LIMIT 回のポーリングで
    // 完了を確認できなかった）場合、shared 側の転送完了は未保証のため
    // `smem_tile` を読まず NaN センチネル（0x7fc00000）を書く。ホスト側
    // ([`tma_execution_probe_cta`]) はビット等値比較で失敗するため
    // 「未完了」を「値が偶然一致しない一般的な不一致」と区別なく検出でき、
    // かつハングせずテストとして終了する。
    if (tid < TILE_ELEMS && tid < out_len) {
        if (timed_out != 0) {
            // `__int_as_float` 等の CUDA math intrinsics は NVRTC 既定の
            // 組み込みヘッダに含まれない可能性があるため（`nvrtc.rs` の
            // 制約と同じ）、bit pattern の再解釈は union で行う。
            union {
                unsigned u;
                float f;
            } nan_bits;
            nan_bits.u = 0x7fc00000u;
            out[tid] = nan_bits.f;
        } else {
            out[tid] = smem_tile[tid];
        }
    }
}
"#;

/// [`tma_nvrtc_compile_probe`] が記録する 1 arch × 1 opcode スコープ
/// （cluster / cta）分のコンパイル結果。
struct CompileProbeResult {
    arch: &'static str,
    /// opcode スコープの識別子。`"cluster"`（[`TMA_PROBE_KERNEL`]）または
    /// `"cta"`（[`TMA_PROBE_KERNEL_CTA`]）。cluster のみを probe すると
    /// sm_121 で CTA スコープ TMA が機能していても記録上「TMA 不可」と
    /// 誤認しうるため両方を区別して記録する（本ファイル冒頭コメント参照）。
    variant: &'static str,
    /// `Ok(())` はコンパイル成功、`Err` は失敗時のエラーメッセージ全文
    /// （`CudaError::Display` 経由。`NvrtcUnavailable` かどうかは
    /// [`CompileProbeResult::is_nvrtc_unavailable`] で判定する）。
    outcome: Result<(), String>,
    is_nvrtc_unavailable: bool,
}

/// `arch` × `variant`（`kernel_src`）向けにカーネルをコンパイルし、
/// 成否とエラー全文を記録する（`--nocapture` での構造化出力・記録表
/// 転記用）。
fn probe_compile(
    variant: &'static str,
    kernel_src: &'static str,
    arch: &'static str,
) -> CompileProbeResult {
    match backend_cuda::compile_ptx(kernel_src, arch) {
        Ok(_ptx) => CompileProbeResult {
            arch,
            variant,
            outcome: Ok(()),
            is_nvrtc_unavailable: false,
        },
        Err(err) => {
            let is_nvrtc_unavailable =
                matches!(err, backend_cuda::CudaError::NvrtcUnavailable { .. });
            CompileProbeResult {
                arch,
                variant,
                outcome: Err(err.to_string()),
                is_nvrtc_unavailable,
            }
        }
    }
}

/// [`tma_execution_probe`]・[`tma_execution_probe_cta`] が実行対象の arch を
/// 選ぶための共通ロジック（PR #634 codex-review 指摘対応）。
///
/// [`PROBE_ARCHS`] を優先順位（`compute_121` → `compute_121a` →
/// `compute_121f`）どおりに試し、最初にコンパイル成功した arch を返す。
/// `compute_121` が拒否され代替 arch のみが成功する環境でも、その arch で
/// 実行可能性を検証できるようにする（#483 受け入れ基準の判定基準
/// 「`compute_121` または `compute_121a`/`compute_121f` でコンパイル・実行
/// 成功」に対応。本ファイル冒頭コメント「cluster / cta 両スコープを probe
/// する理由」と同じ動機）。
///
/// 全 arch が失敗した場合は [`tma_nvrtc_compile_probe`] と同じ
/// `variant=… arch=… result=failure detail=…` 形式で各 arch の失敗を記録
/// してから、全エラーメッセージを連結した理由で `panic` する（実行プローブ
/// 単独で見ても、どの arch が何のエラーで拒否されたか
/// `docs/cuda-tensor-core-design.md` の記録表へ転記できるようにするため。
/// silent green を許さない方針は変えない）。
fn select_compiling_arch(variant: &'static str, kernel_src: &'static str) -> &'static str {
    let mut failures: Vec<String> = Vec::new();
    for arch in PROBE_ARCHS {
        match backend_cuda::compile_ptx(kernel_src, arch) {
            Ok(_ptx) => {
                println!(
                    "tma_compile_probe variant={variant} arch={arch} result=success (selected for execution probe)"
                );
                return arch;
            }
            Err(err) => {
                println!(
                    "tma_compile_probe variant={variant} arch={arch} result=failure detail={err:?}"
                );
                failures.push(format!("arch={arch} detail={err}"));
            }
        }
    }
    panic!(
        "TMA probe kernel (variant={variant}) failed to compile for every arch in \
         PROBE_ARCHS ({PROBE_ARCHS:?}); cannot select an arch to execute. Failures: \
         {}",
        failures.join("; ")
    );
}

/// コンパイルプローブ本体（#483 受け入れ基準 1・2）。
///
/// `compute_121`／`compute_121a`／`compute_121f` の 3 arch × cluster/cta
/// の 2 opcode スコープ（本ファイル冒頭コメント「cluster / cta 両スコープ
/// を probe する理由」参照）の計 6 組み合わせでそれぞれコンパイルを試み、
/// 成否・エラーメッセージ全文を `--nocapture` 前提で構造化出力する
/// （`docs/cuda-tensor-core-design.md` の記録表へ転記する運用）。
///
/// 6 組み合わせすべてが `NvrtcUnavailable`（NVRTC 自体が不在）だった場合は
/// プローブとして無意味（TMA 命令列自体の成否を何も語らない）なので
/// `panic` させる。それ以外（少なくとも 1 組み合わせで NVRTC が実際に
/// ソースを解析した）であれば、TMA 命令列自体の受理・拒否いずれであっても
/// プローブとして有効な記録が得られたとみなしテストを通す。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、sm_121）必須。実測記録は docs/cuda-tensor-core-design.md「TMA sm_121 プローブ」節"]
fn tma_nvrtc_compile_probe() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    println!(
        "environment: name={:?} compute_capability={:?} arch={:?}",
        device.name(),
        device.compute_capability(),
        device.arch()
    );

    let variants: [(&'static str, &'static str); 2] =
        [("cluster", TMA_PROBE_KERNEL), ("cta", TMA_PROBE_KERNEL_CTA)];
    let results: Vec<CompileProbeResult> = variants
        .into_iter()
        .flat_map(|(variant, kernel_src)| {
            PROBE_ARCHS
                .into_iter()
                .map(move |arch| probe_compile(variant, kernel_src, arch))
        })
        .collect();

    for result in &results {
        match &result.outcome {
            Ok(()) => println!(
                "tma_compile_probe variant={} arch={} result=success",
                result.variant, result.arch
            ),
            Err(detail) => println!(
                "tma_compile_probe variant={} arch={} result=failure detail={detail:?}",
                result.variant, result.arch
            ),
        }
    }

    let any_meaningful = results.iter().any(|r| !r.is_nvrtc_unavailable);
    assert!(
        any_meaningful,
        "cluster/cta 2 variant × 3 arch（compute_121/compute_121a/compute_121f）\
         の計 6 組み合わせすべてで NVRTC 自体が不在（NvrtcUnavailable）でした。\
         この実行環境では TMA 命令列の成否について何の記録も得られません。\
         libnvrtc が導入された実機（DGX Spark GB10 等）で再実行してください"
    );
}

/// [`tma_execution_probe`] が host 側で保持する `CUtensorMap`。
///
/// `cudarc::driver::sys::CUtensorMap` はマーカートレイト `DeviceRepr`
/// （orphan rule により本クレート外の型 × 本クレート外のトレイトは
/// 直接 impl できない）を実装していないため、カーネル引数として値渡し
/// できるよう `#[repr(transparent)]` のローカルラッパー型で包む
/// （`gemm_mma.rs` 等の既存カーネル引数はプリミティブ型のみで済んでいた
/// ため、本ファイルが本クレート初の `DeviceRepr` ラッパー実装になる）。
#[repr(transparent)]
struct TensorMapArg(CUtensorMap);

// SAFETY: `DeviceRepr` はマーカートレイト（メソッドを持たない）であり、
// 要求されるのは「カーネル引数としてバイト単位でそのままデバイスへ渡して
// よい POD（Plain Old Data）である」という契約のみ。`CUtensorMap`
// （`cudarc::driver::sys` 側で `#[repr(C)]`・`#[repr(align(128))]`・
// `opaque: [u64; 16]` の POD 構造体として定義済み）をラップした
// `#[repr(transparent)]` 型はこの契約を満たす。値の中身自体は
// `cuTensorMapEncodeTiled`（driver API）が書き込む不透明ディスクリプタで
// あり、Rust 側では解釈せずそのまま右から左へ受け渡すのみ。
unsafe impl cudarc::driver::DeviceRepr for TensorMapArg {}

/// TMA 実行プローブ本体（#483 受け入れ基準 1）。
///
/// [`select_compiling_arch`] で [`PROBE_ARCHS`]（`compute_121` →
/// `compute_121a` → `compute_121f`）を順に試し、最初にコンパイル成功した
/// arch で [`TMA_PROBE_KERNEL`] をロードし、64x64 の f32 global テンソルから
/// `(0,0)` 起点の 16x16 タイルを `cuTensorMapEncodeTiled` で生成した
/// `CUtensorMap` 経由で転送する。転送されたタイルをホストへ回収し
/// ソース領域とのビット等値比較で検証する（純粋コピーのため許容誤差の
/// 概念を持ち込まない = tolerance 変更なしの共通契約と整合。
/// `.claude/rules/coding-rust.md`）。`compute_121` 固定だと、`compute_121`
/// が拒否され代替 arch のみ成功する環境で実行可能性を確認できない
/// （PR #634 codex-review 指摘対応）。
///
/// [`PROBE_ARCHS`] 全 arch でコンパイルが失敗した場合は実行を試みず
/// [`select_compiling_arch`] 内の `panic` で失敗を顕在化させる
/// （コンパイルプローブが失敗を記録した環境で実行プローブだけ silent
/// green になることを避ける）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、sm_121）必須。実測記録は docs/cuda-tensor-core-design.md「TMA sm_121 プローブ」節"]
fn tma_execution_probe() {
    const GLOBAL_M: usize = 64;
    const GLOBAL_N: usize = 64;
    const TILE_M: u32 = 16;
    const TILE_N: u32 = 16;
    const TILE_ELEMS: usize = (TILE_M as usize) * (TILE_N as usize);

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");

    let selected_arch = select_compiling_arch("cluster", TMA_PROBE_KERNEL);
    let ptx = backend_cuda::compile_ptx(TMA_PROBE_KERNEL, selected_arch).expect(
        "TMA probe kernel must compile for the arch select_compiling_arch just selected \
         (recompiling here avoids threading a Ptx value out of the selection helper)",
    );

    // `cudarc::driver::safe::CudaModule`/`CudaFunction` は raw `CUmodule`/
    // `CUfunction` ハンドルを private field に隠しており、`cuLaunchKernelEx`
    // （クラスタ次元指定に必須。本ファイル冒頭コメント「`shared::cluster`」
    // 参照）へ渡せない。そのため本テストのみ `cudarc::driver::result::module`
    // （`cuModuleLoadData`/`cuModuleGetFunction` の薄いラッパー。他クレート
    // コードは `gemm.rs`/`gemm_wmma.rs`/`gemm_mma.rs` すべて safe API 経由
    // であり、raw ハンドルを要する本テストのみ例外的に低レベル API を使う）
    // を直接使う。
    //
    // SAFETY: `load_data` はコンパイル済み PTX 文字列（NUL 終端）を渡す。
    // 呼び出し元スレッドは `CudaDevice::new` が `bind_to_thread()` 済みの
    // primary context を保持しているため（`cudarc` の `CudaContext::new`
    // 実装。本ファイル冒頭コメント「実機前提」）、`cuModuleLoadData` は
    // その context 上へロードされる。`get_function` の `module` は直前に
    // 取得した有効な `CUmodule`（未 unload）。
    let ptx_src = std::ffi::CString::new(ptx.to_src())
        .expect("compiled PTX source must not contain interior NUL bytes");
    let cu_module =
        unsafe { cudarc::driver::result::module::load_data(ptx_src.as_ptr() as *const c_void) }
            .expect("cuModuleLoadData must succeed for a successfully compiled PTX module");
    let fn_name = std::ffi::CString::new("tma_probe_2d").expect("kernel name has no interior NUL");
    let cu_function = unsafe { cudarc::driver::result::module::get_function(cu_module, fn_name) }
        .expect("tma_probe_2d function must be present in the loaded module");

    // global テンソル（64x64 f32、row-major）。境界内タイルのみを扱う
    // ため OOB fill 経路（`CUtensorMapFloatOOBfill`）は NONE のままでよい。
    let mut rng = bench_harness::rng::Xorshift64Star::new(0x7A5A_7473);
    let global_host: Vec<f32> = rng.fill_vec(GLOBAL_M * GLOBAL_N);
    let global_dev = device
        .stream()
        .clone_htod(&global_host)
        .expect("clone_htod must succeed on CUDA-equipped test runner");
    let mut out_dev = device
        .stream()
        .alloc_zeros::<f32>(TILE_ELEMS)
        .expect("alloc_zeros must succeed on CUDA-equipped test runner");

    // SAFETY: `cuTensorMapEncodeTiled` は driver API の FFI 呼び出し。
    // - `tensor_map`: スタック上の `CUtensorMap`（`Default` 相当のゼロ初期化
    //   から書き込み先として渡す。関数内で全フィールドを埋める契約）
    // - `globalAddress`: `global_dev` の device pointer（`DevicePtr::device_ptr`
    //   経由で取得。`SyncOnDrop` ガードは encode 呼び出しの完了まで
    //   スコープに保持し、転送完了前にストリーム同期が走らないようにする）
    // - `globalDim`/`globalStrides`/`boxDim`/`elementStrides`:
    //   64x64 f32・16x16 タイル・row stride 256B（64 要素 x 4B、16B
    //   境界を満たす）。すべてスタック上の配列で FFI 呼び出しの間だけ
    //   生存すればよい
    // `CUresult` は `.result()`（`cudarc::driver::sys` の拡張）で
    // `Result<(), DriverError>` に変換し、成否をそのまま `.expect` で
    // 顕在化させる（他コード同様 panic ではなく明示 fail を優先したいが、
    // 本テストは `#[ignore]` 実機プローブであり `tests/gemm_mma.rs` 等
    // 既存の実機テストと同じ `.expect` 顕在化方針に合わせる）。
    let mut tensor_map = TensorMapArg(unsafe { std::mem::zeroed::<CUtensorMap>() });
    let global_dim: [u64; 2] = [GLOBAL_N as u64, GLOBAL_M as u64];
    let global_strides: [u64; 1] = [(GLOBAL_N * std::mem::size_of::<f32>()) as u64];
    let box_dim: [u32; 2] = [TILE_N, TILE_M];
    let element_strides: [u32; 2] = [1, 1];
    {
        let (global_ptr, _sync_guard) = global_dev.device_ptr(device.stream());
        unsafe {
            sys::cuTensorMapEncodeTiled(
                &mut tensor_map.0 as *mut CUtensorMap,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                2,
                global_ptr as *mut c_void,
                global_dim.as_ptr(),
                global_strides.as_ptr(),
                box_dim.as_ptr(),
                element_strides.as_ptr(),
                CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        }
        .result()
        .expect(
            "cuTensorMapEncodeTiled must succeed for an in-bounds 16x16 tile of a 64x64 f32 tensor",
        );
    }

    let out_len_i32 = TILE_ELEMS as i32;

    // クラスタ次元 (1,1,1) を明示するため `cudarc` の safe `launch_builder`
    // （クラスタ非対応）ではなく `cuLaunchKernelEx` を直接呼ぶ
    // （本ファイル冒頭コメント「`shared::cluster`」参照）。
    //
    // SAFETY: `kernel_params` はロード済み `func`（`tma_probe_2d`）の
    // シグネチャ（`CUtensorMap` 値渡し・`float*`・`int`）と 3 個・型・順序が
    // 1:1 対応する。各要素は「引数値そのものへのポインタ」（driver API の
    // 標準契約）であり、呼び出し完了（`synchronize` 後）まで有効な
    // スタック変数を指す。カーネル内の手動境界チェック（`out_len` ガード。
    // 本ファイル冒頭コメント参照、REQ-8）と合わせて OOB 書き込みが
    // 起きない根拠とする。
    {
        let (out_ptr, _sync_guard) = out_dev.device_ptr_mut(device.stream());
        let mut out_ptr = out_ptr;
        let mut out_len = out_len_i32;
        let mut kernel_params: [*mut c_void; 3] = [
            &mut tensor_map as *mut TensorMapArg as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut out_len as *mut i32 as *mut c_void,
        ];

        let mut cluster_attr = CUlaunchAttribute {
            id: CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
            pad: [0; 4],
            value: CUlaunchAttributeValue { pad: [0; 64] },
        };
        // `CUlaunchAttributeValue` は union。union フィールドへの「書き込み」
        // は安全（読み出しのみ unsafe）。`id` を `CLUSTER_DIMENSION` に設定
        // 済みのため driver 側は `clusterDim` フィールドとして解釈する契約
        // （CUDA Driver API 仕様）。
        cluster_attr.value.clusterDim.x = 1;
        cluster_attr.value.clusterDim.y = 1;
        cluster_attr.value.clusterDim.z = 1;

        let launch_config = CUlaunchConfig {
            gridDimX: 1,
            gridDimY: 1,
            gridDimZ: 1,
            blockDimX: 256,
            blockDimY: 1,
            blockDimZ: 1,
            sharedMemBytes: 0,
            hStream: device.stream().cu_stream(),
            attrs: &mut cluster_attr as *mut CUlaunchAttribute,
            numAttrs: 1,
        };

        // SAFETY: `func` は上で正常ロード済みの `CUfunction`。
        // `kernel_params`/`launch_config` は上記の通り検証済み。
        // `extra` は未使用（null）。
        unsafe {
            sys::cuLaunchKernelEx(
                &launch_config as *const CUlaunchConfig,
                cu_function,
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
        .result()
        .expect("cuLaunchKernelEx must succeed for the TMA probe kernel");
    }

    device
        .stream()
        .synchronize()
        .expect("synchronize must succeed after the TMA probe kernel launch");

    let out_host = device
        .stream()
        .clone_dtoh(&out_dev)
        .expect("clone_dtoh must succeed on CUDA-equipped test runner");

    // ソース領域（global の (0,0)〜(15,15) タイル、row stride 64 要素）との
    // ビット等値比較。純粋コピーのため許容誤差は持ち込まない（本ファイル
    // 冒頭コメント参照）。
    let mut expected = vec![0.0f32; TILE_ELEMS];
    for row in 0..(TILE_M as usize) {
        for col in 0..(TILE_N as usize) {
            expected[row * (TILE_N as usize) + col] = global_host[row * GLOBAL_N + col];
        }
    }
    assert_eq!(
        out_host, expected,
        "TMA 転送結果が期待するタイル（global (0,0) 起点 16x16）とビット単位で \
         一致しませんでした（純粋コピーのため誤差許容なし）"
    );

    println!(
        "tma_execution_probe variant=cluster arch={selected_arch} result=success tile=16x16 \
         global=64x64 bitwise_match=true"
    );

    // SAFETY: `cu_module` はこの時点で以降どこからも参照されない
    // （`cu_function` は呼び出し済みで既にカーネル起動が完了している）。
    // アンロード失敗はプローブの主目的（転送結果の検証）に影響しないため
    // `.ok()` で握りつぶす（テスト末尾のリソース解放であり、失敗しても
    // プロセス終了時に driver がコンテキストごと回収する）。
    unsafe { cudarc::driver::result::module::unload(cu_module) }.ok();
}

/// TMA 実行プローブ本体（cta スコープ版。#483 受け入れ基準 1・PR #634
/// cursor[bot] レビュー指摘対応）。
///
/// [`tma_execution_probe`]（`shared::cluster` スコープ・[`TMA_PROBE_KERNEL`]）
/// と対になる `shared::cta` スコープ版で、[`TMA_PROBE_KERNEL_CTA`] を
/// [`select_compiling_arch`] で [`PROBE_ARCHS`] から選んだ arch でロードし
/// 同じ 64x64 → 16x16 タイル転送・ビット等値比較を行う（本ファイル冒頭
/// コメント「cluster / cta 両スコープを probe する理由」参照）。
/// `compute_121` 固定だと、`compute_121` が拒否され代替 arch のみ成功する
/// 環境で実行可能性を確認できない（[`tma_execution_probe`] と同じ理由。
/// PR #634 codex-review 指摘対応）。`shared::cta` はクラスタ起動を要求しない
/// ため、[`tma_execution_probe`] のように raw `cuLaunchKernelEx` を使う必要
/// はなく `cudarc::driver::result::launch_kernel`（`cuLaunchKernel` の薄い
/// ラッパー）で起動する。ただし `cudarc::driver::safe::CudaModule` は
/// 依然として raw `CUfunction` を公開しないため、モジュールロード自体は
/// [`tma_execution_probe`] と同じ `cudarc::driver::result::module` 経路を
/// 使う。
///
/// [`PROBE_ARCHS`] 全 arch でコンパイルが失敗した場合は実行を試みず
/// [`select_compiling_arch`] 内の `panic` で失敗を顕在化させる
/// （[`tma_execution_probe`] と同じ方針。コンパイルプローブが失敗を記録した
/// 環境で実行プローブだけ silent green になることを避ける）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、sm_121）必須。実測記録は docs/cuda-tensor-core-design.md「TMA sm_121 プローブ」節"]
fn tma_execution_probe_cta() {
    const GLOBAL_M: usize = 64;
    const GLOBAL_N: usize = 64;
    const TILE_M: u32 = 16;
    const TILE_N: u32 = 16;
    const TILE_ELEMS: usize = (TILE_M as usize) * (TILE_N as usize);

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");

    let selected_arch = select_compiling_arch("cta", TMA_PROBE_KERNEL_CTA);
    let ptx = backend_cuda::compile_ptx(TMA_PROBE_KERNEL_CTA, selected_arch).expect(
        "TMA probe kernel (cta variant) must compile for the arch select_compiling_arch \
         just selected (recompiling here avoids threading a Ptx value out of the \
         selection helper)",
    );

    // SAFETY: [`tma_execution_probe`] の同名ステップと同じ契約
    // （呼び出し元スレッドが `CudaDevice::new` で bind 済みの primary
    // context を保持していること）。
    let ptx_src = std::ffi::CString::new(ptx.to_src())
        .expect("compiled PTX source must not contain interior NUL bytes");
    let cu_module =
        unsafe { cudarc::driver::result::module::load_data(ptx_src.as_ptr() as *const c_void) }
            .expect("cuModuleLoadData must succeed for a successfully compiled PTX module");
    let fn_name =
        std::ffi::CString::new("tma_probe_2d_cta").expect("kernel name has no interior NUL");
    let cu_function = unsafe { cudarc::driver::result::module::get_function(cu_module, fn_name) }
        .expect("tma_probe_2d_cta function must be present in the loaded module");

    // global テンソル（64x64 f32、row-major）。[`tma_execution_probe`] と
    // 同一の乱数シードを使う（同じソース領域・期待値計算で cluster/cta
    // 両 variant を対称に比較できるようにするため）。
    let mut rng = bench_harness::rng::Xorshift64Star::new(0x7A5A_7473);
    let global_host: Vec<f32> = rng.fill_vec(GLOBAL_M * GLOBAL_N);
    let global_dev = device
        .stream()
        .clone_htod(&global_host)
        .expect("clone_htod must succeed on CUDA-equipped test runner");
    let mut out_dev = device
        .stream()
        .alloc_zeros::<f32>(TILE_ELEMS)
        .expect("alloc_zeros must succeed on CUDA-equipped test runner");

    // SAFETY: [`tma_execution_probe`] の同名ステップ（`cuTensorMapEncodeTiled`
    // 呼び出し）と同じ契約・同じパラメータ（cluster/cta で TMA
    // ディスクリプタ自体は変わらない。差分は転送命令の shared スコープ
    // 修飾子のみ）。
    let mut tensor_map = TensorMapArg(unsafe { std::mem::zeroed::<CUtensorMap>() });
    let global_dim: [u64; 2] = [GLOBAL_N as u64, GLOBAL_M as u64];
    let global_strides: [u64; 1] = [(GLOBAL_N * std::mem::size_of::<f32>()) as u64];
    let box_dim: [u32; 2] = [TILE_N, TILE_M];
    let element_strides: [u32; 2] = [1, 1];
    {
        let (global_ptr, _sync_guard) = global_dev.device_ptr(device.stream());
        unsafe {
            sys::cuTensorMapEncodeTiled(
                &mut tensor_map.0 as *mut CUtensorMap,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                2,
                global_ptr as *mut c_void,
                global_dim.as_ptr(),
                global_strides.as_ptr(),
                box_dim.as_ptr(),
                element_strides.as_ptr(),
                CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        }
        .result()
        .expect(
            "cuTensorMapEncodeTiled must succeed for an in-bounds 16x16 tile of a 64x64 f32 tensor",
        );
    }

    let out_len_i32 = TILE_ELEMS as i32;

    // `shared::cta` はクラスタ起動を要求しないため（本関数のドキュメント
    // コメント参照）、`cuLaunchKernelEx` + クラスタ属性ではなく通常の
    // `cuLaunchKernel`（`cudarc::driver::result::launch_kernel`）で起動する。
    //
    // SAFETY: `func` は上で正常ロード済みの `CUfunction`。`kernel_params`
    // は [`tma_execution_probe`] と同じ 3 要素（`CUtensorMap` 値渡し・
    // `float*`・`int`）で、ロード済み関数のシグネチャと 1:1 対応する。
    // 各要素は「引数値そのものへのポインタ」（driver API の標準契約）
    // であり、呼び出し完了（`synchronize` 後）まで有効なスタック変数を
    // 指す。カーネル内の手動境界チェック（`out_len` ガード。REQ-8）と
    // 合わせて OOB 書き込みが起きない根拠とする。
    {
        let (out_ptr, _sync_guard) = out_dev.device_ptr_mut(device.stream());
        let mut out_ptr = out_ptr;
        let mut out_len = out_len_i32;
        let mut kernel_params: [*mut c_void; 3] = [
            &mut tensor_map as *mut TensorMapArg as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut out_len as *mut i32 as *mut c_void,
        ];

        unsafe {
            cudarc::driver::result::launch_kernel(
                cu_function,
                (1, 1, 1),
                (256, 1, 1),
                0,
                device.stream().cu_stream(),
                &mut kernel_params,
            )
        }
        .expect("cuLaunchKernel must succeed for the TMA probe kernel (cta variant)");
    }

    device
        .stream()
        .synchronize()
        .expect("synchronize must succeed after the TMA probe kernel launch");

    let out_host = device
        .stream()
        .clone_dtoh(&out_dev)
        .expect("clone_dtoh must succeed on CUDA-equipped test runner");

    // ソース領域（global の (0,0)〜(15,15) タイル、row stride 64 要素）との
    // ビット等値比較。純粋コピーのため許容誤差は持ち込まない
    // （[`tma_execution_probe`] と同じ契約）。
    let mut expected = vec![0.0f32; TILE_ELEMS];
    for row in 0..(TILE_M as usize) {
        for col in 0..(TILE_N as usize) {
            expected[row * (TILE_N as usize) + col] = global_host[row * GLOBAL_N + col];
        }
    }
    assert_eq!(
        out_host, expected,
        "TMA 転送結果（cta variant）が期待するタイル（global (0,0) 起点 16x16）と \
         ビット単位で一致しませんでした（純粋コピーのため誤差許容なし）"
    );

    println!(
        "tma_execution_probe_cta variant=cta arch={selected_arch} result=success tile=16x16 \
         global=64x64 bitwise_match=true"
    );

    // SAFETY: [`tma_execution_probe`] の同名ステップと同じ契約
    // （アンロード失敗はプローブの主目的に影響しないため `.ok()` で
    // 握りつぶす）。
    unsafe { cudarc::driver::result::module::unload(cu_module) }.ok();
}
