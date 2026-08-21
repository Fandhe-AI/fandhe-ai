//! f16 `mma.sync`/`ldmatrix`/`cp.async` GEMM の起動 API（TASK-11.1h・#187）。
//!
//! `CudaMmaGemm` は `kernels_mma::mma_f16_source()`（`m16n8k16` mma・3 ステージ
//! `cp.async` パイプライン）をコンパイル・保持し、以降はホスト側スライスを
//! 渡すだけで GPU 実行できる境界を担う（`gemm_wmma.rs::CudaWmmaGemm` と
//! 同じ責務分割。並行 issue #62/#63 が `gemm.rs`／`gemm_wmma.rs` を編集中の
//! ため、本イシューでは既存ファイルに触れず独立ファイルへ分離する）。
//!
//! ホスト側形状検証は `gemm.rs::validate_gemm_dims`（`pub(crate)`）を
//! そのまま再利用し、判定ロジックを複製しない。加えて本経路固有の
//! `cp.async` 16 バイト整列制約（[`validate_mma_alignment`]）を追加で
//! 検証する（`kernels_mma.rs` 冒頭ドキュメントコメント「整列制約」参照）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::f16;

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm::validate_gemm_dims;
use crate::kernels_mma;
use crate::nvrtc::compile_ptx;

/// `mma.sync`/`ldmatrix`/`cp.async` 経路が要求する compute capability の
/// 下限（major）。
///
/// `cp.async`・`ldmatrix` は LDGSTS（compute capability 8.0+）を要求する
/// （`nvidia-cuda` スキル `references/advanced/features/async-copies.md`
/// 「LDGSTS (CC 8.0+)」。`kernels_mma.rs` 冒頭コメント「命令選定・sm_80+
/// ゲート」）。WMMA 経路（cc>=7.0・`gemm_wmma.rs::MIN_COMPUTE_CAPABILITY_MAJOR`）
/// より厳しい下限であり、独立した定数として保持する。
const MIN_COMPUTE_CAPABILITY_MAJOR: i32 = 8;

/// `kernels_mma::MMA_BLOCK_THREADS` に 1:1 対応するブロック次元。
const MMA_BLOCK_DIM: (u32, u32, u32) = (kernels_mma::MMA_BLOCK_THREADS, 1, 1);

/// `cp.async.cg.shared.global` の 16 バイト転送粒度が要求するグローバル側
/// 整列制約を検証する（`kernels_mma.rs` 冒頭コメント「整列制約」参照）。
///
/// A の行ストライドは `k`、B の行ストライドは `n` であり、共有メモリ側の
/// タイル幅（`MMA_BK`/`MMA_BN`）が共に 8 の倍数であることと合わせて
/// `k % 8 == 0 && n % 8 == 0` を満たさない限り、行境界をまたぐ列オフセット
/// が 16 バイト境界からずれうる。`gemm.rs::validate_tiled_k_bound`
/// （tiled 経路の K 追加検証）と同種の「経路固有の追加検証」パターンで
/// あり、`validate_gemm_dims` の一般契約（`CudaError::InvalidShape`）を
/// 再利用する。
///
/// `pub(crate)`: `tests/gemm_mma.rs` から実機非依存の単体テストとして
/// 直接呼べるようにする（`validate_gemm_dims` と同じ公開範囲方針）。
pub(crate) fn validate_mma_alignment(n: u32, k: u32) -> Result<(), CudaError> {
    if !k.is_multiple_of(8) || !n.is_multiple_of(8) {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "mma.sync/cp.async path requires k % 8 == 0 && n % 8 == 0 \
                 (cp.async 16-byte transfer granularity; kernels_mma.rs \
                 doc comment \"整列制約\"), but got n={n}, k={k}"
            ),
        });
    }
    Ok(())
}

/// CUDA の grid 次元 y/z 成分の上限（65,535。全 compute capability 共通。
/// x 成分の上限は 2^31-1 と大きく実用的に問題にならないため x は検証
/// しない）。
const MAX_GRID_DIM_Y: u32 = 65_535;

/// `mma_launch_config` が構築するグリッドの y 成分（`m.div_ceil(MMA_BM)`）
/// が CUDA の上限（65,535）を超えないことを検証する（PR #255 レビュー
/// 指摘。超過するとホスト側の形状・整列検証はすべて通過した上で、
/// ドライバのカーネル起動が失敗する。`validate_mma_alignment` と同種の
/// 「経路固有の追加検証」パターン）。
///
/// `pub(crate)`: `tests/gemm_mma.rs` から実機非依存の単体テストとして
/// 直接呼べるようにする（`validate_mma_alignment` と同じ公開範囲方針）。
pub(crate) fn validate_mma_grid_bounds(m: u32) -> Result<(), CudaError> {
    let grid_y = m.div_ceil(kernels_mma::MMA_BM);
    if grid_y > MAX_GRID_DIM_Y {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "mma.sync path grid_dim.y (m.div_ceil(MMA_BM)={grid_y}) exceeds CUDA's \
                 {MAX_GRID_DIM_Y} limit for grid dimensions y/z (MMA_BM={}); m={m} is too large",
                kernels_mma::MMA_BM
            ),
        });
    }
    Ok(())
}

/// f16 `mma.sync`/`ldmatrix`/`cp.async` GEMM カーネルのコンパイル済み
/// ハンドルを保持する。
///
/// `stream` は `CudaDevice` から `Arc` クローンで受け取る（`gemm_wmma.rs`
/// と同じ共有契約）。
///
/// `mma_f16`: 常時保持する起動カーネル（[`new`](Self::new)（本番既定
/// コンストラクタ。イシュー #782 でサイズ条件付き swizzle 選択機構を結線
/// した後も base カーネル自体は常に保持する）・
/// [`new_without_swizzle`](Self::new_without_swizzle) では swizzle 無適用
/// base カーネル、[`new_with_swizzle`](Self::new_with_swizzle) では
/// swizzle 変種そのもの。詳細は各コンストラクタ doc comment 参照）。
///
/// `mma_f16_swizzle`: [`new`](Self::new)（本番既定コンストラクタ。イシュー
/// #782 で `new_with_size_conditional_swizzle` 相当のロジックを昇格）が
/// サイズ条件付き適用のために追加コンパイルする swizzle 変種ハンドル。
/// `Some(_)` は `new` が `device.multiprocessor_count()` を取得でき変種を
/// コンパイルできた場合（`launch_f16` は呼び出し形状ごとに [`crate::
/// swizzle::should_apply_swizzle`] で `mma_f16`／`mma_f16_swizzle` の
/// いずれを起動するか判定する）。`None` は
/// [`new_without_swizzle`](Self::new_without_swizzle)・
/// [`new_with_swizzle`](Self::new_with_swizzle)（いずれも単一カーネルを
/// 強制適用／非適用する診断用入口のため追加変種を持たない）、または
/// `new` が SM 数を取得できず安全側で base のみを保持した場合（fail-soft。
/// 下記 [`new`](Self::new) doc comment 参照）を意味する。
///
/// `swizzle_group_width`: L2 再利用のためのタイル→SM 割り当てスウィズル
/// （イシュー #499・#775・#782）のグルーピング幅。`Some(_)` は
/// `mma_f16`／`mma_f16_swizzle` のいずれかが swizzle 変種を保持している
/// ことを意味する（[`swizzle_group_width`](Self::swizzle_group_width)
/// アクセサ経由で可観測にする。#733 の `wmma_tf32_staged` 可用性出力と
/// 同型の起動時診断。`examples/cuda_floor_bench.rs` 参照）。個別呼び出し
/// で実際に swizzle が適用されるかは
/// [`swizzle_applies`](Self::swizzle_applies) を使う。
///
/// `swizzle_compile_error`: [`new`](Self::new) が SM 数実測に成功した
/// にもかかわらず swizzle 変種のソース生成・NVRTC コンパイルに失敗した
/// 場合の理由文字列（`gemm_wmma.rs::CudaWmmaGemm::wmma_f16_opt_error` と
/// 同型の fail-soft 方針。swizzle は base カーネルの可用性とは独立な
/// 性能最適化に過ぎないため、この失敗で `new` 全体を `Err` にはしない）。
/// [`swizzle_unavailable_reason`](Self::swizzle_unavailable_reason) 経由で
/// 読み取れる。
pub struct CudaMmaGemm {
    stream: Arc<CudaStream>,
    mma_f16: CudaFunction,
    mma_f16_swizzle: Option<CudaFunction>,
    swizzle_group_width: Option<u32>,
    swizzle_compile_error: Option<String>,
}

/// [`check_min_compute_capability`] のゲート検査後、`source` を NVRTC
/// コンパイルし `gemm_mma_f16` 関数ハンドルをロードする共通手順
/// （[`CudaMmaGemm::new`]・[`new_with_swizzle`](CudaMmaGemm::new_with_swizzle)・
/// [`new_without_swizzle`](CudaMmaGemm::new_without_swizzle) の 3
/// コンストラクタが、base カーネルのコンパイルに共有する。[`new`
/// ](CudaMmaGemm::new) はさらに swizzle 変種ソース（コンパイル対象文字列
/// のみが異なる）のコンパイルにもこの共通手順を再利用する。イシュー #740
/// で（当時の）`new` が swizzle 変種をコンパイルするようになり、コンパイル
/// 対象ソース文字列のみが異なる同一手順になったため重複を排除した。
/// イシュー #775 で `new` を一旦 base 専用へ戻し、サイズ条件付き変種の
/// コンパイルは `new_with_size_conditional_swizzle`（`internal-diagnostics`
/// feature 限定の実機検証専用入口）へ切り出したが、イシュー #782 で GB10
/// 実機ゲート通過（2026-08-21）を根拠に同ロジックを `new` へ昇格し、
/// 実機検証専用入口としての `new_with_size_conditional_swizzle` は
/// （`new` と重複するため）廃止した。
fn compile_mma_f16(
    device: &CudaDevice,
    source: &str,
) -> Result<(Arc<CudaStream>, CudaFunction), CudaError> {
    let arch = device.arch();
    let ptx = compile_ptx(source, arch)?;
    let mma_f16 = device
        .context()
        .load_module(ptx)?
        .load_function("gemm_mma_f16")?;
    Ok((device.stream().clone(), mma_f16))
}

/// [`MIN_COMPUTE_CAPABILITY_MAJOR`]（8.0）のゲート検査を行う（
/// [`CudaMmaGemm::new`]・[`CudaMmaGemm::new_with_swizzle`] 共通の前段）。
///
/// イシュー #499 で `new_with_swizzle` を追加した際に、両コンストラクタが
/// コンパイルするカーネルソースが異なる（`MMA_F16` 定数 vs
/// `kernels_mma::mma_f16_source_with_swizzle` が生成する変種）一方で cc
/// ゲート自体は同一（ブロックタイル定数・命令選定はどちらも
/// `MMA_M`/`MMA_N`/`MMA_K`/`cp.async`/`ldmatrix` を使う同じ命令セットの
/// ため）であることから、重複を避けて切り出した。
///
/// `pub(crate)`: `gemm_auto.rs::SpecializedMmaKernelHandle::compile` も
/// 同じ `mma.sync`/`ldmatrix`/`cp.async` 命令セットを NVRTC コンパイルする
/// ため同一ゲートを再利用する（PR #685 Bugbot 指摘〈Low〉・codex-review
/// 指摘への対応: `SpecializedMmaKernelHandle::compile` が `CudaMmaGemm::new`
/// と同じカーネルを構築するにもかかわらず cc ゲートを欠いており、旧世代
/// GPU 上で `CudaError::TensorCoreUnsupported` の代わりに不透明な NVRTC
/// コンパイル／起動失敗になっていた）。
pub(crate) fn check_min_compute_capability(device: &CudaDevice) -> Result<(), CudaError> {
    let (major, minor) = device.compute_capability();
    if major < MIN_COMPUTE_CAPABILITY_MAJOR {
        return Err(CudaError::TensorCoreUnsupported {
            detail: format!(
                "mma.sync/ldmatrix/cp.async path requires compute capability \
                 >= {MIN_COMPUTE_CAPABILITY_MAJOR}.0 (cp.async/ldmatrix are \
                 LDGSTS-only, sm_80+), but device reports {major}.{minor}"
            ),
        });
    }
    Ok(())
}

impl CudaMmaGemm {
    /// `device` 上で `mma.sync`/`ldmatrix`/`cp.async` GEMM カーネルを
    /// NVRTC コンパイルし保持するハンドルを構築する（**本番既定
    /// コンストラクタ**。base カーネルに加え、**サイズ条件付き適用**の
    /// L2 再利用スウィズル変種（イシュー #499・#775・#782）を結線する）。
    ///
    /// 手順: (1) `device.compute_capability()` が
    /// [`MIN_COMPUTE_CAPABILITY_MAJOR`]（8.0）未満なら NVRTC コンパイルを
    /// 試みず `CudaError::TensorCoreUnsupported` を返す（`gemm_wmma.rs::new`
    /// と同じ判断。cc 判定をコンパイル前に行うことで、非対応デバイス上での
    /// 無駄な NVRTC 呼び出し・コンパイル失敗の紛れ込みを避ける）。(2)
    /// `kernels_mma::mma_f16_source()`（swizzle 無適用の base カーネル）を
    /// `device.arch()` 向けに `nvrtc::compile_ptx` でコンパイルする。(3)
    /// `device.context().load_module()` → `load_function("gemm_mma_f16")`。
    /// `libnvrtc` 不在時は `CudaError::NvrtcUnavailable` を返す
    /// （`compile_ptx` のプローブゲートを経由。panic しない。本セッションの
    /// 実行環境がまさにこの分岐——CUDA driver はあるが NVRTC はない——で
    /// あり、`tests/gemm_mma.rs` の環境適応テストで確認済み。
    /// `kernels_mma.rs` 冒頭「検証状態」参照）。(4)
    /// `device.multiprocessor_count()` が実測 SM 数を返せた場合、
    /// [`crate::swizzle::select_swizzle_group_width`]（動的幅選択）で
    /// グルーピング幅を決め、`kernels_mma::mma_f16_source_with_swizzle`
    /// 変種を追加コンパイルして `mma_f16_swizzle` に保持する。
    ///
    /// **fail-soft 契約**: SM 数を取得できなかった場合、または変種の
    /// ソース生成・NVRTC コンパイル自体が失敗した場合は、いずれも本関数
    /// 全体を `Err` にはせず fail-soft に base カーネルのみを保持する
    /// （`mma_f16_swizzle = None`。swizzle は base カーネルの可用性とは
    /// 独立な L2 再利用の性能最適化に過ぎないため。`gemm_wmma.rs::
    /// CudaWmmaGemm::new` の `wmma_f16_opt` 分岐と同型の判断）。コンパイル
    /// 失敗時の理由は [`swizzle_unavailable_reason`
    /// ](Self::swizzle_unavailable_reason) から読める。
    ///
    /// `launch_f16` は呼び出し形状ごとに [`crate::swizzle::
    /// should_apply_swizzle`]（総ブロックタイル数
    /// `num_m_blocks * num_n_blocks >= 2048`。イシュー #775 のユーザー
    /// 起票の受け入れ条件〈4096 級のみ適用・512〜2048 は劣化 5% 以内〉を
    /// 承認記録として採用。`docs/perf/cuda-gemm-swizzle-ab.md` §4）で
    /// `mma_f16`（base）／`mma_f16_swizzle`（変種）のいずれを起動するか
    /// 判定する。個別呼び出しで実際に適用されたかは
    /// [`swizzle_applies`](Self::swizzle_applies) で確認できる。SM 数は
    /// `device.multiprocessor_count()` の実測値を動的に使うため、CI 恒久
    /// 検査（`swizzle.rs`）のハードコード値誤用は本コンストラクタの判定に
    /// 影響しない（#758 差し戻し理由(c)の解消）。明示幅指定・強制適用が
    /// 必要な場合（A/B 計測用途）は
    /// [`new_with_swizzle`](Self::new_with_swizzle) を使う。
    ///
    /// **本番結線の経緯**: イシュー #740 で一度無条件適用として本番結線
    /// し、PR #758 レビュー指摘（採用基準の無承認読み替え・結線前必須確認
    /// 〈レジスタスピル・bit 一致・parity 非後退〉未実施・CI 恒久検査の
    /// SM 数入力誤り）により差し戻した。イシュー #775 でサイズ条件付き
    /// 適用ロジック自体は実装したが、結線前必須確認が実機（GB10）到達可能
    /// なセッションで未実施だったため、本コンストラクタへの結線は見送り
    /// `new_with_size_conditional_swizzle`（`internal-diagnostics` feature
    /// 限定の実機検証専用入口）に留めていた。イシュー #782 で 2026-08-21
    /// の GB10 実機再計測（A/B: 4096 で ×1.592・512〜2048 劣化 5% 以内・
    /// bit 一致テスト ok）を根拠にユーザー承認のもと本コンストラクタへ
    /// 結線し `new_with_size_conditional_swizzle`（重複となるため）を
    /// 廃止した。**parity 非後退確認・結線後の `cuda_floor_bench` 実測
    /// （≥50 TFLOPS 確認）・レジスタスピル確認はイシュー #782 の受け入れ
    /// 条件チェックリストで「マージ後確認可」と明記された未解消のマージ後
    /// 確認事項として残る**（`docs/perf/cuda-gemm-swizzle-ab.md` §6.2
    /// 参照）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        // カーネル定数の内部整合性（静的共有メモリの 48KiB 上限・`MMA_BK`
        // の `MMA_K` 整除性・`MMA_STAGES >= 2`）は `kernels_mma.rs` の
        // `const _: () = assert!(...)` でコンパイル時に検査済み（実機
        // コンパイルできない本セッションでも `cargo build` の時点で機械
        // 検出できる代替チェック。本ファイル冒頭コメント参照）。
        //
        // #492 でカーネルソース側の wait_group を「ループ内固定即値
        // （`"n"(STAGES - 2)`）＋ループ外 drain」構造へ整理したことにより、
        // ここでの追加確認はカーネルソース中のハードコード数字即値
        // （旧 `cp.async.wait_group 1;`）との対応検査ではなくなった
        // （ループ内 wait はもはや `MMA_STAGES` 由来の数字即値を持たず、
        // `"n"` 制約を通じてカーネル側が自身で `STAGES - 2` を計算する
        // ため）。かつてここには Rust 側 `MMA_WAIT_GROUP_IMMEDIATE` 定数を
        // その定義式自身（`MMA_STAGES - 2`）と比較するだけの
        // `debug_assert_eq!` があったが、常に真となるトートロジーで
        // 検査価値がなかったため定数ごと撤去した（#492 レビュー指摘）。
        // 非負性（`STAGES - 2` の u32 アンダーフロー回避）は上記
        // コンパイル時 assert（`MMA_STAGES >= 2`）が既に担保している。
        // 段数非依存になった今でも `MMA_K_STEPS_PER_STAGE` はカーネル内
        // `for (int kstep = 0; kstep < BK / MMA_K; ++kstep)` に対応する
        // Rust 側唯一の真実源であり続けるため、引き続き実利用しておく
        // （`kernels_mma.rs` 冒頭ドキュメントコメント参照）。
        debug_assert_eq!(kernels_mma::MMA_K_STEPS_PER_STAGE, 2);

        check_min_compute_capability(device)?;

        let (stream, mma_f16) = compile_mma_f16(device, kernels_mma::mma_f16_source())?;

        // SM 数が実測できた場合のみ swizzle 変種を追加コンパイルする。
        // swizzle はあくまで L2 再利用の性能最適化であり base カーネルの
        // 可用性とは独立であるべきため、ソース生成・コンパイルいずれの
        // 失敗も本関数全体の `Err` へ波及させず fail-soft に
        // `(mma_f16_swizzle: None, swizzle_group_width: None)` へ縮退する
        // （`gemm_wmma.rs::CudaWmmaGemm::new` の `wmma_f16_opt` 分岐と同型
        // の判断。構造体ドキュメンテーションコメント「mma_f16_swizzle」
        // 節参照）。失敗理由は [`swizzle_unavailable_reason`
        // ](Self::swizzle_unavailable_reason) 経由でテストから参照できる
        // ようにする。
        let (mma_f16_swizzle, swizzle_group_width, swizzle_compile_error) =
            match device.multiprocessor_count() {
                Some(num_sms) => {
                    let group_width = crate::swizzle::select_swizzle_group_width(
                        num_sms,
                        kernels_mma::MMA_BM,
                        kernels_mma::MMA_BN,
                    );
                    match kernels_mma::mma_f16_source_with_swizzle(group_width)
                        .and_then(|swizzle_src| compile_mma_f16(device, &swizzle_src))
                    {
                        Ok((_swizzle_stream, swizzle_func)) => {
                            (Some(swizzle_func), Some(group_width), None)
                        }
                        Err(err) => (None, None, Some(err.to_string())),
                    }
                }
                None => (None, None, None),
            };

        Ok(Self {
            stream,
            mma_f16,
            mma_f16_swizzle,
            swizzle_group_width,
            swizzle_compile_error,
        })
    }

    /// `device` 上で、swizzle 変種と同じ [`check_min_compute_capability`]・
    /// NVRTC コンパイル手順で、**swizzle remap を適用しない base カーネル**
    /// （`kernels_mma::mma_f16_source()`）を NVRTC コンパイルし保持する
    /// ハンドルを構築する。
    ///
    /// A/B 計測（`examples/gemm_mma_swizzle_bench.rs`）・bit 一致検証
    /// （本ファイル `mod tests` の swizzle 変種比較テスト）が、
    /// [`new`](Self::new)（デバイス依存で `mma_f16_swizzle` が
    /// `Some`/`None` いずれにもなりうる。イシュー #782 でサイズ条件付き
    /// 適用を結線済み）とは独立に、**常に**swizzle 無適用の base カーネル
    /// へアクセスするための明示的な入口。`mma_f16_swizzle` は持たない
    /// （`launch_f16` は常に `mma_f16`〈base〉を起動する）。
    ///
    /// **`internal-diagnostics` feature（既定 off）でのみコンパイルされる**
    /// （[`new_with_swizzle`](Self::new_with_swizzle) と同じ理由・同じ
    /// feature ゲート方針。通常ビルドの公開 API 面からは除外する）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn new_without_swizzle(device: &CudaDevice) -> Result<Self, CudaError> {
        check_min_compute_capability(device)?;

        let (stream, mma_f16) = compile_mma_f16(device, kernels_mma::mma_f16_source())?;

        Ok(Self {
            stream,
            mma_f16,
            mma_f16_swizzle: None,
            swizzle_group_width: None,
            swizzle_compile_error: None,
        })
    }

    /// `device` 上で、L2 再利用のためのタイル→SM 割り当てスウィズル
    /// （イシュー #499・`kernels_mma::mma_f16_source_with_swizzle`）を
    /// 明示指定の `group_width` で**強制適用**した変種カーネルを NVRTC
    /// コンパイルし保持するハンドルを構築する（**診断用・明示幅指定の
    /// 入口**。[`new`](Self::new)（本番既定コンストラクタ。イシュー #782 で
    /// サイズ条件付き適用〈`should_apply_swizzle` の閾値判定〉を結線済み）
    /// とは異なり、本コンストラクタは形状・SM 数の判定を経ずに指定幅を
    /// 全サイズへ強制適用するため、A/B 計測・bit 一致検証で候補幅
    /// `{8, 16}` を個別に指定・強制適用したい場合の用途に限定される）。
    /// `mma_f16_swizzle` は持たない（swizzle 変種そのものを `mma_f16` に
    /// 格納し、`launch_f16` はサイズ判定を経ずに常にそれを起動する）。
    ///
    /// [`new`](Self::new) と同じ cc ゲート・NVRTC コンパイル手順を共有し
    /// （[`check_min_compute_capability`]）、コンパイルするソース文字列
    /// のみが `kernels_mma::MMA_F16`（変更なし）から
    /// `kernels_mma::mma_f16_source_with_swizzle(group_width)`（M 方向
    /// ブロック割り当てを remap した変種）へ変わる。返す
    /// [`CudaMmaGemm`] は [`new`](Self::new) が返すものと同一の型・API
    /// （`run_f16`／`upload_f16`／`launch_f16`／`download_f16`）を持ち、
    /// grid/block 構成・形状検証・SAFETY 根拠はブロックタイル定数
    /// （`MMA_BM`/`MMA_BN`/`MMA_BK`）を変更しないため共有できる（swizzle
    /// はブロックがどの `(m_block, n_block)` を担当するかの割り当てのみを
    /// 変え、各出力要素のアキュムレート順序・ブロックあたりの計算内容は
    /// 変えない）。
    ///
    /// 任意ソースを受ける公開 API（`new_with_source` 型）は意図的に作らず、
    /// `group_width: u32` のみを受けてカプセル化する（`kernels_mma.rs`
    /// 側で固定文字列アンカーの `replacen` と数値 `format!` 埋め込みのみを
    /// 行う契約を維持し、外部入力を直接カーネルソースへ流し込む経路を
    /// 作らない。`.claude/rules/security.md` A03 インジェクション対策）。
    /// `group_width < 2` は `kernels_mma::mma_f16_source_with_swizzle` が
    /// `CudaError::InvalidShape` で拒否する。
    ///
    /// **`internal-diagnostics` feature（既定 off）でのみコンパイルされる**
    /// （`Cargo.toml` の `[features]` 参照。PR #667 codex-review P1 是正:
    /// `CudaMmaGemm` 自体は `run_f16` 等の安定 API を持つ常時公開の型だが、
    /// 本コンストラクタが返す「未計測の実験カーネル変種」だけは
    /// `lib.rs::diagnostics` モジュールと同じ feature ゲート方針で通常
    /// ビルドの公開 API 面から除外する。ゲートしない場合、doc comment
    /// 上で「opt-in／本番経路から到達不能」と謳っていても、通常ビルドの
    /// crate 外部利用者が feature 指定なしに直接呼べてしまい実態と矛盾
    /// する）。`examples/gemm_mma_swizzle_bench.rs`（`Cargo.toml` の
    /// `required-features` で同 feature を要求）専用の入口であり、
    /// イシュー #782 で本番結線が完了した [`new`](Self::new) とは独立に、
    /// 明示幅指定の A/B 計測用途専用として feature ゲートを維持し続ける
    /// （`docs/perf/cuda-gemm-swizzle-ab.md` 参照）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn new_with_swizzle(device: &CudaDevice, group_width: u32) -> Result<Self, CudaError> {
        check_min_compute_capability(device)?;

        let src = kernels_mma::mma_f16_source_with_swizzle(group_width)?;
        let (stream, mma_f16) = compile_mma_f16(device, &src)?;

        Ok(Self {
            stream,
            mma_f16,
            mma_f16_swizzle: None,
            swizzle_group_width: Some(group_width),
            swizzle_compile_error: None,
        })
    }

    /// L2 再利用スウィズル（イシュー #499・#740・#775・#782）の適用
    /// グルーピング幅を返す（構造体ドキュメンテーションコメント参照）。
    /// `Some(_)` はこのハンドルが swizzle 変種カーネルを保持していることを
    /// 意味する: [`new`](Self::new)（本番既定コンストラクタ。`device.
    /// multiprocessor_count()` の実測に成功した場合。サイズ条件付き適用の
    /// ためこの `Some` は「その形状で必ず適用される」ことまでは意味しない
    /// — 個別呼び出しでの適用有無は
    /// [`swizzle_applies`](Self::swizzle_applies) を使う）・
    /// [`new_with_swizzle`](Self::new_with_swizzle)（診断用・明示幅指定・
    /// 強制適用。`internal-diagnostics` feature 限定）のいずれか。`None`
    /// は [`new`](Self::new) が SM 数を取得できず安全側で base のみを
    /// 保持した場合（fail-soft）、または
    /// [`new_without_swizzle`](Self::new_without_swizzle)（診断用・
    /// 強制非適用）を意味する。
    ///
    /// `examples/cuda_floor_bench.rs` の起動時診断（#733 の
    /// `wmma_tf32_staged` 可用性出力と同型）が、現在選択されている値を
    /// 可観測にするために呼ぶ。feature 非依存の常時公開 API（`new` 自体が
    /// feature 非依存のため）。
    pub fn swizzle_group_width(&self) -> Option<u32> {
        self.swizzle_group_width
    }

    /// [`new`](Self::new)（本番既定コンストラクタ）が
    /// `device.multiprocessor_count()` の実測に成功した（＝ swizzle 変種を
    /// 試みた）にもかかわらず、ソース生成・NVRTC コンパイルに失敗し
    /// swizzle 変種を保持できなかった場合の理由文字列
    /// （構造体ドキュメンテーションコメント「swizzle_compile_error」参照。
    /// `gemm_wmma.rs::CudaWmmaGemm::wmma_f16_opt_unavailable_reason` と
    /// 同型）。swizzle 変種を保持している場合・SM 数自体が取得できず
    /// 試みなかった場合は `None`。
    pub fn swizzle_unavailable_reason(&self) -> Option<&str> {
        self.swizzle_compile_error.as_deref()
    }

    /// `run_f16`/`launch_f16` が形状 `(m, n)` に対して実際に swizzle 変種
    /// カーネルを起動するかを返す（構造体ドキュメンテーションコメント
    /// 「mma_f16_swizzle」参照。イシュー #775・#782）。
    ///
    /// 判定規則（[`should_launch_swizzle_kernel`](Self::should_launch_swizzle_kernel)
    /// と単一の真実源を共有）:
    /// - [`new`](Self::new)（本番既定コンストラクタ）経由・
    ///   `mma_f16_swizzle` が `Some`（SM 数実測に成功しサイズ条件付き
    ///   変種を保持している）: [`crate::swizzle::should_apply_swizzle`] を
    ///   `(m, n)` から導出したブロックタイル数（`MMA_BM`/`MMA_BN` 単位の
    ///   `div_ceil`。`mma_launch_config` の grid 次元と同じ導出式）へ
    ///   適用した結果
    /// - [`new_with_swizzle`](Self::new_with_swizzle) 経由（強制適用。
    ///   `mma_f16_swizzle` は `None` だが `swizzle_group_width` が
    ///   `Some`）: 形状に関わらず常に `true`
    /// - [`new_without_swizzle`](Self::new_without_swizzle)・SM 数未取得時の
    ///   [`new`](Self::new)（いずれも強制非適用。両フィールドとも
    ///   `None`）: 形状に関わらず常に `false`
    ///
    /// `examples/cuda_floor_bench.rs` の起動時診断が、判定対象サイズ
    /// （512/1024/2048/4096）ごとの適用有無を出力するために呼ぶ。
    pub fn swizzle_applies(&self, m: u32, n: u32) -> bool {
        let num_m_blocks = m.div_ceil(kernels_mma::MMA_BM);
        let num_n_blocks = n.div_ceil(kernels_mma::MMA_BN);
        self.should_launch_swizzle_kernel(num_m_blocks, num_n_blocks)
    }

    /// ブロックタイル数 `(num_m_blocks, num_n_blocks)`（`mma_launch_config`
    /// の grid 次元 `(grid_dim.1, grid_dim.0)` と同じ導出式）から、
    /// `launch_f16` が起動すべきカーネルが swizzle 変種か base かを判定
    /// する共通ロジック（[`swizzle_applies`](Self::swizzle_applies) と
    /// `launch_f16` の両方が参照する単一の真実源）。
    ///
    /// `mma_f16_swizzle` が `Some`（[`new`](Self::new)（本番既定
    /// コンストラクタ）が SM 数実測に成功しサイズ条件付き変種を保持して
    /// いる）場合は [`crate::swizzle::should_apply_swizzle`] で判定し、
    /// `None` の場合は `swizzle_group_width` の有無で強制適用
    /// （[`new_with_swizzle`](Self::new_with_swizzle)。`mma_f16` 自体が
    /// swizzle 変種）／強制非適用（[`new_without_swizzle`
    /// ](Self::new_without_swizzle)・SM 数未取得時の `new`。`mma_f16` は
    /// base）のいずれかを返す。
    fn should_launch_swizzle_kernel(&self, num_m_blocks: u32, num_n_blocks: u32) -> bool {
        match &self.mma_f16_swizzle {
            Some(_) => crate::swizzle::should_apply_swizzle(num_m_blocks, num_n_blocks),
            None => self.swizzle_group_width.is_some(),
        }
    }

    /// f16 `mma.sync`/`ldmatrix`/`cp.async` GEMM を実行する。C = A @ B
    /// （`m x k` @ `k x n`）。入出力は `half::f16`、GPU 内部アキュムレートは
    /// f32（`kernels_mma::mma_f16_source()` 参照。数値契約は `CudaWmmaGemm::run_f16`
    /// と同一）。
    ///
    /// ホスト側形状検証を 3 段で行う: `validate_gemm_dims`（naive/tiled/WMMA
    /// と共通の一般契約。スライス長の整合性のみを見るため no-op 形状でも
    /// 常に先行させる）→ no-op 形状（`m==0 || n==0 || k==0`）の早期
    /// return → [`validate_mma_alignment`]／[`validate_mma_grid_bounds`]
    /// （本経路固有の `cp.async` 16 バイト整列制約・grid_dim.y 上限）。
    ///
    /// 整列検証・grid 上限検証を no-op 判定より後に置く（PR #255 レビュー
    /// 指摘）: 例えば `(m,n,k)=(8,7,0)` のような有効な no-op 形状は
    /// `n=7` が 8 の倍数でないため、整列検証を先に行うと実際には
    /// カーネルを起動しない形状まで誤って `CudaError::InvalidShape` で
    /// 拒否してしまう。整列・grid 上限はいずれもカーネル起動時にのみ
    /// 意味を持つ制約であるため、起動しないと決まった時点（no-op 判定）
    /// より後で検証すれば十分。
    ///
    /// グリッド次元は `kernels_mma::MMA_BM`/`MMA_BN` 単位の `div_ceil` で
    /// 構築し、末尾タイルの余剰はカーネル内 REQ-8 境界チェック
    /// （`kernels_mma.rs` 参照）に委ねる。
    pub fn run_f16(
        &self,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f16>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;

        // m==0/n==0（0 次元 grid はドライバが拒否する。`gemm.rs::run_f32_kernel`
        // ・`gemm_wmma.rs::run_f16` と同じ根拠）は起動自体を回避する。
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        // k==0 は A/B が空スライスになるため起動を回避し C = 全 0 を返す
        // （`gemm_wmma.rs::run_f16` の k==0 早期 return と同一契約）。
        if k == 0 {
            return Ok(vec![f16::ZERO; (m as usize) * (n as usize)]);
        }

        validate_mma_alignment(n, k)?;
        validate_mma_grid_bounds(m)?;

        let (a_dev, b_dev) = self.upload_f16(a, b)?;
        let mut c_dev = self.alloc_output_f16(m, n)?;
        self.launch_f16(&a_dev, &b_dev, &mut c_dev, m, n, k)?;
        self.download_f16(&c_dev)
    }

    /// A・B をホスト→デバイスへ転送する（`run_f16` の H2D 部分の切り出し。
    /// ベンチマークが GPU 実行時間のみを計測できるよう、転送とカーネル
    /// 実行を分離する。PR #255 レビュー指摘 —— `examples/gemm_mma_bench.rs`
    /// が転送・バッファ確保込みで TFLOPS を算出していた問題への対処）。
    pub fn upload_f16(
        &self,
        a: &[f16],
        b: &[f16],
    ) -> Result<(CudaSlice<f16>, CudaSlice<f16>), CudaError> {
        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        Ok((a_dev, b_dev))
    }

    /// C 用のゼロ初期化デバイスバッファを確保する（`run_f16` のバッファ
    /// 確保部分の切り出し。[`upload_f16`] と同じ理由でベンチマークから
    /// 再利用できるよう公開する）。
    pub fn alloc_output_f16(&self, m: u32, n: u32) -> Result<CudaSlice<f16>, CudaError> {
        Ok(self
            .stream
            .alloc_zeros::<f16>((m as usize) * (n as usize))?)
    }

    /// デバイス常駐済みの A/B/C バッファに対してカーネルを起動し、完了を
    /// 待つ（H2D/D2H を含まない「GPU 実行のみ」の区間。[`upload_f16`]・
    /// [`alloc_output_f16`] と組み合わせてベンチマークの計測対象を絞る
    /// ために公開する）。
    ///
    /// safe な公開 API であるため、呼び出し元（`run_f16` あるいは
    /// ベンチマーク）の事前検証に依存せず、本関数自身が `run_f16` と同じ
    /// 形状検証（`validate_gemm_dims`・[`validate_mma_alignment`]・
    /// [`validate_mma_grid_bounds`]）およびデバイスバッファ長検証
    /// （`a_dev`/`b_dev`/`c_dev`）を行う（PR #349 codex-review 指摘 P0。
    /// `gemm.rs::launch_tiled_f32` のドキュメンテーションコメント参照。
    /// `gemm_wmma.rs::launch_f16` には同種の指摘があったが本関数は指摘に
    /// 明示されていなかった — 同一パターンの脆弱性のため一貫して修正する）。
    ///
    /// カーネル選択（イシュー #775）: [`should_launch_swizzle_kernel`
    /// ](Self::should_launch_swizzle_kernel) の判定に従い、`mma_f16`
    /// （base）または `mma_f16_swizzle`（swizzle 変種。`Some` の場合の
    /// み）のいずれかを起動する（[`swizzle_applies`](Self::swizzle_applies)
    /// と単一の真実源を共有）。base／swizzle いずれの変種もブロックタイル
    /// 定数・引数構成・カーネル内 REQ-8 境界チェックを共有しており
    /// （swizzle はブロックがどの `(m_block, n_block)` を担当するかの
    /// 割り当てのみを変える。`kernels_mma.rs::mma_f16_source_with_swizzle`
    /// ドキュメンテーションコメント参照）、下記 SAFETY 根拠は両者で共通。
    pub fn launch_f16(
        &self,
        a_dev: &CudaSlice<f16>,
        b_dev: &CudaSlice<f16>,
        c_dev: &mut CudaSlice<f16>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        validate_mma_alignment(n, k)?;
        validate_mma_grid_bounds(m)?;
        crate::gemm::validate_output_len(c_dev.len(), m, n)?;

        let cfg = mma_launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // カーネル選択（構造体ドキュメンテーションコメント
        // 「mma_f16_swizzle」・`should_launch_swizzle_kernel` 参照）。
        // grid_dim は mma_launch_config が (n.div_ceil(MMA_BN),
        // m.div_ceil(MMA_BM), 1) として構築しているため、(num_m_blocks,
        // num_n_blocks) = (grid_dim.1, grid_dim.0)。
        let (num_n_blocks, num_m_blocks) = (cfg.grid_dim.0, cfg.grid_dim.1);
        let kernel = if self.should_launch_swizzle_kernel(num_m_blocks, num_n_blocks) {
            // `mma_f16_swizzle` が `None`（`new_with_swizzle` 経由。
            // `mma_f16` 自体が swizzle 変種）の場合は `mma_f16` へ
            // フォールバックする（`should_launch_swizzle_kernel` の
            // ドキュメンテーションコメント参照）。
            self.mma_f16_swizzle.as_ref().unwrap_or(&self.mma_f16)
        } else {
            &self.mma_f16
        };

        // SAFETY: カーネル引数は a_dev/b_dev/c_dev（それぞれ a.len()/
        // b.len()/(m*n) 要素の確保済みデバイスバッファ）と m_i/n_i/k_i の
        // 5 個・型・個数が、上記で検証済みの m/n/k と 1:1 対応する。
        // カーネル内の手動境界チェック
        // （cp.async src-size ゼロ充填・エピローグ guarded store。
        // kernels_mma.rs 参照、REQ-8）と合わせて OOB 読み書きが起きない
        // 根拠とする。グリッド次元は MMA_BM/MMA_BN 単位の div_ceil で
        // m/n を包含するよう構築しており（mma_launch_config）、末尾タイル
        // の余剰はカーネル内境界チェックで弾かれる。共有メモリは静的
        // `__shared__` 配列のみを使用するため `shared_mem_bytes` は 0 の
        // ままでよい（`kernels_mma.rs` 冒頭コメント「タイル構成」の
        // 41,472B〈#494 のブロックタイル拡大後・#498 のバンクコンフリクト
        // 対策パディング適用後の値〉は per-block 静的上限 48KiB 内であり
        // 動的共有メモリの追加確保・`cudaFuncSetAttribute` opt-in は
        // 不要）。上記のカーネル選択は `blockIdx`→`(m_block, n_block)`
        // remap の有無のみが異なり、引数構成・境界チェック契約は base・
        // swizzle 変種間で同一（本関数ドキュメンテーションコメント参照）。
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(a_dev)
                .arg(b_dev)
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// C をデバイス→ホストへ転送する（`run_f16` の D2H 部分の切り出し。
    /// [`upload_f16`] と同じ理由で公開する）。
    pub fn download_f16(&self, c_dev: &CudaSlice<f16>) -> Result<Vec<f16>, CudaError> {
        Ok(self.stream.clone_dtoh(c_dev)?)
    }
}

/// mma カーネル専用のグリッド次元計算。1 ブロック = C の `MMA_BM x MMA_BN`
/// タイル 1 個を担当するため、`div_ceil(n, MMA_BN)` x `div_ceil(m, MMA_BM)`
/// のグリッドを構築する（`gemm_wmma.rs::wmma_launch_config` と同じ設計。
/// `gemm.rs::launch_config` の「1 スレッド = C の 1 要素」前提とは異なる
/// ため独立関数とする）。
fn mma_launch_config(m: u32, n: u32) -> LaunchConfig {
    let grid_dim = (
        n.div_ceil(kernels_mma::MMA_BN),
        m.div_ceil(kernels_mma::MMA_BM),
        1,
    );
    LaunchConfig {
        grid_dim,
        block_dim: MMA_BLOCK_DIM,
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mma_launch_config_grid_dim_covers_m_and_n_via_div_ceil() {
        // #494 でブロックタイルを MMA_BM=64/MMA_BN=128 へ拡大。
        // 65x129 を覆うには grid (2, 2) が必要
        // （div_ceil(129,128)=2, div_ceil(65,64)=2）。
        let cfg = mma_launch_config(65, 129);
        assert_eq!(cfg.grid_dim, (2, 2, 1));
        assert_eq!(cfg.block_dim, MMA_BLOCK_DIM);
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn mma_launch_config_exact_multiple_shape_has_no_extra_tile() {
        // MMA_BM=64/MMA_BN=128 のちょうど 2 倍（128, 256）で余剰タイルが
        // 出ないことを検査する。
        let cfg = mma_launch_config(128, 256);
        assert_eq!(cfg.grid_dim, (2, 2, 1));
    }

    #[test]
    fn validate_mma_alignment_accepts_multiples_of_eight() {
        assert!(validate_mma_alignment(64, 32).is_ok());
        assert!(validate_mma_alignment(8, 8).is_ok());
    }

    #[test]
    fn validate_mma_alignment_rejects_non_multiple_n() {
        let err = validate_mma_alignment(9, 32).expect_err("n=9 is not a multiple of 8");
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_mma_alignment_rejects_non_multiple_k() {
        let err = validate_mma_alignment(64, 17).expect_err("k=17 is not a multiple of 8");
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_mma_grid_bounds_accepts_shapes_within_limit() {
        // MMA_BM（#494 時点で 64）単位: div_ceil(65_535 * MMA_BM, MMA_BM)
        // = 65_535（上限ちょうど）。定数参照のためタイル値変更時も自動追従。
        assert!(validate_mma_grid_bounds(65_535 * kernels_mma::MMA_BM).is_ok());
    }

    #[test]
    fn validate_mma_grid_bounds_rejects_m_exceeding_grid_y_limit() {
        // MMA_BM（#494 時点で 64）単位: div_ceil(65_535*MMA_BM + 1, MMA_BM)
        // = 65_536 > 65_535。
        let err = validate_mma_grid_bounds(65_535 * kernels_mma::MMA_BM + 1)
            .expect_err("grid_dim.y must exceed CUDA's 65,535 limit");
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    /// `validate_mma_alignment` 単体では k=0/n=7 は整列制約違反として
    /// 拒否される（この関数自体は no-op 形状かどうかを考慮しない）。
    /// `run_f16` が no-op 早期 return をこの検証より前に行うことで
    /// `(m,n,k)=(8,7,0)` のような有効な no-op 形状を誤って拒否しない
    /// 契約は、実機依存の統合テスト
    /// `tests/gemm_mma.rs::mma_f16_accepts_noop_shape_with_misaligned_n_when_k_is_zero`
    /// （`#[ignore]`）で確認する（PR #255 レビュー指摘）。
    #[test]
    fn validate_mma_alignment_rejects_misaligned_n_independent_of_noop_shape() {
        assert!(validate_mma_alignment(7, 0).is_err());
    }

    /// `#define STAGES <kernels_mma::MMA_STAGES>` 行のみを `stages` へ
    /// 書き換えたソースを NVRTC コンパイル・実行する（#492 §5-5 の
    /// 実機必須テスト専用ヘルパー。`kernels_mma.rs::tests::
    /// mma_f16_source_with_stages` と同じ置換方針だが、こちらは
    /// NVRTC コンパイル・カーネル起動まで踏み込む点が異なる）。
    ///
    /// `CudaMmaGemm::new`/`run_f16` を再利用しない理由: それらは常に
    /// `kernels_mma::mma_f16_source()`（`STAGES=3` 固定の文字列）をコンパイル
    /// する ため、段数を差し替えた変種を実行するには本関数のように NVRTC
    /// コンパイル・モジュールロード・起動を直接組み立てる必要がある。
    /// 形状検証（`validate_gemm_dims`・[`validate_mma_alignment`]・
    /// [`validate_mma_grid_bounds`]）・グリッド構築（[`mma_launch_config`]）・
    /// SAFETY 根拠は `launch_f16` と同一（段数はブロックタイル形状・
    /// grid/block 次元に影響しないため）。
    fn run_f16_with_stages(
        device: &CudaDevice,
        stages: u32,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f16>, CudaError> {
        let base_src = kernels_mma::mma_f16_source();
        let from = format!("#define STAGES {}\n", kernels_mma::MMA_STAGES);
        let to = format!("#define STAGES {stages}\n");
        assert_eq!(
            base_src.matches(&from).count(),
            1,
            "kernels_mma::mma_f16_source() 中の `{from:?}` の出現数が 1 ではありません \
             （run_f16_with_stages の前提が崩れています）"
        );
        let src = base_src.replacen(&from, &to, 1);

        let ptx = compile_ptx(&src, device.arch())
            .expect("stage-swapped MMA_F16 source must compile via NVRTC on real hardware");
        let func = device
            .context()
            .load_module(ptx)
            .expect("stage-swapped module load must succeed")
            .load_function("gemm_mma_f16")
            .expect("gemm_mma_f16 must be present in the stage-swapped module");

        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        validate_mma_alignment(n, k)?;
        validate_mma_grid_bounds(m)?;

        let stream = device.stream();
        let a_dev = stream.clone_htod(a)?;
        let b_dev = stream.clone_htod(b)?;
        let mut c_dev = stream.alloc_zeros::<f16>((m as usize) * (n as usize))?;

        let cfg = mma_launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: `launch_f16` と同一の引数構成（a_dev/b_dev/c_dev + m/n/k）
        // であり、段数の違いはカーネル内部の共有メモリ・パイプライン深さ
        // のみに影響し、引数の型・個数・対応関係は変わらない。REQ-8 の
        // 手動境界チェックも段数非依存（本ファイル冒頭コメント・
        // `kernels_mma.rs` 参照）。
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        stream.synchronize()?;
        Ok(stream.clone_dtoh(&c_dev)?)
    }

    /// #492 受け入れ基準（段数可変化）の実機検証: `stages ∈ {2, 3, 4}` の
    /// 各カーネル変種が、小形状・タイル端形状・大形状の全てで**ビット
    /// 一致**の出力を返すことを確認する。段数（パイプライン深さ）は
    /// cp.async の同期タイミングのみを変え、mma/ldmatrix の実行順序・
    /// アキュムレート順序は変えないため（`kernels_mma.rs` の `MMA_STAGES`
    /// 定数直下のドキュメンテーションコメント「正しさ」参照）、
    /// tolerance を使わない bit 等値で主張できる
    /// （`.claude/rules/coding-rust.md` の「バックエンド間数値一致テストの
    /// 許容誤差を単独で緩和しない」契約に抵触しない。段数間比較は
    /// バックエンド間比較ではなく同一バックエンド内の実装詳細比較の
    /// ため、tolerance の対象外）。
    ///
    /// `#[ignore]`: 本セッション（本ファイル冒頭コメント「検証状態」）は
    /// NVRTC 非搭載のため実行できない。DGX Spark GB10 等の実機で
    /// `cargo test -p backend-cuda --lib -- --ignored` から実行する
    /// （`gemm.rs::tests::wmma_tf32_basic_kernel_parity_does_not_regress`
    /// と同じ実行方法）。
    #[test]
    #[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
    fn mma_f16_stage_count_does_not_change_bit_exact_output() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");

        // (m, n, k): 小形状（16x8x16。単一 mma タイル）・タイル端形状
        // （40x72x160。#494 時点の MMA_BM=64/MMA_BN=128/MMA_BK=32 の
        // 非整数倍）・
        // 大形状（256x256x4096。B-0/#491 parity 非後退契約の mma_f16 行と
        // 同一形状。docs/perf/cuda-parity-baseline.md 参照）を横断する
        // （#492 実装計画 §5-5）。
        let shapes: [(u32, u32, u32); 3] = [(16, 8, 16), (40, 72, 160), (256, 256, 4096)];
        let seed: u64 = 9999;

        for &(m, n, k) in &shapes {
            let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
            let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
            let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));

            let mut outputs: Vec<(u32, Vec<f16>)> = Vec::new();
            for stages in [2u32, 3u32, 4u32] {
                let c =
                    run_f16_with_stages(&device, stages, &a, &b, m, n, k).unwrap_or_else(|err| {
                        panic!(
                            "stages={stages} run_f16_with_stages failed for shape \
                             (m={m}, n={n}, k={k}): {err}"
                        )
                    });
                outputs.push((stages, c));
            }

            let (base_stages, base_c) = &outputs[0];
            for (stages, c) in &outputs[1..] {
                assert_eq!(
                    c, base_c,
                    "shape (m={m}, n={n}, k={k}): stages={stages} の出力が \
                     stages={base_stages} と bit 一致しません（段数変更が \
                     mma/ldmatrix の実行順序に影響していないか確認すること）"
                );
            }
        }
    }

    /// イシュー #782（GB10 実機ゲート通過を根拠にサイズ条件付き swizzle
    /// 選択機構を本番既定コンストラクタへ結線）の契約ピン留め: 本番既定
    /// コンストラクタ [`CudaMmaGemm::new`] は、旧 `new_with_size_
    /// conditional_swizzle`（`internal-diagnostics` feature 限定・実機検証
    /// 専用入口。本 PR で `new` へ統合・廃止）が持っていた fail-soft 3
    /// 分岐（SM 数取得失敗／取得成功&コンパイル成功／取得成功&コンパイル
    /// 失敗）と同一の契約を持つことを検査する（構造体ドキュメンテーション
    /// コメント「mma_f16_swizzle」参照）。いずれの分岐でも `new` 全体が
    /// `Err` へ波及しないこと（swizzle は base カーネルの可用性とは独立な
    /// 性能最適化であるという fail-soft 方針。`new` doc comment 参照）を
    /// `swizzle_group_width()`／`swizzle_unavailable_reason()` の組み合わせ
    /// で直接 assert する。`internal-diagnostics` feature 非依存（`new`・
    /// `swizzle_group_width`／`swizzle_unavailable_reason` はどちらも常時
    /// 公開 API）で成立するテストのため feature ゲートしない。
    ///
    /// `#[ignore]`: `CudaDevice::new` が CUDA 実機を要求するため
    /// （本ファイル冒頭コメント「検証状態」）。DGX Spark GB10 等の実機で
    /// `cargo test -p backend-cuda --lib -- --ignored` から実行する。
    #[test]
    #[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
    fn mma_f16_new_wires_size_conditional_swizzle_into_production_constructor() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
        let production = CudaMmaGemm::new(&device)
            .expect("CudaMmaGemm::new must succeed on ignored test runner");

        let expected_group_width_if_compile_succeeds =
            device.multiprocessor_count().map(|num_sms| {
                crate::swizzle::select_swizzle_group_width(
                    num_sms,
                    kernels_mma::MMA_BM,
                    kernels_mma::MMA_BN,
                )
            });

        match (
            device.multiprocessor_count(),
            production.swizzle_group_width(),
            production.swizzle_unavailable_reason(),
        ) {
            (None, actual_group_width, reason) => {
                // 分岐 (a): SM 数取得失敗。
                assert_eq!(
                    actual_group_width, None,
                    "分岐 (a)（SM 数取得失敗）では swizzle_group_width は None のはずです"
                );
                assert_eq!(
                    reason, None,
                    "分岐 (a)（SM 数取得失敗）では swizzle_unavailable_reason は None の \
                     はずです（コンパイル自体を試みていないため）"
                );
            }
            (Some(_), Some(actual_group_width), reason) => {
                // 分岐 (b): SM 数取得成功・コンパイル成功。
                assert_eq!(
                    Some(actual_group_width),
                    expected_group_width_if_compile_succeeds,
                    "分岐 (b)（SM 数取得成功・コンパイル成功）では swizzle_group_width が \
                     select_swizzle_group_width の動的選択幅と一致するはずです"
                );
                assert_eq!(
                    reason, None,
                    "分岐 (b)（コンパイル成功）では swizzle_unavailable_reason は None の \
                     はずです"
                );
            }
            (Some(_), None, reason) => {
                // 分岐 (c): SM 数取得成功・コンパイル失敗（fail-soft 縮退）。
                assert!(
                    reason.is_some(),
                    "分岐 (c)（SM 数取得成功・コンパイル失敗）では \
                     swizzle_unavailable_reason に失敗理由が記録されているはずです"
                );
            }
        }
    }

    /// #499/#775/#782 受け入れ基準（実機検証）: [`CudaMmaGemm::new_with_swizzle`]
    /// が生成する各 `group_width` の変種が、swizzle 無適用の
    /// [`CudaMmaGemm::new_without_swizzle`]（base）と**ビット一致**の
    /// 出力を返すことを確認する。[`CudaMmaGemm::new`]（本番既定
    /// コンストラクタ。イシュー #782 でサイズ条件付き適用機構を結線済み）
    /// についても、`should_launch_swizzle_kernel` の契約
    /// （`swizzle_group_width()` は SM 数実測に成功すれば `Some(_)`・
    /// 閾値未満形状は base 選択・閾値以上形状は swizzle 選択）を本テストで
    /// ピン留めする（旧: 実機検証専用の opt-in 入口
    /// `new_with_size_conditional_swizzle` を検証対象としていたが、イシュー
    /// #782 で同ロジックが `new` へ統合されたため、検証対象を `new` 自身へ
    /// 戻した）。
    ///
    /// swizzle はブロックがどの `(m_block, n_block)` を担当するかの割り当て
    /// のみを変え、各ブロック内部の計算（mma/ldmatrix の発行順序・
    /// アキュムレート順序）は変えないため（`kernels_mma.rs::
    /// mma_f16_source_with_swizzle` ドキュメンテーションコメント参照）、
    /// `mma_f16_stage_count_does_not_change_bit_exact_output` と同じ論法で
    /// tolerance を使わない bit 等値で主張できる（`.claude/rules/
    /// coding-rust.md` の「バックエンド間数値一致テストの許容誤差を単独で
    /// 緩和しない」契約に抵触しない。swizzle 変種間比較はバックエンド間
    /// 比較ではなく同一バックエンド内の実装詳細比較のため tolerance の
    /// 対象外）。
    ///
    /// `group_width` は [`crate::swizzle::select_swizzle_group_width`] の
    /// 動的選択結果（`device.multiprocessor_count()` 実測値ベース）に
    /// 加え、参考として固定候補 `8`/`16` も検査する（実装計画 4 節
    /// 「gemm_mma.rs（起動側）」）。
    ///
    /// `#[ignore]`: 本セッション（本ファイル冒頭コメント「検証状態」）は
    /// NVRTC 非搭載のため実行できない。DGX Spark GB10 等の実機で
    /// `cargo test -p backend-cuda --lib --features internal-diagnostics --
    /// --ignored` から実行する（`--features internal-diagnostics` を欠くと
    /// 下記の理由で本テスト自体がコンパイルされず green と誤認する。PR #667
    /// codex-review P1 是正。`docs/perf/cuda-gemm-swizzle-ab.md` の実機検証
    /// 手順コマンドも同時に是正済み）。
    ///
    /// `internal-diagnostics` feature（既定 off）でのみコンパイルされる
    /// （[`CudaMmaGemm::new_with_swizzle`] 自体が同 feature でゲートされて
    /// いるため）。`Makefile` の `test` ターゲットは `--all-features` の
    /// ため通常の `make test`（コンパイルのみ・`--ignored` なしでは実行
    /// されない）でも本 feature は有効。
    #[cfg(feature = "internal-diagnostics")]
    #[test]
    #[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
    fn mma_f16_swizzle_variant_matches_base_bit_exact_output() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
        let base = CudaMmaGemm::new_without_swizzle(&device)
            .expect("base CudaMmaGemm::new_without_swizzle must succeed on ignored test runner");
        let production = CudaMmaGemm::new(&device)
            .expect("CudaMmaGemm::new must succeed on ignored test runner");

        // `new`（本番既定コンストラクタ）の fail-soft 3 分岐（3255823 で
        // 導入・イシュー #782 で `new` 自身へ統合）を、`swizzle_unavailable_
        // reason()`（doc comment「テストから参照できるようにする」の唯一の
        // 参照元）を使って明示的にピン留めする:
        //   (a) SM 数取得失敗                       → group_width=None・reason=None
        //   (b) SM 数取得成功 かつ 変種コンパイル成功 → group_width=Some(_)・reason=None
        //   (c) SM 数取得成功 かつ 変種コンパイル失敗 → group_width=None・reason=Some(_)
        // 旧テストは (a)/(b) のみを想定し
        // `production.swizzle_group_width() == expected_group_width`
        // （`expected_group_width` はコンパイル成否を考慮しない
        // `select_swizzle_group_width` の純計算）を assert していたため、
        // 実機で (c) が起きた場合に誤って失敗する契約になっていた
        // （Review 指摘）。
        let expected_group_width_if_compile_succeeds =
            device.multiprocessor_count().map(|num_sms| {
                crate::swizzle::select_swizzle_group_width(
                    num_sms,
                    kernels_mma::MMA_BM,
                    kernels_mma::MMA_BN,
                )
            });
        match (
            device.multiprocessor_count(),
            production.swizzle_group_width(),
            production.swizzle_unavailable_reason(),
        ) {
            (None, actual_group_width, reason) => {
                // 分岐 (a)
                assert_eq!(
                    actual_group_width, None,
                    "分岐 (a)（SM 数取得失敗）では swizzle_group_width は None のはずです"
                );
                assert_eq!(
                    reason, None,
                    "分岐 (a)（SM 数取得失敗）では swizzle_unavailable_reason は None の \
                     はずです（コンパイル自体を試みていないため）"
                );
            }
            (Some(_), Some(actual_group_width), reason) => {
                // 分岐 (b)
                assert_eq!(
                    Some(actual_group_width),
                    expected_group_width_if_compile_succeeds,
                    "分岐 (b)（SM 数取得成功・コンパイル成功）では swizzle_group_width が \
                     select_swizzle_group_width の動的選択幅と一致するはずです"
                );
                assert_eq!(
                    reason, None,
                    "分岐 (b)（コンパイル成功）では swizzle_unavailable_reason は None の \
                     はずです"
                );
            }
            (Some(_), None, reason) => {
                // 分岐 (c): 実機の NVRTC 状態次第で発生しうる縮退。
                // fail-soft 契約どおり Err へ波及していないこと・理由が
                // 記録されていることのみを検査する（コンパイル失敗の
                // 発生自体を本テストで強制はできないため）。
                assert!(
                    reason.is_some(),
                    "分岐 (c)（SM 数取得成功・コンパイル失敗）では \
                     swizzle_unavailable_reason に失敗理由が記録されているはずです"
                );
            }
        }

        let num_sms = device.multiprocessor_count().unwrap_or(1).max(1);
        let dynamic_group_width = crate::swizzle::select_swizzle_group_width(
            num_sms,
            kernels_mma::MMA_BM,
            kernels_mma::MMA_BN,
        );

        // (m, n, k): 小形状（16x8x16。単一 mma タイル。grid は 1x1 のため
        // swizzle remap は恒等的に自明）・タイル端形状（80x136x160。
        // MMA_BM=64/MMA_BN=128/MMA_BK=32 に対し
        // grid=(n.div_ceil(MMA_BN), m.div_ceil(MMA_BM))=(2, 2) となり
        // （`mma_launch_config`。両軸とも非整数倍の端数タイルを含む）
        // remap が非自明に効く。旧値 40x72x160 は grid=(1,1) となり remap
        // が恒等写像に縮退していたため是正した〈Bugbot 指摘・PR #667
        // レビュー是正〉）・full_groups 分岐形状（1088x256x2048。
        // `swizzled_block_idx`〈`swizzle.rs`〉は `num_m_blocks` を
        // `group_width` 単位でグルーピングし、`full_groups`（`group_width`
        // 個ぴったり埋まるグループ）と末尾の縮小 `remainder` グループの
        // 2 分岐を持つ。旧値 256x256x2048 は
        // num_m_blocks=m.div_ceil(MMA_BM)=4 のため候補幅 group_width∈
        // {8, 16}（`swizzle::GROUP_WIDTH_CANDIDATES`）のいずれでも
        // full_groups=num_m_blocks/group_width=0 となり、生成 CUDA 側の
        // full_groups リマップ分岐（ベクトル化ロードの境界検査を伴う。
        // REQ-8）が一度も経由されず remainder 分岐のみ検査していた
        // 〈Bugbot 指摘・PR #667 レビュー是正〉。1088=17*MMA_BM により
        // num_m_blocks=17 となり、group_width=8 では full_groups=2・
        // remainder=1、group_width=16 では full_groups=1・remainder=1 と
        // なり両候補幅で full_groups・remainder の両分岐を経由する
        // （k=2048 は実装計画 5 節「実機（引き継ぎ）」の A/B 計測 k 値と
        // 揃える値をそのまま踏襲。4096 は本テストの目的〈bit 一致の
        // 確認〉には過大なため採らない）。
        let shapes: [(u32, u32, u32); 3] = [(16, 8, 16), (80, 136, 160), (1088, 256, 2048)];
        let seed: u64 = 424_242;

        // group_width=8/16 は候補表（swizzle.rs::select_swizzle_group_width
        // の候補）そのもの。dynamic_group_width が候補と一致する場合は
        // 重複計測になるが、テストの単純さを優先し de-dup はしない。
        for group_width in [dynamic_group_width, 8, 16] {
            let variant =
                CudaMmaGemm::new_with_swizzle(&device, group_width).unwrap_or_else(|err| {
                    panic!("group_width={group_width}: new_with_swizzle failed: {err}")
                });

            for &(m, n, k) in &shapes {
                let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
                let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
                let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));

                let base_c = base.run_f16(&a, &b, m, n, k).unwrap_or_else(|err| {
                    panic!("base run_f16 failed for shape (m={m}, n={n}, k={k}): {err}")
                });
                let variant_c = variant.run_f16(&a, &b, m, n, k).unwrap_or_else(|err| {
                    panic!(
                        "group_width={group_width} run_f16 failed for shape \
                         (m={m}, n={n}, k={k}): {err}"
                    )
                });

                assert_eq!(
                    variant_c, base_c,
                    "shape (m={m}, n={n}, k={k}) group_width={group_width}: swizzle \
                     変種の出力が base と bit 一致しません（remap がブロック内部の \
                     計算・アキュムレート順序に影響していないか確認すること）"
                );

                // production（CudaMmaGemm::new）が base と bit 一致する
                // ことを検査する（本テストの 3 形状はいずれも総ブロック
                // タイル数が `should_apply_swizzle` の閾値〈2048〉未満
                // ——16x8x16: 1・80x136x160: 2x2=4・1088x256x2048:
                // 17x2=34——のため常に base カーネルを選択する契約。
                // `swizzle_applies` の閾値未満側の分岐を検証する）。
                if group_width == dynamic_group_width {
                    assert!(
                        !production.swizzle_applies(m, n),
                        "shape (m={m}, n={n}, k={k}): 総ブロックタイル数が \
                         閾値未満のため swizzle_applies は false のはずです"
                    );
                    let production_c = production.run_f16(&a, &b, m, n, k).unwrap_or_else(|err| {
                        panic!(
                            "CudaMmaGemm::new (production) run_f16 failed \
                             for shape (m={m}, n={n}, k={k}): {err}"
                        )
                    });
                    assert_eq!(
                        production_c, base_c,
                        "shape (m={m}, n={n}, k={k}): CudaMmaGemm::new（本番既定 \
                         コンストラクタ）の出力が base と bit 一致しません"
                    );
                }
            }
        }

        // 閾値以上（総ブロックタイル数 >= 2048）の形状で production
        // （CudaMmaGemm::new）の経路が実際に swizzle 変種へディスパッチ
        // することを検証する（実装計画 2 節「gemm_mma.rs」）。
        // m=n=4096・k=32（省メモリ）で
        // num_m_blocks=4096.div_ceil(MMA_BM=64)=64・
        // num_n_blocks=4096.div_ceil(MMA_BN=128)=32 → 総タイル数
        // 64*32=2048（`SWIZZLE_APPLY_TILE_COUNT_THRESHOLD` ちょうど）。
        // `production.swizzle_group_width()` が `Some`（fail-soft 分岐
        // (b)：SM 数取得成功・変種コンパイル成功）の場合のみ
        // `mma_f16_swizzle` が使用可能で本チェックが意味を持つため、
        // それ以外（分岐 (a)/(c)）はスキップする。
        if production.swizzle_group_width().is_some() {
            let (m, n, k) = (4096u32, 4096u32, 32u32);
            assert!(
                production.swizzle_applies(m, n),
                "shape (m={m}, n={n}, k={k}): 総ブロックタイル数が閾値以上の \
                 ため swizzle_applies は true のはずです"
            );

            let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
            let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
            let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));

            let base_c = base.run_f16(&a, &b, m, n, k).unwrap_or_else(|err| {
                panic!("base run_f16 failed for shape (m={m}, n={n}, k={k}): {err}")
            });
            let production_c = production.run_f16(&a, &b, m, n, k).unwrap_or_else(|err| {
                panic!(
                    "CudaMmaGemm::new (production) run_f16 failed for \
                     shape (m={m}, n={n}, k={k}): {err}"
                )
            });
            assert_eq!(
                production_c, base_c,
                "shape (m={m}, n={n}, k={k}): 閾値以上形状で CudaMmaGemm::new \
                 （本番既定コンストラクタ。swizzle 変種を選択するはず）の \
                 出力が base と bit 一致しません"
            );
        }
    }

    /// C-8（#521）受け入れ基準「B-1（#492）可変化カーネルとの接続実証」の
    /// 実機検証: 実デバイスの SMEM 予算から
    /// `gemm_auto::derive_stages_for_device` で段数を導出し、
    /// `run_f16_with_stages`（B-1 の段数可変化ヘルパー）でその導出値を
    /// 使ってカーネルを生成・実行し、既定の `MMA_STAGES=3` 構成の出力と
    /// bit 一致することを確認する。
    ///
    /// `mma_f16_stage_count_does_not_change_bit_exact_output` が段数
    /// `{2,3,4}` を固定リテラルで横断するのに対し、本テストは
    /// **導出ロジックが返した値**を実際にカーネル起動へ結線できることを
    /// 検証する点が異なる（実装計画 §4 の受け入れ基準対応表）。
    ///
    /// `#[ignore]`: 本セッション（本ファイル冒頭コメント「検証状態」）は
    /// NVRTC 非搭載のため実行できない。DGX Spark GB10 等の実機で
    /// `cargo test -p backend-cuda --lib -- --ignored` から実行する。
    ///
    /// **実機実行時の注意**: 現行タイル構成（#494。`MMA_BM=64`/`MMA_BN=128`/
    /// `MMA_BK=32`・f16）では導出段数は 4（`docs/perf/sm121-device-attributes.md`
    /// の検算参照）で、これは静的 SMEM 予算 49,152 バイトをちょうど使い切る
    /// （余裕ゼロ）値である。もし実機のドライバが宣言済み静的確保に加えて
    /// per-block の予約領域（同ドキュメントの `RESERVED_SHARED_MEMORY_PER_BLOCK`
    /// 実測欄参照）を上乗せする場合、`run_f16_with_stages(derived_stages, ...)`
    /// がコンパイル・起動エラーになりうる。その場合の原因は
    /// `derive_stages_for_device`／`derive_pipeline_stages` の結線ではなく
    /// 予算値そのもの（クランプ上限 49,152 が実機の実効上限より大きい）で
    /// ある可能性を先に疑うこと。
    #[test]
    #[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
    fn mma_f16_derived_stage_count_matches_default_stage_count_bit_exact() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");

        let block_m = std::num::NonZeroU32::new(kernels_mma::MMA_BM)
            .expect("kernels_mma::MMA_BM must be non-zero");
        let block_n = std::num::NonZeroU32::new(kernels_mma::MMA_BN)
            .expect("kernels_mma::MMA_BN must be non-zero");
        let block_k = std::num::NonZeroU32::new(kernels_mma::MMA_BK)
            .expect("kernels_mma::MMA_BK must be non-zero");
        let derived_stages = crate::gemm_auto::derive_stages_for_device(
            &device,
            block_m,
            block_n,
            block_k,
            tensor_core::dispatch::DType::F16,
        )
        .expect("derive_stages_for_device must succeed for the current tile configuration");

        let (m, n, k) = (256u32, 256u32, 4096u32);
        let seed: u64 = 9999;
        let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
        let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
        let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));

        let derived_c = run_f16_with_stages(&device, derived_stages.get(), &a, &b, m, n, k)
            .expect("derived stage count must compile and execute");
        let default_c = run_f16_with_stages(&device, kernels_mma::MMA_STAGES, &a, &b, m, n, k)
            .expect("MMA_STAGES-default kernel must compile and execute");

        assert_eq!(
            derived_c,
            default_c,
            "derived stage count {} の出力が既定 MMA_STAGES={} と \
             bit 一致しません（段数導出値とカーネル起動の結線に問題がないか \
             確認すること）",
            derived_stages,
            kernels_mma::MMA_STAGES
        );
    }
}
