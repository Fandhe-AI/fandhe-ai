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
/// `swizzle_group_width`: L2 再利用のためのタイル→SM 割り当てスウィズル
/// （イシュー #499）のグルーピング幅。`Some(_)` は swizzle 適用済み
/// （[`new_with_swizzle`](Self::new_with_swizzle) 経由。
/// `internal-diagnostics` feature 限定）・`None` は無適用
/// （[`new`](Self::new)・[`new_without_swizzle`](Self::new_without_swizzle)
/// いずれも該当。イシュー #740 の本番結線は PR #758 レビュー指摘により
/// 差し戻し済みで、`new` は現時点で swizzle 非適用の base カーネルを
/// 返す。`new` doc comment 参照）を意味する。[`swizzle_group_width`
/// ](Self::swizzle_group_width) アクセサ経由で可観測にする（#733 の
/// `wmma_tf32_staged` 可用性出力と同型の起動時診断。
/// `examples/cuda_floor_bench.rs` 参照）。
pub struct CudaMmaGemm {
    stream: Arc<CudaStream>,
    mma_f16: CudaFunction,
    swizzle_group_width: Option<u32>,
}

/// [`check_min_compute_capability`] のゲート検査後、`source` を NVRTC
/// コンパイルし `gemm_mma_f16` 関数ハンドルをロードする共通手順
/// （[`CudaMmaGemm::new`]・[`new_with_swizzle`](CudaMmaGemm::new_with_swizzle)・
/// [`new_without_swizzle`](CudaMmaGemm::new_without_swizzle) の 3
/// コンストラクタが共有する。イシュー #740 で `new` が swizzle 変種を
/// コンパイルするようになり、3 者ともコンパイル対象ソース文字列のみが
/// 異なる同一手順になったため重複を排除した）。
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
    /// NVRTC コンパイルし保持するハンドルを構築する。
    ///
    /// 手順: (1) `device.compute_capability()` が
    /// [`MIN_COMPUTE_CAPABILITY_MAJOR`]（8.0）未満なら NVRTC コンパイルを
    /// 試みず `CudaError::TensorCoreUnsupported` を返す（`gemm_wmma.rs::new`
    /// と同じ判断。cc 判定をコンパイル前に行うことで、非対応デバイス上での
    /// 無駄な NVRTC 呼び出し・コンパイル失敗の紛れ込みを避ける）。(2)
    /// `kernels_mma::mma_f16_source()`（**swizzle 無適用の base カーネル**）
    /// を `device.arch()` 向けに `nvrtc::compile_ptx` でコンパイルする。
    ///
    /// L2 再利用のためのタイル→SM 割り当てスウィズル（イシュー #499）は
    /// **本番既定へは未結線**（一度イシュー #740 で本番結線したが、
    /// codex-review／Cursor Bugbot 指摘〈PR #758〉により差し戻した）。
    /// 理由: (a) `docs/perf/cuda-gemm-swizzle-ab.md` §4 の採用基準
    /// （size ∈ {2048, 4096} の両方で中央値 TFLOPS 改善）に対し 2048 は
    /// ×0.97〜1.00 で未改善のまま「4096 大幅改善＋中立」を代替基準として
    /// 人間承認なしに読み替えていた、(b) 結線前必須のレジスタスピル確認・
    /// 本番コンストラクタでの bit 一致再検証・parity 再検証が未実施
    /// だった、(c) `select_swizzle_group_width` の CI 恒久検査が
    /// `docs/perf/sm121-device-attributes.md` L58 の RTX 3060 実測値
    /// （sm_121 には流用不可と同ドキュメントが明記）を GB10 実測 SM 数と
    /// 誤って扱っていた（同ドキュメント L79 のとおり GB10 の SM 数は
    /// 本リポでは未実測）。採用基準を字義どおり満たすか、満たさない
    /// 場合は人間承認を伴って採用基準を正式改訂したうえで再結線する
    /// （`docs/perf/cuda-gemm-swizzle-ab.md` §2・§6 参照）。swizzle 適用版
    /// が必要な場合は [`new_with_swizzle`](Self::new_with_swizzle)
    /// （`internal-diagnostics` feature 限定）を使う。(3)
    /// `device.context().load_module()` → `load_function("gemm_mma_f16")`。
    /// `libnvrtc` 不在時は `CudaError::NvrtcUnavailable` を返す
    /// （`compile_ptx` のプローブゲートを経由。panic しない。本セッションの
    /// 実行環境がまさにこの分岐——CUDA driver はあるが NVRTC はない——で
    /// あり、`tests/gemm_mma.rs` の環境適応テストで確認済み。
    /// `kernels_mma.rs` 冒頭「検証状態」参照）。
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

        Ok(Self {
            stream,
            mma_f16,
            swizzle_group_width: None,
        })
    }

    /// `device` 上で、swizzle 変種と同じ [`check_min_compute_capability`]・
    /// NVRTC コンパイル手順で、**swizzle remap を適用しない base カーネル**
    /// （`kernels_mma::mma_f16_source()`）を NVRTC コンパイルし保持する
    /// ハンドルを構築する。
    ///
    /// イシュー #740 で一時的に [`new`](Self::new) が swizzle 既定へ
    /// 切り替わった際、A/B 計測（`examples/gemm_mma_swizzle_bench.rs`）・
    /// bit 一致検証（本ファイル `mod tests` の swizzle 変種比較テスト）が
    /// base カーネルへアクセスする入口として新設した（現在は `new` 自体が
    /// 再び base カーネルを返すため冗長だが、`new_with_swizzle` と対称の
    /// 明示的な入口として・将来 `new` を再結線した場合の base 側アクセス
    /// 手段として維持する）。
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
            swizzle_group_width: None,
        })
    }

    /// `device` 上で、L2 再利用のためのタイル→SM 割り当てスウィズル
    /// （イシュー #499・`kernels_mma::mma_f16_source_with_swizzle`）を
    /// 明示指定の `group_width` で適用した変種カーネルを NVRTC コンパイル
    /// し保持するハンドルを構築する（**診断用・明示幅指定の入口**。
    /// [`new`](Self::new) はイシュー #740 で一時本番既定へ結線したが
    /// PR #758 レビュー指摘により base カーネルへ差し戻し済みのため
    /// 〈`new` doc comment 参照〉、本コンストラクタは A/B 計測・bit 一致
    /// 検証で候補幅 `{8, 16}` を個別に指定したい場合の用途に限定される）。
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
    /// `required-features` で同 feature を要求）専用の入口であり、実機
    /// A/B 計測後に採用確定した段階で feature ゲートを外し安定 API へ
    /// 昇格する（`docs/perf/cuda-gemm-swizzle-ab.md` 参照）。
    #[cfg(feature = "internal-diagnostics")]
    pub fn new_with_swizzle(device: &CudaDevice, group_width: u32) -> Result<Self, CudaError> {
        check_min_compute_capability(device)?;

        let src = kernels_mma::mma_f16_source_with_swizzle(group_width)?;
        let (stream, mma_f16) = compile_mma_f16(device, &src)?;

        Ok(Self {
            stream,
            mma_f16,
            swizzle_group_width: Some(group_width),
        })
    }

    /// L2 再利用スウィズル（イシュー #499・#740）の適用グルーピング幅を
    /// 返す（構造体ドキュメンテーションコメント参照）。`new` はイシュー
    /// #740 で一時 swizzle 既定へ動的選択結線したが、PR #758 レビュー
    /// 指摘により base カーネルへ差し戻し済みのため、現時点で `new` が
    /// 返すハンドルは常に `None`（`new` doc comment 参照）。`Some(_)` は
    /// [`new_with_swizzle`](Self::new_with_swizzle)（診断用・明示幅指定。
    /// `internal-diagnostics` feature 限定）経由のみで観測される。`None`
    /// はそれ以外（`new`・[`new_without_swizzle`](Self::new_without_swizzle)
    /// いずれも該当）を意味する。
    ///
    /// `examples/cuda_floor_bench.rs` の起動時診断（#733 の
    /// `wmma_tf32_staged` 可用性出力と同型）が、現在選択されている値
    /// （差し戻し後の通常ビルドでは常に `None`）を可観測にするために呼ぶ。
    /// feature 非依存の常時公開 API（`new` 自体が feature 非依存のため）。
    pub fn swizzle_group_width(&self) -> Option<u32> {
        self.swizzle_group_width
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
        // 不要）。
        unsafe {
            self.stream
                .launch_builder(&self.mma_f16)
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

    /// #499 受け入れ基準（実機検証）: [`CudaMmaGemm::new_with_swizzle`]
    /// が生成する各 `group_width` の変種が、swizzle 無適用の
    /// [`CudaMmaGemm::new_without_swizzle`]（base）と**ビット一致**の
    /// 出力を返すことを確認する。[`CudaMmaGemm::new`] は現時点では base
    /// カーネルそのもの（イシュー #740 の本番結線は PR #758 レビュー
    /// 指摘により差し戻し済み。`new` 冒頭ドキュメンテーションコメント
    /// 参照）のため、`new` と base の一致検査は自明だが、将来の再結線時に
    /// 誤って `new` が base と乖離した場合を検知する回帰テストとして
    /// 維持する。
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
        // `new`（本番既定コンストラクタ）は現時点では base カーネル
        // そのもの（イシュー #740 の本番結線は PR #758 レビュー指摘に
        // より差し戻し済み）。将来の再結線時に `new` が base と乖離
        // していないかを検知する回帰テストとして、この一致検査を維持する。
        let production = CudaMmaGemm::new(&device)
            .expect("production CudaMmaGemm::new must succeed on ignored test runner");
        // bit 一致検査（下記 production_c vs base_c）だけでは、swizzle は
        // 仕様上ビット一致のため「動的幅カーネルへサイレントに再結線され
        // ても出力が変わらず検知できない」（PR #758 cursor[bot] レビュー
        // 指摘）。`swizzle_group_width()` を直接検査し、`new` が
        // 「swizzle 無適用（`None`）」という差し戻し後の契約から外れて
        // いないかをここで可観測にする。
        assert_eq!(
            production.swizzle_group_width(),
            None,
            "CudaMmaGemm::new の swizzle_group_width が None ではありません \
             （イシュー #740 の本番結線は PR #758 レビュー指摘により差し \
             戻し済みのため、意図しない再結線が起きていないか確認すること）"
        );

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

                // 本番既定コンストラクタ（`new`）が base と bit 一致する
                // ことを検査する（`new` は現時点で base カーネルそのもの
                // のため自明に一致するが、将来の再結線時の回帰検知として
                // 維持する）。
                if group_width == dynamic_group_width {
                    let production_c = production.run_f16(&a, &b, m, n, k).unwrap_or_else(|err| {
                        panic!(
                            "production CudaMmaGemm::new run_f16 failed for shape \
                             (m={m}, n={n}, k={k}): {err}"
                        )
                    });
                    assert_eq!(
                        production_c, base_c,
                        "shape (m={m}, n={n}, k={k}): 本番既定コンストラクタ \
                         CudaMmaGemm::new の出力が base と bit 一致しません"
                    );
                }
            }
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
