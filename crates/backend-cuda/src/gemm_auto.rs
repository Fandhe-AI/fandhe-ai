//! GEMM 自動経路選択の入口（TASK-11.2b・#68）。
//!
//! `fandhe_ai_tensor_core::dispatch::select_gemm_kernel`（規則の純関数実装。
//! `docs/dispatch-rules-design.md` §5 の決定表）を `backend-cuda` の
//! 既存 GEMM 実装（`gemm.rs::CudaGemm`〈naive／tiled〉・
//! `gemm_wmma.rs::CudaWmmaGemm`〈WMMA f16 Tensor Core〉・
//! `gemm_mma.rs::CudaMmaGemm`〈mma.sync f16 Tensor Core。#1150 設計・
//! #1152/#1156 で本構造体へ結線済み〉）へ結線する。
//! `BackendOps` trait 実装（TASK-1.9c・#46）はこの `CudaGemmAuto` を
//! `BackendOps::gemm` から呼ぶだけの構成にできる（実装計画 §3.2
//! 「#46 との境界」）。
//!
//! `DeviceCaps` は [`CudaGemmAuto::new`] 実行時（デバイス初期化）に 1 回
//! だけ `CudaDevice::compute_capability()` から構築し、以降の
//! `run_f32`／`run_f16` 呼び出しでは FFI 照会を繰り返さない
//! （`docs/dispatch-rules-design.md` §2.1「判定タイミング」）。
//!
//! **設計目標**のフォールバック連鎖は `MatrixUnit(mma) → MatrixUnit(wmma)
//! → Tiled → Naive`（`docs/dispatch-rules-design.md` §5.6）。
//! `select_gemm_kernel`（第 1 層。`fandhe_ai_tensor_core::dispatch`）が
//! `KernelKind::MatrixUnit` を返した場合、判定ロジック（第 2 層。
//! `select_f16_matrix_unit_impl`）は `prefer_mma` が `true` のときのみ
//! `CudaMmaGemm → CudaWmmaGemm → Tiled` の優先順位で実装を選ぶ（`mma` は
//! cc ゲート〈`cc >= 8.0`〉に加え事前形状ゲート〈`validate_mma_alignment
//! (n, k)`・`validate_mma_grid_bounds(m)` がいずれも `Ok`〉を満たす場合
//! のみ選ばれ、満たさなければ `wmma`〈`Some` なら〉、`wmma` も
//! `None`／非充足なら tiled へ倒す。fail-safe。`panic!`／`unwrap()` を
//! 使わない。エラー駆動フォールバック〈mma 実行の `Err` を捕捉して wmma
//! へ再試行〉は採らない）。
//!
//! **本番既定（`MMA_PRIORITY_PRODUCTION_ENABLED = true`。#1191）は
//! 上記優先順位を有効化しており、`CudaGemmAuto::run_f16` は cc>=8.0・
//! 整列形状かつ `mma` 構築済みなら `CudaMmaGemm → CudaWmmaGemm →
//! Tiled → Naive` の順に選ぶ**（イシュー #1160: #1156 のユーザー承認
//! 条件「切替前後を同一プロトコル・5 回計測中央値で比較し、後退時は
//! 結線しない」〈§5.6〉自体は GB10 実機（転送込みの auto 経路。
//! `docs/perf/cuda-gemm-auto-f16-mma-switch.md`）で満たすことを確認
//! 済みで（512/1024/2048 は after が base の run-median を上回り、
//! 4096 は #1130 の per-call アロケーション病態下で after の
//! run-median が base の 5 run 範囲内であることを確認）、PR #1179
//! codex-review 指摘（P1）で保留していた K=4096 ストレス形状の
//! 非後退ゲート（`tests/gemm_auto.rs::
//! run_f16_k4096_stress_non_regression_route_aware`）が参照する
//! `ParityPath::MmaF16` baseline 行の ceiling も #1190（PR #1207）で
//! ユーザー承認値が `BASELINES` へ反映済みのため、#1191 で
//! `MMA_PRIORITY_PRODUCTION_ENABLED` を `true` へ復帰した
//! （`docs/perf/cuda-parity-baseline.md` §12.4〜§12.6）。

use std::num::NonZeroU32;

use cudarc::driver::sys::CUdevice_attribute;
use fandhe_ai_tensor_core::dispatch::{
    DType, DeviceCaps, GemmShape, KernelKind, select_gemm_kernel,
};
use half::f16;

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm::CudaGemm;
use crate::gemm_mma::{CudaMmaGemm, validate_mma_alignment, validate_mma_grid_bounds};
// `SpecializedMmaKernelHandle::compile`（本ファイル下部）でのみ使う。
// 同メソッドが `internal-diagnostics` feature（既定 off）でのみコンパイル
// されるため、この import も同 feature でゲートしないと既定ビルドで
// unused import 警告（`-D warnings` で fail）になる。
#[cfg(feature = "internal-diagnostics")]
use crate::gemm_mma::check_min_compute_capability;
use crate::gemm_wmma::CudaWmmaGemm;
use crate::kernels_mma::{
    DimSpec, MMA_K, MMA_SHARED_MEM_BYTES, MMA_STATIC_SMEM_LIMIT_BYTES, MMA_WARP_M, MMA_WARP_N,
    MmaKernelConfig, RenderedMmaKernel, render_mma_f16,
};
// `SpecializedMmaKernelHandle` の非公開フィールド型としてのみ使う
// （上記 `check_min_compute_capability` と同じ理由で feature ゲート）。
#[cfg(feature = "internal-diagnostics")]
use crate::kernels_mma::CompiledMmaKernel;
use crate::nvrtc::{CompiledDims, CudaKernelDescriptor, derive_pipeline_stages};

/// 静的 `__shared__`（動的 SMEM opt-in 非使用）構成のカーネルが従う
/// per-block SMEM 上限。[`MMA_STATIC_SMEM_LIMIT_BYTES`] から
/// 直接導出し、49,152（48KiB）という値をここで独立にハードコードしない
/// （以前は `gemm_auto.rs` と `kernels_mma.rs` の双方が同じリテラルを
/// 個別に持っており、どちらか一方だけが変更されても検出されない静かな
/// 乖離リスクがあった。#521 レビュー指摘）。[`derive_stages_for_device`]
/// はデバイス実測値がこれを上回っても静的上限でクランプする（本モジュール
/// 冒頭コメント 3.1 節の安全側判断: NVRTC コンパイル自体がこの上限超過で
/// 失敗するため、予算側で先に fail-closed に遮断する）。
pub const STATIC_SMEM_BUDGET_CAP_BYTES: u64 = MMA_STATIC_SMEM_LIMIT_BYTES as u64;

// 実使用量が予算上限に収まっていることのコンパイル時契約検査（上記の
// 「唯一の真実源」導出そのものは値の重複を構造的に排除しているが、これは
// 別の検査: 現在のタイル構成〈MMA_SHARED_MEM_BYTES〉が STATIC_SMEM_BUDGET_
// CAP_BYTES を超えていないかを機械検出する使用量ガードである。実機
// コンパイルできない環境でも `cargo build` の時点で検出できる代替チェック
// （`kernels_mma.rs` 側の同種 assert と対になる二重チェック）。
const _: () = assert!(
    MMA_SHARED_MEM_BYTES as u64 <= STATIC_SMEM_BUDGET_CAP_BYTES,
    "gemm_auto::STATIC_SMEM_BUDGET_CAP_BYTES に対して \
     kernels_mma::MMA_SHARED_MEM_BYTES の実使用量が超過している"
);

/// f16 の要素バイト幅（`kernels_mma.rs` の f16 * 2B と同じ根拠）。
/// [`derive_stages_for_device`] の `bytes_per_element` 導出で使う
/// コンパイル時定数。`NonZeroU32::new` は `Option` を返すため `match` で
/// 分解するが、リテラルが非ゼロであることはコンパイル時に確定しており
/// 実行時に `None` 分岐へ到達することはない（#521 レビュー指摘: 旧実装は
/// これを実行時エラーとして扱っており、到達不能なエラーパスを持っていた）。
const BYTES_PER_ELEMENT_F16: NonZeroU32 = match NonZeroU32::new(2) {
    Some(v) => v,
    None => panic!("BYTES_PER_ELEMENT_F16: 2 must be non-zero"),
};

/// F32 の要素バイト幅（将来の mma/tf32 経路を見越した 4 バイト）。
/// [`BYTES_PER_ELEMENT_F16`] と同じ理由でコンパイル時定数とする。
const BYTES_PER_ELEMENT_F32: NonZeroU32 = match NonZeroU32::new(4) {
    Some(v) => v,
    None => panic!("BYTES_PER_ELEMENT_F32: 4 must be non-zero"),
};

/// `device` の `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK` を実行時
/// に取得し、`STATIC_SMEM_BUDGET_CAP_BYTES` でクランプしたうえで
/// [`derive_pipeline_stages`]（`nvrtc.rs`・C-8）へ渡してパイプライン段数を
/// 導出する結線ヘルパー（イシュー #521 実装計画 §4）。
///
/// SMEM 容量をハードコード定数として持たない理由: `docs/perf/
/// sm121-device-attributes.md` の sm_121 実機実測は「未実測・要実機実行」
/// のまま安全側クローズされており（A-2・#482）、存在しない実測値を推定で
/// 定数化することは同ドキュメントの方針に反する。CUTLASS
/// `StageCountAutoCarveout` と同じ「実デバイスの残量から自動算出・固定値
/// ハードコードなし」の思想を採る。
///
/// 属性取得の失敗（例: デバイス問い合わせ自体の driver API エラー）は
/// `CudaError::Driver` として呼び出し元へそのまま伝播する（fail-closed。
/// `From<DriverError>` は `error.rs` に既存）。
///
/// 本関数は導出ロジックと B-1（#492）可変化カーネルとの接続を実証する
/// ためのものであり、既定の本番 GEMM 経路（[`CudaGemmAuto::run_f16`]・
/// `gemm_mma.rs::CudaMmaGemm::run_f16` の `MMA_STAGES=3` 固定）へ導出値を
/// 適用する判断は本イシューのスコープ外（C-10・#527 最良構成選定に委ねる。
/// カーネルソース自体は一切変更していないため、この関数は既定経路の
/// 実行結果・parity ベースラインに影響しない）。
pub fn derive_stages_for_device(
    device: &CudaDevice,
    block_m: NonZeroU32,
    block_n: NonZeroU32,
    block_k: NonZeroU32,
    dtype: DType,
) -> Result<NonZeroU32, CudaError> {
    let smem_budget_bytes = read_clamped_smem_budget_bytes(device)?;
    let bytes_per_element = bytes_per_element_for(dtype);

    derive_pipeline_stages(
        block_m,
        block_n,
        block_k,
        bytes_per_element,
        smem_budget_bytes,
    )
}

/// `device` の `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK` を取得し
/// [`STATIC_SMEM_BUDGET_CAP_BYTES`] でクランプする（[`derive_stages_for_device`]
/// と [`enumerate_tile_candidates_for_device`] の共通結線部分。イシュー
/// #524 で両者が同じ属性取得・クランプ・エラー処理を必要としたため、
/// この結線部分を private ヘルパーへ抽出して重複を避ける）。
///
/// `attribute()` は driver API 契約上 `i32` を返す（cudarc 0.19.8
/// `CudaContext::attribute`）。`MAX_SHARED_MEMORY_PER_BLOCK` は物理的に
/// 負値になり得ないが、ドライバが不正値を返すケースを暗黙の 0 丸め
/// （`unwrap_or(0)`）で握り潰すと、後続の `derive_pipeline_stages` が
/// 返す `min_required` 未達エラーが「予算 0」という誤解を招く診断に
/// なる。`TryFrom` 失敗を明示的な `InvalidKernelDescriptor` として伝播し、
/// fail-closed のまま原因を追跡可能にする。
///
/// `pub(crate)`: イシュー #592（融合 RMSNorm 順伝播カーネル・`rmsnorm.rs`）
/// が persistent block 数導出の SMEM 予算クランプを本関数と共有するため
/// クレート内公開に広げた（#521 の「同じ属性取得・クランプ・エラー処理の
/// 重複を避ける」教訓を GEMM 外のカーネルにも適用する）。
pub(crate) fn read_clamped_smem_budget_bytes(device: &CudaDevice) -> Result<u64, CudaError> {
    let raw_attr = device
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?;
    let raw_attr_u64 = u64::try_from(raw_attr).map_err(|_| CudaError::InvalidKernelDescriptor {
        detail: format!(
            "read_clamped_smem_budget_bytes: CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK \
             returned a negative value ({raw_attr}), which cannot be a valid SMEM budget"
        ),
    })?;
    Ok(raw_attr_u64.min(STATIC_SMEM_BUDGET_CAP_BYTES))
}

/// `dtype` に対応するカーネル入出力の要素バイト幅を返す
/// （[`derive_stages_for_device`]・[`enumerate_tile_candidates`] の共通
/// 導出。`kernels_mma.rs` の f16 * 2B と同じ根拠。F32 は将来の mma/tf32
/// 経路を見越して 4 バイトとする）。
///
/// `DType` は非網羅的でない列挙（`fandhe_ai_tensor_core::dispatch`）であるため
/// match は両変種を明示し、新変種追加時はコンパイルエラーで見落としを
/// 検出する（`_ =>` フォールバックは持たない）。両アームともコンパイル
/// 時定数（[`BYTES_PER_ELEMENT_F16`]／[`BYTES_PER_ELEMENT_F32`]）を返す
/// ため、`NonZeroU32::new` の失敗分岐はコンパイル時に確定して排除される
/// （旧実装は実行時 `ok_or_else` で検査しており、両アームとも非ゼロの
/// リテラルであるため到達不能な `Err` 分岐を持つだけだった。#521 レビュー
/// 指摘）。
fn bytes_per_element_for(dtype: DType) -> NonZeroU32 {
    match dtype {
        DType::F16 => BYTES_PER_ELEMENT_F16,
        DType::F32 => BYTES_PER_ELEMENT_F32,
    }
}

/// タイル候補（block_m/n/k・パイプライン段数）の列挙・枝刈り（Phase C-9a・
/// イシュー #524）。C-8（#521・[`derive_pipeline_stages`]）が導出した
/// 単一構成の段数判定を、候補空間全体の列挙へ拡張する前段実装。
///
/// 参照実装は DeepGEMM（`sm90.hpp`）の候補列挙ヒューリスティクス（block_m
/// の離散集合 × block_n の 16 の倍数刻み × cluster を全列挙し、事前枝刈り
/// で不正構成を早期棄却する方式）だが、DeepGEMM は SM90（Hopper・
/// cluster・swizzle 前提）向けであるため、本カーネル族
/// （sm_121・`kernels_mma.rs` の cp.async + mma.sync 構成）へ以下のとおり
/// 適応する。
///
/// ## 候補空間
///
/// | 軸 | 候補 |
/// |----|------|
/// | block_m | `BLOCK_M_BASE_CANDIDATES`（`{64, 128}`）。`shape.m <=`
/// | `SMALL_M_THRESHOLD`（64）の場合のみ `MMA_WARP_M`（32）を追加する。
/// | 16 は本カーネル族の warp タイル（`MMA_WARP_M` = 32）が構造的に
/// | 分解不能なため候補にしない |
/// | block_n | `MMA_WARP_N`（16）の倍数刻みで `16..=BLOCK_N_MAX_CANDIDATE`
/// | （256） |
/// | block_k | `MMA_K`（16）の倍数 `{16, 32, 64}`
/// | （本カーネルの K ループ構造 `MMA_K_STEPS_PER_STAGE = BK / MMA_K`
/// | の整数制約） |
/// | cluster | 列挙しない（thread block cluster は本カーネル族が
/// | 使用しない機構のため、DeepGEMM の「SM 数が cluster サイズで
/// | 割り切れない」枝刈りは自明に通過する。将来 cluster 導入時の
/// | 拡張点として本コメントに記録する） |
///
/// ## 枝刈り規則（[`TileCandidate`] に残さない条件。§本モジュール
/// ドキュメンテーションコメント §3.2 対応）
///
/// 1. warp 分解不能: `block_m % MMA_WARP_M != 0` または
///    `block_n % MMA_WARP_N != 0` または `block_k % MMA_K != 0`
///    （候補空間の構成段階で構造的に排除されるが、将来の候補空間変更に
///    対する多層防御として実行時にも検査する）。加えて
///    `block_k / MMA_K < MIN_MMA_K_STEPS_PER_STAGE`（2）も棄却する
///    （`kernels_mma.rs` の `MMA_K_STEPS_PER_STAGE`〈`= MMA_BK / MMA_K`〉
///    が 2 以上であることをコンパイル時に要求する `assert!`〈#494
///    受け入れ基準 3 項〉と同じ K ループ構造契約。
///    `block_k == MMA_K`〈16〉だと 1 段あたりの kstep 反復が 1 回になり
///    カーネル側のコンパイル時契約に違反するため、候補段階で先に
///    fail-closed に遮断する。#524 レビュー指摘: 旧実装は整数倍数性のみ
///    検査し「2 以上」を検査していなかった）
/// 2. per-block スレッド数超過:
///    `(block_m / MMA_WARP_M) * (block_n / MMA_WARP_N) * 32 > 1024`
///    （CUDA の per-block スレッド上限。`kernels_mma.rs` の compile-time
///    assert と同じ制約の実行時版）
/// 3. レジスタ不足: `block_m > 128 && block_n > 128` の同時成立
///    （DeepGEMM の事前枝刈りをそのまま踏襲。現在の
///    `BLOCK_M_BASE_CANDIDATES`（最大 128）では block_m が 128 を超える
///    候補自体が生成されないため、この規則は現行候補空間では構造的に
///    発火しない〈dead〉。#524 レビュー指摘。`BLOCK_M_BASE_CANDIDATES` が
///    将来 128 超へ拡張された際の多層防御として意図的に残す。対応する
///    テスト `candidates_never_exceed_128_on_both_dimensions_even_with_huge_budget`
///    は現行候補空間では規則 1・2 由来の恒真検査になるが、規則 3 単独の
///    リグレッション〈規則 3 自体の削除・条件緩和〉も検出できるため
///    保持する）
/// 4. cp.async アライメント不足:
///    `(block_k * bytes_per_element) % 16 != 0` または
///    `(block_n * bytes_per_element) % 16 != 0`（DeepGEMM の
///    「swizzle 64B 未満」枝刈りの本カーネル族への適応。既存テスト
///    `mma_tile_dims_satisfy_cp_async_alignment_granularity`
///    〈`kernels_mma.rs`〉と同じ 16B 粒度契約）。block_n は規則 1 により
///    常に `MMA_WARP_N`（16）の倍数、block_k は規則 1（続き）により
///    常に `MMA_K`（16）の倍数と確定しているため、`bytes_per_element`
///    が 1 以上である限り両積は常に 16 の倍数になり、現行候補空間では
///    この規則も構造的に発火しない〈dead〉。#524 レビュー指摘。将来
///    warp タイル寸法（`MMA_WARP_N`／`MMA_K`）が 16 未満に変更される
///    ケースの多層防御として意図的に残す
/// 5. 段数・SMEM 不成立: [`derive_pipeline_stages`] が `Err`（SMEM
///    予算超過と DeepGEMM 由来の最小段数要求〈3 段・小タイル 4 段〉
///    未達をここで一括棄却する。C-8 実装の再利用であり閾値を二重管理
///    しない）
/// 6. GEMM 形状自体の cp.async 整列制約: `shape.k % 8 != 0` または
///    `shape.n % 8 != 0`（`gemm_mma::validate_mma_alignment` と同一契約。
///    候補のブロックタイル寸法〈block_k/block_n〉が規則 1・4 で
///    8 の倍数〈`MMA_K`/`MMA_WARP_N` はいずれも 16 の倍数〉に確定して
///    いても、実際のグローバルメモリ行ストライドである `shape.k`・
///    `shape.n` 自体が整列要件を満たさなければ `kernels_mma.rs` の
///    `cp.async` は実行できない。この規則は個々の候補ではなく形状
///    全体に対する前提のため、列挙開始前に一括で空 `Vec` を返す形で
///    適用する（#524 レビュー指摘: 旧実装は候補側の整列のみ検査し
///    `shape.k`/`shape.n` を検査していなかったため、例えば
///    `GemmShape::new(4096, 9, 17)` に対して実行不能な候補を返して
///    いた）
///
/// 純関数（副作用なし・決定的）であり、候補ゼロ件は空 `Vec` を返す
/// （`panic!`／`Err` にはしない。ゼロ件時のフォールバック方針は C-9b・
/// #527 の判断に委ねる）。返り値は `(block_m, block_n, block_k)` 昇順の
/// 決定的ソート順で返し、C-9b の比較・テストを安定化する。
///
/// 演算はすべて `u32`/`u64` の `checked_*` でオーバーフロー安全に行う
/// （`.claude/rules/coding-rust.md`。本番経路で `unwrap()`/`expect()`
/// を使わない）。
///
/// ## スコープ境界
///
/// 既定の本番 GEMM 経路（[`CudaGemmAuto::run_f16`]・`gemm_mma.rs` の
/// `MMA_BM/BN/BK`・`MMA_STAGES=3` 固定）への候補適用・結線は行わない
/// （C-9b・#527 のスコープ）。カーネルソース・tolerance 定数・既定経路は
/// 一切変更しないため、本関数の追加は既定経路の実行結果・parity
/// ベースラインに影響しない。候補でのカーネル生成・コンパイル・実測も
/// 行わない（\[実機不要\] タスク）。
pub fn enumerate_tile_candidates(
    shape: GemmShape,
    dtype: DType,
    smem_budget_bytes: u64,
) -> Vec<TileCandidate> {
    // 規則 6: GEMM 形状自体の cp.async 整列制約（本関数ドキュメンテーション
    // コメント §枝刈り規則 6 参照）。`kernels_mma.rs` の起動 API
    // （`gemm_mma.rs::CudaMmaGemm::run_f16`）が要求する契約と同一のもの
    // をここでも検証し、実行不能な候補を一切生成しない（候補側の整列
    // 〈規則 4〉だけでは block_k/block_n しか検査できず、shape.k/shape.n
    // 自体の不整列を見逃す）。
    if validate_mma_alignment(shape.n, shape.k).is_err() {
        return Vec::new();
    }

    let bytes_per_element = bytes_per_element_for(dtype);
    let bpe_u64 = u64::from(bytes_per_element.get());

    // block_m 候補: 基本集合に加え、小 M 形状（`m <= SMALL_M_THRESHOLD`）
    // でのみ warp タイル最小値（`MMA_WARP_M`）を追加する（DeepGEMM の
    // 「小 M 時のみ小 block_m を追加」ヒューリスティクスの適応）。
    let mut block_m_candidates: Vec<u32> = BLOCK_M_BASE_CANDIDATES.to_vec();
    if shape.m <= SMALL_M_THRESHOLD {
        block_m_candidates.push(MMA_WARP_M);
    }

    let mut candidates = Vec::new();
    for &block_m in &block_m_candidates {
        let mut block_n = MMA_WARP_N;
        while block_n <= BLOCK_N_MAX_CANDIDATE {
            for &block_k in &BLOCK_K_CANDIDATES {
                if let Some(candidate) = build_tile_candidate(
                    block_m,
                    block_n,
                    block_k,
                    bytes_per_element,
                    bpe_u64,
                    smem_budget_bytes,
                ) {
                    candidates.push(candidate);
                }
            }
            block_n += MMA_WARP_N;
        }
    }

    // C-9b（#527）のコストモデルが決定的に候補を比較できるよう、常に
    // 同一の昇順（block_m → block_n → block_k）で返す。
    candidates.sort_by_key(|c| (c.block_m.get(), c.block_n.get(), c.block_k.get()));
    candidates
}

/// [`enumerate_tile_candidates`] の実デバイス結線ヘルパー（[`derive_stages_for_device`]
/// と同型。イシュー #524 実装計画 §3.3）。
///
/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK` を取得し
/// `STATIC_SMEM_BUDGET_CAP_BYTES` でクランプしたうえで純関数
/// [`enumerate_tile_candidates`] へ渡す。属性取得の失敗は
/// `CudaError::Driver` として呼び出し元へそのまま伝播する（fail-closed）。
pub fn enumerate_tile_candidates_for_device(
    device: &CudaDevice,
    shape: GemmShape,
    dtype: DType,
) -> Result<Vec<TileCandidate>, CudaError> {
    let smem_budget_bytes = read_clamped_smem_budget_bytes(device)?;
    Ok(enumerate_tile_candidates(shape, dtype, smem_budget_bytes))
}

/// `shape` と `compiled`（[`CompiledDims`]。イシュー #519・C-7）から、
/// `MmaKernelConfig` の `dim_m`/`dim_n`/`dim_k` に渡す
/// [`DimSpec`] の三つ組を導出する（DeepGEMM `runtime_utils.hpp`
/// `get_compiled_dim` 相当の純関数）。
///
/// `compiled` が当該次元を定数化対象として選択している場合は
/// `DimSpec::Static(実 shape 値)` を、非選択（動的）の場合は
/// `DimSpec::Dynamic`（カーネル引数 `m`/`n`/`k` をそのまま使う。受け入れ
/// 基準 2「非定数化次元は実行時引数として渡ること」）を返す。
///
/// 既定の本番 GEMM 経路（[`CudaGemmAuto::run_f16`]）からは呼ばれず
/// （本モジュール §スコープ境界参照）、`mod kernels_mma` 同様 rustc の
/// dead-code 解析が誤検知するため `#[allow(dead_code)]` を付す
/// （`kernels_mma::DimSpec::Static` に対する同種の判断と同じ方針）。
#[allow(dead_code)]
pub fn dim_specs_for(shape: GemmShape, compiled: CompiledDims) -> (DimSpec, DimSpec, DimSpec) {
    let pick = |is_compiled: bool, value: u32| {
        if is_compiled {
            DimSpec::Static(value)
        } else {
            DimSpec::Dynamic
        }
    };
    (
        pick(compiled.m(), shape.m),
        pick(compiled.n(), shape.n),
        pick(compiled.k(), shape.k),
    )
}

/// `shape`／`compiled` から特化 [`MmaKernelConfig`] を構築し、実際に
/// [`render_mma_f16`] へ通してカーネルソースへ展開する（イシュー #519・
/// C-7。実装計画 §3.3）。
///
/// ブロックタイル値・段数は現行本番構成（[`MmaKernelConfig::default`]。
/// `MMA_BM`/`MMA_BN`/`MMA_BK`/`MMA_STAGES`）をそのまま使い、
/// `dim_m`/`dim_n`/`dim_k` のみ [`dim_specs_for`] の結果で上書きする。
/// `render_mma_f16` 内部の `validate_mma_kernel_config`（[`kernels_mma`]
/// 唯一の判定根拠。本関数側で検証ロジックを二重管理しない）が
/// `Static` 値のゼロ・8 の倍数制約等を fail-closed で検査するため、
/// 実行不能な特化構成をそのまま返すことはない。
///
/// 戻り値は `(cfg, rendered)`。`cfg` は [`specialized_mma_descriptor`]
/// が同じタイル値・段数からキャッシュキー用 descriptor を組み立てる際に
/// 再利用する（config とキーの値の単一真実源を保つ）。`RenderedMmaKernel`
/// は `Clone` 導出済みのため、呼び出し元がソースを保持したまま `cfg` も
/// 別途参照できる。
///
/// # スコープ境界
///
/// 既定の本番 GEMM 経路（[`CudaGemmAuto::run_f16`]・
/// `gemm_mma.rs::CudaMmaGemm`）への結線・実行は行わない（最良構成選定
/// C-9b・#527・数値一致回帰 #531 のスコープ。プロセス内 LRU カーネル
/// モジュールキャッシュ〈C-4・#511〉は本関数が内部で呼ぶ
/// `RenderedMmaKernel::compile` 側へ実装済みだが、既定経路への結線自体は
/// 行わない点は変わらない）。カーネルソース・既定経路は一切変更しない
/// ため、本関数の追加は既定経路の実行結果・parity ベースラインに影響
/// しない。
///
/// 理由は [`dim_specs_for`] と同じ（既定経路から未結線のため
/// dead-code 解析が誤検知する）。
#[allow(dead_code)]
pub fn specialized_mma_config(
    shape: GemmShape,
    compiled: CompiledDims,
) -> Result<(MmaKernelConfig, RenderedMmaKernel), CudaError> {
    let (dim_m, dim_n, dim_k) = dim_specs_for(shape, compiled);
    let cfg = MmaKernelConfig {
        dim_m,
        dim_n,
        dim_k,
        ..MmaKernelConfig::default()
    };
    let rendered = render_mma_f16(&cfg)?;
    Ok((cfg, rendered))
}

/// [`specialized_mma_config`] と同じタイル値・段数から
/// [`CudaKernelDescriptor`]（コンパイルキャッシュキーの構成要素。C-1・
/// #504）を構築する（イシュー #519・C-7。実装計画 §3.3）。
///
/// `specialized_mma_config` を内部で呼ぶことで、config とキーの値
/// （ブロックタイル・段数・`compiled_dims`）が同一 shape・同一
/// `CompiledDims` から常に一致した値になることを構造で保証する（config
/// とキーが別々に計算されて静かに乖離するリスクを構造的に排除する）。
///
/// `kernel_name` は固定文字列 `"mma_f16"`（`kernels_mma::render_mma_f16`
/// が展開するカーネル本体と対応）。
///
/// 理由は [`dim_specs_for`] と同じ（既定経路から未結線のため
/// dead-code 解析が誤検知する）。
#[allow(dead_code)]
pub fn specialized_mma_descriptor(
    shape: GemmShape,
    compiled: CompiledDims,
) -> Result<CudaKernelDescriptor, CudaError> {
    let (cfg, _rendered) = specialized_mma_config(shape, compiled)?;
    CudaKernelDescriptor::new_with_compiled_dims(
        "mma_f16",
        shape,
        cfg.bm,
        cfg.bn,
        cfg.bk,
        cfg.stages,
        DType::F16,
        compiled,
    )
}

/// `compiled`（[`CompiledDims`]。C-7・#519）が選択した特化構成で
/// コンパイル済み `mma_f16` カーネルを保持し、複数回の起動へ再利用可能
/// にする公開ハンドル（イシュー #531 実装計画 §3.1・§3.2 点 3）。
///
/// [`RenderedMmaKernel`]／[`CompiledMmaKernel`] 型自体は非公開のまま
/// 維持する（PR #643 codex-review 対応で確立した型レベル封じ込め設計。
/// カーネルソース・`CudaFunction` を crate 外へ渡さない）。本ハンドルは
/// [`CompiledMmaKernel`] を非公開フィールドとして内部に保持するだけの
/// 薄いラッパーであり、[`Self::launch_f16`] 以外に内部状態へ到達する
/// 経路を持たない。
///
/// `gemm_mma.rs::CudaMmaGemm` と同じ「1 回コンパイル・複数回起動」の
/// 形を、テスト・ベンチが特化構成に対しても使えるよう公開する。
/// `STATIC_NK`（M=Dynamic）カーネルを同一 N/K・異なる M で再利用する
/// 回帰検査、`STATIC_MNK` の不一致形状起動が
/// `CudaError::InvalidKernelConfig` で fail-closed に拒否されることの
/// 検査に使う（`tests/specialized_mma_parity.rs` 参照）。
///
/// 本番ディスパッチ経路（[`CudaGemmAuto::run_f16`]）からは呼ばれず
/// （本モジュール §スコープ境界参照）テスト・ベンチ専用の検証用ハンドル
/// である。`Self::compile` が内部で呼ぶ `RenderedMmaKernel::compile` は
/// プロセス内 LRU カーネルモジュールキャッシュ（C-4・#511）を経由する
/// ため、同一形状・同一 `CompiledDims` での 2 回目以降の `Self::compile`
/// 呼び出しは NVRTC 再コンパイルを回避しうる（`tests/
/// specialized_mma_parity.rs::
/// specialized_mma_kernel_handle_compile_reuses_process_local_module_cache`
/// 参照）。
///
/// **`internal-diagnostics` feature（既定 off）でのみコンパイルされる**
/// （PR #685 codex-review P1 指摘の是正: 本ハンドルは crate root から
/// 無条件 re-export されていたため、「テスト・ベンチ専用」というコメント
/// 上の意図に反して通常ビルドの安定した公開 API 面に漏出していた。
/// `diagnostics` モジュール〈本ファイル冒頭 `lib.rs` の同 feature ゲート
/// 参照〉・`gemm_mma.rs::CudaMmaGemm::new_with_swizzle` と同じ方針で
/// `#[cfg(feature = "internal-diagnostics")]` により定義自体を既定ビルド
/// のコンパイル対象から外し、`lib.rs` の re-export も同 feature で
/// ゲートする。外部 integration test（`tests/specialized_mma_parity.rs`）
/// は `Cargo.toml` の `[[test]]` セクションで `required-features =
/// ["internal-diagnostics"]` を指定し、`cargo test --all-features`
/// （CI の test ジョブ・`make test` が使うコマンド）でのみビルド・実行
/// される。feature 無効時は crate 外部からはもちろん crate 内部からも
/// 到達不能になるため `#[allow(dead_code)]` は不要（[`dim_specs_for`]
/// のように dead-code 解析が誤検知する状況ではなくなった）。
///
/// `stream` は [`Self::compile`] 実行時（NVRTC コンパイル・
/// `load_module`／`load_function`）に使った `device` の
/// `Arc<CudaStream>`（延いては `Arc<CudaContext>`）をハンドル内に
/// 保持し、[`Self::launch_f16`] は常にこの `stream` のみで起動する
/// （PR #685 codex-review 指摘〈P0〉への対応: 従来は `launch_f16` が
/// 呼び出しごとに任意の `&CudaDevice` を受け取っており、safe な公開
/// API から `compiled.func`〈コンパイル元 context 由来〉と別 context
/// の `stream`／デバイスバッファを渡して起動できてしまっていた。
/// `CudaFunction` はロード元 `CudaContext` に紐付き、別 context の
/// stream・バッファで起動する不変条件違反は cudarc の型では検出
/// されない〈`kernels_mma.rs::CompiledMmaKernel::launch_f16` の
/// `// SAFETY:` 根拠もバッファ長の一致のみを扱い、context 同一性は
/// 前提としていた〉。コンパイル元の `stream` をハンドル内に固定し
/// 起動時の外部入力から外すことで、この不変条件を型・構造の両面で
/// 強制する）。
#[cfg(feature = "internal-diagnostics")]
pub struct SpecializedMmaKernelHandle {
    compiled: CompiledMmaKernel,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    /// [`Self::compile`] が構築した特化 config（[`specialized_mma_config`]）。
    /// [`Self::launch_f16`] が H2D 転送・出力確保より前に
    /// `validate_launch_shape` で形状検証するための保持（codex-review
    /// P2 指摘への対応。本 struct ドキュメンテーションコメント参照）。
    cfg: MmaKernelConfig,
}

#[cfg(feature = "internal-diagnostics")]
impl SpecializedMmaKernelHandle {
    /// `shape`・`compiled` から特化 config を構築し（[`specialized_mma_config`]）
    /// NVRTC コンパイルまで完了させる（[`RenderedMmaKernel::compile`]）。
    /// コンパイルに使った `device` の `stream`（`Arc<CudaStream>`）を
    /// ハンドル内に保持し、以降の [`Self::launch_f16`] 呼び出しは常に
    /// この `stream`（延いては同一 `CudaContext`）でのみ起動する
    /// （本 struct ドキュメンテーションコメント参照）。
    ///
    /// `compiled` が `Dynamic` としている次元は `shape` の対応する値が
    /// 何であっても焼き込みに影響しない（[`dim_specs_for`] 参照）ため、
    /// `STATIC_NK`（M=Dynamic）を渡す場合の `shape.m` は任意の非ゼロ値で
    /// よい（後続 [`Self::launch_f16`] が実際の起動ごとの M を渡す）。
    /// `Static` 化された次元に `0` を渡すと [`specialized_mma_config`]
    /// 内部の `render_mma_f16`（`kernels_mma::validate_mma_kernel_config`）
    /// が fail-closed で拒否する（[`run_specialized_mma_f16`] ドキュメン
    /// テーションコメント「no-op 形状」参照）。
    pub fn compile(
        device: &CudaDevice,
        shape: GemmShape,
        compiled: CompiledDims,
    ) -> Result<Self, CudaError> {
        // `gemm_mma.rs::CudaMmaGemm::new` と同じ `mma.sync`/`ldmatrix`/
        // `cp.async` 命令セットを NVRTC コンパイルするため同じ cc ゲートを
        // NVRTC コンパイル前に適用する（PR #685 Bugbot 指摘〈Low〉・
        // codex-review 指摘への対応。`check_min_compute_capability` doc
        // comment 参照）。
        check_min_compute_capability(device)?;

        let (cfg, rendered) = specialized_mma_config(shape, compiled)?;
        let compiled_kernel = rendered.compile(device)?;
        Ok(Self {
            compiled: compiled_kernel,
            stream: std::sync::Arc::clone(device.stream()),
            cfg,
        })
    }

    /// コンパイル済みカーネルを `a`/`b`/`m`/`n`/`k` で起動する（先行検証
    /// → H2D 転送 → [`CompiledMmaKernel::launch_f16`] → D2H 回収）。
    ///
    /// 転送・出力確保より前に `self.cfg.validate_launch_shape`・
    /// `crate::gemm::validate_gemm_dims`（host 側スライス長）で早期
    /// fail-closed する（codex-review P2 指摘への対応: 従来はこの
    /// 検査を行わずに `clone_htod`／`alloc_zeros` を先に実行しており、
    /// 無効な起動引数でも GPU 転送・確保〈OOM 等〉が先に発生しえた）。
    /// これは `run_specialized_mma_f16` が呼び出し前に `validate_gemm_dims`
    /// を行うのと同型の多層防御であり、device 側バッファ長・アライメント・
    /// grid/k タイル境界の最終検証は引き続き [`CompiledMmaKernel::launch_f16`]
    /// （唯一の真実源）が担う。本メソッドはその検証を複製・代替しない。
    /// `Dynamic` 次元は起動ごとに異なる値を許容しうるため、同一ハンドルへ
    /// 複数回呼べる設計とする。
    ///
    /// `m==0 || n==0`・`k==0` の no-op 形状は `run_specialized_mma_f16`・
    /// `gemm_mma.rs::CudaMmaGemm::run_f16` と同一契約で本メソッド自身が
    /// 早期 return する（PR #685 Bugbot 再指摘〈Medium〉への対応。本メソッド
    /// 実装のコメント参照）。`CompiledMmaKernel::launch_f16` は `k==0` を
    /// 自身の no-op 契約に含めない設計のため、`k==0` の意味付けは呼び出し元
    /// である本メソッドが担う。
    ///
    /// `device` を引数に取らず [`Self::compile`] 時に保持した
    /// `self.stream` のみを使う（本 struct ドキュメンテーションコメント
    /// 参照。呼び出し元がコンパイル元と異なる `CudaDevice`／`stream` を
    /// 持ち込める経路を型で塞ぐ）。
    pub fn launch_f16(
        &self,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f16>, CudaError> {
        self.cfg.validate_launch_shape(m, n, k)?;
        crate::gemm::validate_gemm_dims(a.len(), b.len(), m, n, k)?;

        // no-op 形状（`m==0 || n==0`）・`k==0`（A/B が空の縮約次元となる
        // ため C は全 0）は `gemm_mma.rs::CudaMmaGemm::run_f16`・
        // `run_specialized_mma_f16`（本ファイル）と同一契約で、H2D 転送・
        // カーネル起動そのものを回避して早期 return する（PR #685 Bugbot
        // 再指摘〈Medium〉への対応: 従来は `k==0` の早期 return を欠いており、
        // `m`/`n` が非ゼロなら常に `validate_mma_alignment` を実行していた
        // ため、`(m,n,k)=(8,7,0)` のような有効な no-op〈`k==0` により
        // C が全 0 となるべき形状〉が `n=7` の非整列を理由に誤って拒否
        // されていた。`CompiledMmaKernel::launch_f16` は `k==0` を自身の
        // no-op 契約に含めない設計〈同 struct doc comment 参照。カーネル
        // 自体は `num_k_tiles=0` として起動しうる別経路〉のため、
        // `k==0` の意味付け〈全 0 の C を返す no-op として扱うか〉は
        // 呼び出し元がここで担う）。
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if k == 0 {
            return Ok(vec![f16::ZERO; (m as usize) * (n as usize)]);
        }

        // `cp.async` 整列制約（`validate_mma_alignment`）・grid_dim.y 上限
        // （`validate_mma_grid_bounds`）も H2D 転送・出力アロケーションより
        // 前で fail-closed する（PR #685 Bugbot 指摘〈Medium〉への対応。
        // 本メソッドドキュメンテーションコメント参照: 従来はこの検査を
        // `CompiledMmaKernel::launch_f16` 内部〈転送後〉に委ねきっており、
        // 不正な `Dynamic` 起動でも先に `clone_htod`／`alloc_zeros` が発生
        // しえた）。上記の no-op／`k==0` 早期 return より後に置くことで、
        // 有効な no-op 形状（例: `n` が 8 の倍数でない `(m,n,k)=(8,7,0)`）
        // を誤って拒否しない。
        validate_mma_alignment(n, k)?;
        validate_mma_grid_bounds(m)?;

        let stream = &self.stream;
        let a_dev = stream.clone_htod(a)?;
        let b_dev = stream.clone_htod(b)?;
        let mut c_dev = stream.alloc_zeros::<f16>((m as usize) * (n as usize))?;
        self.compiled
            .launch_f16(stream, &a_dev, &b_dev, &mut c_dev, m, n, k)?;
        // `launch_f16`（`kernels_mma.rs::CompiledMmaKernel`）は #1013 で
        // 非同期投入のみに契約変更済み。ここが本関数唯一のホストへの
        // readback であるため、同期点を `readback` ヘルパーへ集約する。
        crate::memory::readback(stream, &c_dev)
    }
}

/// `compiled`（[`CompiledDims`]。C-7・#519）が選択した特化構成で
/// [`SpecializedMmaKernelHandle::compile`] → [`SpecializedMmaKernelHandle::launch_f16`]
/// を一括実行する結線ヘルパー（イシュー #531 実装計画 §3.1）。
///
/// `gemm_mma.rs::CudaMmaGemm` は `mma_f16` カーネルを 1 回だけコンパイルし
/// 使い回す設計だが、本関数は**呼び出しごとに [`SpecializedMmaKernelHandle::compile`]
/// を新規に呼ぶ**（`SpecializedMmaKernelHandle` インスタンス自体は使い
/// 回さない。本関数はあくまでテスト・ベンチが特化カーネルの実行結果へ
/// 単発で到達するための経路であり、本番ディスパッチ経路
/// （[`CudaGemmAuto::run_f16`]）ではない）。ただし `compile` 内部の
/// `RenderedMmaKernel::compile` はプロセス内 LRU カーネルモジュール
/// キャッシュ（C-4・#511）を経由するため、同一形状・同一 `compiled` での
/// 呼び出しは NVRTC 再コンパイルを回避しうる。複数回起動しての明示的な
/// 再利用検証には [`SpecializedMmaKernelHandle`] を直接使う）。
///
/// no-op 形状（`m==0 || n==0`）・`k==0` の早期 return は
/// `gemm_mma.rs::CudaMmaGemm::run_f16` と同一契約とする。`compiled` が
/// 対象次元を `Static` 化している場合、`Static(0)` は
/// [`specialized_mma_config`] 内部の `render_mma_f16`（`kernels_mma::
/// validate_mma_kernel_config`）が fail-closed で拒否するため、
/// コンパイルへ進む前にここで no-op 判定を済ませる必要がある
/// （`CudaMmaGemm::run_f16` は既定 config が全次元 `Dynamic` のため
/// この制約を持たないが、本関数は `compiled` 次第で `Static(0)` に
/// 到達しうる点が異なる）。
///
/// 形状検証（`validate_gemm_dims`・[`crate::gemm_mma::validate_mma_alignment`]・
/// [`crate::gemm_mma::validate_mma_grid_bounds`]）も `CudaMmaGemm::run_f16`
/// と同一手順・同一関数を再利用し、判定ロジックを複製しない。
///
/// テスト・ベンチからのみ呼ばれる。[`SpecializedMmaKernelHandle`] と
/// 同じ理由（PR #685 codex-review P1 指摘の是正）で
/// **`internal-diagnostics` feature（既定 off）でのみコンパイルされる**
/// （同 struct ドキュメンテーションコメント参照。feature 無効時は crate
/// 内部からも到達不能になるため `#[allow(dead_code)]` は不要）。
#[cfg(feature = "internal-diagnostics")]
pub fn run_specialized_mma_f16(
    device: &CudaDevice,
    compiled: CompiledDims,
    a: &[f16],
    b: &[f16],
    m: u32,
    n: u32,
    k: u32,
) -> Result<Vec<f16>, CudaError> {
    crate::gemm::validate_gemm_dims(a.len(), b.len(), m, n, k)?;

    // no-op 形状の早期 return（本関数ドキュメンテーションコメント参照。
    // `CudaMmaGemm::run_f16` と同一契約）。
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    if k == 0 {
        return Ok(vec![f16::ZERO; (m as usize) * (n as usize)]);
    }

    crate::gemm_mma::validate_mma_alignment(n, k)?;
    crate::gemm_mma::validate_mma_grid_bounds(m)?;

    let shape = GemmShape::new(m, n, k);
    let handle = SpecializedMmaKernelHandle::compile(device, shape, compiled)?;
    handle.launch_f16(a, b, m, n, k)
}

/// [`enumerate_tile_candidates`] の 1 候補分の枝刈り判定＋構築（規則
/// 1〜5・本関数ドキュメンテーションコメント参照元は同関数の doc を参照）。
/// いずれかの規則で棄却された場合は `None` を返す（列挙側はそのまま
/// スキップする）。
fn build_tile_candidate(
    block_m: u32,
    block_n: u32,
    block_k: u32,
    bytes_per_element: NonZeroU32,
    bpe_u64: u64,
    smem_budget_bytes: u64,
) -> Option<TileCandidate> {
    // 規則 1: warp 分解可能性。
    if !block_m.is_multiple_of(MMA_WARP_M)
        || !block_n.is_multiple_of(MMA_WARP_N)
        || !block_k.is_multiple_of(MMA_K)
    {
        return None;
    }

    // 規則 1（続き）: `MMA_K_STEPS_PER_STAGE`（= block_k / MMA_K）が
    // `MIN_MMA_K_STEPS_PER_STAGE`（2）未満の構成を棄却する
    // （`kernels_mma.rs` のコンパイル時契約 `assert!(MMA_K_STEPS_PER_STAGE
    // >= 2, ...)` と同じ制約。#524 レビュー指摘: block_k == MMA_K〈16〉が
    // block_k % MMA_K == 0 を満たしてしまうため上の多重性検査だけでは
    // 通過してしまっていた）。`block_k` は上の検査で `MMA_K` の倍数と
    // 確定済みのため整数除算は割り切れる。
    if block_k / MMA_K < MIN_MMA_K_STEPS_PER_STAGE {
        return None;
    }

    // 規則 2: per-block スレッド数が CUDA の上限（1024）を超えない。
    let warps_m = block_m / MMA_WARP_M;
    let warps_n = block_n / MMA_WARP_N;
    let threads = warps_m.checked_mul(warps_n)?.checked_mul(32)?;
    if threads > 1024 {
        return None;
    }

    // 規則 3: レジスタ不足の事前枝刈り（両ブロック次元同時 128 超）。
    if block_m > 128 && block_n > 128 {
        return None;
    }

    // 規則 4: cp.async 16B アライメント粒度（swizzle 幅相当）。
    let block_k_bytes = u64::from(block_k).checked_mul(bpe_u64)?;
    let block_n_bytes = u64::from(block_n).checked_mul(bpe_u64)?;
    if block_k_bytes % 16 != 0 || block_n_bytes % 16 != 0 {
        return None;
    }

    // 規則 5: SMEM 予算・段数下限（C-8・derive_pipeline_stages への委譲。
    // 判定ロジックの分裂〈片側だけ緩和される静かな乖離〉を構造的に防ぐ
    // ため、閾値をここで二重管理しない）。
    let block_m_nz = NonZeroU32::new(block_m)?;
    let block_n_nz = NonZeroU32::new(block_n)?;
    let block_k_nz = NonZeroU32::new(block_k)?;
    let stages = derive_pipeline_stages(
        block_m_nz,
        block_n_nz,
        block_k_nz,
        bytes_per_element,
        smem_budget_bytes,
    )
    .ok()?;

    // C-9b（#527）のコストモデルが再計算せずに使える付帯情報として、
    // 1 段あたり／全段合計の SMEM 使用量を導出しておく
    // （`derive_pipeline_stages` 内部の同一計算式と一致させる。
    // `nvrtc.rs` は段数のみを返し内部値を公開しないため、ここで
    // 同じ式を再計算する）。
    let a_tile_elems = u64::from(block_m).checked_mul(u64::from(block_k))?;
    let b_tile_elems = u64::from(block_k).checked_mul(u64::from(block_n))?;
    let tile_elems = a_tile_elems.checked_add(b_tile_elems)?;
    let smem_per_stage = tile_elems.checked_mul(bpe_u64)?;
    let smem_total = smem_per_stage.checked_mul(u64::from(stages.get()))?;

    Some(TileCandidate {
        block_m: block_m_nz,
        block_n: block_n_nz,
        block_k: block_k_nz,
        stages,
        smem_per_stage,
        smem_total,
    })
}

/// block_m 候補の基本集合（[`enumerate_tile_candidates`] §候補空間参照）。
const BLOCK_M_BASE_CANDIDATES: [u32; 2] = [64, 128];

/// 小 M 形状（block_m 候補に [`MMA_WARP_M`] を追加する）の閾値。DeepGEMM
/// の「小 M 時のみ小 block_m を追加」ヒューリスティクスをそのまま採用。
const SMALL_M_THRESHOLD: u32 = 64;

/// block_n 候補の上限（[`MMA_WARP_N`] の倍数刻みで列挙する範囲の上端）。
const BLOCK_N_MAX_CANDIDATE: u32 = 256;

/// block_k 候補（[`MMA_K`] の倍数。独立したリテラルを持たず [`MMA_K`]
/// からの倍数として導出し単一真実源を保つ）。`MMA_K`（16）自体は
/// [`MIN_MMA_K_STEPS_PER_STAGE`] 制約（`block_k / MMA_K >= 2`）により
/// `build_tile_candidate` の規則 1 で常に棄却される（#524 レビュー指摘）。
/// 候補生成側から先に除外せず列挙後の枝刈りに委ねているのは、枝刈り
/// 規則の一覧（本モジュール `enumerate_tile_candidates` の doc）を唯一の
/// 判定根拠にするため（候補空間の構成条件と枝刈り条件が分裂すると
/// どちらか一方だけが更新される静かな乖離を招く。#521 と同種の教訓）。
const BLOCK_K_CANDIDATES: [u32; 3] = [MMA_K, MMA_K * 2, MMA_K * 4];

/// カーネル側の K ループ構造契約（`kernels_mma.rs` の
/// `assert!(MMA_K_STEPS_PER_STAGE >= 2, ...)`〈#494 受け入れ基準 3 項〉）
/// を候補列挙側でも検査するための下限値。カーネル側の
/// `MMA_K_STEPS_PER_STAGE` は固定 `MMA_BK`（32）から導出される定数だが、
/// ここでは候補ごとに異なる `block_k` に対して同じ「1 段あたりの kstep
/// 反復回数は 2 以上」という契約を検査する必要があるため、値（2）のみを
/// 独立した名前付き定数として持つ（`kernels_mma::MMA_K_STEPS_PER_STAGE`
/// 自体を再利用すると `MMA_BK=32` 固定の値〈2〉を意味的に異なる文脈
/// 〈任意の block_k〉へ流用することになり誤解を招くため）。
const MIN_MMA_K_STEPS_PER_STAGE: u32 = 2;

/// GEMM ブロックタイル候補 1 件（Phase C-9a・イシュー #524）。
///
/// [`enumerate_tile_candidates`] が列挙した「構造的制約・SMEM/段数制約を
/// 通過した」候補のみを表現する。フィールドは private + getter とし、
/// 不正な組合せを外部から構築できないようにする（[`derive_stages_for_device`]
/// と同じ不変条件維持方針）。
///
/// 後続 C-9b（#527）の L1/L2 帯域コストモデルが `smem_per_stage`／
/// `smem_total` を消費する想定のため、導出済み付帯情報として保持する
/// （コストモデル側が同じ計算をやり直さずに済むようにする）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCandidate {
    block_m: NonZeroU32,
    block_n: NonZeroU32,
    block_k: NonZeroU32,
    stages: NonZeroU32,
    smem_per_stage: u64,
    smem_total: u64,
}

impl TileCandidate {
    /// ブロックタイルの M 次元。
    pub fn block_m(&self) -> NonZeroU32 {
        self.block_m
    }

    /// ブロックタイルの N 次元。
    pub fn block_n(&self) -> NonZeroU32 {
        self.block_n
    }

    /// ブロックタイルの K 次元。
    pub fn block_k(&self) -> NonZeroU32 {
        self.block_k
    }

    /// `derive_pipeline_stages` が導出したパイプライン段数。
    pub fn stages(&self) -> NonZeroU32 {
        self.stages
    }

    /// 1 段あたりの SMEM 使用量（バイト）。
    pub fn smem_per_stage(&self) -> u64 {
        self.smem_per_stage
    }

    /// 全段合計の SMEM 使用量（バイト）。`smem_per_stage * stages` に一致。
    pub fn smem_total(&self) -> u64 {
        self.smem_total
    }
}

/// L1/L2 帯域コストモデルによる静的タイル構成選定（Phase C-9b・イシュー
/// #527）。[`enumerate_tile_candidates`]（C-9a・#524）が列挙した
/// [`TileCandidate`] 群から、実行時ベンチマークを一切行わずに最良 1 件を
/// 決定的に選ぶ。
///
/// 参照実装は DeepGEMM（`sm90.hpp` の `get_layout_info`／`compare`）だが、
/// 同ドキュメントの帯域定数（`l2_bandwidth_per_cycle = min(64*num_sms,
/// 8e6/1300)`・`l1 = 128*num_sms`）は H100 実測由来のマジックナンバーで
/// あり、本モジュールでは**流用しない**。sm_121 実測値
/// （`docs/perf/sm121-device-attributes.md`・A-2・#482）は本イシュー時点で
/// 「未実測・要実機実行」のまま安全側クローズされているため、帯域定数は
/// [`SM121_MEASURED_BANDWIDTH`] という `Option` 型の注入パラメータで表現
/// し、`None` の間はコストモデル自体を稼働させない（下記 [`select_tile_config`]
/// のフォールバック方針参照）。H100 定数の暗黙流用経路を構造的に排除する。
///
/// ## モデル構造（DeepGEMM 同型の段階分解）
///
/// 1. `num_blocks = ceil_div(M, block_m) * ceil_div(N, block_n)`
/// 2. `num_waves = ceil_div(num_blocks, num_sms)`
/// 3. L2（グローバル→SMEM）総トラフィックから `l2_cycles` を、L1
///    （SMEM→レジスタ、mma.sync オペランド供給）総トラフィックから
///    `l1_cycles` を導出する（下記 §トラフィックモデル）
/// 4. `base_cycles = max(l1_cycles, l2_cycles)`（律速側支配）
/// 5. wave 効率 `num_blocks / (num_waves * num_sms)` で `base_cycles` を
///    補正する（端数 wave を生む構成ほど不利になる）:
///    `cost = base_cycles * num_waves * num_sms / num_blocks`
/// 6. `cost` 昇順で比較する（DeepGEMM `compare` 相当）
///
/// ## トラフィックモデル（本カーネル族〈cp.async + mma.sync・cluster
/// なし・num_groups=1〉への適応。DeepGEMM の Hopper cluster／swizzle
/// 前提の削減ヒューリスティクスはそのまま持ち込めないため、独自に導出
/// する物理モデルを採る）
///
/// - `K_padded = ceil_div(shape.k, block_k) * block_k`（末尾 k-step の
///   境界検査分もブロック単位で丸めた保守的な見積り）
/// - **L2（グローバルメモリ→L2/SMEM）**: 各ブロックはグローバルメモリ
///   から A タイル（`block_m * K_padded` 要素）・B タイル
///   （`block_n * K_padded` 要素）を 1 回ずつロードする（ブロック間の
///   L2 ヒット再利用は保守的にモデル化しない）:
///   `l2_traffic = num_blocks * (block_m + block_n) * K_padded * bytes_per_element`
/// - **L1（SMEM→レジスタ、mma.sync オペランド供給）**: warp タイル分割
///   （`warps_m = block_m / MMA_WARP_M`・`warps_n = block_n / MMA_WARP_N`）
///   により、A タイルの各バイトは `warps_n` 個の N 方向 warp グループに、
///   B タイルの各バイトは `warps_m` 個の M 方向 warp グループにそれぞれ
///   再読み込みされる（同一 SMEM 内容を複数 warp が個別に `ldmatrix`
///   等で読む構造上の再読込係数）:
///   `l1_traffic = num_blocks * (block_m * warps_n + block_n * warps_m) * K_padded * bytes_per_element`
///
/// `l1_traffic >= l2_traffic`（`warps_m, warps_n >= 1`）が構造的に成立し、
/// SMEM 再読込トラフィックがグローバルメモリトラフィックを常に上回る
/// （タイリングによる演算強度向上の物理的根拠と整合する）。
///
/// **L1 トラフィックはタイル構成にほぼ不変である（意図された挙動）**:
/// `warps_n = block_n / MMA_WARP_N`・`warps_m = block_m / MMA_WARP_M` を
/// 代入すると `block_m*warps_n + block_n*warps_m = block_m*block_n *
/// (1/MMA_WARP_N + 1/MMA_WARP_M)` となり、タイル面積（`block_m*block_n`）
/// に比例する。したがって `l1_traffic ∝ num_blocks * block_m * block_n`
/// であり、`block_m`・`block_n` が `shape.m`・`shape.n` をそれぞれ割り切る
/// 場合は `num_blocks * block_m * block_n ≈ shape.m * shape.n`（タイル
/// 構成に依らない定数）に近似される。これは warp タイル（レジスタ
/// ブロッキング係数）がブロックタイル寸法と独立に固定されていることの
/// 帰結であり、物理的にも妥当（1 flop あたりに必要な warp フラグメント
/// 読み出し量はブロック分割の仕方によらずほぼ一定。ブロックタイル拡大の
/// 恩恵はもっぱら L2 側〈グローバルメモリ再読込の削減〉に現れる）。
/// 候補間の L1 側の差は主に「割り切れない形状での `ceil_div` 余剰」から
/// 生じ、タイル面積そのものの差では大きく動かない。実機補正（下記
/// `docs/perf/cuda-gemm-cost-model-selection.md` §1 ステップ 4）で
/// モデル選定と実測が食い違う場合、L1 側の帯域定数を調整しても整列
/// 形状ではランキングをほぼ動かせない点に注意し、補正は L2 側の係数・
/// wave 効率項を優先して見直すこと。
///
/// ## 数値表現
///
/// 浮動小数を使わず、すべての中間値を分子・分母を保持した
/// [`CycleFraction`]（`u128` の `checked_*` 演算）として扱い、比較は
/// 交差乗算で行う（決定性・オーバーフロー安全性の確保。
/// `.claude/rules/coding-rust.md`「本番経路で `unwrap()`/`expect()` を
/// 使わない」）。オーバーフロー・ゼロ除算はすべて `CudaError::
/// InvalidKernelDescriptor` として型付きエラーで返す（`panic!` なし）。
mod cost_model {
    use std::num::{NonZeroU32, NonZeroU64, NonZeroU128};

    use fandhe_ai_tensor_core::dispatch::GemmShape;

    use crate::error::CudaError;
    use crate::kernels_mma::{MMA_BK, MMA_BM, MMA_BN, MMA_STAGES, MMA_WARP_M, MMA_WARP_N};

    use super::TileCandidate;

    /// sm_121 の L1（per-SM）／L2（device-wide）帯域実測値（バイト/サイクル）。
    ///
    /// 単位は `docs/perf/sm121-device-attributes.md` の「単位に関する
    /// 注意」（L2 は device-wide 総帯域・L1 は per-SM 帯域）と厳密に
    /// 揃える。`estimate_candidate_cost` は `l1_bytes_per_cycle_per_sm`
    /// を呼び出し側から渡された `num_sms` 倍して device-wide 換算する。
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MeasuredBandwidth {
        /// L1（SMEM）の SM 単位あたり帯域（バイト/サイクル）。
        pub l1_bytes_per_cycle_per_sm: NonZeroU64,
        /// L2 のデバイス全体帯域（バイト/サイクル）。
        pub l2_bytes_per_cycle_device: NonZeroU64,
    }

    /// sm_121 実測帯域定数（A-2・イシュー #482 が「未実測・要実機実行」の
    /// まま安全側クローズしているため `None`）。
    ///
    /// `None` の間は `select_tile_config` がコストモデルを一切評価せず
    /// 固定選定テーブル（`FIXED_TILE_SELECTION`）へ fail-closed に
    /// フォールバックする。`docs/perf/sm121-device-attributes.md` に実測値
    /// が記入され、`docs/perf/cuda-gemm-cost-model-selection.md` の実機
    /// 比較手順（3 形状中 2 形状以上一致）が完了した時点で、同ドキュメント
    /// を正として本定数を `Some(...)` へ更新する。H100（DeepGEMM
    /// `sm90.hpp`）の帯域定数をここに書き写さないこと（本モジュール冒頭
    /// ドキュメンテーションコメント参照）。
    pub const SM121_MEASURED_BANDWIDTH: Option<MeasuredBandwidth> = None;

    /// [`SM121_MEASURED_BANDWIDTH`] が対象とする compute capability
    /// （major, minor）= sm_121（DGX Spark GB10）。
    ///
    /// [`super::select_tile_config_for_device`] はこの定数と
    /// `device.compute_capability()` を突き合わせてからでなければ
    /// [`SM121_MEASURED_BANDWIDTH`] を [`super::select_tile_config`] へ
    /// 渡さない（codex-review #675 P1 指摘対応: アーキテクチャ検証なしに
    /// sm_121 実測定数を任意デバイスへ適用すると、`SM121_MEASURED_BANDWIDTH`
    /// が `Some` へ更新された将来、sm_80/sm_90 等の他アーキテクチャでも
    /// 未検証の `CostModel` 選定結果を成功として返してしまう）。
    pub const SM121_COMPUTE_CAPABILITY: (i32, i32) = (12, 1);

    /// `estimate_candidate_cost` へ渡すデバイス実行時パラメータ。
    ///
    /// `num_sms` はハードコードせず、呼び出し元（`select_tile_config_for_device`）
    /// が `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT` から実行時に構築する。
    #[derive(Debug, Clone, Copy)]
    pub struct CostModelParams {
        pub num_sms: NonZeroU32,
        pub bandwidth: MeasuredBandwidth,
    }

    /// 分子・分母を保持した非負有理数（サイクル数の中間表現）。
    ///
    /// 浮動小数を経由せず `u128` の `checked_*` 演算のみで構築・比較する
    /// （本モジュールドキュメンテーションコメント §数値表現）。すべての
    /// 演算はオーバーフロー時に `None` を返し、呼び出し元が
    /// `CudaError::InvalidKernelDescriptor` へ変換する。
    #[derive(Debug, Clone, Copy)]
    struct CycleFraction {
        numerator: u128,
        denominator: NonZeroU128,
    }

    impl CycleFraction {
        fn new(numerator: u128, denominator: NonZeroU128) -> Self {
            Self {
                numerator,
                denominator,
            }
        }

        /// `numerator` を追加の整数倍率で拡大する（分母は変えない）。
        fn checked_scale_numerator(self, factor: u128) -> Option<Self> {
            let numerator = self.numerator.checked_mul(factor)?;
            Some(Self::new(numerator, self.denominator))
        }

        /// 分母へ追加の整数除数を乗じる（実質的に値を縮小する）。
        fn checked_scale_denominator(self, divisor: NonZeroU128) -> Option<Self> {
            let denominator = self
                .denominator
                .get()
                .checked_mul(divisor.get())
                .and_then(NonZeroU128::new)?;
            Some(Self::new(self.numerator, denominator))
        }

        /// `self >= other` を交差乗算で判定する（浮動小数を使わない比較。
        /// オーバーフロー時は `None`）。
        fn checked_ge(self, other: Self) -> Option<bool> {
            let lhs = self.numerator.checked_mul(other.denominator.get())?;
            let rhs = other.numerator.checked_mul(self.denominator.get())?;
            Some(lhs >= rhs)
        }

        /// 律速側支配（`max`）。交差乗算比較のオーバーフロー時は `None`。
        fn checked_max(self, other: Self) -> Option<Self> {
            if self.checked_ge(other)? {
                Some(self)
            } else {
                Some(other)
            }
        }
    }

    /// 候補 1 件のコスト評価結果。`cycles` は最終補正後のサイクル数
    /// （分数のまま保持し、[`compare_candidate_costs`] が交差乗算で
    /// 比較する。個々の候補ごとに異なる `num_blocks` を分母に持つため、
    /// 浮動小数への変換は行わない）。
    #[derive(Debug, Clone, Copy)]
    pub struct CandidateCost {
        cycles: CycleFraction,
    }

    /// `a` と `b` の推定サイクル数を比較する（`a < b` なら
    /// `std::cmp::Ordering::Less`）。交差乗算のオーバーフロー時は
    /// `CudaError::InvalidKernelDescriptor` を返す。
    ///
    /// `pub(crate)`: `select_best_tile_candidate`（本モジュール内）に加え、
    /// `cost_model_tests`（`gemm_auto.rs` 側の兄弟モジュール）が
    /// wave 効率補正・L1/L2 律速切り替えの検証で実際の比較結果を検査する
    /// ために公開する（`CandidateCost` のフィールドは private のままとし、
    /// 比較演算のみをテストへ公開する。#527 レビュー指摘: 合成コストを
    /// 破棄して自前で再計算したアサーションはモデル自体を検証しない）。
    pub(crate) fn compare_candidate_costs(
        a: &CandidateCost,
        b: &CandidateCost,
    ) -> Result<std::cmp::Ordering, CudaError> {
        let a_ge_b =
            a.cycles
                .checked_ge(b.cycles)
                .ok_or_else(|| CudaError::InvalidKernelDescriptor {
                    detail: "compare_candidate_costs: サイクル数比較の交差乗算が u128 の範囲を \
                         超過した"
                        .to_string(),
                })?;
        let b_ge_a =
            b.cycles
                .checked_ge(a.cycles)
                .ok_or_else(|| CudaError::InvalidKernelDescriptor {
                    detail: "compare_candidate_costs: サイクル数比較の交差乗算が u128 の範囲を \
                         超過した"
                        .to_string(),
                })?;
        Ok(match (a_ge_b, b_ge_a) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            // 両方 false は起こり得ない（全順序）が、網羅性のため
            // Equal 扱いにフォールバックする（`unreachable!` は使わない。
            // `.claude/rules/coding-rust.md`）。
            (false, false) => std::cmp::Ordering::Equal,
        })
    }

    /// `shape` に対する `candidate` 1 件のコストを見積もる（本モジュール
    /// ドキュメンテーションコメント §モデル構造・§トラフィックモデル参照）。
    ///
    /// `MMA_WARP_M`／`MMA_WARP_N`（`kernels_mma.rs`）から warp 分割数を
    /// 導出する（[`TileCandidate`] 自体は warp 分割数を保持しないため、
    /// [`enumerate_tile_candidates`]〈#524〉の規則 1 が保証する整数除算
    /// 可能性に依拠する）。
    pub fn estimate_candidate_cost(
        shape: GemmShape,
        candidate: &TileCandidate,
        params: &CostModelParams,
        bytes_per_element: NonZeroU32,
    ) -> Result<CandidateCost, CudaError> {
        let overflow_err = || CudaError::InvalidKernelDescriptor {
            detail: format!(
                "estimate_candidate_cost: shape={shape:?} candidate=(block_m={}, block_n={}, \
                 block_k={}) の見積り計算が整数範囲を超過した",
                candidate.block_m(),
                candidate.block_n(),
                candidate.block_k()
            ),
        };

        let block_m = u128::from(candidate.block_m().get());
        let block_n = u128::from(candidate.block_n().get());
        let block_k = u128::from(candidate.block_k().get());
        let bpe = u128::from(bytes_per_element.get());
        let num_sms = u128::from(params.num_sms.get());

        // num_blocks = ceil_div(M, block_m) * ceil_div(N, block_n)。
        let blocks_m = u128::from(shape.m)
            .checked_add(block_m.checked_sub(1).ok_or_else(overflow_err)?)
            .ok_or_else(overflow_err)?
            .checked_div(block_m)
            .ok_or_else(overflow_err)?;
        let blocks_n = u128::from(shape.n)
            .checked_add(block_n.checked_sub(1).ok_or_else(overflow_err)?)
            .ok_or_else(overflow_err)?
            .checked_div(block_n)
            .ok_or_else(overflow_err)?;
        let num_blocks = blocks_m.checked_mul(blocks_n).ok_or_else(overflow_err)?;
        if num_blocks == 0 {
            return Err(CudaError::InvalidKernelDescriptor {
                detail: format!(
                    "estimate_candidate_cost: shape={shape:?} に対する num_blocks が 0 に \
                     なった（M/N のいずれかが 0）"
                ),
            });
        }
        let num_blocks_nz = NonZeroU128::new(num_blocks).ok_or_else(overflow_err)?;

        // num_waves = ceil_div(num_blocks, num_sms)。
        let num_waves = num_blocks
            .checked_add(num_sms.checked_sub(1).ok_or_else(overflow_err)?)
            .ok_or_else(overflow_err)?
            .checked_div(num_sms)
            .ok_or_else(overflow_err)?;

        // K_padded = ceil_div(shape.k, block_k) * block_k。
        let k_steps = u128::from(shape.k)
            .checked_add(block_k.checked_sub(1).ok_or_else(overflow_err)?)
            .ok_or_else(overflow_err)?
            .checked_div(block_k)
            .ok_or_else(overflow_err)?;
        let k_padded = k_steps.checked_mul(block_k).ok_or_else(overflow_err)?;

        // L2 総トラフィック（バイト）。
        let l2_tile_bytes = block_m
            .checked_add(block_n)
            .ok_or_else(overflow_err)?
            .checked_mul(k_padded)
            .ok_or_else(overflow_err)?
            .checked_mul(bpe)
            .ok_or_else(overflow_err)?;
        let l2_traffic = num_blocks
            .checked_mul(l2_tile_bytes)
            .ok_or_else(overflow_err)?;

        // L1（SMEM 再読込）総トラフィック（バイト）。warp 分割数は
        // `enumerate_tile_candidates` 規則 1 により整数除算が保証される。
        let warps_m = u128::from(candidate.block_m().get() / MMA_WARP_M);
        let warps_n = u128::from(candidate.block_n().get() / MMA_WARP_N);
        let l1_tile_bytes_a = block_m
            .checked_mul(warps_n)
            .ok_or_else(overflow_err)?
            .checked_mul(k_padded)
            .ok_or_else(overflow_err)?
            .checked_mul(bpe)
            .ok_or_else(overflow_err)?;
        let l1_tile_bytes_b = block_n
            .checked_mul(warps_m)
            .ok_or_else(overflow_err)?
            .checked_mul(k_padded)
            .ok_or_else(overflow_err)?
            .checked_mul(bpe)
            .ok_or_else(overflow_err)?;
        let l1_tile_bytes = l1_tile_bytes_a
            .checked_add(l1_tile_bytes_b)
            .ok_or_else(overflow_err)?;
        let l1_traffic = num_blocks
            .checked_mul(l1_tile_bytes)
            .ok_or_else(overflow_err)?;

        // 帯域（バイト/サイクル）。L1 は per-SM 実測値を num_sms 倍して
        // device-wide 換算する（本モジュール `MeasuredBandwidth` doc 参照）。
        let l2_bandwidth =
            NonZeroU128::new(u128::from(params.bandwidth.l2_bytes_per_cycle_device.get()))
                .ok_or_else(overflow_err)?;
        let l1_bandwidth_per_sm = u128::from(params.bandwidth.l1_bytes_per_cycle_per_sm.get());
        let l1_bandwidth = l1_bandwidth_per_sm
            .checked_mul(num_sms)
            .and_then(NonZeroU128::new)
            .ok_or_else(overflow_err)?;

        let l2_cycles = CycleFraction::new(l2_traffic, l2_bandwidth);
        let l1_cycles = CycleFraction::new(l1_traffic, l1_bandwidth);
        let base_cycles = l1_cycles.checked_max(l2_cycles).ok_or_else(overflow_err)?;

        // wave 効率補正: cost = base_cycles * num_waves * num_sms / num_blocks。
        let wave_factor = num_waves.checked_mul(num_sms).ok_or_else(overflow_err)?;
        let cycles = base_cycles
            .checked_scale_numerator(wave_factor)
            .ok_or_else(overflow_err)?
            .checked_scale_denominator(num_blocks_nz)
            .ok_or_else(overflow_err)?;

        Ok(CandidateCost { cycles })
    }

    /// `candidates` の中からコスト最小の 1 件を決定的に選ぶ（DeepGEMM
    /// `compare` 相当。サイクル数昇順、同値時は列挙順〈block_m→n→k
    /// 昇順ソート済み〉の先頭を採るタイブレーク）。
    ///
    /// 候補ゼロ件は `Ok(None)`（`panic!` にはしない）。交差乗算の
    /// オーバーフローは握り潰さず `Err` として明示的に返す（コストモデル
    /// 自体が評価不能だったという事実を消さない）。呼び出し元の
    /// [`super::select_tile_config`] は `Ok(None)`・`Err` のいずれも
    /// 「コストモデルで評価不能」として同一に扱い、固定選定テーブルへ
    /// fail-closed にフォールバックする（未検証のまま候補を採用しない
    /// 安全側判断。`Err` を呼び出し元へ伝播させて上位で panic させたり
    /// 未定義の構成を返したりしない、という意味で「握り潰さない」）。
    pub fn select_best_tile_candidate(
        shape: GemmShape,
        candidates: &[TileCandidate],
        params: &CostModelParams,
        bytes_per_element: NonZeroU32,
    ) -> Result<Option<TileCandidate>, CudaError> {
        let mut best: Option<(TileCandidate, CandidateCost)> = None;
        for candidate in candidates {
            let cost = estimate_candidate_cost(shape, candidate, params, bytes_per_element)?;
            best = match best {
                None => Some((*candidate, cost)),
                Some((best_candidate, best_cost)) => {
                    // 厳密な `Less`（`cost < best_cost`）のときのみ更新する
                    // ことで、同値時は先に見つかった候補（列挙順の先頭）
                    // を保つ決定的タイブレークにする。
                    if compare_candidate_costs(&cost, &best_cost)? == std::cmp::Ordering::Less {
                        Some((*candidate, cost))
                    } else {
                        Some((best_candidate, best_cost))
                    }
                }
            };
        }
        Ok(best.map(|(candidate, _)| candidate))
    }

    /// [`FIXED_TILE_SELECTION`] の各 `NonZeroU32` フィールドをコンパイル
    /// 時に構築する（`match` + `panic!` は const context で評価される
    /// ため、実行時にこの分岐へ到達することはない。[`super::
    /// BYTES_PER_ELEMENT_F16`] と同型のコンパイル時確定パターン）。
    const fn nonzero_or_panic(value: u32) -> NonZeroU32 {
        match NonZeroU32::new(value) {
            Some(v) => v,
            None => panic!("gemm_auto::cost_model::nonzero_or_panic: value must be non-zero"),
        }
    }

    /// 実測裏付けのある現行本番構成（#494 実測記録が根拠。
    /// `kernels_mma.rs` の `MMA_BM/BN/BK`・`MMA_STAGES` を単一真実源とし、
    /// リテラルをここで独立に持たない）。[`SM121_MEASURED_BANDWIDTH`] が
    /// `None` の間、[`super::select_tile_config`] はこのテーブル値を返す。
    ///
    /// `TileCandidate` の全フィールドをコンパイル時定数として構築する
    /// （`.claude/rules/coding-rust.md`「本番経路で `unwrap()`/`expect()`
    /// を使わない」の徹底: 旧実装は実行時 `NonZeroU32::new(...).expect(...)`
    /// を使っており、値がコンパイル時に確定しているにも関わらず実行時
    /// panic 経路を持っていた。#527 レビュー指摘）。
    ///
    /// `smem_per_stage`／`smem_total` は f16（バイト幅 2）のカーネル族
    /// （現行本番の `kernels_mma.rs` は f16 専用。#494）を前提に算出する。
    /// `select_tile_config` が `DType::F32` に対して本定数を返す場合、
    /// `block_m`/`block_n`/`block_k`/`stages`（カーネルのブロック分割・
    /// パイプライン構造）は dtype に依存しないため引き続き有効だが、
    /// `smem_per_stage`/`smem_total`（バイト量）は f16 前提の値である
    /// 点に注意（F32 専用の固定テーブルは現時点で存在しない。mma.sync
    /// 経路自体が f16 専用のため）。
    pub const FIXED_TILE_SELECTION: TileCandidate = {
        let block_m = nonzero_or_panic(MMA_BM);
        let block_n = nonzero_or_panic(MMA_BN);
        let block_k = nonzero_or_panic(MMA_BK);
        let stages = nonzero_or_panic(MMA_STAGES);
        let bpe_u64 = 2u64; // f16 固定（現行本番構成は f16 専用。#494）。
        let a_tile_elems = MMA_BM as u64 * MMA_BK as u64;
        let b_tile_elems = MMA_BK as u64 * MMA_BN as u64;
        let smem_per_stage = (a_tile_elems + b_tile_elems) * bpe_u64;
        let smem_total = smem_per_stage * MMA_STAGES as u64;
        TileCandidate {
            block_m,
            block_n,
            block_k,
            stages,
            smem_per_stage,
            smem_total,
        }
    };
}

pub use cost_model::{
    CostModelParams, MeasuredBandwidth, SM121_COMPUTE_CAPABILITY, SM121_MEASURED_BANDWIDTH,
};

/// [`select_tile_config`] の選定根拠（実機検証・ログでの判別用。
/// イシュー #527 実装計画 §3）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileSelectionBasis {
    /// [`cost_model::SM121_MEASURED_BANDWIDTH`] が `Some` で、L1/L2 帯域
    /// コストモデルが候補を評価して選んだことを表す。
    CostModel,
    /// 帯域未実測（`None`）または候補選定不能（候補ゼロ件・オーバー
    /// フロー等）により、固定選定テーブルへフォールバックしたことを表す。
    FixedTable,
}

/// [`select_tile_config`] の返り値。選定されたタイル構成と、その選定
/// 根拠（コストモデル／固定テーブル）の両方を保持する。
#[derive(Debug, Clone, Copy)]
pub struct TileSelection {
    candidate: TileCandidate,
    basis: TileSelectionBasis,
}

impl TileSelection {
    /// 選定されたタイル構成。
    pub fn candidate(&self) -> TileCandidate {
        self.candidate
    }

    /// 選定根拠（[`TileSelectionBasis::CostModel`] か
    /// [`TileSelectionBasis::FixedTable`] か）。
    pub fn basis(&self) -> TileSelectionBasis {
        self.basis
    }
}

/// GEMM タイル構成の静的最良選定（Phase C-9b・イシュー #527）。
///
/// `measured` が `Some`（sm_121 実測帯域が確定済み）の場合のみ
/// [`enumerate_tile_candidates`]（#524）で候補を列挙し
/// `cost_model::select_best_tile_candidate` で最良 1 件を選ぶ。
/// 候補が 0 件、またはコストモデル自体が交差乗算オーバーフローで
/// 評価不能だった場合は、`measured` の有無に関わらず
/// `cost_model::FIXED_TILE_SELECTION`（実測裏付けのある現行本番構成
/// 64/128/32・stages 3）へ fail-closed にフォールバックする（本モジュール
/// `cost_model` ドキュメンテーションコメント参照）。
///
/// `measured` が `None`（未実測。既定は [`SM121_MEASURED_BANDWIDTH`]）の
/// 間は、コストモデルを一切評価せず常に固定選定テーブルを返す。これは
/// H100（DeepGEMM）帯域定数の暗黙流用を構造的に排除するための意図的な
/// 分岐であり、`measured` を「呼び出し元が明示的に注入しない限り
/// コストモデルは動かない」パラメータにしている。
///
/// `measured` が `None`、またはコストモデルが候補ゼロ件／評価不能で
/// 固定選定テーブルへフォールバックする場合、[`cost_model::
/// FIXED_TILE_SELECTION`] を無条件に成功として返さず、`validate_fixed_tile_selection`
/// で呼び出し元が渡した `shape`／`dtype`／`smem_budget_bytes` に対して
/// 実際に適用可能かを検証する（codex-review #675 P1 指摘: 検証なしで
/// 固定テーブルを返すと `smem_budget_bytes` 予算超過・`shape` の
/// mma.sync 整列制約違反・f16 専用メタデータの `DType::F32` への誤流用
/// を「成功した `TileSelection`」として呼び出し元へ伝えてしまい、
/// 公開 API `select_tile_config` の契約〈返す構成は入力制約を満たす〉
/// を破っていた）。満たさない場合は `Err(CudaError::InvalidKernelConfig)`
/// で選定不能を明示し、無効な構成を返さない（fail-closed。
/// `.claude/rules/security.md` A08 と同種の安全側判断）。
///
/// 本関数は選定結果を既定の本番 GEMM 経路（[`CudaGemmAuto::run_f16`]）
/// へ結線・適用しない（スコープ境界。カーネルが `#define` 固定のため、
/// 適用は実機実測・補正判定完了後の後続タスクに委ねる。
/// `docs/perf/cuda-gemm-cost-model-selection.md` 参照）。
pub fn select_tile_config(
    shape: GemmShape,
    dtype: DType,
    smem_budget_bytes: u64,
    num_sms: NonZeroU32,
    measured: Option<MeasuredBandwidth>,
) -> Result<TileSelection, CudaError> {
    let Some(bandwidth) = measured else {
        return fixed_table_selection_if_valid(shape, dtype, smem_budget_bytes);
    };

    let bytes_per_element = bytes_per_element_for(dtype);
    let candidates = enumerate_tile_candidates(shape, dtype, smem_budget_bytes);
    let params = CostModelParams { num_sms, bandwidth };

    match cost_model::select_best_tile_candidate(shape, &candidates, &params, bytes_per_element) {
        Ok(Some(candidate)) => Ok(TileSelection {
            candidate,
            basis: TileSelectionBasis::CostModel,
        }),
        // 候補ゼロ件・コストモデルのオーバーフロー評価不能のいずれも、
        // fail-closed に固定選定テーブルへ倒す（`.claude/rules/security.md`
        // A08「取り込み判断の迂回経路を作らない」と同種の安全側判断:
        // 静的モデルが評価不能な状況で未検証のまま候補を採用しない）。
        // ただしこのフォールバック先自体も無検証では返さない
        // （`fixed_table_selection_if_valid` 参照）。
        Ok(None) | Err(_) => fixed_table_selection_if_valid(shape, dtype, smem_budget_bytes),
    }
}

/// [`cost_model::FIXED_TILE_SELECTION`] を返す前に、呼び出し元が渡した
/// `shape`／`dtype`／`smem_budget_bytes` に対して実際に適用可能かを
/// [`validate_fixed_tile_selection`] で検証する（[`select_tile_config`]
/// の 2 つのフォールバック経路〈`measured = None`／候補評価不能〉が
/// 共有する単一の検証入口。判定ロジックが分裂して片方だけ緩和される
/// 静かな乖離を防ぐ。#521 と同種の教訓）。
fn fixed_table_selection_if_valid(
    shape: GemmShape,
    dtype: DType,
    smem_budget_bytes: u64,
) -> Result<TileSelection, CudaError> {
    validate_fixed_tile_selection(shape, dtype, smem_budget_bytes)?;
    Ok(TileSelection {
        candidate: cost_model::FIXED_TILE_SELECTION,
        basis: TileSelectionBasis::FixedTable,
    })
}

/// [`cost_model::FIXED_TILE_SELECTION`] が `shape`／`dtype`／
/// `smem_budget_bytes` の下で実際に適用可能かを検証する
/// （codex-review #675 P1 指摘対応）。
///
/// 検証する 4 条件（いずれか 1 つでも満たさなければ `Err`）:
/// 1. `dtype == DType::F16`: [`cost_model::FIXED_TILE_SELECTION`] の
///    `smem_per_stage`／`smem_total` は f16（2 bytes/element）前提で
///    コンパイル時算出されている（同定数のドキュメンテーションコメント
///    参照）。`DType::F32` に対して返すとメタデータが実際のバイト量と
///    一致しない。
/// 2. `validate_mma_alignment(shape.n, shape.k)`: `kernels_mma.rs` の
///    cp.async が要求する `n`/`k` 8 の倍数整列（[`enumerate_tile_candidates`]
///    規則 6 と同一契約）。固定テーブルも同じ `kernels_mma.rs` 起動 API
///    を経由するため、この整列を満たさない形状には適用できない。
/// 3. `validate_mma_grid_bounds(shape.m)`: `mma_launch_config` が構築する
///    グリッドの y 成分（`m.div_ceil(MMA_BM)`）が CUDA の grid dim y/z
///    上限（65,535）を超えないか（`gemm_mma.rs::validate_mma_grid_bounds`
///    と同一契約）。固定テーブルも同じ起動 API を経由するため、`m` が
///    `64 * 65535` を超える形状には適用できない（Cursor Bugbot 指摘。
///    PR #675 review。この検証を欠くと `dtype`／`n`／`k`／SMEM 予算の
///    条件をすべて満たしていても、実際のカーネル起動時に grid dim
///    上限超過で失敗する構成を `Ok` として返してしまい、本関数が
///    閉じるはずの「成功したのに使えない構成」という穴が `m` の
///    次元だけ再発する）。
/// 4. `smem_total <= smem_budget_bytes`: 固定テーブルの SMEM 使用量が
///    デバイスの実際の予算を超過していないか。
fn validate_fixed_tile_selection(
    shape: GemmShape,
    dtype: DType,
    smem_budget_bytes: u64,
) -> Result<(), CudaError> {
    if dtype != DType::F16 {
        return Err(CudaError::InvalidKernelConfig {
            detail: format!(
                "select_tile_config: FIXED_TILE_SELECTION の smem_per_stage/smem_total は \
                 f16 (2 bytes/element) 前提で算出済みのため、dtype={dtype:?} には \
                 適用できない（現行本番の kernels_mma.rs は f16 専用。#494）"
            ),
        });
    }

    validate_mma_alignment(shape.n, shape.k)?;
    validate_mma_grid_bounds(shape.m)?;

    let smem_total = cost_model::FIXED_TILE_SELECTION.smem_total();
    if smem_total > smem_budget_bytes {
        return Err(CudaError::InvalidKernelConfig {
            detail: format!(
                "select_tile_config: FIXED_TILE_SELECTION の smem_total={smem_total} が \
                 smem_budget_bytes={smem_budget_bytes} を超過するため適用できない"
            ),
        });
    }

    Ok(())
}

/// [`select_tile_config`] の実デバイス結線ヘルパー
/// （[`enumerate_tile_candidates_for_device`]・[`derive_stages_for_device`]
/// と同型。イシュー #527 実装計画 §3）。
///
/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`（SMEM 予算）・
/// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`（SM 数）を実行時取得し、
/// [`SM121_MEASURED_BANDWIDTH`] とともに [`select_tile_config`] へ渡す。
/// 属性取得の失敗は `CudaError::Driver`／`CudaError::InvalidKernelDescriptor`
/// として呼び出し元へそのまま伝播する（fail-closed。既存ヘルパーと同型）。
///
/// `device.compute_capability()` を `SM121_COMPUTE_CAPABILITY`（sm_121）と
/// 突き合わせ、一致する場合のみ [`SM121_MEASURED_BANDWIDTH`] を
/// [`select_tile_config`] へ渡す。不一致（sm_80/sm_90 等の他アーキテクチャ）
/// の場合は `measured = None` を渡し、[`select_tile_config`] を常に
/// 検証済み固定選定テーブル経路（`fixed_table_selection_if_valid`）へ
/// fail-closed に倒す（codex-review #675 P1 指摘対応: sm_121 実測定数
/// ——将来 `SM121_MEASURED_BANDWIDTH` が `Some` へ更新された時点——を
/// アーキテクチャ検証なしに他デバイスへ適用すると、未検証の `CostModel`
/// 選定結果を成功として返してしまう。現状 `SM121_MEASURED_BANDWIDTH` は
/// `None` のため本検証追加による挙動変化はない）。
pub fn select_tile_config_for_device(
    device: &CudaDevice,
    shape: GemmShape,
    dtype: DType,
) -> Result<TileSelection, CudaError> {
    let smem_budget_bytes = read_clamped_smem_budget_bytes(device)?;
    let raw_num_sms = device
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let num_sms_u32 =
        u32::try_from(raw_num_sms).map_err(|_| CudaError::InvalidKernelDescriptor {
            detail: format!(
                "select_tile_config_for_device: CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT \
                 returned a negative value ({raw_num_sms}), which cannot be a valid SM count"
            ),
        })?;
    let num_sms =
        NonZeroU32::new(num_sms_u32).ok_or_else(|| CudaError::InvalidKernelDescriptor {
            detail: "select_tile_config_for_device: CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT \
                 returned 0, which cannot be a valid SM count"
                .to_string(),
        })?;
    // sm_121 実測定数の他アーキテクチャへの誤流用防止（codex-review #675
    // P1 指摘）。`SM121_MEASURED_BANDWIDTH` は sm_121（DGX Spark GB10）で
    // しか意味を持たない定数のため、compute capability が一致しない
    // デバイスには決して渡さず `None` にフォールバックする（`select_tile_config`
    // の `measured = None` 経路は検証済み固定選定テーブルへ fail-closed
    // に倒れる。上記関数ドキュメンテーションコメント参照）。
    let measured = if device.compute_capability() == SM121_COMPUTE_CAPABILITY {
        SM121_MEASURED_BANDWIDTH
    } else {
        None
    };
    select_tile_config(shape, dtype, smem_budget_bytes, num_sms, measured)
}

/// naive／tiled／WMMA／`mma.sync` の全 GEMM カーネルを保持し、
/// `select_gemm_kernel` の判定結果に従って呼び分ける自動経路選択の入口。
///
/// `wmma` は `Option` とし、cc ゲート非対応・NVRTC コンパイル失敗時は
/// `None` のまま保持する（fail-safe。上記モジュールコメント参照）。
/// `caps` は `wmma`／`mma` の有無とは独立に「cc ゲートを満たすか」だけを
/// 表す（`select_gemm_kernel` の判定材料は cc のみであり、コンパイル
/// 成否は別軸のフォールバックとして扱う。`run_f16` 内のコメント参照）。
///
/// `mma`（[`crate::gemm_mma::CudaMmaGemm`]、`mma.sync`/`ldmatrix`/
/// `cp.async` パイプライン版 f16 Tensor Core 経路）は `wmma` と同型の
/// fail-soft 構築（#1152。`docs/dispatch-rules-design.md` §5.6）: cc
/// ゲート非対応（`CudaError::TensorCoreUnsupported`、cc < 8.0）・NVRTC
/// コンパイル失敗のいずれでも `None` のまま保持する。`run_f16` は
/// `mma` を最優先候補として読み（事前形状ゲート込み。#1156）、`mma` が
/// 使えない場合のフォールバックとして `wmma` を維持する。
pub struct CudaGemmAuto {
    gemm: CudaGemm,
    wmma: Option<CudaWmmaGemm>,
    /// `mma.sync` 系 f16 Tensor Core GEMM。cc ゲート非対応・NVRTC
    /// コンパイル失敗時は `None`（fail-soft。上記構造体コメント参照）。
    mma: Option<CudaMmaGemm>,
    /// `mma` が `None` の場合の構築失敗理由（`CudaError` の `Display`
    /// 文字列）。`gemm_wmma.rs::CudaWmmaGemm::wmma_f16_opt_error` と同じ
    /// 設計判断（PR #256 レビュー指摘: サイレントフォールバックで green
    /// になりうるため、失敗理由をテスト・診断から読める形で退避する）。
    /// `mma` が `Some` の場合は `None`。
    mma_construct_error: Option<String>,
    caps: DeviceCaps,
}

impl CudaGemmAuto {
    /// `device` 上で naive／tiled GEMM（[`CudaGemm`]）を構築し、cc ゲート
    /// を満たす場合のみ WMMA f16 GEMM（[`CudaWmmaGemm`]）の構築を試みる。
    ///
    /// naive／tiled の構築失敗（NVRTC 不在等）はそのまま `Err` として
    /// 呼び出し元へ伝播する（naive／tiled は最終フォールバックであり
    /// 失敗を握り潰すべきではない）。WMMA の構築失敗（cc ゲート非対応の
    /// `TensorCoreUnsupported`・NVRTC コンパイル失敗のいずれも）は
    /// `wmma = None` として握り潰し、`run_f16` を tiled へ倒す
    /// （fail-safe。`docs/dispatch-rules-design.md` §2.2）。
    ///
    /// `mma`（[`CudaMmaGemm`]）も `wmma` と同様の fail-soft で構築する
    /// （#1152。`cc < 8.0` の `TensorCoreUnsupported`・NVRTC コンパイル
    /// 失敗とも `None`。`docs/dispatch-rules-design.md` §5.6）。構築失敗
    /// 理由は `mma_construct_error` へ退避し [`Self::mma_unavailable_reason`]
    /// から読める（`mma` フィールド自体の値は `.ok()` と完全に同値）。
    /// `mma` は `wmma` の後に構築する（`gemm` の構築失敗はそのまま
    /// 呼び出し元へ伝播するため、naive/tiled すら構築できない環境で
    /// 無駄な NVRTC コンパイルを行わない）。
    ///
    /// `CudaMmaGemm::new` は `compile_ptx` 直呼び（LRU カーネル
    /// キャッシュ非経由）で base／swizzle 2 カーネルをコンパイルするため
    /// `CudaGemmAuto::new` 自体の構築コストは増える。ただし `CudaGemmAuto`
    /// を構築する本番経路は現状存在せず（`facade`／`backend-cuda::ops`／
    /// `bench-harness` のいずれからも未参照。`BackendOps::gemm` は f32
    /// tiled 固定）、到達するのは `tests/gemm_auto.rs`・
    /// `tests/dispatch_boundary.rs` の `#[ignore]` テストのみのため、
    /// 利用者向け起動コスト（`startup-bench`）への影響はない。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let gemm = CudaGemm::new(device)?;
        let wmma = CudaWmmaGemm::new(device).ok();
        let (mma, mma_construct_error) = match CudaMmaGemm::new(device) {
            Ok(mma) => (Some(mma), None),
            Err(err) => (None, Some(err.to_string())),
        };
        let caps = DeviceCaps::cuda(device.compute_capability());
        Ok(Self {
            gemm,
            wmma,
            mma,
            mma_construct_error,
            caps,
        })
    }

    /// `mma`（[`CudaMmaGemm`]、`mma.sync` 系 f16 Tensor Core 経路）が
    /// [`Self::new`] 時点でコンパイル・ロードに成功しているかを返す
    /// （`gemm_wmma.rs::CudaWmmaGemm::wmma_f16_opt_available` と同型）。
    ///
    /// 診断・テスト用の読み取り口であり利用者向け切替 API ではない
    /// （REQ-11・`docs/dispatch-rules-design.md` §5.6 判定規則 7）。
    /// `run_f16` は `MatrixUnit` 分岐で本フィールドの有無を
    /// `select_f16_matrix_unit_impl` へ渡し実装選択に用いる（#1156）。
    pub fn mma_available(&self) -> bool {
        self.mma.is_some()
    }

    /// [`Self::mma_available`] が `false` の場合の構築失敗理由
    /// （`mma_construct_error` の公開読み取り口。`CudaWmmaGemm::
    /// wmma_f16_opt_unavailable_reason` と同型）。`mma` が `Some` の場合は
    /// `None`。
    ///
    /// 診断・テスト用の読み取り口であり利用者向け切替 API ではない
    /// （REQ-11・`docs/dispatch-rules-design.md` §5.6 判定規則 7）。
    pub fn mma_unavailable_reason(&self) -> Option<&str> {
        self.mma_construct_error.as_deref()
    }

    /// f32 GEMM を自動経路選択で実行する。
    ///
    /// 現時点の決定表（`select_gemm_kernel`）では f32 は常に
    /// [`KernelKind::Tiled`] を返す（TF32 経路は #62／#186 の実測・承認
    /// まで既定採用を保留。§4）。将来 TF32 経路が有効化された際に
    /// `KernelKind::MatrixUnit` 分岐が生きるよう、match 式で明示的に
    /// 経路を切り分ける構造にしておく（未実装分岐は tiled へフォール
    /// バックし `unreachable!` は使わない。fail-safe）。
    pub fn run_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        let shape = GemmShape::new(m, n, k);
        match select_gemm_kernel(&self.caps, shape, DType::F32) {
            // TF32/f32 Tensor Core 経路は未実装（#62）のため、決定表が
            // 万一 MatrixUnit を返しても tiled へ倒す（現状の決定表は
            // f32 に対し常に Tiled を返すため、この分岐は到達しない
            // 契約だが、`select_gemm_kernel` 側の将来変更に対する
            // fail-safe として保持する）。
            KernelKind::MatrixUnit | KernelKind::Tiled => self.gemm.run_tiled_f32(a, b, m, n, k),
            KernelKind::Naive => self.gemm.run_naive_f32(a, b, m, n, k),
        }
    }

    /// f16 GEMM を自動経路選択で実行する。
    ///
    /// **設計目標**のフォールバック連鎖は `MatrixUnit(mma) →
    /// MatrixUnit(wmma) → Tiled → Naive`（`docs/dispatch-rules-design.md`
    /// §5.6）。第 1 層（`select_gemm_kernel`）が `KernelKind::MatrixUnit`
    /// を返した場合、第 2 層の実装選択は `Self::f16_matrix_unit_impl`
    /// （内部で `select_f16_matrix_unit_impl` を `MMA_PRIORITY_
    /// PRODUCTION_ENABLED` 付きで呼ぶ）が担う。
    ///
    /// **本番既定は `MMA_PRIORITY_PRODUCTION_ENABLED = true`（mma
    /// 優先・#1191 で有効化済み）である**（イシュー #1160: #1156 の
    /// ユーザー承認条件「切替前後を同一プロトコル・5 回計測中央値で
    /// 比較し、後退時は結線しない」〈§5.6〉自体は `run_f16` 経由
    /// 〈転送込み〉の auto 経路で GB10 実機実測し満たすことを確認済み
    /// で（`docs/perf/cuda-gemm-auto-f16-mma-switch.md`）、mma 優先の
    /// 本番有効化を保留していた K=4096 非後退ゲートの `MmaF16`
    /// baseline ceiling も #1190 でユーザー承認・反映済みのため
    /// （PR #1179 codex-review 指摘・[`MMA_PRIORITY_
    /// PRODUCTION_ENABLED`] docblock 参照）、#1191 で `true` へ復帰
    /// した。したがって `mma` が構築済みかつ事前形状ゲート（`n`／`k`
    /// のアラインメント・`m` のグリッド上限）を満たせば mma を優先し、
    /// 満たさなければ `wmma`（`Some` なら）を呼び、`wmma` も `None`
    /// なら tiled を呼ぶ。形状ゲートは呼び出し前の事前判定として行い、
    /// mma 実行が返す `Err` を捕捉して wmma へ再試行するエラー駆動
    /// フォールバックは採らない（カーネル起動失敗を静かに別経路で
    /// 覆い隠さないため。`docs/dispatch-rules-design.md` §5.6
    /// 判定規則 2・3）。
    /// `self.gemm`〈naive／tiled〉は `new` 成功時点で必ず存在するため、
    /// tiled 自体が失敗するケースはカーネル起動時エラー（`CudaError`）
    /// としてそのまま呼び出し元へ返る。
    ///
    /// 万一 `Self::f16_matrix_unit_impl` の判定結果と対応フィールドの
    /// 有無が食い違っても（あり得ない契約だが）tiled へ倒れる fail-safe
    /// な `match` 構造とし、`unwrap()`／`unreachable!` は使わない。
    pub fn run_f16(
        &self,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f16>, CudaError> {
        let shape = GemmShape::new(m, n, k);
        match select_gemm_kernel(&self.caps, shape, DType::F16) {
            KernelKind::MatrixUnit => {
                match (self.f16_matrix_unit_impl(m, n, k), &self.mma, &self.wmma) {
                    (F16MatrixUnitImpl::Mma, Some(mma), _) => mma.run_f16(a, b, m, n, k),
                    (F16MatrixUnitImpl::Wmma, _, Some(wmma)) => wmma.run_f16(a, b, m, n, k),
                    _ => self.gemm.run_tiled_f16(a, b, m, n, k),
                }
            }
            KernelKind::Tiled => self.gemm.run_tiled_f16(a, b, m, n, k),
            KernelKind::Naive => self.gemm.run_naive_f16(a, b, m, n, k),
        }
    }

    /// `run_f16` が `KernelKind::MatrixUnit` 判定時にどの実装
    /// （[`F16MatrixUnitImpl`]）を選ぶかを返す診断用アクセサ。
    ///
    /// 内部で `select_f16_matrix_unit_impl`（単一真実源の純関数）へ
    /// `MMA_PRIORITY_PRODUCTION_ENABLED`（本番既定値）・`self.mma`／
    /// `self.wmma` の有無・形状を渡すだけの薄いラッパー。診断・テスト用
    /// の読み取り口であり利用者向け切替 API ではない（REQ-11・
    /// `docs/dispatch-rules-design.md` §5.6 判定規則 7）。
    ///
    /// `MMA_PRIORITY_PRODUCTION_ENABLED` は `true`（#1191 で有効化
    /// 済み。[`MMA_PRIORITY_PRODUCTION_ENABLED`] docblock 参照）であり、
    /// `self.mma` が `Some` かつ整列形状であれば `Mma` を返す（非整列
    /// 形状・`mma` 未構築時は `Wmma`／`Tiled` を返す）。いずれの値でも
    /// `run_f16` が実際に呼ぶ実装と常に一致する（`run_f16` も同じ
    /// `MMA_PRIORITY_PRODUCTION_ENABLED` を渡すため）。
    ///
    /// codex-review PR #1177 指摘の是正（feature ゲート）: この内部
    /// ディスパッチ実装選択は「診断・テスト用」の意図に反して常時
    /// `pub` になっており、`crate::lib` の無条件 re-export と合わせて
    /// 利用者が依存しうる公開 API 面へ漏出していた。
    /// `SpecializedMmaKernelHandle` 等と同じ `internal-diagnostics`
    /// feature（既定 off）でのみ `pub` とし、通常ビルドでは crate 内部
    /// （`run_f16` 本体・単体テスト）限定の `pub(crate)` に留める。戻り値型
    /// [`F16MatrixUnitImpl`] 自体は feature ゲートせず定義するが、
    /// `crate::lib` の re-export（同 feature ゲート済み）を経由しない限り
    /// crate 外からは到達できない。
    #[cfg(feature = "internal-diagnostics")]
    pub fn f16_matrix_unit_impl(&self, m: u32, n: u32, k: u32) -> F16MatrixUnitImpl {
        select_f16_matrix_unit_impl(
            MMA_PRIORITY_PRODUCTION_ENABLED,
            self.mma.is_some(),
            self.wmma.is_some(),
            m,
            n,
            k,
        )
    }

    #[cfg(not(feature = "internal-diagnostics"))]
    pub(crate) fn f16_matrix_unit_impl(&self, m: u32, n: u32, k: u32) -> F16MatrixUnitImpl {
        select_f16_matrix_unit_impl(
            MMA_PRIORITY_PRODUCTION_ENABLED,
            self.mma.is_some(),
            self.wmma.is_some(),
            m,
            n,
            k,
        )
    }
}

/// [`CudaGemmAuto::run_f16`] が `KernelKind::MatrixUnit` 判定時に選ぶ
/// 第 2 層の実装（`docs/dispatch-rules-design.md` §5.6）。
///
/// `CudaMmaGemm`（`mma.sync`/`ldmatrix`/`cp.async` パイプライン）を
/// 最優先とし、事前形状ゲート非充足または未構築時は `CudaWmmaGemm` へ、
/// それも未構築なら `CudaGemm::run_tiled_f16` へ倒す。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F16MatrixUnitImpl {
    /// `CudaMmaGemm::run_f16`（`mma.sync` 系パイプライン）を使う。
    Mma,
    /// `CudaWmmaGemm::run_f16`（WMMA）を使う。
    Wmma,
    /// `CudaGemm::run_tiled_f16`（tiled）を使う。
    Tiled,
}

/// [`CudaGemmAuto::run_f16`] の `MatrixUnit` 分岐内で実装優先順位を
/// 決める単一真実源の純関数（`docs/dispatch-rules-design.md` §5.6
/// 判定規則 2）。GPU・`CudaDevice` に依存しないため GPU なしの CI でも
/// 単体テスト可能（`f16_matrix_unit_impl_tests`）。
///
/// 判定順序:
/// 1. `mma_available` かつ `validate_mma_alignment(n, k)` かつ
///    `validate_mma_grid_bounds(m)` がいずれも `Ok` → [`F16MatrixUnitImpl::Mma`]
///    （事前形状ゲート。カーネル起動後の `Err` を捕捉してのフォール
///    バックは行わない契約と対になる。§5.6 判定規則 3）
/// 2. 上記を満たさず `wmma_available` → [`F16MatrixUnitImpl::Wmma`]
/// 3. いずれも満たさない → [`F16MatrixUnitImpl::Tiled`]
///
/// `m == 0 || n == 0 || k == 0` の no-op 形状は特別扱いしない
/// （各経路の呼び出し先が同一契約の早期 return を持つため、判定結果が
/// `Wmma`／`Tiled` のいずれに倒れても出力は同じ空／ゼロ Vec。§5.6
/// 判定規則 4）。
///
/// `prefer_mma` は §5.6 で設計した `mma → wmma → tiled` 優先順位を
/// **有効化するかどうか**を呼び出し元（[`CudaGemmAuto::f16_matrix_unit_impl`]）
/// が渡す。`false` の場合は `mma_available`／形状ゲートの成否に関わらず
/// mma 経路を選ばず、常に `wmma → tiled` の従来（#1156 以前）と同じ優先
/// 順位で判定する。
///
/// イシュー #1160: #1156 のユーザー承認条件（`docs/dispatch-rules-
/// design.md` §5.6「性能の引き渡し」節）は「切替前後を同一プロトコル・
/// 5 回計測中央値で比較し、後退時は結線しない」であり、
/// `CudaGemmAuto::run_f16` 経由の auto 経路（転送込み）でこの比較を
/// GB10 実機で実施し非後退を確認した（`docs/perf/
/// cuda-gemm-auto-f16-mma-switch.md`）。本番既定（`CudaGemmAuto::
/// f16_matrix_unit_impl` が渡す [`MMA_PRIORITY_PRODUCTION_ENABLED`]）は
/// PR #1179 codex-review 指摘（P1）を受けて K=4096 非後退ゲートの
/// baseline ceiling 未承認の間 `false` へ差し戻していたが、#1190
/// （PR #1207）で ceiling がユーザー承認値で `BASELINES` へ反映された
/// ため、#1191 で `true`（mma 優先）へ復帰した。`prefer_mma` を引数
/// として明示する構造自体は変更していない。
pub(crate) fn select_f16_matrix_unit_impl(
    prefer_mma: bool,
    mma_available: bool,
    wmma_available: bool,
    m: u32,
    n: u32,
    k: u32,
) -> F16MatrixUnitImpl {
    if prefer_mma
        && mma_available
        && validate_mma_alignment(n, k).is_ok()
        && validate_mma_grid_bounds(m).is_ok()
    {
        return F16MatrixUnitImpl::Mma;
    }
    if wmma_available {
        return F16MatrixUnitImpl::Wmma;
    }
    F16MatrixUnitImpl::Tiled
}

/// [`select_f16_matrix_unit_impl`] の `prefer_mma` 引数へ渡す本番既定値。
///
/// `true`（mma 優先・#1191 で有効化済み）。性能 A/B 自体は
/// `CudaGemmAuto::run_f16` 経由（転送込み）の auto 経路で GB10 実機実測
/// 済みで、512/1024/2048 は after が base の run-median 以上、4096 は
/// #1130 の per-call アロケーション病態下で after の run-median が base
/// の 5 run 範囲内であることを確認している（`docs/perf/
/// cuda-gemm-auto-f16-mma-switch.md`）。
///
/// mma 優先の本番有効化を保留していた K=4096 ストレス形状の非後退
/// ゲート（`crates/backend-cuda/tests/gemm_auto.rs::
/// run_f16_k4096_stress_non_regression_route_aware`）が参照する
/// `ParityPath::MmaF16` baseline 行の `baseline_max_abs_diff_ceiling`／
/// `baseline_max_rel_err_ceiling` は、#1190（PR #1207）でユーザー承認
/// 済みの実機実測値が `BASELINES` へ反映されている（
/// `docs/perf/cuda-parity-baseline.md` §12.4〜§12.6）。
/// baseline 値の追加・更新は実機実測値のみ・人間承認必須（
/// `.claude/rules/coding-rust.md`「テスト・ベンチ」節）であり、この
/// 前提が満たされたことを受けて PR #1179 codex-review 指摘（P1）で
/// `false` へ差し戻していた本定数を #1191 で `true` へ復帰した。
/// 判定ロジック自体（`select_f16_matrix_unit_impl`）は変更していない。
/// 再び `false` へ戻す場合は、この決定を覆す実測根拠を
/// `docs/perf/cuda-gemm-auto-f16-mma-switch.md` に記録し、下記
/// コンパイル時ガード・`tests/gemm_auto.rs::
/// f16_matrix_unit_impl_reports_selected_implementation`・
/// `tests/tensor_core_real_device.rs` の期待値も同時に更新すること。
const MMA_PRIORITY_PRODUCTION_ENABLED: bool = true;

#[cfg(test)]
mod f16_matrix_unit_impl_tests {
    use super::*;

    /// mma・wmma とも未構築なら tiled へ倒れる（fail-safe の最終段。
    /// `prefer_mma` の値に関わらず成立する）。
    #[test]
    fn neither_available_selects_tiled() {
        assert_eq!(
            select_f16_matrix_unit_impl(true, false, false, 16, 16, 16),
            F16MatrixUnitImpl::Tiled
        );
        assert_eq!(
            select_f16_matrix_unit_impl(false, false, false, 16, 16, 16),
            F16MatrixUnitImpl::Tiled
        );
    }

    /// mma 未構築・wmma 構築済みなら wmma を選ぶ。
    #[test]
    fn only_wmma_available_selects_wmma() {
        assert_eq!(
            select_f16_matrix_unit_impl(true, false, true, 16, 16, 16),
            F16MatrixUnitImpl::Wmma
        );
    }

    /// `prefer_mma == true`（mma 優先。#1191 で
    /// `MMA_PRIORITY_PRODUCTION_ENABLED = true` として本番既定化済み）
    /// かつ mma・wmma とも構築済みで整列形状なら mma を優先する
    /// （§5.6 の設計目標）。
    #[test]
    fn prefer_mma_true_both_available_aligned_shape_selects_mma() {
        assert_eq!(
            select_f16_matrix_unit_impl(true, true, true, 16, 16, 16),
            F16MatrixUnitImpl::Mma
        );
    }

    /// `prefer_mma == false`（#1156 以前と同じ wmma 優先。本番既定は
    /// #1191 で `true` へ復帰済みのため、これは
    /// `select_f16_matrix_unit_impl` 自体の網羅テストとして維持する）
    /// では mma・wmma とも構築済み・整列形状でも mma を選ばず
    #[test]
    fn prefer_mma_false_both_available_aligned_shape_selects_wmma() {
        assert_eq!(
            select_f16_matrix_unit_impl(false, true, true, 16, 16, 16),
            F16MatrixUnitImpl::Wmma
        );
    }

    /// `MMA_PRIORITY_PRODUCTION_ENABLED` 自体が `true`（#1191 で
    /// ceiling 反映後に本番有効化済み）であることをコンパイル時に固定
    /// するリグレッションガード（PR #1179 codex-review 指摘〈P1〉対応
    /// の裏返し: `ParityPath::MmaF16` baseline 行の ceiling 反映
    /// （#1190・PR #1207）を前提に `true` へ復帰したこの判断が、他の
    /// テストの `prefer_mma` 明示引数だけでは検知できない形で無断で
    /// `false` へ差し戻されるのを防ぐ。`clippy::assertions_on_constants`
    /// を避けるため `#[test]` ではなく `const` ブロックで表現する）。
    const _: () = assert!(
        MMA_PRIORITY_PRODUCTION_ENABLED,
        "MMA_PRIORITY_PRODUCTION_ENABLED を false へ戻す場合は、\
         #1191 の結線判断（docs/perf/cuda-gemm-auto-f16-mma-switch.md）\
         を覆す実機実測根拠を記録したうえで、\
         tests/gemm_auto.rs::f16_matrix_unit_impl_reports_selected_implementation \
         と tests/tensor_core_real_device.rs の期待値も同時に更新すること"
    );

    /// wmma 未構築でも mma が使えて整列形状なら mma を選ぶ（`prefer_mma
    /// == true` の場合。wmma の有無は mma 選択の必要条件ではない）。
    #[test]
    fn mma_available_without_wmma_aligned_shape_selects_mma() {
        assert_eq!(
            select_f16_matrix_unit_impl(true, true, false, 16, 16, 16),
            F16MatrixUnitImpl::Mma
        );
    }

    /// n が 8 の倍数でない（`validate_mma_alignment` 非充足）場合は
    /// `prefer_mma == true` で mma が使えても wmma へフォールバックする。
    #[test]
    fn mma_available_but_n_misaligned_falls_back_to_wmma() {
        assert_eq!(
            select_f16_matrix_unit_impl(true, true, true, 16, 12, 16),
            F16MatrixUnitImpl::Wmma
        );
    }

    /// k が 8 の倍数でない場合も同様に wmma へフォールバックする。
    #[test]
    fn mma_available_but_k_misaligned_falls_back_to_wmma() {
        assert_eq!(
            select_f16_matrix_unit_impl(true, true, true, 16, 16, 12),
            F16MatrixUnitImpl::Wmma
        );
    }

    /// 形状ゲート非充足かつ wmma も未構築なら tiled まで倒れる。
    #[test]
    fn mma_available_misaligned_and_no_wmma_falls_back_to_tiled() {
        assert_eq!(
            select_f16_matrix_unit_impl(true, true, false, 16, 12, 16),
            F16MatrixUnitImpl::Tiled
        );
    }

    /// grid_dim.y 上限（65,535）超過時は `prefer_mma == true` で mma が
    /// 使えても wmma へ倒す。
    #[test]
    fn mma_available_grid_bounds_exceeded_falls_back_to_wmma() {
        // `validate_mma_grid_bounds` は `m.div_ceil(MMA_BM) <= 65_535` を
        // 要求する（`gemm_mma.rs`）。`MMA_BM` に依存しない形で確実に
        // 上限超過させるため、大きめの m を直接指定する。
        let m = 65_535u32.saturating_mul(256).saturating_add(1);
        assert_eq!(
            select_f16_matrix_unit_impl(true, true, true, m, 16, 16),
            F16MatrixUnitImpl::Wmma
        );
    }

    /// no-op 形状（k == 0）は特別扱いせず、通常どおり判定規則へ通す
    /// （n = 7 は非整列のため wmma へ倒れる。§5.6 判定規則 4）。
    #[test]
    fn noop_shape_is_not_special_cased() {
        assert_eq!(
            select_f16_matrix_unit_impl(true, true, true, 8, 7, 0),
            F16MatrixUnitImpl::Wmma
        );
    }
}

// GPU 不要（純関数 [`enumerate_tile_candidates`] のみを対象とし、
// `CudaDevice` を構築する結線ヘルパー [`enumerate_tile_candidates_for_device`]
// はカバーしない。`nvrtc.rs` の `derive_pipeline_stages` テスト群と同型の
// 方針。イシュー #524 実装計画 §5 ステップ 6）。
#[cfg(test)]
mod tile_candidate_tests {
    use super::*;

    /// 実測不要な代表的 SMEM 予算（`STATIC_SMEM_BUDGET_CAP_BYTES` と同値。
    /// 48KiB。現行本番構成〈64/128/32・f16〉が成立する予算として全テスト
    /// で共通利用する）。
    const FULL_BUDGET_BYTES: u64 = STATIC_SMEM_BUDGET_CAP_BYTES;

    /// 規則 1〜4（構造的制約）を検証する（規則 5 は `derive_pipeline_stages`
    /// が `Ok` を返したことそのものが充足の証拠であるため、列挙結果に
    /// 残っている時点で成立している）。
    fn assert_candidate_satisfies_structural_rules(candidate: &TileCandidate, bpe: u64) {
        let bm = candidate.block_m().get();
        let bn = candidate.block_n().get();
        let bk = candidate.block_k().get();

        // 規則 1: warp 分解可能性。
        assert_eq!(
            bm % MMA_WARP_M,
            0,
            "block_m={bm} must be a multiple of MMA_WARP_M"
        );
        assert_eq!(
            bn % MMA_WARP_N,
            0,
            "block_n={bn} must be a multiple of MMA_WARP_N"
        );
        assert_eq!(bk % MMA_K, 0, "block_k={bk} must be a multiple of MMA_K");
        assert!(
            bk / MMA_K >= MIN_MMA_K_STEPS_PER_STAGE,
            "block_k={bk} must yield at least MIN_MMA_K_STEPS_PER_STAGE \
             ({MIN_MMA_K_STEPS_PER_STAGE}) kstep iterations per stage \
             (kernels_mma.rs MMA_K_STEPS_PER_STAGE >= 2 contract, #494)"
        );

        // 規則 2: per-block スレッド数上限。
        let threads = (bm / MMA_WARP_M) * (bn / MMA_WARP_N) * 32;
        assert!(threads <= 1024, "threads={threads} must not exceed 1024");

        // 規則 3: レジスタ不足の事前枝刈り。
        assert!(
            !(bm > 128 && bn > 128),
            "block_m={bm} and block_n={bn} must not both exceed 128"
        );

        // 規則 4: cp.async 16B アライメント粒度。
        assert_eq!(
            (u64::from(bk) * bpe) % 16,
            0,
            "block_k={bk} * bpe={bpe} must be 16B aligned"
        );
        assert_eq!(
            (u64::from(bn) * bpe) % 16,
            0,
            "block_n={bn} * bpe={bpe} must be 16B aligned"
        );
    }

    /// 受け入れ基準の中核: 代表形状で列挙した全候補が規則 1〜4 を満たし、
    /// `derive_pipeline_stages` が同一予算で `Ok` を返すこと（不正構成が
    /// 候補に残らないこと）。
    #[test]
    fn all_candidates_satisfy_pruning_rules_for_representative_shapes() {
        let bpe_f16 = u64::from(bytes_per_element_for(DType::F16).get());
        for shape in [
            GemmShape::new(4096, 4096, 4096),
            GemmShape::new(2048, 2048, 2048),
            GemmShape::new(1024, 1024, 1024),
            GemmShape::new(32, 4096, 4096),
        ] {
            let candidates = enumerate_tile_candidates(shape, DType::F16, FULL_BUDGET_BYTES);
            assert!(
                !candidates.is_empty(),
                "shape={shape:?} must yield at least one candidate at the full SMEM budget"
            );
            for candidate in &candidates {
                assert_candidate_satisfies_structural_rules(candidate, bpe_f16);
                // 規則 5 の二重検証: 列挙結果の段数が `derive_pipeline_stages`
                // の再計算と一致すること（判定ロジックの分裂検知）。
                let recomputed = derive_pipeline_stages(
                    candidate.block_m(),
                    candidate.block_n(),
                    candidate.block_k(),
                    bytes_per_element_for(DType::F16),
                    FULL_BUDGET_BYTES,
                )
                .expect("a listed candidate must re-derive Ok under the same budget");
                assert_eq!(recomputed, candidate.stages());
            }
        }
    }

    /// 両次元同時 128 超（レジスタ不足）候補は予算を大きくしても出現
    /// しないこと。
    #[test]
    fn candidates_never_exceed_128_on_both_dimensions_even_with_huge_budget() {
        let shape = GemmShape::new(4096, 4096, 4096);
        let candidates = enumerate_tile_candidates(shape, DType::F16, u64::MAX);
        for candidate in &candidates {
            let bm = candidate.block_m().get();
            let bn = candidate.block_n().get();
            assert!(
                !(bm > 128 && bn > 128),
                "found disallowed candidate block_m={bm} block_n={bn}"
            );
        }
    }

    /// スレッド数 1024 超の構成（例: block_m=128 × block_n=256）が候補に
    /// 残らないこと。
    #[test]
    fn candidates_never_exceed_1024_threads_per_block() {
        let shape = GemmShape::new(4096, 4096, 4096);
        let candidates = enumerate_tile_candidates(shape, DType::F16, u64::MAX);
        for candidate in &candidates {
            let bm = candidate.block_m().get();
            let bn = candidate.block_n().get();
            let threads = (bm / MMA_WARP_M) * (bn / MMA_WARP_N) * 32;
            assert!(
                threads <= 1024,
                "found disallowed candidate block_m={bm} block_n={bn} threads={threads}"
            );
            // block_m=128 と block_n=256 の組合せそのものは規則 3
            // （レジスタ不足）でも棄却されるはずだが、規則 2 自体の
            // 有効性を独立に検証するため 1024 上限を明示的に検査する。
        }
        assert!(
            !candidates
                .iter()
                .any(|c| c.block_m().get() == 128 && c.block_n().get() == 256),
            "block_m=128 x block_n=256 (1024 threads, and also >128 on both dims) must be pruned"
        );
    }

    /// 予算が極端に小さい場合（全構成が最小段数未達）に空 `Vec` を返し
    /// panic しないこと。
    #[test]
    fn empty_candidates_when_budget_is_too_small_for_any_configuration() {
        let shape = GemmShape::new(4096, 4096, 4096);
        let candidates = enumerate_tile_candidates(shape, DType::F16, 0);
        assert!(
            candidates.is_empty(),
            "a zero SMEM budget must yield no candidates, not panic"
        );
    }

    /// 現行本番構成（64/128/32）が 48KiB 予算・M=4096 で候補に含まれ、
    /// 段数が 3 以上であること（列挙が現実の有効構成を取りこぼさない
    /// 健全性検査）。
    #[test]
    fn production_tile_configuration_is_present_in_candidates() {
        let shape = GemmShape::new(4096, 4096, 4096);
        let candidates = enumerate_tile_candidates(shape, DType::F16, FULL_BUDGET_BYTES);
        let production = candidates.iter().find(|c| {
            c.block_m().get() == 64 && c.block_n().get() == 128 && c.block_k().get() == 32
        });
        let production =
            production.expect("production tile configuration (64/128/32) must be enumerated");
        assert!(
            production.stages().get() >= 3,
            "production configuration must derive at least 3 pipeline stages"
        );
    }

    /// 小 M 候補（block_m=32）が `m <= 64` の形状でのみ出現すること。
    #[test]
    fn small_block_m_candidate_appears_only_for_small_m_shapes() {
        let small_shape = GemmShape::new(64, 4096, 4096);
        let small_candidates =
            enumerate_tile_candidates(small_shape, DType::F16, FULL_BUDGET_BYTES);
        assert!(
            small_candidates.iter().any(|c| c.block_m().get() == 32),
            "m=64 (<= SMALL_M_THRESHOLD) must include a block_m=32 candidate"
        );

        let large_shape = GemmShape::new(65, 4096, 4096);
        let large_candidates =
            enumerate_tile_candidates(large_shape, DType::F16, FULL_BUDGET_BYTES);
        assert!(
            !large_candidates.iter().any(|c| c.block_m().get() == 32),
            "m=65 (> SMALL_M_THRESHOLD) must not include a block_m=32 candidate"
        );
    }

    /// block_k=[`MMA_K`]（16）はカーネル側の K ループ構造契約
    /// （`kernels_mma.rs` の `MMA_K_STEPS_PER_STAGE >= 2` コンパイル時
    /// assert・#494 受け入れ基準 3 項）に違反するため、`block_m % MMA_K
    /// == 0` を満たしていても候補に残らないこと（#524 レビュー指摘の
    /// 再現テスト: 修正前は `enumerate_tile_candidates(GemmShape::new(
    /// 4096, 4096, 4096), DType::F16, STATIC_SMEM_BUDGET_CAP_BYTES)` で
    /// block_k=16 の候補が 24 件残存していた）。
    #[test]
    fn block_k_equal_to_mma_k_is_always_pruned() {
        let shape = GemmShape::new(4096, 4096, 4096);
        let candidates = enumerate_tile_candidates(shape, DType::F16, FULL_BUDGET_BYTES);
        assert!(
            !candidates.iter().any(|c| c.block_k().get() == MMA_K),
            "block_k == MMA_K ({MMA_K}) violates MMA_K_STEPS_PER_STAGE >= 2 \
             and must never appear in enumerated candidates"
        );
    }

    /// 規則 6 の再現テスト（#524 レビュー指摘、PR #671 対応）: GEMM 形状
    /// 自体が `gemm_mma::validate_mma_alignment` の cp.async 整列制約
    /// （`k % 8 == 0 && n % 8 == 0`）を満たさない場合、候補のブロック
    /// タイル寸法（block_k/block_n）は常に 8 の倍数であっても、実際の
    /// グローバルメモリ行ストライドである `shape.k`/`shape.n` が不整列
    /// なため `kernels_mma.rs` の `cp.async` は実行不能である。
    /// `enumerate_tile_candidates` は shape 全体を棄却して空 `Vec` を
    /// 返さねばならない（修正前は shape.k/shape.n を検査せず実行不能な
    /// 候補を返していた）。
    #[test]
    fn misaligned_shape_yields_no_candidates() {
        // n=9 (% 8 != 0), k=17 (% 8 != 0)。
        let shape = GemmShape::new(4096, 9, 17);
        let candidates = enumerate_tile_candidates(shape, DType::F16, FULL_BUDGET_BYTES);
        assert!(
            candidates.is_empty(),
            "misaligned shape (n=9, k=17) must yield zero candidates, got {candidates:?}"
        );
    }

    /// 規則 6 の境界確認: `n`・`k` の一方のみが不整列な場合も棄却される
    /// こと（`validate_mma_alignment` は `n`/`k` の両方を独立に検査する
    /// ため、片方だけの不整列でも棄却が必要）。
    #[test]
    fn misaligned_shape_on_either_axis_alone_yields_no_candidates() {
        let n_misaligned = GemmShape::new(4096, 9, 4096);
        assert!(
            enumerate_tile_candidates(n_misaligned, DType::F16, FULL_BUDGET_BYTES).is_empty(),
            "n=9 alone (k aligned) must still yield zero candidates"
        );

        let k_misaligned = GemmShape::new(4096, 4096, 17);
        assert!(
            enumerate_tile_candidates(k_misaligned, DType::F16, FULL_BUDGET_BYTES).is_empty(),
            "k=17 alone (n aligned) must still yield zero candidates"
        );
    }

    /// 同一入力での 2 回呼び出しが同一結果（決定性・ソート順）であること。
    #[test]
    fn enumeration_is_deterministic_and_sorted() {
        let shape = GemmShape::new(4096, 4096, 4096);
        let first = enumerate_tile_candidates(shape, DType::F16, FULL_BUDGET_BYTES);
        let second = enumerate_tile_candidates(shape, DType::F16, FULL_BUDGET_BYTES);
        assert_eq!(first, second, "enumeration must be deterministic");

        let mut sorted = first.clone();
        sorted.sort_by_key(|c| (c.block_m().get(), c.block_n().get(), c.block_k().get()));
        assert_eq!(first, sorted, "enumeration must already be in sorted order");
    }

    /// f16/f32 の `bytes_per_element` 差でアライメント枝刈りが変化する
    /// ケース: f32（4B）は f16（2B）よりバイト幅が大きいぶん 16B
    /// アライメント制約を満たしやすく、少なくとも同じ形状・予算で候補が
    /// 空にならないこと。両 dtype とも block_n の最小候補（16）×
    /// bytes_per_element が 16B 境界と整合すること（16*2=32, 16*4=64、
    /// いずれも 16 の倍数）を確認する。
    #[test]
    fn f16_and_f32_candidates_both_satisfy_their_own_alignment_granularity() {
        let shape = GemmShape::new(4096, 4096, 4096);

        let f16_candidates = enumerate_tile_candidates(shape, DType::F16, FULL_BUDGET_BYTES);
        assert!(!f16_candidates.is_empty(), "f16 must yield candidates");
        let bpe_f16 = u64::from(bytes_per_element_for(DType::F16).get());
        for candidate in &f16_candidates {
            assert_candidate_satisfies_structural_rules(candidate, bpe_f16);
        }

        let f32_candidates = enumerate_tile_candidates(shape, DType::F32, FULL_BUDGET_BYTES);
        assert!(!f32_candidates.is_empty(), "f32 must yield candidates");
        let bpe_f32 = u64::from(bytes_per_element_for(DType::F32).get());
        for candidate in &f32_candidates {
            assert_candidate_satisfies_structural_rules(candidate, bpe_f32);
        }
    }
}

// イシュー #519（C-7）: [`dim_specs_for`]／[`specialized_mma_config`]／
// [`specialized_mma_descriptor`] のユニットテスト群。全て GPU 不要（純関数
// ／NVRTC 実コンパイルを伴わない `render_mma_f16` の文字列組み立てのみ）。
#[cfg(test)]
mod compiled_dims_selection_tests {
    use std::collections::HashSet;

    use super::*;

    /// 代表 shape（全次元が異なる値。sentinel 正規化の衝突を避けるため）。
    fn sample_shape() -> GemmShape {
        GemmShape::new(64, 128, 256)
    }

    /// [`CudaKernelCacheKey`] を構築する際の環境パラメータ・ソース文字列
    /// （テスト内では descriptor 側の差分のみを見たいため固定値を使う。
    /// `nvrtc.rs` の `sample_key` と同型の判断。`source` 引数は C-5・#514
    /// で `CudaKernelCacheKey::new` へ追加された）。
    fn cache_key_for(
        descriptor: crate::nvrtc::CudaKernelDescriptor,
    ) -> crate::nvrtc::CudaKernelCacheKey {
        crate::nvrtc::CudaKernelCacheKey::new(
            descriptor,
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
            "// fixed source for descriptor-only comparison".to_string(),
        )
    }

    // 受け入れ基準 1: 次元ごとに定数化／動的を選択できること。
    #[test]
    fn dim_specs_for_selects_static_only_for_compiled_dims() {
        let shape = sample_shape();

        assert_eq!(
            dim_specs_for(shape, CompiledDims::DYNAMIC_ALL),
            (DimSpec::Dynamic, DimSpec::Dynamic, DimSpec::Dynamic)
        );
        assert_eq!(
            dim_specs_for(shape, CompiledDims::STATIC_NK),
            (
                DimSpec::Dynamic,
                DimSpec::Static(shape.n),
                DimSpec::Static(shape.k)
            )
        );
        assert_eq!(
            dim_specs_for(shape, CompiledDims::STATIC_MNK),
            (
                DimSpec::Static(shape.m),
                DimSpec::Static(shape.n),
                DimSpec::Static(shape.k)
            )
        );
        // 任意組合せ（M/K のみ定数化・N は動的）。
        assert_eq!(
            dim_specs_for(shape, CompiledDims::new(true, false, true)),
            (
                DimSpec::Static(shape.m),
                DimSpec::Dynamic,
                DimSpec::Static(shape.k)
            )
        );
    }

    // 受け入れ基準 2: 非定数化次元は実行時引数のまま扱われること。
    // `MmaKernelConfig::validate_launch_shape`（`DimSpec::matches_launch_dim`
    // 経由）は `Dynamic` を常に許容し、`Static` は展開元の値と一致しない
    // 実起動 shape を拒否する（`kernels_mma.rs` の既存契約）。この挙動を
    // 組合せ単位で確認することで「動的次元は任意の実行時引数を受け付ける」
    // ことを間接検証する。
    #[test]
    fn non_compiled_dims_accept_arbitrary_runtime_values() {
        let shape = sample_shape();
        let (cfg, _rendered) = specialized_mma_config(shape, CompiledDims::STATIC_NK)
            .expect("STATIC_NK config for an 8-aligned n/k shape must render successfully");

        // M は動的（任意値で Ok）、N/K は展開元の値と一致する必要がある。
        assert!(cfg.validate_launch_shape(999_999, shape.n, shape.k).is_ok());
        assert!(cfg.validate_launch_shape(shape.m, shape.n, shape.k).is_ok());
        assert!(
            cfg.validate_launch_shape(shape.m, shape.n + 8, shape.k)
                .is_err()
        );
        assert!(
            cfg.validate_launch_shape(shape.m, shape.n, shape.k + 16)
                .is_err()
        );
    }

    // 受け入れ基準 3（中核）: 定数化した次元のみがキャッシュエントリ数へ
    // 反映され、動的次元の値変動ではエントリが増殖しないこと。
    #[test]
    fn static_nk_collapses_varying_m_into_a_single_cache_entry() {
        // n/k は 8 の倍数で固定し M のみ多数変動させる（`STATIC_NK` では
        // M は sentinel 0 に正規化されるため、M の実値差はキャッシュキーへ
        // 反映されない）。
        let keys: HashSet<_> = (1..=8)
            .map(|i| GemmShape::new(i * 8, 128, 256))
            .map(|shape| {
                specialized_mma_descriptor(shape, CompiledDims::STATIC_NK)
                    .expect("n=128/k=256 are 8-aligned so specialization must succeed")
            })
            .map(cache_key_for)
            .collect();
        assert_eq!(
            keys.len(),
            1,
            "STATIC_NK must collapse all M variations into a single cache entry"
        );
    }

    #[test]
    fn static_nk_produces_distinct_entries_for_varying_n() {
        // block_n/block_k はタイル定数（既定 config）で固定なので、shape
        // 側の n を変えても render 検証（8 の倍数）には影響しない。K は
        // 固定・N のみ変動させ、各 N ごとに別エントリになることを確認する。
        let ns = [128u32, 136, 144, 152];
        let keys: HashSet<_> = ns
            .iter()
            .map(|&n| GemmShape::new(64, n, 256))
            .map(|shape| {
                specialized_mma_descriptor(shape, CompiledDims::STATIC_NK)
                    .expect("8-aligned n must specialize successfully")
            })
            .map(cache_key_for)
            .collect();
        assert_eq!(
            keys.len(),
            ns.len(),
            "STATIC_NK must produce one distinct cache entry per distinct N"
        );
    }

    #[test]
    fn static_mnk_produces_distinct_entries_for_varying_m() {
        let ms = [8u32, 16, 24, 32];
        let keys: HashSet<_> = ms
            .iter()
            .map(|&m| GemmShape::new(m, 128, 256))
            .map(|shape| {
                specialized_mma_descriptor(shape, CompiledDims::STATIC_MNK)
                    .expect("valid shape must specialize successfully")
            })
            .map(cache_key_for)
            .collect();
        assert_eq!(
            keys.len(),
            ms.len(),
            "STATIC_MNK must produce one distinct cache entry per distinct M"
        );
    }

    #[test]
    fn dynamic_all_collapses_every_shape_into_a_single_cache_entry() {
        let keys: HashSet<_> = [
            GemmShape::new(64, 128, 256),
            GemmShape::new(4096, 4096, 4096),
            GemmShape::new(8, 8, 16),
        ]
        .into_iter()
        .map(|shape| {
            specialized_mma_descriptor(shape, CompiledDims::DYNAMIC_ALL)
                .expect("DYNAMIC_ALL never depends on shape alignment for its own dims")
        })
        .map(cache_key_for)
        .collect();
        assert_eq!(
            keys.len(),
            1,
            "DYNAMIC_ALL must collapse every shape into a single cache entry"
        );
    }

    // 同一 shape に対して全 8 通りの `CompiledDims` 組合せを適用しても、
    // 組合せ間で誤って衝突・誤ヒットしないこと（A08: 異なる定数化組合せ
    // 間の誤ヒットは誤った特化カーネルの再利用につながる）。
    #[test]
    fn all_eight_compiled_dims_combinations_yield_distinct_entries_for_same_shape() {
        let shape = sample_shape();
        let mut keys = HashSet::new();
        for m in [false, true] {
            for n in [false, true] {
                for k in [false, true] {
                    let descriptor = specialized_mma_descriptor(shape, CompiledDims::new(m, n, k))
                        .expect("sample_shape is 8-aligned on n/k for every combination");
                    keys.insert(cache_key_for(descriptor));
                }
            }
        }
        assert_eq!(
            keys.len(),
            8,
            "all 8 CompiledDims combinations must yield distinct cache entries for one shape"
        );
    }

    // fail-closed: 定数化対象次元の実値が 0 だと `Err` になり panic しない
    // こと。`specialized_mma_descriptor` は内部で `specialized_mma_config`
    // （`render_mma_f16` 経由）を先に呼ぶため、`dim_specs_for` が
    // `DimSpec::Static(0)` を生成するこの入力は `kernels_mma::
    // validate_mma_kernel_config` の「静的次元ゼロ拒否」検査
    // （`CudaError::InvalidKernelConfig`）が先に発火する。`CudaKernelDescriptor
    // ::new` 自身の同種検査（`nvrtc.rs`。`new_rejects_zero_value_for_a_
    // compiled_dim` で直接検証済み）は、descriptor を render を経ずに
    // 直接構築する呼び出し元に対する独立の多層防御であり、本関数のような
    // 合成経路では render 側の検査が先に fail-closed に遮断する
    // （判定ロジックを二重管理しない設計。本関数ドキュメンテーション
    // コメント参照）。
    #[test]
    fn specialized_mma_descriptor_rejects_zero_value_for_a_compiled_dim() {
        let result =
            specialized_mma_descriptor(GemmShape::new(0, 128, 256), CompiledDims::STATIC_MNK);
        assert!(matches!(result, Err(CudaError::InvalidKernelConfig { .. })));
    }

    // fail-closed: N/K を定数化した shape の N/K が 8 の倍数でないと
    // `render_mma_f16`（`kernels_mma::validate_mma_kernel_config`）の
    // 検査で `Err` になり、それが `specialized_mma_config` からそのまま
    // 伝播すること（判定ロジックを二重管理せず render 側へ委譲する契約
    // の検証）。
    #[test]
    fn specialized_mma_config_rejects_non_eight_aligned_static_n_or_k() {
        let n_misaligned =
            specialized_mma_config(GemmShape::new(64, 127, 256), CompiledDims::STATIC_NK);
        assert!(matches!(
            n_misaligned,
            Err(CudaError::InvalidKernelConfig { .. })
        ));

        let k_misaligned =
            specialized_mma_config(GemmShape::new(64, 128, 255), CompiledDims::STATIC_NK);
        assert!(matches!(
            k_misaligned,
            Err(CudaError::InvalidKernelConfig { .. })
        ));
    }

    // 決定性: 同一入力の 2 回呼び出しが同一 `cfg` を返すこと
    // （`enumerate_tile_candidates` の決定性テストと同型の方針。
    // `RenderedMmaKernel` は `PartialEq` を導出していない〈内部ソース
    // 文字列を外部比較材料にしない設計。`kernels_mma.rs` 参照〉ため、
    // 展開元 `cfg`〈`PartialEq` 導出済み〉の一致で決定性を検証する）。
    #[test]
    fn specialized_mma_config_is_deterministic() {
        let shape = sample_shape();
        let (cfg_a, _) = specialized_mma_config(shape, CompiledDims::STATIC_NK).unwrap();
        let (cfg_b, _) = specialized_mma_config(shape, CompiledDims::STATIC_NK).unwrap();
        assert_eq!(cfg_a, cfg_b);
    }
}

/// L1/L2 帯域コストモデル（Phase C-9b・イシュー #527）の GPU 不要
/// ユニットテスト。合成帯域パラメータ（実測不要）でモデル挙動を検証する
/// （実装計画 §4 ステップ 4）。
#[cfg(test)]
mod cost_model_tests {
    use std::num::{NonZeroU32, NonZeroU64};

    use super::cost_model::{
        CostModelParams, FIXED_TILE_SELECTION, MeasuredBandwidth, SM121_MEASURED_BANDWIDTH,
        compare_candidate_costs, estimate_candidate_cost, select_best_tile_candidate,
    };
    use super::*;
    use crate::kernels_mma::MMA_BM;

    /// 実測不要な代表的 SMEM 予算（`tile_candidate_tests::FULL_BUDGET_BYTES`
    /// と同じ根拠。全候補が SMEM 制約で棄却されないよう十分大きい値）。
    const FULL_BUDGET_BYTES: u64 = u64::MAX;

    /// L1/L2 双方に十分な帯域を与える合成パラメータ（律速側の切り替わり
    /// を意図的に起こさないケースで使う既定値）。SM 数は sm_121 に近い
    /// 桁数の合成値だが実測値ではない（テスト専用の合成入力）。
    fn balanced_params() -> CostModelParams {
        CostModelParams {
            num_sms: NonZeroU32::new(64).expect("64 is non-zero"),
            bandwidth: MeasuredBandwidth {
                l1_bytes_per_cycle_per_sm: NonZeroU64::new(128).expect("128 is non-zero"),
                l2_bytes_per_cycle_device: NonZeroU64::new(4096).expect("4096 is non-zero"),
            },
        }
    }

    /// (a) 単調性: 同一形状・同一 SMEM 予算下では、より大きなタイル
    /// （num_blocks が小さい構成）ほどトラフィックが減り、コストモデルが
    /// より小さいサイクル数（より優先）を割り当てる。
    #[test]
    fn larger_tile_traffic_yields_lower_or_equal_cost_than_smaller_tile() {
        let shape = GemmShape::new(4096, 4096, 4096);
        let params = balanced_params();
        let bpe = bytes_per_element_for(DType::F16);

        let candidates = enumerate_tile_candidates(shape, DType::F16, FULL_BUDGET_BYTES);
        assert!(
            candidates.len() >= 2,
            "must have multiple candidates to compare"
        );

        // block_m*block_n が最大の候補（最も少ない num_blocks）と最小の
        // 候補（最も多い num_blocks）を比較する。
        let largest = candidates
            .iter()
            .max_by_key(|c| c.block_m().get() as u64 * c.block_n().get() as u64)
            .expect("candidates is non-empty");
        let smallest = candidates
            .iter()
            .min_by_key(|c| c.block_m().get() as u64 * c.block_n().get() as u64)
            .expect("candidates is non-empty");

        let largest_cost = estimate_candidate_cost(shape, largest, &params, bpe)
            .expect("cost estimation must not overflow for this shape/candidate");
        let smallest_cost = estimate_candidate_cost(shape, smallest, &params, bpe)
            .expect("cost estimation must not overflow for this shape/candidate");

        // largest の推定サイクル数が smallest 以下であることを実際の
        // CandidateCost 比較で検証する（`compare_candidate_costs` は
        // `pub(crate)` としてテストへ公開。#527 レビュー指摘: 合成コスト
        // を破棄して select_best_tile_candidate の結果だけを見るのでは、
        // ペナルティ補正を消してもテストが検出できない）。
        assert_eq!(
            compare_candidate_costs(&largest_cost, &smallest_cost)
                .expect("comparison must not overflow"),
            std::cmp::Ordering::Less,
            "the tile with fewer num_blocks (less total traffic) must have a strictly lower cost"
        );

        // 選定関数自体も一貫して largest を選ぶことを確認する。
        let best = select_best_tile_candidate(shape, &[*largest, *smallest], &params, bpe)
            .expect("comparison must not overflow");
        assert_eq!(
            best,
            Some(*largest),
            "the tile with fewer num_blocks (less total traffic) must be preferred"
        );
    }

    /// (b) L1/L2 律速側の切り替わり: 片側の帯域を極端に絞ると、その側の
    /// `max()` が支配し、L2 トラフィック（`block_m + block_n` に比例）と
    /// L1 トラフィック（`block_m*warps_n + block_n*warps_m` に比例。
    /// `warps_n = block_n/16`・`warps_m = block_m/32` を代入すると
    /// `block_m*block_n` の積に比例）とで有利な候補の順位が入れ替わる
    /// ことを、実際に順位が入れ替わる 1 組の候補ペアで直接検証する
    /// （#527 レビュー指摘: `select_best_tile_candidate` の結果を
    /// `contains` するだけの smoke test では律速切り替え自体を検証
    /// できない）。
    #[test]
    fn cost_model_switches_dominance_between_l1_and_l2_bandwidth() {
        let shape = GemmShape::new(4096, 4096, 4096);
        let bpe = bytes_per_element_for(DType::F16);
        let candidates = enumerate_tile_candidates(shape, DType::F16, FULL_BUDGET_BYTES);

        // 候補 A: (block_m=128, block_n=16) → タイル面積=2048（小・L1
        // トラフィックはタイル面積に比例）・L2 和=144（大・L2 トラフィック
        // は block_m+block_n に比例）。
        // 候補 B: (block_m=64,  block_n=48) → タイル面積=3072（大）・
        // L2 和=112（小）。shape.n=4096 は block_n=48 で割り切れない
        // （ceil_div により余剰ブロックが生じ、B の面積優位はさらに
        // 相殺される）。
        // 面積の大小関係（A<B）と L2 和の大小関係（A>B）が逆転しているため、
        // 律速側の切り替えで最良候補が入れ替わるはずの組。
        let candidate_a = *candidates
            .iter()
            .find(|c| c.block_m().get() == 128 && c.block_n().get() == 16)
            .expect("(128, 16) candidate must exist for this shape/budget");
        let candidate_b = *candidates
            .iter()
            .find(|c| c.block_m().get() == 64 && c.block_n().get() == 48)
            .expect("(64, 48) candidate must exist for this shape/budget");

        // L2 帯域を極端に絞る（L2 律速。L2 和が小さい B が有利になるはず）。
        let l2_starved = CostModelParams {
            num_sms: NonZeroU32::new(64).expect("64 is non-zero"),
            bandwidth: MeasuredBandwidth {
                l1_bytes_per_cycle_per_sm: NonZeroU64::new(1_000_000).expect("non-zero"),
                l2_bytes_per_cycle_device: NonZeroU64::new(1).expect("non-zero"),
            },
        };
        // L1 帯域を極端に絞る（L1 律速。L1 積が小さい A が有利になるはず）。
        let l1_starved = CostModelParams {
            num_sms: NonZeroU32::new(64).expect("64 is non-zero"),
            bandwidth: MeasuredBandwidth {
                l1_bytes_per_cycle_per_sm: NonZeroU64::new(1).expect("non-zero"),
                l2_bytes_per_cycle_device: NonZeroU64::new(1_000_000).expect("non-zero"),
            },
        };

        let cost_a_l2 = estimate_candidate_cost(shape, &candidate_a, &l2_starved, bpe)
            .expect("cost estimation must not overflow");
        let cost_b_l2 = estimate_candidate_cost(shape, &candidate_b, &l2_starved, bpe)
            .expect("cost estimation must not overflow");
        assert_eq!(
            compare_candidate_costs(&cost_b_l2, &cost_a_l2).expect("comparison must not overflow"),
            std::cmp::Ordering::Less,
            "under L2-starved bandwidth, candidate B (smaller L2 traffic) must cost less than A"
        );

        let cost_a_l1 = estimate_candidate_cost(shape, &candidate_a, &l1_starved, bpe)
            .expect("cost estimation must not overflow");
        let cost_b_l1 = estimate_candidate_cost(shape, &candidate_b, &l1_starved, bpe)
            .expect("cost estimation must not overflow");
        assert_eq!(
            compare_candidate_costs(&cost_a_l1, &cost_b_l1).expect("comparison must not overflow"),
            std::cmp::Ordering::Less,
            "under L1-starved bandwidth, candidate A (smaller L1 traffic) must cost less than B"
        );
    }

    /// (c) wave 効率ペナルティ: 端数 wave を生む構成（`num_blocks` が
    /// `num_sms` の倍数から外れる）は、同等のトラフィックを持つ整数 wave
    /// 構成より不利になる。`num_sms` を意図的に `num_blocks` と非整除
    /// にすることで検証する。
    #[test]
    fn wave_efficiency_penalizes_non_integer_wave_counts() {
        // block_m=128, block_n=128 の単一候補で num_blocks を固定し、
        // num_sms を変えて wave 効率だけが変化するケースを作る。
        let shape = GemmShape::new(1024, 1024, 4096); // 8x8 = 64 blocks (128x128 tile)
        let bpe = bytes_per_element_for(DType::F16);
        let candidates = enumerate_tile_candidates(shape, DType::F16, FULL_BUDGET_BYTES);
        let candidate = candidates
            .iter()
            .find(|c| c.block_m().get() == 128 && c.block_n().get() == 128)
            .expect("128x128 candidate must exist for this shape/budget");

        // l1_bytes_per_cycle_per_sm はコメントどおり device-wide 換算時に
        // `num_sms` 倍される（`estimate_candidate_cost` L1 帯域計算参照）ため、
        // num_sms を変えると l1_cycles 自体も変化してしまい L1 律速では
        // base_cycles が num_sms 非依存にならない（#527 レビュー指摘: PR #675
        // codex-review P2・Cursor Bugbot 双方が、旧パラメータ
        // （l1=128・l2=4096）は 128x128 タイルで L1 律速となり base_cycles が
        // num_sms に連動するため wave 効率補正項を削除しても検出できないと
        // 指摘）。L2 帯域（`l2_bytes_per_cycle_device`）は device-wide 値を
        // そのまま使い num_sms に依存しないため、L2 律速（l2_cycles >=
        // l1_cycles）になるよう l2 帯域を十分小さく選ぶことで、num_sms を
        // 63/64 間で変えても base_cycles（= l2_cycles）は真に不変になる。
        // 128x128 タイル（warps_m=4, warps_n=8）では l1_traffic =
        // 6 * l2_traffic が成り立つため、L2 律速の条件は
        // `l1_bytes_per_cycle_per_sm * num_sms >= 6 * l2_bytes_per_cycle_device`。
        // num_sms=63（最小値）でも成立するよう l2 帯域を 1024 に設定する
        // （128 * 63 = 8064 >= 6 * 1024 = 6144）。
        let bandwidth = MeasuredBandwidth {
            l1_bytes_per_cycle_per_sm: NonZeroU64::new(128).expect("non-zero"),
            l2_bytes_per_cycle_device: NonZeroU64::new(1024).expect("non-zero"),
        };

        // num_blocks = 64。num_sms = 64 なら wave 効率 100%（1 wave）。
        let perfect_params = CostModelParams {
            num_sms: NonZeroU32::new(64).expect("non-zero"),
            bandwidth,
        };
        // num_sms = 63 なら num_waves = ceil(64/63) = 2、wave 効率
        // 64/(2*63) ≈ 50.8%（端数 wave によるペナルティが発生する）。
        let imperfect_params = CostModelParams {
            num_sms: NonZeroU32::new(63).expect("non-zero"),
            bandwidth,
        };

        let perfect_cost = estimate_candidate_cost(shape, candidate, &perfect_params, bpe)
            .expect("cost estimation must not overflow");
        let imperfect_cost = estimate_candidate_cost(shape, candidate, &imperfect_params, bpe)
            .expect("cost estimation must not overflow");

        // 上記の帯域選定により L2 律速（base_cycles = l2_cycles）が
        // num_sms=63/64 の両方で成立し、l2_cycles は l2_traffic /
        // l2_bytes_per_cycle_device のみで決まり num_sms に依存しないため、
        // トラフィック（base_cycles）は真に同一で wave 効率補正項のみが
        // 異なる。`compare_candidate_costs` で実際の CandidateCost を
        // 比較し、端数 wave を生む構成（num_sms=63）の方が真に高コストで
        // あることを検証する（#527 レビュー指摘: 補正式を丸ごと削除しても
        // 検出できない自己完結アサーションは wave 効率補正自体を検証
        // しない）。
        assert_eq!(
            compare_candidate_costs(&imperfect_cost, &perfect_cost)
                .expect("comparison must not overflow"),
            std::cmp::Ordering::Greater,
            "num_sms=63 (imperfect wave packing, num_waves=2) must cost strictly more than \
             num_sms=64 (perfect wave packing, num_waves=1) for the same traffic"
        );
    }

    /// (d) 決定性: 同一入力を 2 回評価しても同一の選定結果になる。
    #[test]
    fn select_best_tile_candidate_is_deterministic() {
        let shape = GemmShape::new(4096, 4096, 4096);
        let bpe = bytes_per_element_for(DType::F16);
        let candidates = enumerate_tile_candidates(shape, DType::F16, FULL_BUDGET_BYTES);
        let params = balanced_params();

        let first = select_best_tile_candidate(shape, &candidates, &params, bpe)
            .expect("comparison must not overflow");
        let second = select_best_tile_candidate(shape, &candidates, &params, bpe)
            .expect("comparison must not overflow");
        assert_eq!(
            first, second,
            "identical inputs must yield identical selection"
        );
    }

    /// (e) オーバーフロー安全性: 巨大形状・巨大帯域を与えても `panic!`
    /// せず `Err`（もしくは正常な `Ok`）のいずれかで応答する。
    #[test]
    fn estimate_candidate_cost_does_not_panic_on_huge_inputs() {
        let shape = GemmShape::new(u32::MAX, u32::MAX, u32::MAX);
        let bpe = bytes_per_element_for(DType::F16);
        let candidates = enumerate_tile_candidates(shape, DType::F16, FULL_BUDGET_BYTES);

        let extreme_params = CostModelParams {
            num_sms: NonZeroU32::new(1).expect("non-zero"),
            bandwidth: MeasuredBandwidth {
                l1_bytes_per_cycle_per_sm: NonZeroU64::new(1).expect("non-zero"),
                l2_bytes_per_cycle_device: NonZeroU64::new(1).expect("non-zero"),
            },
        };

        for candidate in &candidates {
            // panic せず Result で応答することのみを検証する（Err/Ok
            // いずれも許容。巨大形状は u128 演算でも桁あふれしうる）。
            let _ = estimate_candidate_cost(shape, candidate, &extreme_params, bpe);
        }

        // select_best_tile_candidate 自体も panic しないことを検証する。
        let _ = select_best_tile_candidate(shape, &candidates, &extreme_params, bpe);
    }

    /// フォールバック検証: `measured = None`（既定の
    /// `SM121_MEASURED_BANDWIDTH`）→ 固定選定テーブル（64/128/32・
    /// stages 3）が返り、選定根拠が `FixedTable` であること。
    #[test]
    fn select_tile_config_falls_back_to_fixed_table_when_bandwidth_unmeasured() {
        assert_eq!(
            SM121_MEASURED_BANDWIDTH, None,
            "sm_121 実測帯域が未実測の間は select_tile_config が常に固定選定 \
             テーブルへフォールバックする契約のテスト前提（A-2・#482 未実測）"
        );

        let shape = GemmShape::new(4096, 4096, 4096);
        let num_sms = NonZeroU32::new(64).expect("64 is non-zero");

        let selection = select_tile_config(
            shape,
            DType::F16,
            FULL_BUDGET_BYTES,
            num_sms,
            SM121_MEASURED_BANDWIDTH,
        )
        .expect("aligned shape/F16/full budget satisfies FIXED_TILE_SELECTION constraints");

        assert_eq!(selection.basis(), TileSelectionBasis::FixedTable);
        let fixed = FIXED_TILE_SELECTION;
        assert_eq!(selection.candidate(), fixed);
    }

    /// フォールバック検証（かつ codex-review #675 P1 指摘の回帰防止）:
    /// 候補ゼロ件（不整列形状）の場合、`measured` を `Some` にしても
    /// コストモデルが評価対象を持たないため固定テーブルへ倒れようと
    /// するが、固定テーブル自体も同じ不整列形状の下では
    /// `validate_mma_alignment` を満たさないため、無検証の
    /// `TileSelection` を返さず `Err(CudaError::InvalidKernelConfig)`
    /// で選定不能を明示する。
    #[test]
    fn select_tile_config_rejects_when_shape_violates_alignment_even_for_fixed_table() {
        // shape.k % 8 != 0 のため enumerate_tile_candidates が規則 6 で
        // 空 Vec を返す形状（tile_candidate_tests の同種フィクスチャと
        // 同じ根拠）。同じ不整列性が FIXED_TILE_SELECTION 側の
        // validate_mma_alignment 検証も失敗させる。
        let unaligned_shape = GemmShape::new(4096, 9, 17);
        let num_sms = NonZeroU32::new(64).expect("64 is non-zero");
        let measured = Some(MeasuredBandwidth {
            l1_bytes_per_cycle_per_sm: NonZeroU64::new(128).expect("non-zero"),
            l2_bytes_per_cycle_device: NonZeroU64::new(4096).expect("non-zero"),
        });

        let err = select_tile_config(
            unaligned_shape,
            DType::F16,
            FULL_BUDGET_BYTES,
            num_sms,
            measured,
        )
        .expect_err(
            "unaligned shape must be rejected instead of silently returning \
             FIXED_TILE_SELECTION (codex-review #675 P1)",
        );
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    /// `SM121_MEASURED_BANDWIDTH` が `None` である間は選定入口が
    /// コストモデルを一切評価しないこと（H100 定数流用の不在を機械検査）。
    /// `select_tile_config` に `measured: None` を渡した場合、内部で
    /// `enumerate_tile_candidates`／コストモデルへ到達せず即座に固定
    /// テーブルを返す実装であることを、`smem_budget_bytes` に意図的に
    /// 不正な値（0。全候補が SMEM 予算超過で棄却される値）を渡しても
    /// 固定テーブルの選定結果に一切影響しないことで間接検証する
    /// （コストモデル経路を通っていれば `smem_budget_bytes=0` は
    /// `enumerate_tile_candidates` の結果に影響するはずだが、`None`
    /// 経路はそもそも `enumerate_tile_candidates` を呼ばない）。
    #[test]
    fn none_bandwidth_bypasses_cost_model_evaluation_entirely() {
        let shape = GemmShape::new(4096, 4096, 4096);
        let num_sms = NonZeroU32::new(64).expect("64 is non-zero");

        // smem_budget_bytes = 0 は enumerate_tile_candidates を通せば
        // 全候補棄却（tile_candidate_tests 参照）になる。None 経路は
        // enumerate_tile_candidates／コストモデルを一切呼ばず、直接
        // fixed_table_selection_if_valid で FIXED_TILE_SELECTION の
        // smem_total を smem_budget_bytes=0 に対して検証するため、
        // どちらの経路を通っても最終的に「予算超過で選定不能」という
        // 同じ結論に到達する（codex-review #675 P1 指摘の回帰防止:
        // 検証共有化前は None 経路がこの budget=0 を無視して
        // FIXED_TILE_SELECTION を無検証で返してしまっていた）。
        let err = select_tile_config(shape, DType::F16, 0, num_sms, None).expect_err(
            "smem_budget_bytes=0 must reject FIXED_TILE_SELECTION instead of returning it \
             unchecked (codex-review #675 P1)",
        );
        assert!(matches!(err, CudaError::InvalidKernelConfig { .. }));
    }

    /// codex-review #675 P1 指摘の回帰防止: `select_tile_config` に
    /// `DType::F32` を渡した場合、`measured = None`（固定テーブル経路）
    /// では `FIXED_TILE_SELECTION` の `smem_per_stage`/`smem_total` が
    /// f16 前提で算出されたメタデータであるため、dtype 不一致として
    /// `Err(CudaError::InvalidKernelConfig)` を返し、無検証の
    /// `TileSelection` を成功として返さないこと。
    #[test]
    fn select_tile_config_rejects_f32_dtype_for_fixed_table() {
        let shape = GemmShape::new(4096, 4096, 4096);
        let num_sms = NonZeroU32::new(64).expect("64 is non-zero");

        let err = select_tile_config(shape, DType::F32, FULL_BUDGET_BYTES, num_sms, None)
            .expect_err(
                "DType::F32 must be rejected because FIXED_TILE_SELECTION's smem metadata is \
                 f16-only (codex-review #675 P1)",
            );
        assert!(matches!(err, CudaError::InvalidKernelConfig { .. }));
    }

    /// Cursor Bugbot 指摘の回帰防止（PR #675 review）:
    /// `validate_fixed_tile_selection` は dtype／`n`・`k` 整列／SMEM 予算は
    /// 検査するが `validate_mma_grid_bounds` を呼んでいなかったため、
    /// `m > 64 * 65535`（`MMA_BM` 単位で grid_dim.y が CUDA の 65,535 上限を
    /// 超える形状）に対しても `dtype`／`n`／`k`／SMEM の条件さえ満たせば
    /// `FIXED_TILE_SELECTION` を「成功した `TileSelection`」として返して
    /// しまっていた（実際のカーネル起動は grid dim 上限超過で失敗する）。
    /// `n`/`k` は 8 の倍数の整列要件を保ちつつ `m` だけを上限超過させ、
    /// `Err(CudaError::InvalidShape)` になることを確認する。
    #[test]
    fn select_tile_config_rejects_m_exceeding_grid_y_limit_for_fixed_table() {
        // gemm_mma.rs::validate_mma_grid_bounds_rejects_m_exceeding_grid_y_limit
        // と同じ根拠・同じ境界値（65_535 * MMA_BM + 1）。
        let oversized_shape = GemmShape::new(65_535 * MMA_BM + 1, 4096, 4096);
        let num_sms = NonZeroU32::new(64).expect("64 is non-zero");

        let err = select_tile_config(
            oversized_shape,
            DType::F16,
            FULL_BUDGET_BYTES,
            num_sms,
            SM121_MEASURED_BANDWIDTH,
        )
        .expect_err(
            "m exceeding 64 * 65535 must be rejected instead of silently returning \
             FIXED_TILE_SELECTION with an unlaunchable grid_dim.y (Cursor Bugbot, PR #675)",
        );
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }
}
