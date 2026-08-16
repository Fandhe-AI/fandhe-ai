//! GEMM 自動経路選択の入口（TASK-11.2b・#68）。
//!
//! `tensor_core::dispatch::select_gemm_kernel`（規則の純関数実装。
//! `docs/dispatch-rules-design.md` §5 の決定表）を `backend-cuda` の
//! 既存 GEMM 実装（`gemm.rs::CudaGemm`〈naive／tiled〉・
//! `gemm_wmma.rs::CudaWmmaGemm`〈WMMA f16 Tensor Core〉）へ結線する。
//! `BackendOps` trait 実装（TASK-1.9c・#46）はこの `CudaGemmAuto` を
//! `BackendOps::gemm` から呼ぶだけの構成にできる（実装計画 §3.2
//! 「#46 との境界」）。
//!
//! `DeviceCaps` は [`CudaGemmAuto::new`] 実行時（デバイス初期化）に 1 回
//! だけ `CudaDevice::compute_capability()` から構築し、以降の
//! `run_f32`／`run_f16` 呼び出しでは FFI 照会を繰り返さない
//! （`docs/dispatch-rules-design.md` §2.1「判定タイミング」）。
//!
//! フォールバック連鎖は `MatrixUnit → Tiled → Naive`（§5.2）。
//! `CudaWmmaGemm::new` が cc ゲート非対応・NVRTC コンパイル失敗のいずれか
//! で `Err` を返した場合は WMMA を候補から外し、以降 `run_f16` は常に
//! tiled 経路を使う（fail-safe。`panic!`／`unwrap()` を使わない）。

use std::num::NonZeroU32;

use cudarc::driver::sys::CUdevice_attribute;
use half::f16;
use tensor_core::dispatch::{DType, DeviceCaps, GemmShape, KernelKind, select_gemm_kernel};

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm::CudaGemm;
use crate::gemm_mma::validate_mma_alignment;
use crate::gemm_wmma::CudaWmmaGemm;
use crate::kernels_mma::{
    MMA_K, MMA_SHARED_MEM_BYTES, MMA_STATIC_SMEM_LIMIT_BYTES, MMA_WARP_M, MMA_WARP_N,
};
use crate::nvrtc::derive_pipeline_stages;

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
/// に取得し、[`STATIC_SMEM_BUDGET_CAP_BYTES`] でクランプしたうえで
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
fn read_clamped_smem_budget_bytes(device: &CudaDevice) -> Result<u64, CudaError> {
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
/// `DType` は非網羅的でない列挙（`tensor_core::dispatch`）であるため
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
/// | `SMALL_M_THRESHOLD`（64）の場合のみ [`MMA_WARP_M`]（32）を追加する。
/// | 16 は本カーネル族の warp タイル（[`MMA_WARP_M`] = 32）が構造的に
/// | 分解不能なため候補にしない |
/// | block_n | [`MMA_WARP_N`]（16）の倍数刻みで `16..=BLOCK_N_MAX_CANDIDATE`
/// | （256） |
/// | block_k | [`MMA_K`]（16）の倍数 `{16, 32, 64}`
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
///    常に [`MMA_WARP_N`]（16）の倍数、block_k は規則 1（続き）により
///    常に [`MMA_K`]（16）の倍数と確定しているため、`bytes_per_element`
///    が 1 以上である限り両積は常に 16 の倍数になり、現行候補空間では
///    この規則も構造的に発火しない〈dead〉。#524 レビュー指摘。将来
///    warp タイル寸法（[`MMA_WARP_N`]／[`MMA_K`]）が 16 未満に変更される
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
/// 行わない（[実機不要] タスク）。
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
/// [`STATIC_SMEM_BUDGET_CAP_BYTES`] でクランプしたうえで純関数
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

/// naive／tiled／WMMA の全 GEMM カーネルを保持し、`select_gemm_kernel`
/// の判定結果に従って呼び分ける自動経路選択の入口。
///
/// `wmma` は `Option` とし、cc ゲート非対応・NVRTC コンパイル失敗時は
/// `None` のまま保持する（fail-safe。上記モジュールコメント参照）。
/// `caps` は `wmma` の有無とは独立に「cc ゲートを満たすか」だけを表す
/// （`select_gemm_kernel` の判定材料は cc のみであり、コンパイル成否は
/// 別軸のフォールバックとして扱う。`run_f16` 内のコメント参照）。
pub struct CudaGemmAuto {
    gemm: CudaGemm,
    wmma: Option<CudaWmmaGemm>,
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
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let gemm = CudaGemm::new(device)?;
        let wmma = CudaWmmaGemm::new(device).ok();
        let caps = DeviceCaps::cuda(device.compute_capability());
        Ok(Self { gemm, wmma, caps })
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
    /// `select_gemm_kernel` が `MatrixUnit` を返しても `wmma` が `None`
    /// （構築失敗。[`Self::new`] 参照）なら tiled へフォールバックする
    /// （§5.2 のフォールバック連鎖 `MatrixUnit → Tiled → Naive` の
    /// 中間段。`self.gemm`〈naive／tiled〉は `new` 成功時点で必ず存在
    /// するため、tiled 自体が失敗するケースはカーネル起動時エラー
    /// （`CudaError`）としてそのまま呼び出し元へ返る）。
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
            KernelKind::MatrixUnit => match &self.wmma {
                Some(wmma) => wmma.run_f16(a, b, m, n, k),
                // cc ゲートは満たすが WMMA 構築が失敗していた場合の
                // fail-safe フォールバック（NVRTC コンパイル失敗等）。
                None => self.gemm.run_tiled_f16(a, b, m, n, k),
            },
            KernelKind::Tiled => self.gemm.run_tiled_f16(a, b, m, n, k),
            KernelKind::Naive => self.gemm.run_naive_f16(a, b, m, n, k),
        }
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
