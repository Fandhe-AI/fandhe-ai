//! Metal デバイス・コマンドキューの基盤（TASK-1.8a・#38）。
//!
//! `tensor-core` の演算グラフノードを MSL カーネルへディスパッチする前段
//! として、システムデフォルトの Metal デバイス取得とコマンドキュー生成を
//! 一箇所にまとめる。MSL ライブラリのコンパイル・パイプライン構築・
//! ディスパッチ経路は本イシューのスコープ外（TASK-1.8b・#39 で
//! `MetalContext` を土台にして追加する）。
//!
//! **移植元**: `docs/spec/03-poc/poc-v2-4-metal-gemm/code/rust/src/metal_gemm.rs`
//! の `MetalGemm::new`（デバイス・キュー取得部分）。PoC は `Option` 返しの
//! `expect` 呼び出しだったが、本実装は [`MetalError`] を返す `Result` 化
//! （coding-rust.md「本番経路で unwrap/expect を使わない」）。

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_tensor_core::DispatchFailureCell;
use fandhe_ai_tensor_core::dispatch::DeviceCaps;
use objc2::Message;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily,
};

use crate::batch_state::{self, BatchMeta};
use crate::buffer::MtlBuffer;
use crate::device::MetalOccupancyInfo;
use crate::error::MetalError;
use crate::pool_pending;
use crate::tile::{self, OccupancyParams};

pub(crate) type MtlDevice = ProtocolObject<dyn MTLDevice>;
pub(crate) type MtlQueue = ProtocolObject<dyn MTLCommandQueue>;
type MtlCommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;
type MtlComputeEncoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// コマンドバッファ共有バッチ 1 個分の状態（イシュー #1017・
/// `docs/backend-metal-command-batching-design.md`）。
///
/// `encoder` は [`MetalContext::encode`] の複数回の呼び出しにまたがって
/// 使い回す（`MTLComputeCommandEncoder` は `endEncoding()` を呼ぶまで
/// 複数回の `dispatchThreadgroups_threadsPerThreadgroup` を受け付ける
/// ため、1 dispatch = 1 エンコーダの旧 `dispatch_sync` と異なり
/// バッチ内では 1 エンコーダを使い回してよい）。`in_flight` は dispatch
/// が参照した `MTLBuffer` を [`MetalContext::synchronize`] 完了まで
/// 生存させる保持列（呼び出し元が関数スコープを抜けても GPU 完了前に
/// drop されないようにする。設計文書 §3.4）。`tokens` は `encode` と
/// 同一ロック区間で登録された [`DispatchFailureCell`]（`tensor-core`）
/// で、`synchronize` が実行時エラーを検出した際にまとめて `set` する
/// （設計文書 §3.7 (2)。「encode 後に別 API で登録」方式は、その間に
/// 別スレッドの `synchronize` が割り込む競合があるため採らない）。
struct Batch {
    cmd_buf: Retained<MtlCommandBuffer>,
    encoder: Option<Retained<MtlComputeEncoder>>,
    meta: BatchMeta,
    in_flight: Vec<Retained<MtlBuffer>>,
    tokens: Vec<DispatchFailureCell>,
}

// SAFETY: `MTLCommandBuffer`／`MTLComputeCommandEncoder`（objc2-metal
// 0.3.2）は `Send`／`Sync` を supertrait に持たない（Apple の Metal
// スレッディング契約が「同時アクセス不可」のみを課しスレッド親和性を
// 持たないことに対応し、objc2-metal は直列化を静的に証明できないため
// 保守的に付けていない）。`Batch` は `MetalContext::batch`
// （`Mutex<BatchSlots>`）を介してのみ到達可能であり、`encode`／
// `flush`／`synchronize`／`Drop` の全経路が `Mutex` のロック下でのみ
// `Batch` の中身へ触れる（`Mutex` によるアクセスの完全直列化）。この
// 直列化により、`Batch` が実際に複数スレッド間を移動しても Metal 側の
// 「同時アクセス不可」契約に違反しない。よって `Send` の付与は安全
// （`MetalContext` 自体は `context_cache.rs::assert_send_sync` が
// `Send + Sync` を要求しており、`Mutex<BatchSlots>: Sync` の成立には
// `Batch: Send` が必要）。イシュー #1017 実装計画 §2.1・
// `docs/backend-metal-command-batching-design.md` §0 参照。
unsafe impl Send for Batch {}

/// [`MetalContext::batch`] が保持する開いているバッチ・commit 済みで
/// 完了待ちのバッチ列。`Mutex` を介してのみアクセスされる（`Batch` の
/// SAFETY コメント参照）ため、フィールド自体に追加の同期は不要。
#[derive(Default)]
struct BatchSlots {
    /// まだ `commit()` していない、encode 中のバッチ（高々 1 個）。
    open: Option<Batch>,
    /// `commit()` 済みで `waitUntilCompleted()` 未実施のバッチ列
    /// （内部ヘルパ `flush_locked`（`encode` の自動 flush・`synchronize`
    /// から呼ばれる）が `open` から移す。複数回の flush が間に挟まると
    /// 2 個以上になりうる）。
    committed: Vec<Batch>,
    /// 保留中のプール返却列（イシュー #1021・設計文書 §3.3「Metal」）。
    /// `PooledMetalHandle::Drop`（`crate::pool`）が `open`／`committed`
    /// の有無を検査した**同一ロック区間内**でここへ push する（TOCTOU
    /// 回避。§3.5「Metal `pending_pool_returns` の排他制御」）。`Send`
    /// 境界は [`crate::pool::RawMetalBuffer`] が `Send`（`Batch` の
    /// SAFETY コメントと同種の根拠。objc2-metal 0.3.2 の `MTLBuffer`
    /// protocol が `Send + Sync` を supertrait に持つ）であるため
    /// `PendingReturns<RawMetalBuffer>` も自動的に `Send` になる。
    pending_pool_returns: pool_pending::PendingReturns<crate::pool::RawMetalBuffer>,
}

/// [`MetalContext::synchronize_with_gpu_timestamps`]（診断専用。イシュー
/// #1276）が返す、完了したバッチ 1 個分の GPU タイムスタンプ。
///
/// `MTLCommandBuffer::GPUStartTime`/`GPUEndTime` は「ホスト時計上の
/// 秒数」（`CFTimeInterval` = `f64`）で、未開始・完了通知未受領時は
/// `0.0` を返す契約（objc2-metal 0.3.2 生成コードのコメント）。診断側
/// はこの `0.0` を「値なし」として扱いたいため、生値をそのまま持たず
/// [`Self::from_raw`] で `None` へ正規化してから保持する（`0.0` を
/// 「ホスト時計の起点」と誤読して区間計算に使う事故を型で防ぐ）。
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct BatchGpuTimestamps {
    gpu_start_secs: Option<f64>,
    gpu_end_secs: Option<f64>,
    /// バッチに記録されていたディスパッチのラベル列（`BatchMeta::
    /// labels`）。診断テスト側が「想定どおり 1 個の GEMM ディスパッチ
    /// だけが載っていたか（singleton `cached_context()` への他
    /// ディスパッチ混入がないか）」を確認するために保持する。
    labels: Vec<&'static str>,
}

#[cfg(test)]
impl BatchGpuTimestamps {
    /// `0.0`（未開始／完了通知未受領）を `None` へ変換して保持する。
    fn from_raw(gpu_start_secs: f64, gpu_end_secs: f64, meta: &BatchMeta) -> Self {
        Self {
            gpu_start_secs: (gpu_start_secs != 0.0).then_some(gpu_start_secs),
            gpu_end_secs: (gpu_end_secs != 0.0).then_some(gpu_end_secs),
            labels: meta.labels().to_vec(),
        }
    }

    /// このバッチに記録されていたディスパッチのラベル列。
    pub(crate) fn labels(&self) -> &[&'static str] {
        &self.labels
    }

    /// `GPUEndTime - GPUStartTime`（秒）。両者が取得できた場合のみ
    /// `Some` を返す（`None` の伝播は診断テスト側の assert で検出する
    /// 設計。ファイル冒頭「GPU タイムスタンプ変種」参照）。
    pub(crate) fn kernel_gpu_secs(&self) -> Option<f64> {
        let start = self.gpu_start_secs?;
        let end = self.gpu_end_secs?;
        Some(end - start)
    }
}

/// Metal デバイスとコマンドキューを保持するハンドル。
///
/// [`crate::buffer::MetalBuffer`] の確保・[`MetalContext::dispatch_sync`]
/// の同期実行はいずれも本構造体が保持する `device` / `queue` を介して
/// 行う。TASK-1.8b（#39）以降のパイプライン構築・エンコーダ結線は
/// 本構造体を土台にして追加される想定であり、公開フィールドは持たせず
/// アクセサ（[`MetalContext::device`] / [`MetalContext::queue`]）経由に
/// 限定する。
///
/// TASK-11.2b（#68）で `caps`（[`DeviceCaps`]）を追加した。`MTLDevice::
/// supportsFamily(MTLGPUFamily::Apple7)` の判定結果を `new` 時に 1 回
/// キャッシュし、[`crate::gemm::MetalGemm::dispatch_backend_auto`] から
/// `fandhe_ai_tensor_core::dispatch::select_gemm_kernel` へそのまま渡せるようにする
/// （`docs/dispatch-rules-design.md` §2.1「判定タイミング: デバイス初期化
/// 時に 1 回」。ディスパッチ呼び出しごとに `supportsFamily` を再照会
/// しない）。
pub struct MetalContext {
    device: Retained<MtlDevice>,
    queue: Retained<MtlQueue>,
    caps: DeviceCaps,
    occupancy_params: Option<OccupancyParams>,
    /// SoC ブランド文字列（`crate::device::probe_soc_brand_string` の実測値。
    /// 例: `"Apple M4 Max"`）。`new` 時に 1 回だけ取得しキャッシュする
    /// （`occupancy_params`・`caps` と同じ判断）。[`Self::
    /// verified_m4_max_gpu_core_count`] が `occupancy_params` の GPU コア数
    /// と組み合わせ、イシュー #1039 の M4 Max 実測厳密一致テーブルの適用
    /// 可否判定に使う（P1・codex-review 指摘・PR #1108 レビュー: GPU コア数
    /// だけでは機種〈例: M3 Max との 40 コア構成の混同〉を一意に識別
    /// できないため）。
    soc_brand: Option<String>,
    /// コマンドバッファ共有バッチの状態（イシュー #1017）。
    /// [`MetalContext::encode`]／[`MetalContext::synchronize`]（内部
    /// ヘルパ `flush_locked` を含む）がこの `Mutex` を介してのみ
    /// アクセスする（[`Batch`] の SAFETY コメント参照）。
    batch: Mutex<BatchSlots>,
    /// イシュー #1099 診断用カウンタ（本番経路の判断には使わず、
    /// テスト・ベンチからバッチング効果を実測するためだけに存在する。
    /// `Relaxed` 加算のみで、ホットパスへの影響は 1 命令程度に留める。
    /// `Mutex<BatchSlots>` の外に置くのは、カウンタ読み出しがロックを
    /// 取らずに行えるようにするため（診断専用であり厳密な同時性は
    /// 要求しない）。[`Self::diagnostic_batch_counters`] 参照。
    diag_encode_calls: AtomicUsize,
    diag_command_buffers: AtomicUsize,
    diag_wait_until_completed: AtomicUsize,
}

/// [`MetalContext::diagnostic_batch_counters`] が返す、ある時点での
/// バッチング診断カウンタのスナップショット（イシュー #1099）。
///
/// テスト・診断専用であり、`facade` の公開 API 面（`docs/compat-api-scope.md`
/// §0「`facade` が唯一のサポートされる公開 API 面」）には含めない。
/// `lib.rs::__diagnostic_batch_counters_snapshot`（`#[doc(hidden)]`）
/// 経由で `backend-metal`／`facade` の実機ベンチ・回帰テストから
/// カウンタ差分を読む用途にのみ使う。
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BatchCountersSnapshot {
    /// [`MetalContext::encode`] の呼び出し回数（dispatch 1 回ごとに 1）。
    pub encode_calls: usize,
    /// 新規コマンドバッファ生成回数（`queue.commandBuffer()` の呼び出し
    /// 回数。バッチが開くたびに 1 回）。
    pub command_buffers: usize,
    /// `waitUntilCompleted()` の呼び出し回数（[`MetalContext::synchronize`]
    /// が完了待ちしたバッチの総数）。
    pub wait_until_completed: usize,
}

impl MetalContext {
    /// システムデフォルトの Metal デバイスを取得し、コマンドキューを
    /// 生成する。デバイスが見つからない・キュー生成に失敗した場合は
    /// [`MetalError`] を返す（PoC-v2-4 の `Option` 返しを型付きエラーへ
    /// 置き換え）。
    ///
    /// `MTLDevice::supportsFamily(MTLGPUFamily::Apple7)`（`simdgroup_matrix`
    /// 対応可否の判定材料。`docs/dispatch-rules-design.md` §2 表）もここで
    /// 1 回だけ評価し [`DeviceCaps`] へキャッシュする（[`Self::caps`]）。
    ///
    /// [`crate::tile::select_with_occupancy`]（イシュー #542）が使う
    /// occupancy 実機値（[`MetalOccupancyInfo::probe`]。GPU コア数は IOKit
    /// FFI・threadgroup memory 上限は `MTLDevice`）もここで 1 回だけ取得し
    /// [`OccupancyParams`] へ写像してキャッシュする（[`Self::occupancy_params`]）。
    /// ディスパッチ経路のホットパスに FFI を持ち込まないための設計（`caps`
    /// と同じ判断）。GPU コア数が取得不能（`None`）な場合は
    /// `occupancy_params` 自体を `None` にし、`select_with_occupancy` の
    /// fail-safe フォールバック（`params: None`）へ委ねる。
    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceUnavailable)?;
        let queue = device
            .newCommandQueue()
            .ok_or(MetalError::CommandQueueCreation)?;
        // `supportsFamily` は objc2-metal が safe メソッドとして提供する
        // （`MTLDevice.rs` 生成コードに `unsafe` プレフィックスなし。
        // `device.rs::probe_all` が同様に他の `MTLDevice` メソッドを
        // unsafe ブロックなしで呼んでいるのと同じ扱い）。判定失敗
        // （このメソッド自体は bool を返すため失敗はしないが、将来
        // API が変わり得ない前提を置かない）時は非対応（Apple7 未満）
        // 扱いに倒す fail-safe とする（§2.2）。
        let apple7_supported = device.supportsFamily(MTLGPUFamily::Apple7);
        let caps = DeviceCaps::metal(apple7_supported);

        let occupancy_info = MetalOccupancyInfo::probe(&device);
        let occupancy_params =
            occupancy_info
                .gpu_core_count
                .map(|gpu_core_count| OccupancyParams {
                    gpu_core_count,
                    max_threadgroup_memory_bytes: occupancy_info.max_threadgroup_memory_bytes,
                });

        let soc_brand = crate::device::probe_soc_brand_string();

        Ok(Self {
            device,
            queue,
            caps,
            occupancy_params,
            soc_brand,
            batch: Mutex::new(BatchSlots::default()),
            diag_encode_calls: AtomicUsize::new(0),
            diag_command_buffers: AtomicUsize::new(0),
            diag_wait_until_completed: AtomicUsize::new(0),
        })
    }

    /// イシュー #1099 診断用: 現時点までのバッチングカウンタ
    /// スナップショットを返す（`lib.rs::__diagnostic_batch_counters_snapshot`
    /// 経由でクレート外のテスト・ベンチから読む唯一の経路）。
    /// `Relaxed` で読むため、他スレッドの同時呼び出しと厳密には順序
    /// 保証しない（診断専用であり本番の正しさに影響しない）。
    #[doc(hidden)]
    pub fn diagnostic_batch_counters(&self) -> BatchCountersSnapshot {
        BatchCountersSnapshot {
            encode_calls: self.diag_encode_calls.load(Ordering::Relaxed),
            command_buffers: self.diag_command_buffers.load(Ordering::Relaxed),
            wait_until_completed: self.diag_wait_until_completed.load(Ordering::Relaxed),
        }
    }

    /// [`crate::buffer::MetalBuffer`] の確保・パイプライン構築
    /// （TASK-1.8b・#39 以降）から参照される Metal デバイスハンドル。
    pub fn device(&self) -> &MtlDevice {
        &self.device
    }

    /// `new` 時にキャッシュした occupancy 判定用実機値（イシュー #542）。
    /// [`crate::tile::select_with_occupancy`] が受け取る形へ写像済みだが、
    /// 本番ディスパッチ入口 [`crate::gemm::MetalGemm::dispatch_auto`] は
    /// 実機実測（M4 Max `select()` 比の非劣化確認）が完了するまで
    /// `tile::select` を使い続ける契約であり、本値を渡さない
    /// （[`crate::gemm::MetalGemm::dispatch_auto`] ドキュメンテーション
    /// コメント参照。codex-review P2・PR #684）。現状は
    /// `examples/gemm_bench.rs` の比較経路からのみ利用される。GPU コア数が
    /// 取得不能だった場合は `None`（`select_with_occupancy` 側の
    /// fail-safe フォールバックで形状のみの選択へ倒れる）。
    pub fn occupancy_params(&self) -> Option<OccupancyParams> {
        self.occupancy_params
    }

    /// イシュー #1039 の M4 Max 実測厳密一致テーブル（`crate::tile` の
    /// `exact_match_cfg`）を適用してよいかどうかを、`new` 時にキャッシュ
    /// した GPU コア数と SoC ブランド文字列の両方から検証する
    /// （`crate::tile::verify_m4_max` への委譲。P1・codex-review 再指摘・
    /// PR #1108 レビュー: GPU コア数だけでは機種〈M3 Max との 40 コア構成
    /// の混同〉を一意に識別できないため）。戻り値は [`crate::tile::
    /// VerifiedM4MaxGpuCoreCount`]（`verify_m4_max` からのみ構築可能な
    /// opaque 型）であり、`crate::gemm::MetalGemm::dispatch_auto` 等の
    /// 本番ディスパッチ入口は、この戻り値をそのまま `tile::
    /// select_for_device` の `gpu_core_count` 引数へ渡す（未検証の生の
    /// GPU コア数を渡してブランド照合を迂回することは型上できない）。
    pub fn verified_m4_max_gpu_core_count(&self) -> Option<tile::VerifiedM4MaxGpuCoreCount> {
        tile::verify_m4_max(
            self.occupancy_params.map(|p| p.gpu_core_count),
            self.soc_brand.as_deref(),
        )
    }

    /// コマンドバッファ生成に使うコマンドキュー
    /// （TASK-1.8b・#39 のディスパッチ経路から参照される）。
    pub fn queue(&self) -> &MtlQueue {
        &self.queue
    }

    /// `new` 時にキャッシュした GPU family 判定結果
    /// （[`crate::gemm::MetalGemm::dispatch_backend_auto`] が
    /// `select_gemm_kernel` へ渡す `DeviceCaps`。TASK-11.2b・#68）。
    pub fn caps(&self) -> DeviceCaps {
        self.caps
    }

    /// バッチの `Mutex<BatchSlots>` をロックする共通ヘルパ。poison を
    /// panic させず [`MetalError::BatchStateUnavailable`] へ変換する
    /// （`context_cache.rs::on_poison` と同じ「本番経路で unwrap/expect
    /// を使わない」判断。`.claude/rules/coding-rust.md`）。
    fn lock_batch(
        &self,
        op: &'static str,
    ) -> Result<std::sync::MutexGuard<'_, BatchSlots>, MetalError> {
        self.batch
            .lock()
            .map_err(|_| MetalError::BatchStateUnavailable {
                detail: format!("{op}: batch mutex poisoned"),
            })
    }

    /// [`crate::pool::PooledMetalHandle::drop`] から呼ばれる、プール
    /// バッファ返却の GPU 完了待ち判定（イシュー #1021・設計文書 §3.3
    /// 「Metal」・§3.5「Metal `pending_pool_returns` の排他制御」）。
    ///
    /// `BatchSlots` の同一ロック区間内で「`open`／`committed` バッチの
    /// 有無を検査する」処理と「`pending_pool_returns` へ追加する（また
    /// はそのまま即時返却する）」処理を行うことで、検査後・追加前に
    /// 別スレッドが `synchronize()` を呼んで空の `pending_pool_returns`
    /// を drain してしまう TOCTOU 競合を構造的に防ぐ（§3.5）。
    ///
    /// 判定ロジック自体（in-flight なら push＋`record_pending_return`、
    /// そうでなければ `Some` を返す）は `pool_pending::PendingReturns::
    /// defer_or_release`（`objc2` 型に触れない純粋ロジック）に委譲する。
    /// 即時返却経路（戻り値が `Some`）は `Mutex<PoolCore<H>>` を要する
    /// `SizeClassPool::put` を呼ぶため、**ロックを解放した後**に
    /// `pool_pending::put_all` へ渡す（§3.5「ロック順序規則」）。
    ///
    /// `lock_batch` が poison を検出した場合（本来到達しない異常系）は、
    /// 返却対象のハンドルをそのまま drop する（プールへ戻せないだけで
    /// メモリ安全性には影響しない。`Drop` から `Result` を返せないための
    /// fail-safe な縮退。`crate::context_cache::on_poison` と同じ
    /// 「panic させない」方針）。
    pub(crate) fn defer_pool_return(
        &self,
        entry: pool_pending::PendingReturn<crate::pool::RawMetalBuffer>,
    ) {
        let Ok(mut slots) = self.lock_batch("defer_pool_return") else {
            return;
        };
        let in_flight = slots.open.is_some() || !slots.committed.is_empty();
        let immediate = slots
            .pending_pool_returns
            .defer_or_release(in_flight, entry);
        drop(slots);
        if let Some(entry) = immediate {
            pool_pending::put_all(vec![entry]);
        }
    }

    /// ディスパッチ内容の記録（バッファ結線・エンコード）だけを行い、
    /// **待たない**（イシュー #1017・`docs/backend-metal-command-
    /// batching-design.md` §3）。同一 [`MetalContext`] に対する連続した
    /// `encode` 呼び出しは、上限（[`batch_state::MAX_DISPATCHES_PER_BATCH`]）
    /// に達するまで同一コマンドバッファ・同一コンピュートエンコーダへ
    /// 積まれる。実際に GPU 完了を待つのは [`Self::synchronize`]（ホスト
    /// 実体化時: `memory.rs::download_inner`／`zero_fill`・`Drop`）のみ。
    ///
    /// `resources` は本 dispatch が参照する `MTLBuffer` 列で、
    /// `synchronize()` が完了するまで [`Batch::in_flight`] へ retain
    /// して生存させる（呼び出し元がバッファを関数スコープで drop しても
    /// GPU 完了前に解放されないようにする。設計文書 §3.4・§2.5）。
    /// `token`（`Some` の場合）は本 dispatch を含むバッチが実行時エラー
    /// になった際に [`fandhe_ai_tensor_core::DispatchFailureCell::set`]
    /// される（`encode` 呼び出しと同一ロック区間で登録するため、登録と
    /// 別スレッドの `synchronize` の間に競合が生じない。設計文書
    /// §3.7 (2)）。
    ///
    /// `commandBuffer()`／`computeCommandEncoder()` は autoreleased な
    /// オブジェクトを返す。Rust バイナリには Cocoa アプリのような周囲の
    /// autorelease pool が存在しないため、`autoreleasepool` で明示的に
    /// 囲まないと繰り返し `encode` のたびに Metal の一時オブジェクトが
    /// プロセス寿命分蓄積する（旧 `dispatch_sync` と同じ理由）。
    pub(crate) fn encode<F>(
        &self,
        label: &'static str,
        resources: &[&MtlBuffer],
        token: Option<&DispatchFailureCell>,
        encode_fn: F,
    ) -> Result<(), MetalError>
    where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    {
        autoreleasepool(|_pool| {
            let mut slots = self.lock_batch("encode")?;

            // イシュー #1099 診断カウンタ: この `encode` 呼び出し自体を
            // 1 カウントする（`batch.meta.record_dispatch` と対応する
            // 呼び出し回数。バッチング効果の実測に使う。本番の正しさ
            // には影響しない `Relaxed` 加算）。
            self.diag_encode_calls.fetch_add(1, Ordering::Relaxed);

            if slots.open.is_none() {
                let cmd_buf = self
                    .queue
                    .commandBuffer()
                    .ok_or(MetalError::CommandBufferCreation)?;
                // イシュー #1099 診断カウンタ: 新規コマンドバッファ生成の
                // たびに 1 カウントする（バッチが開くたび = 1 回）。
                self.diag_command_buffers.fetch_add(1, Ordering::Relaxed);
                slots.open = Some(Batch {
                    cmd_buf,
                    encoder: None,
                    meta: BatchMeta::new(),
                    in_flight: Vec::new(),
                    tokens: Vec::new(),
                });
            }
            // 直前で `is_none()` なら新規構築し `Some` にしたばかりであり、
            // 到達直後に `None` へ戻す経路はこの関数内に存在しない。
            // ただし `.claude/rules/coding-rust.md`「本番経路で
            // unwrap/expect を使わない」に従い、万一の到達不能パスも
            // panic ではなく型付きエラー（`BatchStateUnavailable`）で
            // 呼び出し元へ伝える（codex-review P1 指摘対応）。
            let batch = slots
                .open
                .as_mut()
                .ok_or_else(|| MetalError::BatchStateUnavailable {
                    detail: "encode: open batch missing immediately after construction".to_string(),
                })?;

            if batch.encoder.is_none() {
                let encoder = batch
                    .cmd_buf
                    .computeCommandEncoder()
                    .ok_or(MetalError::ComputeEncoderCreation)?;
                batch.encoder = Some(encoder);
            }
            let encoder =
                batch
                    .encoder
                    .as_ref()
                    .ok_or_else(|| MetalError::BatchStateUnavailable {
                        detail: "encode: encoder missing immediately after construction"
                            .to_string(),
                    })?;

            encode_fn(encoder);

            batch
                .in_flight
                .extend(resources.iter().map(|buf| buf.retain()));
            if let Some(token) = token {
                batch.tokens.push(token.clone());
            }
            batch.meta.record_dispatch(label);

            if batch.meta.should_auto_flush() {
                self.flush_locked(&mut slots);
            }
            Ok(())
        })
    }

    /// 開いているバッチがあれば `endEncoding()` + `commit()` する
    /// （**待たない**）。`slots.open` を `slots.committed` へ移し、後続の
    /// `encode` は新しいコマンドバッファから開始する。
    ///
    /// [`batch_state::BatchMeta::should_auto_flush`] が
    /// [`batch_state::MAX_DISPATCHES_PER_BATCH`] 到達を検出した際の安全弁
    /// （[`Self::encode`] から呼ばれる）、および [`Self::synchronize`] が
    /// 「まず全ての未 commit 分を commit してから待つ」ために呼ぶ内部
    /// ヘルパ。呼び出し元は `self.batch` を既にロック済みであることが
    /// 前提（デッドロック回避のため `&mut BatchSlots` を直接受け取る）。
    fn flush_locked(&self, slots: &mut BatchSlots) {
        if let Some(mut batch) = slots.open.take() {
            if let Some(encoder) = batch.encoder.take() {
                encoder.endEncoding();
            }
            batch.cmd_buf.commit();
            slots.committed.push(batch);
        }
    }

    /// 開いている・commit 済みの全バッチを `waitUntilCompleted()` で
    /// 完了させ、実行時エラーがあれば呼び出し元へ返す
    /// **ホスト実体化時に呼ぶ唯一の同期点**（`memory.rs::download_inner`／
    /// `zero_fill`・`Self::dispatch_sync`・`Drop` から呼ばれる。イシュー
    /// #1017・設計文書 §3.5）。
    ///
    /// 開いているバッチはまず `flush_locked` で commit してから
    /// 待つ。複数バッチにまたがって失敗した場合、**全バッチの登録済み
    /// トークンへ伝播**したうえで最初のエラーを返す（`.claude/rules/
    /// security.md` A08。1 個のエラーで打ち切って残りのバッチの失敗を
    /// 検出し損なわない）。`self.batch` のロックは待機（`waitUntilCompleted`
    /// はブロッキング呼び出し）中も保持し続ける: 他スレッドからの
    /// `encode`／`synchronize` はこの間ブロックされる（正しさを優先し
    /// 並行性は最適化しない設計判断。設計文書 §3.5「同時実行の扱い」）。
    ///
    /// `flush_locked` の `endEncoding()`／`commit()` に加え、本メソッド
    /// 自身の `waitUntilCompleted()`／`status()`／`error()` も
    /// autoreleased なオブジェクトを返しうる（`encode` の
    /// autoreleasepool コメント参照）。`encode` から `should_auto_flush`
    /// 経由で `flush_locked` のみが呼ばれる経路は `encode` 側の
    /// `autoreleasepool` で覆われているが、本メソッドを直接呼ぶ経路
    /// （`memory.rs::download_inner`／`zero_fill`・`Drop`）はそれを
    /// 経由しないため、ここで独自に `autoreleasepool` を張らないと
    /// プロセス寿命分蓄積する（Cursor Bugbot 指摘・PR #1057）。
    ///
    /// 本番経路は `Self::synchronize_observed`（private）を no-op オブザーバで
    /// 呼ぶ薄いラッパー（イシュー #1276）。挙動・エラー伝播・
    /// `pending_pool_returns` 合流順序は本メソッドが唯一の実体だった
    /// 時点から不変（AC-2）。
    pub fn synchronize(&self) -> Result<(), MetalError> {
        self.synchronize_observed(|_cmd_buf, _meta| {})
    }

    /// [`Self::synchronize`] の本体（イシュー #1276 で切り出し）。
    /// 完了したバッチごとに `observe` を 1 回呼ぶ点のみが違い、それ
    /// 以外のロック区間・エラー集約・`pending_pool_returns` 合流順序は
    /// 従来の `synchronize` と完全に同一。
    ///
    /// `observe` はバッチ 1 個につき `waitUntilCompleted()`＋`status()`
    /// 判定の直後・`batch`（`in_flight` retain 列を含む）が drop される
    /// 直前に、`self.batch` のロックを保持したまま呼ばれる（`Batch:
    /// Send` の SAFETY コメントが要求する「`Mutex` 下でのみ `Batch` へ
    /// 触れる」不変条件を維持したまま診断値を読む）。GPU タイムスタンプ
    /// 変種（`gemm_reuse_phase_diag_tests.rs::synchronize_with_gpu_
    /// timestamps`）が `MTLCommandBuffer::GPUStartTime`/`GPUEndTime` を
    /// ここから読む。本番 `synchronize()` は no-op オブザーバを渡すため
    /// 追加の FFI 呼び出しはゼロ（AC-2 の性能非後退根拠）。
    fn synchronize_observed<F>(&self, mut observe: F) -> Result<(), MetalError>
    where
        F: FnMut(&MtlCommandBuffer, &BatchMeta),
    {
        autoreleasepool(|_pool| {
            let mut slots = self.lock_batch("synchronize")?;
            self.flush_locked(&mut slots);
            let batches = std::mem::take(&mut slots.committed);

            let mut first_error: Option<MetalError> = None;
            for batch in batches {
                batch.cmd_buf.waitUntilCompleted();
                // イシュー #1099 診断カウンタ: `waitUntilCompleted()` の
                // 呼び出し回数（バッチ 1 個の完了待ちごとに 1）。
                self.diag_wait_until_completed
                    .fetch_add(1, Ordering::Relaxed);
                if batch.cmd_buf.status() == MTLCommandBufferStatus::Error {
                    let message = batch
                        .cmd_buf
                        .error()
                        .map(|error| error.localizedDescription().to_string())
                        .unwrap_or_else(|| {
                            "no NSError attached to failed command buffer".to_string()
                        });
                    let formatted =
                        batch_state::format_failure_message(batch.meta.labels(), &message);
                    batch_state::propagate_failure(&batch.tokens, &formatted);
                    if first_error.is_none() {
                        first_error =
                            Some(MetalError::CommandBufferExecutionFailed { message: formatted });
                    }
                }
                // 診断オブザーバ（イシュー #1276）: `waitUntilCompleted()`
                // 完了・エラー判定後・`batch` drop 前に呼ぶ（GPU 実行が
                // 完了済みのタイムスタンプを読める最後の地点）。
                observe(&batch.cmd_buf, &batch.meta);
                // `batch`（`in_flight` の retain 列を含む）はこのループの
                // 末尾で drop される。GPU 実行は `waitUntilCompleted()`
                // 直後の時点で完了済みのため、ここで解放してよい。
            }

            // フェーズ (i) の合流（イシュー #1021・設計文書 §3.3
            // 「Metal」）: `waitUntilCompleted()` 完了後、**`Ok`／`Err`
            // いずれで復帰する場合も**、`pending_pool_returns` を
            // `BatchSlots` の同一ロック区間内で `drain_for_merge` して
            // から `put_all_merged`（`SizeClassPool::put_merged` が
            // `pending_return_bytes` の減算とフリーリスト挿入・総量
            // 上限判定を単一ロック区間内で行う。`Mutex<PoolCore<H>>` を
            // 要するため、ここでロックを解放した後に呼ぶ。§3.5「ロック
            // 順序規則」）する。合流はフェーズ (i) 自体が担う入出金で
            // あり「解放処理」ではないため、`first_error` の有無に
            // 関わらず必ず実行する。
            let drained = slots.pending_pool_returns.drain_for_merge();
            drop(slots);
            // 合流経路は `put_all_merged`（`SizeClassPool::put_merged`）を
            // 使う（`drain_for_merge` 由来のエントリは既に
            // `record_pending_return` 済みのため。`put_all`〈即時返却
            // 専用〉へ渡すと対の減算が呼ばれず `pending_return_bytes` が
            // 恒久的に高止まりする。Cursor Bugbot High・codex P2 指摘
            // 対応。PR #1063 追加是正）。
            pool_pending::put_all_merged(drained);

            match first_error {
                Some(err) => Err(err),
                None => Ok(()),
            }
        })
    }

    /// 診断専用（イシュー #1276。`#[cfg(test)] pub(crate)` の可視性は
    /// `gemm.rs::MetalGemm::diag_encode_tiled_nn` と同じ方針）:
    /// [`Self::synchronize_observed`] を GPU タイムスタンプ収集
    /// オブザーバで呼び、完了した各バッチの
    /// [`BatchGpuTimestamps`] を呼ばれた順に返す。
    ///
    /// `crates/backend-metal/src/gemm_reuse_phase_diag_tests.rs`
    /// （#1189 の Layer B 分解）が `commit_wait` 区間内の純カーネル
    /// 専有時間を分離するために使う（同ファイル冒頭コメント「GPU
    /// タイムスタンプ変種」参照）。本番 `synchronize()` は no-op
    /// オブザーバのままのため、本メソッドの追加は本番経路の FFI
    /// 呼び出し回数・挙動を一切変えない（AC-2）。
    #[cfg(test)]
    pub(crate) fn synchronize_with_gpu_timestamps(
        &self,
    ) -> Result<Vec<BatchGpuTimestamps>, MetalError> {
        let mut collected = Vec::new();
        let result = self.synchronize_observed(|cmd_buf, meta| {
            // SAFETY ではなく単純な safe メソッド呼び出し: `GPUStartTime`/
            // `GPUEndTime`（objc2-metal =0.3.2 生成コード
            // `MTLCommandBuffer.rs:407-416`）は `unsafe fn` ではない
            // （`unsafe(method_family = none)` 属性はコード生成マクロの
            // 内部注釈であり呼び出し側に `unsafe` を要求しない）。
            // `CFTimeInterval = c_double`（`objc2-core-foundation`
            // `CFDate.rs`）で `f64` と同値。未開始／完了通知未受領時は
            // 0.0 を返す契約（Apple ドキュメント。生成コードのコメント
            // 参照）ため `BatchGpuTimestamps::from_raw` で `None` へ
            // 変換する。
            let gpu_start = cmd_buf.GPUStartTime();
            let gpu_end = cmd_buf.GPUEndTime();
            collected.push(BatchGpuTimestamps::from_raw(gpu_start, gpu_end, meta));
        });
        result.map(|()| collected)
    }

    /// コンピュートエンコーダを生成し `encode_fn` にディスパッチ内容の
    /// 記録を委ね、即座に完了を待つ同期実行ヘルパ（イシュー #1017 で
    /// `encode` + [`Self::synchronize`] の薄いラッパーへ変更した。
    /// シグネチャ・戻り値の意味は不変であり既存呼び出し元
    /// （`gemm.rs`／`elementwise.rs`／`rmsnorm.rs`／`softmax.rs`）は無
    /// 変更のまま動作する）。同期方式は PoC-v2-4 の計測境界
    /// （`GemmCase::dispatch`）と同一にし、v1 系と揃える（バックエンド間
    /// 比較の計測条件を崩さないため）。
    ///
    /// `resources` を渡さない（`&[]`）のは、このバッファ列が
    /// `synchronize()` 完了まで呼び出し元の関数スコープで生存する契約
    /// （呼び出し元が `dispatch_sync` の戻り待ちを行う）ため、
    /// `in_flight` による追加の生存延長が不要なため
    /// （`sgd.rs::MetalSgd::run` が `Self::encode` を直接使う際は
    /// `param`／`grad`／`velocity` を渡す点と対照的）。
    pub fn dispatch_sync<F>(&self, encode_fn: F) -> Result<(), MetalError>
    where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    {
        self.encode("dispatch_sync", &[], None, encode_fn)?;
        self.synchronize()
    }
}

impl Drop for MetalContext {
    /// プロセス終了・スイートキャッシュ解放時に、開いている・commit 済み
    /// のバッチを `synchronize()` 相当で完了させる（イシュー #1017・
    /// 設計文書 §3.5「同期点」表）。`Drop` からは `Result` を返せない
    /// ため、失敗（GPU fault 等）はここでは伝播せず無視する: `Drop` 時点
    /// で呼び出し元は既に結果を受け取れない一方、GPU 側のコマンド
    /// バッファ実行自体は本メソッドを呼ばなくても最終的に完了する
    /// （破棄後もハンドルが握っていた in-flight バッファの解放を
    /// 妨げないための best-effort な後始末であり、`unwrap`/`expect` の
    /// ような panic 経路ではない）。
    fn drop(&mut self) {
        let _ = self.synchronize();
    }
}
