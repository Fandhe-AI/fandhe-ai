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
use crate::tile::OccupancyParams;

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
    /// コマンドバッファ共有バッチの状態（イシュー #1017）。
    /// [`MetalContext::encode`]／[`MetalContext::synchronize`]（内部
    /// ヘルパ `flush_locked` を含む）がこの `Mutex` を介してのみ
    /// アクセスする（[`Batch`] の SAFETY コメント参照）。
    batch: Mutex<BatchSlots>,
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

        Ok(Self {
            device,
            queue,
            caps,
            occupancy_params,
            batch: Mutex::new(BatchSlots::default()),
        })
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

            if slots.open.is_none() {
                let cmd_buf = self
                    .queue
                    .commandBuffer()
                    .ok_or(MetalError::CommandBufferCreation)?;
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
    /// 開いているバッチはまず [`Self::flush_locked`] で commit してから
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
    /// autoreleased なオブジェクトを返しうる（[`Self::encode`] の
    /// autoreleasepool コメント参照）。`encode` から `should_auto_flush`
    /// 経由で `flush_locked` のみが呼ばれる経路は `encode` 側の
    /// `autoreleasepool` で覆われているが、本メソッドを直接呼ぶ経路
    /// （`memory.rs::download_inner`／`zero_fill`・`Drop`）はそれを
    /// 経由しないため、ここで独自に `autoreleasepool` を張らないと
    /// プロセス寿命分蓄積する（Cursor Bugbot 指摘・PR #1057）。
    pub fn synchronize(&self) -> Result<(), MetalError> {
        autoreleasepool(|_pool| {
            let mut slots = self.lock_batch("synchronize")?;
            self.flush_locked(&mut slots);
            let batches = std::mem::take(&mut slots.committed);

            let mut first_error: Option<MetalError> = None;
            for batch in batches {
                batch.cmd_buf.waitUntilCompleted();
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
                // `batch`（`in_flight` の retain 列を含む）はこのループの
                // 末尾で drop される。GPU 実行は `waitUntilCompleted()`
                // 直後の時点で完了済みのため、ここで解放してよい。
            }
            match first_error {
                Some(err) => Err(err),
                None => Ok(()),
            }
        })
    }

    /// コンピュートエンコーダを生成し `encode_fn` にディスパッチ内容の
    /// 記録を委ね、即座に完了を待つ同期実行ヘルパ（イシュー #1017 で
    /// [`Self::encode`] + [`Self::synchronize`] の薄いラッパーへ変更した。
    /// シグネチャ・戻り値の意味は不変であり既存呼び出し元
    /// （`gemm.rs`／`elementwise.rs`／`rmsnorm.rs`／`softmax.rs`）は無
    /// 変更のまま動作する）。同期方式は PoC-v2-4 の計測境界
    /// （`GemmCase::dispatch`）と同一にし、v1 系と揃える（バックエンド間
    /// 比較の計測条件を崩さないため）。
    ///
    /// `resources` を渡さない（`&[]`）のは、このバッファ列が
    /// `synchronize()` 完了まで呼び出し元の関数スコープで生存する契約
    /// （呼び出し元が `dispatch_sync` の戻り待ちを行う）ため、
    /// [`Batch::in_flight`] による追加の生存延長が不要なため
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
