//! Metal バックエンド。
//!
//! Metal バインディング経路は `wgpu` ではなく `objc2-metal` 直接呼び出しを採用する
//! （TASK-1.8d、#41）。PoC-v2-4 実測（Apple M4 Max）で、同一アルゴリズム（tiled GEMM）
//! 比較において `objc2-metal` 直接実装が `wgpu`（境界検査無効化後）より約 2.3 倍高速
//! （size=4096: 2.123 TFLOPS 対 0.920 TFLOPS）であり、かつ `simdgroup_matrix`（8×8
//! ハードウェア行列演算命令）は WGSL に相当命令が存在せず `wgpu` 経由では原理的に
//! 到達不可能なことを確認済み（Metal 直接はさらに約 1.5 倍、3.134 TFLOPS）。
//! `wgpu` は 124 パッケージロックの大規模依存 + `pollster` を要するのに対し、
//! `objc2` 系は許容依存 3 crate（`.claude/rules/deps-policy.md`）で `unsafe` を
//! FFI 境界に局所化できる。判断根拠・実測値の全体は
//! `docs/spec/03-poc/poc-v2-4-metal-gemm/README.md`「経路選定の比較判断」節（正本）と
//! `docs/backend-metal-wgpu-decision.md`（実装リポ側の要約。REQ-8 境界検査規約との
//! 関係も記載）を参照。「約 2.3 倍」は naga の自動境界検査を無効化した後の wgpu 値との
//! 比較であり、WGSL 側の手動境界チェック自体は維持された状態での計測である
//! （境界検査省略の正当化に用いない。REQ-8）。
//!
//! `tensor-core` の演算グラフノードを MSL カーネル（simdgroup 系命令を含む）へ変換して実行する。
//! バックエンド切替は feature フラグなしの cfg ベース（PoC-v2-5 実証構成。REQ-2）とし、
//! `objc2` / `objc2-foundation` / `objc2-metal` は `cfg(target_os = "macos")` で分離する
//! （非 macOS 環境のビルドに影響を与えない。`.claude/rules/deps-policy.md`）。
//!
//! `backend-cpu` との数値一致は統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」で
//! 検証する。丸め方針（FMA 契約）は Metal `simdgroup_multiply_accumulate` の既定 FMA 契約を
//! CPU 参照実装（`f32::mul_add`）と揃える（PoC-v2-5 の K=4096 ストレスケースで実測確認済み。
//! `.claude/rules/coding-rust.md`）。カーネルの手動境界検査は最適化を理由に省略しない（REQ-8）。
//! FFI 境界の `unsafe`（objc2 系）は必要最小限に留め理由コメントを付す
//! （`.claude/rules/security.md`）。
//!
//! TASK-1.8a（#38）でデバイス・コマンドキュー・バッファ管理の基盤（[`context::MetalContext`]・
//! [`buffer::MetalBuffer`]・[`error::MetalError`]）を実装済み。TASK-1.8b（#39）で MSL 実行時
//! コンパイル・パイプライン構築（[`pipeline`]）・naive GEMM ディスパッチ経路（[`gemm::MetalGemm`]）
//! を追加した。TASK-1.8c（#40）で tiled・simdgroup カーネル（[`gemm::GemmVariant`]・
//! [`gemm::MetalGemm::dispatch_variant`]）と 8 の倍数パディングユーティリティ（[`pad`]）を
//! 追加し、`shaders/gemm.metal` の naive/tiled/simdgroup 3 段すべてを実装済みにした
//! （spec 根拠: `docs/spec/05-tasks.md` TASK-1.1・TASK-1.8）。
//!
//! 非 macOS 環境ではモジュールごとコンパイル対象外になる（feature フラグなしの cfg ベース。
//! PoC-v2-5 実証構成・REQ-2）。[`pad`] のみ `objc2` 系 FFI に触れない純粋関数群のため
//! `cfg(target_os = "macos")` を付けず、Linux（CI・本実装環境）でも単体テストが回る。
//!
//! TASK-1.9a（#44）で [`device`] モジュール（[`device::MetalDeviceProvider`]）を追加した。
//! `tensor_core::device::DeviceProvider` の Metal 実装であり、CPU／CUDA 実装
//! （`backend-cpu::CpuDeviceProvider`／`backend-cuda::device::CudaDeviceProvider`）と
//! 同一 trait で列挙・選択できることを macOS 実機上のテストで検証する。`Device::Metal`
//! 自体が `cfg(target_os = "macos")` 限定のため、本モジュールもクレート全体でこの cfg を
//! 付す（非 macOS 環境のビルドに影響を与えない）。
//!
//! TASK-1.8f（#188）で動的タイル選択（[`tile`]）を追加した。`gemm_simdgroup`（1 threadgroup =
//! 1 simdgroup = C の 8×8 タイル 1 つ）はタイルサイズの自由度がなく、MLX steel カーネル方式
//! （BM/BN/BK/WM/WN のパラメータ化＋行列サイズ別動的選択）の性能差の核心に対応できないため、
//! `shaders/gemm.metal` に MSL function constant でパラメータ化した `gemm_simdgroup_tiled` を追加し、
//! [`gemm::GemmVariant::SimdgroupTiled`]／[`gemm::MetalGemm::dispatch_auto`] から利用する。
//! [`tile`] 自体は `objc2` 系 FFI に触れない純粋関数群のため他モジュールと異なり
//! `cfg(target_os = "macos")` を付けない（[`pad`] と同じ設計判断。Linux でも単体テストが回る）。
//!
//! TASK-1.9b（#45）で [`memory`] モジュール（[`memory::MetalMemory`]）を追加した。
//! `tensor_core::buffer::MemoryOps` の Metal 実装であり、新規 `unsafe` を追加せず
//! 既存の [`buffer::MetalBuffer`]（`new_with_data`／`new_zeroed`／`read_to_vec`）を
//! そのまま再利用する。`StorageModeShared`（UMA）のため CUDA のような明示同期は
//! 不要（`memory.rs` モジュールコメント参照）。
//!
//! TASK-11.2b（#68）で GEMM 自動経路選択入口
//! （[`gemm::MetalGemm::dispatch_backend_auto`]）を追加した。
//! `tensor_core::dispatch::select_gemm_kernel`（#67 が設計した決定的規則。
//! `docs/dispatch-rules-design.md`）が返す経路に従い、`simdgroup_matrix`
//! （[`gemm::MetalGemm::dispatch_auto`] 経由）／tiled／naive を呼び分ける。
//! 判定材料となる `MTLDevice::supportsFamily(MTLGPUFamily::Apple7)` は
//! [`context::MetalContext::new`] 時に 1 回評価しキャッシュする
//! （[`context::MetalContext::caps`]）。既存の [`gemm::MetalGemm::dispatch`]
//! （naive）／[`gemm::MetalGemm::dispatch_variant`]（経路直接指定）は
//! テスト・証跡用途（#70）にそのまま温存する（`docs/dispatch-rules-design.md`
//! §5.4）。
//!
//! TASK-1.9c（#46）で [`ops`] モジュール（[`ops::MetalBackendOps`]）を追加した。
//! `tensor_core::backend_ops::BackendOps` の Metal 実装であり、`gemm` は
//! [`gemm::MetalGemm::dispatch_auto`]（実装済みの動的タイル選択）へ委譲する。
//! elementwise・reduction は GPU カーネル未実装のため
//! `tensor_core::device::BackendError::Unsupported` を返す
//! （out-of-scope-tracking.md 対象）。`device` モジュールと同じく
//! `cfg(target_os = "macos")` 限定。
//!
//! TASK-8.3b（#156）で REQ-8「Metal f16 対 PyTorch MPS f16」の実測対象と
//! なる f16 GEMM カーネル（`shaders/gemm.metal` の `gemm_simdgroup_f16`。
//! A/B は `simdgroup_half8x8`、アキュムレータは `simdgroup_float8x8`
//! （f32 累算。イシュー #380 の実機検証で half 統一から変更）。カーネル
//! 冒頭コメントに精度契約の判断根拠を記載）と、その明示ディスパッチ入口
//! （`gemm::MetalGemm::dispatch_f16_unverified`）を追加した。既存の
//! [`gemm::MetalGemm::dispatch_auto`]／`dispatch_backend_auto`（f32 専用の
//! 自動経路選択）はそのまま変更していない（f16 の自動ディスパッチ統合は
//! 本 TASK のスコープ外。`docs/dispatch-rules-design.md` 参照）。f16 専用の
//! Metal バッファ型 [`half_buffer::MetalHalfBuffer`] を新設し、既存
//! [`buffer::MetalBuffer`]（f32 専用）のシグネチャには一切手を入れていない。
//!
//! イシュー #541（D-7a）で occupancy 目標算出の基盤を追加した:
//! [`device::probe_gpu_core_count`]（IOKit `AGXAccelerator` 実測。
//! `device` モジュールと同じく `cfg(target_os = "macos")` 限定）・
//! [`device::MetalOccupancyInfo`]・[`tile::OccupancyParams`]
//! （`tile::actual_groups`／`tile::is_underoccupied` と合わせ `objc2` 系
//! FFI に触れない純粋関数群）。
//!
//! イシュー #542（D-7b）で [`tile::select_with_occupancy`] を実装した。
//! [`context::MetalContext::new`] が `MetalOccupancyInfo::probe` を 1 回
//! だけ実行して `Option<tile::OccupancyParams>` へ写像・キャッシュする
//! （[`context::MetalContext::occupancy_params`]）。**ただし
//! `gemm::MetalGemm::dispatch_auto`（本番ディスパッチ経路）は現時点では
//! `tile::select`（形状のみ）を呼ぶ**: `ideal_groups` の係数（MFA 経験式
//! 由来の暫定値）は M4 Max 実機での `select()` 比・性能非劣化確認
//! （`docs/perf/metal-gemm-occupancy-select.md` §5）が未完了のため、
//! `select_with_occupancy` への切替は当該実測完了後に別 PR で行う
//! （codex-review P1・PR #684）。GPU コア数取得不能時のフォールバック
//! 挙動（`occupancy_params` が `None` になり `select_with_occupancy` が
//! 形状のみの判定へ fail-safe する）自体は実装・テスト済み。
//!
//! `dispatch_f16_unverified`／`dispatch_f16_prepared_unverified` は関数名に
//! `_unverified` を付け `#[doc(hidden)]` としている（PR #346 codex-review
//! P1-2 指摘）。REQ-2 複合判定を満たすことはイシュー #380 で Metal 実機
//! （M4 Max・macOS 26.6）実測により確認済みだが、production 経路
//! （`dispatch_auto`／`dispatch_backend_auto`）への統合はスコープ外のため
//! `_unverified` suffix・`#[doc(hidden)]` は当面維持する。詳細は
//! [`gemm::MetalGemm::dispatch_f16_unverified`] のドキュメントコメントを
//! 参照）。
//!
//! イシュー #604 で融合 RMSNorm 順伝播カーネル（[`rmsnorm::MetalRmsNorm`]）と
//! online softmax カーネル（[`softmax::MetalSoftmax`]）を MSL で追加した。
//! CUDA 側 G-6（#592）と同一アルゴリズム契約（1 パス／2 パス経路・
//! persistent threadgroup・FMA 契約統一）を採るが、`MetalContext::
//! dispatch_sync` が動的 threadgroup memory 設定 API を経由しないため
//! 1 パス経路はコンパイル時固定長の `threadgroup` 配列を使う（[`row_kernel`]
//! モジュール冒頭コメント参照）。両カーネル・`row_kernel` の経路選択・
//! canonical 融合プラン照合の cfg 非依存部分は [`row_kernel`] に集約し、
//! [`ops::MetalBackendOps::run_fused`] からルーティングする。softmax の
//! CUDA 直接 parity 相手（#594・G-7）は本イシュー時点で未実装のため、
//! 両バックエンドとも CPU 参照実装（REQ-2 統一複合判定）に対する数値一致を
//! 経由した推移的な担保に留まる（`softmax.rs`／`tests/softmax_parity.rs`
//! ドキュメンテーションコメント参照）。

#[cfg(target_os = "macos")]
pub mod buffer;
#[cfg(target_os = "macos")]
pub mod context;
#[cfg(target_os = "macos")]
pub mod device;
#[cfg(target_os = "macos")]
pub mod error;
#[cfg(target_os = "macos")]
pub mod gemm;
#[cfg(target_os = "macos")]
pub mod half_buffer;
#[cfg(target_os = "macos")]
pub mod memory;
#[cfg(target_os = "macos")]
pub mod ops;
pub mod pad;
#[cfg(target_os = "macos")]
pub mod pipeline;
#[cfg(target_os = "macos")]
pub mod rmsnorm;
// `pad`／`tile` と同じ設計判断: `objc2` 系 FFI に触れないため
// `cfg(target_os = "macos")` を付けず Linux でも単体テストが回るが、
// 実際の呼び出し元（`ops.rs`／`rmsnorm.rs`／`softmax.rs`）は macOS 限定
// のため、`pub(crate)` のままだと Linux 単体ビルド（`cargo build`／
// `cargo clippy` の非テストパス）で「クレート内から到達不能」と
// 判定され dead_code lint が誤検知する。`pad`／`tile` に倣い `pub` に
// する（`row_kernel` モジュール自体は公開 API 面〈`facade`〉から到達
// しないため実質的な公開面拡大にはならない）。
pub mod row_kernel;
#[cfg(target_os = "macos")]
pub mod softmax;
pub mod tile;

// `MTLCreateSystemDefaultDevice` は CoreGraphics framework がリンクされた
// バイナリでのみ確実にデバイスを返す（プレーンな CLI バイナリ ―― 本クレートの
// test/bench 実行ファイル等 ―― では `MTLCreateSystemDefaultDevice` が nil を
// 返しうる。Apple の Metal サンプル・Homebrew 経由の CLI ツールが軒並み
// CoreGraphics をリンクしているのはこのため）。`objc2-core-graphics` は
// 許容依存 8 区分（`.claude/rules/deps-policy.md`）に含まれず追加はユーザー
// 承認が要るため、クレート依存を増やさずリンカディレクティブのみで解決する
// （extern ブロック自体は空でよく、`#[link]` 属性がリンク時に
// `-framework CoreGraphics` を linker へ伝搬する）。
#[cfg(target_os = "macos")]
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

#[cfg(target_os = "macos")]
pub use buffer::MetalBuffer;
#[cfg(target_os = "macos")]
pub use context::MetalContext;
#[cfg(target_os = "macos")]
pub use device::MetalDeviceProvider;
#[cfg(target_os = "macos")]
pub use device::{MetalOccupancyInfo, probe_gpu_core_count};
#[cfg(target_os = "macos")]
pub use error::MetalError;
#[cfg(target_os = "macos")]
pub use gemm::{GemmVariant, MetalGemm};
#[cfg(target_os = "macos")]
pub use half_buffer::MetalHalfBuffer;
#[cfg(target_os = "macos")]
pub use memory::MetalMemory;
#[cfg(target_os = "macos")]
pub use ops::MetalBackendOps;
#[cfg(target_os = "macos")]
pub use rmsnorm::MetalRmsNorm;
#[cfg(target_os = "macos")]
pub use softmax::MetalSoftmax;
pub use tile::TileConfig;
