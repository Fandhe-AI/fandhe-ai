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
//! TASK-1.8a（#38）でデバイス・コマンドキュー・バッファ管理の基盤（`context::MetalContext`・
//! `buffer::MetalBuffer`・`error::MetalError`）を実装済み。TASK-1.8b（#39）で MSL 実行時
//! コンパイル・パイプライン構築（`pipeline`）・naive GEMM ディスパッチ経路（`gemm::MetalGemm`）
//! を追加した。TASK-1.8c（#40）で tiled・simdgroup カーネル（`gemm::GemmVariant`・
//! `gemm::MetalGemm::dispatch_variant`）と 8 の倍数パディングユーティリティ（[`pad`]）を
//! 追加し、`shaders/gemm.metal` の naive/tiled/simdgroup 3 段すべてを実装済みにした
//! （spec 根拠: `docs/spec/05-tasks.md` TASK-1.1・TASK-1.8）。
//!
//! 非 macOS 環境ではモジュールごとコンパイル対象外になる（feature フラグなしの cfg ベース。
//! PoC-v2-5 実証構成・REQ-2）。[`pad`] のみ `objc2` 系 FFI に触れない純粋関数群のため
//! `cfg(target_os = "macos")` を付けず、Linux（CI・本実装環境）でも単体テストが回る。
//!
//! TASK-1.9a（#44）で `device` モジュール（`device::MetalDeviceProvider`）を追加した。
//! `fandhe_ai_tensor_core::device::DeviceProvider` の Metal 実装であり、CPU／CUDA 実装
//! （`backend-cpu::CpuDeviceProvider`／`backend-cuda::device::CudaDeviceProvider`）と
//! 同一 trait で列挙・選択できることを macOS 実機上のテストで検証する。`Device::Metal`
//! 自体が `cfg(target_os = "macos")` 限定のため、本モジュールもクレート全体でこの cfg を
//! 付す（非 macOS 環境のビルドに影響を与えない）。
//!
//! TASK-1.8f（#188）で動的タイル選択（[`tile`]）を追加した。`gemm_simdgroup`（1 threadgroup =
//! 1 simdgroup = C の 8×8 タイル 1 つ）はタイルサイズの自由度がなく、MLX steel カーネル方式
//! （BM/BN/BK/WM/WN のパラメータ化＋行列サイズ別動的選択）の性能差の核心に対応できないため、
//! `shaders/gemm.metal` に MSL function constant でパラメータ化した `gemm_simdgroup_tiled` を追加し、
//! `gemm::GemmVariant::SimdgroupTiled`／`gemm::MetalGemm::dispatch_auto` から利用する。
//! [`tile`] 自体は `objc2` 系 FFI に触れない純粋関数群のため他モジュールと異なり
//! `cfg(target_os = "macos")` を付けない（[`pad`] と同じ設計判断。Linux でも単体テストが回る）。
//!
//! TASK-1.9b（#45）で `memory` モジュール（`memory::MetalMemory`）を追加した。
//! `fandhe_ai_tensor_core::buffer::MemoryOps` の Metal 実装であり、新規 `unsafe` を追加せず
//! 既存の `buffer::MetalBuffer`（`new_with_data`／`new_zeroed`／`read_to_vec`）を
//! そのまま再利用する。`StorageModeShared`（UMA）のため CUDA のような明示同期は
//! 不要（`memory.rs` モジュールコメント参照）。
//!
//! TASK-11.2b（#68）で GEMM 自動経路選択入口
//! （`gemm::MetalGemm::dispatch_backend_auto`）を追加した。
//! `fandhe_ai_tensor_core::dispatch::select_gemm_kernel`（#67 が設計した決定的規則。
//! `docs/dispatch-rules-design.md`）が返す経路に従い、`simdgroup_matrix`
//! （`gemm::MetalGemm::dispatch_auto` 経由）／tiled／naive を呼び分ける。
//! 判定材料となる `MTLDevice::supportsFamily(MTLGPUFamily::Apple7)` は
//! `context::MetalContext::new` 時に 1 回評価しキャッシュする
//! （`context::MetalContext::caps`）。既存の `gemm::MetalGemm::dispatch`
//! （naive）／`gemm::MetalGemm::dispatch_variant`（経路直接指定）は
//! テスト・証跡用途（#70）にそのまま温存する（`docs/dispatch-rules-design.md`
//! §5.4）。
//!
//! TASK-1.9c（#46）で `ops` モジュール（`ops::MetalBackendOps`）を追加した。
//! `fandhe_ai_tensor_core::backend_ops::BackendOps` の Metal 実装であり、`gemm` は
//! `gemm::MetalGemm::dispatch_auto`（実装済みの動的タイル選択）へ委譲する。
//! elementwise・reduction は GPU カーネル未実装のため
//! `fandhe_ai_tensor_core::device::BackendError::Unsupported` を返す
//! （out-of-scope-tracking.md 対象）。`device` モジュールと同じく
//! `cfg(target_os = "macos")` 限定。
//!
//! TASK-8.3b（#156）で REQ-8「Metal f16 対 PyTorch MPS f16」の実測対象と
//! なる f16 GEMM カーネル（`shaders/gemm.metal` の `gemm_simdgroup_f16`。
//! A/B は `simdgroup_half8x8`、アキュムレータは `simdgroup_float8x8`
//! （f32 累算。イシュー #380 の実機検証で half 統一から変更）。カーネル
//! 冒頭コメントに精度契約の判断根拠を記載）と、その明示ディスパッチ入口
//! （`gemm::MetalGemm::dispatch_f16_unverified`）を追加した。既存の
//! `gemm::MetalGemm::dispatch_auto`／`dispatch_backend_auto`（f32 専用の
//! 自動経路選択）はそのまま変更していない（f16 の自動ディスパッチ統合は
//! 本 TASK のスコープ外。イシュー #798 で `gemm::MetalGemm::dispatch_f16_auto_unverified`
//! として実現した。`docs/dispatch-rules-design.md` 参照）。f16 専用の
//! Metal バッファ型 `half_buffer::MetalHalfBuffer` を新設し、既存
//! `buffer::MetalBuffer`（f32 専用）のシグネチャには一切手を入れていない。
//!
//! イシュー #541（D-7a）で occupancy 目標算出の基盤を追加した:
//! `device::probe_gpu_core_count`（IOKit `AGXAccelerator` 実測。
//! `device` モジュールと同じく `cfg(target_os = "macos")` 限定）・
//! `device::MetalOccupancyInfo`・[`tile::OccupancyParams`]
//! （`tile::actual_groups`／`tile::is_underoccupied` と合わせ `objc2` 系
//! FFI に触れない純粋関数群）。
//!
//! イシュー #542（D-7b）で [`tile::select_with_occupancy`] を実装した。
//! `context::MetalContext::new` が `MetalOccupancyInfo::probe` を 1 回
//! だけ実行して `Option<tile::OccupancyParams>` へ写像・キャッシュする
//! （`context::MetalContext::occupancy_params`）。**ただし
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
//! （M4 Max・macOS 26.6）実測により確認済みだが、非タイル `gemm_simdgroup_f16`
//! 自体は production 自動経路（`dispatch_auto`／`dispatch_backend_auto`）へ
//! 統合しない方針（イシュー #798 の後方互換方針）のため `_unverified`
//! suffix・`#[doc(hidden)]` は当面維持する。タイル化 f16 カーネル
//! （`gemm_simdgroup_tiled_f16`）は #798 で `gemm::MetalGemm::dispatch_f16_auto_unverified`
//! から動的タイル選択付きで自動経路へ統合済み（明示 `cfg` 指定用の
//! `dispatch_f16_tiled_unverified`／`dispatch_f16_tiled_prepared_unverified`
//! 自体は引き続き `_unverified` suffix・`#[doc(hidden)]` を維持）。詳細は
//! `gemm::MetalGemm::dispatch_f16_unverified`・
//! `gemm::MetalGemm::dispatch_f16_auto_unverified` のドキュメントコメントを
//! 参照）。
//!
//! イシュー #604 で融合 RMSNorm 順伝播カーネル（`rmsnorm::MetalRmsNorm`）と
//! online softmax カーネル（`softmax::MetalSoftmax`）を MSL で追加した。
//! CUDA 側 G-6（#592）と同一アルゴリズム契約（1 パス／2 パス経路・
//! persistent threadgroup・FMA 契約統一）を採るが、`MetalContext::
//! dispatch_sync` が動的 threadgroup memory 設定 API を経由しないため
//! 1 パス経路はコンパイル時固定長の `threadgroup` 配列を使う（`row_kernel`
//! モジュール冒頭コメント参照）。両カーネル・`row_kernel` の経路選択・
//! canonical 融合プラン照合の cfg 非依存部分は `row_kernel` に集約し、
//! `ops::MetalBackendOps::run_fused` からルーティングする。softmax の
//! CUDA 直接 parity 相手（#594・G-7）は本イシュー時点で未実装のため、
//! 両バックエンドとも CPU 参照実装（REQ-2 統一複合判定）に対する数値一致を
//! 経由した推移的な担保に留まる（`softmax.rs`／`tests/softmax_parity.rs`
//! ドキュメンテーションコメント参照）。
//!
//! イシュー #605（Phase G・G-14）で elementwise 5 演算
//! （`elementwise::MetalElementwise`。CUDA 側 #599 の対応版）を追加し、
//! `ops::MetalBackendOps::gemm_bias_act` を GEMM epilogue 実融合カーネル
//! （`gemm::MetalGemm::run_tiled_bias_act_f32`・`shaders/gemm.metal::
//! gemm_tiled_bias_act`）でオーバーライドした。既存 `gemm_naive`／
//! `gemm_tiled`／`gemm_simdgroup`／`gemm_simdgroup_tiled`／
//! `gemm_simdgroup_f16` カーネルには一切触れておらず、GEMM 単体の性能・
//! 数値契約は構造的に非後退（新規カーネルの追加のみ）。
//!
//! イシュー #796 で f16 タイル化カーネル本体（`shaders/gemm.metal::
//! gemm_simdgroup_tiled_f16`。BM/BN/BK/WM/WN・協調ロード・direct-load
//! 分岐・f32 アキュムレータ・2 段エピローグ）を追加し、イシュー #797 で
//! 協調ロードの 8 要素（128bit）ベクトル化・エピローグの barrier 粒度統合
//! （8x8 タイル毎 → サブタイル全体単位へ集約）を行った。
//! 既存 `gemm_simdgroup_f16`（1 threadgroup = 1 simdgroup = C の 8x8
//! タイル 1 つの非タイル化構造）が対 PyTorch MPS f16 で大きく劣後する
//! 主因（親イシュー #787）への対応で、f32 版 `gemm_simdgroup_tiled`
//! （TASK-1.8f・#188）の構造を half 入力対応で移植した。明示 `TileConfig`
//! 指定の単体ディスパッチ入口（`gemm::MetalGemm::dispatch_f16_tiled_unverified`・
//! `gemm::MetalGemm::dispatch_f16_tiled_prepared_unverified`）を追加した
//! （`_unverified` suffix・`#[doc(hidden)]` の判断根拠は
//! `gemm::MetalGemm::dispatch_f16_unverified` と同一）。協調ロードの
//! float4 ベクトル化・エピローグの barrier 粒度最適化は #797、実機
//! 再計測・ベースライン更新は #799 のスコープとする。
//!
//! イシュー #798 で動的タイル選択の自動入口
//! （`gemm::MetalGemm::dispatch_f16_auto_unverified`）を追加し、`gemm_simdgroup_tiled_f16`
//! を `tile::select(m, n, k)` による f32 版 `dispatch_auto` と同型のタイル
//! 構成選択で使う経路を用意した。`GemmVariant` enum へは
//! f16 を統合しない（`dispatch_variant` が `&[f32]` に閉じる既存設計判断を
//! 維持。`gemm::MetalGemm::pipeline_simdgroup_f16` フィールドコメント参照）。
//! 非タイル `gemm_simdgroup_f16`・`dispatch_f16_unverified` 系入口は
//! 削除・置換せず、計測・回帰基線として存置する後方互換方針を採った
//! （`gemm::MetalGemm::dispatch_f16_auto_unverified` ドキュメンテーションコメント
//! 参照）。数値一致回帰は `tests/gemm_f16_auto_parity.rs`（Metal 実機依存・
//! `#[ignore]`）で REQ-2 統一複合判定を検証する契約だが、本 PR 時点で実機
//! 実行は未完了（#799 のスコープ）。**PR #819 codex-review P1 指摘対応**として、
//! 精度未検証カーネルを検証済み production 入口へ結線しない既存の安全境界
//! に従い、`gemm::MetalGemm::dispatch_f16_auto_unverified` 自体も
//! `_unverified` suffix・`#[doc(hidden)]` とし、`ops::MetalBackendOps` や
//! `dispatch_backend_auto`（真の production 自動経路）へは統合していない
//! （#799 の実機検証完了後、別イシューで統合可否を判断する）。`tensor-core`
//! 決定表（`select_gemm_kernel`）・`dispatch_backend_auto` の f16 拡張は
//! 本イシューのスコープ外のまま残す。
//!
//! イシュー #930 で `context_cache` モジュールを追加し、`ops::MetalBackendOps`
//! が演算メソッド呼び出しごとに都度構築していた `MetalContext`／
//! `MetalGemm`／`MetalElementwise`／`MetalRmsNorm`／`MetalSoftmax` を
//! プロセス内キャッシュへ常駐化した（診断 #927 が特定した約 5 ms・N 非依存の
//! 固定オーバーヘッドの解消。CUDA 側 #929 と同型設計）。`gemm::MetalGemm`
//! の tile パイプラインキャッシュ（`tiled_cache`／`tiled_f16_cache`）は
//! `Arc` 経由の複数スレッド共有に対応するため `RefCell` から `Mutex` へ
//! 変更した。カーネルソース・ディスパッチロジック・許容誤差・境界検査は
//! 一切変更していない。
//!
//! イシュー #1017（親 #1015・設計 #1016・`docs/backend-metal-command-
//! batching-design.md`）で `MetalContext` にコマンドバッファ共有バッチ
//! （`context::MetalContext::encode`／`context::MetalContext::synchronize`）を
//! 追加した。既存の
//! `context::MetalContext::dispatch_sync`（`encode` + 即時
//! `synchronize` の薄いラッパーへ変更。シグネチャ・戻り値の意味は不変）
//! を経由する既存呼び出し元（`gemm.rs`／`elementwise.rs`／`rmsnorm.rs`／
//! `softmax.rs`）は無変更のまま後方互換を保つ。`sgd.rs::MetalSgd::run`
//! のみ `encode` へ直接切り替え、`ops::MetalBackendOps::
//! sgd_step_device_tracked`（`tensor-core::BackendOps` の非破壊拡張）
//! から [`fandhe_ai_tensor_core::DispatchFailureCell`] を encode と
//! 同一ロック区間で登録することで、`DeviceParamStore::step` の毎ステップ
//! 呼び出しが個別のコマンドバッファ生成・`waitUntilCompleted` を支払わず
//! 済むようにした（診断: `docs/perf/device-resident-update-bench.md`。
//! ホスト実体化（`memory.rs::download_inner`／`zero_fill`）が唯一の
//! 同期点となる）。`batch_state` モジュールが cfg 非依存の純粋ロジック
//! （ラベル記録・自動 flush 上限・失敗伝播）を担う。

// `context.rs::MetalContext::encode`／`flush`／`synchronize`
// （イシュー #1017）のうち `objc2-metal` の型に触れない部分（ラベル
// 記録・自動 flush 判定・失敗伝播・診断メッセージ整形）を切り出した
// モジュール。`generic_cache`／`row_kernel`／`pad`／`tile` と同じ設計
// 判断で `cfg(target_os = "macos")` を付けず、Linux（本実装環境・CI）
// でも単体テストが回るようにする。
pub(crate) mod batch_state;
// `context.rs::MetalContext::synchronize`／`pool.rs::PooledMetalHandle::
// Drop`（イシュー #1021）が「保留中のプール返却列」へ push・合流する
// 判定ロジック（`objc2` 系 FFI に触れない）を切り出したモジュール。
// `batch_state`／`generic_cache` と同じ設計判断で `cfg(target_os =
// "macos")` を付けず、Linux（本実装環境・CI）でも単体テストが回る
// ようにする（`pool_pending.rs` モジュール冒頭コメント参照）。
#[cfg(target_os = "macos")]
pub mod buffer;
#[cfg(target_os = "macos")]
pub mod context;
pub(crate) mod pool_pending;
// `ops::MetalBackendOps` からのみ参照する内部モジュール（CUDA 側
// `context_cache`〈feat/929-cuda-ctx-cache〉と同じ可視性方針）。
// `MetalContext`／`MetalGemm` 等の公開型はここを経由せず既存の `pub use`
// でも到達可能なため、本モジュール自体は非公開のままでよい。
#[cfg(target_os = "macos")]
pub(crate) mod context_cache;
#[cfg(target_os = "macos")]
pub mod device;
#[cfg(target_os = "macos")]
pub mod elementwise;
#[cfg(target_os = "macos")]
pub mod error;
// `context_cache` のコア判定ロジック（ヒット判定・ビルド・poison 変換）の
// 切り出し（イシュー #930 codex-review 対応）。`pad`／`tile`／`row_kernel`
// と同じ設計判断で `objc2` 系 FFI に触れないため `cfg(target_os = "macos")`
// を付けず、Linux CI でも汎用キャッシュ契約（ヒットの clone・ビルド失敗の
// 非キャッシュ・poison 時 fail-closed）を検証できるようにしてある。
#[cfg(target_os = "macos")]
pub mod gemm;
pub(crate) mod generic_cache;
#[cfg(target_os = "macos")]
pub mod half_buffer;
#[cfg(target_os = "macos")]
pub mod memory;
#[cfg(target_os = "macos")]
pub mod ops;
pub mod pad;
// `crate::buffer::MetalBuffer::alloc_zeroed_pooled`／`alloc_uninit_pooled`
// からのみ到達する `pub(crate)` 面（イシュー #1021）。`tensor-core` の
// どの公開 trait にも属さない低水準アロケータ実装のため非公開のまま。
#[cfg(target_os = "macos")]
pub mod pipeline;
#[cfg(target_os = "macos")]
pub(crate) mod pool;
#[cfg(target_os = "macos")]
pub mod rmsnorm;
#[cfg(target_os = "macos")]
pub(crate) mod sgd;
// `pad`／`tile` と同じ設計判断: `objc2` 系 FFI に触れないため
// `cfg(target_os = "macos")` を付けず Linux でも単体テストが回る。
// ただし `pad`／`tile` と異なり `row_kernel` は経路選択・occupancy 定数・
// 起動検証エラー・canonical FusionPlan 照合などバックエンド内部実装の
// 密度が高いため、`pub`（クレート外部から `fandhe_ai_backend_metal::row_kernel::*`
// として到達可能）にはせず `pub(crate)` を維持する（codex-review P1
// 指摘・PR #714）。実際の呼び出し元（`ops.rs`／`rmsnorm.rs`／
// `softmax.rs`）は macOS 限定のため、Linux 単体ビルド（`cargo build`／
// `cargo clippy` の非テストパス）では `row_kernel` の各項目が
// 「クレート内から到達不能」と判定され dead_code lint が誤検知する。
// これは `pub` へ広げず、`row_kernel.rs` モジュール冒頭の
// `#![cfg_attr(not(target_os = "macos"), allow(dead_code))]`
// （対象を non-macOS ビルドに限定した allow）で個別に抑制する。
//
// 例外: `row_kernel::SOFTMAX_NEG_FLT_MAX`（テスト専用の数値特性ロック値。
// 本番経路〈`ops.rs`／`softmax.rs`〉からは参照されない）は上記の
// `cfg_attr` では救えない（macOS ビルドでも `#[cfg(test)] mod tests` の
// 外からは到達不能なため dead_code になる）。この 1 項目のみ
// `#[cfg(test)]` を個別付与している（macOS/aarch64 ローカル clippy 実測。
// PR「fix(backend): macOS/aarch64 ローカル clippy エラーを解消」）。
pub(crate) mod row_kernel;
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
pub use elementwise::MetalElementwise;
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
