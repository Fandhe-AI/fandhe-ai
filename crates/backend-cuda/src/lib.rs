//! CUDA バックエンド。
//!
//! `tensor-core` の演算グラフノードを NVRTC 経由でコンパイルした CUDA カーネルへ変換して
//! 実行する。バックエンド切替は feature フラグなしの cfg ベース（PoC-v2-5 実証構成。REQ-2）で、
//! 依存する `cudarc` は無条件依存かつ動的ロード方式を用いるため、CUDA toolkit 非搭載環境でも
//! ビルド自体は成立する（実行時のみ toolkit を要求。`.claude/rules/deps-policy.md`）。
//!
//! `backend-cpu` との数値一致は統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」で
//! 検証する。丸め方針（FMA 契約）は NVRTC の既定 FMA 契約を CPU 参照実装（`f32::mul_add`）と
//! 揃える（PoC-v2-5 の K=4096 ストレスケースで実測確認済み。`.claude/rules/coding-rust.md`）。
//! カーネルの手動境界検査は最適化を理由に省略しない（REQ-8）。
//! FFI 境界の `unsafe` は必要最小限に留め理由コメントを付す（`.claude/rules/security.md`）。
//!
//! TASK-1.7a（#32）で、動的ロード・デバイス初期化・NVRTC コンパイルの基盤
//! （`CudaDevice`・`CudaError`・`compile_ptx`）を追加した。`cudarc` の
//! `dynamic-loading` feature は `libcuda`/`libnvrtc` が `dlopen` できない
//! 環境で driver/nvrtc API を直接呼ぶと `Err` ではなく panic するため、
//! 本クレートの初期化入口（`CudaDevice::new`・`CudaDevice::device_count`・
//! `compile_ptx`）は `is_culib_present()` による非 panic プローブで
//! 必ずゲートしてから型付きエラー（`CudaError::DriverUnavailable`／
//! `NvrtcUnavailable`）を返す（`device.rs`／`nvrtc.rs` のドキュメンテーション
//! コメント参照）。これにより CUDA 非搭載環境でも panic しない。
//!
//! TASK-1.9a（#44）で `device` モジュールに [`device::CudaDeviceProvider`]
//! （`tensor_core::device::DeviceProvider` の CUDA 実装）を追加した。上記の
//! `CudaDevice` を内部で経由するため panic 回避ゲートは共通で効く。CPU／Metal
//! 実装（`backend-cpu::CpuDeviceProvider`／`backend-metal::device::MetalDeviceProvider`）
//! と同一 trait で列挙・選択できることを
//! `backend-cpu/tests/device_provider_integration.rs` で検証する。CUDA
//! ドライバ非搭載環境では `is_available() == false`・`enumerate() == Ok(vec![])`
//! を返す（fail-safe。`device.rs` 内コメント参照）。
//!
//! カーネルソース・起動 API は naive 版（#33）・tiled 版（#34。共有メモリ
//! タイリング `TILE=32`）を追加済み。CUDA toolkit 非搭載ビルドの CI 検証は
//! `.github/workflows/ci.yml` の `build-no-cuda-toolkit` ジョブと
//! `scripts/check-cuda-toolkit-absent.sh`（TASK-1.7d・#35）で実装済み。
//! 実機（DGX Spark GB10 等）依存テストの `#[ignore]` 分離は #36 で
//! 完了した（実機での実行導線は `make test-ignored-cuda`。`Makefile`・
//! `README.md` 参照）。f16 向け許容誤差の設計・採用（実質的な許容誤差
//! 変更でありユーザー承認必須）は #36 のスコープ外として未着手のまま
//! 残す（`tests/cpu_cuda_parity.rs` 冒頭コメント参照）。
//! TASK-1.9b（#45）で [`memory`] モジュール（[`memory::CudaMemory`]）を
//! 追加した。`tensor_core::buffer::MemoryOps` の CUDA 実装であり、
//! `CudaDevice` 経由でのみ構築できるため上記の panic 回避ゲートを共有
//! する。既存の `gemm.rs`（`clone_htod`/`alloc_zeros`/`clone_dtoh`）は
//! 演算内部にホスト⇔デバイス転送を抱えたままとし、本イシューでは
//! 載せ替えを行わない（TASK-1.9c・#46 のスコープ）。`BackendOps`
//! トレイト自体（カーネルディスパッチ）へのフルマッピングも
//! TASK-1.9c（#46）のスコープであり、本クレートではまだ扱わない
//! （spec 根拠: `docs/spec/05-tasks.md` TASK-1.7・TASK-1.9）。
//!
//! TASK-11.1b（#61）で f16 Tensor Core（WMMA）GEMM カーネル
//! （[`CudaWmmaGemm`]）を追加した。設計は `docs/cuda-tensor-core-design.md`
//! （#60）で確定済み（方式 A: `#include <mma.h>` の WMMA C++ API・
//! `m16n16k16` fragment・f32 アキュムレート）。naive／tiled 経路
//! （`kernels.rs`／`gemm.rs`）とは別ファイル（`kernels_wmma.rs`／
//! `gemm_wmma.rs`）に分離している。
//!
//! TASK-11.1c（#62）で WMMA（Tensor Core）を用いた TF32/f32 GEMM
//! （[`CudaGemm::run_wmma_tf32`]）を追加した。設計は `docs/cuda-tensor-core-design.md`
//! （#60）を正本とし、fragment `m16n16k8`（TF32 精度・f32 累算）・方式 A
//! （WMMA C++ API `<mma.h>`）を採用する（REQ-11）。TF32 は f32 の仮数部
//! 23bit を 10bit に丸めて Tensor Core へ投入するため、統一複合判定
//! （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）は TF32 前提の複合指標
//! として適用する（REQ-2、`.claude/rules/coding-rust.md`）。f16 WMMA 経路
//! （#61）とは異なり、TF32 経路は naive／tiled 経路と同じ `kernels.rs`／
//! `gemm.rs` に実装している。
//! TASK-11.1d（#63）で WMMA(TF32)／f16 WMMA の共有メモリ・タイル最適化版
//! カーネル（`kernels_wmma_opt::WMMA_TF32_F32_OPT`／`WMMA_F16_OPT`）を追加
//! した。ブロックタイル 64×64・warp あたり fragment 2×2 個（レジスタ
//! ブロッキング）・バンクコンフリクト回避パディング・`__syncthreads()`
//! ベースのダブルバッファリングを適用する（設計は `docs/cuda-tensor-core-
//! design.md` 4.2 節を正本とする）。公開 API（`CudaGemm::run_wmma_tf32`／
//! `CudaWmmaGemm::run_f16`）のシグネチャは変更せず、opt カーネルが
//! `new` 時点でコンパイル・ロードに成功していれば優先的に使用し、失敗
//! していれば #61/#62 の基本 WMMA カーネルへ自動フォールバックする
//! （`kernels_wmma_opt.rs` 冒頭ドキュメントコメント参照）。実機実測での
//! 数値一致・性能確定は #64 のスコープ。
//!
//! TASK-11.1h（#187）で `mma.sync`/`ldmatrix`/`cp.async` PTX 直叩き経路
//! （[`CudaMmaGemm`]）を追加した。WMMA 経路（cc>=7.0）より厳しい
//! compute capability 8.0+ ゲートを持つ独立経路であり、`kernels_mma.rs`／
//! `gemm_mma.rs` に分離している（並行 issue #62/#63 が `gemm.rs`／
//! `gemm_wmma.rs`／`kernels.rs`／`kernels_wmma.rs` を編集中のため）。
//! XOR swizzle によるバンクコンフリクト低減は未実装（`kernels_mma.rs`
//! 冒頭コメント「タイル構成」参照。コンパイル未検証環境でのリスク
//! 最小化判断）。
//!
//! ディスパッチ規則（naive／tiled／f16 WMMA／TF32 WMMA／`mma.sync` の
//! どの経路をいつ選ぶか）は TASK-11.2（#66）のスコープであり本クレートでは
//! 未実装。
//!
//! TASK-11.2b（#68）で GEMM 自動経路選択の入口（[`CudaGemmAuto`]）を
//! 追加した。`tensor_core::dispatch::select_gemm_kernel`（#67 が設計した
//! 決定的規則。`docs/dispatch-rules-design.md`）の結果に従い、naive／
//! tiled（`CudaGemm`）・WMMA f16（`CudaWmmaGemm`）を呼び分ける。TF32/f32
//! 経路（`CudaGemm::run_wmma_tf32`・#62）・`mma.sync` 経路（`CudaMmaGemm`・
//! #187）は、決定表（設計文書 §4）が TF32 既定採用を #186（TASK-11.1g）の
//! ユーザー承認まで保留と定めているため、現時点の `select_gemm_kernel` の
//! 自動経路には含めない（f32 は常に Tiled）。既存の `CudaGemm`／
//! `CudaWmmaGemm`／`CudaMmaGemm` の直接指定 API はテスト・証跡用途
//! （#70）にそのまま温存する（設計文書 §5.4）。
//!
//! TASK-1.9c（#46）で [`ops`] モジュール（[`ops::CudaBackendOps`]）を追加した。
//! `tensor_core::backend_ops::BackendOps` の CUDA 実装であり、`gemm` は
//! [`CudaGemm::run_tiled_f32`] へ委譲する（既定カーネル変種の選択は保守的に
//! tiled 固定とし、`CudaGemmAuto` を介した Tensor Core 経路の自動選択への
//! 切替は別スコープ）。イシュー #599 で elementwise 5 演算（`add`／`mul`／
//! `relu`／`exp`／`tanh`。[`elementwise::CudaElementwise`]）を実装し、
//! `gemm_bias_act` を GEMM epilogue 融合カーネル
//! （[`CudaGemm::run_tiled_bias_act_f32`]）で実融合化した（`bias`
//! ブロードキャスト形状の非厳密一致ケースは非融合合成へフォールバックする。
//! `ops::gemm_bias_act_route` 参照）。reduction（`sum`／`max`）は GPU
//! カーネル未実装のまま `tensor_core::device::BackendError::Unsupported`
//! を返す（out-of-scope-tracking.md 対象）。
//!
//! イシュー #499（GEMM 性能改善ツリー #479 の後続）で L2 再利用のための
//! タイル→SM 割り当てスウィズル（[`swizzle`]・`kernels_mma::
//! mma_f16_source_with_swizzle`）を opt-in・`internal-diagnostics`
//! feature（既定 off）ゲート経路として追加した（#497 と同型の判断。本
//! セッション実行環境〈RTX 3060・NVRTC 非搭載〉では実機 A/B 計測が
//! できなかったため）。
//!
//! イシュー #740 で GB10 実機 A/B 計測（4096: ×1.5957・group_width=8、
//! 512〜2048 は 0.97〜1.00 倍とほぼ中立。`docs/perf/cuda-gemm-swizzle-ab.md`
//! §6 参照）を根拠に一時的に `gemm_mma::CudaMmaGemm::new`（本番既定
//! コンストラクタ）へ本番結線したが、PR #758 レビュー指摘（採用基準
//! 〈2048/4096 両方の改善〉未達のまま代替基準へ読み替えていたこと・
//! 結線前必須確認〈spill／bit 一致／parity〉未実施・CI 恒久検査が
//! GB10 未実測の SM 数を実測値と誤扱いしていたこと）により差し戻した
//! （`docs/perf/cuda-gemm-swizzle-ab.md` §2 参照）。`new` は現在
//! swizzle 無適用の base カーネル（`kernels_mma::mma_f16_source()`）を
//! 返す。swizzle 適用版は引き続き `new_with_swizzle`（`internal-
//! diagnostics` feature 限定）から opt-in で利用できる。§4 の採用基準を
//! 字義どおり満たすか、人間承認を伴って基準を正式改訂したうえで
//! 再結線を検討する。
//!
//! Phase C-1（#504。親イシュー #503「CUDA JIT shape 特化・コンパイル
//! キャッシュ・静的タイル選定」の先頭タスク）で [`CudaKernelDescriptor`]・
//! [`CudaKernelCacheKey`]・[`nvrtc_version`] を追加した。カーネル特化
//! パラメータ（shape・ブロックタイル・パイプライン段数・dtype）とコンパイル
//! キャッシュのキー（上記 + compute capability・NVRTC バージョン・
//! コンパイルフラグ + 最終レンダー済みカーネルソース）を表す `Hash + Eq` な
//! 型であり、後続タスク（C-2 自作ハッシュ・ディレクトリ命名 #506、C-4
//! プロセス内 LRU #511、C-6 テンプレート展開 #516）が共通に使う「キーの
//! 単位」を確定する。ソース断片の取り込みによるキャッシュ無効化は C-5
//! （#514）で実装済み（`nvrtc.rs::CudaKernelCacheKey` ドキュメンテーション
//! コメント参照。`new`／`from_device` への `source` 必須引数追加は意図
//! した破壊的変更であり、移行契約は同コメントの「破壊的変更の意図的な
//! 受容」節を参照）。本タスクではキャッシュ本体・ディレクトリ命名・
//! テンプレート展開は実装しない（`nvrtc.rs` ドキュメンテーションコメント
//! 参照）。
//!
//! イシュー #592 で融合 RMSNorm 順伝播カーネル（[`rmsnorm::CudaRmsNorm`]）を
//! 追加した。TileKernels engram gate カーネルの構造イディオム（1 CTA =
//! 1 warp・`__syncthreads()` 不使用・persistent block）を転用し、
//! `out = x * rsqrt(sum(x^2) * inv_n + eps) * w` を 1 カーネルで完結させる
//! （中間テンソルの HBM 非書き出し。`kernels_rmsnorm.rs` 冒頭ドキュメント
//! コメント参照）。行長が SMEM 予算に収まる場合は動的共有メモリ常駐の
//! 1 パス経路、収まらない場合は global 再読の 2 パス経路（[`rmsnorm::
//! rmsnorm_route`]。予算は `gemm_auto::read_clamped_smem_budget_bytes` と
//! 同一のクランプ済み値を単一の真実源として共有する）を選ぶ。persistent
//! block 数は sm_121 の SMEM/SM 上限を実行時属性取得から導出する
//! （Hopper 固定値を流用しない。`docs/perf/sm121-device-attributes.md`
//! C-8 注記と同方針）。`CudaBackendOps::run_fused`（`ops.rs`）は canonical
//! RMSNorm 融合プラン（`x * rsqrt(sum(x^2))`。mean 化・eps・weight を含ま
//! ない厳密形状。`rmsnorm::match_rmsnorm_plan`）検出時のみ本カーネルへ
//! ルーティングし、一致しないプランはデフォルトの `Unsupported`
//! フォールバックのまま維持する。
//!
//! イシュー #594 で online softmax（FlashAttention-2 型）順伝播カーネル
//! （[`softmax::CudaSoftmax`]）を追加した。RMSNorm と同じ構造イディオム
//! （1 CTA = 1 warp・persistent block・1 パス／2 パスの 2 経路）を転用し、
//! `log2(e)` 事前スケール + `exp2f`（`expf` 不使用）・オンライン最大値
//! 更新の補正係数スキップ・有限マージンの境界マスク（`-INFINITY` 不
//! 使用）で数値安定性を確保する（`kernels_softmax.rs` 冒頭コメント参照）。
//! `CudaBackendOps::run_fused`（`ops.rs`）は canonical softmax 融合プラン
//! （`exp(x-max(x))/sum(...)`。最終軸または全軸縮約の厳密形状。
//! `softmax::match_softmax_plan`）検出時のみ本カーネルへルーティングする。
//!
//! Phase C-4（#511。親イシュー #503 の最終タスク）で [`module_cache`]
//! （非公開 `mod`。`pub use` で再公開しない内部実装詳細）を追加し、
//! `kernels_mma.rs::RenderedMmaKernel::compile` をプロセス内 LRU
//! （ロード済みモジュールハンドル再利用）→ ディスクキャッシュ
//! （`nvrtc.rs::load_cache_entry`／`store_cache_entry`。C-3・#509）→
//! NVRTC 直コンパイルの 3 段フォールバックへ結線した。これにより
//! `gemm_auto.rs::SpecializedMmaKernelHandle::compile`（従来は呼び出し
//! ごとに NVRTC コンパイルしていた shape 特化経路）が自動的に再利用化
//! される。容量は環境変数 `RUST_AI_CUDA_MODULE_CACHE_CAPACITY`
//! （既定 32・上限 1024）で調整可能（`module_cache.rs` ドキュメンテー
//! ションコメント参照）。ディスクキャッシュ関連の失敗（`workspace_root`
//! 解決不能・fs I/O 失敗）はコンパイル失敗にせず「ディスクキャッシュ
//! なしの縮退運転」へフォールバックする fail-safe 方針を採る
//! （`kernels_mma.rs::RenderedMmaKernel::compile` ドキュメンテーション
//! コメント参照）。固定ソースの一回コンパイル経路（`CudaGemm::new`・
//! `CudaWmmaGemm::new`・`CudaMmaGemm::new`・elementwise/transpose 群）は
//! インスタンス構築時 1 回のみのコンパイルであり本タスクでは結線しない
//! （拡大は効果に対しリスク過大と判断。実装計画 §3.4 スコープ境界）。

pub mod device;
mod elementwise;
mod error;
mod gemm;
mod gemm_auto;
mod gemm_mma;
mod gemm_wmma;
mod kernels;
mod kernels_elementwise;
mod kernels_mma;
mod kernels_rmsnorm;
mod kernels_softmax;
mod kernels_transpose;
mod kernels_wmma;
mod kernels_wmma_opt;
pub mod memory;
mod module_cache;
mod nvrtc;
mod ops;
mod rmsnorm;
mod softmax;
mod swizzle;
mod transpose;

pub use device::{CudaDevice, CudaDeviceProvider};
pub use elementwise::CudaElementwise;
pub use error::CudaError;
pub use gemm::CudaGemm;
pub use gemm_auto::{
    CostModelParams, CudaGemmAuto, MeasuredBandwidth, SM121_MEASURED_BANDWIDTH, TileCandidate,
    TileSelection, TileSelectionBasis, derive_stages_for_device, enumerate_tile_candidates,
    enumerate_tile_candidates_for_device, select_tile_config, select_tile_config_for_device,
};
// `SpecializedMmaKernelHandle`／`run_specialized_mma_f16` はテスト・ベンチ専用の
// 検証用ハンドル（`gemm_auto.rs` 冒頭ドキュメンテーションコメント参照。本番
// ディスパッチ経路〈`CudaGemmAuto::run_f16`〉からは呼ばれない）。PR #685
// codex-review P1 指摘の是正: 従来は上記ブロックへ無条件 re-export しており、
// コメント上「テスト・ベンチ専用」の意図に反して通常ビルドの安定した公開
// API 面へ漏出していた。`diagnostics` モジュール（本ファイル下部）と同じ
// `internal-diagnostics` feature（既定 off）でゲートし、`tests/
// specialized_mma_parity.rs` は `Cargo.toml` の `[[test]]` セクションで
// `required-features = ["internal-diagnostics"]` を指定して到達する
// （`cargo test --all-features` でのみビルド・実行される）。
#[cfg(feature = "internal-diagnostics")]
pub use gemm_auto::{SpecializedMmaKernelHandle, run_specialized_mma_f16};
pub use gemm_mma::CudaMmaGemm;
pub use gemm_wmma::CudaWmmaGemm;
pub use memory::CudaMemory;
pub use nvrtc::{
    CompiledDims, CudaKernelCacheKey, CudaKernelDescriptor, MAX_PIPELINE_STAGES, compile_ptx,
    derive_pipeline_stages, nvrtc_version,
};
pub use ops::CudaBackendOps;
pub use rmsnorm::{CudaRmsNorm, RmsNormShape};
pub use softmax::CudaSoftmax;
pub use transpose::CudaTranspose;

/// `kernels_mma`／`kernels_wmma_opt`（非公開 `mod`。カーネル本体は crate
/// 外から直接呼ばせない）が持つブロックタイル定数を、診断専用の安定関数
/// として公開する境界。イシュー #486 の `examples/gemm_profile_target.rs`
/// occupancy 概算がこのタイル値を必要とするが、値を手元転記すると出典側の
/// 変更を機械的に検知できない（値が乖離しても診断ツールが静かに誤った
/// 参考値を出し続ける）ため、`kernels_mma::MMA_BM`／`_BN`・
/// `kernels_wmma_opt::WMMA_TF32_OPT_BLOCK_M`／`_N` を crate 内部でのみ
/// `use` し、値そのものを返す関数だけを公開する。
///
/// **`internal-diagnostics` feature（既定 off）でのみコンパイルされる**
/// （`Cargo.toml` の `[features]` 参照。PR #637 codex-review P1 指摘の是正:
/// 生の内部定数はおろか、この安定関数群自体も非公開カーネルのタイル形状を
/// crate 外へ伝える契約になってしまうため、`pub mod` として常時公開せず
/// feature ゲートで既定ビルドの公開 API 面から完全に除外する。コメントで
/// 「SemVer 互換性保証対象外」と宣言するだけでは Rust の通常の公開 API で
/// ある以上、戻り値の意味・関数自体が利用者との契約になってしまうため
/// 不十分と判断した）。`examples/gemm_profile_target.rs`
/// （occupancy 概算専用）は `Cargo.toml` の `required-features` で本
/// feature を要求するため、`cargo build --example gemm_profile_target
/// --features internal-diagnostics` でのみビルドできる。通常の利用者は
/// [`CudaGemm`]／[`CudaMmaGemm`]／[`ops::CudaBackendOps`] 等の安定 API を
/// 経由してバックエンドを利用し、本 feature を有効化する必要はない。
#[cfg(feature = "internal-diagnostics")]
pub mod diagnostics {
    use crate::{kernels_mma, kernels_wmma_opt, swizzle};

    // イシュー #742: TF32 opt-staged 段数スイープ example
    // （`examples/gemm_wmma_tf32_staged_stages_bench.rs`）専用の再公開。
    // `kernels_wmma_opt` は非公開 `mod` のため、本モジュール（
    // `internal-diagnostics` feature 配下）を経由しないと crate 外部から
    // 到達できない（上記関数群と同じ「非公開モジュールへの薄い診断用
    // ラッパー」方針）。本番経路（`gemm.rs` の 3 段フォールバック選択・
    // `CudaGemm::run_wmma_tf32`）はこの再公開に一切依存しない。
    //
    // イシュー #743 追補（PR #769 Bugbot 指摘 review id 4978031442 の
    // 是正）: `render_wmma_tf32_staged`／`RenderedWmmaTf32StagedKernel`／
    // `CompiledWmmaTf32StagedKernel`（**static** 共有メモリ変種。本番経路
    // と同一の `__shared__` 宣言・同一 occupancy）も併せて再公開する。
    // `examples/gemm_profile_target.rs` の `--b-pad` 計測が動的共有メモリ
    // 変種（`render_wmma_tf32_staged_dyn`。`c_tile` を `as_tile`/`bs_tile`
    // へエイリアスし約 29KiB・3 blocks/SM）だけを使って本番の静的変種
    // （44.8〜45.6KiB・2 blocks/SM）と比較していたため、`b_pad` の効果と
    // dyn/static の occupancy 差が交絡していた（ncu 実測がどちらの要因か
    // 切り分けられない）。static 変種を config 経由で `b_pad` を変えて
    // 起動できるようにし、本番と同一レイアウトのまま `b_pad` のみを
    // 変数化して切り分ける。
    pub use kernels_wmma_opt::{
        CompiledWmmaTf32StagedDynKernel, CompiledWmmaTf32StagedKernel,
        RenderedWmmaTf32StagedDynKernel, RenderedWmmaTf32StagedKernel, WmmaTf32StagedKernelConfig,
        render_wmma_tf32_staged, render_wmma_tf32_staged_dyn, wmma_tf32_staged_dyn_smem_bytes,
    };

    /// `wmma_tf32`（WMMA(TF32) opt）カーネルのブロックタイル形状
    /// `(block_m, block_n)`。`examples/gemm_profile_target.rs` の
    /// occupancy 概算専用。
    pub fn wmma_tf32_opt_block_tile() -> (u32, u32) {
        (
            kernels_wmma_opt::WMMA_TF32_OPT_BLOCK_M,
            kernels_wmma_opt::WMMA_TF32_OPT_BLOCK_N,
        )
    }

    /// `mma_f16`（`mma.sync` f16 パイプライン）カーネルのブロックタイル
    /// 形状 `(block_m, block_n)`。`examples/gemm_profile_target.rs` の
    /// occupancy 概算専用。
    pub fn mma_f16_block_tile() -> (u32, u32) {
        (kernels_mma::MMA_BM, kernels_mma::MMA_BN)
    }

    /// イシュー #499: L2 再利用のためのタイル→SM 割り当てスウィズルの
    /// グルーピング幅動的選択（`swizzle::select_swizzle_group_width`）を
    /// `mma_f16` のブロックタイル（`MMA_BM`/`MMA_BN`）に対して適用した
    /// 結果を返す。`swizzle` は非公開 `mod`（`lib.rs` の `mod swizzle;`）
    /// のため、crate 外部（`examples/gemm_mma_swizzle_bench.rs`）から
    /// 到達するにはこの diagnostics 経由の薄いラッパーが必要
    /// （`mma_f16_block_tile`・`wmma_tf32_opt_block_tile` と同じ理由・
    /// 同じ feature ゲート方針）。`gemm_mma::CudaMmaGemm::new`（本番既定
    /// コンストラクタ）はイシュー #740 で一時この式を直接呼ぶよう結線
    /// されたが、PR #758 レビュー指摘により差し戻し済み（`new` は現在
    /// swizzle 無適用の base カーネルを返す。`gemm_mma.rs::CudaMmaGemm::
    /// new` ドキュメンテーションコメント参照）。本関数は A/B 計測
    /// （`examples/gemm_mma_swizzle_bench.rs`）専用の診断用ラッパーで
    /// あり続ける。
    pub fn mma_swizzle_group_width(num_sms: u32) -> u32 {
        swizzle::select_swizzle_group_width(num_sms, kernels_mma::MMA_BM, kernels_mma::MMA_BN)
    }

    /// イシュー #741: [`mma_swizzle_group_width`] の TF32 opt-staged 版。
    /// `swizzle::select_swizzle_group_width` を TF32 opt-staged の
    /// ブロックタイル（`WMMA_TF32_STAGED_BLOCK_M`/`_N`。64×64）に対して
    /// 適用する。`mma_f16` のブロックタイル（64×128）と異なるため専用
    /// ラッパーが必要（`swizzle.rs` 本体は無変更。#740 とのコンフリクト
    /// 回避）。`examples/gemm_wmma_tf32_swizzle_bench.rs` から到達する。
    pub fn wmma_tf32_staged_swizzle_group_width(num_sms: u32) -> u32 {
        swizzle::select_swizzle_group_width(
            num_sms,
            kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_M,
            kernels_wmma_opt::WMMA_TF32_STAGED_BLOCK_N,
        )
    }

    /// プロセス内 LRU カーネルモジュールキャッシュ（イシュー #511・C-4。
    /// `crate::module_cache`。非公開 `mod` のため crate 外部から直接
    /// 到達できない）のヒット件数。`crate::module_cache::
    /// KernelModuleCache::global` の初期化自体が失敗した場合（不正な
    /// `RUST_AI_CUDA_MODULE_CACHE_CAPACITY`）は `None` を返す。
    ///
    /// `tests/specialized_mma_parity.rs`（`#[ignore]` 実機テスト）が
    /// 「同一形状・同一 `CompiledDims` での 2 回目以降の
    /// `SpecializedMmaKernelHandle::compile` が NVRTC 再コンパイルを
    /// 回避してプロセス内 LRU をヒットする」ことを検証するための観測点
    /// （`wmma_tf32_opt_block_tile` 等と同じ「非公開モジュールへの薄い
    /// 診断用ラッパー」方針。`internal-diagnostics` feature〈既定 off〉
    /// でのみコンパイルされる）。
    pub fn module_cache_hit_count() -> Option<u64> {
        crate::module_cache::KernelModuleCache::global()
            .ok()
            .map(|cache| cache.hit_count())
    }

    /// プロセス内 LRU カーネルモジュールキャッシュのミス件数。
    /// [`module_cache_hit_count`] と同じ理由・同じ feature ゲート方針。
    pub fn module_cache_miss_count() -> Option<u64> {
        crate::module_cache::KernelModuleCache::global()
            .ok()
            .map(|cache| cache.miss_count())
    }
}
