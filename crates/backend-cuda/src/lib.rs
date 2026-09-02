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
//! （`fandhe_ai_tensor_core::device::DeviceProvider` の CUDA 実装）を追加した。上記の
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
//! 追加した。`fandhe_ai_tensor_core::buffer::MemoryOps` の CUDA 実装であり、
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
//! 追加した。`fandhe_ai_tensor_core::dispatch::select_gemm_kernel`（#67 が設計した
//! 決定的規則。`docs/dispatch-rules-design.md`）の結果に従い、naive／
//! tiled（`CudaGemm`）・WMMA f16（`CudaWmmaGemm`）を呼び分ける。TF32/f32
//! 経路（`CudaGemm::run_wmma_tf32`・#62）・`mma.sync` 経路（`CudaMmaGemm`・
//! #187）は、決定表（設計文書 §4）が TF32 既定採用を #186（TASK-11.1g）の
//! ユーザー承認まで保留と定めているため、現時点の `select_gemm_kernel` の
//! 自動経路には含めない（f32 は常に Tiled）。既存の `CudaGemm`／
//! `CudaWmmaGemm`／`CudaMmaGemm` の直接指定 API はテスト・証跡用途
//! （#70）にそのまま温存する（設計文書 §5.4）。
//!
//! TASK-1.9c（#46）で `ops` モジュール（[`ops::CudaBackendOps`]）を追加した。
//! `fandhe_ai_tensor_core::backend_ops::BackendOps` の CUDA 実装であり、`gemm` は
//! [`CudaGemm::run_tiled_f32`] へ委譲する（既定カーネル変種の選択は保守的に
//! tiled 固定とし、`CudaGemmAuto` を介した Tensor Core 経路の自動選択への
//! 切替は別スコープ）。イシュー #599 で elementwise 5 演算（`add`／`mul`／
//! `relu`／`exp`／`tanh`。[`elementwise::CudaElementwise`]）を実装し、
//! `gemm_bias_act` を GEMM epilogue 融合カーネル
//! （[`CudaGemm::run_tiled_bias_act_f32`]）で実融合化した（`bias`
//! ブロードキャスト形状の非厳密一致ケースは非融合合成へフォールバックする。
//! `ops::gemm_bias_act_route` 参照）。reduction（`sum`／`max`）は GPU
//! カーネル未実装のまま `fandhe_ai_tensor_core::device::BackendError::Unsupported`
//! を返す（out-of-scope-tracking.md 対象）。
//!
//! イシュー #499（GEMM 性能改善ツリー #479 の後続）で L2 再利用のための
//! タイル→SM 割り当てスウィズル（`swizzle`・`kernels_mma::
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
//! GB10 未実測の SM 数を実測値と誤扱いしていたこと）により差し戻した。
//!
//! イシュー #775 で 2026-08-20 GB10 実機再計測（4096: base 34.4089 →
//! swizzle(動的幅 g8) 54.3055 TFLOPS・×1.578 が安定再現。512〜2048 は
//! ×0.979〜0.992）を根拠に、**サイズ条件付き適用**（総タイル数
//! `num_m_blocks * num_n_blocks >= 2048`。`swizzle::should_apply_swizzle`）
//! のロジック自体は実装したが、`gemm_mma::CudaMmaGemm::new`（本番既定
//! コンストラクタ）への結線は見送った。#758 差し戻し理由（採用基準の無承認
//! 読み替え・結線前必須確認未実施・CI 恒久検査の SM 数入力誤り）のうち
//! 採用基準はイシュー #775 のユーザー起票の受け入れ条件を承認記録として
//! 明記し、SM 数は `device.multiprocessor_count()` 実測値を動的に使うこと
//! でハードコード依存を解消したが、結線前必須確認（レジスタスピル・
//! bit 一致・parity 非後退・`cuda_floor_bench` 実測）は当時 GB10 実機到達
//! 可能なセッションで未実施だったため、`new` は base 専用のまま維持し、
//! サイズ条件付き適用のコンパイル・`launch_f16` ディスパッチは opt-in・
//! `internal-diagnostics` feature 限定の `gemm_mma::CudaMmaGemm::
//! new_with_size_conditional_swizzle`（実機検証専用入口）からのみ到達
//! できるようにした。
//!
//! イシュー #782 で 2026-08-21 GB10 実機再計測（A/B: 4096 で ×1.592・
//! 512〜2048 は劣化 5% 以内・`mma_f16_swizzle_variant_matches_
//! base_bit_exact_output` bit 一致 ok。`docs/perf/cuda-gemm-swizzle-ab.md`
//! §6.2 参照）を根拠に、A/B 実測・bit 一致の 2 項目についてユーザー承認の
//! もと `gemm_mma::CudaMmaGemm::new` へサイズ条件付き適用機構を結線した。
//! `new_with_size_conditional_swizzle` は `new` と重複するため廃止し、
//! 明示幅指定・強制適用の診断用入口（`new_with_swizzle`／
//! `new_without_swizzle`）のみを opt-in・`internal-diagnostics` feature
//! 限定のまま残す。**parity 非後退確認・結線後の `cuda_floor_bench` 実測
//! （≥50 TFLOPS 確認）・レジスタスピル確認は #782 の受け入れ条件
//! チェックリストで「マージ後確認可」と明記されていたが、PR #784
//! codex-review 指摘への対応として結線済みコード自身に対するマージ前
//! 検証（2026-08-21・DGX Spark GB10 実機）で全項目解消済み**
//! （`docs/perf/cuda-gemm-swizzle-ab.md` §6.3 参照）。
//! PR #784 codex-review P1 是正で、`swizzle::should_apply_swizzle` は
//! 上記の総タイル数閾値に加えて M/N 各軸のブロック数が実測点
//! M=N=K=4096 相当以上（`swizzle::SWIZZLE_APPLY_MIN_M_BLOCKS`/
//! `SWIZZLE_APPLY_MIN_N_BLOCKS`）であることも要求するよう改訂した。実測
//! 承認範囲が正方形形状（M=N=K）に限られるため、非正方形形状（例:
//! M=32768, N=512）への外挿を base 経路へフォールバックさせる（`swizzle.rs`
//! ドキュメンテーションコメント参照）。
//!
//! **PR #784 codex-review 追加指摘の是正**: 上記 M/N 軸別ガードは K を
//! 見ないため、M=N=4096, K=8 のような未検証形状（メモリアクセス量・L2
//! 再利用特性が実測承認点 M=N=K=4096 と大きく異なる）にも適用してしまう
//! 不備が残っていた。K の生値をそのまま下限とする
//! `swizzle::SWIZZLE_APPLY_MIN_K`（実測承認点の K=4096）を追加し、
//! M/N 軸別ガード・総タイル数条件と AND で結合することで、実測承認範囲
//! （M=N=K=4096 相当以上）を超える適用を防いだ（`swizzle.rs::
//! should_apply_swizzle` ドキュメンテーションコメント参照）。
//! [`gemm_mma::CudaMmaGemm::swizzle_applies`]／`launch_f16` の呼び出し
//! シグネチャも `k` を受け取る形へ更新した。
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
//! Phase C-4（#511。親イシュー #503 の最終タスク）で `module_cache`
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
//! コメント参照）。
//!
//! イシュー #1024 で結線範囲を `CudaGemm::new`（f32 GEMM 本番経路。naive
//! f32/f16・tiled f32/f16・tiled_bias_act_f32・wmma_tf32・wmma_tf32_opt・
//! wmma_tf32_staged の 8 カーネル＋サイズ条件付き swizzle 変種）へ拡張
//! した。共通の 3 段フォールバック実装は
//! `module_cache::load_function_cached` へ抽出し、
//! `kernels_mma.rs::RenderedMmaKernel::compile` と `gemm.rs::CudaGemm::new`
//! の双方がこれを呼ぶ（`gemm.rs::GemmKernelSpec::descriptor` 参照）。
//! `CudaWmmaGemm::new`・`CudaMmaGemm::new`・elementwise/transpose 群は
//! 未結線のまま残す（本イシューのスコープ外。将来の横展開は別イシューで
//! 判断する）。ビルド時 PTX 事前埋め込み（candle の `build.rs` 事前生成
//! 方式）は CUDA toolkit 非搭載環境でのビルド成立契約（REQ-2・PoC-v2-5）
//! と衝突するため不採用と判断した（`docs/backend-cuda-ptx-embedding-
//! decision.md` 参照）。

//! イシュー #801（refactor）で TF32 `mma.sync`(m16n8k8)/`ldmatrix`/`cp.async`
//! 経路（[`CudaMmaTf32Gemm`]。`kernels_mma_tf32.rs`／`gemm_mma_tf32.rs`）を
//! 追加した。既存 TF32 本番経路（`CudaGemm::run_wmma_tf32` の WMMA C++ API
//! ベース 3 段選択）は無変更のまま並存させる独立経路であり、本番
//! ディスパッチ（`ops.rs`／`gemm.rs`／`gemm_auto.rs`）へは結線しない
//! （`kernels_mma_tf32.rs` 冒頭ドキュメンテーションコメント「位置づけ・
//! 非結線」参照）。数値一致回帰・parity 非後退契約・本番採否判断は後続
//! イシュー #802 のスコープ。
//!
//! イシュー #929 で `context_cache`（プロセス内キャッシュ）を追加した。
//! 上記の「固定ソースの一回コンパイル経路はインスタンス構築時 1 回のみの
//! コンパイルであり結線しない」判断（Phase C-4・イシュー #511 時点）は
//! shape 特化コンパイルキャッシュ（`module_cache.rs`）のスコープ判断
//! だったが、`ops::CudaBackendOps` はそのインスタンス自体を演算呼び出し
//! ごとに都度構築していたため、2 回目以降の呼び出しでも `CudaContext`
//! 生成・NVRTC コンパイルが繰り返し発生していた（`scripts/bench/
//! framework-compare/results/summary.md:177` 実測: サイズ非依存の固定
//! オーバーヘッド約 440〜460ms）。`context_cache` は `ordinal` をキーに
//! `CudaDevice`／`CudaGemm`／`CudaElementwise`／`CudaRmsNorm`／
//! `CudaSoftmax` をプロセスワイドに常駐させ、`ops.rs`（`device_handle`・
//! 各演算メソッド）・`device.rs`（`CudaDeviceProvider::probe`）の両方から
//! 参照する。カーネルソース・許容誤差・境界検査には触れない（構造的に
//! 数値一致契約は不変）。
//!
//! イシュー #1042（親ツリー #1029 Phase 2）で [`precision`] モジュールを
//! 追加した。`ops::CudaBackendOps::gemm`（公開 GEMM 経路）は既定で
//! `CudaGemm::run_tiled_f32`（FP32 厳密）を使うが、
//! `precision::set_tf32_gemm_enabled(true)`（`facade::
//! set_cuda_tf32_gemm_enabled` から到達する composition root 公開 API）
//! で opt-in すると `CudaGemm::run_wmma_tf32`（TF32 Tensor Core。既存の
//! WMMA 本番経路）へプロセスワイドに切り替わる。既定 OFF・fail-closed
//! （TF32 カーネル使用不能時は FP32 へ黙示フォールバックしない）の契約
//! は `precision.rs` モジュール冒頭コメントを正とする。

mod context_cache;
pub mod device;
mod elementwise;
mod error;
mod gemm;
mod gemm_auto;
// イシュー #1035: f32 GEMM の形状別カーネル選択（simple / double-buffer /
// split-K）ヒューリスティック。GPU 資源を要さない純関数のみで構成し、
// `gemm.rs` の opt-in コンストラクタ（`internal-diagnostics` feature
// 限定）から参照される。本番既定経路（`CudaGemm::new`）へは未結線
// （実装計画 §3・§8 参照）。
mod gemm_variant;
mod kernels_gemm_variants;
// イシュー #1035: 上記 2 モジュールを実際に NVRTC コンパイル・起動する
// opt-in 実行経路。`internal-diagnostics` feature（既定 off）限定
// （`gemm_variant_selection.rs` 冒頭ドキュメントコメント参照）。
#[cfg(feature = "internal-diagnostics")]
pub mod gemm_variant_selection;
// イシュー #926: CUDA GEMM の固定初期化コスト（tape_for 初期化コスト。
// フレームワーク横並びベンチ・PR #915 実測 440〜460 ms 帯）のフェーズ分解
// 診断テスト。`kernels`／`kernels_wmma_opt`（非公開 `mod`）へ到達するため
// integration test ではなくクレートルートの兄弟モジュールとして配置する
// （`init_cost_diag_tests.rs` 冒頭ドキュメンテーションコメント参照。
// `jit_cache_bench_tests.rs`〈#534・`nvrtc.rs` 子モジュール〉と同型の判断）。
// プロダクションコード（`ops.rs`・`gemm.rs`・`device.rs` 等）は本イシューで
// 変更しない（改善実装は Phase 2 のスコープ）。
mod gemm_mma;
mod gemm_mma_tf32;
mod gemm_wmma;
// イシュー #956: #946（`context_cache` プロセス内キャッシュ）反映後の
// fresh モード GEMM で N=2048 のみに現れる約 166 ms の再現性ある固定
// コストのフェーズ分解診断テスト。`init_cost_diag_tests`（#926）と同じ
// 理由（`kernels` 系非公開 `mod`・`context_cache` 非公開 `mod` への到達）
// でクレートルートの兄弟モジュールとして配置する
// （`fresh_overhead_diag_tests.rs` 冒頭ドキュメンテーションコメント
// 参照）。プロダクションコード（`gemm.rs`・`memory.rs`・
// `context_cache.rs` 等）は本イシューで変更しない。
#[cfg(test)]
mod fresh_overhead_diag_tests;
#[cfg(test)]
mod init_cost_diag_tests;
mod kernels;
mod kernels_elementwise;
mod kernels_mma;
mod kernels_mma_tf32;
mod kernels_mse;
mod kernels_rmsnorm;
mod kernels_sgd;
mod kernels_softmax;
mod kernels_tiled_pipeline;
mod kernels_transpose;
mod kernels_wmma;
mod kernels_wmma_opt;
pub mod memory;
mod module_cache;
mod mse;
// イシュー #1024: `module_cache`／NVRTC ディスクキャッシュへの結線
// （`gemm.rs::CudaGemm::new`）を実機で検証する `#[ignore]` テスト。
// `context_cache`（非公開 `mod`）へ到達する必要があるため
// `init_cost_diag_tests` と同じ理由でクレートルートの兄弟モジュールとして
// 配置する。
#[cfg(test)]
mod module_cache_wiring_tests;
mod nvrtc;
mod ops;
mod pool;
pub mod precision;
mod rmsnorm;
mod sgd;
mod softmax;
mod swizzle;
mod transpose;

pub use device::{CudaDevice, CudaDeviceProvider};
pub use elementwise::CudaElementwise;
pub use error::CudaError;
pub use gemm::CudaGemm;
// `TiledF32Kernel`（`Classic`/`Pipeline`）は `CudaGemm::run_tiled_f32` 系
// 3 入口が実際にどちらのカーネルへ分岐したかを表す観測用の型（イシュー
// #1137。`gemm.rs::TiledF32Kernel` 冒頭ドキュメンテーションコメント参照）。
// `wmma_tf32_staged_swizzle_group_width`（数値ヘルパー）と異なり、こちらは
// 内部カーネルディスパッチの実装詳細（Classic/Pipeline の二択）そのものを
// 型として外部公開することになるため、`SpecializedMmaKernelHandle` 等と
// 同じ `internal-diagnostics` feature（既定 off）限定で公開する
// （codex-review P1 指摘・PR #1164。`CudaGemm::tiled_f32_kernel_for` の
// みが本型を返す唯一の公開経路で、同メソッドも同 feature でゲートする。
// `tests/cpu_cuda_tiled_pipeline_parity.rs` は既に `required-features =
// ["internal-diagnostics"]`〈`Cargo.toml`〉のためこのゲート化による
// 通常 CI ジョブへの影響はない）。
#[cfg(feature = "internal-diagnostics")]
pub use gemm::TiledF32Kernel;
pub use mse::CudaMse;
// `TiledPipelineFunction`／`CudaGemm::compile_tiled_pipeline_variant`／
// `CudaGemm::launch_tiled_pipeline_f32` はベンチ専用の常駐 API（イシュー
// #1033）。**本番既定経路（`CudaGemm::new`）は #1137 で `run_tiled_f32`
// 系 3 入口へ形状条件付きに結線済み**（`gemm.rs::CudaGemm::
// select_tiled_f32_kernel` 参照。整列形状〈`n % 4 == 0 && k % 4 == 0`〉
// かつ `new` 時のコンパイル成功時のみ pipeline へ分岐し、それ以外は
// classic へ fail-closed にフォールバックする）。以下で feature ゲートする
// のは `TiledPipelineFunction` 型そのもの・`compile_tiled_pipeline_variant`・
// `launch_tiled_pipeline_f32`（常駐ハンドルを外部から明示的に扱う診断・
// ベンチ専用 API）であり、本番結線自体はこれらに依存しない
// （`gemm.rs::TiledPipelineFunction` 冒頭ドキュメンテーションコメント
// 参照）。
// PR #1071 codex-review P1 指摘の是正: 従来は本ブロックへ無条件 re-export
// しており、`SpecializedMmaKernelHandle` と同じ「テスト・ベンチ専用」の
// 意図に反して通常ビルドの安定した公開 API 面へ漏出していた。
// `SpecializedMmaKernelHandle` と同じ `internal-diagnostics` feature
// （既定 off）でゲートし、`examples/gemm_tiled_pipeline_bench.rs`・
// `tests/cpu_cuda_tiled_pipeline_parity.rs` の常駐 API 使用箇所は
// `Cargo.toml` の `[[example]]`/`[[test]]` セクションで
// `required-features = ["internal-diagnostics"]` を指定して到達する
// （`cargo test --all-features` でのみビルド・実行される）。
// `TiledPipelineFunction` を返す・受け取る公開関数（`compile_tiled_pipeline_
// variant`・`launch_tiled_pipeline_f32`。`gemm.rs` 側で同 feature ゲート
// 済み）以外にこの型を公開面へ露出する経路はない。
#[cfg(feature = "internal-diagnostics")]
pub use gemm::TiledPipelineFunction;
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
pub use gemm_mma_tf32::CudaMmaTf32Gemm;
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
    use crate::{kernels_mma, kernels_mma_tf32, kernels_wmma_opt, swizzle};

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

    // イシュー #782 codex-review P1 是正（本番結線前必須のレジスタスピル
    // 確認が未実施のまま残っていた不備）: `examples/mma_ptx_dump.rs`
    // （NVRTC で PTX を生成しファイルへダンプし、DGX 実機の
    // `ptxas -arch=sm_121 -v` へオフラインで掛けてレジスタ使用量・スピル
    // を観測する診断バイナリ）がカーネルソース文字列を必要とする。
    // `kernels_mma` は非公開 `mod` のため、`mma_f16_source`／
    // `mma_f16_source_with_swizzle`（いずれも `kernels_mma.rs` 側では
    // `pub fn` だが到達境界がない）を本モジュール経由でのみ crate 外部
    // （example）へ公開する（上記関数群と同じ「非公開モジュールへの薄い
    // 診断用ラッパー」方針）。本番経路（`gemm_mma.rs::CudaMmaGemm::new`・
    // `new_with_swizzle`・`new_without_swizzle`）はこの再公開に依存せず
    // `kernels_mma` を直接 `use` し続ける。
    pub use kernels_mma::{mma_f16_source, mma_f16_source_with_swizzle};

    // イシュー #803: warp タイル拡大候補（`docs/perf/
    // cuda-gemm-mma-warp-tile-register-budget.md` §3.1 候補表）のレジスタ
    // 収支を `examples/mma_ptx_dump.rs` から実機 `ptxas -v` で観測するための
    // 再公開。`mma_f16_source`／`mma_f16_source_with_swizzle` と同じ「非公開
    // モジュールへの薄い診断用ラッパー」方針（本モジュール冒頭コメント
    // 参照）。本番経路（`gemm_mma.rs`）はこの再公開に依存しない（#804 の
    // 本番結線まで `MMA_WARP_TILES_M`/`_N` 定数自体は無変更）。
    pub use kernels_mma::mma_f16_source_with_warp_tiles;

    // イシュー #804: ブロックタイル拡大・ステージ数増候補（実装計画
    // Step 1 の候補表。`docs/perf/cuda-gemm-mma-block-tile-stages.md`）を
    // `examples/mma_ptx_dump.rs` から観測するための再公開。
    // `mma_f16_source_with_warp_tiles`（#803・#822）と同じ「非公開
    // モジュールへの薄い診断用ラッパー」方針。本番経路（`gemm_mma.rs`）は
    // この再公開に依存しない（`MMA_BM`/`MMA_BN`/`MMA_STAGES` 等の本番定数は
    // 本イシュー時点で実機到達不能のため無変更のまま。計画 Step F 参照）。
    pub use kernels_mma::mma_f16_source_with_block_tile;

    // イシュー #840: #804 が整備したブロックタイル拡大・ステージ数増
    // 候補ソース生成（上記 `mma_f16_source_with_block_tile`）を実際に
    // NVRTC コンパイル・起動して実機 A/B 計測するためのランナー型・
    // レイアウト導出ヘルパーの再公開。`examples/
    // gemm_mma_block_tile_bench.rs`（診断専用 A/B ランナー）専用。
    // `mma_f16_source_with_block_tile` と同じ「非公開モジュールへの薄い
    // 診断用ラッパー」方針。本番経路（`gemm_mma.rs`）はこの再公開・型に
    // 一切依存しない（本番定数〈`MMA_BM`/`MMA_BN`/`MMA_STAGES`〉・
    // `gemm_mma.rs` 本番コンストラクタは本イシューで無変更。採否判断・
    // 本番結線は後続イシュー #842 のスコープ）。
    pub use kernels_mma::{
        CompiledMmaF16BlockTileKernel, MmaBlockTileLayout, RenderedMmaF16BlockTileKernel,
        render_mma_f16_block_tile,
    };

    // イシュー #855: `render_mma_f16_block_tile` と同じ候補パラメータで、
    // 静的予算以下でも `extern __shared__` 動的 SMEM 変換を強制適用する
    // 対照実験用ランナー（`kernels_mma.rs::mma_f16_source_with_block_tile_
    // forced_dynamic_smem` ドキュメンテーションコメント「目的」節参照）。
    // 実機観測で「変換そのものの欠陥」か「候補定数側の潜在バグ」かを
    // 切り分けるための唯一の呼び出し元は
    // `examples/gemm_mma_block_tile_bench.rs`。本番経路はこの再公開に
    // 一切依存しない。
    pub use kernels_mma::render_mma_f16_block_tile_forced_dynamic_smem;

    // derive_mma_block_tile_layout は非公開関数（`kernels_mma.rs` 内部の
    // レイアウト導出ロジックの単一の真実源）だが、`examples/
    // gemm_mma_block_tile_bench.rs` は候補表定義・opt-in 予算比較・
    // 除外ログ出力に導出結果（`MmaBlockTileLayout`）を必要とするため、
    // 薄いラッパー関数として再公開する（`mma_swizzle_group_width` 等の
    // 「非公開ロジックへの薄い診断用関数ラッパー」方針と同型）。
    pub fn mma_f16_block_tile_layout(
        bm: u32,
        bn: u32,
        bk: u32,
        stages: u32,
        warp_tiles_m: u32,
        warp_tiles_n: u32,
    ) -> Result<MmaBlockTileLayout, crate::error::CudaError> {
        kernels_mma::derive_mma_block_tile_layout(bm, bn, bk, stages, warp_tiles_m, warp_tiles_n)
    }

    // イシュー #840 codex-review 是正: `examples/gemm_mma_block_tile_bench.rs`
    // の比較基準行（現行本番構成）が `threads`/`smem_bytes`/
    // `needs_dynamic_smem` をハードコードしたリテラルではなく、候補行と
    // 同じ「単一の真実源」（`derive_mma_block_tile_layout`）経由で取得
    // できるようにするための薄いラッパー。`MMA_STAGES`・パディング定数
    // （`MMA_A_PAD`/`_B_PAD`）が将来変更された際にベンチ CSV・
    // `docs/perf/*.md` §4.1 転記値が自動追従するようにする。
    pub fn mma_f16_block_tile_layout_production()
    -> Result<MmaBlockTileLayout, crate::error::CudaError> {
        kernels_mma::derive_mma_block_tile_layout(
            kernels_mma::MMA_BM,
            kernels_mma::MMA_BN,
            kernels_mma::MMA_BK,
            kernels_mma::MMA_STAGES,
            kernels_mma::MMA_WARP_TILES_M,
            kernels_mma::MMA_WARP_TILES_N,
        )
    }

    // イシュー #806: TF32 生 mma.sync 経路（`kernels_mma_tf32.rs`）の
    // ベースソース・ブロックタイル拡大候補を `examples/
    // mma_tf32_ptx_dump.rs` から観測するための再公開。
    // `mma_f16_source`/`mma_f16_source_with_block_tile`（#782・#804）と
    // 同じ「非公開モジュールへの薄い診断用ラッパー」方針。本番経路
    // （`gemm_mma_tf32.rs::CudaMmaTf32Gemm`）はこの再公開に依存しない
    // （`MMA_TF32_BM`/`MMA_TF32_BN`/`MMA_TF32_STAGES` 等の本番定数は
    // 本イシュー時点で実機到達不能のため無変更のまま）。
    pub use kernels_mma_tf32::{mma_tf32_source, mma_tf32_source_with_block_tile};

    // イシュー #841: #806（PR #832）が整備した TF32 生 `mma.sync` ブロック
    // タイル拡大・ステージ数増候補ソース生成（上記
    // `mma_tf32_source_with_block_tile`）を実際に NVRTC コンパイル・起動
    // して実機 A/B 計測するためのランナー型・レイアウト導出ヘルパーの
    // 再公開。`examples/gemm_mma_tf32_block_tile_bench.rs`（診断専用 A/B
    // ランナー）専用。`CompiledMmaF16BlockTileKernel`/
    // `RenderedMmaF16BlockTileKernel`/`render_mma_f16_block_tile`（#840）と
    // 同じ「非公開モジュールへの薄い診断用ラッパー」方針。本番経路
    // （`gemm_mma_tf32.rs::CudaMmaTf32Gemm`）はこの再公開・型に一切依存
    // しない（`MMA_TF32_BM`/`MMA_TF32_BN`/`MMA_TF32_STAGES` 等の本番定数は
    // 本イシューで無変更。`CudaMmaTf32Gemm` 自体が #839 で不採用〈凍結〉
    // 判断済み）。
    pub use kernels_mma_tf32::{
        CompiledMmaTf32BlockTileKernel, MmaTf32BlockTileLayout, RenderedMmaTf32BlockTileKernel,
        render_mma_tf32_block_tile,
    };

    // derive_mma_tf32_block_tile_layout は非公開関数（`kernels_mma_tf32.rs`
    // 内部のレイアウト導出ロジックの単一の真実源）だが、`examples/
    // gemm_mma_tf32_block_tile_bench.rs` は候補表定義・opt-in 予算比較・
    // 除外ログ出力に導出結果（`MmaTf32BlockTileLayout`）を必要とするため、
    // 薄いラッパー関数として再公開する（`mma_f16_block_tile_layout`
    // （#840）と同型）。
    pub fn mma_tf32_block_tile_layout(
        bm: u32,
        bn: u32,
        bk: u32,
        stages: u32,
        warp_tiles_m: u32,
        warp_tiles_n: u32,
    ) -> Result<MmaTf32BlockTileLayout, crate::error::CudaError> {
        kernels_mma_tf32::derive_mma_tf32_block_tile_layout(
            bm,
            bn,
            bk,
            stages,
            warp_tiles_m,
            warp_tiles_n,
        )
    }

    // イシュー #841: `examples/gemm_mma_tf32_block_tile_bench.rs` の比較
    // 基準行（現行本番構成）が `threads`/`smem_bytes`/`needs_dynamic_smem`
    // をハードコードしたリテラルではなく、候補行と同じ「単一の真実源」
    // （`derive_mma_tf32_block_tile_layout`）経由で取得できるようにする
    // ための薄いラッパー（`mma_f16_block_tile_layout_production`（#840）と
    // 同型）。`MMA_TF32_STAGES`・パディング定数（`MMA_TF32_A_PAD`/`_B_PAD`）
    // が将来変更された際にベンチ CSV・`docs/perf/*.md` 転記値が自動追従
    // するようにする。
    pub fn mma_tf32_block_tile_layout_production()
    -> Result<MmaTf32BlockTileLayout, crate::error::CudaError> {
        kernels_mma_tf32::derive_mma_tf32_block_tile_layout(
            kernels_mma_tf32::MMA_TF32_BM,
            kernels_mma_tf32::MMA_TF32_BN,
            kernels_mma_tf32::MMA_TF32_BK,
            kernels_mma_tf32::MMA_TF32_STAGES,
            kernels_mma_tf32::MMA_TF32_WARP_TILES_M,
            kernels_mma_tf32::MMA_TF32_WARP_TILES_N,
        )
    }

    /// `mma_tf32`（TF32 `mma.sync` 経路）カーネルのブロックタイル形状
    /// `(block_m, block_n)`。`examples/mma_tf32_ptx_dump.rs` の occupancy
    /// 概算専用（`mma_f16_block_tile` と同じ「非公開モジュールへの薄い
    /// 診断用ラッパー」方針）。
    pub fn mma_tf32_block_tile() -> (u32, u32) {
        (kernels_mma_tf32::MMA_TF32_BM, kernels_mma_tf32::MMA_TF32_BN)
    }

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

    /// tiled f32（本番既定 f32 経路 `TILED_F32`。イシュー #1030／#1034 の
    /// occupancy 診断対象）カーネルのブロックタイル形状
    /// `(block_m, block_n)`。`examples/gemm_profile_target.rs` の
    /// `Path::TiledF32`／`Path::TiledF32Swizzle` occupancy 概算専用
    /// （`wmma_tf32_opt_block_tile`・`mma_f16_block_tile` と同じ理由）。
    ///
    /// `crate::kernels::TILE`（32。レジスタブロッキング導入前の旧
    /// 素朴カーネルが使っていたタイル一辺）ではなく
    /// `crate::kernels::TILED_F32_BM`／`TILED_F32_BN`（64。#1032 の
    /// レジスタブロッキング適用後、実際に起動する `launch_tiled_f32`
    /// が使うブロックタイル一辺。`gemm.rs` の `tiled_f32_launch_config`
    /// 参照）を返す。旧定数のままだと `actual_blocks`／`blocks_per_sm`
    /// の occupancy 見積りが実カーネルの 4 倍（(64/32)^2）過大になり、
    /// ncu 実測との突合が成立しない（Cursor Bugbot 指摘: PR #1090）。
    pub fn tiled_f32_block_tile() -> (u32, u32) {
        (crate::kernels::TILED_F32_BM, crate::kernels::TILED_F32_BN)
    }

    /// イシュー #499: L2 再利用のためのタイル→SM 割り当てスウィズルの
    /// グルーピング幅動的選択（`swizzle::select_swizzle_group_width`）を
    /// `mma_f16` のブロックタイル（`MMA_BM`/`MMA_BN`）に対して適用した
    /// 結果を返す。`swizzle` は非公開 `mod`（`lib.rs` の `mod swizzle;`）
    /// のため、crate 外部（`examples/gemm_mma_swizzle_bench.rs`）から
    /// 到達するにはこの diagnostics 経由の薄いラッパーが必要
    /// （`mma_f16_block_tile`・`wmma_tf32_opt_block_tile` と同じ理由・
    /// 同じ feature ゲート方針）。`gemm_mma::CudaMmaGemm::new`（本番既定
    /// コンストラクタ。イシュー #782 でサイズ条件付き適用機構を結線済み）
    /// がこの式を直接呼ぶ（`gemm_mma.rs::CudaMmaGemm::new` ドキュメンテー
    /// ションコメント参照。サイズ条件付き適用は `swizzle::
    /// should_apply_swizzle` が別途判定する）。本関数自体は A/B 計測
    /// （`examples/gemm_mma_swizzle_bench.rs`）専用の診断用ラッパーで
    /// あり続ける（本番経路とは独立に固定候補 `{8,16}` を個別計測する
    /// 用途のため）。
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

    /// イシュー #1034: [`mma_swizzle_group_width`] の tiled f32（本番既定
    /// f32 経路 `TILED_F32`）版。`swizzle::select_swizzle_group_width` を
    /// tiled f32 のブロックタイル（`kernels::TILE` x `kernels::TILE`。
    /// 32×32）に対して適用する（`mma_f16`・TF32 opt-staged と異なる
    /// ブロックタイルのため専用ラッパーが必要。`swizzle.rs` 本体は無
    /// 変更）。`examples/gemm_tiled_f32_swizzle_bench.rs`・
    /// `gemm.rs::CudaGemm::new_with_tiled_f32_swizzle`（`internal-
    /// diagnostics` feature 限定・診断用 opt-in 入口。本番既定コンスト
    /// ラクタ `CudaGemm::new` はこの経路を呼ばない）を呼ぶ側から到達する。
    pub fn tiled_f32_swizzle_group_width(num_sms: u32) -> u32 {
        swizzle::select_swizzle_group_width(num_sms, crate::kernels::TILE, crate::kernels::TILE)
    }

    /// イシュー #1034: `kernels::tiled_f32_source_with_swizzle`（非公開
    /// `mod kernels` 配下）の crate 外部（`examples/
    /// gemm_tiled_f32_swizzle_bench.rs` 等の診断用バイナリ）からの到達用
    /// 再公開。`kernels` は本番カーネルソースの内部表現のため非公開の
    /// ままとし（PR #637 codex-review 指摘と同じ理由）、本モジュール経由
    /// のみを唯一の公開境界とする（`wmma_tf32_f32_staged_source_with_
    /// swizzle` の再公開と同じ方針）。
    pub use crate::kernels::tiled_f32_source_with_swizzle;

    // イシュー #856: `examples/wmma_tf32_staged_ptx_dump.rs`
    // （TF32 opt-staged base／swizzle 変種のレジスタ・スピル差分を
    // 実機 `ptxas -v` で観測する診断バイナリ。`mma_ptx_dump.rs` と同型）
    // からの到達用再公開。`kernels_wmma_opt` は非公開 `mod` のため、本
    // モジュール経由でないと crate 外部（example）へ到達できない
    // （`mma_f16_source`/`mma_f16_source_with_swizzle` と同じ「非公開
    // モジュールへの薄い診断用ラッパー」方針）。本番経路（`gemm.rs::
    // CudaGemm::new`）はこの再公開に依存せず `kernels_wmma_opt` を直接
    // `use` し続ける。
    pub use kernels_wmma_opt::{
        wmma_tf32_f32_staged_source, wmma_tf32_f32_staged_source_with_swizzle,
    };

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
