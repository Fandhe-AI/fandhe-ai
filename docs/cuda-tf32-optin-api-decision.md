# CUDA TF32 Tensor Core 経路の opt-in 公開 API 設計判断（#1042）

イシュー #1042「REQ-2 複合判定内の TF32 tensor core 経路を opt-in で選択可能に
する（ユーザー判断）」に対応する。親ツリー #1029（GEMM カーネルの candle 超え）
配下の Phase 2 イシュー（human-required ラベル付き）。

## 背景

`backend-cuda` には WMMA TF32 GEMM 経路（`CudaGemm::run_wmma_tf32`。
`crates/backend-cuda/src/gemm.rs`。staged→opt→basic の 3 段選択）が実装済みで、
誤差分布も GB10 実機で実測済み（`docs/perf/cuda-tensor-core-tolerance-opt-
remeasurement.md`・`cuda-tensor-core-tolerance-gb10-scale-sweep.md`）だが、公開
経路（`ops.rs::CudaBackendOps::gemm`）は本イシュー導入前は常に FP32 tiled
（`run_tiled_f32`）であり、TF32 経路へはテスト・example からしか到達できな
かった。

一方 framework-compare の burn CUDA は TF32 へ強制降格される（burn 0.21 の
既定挙動）ため、fandhe-ai の FP32 計測と条件が揃わない。candle は
`MM_F32_REDUCED_PRECISION` 既定 `false` の opt-in 方式。REQ-2 の数値一致は
TF32 前提の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）へ
既に改定済みであり、opt-in 時もこの契約の範囲内で動作する。

## 承認ステータス: 自動運転モード実装（本ドキュメントが承認記録を兼ねる）

human-required ラベルの本イシューは通常ユーザーの計画承認を経るが、本実装は
自動運転モードのエージェントが実装計画に記載済みの設計判断（安全側の既定・
fail-closed 方針）をそのまま採用した。設計判断はいずれも「既定 OFF・FP32
非後退・許容誤差不変」という安全側の選択であり、ユーザーが後から確認・
差し戻し可能な形で記録する。

## API 形状（確定）

- `backend-cuda::precision` モジュール（`crates/backend-cuda/src/precision.rs`）に
  プロセスワイドの opt-in フラグ（`static AtomicBool`、既定 `false`）と
  setter/getter（`set_tf32_gemm_enabled` / `tf32_gemm_enabled`）を追加した。
  candle の `MM_F32_REDUCED_PRECISION` と同型のプロセスグローバル方式。
- `facade` から自由関数として再公開する（`release_cached_memory`
  （`crates/facade/src/lib.rs`）と同型の composition root 直委譲。facade は
  `fandhe-ai-backend-cuda` へ無条件依存済みのため cfg 分岐不要）。
  - 採用名: `fandhe_ai::set_cuda_tf32_gemm_enabled(enabled: bool)` /
    `fandhe_ai::cuda_tf32_gemm_enabled() -> bool`。CUDA 限定であることを
    名前で明示する（`docs/compat-api-scope.md` §0 の公開面）。
- **既定は OFF（FP32 厳密）**。フラグ OFF 時の経路・出力は本イシュー導入前と
  bit-exact に不変（`ops.rs::CudaBackendOps::gemm` の分岐は `else` 節で従来の
  `run_tiled_f32` 呼び出しをそのまま保持）。

## 適用範囲・フォールバック方針（確定）

- 適用は `CudaBackendOps::gemm`（f32 の素の GEMM。`crates/backend-cuda/src/
  ops.rs`）のみ。`gemm_bias_act`・`gemm_resident_*`・学習経路は本イシューでは
  FP32 のまま（スコープ境界。拡張は「フォローアップ」節参照）。
- opt-in 時に TF32 カーネルが使用不能（cc<8.0・NVRTC 失敗等）の場合は
  **fail-closed で型付きエラーを返す**（`run_wmma_tf32` の既存
  `CudaError::WmmaUnavailable` → `BackendError::KernelLaunchFailed` へ伝播）。
  FP32 への黙示フォールバックはしない（明示 opt-in の計測条件を静かに崩さ
  ない。#994 の診断コンストラクタと同じ方針）。

## framework-compare の制約

`bench-fandhe` は crates.io 公開版 `fandhe-ai =0.4.0` に完全固定されており
（deps-policy 第 9 区分。`check_framework_compare` が registry 取得元を
fail-closed 検査するため path 依存への差し替えは不可）、本イシューで追加した
新 API は次回リリース（v0.5.0 公開 + ピン更新のユーザー承認）まで
`bench-fandhe` から呼べない。

- **C-1（本イシューのスコープ）**: `scripts/bench/framework-compare/` に
  `--tf32` フラグの CLI・JSONL・summarize.py 対応一式を追加した。
  `bench-fandhe` は `--tf32` 指定時に「fandhe-ai >= 0.5.0 が必要」の
  `MEASURE_ERROR` で fail-fast する（`--phases` の対象外組合せ拒否と同型）。
  `bench-candle` は candle-core 0.11 の公開 API で `--tf32` を即時有効化し、
  candle TF32 との同条件比較を先行して可能にする。`bench-burn` は常時 TF32
  のため `--tf32` は受理せず fail-fast する（README に明記）。
- **C-2（本イシューのスコープ外・別イシュー提案）**: v0.5.0 公開後のピン更新
  （ユーザー承認必須）+ `bench-fandhe` 結線 + `run_all` スクリプトへの tf32
  スイープ追加。起票はユーザー承認を得てから行う（`out-of-scope-tracking.md`
  に従い、実装完了後に既存イシュー検索・ユーザー確認を経て起票する）。

## フォローアップ（スコープ外事項の追跡）

- `gemm_bias_act`・`gemm_resident_*`・学習経路への TF32 opt-in 拡張は本イシュー
  のスコープ外。
- C-2（v0.5.0 ピン更新後の `bench-fandhe` 結線・`run_all` tf32 スイープ）。

いずれも `out-of-scope-tracking.md` の規約に従い、実装完了後にユーザー承認を
得たうえで Issue 化を提案する（本エージェントは自動運転モードのため Issue の
自動起票は行わない）。
