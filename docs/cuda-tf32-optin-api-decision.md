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

**追補（2026-08-31・イシュー #1011）**: 承認済みピンを `fandhe-ai =0.5.0`
（crates.io 公開済み・`release-all.yml` run 33388884217）へ更新した（前提
条件は充足）。ただし `bench-fandhe`（`main.rs`）側の呼び出し結線・`run_all`
の tf32 スイープ追加自体は本更新のスコープ外で未実施のため、`--tf32` は
引き続き `MEASURE_ERROR` で fail-fast する。C-2 本体は別イシューのまま。

## フォローアップ（スコープ外事項の追跡）

- `gemm_bias_act`・`gemm_resident_*`・学習経路への TF32 opt-in 拡張は本イシュー
  のスコープ外。
- C-2（v0.5.0 ピン更新後の `bench-fandhe` 結線・`run_all` tf32 スイープ）。

いずれも `out-of-scope-tracking.md` の規約に従い、実装完了後にユーザー承認を
得たうえで Issue 化を提案する（本エージェントは自動運転モードのため Issue の
自動起票は行わない）。

## 追補（イシュー #1355）: 第 3 モード 3×TF32

親ツリー #1354・承認元 #1338（ユーザー承認 2026-09-06）に基づき、`crate::
precision::CudaGemmPrecision` を `bool`（TF32 単発の 2 値）から 3 モード
（`Fp32Strict`／`Tf32`／`Tf32x3`）の enum へ拡張した。3×TF32（split-single
法。hi/lo 分割・3 回の `mma.sync` 累積）の設計・数値契約の詳細は
`docs/cuda-tf32x3-split-single-decision.md` を参照する（本 doc では API 形状の
要点のみ記す）。

- **API 形状**: `precision::set_gemm_precision(CudaGemmPrecision)`／
  `precision::gemm_precision() -> CudaGemmPrecision` を新設。facade は
  `fandhe_ai::set_cuda_gemm_precision`／`fandhe_ai::cuda_gemm_precision`
  として再公開する（`set_cuda_tf32_gemm_enabled`／`cuda_tf32_gemm_enabled`
  と並存）。
- **互換ラッパーの意味論**（公開 API 非破壊。`precision.rs` モジュール冒頭
  コメントが正）: `set_tf32_gemm_enabled(true)` は `Tf32` へ、
  `set_tf32_gemm_enabled(false)` はどのモードからでも `Fp32Strict` へ設定
  する。`tf32_gemm_enabled()` は `Tf32` のときのみ `true`（`Tf32x3` では
  `false`）。
- **fail-closed の範囲**: `Tf32x3` opt-in 時にカーネルが使用不能
  （cc<8.0・NVRTC コンパイル失敗・cp.async 16 バイト整列制約〈`n%4==0 &&
  k%4==0`〉不成立）な場合は `BackendError::KernelLaunchFailed`（
  `"3xTF32 gemm unavailable (fail-closed): …"` 接頭辞）を返し、FP32／単発
  TF32 への黙示フォールバックはしない（`Tf32` opt-in の既存契約と同型）。
- **framework-compare 対象外**: `bench-fandhe` は crates.io 公開版
  `fandhe-ai =0.7.0` に完全固定されており、本イシューで追加した
  `set_cuda_gemm_precision` は次回リリース + ピン更新（ユーザー承認）まで
  `bench-fandhe` から呼べない（`--tf32` フラグの 3×TF32 対応は本イシューの
  スコープ外。上記 C-2 と同型のフォローアップ）。
- **適用範囲**: 素の `CudaBackendOps::gemm` のみ（`gemm_bias_act`・
  `gemm_resident_*`・学習経路は対象外。既存の適用範囲契約を継承）。

## 追補（イシュー #1983）: C-2（`bench-fandhe` 結線・`run_all_cuda.sh` TF32 スイープ）実施

上記「framework-compare の制約」節の C-2（v0.5.0 ピン更新後の `bench-fandhe`
結線・`run_all` tf32 スイープ追加）を実施した。承認済みピンは `fandhe-ai
=0.9.0` まで進み `fandhe_ai::set_cuda_tf32_gemm_enabled`／
`cuda_tf32_gemm_enabled` は crates.io 公開版へ cfg ゲートなしで収録済みの
ため、`--managed`／`--pinned-h2d`／`--graph`（v0.9.0 未収録の API を要する）
と異なり追加の cargo feature 導入は不要だった。

- **受理条件**: `bench-fandhe --tf32` は `--task gemm --device cuda`
  （fresh／reuse とも）の素の GEMM のみを受理する（`validate_tf32_flag`
  関数。`--managed` と同型の allowlist 方式）。以下はすべて `MEASURE_ERROR`
  で fail-closed 拒否する: `task != "gemm"` または `device != "cuda"`
  （TF32 opt-in が `CudaBackendOps::gemm` のみに効き `gemm_bias_act`／
  `gemm_resident_*`／train／infer は FP32 のままのため誤ラベル行を防ぐ）・
  `--phases`（`PhaseRecord` に `tf32` キーがない）・`--device-checksum`
  （`matmul_checksum`／`gemm_checksum` 経路の TF32 挙動未検証）・
  `--managed`／`--pinned-h2d`（`summarize.py` (a-tf32) 節が managed／pinned
  を区別しないため複合条件行の混入を防ぐ）。
- **有効化**: 受理時は `set_cuda_tf32_gemm_enabled(true)` を呼び
  `cuda_tf32_gemm_enabled()` で読み戻し確認する（`--managed` と同一の
  fail-closed 確認パターン）。`--tf32` なしの既定行は setter／getter を
  一切呼ばず、結線前と bit 同一のまま不変（`Record.tf32` は `cli.tf32`
  の値をそのまま反映し、`false` なら従来どおり JSONL に `tf32` キーを
  emit しない）。
- **要素単位検証は不変**: `reference.verify_strict(&out)`（救済項なし）の
  まま変更しない。TF32 は結合順序・精度が異なるため大きい N で
  `fail_count > 0` になりうるが、これは想定内の記録事項であり是正対象では
  ない（`summarize.py` (a-tf32) 節が「無効: …」と表示する既存挙動に委ねる。
  tolerance・baseline は不変）。
- **`run_all_cuda.sh` へ (a-tf32) スイープを追加**: `bench-fandhe`・
  `bench-candle`（C-1 で既に `--tf32` 結線済み）の `gemm cuda`（fresh・
  N=256〜4096）に `--tf32` を付けて実行する。`bench-burn` は `--tf32` を
  常に `MEASURE_ERROR` で拒否する仕様のため対象外（(a') ブロックと同じ
  理由: 対象外の既知失敗で `skipped-cuda.log` を汚さない）。burn の cuda
  gemm 行は通常スイープ（(a)）で既に `tf32:true` として記録されるため、
  `summarize.py` の (a-tf32) 節には fandhe-ai／candle／burn の 3
  フレームワークが並ぶ。
- **対象外のまま**: 3×TF32（`CudaGemmPrecision::Tf32x3`／
  `set_cuda_gemm_precision`）の `--tf32` 対応は引き続きスコープ外（上記
  「追補（イシュー #1355）」節のフォローアップと同じ）。新規 Issue は
  起票せず本追補への記録のみとする。
- **実機実測**: 本イシューの実装エージェント実行環境に DGX Spark GB10
  実機への到達手段がなく、`gemm_tf32_cuda_smoke`（`#[ignore]`）・
  `run_all_cuda.sh` の (a-tf32) 実数値取得は未実施のまま GB10 セッションへ
  申し送る（数値を捏造しない）。
  - **2026-09-18 GB10 実測済み（(a-tf32) のみ）**: `run_all_cuda.sh`
    （(a-tf32) スイープ込み・registry ピン `fandhe-ai =0.9.0`・転送元
    コミット `536c56a8`・専有 1 セッション）を GB10 で実行し、成果物を
    `docs/perf/logs/framework-compare-cuda-tf32-sweep-1983/` へ収納した。
    (a-tf32) 節に N=256〜4096 の全 5 形状で fandhe-ai（`--tf32`）・candle
    （`--tf32`）・burn（`tf32:true`）の 3 行が並ぶことを確認（受け入れ条件
    充足）。全 112 セル完走・`skipped-cuda.log` 0 行。fandhe-ai `--tf32`
    行は `verify_strict`（救済項なし・`bound=0`）により全形状
    `fail_count > 0` の「無効」表示となるが、これは上記「要素単位検証は
    不変」のとおり想定内の記録事項（tolerance・baseline・判定式は不変）。
    burn 行の `fail_count`／`rescued` は 0.9.0 正式再計測
    （`docs/perf/logs/framework-compare-0.9.0-remeasure/gb10/`）の burn 行と
    全 5 形状で一致。単発 run・採否判定なし・正式系列は置き換えない。
    `gemm_tf32_cuda_smoke`（`#[ignore]`）の GB10 実行は引き続き未実施。
