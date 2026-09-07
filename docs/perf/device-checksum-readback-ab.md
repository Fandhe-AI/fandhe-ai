# checksum のデバイス側 f64 reduction 化（イシュー #1339）

## §0 位置づけ

`docs/perf/cuda-gemm-reuse-phase-breakdown.md`・`metal-gemm-reuse-phase-breakdown.md`
の実測（イシュー #1182・#1189）は、framework-compare の gemm reuse 計測窓のうち
`host_copy`（`C` の D2H）と `checksum`（縮退検出用の全要素和をホスト側 `f64` 逐次和
で求め直す処理）が `iter_total` の約 66〜75% を占め、candle 比未達の主因が GEMM
カーネル自体ではなく**ハーネスの診断コスト**であることを確定した（イシュー #1338
でユーザー承認済み、本イシュー〈#1339〉の着手根拠）。

本イシューは、縮退検出契約（全反復での checksum 検査・末尾反復での要素単位 parity）
を維持したまま、checksum を GPU 上の reduction で求めホストへの読み戻しを 8 バイト
に縮小する経路を bench-fandhe・bench-candle の両ハーネスへ実装する。

**本ドキュメント時点のスコープ（重要）**: 本イシューの実装は以下 2 段階に分割された。

1. **本 PR で完了**: 公開 API（`fandhe_ai_tensor_core::{ChecksumReadout, GemmChecksum}`・
   `BackendOps::gemm_checksum`・`fandhe_ai_autodiff::Var::matmul_checksum`・facade
   再エクスポート）の追加、**`backend-cpu` での実装**（`gemm` と bit 同一の `C`・
   checksum はホスト f64 逐次和と bit 一致）、両ハーネス（`bench-fandhe`
   `--device-checksum` + `device-checksum` feature・`bench-candle`
   `--device-checksum`）の結線、CPU 経由の自動テストによる正しさの検証。
2. **後続イシューへ引き継ぎ（未実装。本 PR のスコープ外）**: `backend-cuda`・
   `backend-metal` の `BackendOps::gemm_checksum` は本 PR 時点でデフォルト実装
   （常に `BackendError::Unsupported` を返す fail-safe）のままであり、GPU 上の
   double reduction（CUDA）・Neumaier 補償和 reduction（Metal）カーネル自体は
   実装していない。したがって **CUDA／Metal での H2D／D2H 削減効果の実機実測は
   本ドキュメントには含まれない**（実測を伴わない性能改善の主張は行わない。
   `.claude/rules/security.md` A08「数値を捏造しない」）。

CPU バックエンドは元々ホスト常駐でデバイス転送を持たないため、`--device-checksum`
による H2D／D2H 削減効果は原理上存在しない（CPU 側の実装は「GPU バックエンドと
同じ API 面を満たす」ための意味論的対称性の確認に留まる。§2 参照）。

## §1 実装内容

### §1.1 公開 API（`tensor-core`／`autodiff`／`facade`）

- `fandhe_ai_tensor_core::backend_ops::{ChecksumReadout, GemmChecksum}`（新規型）と
  `BackendOps::gemm_checksum`（デフォルトメソッド。既定 `Unsupported`）を追加した。
  `ChecksumReadout::ChecksumOnly` は checksum のみ、`WithOutput` は `C` 全体も
  ホストへ download する契約（GPU バックエンドは `ChecksumOnly` のとき `C` の
  download を回避する契約。実装は §1.2 参照）。
- `fandhe_ai_autodiff::Var::matmul_checksum`（新規）は `Var::matmul` と同じ
  「①クロステープ検査 → ②shape 検査 → ③入力実体化」を経て `ops().gemm_checksum`
  を呼ぶ。`matmul` と異なり **tape へノードを記録しない**（backward の対象外。
  ベンチ計測専用の入口という位置づけ）。
- `facade`（`fandhe_ai`）は `ChecksumReadout`／`GemmChecksum` を 1 行 `pub use` で
  再エクスポートし、`Var::matmul_checksum` は既存の `Var` 再エクスポート経由で
  到達可能（REQ-12「`Tape`／`BackendOps`／`new_with_ops` を再エクスポートしない」
  契約は不変。`crates/facade/tests/api_surface.rs` で機械的に固定）。

### §1.2 backend-cpu

`CpuBackendOps::gemm_checksum` は `gemm`（`gemm_into_slice`）と同一カーネル呼び出し
で `C` を求め、checksum は `out.iter().map(|&x| x as f64).sum()`（先頭から逐次
`f64` 昇格・固定順序）で計算する。`WithOutput` の `output` は `gemm` と bit 同一
（`crates/backend-cpu/src/ops.rs::gemm_checksum_tests` で検証）。

### §1.3 backend-cuda／backend-metal

**未実装（本 PR のスコープ外）**。`tensor-core::backend_ops::BackendOps::gemm_checksum`
のデフォルト実装（常に `Err(BackendError::Unsupported(..))`）のまま。`Var::matmul_checksum`
はこのエラーをそのまま呼び出し元へ伝播する（判定迂回経路を作らない。
`.claude/rules/security.md` A08）ため、`--device cuda`／`--device metal` で
`--device-checksum` を指定した `bench-fandhe` は `MEASURE_ERROR` になる
（`run_gemm_device_checksum`／`run_gemm_reuse_device_checksum` 内の
`a.matmul_checksum(&b, ..)` 呼び出しでエラーが伝播する）。

引き継ぎ設計（実装計画時点のメモ。実装時に再検討可）:

- **CUDA**: 既存 `gemm.rs::run_tiled_f32`（H2D → カーネル → D2H）から D2H 前の
  デバイスバッファを取り出す経路を新設し、`kernels_mse.rs`（2 段 grid-stride +
  warp butterfly reduction。`atomicAdd` 不使用で決定的）と同型の `double`
  アキュムレータ reduction カーネルを追加して `C` を読み戻す前に checksum
  （8 バイト）を求める。
- **Metal**: MSL は `double` 非対応のため、`rmsnorm.metal::rmsnorm_kahan_add`
  （Neumaier 改良版 Kahan 補償和）と同型の (hi, lo) 2 float reduction を
  `gemm_simdgroup_tiled` の出力（`pad8` パディング後の `m_eff × n_eff` バッファ、
  論理領域 `m × n` のみ手動境界チェックしながら読む）に対して適用する。
- 両バックエンドとも `WithOutput` 選択時のみ続けて `C` を readback する。

## §2 CPU での検証結果（実測。捏造なし）

- `crates/backend-cpu/src/ops.rs::gemm_checksum_tests`（`cargo test -p
  fandhe-ai-backend-cpu`）: `ChecksumOnly` の checksum がホスト `f64` 逐次和と
  完全一致・`output` が `None` であること／`WithOutput` の `output` が `gemm`
  と bit 同一であることを検証（2 件、pass）。
- `crates/autodiff/tests/matmul_checksum.rs`（`cargo test -p fandhe-ai-autodiff
  --test matmul_checksum`）: `Var::matmul_checksum` が `Var::matmul` と同一の
  checksum／bit 同一の `output` を返すこと、クロステープ・shape 不整合の検証
  順序が `matmul` と同じであることを検証（4 件、pass）。
- `crates/facade/tests/matmul_checksum_readout.rs`: composition root（`fandhe_ai::
  tape()`）経由で CPU バックエンドまで到達することを検証（1 件、pass）。
- `scripts/bench/framework-compare/bench-fandhe`（`device-checksum` feature
  有効ビルド。`--config patch.crates-io.fandhe-ai.path=<crates/facade 絶対パス>`
  併用）: `device_checksum_matches_legacy_checksum_fresh_and_reuse`（cargo test）
  が `--device-checksum` の有無で同一入力の checksum が一致することを fresh／
  reuse 双方で検証（pass）。加えて手動実行で JSONL 出力を確認した:

  ```text
  # --device-checksum あり（N=64・cpu・fresh）
  {"framework":"fandhe-ai", ..., "checksum":-18.685036, ...,
   "parity_total":4096,"parity_fail_count":0, ...,"device_checksum":true}
  # --device-checksum あり（N=64・cpu・reuse）
  {"framework":"fandhe-ai", ..., "checksum":-18.685036, ...,"init_s":0.004999500,
   "parity_total":4096,"parity_fail_count":0, ...,"device_checksum":true}
  ```

  （実行環境: Apple Silicon Mac、`cargo run --release` 相当ではなく `cargo run`
  〈debug〉。本ドキュメントは正しさの確認が目的であり、debug ビルドの実行時間は
  性能値として扱わない）。

- `scripts/bench/framework-compare` の python 側集計ツール（`summarize.py`・
  `compare_gemm_gate.py`・`compare_ab.py`・`compare_gemm_ab.py`・
  `compare_managed_ab.py`）に `device_checksum` フィールドの型検証・既定除外を
  追加し、既存の `_test.py` 群（296 件。`summarize_test.py` 205・
  `compare_ab_test.py` 28・`compare_gemm_ab_test.py` 28・
  `compare_gemm_gate_test.py` 16・`compare_managed_ab_test.py` 40）が全て
  pass することを確認した（既存ゲート・A/B ツールへの回帰なし）。

## §3 スコープ外・引き継ぎ

- **CUDA／Metal の `BackendOps::gemm_checksum` 実装**（§1.3 の引き継ぎ設計）。
  実装後に本ドキュメントへ実機（DGX Spark GB10・M4 Max）での gemm reuse
  5 回計測中央値の前後比較（AC-3）・checksum 複合判定の実測結果を追記する。
- **`gemm --mode reuse --phases` の device checksum 対応**（区間定義の再設計が
  必要。本イシューのスコープ外のまま）。
- **正式ゲート（#1031/#1037/#1117）の判定境界を device checksum 経路へ切り替える
  か**は判定ロジック変更（ユーザー判断事項）であり、本イシューでは既定判定を
  変更しない（`summarize.py`／`compare_gemm_gate.py` は `device_checksum:true`
  行を既定で除外したまま）。
- **`compare_checksum_ab.py`（off/on・両フレームワーク同時適用の専用集計ツール）
  ・`run_ab_checksum_device.sh`（実機 A/B 計測スクリプト）は未着手**。CUDA／Metal
  実装完了後、`run_ab_managed_cuda.sh`／`compare_managed_ab.py` を雛形に追加する。
- **fandhe reuse の A/B 毎反復 H2D と candle の 1 回アップロードの非対称**
  （既存・本イシューのスコープ外）。
