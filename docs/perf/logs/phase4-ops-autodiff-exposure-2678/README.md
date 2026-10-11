# Phase 4 の演算・自動微分の facade 公開の実機 parity 申し送り（イシュー #2678）

対象は `Var` の委譲メソッド（FFT・低精度 forward・逆三角／双曲線・非有限値・累積・順序統計・ヒストグラム／二分探索・
形状・索引付き更新・テンソル積・pad モード・活性化）と facade `Tape` の `bincount`・`bincount_weighted`・`jacobian`・`hessian`・
`backward_detect_anomaly`。本エージェントの実行環境には CUDA／Metal 実機への到達手段がないため、実機テストは
**未実測**のまま GB10・M4 Max セッションへ申し送る。CPU 側の正しさは Linux で実行可能な
`crates/facade/tests/phase4_ops_facade.rs`（閉形式・手計算値）と各内部クレートのテストで検証済み。tolerance・baseline は変更しない。

> CUDA は #2992 で実測済み（2026-10-11・`784b4e79`。末尾「CUDA 実測記入欄（#2992）」参照）。以下の「未実測」は記入時点の記述。

> Metal は #2993 で実測済み（2026-10-11・対象コミット `c8e1ceb1`・Mac mini〈Apple M4〉。末尾「Metal 実測記入欄（#2993）」参照）。

## 新しい数値経路がないこと

公開は内部の自由関数（`crates/autodiff/src/<module>.rs`）への 1 行委譲だけで、新規 `Op`・`BackendOps` メソッド・カーネルを足していない。
このため facade 経由の実機 parity は、各機能の内部クレート側 `*_backend_parity.rs`（`#[ignore]`）の結果がそのまま適用される。

## 判定契約

REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`fandhe_ai_backend_cpu::assert_parity`）。結合順序が単一の連続 K ループと
異なる経路（Tensor Core・split-K 等）は `.claude/rules/coding-rust.md` の baseline 非後退方式に従う。新しい tolerance 定数は作らない。

## 実行コマンド

各機能の `crates/facade/tests/<名前>_backend_parity.rs`（`#[ignore]`）を実機で `--ignored` 実行する。機能ごとの手順・実測の詳細は
`docs/perf/logs/` の各機能のログ（`trig-ops-2634`・`fft-rfft-irfft-2631` 等）が正。

```
# CUDA（DGX Spark GB10）
for t in fft_ops low_precision_ops trig_ops nonfinite_ops cumulative_ops stat_reduce_ops binning_ops shape_view_ops \
         indexed_update_ops tensor_product_ops pad_modes_ops activation_scalar_ops softmin_threshold_ops \
         jacobian_hessian gradcheck_anomaly; do
  cargo test -p fandhe-ai --test ${t}_backend_parity -- --ignored cuda_
done
# Metal（M4 Max）: 上と同じ対象に `-- --ignored metal_`
```

## 結果記入欄（未実測）

| 環境 | 対象 | 結果 | 日付 |
|---|---|---|---|
| GB10 | `low_precision_ops` を除く 14 機能の `*_backend_parity.rs`（`--ignored cuda_`）。14 ファイル・19 テスト（詳細は下記「CUDA 実測記入欄（#2992）」） | pass（19/19） | 2026-10-11 |
| Mac mini（Apple M4） | `low_precision_ops` を除く 14 機能の `*_backend_parity.rs`（`--ignored metal_`）。14 ファイル・19 テスト（詳細は下記「Metal 実測記入欄（#2993）」） | pass（19/19） | 2026-10-11 |
| GB10・M4 Max | `low_precision_ops` のみ実測済み（#2629。両実機 pass・他機能は未実測のまま。`docs/perf/logs/low-precision-ops-2628/README.md`） | pass | 2026-10-11 |

## CUDA 実測記入欄（#2992）

CUDA（GB10）は #2992 で実測済み（2026-10-11・対象コミット `784b4e79`）。Metal は #2993 で実測済み（下記「Metal 実測記入欄（#2993）」）。本記録が各機能ログの GB10「未実測」行より優先する（機能ごとの行の更新は追従課題）。

- 判定規則（測定前に固定）: pass は各ファイルが exit 0 かつ `N passed; 0 failed`。fail は 1 件以上のテスト失敗（`tape_for(Device::Cuda(0))` の失敗・`Unsupported` を含む）。undetermined はノード到達不能またはビルド不能。CUDA vs CPU の不一致は判定不能にせず通常の parity 失敗として扱う
- 実行コマンド（ファイルごと）: `cargo test -p fandhe-ai --release --test <名前>_backend_parity -- --ignored --nocapture --test-threads=1 cuda_`（`--release --nocapture --test-threads=1` は判定に影響しない実行オプション。`timeout 1200` 付き）
- 環境: NVIDIA GB10・driver 580.173.02・CUDA 13.0（V13.0.88）・rustc 1.97.0。計測前後で GPU 使用率 0%・常駐プロセス 2 件（同一 pid）で変化なし。`.rev-stamp` は転送後・実行後とも `784b4e79` 一致
- 転送: `docs/real-hardware-verification-env.md` §3 の rsync（`--exclude 'target/'` を併用）。生ログ・`env_info.txt`（マスク済み）は `cuda/` 配下
- 判定契約・tolerance・baseline は不変（本記録は実測のみ。コード・テスト変更なし）

| ファイル（`crates/facade/tests/`） | 関数 | 結果 | 備考 |
|---|---|---|---|
| `fft_ops_backend_parity.rs` | `cuda_fft_matches_cpu_reference` | pass | 3 passed |
| 同上 | `cuda_fft_c2c_matches_cpu_reference` | pass | |
| 同上 | `cuda_stft_matches_cpu_reference` | pass | |
| `trig_ops_backend_parity.rs` | `cuda_trig_ops_matches_cpu_reference` | pass | |
| `nonfinite_ops_backend_parity.rs` | `cuda_nonfinite_ops_matches_cpu_reference` | pass | |
| `cumulative_ops_backend_parity.rs` | `cuda_cumulative_ops_match_cpu_reference` | pass | |
| `stat_reduce_ops_backend_parity.rs` | `cuda_stat_reduce_ops_match_cpu_reference` | pass | |
| `binning_ops_backend_parity.rs` | `cuda_binning_ops_match_cpu_reference` | pass | 実行 0.00s（他は 0.3〜0.6s）。ビルド済みで `CudaBackendOps::new(0)` を通過しており skip ではない。経路確認は追従課題 |
| `shape_view_ops_backend_parity.rs` | `cuda_shape_view_ops_match_cpu_reference` | pass | |
| `indexed_update_ops_backend_parity.rs` | `cuda_indexed_update_ops_match_cpu_reference` | pass | |
| `tensor_product_ops_backend_parity.rs` | `cuda_tensor_product_ops_match_cpu_reference` | pass | |
| `pad_modes_ops_backend_parity.rs` | `cuda_pad_modes_ops_match_cpu_reference` | pass | |
| `activation_scalar_ops_backend_parity.rs` | `cuda_activation_scalar_ops_matches_cpu_reference` | pass | |
| `softmin_threshold_ops_backend_parity.rs` | `cuda_bit_exact_forward_matches_cpu_reference` | pass | 4 passed（bit 一致判定） |
| 同上 | `cuda_bit_exact_backward_matches_cpu_reference` | pass | bit 一致判定 |
| 同上 | `cuda_req2_forward_matches_cpu_reference_within_tolerance` | pass | 統一複合判定 |
| 同上 | `cuda_req2_backward_matches_cpu_reference_within_tolerance` | pass | 統一複合判定 |
| `jacobian_hessian_backend_parity.rs` | `cuda_jacobian_hessian_match_cpu_reference` | pass | |
| `gradcheck_anomaly_backend_parity.rs` | `cuda_gradcheck_anomaly_match_cpu_reference` | pass | |

fail は 0 件のため「別 issue 候補」（原因切り分け）は無し。

## Metal 実測記入欄（#2993）

Metal（Mac mini・Apple M4）は #2993 で実測済み（2026-10-11・対象コミット `c8e1ceb1`）。計測機体は 2026-10-11 の所有者指示（`docs/real-hardware-verification-env.md` §7）により開発機の M4 Max ではなく Mac mini とする。本記録が各機能ログの Metal「未実測」行より優先する（機能ごとの行の更新は追従課題）。

- 判定規則（実測前の実装計画で固定。README への記入は実測後）: pass は各ファイルが exit 0 かつ `N passed; 0 failed` で、`running N tests` の N が期待件数（`fft_ops` 3・`softmin_threshold_ops` 4・他 1。計 19）と一致すること。fail は 1 件以上のテスト失敗（`tape_for(Device::Metal)` の失敗・`Unsupported` を含む）。undetermined はノード到達不能またはビルド不能。Metal vs CPU の不一致は自前バックエンド間の比較なので判定不能にせず通常の parity 失敗として扱う。softmin の bit 一致 2 件の fail は「bit 一致 fail」と記録し REQ-2 pass へ読み替えない
- 実行コマンド（ファイルごと）: `cargo test -p fandhe-ai --release --test <名前>_backend_parity -- --ignored --nocapture --test-threads=1 metal_`（`--release --nocapture --test-threads=1` は判定に影響しない実行オプション。`CARGO_TARGET_DIR` はノード側の共有ターゲット）
- 環境: Apple M4・macOS 27.0.1（26A434）・rustc 1.99.0。計測前に他の cargo／rustc プロセスが無いことを確認。load average は計測前 3.15（1 分）、計測後 13.02（1 分。本計測自身のビルドを含む）
- 転送: `docs/real-hardware-verification-env.md` §3・§7 の rsync（`--exclude 'target/'` を併用）。`.rev-stamp` は転送後・実行後とも `c8e1ceb1`（全桁一致）。生ログ・`env_info.txt`（マスク済み）は `metal/` 配下
- 判定契約・tolerance・baseline は不変（本記録は実測のみ。コード・テスト変更なし）

| ファイル（`crates/facade/tests/`） | 関数 | 結果 | 備考 |
|---|---|---|---|
| `fft_ops_backend_parity.rs` | `metal_fft_matches_cpu_reference` | pass | 3 passed |
| 同上 | `metal_fft_c2c_matches_cpu_reference` | pass |  |
| 同上 | `metal_stft_matches_cpu_reference` | pass |  |
| `trig_ops_backend_parity.rs` | `metal_trig_ops_matches_cpu_reference` | pass |  |
| `nonfinite_ops_backend_parity.rs` | `metal_nonfinite_ops_matches_cpu_reference` | pass |  |
| `cumulative_ops_backend_parity.rs` | `metal_cumulative_ops_match_cpu_reference` | pass |  |
| `stat_reduce_ops_backend_parity.rs` | `metal_stat_reduce_ops_match_cpu_reference` | pass |  |
| `binning_ops_backend_parity.rs` | `metal_binning_ops_match_cpu_reference` | pass | 実行 0.00s（他は 0.05〜0.47s）。ビルド済みで実行され skip ではない。カーネル経路の確認は追従課題 |
| `shape_view_ops_backend_parity.rs` | `metal_shape_view_ops_match_cpu_reference` | pass |  |
| `indexed_update_ops_backend_parity.rs` | `metal_indexed_update_ops_match_cpu_reference` | pass |  |
| `tensor_product_ops_backend_parity.rs` | `metal_tensor_product_ops_match_cpu_reference` | pass |  |
| `pad_modes_ops_backend_parity.rs` | `metal_pad_modes_ops_match_cpu_reference` | pass |  |
| `activation_scalar_ops_backend_parity.rs` | `metal_activation_scalar_ops_matches_cpu_reference` | pass |  |
| `softmin_threshold_ops_backend_parity.rs` | `metal_bit_exact_forward_matches_cpu_reference` | pass | 4 passed（bit 一致判定） |
| 同上 | `metal_bit_exact_backward_matches_cpu_reference` | pass | bit 一致判定 |
| 同上 | `metal_req2_forward_matches_cpu_reference_within_tolerance` | pass | 統一複合判定 |
| 同上 | `metal_req2_backward_matches_cpu_reference_within_tolerance` | pass | 統一複合判定 |
| `jacobian_hessian_backend_parity.rs` | `metal_jacobian_hessian_match_cpu_reference` | pass |  |
| `gradcheck_anomaly_backend_parity.rs` | `metal_gradcheck_anomaly_match_cpu_reference` | pass |  |

fail は 0 件のため「別 issue 候補」（原因切り分け）は無し。なお 1 テスト内では最初の fail で停止するため、pass の場合のみ全 op が評価済みである。

## 保留（本イシューでは公開していない）

- `hinge_embedding_loss`・`soft_margin_loss`・`multilabel_margin_loss`: `Reduction` を `fandhe_ai::nn::loss::Reduction` で名指しできる前提（#2602 未マージ）
- `Tape::gradcheck` と `GradcheckOptions`／`GradcheckReport`: 決定記録に facade シグネチャ（レシーバ・テープ生成の受け方）が未記載
- 行 12〜15（3D プーリング・ConvTranspose3d／MaxUnpool・Fold／Unfold・LRN／weight_norm／spectral_norm）: 承認の対象外
