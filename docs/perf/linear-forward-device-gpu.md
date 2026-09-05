# `linear_forward_device` の CUDA／Metal 実装・実機実測（イシュー #1216）

親イシュー #1135（Phase 5・横並び再計測サイクルの残存未達の解消）配下。
`docs/inference-forward-fixed-cost-design.md` §3.2「段階 B」で CPU
実装・設計は確定済みだったが、CUDA／Metal 実装は実機検証環境が必要な
ため out-of-scope として引き継がれていた（同 doc §4 旧版）。本文書は
その実装・実機実測の記録を持つ。

## §1 目的

`BackendOps::linear_forward_device`（`a`／`w`／`bias`／戻り値をすべて
`DeviceBuffer` のまま扱い `y = act(a @ w + bias)` を計算する非破壊
拡張メソッド）を CUDA・Metal へ実装し、多層 MLP 推論チェーンで層間の
D2H→H2D（`gemm_resident_rhs*` 系が持つ「`a` を毎回 upload・結果を
毎回 download」という構造）を解消できることを実機で確認する
（`docs/perf/train-step-phase-breakdown.md` §15.5 の仮説の検証対象）。

## §2 実装要約

| バックエンド | 実装箇所 | 再利用したカーネル |
|---|---|---|
| CUDA | `crates/backend-cuda/src/ops.rs::CudaBackendOps::linear_forward_device` | `CudaGemm::launch_tiled_bias_act_f32_resident`（`gemm_resident_rhs_act` と同一） |
| Metal | `crates/backend-metal/src/ops.rs::MetalBackendOps::linear_forward_device` | `MetalGemm::dispatch_strided_bias_act_prepared`（`gemm_resident_rhs` と同一） |

`gemm_resident_rhs*` 系（`w`／`bias` のみ常駐・`a`／戻り値はホスト
常駐）との違いは、`a` も呼び出し元が既にデバイスへ置いた
`DeviceBuffer` として受け取り、戻り値も `DeviceBuffer` のまま返す点
のみ。カーネル本体・launch config・epilogue（bias 加算・ReLU）は
完全に共有するため、同一入力に対して両者は **bit 完全一致**する
契約（§3 (b) で実機確認）。

### CUDA 固有の差分

- **世代検査の対象に `a` を追加**: `gemm_resident_rhs*` は `a` を
  呼び出し内で毎回 upload するため世代検査の対象外だったが、
  `linear_forward_device` の `a` は呼び出しを跨いで生存する常駐
  バッファのため、`w`・`bias` と同じく `resident_generations` へ
  含める（poison／stale generation 検査を `a` にも適用）
- **出力バッファの確保元**: 呼び出し元へ escape する戻り値のため
  `CudaMemory::new(&device)`（`gemm_resident_rhs` が使う、呼び出し
  内で死ぬ一時 tracker）ではなく `static_cuda_memory`（`memory_ops()`
  と同一インスタンス）で確保する（REQ-14 の単一計測系列。`docs/
  device-resident-update-design.md` §3.3d）。CPU 実装が
  `shared_cpu_memory()` を使うのと同じ判断
- `k == 0` は `Linear::new` が `in_features == 0` を構築時に拒否する
  ため到達不能と判断し、フォールバックを設けず型付きエラーで拒否
  （`gemm_resident_rhs` と同じ判断）

### Metal 固有の差分

- 世代検査の機構自体が Metal 側に存在しない（`gemm_resident_rhs` も
  同様）ため CUDA と異なり追加の世代検査はない
- 出力バッファは `static_metal_memory()`（`memory_ops()` と同一
  インスタンス・`context_cache::cached_context()` 共有）で確保
- **同期契約**: `dispatch_strided_bias_act_prepared` はコマンド
  バッファへ積むのみで待たない。次層の dispatch は同一コマンド
  バッファ内（または `should_auto_flush` 分割後の後続コマンド
  バッファ。同一キューの serial 実行順）に積まれるため、前層出力を
  次層の入力として読む順序は GPU 側で保証される（イシュー #1017 の
  設計文書が「実機で確認する必要がある」と留保していた項目。§3 (c)
  で実機確認済み）

## §3 テスト（実機必須。`#[ignore]`）

| ファイル | 内容 |
|---|---|
| `crates/backend-cuda/tests/linear_forward_device_real_device.rs` | (a)〜(d) + record-only ベンチ |
| `crates/backend-metal/tests/linear_forward_device_parity.rs` | 同上 |

- **(a) CPU 参照との複合判定**: `(1,1,1)`・`(4,8,4)`・`(37,65,33)`・
  `(64,784,256)`・`(64,256,10)` × bias 有無 × `Activation::{None,Relu}`。
  参照は `CpuBackendOps::gemm_bias_act`。判定は REQ-2 統一複合判定
  （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）
- **(b) 同バックエンド `gemm_resident_rhs_act` との bit 一致**: 同一
  形状群で `download` 後 `assert_eq!`（tolerance 不使用）
- **(c) 2 層チェーン**: batch 64・784→256→ReLU→10。upload 1 回 →
  `linear_forward_device` ×2 → download 1 回。参照は CPU の
  `gemm_bias_act` 連鎖
- **(d) fail-closed**: `w` 形状不整合・`bias` 形状不一致・CPU 側
  `DeviceBuffer` を `a` に渡した `DeviceMismatch`・`m == 0` の早期
  return が空 `[0, n]` を返すこと

CI 実行可能なユニットテスト（実機不要。`crates/backend-cuda/src/
ops.rs::ops::tests`）: poison 済み ordinal・`w`／`a` の stale
generation・CPU バッファの `DeviceMismatch`・別 ordinal の
`DeviceMismatch` を早期 return 分岐（`m == 0 || n == 0`）経由で検証。
`build-no-cuda-toolkit` ジョブでも green（CUDA driver 呼び出し前に
拒否されるため）。

## §4 実機実測

### 環境

| 項目 | 値 |
|---|---|
| コミット | `ab0b77d0b23369603c02ee9ee2335d8488fde791`（`perf/1216-linear-forward-device-gpu` ブランチの base） |
| rustc | `1.96.0 (ac68faa20 2026-05-25)` |
| Metal 実機 | Apple M4 Max（macOS 26.6.2）。実ホスト名は書かない（`docs/real-hardware-verification-env.md` 方針） |
| CUDA 実機 | 本エージェント実行環境に CUDA toolkit・GPU 実機がないため未実測（下記参照） |

### Metal（M4 Max 実機・2026-09-06）

```sh
cargo test -p fandhe-ai-backend-metal --release --test linear_forward_device_parity -- --ignored --nocapture
```

parity テスト (a)〜(d) はすべて pass（CPU 参照一致・`gemm_resident_
rhs_act` との bit 完全一致・2 層チェーンのコマンドバッファ共有同期
確認・fail-closed 系列とも green）。

backend レベル before/after ベンチ（2 層チェーン。batch 64・
784→256→ReLU→10・warmup 20・iters 20・5 trial 中央値。旧経路:
`gemm_resident_rhs_act` を層ごとに呼び毎回 upload/download・新経路:
`mem.upload` 1 回 → `linear_forward_device` ×2 → `mem.download` 1 回）
を 5 回実行した結果:

| run | before_median_s | after_median_s | speedup_x |
|---|---|---|---|
| 1 | 0.000763 | 0.000576 | 1.326 |
| 2 | 0.000533 | 0.000295 | 1.806 |
| 3 | 0.000559 | 0.000434 | 1.288 |
| 4 | 0.000429 | 0.000288 | 1.488 |
| 5 | 0.000384 | 0.000266 | 1.443 |

5 回とも新経路が旧経路を上回る（speedup 1.29〜1.81 倍。中央値
1.44 倍）。record-only 方針（hard assert なし）のとおり、本結果は
「非後退どころか明確な改善」であることの記録に留め、判定はこの表を
根拠に人間が行う。

### CUDA（DGX Spark GB10 実機）

**未実測**。本エージェント実行環境に CUDA toolkit・GPU 実機がない
ため、下記コマンドを実機セッションで実行し本節を追記すること
（`docs/real-hardware-verification-env.md` §3 の rsync 手順。
`docs/perf/cuda-tf32-optin-parity.md` と同じ「未実測明記」方針）:

```sh
cargo test -p fandhe-ai-backend-cuda --release --test linear_forward_device_real_device -- --ignored --nocapture
```

CI 実行可能なユニットテスト（§3）は本エージェント実行環境
（CUDA toolkit 非搭載・`build-no-cuda-toolkit` 相当）で pass 済み。

### 既存回帰（非後退確認）

```sh
cargo test -p fandhe-ai-backend-metal --release -- --ignored
```

`gemm_resident_*`・`backend_ops_real_device`・`sgd_device_*` 等の
既存実機テストは非後退（全 pass）。ただし `command_batching.rs` の
一部テスト（`pool_reuse_zero_fill_does_not_synchronize_open_batch`
等）が、全 `#[ignore]` テストを一括実行した際に低頻度で flaky に
失敗する事象を観測した。個別実行・本変更の有無いずれでも再現する
（`git stash` で本変更を外した状態でも同種の別テストが同様に flaky
失敗することを確認済み）ため、本イシューの変更が原因の regression
ではなく既存の pre-existing flakiness と判断する。修正はスコープ外
（`out-of-scope-tracking.md` の規約に従い、ユーザー承認のうえ別
イシューで追跡することを推奨）。

## §5 Phase 2（facade／autodiff 結線）の判断

実装計画 Step 9 は「CUDA または Metal のいずれかで after が before
より速い（中央値ベース）ことを実施条件」としていた。Metal は §4 の
とおり 5 回すべてで明確な改善（中央値 1.44 倍）を確認できたため、
条件自体は満たしている。

しかし本 PR では **Phase 2（`DeviceParamStore::forward_device_chain`
の追加・`Sequential::predict_resident` の内部差し替え）は実施しない**:

- CUDA 実機実測が本エージェント実行環境の制約により未完了であり、
  「CUDA／Metal 両方の結線判断材料が揃ってから行う」という元の設計
  文書 §3.2 の前提（第三者レビュー観点での対称性）を崩さないため
- `facade`／`autodiff` への結線は plan 型（`LinearChainStep` の
  導入・層列からの plan 構築・`Sigmoid`/`Tanh` 混在層での不採用
  判定等）を伴う独立した変更単位であり、backend 実装・実機実測
  （本 PR の主眼）と 1 PR に混在させると差分の見通しが悪化するため

Phase 2 の実施自体は妥当（backend 実装は完了・Metal 実測は明確な
改善）と判断しており、後続イシュー #1217（bench-fandhe の infer に
`predict_resident` reuse モードを追加。本イシューに依存）または
新規イシューでの実施を推奨する。CUDA 実機実測が得られ次第、本文書
§4 に追記し、Phase 2 着手の最終判断材料とすること。
