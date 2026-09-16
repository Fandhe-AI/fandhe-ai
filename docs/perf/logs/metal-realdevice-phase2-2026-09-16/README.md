# Metal（Apple M4 Max）実機 Phase 2 機能群 parity 実測（2026-09-16）

低レイヤー診断・機能網羅ツリー（ルート #1570）Phase 2 で「M4 Max 実機実測は
未実施のまま Mac セッションへ申し送り」となっていた `#[ignore]` テスト群を、
本セッション（Apple M4 Max・origin/main `3e43bbd0`）で一括実行した記録。
実装コードは一切変更していない（記録・ログ・doc 追記のみ）。tolerance／
baseline／判定規則の事後緩和も行っていない。

- 環境: `env_info.txt`（内部ホスト名・ユーザー名・絶対パスは含めない。
  ログ内の絶対パスは `/Users/<user>/…`・ホスト名は `masked` へ置換済み）
- 負荷: **共有負荷下**（load average 約 4〜30。他セッションの並走あり）。
  本ディレクトリの項目はすべて parity・bit 一致・決定性の確認であり負荷
  非依存のため、判定には影響しない（性能 A/B は本ディレクトリの対象外）
- ビルド: `CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai`・`--release`。
  backend-metal 側は `make test-ignored-metal` と同じ `--all-features`
  で統一（parity の数値経路は feature 非依存）。conv ランブックの
  `run_ignored_tests_metal.sh` はスクリプト記載どおり feature 指定なし
- 実行スクリプト: `run_phase2.sh`（d・e 群の一括実行。`summary.tsv` を生成）

## 1. 全体非後退（`make test-ignored-metal`）

| ログ | 内容 | 結果 |
|---|---|---|
| `make-test-ignored-metal.log` | `make test-ignored-metal` そのまま（1 回目） | lib 91 pass → `command_batching.rs` で 1 FAIL（`pool_reuse_zero_fill_does_not_synchronize_open_batch`）となり cargo test の既定動作（fail-fast）で **以降のバイナリは未実行** |
| `command_batching_isolation.log` | 上記 FAIL の切り分け（`--test-threads=1` ×3・既定並列 ×1） | 直列 3/3 pass・既定並列 1/1 FAIL（再現） |
| `make-test-ignored-metal-nofailfast.log` | 同コマンドに `--no-fail-fast` を付けた取り直し（全バイナリ） | **398 pass・5 FAIL**（内訳は §3） |

`command_batching::pool_reuse_zero_fill_does_not_synchronize_open_batch` は
2 つ目の assert（`alloc_zeroed` 自体は `encode()` を呼ばない: `encode` カウンタ
6 ≠ 5）で失敗する。同一バイナリの他 4 テストが singleton `MetalContext` の
診断カウンタ（`__diagnostic_batch_counters_snapshot`）を並列に進めるための
**並列干渉**で、`docs/backend-metal-command-batching-design.md` §4.3 の
実機記録も `--ignored --test-threads=1` で green とされている。`--no-fail-fast`
の取り直しでは（スケジューリング差により）pass。演算の非後退とは無関係。
テストハーネス上の所見として記録する（`make test-ignored-metal` は本バイナリ
に `--test-threads=1` を付けておらず、また `--no-fail-fast` でもないため
1 バイナリでも FAIL すると完全な一覧を出せない。Makefile は本 PR では変更
しない）。

既知 FAIL との突合: split-K 8 形状の既知 FAIL（`docs/perf/metal-gemm-splitk-
two-pass.md` §5.5）は #1512 の baseline 方式再切替により解消済みで、
`gemm_splitk_parity` は 1/1 pass。`gemm_hfrag_parity`（4 pass）・
`gemm_te_parity`（5 pass）も pass。


### 1a. 非 `#[ignore]` lib テストの M4 Max での FAIL（pre-push フックのブロッカー）

`cargo test --workspace --all-features`（`lefthook.yml` pre-push の `test` と同一）を本機で
実行すると、`crates/backend-metal/src/ops.rs` の macOS 限定・非 `#[ignore]` テスト 3 件
（`{max,avg,adaptive_avg}_pool2d_rejects_huge_broadcast_view_input_without_panicking`。
PR #1888 で追加）が `called Result::unwrap() on an Err value: ElementCountOverflow`
で FAIL する（backend-metal lib: 549 pass・3 fail・91 ignored）。panic 位置は fixture の
`base.broadcast_to(&[1, 1, 1usize << 62, 4]).unwrap()` で、`broadcast_to` 自体が要素数
overflow を拒否するため検証対象（`checked_bytes_for`）に到達しない。`cfg(target_os =
"macos")` 限定ファイルのため Linux CI では未コンパイル。本 PR の変更（docs／ログのみ）
とは無関係だが、本フックにより Mac からの push が拒否される（PR #1888 へ所見コメント
投稿済み。コード変更は本 PR に含めない）。再現:
`cargo test -p fandhe-ai-backend-metal --all-features --lib rejects_huge_broadcast_view`。

## 2. 高優先（Phase 2 機能 parity）結果一覧

pass 数は各ログ末尾の `test result:` 行の実測値。`metal_` はテスト名フィルタ
（同一バイナリに CUDA 実機必須の `cuda_*` テストが混在するため）。

### 2a. Conv（#1771・親 #1645／#1606）→ `docs/perf/logs/conv-realdevice-1771/metal/`

| ログ（run1／run2 とも同結果） | pass | fail | 失敗テスト | 判定 |
|---|---|---|---|---|
| `im2col_col2im_parity.log` | 6 | 0 | — | 規則 1) PASS（bit 完全一致） |
| `conv2d_backend_parity.log` | 1 | 1 | `metal_conv2d_backward_matches_cpu` | forward PASS・backward は `sum` 未実装で判定不能 |
| `conv1d_backend_parity.log` | 2 | 2 | `metal_conv1d_backward_matches_cpu`・`metal_conv1d_matches_manual_reshape_conv2d_bit_exact` | forward（groups/dilation 込み）PASS・backward／bit 一致は `sum` 未実装で判定不能 |
| `nn_conv_backend_parity.log` | 5 | 1 | `metal_sequential_conv1d_matches_manual_reshape_conv2d_bit_exact` | nn 層 forward／backward（conv1d・conv2d）・record-only 学習ループは PASS |
| `check_determinism.log` | — | — | — | 規則 4) PASS（4 ログ群とも run-to-run bit 同一。ただし pass したテストが出力した `fold_bits=`／`bits=` 行のみが比較対象） |

**verdict = FAIL（規則 2) 未成立・テスト構造起因）**。規則 1)・4) は成立。
規則 2)・3) を担う 4 テストは、Metal tape 上の loss 縮約 `Var::sum` が
`MetalBackendOps::sum: reduction カーネル未実装` の `Unsupported` を返すため
比較に到達しない（§3.1）。conv 演算自体の数値不一致は 1 件も観測されて
いない（forward は全 pass）。

### 2b. BatchNorm（#1736）→ `docs/perf/logs/metal-batch-norm-1736/`

| ログ | pass | fail | 判定 |
|---|---|---|---|
| `batch_norm_parity.log`（backend-metal） | 9 | 0 | PASS（REQ-2・決定性含む） |
| `batch_norm_backend_parity.log`（facade `metal_batch_norm_train_forward_matches_cpu`） | 1 | 0 | PASS |

### 2c. Pooling（#1730・追従 PR #1888 配線済み）→ `docs/perf/logs/metal-pooling-1730/`

| ログ | pass | fail | 判定 |
|---|---|---|---|
| `pooling_parity.log`（backend-metal） | 6 | 0 | PASS（Max は bit 完全一致・Avg／Adaptive は soft-f64 で bit 完全一致） |
| `facade_pooling_backend_parity.log`（本ディレクトリ。`metal_` 3 件） | 3 | 0 | PASS（facade → `MetalBackendOps` 配線経由） |

### 2d. bit 一致契約の演算群

| ログ | pass | fail | 失敗テスト | 判定 |
|---|---|---|---|---|
| `backend-metal_sort_topk_parity.log` | 4 | 0 | — | PASS |
| `backend-metal_scan_parity.log` | 2 | 0 | — | PASS |
| `backend-metal_gather_scatter_parity.log` | 19 | 1 | `gather_backward_matches_cpu_tape` | forward／scatter 系 PASS・backward は `sum` 未実装で判定不能 |
| `backend-metal_cast_parity.log` | 1 | 0 | — | PASS |
| `backend-metal_unique_parity.log` | 1 | 0 | — | PASS |
| `backend-metal_constant_pad_parity.log` | 4 | 1 | `pad_backward_matches_cpu_tape` | forward PASS・backward は `sum` 未実装で判定不能 |
| `backend-metal_interpolate_parity.log` | 4 | 2 | `interpolate_backward_matches_cpu_tape`・`interpolate_bilinear_backward_matches_cpu_tape` | forward（nearest bit・bilinear REQ-2）PASS・backward は `sum` 未実装で判定不能 |
| `backend-metal_gemm_batched_parity.log` | 13 | 0 | — | PASS |
| `facade_sort_topk_backend_parity.log` | 2 | 0 | — | PASS |
| `facade_index_ops_backend_parity.log`（where／masked_fill／narrow／one_hot） | 4 | 0 | — | PASS |
| `facade_cast_backend_parity.log` | 1 | 0 | — | PASS |
| `facade_unique_backend_parity.log` | 1 | 0 | — | PASS |
| `facade_constant_pad_backend_parity.log` | 1 | 0 | — | PASS |
| `facade_interpolate_backend_parity.log` | 2 | 0 | — | PASS |
| `facade_batched_matmul_backend_parity.log`（forward／backward） | 2 | 0 | — | PASS |

### 2e. REQ-2 統一複合判定の演算群（facade・`metal_` フィルタ）

| ログ | pass | fail | 失敗テスト | 判定 |
|---|---|---|---|---|
| `backend-metal_scalar_op_parity.log` | 11 | 0 | — | PASS |
| `facade_scalar_ops_backend_parity.log` | 6 | 0 | — | PASS |
| `facade_scalar_unary_transcendental_backend_parity.log` | 3 | 0 | — | PASS |
| `facade_activation_gelu_softplus_backend_parity.log` | 3 | 0 | — | PASS |
| `facade_bce_backend_parity.log` | 1 | 0 | — | PASS |
| `facade_nll_kl_div_backend_parity.log` | 3 | 0 | — | PASS |
| `facade_huber_backend_parity.log` | 1 | 0 | — | PASS |
| `facade_dropout_backend_parity.log` | 1 | 0 | — | PASS |
| `facade_attention_backend_parity.log` | 1 | 1 | `metal_sdpa_backward_dq_matches_cpu` | forward PASS・backward は `sum` 未実装で判定不能 |
| `facade_mha_backend_parity.log` | 2 | 0 | — | PASS |
| `facade_compat_sequential_layers_backend_parity.log`（norm＋Embedding／BatchNorm2d／MHA） | 3 | 0 | — | PASS |
| `facade_no_grad_detach_backend_parity.log` | 0 | 1 | `metal_detach_weight_grad_matches_cpu` | `sum` 未実装で判定不能（detach 機構自体の不一致は未観測） |
| `facade_backward_accumulate_backend_parity.log` | 0 | 1 | `metal_backward_accumulate_weight_grad_matches_cpu` | `sum` 未実装で判定不能（accumulate 機構自体の不一致は未観測） |
| `facade_device_transfer_backend_parity.log` | 1 | 0 | — | PASS |
| `facade_device_enumeration.log`（`#[ignore]` なし・常時実行） | 5 | 0 | — | PASS（Metal デバイス列挙成立） |

### 2f. TypedOps（#1705／#1706）

| ログ | pass | fail | 判定 |
|---|---|---|---|
| `backend-metal_typed_ops_f16_parity.log` | 4 | 0 | PASS（REQ-2） |
| `backend-metal_typed_ops_bf16_parity.log`（`metal-typed-bf16-probe-1706/typed_ops_bf16_parity.log` へ複製） | 2 | 0 | PASS（REQ-2） |
| `metal-typed-bf16-probe-1706/typed_bf16_probe.log` | 5 | 0 | 非 gating。P0〜P4 の記録は同ディレクトリ README・`docs/backend-dtype-dispatch-design.md` §15.6 |

## 3. FAIL の分析

### 3.1 Metal `sum` 未実装に起因する判定不能（11 テスト）

対象: §2a の 4 件・§2d の 4 件・§2e の 3 件（`*_backward_matches_cpu_tape`・
`metal_*_backward_*`・`*_matches_manual_reshape_conv2d_bit_exact`・
`metal_detach_weight_grad_matches_cpu`・`metal_backward_accumulate_weight_grad_matches_cpu`）。
panic メッセージはすべて
`Backend(Unsupported("MetalBackendOps::sum: reduction カーネル未実装（TASK-1.9c スコープ外）"))`。

- 機構: これらのテストは Metal tape（`Tape::new_with_ops(Box::new(MetalBackendOps::new()))`
  または `fandhe_ai::tape_for(Device::Metal(0))`）上で `Var::sum(None)` を
  スカラー loss として backward を起動する。`Var::sum`
  （`crates/autodiff/src/var.rs:918`）は `self.tape.ops().sum(..)` の結果を
  そのまま `?` で返し、`min_with_fallback`／`cumsum` のようなホスト
  フォールバックを持たない。一方 `MetalBackendOps::sum`
  （`crates/backend-metal/src/ops.rs:2644`）は設計どおり常に `Unsupported`
  を返す（`docs/backend-dtype-dispatch-design.md` §14.2「`sum`／`max` は
  `Unsupported` を継承する」・Metal f32 reduction カーネルは未実装で
  スコープ外として記録済み）
- 帰結: 11 テストは Linux 上で実機到達なしに作成され「Metal tape 上で
  `sum` が成功する」前提を置いていた。比較段階に到達していないため
  **演算自体の数値不一致は 1 件も観測されていない**（各演算の forward
  parity・同じ演算の `mse_loss` 経由 backward〈`nn_conv_backend_parity` の
  `metal_sequential_conv{1d,2d}_backward_matches_cpu`〉は pass）。
  判定としては FAIL（判定不能）のまま記録し、規則の事後緩和は行わない
- 公開面への含意: `tape_for(Device::Metal(0))` 上の `Var::sum` は現状
  `Unsupported` で失敗する（CUDA・CPU は成功）。是正候補（Metal reduction
  カーネル実装／`Var::sum` のホストフォールバック／テスト側の loss を
  `mse_loss` 等へ書き換え）はいずれもコード変更であり本 PR の対象外。
  `out-of-scope-tracking.md` に従い、切り出し先の起票可否はユーザー判断へ
  回す（本 PR では起票しない）

### 3.2 split-K 自動判定入口 `auto_entry_falls_back_to_classic_not_eligible_for_non_split_k_shapes`（#1513・1 テスト）

`docs/perf/logs/metal-gemm-splitk-auto-entry-1513/auto_entry.log`。事前登録
規則 2(2) の **FAIL**。`(512,512,512)` は `Classic{NotEligible}`＋CPU 参照
bit 一致を通過したが、`(64,64,63)` で `dispatch_split_k_strided_prepared` が
`strided tiled GEMM route ineligible: m/n/k must all be multiples of 8` の
`Err` を返し panic した。`dispatch_split_k_strided_prepared`
（`crates/backend-metal/src/gemm.rs`）は split-K 判定・Classic フォールバック
より前に `strided_tiled_eligibility`（8 の倍数検査）を通すため、k=63 の
fixture では `SplitKRoute::Classic` が構造的に到達不能であり、テストの
期待（NON_ELIGIBLE 形状は必ず Classic）と実装契約が食い違っている
（#1513 で実機未測のまま作成された fixture の前提不成立）。positive 側
`auto_entry_dispatches_split_k_for_eligible_shapes_and_matches_baseline`
（11 形状 × 4 パターン）は pass。本番経路 `dispatch_auto`／
`MetalBackendOps::gemm` は別ルーティングで、`cpu_metal_parity`・
`gemm_strided_parity`（15 pass）等は全 pass。「8 の倍数でない形状を
`Err` ではなく `Classic` に分類すべきか」は入口の契約に関する判断で
ユーザーへ回す。`run_auto_entry.sh` は `set -e` により本 FAIL で中断した
ため（`uptime_during.log` は 1 本目のみ）、`bit_match.log`／`parity.log` は
スクリプト記載と同一コマンドを手動実行して補完した（2 pass・1 pass）。

### 3.3 `command_batching::pool_reuse_zero_fill_does_not_synchronize_open_batch`

§1 参照（並列干渉。直列 3/3 pass）。

## 4. 実測していない項目

- CUDA（DGX Spark GB10）側の同名テスト群（`cuda_*`）はすべて対象外
- #1693／#1694（thread_elements A/B）は指示により未着手
- 中優先（#1691／#1580／#1562／#1563／#1696）は別 PR
