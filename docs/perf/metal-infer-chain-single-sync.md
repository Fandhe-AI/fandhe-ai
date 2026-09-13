# Metal 推論 forward チェーン単一同期化（イシュー #1580）

設計 `docs/inference-chain-single-sync-design.md`（イシュー #1579）決定 9
の事前登録規則に従い、`Sequential::predict_resident`（推論 forward。
Metal）を「層境界ごとに D2H→H2D」から「入力を 1 回だけ `upload`・各層は
`linear_forward_device_tracked`（encode-only）・チェーン末尾で 1 回だけ
`download`」へ置き換えた変更の実測記入欄。

## 1. 実装記録

コア実装（チェーン経路本体）は #1688（PR #1788）で着地済み。本 PR（#1580）
が追加したのは実機テスト・A/B スキャフォールド・本 perf doc のみで、下記
以外のコード変更は行っていない。

| ファイル | 変更概要 | 帰属 |
|---|---|---|
| `crates/tensor-core/src/backend_ops.rs` | `BackendOps::linear_forward_device_tracked`（デフォルトメソッド。`token` を無視して `linear_forward_device` へ委譲） | #1688（PR #1788）で実装済み |
| `crates/autodiff/src/optim/device_store.rs` | `ChainStep` 型エイリアス（非 pub）・`DeviceParamStore::predict_device_chain`（`upload` 1 回・各層 `linear_forward_device_tracked`・末尾 `download` 1 回。決定 4 の 3 ケース状態遷移） | #1688（PR #1788）で実装済み |
| `crates/facade/src/compat/sequential.rs` | private ヘルパー `build_device_chain_steps` を追加し、`predict_resident` を「チェーン優先・`Unsupported` 検出でフォールバック」へ差し替え | #1688（PR #1788）で実装済み |
| `crates/backend-metal/src/ops.rs` | `linear_forward_device_impl(.., token: Option<&DispatchFailureCell>)` への切り出し・`linear_forward_device_tracked` の `encode_strided_bias_act_prepared_with_c_offset` への配線 | #1688（PR #1788）で実装済み |
| `crates/backend-metal/tests/linear_forward_device_parity.rs` | `linear_forward_device_tracked_matches_linear_forward_device_and_leaves_token_unset`（`#[ignore]`）を追加 | 本 PR（#1580） |
| `crates/facade/tests/infer_device_chain_metal.rs`（新規） | `cfg(target_os = "macos")` + 理由付き `#[ignore]`。新チェーン経路 vs 手動 per-op チェーン（公開 API のみで再現）の bit 同一・run-to-run bit 同一・`diagnostic_batch_counters()` によるディスパッチ数検証（`encode_calls` 差分 2・`wait_until_completed` 差分 1）・record-only ベンチ | 本 PR（#1580） |
| `scripts/bench/framework-compare/run_ab_infer_chain_metal.sh`（新規） | `bench-fandhe --task infer --device metal --mode {fresh,reuse}` を before/after 2 本の path patch ビルドで交互実行する record-only スクリプト | 本 PR（#1580） |
| `scripts/bench/framework-compare/judge_infer_ab.py`（新規） | `--task infer` 専用の自己完結 A/B 判定ツール（`compare_gemm_ab.py` は `--task train` 限定で `--phases` を受け付ける設計のため、他イシューと共有する同ツールは変更せず新設） | 本 PR（#1580） |

## 2. 事前登録判定規則（設計 決定 9 の転記）

- 判定セル: `bench-fandhe --task infer --device metal --mode reuse`（5 round・起動順反転）
- 対照（非判定・参考記録のみ）: `--mode fresh`
- 判定: 全 5 round 中央値の比（`after_median / before_median`）が `<= 1.00`、かつ checksum が全 round で完全一致（`==`）
  → ADOPT（`Sequential::predict_resident` のチェーン優先経路を維持）
- 上記いずれか不成立 → REJECT
- 判定セルの計測行が「ちょうど 5 件」でない・`warmup`/`iters` 不一致 等 → undetermined
- 事後緩和禁止（.claude/rules/coding-rust.md「テスト・ベンチ」節）

補助検証（実機 `#[ignore]` テスト。`crates/facade/tests/infer_device_chain_metal.rs`）:

- `predict_resident_device_chain_matches_manual_per_layer_chain_bit_exact_metal`: 新経路 vs 手動 per-op チェーン（公開 API のみで再現）の bit 完全一致
- `predict_resident_device_chain_is_run_to_run_bit_identical_metal`: run-to-run bit 同一
- `predict_resident_device_chain_dispatch_counters_metal`: `encode_calls` 差分 2・`wait_until_completed` 差分 1（層ごとの同期点が生じていないことの確認）

## 3. 実機実測（M4 Max）

**本セッションの実行環境には Apple Silicon 実機がないため、実測ログ自体は
生成できない**（`docs/perf/` 配下の他の多数の record-only ドキュメントと
同じ「記入欄のみ」の扱い）。後続の Mac セッションが以下を実行し、本節へ
転記すること。

### 3.1 実機 `#[ignore]` テスト

`infer_device_chain_metal.rs` 内の 4 テストは同一プロセス内の
`MetalContext` シングルトンカウンタを共有するため、`METAL_SINGLETON_LOCK`
（ファイル内 `static Mutex`）で Metal 実行区間を直列化している
（codex-review 指摘対応。`--test-threads` の指定有無に依らず
`predict_resident_device_chain_dispatch_counters_metal` のカウンタ差分
計測が他テストの dispatch に汚染されない）。

```sh
cargo test -p fandhe-ai-backend-metal --release --test linear_forward_device_parity -- --ignored --nocapture
cargo test -p fandhe-ai --release --test infer_device_chain_metal -- --ignored --nocapture
```

- `linear_forward_device_tracked_matches_linear_forward_device_and_leaves_token_unset`: 記入欄（pass/fail・実行ログ）
- `predict_resident_device_chain_matches_manual_per_layer_chain_bit_exact_metal`: 記入欄
- `predict_resident_device_chain_is_run_to_run_bit_identical_metal`: 記入欄
- `predict_resident_device_chain_dispatch_counters_metal`: 記入欄（`encode_calls` 差分・`wait_until_completed` 差分の実測値）
- `predict_resident_device_chain_bench_metal`（record-only）: 記入欄（`old_median_s`／`new_median_s`／`speedup_x`）

### 3.2 framework-compare A/B（`run_ab_infer_chain_metal.sh`）

```sh
AB_BEFORE_FACADE_PATH=/absolute/path/to/before/crates/facade \
AB_AFTER_FACADE_PATH=/absolute/path/to/after/crates/facade \
  bash scripts/bench/framework-compare/run_ab_infer_chain_metal.sh 1580
```

- `mode=reuse`（判定対象）: `before_median_s` / `after_median_s` / `ratio` / `checksum_exact_match` — 記入欄
- `mode=fresh`（対照）: 同上（参考記録） — 記入欄
- 総合判定（ADOPT／REJECT／undetermined）: 記入欄
- 生ログ・JSONL・env_info の保存先: `docs/perf/logs/metal-infer-chain-single-sync-1580/`（本 PR 時点では未作成。実測時に Mac セッションが作成する）

## 4. 出典

`docs/inference-chain-single-sync-design.md`（イシュー #1579・決定 9）・
`docs/inference-forward-fixed-cost-design.md`・`docs/perf/
linear-forward-device-gpu.md`・`docs/perf/infer-reuse-phase-breakdown.md`・
`docs/backend-metal-command-batching-design.md`
