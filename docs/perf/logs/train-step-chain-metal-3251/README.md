# 学習ステップ全体の encode-only チェーン（Metal）実機記録（イシュー #3251）

親 #3248「学習 1 step の同期を 1 回へ」。`DeviceParamStore::train_step_device_chain`
（`crates/autodiff/src/optim/device_store/train_chain.rs`）が、forward・MSE・backward・SGD を
`_tracked` 部品だけで 1 区間に積み、同期を loss の読み出し 1 回へ集約することを Mac mini で確かめた記録。
設計・公開形は `docs/compat-resident-train-step-decision.md` §11、カウンタの導出は
`docs/backend-metal-command-batching-design.md` §7.7。

## 判定規則

`RULE.txt`（事前登録。実測値を含まない時点で固定）。A1・A2 は Tape 経路との bit 一致、A3 は wait 0／1・encode 10。

| 行 | テスト（`fandhe-ai` の lib・`--release`・`--ignored --exact`） | 内容 |
|---|---|---|
| A1 | `metal_chain_matches_tape_path_bit_exact_two_layers` | 784-256-10・batch 64・momentum 0／0.9（dampening・weight_decay）／0.9（nesterov）・各 4 step |
| A2 | `metal_chain_matches_tape_path_bit_exact_three_layers` | 784-128-64-10・batch 64・momentum 0.9／0（L2 以降の d_input の連鎖） |
| A3 | `metal_chain_step_issues_exactly_one_wait` | warmup 3 step の後の 1 step の encode／command_buffer／wait |

## 手順

1. `docs/perf/logs/metal-infer-bimodal-macmini-3183/stage-tree.sh <commit> <dir>` で展開 → rsync
   （`.git`・`.env*`・`real-hardware-verification-env.local.md` を除外）→ ノード側で `.tree-manifest` を
   `shasum -a 256 -c` で照合。
2. ビルドを 1 回だけ（`build.log`）→ ノード側で `run_chain.sh <out>` を実行（A1〜A3 を各 5 run）。
3. ログ・env・`.rev-stamp` を回収し、ホスト名・ユーザー名・絶対パスを `<metal-node>`・`<home>` へ置換。

## 結果（Mac mini・Apple M4・2026-10-24）

測定ツリー: 作業中コミット `f15af2a987783ef9af414cf35639886e947617e7`（`.tree-manifest` 12047 件を全件照合・非 OK 0 件。
このコミットは最終コミットへ squash したため履歴には残らない）。
測定後に変えたのは `docs/` のみで、`crates/`・`Cargo.toml`・`Cargo.lock` は同一。耐久的な照合手段として
測定ツリーの `crates/` の tree hash を記録する: **`12e9049f12d5ca42738aaac049fbd66182286ac3`**
（`git rev-parse <コミット>:crates` で照合できる。squash マージでも内容が同じなら一致する）。
`.tree-manifest` は測定後に `docs/` が変わったため最終コミットからは再現しない（検証できるのは `crates/` 部分）。
invalid の run は 0 件。

| 行 | 5 run の結果 |
|---|---|
| A1 | 5 / 5 pass（loss と全パラメータが Tape 経路と bit 一致） |
| A2 | 5 / 5 pass（同上） |
| A3 | 5 / 5 pass。全 run で `chain: encode=10 command_buffers=1 wait=0 / read: wait=1` |

- 学習 1 step の `waitUntilCompleted` は 7 回（従来の `Tape` 経路。`docs/perf/train-step-phase-breakdown.md` §18.3）から
  **1 回**（loss を読む download）へ減った。チェーン区間は wait 0。
- encode 10 は机上導出（forward 2 + MSE forward 2 + MSE backward 1 + dW 2 + d_input 1 + ReLU マスク 1 + SGD 1）と一致。
- 性能（0.65〜0.9 ms の目安）は本 issue の範囲外で未計測。

## 不一致

なし。

## 同梱物

`macmini/`: `.rev-stamp`・`tree-manifest-check.txt`・`env_info.txt`・`build.log`・`driver.log`・
`A{1..3}.run{1..5}.log`（末尾に `rc=`）・`DONE`。
