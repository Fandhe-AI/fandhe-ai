# CUDA の学習チェーン（同期・確保・H2D 0）実機記録（イシュー #3252）

親 #3248「学習 1 step の同期を 1 回へ」。`DeviceParamStore::train_step_device_chain`
（`crates/autodiff/src/optim/device_store/train_chain.rs`）を CUDA で動かし、定常状態の区間内にホスト同期・D2H・
H2D・デバイス確保を持たず、同期が loss の読み出し 1 回だけになることを GB10 で確かめた記録。
設計・実装は `docs/compat-resident-train-step-decision.md` §12、部品の契約は
`docs/train-step-resident-parts-design.md` §3（#3252 追記）。

## 判定規則

`RULE.txt`（事前登録。実測値を含まない時点で固定）。

| 行 | テスト（`--release --lib --ignored --exact --test-threads=1`） | 内容 |
|---|---|---|
| B1 | `fandhe-ai-backend-cuda`: `train_step_chain_diag_tests_3252::train_chain_window_has_no_sync_alloc_or_h2d_and_loss_read_is_one_sync` | 784-256-10・batch 64・momentum 0.9。窓の `stream_syncs`・`d2h_calls`・`h2d_calls`・`mem_uploads`・`mem_allocs` が 0、loss 読み出しで同期 1・D2H 1 |
| B2 | `fandhe-ai`: `compat::sequential::train_step_chain_tests_3251::cuda::cuda_chain_matches_tape_path_bit_exact_two_layers` | 784-256-10・momentum 0／0.9（dampening・weight_decay）／0.9（nesterov）・各 4 step。Tape 経路と bit 一致 |
| B3 | `fandhe-ai`: `...::cuda::cuda_chain_matches_tape_path_bit_exact_three_layers` | 784-128-64-10・momentum 0.9／0（L2 以降の d_input の連鎖） |

## 手順

1. 作業ツリーを GB10（spark-dbd9）の専用ディレクトリへ rsync（`target/`・`.git`・`docs/spec` を除外。openrsync は
   protect フィルタを無視するため既存ツリーを触らない別ディレクトリへ置く）。
2. ビルドを 1 回だけ（`gb10/build.log`。判定外）。
3. `run_chain.sh <out>` で B1〜B3 を各 5 run（独立プロセス）。ログ・`env_info.txt` を回収し、`/home/<user>` を `<HOME>` へ置換。

## 結果（GB10・NVIDIA GB10・driver 580.173.02・2026-10-24）

- 測定ツリー: ベース `c9c25c7850bf6fa4773a472ac14fd5770829ee4c`（#3565 マージ後の main）に本変更（`crates/` の差分）を
  載せた未コミットの作業ツリー。変更の `git diff --cached -- crates | shasum -a 256` は
  `570f96cc93cd3918421e4aeb91cf88d68771c5dee29e0e8e9d213e5886d77f1d`（`.rev-stamp` にも記録）。測定後に変えたのは `docs/` のみ
  （B1 のテストから `kernel_launches >= 1` の対照断言 1 行を測定前に除いた。同カウンタは tiled 起動を数えないため。判定規則に影響しない）
- invalid の run は 0 件。5 run × 3 行・15 件すべて `rc=0`

| 行 | 5 run の結果 |
|---|---|
| B1 | 5 / 5 pass。全 run で窓 `{stream_syncs: 0, d2h_calls: 0, h2d_calls: 0, mem_uploads: 0, mem_allocs: 0}`、loss 読み出し `{stream_syncs: 1, d2h_calls: 1}`、初回 step `mem_allocs: 11`・`stream_syncs: 0` |
| B2 | 5 / 5 pass（loss と全パラメータが Tape 経路と bit 一致） |
| B3 | 5 / 5 pass（同上） |

- 学習 1 step の stream 同期は loss を読む 1 回だけ。窓内のホスト同期・pageable H2D・デバイス確保は 0。
- 計測前の GPU には常駐サービス（vLLM 2 プロセス・ComfyUI）の compute プロセスが居た（`env_info.txt`）。本記録の判定は件数と bit 一致で、
  時間を測らないため GPU 負荷に依存しない。fandhe 系の他プロセスは無い。
- カウンタの範囲外（cudarc `CudaSlice::Drop` の非 async-alloc 環境の同期）はコード監査で補った（決定記録 §12。GB10 は async-alloc が有効）。
- 性能（目安 0.18〜0.25 ms）は本 issue の受け入れ条件に含まれないため未計測。

## 不一致

なし。

## 同梱物

`gb10/`: `.rev-stamp`・`env_info.txt`・`build.log`・`B{1..3}.run{1..5}.log`（末尾に `rc=`）・`DONE`。
