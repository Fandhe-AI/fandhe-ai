# サンプリング版 speculative decoding（#3102）CUDA／Metal 実機 parity 実測記録

`docs/facade-speculative-decoding-batching-design.md` §20.7・§27 参照。
`crates/facade/tests/speculative_batching_backend_parity.rs` のうち CUDA（`CudaBackendOps`）・
Metal（`MetalBackendOps`。`cfg(target_os = "macos")` 限定）を対象とする `#[ignore]` 2 テストを、
2026-10-27（UTC）に GB10（`<cuda-node>`）と Mac mini（`<metal-node>`）の実機で実測した。
2026-10-27 の reopen（所有者判断）で、これらを測定対象に加えることが確定している（設計 §27）。

## 測定コマンド

`.rev-stamp` で転送リビジョンを突合したうえで、同期ツリー外の `CARGO_TARGET_DIR` を使って実行した。
`--exact` で対象を 1 本に絞る（CUDA テストは cfg が付かず Mac でもコンパイルされるため、0 本実行を pass と誤認しない）。

```sh
# CUDA（GB10 実機上。debug ビルド）
env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
  cargo test -p fandhe-ai --test speculative_batching_backend_parity \
  cuda_speculative_sampling_matches_cpu -- --ignored --exact --nocapture

# Metal（Mac mini 実機上。zsh -lc で PATH を通す。debug ビルド）
CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
  cargo test -p fandhe-ai --test speculative_batching_backend_parity \
  metal_speculative_sampling_matches_cpu -- --ignored --exact --nocapture
```

CPU（`CpuBackendOps`）対 NaiveOps は属性なしテスト `cpu_backend_ops_matches_naive_ops_for_speculative_sampling`
（と再生ハーネスの検出力を確かめる `replay_comparison_detects_tampering`）で CI 実行される。

## 方式

- 設計 §20.7 の推奨方式。参照側（CPU）で `generate_speculative` をサンプリング設定
  （`Temperature(0.8)`・`TopK(3)`）で実行し、target・draft の `forward_step` 入力 id 列を記録する。
  比較側（実機）のモデルへ同じ id 列を再生し、呼び出しごとの logits を比べる。検証 forward（`L = k + 1`）と
  巻き戻し後の再 forward もそのままデバイスで通す
- 判定は呼び出しごとの logits に対する REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
  `fandhe_ai_backend_cpu::parity::assert_parity`）。呼び出し数・`seq_before`・ids・shape の一致も確かめる
- **token 列の一致は比べない**（契約でも仮説 H1 でもない。サンプリングの CDF 境界で logits の微差により反転しうる）
- 新規カーネルは無く、tolerance・baseline・係数は新設・変更していない

## 実測記録

| 日付（UTC） | 実機 | 結果 | 備考 |
|---|---|---|---|
| 2026-10-27 | GB10（`<cuda-node>`・CUDA 13.0・rustc 1.97.0） | `running 1 test`・`cuda_speculative_sampling_matches_cpu ... ok`・`1 passed; 0 failed; 9 filtered out`・5.18s・終了コード 0 | 測定 sha `343d88ee7e7a1cbedb10f9ace03658963483a6be`（ノード側 `.rev-stamp` と一致） |
| 2026-10-27 | Mac mini（`<metal-node>`・Apple M4・rustc 1.99.0） | `running 1 test`・`metal_speculative_sampling_matches_cpu ... ok`・`1 passed; 0 failed; 13 filtered out`・17.36s・終了コード 0 | 測定 sha 同上（ノード側 `.rev-stamp` と一致） |

## 注記

- テストは `SpecStats` を出力しない。pass は、ハーネス内部の掃引の assert（`runs >= 256`・`verify_calls > 0`・`replays > 0`）と、
  呼び出しごとの `assert_parity` がすべて成立したことを意味する。件数は記録していない（stats を出すためのテスト書き換えはしていない）
- 実機ロック（`~/work/.fandhe-hw-lock`）の保持時間: GB10 は 10:46:19〜10:48:01、Mac mini は 10:46:20〜10:48:02（いずれも UTC）
- GB10 には常駐サービス（vLLM・ComfyUI。計 3 プロセス）があり、実行前後とも `utilization.gpu` は 0 %。常駐サービスは停止していない
- 代替機（Linux 機・M4 Max 開発機）の値は使っていない。tolerance 定数・baseline・判定規則・テストコードは変更していない
- 転送前に、ローカル CPU で `speculative_sampling_facade`（7 件）と `speculative_batching_backend_parity`（属性なし 6 件）が pass することを確認した
