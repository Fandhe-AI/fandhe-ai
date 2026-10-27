# GB10 で同一 GPU を共有する 2 プロセスの分散学習（#3149）実測記録

親 #3143。DDP 初期形（ホスト側 std TCP の ring all-reduce。`docs/facade-distributed-host-collective-decision.md`）は、#3148 で
CPU・loopback・複数プロセスの正しさを CI で確認済み。本ディレクトリは CUDA 版の実機確認の記録枠である。
手元の CUDA 実機は GB10（GPU 1 基）だけで複数 GPU の DDP は検証できないため、同一 GPU（`Device::Cuda(0)`）の上で
2 プロセスがそれぞれ自分の CUDA コンテキストを持ち、勾配をホスト経由で all-reduce する経路を確かめる。

**2026-10-27 に GB10 で実測した。独立 5 run すべてで `RULE.txt` の合否規則を満たした（pass）**。6 比較（SGD・Adam・AdamW × `single_dev`・`single_cpu`）は全 run で `fail=0`、
子プロセスは正常終了、elapsed は 1.5〜1.6 秒、待受は `127.0.0.1` のみ。事前宣言は同ディレクトリの `RULE.txt`（実測前に固定し、事後に緩めない。本実測でも変更していない）。対象外: 性能（スケーリング効率・格上げ条件 (f)）・NCCL・複数 GPU。

## 対象テスト

| テスト（`crates/facade/tests/distributed_cuda_multiprocess_real_device.rs`） | 意図 |
|---|---|
| `cuda_two_process_ddp_on_shared_gpu_matches_single_cuda_and_cpu`（`#[ignore]`） | W=2・`Device::Cuda(0)` 共有。SGD・Adam・AdamW × 5 ステップの最終パラメータを、1 プロセス CUDA（`single_dev`）・1 プロセス CPU（`single_cpu`）と REQ-2 統一複合判定で比較する |
| `harness_self_check_on_cpu_device_two_process` | 同じハーネスを `Device::Cpu` で回す CI 用の自己検査（配線の誤りを実機前に捕まえる） |
| `child_rank_entry_device` | 子プロセスのエントリ。環境変数の合図が無ければ no-op（`#[ignore]` なし） |

待受は loopback のみ。リング待受の IP は集合点との接続の `local_addr` から導かれる（記録 §13.2）ので loopback になる。
テストは `is_non_loopback_allowed() == false` と実アドレスが loopback であることを assert し、`rdv_addr` を出力する。

## 既知のリスク

分散ステップは `mean_reduce_grads` が合成した `Gradients`（`resident_fingerprint = None`）を使うため、**非 resident の step 経路**を通る
（記録 §15）。一方 `single_dev` は resident 経路を通る。既存の CUDA 実機テストがホスト側の合成 `Gradients` から CUDA の
`DeviceParamStore::step*` を通しているかは本実装では確定できていない（`crates/facade/tests/distributed_facade.rs` は CPU）。
通していなければ #3149 がこの経路を CUDA で通す最初のテストになり、fail の原因は collective ではなく step 経路の可能性がある。
切り分けには `ref_single_dev_vs_single_cpu` 行（判定対象外）を使う。

## 測定コマンド（`$CUDA_NODE` 形式。`docs/real-hardware-verification-env.md` §3・§4・§6.1 に従う）

```sh
# 1. 転送（§3 の除外リストどおり。.env*・settings.local.json・.local.md は渡さない）。.rev-stamp を作ってから rsync し、
#    ssh "$CUDA_NODE" 'cat ~/work/rust-ai-library-run/.rev-stamp' で転送リビジョンを確認する。
# 2. 事前にビルドだけ済ませる（計測中はビルドを並走させない）
ssh "$CUDA_NODE" 'cd ~/work/rust-ai-library-run && env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai cargo test -p fandhe-ai --release --test distributed_cuda_multiprocess_real_device --no-run'
# 3. 実行前後に GPU の使用状況を記録する（実行中に 1 回取れれば、同じ GPU に 2 つの PID が載っている証跡になる）
ssh "$CUDA_NODE" 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
# 4. 本実行（独立プロセスで 5 回）
ssh "$CUDA_NODE" 'cd ~/work/rust-ai-library-run && env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai cargo test -p fandhe-ai --release --test distributed_cuda_multiprocess_real_device -- --ignored --nocapture cuda_two_process'
# 5. env_info.txt（GPU・driver・CUDA・rustc）を取る
```

## ログ書式（`--nocapture` の出力）

```
DDP_GB10 rdv_addr=<ip:port> loopback=true non_loopback_allowed=false
DDP_GB10 device=<cuda0|cpu> opt=<sgd|adam|adamw> vs=<single_dev|single_cpu|ref_single_dev_vs_single_cpu> total=<n> fail=<k> max_abs=<v> max_rel=<v> mean_abs=<v>
DDP_GB10 bits_equal kind=<rank0_vs_rank1|runA_vs_runB> opt=<..> value=<true|false>
DDP_GB10 device=<..> elapsed_s=<秒>
```

## 手順

1. 実測前に `RULE.txt` をコミットして固定する（本 PR で固定済み）。
2. 上記コマンドを独立プロセスで 5 回実行し、全出力を保存する。計測中はビルドや別の GPU 負荷を並走させない。
3. 合否は「5 回すべて pass で pass」。`DDP_GB10` 行の数値は 5 回の中央値を記録し、不一致があれば事実として記録する。
4. ログは本ディレクトリへ置く。cargo や panic のメッセージには絶対パスが出るので、収録前に sed で `$HOME`・ユーザー名・ホスト名を
   `<home>`・`<user>`・`<cuda-node>` へ置換する。置換後に、ログ内に `$HOME` の絶対パス（Linux・macOS のホーム配下）が残っていないことを確認する。`.rev-stamp` を同梱する。
5. 下の記入欄を埋め、`docs/README.md` の `logs/ddp-host-collective-gb10-3149/` の行へ要約を追記する。

## fail したとき

- tolerance 定数・判定式・カーネルは変えない（`.claude/rules/coding-rust.md`）。baseline 切り替えも行わない。
- `DDP_GB10` 行の `opt`・`vs`・`total`・`fail`・`max_abs`・`max_rel`・`mean_abs` を転記して事実を記録する。
- 原因調査は別 issue として所有者へ上げる（起票の可否は所有者が判断する。`.claude/rules/out-of-scope-tracking.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-27 | DGX Spark GB10（<cuda-node>・sm_121。driver 580.173.02・CUDA 13.0.88・rustc 1.97.0） | 上記 4（独立 5 run。`run1.log`〜`run5.log`） | **pass**（5 run × 6 比較すべて `fail=0`。各 run の子プロセス rc=0・`1 passed; 0 failed`・elapsed 1.5〜1.6 秒） | 対象 `578e820d2`（dirty=0。`.rev-stamp`）。hw-lock 保持 2026-10-27 12:16:32Z〜12:17:29Z（UTC）。下の「実測値」参照 |

本行が先頭の「未実測」行を引き継ぐ。

## 実測値（5 run の中央値。`max_abs`・`max_rel`・`mean_abs`）

| vs | opt | max_abs | max_rel | mean_abs |
|---|---|---|---|---|
| single_dev | sgd | 1.490e-08 | 2.439e-07 | 1.472e-10 |
| single_dev | adam | 2.980e-08 | 1.869e-06 | 2.466e-10 |
| single_dev | adamw | 1.490e-08 | 4.225e-05 | 1.527e-10 |
| single_cpu | sgd | 1.490e-08 | 2.979e-07 | 2.372e-10 |
| single_cpu | adam | 2.980e-08 | 1.869e-06 | 2.466e-10 |
| single_cpu | adamw | 1.490e-08 | 4.225e-05 | 1.527e-10 |
| ref_single_dev_vs_single_cpu（参考・判定対象外） | sgd | 1.490e-08 | 2.439e-07 | 9.225e-11 |
| 同上 | adam | 0 | 0 | 0 |
| 同上 | adamw | 0 | 0 | 0 |

`total=212`（全要素数）・`fail=0` は全 run 共通。`bits_equal`（`rank0_vs_rank1`・`runA_vs_runB`）は全 run・全 optimizer で `true`（記録のみ。判定には使わない）。

## 実測の付記

- loopback: 全 run で `rdv_addr=127.0.0.1:<port> loopback=true non_loopback_allowed=false`（ポートは run ごとに異なる）。ログ内の IPv4 は `127.0.0.1` のみ。
- 2 PID の証跡: `gpu-apps-run*.txt`（1 秒間隔の `nvidia-smi --query-compute-apps`）に、テストバイナリの異なる 2 PID が同一時刻に載った行が各 run で 1 サンプルある。run が 1.5 秒程度と短いため取れたサンプルは 1 回だけで、`used_memory` は取得できた（親 約 204 MiB・子 約 11 MiB は run1 の例）。
- 常駐サービス: 計測前後の GPU 使用は常駐サービス 3 プロセス（<resident-service>。実名は公開しない）が計測前後とも載っていたが、計測前後の `utilization.gpu` は 0%。計測は hw-lock 保持中・ビルド完了後に行い、ビルドは並走させていない。
- 手順の差異: 転送の rsync に `--delete-excluded` を付けていない（openrsync が protect フィルタを無視してノード側の別ディレクトリを消すため。#3138 と同じ扱い）。`.claude/worktrees/`・`target/` を追加で除外した。サンプラーは run ごとに開始・停止し、`rc=` はパイプを挟まず `cargo test` の直後に取った。
- 不変: tolerance 定数・parity baseline・FMA 契約・カーネル・テストコード・`RULE.txt` は変更していない。
- ログの加工: ホームの絶対パスを `<home>`、常駐サービスのプロセス名を `<resident-service>` に置換した以外は無加工。
