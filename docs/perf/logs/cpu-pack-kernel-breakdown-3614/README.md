# 6 形状の A pack・B pack・kernel 時間内訳の計測部品と規則（イシュー #3614）

NEON 直接カーネル (A) を再検討するかの前提データ（pack が主因か）を測るための事前登録規則・診断パッチ・ランナー・集計。
決定記録は [`docs/perf/cpu-gemm-neon-direct-decision.md`](../../cpu-gemm-neon-direct-decision.md) §7.1。

## 状態

- **未実測**。本ディレクトリは計測部品と事前登録規則だけを収録している。実機（GB10・Mac mini）の計測ログ・`aggregate.md`・判定は未記録。
- (A) を再検討するかの所有者判断は未記録（所有者確認待ち）。

## 収録物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に固定。計測後に変えない） |
| `diag_timing.patch` | 区間計時の診断パッチ（`gemm_blis/diag_pack_timer.rs` の新設・`run_two_d_job`／`gemm_blis_ic_loop` の 4 か所・example の追加出力）。計測用の複製ツリーにだけ当て、main へは当てない。sha256 は `patch_sha256.txt` |
| `run_breakdown.sh` | 機体上のランナー（`--dry-run` あり。bash 3.2 互換。負荷ゲート閾値は Mac mini 5.0・GB10 10.0） |
| `aggregate.py`・`aggregate_test.py` | fail-closed 集計（stdlib のみ）と合成 fixture のテスト（CI の python unittest ステップで実行） |
| 計測 example | `crates/backend-cpu/examples/cpu_pack_kernel_breakdown.rs`（base 腕。bit ハッシュと wall のみ） |

## 手順

1. 計測部品のコミットを `.rev-stamp`（HEAD とツリー指紋）に書き、専用ディレクトリへ rsync する（`.gitignore` フィルタ・`.env*`・`*.local.md` 除外）。
2. 両機体で実機ロック（`~/work/.fandhe-hw-lock`）を取り、保持している間に `BASE=… WORK=… bash run_breakdown.sh <macmini|gb10>` を実行する（Mac mini は `zsh -lc`・切り離しは nohup のサブシェル。2 機体の計測をビルドと並走させない）。ホスト名は Git 管理外の `docs/real-hardware-verification-env.local.md` を参照し、収録物へ書かない。
3. 回収したログのホーム絶対パスをマスクし、`python3 -I aggregate.py <machine_dir>... > aggregate.md` を実行する。fail-closed で止まった場合は規則を緩めず、その状態を決定記録へ記す。
4. 診断パッチの健全性（開発機での確認のみ・記録には使わない）: `git apply --check`、timed／base の release ビルド、`bits_hash` の 6 形状＋参照の完全一致。timed 腕のツリーでは `determinism_inventory` が通らない（診断用 `Mutex` のため。これが診断パッチを main に入れない理由）。

## 限界

`RULE.txt` §13 を参照（plain GEMM 入口であり `bias_act` 融合経路ではない・L1 dX が本番で実行されるかは未確認・kernel 区間は端タイルのコピーを含む）。
