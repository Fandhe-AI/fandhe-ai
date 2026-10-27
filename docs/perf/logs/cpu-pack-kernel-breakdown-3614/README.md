# 6 形状の A pack・B pack・kernel 時間内訳の計測部品と規則（イシュー #3614）

NEON 直接カーネル (A) を再検討するかの前提データ（pack が主因か）を測るための事前登録規則・診断パッチ・ランナー・集計。
決定記録は [`docs/perf/cpu-gemm-neon-direct-decision.md`](../../cpu-gemm-neon-direct-decision.md) §7.1。

## 状態

- **実測済み（2026-10-27）**。GB10 と Mac mini で 5 round 完走（負荷ゲート各 20/20 pass、`DONE` あり、`run.out` に完走を記録）。ログは `gb10/`・`macmini/`、集計は `aggregate.md`（終了コード 0 は `aggregate.exit`）。
- 判定（native 腕）: GB10 **NOT_PACK_MAIN**（S_A 中央値 0.1492・0/5）、Mac mini **NOT_PACK_MAIN**（S_A 中央値 0.1737・0/5）。詳細と注記（L2 の分解能不足・摂動）は決定記録 §7.1。
- 所有者判断済み（2026-10-28。出典: https://github.com/Fandhe-AI/fandhe-ai/issues/3614#issuecomment-6445943222）: 両機体とも NOT_PACK_MAIN のため NEON 直接カーネル (A) は再検討しない。本ディレクトリの計測値・`RULE.txt` は変えていない。

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
実際に使った形: ノード側で `CARGO_TARGET_DIR` が export されているとバイナリ位置（`${BASE}/target`・`${TIMED}/target`）と食い違い、base と timed が target を共有するため、`env -u CARGO_TARGET_DIR -u RAYON_NUM_THREADS` で包んで起動した。BASE は `git archive` の展開物（`.git` なし）、WORK は BASE の外。
3. 回収したログのホーム絶対パスをマスクし、`python3 -I aggregate.py <machine_dir>... > aggregate.md` を実行する。fail-closed で止まった場合は規則を緩めず、その状態を決定記録へ記す。
4. 診断パッチの健全性（開発機での確認のみ・記録には使わない）: `git apply --check`、timed／base の release ビルド、`bits_hash` の 6 形状＋参照の完全一致。timed 腕のツリーでは `determinism_inventory` が通らない（診断用 `Mutex` のため。これが診断パッチを main に入れない理由）。

## 限界

`RULE.txt` §13 を参照（plain GEMM 入口であり `bias_act` 融合経路ではない・L1 dX が本番で実行されるかは未確認・kernel 区間は端タイルのコピーを含む）。
