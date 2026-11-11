# 診断 (3): <cuda-node> の MLP 3 セル（G-CPU-TRN・G-CPU-INF・G-CUDA-INF）（#4147）

## 状態

**道具立ての時点: 実測なし。** 本ディレクトリは測る前に固定する部品（変換スクリプト・起動キット・CUDA 純カーネル計測）だけを置く。
`RULE.txt` は道具立てのマージ後、その main の SHA を `tree_sha` にして**単独コミット**で固定する（計測より前）。
`round{i}/`・`env/`・`aggregate.md`・`aggregate.tsv` は <cuda-node> での実測で追加する。
実測値・分類・確度は、実測するまで書かない（推定値で埋めない）。**<cuda-node> に到達できる環境でのみ実測し、`Closes #4147` は実測と判定を含む最後の PR だけに付ける。**

雛形と規則の一般論は `docs/perf/logs/diag-0140-template/README.md`、式の定義は `docs/perf/loss-attribution-matrix.md` §10.1。
派生元は `docs/perf/logs/diag-0140-mlp-macmini-4146/`（Mac mini の CPU 2 セル）。

## 対象

| セル | 機体 | scoreboard の比 | 現分類 | 相手 | 窓 |
|---|---|---|---|---|---|
| G-CPU-TRN | <cuda-node> | 0.51 | 自分たちの制約（確度低） | PyTorch CPU | train size=64 |
| G-CPU-INF | <cuda-node> | 0.75 | 分類保留 | PyTorch CPU | infer size=64 |
| G-CUDA-INF | <cuda-node> | 0.51 | 自分たちの制約（確度低） | PyTorch CUDA（candle は参考） | infer size=64 |

出典: `docs/perf/logs/framework-compare-0.14.0-remeasure/scoreboard/unreached_0140.tsv`。分類の反映はこの issue では行わない（#4168）。

## 注意（読み方）

- tree は PR 後の main の SHA（RULE の `tree_sha`）。`git diff v0.14.0 <tree_sha> -- crates/backend-cpu/src crates/backend-cuda/src` が 0 件であることを RULE 固定前に確認し、0 件でなければ停止する（bench-fandhe は crates.io 版 `fandhe-ai =0.14.0` を使うため、workspace の `cpu_jobgrid_bench`・新設 example と同一のカーネルコードを測る前提）。
- scoreboard の train reuse は常駐チェーン（`train_step_resident`。第 1 層の入力勾配 dX を計算しない。#3245）。fresh の窓は L1 の dX を計算するので、F2 の fresh 窓にはその余分な仕事が含まれる。
- 純カーネルは GEMM のみ（ReLU・MSE・SGD・readout・bias は両側とも含まない）。学習は l1_fwd・l2_fwd・l2_dw・l2_dx・l1_dw の 5 形状（l1_dx は含めない）、推論は l1_fwd・l2_fwd。
  - CPU 2 セル: 時計 `wall`（単発 GEMM を温めた状態で測った値の和。step 内の状態とは一致しない）。
  - G-CUDA-INF: 時計 `cuda_event`（event 対で 1 launch を挟む。H2D・出力確保・event 生成は区間外）。fandhe 側は `crates/backend-cuda/examples/cuda_mlp_event_bench_4147.rs`（`CudaGemm::launch_tiled_f32`＝本番と同じ `select_tiled_f32_kernel` 選択の純 GEMM。epilogue 融合は含まない）、相手側は `opp_kernel_cuda_mlp_diag_4147.py`（cuBLAS。既存の `opp_kernel_cuda_diag.py` は変更せず importlib で再利用）。
- `kernel_owner=hardware` は測る前の宣言（相手の GEMM が閉源ベンダーライブラリ〈CPU は NVPL、CUDA は cuBLAS〉で、REQ-1 により fandhe-ai は使わない）。`docs/perf/logs/cpu-gemm-resident-reuse-3646/` は G-CPU-INF を「自分たちの制約（条件つき）」と記録しており、**ラベルが食い違う**。kernel_owner は F1 ≥ 1 のときの表示ラベルだけを決め、F1・F2 の値は常に記録されるため、所有者が読み替えても再計測は不要。所有者確認が望ましい。
- ゲートは 0.14.0 GB10 と同一（compute プロセス 0 件、GPU 使用率 0 が 3 サンプル連続、load1 < 1.0、30 秒間隔で最大 30 分）。不通過は TIMEOUT を `gate.txt` に書き、そのラウンドを正式系列から外す（確度は常に低）。
- ノイズ帯 NB-RANGE5 は `docs/perf/noise-band-decision.md`（案・所有者承認待ち）。承認後に定義が変わったら RULE.v2.txt で反映する。
- record_only（分類に使わない）: `bench-fandhe --phases`（CPU train／infer、CUDA infer の reuse）。

## ファイル

| ファイル | 役割 |
|---|---|
| `RULE.txt` | 測る前に固定する規則（道具立てのマージ後に単独コミットで追加） |
| `RULE.template.txt`・`aggregate_diag.py` | 雛形の無改変コピー（sha256 を RULE の PARAMS に固定）。`aggregate_diag.py` は #4208 で時計不一致のセルを分類保留にする修正後の雛形に合わせ、#4237 で時計が揃っていても純カーネルが窓を上回るラウンドを止めずに F1 を比較不能とする修正後の雛形（所有者判断 2026-11-11）に合わせた（雛形と byte 一致。RULE は未固定のため再登録は不要） |
| `to_values_4147.py` | `raw/` → `values.tsv` の変換（厳格パース・`--self-test`） |
| `run_round.sh` | 1 round の起動（ゲート → 計測 → `gate.txt`・`raw/`） |
| `opp_kernel_cuda_mlp_diag_4147.py` | PyTorch(cuBLAS) の MLP 推論 2 形状を CUDA event で測る（`--self-test`） |

## 手順（実測・<cuda-node>）

1. 送信側で `.rev-stamp` を作り rsync で転送する（`docs/real-hardware-verification-env.md` §3）。ノード側の `.rev-stamp`・tree_sha・`--fingerprint` が RULE と一致することを計測前に確認する（1 件でも違えば停止）。同 §4.11 を適用してからビルドする。
2. 実機ロックを取得する。系列前に 1 回だけビルド: framework-compare の `bench-fandhe`（`--locked`）、`cargo build --release -p fandhe-ai-backend-cpu --example cpu_jobgrid_bench`、`cargo build --release -p fandhe-ai-backend-cuda --features internal-diagnostics --example cuda_mlp_event_bench_4147`。compute プロセス 0 件を確認して記録する。
3. `ROOT=… PY=… BP=… EXPECT_SHA=… HW_LOCK_OWNER_FILE=… HW_LOCK_HELD=1 bash run_round.sh <i> <記録先>/round<i>` を 1〜5 で回す。run の差し替え・除外・上書きは禁止。失敗したラウンドは別ラベルで取り直し旧系列を残す。到達不能な量は `unavailable` と記録する。
4. 各 round の後に `python3 -I to_values_4147.py --round-dir <記録先>/round<i>`。
5. 回収時にパス・ホスト名・ユーザー名を `<home>`・`<cuda-node>` に置換し、`env/` にツリー指紋・`torch.__config__.show()`・BLAS 経路（`blas_probe/run_probe.sh` を計測外で実行）・`.rev-stamp`・`nvidia-smi` 概要を置く。
6. `python3 -I aggregate_diag.py --dir <記録先>` で `aggregate.md`・`aggregate.tsv` を生成する。
7. 結果欄に 3 セルの F1／F2 の run 別値と中央値、観測比と scoreboard の比のドリフト（10% 超は記録）、分類、確度、ノイズ帯の判定を書く。F1・F2 ともに閾値未満の「分類保留」はそのまま記録する（後続は #4149・#4150）。
8. 実機ロックを解除する（失敗時も）。

## 結果

（実測後に追記。道具立て時点は空。）
