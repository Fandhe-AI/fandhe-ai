# <cuda-node> の G-CUDA-INF の kernel_wait を GPU 実行と待ちに分ける診断（#4150）

## 状態

**PR-1 時点: 実測なし。** 本ディレクトリは測る前に固定する部品（RULE 雛形・集計スクリプト・1 round の起動キット・nsys 補助）だけを置く。
診断 example は `crates/backend-cuda/examples/cuda_infer_chain_event_diag_4150.rs`。
`RULE.txt`（PR-2・単独コミット）、`round{i}/`・`aggregate.md`・`aggregate.tsv`・結果欄（PR-3・<cuda-node> での実測）は後続で追加する。
実測値・内訳・分類・確度は、実測するまで書かない（推定値で埋めない）。#4150 は実測と記録の PR-3 で閉じる（部品の PR は `Refs #4150`）。
PR-1 の担当環境は GB10 に到達できないため、単体テスト・`--self-test`・構文検査までしか行っていない。GB10 での動作（NULL stream 前提、event の摂動量、`nsys` の可用性）は未検証。

## 目的

0.14.0 の G-CUDA-INF（batch 64）は fandhe-ai reuse 0.0785 ms 対 PyTorch 0.0401 ms（比 0.51）。0.11.0 の診断（`docs/perf/infer-reuse-phase-breakdown.md` §12）では
chain 95.9 µs のうち download 69.2 µs、導出 `kernel_wait`（= download − 2 回目 download）63.2 µs だが、これは**導出値**で、GPU が実際に計算している時間か、
起動遅延・間隙・同期復帰といった待ちかは未分離（`docs/perf/infer-chain-fixed-cost-reduction-3205.md` §3 は「小 m のタイル占有率」と推定止まり）。
注意: 上の 0.0785 ms は 0.14.0 の本番経路（層出力を into 経路で再利用）で測った値で、#3189 の呼び出し列（P・E 腕。層ごとに確保＋ゼロ初期化・入力 H2D あり）には対応しない。
そこで #4188 で本番相当の腕（PR・ER）を足し、2 系列（`seq3189`＝P・E、`prod0140`＝PR・ER）を分けて記録する。0.0785 ms に対応するのは `prod0140`（下の「本番列と 0.14.0 G-CUDA-INF の区間対応」）。
本診断は record_only の内訳を取る（閾値なし）。結果は推論 Graph（reuse batch 64 で 0.861・undetermined）や学習 Graph（#3258・ADOPT・既定 OFF）の読みの材料にするが、Graph 腕は追加しない。

## 腕

| 腕 | 内容 |
|---|---|
| P | 系列 `seq3189`。3189 と同じ呼び出し列。host `Instant` のみ（`kernel_wait_derived` の再現）。層ごとに出力の確保＋ゼロ初期化・drop を含み、層出力を再利用する 0.14.0 の本番 `predict_device_chain`（#3205）とはこの点で異なる |
| E | 系列 `seq3189`。P と同じチェーンに CUDA event 4 本。GPU 側の幅（`up_gpu`・`l1_span`・`l2_span`・`chain_gpu`。層出力のゼロ初期化を含む） |
| PR | 系列 `prod0140`。0.14.0 本番の `DeviceParamStore::predict_device_chain_resident` の 2 回目以降（R は reuse）を再現する。常駐入力（区間外で 1 回 upload）・層出力 h／y は区間外で 1 回確保して使い回し、区間内は L1 into 発行・L2 into 発行・download のみ（確保・ゼロ初期化・H2D・drop を含まない）。host `Instant` のみ |
| ER | 系列 `prod0140`。PR と同じ列に CUDA event 3 本（`l1_span`・`l2_span`・`chain_gpu` = e0→e2。層出力のゼロ初期化を含まない） |
| I | 共通。drain 後に event 対で、同じ融合カーネルを into 経路（`linear_forward_device_into_tracked`・事前確保した出力）で層ごとに 1 回挟む（`l1_iso`・`l2_iso`。確保・ゼロ初期化・H2D は区間外） |
| F | 共通。固定費（`sync_idle`・`launch_floor_1x1`・`tiny_roundtrip`・`event_record_pair`）。`sync_idle`・`tiny_roundtrip`（1 要素 kernel の発行＋D2H＋同期を 1 区間）・`event_record_pair` は #4149 の `cuda_fixed` と区間定義が同じ。1x1 カーネルは into 経路（区間内に device 側の確保なし）。`launch_floor_1x1`（1x1 linear の単発発行）は #4149 の `launch_floor_add`（1 要素 add 100 回の 1 回あたり）と異なる |

## 本番列と 0.14.0 G-CUDA-INF（reuse）の区間対応

0.14.0 の G-CUDA-INF reuse の窓は `scripts/bench/framework-compare/bench-fandhe/src/main.rs::run_infer_reuse` で、入力 x を init で 1 回だけ `upload_resident` し、窓の中は `predict_resident_input` と `checksum_tensor` だけ（常駐入力。#3635）。
Issue #4188 の文言は `predict_device_chain`（ホスト入力版）を挙げるが、0.0785 ms と対応づけられるのは常駐入力版なので、本番列（PR・ER）は常駐入力版を再現する。両者は同じ `predict_device_chain_once` を共有し、違いは「入力の upload」と「第 1 層の発行直後の入力 drop」だけで、常駐入力ではどちらも起きず Graph capture 経路も通らない。

| 項目 | bench の窓（`run_infer_reuse`） | 本番列（PR の `total`） |
|---|---|---|
| 含む | `predict_resident_input`（L1 into 発行・L2 into 発行・出力の download）＋ `checksum_tensor` | L1 into 発行・L2 into 発行・download（`total = l1_issue + l2_issue + download`） |
| 含まない（本番列に入らない） | — | facade・autodiff の上位処理（poison・device・tape 検査、`checked_resident_buffer`、steps の解決、`try_lock`・key 照合・scratch の受け渡し、tape 生成）と host の checksum |
| 初回呼び出し | warmup 20／iters 20 の定常状態 | 初回のプローブは区間外（warmup 50／iters 200 の定常状態だけ） |

- 上位の固定費の参考値として 0.11.0 の `docs/perf/infer-reuse-phase-breakdown.md` §12.2 の 2.9 µs があるが、これは **0.11.0・ホスト入力版での実測・参考**で、0.14.0 の値として使わない。
- ホスト入力版 `predict_device_chain` の 2 回目以降は、本番列に「入力の確保＋H2D＋L1 発行後の drop」が加わったもの。本診断では**未計測**（必要なら 1 腕で追加できる）。
- 実測前なので数値は書かない。

## 読み方の注意

- TIMEOUT／INVALID の round を含む系列は `status=partial`・表題「参考」で、確度を出さない（RULE 項 3）。
- 分類と確度は系列ごとに RULE 項 5 の規則で機械的に出す（`event_perturbation > 0.10` なら確度は必ず低、`share_exec ≥ 0.5` は GPU 実行支配）。事後に動かさない。
- `E_sum`（`l1_iso + l2_iso`）は起動立ち上がりを含む融合カーネルの GPU 実行幅の実測で、出力のゼロ初期化を含まない。`chain_gpu` は層出力のゼロ初期化を含むため、その分は `gpu_gap` と `share_wait` 側に入る。`gpu_tail`・`wait_resid`・`share_*` は**推定**で、タグを付けて読む。
- 相手側（#4149 の PyTorch CUDA）との比較は、#4149 の実測が済んでいる場合だけ `--opp-tsv` で出す。無ければ unavailable。#4149 の実測が済んでいなければ、本イシューを閉じる前提条件が未充足。
- 相手側との対応は本番列（`prod0140`。H2D を含まず、#4149 の X が device 上であることと揃う）と取り、#3189 列は参考として並べる。
- 非対称（本番列）: fandhe の `l1_iso`／`l2_iso` は bias＋ReLU 融合込みの同じカーネル（into 経路）、PyTorch の `mm_event` は純 GEMM。fandhe の窓は facade 上位の固定費を含まない。
- 非対称（#3189 列のみ）: fandhe の窓は H2D を含み、PyTorch 側は X が device 上で H2D を含まない。fandhe の `chain_gpu` は層出力のゼロ初期化を含む。
- `internal-diagnostics` 有効時、`CudaDevice` は常に Legacy（NULL）stream を使うため、example が得た stream に event を記録すると本番チェーンと同じ stream に差し込める。example は冒頭で NULL stream を前提検査する。
- event 記録自体が計測を摂動する。P 腕と E 腕の `download` の差で見積もり、集計が `event_perturbation` として出す。
- nsys は時間を汚すため補助（record_only）に限り、判定に使わない。
- ノイズ帯 NB-RANGE5 は `docs/perf/noise-band-decision.md`（案・所有者承認待ち。#4142）。承認後に定義が変われば `RULE.v2.txt` で反映する。
- 同じ GB10 を #4147・#4149 の実測も使う。実機ロックで直列化し、並走させない。

## ファイル

| ファイル | 役割 |
|---|---|
| `RULE.template.txt` | 測る前に固定する規則の雛形（`aggregate_4150.py --check-template` で検査） |
| `aggregate_4150.py` | 厳格パースの集計（`--dir`・`--self-test`・`--check-template`・`--check-rule`・`--fingerprint`・`--opp-tsv`）。sha256 を RULE に固定 |
| `run_round.sh` | 1 round の起動（precheck → ビルド → ゲート → ビルド済みバイナリの直接実行。ゲート後に cargo を呼ばない） |
| `nsys_aux_4150.sh` | 全 round の後に 1 回の nsys 補助（record_only。不通は unavailable） |
| `crates/backend-cuda/examples/cuda_infer_chain_event_diag_4150.rs` | 計測本体（P／E／PR／ER／I／F 腕）。sha256 を RULE に固定 |

## 手順

1. （PR-2）PR-1 のマージ後の main SHA の git archive で `tree_sha`・`tree_fingerprint`・各 sha256 を埋め、
   `python3 -I aggregate_4150.py --check-rule RULE.txt` でプレースホルダ 0・sha 一致を確認して `RULE.txt` を**単独コミット**する。
   `git diff v0.14.0 <tree_sha> -- crates/backend-cuda/src` の差分有無を確認して RULE に記す。
2. （PR-3・<cuda-node>）実機ロックを取り、ツリー指紋を RULE と照合する。
   `ROOT=… EXPECT_SHA=… EXPECT_EXAMPLE_SHA=… HW_LOCK_OWNER_FILE=… HW_LOCK_HELD=1 bash run_round.sh <i> <記録先>/round<i>` を 1〜5 で回す。run の差し替え・除外・上書きは禁止。
3. 全 round の後に `ROOT=… OUT=<記録先>/nsys HW_LOCK_HELD=1 bash nsys_aux_4150.sh` を 1 回実行する（計時外）。
4. 回収時にパス上のユーザー名・ホスト名をマスクする（`<home>`・`<cuda-node>`）。
5. `python3 -I aggregate_4150.py --dir <記録先> [--opp-tsv <#4149 の相手値>]` で `aggregate.md`・`aggregate.tsv` を生成し、下の結果欄と
   `docs/perf/loss-attribution-matrix.md` の G-CUDA-INF 行、`docs/perf/infer-chain-fixed-cost-reduction-3205.md` §3 の「推定」を実測値＋タグで更新する（未測定は未測定のまま）。

## 結果

（実測後に追記。PR-1 時点は空。）
