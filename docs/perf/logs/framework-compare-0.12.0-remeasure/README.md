# 0.12.0 の 6 フレームワーク同席計測（イシュー #3273）

**状態: 実測済み（2026-10-27。GB10 5 ラウンド・Mac mini 系列 A／B・#3256／#3272 の実機テスト）。** 事前登録規則 `RULE.txt` は計測前の 2026-10-27T07:17:00Z に固定し、コミット `593172a7c` で push した後に系列を始めた。受け入れ条件 A1〜A6 は下表のとおり。Mac mini 側の生ログは `../framework-compare-0.12.0-macmini-remeasure/`。Mac mini と GB10 の値は同じ表に混ぜていない（機体ごとに節を分ける）。

## 目的

0.12.0 で bench-fandhe の非対称を是正した効果（#3265・#3272。`docs/framework-compare-harness-decision.md` §7）を、6 フレームワーク（fandhe-ai・candle・burn・PyTorch・TensorFlow・SciPy）同席の正式計測で確かめる。判定規則は 0.11.0 から変えない（比 = 同一 run 内の最速の有効な相手 / fandhe-ai reuse の 5 run 中央値。1 超で 1 位・0.90 以上で僅差・parity fail は判定不能）。対戦成績・対応表の更新は #3274 の範囲で、本イシューでは行わない。

## 受け入れ条件

| # | 要件 | 状態 | 根拠 |
|---|---|---|---|
| A1 | 全 run の tree が v0.12.0・rev-stamp が同一 | **達成** | GB10 5 run・Mac mini 系列 A／B の計 15 run で `tree.txt`・`tree-old.txt` が `└── fandhe-ai v0.12.0`、`rev-stamp.txt` が新腕 `bdaca14ccff7a01a4582eb84c6655f025a23208f`、`rev-stamp-old.txt` が旧腕 `0ae5670359c0f9d7a82f8bbd71791fb516713627`。`aggregate/check_identity.py` が 3 系列とも rc=0（行数 GB10 112／8／28／28、Mac mini 48／26／32 を含む） |
| A2 | 5 run 中央値を新・旧・0.11.0 で並べる | **達成** | `gb10/*-median5.jsonl`（新腕・extra・Python・旧腕・旧腕置換）と Mac mini 側の同名ファイル。3 条件の比を下の「結果」に並べた |
| A3 | 全セルの比と 1.0 以下の理由 | **達成（1 セルは分類保留）** | 全 27 セルの比は `scoreboard/gen_0120.out`（新条件）・`gen_0120_oldarm.out`（旧条件）・`aggregate/cell_table.md`。1.0 以下の 17 セルの理由は「1.0 以下のセルの理由」。G-CPU-INF は数字で 3 分類に割り当てられず「分類保留」（所有者判断 2026-10-27 で確定。`scoreboard/unreached_0120.tsv`） |
| A4 | 事前登録規則を計測前に固定 | **達成** | `RULE.txt`（固定 2026-10-27T07:17:00Z・コミット `593172a7c`）。系列の開始は GB10 07:18:50Z・Mac mini 07:17:47Z。固定後に規則・期待行数・tolerance は変えていない（RULE.v2 なし） |
| A5 | 機体を混ぜない | **達成** | Mac mini は run ごとに `machdep.cpu.brand_string == "Apple M4"` を検査（`macmini-run-0120.sh`）。GB10 は `env_info.txt` が `NVIDIA GB10, 580.173.02`。M4 Max の値・Linux の代替値は使っていない |
| A6 | #3256 の実機テスト結果（#3272 を含む） | **達成** | GB10: C1〜C9 各 5 run・45/45 pass、#3272 の `tests::resident_paths_match_tape_paths_cuda` 5/5 pass。Mac mini: M1〜M6 各 5 run・30/30 pass、`tests::resident_paths_match_tape_paths_metal` 5/5 pass。全 85 ログが `running 1 test`・`1 passed`・`rc=0` を満たし、invalid（foreign の cargo／rustc）は 0 件 |

## 計測条件

| 項目 | GB10（`<cuda-node>`） | Mac mini（`<metal-node>`） |
|---|---|---|
| 機体 | NVIDIA GB10・driver 580.173.02（`env_info.txt`）・rustc 1.97.1 | Apple M4・10 コア・16 GB・rustc 1.99.0 |
| ノード選定 | 0.11.0 の GB10 計測ノードの記録はない。指示の第一候補のノードを使い、途中で変えていない。計測時の compute プロセスは 0 件（全 run の `env_info.txt`）。別の GB10 ノードには常駐の推論サービスが居たため使わなかった | 計測専用機（0.11.0 と同じ機体） |
| 腕 | 新条件 `bdaca14cc`（計測直前の origin/main。`ff0ac8e4e` 以降）・旧条件 `0ae567035`（#3272 直前）。`git archive` で展開し `.rev-stamp`（40 桁）を置いて転送。共有ファイル（Cargo.lock 等）の新旧一致・腕判別 grep（新 1 件・旧 0 件）は run ごとに検査 | 同左 |
| registry ピン | `fandhe-ai =0.12.0`・`candle-core =0.11.0`・`burn =0.21.0` | 同左 |
| Python 3 FW | `~/work/.venv-bench` を新規作成（Python 3.12.3）: torch 2.14.0+cu130・tensorflow 2.21.0（CPU のみ）・scipy 1.18.1・numpy 2.5.4・keras 3.15.1（`env/pip-freeze-venv-bench.txt`）。0.11.0 の GB10 の numpy の版は記録がない | uv の Python 3.12.15。`venv-torch` = torch 2.14.0・scipy 1.18.1・numpy 2.5.4、`venv-tf` = tensorflow 2.16.2・tensorflow-metal 1.2.0・numpy 1.26.4・keras 3.15.1（Mac mini 側 `env/`）。0.11.0 README の記録と同じ |
| bench_py.py | sha256 `9fd2c08f…2409b`（`docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py`。run ごとに照合） | 同左 |
| 既定設定 | SME OFF・Graph capture OFF（既定のまま） | 同左 |
| 日時 | 2026-10-27 07:18:50Z〜07:55:04Z（5 ラウンド） | 系列 A 07:17:47Z〜07:26:03Z・系列 B 07:26:03Z〜07:37:16Z |
| ゲート | 各ラウンド前に load1 &lt; 1.0 かつ GPU 0% の 3 連続（`gb10/gate.log` 32 行）。ラウンド開始時 load1 = 0.28／0.38／0.83／0.49／0.30・GPU 0%。`GATE-INVALID`・`GATE-TIMEOUT` なし | 系列 B は各 run 前に load1 &lt; 3.0（`gate.log` 8 行。run1・run4・run5 の前に 3.47・4.88・3.63 で 1 分ずつ待機）。開始時 load1 は系列 A 1.86／3.08／2.97／4.51／3.07、系列 B 2.01／1.87／2.98／2.52／2.08 |
| 完走 | `gb10/loop.log` `ALL-DONE completed=5/5`。skipped 0 件 | 系列 A・B とも `ALL-DONE completed=5/5`。全 10 run の最終行 `done. rows=48 pyrows=32 oldrows=26 skipped=4`（skipped は burn Metal GEMM N=512〜4096 fresh の既知の記録拒否だけ） |
| 正式系列 | 5 ラウンド（RULE どおり） | 系列 B が 5/5 完走したため系列 B を正式、系列 A は対照（RULE どおり） |
| 実機ロック | `~/work/.fandhe-hw-lock`（owner `#3273 2026-10-27T07:03:07Z`）を 07:03:07Z に取得し、venv 構築・転送・ビルド・スモーク・系列・実機テスト・回収の後 07:58:26Z に解除 | 同 07:03:07Z 取得・07:56:09Z 解除 |

- **版の照合（JSONL の `version`）**: 全 15 run の Python 3 FW 行の `(framework, version)` 集合が 0.11.0 と一致した（GB10 `pytorch 2.14.0+cu130・tensorflow 2.21.0・scipy 1.18.1`、Mac mini 系列 A・B `pytorch 2.14.0・tensorflow 2.16.2・scipy 1.18.1`）。`check_identity.py` は収録後のツリーで再実行して 3 系列とも rc=0。
- 行数スモーク（RULE 固定前に 1 round。値は判定に使わない）は `smoke/`（GB10）・Mac mini 側 `smoke/` に置いた。行数は期待値どおり（RULE.txt に記入済み）。
- 旧腕は round 内の位置が新腕と違う（GB10 は stage2 の後・stage3 の前、Mac mini は Rust 段の直後）。RULE に偏りの可能性として明記した。
- **判定不能 1 セル**: GB10 PyTorch CPU gemm N=4096 の 1 要素が複合判定を外れる（`parity_fail_count` 1／16777216。0.11.0 と同じ）。順位に含めない。fandhe-ai／candle の CUDA TF32 opt-in 行は f32 判定から外す（0.11.0 と同じ扱い）。
- **セッションの違い（Mac mini CPU）**: Mac mini の gemm CPU 行は、全フレームワークの窓が 0.11.0 の約 0.5〜0.8 倍だった（例: gemm CPU N=256 の PyTorch 0.320→0.167 ms・candle 0.349→0.228 ms・fandhe-ai reuse 0.307→0.237 ms。5 run 中央値同士）。Python 3 FW の版は 0.11.0 と同じ。セル表の「0.11.0」列の比は同一 run 内比なので比較できるが、fandhe-ai の時間の 0.11.0 比（`gen_0120.out` の「0.11.0 比」）は機体の状態の差を含む。

## 結果（同一 run 内比の 5 run 中央値。判定は `scoreboard/gen_0120.py`）

| 条件 | 1 位 | 僅差 | 負け | 判定不能 | ノイズ帯 | 計 |
|---|---|---|---|---|---|---|
| 新条件（0.12.0・bench-fandhe 是正後） | **10** | 4 | 13 | 1 | 0 | 27 |
| 旧条件（0.12.0・#3272 直前の bench-fandhe） | 5 | 7 | 15 | 1 | 0 | 27 |
| 0.11.0（Mac mini 系列 B・GB10 0.11.0 正式系列） | 6 | 1 | 20 | 1 | 1 | 27 |

- 新条件の 1 位 10 セル: Mac mini gemm Metal N=256・512、train Metal、infer Metal、GB10 gemm CUDA N=256〜4096（5 セル）、train CUDA。
- 旧条件との差が大きいセル: Mac mini train Metal 0.32→1.01、GB10 train CUDA 0.98→2.01、GB10 gemm CUDA N=256〜2048 0.89〜0.98→1.12〜1.17（いずれも新条件が速い）。逆に **GB10 gemm CPU の 5 セルは新条件が旧条件より大きく遅い**（旧条件 1.42／0.92／1.20／1.65／1.19 → 新条件 0.71／0.68／0.87／0.92／0.96）。下の「GB10 CPU gemm の新旧差」を参照。
- 「ノイズ帯」（系列 A と系列 B で区分が反転した行）は 0 件。

### 3 条件のセル表（`aggregate/cell_table.md`）

各列は「区分 比（vs 最速相手）」。比は同一 run 内比の 5 run 中央値。旧条件は `swap_fandhe_arm.py` で fandhe-ai 行を同じ run の旧腕の行に置き換えた入力に、同じ生成器をかけた値。
#### Mac mini（M4）

| セル | 新条件（0.12.0） | 旧条件（0.12.0・旧腕） | 0.11.0 | 新条件の run 別比 |
|---|---|---|---|---|
| gemm Metal N=256 | 1 位 1.16×（vs candle） | 僅差 0.93×（vs candle） | 2 位 0.84×（vs candle） | 1.16 1.17 1.13 1.14 1.16 |
| gemm Metal N=512 | 1 位 1.06×（vs candle） | 2 位 0.84×（vs candle） | 2 位 0.80×（vs candle） | 0.98 1.08 1.06 1.05 1.09 |
| gemm Metal N=1024 ★ | 僅差 0.99×（vs candle） | 2 位 0.86×（vs candle） | 2 位 0.69×（vs candle） | 0.95 1.01 0.99 1.00 0.97 |
| gemm Metal N=2048 ★ | 僅差 0.94×（vs candle） | 2 位 0.81×（vs candle） | 2 位 0.73×（vs candle） | 0.94 0.94 0.95 0.94 0.94 |
| gemm Metal N=4096 ★ | 2 位 0.85×（vs candle） | 2 位 0.75×（vs candle） | 2 位 0.58×（vs candle） | 0.88 0.85 0.91 0.85 0.85 |
| gemm CPU N=256 ★ | 3 位 0.70×（vs pytorch） | 3 位 0.68×（vs pytorch） | 1 位 1.04×（vs pytorch） | 0.68 0.70 0.70 0.74 0.70 |
| gemm CPU N=512 ★ | 2 位 0.77×（vs pytorch） | 2 位 0.78×（vs pytorch） | 1 位 1.04×（vs pytorch） | 1.05 0.77 0.77 0.77 0.77 |
| gemm CPU N=1024 ★ | 2 位 0.67×（vs pytorch） | 2 位 0.68×（vs pytorch） | 2 位 0.89×（vs pytorch） | 0.71 0.66 0.66 0.67 0.69 |
| gemm CPU N=2048 ★ | 2 位 0.53×（vs pytorch） | 2 位 0.53×（vs pytorch） | 2 位 0.69×（vs pytorch） | 0.54 0.54 0.53 0.52 0.52 |
| train Metal | 1 位 1.01×（vs pytorch） | 4 位 0.32×（vs pytorch） | 3 位 0.32×（vs pytorch） | 1.01 1.01 1.03 1.02 0.99 |
| infer Metal | 1 位 1.02×（vs candle） | 僅差 0.94×（vs candle） | 1 位 1.13×（vs candle） | 1.03 1.01 1.02 1.01 1.03 |
| train CPU ★ | 3 位 0.25×（vs pytorch） | 3 位 0.26×（vs pytorch） | 4 位 0.20×（vs pytorch） | 0.25 0.26 0.25 0.25 0.26 |
| infer CPU ★ | 4 位 0.15×（vs pytorch） | 4 位 0.15×（vs pytorch） | 2 位 0.25×（vs pytorch） | 0.16 0.15 0.15 0.15 0.15 |

#### GB10

| セル | 新条件（0.12.0） | 旧条件（0.12.0・旧腕） | 0.11.0 | 新条件の run 別比 |
|---|---|---|---|---|
| gemm CUDA N=256 | 1 位 1.17×（vs candle） | 2 位 0.89×（vs candle） | 2 位 0.84×（vs candle） | 1.19 1.15 1.17 1.17 1.16 |
| gemm CUDA N=512 | 1 位 1.12×（vs candle） | 僅差 0.95×（vs candle） | 3 位 0.42×（vs candle） | 1.06 1.12 1.15 1.15 1.12 |
| gemm CUDA N=1024 | 1 位 1.15×（vs candle） | 僅差 0.96×（vs candle） | 3 位 0.40×（vs candle） | 1.14 1.15 1.16 1.21 1.15 |
| gemm CUDA N=2048 | 1 位 1.13×（vs burn） | 僅差 0.98×（vs burn） | 3 位 0.48×（vs burn） | 1.14 1.11 1.15 1.13 1.12 |
| gemm CUDA N=4096 | 1 位 1.92×（vs burn） | 1 位 1.72×（vs burn） | 1 位 1.01×（vs burn） | 1.93 1.85 1.93 1.91 1.92 |
| gemm CPU N=256 ★ | 3 位 0.71×（vs pytorch） | 1 位 1.42×（vs pytorch） | 3 位 0.74×（vs pytorch） | 0.68 0.69 0.71 0.82 0.75 |
| gemm CPU N=512 ★ | 2 位 0.68×（vs candle） | 僅差 0.92×（vs candle） | 2 位 0.70×（vs candle） | 0.77 0.68 0.61 0.95 0.63 |
| gemm CPU N=1024 ★ | 2 位 0.87×（vs candle） | 1 位 1.20×（vs candle） | 僅差 0.98×（vs candle） | 0.82 0.87 0.84 0.88 0.89 |
| gemm CPU N=2048 ★ | 僅差 0.92×（vs tensorflow） | 1 位 1.65×（vs tensorflow） | 1 位 1.06×（vs tensorflow） | 0.99 0.92 0.89 1.09 0.91 |
| gemm CPU N=4096 ★ | 僅差 0.96×（vs tensorflow） | 1 位 1.19×（vs tensorflow） | 1 位 1.21×（vs tensorflow） | 0.96 0.97 0.94 0.97 0.96 |
| train CUDA | 1 位 2.01×（vs candle） | 僅差 0.98×（vs candle） | 2 位 0.83×（vs candle） | 2.00 2.01 2.04 2.00 2.05 |
| infer CUDA ★ | 3 位 0.51×（vs pytorch） | 3 位 0.44×（vs pytorch） | 3 位 0.40×（vs pytorch） | 0.51 0.51 0.52 0.52 0.51 |
| train CPU ★ | 2 位 0.55×（vs pytorch） | 2 位 0.48×（vs pytorch） | 3 位 0.35×（vs pytorch） | 0.55 0.56 0.49 0.60 0.46 |
| infer CPU ★ | 2 位 0.78×（vs pytorch） | 2 位 0.68×（vs pytorch） | 2 位 0.60×（vs pytorch） | 0.70 0.71 0.78 1.00 0.84 |

- 新条件: `tally 10 4 13 invalid 1 noise 0 total 27`
- 旧条件: `tally 5 7 15 invalid 1 noise 0 total 27`
- 0.11.0: `tally 6 1 20 invalid 1 noise 1 total 27`
- ★ = 新条件の比が 1.0 以下

### GB10 CPU gemm の新旧差（実測）

切り分けの設計と状態（GB10 実測は未実施・原因は未特定）は #3646（`../cpu-gemm-resident-reuse-3646/README.md`・`docs/perf/cpu-gemm-reuse-fresh-inversion.md` §9）。本節の数値と判定は変えない。

fandhe-ai の 5 run 中央値（ms）。新腕の reuse は同じ round の新腕 fresh より遅く、旧腕は reuse と fresh が並ぶ。fresh 同士は新旧でほぼ同じ（位置の違う段で測っても差は 0〜5%）。

| N | 新腕 fresh | 新腕 reuse | 新腕 reuse/fresh | 旧腕 fresh | 旧腕 reuse | 旧腕 reuse/fresh | 0.11.0 reuse |
|---|---|---|---|---|---|---|---|
| 256 | 0.196 | 0.351 | 1.79 | 0.189 | 0.178 | 0.95 | 0.367 |
| 512 | 1.832 | 2.311 | 1.26 | 1.796 | 1.770 | 0.99 | 2.522 |
| 1024 | 4.641 | 6.344 | 1.37 | 4.637 | 4.475 | 0.97 | 5.598 |
| 2048 | 17.598 | 30.974 | 1.76 | 16.757 | 17.105 | 1.02 | 26.301 |
| 4096 | 125.016 | 153.687 | 1.23 | 123.672 | 123.908 | 1.00 | 121.482 |

- 新腕の gemm reuse は `matmul_resident` → readout → checksum の窓（`docs/framework-compare-harness-decision.md` §7）。N=4096 の新腕は reuse と fresh を同じ stage2 で続けて測っており、位置の違いでは説明できない。原因（CPU の常駐経路の中のどの区間か）は本計測では切り分けていない。
- Mac mini の CPU gemm は新旧とも reuse と fresh が並ぶ（例: N=2048 新腕 36.69／36.32 ms、旧腕 36.12／35.91 ms）。この差は GB10 の CPU だけに出ている。

### 1.0 以下のセルの理由（新条件・17 セル）

分類は `docs/perf/loss-attribution-matrix.md` §10.5 の 3 区分（ハードの上限／計測窓の定義／自分たちの制約）。数字の「0.12.0」は本計測の 5 run 中央値（実測）、§10.3〜§10.5 の値は 0.11.0 時点の実測・推定（タグは同節のまま）。(3)(4) は §10.3・§10.4 の「再現で届く比」「工夫で届く比」で、目標値の列であり合否には使わない。制約の緩和は提案しない。

| セル | 新条件の比 | (3)／(4)（§10.3・§10.4） | 分類 | 数字 |
|---|---|---|---|---|
| Mac mini gemm Metal N=1024 | 僅差 0.99 | F2 0.76／F4 1.00（仮想上限） | 計測窓の定義 | 0.12.0: fandhe-ai reuse 2.354 ms 対 candle fresh 2.300 ms。0.11.0 の自作カーネル kernel_gpu 0.861 ms（2.50 TFLOPS）は candle excl 0.950 ms（2.26 TFLOPS）より速い（`MM-182`）。新条件で upload が窓から外れ、残る固有分は readout（0.11.0 の host_copy 0.241 ms）と checksum。F4（upload・host_copy を 0）の 1.00 と実測 0.99 が並ぶ |
| Mac mini gemm Metal N=2048 | 僅差 0.94 | F2 0.81／F3 0.76〜0.78・F4 0.84 | 自分たちの制約 | 0.12.0: 10.113 ms（窓で 1.70 TFLOPS）対 candle 9.550 ms（1.80 TFLOPS）。0.11.0 の kernel_gpu 2.13 TFLOPS 対 candle excl 3.15・MPS 純時間 3.237 TFLOPS（`MM-182`・`MM-179`）。MPS は REQ-1 により使わない構成。実測 0.94 は (4) の推定 0.84 を上回った |
| Mac mini gemm Metal N=4096 | 2 位 0.85 | F2 0.62／F3 0.71〜0.73・F4 0.70 | 自分たちの制約 | 0.12.0: 65.389 ms（2.10 TFLOPS）対 candle 55.601 ms（2.47 TFLOPS）。0.11.0 の kernel_gpu 2.11 TFLOPS（最良タイル 2.58）対 candle excl 3.24・MPS 3.429 TFLOPS（`MM-182`・`MM-181`・`MM-179`）。REQ-1（MPS 不使用） |
| Mac mini gemm CPU N=256 | 3 位 0.70 | Accelerate は再現不能・F2 0.95／SME は対象外で未測定 | ハードの上限 | 0.12.0: fandhe-ai 0.237 ms 対 PyTorch 0.167 ms。相手の純 `mm` は 0.0240 ms（閉源 Accelerate・`MM-179`）。0.11.0 の 1 位（1.04）は PyTorch の窓が 0.320 ms だったセッションの値で、本計測では PyTorch の窓が 0.52 倍（`MM-179` が記録した 0.52〜0.54 倍の窓と同じ水準） |
| Mac mini gemm CPU N=512 | 2 位 0.77 | F2 1.05／SME ON の A/B で PyTorch 比 1.307／1.251 | 自分たちの制約 | 0.12.0: 0.951 ms 対 PyTorch 0.731 ms。SME は既定 OFF（`SME_PRODUCTION_ENABLED = false`）。SME ON の A/B（`MM-176`）は同一ラウンド内の PyTorch 比 0.737／0.742 → 1.307／1.251 |
| Mac mini gemm CPU N=1024 | 2 位 0.67 | F2 0.91／SME ON で 1.406／1.246 | 自分たちの制約 | 0.12.0: 5.423 ms 対 PyTorch 3.649 ms。SME 既定 OFF。`MM-176` の PyTorch 比 0.650／0.619 → 1.406／1.246 |
| Mac mini gemm CPU N=2048 | 2 位 0.53 | F2 0.68／SME ON で 1.109／1.030 | 自分たちの制約 | 0.12.0: 36.688 ms（468 GFLOPS）対 PyTorch 19.231 ms（893 GFLOPS）。SME 既定 OFF。`MM-176` の PyTorch 比 0.525／0.494 → 1.109／1.030 |
| Mac mini train CPU | 3 位 0.25 | (3) 未測定／SME ON 0.90〜0.91・F3 0.22 | 自分たちの制約（確度低） | 0.12.0: fandhe-ai reuse 0.508 ms 対 PyTorch 0.129 ms。§10.5.2: fandhe-ai は第 1 層の dX を計算してから捨てる形（推定）、L2 の細い形状の単スレッド 29〜40 GFLOPS（`MM-177`）。相手の純カーネルは未測定で、ハードの上限は数字で示せない |
| Mac mini infer CPU | 4 位 0.15 | (3) 未測定／F3 0.29（REJECT） | 自分たちの制約（確度低） | 0.12.0: 0.205 ms 対 PyTorch 0.031 ms。SME は m=64 に効かない（0.9821／1.0100・`MM-176`）。L2 形状の単スレッド 29〜40 GFLOPS（`MM-177`）。相手の純カーネルは未測定 |
| GB10 gemm CPU N=256 | 3 位 0.71 | F2 1.39／affinity REJECT | 計測窓の定義 | 0.12.0: 新腕 reuse/fresh 1.79（上表）。旧腕（同じ v0.12.0）は同一 run 内比 1.42 で 1 位。fresh 0.196 ms 対 PyTorch 0.259 ms（中央値同士・参考） |
| GB10 gemm CPU N=512 | 2 位 0.68 | F2 1.01 | 計測窓の定義 | 0.12.0: 新腕 reuse/fresh 1.26。旧腕は 0.92（僅差）。fresh 1.832 ms 対 candle 1.654 ms（中央値同士・参考）で fresh でも届かない分は相手の経路が未測定（§10.3.4 (1)） |
| GB10 gemm CPU N=1024 | 2 位 0.87 | F2 1.35 | 計測窓の定義 | 0.12.0: 新腕 reuse/fresh 1.37。旧腕は 1.20 で 1 位 |
| GB10 gemm CPU N=2048 | 僅差 0.92 | F2 1.68 | 計測窓の定義 | 0.12.0: 新腕 reuse/fresh 1.76。旧腕は 1.65 で 1 位（0.11.0 も 1.06 で 1 位） |
| GB10 gemm CPU N=4096 | 僅差 0.96 | F2 1.20 | 計測窓の定義 | 0.12.0: 新腕 reuse/fresh 1.23。旧腕は 1.19 で 1 位（0.11.0 も 1.21 で 1 位） |
| GB10 infer CUDA | 3 位 0.51 | (3) 未測定／Graph capture 0.861（undetermined）・F3 0.47 | 自分たちの制約（確度低） | 0.12.0: fandhe-ai reuse 0.0776 ms 対 PyTorch 0.0400 ms・candle 0.0413 ms。0.11.0 の chain 95.9 µs のうち導出 kernel_wait 63.2 µs、launch 往復 6.14 µs（`GB-189`）。相手は cuBLAS（閉源）で、REQ-1 により使わない構成。相手側の区間分解は未測定 |
| GB10 train CPU | 2 位 0.55 | (3) 未測定／F3 0.39 | 自分たちの制約（確度低） | 0.12.0: 0.734 ms 対 PyTorch 0.363 ms。§10.5.3: dX を計算してから捨てる形（推定）・NEON のみ（SME 非対応）。相手の経路は未確認 |
| GB10 infer CPU | 2 位 0.78 | (3) 未測定／F3 0.77（REJECT） | **分類保留** | 0.12.0: 0.167 ms 対 PyTorch 0.131 ms。0.11.0 で記録された reuse が fresh より遅い逆転は本計測では見えない（reuse 0.1667・fresh 0.1698 ms）。相手の経路と純カーネルは未測定で、制約の欄にも該当する項目がない（§10.5.3）。3 区分のどれにも数字で割り当てられない |

## 実機テスト（#3256 の C1〜C9・M1〜M6、#3272）

時間系列の後、同じ実機ロックの中で新腕の展開ツリー（`bdaca14cc`）を使って実行した。判定は `../resident-chain-parity-3256/RULE.txt`（各行 5 run・独立プロセス・`--ignored --exact --test-threads=1`）。#3272 の 2 本は RULE.txt に X1 として登録した（同じ判定）。ビルドは `cargo test --no-run` で測定前に 1 回だけ（`build.log`・判定外）。

| 行 | 機体 | テスト | 5 run の結果 |
|---|---|---|---|
| C1〜C4 | GB10 | `cuda_public_api_*`・`cuda_upload_resident_with_cpu_tape_is_device_mismatch` | 各 5/5 pass |
| C5 | GB10 | `cuda_tf32_mode_chain_matches_tape_path_bit_exact` | 5/5 pass |
| C6・C7 | GB10 | #3252 の窓 0・読み出し同期 1 | 各 5/5 pass（C7 は毎回 window 全 0・read `stream_syncs: 1, d2h_calls: 1`） |
| C8 | GB10 | TF32 モードのチェーン（記録専用） | 5/5 `TF32OBS outcome=Ok`（初回 `mem_allocs` 11・同期 0・D2H 0） |
| C9 | GB10 | P7 の wait と内部 tape を捨てた後の read | 5/5 pass |
| X1 | GB10 | `bench-fandhe` `tests::resident_paths_match_tape_paths_cuda` | 5/5 pass |
| M1〜M6 | Mac mini | Metal の公開入口の bit 一致・同期回数 | 各 5/5 pass（M5 は毎回 `encode=10 command_buffers=1 wait=0 / read: wait=1`） |
| X1 | Mac mini | `bench-fandhe` `tests::resident_paths_match_tape_paths_metal` | 5/5 pass |

合計 85 ログ（GB10 50・Mac mini 35）。全ログが `running 1 test`・`1 passed`・末尾 `rc=0` を満たす（0 件実行の素通りなし）。invalid（foreign の cargo／rustc 検出）は 0 件。生ログは `tests-cuda/`・Mac mini 側 `tests-metal/`（ホスト名・ユーザー名・絶対パスはマスク済み）。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | GB10 の事前登録規則（固定 2026-10-27T07:17:00Z） |
| `scripts/dgx-run-0120.sh`・`scripts/dgx-0120-loop.sh` | 1 round・5 round ループ（本計測で使った版。変更なし） |
| `gb10/run{1..5}/` | 各 round の JSONL（`results-dgx-0.12.0.jsonl`＝run_all_cuda.sh 112 行、`-extra` 8 行、`-py` 28 行、`results-dgx-old-0.12.0.jsonl`＝旧腕 28 行、`results-dgx-oldarm-swapped-0.12.0.jsonl`＝旧腕置換の集計入力）・`tree.txt`・`tree-old.txt`・`rev-stamp*.txt`・`env_info.txt`・`uptime_*.txt`・`gpu_util_before.txt`・`run_all_cuda.log`・`*.err`・`skipped-dgx-0.12.0.log` |
| `gb10/run{1..5}.log`・`gb10/loop.log`・`gb10/gate.log`・`gb10/build-*.log` | round のログ・ループのログ・ゲート・系列前のビルド |
| `gb10/*-median5.jsonl` | 5 run 中央値（新腕 58 セル・extra 8・Python 28・旧腕 28・旧腕置換 66） |
| `smoke/` | RULE 固定前の行数スモーク（判定に使わない） |
| `tests-cuda/` | #3256（`3256/`）・#3272（`3272/`）の実機テストのログ |
| `env/pip-freeze-venv-bench.txt` | GB10 の venv の `pip freeze` |
| `aggregate/median_rounds.py`・`swap_fandhe_arm.py`・`check_identity.py` | 集計器（変更なし） |
| `aggregate/cell_table.py`・`cell_table.md` | 3 条件の判定一覧を 1 枚の表にする生成器（本 PR で追加）とその出力 |
| `scoreboard/gen_0120.out`・`gen_0120_oldarm.out` | 新条件・旧条件の判定一覧（`gen_0120.py` の標準出力。#3274 の `run_0120.sh` で再生成して byte 一致を確認） |
| `scoreboard/gen_args.txt` | 負荷・日付・skipped の注記引数（実測値を記入済み） |
| `scoreboard/gen_0120.py`・`body_0120.html`・`style.css` | スコアボード生成器とテンプレート（#3274） |
| `scoreboard/unreached_0120.tsv` | 新条件で 1 位でない 17 セルの届かなかった理由（3 分類＋分類保留。#3274） |
| `scoreboard/run_0120.sh`・`check_0120.py` | 判定一覧 2 本と HTML の一括生成・HTML と判定一覧の機械照合（#3274） |
| `scoreboard/out/scoreboard_0120.html` | 0.12.0 のスコアボード（1 ファイル・CSS インライン。#3274 で生成。公開はメインのセッション） |

## 再現手順

```bash
# 0. 展開ツリー（新腕は全体、旧腕は framework-compare だけ）。.rev-stamp は 40 桁 1 行
git archive bdaca14ccff7a01a4582eb84c6655f025a23208f | tar -x -C new && git rev-parse bdaca14cc > new/.rev-stamp
git archive 0ae5670359c0f9d7a82f8bbd71791fb516713627 scripts/bench/framework-compare | tar -x -C old && echo 0ae5670359c0f9d7a82f8bbd71791fb516713627 > old/.rev-stamp
# 1. GB10: ~/work/m3273/{new,old}、~/work/dgx-run-0120.sh、~/work/bench_py.py、~/work/.venv-bench を置き、実機ロックを取ってから
ROOT=~/work/m3273/new OLD_ROOT=~/work/m3273/old EXPECT_NEW_SHA=… EXPECT_OLD_SHA=… setsid nohup bash dgx-0120-loop.sh > loop-gb10.log 2>&1 &
# 2. 集計（このディレクトリで）
g=gb10; python3 aggregate/median_rounds.py $g/results-dgx-0.12.0-median5.jsonl $g/run{1..5}/results-dgx-0.12.0.jsonl   # extra・py・old も同様
for i in 1 2 3 4 5; do python3 aggregate/swap_fandhe_arm.py $g/run$i/results-dgx-0.12.0.jsonl $g/run$i/results-dgx-0.12.0-extra.jsonl $g/run$i/results-dgx-old-0.12.0.jsonl $g/run$i/results-dgx-oldarm-swapped-0.12.0.jsonl; done
python3 aggregate/check_identity.py <NEW_SHA> <OLD_SHA> results-dgx-0.12.0.jsonl=112,results-dgx-0.12.0-extra.jsonl=8,results-dgx-py-0.12.0.jsonl=28,results-dgx-old-0.12.0.jsonl=28 gb10/run{1..5}
# 3. 判定一覧（新条件）。旧条件は run ディレクトリの results-*-0.12.0.jsonl を旧腕置換版に差し替えた一時ツリーへ同じ生成器をかける
M=../framework-compare-0.12.0-macmini-remeasure
python3 scoreboard/gen_0120.py --m4 $M/macmini-series-b/results-macmini-0.12.0-median5.jsonl --m4-py $M/macmini-series-b/results-macmini-py-0.12.0-median5.jsonl \
  --m4-alt $M/macmini-series-a/results-macmini-0.12.0-median5.jsonl --m4-py-alt $M/macmini-series-a/results-macmini-py-0.12.0-median5.jsonl \
  --m4-runs $M/macmini-series-b --m4-alt-runs $M/macmini-series-a --gb gb10/results-dgx-0.12.0-median5.jsonl \
  --gb-extra gb10/results-dgx-0.12.0-extra-median5.jsonl --gb-py gb10/results-dgx-py-0.12.0-median5.jsonl --gb-runs gb10 \
  --m4-old $M/macmini-series-b/results-macmini-old-0.12.0-median5.jsonl --gb-old gb10/results-dgx-old-0.12.0-median5.jsonl \
  --m4-date 2026-10-27 --gb-date 2026-10-27 --out /tmp/sb.html > scoreboard/gen_0120.out
# 3'. #3274: 判定一覧 2 本（新条件・旧条件の一時ツリー）と HTML を一括で作り、HTML と判定一覧を照合する（下の「スコアボード」節）
bash scoreboard/run_0120.sh
python3 aggregate/cell_table.py scoreboard/gen_0120.out scoreboard/gen_0120_oldarm.out ../framework-compare-0.11.0-macmini-remeasure/scoreboard/gen_0110_macmini.out > aggregate/cell_table.md
# 4. 実機テスト（新腕の全体ツリーで。系列の後・同じロック内）
bash docs/perf/logs/resident-chain-parity-3256/run_rows.sh <out> cuda   # Mac mini は metal
cargo test -p bench-fandhe --release --locked tests::resident_paths_match_tape_paths_cuda -- --ignored --exact --test-threads=1 --nocapture  # framework-compare で
```

## スコアボード（#3274）

**状態: 生成済み（2026-10-27）。** `scoreboard/run_0120.sh` が、`gen_args.txt` の注記引数（上の計測条件の値。`gate.log`・`uptime_*.txt`・`loop.log`・`skipped-*.log` と照合済み）を付けて `gen_0120.py` を 2 回かける。

| 出力 | 内容 |
|---|---|
| `scoreboard/gen_0120.out` | 新条件の判定一覧。`tally 10 4 13 invalid 1 noise 0 total 27` |
| `scoreboard/gen_0120_oldarm.out` | 旧条件の判定一覧。run ディレクトリの主 JSONL を旧腕置換版（`*-oldarm-swapped-0.12.0.jsonl`）にした一時ツリーで生成する。GB10 の置換版は extra を含むため extra は空ファイル、Mac mini は系列 A・B の両方を置換する。`tally 5 7 15 invalid 1 noise 0 total 27` |
| `scoreboard/out/scoreboard_0120.html` | 新条件の HTML。旧条件・0.11.0（`../framework-compare-0.11.0-macmini-remeasure/scoreboard/gen_0110_macmini.out`・`tally 6 1 20 invalid 1 noise 1 total 27`）の tally を 3 条件表に並べる |

- 再生成した 2 本の判定一覧は、#3273 でコミットした版と byte 一致した（`run_0120.sh` が `cmp` で検査し、不一致なら止める）。勝敗は上の「結果」の 3 条件（10/4/13/1・5/7/15/1・6/1/20/1）と一致する。
- `check_0120.py` が、HTML の判定行 27 行（Mac mini 13・GB10 14）の順序・区分・比・相手・0.11.0 比、先頭の tally、3 条件の tally 表、届かなかったセルの表（TSV と同じ 17 行）、GB10 CPU gemm の注記の比を判定一覧と照合する。未実測の placeholder・実ホスト名・フォント以外の外部参照・script が無いことも確かめる。
- 生成器の変更は表示だけで、`judge`／`paired_ratio`・判定規則・tolerance は変えていない。追加したのは、旧条件・前版の tally 表（`--oldarm-out`／`--prev-out`）、GB10 CPU gemm の新旧の reuse／fresh の注記（`--gb-old` のデータから生成）、届かなかったセルの表示専用ラベル「分類保留」、出力の UTF-8 指定。テンプレートには `<meta charset="utf-8">` を足した（file:// で開いたときの文字化け防止）。外部参照は Google Fonts の stylesheet だけ。
- 所有者判断（2026-10-27）の注記: 新条件を正式値・旧条件を併記、GB10 CPU gemm の後退は注記して切り分けは #3646、G-CPU-INF は「分類保留（数字で 3 分類に割り当てられなかった）」、相互運用行は「部分的」、A3-e・A3-f・C4・C8 は対象外、`docs/perf/framework-compare-feature-matrix-0.12.0.md` §6 の 9 件は未回答の申し送り。
- 届かなかったセル 17 件の内訳: ハードの上限 1・計測窓の定義 6（Mac mini gemm Metal N=1024 と GB10 CPU gemm 5 セル）・自分たちの制約 9（うち確度低 4）・分類保留 1（G-CPU-INF）。

### 公開先

0.12.0 用の claude.ai Artifact を新しく作る（所有者判断 2026-10-27）。0.11.0 版のページ（`../framework-compare-0.11.0-macmini-remeasure/README.md`「公開先」）は上書きせず、GitHub Pages には載せない。公開はメインのセッションが `scoreboard/out/scoreboard_0120.html` を使って行う。2026-10-27 に公開した: <https://claude.ai/artifact/MY9opk9bGoA8fxu4qaNWMv>（非公開。共有範囲は所有者が設定する）。公開版は `scoreboard/out/scoreboard_0120.html` から、先頭の `charset`・`viewport` meta の 2 行だけを除いたもの（公開側の骨組みが同じ meta を付けるため）。

## 経緯（2026-10-27 の事前確認）

最初の事前確認（PR #3640）では、両機体とも Python 3 FW の環境（GB10 の TensorFlow・SciPy、Mac mini の `venv-tf`・`bench_py.py` の配置先）が無く、計測を見送った。2026-10-27 の所有者判断（#3273）で、0.11.0 と同じ版での再構築が承認され、本計測を行った。同じ版がすべて手に入ったため、代用した版はない。

## 所有者判断を仰ぐ点

所有者判断（2026-10-27・#3274）: 1 は「新条件を正式値とし旧条件を併記。後退は注記で触れ、切り分けは #3646」、2 は「G-CPU-INF は分類保留（数字で 3 分類に割り当てられなかった）と明記」で確定した。3〜5 は未回答。

1. **GB10 CPU gemm の新旧差**: 新条件の bench-fandhe では GB10 の CPU gemm reuse が同じ round の fresh より 1.23〜1.79 倍遅く、旧条件で 1 位だった 4 セル（N=256・1024・2048・4096）が負け・僅差になった。判定規則どおり新条件の値を正式とするが、#3274 の対戦成績でこの 5 セルをどう扱うか（旧条件の併記で足りるか、常駐経路の CPU 側の切り分けを別イシューにするか）。
2. **G-CPU-INF の分類保留**: 数字で 3 区分に割り当てられない。相手の経路の確認（#3191 の (1) が未測定）を待つか。
3. **Mac mini CPU のセッション差**: gemm CPU で全フレームワークの窓が 0.11.0 の約 0.5〜0.8 倍で、gemm CPU N=256・512 は 0.11.0 の 1 位から負けに変わった。N=256 の窓は fandhe-ai が 0.77 倍、PyTorch が 0.52 倍で、相手の窓の縮小のほうが大きい。
4. **GB10 の計測個体**: GB10 の venv は、PR #3640 の事前確認で PyTorch を見つけた個体とは別の GB10 に新規作成した（0.11.0 の GB10 の計測個体は記録がなく、同一個体かは不明）。
5. **勝ちセルの (3)(4) との突き合わせ**: 1.0 以下の 17 セルは (3)(4) と並べたが、1 位の 10 セルの (3)(4) 列は作っていない。#3274 のスコアボード（`--unreached-tsv`）で扱うか。
