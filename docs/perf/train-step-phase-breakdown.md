# 学習 1 step フェーズ分解の実機実測と支配項の確定（イシュー #1010）

## 1. 目的・対応

イシュー #1010「CPU / CUDA / Metal の学習 1 step フェーズ分解を実機実測して
支配項を確定する」（親 #1008「学習・推論ループの固定費の診断と除去」）に
対応する実測記録。#1008 は「GEMM 単体差で説明できない 20〜40 倍差の主因は
固定費」と仮定して #1011〜#1028 を並べたが、どの区間が支配的かの実測は
本イシュー以前には存在しなかった。

計測ハーネスはイシュー #1009（PR #1055・2026-08-29 マージ済み）で実装済みの
`scripts/bench/framework-compare/bench-fandhe --task train --device <device>
--mode <fresh|reuse> --phases`。区間定義・JSONL スキーマ・`summarize.py`
(b'') 節の読み方は `scripts/bench/framework-compare/README.md`「`train
--phases`」節を正とし、本ドキュメントでは二重管理しない。

## 2. 計測プロトコル

- モデル: 784→256→10（ReLU）・バッチ 64・MSE・SGD lr=0.01（`--size 64`）
- `warmup=20` / `iters=80`（producer 側の既定値。`bench_common` 参照）
- **5 回計測**（`.claude/rules/coding-rust.md`）: 1 ラン目を表の生成元
  （`results-*-phases-0.4.0.jsonl`）、残り 4 ラン分を
  `results-*-phases-0.4.0-extra.jsonl` へ分離（`run_all.sh`/`run_all_cuda.sh`
  と異なり `results/raw/results*.jsonl` を初期化しないため専用ファイル名を
  用いた。環境 4/5 の既存計測と同じ「1 ラン目固定＋extra 分離」方針）
- 本表の数値は各ランの `median_s` 5 個の中央値（`statistics.median`）と
  範囲（min–max）。summarize.py 生成表（1 ラン目のみ）は §5 に個別記載
- 実行コマンド（両環境共通のループ本体）:

  ```bash
  ./target/release/bench-fandhe --task train --device <cpu|metal|cuda> \
    --size 64 --mode <fresh|reuse> --phases --out <dest.jsonl>
  ```

- 計測環境:
  - 環境 A: Apple M4 Max・macOS 26.6.2 (25G83)・rustc/cargo 1.96.0
    （ローカル直接実行。デバイス cpu・metal）
  - 環境 B: DGX Spark GB10（sm_121）・NVIDIA driver 580.173.02・
    CUDA 13.0 V13.0.88（nvcc）・Ubuntu 24.04 aarch64・rustc/cargo 1.97.0
    （SSH リモート実行。デバイス cuda・cpu。`docs/real-hardware-
    verification-env.md` §2〜4 準拠）
- 計測日: 2026-08-31
- **版差の注記**: `fandhe-ai =0.4.0`（crates.io 公開版。2026-08-29 公開）を
  計測している。現 main（#1055 以降・PR #1078〜#1081 のマージ後）とは
  内訳が異なる。詳細は §7 を参照

## 3. 内訳表（5 ラン中央値・範囲・step_total 比）

生データ: `results/raw/results-m4max-phases-0.4.0.jsonl`（+ `-extra.jsonl`）・
`results/raw/results-dgx-phases-0.4.0.jsonl`（+ `-extra.jsonl`）。両環境とも
`skipped-*-phases-0.4.0.log` は空（全実行成功。捏造していないことの証跡）、
`python3 summarize.py <jsonl> --strict` は exit 0（§5 の 1 ラン目単体で確認）。

「step_total 比」は当該行の 5 ラン中央値 ÷ step_total の 5 ラン中央値。
sub-µs の値（`0.0 µs` 表示）は `_safe_phase_time_s` により有効値として扱う
（9 桁固定小数シリアライズ自体は ns 単位を表現できるため丸まらないが、
sub-100 ns 区間は計時クロックの分解能未満の標本で連続する `Instant::now()`
が同一時刻を返し区間長が 0 と計測されることがあるため。`summarize.py`
`_safe_phase_time_s` docstring 参照）。

### 3.1 CPU / Apple M4 Max

| フェーズ | fresh 中央値 (範囲) | fresh 比 | reuse 中央値 (範囲) | reuse 比 |
|---|---|---|---|---|
| tape_build | 0.0 µs [0.0, 0.0] | 0.0% | 0.0 µs [0.0, 0.0] | 0.0% |
| leaf_register | 0.4 µs [0.4, 0.5] | 0.0% | 0.2 µs [0.1, 0.2] | 0.0% |
| forward / forward_resident | 726.5 µs [705.0, 736.7] | 4.5% | 726.4 µs [712.1, 731.3] | 4.5% |
| loss_readout | 0.0 µs [0.0, 0.0] | 0.0% | 0.0 µs [0.0, 0.0] | 0.0% |
| **backward** | **15.413 ms [15.385, 15.457]** | **95.2%** | **15.396 ms [15.374, 15.439]** | **94.9%** |
| param_readout | 19.1 µs [18.9, 21.1] | 0.1% | — | — |
| host_sgd | 29.8 µs [29.0, 30.9] | 0.2% | — | — |
| apply_params | 0.2 µs [0.1, 0.3] | 0.0% | — | — |
| device_update | — | — | 112.6 µs [110.2, 122.8] | 0.7% |
| tape_drop | 0.8 µs [0.7, 0.9] | 0.0% | 0.8 µs [0.7, 0.8] | 0.0% |
| **step_total** | **16.199 ms [16.133, 16.239]** | 100.0% | **16.224 ms [16.172, 16.300]** | 100.0% |
| init_s（reuse のみ） | — | — | 101.9 µs（1 ラン目実測。§5 参照） | — |

### 3.2 Metal / Apple M4 Max

| フェーズ | fresh 中央値 (範囲) | fresh 比 | reuse 中央値 (範囲) | reuse 比 |
|---|---|---|---|---|
| tape_build | 17.4 µs [15.9, 19.0] | 0.1% | 18.7 µs [17.6, 21.7] | 0.1% |
| leaf_register | 0.5 µs [0.5, 0.6] | 0.0% | 0.2 µs [0.2, 0.3] | 0.0% |
| forward / forward_resident | 2.233 ms [2.159, 2.268] | 12.3% | 1.776 ms [1.708, 1.838] | 9.5% |
| loss_readout | 0.0 µs [0.0, 0.0] | 0.0% | 0.0 µs [0.0, 0.0] | 0.0% |
| **backward** | **15.812 ms [15.742, 15.903]** | **87.0%** | **15.746 ms [15.726, 15.837]** | **84.1%** |
| param_readout | 20.2 µs [19.5, 20.6] | 0.1% | — | — |
| host_sgd | 30.3 µs [28.8, 30.9] | 0.2% | — | — |
| apply_params | 0.2 µs [0.2, 0.2] | 0.0% | — | — |
| device_update | — | — | 1.253 ms [1.193, 1.282] | 6.7% |
| tape_drop | 0.8 µs [0.7, 0.8] | 0.0% | 1.1 µs [1.0, 1.2] | 0.0% |
| **step_total** | **18.180 ms [18.072, 18.254]** | 100.0% | **18.727 ms [18.713, 18.950]** | 100.0% |
| init_s（reuse のみ） | — | — | 26.199 ms（1 ラン目実測。§5 参照） | — |

### 3.3 CPU / DGX Spark GB10（Ubuntu aarch64）

| フェーズ | fresh 中央値 (範囲) | fresh 比 | reuse 中央値 (範囲) | reuse 比 |
|---|---|---|---|---|
| tape_build | 0.2 µs [0.2, 0.2] | 0.0% | 0.1 µs [0.1, 0.1] | 0.0% |
| leaf_register | 1.1 µs [1.1, 1.2] | 0.0% | 0.1 µs [0.1, 0.2] | 0.0% |
| forward / forward_resident | 2.016 ms [1.614, 2.427] | 14.1% | 1.460 ms [1.346, 1.590] | 10.4% |
| loss_readout | 0.1 µs [0.1, 0.2] | 0.0% | 0.0 µs [0.0, 0.1] | 0.0% |
| **backward** | **11.992 ms [11.871, 12.037]** | **84.1%** | **12.478 ms [12.302, 12.492]** | **89.1%** |
| param_readout | 32.7 µs [28.1, 33.6] | 0.2% | — | — |
| host_sgd | 47.1 µs [43.8, 57.2] | 0.3% | — | — |
| apply_params | 0.5 µs [0.4, 0.5] | 0.0% | — | — |
| device_update | — | — | 116.2 µs [115.1, 116.6] | 0.8% |
| tape_drop | 2.2 µs [2.1, 2.3] | 0.0% | 1.7 µs [1.7, 1.7] | 0.0% |
| **step_total** | **14.255 ms [14.001, 14.522]** | 100.0% | **14.010 ms [13.804, 14.182]** | 100.0% |
| init_s（reuse のみ） | — | — | 618.0 µs（1 ラン目実測。§5 参照） | — |

### 3.4 CUDA / DGX Spark GB10（sm_121）

| フェーズ | fresh 中央値 (範囲) | fresh 比 | reuse 中央値 (範囲) | reuse 比 |
|---|---|---|---|---|
| tape_build | 3.3 µs [3.3, 3.4] | 0.0% | 2.9 µs [2.8, 3.1] | 0.0% |
| leaf_register | 0.9 µs [0.9, 1.0] | 0.0% | 0.1 µs [0.1, 0.1] | 0.0% |
| forward / forward_resident | 232.3 µs [230.8, 233.3] | 1.9% | 263.5 µs [261.6, 266.4] | 2.2% |
| loss_readout | 0.0 µs [0.0, 0.0] | 0.0% | 0.0 µs [0.0, 0.0] | 0.0% |
| **backward** | **11.884 ms [11.878, 11.916]** | **97.3%** | **11.895 ms [11.778, 11.918]** | **97.3%** |
| param_readout | 34.2 µs [30.7, 39.8] | 0.3% | — | — |
| host_sgd | 53.1 µs [52.7, 53.7] | 0.4% | — | — |
| apply_params | 0.2 µs [0.2, 0.3] | 0.0% | — | — |
| device_update | — | — | 57.8 µs [57.0, 58.4] | 0.5% |
| tape_drop | 2.5 µs [2.4, 2.5] | 0.0% | 3.0 µs [2.9, 3.1] | 0.0% |
| **step_total** | **12.211 ms [12.205, 12.245]** | 100.0% | **12.224 ms [12.108, 12.249]** | 100.0% |
| init_s（reuse のみ） | — | — | 214.073 ms（1 ラン目実測。§5 参照） | — |

## 4. 支配項トップ 3（バックエンド別・確度付き）

判定基準: (i) 5 ラン中の比率のばらつきが小さいか、(ii) fresh/reuse 間で
同じ結論になるか、(iii) 区間の定義上その区間に閉じた API 呼び出しか
（README「実行時間区間の定義」節参照）。3 条件を満たせば「高」、
1〜2 条件のみなら「中」、区間の定義に前提が残る場合は「低」とする。

### CPU（M4 Max・DGX Spark 共通）

1. **backward（確度: 高）**: 83.6〜95.2%。両環境・両 mode で一貫して圧倒的
   多数を占め、5 ラン範囲の幅も相対的に小さい。CPU の backward は
   `crates/backend-cpu` の VJP 経路（matmul 転置・reduction・rayon 分割を
   含む）に完全に閉じており、区間帰属の曖昧さがない
2. **forward / forward_resident（確度: 中）**: 4.5〜14.1%。DGX Spark CPU
   fresh の範囲が [1.614, 2.427] ms と比較的広く（他 20 コア環境の
   スケジューリング揺らぎと推定）、M4 Max ほど安定しない
3. **device_update / param_readout+host_sgd+apply_params（確度: 中）**:
   合計で 0.3〜1.0% 程度。数値自体は小さいが両環境で一貫した順位

### Metal（M4 Max）

1. **backward（確度: 高）**: 84.1〜87.0%。CPU ほど支配的ではないが依然
   最大区間で、5 ラン範囲の幅も小さい
2. **forward / forward_resident（確度: 中）**: 9.5〜12.3%。CPU の同区間
   より比率が高く、Metal のディスパッチオーバーヘッドが上乗せされている
   と推定されるが、本区間はコマンドバッファのエンコード〜完了待ちを含み
   `docs/backend-metal-command-batching-design.md` のバッチング境界と
   厳密には対応しないため「中」とする
3. **device_update（reuse のみ。確度: 低）**: 6.7%。README の制約注記
   （reuse の非同期発行分は次 step の `forward_resident` へ計上されうる）
   により、本区間の実測値は実際の更新コストを過小評価している可能性が
   ある。真の支配項順位は forward_resident との合算で見るべき

### CUDA（DGX Spark GB10）

1. **backward（確度: 高）**: 97.3%（fresh/reuse とも同値）。3 バックエンド
   中最も支配的で、5 ラン範囲も 12.205〜12.249 ms と極めて狭い
2. **forward / forward_resident（確度: 中）**: 1.9〜2.2%。README の制約
   注記（同期待ちは `loss_readout` へ計上）により、CUDA 側の非同期実行
   モデル（`docs/backend-cuda-async-execution-design.md`）下での区間帰属
   には限界がある
3. **device_update（reuse のみ。確度: 低）**: 0.5%。Metal と同じ理由
   （非同期発行分の次 step 計上の可能性）で「低」とする

## 5. summarize.py 生成表（1 ラン目・`--strict` exit 0）

<details>
<summary>M4 Max（<code>results-m4max-phases-0.4.0.jsonl</code>）</summary>

```
$ python3 summarize.py results/raw/results-m4max-phases-0.4.0.jsonl --strict
（exit 0）
```

#### CPU / fresh

| フェーズ | 中央値 | Q1 | Q3 | step_total 比 |
| --- | --- | --- | --- | --- |
| tape_build | 0.0 µs | 0.0 µs | 0.0 µs | 0.0% |
| leaf_register | 0.4 µs | 0.4 µs | 0.4 µs | 0.0% |
| forward | 705.0 µs | 627.0 µs | 752.1 µs | 4.4% |
| loss_readout | 0.0 µs | 0.0 µs | 0.0 µs | 0.0% |
| backward | 15.385 ms | 15.358 ms | 15.413 ms | 95.4% |
| param_readout | 18.9 µs | 18.8 µs | 19.0 µs | 0.1% |
| host_sgd | 29.0 µs | 27.8 µs | 30.1 µs | 0.2% |
| apply_params | 0.1 µs | 0.1 µs | 0.2 µs | 0.0% |
| tape_drop | 0.7 µs | 0.7 µs | 0.8 µs | 0.0% |
| step_total | 16.133 ms | 16.052 ms | 16.197 ms | 100.0% |

#### CPU / reuse（初期化 init_s: 101.9 µs）

| フェーズ | 中央値 | Q1 | Q3 | step_total 比 |
| --- | --- | --- | --- | --- |
| tape_build | 0.0 µs | 0.0 µs | 0.0 µs | 0.0% |
| leaf_register | 0.1 µs | 0.1 µs | 0.2 µs | 0.0% |
| forward_resident | 726.4 µs | 634.8 µs | 768.8 µs | 4.5% |
| loss_readout | 0.0 µs | 0.0 µs | 0.0 µs | 0.0% |
| backward | 15.381 ms | 15.312 ms | 15.424 ms | 94.9% |
| device_update | 110.2 µs | 109.8 µs | 111.8 µs | 0.7% |
| tape_drop | 0.7 µs | 0.7 µs | 0.9 µs | 0.0% |
| step_total | 16.211 ms | 16.054 ms | 16.287 ms | 100.0% |

#### Metal / fresh

| フェーズ | 中央値 | Q1 | Q3 | step_total 比 |
| --- | --- | --- | --- | --- |
| tape_build | 18.9 µs | 15.4 µs | 25.2 µs | 0.1% |
| leaf_register | 0.5 µs | 0.5 µs | 0.7 µs | 0.0% |
| forward | 2.228 ms | 2.123 ms | 2.374 ms | 12.2% |
| loss_readout | 0.0 µs | 0.0 µs | 0.1 µs | 0.0% |
| backward | 15.812 ms | 15.768 ms | 15.904 ms | 86.8% |
| param_readout | 19.5 µs | 19.3 µs | 20.3 µs | 0.1% |
| host_sgd | 28.8 µs | 28.0 µs | 29.8 µs | 0.2% |
| apply_params | 0.2 µs | 0.2 µs | 0.2 µs | 0.0% |
| tape_drop | 0.7 µs | 0.7 µs | 0.8 µs | 0.0% |
| step_total | 18.212 ms | 18.038 ms | 18.379 ms | 100.0% |

#### Metal / reuse（初期化 init_s: 26.199 ms）

| フェーズ | 中央値 | Q1 | Q3 | step_total 比 |
| --- | --- | --- | --- | --- |
| tape_build | 18.4 µs | 16.2 µs | 24.2 µs | 0.1% |
| leaf_register | 0.2 µs | 0.2 µs | 0.2 µs | 0.0% |
| forward_resident | 1.776 ms | 1.460 ms | 2.072 ms | 9.5% |
| loss_readout | 0.0 µs | 0.0 µs | 0.1 µs | 0.0% |
| backward | 15.730 ms | 15.517 ms | 15.831 ms | 84.1% |
| device_update | 1.222 ms | 1.097 ms | 1.411 ms | 6.5% |
| tape_drop | 1.1 µs | 1.0 µs | 1.3 µs | 0.0% |
| step_total | 18.713 ms | 18.353 ms | 19.321 ms | 100.0% |

</details>

<details>
<summary>DGX Spark GB10（<code>results-dgx-phases-0.4.0.jsonl</code>）</summary>

```
$ python3 summarize.py results/raw/results-dgx-phases-0.4.0.jsonl --strict
（exit 0）
```

#### CPU / fresh

| フェーズ | 中央値 | Q1 | Q3 | step_total 比 |
| --- | --- | --- | --- | --- |
| tape_build | 0.2 µs | 0.2 µs | 0.2 µs | 0.0% |
| leaf_register | 1.1 µs | 1.0 µs | 1.3 µs | 0.0% |
| forward | 2.024 ms | 1.374 ms | 2.710 ms | 14.2% |
| loss_readout | 0.1 µs | 0.0 µs | 0.2 µs | 0.0% |
| backward | 11.918 ms | 11.855 ms | 12.006 ms | 83.6% |
| param_readout | 29.7 µs | 27.4 µs | 33.6 µs | 0.2% |
| host_sgd | 57.2 µs | 42.2 µs | 422.9 µs | 0.4% |
| apply_params | 0.4 µs | 0.4 µs | 0.5 µs | 0.0% |
| tape_drop | 2.2 µs | 1.8 µs | 214.1 µs | 0.0% |
| step_total | 14.255 ms | 13.602 ms | 14.879 ms | 100.0% |

#### CPU / reuse（初期化 init_s: 618.0 µs）

| フェーズ | 中央値 | Q1 | Q3 | step_total 比 |
| --- | --- | --- | --- | --- |
| tape_build | 0.1 µs | 0.1 µs | 0.2 µs | 0.0% |
| leaf_register | 0.1 µs | 0.1 µs | 0.2 µs | 0.0% |
| forward_resident | 1.468 ms | 1.354 ms | 1.763 ms | 10.5% |
| loss_readout | 0.1 µs | 0.0 µs | 0.1 µs | 0.0% |
| backward | 12.463 ms | 12.217 ms | 12.482 ms | 89.0% |
| device_update | 115.9 µs | 114.6 µs | 116.2 µs | 0.8% |
| tape_drop | 1.7 µs | 1.6 µs | 1.9 µs | 0.0% |
| step_total | 14.010 ms | 13.813 ms | 14.376 ms | 100.0% |

#### CUDA / fresh

| フェーズ | 中央値 | Q1 | Q3 | step_total 比 |
| --- | --- | --- | --- | --- |
| tape_build | 3.3 µs | 3.2 µs | 3.4 µs | 0.0% |
| leaf_register | 0.9 µs | 0.9 µs | 1.0 µs | 0.0% |
| forward | 232.5 µs | 231.4 µs | 233.9 µs | 1.9% |
| loss_readout | 0.0 µs | 0.0 µs | 0.0 µs | 0.0% |
| backward | 11.916 ms | 11.897 ms | 11.961 ms | 97.3% |
| param_readout | 37.8 µs | 32.4 µs | 42.6 µs | 0.3% |
| host_sgd | 53.3 µs | 52.1 µs | 53.9 µs | 0.4% |
| apply_params | 0.2 µs | 0.2 µs | 0.3 µs | 0.0% |
| tape_drop | 2.4 µs | 2.3 µs | 2.5 µs | 0.0% |
| step_total | 12.245 ms | 12.227 ms | 12.295 ms | 100.0% |

#### CUDA / reuse（初期化 init_s: 214.073 ms）

| フェーズ | 中央値 | Q1 | Q3 | step_total 比 |
| --- | --- | --- | --- | --- |
| tape_build | 3.1 µs | 3.0 µs | 3.3 µs | 0.0% |
| leaf_register | 0.1 µs | 0.1 µs | 0.1 µs | 0.0% |
| forward_resident | 266.4 µs | 265.4 µs | 267.9 µs | 2.2% |
| loss_readout | 0.0 µs | 0.0 µs | 0.0 µs | 0.0% |
| backward | 11.918 ms | 11.904 ms | 11.940 ms | 97.3% |
| device_update | 57.0 µs | 56.7 µs | 57.2 µs | 0.5% |
| tape_drop | 2.9 µs | 2.8 µs | 3.0 µs | 0.0% |
| step_total | 12.249 ms | 12.235 ms | 12.273 ms | 100.0% |

</details>

## 6. イシューの候補仮説の採否

イシュー #1010 が挙げていた候補仮説（`to_vec`/`from_slice`/
`apply_parameters` の再確保・小形状での rayon 分割オーバーヘッド・`bind`
の毎 step 再構築）は、3 バックエンド全てで **棄却** される。これらの
候補が該当しうる区間（`param_readout`・`host_sgd`・`apply_params`・
`leaf_register`）は合計しても 0.1〜1.0% に留まり、支配項は一貫して
**backward 単独**（83.6〜97.3%）である。

`bind`（`leaf_register` に含まれる。README「実行時間区間の定義」参照）は
全環境・全 mode で 0.0〜1.3 µs であり、step_total 比 0.0% で計測誤差の
範囲内にとどまる。#1026 で `bind`/`to_vec` 系の最適化が既に main へ
マージ済みだが、今回の 0.4.0 計測（#1026 以前のビルド）時点でも当該区間
はそもそも支配的ではなかったことが確認できる。

**backward 内部の切り分け**: CPU の backward（VJP の転置コピー・小形状
GEMM の rayon 分割・reduction 等）は本ハーネスでは区間分解できない
（0.4.0 は crates.io 公開版のため内部計測点を持たない）。macOS
`sample`/`xctrace`・Linux `perf` によるホットスポット採取は
**未実施**（本セッションが到達できた環境・時間内では実施しなかった。
捏造しない方針により「未実施」と明記する）。

## 7. 版差の注記

本計測は `fandhe-ai =0.4.0`（crates.io 公開版。2026-08-29 公開）を対象と
する。現 main（本ドキュメント作成時点）は以下の PR が既にマージ済みで
あり、0.4.0 とは学習 1 step の内訳が異なりうる:

| PR | 内容 |
|---|---|
| #1078 | MSE loss の reduction を単一カーネルへ融合 |
| #1079 | Linear の epilogue 融合（bias + ReLU）を学習経路へ結線 |
| #1080 | view 系ノード（reshape/transpose）の再計算方式化 |
| #1081 | 学習ループで tape を再利用可能にする（ノードクリア API） |
| #1026・#1027 | `bind`/`to_vec` 系の再確保削減（イシュー本文の候補仮説対象） |

これらは主に forward・tape_build・param 更新経路に効くと想定される
変更であり、本計測が示す「backward が支配的」という結論そのものへの
影響は小さいと考えられるが、実際の比率は再計測でのみ確定できる。次回
crates.io 公開後、同一手順（`--task train --phases`・5 ラン中央値）での
再計測を推奨する（#1083 と同じ枠組み。新規 Issue は起票せず、優先順位
提案として #1008 へコメントする）。

**v0.6.0 ピンでの再計測は §10〜§13（イシュー #1145）を参照。**

## 8. 後続 Issue 優先順位の更新案

#1008 配下の残る open Issue（#1011 CUDA 同期廃止・#1015 Metal コマンド
バッファ統合・#1025 CUDA fresh N=2048 固有オーバーヘッド）は、いずれも
本計測が「支配項ではない」と示した区間（forward・device_update・
loss_readout 付近の同期待ち）に対する最適化である。backward が
83.6〜97.3% を占める現状を踏まえると、これらの相対優先度は当初想定より
下がる可能性がある。一方で backward 自体の内部最適化（matmul VJP の
transpose コピー・小形状閾値・reduction 融合）を対象にした Issue は
#1008 配下に現時点で存在しない。

提案（`out-of-scope-tracking.md` に従い提案に留め、Issue の付け替え・
新規起票は行わない。#1008 へのコメントで提示する）:

1. **最優先候補**: backward 内部（`matmul` VJP・transpose ゼロコピー
   経路の gemm 結線状況・reduction）のプロファイリングと最適化候補の
   洗い出し（新規 Issue 化はユーザー判断に委ねる）
2. #1011（CUDA 同期廃止）・#1015（Metal コマンドバッファ統合）は
   forward/device_update 区間（CUDA 1.9〜2.2%・Metal 6.7〜12.3%）が
   対象であり、backward 最適化と比べ総 step_total 短縮への寄与は小さい
   と推定されるため優先度を下げる候補
3. #1025（CUDA fresh N=2048 固有オーバーヘッド）は GEMM 単体の形状依存
   問題であり、本計測（学習 1 step、size=64 固定）の対象外。優先順位の
   判断材料としては別軸で扱う

### 8.1 更新履歴（2026-09-05・イシュー #1151・v0.6.0 実測反映）

上記の提案 1〜3 は 0.4.0 実測（イシュー #1010）時点のもの。#1008 は
クローズ済み（配下の #1011・#1015・#1025 はいずれも完了）であり、
本節の提案 2・3 が対象とした Issue は既に解消済みである。backward が
支配項であるという結論・提案 1（backward 内部の最適化候補洗い出し）は
v0.6.0 実測（§11・§12）でも変わらない。v0.6.0 実測を踏まえた更新版の
後続 Issue 優先順位・起票案は §15.4・§15.6 を参照。

## 9. 検証結果サマリ

- `scripts/bench/framework-compare/summarize_test.py`: 全 pass（新規
  8 テスト追加。`_safe_phase_time_s` の境界値・phase 行の 0 値許容・
  `step_total`/`init_s` の 0 値は引き続き無効）
- `python3 summarize.py results/raw/results-m4max-phases-0.4.0.jsonl
  --strict`: exit 0
- `python3 summarize.py results/raw/results-dgx-phases-0.4.0.jsonl
  --strict`: exit 0
- `python3 summarize.py --strict`（既存 `results/raw/*.jsonl` 全件。
  回帰確認）: 変更前後で stdout/stderr が完全一致（既存 exit 2 は
  Burn(wgpu) Metal GEMM の既知不一致 `docs/perf/burn-wgpu-metal-gemm-
  zero-result.md` によるもので本 PR と無関係）
- `results/raw/skipped-m4max-phases-0.4.0.log`・
  `results/raw/skipped-dgx-phases-0.4.0.log`: いずれも空（全実行成功）
- `cargo fmt --all --check && cargo clippy --workspace --all-targets --
  -D warnings && cargo test -p bench-common -p bench-fandhe`
  （`scripts/bench/framework-compare/` workspace。Rust 側は無変更のため
  回帰確認のみ）: pass

## 10. v0.6.0 再計測（イシュー #1145）

### 10.1 目的・対応

イシュー #1145「v0.6.0 ピンの学習・推論 1 step を CPU/CUDA/Metal でフェーズ
分解実測し支配項を更新する」に対応する再計測記録。§7 のとおり §1〜9 は
`fandhe-ai =0.4.0`（2026-08-29 公開）時点の実測であり、その後 main へ
マージされた以下の PR は本計測が示す内訳（backward 支配）に影響しうる:

| PR | 内容 |
|---|---|
| #1078 | MSE loss の reduction を単一カーネルへ融合 |
| #1079 | Linear の epilogue 融合（bias + ReLU）を学習経路へ結線 |
| #1080 | view 系ノード（reshape/transpose）の再計算方式化 |
| #1081 | 学習ループで tape を再利用可能にする（ノードクリア API） |
| #1011 | CUDA 都度同期の廃止 |
| #1059 | resident forward/backward 経路の変更（reuse モード） |

対象は `fandhe-ai =0.6.0`（crates.io 公開版。2026-09-02 公開。
`scripts/bench/framework-compare/bench-fandhe/Cargo.toml` の現行ピン）。
**単一系列（registry 解決）のみを対象とし、HEAD path patch の参考系列は
本イシューの対象外とする**。本イシュー実装時点の HEAD の Rust コードは
v0.6.0 タグと同一ではない（`git diff v0.6.0..HEAD -- crates/` は
`crates/backend-cpu/src/gemm_blis/mod.rs`・`crates/backend-cpu/src/rmsnorm.rs`
〈CPU GEMM 本番経路変更 #1174 を含む〉、`crates/backend-cuda`・
`crates/backend-metal` 配下の多数のカーネル・テスト変更を含む差分を
示す）。参考系列を追加しない理由は「HEAD と v0.6.0 のコードが同一だから」
ではなく、本イシューのスコープが registry の `=0.6.0` 版の再計測に限定
されるためである。したがって §10〜§12 の実測値・変化率は
**`fandhe-ai =0.6.0`（registry 解決）というピン留めされたバージョンに
限定された結果**であり、本 PR 時点の HEAD（`main`）における性能を
示すものではない点に注意する。

### 10.2 計測プロトコル

- モデル・warmup/iters・5 回計測方針は §2 と同一（`--size 64`・
  `warmup=20`・`iters=80`・1 ラン目を主表の生成元、残り 4 ラン分を
  `-extra.jsonl` へ分離）
- 実行コマンド（両環境共通）:

  ```bash
  ./target/release/bench-fandhe --task train --device <cpu|metal|cuda> \
    --size 64 --mode <fresh|reuse> --phases --out <dest.jsonl>
  ```

- 計測環境:
  - 環境 18: Apple M4 Max・macOS 26.6.2 (25G83)・rustc/cargo 1.96.0
    （ローカル直接実行。デバイス cpu・metal）
  - 環境 19: DGX Spark GB10（sm_121）・NVIDIA driver 580.173.02・
    CUDA 13.0 V13.0.88（nvcc）・Ubuntu 24.04 aarch64・rustc/cargo 1.97.0
    （SSH リモート実行。デバイス cuda・cpu。イシュー固有の隔離ディレクトリ
    `~/work/rust-ai-library-run-1145`・`CARGO_TARGET_DIR` 分離で他 Issue の
    並行実行との競合を避けた。`docs/real-hardware-verification-env.md`
    §2〜4 準拠）
- 計測日: 2026-09-05
- 実行前後で `nvidia-smi --query-gpu=utilization.gpu`・
  `--query-compute-apps` を確認し、GPU 使用率 0%・既知の常駐プロセス
  （計測に無関係と確認済み）以外のプロセスがないことを確認した。
  M4 Max はローカルマシン上で他
  worktree（別 Issue）の cargo テストが並行実行されており負荷混入が
  あったため §10.4 に記録する（値は平滑化・除外しない）
- `cargo tree -p bench-fandhe --depth 1 --locked` は両環境とも
  `fandhe-ai v0.6.0`（registry 解決・`(path: …)` なし）を確認済み

### 10.3 集計スニペット（再現用）

```python
import json, statistics, collections
rows = [json.loads(l) for f in (
    "results/raw/results-<env>-phases-0.6.0.jsonl",
    "results/raw/results-<env>-phases-0.6.0-extra.jsonl",
) for l in open(f) if l.strip()]
g = collections.defaultdict(list)
for r in rows:
    assert r["task"] == "train_phases" and r["version"] == "0.6.0"
    g[(r["device"], r["mode"], r["phase"])].append(r["median_s"])
for k, v in sorted(g.items()):
    assert len(v) == 5, k
    print(k, statistics.median(v), min(v), max(v))
```

「step_total 比」の定義は §3 と同一（当該行の 5 ラン中央値 ÷ step_total
の 5 ラン中央値）。reuse の `init_s` は 1 ラン目実測値（§3 と同じ扱い）。

### 10.4 計測時の負荷混入について

M4 Max 実測中、同一マシン上で別 worktree（別 Issue の並列実装セッション）
の `cargo test` が CPU を占有する時間帯があった（計測直前に `ps` で
確認・load average 約 5〜7）。DGX Spark GB10 は計測前後とも
`utilization.gpu` 0%・既知の常駐プロセス（計測に無関係と確認済み）
以外のプロセスなしを確認しており、負荷混入は M4 Max 側のみと考えられる。
`min–max` の範囲を平滑化せずそのまま記載することで影響を可視化した
（§11 の範囲列を参照。とくに `forward`/`tape_build` 系の µs オーダーの
フェーズで範囲が広い傾向がある）。値そのものの再計測は行わず、範囲を
含めて実測事実として記録する。

## 11. 内訳表（v0.6.0・5 ラン中央値・範囲・step_total 比）

生データ: `results/raw/results-m4max-phases-0.6.0.jsonl`（+ `-extra.jsonl`）・
`results/raw/results-dgx-phases-0.6.0.jsonl`（+ `-extra.jsonl`）。両環境とも
`skipped-*-phases-0.6.0.log` は空（全実行成功。捏造していないことの証跡）、
`python3 summarize.py <jsonl> --strict` は主表（1 ラン目）で exit 0。
`-extra.jsonl`（2〜5 ラン目・4 ラン分を 1 ファイルへ集約）を単独で
`--strict` に渡すと exit 2 になるが、これは §3 の 0.4.0 実測（
`results-*-phases-0.4.0-extra.jsonl`）と同一の既知挙動であり本計測固有の
不具合ではない（`--strict` の単一ファイル検証は 1 run 1 行を前提とする
ため、同一 phase_index が複数ラン分重複する `-extra.jsonl` は「重複」
として無効行扱いになる。§14 参照）。

### 11.1 CPU / Apple M4 Max

| フェーズ | fresh 中央値 (範囲) | fresh 比 | reuse 中央値 (範囲) | reuse 比 |
|---|---|---|---|---|
| tape_build | 0.0 µs [0.0 µs, 0.1 µs] | 0.0% | 0.0 µs [0.0 µs, 0.0 µs] | 0.0% |
| leaf_register | 0.6 µs [0.5 µs, 1.4 µs] | 0.0% | 0.2 µs [0.2 µs, 0.4 µs] | 0.0% |
| forward | 562.7 µs [510.0 µs, 621.4 µs] | 3.3% | 501.3 µs [481.6 µs, 520.7 µs] | 6.3% |
| loss_readout | 0.0 µs [0.0 µs, 0.1 µs] | 0.0% | 0.0 µs [0.0 µs, 0.1 µs] | 0.0% |
| backward | 16.540 ms [16.489 ms, 16.959 ms] | 96.5% | 7.266 ms [7.244 ms, 7.337 ms] | 91.7% |
| param_readout | 21.7 µs [21.3 µs, 25.1 µs] | 0.1% | — | — |
| host_sgd | 33.2 µs [32.1 µs, 34.3 µs] | 0.2% | — | — |
| apply_params | 0.4 µs [0.3 µs, 0.8 µs] | 0.0% | — | — |
| device_update | — | — | 137.7 µs [137.4 µs, 138.8 µs] | 1.7% |
| tape_drop | 0.8 µs [0.8 µs, 1.9 µs] | 0.0% | 0.5 µs [0.4 µs, 0.7 µs] | 0.0% |
| **step_total** | **17.137 ms [17.125 ms, 17.528 ms]** | 100.0% | **7.926 ms [7.911 ms, 7.998 ms]** | 100.0% |
| init_s（reuse のみ） | — | — | 185.0 µs（1 ラン目実測） | — |

### 11.2 Metal / Apple M4 Max

| フェーズ | fresh 中央値 (範囲) | fresh 比 | reuse 中央値 (範囲) | reuse 比 |
|---|---|---|---|---|
| tape_build | 38.7 µs [16.7 µs, 71.7 µs] | 0.2% | 33.2 µs [14.2 µs, 54.5 µs] | 0.4% |
| leaf_register | 0.9 µs [0.5 µs, 1.6 µs] | 0.0% | 0.3 µs [0.1 µs, 0.4 µs] | 0.0% |
| forward | 1.985 ms [1.882 ms, 2.013 ms] | 10.5% | 1.375 ms [1.304 ms, 1.426 ms] | 15.3% |
| loss_readout | 0.1 µs [0.0 µs, 0.1 µs] | 0.0% | 0.0 µs [0.0 µs, 0.1 µs] | 0.0% |
| backward | 16.871 ms [16.718 ms, 17.087 ms] | 88.9% | 7.548 ms [7.474 ms, 7.652 ms] | 84.1% |
| param_readout | 22.6 µs [21.1 µs, 25.3 µs] | 0.1% | — | — |
| host_sgd | 31.8 µs [31.2 µs, 34.5 µs] | 0.2% | — | — |
| apply_params | 0.4 µs [0.2 µs, 0.8 µs] | 0.0% | — | — |
| device_update | — | — | 84.5 µs [69.2 µs, 101.3 µs] | 0.9% |
| tape_drop | 1.0 µs [0.7 µs, 1.7 µs] | 0.0% | 0.5 µs [0.3 µs, 0.9 µs] | 0.0% |
| **step_total** | **18.985 ms [18.776 ms, 19.259 ms]** | 100.0% | **8.979 ms [8.918 ms, 9.228 ms]** | 100.0% |
| init_s（reuse のみ） | — | — | 28.878 ms（1 ラン目実測） | — |

### 11.3 CPU / DGX Spark GB10（Ubuntu aarch64）

| フェーズ | fresh 中央値 (範囲) | fresh 比 | reuse 中央値 (範囲) | reuse 比 |
|---|---|---|---|---|
| tape_build | 0.2 µs [0.2 µs, 0.3 µs] | 0.0% | 0.3 µs [0.2 µs, 0.3 µs] | 0.0% |
| leaf_register | 1.3 µs [1.2 µs, 1.3 µs] | 0.0% | 0.3 µs [0.2 µs, 0.3 µs] | 0.0% |
| forward | 1.591 ms [1.416 ms, 1.657 ms] | 11.7% | 1.644 ms [1.300 ms, 1.836 ms] | 20.3% |
| loss_readout | 0.1 µs [0.1 µs, 0.1 µs] | 0.0% | 0.1 µs [0.1 µs, 0.2 µs] | 0.0% |
| backward | 11.606 ms [11.538 ms, 11.729 ms] | 85.1% | 6.082 ms [6.043 ms, 6.195 ms] | 75.1% |
| param_readout | 32.4 µs [31.2 µs, 33.4 µs] | 0.2% | — | — |
| host_sgd | 257.8 µs [251.9 µs, 285.4 µs] | 1.9% | — | — |
| apply_params | 0.5 µs [0.4 µs, 0.6 µs] | 0.0% | — | — |
| device_update | — | — | 261.8 µs [176.6 µs, 268.8 µs] | 3.2% |
| tape_drop | 1.7 µs [1.6 µs, 1.8 µs] | 0.0% | 1.5 µs [1.4 µs, 1.6 µs] | 0.0% |
| **step_total** | **13.631 ms [13.539 ms, 13.709 ms]** | 100.0% | **8.093 ms [7.543 ms, 8.374 ms]** | 100.0% |
| init_s（reuse のみ） | — | — | 1.106 ms（1 ラン目実測） | — |

### 11.4 CUDA / DGX Spark GB10（sm_121）

| フェーズ | fresh 中央値 (範囲) | fresh 比 | reuse 中央値 (範囲) | reuse 比 |
|---|---|---|---|---|
| tape_build | 3.9 µs [3.4 µs, 5.0 µs] | 0.0% | 3.0 µs [2.6 µs, 3.1 µs] | 0.1% |
| leaf_register | 1.1 µs [1.0 µs, 1.2 µs] | 0.0% | 0.1 µs [0.1 µs, 0.1 µs] | 0.0% |
| forward | 181.3 µs [177.1 µs, 188.1 µs] | 1.5% | 154.9 µs [149.4 µs, 158.3 µs] | 2.8% |
| loss_readout | 0.0 µs [0.0 µs, 0.0 µs] | 0.0% | 0.0 µs [0.0 µs, 0.0 µs] | 0.0% |
| backward | 11.500 ms [11.393 ms, 12.951 ms] | 97.5% | 5.398 ms [5.395 ms, 5.465 ms] | 96.4% |
| param_readout | 37.8 µs [27.3 µs, 40.9 µs] | 0.3% | — | — |
| host_sgd | 60.7 µs [52.9 µs, 61.4 µs] | 0.5% | — | — |
| apply_params | 0.3 µs [0.2 µs, 0.3 µs] | 0.0% | — | — |
| device_update | — | — | 40.6 µs [40.2 µs, 67.8 µs] | 0.7% |
| tape_drop | 2.2 µs [2.0 µs, 2.2 µs] | 0.0% | 0.9 µs [0.9 µs, 1.0 µs] | 0.0% |
| **step_total** | **11.789 ms [11.658 ms, 13.244 ms]** | 100.0% | **5.598 ms [5.592 ms, 5.670 ms]** | 100.0% |
| init_s（reuse のみ） | — | — | 211.249 ms（1 ラン目実測） | — |

## 12. 0.4.0 → 0.6.0 の差分

`step_total` は 8 系列（4 環境 × fresh/reuse）中 6 系列で短縮した
（-3.5%〜-54.2%）。とくに reuse モード（4 系列全て）は #1078〜#1081・
#1059 の効果で 0.4.0 比 -42.2%〜-54.2% と大幅に短縮している。一方
fresh モードは 4 系列中 2 系列（M4 Max の cpu +5.8%・metal +4.4%）で
増加しており、残り 2 系列（DGX Spark GB10 の cpu -4.4%・cuda -3.5%）の
短縮とは傾向が異なる。M4 Max fresh の増加は §10.4 記載の計測時負荷
混入（同一マシン上の別 worktree の `cargo test` 並行実行）が寄与した
可能性があり、fresh モードでの 0.4.0 → 0.6.0 比較は reuse ほど強い
根拠を持たない（比較上の制約として明記する）。

**支配項トップ 3 の再判定**: backward は全 8 系列（4 環境 × fresh/reuse）
で引き続き 75.1%〜97.5% を占め、支配項であるという §4 の結論は不変。
唯一 DGX CPU reuse で 89.1%→75.1%（-13.9pt）とやや低下し forward_resident
が 10.4%→20.3%（+9.9pt）へ増加した。この変化を §4 の確度基準
（(i) 5 ラン中の比率のばらつきが小さいか、(ii) fresh/reuse 間で同じ
結論になるか、(iii) 区間の定義上その区間に閉じた API 呼び出しか）で
再評価すると、(i) は本計測の DGX CPU reuse の範囲 [7.543 ms, 8.374 ms]
（step_total 比で見て backward 単独の 5 ラン範囲も §11.3 のとおり
[6.043 ms, 6.195 ms] と相対的には安定）で大きな乱れはなく、(ii) は
同環境の fresh（85.1%）と reuse（75.1%）で backward が引き続き最大項
という結論自体は一致するが比率の差が §4 時点（0.4.0）より拡大しており
一致の強さはやや弱まった、(iii) は区間定義（CPU backward は
`crates/backend-cpu` の VJP 経路に閉じる）自体に変更はない。3 条件の
うち (iii) は満たし (i)(ii) は「弱まったが崩れてはいない」というのが
実測に即した評価であり、§4 の「高（3 条件満たす）」から「中
（1〜2 条件のみ）」への格下げが妥当な範囲というのが実測ベースの結論
である。M4 Max Metal reuse の
`device_update` は 6.7%→0.9%（-5.7pt）まで縮小しており、#1015/#1017
（Metal コマンドバッファ統合）以前の計測との単純比較はできないものの、
§8 提案 2（#1011・#1015 の優先度を下げる候補）の妥当性を補強する材料
になりうる。改善提案・原因分析そのものは後続 Issue（#1118 配下）へ
引き継ぐ（下記スコープ外参照）。

### 12.1 cpu / fresh (Apple M4 Max)

| フェーズ | 0.4.0 中央値 (比) | 0.6.0 中央値 (比) | 変化率 | 比の変化 |
|---|---|---|---|---|
| tape_build | 0.0 µs (0.0%) | 0.0 µs (0.0%) | +2.4% | -0.0pt |
| leaf_register | 0.4 µs (0.0%) | 0.6 µs (0.0%) | +33.6% | +0.0pt |
| forward | 726.5 µs (4.5%) | 562.7 µs (3.3%) | -22.5% | -1.2pt |
| loss_readout | 0.0 µs (0.0%) | 0.0 µs (0.0%) | +0.0% | -0.0pt |
| backward | 15.413 ms (95.1%) | 16.540 ms (96.5%) | +7.3% | +1.4pt |
| param_readout | 19.1 µs (0.1%) | 21.7 µs (0.1%) | +13.6% | +0.0pt |
| host_sgd | 29.8 µs (0.2%) | 33.2 µs (0.2%) | +11.5% | +0.0pt |
| apply_params | 0.2 µs (0.0%) | 0.4 µs (0.0%) | +100.5% | +0.0pt |
| tape_drop | 0.8 µs (0.0%) | 0.8 µs (0.0%) | +11.2% | +0.0pt |
| **step_total** | 16.199 ms (100.0%) | 17.137 ms (100.0%) | +5.8% | +0.0pt |

### 12.2 cpu / reuse (Apple M4 Max)

| フェーズ | 0.4.0 中央値 (比) | 0.6.0 中央値 (比) | 変化率 | 比の変化 |
|---|---|---|---|---|
| tape_build | 0.0 µs (0.0%) | 0.0 µs (0.0%) | +2.4% | +0.0pt |
| leaf_register | 0.2 µs (0.0%) | 0.2 µs (0.0%) | +25.1% | +0.0pt |
| forward_resident | 726.4 µs (4.5%) | 501.3 µs (6.3%) | -31.0% | +1.8pt |
| loss_readout | 0.0 µs (0.0%) | 0.0 µs (0.0%) | +2.4% | +0.0pt |
| backward | 15.396 ms (94.9%) | 7.266 ms (91.7%) | -52.8% | -3.2pt |
| device_update | 112.6 µs (0.7%) | 137.7 µs (1.7%) | +22.3% | +1.0pt |
| tape_drop | 0.8 µs (0.0%) | 0.5 µs (0.0%) | -27.9% | +0.0pt |
| **step_total** | 16.224 ms (100.0%) | 7.926 ms (100.0%) | -51.1% | +0.0pt |

### 12.3 metal / fresh (Apple M4 Max)

| フェーズ | 0.4.0 中央値 (比) | 0.6.0 中央値 (比) | 変化率 | 比の変化 |
|---|---|---|---|---|
| tape_build | 17.4 µs (0.1%) | 38.7 µs (0.2%) | +122.1% | +0.1pt |
| leaf_register | 0.5 µs (0.0%) | 0.9 µs (0.0%) | +69.0% | +0.0pt |
| forward | 2.233 ms (12.3%) | 1.985 ms (10.5%) | -11.1% | -1.8pt |
| loss_readout | 0.0 µs (0.0%) | 0.1 µs (0.0%) | +47.6% | +0.0pt |
| backward | 15.812 ms (87.0%) | 16.871 ms (88.9%) | +6.7% | +1.9pt |
| param_readout | 20.2 µs (0.1%) | 22.6 µs (0.1%) | +11.9% | +0.0pt |
| host_sgd | 30.3 µs (0.2%) | 31.8 µs (0.2%) | +5.0% | +0.0pt |
| apply_params | 0.2 µs (0.0%) | 0.4 µs (0.0%) | +99.5% | +0.0pt |
| tape_drop | 0.8 µs (0.0%) | 1.0 µs (0.0%) | +27.4% | +0.0pt |
| **step_total** | 18.180 ms (100.0%) | 18.985 ms (100.0%) | +4.4% | +0.0pt |

### 12.4 metal / reuse (Apple M4 Max)

| フェーズ | 0.4.0 中央値 (比) | 0.6.0 中央値 (比) | 変化率 | 比の変化 |
|---|---|---|---|---|
| tape_build | 18.7 µs (0.1%) | 33.2 µs (0.4%) | +77.1% | +0.3pt |
| leaf_register | 0.2 µs (0.0%) | 0.3 µs (0.0%) | +59.3% | +0.0pt |
| forward_resident | 1.776 ms (9.5%) | 1.375 ms (15.3%) | -22.6% | +5.8pt |
| loss_readout | 0.0 µs (0.0%) | 0.0 µs (0.0%) | +0.0% | +0.0pt |
| backward | 15.746 ms (84.1%) | 7.548 ms (84.1%) | -52.1% | -0.0pt |
| device_update | 1.253 ms (6.7%) | 84.5 µs (0.9%) | -93.3% | -5.7pt |
| tape_drop | 1.1 µs (0.0%) | 0.5 µs (0.0%) | -58.5% | -0.0pt |
| **step_total** | 18.727 ms (100.0%) | 8.979 ms (100.0%) | -52.1% | +0.0pt |

### 12.5 cpu / fresh (DGX Spark GB10)

| フェーズ | 0.4.0 中央値 (比) | 0.6.0 中央値 (比) | 変化率 | 比の変化 |
|---|---|---|---|---|
| tape_build | 0.2 µs (0.0%) | 0.2 µs (0.0%) | +16.7% | +0.0pt |
| leaf_register | 1.1 µs (0.0%) | 1.3 µs (0.0%) | +14.4% | +0.0pt |
| forward | 2.016 ms (14.1%) | 1.591 ms (11.7%) | -21.1% | -2.5pt |
| loss_readout | 0.1 µs (0.0%) | 0.1 µs (0.0%) | +6.7% | +0.0pt |
| backward | 11.992 ms (84.1%) | 11.606 ms (85.1%) | -3.2% | +1.0pt |
| param_readout | 32.7 µs (0.2%) | 32.4 µs (0.2%) | -0.7% | +0.0pt |
| host_sgd | 47.1 µs (0.3%) | 257.8 µs (1.9%) | +447.5% | +1.6pt |
| apply_params | 0.5 µs (0.0%) | 0.5 µs (0.0%) | +10.0% | +0.0pt |
| tape_drop | 2.2 µs (0.0%) | 1.7 µs (0.0%) | -20.7% | -0.0pt |
| **step_total** | 14.255 ms (100.0%) | 13.631 ms (100.0%) | -4.4% | +0.0pt |

### 12.6 cpu / reuse (DGX Spark GB10)

| フェーズ | 0.4.0 中央値 (比) | 0.6.0 中央値 (比) | 変化率 | 比の変化 |
|---|---|---|---|---|
| tape_build | 0.1 µs (0.0%) | 0.3 µs (0.0%) | +216.7% | +0.0pt |
| leaf_register | 0.1 µs (0.0%) | 0.3 µs (0.0%) | +83.3% | +0.0pt |
| forward_resident | 1.460 ms (10.4%) | 1.644 ms (20.3%) | +12.6% | +9.9pt |
| loss_readout | 0.0 µs (0.0%) | 0.1 µs (0.0%) | +166.7% | +0.0pt |
| backward | 12.478 ms (89.1%) | 6.082 ms (75.1%) | -51.3% | -13.9pt |
| device_update | 116.2 µs (0.8%) | 261.8 µs (3.2%) | +125.4% | +2.4pt |
| tape_drop | 1.7 µs (0.0%) | 1.5 µs (0.0%) | -8.6% | +0.0pt |
| **step_total** | 14.010 ms (100.0%) | 8.093 ms (100.0%) | -42.2% | +0.0pt |

### 12.7 cuda / fresh (DGX Spark GB10)

| フェーズ | 0.4.0 中央値 (比) | 0.6.0 中央値 (比) | 変化率 | 比の変化 |
|---|---|---|---|---|
| tape_build | 3.3 µs (0.0%) | 3.9 µs (0.0%) | +15.8% | +0.0pt |
| leaf_register | 0.9 µs (0.0%) | 1.1 µs (0.0%) | +11.9% | +0.0pt |
| forward | 232.3 µs (1.9%) | 181.3 µs (1.5%) | -22.0% | -0.4pt |
| loss_readout | 0.0 µs (0.0%) | 0.0 µs (0.0%) | +50.0% | +0.0pt |
| backward | 11.884 ms (97.3%) | 11.500 ms (97.5%) | -3.2% | +0.2pt |
| param_readout | 34.2 µs (0.3%) | 37.8 µs (0.3%) | +10.7% | +0.0pt |
| host_sgd | 53.1 µs (0.4%) | 60.7 µs (0.5%) | +14.4% | +0.1pt |
| apply_params | 0.2 µs (0.0%) | 0.3 µs (0.0%) | +13.3% | +0.0pt |
| tape_drop | 2.5 µs (0.0%) | 2.2 µs (0.0%) | -13.2% | -0.0pt |
| **step_total** | 12.211 ms (100.0%) | 11.789 ms (100.0%) | -3.5% | +0.0pt |

### 12.8 cuda / reuse (DGX Spark GB10)

| フェーズ | 0.4.0 中央値 (比) | 0.6.0 中央値 (比) | 変化率 | 比の変化 |
|---|---|---|---|---|
| tape_build | 2.9 µs (0.0%) | 3.0 µs (0.1%) | +4.5% | +0.0pt |
| leaf_register | 0.1 µs (0.0%) | 0.1 µs (0.0%) | -14.3% | +0.0pt |
| forward_resident | 263.5 µs (2.2%) | 154.9 µs (2.8%) | -41.2% | +0.6pt |
| loss_readout | 0.0 µs (0.0%) | 0.0 µs (0.0%) | +0.0% | +0.0pt |
| backward | 11.895 ms (97.3%) | 5.398 ms (96.4%) | -54.6% | -0.9pt |
| device_update | 57.8 µs (0.5%) | 40.6 µs (0.7%) | -29.7% | +0.3pt |
| tape_drop | 3.0 µs (0.0%) | 0.9 µs (0.0%) | -68.0% | -0.0pt |
| **step_total** | 12.224 ms (100.0%) | 5.598 ms (100.0%) | -54.2% | +0.0pt |

## 13. infer について

イシュー #1145 のタイトルは「学習・推論 1 step」だが、**推論のフェーズ
分解は本イシューでは未実施**である。`scripts/bench/framework-compare/
bench-fandhe/src/main.rs` の `phases_with_gemm_fresh_or_infer_is_measure_error`
テスト（`dispatch()` を通した固定回帰。イシュー #1182 で `gemm --mode
reuse` が `--phases` 対象へ追加された際に「`infer`・`gemm --mode fresh`
は引き続き拒否される」ことを固定する目的で維持されている）が示すとおり、
`--phases` フラグは `--task train`（fresh/reuse）と `--task gemm --mode
reuse` に限定されており、`--task infer --phases` は
`MEASURE_ERROR`（fail-fast）となる。ハーネス側にこの制約を外す変更を
加えることは本イシューのスコープ外（下記参照）。

推論の学習経路との対比・要因分析が必要な場合は、以下の既存データを
参照する:

- `results-m4max-0.6.0.jsonl`・`results-dgx-0.6.0.jsonl` の
  `task:"infer"` 行（`scripts/bench/framework-compare/results/raw/`。
  PR #1127 で v0.6.0 について 5 回計測済み）
- PR #1127 の candle 比判定表（親 #1118 で「train/infer とも全バックエンド
  で candle 比未達」と確定した根拠）

推論の未達要因分析・`--phases` の infer 対応拡張は #1118 配下の後続
Issue に引き継ぐ（out-of-scope-tracking.md に従い、本 PR では新規 Issue
起票・既存 Issue への付け替えは行わず、PR 本文に切り出し先として記載
する）。

## 14. 検証結果サマリ（v0.6.0 再計測）

- `cargo tree -p bench-fandhe --depth 1 --locked`: 両環境とも
  `fandhe-ai v0.6.0`（registry 解決。`(path: …)` なし）。実行前後で
  `Cargo.lock` に差分なし
- 生データ件数: `results-m4max-phases-0.6.0.jsonl`（36 行）・
  `results-m4max-phases-0.6.0-extra.jsonl`（144 行）・
  `results-dgx-phases-0.6.0.jsonl`（36 行）・
  `results-dgx-phases-0.6.0-extra.jsonl`（144 行）。計 360 行
- `results/raw/skipped-m4max-phases-0.6.0.log`・
  `results/raw/skipped-dgx-phases-0.6.0.log`: いずれも 0 バイト
  （全実行成功。捏造していないことの証跡）
- `python3 summarize.py <file> --strict`: 主表 2 ファイル
  （`results-m4max-phases-0.6.0.jsonl`・`results-dgx-phases-0.6.0.jsonl`）
  は exit 0。`-extra.jsonl` 2 ファイルは exit 2（§11 に記載のとおり
  0.4.0 実測と同一の既知挙動で本計測固有ではない）
- `python3 summarize.py --strict`（既存 `results/raw/*.jsonl` 全件。
  回帰確認）: 新規 4 ファイル分の集計セクションが追加された以外は
  変更前と完全一致（既存の exit 2 は Burn(wgpu) Metal GEMM の既知不一致
  `docs/perf/burn-wgpu-metal-gemm-zero-result.md` によるもので本 PR と
  無関係。新規ファイル自体も同じ理由〈gemm 行なし〉で strict 全体判定を
  変えない）
- `python3 summarize_test.py`: 全 pass（Python 側無変更の回帰確認）
- 決定性: 新規 360 行すべての `checksum` が `0.080541` で一致
  （PR #1127 の `task:"train"` 行・§3 の 0.4.0 phases 行と同一値）
- クロスチェック: 各 (device, mode) の `step_total` 5 ラン中央値が
  PR #1127（`results-*-0.6.0.jsonl` の `task:"train"` 行。M4 Max
  cpu 17.30/8.07 ms・metal 19.14/9.11 ms、DGX cuda 11.69/5.50 ms・
  cpu 13.43/8.00 ms〈fresh/reuse〉）と同オーダー（本計測: cpu
  17.137/7.926 ms・metal 18.985/8.979 ms・cuda 11.789/5.598 ms・
  DGX cpu 13.631/8.093 ms）であることを確認。DGX cpu reuse は
  8.093 ms と #1127 の 8.00 ms よりやや高いが範囲 [7.543, 8.374] 内で
  あり、DGX 側は §10.2/10.4 のとおり常駐プロセス以外のプロセスなし
  （負荷混入なし）を確認済みのため、許容範囲の統計的ばらつきと判断する
- `cargo fmt --all --check`（`scripts/bench/framework-compare/`
  workspace）: pass
- `cargo clippy -p bench-common -p bench-fandhe --all-targets --locked
  -- -D warnings`（candle/burn は対象外。ビルド時間短縮のため
  `--workspace` は使わない）: pass
- `cargo test -p bench-common -p bench-fandhe --locked`: 12 passed
  （実機依存 4 件は `#[ignore]`）。Rust 側は無変更のため回帰確認のみ
- 内部情報漏えい: 新規・変更ファイルを `docs/real-hardware-
  verification-env.local.md` の実値（SSH ホスト名・IP・ユーザー名・
  常駐サービスパス等）で grep し 0 件を確認

## 15. 支配項トップ 3 の確定と改善 issue 起票案（イシュー #1151）

### 15.1 目的・対応

親 #1118（学習・推論の candle 比未達の解消）配下、依存 #1145（本
ドキュメント §10〜§14。v0.6.0 ピンでの学習 1 step フェーズ分解実測）が
完了したことを受け、本節は §11 の実測値から支配項トップ 3 を確定し
（§15.2）、backward 内部のコード事実に基づく推定（§15.3。実測ではない
ことを明記）・infer 未達の仮説（§15.5）・改善 issue 起票案（§15.6）を
記録する。§8.1 のとおり #1008 配下は解消済みであり、本節が v0.6.0
実測を踏まえた後続 Issue 優先順位の更新版（§15.4）を提供する。

### 15.2 train 支配項トップ 3（確定表・v0.6.0・§11 出典）

判定基準は §4 と同一（(i) 5 ラン中の比率のばらつきが小さいか、
(ii) fresh/reuse 間で同じ結論になるか、(iii) 区間の定義上その区間に
閉じた API 呼び出しか。3 条件で「高」・1〜2 条件で「中」・前提が残れば
「低」）。§12 で述べたとおり DGX CPU reuse の backward は 0.4.0 → 0.6.0
で (ii) の一致の強さが弱まったため「高」から「中」へ格下げしている。

| 環境 / mode | 1 位 | 2 位 | 3 位 | 確度（1 位） |
|---|---|---|---|---|
| CPU / M4 Max fresh | backward 96.5% | forward 3.3% | host_sgd 0.2% | 高 |
| CPU / M4 Max reuse | backward 91.7% | forward_resident 6.3% | device_update 1.7% | 高 |
| Metal / M4 Max fresh | backward 88.9% | forward 10.5% | tape_build/host_sgd 各 0.2% | 高 |
| Metal / M4 Max reuse | backward 84.1% | forward_resident 15.3% | device_update 0.9% | 高 |
| CPU / DGX GB10 fresh | backward 85.1% | forward 11.7% | host_sgd 1.9% | 中 |
| CPU / DGX GB10 reuse | backward 75.1% | forward_resident 20.3% | device_update 3.2% | **中**（§12 で高→中に格下げ） |
| CUDA / DGX GB10 fresh | backward 97.5% | forward 1.5% | host_sgd 0.5% | 高 |
| CUDA / DGX GB10 reuse | backward 96.4% | forward_resident 2.8% | device_update 0.7% | 高 |

バックエンド別差異（実測事実）:

- **CUDA は一極集中**（1.5〜2.8% まで 2 位以下が縮小）。非同期実行
  モデル下で forward・device_update の同期待ちが `loss_readout` へ
  計上される §4 の制約注記のとおり、2 位以下の絶対比較には限界がある
- **Metal・DGX CPU は forward 比が相対的に高い**（reuse で 15.3%・
  20.3%）。§12 のとおり DGX CPU reuse は 0.4.0 比 forward_resident が
  +9.9pt・backward が -13.9pt で、支配項の一極集中度が他系列より弱い
- **fresh:reuse の backward 比はおよそ 2:1**（§11 各表参照。例: CPU
  M4 Max 16.540 ms : 7.266 ms ≈ 2.28、CUDA DGX 11.500 ms : 5.398 ms ≈
  2.13）。8 系列全てで同傾向

### 15.3 backward 内部の推定（コード事実 + FLOP 算術。実測ではない）

`crates/autodiff/src/grad.rs::vjp` の VJP 経路（コード事実。origin/main
時点。`git diff v0.6.0..origin/main -- crates/autodiff crates/facade` は
空のため v0.6.0 本番経路にそのまま当たる）:

- `Op::MatMul`・`Op::LinearAct`（fresh 経路。#1044 の epilogue 融合
  ノード）は `matmul_vjp(a, b, g)`（`crates/autodiff/src/grad.rs:410`）
  が `d_input = eval::matmul(g, bᵀ)`・`d_weight = eval::matmul(aᵀ, g)`
  の**両方をホスト参照実装で計算**する（呼び出しは同 327 行）
- `Op::LinearResident`（reuse 経路）は d_input のみ
  `ops.gemm_resident_lhs(w_dev, gᵀ)`（デバイス GEMM。同 263〜266 行）、
  d_weight は `eval::matmul(xᵀ, g)`（同 276 行。ホスト参照実装）
- `crates/autodiff/src/eval.rs::matmul`（218〜242 行）は **scalar 三重
  ループ（`f32::mul_add`・`rayon` なし・`BackendOps` 非経由）** の
  ホスト参照実装。イシュー #1046 で転置 view の repack は除去済みだが
  演算自体は scalar のまま

推定（実測ではなく FLOP 算術に基づく仮説）: 本計測の層 1
（784→256・batch 64）の d_weight（xᵀ·g: [784,64]×[64,256]）は
784×64×256×2 ≈ 25.7 MFLOP、fresh の d_input（g·W1ᵀ: [64,256]×[256,784]）
も同オーダー。reuse backward ≈ scalar GEMM 呼び出し 1 個相当
（d_weight のみホスト scalar）、fresh ≈ 2 個相当（d_input・d_weight とも
ホスト scalar）と仮定すると:

- 3 バックエンドでほぼ同値（§15.2 観察）: backward の支配的コストが
  `eval::matmul`（バックエンド非依存のホスト参照実装）にあるなら、
  CPU/CUDA/Metal で数値が近いことと整合する
- fresh:reuse ≒ 2:1（§15.2 観察）: 呼び出し回数が 2 個 vs 1 個という
  仮定と整合する
- 実測値（reuse 5.4〜7.5 ms・fresh 11.5〜16.9 ms）を層 1 のみの
  25.7 MFLOP に当てはめると scalar 換算で概算 3〜5 GFLOP/s 相当となり
  （層 2〜3 分の追加 FLOP・その他区間内オーバーヘッドを含まない粗い
  概算のため幅を持たせる）、rayon 未使用の scalar 三重ループとして
  桁レベルでは矛盾しない

この推定は §15.2 の 2 つの実測パターン（バックエンド非依存・
fresh:reuse ≒ 2:1）を同時に説明する仮説として提示するものであり、
#1145 のハーネス（`--phases`）は backward 区間の内部をさらに分解
できないため実測による裏付けはない。参考計測（`eval::matmul` を層 1
VJP 形状で単体計時するミクロベンチマーク）は本 PR の作業環境では
実施していない（未計測。実施は起票案 A の実装時に譲る）。

### 15.4 後続 Issue 優先順位の更新案（v0.6.0 反映後）

§8 の 3 項目（0.4.0 実測時点）は #1008 クローズ（#1011/#1015/#1025
完了）により対象自体が解消済みである。v0.6.0 実測（§11・§12）を踏まえた
更新版:

1. **最優先**: backward 内部の VJP GEMM をホスト scalar 参照実装
   （`eval::matmul`）からバックエンド別 GEMM（`BackendOps::gemm` 系）へ
   切り替える（§15.3 の推定が正しければ 3 バックエンド共通の支配項へ
   直接効く）。起票案 A〜E
2. reuse 経路の勾配のデバイス常駐化（D2H→H2D 往復排除）は 1 の効果測定
   後に評価する（起票案 B）
3. `device_update`／`host_sgd` は全系列で 3% 未満（§11）であり、
   #1015/#1017（Metal コマンドバッファ統合）が既に縮小させている
   （§12: Metal reuse `device_update` 6.7%→0.9%）ことも踏まえ、単独の
   追加最適化としての優先度は低い

### 15.5 infer 未達の仮説（実測不能事項を明記）

§13 のとおり `--task infer --phases` は `MEASURE_ERROR`（ハーネス制約）
のため、infer の内訳は本 issue でも実測できない。以下は既存計測値と
コード事実に基づく仮説であり、内訳比率は実測していない。

**コード事実**（`scripts/bench/framework-compare/bench-fandhe/src/
main.rs::run_infer`。origin/main 時点）:

- CPU 経路は `Sequential::predict`（facade の tape 不要経路。層ごとに
  `forward_host` を直接呼ぶ非融合合成〈`gemm` → `add`〉であり、
  呼び出し毎に `CpuBackendOps::new()` を構築する）
- CUDA/Metal 経路は `make_tape` 後 `tape.var(&x_data)` +
  `model.forward(&tape, &x)`。`Linear::bind`（facade）が `Tape::var` で
  weight/bias を**毎回 clone**するため毎回 H2D が発生し、演算ごとの
  戻り値も D2H される
- `BackendOps::linear_forward_device`（活性化のデバイス常駐チェーン。
  `docs/inference-forward-fixed-cost-design.md` §3.2 段階 B）は **CPU
  のみ実装**（`crates/backend-cpu/src/ops.rs:493`）。CUDA/Metal は
  同 doc §4 のスコープ外項目のまま未実装で、`Sequential::predict_
  resident`（facade）は公開済みだが bench-fandhe の infer 計測では
  使われていない
- 対比として `bench-candle::run_infer` は weight・入力ともデバイス
  常駐で計測外に構築し、計測内は forward + 1 回の readout のみ

**既存計測値**（`results-dgx-0.6.0.jsonl`／`results-m4max-0.6.0.jsonl`。
PR #1127 実測）: DGX CPU 307.0 µs vs candle 248.0 µs（0.81 倍）・CUDA
156.9 vs 42.4 µs（0.27 倍）、M4 Max CPU 567.5 vs 202.8 µs（0.36 倍）・
Metal 805.9 vs 409.8 µs（0.51 倍）

**仮説**: GPU（CUDA/Metal）は毎回の weight clone + H2D + 演算ごとの
D2H が固定費として重く、値が小さい形状（本計測 size=64）ほど相対的な
未達率が大きい（CUDA 0.27 倍が最も未達）ことと整合しうる。CPU は
非融合 3 起動 + 呼び出し毎の `CpuBackendOps::new()` が固定費として
乗るが、train の backward（infer には存在しない区間）が支配項である
という §15.2〜§15.3 の知見は infer には直接適用できない（train と
infer は異なるコード経路であり、backward 最適化が infer の未達を
解消する保証はない）。

**実測不能事項（明記）**: GPU 経路の H2D/D2H が infer 総時間に占める
実際の比率、CPU 経路の非融合合成分と `CpuBackendOps::new()` 構築分の
内訳、CPU 1.24〜2.8 倍差（0.81/0.36 倍の逆数）の具体的な内訳。いずれも
`--phases` の infer 対応拡張（起票案 G）なしには実測できない。

### 15.6 改善 issue 起票案

各 issue 本文に共通で転記する条件・注意（起票時に本文へ含めた）:

- 本番結線は事前承認済み・性能低下の可能性がある変更は結線前後で
  同一プロトコル 5 回計測中央値の比較を PR 本文と `docs/perf` に記録する
- tolerance / baseline / 依存の変更はユーザー承認必須
- 内部ホスト名等を書かない
- (i) `eval::matmul`（scalar・固定 k 順）から BLIS/GPU GEMM へ
  切り替えると累積順序が変わり勾配が bit 一致しなくなりうる。既存の
  `assert_eq!` 系勾配テストを複合判定へ変える必要が生じた場合は
  「許容誤差の新設」としてユーザー承認対象（勝手に緩めない）
- (ii) framework-compare は `fandhe-ai =0.6.0` registry 固定
  （fail-closed）のため HEAD の before/after は同ハーネスで測れない。
  #1142/#1147 の「参考系列」手順（非コミットの一時 path 差し替え）に
  従うか、リポ内 bench（`crates/facade/tests/*bench*` 等）で計測する

| # | タイトル | スコープ | 依存 | 支配項 |
|---|---|---|---|---|
| A | `matmul_vjp`／`Op::LinearResident` d_weight の `eval::matmul` 呼び出しを `BackendOps::gemm` 経由へ切り替える | grad.rs の 3 箇所（LinearResident d_weight・matmul_vjp の d_input/d_weight）を `ops.gemm` へ結線。転置は当面 `contiguous()` 再パックを許容 | なし | backward（fresh・reuse 双方） |
| B | reuse 経路の grad をホストへ落とさずデバイス常駐のまま `device_update` へ渡す | `Op::LinearResident` の d_weight を GPU 計算後 D2H せず grad upload へ直結。A の実測で寄与が小さければ見送り可 | A | backward・device_update（reuse） |
| C | CPU BLIS GEMM に VJP 専用の NT/TN 2 パターン入口を追加する | `matmul-vjp-zero-copy-decision.md` §3.2 項目 1 を NT（d_input）／TN（d_weight）の 2 パターンに限定 | A | backward |
| D | CUDA GEMM に NT/TN 限定の転置入口を追加し VJP 経路へ結線する | 同 §3.2 項目 2 を NT/TN 限定。GB10 実機実測必須 | A | backward（CUDA） |
| E | Metal GEMM の NT/TN strided 結線を VJP 経路へ適用する | 同 §3.2 項目 3・4 を NT/TN 限定。#1187（転置タイル variant 自動ルーティング）を依存先として参照 | A・#1187 | backward（Metal） |
| F | `linear_forward_device`（活性化デバイス常駐チェーン）を CUDA/Metal に実装する | `inference-forward-fixed-cost-design.md` §4 の未実装項目。`#[ignore]` parity + 実機実測 | なし | forward・infer |
| G | bench-fandhe の infer に `predict_resident` 使用の reuse モードと `--phases` 対応を追加する | ハーネス拡張。採否・summary.md への反映方針は当該 issue で判断 | F | infer |
| H | CPU 推論 `predict` 経路をプロファイルし融合 `gemm_bias_act` 適用可否を判断する | ホットスポット採取。bit-exactness 契約の見直しはユーザー判断事項として提示 | なし | infer（CPU） |
| I | 非学習葉（入力 x）への d_input 伝播スキップの設計判断 | fresh の d_input（≈25.7 MFLOP）は入力 x が非学習葉のため学習には不要。`Gradients::get` の葉勾配契約との両立を検討する設計判断 issue（実装は別途） | なし | backward |

起票結果（#1135 配下・`phase:5` ラベル）:

| # | issue 番号 |
|---|---|
| A | #1211 |
| B | #1212 |
| C | #1213 |
| D | #1214 |
| E | #1215 |
| F | #1216 |
| G | #1217 |
| H | #1218 |
| I | #1219 |

### 15.7 spec 整合の確認

- REQ-8（`docs/spec/04-requirements.md`）は GEMM 5 行の対 PyTorch 下限の
  みを定め、train/infer 自体には下限を置いていない。本節の candle 比
  ゲートは #1051 のハーネス基準であり spec の受け入れ基準ではないため、
  本節の提案は REQ-8 の受け入れ基準と矛盾しない
- 起票案 A〜E は FMA 契約（CPU 参照実装 `f32::mul_add`・GPU 既定 FMA
  契約）・REQ-2 複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5
  未満）を変更しない前提で書かれている（上記 (i) の注意）。tolerance
  定数自体の変更が必要になった場合はユーザー承認必須と明記済み
- カーネル側の手動境界チェック省略はいずれの起票案にも含まれない
- 依存追加は起票案に含まれない（既存の許容依存区分・自作コア方針の
  範囲内での実装を前提とする）
- 以上より本節は spec 変更提案を必要としないと結論する

### 15.8 検証結果サマリ

- §10.3 の集計スニペットを `results/raw/results-{m4max,dgx}-phases-
  0.6.0*.jsonl` に対し再実行し、§15.2 の中央値・比率が §11 の値と
  一致することを確認した（転記元の再計算による裏付け。exit 0）
- `git diff origin/main -- docs/perf/train-step-phase-breakdown.md` で
  §8 本文（既存文言）に削除行がないこと、§9〜§14 の見出し番号が不変
  であることを確認した
- `grep -n "^## \|^### " docs/perf/train-step-phase-breakdown.md` で
  §8.1・§15.1〜§15.8 の見出しが想定順序どおりであることを確認した
- 新規・変更ファイルを内部ホスト名等で grep し 0 件を確認した
  （`docs/real-hardware-verification-env.local.md` は本作業環境に
  存在しないため、`.example` のプレースホルダ名で代替確認した）
- Rust コード・`results/summary.md`・`docs/spec/`・依存・tolerance
  定数はいずれも無変更（本節は docs と GitHub issue 起票のみ）
