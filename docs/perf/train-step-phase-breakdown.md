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
