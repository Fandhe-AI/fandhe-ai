# backward 非 GEMM 内訳の集計（イシュー #2100・machine=smoke-x86）

判定種別: **record_only**（診断のみ。ADOPT／REJECT は判定しない）。

集計方法: 各 run で step 20..99 の中央値 → その 3 run の中央値。単位は µs（`total`／`gemm` 等の壁時計時間）。

注記: `gemm` は resident grad staging 書き込み（`fill`。bias 縮約を含みうる）を含む（旧診断 §4 と比較可能）。`fill` は `gemm` の内訳で二重計上しない。`非 GEMM 下界` は fill 内の bias 縮約を GEMM 側へ含むため過小評価になりうる。bias 縮約は GEMM 計算と分離計時していないため、fill 全体を非 GEMM 側へ戻した `非 GEMM 上界`（total − gemm + fill）を併記する（真値は下界〜上界の間）。`残差` = total − Σ(gemm+mask+ewise+transpose+materialize+accumulate+loss)（走査ループ・grads clone・checkpoint 再解放・未計装 arm 等）。GPU デバイスは非同期処理が同期点のカテゴリへ計上される。

## device=cpu mode=fresh

- 負荷ゲート: pass
- 計装オーバーヘッド比（記録のみ）: step_total 計装あり÷なし の run 中央値 = 1.116

| カテゴリ | µs | total 比 |
|---|---:|---:|
| total | 550.00 | 100.0% |
| gemm | 486.57 | 88.5% |
| mask | 11.91 | 2.2% |
| ewise | 8.50 | 1.5% |
| transpose | 0.00 | 0.0% |
| materialize | 0.63 | 0.1% |
| accumulate | 0.34 | 0.1% |
| loss | 35.60 | 6.5% |
| (fill ⊂ gemm) | 0.00 | 0.0% |
| vjp（参考） | 545.30 | 99.1% |
| **非 GEMM 下界 = total − gemm** | 65.24 | 11.9% |
| **非 GEMM 上界 = total − gemm + fill** | 65.24 | 11.9% |
| 残差 | 6.44 | 1.2% |

呼び出し回数（run 中央値）: vjp=10 gemm=2 mask=1

## device=cpu mode=reuse

- 負荷ゲート: pass
- 計装オーバーヘッド比（記録のみ）: step_total 計装あり÷なし の run 中央値 = 0.839

| カテゴリ | µs | total 比 |
|---|---:|---:|
| total | 547.48 | 100.0% |
| gemm | 458.27 | 83.7% |
| mask | 35.17 | 6.4% |
| ewise | 8.28 | 1.5% |
| transpose | 0.81 | 0.1% |
| materialize | 0.40 | 0.1% |
| accumulate | 0.23 | 0.0% |
| loss | 24.86 | 4.5% |
| (fill ⊂ gemm) | 214.24 | 39.1% |
| vjp（参考） | 544.21 | 99.4% |
| **非 GEMM 下界 = total − gemm** | 77.60 | 14.2% |
| **非 GEMM 上界 = total − gemm + fill** | 299.08 | 54.6% |
| 残差 | 19.46 | 3.6% |

呼び出し回数（run 中央値）: vjp=7 gemm=4 mask=1

## 収録範囲（再検証の可否）

本ディレクトリには集計結果（`iterations.csv`・本ファイル）・`gate.tsv`・`env_info.txt`・`RULE.txt` のみを収録し、集計器が読む元ログ（`run*-instr.err`・計装あり／なしの JSONL）は**収録していない**（スモークは配線確認用の使い捨て実行で、ホスト名・絶対パスのマスク済み元ログを保存していない）。したがって本値と、計装あり／なし JSONL の checksum 一致は集計時に検査済みだが、本ディレクトリだけからは再検証できない。`iterations.csv` は step ごとの DIAG 値を保持するので、中央値・非 GEMM 上下界の再計算は可能。実機（m4max・gb10）の収録ではマスク済み元ログも収録する。
