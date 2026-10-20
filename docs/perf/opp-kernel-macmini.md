# Mac mini で相手の純カーネル時間と Python 側 checksum を測る（イシュー #3179）

## 状態

診断基盤と事前登録規則まで完了。**実測は未実施（unavailable）**。この作業環境から Mac mini へ到達できなかったため、
数値は一切埋めていない（推定値で埋めない。`logs/opp-kernel-macmini-3179/RULE.txt` 項 13）。

## 目的

0.11.0 の Mac mini 系列 B では、PyTorch の計測窓に matmul のディスパッチ・`.cpu().numpy()`・`checksum()`
（`np.cumsum(astype(float64))`）が全部入っている。相手の純カーネルがハード上限にどれだけ近いかを実測で言うために、
窓を区間に分けて測る。診断のみで、`bench_py.py`・相手側ベンチ・判定規則は変えない。

## 方法

正は `logs/opp-kernel-macmini-3179/RULE.txt`。ツールは `scripts/bench/framework-compare/opp_kernel_torch_diag.py`、
実行は `logs/opp-kernel-macmini-3179/run-macmini.sh`。

## 結果（すべて unavailable）

| 項目 | 値 | 出典 |
|---|---|---|
| CPU／MPS の `torch.mm` 純時間（正方 5 サイズ・MLP 6 形状） | unavailable | 未実測 |
| Python `checksum()` の ns／要素（Mac mini） | unavailable（参考: 推定 約 3.5 ns／要素・実測ではない） | 未実測 |
| Rust 逐次総和の ns／要素（Mac mini） | unavailable。#3182 の値と突き合わせる（#3182 待ち）。約 0.56 は M4 Max の値で Mac mini の値ではない | 未実測 |
| 窓の再現（window ÷ 系列 B） | unavailable | 未実測 |

## 約 480 GFLOPS（PyTorch CPU N=2048）の切り分け

未実施。総和で説明できる分・スレッド腕で変わる分・numpy 経由の Accelerate との差を並べる設計だが、
**分けられたかどうかは未確定**。第三者公開値は別機体のため、ビルド差はこの機体だけでは決着しない可能性が高い。

## 残課題

- Mac mini での実測（`run-macmini.sh`）と本 doc の結果表の記入
- N=256／512 のライブラリ同士の比は fandhe 側の純カーネル（#3180／#3182）待ちで空欄。#3191 へ申し送る
