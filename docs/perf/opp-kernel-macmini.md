# Mac mini で相手の純カーネル時間と Python 側 checksum を測る（イシュー #3179）

## 状態

**実測済み**（2026-10-21 UTC・Mac mini〈Apple M4・4P+6E・16 GB〉・macOS 27.0.1）。診断のみ（record_only）で、判定・閾値は設けない。
出典はすべて `logs/opp-kernel-macmini-3179/macmini/` の `aggregate.md`（Python 側。以下「A」）と `aggregate_rust.md`（Rust 側。以下「R」）。

## 目的

0.11.0 の Mac mini 系列 B では、PyTorch の計測窓に matmul のディスパッチ・`.cpu().numpy()`・`checksum()`
（`np.cumsum(astype(float64))`）が全部入っている。相手の純カーネルがハード上限にどれだけ近いかを実測で言うために、
窓を区間に分けて測る。`bench_py.py`・相手側ベンチ・判定規則は変えていない。

## 方法と計測条件

- 規則: `logs/opp-kernel-macmini-3179/RULE.txt`。ツール: `scripts/bench/framework-compare/opp_kernel_torch_diag.py`。実行: `logs/opp-kernel-macmini-3179/run-macmini.sh`
- 腕 8 つ（base CPU／base MPS／threads 1・4・10／VECLIB 1・4／numpy）× 5 run（run ごとに起動順をローテーション）＋ Rust 25 回（5 サイズ × 5 run）
- 計測ツリー: `macmini/rev-stamp.txt` = `4c9abce7ed8e2f05561ef4b42e047602f34dca0c`。bench-fandhe は fandhe-ai v0.11.0（registry 版。`macmini/rust/tree.txt`）
- 負荷ゲート: 65 プロセスすべて GATE-PASS（通過時 load1 は 1.54〜2.72。`macmini/gate.log`）。GATE-TIMEOUT／GATE-INVALID／unavailable は 0 件
- 版: Python 3.12.15・torch 2.14.0・numpy 2.5.4（`macmini/versions.txt`）。torch は `BLAS_INFO=accelerate`、numpy も Accelerate（`macmini/env.txt`）
- `bench_py.py` の sha256 は `9fd2c08f…2409b` で一致。系列 B との checksum 照合も通過（`aggregate ok: 202 entries`）
- **RULE 項 2 からの逸脱**: 0.11.0 計測の作業ディレクトリ（fill_vec の `.npy` キャッシュ）は Mac mini から消えていた。同じ版の venv（`thread-3178` 用）と、同じ `bench_py.py` を転送ツリーに置き、キャッシュは seed 決定的に作り直した。値の同一性は系列 B との checksum 照合で検査済み。0.11.0 当時の作業ディレクトリそのものではない

## 結果

### CPU／MPS の `torch.mm` 純時間（`mm_pure`。5 run 中央値。run 別は `aggregate.md`）

| N | CPU 時間 (ms) | CPU GFLOPS | MPS 時間 (ms) | MPS GFLOPS |
|---|---|---|---|---|
| 256 | 0.0240 | 1398 | 0.2462 | 136 |
| 512 | 0.1520 | 1766 | 0.3050 | 880 |
| 1024 | 1.1404 | 1883 | 0.9344 | 2298 |
| 2048 | 9.7186 | 1768 | 5.3075 | 3237 |
| 4096 | 73.513 | 1870 | 40.080 | 3429 |

- MPS の小サイズは 1 回ごとの同期の固定費が支配する（`mm_pure_batch10`＝10 回続けて同期して 10 で割る参考値は N=256 で 648 GFLOPS、N=512 で 2061 GFLOPS）。判定には使わない（RULE 項 5）
- MLP 6 形状の CPU `mm_pure`（GFLOPS）: L1 fwd 1427／L1 dW 1519／L1 dX 1041／L2 fwd 136／L2 dW 179／L2 dX 183。MPS はいずれも 1.6〜106 GFLOPS で固定費が支配（L2 系は 1.6〜1.8）。時間は `aggregate.md`
- CPU の `readout`（`.cpu().numpy()`）は 0.46 µs 前後で N に依らない（ビューを作るだけ）。MPS の `readout` は N=2048 で 0.79 ms・N=4096 で 2.31 ms

### Python `checksum()` の ns／要素（毎回新しく実体化した配列。`aggregate.md`）

| N | CPU | MPS | 推定（実測ではない） |
|---|---|---|---|
| 256 | 2.200 | 2.026 | 約 3.5（範囲 2.6〜4.2） |
| 512 | 2.206 | 2.147 | 同 |
| 1024 | 2.312 | 2.217 | 同 |
| 2048 | 2.388 | 2.370 | 同 |
| 4096 | 2.497 | 2.462 | 同 |

- 実測は 2.0〜2.5 ns／要素で、推定の約 3.5 より小さい（推定の範囲 2.6〜4.2 の下限も下回る）。小形状（MLP の L2 fwd）では 5.2〜6.9 に上がる（要素数が少なく固定費が見える）
- 窓の検算: CPU の Σ区間（mm_pure+readout+checksum）÷ window は 0.98〜1.02（正方 5 サイズ）。窓は区間の和で説明できる

### Rust 逐次総和の ns／要素（R。Mac mini・`--device cpu`・reuse --phases・5 run 中央値）と #3182 F2・Python の比

| N | Rust（R） | #3182 F2（Metal 計測・checksum はホスト側処理） | Python CPU（A） | Python ÷ Rust |
|---|---|---|---|---|
| 256 | 0.556 | 0.8335 | 2.200 | 3.96 |
| 512 | 0.603 | 0.6028 | 2.206 | 3.66 |
| 1024 | 0.605 | 0.5403 | 2.312 | 3.82 |
| 2048 | 0.604 | 0.5329 | 2.388 | 3.95 |
| 4096 | 0.599 | 0.5148 | 2.497 | 4.17 |

- Rust の値は Mac mini の実測で、M4 Max の約 0.56 とは別に出した。#3182 F2 とは N=512 で一致し、N=1024 以上で 0.06〜0.09 ns 高く、N=256 では低い（計測のデバイス・経路が違うため差は説明していない）
- 窓の中の総和は Python が Rust の約 3.7〜4.2 倍。推定（約 5 倍）より小さいが、Python 側が不利になる向きは確認できた

### 窓の再現（window ÷ 系列 B。CPU）

N=256／512／1024／2048 で 0.522／0.523／0.522／0.543。**今回の窓は 0.11.0 系列 B の約半分の時間**で、同じ `bench_py.py` の窓なのに再現していない（原因は未特定。MPS も 0.52〜0.65）。

## 約 480 GFLOPS（PyTorch CPU N=2048）の切り分け

480 GFLOPS は系列 B の窓の値（17.18 GFLOP ÷ 約 35.8 ms。今回の窓 19.42 ms は系列 B の 0.543 倍）。今回の結果を 3 軸で並べる。

1. **総和（readout＋checksum）**: N=2048 の窓 19.42 ms のうち checksum が 10.02 ms（約 52%）・readout は約 0。N=4096 では 41.9 ms／115.5 ms（約 36%）。窓の GFLOPS（885／1190）は純 `mm` の GFLOPS（1768／1870）の約半分〜3 分の 2 に下がる。総和は窓の GFLOPS を下げる最大の要因だが、これだけで 480 には届かない（総和を窓から除いても今回は 1768 GFLOPS）
2. **スレッド数（N=2048 CPU の `mm_pure` GFLOPS）**: 既定 1768／threads1 1678／threads4 1810／threads10 1872／VECLIB 1 は 1575／VECLIB 4 は 1878。N=512 は全腕で 1765〜1766 でほぼ同じ。2048 の run 間ばらつきは大きい（既定の run 別 9.1〜10.8 ms）。値だけを記録する
3. **numpy 経由の Accelerate**（`np.matmul`）: N=2048 で 1888 GFLOPS、N=512 で 1765 GFLOPS。PyTorch 既定（1768／1766）と同水準で、ビルド差としては出なかった

**結論: 切り分けられなかった。** 純カーネルは 1.77 TFLOPS 前後で、第三者公開値（M4 系で約 1.5 TFLOPS）と同水準以上、numpy の Accelerate とも差がない。窓の GFLOPS は総和の分だけ下がるが、今回の窓は 885 GFLOPS で系列 B の 480 GFLOPS を再現していない。系列 B との約 2 倍の差は、この診断が測る 3 軸（総和・スレッド・ビルド）では説明できない。差の原因（系列 B 当時の機体状態・作業ディレクトリ・実行条件の違い等）は未検証で、推測は書かない。第三者公開値は別機体のため、ビルド差はこの機体だけでは決着しない。

## #3191 への申し送り

- 相手側（PyTorch）の窓の中身: checksum が窓の約 36〜52%（CPU・N=2048〜4096）。相手の純カーネルは CPU で約 1.77 TFLOPS・MPS で約 3.2〜3.4 TFLOPS（N=2048〜4096）
- 窓が系列 B の約 0.52〜0.54 倍で再現した点は「再現で届く値」表の列を決める前に所有者判断が要る（系列 B の値をそのまま使うか、今回の窓で更新するか）
- N=256／512 のライブラリ同士の比は fandhe 側の純カーネル（#3180／#3182）と合わせて #3191 で出す

## 残課題

- 系列 B の窓との約 2 倍の差の原因調査（系列 B と同じ作業ディレクトリ・同じ起動条件の再現が必要）
- Metal の candle の純カーネルは #3181、GB10 の cuBLAS は #3185
