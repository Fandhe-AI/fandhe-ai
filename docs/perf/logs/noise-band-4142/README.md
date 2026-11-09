# 僅差セルのノイズ帯の試算（#4142）

`docs/perf/noise-band-decision.md`（決定記録。状態は所有者承認待ち）の試算表を再現する部品。計測ではなく、0.14.0 の既存 5 run の JSONL を読むだけで、追加の実機計測はない。

## 中身

| ファイル | 役割 |
|----------|------|
| `compute_noise_band.py` | NB-RANGE5（run 別対戦相手比の最小〜最大）と NB-IQR（iter 内 q1/q3 の帯）を TSV で出す。Python 標準ライブラリのみ |
| `noise_band_0140_macmini.tsv` | Mac mini（Apple M4）13 セルの出力 |
| `noise_band_0140_gb10.tsv` | GB10 14 セルの出力 |

TSV の列: `cell`（task device size）／`runs`（run1〜run5 の比）／`median`／`min`／`max`／`cross`（帯が含む区分境界。なければ `-`）／`iqr_lo`・`iqr_hi`（NB-IQR。run 間の中央値）。

## 入力の出所

- Mac mini: `../framework-compare-0.14.0-macmini-remeasure/macmini-series-b/run{1..5}/results-macmini-0.14.0.jsonl` と `results-macmini-py-0.14.0.jsonl`
- GB10: `../framework-compare-0.14.0-remeasure/gb10/run{1..5}/results-dgx-0.14.0.jsonl`・`results-dgx-0.14.0-extra.jsonl`・`results-dgx-py-0.14.0.jsonl`

機体のホスト名は公開テキストに書かない（Mac mini は `<metal-node>`）。

## 実行

```sh
cd docs/perf/logs/noise-band-4142
python3 -I compute_noise_band.py --self-test
python3 -I compute_noise_band.py --check-verbatim ../framework-compare-0.14.0-remeasure/scoreboard/gen_0140.py
python3 -I compute_noise_band.py --profile macmini --runs-dir ../framework-compare-0.14.0-macmini-remeasure/macmini-series-b > noise_band_0140_macmini.tsv
python3 -I compute_noise_band.py --profile gb10 --runs-dir ../framework-compare-0.14.0-remeasure/gb10 > noise_band_0140_gb10.tsv
```

出力は決定的で、再生成すると同梱の TSV と byte 一致する。

## 判定側との一致

- run 別比の式は `gen_0140.py::paired_ratio` と同じ（`load`・`key`・`index` は逐語コピーで、`--check-verbatim` が確かめる）。
- 小数 2 桁に丸めると、Mac mini の run 別比は `../framework-compare-0.14.0-macmini-remeasure/aggregate/gen_0140_macmini.out` の `run別` の値、GB10 は `../framework-compare-0.14.0-remeasure/scoreboard/gen_0140.out` の値と一致する（13＋14 セル）。
- fail-closed: 判定側（fandhe-ai reuse）の `parity_fail_count > 0`・run 欠損・有効な相手の不在・版違いは停止する。相手側の parity 不合格は最速に含めない。

## 限界

N=5・単一機体・単一日の揺れだけを見る。詳細は決定記録の §5。
