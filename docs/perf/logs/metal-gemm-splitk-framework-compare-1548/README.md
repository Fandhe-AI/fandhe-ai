# イシュー #1548: split-K 既定 ON の事後監視（framework-compare gemm／train metal・同一バイナリ runtime 切替 A/B）

本ディレクトリはイシュー #1548（#1544 の既定 ON 切替に対する低負荷時の
事後監視）の M4 Max 実機実測成果物置き場。#1517（2 worktree の定数
フリップ方式）と異なり、#1546 の実行時トグル（`bench-fandhe
--metal-split-k on|off`。feature `metal-split-k-toggle`）を使い**単一
バイナリ**で before=`off`／after=`on` を run 単位 interleave 計測する
（`scripts/bench/framework-compare/run_ab_splitk_metal.sh`）。

## 事前登録判定規則（計測開始前に固定。計測後の追加・緩和は行わない）

- セル: gemm 8 セル（N=512/1024/2048/4096 × fresh/reuse）＋ train 2 セル
  （fresh/reuse）。各腕 5 round・中央値。
- 規則 1（非後退）: セルごとに `after(on)/before(off)` 中央値比
  `ratio <= 1.00`（#1517 §3 と同一）。
- 規則 2（checksum）: 全セルで両腕の checksum が完全一致すること。
  ただし split-K 到達セル（train fresh の L2 forward。#1517 §2 帰属表）は
  経路が異なるため bit 一致しないことが既知であり、checksum の**不一致
  自体は数値不良ではない**（受け入れは #1511 baseline 方式で別途担保）。
  不一致セルは帰属表と突合し「到達セルのみで不一致」であることを確認する。
- 負荷: 専有ゲートなし（record_only）。`uptime`／`pmset -g therm` を記録。
- 結果は**記録のみ**。#1544 のユーザー判断（既定 ON）を変更する入力に
  しない（`docs/backend-metal-splitk-decision.md` §5 の判定記録も不変）。
- 計測中はビルド（cargo）を並走させない（メモリ
  `mac-session-bash32-pitfall`）。

## 実行手順

```sh
cd scripts/bench/framework-compare
AB_PATCH_FACADE_PATH="$(cd ../../../crates/facade && pwd)" \
  bash run_ab_splitk_metal.sh splitk-monitor-1548
python3 compare_gemm_ab.py --task gemm --threshold 1.00 --per-run \
  results/raw/results-m4max-splitk-ab-off-splitk-monitor-1548-gemm.jsonl \
  results/raw/results-m4max-splitk-ab-on-splitk-monitor-1548-gemm.jsonl
python3 compare_gemm_ab.py --task train --threshold 1.00 --per-run --phases \
  results/raw/results-m4max-splitk-ab-off-splitk-monitor-1548-phases.jsonl \
  results/raw/results-m4max-splitk-ab-on-splitk-monitor-1548-phases.jsonl \
  results/raw/results-m4max-splitk-ab-off-splitk-monitor-1548-train.jsonl \
  results/raw/results-m4max-splitk-ab-on-splitk-monitor-1548-train.jsonl
```

結果は `docs/perf/metal-gemm-splitk-framework-compare-1517.md` の追記節へ
転記する。内部ホスト名は含めない。
