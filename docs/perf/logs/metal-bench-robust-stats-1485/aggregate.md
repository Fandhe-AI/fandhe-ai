# イシュー #1485: `phase1_round_stats` 判定不変・キー順の機械検査結果

M4 Max 実機 1 run（`phase1_only_run1.log`。record_only モード・共有負荷下・
load average 1min≈3.84）から `phase1_round_stats` 行を抽出し、以下を検査
した（検査スクリプトは `docs/perf/logs/metal-bench-robust-stats-1485/README.md`
の「判定不変の確認方法」節参照。標準ライブラリのみの Python3 スクリプト）。

| size | spread | gate | within_gate | spread<=gate | trimmed_spread_k1 | iqr_spread | mad_spread |
|---|---|---|---|---|---|---|---|
| 256 | 2.1418e-01 | 5.0000e-02 | false | false | 1.9120e-1 | 1.3225e-2 | 1.9510e-2 |
| 512 | 5.9134e-01 | 5.0000e-02 | false | false | 4.7441e-1 | 5.2024e-2 | 7.1282e-2 |
| 1024 | 1.2784e-01 | 5.0000e-02 | false | false | 6.8816e-2 | 3.2653e-2 | 4.4504e-2 |
| 2048 | 3.6217e-01 | 5.0000e-02 | false | false | 1.9826e-1 | 1.4264e-1 | 4.3319e-2 |
| 4096 | 5.9607e-01 | 5.0000e-02 | false | false | 5.7022e-2 | 3.7271e-2 | 3.9314e-2 |

## 判定不変の確認結果

- 行数: 5（size=256/512/1024/2048/4096 の順。期待どおり）
- キー順: 全 5 行で `size rounds spread gate within_gate median_secs
  min_secs min_round_idx max_secs max_round_idx round_medians_secs
  trimmed_spread_k1 iqr_spread mad_spread` の順に一致（既存ログ
  `docs/perf/logs/metal-gemm-transpose-route-ab-1242/1267-attempt1-run.log`
  L12 等のキー順と同一）
- `within_gate` 一致検査: 全 5 行で `within_gate == (spread <= gate)`
  （上表の `within_gate` 列と `spread<=gate` 列が完全一致） → 補助統計
  （`trimmed_spread_k1`／`iqr_spread`／`mad_spread`）が判定に一切影響
  していないことを機械的に確認
- 共有負荷下（load average 1min 3.84〜7.91。上昇傾向）のため全 5 サイズで
  `within_gate=false`。ゲート成立自体は本イシューの目的ではない
  （`docs/perf/metal-bench-noise-protocol.md` §8.6・§8.8 参照）

## 実行結果全文（`ALL CHECKS PASS`）

```
size=256 spread=2.1418e-01 gate=5.0000e-02 within_gate=false (spread<=gate: False) trimmed_spread_k1=1.9120e-1 iqr_spread=1.3225e-2 mad_spread=1.9510e-2
size=512 spread=5.9134e-01 gate=5.0000e-02 within_gate=false (spread<=gate: False) trimmed_spread_k1=4.7441e-1 iqr_spread=5.2024e-2 mad_spread=7.1282e-2
size=1024 spread=1.2784e-01 gate=5.0000e-02 within_gate=false (spread<=gate: False) trimmed_spread_k1=6.8816e-2 iqr_spread=3.2653e-2 mad_spread=4.4504e-2
size=2048 spread=3.6217e-01 gate=5.0000e-02 within_gate=false (spread<=gate: False) trimmed_spread_k1=1.9826e-1 iqr_spread=1.4264e-1 mad_spread=4.3319e-2
size=4096 spread=5.9607e-01 gate=5.0000e-02 within_gate=false (spread<=gate: False) trimmed_spread_k1=5.7022e-2 iqr_spread=3.7271e-2 mad_spread=3.9314e-2
ALL CHECKS PASS: key order matches / within_gate == (spread <= gate) for all 5 rows
```
