# イシュー #1555 実測記録（Metal resident weight grad 実装）

## 目的

イシュー #1555「Metal 側 `gemm_fp32_strict_into` 実装・reuse 学習 weight 勾配のデバイス常駐 staging 直接書き込み」の実装確認・性能検証。

## 実行環境

- デバイス: Apple M4 Max
- 実施日: 2026-09-12
- 実行モード: record_only（判定ゲート opt-out）
- 共有負荷: load1 ≈ 7.9〜8.5（Mac セッション内の他作業）

## ディレクトリ構成

- `bitdump/`: 数値完全一致確認
  - `main_raw.log`, `branch_raw.log`: 各系列の実行ログ
  - `main_bits.txt`, `branch_bits.txt`: bit ダンプ出力（4462 行の loss・勾配・パラメータ値）
- `ab/`: framework-compare train A/B 計測
  - `results-before-1555-train.jsonl`, `results-after-1555-train.jsonl`: 計測結果（5 round）
  - `results-before-1555-phases.jsonl`, `results-after-1555-phases.jsonl`: フェーズ分解結果
  - `compare-train-1555.md`: 集計済み比較表
  - `run_ab_1555.sh`: 計測スクリプト（参考用・ローカル絶対パスをマスク）
  - `sha-before.txt`, `sha-after.txt`: 各腕のバイナリ sha256
  - `uptime.log`, `pmset_therm_before.txt`, `pmset_therm_after.txt`: 環境情報
  - `tree-before.txt`, `tree-after.txt`: `cargo tree` 出力（依存一致確認）
- `env_info.txt`: 実行環境・デバイス属性
- その他: `measure.log`（計測実行ログ）

## 実測内容

### 数値一致（bit dump）

main（sha `4e68a4dc`）と branch（feat/metal-resident-weight-grad）の `metal_reuse_step_grad_bit_dump` 出力を比較。4462 行（loss・各 step の重み勾配・パラメータ・最終パラメータ）すべてが **bit 完全一致**（差分 0）。

### framework-compare train A/B

before（origin/main の `crates/facade` path patch）・after（本ブランチ）を同一マシン上で交互実行（5 round・run 単位で順序反転・各腕別バイナリ sha256 記録）。

**判定対象: reuse step_total セル**
- 比率: 0.8784 倍（after 1.196 ms vs before 1.361 ms）
- 5 run 内比率: 0.8580・0.8784・0.8864・0.8632・0.8773（全 run で改善・符号一貫）
- checksum: 完全一致
- 事前登録規則「reuse `step_total` の after/before ≤ 1.00」: **充足（非後退）**

**対照セル: fresh（resident 経路非到達）**
- 比率: 1.0012 倍（after 1.446 ms vs before 1.444 ms）
- 5 run 内比率: 0.9896・0.9485・0.9719・1.0092・1.0024（符号不一致・ノイズ帯）
- 機械判定は「後退」だが、fresh は設計上 resident 経路へ非到達のため計測環境ノイズと帰属
- **規則の緩和ではない**点を明記（#1448/#1506 の教訓）

### フェーズ分解診断

reuse セルの各フェーズ時間（単発・非判定）:

| フェーズ | before (µs) | after (µs) | 比率 |
|---|---|---|---|
| backward | 657.3 | 444.6 | 0.676 |
| device_update | 57.8 | 139.2 | 2.407 |
| step_total | 1338 | 1210 | 0.905 |

device_update の 2.407 倍増加は**同期点の移動**に起因（backward 内の GPU 完了待ちが bias `upload_into` の synchronize へ移動）。全体 step_total は減少（0.905 倍）のため総和は削減。

## 事前登録判定規則

- **Tier 1（必須）**: reuse `step_total` の after/before ≤ 1.00（フレームワーク機械判定）→ **充足**
- **参考**: fresh は対照セル（resident 経路非到達）のため符号不一致は正常

## 参考資料

実装詳細・実測解釈の詳細は `docs/perf/train-resident-grad-device-update.md` §5 を参照。
カウンタ変遷は `docs/backend-metal-command-batching-design.md` §7.2 を参照。
