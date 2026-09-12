# イシュー #1577 実測ログ

`run_ab_1577.sh`: framework-compare train A/B（`--task train --mode
{fresh,reuse}`。5 round・run 単位で before/after 起動順反転・
`--phases` 診断 1 回・`compare_gemm_ab.py --task train --device
<cpu|metal>` で集計）。`FC`／`BEFORE_FACADE`／`AFTER_FACADE`／`OUT`／
`DEVICE`（`cpu`／`metal`／`cuda`）／`LABEL`／`ROUNDS` を環境変数で渡す。
`before-tree`（`git archive origin/main` の非 git 展開ツリー）はログには
含めない（`before-tree/crates/facade` を指すだけの一時ディレクトリ）。

## round1 と round2

- `round1/`: `elementwise_mul_mask` の初版実装（`MaskReadOperand` が
  `Contig` 分岐を持たず、連続入力も含め全読み出しが `checked_mul`／
  `checked_add`／`isize`↔`usize` 変換を伴う `View` 相当の経路を経由して
  いた版）の A/B。孤立マイクロベンチで「連続経路が旧実装（`dense_vec`
  zip）より約 5.6 倍遅い」ことが判明し（`docs/perf/
  train-reuse-relu-mask-stride.md` §5.1 参照）、metal reuse セルが
  事前登録規則の `ratio<=1.00` を満たさず（1.0131）**REJECT**。
- `round2/`: 上記を是正した最終実装（`Contig`／`View`〈`usize` プレーン
  演算〉／`Owned` の 3 分岐・全 contig 高速経路・`Contig`×`View` 混在の
  rank-2 専用経路）の A/B。cpu reuse 0.9196・metal reuse 0.9653 と
  両対象セルが規則を満たし **ADOPT**（`round2/compare-train-{cpu,
  metal}.md` が `compare_gemm_ab.py` の判定表そのもの）。

- `results-{before,after}-{cpu,metal}-train.jsonl`: 5 round 分の
  `--task train`（`--phases` なし）出力。
- `results-{before,after}-{cpu,metal}-phases.jsonl`: 各腕 1 回の
  `--phases` 診断出力（backward／device_update 等の内訳。判定には
  用いない）。
- `uptime-{cpu,metal}.log`: 各 run 前後の `uptime`（load average）記録。
- `compare-train-{cpu,metal}.md`（round2 のみ）:
  `compare_gemm_ab.py --task train --device <device> --threshold 1.00
  --per-run --phases` の生出力（判定表・run 内比・フェーズ分解）。
  round1 は `--device` フラグを渡し忘れ（既定 `metal`）ていたため
  cpu 側が「不正または欠損した 'device' フィールド」で全行 reject
  され `判定不能` になった生成物のみが残っており、本ディレクトリには
  含めていない（本文は `docs/perf/train-reuse-relu-mask-stride.md`
  §4 補足・§5.1 に記録済み。round1 の判定は中央値を Python で直接
  集計する代替方式で確定した）。
- `bitdump-{cpu,metal}-{before,after}.txt`: `crates/facade/tests/
  {cpu,metal}_reuse_step_grad_bit_dump.rs` を before／after 両ツリーで
  `--release -- --ignored --nocapture` 実行した標準出力（loss・各 step
  の重み勾配・パラメータの bit 表現。各 4462 行・`diff` で完全一致
  確認済み。round1／round2 でコードは変わるが `elementwise_mul_mask`
  の入出力契約は不変のため、bit ダンプは round を分けず 1 系列のみ
  〈round2 実装時点〉を採取した）。

## 集計（round2・最終実装）

| device | mode | before median (s) | after median (s) | ratio (after/before) | checksum |
|---|---|---|---|---|---|
| cpu | fresh（ガードセル） | 0.000775 | 0.000803 | 1.0356 | 完全一致 |
| cpu | reuse（対象セル） | 0.000961 | 0.000883 | **0.9196** | 完全一致 |
| metal | fresh（ガードセル） | 0.001537 | 0.001467 | 0.9545 | 完全一致 |
| metal | reuse（対象セル） | 0.001060 | 0.001023 | **0.9653** | 完全一致 |

GB10（CUDA）は本実装エージェントの実行環境に到達手段がなく未実測のまま
記入欄を残す（`docs/real-hardware-verification-env.local.md` 参照可能な
Mac セッションでの追加実測を推奨）。
