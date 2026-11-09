# <metal-node> の単スレッド GEMM で NR=12 の端数が無駄になっているかの診断（#4152）

## 状態

**PR-1 時点: 実測なし。** 本ディレクトリは測る前に固定する部品（RULE 雛形・集計スクリプト・1 round の起動キット）だけを置く。
`RULE.txt`（PR-2・単独コミット）、`round{i}/`・`aggregate.md`・`aggregate.tsv`（PR-3・<metal-node> での実測）は後続で追加する。
実測値・判定は、実測するまで書かない。#4152 は実測と判定の記録（PR-3）で閉じる（部品の PR は `Refs #4152`）。

## 目的

MM-177（`docs/perf/cpu-gemm-m64-jobgrid-diagnosis.md` §3）で、<metal-node>（Apple M4）の単スレッド GEMM は L2 の細い形状（n=10）が
29〜40 GFLOPS にとどまる（正方 512 は 117.5）と記録された。aarch64 の既定タイルは NEON の MR=8・NR=12 なので、n=10 は 12 レーンのうち 2 レーンが
ゼロ詰めになり、端タイル（`nr_eff < NR`）はスタックバッファへのコピーインとコピーアウトの往復も通る。この 2 つが低 GFLOPS の原因かを、
n を掃引した時間から次の式で判定する（式と閾値の正は `RULE.txt`。雛形は `RULE.template.txt`）。

- `P = t(24) − t(12)`（NR 1 列パネルの限界費用）、`t_ideal(10) = t(12) − P·2/12`、`waste = t(10) − t_ideal(10)`、`w = waste/t(10)`
- `g = waste / (t(10) − t_peak(10))`（正方 512 との差のうち端数で説明できる割合）
- 分類: 判定不能（P ≤ 0）／主因（min g ≥ 0.50）／寄与あり（min w ≥ 0.03。0.10 以上は「大」）／無視できる（max w < 0.03）／判定不能（揺れの中）

事前の予想（判定には使わない）: 端数が無駄にする分は kernel 時間の 2/12 が上限で、窓全体では 6〜7% 程度。MM-177 の 3〜4 倍の差を端数だけで説明できる見込みは薄い。

## 対象外

- L2 dX（64×256×10 Nt）の細さは k=10 の側にあり、NR とは別の原因（別の診断が要る。起票は所有者の承認が要る）
- L1（n=256。256 mod 12 = 4）の固定費は #4151 と重なるため判定しない（record_only で曲線だけ記録）
- GB10 での同じ診断（x86 の NR は AVX2 で 16・AVX-512 で 32）。Linux での実行は動作確認にしかならず、`nr≠12` で集計が止まる
- kernel 腕（1 タイル単体の計時）は実施しない（#3614 で区間の分解能不足と判定済み）。レーンの無駄とコピー往復の内訳の分離は本診断の範囲外
- `unreached_0140.tsv` と `loss-attribution-matrix.md` の分類の付け直しは #4168。NB-RANGE5 の所有者承認は #4142

## 読み方の注意

- 判定系列は `t1`（`RAYON_NUM_THREADS=1`）。`native` は n で job grid の列帯数が変わり交絡するため record_only
- SME は既定 OFF（`SME_PRODUCTION_ENABLED=false`・`SME_THIN_M64_ENABLED=false`）で、しきい値（m・n ≥ 256）にも届かないため n の掃引は NEON 経路だけを通る
- n=10 の t1 GFLOPS が MM-177 §3 の値（L2 fwd 32.5・L2 dW 28.8）の 0.5〜2 倍の外なら、環境差の可能性を注記し確度を「低」にする（判定は止めない）
- ノイズ帯 NB-RANGE5 は `docs/perf/noise-band-decision.md`（案・所有者承認待ち。#4142）。承認後に定義が変われば `RULE.v2.txt` で反映する
- 同じ <metal-node> を他の診断（#4146・#4148 ほか）も使いうる。hw-lock で直列化し、並走させない。公開面に実機のホスト名を書かない

## ファイル

| ファイル | 役割 |
|---|---|
| `RULE.template.txt` | 測る前に固定する規則の雛形（`aggregate_4152.py --check-template` で検査。`<…>` にするのは SHA・日時・ハッシュだけ） |
| `aggregate_4152.py` | 厳格パースの集計（`--dir`・`--self-test`・`--check-template`・`--check-rule`・`--fingerprint`）。sha256 を RULE に固定 |
| `run_round.sh` | 1 round の起動（ゲート → t1 → native。奇数 round は n 昇順・偶数 round は降順） |
| `crates/backend-cpu/examples/cpu_gemm_nr_tail_4152.rs` | 計測本体。公開入口 `gemm_blis_parallel{,_tn}` を n 掃引で計時し `bits_hash` を記録。sha256 を RULE に固定 |

## 手順

1. （PR-2）PR-1 のマージ後の main SHA の git archive を展開し、`python3 -I aggregate_4152.py --fingerprint <tree-dir>` で
   `tree_fingerprint` を得る。`tree_sha`・`example_sha256`・`aggregate_script_sha256`・`fixed-at-utc` を埋めて
   `python3 -I aggregate_4152.py --check-rule RULE.txt` で確認し、`RULE.txt` を**測る前に単独コミット**する。
2. （PR-3・<metal-node>）hw-lock を取り、`python3 -I aggregate_4152.py --check-tree <tree-dir> --rule RULE.txt` で example_sha256・tree_fingerprint を RULE と照合する（`run_round.sh` も `<記録先>/RULE.txt` で同じ照合を毎 round 行い、不一致なら計測しない）。`cargo build --release -p fandhe-ai-backend-cpu --example cpu_gemm_nr_tail_4152` を済ませ、
   `ROOT=… EXPECT_SHA=… HW_LOCK_HELD=1 bash run_round.sh <i> <記録先>/round<i>` を 1〜5 で回す。run の差し替え・除外・上書きは禁止。
3. 回収時にパス上のユーザー名・ホスト名をマスクする。
4. `python3 -I aggregate_4152.py --dir <記録先>` で `aggregate.md`・`aggregate.tsv` を生成し、下の結果欄へ判定を転記して `Closes #4152` とする。

## 結果

（実測後に追記。PR-1 時点は未実測。）
