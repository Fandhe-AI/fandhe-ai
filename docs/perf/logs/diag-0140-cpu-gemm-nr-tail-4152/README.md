# <metal-node> の単スレッド GEMM で NR=12 の端数が無駄になっているかの診断（#4152）

## 状態

**実測済み（2026-11-10・<metal-node>・5 round すべて PASS）。** 規則は `RULE.txt`（#4212 で単独コミット）、実測は `round{1..5}/`、
集計は `aggregate.md`・`aggregate.tsv`。判定は下の「結果」。#4152 は本 PR（実測と判定の記録）で閉じる条件を満たした。

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

実測: 2026-11-10・<metal-node>（Apple M4）・tree `a52d4cfc15b81903a6fe471644ab21c3db18df78`（ツリー指紋は RULE と一致。`round{i}.log` の `tree OK`）。
5 round とも hw-lock 取得済み・ゲート PASS（load1 2.22。`round{i}/gate.txt`。`drive.status` は hw-lock を取ったまま round 1〜5 を順に回した起動側の記録）。差し替え・除外・再実行はしていない。
集計は `python3 -I aggregate_4152.py --dir .`（status: complete）。生出力のホーム・ホスト名・ユーザー名はマスク済み。

> **他ジョブとの重なりの確認（2026-11-10 追記。所有者判断「他ジョブと重なった系列は測り直す」に伴う確認）:** #4151・#4153・#4154 と重なった他ジョブ
> （hw-lock を取らない wgpu の GPU GEMM。03:37:56Z 起動。作業ディレクトリと driver の作成は 03:26:47Z 以降）の driver との重なりはない
> （本系列は 03:08:26〜03:08:28Z）。03:08Z 時点の ps が無いため、ほかのジョブの有無は確かめられない。測り直しは不要（所有者判断）。

| family（t1・n=10） | w 中央値 [最小〜最大] | g 中央値 [最小〜最大] | r10_12 中央値 | 分類 | 確度 |
|---|---|---|---|---|---|
| l2_fwd（Nn・64×10×256） | 0.092 [0.088〜0.095] | 0.124 [0.119〜0.129] | 1.032 | 寄与あり | 通常 |
| l2_dw（Tn・256×10×64） | 0.150 [0.147〜0.156] | 0.197 [0.189〜0.201] | 1.120 | 寄与あり（大） | 通常 |

- **NR=12 の端数は L2 の細い形状の時間に寄与している**が、主因ではない。端数の無駄 `w` は n=10 の時間の約 9%（fwd）・約 15%（dW）で、
  5 round の帯が `w_small`（0.03）を下回らない（dW は `w_large`〈0.10〉も下回らない）。正方 512 の水準との差のうち端数で説明できる割合 `g` は
  0.12〜0.20 で、`g_main`（0.50）に届かない。MM-177 の 3〜4 倍の差の大半は端数以外にある。
- n=10 は n=12 より仕事が少ないのに時間が長い（`r10_12` 1.03・1.12）。ゼロ詰めだけなら `r10_12` は 1.0 になるはずで、1.0 を超える分は端タイル固有の追加費用（コピー往復など。内訳は未分離）がある形である。
- 健全性: n=10 の t1 GFLOPS 中央値は l2_fwd 29.9（MM-177 §3 の 32.5 の 0.92 倍）・l2_dw 25.7（同 28.8 の 0.89 倍）で、`sanity_lo`〜`sanity_hi`（0.5〜2.0）の内側。確度は「通常」のまま。
- 事前の予想（窓全体で 6〜7% 程度。判定には使わない）より `w` は大きかった（fwd 約 9%・dW 約 15%）。dW は予想の上限（kernel 時間の 2/12 ≈ 0.167）に近い。
  レーンの無駄とコピー往復のどちらが効いているかは本診断の範囲外（kernel 腕は実施しない。`対象外` 参照）。
- record_only（判定に使わない）: t1 の GFLOPS は n=12 が n=11・13 より高く、n=13 で落ちる（次の NR 境界の端数）。native 系列は n で job grid が変わるため比べない。
  L1（n=256・264・276）は #4151 と重なるので判定しない。
- ノイズ帯 NB-RANGE5 は #4142 で 2026-11-10 に定義を変えずに承認された（`docs/perf/noise-band-decision.md` §6）ため、`RULE.v2.txt` は置かない。
- 分類の反映（`unreached_0140.tsv`・`loss-attribution-matrix.md` の MM-CPU-TRN 行）は #4168 が行う。本 PR では書き換えない。
