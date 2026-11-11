# <metal-node> の M-MTL-TRN（train Metal・相手 PyTorch MPS）の内訳と A/B 要否の判定（#4154）

## 状態

**判定は保留（旧系列を参考値とする。測り直しは見送り）。** 2026-11-10 の系列（下の「結果」）は全 5 ラウンドが他ジョブ（hw-lock を取らない wgpu の GPU GEMM・CPU 100%。所有者の別セッションのジョブで 03:37:56Z に起動。03:38:00〜03:41:02Z は N=2048、03:41:02〜04:00:14Z は N=4096 の GEMM を実行していた）と重なったため**参考**に下げた
（所有者判断（#4151・#4153・#4154 の 2026-11-10 のコメント「他ジョブと重なった系列は測り直す」））。部品は PR-1（#4190）、RULE の固定は PR-2（#4216。前提ゲートの雛形修正は #4210）、旧系列の実測は PR-3（#4218）。
旧系列の数値は消さずに残す。

測り直しは見送り、旧系列を参考値とする（所有者判断（2026-11-10「今の値を参考値として記録する。測定のハードルが高そうなので、できる範囲にしたい」））。RULE.v2 は固定していない。経緯:
他ジョブの前提ゲート（`docs/perf/logs/macmini-foreign-job-gate/`・PR #4227）で各ラウンドを挟む試走 1（2026-11-10T16:41:25〜17:26:07Z。展開先は名前が `.noindex` で終わる新しいディレクトリ・
ビルドの前後で tree_fingerprint が PARAMS と一致）は、round 1 が終了後のゲートで `mediaanalysisd`（%CPU 120.1）に当たって「INVALID foreign-after」、
round 2 は開始前のゲートが 60 回（30 分）とも `mediaanalysisd`（%CPU 82.8〜123.9。うち 1 回は NotificationCenter も）に当たって TIMEOUT になり、round 3 の待ちの途中で止めた
（試走は捨てる前提で、値は見ずにゲートの記録だけを読んだ。リポジトリには入れない）。`mediaanalysisd` は Spotlight の埋め込み生成の依頼
（依頼元 `spotlightknowledged`）を処理していた。この埋め込み生成は 2026-10-11 に始まったバックログで、2026-11-10 時点の進み具合は 0.756。
16:44Z から 40 分以上、毎分 100 件前後の依頼が続いた。書き込み直後の数秒の山は `.noindex` で避けられるが（#4151 の `RULE.v2.txt` 項 10 (e)）、
このバックログの処理は展開先では避けられない。

## 目的

M-MTL-TRN の 0.14.0 の比は 0.96×（fandhe-ai reuse 0.510 ms 対 PyTorch MPS 0.490 ms）で、分類は「分類保留」（相手の経路が未測定）。
fandhe 側の内訳は #3612 で実測済み（`docs/compat-resident-train-step-decision.md` §17.6。Δ約 14 µs は A0 窓のノイズ幅に埋もれる）。
本 issue は**相手側**の内訳を同じセッションで取り、

1. 窓を wall の 3 区間（発行 I・待ち S・残差 R）に分けて fandhe 側と同じ定義で比べ、
2. NB-RANGE5（`docs/perf/noise-band-decision.md`・所有者承認待ち）を当てて、A/B を立てるかを判定する。

判定は N（帯が 1.0 を含む）／W（1 位）／A-S／A-I／A-R（その区間の A/B 候補）／H（保留）。式と閾値は `RULE.template.txt` 項 8。

## 読み方の注意

- 主軸は時計が揃う **wall の区間**。`torch.mps.Event` の GPU span `G_t` は時計 `mps_event`（共有イベントを host が観測した時刻の差と推定）で、
  fandhe の `gpu_busy`（`gpu_timestamp`）とは別の時計。aggregate_diag.py の F1 は「時計不一致・分類保留」になる見込みで、事後に読み替えない。
- イベントの record がコマンドバッファの区切りを変えうるため、event 腕の値は split 腕・bench_py の窓と混ぜない（別プロセス・別の値）。
- 判定は `G_t` に依存しない。`G_t` が unavailable になるのは、学習ステップを含まないプローブで `torch.mps.Event` が無い・使えないと分かった場合だけ（推定で埋めない）。本計測中の失敗（学習ステップ・同期・`item()`・`elapsed_time`）は unavailable に読み替えず event 腕と round を失敗させる。
- event 腕の span の終端は `e1.record()` の後のデバイス全体の `torch.mps.synchronize()`（SGD 更新と `e1` を含む）。`loss.item()` は forward の結果しか待たないので完了待ちに使わない。モックの self-test が検査するのは呼び出し順と完了待ちだけで、MPS の `elapsed_time` の値の正しさは検査できない。
- B1（明示同期）は B0（本番の同期位置）に対し観測負荷を持つ。fandhe 側 I_f・S_f は B1 の値で、負荷は B0 total との比較で記録する。
- F2（fresh 窓の再現）は使わない: torch も parameter を常駐させて更新しており、fandhe の fresh 窓は相手の構成の再現にならない。
- 同じ <metal-node> を他の診断も使いうる。hw-lock で直列化し、並走させない。ビルドは `prebuild.sh` で round 1 の前に 1 回だけ。
- **Metal System Trace（`xctrace`）は採らない**: Linux でエクスポート形式を検証できずパーサの正しさを保証できない、成果物が大きくリポに置けない、
  計測の摂動が大きい、の 3 点による。同じ時計での F1 が必要になった場合は所有者判断で別 issue にする。
- 実機（MPS）経路は PR-1 では Linux で実行できず**未検証**。<metal-node> での初回実行で不具合が出たら、PR-2 の前に修正 PR を出す。

## ファイル

| ファイル | 役割 |
|---|---|
| `RULE.template.txt` | 測る前に固定する規則の雛形（`aggregate_4154.py --check-template` で検査）。項 8 が判定式 |
| `aggregate_4154.py` | 厳格パースの集計・判定（`--dir`・`--write-values`・`--self-test`・`--check-template`・`--check-rule`）。sha256 を RULE に固定 |
| `aggregate_diag.py` | 雛形 `diag-0140-template/aggregate_diag.py` の無改変コピー（F1 の分類・`--fingerprint`）。時計不一致のセルを窓超過で異常終了させず分類保留にする修正（#4190 のレビュー指摘）は、先に本コピーへ入れ、#4208 で雛形へ移して雛形と一致させた。sha256 を RULE の PARAMS に固定 |
| `prebuild.sh` | round 1 の前に 1 回: bench-fandhe（release）と backend-metal lib テストバイナリのビルド・パス記録 |
| `run_round.sh` | 1 round の起動（ゲート → env → 腕。奇数・偶数 round で順序を交互にする） |
| `scripts/bench/framework-compare/torch_mps_train_diag.py` | PyTorch 側の計測本体（`env`／`measure --arm split\|event\|floor`・`--self-test`）。sha256 を RULE に固定 |

## 手順

1. （PR-2）PR-1（#4190）のマージコミットの git archive で `tree_sha`・`tree_fingerprint`（`aggregate_diag.py --fingerprint`）・各 sha256 を埋め、
   `python3 -I aggregate_4154.py --check-rule RULE.txt` で確認して `RULE.txt` を**単独コミット**する。
   項 2 に `git diff --stat v0.14.0 <tree_sha> -- crates/backend-metal crates/facade ':(exclude)crates/facade/tests'` が空であることを事実として書き、
   `crates/facade/tests` の差は一覧を記録する（空でなくても止めない）。
2. （PR-3・<metal-node>）hw-lock を取り、ツリー指紋を RULE と照合する（不一致なら停止）。`ROOT=… OUT=… prebuild.sh` を 1 回、
   続いて `ROOT=… D=… EXPECT_SHA=… TESTBIN=… HW_LOCK_HELD=1 bash run_round.sh <i> <記録先>/round<i>` を 1〜5 で回す。run の差し替え・除外・上書きは禁止。
3. 回収時にパス上のユーザー名・ホスト名をマスクし、`env/` にツリー指紋・`.rev-stamp`・`build-sha256.txt` を置く。
4. `python3 -I aggregate_4154.py --dir <記録先>` で判定を出す。`G_t` が取れた round は `--write-values <記録先>/round<i>` で values.tsv を作り、
   `python3 -I aggregate_diag.py --dir <記録先>` の出力（F1・分類保留の理由）も並べて置く。
5. 下の結果欄と `docs/perf/loss-attribution-matrix.md` §10.11 の M-MTL-TRN の相手側、`docs/compat-resident-train-step-decision.md` §17.6 の
   「torch 側の内訳は未測定」への参照を、実測値とタグで更新する（未測定は未測定のまま）。
6. 判定が A-S／A-I／A-R の場合は、追加起票を所有者に提案するコメントを #4154 に書く（起票は承認後）。`unreached_0140.tsv` への反映は #4168。

## 結果

> **他ジョブ（hw-lock を取らない wgpu の GPU GEMM・CPU 100%）と重なった系列・参考値（判定は保留。測り直しは見送り）。** 全 5 ラウンド（round 1 03:52:39〜03:52:45Z、round 2 〜03:52:49Z、round 3 〜03:52:52Z、
> round 4 〜03:54:26Z、round 5 〜03:54:30Z）が、他ジョブの N=4096 の GEMM（03:41:02〜04:00:14Z）と重なった。load1 のゲート（3.0 未満）は
> 通ったが、このゲートでは重なりを検出できない。下の判定・確度は旧系列のもので、判定には使わない（参考値。上の「状態」。所有者判断（#4151・#4153・#4154 の 2026-11-10 のコメント「他ジョブと重なった系列は測り直す」））。

計測: 2026-11-10T03:52:39Z〜03:54:30Z（<metal-node>・Apple M4・torch 2.14.0）。5 ラウンドとも PASS（load1 2.65〜2.95。直前に同じ機体で
#4153・#4151 の計測が続けて走っていたため、ゲートの閾値 3.0 に近い。後に、同じ時間帯に他ジョブが動いていたことが分かった。上の注記）。prebuild は計測前に 1 回（`env/build-*`）。計測後のツリー指紋は
RULE の値と一致（`env/tree-fingerprint.txt`・`env/rev-stamp.txt`）。判定は `aggregate_4154.md`／`aggregate_4154.tsv`、
`aggregate_diag.py` の参考出力は `aggregate.md`／`aggregate.tsv`、ラウンド別の値は `round{i}/values.tsv`（`--write-values`）。

**旧系列の判定（参考・保留）: N（A/B を立てない）／確度: 高**（全 round PASS・全 round で TW の checksum と TS の最終 loss が一致）

| run | W_f | W_t | 観測比 | I_f | I_t | S_f | S_t | R_f | R_t | Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 822.9 | 489.3 | 0.5946 | 10.9 | 179.7 | 661.9 | 291.9 | 150.1 | 17.7 | 333.6 |
| 2 | 475.7 | 494.6 | 1.0398 | 10.9 | 180.3 | 431.2 | 292.5 | 33.5 | 21.8 | −18.9 |
| 3 | 473.3 | 486.2 | 1.0273 | 11.0 | 179.5 | 440.5 | 292.0 | 21.8 | 14.7 | −12.9 |
| 4 | 474.8 | 564.9 | 1.1897 | 10.9 | 192.3 | 438.8 | 291.6 | 25.1 | 81.0 | −90.1 |
| 5 | 473.0 | 500.8 | 1.0589 | 10.9 | 182.6 | 433.8 | 290.0 | 28.2 | 28.2 | −27.9 |

（単位 µs/step。fandhe 側 I=prep+encode・S=commit_wait〈B1〉・R=窓−I−S。torch 側 I=step 本体を出し終えるまで・S=synchronize が返るまで）

- 観測比の中央値 1.0398、NB-RANGE5 の帯 0.5946〜1.1897。帯が 1.0 を含むので規則 1 の N。スコアボードの 0.96 から 8.3% 離れた（10% 以内なのでドリフトの記録なし）。
- **帯が 1.0 を含むのは round 1 の 1 本による。** round 2〜5 の観測比は 4 本とも 1.0 を超え（1.027〜1.190）、round 1 だけ fandhe の W_f が 822.9 µs
  （ほかは 473〜476 µs）、B1 の gpu_busy も 460 µs（ほかは 234 µs）だった。round 1 は奇数 round で fandhe の腕（A0）が最初に走る順序で、
  #4151 の計測が終わった直後に始まった。RULE は外れ値を除かない（事後に除外・再実行しない）ので、そのまま判定した。
  #3612（`docs/compat-resident-train-step-decision.md` §17.6）でも A0 の round 1 が外れ値（871.7 µs）だった。
- 区間の中央値（列ごと。µs/step）: fandhe I_f 10.9・S_f 438.8・R_f 28.2、torch I_t 180.3・S_t 291.9・R_t 21.8。
  torch は step 本体の発行（I_t）に約 180 µs かかり、待ち（S_t）は fandhe より約 147 µs 短い。fandhe は発行が約 11 µs と小さく、待ちが大きい
  （1 コマンドバッファを最後に待つ常駐チェーンの構成どおり）。round 2〜5 では ΔI（−168〜−181）と ΔS（+139〜+148）がほぼ相殺し、Δ は −13〜−90 µs。
- 補助（推定タグ。時計が揃わないため区間と足さず、確定に使わない）: G_t（MPS イベントの GPU span）中央値 423.9 µs・fandhe の gpu_busy 234.0 µs、
  F_roundtrip（1 要素 add + item）182.8 µs・fandhe の host_fixed 202.7 µs、空の synchronize 0.1 µs。B0／B1 の total 451.6／450.2 µs。
- `aggregate_diag.py`（参考）: F1 の時計不一致で「分類保留・確度低」（RULE 項 4 の想定どおり。values.tsv の純カーネルは時計が違う）。
- A/B を立てる判定ではないため、追加起票の提案はしない（閉じる条件 2 は該当なし）。`unreached_0140.tsv` への反映は #4168。
