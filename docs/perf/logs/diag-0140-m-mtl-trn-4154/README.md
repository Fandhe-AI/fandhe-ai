# <metal-node> の M-MTL-TRN（train Metal・相手 PyTorch MPS）の内訳と A/B 要否の判定（#4154）

## 状態

**PR-1 時点: 実測なし。** 本ディレクトリは測る前に固定する部品（RULE 雛形・集計スクリプト・起動キット）だけを置く。
`RULE.txt`（PR-2・単独コミット）、`round{i}/`・`env/`・`aggregate_4154.md`・`aggregate_4154.tsv`（PR-3・<metal-node> での実測）は後続で追加する。
実測値・内訳は、実測するまで書かない（推定値で埋めない）。#4154 は実測と判定の記録（PR-3）で閉じる。部品の PR は `Refs #4154`。

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
- 判定は `G_t` に依存しない。`G_t` が取れなければ unavailable のまま残す（推定で埋めない）。
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
| `aggregate_diag.py` | 雛形 `diag-0140-template/aggregate_diag.py` の逐語コピー（F1 の分類・`--fingerprint`）。sha256 を RULE の PARAMS に固定 |
| `prebuild.sh` | round 1 の前に 1 回: bench-fandhe（release）と backend-metal lib テストバイナリのビルド・パス記録 |
| `run_round.sh` | 1 round の起動（ゲート → env → 腕。奇数・偶数 round で順序を交互にする） |
| `scripts/bench/framework-compare/torch_mps_train_diag.py` | PyTorch 側の計測本体（`env`／`measure --arm split\|event\|floor`・`--self-test`）。sha256 を RULE に固定 |

## 手順

1. （PR-2）PR-1 のマージ後の main SHA の git archive で `tree_sha`・`tree_fingerprint`（`aggregate_diag.py --fingerprint`）・各 sha256 を埋め、
   `python3 -I aggregate_4154.py --check-rule RULE.txt` で確認して `RULE.txt` を**単独コミット**する。
   項 2 に `git diff --stat v0.14.0 <tree_sha> -- crates/backend-metal crates/facade` が空であることを事実として書く。
2. （PR-3・<metal-node>）hw-lock を取り、ツリー指紋を RULE と照合する（不一致なら停止）。`ROOT=… OUT=… prebuild.sh` を 1 回、
   続いて `ROOT=… D=… EXPECT_SHA=… TESTBIN=… HW_LOCK_HELD=1 bash run_round.sh <i> <記録先>/round<i>` を 1〜5 で回す。run の差し替え・除外・上書きは禁止。
3. 回収時にパス上のユーザー名・ホスト名をマスクし、`env/` にツリー指紋・`.rev-stamp`・`build-sha256.txt` を置く。
4. `python3 -I aggregate_4154.py --dir <記録先>` で判定を出す。`G_t` が取れた round は `--write-values <記録先>/round<i>` で values.tsv を作り、
   `python3 -I aggregate_diag.py --dir <記録先>` の出力（F1・分類保留の理由）も並べて置く。
5. 下の結果欄と `docs/perf/loss-attribution-matrix.md` §10.11 の M-MTL-TRN の相手側、`docs/compat-resident-train-step-decision.md` §17.6 の
   「torch 側の内訳は未測定」への参照を、実測値とタグで更新する（未測定は未測定のまま）。
6. 判定が A-S／A-I／A-R の場合は、追加起票を所有者に提案するコメントを #4154 に書く（起票は承認後）。`unreached_0140.tsv` への反映は #4168。

## 結果

（実測後に追記。PR-1 時点は空。）
