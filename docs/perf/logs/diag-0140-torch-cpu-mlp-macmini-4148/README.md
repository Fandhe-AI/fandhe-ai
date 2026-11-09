# <metal-node> の PyTorch CPU の MLP 経路と純カーネル時間（#4148）

## 状態

**PR-1 時点: 実測なし。** 本ディレクトリは測る前に固定する部品（RULE 雛形・集計スクリプト・1 round の起動キット）だけを置く。
`RULE.txt`（PR-2・単独コミット）、`round{i}/`・`env/`・`aggregate.md`・`aggregate.tsv`（PR-3・<metal-node> での実測）は後続で追加する。
実測値・内訳は、実測するまで書かない（推定値で埋めない）。#4148 は実測と記録の PR-3 で閉じる（部品の PR は `Refs #4148`）。

## 目的

PyTorch CPU（Apple M4・Accelerate）の窓を、判定ではなく内訳として記録する（record_only。閾値なし）。

1. MLP train（batch 64）・MLP infer の 1 窓を「純カーネル（BLAS）／非 GEMM の要素演算／残差（dispatch・autograd・Python 固定費）」に分ける。
2. GEMM N=256 の窓（相手の純 mm 0.024 ms が窓の 7.5%）を「純 mm／readout／checksum／残差」に分け、torch の mm と
   Accelerate `cblas_sgemm` 直呼びの差（torch ラッパー費）を出す。
3. 各値に証拠タグ（実測／推定／未測定）を付ける。残差は「推定」。到達不能な量は unavailable のまま空欄にする。

## 読み方の注意

- 時間は **計時腕**（`measure`）の値だけ。`profile` 腕（`torch.profiler`）は計測を摂動するため、op 数と `aten::mm` が子 op を持つか
  （BLAS 呼び出しの直下か）だけを使い、時間値は混ぜない。
- 純カーネルは `torch.mm` を 10 回まとめた 1 回あたり（dispatch を償却した値）。単発の `mm_pure` は上限側の参考として別に出す。
- 学習の GEMM は l1_fwd・l2_fwd・l2_dw・l2_dx・l1_dw の 5 形状（l1_dx は PyTorch が計算しない）。推論は l1_fwd・l2_fwd。
- 残差（推定）には autograd エンジン・Python 解釈・メモリ確保・backward の要素演算が混ざる。`dispatch_floor_est` は区間と重複するため和に入れない。
- BLAS 経路が Accelerate でなくても宣言は動かさず、実測時に `env/`（`torch.__config__.show()`・`otool -L`）の事実をここへ注記する。
- Accelerate 直呼びは固定パス・固定シンボル（`cblas_sgemm`）だけを ctypes で呼ぶ。結果が `torch.mm` と複合判定で一致しなければ unavailable にする。
- ノイズ帯 NB-RANGE5 は `docs/perf/noise-band-decision.md`（案・所有者承認待ち。#4142）。承認後に定義が変われば `RULE.v2.txt` で反映する。
- issue 本文の内訳（206／452／118 µs）は 0.11.0 時点の fandhe 側（MM-180）の値で、本 issue の PyTorch 内訳とは別物。
- 同じ <metal-node> を #4146 の実測も使いうる。hw-lock で直列化し、並走させない。

## ファイル

| ファイル | 役割 |
|---|---|
| `RULE.template.txt` | 測る前に固定する規則の雛形（`aggregate_4148.py --check-template` で検査） |
| `aggregate_4148.py` | 厳格パースの集計（`--dir`・`--self-test`・`--check-template`・`--fingerprint`）。sha256 を RULE に固定 |
| `run_round.sh` | 1 round の起動（ゲート → env → measure → profile） |
| `scripts/bench/framework-compare/torch_cpu_mlp_diag.py` | 計測本体（`env`／`measure`／`profile`・`--self-test`）。sha256 を RULE に固定 |

## 手順

1. （PR-2）PR-1 のマージ後の main SHA の git archive で `tree_sha`・`tree_fingerprint`・各 sha256 を埋め、
   `aggregate_4148.py --check-template` でプレースホルダ 0 を確認して `RULE.txt` を**単独コミット**する。
2. （PR-3・<metal-node>）hw-lock を取り、ツリー指紋を RULE と照合する。
   `ROOT=… D=… EXPECT_SHA=… HW_LOCK_HELD=1 bash run_round.sh <i> <記録先>/round<i>` を 1〜5 で回す。run の差し替え・除外・上書きは禁止。
3. 回収時にパス上のユーザー名・ホスト名をマスクし、`env/` にツリー指紋・`.rev-stamp` を置く。
4. `python3 -I aggregate_4148.py --dir <記録先>` で `aggregate.md`・`aggregate.tsv` を生成し、下の結果欄と
   `docs/perf/loss-attribution-matrix.md` の train／infer／M-CPU-G256 の相手（1）欄を実測値＋タグで更新する（未測定は未測定のまま）。

## 結果

（実測後に追記。PR-1 時点は空。）
