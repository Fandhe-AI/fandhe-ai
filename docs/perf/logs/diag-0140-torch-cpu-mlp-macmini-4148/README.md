# <metal-node> の PyTorch CPU の MLP 経路と純カーネル時間（#4148）

## 状態

**PR-3（2026-11-11 実測）: 実測済み。** 本系列は `series-formal-1/`（5 round・全 PASS・正式）。規則は `RULE.txt`（2026-11-11T01:18:14Z 固定・PR #4239）。
部品の PR は `Refs #4148`、本 PR で #4148 を閉じる。値と内訳は下の「結果」と `series-formal-1/aggregate.md`。

## 目的

PyTorch CPU（Apple M4・Accelerate）の窓を、判定ではなく内訳として記録する（record_only。閾値なし）。

1. MLP train（batch 64）・MLP infer の 1 窓を「純カーネル（BLAS）／非 GEMM の要素演算／残差（dispatch・autograd・Python 固定費）」に分ける。
2. GEMM N=256 の窓（相手の純 mm 0.024 ms が窓の 7.5%）を「純 mm／readout／checksum／残差」に分け、torch の mm と
   Accelerate `cblas_sgemm` 直呼びの差（`torch_vs_ctypes_gap`）を出す。Accelerate 側は毎回 Python から ctypes で呼ぶため、差には ctypes の引数変換費（14 引数）が含まれ、torch ラッパー費としては帰属させない（負値もありうる）。
3. 各値に証拠タグ（実測／推定／未測定）を付ける。残差は「推定」。到達不能な量は unavailable のまま空欄にする。

## 読み方の注意

- 時間は **計時腕**（`measure`）の値だけ。`profile` 腕（`torch.profiler`）は計測を摂動するため、op 数と `aten::mm` が子 op を持つか
  （BLAS 呼び出しの直下か）だけを使い、時間値は混ぜない。
- 純カーネル相当は `torch.mm(out=)` を 10 回まとめた 1 回あたり。出力は再利用し（Accelerate 直呼びと同条件）、dispatch 費用は含む「dispatch 込みの torch.mm 時間」で、純カーネル時間そのものではない。そのため残差（推定）は下限側に読む。単発の `mm_pure` は上限側の参考として別に出す。
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
   `python3 -I aggregate_4148.py --check-rule RULE.txt`（`parse_rule`。プレースホルダ 0・sha 一致を検査。`--check-template` は雛形専用でプレースホルダ必須のため固定済みには使えない）で確認して `RULE.txt` を**単独コミット**する。
2. （PR-3・<metal-node>）hw-lock を取り、ツリー指紋を RULE と照合する。
   `ROOT=… D=… EXPECT_SHA=… HW_LOCK_HELD=1 bash run_round.sh <i> <記録先>/round<i>` を 1〜5 で回す。run の差し替え・除外・上書きは禁止。
3. 回収時にパス上のユーザー名・ホスト名をマスクし、`env/` にツリー指紋・`.rev-stamp` を置く。
4. `python3 -I aggregate_4148.py --dir <記録先>` で `aggregate.md`・`aggregate.tsv` を生成し、下の結果欄と
   `docs/perf/loss-attribution-matrix.md` の train／infer／M-CPU-G256 の相手（1）欄を実測値＋タグで更新する（未測定は未測定のまま）。

## 結果

記録先は `series-formal-1/`（`aggregate.md`・`aggregate.tsv`・`round1〜5/`・`round<i>.log`・`drive.status`・`env/`・`RULE.txt` の写し）。
集計は `python3 -I aggregate_4148.py --dir series-formal-1`（rc=0）。record_only のため分類・判定はしない。

### 系列

- 系列の開始: 2026-11-11T01:43:35Z、終了: 01:44:47Z。round 1〜5 とも開始前・終了後の他ジョブのゲートが CLEAN で、`gate.txt` は PASS（`drive.status`）。
- 系列の前の確認（`env/preflight.txt`）: `mediaanalysisd` は 2 分間隔で 3 回とも %CPU 0.0、spotlightknowledged の「Submitting text processing」は直近 10 分で 0 件。
- 系列の始め方（`env/series-start.txt`）と kit の照合（`env/kit-verify.txt`。sha-rc=0・gate-selftest-rc=0・wrap-selftest-rc=0）。
- torch 2.14.0。BLAS 経路は `BLAS_INFO=accelerate`（各 round の `raw/env.json` の `torch_config`）で、宣言どおり Accelerate。
- 系列の途中で値は見ていない（見たのは `drive.status` と `gate.txt` だけ）。

### 区間内訳（5 round の中央値。ms。比は窓に対する比の run 間の中央値）

| section | 窓 | 純 `mm`（10 回まとめ・1 回あたりの和） | 非 GEMM の要素演算・更新など（実測） | 残差（推定） |
|---|---|---|---|---|
| train | 0.13677（最小〜最大 0.11654〜0.13758） | 0.04235（0.310） | ew_fwd 0.00833（0.061）・update 0.04152（0.302）・loss_item 0.00012 | 0.04419（0.325） |
| infer | 0.03169（0.03165〜0.03325） | 0.02089（0.659） | ew_fwd 0.00485（0.151）・readout 0.00058・checksum 0.00229（0.072） | 0.00304（0.096） |
| gemm256 | 0.16440（0.16427〜0.16502） | 0.02311（0.141） | readout 0.00046・checksum 0.13204（0.803） | 0.00879（0.053） |

- train の窓は run 間で 0.11654〜0.13758 ms に広がる（実測のまま。除外・説明はしない）。
- gemm256 の窓の 80% は Python の checksum で、純 `mm` は 14%。Accelerate `cblas_sgemm` の ctypes 直呼び（10 回まとめ）は 0.02339 ms で、`torch_vs_ctypes_gap` は −0.00028 ms（ctypes の引数変換費を含むため torch のラッパー費には帰属させない）。
- profiler 腕（op 数のみ）: step あたりの aten op は train 99（最上位 34）・infer 19（10）・gemm256 10（6）。`aten::mm` は 3 section とも子 op を持つ。
- `dispatch_floor_est` は区間と重なるため和に入れていない（train 0.01152・infer 0.00334 ms・推定）。

### ツリーの指紋（開示）

- 試走 1 の前に、送信側とノード側の `tree_fingerprint` が RULE の値と一致することを確かめた（RULE 項 8 (d)）。本系列の直前には取り直していない。
- 計測後のツリーには `scripts/bench/framework-compare/__pycache__/checksum_contract.cpython-312.pyc` が 1 件増えていた（files=2702）。
  生成は 2026-11-11T01:02:48Z で、試走 1 の round 1 のとき（本系列の前）。`__pycache__` を除いた写しの指紋は RULE の値と一致し、`.rev-stamp` も `tree_sha` と一致する（`series-formal-1/env/trees.txt`）。#4151 の測り直しと同じ扱い。
