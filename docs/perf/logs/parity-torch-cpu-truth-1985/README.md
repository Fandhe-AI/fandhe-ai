# PyTorch cpu N=4096 parity fail 1 要素の真値突合（イシュー #1985）

## 位置づけ

0.9.0 正式再計測（#1967。`docs/perf/logs/framework-compare-0.9.0-remeasure/`）で
PyTorch cpu N=4096 gemm は `parity_fail_count=1`（`max_abs 8.96e-5`・第 3 項 bound
`3.05e-5`・rescued 643）となり、スコアボード規則上「判定不能」であった。本ディレクトリは
その fail 要素を #1184 と同じ手順（`Fraction` 厳密真値・f32 FMA 逐次参照の bit 再現）で
突合した**事実の記録**である。判定・tolerance・判定式（`bench-common::parity`・
`bench_py.py`・`compare_gemm_gate.py`）は変更していない。**採否判定は書かない**。

## 事前登録規則

`RULE.txt`（2026-09-18T01:48:35Z に固定）のとおり。fail 要素が 1 件でない場合も是正・
再試行せず記録のみ。救済可否表は `c ∈ {0.5, 1.0, 1.5}` × `{線形 K, √K}`
（`u=2^-24`・`S_A`／`S_B` は `bench-common/parity.rs::ScaledAbsTolerance::from_inputs`
と同じ大域 `max|A|`／`max|B|`）。

## 実行コマンド（DGX Spark GB10・venv・CPU 既定スレッド数）

```bash
# ノード上。venv は docs/perf/logs/lowlayer-diagnosis-2026-09-12/scripts/dgx-venv.sh が構築したもの
# （torch 2.14.0+cu130）。OMP_NUM_THREADS 未設定（torch.get_num_threads()=20）
cd scripts/bench/framework-compare
uptime > uptime_before.txt
python3 parity_torch_truth.py --self-test 2>&1 | tee self-test.log          # rc=0
python3 parity_torch_truth.py --n 4096 --device cpu \
  --out truth-torch-cpu-4096.json 2>&1 | tee run.log                        # 1 回のみ
uptime > uptime_after.txt
```

スクリプト本体は `scripts/bench/framework-compare/parity_torch_truth.py`（本イシューで
新設。docstring に参照実装の再現方法・二重丸めの扱い・使い方を記載）。Mac 側でも
`--self-test` を実行し rc=0（torch 不在のため torch 経路は skip）を確認した。

## 生成物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則 |
| `run.log` | 本実行のログ（集計 1 行・fail 要素 1 行） |
| `self-test.log` | ノード venv での `--self-test`（torch 経路含め全 ok） |
| `truth-torch-cpu-4096.json` | 集計・fail 要素の全数値（bits・Fraction 真値・救済表・env） |
| `truth-torch-cpu-4096.md` | 同 Markdown 版 |
| `uptime_before.txt`／`uptime_after.txt` | 負荷記録 |
| `env_info.txt` | 実行環境（hostname masked） |

## 結果

### 集計（0.9.0 系列の再現）

| total | fail_count | max_abs_err | max_rel_err | scaled_abs_bound | scaled_abs_rescued |
|---:|---:|---:|---:|---:|---:|
| 16777216 | 1 | 8.964539e-05 | 1.685575e+00 | 3.051757e-05 | 643 |

`fail_count=1`・rescued 643・bound 3.05e-5 は 0.9.0 系列（`results-dgx-py-0.8.0.jsonl`
以来）と同一であり、同一ノードでの 1 回実行で再現した（fail 要素は 1 件のまま）。
`S_A=0.5`・`S_B=0.49999994`・`u=2^-24`。

### fail 要素の真値突合

| idx | row | col | actual（PyTorch。bits） | ref（f32 FMA 逐次。bits） | exact truth | \|actual−truth\| | \|ref−truth\| | \|ref−actual\| | 真値に近い側 | ref_fma_bit_match |
|---:|---:|---:|---|---|---:|---:|---:|---:|---|---|
| 343838 | 83 | 3870 | −1.325470209e-02（0xbc592a40） | −1.322097145e-02（0xbc589cc6） | −1.325068847e-02（= −1864868614407/2^47） | 4.014e-06 | 2.972e-05 | 3.373e-05 | actual（PyTorch） | True |

- `ref_fma_bit_match=True`: `bench_py.py` 方式（k 昇順に f64 加算 → f32 丸め）の参照値は
  真の f32 FMA 逐次累積（各ステップ有理数厳密・1 回丸め）と bit 一致した（二重丸め境界
  ケースではない）。真の FMA 参照でも当該要素は fail のまま（`pass_under_exact_fma_ref=false`）
- `max|partial|=9.65`（k 昇順の部分和の最大絶対値）に対し最終値は 1.3e-2 と 3 桁小さく、
  `|ref−truth|=2.97e-5` は `√K·ulp(max|partial|)=6.1e-5` の範囲内（数値事実のみ）

### 救済可否表（`d=|ref−actual|=3.373e-05 <= bound`）

| c | 形 | bound | idx=343838 |
|---:|---|---:|---|
| 0.5 | 線形 K（**現行契約**） | 3.051757e-05 | fail |
| 0.5 | √K | 4.768371e-07 | fail |
| 1.0 | 線形 K | 6.103515e-05 | 救済 |
| 1.0 | √K | 9.536742e-07 | fail |
| 1.5 | 線形 K | 9.155272e-05 | 救済 |
| 1.5 | √K | 1.430511e-06 | fail |

- `d / bound(c=0.5, 線形 K) = 1.105`。線形 K 形で救済される最小の c は 0.553（`0.5 × 1.105`）
- **issue 本文の見積りとの相違**: issue 本文は「現行 bound を超えるには c≈1.5 が必要」と
  書いているが、実測では線形 K 形は c=1.0 で救済される（c=1.5 も救済）。この差は事実として
  記録する（見積りの根拠との差の原因は本記録では断定しない）。√K 形は c=1.5 でも fail

## 併記事項（採否判定ではない）

issue 本文どおり、**「spec 上正当な判定不能として現状維持」を第一候補として併記する**。
根拠は spec REQ-2（2026-09-12 追記・fandhe-ai-spec#64）が「第三者比較対象の出力が統一複合
判定を外れた場合は比較データの妥当性上の判定不能であり fandhe-ai 側の REQ-2 違反ではない」
と定めていること、および 1 外れ値（16777216 要素中 1 件）に係数を合わせるのは事後緩和に
あたることである。係数 c の変更・線形 K／√K 形の選択はいずれもユーザー承認事項であり、
本記録は判断材料の提示に留める。

## 限界

- 1 回実行（RULE どおり）。PyTorch CPU GEMM の結果はスレッド数・oneDNN のブロッキングに
  依存しうるため、他のスレッド数・他ホストでの fail 要素の再現性は本記録の対象外
- fail 要素 1 件のみの突合であり、pass 側 643 件（現行第 3 項で救済された要素）の真値距離は
  集計していない

## 秘密情報・内部実値の非混入確認

内部ホスト名（ノード名・内部ドメイン）・ユーザー名付きメールアドレス・ホーム
ディレクトリの絶対パスを対象とした `grep -rnE` が本ディレクトリで 0 件であることを
確認済み（`run.log`／`self-test.log`／JSON の `env` にもホスト名・絶対パスは含まれない）。
