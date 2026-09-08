# CPU GEMM `TwoDDynamic` vs `RowPanel` 両実機 A/B（1024/2048/4096）・採用ゲート判定（イシュー #1312）

## 状態: 実機実測完了。判定は **undetermined**（M4 Max 専有ゲート未通過。DGX 単独では ADOPT しない規則。#1313 へ結線せず記録のみで引き継ぐ）。**#1313 で M4 Max 専有ゲート通過後の Phase 0 再計測を実施し、jpw=2・jpw=4 とも Tier 1 条件を充足したため ADOPT へ確定・本番結線済み**（「#1313 追記」節参照）。

## 位置づけ

起票元 #1312・親系列 #1307/#1303/#1283。前提 #1311（実装。`GemmDriverVariant::TwoDDynamic`・
`gemm_blis_parallel_two_d_dynamic_with_params`・`TWO_D_JOBS_PER_WORKER=2` を追加）。
設計 `docs/cpu-gemm-2d-dynamic-partition-design.md` §9・§11 の採用ゲートに基づく。
後続 #1313（本番結線。本イシューが undetermined と判定したため「結線せず close」で引き継ぐ）。

先行 #1305（`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §17）は DGX Spark GB10
（Cortex-X925 ×10 + Cortex-A725 ×10 の異種コア）で `RowPanel`（本番既定・静的等分割の行パネル）
の N=1024・T=10 が T=8 比 0.5094 倍という非単調性を持つことを確定し、H1（異種コア由来）と
結論した。本イシューは (mc, nc) 2D job 動的分配（`TwoDDynamic`）がこの非単調性を緩和するか、
かつ本番既定を上回るかを両実機・5 回独立プロセス中央値で判定する。

## 実行環境

| 実機 | 役割 | 備考 |
|---|---|---|
| DGX Spark GB10（Grace CPU・20 論理コア） | Tier 1 主判定・AC-2（非単調性）判定 | 専有ゲート通過（load average < 6 を 2 回連続で確認。`docs/perf/logs/cpu-gemm-2d-dynamic-ab-1312/gate-dgx.log`） |
| Apple M4 Max（16 論理コア） | Tier 1 主判定（両実機一致要件） | **専有ゲート不通過**（1 分 load average が計測直前 59.65〜35 台・約 22 分の手動観測後も 11.84 で閾値 6.0 の約 2 倍。他セッション並走による共有負荷。`docs/perf/logs/cpu-gemm-2d-dynamic-ab-1312/gate-m4max.log`。**実行プロトコルに関する注記は下記参照**） |

対象コミット: `f719012`（`test(backend-cpu): TwoDDynamic の jobs_per_worker 別 A/B ハーネスを追加する`）。

### M4 Max 低負荷ゲートの実行プロトコルに関する注記

計画 §4.1 条件 7 は「スクリプト化した 60 秒間隔ポーリング・1 分 load average < 6.0 が
2 回連続・最大 30 分待機」という自動プロトコル（#1305/#1319 が `m4max_orchestrate.sh` 等で
実装した方式）を規定しているが、**本イシューではこのフル自動プロトコルを実行できなかった**。
本エージェント実行環境のサンドボックス制約（バックグラウンド実行を伴わない長時間 `sleep`
の直接実行禁止・`while` ループを含む複雑なコマンド構成の拒否）により、機械的なポーリング
スクリプトを起動できなかったため。代わりに、他のツール呼び出しの合間に手動で複数回
`uptime` を取得した:

| 経過時間 | 1 分 load average |
|---|---|
| 計測直前（02:08:31Z） | 59.65 |
| +数十秒 | 55.67 → 46.69 → 37.47（3 サンプル） |
| +約 21 分（02:29:51Z） | 13.56 |
| +約 22 分 | 12.79 → 11.85 → 11.84（3 サンプル） |

単調減少はしているが、約 22 分間の観測を通じて閾値 6.0 を一度も下回らなかった（最終値
11.84 は閾値の約 2 倍）。したがって「2 回連続 <6.0」は成立せず、低負荷ゲートは**不成立**と
判断する。この判断はフル自動プロトコル（60 秒間隔・最大 30 分）の完全な代替ではなく、
手動サンプリングによる代替確認である点を明記する（詳細な全サンプルは
`docs/perf/logs/cpu-gemm-2d-dynamic-ab-1312/gate-m4max.log`）。

## 事前宣言ゲート（計測前に固定。§4 は計画の記載を転記したもの）

### Tier 1（結線判定。ADOPT を決める。対 `RowPanel`）

同一プロセス内 round-robin 計測の 5 回独立プロセス中央値比 `TwoDDynamic(jpw) / RowPanel`
（`RAYON_NUM_THREADS` 未設定＝既定全コア。DGX 20・M4 Max 16）:

1. N=1024・N=2048 の両方で比 ≥ 1.00、かつ N=4096 で比 ≥ 0.95
2. 単一の固定 `jobs_per_worker`（2 または 4）が条件 1 を **M4 Max・GB10 の両方**で満たす場合
   のみ ADOPT。片実機のみは REJECT
3. ノイズガード: 中央値比 ≥ 1.00 でもペアワイズ勝ち run 数が 5 中 3 未満の形状があれば
   **undetermined**
4. 中止条件: DGX N=1024・T=10 で jpw 2・4 のいずれも対 `RowPanel` 比 < 1.00 → REJECT
5. 前提: 両実機で `gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_large`（`#[ignore]`）
   pass、`cargo test -p fandhe-ai-backend-cpu --lib gemm_blis --release` green
6. 条件付き採用: 述語（`n`・`m`・`effective_num_threads` のみ）で表現できる部分集合が両実機で
   条件 1 を満たす場合に限り明記する
7. **M4 Max 専有性**: 低負荷ゲート不通過なら「共有負荷下」と明記し計測は続行するが、
   **ADOPT／条件付き ADOPT は M4 Max のゲート通過が必須**（不通過なら undetermined として
   #1313 へ「結線せず記録のみ」を引き継ぐ）。REJECT は DGX（ゲート通過）＋同方向の M4 Max
   結果で確定してよい

### Tier 2（#1041 ゲート。対 gemm crate。#1313 への参考値・結線判定には使わない）

N=1024/2048 で「候補の GFLOP/s ÷ gemm crate の tflops_median × 1000」が ≥ 1.00 かつ N=4096 で
`RowPanel` 比非劣化。

### AC-2 判定基準（非単調性の解消／残存）

DGX・N=1024 の `TwoDDynamic(jpw)(T=10) / TwoDDynamic(jpw)(T=8)`: ≥ 1.00（5 run 中 ≥3 run で
1.0 以上）→「解消」、0.90 未満 →「残存」、その間は「部分的緩和」。

---

## 実測結果

### Tier 1: DGX Spark GB10（`RAYON_NUM_THREADS` 未設定＝既定全コア 20）

出典: `docs/perf/logs/cpu-gemm-2d-dynamic-ab-1312/ab-{1024-2048,4096}-dgx-Tdefault-run{1..5}.txt`・
`aggregate.py` 出力。

| size | candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | 勝ち run 数 |
|---|---|---|---|---|---|
| 1024 | RowPanel | 0 | 543.901 | 1.0000 | — |
| 1024 | TwoDDynamic | 2 | 713.834 | **1.3124** | 5/5 |
| 1024 | TwoDDynamic | 4 | 693.188 | **1.2745** | 5/5 |
| 2048 | RowPanel | 0 | 712.477 | 1.0000 | — |
| 2048 | TwoDDynamic | 2 | 1285.474 | **1.8042** | 5/5 |
| 2048 | TwoDDynamic | 4 | 1172.275 | **1.6454** | 5/5 |
| 4096 | RowPanel | 0 | 1112.779 | 1.0000 | — |
| 4096 | TwoDDynamic | 2 | 1214.659 | **1.0916** | 5/5 |
| 4096 | TwoDDynamic | 4 | 1300.746 | **1.1689** | 5/5 |

DGX は条件 1・条件 3 を jpw=2・jpw=4 の両方で満たす（全形状で比 ≥1.00〈4096 も ≥0.95 を大きく
上回る〉・勝ち run 数は全形状・両 jpw とも 5/5）。

### Tier 1: Apple M4 Max（`RAYON_NUM_THREADS` 未設定＝既定全コア 16。**共有負荷下**）

出典: `docs/perf/logs/cpu-gemm-2d-dynamic-ab-1312/ab-{1024-2048,4096}-m4max-Tdefault-run{1..5}.txt`。

| size | candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | 勝ち run 数 |
|---|---|---|---|---|---|
| 1024 | RowPanel | 0 | 393.673 | 1.0000 | — |
| 1024 | TwoDDynamic | 2 | 467.768 | 1.1882 | 5/5 |
| 1024 | TwoDDynamic | 4 | 448.046 | 1.1381 | 4/5 |
| 2048 | RowPanel | 0 | 430.647 | 1.0000 | — |
| 2048 | TwoDDynamic | 2 | 510.529 | 1.1855 | 5/5 |
| 2048 | TwoDDynamic | 4 | 500.705 | 1.1627 | 5/5 |
| 4096 | RowPanel | 0 | 408.255 | 1.0000 | — |
| 4096 | TwoDDynamic | 2 | 396.482 | 0.9712 | 3/5 |
| 4096 | TwoDDynamic | 4 | 422.861 | 1.0358 | 5/5 |

数値上は M4 Max も条件 1・条件 3 を jpw=2・jpw=4 の両方でぎりぎり満たす（N=4096 の jpw=2 は
勝ち run 数 3/5 とノイズガード閾値ちょうど）。**しかしこの計測は §4.1 条件 7 の低負荷ゲート
（1 分 load average < 6 を 2 回連続）を通過していない**（実測時 1 分 load average は概ね
30〜60（他セッション並走の共有負荷下。`gate-m4max.log`）で推移しており、GB10 単独より 1〜2 桁
高い）。条件 7 は「ADOPT／条件付き ADOPT は M4 Max のゲート通過が必須」と明記しているため、
数値が条件を満たしているように見えても本イシューでは ADOPT の根拠にしない。

### DGX 補助軸: T=8・T=10（AC-2・非単調性クロスチェック用）

| threads | size | candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | 勝ち run 数 |
|---|---|---|---|---|---|---|
| 8 | 1024 | RowPanel | 0 | 722.691 | 1.0000 | — |
| 8 | 1024 | TwoDDynamic | 2 | 711.040 | 0.9839 | 1/5 |
| 8 | 1024 | TwoDDynamic | 4 | 675.314 | 0.9344 | 0/5 |
| 10 | 1024 | RowPanel | 0 | 346.559 | 1.0000 | — |
| 10 | 1024 | TwoDDynamic | 2 | 610.148 | **1.7606** | 4/5 |
| 10 | 1024 | TwoDDynamic | 4 | 583.863 | **1.6847** | 3/5 |

T=10 では `TwoDDynamic` が jpw=2・jpw=4 とも対 `RowPanel` 比 ≥1.00 であり、§4.1 条件 4 の
REJECT トリガー（「DGX N=1024・T=10 で jpw 2・4 のいずれも対 `RowPanel` 比 < 1.00」）は
**発火しない**。

T=8 では逆に `TwoDDynamic` が `RowPanel` をわずかに下回る（0.93〜0.98 倍・勝ち run 数
0〜1/5）。これは `RowPanel` 自体が T=8 で既に高速（722.691 GFLOP/s。T=10 の 346.559 の
約 2.1 倍）という #1305 の非単調性そのものが分母側にあるためで、`TwoDDynamic` 単体の絶対
性能（T=8: jpw2=711.040、T=10: jpw2=610.148）は T=8→T=10 でおよそ 14% しか落ちておらず、
`RowPanel` の 52% 落ち込みに比べて遥かに滑らかである（下記 AC-2 参照）。

### AC-2: 非単調性の解消／残存判定（DGX・N=1024）

| candidate | jobs_per_worker | T=10 median | T=8 median | T10/T8 比 | 判定（事前宣言基準） |
|---|---|---|---|---|---|
| RowPanel | 0 | 346.559 | 722.691 | 0.4795 | （対照。#1305 §17.3 の 0.5094 とオーダー一致・同ハーネスでの再現を確認） |
| TwoDDynamic | 2 | 610.148 | 711.040 | 0.8581 | **残存**（0.90 未満） |
| TwoDDynamic | 4 | 583.863 | 675.314 | 0.8646 | **残存**（0.90 未満） |

事前宣言した閾値（≥1.00 で解消・<0.90 で残存・その間で部分的緩和）に機械的に適用すると、
T=10/T=8 軸では **「残存」** と判定される。ただし比の絶対値は `RowPanel` の 0.48 から
0.86 前後へ大きく改善しており、閾値のすぐ外側（0.90 未満だが 0.86 と近接）である点は
注記する。

参考: DGX T=20（既定・全コア）対 T=8 の比（`TwoDDynamic` が全コア使用時の劣化〈#1305 §17.5〉
をどう変えるか）:

| candidate | jobs_per_worker | T=20 median | T=8 median | T20/T8 比 |
|---|---|---|---|---|
| RowPanel | 0 | 543.901 | 722.691 | 0.7527 |
| TwoDDynamic | 2 | 713.834 | 711.040 | **1.0039** |
| TwoDDynamic | 4 | 693.188 | 675.314 | **1.0265** |

全コア（T=20）対 T=8 の軸では `TwoDDynamic` はほぼ等速〜わずかに上回る（比 1.00〜1.03）のに
対し、`RowPanel` は T=8 比で依然 25% 落ち込む（0.75）。**全コア使用時の劣化（#1305 §17.5 の
軸）は `TwoDDynamic` でほぼ解消しているが、T=10 という中間スレッド数での非単調性
（#1305 §17.3 の軸）は「残存」（閾値未達だが改善方向）という、軸によって異なる結論になる。**

### Tier 2（参考。対 gemm crate。結線判定には使わない）

`oss-gemm-compare --sizes 1024,2048,4096` 5 回独立プロセス中央値。出典:
`docs/perf/logs/cpu-gemm-2d-dynamic-ab-1312/oss-{dgx,m4max}-run{1..5}.jsonl`。

| 実機 | size | gemm crate (TFLOP/s) | RowPanel (TFLOP/s) | RowPanel/gemm |
|---|---|---|---|---|
| DGX | 1024 | 0.5958 | 0.5351 | 0.898 |
| DGX | 2048 | 0.6879 | 0.6928 | 1.007 |
| DGX | 4096 | 0.7672 | 1.0937 | 1.426 |
| M4 Max | 1024 | 0.4553 | 0.1961 | 0.431 |
| M4 Max | 2048 | 0.4304 | 0.4311 | 1.002 |
| M4 Max | 4096 | 0.6010 | 0.5763 | 0.959 |

（`self_gemm_blis_parallel` の `impl_threads` は実機既定値〈DGX 20・M4 Max 16〉。
`matrixmultiply` は `output_match=false` の既知限界〈K≥1024 の丸め差。
`docs/oss-comparison-harness-decision.md`〉があるため参照に含めない。**`gemm` crate 側にも
同様の留保がある**: 上表の DGX `2048`・`4096` 行と M4 Max `4096` 行は、同梱 JSONL
（`oss-{dgx,m4max}-run{1..5}.jsonl`）で該当 size の 5 run すべてが `output_match=false`
（DGX 1024 と M4 Max 1024／2048 のみ `output_match=true`）であり、これらの行の
「TFLOP/s」「RowPanel/gemm」は出力不一致下での参考値に留まる（Tier 2 が採否判定に
使わない参考値であるという既存の位置づけに変わりはないが、行単位でどこまで信頼できるかを
明示するための追記）。M4 Max は共有負荷下で
`--sizes 1024,2048,4096` 一括実行が 2 分タイムアウトへ到達した run が複数あったため、
run 1・run 3 はサイズ別（1024,2048 → 4096 を別プロセスで追加実行）に分割して完走させた
〈`oss-m4max-run{1,3}.err` 参照〉。各 (impl, size) の 5 サンプルはいずれも 5 個の独立
プロセス実行から得ており、中央値の算出方法自体は他の run と変わらない。）

`TwoDDynamic` の GFLOP/s（AB ハーネス実測値・上記主表）を同じ gemm crate 分母で換算した
参考比（`oss-gemm-compare` 自体は variant 選択非対応のためこの換算による推定に留まる）:

| 実機 | size | jobs_per_worker | TwoDDynamic/gemm（推定） | Tier 2 ゲート（1024/2048≥1.00 かつ 4096 非劣化） |
|---|---|---|---|---|
| DGX | 1024 | 2 | 1.198 | 満たす |
| DGX | 1024 | 4 | 1.163 | 満たす |
| DGX | 2048 | 2 | 1.868 | 満たす |
| DGX | 2048 | 4 | 1.704 | 満たす |
| DGX | 4096 | 2 | 1.583 | 非劣化（RowPanel比 1.426 を上回る） |
| DGX | 4096 | 4 | 1.695 | 非劣化 |
| M4 Max | 1024 | 2 | 1.027 | 満たす |
| M4 Max | 1024 | 4 | 0.984 | **満たさない**（<1.00） |
| M4 Max | 2048 | 2 | 1.186 | 満たす |
| M4 Max | 2048 | 4 | 1.163 | 満たす |
| M4 Max | 4096 | 2 | 0.660 | RowPanel比 0.679 をわずかに下回る（劣化） |
| M4 Max | 4096 | 4 | 0.704 | 非劣化（RowPanel比 0.679 を上回る） |

（Tier 2 は §4.2 のとおり参考値であり、本イシューの ADOPT／REJECT／undetermined 判定には
使わない。M4 Max が §4.1 条件 7 の専有ゲートを通過していない点は Tier 2 にも同様に適用される
留保である。）

## 判定

### Tier 1 条件の機械的適用

1. **条件 5（前提）**: 両実機で `gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_large`
   pass（`bit-exact-large-{dgx,m4max}.txt`）・`cargo test --lib gemm_blis --release` green
   （`unit-test-{dgx,m4max}.txt`）を確認済み
2. **条件 4（中止条件）**: DGX N=1024・T=10 で jpw=2（1.7606）・jpw=4（1.6847）とも
   対 `RowPanel` 比 ≥1.00 のため **発火しない**
3. **条件 1・3（数値のみ）**: DGX は jpw=2・jpw=4 とも全形状で満たす（勝ち run 数 5/5）。
   M4 Max も数値上は jpw=2・jpw=4 とも満たす（N=4096・jpw=2 のみ勝ち run 数 3/5 でノイズ
   ガード閾値ちょうど）
4. **条件 7（M4 Max 専有性）**: **不通過**。1 分 load average が計測全体を通じて概ね
   30〜60（閾値 6 の 5〜10 倍）で推移しており、他セッション並走による共有負荷であることが
   明らか（`gate-m4max.log`）。条件 7 は「ADOPT／条件付き ADOPT は M4 Max のゲート通過が
   必須（不通過なら undetermined として #1313 へ「結線せず記録のみ」を引き継ぐ）」と
   計測前に明記しており、DGX 単独の良好な結果や M4 Max の数値そのものが良好に見えることを
   もって ADOPT へ倒さない

### 最終判定: **undetermined（条件付き採用にも至らない）**

- **REJECT ではない**: DGX（専有ゲート通過）の結果は jpw=2・jpw=4 とも全形状・全 run で
  `RowPanel` を上回り（勝ち run 数 5/5）、条件 4 の中止条件も発火していない
- **ADOPT でも条件付き ADOPT でもない**: 条件 7（M4 Max 専有性）が計測前の契約として
  ADOPT 系列の必須条件としているため、M4 Max 側データが数値上良好であっても未確定として
  扱う。条件付き採用（§4.1 条件 6）についても同じ理由で確定できない（条件 6 も「両実機で」
  述語充足を要求し、M4 Max 側が専有ゲート未通過のままでは判定を確定できない）
- 本イシューの結論としては **「#1313 へは『結線せず記録のみ』として引き継ぐ」**（設計
  `docs/cpu-gemm-2d-dynamic-partition-design.md` §11「M4 Max が計測中に専有できない場合の
  扱い」がこの規則を事前に明記済み）

### #1313 への引き継ぎ

- 判定: undetermined。本番結線（`gemm_blis_parallel`／`gemm_blis_bias_act_parallel` への
  `TwoDDynamic` 分岐追加）は **実施しない**
- 参考情報として記録する数値的シグナル（結線判断の根拠にはしないが、再計測時の出発点として
  有用）:
  - DGX（専有ゲート通過）は jpw=2・jpw=4 とも全形状で `RowPanel` を明確に上回る
    （1.09〜1.80 倍）
  - AC-2（DGX・N=1024・T10/T8 非単調性）は「残存」判定だが、`RowPanel` の 0.48 から
    `TwoDDynamic` の 0.86 前後へ大幅に改善（閾値 0.90 のすぐ外側）
  - DGX の全コア軸（T20/T8）では `TwoDDynamic` がほぼ解消（比 1.00〜1.03）する一方、
    `RowPanel` は 0.75 のまま
  - M4 Max は専有ゲート未通過のため確証はないが、共有負荷下でも jpw=2・jpw=4 とも大半の
    形状で `RowPanel` を上回る方向性を示した（N=4096・jpw=2 のみ僅かに下回る 3/5）
- 再計測を行う場合の推奨: M4 Max の専有ゲート（1 分 load average < 6 を 2 回連続）が
  実際に通過できるタイミングで `gemm_blis_two_d_dynamic_ab_1024_2048`／`_4096` を
  再実行し、本ドキュメントの Tier 1 表と同じ形式で追記する。DGX 側の再計測は本イシューの
  結果を正として扱ってよい（専有ゲート通過済み・条件 4 の中止条件も発火していない）
- `jobs_per_worker` の候補: 数値上は jpw=2・jpw=4 のいずれも大差なく良好（DGX は jpw=2 が
  N=1024/2048 でやや優勢・jpw=4 が N=4096 でやや優勢。M4 Max も同様の傾向）。再計測で ADOPT
  と判定された場合、既存の `TWO_D_JOBS_PER_WORKER`（#1311 導入時点の既定値 2）をそのまま
  採用するか、N=4096 での優位性を踏まえて jpw=4 を検討するかは #1313 の裁量とする

## #1313 追記（Phase 0 再計測・ADOPT 確定・本番結線）

### Phase 0: Apple M4 Max 専有ゲート付き再計測（2026-09-08）

計画に基づき、M4 Max 専有ゲート（1 分 load average < 6.0 を 2 回連続・最大 30 分待機）を
`setsid nohup` で切り離したオーケストレーションスクリプトで 1 回だけ試みた
（`docs/perf/logs/cpu-gemm-2d-dynamic-wiring-1313/m4max_orchestrate.sh`）。attempt=8（開始
から約 8 分後）で load average 3.40 を記録しゲート **通過**（`gate-m4max.log`）。前提 2 件
（`cargo test -p fandhe-ai-backend-cpu --lib gemm_blis --release` green・
`gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_large`〈`--ignored`〉pass）を確認後、
`gemm_blis_two_d_dynamic_ab_1024_2048`／`_4096`（T=default=16）を実行した。

| candidate | jobs_per_worker | N=1024 比 | N=2048 比 | N=4096 比 | ノイズガード（勝ち run） |
|---|---|---|---|---|---|
| TwoDDynamic | 2 | 1.2327 | 1.3088 | 1.0261 | 5/5・5/5・4/5 |
| TwoDDynamic | 4 | 1.2261 | 1.3120 | 1.0917 | 5/5・5/5・5/5 |

（`docs/perf/logs/cpu-gemm-2d-dynamic-wiring-1313/aggregate.md` 参照）

Tier 1 条件（本ファイル §「事前宣言ゲート」・#1312 の文言をそのまま転記した
`docs/perf/logs/cpu-gemm-2d-dynamic-wiring-1313/env_info.txt` を正とする）を機械的に適用:

1. **条件 1**（N=1024/2048 で比 >=1.00・N=4096 で比 >=0.95）: jpw=2・jpw=4 とも **充足**
2. **条件 2**（単一固定 jpw が両実機で条件 1 を充足）: DGX（#1312）は jpw=2・jpw=4 とも
   全形状で `RowPanel` を上回り条件 1 を満たす。M4 Max（本計測）も jpw=2・jpw=4 とも条件 1
   を満たす。よって**両実機で条件 2 が jpw=2・jpw=4 の両方について成立**
3. **条件 3**（ノイズガード。勝ち run >=3/5）: jpw=2 は N=4096 で 4/5・他は 5/5 で充足。
   jpw=4 は全形状 5/5 で充足
4. **条件 4**（中止条件。DGX N=1024・T=10）: #1312 で非発火を確認済み（再計測しない。本
   ファイル §「#1313 への引き継ぎ」参照）
5. **条件 5**（前提）: 両実機で充足（M4 Max は本計測・DGX は #1312）
6. **条件 7**（M4 Max 専有性）: **通過**（本計測。#1312 とは異なり成立）

**最終判定: ADOPT 確定**。既存の `TWO_D_JOBS_PER_WORKER`（#1311 導入時点の既定値 2）は
jpw=2 が条件 1〜3 を単独で満たすため変更不要と判断し、そのまま採用した（計画「Phase 0 で
jpw=2 が条件 1 を満たさず jpw=4 のみ満たす場合に限り 4 へ変更する」の分岐は非該当）。

### 本番結線

`crates/backend-cpu/src/gemm_blis/mod.rs` に単一 const ゲート
`TWO_D_DYNAMIC_PRODUCTION_ENABLED = true` を追加し、`gemm_blis_parallel_with_transpose`
（`gemm_blis_parallel`／`_nt`／`_tn` の共通実装本体）・`gemm_blis_bias_act_parallel` から
`dispatch_two_d_dynamic` を呼ぶよう分岐した（`false` の場合は #1313 以前と同一の静的行
パネル分割 `par_chunks_mut` へ戻る。`thread_limit::BIG_CORE_LIMIT_ENABLED` と同型の設計）。
`partition` モジュール・`TwoDJob`／`split_c_into_jobs`／`run_two_d_job`／
`gemm_blis_two_d_dynamic_region`／`dispatch_two_d_dynamic`（3 arch 版）／
`TWO_D_JOBS_PER_WORKER` の `#[cfg(test)]` を解除し本番到達可能にした。`partition.rs` 内の
`split_evenly`／`row_ranges_for_workers`（#753 の `gemm_blis_parallel_2d_with_blocks` 専用
ヘルパー）は個別に `#[cfg(test)]` を付与し維持した。

回帰テスト（`cargo test -p fandhe-ai-backend-cpu --lib`。289 件 pass・0 fail）・統合テスト
（`gemm_blis_parity`・`gemm_epilogue_parity`・`gemm_transposed_parity`。既存の「本番入口
`gemm_blis_parallel`／`gemm_blis_bias_act_parallel` vs `gemm_naive` bit 完全一致」回帰が
`TwoDDynamic` 経路を自動的に検証する構成のため新規テスト追加は不要と判断した）・
`cargo clippy -p fandhe-ai-backend-cpu --all-targets -- -D warnings`（0 error）・
`cargo fmt --all -- --check`（差分なし）はすべて通過を確認した。

### framework-compare gemm cpu before/after（両実機）

`compare_gemm_ab.py --device cpu`（README「`compare_gemm_ab.py --device cpu`」節の手順。
before = 結線前の origin/main HEAD `fddca17`〈`GEMM_GATE_PATCH_FACADE_PATH` 経由の path
patch〉・after = 本ブランチ HEAD）を N=512/1024/2048 × fresh/reuse の全 6 セルで実行した。

Apple M4 Max（`docs/perf/logs/cpu-gemm-2d-dynamic-wiring-1313/framework-compare/compare_gemm_ab-m4max-1313.md`）:

| size/mode | after/before | checksum | 判定 |
|---|---|---|---|
| 512/fresh | 0.8954 | 完全一致 | 非後退 |
| 512/reuse | 0.8789 | 完全一致 | 非後退 |
| 1024/fresh | 0.8792 | 完全一致 | 非後退 |
| 1024/reuse | 0.8593 | 完全一致 | 非後退 |
| 2048/fresh | 0.8575 | 完全一致 | 非後退 |
| 2048/reuse | 0.8385 | 完全一致 | 非後退 |

DGX Spark GB10（`docs/perf/logs/cpu-gemm-2d-dynamic-wiring-1313/framework-compare/compare_gemm_ab-dgx-1313.md`）:

| size/mode | after/before | checksum | 判定 |
|---|---|---|---|
| 512/fresh | 0.9643 | 完全一致 | 非後退 |
| 512/reuse | 0.9003 | 完全一致 | 非後退 |
| 1024/fresh | 0.8142 | 完全一致 | 非後退 |
| 1024/reuse | 0.7603 | 完全一致 | 非後退 |
| 2048/fresh | 0.6504 | 完全一致 | 非後退 |
| 2048/reuse | 0.6015 | 完全一致 | 非後退 |

両実機・全 12 セルが非後退（ratio 0.60〜0.96。すべて改善方向）・checksum 完全一致・
`parity_fail_count=0` を確認した。決定規則（#1364 と同一。両実機・全判定可能 reuse セルで
`ratio <= 1.05` かつ checksum 完全一致）を満たすため **ADOPT を確定**し、
`TWO_D_DYNAMIC_PRODUCTION_ENABLED = true` を維持する（差し戻しは不要）。

### 実行ログ

`docs/perf/logs/cpu-gemm-2d-dynamic-wiring-1313/`（Phase 0 の gate/uptime/aggregate・
env_info）・`docs/perf/logs/cpu-gemm-2d-dynamic-wiring-1313/framework-compare/`
（両実機の JSONL・manifest・compare 結果）。DGX 側は本イシュー専用の隔離ディレクトリ
（`~/work/fc-1313/`）で作業し、計測後に削除した。内部ホスト名は記録していない。

## スコープ外

- `oss-gemm-compare` の variant 選択オプション追加
- `jobs_per_worker=8` のスイープ（計画で対象外と明記）
- x86_64 実機
- M4 Max T=12（P コア数限定）の追加スレッド軸（時間制約により未実施）
- DGX T=8 での `TwoDDynamic` 後退（0.93〜0.98 倍）の根本原因診断（T=1 staging 単離診断は
  REJECT 確定時のみ実施する契約〈計画 §4.4〉であり、本イシューは undetermined 確定のため
  未実施のまま記録する）
- `#1313` へのコメント投稿（本イシューの引き継ぎ内容は本ドキュメントに記録済み。GitHub
  への書き込みは本エージェントの権限外のため実施しない）

## 出典

イシュー #1312・#1311/#1307/#1305/#1303/#1283・`docs/cpu-gemm-2d-dynamic-partition-design.md`
§9/§11・`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §17・
`crates/backend-cpu/src/gemm_blis/mod.rs`（`AbCandidate`・`gemm_blis_two_d_dynamic_ab_1024_2048`・
`gemm_blis_two_d_dynamic_ab_4096`。コミット `f719012`）・
`docs/perf/logs/cpu-gemm-2d-dynamic-ab-1312/`（実行ログ・`aggregate.py`・`aggregate.md`・
`env_info*`）
