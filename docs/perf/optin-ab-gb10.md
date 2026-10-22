# GB10 の未計測 opt-in 候補 A/B（イシュー #3190）

状態: **B1〜B4・B6 は 2026-10-20、B5 と B4 の有効系列は再開分として 2026-10-21 に GB10 で実測済み**（B3 の再計測は #3205 へ切り出し）。
B1・B2 は #3184（Mac mini）の実測と機体別に併記した（下記）。B4 は有効系列（`gb10-r2`）で集計し、B5 は**不採用**（4 形状すべて 5/5 後退）。
main の定数・環境変数既定は変更していない（`crates/` 差分なし）。規則は `logs/optin-ab-gb10-3190/RULE.txt`（計測前に単独コミット `230937f5f` で固定。
継承 RULE 6 件は 1 字も変えず sha256 を記録）。判定は継承（または補遺）RULE の文言を機械的に当てはめた。兄弟 #3184（Mac mini 版）と共通のゲートは B1=G1・B2=G2。

時刻の注記: ログのタイムスタンプは UTC、`RULE.txt` の `fixed-at: 2026-10-21` は JST（日付表記のみ）。規則を固定したコミット `230937f5f` の作者日時は 2026-10-20T15:20:29Z で、最初の計測ログ（B1 の負荷ゲート 15:22:50Z）より前。
再開分の事前登録（`RULE-reopen.txt`・`aggregate.py` の融合 1 形の受理・`aggregate_b5.py`・`on-boundary-predicate.patch`）はコミット `fbf6ecbbe`（作者日時 2026-10-21T13:43:24Z）で固定し、
GB10 上の最初のログ（`preflight-procs.txt` 13:44:22Z）より前。(a) のコマンド訂正 2 件（`RULE-reopen-2.txt`・`-3.txt`）は、それぞれ再実行の前にコミットした（(a) の再実行は 13:56Z・13:58Z、TFLOPS 計測は 13:59Z 以降）。

## 条件差

| 項目 | 継承 RULE の想定 | 本計測 |
|---|---|---|
| 機体 | GB10（20 コア・sm_121・CUDA 13.0） | 同じ |
| ツリー | git 付き | `git bundle` で送り、before／after／runner の 3 clone。HEAD `d89a09370`（`origin/main` `42107efc1` + 補遺規則 + A/B 基盤の追加のみ）。再開分は runner／b5-base（`230937f5f`）／b5-head（base + patch）の 3 clone で、`b5/rev-stamp.txt` に記録 |
| B4 の registry 腕 | `fandhe-ai =0.9.0` と記載 | framework-compare の現行ピン `=0.11.0`（record_only のため判定不変） |
| 負荷 | 継承登録値（CPU: load1 < 8.0／CUDA: 専有 load1 < 1.0 かつ GPU 0%） | 同じ。GB10 用の外側ゲートは追加せず |

## ゲート別判定

| ID | 対象 | 継承 RULE | GB10 判定 | Mac mini（#3184） | 総合 |
|---|---|---|---|---|---|
| B1（= #3184 の G1） | `HOST_ARENA_DEFAULT_ENABLED` | 2104 | **undetermined** | **REJECT**（#3184 再開分 `3184-macmini-r2-g1`。正式系列で train reuse 1.0090・5/5 後退。初回 3 系列は record_only） | **REJECT**（2104 は「いずれかの機体で 5/5 一貫の後退セルがあれば REJECT」） |
| B2（= #3184 の G2） | `REDUCTION_SEQUENTIAL_FALLBACK_ENABLED` | 2101→2102 | **undetermined**（T_gb10 = 16384・Stage 0 PROCEED） | Stage 0: T_macmini = 16384（PROCEED）／Stage 1: undetermined（record_only） | undetermined。最終 T = min(16384, 16384) = 16384 |
| B3 | 推論 Graph capture | 2115 | **undetermined**（性能は全セル改善・R1(b) 失敗）。**再計測は #3205 へ切り出し** | 対象外 | undetermined |
| B4 | N=256 起動固定費の内訳 | 2109 | **有効系列（`gb10-r2`・`aggregate.py` exit 0）**: H1・H3 支持、H2・H4・H5 未確定、支配項は readback | 対象外 | 診断（採否ではない） |
| B5 | 境界述語の事前計算（MMA_F16） | 補遺 8 | **不採用**（前提ゲート (a)(b) 成立・4 形状すべて 5/5 後退） | 対象外 | 不採用（#537 へ） |
| B6 | readback `pinned-reuse` の N=256 | 補遺 9（2108 の文言） | **REJECT**（ratio 1.0475・5/5 後退） | 対象外 | REJECT |

ADOPT になったゲートはない。従って #3227・#3205 が反転できる定数は現時点でない。
総合判定は、2104／2102 の 2 機体規則を機械的に当てはめた（ADOPT には両機体の成立が要る。REJECT には、どちらかの機体の**正式系列**で 5/5 round 一貫の後退が要る。
#3184 の初回 Mac mini 系列は補遺 RULE 3 の負荷条件を満たさず record_only だったため根拠にしなかったが、再開分〈事前登録コミット `c80d9818`・round 前クールダウンで全 round の load1 < 3.0〉で正式系列が成立した）。
再開分の Mac mini 正式系列で B1 の train reuse が 5/5 一貫の後退（1.0090）となったため、B1 の総合は REJECT に更新した（GB10 は 5/5 一貫の後退がなく undetermined のまま）。B2 は Mac mini も undetermined のまま。
数値の出典: GB10 は本書下記、Mac mini は `optin-ab-macmini.md`（#3184。初回は PR #3509、再開分は「再開分の結果」節）。

### B1 host arena（2104）

HEAD の `HOST_ARENA_DEFAULT_ENABLED` だけを `true` にした after（`on-host-arena.patch`）。`2104-gb10-3190`、正式系列（5/5 round が load1 < 8.0）。

| セル | before 中央値 | after 中央値 | ratio | checksum |
|---|---|---|---|---|
| train fresh | 1.118 ms | 967.3 us | 0.8648 | 完全一致 |
| train reuse | 828.1 us | 842.4 us | 1.0172 | 完全一致 |
| infer fresh | 178.2 us | 178.1 us | 0.9995 | 完全一致 |
| infer reuse | 182.8 us | 183.4 us | 1.0035 | 完全一致 |

全セル非後退でない（reuse 2 セルが ratio > 1.00）ため ADOPT 条件を満たさず、5/5 round 一貫した後退（全 round ratio > 1.00）もないため REJECT でもない。
train fresh は 0.82〜0.91 で一貫して改善した（fresh 側にだけ効く arena の想定どおり）。reuse は arena が既に確保を済ませているため効かない。
出典: `logs/tape-arena-reuse-2104/compare-{train,infer}-2104-cpu-2104-gb10-3190.md`。

**機体別の併記（ratio = after/before の 5 round 中央値）**。Mac mini は #3184 の 3 系列（g1＝初回・g1w＝ウォーム・g1f＝再実行。いずれも record_only）で、系列間で符号が揃わない。

| セル | GB10（正式系列） | Mac mini g1 | Mac mini g1w | Mac mini g1f |
|---|---|---|---|---|
| train fresh | 0.8648 | 1.0115 | 1.0291（5/5 後退） | 1.0057 |
| train reuse | 1.0172 | 0.9887 | 0.9934 | 1.0067 |
| infer fresh | 0.9995 | 0.9993 | 1.0007 | 1.0275 |
| infer reuse | 1.0035 | 1.0022 | 1.0090 | 0.9671 |

両機体とも ADOPT 条件（全セル非後退）を満たさない。REJECT の根拠になりうる 5/5 一貫の後退は、Mac mini g1w の train fresh のみ（record_only）。GB10 の train fresh は 0.8648 と改善、Mac mini は 1.0057〜1.0291 と悪化方向で
**機体間で符号が割れている**（GB10 は 20 コア、Mac mini は P4+E6 で arena 確保の相対コストが違う可能性。推定であり本計測では切り分けていない）。

再開分（#3184。正式系列 `3184-macmini-r2-g1`）の Mac mini: train fresh 1.0115・train reuse **1.0090（5/5 後退。1.0153・1.0036・1.0420・1.0025・1.0060）**・infer fresh 0.9961・infer reuse 1.0181。上の表の g1／g1w／g1f は record_only の旧系列。

### B2 縮約逐次化（2101→2102）

- Stage 0（GB10 単機・5 プロセス）: **T_gb10 = 16384**。checksum は全セル一致。T 未満の max r = 1.0000。違反点は sum_axis0/1 = 16384、sum_all/mean_all = 65536、max_all = 131072。
  利得（5/5 run で r < 1.00・T 未満）は 12 セルで成立。T_gb10 > 640 のため Stage 1 へ進んだ。集計は `aggregate_sweep_single.py`（継承の関数を import して 1 機体で評価）。
  出典: `logs/elemental-reduction-threshold-2101/gb10/aggregate.md`。
- Stage 1（T = 16384・`on-reduction-gb10.patch`・`2102-gb10-3190`）: 10 セルのうち gemm 2048/fresh（1.0192）と infer reuse（1.0049）の 2 セルが ratio > 1.00、いずれも 5/5 一貫ではない。
  他は ratio <= 1.00（train は 2 セルとも 0.79〜0.99）で checksum は全て完全一致。正式系列（5/5 round が gate_ok=1）。**GB10 機体判定は undetermined**。
  gemm 2048/fresh の before は spread が大きく（15.9〜16.4 ms、2048/reuse は 19.9〜27.6 ms）、負荷ノイズの疑いがある（compare 表の注記）。T の再選定や再判定はしない（RULE）。
- 最終 T = min(T_macmini, T_gb10) = min(16384, 16384) = **16384**（T_macmini は #3184 の Stage 0。`optin-ab-macmini.md` G2）。GB10 の Stage 1 は同じ T = 16384 で計測済みなので、**再計測は不要**。MSE backward は適用外。

**機体別の併記（Stage 1・T = 16384。ratio = after/before の 5 round 中央値）**。Mac mini は record_only（正式系列なし）。

| セル | GB10 | Mac mini（g2f） |
|---|---|---|
| train 64／fresh | 0.7889 | 0.9997 |
| train 64／reuse | 0.9890 | 0.9933 |
| infer 64／fresh | 0.9600 | 0.9808 |
| infer 64／reuse | 1.0049 | 0.9343 |
| gemm 1024／fresh | 0.8800 | 0.9861 |
| gemm 1024／reuse | 0.9980 | 0.9862 |
| gemm 2048／fresh | 1.0192 | 0.9980 |
| gemm 2048／reuse | 0.9932 | 1.0065 |
| gemm 4096／fresh | 0.9869 | 1.0001 |
| gemm 4096／reuse | 0.9908 | 1.0002 |

（セルは train 2・infer 2・gemm 6 の計 10。GB10 の値は `logs/elemental-reduction-ab-2102/gb10/compare-{train,infer,gemm}-2102-cpu-2102-gb10-3190.md`、Mac mini は `optin-ab-macmini.md` G2。）
ratio > 1.00 のセルは GB10 が 2（gemm 2048/fresh 1.0192・infer 64/reuse 1.0049）、Mac mini が 3（gemm 2048/reuse 1.0065・4096/fresh 1.0001・4096/reuse 1.0002）で、いずれも 5/5 一貫の後退ではない
（GB10 の run 内比は compare 表のとおり全セルで「符号一貫: いいえ」）。両機体とも ADOPT 不成立、REJECT 不成立。

### B3 推論チェーンの Graph capture（2115）

詳細は `docs/perf/infer-chain-graphcapture-cuda-ab.md` §3。要点:

| batch/mode | ratio | 5 run 一貫 < 1.00 |
|---|---|---|
| 64/fresh | 0.9243 | はい |
| 64/reuse | 0.8610 | はい |
| 1024/fresh | 0.9763 | はい |
| 1024/reuse | 0.9270 | はい |
| 4096/fresh | 0.9955 | いいえ |
| 4096/reuse | 0.9761 | はい |

- R0 合格・R1(a) 合格（PASS 5 行）・R2 合格（bit ダンプ完全一致）。**R1(b) 不合格**: 事前登録コマンドでは `predict_device_chain_cuda_bit_identity` が ON で 4 件失敗
  （`DeviceContextCaptureInProgress`。並列テストの別スレッドが capture 中のデバイスを使うとエラーを返す）。bit 不一致ではないので REJECT には当たらず、ADOPT 条件 5 を欠くため undetermined。
  `--test-threads=1` の診断再実行は pass したが判定には使わない。
- 示唆: capture 競合の扱い（待機またはフォールバック）を直せば、性能面は ADOPT 条件 1〜4 を満たしている。
- **再開分の扱い（B3 は再計測しない）**: 是正（capture 競合の待機またはフォールバック）と、その後の R1(b) の別 label・新規事前登録による再判定は **#3205 で行う**（`RULE-reopen.txt` 0）。
  本イシューは B3 の結果を上記のまま undetermined で残す。
- 作業ミスの開示: 第 1 系列（`2115-gb10-3190`）は環境変数設定の不具合で計測ゼロのまま失敗（結果未確認・`invalid-attempt1-...` に保存）、第 2 系列（`3190b`）の
  compare 表・JSONL を作業ツリーの掃除で消してしまい、実行時の標準出力ログから無加工で再構成した。以下の B6 も同様。

### B4 N=256 起動固定費の内訳（2109）

`docs/perf/cuda-gemm-small-launch-cost.md` §7.1（有効系列）。再開分で `aggregate.py` の Layer B 先頭行融合への対応（融合 1 形だけを受理。fail-closed は弱めず、`--self-test` に fixture を追加。
規則は `RULE-reopen.txt` 1）を入れ、空の別ディレクトリ `logs/cuda-gemm-small-launch-cost-2109/gb10-r2/` に `orchestrate.sh` を 1 回だけ実行した。`aggregate.py` は **exit 0**（有効系列）。
専有ゲートは全 30 run PASS。旧 `gb10/` は RULE 7 により無効のまま残してあり、遡って有効にはしていない。

| 項目 | 値（5 run 中央値） |
|---|---|
| HEAD reuse／candle fresh／gap | 90.720／75.088／15.632 µs |
| L0／L1／L2 | 52.737／48.752／52.208 µs |
| 床（host→kernel 往復） | 6.096 µs（L0 の 11.6%） |
| H1 毎反復 H2D | 22.896 µs（L0 比 43.4%・gap 比 146.5%）**支持** |
| H3 D2H readback | 25.056 µs（47.5%・160.3%）**支持** |
| H2・H4a・H5 | 3.948／4.144／3.625 µs（gap 比 25.3%／26.5%／23.2%）未確定 |
| 支配項 | L2 readback 25.408 µs（L0 比 48.2%） |

含意: N=256 の差は launch 往復より転送が支配的（H1+H3 は L0 の 90.9%）。B6 の REJECT（readback 宛先を変えても縮まらない）と整合する。#3214・#3259 の前提として、入出力の resident 化（計測境界の対称化）を先に検討する。
旧無効系列の参考値（gap 14.2 µs・H1 44.8%・H3 46.0%）とも向きが一致した。

### B5 境界述語の事前計算（#537）

`docs/perf/cuda-gemm-boundary-predicate-ab.md`「実測結果」。`on-boundary-predicate.patch`（A/B ロードの行述語とエピローグ述語を入口で 1 回ビット化し、K 方向の述語は stage ごとに AND。REQ-8 の検査は前倒しで、省略しない）を実装し、事前登録どおりに実施した。

- 前提 (a) parity 非後退: **成立**（111 件。base ok → head FAILED は 0 件。`mma_f16_k4096_stress` は base でも FAILED）。コマンドは 2 回訂正した（開示は同 doc）。
- 前提 (b) ptxas -v: **成立**（base 58／head 66 registers・どちらも spill 0・lmem 0・stack 0）。
- TFLOPS（5 プロセス・round 内で順序反転・専有ゲート全 PASS）: head/base は 512 = 0.9519、1024 = 0.8293、2048 = 0.7670、4096 = 0.7294。**4 形状すべて 5/5 後退**。
- 判定: **不採用**。main のカーネルは変えない（#537 は「この形では採らない」で閉じられる）。

### B6 readback `pinned-reuse` の N=256

`docs/perf/cuda-gemm-readback-reuse-2108.md` §5.1。ratio 1.0475（5 run 内比 1.0724・1.0500・1.0498・1.0472・1.0724 の 5/5 後退）・checksum 完全一致・専有ゲート成立 → **REJECT**。
標準出力に他プロセス検出の警告が出たが、専有ゲート（load1 2.92 から待機し 0.39 で 3 連続通過・GPU 0%）は成立している（詳細は 2108 doc §5.1）。#3186 の N>=1024（1.0982〜1.1254）と合わせ、全 N で pinned-reuse は後退した。

## 申し送り

- #3227: ADOPT 条件が成立したゲートはない。B1 は #3184 再開分の Mac mini 正式系列で REJECT（train reuse が 5/5 後退・1.0090。GB10 の train fresh は 0.8648 と改善で機体間の符号は割れる）、B2 は両機体とも undetermined（最終 T = 16384）。B1 の既定は OFF の維持が妥当。
- #3205: B3 の capture 競合（`DeviceContextCaptureInProgress`）の是正と、その後の再判定（別 label・新規事前登録）はこちらで行う。性能は改善方向（reuse 0.86〜0.98）。
- #3214・#3259: B4 の有効系列（転送 H1+H3 が L0 の 90.9%）を前提に使う。入出力の resident 化（計測境界の対称化）を先に検討する。
- #537: B5 の実測で「不採用」。クローズ可能（採らない判断と実測は `cuda-gemm-boundary-predicate-ab.md`）。別案を測るなら新規事前登録で。
- 対象外にしたもの（起票は承認後）: 2113 の GB10 対照セル（#3184 の G3）、継承スクリプトの負荷閾値の機体別化、2109 の registry 腕ラベルの現行版への更新、
  `aggregate_sweep.py` の単機モード、B3 の capture 競合の是正（#3205）、B5 が採用候補になった場合の本番導入（不採用のため発生しない）。
- 訂正（前版の記述）: 前版の申し送りは「2101 RULE が求める CUDA reduce 系 `--ignored` parity の同セッション実施」を未実施事項に挙げていたが、
  2101／2102 の RULE.txt を `parity`・`ignored` で検索しても該当は 2101 の `reduction_threshold_sweep -- --ignored --nocapture`（Stage 0 の計測コマンド。10 行目）の 1 件のみで、
  CUDA reduce 系の parity を同セッションで行う要件の文言は見当たらない。誤記と判断して取り下げた（推定で補わない）。
