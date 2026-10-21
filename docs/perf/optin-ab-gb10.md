# GB10 の未計測 opt-in 候補 A/B（イシュー #3190）

状態: **B1〜B4・B6 は GB10 で実測済み（2026-10-20）。B5（境界述語の事前計算）は未実施**。
main の定数・環境変数既定は変更していない（`crates/` 差分なし）。規則は `logs/optin-ab-gb10-3190/RULE.txt`（計測前に単独コミット `230937f5f` で固定。
継承 RULE 6 件は 1 字も変えず sha256 を記録）。判定は継承（または補遺）RULE の文言を機械的に当てはめた。兄弟 #3184（Mac mini 版）と共通のゲートは B1=G1・B2=G2。

時刻の注記: ログのタイムスタンプは UTC、`RULE.txt` の `fixed-at: 2026-10-21` は JST（日付表記のみ）。規則を固定したコミット `230937f5f` の作者日時は 2026-10-20T15:20:29Z で、最初の計測ログ（B1 の負荷ゲート 15:22:50Z）より前。

## 条件差

| 項目 | 継承 RULE の想定 | 本計測 |
|---|---|---|
| 機体 | GB10（20 コア・sm_121・CUDA 13.0） | 同じ |
| ツリー | git 付き | `git bundle` で送り、before／after／runner の 3 clone。HEAD `d89a09370`（`origin/main` `42107efc1` + 補遺規則 + A/B 基盤の追加のみ） |
| B4 の registry 腕 | `fandhe-ai =0.9.0` と記載 | framework-compare の現行ピン `=0.11.0`（record_only のため判定不変） |
| 負荷 | 継承登録値（CPU: load1 < 8.0／CUDA: 専有 load1 < 1.0 かつ GPU 0%） | 同じ。GB10 用の外側ゲートは追加せず |

## ゲート別判定

| ID | 対象 | 継承 RULE | GB10 判定 | Mac mini（#3184） | 総合 |
|---|---|---|---|---|---|
| B1 | `HOST_ARENA_DEFAULT_ENABLED` | 2104 | **undetermined** | 未実測 | undetermined |
| B2 | `REDUCTION_SEQUENTIAL_FALLBACK_ENABLED` | 2101→2102 | **undetermined**（T_gb10 = 16384・Stage 0 PROCEED） | 未実測 | undetermined |
| B3 | 推論 Graph capture | 2115 | **undetermined**（性能は全セル改善・R1(b) 失敗） | 対象外 | undetermined |
| B4 | N=256 起動固定費の内訳 | 2109 | 系列は**無効**（`aggregate.py` exit 1）・参考値あり | 対象外 | 診断 |
| B5 | 境界述語の事前計算（MMA_F16） | 補遺 8 | **未実施** | 対象外 | undetermined |
| B6 | readback `pinned-reuse` の N=256 | 補遺 9（2108 の文言） | **REJECT**（ratio 1.0475・5/5 後退） | 対象外 | REJECT |

ADOPT になったゲートはない。従って #3227・#3205 が反転できる定数は現時点でない。

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

### B2 縮約逐次化（2101→2102）

- Stage 0（GB10 単機・5 プロセス）: **T_gb10 = 16384**。checksum は全セル一致。T 未満の max r = 1.0000。違反点は sum_axis0/1 = 16384、sum_all/mean_all = 65536、max_all = 131072。
  利得（5/5 run で r < 1.00・T 未満）は 12 セルで成立。T_gb10 > 640 のため Stage 1 へ進んだ。集計は `aggregate_sweep_single.py`（継承の関数を import して 1 機体で評価）。
  出典: `logs/elemental-reduction-threshold-2101/gb10/aggregate.md`。
- Stage 1（T = 16384・`on-reduction-gb10.patch`・`2102-gb10-3190`）: 12 セルのうち gemm 2048/fresh（1.0192）と infer reuse（1.0049）の 2 セルが ratio > 1.00、いずれも 5/5 一貫ではない。
  他は ratio <= 1.00（train は 2 セルとも 0.79〜0.99）で checksum は全て完全一致。正式系列（5/5 round が gate_ok=1）。**GB10 機体判定は undetermined**。
  gemm 2048/fresh の before は spread が大きく（15.9〜16.4 ms、2048/reuse は 19.9〜27.6 ms）、負荷ノイズの疑いがある（compare 表の注記）。T の再選定や再判定はしない（RULE）。
- 最終 T は Mac mini の Stage 0 が出てから min(T_macmini, 16384)。16384 より小さくなれば Stage 1 は #3227 側で測り直す。MSE backward は適用外。

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
- 示唆: capture 競合の扱い（待機またはフォールバック）を直せば、性能面は ADOPT 条件 1〜4 を満たしている。直したうえでの再判定は別 label・新規事前登録で行う。
- 作業ミスの開示: 第 1 系列（`2115-gb10-3190`）は環境変数設定の不具合で計測ゼロのまま失敗（結果未確認・`invalid-attempt1-...` に保存）、第 2 系列（`3190b`）の
  compare 表・JSONL を作業ツリーの掃除で消してしまい、実行時の標準出力ログから無加工で再構成した。以下の B6 も同様。

### B4 N=256 起動固定費の内訳（2109）

`docs/perf/cuda-gemm-small-launch-cost.md` §7。`aggregate.py` が Layer B ログの先頭行融合（`test ... ... DIAG2109 ...`）で exit 1 となったため RULE 7 により系列は無効。
参考値（正規化コピー）: reuse 90.1 us・candle fresh 75.9 us・gap 14.2 us、支配項は readback（24.1 us）と h2d_a／h2d_b（各 約 11〜12 us）、
launch 往復の床は 6.1 us。H1・H3（転送）が仮説の向きに大きい（RULE 判定ではない参考）。重要な含意: N=256 の差は launch 往復より転送が支配的で、B6 の REJECT（readback 宛先を変えても縮まらない）と整合する。
#3214・#3259 の前提として、入出力の resident 化（計測境界の対称化）を先に検討する。

### B5 境界述語の事前計算（#537）

未実施（undetermined）。補遺 RULE 8 の打ち切り条件に従い、`on-boundary-predicate.patch` を作っていないため前提ゲート・TFLOPS 計測とも行っていない。
`MMA_F16` は f16 経路で f32 の N=256 には効かず、`kernels_mma.rs` は #537 の記述時から `LOAD_A_STAGE_GROUP` 方式へ変わっている（行番号が古い）。推定で採否を書かない。

### B6 readback `pinned-reuse` の N=256

`docs/perf/cuda-gemm-readback-reuse-2108.md` §5.1。ratio 1.0475（5 run 内比 1.0724・1.0500・1.0498・1.0472・1.0724 の 5/5 後退）・checksum 完全一致・専有ゲート成立 → **REJECT**。
標準出力に他プロセス検出の警告が出たが、専有ゲート（load1 2.92 から待機し 0.39 で 3 連続通過・GPU 0%）は成立している（詳細は 2108 doc §5.1）。#3186 の N>=1024（1.0982〜1.1254）と合わせ、全 N で pinned-reuse は後退した。

## 申し送り

- #3227: ADOPT 条件が成立したゲートはない。B1（train fresh に利得あり・reuse は中立〜微悪化）と B2（T_gb10 = 16384）は Mac mini の結果と合わせて判定する。
- #3205: B3 の capture 競合（`DeviceContextCaptureInProgress`）の是正が先。性能は改善方向（reuse 0.86〜0.98）。
- #3214・#3259: B4 の参考値（転送 H1+H3 が L0 の約 9 割。gap への寄与は未分離）を前提に使う。ただし系列は無効なので、必要なら再計測する。
- 対象外にしたもの（起票は承認後）: 2113 の GB10 対照セル（#3184 の G3）、継承スクリプトの負荷閾値の機体別化、2109 の registry 腕ラベルの現行版への更新、
  `aggregate_sweep.py` の単機モード、`aggregate.py`（2109）の先頭行融合対応、B5 の実装と計測、B3 の capture 競合の是正、2101 RULE が求める CUDA reduce 系 `--ignored` parity の同セッション実施（未実施）。
