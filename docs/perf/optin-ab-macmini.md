# Mac mini の未計測 opt-in 候補 A/B（イシュー #3184）

状態: **実測済み（2026-10-21・Mac mini M4・main `26e60f63`）。ただし補遺 RULE 3 の「全 round で load1 < 3.0」を満たす正式系列は 1 つも成立せず、全ゲートの Mac mini 判定は undetermined（record_only）**。
G6 は継承集計器が入力形式を受理せず INCOMPLETE、G7 は集計器が fail-closed で停止した。値と生ログは全て収録した（`logs/optin-ab-macmini-3184/`）。
main の定数・環境変数既定は変更していない。規則は `logs/optin-ab-macmini-3184/RULE.txt`（計測前に単独コミットで固定。継承 RULE も補遺も計測後に緩めていない）。

## 条件差

| 項目 | 継承 RULE の想定 | 本計測 |
|---|---|---|
| 機体 | M4 Max + GB10 | Mac mini M4（P4+E6、GPU 10 コア、macOS 27.0.1、rustc 1.99.0。Apple Silicon 枠として扱う。`logs/optin-ab-macmini-3184/macmini/uname.txt`・`rev-stamp.txt`） |
| 負荷 | 継承登録値（8.0／exclusive 4.0／record_only） | 継承スクリプトは登録値のまま。外側で系列開始前に load1 < 3.0 を待ち、事後に round 別 load1 < 3.0 を検査（`machine_gate_check.py`） |
| 外側ゲート | — | 全系列で成立（load1 1.4〜2.9。`macmini/machine_gate.log`）。並走プロセスは全系列で 0 件（`macmini/preflight-procs.txt`） |
| 事後検査（round 別 load1 < 3.0） | — | **全系列で不成立**（`macmini/machine_gate_ok.txt`） |
| verified_m4_max_gpu_core_count | 解決済み | G6 の `env_info.txt` に gpu_cores=10 を記録（`mtl_architecture_name` は未取得） |

事後検査が全系列で不成立になった理由: 継承スクリプトは **自身のビルド直後、または直前 round の計測直後** に round 開始の load1 を読む。10 コア全利用の計測（CPU GEMM・縮約・Metal host 側）では load1 の 1 分平均が自分自身の負荷で 5〜8 に上がり、round 間 30 秒待ちでは < 3.0 に戻らない。継承スクリプトの閾値は変えない規則のため、この条件は実測前の運用では満たせなかった。補遺 RULE 3 は「どちらかが欠けた系列は record_only で判定は undetermined」と定めているため、その規則どおりに扱う。

CPU 系 G1／G2 Stage 1 は、初回（label `g1`）が自分のビルド由来の load1 5.65 で不成立だったため、別 label でウォーム走行（`*w`、ビルドだけ済ませる）→ target 改名 → 正式 label（`*f`）の再実行も行った（`session-scripts/cpu_session2.sh`）。`*f` でも round 開始の load1 は 6〜8 で不成立。旧系列は全て残してあり、系列の差し替え・セル除外はしていない。

## ゲート別判定

「継承 RULE のみで読むと」は参考であり判定ではない（補遺 RULE 3 により正式系列でないため）。

| ゲート | 継承 RULE | Mac mini 判定 | 継承 RULE のみで読むと（参考） | GB10（#3190） | 総合 |
|---|---|---|---|---|---|
| G1 `HOST_ARENA_DEFAULT_ENABLED` | 2104 | undetermined（record_only） | ADOPT 不成立。train fresh が ratio > 1.00（g1 1.0115・g1w 1.0291〈5/5 round 後退〉・g1f 1.0057）。5/5 一貫の後退は g1w の train fresh のみで、g1・g1f は一貫せず | undetermined（[optin-ab-gb10.md](./optin-ab-gb10.md) B1） | undetermined |
| G2 `REDUCTION_SEQUENTIAL_FALLBACK_ENABLED` | 2101／2102 | Stage 0: T_macmini = 16384（PROCEED）。Stage 1: undetermined（record_only） | ADOPT 不成立。gemm 2048/reuse 1.0065・4096/fresh 1.0001・4096/reuse 1.0002 が後退。5/5 一貫の後退はなし | undetermined（同 B2。T_gb10 = 16384） | undetermined。最終 T = min(16384, 16384) = 16384 の候補 |
| G3 `TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED` | 2113 | 前提ゲート (a)(b)(c) 成立。性能判定は undetermined（record_only） | ADOPT 不成立（判定セル train fresh が 1.0002 で ratio > 1.00）。REJECT 条件（5/5 一貫の後退）も不成立。train reuse は 0.8941 で 5/5 round 改善 | 対象外（未実測） | undetermined |
| G4 `READBACK_DEST_DEFAULT` | 2112 | undetermined（record_only） | REJECT 相当（専有ゲートは成立。gemm reuse N=4096 が 1.0020 > 1.00。N=1024 は 0.9887、infer reuse は 0.9932） | 対象外 | undetermined |
| G5 readout legacy／borrowed | §15.1（#3265 へ） | undetermined（record_only） | ADOPT 相当（6 セル全て borrowed/legacy ≤ 1.00・checksum 完全一致。0.9462〜0.9928） | 対象外 | undetermined。#3265 には参考値として渡す |
| G6 steel 候補 | 2111 | INCOMPLETE（前提ゲートは成立。集計器が run ログ 5 本を「成功行なし」と判定） | 集計器の判定が出ないため読めない。生ログの head/base 比は下記（参考値） | 対象外 | undetermined |
| G7 `METAL_DEVICE_VERIFY_CACHE_DEFAULT_ENABLED` | 2114（ADOPT 判定なし） | record_only。集計器が fail-closed で停止（H1〜H4 は未確定） | — | 対象外 | record_only |

機体間で判定が割れたゲートはない（GB10 側が未実測または undetermined のため）。#3227 の対象は本表で ADOPT 条件が成立したゲートのみで、現状なし。

## ゲート別の実測

全て 5 round・checksum 完全一致（`--require-checksum-exact`）。欠損・round 数 ≠ 5 はなし。比 = after/before の 5 round 中央値。

### G1 host arena（train／infer size=64、cpu）

比（train fresh／train reuse／infer fresh／infer reuse）:

| 系列 | train fresh | train reuse | infer fresh | infer reuse |
|---|---|---|---|---|
| g1（初回） | 1.0115 | 0.9887 | 0.9993 | 1.0022 |
| g1w（ウォーム） | 1.0291（5/5 後退） | 0.9934 | 1.0007 | 1.0090 |
| g1f（再実行） | 1.0057 | 1.0067 | 1.0275 | 0.9671 |

生ログ: `logs/optin-ab-macmini-3184/host-arena/`。3 系列で符号が揃わず、train reuse は 0.9887→1.0067 と系列間で反転する。サブミリ秒セルのノイズが ±3% 程度あるため、この条件では方向を言えない。

### G2 縮約逐次化

- Stage 0（`reduction/stage0/`、5 プロセス独立）: T_macmini = 16384。T 未満の全セルの最大 r = 1.0236（R_MAX 1.05）。利得（5/5 run で r < 1.00）は sum_all 4096／8192、sum_axis0 640〜4096、sum_axis1 640〜8192、max_all 8192、mean_all 8192。継承集計 `aggregate_sweep.py` を単機で呼ぶ補助 `aggregate_sweep_single.py` で算出（判定ロジックは再実装していない）。
- Stage 1（T = 16384、`reduction/stage1/`、系列 g2f。g2w は誤って中断した走行で結果が未完のため参考扱い）: train 64／fresh 0.9997・reuse 0.9933、infer 64／fresh 0.9808・reuse 0.9343、gemm 1024／fresh 0.9861・reuse 0.9862、2048／fresh 0.9980・reuse 1.0065、4096／fresh 1.0001・reuse 1.0002。MSE backward は適用外（2102 の除外節）。
- 最終 T の候補は min(T_macmini, T_gb10) = min(16384, 16384) = 16384。T_macmini と T_gb10 が一致したため Stage 1 の再計測は不要になる見込み（GB10 の Stage 1 は #3190 で undetermined）。

### G3 train forward encode-only

前提ゲート（`encode-only/prereq/`）:

| 項目 | 結果 |
|---|---|
| (a) backend-metal `train_forward_encode_only_parity` | 4 passed |
| (a) facade `metal_train_forward_encode_only_bit_identity` | 1 passed（to_bits 完全一致） |
| (b) カウンタ | ON 11/6/6、OFF 11/7/7、`backward_dinput_phase` 5/3/3 を再現 |
| (c) 既存 `#[ignore]` 群 | 全て after／before で同じ結果。`command_batching`（`pool_reuse_zero_fill_does_not_synchronize_open_batch` が失敗）と `device_param_store_backend_parity`（CUDA 前提の 2 件が失敗）は before でも失敗する既存の失敗で、非後退 |

性能（`encode-only/`、Metal、5 round）: train fresh 1.0002（後退側・5/5 一貫せず）、train reuse **0.8941**（全 5 round が 0.8867〜0.8991 で改善）。infer は record_only（fresh 1.0100・reuse 1.0412。判定に使わない）。GB10 の対照セル（cpu／cuda の A/A）は #3190 の対象外で未実測。

### G4 readback 宛先

`readback/`。専有ゲート（load1 < 4.0 を 2 回連続）は成立。gemm reuse N=1024 0.9887・**N=4096 1.0020**・infer reuse 0.9932。生 JSONL は gemm と infer が混在するため、`split_by_task.py` で task だけを基準に機械的に分割してから 2112 README §3 のコマンドを実行した（行の値と順序は不変。継承 `compare_gemm_ab.py` は混在ファイルを「判定不能」とする）。凍結定数（8 MiB・8 スレッド）は変更していない。

### G5 readout legacy／borrowed

`readout/`。専有ゲート成立。N=1024／2048／4096 × fresh／reuse の borrowed/legacy は 0.9754／0.9462／0.9691／0.9658／0.9800／0.9928（全セル checksum 完全一致）。継承 §15.1 のみで読むと ADOPT 相当だが、#1520 の REJECT を覆すものではなく「専有 Mac mini での条件を変えた再測定」であり、round 開始の load1 が 3.0 を超えた系列（G5 は round 3〜5 が 3.67〜5.15）のため判定は undetermined。#3265 には参考値として渡す。

### G6 steel 候補

`steel/`。前提ゲート 4 本（bit 一致 3・parity 1）は全て成功。5 run は完走（各 run が `test result: ok. 1 passed`）。継承 `aggregate.py` の `check_run_log` は、`--nocapture` で `test ... ... N=512 arm=...` と出力が同じ行に続く形を受理せず（「対象テストの成功行が無い」）、全 arm が INCOMPLETE になった。継承集計器は編集しない規則のため、判定は INCOMPLETE のまま。

参考値（生ログ `kernel_gpu_run*.log` の `head_over_base_kernel_gpu` を 5 run の中央値にしたもの。**判定ではない**。全 arm・全 N で bit_identical=true・run 間ハッシュ一致）:

| N | T0U | LU | T0U-LU | T0U-LU-FB |
|---|---|---|---|---|
| 512 | 1.0185 | 0.9979 | 1.0347 | 0.9989 |
| 1024 | 0.8976 | 1.0000 | 0.8892 | 0.8511 |
| 2048 | 0.7639 | 1.0003 | 0.7567 | 0.7250 |
| 4096 | 0.7746 | 1.0010 | 0.7575 | 0.7231 |

Mac mini（M4）では本番選択が 32×32 タイルのため、candle 相当の 64×64 タイル（T0U）が N ≥ 1024 で 10〜24% 速く見える。M4 Max とは別の解決先になるため、機体固有の結果である。N=512 の結論には、candle のデバイス区分の仮定が未確認（`metal-gemm-steel-candidates.md` §3。影響は N=512 のみ）という注記が付く。

REQ-8 の確認: 固定 4 arm は `metal-gemm-steel-candidates.md` §2 の構成（タイル選択・`UNROLL_ACC`・`UNROLL_LOAD`・`FINE_BARRIER`）で、§4 が align_M/N/K 分岐ロード（#808 で不採用・REQ-8）を除外している。`UNROLL_LOAD` は整列可否で分岐しない。したがって手動境界チェックを外す arm は含まれない。境界検査を省く候補は採用しない。

### G7 device verify cache

`verify-cache/`。5 run を record_only で実行した（`host_kind` は `m4max` しか受け付けないため、トークンを流用して `--out` で出力先を指定した）。`aggregate.py` は run2 の `tape_build_micro`／`tape_new` の `median_s` が 0.0（有限の正値でない）として fail-closed で停止した。H1〜H3 と checksum 規則は評価できず、H1〜H4 は未確定。H4 の (c) `infer_fixed_cost_diag_wall_vs_gpu_busy` は実行していない。

## 申し送り

- #3227: 本表に ADOPT 条件が成立したゲートはない。G3 の train reuse（0.8941、5/5 改善）は材料になるが、train fresh が 1.0002 で判定セルの条件を満たさないため、正式系列の再計測が必要。
- #3265: G5 は参考値（上記）。正式な再測定には、負荷で round 開始値が上がらない計測か、補遺 RULE 3 の条件の見直し（所有者判断）が必要。
- 正式系列を成立させるには、継承スクリプトを編集せずに round 開始の load1 を < 3.0 にする方法がない（計測自身が 10 コアを使う）。補遺 RULE 3 の事後検査を「計測に入る前の外側ゲート値」へ変えるか、継承スクリプトへ機体別閾値を入れる改修が必要で、いずれもスコープ外として切り出した。
- 継承 `metal-gemm-candidate-ab-2111/aggregate.py` の `check_run_log` と `--nocapture` 出力形式の不整合（G6）、`metal-tape-build-infer-phase-2114` の `tape_new` の `median_s` = 0.0（G7）は、継承側の別対応が必要。
