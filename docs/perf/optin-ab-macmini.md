# Mac mini の未計測 opt-in 候補 A/B（イシュー #3184）

状態: **再開分の実測済み（2026-10-22 JST。Mac mini M4・事前登録コミット `c80d9818`）。正式系列が成立し、G1・G4 は REJECT（Mac mini 単機で確定）、G2・G3 は undetermined、G5 は継承集計器が ADOPT（#1520 を覆さず #3265 へ）、G6 は arm 別の判定語彙が出た、G7 は record_only（H1〜H3 未確定・H4 支持）**。詳細は「再開分の結果」節。
初回（2026-10-21・main `26e60f63`）は正式系列が成立せず全ゲート undetermined だった（旧系列として残す。「初回の記録」節）。
main の定数・環境変数既定は変更していない。規則は `logs/optin-ab-macmini-3184/RULE.txt`（初回）と `RULE-reopen.txt`（再開分。いずれも計測前に固定。継承 RULE も補遺も計測後に緩めていない）。

## 再開分の結果（2026-10-22 JST／UTC 2026-10-21・Mac mini M4・事前登録コミット `c80d9818`）

初回は正式系列が成立せず判定が出なかったため、補遺 2（`logs/optin-ab-macmini-3184/RULE-reopen.txt`）を計測前に固定して取り直した。
継承 RULE と元の補遺は 1 字も変えていない（修正前の sha256 を RULE-reopen.txt に記録）。事前登録コミット `c80d9818` の作者日時は 2026-10-21T18:16:02Z で、最初の実機ログ（`macmini-r2/steps.log` の 18:16:41Z）より 39 秒前。
before／after／diag の 3 clone と GB10 の 2 clone は全て同じ `c80d9818` で、`git status --porcelain` は 0 件（`macmini-r2/rev-stamp.txt`・`gb10/rev-stamp.txt`）。系列は新 label（`3184-macmini-r2-*`・`3184-gb10-g3-*`）で 1 回ずつ実行し、差し替え・追加・セル除外はない。

### 変更点（判定を緩めない・条件を足すだけ）

| 変更 | 内容 |
|---|---|
| round 前クールダウン hook | `scripts/bench/framework-compare/ab_round_cooldown.sh`。`AB_ROUND_COOLDOWN_MAX_LOAD1` 設定時のみ round ごとに load1 < 閾値を待ち、`cooldown-*.log` へ round 別 load1 を記録する。未設定なら no-op（self-test `ab_round_cooldown_test.sh` で確認）。継承 5 スクリプトへは追加行のみ。継承の閾値 8.0・`gate_ok`・series 判定は不変なので、継承 gate log は `series=official` かつ load1 < 3.0 が同じ行に並ぶ |
| G6 の外側待機 | run ごとに `orchestrate.sh` の外で load1 < 3.0 を待ち、float で `steel-r2/cooldown.log` に記録 |
| 2111 集計器 | `--nocapture` で融合する最初の出力行（`test NAME ... N=...`）を 1 ログにつき 1 回だけ受理。FAILED／panicked・`1 passed; 0 failed`・独立 `ok` 行の検査は不変（拒否ケースを self-test に固定） |
| 2114 `orchestrate.sh` | host_kind に `macmini` を追加（`m4max` と同じ record_only の経路） |
| `machine_gate_check.py` | cooldown ログ形式を追加。`cooldown=` が pass 以外の行は 0 として扱う（厳しくなる方向のみ。判定式 < 3.0・5 round 完備は不変） |

### 条件差（初回との違い）

| 項目 | 初回 | 再開分 |
|---|---|---|
| round 開始時の load1 | 継承スクリプトの直前ビルド・前 round の負荷で 3.0 を超えた | 全系列・全 round で < 3.0（G1 2.48〜2.52、G2 2.38〜2.71、G3 2.8〜2.9、G4 2.28〜2.65、G5 2.29〜2.68、G6 1.62〜2.78。`macmini-r2/machine_gate_ok.txt`） |
| G4／G5 の round 別記録 | uptime の 30 秒サンプルのみ（判定不能） | `cooldown-*.log` に round 別 load1。全 5 round が `cooldown=pass` |
| 外側ゲート | 成立 | 成立（load1 < 3.0 かつ並走プロセス 0 件。`macmini-r2/machine_gate.log`・`preflight-procs.txt`）。同じ Mac mini を使う他セッションの計測とは並走していない |
| G2 Stage 0 | 実施（T_macmini = 16384） | **再計測しない**（2101 RULE が Stage 0 の負荷を record_only と定め、run の追加による再判定を行わないため。T_macmini = 16384 を継承）。初回 `stage0/env_info.txt` の load1 推移（2.92→5.60）は参考値で、Stage 0 の判定には効かない |
| G3 前提ゲート | rev `26e60f63` | 新しい rev `c80d9818` で取り直した（初回の結果は流用しない） |
| G3 GB10 対照セル | 未実測 | 実測（cpu・cuda × train {fresh, reuse}。継承ゲート 8.0。Mac mini 用の 3.0 は GB10 に適用しない） |
| G6 `mtl_architecture_name` | 未取得 | `applegpu_g16g`（`p0_device_attributes` の出力を転記。`steel-r2/env_info.txt`） |
| G7 `host_kind` | `m4max`（流用） | `macmini` |

### ゲート別判定（再開分）

比 = after/before（G5 は borrowed/legacy）の 5 round 中央値。全セル checksum 完全一致（`--require-checksum-exact`）、5 round 完備、非有限値なし。継承 RULE の文言を機械的に当てはめた。

| ゲート | Mac mini 判定（再開分） | 根拠 | GB10（#3190／本再開分） | 総合 |
|---|---|---|---|---|
| G1 `HOST_ARENA_DEFAULT_ENABLED`（2104） | **REJECT**（Mac mini 単機で確定） | 正式系列（gate 8.0 pass・全 round load1 < 3.0）で train reuse が 1.0090、5 round 全てが ratio > 1.00（1.0153・1.0036・1.0420・1.0025・1.0060）。他セル: train fresh 1.0115（4/5）・infer fresh 0.9961・infer reuse 1.0181（3/5） | undetermined（#3190 B1。train fresh 0.8648 の改善は別機体の値で判定を覆さない） | **REJECT**（2104 は「いずれかの機体で 5/5 一貫の後退セルが 1 つでもあれば REJECT」） |
| G2 `REDUCTION_SEQUENTIAL_FALLBACK_ENABLED`（2102 Stage 1。T = 16384） | undetermined | 後退セルが 2 つ（gemm 2048/reuse 1.0114〈4/5 round 後退〉・gemm 4096/fresh 1.0010〈3/5〉）で ADOPT 不成立。5/5 一貫の後退はなく REJECT 不成立。train 64／fresh 0.9978・reuse 0.9951、infer 64／fresh 0.9657・reuse 0.9473、gemm 1024／fresh 0.9860・reuse 0.9930、2048／fresh 0.9902、4096／reuse 0.9983 | undetermined（#3190 B2） | undetermined。最終 T = min(16384, 16384) = 16384 の候補は不変 |
| G3 `TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED`（2113） | undetermined。前提ゲート (a)(b)(c) は成立 | train fresh 1.0077（後退・5 round 中 3 回のみ > 1.00）で ADOPT 不成立。train reuse **0.9000**（全 5 round が 0.8835〜0.9060 で改善）。5/5 一貫の後退なし。infer は record_only（fresh 0.9977・reuse 0.9434） | 対照セル（A/A）: cpu fresh 0.9771・reuse 1.0138、cuda fresh 1.0071・reuse 0.9944。いずれも 5/5 一貫の後退でなく、非後退でないセルがあるため undetermined | undetermined |
| G4 `READBACK_DEST_DEFAULT`（2112） | **REJECT**（専有ゲート成立の系列） | 専有ゲート（load1 < 4.0 を 2 回連続）成立・全 round `cooldown=pass`。gemm reuse N=1024 1.0228・N=4096 1.0074・infer reuse 1.1604（全て > 1.00）。2112 は「専有ゲートが成立した系列でいずれかのセルが ratio > 1.00」で REJECT | 対象外 | **REJECT**（Mac mini を M4 Max 枠として扱う補遺 RULE 4 による） |
| G5 readout legacy／borrowed（§15.1） | 継承集計器の総合判定は ADOPT（6 セル全て ≤ 1.00） | borrowed/legacy は 1024／fresh 0.9592・reuse 0.9688、2048／fresh 0.9773・reuse 0.9679、4096／fresh 0.9752・reuse 0.9750。専有ゲート成立・全 round `cooldown=pass`。#1520 の REJECT は覆さない（条件を変えた再測定）ため、採否は #3265 | 対象外 | #3265 へ「専有 Mac mini での再測定」として渡す。本イシューは既定を変えない |
| G6 steel 候補（2111） | 正式系列（gate OK・`reference_only=False`）。LU: UNDETERMINED、T0U: **REJECT**、T0U-LU: **REJECT**、T0U-LU-FB: UNDETERMINED | T0U／T0U-LU は N=512 が 1.0378／1.0547 で 5/5 run 符号一貫の後退（REJECT の条件）。N=1024 以上は T0U 0.8987／0.7657／0.7748、T0U-LU 0.8903／0.7587／0.7575 と 10〜24% 速い。N=512 の結論には candle デバイス区分の仮定が未確認という注記が付く | 対象外 | Mac mini 固有の結果（M4 では本番が 32×32 タイルのため） |
| G7 `METAL_DEVICE_VERIFY_CACHE_DEFAULT_ENABLED`（2114。ADOPT 判定なし） | record_only。H1〜H3 は**未確定**、H4 は**支持** | 集計器が fail-closed で停止（run2 の `provider_select` の `median_s` = 0.0。`tape_new` は median 4.1e-8・q1 0.0 で、計時分解能 41.67 ns を下回る）。集計器の正値検査は変えていないため H1〜H3 は推定で埋めない。H4 は (c) の出力を転記: host_fixed 368.7 us が wall 588.6 us の 63%（>= 50% で支持） | 対象外 | record_only |

機体間で判定が割れたゲート: G1（Mac mini REJECT／GB10 undetermined。GB10 の train fresh は 0.8648 と改善、Mac mini の train reuse は後退。2104 の規則上は REJECT が優先）。

### 生ログ・集計の所在

| ゲート | 収録先（`logs/optin-ab-macmini-3184/`） |
|---|---|
| G1 | `host-arena-r2/`（compare-{train,infer}・`load-gate`・`cooldown`・生 JSONL） |
| G2 Stage 1 | `reduction/stage1-r2/`（`on-reduction-filled.sha256`・`patch_apply.log` を含む。Stage 0 は初回の `reduction/stage0/` を継承） |
| G3 | `encode-only-r2/`（`prereq/` に前提ゲート (a)(b)(c)）・GB10 対照は `encode-only-gb10/` |
| G4 | `readback-r2/`（`split/`・`compare-gemm-readback.md`・`compare-infer-readback.md`） |
| G5 | `readout-r2/`（`compare-readout.md`） |
| G6 | `steel-r2/`（`aggregate.md`・`cooldown.log`・`env_info.txt`） |
| G7 | `verify-cache-r2/`（`aggregate.err`・`h4c.log`・`env_info.txt`） |
| 機体の証跡 | Mac mini は `macmini-r2/`、GB10 は `gb10/`（`uname.txt`・`rev-stamp.txt`・`steps.log`・`machine_gate*`・`preflight-procs*`） |

G3 前提ゲートの (c): `device_param_store_backend_parity` は after／before とも同じ 2 件（`device_resident_matches_host_sgd_on_cuda_across_100_steps`・`grad_readout_contract_on_cuda`。CUDA 前提で Mac では動かない）が失敗する既存の失敗で、非後退。その他の群は全て rc 0。
REQ-8 の確認（G6）は初回と同じ（固定 4 arm は手動境界チェックを外さない）。

### 申し送り（再開分）

- #3227: 反転できるゲートはない（ADOPT は両機体の成立が要り、GB10 側が undetermined）。G1 と G4 は Mac mini の正式系列で REJECT が確定したため、既定 OFF の維持が妥当。G3 は train reuse が 5/5 改善（0.9000）だが train fresh が 1.0077 で未成立。
- #3265: G5 は専有 Mac mini の正式系列で全 6 セル ≤ 1.00（0.9592〜0.9773）。#1520 の REJECT を覆す材料として渡す（採否は #3265 の判断）。
- 初回系列（record_only）は無効にせず残してある。旧 `steel/` を修正後の集計器で再集計すると判定語彙が出るが、旧系列は遡って有効にしない（RULE-reopen.txt 1.(ii)）。
- スコープ外: `tape_new` が計時分解能を下回る問題（2114 の集計器と計測粒度の見直し）は #3618 で計測側を修正済み（まとめ計時。実機再測定は未実施。「G7 の分解能を上げた再計測（#3618）」節）。ruleset の required check の追加・変更は不要だった（workflow は触っていない）。

---

## 初回（2026-10-21）の記録（旧系列・record_only）

以下は初回系列の記録で、判定の正は上の再開分。

## 条件差

| 項目 | 継承 RULE の想定 | 本計測 |
|---|---|---|
| 機体 | M4 Max + GB10 | Mac mini M4（P4+E6、GPU 10 コア、macOS 27.0.1、rustc 1.99.0。Apple Silicon 枠として扱う。`logs/optin-ab-macmini-3184/macmini/uname.txt`・`rev-stamp.txt`） |
| 負荷 | 継承登録値（8.0／exclusive 4.0／record_only） | 継承スクリプトは登録値のまま。外側で系列開始前に load1 < 3.0 を待ち、事後に round 別 load1 < 3.0 を検査（`machine_gate_check.py`） |
| 外側ゲート | — | 全系列で成立（load1 1.4〜2.9。`macmini/machine_gate.log`）。並走プロセスは全系列で 0 件（`macmini/preflight-procs.txt`） |
| 事後検査（round 別 load1 < 3.0） | — | **全系列で不成立または判定不能**（round 番号付きログの系列は不成立、G4／G5 は uptime の定期サンプルのみで round と対応しないため判定不能。`macmini/machine_gate_ok.txt`） |
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

`readout/`。専有ゲート成立。N=1024／2048／4096 × fresh／reuse の borrowed/legacy は 0.9754／0.9462／0.9691／0.9658／0.9800／0.9928（全セル checksum 完全一致）。継承 §15.1 のみで読むと ADOPT 相当だが、#1520 の REJECT を覆すものではなく「専有 Mac mini での条件を変えた再測定」であり、round 別の load1 を確認できない系列（G5 は 30 秒間隔の uptime 定期サンプル 5 件のみで round 開始値ではなく、うち 3.67〜5.15 が 3.0 超。round との対応は判定不能）のため判定は undetermined。#3265 には参考値として渡す。

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

## G7 の分解能を上げた再計測（#3618）

- 状態: **未実測**（Mac mini での再測定は本変更の時点で未実施。H1〜H3 は引き続き未確定。推定で埋めない）。
- 原因: `Instant` は macOS で `clock_gettime(CLOCK_UPTIME_RAW)`（rust-src `library/std/src/sys/time/unix.rs` 55〜67 行目）。再開分の実ログの離散値（4.1e-8・4.2e-8・8.3e-8）から 1 tick は約 41.7 ns と推定している（実測は `timer_resolution_probe` を Mac mini で実行して確定させる）。集計器は 1 セルでも `median_s` が 0 なら全体を止める。
- 対応（計測方法のみ。判定式・閾値・集計器は不変）: 4 セル（`tape_build_micro.tape_new`・`reuse_decomposed_off.tape_new`・`reuse_decomposed_on.tape_new`・`reuse_decomposed_on.provider_select`）を K = 1000 回まとめ計時した平均にし、従来値は `single_shot_*` に残す。バッチはカウンタ窓・`iter_total`・`tape_build` 窓の外で実行する。規則は `docs/perf/logs/optin-ab-macmini-3184/RULE-g7-r3.txt`、手順は `session-scripts/g7_r3_session.sh`。
- 開発機（M4 Max）での動作確認のみ実施: 4 セルとも `median_s` > 0（tape_new 約 11〜14 ns・キャッシュ ON の provider_select 約 2 ns）、集計器の `--self-test` は変更なしで OK。この値は結果に使わない。
- 次の手順: 事前登録コミットを Mac mini へ転送し `g7_r3_session.sh` を 1 回実行する（label `3184-macmini-r3-g7`）。H4 の再実行と #2114 の M4 Max 実測は対象外。
