# sm_121 向け GEMM 候補の再分類と設計記録（#2130）

## 0. 位置づけ・スコープ

- 対応イシュー #2130（親 #2122「sm_121 可用命令の実機プローブ」・ルート #2121 Phase 4）。**設計のみ**の記録であり、候補カーネルの実装・GB10 実機実測・新規依存の追加は行わない。tolerance 定数・`ParityBaseline`・既存の採否判断（3×TF32 のユーザー判断を含む）・facade 公開 API は変更しない
- 目的: CUDA f32 GEMM の最適化候補のうち REJECT／非推奨となった 4 件（StreamK #1359・persistent #1347・TMA Stage 1 #1975/#1976・3×TF32 #1356）について、不採用の原因が (H) ハードウェア限界か (I) 実装の不完全さか (D) 設計・机上モデルの前提かを既存の確定記録だけで再分類し、再挑戦すべき候補とそうでない候補を分ける。あわせて新規候補の opt-in 設計と本番結線の判断フローを定める
- **#2122 は本 doc 作成時点で未完了**（プローブ結果なし）。本 doc は既存の確定記録だけで成立する構成とし、#2122 で確定する項目は §2.3 の差し込み欄を「未確定」のまま置く。推定で埋めない。#2122 完了時は同 issue 側で §2.3 を更新する
- 出典の実測値は各 doc を正とし、本 doc は転記と分類のみを行う

## 1. 前提の訂正

イシュー本文の想定と既存の確定記録が食い違う点を、黙って直さずここに記録する。

1. **tcgen05／TMEM／wgmma・cluster**: イシューは sm_121 で可用と想定しているが、`docs/cuda-tensor-core-design.md` §11.1（CUTLASS v4.7.0 一次ソースの静的読解）は、tcgen05／TMEM／wgmma は SM120/sm_121 で**不可**、cluster は実用上 **1×1×1 のみ**と記録している。またイシューは sm_121 を「Blackwell datacenter」と呼ぶが、同 §11 は SM12x を DC Blackwell（SM100 系）と区別している。#2122 が §11 を実機で覆さない限り、tcgen05／TMEM 系の候補は列挙対象外とする（§4）
2. **TMA Stage 1 の N=256 後退**は、ゲート C（GPU-only の純カーネル時間）での後退である。ゲート A（bit 一致）は全 PASS、ゲート B は 0 fail（`docs/backend-cuda-tma-gemm-load-design.md` §10.8）
3. **3×TF32 の P1（厳密ゼロ fail）不成立**は、既に GB10（sm_121）実機で実測済みである（#1356。`docs/perf/cuda-tensor-core-tolerance-tf32x3-gb10.md` §10・§11）。2026-09-08 のユーザー判断で「opt-in 維持・非推奨」・baseline 不承認が確定している（`docs/cuda-tf32x3-split-single-decision.md` §9）。受入基準「sm_121 で検証」は既存実測の再掲と原因分類で満たし、本 issue では再計測しない

## 2. 確定済み事実の集約

### 2.1 機能可否（出典付き・推定なし）

| 機能 | sm_121 での状態 | 出典 |
|---|---|---|
| `mma.sync`・`ldmatrix`・`cp.async` | 可（実装実績あり） | `cuda-tensor-core-design.md` §11.1 |
| TMA（`cp.async.bulk.tensor`） | 可（GB10 実機で NVRTC compile・CTA 実行・cluster 実行の 3 プローブが成立。`shared::cta` と cluster variant の両方で bit 一致。要素座標・部分 OOB・smem 配置ダンプの意味論プローブ 3 件は未実装で、その範囲の検証は未了） | `backend-cuda-tma-gemm-load-design.md` §10.8 |
| wgmma・tcgen05・TMEM | 不可（静的読解。実機での再確認は #2122 待ち） | `cuda-tensor-core-design.md` §11.1 |
| cluster（実用） | 1×1×1 のみ（静的読解。launch 可否の実機確認は #2122 待ち） | 同 §11.1 |
| `setmaxnreg` | **未実測**（プローブ実装はあるが実機実行は未了） | 同 §13・`backend-cuda-tma-gemm-load-design.md` §2 F2 |
| `has_async_alloc()` | true | `docs/perf/lowlayer-diagnosis-2026-09-12.md` |

### 2.2 デバイス属性

SM 数 48・L2 25,165,824 B（24 MiB）・global 実効帯域 212.34 GB/s・L2 実効帯域 1237.62 GB/s（`docs/perf/sm121-device-attributes.md`）。常駐ブロック数は 64×64 タイルで 3 block/SM（grid_capacity 144）・128×64 タイルで 2 block/SM（同 96）で、机上見積りと実測が一致している（`docs/perf/cuda-gemm-tiled-pipeline.md`「#1347」節）。

### 2.3 #2122 確定値の差し込み欄（未確定）

| 項目 | 状態 | 依存する候補 |
|---|---|---|
| cluster サイズ >1 の launch 可否 | 未確定（#2122 待ち） | C3(b) |
| DSMEM（`mapa`／`ld.shared::cluster`）の可否 | 未確定（#2122 待ち） | C3(b) |
| `setmaxnreg` の受理・実行 | 未確定（#2122 待ち） | C2 |
| SM120 系の追加 mma 形状（f8f6f4・block-scaled。sm_121a 要否） | 未確定（#2122 待ち） | 本 issue では対象外（§4） |
| `CLOCK_RATE` 等の未実測属性 | 未確定（#2122 待ち） | bytes/cycle 換算を要する分析全般 |

## 3. REJECT 理由の再分類

分類語彙は 3 値: **(H)** ハードウェア限界／**(I)** 実装の不完全さ／**(D)** 設計・机上モデル起因（HW でも実装欠陥でもない）。

| 候補 | 分類 | 再挑戦可否 | 依存する #2122 項目 |
|---|---|---|---|
| StreamK（#1359） | (D)／(I)。HW 限界の根拠なし | 可（C3） | cluster・DSMEM（C3(b) のみ） |
| persistent（#1347） | (D)。HW 起因ではない | 価値低（Stream-K 側で扱う） | なし |
| TMA Stage 1（#1975/#1976） | (I) が主。HW 限界の根拠なし | 可（C1） | なし |
| 3×TF32（#1356） | (H)＋(D) の複合の可能性（判別実験前は確定しない） | 候補として再興しない | なし |

### 3.1 StreamK（#1359）

**再掲**（`docs/cuda-streamk-decision.md`「#1359 実測結果」・`docs/perf/cuda-gemm-tiled-pipeline-streamk.md` §6）:

- 決定性は充足: `cpu_cuda_tiled_pipeline_streamk_parity -- --ignored` の 8 テストが GB10 で全 PASS（固定順序 fixup）
- ゲート B-1 は 16 行中 12 行で `fail_count > 0`
- ゲート C は N=1024 で 1.0271 倍（基準 ≥1.05 未達）・N=2048 で 0.9395 倍（基準 ≥1.00 未達）。ゲート D も未達
- `streamk_plan` ログ: N=1024 は grid 144・`max_contributors=3`（fixup 往復約 5.25 MiB）、N=2048 は `max_contributors=10`

**机上 wave 値**（`cuda-streamk-decision.md` §2。64×128 タイル・2 blocks/SM 仮定）: quantization loss は N=4096 で 3.0%・2048 で 11.1%・1024 で 33.3%。

**分類**: fixup を別カーネルにしたのは「カーネル内 fixup は CTA 間の co-residency が保証されずデッドロックしうる」という設計選択である（`cuda-gemm-tiled-pipeline-streamk.md` §6.8）。末尾 wave 短縮の利得を fixup 固定費が相殺・逆転したのが実測の結論で、ハードウェア限界を示す記録はない。したがって (D)／(I) に分類する。co-residency を保証する起動（cooperative launch。unsafe 承認整理は #2127）、または #2122 で cluster>1 と DSMEM が確定した場合のクラスタ内還元で選択肢が変わるため、再挑戦余地がある（C3）。B-1 の `fail_count > 0` は K 分割に固有の結合順序差であり、tolerance・`ParityBaseline` の扱いはユーザー承認事項として切り分ける（本 doc では判断しない）。

**AC1（wave 定量化・CTA→SM 分布）**: 上記の机上値は再掲のみで、**CTA→SM の実分布は未実測**である。判別プローブを次のとおり設計する。各 CTA が `%smid` と `%globaltimer` の開始・終了値を記録するカーネルを `internal-diagnostics` feature 限定・`#[ignore]` で用意し、wave 境界と SM 遊休時間を可視化する。決定性の再確認は既存の 8 テスト（再実行で bit 一致）をそのまま使う。実測は後続 issue（§7）に委ねる。

### 3.2 persistent タイルキュー（#1347）

**再掲**（`docs/perf/cuda-gemm-tiled-pipeline.md`「#1347」節）: N=1024 で 64×64 が 1.0063 倍・128×64 が 1.0182 倍（基準 ≥1.05 未達）。N=4096 の 64×64 は 0.9637 倍。占有率は机上見積りと実測が一致（§2.2）。

**机上再検算（新規）**: タイルが原子的な作業単位のまま（K 分割なし）で、総タイル数が常駐スロット数を超える場合、タイル取得を動的にしても最終 wave の長さは変わらない。N=1024 の数値で示すと、64×64 は 256 タイル／144 スロット（約 1.78 wave）、128×64 は 128 タイル／96 スロット（約 1.33 wave）で、どちらも最後の端数分のタイルが 1 タイル分の実行時間を要する構造は動的取得でも残る。persistent が回復できるのは、タイルごとの実行時間ばらつきによる遊休に限られる。#1347 の 3 つの原因仮説はこの原理的な点を問うていなかった。

**分類**: (D)。期待効果の机上モデルが、K 分割なしの persistent に wave 回復を仮定していた。HW 起因ではなく、sm_121 固有の再挑戦価値は低い（wave 回復は K 分割を伴う Stream-K 側で扱う）。SM 間の実行時間ばらつきによる限定的効果の有無は、§3.1 のプローブ出力で併せて確認できる。

### 3.3 TMA Stage 1（#1975/#1976）

**再掲**（`backend-cuda-tma-gemm-load-design.md` §10.8・`docs/perf/logs/cuda-tma-stage1-1975/aggregate.md`）:

- ゲート A 全 PASS・ゲート B 0 fail
- ゲート C の N=256 は none 腕 0.9857・B64 腕 0.9627（いずれも 5/5 で後退。絶対差 0.027 TFLOPS≒1.3%）。N≥512 では同タイル（64×64）の cp.async 版に対し 7.6〜20.7% 改善
- 本番構成（128×64）に対しては N=2048／4096 で下回る（none 腕 0.9810／0.9282）

**AC3（原因切り分け）**:

- **tensor map の encode 固定費仮説は棄却**: ゲート C は `prepare_tiled_pipeline_tma_maps` で事前 encode し、計測区間外に置いている（`crates/backend-cuda/examples/gemm_tiled_pipeline_bench.rs` 冒頭コメント）。したがって N=256 の後退の説明には使えない。ただし本番到達時のホスト側 encode 費は別問題として残る（tensor map キャッシュは未実装。§10.8 の申し送り）
- **残仮説と判別実験（設計のみ・実測は後続）**:
  - (a) N=256 は 64×64 タイルで 16 CTA／48 SM の遅延律速域にあり、単一 elected thread の TMA 発行と mbarrier 待ちのレイテンシが cp.async より長い。ncu の `smsp__cycles_active`・stall 理由（`smsp__pcsamp_warps_issue_stalled_*`）の比較と、段数 2／3／4 の感度 A/B で判別する
  - (b) 密レイアウト（パディングなし）の smem バンク衝突。`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` を none／b64／cp.async で比較する。N≥512 で none 腕が速い事実から、支配的でない可能性も併記する
  - (c) 記述子取得のレイテンシ。`prefetch.tensormap` の有無で A/B する

**分類**: (I) が主（形状条件・タイル選択・段数の未調整）。HW 限界を示す根拠はない。

### 3.4 3×TF32（#1356）

**再掲**（`cuda-tensor-core-tolerance-tf32x3-gb10.md` §10・§11・`cuda-tf32x3-split-single-decision.md` §9）: P1 は FAIL（512³・seed 5004 で 212/262144、4096³・seed 9001 で 151916/16777216）。P4 は 5 形状すべてで `mma_tf32x3 / f32_simt` が 0.721〜0.930 倍。ユーザー判断は「opt-in 維持・非推奨」・baseline 不承認・テストは厳密判定のままで確定済み。**この判断は本 doc で変更しない**。

**判別実験（設計のみ）**: fail の原因が Tensor Core の累積意味論（HW。内部の積和の丸め・整列が IEEE f32 FMA と異なる可能性）か、設計要素（`lo·lo` の省略・累積順序・`__float_to_tf32` の丸め）かを分けるため、同一の 3-pass 分割をホスト側で f32／f64 で模擬した参照を作り、GPU 出力との差を比較する。模擬参照と GPU が一致して f32 SIMT との差だけが残れば設計要素、GPU だけが外れれば累積意味論側と読む。

**分類**: (H)＋(D) の複合の可能性（確定は判別実験後）。候補としては再興しない。

## 4. 新規候補の列挙

根拠が既存記録にあるものだけを挙げる。

| 候補 | 根拠 | #2122 依存 | 想定効果・対象形状 | 事前登録ゲート | opt-in 方式 |
|---|---|---|---|---|---|
| C1: TMA Stage 2（128×64 タイルへの TMA 適用）＋形状条件 N≥512。tensor map キャッシュを同時設計 | `backend-cuda-tma-gemm-load-design.md` §10.8「再評価の仮説」 | なし（TMA は確定済み） | 本番構成と同一タイルで比較可能。N≥512 で Stage 1 の改善（最大 20.7%）が本番比でも出るか | A〜D | 初期は P-diag、合格後に形状分岐 |
| C2: producer/consumer 非対称レジスタ（warp specialization）＋TMA | 同 §2 F2・§7 | **`setmaxnreg` の受理・実行の確定が前提**（未確定なら着手不可） | 非同期オーバーラップの上振れ | A〜D | P-diag |
| C3: Stream-K の fixup 固定費削減。(a) co-residency 保証付きのカーネル内 fixup（cooperative launch。raw FFI の `unsafe` を伴うため #2127 の整理と security-auditor 監査が前提）／(b) cluster>1 と DSMEM が確定した場合のクラスタ内固定順序還元（クラスタ内 CTA は同時スケジュールされ、デッドロック要因を排除できる） | `cuda-gemm-tiled-pipeline-streamk.md` §6.8 | (b) のみ cluster・DSMEM | N=1024／2048 で wave 損失 11〜33% の回復余地。固定順序で決定性を維持 | A〜D（A は決定性・full タイル bit 一致。B-1 必須・fail>0 の扱いは承認事項） | P-diag |
| C4: 128×64 版 Stream-K | 同 §6.8 | なし | C3 の成否に従属。ゲート C が 64×64 で FAIL のため優先度低 | A〜D | P-diag |

**対象外**:

- tcgen05／TMEM／wgmma 系（§1 の 1）
- cluster multicast（1×1×1 のみ。#2122 で覆るまで）
- 3×TF32 の再興（ユーザー判断確定済み）
- f8f6f4・block-scaled 等の narrow-precision mma（精度契約・REQ-2 の変更を伴うため本 issue の範囲外。必要なら spec 側への提案）

いずれの候補も既存の cudarc `driver::sys` API（`cuTensorMapEncodeTiled`・`cuLaunchKernelEx`・cooperative launch）の範囲で設計し、新規依存は追加しない。

## 5. opt-in enable／disable パラメータ設計（AC5）

既存の 2 パターンから選ぶ。

- **P-diag（診断限定・本番非到達）**: `internal-diagnostics` feature 限定の `compile_*_variant`／`run_*`／`launch_*` を置き、計算本体の文字列を既存カーネルと共有して bit 同一を担保する（persistent #1346・Stream-K #1358・TMA #1975 と同型）。C1〜C4 の初期実装はすべてこれとする
- **P-prod（本番到達 opt-in）**: `crates/backend-cuda/src/precision.rs` の `AtomicU8` 方式（既定 OFF・fail-closed）。opt-in が OFF の間は候補経路へ入らず、ON かつ候補が失敗した場合は既存経路へ黙示的に落とさず型付きエラーで拒否する（P-prod の失敗時契約）。ゲート A〜D 全合格後にのみ検討する
- **結線形態**: 合格後は `select_tiled_f32_kernel`／`select_tiled_pipeline_handle` の形状条件分岐（`TILED_PIPELINE_128X64_PRODUCTION_ENABLED` と同型の `const` スイッチ＋N／K 閾値）を第一候補とする。**facade への新規公開 API は本 issue でも後続の初期実装でも追加しない**（公開 API 非破壊のガードレール。追加が必要になればユーザー承認）
- **失敗時契約（選択方式ごと）**: (1) 診断入口（P-diag）と P-prod の明示 opt-in 経路は、コンパイル失敗・整列不成立を型付きエラーで拒否し、既存経路へ黙示フォールバックしない。(2) 形状条件による本番の自動選択（`const` スイッチ＋N／K 閾値。明示 opt-in ではない）に限り、コンパイル失敗・整列不成立時は既存の cp.async pipeline／classic へフォールバックする（既存の `if let` fail-closed 方針と同型。条件外形状は従来経路のまま）。両者を混在させず、結線 issue で採用する方式を 1 つ明記する

## 6. 本番結線の判断フロー

事前登録ゲートは既存と同形とする。

- **A**: 候補の種別で判定を分ける
  - 計算順序（K 連鎖）を維持する候補（C1・C2 等）: bit 一致（同タイルの cp.async 版 vs 候補。端あり・転置 4 パターン）
  - K 連鎖を分割する候補（Stream-K の C3・C4）: 残タイルは非 Stream-K 版と bit 同一にならない（`cuda-gemm-tiled-pipeline-streamk.md` §5）ため bit 一致は要求しない。既存の決定性検査（`streamk_repeated_launch_is_deterministic` 等。再実行で bit 一致）と、分割されない full タイルの非 Stream-K 版との bit 一致を適用する。残タイルは下記ゲート B-1 の複合判定（承認済み方式）で判定する
- **B**: parity 非後退。2 段で判定する
  - B（既存回帰）: `tests/parity_nonregression.rs` 等の既存テストで 0 fail
  - B-1（Stream-K 候補 C3・C4 で必須）: 候補自身の残タイルを CPU 参照実装と複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で比較し、全行 `fail_count == 0`。`fail_count > 0` の行が 1 件でもあれば B-1 は FAIL とし、baseline 方式（`ParityBaseline`）への変更・tolerance の変更はユーザー承認を条件とする（承認なしに採用しない）
- **C**: GPU-only の純カーネル時間の 5 回中央値。N≥1024 のいずれかで ≥1.05 かつ全計測形状で後退なし
- **D**: 結線後の本番ディスパッチ非後退（framework-compare の gemm cuda を同一 HEAD の base／after で比較する。`backend-cuda-tma-gemm-load-design.md` §6 のゲート D と同型だが、判定条件は候補の結合順序で分ける）
  - K 連鎖を分割しない候補（C1・C2 等。ゲート A で bit 一致を要求できるもの）: base／after の checksum 完全一致と性能非後退の両方を満たすこと
  - K 連鎖を分割する候補（Stream-K の C3・C4）: 残タイルの結合順序が非 Stream-K 版と異なり bit 一致しない（ゲート A・B-1 で許容済み）ため、checksum 完全一致は要求しない。代わりに (1) base／after の出力を複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で比較して全行 `fail_count == 0`（ゲート B-1 と同じ承認済み方式。tolerance・baseline は変更しない。framework-compare の checksum は全要素和で要素単位の `fail_count` を得られず誤差相殺を見逃すため判定に使わない。base／after それぞれ同一の決定的シード入力で出力行列の全要素をファイル等へ保存し〈checksum 出力とは別に保存するハーネス側の追加が必要。ハーネス未対応なら本ゲートは未達扱い〉、同一形状・同一入力について要素単位で複合判定して不合格要素数を数える）、(2) after 側の同一入力での再実行 checksum が一致（決定性維持）、(3) 性能非後退、のすべてを満たすこと

**no-go**: A の不一致 1 件（C3・C4 は決定性検査の失敗または full タイルの bit 不一致 1 件）、B-1 の `fail_count > 0`（承認なき baseline 化は不可）、または C の後退 1 形状で REJECT（opt-in 維持）。判定基準は実測前に固定し、事後に変更しない。tolerance・baseline の変更が必要になった時点で停止し、ユーザー承認へ回す。

**フロー**: #2122 確定 → 候補の着手可否判定（依存項目）→ P-diag 実装 issue → GB10 実測 issue → ゲート A〜D → 本番結線 issue（P-prod／形状分岐）。

## 7. 本番結線スケジュール（起票候補・ユーザー承認待ち）（AC6）

TMA・`setmaxnreg`・Stream-K・persistent・cluster で open issue を検索したが、本 issue の候補に該当するものは存在しない。`.claude/rules/out-of-scope-tracking.md` に従い、ユーザー承認なしには起票せず、候補として記録する。親は #2121 配下を想定する。

| 起票候補 | dependsOn | 実施ゲート | 優先度 |
|---|---|---|---|
| test(backend-cuda): CTA→SM 分布プローブ（`%smid`・`%globaltimer`。§3.1） | なし | 診断のみ | 高 |
| test(backend-cuda): TMA N=256 判別実験（§3.3 の (a)〜(c)） | なし | 診断のみ | 高 |
| perf(backend-cuda): TMA Stage 2（128×64）opt-in 実装（C1） | 上記の判別実験 | A〜D | 中 |
| test(backend-cuda): Stage 2 の GB10 実測・採否 | C1 の実装 | C・D | 中 |
| test(backend-cuda): 3×TF32 累積意味論の判別実験（§3.4） | なし | 診断のみ | 低 |
| perf(backend-cuda): Stream-K の cooperative fixup（C3(a)） | #2127・security-auditor 監査 | A〜D | 中 |
| perf(backend-cuda): クラスタ内還元（C3(b)） | #2122（cluster・DSMEM の確定） | A〜D | #2122 次第 |
| perf(backend-cuda): warp specialization＋TMA（C2） | #2122（`setmaxnreg` の確定） | A〜D | #2122 次第 |

C1〜C3 は raw FFI（`CUtensorMap` の `DeviceRepr` 実装・`cuLaunchKernelEx`・cooperative launch）を伴いうるため、後続の実装 PR では security-auditor の監査を必須とする。

## 8. スコープ外

- 候補カーネルの実装・GB10 実測・A/B 計測
- Issue の起票
- facade 公開 API の追加
- tolerance・baseline の変更
- narrow-precision（f8f6f4・block-scaled）mma の採用検討
- `docs/spec` の編集

## 9. 出典

- `docs/cuda-tensor-core-design.md` §11・§13
- `docs/perf/sm121-device-attributes.md`
- `docs/cuda-streamk-decision.md` §2・§5・§6・「#1359 実測結果」
- `docs/perf/cuda-gemm-tiled-pipeline-streamk.md` §6
- `docs/perf/cuda-gemm-tiled-pipeline.md`「#1347」節
- `docs/perf/cuda-gemm-tiled-pipeline-persistent.md` §5.1
- `docs/backend-cuda-tma-gemm-load-design.md` §2・§6・§7・§10.8
- `docs/cuda-tf32x3-split-single-decision.md` §9
- `docs/perf/cuda-tensor-core-tolerance-tf32x3-gb10.md` §10・§11
- `docs/perf/lowlayer-diagnosis-2026-09-12.md`
- ログ: `docs/perf/logs/cuda-tiled-pipeline-streamk-1359/`・`docs/perf/logs/cuda-tiled-pipeline-persistent-1347/`・`docs/perf/logs/cuda-tma-stage1-1975/`・`docs/perf/logs/cuda-gemm-tf32x3-1356/`・`docs/perf/logs/lowlayer-diagnosis-2026-09-12/`
- `crates/backend-cuda/examples/gemm_tiled_pipeline_bench.rs`・`crates/backend-cuda/src/precision.rs`・`crates/backend-cuda/src/gemm.rs`
