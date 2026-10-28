# GB10 CPU GEMM の reuse>fresh 逆転の帰属（GB10 実測済み。2026-10-21）

イシュー #3187。対応表の GEMM（GB10 CPU）で N=256／512 が負けている（reuse の対最速比が
0.74／0.70）。0.11.0 の GB10 実測では reuse が fresh より遅い（N=256〜2048 の全点で
およそ 1.3〜1.9 倍）。reuse を fresh 並みにできれば判定行は算術上「負け」から「勝ちまたは
同等」へ変わる見込みである。本書はその原因を区間別に帰属するための診断基盤と、
`docs/perf/cpu-gemm-gb10-affinity-ab.md`（#2117）の A/B を同一セッションで実行する手順をまとめる。

## 1. 状態

| 項目 | 状態 |
|---|---|
| フェーズ分解テスト（`crates/facade/tests/cpu_gemm_reuse_fresh_diag.rs`） | 実装済み（CI 実行の bit 一致テスト 1 本 + 実機用 `#[ignore]` 1 本） |
| 事前登録の判定規則（`docs/perf/logs/cpu-gemm-reuse-fresh-3187/RULE.txt`） | 固定済み（実測前）。本実測で変更していない |
| 計測・集計スクリプト（`orchestrate.sh`／`aggregate.py`） | 実装済み（`aggregate.py --self-test` OK） |
| GB10 での 5 run 実測と帰属判定（AC1） | **実測済み（2026-10-21）。5 run すべて負荷ゲート通過・checksum 全 bit 一致・n_samples=80。結果は §6** |
| affinity A/B（#2117 の RULE.txt どおり 14 セル 5 round。AC2） | **実測済み（2026-10-21）。正式系列・verdict は REJECT。結果は §4** |
| `.rev-stamp` による計測ツリーの同定（AC3） | 実測済み。runner・before・after の 3 ツリーに `.rev-stamp`（HEAD `d56c64ba87e825e0fa6b6e0958c456214d9083f6`・dirty=0）を置き、ツリー指紋を Mac 側 `git archive` 展開物と突合した（`logs/cpu-gemm-reuse-fresh-3187/rev-stamp-verification-3187.md`）。`env_info.txt` の `rule_sha256` は RULE.txt と一致 |
| `GB10_AFFINITY_ENABLED`（AC4） | false のまま（`crates/backend-cpu/src/gb10_affinity.rs` 無変更。REJECT のため反転しない） |

実測ログは `docs/perf/logs/cpu-gemm-reuse-fresh-3187/gb10/`（5 run）・`gb10-minflt/`（補助 1 run）・
`docs/perf/logs/cpu-gb10-affinity-ab-2117/gb10/`（A/B）。実測コミットは `d56c64ba8`（本イシューの実装ブランチの起点）。

## 2. 仮説

| 記号 | 仮説 | 根拠・予測 |
|---|---|---|
| H-A | 結果蓄積: reuse では各反復の結果ノードが Tape に積まれ解放されず、毎回新しい出力バッファを確保する。fresh は反復ごとに Tape を drop して出力を解放する | reuse−fresh の差は結果バイト数にほぼ比例する（約 0.66〜0.74 ms／MB。`cpu-gemm-candle-gate-remeasurement.md` §16.2 の数値から算術）。first-touch のページフォールトと整合するが CPU では未検証 |
| H-P | ページ確保の機構: 解放済みブロックの再利用（glibc）対 mmap 新規確保 | `GEMM_REUSE_MINFLT=1` の補助 run で minflt／iter を N²×4／PAGESIZE と並べる（判定外） |
| H-R | readout 方式 | CPU の既定は `host_view` の借用。legacy 腕で差を観測する（補助） |
| H-C | checksum 側のキャッシュ・TLB の差 | `checksum` 区間差で観測 |

コード事実（読解。本 PR は本番コードを変更しない）:

- fresh（`bench-fandhe::run_gemm`）の計測窓は「tape 構築 → `tape.var`×2（N² コピー）→ `matmul` →
  readout → checksum」。`Tape` の drop は `elapsed` 取得後（窓の外）に起きる
- reuse（`run_gemm_reuse`）は同じ Tape で `matmul` → readout → checksum を繰り返し、結果ノードは
  Tape に積まれ続ける
- `Var::matmul` は eager（`materialize_fallible` → `matmul_forward` → `push_eager`）で、`matmul` 区間に
  カーネル時間が入る
- CPU GEMM の出力確保は `zeroed_output` → `take_zeroed_f32`。host arena（#2104）の回収は `Tape` の
  Drop／reset と一意所有の場合に限られるため、reuse の計測窓では arena が効かない（hit=0）と予測する

## 3. 計測方法

- 診断テストは bench-fandhe と同じ入力（`SEED_A = 0xA11CE`・`SEED_B = 0xB0B`）、warmup 20＋計測 20
  を 4 ラウンド（腕順をラウンドごとに反転）で 1 プロセスあたり 80 サンプル。独立 5 プロセス
- 腕: `fresh`／`reuse`／`fresh_hold`（Tape を保持し結果蓄積を fresh 側で再現する因果腕）／
  `fresh_arena`・`reuse_arena`（host arena ON）／`backend_drop`・`backend_hold`（`BackendOps::gemm`
  を直接呼び結果を捨てる／保持）／`fresh_legacy`・`reuse_legacy`（legacy readout）
- 判定規則・ゲート・ドリフトガードは RULE.txt が正（本書で二重管理しない）
- 実行: `bash docs/perf/logs/cpu-gemm-reuse-fresh-3187/orchestrate.sh gb10` →
  `python3 docs/perf/logs/cpu-gemm-reuse-fresh-3187/aggregate.py <dir> > <dir>/aggregate.md`
- 補助（ページ確保）: `GEMM_REUSE_MINFLT=1 bash .../orchestrate.sh gb10 --out <別ディレクトリ>`
- 0.11.0 境界との差は `git diff v0.11.0..<sha> --stat -- crates/{backend-cpu,autodiff,tensor-core,facade}/src`
  で示す（`cpu-gemm-candle-gate-remeasurement.md` §16.1 と同方式）

## 4. affinity A/B（#2117）

判定規則は `docs/perf/logs/cpu-gb10-affinity-ab-2117/RULE.txt`、実行は
`scripts/bench/framework-compare/run_ab_gb10_affinity_cpu.sh`（ラベル `2117-gb10-3187`）。
結果の全 14 セルは `cpu-gemm-gb10-affinity-ab.md` に記入した。

- 系列: 正式（`series=official`・`precheck=assert`）。5 round すべて `gate=pass gate_ok=1`
  （`logs/cpu-gb10-affinity-ab-2117/gb10/load_gate-2117-cpu-2117-gb10-3187.log`）
- 機構発火: before は `enabled=false`・`pool_active=false`、after は `enabled=true`・`env_override=false`・
  大コア 10 個（`[5,6,7,8,9,15,16,17,18,19]`）・`pool_active=true`（`affinity-report-2117-*`）
- checksum: 14 セルすべてで完全一致
- 判定: **REJECT**。中央値 ratio（after/before）が 1.00 を超えたセルは 6 つ
  （gemm 512/fresh 1.0857・512/reuse 1.0341・1024/reuse 1.0043・2048/fresh 1.0449、train size=64 の
  fresh 1.0662・reuse 1.1115）。train reuse は 5 round すべてで 1.00 超。RULE どおり、ガードセルの除外や
  閾値の事後調整はしていない
- 処置セル（N=256）は fresh 0.8352・reuse 0.9979 で非後退、infer は 0.7470／0.7704 で改善しており、
  効く形状と悪化する形状が混在する（既定 ON にはできない）
- `GB10_AFFINITY_ENABLED` は変えていない（false のまま）。`GB10_AFFINITY_MAX_WORK` の再チューニングは
  RULE の対象外（別イシュー）

## 5. 申し送り

> #3226 の是正機構（既定 OFF のゲート）は実装済み。RULE.txt 固定済み・A/B は未実施（§8 を参照）。

| 項目 | 担当 |
|---|---|
| 逆転の是正（glibc の mmap 閾値・出力プール・Tape 結果の解放・prefault 等）。**§6 の結論から、是正の対象は「出力バッファの新規確保（first-touch のページフォールト）」。既存の host arena（#2104）は reuse で hit=0 のため効かない**。本書は列挙のみで実装しない | #3226 |
| affinity の既定反転 | #3227。**A/B が REJECT のため反転の前提が成立していない**（`GB10_AFFINITY_ENABLED` は false のまま）。#3227 は判断が必要 |
| 0.11.0 境界の同一セッション照合（framework-compare の `bench-fandhe` を fresh／reuse で各 5 run。record-only） | **本イシューでは対象外**（`orchestrate.sh` に boundary モードが無く、AC にも含まれない）。必要なら別イシュー |

## 6. 結果（GB10・2026-10-21）

出典: `docs/perf/logs/cpu-gemm-reuse-fresh-3187/gb10/aggregate.md`（5 run 中央値。単位 us）、
補助は `gb10-minflt/aggregate.md`。

| N | ratio (reuse/fresh) | G | 帰属 | rho_FH | rho_B | arena A/B |
|---|---|---|---|---|---|---|
| 256 | 1.434（fresh 179.9／reuse 258.0） | 79.4 | matmul（出力確保・結果蓄積）100% | 0.77（支持） | 0.72（支持） | fresh_arena hit=780・reuse_arena hit=0。host arena では是正できない（240.4 対 258.0） |
| 512 | 1.563（794.2／1241.6） | 459.1 | matmul 100% | 1.01（支持） | 0.94（支持） | 同上（1292.8 対 1241.6） |
| 1024 | 1.495（3105.5／4641.2） | 1370.9 | matmul 115% | 0.96（支持） | 1.03（支持） | 同上（4578.0 対 4641.2） |
| 2048 | 1.116（16091.6／17952.4） | 1881.8 | matmul 101% | 1.06（支持） | 0.97（支持） | 同上（17783.9 対 17952.4） |

総合: **逆転は 4 サイズすべてで再現（ratio 1.12〜1.56）。原因は matmul 区間の出力確保（結果蓄積）。**

- 区間差は matmul 区間にほぼ集中する（D_matmul 79.6／459.2／1581.5／1891.2 us に対し、D_readout は ±0.1 us 以内。
  D_checksum は N=1024 だけ -210.6 us で、fresh 側の checksum に二峰性〈757／542 us〉があるため。G の符号は変わらない）
- 因果腕: 結果を保持するだけで（fresh_hold）fresh の matmul が reuse の水準まで悪化し（rho_FH 0.77〜1.06）、
  Tape を介さず `BackendOps::gemm` の結果を保持するだけでも同じ（backend_hold。rho_B 0.72〜1.03）。
  tape／autodiff の機構ではなく、**結果を保持し続けることで毎回新しい出力バッファが必要になる**ことが原因
- ページ確保（補助・判定外。`gb10-minflt/aggregate.md`、pagesize 4096 B。minflt／iter を N²×4／PAGESIZE で割った比）:
  fresh と backend_drop は 0.00（解放済みブロックを再利用）、reuse は 1.37（N=256）・1.73・1.81・1.84、
  backend_hold は 0.60・1.00（N=512）・1.92・1.26。結果を保持する腕だけが毎反復ほぼ出力 1 枚分のページを
  新規に first-touch しており、機構は H-P（新規確保対解放済みブロック再利用）と整合する。
  比が 1 を超える点（1.4〜1.9）は出力以外のページ（packing 用領域等）の寄与を含みうるが、本計測では切り分けていない
- readout 方式（H-R）と checksum 側（H-C）は支持されない（D_readout ≈ 0。legacy 腕は読み出しコピー分が増えるだけで
  fresh／reuse の差は縮まない）
- arena A/B: reuse_arena の hit は 0（事前予測どおり）。`|reuse_arena − reuse| < 0.5×(reuse − fresh)` を 4 サイズとも満たし、
  「既存の host arena（#2104）では reuse の逆転を是正できない」を記録する

## 7. 限界

- 5 run は同一ホスト・同一セッション。GB10 の専有ゲート（load1<1.0）は 5 run すべて通過した（`load_gate_status.txt`: pass×5。補助 1 run も pass）。record_only の系列はない
- arena 統計は warmup を含むブロック全体の累計
- N=4096 の同型の帰属、M4 Max の正式系列は対象外（Issue は N=256〜2048、GB10 限定）

## 8. 是正（#3226）

### 8.1 設計

- 機構: `crates/autodiff/src/eager_release.rs`。`Var::matmul`（通常版）が出力を確保する**前**に、直前の
  `Op::MatMul` 結果が consumer を持たなければ `recompute = true` で値を手放し、`recycle_tensor_f32` へ渡す
  （arena OFF なら drop され glibc が再利用、ON なら次の `take_zeroed_f32` が hit）。確保の前に解放するので
  定常状態は 1 枠に収まる
- ゲート: `EAGER_MATMUL_RELEASE_DEFAULT_ENABLED = false`（既定 OFF・CPU 限定・新規 `unsafe` なし・公開 API 追加なし）。
  A/B の after 腕はこの 1 行だけを反転する。既定の反転は ADOPT 後の #3227
- bit 同一: 解放は checkpoint と同じ recompute 契約（`docs/autodiff-retain-graph-accumulate-decision.md` 追補・
  `docs/autodiff-checkpoint-design.md` 追補）。再利用バッファは `take_zeroed_f32` で全要素ゼロ埋め。
  `create_graph` は親ノードを `materialize_fallible` だけで読むためゲート ON でも `Ok` のまま
- テスト: 単体（`eager_release.rs`）・CI の bit 一致（`crates/facade/tests/cpu_gemm_reuse_release_diag.rs`。
  4 通りの全反復・NaN 番兵の残値非混入・backward／create_graph／checkpoint 共存）
- 採らなかった案: prefault／並列 first-touch（#1481 で REJECT・フォルトの場所を動かすだけ）・mmap 閾値／madvise
  （libc FFI と `unsafe` が要る）・arena 単独（結果が保持される間は返却が起きない）

### 8.2 事前登録規則と実測の状態

| 項目 | 状態 |
|---|---|
| 事前登録規則（`docs/perf/logs/cpu-gemm-reuse-release-3226/RULE.txt`） | 固定済み（計測より前にコミット） |
| GB10・Mac mini の 5 round A/B（RULE の ADOPT／REJECT 判定） | **実施済み（2026-10-23）。判定は REJECT**（§8.4）。既定は OFF のまま |
| 補助 diag（GB10・1 run・record-only・判定外） | 実施済み（下記）。`diag-gb10/run1.jsonl` |

### 8.3 補助 diag の結果（GB10・1 run・record-only。RULE の判定ではない）

`cpu_gemm_reuse_release_phases`（release ビルド・warmup 20 + 計測 20 反復の中央値・minflt は未取得〈-1〉）。
checksum bit は全腕・全サイズで一致。時間は 1 反復（matmul + readout + checksum）の中央値。

| N | fresh (ms) | reuse (ms) | reuse_release (ms) | reuse_release_arena (ms) |
|---|---|---|---|---|
| 256 | 0.176 | 0.350 | 0.201 | 0.177 |
| 512 | 0.842 | 1.397 | 0.860 | 0.817 |
| 1024 | 3.673 | 4.448 | 3.262 | 3.148 |
| 2048 | 16.108 | 18.385 | 16.856 | 16.516 |

解放により reuse は fresh 並みへ戻る方向（N=256〜512 で reuse 比 0.57〜0.62、N=1024 は fresh より速い）。
単一 run・単一セッションのため後退セルの有無・ADOPT 判定は言えない。RULE の A/B（5 round・両機体）が
次の作業。Mac mini の train CPU（reuse が +0.169 ms）は本機構の予測対象ではない。

### 8.4 A/B 結果と判定（#3226・2026-10-23）

計測 sha `3c5549027edd58c15917432add08dfd2d34f9d6d`（before は main 相当・after は `on-arm.patch` の 1 行差分）。
GB10 は 14 セル・Mac mini は 12 セル、各 5 round・round ごとに起動順を反転。両機体とも 5/5 round が `gate_ok=1`
（GB10 load1 < 1.0・Mac mini load1 < 3.0）・precheck 通過（before は `released=false`・after は `released=true`）・
checksum 全一致。`.rev-stamp`・ツリー指紋・`rule_sha256` は `logs/cpu-gemm-reuse-release-3226/rev-stamp-verification-3226.md` で突合済み。
生ログと最終判定（`judge-3226-combined.md`）は `logs/cpu-gemm-reuse-release-3226/{gb10,macmini}/`。

**判定: REJECT**（RULE: 5/5 round で後退したセルがある）。後退セルは次の 2 つ。

| 機体 | セル | before 中央値 | after 中央値 | ratio | round 別 ratio |
|---|---|---|---|---|---|
| GB10 | gemm reuse N=4096（非処置ガード） | 124.84 ms | 138.01 ms | 1.106 | 1.098 1.107 1.102 1.063 1.094 |
| GB10 | train reuse size=64 | 0.825 ms | 0.860 ms | 1.042 | 1.036 1.036 1.045 1.085 1.027 |

処置セル（GB10 gemm reuse）は ADOPT 条件の ratio <= 0.95 を 4 セルとも満たした（5/5 round で ratio < 1.00）。

| GB10 gemm reuse | before (ms) | after (ms) | ratio | reuse/fresh（before → after） |
|---|---|---|---|---|
| N=256 | 0.364 | 0.202 | 0.554 | 1.818 → 1.018 |
| N=512 | 1.654 | 0.933 | 0.564 | 1.663 → 0.947 |
| N=1024 | 5.218 | 4.276 | 0.819 | 1.235 → 1.050 |
| N=2048 | 27.02 | 16.88 | 0.625 | 1.620 → 1.011 |

その他の中央値後退セル（5/5 ではない・判定に直接は効かないが ADOPT 条件の「非処置は ratio <= 1.00」は満たさない）:

- GB10: gemm fresh N=2048（1.001）・infer fresh（1.036）。
- Mac mini: gemm fresh N=256／512／1024（1.016／1.007／1.000 超）・gemm reuse N=256（1.067。round 別 0.996〜1.128）・infer fresh／reuse（1.003／1.004）・train reuse（1.006）。
  Mac mini の処置相当 reuse は N=1024／2048 が 0.958／0.949 で改善、N=512 は 0.991。

副次（判定外・記録のみ）:

- 解放後の GB10 reuse/fresh 比は上表のとおり 1.0 近傍（N=256／512／2048 は 1.02／0.95／1.01）へ戻った。是正機構としての効果は確認できたが、
  N=4096（出力 64 MiB）では解放した出力の再確保が毎反復の確保コストとして残り逆に遅くなる。ゲートの対象を出力サイズで絞る案は本 RULE の対象外
  （RULE を事後に変えずに別イシューで事前登録して測る）。
- Mac mini の train CPU: 本系列では reuse が fresh より速く（before 0.673 ms vs 0.696 ms）、先行の +0.169 ms 差は再現しなかった。
  本機構の予測対象でもない（train は step ごとに Tape を作り直す）。原因特定は未実施。
- minflt 補助 run（GB10・1 回・`gb10/minflt-run.txt`）: reuse の minflt/iter は N=512 で 925・N=1024 で 3169・N=2048 で 7719 に対し、
  reuse_release は 0.00（fresh 並み）。「minflt/iter が 0 に近づく」予測と一致。
- 既定は OFF のまま。REJECT のため #3227（既定反転）は進めない。

## 9. 常駐入力経路の逆転（#3646）

0.12.0 の同席計測（#3273）で、bench の `gemm --mode reuse` を常駐入力経路（`upload_resident` → `matmul_resident`。#3272）に
切り替えると、GB10 の CPU gemm だけ reuse が同 round の fresh より 1.23〜1.79 倍遅くなった
（`logs/framework-compare-0.12.0-remeasure/README.md`「GB10 CPU gemm の新旧差」）。旧条件（ホスト入力・Tape 経路）では
並んでおり、Mac mini は新旧とも並ぶ。本節は切り分けの設計と状態を記録する。

### 9.1 状態（先に結論）

**2026-10-28 に GB10 専有（compute プロセス 0 件）で取り直した正式系列で判定した**（`logs/cpu-gemm-resident-reuse-3646/gb10-excl/`。§9.7）。
RULE.txt（ブリッジ節のみ RULE.v2.txt）・閾値・帰属規則・`aggregate.py` は変えていない。6 系列すべて `excl=pass`・診断主系列の全 run が `gate=pass`・checksum bit 一致・各 judge exit 0。

先行の実測（`gb10/` ほか。PR #3669）は常駐 GPU プロセス 3 件がある機体で取ったため**参考記録**に据え置く（§9.3b）。両者の数値は近く、結論の向きは同じだった。

- **bench の逆転は GB10 で再現した**（専有系列: bench-fandhe の reuse iter_total / fresh = N=256〜4096 で 2.03・1.35・1.33・1.74・1.23。§9.7）
- **原因を特定できたのは N=4096 だけ**: 帰属は H1（`matmul_resident` の `download` = 出力 N² の新規確保＋コピー）。rho_DL 0.72（逆転幅 25.5 ms のうち `download` が約 18.4 ms）。
  rho_DROP 0.25・rho_K 0.01・rho_IN 0.00 は未達。minflt 補助では `download` が 16385 回/反復のページフォールト
- **N=256〜2048 は原因を特定できなかった**: 診断の `resident_public` は fresh より 2.1〜6.4% 遅いだけで、bench の逆転幅の 1/8〜1/20 しか写せず、ブリッジ v2 が fail
  （G_bench / G_diag = 8.1〜20.1）。帰属は「保留（ブリッジ fail）」。試した仮説（H1・H1b・H2/H3・K・H4 補助）の値と未検証の候補は §9.7・§9.5 に分けて記録
- **G-CPU-INF の分類: 自分たちの制約（REQ-1。条件つき）**。純カーネル差が D の 134〜192%、窓の構成差は負（§9.4）。条件は、相手経路（NVPL）の確認が RULE 固定手順
  （LD_DEBUG・perf）では取れず、追加確認の LD_PRELOAD shim によること。所有者はこの追加確認を認めた（2026-10-28。§9.4）。分類は「保留」へ読み替えない
- 修正は実装していない（方針提示のみ。§9.5）。tolerance・parity baseline・判定規則・本番コード・`aggregate.py` の閾値は変更していない
- 規則の変更は `RULE.v2.txt` の 1 件のみ（ブリッジ節の置換。bench-fandhe が `gemm --mode fresh --phases` を拒否するため RULE.txt のブリッジは実行不能だった。実測前に別コミットで固定）

### 9.2 コード上の事実（読解。実測ではない）

新腕の窓にだけ入る処理は 3 つ。カーネル本体（`gemm_blis_parallel`）と入力の寿命は両腕で同じ。

1. `matmul_resident` が `gemm_device` の後に呼ぶ `MemoryOps::download`（CPU では `data().to_vec()`。N² の新規確保＋単一スレッド memcpy）
2. `gemm_device` が返す出力 `DeviceBuffer` を、関数を抜けるときに窓内で drop すること（大きいサイズでは munmap）
3. `zeroed_output`（calloc 相当）の新規確保

旧腕は出力 Vec がそのまま `Tensor` になり `host_view` で借用されるためコピーがない。
計測ツリーの等価性: `git diff --stat v0.12.0 HEAD -- crates/backend-cpu/src crates/autodiff/src/optim/resident_input.rs crates/autodiff/src/optim/device_store.rs crates/facade/src/lib.rs crates/tensor-core/src/alloc.rs` は実装時点で空（診断は 0.12.0 の bench と同じコードを測る）。

### 9.3 仮説と予測（RULE.txt に事前登録）

| 記号 | 仮説 | 支持の条件 |
|---|---|---|
| H1 | `download` の追加確保とコピー（first-touch＋単一スレッド memcpy） | rho_DL ≥ 0.5 |
| H1b | `gemm_device` 出力の窓内解放（munmap・TLB shootdown） | rho_DROP ≥ 0.5 |
| H2・H3 | 入力の寿命・ページ配置 | rho_IN ≥ 0.5 |
| H4 | スレッド親和性 | 補助系列 taskset で ratio が 1.0 付近へ動くか（判定外） |

先行記録（#3187 §6・#3226 §8.4）は、出力の確保・解放パターンが GB10 の glibc・ページフォールト機構に当たることを示しており、
H1・H1b が最有力だが、これは先行記録からの見込みで本件の実測結果ではない。

### 9.3b 参考記録（非専有・PR #3669。GB10・2026-10-28。5 run 中央値。生値・run 別は `logs/cpu-gemm-resident-reuse-3646/` の各 `aggregate.md`・`bridge_judge.md`）

**常駐 GPU プロセス 3 件がある機体で取った参考記録。RULE の専有条件を満たさないため、判定・帰属には使わない。正式系列は §9.7（値は残し、書き換えない）。**


診断の正式系列（`gb10/aggregate.md`。単位 us）。

| N | fresh | resident_public | ratio | 再現 | G | rho_DL | rho_DROP | rho_K | rho_IN | 帰属 |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 167.9 | 162.9 | 0.970 | 非再現 | — | — | — | — | — | なし |
| 512 | 787.8 | 831.0 | 1.055 | 再現 | 43.2 | — | — | — | — | 保留（ブリッジ fail） |
| 1024 | 3054.7 | 3208.3 | 1.050 | 再現 | 153.7 | — | — | — | — | 保留（ブリッジ fail） |
| 2048 | 15951.8 | 16583.5 | 1.040 | 再現 | 631.7 | — | — | — | — | 保留（ブリッジ fail） |
| 4096 | 124751.6 | 151388.7 | 1.214 | 再現 | 26637.1 | **0.78** | 0.36 | 0.00 | 0.02 | **H1（download）支持** |

ブリッジ v2（`gb10-bridge/bridge_judge.md`。RULE.v2.txt。bench-fandhe `fandhe-ai =0.12.0`・単位 us）。

| N | bench fresh | bench reuse（iter_total） | bench ratio | G_bench | G_diag | G_bench / G_diag | 判定 |
|---|---|---|---|---|---|---|---|
| 256 | 196.2 | 334.8 | 1.71 | 138.7 | -5.0 | — | fail |
| 512 | 1033.8 | 1698.6 | 1.64 | 664.8 | 43.2 | 15.4 | fail |
| 1024 | 4140.2 | 6077.6 | 1.47 | 1937.4 | 153.7 | 12.6 | fail |
| 2048 | 16441.9 | 25865.6 | 1.57 | 9423.8 | 631.7 | 14.9 | fail |
| 4096 | 121372.3 | 151343.1 | 1.25 | 29970.8 | 26637.1 | 1.13 | **pass** |

読み取り（事実）:

- N=4096 は bench と診断の窓が一致し、逆転の 78% が `download`。`gemm_device`（カーネル＋確保）は fresh の matmul と同じ（113.0 ms 対 113.0 ms）で、
  窓内 drop を足しても +9.6 ms（rho_DROP 0.36）、入力の再アップロードは差を動かさない（rho_IN 0.02）。minflt 補助（1 run・判定外）では `download` が
  16385 回/反復、`gemm_device` と fresh の matmul が各約 38k 回/反復で、`download` の 20.9 ms は約 1.28 us/回のページフォールトに相当する
- N=256〜2048 は診断が写せていない。bench の reuse では `--phases` の matmul 区間だけで、診断の fresh の iter_total 全体より長い
  （N=256: 300.5 us 対 167.9 us）。つまり bench の reuse 窓の `matmul_resident` 呼び出し自体が、診断の `resident_public` の同区間より
  N=256〜2048 で 1.75〜2.33 倍遅い（300.5／128.8・1541.8／669.1・5531.3／2482.8・23690.3／13570.0 us）。N=4096 は同じ（142.7 ms 対 142.6 ms）。
  reuse の to_tensor・host_copy は 1 us 未満（`gb10-bridge/bridge_judge.md` の matmul 区間表）
- 補助系列（判定外）: tunables（`mmap_threshold`・`trim_threshold` を引き上げ）は ratio が N=256 で 0.945・N=512 で 1.045（<= 1.05）、
  N=1024・2048・4096 で 1.142・1.254・1.221（> 1.05。RULE の規則どおり「それだけでは説明できない」）。N=2048 の fresh が 20597 us（正式系列 15952 us）と
  大きく振れており、この系列は参考値。taskset（大コア集合 5-9,15-19）は ratio を 0.950・1.037・1.040・1.034・1.201 とほぼ動かさず、H4（スレッド親和性）は支持されない
- 否定・未達の仮説（N=4096）: H1b（窓内 drop。rho 0.36）・H2/H3（入力の寿命・配置。rho 0.02）・gemm_device 内の確保かカーネル（rho 0.00）

### 9.4 G-CPU-INF の分類（要件 2）

**正式系列（専有。§9.7）の分類: 自分たちの制約（REQ-1。条件つき）**。規則（RULE.txt 末尾）どおり、GB10 同一セッションの 5 run 中央値で D と成分を求めた
（`gb10-excl/gb10-gcpuinf/gcpuinf_judge.md`。単位 us）。

| fandhe の窓 | 窓 | D（対 PyTorch 窓 126.7） | 純カーネル差 | 窓の構成差 | 純カーネル差 / D | 構成差 / D |
|---|---|---|---|---|---|---|
| fresh 通常行 | 171.6 | +44.9 | +59.9 | -15.1 | 1.34 | -0.34 |
| reuse 通常行 | 161.7 | +34.9 | +66.2 | -31.3 | 1.90 | -0.90 |
| reuse --phases iter_total | 161.2 | +34.5 | +66.2 | -31.7 | 1.92 | -0.92 |

- 純カーネル差 = fandhe 層カーネル合計（#2105 診断の層別区間。fresh 177.1・reuse 183.4）- PyTorch mm 合計（l1_fwd 106.2 + l2_fwd 11.0 = 117.2）。D の 50% 以上（134〜192%）
- 窓の構成差は負で 50% に届かない。したがって「計測窓の定義」ではない
- 相手経路: `libtorch_cpu.so` は NVPL（`BLAS_INFO=nvpl`）に動的リンクされている（`blas_path.log` の `ldd`）。RULE 固定手順の LD_DEBUG は sgemm 系の解決先を拾えず、perf は
  `perf_event_paranoid=4` で拒否された（sudo は使っていない）。追加確認として LD_PRELOAD の観測専用 shim（RULE 固定手順に対する追加確認。セッション後・計測外・時間は測らない）で
  `torch.mm` 2 回が `sgemm_`（m=256,n=64,k=784／m=10,n=64,k=256）を 1 回ずつ呼び、転送先が `libnvpl_blas_lp64_gomp.so.0` であることを確認した
  （`gb10-excl/gb10-gcpuinf/blas_probe/probe_out.txt`。mm なしの import のみでは呼び出し 0 回。PR #3669 の出力と sha256 まで同一）。
  この分類が追加確認に依ることは条件として残す（RULE 固定手順のみなら「経路を確認できない」に当たり「保留」。所有者が追加確認を認めない場合は保留に読み替える）
- **所有者判断（2026-10-28）: 追加確認の shim を、相手経路の確認として認める。** 理由は、shim が `torch.mm` の実行時の呼び出し先を直接観測しており、シンボル解決を間接に見る
  LD_DEBUG より強い確認であること、観測だけで時間を測らず閾値・計測値・判定式に触れないこと、perf のために共有の GB10 で sudo によりカーネル設定を変えるのは割に合わないこと。
  分類は「自分たちの制約（REQ-1。条件つき）」のまま確定し、「保留」へは読み替えない。「条件つき」の表記は、確認手段が事前登録の外だったことを示すために残す。
  今後の RULE では、LD_PRELOAD の観測専用 shim を相手経路の確認手段として事前登録する
- 条件・限界は参考記録と同じ: (1) 層カーネルは bias／活性化融合込みで PyTorch は mm のみ（純カーネル差は過大になりうる）。(2) 層別区間は別プロセスの値。
  (3) PyTorch 窓の run 別は 114.4〜141.0 で D の 2 割を超える幅。(4) 「ハードの上限」を示す数値（理論 FLOPS 比など）は取っておらず、この分類はそれを否定も肯定もしない。
  D の縮小手段は本計測から因果的に断定しない
- 0.12.0 の正式値・tally は変えない（#3274 の範囲）

**参考記録（非専有・PR #3669）の分類**（値は残す。上の正式系列と向きは同じ）:


**分類: 自分たちの制約（REQ-1。条件つき）**。規則（RULE.txt 末尾）どおり、GB10 同一セッションの 5 run 中央値で D と成分を求めた（`gb10-gcpuinf/gcpuinf_judge.md`。単位 us）。

| fandhe の窓 | 窓 | D（対 PyTorch 窓 122.9） | 純カーネル差 | 窓の構成差 | 純カーネル差 / D | 構成差 / D |
|---|---|---|---|---|---|---|
| fresh 通常行 | 173.6 | +50.7 | +68.9 | -18.2 | 1.36 | -0.36 |
| reuse 通常行 | 159.3 | +36.4 | +73.2 | -36.8 | 2.01 | -1.01 |
| reuse --phases iter_total | 164.5 | +41.7 | +73.2 | -31.5 | 1.76 | -0.76 |

- 純カーネル差 = fandhe 層カーネル合計（#2105 診断の層別区間。fresh 177.8・reuse 182.1）- PyTorch mm 合計（l1_fwd 98.0 + l2_fwd 10.9 = 108.9）
- 純カーネル差が D の 50% 以上（136〜201%）。窓の構成差は負で 50% に届かない。したがって「計測窓の定義」ではない
- 相手経路の実行時確認: `libtorch_cpu.so` は NVPL（NVIDIA の閉源 BLAS。`BLAS_INFO=nvpl`）に動的リンクされ、LD_PRELOAD の観測専用 shim で PyTorch の `torch.mm`
  （64×784 @ 784×256 と 64×256 @ 256×10）がどちらも `sgemm_`（m=256,n=64,k=784／m=10,n=64,k=256）を 1 回ずつ呼び、転送先が
  `libnvpl_blas_lp64_gomp.so.0` であることを確認した（`gb10-gcpuinf/blas_probe/probe_out.txt`。mm なしの import のみでは呼び出し 0 回）。
  perf は `perf_event_paranoid=4` で使えなかった（未確認）。よって「相手経路が閉源ライブラリと実行時に確認できた」条件は満たす
- 条件・限界: (1) fandhe 層カーネルは bias／活性化を融合した層の区間で、PyTorch 側は mm のみのため、純カーネル差は mm 以外の分だけ過大になりうる。
  (2) 層別区間は別プロセス（#2105 診断）の値で、窓の値と同一プロセスではない。(3) PyTorch の窓は bench_py と同一プロトコルの infer（run 別 120.4〜138.9）で、
  5 run 間の幅が D の 2 割を超える。(4) 「ハードの上限」を示す数値（GB10 の理論 FLOPS 比など）は本件で取っておらず、この分類はそれを否定も肯定もしない。
  分類は G-CUDA-INF（cuBLAS）の先例と `loss-attribution-matrix.md` §10.5 の定義に揃えた規則による条件つきの分類であり、D の縮小手段（自作カーネルの改善・固定費の削減等）については本計測から因果的に断定しない（fandhe 側は bias／活性化を含む層区間、PyTorch 側は mm のみで、層処理の差や計測条件の差を純 GEMM 差から分離できていないため）。また専有条件未達のため参考記録である（§9.1）
- 0.12.0 の正式値・tally は変えない（#3274 の範囲）

### 9.5 修正方針の提案（要件 3。#3646 では実装していない）

専有の正式系列（§9.7）で H1（`download`）を支持するのは N=4096 だけ（rho_DL 0.72）。提案はその範囲に限る。修正 issue の起票・実装は所有者の承認後で、本 PR では行わない。

**所有者判断（2026-10-28）**: 次の 2 件を #3018 Phase 14（#3673）として起票した。#3671 → #3672 の順に直列で進める。

- #3671: N=256〜2048 で、bench の reuse 窓が診断より遅い原因の切り分け（下記の未検証の候補）。先に行う。原因が bench の計測窓の側にあると直し方が (c) に変わり、(a) では負けセルの大半が残るため
- #3672: (a) の実装と A/B。形は `MemoryOps` に「所有権ごと渡す版の `download`」を足すもの（下記「保守性の判断」の前者）。既定の実装は今のコピー版に委ね、CPU バックエンドだけが上書きする。
  facade・autodiff に CPU 専用の分岐は置かない。A/B は GB10 の専有と Mac mini で、別 RULE を事前登録して行う

**候補**

- (a) CPU の `matmul_resident` で `gemm_device` の出力 Vec を `Tensor` へ所有権ごと移す（`download` のコピーと新規確保を除く。bit 同一）。N=4096 で最大 20.9 ms（診断の `download` 区間）を削れる見込み。ただし `Tensor` が出力バッファを抱え続けるため、解放のタイミング（窓内 drop。rho_DROP 0.36）が `download` の分だけ移る可能性がある。見込みは実測前の読みで未検証
- (b) 出力バッファの再利用（store 側のプール）
- (c) bench 側の窓の見直し（判定規則・D-C の範囲に関わるため所有者判断）
- (d) N=256〜2048 は別途、診断が写していない費用の切り分けが先（下記）。(a) では説明できない

**性能規則の結論**（ADOPT／REJECT）: 今回は出さない。(a)(b) は将来の A/B を別 RULE で事前登録し、N=4096 の `download` 区間と bench の iter_total を分けて判定する。
N=256〜2048 の逆転は (a) の対象外で、先に原因の切り分け診断（bench の reuse 窓の `matmul_resident` 呼び出しが診断の同区間より 1.75〜2.33 倍遅い理由）が要る。
未検証の候補は、bench が反復ごとに窓の外で行う全要素の検算（`verify_strict`・checksum）が次反復の出力ページ・キャッシュに及ぼす影響、
bench プロセス固有のアロケータ状態、反復ごとの `Tape::reset` の有無の差。いずれも本件では測っていない。

**保守性の判断**（性能規則の結論とは別）: (a) は `MemoryOps` へ所有権移動版 `download` を足すか、CPU バックエンドだけの分岐にするかを選ぶ必要がある。
前者は内部 trait の変更（公開 API の `facade` には出ない見込み・`unsafe` は不要の見込み）、後者は分岐が増えるが影響は CPU 内に閉じる。
どちらも `bit` 同一であることの確認（既存の parity テスト）を伴う。#3646 の時点では採否の結論を出していない。所有者は前者を選んだ（2026-10-28。上記・#3672）。

### 9.6 限界

- 参考値で H1 を支持できたのは N=4096 だけ（帰属は保留）。N=512〜2048 の診断 ratio は 1.04〜1.06 で、bench の 1.47〜1.64 を説明できず、N=256 は診断が非再現。ブリッジ v2 は fail
- 診断の `resident_*` 腕は内部 API（`MemoryOps::download` 等）を使い、bench の公開 API の窓と同一ではない。ブリッジ v2 で一致を確認できたのは N=4096 のみ。
  N≤2048 の不一致の原因は未特定（§9.5 の候補は未検証）
- 先行の実測（§9.3b。PR #3669）は GPU 常駐プロセス 3 件がある機体で参考記録。正式判定は専有系列（§9.7）だけで、そこでは compute プロセス 0 件を計測の前・途中（30 秒間隔 173 回）・後で記録した（`excl_check.py`）
- 補助系列 tunables の N=2048 は fresh が大きく振れており参考値。minflt は 1 run
- G-CPU-INF は fandhe の層別区間が別プロセスの値・PyTorch が mm のみ・PyTorch 窓の run 間幅が大きい、という限界つきの規則適用（§9.4）。perf は使えなかった
- 旧腕（Tape 経路 reuse＋reset）は本診断に含めていない
- RULE.v2.txt によるブリッジ節の置換は実測前に固定したが、判定式（`G_bench`／`G_diag` の比 0.5〜2.0）は bench が出す値に合わせて新設したもので、RULE.txt の「matmul 区間差」式を実測していない
- 専有系列の限界: (1) 専有判定は compute-apps（`nvidia-smi --query-compute-apps`）の 0 件であり、compute 以外の GPU 利用や他ユーザーの CPU 負荷は負荷ゲート（load1 < 1.0）が見る範囲に限る。
  (2) 機体は 1 台・1 セッションで、機体間・セッション間の再現は取っていない。(3) perf が使えず、LD_DEBUG も sgemm 系の解決先を拾えなかったため、G-CPU-INF の相手経路は追加確認の shim に依る。
  (4) N=256〜2048 の帰属は規則どおり保留で、§9.5 の候補は未検証

### 9.7 専有の正式系列の結果（GB10・2026-10-28。5 run 中央値。生値は `logs/cpu-gemm-resident-reuse-3646/gb10-excl/`）

**位置づけ**: 3 回目の reopen（所有者判断 2026-10-28）に従い、compute プロセス 0 件の GB10（`<cuda-node>`）で独立に取り直した。
同定は `rev-stamp-verification-3646.md`「専有系列」節（`.rev-stamp` `555a445a…`・ツリー指紋 5e8b85d2… が Mac 側と一致・RULE の sha256 が不変）。
専有は `session-scripts/excl_status.txt` の 6 系列すべて `excl=pass`（`gpu_poll.log` 173 件・境界 12 件が `rc=0 apps=0`）。

診断の正式系列（`gb10-excl/gb10/aggregate.md`。単位 us）。

| N | fresh | resident_public | ratio | G | rho_DL | rho_DROP | rho_K | rho_IN | 帰属 |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 184.9 | 196.8 | 1.064 | 11.9 | — | — | — | — | 保留（ブリッジ fail） |
| 512 | 1181.5 | 1256.5 | 1.064 | 75.0 | — | — | — | — | 保留（ブリッジ fail） |
| 1024 | 4251.5 | 4341.1 | 1.021 | 89.6 | — | — | — | — | 保留（ブリッジ fail） |
| 2048 | 16611.3 | 17247.8 | 1.038 | 636.5 | — | — | — | — | 保留（ブリッジ fail） |
| 4096 | 125350.8 | 150853.5 | 1.203 | 25502.6 | **0.72** | 0.25 | 0.01 | 0.00 | **H1（download）支持** |

ブリッジ v2（`gb10-excl/gb10-bridge/bridge_judge.md`。単位 us）。

| N | bench fresh | bench reuse（iter_total） | bench ratio | G_bench | G_diag | G_bench / G_diag | 判定 |
|---|---|---|---|---|---|---|---|
| 256 | 176.9 | 359.7 | 2.03 | 182.8 | 11.9 | 15.4 | fail |
| 512 | 1754.2 | 2363.1 | 1.35 | 609.0 | 75.0 | 8.1 | fail |
| 1024 | 4782.2 | 6371.4 | 1.33 | 1589.2 | 89.6 | 17.7 | fail |
| 2048 | 17353.4 | 30162.8 | 1.74 | 12809.4 | 636.5 | 20.1 | fail |
| 4096 | 124243.7 | 153183.5 | 1.23 | 28939.8 | 25502.6 | 1.135 | **pass** |

読み取り（事実）:

- N=4096: bench と診断の窓が一致（比 1.135）。`gemm_device`（113.7 ms）は fresh の matmul（113.2 ms）と同じ、窓内 drop を足しても +6.4 ms（rho_DROP 0.25）、
  入力の再アップロードは差を動かさない（rho_IN 0.00）、`backend_drop` のカーネル呼び出しは 114.2 ms。`download` 区間は 19.7 ms で、
  `gemm_device_download` 133.0 ms と `gemm_device_hold` 114.6 ms の差は 18.4 ms（rho_DL の分子）。minflt 補助（1 run・判定外）で `download` は 16385 回/反復、
  `gemm_device` と fresh の matmul は各約 37.8k 回/反復で、`download` の 19.7 ms は約 1.2 us/回に相当する
- N=256〜2048: bench の reuse `--phases` の matmul 区間は診断の `resident_public` の同区間より 1.63〜2.04 倍長い
  （321.8／157.9・2168.9／1077.4・5827.2／3582.4・28000.0／14224.7 us）。N=4096 は同じ（144.6 ms 対 142.2 ms）。reuse の to_tensor・host_copy は 2 us 未満
- 診断の N=256 ratio は専有系列で 1.064（参考記録では 0.970 で非再現）。N=256 は fresh の run 間幅が大きく（177.2〜209.3 us）、中央値の差 11.9 us はその幅に収まる。原因は特定していない
- 補助系列（判定外。各 `aggregate.md`）: tunables（`mmap_threshold`・`trim_threshold` を引き上げ）は ratio が N=256〜4096 で 0.847・1.094・1.023・1.244・1.196。
  RULE の規則では N=256・1024 が <= 1.05 で、N=512・2048・4096 は 1.05 超（N=2048 の fresh が 20643 us と振れており参考値）。N=4096 は 1.196 のまま「それだけでは説明できない」。taskset（大コア集合）は 0.968・1.039・1.075・1.027・1.188 で
  ratio を動かさず、H4（スレッド親和性）は支持されない。minflt は 1 run（ratio 0.620・1.157・1.010・1.029・1.204）
- 否定・未達の仮説（N=4096）: H1b（窓内 drop。rho 0.25）・H2/H3（入力の寿命・配置。rho 0.00）・`gemm_device` 内の確保かカーネル（rho 0.01）・H4（taskset）

**要件 1 の整理（特定できたこと・できなかったこと）**

| 範囲 | 結果 |
|---|---|
| N=4096 | 特定: H1（`download` の追加確保＋コピー） |
| N=256〜2048 | 特定できず。試して否定・未達: H1（診断の `download` は 6〜613 us で bench の差を説明できない）・H1b・H2/H3・K・H4。未検証の候補は §9.5（bench の `verify_strict`・checksum が次反復へ及ぼす影響、bench プロセス固有のアロケータ状態、`Tape::reset` の有無） |

参考記録（§9.3b。非専有）との差: 向き・結論は同じ（N=4096 のみ H1 支持、N≤2048 ブリッジ fail）。rho_DL は 0.78→0.72、rho_DROP は 0.36→0.25。数値の細部の差は
機体の状態差を含みうるため、どちらが「真」かはここでは断定しない。G-CPU-INF の分類も同じ（§9.4）。

## 10. N=256〜2048 の窓の差の切り分け（#3671）

### 10.1 状態（先に結論）

**GB10 の専有の正式系列を実測した（2026-10-28・`<cuda-node>`・compute プロセス 0 件）。判定は機械的で、N=1024 は `bench_tunables` が支持され、
N=256・512 は「特定できず」、N=2048 は bridge_v2 が pass して現象が再現しなかった。** 6 系列（`diag`・`solo`・`base`・`tunables`・`headpatch`・`minflt`）は
すべて `excl=pass`、`aggregate.py`（`gb10`・`gb10-solo`）と `bench_judge.py` は `OFFICIAL: yes`、`steps.log` は `done.`（rc なし）で完走した
（同定は `logs/cpu-gemm-reuse-window-3671/rev-stamp-verification-3671.md`。固定コミット `302f6ae9…`・`dirty=0`・ツリー指紋が Mac 側と一致・RULE の sha256 が不変）。

0.12.0 の正式値と #3274 の tally は変えていない。規則・閾値（0.5〜2.0）・tolerance・parity baseline・bench の窓・本番コードも変えていない。
Linux x86・M4 Max の値では代替していない。修正の実装は本 Issue の対象外（§10.6）。

| 項目 | 状態 |
|---|---|
| 読解で確定した窓の差 | 窓の中は同じ。差は窓の外（bench の `verify_strict`）とプロセス構成（10.2） |
| `Tape::reset` | bench も診断も呼ばないため比べる差分なし（腕は作らない） |
| 診断側の腕・bench 側の腕の GB10 専有系列 | **実測済み**（10.4。専有 6 系列すべて pass） |
| bridge_v2 | N=256・512・1024 は fail（再現）／N=2048 は **pass（再現せず）**／N=4096（対照）は pass |
| 原因 | N=1024: `bench_tunables`（アロケータ）が支持。N=256・512: **特定できず**。N=2048: 再現せず帰属しない（10.5） |

### 10.2 コード上の事実（読解。実測ではない）

| 項目 | bench `measure_gemm_reuse_resident` | 診断 `resident_public`（#3646） |
|---|---|---|
| 窓の中 | `matmul_resident` → `readout_tensor` → `slice` → 全要素 checksum | `matmul_resident` → `contiguous` → `as_slice` → checksum（実質同じ） |
| 窓の外（反復ごと） | `validate_gemm_checksum`・`reference.verify_strict(out)`（参照 N² と出力 N² の全要素比較・単一スレッド）・drop | `record_checksum`（bit 比較のみ）・drop |
| `Tape::reset` | 呼ばない | 呼ばない |
| 常駐データ | `a_data`・`b_data`・`a_host`・`b_host`・`GemmReference.c`（N² の f32 が 5 本）。init より前に並列（scoped worker）の O(N³) 参照 GEMM（`RULE.v3.txt` で「単一スレッド」の誤記を訂正） | `a`・`b` と常駐コピーのみ |
| 反復構造 | 1 プロセス 1 腕 1 N（init 1 回＋warmup 19＋計測 20） | 1 プロセスで腕を交互に回す（各ブロック warmup 20＋計測 20） |
| ビルド | framework-compare workspace・registry `fandhe-ai =0.12.0` | 本体 workspace・path の HEAD・test harness |

§9.7 では fresh 自体も bench と診断で違う（bench fresh / diag fresh = 0.96・1.48・1.12・1.04）。プロセス全体に効く要因を常駐経路の原因として
数えないよう、reuse 側だけにある超過分 E(N) = (bench reuse / diag resident_public) ÷ (bench fresh / diag fresh) を記録する（判定には使わない）。

### 10.3 仮説・腕・予測（`RULE.txt`・`RULE.v2.txt` に事前登録）

| 仮説 | 腕（1 要因） | 支持の条件（#3646 ブリッジ v2 と同じ 0.5〜2.0。新しい閾値は作らない） |
|---|---|---|
| H-V 検算が次反復のキャッシュ・ページに及ぼす影響 | `rp_verify_touch`（窓の外で N² を全要素比較） | G_bench / G_X が 0.5〜2.0 |
| H-V の分解（空き時間のみ） | `rp_verify_gap`（同じ回数を L1 上で） | 同上 |
| H-A1 常駐量 | `rp_hold_footprint` | 同上 |
| H-A2 前処理の確保・解放履歴 | `rp_ref_compute` | 同上 |
| H-P 腕の交互実行・反復構造 | `rp_solo`（別系列） | 同上 |
| H-A3 bench プロセスのアロケータ | `bench_tunables` | G_Y / G_diag が 0.5〜2.0 |
| H-B ビルド成果物の差 | `bench_headpatch` | 同上 |

予測: H-V が主因なら `rp_verify_touch` が bench に近づく。N=4096 は 1 反復の GEMM（約 113 ms）に比べ検算の費用とキャッシュの影響が小さく、
ブリッジが崩れない（§9.7 の観測 1.135 と矛盾しない）。空き時間が主因なら `rp_verify_gap` も近づき、追い出しが主因なら `rp_verify_touch` だけが近づく。

### 10.4 正式系列の結果（GB10・2026-10-28。5 run 中央値。生値は `logs/cpu-gemm-reuse-window-3671/gb10-excl/`）

**位置づけ**: compute プロセス 0 件の GB10（`<cuda-node>`）で 1 本の切り離しセッション（hw-lock 取得後）として取得した。
専有は `session-scripts/excl_status.txt` の 6 系列すべて `excl=pass`（`gpu_poll.log` 135 件すべて `apps=0`）。負荷ゲートは diag・solo の全 run と
bench の 75 ケースすべて `gate=pass`。bench の `rc.log` は 3 腕とも 75 行・全行 `rc=0`。run 別の値（run 番号付き）は各 `aggregate.md` を正とする。

診断の正式系列（`gb10-excl/gb10/aggregate.md`。iter_total・単位 us。[ ] は 5 run の min–max）。

| N | fresh | resident_public | rp_verify_touch | rp_verify_gap | rp_hold_footprint | rp_ref_compute | rp_solo（`gb10-solo`） |
|---|---|---|---|---|---|---|---|
| 256 | 193.7 [156.6–216.3] | 173.6 [156.4–191.2] | 211.4 [159.6–213.2] | 170.0 [151.9–200.4] | 168.4 [155.5–176.7] | 156.5 [156.0–165.9] | 191.7 [155.1–199.5] |
| 512 | 1135.4 [1075.6–1255.1] | 1161.7 [1016.7–1294.9] | 1405.5 [1306.5–1413.2] | 1270.1 [1146.5–1459.5] | 1171.6 [1110.8–1263.5] | 917.9 [744.0–945.2] | 1159.0 [815.8–1547.4] |
| 1024 | 4325.0 [4065.4–4503.5] | 4493.8 [4319.8–4680.8] | 4617.9 [4137.6–4761.5] | 4359.2 [4164.2–4531.4] | 4271.9 [4227.0–4594.9] | 4202.1 [4007.5–4785.5] | 4172.6 [3825.1–4360.5] |
| 2048 | 16452.0 [16336.6–16553.9] | 21849.4 [21500.3–22304.7] | 25737.4 [25580.4–26161.6] | 21553.0 [21402.1–22002.6] | 21483.5 [21457.2–21689.4] | 21582.3 [21310.1–21932.8] | 16611.1 [16341.7–17534.3] |
| 4096 | 125358.1 [125219.7–126165.5] | 151215.5 [150033.5–152056.2] | 150624.5 [150190.3–151166.0] | 150660.7 [149806.5–151879.1] | 150379.8 [149486.5–150721.7] | 150767.5 [149981.1–151058.1] | 151489.3 [149321.1–151879.4] |

`bench_judge.py` の判定（`gb10-excl/bench_judge.md`。`OFFICIAL: yes`。単位 us。G は reuse − fresh。bridge_v2 は G_bench / G_diag が 0.5〜2.0 で pass）。

| N | bench fresh | bench reuse | G_bench | diag fresh | resident_public | G_diag | G_bench / G_diag | bridge_v2 | E(N)（記録のみ） | matmul 区間 / G_bench（記録のみ） |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 187.3 | 347.8 | +160.6 | 193.7 | 173.6 | −20.1 | −7.996 | fail | 2.072 | 1.895 |
| 512 | 1709.9 | 2356.4 | +646.5 | 1135.4 | 1161.7 | +26.3 | 24.606 | fail | 1.347 | 3.350 |
| 1024 | 4699.7 | 6272.2 | +1572.5 | 4325.0 | 4493.8 | +168.8 | 9.314 | fail | 1.284 | 3.647 |
| 2048 | 17806.8 | 25802.9 | +7996.2 | 16452.0 | 21849.4 | +5397.4 | 1.481 | **pass** | 1.091 | 2.930 |
| 4096（対照・判定外） | 126705.6 | 152389.7 | +25684.1 | 125358.1 | 151215.5 | +25857.4 | 0.993 | pass | 0.997 | 5.599 |

診断側の腕 X（H-V・H-A1・H-A2・H-P）。支持の条件は G_bench / G_X が 0.5〜2.0。G_X は X.iter_total − 同じ診断プロセスの fresh（us）。

| N | rp_verify_touch G_X・比・支持 | rp_verify_gap | rp_hold_footprint | rp_ref_compute | rp_solo |
|---|---|---|---|---|---|
| 256 | +17.8・9.040・no | −23.7・−6.766・no | −25.3・−6.343・no | −37.2・−4.316・no | −2.0・−82.250・no |
| 512 | +270.0・2.394・no | +134.7・4.800・no | +36.1・17.887・no | −217.5・−2.972・no | +23.6・27.430・no |
| 1024 | +292.9・5.368・no | +34.2・46.033・no | −53.2・−29.581・no | −122.9・−12.799・no | −152.4・−10.316・no |
| 2048 | +9285.4・0.861・**yes** | +5101.0・1.568・**yes** | +5031.5・1.589・**yes** | +5130.3・1.559・**yes** | +159.1・50.251・no |
| 4096（対照） | +25266.4・1.017・yes | +25302.6・1.015・yes | +25021.6・1.026・yes | +25409.3・1.011・yes | +26131.1・0.983・yes |

bench 側の腕 Y（H-A3・H-B）。支持の条件は G_Y / G_diag が 0.5〜2.0。G_Y は同じ腕の reuse − fresh（us）。

| N | bench_tunables G_Y・比・支持 | bench_headpatch G_Y・比・支持 |
|---|---|---|
| 256 | −0.6・0.030・no | +190.5・−9.489・no |
| 512 | −69.9・−2.661・no | +738.6・28.112・no |
| 1024 | +232.6・1.378・**yes** | +1603.6・9.499・no |
| 2048 | +1433.7・0.266・no | +12563.2・2.328・no |
| 4096（対照） | +27625.2・1.068・yes | +27019.2・1.045・yes |

記録のみの量（判定に使わない。単位 us）:

| N | bench_base reuse の中央値 | bench_tunables reuse | bench_headpatch reuse | bench_base の reuse − `--phases` の iter_total | bench の checksum 区間（base） |
|---|---|---|---|---|---|
| 256 | 352.1 | 164.8 | 362.0 | +4.3 | 33.6 |
| 512 | 2465.7 | 1686.0 | 2527.4 | +109.3 | 188.9 |
| 1024 | 6410.3 | 4677.4 | 6327.7 | +138.1 | 540.8 |
| 2048 | 25265.4 | 17978.5 | 27935.0 | −537.5 | 2129.5 |
| 4096 | 152758.5 | 152084.8 | 153021.7 | +368.8 | 8594.9 |

補助系列 `minflt`（1 run・判定外。`OFFICIAL: no` が正常）は `gb10-excl/gb10-minflt/`（最大 RSS 685172 KiB・minor page fault 24423265 回。診断全体のプロセスの値で、腕・N の別は付かない）。

### 10.5 性能規則の結論（`bench_judge.md` の結論を機械的に転記。因果の断定ではない）

| N | 結論 |
|---|---|
| 256 | **特定できず**（どの腕も支持されない。各比は 10.4） |
| 512 | **特定できず**（どの腕も支持されない。各比は 10.4） |
| 1024 | 支持された腕 = **`bench_tunables`**（H-A3。G_Y / G_diag = 1.378。交互作用は断定しない） |
| 2048 | **現象が再現しない**（bridge_v2 = pass）。帰属は行わない |
| 4096 | 対照（判定外）。bridge_v2 = pass（§9.7 の download への帰属と矛盾しない） |

読み取り（事実の整理。規則の結論ではない）:

- bench の `reuse` と `fresh` の差 G_bench は N=256〜2048 で +160.6・+646.5・+1572.5・+7996.2 us と正で、診断の `resident_public` − `fresh`（−20.1・+26.3・+168.8・+5397.4 us）より N=256〜1024 で桁違いに大きい。N=2048 だけは比が 1.481 で、診断側でも `resident_public` が `fresh` より 5.4 ms 遅く現象が再現した（§9.7 の同 N の ratio 1.038 とは異なる値で、系列間の差は本 Issue で原因を調べていない）
- 診断の窓の外を再現する腕（`rp_verify_touch`・`rp_verify_gap`・`rp_hold_footprint`・`rp_ref_compute`・`rp_solo`）は N=256〜1024 のいずれでも bench の差の 0.5〜2.0 倍に入らない。これらの腕の iter_total は bench reuse の 0.39〜0.74 倍にとどまる。N=2048 で 4 腕が「支持」になったのは、N=2048 では診断の `resident_public` 自体が bench と同じ向きの差を示した（比 1.481）ことによる。`rp_solo` だけが N=2048 で fresh 並み（G_X +159 us）に戻る（支持 no）
- `rp_verify_touch` と `rp_verify_gap` の組み合わせ（キャッシュとページ／空き時間の分解）は記録のみ。どちらも N=256〜1024 で支持されないため、検算そのものを本因と読める根拠はこの系列にない（因果は断定しない）
- `bench_headpatch`（path patch）は bench_base と同じ遅さ（reuse 中央値 362.0・2527.4・6327.7・27935.0 us）で、H-B は支持されない。v0.12.0 との差は `src` にはなく（`rev-stamp-verification-3671.md`）、registry 版か path 版かでは変わらない
- `bench_tunables`（`GLIBC_TUNABLES`）では bench の reuse 中央値が fresh 並みまで下がる（記録のみの表: 164.8・1686.0・4677.4・17978.5 us。bench_base は 352.1・2465.7・6410.3・25265.4 us）。ただし機械判定の支持は N=1024 だけである。N=256 は G_diag が負（−20.1 us）で比が意味を持たず、N=512 は G_Y が負（−69.9 us）で比が −2.661、N=2048 は G_Y/G_diag = 0.266 で閾値外のため「支持なし」とされた。**この表は記録であり、規則の結論ではない。** N=4096 では tunables でも reuse が fresh より 27.6 ms 遅いままで、§9.7 の download（H1）への帰属と矛盾しない
- 窓の外の空き時間（`gap` 区間）は `rp_verify_touch` と `rp_verify_gap` で N=256・512 はほぼ同じ（7.7／7.1 us・104.7／102.4 us）。N=2048・4096 では touch のほうが長い（1657.7 vs 457.4 us・3553.0 vs 1811.9 us）。空き時間が揃っているかの記録で、判定には使わない
- 規則どおり、複数の腕が支持された場合の交互作用や、「支持なし」の N の原因は断定しない。本結論は「N=256・512 の原因は特定できていない」を含む

### 10.6 保守性の判断（性能規則の結論とは別項）

修正は本 Issue では実装しない。所有者へ示す方針は次のとおり（提案。**承認は得ていない**）。

1. **bench の計測窓の側が原因という結論は得られていない**。窓の外の検算を診断側に再現した腕は N=256〜1024 の差を再現せず、窓の外の検算・保持量・前処理・プロセス構成のどれも、この系列では帰属先にならなかった。よって §9.5 (c)（窓の見直し）を、この結果だけで所有者判断へ上げる根拠は弱い
2. 一方、`GLIBC_TUNABLES` を付けた bench の reuse が fresh 並みに下がる事実（記録のみ。N=1024 だけが機械判定で支持）は、アロケータ（mmap／trim の閾値）が候補であることを示す。ただし N=256・512 は機械判定が成立せず、N=2048 は閾値外で、規則の結論としては「N=1024 のみ支持・N=256・512 は特定できず」である。**次の切り分けは、bench-fandhe の診断用ノブ（例: 反復内の検算 `verify_strict` をループの後へ移す腕・`GLIBC_TUNABLES` を N 別に当てる腕）を追加して N=256・512 を直接測る**ことを提案する（実装は別 Issue。本 Issue では bench-fandhe のソースを変更していない）
3. N=4096 の download の修正（#3672）は本結果に依存しない（N=4096 の帰属は §9.7 のとおり H1。本系列の対照でも bridge_v2 pass・tunables でも差が残る）

### 10.7 限界

- 診断の `rp_verify_touch` は `verify_strict` の**メモリアクセスパターンの複製**であり、parity 契約ではない（bench-common へは path 依存できない）
- `Tape::reset` はノード数を公開 API で数えられないため、読解（reset 非呼び出し）と葉数不変の CI テストのみが根拠
- 腕は 1 要因ずつで、交互作用は断定しない。支持が複数ならすべて併記する
- 取得は `<cuda-node>` 1 台・1 回。系列間の差（N=2048 の diag の ratio が §9.7 と異なる点）は本 Issue では調べていない。別機体・別セッションでの再現は未実測
- N=256 は G_diag が負で、G_bench / G_diag・G_Y / G_diag の比が意味を持たない。判定器は規則どおり機械的に「支持なし」を返した
- bench の `bench_tunables` 腕は 1 つのリテラル（`glibc.malloc.mmap_threshold=33554432:glibc.malloc.trim_threshold=1073741824`）だけで、閾値別の感度は測っていない
- ノードに `.git` が無く、`env_info.txt` の `diff_stat_v0.12.0_head` は「未実測」（v0.12.0 との差は Mac で取り `rev-stamp-verification-3671.md` に記録）
- `rp_solo` の G_X は主診断系列の fresh を使う（RULE の定義どおり）

## 11. #3672: CPU の matmul_resident で download のコピーを除く

### 11.1 状態（先に結論）

- **実装は完了（PR #3677）。実機 A/B は 2026-10-28 に GB10 専有（`<cuda-node>`）と Mac mini（`<metal-node>`）の両機体で RULE どおりに実測し、性能規則の結論は REJECT**（`logs/cpu-gemm-download-owned-3672/judge-3672.md`。`ab_judge.py` の出力そのまま）
- REJECT の理由は、RULE の「5/5 run で後退（全 run で ratio > 1.00）したセルがある」に当たる非処置セルが 3 つあること。GB10 の診断 `resident_split` N=512（中央値 ratio 1.0743）、Mac mini の診断 `resident_public` N=256（同 1.0872）と診断 `gemm_device_download` N=4096（同 1.0021）。**処置 2 セルは大きく改善し（GB10 bench reuse N=4096 は ratio 0.7997、診断 `resident_public` N=4096 は 0.8220。ともに 5/5 run で < 0.95）、rho_REM は 1.28、checksum は全件一致**。ADOPT に必要な非処置セルの条件（全セル中央値 ratio ≤ 1.00）を満たしたセルは多いが、ノイズ級の > 1.00 が両機体に残った（`judge-3672.md` の「ADOPT を妨げた項目」に 18 件〈GB10 4・Mac mini 14〉を列挙）
- 事前登録規則は `logs/cpu-gemm-download-owned-3672/RULE.txt`（sha256 `03d4c5f4f3dd08d9f076bfa879ccb8ca9c983005bf5ea5e06663cb1f2ed793be`。実測前に単独コミットで固定）と `RULE.v2.txt`（sha256 `f07439fd88ee335028dee3d6a2bb1b495281099c4ab7a1d82fa804538a9d1ddc`。セル集合の解釈の固定。閾値は不変。計測部品のコミットより前に単独コミットで固定）
- 専有・ゲート: GB10 は 5 run 全てで load1 < 1.0、compute プロセス 0 件（poller 61 件・前後のスナップショット。`excl_check.py --series main` は `main excl=pass`）。Mac mini は 5 run 全てで load1 < 3.0。腕の同定（`.rev-stamp`・ツリー指紋・バイナリ sha256）は `logs/cpu-gemm-download-owned-3672/rev-stamp-verification-3672.md`

### 11.2 設計（所有者判断 §9.5 (a) の実装）

- `tensor-core::MemoryOps` に既定実装つき `download_owned(&self, DeviceBuffer<f32>)` を追加（既定は `download(&buffer)` へ委譲して drop。CUDA・Metal の挙動は不変。公開 API の非破壊拡張）
- `backend-cpu` の `CpuMemory` と `CpuBackendOps` の 2 か所で上書き。`CpuBufferHandle::take_data` が prepack キャッシュを先に破棄（#3200 の不変条件）して `Vec` を取り出し、`Tensor::new` へ移す。device・ダウンキャスト・長さの検査は `download` と同じで、異種ハンドルは `DeviceMismatch`
- `autodiff::matmul_resident` は `mem.download_owned(out)` に変えただけ。facade 公開面・CPU 専用分岐なし・`unsafe` なし
- `PooledMemory` は上書きせず既定実装（プール契約を保つ）

### 11.3 bit 一致の確認

- `memory::tests::download_owned_is_bit_exact_and_moves_vec`: NaN・±0.0・subnormal・±inf を含む入力で `download` と bit 一致、`Vec` のポインタ一致、解放計上が 0 へ戻る
- `ops::...cpu_backend_ops_download_owned_moves_vec_and_matches_download`: ホットパスの `CpuBackendOps` 経由でも同じ
- 空バッファ・異種ハンドル・非 CPU device の拒否、`take_data` の抜け殻、既定実装の委譲（tensor-core）
- 既存の `cpu_gemm_resident_reuse_diag.rs::cpu_gemm_resident_arms_match_public_path_bit_exact`（コピー経路の腕と公開経路の bit 一致を hard assert）が、変更後は旧経路と新経路の bit 一致検査として働く

### 11.4 事前登録（要約）

before = マージベース・after = PR head の 2 木を ABBA・5 run・N=256〜2048・4096。処置セルは GB10 の bench reuse と診断 `resident_public` の N=4096。ADOPT は処置 2 セルが中央値 ratio ≤ 0.95 かつ 5/5 run < 1.00・rho_REM ≥ 0.5・非処置セルが ≤ 1.00・checksum 全一致。詳細は RULE.txt を正とする。

### 11.5 結果（2026-10-28。5 run 中央値。生値・全セルは `logs/cpu-gemm-download-owned-3672/judge-3672.md`、機体別の生ログは `gb10/`・`macmini/`）

before = `a7aac2f02`（マージベース）、after = `f7ce20fc9`（PR head。tree は `e03f81957` と同一）。ratio = after / before の中央値（run ごとの ratio の中央値）。ABBA・5 run・独立プロセス。

**GB10（`<cuda-node>`・専有 pass）**

| セル | N | before (s) | after (s) | 中央値 ratio | run ごとの ratio |
|---|---|---|---|---|---|
| bench reuse iter_total（**処置**） | 4096 | 0.153438 | 0.122702 | **0.7997** | 0.8221 0.7997 0.7861 0.8109 0.7923 |
| 診断 resident_public iter_total（**処置**） | 4096 | 0.15118 | 0.124936 | **0.8220** | 0.8322 0.8187 0.8220 0.8181 0.8298 |
| bench reuse iter_total（非処置） | 256 | 0.000357577 | 0.000182137 | 0.5115 | 0.5115 0.4646 0.5335 0.5364 0.4287 |
| 同 | 512 | 0.00239146 | 0.00185485 | 0.7731 | 0.8438 0.7628 0.7381 0.7731 0.8793 |
| 同 | 1024 | 0.00637541 | 0.00442637 | 0.6675 | 0.8051 0.6675 0.7674 0.6261 0.6537 |
| 同 | 2048 | 0.0301476 | 0.0166738 | 0.5407 | 0.5235 0.5785 0.5407 0.5297 0.5652 |
| 診断 resident_public（非処置） | 256 | 0.000183712 | 0.00018104 | 1.0030 | 1.0023 1.0098 1.0972 1.0030 0.8847 |
| 同 | 512 | 0.00109195 | 0.0011259 | 1.0143 | 1.0143 1.0223 0.9976 0.9490 1.0606 |
| 同 | 1024 | 0.00419371 | 0.00388986 | 0.9341 | 0.9912 0.9341 1.0569 0.8240 0.7939 |
| 同 | 2048 | 0.017263 | 0.0164745 | 0.9543 | 0.9606 0.9541 0.9546 0.9348 0.9543 |
| 診断 resident_split（非処置・REJECT 条件） | 512 | 0.00112684 | 0.00125103 | 1.0743 | 1.1748 1.0062 1.0444 1.1613 1.0743 |

**Mac mini（`<metal-node>`・5 run とも load1 < 3.0）**

| セル | N | before (s) | after (s) | 中央値 ratio | run ごとの ratio |
|---|---|---|---|---|---|
| bench reuse iter_total（非処置） | 256 / 512 / 1024 / 2048 / 4096 | 各セルは `judge-3672.md` | | 0.9712 / 0.9828 / 0.9825 / 0.9794 / 0.9919 | |
| 診断 resident_public（非処置・REJECT 条件） | 256 | 0.000230708 | 0.00024875 | 1.0872 | 1.0732 1.0872 1.1031 1.1208 1.0679 |
| 診断 resident_public（非処置） | 4096 | 0.265233 | 0.263727 | 0.9952 | 0.9949 0.9923 0.9980 0.9952 0.9959 |
| 診断 gemm_device_download（非処置・REJECT 条件） | 4096 | `judge-3672.md` | | 1.0021 | 全 5 run で > 1.00 |

**rho（GB10 N=4096。`ab_judge.py` の出力）**: X_before 0.0370215 s・X_after 0.0116697 s・DL_before 0.0197884 s、**rho_REM = 1.2811（≥ 0.5）**。bench の iter_total 差から rho_BENCH = 1.5532、rho_OFFSET = −0.5532（相殺は起きず、窓の短縮は download 区間より大きい）。

**checksum**: 診断の iter_total（全 8 腕 × 5 N）と bench の iter_total（5 N）の checksum は、両機体で before・after の全 run が完全一致（不一致 0 件）。

### 11.6 性能規則の結論（`judge-3672.md` の結論を機械的に転記）

**REJECT**。RULE.txt の「5/5 run で後退（全 run で ratio > 1.00）したセルがある」に 3 セル（上記）が当たった。処置セルの条件（中央値 ratio ≤ 0.95 かつ 5/5 run < 1.00）と rho_REM ≥ 0.5 は満たし、checksum も全件一致したが、RULE の REJECT 条件が ADOPT の成否より先に適用される。規則は緩めていない。

事実の記録（因果の断定ではない）:
- 処置 2 セルは約 18〜20% 短縮（ratio 0.80〜0.82）。N=4096 の download 区間（DL_before 19.8 ms）を除くと X が 37.0 ms から 11.7 ms へ減った（rho_REM 1.28）
- GB10 の bench reuse は N=256〜2048 でも 0.51〜0.77 へ短縮した（処置セル外。RULE では非処置セル）
- REJECT 条件に当たった 3 セルのうち 2 つ（GB10 診断 `resident_split` N=512・Mac mini 診断 `resident_public` N=256）は小 N で 7〜9% の後退、Mac mini の診断 `gemm_device_download` N=4096 は ratio 1.0021（run 間のばらつきが小さく 5/5 が > 1.00 になった）。3 件とも本 Issue では原因を切り分けていない（計測ノイズの可能性を含む。RULE のとおり判定に使い、除外していない）

### 11.7 保守性の判断（11.6 とは別項）

**所有者判断（2026-10-29）: 変更（PR #3677 の `download_owned`）は main に残す**。性能規則の結論（11.6 の REJECT）は書き換えず、記録としてそのまま残す。残すのは、処置 2 セルの改善（約 18〜20%）・全 checksum 一致・既定実装による他バックエンドの無改修を踏まえた保守性の判断であり、規則の判定を覆すものではない。下の論点 2 つは Phase 15（#3682）で扱う: (1) は #3681（非処置セルを `matmul_resident` の経路で絞った規則を別に事前登録して再計測。#3672 の RULE と判定は変えない）、(2) は #3679（5/5 run で後退した 3 セルが再現する差かの確認）。N=256・512 の原因（§10）は #3680。

以下は判断前に記した提案（記録として残す）。

内部 trait への追加は 1 メソッド・CPU の上書きは 2 か所で、既定実装により他バックエンドは無改修。処置セルの改善（約 18〜20%）と全 checksum 一致を踏まえると、**変更を残すこと自体は保守性・性能の両面で妥当と考えるが、性能規則の結論は REJECT のままで、採否はユーザー判断**。判断のために論点を 2 つ挙げる。(1) RULE.v2 で「非処置セル = bench と診断の全腕の iter_total」と厳しい側に固定したため、Mac mini の小 N などノイズ級の差（中央値 ratio が 1.00 を 0.1〜9% 上回る）も ADOPT を妨げる（18 件）。規則の趣旨が「`matmul_resident` 経路の後退検出」なら、対象セルの絞り方は別 Issue で事前登録し直して再計測すべきで、事後に緩めない。(2) GB10 診断 `resident_split` N=512 と Mac mini 診断 `resident_public` N=256 の小 N 後退は再現性のある差の可能性があり（5/5 run で後退）、小 N の経路（#3671 の対象）として別途確認する価値がある。将来 Metal（unified memory）が同じ上書きの候補になりうるが提案のみ。

### 11.8 限界

- 腕の木は git archive の展開木で、ビルドは各ノードで 1 回ずつ（`build_ab_arms.sh`）。計測中のビルド並走はない。診断の反復数は `iters_for` の既定のまま
- 計測部品（`build_ab_arms.sh`・`orchestrate.sh`・`ab_judge.py`・セッションスクリプト）は計測より前にコミットした。例外として、計測後に `ab_judge.py` を 1 点修正した（判定に使わない区間行の `median_s` が 0.0 のとき入力不備にしていた過剰な検査を外した。閾値・判定式は不変。詳細は `rev-stamp-verification-3672.md`）
- RULE.v2.txt の `fixed-at-utc` は記入時の誤り（13:30Z と記載、実際は 14:00Z 頃のコミット）。計測開始（14:15Z）より前に固定した事実は不変（`rev-stamp-verification-3672.md`）
- `orchestrate.sh` の最終行の案内文は絶対パスを含む（回収時にマスクした）
- GB10 は専有 pass だが、診断・bench とも 1 機体 1 系列（5 run）。小 N の差は run 間のばらつきが大きく、本 Issue の対象（N=4096 の逆転）の判定には影響しないが、小 N の後退の有無は別途の確認が要る
- autodiff 内の他の `download(` 呼び出し、`PooledMemory`・Metal の `download_owned` 上書き、出力バッファのプール（§9.5 (b)）、bench 窓の見直し（§9.5 (c)）、N=256〜2048 の逆転（#3671）は本 Issue の対象外
