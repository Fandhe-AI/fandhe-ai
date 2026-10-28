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

2026-10-28 に GB10 で RULE.txt（ブリッジ節のみ RULE.v2.txt）どおり実測した（`logs/cpu-gemm-resident-reuse-3646/`。全 run が負荷ゲート通過・checksum bit 一致）。

- **bench の逆転は GB10 で再現した**（bench-fandhe の reuse iter_total / fresh = 1.25〜1.71。§9.3b）
- **原因を特定できたのは N=4096 だけ**: `matmul_resident` の `download`（出力 N² の新規確保＋コピー）が逆転幅の 78%（rho_DL 0.78）。minflt 補助系列では
  `download` が 16385 回/反復のページフォールト（64 MiB ÷ 4 KiB）を起こしている
- **N=256〜2048 の原因は未特定**: 診断の `resident_public` は fresh より 0〜6% 遅いだけで、bench の逆転幅（37〜57%）の 1/12〜1/15 しか写せず、ブリッジが fail。
  診断が写していない費用が bench の窓にある（候補は §9.3b・§9.6。いずれも未検証）。試して否定した仮説は §9.3b に記録
- G-CPU-INF は「自分たちの制約（REQ-1）」と分類した（条件つき。§9.4）
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

### 9.3b 実測結果（GB10・2026-10-28。5 run 中央値。生値・run 別は `logs/cpu-gemm-resident-reuse-3646/` の各 `aggregate.md`・`bridge_judge.md`）

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
  分類の根拠は G-CUDA-INF（cuBLAS）の先例と `loss-attribution-matrix.md` §10.5 の定義に揃えた規則適用であり、D の縮小は NVPL 相当の自作カーネル改善（REQ-1 の範囲外の依存を使わない制約下）でしか得られない
- 0.12.0 の正式値・tally は変えない（#3274 の範囲）

### 9.5 修正方針の提案（要件 3。実装はしていない・所有者判断待ち）

実測で帰属が決まったのは N=4096（H1: `download`）だけ。提案はその範囲に限る。修正 issue の起票・実装は所有者の承認後で、本 PR では行わない。

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
どちらも `bit` 同一であることの確認（既存の parity テスト）を伴う。現時点で採否の結論は出していない。

### 9.6 限界

- 帰属を出せたのは N=4096 だけ。N=512〜2048 の診断 ratio は 1.04〜1.06 で、bench の 1.47〜1.64 を説明できず、N=256 は診断が非再現。ブリッジ v2 は fail
- 診断の `resident_*` 腕は内部 API（`MemoryOps::download` 等）を使い、bench の公開 API の窓と同一ではない。ブリッジ v2 で一致を確認できたのは N=4096 のみ。
  N≤2048 の不一致の原因は未特定（§9.5 の候補は未検証）
- 実機に GPU 常駐プロセス 3 件があり専有ではない（RULE の専有確認は満たせず、`utilization.gpu` 0 %・%CPU 3.9 以下・新規プロセスなしを記録）。判定ゲートは load1 < 1.0（全 run 通過）
- 補助系列 tunables の N=2048 は fresh が大きく振れており参考値。minflt は 1 run
- G-CPU-INF は fandhe の層別区間が別プロセスの値・PyTorch が mm のみ・PyTorch 窓の run 間幅が大きい、という限界つきの規則適用（§9.4）。perf は使えなかった
- 旧腕（Tape 経路 reuse＋reset）は本診断に含めていない
- RULE.v2.txt によるブリッジ節の置換は実測前に固定したが、判定式（`G_bench`／`G_diag` の比 0.5〜2.0）は bench が出す値に合わせて新設したもので、RULE.txt の「matmul 区間差」式を実測していない
