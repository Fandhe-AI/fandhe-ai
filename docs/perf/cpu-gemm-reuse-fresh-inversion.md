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
| GB10・Mac mini の 5 round A/B（RULE の ADOPT／REJECT 判定） | **未実施**（A/B スクリプト・Mac mini 計測は本ブランチに含まれない。判定は未確定で、既定は OFF のまま） |
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
