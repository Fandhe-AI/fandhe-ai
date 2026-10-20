# GB10 CPU GEMM の reuse>fresh 逆転の帰属（診断基盤。実機実測は未実施）

イシュー #3187。対応表の GEMM（GB10 CPU）で N=256／512 が負けている（reuse の対最速比が
0.74／0.70）。0.11.0 の GB10 実測では reuse が fresh より遅い（N=256〜2048 の全点で
およそ 1.3〜1.9 倍）。reuse を fresh 並みにできれば判定行は算術上「負け」から「勝ちまたは
同等」へ変わる見込みである。本書はその原因を区間別に帰属するための診断基盤と、
`docs/perf/cpu-gemm-gb10-affinity-ab.md`（#2117）の A/B を同一セッションで実行する手順をまとめる。

## 1. 状態

| 項目 | 状態 |
|---|---|
| フェーズ分解テスト（`crates/facade/tests/cpu_gemm_reuse_fresh_diag.rs`） | 実装済み（CI 実行の bit 一致テスト 1 本 + 実機用 `#[ignore]` 1 本） |
| 事前登録の判定規則（`docs/perf/logs/cpu-gemm-reuse-fresh-3187/RULE.txt`） | 固定済み（実測前） |
| 計測・集計スクリプト（`orchestrate.sh`／`aggregate.py`） | 実装済み（`aggregate.py --self-test` あり。M4 Max 手元機で JSONL 形式の疎通のみ確認し、数値は記録に使わない） |
| GB10 での 5 run 実測と帰属判定（AC1） | **未実施（実機セッションへ申し送り）** |
| affinity A/B（#2117 の RULE.txt どおり 14 セル 5 round。AC2） | **未実施（verdict は undetermined のまま）** |
| `.rev-stamp` による計測ツリーの同定（AC3） | 実測時に `orchestrate.sh` が `env_info.txt` へ記録する（`.rev-stamp` 無しは開始前に停止） |
| `GB10_AFFINITY_ENABLED`（AC4） | false のまま（`crates/backend-cpu/src/gb10_affinity.rs` 無変更） |

本書に実測値は載せていない。数値は §6 を実機セッションで RULE.txt の規則に沿って埋める。

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
`scripts/bench/framework-compare/run_ab_gb10_affinity_cpu.sh`。手順は同ディレクトリ README.md。
結果は `cpu-gemm-gb10-affinity-ab.md` に記入する。ADOPT でも本イシューでは
`GB10_AFFINITY_ENABLED` を変えない（既定反転は #3227）。

## 5. 申し送り

| 項目 | 担当 |
|---|---|
| 逆転の是正（glibc の mmap 閾値・出力プール・Tape 結果の解放・prefault 等）。本書は列挙のみで実装しない | #3226 |
| affinity の既定反転（A/B が ADOPT の場合） | #3227 |
| 0.11.0 境界の同一セッション照合（framework-compare の `bench-fandhe` を fresh／reuse で各 5 run。record-only） | 実機セッション（本 PR では `orchestrate.sh` に `boundary` を実装していない） |

## 6. 結果（未実測）

| N | ratio (reuse/fresh) | G | 帰属 | rho_FH | rho_B | arena A/B |
|---|---|---|---|---|---|---|
| 256 | 未実測 | — | — | — | — | — |
| 512 | 未実測 | — | — | — | — | — |
| 1024 | 未実測 | — | — | — | — | — |
| 2048 | 未実測 | — | — | — | — | — |

総合: **未確定（実測未実施）**

## 7. 限界

- 5 run は同一ホスト・同一セッション。GB10 の専有ゲート（load1<1.0）を通らない run がある場合は参考扱い
- arena 統計は warmup を含むブロック全体の累計
- N=4096 の同型の帰属、M4 Max の正式系列は対象外（Issue は N=256〜2048、GB10 限定）
