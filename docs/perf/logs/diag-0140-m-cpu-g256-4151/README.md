# #4151 M-CPU-G256（<metal-node>・Mac mini・CPU gemm N=256）の固定費の診断

状態: **PR-1 時点は実測なし**。本 PR は計測の部品（example・patch 2 本・起動キット・集計スクリプト・RULE 雛形）だけを置く。
RULE の固定（PR-2）・5 ラウンドの実測と判定（PR-3）は Mac mini のセッションが行い、PR-3 が #4151 を閉じる。
結果欄は実測後に追記する。

## 目的

スコアボード 0.14.0 の M-CPU-G256 は 0.69×（fandhe-ai reuse 0.238 ms・PyTorch fresh 0.164 ms）で、
`unreached_0140.tsv` では「ハードの上限」、理由欄は「SME は対象外で未測定」。この記載は古い:
#3232 の A/B（`docs/perf/logs/sme-default-on-ab-3232/aggregate-macmini.md:26`）で N=256 fresh は SME ON で
0.888 倍（fandhe ON 0.184 ms）まで測られている。それでも約 12% 足りない。相手の純粋な `mm` は 0.0240 ms
（`MM-179`）で、fandhe-ai の 0.2 ms 前後は 33.5 MFLOP に対し約 170 GFLOPS にしかならない。
閉源の Accelerate やハードの上限ではなく、自分たちの GEMM 駆動部の固定費（worker の起床・fork-join・pack・
C の staging・確保・SME の `smstart`／`smstop`）が主因である可能性があり、`loss-attribution-matrix.md:120` の
`THREAD`・`FIX` は推定または未確定のまま残っている。

SME OFF（0.14.0 の本番構成）と ON の両方で、N=256 の 1 回の呼び出しを区間に分けて測り、事前登録した数値規則で
「ハードの上限」が成り立つかを判定する。

## 中身

| ファイル | 内容 |
|---|---|
| `crates/backend-cpu/examples/cpu_gemm_n256_fixed_cost_4151.rs` | 計測バイナリ（base 腕）。系列 hot・gap・serial_t1・forkjoin_floor_hot／gap・cold_first_call。出力は `ROW4151`／`ENV4151`／`END4151` |
| `on-arm.patch` | `SME_PRODUCTION_ENABLED` を true にする 1 行（#3232 の OFF patch の逆向き。ON は計測専用ツリーでしか作れない） |
| `diag_timing.patch` | 区間計時（`gemm_blis/diag_fixed_cost_timer.rs` の追加と数か所のフック）。**main には当てない**（`determinism_inventory` が通らない） |
| `patch_sha256.txt` | 2 本の patch の sha256 |
| `run_round.sh` | 1 ラウンドの起動キット（bash 3.2 互換・`--dry-run` あり） |
| `aggregate_4151.py` | 厳格パースの集計（標準ライブラリのみ・`python3 -I`・`--self-test`・`--check-template`・`--fingerprint`） |
| `RULE.template.txt` | 事前登録規則の雛形（`## PARAMS` はプレースホルダ） |

## 4 腕

| 腕 | 作り方 |
|---|---|
| off-base | clean archive |
| on-base | off-base + `on-arm.patch` |
| off-timed | off-base + `diag_timing.patch` |
| on-timed | off-base + `on-arm.patch` + `diag_timing.patch` |

`bits_hash`（GEMM 系列は、計時フック有効のまま走る計測ループ最後の 1 回〈直前に C をゼロへ戻す〉の出力の `to_bits()` の FNV-1a 64。GEMM 出力のない系列は起動直後の 1 回の値）が 4 腕・全系列で一致することを確かめる。
計時フックと SME の有無が数値を変えないことの検査で、不一致は FAIL。ON ツリーでは `cargo test --lib` を走らせない
（ドリフト検出テスト `sme_production_disabled_after_gb10_reject` が作りの上で落ちる）。

## 区間の定義

時計は `Instant`（macOS の分解能は約 41.7 ns なので、マイクロカーネル 1 回ごとには計らずループ単位で計る）。

| 区間 | 取り方 |
|---|---|
| pre（grid・split を内訳に持つ） | 入口から `par_iter_mut` の直前まで。`grid` は入口から `job_grid` 直後まで、`split` はそこから `split_c_into_jobs` 直後まで |
| fork_wake（min・max） | `par_iter_mut` の呼び出しから各 worker の最初の job が始まるまで |
| stage_in／stage_out | `c_local` の確保と copy-in／copy-out（job 合計） |
| alloc | `PanelBuffers::new`（job 合計） |
| pack_b／pack_a | B pack／A pack ループ（job 合計） |
| kernel | jr／ir ループ全体（端タイルのコピーを含む。job 合計） |
| join | 最後の job の終了から `par_iter_mut` が戻るまで |
| region_wall | `par_iter_mut` の呼び出しから戻りまで |
| n_calls | マイクロカーネルの呼び出し回数（ir×jr のタイル数）。集計が手計算の下限と突き合わせる |

導出量（`aggregate_4151.py` が計算）: `kernel_ideal = Σkernel ÷ P_used`・`kernel_crit = max_worker Σkernel`・
`imbalance`・`F_drv = G − kernel_ideal`・`perturbation`。`kernel_ideal`・`imbalance`・区間和の整合性検査用の値は、P_used が呼び出しごとに変わりうるため診断パッチが region ごとに計算してから中央値に集約する（中央値同士の比は使わない）。診断式 F1・F2・F1_drv と判定規則は `RULE.template.txt` 項 6。

SME の `smstart`／`smstop` は、マイクロカーネルが 1 つの `asm!` の中で対にしているため、新しい `asm!`
を足さずに単独では計れない（`docs/perf/cpu-sme-thread-scaling-macmini.md` §7）。既存の `#[ignore]` 診断
`sme_microcall_fixed_cost_probe`（`kc=0` の直接計時と `kc≥1` の切片。前回 44.5〜89.8 ns）を再利用し、
`n_calls × c_sme ÷ P_used` を**上限**として別欄に出す（kernel の内側。`F_drv` には足さない）。

## 手順（Mac mini のセッション）

1. PR-1（#4192）のマージコミットの SHA で `RULE.template.txt` を埋める。`tree_sha`・`tree_fingerprint`
   （`python3 -I aggregate_4151.py --fingerprint <tree-dir>`）・`aggregate_script_sha256`・patch の sha256 を入れ、
   `git diff --name-only v0.14.0 <tree_sha> -- crates/backend-cpu/src` の一覧を記録して、差が
   `crates/backend-cpu/src/gemm_blis/mod.rs`・`crates/backend-cpu/src/lib.rs` の 2 ファイル（数値・実行経路を変えない診断 API
   `sme_production_enabled()` の追加と再エクスポート）だけであることを確かめる。backend-cpu 以外の `crates/*/src/*` の差は
   一覧を記録するだけで止めない（RULE 項 2）。そのうえで**測る前に単独コミット**（PR-2）。
2. ビルド（ゲートの前に済ませる。計測中に並走させない）: 4 腕のツリーを作り、ツリーごとに別の target dir で
   `env -u CARGO_TARGET_DIR -u RAYON_NUM_THREADS cargo build --release -p fandhe-ai-backend-cpu --example cpu_gemm_n256_fixed_cost_4151`。
   off-base ツリーでは `cargo test --release -p fandhe-ai-backend-cpu --lib --no-run` も済ませる。
   bench-fandhe は `scripts/bench/framework-compare/run_ab_sme_default_on_cpu.sh` と同じ要領で、
   `cargo build --release -p bench-fandhe --config 'patch.crates-io.fandhe-ai.path="<tree>/crates/facade"'` を off-base と on-base に向けて腕ごとに別の `--target-dir` で作り（生成物は `<target-dir>/release/bench-fandhe`。dev プロファイルのバイナリは未最適化で、release の example の G と混ぜると診断式を誤るため `--release` 必須）、
   `cargo tree` で path patch が効いたことを確かめる。
3. 5 ラウンド: `run_round.sh <i> <round-dir>`（環境変数はファイル冒頭）。ホスト名・ホームの絶対パスは raw をマスクする。
4. `values.tsv`（quantity／value の 2 列）を raw から作り、変換手順をここに残す。W・Wf は bench-fandhe の reuse／fresh の
   median（ms。off／on）、O は bench_py.py の fresh、K_opp は `torch_cpu_mlp_diag.py` の `mm_pure`、`c_sme_*` は probe の値（ns）。
   gemm 256 の bench-fandhe に `--phases` は無いため、窓の内訳の record_only は取れない（unavailable）。
5. `python3 -I aggregate_4151.py --dir <記録先>` で `aggregate.md`／`aggregate.tsv` を作る。

## 結果

（PR-3 で追記）

## Phase 6 への申し送り（PR-3 で埋める）

本 issue はスコアボード・`unreached_0140.tsv`・`loss-attribution-matrix.md` を書き換えない。Phase 6（#4173 配下）の
スコアボード issue が行う書き換えのために、PR-3 で次を残す。

- `unreached_0140.tsv` の M-CPU-G256 行と `loss-attribution-matrix.md` §10.3.2 にある「SME は対象外で未測定」を、
  #3232 の実測（ON 0.888・0.184 ms）と本診断の結果に置き換える文案。
- 分類が変わる場合の新しい分類と理由欄の文案。

## 限界

- plain GEMM の入口で、`bias_act` 融合経路ではない。`kernel` は端タイルのコピーを含む。
- E コアと P コアは分けられない（macOS に正式な pinning の手段がない。#3175 §5）。
- `gap_us`（既定 50 µs のスピン待ち）が実ベンチ窓の直列区間の起床コストを再現できているかは未検証。
- Linux での確認はビルド・単体テスト・`--self-test`・`--dry-run` まで。Mac mini での実行、SME 側の `n_calls`・`c_sme`、
  ノード側の `tree_fingerprint` 一致、torch 経路は未検証。
- ノイズ帯 NB-RANGE5 は #4142 で所有者の承認待ち。定義が変われば `RULE.v2.txt` で改訂し、旧版も残す。
