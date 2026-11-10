# #4151 M-CPU-G256（<metal-node>・Mac mini・CPU gemm N=256）の固定費の診断

状態: **判定は保留（測り直し待ち）**。2026-11-10 の系列（下の「結果」）は全 5 ラウンドが他ジョブ（hw-lock を取らない wgpu の GPU GEMM・CPU 100%。所有者の別セッションのジョブで 03:37:56Z に起動。03:38:00〜03:41:02Z は N=2048、03:41:02〜04:00:14Z は N=4096 の GEMM を実行していた）と重なったため**参考**に下げた
（所有者判断（#4151・#4153・#4154 の 2026-11-10 のコメント「他ジョブと重なった系列は測り直す」））。部品は PR-1（#4190）、RULE の固定は PR-2（#4214。雛形の修正は #4210）、旧系列の実測は PR-3（#4217）。
旧系列の数値は消さずに残す。測り直しは他ジョブのゲートを足した RULE.v2 で、旧系列と別のラベルの系列で行う。
測り直しの規則は `RULE.v2.txt`（項 10。他ジョブの前提ゲート〈`docs/perf/logs/macmini-foreign-job-gate/`・PR #4227〉で各ラウンドを前後から挟み、系列 `series-remeasure-1` を記録先 `series-remeasure-1/` に置く）。固定前の試走 1 で、作業ディレクトリへのファイル書き込みが Spotlight の埋め込み生成（`mediaanalysisd`・%CPU 約 120）を呼び、全ラウンドが終了後のゲートで INVALID になったため、展開先は名前が `.noindex` で終わるディレクトリに作り直した（事実確認と試走 2 の記録は `RULE.v2.txt` 項 10 (e)）。

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
| `RULE.v2.txt` | 他ジョブの前提ゲートを足して測り直すための改訂版（項 1〜9 と PARAMS は fixed-at-utc を除き `RULE.txt` の逐語コピー。項 10 に部品の sha256・`.noindex` の展開先・系列の回し方・試走と事実確認の記録）。新系列 `series-remeasure-1/` の集計は `--rule RULE.v2.txt` |

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
   変換は同梱の `to_values_4151.py`（PR-3 で追加。RULE 固定後に書いたもので、RULE の sha256 固定の対象外）で行う:
   `python3 -I to_values_4151.py <記録先>/round1 … round5`。W_off／Wf_off は `raw/rust.jsonl`、W_on／Wf_on は `raw/rust-on.jsonl` の
   mode=reuse／fresh の `median_s`×1000、O は `raw/py.jsonl` の fresh、K_opp は `raw/measure_gemm256.jsonl` の item=mm_pure。
   `c_sme_kc0_ns` は probe の SME kc=0 の値、`c_sme_intercept_ns` は SME kc≥1（1〜256）の値を kc に対して**無重みの最小二乗**で
   当てた切片（#3175 の `aggregate.py` の `lsq_line` と同じ方法。この選び方は PR-3 の判断）。`c_sme_*` は「SME ON の場合」の
   上限欄にだけ使い、分類を動かさない。欠損・重複・非有限・0／負・parity fail は例外で止まり、既存の values.tsv は上書きしない。
5. `python3 -I aggregate_4151.py --dir <記録先>` で `aggregate.md`／`aggregate.tsv` を作る。

## 結果

> **他ジョブと重なった系列・参考（判定は保留・測り直し待ち）。** 全 5 ラウンド（round 1 03:50:42〜03:50:52Z、round 2 〜03:51:56Z、
> round 3 〜03:52:31Z、round 4 〜03:52:35Z、round 5 〜03:52:39Z）が、他ジョブ（hw-lock を取らない wgpu の GPU GEMM・CPU 100%。所有者の別セッションのジョブで 03:37:56Z に起動。03:38:00〜03:41:02Z は N=2048、03:41:02〜04:00:14Z は N=4096 の GEMM を実行していた）の N=4096 の区間と重なった。load1 のゲート（3.0 未満）は
> 通ったが、このゲートでは重なりを検出できない。下の判定・確度・申し送りの文案は旧系列のもので、測り直すまで使わない（所有者判断（#4151・#4153・#4154 の 2026-11-10 のコメント「他ジョブと重なった系列は測り直す」））。

計測: 2026-11-10T03:50:46Z〜03:52:39Z（<metal-node>・Apple M4・RAYON_NUM_THREADS 未設定）。5 ラウンドとも PASS（load1 2.36〜2.94。
直前に同じ機体で #4153 の計測が走っていたため、ゲートの閾値 3.0 に近い。後に、同じ時間帯に他ジョブが動いていたことが分かった。上の注記）。集計は `aggregate.md`／`aggregate.tsv`、
ラウンド別の窓の値は `round{i}/values.tsv`、計測環境（ビルドログ・`cargo tree` で path patch を確かめた記録・4 腕の指紋）は `env/`。

**旧系列の判定（参考・保留）: 自分たちの制約（確度低）**（前分類「ハードの上限」は測り直すまで据え置く）

| 式（OFF 腕＝0.14.0 の本番構成） | 中央値 | ラウンド別 | 閾値 1.0 に対して |
|---|---|---|---|
| F2 | 0.6338 | 0.6338・0.6565・0.6224・0.6455・0.6283 | 下 |
| F1 | 2.1531 | 2.0360・2.1531・1.7651・2.1573・2.4679 | 上 |
| F1_drv | 1.3671 | 1.3223・1.4251・1.1433・1.3671・1.4769 | 上 |

- 観測比 O÷W_off の中央値は 0.6822（スコアボード 0.69 から 1.1%）。
- 理由（`aggregate_4151.py` の出力）: GEMM 駆動部の固定費。「ハードの上限」は成り立たない（F1・F1_drv が全ラウンドで 1.0 を超える）。
- 内訳（timed 腕・hot 系列・ラウンド中央値・ms）: OFF は G 0.1944 のうち kernel_ideal（Σkernel÷P_used）0.0704、
  F_drv（G − kernel_ideal）0.1240。ON は G 0.1299・kernel_ideal 0.0306・F_drv 0.0972。区間別の ns は `aggregate.md` の表。
  OFF の job 合計で大きいのは kernel 573 µs・pack_b 116 µs・alloc 101 µs・pack_a 81 µs（P_used 8・jobs 20・呼び出し 704 回）。
- **確度低の理由**: on 腕の計時の摂動（timed÷base−1）が上限 0.10 を超えたラウンドがある（round 1 は +0.213、round 2 は −0.190。
  on-base の hot 中央値が 105.8〜131.0 µs の間で揺れた）。OFF 腕の摂動は全ラウンドで 0.10 以内（中央値 −0.021）。
- SME ON の場合（記録のみ。分類は動かさない）: F2 0.8392・F1 1.9870・F1_drv 1.8816。`smstart`／`smstop` を含む 1 呼び出し固定費の
  上限 × 呼び出し回数 ÷ P_used は kc=0 直接計時で 0.0013 ms、切片で 0.0026 ms（kernel の内側。F_drv には足さない）。
- probe の SME kc=0 は round 1 だけ 142.5 ns（ほかは 42.6〜45.4 ns）。差し替えずにそのまま記録した。切片は 88.2〜92.0 ns。
- 計測後の off-base ツリーには `scripts/bench/framework-compare/__pycache__/checksum_contract.cpython-312.pyc` が 1 件増えた
  （round 1 の `torch_cpu_mlp_diag.py` 実行で生成）。これを除いた指紋は RULE の `tree_fingerprint` と一致する（`env/trees.txt`）。
  計測前の照合は 4 腕とも一致していた。

## Phase 6 への申し送り

本 issue はスコアボード・`unreached_0140.tsv`・`loss-attribution-matrix.md` を書き換えない。Phase 6（#4173 配下）の
スコアボード issue が行う書き換えのために、次の文案を残す（採否は Phase 6 が決める）。**下の文案は他ジョブと重なった旧系列に基づくため保留で、
測り直しの結果で書き直す。**

- 「SME は対象外で未測定」の置き換え文案: 「SME ON は #3232 で N=256 fresh 0.888 倍（fandhe ON 0.184 ms）。#4151 の診断（2026-11-10・
  Mac mini・5 ラウンド）では ON の F2 0.84・F1 1.99（記録のみ）」。
- 分類の文案: **自分たちの制約（確度低）**。理由欄「GEMM 駆動部の固定費（#4151: OFF の F1 2.15・F1_drv 1.37 が全ラウンドで 1.0 超、
  F_drv 0.124 ms／G 0.194 ms）。確度低は on 腕の計時摂動が上限超過のため」。

## 限界

- plain GEMM の入口で、`bias_act` 融合経路ではない。`kernel` は端タイルのコピーを含む。
- E コアと P コアは分けられない（macOS に正式な pinning の手段がない。#3175 §5）。
- `gap_us`（既定 50 µs のスピン待ち）が実ベンチ窓の直列区間の起床コストを再現できているかは未検証。
- PR-1 時点の確認は Linux でのビルド・単体テスト・`--self-test`・`--dry-run` まで。Mac mini での実行・SME 側の `n_calls`・`c_sme`・
  ノード側の `tree_fingerprint` 一致・torch 経路は PR-3 の実測で通った。
- ノイズ帯 NB-RANGE5 は #4142 で定義を変えずに承認済み（#4209）。
