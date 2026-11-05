# 0.14.0 の 6 フレームワーク同席計測（GB10）

**状態: 実測済み（2026-11-05 UTC。GB10 5 ラウンド・5/5 完走）。** crates.io `fandhe-ai =0.14.0`（2026-11-05 公開）で framework-compare の GB10 セルを取り直した（#3875）。事前登録規則 `RULE.txt` は計測前の 2026-11-05T16:00:58Z に固定し、以後編集していない（RULE.v2.txt なし）。計測腕（ARM）は main の #4063 マージ `18a7f4b87fe61029a179cb1922cc82928440f2c9`（単腕）。Mac mini 側（#3874）は別セッションの別ディレクトリ・別 PR で、本ディレクトリには Mac mini の値を混ぜていない。スコアボード HTML の生成（#3876）・Artifact の公開・対応表の更新は範囲外。

時系列（RULE の固定が計測より前であることの根拠。時刻はノードと送信側の `date -u`）:

| 時刻（UTC） | できごと | 根拠 |
|---|---|---|
| 15:54:09 | 実機ロック取得（owner `#3875 rel-0.14.0 2026-11-05T15:54:09Z`）・`git archive` で計測ツリーを展開 | 全 run の `env_info.txt` |
| 15:56:14〜16:00:22 | 行数スモーク（ビルド込み 1 round・値は判定に使わない） | `smoke/smoke-gb10.out` |
| 16:00:58 | RULE 固定・最初のコミット `e533572a7` | `RULE.txt`・コミット時刻 |
| 16:02:55 | push の到達を確認 | リモートの HEAD |
| 16:03:04 | 系列の前の専有確認（compute プロセスなし） | `gb10/compute_apps_before_series.txt` |
| 16:03:05 | 最初のゲートのサンプル | `gb10/gate.log` |
| 16:04:05〜16:39:35 | run1 開始〜run5 終了 | `gb10/run{1..5}.log`・`gb10/loop.log` |
| 16:39:35 | 実機ロック解除（`lock-released`） | `gb10/loop.log` |

## 目的

0.13.0 の計測（`../framework-compare-0.13.0-remeasure/`）を、registry ピンだけ `=0.14.0` に上げて取り直す。旧条件腕は置かない単腕構成で、判定規則は 0.13.0 から変えない（比 = 同一 run 内の最速の有効な相手 / fandhe-ai reuse の 5 run 中央値。1 超で 1 位・0.90 以上で僅差・相手側の parity fail は判定不能。判定側 fandhe-ai の要素検証不合格は生成停止）。「0.13.0 比」は同一セル・同一モードの fandhe-ai 中央値同士の比（日またぎで負荷の揺れを含む。表示のみで勝敗には使わない）。

## 0.13.0 版からの差分

| 項目 | 0.13.0 | 0.14.0 |
|---|---|---|
| 腕 | `5d05a5023`（#3709） | `18a7f4b87`（#4063） |
| 版 | `fandhe-ai v0.13.0` の検査・出力名 | `fandhe-ai v0.14.0` の検査・出力名 |
| 前版比の参照先 | 0.12.0 の GB10 中央値 3 ファイル | 0.13.0 の GB10 中央値 3 ファイル（sha256 を RULE に固定。`aggregate/judge_gb10_0140.py` が照合） |
| 前版の勝敗 tally | `framework-compare-0.12.0-remeasure/scoreboard/gen_0120.out` | `framework-compare-0.13.0-remeasure/scoreboard/gen_0130.out`（`tally 13 3 11 invalid 1 noise na total 27`） |
| 専有の確認（#3875 の指示） | compute プロセスを `env_info.txt` に記録するだけ | 記録に加え、系列の前と各 round の前に 1 件でも出たら「専有未達」で打ち切る。vLLM・ComfyUI・Kokoro の常駐件数とロック owner も `env_info.txt` に記録する |
| 実機ロック | 手で取り外し | 各 round でロック owner を検査し、系列の終了時に自動で外す（失敗時も） |
| 判定一覧の出力元 | `scoreboard/gen_0130.py`（Mac mini 必須・HTML も生成） | `aggregate/judge_gb10_0140.py`（判定関数を字句そのままで GB10 の 14 セルに限定。下の「判定器の検証」） |
| RULE の書き方 | 判定・前版比の定義を Mac mini 側の RULE へ委ねる | 同じ文を本 RULE へ写して自己完結にした（中身は同じ） |

判定規則・N=5・ゲート・期待行数・既定設定（SME OFF・Graph capture OFF）・tolerance・baseline は 0.13.0 と同一。

## 受け入れ条件

| # | 要件 | 状態 | 根拠 |
|---|---|---|---|
| A1 | 全 run の tree が v0.14.0・rev-stamp が同一 | **達成** | 5 run とも `tree.txt` が `└── fandhe-ai v0.14.0`、`rev-stamp.txt` が `18a7f4b87fe61029a179cb1922cc82928440f2c9`。`aggregate/check_identity.py` が rc=0（行数 112／8／28 を含む） |
| A2 | 5 run 中央値と 0.13.0 比 | **達成** | `gb10/*-median5.jsonl`・`gb10/judge_0140.out`。median5 は run1〜run5 から再生成して byte 一致を確認した |
| A3 | 全セルの比 | **達成** | GB10 14 セルの比は `gb10/judge_0140.out`。1.0 以下の 4 セルは数字のみ載せた（3 分類の割り当ては 0.13.0 では `scoreboard/unreached_0130.tsv` にあり、0.14.0 は #3876 の範囲） |
| A4 | 事前登録規則を計測前に固定 | **達成** | `RULE.txt`（固定 2026-11-05T16:00:58Z）。固定後に規則・スクリプト・期待行数・tolerance は変えていない |
| A5 | 機体を混ぜない・専有 | **達成** | 5 run の `env_info.txt` 先頭行が `NVIDIA GB10, 580.173.02` で一致。compute プロセスは系列の前・全 5 run の前・系列の後とも 0 件（`gb10/compute_apps_*.txt`・`env_info.txt`）。常駐は vllm・comfyui・kokoro とも 0。ノードは 1 台で途中で変えていない |
| A6 | 判定側の要素検証 | **達成（gemm のみ）** | fandhe-ai の gemm 行（reuse・fresh の 20 行 × 5 run と中央値）の `parity_fail_count` 合計 0（`gb10/verify_0140.txt`）。train／infer の行には要素検証の欄がない（0.13.0 と同じ出力仕様） |
| A7 | ホスト名を含まない | **達成** | 公開テキストはノードを `<cuda-node>`、ホームを `<home>` と書く。回収後に PR 差分全体をユーザー名・`/home/`・`/Users/`・内部ドメイン・SSH 別名で grep して、規則の文のほかに 0 件 |

## 事前確認（計測前の確認項目と結果）

- [x] `scripts/bench/framework-compare` の `5d05a5023` からの差分が指定の範囲に収まる（`git diff 5d05a5023 18a7f4b87`）:
  - `*.rs` は `bench-fandhe/src/main.rs` の 1 ファイルだけで、`VERSION` 定数（`"0.13.0"`→`"0.14.0"`）とピンの版を述べた文字列・コメントだけ。`0.13.0` を `0.14.0` に置換した旧版と byte 一致
  - `Cargo.toml` は `bench-fandhe/Cargo.toml` の 1 ファイルだけで、`fandhe-ai` の版の行とコメントだけ（同じ置換で byte 一致）
  - `Cargo.lock` は fandhe-ai 系 7 クレート（fandhe-ai・autodiff・backend-cpu・backend-cuda・backend-metal・onnx-interop・tensor-core）の version／checksum、`fandhe-ai` の dependencies への `fandhe-ai-backend-rocm` の追加、新規エントリ `fandhe-ai-backend-rocm 0.14.0`（依存は既存の `libloading 0.9.0` と自社クレート）だけ。外部 crate の増減・版変更は 0
  - `deny.toml` は #3962 の RUSTSEC-2026-0429 の ignore 追加とコメントだけ（計測には影響しない）
  - ほかの差（`run_all_cuda.sh`・`run_all.sh`・`run_ab_*.sh`・`compare_gemm_ab.py`・`README.md`）はコメント・docstring・文書の版表記だけ。stage1 の本体である `run_all_cuda.sh` の差はコメント 1 行
- [x] 常駐入力の判別式 `^[[:space:]]+let c0 = tape\.matmul_resident\(` が計測ツリーの `bench-fandhe/src/main.rs` で 1 件（固定前に `grep -Ec` で確認。`dgx-run-0140.sh` が run ごとに検査し、全 run を通過）
- [x] `bench_py.py` の sha256 が `9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b`（0.13.0 と同値。run ごとに照合し全 run を通過）
- [x] Python 3 FW の版が 0.13.0 と同じ: `env/pip-freeze-venv-bench.txt` が 0.13.0 の同名ファイルと byte 一致（`pip freeze` ＋ `python --version`。固定前に取得）。JSONL の `version` 欄は `pytorch 2.14.0+cu130`・`tensorflow 2.21.0`・`scipy 1.18.1`
- [x] 行数スモーク: stage1 112・skipped 0・stage2 8・stage3 28・rc 0・`fandhe-ai v0.14.0`（`smoke/`。値は判定に使わない）。`extra.err`・`py.err` は 0 バイトで、`backend-rocm` が通常依存になったことによる新しい stderr 出力はなし
- [x] 計測ツリーの同一性: ノードの `crates`・`scripts/bench/framework-compare` の全ファイルの sha256 一覧のハッシュ（`target/` と実行で生じる `results-cuda.jsonl`・`skipped-cuda.log` を除く）が `git archive 18a7f4b87` の展開と一致（`b0ca0cb0…`）。`.rev-stamp` は `18a7f4b87…`。展開先は新規のディレクトリで、`target/` もツリー内に新規（共有 target を使っていない）

### 判定器の検証

`aggregate/judge_gb10_0140.py` の `paired_ratio`・`judge` は `gen_0130.py` と docstring の 1 文（他の呼び出し元に触れる部分）を除いて字句で同じ。固定前に次を確認した:

- `--regress-0130`: 0.13.0 の GB10 データ（前版 0.12.0）へ当てて `gen_0130.out` の GB10 14 行と byte 一致し、tally が `tally 10 0 4 invalid 1 noise na total 14`
- `--self-test`: 判定側の要素検証不合格（中央値・run）・欠損 run・有効な相手の不在・版の不一致・`parity_fail_count` の欠損や不正値・前版 tally や sha256 の不一致で止まる

gen_0130.py に無い入口検査（gemm の `parity_fail_count` の非負整数検査・各 run の compute プロセス 0 件）は fail-closed にする方向だけで、区分は変えない。

## 計測条件

| 項目 | GB10（`<cuda-node>`） |
|---|---|
| 機体 | NVIDIA GB10・driver 580.173.02（`env_info.txt`。5 run 一致。0.13.0 と同じ）。専有のノードは #3875 で指定 |
| 腕 | ARM `18a7f4b87`（単腕）。`git archive` で展開し `.rev-stamp`（40 桁）を置いた |
| registry ピン | `fandhe-ai =0.14.0`・`candle-core =0.11.0`・`burn =0.21.0`（`tree.txt` で run ごとに `fandhe-ai` を検査） |
| Python 3 FW | pytorch 2.14.0+cu130・tensorflow 2.21.0（CPU のみ）・scipy 1.18.1（JSONL の `version` 欄）。全パッケージが 0.13.0 と同じ（`env/`） |
| toolchain | rustc 1.97.1（ノード） |
| 既定設定 | SME OFF・Graph capture OFF（既定のまま） |
| 日時 | 2026-11-05 16:04:05Z〜16:39:35Z（run1 開始〜run5 終了） |
| ゲート | 各ラウンド前に load1 < 1.0 かつ GPU 0% の 3 連続（`gb10/gate.log` 37 行）。ラウンド開始時 load1 = 0.45／0.62／0.53／0.32／0.63・GPU 0%。run1 は待機なし（3 サンプル）、run2〜5 は 3〜5 分待機した（最初のサンプルは load1 5.91／5.22／6.49／5.30 で直前ラウンドの終了直後。run5 の前は途中で 1.02・1.09 に戻って数え直した）。全サンプルで GPU 0%。`GATE-INVALID`・`GATE-TIMEOUT` なし |
| 完走 | `gb10/loop.log` `ALL-DONE completed=5/5`。skipped 0 件（`skipped-dgx-0.14.0.log` 5 本とも 0 バイト） |
| 実機ロック | `~/work/.fandhe-hw-lock`（owner `#3875 rel-0.14.0 2026-11-05T15:54:09Z`）。16:39:35Z に解除 |
| 計測中の接続 | 完了待ちは系列開始の直後に張った SSH 1 本（ノード上で `sleep 60` のループ）。新しいログインを計測中に作らない。ビルドは系列前に 1 回だけ（`gb10/build.log` は 3 ターゲットとも no-op の `Finished`。実ビルドはスモークの前） |

- **判定不能 1 セル**: PyTorch CPU gemm N=4096 の 1 要素が複合判定を外れる（`parity_fail_count` 1／16777216。5 run とも。0.13.0 と同じ）。順位に含めない。fandhe-ai／candle の CUDA TF32 opt-in 行は f32 判定から外す（0.13.0 と同じ扱い）。
- **ノイズ帯**: 対照系列なし・未算出（判定一覧の tally は `noise na`。0 件ではない）。

## 結果（同一 run 内比の 5 run 中央値。判定は `aggregate/judge_gb10_0140.py`）

| 条件 | 1 位 | 僅差 | 負け | 判定不能 | ノイズ帯 | 計 |
|---|---|---|---|---|---|---|
| 0.14.0（GB10 14 セル） | **10** | 0 | 4 | 1 | 未算出（対照系列なし） | 14 |
| 0.13.0（GB10 14 セル。`gen_0130.out` の GB10 行から数えた値。`--regress-0130` の tally と同じ） | 10 | 0 | 4 | 1 | 未算出 | 14 |

`gb10/judge_0140.out` の tally 行: `tally 10 0 4 invalid 1 noise na total 14`。

前版 tally `tally 13 3 11 invalid 1 noise na total 27`（`../framework-compare-0.13.0-remeasure/scoreboard/gen_0130.out`）は Mac mini を含む 27 セルの値で、上の 14 セルと数を直接比べない。両機体を合わせた 0.14.0 の tally は #3876 が出す。

### 0.13.0 からの区分変化

**なし**（14 セルすべて区分が同じ）。比の変化が大きいのは gemm CPU N=512 の 1.14→1.00、N=2048 の 1.74→1.62、N=256 の 0.72→0.83（下の表）。

### セル表（`gb10/judge_0140.out`・`gb10/tables_0140.md`）

各列は「区分 比（vs 最速相手）」。比は同一 run 内比の 5 run 中央値。★ = 0.14.0 の比が 1.0 以下。「0.13.0 比」は fandhe-ai 中央値同士の時間比（推論はスループット比。<1 が高速化・推論は >1 が改善）。

| セル | 0.14.0 | 0.13.0 | 0.14.0 の run 別比 | 0.13.0 比（fandhe-ai 中央値同士） |
|---|---|---|---|---|
| gemm CUDA N=256 | 1 位 1.14×（vs candle） | 1 位 1.16×（vs candle） | 1.14 1.14 1.16 1.17 1.14 | 1.01× |
| gemm CUDA N=512 | 1 位 1.14×（vs candle） | 1 位 1.14×（vs candle） | 1.14 1.12 1.14 1.15 1.17 | 0.99× |
| gemm CUDA N=1024 | 1 位 1.18×（vs candle） | 1 位 1.15×（vs candle） | 1.13 1.20 1.18 1.15 1.19 | 1.00× |
| gemm CUDA N=2048 | 1 位 1.14×（vs burn） | 1 位 1.13×（vs burn） | 1.15 1.12 1.14 1.14 1.15 | 1.00× |
| gemm CUDA N=4096 | 1 位 1.99×（vs burn） | 1 位 1.97×（vs burn） | 1.99 2.00 1.98 2.04 1.99 | 1.01× |
| gemm CPU N=256 ★ | 3 位 0.83×（vs candle） | 3 位 0.72×（vs pytorch） | 0.85 0.87 0.62 0.69 0.83 | 1.02× |
| gemm CPU N=512 | 1 位 1.00×（vs candle） | 1 位 1.14×（vs candle） | 1.00 1.04 0.96 1.10 0.98 | 0.94× |
| gemm CPU N=1024 | 1 位 1.20×（vs candle） | 1 位 1.25×（vs candle） | 1.18 1.24 1.20 1.08 1.30 | 1.03× |
| gemm CPU N=2048 | 1 位 1.62×（vs tensorflow） | 1 位 1.74×（vs tensorflow） | 1.60 1.73 1.55 1.62 1.66 | 1.05× |
| gemm CPU N=4096 | 1 位 1.17×（vs tensorflow） | 1 位 1.20×（vs tensorflow） | 1.17 1.16 1.17 1.17 1.19 | 1.04× |
| train CUDA | 1 位 2.00×（vs candle） | 1 位 1.99×（vs candle） | 2.00 2.03 2.00 1.89 2.02 | 1.00× |
| infer CUDA ★ | 3 位 0.51×（vs pytorch） | 3 位 0.51×（vs pytorch） | 0.51 0.50 0.52 0.52 0.51 | 1.00× |
| train CPU ★ | 2 位 0.51×（vs pytorch） | 2 位 0.52×（vs pytorch） | 0.54 0.59 0.51 0.51 0.49 | 0.98× |
| infer CPU ★ | 2 位 0.75×（vs pytorch） | 2 位 0.76×（vs pytorch） | 0.78 0.83 0.75 0.64 0.61 | 0.95× |

### 目立つセル（数字のみ。原因の切り分けは本計測では行っていない）

時間は 5 run 中央値（ms）。0.13.0（2026-10-31）との比は日またぎの負荷・機体の状態の揺れを含み、この表からライブラリの変更の効果は分けられない。

| セル | fandhe-ai reuse 0.13.0→0.14.0 | 相手（同じ期間） | 読み方 |
|---|---|---|---|
| gemm CPU N=512 | 1.7451→1.6402 | candle 1.9814→1.6465 | 比は 1.14→**1.00**（厳密には 1.0038）。規則どおり 1 位だが、run 別は 1.00 1.04 0.96 1.10 0.98 で 2 run が 1 未満。fandhe-ai が速くなった以上に candle の窓が縮んだ。ノイズ帯は未算出で、この 1 位が揺れの中にあるかは示せない |
| gemm CPU N=2048 | 16.6114→17.5224（1.05 倍） | tensorflow 28.5688→28.2675・candle 33.5367→33.7146 | 1 位のまま 1.74→1.62。相手がほぼ同じなのに fandhe-ai だけが遅い側へ動いた |
| gemm CPU N=4096 | 121.9729→126.9719（1.04 倍） | tensorflow 147.0190→148.3057・candle 220.2540→218.6878 | 1 位のまま 1.20→1.17。同上 |
| gemm CPU N=1024 | 4.4427→4.5736（1.03 倍） | candle 5.4362→5.5461 | 1 位のまま 1.25→1.20。fresh は 3.895→4.512（1.16 倍） |
| gemm CPU N=256 | 0.3656→0.3742（1.02 倍） | PyTorch 0.2635→0.3459・candle 0.3416→0.3306 | 3 位のまま 0.72→0.83。最速の相手が PyTorch から candle に変わった（PyTorch の窓が 1.31 倍） |
| infer CPU | 0.1621→0.1713（スループット 0.95 倍） | PyTorch 0.1256→0.1333 | 2 位のまま 0.76→0.75。run 別は 0.78 0.83 0.75 0.64 0.61 で後半の run ほど低い |
| infer CUDA（参考） | 0.0785→0.0785 | burn 0.2008→0.3994（最速の相手ではない）・PyTorch 0.0402→0.0401 | 判定に影響なし。burn の窓だけが約 2 倍になった原因は未確認 |

### GB10 CPU gemm の reuse／fresh（実測）

fandhe-ai の 5 run 中央値（ms）。reuse が fresh より遅いのは N=256（1.87）で、ほかの 4 サイズは 0.92〜1.01。

| N | 0.14.0 fresh | 0.14.0 reuse | 0.14.0 reuse/fresh | 0.13.0 fresh | 0.13.0 reuse | 0.13.0 reuse/fresh |
|---|---|---|---|---|---|---|
| 256 | 0.200 | 0.374 | 1.87 | 0.191 | 0.366 | 1.91 |
| 512 | 1.788 | 1.640 | 0.92 | 1.750 | 1.745 | 1.00 |
| 1024 | 4.512 | 4.574 | 1.01 | 3.895 | 4.443 | 1.14 |
| 2048 | 17.510 | 17.522 | 1.00 | 16.720 | 16.611 | 0.99 |
| 4096 | 126.061 | 126.972 | 1.01 | 121.661 | 121.973 | 1.00 |

- N=256 の reuse が fresh より遅い後退は 0.14.0 でも残る。区間の切り分けは本計測では行っていない。

### 1.0 以下のセル（0.14.0・4 セル）

gemm CPU N=256（3 位 0.83）・infer CUDA（3 位 0.51）・train CPU（2 位 0.51）・infer CPU（2 位 0.75）。0.13.0 で 1.0 以下だった GB10 の 4 セルと同じ。本計測は原因の切り分けを行っていないので分類は書かない（0.13.0 の分類は `../framework-compare-0.13.0-remeasure/README.md`「1.0 以下のセルの理由」。0.14.0 の分類は #3876 の範囲）。

## 既知の訂正

なし。学習の最終 checksum は fandhe-ai 0.080541（CPU・CUDA）、他 5 フレームワークは 0.117549〜0.117643 で、0.13.0 の README の記述と同じ（burn の GB10 CUDA が 0.117643）。

## 実機テスト（#3256 の C1〜C9、#3272 の X1）

未実施（範囲外）。RULE.txt のとおり。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（固定 2026-11-05T16:00:58Z） |
| `scripts/dgx-run-0140.sh`・`scripts/dgx-0140-loop.sh` | 1 round・5 round ループ（本計測で使った版。ノードの sha256 がコミットと一致することを系列の起動時に照合した） |
| `gb10/run{1..5}/` | 各 round の JSONL（`results-dgx-0.14.0.jsonl`＝run_all_cuda.sh 112 行、`-extra` 8 行、`-py` 28 行）・`tree.txt`・`rev-stamp.txt`・`env_info.txt`・`uptime_*.txt`・`gpu_util_before.txt`・`run_all_cuda.log`・`*.err`・`skipped-dgx-0.14.0.log` |
| `gb10/run{1..5}.log`・`gb10/loop.log`・`gb10/gate.log`・`gb10/build.log`・`gb10/compute_apps_*.txt` | round のログ・ループのログ・ゲート・系列前のビルド・系列の前後の専有記録 |
| `gb10/*-median5.jsonl` | 5 run 中央値（本体・extra・Python） |
| `gb10/judge_0140.out` | 判定一覧（`judge_gb10_0140.py` の標準出力そのまま。再実行で byte 一致を確認） |
| `gb10/verify_0140.txt`・`gb10/tables_0140.md` | 要素検証と専有の集計・本 README の表の元（同スクリプトの出力） |
| `gb10/SHA256SUMS` | `gb10/` の全ファイルの sha256（マスク後の内容） |
| `smoke/` | RULE 固定前の行数スモーク（判定に使わない） |
| `env/pip-freeze-venv-bench.txt` | `pip freeze` ＋ `python --version`（0.13.0 と byte 一致） |
| `aggregate/median_rounds.py`・`check_identity.py`・`judge_gb10_0140.py` | 集計器（0.13.0 と byte 一致）・同一性検査（版の文字列のみ変更）・GB10 限定の判定器 |

ログのマスク: `/home/<ユーザー名>` は `<home>`、ノード名は `<cuda-node>`。`smoke/smoke-gb10.out` に cargo の絶対パスがあり、ほかの生ログに該当はなかった。JSONL の値は変えていない。

## 再現手順

```bash
# 0. 展開ツリー（単腕）。.rev-stamp は 40 桁 1 行
PIN=18a7f4b87fe61029a179cb1922cc82928440f2c9
git archive $PIN | tar -x -C tree && echo $PIN > tree/.rev-stamp
# 1. GB10: ~/work/r0140/tree、~/work/dgx-run-0140.sh、~/work/dgx-0140-loop.sh、~/work/bench_py.py、~/work/.venv-bench を置き、
#    実機ロック（mkdir ~/work/.fandhe-hw-lock と owner ファイル）を取ってから
mkdir -p ~/work/dgx-0140-logs
ROOT=~/work/r0140/tree EXPECT_SHA=$PIN setsid nohup bash ~/work/dgx-0140-loop.sh > ~/work/dgx-0140-logs/loop.log 2>&1 &
#    ログは ~/work/dgx-0140-logs/（run{1..5}/・gate.log・build.log）。回収して gb10/ に置く
# 2. 集計（このディレクトリで）
g=gb10; python3 aggregate/median_rounds.py $g/results-dgx-0.14.0-median5.jsonl $g/run{1..5}/results-dgx-0.14.0.jsonl   # extra・py も同様
python3 aggregate/check_identity.py $PIN results-dgx-0.14.0.jsonl=112,results-dgx-0.14.0-extra.jsonl=8,results-dgx-py-0.14.0.jsonl=28 gb10/run{1..5}
python3 aggregate/judge_gb10_0140.py --emit-verify gb10/verify_0140.txt --emit-md gb10/tables_0140.md > gb10/judge_0140.out
```

## スコアボード

本ディレクトリに `scoreboard/` は作らない（#3876 の範囲）。#3876 は `gb10/run{1..5}/`・`gb10/*-median5.jsonl` を入力に使える（配置と命名は 0.13.0 と同じ）。

## 所有者判断を仰ぐ点

1. **gemm CPU N=512 の 1 位（1.0038）**: 規則どおり 1 位だが、run 別は 2 run が 1 未満で、対照系列が無いため揺れとの区別はできない。規則は変えていない。
2. **gemm CPU N=1024〜4096 の fandhe-ai の 1.03〜1.05 倍**: 相手はほぼ同じなのに fandhe-ai だけが遅い側に動いた。前版比は表示のみで勝敗には使わない規則で、区分は変わっていない。日またぎの揺れとライブラリの変更を本計測からは分けられない。切り分けをするかどうかは判断待ち。
