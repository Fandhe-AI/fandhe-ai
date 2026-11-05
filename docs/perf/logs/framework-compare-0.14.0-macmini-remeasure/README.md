# 0.14.0 の 6 フレームワーク同席計測 — Mac mini（Apple M4）

**状態: 実測済み（2026-11-05 UTC。系列 B 5/5 完走）。** crates.io `fandhe-ai =0.14.0` を Mac mini の系列 B で計測した（#3874。GB10 は #3875 が別ディレクトリ・別 PR で行う）。GB10 の値とは混ぜない。事前登録規則 `RULE.txt` は 2026-11-05T15:56:36Z に固定し（コミット `db6d816e7`）、以後編集していない。`RULE.v2.txt` はない。スコアボード HTML の生成（#3876）・Artifact の公開・対応表の更新はしていない（範囲外）。

## 0.13.0 版からの差分

規則の違いは腕・版・前版比の参照先・前版 tally の参照先の 4 点だけ（`RULE.txt`）。判定規則・N=5・ゲート・期待行数・既定設定（SME OFF・Graph capture OFF）・tolerance・baseline は同一。そのほかの違いは次のとおりで、いずれも規則の変更ではない。

| 項目 | 0.13.0 | 0.14.0 |
|---|---|---|
| 腕 | `5d05a5023`（#3709） | `18a7f4b87`（#4063） |
| 前版比の参照先 | 0.12.0 の median5（Mac mini 2 本） | 0.13.0 の median5（Mac mini 2 本。sha256 は `RULE.txt` に固定） |
| 前版 tally の参照先 | `gen_0120.out` | `gen_0130.out`（27 セル。Mac mini の比較は同ファイルの Mac mini 13 行） |
| RULE の事前確認の文面 | `*.rs` の差 0 件 | `VERSION` 定数と版の文字列・コメントだけ（`bench-fandhe` の版定数が JSONL の `version` 欄になるため） |
| 判定一覧の出力元 | GB10 側の `gen_0130.py`（`--gb` 必須・27 セル） | `aggregate/judge_macmini_0140.py`（Mac mini 13 セル。`load`・`key`・`index`・`metric`・`better`・`paired_ratio`・`judge` は `gen_0130.py` の逐語コピー） |
| 起動役 | 手でロック・手で解放 | `scripts/macmini-0140-series.sh`（ロックの owner 確認・環境の記録・失敗時も trap で解放。ループ本体は版文字列以外 0.13.0 と同じ） |
| 収録の追加 | なし | `env/env_info_*.txt`・`env/tree-fingerprint-*.txt`・`SHA256SUMS` |
| パスのマスク | `~` | `<home>` |

## 事前確認（RULE 固定前に確かめた結果）

`scripts/bench/framework-compare` の 0.13.0 腕 `5d05a50235138645b0b069060ed54dfd2c46db7f` と ARM `18a7f4b87fe61029a179cb1922cc82928440f2c9` の差分（`git diff`）:

- [x] `*.rs`: 変更は `bench-fandhe/src/main.rs` だけ。`0.13.0` を `0.14.0` に読み替えると差 0 行。変わるのは `VERSION` 定数・`MEASURE_ERROR` の文言・コメントの版の記述だけ。
- [x] `Cargo.toml`: 変更は `bench-fandhe/Cargo.toml` だけ。読み替えで差 0 行（`fandhe-ai` の版の行とコメント）。
- [x] `Cargo.lock`: パッケージ集合の差は fandhe-ai 系 7 クレートの版（0.13.0→0.14.0）と checksum、新規の `fandhe-ai-backend-rocm 0.14.0` の 1 件だけ。パッケージ数 658→659。外部 crate の増減・版変更 0（`libloading 0.9.0` は既に lock にある）。
- [x] `deny.toml`: `RUSTSEC-2026-0429` の ignore 追加と、その説明コメントだけ（#3962）。計測に影響しない。
- [x] 上の一覧にない差分（記録のみ）: `run_all.sh`・`run_all_cuda.sh`・`run_ab_*.sh` 7 本・`compare_gemm_ab.py` は読み替えで差 0 行。`README.md` はピン履歴の追記。本計測のスクリプトはこれらを呼ばない。
- [x] 常駐入力の判別式 `^[[:space:]]+let c0 = tape\.matmul_resident\(` は計測ツリーで 1 件（`env/env_info_before.txt`・`env_info_after.txt` の `arm_re_count=1`）。run ごとの検査は 0.13.0 と同じ `grep -Eq` で「1 件以上」を見る。件数の厳密な確認は系列の前後の 2 回で、前後のツリー指紋は一致している。
- [x] `bench_py.py` の sha256 は `9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b`（0.13.0 と同値。run ごとに照合）。
- [x] Python 3 FW: `env/pip-freeze-venv-torch.txt`・`pip-freeze-venv-tf.txt` は 0.13.0 の `env/` と byte 一致（RULE 固定前にハッシュで確認し、計測後に取得して `cmp` で再確認）。Python 3.12.15。JSONL の `version` 欄は pytorch 2.14.0・tensorflow 2.16.2・scipy 1.18.1 で 0.13.0 と同じ。
- [x] 判定スクリプトの再現: `judge_macmini_0140.py --check-verbatim` が `gen_0130.py` との字句一致を確認。0.13.0 のデータ（`--run-ver 0.13.0`・前版は 0.12.0 の 2 本）で流した 13 行が `gen_0130.out` の Mac mini 13 行と byte 一致し、tally は `3 3 7`（0.13.0 README の値と同じ）。`--self-test` が通る（判定側の不合格・run 欠損・sha256 不一致で止まること、相手側の不合格が判定不能に数えられること、区分の境界）。
- [x] 行数スモーク（RULE 固定前・ビルド込み 1 round・値は判定に使わない）: 2026-11-05T15:51:48Z〜15:56:08Z、rows 48・pyrows 32・skipped 4（burn Metal GEMM 512〜4096 fresh のみ）・rc 0。`smoke/` に収録。

## 計測の要点

- 機体: Mac mini（`<metal-node>`・Apple M4・10 コア・16 GB）。`macmini-run-0140.sh` が run ごとに `machdep.cpu.brand_string == "Apple M4"` を検査する（5/5 完走。M4 Max の値は使っていない）。
- 腕: 単腕 ARM `18a7f4b87fe61029a179cb1922cc82928440f2c9`。全 5 run の `tree.txt` が `fandhe-ai v0.14.0`、`rev-stamp.txt` が RULE の SHA と一致（`aggregate/check_identity.py` が rc=0。行数 Rust 48・Python 32）。計測ツリーの指紋（`env/tree-fingerprint-*.txt`）は系列の前後で同じで、`git archive` の ARM の framework-compare とも一致する。
- ビルド: 計測ツリーの `target/` を新規に作り、ロック内の行数スモークで 1 回ビルドした（1 分 35 秒）。系列の事前ビルドは `Finished … in 0.57s`（再コンパイルなし）で、計測中のビルド・並走はしていない。`CARGO_TARGET_DIR` は未設定（`env/env_info_*.txt`）。ツールチェーンは rustc 1.99.0 で、0.13.0 の計測ツリーの `.rustc_info.json` と同じ。
- 系列: **系列 B のみ**（各 run 前に load1 < 3.0 を最大 30 分待つゲート付き・5 run）。**系列 A（対照）は行わない**ため、ノイズ帯は「対照系列なし・未算出」（0 件ではない）。
- 日時・負荷: 2026-11-05 15:56:54Z〜16:08:35Z（`gate.log` の最初〜run5 終了。`loop-macmini.log`）。開始時 load1 = 2.37／1.89／2.04／1.97／2.34（`gate.log` 10 行・各 run の `uptime_before.txt`）。全 5 run の前で最初のサンプルが 3.63・3.12・3.07・3.65・3.63 で、1 分ずつ待ってから通過した。`GATE-INVALID`・`GATE-TIMEOUT` なし。load5／load15 は系列の間 6.73→3.21／5.20→3.92 で、スモークのビルドと直前の run の残りを含む。
- 完走・skipped: `loop-macmini.log` が `ALL-DONE completed=5/5`、全 5 run の最終行が `done. rows=48 pyrows=32 skipped=4`。skipped は burn Metal GEMM N=512〜4096 fresh の既知の記録拒否 4 件（`MEASURE_ERROR: gemm checksum is degenerate (0)`）だけで、5 run の `skipped-macmini-0.14.0.log` は同一内容・0.13.0 の同ログとも同一。
- 要素検証: 判定側（fandhe-ai reuse）は gemm 9 セル × 5 run の 45 行すべて `parity_fail_count = 0`（要素総数 139,591,680）。相手側を含めて不合格の行は 0 件で、判定不能は 0。**`parity_fail_count` は gemm の行だけにあり、train・infer の行にはない**（0.13.0 も同じ）。train・infer の値の確認は `checksum` で、全 80 セルが 5 run で一致し（`checksums_agree`）、0.13.0 の中央値行とも一致する。ただし checksum は丸めた値で、bit 同一の証拠ではない。
- 実機ロック: `~/work/.fandhe-hw-lock`（owner `#3874 rel-0.14.0 2026-11-05T15:51:45Z`）。行数スモークの前に取り、16:08:39Z に `macmini-0140-series.sh` の trap で解放した。
- 実機テスト（#3256 の M1〜M6・#3272 の X1）: 未実施（範囲外。`RULE.txt`）。

## 結果（同一 run 内比の 5 run 中央値。判定は `aggregate/judge_macmini_0140.py`）

| 条件 | 1 位 | 僅差 | 負け | 判定不能 | ノイズ帯 | 計 |
|---|---|---|---|---|---|---|
| 0.14.0（Mac mini 13 セル） | **3** | 3 | 7 | 0 | 未算出（対照系列なし） | 13 |
| 0.13.0 の Mac mini 13 セル（`gen_0130.out` の Mac mini 行） | 3 | 3 | 7 | 0 | 未算出（対照系列なし） | 13 |

`gen_0140_macmini.out` の tally 行: `tally 3 3 7 invalid 0 noise na total 13`。前版の全体の tally は `tally 13 3 11 invalid 1 noise na total 27`（GB10 を含む 27 セル。`../framework-compare-0.13.0-remeasure/scoreboard/gen_0130.out`）で、本 tally と直接比べる値ではない。

**0.13.0 からの区分変化は 13 セルとも 0 件**（`aggregate/compare_0130.txt`）。

### セル表（`aggregate/gen_0140_macmini.out`）

各列は「区分 比（vs 最速相手）」。比は同一 run 内比の 5 run 中央値。★ = 0.14.0 の比が 1.0 以下。「0.13.0 比」は fandhe-ai 中央値同士の時間比（推論はスループット比。<1 が高速化・推論は >1 が改善）で、表示のみ・勝敗には使わない。2026-10-31 の計測との比なので、計測日の負荷・機体の状態の揺れを含む（ライブラリの変更だけの効果ではない）。

| セル | 0.14.0 | 0.13.0 | 0.14.0 の run 別比 | 0.13.0 比（fandhe-ai 中央値同士） |
|---|---|---|---|---|
| gemm Metal N=256 | 1 位 1.13×（vs candle） | 1 位 1.15×（vs candle） | 1.16 1.13 1.11 1.13 1.30 | 0.99× |
| gemm Metal N=512 | 1 位 1.07×（vs candle） | 1 位 1.08×（vs candle） | 1.49 1.07 1.23 1.07 0.95 | 1.01× |
| gemm Metal N=1024 ★ | 僅差 0.98×（vs candle） | 僅差 0.99×（vs candle） | 0.98 0.99 1.12 0.97 0.97 | 1.02× |
| gemm Metal N=2048 ★ | 僅差 0.94×（vs candle） | 僅差 0.94×（vs candle） | 0.94 0.94 0.94 0.94 0.94 | 1.00× |
| gemm Metal N=4096 ★ | 2 位 0.88×（vs candle） | 2 位 0.88×（vs candle） | 0.88 0.85 0.89 0.87 0.88 | 1.03× |
| gemm CPU N=256 ★ | 3 位 0.69×（vs pytorch） | 3 位 0.71×（vs pytorch） | 0.69 0.69 0.69 0.72 0.68 | 1.01× |
| gemm CPU N=512 ★ | 2 位 0.75×（vs pytorch） | 2 位 0.76×（vs pytorch） | 0.75 0.74 0.76 0.77 0.74 | 1.00× |
| gemm CPU N=1024 ★ | 2 位 0.66×（vs pytorch） | 2 位 0.70×（vs pytorch） | 0.65 0.84 0.66 0.69 0.65 | 1.00× |
| gemm CPU N=2048 ★ | 2 位 0.53×（vs pytorch） | 2 位 0.52×（vs pytorch） | 0.53 0.54 0.53 0.52 0.53 | 1.01× |
| train Metal ★ | 僅差 0.96×（vs pytorch） | 僅差 0.95×（vs pytorch） | 0.96 0.96 0.95 0.99 0.96 | 0.99× |
| infer Metal | 1 位 1.02×（vs candle） | 1 位 1.02×（vs candle） | 1.03 0.99 1.00 1.03 1.02 | 1.00× |
| train CPU ★ | 3 位 0.24×（vs pytorch） | 3 位 0.28×（vs pytorch） | 0.24 0.23 0.28 0.24 0.23 | 1.01× |
| infer CPU ★ | 4 位 0.15×（vs pytorch） | 4 位 0.15×（vs pytorch） | 0.15 0.16 0.15 0.16 0.15 | 1.00× |

### 前版比で目立つセル

いずれも区分は変わらない。値は 5 run 中央値（JSONL の `median_s`）。理由は調べていない（本計測は区間の切り分けをしない）。

| セル | fandhe-ai reuse（0.13.0→0.14.0） | 最速の相手（0.13.0→0.14.0） | 比の動き |
|---|---|---|---|
| gemm Metal N=4096 | 62.712→64.379 ms（1.03×） | candle 55.303→56.103 ms | 0.88→0.88 |
| gemm Metal N=1024 | 2.308→2.345 ms（1.02×） | candle 2.298→2.278 ms | 0.99→0.98 |
| train CPU | 0.4957→0.5017 ms（1.01×） | PyTorch 0.1365→0.1185 ms | 0.28→0.24 |
| gemm CPU N=1024 | 5.329→5.338 ms（1.00×） | PyTorch 3.734→3.514 ms | 0.70→0.66 |

train CPU と gemm CPU N=1024 の比の低下は、fandhe-ai がほぼ動いておらず、相手 PyTorch の窓が縮んだ分による。gemm Metal N=512 は run 別比の幅が 0.95〜1.49 と広く、run5 だけ 1.0 を下回る（中央値は 1.07 で 1 位）。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（固定 2026-11-05T15:56:36Z・コミット `db6d816e7`） |
| `scripts/macmini-run-0140.sh`・`scripts/macmini-0140b-loop.sh` | 1 round・系列 B（0.13.0 版の版文字列だけを変えたもの） |
| `scripts/macmini-0140-series.sh` | 起動役（ロックの owner 確認・環境の記録・解放） |
| `macmini-series-b/run{1..5}/` | 各 run の `results-macmini-0.14.0.jsonl`（48 行）・`results-macmini-py-0.14.0.jsonl`（32 行）・`tree.txt`・`rev-stamp.txt`・`run.log`・`uptime_*.txt`・`skipped-macmini-0.14.0.log` |
| `macmini-series-b/results-macmini-0.14.0-median5.jsonl`・`results-macmini-py-0.14.0-median5.jsonl` | 5 run 中央値（`median_rounds.py` の再実行で byte 一致を確認） |
| `macmini-series-b/gate.log`・`macmini-series-b/build/` | ゲートのログ・事前ビルドの検査とビルドのログ（結果 JSONL は作らない） |
| `loop-macmini.log` | 系列 B のループと起動役のログ |
| `smoke/` | RULE 固定前の行数スモーク（判定に使わない） |
| `env/` | `pip-freeze-venv-*.txt`・`env_info_before.txt`／`env_info_after.txt`・`tree-fingerprint-before.txt`／`tree-fingerprint-after.txt` |
| `aggregate/median_rounds.py` | 0.13.0 版のコピー（変更なし） |
| `aggregate/check_identity.py` | 0.13.0 版の版文字列だけを変えたもの |
| `aggregate/judge_macmini_0140.py`・`gen_0140_macmini.out`・`compare_0130.txt` | 判定スクリプト・判定一覧・0.13.0 との区分の突き合わせ |
| `SHA256SUMS` | `README.md` と自身を除く全ファイルの sha256（マスク後の値。`shasum -a 256 -c SHA256SUMS`） |

作業ディレクトリ名 `macmini-0140b/` は収納時に `macmini-series-b/` へ改名した（0.13.0 と同じ）。回収時にパスのユーザー名を `<home>` へマスクした（`smoke/smoke-macmini.out` の `Compiling` 行のみ）。

## 再現手順

```bash
# 0. 展開ツリー（単腕）。.rev-stamp は 40 桁 1 行
PIN=18a7f4b87fe61029a179cb1922cc82928440f2c9
git archive $PIN | tar -x -C "${HOME}/work/r0140/tree" && echo $PIN > "${HOME}/work/r0140/tree/.rev-stamp"
# 1. 作業ディレクトリ D に scripts/ の 3 本・py/bench_py.py・venv-torch・venv-tf を置き、実機ロックを取ってから
#    （macOS は bash 3.2・setsid なし。ROOT・D・EXPECT_SHA は export が必要。実行は zsh -lc で包む）
export ROOT="${HOME}/work/r0140/tree" D="${HOME}/work/r0140" EXPECT_SHA=$PIN
(nohup bash "${D}/macmini-0140-series.sh" "<ロックの owner>" > "${D}/loop-macmini.log" 2>&1 < /dev/null &)
# 2. 集計（本ディレクトリで。macmini-series-b は作業ディレクトリの macmini-0140b を改名したもの）
M=macmini-series-b; L=..
python3 aggregate/median_rounds.py $M/results-macmini-0.14.0-median5.jsonl $M/run{1..5}/results-macmini-0.14.0.jsonl   # py も同様
python3 aggregate/check_identity.py $PIN results-macmini-0.14.0.jsonl=48,results-macmini-py-0.14.0.jsonl=32 $M/run{1..5}
P=$L/framework-compare-0.13.0-macmini-remeasure/macmini-series-b
python3 aggregate/judge_macmini_0140.py --m4-dir $M --prev $P/results-macmini-0.13.0-median5.jsonl $P/results-macmini-py-0.13.0-median5.jsonl \
  --prev-out $L/framework-compare-0.13.0-remeasure/scoreboard/gen_0130.out --compare-out aggregate/compare_0130.txt > aggregate/gen_0140_macmini.out
```
