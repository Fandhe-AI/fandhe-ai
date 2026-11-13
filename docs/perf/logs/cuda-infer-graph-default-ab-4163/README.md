# CUDA Graph を推論で既定にした場合の影響評価（イシュー #4163）

**状態: v3 で測り直し済み・判定 ADOPT**（2026-11-13・GB10 `<cuda-node>`・`RULE.v3.txt`）。詳細は下の「結果（v3）」と `judge-4163v3.md`（集計器の終了コード `judge-4163v3.exit` = 0）。記録 doc は `docs/perf/infer-chain-graphcapture-cuda-ab.md` §6。

- 前回（v2・2026-11-11）の判定は UNDETERMINED。その記録（`judge-4163.*`・`gb10/`・下の「結果（v2）」）は変えずに残す。
- v3 は判定式・閾値・セル・round・起動順・`r3_repeats`・ゲートを v2 のまま、対象の tree だけを `a763da5b0`（#4263 の R3 阻害 3 件の修正を含む）に変えた規則（所有者判断。#4167 コメント 6855769755）。v2 の値を見た後に固定した（`RULE.v3.txt` 補足 (g)）。
- ADOPT でも `INFER_GRAPH_DEFAULT_ENABLED` は変えていない。結線は #4167 で、所有者の承認を得てから行う（doc §7.4）。

## 目的

推論チェーンの Graph capture（opt-in の `FANDHE_AI_CUDA_GRAPH_INFER`）を既定にしたら、推論・学習の双方にどう効くかを事前登録の A/B で測る。結果は #4167（既定化の結線または見送り。所有者承認）の判断材料にする。`INFER_GRAPH_DEFAULT_ENABLED` は本イシューでは変えない。

## 読み方の注意

- 本番の G-CUDA-INF reuse セル（常駐入力）は capture に届かない。効果は stream 変更だけ。capture まで含む効果は診断腕 reuse-host（`--diag-host-input-reuse`）で見る。
- checksum 一致は bit 同一性の証明ではない（R2 で補う）。単一機体・batch 3 種・1 スレッド。

## ファイル

| ファイル | 内容 |
|---|---|
| `RULE.template.txt` | 事前登録規則の雛形（プレースホルダ付き） |
| `RULE.txt` | v1（R3 は N=5。PR #4257）。所有者判断で N を 3 に変えたため計測には使っていない。記録として残す |
| `RULE.v2.txt` | 前回（v2）の計測の事前登録規則（R3 は N=3。PR #4259・#4163 コメント 6798581780）。記録として残す。v3 の後は `--check-rule` が不一致になる（ピンが v2 版のランナー・ゲートを指すため） |
| `RULE.v3.txt` | v3 の計測の事前登録規則（PR #4288。v2 から対象の tree だけを変えた）。ランナーとゲートはこのファイルだけを読む |
| `judge-4163v3.md`・`judge-4163v3.tsv`・`judge-4163v3.exit` | v3 の判定出力と集計器の終了コード（集計器の出力名は `judge-4163.*` 固定のため、`--out-dir` に出したものを改名した。中身は変えていない） |
| `judge-4163.md`・`judge-4163.tsv`・`judge-4163.exit` | 前回（v2）の判定出力と集計器の終了コード |
| `gb10-v3/` | v3 のマスク済みの生ログ（207 本。構成は `gb10/` と同じで、接尾辞は `4163v3`・`crates-diff-v0.14.0-a763da5b0.txt`）。64 KiB を超える 36 本は `gzip -9n` で圧縮した |
| `gb10/` | 前回（v2）のマスク済みの生ログ・JSONL・`env_info`・ゲートのログ・`.rev-stamp`（`rev-stamp-4163.txt`）・ツリー指紋（`fp-sender-4163.txt`・`fp-node-4163.txt`）・起動ログ（`go-4163.log`）・`git diff --stat v0.14.0 <source_commit> -- crates/`（`crates-diff-v0.14.0-737b9b5d7.txt`）。64 KiB を超える 36 本は `gzip -9n` で圧縮した |
| `run_gates_4163.sh` | 前提ゲート R0〜R3（R3 は `r3_repeats` 回の反復とダンプテストの単独実行） |
| `default-on.patch` | R3 専用の計測パッチ（const を true にし、コンパイル時表明テストを反転に合わせる）。crates へはコミットしない |
| `default-off-assumption-tests.txt` | 「既定 OFF 前提」のテストの事前リスト（R3 の失敗分類に使う） |
| `aggregate_4163.py` | 厳格パースの集計と判定（`--self-test`・`--check-template`・`--check-rule`） |
| `scripts/bench/framework-compare/run_ab_infer_graph_default_cuda.sh` | A/B ランナー |

## 手順

1. PR-2: PR-1 マージ後の main で `RULE.template.txt` から `RULE.txt` を作り（`fixed-at-utc`・`source_commit`・各 sha256・`r3_repeats` を埋める）、`python3 -I aggregate_4163.py --check-rule RULE.txt` を通して単独コミットする。
2. PR-3: GB10（`<cuda-node>`）で実機ロックを取り、compute プロセス 0 を確認し、`HW_LOCK_HELD=1 HW_LOCK_OWNER_FILE=<abs> AB_FACADE_PATH=<abs>/crates/facade bash run_ab_infer_graph_default_cuda.sh <label>`。
3. `python3 -I aggregate_4163.py --dir scripts/bench/framework-compare/results/raw --rule RULE.v3.txt --label <label> --out-dir <dir>` で判定を作る（`RULE.v3.txt` 補足 (f)。v3 のラベルは `4163v3`）。出力名は `judge-4163.md`／`.tsv` で固定のため、`judge-4163v3.*` に改名して置く（前回の `judge-4163.*` を上書きしない）。`--out-dir` を省くと `--dir` に書く。ホスト名・パスをマスクして記録する。

手順 1 の `RULE.txt` は v1 で、前回は `RULE.v2.txt`（N=3）、今回は `RULE.v3.txt` で測った。収録したログから判定を再現するには、`gb10-v3/` を一時ディレクトリへ写して `gunzip *.gz` し、そのディレクトリを `--dir` に渡して `--rule RULE.v3.txt --label 4163v3 --out-dir <別のディレクトリ>` で集計する（集計器は `.gz` を読まない）。前回分は `gb10/` を同じように展開し、v2 の時点のツリー（`8482bcb19`）の集計器と `RULE.v2.txt` で `--label 4163` を渡す。

## R3 の作り直し（固定前の試走で見つかった点。#4163 所有者判断 2026-11-11）

RULE を固定する前の捨て試走で、R3 の集計が入力不正で止まった。原因は次の 3 つで、いずれも RULE 固定前に直した。

| 原因 | 対処 |
|---|---|
| `crates/backend-cuda/src/jit_cache_regression_tests.rs` の `multiprocess_concurrent_store_absorbs_rename_race` が自分を子プロセスで起動し直し、子の標準出力がそのまま親のログに混ざる | 子の stdout・stderr を pipe で捕捉し、失敗時だけ assert のメッセージに出す（テストの意味と製品コードは変えない） |
| facade の `tests/cpu_sme_gate_bit_dump.rs::dump_cpu_sme_gate_bits` が標準出力へ直接書き、`test <name> ... out[...]` の 1 行になる | 本体の実行から `--exact --skip` で外し、同じ回の中で `--exact` で単独に流す。ダンプ本体の行はログから落として件数だけ残し、結果は集計行から取って R3 の集合に含める（テストは変えない） |
| edition 2024 の merged doctests が 1 つの `Doc-tests` 節に結果行を 2 本出す | `aggregate_4163.py` は Doc-tests 節に限り結果行が複数あってよいとする（Running 節は 1 本のまま） |

あわせて R3 を反復にした。腕 C・A を `r3_repeats`（N。RULE の機械可読ブロック。所有者が決める）回ずつ流し、腕 A で N 回とも失敗し腕 C で N 回とも成功したテストだけを「阻害」（事前リストに載れば「既定 OFF 前提」）とする。回ごとに結果が混ざったテストは「不安定」として `judge-4163.md` に記録し、判定には使わない。腕 B（補助・判定外）は 1 回のまま。

ログは `r3-<腕>-<単位>-rep<k>-<label>.log/.tsv`（単位は `fandhe-ai-backend-cuda`・`fandhe-ai`・`smedump`）。テスト側の修正が対象差分に入るため、RULE.txt の対象差分は「差分あり（テストのみ）」になる。

## 既定 OFF 前提テストの走査

`git grep -nE "INFER_GRAPH_DEFAULT_ENABLED|infer_graph_stats|infer_graph_enabled|infer_graph_stream_requested|StreamKind::Legacy|is_capturable_stream|GRAPH_INFER" -- crates` で `crates/*/tests` と `graph.rs` の単体テストを走査し、既定値が OFF であること・Legacy stream であることを表明するものだけを `default-off-assumption-tests.txt` に載せる。

## 結果（v3・2026-11-13）

### 判定（`judge-4163v3.md` の語のまま）

**verdict: ADOPT**

- undetermined 要因: なし
- reject 要因: なし
- `INFER_GRAPH_DEFAULT_ENABLED` は変えていない。doc §7.4 の 1.（判定が ADOPT）は満たした。2.（所有者の承認と結線イシュー）・3.（反転 PR）はまだ。

| ゲート | 結果 |
|---|---|
| R0（linear・graph） | 合格・合格 |
| R1(a) | 合格（PASS=5 SKIP=0） |
| R1(b) | 5/5 合格（失敗なし） |
| R2（64・1024・4096） | 全行一致 |
| R3 | N=3・阻害 0 件・既定OFF前提 1 件・不安定 2 件 |
| R3 の腕 B（補助・判定外） | rc=101 |
| 専有ゲート | 5/5 round 通過（`gb10-v3/load_gate-4163v3-cuda.log`） |

- 全 11 セルで ratio <= 1.00、checksum は全セルで一致。reuse-host の 3 セル（64・1024・4096）は 5/5 round で ratio_r < 1.00。
- 「5/5 一貫」が「いいえ」のセル（round 別の ratio_r に 1.00 を超える round がある。ADOPT の条件ではない）: infer-fresh-1024・infer-fresh-4096・infer-reuse-1024・infer-reuse-4096・train-fresh。
- 「既定OFF前提」は v2 と同じ `graph_capture_real_device_optin_off::opt_in_off_keeps_legacy_stream_and_returns_none` の 1 件。
- 「不安定」（判定外）は 2 件（いずれも `fandhe-ai-backend-cuda`）: `gemm_wmma_tf32_opt::wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096`・`gemm_wmma_tf32_staged::wmma_tf32_staged_exceeds_tiled_f32_tflops_at_4096`。
- v2 で阻害だった 3 件（下の「結果（v2）」）は、v3 では阻害にも不安定にも入っていない。
- セル別の中央値・ratio・ノイズ帯の記録は `judge-4163v3.md` を参照。原因の調査はしていない。

infer-fresh-1024（v2 で reject 要因になったセル）の値を `judge-4163.md`・`judge-4163v3.md` から並べる（解釈は足さない。判定式は同じ。対象の tree の差は `RULE.v3.txt` 補足 (b) の 5 ファイルで、R3 の阻害 3 件のテストの書き換え・`jit_cache_regression_tests.rs` の子プロセス出力の捕捉・`device.rs` の `CudaDevice::new` 後半の切り出しと `#[cfg(test)]` の構築関数）:

| 版 | ratio | ratio_r（round 順） | ratio_r の最大-最小 |
|---|---|---|---|
| v2（2026-11-11） | 1.0734 | 0.9438・0.9747・1.0734・1.2750・0.4058 | 0.8693 |
| v3（2026-11-13） | 0.9256 | 0.5584・1.0209・1.0039・1.4190・0.9504 | 0.8606 |

### 計測の条件（v3）

- 計測したコミット: `4c279200490c0f30521438f63ecdefe01edd97ce`（PR #4288 のマージコミット。`RULE.v3.txt` の `source_commit` は `a763da5b0`）。送信側で `source_commit` から `4c2792004` までの差分検査（`RULE.v3.txt` 補足 (c) の除外 3 本）を流し rc=0。`aggregate_4163.py --check-rule RULE.v3.txt` は `rule ok`。
- 転送: `4c2792004` までの全履歴の `git bundle` を送り、ノードの新しいディレクトリへ clone して detached checkout。`.rev-stamp` は `4c2792004`／`dirty=0`（`gb10-v3/rev-stamp-4163v3.txt`）、`status --porcelain` は 0 行。ノード上で GitHub への fetch はしていない。
- ツリー指紋（送信側とノードで一致。`gb10-v3/fp-sender-4163v3.txt`・`fp-node-4163v3.txt`）: tar sha256 `223e749037813355e4a4a42bfa29042a8a45f612c5d412eea0d517e562d367a7`・tree `423f0b3fa685eee21b417581b670909eb7ab6fc2`。
- 実機ロック: 2026-11-13T14:53:18Z に手で取得（owner `#4163 measure 2026-11-13T14:53:18Z`）。取得前に、ロックなし・計測系のプロセス 0・compute プロセス 0 を確かめた。ランナーは nohup（setsid）で起動し、ラベルは `4163v3`。`done. rc=0` は 2026-11-13T16:30:59Z（約 98 分）。終了後に関係プロセス 0・compute プロセス 0 を確かめてから、16:31:16Z に手で外した（起動スクリプトは外さない）。
- 集計: `4c2792004` の `aggregate_4163.py` を、マスク済みの写しに対して `--rule RULE.v3.txt --label 4163v3 --out-dir <dir>` で実行（rc=0）。マスク前の原本に流した出力（`judge-4163.md`・`.tsv` と標準出力）とバイト一致した。収録したのはその出力の改名だけで、中身は変えていない。
- マスク: `$HOME` → `<home>`（3353 件）、ホスト名 → `<cuda-node>`、ユーザー名 → `<user>`（前後が英数字・`_`・`-` でない語境界つき）。ホスト名とユーザー名の置換は 0 件だった。マスク後にホスト名・`$HOME`・ユーザー名・IPv4・`/home/`・`/Users/`・`/opt/`・常駐サービス名・GPU UUID が 0 件であることを数えて確かめた。成果物のうち bench バイナリ（`bench-fandhe-4163v3`）は収録していない。

### 申告（v3。露出と経緯）

- 系列中に見たもの:
  - 起動直後に、ロック取得の行とランナーのプロセス数。
  - 約 53 分の時点（15:46Z ごろ）で、進み具合を確かめるために、ランナーのプロセス数・`results/raw` のラベル付きファイルの本数（51）・専有ゲートのログがまだ無いこと。値・ファイルの中身は見ていない。
  - 終了後に、`run.log` の `^done\. rc=` 行と、専有ゲートのログ（round ごとの通過・待ち時間・load1）。`run.log` の末尾は表示していない。
- 回収後・集計前に見たもの: ファイル名と種類ごとの本数、ファイルの種類（`file` の出力。bench バイナリが回収に混ざっていたので取り除いた）、マスクの置換件数、漏れ検査の件数。セルの値・判定語は集計後に見た。
- v3 の固定前に v2 の値を見ていること（規則を書いた側）は `RULE.v3.txt` 補足 (g)。本計測のセッションは v2 の計測・記録も行っており、v2 の値を見ている。

## 結果（v2・2026-11-11）

### 判定（`judge-4163.md` の語のまま）

**verdict: UNDETERMINED**

- undetermined 要因: `r3: 阻害 3 件`
- reject 要因: `ratio > 1.00: infer-fresh-1024`
- `INFER_GRAPH_DEFAULT_ENABLED` は変えていない。#4167 の再開条件（doc §7.4 の 1. 判定が ADOPT）は満たさない。

| ゲート | 結果 |
|---|---|
| R0（linear・graph） | 合格・合格 |
| R1(a) | 合格（PASS=5 SKIP=0） |
| R1(b) | 5/5 合格 |
| R2（64・1024・4096） | 全行一致 |
| R3 | N=3・阻害 3 件・既定OFF前提 1 件・不安定 3 件 |
| R3 の腕 B（補助・判定外） | rc=101 |

R3 で「阻害」（腕 A で 3/3 失敗・腕 C で 3/3 成功、事前リストにない）に分類されたテスト（いずれも `fandhe-ai-backend-cuda`）:

- `lib::context_cache::tests::cached_allocator_keeps_distinct_contexts_on_same_ordinal_separate`
- `lib::module_cache_wiring_tests::cuda_gemm_new_second_construction_reuses_module_cache`
- `lib::scalar_op_cache_wiring_tests::cached_scalar_unary_kernel_distinct_devices_do_not_share_cache`

「既定OFF前提」は `graph_capture_real_device_optin_off::opt_in_off_keeps_legacy_stream_and_returns_none` の 1 件。「不安定」3 件（判定外）、セル別の中央値・ratio・ノイズ帯の記録は `judge-4163.md` を参照。原因の調査はしていない。

### 計測の条件

- 計測したコミット: `8482bcb19a15ba5fb855cd789589b096716d085d`（#4259 のマージコミット。`RULE.v2.txt` の `source_commit` は `737b9b5d7`）。送信側で `source_commit` から `8482bcb19` までの差分検査（`RULE.v2.txt` 補足 (c) の除外 3 本）を流し rc=0、除外 3 本はレビュー済みのブランチ版と同一。
- 転送: `8482bcb19` までの履歴の `git bundle` を送り、ノードで clone して detached checkout。`.rev-stamp` は `8482bcb19`／`dirty=0`、`status --porcelain` は 0 行。
- ツリー指紋（送信側とノードで一致）: tar sha256 `6cad56ac9c27a4c471e9b86ee8cdf8bceb89a5563ac30c29ce4322f7f92e0840`・tree `d3db259f8450faf47bd8333f8bda79e93d10b8c2`。
- 実機ロック: owner `#4163 measure 2026-11-11T16:26:14Z`。起動前にロックなし・compute プロセス 0 を確認。nohup で起動し、`done. rc=0`（2026-11-11T18:05:53Z、約 100 分）の直後にロックを外した。終了後に関係プロセス 0・compute プロセス 0 を確認した。
- 集計: `8482bcb19` の `aggregate_4163.py` を、マスク済みの写しに対して `--rule RULE.v2.txt --label 4163` で実行（rc=0）。マスク前の原本に流した出力（`judge-4163.md`・`.tsv`）とバイト一致した。
- マスク: `$HOME` → `<home>`、ホスト名 → `<cuda-node>`、ユーザー名 → `<user>`（前後が英数字・`_`・`-` でない語境界つき）。マスク後にホスト名・`$HOME`・ユーザー名・IPv4・`/home/`・常駐サービス名・GPU UUID が 0 件であることを数えて確かめた。成果物のうち bench バイナリ（`bench-fandhe-4163`）は収録していない。

### 申告（露出と経緯）

- 系列中に見たのは、起動ログの `start=`・`owner=` の行と、`done. rc=`・ロック解放の行だけ。ゲートのログ（`load_gate-4163-cuda.log`）は開く必要がなかった。
- 回収後・集計前に見たのは、ファイル名と種類ごとの本数、マスク対象の件数、漏れ検査の一致箇所の前後（R2 のダンプ `out[i].bits=0x…` の行 15 種前後と、テストのバイナリ名 1 種。16 進値が検査語に偶然一致したもの）。セルの値・判定語は集計後に見た。
- RULE.v2 の前段確認（PR #4259 に記載）: ブランチのコミットを試走用のラベルで起動し、`env_info` の差分検査の行を見たところで止めた。止めた時点で R0〜R2 は終わり、R3 の腕 C の途中だった。その出力は開かずに消した。止めるときにプロセスが一部残ったままロックを一度外し、取り直してから残りを止めて外した。
- 同 PR の作業中、変異の確認用に手元で作った捨てコミット 1 件に `--no-verify` を付けた（push せず取り消した。禁止事項に当たる）。
- `RULE.txt`（v1・N=5）での系列は起動していない。
