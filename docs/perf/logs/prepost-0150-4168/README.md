# 結線後の前後比較（v0.14.0 対 main HEAD）と 14 セルの分類の付け直し（#4168）

Phase 4（#4165〜#4167）の結線後の main HEAD が 0.15.0 でどこまで届きそうかを、v0.14.0 と同じ run 内の前後比較で見積もる。合わせて `unreached_0140.tsv` の 14 セルの分類・理由を 0.15.0 用に直した原稿を作る。ADOPT／REJECT の二値判定は行わない（腕比は参考で合格ラインなし）。

## 状態

**部品のみ・未実測（PR-1b で結線後の main に合わせて更新済み）。** ランナー・集計器・テスト・RULE 雛形・起動役・分類の原稿（比は未記入）を置いた段階で、実機の値は一つもない。#4168 は実測と記録の PR で閉じる（部品の PR は `Refs #4168`）。M4 Max は対象外（2026-11-14 所有者指示）。

- 検証済み（Linux）: 集計器の単体テスト 33 件・`--check-verbatim`（`gen_0140.py` と 7 関数が字句一致）・`--regress-0140`（0.14.0 の実データを単腕で流し `gen_0140.out` の 27 行と比・区分・run 別が一致）・ランナーの `--print-expected-consts`（ゲート定数 29 件）・`--print-tool-sha256`・`--dry-run`・`--smoke`（GPU なしのスタブで正常・失敗・ゲート失敗・拒否の各経路と結果ファイルの削除を確認）・`bash -n`・`shellcheck`。
- **未検証**: 実機でのビルド・ゲート・計測、GB10 の専有確認、Mac mini のロック取得・解放と他ジョブゲート（`foreign_job_gate.py`）、マスクの grep 検査、`.rev-stamp` と `bench_py.py` の照合。

## 前提の変遷（結線は 0 件ではなくなった）

- PR #4205（2026-11-09）の時点は「結線 0 件」の前提だった。その後、Phase 4 の結線が main に入った: #4166 の Metal STEEL（PR #4255・`STEEL_LARGE_N_ENABLED = true`・結線後の前後比較は PASS で記録 PR #4271）、#4165 の SME 既定 ON（#4156 が ADOPT〈#4273〉・PR #4284・`SME_PRODUCTION_ENABLED = true`）と N=256 の SME スケジュール t4-jpw1（#4162 が ADOPT〈#4283〉・`SME_SMALL_SCHED_ENABLED = true`）、推論スコープ内 jpw1（#4161 が #4266 で ADOPT〈#4269〉・PR #4270・aarch64 Linux だけ `INFER_SCOPE_JPW1_ENABLED` が ON）、#4167 の推論 Graph 既定 ON（#4163 v3 が ADOPT〈#4289〉・PR #4290・`INFER_GRAPH_DEFAULT_ENABLED = true`）。
- 旧ランナーの事前確認は「baseline と同じ値・後から増えた定数は false」だったため、いまの main では `precheck_fail` で止まる。そこで事前確認を「RULE の `[期待するゲート定数]` との完全一致」に変えた（結線は想定内。baseline との差は `consts-diff.txt` に記録だけする）。`INFER_SCOPE_JPW1_ENABLED` は値が 2 行にわたる `cfg!(...)` なので、継続行をつなぐ抽出に直した。
- 結線後の仮説は RULE.template.txt の `[結果の扱い]` に表明する（合格ラインではない）: Mac mini の CPU gemm は SME 既定 ON で、Metal G1024〜G4096 は steel で、GB10 は jpw1 と推論 Graph 既定 ON で動く見込み。G-CUDA-INF の本番セル（常駐入力）は capture に届かず、created stream 化の分しか動かない見込み（`docs/perf/infer-chain-graphcapture-cuda-ab.md` §7.5）。
- 道具の追加（PR-1b）: 環境変数 `FANDHE_AI_*`／`RUST_AI_*` の混入拒否、道具の sha256 の RULE 固定、Mac mini の他ジョブゲート（#4166 と同じ `foreign_job_gate.py`。各ラウンドの前後。所有者承認 2026-11-14）。
- NB-RANGE5・NB-SEP の定義は承認済み（#4142 issuecomment-6757497844・#4209）。本 issue では記録だけで、判定と分類には使わない（区分の確定は Phase 6）。#4142・#4143 の承認待ち・#4167 の見送りという旧記述は古い。
- 診断（#4146・#4148・#4149・#4150・#4153・#4154）は実測済みの記録がある（`docs/perf/loss-attribution-matrix.md` §10）。原稿の分類列は触っていない（診断を根拠にした分類の付け直しは、記録 PR が RULE「分類の付け直し」に従って決める）。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.template.txt` | 規則の雛形。実体化の手順を冒頭に書いた。after SHA・固定時刻・行数スモーク・`[期待するゲート定数]`・`[道具の sha256]` は `<要記入`（残っていればランナーと判定器が拒否する） |
| `orchestrate.sh` | 実機の起動役（ロック取得・ランナー・判定器・収録・マスク検査・SHA256SUMS）。`--dry-run` あり |
| `unreached_0150.draft.tsv` | 0.15.0 用の分類の原稿（14 行・結線後の前提に直した）。比の列は `<要記入（#4168 実測後）>` |
| `../../../../scripts/bench/framework-compare/run_ab_prepost_0150_4168.sh` | A/B ランナー（#3266 の `run_ab_bench_fandhe_symmetry.sh` から派生。元は凍結） |
| `../../../../scripts/bench/framework-compare/judge_prepost_0150_4168.py`・`_test.py` | 集計器と単体テスト |

実測後に `judge-4168-{macmini,gb10}.md`・`.exit`・`ratios.tsv`・`run{1..5}/`・`env_info.txt`・`gate.log`・`consts-diff.txt`・`foreign-gate/`（Mac mini）・`hwlock.log`・`SHA256SUMS` が各機体のディレクトリに加わる。

## 腕

| 腕 | 内容 |
|---|---|
| before | registry の fandhe-ai `=0.14.0`（path patch なし・`--locked`）。0.14.0 スコアボードと同じライブラリなので `unreached_0140.tsv` の比と並べて読める |
| after | 同じ BASE の同じ `bench-fandhe` ソースを、`crates/facade` への path patch でビルド。`fandhe-ai` と `fandhe-ai-*` のすべてが `crates/` に解決されることを `cargo tree --depth 2` で確認する |

相手（candle・burn・Python）は BASE からビルドして両腕で共用する。after のビルドは `Cargo.lock` を書き換えるため、ランナーが退避し終了時に復元して差分 0 を確かめる。

## 手順（順序が決まっている）

1. **PR-1b**（本更新）をマージする。
2. **行数スモーク**（RULE.txt の実体化前に、両機で PR-1b のマージコミットの clone〔BASE〕から。`--smoke` は RULE.txt を読まず、実ラウンドと同じビルド・セル・環境変数の拒否・ハードロック・機体の同定・ゲートを 1 ラウンドだけ流す）。報告するのは **rows／pyrows／skipped／rc だけ**で、値は見ない・残さない（結果ファイルは終了時に削除され、判定器は `--smoke` の出力を拒否する）。`AB_SMOKE_SUMMARY` は未存在の絶対パスで、件数・rc・UTC 時刻・機体プロファイルだけの要約が残る。

   ```
   # Mac mini（<metal-node>。Apple M4。他ジョブゲートつき）
   AB_PROFILE=macmini AB_BASE_ROOT=<BASE> AB_RUNNER_ROOT=<BASE> AB_PY_TORCH=<abs> AB_PY_TF=<abs> AB_SMOKE_SUMMARY=<abs> bash <BASE>/docs/perf/logs/prepost-0150-4168/orchestrate.sh --smoke
   # GB10（<cuda-node>。compute プロセス 0 の専有）
   AB_PROFILE=gb10 AB_BASE_ROOT=<BASE> AB_RUNNER_ROOT=<BASE> AB_PY=<abs> AB_SMOKE_SUMMARY=<abs> bash <BASE>/docs/perf/logs/prepost-0150-4168/orchestrate.sh --smoke
   ```

   あわせて両機でランナーの `--print-expected-consts`・`--print-tool-sha256`（RULE.txt 不要）を実行し、出力が開発機の生成結果と完全一致することを確かめる（mac の BSD awk での抽出の確認を兼ねる）。
3. **RULE.txt の単独コミット**: 開発機で BASE（計測時点の `origin/main`）を確定し clone して `.rev-stamp` を置く。`RULE.template.txt` から `RULE.txt` を作る（固定時刻 UTC・after SHA・行数スモークの結果）。`[期待するゲート定数]`・`[道具の sha256]` の 2 ブロックは、BASE で次の 2 コマンドの出力をそのまま貼る（機体に依存しない）。**RULE.txt だけを単独コミットして push する**。

   ```
   AB_BASE_ROOT=<BASE の絶対パス> AB_RUNNER_ROOT=<BASE の絶対パス> bash <BASE>/scripts/bench/framework-compare/run_ab_prepost_0150_4168.sh --print-expected-consts
   AB_BASE_ROOT=<BASE の絶対パス> AB_RUNNER_ROOT=<BASE の絶対パス> bash <BASE>/scripts/bench/framework-compare/run_ab_prepost_0150_4168.sh --print-tool-sha256
   ```
4. **GB10**（`<cuda-node>`・compute プロセス 0 の専有）の次に **Mac mini**（`<metal-node>`・Apple M4 のみ。他ジョブゲートつき）の順に `AB_PROFILE=... bash docs/perf/logs/prepost-0150-4168/orchestrate.sh` を実行する。計測中はビルドを並走させない。
5. 判定器の出力・ログ・SHA256SUMS を収録し、`--fill-draft` で原稿の比を埋め、RULE の「分類の付け直し」の規則で分類・理由を確定する。ここで README の状態を更新し `Closes #4168`。

## マスク規則

公開面には機体名を `<metal-node>`・`<cuda-node>` とだけ書く。ホスト名・ユーザー名・ホームの絶対パス・IP は収録前に置換し、残りが 1 件でもあれば収録しない（`orchestrate.sh`）。実名は `docs/real-hardware-verification-env.local.md` を見るだけで転記しない。

## スコープ外

- #4156〜#4163 の再オープンと実測、既定の反転・SME の本番化の追加（結線は #4165〜#4167 で済んでおり、本 issue は前後比較だけを行う）。NB-RANGE5・NB-SEP を判定・分類に使うこと（区分の確定は Phase 6）。
- 0.15.0 の正式な `unreached_0150.tsv` とスコアボードの生成（#4179）、公開（#4174・#4175）、ピン更新（#4176）。
- `framework-compare-0.14.0-remeasure/scoreboard/` の変更（#4139 の範囲）。
