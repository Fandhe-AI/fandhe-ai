# 結線後の前後比較（v0.14.0 対 main HEAD）と 14 セルの分類の付け直し（#4168）

Phase 4（#4165〜#4167）の結線後の main HEAD が 0.15.0 でどこまで届きそうかを、v0.14.0 と同じ run 内の前後比較で見積もる。合わせて `unreached_0140.tsv` の 14 セルの分類・理由を 0.15.0 用に直した原稿を作る。ADOPT／REJECT の二値判定は行わない（腕比は参考で合格ラインなし）。

## 状態

**部品のみ・未実測（PR-1）。** ランナー・集計器・テスト・RULE 雛形・起動役・分類の原稿（比は未記入）を置いた段階で、実機の値は一つもない。#4168 は実測と記録の PR-2 で閉じる（部品の PR は `Refs #4168`）。

- 検証済み（Linux）: 集計器の単体テスト 17 件・`--check-verbatim`（`gen_0140.py` と 7 関数が字句一致）・`--regress-0140`（0.14.0 の実データを単腕で流し `gen_0140.out` の 27 行と比・区分・run 別が一致）・ランナーの `--dry-run`（RULE の未記入拒否・ゲート定数 29 件の照合・`cargo tree --depth 2` による after 腕の path 解決確認）・`bash -n`・`shellcheck`・`check-forbidden-deps.sh`（framework-compare の契約検査）。
- **未検証**: 実機でのビルド・ゲート・計測、GB10 の専有確認、Mac mini のロック取得・解放、マスクの grep 検査、`.rev-stamp` と `bench_py.py` の照合。

## 事前調査の結果（2026-11-09）

- 依存 3 件はいずれも結線 0 件で閉じた: #4165（PR #4203。CPU 系の ADOPT 0 件・ゲート定数は false）、#4166（PR #4204。Metal STEEL の結線は保留・`STEEL_LARGE_N_ENABLED=false`）、#4167（PR #4202。CUDA Graph の推論既定化は見送り）。
- Phase 2・3（#4146〜#4163）は道具立ての PR だけで実測前に閉じており、分類を変える実測の根拠はない。NB-RANGE5・NB-SEP（#4142）と #4143 は所有者承認待ち。
- v0.14.0 計測腕（`18a7f4b87`）から main までの差: 新しいゲート定数は `STEEL_LARGE_N_ENABLED`・`SME_SMALL_SCHED_ENABLED` の 2 つでどちらも false。`bench-fandhe` の差は opt-in の `--diag-*` の追加が中心（`bench-fandhe/src/main.rs` の追加 250 行・削除 9 行。削除側は reuse の既定経路の呼び出し行の整理を含むため、実機セッションで `bench-fandhe-diff-stat.txt` と差分を読んで確認する）。事前仮説は after ≈ before だが、既定経路のコードにも差分があるため断定しない。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.template.txt` | 規則の雛形。実体化の手順を冒頭に書いた。after SHA・固定時刻・行数スモークは `<要記入`（残っていればランナーと判定器が拒否する） |
| `orchestrate.sh` | 実機の起動役（ロック取得・ランナー・判定器・収録・マスク検査・SHA256SUMS）。`--dry-run` あり |
| `unreached_0150.draft.tsv` | 0.15.0 用の分類の原稿（14 行）。比の列は `<要記入（#4168 実測後）>` |
| `../../../../scripts/bench/framework-compare/run_ab_prepost_0150_4168.sh` | A/B ランナー（#3266 の `run_ab_bench_fandhe_symmetry.sh` から派生。元は凍結） |
| `../../../../scripts/bench/framework-compare/judge_prepost_0150_4168.py`・`_test.py` | 集計器と単体テスト |

実測後に `judge-4168-{macmini,gb10}.md`・`.exit`・`ratios.tsv`・`run{1..5}/`・`env_info.txt`・`gate.log`・`hwlock.log`・`SHA256SUMS` が各機体のディレクトリに加わる。

## 腕

| 腕 | 内容 |
|---|---|
| before | registry の fandhe-ai `=0.14.0`（path patch なし・`--locked`）。0.14.0 スコアボードと同じライブラリなので `unreached_0140.tsv` の比と並べて読める |
| after | 同じ BASE の同じ `bench-fandhe` ソースを、`crates/facade` への path patch でビルド。`fandhe-ai` と `fandhe-ai-*` のすべてが `crates/` に解決されることを `cargo tree --depth 2` で確認する |

相手（candle・burn・Python）は BASE からビルドして両腕で共用する。after のビルドは `Cargo.lock` を書き換えるため、ランナーが退避し終了時に復元して差分 0 を確かめる。

## 実機セッションの手順（PR-2）

1. 開発機で BASE（計測時点の `origin/main` の SHA）を確定し、clone して `.rev-stamp` を置く。`RULE.template.txt` から `RULE.txt` を作り（固定時刻 UTC・after SHA・行数スモークの結果）、**RULE.txt だけを単独コミットして push する**。
2. GB10（`<cuda-node>`・compute プロセス 0 の専有）→ Mac mini（`<metal-node>`・Apple M4 のみ）の順に `AB_PROFILE=... bash docs/perf/logs/prepost-0150-4168/orchestrate.sh` を実行する。計測中はビルドを並走させない。
3. 判定器の出力・ログ・SHA256SUMS を収録し、`--fill-draft` で原稿の比を埋め、RULE の「分類の付け直し」の規則で分類・理由を確定する。ここで README の状態を更新し `Closes #4168`。

## マスク規則

公開面には機体名を `<metal-node>`・`<cuda-node>` とだけ書く。ホスト名・ユーザー名・ホームの絶対パス・IP は収録前に置換し、残りが 1 件でもあれば収録しない（`orchestrate.sh`）。実名は `docs/real-hardware-verification-env.local.md` を見るだけで転記しない。

## スコープ外

- #4156〜#4163 の再オープンと実測、#4142（NB-RANGE5・NB-SEP）・#4143 の所有者承認、既定の反転・SME の本番化。
- 0.15.0 の正式な `unreached_0150.tsv` とスコアボードの生成（#4179）、公開（#4174・#4175）、ピン更新（#4176）。
- `framework-compare-0.14.0-remeasure/scoreboard/` の変更（#4139 の範囲）。
