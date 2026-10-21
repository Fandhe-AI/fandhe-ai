# metal-infer-bimodal-macmini-3183

イシュー #3183（Mac mini の infer Metal reuse の二峰性の原因特定。診断のみ）の計測ログ置き場。
背景・仮説・限界・結果は `docs/perf/metal-infer-bimodal-macmini.md`。

## 構成

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録の判定規則（実測前に単独コミットで固定。事後に緩めない） |
| `orchestrate.sh` | テストバイナリ 2 本を 1 回だけビルドし、run ごとに腕の起動順をローテーションして独立 5 回実行する。`run{1..5}.jsonl`・`env_info.txt`・`env_run{N}.txt`・`load_gate*.{log,txt}`・`power_run{N}.txt` を残す（`macmini` のみ受け付ける。既存 run があれば中止） |
| `aggregate.py` | 5 run を集計（python3 標準ライブラリのみ。n≠500・非有限／負値・JSON 不正・腕／run／ゲート状態の欠落・checksum 不一致は fail-closed。`--self-test` あり） |
| `macmini/` | Mac mini の実測生ログ（第 1 ラウンド・2026-10-21。`rev-stamp.txt`・`prereq_bit_exact.log`・`aggregate.md`・`bench_control/` を含む） |
| `RULE2.txt` | 第 2 ラウンドの事前登録規則（RULE.txt は変えず別に固定。実測前に単独コミットで固定） |
| `stage-tree.sh` | 転送ツリーを `git archive` から作り、`.rev-stamp`・`.tree-manifest` を書く（gitfile 混入とツリー指紋の欠落への対処） |
| `macmini-r2/` | 第 2 ラウンドの実測生ログ（`head/`・`v0110/`・`bench_control/`・`run_meta.txt`・`aggregate.md`）と、第 1 ラウンドへの事後適用 `posthoc-r1.md`（事後・探索で判定に使わない） |

## 第 2 ラウンドの手順（`--r2`）

第 1 ラウンドで残った課題（commit_wait の内訳・B 水準の再現・版差・実行順）を `RULE2.txt` で切り分ける。条件は HEAD の R／D／Don／G、v0.11.0 ツリーの R／D／Don、bench 対照の Bc／BcT の 9 条件を run ごとに巡回ローテーションする。

```bash
# 1. 転送ツリーを作る（dest は mktemp -d で作った空ディレクトリ。同じ呼び出しの中で rsync して削除する）
bash docs/perf/logs/metal-infer-bimodal-macmini-3183/stage-tree.sh HEAD "${stage}"
bash docs/perf/logs/metal-infer-bimodal-macmini-3183/stage-tree.sh v0.11.0 "${stage_v0}" \
  --overlay crates/facade/tests/metal_infer_bimodal_diag.rs
# 2. rsync（`.git` は末尾スラッシュなしで gitfile も除く。.env*・*.local.md は渡さない）
rsync -a --delete --exclude '.git' --exclude '.env*' --exclude 'real-hardware-verification-env.local.md' \
  "${stage}/" "${METAL_NODE}":~/work/rust-ai-library-run-3183/
rsync -a --delete --exclude '.git' --exclude '.env*' --exclude 'real-hardware-verification-env.local.md' \
  "${stage_v0}/" "${METAL_NODE}":~/work/rust-ai-library-run-3183-v0110/
# 3. ノード側（zsh -lc。nohup で切り離して生存確認。他の計測と衝突しないよう issue 番号付きの専用ディレクトリを使う。HEAD は CARGO_TARGET_DIR=$HOME/work/target-3183、v0110 は V0_TREE・V0_TARGET_DIR（target-3183-v0110）、
#    bench は FC_TARGET_DIR（target-3183-fc）の別 target。ビルドと計測を並走させない）
V0_TREE="${HOME}/work/rust-ai-library-run-3183-v0110" V0_TARGET_DIR="${HOME}/work/target-3183-v0110" \
  bash docs/perf/logs/metal-infer-bimodal-macmini-3183/orchestrate.sh macmini --r2 --out "${HOME}/work/out-3183-r2"
# 3b. 計測出力（ノードの ~/work/out-3183-r2/）を開発機のリポジトリ内 macmini-r2/ へ回収する（posthoc-r1.md を消さないため --delete は付けない。再計測前に既存の head/・v0110/・bench_control/ を退避または削除する。集計は回収後のこのディレクトリを読む）
rsync -a "${METAL_NODE}":~/work/out-3183-r2/ docs/perf/logs/metal-infer-bimodal-macmini-3183/macmini-r2/
# 4. 集計（出力はそのまま macmini-r2/aggregate.md にする）
python3 -I docs/perf/logs/metal-infer-bimodal-macmini-3183/aggregate.py --rule 2 \
  docs/perf/logs/metal-infer-bimodal-macmini-3183/macmini-r2
```

- `orchestrate.sh` はツリーの `.rev-stamp`（1 行目 = HEAD）を env_info の `git_head` に書き、`.tree-manifest` を `shasum -a 256 -c` で照合する（無い・不一致なら計測前に停止）。`tree_fingerprint` は `.tree-manifest` の sha256
- v0.11.0 ツリーがビルドできない、またはノードの空きが 20 GiB 未満の場合は `v0110/NOT_RUN.txt` に理由を残して残りの条件で続ける
- openrsync の `--delete-excluded` は DGX 側の framework-compare の target を消すことがあるため、bench-fandhe のビルドは `--r2` 内で必ずやり直す
- 開発機（M4 Max）でのスモークは出力形式の確認用で、計測として扱わずコミットしない

## 実行手順（第 1 ラウンド。Mac mini セッション）

転送・SSH は `docs/real-hardware-verification-env.md` §7 に従う（`METAL_NODE` の実値は `docs/real-hardware-verification-env.local.md`）。計測中はほかのビルド・計測を並走させない。

```bash
# 0. 前提確認: 分解と公開 API の bit 一致（既存テスト）
cargo test --release -p fandhe-ai --test metal_infer_tape_build_fixedcost_diag \
  -- --ignored --test-threads=1 --nocapture metal_infer_decomposition_matches_public_api_bit_exact
# 1. 独立 5 run（nohup で切り離し、生存確認する。macOS には setsid がない。第 1 ラウンドも .rev-stamp・.tree-manifest が要る）
bash docs/perf/logs/metal-infer-bimodal-macmini-3183/orchestrate.sh macmini
# 2. 集計
python3 -I docs/perf/logs/metal-infer-bimodal-macmini-3183/aggregate.py \
  docs/perf/logs/metal-infer-bimodal-macmini-3183/macmini \
  > docs/perf/logs/metal-infer-bimodal-macmini-3183/macmini/aggregate.md
```

- 同セッションで bench-fandhe（crates.io 0.11.0・変更なし）の `infer metal 64 reuse` を 5 回実行し `macmini/bench_control/` へ記録する（記録のみ）
- 収録前にホスト名は `masked`、`$HOME` は `<home>` に置換する（`orchestrate.sh` の `mask`）。実ホスト名はコミットしない
- 開発機（M4 Max）でのスモークは出力形式の確認用で、計測として扱わずコミットしない
