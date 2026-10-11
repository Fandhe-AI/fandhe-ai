---
paths:
    - ".github/workflows/**"
---

# GitHub Actions ワークフロー編集ルール

## 目的

CI の実行時間・課金分数・ジョブ数を増やさずにワークフローを追加・変更できるようにする。
runner 方針・fail-closed 集約・SHA 固定などの基本規約は `ci.md` が正であり、本ルールは「発火条件」「必須チェック」「追加時のチェックリスト」「計測」を扱う。

## runner と visibility

- 本リポジトリは **public**（`gh repo view --json visibility`）。public は GitHub ホステッド（`ubuntu-latest`）が正
- self-hosted は使わない。例外は ai-review の `review` ジョブ（codex 専用 runner。`ai-review.yml` が呼ぶ reusable workflow 側で決まり、本リポ側の yml には runner 宣言が現れない）のみ。public で self-hosted を使うと fork PR 経由で任意コードが実行されうるため、例外を増やさない
- 逆戻りは `scripts/check-workflow-runner-policy.sh`（ci.yml の `runner-policy` ジョブ・`make runner-policy`）が検知する。macOS / Windows が必要な場合もまず Linux ホステッドのクロスターゲット検査（`build` ジョブ参照）で代替できないか検討する
- 詳細は `ci.md`「runner」節を参照

## 必須チェックと集約ジョブ

- ruleset `main-protection`（id `20587668`、classic branch protection は無い）の required status checks は `gh api repos/Fandhe-AI/fandhe-ai/rulesets/20587668` で確認する。`ci-complete` に加え、`ci.yml` の各ジョブ名（`name:`）、`rust-ci / ...`、`codex / preflight`・`codex / review`・`codex / post_feedback`、`Cursor Bugbot` が登録されている
- `ci.yml` のジョブを追加・改名する場合は `ci-complete` の `needs` と完遂判定への追加に加え、**ruleset 側の required contexts の更新**が必要（人間承認が要る設定変更のため `gh api` で単独更新しない）。更新を怠ると、追加ジョブは client-only context で自動マージを止め、改名は旧 context が永久に pending になる（`ci.md`「運用上の注意」）
- 必須チェックを含む workflow に**workflow 単位の `paths` / `paths-ignore` を `pull_request` へ付けない**（起動されないと必須チェックが pending のままマージ不能になる）。`push`（required checks 無し）にのみ付けてよい。PR で重い処理を省くときは、ジョブは起動して success させ、ジョブ内の変更検知で重いステップを省く（`scripts/ci-detect.sh` の `has_cargo` 出力。Markdown のみの変更では cargo 系ステップをスキップ）
- reusable workflow 呼び出しジョブ（`rust-ci`・`codex`）を `if:` で丸ごとスキップしない（内部の required context が報告されず pending になる）

## 発火条件の方針

- `ci.yml`: `pull_request`（`branches: [main]`）と `push`（`branches: [main]`）。push は Markdown のみの変更なら `paths`（否定パターン）で起動しない。`site/**`・`crates/**` は docs-site のテスト・doc 属性が実読込しうるため、また `docs/` 配下の `.json`（テスト入力）・`.sh` は文書ではないため、省略対象にしない
- `concurrency`: PR は `cancel-in-progress: true`、main への push と release / pages-deploy は取り消さない（`cancel-in-progress: ${{ github.event_name == 'pull_request' }}` の形）
- schedule は頻度を必要最小限にする。日次: `guardrail-regression-schedule.yml`・`clippy-nocache-schedule.yml`・`update-external.yml`、月次: `verification-gate-bench.yml`（週次では 180 分の timeout に到達し続けたため月次化）。閾値判定を持たない重い計測を高頻度で回さない
- draft PR: `ai-review.yml` は reusable workflow 呼び出しのため `if:` でスキップしない（上記）。draft の扱いを変える場合は reusable workflow 側（Fandhe-AI/actions）で対応する
- `merge_group` は使っていない（merge queue 未使用）。導入する場合は必須チェックの報告タイミングを確認してから

## 新規ジョブ・ワークフロー追加のチェックリスト

- [ ] 既存ワークフローへのジョブ追加を優先する（ワークフロー数を増やさない）。同一トリガーの小さなワークフローは統合を検討する
- [ ] 検査ロジックは `scripts/` に置き Makefile と共用する（CI 独自にロジックを書かない）。`make ci` の依存にも追加する（`makefile.md`）
- [ ] `ci-complete` の `needs` と完遂判定へ追加し、ruleset の更新が必要なことを PR 本文でユーザーに伝える
- [ ] 不要な `needs` を作らない（既存の固有ジョブは互いに独立で並列実行）。setup 時間が支配的な小ジョブは統合を検討する
- [ ] `runs-on: ubuntu-latest`・`timeout-minutes` 必須・`permissions` は最小・`persist-credentials: false`（`ci.md`）
- [ ] 第三者 action はコミット SHA で固定する。`Fandhe-AI/actions` は `@latest`（`ci.md`）
- [ ] checkout は `fetch-depth` 既定（1）。submodule は取得しない。差分が要るときだけ必要最小の深さにする
- [ ] cargo 系ジョブは registry / target キャッシュを付ける（キー設計は `ci.yml` 冒頭コメント、方針は下記「キャッシュ方針」）。重複した `cargo build` を増やさない
- [ ] lefthook と同じ検査を CI にもフックにも置かない（フックに置くのは 1 秒未満のものだけ。`git-hooks.md`）
- [ ] 追加後に `actionlint` と `bash scripts/check-workflow-runner-policy.sh check` を通す

## キャッシュ方針

GitHub Actions cache はリポジトリ単位で上限 **20GB**（本リポジトリは 2026-10-11 に 10GB から引き上げ）。超えると最終アクセスの古い順に追い出され（LRU）、7 日アクセスの無いものは削除される。Actions の予算は **$10/月（超過時停止）** で、**10GB を超えた分は 1GB あたり月 $0.07 の課金**になる。20GB は余裕分であり、常用は 10GB 以内に収める。

- **保存は main の push のみ**。target キャッシュは `actions/cache/restore` + `actions/cache/save`（`if: ... && github.ref == 'refs/heads/main' && steps.<restore>.outputs.cache-hit != 'true'`）に分離し、PR は restore のみにする。キーはソースを含まない（toolchain 実解決版数 + `Cargo.lock`）ため PR は初回 push から main のキャッシュに exact hit する。`Cargo.lock` を変える PR だけが restore-keys の前方一致（main の旧キー）からの差分ビルドになる
- **キーに SHA・run_id 等の一意値を入れない**。restore-keys は同一 toolchain 版数スコープの前方一致に限定する（イシュー #917）
- **workspace メンバーの成果物を保存しない**。cargo を実行する最後のステップの後・save の前に `scripts/ci-prune-target.sh` を置く（統合テスト・example バイナリ、メンバーの rlib/rmeta、incremental、`doc/`・`tmp/` を削除し依存 crate の成果物だけを残す。`deps/` はメンバーの `.fingerprint/<package>-<hex>` の hex で判定し、target 名では判定しない〈統合テスト `cast` と依存 crate `cast` のような同名衝突で依存側を消すため〉。保存はキー未保存時の 1 回だけで、メンバー成果物は次の run では必ず古いため価値が無い）。workflow 共通 env の `CARGO_INCREMENTAL=0`・`CARGO_PROFILE_DEV_DEBUG=line-tables-only` も維持する
- **保存内容（prune・env・プロファイル・ビルドコマンド）を変えたらキーの版セグメント（`target-<job-id>-v<N>-`）を上げる**。既存キーに exact hit すると保存がスキップされ、新しい内容が永久に保存されない。旧版への restore-keys フォールバックは付けない（中身の fingerprint が合わず、復元時間だけがかかる）。同じキーを使う他 workflow（`guardrail-regression-schedule.yml`）も同時に上げ、そちらは restore のみにする
- **復元が節約時間より長いキャッシュは持たない**。`Cache Size` と復元ステップの所要時間、ヒット時とミス時のジョブ時間を比較して判断する
- 共通側 reusable workflow（`rust-ci` の `rust-base-ci-*` キー）の設計は Fandhe-AI/actions 側を正とし、本リポジトリでは変更しない（同様の prune・`CARGO_INCREMENTAL=0` は共通側で実装済み）

計測:

```bash
# キャッシュ一覧（キー・ref・サイズ MB・作成・最終アクセス）と使用量
gh api --paginate "repos/Fandhe-AI/fandhe-ai/actions/caches?per_page=100" \
  -q '.actions_caches[]|[.key,.ref,(.size_in_bytes/1048576|floor),.created_at,.last_accessed_at]|@tsv'
gh api repos/Fandhe-AI/fandhe-ai/actions/cache/usage
# ジョブごとの復元・保存結果（exact は primary key と一致、restore-keys 経由は別キー、miss は Cache not found）
gh run view <run-id> --log --job <job-id> | grep -E "Cache (restored from key|not found|saved with key)|Cache Size|prune 完了|target: "
```

`scripts/ci-prune-target.sh` は削除前後の target サイズとプロファイル別内訳をログとジョブサマリに出す。

### 実測（2026-10-11）

変更前（直近 13 run。PR 7・main push 6）:

| キャッシュ | 保存サイズ | 状況 |
|-----------|-----------|------|
| `target-build-no-cuda-toolkit-*` | 10.5GB（9,999MiB） | 単独で上限 10GB をほぼ使い切り、保存のたびに他のキャッシュを追い出していた。PR も自ブランチに同サイズを保存。exact hit でも復元に 5〜7 分（ダウンロード約 1.5 分 + 展開）かかり、節約できるのは依存ビルド約 15 秒のみ。ジョブ時間はヒット時 16〜20 分、ミス時 16 分で、ヒットの方が遅かった |
| `target-verification-gates-*` | 1.1GB | 13 run 中 4 回 miss（追い出し） |
| `target-build-*` | 0.5GB | 毎回 exact hit |
| `target-guardrail-regression-*` | 0.13GB | 13 run 中 2 回 miss |
| `rust-base-ci-*-test-v3-*`（rust-ci） | — | 13 run 中 4 回 miss。main で保存した直後に追い出され一覧に残らない |
| registry 系・`cargo-deny-bin` 等 | 合計 0.25GB 未満 | ほぼ exact hit |

変更後（PR #3010。計測のため一時的に PR からも保存した 1 回目〈miss〉と 2 回目〈exact hit〉）:

| ジョブ | 保存サイズ（前 → 後） | prune 前 → 後（ディスク上） | ジョブ時間（前 → 後・hit 時） |
|-------|---------------------|---------------------------|-----------------------------|
| build-no-cuda-toolkit | 10.5GB → 116MB | 23.8GB → 0.5GB | 16〜20 分 → 6.6 分（miss 時 7.7 分） |
| verification-gates | 1.1GB → 143MB | 3.7GB → 0.5GB | 約 10.5 分 → 10.5 分 |
| build | 520MB → 245MB | 1.9GB → 1.1GB | 1.5〜2 分 → 1.8 分 |
| guardrail-regression | 125MB → 54MB | 0.4GB → 0.2GB | 約 20 秒 → 22 秒 |

target キャッシュ 4 件の合計は 11.8GB → 0.56GB。rust-ci（`rust-base-ci-*`）・registry 系を合わせても main のキャッシュは 1GB 未満で、PR ごとの保存が無くなるため上限 20GB はもちろん課金の発生する 10GB にも届かない。build-no-cuda-toolkit の短縮は主に debuginfo 削減（統合テストバイナリのリンク・書き込み量の減少）と、巨大キャッシュの復元・保存（各 3〜7 分）が無くなったことによる。

## 計測方法

```bash
# ワークフロー全体の所要時間（直近 30 件）
gh run list --workflow ci.yml --limit 30 --json databaseId,conclusion,createdAt,updatedAt,event
# ジョブ別の所要時間
gh run view <run-id> --json jobs -q '.jobs[]|{name,s:((.completedAt|fromdate)-(.startedAt|fromdate))}'
```

## 棚卸し結果（2026-10-11）

| ワークフロー | トリガー | 直近 30 件の所要時間（中央値 / 最大） |
|-------------|---------|----------------------------------|
| `ci.yml` | PR・push（main） | 約 17 分 / 約 21 分（最長ジョブ `cargo build / test (CUDA toolkit 非搭載検証)` 約 17〜20 分、`rust-ci / cargo test` 約 13 分、`検証ゲート` 約 11 分。ジョブ合計 約 52 分） |
| `ai-review.yml` | PR | 約 1 分 |
| `docs-site.yml` | push（site/・docs-site 変更） | 約 0.6 分 |
| `guardrail-regression-schedule.yml` | 日次 | 約 0.7 分 |
| `clippy-nocache-schedule.yml` | 日次 | 約 1.2 分 |
| `update-external.yml` | 日次 | 約 1.7 分 |
| `verification-gate-bench.yml` | 月次（旧 週次） | 約 180 分（直近 10 回中 8 回が timeout で cancelled） |
| `release.yml` / `release-all.yml` | 手動 | verify 約 1〜2 分（publish 時は承認待ちを含み最大 4 時間） |

直近 40 件の main コミットのうち 39 件が docs 系の変更（多くは Markdown）だったため、Markdown のみの push は CI を起動せず、PR は固有ジョブの cargo 系ステップを省く。
