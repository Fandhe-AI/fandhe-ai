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
- [ ] cargo 系ジョブは registry / target キャッシュを付ける（キー設計は `ci.yml` 冒頭コメント）。重複した `cargo build` を増やさない
- [ ] lefthook と同じ検査を CI にもフックにも置かない（フックに置くのは 1 秒未満のものだけ。`git-hooks.md`）
- [ ] 追加後に `actionlint` と `bash scripts/check-workflow-runner-policy.sh check` を通す

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
