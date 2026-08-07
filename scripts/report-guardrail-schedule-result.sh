#!/usr/bin/env bash
#
# guardrail 2 層検証（scripts/run-guardrail-regression.sh）の schedule 定期実行結果を
# GitHub Issue へ可視化する単一ソース（TASK-6.1b・docs/spec/05-tasks.md、イシュー #148）。
#
# 背景（.claude/rules/ci.md・.claude/rules/security.md A08/A09）:
#   schedule 実行は push/PR ゲートと違い人の目に触れないため、失敗を放置すると
#   判定器の正誤継続確認（TASK-6.1）が機能しなくなる。本スクリプトは 2 層検証の
#   合否結果を受け取り、失敗時のみ追跡用 Issue への起票・追記、復旧時のクローズを行う
#   「純粋な可視化」であり、ガードレール 3 分岐判定そのもの・除外リスト・許容誤差には
#   一切関与しない（迂回経路にはならない。security.md A08）。
#
# 呼び出し元:
#   - .github/workflows/guardrail-regression-schedule.yml の report ジョブ
#     （schedule/workflow_dispatch 実行後、常に呼ばれる。needs 経由の 2 層検証結果を
#     RESULT env で受け取る）
#   - .github/workflows/ci.yml の guardrail-regression ジョブ（self-test のみ。
#     報告ロジック自体の退行を push/PR 契機で常時検知するため。TASK-6.1a との連携）
#
# サブコマンド:
#   report      環境変数 RESULT・RUN_URL・GH_TOKEN・GH_REPO を読み、固定タイトルの
#               open Issue をタイトル完全一致で検索した上で起票／追記／クローズを行う
#               （gh CLI 経由。ネットワーク必須）
#   self-test   ネットワーク不要。(1) 不正サブコマンド・必須環境変数欠落時の fail-closed
#               終了、(2) PATH 先頭に置いた gh スタブでの 4 分岐
#               （失敗→新規起票 / 失敗→既存へ追記 / 成功→クローズ / 成功→何もしない）
#               が意図どおり呼び分けられることを検証する
set -euo pipefail

# 固定タイトル。gh issue list の --json 出力からこの文字列と完全一致するものだけを
# 対象にする（search クエリの部分一致による誤検知・別 Issue への誤爆を避けるため）。
ISSUE_TITLE='ci(guardrail): schedule 定期実行の失敗検知（TASK-6.1b）'

usage() {
  echo "usage: $0 {report|self-test}" >&2
  exit 2
}

# 固定タイトルに完全一致する open Issue の番号を 1 件だけ返す（無ければ空文字）。
# 複数一致は起票ロジックの前提が崩れているため fail-closed で異常終了する。
#
# --limit は gh CLI 既定の 30 件で打ち切られると、固定タイトルの追跡 Issue が
# ページ外に出た際に「既存 Issue なし」と誤判定してしまう（review 指摘・#148）。
# 本リポの open Issue 数は運用上 200 件を大きく超えない想定のため、十分大きい
# 固定値で明示し、既定値依存の暗黙的な打ち切りを避ける。
#
# gh 側は number/title の TSV 化のみを `--jq`（gh CLI 内蔵の Go 実装。外部 jq 非依存）
# で行い、タイトル完全一致・複数一致 fail-closed の判定は bash 側に残す
# （self-hosted runner に外部 jq が未導入でも壊れないようにするため。gh は Issue 操作で
# 既に必須のため新規ツール依存の追加にはならない。review 指摘・#148 収束指針）。
find_open_issue() {
  local tsv
  tsv=$(gh issue list --state open --limit 200 --json number,title \
        --jq '.[] | [.number, .title] | @tsv')
  local numbers=""
  local count=0
  local number title
  while IFS=$'\t' read -r number title; do
    [ -z "${number}" ] && continue
    if [ "${title}" = "${ISSUE_TITLE}" ]; then
      numbers="${number}"
      count=$((count + 1))
    fi
  done <<<"${tsv}"
  if [ "${count}" -gt 1 ]; then
    echo "NG: 固定タイトルに一致する open Issue が ${count} 件あります（想定は 0 または 1 件）" >&2
    return 1
  fi
  printf '%s' "${numbers}"
}

cmd_report() {
  : "${RESULT:?RESULT が未設定です（2 層検証ジョブの result を渡してください）}"
  : "${RUN_URL:?RUN_URL が未設定です}"
  : "${GH_TOKEN:?GH_TOKEN が未設定です}"
  : "${GH_REPO:?GH_REPO が未設定です}"

  local now
  now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  local existing
  existing=$(find_open_issue)

  # fail-closed: success 以外（failure/cancelled/skipped）はすべて失敗として扱う
  # （.claude/rules/security.md A08。中間状態を成功扱いにしない）。
  if [ "${RESULT}" != "success" ]; then
    local body_file
    body_file=$(mktemp)
    trap 'rm -f "${body_file}"' RETURN
    cat >"${body_file}" <<EOF
guardrail 2 層検証の schedule 定期実行が失敗しました（TASK-6.1b）。

- 結果: ${RESULT}
- 検知時刻（UTC）: ${now}
- 実行ログ: ${RUN_URL}

2 層検証（REQ-4 判定器単体・REQ-5 除外適用後）の継続確認が損なわれています。
scripts/run-guardrail-regression.sh の layer1/layer2 のログを確認してください。
EOF
    if [ -n "${existing}" ]; then
      gh issue comment "${existing}" --body-file "${body_file}"
      echo "NG: 既存 Issue #${existing} に失敗を追記しました"
    else
      gh issue create --title "${ISSUE_TITLE}" --body-file "${body_file}"
      echo "NG: 新規 Issue を起票しました"
    fi
    # report ジョブ自体を失敗させ、run 一覧上でも可視化する（受け入れ条件「失敗が
    # 検知可能」を GitHub Actions の run ステータスと Issue の二重で満たす）。
    return 1
  fi

  # 復旧確認: 既存 open Issue があればクローズし、無ければ何もしない
  # （success のたびに Issue を作る必要はない）。
  if [ -n "${existing}" ]; then
    local body_file
    body_file=$(mktemp)
    trap 'rm -f "${body_file}"' RETURN
    cat >"${body_file}" <<EOF
guardrail 2 層検証の schedule 定期実行が成功しました。復旧を確認したためクローズします。

- 検知時刻（UTC）: ${now}
- 実行ログ: ${RUN_URL}
EOF
    gh issue comment "${existing}" --body-file "${body_file}"
    gh issue close "${existing}"
    echo "OK: 既存 Issue #${existing} の復旧を確認しクローズしました"
  else
    echo "OK: 2 層検証は成功しており、既存の失敗 Issue もありません"
  fi
}

# gh スタブを使い、ネットワーク接続なしで report サブコマンドの分岐ロジックを検証する。
# スタブは呼び出し引数を記録し、期待どおりの gh コマンド列（issue list → create/comment/close）
# が発行されたかを assert する。
self_test_stub_dir=""
cleanup_self_test() {
  [ -n "${self_test_stub_dir}" ] && rm -rf "${self_test_stub_dir}"
}

# 引数検証・必須環境変数検証（ネットワーク不要）を確認する。
self_test_arg_validation() {
  local failed=0

  if bash "$0" no-such-subcommand >/dev/null 2>&1; then
    echo "NG: 不正サブコマンドが exit 0 になりました" >&2
    failed=1
  else
    echo "OK: 不正サブコマンドは非 0 終了します"
  fi

  if env -u RESULT -u RUN_URL -u GH_TOKEN -u GH_REPO bash "$0" report >/dev/null 2>&1; then
    echo "NG: 必須環境変数欠落時に exit 0 になりました" >&2
    failed=1
  else
    echo "OK: 必須環境変数欠落時は非 0 終了します"
  fi

  return "${failed}"
}

# gh スタブ 1 個を配置し、$1 に渡した issue_list_tsv（`gh issue list --jq '... @tsv'`
# の応答。number<TAB>title の行形式）を返させる。実 gh の `--jq` は TSV 変換までを
# 内蔵実装（外部 jq 非依存）で行うため、スタブも同じ形の出力を返せば find_open_issue
# 側のパース（bash 側でのタイトル完全一致・複数一致判定）をネットワーク不要で検証できる。
# 記録された呼び出しログは ${self_test_stub_dir}/calls.log に蓄積される。
setup_gh_stub() {
  local issue_list_tsv="$1"
  self_test_stub_dir=$(mktemp -d)
  : >"${self_test_stub_dir}/calls.log"
  cat >"${self_test_stub_dir}/gh" <<STUB
#!/usr/bin/env bash
echo "\$*" >> "${self_test_stub_dir}/calls.log"
case "\$1 \$2" in
"issue list")
  printf '%s\n' "${issue_list_tsv}"
  ;;
"issue create"|"issue comment"|"issue close")
  exit 0
  ;;
*)
  echo "unexpected gh invocation: \$*" >&2
  exit 1
  ;;
esac
STUB
  chmod +x "${self_test_stub_dir}/gh"
}

# 4 分岐（失敗→新規起票／失敗→既存へ追記／成功→クローズ／成功→何もしない）を検証する。
self_test_branches() {
  local failed=0

  # 分岐 1: 失敗 + 既存 Issue なし → 新規起票（gh issue create が呼ばれる）。
  setup_gh_stub ''
  if PATH="${self_test_stub_dir}:${PATH}" RESULT=failure RUN_URL=https://example.invalid/run/1 GH_TOKEN=dummy GH_REPO=dummy/dummy bash "$0" report >/dev/null 2>&1; then
    echo "NG: 失敗時に report が exit 0 になりました" >&2
    failed=1
  elif grep -q '^issue create' "${self_test_stub_dir}/calls.log"; then
    echo "OK: 失敗＋既存 Issue なし → issue create が呼ばれました"
  else
    echo "NG: 失敗＋既存 Issue なしで issue create が呼ばれませんでした" >&2
    failed=1
  fi
  cleanup_self_test

  # 分岐 2: 失敗 + 既存 Issue あり → 追記（gh issue comment が呼ばれ、create は呼ばれない）。
  setup_gh_stub "$(printf '42\t%s' "${ISSUE_TITLE}")"
  if PATH="${self_test_stub_dir}:${PATH}" RESULT=failure RUN_URL=https://example.invalid/run/2 GH_TOKEN=dummy GH_REPO=dummy/dummy bash "$0" report >/dev/null 2>&1; then
    echo "NG: 失敗時に report が exit 0 になりました" >&2
    failed=1
  elif grep -q '^issue comment 42' "${self_test_stub_dir}/calls.log" && ! grep -q '^issue create' "${self_test_stub_dir}/calls.log"; then
    echo "OK: 失敗＋既存 Issue あり → issue comment のみ呼ばれました（重複起票なし）"
  else
    echo "NG: 失敗＋既存 Issue ありの呼び出しが期待と異なります" >&2
    failed=1
  fi
  cleanup_self_test

  # 分岐 3: 成功 + 既存 Issue あり → クローズ（comment → close の順で呼ばれる）。
  setup_gh_stub "$(printf '43\t%s' "${ISSUE_TITLE}")"
  if PATH="${self_test_stub_dir}:${PATH}" RESULT=success RUN_URL=https://example.invalid/run/3 GH_TOKEN=dummy GH_REPO=dummy/dummy bash "$0" report >/dev/null 2>&1; then
    if grep -q '^issue close 43' "${self_test_stub_dir}/calls.log"; then
      echo "OK: 成功＋既存 Issue あり → issue close が呼ばれました"
    else
      echo "NG: 成功＋既存 Issue ありで issue close が呼ばれませんでした" >&2
      failed=1
    fi
  else
    echo "NG: 成功時に report が非 0 終了しました" >&2
    failed=1
  fi
  cleanup_self_test

  # 分岐 4: 成功 + 既存 Issue なし → 何もしない（create/comment/close いずれも呼ばれない）。
  setup_gh_stub ''
  if PATH="${self_test_stub_dir}:${PATH}" RESULT=success RUN_URL=https://example.invalid/run/4 GH_TOKEN=dummy GH_REPO=dummy/dummy bash "$0" report >/dev/null 2>&1; then
    if grep -qE '^issue (create|comment|close)' "${self_test_stub_dir}/calls.log"; then
      echo "NG: 成功＋既存 Issue なしで不要な issue 操作が呼ばれました" >&2
      failed=1
    else
      echo "OK: 成功＋既存 Issue なし → 何もしません"
    fi
  else
    echo "NG: 成功時に report が非 0 終了しました" >&2
    failed=1
  fi
  cleanup_self_test

  return "${failed}"
}

# 分岐 5（fail-closed）: 固定タイトルに一致する open Issue が 2 件以上ある異常状態を
# 検証する。find_open_issue 内部で異常終了するため report は必ず非 0 で終了し、
# かつ issue create/comment/close のいずれも呼ばれてはならない（起票ロジックの
# 前提が崩れた状態で誤った Issue 操作を実行しないことの保証。review 指摘・#148）。
self_test_duplicate_issue_fail_closed() {
  local failed=0

  setup_gh_stub "$(printf '50\t%s\n51\t%s' "${ISSUE_TITLE}" "${ISSUE_TITLE}")"
  if PATH="${self_test_stub_dir}:${PATH}" RESULT=failure RUN_URL=https://example.invalid/run/5 GH_TOKEN=dummy GH_REPO=dummy/dummy bash "$0" report >/dev/null 2>&1; then
    echo "NG: 固定タイトルの open Issue が複数存在するのに report が exit 0 になりました" >&2
    failed=1
  elif grep -qE '^issue (create|comment|close)' "${self_test_stub_dir}/calls.log"; then
    echo "NG: 複数一致の fail-closed 時に issue 操作が呼ばれました" >&2
    failed=1
  else
    echo "OK: 固定タイトルの open Issue が複数存在する場合は fail-closed で異常終了し、issue 操作は呼ばれません"
  fi
  cleanup_self_test

  return "${failed}"
}

cmd_self_test() {
  local failed=0
  self_test_arg_validation || failed=1
  self_test_branches || failed=1
  self_test_duplicate_issue_fail_closed || failed=1

  if [ "${failed}" -ne 0 ]; then
    echo "NG: self-test に失敗しました" >&2
    return 1
  fi
  echo "OK: self-test すべて成功"
}

case "${1:-}" in
report)
  cmd_report
  ;;
self-test)
  cmd_self_test
  ;;
*)
  usage
  ;;
esac
