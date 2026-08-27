#!/usr/bin/env bash
#
# キャッシュなしフルビルド clippy（scripts/report-clippy-nocache-schedule-result.sh を
# 呼び出す .github/workflows/clippy-nocache-schedule.yml）の schedule 定期実行結果を
# GitHub Issue へ可視化する単一ソース（イシュー #918）。
#
# 背景（.claude/rules/ci.md・.claude/rules/security.md A08/A09）:
#   main の push CI（rust-base-ci reusable workflow の clippy ジョブ）はキャッシュ命中時に
#   検査対象が実質的に狭まり、toolchain 更新（clippy 1.98 の新 lint 等）起因の違反が
#   main 上で潜伏したまま検出されない「キャッシュ偽陰性」が起こりうる（#912 の対応漏れが
#   キャッシュミスした PR ブランチ #913/#914 で初めて顕在化した実績）。本スクリプトは
#   キャッシュなしフルビルド clippy の schedule 実行結果を受け取り、失敗時のみ追跡用
#   Issue への起票・追記、復旧時のクローズを行う「純粋な可視化」であり、clippy の
#   検査コマンド・lint 設定そのものには一切関与しない（迂回経路にはならない。
#   security.md A08）。
#
#   scripts/report-guardrail-schedule-result.sh（TASK-6.1b・イシュー #148）と同型の
#   専用スクリプトとして新設した。既存スクリプトは固定タイトル・本文が guardrail 固有で
#   self-test も密結合のため、汎用化改修は guardrail 側 workflow への変更混入・迂回
#   リスクがあり避けた（out-of-scope-tracking.md。共通化リファクタは別イシューの検討
#   対象として PR で言及するに留める）。
#
# 呼び出し元:
#   - .github/workflows/clippy-nocache-schedule.yml の report ジョブ
#     （schedule/workflow_dispatch 実行後、常に呼ばれる。needs 経由の clippy ジョブ結果を
#     RESULT env で受け取る）
#
# サブコマンド:
#   report      環境変数 RESULT・RUN_URL・GH_TOKEN・GH_REPO を読み、固定タイトルの
#               open Issue をタイトル完全一致で検索した上で起票／追記／クローズを行う
#               （gh CLI 経由。ネットワーク必須）
#   self-test   ネットワーク不要。(1) 不正サブコマンド・必須環境変数欠落時の fail-closed
#               終了、(2) PATH 先頭に置いた gh スタブでの分岐
#               （失敗→新規起票 / 失敗→既存へ追記 / 成功→クローズ / 成功→何もしない /
#               複数一致 fail-closed）が意図どおり呼び分けられることを検証する
set -euo pipefail

# 固定タイトル。gh issue list の --json 出力からこの文字列と完全一致するものだけを
# 対象にする（search クエリの部分一致による誤検知・別 Issue への誤爆を避けるため）。
ISSUE_TITLE='ci(rust): キャッシュなしフルビルド clippy 定期検証の失敗検知（#918）'

usage() {
  echo "usage: $0 {report|self-test}" >&2
  exit 2
}

# 固定タイトルに完全一致する open Issue の番号を 1 件だけ返す（無ければ空文字）。
# 複数一致は起票ロジックの前提が崩れているため fail-closed で異常終了する。
#
# --limit は gh CLI 既定の 30 件で打ち切られると、固定タイトルの追跡 Issue が
# ページ外に出た際に「既存 Issue なし」と誤判定してしまう（report-guardrail-schedule-
# result.sh と同一方針）。gh issue list --limit は指定件数まで自動でページングして
# 取得する（gh CLI 実装）ため、既定値依存の暗黙的な打ち切りを避ける目的で、本リポの
# 想定運用規模を大きく超える 1000 件を明示指定し「先頭 N 件のみ検査して見落とす」
# リスクを実質的に排除する（コードレビュー指摘 #940。以前の 200 件固定では想定超過時に
# 既存 Issue を見落とし重複起票しうる余地があったための引き上げ）。
#
# gh 側は number/title の TSV 化のみを `--jq`（gh CLI 内蔵の Go 実装。外部 jq 非依存）
# で行い、タイトル完全一致・複数一致 fail-closed の判定は bash 側に残す
# （runner に外部 jq が未導入でも壊れないようにするため）。
find_open_issue() {
  local tsv
  tsv=$(gh issue list --state open --limit 1000 --json number,title \
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
  : "${RESULT:?RESULT が未設定です（clippy ジョブの result を渡してください）}"
  : "${RUN_URL:?RUN_URL が未設定です}"
  : "${GH_TOKEN:?GH_TOKEN が未設定です}"
  : "${GH_REPO:?GH_REPO が未設定です}"
  : "${EVENT_NAME:?EVENT_NAME が未設定です（github.event_name を渡してください）}"

  local now
  now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  # Issue の起票・追記・クローズ（状態変更）は schedule 実行に限定する（コードレビュー
  # 指摘 #940）。workflow_dispatch は任意のブランチ／コミットで手動実行できるため、
  # 状態変更を許すと検証用ブランチでの手動実行の失敗が main 用の固定タイトル障害
  # Issue を誤って起票・追記したり、成功が未復旧の main の Issue を誤ってクローズ
  # したりしうる（定期検証結果の fail-closed な可視化という運用契約を壊す）。
  # schedule トリガーは GitHub Actions の仕様上 default branch 上の workflow 定義から
  # しか発火しない（本スクリプト冒頭コメント・.github/workflows/clippy-nocache-
  # schedule.yml 冒頭コメント）ため、EVENT_NAME == schedule の判定のみでブランチ限定
  # の意図を満たす。workflow_dispatch は run 自体の exit code（サマリ出力のみ）で
  # 可視化し、Issue 側は一切変更しない。
  if [ "${EVENT_NAME}" != "schedule" ]; then
    echo "NOTE: EVENT_NAME=${EVENT_NAME} のため Issue の起票・追記・クローズは行いません（schedule 実行専用。結果: ${RESULT}）"
    if [ "${RESULT}" != "success" ]; then
      echo "NG: キャッシュなしフルビルド clippy が失敗しました（手動実行のため Issue 可視化はスキップ）: ${RUN_URL}" >&2
      return 1
    fi
    echo "OK: キャッシュなしフルビルド clippy は成功しました（手動実行のため Issue 可視化はスキップ）"
    return 0
  fi

  local existing
  existing=$(find_open_issue)

  # fail-closed: success 以外（failure/cancelled/skipped）はすべて失敗として扱う
  # （.claude/rules/security.md A08。中間状態を成功扱いにしない）。
  if [ "${RESULT}" != "success" ]; then
    local body_file
    body_file=$(mktemp)
    trap 'rm -f "${body_file}"' RETURN
    cat >"${body_file}" <<EOF
キャッシュなしフルビルド clippy の schedule 定期検証が失敗しました（#918）。

- 結果: ${RESULT}
- 検知時刻（UTC）: ${now}
- 実行ログ: ${RUN_URL}

main の push CI（rust-base-ci の clippy ジョブ）ではキャッシュ命中により
検査対象が狭まっている可能性があります（キャッシュ偽陰性）。ローカルまたは
キャッシュなし環境で以下を再現して確認してください。

    cargo clippy --workspace --all-targets --all-features -- -D warnings
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
キャッシュなしフルビルド clippy の schedule 定期検証が成功しました。復旧を確認したためクローズします。

- 検知時刻（UTC）: ${now}
- 実行ログ: ${RUN_URL}
EOF
    gh issue comment "${existing}" --body-file "${body_file}"
    gh issue close "${existing}"
    echo "OK: 既存 Issue #${existing} の復旧を確認しクローズしました"
  else
    echo "OK: キャッシュなしフルビルド clippy は成功しており、既存の失敗 Issue もありません"
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

  if env -u RESULT -u RUN_URL -u GH_TOKEN -u GH_REPO -u EVENT_NAME bash "$0" report >/dev/null 2>&1; then
    echo "NG: 必須環境変数欠落時に exit 0 になりました" >&2
    failed=1
  else
    echo "OK: 必須環境変数欠落時は非 0 終了します"
  fi

  if env -u EVENT_NAME RESULT=success RUN_URL=https://example.invalid/run/0 GH_TOKEN=dummy GH_REPO=dummy/dummy bash "$0" report >/dev/null 2>&1; then
    echo "NG: EVENT_NAME 欠落時に exit 0 になりました" >&2
    failed=1
  else
    echo "OK: EVENT_NAME 欠落時は非 0 終了します"
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
  if PATH="${self_test_stub_dir}:${PATH}" EVENT_NAME=schedule RESULT=failure RUN_URL=https://example.invalid/run/1 GH_TOKEN=dummy GH_REPO=dummy/dummy bash "$0" report >/dev/null 2>&1; then
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
  if PATH="${self_test_stub_dir}:${PATH}" EVENT_NAME=schedule RESULT=failure RUN_URL=https://example.invalid/run/2 GH_TOKEN=dummy GH_REPO=dummy/dummy bash "$0" report >/dev/null 2>&1; then
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
  if PATH="${self_test_stub_dir}:${PATH}" EVENT_NAME=schedule RESULT=success RUN_URL=https://example.invalid/run/3 GH_TOKEN=dummy GH_REPO=dummy/dummy bash "$0" report >/dev/null 2>&1; then
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
  if PATH="${self_test_stub_dir}:${PATH}" EVENT_NAME=schedule RESULT=success RUN_URL=https://example.invalid/run/4 GH_TOKEN=dummy GH_REPO=dummy/dummy bash "$0" report >/dev/null 2>&1; then
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
# 前提が崩れた状態で誤った Issue 操作を実行しないことの保証）。
self_test_duplicate_issue_fail_closed() {
  local failed=0

  setup_gh_stub "$(printf '50\t%s\n51\t%s' "${ISSUE_TITLE}" "${ISSUE_TITLE}")"
  if PATH="${self_test_stub_dir}:${PATH}" EVENT_NAME=schedule RESULT=failure RUN_URL=https://example.invalid/run/5 GH_TOKEN=dummy GH_REPO=dummy/dummy bash "$0" report >/dev/null 2>&1; then
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

# EVENT_NAME != schedule（workflow_dispatch 等）では、失敗・成功いずれの RESULT でも
# gh issue list/create/comment/close が一切呼ばれない（find_open_issue 自体を呼ばない
# ため gh コマンドが 1 度も起動されない）こと、かつ RESULT に応じた exit code
# （失敗→非 0／成功→0）だけは維持されることを検証する（コードレビュー指摘 #940 の
# P1: 手動実行が main の定期検証 Issue を誤更新することを防ぐ本体）。
self_test_workflow_dispatch_no_issue_mutation() {
  local failed=0

  # 手動実行 + 失敗 → Issue 操作なし・run 自体は非 0 終了（可視化は run ステータスのみ）。
  setup_gh_stub ''
  if PATH="${self_test_stub_dir}:${PATH}" EVENT_NAME=workflow_dispatch RESULT=failure RUN_URL=https://example.invalid/run/6 GH_TOKEN=dummy GH_REPO=dummy/dummy bash "$0" report >/dev/null 2>&1; then
    echo "NG: workflow_dispatch + 失敗で report が exit 0 になりました" >&2
    failed=1
  elif [ -s "${self_test_stub_dir}/calls.log" ]; then
    echo "NG: workflow_dispatch 実行で gh コマンドが呼ばれました（Issue 状態変更の可能性）" >&2
    failed=1
  else
    echo "OK: workflow_dispatch + 失敗 → gh は一切呼ばれず、report は非 0 終了します"
  fi
  cleanup_self_test

  # 手動実行 + 成功 → Issue 操作なし・run 自体は 0 終了。
  setup_gh_stub ''
  if PATH="${self_test_stub_dir}:${PATH}" EVENT_NAME=workflow_dispatch RESULT=success RUN_URL=https://example.invalid/run/7 GH_TOKEN=dummy GH_REPO=dummy/dummy bash "$0" report >/dev/null 2>&1; then
    if [ -s "${self_test_stub_dir}/calls.log" ]; then
      echo "NG: workflow_dispatch + 成功で gh コマンドが呼ばれました（Issue 状態変更の可能性）" >&2
      failed=1
    else
      echo "OK: workflow_dispatch + 成功 → gh は一切呼ばれず、report は 0 終了します"
    fi
  else
    echo "NG: workflow_dispatch + 成功で report が非 0 終了しました" >&2
    failed=1
  fi
  cleanup_self_test

  return "${failed}"
}

cmd_self_test() {
  local failed=0
  self_test_arg_validation || failed=1
  self_test_branches || failed=1
  self_test_duplicate_issue_fail_closed || failed=1
  self_test_workflow_dispatch_no_issue_mutation || failed=1

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
