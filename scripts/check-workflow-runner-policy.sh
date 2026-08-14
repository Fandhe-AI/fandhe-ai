#!/usr/bin/env bash
#
# self-hosted runner の逆戻り防止を fail-closed で検査する単一ソース（イシュー #472）。
# #457 Phase 2（#465〜#469）で .github/workflows/ の全ジョブが runs-on: ubuntu-latest へ
# 移行完了した（.claude/rules/ci.md「runner（GitHub ホステッド既定）」節）が、以後
# self-hosted を再導入しても機械的に検知する仕組みがなかった。fandhe-frontend の
# workflow_runner_policy.rs（runner 宣言を機械列挙し唯一の許容形と突合する fail-closed
# 検査）の設計思想を、本リポの既存契約検査パターン（scripts/check-forbidden-deps.sh:
# 単一ソースのシェルスクリプト + scripts/testdata/ fixture + self-test サブコマンド +
# ci.yml ジョブと Makefile の共用）へ移植したもの。
#
# 呼び出し元:
#   - .github/workflows/ci.yml の runner-policy ジョブ（self-test → check の順で呼ぶ）
#   - Makefile の runner-policy ターゲット（CI と同一判定をローカル再現）
#
# サブコマンド:
#   check      .github/workflows/ 配下の *.yml・*.yaml を対象に runner 宣言を検査する
#   self-test  scripts/testdata/ の固定 fixture（clean/forbidden/unknown）に対し
#              ポジティブ・ネガティブ判定を行い、本スクリプトの検査ロジック自体の
#              退行を検出する（受け入れ条件「self-hosted 再導入で CI が fail する」の
#              機械検証）
#
# 検査方針（fail-closed。許可リスト方式ではなく「唯一の許容形に一致しなければ違反」）:
#   1. 禁止トークン検査: 各行のコメント（# 以降）を除去した後 `self-hosted` の出現を
#      違反とする
#   2. 唯一の許容形検査: コメント除去後の `runs-on:`／`runner-label:`／
#      `post-feedback-runner-label:`（引用キー `"runs-on":`／`'runs-on':`・コロン前
#      空白 `runs-on :` 等の有効な YAML 表記の等価系を含む。イシュー #472 P1
#      codex-review 指摘）の宣言行を全て列挙し、値が `ubuntu-latest` のスカラー
#      完全一致でなければ違反とする（larger runner 名・matrix/式展開
#      `${{ ... }}`・ブロックシーケンス形式の値なし行は「未知の形」として違反側に倒す。
#      パースの取りこぼしで fail-open にならない設計）
#   3. 未知のキー表記検査（イシュー #472 P0 codex-review 指摘）: 正規表現による
#      YAML 構文の完全な再実装（エスケープ・複合キー表記の意味解析）は避け、代わりに
#      「本検査が安全に解釈できないキー表記が存在する」こと自体を検出して無条件に
#      違反とする（許容形かどうかを判定できない＝違反、という fail-closed 側への
#      倒し方。値を読み取れないので `ubuntu-latest` 判定を試みない）:
#        a. バックスラッシュを含む二重引用キー（例: `"runs\x2Don": ...`）。YAML の
#           double-quoted scalar は `\xHH`／`\uHHHH` 等のエスケープをデコードするため、
#           キーの字面が `runs-on` 等と一致しなくても GitHub Actions からは一致した
#           キーとして解釈されうる。エスケープの意味解析（デコード）はせず、
#           「エスケープを含む引用キーが 1 つでも存在する」こと自体を通常の
#           workflow では起こり得ない異常な表記として違反にする
#        b. YAML の明示キー形式（`? key` インジケータ）。`? runs-on` \n
#           `: ubuntu-latest-8-cores` のように、キーと値が別行かつキー行に `:` が
#           現れない複合マッピングは唯一の許容形検査（行内 `key:` 一致）を素通りする。
#           `?` インジケータ自体が通常の workflow ファイルでは使われないため、
#           出現した時点で無条件に違反とする
#
# 例外の扱い（意図的に検査対象外となる構成。.claude/rules/ci.md「codex-review」節）:
#   codex-review wrapper の codex 実行ジョブは runner-label を非指定とし reusable
#   workflow の既定値 `codex`（codex-home 方式の認証情報を持つ self-hosted 専用 runner）
#   に委譲する設計のため、本リポ側ファイルに runner 宣言が現れず本検査の対象外と
#   なる。将来 `codex` 等のラベルを明示する場合は、それは本スクリプトの許容形の
#   意図的な拡張（レビュー必須）を要する。実機（CUDA/Metal）ジョブの将来的な例外追加
#   （ci.md「実機依存」節）も同様に、その時点での許容形拡張として扱う。
#
# fail-closed の追加防御:
#   .github/workflows/ が存在しない・対象ファイルが 0 件の場合も失敗とする
#   （ディレクトリ改名等で検査が空振りしても green にならない）。
set -euo pipefail

ALLOWED_RUNNER_VALUE='ubuntu-latest'
# runner 宣言のキー名（コロン込み・行頭の空白は許容するため正規表現側で吸収する）。
# YAML の mapping key は素の識別子以外に引用キー（`"runs-on":` / `'runs-on':`）や
# コロン前の空白（`runs-on :`）でも同じキーとして解釈される（イシュー #472 P1
# codex-review 指摘）。これらは文字列としては非引用・コロン直後のパターンに一致
# しないため、素朴な `key:` 一致だけでは fail-closed の前提（唯一の許容形以外は
# 違反）を素通りしてしまう。キー名の前後に任意 1 文字の引用符（`"`／`'`）・
# コロン前の空白を許容するよう拡張し、有効な YAML 表記の等価系をすべて宣言として
# 検出する（本物の YAML パーサーを新規依存として導入しないため、正規表現側の
# 網羅で fail-closed を維持する。deps-policy.md の許容依存 8 区分に YAML パーサーは
# 含まれない）。
RUNNER_KEY_PATTERN='["'\''"]?(runs-on|runner-label|post-feedback-runner-label)["'\''"]? *:'

# 未知のキー表記検査（イシュー #472 P0 codex-review 指摘）用パターン。
# 「唯一の許容形検査」が読み取れる字面の宣言行を前提としているのに対し、この 2 つは
# 本検査がそもそも安全に意味解析できない YAML 表記であることそのものを検出する
# （検出できたら直ちに違反。デコードや意味解釈は行わない。上記コメント方針 3 参照）。
#
# a. バックスラッシュエスケープを含む二重引用キー: `"` で始まり `\` を含み `"` で
#    閉じたのち任意個の空白を挟んで `:` が続く行。単純な文字列内バックスラッシュ
#    （例: Windows パスを値として書く場合）は値側であって mapping key ではないため
#    本パターン（`:` が直後に続くもの）には一致しない。
ESCAPED_QUOTED_KEY_PATTERN='"[^"]*\\[^"]*" *:'
# b. YAML 明示キー形式のインジケータ（`? key` の `?`）。行頭（前後の空白を許容）に
#    `?` があり、その直後が空白または行末であるもの。フロー内 `?`（`{? a: b}` 等）は
#    本リポの workflow ファイルでは使用実績がなく、fail-closed 側に倒すことを優先し
#    区別しない。
EXPLICIT_KEY_INDICATOR_PATTERN='^[[:space:]]*\?([[:space:]]|$)'

usage() {
  echo "usage: $0 {check|self-test}" >&2
  exit 2
}

# 1 ファイル分のテキストに対する検査本体。check（実ファイル）と self_test（固定
# fixture）の双方から呼ぶ共通経路とすることで、self-test がパターン退行を確実に
# 検出できるようにする（check-forbidden-deps.sh の check_tree_text と同型の設計）。
check_workflow_text() {
  local label="$1"
  local text="$2"
  # `#` 以降のコメントを除去したテキストに対して検査する（コメント内の self-hosted
  # 言及・説明文を誤検出しない）。行ごとにシェルのパラメータ展開（`${line%%#*}`）で
  # 除去する（sed 起動なしで完結させ、SC2001 のスタイル指摘も回避する）。
  local stripped=""
  local line
  while IFS= read -r line; do
    stripped+="${line%%#*}"$'\n'
  done <<<"${text}"
  local failed=0

  # 1. 禁止トークン検査。
  if echo "${stripped}" | grep -qw 'self-hosted'; then
    echo "::error::${label} に self-hosted runner の宣言が含まれています（.claude/rules/ci.md「runner」節。GitHub ホステッド既定への移行〈#457 Phase 2〉に反する）:" >&2
    echo "${stripped}" | grep -nw 'self-hosted' >&2
    failed=1
  fi

  # 2. 唯一の許容形検査。runner 宣言行を全て抽出し、値が ubuntu-latest の
  # スカラー完全一致でない行を違反とする。
  #    - 値なし行（ブロックシーケンス形式 `runs-on:` のみで次行以降に `- xxx`）は
  #      キーの直後に値が続かないため、この抽出パターンでは「値が空/一致しない」
  #      側に自然に倒れ違反として検出される
  #    - 式展開（`${{ ... }}`）・larger runner 名（`ubuntu-latest-8-cores` 等）は
  #      いずれも ubuntu-latest との完全一致にならないため違反として検出される
  local declarations
  declarations=$(echo "${stripped}" | grep -E "${RUNNER_KEY_PATTERN}" || true)
  if [ -n "${declarations}" ]; then
    local bad_lines
    bad_lines=$(echo "${declarations}" | grep -vE ": *${ALLOWED_RUNNER_VALUE} *\$" || true)
    if [ -n "${bad_lines}" ]; then
      echo "::error::${label} に唯一の許容形（runner-label/runs-on = ${ALLOWED_RUNNER_VALUE} 完全一致）以外の runner 宣言が含まれています:" >&2
      echo "${bad_lines}" >&2
      failed=1
    fi
  fi

  # 3. 未知のキー表記検査。上記コメント方針 3 参照。値が何であるかに関わらず、
  #    本検査が安全に意味解析できないキー表記が 1 行でも存在すれば違反とする
  #    （デコードして許容形か判定する、という経路を作らない）。
  local escaped_key_lines
  escaped_key_lines=$(echo "${stripped}" | grep -E "${ESCAPED_QUOTED_KEY_PATTERN}" || true)
  if [ -n "${escaped_key_lines}" ]; then
    echo "::error::${label} にエスケープを含む二重引用キーが見つかりました（YAML の \\xHH／\\uHHHH 等のエスケープはキー名を変えうるため、本検査では意味解析せず無条件に違反とします。.claude/rules/ci.md「runner」節）:" >&2
    echo "${escaped_key_lines}" >&2
    failed=1
  fi

  local explicit_key_lines
  explicit_key_lines=$(echo "${stripped}" | grep -nE "${EXPLICIT_KEY_INDICATOR_PATTERN}" || true)
  if [ -n "${explicit_key_lines}" ]; then
    echo "::error::${label} に YAML 明示キー形式（'?' インジケータ）が見つかりました（キーと値が別行の複合マッピングは唯一の許容形検査を素通りしうるため、出現した時点で無条件に違反とします）:" >&2
    echo "${explicit_key_lines}" >&2
    failed=1
  fi

  if [ "${failed}" -ne 0 ]; then
    return 1
  fi
  echo "OK: ${label} は runner 契約（${ALLOWED_RUNNER_VALUE} 完全一致・self-hosted 不在・既知のキー表記のみ）に適合"
}

check() {
  local workflow_dir=".github/workflows"
  if [ ! -d "${workflow_dir}" ]; then
    echo "::error::${workflow_dir} が見つかりません（fail-closed: 検査対象ディレクトリの消失を違反として扱う）" >&2
    return 1
  fi

  local files=()
  local f
  while IFS= read -r -d '' f; do
    files+=("${f}")
  done < <(find "${workflow_dir}" -maxdepth 1 -type f \( -name '*.yml' -o -name '*.yaml' \) -print0 | sort -z)

  if [ "${#files[@]}" -eq 0 ]; then
    echo "::error::${workflow_dir} に *.yml/*.yaml が見つかりません（fail-closed: 対象 0 件を違反として扱う）" >&2
    return 1
  fi

  local failed=0
  for f in "${files[@]}"; do
    if ! check_workflow_text "${f}" "$(cat "${f}")"; then
      failed=1
    fi
  done

  if [ "${failed}" -ne 0 ]; then
    return 1
  fi
  echo "OK: ${#files[@]} 件の workflow ファイルすべてが runner 契約に適合"
}

self_test() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local clean="${script_dir}/testdata/workflow-runner-clean.yml"
  local forbidden="${script_dir}/testdata/workflow-runner-forbidden.yml"
  local unknown="${script_dir}/testdata/workflow-runner-unknown.yml"
  local failed=0

  for f in "${clean}" "${forbidden}" "${unknown}"; do
    if [ ! -f "${f}" ]; then
      echo "NG: self-test fixture が見つかりません（${f}）" >&2
      return 1
    fi
  done

  # ポジティブ: ubuntu-latest のみ・コメント内 self-hosted を含む clean fixture は
  # pass すること（コメント誤検出がないことの検証を兼ねる）。
  if check_workflow_text "self-test clean fixture" "$(cat "${clean}")" >/dev/null; then
    echo "self-test OK: clean fixture は pass する"
  else
    echo "self-test NG: clean fixture が誤って fail した（検査ロジックが誤検出している）" >&2
    failed=1
  fi

  # ネガティブ: 非コメント行に self-hosted を含む forbidden fixture は fail すること
  # （受け入れ条件「self-hosted 再導入で CI が fail する」の機械検証）。
  if check_workflow_text "self-test forbidden fixture" "$(cat "${forbidden}")" >/dev/null 2>&1; then
    echo "self-test NG: forbidden fixture が誤って pass した（検査ロジックが退行している）" >&2
    failed=1
  else
    echo "self-test OK: forbidden fixture は fail する"
  fi

  # ネガティブ: 許容形以外の runner 宣言（larger runner 名・ブロックシーケンス形式等）
  # を含む unknown fixture は fail すること（fail-open 側への取りこぼしがないことの検証）。
  if check_workflow_text "self-test unknown fixture" "$(cat "${unknown}")" >/dev/null 2>&1; then
    echo "self-test NG: unknown fixture が誤って pass した（検査ロジックが退行している）" >&2
    failed=1
  else
    echo "self-test OK: unknown fixture は fail する"
  fi

  if [ "${failed}" -ne 0 ]; then
    return 1
  fi
  echo "OK: self-test すべて pass"
}

main() {
  local subcommand="${1:-}"
  case "${subcommand}" in
    check)
      check
      ;;
    self-test)
      self_test
      ;;
    *)
      usage
      ;;
  esac
}

main "$@"
