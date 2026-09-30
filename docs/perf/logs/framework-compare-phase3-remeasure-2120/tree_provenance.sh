# shellcheck shell=bash
# tree_provenance.sh — 計測ツリーの出典（.rev-stamp ↔ git 実体）照合ヘルパー（イシュー #2120・PR #2466 レビュー）。
#
# 役割: orchestrate.sh が腕 B／C のツリー（PRE_TREE／HEAD_TREE）について、記録する出典（.rev-stamp・rev_B／rev_C）が
#   ビルド・計測する実際のツリーと一致することを計測開始前（スイッチ抽出・ビルドの前）と全 run 後に照合する。
#   .rev-stamp は手で書く値のため、古い stamp を残したツリーや変更済みのツリーを正式な腕として記録しないよう、
#   git の実体（rev-parse HEAD・status）と突き合わせ、1 つでも外れれば非 0 を返す（呼び出し側が停止する）。
# 呼び出し元: orchestrate.sh（source して verify_tree_provenance を呼ぶ）・orchestrate_selftest.sh（一時 git リポジトリで検査）。
# 照合内容（RULE.txt「計測腕」の追記と同じ。ここを変えるときは RULE.txt と self-test も同時に改める）:
#   1. ツリーは git 作業ツリーのルートである（rev-parse --show-toplevel がツリー自身。包含する別リポジトリの HEAD を借用しない）
#   2. .rev-stamp は 7〜40 桁の小文字 16 進で、git rev-parse HEAD（40 桁）がその値で始まる
#   3. 期待する完全な commit が与えられた場合（腕 B）は HEAD がそれと完全一致する
#   4. git status（未追跡ファイルを含む・submodule は除外）に変更がない。例外は計測手順が意図的に加える次の 2 つだけ:
#      - ツリールートの未追跡 .rev-stamp（手順 §5 がツリーへ書く。コミットしない）
#      - LOGD をツリー内に置く場合の LOGD 配下の未追跡ファイル（計測出力。LOGD はツリーの docs/ 配下に限る）
#      submodule（docs/spec）はビルドに使わないため除外する（--ignore-submodules=all）。
#      Cargo.lock は patch ビルドで一時的に書き換わるが、orchestrate.sh がビルドごとに復元し sha256 で突合するため、
#      照合はビルド前と復元後に行う（この関数は Cargo.lock を特別扱いしない）。
#   5. assume-unchanged／skip-worktree のファイルがない（これらは git status に変更が現れなくなるため）
# bash 3.2（macOS 既定）でも動くように mapfile・連想配列・${var,,} を使わない。

# verify_tree_provenance <label> <tree> <stamp> <expected_full|-> <logd>
#   tree・logd は pwd -P で正規化済みの絶対パスを渡す（orchestrate.sh が cd 前に正規化する）。
#   成功時は PROV_HEAD に 40 桁の HEAD を設定して 0 を返す。失敗時は理由を標準エラーへ出して 1 を返す。
verify_tree_provenance() {
  local label=$1 tree=$2 stamp=$3 expect=$4 logd=$5
  local top head rel="" tmp entry xy path bad=0 flagged
  PROV_HEAD=""
  command -v git >/dev/null 2>&1 || { echo "ERROR: 腕 ${label}: git が無い（出典を照合できないため計測しない）" >&2; return 1; }
  top="$(git -C "${tree}" rev-parse --show-toplevel 2>/dev/null)" \
    || { echo "ERROR: 腕 ${label}: ${tree} は git 作業ツリーではない（.git を含めて用意すること。転送時は clone 等で .git ごと）" >&2; return 1; }
  top="$( (CDPATH='' cd -- "${top}" 2>/dev/null && pwd -P) )" || { echo "ERROR: 腕 ${label}: 作業ツリーのルートを解決できない" >&2; return 1; }
  if [[ "${top}" != "${tree}" ]]; then
    echo "ERROR: 腕 ${label}: ${tree} は git 作業ツリーのルートではない（ルート ${top}。包含する別リポジトリの HEAD を借用しない）" >&2
    return 1
  fi
  head="$(git -C "${tree}" rev-parse --verify -q 'HEAD^{commit}' 2>/dev/null)" \
    || { echo "ERROR: 腕 ${label}: ${tree} の HEAD を解決できない" >&2; return 1; }
  [[ "${head}" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: 腕 ${label}: HEAD が 40 桁の sha ではない: ${head}" >&2; return 1; }
  [[ "${stamp}" =~ ^[0-9a-f]{7,40}$ ]] \
    || { echo "ERROR: 腕 ${label}: .rev-stamp が 7〜40 桁の小文字 16 進 sha ではない: '${stamp}'" >&2; return 1; }
  if [[ "${head}" != "${stamp}"* ]]; then
    echo "ERROR: 腕 ${label}: .rev-stamp（${stamp}）と ${tree} の git rev-parse HEAD（${head}）が一致しない（古い stamp の可能性）" >&2
    return 1
  fi
  if [[ "${expect}" != "-" && "${head}" != "${expect}" ]]; then
    echo "ERROR: 腕 ${label}: HEAD ${head} が規定の commit ${expect} と一致しない" >&2
    return 1
  fi
  # LOGD がこのツリー内なら、その配下の未追跡ファイルだけを計測出力として除外する（docs/ 配下に限る）
  if [[ "${logd}" == "${tree}" ]]; then
    echo "ERROR: 腕 ${label}: LOGD がツリーのルートそのもの（計測出力を docs/ 配下以外へ置かない）" >&2; return 1
  fi
  if [[ "${logd}" == "${tree}/"* ]]; then
    rel="${logd#"${tree}/"}/"
    if [[ "${rel}" != docs/* ]]; then
      echo "ERROR: 腕 ${label}: LOGD（${logd}）がツリー内の docs/ 配下ではない（ソース・スクリプトの隣に計測出力を置かない）" >&2
      return 1
    fi
  fi
  tmp="$(mktemp)" || return 1
  if ! git -C "${tree}" --no-optional-locks status --porcelain=v1 -z --untracked-files=all --ignore-submodules=all > "${tmp}" 2>/dev/null; then
    rm -f "${tmp}"; echo "ERROR: 腕 ${label}: git status に失敗" >&2; return 1
  fi
  while IFS= read -r -d '' entry; do
    xy="${entry:0:2}"; path="${entry:3}"
    [[ "${xy}" == "??" && "${path}" == ".rev-stamp" ]] && continue
    [[ -n "${rel}" && "${xy}" == "??" && "${path}" == "${rel}"* ]] && continue
    [[ "${bad}" -eq 0 ]] && echo "ERROR: 腕 ${label}: 作業ツリーに変更がある（計測しない。.rev-stamp と LOGD 配下の未追跡ファイル以外は不可）:" >&2
    echo "  ${entry}" >&2
    bad=1
  done < "${tmp}"
  rm -f "${tmp}"
  [[ "${bad}" -eq 0 ]] || return 1
  tmp="$(mktemp)" || return 1
  if ! git -C "${tree}" ls-files -v > "${tmp}" 2>/dev/null; then
    rm -f "${tmp}"; echo "ERROR: 腕 ${label}: git ls-files に失敗" >&2; return 1
  fi
  flagged="$(grep -E '^([a-z]|S) ' "${tmp}" | head -5)"
  rm -f "${tmp}"
  if [[ -n "${flagged}" ]]; then
    echo "ERROR: 腕 ${label}: assume-unchanged／skip-worktree のファイルがある（git status に変更が現れないため照合できない）:" >&2
    while IFS= read -r entry; do echo "  ${entry}" >&2; done <<< "${flagged}"
    return 1
  fi
  # shellcheck disable=SC2034 # 呼び出し側（orchestrate.sh・self-test）が読む出力変数
  PROV_HEAD="${head}"
  return 0
}
