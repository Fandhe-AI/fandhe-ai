#!/usr/bin/env bash
#
# 依存禁止リスト（burn 系一式・cubecl・candle・tch・ndarray。.claude/rules/deps-policy.md）の
# 混入を fail-closed で検査する単一ソース（TASK-1.2・docs/spec/05-tasks.md）。
#
# 呼び出し元:
#   - .github/workflows/ci.yml の deps-forbidden ジョブ（self-test → lock-all → tree の順で呼ぶ）
#   - Makefile の deps-forbidden ターゲット（CI と同一判定をローカル再現）
#
# サブコマンド:
#   lock <path>  Cargo.lock の `name = "<crate>"` 行を検査する（Cargo.toml 未追加時は
#                呼び出し側で存在チェックしてからこのスクリプトを呼ぶ想定）
#   lock-all     本リポジトリが持つ検査対象の全 Cargo.lock（本体 workspace ルート・
#                scripts/bench/oss-gemm-compare/ の OSS 直接比較ハーネス〈許容依存
#                第 9 区分。.claude/rules/deps-policy.md〉）をまとめて検査する。
#                scripts/bench/framework-compare/（第 9 区分の適用範囲拡張。
#                フレームワーク横並びベンチ）の Cargo.lock は、比較対象という性質上
#                依存禁止リストのクレート（candle-*・burn-*・cubecl・ndarray・tch 等）を
#                意図的に含むため、lock-all の走査対象に**含めない**（意図的除外。
#                deps-policy.md 第 9 区分の該当行参照。本体 workspace への混入は
#                引き続きルート Cargo.lock・cargo tree 検査で fail-closed に検出される）。
#                対象パスの列挙をこの 1 箇所に集約し、呼び出し側（ci.yml・Makefile）で
#                個別パスをハードコードしない（「CI と同一判定をローカル再現」を
#                二重管理なしで満たすため）。ルート Cargo.lock は workspace 骨格
#                構築前の不在を許容し notice でスキップするが、ハーネスの Cargo.lock は
#                第 9 区分有効化後は常時存在すべきものとして fail-closed（不在はエラー）
#                とする
#   tree         `cargo tree` 出力を検査する（cargo 必須。呼び出し側で Cargo.toml の
#                有無を判定してから呼ぶ想定。--target all で cfg(target_os = "macos")
#                限定の Metal 系依存も検査範囲に含める）
#   self-test    scripts/testdata/ の固定 fixture に対しネガティブ・ポジティブ判定を行い、
#                本スクリプトの検査ロジック自体の退行（パターン破損等）を検出する
#                （受け入れ条件「禁止クレート混入時に fail-closed で失敗する」の機械検証）。
#                lock-all は check_lock を再利用する薄いラッパーのため、検査ロジック
#                自体は本 self-test の対象で足り、専用 fixture は追加しない
#
# 禁止クレート名の候補はここ 1 箇所だけに定義し、lock / tree / self-test の全パターンを
# 導出する（計画どおり「正規表現はスクリプト内 1 箇所に定義し共用する」を満たす）。
# candle は `candle-nn`・`candle-transformers` 等の同系列クレートも検出対象に含める
# （deps-policy.md の禁止対象「candle」の安全側解釈。検出範囲拡張は fail-closed の強化であり
# ガードレール緩和ではないため承認不要）。`[a-z0-9-]+` は数字を含むサフィックス
# （例: burn2 のような将来の命名）も取りこぼさないための表記。
set -euo pipefail

FORBIDDEN_CRATES_ALT='burn|burn-[a-z0-9-]+|cubecl|cubecl-[a-z0-9-]+|candle|candle-[a-z0-9-]+|tch|ndarray'

# Cargo.lock の `name = "<crate>"` 行に対する完全一致パターン。
FORBIDDEN_LOCK_PATTERN="^name = \"(${FORBIDDEN_CRATES_ALT})\"\$"

# `cargo tree --prefix none` 出力（`<crate> v<version>` 形式）に対する行頭一致パターン。
FORBIDDEN_TREE_PATTERN="^(${FORBIDDEN_CRATES_ALT}) v"

usage() {
  echo "usage: $0 {lock <Cargo.lock のパス>|lock-all|tree|self-test}" >&2
  exit 2
}

check_lock() {
  local lock_path="$1"
  if [ ! -f "${lock_path}" ]; then
    echo "NG: ${lock_path} が見つかりません" >&2
    return 1
  fi
  if grep -qE "${FORBIDDEN_LOCK_PATTERN}" "${lock_path}"; then
    echo "::error::依存禁止リストのクレートが ${lock_path} に含まれています（.claude/rules/deps-policy.md）:" >&2
    grep -E "${FORBIDDEN_LOCK_PATTERN}" "${lock_path}" >&2
    return 1
  fi
  echo "OK: ${lock_path} に依存禁止リストの混入なし"
}

# 本リポジトリが持つ全 Cargo.lock を一括検査する（lock-all サブコマンド本体）。
# 対象パスの列挙をこの関数 1 箇所に集約する（呼び出し元コメント参照）。
# check_lock（既存ロジック）をそのまま再利用し、新しい正規表現・grep 経路は
# 追加しない。
check_lock_all() {
  local failed=0

  # 本体 workspace のルート Cargo.lock。workspace 骨格構築前（TASK-1.1 未着手時）は
  # 不在を許容し notice でスキップする（deps-forbidden ジョブの既存挙動を踏襲）。
  if [ -f "Cargo.lock" ]; then
    check_lock "Cargo.lock" || failed=1
  else
    echo "::notice::Cargo.lock が未追加のため依存禁止検査をスキップしました（workspace 作成後に有効化されます）"
  fi

  # OSS 直接比較ハーネス（scripts/bench/oss-gemm-compare/。許容依存第 9 区分。
  # .claude/rules/deps-policy.md）の Cargo.lock。第 9 区分有効化後は常時存在すべき
  # ものとして fail-closed（不在はエラー。本体 Cargo.lock と異なり notice スキップしない）
  # とする。
  local oss_gemm_compare_lock="scripts/bench/oss-gemm-compare/Cargo.lock"
  if [ -f "${oss_gemm_compare_lock}" ]; then
    check_lock "${oss_gemm_compare_lock}" || failed=1
  else
    echo "::error::${oss_gemm_compare_lock} が見つかりません（許容依存第 9 区分は有効化済みのため必須。.claude/rules/deps-policy.md）" >&2
    failed=1
  fi

  if [ "${failed}" -ne 0 ]; then
    return 1
  fi
}

# cargo tree 形式（`<crate> v<version>` 行）のテキストに対する検査本体。
# check_tree（実際の cargo tree 出力）と self_test（固定 fixture）の双方から呼ぶ共通経路
# とすることで、self-test が FORBIDDEN_TREE_PATTERN の退行も確実に検出できるようにする。
check_tree_text() {
  local label="$1"
  local text="$2"
  if echo "${text}" | grep -qE "${FORBIDDEN_TREE_PATTERN}"; then
    echo "::error::依存禁止リストのクレートが ${label} に含まれています（.claude/rules/deps-policy.md）:" >&2
    echo "${text}" | grep -E "${FORBIDDEN_TREE_PATTERN}" >&2
    return 1
  fi
  echo "OK: ${label}に依存禁止リストの混入なし"
}

check_tree() {
  # --locked: Cargo.lock の意図しない書き換え・runner 汚染を防止する（.claude/rules/ci.md の
  # cargo deny と同一方針）。--target all: cfg(target_os = "macos") 限定の Metal 系依存
  # （objc2 配下等）も検査対象に含める。
  local tree_output
  # -e normal,build,dev: Cargo.lock 検査（dev-dependencies も含む全依存を対象）との
  # カバレッジ非対称をなくすため、開発依存（criterion 等）も検査対象に含める。
  if ! tree_output=$(cargo tree --workspace --all-features --locked -e normal,build,dev --target all --prefix none 2>&1); then
    echo "NG: cargo tree の実行に失敗しました:" >&2
    echo "${tree_output}" >&2
    return 1
  fi
  check_tree_text "cargo tree 出力" "${tree_output}"
}

self_test() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local lock_clean="${script_dir}/testdata/cargo-lock-clean.txt"
  local lock_forbidden="${script_dir}/testdata/cargo-lock-forbidden.txt"
  local tree_clean="${script_dir}/testdata/cargo-tree-clean.txt"
  local tree_forbidden="${script_dir}/testdata/cargo-tree-forbidden.txt"
  local failed=0

  for f in "${lock_clean}" "${lock_forbidden}" "${tree_clean}" "${tree_forbidden}"; do
    if [ ! -f "${f}" ]; then
      echo "NG: self-test fixture が見つかりません（${f}）" >&2
      return 1
    fi
  done

  # ポジティブ: 禁止クレートを含まない fixture は pass（exit 0）すること。
  if check_lock "${lock_clean}" >/dev/null; then
    echo "self-test OK: lock clean fixture は pass する"
  else
    echo "self-test NG: lock clean fixture が誤って fail した（検査ロジックが誤検出している）" >&2
    failed=1
  fi

  # ネガティブ: 禁止クレートを含む fixture は fail（exit 非 0）すること
  # （受け入れ条件「禁止クレート混入時に fail-closed で失敗する」の機械検証）。
  if check_lock "${lock_forbidden}" >/dev/null 2>&1; then
    echo "self-test NG: lock forbidden fixture が誤って pass した（検査ロジックが退行している）" >&2
    failed=1
  else
    echo "self-test OK: lock forbidden fixture は fail する"
  fi

  # FORBIDDEN_TREE_PATTERN（cargo tree 検査）は check_lock 経由では検証されないため、
  # check_tree_text を同じ fixture 方式で直接検証する（tree 側の退行取りこぼし防止）。
  if check_tree_text "self-test tree clean fixture" "$(cat "${tree_clean}")" >/dev/null; then
    echo "self-test OK: tree clean fixture は pass する"
  else
    echo "self-test NG: tree clean fixture が誤って fail した（検査ロジックが誤検出している）" >&2
    failed=1
  fi

  if check_tree_text "self-test tree forbidden fixture" "$(cat "${tree_forbidden}")" >/dev/null 2>&1; then
    echo "self-test NG: tree forbidden fixture が誤って pass した（検査ロジックが退行している）" >&2
    failed=1
  else
    echo "self-test OK: tree forbidden fixture は fail する"
  fi

  if [ "${failed}" -ne 0 ]; then
    return 1
  fi
  echo "OK: self-test すべて pass"
}

main() {
  local subcommand="${1:-}"
  case "${subcommand}" in
    lock)
      local lock_path="${2:-}"
      [ -n "${lock_path}" ] || usage
      check_lock "${lock_path}"
      ;;
    lock-all)
      check_lock_all
      ;;
    tree)
      check_tree
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
