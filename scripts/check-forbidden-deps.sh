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
#                意図的に含むため、禁止リストの grep 検査（check_lock）ではなく
#                **専用の fail-closed 契約検査**（check_framework_compare）を適用する:
#                (1) Cargo.lock の存在（不在はエラー）、(2) 同ディレクトリの Cargo.toml が
#                独自の [workspace] を宣言していること（本体 workspace への構造的
#                非混入）、(3) 承認済み比較対象のピン（burn 0.21.0・candle-core 0.11.0・
#                fandhe-ai 0.5.0。deps-policy.md 第 9 区分の承認バージョン）が
#                Cargo.lock に存在すること（承認外バージョンへのドリフトを検出。
#                加えて各エントリが `source = "registry+https://github.com/rust-lang/
#                crates.io-index"` を伴うこと＝path/git 依存への差し替えで source/
#                checksum 行が消える・書き換わるケースを fail-closed に検出する。
#                イシュー #982）、(4) 各メンバー crate の [dependencies] が承認済み
#                allowlist（比較対象の =x.y.z 完全固定 + bench-common の path 依存）の
#                範囲内であること（`tch` 等 allowlist 外の直接依存追加・ドット付きキー
#                宣言・完全固定でないバージョン指定に加え、`@=` 付き承認済みエントリが
#                `path`/`git`/`registry`/`rev`/`branch`/`tag`/`package` キーで非
#                registry 取得元へ差し替えられていることを検出。`deny.toml` の
#                `allow-wildcard-paths = true`（bench-common 用）は path 依存自体を
#                止めないため、この manifest 層検査が承認済み比較対象の取得元を守る
#                唯一の防御となる。イシュー #982）、(5) 各 Cargo.toml のセクション
#                ヘッダが allowlist の範囲内であること（[dev-dependencies]・
#                [build-dependencies]・[target.'cfg'.dependencies]・
#                [dependencies.<crate>] 等の代替依存宣言経路を遮断）、
#                (6) workspace members 宣言が期待値と完全一致すること、
#                (7) ディレクトリ配下の Cargo.toml ファイル集合が契約と一致すること
#                （未登録 member crate の追加を遮断）。
#                承認済みピン以外への変更・検査の緩和はユーザー承認必須
#                （deps-policy.md）。本体 workspace への混入は引き続きルート
#                Cargo.lock・cargo tree 検査で fail-closed に検出される。
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

# framework-compare の承認済み比較対象（`@=` 付き allowlist エントリ）が
# crates.io registry 以外から取得されていることを示す TOML キーの候補
# （check_manifest_deps_text が使う。イシュー #982）。`path`/`git` は取得元の
# 直接指定、`registry` は代替レジストリ指定、`rev`/`branch`/`tag` は git 依存の
# 付随キー（`git` を伴わず単独で現れても取得元差し替えの兆候として扱う）、
# `package` は rename（`fandhe-ai = { package = "...", path = "..." }` 等）による
# allowlist 名の偽装を防ぐため対象に含める。
ORIGIN_KEY_ALT='path|git|registry|rev|branch|tag|package'

# フレームワーク横並びベンチ（許容依存第 9 区分の適用範囲拡張。
# .claude/rules/deps-policy.md「ベンチ比較対象（フレームワーク横並び）」）の
# 承認済みピン。`<crate>=<version>` 形式のスペース区切り。ここを緩める・削る変更は
# ユーザー承認必須（検査対象の追加は fail-closed の強化であり承認不要）。
FRAMEWORK_COMPARE_DIR="scripts/bench/framework-compare"
FRAMEWORK_COMPARE_PINS="burn=0.21.0 candle-core=0.11.0 fandhe-ai=0.5.0"

# 同ベンチの各メンバー crate が [dependencies] に宣言してよい直接依存の allowlist
# （`<manifest 相対パス>:<crate>[@=version]...` 形式）。承認済み比較対象（=x.y.z 完全
# 固定必須）と workspace 内の path 依存 bench-common のみを許容し、これ以外の直接依存
# （禁止リストの `tch` を含む任意のクレート）の追加を fail-closed に検出する。
# allowlist の拡張はユーザー承認必須（deps-policy.md 第 9 区分）。
FRAMEWORK_COMPARE_MANIFEST_ALLOWLIST="\
bench-common/Cargo.toml:
bench-fandhe/Cargo.toml:bench-common,fandhe-ai@=0.5.0
bench-candle/Cargo.toml:bench-common,candle-core@=0.11.0
bench-burn/Cargo.toml:bench-common,burn@=0.21.0"

# 同ベンチ workspace の members 宣言の期待値（完全一致で検査する。member の追加・
# 削除・並び替えはユーザー承認必須の契約変更として fail-closed に検出する）。
FRAMEWORK_COMPARE_EXPECTED_MEMBERS='members = ["bench-common", "bench-fandhe", "bench-candle", "bench-burn"]'

# 各 Cargo.toml に出現してよい TOML セクションヘッダの allowlist（完全一致）。
# [dev-dependencies]・[build-dependencies]・[target.'cfg(...)'.dependencies]・
# [dependencies.<crate>]（ドット付きセクション）等、[dependencies] 以外の経路での
# 依存宣言をセクション単位で fail-closed に遮断する（allowlist 外のセクションは
# 内容を問わずエラー）。
FRAMEWORK_COMPARE_MEMBER_SECTIONS="[package],[dependencies],[features]"
FRAMEWORK_COMPARE_ROOT_SECTIONS="[workspace],[profile.release]"

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

  # フレームワーク横並びベンチ（scripts/bench/framework-compare/。第 9 区分の
  # 適用範囲拡張）: 禁止リスト grep の代わりに専用の fail-closed 契約検査を適用する
  # （関数冒頭コメント参照。存在・[workspace] 隔離・承認済みピン・直接依存
  # allowlist・セクション allowlist・members 完全一致・Cargo.toml 集合一致の 7 点）。
  check_framework_compare || failed=1

  if [ "${failed}" -ne 0 ]; then
    return 1
  fi
}

# Cargo.lock 形式テキストに `name = "<crate>"` + `version = "<version>"` +
# `source = "registry+https://github.com/rust-lang/crates.io-index"` が
# **同一の [[package]] エントリ内に揃って**存在することを検査する（テキスト入力版。
# self-test から固定文字列で直接検証できるよう、ファイル I/O と分離する）。
# source 行の必須化はイシュー #982: path 依存は [[package]] エントリから
# source/checksum 行そのものが消え、git 依存は `source = "git+…"` になるため、
# version 一致だけでは非 registry 取得元への差し替えを検出できない
# （check_manifest_deps_text の非 registry 取得元検出と対をなす Cargo.lock 層の防御）。
# エントリ単位での判定が必須な理由（イシュー #982 レビュー指摘・PR #991
# codex-review / Cursor Bugbot 双方が独立検出）: 旧実装は `grep -A3` で crate 名に
# 一致する全 [[package]] ブロックの断片を 1 つの文字列へ連結したうえで、
# version・source を別々に（`grep -qF` 2 回で）検査していた。このため
# 「承認バージョンのエントリが path/git 依存（source 行なし）」かつ
# 「同名クレートの別バージョンのエントリが registry 依存」という Cargo.lock が
# 現れると、version 条件は前者のブロックで、source 条件は後者のブロックで
# それぞれ成立してしまい、承認バージョン自体が非 registry 取得元へ差し替えられた
# ケースを検出できない fail-open だった（本リポの Cargo.lock に実際に
# candle-core 0.10.2 / 0.11.0 のような同名複数バージョンが併存することを
# Cursor Bugbot が指摘）。Cargo.lock の [[package]] エントリは空行区切りの
# パラグラフのため、awk の paragraph mode（RS=""）でエントリ単位に分割し、
# name・version・source の 3 条件を同一エントリ内でのみ判定する。
check_lock_pin_text() {
  local label="$1"
  local text="$2"
  local crate="$3"
  local version="$4"
  local result
  result=$(echo "${text}" | awk -v crate="${crate}" -v version="${version}" '
    BEGIN { RS = ""; FS = "\n"; found_name_version = 0; ok = 0 }
    {
      has_name = 0; has_version = 0; has_source = 0
      for (i = 1; i <= NF; i++) {
        line = $i
        if (line == "name = \"" crate "\"") has_name = 1
        if (line == "version = \"" version "\"") has_version = 1
        if (line == "source = \"registry+https://github.com/rust-lang/crates.io-index\"") has_source = 1
      }
      if (has_name && has_version) {
        found_name_version = 1
        if (has_source) { ok = 1 }
      }
    }
    END {
      if (ok) { print "OK"; exit }
      if (found_name_version) { print "NO_SOURCE"; exit }
      print "NOT_FOUND"
    }
  ')
  case "${result}" in
    OK)
      echo "OK: ${label} に承認済みピン ${crate} ${version}（registry 取得元）が存在"
      return 0
      ;;
    NO_SOURCE)
      echo "::error::${label} の承認済みピン ${crate} ${version} が crates.io registry 取得元ではありません（path/git 依存等への差し替えの可能性。.claude/rules/deps-policy.md 第 9 区分）" >&2
      return 1
      ;;
    *)
      echo "::error::${label} に承認済みピン ${crate} ${version} が見つかりません（承認外バージョンへのドリフト、または比較対象の削除。.claude/rules/deps-policy.md 第 9 区分）" >&2
      return 1
      ;;
  esac
}

# Cargo.toml 形式テキストの [dependencies] セクションを、許容された直接依存の
# allowlist（カンマ区切り `<crate>` または `<crate>@=<version>`。`@=` 付きは
# `"=<version>"` の完全固定宣言を要求する）と突合する（テキスト入力版。self-test
# から固定文字列で直接検証できるよう、ファイル I/O と分離する）。allowlist 外の
# 直接依存（禁止リストの `tch` を含む任意のクレート）・完全固定でないバージョン
# 指定を fail-closed に検出する。`@=` 付きエントリ（＝承認済み比較対象。registry
# 取得・完全固定必須の意味）は、`fandhe-ai = { version = "=0.5.0", path = "…" }` の
# ように完全固定文字列を保ったまま取得元だけを path/git 等へ差し替える迂回も
# 検出する（イシュー #982。ORIGIN_KEY_ALT 参照）。
check_manifest_deps_text() {
  local label="$1"
  local text="$2"
  local allowlist="$3" # 例: "bench-common,burn@=0.21.0"（空文字列 = 直接依存なし）
  local failed=0

  # [dependencies] セクションのみを抜き出し、依存宣言行（`name = ...`）を列挙する。
  # [features]・[package] 等の他セクションは対象外。
  local dep_lines
  # ドット付きキー（`tch.version = "..."` 形式）も依存宣言として拾う（名前は最初の
  # `.` までで切り出して allowlist と突合する）。
  # インデント付きのセクションヘッダ・キー宣言も TOML として有効なため、行頭の
  # 空白を許容して走査する（インデントによる検査すり抜けの防止）。
  # セクションヘッダは行末コメント（`[dependencies]  # ...`）付きでも TOML として
  # 有効なため、ヘッダ判定はコメントの有無を許容する（コメント付きヘッダ配下の
  # 依存宣言が走査から漏れる fail-open の防止）。
  dep_lines=$(echo "${text}" | awk '
    /^[[:space:]]*\[dependencies\][[:space:]]*(#.*)?$/ { in_deps = 1; next }
    /^[[:space:]]*\[/ { in_deps = 0 }
    in_deps && /^[[:space:]]*[a-zA-Z0-9_.-]+[[:space:]]*=/ { print }
  ')

  local line raw_key name entry entry_name entry_version found
  while IFS= read -r line; do
    [ -n "${line}" ] || continue
    raw_key="${line%%=*}"
    # 依存名の前後空白を除去する（ドット付きキーはこの時点ではまだ切り出さない。
    # 取得元キー検出でドット以降のサフィックスも使うため raw_key に残しておく）。
    raw_key="$(echo "${raw_key}" | tr -d '[:space:]')"
    name="${raw_key%%.*}"
    found=0
    local IFS_SAVE="${IFS}"
    IFS=','
    for entry in ${allowlist}; do
      IFS="${IFS_SAVE}"
      entry_name="${entry%%@*}"
      if [ "${name}" = "${entry_name}" ]; then
        found=1
        if [ "${entry}" != "${entry_name}" ]; then
          # `@=` 付きエントリ: `"=<version>"` の完全固定宣言を要求する。
          entry_version="${entry#*@}"
          if ! echo "${line}" | grep -qF "\"${entry_version}\""; then
            echo "::error::${label} の直接依存 ${name} が承認済みの完全固定 ${entry_version} で宣言されていません（.claude/rules/deps-policy.md 第 9 区分）: ${line}" >&2
            failed=1
          fi
          # 非 registry 取得元の検出（イシュー #982）: (a) ドット付きキー
          # （`fandhe-ai.path = "…"` 形式）は raw_key のサフィックスが取得元キー
          # 集合に含まれるかで判定する。(b) インライン table
          # （`fandhe-ai = { version = "=…", path = "…" }` 形式）は行内の
          # キー境界（`{`・`,`・空白の直後）で取得元キーが出現するかを判定する
          # （`default-features`・`features` 等の無関係キーを誤検出しないための
          # 境界指定）。
          if [ "${raw_key}" != "${name}" ] && echo "${raw_key#*.}" | grep -qE "^(${ORIGIN_KEY_ALT})\$"; then
            echo "::error::${label} の直接依存 ${name} が非 registry 取得元キー（${raw_key#*.}）で宣言されています（承認済み比較対象は crates.io registry からの取得が必須。.claude/rules/deps-policy.md 第 9 区分）: ${line}" >&2
            failed=1
          elif echo "${line}" | grep -qE "(^|[{,[:space:]])(${ORIGIN_KEY_ALT})[[:space:]]*="; then
            echo "::error::${label} の直接依存 ${name} が非 registry 取得元キーを伴って宣言されています（承認済み比較対象は crates.io registry からの取得が必須。.claude/rules/deps-policy.md 第 9 区分）: ${line}" >&2
            failed=1
          fi
        fi
        break
      fi
      IFS=','
    done
    IFS="${IFS_SAVE}"
    if [ "${found}" -eq 0 ]; then
      echo "::error::${label} に allowlist 外の直接依存 ${name} が宣言されています（承認済み比較対象以外の依存追加はユーザー承認必須。.claude/rules/deps-policy.md 第 9 区分）: ${line}" >&2
      failed=1
    fi
  done <<EOF_DEPS
${dep_lines}
EOF_DEPS

  if [ "${failed}" -ne 0 ]; then
    return 1
  fi
  echo "OK: ${label} の直接依存は allowlist（${allowlist:-なし}）の範囲内"
}

# Cargo.toml 形式テキストのセクションヘッダ（`[...]` 行）が allowlist（カンマ区切り・
# 完全一致）の範囲内であることを検査する（テキスト入力版）。[dev-dependencies]・
# [build-dependencies]・[target.'cfg(...)'.dependencies]・[dependencies.<crate>] 等、
# [dependencies] 以外の経路での依存宣言セクションを内容を問わず fail-closed に
# 遮断する。
check_manifest_sections_text() {
  local label="$1"
  local text="$2"
  local allowed="$3" # 例: "[package],[dependencies],[features]"
  local failed=0

  local section entry found
  while IFS= read -r section; do
    [ -n "${section}" ] || continue
    found=0
    local IFS_SAVE="${IFS}"
    IFS=','
    for entry in ${allowed}; do
      IFS="${IFS_SAVE}"
      if [ "${section}" = "${entry}" ]; then
        found=1
        break
      fi
      IFS=','
    done
    IFS="${IFS_SAVE}"
    if [ "${found}" -eq 0 ]; then
      echo "::error::${label} に allowlist 外のセクション ${section} が宣言されています（[dependencies] 以外の依存宣言経路はセクション単位で禁止。.claude/rules/deps-policy.md 第 9 区分）" >&2
      failed=1
    fi
  done <<EOF_SECTIONS
$(echo "${text}" | grep -E '^[[:space:]]*\[' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/[[:space:]]*#.*$//')
EOF_SECTIONS

  if [ "${failed}" -ne 0 ]; then
    return 1
  fi
  echo "OK: ${label} のセクションは allowlist（${allowed}）の範囲内"
}

# フレームワーク横並びベンチ（scripts/bench/framework-compare/）専用の fail-closed
# 契約検査（lock-all から呼ぶ。呼び出し元コメントの (1)〜(7)）。
# 同 workspace の Cargo.lock は比較対象として禁止リストのクレートを意図的に含むため
# check_lock（禁止リスト grep）は適用せず、代わりに「本体 workspace への構造的
# 非混入」と「承認済みピンからのドリフト検出」を fail-closed で検査する。
check_framework_compare() {
  local dir="${FRAMEWORK_COMPARE_DIR}"
  local lock="${dir}/Cargo.lock"
  local manifest="${dir}/Cargo.toml"
  local failed=0

  # (1) Cargo.lock の存在（第 9 区分の適用範囲拡張が有効化済みのため必須。
  #     再現性担保のコミット対象。不在はエラー）。
  if [ ! -f "${lock}" ]; then
    echo "::error::${lock} が見つかりません（第 9 区分の適用範囲拡張は有効化済みのため必須。.claude/rules/deps-policy.md）" >&2
    return 1
  fi

  # (2) 独自 [workspace] の宣言（本体 workspace への構造的非混入。宣言が消えると
  #     cargo が親 workspace を探索し、本体 Cargo.lock へ依存が混入しうる）。
  if [ ! -f "${manifest}" ] || ! grep -q '^\[workspace\]' "${manifest}"; then
    echo "::error::${manifest} が独自の [workspace] を宣言していません（本体 workspace への構造的非混入の契約。.claude/rules/deps-policy.md 第 9 区分）" >&2
    failed=1
  fi

  # (3) 承認済み比較対象のピンが Cargo.lock に存在すること。
  local pin crate version
  for pin in ${FRAMEWORK_COMPARE_PINS}; do
    crate="${pin%%=*}"
    version="${pin#*=}"
    check_lock_pin_text "${lock}" "$(cat "${lock}")" "${crate}" "${version}" || failed=1
  done

  # (4) 各メンバー crate の [dependencies] が allowlist（承認済み比較対象の完全固定 +
  #     bench-common の path 依存）の範囲内であること。allowlist 外の直接依存
  #     （禁止リストの `tch` を含む）の追加・完全固定でないバージョン指定を
  #     fail-closed に検出する。
  local mapping member_manifest member_allowlist
  while IFS= read -r mapping; do
    [ -n "${mapping}" ] || continue
    member_manifest="${dir}/${mapping%%:*}"
    member_allowlist="${mapping#*:}"
    if [ ! -f "${member_manifest}" ]; then
      echo "::error::${member_manifest} が見つかりません（framework-compare のメンバー構成が契約から変更されています。.claude/rules/deps-policy.md 第 9 区分）" >&2
      failed=1
      continue
    fi
    check_manifest_deps_text "${member_manifest}" "$(cat "${member_manifest}")" "${member_allowlist}" || failed=1
  done <<EOF_MAPPING
${FRAMEWORK_COMPARE_MANIFEST_ALLOWLIST}
EOF_MAPPING

  # (5) 各 Cargo.toml のセクションヘッダが allowlist の範囲内であること
  #     （[dev-dependencies]・[build-dependencies]・[target.'cfg'.dependencies]・
  #     [dependencies.<crate>] 等、(4) の [dependencies] 走査に乗らない依存宣言経路を
  #     セクション単位で遮断する）。
  check_manifest_sections_text "${manifest}" "$(cat "${manifest}")" "${FRAMEWORK_COMPARE_ROOT_SECTIONS}" || failed=1
  while IFS= read -r mapping; do
    [ -n "${mapping}" ] || continue
    member_manifest="${dir}/${mapping%%:*}"
    if [ -f "${member_manifest}" ]; then
      check_manifest_sections_text "${member_manifest}" "$(cat "${member_manifest}")" "${FRAMEWORK_COMPARE_MEMBER_SECTIONS}" || failed=1
    fi
  done <<EOF_MAPPING2
${FRAMEWORK_COMPARE_MANIFEST_ALLOWLIST}
EOF_MAPPING2

  # (6) workspace members 宣言が期待値と完全一致すること（allowlist 未登録の新規
  #     member crate を workspace へ追加して依存を持ち込む迂回を遮断する）。
  local members_line
  members_line=$(grep -E '^members = ' "${manifest}" | sed -e 's/[[:space:]]*$//' || true)
  if [ "${members_line}" != "${FRAMEWORK_COMPARE_EXPECTED_MEMBERS}" ]; then
    echo "::error::${manifest} の members 宣言が契約と一致しません（member の追加・削除はユーザー承認必須。.claude/rules/deps-policy.md 第 9 区分）。期待: ${FRAMEWORK_COMPARE_EXPECTED_MEMBERS} / 実際: ${members_line:-（members 行なし）}" >&2
    failed=1
  else
    echo "OK: ${manifest} の members 宣言は契約と一致"
  fi

  # (7) ディレクトリ配下の Cargo.toml が契約どおりのファイル集合であること
  #     （members 宣言に載らない場所への Cargo.toml 追加も検出する）。
  local expected_manifests actual_manifests
  expected_manifests=$(printf '%s\n' \
    "${dir}/Cargo.toml" \
    "${dir}/bench-common/Cargo.toml" \
    "${dir}/bench-fandhe/Cargo.toml" \
    "${dir}/bench-candle/Cargo.toml" \
    "${dir}/bench-burn/Cargo.toml" | sort)
  # target/（ビルド生成物。.gitignore 対象）配下はベンダーされた依存の Cargo.toml を
  # 含みうるため除外する（リポジトリにコミットされる範囲が検査対象）。
  actual_manifests=$(find "${dir}" -path "${dir}/target" -prune -o -name Cargo.toml -print | sort)
  if [ "${actual_manifests}" != "${expected_manifests}" ]; then
    echo "::error::${dir} 配下の Cargo.toml 集合が契約と一致しません（crate の追加・削除はユーザー承認必須。.claude/rules/deps-policy.md 第 9 区分）。" >&2
    echo "期待:" >&2
    echo "${expected_manifests}" >&2
    echo "実際:" >&2
    echo "${actual_manifests}" >&2
    failed=1
  else
    echo "OK: ${dir} 配下の Cargo.toml 集合は契約と一致"
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

  # check_lock_pin_text（framework-compare の承認済みピン検査）も同じ固定入力方式で
  # 直接検証する（新設の検査経路の退行取りこぼし防止。ファイル fixture は増やさず
  # インライン文字列で足りる）。source 行は crates.io registry 形式を含める
  # （イシュー #982 で source 必須化したため、承認済みバージョン fixture が
  # source 不在で誤って fail しないようにする）。
  local pin_ok_text='[[package]]
name = "burn"
version = "0.21.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "deadbeef"'
  local pin_drift_text='[[package]]
name = "burn"
version = "0.22.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "deadbeef"'
  if check_lock_pin_text "self-test pin fixture" "${pin_ok_text}" "burn" "0.21.0" >/dev/null; then
    echo "self-test OK: pin fixture（承認済みバージョン）は pass する"
  else
    echo "self-test NG: pin fixture（承認済みバージョン）が誤って fail した" >&2
    failed=1
  fi
  if check_lock_pin_text "self-test pin drift fixture" "${pin_drift_text}" "burn" "0.21.0" >/dev/null 2>&1; then
    echo "self-test NG: pin drift fixture が誤って pass した（ドリフト検出が退行している）" >&2
    failed=1
  else
    echo "self-test OK: pin drift fixture は fail する"
  fi

  # check_lock_pin_text の非 registry 取得元検出（イシュー #982）: path 依存化で
  # source/checksum 行そのものが消えるケースと、git 依存で source = "git+…" に
  # 書き換わるケースの双方を fail-closed に検出すること。
  local pin_path_dep_text='[[package]]
name = "burn"
version = "0.21.0"
dependencies = [
 "burn-core",
]'
  local pin_git_dep_text='[[package]]
name = "burn"
version = "0.21.0"
source = "git+https://github.com/tracel-ai/burn?rev=deadbeef#deadbeef"'
  if check_lock_pin_text "self-test pin path-dep fixture" "${pin_path_dep_text}" "burn" "0.21.0" >/dev/null 2>&1; then
    echo "self-test NG: pin path-dep fixture が誤って pass した（source 行不在＝path 依存の検出が退行している）" >&2
    failed=1
  else
    echo "self-test OK: pin path-dep fixture は fail する"
  fi
  if check_lock_pin_text "self-test pin git-dep fixture" "${pin_git_dep_text}" "burn" "0.21.0" >/dev/null 2>&1; then
    echo "self-test NG: pin git-dep fixture が誤って pass した（git source の検出が退行している）" >&2
    failed=1
  else
    echo "self-test OK: pin git-dep fixture は fail する"
  fi

  # check_lock_pin_text のエントリ横断 fail-open 回帰検出（イシュー #982・PR #991
  # codex-review / Cursor Bugbot 指摘）: 承認バージョンのエントリが path 依存
  # （source 行なし）で、かつ同名クレートの別バージョンのエントリが registry
  # 依存という Cargo.lock（[[package]] は空行区切り）を与えたとき、version 条件と
  # source 条件をエントリ横断でそれぞれ満たしてしまい誤って pass する退行を防ぐ。
  local pin_cross_entry_text='[[package]]
name = "burn"
version = "0.21.0"
dependencies = [
 "burn-core",
]

[[package]]
name = "burn"
version = "0.22.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "deadbeef"'
  if check_lock_pin_text "self-test pin cross-entry fixture" "${pin_cross_entry_text}" "burn" "0.21.0" >/dev/null 2>&1; then
    echo "self-test NG: pin cross-entry fixture が誤って pass した（同名クレートの別バージョン registry エントリを誤検出し fail-open している）" >&2
    failed=1
  else
    echo "self-test OK: pin cross-entry fixture は fail する"
  fi

  # check_manifest_deps_text（framework-compare の直接依存 allowlist 検査）も同方式で
  # 直接検証する（allowlist 外依存・非完全固定の検出退行の防止）。
  local manifest_ok_text='[package]
name = "bench-burn"

[dependencies]
bench-common = { path = "../bench-common" }
burn = { version = "=0.21.0", default-features = false }

[features]
default = ["metal"]'
  local manifest_extra_dep_text='[dependencies]
bench-common = { path = "../bench-common" }
burn = { version = "=0.21.0" }
tch = { version = "=0.22.0" }'
  local manifest_unpinned_text='[dependencies]
bench-common = { path = "../bench-common" }
burn = { version = "0.21" }'
  if check_manifest_deps_text "self-test manifest fixture" "${manifest_ok_text}" "bench-common,burn@=0.21.0" >/dev/null; then
    echo "self-test OK: manifest fixture（allowlist 内・完全固定）は pass する"
  else
    echo "self-test NG: manifest fixture（allowlist 内・完全固定）が誤って fail した" >&2
    failed=1
  fi
  if check_manifest_deps_text "self-test manifest extra-dep fixture" "${manifest_extra_dep_text}" "bench-common,burn@=0.21.0" >/dev/null 2>&1; then
    echo "self-test NG: manifest extra-dep fixture が誤って pass した（allowlist 外依存の検出が退行している）" >&2
    failed=1
  else
    echo "self-test OK: manifest extra-dep fixture は fail する"
  fi
  if check_manifest_deps_text "self-test manifest unpinned fixture" "${manifest_unpinned_text}" "bench-common,burn@=0.21.0" >/dev/null 2>&1; then
    echo "self-test NG: manifest unpinned fixture が誤って pass した（完全固定検査が退行している）" >&2
    failed=1
  else
    echo "self-test OK: manifest unpinned fixture は fail する"
  fi

  # ドット付きキー（`tch.version = ...`）による [dependencies] 内の依存宣言も
  # allowlist 外として検出されること。
  local manifest_dotted_text='[dependencies]
bench-common = { path = "../bench-common" }
tch.version = "=0.22.0"'
  if check_manifest_deps_text "self-test manifest dotted-key fixture" "${manifest_dotted_text}" "bench-common,burn@=0.21.0" >/dev/null 2>&1; then
    echo "self-test NG: manifest dotted-key fixture が誤って pass した（ドット付きキーの検出が退行している）" >&2
    failed=1
  else
    echo "self-test OK: manifest dotted-key fixture は fail する"
  fi

  # check_manifest_sections_text（セクション allowlist 検査）: [dev-dependencies]・
  # [dependencies.<crate>] 等の代替依存宣言セクションが遮断されること。
  local sections_ok_text='[package]
name = "x"

[dependencies]

[features]'
  local sections_dev_text='[package]

[dependencies]

[dev-dependencies]'
  local sections_dotted_text='[package]

[dependencies.tch]'
  if check_manifest_sections_text "self-test sections fixture" "${sections_ok_text}" "[package],[dependencies],[features]" >/dev/null; then
    echo "self-test OK: sections fixture（allowlist 内）は pass する"
  else
    echo "self-test NG: sections fixture（allowlist 内）が誤って fail した" >&2
    failed=1
  fi
  if check_manifest_sections_text "self-test sections dev fixture" "${sections_dev_text}" "[package],[dependencies],[features]" >/dev/null 2>&1; then
    echo "self-test NG: sections dev fixture が誤って pass した（[dev-dependencies] の遮断が退行している）" >&2
    failed=1
  else
    echo "self-test OK: sections dev fixture は fail する"
  fi
  if check_manifest_sections_text "self-test sections dotted fixture" "${sections_dotted_text}" "[package],[dependencies],[features]" >/dev/null 2>&1; then
    echo "self-test NG: sections dotted fixture が誤って pass した（[dependencies.<crate>] の遮断が退行している）" >&2
    failed=1
  else
    echo "self-test OK: sections dotted fixture は fail する"
  fi

  # インデント付きセクションヘッダ（TOML として有効）が検査をすり抜けないこと。
  local sections_indented_text='[package]

  [dev-dependencies]'
  if check_manifest_sections_text "self-test sections indented fixture" "${sections_indented_text}" "[package],[dependencies],[features]" >/dev/null 2>&1; then
    echo "self-test NG: sections indented fixture が誤って pass した（インデント付きヘッダの検出が退行している）" >&2
    failed=1
  else
    echo "self-test OK: sections indented fixture は fail する"
  fi

  # 行末コメント付きの [dependencies] ヘッダ配下の依存宣言も走査に乗ること。
  local manifest_comment_header_text='[dependencies] # comment
tch = { version = "=0.22.0" }'
  if check_manifest_deps_text "self-test manifest comment-header fixture" "${manifest_comment_header_text}" "bench-common,burn@=0.21.0" >/dev/null 2>&1; then
    echo "self-test NG: manifest comment-header fixture が誤って pass した（コメント付きヘッダの走査が退行している）" >&2
    failed=1
  else
    echo "self-test OK: manifest comment-header fixture は fail する"
  fi

  # インデント付きの依存キー宣言も [dependencies] 走査に乗ること。
  local manifest_indented_dep_text='[dependencies]
  tch = { version = "=0.22.0" }'
  if check_manifest_deps_text "self-test manifest indented-dep fixture" "${manifest_indented_dep_text}" "bench-common,burn@=0.21.0" >/dev/null 2>&1; then
    echo "self-test NG: manifest indented-dep fixture が誤って pass した（インデント付き依存宣言の検出が退行している）" >&2
    failed=1
  else
    echo "self-test OK: manifest indented-dep fixture は fail する"
  fi

  # check_manifest_deps_text の非 registry 取得元検出（イシュー #982）: `@=` 付き
  # 承認済みエントリが完全固定バージョン文字列を保ったまま path/git 依存へ差し替え
  # られる、現行検査（バージョン文字列一致のみ）がすり抜けていた形そのもの。
  local manifest_inline_path_dep_text='[dependencies]
bench-common = { path = "../bench-common" }
fandhe-ai = { version = "=0.5.0", path = "../../../crates/facade" }'
  local manifest_inline_git_dep_text='[dependencies]
bench-common = { path = "../bench-common" }
fandhe-ai = { version = "=0.5.0", git = "https://github.com/Fandhe-AI/fandhe-ai", rev = "deadbeef" }'
  local manifest_dotted_origin_text='[dependencies]
bench-common = { path = "../bench-common" }
fandhe-ai.path = "../../../crates/facade"'
  if check_manifest_deps_text "self-test manifest inline path-dep fixture" "${manifest_inline_path_dep_text}" "bench-common,fandhe-ai@=0.5.0" >/dev/null 2>&1; then
    echo "self-test NG: manifest inline path-dep fixture が誤って pass した（非 registry 取得元〈インライン table〉の検出が退行している）" >&2
    failed=1
  else
    echo "self-test OK: manifest inline path-dep fixture は fail する"
  fi
  if check_manifest_deps_text "self-test manifest inline git-dep fixture" "${manifest_inline_git_dep_text}" "bench-common,fandhe-ai@=0.5.0" >/dev/null 2>&1; then
    echo "self-test NG: manifest inline git-dep fixture が誤って pass した（非 registry 取得元〈インライン table・git〉の検出が退行している）" >&2
    failed=1
  else
    echo "self-test OK: manifest inline git-dep fixture は fail する"
  fi
  if check_manifest_deps_text "self-test manifest dotted origin-key fixture" "${manifest_dotted_origin_text}" "bench-common,fandhe-ai@=0.5.0" >/dev/null 2>&1; then
    echo "self-test NG: manifest dotted origin-key fixture が誤って pass した（非 registry 取得元〈ドット付きキー〉の検出が退行している）" >&2
    failed=1
  else
    echo "self-test OK: manifest dotted origin-key fixture は fail する"
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
