#!/usr/bin/env bash
# 0.15.0 同席計測（#4177 Mac mini・#4178 GB10）: 道具の自己テスト（両機体のディレクトリで同一内容。実機なし・モックで Linux 上でも動く）。
# 検査する範囲:
#   1. 環境変数の検査（env_guard.sh）: FANDHE_AI_*・RAYON_NUM_THREADS の検出、名前だけを出し値を出さないこと、紛らわしい名前を誤検出しないこと。
#   2. マスク・漏れ検査・SHA256SUMS（mask_lib.sh）: ユーザー名の語境界（別の語の一部は置換も検出もしない）、ホーム・ホスト名の置換先、
#      検査失敗が fail-closed になること、IPv4 検査が pip freeze の 4 区切り版番号を除くこと。
#   3. venv の freeze 検査（venv_preflight.sh）: macmini（uv pip freeze）・gb10（pip freeze + python --version）の byte 一致・不一致・不在。
#   4. 他ジョブゲートのラッパー（foreign_gate_wrapper.sh。Mac mini 側のディレクトリにだけある）: CLEAN／FOREIGN／タイムアウト／
#      ゲート自身の失敗／sha256 不一致の判定分岐。python3・sysctl・sleep をモックに差し替え、固定済みの 2 ファイルの中身は変えずに本物を使う。
# 使い方: bash self-test.sh（全部通れば rc 0。FAIL が 1 件でもあれば rc 1。値は表示しない）。
# bash 3.2 互換（連想配列・mapfile・${v,,} なし）。
# shellcheck disable=SC2015,SC2016,SC2030,SC2031,SC2034
set -uo pipefail
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SD}/../../../../.." && pwd)}"
# shellcheck source=/dev/null
. "${SD}/env_guard.sh"
# shellcheck source=/dev/null
. "${SD}/mask_lib.sh"
# shellcheck source=/dev/null
. "${SD}/venv_preflight.sh"
T="$(mktemp -d "${TMPDIR:-/tmp}/selftest-0150.XXXXXX")" || exit 1
trap 'chmod -R u+rwx "${T}" 2>/dev/null; rm -rf "${T}"' EXIT
N=0; BAD=0
ok() { N=$((N + 1)); }
ng() { N=$((N + 1)); BAD=$((BAD + 1)); echo "self-test FAIL: $1" >&2; }
expect_rc() { # <名前> <期待 rc> <実際の rc>
  if [ "$2" = "$3" ]; then ok; else ng "$1（期待 rc=$2・実際 rc=$3）"; fi
}
expect_eq() { if [ "$2" = "$3" ]; then ok; else ng "$1"; fi; }

# ---- 1. 環境変数の検査
envg() { env -i PATH="${PATH}" "$@" bash -c '. "$0"; env_guard_check' "${SD}/env_guard.sh"; }
out="$(envg 2>/dev/null)"; rc=$?
expect_rc "env: 何も無ければ rc 0" 0 "${rc}"
expect_eq "env: 記録 2 行" "fandhe_ai_env=none
rayon_num_threads_env=none" "${out}"
for spec in "FANDHE_AI_SME=0" "FANDHE_AI_X=" "RAYON_NUM_THREADS=4" "FANDHE_AI_CUDA_INFER_GRAPH=1"; do
  err="$(envg "${spec}" 2>&1 >/dev/null)"; rc=$?
  expect_rc "env: ${spec%%=*} を検出して rc 3" 3 "${rc}"
  case "${err}" in *"${spec%%=*}"*) ok ;; *) ng "env: 名前が stderr に出ない（${spec%%=*}）" ;; esac
done
err="$(envg FANDHE_AI_Y=secretvalue123 2>&1 >/dev/null)"
case "${err}" in *secretvalue123*) ng "env: 値を表示している" ;; *) ok ;; esac
for spec in "FANDHE_AI=1" "MY_FANDHE_AI_X=1" "RAYON_NUM_THREADS_X=1" "XRAYON_NUM_THREADS=1" "RAYON_THREADS=2"; do
  envg "${spec}" > /dev/null 2>&1; rc=$?
  expect_rc "env: ${spec%%=*} は誤検出しない" 0 "${rc}"
done

# ---- 2. マスク・漏れ検査・SHA256SUMS
HOME_X=/nonexistent-home
check_mask_pair() { # <host> <user>
  local host=$1 user=$2 d line i f
  d="${T}/mask-$$-${RANDOM}"; mkdir -p "${d}"
  i=0
  for line in "run by ${user} at 00:00" "${user}" "${user}: done" "owner=${user}" "path ${user}/x" "${user} ${user} ${user}" "on ${host} now" "home ${HOME_X}/${user}/w"; do
    i=$((i + 1)); mkdir -p "${d}/pos${i}"; f="${d}/pos${i}/a.txt"; printf '%s\n' "${line}" > "${f}"
    [ "$(leak_count "${d}/pos${i}" "${HOME_X}" "${host}" "" "${user}" 2> /dev/null)" != "0" ] && ok || ng "mask: 検出されるべき行 ${i}"
    mask_file "${f}" "${HOME_X}" "${host}" "" "${user}" "<cuda-node>" && ok || ng "mask: マスク失敗 ${i}"
    [ "$(leak_count "${d}/pos${i}" "${HOME_X}" "${host}" "" "${user}" 2> /dev/null)" = "0" ] && ok || ng "mask: マスク後に残る行 ${i}"
  done
  # ユーザー名が別の語の一部なら、置換も検出もしない
  i=0
  for line in "x${user}y" "${user}_z" "z_${user}" "${user}9q" "${user}-z" "z-${user}" "z-${user}-w"; do
    i=$((i + 1)); mkdir -p "${d}/neg${i}"; f="${d}/neg${i}/a.txt"; printf '%s\n' "${line}" > "${f}"
    [ "$(leak_count "${d}/neg${i}" "${HOME_X}" "__nohost__" "" "${user}" 2> /dev/null)" = "0" ] && ok || ng "mask: 検出されないべき行 ${i}"
    mask_file "${f}" "${HOME_X}" "__nohost__" "" "${user}" "<cuda-node>"
    [ "$(cat "${f}")" = "${line}" ] && ok || ng "mask: 書き換えられないべき行 ${i}"
  done
  rm -rf "${d}"
}
check_mask_pair "thost-01.example" "tuser9"
check_mask_pair "$(hostname 2> /dev/null || echo __nohost__)" "$(id -un 2> /dev/null || echo __nouser__)"
# 置換先（ホーム・ホスト名・ユーザー名）
mkdir -p "${T}/m1"; printf '%s\n' "p ${HOME_X}/w on thost-01.example by tuser9 (thost-01)" > "${T}/m1/a.log"
mask_file "${T}/m1/a.log" "${HOME_X}" "thost-01.example" "thost-01" "tuser9" "<metal-node>"
expect_eq "mask: 置換先" "p <home>/w on <metal-node> by <user> (<metal-node>)" "$(cat "${T}/m1/a.log")"
# ホームの特殊文字（正規表現として解釈されない）
mkdir -p "${T}/m2"; printf 'p /tmp/a.b+c/x\n' > "${T}/m2/a.log"
mask_file "${T}/m2/a.log" "/tmp/a.b+c" "" "" "" "<cuda-node>"
expect_eq "mask: ホームの特殊文字" "p <home>/x" "$(cat "${T}/m2/a.log")"
# IPv4 検査: 通常の txt は検出、pip-freeze-* は 4 区切りの版番号でも検出しない
mkdir -p "${T}/ip1" "${T}/ip2"; printf 'ip 10.1.2.3 x\n' > "${T}/ip1/a.txt"; printf 'nvidia-cudnn-cu13==9.20.0.48\n' > "${T}/ip2/pip-freeze-venv-bench.txt"
expect_eq "ip: txt の IPv4 を検出" 1 "$(leak_count "${T}/ip1" "" "" "" "" 2> /dev/null)"
expect_eq "ip: pip freeze の 4 区切りは対象外" 0 "$(leak_count "${T}/ip2" "" "" "" "" 2> /dev/null)"
# 検査失敗は「漏れなし」にしない（root では読み書きの拒否が起きないので飛ばす）
leak_count "${T}/__absent__" "" "" "" "" > /dev/null 2>&1; expect_rc "leak: 対象が無ければ rc 2" 2 "$?"
if [ "$(id -u)" != "0" ]; then
  mkdir -p "${T}/unread"; printf 'x\n' > "${T}/unread/a.txt"; chmod 000 "${T}/unread/a.txt"
  leak_count "${T}/unread" "/nonexistent-home" "__nohost__" "" "tuser9" > /dev/null 2>&1; expect_rc "leak: 読めないファイルで rc 2" 2 "$?"
  chmod 600 "${T}/unread/a.txt"
  mkdir -p "${T}/ro"; printf 'tuser9\n' > "${T}/ro/a.txt"; chmod 444 "${T}/ro/a.txt"
  mask_tree "${T}/ro" "/nonexistent-home" "" "" "tuser9" "<cuda-node>" > /dev/null 2>&1; rc=$?
  if [ "${rc}" -ne 0 ]; then ok; else ng "mask: 書けないファイルで mask_tree が成功を返す"; fi
  chmod 644 "${T}/ro/a.txt"
fi
# mask_tree は jsonl を書き換えない（漏れていれば leak_count が止める）
mkdir -p "${T}/mt"; printf 'tuser9\n' > "${T}/mt/a.jsonl"; printf 'tuser9\n' > "${T}/mt/a.out"
mask_tree "${T}/mt" "/nonexistent-home" "" "" "tuser9" "<cuda-node>"
expect_eq "mask_tree: jsonl は不変" "tuser9" "$(cat "${T}/mt/a.jsonl")"
expect_eq "mask_tree: out は置換" "<user>" "$(cat "${T}/mt/a.out")"
expect_eq "mask_tree 後の jsonl の漏れは検出される" 1 "$(leak_count "${T}/mt" "" "" "" "tuser9" 2> /dev/null)"
# SHA256SUMS: README.md と自身を除く全ファイル。検証が通る
mkdir -p "${T}/ss/a"; printf 'r\n' > "${T}/ss/README.md"; printf 'x\n' > "${T}/ss/a/b.txt"; printf 'y\n' > "${T}/ss/c.log"
write_sums "${T}/ss"; rc=$?; expect_rc "sums: 作成" 0 "${rc}"
if grep -q README "${T}/ss/SHA256SUMS" || grep -q SHA256SUMS "${T}/ss/SHA256SUMS"; then ng "sums: README.md か自身が入っている"; else ok; fi
expect_eq "sums: 2 行" 2 "$(wc -l < "${T}/ss/SHA256SUMS" | tr -d ' ')"
if command -v sha256sum > /dev/null 2>&1; then (cd "${T}/ss" && sha256sum -c SHA256SUMS > /dev/null 2>&1); else (cd "${T}/ss" && shasum -a 256 -c SHA256SUMS > /dev/null 2>&1); fi
expect_rc "sums: 検証が通る" 0 "$?"

# ---- 3. venv の freeze 検査
# 偽の uv（macmini）: `uv pip freeze --python <venv>/bin/python` → <venv>/freeze.fixture を出す
MB="${T}/mockbin"; mkdir -p "${MB}"
cat > "${MB}/uv" <<'EOF'
#!/bin/sh
# 偽の uv: --python の値の bin/ の親にある freeze.fixture を出す
while [ $# -gt 0 ]; do if [ "$1" = "--python" ]; then py=$2; shift; fi; shift; done
cat "$(dirname "${py}")/../freeze.fixture"
EOF
chmod +x "${MB}/uv"
mkvenv() { # <dir> <fixture 内容>: bin/python は pip freeze と --version を偽装する
  mkdir -p "$1/bin"; printf '%s\n' "$2" > "$1/freeze.fixture"
  cat > "$1/bin/python" <<'EOF'
#!/bin/sh
here="$(cd "$(dirname "$0")/.." && pwd)"
if [ "$1" = "--version" ]; then echo "Python 3.12.3"; exit 0; fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "freeze" ]; then cat "${here}/freeze.fixture"; exit 0; fi
exit 9
EOF
  chmod +x "$1/bin/python"
}
FX="torch==2.14.0
numpy==2.5.4"
mkvenv "${T}/pv/venv-torch" "${FX}"; mkvenv "${T}/pv/venv-tf" "tensorflow==2.16.2"
mkdir -p "${T}/ref"; printf '%s\n' "${FX}" > "${T}/ref/pip-freeze-venv-torch.txt"; printf '%s\n' "tensorflow==2.16.2" > "${T}/ref/pip-freeze-venv-tf.txt"
( PATH="${MB}:${PATH}"; export FREEZE_REF_DIR="${T}/ref"; vp_check macmini "${T}/pv" > /dev/null 2>&1 ); expect_rc "venv macmini: byte 一致" 0 "$?"
printf '%s\n' "tensorflow==2.16.3" > "${T}/pv/venv-tf/freeze.fixture"
( PATH="${MB}:${PATH}"; export FREEZE_REF_DIR="${T}/ref"; vp_check macmini "${T}/pv" > /dev/null 2>&1 ); expect_rc "venv macmini: 1 つ不一致なら失敗" 1 "$?"
printf '%s\n' "tensorflow==2.16.2" > "${T}/pv/venv-tf/freeze.fixture"
rm -f "${T}/ref/pip-freeze-venv-tf.txt"
( PATH="${MB}:${PATH}"; export FREEZE_REF_DIR="${T}/ref"; vp_check macmini "${T}/pv" > /dev/null 2>&1 ); expect_rc "venv macmini: 参照 freeze が無ければ失敗" 1 "$?"
printf '%s\n' "tensorflow==2.16.2" > "${T}/ref/pip-freeze-venv-tf.txt"
( PATH="${MB}:${PATH}"; export FREEZE_REF_DIR="${T}/ref"; vp_check macmini "${T}/nopv" > /dev/null 2>&1 ); expect_rc "venv macmini: venv が無ければ失敗" 1 "$?"
printf '%s\n%s\n' "${FX}" "Python 3.12.3" > "${T}/ref/pip-freeze-venv-bench.txt"
mkvenv "${T}/gv" "${FX}"
( export FREEZE_REF_DIR="${T}/ref"; vp_check gb10 "${T}/gv" > /dev/null 2>&1 ); expect_rc "venv gb10: freeze + python --version が byte 一致" 0 "$?"
printf '%s\n%s\n' "${FX}" "Python 3.12.4" > "${T}/ref/pip-freeze-venv-bench.txt"
( export FREEZE_REF_DIR="${T}/ref"; vp_check gb10 "${T}/gv" > /dev/null 2>&1 ); expect_rc "venv gb10: python の版が違えば失敗" 1 "$?"
( export FREEZE_REF_DIR="${T}/ref"; vp_check gb10 "${T}/nogv" > /dev/null 2>&1 ); expect_rc "venv gb10: venv が無ければ失敗" 1 "$?"
# 個別指定（VENV_TORCH・VENV_TF）: PY_VENV_DIR の配置と違っても指定した場所を検査する
mkvenv "${T}/alt/t" "${FX}"; mkvenv "${T}/alt/f" "tensorflow==2.16.2"
( PATH="${MB}:${PATH}"; export FREEZE_REF_DIR="${T}/ref" VENV_TORCH="${T}/alt/t" VENV_TF="${T}/alt/f"; vp_check macmini "${T}/nopv" > /dev/null 2>&1 ); expect_rc "venv macmini: VENV_TORCH・VENV_TF の個別指定" 0 "$?"
( PATH="${MB}:${PATH}"; export FREEZE_REF_DIR="${T}/ref" VENV_TORCH="${T}/alt/t" VENV_TF="${T}/alt/none"; vp_check macmini "${T}/pv" > /dev/null 2>&1 ); expect_rc "venv macmini: VENV_TF の指定先が無ければ失敗" 1 "$?"
# 既定の参照先（FREEZE_REF_DIR なし）が、リポジトリ内の 0.14.0 の収録 freeze に解決する
for spec in "macmini pip-freeze-venv-torch.txt" "macmini pip-freeze-venv-tf.txt" "gb10 pip-freeze-venv-bench.txt"; do
  rd="$(unset FREEZE_REF_DIR; vp_ref_dir "${spec%% *}")"
  [ -f "${rd}/${spec##* }" ] && ok || ng "venv: 既定の参照先に ${spec##* } が無い（${spec%% *}）"
done
( export FREEZE_REF_DIR="${T}/ref"; vp_check bogus "${T}/gv" > /dev/null 2>&1 ); expect_rc "venv: 未知の profile は失敗" 1 "$?"

# ---- 4. 他ジョブゲートのラッパー（Mac mini 側のみ）
if [ -f "${SD}/foreign_gate_wrapper.sh" ]; then
  # shellcheck source=/dev/null
  . "${SD}/foreign_gate_wrapper.sh"
  cat > "${MB}/python3" <<'EOF'
#!/bin/sh
# 偽の python3: foreign_job_gate.py の代わり。$MOCK_DIR/gate.seq の n 行目（最後の行は繰り返す）に従って CLEAN／FOREIGN／ERROR 等を返す
while [ $# -gt 0 ]; do case "$1" in --phase) ph=$2; shift ;; --out) out=$2; shift ;; esac; shift; done
n=$(cat "${MOCK_DIR}/gate.n" 2>/dev/null || echo 0); n=$((n + 1)); echo "${n}" > "${MOCK_DIR}/gate.n"
line=$(sed -n "${n}p" "${MOCK_DIR}/gate.seq"); [ -n "${line}" ] || line=$(tail -n 1 "${MOCK_DIR}/gate.seq")
case "${line}" in
  CLEAN) echo "CLEAN ${ph} t" > "${out}"; exit 0 ;;
  FOREIGN) echo "FOREIGN ${ph} t" > "${out}"; exit 1 ;;
  MISMATCH) echo "FOREIGN ${ph} t" > "${out}"; exit 0 ;;
  NOOUT) exit 0 ;;
  *) exit 2 ;;
esac
EOF
  cat > "${MB}/sysctl" <<'EOF'
#!/bin/sh
# 偽の sysctl: -n vm.loadavg だけ。$MOCK_DIR/load.seq の n 行目（最後の行は繰り返す）を load1 として `{ L 1.0 1.0 }` を出す
n=$(cat "${MOCK_DIR}/load.n" 2>/dev/null || echo 0); n=$((n + 1)); echo "${n}" > "${MOCK_DIR}/load.n"
l=$(sed -n "${n}p" "${MOCK_DIR}/load.seq"); [ -n "${l}" ] || l=$(tail -n 1 "${MOCK_DIR}/load.seq")
echo "{ ${l} 1.0 1.0 }"
EOF
  cat > "${MB}/sleep" <<'EOF'
#!/bin/sh
echo "$1" >> "${MOCK_DIR}/sleep.log"
EOF
  chmod +x "${MB}/python3" "${MB}/sysctl" "${MB}/sleep"
  CASE=0
  # run_case <gate.seq の内容> <load.seq の内容> <before|after>: 結果を CASE_RC・CASE_LOG・CASE_SLEEPS に入れる
  run_case() {
    CASE=$((CASE + 1)); local md="${T}/case${CASE}"; mkdir -p "${md}"
    printf '%s\n' "$1" > "${md}/gate.seq"; printf '%s\n' "$2" > "${md}/load.seq"; : > "${md}/sleep.log"; : > "${md}/gate.log"
    (
      PATH="${MB}:${PATH}"; MOCK_DIR="${md}"; export MOCK_DIR
      fgw_init "${REPO_ROOT}" "${md}/fg" > /dev/null 2>&1 || exit 90
      if [ "$3" = before ]; then fgw_before 1 3.0 "${md}/gate.log"; else fgw_after 1 "${md}/gate.log"; fi
    ); CASE_RC=$?
    CASE_LOG="$(cat "${md}/gate.log")"; CASE_SLEEPS="$(wc -l < "${md}/sleep.log" | tr -d ' ')"
  }
  has() { case "${CASE_LOG}" in *"$1"*) ok ;; *) ng "gate.log に「$1」が無い（$2）" ;; esac; }
  hasnt() { case "${CASE_LOG}" in *"$1"*) ng "gate.log に「$1」がある（$2）" ;; *) ok ;; esac; }
  # 前: CLEAN かつ load1 < 3.0 → 通過（待たない）
  run_case "CLEAN" "1.20" before
  expect_rc "gate before: CLEAN・load1 低 → 通過 rc 0" 0 "${CASE_RC}"; expect_eq "gate before: 待たない" 0 "${CASE_SLEEPS}"
  has "round1 gate=pass threshold=3.0 waited_s=0 load1=1.20" "通過行"; has "round1 foreign-before=clean" "前 CLEAN 行"
  # 前: FOREIGN が 2 回続いてから CLEAN → 30 秒を 2 回待って通過
  run_case "FOREIGN
FOREIGN
CLEAN" "1.20" before
  expect_rc "gate before: FOREIGN 2 回のあと CLEAN → 通過" 0 "${CASE_RC}"; expect_eq "gate before: 30 秒を 2 回待つ" 2 "${CASE_SLEEPS}"
  has "waited_s=60" "待った秒数"
  # 前: ずっと FOREIGN → 30 分（60 回）待って GATE-TIMEOUT rc 1
  run_case "FOREIGN" "1.20" before
  expect_rc "gate before: FOREIGN が続く → GATE-TIMEOUT rc 1" 1 "${CASE_RC}"; expect_eq "gate before: 60 回待つ" 60 "${CASE_SLEEPS}"
  has "round1 GATE-TIMEOUT reason=foreign-job" "タイムアウト行"; hasnt "foreign-before=clean" "通過行を書かない"
  # 前: CLEAN でも load1 が高いまま → GATE-TIMEOUT reason=load1
  run_case "CLEAN" "5.50" before
  expect_rc "gate before: load1 が高いまま → GATE-TIMEOUT" 1 "${CASE_RC}"; has "reason=load1" "load1 の理由"
  # 前: load1 が 3.0 ちょうどは通さない（< 3.0）
  run_case "CLEAN" "3.0" before
  expect_rc "gate before: load1 = 3.0 は不通過" 1 "${CASE_RC}"
  # 前: load1 が下がれば通過
  run_case "CLEAN" "4.0
2.9" before
  expect_rc "gate before: load1 が 2.9 に下がれば通過" 0 "${CASE_RC}"; expect_eq "gate before: 1 回待つ" 1 "${CASE_SLEEPS}"
  # 前: ゲート自身の失敗（rc 2・出力不一致・出力なし）→ GATE-INVALID rc 2
  for seq in ERROR MISMATCH NOOUT; do
    run_case "${seq}" "1.20" before
    expect_rc "gate before: ${seq} → GATE-INVALID rc 2" 2 "${CASE_RC}"; has "round1 GATE-INVALID reason=foreign-gate-error" "${seq}"
  done
  # 前: load1 が数値で取れない → GATE-INVALID reason=load1-unavailable
  run_case "CLEAN" "abc" before
  expect_rc "gate before: load1 が数値でない → GATE-INVALID" 2 "${CASE_RC}"; has "reason=load1-unavailable" "load1 不明"
  # 後: CLEAN → 0
  run_case "CLEAN" "1.20" after
  expect_rc "gate after: CLEAN → rc 0" 0 "${CASE_RC}"; has "round1 foreign-after=clean" "後 CLEAN 行"
  # 後: FOREIGN → rc 1（そのラウンドは無効）
  run_case "FOREIGN" "1.20" after
  expect_rc "gate after: FOREIGN → rc 1" 1 "${CASE_RC}"; has "round1 GATE-INVALID reason=foreign-after" "後 FOREIGN 行"; hasnt "foreign-after=clean" "CLEAN 行を書かない"
  # 後: ゲート自身の失敗 → rc 2
  for seq in ERROR MISMATCH NOOUT; do
    run_case "${seq}" "1.20" after
    expect_rc "gate after: ${seq} → rc 2" 2 "${CASE_RC}"; has "round1 GATE-INVALID reason=foreign-gate-error" "${seq}"
  done
  # sha256 不一致・欠落は使わない（fgw_init が失敗する）
  mkroot() { # <dir>: 本物の 2 ファイルを写した ROOT を作る
    mkdir -p "$1/docs/perf/logs/macmini-foreign-job-gate" "$1/scripts/bench/framework-compare"
    cp "${REPO_ROOT}/${FGW_GATE_REL}" "$1/${FGW_GATE_REL}"; cp "${REPO_ROOT}/${FGW_LIB_REL}" "$1/${FGW_LIB_REL}"
  }
  mkroot "${T}/rt0"; ( fgw_init "${T}/rt0" "${T}/rt0-fg" > /dev/null 2>&1 ); expect_rc "gate init: 固定済みの 2 ファイルなら成功" 0 "$?"
  mkroot "${T}/rt1"; echo "# x" >> "${T}/rt1/${FGW_GATE_REL}"
  ( fgw_init "${T}/rt1" "${T}/rt1-fg" > /dev/null 2>&1 ); expect_rc "gate init: foreign_job_gate.py の sha256 不一致で失敗" 1 "$?"
  mkroot "${T}/rt2"; echo "# x" >> "${T}/rt2/${FGW_LIB_REL}"
  ( fgw_init "${T}/rt2" "${T}/rt2-fg" > /dev/null 2>&1 ); expect_rc "gate init: ab_foreign_gate.sh の sha256 不一致で失敗" 1 "$?"
  mkroot "${T}/rt3"; rm -f "${T}/rt3/${FGW_LIB_REL}"
  ( fgw_init "${T}/rt3" "${T}/rt3-fg" > /dev/null 2>&1 ); expect_rc "gate init: ファイルが無ければ失敗" 1 "$?"
  mkroot "${T}/rt4"; mkdir -p "${T}/rt4-fg"
  ( fgw_init "${T}/rt4" "${T}/rt4-fg" > /dev/null 2>&1 ); expect_rc "gate init: 出力ディレクトリが既にあれば失敗（差し替え禁止）" 1 "$?"
  ( fgw_init "relative/root" "${T}/rt5-fg" > /dev/null 2>&1 ); expect_rc "gate init: 相対パスの ROOT は失敗" 1 "$?"
  # 本番の固定値が RULE（#4168）の値と一致している（ラッパーの定数のドリフト検出）
  rule="${REPO_ROOT}/docs/perf/logs/prepost-0150-4168/RULE.txt"
  if [ -f "${rule}" ]; then
    grep -q "^${FGW_GATE_SHA256}  ${FGW_GATE_REL}\$" "${rule}" && ok || ng "ラッパーの foreign_job_gate.py の固定値が #4168 の RULE と違う"
    grep -q "^${FGW_LIB_SHA256}  ${FGW_LIB_REL}\$" "${rule}" && ok || ng "ラッパーの ab_foreign_gate.sh の固定値が #4168 の RULE と違う"
  fi
fi

echo "self-test: ${N} 件中 FAIL ${BAD} 件"
[ "${BAD}" -eq 0 ]
