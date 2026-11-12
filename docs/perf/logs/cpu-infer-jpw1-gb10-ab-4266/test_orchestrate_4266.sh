#!/usr/bin/env bash
# #4266 orchestrate.sh の前提検査を、実機なしで通す試験（開発機・Linux のどちらでも動く。bash 3.2 可）。
# 役割: 固定前の捨て試走の代わりに、orchestrate.sh が #4161 から変えた部分（RULE.v2.txt を読む・本スクリプト自身と patch の
#   sha256 照合）と、その前後の前提検査が、本物の RULE.v2.txt・patch で通ることを確かめる。
#   GB10・CUDA・cargo は使わない。nvidia-smi と torch の python は偽物を置き、計測対象 tree の代わりに小さな tar を使う
#   （infer_scope.rs は本リポジトリの実物を入れるので、patch の当たり方は本物と同じ）。
#   成功の形: 前提検査をすべて通り、2 腕のツリーを作って patch を当て、ビルドの最初（Cargo.lock の退避）で止まる。
# 使い方: bash docs/perf/logs/cpu-infer-jpw1-gb10-ab-4266/test_orchestrate_4266.sh   -> 最後に "test-orchestrate ok"
set -u
export LC_ALL=C
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../../.." && pwd)"
T="$(mktemp -d)" || exit 1
trap 'rm -rf "${T}"' EXIT
ng() { echo "NG: $*" >&2; exit 1; }
if command -v sha256sum >/dev/null 2>&1; then sha() { sha256sum "$1" | awk '{print $1}'; }; else sha() { shasum -a 256 "$1" | awk '{print $1}'; }; fi

# ---- 偽のコマンド（nvidia-smi は GB10 として空きを返す。sha256sum が無い機体では shasum で代わりを置く）
mkdir -p "${T}/bin" "${T}/home"
cat > "${T}/bin/nvidia-smi" <<'EOF'
#!/bin/sh
case "$*" in *query-gpu=name*) echo "NVIDIA GB10" ;; *utilization*) echo "0" ;; *) : ;; esac
EOF
printf '#!/bin/sh\nexit 0\n' > "${T}/bin/fake-python"
if ! command -v sha256sum >/dev/null 2>&1; then printf '#!/bin/sh\nexec shasum -a 256 "$@"\n' > "${T}/bin/sha256sum"; fi
chmod +x "${T}/bin/"*

# ---- 計測対象 tree の代わり（infer_scope.rs は実物。Cargo.lock は置かない = ビルドの最初で止まる）
SRC="${T}/src"
mkdir -p "${SRC}/crates/backend-cpu/src/gemm_blis"
cp "${REPO}/crates/backend-cpu/src/gemm_blis/infer_scope.rs" "${SRC}/crates/backend-cpu/src/gemm_blis/" || ng "infer_scope.rs が無い"
echo "dummy" > "${SRC}/README"
(cd "${SRC}" && tar -cf "${T}/tree.tar" .) || ng "tar 作成失敗"
FP="$(sha "${T}/tree.tar")"
FAKESHA="0123456789abcdef0123456789abcdef01234567"

# mk_logdir <名前> : 本物の道具を LOGDIR に置き、RULE.v2.txt の tree_sha・tree_fingerprint だけを偽の tree に合わせる
mk_logdir() {
  local L="${T}/$1/log"
  mkdir -p "${L}" "${T}/$1/base" "${T}/$1/work"
  cp "${HERE}/RULE.v2.txt" "${HERE}/aggregate.py" "${HERE}/orchestrate.sh" "${HERE}/ab_patch_inf_jpw1.diff" "${L}/"
  sed -e "s/^  tree_sha: .*/  tree_sha: ${FAKESHA}/" -e "s/^  tree_fingerprint: .*/  tree_fingerprint: ${FP}/" \
    "${HERE}/RULE.v2.txt" > "${L}/RULE.v2.txt"
  tar -C "${T}/$1/base" -xf "${T}/tree.tar"
  echo "${FAKESHA}" > "${T}/$1/base/.rev-stamp"
  echo "#4266 measure test" > "${T}/$1/owner"
}
# run_o <名前> -> 終了コード（出力は ${T}/<名前>.log）
run_o() {
  HOME="${T}/home" PATH="${T}/bin:${PATH}" BASE="${T}/$1/base" LOGDIR="${T}/$1/log" WORK="${T}/$1/work" \
    ARCHIVE="${T}/tree.tar" PY="${T}/bin/fake-python" BP="${T}/bin/fake-python" \
    HW_LOCK_OWNER_FILE="${T}/$1/owner" HW_LOCK_HELD=1 \
    bash "${T}/$1/log/orchestrate.sh" > "${T}/$1.log" 2>&1
}
no_outputs() { [ ! -e "${T}/$1/log/prereq" ] && [ ! -e "${T}/$1/log/tree_verify.txt" ] && [ ! -e "${T}/$1/log/patch_sha256.txt" ]; }

# ---- 本物の RULE.v2.txt の PARAMS: プレースホルダ検査に当たらない・aggregate.py の parse_params が通る
if grep -E '^  [A-Za-z0-9_-]+:.*<[^>]*>' "${HERE}/RULE.v2.txt" > /dev/null; then ng "RULE.v2.txt がプレースホルダ検査に当たる"; fi
(cd "${HERE}" && python3 -B -I -c 'import sys; sys.path.insert(0, "."); import aggregate as a; a.parse_params("RULE.v2.txt", a.self_sha())') \
  || ng "aggregate.py の parse_params が本物の RULE.v2.txt を読めない（自身の sha256 を含む）"
[ "$(sha "${HERE}/orchestrate.sh")" = "$(sed -n 's/^  orchestrate_script_sha256: *//p' "${HERE}/RULE.v2.txt")" ] || ng "orchestrate.sh のピンが不一致"
[ "$(sha "${HERE}/ab_patch_inf_jpw1.diff")" = "$(sed -n 's/^  patch_sha256: *//p' "${HERE}/RULE.v2.txt")" ] || ng "patch のピンが不一致"

# ---- 1. 正常: 前提検査を通り、2 腕のツリーを作り、ビルドの最初で止まる
mk_logdir good
run_o good && ng "good: 最後まで通った（Cargo.lock の退避で止まるはず）"
grep -q 'Cargo.lock の退避失敗' "${T}/good.log" || { cat "${T}/good.log" >&2; ng "good: 前提検査で止まった"; }
grep -q 'self-test OK' "${T}/good.log" || ng "good: aggregate.py の self-test が走っていない"
[ "$(grep -c 'INFER_SCOPE_JPW1_ENABLED: bool = false;' "${T}/good/log/tree_verify.txt")" = 1 ] || ng "good: off 腕の定数"
[ "$(grep -c 'INFER_SCOPE_JPW1_ENABLED: bool = true;' "${T}/good/log/tree_verify.txt")" = 1 ] || ng "good: inf-jpw1 腕の定数"
[ "$(awk '{print $1}' "${T}/good/log/patch_sha256.txt")" = "$(sed -n 's/^  patch_sha256: *//p' "${HERE}/RULE.v2.txt")" ] \
  || ng "good: patch_sha256.txt が RULE.v2.txt と違う"

# ---- 2. orchestrate.sh を 1 行変えると、自身の sha256 で止まり何も書かない
mk_logdir orch
echo "# changed" >> "${T}/orch/log/orchestrate.sh"
run_o orch && ng "orch: 起動した"
grep -q 'orchestrate.sh の sha256 が RULE.v2.txt と不一致' "${T}/orch.log" || ng "orch: 自身の sha256 で止まらない"
no_outputs orch || ng "orch: 出力を書いた"

# ---- 3. patch を変えると、patch の sha256 で止まる
mk_logdir patch
sed 's/= true;/= true; \/\/ x/' "${HERE}/ab_patch_inf_jpw1.diff" > "${T}/patch/log/ab_patch_inf_jpw1.diff"
run_o patch && ng "patch: 起動した"
grep -q 'ab_patch_inf_jpw1.diff の sha256 が RULE.v2.txt と不一致' "${T}/patch.log" || ng "patch: patch の sha256 で止まらない"
no_outputs patch || ng "patch: 出力を書いた"

# ---- 4. patch が無いときも止まる
mk_logdir nopatch
rm "${T}/nopatch/log/ab_patch_inf_jpw1.diff"
run_o nopatch && ng "nopatch: 起動した"
grep -q 'ab_patch_inf_jpw1.diff の sha256 が RULE.v2.txt と不一致' "${T}/nopatch.log" || ng "nopatch: 止まらない"

# ---- 5. PARAMS の orchestrate_script_sha256 が無い・patch_sha256 が無い
for k in orchestrate_script_sha256 patch_sha256; do
  mk_logdir "no-${k}"
  sed -i.bak "/^  ${k}:/d" "${T}/no-${k}/log/RULE.v2.txt" && rm -f "${T}/no-${k}/log/RULE.v2.txt.bak"
  run_o "no-${k}" && ng "no-${k}: 起動した"
  grep -q 'orchestrate_script_sha256 / patch_sha256 が空' "${T}/no-${k}.log" || ng "no-${k}: 空で止まらない"
done

# ---- 6. v1 から引き継いだ検査も働く（.rev-stamp の不一致・ロック未宣言）
mk_logdir rev
echo "ffffffffffffffffffffffffffffffffffffffff" > "${T}/rev/base/.rev-stamp"
run_o rev && ng "rev: 起動した"
grep -q '.rev-stamp が RULE と不一致' "${T}/rev.log" || ng "rev: .rev-stamp で止まらない"
mk_logdir nolock
HOME="${T}/home" PATH="${T}/bin:${PATH}" BASE="${T}/nolock/base" LOGDIR="${T}/nolock/log" WORK="${T}/nolock/work" \
  ARCHIVE="${T}/tree.tar" PY="${T}/bin/fake-python" BP="${T}/bin/fake-python" HW_LOCK_OWNER_FILE="${T}/nolock/owner" \
  bash "${T}/nolock/log/orchestrate.sh" > "${T}/nolock.log" 2>&1 && ng "nolock: 起動した"
grep -q '実機ロック未取得' "${T}/nolock.log" || ng "nolock: ロック未宣言で止まらない"

echo "test-orchestrate ok"
