#!/usr/bin/env bash
# エンジン単体の対照 v0.15.0（#4314）: orchestrate.sh の分岐の自己テスト（実機なし・モックで Linux 上でも動く。計測しない）。
# 検査する範囲:
#   1. orchestrate.sh series の分岐: 5/5 完走（ALL-DONE・gate.txt が PASS で始まる）／前ゲートのタイムアウト／ゲート自身の失敗／
#      ラウンド後の FOREIGN／env guard／freeze 不一致（開始時・終了時）／ロック owner 不一致／終了後のロック解放／再実行の拒否。
#      python3・sysctl・uv・sleep・ハーネス・venv の python をモックに差し替える。他ジョブゲートの 2 ファイル（foreign_job_gate.py・
#      ab_foreign_gate.sh）は本物をリポジトリから写す（中身は変えない。python3 のモックが foreign_job_gate.py の代わりに答える）。
#   2. orchestrate.sh prepare: freeze 不一致ならビルドしない。
#   3. 同梱のコピーが元と byte 一致すること（foreign_gate_wrapper.sh・env_guard.sh は #4177、aggregate.py は 2026-11-08 版）と、aggregate.py --self-test。
# 使い方: bash self-test.sh（全部通れば rc 0。FAIL が 1 件でもあれば rc 1）。
# bash 3.2 互換（連想配列・mapfile・${v,,} なし）。
# shellcheck disable=SC2015,SC2016,SC2034
set -uo pipefail
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SD}/../../../../.." && pwd)}"
T="$(mktemp -d "${TMPDIR:-/tmp}/selftest-4314.XXXXXX")" || exit 1
trap 'rm -rf "${T}"' EXIT
N=0; BAD=0
ok() { N=$((N + 1)); }
ng() { N=$((N + 1)); BAD=$((BAD + 1)); echo "self-test FAIL: $1" >&2; }
expect_rc() { if [ "$2" = "$3" ]; then ok; else ng "$1（期待 rc=$2・実際 rc=$3）"; fi; }
GATE_REL=docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py
LIB_REL=scripts/bench/framework-compare/ab_foreign_gate.sh

# ---- モック（PATH の先頭に置く）
MB="${T}/mockbin"; mkdir -p "${MB}"
cat > "${MB}/python3" <<'EOF'
#!/bin/sh
# 偽の python3: foreign_job_gate.py の代わり。$MOCK_DIR/gate.seq の n 行目（最後の行は繰り返す）で CLEAN／FOREIGN／ERROR を返す
while [ $# -gt 0 ]; do case "$1" in --phase) ph=$2; shift ;; --out) out=$2; shift ;; esac; shift; done
n=$(cat "${MOCK_DIR}/gate.n" 2>/dev/null || echo 0); n=$((n + 1)); echo "${n}" > "${MOCK_DIR}/gate.n"
line=$(sed -n "${n}p" "${MOCK_DIR}/gate.seq"); [ -n "${line}" ] || line=$(tail -n 1 "${MOCK_DIR}/gate.seq")
case "${line}" in
  CLEAN) echo "CLEAN ${ph} t" > "${out}"; exit 0 ;;
  FOREIGN) echo "FOREIGN ${ph} t" > "${out}"; exit 1 ;;
  *) exit 2 ;;
esac
EOF
cat > "${MB}/sysctl" <<'EOF'
#!/bin/sh
# 偽の sysctl: vm.loadavg は $MOCK_DIR/load の値、それ以外は 0
case "$*" in *vm.loadavg*) echo "{ $(cat "${MOCK_DIR}/load") 1.0 1.0 }" ;; *) echo 0 ;; esac
EOF
cat > "${MB}/uv" <<'EOF'
#!/bin/sh
# 偽の uv: `uv pip freeze --python <venv>/bin/python` だけ。<venv>/freeze.txt を出す
[ "$1 $2 $3" = "pip freeze --python" ] || exit 2
cat "$(dirname "$(dirname "$4")")/freeze.txt"
EOF
cat > "${MB}/sleep" <<'EOF'
#!/bin/sh
echo "$1" >> "${MOCK_DIR}/sleep.log"
EOF
cat > "${MB}/cargo" <<'EOF'
#!/bin/sh
echo "cargo called" >> "${MOCK_DIR}/cargo.log"; exit 1
EOF
chmod +x "${MB}"/*

# mkcase <名前> <gate.seq> <load1>: 一時の HOME（ロック）・D（tree・target・venv-mlx）・VENV_TORCH・FREEZE_REF_DIR を作る
OWN="#4314 engine-0150-macmini 2026-01-01T00:00:00Z"
mkcase() {
  C="${T}/$1"; mkdir -p "${C}/home/work/.fandhe-hw-lock" "${C}/mock" "${C}/ref"
  printf '%s' "${OWN}" > "${C}/home/work/.fandhe-hw-lock/owner"
  printf '%s\n' "$2" > "${C}/mock/gate.seq"; printf '%s' "$3" > "${C}/mock/load"; : > "${C}/mock/sleep.log"
  CD="${C}/d"; mkdir -p "${CD}/tree/$(dirname "${GATE_REL}")" "${CD}/tree/$(dirname "${LIB_REL}")"
  cp "${REPO_ROOT}/${GATE_REL}" "${CD}/tree/${GATE_REL}"; cp "${REPO_ROOT}/${LIB_REL}" "${CD}/tree/${LIB_REL}"
  echo "sha=22a475e2df0cfcbf5854d827a79ddc1e7df4a4b9 dirty=0" > "${CD}/tree/.rev-stamp"
  mkdir -p "${CD}/target-oss/release" "${CD}/target-metal/release/examples"
  for b in target-oss/release/oss-gemm-compare target-metal/release/examples/metal_gemm_bench target-metal/release/examples/gemm_f32_prepared_bench; do
    printf '#!/bin/sh\necho mock-bench\nexit 0\n' > "${CD}/${b}"; chmod +x "${CD}/${b}"
  done
  for v in "${CD}/venv-mlx" "${C}/venv-torch"; do
    mkdir -p "${v}/bin"; printf '#!/bin/sh\necho mock-python\n' > "${v}/bin/python"; chmod +x "${v}/bin/python"
  done
  echo "mlx==0.32.4" > "${CD}/venv-mlx/freeze.txt"; cp "${CD}/venv-mlx/freeze.txt" "${C}/ref/pip-freeze-venv-mlx.txt"
  echo "torch==2.14.0" > "${C}/venv-torch/freeze.txt"; cp "${C}/venv-torch/freeze.txt" "${C}/ref/pip-freeze-venv-torch.txt"
}
# runo <mode> [NAME=VALUE ...]: 直前の mkcase の環境で orchestrate.sh を走らせ、OUT_TXT・RC を設定する
runo() {
  local mode=$1; shift
  OUT_TXT="$(env -i PATH="${MB}:${PATH}" HOME="${C}/home" MOCK_DIR="${C}/mock" D="${CD}" VENV_TORCH="${C}/venv-torch" FREEZE_REF_DIR="${C}/ref" "$@" \
    bash "${SD}/orchestrate.sh" "${mode}" "${OWNER_ARG:-${OWN}}" 2>&1)"; RC=$?
}
has() { case "${OUT_TXT}" in *"$1"*) ok ;; *) ng "出力に「$1」が無い（$2）" ;; esac; }
hasnt() { case "${OUT_TXT}" in *"$1"*) ng "出力に「$1」がある（$2）" ;; *) ok ;; esac; }
lock_gone() { [ ! -e "${C}/home/work/.fandhe-hw-lock" ] && ok || ng "ロックが残っている（$1）"; }
lock_kept() { [ -f "${C}/home/work/.fandhe-hw-lock/owner" ] && ok || ng "ロックが消えた（$1）"; }

# ---- 1. series
# (1) 全部 CLEAN・load1 低 → 5/5 完走
mkcase c1 "CLEAN" "1.20"; runo series
expect_rc "series: 5/5 → rc 0" 0 "${RC}"; has "ALL-DONE completed=5/5" "完走"; hasnt "SERIES-ABORTED" "打ち切りなし"; lock_gone "完走後"
for i in 1 2 3 4 5; do
  [ "$(head -n 1 "${CD}/out/round${i}/gate.txt" | cut -d' ' -f1)" = PASS ] && ok || ng "series: round${i}/gate.txt の 1 行目が PASS でない"
  [ -f "${CD}/out/round${i}/mps.exit" ] && ok || ng "series: round${i} のハーネスが走っていない"
done
grep -q 'round5 foreign-after=clean' "${CD}/out/gate.log" && ok || ng "series: gate.log に round5 の後 CLEAN が無い"
grep -q 'fandhe_ai_env=none' "${CD}/env/env_info_before.txt" && ok || ng "series: env guard の記録が env_info_before にない"
for f in pip-freeze-venv-torch-before pip-freeze-venv-mlx-before pip-freeze-venv-torch-after pip-freeze-venv-mlx-after; do
  [ -f "${CD}/env/${f}.txt" ] && ok || ng "series: env/${f}.txt が無い"
done
# 再実行（out/round1 が既にある）→ 始めない
mkdir -p "${C}/home/work/.fandhe-hw-lock"; printf '%s' "${OWN}" > "${C}/home/work/.fandhe-hw-lock/owner"; runo series
expect_rc "series: 再実行は rc 2" 2 "${RC}"; has "out/round1 が既にある" "再実行の拒否"; lock_gone "再実行の拒否後"
# (2) 前ゲートがずっと FOREIGN → 60 回待って GATE-TIMEOUT・打ち切り
mkcase c2 "FOREIGN" "1.20"; runo series
expect_rc "series: 前 TIMEOUT → rc 1" 1 "${RC}"; has "GATE-TIMEOUT before round1" "タイムアウト"; has "SERIES-ABORTED completed=0/5" "打ち切り"
hasnt "ALL-DONE" "完走しない"; [ "$(wc -l < "${C}/mock/sleep.log" | tr -d ' ')" = 60 ] && ok || ng "series: 30 秒 × 60 回待っていない"
[ ! -f "${CD}/out/round1/oss.exit" ] && ok || ng "series: タイムアウトなのにハーネスが走った"
[ "$(cut -d' ' -f1 "${CD}/out/round1/gate.txt")" = TIMEOUT ] && ok || ng "series: gate.txt が TIMEOUT でない"; lock_gone "タイムアウト後"
# (2b) load1 が高いまま → タイムアウト（reason=load1）
mkcase c2b "CLEAN" "5.50"; runo series
expect_rc "series: load1 高 → rc 1" 1 "${RC}"; grep -q 'GATE-TIMEOUT reason=load1' "${CD}/out/gate.log" && ok || ng "series: reason=load1 が無い"
# (3) ゲート自身の失敗 → GATE-INVALID・打ち切り
mkcase c3 "ERROR" "1.20"; runo series
expect_rc "series: ゲート失敗 → rc 1" 1 "${RC}"; has "GATE-INVALID before round1" "ゲート失敗"; has "SERIES-ABORTED completed=0/5" "打ち切り"
[ ! -f "${CD}/out/round1/oss.exit" ] && ok || ng "series: ゲート失敗なのにハーネスが走った"; lock_gone "ゲート失敗後"
# (4) round2 の後が FOREIGN（ゲート呼び出し: 1 前 2 後 3 前 4 後）→ 打ち切り・round3 は始まらない
mkcase c4 "CLEAN
CLEAN
CLEAN
FOREIGN" "1.20"; runo series
expect_rc "series: round2 の後 FOREIGN → rc 1" 1 "${RC}"; has "SERIES-ABORTED completed=1/5 (round2 は無効: foreign-after)" "後 FOREIGN の打ち切り"
hasnt "ALL-DONE" "完走しない"; [ ! -d "${CD}/out/round3" ] && ok || ng "series: 打ち切り後に round3 が始まった"
grep -q 'round2 GATE-INVALID reason=foreign-after' "${CD}/out/gate.log" && ok || ng "series: gate.log に foreign-after が無い"
grep -q '^INVALID-AFTER' "${CD}/out/round2/gate.txt" && ok || ng "series: round2/gate.txt に無効の印が無い"; lock_gone "後 FOREIGN 後"
# (4b) round1 の後のゲート失敗 → 打ち切り
mkcase c4b "CLEAN
ERROR" "1.20"; runo series
expect_rc "series: 後のゲート失敗 → rc 1" 1 "${RC}"; has "SERIES-ABORTED completed=0/5 (round1 は無効: foreign-gate-error)" "後のゲート失敗"
# (5) env guard → 始めない
for spec in FANDHE_AI_SME=0 RAYON_NUM_THREADS=4 FANDHE_AI_X=; do
  mkcase "c5-${spec%%=*}" "CLEAN" "1.20"; runo series "${spec}"
  expect_rc "series: ${spec%%=*} が設定されていれば rc 3" 3 "${RC}"; has "ENV-GUARD FAILED" "env guard"
  [ ! -d "${CD}/out/round1" ] && ok || ng "series: env guard で止まらず round1 を作った"; lock_gone "env guard 後"
done
# (6) freeze 不一致（開始時）→ 始めない
mkcase c6 "CLEAN" "1.20"; echo "mlx==0.32.1" > "${CD}/venv-mlx/freeze.txt"; runo series
expect_rc "series: venv-mlx の freeze 不一致 → rc 1" 1 "${RC}"; has "FREEZE-MISMATCH venv-mlx" "mlx の不一致"; has "FREEZE CHECK FAILED" "始めない"
[ ! -d "${CD}/out/round1" ] && ok || ng "series: freeze 不一致で round1 を作った"
mkcase c6b "CLEAN" "1.20"; echo "torch==2.13.0" > "${C}/venv-torch/freeze.txt"; runo series
expect_rc "series: venv-torch の freeze 不一致 → rc 1" 1 "${RC}"; has "FREEZE-MISMATCH venv-torch" "torch の不一致"
# (6c) freeze の取得失敗（uv の失敗）→ 始めない
mkcase c6c "CLEAN" "1.20"; rm -f "${C}/venv-torch/freeze.txt"; runo series
expect_rc "series: freeze が取れなければ rc 1" 1 "${RC}"; has "FREEZE-FAILED venv-torch" "取得失敗"
# (6d) 終了時の freeze 不一致 → ALL-DONE を出さず非 0（5 ラウンド目の後に venv を差し替えるハーネスのモック）
mkcase c6d "CLEAN" "1.20"
printf '#!/bin/sh\necho mock-bench\n[ -f "%s/out/round5/oss.exit" ] && echo "torch==2.99.0" > "%s/freeze.txt"\nexit 0\n' "${CD}" "${C}/venv-torch" > "${CD}/target-metal/release/examples/gemm_f32_prepared_bench"
runo series
expect_rc "series: 終了時 freeze 不一致 → rc 1" 1 "${RC}"; has "SERIES-INVALID completed=5/5 reason=freeze-after" "終了時の不一致"; hasnt "ALL-DONE" "完走扱いにしない"
# (7) ロック owner 不一致 → 始めない・他人のロックに触らない
mkcase c7 "CLEAN" "1.20"; OWNER_ARG="#4314 engine-0150-macmini 2026-02-02T00:00:00Z" runo series
expect_rc "series: owner 不一致 → rc 2" 2 "${RC}"; has "LOCK-NOT-MINE" "owner 不一致"; lock_kept "owner 不一致"
[ ! -d "${CD}/out" ] && ok || ng "series: owner 不一致で out を作った"
# (7b) ロックが無い → 始めない
mkcase c7b "CLEAN" "1.20"; rm -rf "${C}/home/work/.fandhe-hw-lock"; runo series
expect_rc "series: ロックなし → rc 2" 2 "${RC}"; has "LOCK-NOT-MINE" "ロックなし"
# (7c) 相対パス → 始めない
mkcase c7c "CLEAN" "1.20"
OUT_TXT="$(env -i PATH="${MB}:${PATH}" HOME="${C}/home" MOCK_DIR="${C}/mock" D="${CD}" VENV_TORCH="rel/venv" FREEZE_REF_DIR="${C}/ref" bash "${SD}/orchestrate.sh" series "${OWN}" 2>&1)"; RC=$?
expect_rc "series: VENV_TORCH が相対パス → rc 2" 2 "${RC}"
# (7d) ハーネスの実行ファイルが無い → 始めない・ロックは外す
mkcase c7d "CLEAN" "1.20"; rm -f "${CD}/target-oss/release/oss-gemm-compare"; runo series
expect_rc "series: ハーネスが無い → rc 2" 2 "${RC}"; has "missing" "ハーネスなし"; lock_gone "ハーネスなし"
# (7e) ゲートの 2 ファイルの sha256 不一致 → 始めない
mkcase c7e "CLEAN" "1.20"; echo "# x" >> "${CD}/tree/${GATE_REL}"; runo series
expect_rc "series: ゲートの sha256 不一致 → rc 1" 1 "${RC}"; has "FOREIGN-GATE PRECHECK FAILED" "sha 不一致"
[ ! -d "${CD}/out/round1" ] && ok || ng "series: sha 不一致で round1 を作った"
# (8) ハーネスの非 0 終了は記録のみ（判定は aggregate.py。前回と同じ）
mkcase c8 "CLEAN" "1.20"; printf '#!/bin/sh\nexit 1\n' > "${CD}/target-oss/release/oss-gemm-compare"; runo series
expect_rc "series: oss の非 0 終了は記録のみ → rc 0" 0 "${RC}"; grep -q 'exit=1' "${CD}/out/round1/oss.exit" && ok || ng "series: oss.exit に exit=1 が無い"

# ---- 2. prepare: freeze 不一致ならビルドしない
mkcase p1 "CLEAN" "1.20"; rm -rf "${CD}/target-oss" "${CD}/target-metal"; echo "numpy==1.0" > "${CD}/venv-mlx/freeze.txt"; runo prepare
expect_rc "prepare: freeze 不一致 → rc 1" 1 "${RC}"; [ ! -f "${C}/mock/cargo.log" ] && ok || ng "prepare: freeze 不一致なのに cargo を呼んだ"; lock_kept "prepare はロックを外さない"
mkcase p2 "CLEAN" "1.20"; runo prepare
expect_rc "prepare: target が既にあれば rc 2" 2 "${RC}"

# ---- 3. 同梱のコピーと aggregate.py
cmpf() { cmp -s "$1" "$2" && ok || ng "byte 一致しない: ${1#"${REPO_ROOT}"/} と ${2#"${REPO_ROOT}"/}"; }
P177="${REPO_ROOT}/docs/perf/logs/framework-compare-0.15.0-macmini-remeasure/scripts"
cmpf "${SD}/foreign_gate_wrapper.sh" "${P177}/foreign_gate_wrapper.sh"
cmpf "${SD}/env_guard.sh" "${P177}/env_guard.sh"
cmpf "${SD}/aggregate.py" "${REPO_ROOT}/docs/perf/oss-comparison/2026-11-08-macmini/scripts/aggregate.py"
if command -v python3 > /dev/null 2>&1; then
  python3 -I "${SD}/aggregate.py" --self-test "${REPO_ROOT}" > "${T}/agg.txt" 2>&1 && grep -q '^self-test ok$' "${T}/agg.txt" && ok || ng "aggregate.py --self-test が通らない"
fi
for f in orchestrate.sh self-test.sh foreign_gate_wrapper.sh env_guard.sh; do bash -n "${SD}/${f}" && ok || ng "bash -n ${f}"; done

echo "self-test: ${N} 件中 FAIL ${BAD} 件"
[ "${BAD}" -eq 0 ]
