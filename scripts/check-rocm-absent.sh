#!/usr/bin/env bash
#
# ROCm「非搭載」であることを fail-closed で検査する単一ソース（イシュー #3154）。
#
# 背景: backend-rocm は libloading による動的ロード（ビルド時リンクなし）を採用しており、
# 「ROCm 非搭載環境でもビルド・テストが成立する」契約である
# （docs/backend-rocm-hip-ffi-decision.md §1・§4）。runner に ROCm が導入されると
# この契約の検証が暗黙に消えるため、本スクリプトで非搭載自体を機械検査する。
# scripts/check-cuda-toolkit-absent.sh と同型（同一方針）。
#
# 呼び出し元:
#   - .github/workflows/ci.yml の build-no-cuda-toolkit ジョブ（self-test → assert）
#   - scripts/build-no-cuda.sh（make build-no-cuda。probe でスキップ分岐）
#
# 判定対象（ROCm の構成物）:
#   - PATH 上の hipcc・hipconfig・rocminfo
#   - 標準インストールディレクトリ（/opt/rocm*）
#   - ldconfig 登録済み共有ライブラリ（libamdhip64.so・libhiprtc.so）
#   - 環境変数 ROCM_PATH・HIP_PATH・ROCM_HOME が実在ディレクトリを指す
#
# サブコマンド:
#   probe      非搭載なら exit 0・搭載なら exit 1（メッセージなし）
#   assert     搭載を検出したら検出内容を明示して exit 1（CI 用 fail-closed）
#   self-test  検査ロジックの退行検出（実環境に依存しないヘルメティックな検証）
#
# 環境変数（self-test 用の注入ポイント。通常運用では既定値のまま使う）:
#   ROCM_BINS        検出するコマンド名（空白区切り。既定: hipcc hipconfig rocminfo）
#   LDCONFIG_BIN     ldconfig コマンド名／パス（既定: ldconfig）
#   ROCM_ROOT_GLOBS  標準インストールディレクトリの glob（空白区切り。既定: /opt/rocm*）
set -euo pipefail

ROCM_BINS="${ROCM_BINS:-hipcc hipconfig rocminfo}"
LDCONFIG_BIN="${LDCONFIG_BIN:-ldconfig}"
ROCM_ROOT_GLOBS="${ROCM_ROOT_GLOBS:-/opt/rocm*}"

usage() {
  echo "usage: $0 {probe|assert|self-test}" >&2
  exit 2
}

# ROCm 構成物の検出結果を改行区切りで標準出力へ書き出す（空なら非搭載）。
detect_rocm() {
  local found=""
  local bin root

  # shellcheck disable=SC2086 # ROCM_BINS は空白区切りの列として展開する意図
  for bin in ${ROCM_BINS}; do
    if command -v "${bin}" >/dev/null 2>&1; then
      found="${found}${bin}: $(command -v "${bin}")"$'\n'
    fi
  done

  # shellcheck disable=SC2086 # ROCM_ROOT_GLOBS は空白区切りの glob 列として展開する意図
  for root in ${ROCM_ROOT_GLOBS}; do
    if [ -d "${root}" ]; then
      found="${found}ROCm ディレクトリ: ${root}"$'\n'
    fi
  done

  if command -v "${LDCONFIG_BIN}" >/dev/null 2>&1; then
    local ldconfig_out
    ldconfig_out=$("${LDCONFIG_BIN}" -p 2>/dev/null || true)
    # here-string で grep に渡す（`echo | grep -q` は大量出力時に SIGPIPE で pipefail 下の
    # 誤判定を起こしうる。check-cuda-toolkit-absent.sh と同じ理由）。
    if grep -q 'libamdhip64\.so' <<<"${ldconfig_out}"; then
      found="${found}ldconfig: libamdhip64.so 登録あり"$'\n'
    fi
    if grep -q 'libhiprtc\.so' <<<"${ldconfig_out}"; then
      found="${found}ldconfig: libhiprtc.so 登録あり"$'\n'
    fi
  fi

  local var
  for var in ROCM_PATH HIP_PATH ROCM_HOME; do
    local val="${!var:-}"
    if [ -n "${val}" ] && [ -d "${val}" ]; then
      found="${found}${var}: ${val}"$'\n'
    fi
  done

  printf '%s' "${found}"
}

cmd_probe() {
  local found
  found=$(detect_rocm)
  [ -z "${found}" ]
}

cmd_assert() {
  local found
  found=$(detect_rocm)
  if [ -n "${found}" ]; then
    echo "::error::ROCm の構成物を検出しました。本ジョブは ROCm 非搭載環境での検証を前提とする（イシュー #3154）:" >&2
    echo "${found}" >&2
    return 1
  fi
  echo "OK: ROCm の構成物は検出されませんでした（非搭載を確認）"
}

# 検査ロジックの退行を検出する。引数: 説明・期待（present|absent）・注入環境（env の引数列）。
# 全注入変数を既定で「存在しない」値に固定し、1 分岐だけを有効化して単独検証する。
expect_probe() {
  local desc="$1" expect="$2"
  shift 2
  local rc=0
  (
    export ROCM_BINS="nonexistent-rocm-bin" LDCONFIG_BIN="/nonexistent-ldconfig" \
      ROCM_ROOT_GLOBS="/nonexistent-rocm-root-*" ROCM_PATH="" HIP_PATH="" ROCM_HOME=""
    local kv
    for kv in "$@"; do declare -x "${kv}"; done
    cmd_probe
  ) >/dev/null 2>&1 || rc=$?
  # probe は非搭載なら 0。
  if [ "${expect}" = "present" ] && [ "${rc}" -eq 0 ]; then
    echo "NG: self-test ${desc}: 非搭載と判定されました（退行）" >&2
    return 1
  fi
  if [ "${expect}" = "absent" ] && [ "${rc}" -ne 0 ]; then
    echo "NG: self-test ${desc}: 搭載と誤判定されました（退行）" >&2
    return 1
  fi
  echo "OK: self-test ${desc}"
}

self_test() {
  local failed=0
  local tmpdir
  tmpdir=$(mktemp -d)
  # shellcheck disable=SC2064 # tmpdir はこの時点の値で固定して trap に渡す意図
  trap "rm -rf '${tmpdir}'" RETURN

  # (a) 偽 hipcc を PATH 上のコマンドとして注入
  local fake_bin="${tmpdir}/fake-hipcc"
  printf '#!/bin/sh\necho fake\n' >"${fake_bin}"
  chmod +x "${fake_bin}"
  expect_probe "(a) 偽 hipcc 注入で搭載判定" present "ROCM_BINS=${fake_bin}" || failed=1

  # (b) libamdhip64.so のみの偽 ldconfig
  local ld_hip="${tmpdir}/ldconfig-hip"
  printf '#!/bin/sh\necho "\tlibamdhip64.so.6 (libc6,x86-64) => /fake/libamdhip64.so.6"\n' >"${ld_hip}"
  chmod +x "${ld_hip}"
  expect_probe "(b) libamdhip64.so 登録で搭載判定" present "LDCONFIG_BIN=${ld_hip}" || failed=1

  # (b2) libhiprtc.so のみの偽 ldconfig
  local ld_rtc="${tmpdir}/ldconfig-rtc"
  printf '#!/bin/sh\necho "\tlibhiprtc.so.6 (libc6,x86-64) => /fake/libhiprtc.so.6"\n' >"${ld_rtc}"
  chmod +x "${ld_rtc}"
  expect_probe "(b2) libhiprtc.so 登録で搭載判定" present "LDCONFIG_BIN=${ld_rtc}" || failed=1

  # (c) 全注入変数を「存在しない」値にすると非搭載判定
  expect_probe "(c) 全注入変数が存在しないとき非搭載判定" absent "ROCM_PATH=" || failed=1

  # (d) 先頭行でマッチし 64KiB 超の出力を続ける偽 ldconfig（pipefail 退行の検出）
  local ld_large="${tmpdir}/ldconfig-large"
  {
    printf '#!/bin/sh\n'
    printf 'echo "\tlibamdhip64.so.6 (libc6,x86-64) => /fake/libamdhip64.so.6"\n'
    printf 'i=0\n'
    # shellcheck disable=SC2016 # 生成先スクリプト内で展開させたい変数参照のため意図的にシングルクォート
    printf 'while [ "$i" -lt 4000 ]; do\n'
    # shellcheck disable=SC2016 # 同上
    printf '  echo "\tlibfiller%%d.so.1 (libc6,x86-64) => /fake/libfiller%%d.so.1" "$i" "$i"\n'
    # shellcheck disable=SC2016 # 同上
    printf '  i=$((i + 1))\n'
    printf 'done\n'
  } >"${ld_large}"
  chmod +x "${ld_large}"
  expect_probe "(d) 大量出力＋先頭マッチで搭載判定" present "LDCONFIG_BIN=${ld_large}" || failed=1

  # (e) 実在ディレクトリを ROCM_ROOT_GLOBS へ注入
  local fake_root="${tmpdir}/rocm-root"
  mkdir -p "${fake_root}"
  expect_probe "(e) ROCM_ROOT_GLOBS の実在ディレクトリで搭載判定" present "ROCM_ROOT_GLOBS=${fake_root}" || failed=1

  # (f) 実在ディレクトリを環境変数へ注入（3 変数それぞれ単独）
  local var
  for var in ROCM_PATH HIP_PATH ROCM_HOME; do
    expect_probe "(f) ${var} の実在ディレクトリで搭載判定" present "${var}=${fake_root}" || failed=1
  done

  # (g) 存在しないディレクトリを指す環境変数は無視される
  expect_probe "(g) 存在しないディレクトリを指す ROCM_PATH は非搭載判定" absent "ROCM_PATH=${tmpdir}/none" || failed=1

  if [ "${failed}" -ne 0 ]; then
    echo "NG: self-test に失敗しました" >&2
    return 1
  fi
  echo "OK: self-test すべて成功"
}

case "${1:-}" in
probe)
  cmd_probe
  ;;
assert)
  cmd_assert
  ;;
self-test)
  self_test
  ;;
*)
  usage
  ;;
esac
