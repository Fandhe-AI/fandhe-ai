#!/usr/bin/env bash
#
# CUDA toolkit「非搭載」であることを fail-closed で検査する単一ソース
# （TASK-1.7d・docs/spec/05-tasks.md、イシュー #35）。
#
# 背景: backend-cuda は cudarc の dynamic-loading feature による無条件依存＋動的ロード方式
# （PoC-v2-3・PoC-v2-5）を採用しており、「CUDA toolkit 非搭載環境でもビルドは成立し、
# 実行時のみ toolkit を要求する」契約である（crates/backend-cuda/src/lib.rs 冒頭コメント）。
# この契約は runner に toolkit が導入されると暗黙に検証されなくなるため、本スクリプトで
# 「非搭載であること自体」を機械検査し、CI の build-no-cuda-toolkit ジョブで assert する。
#
# 呼び出し元:
#   - .github/workflows/ci.yml の build-no-cuda-toolkit ジョブ（self-test → assert の順で呼ぶ）
#   - Makefile の build-no-cuda ターゲット（CI と同一判定をローカル再現。probe でスキップ分岐）
#
# 判定対象は toolkit の構成物に限定する（ドライバ libcuda.so は対象外。TASK-1.7d の主旨は
# 「toolkit 非搭載」の検証であり、ドライバの有無は問わない）:
#   - nvcc（PATH 上のコンパイラ本体）
#   - toolkit 標準インストールディレクトリ（/usr/local/cuda*・/opt/cuda*）
#   - ldconfig 登録済み共有ライブラリ（libnvrtc.so・libcudart.so）
#   - 環境変数 CUDA_HOME・CUDA_PATH が実在ディレクトリを指す
#
# サブコマンド:
#   probe      非搭載なら exit 0・搭載なら exit 1（メッセージなし。Makefile のスキップ分岐用）
#   assert     搭載を検出したら検出内容を明示して exit 1（CI 用 fail-closed）
#   self-test  検査ロジック自体の退行検出。偽 nvcc・偽 ldconfig をスタブ注入し「搭載」判定に
#              なること／全注入変数を「存在しない」値に固定し「非搭載」判定になることを検証する
#              （実環境の toolkit 有無に依存しないヘルメティックな self-test）
#
# 環境変数（self-test 用の注入ポイント。通常運用では既定値のまま使う）:
#   NVCC_BIN         nvcc コマンド名／パス（既定: nvcc）
#   LDCONFIG_BIN      ldconfig コマンド名／パス（既定: ldconfig）
#   CUDA_ROOT_GLOBS  toolkit 標準インストールディレクトリの glob（空白区切り。既定は下記）
set -euo pipefail

NVCC_BIN="${NVCC_BIN:-nvcc}"
LDCONFIG_BIN="${LDCONFIG_BIN:-ldconfig}"
CUDA_ROOT_GLOBS="${CUDA_ROOT_GLOBS:-/usr/local/cuda* /opt/cuda*}"

usage() {
  echo "usage: $0 {probe|assert|self-test}" >&2
  exit 2
}

# toolkit 構成物の検出結果を改行区切りで標準出力へ書き出す（空なら非搭載）。
# probe / assert / self-test の全モードから共通で呼ぶことで検出ロジックを一元化する。
detect_toolkit() {
  local found=""

  if command -v "${NVCC_BIN}" >/dev/null 2>&1; then
    found="${found}nvcc: $(command -v "${NVCC_BIN}")"$'\n'
  fi

  # shellcheck disable=SC2086 # CUDA_ROOT_GLOBS は空白区切りの glob 列として展開する意図
  for root in ${CUDA_ROOT_GLOBS}; do
    if [ -d "${root}" ]; then
      found="${found}toolkit ディレクトリ: ${root}"$'\n'
    fi
  done

  if command -v "${LDCONFIG_BIN}" >/dev/null 2>&1; then
    local ldconfig_out
    ldconfig_out=$("${LDCONFIG_BIN}" -p 2>/dev/null || true)
    if echo "${ldconfig_out}" | grep -q 'libnvrtc\.so'; then
      found="${found}ldconfig: libnvrtc.so 登録あり"$'\n'
    fi
    if echo "${ldconfig_out}" | grep -q 'libcudart\.so'; then
      found="${found}ldconfig: libcudart.so 登録あり"$'\n'
    fi
  fi

  if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}" ]; then
    found="${found}CUDA_HOME: ${CUDA_HOME}"$'\n'
  fi
  if [ -n "${CUDA_PATH:-}" ] && [ -d "${CUDA_PATH}" ]; then
    found="${found}CUDA_PATH: ${CUDA_PATH}"$'\n'
  fi

  # 情報表示のみ（判定には使わない）。libcuda.so はドライバであり toolkit ではないため
  # 「非搭載」判定を左右してはならない（TASK-1.7d の主旨に判定を限定する）。
  if command -v "${LDCONFIG_BIN}" >/dev/null 2>&1; then
    if "${LDCONFIG_BIN}" -p 2>/dev/null | grep -q 'libcuda\.so'; then
      echo "情報: libcuda.so（ドライバ）を検出しましたが、toolkit ではないため判定対象外です" >&2
    fi
  fi

  printf '%s' "${found}"
}

cmd_probe() {
  local found
  found=$(detect_toolkit)
  [ -z "${found}" ]
}

cmd_assert() {
  local found
  found=$(detect_toolkit)
  if [ -n "${found}" ]; then
    echo "::error::CUDA toolkit の構成物を検出しました。本ジョブは toolkit 非搭載環境での検証を前提とする。runner へ toolkit を導入した場合はコンテナ内検証への切替を検討すること（TASK-1.7d・イシュー #35）:" >&2
    echo "${found}" >&2
    return 1
  fi
  echo "OK: CUDA toolkit の構成物は検出されませんでした（非搭載を確認）"
}

# 検査ロジック自体の退行（判定条件の破損等）を、実環境の toolkit 有無に依存せず検出する。
self_test() {
  local failed=0
  local tmpdir
  tmpdir=$(mktemp -d)
  # shellcheck disable=SC2064 # tmpdir はこの時点の値で固定して trap に渡す意図
  trap "rm -rf '${tmpdir}'" RETURN

  # (a) 偽 nvcc スタブを注入 → 「搭載」判定になること
  local fake_nvcc="${tmpdir}/nvcc"
  printf '#!/bin/sh\necho fake-nvcc\n' >"${fake_nvcc}"
  chmod +x "${fake_nvcc}"
  if (NVCC_BIN="${fake_nvcc}" LDCONFIG_BIN="/nonexistent-ldconfig" CUDA_ROOT_GLOBS="/nonexistent-cuda-root-*" \
    CUDA_HOME="" CUDA_PATH="" cmd_probe) >/dev/null 2>&1; then
    echo "NG: self-test(a) 偽 nvcc 注入時に非搭載（probe 成功）と判定されました（退行）" >&2
    failed=1
  else
    echo "OK: self-test(a) 偽 nvcc 注入時に搭載と判定されました"
  fi

  # (b) libnvrtc.so を出力するだけの偽 ldconfig スタブを注入 → 「搭載」判定になること
  local fake_ldconfig="${tmpdir}/ldconfig"
  printf '#!/bin/sh\necho "\tlibnvrtc.so.12 (libc6,x86-64) => /fake/libnvrtc.so.12"\n' >"${fake_ldconfig}"
  chmod +x "${fake_ldconfig}"
  if (NVCC_BIN="/nonexistent-nvcc" LDCONFIG_BIN="${fake_ldconfig}" CUDA_ROOT_GLOBS="/nonexistent-cuda-root-*" \
    CUDA_HOME="" CUDA_PATH="" cmd_probe) >/dev/null 2>&1; then
    echo "NG: self-test(b) 偽 ldconfig（libnvrtc.so 登録）注入時に非搭載と判定されました（退行）" >&2
    failed=1
  else
    echo "OK: self-test(b) 偽 ldconfig（libnvrtc.so 登録）注入時に搭載と判定されました"
  fi

  # (c) 全注入変数を「存在しない」値に固定 → 「非搭載」判定になること
  if (NVCC_BIN="/nonexistent-nvcc" LDCONFIG_BIN="/nonexistent-ldconfig" CUDA_ROOT_GLOBS="/nonexistent-cuda-root-*" \
    CUDA_HOME="" CUDA_PATH="" cmd_probe) >/dev/null 2>&1; then
    echo "OK: self-test(c) 全注入変数「存在しない」時に非搭載と判定されました"
  else
    echo "NG: self-test(c) 全注入変数「存在しない」時に搭載と誤判定されました（退行）" >&2
    failed=1
  fi

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
