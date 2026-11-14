#!/usr/bin/env bash
# 0.15.0 同席計測（#4177 Mac mini・#4178 GB10）: 比較相手の venv が 0.14.0 と同じ中身かを計測前に確かめる事前検査（両機体で同一内容）。
# 役割: Python の比較相手（PyTorch・TensorFlow・SciPy）の版が 0.14.0 の計測と違えば、0.14.0 → 0.15.0 の差に相手側の差が混ざる。
#   venv の freeze を取り、0.14.0 の収録 freeze と byte 一致するか見る。不一致・不在・取得失敗は非 0 で止める（作り直しは別の承認事項）。
# freeze の形式（0.14.0 と同じ）:
#   macmini: `uv pip freeze --python <venv>/bin/python`（env/pip-freeze-venv-{torch,tf}.txt。python の版行なし）
#   gb10   : `<venv>/bin/python -m pip freeze` の直後に `<venv>/bin/python --version`（env/pip-freeze-venv-bench.txt。末尾が `Python 3.12.3`）
# 使い方（直接実行。計測開始の前に機体で 1 回、系列の起動時にも呼ばれる）:
#   bash venv_preflight.sh macmini <PY_VENV_DIR>   # PY_VENV_DIR 直下の venv-torch・venv-tf を検査
#   bash venv_preflight.sh gb10 <venv ディレクトリ>   # 例: ~/work/.venv-bench
#   環境変数 VENV_TORCH・VENV_TF（macmini のみ・任意）: venv-torch・venv-tf の場所を個別に指定する（既定は <PY_VENV_DIR>/venv-torch・venv-tf。
#     #4168 は 2 つの python を別々に渡していたため、`venv4168.noindex` の配置が違っても指定できるようにしてある）
#   環境変数 FREEZE_REF_DIR: 0.14.0 の収録 env/ の場所（既定は本スクリプトの位置から見たリポジトリ内の収録。
#     スクリプトだけを機体へ平置きした場合は指定が必要。無ければ止まる）
#   環境変数 VP_KEEP_DIR: 取得した freeze を残すディレクトリ（任意。collect.sh が使う）
# 呼び出し元: macmini-0150-series.sh・dgx-0150-loop.sh・collect.sh・self-test.sh。
# bash 3.2 互換。

vp_dir() { cd "$(dirname "${BASH_SOURCE[0]}")" && pwd; }

# vp_freeze <macmini|gb10> <venv-dir> <出力ファイル>: freeze を 0.14.0 と同じ形式で書く。失敗は非 0。
vp_freeze() {
  case "$1" in
    macmini)
      command -v uv > /dev/null 2>&1 || { echo "venv-preflight: uv が無い" >&2; return 1; }
      [ -x "$2/bin/python" ] || { echo "venv-preflight: $2/bin/python が無い" >&2; return 1; }
      uv pip freeze --python "$2/bin/python" > "$3" || return 1 ;;
    gb10)
      [ -x "$2/bin/python" ] || { echo "venv-preflight: $2/bin/python が無い" >&2; return 1; }
      { "$2/bin/python" -m pip freeze --disable-pip-version-check && "$2/bin/python" --version; } > "$3" || return 1 ;;
    *) echo "venv-preflight: profile は macmini か gb10" >&2; return 1 ;;
  esac
  [ -s "$3" ] || { echo "venv-preflight: freeze が空" >&2; return 1; }
}

# vp_ref_dir <profile>: 0.14.0 の収録 env/ の場所。
vp_ref_dir() {
  if [ -n "${FREEZE_REF_DIR:-}" ]; then printf '%s\n' "${FREEZE_REF_DIR}"; return 0; fi
  case "$1" in
    macmini) printf '%s\n' "$(vp_dir)/../../framework-compare-0.14.0-macmini-remeasure/env" ;;
    gb10) printf '%s\n' "$(vp_dir)/../../framework-compare-0.14.0-remeasure/env" ;;
  esac
}

# vp_check_one <profile> <venv-dir> <名前> <参照ファイル>: 取得して cmp。0 = byte 一致。
vp_check_one() {
  local got rc=0
  got="$(mktemp)" || return 1
  if ! vp_freeze "$1" "$2" "${got}"; then rm -f "${got}"; echo "venv-preflight: $3 freeze を取得できない" >&2; return 1; fi
  if [ ! -f "$4" ]; then rm -f "${got}"; echo "venv-preflight: 参照 freeze が無い（FREEZE_REF_DIR を確かめる）: $(basename "$4")" >&2; return 1; fi
  if cmp -s "${got}" "$4"; then
    echo "venv-preflight: $3 identical（0.14.0 の収録 freeze と byte 一致）"
  else
    echo "venv-preflight: $3 DIFFERENT（0.14.0 の収録 freeze と不一致。測らずに止まる。差分の先頭 20 行）" >&2
    diff "${got}" "$4" 2>&1 | head -n 20 >&2
    rc=1
  fi
  if [ -n "${VP_KEEP_DIR:-}" ]; then cp "${got}" "${VP_KEEP_DIR}/pip-freeze-venv-$3.txt" || rc=1; fi
  rm -f "${got}"
  return "${rc}"
}

# vp_check <profile> <venv の親（macmini）か venv 本体（gb10）>: 全件一致で 0。
vp_check() {
  local ref bad=0
  ref="$(vp_ref_dir "$1")"
  case "$1" in
    macmini)
      vp_check_one macmini "${VENV_TORCH:-$2/venv-torch}" torch "${ref}/pip-freeze-venv-torch.txt" || bad=1
      vp_check_one macmini "${VENV_TF:-$2/venv-tf}" tf "${ref}/pip-freeze-venv-tf.txt" || bad=1 ;;
    gb10)
      vp_check_one gb10 "$2" bench "${ref}/pip-freeze-venv-bench.txt" || bad=1 ;;
    *) echo "venv-preflight: profile は macmini か gb10" >&2; return 1 ;;
  esac
  return "${bad}"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  [ $# -eq 2 ] || { echo "使い方: venv_preflight.sh <macmini|gb10> <venv ディレクトリ>" >&2; exit 2; }
  case "$2" in /*) ;; *) echo "venv ディレクトリは絶対パス" >&2; exit 2 ;; esac
  vp_check "$1" "$2"
fi
