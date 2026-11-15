#!/usr/bin/env bash
# 0.15.0 同席計測（#4177 Mac mini・#4178 GB10）: 環境変数の検査（source 専用。両機体で同一内容）。
# 役割: 0.15.0 は SME・Graph・jpw1・STEEL を「既定のまま」測る（既定 ON の効果を測るため、環境変数で既定を変えない）。
#   run の前に `FANDHE_AI_` で始まる変数と `RAYON_NUM_THREADS` が 1 つも設定されていないことを確かめ、
#   あれば非 0 で止める。値は読まず、名前だけを標準エラーへ出す（値は記録しない）。空文字で設定された変数も「設定あり」とみなす。
#   GB10 は #4299 で env=0/1 を使った直後のため、残留を特に疑う。
# 呼び出し元: macmini-run-0150.sh・macmini-0150-series.sh・dgx-run-0150.sh・dgx-0150-loop.sh・self-test.sh。
# 使い方: ENVG="$(env_guard_check)" || exit 3   # 成功時の標準出力が env 記録（2 行）。そのまま env_info に書く。
# 戻り値: 0 = 設定なし／3 = 設定あり（名前を stderr へ）／2 = 検査自体の失敗（fail-closed）。
# 複数行の値の途中に `FANDHE_AI_X=` で始まる行があると誤検出する（止める側に倒れるので許容）。
# bash 3.2 互換。

env_guard_check() {
  local lines g
  lines="$(env | grep -E '^(FANDHE_AI_[^=]*|RAYON_NUM_THREADS)=')"; g=$?
  case "${g}" in
    0)
      echo "env-guard: 設定されている変数（名前のみ）:" >&2
      printf '%s\n' "${lines}" | sed -e 's/=.*//' -e 's/^/  /' >&2
      return 3 ;;
    1) ;;
    *) echo "env-guard: 検査に失敗（grep rc=${g}）" >&2; return 2 ;;
  esac
  printf 'fandhe_ai_env=none\nrayon_num_threads_env=none\n'
}
