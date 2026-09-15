#!/usr/bin/env bash
# イシュー #1771（PR #1882 レビュー指摘）: 実機ランブックの run-to-run
# 決定性チェック（README「事前登録判定規則」4)）は各 `#[ignore]`
# テストの `--nocapture` 出力から `bits=`／`fold_bits=` 行を `grep`
# で抽出し、2 回起動の出力を `diff` して run-to-run bit 同一を確認
# する設計である。対象テストに 1 行も `bits=`／`fold_bits=` 出力が
# 無いログ（抽出漏れ・テスト未実行・空ログ）の場合、空の抽出結果
# 同士が自明に diff 一致してしまい起動間の変化を検出できない。
#
# 本スクリプトは diff の前に次を検査してから diff する:
#   (a) 各ログの抽出行数が 0 でないこと（期待するケース行が実際に
#       出力されていることの確認）
#   (b) 対応する 2 ログ間で抽出行数が一致すること
# いずれかが不成立なら FAIL とし diff 自体を行わない（空同士の
# 自明一致を PASS と誤認しないため）。
#
# 使い方: `run_ignored_tests_cuda.sh`／`run_ignored_tests_metal.sh`
# を出力先を変えて 2 回実行し（例: 1 回目の実行後に
# `cuda/ignored` を `cuda/ignored-run1` へ退避してから 2 回目を実行
# し `cuda/ignored-run2` へ退避する）、対応する `*.log` ファイル群を
# 持つ 2 ディレクトリを渡す:
#
#   check_determinism.sh <run1_dir> <run2_dir>
#
# 自己検証（GPU 実機不要。ロジック自体の検証のみ）:
#
#   check_determinism.sh --self-test
set -u

extract() {
  # bits=/fold_bits= 行を抽出する（ファイルが存在しない・grep が
  # 何も見つけない場合も非ゼロ終了で落とさない）。
  grep -E 'bits=|fold_bits=' "$1" 2>/dev/null
  return 0
}

count_nonblank() {
  # `extract` の出力（空文字列を含みうる）の非空行数を数える。
  # `grep -c .` は 0 件時に非ゼロ終了するため `|| true` で吸収する。
  printf '%s\n' "$1" | grep -c . || true
}

compare_dirs() {
  local run1=$1
  local run2=$2
  local overall=0
  local found_any=0

  shopt -s nullglob
  local log1 name log2 lines1 lines2 count1 count2
  for log1 in "$run1"/*.log; do
    found_any=1
    name=$(basename "$log1")
    log2="$run2/$name"
    if [[ ! -f "$log2" ]]; then
      echo "FAIL: $name が run2 側（$run2）に存在しない" >&2
      overall=1
      continue
    fi

    lines1="$(extract "$log1")"
    lines2="$(extract "$log2")"
    count1=$(count_nonblank "$lines1")
    count2=$(count_nonblank "$lines2")

    if [[ "$count1" -eq 0 || "$count2" -eq 0 ]]; then
      echo "FAIL: $name の bits=/fold_bits= 抽出行が 0 件（run1=$count1 run2=$count2）——" \
        "空の抽出結果同士は自明に一致するため diff は行わない（PR #1882 レビュー指摘）" >&2
      overall=1
      continue
    fi
    if [[ "$count1" -ne "$count2" ]]; then
      echo "FAIL: $name の抽出行数が run1/run2 で不一致（run1=$count1 run2=$count2）" >&2
      overall=1
      continue
    fi

    if diff <(printf '%s\n' "$lines1") <(printf '%s\n' "$lines2") >/dev/null; then
      echo "PASS: $name run-to-run bit 同一（${count1} 行）"
    else
      echo "FAIL: $name run-to-run で差分あり（下記 diff 参照）" >&2
      diff <(printf '%s\n' "$lines1") <(printf '%s\n' "$lines2") >&2 || true
      overall=1
    fi
  done

  if [[ "$found_any" -eq 0 ]]; then
    echo "FAIL: $run1 に *.log が 1 件も無い" >&2
    overall=1
  fi

  return "$overall"
}

self_test() {
  local tmp
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' RETURN

  # ケース 1: 正常系（両ログとも同じ 2 行）——PASS になるはず。
  mkdir -p "$tmp/case1/run1" "$tmp/case1/run2"
  printf 'case[a].bits=0x1\ncase[b].fold_bits=0x2\n' >"$tmp/case1/run1/x.log"
  printf 'case[a].bits=0x1\ncase[b].fold_bits=0x2\n' >"$tmp/case1/run2/x.log"
  if compare_dirs "$tmp/case1/run1" "$tmp/case1/run2" >/dev/null 2>&1; then
    echo "self-test 1/4 OK（正常系 PASS）"
  else
    echo "self-test 1/4 NG（正常系が FAIL 判定になった）" >&2
    return 1
  fi

  # ケース 2: 空ログ同士（抽出行 0 件）——本スクリプトの主目的:
  # 空同士の自明一致を PASS にしてはならない（レビュー指摘の再現）。
  mkdir -p "$tmp/case2/run1" "$tmp/case2/run2"
  : >"$tmp/case2/run1/x.log"
  : >"$tmp/case2/run2/x.log"
  if compare_dirs "$tmp/case2/run1" "$tmp/case2/run2" >/dev/null 2>&1; then
    echo "self-test 2/4 NG（空ログ同士が PASS 判定になった。レビュー指摘の再発）" >&2
    return 1
  else
    echo "self-test 2/4 OK（空ログ同士は FAIL 判定）"
  fi

  # ケース 3: 行数不一致（片方だけ行が欠落）——FAIL になるはず。
  mkdir -p "$tmp/case3/run1" "$tmp/case3/run2"
  printf 'case[a].bits=0x1\ncase[b].fold_bits=0x2\n' >"$tmp/case3/run1/x.log"
  printf 'case[a].bits=0x1\n' >"$tmp/case3/run2/x.log"
  if compare_dirs "$tmp/case3/run1" "$tmp/case3/run2" >/dev/null 2>&1; then
    echo "self-test 3/4 NG（行数不一致が PASS 判定になった）" >&2
    return 1
  else
    echo "self-test 3/4 OK（行数不一致は FAIL 判定）"
  fi

  # ケース 4: 実際に値が異なる（run-to-run 非決定性の検出）——FAIL。
  mkdir -p "$tmp/case4/run1" "$tmp/case4/run2"
  printf 'case[a].bits=0x1\n' >"$tmp/case4/run1/x.log"
  printf 'case[a].bits=0x9\n' >"$tmp/case4/run2/x.log"
  if compare_dirs "$tmp/case4/run1" "$tmp/case4/run2" >/dev/null 2>&1; then
    echo "self-test 4/4 NG（値の相違が PASS 判定になった）" >&2
    return 1
  else
    echo "self-test 4/4 OK（値の相違は FAIL 判定）"
  fi

  echo "self-test: 全 4 ケース OK"
  return 0
}

main() {
  if [[ "${1:-}" == "--self-test" ]]; then
    self_test
    exit $?
  fi
  local run1=${1:-}
  local run2=${2:-}
  if [[ -z "$run1" || -z "$run2" ]]; then
    echo "usage: $0 <run1_dir> <run2_dir>" >&2
    echo "       $0 --self-test" >&2
    exit 2
  fi
  if [[ ! -d "$run1" || ! -d "$run2" ]]; then
    echo "FAIL: ディレクトリが存在しない（run1=$run1 run2=$run2）" >&2
    exit 1
  fi
  if compare_dirs "$run1" "$run2"; then
    echo "PASS: 全ログ群で run-to-run 決定性チェック成立"
    exit 0
  else
    echo "FAIL: 1 件以上のログ群で run-to-run 決定性チェック不成立" >&2
    exit 1
  fi
}

main "$@"
