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
# **PR #1882 追加レビュー指摘の是正**（2 点）:
#   (1) 旧実装は run1 側に存在する `*.log` のみを走査していたため、
#       run2 側にのみ存在するログ（run1 側で取りこぼした・退避時の
#       操作ミス等）を一度も検査しないまま見逃していた。本版は
#       run1／run2 両ディレクトリの `*.log` の**和集合**を走査する
#       （run2 のみに存在するログは「run1 側に無い」として FAIL）。
#   (2) 旧実装は「run1・run2 のどちらに何のログがあるか」だけを見て
#       おり、README「対象テスト一覧」1)〜2) が定める本来存在すべき
#       4 グループ分のログファイル名を一切知らなかった。このため、
#       ある期待ログが run1・run2 の**両方**から丸ごと欠落した場合
#       （テストバイナリのビルド失敗・実行忘れ・出力先パスの取り違え
#       等）、走査対象の和集合にすら現れず検出できなかった。本版は
#       `--expect-logs <name1,name2,...>`（拡張子なしのログ名を
#       カンマ区切りで列挙。`run_ignored_tests_{cuda,metal}.sh` の
#       `run_case` 呼び出し名と同一の静的な既知集合）を受け取り、
#       列挙した各名前が run1・run2 双方に `<name>.log` として存在し
#       抽出行数が 0 でないことを追加検査する。
#
#   なお、「同一ケースの `print_fold_bits` 呼び出しが run1・run2 の
#   両方で同一に欠落する」場合（既存ログ自体は存在し行数も run1/run2
#   間で一致するが、本来あるべき行が両者から同じだけ欠けている
#   ケース）は、run1 と run2 の相互比較だけでは原理的に検出できない
#   （比較対象となる「期待される行数・ラベル集合」の独立した正が
#   このランブックには無いため）。これは対象がテスト本番コードでは
#   なく診断専用ログ・スクリプトであることを踏まえた既知の残存限界
#   として明記する（`--expect-logs` はログファイル単位の欠落のみを
#   閉じる。ログ内の特定ケース単位の欠落は閉じない）。
#
# 使い方: `run_ignored_tests_cuda.sh`／`run_ignored_tests_metal.sh`
# を出力先を変えて 2 回実行し（例: 1 回目の実行後に
# `cuda/ignored` を `cuda/ignored-run1` へ退避してから 2 回目を実行
# し `cuda/ignored-run2` へ退避する）、対応する `*.log` ファイル群を
# 持つ 2 ディレクトリを渡す:
#
#   check_determinism.sh [--expect-logs <name1,name2,...>] <run1_dir> <run2_dir>
#
# README「対象テスト一覧」1)〜2) の 4 グループ（CUDA／Metal 共通の
# ログ名）を検査対象へ明示的に含めるには例えば:
#
#   check_determinism.sh \
#     --expect-logs im2col_col2im_parity,conv2d_backend_parity,conv1d_backend_parity,nn_conv_backend_parity \
#     cuda/ignored-run1 cuda/ignored-run2
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

# compare_dirs <run1_dir> <run2_dir> [expect_csv]
#
# `expect_csv`（空文字列可）は `--expect-logs` で渡された拡張子なし
# ログ名のカンマ区切り一覧。指定された各名前は、run1／run2 双方の
# `*.log` 走査結果に含まれていなくても検査対象へ強制的に含める
# （PR #1882 追加レビュー指摘 (2) の是正）。
compare_dirs() {
  local run1=$1
  local run2=$2
  local expect_csv=${3:-}
  local overall=0
  local found_any=0

  # 走査対象名の集合を構築する: run1／run2 に実在する *.log の
  # basename（拡張子なし）の和集合 ∪ --expect-logs で明示された名前
  # （PR #1882 追加レビュー指摘 (1)(2) の是正）。連想配列で重複排除
  # する（bash 4+ 前提。self-repair 検証環境・GB10・Mac とも bash 4+
  # を想定。連想配列非対応シェルでの `set -u` 未定義変数エラーを
  # 早期に顕在化させるため fallback は用意しない）。
  local -A names=()
  local f name
  shopt -s nullglob
  for f in "$run1"/*.log "$run2"/*.log; do
    found_any=1
    name=$(basename "$f" .log)
    names["$name"]=1
  done
  shopt -u nullglob

  if [[ -n "$expect_csv" ]]; then
    local IFS=','
    local expect_name
    for expect_name in $expect_csv; do
      [[ -n "$expect_name" ]] && names["$expect_name"]=1
    done
  fi

  if [[ "${#names[@]}" -eq 0 ]]; then
    echo "FAIL: $run1／$run2 に *.log が 1 件も無く --expect-logs も未指定" >&2
    return 1
  fi

  local log1 log2 lines1 lines2 count1 count2
  for name in "${!names[@]}"; do
    log1="$run1/${name}.log"
    log2="$run2/${name}.log"

    if [[ ! -f "$log1" && ! -f "$log2" ]]; then
      echo "FAIL: ${name}.log が run1（$run1）・run2（$run2）の両方に存在しない" \
        "（--expect-logs で期待されたログがまるごと欠落。PR #1882 レビュー指摘）" >&2
      overall=1
      continue
    fi
    if [[ ! -f "$log1" ]]; then
      echo "FAIL: ${name}.log が run1 側（$run1）に存在しない（run2 側にのみ存在）" >&2
      overall=1
      continue
    fi
    if [[ ! -f "$log2" ]]; then
      echo "FAIL: ${name}.log が run2 側（$run2）に存在しない" >&2
      overall=1
      continue
    fi

    lines1="$(extract "$log1")"
    lines2="$(extract "$log2")"
    count1=$(count_nonblank "$lines1")
    count2=$(count_nonblank "$lines2")

    if [[ "$count1" -eq 0 || "$count2" -eq 0 ]]; then
      echo "FAIL: ${name}.log の bits=/fold_bits= 抽出行が 0 件（run1=$count1 run2=$count2）——" \
        "空の抽出結果同士は自明に一致するため diff は行わない（PR #1882 レビュー指摘）" >&2
      overall=1
      continue
    fi
    if [[ "$count1" -ne "$count2" ]]; then
      echo "FAIL: ${name}.log の抽出行数が run1/run2 で不一致（run1=$count1 run2=$count2）" >&2
      overall=1
      continue
    fi

    if diff <(printf '%s\n' "$lines1") <(printf '%s\n' "$lines2") >/dev/null; then
      echo "PASS: ${name}.log run-to-run bit 同一（${count1} 行）"
    else
      echo "FAIL: ${name}.log run-to-run で差分あり（下記 diff 参照）" >&2
      diff <(printf '%s\n' "$lines1") <(printf '%s\n' "$lines2") >&2 || true
      overall=1
    fi
  done

  if [[ "$found_any" -eq 0 && -z "$expect_csv" ]]; then
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
    echo "self-test 1/6 OK（正常系 PASS）"
  else
    echo "self-test 1/6 NG（正常系が FAIL 判定になった）" >&2
    return 1
  fi

  # ケース 2: 空ログ同士（抽出行 0 件）——本スクリプトの主目的:
  # 空同士の自明一致を PASS にしてはならない（レビュー指摘の再現）。
  mkdir -p "$tmp/case2/run1" "$tmp/case2/run2"
  : >"$tmp/case2/run1/x.log"
  : >"$tmp/case2/run2/x.log"
  if compare_dirs "$tmp/case2/run1" "$tmp/case2/run2" >/dev/null 2>&1; then
    echo "self-test 2/6 NG（空ログ同士が PASS 判定になった。レビュー指摘の再発）" >&2
    return 1
  else
    echo "self-test 2/6 OK（空ログ同士は FAIL 判定）"
  fi

  # ケース 3: 行数不一致（片方だけ行が欠落）——FAIL になるはず。
  mkdir -p "$tmp/case3/run1" "$tmp/case3/run2"
  printf 'case[a].bits=0x1\ncase[b].fold_bits=0x2\n' >"$tmp/case3/run1/x.log"
  printf 'case[a].bits=0x1\n' >"$tmp/case3/run2/x.log"
  if compare_dirs "$tmp/case3/run1" "$tmp/case3/run2" >/dev/null 2>&1; then
    echo "self-test 3/6 NG（行数不一致が PASS 判定になった）" >&2
    return 1
  else
    echo "self-test 3/6 OK（行数不一致は FAIL 判定）"
  fi

  # ケース 4: 実際に値が異なる（run-to-run 非決定性の検出）——FAIL。
  mkdir -p "$tmp/case4/run1" "$tmp/case4/run2"
  printf 'case[a].bits=0x1\n' >"$tmp/case4/run1/x.log"
  printf 'case[a].bits=0x9\n' >"$tmp/case4/run2/x.log"
  if compare_dirs "$tmp/case4/run1" "$tmp/case4/run2" >/dev/null 2>&1; then
    echo "self-test 4/6 NG（値の相違が PASS 判定になった）" >&2
    return 1
  else
    echo "self-test 4/6 OK（値の相違は FAIL 判定）"
  fi

  # ケース 5（追加レビュー指摘 (1) の再現）: ログが run2 側にのみ
  # 存在する（run1 側には無い）——旧実装は run1/*.log しか走査せず
  # このケース自体を一度も検査しないまま見逃していた。FAIL になる
  # はず。
  mkdir -p "$tmp/case5/run1" "$tmp/case5/run2"
  printf 'case[a].bits=0x1\n' >"$tmp/case5/run2/only_in_run2.log"
  if compare_dirs "$tmp/case5/run1" "$tmp/case5/run2" >/dev/null 2>&1; then
    echo "self-test 5/6 NG（run2 側のみに存在するログが見逃された。追加レビュー指摘 (1) の再発）" >&2
    return 1
  else
    echo "self-test 5/6 OK（run2 側のみに存在するログは FAIL 判定）"
  fi

  # ケース 6（追加レビュー指摘 (2) の再現）: --expect-logs で期待した
  # ログが run1・run2 の両方からまるごと欠落している——旧実装は
  # 「run1／run2 に実在するログ」しか知らないためこのケースを検出
  # できなかった。FAIL になるはず。
  mkdir -p "$tmp/case6/run1" "$tmp/case6/run2"
  printf 'case[a].bits=0x1\n' >"$tmp/case6/run1/present.log"
  printf 'case[a].bits=0x1\n' >"$tmp/case6/run2/present.log"
  if compare_dirs "$tmp/case6/run1" "$tmp/case6/run2" "present,missing_everywhere" >/dev/null 2>&1; then
    echo "self-test 6/6 NG（両側から丸ごと欠落した期待ログが見逃された。追加レビュー指摘 (2) の再発）" >&2
    return 1
  else
    echo "self-test 6/6 OK（--expect-logs で期待し両側から欠落したログは FAIL 判定）"
  fi

  echo "self-test: 全 6 ケース OK"
  return 0
}

main() {
  local expect_csv=""
  local args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --self-test)
        self_test
        exit $?
        ;;
      --expect-logs)
        expect_csv=${2:-}
        shift 2
        ;;
      --expect-logs=*)
        expect_csv=${1#--expect-logs=}
        shift
        ;;
      *)
        args+=("$1")
        shift
        ;;
    esac
  done

  local run1=${args[0]:-}
  local run2=${args[1]:-}
  if [[ -z "$run1" || -z "$run2" ]]; then
    echo "usage: $0 [--expect-logs <name1,name2,...>] <run1_dir> <run2_dir>" >&2
    echo "       $0 --self-test" >&2
    exit 2
  fi
  if [[ ! -d "$run1" || ! -d "$run2" ]]; then
    echo "FAIL: ディレクトリが存在しない（run1=$run1 run2=$run2）" >&2
    exit 1
  fi
  if compare_dirs "$run1" "$run2" "$expect_csv"; then
    echo "PASS: 全ログ群で run-to-run 決定性チェック成立"
    exit 0
  else
    echo "FAIL: 1 件以上のログ群で run-to-run 決定性チェック不成立" >&2
    exit 1
  fi
}

main "$@"
