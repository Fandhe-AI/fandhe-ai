#!/usr/bin/env bash
# イシュー #4163: 前提ゲート R0〜R3 の実行（GB10 専用。規則の正本は同ディレクトリの RULE.txt）。
# scripts/bench/framework-compare/run_ab_infer_graph_default_cuda.sh から計測の前に呼ばれる。
# 各ログ末尾に rc= 行を残し、判定は aggregate_4163.py が生ログで行う（本スクリプトの終了コードは補助）。
#   R0: 非後退（env なし）  R1(a): env=1 の capture 実機テスト  R1(b): env=1 の並列推論テスト 5 回
#   R2: off／on の 2 プロセスでダンプ全行一致  R3: 既定 ON 波及（腕 C = tree、腕 A = tree + default-on.patch）
# 引数: <label> <out_dir> <runner_root> <target_dir>。bash 3.2 互換（mapfile・連想配列・eval 不使用）。
set -u
LABEL=${1:-}
OUT=${2:-}
ROOT=${3:-}
TDIR=${4:-}
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ || -z "$OUT" || -z "$ROOT" || -z "$TDIR" ]]; then
  echo "usage: $0 <label> <out_dir> <runner_root> <target_dir>" >&2
  exit 1
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
PATCH="$HERE/default-on.patch"
[[ -f "$PATCH" ]] || { echo "error: default-on.patch が無い" >&2; exit 1; }
MANIFEST="$ROOT/Cargo.toml"
CT=(cargo test --manifest-path "$MANIFEST" --locked --release --target-dir "$TDIR")
FAILED=0

# env_mode: off = 3 変数とも未設定、on = INFER=1 のみ、optout = INFER=0 のみ（補助）。
gate_run() { # <log> <env_mode> <cmd...>
  local log=$1 mode=$2 rc
  shift 2
  case "$mode" in
    on) env -u FANDHE_AI_CUDA_GRAPH_STEP -u FANDHE_AI_CUDA_GRAPH_TRAIN FANDHE_AI_CUDA_GRAPH_INFER=1 "$@" >"$log" 2>&1 ;;
    optout) env -u FANDHE_AI_CUDA_GRAPH_STEP -u FANDHE_AI_CUDA_GRAPH_TRAIN FANDHE_AI_CUDA_GRAPH_INFER=0 "$@" >"$log" 2>&1 ;;
    *) env -u FANDHE_AI_CUDA_GRAPH_STEP -u FANDHE_AI_CUDA_GRAPH_TRAIN -u FANDHE_AI_CUDA_GRAPH_INFER "$@" >"$log" 2>&1 ;;
  esac
  rc=$?
  echo "rc=${rc}" >>"$log"
  return "$rc"
}
note_fail() { echo "gate ${1} FAILED rc=${2}" >>"$OUT/skipped-gates-${LABEL}.log"; FAILED=1; }

# ---- R0 ----
gate_run "$OUT/r0-linear-${LABEL}.log" off "${CT[@]}" -p fandhe-ai-backend-cuda --test linear_forward_device_real_device -- --ignored --nocapture --test-threads=1
rc=$?; [[ $rc -eq 0 ]] || note_fail r0-linear $rc
gate_run "$OUT/r0-graph-${LABEL}.log" off "${CT[@]}" -p fandhe-ai-backend-cuda --test graph_capture_real_device -- --ignored --nocapture --test-threads=1
rc=$?; [[ $rc -eq 0 ]] || note_fail r0-graph $rc

# ---- R1(a) ----
gate_run "$OUT/r1a-capture-${LABEL}.log" on "${CT[@]}" -p fandhe-ai-backend-cuda --test infer_graph_capture_real_device -- --ignored --nocapture --test-threads=1
rc=$?; [[ $rc -eq 0 ]] || note_fail r1a $rc

# ---- R1(b): 並列（--test-threads 指定なし）を 5 回。失敗しても 5 回すべて実行してログへ残す ----
for i in 1 2 3 4 5; do
  gate_run "$OUT/r1b-run${i}-${LABEL}.log" on "${CT[@]}" -p fandhe-ai --test predict_device_chain_cuda_bit_identity -- --ignored --nocapture
  rc=$?; [[ $rc -eq 0 ]] || note_fail "r1b-run${i}" $rc
done

# ---- R2: off／on の 2 プロセスでダンプ全行一致（行数 batch*10 は aggregate が検査） ----
for b in 64 1024 4096; do
  for mode in off on; do
    gate_run "$OUT/r2-${mode}-${b}-${LABEL}.log" "$mode" "${CT[@]}" -p fandhe-ai --test predict_device_chain_cuda_graph_bit_identity -- --ignored --exact --nocapture "predict_resident_bit_dump_cuda_graph_batch_${b}"
    rc=$?; [[ $rc -eq 0 ]] || note_fail "r2-${mode}-${b}" $rc
    sed -nE 's/^(test [^ ]+ \.\.\. )?(out\[[0-9]+\]\.bits=0x[0-9a-fA-F]+)[[:space:]]*$/\2/p' "$OUT/r2-${mode}-${b}-${LABEL}.log" >"$OUT/r2-dump-${mode}-${b}-${LABEL}.txt" || true
  done
done

# ---- R3: 既定 ON の波及。tree を別ディレクトリへ展開し、腕 A/B にだけパッチを当てる（ROOT の作業ツリーは触らない） ----
# 反復回数 N は RULE.txt の `  r3_repeats: <正の整数>`（所有者が決める。#4163 所有者判断 2026-11-11）。腕 C・A を N 回ずつ流し、
# aggregate_4163.py が「腕 A で毎回失敗・腕 C で毎回成功」だけを阻害、結果が回ごとに混ざったテストを不安定（判定外）に分ける。
R3_REPEATS="$(sed -n 's/^  r3_repeats: \([1-9][0-9]*\)$/\1/p' "$HERE/RULE.txt" 2>/dev/null | head -n1)"
if [[ -z "$R3_REPEATS" ]]; then
  echo "error: RULE.txt に r3_repeats（正の整数）が無い -> R3 を実行しない" >&2
  note_fail r3-repeats 1
fi
# 標準出力へ直接書くダンプテスト（facade の tests/cpu_sme_gate_bit_dump.rs。高速化のための意図した作りでテストは変えない）は
# libtest の捕捉を通らず "test <name> ... out[...]" の 1 行になって結果行の抽出を壊す。本体の実行から --exact --skip で外し、
# 同じ回の中で --exact で単独に流す。そのログの結果も R3 の集合に含める（R3 の対象は狭まらない）。
SME_DUMP_TEST=dump_cpu_sme_gate_bits
# テスト単位の結果を bin<TAB>name<TAB>status へ機械抽出する。bin は "Running unittests ..." が lib、
# "Running tests/x.rs" が x、"Doc-tests <crate>" が doctest。名前は "test " の後から最初の " ... " の前までの全体
# （doctest は "src/file.rs - item (line N)" で先頭トークンだけでは衝突する）。status は ok／FAILED／ignored（理由付きは ignored へ正規化）。
# 許可集合の外のステータスは抽出せず素通しし、aggregate_4163.py が未知ステータスとして UNDETERMINED にする。
extract_tsv() { # <log>
  awk '/^ *Running /{ if ($2 == "unittests") { b = "lib" } else { b = $2; sub(/.*\//, "", b); sub(/\.rs$/, "", b) } next }
       /^ *Doc-tests /{ b = "doctest"; next }
       /^test .* \.\.\. [^ ]/{
         line = substr($0, 6); k = index(line, " ... ")
         name = substr(line, 1, k - 1); st = substr(line, k + 5)
         if (st ~ /^ignored/) { st = "ignored" }
         print b "\t" name "\t" st }' "$1"
}
# ダンプ単独実行のログからダンプ本体（行頭の out[...] 行）を落とし、落とした行数を 1 行残す（gate_run の rc= 行は呼び出し側で末尾に付け直す）。
drop_dump_lines() { # <raw log>
  grep -v '^rc=[0-9]*$' "$1" | grep -Ev '^out\[[^]]*\]\[[0-9]+\]\.bits=0x[0-9a-fA-F]+$'
  echo "dump_lines_dropped=$(grep -cE '^out\[[^]]*\]\[[0-9]+\]\.bits=0x[0-9a-fA-F]+$' "$1")"
}
# ダンプ単独実行の結果を 1 行の TSV にする。status は集計行（ちょうど 1 件実行・ignored 0）から取り、
# それ以外の形（集計行が無い・複数・件数が違う）は何も出さない（aggregate_4163.py が「テスト結果が 0 件」で UNDETERMINED にする）。
extract_smedump_tsv() { # <log>
  awk -v n="$SME_DUMP_TEST" '
    /^test result: ok\. 1 passed; 0 failed; 0 ignored/ { st = "ok"; c++ }
    /^test result: FAILED\. 0 passed; 1 failed; 0 ignored/ { st = "FAILED"; c++ }
    /^test result: / { all++ }
    END { if (c == 1 && all == 1) print "cpu_sme_gate_bit_dump\t" n "\t" st }' "$1"
}
r3_arm() { # <arm: C|A|B> <env_mode> <reps>
  local arm=$1 mode=$2 reps=$3 dir crate log rc k raw
  dir="$(mktemp -d "${TMPDIR:-/tmp}/r3-${arm}.XXXXXX")" || { note_fail "r3-${arm}-mktemp" 1; return; }
  if ! git -C "$ROOT" archive HEAD | tar -x -C "$dir"; then
    rm -rf "$dir"; note_fail "r3-${arm}-archive" 1; return
  fi
  if [[ "$arm" != "C" ]]; then
    if ! (cd "$dir" && patch -p1 <"$PATCH" >/dev/null); then
      rm -rf "$dir"; note_fail "r3-${arm}-patch" 1; return
    fi
  fi
  k=1
  while [[ $k -le $reps ]]; do
    for crate in fandhe-ai-backend-cuda fandhe-ai; do
      log="$OUT/r3-${arm}-${crate}-rep${k}-${LABEL}.log"
      # --nocapture は付けない: 付けるとテスト自身の出力が "test <name> ... " の行に混ざり、結果行の抽出が壊れる。
      # 通過テストの出力は libtest が捕捉し、"test <name> ... ok|FAILED|ignored" が 1 行で出る（失敗時の出力は末尾の failures 節）。
      # 1 回目でビルドし、2 回目以降は同じ target-dir のバイナリを再実行する（ソースは回の間で変えない）。
      gate_run "$log" "$mode" cargo test --manifest-path "$dir/Cargo.toml" --locked --release --target-dir "${TDIR}-r3-${arm}" \
        -p "$crate" --no-fail-fast -- --include-ignored --test-threads=1 --exact --skip "$SME_DUMP_TEST"
      rc=$?
      # 腕 C／A の rc は 0（全通過）か 101（テスト失敗）以外を異常終了として前提ゲートの不合格にする
      # （101 の中身は aggregate_4163.py が FAILED 件数・集計行と照合する。腕 B は判定外）。
      if [[ "$arm" != "B" && $rc -ne 0 && $rc -ne 101 ]]; then note_fail "r3-${arm}-${crate}-rep${k}" "$rc"; fi
      extract_tsv "$log" >"$OUT/r3-${arm}-${crate}-rep${k}-${LABEL}.tsv"
    done
    if [[ "$arm" != "B" ]]; then
      # ダンプテストの単独実行。ダンプ本体（^out[ 行。約 670 万行）はログに残さず件数だけ記録する。
      # 先頭のダンプ行は libtest の "test <name> ... " に続いて同じ行に出るため、status は行ではなく
      # 当該バイナリの集計行（1 件だけ実行）から取る。集計行が 1 件実行の形でなければ TSV を空にし、aggregate が未完として扱う。
      log="$OUT/r3-${arm}-smedump-rep${k}-${LABEL}.log"
      raw="$log.raw"
      gate_run "$raw" "$mode" cargo test --manifest-path "$dir/Cargo.toml" --locked --release --target-dir "${TDIR}-r3-${arm}" \
        -p fandhe-ai --test cpu_sme_gate_bit_dump -- --include-ignored --exact "$SME_DUMP_TEST" --test-threads=1
      rc=$?
      { drop_dump_lines "$raw"; echo "rc=${rc}"; } >"$log"
      rm -f "$raw"
      if [[ $rc -ne 0 && $rc -ne 101 ]]; then note_fail "r3-${arm}-smedump-rep${k}" "$rc"; fi
      extract_smedump_tsv "$log" >"$OUT/r3-${arm}-smedump-rep${k}-${LABEL}.tsv"
    fi
    k=$((k + 1))
  done
  rm -rf "$dir"
}
if [[ -n "$R3_REPEATS" ]]; then
  r3_arm C off "$R3_REPEATS"
  r3_arm A off "$R3_REPEATS"
  # 補助（判定外）: パッチ適用ビルドで opt-out が効くこと。1 回だけ。
  r3_arm B optout 1
fi

echo "gates done failed=${FAILED}"
[[ "$FAILED" -eq 0 ]]
