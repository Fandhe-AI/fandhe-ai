#!/bin/bash
# イシュー #2118: SME_MIN_K 候補（64／128／256）の計測専用ツリー作成と
# 検証の共有ヘルパー。`orchestrate_m4max.sh`（Apple M4 Max）と
# `gb10/orchestrate_gb10.sh`（DGX Spark GB10）が source する（bash 3.2 でも
# 動くよう連想配列・mapfile は使わない）。判定規則は同ディレクトリの
# RULE.txt（事前登録）が正で、本ファイルは規則の実行側のみを担う。
#
# 役割: 事前登録コミット（SME2118_REGISTERED_BASE）を `git archive` で一時ディレクトリへ展開して before ツリーと
# after(K) ツリーを作り、on-arm-k{K}.patch を after にだけ適用したうえで
# 「差分が gemm_blis/mod.rs の 1 ファイルのみ」「定数 2 行がパッチどおり」
# を fail-closed で assert する（RULE.txt §1。run_ab_sme_cpu.sh は after の
# 差分を検証しないため、その分をここで補う）。main の定数は変更しない。
# 呼び出し元は本ファイルを source する 2 本の orchestrate_*.sh（と self-test の selftest_collect.sh）のみ。

# ロケール固定（RULE.txt §15）: 本ファイルと呼び出し元は、コマンドの出力（cargo の結果行・git／patch の
# 終了コード・awk の数値比較・case の文字範囲・sort 順）に依存する。日本語ロケール等のホストで文言・
# 小数点・照合順が変わらないよう C へ固定する（source した呼び出し元と子プロセスにも及ぶ。cargo・rustc は
# ローカライズされず、python3 は 3.7 以降 C ロケールで UTF-8 モードになるため JSONL・日本語ログの読み書きは不変）。
# 呼び出し元の orchestrate_*.sh も冒頭で同じ export を行う（source 前の処理を含めて固定するため）。
export LC_ALL=C

SME2118_MOD_REL="crates/backend-cpu/src/gemm_blis/mod.rs"

# 事前登録 before コミット（RULE.txt ヘッダ「登録時点の main HEAD」。before 腕の基準）。
# 実行時の HEAD ではなく本 sha を固定で使い、main の前進や作業ブランチの差異で
# before/after の基準が事後にドリフトしないようにする（RULE.txt §1・§10）。
SME2118_REGISTERED_BASE="0b25525fa4026b951021a5d0da3a9d613b507502"

# sme2118_resolve_base <repo_root>: 登録 sha がリポジトリに存在するか検証する（fail-closed）。
# 存在しなければ（浅い clone 等）計測せず停止する。HEAD へのフォールバックはしない。
sme2118_resolve_base() {
  local repo=$1
  git -C "$repo" cat-file -e "${SME2118_REGISTERED_BASE}^{commit}" 2>/dev/null \
    || { echo "error: 事前登録 before コミット ${SME2118_REGISTERED_BASE} がリポジトリに無い（git fetch で取得。HEAD へのフォールバックはしない）" >&2; return 1; }
}

# sme2118_rt_abnormal_count <cargo_test_log>: cargo test のログから「テスト失敗以外の異常終了」行数を返す。
# cargo はテスト失敗時にも `error: test failed` と `process didn't exit successfully ... (exit status: 101)`
# を出すため、これらは異常扱いにしない（既知 FAIL のみでも rt_verdict が regression-suspect になるのを防ぐ）。
# 異常とするのは、コンパイル失敗（could not compile）と、終了状態が 101 以外（シグナル・abort 等）の
# `process didn't exit successfully` 行。
sme2118_rt_abnormal_count() {
  grep -E "could not compile|error: could not|process didn't exit successfully" "$1" \
    | grep -cvE '\((exit status|exit code): 101\)'
}

# K の allowlist（A03 インジェクション対策。パッチ名・LABEL へ埋め込むため）
sme2118_validate_k() {
  case "${1:-}" in
    64 | 128 | 256) return 0 ;;
    *)
      echo "error: K は 64／128／256 のいずれか（got: ${1:-<empty>}）" >&2
      return 1
      ;;
  esac
}

# LABEL は run_ab_sme_cpu.sh と同じ allowlist（英数字・._-）
sme2118_validate_label() {
  case "${1:-}" in
    '' | *[!A-Za-z0-9._-]*)
      echo "error: LABEL は [A-Za-z0-9._-]+ のみ（got: ${1:-<empty>}）" >&2
      return 1
      ;;
    *) return 0 ;;
  esac
}

sme2118_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1"
  else
    shasum -a 256 "$1"
  fi
}

# sme2118_tree_fingerprint <work_dir>: <work_dir>/before と <work_dir>/after の差分エントリを 1 行 1 件で標準出力へ出す。
# diff の出力文言（ロケールで翻訳される）は解析しない（RULE.txt §15。PR #2465 PRRT_kwDOTuUCJc6nj-IA）。
#   - 片側にしか無いパス（.orig・.rej の混入、ファイル／ディレクトリの増減）: `only: <path>`
#   - 両側にあり種別（ファイル／ディレクトリ／リンク）が異なる、または内容が異なる: `<path>`（内容は cmp -s の終了コードで判定）
# パスは各ツリー起点の相対パス（先頭の ./ を除く）。改行を含むパス名は git archive の展開物に無い前提。
# 作業用のパス一覧は <work_dir> 直下（before／after の外）に置く。
# 戻り値: 列挙に成功すれば 0（差分の有無は問わない）。find・sort・comm のいずれかが失敗すれば 1
# （列挙の欠けを「差分なし」と取り違えないよう、各段の終了コードを個別に確認する）。
sme2118_tree_fingerprint() {
  local work=$1 t p only_b only_a common
  for t in before after; do
    (cd "$work/$t" && find . -print) >"$work/fingerprint-$t.raw" || return 1
    LC_ALL=C sort "$work/fingerprint-$t.raw" >"$work/fingerprint-$t.lst" || return 1
  done
  # comm は C ロケールでソート済みの入力を要求する
  only_b=$(LC_ALL=C comm -23 "$work/fingerprint-before.lst" "$work/fingerprint-after.lst") || return 1
  only_a=$(LC_ALL=C comm -13 "$work/fingerprint-before.lst" "$work/fingerprint-after.lst") || return 1
  common=$(LC_ALL=C comm -12 "$work/fingerprint-before.lst" "$work/fingerprint-after.lst") || return 1
  # 片側のみのパス
  while IFS= read -r p; do
    case "$p" in '') continue ;; esac
    printf 'only: %s\n' "${p#./}"
  done <<EOF
$only_b
$only_a
EOF
  # 両側にあるパス: 種別の一致と内容の一致（cmp -s の終了コードのみを見る）
  while IFS= read -r p; do
    case "$p" in '' | .) continue ;; esac
    if [ -d "$work/before/$p" ] && [ -d "$work/after/$p" ]; then
      continue
    fi
    if [ -L "$work/before/$p" ] || [ -L "$work/after/$p" ]; then
      if [ "$(readlink "$work/before/$p")" != "$(readlink "$work/after/$p")" ]; then printf '%s\n' "${p#./}"; fi
      continue
    fi
    if [ -f "$work/before/$p" ] && [ -f "$work/after/$p" ]; then
      cmp -s "$work/before/$p" "$work/after/$p" || printf '%s\n' "${p#./}"
      continue
    fi
    printf '%s\n' "${p#./}"
  done <<EOF
$common
EOF
  return 0
}

# sme2118_prepare_trees <repo_root> <work_dir> <k> <patch_file> <out_dir>
#   <work_dir>/before と <work_dir>/after を作る。パッチ sha256・指紋差分・定数行・成功記録を
#   <out_dir>/patch_sha256.txt・tree_diff.txt・gate_constant.txt・tree_verify.txt へ記録する
#   （aggregate.py はこれらの記録を P-TREE の前提として照合する。RULE.txt §15）。
sme2118_prepare_trees() {
  local repo=$1 work=$2 k=$3 patch=$4 out=$5
  sme2118_validate_k "$k" || return 1
  [ -f "$patch" ] || { echo "error: パッチが無い: $patch" >&2; return 1; }
  mkdir -p "$work/before" "$work/after" "$out" || return 1
  sme2118_resolve_base "$repo" || return 1
  local head=$SME2118_REGISTERED_BASE
  git -C "$repo" archive "$head" | tar -x -C "$work/before" || return 1
  git -C "$repo" archive "$head" | tar -x -C "$work/after" || return 1
  {
    echo "head=${head}"
    echo "current_head=$(git -C "$repo" rev-parse HEAD)"
    sme2118_sha256 "$patch" | awk '{print "patch_sha256=" $1}'
  } >"$out/patch_sha256.txt"
  (cd "$work/after" && patch -p1 --forward <"$patch") >"$out/patch_apply.log" 2>&1 \
    || { echo "error: パッチ適用に失敗（$out/patch_apply.log）" >&2; return 1; }
  # 指紋差分: 差分エントリは mod.rs の 1 件のみ（.orig 等の混入・ファイル増減も検出）。
  # 判定は文言の解析ではなく、差分エントリ一覧がちょうど mod.rs の 1 行であることの完全一致で行う。
  local diffs
  diffs=$(sme2118_tree_fingerprint "$work") \
    || { echo "error: ツリーの指紋差分の列挙に失敗" >&2; return 1; }
  printf '%s\n' "$diffs" >"$out/tree_diff.txt"
  if [ "$diffs" != "${SME2118_MOD_REL}" ]; then
    echo "error: ツリー差分が mod.rs 1 件のみでない（$out/tree_diff.txt）" >&2
    return 1
  fi
  # 定数 2 行の assert（before は現行値・after はパッチどおり）
  local b_en b_k a_en a_k
  b_en=$(grep -E '^const SME_PRODUCTION_ENABLED: bool = ' "$work/before/$SME2118_MOD_REL")
  b_k=$(grep -E '^const SME_MIN_K: usize = ' "$work/before/$SME2118_MOD_REL")
  a_en=$(grep -E '^const SME_PRODUCTION_ENABLED: bool = ' "$work/after/$SME2118_MOD_REL")
  a_k=$(grep -E '^const SME_MIN_K: usize = ' "$work/after/$SME2118_MOD_REL")
  {
    echo "before: ${b_en} / ${b_k}"
    echo "after:  ${a_en} / ${a_k}"
  } >"$out/gate_constant.txt"
  [ "$b_en" = "const SME_PRODUCTION_ENABLED: bool = false;" ] \
    && [ "$b_k" = "const SME_MIN_K: usize = 64;" ] \
    && [ "$a_en" = "const SME_PRODUCTION_ENABLED: bool = true;" ] \
    && [ "$a_k" = "const SME_MIN_K: usize = ${k};" ] \
    || { echo "error: 定数行がパッチの意図と一致しない（$out/gate_constant.txt）" >&2; return 1; }
  # 成功記録（P-TREE。全 assert 通過後にだけ書く）
  echo "trees_ok head=${head} K=${k}" >"$out/tree_verify.txt" || return 1
  echo "trees ok: head=${head} K=${k}"
}

# 収録前のマスク（RULE.txt §12。作業ディレクトリ→<work>・$HOME→<home>）。標準入力を標準出力へ。
# ホスト名の内容置換はしない（gb10・mac 等の短い名前が LABEL や単語を壊すため。ホスト名は
# env_info に `hostname=masked` と書くだけで、収録テキストへ出さない）。
sme2118_mask() {
  local work=${1:-/nonexistent-work-dir}
  sed -e "s#${work}#<work>#g" -e "s#${HOME}#<home>#g"
}

# sme2118_collect_r1r2 <bench_dir> <label> <dest_dir> <work_dir>
#   run_ab_sme_cpu.sh の成果物（バイナリ・target を除く）をマスクして dest へ収録する。
#   fail-closed（RULE.txt §13）: 集計（aggregate.py）が読む必須成果物のいずれかが欠損・空、または
#   コピー（マスク）が 1 件でも失敗したら、欠損を列挙して非ゼロで返す。呼び出し側は戻り値を必ず
#   確認し、env_info／rt_result へ終了コードを記録して非ゼロ終了へ伝播する（aggregate.py は
#   収録の終了コード 0 の記録がない系列を判定不能にする）。
#   必須: results/raw の 6 JSONL（before/after x gemm/train/infer）・load-gate ログ・
#   compare-exit ログ・bench 直下の compare-{task}-1978-cpu-{label}.{md,err}（task 3 種）。
#   任意（存在すれば収録）: uptime・skipped 等その他の成果物。
sme2118_collect_r1r2() {
  local bench=$1 label=$2 dest=$3 work=$4 f base task arm rc=0 seen=" "
  mkdir -p "$dest" || return 1
  local raw="$bench/results/raw"
  # 必須成果物の存在確認（JSONL・md・ゲート／終了コードログは空も不可。err は正常時空なので存在のみ）
  for task in gemm train infer; do
    for arm in before after; do
      [ -s "$raw/results-${arm}-${label}-cpu-${task}.jsonl" ] \
        || { echo "error: 必須成果物が無い／空: results-${arm}-${label}-cpu-${task}.jsonl" >&2; rc=1; }
    done
    [ -s "$bench/compare-${task}-1978-cpu-${label}.md" ] \
      || { echo "error: 必須成果物が無い／空: compare-${task}-1978-cpu-${label}.md" >&2; rc=1; }
    [ -e "$bench/compare-${task}-1978-cpu-${label}.err" ] \
      || { echo "error: 必須成果物が無い: compare-${task}-1978-cpu-${label}.err" >&2; rc=1; }
  done
  for base in "load-gate-1978-cpu-${label}.log" "compare-exit-1978-cpu-${label}.log"; do
    [ -s "$raw/$base" ] || { echo "error: 必須成果物が無い／空: $base" >&2; rc=1; }
  done
  # 収録（存在するものは必須・任意を問わず全て。1 件でもマスク／書き込みに失敗すれば rc=1）
  for f in "$raw"/*"-${label}"* "$bench"/compare-*"-${label}".md "$bench"/compare-*"-${label}".err; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    case "$base" in bench-fandhe-*) continue ;; esac
    case "$seen" in *" $base "*) continue ;; esac
    seen="${seen}${base} "
    if ! sme2118_mask "$work" <"$f" >"$dest/$base"; then
      echo "error: 収録（マスク）に失敗: $base" >&2
      rm -f "$dest/$base"
      rc=1
    fi
  done
  return "$rc"
}
