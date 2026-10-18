#!/usr/bin/env bash
# イシュー #3166: Mac mini（Apple M4）での Transformer・Functional・可変長系列 7 群（8 テスト）の
# Metal 実機 parity 実行スクリプト。手動実行専用（CI からは呼ばない）。
# 規則は同ディレクトリの RULE.txt（事前登録）が正。判定は各テストの cargo test 終了コードと出力件数。
# docs/perf/logs/rocm-parity-3159/run_rocm_parity.sh と同型。macOS 標準 bash 3.2 で動くよう、
# 連想配列・mapfile・${var,,}・sed -i は使わない。
#
# 終了コード: 0=全 pass／1=fail またはマスク漏れ（記録は残る。マスク漏れのファイルは削除）／
#             2=引数・前提不正（Darwin 以外、--list の選択件数が 1 でない場合を含む）。
# 使い方（Mac mini 上。zsh -lc で CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai を指定して呼ぶ）:
#   bash run_metal_parity.sh <tree-abs-path> <out-abs-path>
#   tree: 展開済みリポジトリ（直下に .rev-stamp が必要）／out: 未存在の出力先
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <tree-abs-path> <out-abs-path>" >&2
  exit 2
fi
TREE=$1
OUT=$2
# 引数は絶対パス + 文字種 allowlist で検査する（A03）。改行を含む値は =~ の全体一致で拒否する。
for p in "$TREE" "$OUT"; do
  if ! [[ $p =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "絶対パスかつ許可文字のみが必要" >&2
    exit 2
  fi
done
if [ "$(uname -s)" != "Darwin" ]; then
  echo "Darwin（Mac mini）以外では実行しない" >&2
  exit 2
fi
[ -d "$TREE" ] || { echo "tree が存在しない" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "出力先が既に存在する（上書きしない）" >&2; exit 2; }
[ -f "$TREE/.rev-stamp" ] || { echo ".rev-stamp が無い" >&2; exit 2; }

# .rev-stamp は「40 桁の小文字 16 進数 + 任意の末尾改行 1 つ」だけであること。
# 末尾に番兵 x を付けて $(...) の改行除去を避け、CR・空白・余計な行を全て拒否する。
export LC_ALL=C
STAMP_SIZE=$(wc -c <"$TREE/.rev-stamp" | tr -d ' ')
STAMP=$(
  cat "$TREE/.rev-stamp"
  printf x
)
STAMP=${STAMP%x}
STAMP_BODY=${STAMP%$'\n'}
if { [ "$STAMP_SIZE" -ne 40 ] && [ "$STAMP_SIZE" -ne 41 ]; } || ! [[ $STAMP_BODY =~ ^[0-9a-f]{40}$ ]]; then
  echo ".rev-stamp が 40 桁の小文字 16 進数 1 行だけでない" >&2
  exit 2
fi

mkdir -p "$OUT/logs"
cp "$TREE/.rev-stamp" "$OUT/.rev-stamp"

# 環境情報は allowlist で取得する。hostname は取らず masked と固定で書く。
env_info() { # env_info <label>
  echo "== $1 =="
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "hostname: masked"
  echo "sw_vers: $( (sw_vers | tr '\n' ' ') 2>/dev/null || echo 取得不能)"
  echo "cpu: $( (sysctl -n machdep.cpu.brand_string) 2>/dev/null || echo 取得不能)"
  echo "rustc: $( (rustc -V) 2>/dev/null || echo 取得不能)"
  echo "cargo: $( (cargo -V) 2>/dev/null || echo 取得不能)"
  echo "load: $( (uptime | sed -e 's/.*load averages*: *//') 2>/dev/null || echo 取得不能)"
  echo "rev_stamp: $STAMP_BODY"
}
env_info before >"$OUT/env_info.raw"

# 他の cargo／rustc が走っていれば並走させない（RULE.txt）。
if pgrep -x cargo >/dev/null 2>&1 || pgrep -x rustc >/dev/null 2>&1; then
  echo "他の cargo／rustc プロセスが動作中のため停止" >&2
  exit 2
fi
echo "concurrent_cargo_rustc: none" >>"$OUT/env_info.raw"

cd "$TREE"
# ビルドを先に終え、計測中にコンパイルを並走させない。
cargo test -p fandhe-ai --lib --no-run
cargo test -p fandhe-ai --test compat_sequential_layers_backend_parity --no-run
cargo test -p fandhe-ai --test merge_ops_backend_parity --no-run
cargo test -p fandhe-ai --test packed_sequence_backend_parity --no-run

# 行: 群|ターゲット引数|テストのフルパス（関数名がログのファイル名になる）。
CASES='mha-config-sequential-2530|--test compat_sequential_layers_backend_parity|metal_multihead_attention_config_matches_cpu
transformer-decoder-sequential-2532|--test compat_sequential_layers_backend_parity|metal_transformer_decoder_layer_matches_cpu
transformer-sequential-2533|--test compat_sequential_layers_backend_parity|metal_transformer_matches_cpu
functional-graph-2665|--lib|compat::functional::tests::metal_functional_graph_matches_cpu_reference
functional-fit-2667|--lib|compat::functional::fit_parity_tests::metal_parameter_gradients_match_cpu_reference
merge-ops-2666|--test merge_ops_backend_parity|metal_merge_ops_match_cpu_reference
merge-ops-2666|--lib|compat::functional::merge_tests::metal_merge_graph_matches_cpu_reference
packed-sequence-2647|--test packed_sequence_backend_parity|metal_packed_sequence_matches_cpu_reference'

printf 'group\ttarget\ttest\texit_code\trunning\tresult\tverdict\n' >"$OUT/summary.tsv"
ANY_FAIL=0
while IFS='|' read -r GROUP TARGET TEST; do
  FN=${TEST##*::}
  mkdir -p "$OUT/logs/$GROUP"
  # 選択の検証: --list で ": test" 終わりの行がちょうど 1 行であること（0 または 2 以上は実行しない）。
  # TARGET は上の固定文字列のみ（ユーザー入力ではない）なので単語分割で展開する。
  # shellcheck disable=SC2086
  N=$(cargo test -p fandhe-ai $TARGET -- --ignored --exact --list "$TEST" 2>/dev/null | grep -c ': test$' || true)
  if [ "$N" -ne 1 ]; then
    echo "--list の選択件数が 1 でない: $TEST ($N)" >&2
    printf '%s\t%s\t%s\t-\t-\t-\tundetermined\n' "$GROUP" "$TARGET" "$TEST" >>"$OUT/summary.tsv"
    exit 2
  fi
  set +e
  # shellcheck disable=SC2086
  cargo test -p fandhe-ai $TARGET -- --ignored --exact --nocapture --test-threads=1 "$TEST" \
    >"$OUT/logs/$GROUP/$FN.raw" 2>&1
  RC=$?
  set -e
  echo "$RC" >"$OUT/logs/$GROUP/$FN.exit_code"
  RUNNING=$(grep -c '^running 1 test$' "$OUT/logs/$GROUP/$FN.raw" || true)
  RESULT=$(grep -E '^test result: ' "$OUT/logs/$GROUP/$FN.raw" | head -n 1 || true)
  if [ "$RC" -eq 0 ] && [ "$RUNNING" -eq 1 ] && [[ $RESULT == "test result: ok. 1 passed; 0 failed"* ]]; then
    V=pass
  else
    V=fail
    ANY_FAIL=1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$GROUP" "$TARGET" "$TEST" "$RC" "$RUNNING" "${RESULT:-none}" "$V" >>"$OUT/summary.tsv"
done <<EOF
$CASES
EOF

env_info after >>"$OUT/env_info.raw"

# マスクしてから収録し、残っていれば fail-closed で止める（残ったファイルは raw ごと削除する）。
HN=$(hostname)
LEAK=0
mask() { # mask <raw> <final>
  sed -e "s|$HN|<metal-node>|g" -e "s|$HOME|~|g" -e "s|${USER:-__nouser__}|<user>|g" "$1" >"$2"
  rm -f "$1"
  for s in "$HN" "$HOME"; do
    if [ -n "$s" ] && grep -Fq -- "$s" "$2"; then
      echo "マスク漏れ: $2" >&2
      rm -f "$2"
      LEAK=1
      break
    fi
  done
}
mask "$OUT/env_info.raw" "$OUT/env_info.txt"
for raw in "$OUT"/logs/*/*.raw; do
  mask "$raw" "${raw%.raw}.log"
done

echo "done: any_fail=$ANY_FAIL leak=$LEAK"
if [ "$LEAK" -ne 0 ] || [ "$ANY_FAIL" -ne 0 ]; then
  exit 1
fi
