#!/usr/bin/env bash
# イシュー #2049: `SME_PRODUCTION_ENABLED`（`crates/backend-cpu/src/
# gemm_blis/mod.rs:3085`。既定 `false`）を `true` にした after ツリーと
# `false` の before ツリーの間で、`crates/facade/tests/
# cpu_sme_gate_bit_dump.rs::dump_cpu_sme_gate_bits`（gemm512/1024/2048・
# train size=64（reuse・L1 d_weight が到達）・infer size=64（非到達
# 対照）の全出力を `out[<label>][<i>].bits=0x........` 形式で 1 要素
# 1 行印字する `#[ignore]` テスト）を実行し、`^out\[` 行を抽出して
# `diff`／`sha256sum` で bit 完全一致を確認する。
#
# `docs/perf/logs/infer-chain-single-sync-cuda-1689/run_bitdump.sh`
# （before/after 2 ツリー実行 → 抽出 → 期待行数検査 → diff の手本）と
# 同型。`cpu_sme_gate_bit_dump.rs` は本イシューで新設したファイルの
# ため before ツリー（main）には存在しない想定であり、after ツリーの
# 同ファイルを before ツリーの同パスへコピーしてから実行する
# （`fandhe_ai`／`fandhe_ai_autodiff`／`fandhe_ai_tensor_core`／
# `bench_harness` の公開 API のみに依存するテストのため、before ツリー
# の公開面でそのままコンパイルできる設計。同スクリプトの「コピー
# 手順」節と同じ理由）。
#
# **本スクリプト自体は `SME_PRODUCTION_ENABLED` を切り替えない**。
# after ツリーは呼び出し側が用意する（`docs/perf/cpu-gemm-sme-fmopa-
# microkernel.md` の `on-arm.patch`〈`SME_PRODUCTION_ENABLED` のみ
# `false` → `true` へ反転する計測専用パッチ〉と同型のパッチを適用した
# 別ワークツリーを用意し、`AFTER_TREE` にそのパスを渡すこと）。
#
# 期待行数は `cpu_sme_gate_bit_dump.rs` の `EXPECTED_TOTAL_LINES`
# （ファイル冒頭 doc の内訳と同一。`expected_total_lines_matches_
# documented_breakdown` テストで固定済み）と同じ計算式から機械的に
# 決まる 6,726,847 行（= 512*512 + 1024*1024 + 2048*2048〈gemm。
# 5,505,024〉 + 3 * (1 + 203,530 + 203,530)〈train。1,221,183〉 +
# 64*10〈infer。640〉）。ラベル別の内訳も同じ式から導出して検証する
# （下記 `EXPECTED_*_LINES` 定数）。
#
# **dump 本体（`bitdump/*_bits.txt` 等。数百万行規模）はコミットしない**
# （このディレクトリの `.gitignore` は設けず、本コメントを正とする。
# `bitdump/summary.txt`〈行数・sha256 のみの小さい要約〉もコミット
# 対象外）。
#
# 使い方（GB10 実機。絶対パス必須）:
#   BEFORE_TREE=/home/<user>/work/rust-ai-library-run-2049-before \
#   AFTER_TREE=/home/<user>/work/rust-ai-library-run-2049-after \
#     ./run_bitdump.sh
#
# `--dry-run`: cargo を起動せず、引数検査・期待行数の表示のみ行う。
set -u
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SELF_DIR}/bitdump"
mkdir -p "$OUT_DIR"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

TEST_REL_PATH="crates/facade/tests/cpu_sme_gate_bit_dump.rs"

# gemm512 + gemm1024 + gemm2048
readonly EXPECTED_GEMM_LINES=$((512 * 512 + 1024 * 1024 + 2048 * 2048))
# train: 1 step あたり loss(1) + grad(w1+b1+w2+b2) + param(同数)。
# w1=784*256, b1=256, w2=256*10, b2=10 → 203,530 要素。3 step 分。
readonly EXPECTED_TRAIN_PARAM_ELEMS=$((784 * 256 + 256 + 256 * 10 + 10))
readonly EXPECTED_TRAIN_STEP_LINES=$((1 + EXPECTED_TRAIN_PARAM_ELEMS + EXPECTED_TRAIN_PARAM_ELEMS))
readonly EXPECTED_TRAIN_LINES=$((3 * EXPECTED_TRAIN_STEP_LINES))
# infer: バッチ 64 * D_OUT 10
readonly EXPECTED_INFER_LINES=$((64 * 10))
readonly EXPECTED_BITDUMP_LINES=$((EXPECTED_GEMM_LINES + EXPECTED_TRAIN_LINES + EXPECTED_INFER_LINES))

echo "期待行数: gemm=${EXPECTED_GEMM_LINES} train=${EXPECTED_TRAIN_LINES} infer=${EXPECTED_INFER_LINES} 合計=${EXPECTED_BITDUMP_LINES}" >&2

if [[ "$EXPECTED_BITDUMP_LINES" -ne 6726847 ]]; then
  echo "エラー: 期待行数の算出式が本コメント記載の 6726847 と一致しない（計算結果: ${EXPECTED_BITDUMP_LINES}）" >&2
  exit 1
fi

# `--dry-run` でも BEFORE_TREE／AFTER_TREE の未指定・絶対パス検査は
# 通す（「引数検査」自体が dry-run の目的のため）。実ツリーの存在
# 確認（`Cargo.toml`・テストファイル）と cargo の起動のみ dry-run では
# 省く。
if [[ -z "${BEFORE_TREE:-}" || -z "${AFTER_TREE:-}" ]]; then
  echo "usage: BEFORE_TREE=<absolute path> AFTER_TREE=<absolute path> $0 [--dry-run]" >&2
  exit 1
fi
for v in BEFORE_TREE AFTER_TREE; do
  tree_path="${!v}"
  if [[ "$tree_path" != /* ]]; then
    echo "error: $v must be an absolute path (got: $tree_path)" >&2
    exit 1
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] 引数検査・期待行数の検査のみ完了（cargo は起動していない）" >&2
  exit 0
fi

for v in BEFORE_TREE AFTER_TREE; do
  tree_path="${!v}"
  if [[ ! -f "$tree_path/Cargo.toml" ]]; then
    echo "error: $v/Cargo.toml not found ($tree_path)" >&2
    exit 1
  fi
done

if [[ ! -f "$AFTER_TREE/$TEST_REL_PATH" ]]; then
  echo "error: after ツリーに $TEST_REL_PATH が見つからない（$AFTER_TREE）" >&2
  exit 1
fi
if [[ ! -f "$BEFORE_TREE/$TEST_REL_PATH" ]]; then
  echo "before ツリーに $TEST_REL_PATH が存在しないため after ツリーからコピーする（イシュー #2049 新設ファイル）" >&2
  mkdir -p "$(dirname "$BEFORE_TREE/$TEST_REL_PATH")"
  cp "$AFTER_TREE/$TEST_REL_PATH" "$BEFORE_TREE/$TEST_REL_PATH"
fi

extract_bits_lines() {
  grep -E '^out\['
}

run_arm() { # run_arm <label: before|after> <tree_path>
  # 出力（gemm2048 単体で 419 万行・全体で 672 万行超）をシェル変数へ
  # 抱えると数百 MB のコピーを何度も発生させるため、cargo の標準出力を
  # 直接ファイルへリダイレクトし、以後はファイル間の `grep`／`wc`／
  # `sha256sum`／`diff`／`cmp` のみで処理する（`infer-chain-single-
  # sync-cuda-1689/run_bitdump.sh` は 640 行想定のため変数経由でも
  # 問題なかったが、本スクリプトの規模では変数バッファリングを避ける）。
  local label=$1
  local tree=$2
  echo "[${label}] $tree で dump_cpu_sme_gate_bits を実行する..." >&2
  local rc
  (cd "$tree" && cargo test -p fandhe-ai --release --test cpu_sme_gate_bit_dump \
    -- --ignored --nocapture --exact dump_cpu_sme_gate_bits) >"$OUT_DIR/${label}_raw.log" 2>&1
  rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "エラー: ${label} ツリーの dump_cpu_sme_gate_bits が失敗した（exit ${rc}。$OUT_DIR/${label}_raw.log を参照）" >&2
    exit 1
  fi
  extract_bits_lines <"$OUT_DIR/${label}_raw.log" >"$OUT_DIR/${label}_bits.txt"
  if [[ ! -s "$OUT_DIR/${label}_bits.txt" ]]; then
    echo "エラー: ${label} の出力からビット行を抽出できなかった" >&2
    exit 1
  fi
}

run_arm before "$BEFORE_TREE"
run_arm after "$AFTER_TREE"

before_lines="$(wc -l <"$OUT_DIR/before_bits.txt" | tr -d ' ')"
after_lines="$(wc -l <"$OUT_DIR/after_bits.txt" | tr -d ' ')"
echo "before_lines=$before_lines after_lines=$after_lines expected=$EXPECTED_BITDUMP_LINES" >"$OUT_DIR/line_counts.txt"

# diff が 0（テキストとして一致）でも、before/after が同一箇所で行を
# 欠落させたまま揃っていれば `diff` は差分なしと誤判定しうる
# （`infer-chain-single-sync-cuda-1689/run_bitdump.sh` §末尾と同じ
# 懸念）。成功判定の前に両ファイルが期待行数と一致することを検査する。
if [[ "$before_lines" -ne "$EXPECTED_BITDUMP_LINES" || "$after_lines" -ne "$EXPECTED_BITDUMP_LINES" ]]; then
  echo "判定不能: before_lines=${before_lines} after_lines=${after_lines} が期待行数 ${EXPECTED_BITDUMP_LINES} と一致しない（$OUT_DIR/line_counts.txt を参照）" >&2
  exit 2
fi

# ラベル別の行数・sha256（切り分け用の要約。dump 本体はコミットしない
# ためここへ集約する）。
{
  echo "# cpu_sme_gate_bit_dump ラベル別要約（イシュー #2049）"
  echo
  for label_pattern in 'gemm512' 'gemm1024' 'gemm2048' 'train\.' 'infer'; do
    label_disp="${label_pattern//\\/}"
    before_n="$(grep -cE "^out\[${label_pattern}" "$OUT_DIR/before_bits.txt")"
    after_n="$(grep -cE "^out\[${label_pattern}" "$OUT_DIR/after_bits.txt")"
    before_sha="$(grep -E "^out\[${label_pattern}" "$OUT_DIR/before_bits.txt" | sha256sum | awk '{print $1}')"
    after_sha="$(grep -E "^out\[${label_pattern}" "$OUT_DIR/after_bits.txt" | sha256sum | awk '{print $1}')"
    echo "## ${label_disp}"
    echo "- before: lines=${before_n} sha256=${before_sha}"
    echo "- after:  lines=${after_n} sha256=${after_sha}"
    if [[ "$before_n" -eq "$after_n" && "$before_sha" == "$after_sha" ]]; then
      echo "- 判定: 一致"
    else
      echo "- 判定: **不一致**"
    fi
    echo
  done
} >"$OUT_DIR/summary.txt"

before_sha_all="$(sha256sum "$OUT_DIR/before_bits.txt" | awk '{print $1}')"
after_sha_all="$(sha256sum "$OUT_DIR/after_bits.txt" | awk '{print $1}')"
echo "before_sha256=$before_sha_all after_sha256=$after_sha_all" >>"$OUT_DIR/line_counts.txt"

# diff/cmp の出力も変数へ溜めずファイルへ直接リダイレクトする
# （不一致時は数百万行規模になりうるため。run_arm と同じ方針）。
diff "$OUT_DIR/before_bits.txt" "$OUT_DIR/after_bits.txt" >"$OUT_DIR/bitdump_diff.txt" 2>&1
diff_rc=$?
cmp "$OUT_DIR/before_bits.txt" "$OUT_DIR/after_bits.txt" >"$OUT_DIR/bitdump_cmp.txt" 2>&1
cmp_rc=$?

if [[ "$diff_rc" -eq 0 && "$cmp_rc" -eq 0 ]]; then
  echo "OK: before/after は ${before_lines} 行すべて bit 同一（sha256=${before_sha_all}）" >&2
  exit 0
elif [[ "$diff_rc" -eq 1 || "$cmp_rc" -eq 1 ]]; then
  echo "NG: before/after の出力が bit 同一でない（diff は $OUT_DIR/bitdump_diff.txt・cmp は $OUT_DIR/bitdump_cmp.txt・ラベル別要約は $OUT_DIR/summary.txt を参照）" >&2
  exit 1
else
  echo "エラー: diff／cmp の実行自体が失敗した（diff_rc=${diff_rc}, cmp_rc=${cmp_rc}）。bit 同一性は判定不能" >&2
  exit 3
fi
