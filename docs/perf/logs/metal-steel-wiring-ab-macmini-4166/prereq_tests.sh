#!/bin/bash
# #4166 の前提の確認（RULE.txt 項 3）。Mac mini 上で、formal-1 の前に 1 回だけ、hw-lock の下で流す。値は記録しない。
# 対象は W/after（clean な 9eba21749。定数 true）。すべて --release --all-features（required-features で
# ゲートされたテストが黙ってビルドされない false-green を避ける。Makefile の test-ignored-metal と同じ）。
#   1. lib: gemm_steel_large_n_diag_tests::steel_large_n_default_is_on と ::steel_large_n_reach_trace（--exact）
#   2. lib: gemm_tile_macmini_table_diag_tests:: の ignored 全件
#   3. tests/ の名前に parity を含む 59 本のテストターゲットの ignored 全件
# 各単位で、先に `--ignored --list` で件数を数え、本実行の `test result: ok. <その件数> passed; 0 failed` を要求する
# （件数 0 の単位は「ignored テスト 0 件」として記録し、失敗にしない）。終了コードが 0 でなければ失敗。
# hw-lock が取れなければ W/prereq を作らずに終了コード 5 で終わり、W/prereq-attempts.log に lock-busy を残す（再試行できる）。
# 出力: W/prereq/summary.tsv（単位・件数・passed・failed・rc・結果）と W/prereq/result.txt（all-pass か fail）。
# ログは W/prereq/*.log に残すが、記録 PR には summary と result だけを収める（値は記録しない）。
# 使い方: bash prereq_tests.sh <W>
set -u
W=$1
P="${W}/prereq"
LOCK="${HOME}/work/.fandhe-hw-lock"
PKG=fandhe-ai-backend-metal
PARITY="adam_device_parity batch_norm_parity bce_parity cast_parity constant_pad_parity cpu_metal_f16_parity
cpu_metal_f16_tiled_parity cpu_metal_parity elementwise_device_parity fused_elementwise_parity gather_scatter_parity
gemm_auto_parity gemm_batched_parity gemm_bias_act_parity gemm_device_input_parity gemm_dynamic_tile_parity
gemm_f16_auto_parity gemm_fp32_strict_into_parity gemm_hfrag_parity gemm_naive_parity gemm_resident_parity
gemm_simdgroup_parity gemm_splitk_auto_entry_parity gemm_splitk_parity gemm_strided_parity gemm_te_parity
gemm_transposed_parity grouped_optimizer_step_parity huber_parity im2col_col2im_parity interpolate_parity
kl_div_parity layer_norm_parity linalg_model_parity linalg_parity linalg_parity_inputs linalg_parity_inputs_guard
linear_forward_device_parity log_softmax_backward_parity mse_device_parity mse_parity nll_parity norm_backward_parity
pooling_parity reduce_parity relu_mask_device_parity rmsnorm_parity rnn_cell_parity scalar_op_parity scan_parity
sgd_device_parity softmax_parity sort_topk_parity splitk_parity_baseline_contract train_forward_encode_only_parity
typed_ops_bf16_parity typed_ops_f16_parity unique_parity where_masked_fill_parity"
utc() { date -u +%FT%TZ; }
[ -d "${W}/after" ] || { echo "error: ${W}/after が無い" >&2; exit 2; }
[ -e "${P}" ] && { echo "error: 既にある: ${P}" >&2; exit 2; }
n=0; for _ in ${PARITY}; do n=$((n + 1)); done
[ "${n}" = 59 ] || { echo "error: parity の対象が 59 本ではない: ${n}" >&2; exit 2; }
# hw-lock が取れなかった試みは W/prereq を作らず、W/prereq-attempts.log にだけ残す（ロックが外れた後に同じ展開で再試行できる）。
if ! mkdir "${LOCK}" 2>/dev/null; then
  echo "lock-busy owner=$(cat "${LOCK}/owner" 2>/dev/null) $(utc)" >> "${W}/prereq-attempts.log"
  echo "error: hw-lock が取れない（W/prereq は作っていない。ロックが外れたら再実行できる）" >&2
  exit 5
fi
release() { rm -f "${LOCK}/owner"; rmdir "${LOCK}" 2>/dev/null; }
trap release EXIT
printf '#4166 prereq %s\n' "$(utc)" > "${LOCK}/owner"
echo "lock-acquired $(utc)" >> "${W}/prereq-attempts.log"
mkdir "${P}" || { echo "error: ${P} を作れない（既にある）" >&2; exit 2; }
export CARGO_TARGET_DIR="${W}/target-prereq"
unset RAYON_NUM_THREADS
cd "${W}/after" || exit 2
printf 'unit\tlisted\tpassed\tfailed\trc\tresult\n' > "${P}/summary.tsv"
ALL=0
# run_unit <単位名> <cargo test の対象指定...> -- <libtest の引数...>
run_unit() {
  local name=$1; shift
  local sel=() lt=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do sel+=("$1"); shift; done
  shift
  lt=("$@")
  local log="${P}/${name}.log" listed passed failed rc res
  cargo test -p "${PKG}" --release --all-features "${sel[@]}" -- --ignored --list ${lt[@]+"${lt[@]}"} > "${log}.list" 2> "${log}.list.err"
  rc=$?
  listed=$(grep -c ': test$' "${log}.list")
  if [ "${rc}" != 0 ]; then
    res=fail-list
  else
    cargo test -p "${PKG}" --release --all-features "${sel[@]}" -- --ignored --test-threads=1 ${lt[@]+"${lt[@]}"} > "${log}" 2>&1
    rc=$?
    local line
    line=$(grep -E '^test result: ' "${log}" | tail -1)
    passed=$(echo "${line}" | sed -n 's/.* \([0-9][0-9]*\) passed;.*/\1/p')
    failed=$(echo "${line}" | sed -n 's/.* \([0-9][0-9]*\) failed;.*/\1/p')
    if [ "${rc}" = 0 ] && [ "$(grep -cE '^test result: ' "${log}")" = 1 ] && [ "${passed}" = "${listed}" ] && [ "${failed}" = 0 ]; then
      if [ "${listed}" = 0 ]; then res=pass-no-ignored; else res=pass; fi
    else
      res=fail
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${name}" "${listed}" "${passed:-NA}" "${failed:-NA}" "${rc}" "${res}" >> "${P}/summary.tsv"
  case "${res}" in pass|pass-no-ignored) ;; *) ALL=1 ;; esac
}
echo "begin $(utc) rustc=$(rustc -V) macos=$(sw_vers -productVersion) hw_model=$(sysctl -n hw.model)" > "${P}/env.txt"
run_unit steel_large_n --lib -- --exact gemm_steel_large_n_diag_tests::steel_large_n_default_is_on gemm_steel_large_n_diag_tests::steel_large_n_reach_trace
run_unit macmini_table --lib -- gemm_tile_macmini_table_diag_tests::
for t in ${PARITY}; do
  run_unit "${t}" --test "${t}" --
done
# 1 の 2 件は必ず数えられていること（--exact の名前違いで 0 件になる false-green を防ぐ）。
if [ "$(awk -F'\t' '$1=="steel_large_n"{print $2}' "${P}/summary.tsv")" != 2 ]; then ALL=1; fi
if [ "$(awk -F'\t' '$1=="macmini_table"{print $2}' "${P}/summary.tsv")" = 0 ]; then ALL=1; fi
echo "end $(utc)" >> "${P}/env.txt"
if [ "${ALL}" = 0 ]; then echo "all-pass $(utc)" > "${P}/result.txt"; else echo "fail $(utc)" > "${P}/result.txt"; fi
cat "${P}/result.txt"
exit "${ALL}"
