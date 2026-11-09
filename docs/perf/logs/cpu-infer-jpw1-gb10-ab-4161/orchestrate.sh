#!/usr/bin/env bash
# #4161 GB10 推論限定 jpw1 A/B の実行キット（<cuda-node> = DGX GB10。この PR 時点では未実行）。
# 役割: RULE.txt に従い、腕ツリー作成 → ビルド（計測前に逐次で全部）→ 前提 report → 5 ラウンド計測を行い、
#   $LOGDIR/{round<i>/{gate.txt,raw/},prereq/,tree-<arm>.txt,tree_verify.txt,patch_sha256.txt} を作る。
#   集計は aggregate.py（計測と分け、計測中に余計な処理を走らせない）。
# 派生元: cpu-small-m-gemm-ab-3201/orchestrate.sh（腕ツリー・path patch・Cargo.lock の退避と復元）と
#   diag-0140-mlp-gb10-4147/run_round.sh（GB10 専有ゲート・実機ロックの作法）。
# 環境変数: BASE=計測対象 tree の展開先（.rev-stamp に完全 SHA。ログとは別の絶対パス）
#           LOGDIR=記録先（RULE.txt と aggregate.py を置いた docs/perf/logs/cpu-infer-jpw1-gb10-ab-4161 の複製）
#           WORK=腕ツリーとビルド成果物の置き場（絶対パス。空のディレクトリ）
#           ARCHIVE=tree_fingerprint を作った git archive 本体（tar。絶対パス）
#           PY=torch 入りの python  BP=bench_py.py の絶対パス
#           HW_LOCK_OWNER_FILE=実機ロックの owner ファイル  HW_LOCK_HELD=1（取得済みの宣言）
# 計測中はビルド・集計・転送を並走させない。run の差し替え・上書きは禁止（既存ファイルがあれば停止）。
set -uo pipefail

validate_abs() { # $1=変数名 $2=値。絶対パスのみ。引用符・バックスラッシュ・改行を拒否（A03）
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
BASE="${BASE:?計測対象 tree}"; LOGDIR="${LOGDIR:?記録先}"; WORK="${WORK:?作業ディレクトリ}"
ARCHIVE="${ARCHIVE:?git archive の tar}"; PY="${PY:?torch 入り python}"; BP="${BP:?bench_py.py}"; LOCKF="${HW_LOCK_OWNER_FILE:?実機ロックの owner ファイル}"
for v in BASE ARCHIVE LOGDIR WORK PY BP LOCKF; do validate_abs "${v}" "${!v}"; done
[ "${HW_LOCK_HELD:-}" = "1" ] || { echo "実機ロック未取得（HW_LOCK_HELD=1 が必要）"; exit 2; }
[ -z "${RAYON_NUM_THREADS:-}" ] || { echo "RAYON_NUM_THREADS が設定されている（拒否）"; exit 2; }
[ -z "${GEMM_GATE_PATCH_FACADE_PATH:-}" ] || { echo "GEMM_GATE_PATCH_FACADE_PATH が非空（拒否）"; exit 2; }

RULE="${LOGDIR}/RULE.txt"
PATCH="${LOGDIR}/ab_patch_inf_jpw1.diff"
ARMS=(off inf-jpw1)
ROUNDS=5
FCREL="scripts/bench/framework-compare"

# ---- 前提検査（fail-closed）
[ -s "${RULE}" ] || { echo "RULE.txt が無い（先に単独コミットで固定する）"; exit 1; }
if grep -E '^  [A-Za-z0-9_-]+:.*<[^>]*>' "${RULE}" > /dev/null; then echo "RULE.txt にプレースホルダが残っている"; exit 1; fi
SHA="$(sed -n 's/^  tree_sha: *//p' "${RULE}" | head -n1)"
FP="$(sed -n 's/^  tree_fingerprint: *//p' "${RULE}" | head -n1)"
[ -n "${SHA}" ] && [ -n "${FP}" ] || { echo "RULE.txt の tree_sha / tree_fingerprint が空"; exit 1; }
[ "$(cat "${BASE}/.rev-stamp")" = "${SHA}" ] || { echo ".rev-stamp が RULE と不一致"; exit 1; }
# fingerprint = アーカイブ本体の sha256。さらに展開内容が BASE と一致することを複製・ビルド前に確かめる。
[ "$(sha256sum "${ARCHIVE}" | awk '{print $1}')" = "${FP}" ] || { echo "ARCHIVE の sha256 が tree_fingerprint と不一致"; exit 1; }
VER="$(mktemp -d)" || exit 1
tar -C "${VER}" -xf "${ARCHIVE}" || { rm -rf "${VER}"; echo "ARCHIVE の展開失敗"; exit 1; }
diff -r -x target -x .git -x .rev-stamp "${VER}" "${BASE}" > /dev/null || { rm -rf "${VER}"; echo "BASE の内容が ARCHIVE と不一致"; exit 1; }
rm -rf "${VER}"
[ -f "${PATCH}" ] || { echo "patch が無い"; exit 1; }
[ ! -e "${WORK}/trees" ] || { echo "${WORK}/trees が既にある（上書き禁止）"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'GB10' || { echo "機体が GB10 ではない"; exit 1; }
[ -s "${LOCKF}" ] || { echo "実機ロックの owner が空"; exit 1; }
python3 -I "${LOGDIR}/aggregate.py" --self-test || { echo "aggregate.py の self-test 失敗"; exit 1; }
"${PY}" -c 'import torch,sys; sys.exit(0 if torch.__version__.startswith("2.14.0") else 1)' || { echo "torch の版が違う"; exit 1; }

# ---- 腕ツリーの作成（git archive の複製に patch を当てる。BASE 自体は触らない）
mkdir -p "${WORK}/trees" "${LOGDIR}/prereq"
sha_of() { sha256sum "$1" | awk '{print $1}'; }
for arm in "${ARMS[@]}"; do
  T="${WORK}/trees/${arm}"
  mkdir -p "${T}"
  tar -C "${BASE}" --exclude=target --exclude=.git -cf - . | tar -C "${T}" -xf - || { echo "複製失敗 ${arm}"; exit 1; }
  if [ "${arm}" = "inf-jpw1" ]; then
    (cd "${T}" && git apply --check "${PATCH}" 2>/dev/null || patch -p1 --dry-run < "${PATCH}" > /dev/null) || { echo "patch が当たらない"; exit 1; }
    (cd "${T}" && patch -p1 < "${PATCH}" > /dev/null) || { echo "patch 適用失敗"; exit 1; }
    echo "$(sha_of "${PATCH}")  ab_patch_inf_jpw1.diff" > "${LOGDIR}/patch_sha256.txt"
  fi
  grep -n 'INFER_SCOPE_JPW1_ENABLED: bool' "${T}/crates/backend-cpu/src/gemm_blis/infer_scope.rs" >> "${LOGDIR}/tree_verify.txt"
done

# ---- ビルド（計測前に逐次で全部）。Cargo.lock は退避して必ず復元する。
FCLOCK="${WORK}/Cargo.lock.fc.orig"
restore_lock() { [ -f "${FCLOCK}" ] && cp "${FCLOCK}" "${BASE}/${FCREL}/Cargo.lock" 2>/dev/null; true; }
trap restore_lock EXIT
# 固定依存の比較は 2 つに分ける。path patch で registry 由来から path 由来へ置換される自作クレート（fandhe-ai 系）は
# 置換で source/checksum が消えるのが正常。(1) 自作クレート以外は name/version/checksum がビルド前後で完全一致、
# (2) 自作クレートは置換後に source/checksum を持たない（= path 由来）ことだけを検証する。
lock_pins() { awk '/^\[\[package\]\]/{n="";v=""} /^name = /{n=$3} /^version = /{v=$3} /^checksum = /{ if (n !~ /^"fandhe-ai(-|"$)/) print n, v, $3}' "$1" | sort; }
own_pkgs_not_path() { # 置換後 lock で source/checksum を持つ fandhe-ai 系パッケージ名（空なら全て path 由来）
  awk '/^\[\[package\]\]/{if(n ~ /^"fandhe-ai(-|"$)/ && bad) print n; n="";bad=0} /^name = /{n=$3} /^(source|checksum) = /{bad=1} END{if(n ~ /^"fandhe-ai(-|"$)/ && bad) print n}' "$1"
}
cp "${BASE}/${FCREL}/Cargo.lock" "${FCLOCK}" || { echo "Cargo.lock の退避失敗"; exit 1; }
for arm in "${ARMS[@]}"; do
  T="${WORK}/trees/${arm}"
  PATCHCFG="patch.crates-io.fandhe-ai.path=\"${T}/crates/facade\""
  # 複製元の Cargo.lock は fandhe-ai を registry 固定にしており、path patch と --locked は衝突する。
  # 先に patch 反映の lockfile を（--locked なしで）生成し、registry 由来の固定が動いていないことを確かめてから固定ビルドする。
  (cd "${T}/${FCREL}" && cargo metadata --format-version 1 --config "${PATCHCFG}" > /dev/null 2> "${LOGDIR}/prereq/lock-${arm}.log") || { echo "lockfile 生成失敗 ${arm}"; exit 1; }
  [ "$(lock_pins "${FCLOCK}")" = "$(lock_pins "${T}/${FCREL}/Cargo.lock")" ] || { echo "lockfile の registry 固定が変化した ${arm}"; exit 1; }
  [ -z "$(own_pkgs_not_path "${T}/${FCREL}/Cargo.lock")" ] || { echo "自作クレートが path 置換されていない ${arm}"; exit 1; }
  grep -q '^name = "fandhe-ai"$' "${T}/${FCREL}/Cargo.lock" || { echo "fandhe-ai が lock に無い ${arm}"; exit 1; }
  (cd "${T}/${FCREL}" && cargo build --release --locked -p bench-fandhe --config "${PATCHCFG}" \
      >"${LOGDIR}/prereq/build-${arm}.log" 2>&1) || { echo "bench-fandhe ビルド失敗 ${arm}"; exit 1; }
  cp "${T}/${FCREL}/target/release/bench-fandhe" "${WORK}/bench-fandhe-${arm}" || exit 1
  (cd "${T}/${FCREL}" && cargo tree --locked --config "${PATCHCFG}" -p bench-fandhe --depth 1 > "${LOGDIR}/tree-${arm}.txt" 2>&1)
  # 機構発火の前提 report（腕ごとに 1 件だけ --exact。全件実行はしない）
  (cd "${T}" && cargo test --release -p fandhe-ai --test infer_scope_jpw1_4161 -- --ignored --exact --nocapture report_infer_jpw1_probe \
      > "${LOGDIR}/prereq/${arm}.log" 2>&1) || { echo "前提 report 失敗 ${arm}"; exit 1; }
done
restore_lock

# ---- ゲート（4147 の run_round.sh と同じ条件）
gate() { # $1=記録先 round ディレクトリ
  local rd="$1" i=0 apps util l zero=0
  while [ "${i}" -le 60 ]; do
    if ! apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1); then echo "INVALID nvidia-smi-apps-failed" > "${rd}/gate.txt"; return 2; fi
    if ! util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>&1); then echo "INVALID nvidia-smi-util-failed" > "${rd}/gate.txt"; return 2; fi
    l=$(awk '{print $1}' /proc/loadavg)
    if ! printf '%s' "${l}" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then echo "INVALID load1-not-numeric" > "${rd}/gate.txt"; return 2; fi
    if [ -z "${apps}" ] && [ "$(printf '%s' "${util}" | tr -d ' ')" = "0" ] && awk -v x="${l}" 'BEGIN{exit !(x < 1.0)}'; then
      zero=$((zero + 1))
    else
      zero=0
    fi
    if [ "${zero}" -ge 3 ]; then echo "PASS load1=${l} compute-apps=0 gpu-util=0x3 hw-lock=held $(date -u +%FT%TZ)" > "${rd}/gate.txt"; return 0; fi
    sleep 30; i=$((i + 1))
  done
  echo "TIMEOUT load1=${l} $(date -u +%FT%TZ)" > "${rd}/gate.txt"; return 1
}

# ---- 5 ラウンド（腕順はラウンドごとにローテーション）
for r in $(seq 1 "${ROUNDS}"); do
  RD="${LOGDIR}/round${r}"
  [ ! -e "${RD}/raw" ] || { echo "${RD}/raw が既にある（上書き禁止）"; exit 2; }
  mkdir -p "${RD}/raw"
  gate "${RD}" || { echo "ラウンド ${r} のゲート不通過: $(cat "${RD}/gate.txt")（正式系列から外れる）"; continue; }
  if [ $((r % 2)) -eq 1 ]; then ORDER=(off inf-jpw1); else ORDER=(inf-jpw1 off); fi
  for arm in "${ORDER[@]}"; do
    for t in train infer; do for m in fresh reuse; do
      "${WORK}/bench-fandhe-${arm}" --task "${t}" --device cpu --size 64 --mode "${m}" \
        --out "${RD}/raw/${arm}-${t}-${m}.jsonl" 2>> "${RD}/raw/${arm}.err" \
        || { echo "bench-fandhe ${arm} ${t} ${m} 失敗"; exit 1; }
    done; done
  done
  for t in train infer; do
    "${PY}" "${BP}" --framework pytorch --task "${t}" --device cpu --size 64 --out "${RD}/raw/pytorch-${t}-fresh.jsonl" \
      > /dev/null 2>> "${RD}/raw/pytorch.err" || { echo "bench_py cpu ${t} 失敗"; exit 1; }
  done
  echo "round ${r} done $(date -u +%FT%TZ)"
done
echo "計測完了。次に python3 -I ${LOGDIR}/aggregate.py ${LOGDIR}"
