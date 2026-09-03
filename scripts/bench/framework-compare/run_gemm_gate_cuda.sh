#!/bin/bash
# GEMM 目標達成ゲート（#1031）の 5 回計測スイープ（イシュー #1142）。
#
# `run_all_cuda.sh` は GEMM を N ごとに 1 回しか起動しないため、そのままでは
# coding-rust.md「ベンチは 5 回計測の中央値」を満たせない。本スクリプトは
# N=1024/2048/4096 それぞれについて `bench-fandhe gemm cuda <N> reuse` と
# `bench-candle gemm cuda <N> fresh`（candle は reuse 非対応。README「計測
# プロトコル」節）を交互に 5 回ずつ起動し、`compare_gemm_gate.py` が run 間
# 中央値を算出できる形で JSONL に記録する（fandhe/candle を run 内で交互に
# 実行し、熱・クロック状態の偏りを両フレームワークで揃える）。
#
# 呼び出し例（GB10 実機。ユーザー承認・別セッション）:
#   # 正式系列（承認済みピン。registry 解決のまま計測）
#   bash run_gemm_gate_cuda.sh 0.6.0
#   # 参考系列（次回 crates.io 公開前の見込み値。crates/facade へ path 差し替え
#   # てビルド＋計測を本スクリプト内の 1 invocation で完結させる）
#   GEMM_GATE_PATCH_FACADE_PATH="$HOME/work/rust-ai-library-run/crates/facade" \
#     bash run_gemm_gate_cuda.sh head-<short sha>
#
# バイナリ同一性検証（イシュー #1166 codex-review 指摘。PRRT_kwDOTuUCJc6euL3W）:
# 過去に、参考系列ビルド確認のため引数なし `cargo tree` を挟んだ結果 Cargo.lock
# が registry 解決へ暗黙に再ロックされ、後続の `run_gemm_gate_cuda.sh`（ビルド
# 省略なし）が意図せず registry 版 binary へ差し替えて計測してしまう事故が
# 実際に発生した（`docs/perf/logs/cuda-gemm-candle-gate-1142/env_info.txt`
# 「参考系列ビルドの事故と対処」節）。ラベルだけでは系列を機械的に識別でき
# ないため、本スクリプトは以下の 2 点で対処する:
#   1. `GEMM_GATE_PATCH_FACADE_PATH` を指定すると、参考系列のビルド（`--config
#      patch.crates-io.fandhe-ai.path=...`）と計測を本スクリプト内の同一
#      invocation で不可分に実行する（ビルドと計測の間に任意の `cargo`
#      コマンドが割り込む窓を作らない。外部での事前ビルド＋
#      `GEMM_GATE_SKIP_BUILD=1` の 2 段構成は廃止）
#   2. ビルド直後に `target/release/bench-fandhe` の sha256 と依存解決元
#      （`cargo tree -p bench-fandhe --depth 1` の `fandhe-ai` 行。path か
#      registry か）を `results/raw/manifest-dgx-gemm-gate-<label>.jsonl`
#      へ記録し、計測ループ開始直前に再計算した sha256 と突き合わせる
#      （fail-closed。不一致・manifest 欠落なら測定を一切実行せず exit 1）
#
# GEMM_GATE_SKIP_BUILD=1 は「同一 label で直前に成功した本スクリプト実行が
# 残した manifest とバイナリが一致する場合に限り」ビルドを省略する用途
# （失敗 run の再実行等）。manifest が無い・一致しない場合は測定せず fail-closed
# で終了する（すり替わった binary で判定不能な性能値を確定させない。
# security.md A08）。
#
# 出力は `run_all_cuda.sh` と同じ「失敗を捏造しない」方針（skipped ログへ
# 記録。security.md A08）。
set -u
cd "$(dirname "$0")"

LABEL=${1:-}

# A03 インジェクション対策: ラベルはファイル名へ直接埋め込むため、
# 英数字・`._-` のみを許可する allowlist で検証する（run_ab_train_cuda.sh
# と同じ方針）。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. 0.6.0 or head-abc1234)" >&2
  echo "  optional: GEMM_GATE_PATCH_FACADE_PATH=<crates/facade 絶対パス> でビルド時に patch.crates-io.fandhe-ai.path を適用（参考系列）" >&2
  exit 1
fi

OUT="results/raw/results-dgx-gemm-gate-${LABEL}.jsonl"
SKIP="results/raw/skipped-dgx-gemm-gate-${LABEL}.log"
MANIFEST="results/raw/manifest-dgx-gemm-gate-${LABEL}.json"
mkdir -p results/raw
: > "$OUT"
: > "$SKIP"

ANY_FAILED=0

# 測定対象バイナリの sha256 を計算する（GNU coreutils の sha256sum を優先し、
# macOS 等の非搭載環境では shasum -a 256 へフォールバックする）。
sha256_of() {
  local f=$1
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  else
    shasum -a 256 "$f" | awk '{print $1}'
  fi
}

# `cargo tree -p bench-fandhe --depth 1` の `fandhe-ai` 依存行から解決元
# （path 差し替えか registry か）を抽出する。path 解決時は crate 名の直後に
# `(<絶対パス>)` が付く（cargo 標準表示）。呼び出し時は必ずビルドと同じ
# `${CARGO_CONFIG_ARGS[@]}`（`GEMM_GATE_PATCH_FACADE_PATH` 指定時の
# `--config patch.crates-io.fandhe-ai.path=...`）を渡すこと。引数なしで
# 呼ぶと path patch が適用されないまま解決され、path ビルドでも registry と
# 誤記録しうる（#1166 codex-review 指摘・Cursor Bugbot 指摘）。
# 取得失敗（fandhe-ai 行が見当たらない）は戻り値 1 を返す（呼び出し側で
# fail-closed に扱う。"unknown" を返して fail-open させない。security.md A08）。
fandhe_ai_source_desc() {
  local line path_part
  line=$(cargo tree -p bench-fandhe --depth 1 "$@" 2>/dev/null | grep -E '^[├└]── fandhe-ai ' || true)
  if [[ -z "$line" ]]; then
    return 1
  elif [[ "$line" == *"("* ]]; then
    path_part=${line#*(}
    path_part=${path_part%)}
    echo "path:${path_part}"
  else
    echo "registry"
  fi
}

# ビルド直後に呼び、`target/release/bench-fandhe` の sha256・依存解決元を
# manifest へ記録する（fail-closed 検証の基準値。#1166 対応）。
# 依存解決元は `${CARGO_CONFIG_ARGS[@]}`（ビルド時と同一の `--config`）を
# 付けた `cargo tree` で取得し、取得失敗時・現在の invocation の意図
# （`GEMM_GATE_PATCH_FACADE_PATH` の有無＝参考系列か正式系列か）と矛盾する
# 場合は manifest を書かず fail-closed で終了する（#1166 codex-review 指摘・
# Cursor Bugbot 指摘: ラベルだけでは系列を機械的に識別できないため、記録の
# 時点でも系列契約を照合する）。
record_manifest() {
  local fandhe_sha source_desc
  if [[ ! -x "./target/release/bench-fandhe" ]]; then
    echo "manifest 記録失敗: ./target/release/bench-fandhe が見つからない" >&2
    exit 1
  fi
  fandhe_sha=$(sha256_of "./target/release/bench-fandhe")
  if ! source_desc=$(fandhe_ai_source_desc "${CARGO_CONFIG_ARGS[@]}"); then
    echo "ERROR: fandhe-ai の依存解決元を cargo tree から取得できない" \
         "('cargo tree -p bench-fandhe --depth 1' に fandhe-ai 行が見当たらない)。" >&2
    echo "  依存元不明のまま manifest を記録すると、後続の verify_manifest が" >&2
    echo "  系列（registry/path）の不整合を検出できなくなる。性能値を捏造しない" >&2
    echo "  ため測定を中止する（fail-closed。security.md A08）。" >&2
    exit 1
  fi
  if [[ -n "${GEMM_GATE_PATCH_FACADE_PATH:-}" ]]; then
    if [[ "$source_desc" != "path:${GEMM_GATE_PATCH_FACADE_PATH}" ]]; then
      echo "ERROR: GEMM_GATE_PATCH_FACADE_PATH 指定時（参考系列）は fandhe_ai_source が" >&2
      echo "  'path:${GEMM_GATE_PATCH_FACADE_PATH}' である必要があるが actual=$source_desc。" >&2
      echo "  ビルドと cargo tree で解決結果が食い違っている（fail-closed。#1166）。" >&2
      exit 1
    fi
  elif [[ "$source_desc" != "registry" ]]; then
    echo "ERROR: GEMM_GATE_PATCH_FACADE_PATH 未指定時（正式系列）は fandhe_ai_source が" >&2
    echo "  'registry' である必要があるが actual=$source_desc。" >&2
    echo "  正式系列（label '${LABEL}'）として結果を確定させる場合は依存解決が" >&2
    echo "  registry になるよう Cargo.lock・環境を確認すること（fail-closed。#1166）。" >&2
    exit 1
  fi
  cat > "$MANIFEST" <<JSON
{"label":"${LABEL}","bench_fandhe_sha256":"${fandhe_sha}","fandhe_ai_source":"${source_desc}","recorded_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
  echo "== manifest 記録: $MANIFEST =="
  cat "$MANIFEST"
}

# 計測ループ開始直前に呼び、現在の `target/release/bench-fandhe` の sha256 が
# manifest 記録値と一致するか検証する（fail-closed。不一致・manifest 欠落・
# バイナリ欠落はいずれも測定を実行せず exit 1）。
verify_manifest() {
  if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: $MANIFEST が見つからない。GEMM_GATE_SKIP_BUILD=1 は同一 label での" >&2
    echo "  直前の成功実行が記録した manifest（バイナリ sha256 検証の基準値）を前提と" >&2
    echo "  する。GEMM_GATE_SKIP_BUILD を外して再実行するか、参考系列は" >&2
    echo "  GEMM_GATE_PATCH_FACADE_PATH=<facade 絶対パス> bash $0 <label> で" >&2
    echo "  ビルドと計測を 1 invocation で実行すること（#1166）。" >&2
    exit 1
  fi
  if [[ ! -x "./target/release/bench-fandhe" ]]; then
    echo "ERROR: ./target/release/bench-fandhe が見つからない。" >&2
    exit 1
  fi
  local expected actual
  expected=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('bench_fandhe_sha256',''))" "$MANIFEST" 2>/dev/null || true)
  actual=$(sha256_of "./target/release/bench-fandhe")
  if [[ -z "$expected" || "$expected" != "$actual" ]]; then
    echo "ERROR: bench-fandhe のバイナリ sha256 が manifest ($MANIFEST) と不一致。" >&2
    echo "  expected=${expected:-<なし>} actual=$actual" >&2
    echo "  label '${LABEL}' で最後に検証された時点から binary が別物へ差し替わって" >&2
    echo "  いる可能性がある（例: 素の 'cargo build'/'cargo tree' が Cargo.lock を" >&2
    echo "  registry 解決へ暗黙に再ロックし、意図しない登録版 binary へ差し替えた" >&2
    echo "  事故。docs/perf/logs/cuda-gemm-candle-gate-1142/env_info.txt 参照）。" >&2
    echo "  性能値を捏造しないため測定を中止する（fail-closed。security.md A08）。" >&2
    exit 1
  fi
  echo "== バイナリ sha256 検証 OK: bench-fandhe=${actual}（label '${LABEL}'） =="

  # 依存解決元（registry/path）の照合。ビルドと同じ `${CARGO_CONFIG_ARGS[@]}`
  # で `cargo tree` を実行し、manifest 記録値・現在の invocation の意図
  # （`GEMM_GATE_PATCH_FACADE_PATH` の有無）の双方と一致するかを検証する。
  # `verify_manifest` は sha256 一致（バイナリの同一性）のみを見ており、
  # manifest の `label`・`fandhe_ai_source` と「正式系列=registry／参考系列=
  # 指定 path」という計測契約そのものは照合していなかった（#1166
  # codex-review 指摘・Cursor Bugbot 指摘）。sha256 が一致していても、
  # manifest 記録後に `GEMM_GATE_SKIP_BUILD=1` 付きで
  # `GEMM_GATE_PATCH_FACADE_PATH` の有無を変えて再実行すれば系列の取り違え
  # が起こりうるため、ここで明示的に塞ぐ。
  local expected_source actual_source
  expected_source=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('fandhe_ai_source',''))" "$MANIFEST" 2>/dev/null || true)
  if ! actual_source=$(fandhe_ai_source_desc "${CARGO_CONFIG_ARGS[@]}"); then
    echo "ERROR: 検証時点で fandhe-ai の依存解決元を cargo tree から取得できない。" >&2
    echo "  性能値を捏造しないため測定を中止する（fail-closed。security.md A08）。" >&2
    exit 1
  fi
  if [[ -z "$expected_source" || "$expected_source" != "$actual_source" ]]; then
    echo "ERROR: fandhe_ai_source が manifest ($MANIFEST) と不一致。" >&2
    echo "  expected=${expected_source:-<なし>} actual=$actual_source" >&2
    echo "  性能値を捏造しないため測定を中止する（fail-closed。security.md A08）。" >&2
    exit 1
  fi
  if [[ -n "${GEMM_GATE_PATCH_FACADE_PATH:-}" ]]; then
    if [[ "$actual_source" != "path:${GEMM_GATE_PATCH_FACADE_PATH}" ]]; then
      echo "ERROR: GEMM_GATE_PATCH_FACADE_PATH 指定時（参考系列）は fandhe_ai_source が" >&2
      echo "  'path:${GEMM_GATE_PATCH_FACADE_PATH}' である必要があるが actual=$actual_source。" >&2
      echo "  label '${LABEL}' を参考系列として計測する意図と manifest の記録内容が" >&2
      echo "  食い違っている（fail-closed。#1166）。" >&2
      exit 1
    fi
  elif [[ "$actual_source" != "registry" ]]; then
    echo "ERROR: GEMM_GATE_PATCH_FACADE_PATH 未指定時（正式系列）は fandhe_ai_source が" >&2
    echo "  'registry' である必要があるが actual=$actual_source。" >&2
    echo "  label '${LABEL}' を正式系列（README「ファイル名ラベルが唯一の系列識別" >&2
    echo "  手段」契約）として結果を確定させることはできない（fail-closed。#1166）。" >&2
    exit 1
  fi
  echo "== 依存元検証 OK: fandhe_ai_source=${actual_source}（label '${LABEL}'） =="
}

run() { # run <binary> <task> <device> <size> [mode] [extra_flag]
  local bin=$1 task=$2 device=$3 size=$4 mode=${5:-fresh} extra_flag=${6:-}
  echo "== $bin $task $device size=$size mode=$mode extra=${extra_flag:-none} =="
  if ! "./target/release/$bin" --task "$task" --device "$device" --size "$size" --mode "$mode" ${extra_flag:+"$extra_flag"} --out "$OUT" 2>err.tmp; then
    echo "$bin task=$task device=$device size=$size mode=$mode extra=${extra_flag:-none} : $(cat err.tmp)" >> "$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f err.tmp
}

# `GEMM_GATE_PATCH_FACADE_PATH` 由来の `--config` 引数を、ビルド（分岐の
# 内側）と依存解決元の検証（`record_manifest`/`verify_manifest`。ビルド
# 分岐・SKIP_BUILD 分岐どちらでも実行される）の両方で同一に使い回すため、
# 分岐に入る前に 1 箇所で組み立てる。ここで組み立てず `fandhe_ai_source_desc`
# を引数なしで呼ぶと、path patch 適用ビルドでも `cargo tree` 側だけ patch
# なしで解決され依存元を誤記録しうる（#1166 codex-review 指摘・Cursor
# Bugbot 指摘）。
CARGO_CONFIG_ARGS=()
if [[ -n "${GEMM_GATE_PATCH_FACADE_PATH:-}" ]]; then
  # 参考系列: crates/facade（HEAD ツリー）へ path 差し替えてビルド・依存解決
  # する。`--config` は本 invocation にのみ適用され Cargo.lock へ永続化され
  # ない（[patch] セクション・.cargo/config.toml は一切コミットしない。
  # deps-policy.md 第 9 区分「承認済みピンの完全固定」を壊さないため）。
  # TOML 文字列値として埋め込むため、二重引用符・バックスラッシュを含む
  # 値は不正な `--config` を生成しうるので拒否する（A03 インジェクション
  # 対策。ラベルと同じ fail-closed 方針）。
  if [[ "$GEMM_GATE_PATCH_FACADE_PATH" == *'"'* || "$GEMM_GATE_PATCH_FACADE_PATH" == *'\'* ]]; then
    echo "ERROR: GEMM_GATE_PATCH_FACADE_PATH に '\"' または '\\' を含めることはできない" >&2
    exit 1
  fi
  echo "   (GEMM_GATE_PATCH_FACADE_PATH=${GEMM_GATE_PATCH_FACADE_PATH})"
  CARGO_CONFIG_ARGS+=(--config "patch.crates-io.fandhe-ai.path=\"${GEMM_GATE_PATCH_FACADE_PATH}\"")
fi

if [[ "${GEMM_GATE_SKIP_BUILD:-0}" != "1" ]]; then
  echo "== build bench-fandhe =="
  FANDHE_BUILD_ARGS=(build --release -p bench-fandhe "${CARGO_CONFIG_ARGS[@]}")
  if ! cargo "${FANDHE_BUILD_ARGS[@]}" 2>build-err.tmp; then
    tail -40 build-err.tmp
    echo "bench-fandhe BUILD FAILED: $(tail -3 build-err.tmp | tr '\n' ' ')" >> "$SKIP"
    rm -f build-err.tmp
    echo "done. results in $OUT ; failures (if any) in $SKIP"
    exit 1
  fi
  rm -f build-err.tmp

  # ビルド直後（他の cargo コマンドを一切挟まず）に manifest を記録する。
  # 参考系列ビルドの事故（cargo tree 挿入で Cargo.lock が registry 解決へ
  # 暗黙に再ロックされた事例）を踏まえ、bench-candle のビルド前に基準値を
  # 固定する。
  record_manifest

  echo "== build bench-candle (cuda) =="
  if ! cargo build --release -p bench-candle --no-default-features --features cuda 2>build-err.tmp; then
    tail -40 build-err.tmp
    echo "bench-candle BUILD FAILED: $(tail -3 build-err.tmp | tr '\n' ' ')" >> "$SKIP"
    rm -f build-err.tmp
    echo "done. results in $OUT ; failures (if any) in $SKIP"
    exit 1
  fi
  rm -f build-err.tmp
else
  echo "== GEMM_GATE_SKIP_BUILD=1: ビルドを省略（既存 ./target/release バイナリを使用） =="
fi

# 計測ループ開始直前に、bench-fandhe binary が manifest 記録値と一致することを
# 必ず検証する（ビルド直後の分岐・SKIP_BUILD 分岐のいずれでも実行し、ビルド
# 〜計測の間に別プロセスが binary をすり替えた場合も検出する。#1166）。
verify_manifest

# nvidia-smi のスナップショットを実行ログへ残す（GPU 競合検出用の参考情報。
# 判定はしない。競合が疑われる run は人間・エージェントが確認して破棄する）。
echo "== nvidia-smi (before) =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 || true

for run_i in 1 2 3 4 5; do
  for n in 1024 2048 4096; do
    run bench-fandhe gemm cuda "$n" reuse
    run bench-candle gemm cuda "$n" fresh
  done
done

echo "== nvidia-smi (after) =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 || true

echo "done. results in $OUT ; failures (if any) in $SKIP"

if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
