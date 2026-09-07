#!/bin/bash
# イシュー #1306: Metal GEMM の framework-compare 実践規模計測を
# before（正式系列 `fandhe-ai =0.7.0`。registry 解決の承認済みピン）/
# after（参考系列 HEAD。`crates/facade` への path patch）の 2 バイナリで
# 交互起動し、N=512/1024/2048/4096 × fresh/reuse を 5 回ずつ計測する。
#
# 「結線前後」の呼称について（`docs/perf/metal-gemm-n4096-kernel-gap.md`
# §19.1 に詳細）: 依存 #1304（E2〜E4 の候補組み込み判断）は
# `tile::CANDIDATES`／`tile::select`／`select_for_device` を一切変更して
# いない（本番既定は不変）ため、本スクリプトは字義通りの「結線前後」の
# コード差分を計測するものではなく、v0.7.0 → HEAD の Metal 側変更群
# （E2〜E8 の function constant・候補追加等）が本番既定経路の性能を
# 後退させていないかを確認する 0.7.0 ↔ HEAD 非後退確認である。
#
# 呼び出し例（M4 Max 実機。ユーザー承認・別セッション。低負荷時間帯に
# `uptime` を確認してから実行する）:
#   AB_PATCH_FACADE_PATH="$(cd ../../../crates/facade && pwd)" \
#     bash run_ab_gemm_metal.sh head-<short sha>
#
# 設計は `run_ab_managed_cuda.sh`（同一バイナリ off/on を run 単位で交互
# 起動・`Cargo.lock` backup/restore trap）と `run_gemm_gate.sh`（Metal の
# `pmset -g therm`／`uptime` 記録・fail-closed の一時ファイル→原子的 mv・
# バイナリ sha256／依存解決元の manifest 記録）を、before/after 2 バイナリ
# 方式（`compare_ab.py` の 2 ファイル比較と同型）へ合成したもの。
# `compare_gemm_gate.py`（対 candle）・`compare_managed_ab.py`（同一
# バイナリのフラグ切替）はいずれも本用途（同一 fandhe-ai バージョンを
# 名乗る 2 ビルドの比較）には流用できない（`compare_ab.py` は
# `framework_version` 一致を fail-closed 拒否するため）ので、専用の
# `compare_gemm_ab.py` を別途用意する。
#
# 出力は「失敗を捏造しない」方針（security.md A08）: 全 run 成功時のみ
# 一時ファイルを正規パスへ原子的に反映する。1 件でも失敗すれば正規パスは
# 変更せず、不完全な結果は `.failed-<UTC>` へ退避する。
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LABEL=${1:-}

# A03 インジェクション対策: ラベルはファイル名へ直接埋め込むため、
# 英数字・`._-` のみを許可する allowlist で検証する。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. head-abc1234)" >&2
  echo "  env AB_PATCH_FACADE_PATH=<absolute path to HEAD's crates/facade> is required" >&2
  exit 1
fi

# `AB_PATCH_FACADE_PATH` の検証（A03・A08。`run_ab_managed_cuda.sh` と同一
# 方針）: 未設定・相対パス・`"`／`\` 混入・Cargo.toml 不在・crate 名不一致
# のいずれかなら fail-closed で exit 1。
if [[ -z "${AB_PATCH_FACADE_PATH:-}" ]]; then
  echo "error: AB_PATCH_FACADE_PATH is required (absolute path to HEAD's crates/facade; issue #1306)" >&2
  exit 1
fi
if [[ "$AB_PATCH_FACADE_PATH" != /* ]]; then
  echo "error: AB_PATCH_FACADE_PATH must be an absolute path (got: $AB_PATCH_FACADE_PATH)" >&2
  exit 1
fi
if [[ "$AB_PATCH_FACADE_PATH" == *'"'* || "$AB_PATCH_FACADE_PATH" == *'\'* ]]; then
  echo "error: AB_PATCH_FACADE_PATH must not contain '\"' or '\\'" >&2
  exit 1
fi
if [[ ! -f "$AB_PATCH_FACADE_PATH/Cargo.toml" ]]; then
  echo "error: AB_PATCH_FACADE_PATH/Cargo.toml not found ($AB_PATCH_FACADE_PATH)" >&2
  exit 1
fi
if ! grep -qE '^\s*name\s*=\s*"fandhe-ai"\s*$' "$AB_PATCH_FACADE_PATH/Cargo.toml"; then
  echo "error: AB_PATCH_FACADE_PATH/Cargo.toml does not declare name = \"fandhe-ai\" ($AB_PATCH_FACADE_PATH)" >&2
  exit 1
fi
PATCH_CONFIG="patch.crates-io.fandhe-ai.path=\"${AB_PATCH_FACADE_PATH}\""

AB_ROUNDS=${AB_ROUNDS:-5}
if [[ ! "$AB_ROUNDS" =~ ^[0-9]+$ || "$AB_ROUNDS" -lt 1 ]]; then
  echo "error: AB_ROUNDS must be a positive integer (got: $AB_ROUNDS)" >&2
  exit 1
fi

SIZES=(512 1024 2048 4096)
MODES=(fresh reuse)

# before 側の保存先も LABEL でスコープする（codex-review P2 指摘。
# LABEL 非依存だと別 label で次の A/B を実行した際に過去の before が
# 上書きされ、当該 label の交互計測ペア〈before/after〉を再現できなく
# なる。before バイナリ自体は常に registry pin fandhe-ai =0.7.0 だが、
# 「どの label 実行で計測した before データか」を追跡できることが目的）。
OUT_BEFORE="results/raw/results-m4max-gemm-ab-before-0.7.0-${LABEL}.jsonl"
OUT_AFTER="results/raw/results-m4max-gemm-ab-after-${LABEL}.jsonl"
SKIP="results/raw/skipped-m4max-gemm-ab-${LABEL}.log"
MANIFEST="results/raw/manifest-m4max-gemm-ab-${LABEL}.json"
mkdir -p results/raw

# 一時ファイルへ書き、全 run 成功時にのみ原子的に正規パスへ反映する
# （#1166 の教訓と同型。fail-closed。security.md A08）。
OUT_BEFORE_TMP="${OUT_BEFORE}.tmp"
OUT_AFTER_TMP="${OUT_AFTER}.tmp"
SKIP_TMP="${SKIP}.tmp"
: > "$OUT_BEFORE_TMP"
: > "$OUT_AFTER_TMP"
: > "$SKIP_TMP"

ANY_FAILED=0

sha256_of() {
  local f=$1
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  else
    shasum -a 256 "$f" | awk '{print $1}'
  fi
}

# `cargo tree -p bench-fandhe --depth 1` の `fandhe-ai` 行から解決元
# （path か registry か）を抽出する（#1166 事故対応と同型のハードゲート）。
fandhe_ai_source_desc() {
  local line path_part
  line=$(cargo tree -p bench-fandhe --depth 1 "$@" 2>/dev/null | grep -E "^[├└]── fandhe-ai " || true)
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

# `Cargo.lock` を退避し、異常終了含め終了時に必ず復元する（trap。
# `[patch]` は CLI 引数のみで与え Cargo.lock／.cargo/config.toml は変更
# しない契約。deps-policy.md 第 9 区分）。
LOCK_BACKUP="$(mktemp)"
cp Cargo.lock "$LOCK_BACKUP"
restore_lock() {
  cp "$LOCK_BACKUP" Cargo.lock
  rm -f "$LOCK_BACKUP"
}
trap restore_lock EXIT

echo "== build bench-fandhe (before: registry pin fandhe-ai =0.7.0) =="
# `--target-dir target` を明示（codex-review P2 指摘）: CARGO_TARGET_DIR
# 環境変数や .cargo/config.toml の build.target-dir 設定が有効な環境では
# cargo の実際の出力先が本スクリプトが固定コピー元とする
# `target/release/bench-fandhe` と一致せず、以降の `cp` が古いバイナリを
# 拾ったまま before/after 双方を計測しうる（cargo tree・sha256 検証の
# いずれでも検出不能）。target-dir を明示固定することで一致を保証する。
if ! cargo build --release -p bench-fandhe --target-dir target 2>build-err.tmp; then
  tail -40 build-err.tmp
  echo "bench-fandhe (before) BUILD FAILED: $(tail -3 build-err.tmp | tr '\n' ' ')" >&2
  rm -f build-err.tmp
  exit 1
fi
rm -f build-err.tmp
BEFORE_SOURCE="$(fandhe_ai_source_desc || true)"
if [[ "$BEFORE_SOURCE" != "registry" ]]; then
  echo "error: before ビルドの fandhe-ai が registry 解決ではない (actual=${BEFORE_SOURCE:-<取得失敗>})" >&2
  echo "  承認済みピン fandhe-ai =0.7.0（registry）以外での before 確定はできない（fail-closed）。" >&2
  exit 1
fi
cp target/release/bench-fandhe target/release/bench-fandhe-ab-before
BEFORE_SHA="$(sha256_of target/release/bench-fandhe-ab-before)"
echo "bench-fandhe-ab-before sha256: $BEFORE_SHA (source: $BEFORE_SOURCE)"

echo "== build bench-fandhe (after: HEAD path patch) =="
# 同上（codex-review P2 指摘。#1306:166 相当箇所）。
if ! cargo build --release -p bench-fandhe --target-dir target --config "$PATCH_CONFIG" 2>build-err.tmp; then
  tail -40 build-err.tmp
  echo "bench-fandhe (after) BUILD FAILED: $(tail -3 build-err.tmp | tr '\n' ' ')" >&2
  rm -f build-err.tmp
  exit 1
fi
rm -f build-err.tmp
AFTER_SOURCE="$(fandhe_ai_source_desc --config "$PATCH_CONFIG" || true)"
if [[ "$AFTER_SOURCE" != "path:${AB_PATCH_FACADE_PATH}" ]]; then
  echo "error: after ビルドの fandhe-ai が期待した path 解決ではない (expected=path:${AB_PATCH_FACADE_PATH} actual=${AFTER_SOURCE:-<取得失敗>})" >&2
  exit 1
fi
cp target/release/bench-fandhe target/release/bench-fandhe-ab-after
AFTER_SHA="$(sha256_of target/release/bench-fandhe-ab-after)"
echo "bench-fandhe-ab-after sha256: $AFTER_SHA (source: $AFTER_SOURCE)"

SCRIPT_REPO_HEAD_SHA="$(git -C "$SCRIPT_DIR/../../.." rev-parse HEAD 2>/dev/null || echo unknown)"
# after ビルドが実際に取り込んだ facade のコミット（AB_PATCH_FACADE_PATH
# 側の worktree の HEAD）。$AB_PATCH_FACADE_PATH は $SCRIPT_DIR と異なる
# worktree／コミットを指しうるため、`git -C "$AB_PATCH_FACADE_PATH"` で
# 個別に取得する（`git rev-parse --show-toplevel` でリポジトリルートを
# 解決してから rev-parse HEAD する。crates/facade 配下からでも解決可能）。
AFTER_SOURCE_HEAD_SHA="$(git -C "$AB_PATCH_FACADE_PATH" rev-parse HEAD 2>/dev/null || echo unknown)"
MANIFEST_TMP="${MANIFEST}.tmp"
cat > "$MANIFEST_TMP" <<JSON
{"label":"${LABEL}","device":"metal","script_repo_head_sha":"${SCRIPT_REPO_HEAD_SHA}","after_source_head_sha":"${AFTER_SOURCE_HEAD_SHA}","before_sha256":"${BEFORE_SHA}","before_source":"${BEFORE_SOURCE}","after_sha256":"${AFTER_SHA}","after_source":"${AFTER_SOURCE}","recorded_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
# manifest は計測開始前に確定させず一時ファイルへ書く。
# 全 run 成功後に原子的に正規パスへ反映する。同一 label 再実行時に
# 計測前上書きした正規 manifest が計測失敗後も残存し、旧成功結果へ
# 新しいバイナリ hash・コミット情報が誤って紐付く問題を解消する
#〈codex-review P2 指摘〉）。
echo "== manifest（一時ファイル）記録: $MANIFEST_TMP =="
cat "$MANIFEST_TMP"

verify_binaries() {
  local now_before now_after
  now_before="$(sha256_of target/release/bench-fandhe-ab-before)"
  now_after="$(sha256_of target/release/bench-fandhe-ab-after)"
  if [[ "$now_before" != "$BEFORE_SHA" || "$now_after" != "$AFTER_SHA" ]]; then
    echo "error: bench-fandhe-ab-before/after のバイナリが計測中に変化した（sha256 不一致）" >&2
    exit 1
  fi
}

run() { # run <binary> <out_tmp> <size> <mode>
  local bin=$1 out=$2 size=$3 mode=$4
  verify_binaries
  echo "== $bin gemm metal size=$size mode=$mode =="
  if ! "./target/release/$bin" --task gemm --device metal --size "$size" --mode "$mode" --out "$out" 2>err.tmp; then
    echo "$bin gemm metal size=$size mode=$mode : $(cat err.tmp)" >> "$SKIP_TMP"
    echo "  -> FAILED (recorded in $SKIP_TMP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f err.tmp
}

echo "== metal status (before loop) =="
sysctl -n machdep.cpu.brand_string 2>&1 || true
pmset -g therm 2>&1 || true
uptime 2>&1 || true

# run 単位で before/after を交互起動する。偶数 run_i では順序を反転し
# （after→before）、起動順序自体の系統誤差（熱・クロックの片寄り）を
# 均す（`run_ab_managed_cuda.sh` の off→on 固定順より一段厳格な対策）。
for run_i in $(seq 1 "$AB_ROUNDS"); do
  for size in "${SIZES[@]}"; do
    for mode in "${MODES[@]}"; do
      if (( run_i % 2 == 1 )); then
        run bench-fandhe-ab-before "$OUT_BEFORE_TMP" "$size" "$mode"
        run bench-fandhe-ab-after "$OUT_AFTER_TMP" "$size" "$mode"
      else
        run bench-fandhe-ab-after "$OUT_AFTER_TMP" "$size" "$mode"
        run bench-fandhe-ab-before "$OUT_BEFORE_TMP" "$size" "$mode"
      fi
    done
  done
  echo "== run $run_i/$AB_ROUNDS 完了時点の status =="
  pmset -g therm 2>&1 || true
  uptime 2>&1 || true
done

echo "== metal status (after loop) =="
pmset -g therm 2>&1 || true
uptime 2>&1 || true

if [[ "$ANY_FAILED" -eq 0 ]]; then
  mv -f "$OUT_BEFORE_TMP" "$OUT_BEFORE"
  mv -f "$OUT_AFTER_TMP" "$OUT_AFTER"
  mv -f "$SKIP_TMP" "$SKIP"
  mv -f "$MANIFEST_TMP" "$MANIFEST"
  echo "done. before results in $OUT_BEFORE ; after results in $OUT_AFTER ; failures (if any) in $SKIP ; manifest in $MANIFEST"
else
  FAIL_TS=$(date -u +%Y%m%dT%H%M%SZ)
  mv -f "$OUT_BEFORE_TMP" "results/raw/results-m4max-gemm-ab-before-0.7.0-${LABEL}.failed-${FAIL_TS}.jsonl"
  mv -f "$OUT_AFTER_TMP" "results/raw/results-m4max-gemm-ab-after-${LABEL}.failed-${FAIL_TS}.jsonl"
  mv -f "$SKIP_TMP" "results/raw/skipped-m4max-gemm-ab-${LABEL}.failed-${FAIL_TS}.log"
  mv -f "$MANIFEST_TMP" "results/raw/manifest-m4max-gemm-ab-${LABEL}.failed-${FAIL_TS}.json"
  # 正規 $MANIFEST は計測前に一切書き換えないため（一時ファイルのみ更新）、
  # 同一 label 再実行が失敗しても直前の成功結果に紐づく正規 manifest は
  # 保持されたまま残る（codex-review P2 指摘の解消）。
  echo "FAILED: $ANY_FAILED run(s) failed; partial/unreliable data kept for diagnosis (${FAIL_TS}). $OUT_BEFORE/$OUT_AFTER/$MANIFEST left untouched (fail-closed. security.md A08)." >&2
  exit 1
fi
