#!/bin/bash
# イシュー #1517: Metal split-K 本番結線（イシュー #1516・
# `crate::tile::SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED`）の結線前後
# framework-compare 実践規模 A/B。
#
# `run_ab_gemm_metal.sh`（イシュー #1306。before=crates.io 承認ピン
# registry／after=HEAD path patch）とは異なり、本スクリプトは**両腕とも
# `crates/facade` への path patch**（`AB_BEFORE_FACADE_PATH`＝ゲート
# `false` の worktree・`AB_AFTER_FACADE_PATH`＝ゲート `true` の worktree）
# を用いる。理由（`docs/backend-metal-splitk-decision.md` §5・実装計画
# §2）:
#   - 「結線前後」の実体は crates.io 承認ピンの有無ではなく
#     `SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED` 定数の `false`/`true`
#     そのものである。承認ピン ↔ HEAD 比較（`run_ab_gemm_metal.sh`）では
#     ゲート以外の差分（E2〜E8 等）が混入し、字義通りの「結線前後」計測
#     にならない（`docs/perf/metal-gemm-n4096-kernel-gap.md` §19.1 と
#     同型の教訓）。
#   - facade 公開 API での runtime トグルは本イシューのスコープ外
#     （実装計画 §8「スコープ外」）。
#
# 「before==after で計測対象なし」の再発防止（`docs/perf/train-step-
# phase-breakdown.md` §5.11 の教訓）として、両腕の `tile.rs` 定数値・
# 絶対パス・ビルド後バイナリの sha256 のいずれかが一致すれば fail-closed
# で停止する（下記「差分ガード」節）。
#
# gemm 8 セル（N=512/1024/2048/4096 × fresh/reuse）に加え、train 2 セル
# （`--task train` × fresh/reuse。`docs/perf/metal-gemm-splitk-framework-
# compare-1517.md` §2 の帰属表参照）を計測する。gemm と train は
# `compare_gemm_ab.py --task <t>` の task 別 fail-closed 検証（他タスク
# の行を警告つきで除外し 1 件でもあれば判定不能にする。`load_rows`
# docstring）と整合させるため、最初からタスク別の JSONL（`-gemm.jsonl`／
# `-train.jsonl`）へ出力を分離する（PR #1531 是正）。`--phases`（train
# のみ・診断用・各腕 1 回）はさらに別ファイルへ出力し、5 回計測の
# 「ちょうど 5 件」契約を汚さない。
#
# 専有ゲートは要求しない（record_only。ルート #1509 運用方針）。
# `uptime`／`pmset -g therm` を各 run の前後で記録し負荷推移を残す。
#
# 呼び出し例（M4 Max 実機。Mac セッション）:
#   AB_BEFORE_FACADE_PATH="<gate=false worktree>/crates/facade" \
#   AB_AFTER_FACADE_PATH="<gate=true worktree>/crates/facade" \
#     bash run_ab_splitk_metal.sh splitk-1517
#
# 出力は「失敗を捏造しない」方針（security.md A08）: 全 run 成功時のみ
# 一時ファイルを正規パスへ原子的に反映する。1 件でも失敗すれば正規パスは
# 変更せず、不完全な結果は `.failed-<UTC>` へ退避する。
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LABEL=${1:-}

# A03 インジェクション対策: ラベルはファイル名へ直接埋め込むため、
# 英数字・`._-` のみを許可する allowlist で検証する
# （`run_ab_gemm_metal.sh` と同一方針）。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. splitk-1517)" >&2
  echo "  env AB_BEFORE_FACADE_PATH=<absolute path to crates/facade with gate=false> is required" >&2
  echo "  env AB_AFTER_FACADE_PATH=<absolute path to crates/facade with gate=true> is required" >&2
  exit 1
fi

DRY_RUN=0
if [[ "${AB_DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=1
fi

# `AB_*_FACADE_PATH` の検証（A03・A08。`run_ab_gemm_metal.sh` の
# `AB_PATCH_FACADE_PATH` 検証と同一方針を両腕に適用する）: 未設定・
# 相対パス・`"`／`\` 混入・空白混入（`--config` の TOML 文字列へそのまま
# 埋め込むため）・Cargo.toml 不在・crate 名不一致のいずれかなら
# fail-closed で exit 1。
validate_facade_path() { # validate_facade_path <var_name> <value>
  local name=$1 value=$2
  if [[ -z "$value" ]]; then
    echo "error: $name is required (absolute path to a crates/facade worktree; issue #1517)" >&2
    exit 1
  fi
  if [[ "$value" != /* ]]; then
    echo "error: $name must be an absolute path (got: $value)" >&2
    exit 1
  fi
  if [[ "$value" == *'"'* || "$value" == *'\'* || "$value" == *' '* ]]; then
    echo "error: $name must not contain '\"', '\\', or a space (got: $value)" >&2
    exit 1
  fi
  if [[ ! -f "$value/Cargo.toml" ]]; then
    echo "error: $name/Cargo.toml not found ($value)" >&2
    exit 1
  fi
  if ! grep -qE '^\s*name\s*=\s*"fandhe-ai"\s*$' "$value/Cargo.toml"; then
    echo "error: $name/Cargo.toml does not declare name = \"fandhe-ai\" ($value)" >&2
    exit 1
  fi
}
validate_facade_path AB_BEFORE_FACADE_PATH "${AB_BEFORE_FACADE_PATH:-}"
validate_facade_path AB_AFTER_FACADE_PATH "${AB_AFTER_FACADE_PATH:-}"

# 両腕が同一パスを指す誤操作の fail-closed 検出（`realpath` があれば
# シンボリックリンク経由の別名一致も検出する。無ければ文字列比較のみ）。
BEFORE_REAL="$AB_BEFORE_FACADE_PATH"
AFTER_REAL="$AB_AFTER_FACADE_PATH"
if command -v realpath >/dev/null 2>&1; then
  BEFORE_REAL="$(realpath "$AB_BEFORE_FACADE_PATH" 2>/dev/null || echo "$AB_BEFORE_FACADE_PATH")"
  AFTER_REAL="$(realpath "$AB_AFTER_FACADE_PATH" 2>/dev/null || echo "$AB_AFTER_FACADE_PATH")"
fi
if [[ "$BEFORE_REAL" == "$AFTER_REAL" ]]; then
  echo "error: AB_BEFORE_FACADE_PATH と AB_AFTER_FACADE_PATH が同一パスを指している（結線前後の対照にならない。fail-closed）" >&2
  exit 1
fi

# 差分ガード（実装計画 §1「設計判断」表・「before==after 再発防止」）:
# 各腕の `crates/backend-metal/src/tile.rs`
# （`<facade path>/../backend-metal/src/tile.rs`）を読み、
# `SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED` の宣言行が before=false・
# after=true であることを機械検証する。`grep -E` の alternation は
# BRE エスケープ不要（`(false|true)`）。
gate_value_of() { # gate_value_of <facade_path>
  local facade=$1 tile_rs
  tile_rs="$(cd "$facade/../backend-metal" 2>/dev/null && pwd)/src/tile.rs"
  if [[ ! -f "$tile_rs" ]]; then
    echo "error: tile.rs not found relative to facade path ($facade)" >&2
    return 1
  fi
  local line
  line="$(grep -E '^pub\(crate\) const SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED: bool = (false|true);$' "$tile_rs" || true)"
  if [[ -z "$line" ]]; then
    echo "error: SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED の宣言行を $tile_rs から特定できなかった（フォーマット変更の可能性。fail-closed）" >&2
    return 1
  fi
  if [[ "$line" == *"= false;" ]]; then
    echo "false"
  else
    echo "true"
  fi
}
BEFORE_GATE="$(gate_value_of "$AB_BEFORE_FACADE_PATH")" || exit 1
AFTER_GATE="$(gate_value_of "$AB_AFTER_FACADE_PATH")" || exit 1
if [[ "$BEFORE_GATE" != "false" || "$AFTER_GATE" != "true" ]]; then
  echo "error: 期待するゲート値と異なる（before(expected=false)=${BEFORE_GATE}, after(expected=true)=${AFTER_GATE}）。結線前後の対照にならないため fail-closed で停止する。" >&2
  exit 1
fi
echo "gate check: before=SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED=false / after=SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED=true"

# 参考記録（fail-closed の対象ではない。tile.rs 定数フリップ以外の
# 差分が両腕に混入していないかを人間が確認するための情報のみ）。
#
# README が想定する運用（同一 HEAD sha から一方の worktree だけ tile.rs
# の定数を書き換えるコミットなしの一時フリップ）では、①両腕は別
# worktree のため `git rev-parse --show-toplevel` が一致せずクロス
# HEAD-sha diff は計算できない、②両腕の `git rev-parse HEAD` はそもそも
# 同一（コミットなしフリップのため）で計算できたとしても常に空になる
# ——という 2 重の理由でクロス HEAD-sha diff は実運用では機能しない
# （advisor 指摘）。代わりに **各腕の worktree 内の未コミット差分**
# （tile.rs 以外）を個別に記録する: これは「フリップ以外にもコミット
# されていない変更が紛れ込んでいないか」を検出できる。加えてクロス
# HEAD-sha diff も（両腕が別コミットを指す構成で使われた場合に備え）
# 参考として試みるが、`--show-toplevel` の一致は要求しない
# （2>/dev/null で失敗時は結果空）。
non_tile_uncommitted_diff_lines() { # non_tile_uncommitted_diff_lines <facade_path>
  local facade=$1 root
  root="$(git -C "$facade" rev-parse --show-toplevel 2>/dev/null || echo "")"
  if [[ -z "$root" ]]; then
    echo "unknown"
    return
  fi
  git -C "$root" diff --stat HEAD -- ':!crates/backend-metal/src/tile.rs' 2>/dev/null | wc -l | tr -d ' '
}
BEFORE_HEAD_SHA="$(git -C "$AB_BEFORE_FACADE_PATH" rev-parse HEAD 2>/dev/null || echo unknown)"
AFTER_HEAD_SHA="$(git -C "$AB_AFTER_FACADE_PATH" rev-parse HEAD 2>/dev/null || echo unknown)"
BEFORE_NON_TILE_UNCOMMITTED="$(non_tile_uncommitted_diff_lines "$AB_BEFORE_FACADE_PATH")"
AFTER_NON_TILE_UNCOMMITTED="$(non_tile_uncommitted_diff_lines "$AB_AFTER_FACADE_PATH")"
CROSS_SHA_NON_TILE_DIFF="unknown"
if [[ "$BEFORE_HEAD_SHA" != "unknown" && "$AFTER_HEAD_SHA" != "unknown" ]]; then
  BEFORE_ROOT_FOR_CROSS="$(git -C "$AB_BEFORE_FACADE_PATH" rev-parse --show-toplevel 2>/dev/null || echo "")"
  if [[ -n "$BEFORE_ROOT_FOR_CROSS" ]]; then
    CROSS_SHA_NON_TILE_DIFF="$(git -C "$BEFORE_ROOT_FOR_CROSS" diff --stat "$BEFORE_HEAD_SHA" "$AFTER_HEAD_SHA" -- ':!crates/backend-metal/src/tile.rs' 2>/dev/null | wc -l | tr -d ' ')"
    CROSS_SHA_NON_TILE_DIFF="${CROSS_SHA_NON_TILE_DIFF:-unknown}"
  fi
fi
echo "note: before 腕の tile.rs 以外の未コミット差分行数: ${BEFORE_NON_TILE_UNCOMMITTED}"
echo "note: after 腕の tile.rs 以外の未コミット差分行数: ${AFTER_NON_TILE_UNCOMMITTED}"
echo "note: before/after 間の tile.rs 以外のクロス HEAD-sha 差分行数（同一 repo 内かつ両腕が別コミットを指す場合のみ計算。同一コミットの一時フリップ運用では常に unknown/0）: ${CROSS_SHA_NON_TILE_DIFF}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "AB_DRY_RUN=1: バリデーションのみ完了（cargo/pmset/sysctl は実行しない）。"
  exit 0
fi

AB_ROUNDS=${AB_ROUNDS:-5}
if [[ ! "$AB_ROUNDS" =~ ^[0-9]+$ || "$AB_ROUNDS" -lt 1 ]]; then
  echo "error: AB_ROUNDS must be a positive integer (got: $AB_ROUNDS)" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required (used to parse 'cargo build --message-format=json' artifact paths)" >&2
  exit 1
fi

SIZES=(512 1024 2048 4096)
MODES=(fresh reuse)

# P1 是正（codex-review・Cursor Bugbot 指摘。イシュー #1517 PR #1531）:
# `compare_gemm_ab.py` は `--task`（gemm/train）で指定したタスク以外の
# 行を `_valid_cell_identity` で不正行として警告つき除外し、1 件でも
# warning があれば fail-closed（終了コード 2）で判定不能にする
# （`load_rows` docstring・`main` の `if warnings: return 2`）。
# 当初 gemm 8 セルと train 2 セルを同一 before/after JSONL へ追記して
# いたため、README の手順どおり `--task gemm`／`--task train` のいずれで
# 集計しても相手タスクの行が必ず警告対象になり判定不能になっていた。
# gemm と train を最初からタスク別ファイルへ分離して出力し、
# `compare_gemm_ab.py --task <t>` にはそのタスク単独のファイルだけを
# 渡す契約へ変更する。
OUT_BEFORE_GEMM="results/raw/results-m4max-splitk-ab-before-${LABEL}-gemm.jsonl"
OUT_AFTER_GEMM="results/raw/results-m4max-splitk-ab-after-${LABEL}-gemm.jsonl"
OUT_BEFORE_TRAIN="results/raw/results-m4max-splitk-ab-before-${LABEL}-train.jsonl"
OUT_AFTER_TRAIN="results/raw/results-m4max-splitk-ab-after-${LABEL}-train.jsonl"
OUT_BEFORE_PHASES="results/raw/results-m4max-splitk-ab-before-${LABEL}-phases.jsonl"
OUT_AFTER_PHASES="results/raw/results-m4max-splitk-ab-after-${LABEL}-phases.jsonl"
SKIP="results/raw/skipped-m4max-splitk-ab-${LABEL}.log"
MANIFEST="results/raw/manifest-m4max-splitk-ab-${LABEL}.json"
mkdir -p results/raw

OUT_BEFORE_GEMM_TMP="${OUT_BEFORE_GEMM}.tmp"
OUT_AFTER_GEMM_TMP="${OUT_AFTER_GEMM}.tmp"
OUT_BEFORE_TRAIN_TMP="${OUT_BEFORE_TRAIN}.tmp"
OUT_AFTER_TRAIN_TMP="${OUT_AFTER_TRAIN}.tmp"
OUT_BEFORE_PHASES_TMP="${OUT_BEFORE_PHASES}.tmp"
OUT_AFTER_PHASES_TMP="${OUT_AFTER_PHASES}.tmp"
SKIP_TMP="${SKIP}.tmp"
: > "$OUT_BEFORE_GEMM_TMP"
: > "$OUT_AFTER_GEMM_TMP"
: > "$OUT_BEFORE_TRAIN_TMP"
: > "$OUT_AFTER_TRAIN_TMP"
: > "$OUT_BEFORE_PHASES_TMP"
: > "$OUT_AFTER_PHASES_TMP"
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

fandhe_ai_source_desc() { # fandhe_ai_source_desc [追加の cargo tree 引数...]
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

# `Cargo.lock` の退避・復元（`run_ab_gemm_metal.sh` と同一方針。deps-
# policy.md 第 9 区分「`[patch]` は CLI 引数のみで与え Cargo.lock は
# 変更しない」契約）。
LOCK_BACKUP="$(mktemp)"
if ! cp Cargo.lock "$LOCK_BACKUP"; then
  echo "error: cp Cargo.lock '$LOCK_BACKUP' (backup) に失敗した" >&2
  rm -f "$LOCK_BACKUP"
  exit 1
fi
if [[ "$(sha256_of Cargo.lock)" != "$(sha256_of "$LOCK_BACKUP")" ]]; then
  echo "error: Cargo.lock のバックアップ内容が元ファイルと一致しない（不完全な退避の可能性）" >&2
  rm -f "$LOCK_BACKUP"
  exit 1
fi
restore_lock() {
  if ! cp "$LOCK_BACKUP" Cargo.lock; then
    echo "error: Cargo.lock の復元に失敗した。バックアップを保持する: $LOCK_BACKUP" >&2
    return 1
  fi
  rm -f "$LOCK_BACKUP"
}
restore_lock_trap() {
  local code=$?
  if ! restore_lock; then
    if [[ "$code" -eq 0 ]]; then
      code=1
    fi
  fi
  exit "$code"
}
trap restore_lock_trap EXIT

build_bench_fandhe() { # build_bench_fandhe <out_exe_pathvar> [追加の cargo build 引数...]
  local __out_var=$1
  shift
  local msg_file
  msg_file="$(mktemp)"
  if ! cargo build --release -p bench-fandhe --target-dir target --message-format=json "$@" >"$msg_file" 2>build-err.tmp; then
    tail -40 build-err.tmp
    echo "bench-fandhe BUILD FAILED: $(tail -3 build-err.tmp | tr '\n' ' ')" >&2
    rm -f build-err.tmp "$msg_file"
    exit 1
  fi
  rm -f build-err.tmp
  local exe
  exe="$(jq -rs '[.[] | select(.reason == "compiler-artifact" and .target.name == "bench-fandhe" and (.target.kind[]? == "bin") and .executable != null)] | last | .executable // empty' "$msg_file")"
  rm -f "$msg_file"
  if [[ -z "$exe" || ! -f "$exe" ]]; then
    echo "error: bench-fandhe ビルド成果物のパスを 'cargo build --message-format=json' から特定できなかった（jq 抽出結果: '${exe:-<空>}')" >&2
    exit 1
  fi
  printf -v "$__out_var" '%s' "$exe"
}

BEFORE_CONFIG="patch.crates-io.fandhe-ai.path=\"${AB_BEFORE_FACADE_PATH}\""
AFTER_CONFIG="patch.crates-io.fandhe-ai.path=\"${AB_AFTER_FACADE_PATH}\""

echo "== build bench-fandhe (before: SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED=false path patch) =="
build_bench_fandhe BEFORE_EXE --config "$BEFORE_CONFIG"
BEFORE_SOURCE="$(fandhe_ai_source_desc --config "$BEFORE_CONFIG" || true)"
if [[ "$BEFORE_SOURCE" != "path:${AB_BEFORE_FACADE_PATH}" ]]; then
  echo "error: before ビルドの fandhe-ai が期待した path 解決ではない (expected=path:${AB_BEFORE_FACADE_PATH} actual=${BEFORE_SOURCE:-<取得失敗>})" >&2
  exit 1
fi
mkdir -p target/release
if ! cp "$BEFORE_EXE" target/release/bench-fandhe-splitk-ab-before; then
  echo "error: cp '$BEFORE_EXE' target/release/bench-fandhe-splitk-ab-before に失敗した" >&2
  exit 1
fi
BEFORE_SHA="$(sha256_of target/release/bench-fandhe-splitk-ab-before)"
echo "bench-fandhe-splitk-ab-before sha256: $BEFORE_SHA (source: $BEFORE_SOURCE, exe: $BEFORE_EXE)"

echo "== build bench-fandhe (after: SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED=true path patch) =="
build_bench_fandhe AFTER_EXE --config "$AFTER_CONFIG"
AFTER_SOURCE="$(fandhe_ai_source_desc --config "$AFTER_CONFIG" || true)"
if [[ "$AFTER_SOURCE" != "path:${AB_AFTER_FACADE_PATH}" ]]; then
  echo "error: after ビルドの fandhe-ai が期待した path 解決ではない (expected=path:${AB_AFTER_FACADE_PATH} actual=${AFTER_SOURCE:-<取得失敗>})" >&2
  exit 1
fi
if ! cp "$AFTER_EXE" target/release/bench-fandhe-splitk-ab-after; then
  echo "error: cp '$AFTER_EXE' target/release/bench-fandhe-splitk-ab-after に失敗した" >&2
  exit 1
fi
AFTER_SHA="$(sha256_of target/release/bench-fandhe-splitk-ab-after)"
echo "bench-fandhe-splitk-ab-after sha256: $AFTER_SHA (source: $AFTER_SOURCE, exe: $AFTER_EXE)"

# 「before==after で計測対象なし」の再発防止（差分ガードの最終段。
# バイナリ自体が同一なら、ゲート値の grep 検証をすり抜けた別種の
# 取り違えが起きている可能性がある）。
if [[ "$BEFORE_SHA" == "$AFTER_SHA" ]]; then
  echo "error: before/after のビルド成果物が bit 同一（sha256 一致）。結線前後の対照になっていない（fail-closed）。" >&2
  exit 1
fi

SCRIPT_REPO_HEAD_SHA="$(git -C "$SCRIPT_DIR/../../.." rev-parse HEAD 2>/dev/null || echo unknown)"
MANIFEST_TMP="${MANIFEST}.tmp"
cat > "$MANIFEST_TMP" <<JSON
{"label":"${LABEL}","device":"metal","script_repo_head_sha":"${SCRIPT_REPO_HEAD_SHA}","before_source_head_sha":"${BEFORE_HEAD_SHA}","after_source_head_sha":"${AFTER_HEAD_SHA}","before_gate":"${BEFORE_GATE}","after_gate":"${AFTER_GATE}","before_sha256":"${BEFORE_SHA}","before_source":"${BEFORE_SOURCE}","after_sha256":"${AFTER_SHA}","after_source":"${AFTER_SOURCE}","before_non_tile_uncommitted_diff_lines":"${BEFORE_NON_TILE_UNCOMMITTED}","after_non_tile_uncommitted_diff_lines":"${AFTER_NON_TILE_UNCOMMITTED}","cross_sha_non_tile_diff_lines":"${CROSS_SHA_NON_TILE_DIFF}","recorded_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
echo "== manifest（一時ファイル）記録: $MANIFEST_TMP =="
cat "$MANIFEST_TMP"

verify_binaries() {
  local now_before now_after
  now_before="$(sha256_of target/release/bench-fandhe-splitk-ab-before)"
  now_after="$(sha256_of target/release/bench-fandhe-splitk-ab-after)"
  if [[ "$now_before" != "$BEFORE_SHA" || "$now_after" != "$AFTER_SHA" ]]; then
    echo "error: bench-fandhe-splitk-ab-before/after のバイナリが計測中に変化した（sha256 不一致）" >&2
    exit 1
  fi
}

run_gemm() { # run_gemm <binary> <out_tmp> <size> <mode>
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

run_train() { # run_train <binary> <out_tmp> <mode>
  local bin=$1 out=$2 mode=$3
  verify_binaries
  echo "== $bin train metal mode=$mode =="
  if ! "./target/release/$bin" --task train --device metal --mode "$mode" --out "$out" 2>err.tmp; then
    echo "$bin train metal mode=$mode : $(cat err.tmp)" >> "$SKIP_TMP"
    echo "  -> FAILED (recorded in $SKIP_TMP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f err.tmp
}

echo "== metal status (before loop) =="
sysctl -n machdep.cpu.brand_string 2>&1 || true
pmset -g therm 2>&1 || true
uptime 2>&1 || true

# run 単位で before/after を交互起動する。偶数 run_i では順序を反転する
# （`run_ab_gemm_metal.sh` と同一方針）。gemm 8 セル・train 2 セルは
# 同一 run ループ内で実行するが、出力先はタスク別ファイル
# （`OUT_*_GEMM`／`OUT_*_TRAIN`）へ分離する（P1 是正。上記コメント参照）。
# 分離後も各ファイル内での append 順は run 番号のままのため、
# `compare_gemm_ab.py --per-run` の「append 順＝run 順」前提は
# タスクごとに維持される。
for run_i in $(seq 1 "$AB_ROUNDS"); do
  for size in "${SIZES[@]}"; do
    for mode in "${MODES[@]}"; do
      if (( run_i % 2 == 1 )); then
        run_gemm bench-fandhe-splitk-ab-before "$OUT_BEFORE_GEMM_TMP" "$size" "$mode"
        run_gemm bench-fandhe-splitk-ab-after "$OUT_AFTER_GEMM_TMP" "$size" "$mode"
      else
        run_gemm bench-fandhe-splitk-ab-after "$OUT_AFTER_GEMM_TMP" "$size" "$mode"
        run_gemm bench-fandhe-splitk-ab-before "$OUT_BEFORE_GEMM_TMP" "$size" "$mode"
      fi
    done
  done
  for mode in "${MODES[@]}"; do
    if (( run_i % 2 == 1 )); then
      run_train bench-fandhe-splitk-ab-before "$OUT_BEFORE_TRAIN_TMP" "$mode"
      run_train bench-fandhe-splitk-ab-after "$OUT_AFTER_TRAIN_TMP" "$mode"
    else
      run_train bench-fandhe-splitk-ab-after "$OUT_AFTER_TRAIN_TMP" "$mode"
      run_train bench-fandhe-splitk-ab-before "$OUT_BEFORE_TRAIN_TMP" "$mode"
    fi
  done
  echo "== run $run_i/$AB_ROUNDS 完了時点の status =="
  pmset -g therm 2>&1 || true
  uptime 2>&1 || true
done

# `--phases`（train のみ・診断用・各腕 1 回。本体セルの「ちょうど 5 件」
# 契約を汚さないよう別ファイルへ出力する。実装計画 §1「設計判断」表）。
for mode in "${MODES[@]}"; do
  verify_binaries
  echo "== bench-fandhe-splitk-ab-before train --phases mode=$mode =="
  if ! ./target/release/bench-fandhe-splitk-ab-before --task train --device metal --mode "$mode" --phases --out "$OUT_BEFORE_PHASES_TMP" 2>err.tmp; then
    echo "bench-fandhe-splitk-ab-before train --phases mode=$mode : $(cat err.tmp)" >> "$SKIP_TMP"
    echo "  -> FAILED (recorded in ${SKIP_TMP}。--phases は診断用のため ANY_FAILED には計上しない)"
  fi
  rm -f err.tmp
  echo "== bench-fandhe-splitk-ab-after train --phases mode=$mode =="
  if ! ./target/release/bench-fandhe-splitk-ab-after --task train --device metal --mode "$mode" --phases --out "$OUT_AFTER_PHASES_TMP" 2>err.tmp; then
    echo "bench-fandhe-splitk-ab-after train --phases mode=$mode : $(cat err.tmp)" >> "$SKIP_TMP"
    echo "  -> FAILED (recorded in ${SKIP_TMP}。--phases は診断用のため ANY_FAILED には計上しない)"
  fi
  rm -f err.tmp
done

echo "== metal status (after loop) =="
pmset -g therm 2>&1 || true
uptime 2>&1 || true

MV_FAILED=0
mv_checked() { # mv_checked <src> <dst>
  if ! mv -f "$1" "$2"; then
    echo "error: mv -f '$1' '$2' に失敗した" >&2
    MV_FAILED=$((MV_FAILED + 1))
  fi
}

if [[ "$ANY_FAILED" -eq 0 ]]; then
  mv_checked "$OUT_BEFORE_GEMM_TMP" "$OUT_BEFORE_GEMM"
  mv_checked "$OUT_AFTER_GEMM_TMP" "$OUT_AFTER_GEMM"
  mv_checked "$OUT_BEFORE_TRAIN_TMP" "$OUT_BEFORE_TRAIN"
  mv_checked "$OUT_AFTER_TRAIN_TMP" "$OUT_AFTER_TRAIN"
  mv_checked "$OUT_BEFORE_PHASES_TMP" "$OUT_BEFORE_PHASES"
  mv_checked "$OUT_AFTER_PHASES_TMP" "$OUT_AFTER_PHASES"
  mv_checked "$SKIP_TMP" "$SKIP"
  mv_checked "$MANIFEST_TMP" "$MANIFEST"
  if [[ "$MV_FAILED" -ne 0 ]]; then
    echo "error: $MV_FAILED 件の mv が失敗した。新旧結果混在の可能性があるため、正規パスの内容を手動確認すること（fail-closed。security.md A08）。" >&2
    exit 1
  fi
  echo "done. gemm before/after results in $OUT_BEFORE_GEMM / $OUT_AFTER_GEMM ; train before/after results in $OUT_BEFORE_TRAIN / $OUT_AFTER_TRAIN ; phases (diagnostic) in $OUT_BEFORE_PHASES/$OUT_AFTER_PHASES ; failures (if any) in $SKIP ; manifest in $MANIFEST"
else
  FAIL_TS=$(date -u +%Y%m%dT%H%M%SZ)
  mv_checked "$OUT_BEFORE_GEMM_TMP" "results/raw/results-m4max-splitk-ab-before-${LABEL}-gemm.failed-${FAIL_TS}.jsonl"
  mv_checked "$OUT_AFTER_GEMM_TMP" "results/raw/results-m4max-splitk-ab-after-${LABEL}-gemm.failed-${FAIL_TS}.jsonl"
  mv_checked "$OUT_BEFORE_TRAIN_TMP" "results/raw/results-m4max-splitk-ab-before-${LABEL}-train.failed-${FAIL_TS}.jsonl"
  mv_checked "$OUT_AFTER_TRAIN_TMP" "results/raw/results-m4max-splitk-ab-after-${LABEL}-train.failed-${FAIL_TS}.jsonl"
  mv_checked "$OUT_BEFORE_PHASES_TMP" "results/raw/results-m4max-splitk-ab-before-${LABEL}-phases.failed-${FAIL_TS}.jsonl"
  mv_checked "$OUT_AFTER_PHASES_TMP" "results/raw/results-m4max-splitk-ab-after-${LABEL}-phases.failed-${FAIL_TS}.jsonl"
  mv_checked "$SKIP_TMP" "results/raw/skipped-m4max-splitk-ab-${LABEL}.failed-${FAIL_TS}.log"
  mv_checked "$MANIFEST_TMP" "results/raw/manifest-m4max-splitk-ab-${LABEL}.failed-${FAIL_TS}.json"
  if [[ "$MV_FAILED" -ne 0 ]]; then
    echo "error: $MV_FAILED 件の失敗結果退避 mv も失敗した（診断用データが一部欠落している可能性がある）。" >&2
  fi
  echo "FAILED: $ANY_FAILED run(s) failed; partial/unreliable data kept for diagnosis (${FAIL_TS}). $OUT_BEFORE_GEMM/$OUT_AFTER_GEMM/$OUT_BEFORE_TRAIN/$OUT_AFTER_TRAIN/$MANIFEST left untouched (fail-closed. security.md A08)." >&2
  exit 1
fi
