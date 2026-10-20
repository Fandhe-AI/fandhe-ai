#!/usr/bin/env bash
# イシュー #3188: GB10 ノード上で 1 系列（2106 -> 2105）を直列実行するラッパー。
# 判定規則は同ディレクトリの RULE.txt（計測前に固定）が正。継承の orchestrate.sh を呼ぶだけで、
# ビルド・計測・マスクの実体は 2105/2106 側にある。
# 使い方: measure_series.sh <pre|head> [both|2106|2105]（省略時 both。2106 のみ・2105 のみの片側計測も可）
# 前提: ~/work/fandhe-3188-<label> に git archive 展開済みのツリーがある（.git なし）。
# ラベルは固定 allowlist、期待 SHA はラベルから引く（外部入力から組み立てない）。
set -euo pipefail

label="${1:-}"
case "$label" in
  pre) expect_sha="17c7767b588a8c4c52eccccc39ba880519962d53" ;;
  head) expect_sha="4f6716f638a6997559847480ddf93cf6924a96e3" ;;
  *) echo "usage: $0 <pre|head> [2106|2105]  (2106: 2106 のみ, 2105: 2105 のみ, 省略: 両方)" >&2; exit 2 ;;
esac
only="${2:-both}"
case "$only" in both | 2106 | 2105) ;; *) echo "invalid selector" >&2; exit 2 ;;
esac

export PATH="$HOME/.cargo/bin:$PATH"
tree="$HOME/work/fandhe-3188-$label"
out="$HOME/work/fandhe-3188-out/$label"
export CARGO_TARGET_DIR="$HOME/work/target-fandhe-3188-$label"

[ "$(cat "$tree/.rev-stamp")" = "$expect_sha" ] || { echo ".rev-stamp 不一致" >&2; exit 1; }

fingerprint() {
  (cd "$tree" && find Cargo.toml Cargo.lock rust-toolchain.toml crates \
    docs/perf/logs/cpu-reuse-device-update-2106 docs/perf/logs/cpu-predict-resident-fixedcost-2105 \
    -type f ! -path '*/target/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
}

mkdir -p "$out"
if [ "$only" = both ]; then
  fingerprint > "$out/tree_sha256_before.txt"
fi

cd "$tree"
if [ "$only" = both ] || [ "$only" = 2106 ]; then
  # 区間帰属の前提となる bit 一致テストを aarch64 で先に確認する（計測とビルドを重ねない）。
  cargo test --release --locked -p fandhe-ai --test cpu_reuse_device_update_diag 2>&1 | tail -n 15 > "$out/aarch64_ci_test_2106.txt"
  bash docs/perf/logs/cpu-reuse-device-update-2106/orchestrate.sh gb10 --out "$out/device-update-2106"
fi
if [ "$only" = both ] || [ "$only" = 2105 ]; then
  cargo test --release --locked -p fandhe-ai --test cpu_predict_resident_fixedcost_diag \
    cpu_predict_resident_decomposition_matches_public_api_bit_exact 2>&1 | tail -n 15 > "$out/aarch64_ci_test_2105.txt"
  bash docs/perf/logs/cpu-predict-resident-fixedcost-2105/orchestrate.sh gb10 --out "$out/predict-resident-2105"
fi

if [ "$only" = both ]; then
  fingerprint > "$out/tree_sha256_after.txt"
fi
cp "$tree/.rev-stamp" "$out/.rev-stamp"
echo "done."
