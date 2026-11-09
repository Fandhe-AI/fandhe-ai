#!/usr/bin/env bash
# #4154 の計測前ビルド（<metal-node> = Mac mini。round 1 の前に 1 回だけ。計測キット。PR-1 時点では未実行）。
# 役割: A0 用の bench-fandhe（release）と B0／B1 用の backend-metal lib テストバイナリを作り、
#   バイナリのパスと sha256 を ${OUT} に記録する。計測中のビルドを禁じるため run_round.sh から分離する。
# 環境変数: ROOT=計測ツリー（tree_sha の git archive 展開先）  OUT=記録先 env ディレクトリ（絶対パス。無ければ作る）
# 出力: ${OUT}/test-bin.path（B0／B1 のテストバイナリの絶対パス）・${OUT}/build-sha256.txt
# bash 3.2 互換。外部入力をシェル文字列へ展開しない。--locked で Cargo.lock を書き換えない。
set -uo pipefail
validate_abs() { # $1=変数名 $2=値。絶対パスのみ。引用符・バックスラッシュ・改行を拒否（A03）
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
ROOT="${ROOT:?計測ツリー}"; OUT="${OUT:?記録先 env ディレクトリ}"
validate_abs ROOT "${ROOT}"; validate_abs OUT "${OUT}"
[ ! -e "${OUT}/test-bin.path" ] || { echo "${OUT}/test-bin.path が既にある（上書き禁止）"; exit 2; }
mkdir -p "${OUT}"
[ "$(sysctl -n machdep.cpu.brand_string)" = "Apple M4" ] || { echo "機体が Apple M4 ではない"; exit 1; }

cargo build --release --locked --manifest-path "${ROOT}/scripts/bench/framework-compare/Cargo.toml" -p bench-fandhe \
  > "${OUT}/build-bench-fandhe.log" 2>&1 || { echo "bench-fandhe のビルド失敗"; exit 1; }

# --no-run --message-format=json の executable 行から lib テストバイナリのパスを得る（python3 の標準 json で解析）。
(cd "${ROOT}" && cargo test --release --locked -p fandhe-ai-backend-metal --lib --no-run --message-format=json) \
  > "${OUT}/build-metal-test.json" 2> "${OUT}/build-metal-test.log" || { echo "backend-metal テストのビルド失敗"; exit 1; }
BIN=$(python3 -I -c '
import json, sys
exe = []
for line in open(sys.argv[1], encoding="utf-8"):
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m.get("profile", {}).get("test") \
            and m.get("target", {}).get("name") == "fandhe_ai_backend_metal" and "lib" in m["target"].get("kind", []):
        exe.append(m["executable"])
if len(exe) != 1:
    sys.exit("lib テストバイナリがちょうど 1 つではない: %d" % len(exe))
print(exe[0])
' "${OUT}/build-metal-test.json") || { echo "テストバイナリのパス取得失敗"; exit 1; }
validate_abs BIN "${BIN}"
[ -x "${BIN}" ] || { echo "テストバイナリが実行できない"; exit 1; }
printf '%s\n' "${BIN}" > "${OUT}/test-bin.path"
{ shasum -a 256 "${BIN}"; shasum -a 256 "${ROOT}/scripts/bench/framework-compare/target/release/bench-fandhe"; } > "${OUT}/build-sha256.txt"
echo "prebuild done $(date -u +%FT%TZ)"
