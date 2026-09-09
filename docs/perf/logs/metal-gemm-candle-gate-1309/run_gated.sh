#!/bin/bash
# イシュー #1309: Metal GEMM candle 比ゲート（Phase 3 反映後）の
# 正式系列・参考系列を負荷ゲート付きで直列実行するオーケストレーション記録。
#
# 実際の実行は本スクリプトを対話的に分割して実施した（正式系列は #1438 の
# pin guard により registry 解決ビルドが構造的に不可能なため、archive した
# 61b8b65（#1438 直前コミット）のツリーで registry 解決ビルド、参考系列は
# 本 worktree の HEAD ツリーへ GEMM_GATE_PATCH_FACADE_PATH で path patch。
# §2「中心的な構造問題」参照）。本ファイルは再現用の記録として残す。
#
# 事前宣言規則（§3）: 1 分 load average < 4.0 を 30 秒間隔で 2 回連続確認して
# 通過。待機は 60 秒開始・1.5 倍ずつ増加・最大 10 回。系列間で再判定する。
set -u
LOGDIR="$(cd "$(dirname "$0")" && pwd)"
SCRATCH_FC="/private/tmp/claude-501/-Users-nancy-fandhe-library-rust-ai-library/bac57b76-f1a4-4186-aea8-8f5e06b5dc10/scratchpad/fc-61b8b65/scripts/bench/framework-compare"
FACADE_PATH="$1"  # 呼び出し時に crates/facade の絶対パスを渡す

echo "== 正式系列: 61b8b65（#1438 直前）アーカイブツリーで registry 解決ビルド + 計測 ==" >&2
( cd "$SCRATCH_FC" && bash run_gemm_gate_metal.sh 0.7.0-1309 )

echo "== 参考系列開始前の負荷ゲート再判定 ==" >&2
"$LOGDIR/wait_gate.sh"

echo "== 参考系列: HEAD 797030e を crates/facade へ path patch + 計測 ==" >&2
GEMM_GATE_PATCH_FACADE_PATH="$FACADE_PATH" bash run_gemm_gate_metal.sh head-797030e-1309
