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
#
# 是正（PR #1467 codex-review 指摘）: 従来は参考系列開始前にのみ
# wait_gate.sh を「呼び出すだけ」で結果（標準出力の passed/unmet・
# 終了コード）を検査していなかったため、(a) wait_gate.sh の呼び出し自体が
# 失敗（実行権限欠如等）しても検知されず、(b) 呼び出しが成功しても
# gate_result=unmet のまま計測を開始できてしまい、(c) 正式系列側には
# そもそもゲート呼び出しがなかった。§14.1「各系列の前に負荷ゲートを再判定
# する」契約に沿い、両系列の直前で bash 経由（実行権限に依存しない）で
# wait_gate.sh を呼び、戻り値（標準出力）と終了コードの両方を検査して
# 未通過・呼び出し失敗時は非ゼロ終了で停止する run_gate() を導入した。
#
# 是正（PR #1467 codex-review P1 指摘）: 正式系列の archive 展開先
# （SCRATCH_FC）が特定ユーザー・セッションの /private/tmp 配下に固定され
# ていたため、その一時ディレクトリが消えた後や別環境では正式系列を
# 再実行できなかった（AGENTS.md「ハードコード回避」）。呼び出し時の
# 第 2 引数または環境変数 FC_ARCHIVE_DIR で archive 展開先を受け取り、
# 負荷ゲート待機前（重い処理に入る前）に存在検証するよう変更した。
set -eu
LOGDIR="$(cd "$(dirname "$0")" && pwd)"
FACADE_PATH="$1"  # 呼び出し時に crates/facade の絶対パスを渡す
SCRATCH_FC="${2:-${FC_ARCHIVE_DIR:-}}"  # 61b8b65 アーカイブ展開先（第 2 引数 or FC_ARCHIVE_DIR 環境変数で受け取る）

if [ -z "$SCRATCH_FC" ]; then
  echo "エラー: 正式系列の archive 展開先が指定されていません。呼び出し時の第 2 引数または環境変数 FC_ARCHIVE_DIR で crates/facade を含む 61b8b65 アーカイブ展開先（scripts/bench/framework-compare まで）の絶対パスを渡してください。" >&2
  exit 1
fi
if [ ! -d "$SCRATCH_FC" ]; then
  echo "エラー: 指定された archive 展開先が存在しません: $SCRATCH_FC" >&2
  exit 1
fi
if [ ! -f "$SCRATCH_FC/run_gemm_gate_metal.sh" ]; then
  echo "エラー: 指定された archive 展開先に run_gemm_gate_metal.sh が見つかりません: $SCRATCH_FC" >&2
  exit 1
fi

# 負荷ゲート判定を実行し、通過しなければ（呼び出し自体の失敗も含め）
# 非ゼロで終了してこのオーケストレーションスクリプト自体を止める。
# `bash "$LOGDIR/wait_gate.sh"` で明示的にインタプリタを指定することで、
# ファイルの実行権限ビットに依存せず呼び出せるようにしている。
run_gate() {
  local label="$1"
  local result
  echo "== ${label}開始前の負荷ゲート判定 ==" >&2
  if ! result="$(bash "$LOGDIR/wait_gate.sh")"; then
    echo "エラー: 負荷ゲートの判定呼び出しに失敗しました（${label}）。中止します。" >&2
    exit 1
  fi
  if [ "$result" != "passed" ]; then
    echo "エラー: 負荷ゲート未通過（${result}）のため中止します（${label}）。" >&2
    exit 1
  fi
  echo "== ${label}開始前の負荷ゲート: 通過 ==" >&2
}

run_gate "正式系列"
echo "== 正式系列: 61b8b65（#1438 直前）アーカイブツリーで registry 解決ビルド + 計測 ==" >&2
( cd "$SCRATCH_FC" && bash run_gemm_gate_metal.sh 0.7.0-1309 )

run_gate "参考系列"
echo "== 参考系列: HEAD 797030e を crates/facade へ path patch + 計測 ==" >&2
GEMM_GATE_PATCH_FACADE_PATH="$FACADE_PATH" bash run_gemm_gate_metal.sh head-797030e-1309
