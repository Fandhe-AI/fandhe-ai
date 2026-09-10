# イシュー #1517: split-K 結線前後 framework-compare A/B 実行手順（Mac）

本ディレクトリはイシュー #1517（Metal split-K 本番結線の framework-compare
実践規模 A/B）の実機実測成果物置き場。本ラン（Linux・CI 環境）は
スキャフォールド（スクリプト・帰属表・事前登録判定規則）のみを用意し、
実測は Apple M4 Max 実機を持つ Mac セッションへ引き継ぐ（ルート #1509
の運用方針・メモリ `issue-1509-linux-side-policy`）。

## 前提

- `crates/backend-metal/src/tile.rs::SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED`
  が `false`（既定）の worktree と `true` の worktree の 2 本が必要。
  - `false` 側: 通常は `origin/main`（#1530 マージ済み時点でゲート
    `false`）をそのまま checkout した worktree。
  - `true` 側: 同じ HEAD から `tile.rs` の当該定数のみを `false` → `true`
    へ書き換えた worktree（**定数フリップ以外の差分を含めないこと**。
    `run_ab_splitk_metal.sh` は `git diff --stat -- ':!crates/backend-
    metal/src/tile.rs'` を参考記録として出力するので、これが空でない
    場合は疑うこと）。
  - **定数を `true` へフリップする際は `crates/backend-metal/tests/
    gemm_splitk_auto_wiring.rs`（ドリフト検出テスト
    `split_k_dispatch_auto_production_enabled_is_false_by_default`）が
    落ちる前提を理解した上で行うこと**（一時的な計測用 worktree に限る。
    本番ブランチへコミットしない）。

## 実行手順

```sh
# 1. gate=false / gate=true の 2 worktree を用意する（例: git worktree add）
FACADE_BEFORE="<gate=false worktree>/crates/facade"
FACADE_AFTER="<gate=true worktree>/crates/facade"

# 2. 負荷を確認する（専有ゲートは受け入れ条件ではないが、記録のため）
uptime

# 3. A/B 計測（gemm 8 セル + train 2 セル、5 round）
cd docs/perf/logs/metal-gemm-splitk-framework-compare-1517
AB_BEFORE_FACADE_PATH="$FACADE_BEFORE" \
AB_AFTER_FACADE_PATH="$FACADE_AFTER" \
  ./orchestrate.sh splitk-1517-run1

# 4. 集計（gemm）。--per-run は後退セルが出た場合に §3 rule (b) の
#    「run 単位で符号一貫しているか」を機械的に確認するための追加列
#    （非後退セルのみなら省略してよい。判定〈終了コード〉には影響しない）
cd ../../../../scripts/bench/framework-compare
python3 compare_gemm_ab.py --task gemm --threshold 1.00 --per-run \
  results/raw/results-m4max-splitk-ab-before-splitk-1517-run1.jsonl \
  results/raw/results-m4max-splitk-ab-after-splitk-1517-run1.jsonl

# 5. 集計（train。--phases 診断表つき。--per-run も併用可）
python3 compare_gemm_ab.py --task train --threshold 1.00 --per-run \
  --phases \
  results/raw/results-m4max-splitk-ab-before-splitk-1517-run1-phases.jsonl \
  results/raw/results-m4max-splitk-ab-after-splitk-1517-run1-phases.jsonl \
  results/raw/results-m4max-splitk-ab-before-splitk-1517-run1.jsonl \
  results/raw/results-m4max-splitk-ab-after-splitk-1517-run1.jsonl
```

`--per-run` は run 単位（append 順＝run 順）の `after_k/before_k` 比
5 件と「5 run 全てで比 > 1.00（符号一貫）」フラグを追加列として表示する
（既定 off・既定出力はバイト不変。判定〈終了コード・verdict〉には影響
しない診断列。`docs/perf/metal-gemm-splitk-framework-compare-1517.md`
§3 rule (b) 参照）。判定結果は `docs/perf/metal-gemm-splitk-framework-compare-1517.md`
§5/§6 へ転記し、`env_info.txt` を埋めること。

## 結線維持可否の判定

`docs/perf/metal-gemm-splitk-framework-compare-1517.md` §3「事前登録判定
規則」に従う。後退が確認された場合は `docs/backend-metal-splitk-
decision.md` §5 手順⑤（ゲート `false` への差し戻し）を実施すること。
