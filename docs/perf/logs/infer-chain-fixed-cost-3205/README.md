# 推論チェーンの固定費の削減（GB10・イシュー #3205）の計測基盤と生ログ

数値の正式記録は `docs/perf/infer-chain-fixed-cost-reduction-3205.md`。本ディレクトリは事前登録規則（`RULE.txt`。計測前に固定、
本 PR で編集していない）・集計スクリプト・GB10 の生ログ。B3（推論 Graph capture）の再判定は別規則
`../infer-chain-graphcapture-rejudge-3205/`（`RULE.txt` を計測前に単独コミットした）。

## 構成

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（ADOPT は 3 条件: 中央値の比 <= 1.00・5 run 中 3 run 以上で after が速い・別バイナリ間の bit ダンプ sha256 一致） |
| `aggregate_breakdown.py` | 層の発行の下位分解の集計（標準ライブラリのみ。5 run 揃い・欠損・非数・負値は例外。`python3 -I aggregate_breakdown.py --self-test`） |
| `gb10/breakdown/` | 下位分解（`crates/backend-cuda/tests/infer_chain_issue_breakdown_diag_3205.rs`）5 run の生ログ・`load_gate.log`・`aggregate.md` |
| `gb10/ab/` | 変更前後の A/B（`run_ab_infer_chain_cuda.sh 3205-gb10-reuse`）: `compare-infer-3205-gb10-reuse.md`・各腕の JSONL・ゲートログ・uptime・nvidia-smi |
| `gb10/ab-ungated-first-attempt/` | 負荷ゲート前に取った最初の A/B（ビルド直後で load1 が高かった）。**判定に使わない**。ゲート付きで取り直したため invalid 試行として残す（0.9076。取り直しは 0.9038） |
| `gb10/bitdump/` | 別バイナリ間の bit ダンプ（`predict_resident_bit_dump_cuda_3205`・640 要素）。`sha256.txt` が before／after で一致 |
| `gb10/vs-peers/` | 同一セッションの相手比（fandhe-ai before／after・candle 0.11.0・PyTorch 2.14.0+cu130。5 run。record-only） |
| `gb10/arm-identification.txt` | 腕の同定（before／after の SHA・ツリー指紋の突合結果） |

## 実行手順（GB10。`docs/real-hardware-verification-env.md`）

1. `git archive` で before（変更直前の main）・after（HEAD）を別ディレクトリへ展開し、各ルートへ `.rev-stamp`（フル SHA）を置いて転送する。
2. `PATH=/usr/local/cuda/bin:$PATH` を明示し、`CARGO_TARGET_DIR` はツリーの外に置く。計測中に他のビルドを並走させない。
3. 負荷ゲート（load1 < 1.0 かつ GPU utilization 0%）を通ってから計測する。
4. A/B: `AB_BEFORE_FACADE_PATH=<before>/crates/facade AB_AFTER_FACADE_PATH=<after>/crates/facade bash scripts/bench/framework-compare/run_ab_infer_chain_cuda.sh <label>`（5 round・run ごとに起動順を反転）。
5. bit ダンプ: 両ツリーで `cargo test --release -p fandhe-ai --test predict_device_chain_cuda_fixed_cost_bit_identity_3205 -- --ignored --exact predict_resident_bit_dump_cuda_3205 --nocapture --test-threads=1` を実行し、`out[<i>].bits=` を抽出して sha256 を突き合わせる（before ツリーには同テストファイルを置く）。
6. ホスト名・`$HOME`・ユーザー名は収録前にマスクした（残存 0 件を検索で確認）。

## 限界

- 下位分解の確保・解放は GPU idle での単独計時で、チェーン内（カーネル実行と重なる）の値とは一致しない。`*_other_derived` は中央値差の導出値。
- GB10 には他プロセス（常駐サービス）の GPU メモリ使用がある（`nvidia-smi` の記録。utilization は 0〜数 %）。
- bench-fandhe の infer の checksum は candle・PyTorch の値と定義が異なる（before／after 間では完全一致。本件では調査していない）。
