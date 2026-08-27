# framework-compare: Rust ML フレームワーク横並びベンチマーク

fandhe-ai を candle・Burn と同一プロトコルで横並び比較する独立ベンチ workspace。
`scripts/bench/oss-gemm-compare/`（イシュー #755。許容依存第 9 区分）と同じく**本体 workspace 外の独立 Cargo workspace** であり、ルート `Cargo.toml` / `Cargo.lock` には一切現れない。本 workspace の `Cargo.lock` は比較対象として依存禁止リストのクレート（`candle-core`・`burn` と、その推移的依存の `cubecl`・`ndarray`・`tch` 等）を**意図的に含む**ため、`scripts/check-forbidden-deps.sh lock-all` は禁止リスト grep の代わりに専用の fail-closed 契約検査（Cargo.lock の存在・独自 `[workspace]` 宣言・承認済みピンのドリフト検出）を適用する（`.claude/rules/deps-policy.md`「第 9 区分」の適用範囲拡張、および `docs/framework-compare-harness-decision.md` を参照）。依存監査（advisories / bans / licenses / sources）は専用の `deny.toml` を対象に CI（`deps-forbidden` ジョブ）で毎回実行される。

実測記録（`results/summary.md`・raw JSONL）は「実行資産は scripts/bench・記録は docs/perf」の区分の例外として、再現に必要な生成物一式を本ディレクトリ配下で管理する（`docs/perf/` の実測記録群と同趣旨のコミット済み一次データ）。

## 比較対象

| フレームワーク | クレート | バージョン | デバイス |
| --- | --- | --- | --- |
| fandhe-ai | `fandhe-ai`（facade。crates.io 版） | =0.3.0 | CPU / Metal / CUDA（`tape_for(Device::…)`） |
| candle | `candle-core` | =0.11.0 | CPU / Metal（`metal` feature）/ CUDA（`cuda` feature） |
| Burn | `burn` | =0.21.0 | CPU（ndarray）/ Metal（wgpu）/ CUDA（cubecl） |
| tch-rs | — | 未計測 | libtorch 依存のため省略 |

計測済み環境は 2 系統（詳細・結果は `results/summary.md`）:

- 環境 1: Apple M4 Max / macOS（CPU + Metal）→ `results/raw/results.jsonl`
- 環境 2: DGX Spark（NVIDIA GB10。CUDA + ARM CPU）→ `results/raw/results-dgx.jsonl`。CUDA ホストでは `./run_all_cuda.sh` を使う（bench-candle / bench-burn は `--no-default-features --features cuda` でビルドされる。fandhe-ai は cfg + 実行時プローブのため feature 指定不要）

## 計測タスク

すべて f32・決定的シード（xorshift64\* を `bench-common` に自前実装、全フレームワークで同一シード・同一生成式の入力）。

- **(a) GEMM**: C = A×B、N = 256 / 512 / 1024 / 2048（GPU は 4096 も）。指標: 中央値・Q1・Q3、GFLOP/s（2N³/median）
- **(b) MLP 学習**: 784→256→10（ReLU）、バッチ 64、合成データ、MSE、手動 SGD（lr 0.01）、100 ステップ。先頭 20 ステップを warmup として除外し、残り 80 ステップの 1 ステップ時間の中央値・Q1・Q3
- **(c) 推論**: 同 MLP の forward のみ、バッチ 64。1 回あたり時間の中央値・Q1・Q3、推論/秒

## 計測プロトコル（fandhe-ai の計測規約に準拠・拡張）

- warmup 20 回 → 計測 20 回、中央値 + Q1/Q3（学習は 100 ステップ中、先頭 20 を warmup）
- **同期の統一**: 計測区間の終端で必ず結果テンソルをホストへ実体化して全要素を読み出す
  （fandhe-ai: `to_tensor()` + `contiguous().as_slice()` / candle: `to_vec2()` / Burn: `into_data()`）。
  GPU の非同期実行を計測漏れさせない。読み出した checksum は JSON に記録し、フレームワーク間の数値一致確認に使う
- 計測ごとに新しい計算グラフを作る（fandhe-ai は毎回新しい `tape()` / `tape_for(Device::…)`。
  この条件は fandhe-ai の CUDA で tape ごとの初期化コスト約 440〜460 ms を毎回計測区間に含める。`results/summary.md` 環境 2 の備考を参照）
- 重み初期化: candle / Burn は共有 RNG（同一シード）で同一の重み。fandhe-ai の `Sequential::add_linear` は
  内部初期化（シード指定）のため重みの値自体は異なるが、実行時間には影響しない（同一アーキテクチャ・同一入力）

## 使い方

```bash
cd scripts/bench/framework-compare
./run_all.sh                 # macOS: cpu + metal 全組み合わせ → results/raw/results.jsonl
./run_all_cuda.sh            # CUDA ホスト: cuda + cpu 全組み合わせ → results/raw/results-cuda.jsonl
# 個別実行:
cargo run --release -p bench-fandhe -- --task gemm --device metal --size 2048
# 集計（JSONL → Markdown 表。既定は results/raw/*.jsonl 全件を標準出力へ。
# コミット済みの results/summary.md は既定動作では上書きされない）:
python3 summarize.py
python3 summarize.py results/raw/results.jsonl --out /tmp/tables.md   # 入力・出力の明示
```

失敗した組み合わせは `results/raw/skipped.log`（CUDA は `skipped-cuda.log`）に理由付きで記録される（数値の捏造はしない）。
集計は `results/summary.md` を参照。

## 依存ポリシー上の位置づけ

- 本 workspace は許容依存第 9 区分（ベンチ比較対象）の適用範囲拡張として、`candle-core =0.11.0`・`burn =0.21.0` を**本ディレクトリ限定**で保持する（`.claude/rules/deps-policy.md`）
- 本体 workspace（ルート `Cargo.toml` / `Cargo.lock`）への混入は引き続き禁止であり、ルート `Cargo.lock` / `cargo tree` に対する `scripts/check-forbidden-deps.sh` の検査で fail-closed に検出される
- 承認記録（2026-08-28 ユーザー承認・PR #915）・ライセンス実測・統制の全体像は `docs/framework-compare-harness-decision.md` と `docs/license-matrix.md` 8b 節を参照
