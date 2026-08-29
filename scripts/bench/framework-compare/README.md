# framework-compare: Rust ML フレームワーク横並びベンチマーク

fandhe-ai を candle・Burn と同一プロトコルで横並び比較する独立ベンチ workspace
（プロトコル同一性の範囲・例外は「計測プロトコル」節の `--mode fresh|reuse` の説明を参照。
Metal / CUDA の `fresh` GEMM 比較は例外にあたる）。
`scripts/bench/oss-gemm-compare/`（イシュー #755。許容依存第 9 区分）と同じく**本体 workspace 外の独立 Cargo workspace** であり、ルート `Cargo.toml` / `Cargo.lock` には一切現れない。本 workspace の `Cargo.lock` は比較対象として依存禁止リストのクレート（`candle-core`・`burn` と、その推移的依存の `cubecl`・`ndarray`・`tch` 等）を**意図的に含む**ため、`scripts/check-forbidden-deps.sh lock-all` は禁止リスト grep の代わりに専用の fail-closed 契約検査（Cargo.lock の存在・独自 `[workspace]` 宣言・承認済みピンのドリフト検出）を適用する（`.claude/rules/deps-policy.md`「第 9 区分」の適用範囲拡張、および `docs/framework-compare-harness-decision.md` を参照）。依存監査（advisories / bans / licenses / sources）は専用の `deny.toml` を対象に CI（`deps-forbidden` ジョブ）で毎回実行される。

実測記録（`results/summary.md`・raw JSONL）は「実行資産は scripts/bench・記録は docs/perf」の区分の例外として、再現に必要な生成物一式を本ディレクトリ配下で管理する（`docs/perf/` の実測記録群と同趣旨のコミット済み一次データ）。

## 比較対象

| フレームワーク | クレート | バージョン | デバイス |
| --- | --- | --- | --- |
| fandhe-ai | `fandhe-ai`（facade。crates.io 版） | =0.4.0 | CPU / Metal / CUDA（`tape_for(Device::…)`） |
| candle | `candle-core` | =0.11.0 | CPU / Metal（`metal` feature）/ CUDA（`cuda` feature） |
| Burn | `burn` | =0.21.0 | CPU（ndarray）/ Metal（wgpu）/ CUDA（cubecl） |
| tch-rs | — | 未計測 | libtorch 依存のため省略 |

計測済み環境は 3 系統（詳細・結果は `results/summary.md`）:

- 環境 1: Apple M4 Max / macOS（CPU + Metal）→ `results/raw/results.jsonl`
- 環境 2: DGX Spark（NVIDIA GB10。CUDA + ARM CPU）→ `results/raw/results-dgx.jsonl`。CUDA ホストでは `./run_all_cuda.sh` を使う（bench-candle / bench-burn は `--no-default-features --features cuda` でビルドされる。fandhe-ai は cfg + 実行時プローブのため feature 指定不要）
- 環境 3: NVIDIA GeForce RTX 3060（12 GiB）/ Linux（CUDA。デバイス/tape 再利用モードの fresh/reuse 比較用）→ `results/raw/results-rtx3060.jsonl`（イシュー #925）

## 計測タスク

すべて f32・決定的シード（xorshift64\* を `bench-common` に自前実装、全フレームワークで同一シード・同一生成式の入力）。

- **(a) GEMM**: C = A×B、N = 256 / 512 / 1024 / 2048（GPU は 4096 も）。指標: 中央値・Q1・Q3、GFLOP/s（2N³/median）
- **(b) MLP 学習**: 784→256→10（ReLU）、バッチ 64、合成データ、MSE、手動 SGD（lr 0.01）、100 ステップ。先頭 20 ステップを warmup として除外し、残り 80 ステップの 1 ステップ時間の中央値・Q1・Q3
- **(c) 推論**: 同 MLP の forward のみ、バッチ 64。1 回（= 1 バッチ）あたり時間の中央値・Q1・Q3、バッチ/秒（`throughput_per_s` = 1/median。1 バッチ = 64 件の forward であり、1 件あたりの推論/秒ではない）

## 計測プロトコル（fandhe-ai の計測規約に準拠・拡張）

- warmup 20 回 → 計測 20 回、中央値 + Q1/Q3（学習は 100 ステップ中、先頭 20 を warmup）
- **同期の統一**: 計測区間の終端で必ず結果テンソルをホストへ実体化して全要素を読み出す
  （fandhe-ai: `to_tensor()` + `contiguous().as_slice()` / candle: `to_vec2()` / Burn: `into_data()`）。
  GPU の非同期実行を計測漏れさせない。読み出した checksum は JSON に記録し、フレームワーク間の数値一致確認に使う
- 計測ごとに新しい計算グラフを作る（fandhe-ai は毎回新しい `tape()` / `tape_for(Device::…)`。
  この条件は fandhe-ai の CUDA で tape ごとの初期化コスト約 440〜460 ms を毎回計測区間に含める。`results/summary.md` 環境 2 の備考を参照）
- 重み初期化: candle / Burn は共有 RNG（同一シード）で同一の重み。fandhe-ai の `Sequential::add_linear` は
  内部初期化（シード指定）のため重みの値自体は異なるが、実行時間には影響しない（同一アーキテクチャ・同一入力）
- **`fresh` の計測範囲は fandhe-ai と candle / Burn とで非対称（イシュー #925 レビュー指摘）**:
  「計測ごとに新しい計算グラフを作る」は fandhe-ai の `tape()` / `tape_for(Device::…)` にのみ適用され、
  デバイス・入力テンソルの構築を毎回計測区間内で行う。一方 candle（`bench-candle/src/main.rs`）・
  Burn（`bench-burn/src/main.rs`）は `Device` と入力 `Tensor` をループ外（計測開始前）で 1 回だけ構築し、
  計測区間は `matmul` + ホスト実体化のみを含む。したがって CPU（`tape()` はデバイス選択コストを持たない）
  では実質的に同一プロトコルとみなせるが、**Metal / CUDA では `fresh` の GFLOP/s はフレームワーク間で
  プロトコル同一とは言えない**（fandhe-ai 側にのみ毎回のデバイス/tape 構築コストが乗る）。上記 GEMM
  比較表（Metal・CUDA 環境 2）の fandhe-ai 行はこの固定オーバーヘッドを含んだ数値であり、GEMM カーネル
  単体の速度差として解釈しない。フレームワーク間でプロトコルが完全一致する比較には次の `--mode reuse`
  （`gemm` タスクのみ）を使う

### `--mode fresh|reuse`（イシュー #925。デバイス/tape 再利用モード）

- 既定は `fresh`（上記「計測ごとに新しい計算グラフを作る」プロトコルと完全に同一。既存 JSONL・集計表との互換維持。
  ただし上記のとおり Metal / CUDA では candle / Burn とプロトコル同一ではない点に注意）
- `reuse`（`bench-fandhe` の `gemm` タスクのみ対応。`bench-candle` / `bench-burn` は
  デバイス再利用が API 上の既定設計のため対象外で MEASURE_ERROR を返す）:
  tape/デバイスを 1 回だけ構築し、その構築 + 葉 Var 登録 + 初回 matmul + ホスト実体化までの
  経過時間を `init_s`（JSONL のフィールド。初期化 1 回分のコスト）として分離記録したうえで、
  同一 tape 上で warmup 残り 19 回 → 計測 20 回を回し、`median_s`/`q1_s`/`q3_s` を
  「カーネル実行時間」として記録する。この計測区間（`matmul` + ホスト実体化のみ）は
  candle / Burn の計測区間（デバイス・入力テンソルをループ外で構築済みの `matmul` +
  ホスト実体化）と一致するため、**Metal / CUDA で fandhe-ai を candle / Burn とプロトコル
  同一で GFLOP/s 比較したい場合は `reuse` モードの `median_s`/GFLOP/s を用いる**（`fresh` の
  GFLOP/s ではない）
- **tape 上のノード蓄積に関する注意**: 葉 Var（A・B）は tape 上に 1 回だけ登録して使い回すが、
  matmul の結果ノードは呼ぶたびに tape へ蓄積される（N=2048 で約 16 MiB/回 × 40 回 ≒ 640 MiB。
  N=4096 でも約 2.6 GiB で対象 GPU メモリ内に収まる。長時間・大サイズの reuse 計測では
  メモリ使用量の増加に留意する）
- 使用例: `cargo run --release -p bench-fandhe -- --task gemm --device cuda --size 2048 --mode reuse`

### 要素単位検証（イシュー #970）

`(a)` GEMM の checksum（全要素和）は、要素の入れ替わりや正負誤差の相殺で偶然一致しうる破損を
見逃す。3 バイナリ（`bench-fandhe`/`bench-candle`/`bench-burn`）は `gemm` タスクの各反復で、
結果を参照実装と**要素単位**で突合し、反復間の worst-case を JSONL の 4 フィールド
（`parity_total`・`parity_fail_count`・`parity_max_abs_err`・`parity_max_rel_err`）として記録する。

- **参照実装**: `bench-common::GemmReference`。本体 `backend-cpu::parity::matmul_reference_fma`
  と同じ FMA 契約（f32 `mul_add`・逐次 k 昇順の演算順序固定）を持つ自前 GEMM を、行ブロック分割で
  `std::thread::scope` 並列化したもの（各 `c[i][j]` の累積鎖は k 昇順のまま = 逐次実装と bit 完全
  一致。`bench-common::parity::tests::compute_is_bit_identical_to_sequential_k_ascending`）。
  fandhe-ai 0.4.0（crates.io 版）の facade は parity API を公開しておらず、candle/Burn を参照に
  すると別途バイナリ間で結果を受け渡す仕組みが要る。自前参照は各バイナリが自己完結で計算できる
  ため採用した（f64 累積の参照は「真値との差」という別の指標になり本体契約と整合しないため不採用。
  結果テンソルをファイルへダンプして summarize.py 側で突合する方式は N=4096 で 64 MiB/行になり
  コミット・転送が非現実的なため不採用）
- **閾値**: `PARITY_ABS_TOL = 1e-5`・`PARITY_REL_TOL = 1e-3`。本体の数値一致契約
  （`.claude/rules/coding-rust.md`「バックエンド構成」節）と同値であり、緩和はユーザー承認必須
- **タイミング**（x86_64・12 コアホストでの実測値。`std::thread::available_parallelism()`
  ベースで並列化されるため実効値は環境依存）: 参照 GEMM の計算は warmup 前・計測窓の外で
  1 回だけ（N=1024 で約 210 ms、N=4096 で約 12.8 s）。要素単位の比較（`compare_elementwise`。
  単スレッド）自体は毎反復（warmup 含む）行うが `start.elapsed()` の**後**（O(n²) の比較コストが
  O(n³) の GEMM 計測時間へ混入しないようにするため。checksum の計算・`validate_gemm_checksum`
  は従来どおり計測窓内のまま変更しない）で、N=4096 で 1 回あたり約 61 ms（40 反復合計で
  約 2.4 s）。合計するとバイナリ 1 回起動あたり N=4096 で参照計算 + 全反復比較の合計は
  約 15 秒程度であり、毎反復ではなくバイナリ 1 回起動（`run_all*.sh` の 1 組み合わせ）あたりの
  追加コストとしては許容範囲と判断した（GEMM 自体の計測窓には影響しない）
- **`summarize.py` の判定**: `parity_fail_count > 0`、または 4 フィールドの型・値が不正（`null` 含む）
  な行を「無効（要素誤差超過）」として表で表示し GFLOP/s を `-` にする（`parity_status`）。本フィールド
  追加前の JSONL（キー欄自体が無い）は「無効」ではなく「未検証（旧形式）」として区別する（キー欠損と
  `null` を混同しない。データ有効性節・`--strict` 対象）
- `train`/`infer` タスクは対象外（fandhe-ai の重み初期化が candle/Burn と異なる設計のため checksum
  同様に比較不能。§「計測プロトコル」重み初期化の節を参照）

## 使い方

```bash
cd scripts/bench/framework-compare
./run_all.sh                 # macOS: cpu + metal 全組み合わせ（+ metal reuse スイープ）→ results/raw/results.jsonl
./run_all_cuda.sh            # CUDA ホスト: cuda + cpu 全組み合わせ（+ cuda reuse スイープ）→ results/raw/results-cuda.jsonl
# 個別実行:
cargo run --release -p bench-fandhe -- --task gemm --device metal --size 2048
cargo run --release -p bench-fandhe -- --task gemm --device cuda --size 2048 --mode reuse
# 集計（JSONL → Markdown 表。既定は results/raw/*.jsonl 全件を標準出力へ。
# reuse 行が存在するファイルには (a') 節が追加される。
# コミット済みの results/summary.md は既定動作では上書きされない）:
python3 summarize.py
python3 summarize.py results/raw/results.jsonl --out /tmp/tables.md   # 入力・出力の明示
```

失敗した組み合わせは `results/raw/skipped.log`（CUDA は `skipped-cuda.log`）に理由付きで記録される（数値の捏造はしない）。
`summarize.py` はこの節を集計対象として渡した各入力 JSONL と同一ディレクトリの `skipped*.log` からのみ収集する（入力省略時は従来どおり `results/raw/` 配下が対象。イシュー #971）。
集計は `results/summary.md` を参照。

`summarize.py` は GEMM の checksum（全フレームワーク・全 mode で同一入力のため本来一致するはず）を
size ごとに相互突合し、参照値と外れる行を表で「（無効: checksum 不一致）」表示する
（既定では stderr へ警告のみ、`--strict` を付けると不一致 1 件以上で終了コード 2）。
これとは独立に、要素単位検証（イシュー #970。前節参照）の閾値超過も同じ表で「（無効: 要素誤差超過
fail=<k>/<total>, max_abs=<e>, max_rel=<e>）」と表示し、`--strict` の対象にする（両方に該当する行は
理由を併記する）。各バイナリ側にも `bench-common::validate_gemm_checksum` による縮退 checksum
（全ゼロ・非有限）の emit 前ガードがある（`skipped.log` に理由付きで記録される）。**既知の無効データ**:
Burn(wgpu) Metal GEMM の N>=512 は upstream 既知バグ（`docs/perf/burn-wgpu-metal-gemm-zero-result.md`。
イシュー #965）により結果テンソル全ゼロを返すため無効（`results/summary.md`「データ有効性の注記」参照）。
コミット済みの raw JSONL（`results/raw/*.jsonl`）は本フィールド追加前に計測されたものであり、
要素単位検証は「未検証（旧形式）」表示になる（本 PR で数値を捏造・再計測はしていない。
次回再計測キャンペーンから要素単位検証が有効になる）。**「未検証（旧形式）」行も要素単位検証を
一度も受けていない点では検証済みと同列に扱えないため `--strict` の対象に含まれる**（既定の
非 `--strict` 実行では引き続き警告表示のみで終了コード 0）。このため、コミット済みの旧形式
JSONL（`results/raw/*.jsonl`）に対して `--strict` を付けて実行すると終了コード 2 になる
（`run_all*.sh`・CI は `summarize.py` を `--strict` なしでのみ呼ぶため、この経路は影響を受けない。
要素単位検証つきで再計測した JSONL のみが `--strict` を通過する）。

## 依存ポリシー上の位置づけ

- 本 workspace は許容依存第 9 区分（ベンチ比較対象）の適用範囲拡張として、`candle-core =0.11.0`・`burn =0.21.0` を**本ディレクトリ限定**で保持する（`.claude/rules/deps-policy.md`）
- 本体 workspace（ルート `Cargo.toml` / `Cargo.lock`）への混入は引き続き禁止であり、ルート `Cargo.lock` / `cargo tree` に対する `scripts/check-forbidden-deps.sh` の検査で fail-closed に検出される
- 承認記録（2026-08-28 ユーザー承認・PR #915）・ライセンス実測・統制の全体像は `docs/framework-compare-harness-decision.md` と `docs/license-matrix.md` 8b 節を参照
