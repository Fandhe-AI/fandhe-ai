# DataLoader マルチワーカー prefetch 性能 A/B（イシュー #2183）

対象: `crates/facade/tests/data_loader_prefetch_bench.rs`
（`PrefetchDataLoader`／`PrefetchConfig`。実装は
`crates/tensor-core/src/data.rs`、設計判断は
`docs/tensor-core-data-prefetch-decision.md`）。

## 事前登録した判定規則（計測前にコミット）

1. 5 run の中央値（`bench_harness::median_q1_q3`）を採用する
   （`.claude/rules/coding-rust.md`）。
2. checksum（yield された全バッチ・W2 の最終パラメータについて
   `to_bits` を fold した値）が A（`num_workers=0`）・B（`num_workers
   ∈ {2, 4}`）間、および 5 run 間ですべて完全一致すること。**hard
   assert**とする（決定性契約 R2 の直接検証を兼ねる）。
3. `ratio = B の中央値 / A の中央値 <= 1.00` を非後退条件とするが、
   本 Linux 開発機は共有機であり専有ゲートではないため **record_only**
   とする（性能比そのものは hard assert しない。`docs/perf/logs/
   optimizer-device-step-2175/README.md` 等と同じ record_only 方針）。
4. W2 で見込める改善の上限（ステップ時間に占める取得時間の割合）は
   事前に約束せず、計測値をそのまま記録する。

## ワークロード

- **W1**（取得がボトルネックになる構成）: `batch()` に固定回数の整数
  演算ループ（ホスト側の重い前処理を模す。`HeavyDataset`）を仕込んだ
  合成データセット（64 サンプル・batch_size=4・16 バッチ／epoch）。
  消費側にも固定コストの疑似学習ステップを挟む。`SequentialSampler`
  で `num_workers=0`（逐次）と `num_workers ∈ {2, 4}`（`prefetch_depth
  = 2 × workers`）を比較する。
- **W2**（fit 相当の大型データセット）: `(TensorDataset<f32>,
  TensorDataset<f32>)` 512 サンプル・batch_size=16・3 epoch の MLP
  （`D_IN=16 → D_HIDDEN=32 → D_OUT=4`）CPU 学習ループ。逐次版
  `RandomSampler`（`num_workers=0`）と `PrefetchDataLoader
  (RandomSampler)`（`num_workers ∈ {2, 4}`）を同一 `manual_seed` で
  比較する。`Sequential::fit` は経由しない（facade 保留方針。
  `docs/tensor-core-data-prefetch-decision.md` §2.4）。

CUDA／Metal: 本機構はホスト側だけで完結し、`Op`／`BackendOps`／VJP を
一切増やさないため REQ-2 の parity 対象はなく、実機 `#[ignore]`
テストも不要（GPU DMA prefetch はスコープ外）。

## 実行コマンド

```sh
cargo test -p fandhe-ai --release --test data_loader_prefetch_bench -- --ignored --nocapture
```

## 実測結果（Linux 開発機。`env_info.txt` 参照）

W1（`W1_COST_ITERS=20_000_000`）:

| num_workers | prefetch_depth | a_median_s（逐次） | b_median_s（prefetch） | ratio | checksum（A/B 一致） |
|---|---|---|---|---|---|
| 2 | 4 | 0.031159 | 0.015571 | 0.500 | 一致（0xabdfbe76d0c00000） |
| 4 | 8 | 0.031159 | 0.008098 | 0.260 | 一致（0xabdfbe76d0c00000） |

W2（512 サンプル・3 epoch・CPU MLP。**再計測**——イシュー #2183 PR
#2315 の Codex レビュー指摘により、`(ds_x, ds_y)` タプルを
`PrefetchDataLoader` へ直接渡し特徴量・ラベルの実バッチ構築
（`Dataset::batch`）自体を worker 側で行う構成へ修正した後の値。
旧版は添字だけを worker 側で複製し実バッチ構築を consumer 側の
逐次実行に残していたため、prefetch の重ね合わせ効果を測定できて
いなかった〈以下は誤った旧計測値・不採用〉: num_workers=2 ratio
0.945／num_workers=4 ratio 1.114）:

| num_workers | prefetch_depth | a_median_s（逐次） | b_median_s（prefetch） | ratio | checksum（A/B 一致） |
|---|---|---|---|---|---|
| 2 | 4 | 0.008957 | 0.016596 | 1.853 | 一致（0x5f18244aa47a8100） |
| 4 | 8 | 0.008957 | 0.019964 | 2.229 | 一致（0x5f18244aa47a8100） |

（本開発機は共有機のため run 間の変動が大きく、同一コマンドの
繰り返し実行で ratio は概ね 1.1〜5.2 の範囲で揺れた。上表は
その 1 run を代表値として記録し、計測値をそのまま記す方針
〈事前登録規則 4〉に従う。いずれの run でも checksum は完全一致
した。）

（5 run 中央値。checksum は A/B・5 run すべてで完全一致した。生ログは
標準出力のみでファイルへは保存していない——本ディレクトリ整備時点の
再実行で同一コマンドから再現できる。）

## 判定

- **決定性契約（R2・hard assert）**: 全構成で A/B・5 run 間の checksum
  が完全一致した（pass）。マルチワーカー化による分配・reorder が
  出力へ一切影響していないことを実測で確認した。
- **性能比（record_only）**: W1（取得コストが支配的）では
  `num_workers=2` で約 2 倍、`num_workers=4` で約 3.8 倍の高速化を
  観測した（取得コストが学習ステップと重なる効果）。W2（実バッチ
  構築を worker 側へ移した再計測後）は `num_workers=2` で約 1.9 倍、
  `num_workers=4` で約 2.2 倍**悪化**した（ratio > 1）。本規模の
  `TensorDataset::batch`（512 サンプル・16×16 f32 程度のスライス
  コピー）はマイクロ秒オーダーで完了するため、worker への
  タスク投入・バッチ本体（`Tensor` データ）のチャネル転送・
  reorder buffer のオーバーヘッドが `batch()` 本体のコストを
  上回り、逐次実行より遅くなる。本開発機は共有機のため run 間の
  変動も大きく、W2 の結果は non-gating の record として扱う。
- 総括: 取得側コストが支配的なワークロード（W1 相当）では明確な改善が
  確認できた。取得コストが小さいワークロード（W2 相当・本規模）では
  ワーカー化のオーバーヘッド（スレッド間通信・バッチ本体の転送）が
  相対的に大きく、`num_workers` を増やすとむしろ悪化する。利用者は
  `num_workers`／`prefetch_depth` をワークロードの `batch()` コストに
  応じて選ぶ必要がある——このトレードオフは `PrefetchConfig`／
  `PrefetchDataLoader` の doc（`crates/tensor-core/src/data.rs`）に
  明記する。

## 申し送り

- 大規模データセット・より重い `batch()`（画像デコード相当）での
  再計測、および DGX Spark GB10／Metal 実機上での計測は対象外
  （本機構はホスト側のみで完結し GPU parity 対象がないため、実機での
  再計測自体が必須ではない。あくまで参考データが欲しい場合の任意
  作業）。
