# Metal GEMM 動的タイル選択 計測記録（#188・TASK-1.8f）

イシュー #188「perf(backend-metal): TASK-1.8f 動的タイル選択（行列サイズ別パラメータ化）の実装」の実測記録テンプレート。
受け入れ条件「動的タイル選択（`dispatch_auto`）が simdgroup 版（TASK-1.8c・#40）比で性能向上を示す実測記録」に対応する。

> **選択閾値は #744 で是正済み**: 本ファイルの実測値・本文は当時の記録のまま変更していないが、
> `crate::tile::select`（`select_with_occupancy` 段 1）の正方大形状閾値（下記「実測結果」節の
> 64x64 staged 優位の前提）はその後の staged 経路変更（#533/#538/#572）で実測が逆転したため、
> イシュー #744・2026-08-19 M4 Max 実機実測に基づき是正済み。是正の判断根拠・実測値は
> `docs/perf/metal-tile-select-correction.md` を参照。

## 状態: MSL 構文検証・数値一致は実機検証済み（イシュー #380）。**TFLOPS 実測は #381 で完了**

本ファイルは当初 Linux worktree で作成され、Metal 実機（Apple Silicon）が同一セッションで使用できなかった
ため計測手順・記録テンプレートのみを整備していた。イシュー #380 で Apple Silicon 実機
（M4 Max・macOS 26.6・`stable-aarch64-apple-darwin`）を用い、`gemm_simdgroup_tiled` を含む `gemm.metal` 全体
が `MetalGemm::new` の `newLibraryWithSource` で実機コンパイル成功し（**MSL 構文検証は完了**。当初懸念して
いた「実機での最初の実行が構文検証を兼ねる」は成立し、pass した）、`gemm_dynamic_tile_parity.rs`
（全タイル候補の function constant 組合せを含む 6 件）が数値一致で PASS することを確認済み
（`docs/backend-metal-real-device-testing.md`）。イシュー #381 で本ファイルの主目的である
simdgroup 版と `dispatch_auto` の性能比較を実機実測し、下記「実測結果」節を記入した
（受け入れ条件 A は達成。B は大規模形状〈size=2048・4096、縦長・横長〉で `auto/simdgroup` が
3 ラン全て 1.00 超と明確に達成する一方、size=256・512 では複数ランで 1.00 未満となり未達。
詳細は「受入条件 B の判定」節）。実機 CI 整備自体はイシュー #42
（TASK-1.8e）のスコープ。

## 計測手順（Apple Silicon 実機）

```sh
git fetch origin
git checkout perf/188-metal-dynamic-tile   # 本イシューの実装ブランチ
cargo run -p fandhe-ai-backend-metal --example gemm_bench --release
```

出力形式（`examples/gemm_bench.rs` 参照）:

- `size=<N>` 行: 正方形状（256/512/1024/2048/4096）で naive/tiled/simdgroup/dynamic-tile-auto の TFLOPS と
  `auto_over_simdgroup`・`simdgroup_over_naive` 比
- `shape=(<M>x<N>x<K>)` 行: 縦長（4096x512x512）・横長（512x4096x512）で simdgroup と dynamic-tile-auto の比較
  （`crate::tile::select` の tall/wide 分岐の実測対象）
- `size=<N> candidate=<label>` 行: `GemmVariant::SimdgroupTiled` 候補構成（64x64 staged・32x32 staged・
  32x32 direct）を size=2048 固定形状で明示比較（協調ロード有無の実測比較）

数値一致確認（受け入れ条件に必須の前提）:

```sh
cargo test -p fandhe-ai-backend-metal -- --ignored --nocapture
```

`tests/gemm_dynamic_tile_parity.rs` の全ケース（候補構成別・直接ロード経路・境界形状・`dispatch_auto`・
K ストレスケース）が PASS することを先に確認してから性能値を採用する。

## 実測結果（イシュー #381・2026-08-10 実測）

### 計測環境

| 項目 | 値 |
|------|-----|
| チップ | Apple M4 Max（GPU コア 40・メモリ 64GB。`sysctl -n hw.model` = `Mac16,6`） |
| OS | macOS 26.6 build 25G72 |
| rustc | `rustc 1.96.0 (ac68faa20 2026-05-25)` |
| 計測コミット SHA | `3f7203975887ef3836a003db888b56c29232ccf6`（origin/main。#380 の PR #434 マージ済み） |
| 計測プロトコル | `bench-harness::protocol::run`（`MeasurementConfig::default()` = warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1） |
| 決定的シード | `0xC0FFEE`（`crates/backend-metal/examples/gemm_bench.rs::SEED`） |
| 計測範囲 | 1 ディスパッチごとに A・B のアップロードと C の readback を含む（`MetalGemm::dispatch_variant`／`dispatch_auto` の入口が「1 回の呼び出しで完結」する設計のため。バッファ常駐前提の PoC-v2-4 とは計測範囲が異なる。下記「PoC-v2-4・REQ-8 との関係」参照） |
| 計測衛生 | AC 電源接続。外部 4K/6K ディスプレイ（6400x3600）接続状態（コンポジタ負荷が常時存在。他 GPU 負荷アプリ〈ブラウザ動画・Xcode ビルド・ローカル LLM〉は終了して計測） |
| 統計の扱い | 同一コマンドを 3 回実行（run1/run2/run3）。**表の値は run1 を canonical として採用**（`MeasurementConfig::default()` 自体が warmup 20・計測 20・中央値を内包するため単独で「5 回計測の中央値」下限を満たす。`.claude/rules/coding-rust.md`）。run2/run3 は再現性確認用とし、ラン間最大偏差（%）のみを付記する。生ログは本ファイル末尾の Appendix 参照 |

### 正方形状（naive/tiled/simdgroup/dynamic-tile-auto）

| size | naive TFLOPS | tiled TFLOPS | simdgroup TFLOPS | dynamic-tile-auto TFLOPS | auto/simdgroup | ラン間最大偏差（auto/simdgroup） |
|------|------|------|------|------|------|------|
| 256  | 0.0697 | 0.1382 | 0.1496 | 0.1320 | 0.8823 | 15.7%（0.8145〜0.9531） |
| 512  | 0.3988 | 0.4913 | 0.5484 | 0.5216 | 0.9510 | 60.8%（0.9352〜1.5137。run3 が外れ値。後述） |
| 1024 | 0.7755 | 0.9949 | 1.1171 | 1.2909 | 1.1556 | 5.8%（1.1556〜1.2230） |
| 2048 | 0.8475 | 1.3602 | 1.6015 | 2.5001 | 1.5611 | 16.7%（1.5611〜1.8214） |
| 4096 | 0.9198 | 1.2207 | 1.7432 | 3.0283 | 1.7372 | 31.6%（1.7372〜2.2866） |

### 非正方形状（縦長・横長）

| shape (MxNxK) | simdgroup TFLOPS | dynamic-tile-auto TFLOPS | auto/simdgroup | ラン間最大偏差（auto/simdgroup） |
|------|------|------|------|------|
| 4096x512x512（縦長） | 0.9223 | 1.1244 | 1.2191 | 16.3%（1.2191〜1.4174） |
| 512x4096x512（横長） | 0.9446 | 1.1782 | 1.2473 | 11.1%（1.2473〜1.3858） |

### 候補構成別（size=2048 固定・協調ロード有無比較）

| candidate | BM | BN | BK | WM | WN | staged | TFLOPS | ラン間最大偏差 |
|-----------|----|----|----|----|----|--------|--------|------|
| bm64_bn64_bk16_staged | 64 | 64 | 16 | 2 | 2 | true | 2.3572 | 13.5%（2.0388〜2.3572） |
| bm32_bn32_bk16_staged | 32 | 32 | 16 | 2 | 2 | true | 2.4030 | 12.4%（2.1053〜2.4030） |
| bm32_bn32_bk16_direct | 32 | 32 | 16 | 2 | 2 | false | 1.8663 | 15.4%（1.5783〜1.8663） |

## 受入条件 B の判定（`dispatch_auto` が simdgroup 単独と同等以上か）

- **判定方法**: `auto_over_simdgroup` 比で判定する（PoC-v2-4 の絶対値とは比較しない。計測範囲が異なるため）。
- **判定対象**: 見出しどおり `dispatch_auto` が simdgroup 単独と同等以上かを、実測した全形状
  （正方形状 5 種・縦長・横長）で判定する。カーネル時間が支配的で選択差が現れる size=2048・4096・
  縦長・横長では明確に達成している一方、size=256・512（オーバーヘッド支配領域）では複数ランで
  1.00 を割り込んでおり、**全形状を通した受入条件 B は「部分達成（未達を含む）」と記録する**
  （小規模形状を対象から除外して「達成」と結論づけない）。
- **size=2048・4096・縦長・横長: `auto/simdgroup` は 3 ラン全てで 1.00 を上回る**
  （size=2048: 1.56〜1.82／size=4096: 1.74〜2.29／縦長・横長: 1.22〜1.42）。この範囲では
  `dispatch_auto` が simdgroup 単独比で明確に性能向上を示している。
- **size=256・512: 複数ランで 1.00 未満（未達）**。`tile::select` は `SMALL=64` 未満でのみ
  単一 simdgroup 8x8 を返すため、256・512 では 32x32 系候補が選ばれる。size=256 は 3 ラン全て
  1.0 未満（0.8145〜0.9531）、size=512 も run1・run2 が 0.9352〜0.9510 と 1.0 未満（run3 のみ
  1.51 と跳ねている）。これは 1 ディスパッチあたりのアップロード・readback・コマンドバッファ
  投入コストがカーネル実行時間に対し支配的なオーバーヘッド起因と考えられるが、**原因の如何に
  かかわらず利用者が観測する `dispatch_auto` の性能低下であることに変わりはなく**、この事実
  自体は「劣化なし」として扱わない。対応（`tile::select` の候補選択改善・小規模形状の別経路化等）
  は #382 の境界形状データを踏まえて人間承認のもとで検討する（下記「選択閾値の確定」節）。
- **候補構成別（size=2048）の診断**: 3 ラン全てで `bm32_bn32_bk16_staged`（32x32・協調ロードあり）が
  `bm64_bn64_bk16_staged`（64x64・協調ロードあり）と近接またはわずかに上回り（run1: 2.4030 vs 2.3572、
  run2: 2.2058 vs 2.0388、run3: 2.1053 vs 2.2055〈この回のみ 64x64 が上回る〉）、`bm32_bn32_bk16_direct`
  （協調ロードなし）は 3 ラン全てで両 staged 構成より明確に低い（1.58〜1.87）。`tile::select` の
  `LARGE=512` 分岐は size=2048（m=n=2048 ≥ 512）で `CANDIDATES[0]`（64x64 staged）を選択するが、
  実測では 32x32 staged が同等かやや優位という傾向が 3 ラン中 2 ラン（run1・run2）で観測された
  （run3 は僅差で逆転）。**この差は境界付近で明確な劣化と言えるほど大きくなく、また `tile.rs` の
  閾値・`CANDIDATES` は本イシューでは変更しない**（下記「選択閾値の確定」参照）。
- **熱・実行順序バイアス（既知の制約）**: 単一ラン内の実行順は naive→tiled→simdgroup→auto
  固定であり、`auto` が最も GPU が温まった状態で計測される。size=512 の run3（auto/simdgroup=1.51）
  はこのバイアスが強く出た可能性がある一方、naive/tiled/simdgroup 自体も run3 で軒並み低い値
  （naive 0.19、simdgroup 0.30）を示しており、単なる実行順バイアスというより外部ディスプレイの
  コンポジタ負荷等による当該ラン全体のスループット低下の影響が大きいと考えられる。example の
  実行順・計測条件は変更していない（計画の判定規則どおり）。
- **後続対応の提案**（PR 本文の対象外節に記載。自動運転モードのため新規 Issue は起票しない）:
  size=2048 帯での 32x32 staged と 64x64 staged の優劣が僅差である点、`LARGE=512` 境界
  付近の候補選択の妥当性、および size=256・512 で `auto/simdgroup` が 1.00 を割り込む
  受入条件 B 未達領域への対応（候補選択改善・小規模形状の別経路化等）は、境界形状 TFLOPS の
  実測を扱う #382 の判断材料として引き継ぐ。

## 選択閾値の確定（#382・人間承認を前提とする。本イシューでは変更しない）

`crates/backend-metal/src/tile.rs` の `select` 関数・`CANDIDATES` は下記の暫定値のままである
（実測前の初期値。MLX steel の実装傾向を参考にした推定）:

- 微小形状しきい値: `SMALL = 64`（`m/n/k` のいずれかがこれ未満なら単一 simdgroup 8x8）
- 大形状しきい値: `LARGE = 512`（`m`・`n` ともこれ以上なら 64x64 staged）
- 縦長・横長判定: `ASPECT_RATIO = 2`（`m >= n*2` で縦長、`n >= m*2` で横長）

**本イシュー（#381）では観測事実の記録に留める。閾値の確定・変更は境界形状データ（#382）と
人間承認を前提とする**（`.claude/rules/deps-policy.md`・`security.md` の「ガードレール閾値・
テスト許容誤差の変更はユーザー承認必須」と同趣旨。ここでの `tile.rs` 定数もユーザー承認なしに
実装エージェントが変更してよい対象ではないと判断した）。観測された傾向（事実の記録）:

- `LARGE=512` 境界の上（size=2048）で `CANDIDATES[0]`（64x64 staged）が選ばれるが、実測では
  32x32 staged が同等かやや優位な回が多かった（3 ラン中 2 ラン）。境界を下げる・候補順を
  入れ替える等の判断は #382 の境界形状（256〜1024・アスペクト比 1.5〜3 帯）の追加データを
  待って行う。
- `SMALL=64` 未満（m/n/k のいずれかが 64 未満）の形状は本ベンチでは未計測（size=256・512 は
  m=n=k がいずれも 256・512 であり `SMALL=64` 分岐の実測にはならない。より小さい形状の実測は
  #382 のスコープ）。一方、`LARGE=512` 未満・`SMALL=64` 以上の中間帯である size=256・512 実測
  では `auto/simdgroup` がオーバーヘッド支配で 1.0 を割り込む回があった（上記「受入条件 B の
  判定」節）。これは 32x32 系候補自体の遅さではなくディスパッチ 1 回あたりの固定コストに
  よるものと考えられるが、`SMALL` 閾値自体の妥当性についてはここでは判断材料としない
  （`SMALL=64` 未満の実測データがないため）。

## PoC-v2-4・REQ-8 との関係

`docs/performance-targets.md:28`（Metal f32 対 PyTorch MPS 23.2%・確定）は PoC-v2-4（バッファ常駐前提の
計測）由来であり、本イシューの実測は計測範囲（アップロード・readback を含む）が異なるため、
本実測の絶対 TFLOPS が PoC-v2-4 実測値（naive 1.271／tiled 2.123／simdgroup 3.134 TFLOPS @4096）より
低く出ていることは REQ-8 の確定行に影響しない。

## 未実施・後続作業

- ~~本ファイルの「実測結果」節は Apple Silicon 実機での `cargo run --release` 実行後に埋める~~ → **#381 で完了**
- ~~選択閾値の確定後、`crate::tile::select`/`CANDIDATES` のコメント（「暫定値」の記述）を実測確定版へ更新する
  （境界形状データを扱う #382 の結果を待つ後続作業。本イシューでは変更しない）~~ → **#382 は完了・ただし
  コメント更新は未実施のまま残存**。#382 は境界形状（256〜1024）の実測から `METAL_SIMDGROUP_MIN_DIM`
  （`crates/tensor-core/src/dispatch.rs`、現行値 512）の**変更提案（384 への引き下げ）を記録するに留め、
  コード変更・Issue 起票は行わないと確定した**（出典: `docs/perf/dispatch-boundary-measurement.md`
  「`METAL_SIMDGROUP_MIN_DIM` の妥当性判定（#382）」節。受入基準どおり「記録に留め、実施は別レビュー・
  別 PR・ユーザー承認」）。したがって本ファイルが指す `crate::tile::select`/`CANDIDATES` のコメント更新
  （動的タイル選択閾値側の「暫定値」記述の実測確定版への更新）も、#382 の変更提案が実際にコードへ適用
  される別 PR（ユーザー承認後）まで着手できず、**残存項目として据え置く**（イシュー #387 の総括反映では
  ドキュメントのみを対象としコード変更を含めないため、本イシューでも実施しない）
- 実機 CI 整備（TASK-1.8e・#42）と関連付けて追跡する（`.claude/rules/out-of-scope-tracking.md`）

## Phase D 完了時点再計測（#547）

イシュー #547「bench(backend-metal): Phase D 完了時点の f32/f16 スループットと対 PyTorch MPS 比を
再計測・記録」の実測記録。GEMM 最適化ツリー（ルート #479）の Phase D（親 #530）子イシュー
D-1〜D-9・D-11（#547 自身は D-10）が全て完遂した時点でのスナップショット計測であり、
A-1（#481・`docs/perf/gemm-optimization-baseline.md`）で確定した基準系列に対する改善率を算出する。

### 状態: 未計測。実機セッションで消化

本節は Linux worktree で作成され、Metal 実機（Apple Silicon）が同一セッションで使用できないため
計測手順・記録テンプレートのみを整備する（Phase D 先例 `docs/perf/metal-gemm-float4-staged-load.md`
等と同方式）。実機到達可能なセッションが下記手順で計測し、本節の表を実測値で埋める。

### Phase D 有効変更の一覧（本計測が含む経路の明記）

計測時点で `MetalGemm::dispatch_auto`／`dispatch_f16_prepared_unverified` の本番経路に**適用済み**の
変更と、実装済みだが本番経路には**未適用**の変更を区別して記録する（計測結果の解釈に必須）。

| 子イシュー | 内容 | 本番経路への適用状態 |
|---|---|---|
| D-1（#532） | `tile::CANDIDATES` へ MLX classic 未収録 3 構成を追加（計 8 構成） | **適用済み**（`crates/backend-metal/src/tile.rs::CANDIDATES`） |
| D-2（#533） | staged ロードの `float4` ベクトル化 | **適用済み**（`shaders/gemm.metal` の staged ロード経路。function constant 分岐なし・無条件） |
| D-3（#535） | `TileConfig::validate` へ整除制約検証を追加 | **適用済み**（実機非依存の検証ロジック） |
| D-4（#536） | 蛇行（serpentine）走査順の移植 | **適用済み**（`shaders/gemm.metal` epilogue。無条件） |
| D-5（#538） | threadgroup memory のパディング（`TGP_PAD`） | **適用済み**（`staged=true` の全候補で `TGP_PAD_ELEMS=4` が有効。`tile.rs::TileConfig::pad`） |
| D-6（#540） | threadgroup ID スウィズルの実験 | **未適用**（`tile::SWIZZLE_ENABLED = false` が本番既定。PR #661 codex-review 指摘: 未検証のスウィズルを本番経路へ無条件適用しない） |
| D-7a（#541） | occupancy 目標算出の仕組み | 算出機構のみ（`dispatch_auto` からは未参照） |
| D-7b（#542） | occupancy 判定のタイル選択への組み込み | **未適用**（`docs/perf/metal-gemm-occupancy-select.md`「状態」節: `MetalGemm::dispatch_auto` は `tile::select`〈形状のみ〉を呼び続けており `select_with_occupancy` は未接続。非劣化確認後に別 PR で切替予定） |
| D-8（#544） | Morton 順マッピングの適用余地調査 | **不採用**（`docs/backend-metal-morton-mapping-decision.md`: 標準 `simdgroup_matrix` API 下では適用不可と判断） |
| D-9（#546） | 非公式 `simdgroup_async_copy` 系 AIR intrinsic | **不採用**（`docs/backend-metal-async-copy-decision.md`） |
| D-11（#549） | MLX classic 対比・NAX 経路非適用の記録 | ドキュメントのみ（コード変更なし） |

まとめ: 本計測は D-1〜D-5（CANDIDATES 拡充・float4 ベクトル化・整除検証・serpentine 走査・
TGP パディング）を含む経路の実測であり、D-6（swizzle）・D-7b（occupancy 選択）は未適用のまま。
`tile::select` の選択ロジック自体（`SMALL`/`LARGE`/`ASPECT_RATIO` 閾値）は #381（D-10 の前身相当）
時点から変更していない。

### 計測手順（Apple Silicon 実機）

数値一致確認（性能値採用の前提）:

```sh
cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture
```

全ケース PASS を確認してからベンチを実行する。

```sh
cargo run -p fandhe-ai-backend-metal --example gemm_bench --release
cargo run -p fandhe-ai-backend-metal --example gemm_f16_bench --release
```

上記 2 コマンドを**各 5 回独立実行**し、size ごとに 5 個の TFLOPS 値の中央値を採用する
（`docs/perf/metal-f16-vs-mps-f16.md` §「実測結果」と同一方式。`MeasurementConfig::default()`
自体が warmup 20・計測 20・中央値を内包するため、5 プロセス独立実行との組み合わせで
「5 回計測の中央値」下限〈`.claude/rules/coding-rust.md`〉を二重に満たす）。

PyTorch 側は一時 venv（リポジトリ管理外。`.venv-mps-bench` 先例）で以下を実行する:

```sh
python3 -m venv .venv-mps-bench
source .venv-mps-bench/bin/activate
pip install torch
python3 scripts/bench/gemm_bench_torch_mps_f32.py
python3 scripts/bench/gemm_bench_torch_mps_f16.py
```

Rust 側と同様に各 5 回独立実行し、size ごとの中央値を採用する。

計測衛生（#381・#383 先例と同方式）: AC 電源接続、外部ディスプレイのコンポジタ負荷を許容するが
他 GPU 負荷アプリ（ブラウザ動画・Xcode ビルド・ローカル LLM 等）は終了する。Rust 側・PyTorch 側の
同時実行を避け、各ラン前後に `pgrep -fl "gemm_bench|gemm_f16_bench|gemm_bench_torch_mps"` で他
プロセスとの競合がないことを確認する（競合検出時は破棄・取り直す）。

### 計測環境（実測時に記入）

| 項目 | 値 |
|------|-----|
| チップ | （未計測） |
| OS | （未計測） |
| rustc | （未計測） |
| torch | （未計測） |
| 計測コミット SHA | （未計測） |
| 計測プロトコル | `bench-harness::protocol::run`（warmup 20・計測 20・中央値）を 5 回独立実行し size ごとに中央値採用（Rust・PyTorch 双方） |
| 決定的シード | `0xC0FFEE` |
| 同期境界 | Rust: コマンドバッファ完了待ち／PyTorch: `torch.mps.synchronize()` |

### f32 実測結果（対 系列 (b) #381 run1 改善率）

分母は `docs/perf/gemm-optimization-baseline.md` §2 の系列 (b)（#381 canonical run1。同一計測境界
「1 ディスパッチごとに A・B アップロード＋C readback を含む」）。系列 (a)（PoC-v2-4・バッファ常駐）
とは計測境界が異なるため比較しない（同ドキュメント §2「基準系列の決定」）。

| size | naive TFLOPS | tiled TFLOPS | simdgroup TFLOPS | dynamic-tile-auto TFLOPS | 対 #381 run1 改善率（dynamic-tile-auto） |
|------|------|------|------|------|------|
| 512  | （未計測） | （未計測） | （未計測） | （未計測） | 分母 0.5216 |
| 1024 | （未計測） | （未計測） | （未計測） | （未計測） | 分母 1.2909 |
| 2048 | （未計測） | （未計測） | （未計測） | （未計測） | 分母 2.5001 |
| 4096 | （未計測） | （未計測） | （未計測） | （未計測） | 分母 3.0283 |

改善率 = 本計測 dynamic-tile-auto TFLOPS ÷ 上記分母（#381 run1）。

### f32 対 PyTorch MPS 比（参考値・計測境界差の注記付き）

**REQ-8 の分母・分子には使わない**。`dispatch_auto` は転送込み境界のため、`docs/performance-targets.md`
§4 の同期方式契約（ホスト転送を伴わない完了待ち）を単独では満たさない。§4 準拠の f32 prepared
入口整備・確定計測は Phase F の #572 のスコープ（`docs/perf/gemm-optimization-baseline.md` §2 参照）。
#572 で追加した `MetalGemm::dispatch_tiled_prepared`（§4 準拠 prepared 入口）による確定計測は
`docs/perf/metal-floor-remeasurement.md` へ記録する（本節の参考値とは別系列）。

| size | Metal f32 TFLOPS（dynamic-tile-auto。5 回中央値） | PyTorch MPS f32 TFLOPS（5 回中央値） | Metal/PyTorch 比（参考値） |
|------|------|------|------|
| 512  | （未計測） | （未計測） | （未計測） |
| 1024 | （未計測） | （未計測） | （未計測） |
| 2048 | （未計測） | （未計測） | （未計測） |
| 4096 | （未計測） | （未計測） | （未計測） |

### f16 実測結果（対 #383 改善率・対 MPS f16 比）

分母は `docs/perf/metal-f16-vs-mps-f16.md`「実測結果（イシュー #383）」節（同一計測境界・
`dispatch_f16_prepared_unverified`。§4 準拠の prepared 境界のため対 MPS 比も直接比較可）。

| size | Metal f16 TFLOPS（5 回中央値） | 対 #383 改善率 | PyTorch MPS f16 TFLOPS（5 回中央値） | Metal/PyTorch f16 比 |
|------|------|------|------|------|
| 512  | （未計測） | 分母 1.1554 | （未計測） | （未計測） |
| 1024 | （未計測） | 分母 2.1777 | （未計測） | （未計測） |
| 2048 | （未計測） | 分母 2.4426 | （未計測） | （未計測） |
| 4096 | （未計測） | 分母 2.2411 | （未計測） | （未計測） |

改善率 = 本計測 Metal f16 TFLOPS ÷ 上記分母（#383）。Metal/PyTorch f16 比 = 本計測 Metal f16
TFLOPS ÷ 本計測 PyTorch MPS f16 TFLOPS（REQ-8 の比較定義。size ごとの比の中央値ではない）。

### REQ-8 下限値との関係（変更しない）

**本節は REQ-8 下限値（Metal f32 23.2%・f16 18.6% 等の既定行。`docs/performance-targets.md` §2）を
一切変更しない**。改善率・対 MPS 比の算出結果を下限値へ反映する判断は Phase F の人間承認タスク
（#577）へ申し送る（`docs/perf/gemm-optimization-baseline.md` §0「本ドキュメントは REQ-8 の下限値・
実測比率の数値を一切変更しない」と同方針）。

## Appendix: 生ログ（3 ラン分）

### run1（canonical）

```text
size=256 naive_tflops=0.0697 tiled_tflops=0.1382 simdgroup_tflops=0.1496 dynamic_tile_auto_tflops=0.1320 auto_over_simdgroup=0.8823 simdgroup_over_naive=2.1454
size=512 naive_tflops=0.3988 tiled_tflops=0.4913 simdgroup_tflops=0.5484 dynamic_tile_auto_tflops=0.5216 auto_over_simdgroup=0.9510 simdgroup_over_naive=1.3752
size=1024 naive_tflops=0.7755 tiled_tflops=0.9949 simdgroup_tflops=1.1171 dynamic_tile_auto_tflops=1.2909 auto_over_simdgroup=1.1556 simdgroup_over_naive=1.4405
size=2048 naive_tflops=0.8475 tiled_tflops=1.3602 simdgroup_tflops=1.6015 dynamic_tile_auto_tflops=2.5001 auto_over_simdgroup=1.5611 simdgroup_over_naive=1.8898
size=4096 naive_tflops=0.9198 tiled_tflops=1.2207 simdgroup_tflops=1.7432 dynamic_tile_auto_tflops=3.0283 auto_over_simdgroup=1.7372 simdgroup_over_naive=1.8953
shape=(4096x512x512) simdgroup_tflops=0.9223 dynamic_tile_auto_tflops=1.1244 auto_over_simdgroup=1.2191
shape=(512x4096x512) simdgroup_tflops=0.9446 dynamic_tile_auto_tflops=1.1782 auto_over_simdgroup=1.2473
size=2048 candidate=bm64_bn64_bk16_staged tflops=2.3572
size=2048 candidate=bm32_bn32_bk16_staged tflops=2.4030
size=2048 candidate=bm32_bn32_bk16_direct tflops=1.8663
```

### run2

```text
size=256 naive_tflops=0.1162 tiled_tflops=0.1341 simdgroup_tflops=0.1429 dynamic_tile_auto_tflops=0.1362 auto_over_simdgroup=0.9531 simdgroup_over_naive=1.2297
size=512 naive_tflops=0.3969 tiled_tflops=0.4834 simdgroup_tflops=0.5362 dynamic_tile_auto_tflops=0.5015 auto_over_simdgroup=0.9352 simdgroup_over_naive=1.3509
size=1024 naive_tflops=0.7330 tiled_tflops=0.9744 simdgroup_tflops=1.1431 dynamic_tile_auto_tflops=1.3981 auto_over_simdgroup=1.2230 simdgroup_over_naive=1.5594
size=2048 naive_tflops=0.8807 tiled_tflops=1.1792 simdgroup_tflops=1.5241 dynamic_tile_auto_tflops=2.4505 auto_over_simdgroup=1.6078 simdgroup_over_naive=1.7307
size=4096 naive_tflops=0.8537 tiled_tflops=0.9717 simdgroup_tflops=1.0409 dynamic_tile_auto_tflops=2.1868 auto_over_simdgroup=2.1008 simdgroup_over_naive=1.2193
shape=(4096x512x512) simdgroup_tflops=0.7066 dynamic_tile_auto_tflops=1.0015 auto_over_simdgroup=1.4174
shape=(512x4096x512) simdgroup_tflops=0.7231 dynamic_tile_auto_tflops=1.0021 auto_over_simdgroup=1.3858
size=2048 candidate=bm64_bn64_bk16_staged tflops=2.0388
size=2048 candidate=bm32_bn32_bk16_staged tflops=2.2058
size=2048 candidate=bm32_bn32_bk16_direct tflops=1.5783
```

### run3

```text
size=256 naive_tflops=0.0836 tiled_tflops=0.1060 simdgroup_tflops=0.1195 dynamic_tile_auto_tflops=0.0973 auto_over_simdgroup=0.8145 simdgroup_over_naive=1.4296
size=512 naive_tflops=0.1896 tiled_tflops=0.2512 simdgroup_tflops=0.2978 dynamic_tile_auto_tflops=0.4507 auto_over_simdgroup=1.5137 simdgroup_over_naive=1.5705
size=1024 naive_tflops=0.7722 tiled_tflops=0.9702 simdgroup_tflops=1.1325 dynamic_tile_auto_tflops=1.3392 auto_over_simdgroup=1.1825 simdgroup_over_naive=1.4666
size=2048 naive_tflops=0.9334 tiled_tflops=0.9399 simdgroup_tflops=1.1855 dynamic_tile_auto_tflops=2.1593 auto_over_simdgroup=1.8214 simdgroup_over_naive=1.2702
size=4096 naive_tflops=0.6146 tiled_tflops=0.8708 simdgroup_tflops=1.0922 dynamic_tile_auto_tflops=2.4974 auto_over_simdgroup=2.2866 simdgroup_over_naive=1.7770
shape=(4096x512x512) simdgroup_tflops=0.7657 dynamic_tile_auto_tflops=1.0448 auto_over_simdgroup=1.3646
shape=(512x4096x512) simdgroup_tflops=0.7828 dynamic_tile_auto_tflops=1.0585 auto_over_simdgroup=1.3521
size=2048 candidate=bm64_bn64_bk16_staged tflops=2.2055
size=2048 candidate=bm32_bn32_bk16_staged tflops=2.1053
size=2048 candidate=bm32_bn32_bk16_direct tflops=1.6738
```
