# Metal GEMM の framework-compare ギャップ分析（#937）

イシュー #937「perf(backend-metal): 大サイズ GEMM の性能改善検討（candle 比 約 2 倍差）」の実装記録。

## 状態: 分析・採否判断は確定（Linux worktree）。改善効果の実測は Mac 実機セッション待ち

本イシューの実装（`crates/backend-metal/src/context_cache.rs` によるプロセス内常駐
キャッシュ）は、後述 §2.1 のとおり**イシュー #930・PR #948 で既に main へ実装済み**
だった。したがって本 PR はコード変更を伴わず、(a) #930/#948 が本イシューの受け入れ
条件 1（改善実装）を既に満たしていることの確認、(b) ギャップの定量分解・残存候補の
採否判断、(c) REQ-8 との突合（受け入れ条件 2）を記録する docs-only の PR とする。

## 1. 出典・現象の再整理

`scripts/bench/framework-compare/results/summary.md`「(a) GEMM」Metal 節（計測環境
Apple M4 Max・macOS 26.6.2・計測日 2026-08-28・PR #915）:

| N | fandhe-ai | candle 0.11.0 | Burn 0.21.0 |
|---|-----------|---------------|-------------|
| 256 | 5.441 ms（6.2 GFLOP/s） | 257.6 µs（130.2） | 1.493 ms（22.5） |
| 512 | 5.724 ms（46.9） | 519.0 µs（517.2） | 3.044 ms（88.2） |
| 1024 | 7.894 ms（272.1） | 1.086 ms（1976.7） | 3.613 ms（594.3） |
| 2048 | 14.576 ms（1178.6） | 4.958 ms（3464.9） | 5.548 ms（3096.7） |
| 4096 | 46.415 ms（2961.1） | 23.635 ms（5815.1） | 13.194 ms（10416.9） |

N=4096 で candle 比 約 2.0 倍・Burn 比 約 3.5 倍の実行時間差（イシュー #937 タイトルの
「candle 比 約 2 倍差」と整合）。

## 2. ギャップの分解

### 2.1 per-call 資源再構築の固定費（本イシュー実装時点で解消済み）

`scripts/bench/framework-compare/bench-fandhe/Cargo.toml` は `fandhe-ai = "=0.3.0"`
（crates.io 公開版。deps-policy.md 第 9 区分の適用範囲拡張）へ完全固定している。
v0.3.0 は 2026-08-23 公開（`docs/crates-io-publishing-order.md` §10）であり、
本イシューが解消対象とした `MetalBackendOps` の per-call `MetalContext::new()`／
`MetalGemm::new()` 都度構築（診断 #927・`docs/perf/metal-fixed-overhead-diagnosis.md`
が特定した「約 5 ms・N 非依存」の固定費）は、その後のイシュー #930・PR #948
（`crates/backend-metal/src/context_cache.rs`。2026-08-28 マージ、本 doc 執筆時点で
main 上に存在）で解消されている。**つまり summary.md の上表は #930/#948 適用前の
コード（v0.3.0）で計測された数値であり、#930/#948 による改善は現行の
framework-compare ハーネスにはまだ反映されていない。**

summary.md 自身も同一事実を計測ノートとして記録済み（同ファイル「Metal 表のプロト
コル注意（イシュー #925 レビュー指摘）」節）: fandhe-ai 行は `--mode fresh`
（既定）計測であり、`tape_for(Device::Metal)` をループの各回内で再構築する一方、
candle／Burn はデバイス・入力テンソルをループ外で 1 回だけ構築する。これは
「GEMM カーネル単体の速度としてではなく、fandhe-ai の『計測ごとに新規グラフを
作る』運用コストを含む数値」（summary.md 該当節）であり、#927/#930 が特定・解消
した固定費と同一の構造要因である。

固定費の実測値（約 5 ms・N 非依存。#927 診断）は N=256〜512 のギャップ（fandhe-ai
5.441/5.724 ms 対 candle 257.6/519.0 µs、差 約 5.2/5.2 ms）とほぼ一致する。N=4096
では 46.415 ms 対 23.635 ms（差 約 22.8 ms）であり、固定費 5 ms を差し引いても
約 17.8 ms の差が残る。したがって**固定費は N=256〜512 のギャップの大半を説明する
一方、N=4096（本イシュー表題の「大サイズ」）のギャップは固定費だけでは説明できず、
§2.2・§2.3 の要因が主要因である**。

### 2.2 計測境界の非対称

`scripts/bench/framework-compare/bench-fandhe/src/main.rs` の GEMM 計測窓は毎
イテレーションで A/B のホスト→デバイスアップロード（N=4096・f32 正方で
4096×4096×4 バイト = 64 MiB × 2 = 128 MiB）＋`dispatch_auto` のカーネル実行＋C
readback を含む（`MetalGemm::dispatch_auto` 内で共通処理。#927 §2「計測窓の定義」）。
candle・Burn 側がループ外でデバイス常駐化した入力を使う（§2.1 引用のプロトコル
注意）ため、この転送コストは fandhe-ai 側にのみ計測窓へ乗る非対称がある。

`docs/perf/metal-fixed-overhead-diagnosis.md` §3 の P6/P7 分解（P6: 都度構築
end-to-end、P7: 資源再利用のみ）はいずれも転送・バッファ確保を含む設計のため
（同 doc P6/P7 定義: 「A/B のホスト→デバイスアップロード・Metal バッファ確保・
カーネル実行・同期・C readback を反復ごとに行う」）、P6−P7 の差分（固定費）には
転送コストは現れない。すなわち転送コストの寄与分は §2.1 の固定費とは別勘定であり、
本 doc 執筆時点（Linux worktree）では定量化できない実測待ち事項である（§5 手順
「gemm_f32_prepared_bench」参照）。

### 2.3 カーネル本体差

`docs/perf/metal-gemm-bottleneck-diagnosis.md`（#487）の解析値は、staged タイルの
arithmetic intensity が size に依らずほぼ一定（15.06〜15.88 FLOP/byte。同 doc
§3.2「解析値からの暫定観察」）であることを示す。GEMM の理論 arithmetic intensity
は size に比例して増大するはずであり、この頭打ちは K タイルごとの device メモリ
再ロード構造（threadgroup 間・K タイル間のキャッシュ再利用を論理ロード量の分母に
含めない計測方式に起因。同 doc §5.2 コメント）を示唆する。

candle は MLX Steel 系カーネル、Burn は CubeCL 生成カーネルを使用しており、
本リポジトリの自作カーネルとは異なるタイル・パイプライン構成を採る。残存する
カーネル本体側の改善候補と既存の採否判断の突合:

| 候補 | 既存判断 | 状態 |
|------|---------|------|
| アラインメント特化ロード分岐（`align_M/N/K` function constant） | `docs/backend-metal-aligned-load-decision.md`（#752 保留 → #808 格下げ） | 不採用（格下げ済み） |
| `simdgroup_async_copy` 系 AIR intrinsic | `docs/backend-metal-async-copy-decision.md`（#546） | 不採用（非公式 intrinsic） |
| MLX classic 経路・NAX 経路 | `docs/backend-metal-mlx-classic-nax-decision.md`（#549） | 不採用 |
| Morton 順レーン→要素マッピング | `docs/backend-metal-morton-mapping-decision.md`（#544） | 適用不可（標準 simdgroup_matrix API 制約） |
| split-K ディスパッチ分岐 | `docs/backend-metal-splitk-decision.md`（#810）・`docs/perf/metal-gemm-splitk-shapes.md` | 対象形状は K 支配的非正方のみ。N=4096 正方（本イシュー対象）は**対象形状外** |

上記のとおり、カーネル本体の再設計候補（2 段パイプライン化・double-buffering 等の
Steel/CubeCL 型構成）は既存の採否判断がいずれも不採用・適用不可・対象外であり、
未検討の候補は「本体アーキテクチャの再設計」のみに絞られる。これは実測 A/B が
必須（Mac 実機）かつ既存カーネル・境界検査（`.claude/rules/coding-rust.md`
「カーネル実装の境界検査」節）を変更しない前提での再設計コストが大きいため、
本イシューでは実装せず §6 のスコープ外事項として記録する。

## 3. 改善候補の採否判断（受け入れ条件 1: 改善実装または不採用判断）

| 候補 | 採否 | 根拠 |
|------|------|------|
| 候補 1: コンテキスト／パイプラインキャッシュ（per-call 固定費解消） | **実装済み**（#930/#948。本 doc §2.1） | N=256〜512 ギャップの主要因を解消。本イシューの受け入れ条件 1 を充足 |
| 候補 2: デバイス常駐テンソル化・アップロード削減（facade/autodiff 側） | 保留（本イシュー対象外） | facade/autodiff の API 設計変更を伴い、GEMM バックエンドの改善スコープを超える（§6） |
| 候補 3: カーネル本体の再設計（2 段パイプライン化等） | 保留（実測 A/B 必須） | §2.3 の既存採否判断（#544/#546/#549/#808/#810）といずれも重複せず、新規の実測 A/B が前提。Mac 実機セッションへ申し送り（§6） |
| 候補 4: split-K | **対象外** | 対象形状（K 支配的非正方）が本イシューの対象（N=4096 正方）と異なる（#810） |

## 4. REQ-8 との突合（受け入れ条件 2）

REQ-8 の性能下限は「対 PyTorch」を分母とする指標であり、candle・Burn は REQ-8 の
判定対象ではない（`docs/perf/performance-floor-decision.md` は一貫して「Metal f32
対 PyTorch MPS」等 PyTorch を分母とする表記のみを持つ。同 doc §10 追補〈#577〉が
現行確定値）。

現行の確定下限（`docs/perf/performance-floor-decision.md` §10・#577）:

| ペア | 段階 | 下限 |
|------|------|------|
| Metal f32 対 PyTorch MPS | 最適化後 | 10% |
| Metal f16 対 PyTorch MPS f16 | 最適化後 | 15% |

summary.md の fandhe-ai Metal f32・N=4096 実測 2961.1 GFLOP/s（= 2.9611 TFLOPS、
`--mode fresh` の転送込み値）を、`docs/perf/metal-floor-remeasurement.md`
（size=4096 行）の PyTorch MPS f32 参照値 12.0447 TFLOPS（5 回中央値）と比較すると:

```
2.9611 / 12.0447 ≒ 24.58%
```

現行下限 10% を明確に上回っており**下限抵触なし**。ただし本比較は分母（PyTorch MPS
f32）と分子（fandhe-ai `--mode fresh`）の計測プロトコル・実行日が異なる概算であり
（`metal-floor-remeasurement.md` の値は #577/#799 系列の専用計測手順、summary.md の
値は framework-compare ハーネスの別計測）、REQ-8 の正式な下限再確定手順
（`performance-floor-decision.md` §4「再確定条件・手順」）を経た値ではない。
**本イシューでは下限値自体の変更・再確定は行わない**（変更は人間承認タスク。
`.claude/rules/security.md`「自己修復ループ固有のガードレール」・
`.claude/rules/out-of-scope-tracking.md` に従い、必要であればユーザー承認を得て
別途 REQ-8 再確定タスクとして扱う）。

## 5. Mac 実機計測手順（申し送りテンプレート・未計測欄）

以下は既存バイナリの再利用のみで完結する（新規実装不要）。5 回独立実行・中央値・
Q1/Q3 を記録する（TASK-8.1 プロトコル・`.claude/rules/coding-rust.md`「テスト・
ベンチ」節）。

### 5.1 #930/#948 適用後の framework-compare 再計測（最優先）

`fandhe-ai = "=0.3.0"`（crates.io 公開版）を pin したままでは #930/#948 の効果が
現れない（§2.1）。再計測には以下いずれかが必要:

1. `fandhe-ai` の次期バージョン公開（#930/#948 を含む版）を待って pin を更新する、または
2. 一時的に `scripts/bench/framework-compare/bench-fandhe/Cargo.toml` の
   `fandhe-ai` 依存を path 依存へ切り替えて計測する（deps-policy.md の
   `=x.y.z` 完全固定・独立 workspace 限定の条件を一時的に外れるため、計測後は
   `=0.3.0` pin へ戻すか、この切り替え自体をユーザー承認事項として扱う）

再計測後、summary.md 「Metal」節と本 doc §2.1 の表を突き合わせ、N=256〜512 の
ギャップが 5 ms 固定費分縮小したか（`--未計測--`）を記録する。

### 5.2 `gemm_bench`（`dispatch_auto`・転送込み境界）改善前後比較

```sh
cargo run -p fandhe-ai-backend-metal --example gemm_bench --release
```

`gemm_bench.rs` は `--size` 等の CLI オプションを持たず、size=256/512/1024/2048/4096
（正方）をソース内の固定配列（`crates/backend-metal/examples/gemm_bench.rs` 内
`for size in [256usize, 512, 1024, 2048, 4096]`）で計測する。上記コマンドの実行
だけで当該サイズ集合が得られる。

改善前（v0.3.0 相当）／改善後（#930/#948 適用後）で 5 回計測・中央値比較: `--未計測--`

### 5.3 `gemm_f32_prepared_bench`（カーネル純境界。転送を含まない）

```sh
cargo run -p fandhe-ai-backend-metal --example gemm_f32_prepared_bench --release
```

`gemm_f32_prepared_bench.rs` も `--size` オプションは持たず、ソース内の固定配列
（`crates/backend-metal/examples/gemm_f32_prepared_bench.rs` 内
`for size in [512usize, 1024, 2048, 4096]`。256 は対象外）で size=512/1024/2048/4096
を計測する。

§2.2 の転送コスト寄与分を分離するため、prepared（デバイス常駐入力）境界と
`--mode fresh` の e2e 境界の差分を記録する: `--未計測--`

### 5.4 `scripts/bench/framework-compare/run_all.sh` 横並び再計測

§5.1 の pin 更新後に全体再計測し、summary.md を更新する: `--未計測--`

## 6. スコープ外（記録のみ・本 PR へ混入禁止）

`.claude/rules/out-of-scope-tracking.md` に従い、以下は Issue 起票をユーザー承認後に
行う事項として記録する（自動運転中は起票しない）。

- **framework-compare ハーネスの `fandhe-ai` pin 更新・再計測**（§5.1）: #930/#948
  適用後の実際の改善幅が現行 summary.md に未反映であるため、次回 crates.io 公開後の
  再計測、または path 依存への一時切り替えの是非をユーザーへ確認する
- **facade/autodiff 側のデバイス常駐テンソル化・アップロード削減**（候補 2・§3）:
  MLP 学習・推論の大差（metal 48.845 ms/step 対 candle 0.7518 ms）の主因候補でもあり、
  GEMM 単体の改善スコープを超える設計変更
- **カーネル本体の再設計**（候補 3・§3）と Mac 実機 A/B: 2 段パイプライン化・
  double-buffering 等、既存の採否判断（#544/#546/#549/#808/#810）と重複しない
  新規候補の検討
- **REQ-8 下限値の引き上げ再確定**（§4）: 人間承認タスク

## 7. 出典

- `scripts/bench/framework-compare/results/summary.md`（PR #915・計測日 2026-08-28）
- `docs/perf/metal-fixed-overhead-diagnosis.md`（#927）
- `crates/backend-metal/src/context_cache.rs`・`crates/backend-metal/src/ops.rs`
  冒頭コメント（#930・PR #948）
- `docs/facade-device-handle-design.md` §2.4（#930/#948 実装済みの追認記録。#931）
- `docs/perf/metal-gemm-bottleneck-diagnosis.md`（#487）
- `docs/perf/metal-floor-remeasurement.md`（size=4096 実測系列）
- `docs/perf/performance-floor-decision.md` §10（#577。現行確定下限）
- `docs/backend-metal-aligned-load-decision.md`（#808）・
  `docs/backend-metal-async-copy-decision.md`（#546）・
  `docs/backend-metal-mlx-classic-nax-decision.md`（#549）・
  `docs/backend-metal-morton-mapping-decision.md`（#544）・
  `docs/backend-metal-splitk-decision.md`（#810）・
  `docs/perf/metal-gemm-splitk-shapes.md`（#810）
- `docs/crates-io-publishing-order.md` §10（v0.3.0 公開日）
