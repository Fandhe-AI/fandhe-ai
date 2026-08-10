# GEMM 経路選択 境界形状 実測記録（#69・TASK-11.2c）

イシュー #69「test(backend): TASK-11.2c 境界形状の実測再検証」の実測記録テンプレート。
受け入れ条件「境界値の実測記録と採用した閾値の根拠が残されている」に対応する。

対象は `crates/tensor-core/src/dispatch.rs::select_gemm_kernel`（TASK-11.2b・#68 で実装済みの
自作ディスパッチ規則）が参照する暫定閾値 3 つ（`METAL_SIMDGROUP_MIN_DIM`・`CUDA_WMMA_MIN_CC`・
CUDA の「形状下限なし」設計）。設計文書は `docs/dispatch-rules-design.md`（§2〜§4・#67）。

## 状態: Metal 境界形状 TFLOPS は #382 で実測完了。CUDA 3 節は #388 ツリー（#389/#390）担当で記入待ち

Metal（Apple Silicon 実機）側は #382 で実測を完了した（下記「実測結果」節・
「`METAL_SIMDGROUP_MIN_DIM` の妥当性判定（#382）」節）。CUDA（DGX Spark GB10 等・NVRTC 搭載）
側は本セッションに実機がなく、#388 ツリー配下の #389/#390 が引き続き担当する（推定値で
「CUDA: 小形状」「CUDA: 大形状」表を埋めない）。

過去セッション（Linux worktree、実機なし）で検証済みだった以下は今回も再確認済み:

- `cargo build --workspace --locked` — `cudarc` 動的ロード契約（CUDA toolkit 非搭載でもビルド成立）
  を崩していない
- `cargo fmt --all -- --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p backend-metal --release`（`#[ignore]` テストは通常実行から除外される）

## 計測手順

### Metal（Apple Silicon 実機）

```sh
git fetch origin
git checkout test/69-dispatch-boundary-measurement   # 本イシューの実装ブランチ
cargo test -p backend-metal --release --test dispatch_boundary -- --ignored --nocapture
```

出力形式（`crates/backend-metal/tests/dispatch_boundary.rs` 参照）:

- `dispatch_boundary_record dim=<N> path=tiled tflops=...` / `path=simdgroup_auto tflops=...
  simdgroup_auto_over_tiled=<比率>` 行: `min(M,N,K)` = 256/384/448/512/576/640/768/1024（正方）での
  2 経路比較
- `dispatch_boundary_route_record dim=<N> min_dim_threshold=512 expected_kernel=<KernelKind>
  route_verified=false result=parity_pass` 行: 各境界形状で `dispatch_backend_auto` の出力が
  CPU 参照実装との複合判定に通過したことの記録。`expected_kernel` は `select_gemm_kernel` を
  独立に呼んだ参考値であり、`route_verified=false` が示すとおり実機が実際にその経路を選んだ
  ことの検証ではない（`dispatch_backend_auto` はカーネル種別を返さないため比較不可。PR レビュー
  指摘）。経路選択ロジック自体は `tensor-core` 側の `#[cfg(test)]` が別途網羅する

### CUDA（DGX Spark GB10・compute capability 8.0 以上・NVRTC 搭載）

```sh
git fetch origin
git checkout test/69-dispatch-boundary-measurement
cargo test -p backend-cuda --release --test dispatch_boundary -- --ignored --nocapture
```

出力形式（`crates/backend-cuda/tests/dispatch_boundary.rs` 参照）:

- `dispatch_boundary_record dim=<N> path=tiled_f32|wmma_tf32_opt|tiled_f16|wmma_f16_opt
  tflops=... matrix_unit_over_tiled=<比率>` 行: 小形状 128/256/384/512 での MatrixUnit(WMMA) 対
  Tiled 比較（「形状下限なし」規則の実測根拠）
- `dispatch_boundary_record dim=<N> path=wmma_f16_opt|mma_sync_f16 tflops=... mma_over_wmma=<比率>`
  行: 大形状 2048/4096 での `mma.sync` パイプライン対基本 WMMA 比較（TMA 選好整理の実測根拠）

数値一致確認（受け入れ条件に必須の前提。性能値採用より先に実施すること）:

```sh
cargo test -p backend-metal --release -- --ignored --nocapture   # gemm_auto_parity.rs・dispatch_boundary.rs 双方
cargo test -p backend-cuda --release -- --ignored --nocapture    # gemm_auto.rs・cpu_cuda_*_parity.rs 双方
```

`tests/gemm_auto_parity.rs`（Metal）・`tests/gemm_auto.rs`／`tests/cpu_cuda_wmma_parity.rs`（CUDA）の
既存 `#[ignore]` ケースが全て PASS することを先に確認する（`dispatch_boundary.rs` 自体も
`assert_parity` で複合判定を検証するが、経路選択の網羅的な数値一致検証は既存ファイルが担当する）。

## 実測結果

### 計測環境

| 項目 | 値 |
|------|-----|
| Metal GPU | Apple M4 Max（GPU 40 コア。`machdep.cpu.brand_string` / `system_profiler SPDisplaysDataType`） |
| Metal OS | macOS 26.6（BuildVersion 25G72） |
| CUDA GPU | 未実施（#389/#390 担当。DGX Spark GB10 実機は本セッションに無し） |
| CUDA driver / NVRTC バージョン | 未実施（#389/#390 担当） |
| rustc | `rustc 1.96.0 (ac68faa20 2026-05-25)` |
| 計測コミット SHA | `2264653`（`origin/main` 分岐元。`git rev-parse HEAD` は作業ブランチ先頭コミット） |
| 計測プロトコル | `bench-harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1）を **ラン間 5 回反復**し、各パス（`tiled`／`simdgroup_auto`）の TFLOPS をラン別中央値からさらに中央値化した値を採用（`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値」を「プロトコル内 20/20 中央値」×「ラン間 5 点の中央値」の二重適用として解釈。ステップ 3〜4 参照） |
| 計測範囲 | 転送（`clone_htod`×2 相当のバッファアップロード）＋カーネル実行の合算計測。転送時間のみの補正は行わない（`tests/dispatch_boundary.rs` 冒頭コメント「転送時間補正なし（既知の限定事項）」）。`BOUNDARY_DIMS` は全て 8 の倍数のため `dispatch_variant` の `pad8`（`SimdgroupTiled` のみ適用）は no-op であり、両経路は同一 dtype（f32）・同一バイト数の転送となる |
| 計測衛生 | AC 電源接続・`pgrep -fl cargo` で他エージェントの `cargo`/`gemm_bench`/`dispatch_boundary`/`xcodebuild` プロセスが無いことを各ラン開始前に確認（無関係な `npm exec @playwright/mcp` プロセスのみ検出。競合なし）。外部ディスプレイ（DELL P3225QE・6400×3600、UI 3200×1800@60Hz）を接続した状態で計測（コンポジタ負荷は #381 と同一構成のまま） |
| 統計の扱い | 表の「ラン間偏差」列は 5 本の `simdgroup_auto_over_tiled` 比のうち最大値と最小値の差を中央値で正規化した割合（%）。「ratio」列はパス別 TFLOPS 中央値同士の比（ラン別比率の中央値ではない。ステップ 4 参照） |

### Metal: 境界形状（tiled 対 simdgroup_matrix 動的タイル選択）

`select_gemm_kernel` 期待経路は `dispatch_boundary_route_record` 行から独立に計算した参考値
（`route_verified=false`）であり、実機が実際にその経路を選んだことの検証結果ではない
（`MetalGemm::dispatch_backend_auto` はカーネル種別を返さないため比較不可。上記「計測手順」節
参照）。「parity」列は `dispatch_backend_auto` の出力と CPU 参照実装との数値一致のみを表す。

下表の TFLOPS は転送時間補正なし（合算計測。`crates/backend-metal/tests/dispatch_boundary.rs`
冒頭コメント「転送時間補正なし（既知の限定事項）」参照）。`tiled`／`simdgroup_auto` は同一
dtype・同一バイト数の転送のため比が壊れるほどの歪みはないが、転送コストが一定量
`auto/tiled` 比を 1.0 側へ寄せるバイアスとして残るため、クロスオーバー特定（実測は 384）は
このバイアスを踏まえて解釈すること（実際のクロスオーバーはこのバイアスにより実測値よりやや
小さい形状にある可能性がある。過小評価側にバイアスされる）。

「ラン間偏差」は 5 ラン分の `simdgroup_auto_over_tiled` 比（テスト自身の出力。パス別 TFLOPS の
ラン内比）の min/max とその開き（中央値比の何 % か）。「ratio」列（見出し比率）はパス別 TFLOPS
中央値同士の比 `auto_median / tiled_median` であり、ラン別比率の中央値ではない（ラン別に取ると
偏差の大きい形状で結果が変わりうるため、`docs/perf/dispatch-boundary-measurement.md` では固定した
算出方法を採る。ステップ 4）。

| min(M,N,K) | tiled TFLOPS（5 ラン中央値） | simdgroup_auto TFLOPS（5 ラン中央値） | auto/tiled | ラン間偏差（min–max, %） | `select_gemm_kernel` 期待経路（参考・未検証） | parity |
|------|------|------|------|------|------|------|
| 256  | 0.103 | 0.095 | 0.922 | 0.915–1.501（63.9%） | Tiled | PASS |
| 384  | 0.189 | 0.226 | 1.196 | 1.183–1.220（3.1%） | Tiled | PASS |
| 448  | 0.220 | 0.311 | 1.414 | 1.390–1.435（3.2%） | Tiled | PASS |
| 512  | 0.250 | 0.275 | 1.100 | 1.068–1.105（3.4%） | MatrixUnit | PASS |
| 576  | 0.267 | 0.347 | 1.300 | 1.285–1.312（2.1%） | MatrixUnit | PASS |
| 640  | 0.287 | 0.425 | 1.481 | 1.462–1.482（1.4%） | MatrixUnit | PASS |
| 768  | 0.370 | 0.783 | 2.116 | 1.367–2.236（50.6%） | MatrixUnit | PASS |
| 1024 | 0.548 | 0.969 | 1.768 | 1.238–1.780（30.6%） | MatrixUnit | PASS |

`parity` 列は「数値一致ゲート」（`cargo test -p backend-metal --release --test gemm_auto_parity -- --ignored`・
`--test dispatch_boundary -- --ignored ... dispatch_backend_auto_matches_reference_at_boundary_shapes`）が
全 PASS したことに基づく（実測日時点で両ゲートとも 8 形状 × 該当ケース全て green）。

**dim=256 の符号一貫性に注意**: `simdgroup_auto_over_tiled` は 5 ラン中 4 ラン（0.915〜0.945）で
1.0 未満だが、1 ラン（run5）のみ 1.501 と 1.0 を大きく上回った（straddle）。384 以上は 5 ラン
すべてで符号が一貫（min/max とも同じく 1.0 の同じ側）であり、クロスオーバー自体は 384 で確定
できる（256 はクロスオーバーより小さい非決定 dim。下記「`METAL_SIMDGROUP_MIN_DIM` の妥当性判定」
節）。dim=256 の外れ値 1 点は Appendix の生データから `tiled` 側（分母）の計測揺れに起因すると
判断できるが、この揺れ自体は要調査事項として明記する。

`crates/backend-cuda/tests/dispatch_boundary.rs` の各計測は計算のみ時間の正値ガード
（`assert!(... > 0.0)`。`tests/tensor_core_real_device.rs:235-239` と同じ実装）を持つ。
実機実行で dim=128 付近においてこのガードが失敗した場合はテストのバグではなく、
「その形状では計算時間が転送時間（`clone_htod` × 2 + `alloc_zeros` + `synchronize` +
`clone_dtoh`）と同程度以下」というプロトコル上の観測結果である。その場合は該当形状の
TFLOPS 値は「計測不能」として記録し、より大きな warmup/計測回数か、より大きな最小形状
から実測を再開することを検討する（閾値・許容誤差の変更はしない）。

## `METAL_SIMDGROUP_MIN_DIM` の妥当性判定（#382）

本節は #382 の受入基準 B（据え置き／変更提案の実測根拠付き記録）に対応する。**コード
（`crates/tensor-core/src/dispatch.rs`）・Issue 起票は本節の記録に基づいて行わない**（変更の
要否判断はここに記録するが、実施は別レビュー・別 PR・ユーザー承認を経る。実装計画のスコープ
境界 1〜3 節）。

### 比較軸の明示

本節の比較軸は `GemmVariant::Tiled` 対 `MetalGemm::dispatch_auto`（`simdgroup_matrix` 動的
タイル選択経路）であり、`docs/perf/metal-gemm-dynamic-tile.md`（#381）の `auto/simdgroup`
（simdgroup 単独カーネル対 `dispatch_auto`）とは**比較対象が異なる**。両ドキュメントの比率を
同一指標として混同しないこと。

`select_gemm_kernel` の期待経路（上表の当該列）は `tensor-core::dispatch::select_gemm_kernel`
を独立に呼んで求めた参考値であり、`MetalGemm::dispatch_backend_auto` が実機上で実際にその経路を
選択したことの検証結果ではない（`route_verified=false`。上表脚注・`tests/dispatch_boundary.rs`
冒頭コメント「位置づけ」参照）。

### 判定規則と適用結果

| 観測 | 記録する結論 |
|------|-------------|
| クロスオーバー（`ratio` が 1.0 を跨ぐ最小 dim）が 448〜576 の範囲にあり、かつ決定に関わる dim で符号一貫性が 5/5 | 据え置き（512 妥当）。「暫定」表記の確定は承認事項として保留 |
| クロスオーバーが 384 以下、または 768 以上に明確に外れ、かつ決定に関わる全 dim で符号一貫性が 5/5 | 変更提案を記録（提案値 = ラン 5 本すべてで `ratio > 1.0` となる最小の実測 dim） |
| いずれかの決定 dim で符号が straddle する（ラン間で 1.0 の上下に分かれる）／単調でない | **判定保留（据え置き）**。ラン間偏差を根拠として明記 |
| `ratio` が全 dim で 1.0 未満（simdgroup 経路が全域で不利） | 据え置き＋要調査として記録 |

**適用結果: 変更提案を記録（提案値 = `METAL_SIMDGROUP_MIN_DIM` を 384 相当へ引き下げ）**。
コードは変更しない（実装計画のスコープ境界 1）。

クロスオーバー（`ratio` が 1.0 を跨ぐ最小 dim）は実測データ上 **384**（5 ラン全て 1.183〜1.220
で 1.0 を上回り符号一貫。ラン間偏差 3.1%）である。448 以上（448・512・576・640・768・1024）も
全て 5/5 で 1.0 超であり、384 が「1.0 を上回る」実測レンジの下端になっている。この結果は判定
規則の 2 行目「クロスオーバーが 384 以下…に明確に外れ、かつ決定に関わる全 dim で符号一貫性が
5/5」に該当する（クロスオーバーを画定する決定 dim は 384 であり、384 は符号一貫性 5/5 を満た
す）。したがって **512 は保守的すぎる可能性があり、384 への引き下げを変更提案として記録する**
（提案値の定義「ラン 5 本すべてで `ratio > 1.0` となる最小の実測 dim」に厳密に一致）。

**dim=256 の扱い（決定 dim ではない）**: 256 はクロスオーバー（384）より小さい形状であり、
上記 2 行の判定基準はいずれも「クロスオーバー dim・その上方」の符号一貫性のみを要求するため、
256 の挙動は本判定の分類（384 以下 vs 448〜576 vs 768 以上）を左右しない。256 は 5 ラン中 4 ラン
（0.915〜0.945）で 1.0 未満だが 1 ラン（run5）のみ 1.501 と 1.0 を上回った（ラン間偏差 63.9%）。
Appendix の生データによると、この揺れは `simdgroup_auto` 側（0.094〜0.096 で 5 ラン安定）ではなく
**分母の `tiled` 側**（run5 のみ 0.063、他 4 ラン 0.101〜0.104）に起因する外れ値であり、
「simdgroup 経路が優位に転じた」ことを示すものではない。真のクロスオーバー位置が 256〜384 の
間（実測解像度の制約で 257〜383 は未計測）にあるか、256 未満にあるかは本実測では確定できない
ため、これを**要調査事項として記録するに留め**、変更提案の値は実測で確認できた最小の dim
（384）に留める（256 未満への外挿はしない）。

### 補足事項

- dim=768・1024 のラン間偏差（50.6%／30.6%）は他の中間形状（384〜640 は 1.4〜3.4%）と比べて
  大きいが、5 ラン全てで符号（`ratio > 1.0`）は一貫しており、上記分類（768 以上も 1.0 超）に
  影響しない。外部ディスプレイのコンポジタ負荷・GPU サーマルスロットリング等の環境要因が
  寄与している可能性があるが、本節は分類結果への影響がないため深掘りしない。
- 数値一致（parity）は全 8 形状で PASS しており、性能値の採用条件（ステップ 2 のゲート）を満たす。

### 「採用閾値の根拠表」との整合

下記「採用閾値の根拠表」Metal 行の「実測後の判定基準」列にある「クロスオーバー形状が 512 から
大きく外れる場合は閾値をクロスオーバー形状へ更新する後続 PR を起票する」という記述は、#382 の
受入基準（変更提案は**記録**に留め、実施は別レビュー・別 PR・ユーザー承認とする）および
`.claude/rules/out-of-scope-tracking.md`（Issue 起票はユーザー承認必須）と整合しないため、本節・
該当表セルの追記により「#382 では変更提案の記録に留め、コード変更・Issue 起票は行わない」旨へ
読み替える。

### CUDA: 小形状（tiled 対 WMMA・形状下限なし規則の検証）

| dim | tiled f32 TFLOPS | WMMA TF32(opt) TFLOPS | tf32 matrix_unit/tiled | tiled f16 TFLOPS | WMMA f16(opt) TFLOPS | f16 matrix_unit/tiled |
|------|------|------|------|------|------|------|
| 128  | | | | | | |
| 256  | | | | | | |
| 384  | | | | | | |
| 512  | | | | | | |

### CUDA: 大形状（WMMA 対 mma.sync パイプライン・TMA 選好整理の検証）

| dim | WMMA f16(opt) TFLOPS | mma.sync f16 TFLOPS | mma/wmma |
|------|------|------|------|
| 2048 | | | |
| 4096 | | | |

## 採用閾値の根拠表

| 閾値・設計 | 現行値 | 根拠（v1 参考値の出典＋v2 保守設計の理由） | 実測後の判定基準 |
|---|---|---|---|
| `METAL_SIMDGROUP_MIN_DIM`（`crates/tensor-core/src/dispatch.rs`） | `512` | v1 PoC-8 実測は 256/512 の 2 点のみ（`docs/dispatch-rules-design.md` §3.1「同上 `:75`」）。境界形状（384・640 等）は未計測のため 512 を暫定的に踏襲し、閾値未満は保守的に tiled へ倒す設計（§3.2） | **#382 で実測完了・変更提案あり（提案値 384）**。実測クロスオーバーは 384（5 ラン全て `auto/tiled > 1.0`。448 以上も同様）であり、判定規則「クロスオーバーが 384 以下…かつ決定 dim で符号一貫性 5/5」に該当する。dim=256（クロスオーバー未満・非決定 dim）はラン間で符号が割れた（1 ラン外れ値）が分類には影響しない（詳細は上記「`METAL_SIMDGROUP_MIN_DIM` の妥当性判定（#382）」節）。**#382 では変更提案の記録に留め、コード変更・Issue 起票は行わない**（実施は別レビュー・別 PR・ユーザー承認） |
| `CUDA_WMMA_MIN_CC`（`(7, 0)`） | `(7, 0)` | WMMA は Volta（cc 7.0）以降という一般的な NVIDIA アーキテクチャ世代対応（`docs/dispatch-rules-design.md` §2 表）。cc 世代境界自体の実機再確認は本イシューのスコープだが、cc 7.x の実機（Volta/Turing）が本セッション・後続実機いずれにも存在しないため世代境界の実測は据え置く | 実機実測の対象外（compute capability 境界の実測には該当世代の実機が必要）。RTX 3060（cc 8.6）・GB10（cc 12.1）いずれも `cc >= 7.0` を満たすため、本イシューの実測は「ゲートを満たした場合に MatrixUnit が有利であること」の検証に限定し、`(7, 0)` 自体は変更しない |
| CUDA の「形状下限なし」設計（§3.2） | 形状閾値なし（HW ゲートのみ） | GB10 実測で最小形状 256 でも accelerated が unit の約 1.4〜1.6 倍優位（`docs/dispatch-rules-design.md` §3.1 `:126`・`:140`）という v1 CubeCL 前提の参考値 | 上表「CUDA: 小形状」の `matrix_unit/tiled` が 128/256/384/512 のいずれでも 1.0 を上回れば設計を維持する。1.0 を下回る形状が観測された場合（小形状で tiled が有利に逆転）は Metal と同様の形状閾値導入を検討し、別レビュー・別 PR で提案する（本イシューでは導入しない） |
| CUDA「TMA 選好はディスパッチ条件でなくカーネル内部チューニング」の整理（§3.2「TMA の扱い」） | `select_gemm_kernel` は `mma.sync` パイプラインと基本 WMMA を区別しない（`CudaGemmAuto::run_f16` は `CudaWmmaGemm` のみを呼ぶ） | v1 実測（`poc-8-matrix-unit/README.md:125`）は M=N=K=2048/4096 で TMA 系候補が最速だが、これは「同じ Tensor Core 経路内でのカーネル変種選択」であり「Tensor Core 経路を使うか否か」の分岐条件ではないという v2 設計判断 | 上表「CUDA: 大形状」で `mma_over_wmma` が 1.0 を上回れば整理を維持する（パイプライン差は経路選択の分岐にしない設計のまま、カーネル内部の既定実装をどちらにするかの判断材料として別途記録する）。1.0 を大きく下回る場合（mma パイプラインが未成熟で基本 WMMA より遅い）は `CudaGemmAuto::run_f16` の既定カーネル選択を見直す別 Issue を起票する |

閾値変更が必要と判断した場合は、本表の「実測後の判定基準」に従い `crates/tensor-core/src/dispatch.rs`
の定数・ドキュメンテーションコメントを更新する別レビュー・別 PR で対応する（ガードレール閾値・テスト
許容誤差の変更ではないため本イシュー実装フローの`.claude/rules/security.md` 対象外だが、
`.claude/rules/delegation-impl.md`「実装 Agent にガードレール閾値・テスト許容誤差を緩和させない」との
混同を避けるため、本イシュー内では変更しない）。

## Appendix: Metal 境界形状 5 ラン生データ（#382）

`cargo test -p backend-metal --release --test dispatch_boundary -- --ignored --nocapture
boundary_shapes_tflops_record` を 5 回反復した各ランの `dispatch_boundary_record` 出力から
`path=tiled`／`path=simdgroup_auto` の TFLOPS と `simdgroup_auto_over_tiled` を抜粋する
（`BenchReport::to_json` の完全な `report=` 行〈warmup/計測サンプル全数〉はサイズが大きいため
scratchpad（`382-run{1..5}.log`。本 worktree 外・非コミット）に一次証跡として保存し、本表は
そこから再計算可能な形で転記した値）。

| ラン | dim | tiled TFLOPS | simdgroup_auto TFLOPS | simdgroup_auto_over_tiled |
|------|-----|------|------|------|
| 1 | 256 | 0.101 | 0.096 | 0.945 |
| 1 | 384 | 0.192 | 0.228 | 1.186 |
| 1 | 448 | 0.218 | 0.313 | 1.435 |
| 1 | 512 | 0.248 | 0.265 | 1.068 |
| 1 | 576 | 0.263 | 0.339 | 1.287 |
| 1 | 640 | 0.289 | 0.425 | 1.470 |
| 1 | 768 | 0.350 | 0.783 | 2.236 |
| 1 | 1024 | 0.548 | 0.969 | 1.769 |
| 2 | 256 | 0.104 | 0.095 | 0.915 |
| 2 | 384 | 0.189 | 0.231 | 1.220 |
| 2 | 448 | 0.221 | 0.311 | 1.407 |
| 2 | 512 | 0.248 | 0.274 | 1.105 |
| 2 | 576 | 0.264 | 0.347 | 1.312 |
| 2 | 640 | 0.285 | 0.423 | 1.481 |
| 2 | 768 | 0.619 | 0.900 | 1.455 |
| 2 | 1024 | 0.651 | 0.988 | 1.517 |
| 3 | 256 | 0.104 | 0.095 | 0.917 |
| 3 | 384 | 0.187 | 0.225 | 1.200 |
| 3 | 448 | 0.216 | 0.304 | 1.404 |
| 3 | 512 | 0.250 | 0.276 | 1.103 |
| 3 | 576 | 0.267 | 0.350 | 1.310 |
| 3 | 640 | 0.287 | 0.425 | 1.482 |
| 3 | 768 | 0.354 | 0.734 | 2.072 |
| 3 | 1024 | 0.652 | 1.157 | 1.774 |
| 4 | 256 | 0.103 | 0.094 | 0.916 |
| 4 | 384 | 0.189 | 0.224 | 1.183 |
| 4 | 448 | 0.220 | 0.311 | 1.414 |
| 4 | 512 | 0.251 | 0.277 | 1.104 |
| 4 | 576 | 0.267 | 0.347 | 1.300 |
| 4 | 640 | 0.291 | 0.425 | 1.462 |
| 4 | 768 | 0.370 | 0.636 | 1.719 |
| 4 | 1024 | 0.544 | 0.968 | 1.780 |
| 5 | 256 | 0.063 | 0.094 | 1.501 |
| 5 | 384 | 0.190 | 0.226 | 1.188 |
| 5 | 448 | 0.226 | 0.314 | 1.390 |
| 5 | 512 | 0.253 | 0.275 | 1.089 |
| 5 | 576 | 0.267 | 0.344 | 1.285 |
| 5 | 640 | 0.285 | 0.423 | 1.482 |
| 5 | 768 | 0.630 | 0.862 | 1.367 |
| 5 | 1024 | 0.532 | 0.658 | 1.238 |

run5 の dim=256 tiled=0.063（他ラン 0.101〜0.104）は他ランと比べ明確な外れ値であり、これが
`simdgroup_auto_over_tiled=1.501`（dim=256 のラン間符号不一致の原因）を作っている（分子の
`simdgroup_auto` は 5 ラン 0.094〜0.096 で安定しており、分母側の外れ値が比率を押し上げた）。
run2/3/5 の dim=768・1024 で tiled TFLOPS が run1/4 の約 1.7〜1.9 倍に跳ねている点も、ラン間
偏差 50.6%／30.6%（上表）の実体である。いずれも 384 以上の符号一貫性（クロスオーバー判定の
根拠）には影響していない。dim=256 は判定の分類（クロスオーバー ≤384）を左右しない非決定 dim
であり、上記「`METAL_SIMDGROUP_MIN_DIM` の妥当性判定（#382）」節の要調査事項として記録するに
留める。

## 未実施・後続作業

- CUDA（DGX Spark GB10・NVRTC 搭載）の「実測結果」3 節（計測環境の CUDA 行・小形状表・大形状表）
  は #388 ツリー配下の #389/#390 が `cargo test -p backend-cuda --release --test dispatch_boundary
  -- --ignored --nocapture` 実行後に埋める（本イシュー #382 では推定値を記入しない）
- `METAL_SIMDGROUP_MIN_DIM` の妥当性判定（#382）節で要調査事項として記録した「384 未満（256 含む）
  でのクロスオーバー位置の再確認」は、閾値変更・Issue 起票を伴わない範囲では #382 のスコープ外
  （承認前提。上記「妥当性判定」節参照）
- 実測に基づく閾値変更（`METAL_SIMDGROUP_MIN_DIM` の更新・CUDA 形状閾値の導入検討）は上記「採用閾値の
  根拠表」の判定基準に従い、別レビュー・別 PR・ユーザー承認を経て行う（本イシューのスコープ外）
- 証跡整備（カーネルソース内命令＋ベンチログの体系化）は #70（TASK-11.3）が担当する
