# `gemm_simdgroup_tiled` 協調ロードレイアウト候補（イシュー #1298）

## 0. 目的・スコープ

`docs/perf/metal-gemm-n4096-kernel-gap.md` §5 が未実施のまま残していた E4
（協調ロード再構成）を実装する。`gemm_simdgroup_tiled`（f32・本番既定経路）
の staged 経路（`USE_TGP_STAGING=true`）が threadgroup 全スレッドで A/B
タイルを float4 単位で協調ロードする際の「スレッド → 要素割当（レイアウト）」
「行末パディング幅」を function constant で切替可能な候補として追加し、全
候補が現行方式と bit 同一であることを実機 `#[ignore]` テストで自己検証する。

**本イシューのスコープは機構の実装と bit 一致の自己検証のみ**。各候補の
純カーネル時間比較・`tile::select`／本番既定への結線判断は後続イシュー
#1300／#1302／#1304 のスコープであり、本ドキュメントでは行わない。

非公式 `simdgroup_async_copy` 系 AIR intrinsic は使わない
（`docs/backend-metal-async-copy-decision.md` の不採用判断を維持）。レーン
レベル Morton 順マッピングは標準 `simdgroup_matrix` API 下では適用不可
（`docs/backend-metal-morton-mapping-decision.md`）。

## 1. 候補表

| 軸 | function constant | 値 | 意味 |
|---|---|---|---|
| レイアウト | `COOP_LOAD_LAYOUT`（**index 14**・uint） | `0`（**既定**） | 現行: `idx = vi*4`。連続スレッドが同一行内の連続 float4 グループを担当（行優先・MLX steel `BlockLoader` 同型） |
| | | `1` | 行ストライド割当: `r = vi % rows; chunk = vi / rows; idx = r*row_len + chunk*4`。連続スレッドが同一列チャンク内の連続行を担当 |
| パディング | `TGP_PAD`（既存 index 6） | `0`／`4`（**既定**）／`8` | 行末パディング要素数（f32）。従来は `TileConfig::pad()` 導出の固定 2 値（0/4）のみだったが、本イシューで 3 択（`crate::tile::TgpPad`）へ一般化した |

候補ラベル（`L<layout>-P<pad>`）: `L0-P4`（**現行 = 本番既定**）・`L0-P0`・
`L0-P8`・`L1-P0`・`L1-P4`・`L1-P8` の 6 候補（必須実機検証対象。T1〜T5 が
使う head 集合）。

**バンク競合回避（XOR swizzle）軸は本 PR では実装しない**（§6「スコープ外」
参照。計画では `COOP_SMEM_SWIZZLE`〈index 15〉として任意実装するステップ
（S7）を用意していたが、時間制約により着手せず、index 15 は未割当のまま
残す）。

## 2. 設計

### 2.1 MSL（`crates/backend-metal/src/shaders/gemm.metal`）

- function constant を index 14（`COOP_LOAD_LAYOUT`・uint）として追加
  （`#else`〈本番既定〉側・`#ifdef GEMM_SPEC_ENABLED`〈イシュー #1288 E2
  試作経路〉側の両方、1:1 対応）。
- 協調ロードの「スレッド添字 `vi` → 非パディング平坦添字 `idx`」割当を
  `coop_load_flat_index(vi, rows, row_len)` ヘルパへ集約した:
  ```metal
  inline uint coop_load_flat_index(uint vi, uint rows, uint row_len) {
      if (COOP_LOAD_LAYOUT == 1) {
          uint r = vi % rows;
          uint chunk = vi / rows;
          return r * row_len + chunk * 4;
      }
      return vi * 4;
  }
  ```
  4 箇所の協調ロード分岐（A-NN・A-T・B-NN・B-T）の `uint idx = vi * 4;` を
  それぞれ `coop_load_flat_index(vi, rows, row_len)` へ置換した（`rows`/
  `row_len` は箇所ごとに異なる: A-NN は `(BM, BK)`・A-T は `(BK, BM)`・
  B-NN は `(BK, BN)`・B-T は `(BN, BK)`）。それ以外の行（`r`/`kk` 導出・
  `dst_idx`・`group_in_bounds`・スカラーフォールバック・float4 ロード）は
  一切変更していない。
- `TGP_PAD`（index 6）は従来 `cfg.pad()`（`TileConfig::pad()`。0/4 固定）を
  直接渡していたが、`crate::tile::CoopLoadConfig::pad_elems`（0/4/8）から
  渡すよう一般化した。`shaders/gemm.metal` 側の宣言テキスト自体は無変更
  （渡す値の出所のみ Rust 側で変更）。

### 2.2 bit 一致の論拠（#536/#538/#745/#1282/#1288/#1293 と同じ論法）

- **レイアウト軸**: 「どのスレッドがどの float4 グループを書くか」の割当
  （`vi` から `idx` への全単射）を変えるだけで、共有メモリ上の各要素の
  値・格納位置は不変。両レイアウトとも `idx` は同じ像集合
  `{0, 4, 8, ..., rows*row_len-4}` を過不足なく被覆する全単射であることを
  Rust 側モデル `coop_load_flat_index_model`（`tile.rs`）と単体テスト
  `coop_load_flat_index_model_is_bijection_for_all_candidates_and_patterns`
  （`CANDIDATES` 全候補 × A-NN/A-T/B-NN/B-T の `(rows, row_len)` 全組合せ）
  で固定した（Linux 実行可能。デバイス前の唯一の全単射証明）。
  `threadgroup_barrier` 後の `simdgroup_load`／MMA 発行順・オペランド列は
  一切変わらないため、数値はビット単位で不変。
- **パディング軸**: `lda`/`ldb` の変更は既に #538 で「`simdgroup_load` は
  パディング列を一切読まない」ことが確立済み。0／8 でも同様（`row_len` が
  4 の倍数のため float4 アラインメントも維持される。`TgpPad` の 3 値
  〈0/4/8〉はいずれも 4 の倍数）。

### 2.3 Rust（`crates/backend-metal/src/tile.rs`）

- `pub enum CoopLoadLayout { RowLinear, RowStrided }` + `as_u32()`。
- `pub enum TgpPad { Zero, Four, Eight }` + `elems()`（0/4/8）。値域を型で
  保証し、`TileConfigError` への variant 追加・実行時検証を不要にする
  （`FragLoadKSteps` と同じ設計判断）。
- `pub struct CoopLoadConfig { pub layout: CoopLoadLayout, pub pad: TgpPad }`
  + `pub const DEFAULT`（`RowLinear`・`Four`）・`pub(crate) fn pad_elems(&self,
  cfg: TileConfig) -> u32`（`!cfg.staged` なら常に `0`）。
- `pub(crate) const COOP_LOAD_CONFIG: CoopLoadConfig = CoopLoadConfig::DEFAULT`
  （本番既定。`MetalGemm::new` が渡す）。
- `TileConfig::shared_mem_bytes_for_pad(&self, pattern, pad_elems: u32) ->
  u32`: 既存 `shared_mem_bytes_for(pattern)` の本体をパディング幅引数化
  して移設。`shared_mem_bytes_for(pattern)` は `self.pad()` を渡す薄い
  ラッパーへ変更（戻り値は完全に非後退。単体テスト
  `shared_mem_bytes_for_pad_matches_shared_mem_bytes_for_at_default_pad` で
  固定）。

### 2.4 パイプライン構築（`pipeline.rs`／`spec_source.rs`／`gemm.rs`）

- `pipeline::GemmGateConstants` へ `tgp_pad_elems: u32`・
  `coop_load_layout: u32` を追加。`make_pipeline_with_constants` は
  index 6（`TGP_PAD`）を `cfg.pad()` ではなく `gates.tgp_pad_elems` から
  設定し、index 14（`COOP_LOAD_LAYOUT`）を `gates.coop_load_layout` から
  設定する（既存の 1 つの `unsafe` ブロック内へ 2 行追加。新規 `unsafe`
  ブロックは作らない）。
- `spec_source::SpecializationParams` へ同名 2 フィールドを追加し、
  `#define GEMM_SPEC_TGP_PAD`（従来の `cfg.pad()` 直書きから
  `params.tgp_pad_elems` へ変更）・`#define GEMM_SPEC_COOP_LOAD_LAYOUT`
  （新規）を出力する。既存 7 引数の `SpecializationParams::new` は変更
  せず（既定値 `cfg.pad()`／`0` を設定）、`pipeline.rs` 側は struct update
  構文で `GemmGateConstants` の実効値を上書きする。
- `MetalGemm` に `coop_load: tile::CoopLoadConfig` フィールドを追加し、
  `pub fn new_with_coop_load(ctx, coop_load)` を新設（`new_with_frag_load`
  と同型）。`pipeline_for_tile` は事前検証（`shared_mem_bytes_for_pad`）・
  ゲート導出・`encode_dispatch_tiled` への `tgp_pad_elems` 伝播のすべてで
  `self.coop_load.pad_elems(candidate)` を単一の出所として使う（確保量と
  カーネルが実際にアクセスする範囲を一致させる fail-closed 契約）。
- `pipeline_for_tile_f16` は `tgp_pad_elems: candidate.pad()`・
  `coop_load_layout: 0` を固定で渡す（no-op 契約。f16 カーネルは
  `COOP_LOAD_LAYOUT` を参照しないが `TGP_PAD` は共有メモリレイアウト
  導出に使うため、協調ロードパディング候補を経由しない従来値のまま）。

## 3. bit 一致自己検証結果

Linux で実行可能な範囲（型検査・Rust 側全単射モデル・パイプライン構築の
単体テスト・シェーダソース証跡）は本セッションで実行し、全て green を
確認した:

```
cargo test -p fandhe-ai-backend-metal            # 251 passed; 0 failed; 41 ignored
cargo test -p fandhe-ai-backend-metal --test shader_source_evidence  # 36 passed; 0 failed
cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin  # 成功（macOS 向けクロス型検査）
cargo clippy --workspace --all-targets --all-features -- -D warnings  # クリーン
cargo fmt --all -- --check                        # クリーン
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked                                    # クリーン
RUSTDOCFLAGS="-D warnings" cargo doc -p fandhe-ai-backend-metal -p fandhe-ai-backend-cpu --no-deps --locked --target aarch64-apple-darwin  # クリーン
```

新規追加した Linux 実行可能テスト（抜粋）:

- `tile::tests::coop_load_flat_index_model_is_bijection_for_all_candidates_and_patterns`
- `tile::tests::coop_load_config_default_is_current_path`
- `tile::tests::coop_load_pad_elems_is_zero_when_not_staged`
- `tile::tests::coop_load_default_pad_elems_matches_tile_config_pad`
- `tile::tests::shared_mem_bytes_for_pad_matches_shared_mem_bytes_for_at_default_pad`
- `tile::tests::shared_mem_bytes_for_pad_eight_is_at_least_pad_four_for_staged_candidates`
- `spec_source::tests::coop_load_axis_overrides_are_reflected_in_generated_defines`
- `shader_source_evidence::gemm_metal_source_declares_spec_ifdef_block_with_all_fifteen_defines`
- `shader_source_evidence::gemm_metal_source_else_branch_retains_all_fifteen_function_constants`
- `shader_source_evidence::gemm_simdgroup_tiled_source_uses_coop_load_flat_index_helper`
- `shader_source_evidence::gemm_simdgroup_tiled_f16_source_does_not_reference_coop_load_constants`

**実機（Metal・Apple Silicon）での bit 一致自己検証（T1〜T6。
`crates/backend-metal/src/gemm.rs::tests` に実装済みの `#[ignore]` テスト）
は本エージェントの実行環境に macOS 実機が無いため未実行のまま明記する**:

| テスト | 内容 |
|---|---|
| `coop_load_bit_match_all_candidates` | `tile::CANDIDATES` 全 9 候補 × N∈{512,1024,2048,4096} × 必須 5 head（`L0-P0`/`L0-P8`/`L1-P0`/`L1-P4`/`L1-P8`）で `dispatch_tiled_prepared` の bit 一致・フォールバック非経由を検証 |
| `coop_load_bit_match_dispatch_auto` | 本番自動選択経路 `dispatch_auto` で N=512〜4096 × 全 head |
| `coop_load_transposed_bit_match` | NT/TN/TT を N=1024・`CANDIDATES[3]`／`CANDIDATES[5]`（bk=32）× 全 head |
| `coop_load_bit_match_boundary_shape` | 端数形状（M=1032・N=1048・K=1032）× 全候補 × 全 head |
| `coop_load_f16_path_is_noop` | `gemm_simdgroup_tiled_f16` が `COOP_LOAD_LAYOUT` 非参照であることの実機証明（base vs L1-P8） |
| `coop_load_default_matches_production_constants` | `MetalGemm::new(...).coop_load() == tile::COOP_LOAD_CONFIG == CoopLoadConfig::DEFAULT` |

これら 6 テストの実機実行、および既存 parity／bit-match 群（swizzle・
fine-barrier・unroll-acc・source-specialized・frag-load・transposed 各種）
の非後退確認は、実機アクセスを持つ後続セッション（#1300 等）が
`docs/real-hardware-verification-env.md` の手順に従って実行し、本ドキュメント
または後続ドキュメントへ追記すること:

```sh
cargo test -p fandhe-ai-backend-metal --release --lib -- --ignored --nocapture coop_load
```

## 4. env_info

- 本 PR の実装・型検査・Linux 実行可能テストは Linux コンテナ環境
  （macOS 実機なし）で実施した。実機実測を含まないため CPU/GPU 型番・
  `uptime`／load average の記録は該当なし。
- 実機実行時は `docs/real-hardware-verification-env.md` の手順に従い、
  実行前後の `uptime` を記録すること。bit 一致検査自体は決定的な出力
  比較（`to_bits()` 厳密一致）のため GPU 負荷とは独立に成立する
  （純カーネル時間の A/B 計測とは異なり、共有負荷下でも判定結果は
  変わらない）。

## 5. #1300 への引き継ぎ

- `MetalGemm::new_with_coop_load(ctx, coop_load)` で base（`tile::
  COOP_LOAD_CONFIG`）/head（任意の `tile::CoopLoadConfig`）の 2
  インスタンスを構築できる（`new_with_frag_load` と同型）。
- head の作り方: `tile::CoopLoadConfig { layout: tile::CoopLoadLayout::
  RowStrided, pad: tile::TgpPad::Eight }` のように `layout`／`pad` を
  任意に組み合わせる（6 候補に限らず、`RowLinear`/`RowStrided` ×
  `Zero`/`Four`/`Eight` の全 6 通りが構築可能）。
- 純カーネル時間比較は GPU タイムスタンプ経路（`MetalContext::
  synchronize_with_gpu_timestamps`。イシュー #1276）・
  `gemm_reuse_phase_diag_tests::measure_one_phase_trial` を使う。
  `crates/backend-metal/src/gemm_frag_load_diag_tests.rs`（E3 の同型
  診断テストファイル）を雛形にすること（複数候補の `MetalGemm` を
  1 コンテキストで構築し、trial ごとに交互計測して中央値・base 比を
  出力する構成）。
- `tile::select`／`tile::CANDIDATES` への組み込み（本番結線）は行って
  いない。有効性が確認された場合の結線判断は #1302／#1304 のスコープ。

## 6. スコープ外（PR 本文へ記録。新規 Issue は起票しない）

- **バンク競合回避（XOR swizzle）軸（計画 S7）**: 時間制約により未実装。
  `COOP_LOAD_LAYOUT`（index 14）実装後の残り時間で着手する予定だった
  `COOP_SMEM_SWIZZLE`（index 15）・`tg_tile_offset` ヘルパ・20 箇所の
  添字置換は行っていない。index 15 は未割当のまま残す（`CoopLoadConfig`
  にも `smem_swizzle` フィールドを追加していない）。実装する場合は本
  ドキュメント §1〜§2 の設計と同じ論法（格納位置の純粋な置換であり
  `simdgroup_load(ptr, ld)` の契約を満たす限り bit 不変）が適用できる
  はずだが、実装・自己検証は行っていない。
- 各候補の純カーネル時間比較・有効性判断・`tile::select` 候補表への
  組み込み・本番結線（#1300／#1302／#1304）。
- `gemm_simdgroup_tiled_f16` の協調ロードレイアウト切替（本 issue は f32
  経路のみ。f16 は no-op 契約）。
- 実機（Metal・Apple Silicon）での T1〜T6 の実行・非後退確認（§3 参照。
  実機アクセスを持つ後続セッションへ引き継ぐ）。
