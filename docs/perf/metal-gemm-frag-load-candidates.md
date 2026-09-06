# `gemm_simdgroup_tiled` フラグメントロード方式候補（イシュー #1293）

## 0. 目的・スコープ

`docs/perf/metal-gemm-n4096-kernel-gap.md` §5 が未実施のまま残していた E3
（フラグメントロード方式変更）を実装する。`gemm_simdgroup_tiled`（f32・本番
既定経路）の staged／direct-load 両経路に「ロード元（threadgroup 経由／
device 直接）」「ロード粒度（K 方向の一括ロード段数）」を function constant
で切替可能な候補として追加し、全候補が現行方式と bit 同一であることを実機
`#[ignore]` テストで自己検証する。

**本イシューのスコープは機構の実装と bit 一致の自己検証のみ**。各候補の
性能実測・`tile::select`／本番既定への結線判断は兄弟イシュー #1295 のスコー
プであり、本ドキュメントでは行わない。

非公式 `simdgroup_async_copy` 系 AIR intrinsic は使わない
（`docs/backend-metal-async-copy-decision.md` の不採用判断を維持）。

## 1. 候補表

`USE_TGP_STAGING`（既存 function constant・index 5。`TileConfig.staged` から
畳み込まれる）が既に「threadgroup 経由／device 直接」の切替軸を担っている
ため、本イシューでは新しい mode 定数を増やさず、各経路の内部構造（K 方向の
一括ロード幅）を切り替える 2 定数のみを追加した。

| 候補ラベル | `USE_TGP_STAGING` | `FRAG_LOAD_DEVICE_HOISTED`（index 12） | `FRAG_LOAD_KSTEPS`（index 13） | 実体 |
|---|---|---|---|---|
| `tgp-k1`（**現行 = 本番既定**） | true | 無視（no-op） | 1 | 既存 staged 経路（バイト同一） |
| `tgp-k2` | true | 無視（no-op） | 2 | staged 経路で 1 kk 反復に 16 幅（8 幅 × 2）のフラグメントを一括ロードして MMA |
| `device-legacy`（**現行**） | false | false | 無視（no-op） | 既存 direct-load 経路（バイト同一。#536 蛇行走査） |
| `device-hoisted-k1` | false | true | 1 | device メモリから `a_frag[acc_rows]`/`b_frag[acc_cols]` を kk 先頭で一括ロード（#745 型 hoisting の device 版） |
| `device-hoisted-k2` | false | true | 2 | 同上 + 16 幅 |

## 2. 設計

### 2.1 MSL（`crates/backend-metal/src/shaders/gemm.metal`）

- function constant を index 12（`FRAG_LOAD_DEVICE_HOISTED`・bool）・13
  （`FRAG_LOAD_KSTEPS`・uint）として追加（`#else`〈本番既定〉側・`#ifdef
  GEMM_SPEC_ENABLED`〈イシュー #1288 E2 試作経路〉側の両方、1:1 対応）。
- **staged 経路**: 既存の 8 幅単一フラグメント kk ループを
  `if (FRAG_LOAD_KSTEPS == 2) { <K=2 段ブロック> } else { <既存ループ。
  テキスト無変更> }` で包んだ。K=2 段ブロックは `a_frag2[2][ACC_ROWS_CAP]`/
  `b_frag2[2][ACC_COLS_CAP]` を使い、16 幅刻みの主ループ（ks 0→1 昇順で
  ロード・MMA 発行）+ 8 幅刻みの端数ループ（BK が 16 の倍数でない残り）
  という構成。
- **direct-load 経路**: 既存の legacy ブロック（`bk_full8`・8 幅刻み・蛇行
  走査）を `if (FRAG_LOAD_DEVICE_HOISTED) { <hoisted ブロック> } else {
  <既存ブロック。テキスト無変更> }` で包んだ。hoisted ブロックは
  `kstep = (FRAG_LOAD_KSTEPS == 2) ? 16 : 8` で kk を進め、1 反復あたりの
  実段数 `ks_count = min(kstep, bk_full8 - kk) / 8`（1 または 2）を動的に
  決めることで、K=2 かつ端数が生じる場合も同一ループ内で ks_count=1 として
  処理する（別の端数ループを持たない設計。境界チェック文言
  `if (a_row < dims.m) {`/`if (b_col < dims.n) {` の重複を避けるため）。
- **REQ-8 境界チェック**: hoisted ブロックの `a_row < dims.m`/`b_col <
  dims.n` 判定式は legacy 経路と綴りまで完全に同一。
  `tests/shader_source_evidence.rs`
  `gemm_simdgroup_tiled_source_retains_req8_boundary_guards_in_both_unroll_variants`
  がこの判定式の出現数（`if (a_row < dims.m) {`/`if (b_col < dims.n) {`）
  を legacy 経路の unroll/非 unroll 各 1 回 + hoisted ブロック 1 回の
  計 3 回として固定する（#1282 導入時点の 2 回から増加）。
- **bit 一致の論拠**（#536/#538/#745/#809/#1282/#1288 と同じ論法）:
  `acc[r][c_]` ごとの K 方向累算オペランド列（値・kk 昇順）はロード元・
  一括ロード数・ループ構造を変えても不変なため、
  `simdgroup_multiply_accumulate` の結果はビット単位で一致する。0 埋め
  （境界外要素）の契約も staged/direct 双方で従来と同一のまま変えない。
- `gemm_simdgroup_tiled_f16` はいずれの定数も参照しない（no-op 契約。
  `pipeline_for_tile_f16` は常に既定値を渡す）。

### 2.2 Rust（`tile.rs`・`pipeline.rs`・`spec_source.rs`・`gemm.rs`）

- `tile::FragLoadKSteps`（`One`/`Two`。値域を型で保証し実行時検証・新規
  `TileConfigError`/`MetalError` variant を不要にする）・
  `tile::FragLoadConfig { device_hoisted, ksteps }`・`tile::FRAG_LOAD_CONFIG`
  （本番既定 `DEFAULT`）を追加。`TileConfig` 自体へフィールドは追加しない
  （`SWIZZLE_ENABLED`/`FINE_BARRIER_ENABLED`/`UNROLL_ACC_ENABLED`/
  `SOURCE_SPECIALIZATION_ENABLED` と同じ instance ゲート方式）。
- **可視性（`pub`）**: `FragLoadKSteps`/`FragLoadConfig` は他の内部
  `pub(crate)` 定数群と異なり `pub` にした。`MetalGemm::new_with_frag_load`
  （`Two` を実際に構築する唯一の入口）を他の `new_with_*` 入口
  （`new_with_swizzle`／`new_with_fine_barrier`／`new_with_unroll_acc`／
  `new_with_source_specialization`）と同型の `pub fn` にする以上、引数型も
  同じ可視性が必要。`pub(crate)` のまま維持しようとすると、`Two` バリアント
  が crate 内のどこからも構築されなくなり（bool ゲートと異なり enum の
  未構築バリアントは dead_code 検査の対象になる）、他クレートからの通常の
  依存ビルド（`cargo clippy -p fandhe-ai-backend-cuda --all-targets` 等が
  `fandhe-ai-backend-metal` を lib として通常ビルドする経路。`cfg(test)` が
  付かない）で dead_code エラーになる（`.claude/rules/coding-rust.md`
  「`#[allow]` の安易な追加で黙らせない」方針により `#[allow(dead_code)]`
  は使わない。`tile::verify_m4_max` doc comment の前例と同じ判断軸）。
- `pipeline::GemmGateConstants` に `frag_load_device_hoisted`/
  `frag_load_ksteps` を追加し、`make_pipeline_with_constants`（既存の 1 つの
  `unsafe` ブロック内。新規 `unsafe` ブロックは追加していない）で index 12/13
  を設定。`spec_source::SpecializationParams` にも同 2 フィールドを追加し
  `#define GEMM_SPEC_FRAG_LOAD_DEVICE_HOISTED`/`GEMM_SPEC_FRAG_LOAD_KSTEPS`
  を出力（E2 特殊化経路との整合）。
- `MetalGemm` に `frag_load: tile::FragLoadConfig` フィールド・
  `new_with_frag_load(ctx, frag_load)` 入口を追加。`pipeline_for_tile` が
  `self.frag_load` から `GemmGateConstants` へ展開する。`pipeline_for_tile_f16`
  は常に既定値（no-op）を渡す。

## 3. bit 一致自己検証（実機実測）

M4 Max 実機（`cargo test -p fandhe-ai-backend-metal --release --lib --
--ignored --nocapture frag_load`）で以下 4 テストが全 pass:

```
running 4 tests
test gemm::tests::frag_load_transposed_bit_match ... ok
test gemm::tests::frag_load_on_off_bit_match_dispatch_auto ... ok
test gemm::tests::frag_load_tgp_vs_device_same_shape_bit_match ... ok
test gemm::tests::frag_load_on_off_bit_match_all_candidates ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 274 filtered out; finished in 11.42s
```

- **T1**（`frag_load_on_off_bit_match_all_candidates`）: `tile::CANDIDATES`
  全 9 候補 × N=512/1024/2048/4096 で base（`tile::FRAG_LOAD_CONFIG`）と
  head ∈ {tgp-k2, device-hoisted-k1, device-hoisted-k2} の
  `dispatch_tiled_prepared` 出力が bit 同一。
- **T2**（`frag_load_tgp_vs_device_same_shape_bit_match`）: staged 候補と
  `TileConfig { staged: false, ..cfg }` twin（device-legacy／
  device-hoisted-k1／device-hoisted-k2）の同一タイル形状（N=1024）出力比較
  （threadgroup 経由／device 直接の同一形状対比。#1295 が実際に性能比較する
  軸）。
- **T3**（`frag_load_on_off_bit_match_dispatch_auto`）: 本番自動選択経路
  （`dispatch_auto`）でも base/head（device-hoisted-k2）が bit 同一
  （N=512〜4096）。
- **T4**（`frag_load_transposed_bit_match`）: NT/TN/TT
  （`dispatch_strided_tiled_prepared`）を N=1024 で staged K=2・
  device-hoisted-k2 の両方で比較（`TRANS_A`/`TRANS_B` 分岐を含む新ブロック
  の転置ロード bit 一致を確認する必須ケース）。

### 非後退確認（既存テスト。実機実測）

同一実機で以下も全 pass（本イシューによる退行がないことを確認）:

- `unroll_acc_on_off_bit_match_all_candidates`／
  `unroll_acc_on_off_bit_match_dispatch_auto`／
  `unroll_acc_effective_matches_candidate_acc_product_threshold`
- `source_specialized_on_off_bit_match_all_candidates`／
  `source_specialized_on_off_bit_match_dispatch_auto`／
  `source_specialized_route_populates_only_spec_cache`
- `tests/gemm_dynamic_tile_parity.rs`（11 tests）／
  `tests/gemm_simdgroup_parity.rs`（9 tests）／
  `tests/gemm_transposed_parity.rs`（5 tests）／
  `tests/cpu_metal_parity.rs`（5 tests）／
  `tests/gemm_fine_barrier_bit_match.rs`（2 tests）／
  `tests/gemm_swizzle_bit_match.rs`（2 tests）

いずれも green（全体で 0 failed）。

## 4. env_info

- CPU/GPU: Apple M4 Max（`sysctl -n machdep.cpu.brand_string`）
- OS: macOS 26.6.2（BuildVersion 25G83）
- rustc: 1.96.0（ac68faa20 2026-05-25）
- 実行時刻の `uptime`: `2:09  up 18 days, 13:22, 20 users, load averages: 3.98 3.50 3.39`
  （複数セッションが並走する共有環境。load average が高めのため、性能を
  伴わない bit 一致検査〈GPU 負荷非依存〉のみを実施した本イシューの結果には
  影響しないが、#1295 の性能実測では低負荷環境での再測定が必要）

内部ホスト名は記載しない（`docs/real-hardware-verification-env.md`）。

## 5. #1295 への引き継ぎ

- **A/B 入口**: `MetalGemm::new_with_frag_load(&ctx, tile::FragLoadConfig {
  device_hoisted, ksteps })` が公開済み。base/head の 2 インスタンスを同一
  プロセス内に構築して interleaved に A/B 計測できる（`new_with_swizzle`
  等と同型）。
- **純カーネル時間の計測**: #1275 の GPU タイムスタンプ経路
  （`MetalContext::synchronize_with_gpu_timestamps`）が利用可能。
- **`{staged: false}` twin の作り方**: `TileConfig { staged: false, ..cfg }`
  （§3 T2 と同じ構成）。`validate` が通らない場合は候補から除外する
  （本ドキュメント T2 の設計を踏襲）。
- N=1024/2048/4096 の純カーネル時間比較・`tile::select`／本番既定への結線
  判断は本イシューのスコープ外（#1295）。

### #1295 結果

N=1024/2048/4096 の 5 回計測中央値実測完了（M4 Max。5 候補: `tgp-k1`〈=
本番既定〉・`tgp-k2`・`device-legacy`・`device-hoisted-k1`・
`device-hoisted-k2`）。**本番既定 `tgp-k1` が全 N で最速で、他 4 候補は
いずれも `tile::select` への組み込み対象外（組み込み不可）**と判定・
`tile::FRAG_LOAD_CONFIG` は不変のまま維持。詳細・数値は
`docs/perf/metal-gemm-n4096-kernel-gap.md` §10 を参照。
