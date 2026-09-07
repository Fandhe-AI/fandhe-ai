# CPU GEMM B 側 laneq ベクトル転置版マイクロカーネルの実装記録

イシュー #1317（親: #1316。後続: #1318「両実機実測・採否判定」）。

**本イシュー（#1317）は候補の実装・bit 完全一致回帰への統合までを扱う。
両実機（DGX Spark GB10・Apple M4 Max）実測・採否判定・本番結線は #1318 が
引き継ぐ**（§6 参照）。

## 1. 目的・位置づけ

`crates/backend-cpu/src/gemm_blis/microkernel/neon.rs` には既定カーネル
`compute`（A 側レーン参照・行優先 acc・#559/#561）と、#748 で併設された
B 側レーン参照変種 `compute_b_laneq`（`vfmaq_laneq_f32` のレーン参照
オペランドを B 側にし、acc を列優先 `acc[j][h]` で保持）がある。

`compute_b_laneq` は row-major の C タイルと列優先 acc の間の転置を、
スタック上 `[f32; 4]` を介した **スカラー gather/scatter**（入口 96 要素・
出口 96 要素）で実装しており、#748 のドキュメント（`docs/perf/cpu-gemm-neon-b-laneq-fma.md`
スコープ外節・`neon.rs` 冒頭 #748 節）でも「`vzip`/`vtrn` 系ベクトル化転置は
追加最適化候補として out-of-scope」と明記されていた。

`docs/perf/cpu-gemm-candle-cpu-retune.md` §8 の次候補 1（B 側 laneq の
ベクトル転置化）が本イシューの起票元。`SharedB`／`SharedBPcOuter`（#1144）・
`IcDynamic`（#1367）はいずれも本番既定 `RowPanel` に届かず REJECT 済みで、
残る候補がマイクロカーネル側の改善である。

## 2. 設計

### 2.1 新カーネル `compute_b_laneq_vec`（`neon.rs`）

`compute_b_laneq` と同型の構造（列優先 acc `[[float32x4_t; 2]; NR]`・
`vld1q_f32_x2`/`x3` ロード・k=4 アンロール＋2 段ソフトウェアパイプライン・
端数ループ）を保ちつつ、C タイルの入口/出口転置をスカラー gather/scatter
から `transpose_4x4`（`vtrn1q_f32`/`vtrn2q_f32` → `vreinterpretq_f64_f32` →
`vtrn1q_f64`/`vtrn2q_f64` → `vreinterpretq_f32_f64` の 8 命令構成）による
in-register 4×4 転置へ置き換える。

- **入口転置**: 行グループ h∈{0,1}（MR=8 固定）× 列グループ g∈0..3（NR=12
  固定）の 6 ブロックについて、row-major の 4 行（各 4 要素が連続）を
  `vld1q_f32` で直接ロードし `transpose_4x4` で列ベクトル 4 本へ変換して
  `acc[4g+j][h]` へ格納する。
- **出口転置**: 転置は対合（自己逆写像）であるため、同じ `transpose_4x4`
  を列優先 acc → row-major C の変換にもそのまま適用できる。
- **k ループの共有**: `compute_b_laneq` の `fma_col!`／`load_step!` マクロと
  主・端数ループを `b_laneq_k_loop`（`#[inline(always)] unsafe fn`）へ
  抽出し、`compute_b_laneq`・`compute_b_laneq_vec` の両方が呼ぶ。これにより
  両関数の差分が C タイル入口/出口の転置方式のみであることをコードで保証
  する（挙動不変のリファクタは既存テスト `compute_b_laneq_matches_compute_bit_exact`
  が保護）。

### 2.2 bit 完全一致契約の整理

- 対 `compute_b_laneq`（#748）: FMA 連鎖が完全に同一（`b_laneq_k_loop` を
  共有）で、転置は `vtrn*`／`vreinterpretq_*` による純粋なデータ移動
  （丸めなし・NaN payload/符号も bit 単位で保存）のみであるため、
  **NaN を含む全入力で bit 完全一致**する
  （`compute_b_laneq_vec_matches_compute_b_laneq_bit_exact_including_nan`
  で検証。`transpose_4x4_roundtrip_preserves_bits` で転置自体の bit 保存も
  独立に検証）。
- 対 `compute`（既定・A レーン参照）: #748 と同じ IEEE-754-2008 §6.2 の
  NaN 選択規則が実装依存という理由により、**有限値入力に限り bit 完全
  一致**（`compute_b_laneq_vec_matches_compute_bit_exact`）。

### 2.3 公開入口・`Microkernel` トークン

`compute_b_laneq` と同型の 2 段構成:

- `kernel_b_laneq_vec_with_ldc`: 境界検査つき（`Result` を返す。本番
  コンパイル対象）。
- `#[cfg(test)] kernel_b_laneq_vec`: `assert!` 検査版・`ldc = NR` 密パッキング
  契約固定（テスト専用。「本番経路 panic 禁止」規約対応で `#[cfg(test)]`
  限定）。

`microkernel.rs` に `NeonBLaneqVecKernel`（`#[cfg(all(target_arch = "aarch64", test))]`）
を追加。`NeonBLaneqKernel`（#748）と同じ理由で `run_with_ldc` から
`kernel_b_laneq_vec_with_ldc` へ直接委譲し、デフォルト実装のヒープ確保
ギャザー/スキャッタに頼らない（`Neon12x8Kernel` で判明した A/B 公平性
問題を回避）。

### 2.4 `GemmDriverVariant::RowPanelBLaneqVec`（`mod.rs`）

`GemmDriverVariant` に `RowPanelBLaneqVec`（`#[cfg(target_arch = "aarch64")]`）
を追加し、新ヘルパー `gemm_blis_parallel_row_panel_with_kernel<K: Microkernel>`
（`RowPanel` 用 `gemm_blis_parallel_row_panel_with_blocks` と同一の行パネル
分割・rayon 並列化ロジックを持つが、`dispatch_region`〈本番実行時 ISA
検出〉を経由せず呼び出し元が指定した `K` へ直接 `gemm_blis_region` を
呼ぶ）へ委譲する。これにより `RowPanel` との A/B 比較の差分をマイクロ
カーネル自体（C タイル転置のベクトル化）だけに限定できる。

`ALL_VARIANTS` 相当は `all_gemm_driver_variants()`（aarch64 版／非 aarch64
版の 2 定義。x86_64 で `let mut v = vec![…]; #[cfg(aarch64)] v.push(…)` と
すると `unused_mut` lint が `-D warnings` で落ちる罠を避ける方式）へ一元化
し、A/B ハーネス 2 箇所（`gemm_blis_variant_ab_1024_2048`／`_4096`）・全候補
bit 一致回帰テストの計 3 箇所から参照する。

## 3. スコープ外（不変事項）

- `pack.rs`（`pack_a`／`pack_b`）のレイアウトは変更しない。
- 本番 dispatch（`mod.rs` aarch64 版 `dispatch_region` → `NeonKernel`）・
  `gemm_blis_parallel`・`gemm_blis_bias_act_parallel` は変更しない。
- `GemmDriverVariant::RowPanel` は引き続き `dispatch_region` を呼ぶ
  （本番と分岐完全一致という A/B 基準線の価値を保つ）。
- #748 のスカラー転置カーネル `NeonBLaneqKernel` を独立 variant として
  併記する対応（転置コストを単独で分離する目的）は受入基準外のため
  行っていない。

## 4. テスト

### 4.1 `neon.rs` 単体（`#[cfg(target_arch = "aarch64")]` モジュール内）

- `kernel_b_laneq_vec_matches_hand_computed_subset`（手計算 2×2）
- `kernel_b_laneq_vec_with_larger_ldc_matches_tight_packing_and_preserves_gap`
  （`ldc > NR` でギャップ列非破壊・`ldc = NR` と bit 完全一致）
- `kernel_b_laneq_vec_rejects_ldc_smaller_than_nr`（`Result::Err`。panic しない）
- `compute_b_laneq_vec_matches_compute_bit_exact`（既定 `kernel_with_ldc` と
  有限値 bit 完全一致。kc_len グリッド `[4,5,6,7,32,33,34,35]` で k%4 の
  剰余 0..3 を網羅）
- `compute_b_laneq_vec_matches_compute_b_laneq_bit_exact_including_nan`
  （§2.2 の最重要検証。NaN payload・符号違いを含む混入入力で `compute_b_laneq`
  と bit 完全一致）
- `compute_b_laneq_vec_nan_input_does_not_panic`
- `transpose_4x4_roundtrip_preserves_bits`（2 回転置で恒等・NaN payload
  含む bit 保存）

### 4.2 `mod.rs`

- `neon_8x12_and_12x8_match_scalar_forced_bit_exact` に `NeonBLaneqVecKernel`
  の `ScalarKernel` 強制経路との bit 一致比較を追加。
- `gemm_blis_parallel_variant_all_candidates_match_naive_bit_exact`:
  `all_gemm_driver_variants()` 経由で `RowPanelBLaneqVec` も 5 形状 × スレッド
  数 1/2/3/16 で `gemm_naive` と bit 一致。
- `gemm_blis_row_panel_b_laneq_vec_matches_row_panel_bit_exact_across_shapes_and_threads`
  （aarch64 cfg）: 8 形状（端タイル・`n%12≠0`・`k%KC≠0`・`k=0`・512³）×
  スレッド数 1/2/3/16・非ゼロ C 初期値で `RowPanel` と bit 一致。
- `gemm_blis_row_panel_b_laneq_vec_matches_row_panel_bit_exact_large`
  （`#[ignore]`・aarch64 cfg）: 1024/2048/4096 正方・release。

## 5. 検証コマンド

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p fandhe-ai-backend-cpu --all-features
cargo test -p fandhe-ai-backend-cpu --release -- --ignored gemm_blis_row_panel_b_laneq_vec_matches_row_panel_bit_exact_large --nocapture
cargo check -p fandhe-ai-backend-cpu --tests --all-features --target x86_64-unknown-linux-gnu
make check-cross-cpu-tests
cargo check -p fandhe-ai-backend-cpu --tests --target aarch64-unknown-linux-gnu
RUSTDOCFLAGS=-Dwarnings cargo doc -p fandhe-ai-backend-cpu --no-deps
cargo test --workspace
```

いずれも本エージェント実行環境（arm64 macOS・native aarch64）で実行済み
（全 green。`gemm_blis_row_panel_b_laneq_vec_matches_row_panel_bit_exact_large`
含む）。

## 6. スモーク実行（採否根拠にしない）

`gemm_blis_variant_ab_1024_2048`（`--release -- --ignored … --nocapture`）を
共有負荷下の本エージェント実行環境で 1 回実行し、`variant=RowPanelBLaneqVec`
の行が出力されることのみ確認した（実行値は共有負荷下のローカル smoke で
あり、正式な採否根拠にはしない）。

## 7. 両実機実測・採否判定（#1318 記入欄）

| 実機 | N=1024 | N=2048 | N=4096 | 対 RowPanel 比 | 実施日 |
|------|--------|--------|--------|----------------|--------|
| DGX Spark GB10 | 未実測 | 未実測 | 未実測 | — | — |
| Apple M4 Max | 未実測 | 未実測 | 未実測 | — | — |

採否（ADOPT / REJECT）: 未判定（#1318 が実施）。
