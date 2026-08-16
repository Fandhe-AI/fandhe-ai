# NEON マイクロカーネル MR=8, NR=12（24 accumulator）拡張の実測記録（#559）

イシュー #559「NEON マイクロカーネルを MR=8, NR=12（24 accumulator）へ拡張」の実装記録。
`crates/backend-cpu/src/gemm_blis/microkernel/neon.rs` の既定マイクロカーネルを、旧
MR=8×NR=8（アキュムレータ `float32x4_t` × 16 本）から MR=8×NR=12（同 24 本）へ拡張し、
Apple M1 系 firestorm コア向けの MR=12×NR=8 変種（同じく acc 24 本）を A/B 計測専用として
併設した。依存イシュー #552（`vfmaq_laneq_f32` 化）の技法をそのまま踏襲し、レーン選択 FMA
方式・累積契約（p 昇順・レーン間縮約なし）は変えずタイル形状のみを拡張している。

**本ドキュメントは REQ-8 の下限値・数値一致許容誤差を一切変更しない**。

## 状態: 実装・クロス型検査・x86 側リグレッションまで完了。aarch64 実機での bit 一致・A/B スループット実測は未実施（環境ゲートで未達）

### 実行環境ゲート判定（本イシュー実装セッション時点）

受け入れ基準の bit 完全一致・A/B 実測は aarch64 実機（Apple M4 Max・DGX Spark GB10 の
Grace CPU）でのみ有効という前提のもと、実装セッション開始時に以下を判定した
（#552・`docs/perf/cpu-gemm-neon-laneq-fma.md` と同一のゲート判定手順）:

1. `uname -sm` → `Linux x86_64`（本開発環境。実測）。aarch64 ではないため NEON 経路は実行不能。
2. `docs/real-hardware-verification-env.local.md`（Git 管理外のローカル用ファイル）→ 不存在（実測）。
   実機（M4 Max ローカル / DGX Spark GB10 の Grace CPU）への接続情報が定義されていないため到達不可。
3. `qemu-aarch64` の存在確認（`which`）→ 不在（実測）。よって
   `cargo test --target aarch64-unknown-linux-gnu` によるエミュレーション実行も本環境では不可能。

**結論**: 本セッションでは aarch64 実機に到達できないため、実装・コンパイル検証（クロス
`cargo check`／`cargo clippy`）・x86_64 側のリグレッション確認（scalar/AVX2 経路に影響がない
こと）までを実施し、**実測値の捏造・placeholder 値での完了扱いは行わない**（fail-closed。
#552・#488 と同じ前例）。実機での bit 一致・A/B スループット計測は aarch64 実機へアクセス
可能な後続セッション・Agent（`bench-runner` 委譲想定）が引き継いで実施する。

## 変更内容（実装済み・コンパイル検証済み）

- `crates/backend-cpu/src/gemm_blis/microkernel/neon.rs`:
  - 既定カーネル（`kernel`）を MR=8×NR=8 から **MR=8×NR=12**（acc `[[float32x4_t; 3]; 8]`
    = 24 本、B 3 本（`b0/b1/b2`）、A 2 本（`a0/a1`）= 計 29 本。v0〜v31 の 32 本に収まる）へ
    拡張した。C タイルロード／ストア・k ループの B/A ロード・`fma_row!` マクロを 3 レジスタ
    ぶんへ更新し、静的検査（`assert!(MR * NR <= 256)`・`assert!(MR == 8 && NR == 12)`）・
    冒頭 3 つの境界 `assert_eq!`・SAFETY コメントのオフセット上界証明を新形状に合わせて
    更新した（REQ-8 手動境界検査は省略していない）。
  - **12×8 変種**（`kernel_12x8`。定数 `MR_12X8 = 12`／`NR_12X8 = 8`）を新規追加した。
    acc `[[float32x4_t; 2]; 12]` = 24 本、A 3 本（`a0/a1/a2`）、B 2 本 = 計 29 本。既定
    カーネルと同じ p 昇順 FMA 連鎖契約・境界検査を持つが、駆動経路
    （`gemm_blis::dispatch_region`）には接続しない A/B 計測専用コード（イシュー #559 §2.3）。
- `crates/backend-cpu/src/gemm_blis/microkernel.rs`: `NeonKernel`（既定 8×12。`neon::{MR,NR}`
  経由で自動追従）に加え、`Neon12x8Kernel`（`cfg(target_arch = "aarch64")` の ZST。
  `Microkernel` trait 実装で `neon::kernel_12x8` へ配線）を追加した。
- `crates/backend-cpu/src/gemm_blis/mod.rs`:
  - `panel_capacity_upper_bounds_all_block_iterations` の `KERNEL_DIMS` へ `(8, 12)`・
    `(12, 8)` を追加（`(4,4)`・`(6,16)`・`(8,32)` は既存のまま）。
  - `neon_8x12_and_12x8_match_scalar_forced_bit_exact`（aarch64 限定・通常テスト）: 既定
    8×12・12×8 変種いずれも `ScalarKernel` 強制経路と bit 完全一致することを検証する。
  - `neon_8x12_vs_12x8_ab_median_throughput`（aarch64 限定・`#[ignore]`）: 512/1024/2048
    平方形状で両カーネルを 5 回計測し中央値を標準出力へ報告する（`.claude/rules/coding-rust.md`
    の 5 回中央値規約）。採用可否は本テストの出力を見た人間／後続セッションが判断する
    （テスト自体は勝敗を assert しない）。
- `crates/backend-cpu/tests/gemm_blis_parity.rs`: コメント「neon 8x8」を「neon 8x12（既定。
  #559 で 8x8 から拡張）」へ更新。形状グリッド（`SHAPE_GRID_M`／`SHAPE_GRID_N`）へ NR=12
  境界跨ぎ形状（n=11,12,13）・MR=8 境界跨ぎ形状（m=11,12,13）を追加した。

## 設計判断の要点

- **レジスタ収支**: 8×12・12×8 いずれも acc 24 本 + オペランド 5 本 = 計 29 本で aarch64
  の v0〜v31（32 本）に収まる。コンパイラのスピル余地は小さいが、実効性能・スピル有無は
  実機計測でのみ確認可能（x86_64 環境では判定不能。fail-closed 引き継ぎ対象）。
- **bit 完全一致契約（REQ-2）**: C 各要素の累積は「p 昇順の FMA 連鎖・レーン間縮約なし」の
  まま形状だけを変えるため、`gemm_naive` との bit 完全一致は理論上維持される。ただし
  受け入れ基準に従い、実機での parity 実測で不一致が出た場合は **実装を revert して差分の
  性質（最大 ULP 差・発生形状）を報告し停止する**（tolerance の変更・複合判定への降格は
  行わない。採用可否はユーザー承認事項）。
- **端タイル（NR=12 の非 8 の倍数）**: NR=12 は 4 の倍数だが 8 の倍数でないため端タイル
  （`nr_eff < 12`）の頻度が変わるが、`pack_b` のゼロ充填と C 書き戻しの `nr_eff` 制限で
  既存ドライバがそのまま正しく扱う（ドライバ変更なし。`pack.rs` は MR/NR ジェネリック）。
- **12×8 変種の位置づけ**: 駆動経路（`dispatch_region`）には接続しない A/B 専用コードとして
  残る。実機 A/B 実測で 12×8 が優位と判明した場合、既定（`neon::{MR, NR, kernel}`）の入替は
  後続セッションのスコープとする（本 PR 本文で追跡を明示）。

## 検証済み事項（本セッションで実施）

| 検証 | コマンド | 結果 |
|---|---|---|
| フォーマット | `cargo fmt --all` | 差分なし（PostToolUse hook 自動適用込み） |
| lint（x86_64） | `cargo clippy --workspace --all-targets --all-features -- -D warnings` | 警告なし |
| NEON クロス型検査 | `cargo check -p backend-cpu --target aarch64-unknown-linux-gnu` | 成功 |
| NEON クロス型検査 | `cargo check -p backend-cpu --target aarch64-apple-darwin` | 成功 |
| NEON クロス clippy | `cargo clippy -p backend-cpu --target aarch64-unknown-linux-gnu --all-targets -- -D warnings` | 警告なし |
| リグレッション | `cargo test --workspace` | 全 pass（x86_64 では scalar/AVX2/AVX-512 経路。`KERNEL_DIMS` 更新後の `panel_capacity` 検証・parity グリッド追加分を含む） |

## 未実測（fail-closed・後続セッションへの引き継ぎ事項）

以下 2 項目は受け入れ基準だが、本セッションでは aarch64 実機に到達できないため未実施:

1. **bit 完全一致**: `cargo test -p backend-cpu --release --test gemm_blis_parity` および
   `neon_8x12_and_12x8_match_scalar_forced_bit_exact`（lib 単体テスト）を aarch64 実機
   （M4 Max または Grace CPU）で実行する。aarch64 では `Isa::detect` が無条件に NEON（既定
   8×12）を選ぶため、`gemm_blis_parity` の実行自体が新形状 NEON 経路の `gemm_naive` との
   bit 完全一致検証になる。**不一致の場合は実装を revert し、tolerance の変更・テスト側の
   調整で通すことは行わない**（`.claude/rules/coding-rust.md`）。
2. **A/B スループット比較**: `neon_8x12_vs_12x8_ab_median_throughput`（`--ignored`）を
   `--release` 実行し、8×12（既定）と 12×8（firestorm 型）の中央値を比較する。良い方を
   既定として採用するかはこの実測結果を見た人間／後続セッションが判断する（不採用の判断
   も含め、実測なしに達成を偽装しない）。

### 再現コマンド（後続セッション向け）

```bash
# bit 完全一致（aarch64 実機）
cargo test -p backend-cpu --release --test gemm_blis_parity
cargo test -p backend-cpu --release --lib gemm_blis::tests::neon_8x12_and_12x8_match_scalar_forced_bit_exact

# A/B スループット（aarch64 実機。--ignored ハーネス使用）
cargo test -p backend-cpu --release --lib gemm_blis::tests::neon_8x12_vs_12x8_ab_median_throughput -- --ignored --nocapture
```

実機接続手順は `docs/real-hardware-verification-env.md` §3-4（rsync 転送・除外フィルタ厳守）を参照する。

## スコープ外（本 PR に含めない）

- k アンロールは #561、MC/KC/NC 再選定は #564 の対象（イシュー #559 では扱わない）。
- 12×8 変種の採用判断後の削除／既定入替は、実機 A/B 実測を行う後続セッションのスコープ。
