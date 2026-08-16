# NEON マイクロカーネル k=4 アンロール＋ソフトウェアパイプライン導入の実測記録（#561）

イシュー #561「NEON マイクロカーネルに k=4 アンロール＋ソフトウェアパイプラインを導入」の実装記録。
`crates/backend-cpu/src/gemm_blis/microkernel/neon.rs` の既定カーネル（`kernel`。MR=8×NR=12・#559）・
12×8 A/B 対抗変種（`kernel_12x8`）の k ループを、単純な `for p in 0..kc_len` の 1 ステップ構造から、
BLIS armv8a 参照実装の `k_iter = k/4`（主ループ）・`k_left = k%4`（端数ループ）分離技法へ書き換えた。
主ループは 4 ステップ（p, p+1, p+2, p+3）を 1 チャンクとして展開し、各ステップの A/B ロードを直前
ステップの FMA 列の合間へ先出し発行する 2 段ソフトウェアパイプライン構造とした。依存イシュー #559
（PR #693・コミット 82d86e1）の上に実装した。

**本ドキュメントは REQ-8 の下限値・数値一致許容誤差を一切変更しない**。

## 状態: 実装・クロス型検査・x86 側リグレッションまで完了。aarch64 実機での bit 一致・A/B スループット実測は未実施（環境ゲートで未達）

### 実行環境ゲート判定（本イシュー実装セッション時点）

受け入れ基準の bit 完全一致・効果ベンチは aarch64 実機（Apple M4 Max・DGX Spark GB10 の
Grace CPU）でのみ有効という前提のもと、実装セッション開始時に以下を判定した
（#552・#559 と同一のゲート判定手順）:

1. `uname -sm` → `Linux x86_64`（本開発環境。実測）。aarch64 ではないため NEON 経路は実行不能。
2. `docs/real-hardware-verification-env.local.md`（Git 管理外のローカル用ファイル）→ 不存在（実測）。
   実機（M4 Max ローカル / DGX Spark GB10 の Grace CPU）への接続情報が定義されていないため到達不可。
3. `qemu-aarch64` の存在確認（`which`）→ 不在（実測）。よって
   `cargo test --target aarch64-unknown-linux-gnu` によるエミュレーション実行も本環境では不可能。

**結論**: 本セッションでは aarch64 実機に到達できないため、実装・コンパイル検証（クロス
`cargo check`／`cargo clippy`）・x86_64 側のリグレッション確認（scalar/AVX2/AVX-512 経路に
影響がないこと）までを実施し、**実測値の捏造・placeholder 値での完了扱いは行わない**
（fail-closed。#552・#559・#488 と同じ前例）。実機での bit 一致・効果ベンチ計測は aarch64
実機へアクセス可能な後続セッション・Agent（`bench-runner` 委譲想定）が引き継いで実施する。

## 変更内容（実装済み・コンパイル検証済み）

- `crates/backend-cpu/src/gemm_blis/microkernel/neon.rs`:
  - `kernel`（既定 8×12）・`kernel_12x8`（12×8 A/B 対抗変種）いずれも k ループを
    `k_main = kc_len - (kc_len % 4)` の主ループ（4 ステップチャンク展開・先読みインター
    リーブ）と、`k_main..kc_len` の端数ループ（#559 と同一の 1 ステップ構造をそのまま
    維持）へ分離した。
  - **acc 分割なし**: アキュムレータ（`acc[i][j]`）は主ループ・端数ループを通じて単一の
    まま、FMA の演算順序（p 昇順・行 i 昇順・`[0]`→`[1]`→`[2]`）を一切変えていない。
    変更したのはロード（`vld1q_f32`）の発行位置（ソースコード上の並び）のみ。
  - 先読みはチャンク内（p〜p+3、p は 4 の倍数）に限定し、チャンク境界を越えて次チャンク・
    次領域を読み出さない構造にした（SAFETY コメントで `p+3 <= k_main-1 < kc_len` を証明）。
  - 冒頭モジュールコメントに `## k=4 アンロール＋ソフトウェアパイプライン（イシュー #561）`
    節を追加し、技法・bit 完全一致契約が保たれる理由・先読み境界・レジスタ収支のトレード
    オフを記録した。
- `crates/backend-cpu/tests/gemm_blis_parity.rs`: `SHAPE_GRID_K` に `2`・`4`・`6` を追加し
  （既存の `1, 3, 255, 257, 700` と合わせ k%4 の剰余 0/1/2/3 を全網羅）、コメントで根拠を
  記載した。
- `crates/backend-cpu/src/gemm_blis/mod.rs`: `neon_8x12_and_12x8_match_scalar_forced_bit_exact`
  の k を単一の 700 から `[700, 701, 702, 703]` の反復へ拡張した。元の k=700 は KC=256 の
  ブロック分割で各領域の kc_len が 256/256/188（すべて 4 の倍数）となり端数ループを一切
  通らないため、k=701〜703（最終領域 kc_len が 189/190/191 で k%4 が 1/2/3）を追加して
  新設の端数分離ロジックを実際に検証対象へ含めた。

## 設計判断の要点

- **bit 完全一致契約（REQ-2）の維持**: ロードの並べ替えは丸め・演算順序に影響しないため、
  `p` 昇順の FMA 連鎖・レーン間縮約なしという契約（#552・#559 で確立）はそのまま保たれる
  理論的根拠がある。実機での実測は下記「未実測」節を参照。
- **先読み境界の設計**: 主ループはチャンク内（4 ステップ）でのみ先読みし、チャンク境界・
  端数領域への越境読み出しを構造的に排除した。端数ループ自体は変更せず #559 の実装をその
  まま流用したため、この部分の回帰リスクはない。
- **レジスタ収支のトレードオフ（cross 型コンパイル済みバイナリで実測 — 実機不要で判定可能）**:
  既定カーネルは acc 24 本 + 現行ステップのオペランド 5 本で 29 本（v0〜v31 の 32 本以内）だが、
  先読み対象ステップのオペランド（最大 5 本）が一時的に重複して生存するため、チャンク内の
  短時間ではあるが 32 本を超えうる（スピルの恐れ。イシュー #561 の実装計画 §3.3 で事前に
  想定済みのリスク）。この懸念は「コンパイル済み機械語にスピル命令（NEON ベクタレジスタの
  スタック退避 `str`／`stur` の `[sp, ...]` オペランド）が存在するか」で実機なしに判定
  できる。`cargo build -p backend-cpu --release --target aarch64-unknown-linux-gnu` の成果物
  （`target/aarch64-unknown-linux-gnu/release/libbackend_cpu.rlib`）を
  `llvm-objdump --disassemble-symbols=<mangled `neon::kernel`>` で逆アセンブルし確認した:
  主ループ（アドレス `0x94`〜`0x34c`。逆方向分岐 `b.lo 0x94` で確認できるループ本体）は
  `fmla` 命令 96 個（4 ステップ × 8 行 × 3 レジスタ）を含むが、`[sp` を参照する命令は
  **0 件**（`vld1q_f32` に対応する `ldr qN, [x10]` 等の通常ロードのみ存在し、スタックへの
  ベクタレジスタ退避は一切生成されていない）。よって本 PR の cross コンパイル成果物では
  スピルは発生していないと確認できる（LLVM がレジスタ割付を成功させた）。ただしこれは
  「この機械語列にスピル命令が存在しない」という静的事実の確認であり、**実行時の
  スループット・レイテンシ（キャッシュミス・パイプラインストール等）は依然として aarch64
  実機でのみ計測可能**なため、下記「未実測」節の効果ベンチは省略しない。
- **12×8 変種の扱い**: 既定カーネルと同型の k=4 アンロール構造を適用した（#559 が残した
  8×12 vs 12×8 の A/B 比較の公平性を保つため、片側のみ最適化しない）。

## 検証済み事項（本セッションで実施）

| 検証 | コマンド | 結果 |
|---|---|---|
| フォーマット | `cargo fmt --all` | 差分なし（PostToolUse hook 自動適用込み） |
| lint（x86_64） | `cargo clippy --workspace --all-targets --all-features -- -D warnings` | 警告なし |
| NEON クロス型検査 | `cargo check -p backend-cpu --target aarch64-unknown-linux-gnu` | 成功 |
| NEON クロス型検査 | `cargo check -p backend-cpu --target aarch64-apple-darwin` | 成功 |
| NEON クロス型検査（テスト込み） | `cargo check -p backend-cpu --target aarch64-unknown-linux-gnu --tests` | 成功 |
| NEON クロス clippy | `cargo clippy -p backend-cpu --target aarch64-unknown-linux-gnu --all-targets -- -D warnings` | 警告なし |
| NEON レジスタスピル静的検査 | `cargo build -p backend-cpu --release --target aarch64-unknown-linux-gnu` 後 `llvm-objdump --disassemble-symbols=<mangled symbol>` で `neon::kernel` を逆アセンブルし主ループ内 `[sp` 参照を検査 | 主ループ（`fmla` 96 個）内に `[sp` 参照 0 件（スピルなし） |
| リグレッション | `cargo test --workspace` | 全 pass。`SHAPE_GRID_K` へ追加した k=2/4/6 は x86_64 では scalar/AVX2/AVX-512 経路（`gemm_naive` との bit 完全一致検証）を通じて実行済み。`neon_8x12_and_12x8_match_scalar_forced_bit_exact`（`#[cfg(target_arch = "aarch64")]` 限定の lib テスト）のみコンパイル対象外のため x86_64 側では実行されない（NEON 経路の実行検証は下記「未実測」節） |

## 未実測（fail-closed・後続セッションへの引き継ぎ事項）

以下 2 項目は受け入れ基準だが、本セッションでは aarch64 実機に到達できないため未実施:

1. **bit 完全一致**: `cargo test -p backend-cpu --release --test gemm_blis_parity`
   （拡張後の `SHAPE_GRID_K = [1, 2, 3, 4, 6, 255, 257, 700]` を含む）および
   `neon_8x12_and_12x8_match_scalar_forced_bit_exact`（lib 単体テスト。k=700〜703 反復）を
   aarch64 実機（M4 Max または Grace CPU）で実行する。aarch64 では `Isa::detect` が無条件に
   NEON（既定 8×12）を選ぶため、`gemm_blis_parity` の実行自体が新しい k=4 アンロール経路の
   `gemm_naive` との bit 完全一致検証になる。**不一致の場合は実装を revert し、tolerance の
   変更・テスト側の調整で通すことは行わない**（`.claude/rules/coding-rust.md`）。
2. **効果ベンチ（スループット比較）**: 既存の `#[ignore]` テスト
   `neon_8x12_vs_12x8_ab_median_throughput`（`--ignored --nocapture`・5 回計測中央値・計測順
   交互化済み。新規ハーネス追加なしで既存資産をそのまま再利用）を、変更前コミット
   （`82d86e1`。#559 マージ時点）と本イシュー実装後の HEAD の双方で `--release` 実行し、
   512/1024/2048 平方形状の中央値スループットを比較する。再現コマンド:

   `neon_8x12_vs_12x8_ab_median_throughput` は `crates/backend-cpu/src/gemm_blis/mod.rs`
   の `#[cfg(test)] mod tests` 内に定義された **lib テスト**（`tests/gemm_blis_parity.rs`
   の統合テストではない）ため、`--lib` で対象を絞る:

   ```bash
   # 変更前（82d86e1）
   git worktree add /tmp/neon-k4-baseline 82d86e1
   cd /tmp/neon-k4-baseline
   cargo test -p backend-cpu --release --lib \
     -- --ignored neon_8x12_vs_12x8_ab_median_throughput --nocapture

   # 変更後（本 PR HEAD）
   cd <本リポジトリの作業ブランチ>
   cargo test -p backend-cpu --release --lib \
     -- --ignored neon_8x12_vs_12x8_ab_median_throughput --nocapture
   ```

   良化・悪化いずれであっても実測値をそのまま記録する（**placeholder 値・捏造値での達成
   扱いは行わない**）。レジスタスピルで悪化が確認された場合は、アンロール自体
   （k_main/k_left 分離）は維持したままロード配置（先読みインターリーブ）のみ簡素化して
   再計測する余地を残す（イシュー #561 の実装計画 §3.3・§8 リスク 1）。

上記 2 項目は aarch64 実機へアクセス可能な後続セッション・Agent（`bench-runner` 委譲想定）
が引き継いで実施する。
