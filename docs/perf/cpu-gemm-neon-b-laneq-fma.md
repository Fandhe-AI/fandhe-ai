# NEON マイクロカーネル B 側レーン参照 FMA 変種の実装記録（#748）

イシュー #748「NEON マイクロカーネルのレーン参照 FMA 化（`vfmaq_laneq_f32`）」の実装記録。
`crates/backend-cpu/src/gemm_blis/microkernel/neon.rs` の既定カーネル（`kernel`。MR=8×NR=12・
A 側レーン参照・#559/#561）に対し、`vfmaq_laneq_f32` のレーン参照オペランドを **B 側**へ
入れ替えた変種（`compute_b_laneq`／`kernel_b_laneq_with_ldc`／`kernel_b_laneq`）を追加した。
gemm crate（faer 実体）・matrixmultiply が採る技法（B の 1 ベクトルロードの各レーンを
`vfmaq_laneq_f32` のレーン参照として複数列分の FMA へ供給する）に相当する。帰結として
アキュムレータ配置は行優先（既定。`acc[i][g]` = C の行 i・列 4g..4g+4）から
列優先（`acc[j][h]` = C の列 j・行 4h..4h+4）へ転置される。

**本ドキュメントは REQ-8 の下限値・数値一致許容誤差を一切変更しない**。

## 状態: 実装・クロス型検査・x86 側リグレッションまで完了。aarch64 実機での bit 完全一致・
A/B スループット実測は未実施（環境ゲートで未達）

### 実行環境ゲート判定（本イシュー実装セッション時点）

受け入れ基準の効果ベンチは aarch64 実機（Apple M4 Max・DGX Spark GB10 の Grace CPU）でのみ
有効という前提のもと、実装セッション開始時に以下を判定した（#552・#559・#561 と同一の
ゲート判定手順）:

1. `uname -sm` → `Linux x86_64`（本開発環境。実測）。aarch64 ではないため NEON 経路は
   実行不能。
2. `docs/real-hardware-verification-env.local.md`（Git 管理外のローカル用ファイル）→
   不存在（実測）。実機（M4 Max ローカル / DGX Spark GB10 の Grace CPU）への接続情報が
   定義されていないため到達不可。
3. `qemu-aarch64` の存在確認（`which`）→ 不在（実測）。よって
   `cargo test --target aarch64-unknown-linux-gnu` によるエミュレーション実行も本環境では
   不可能。

**結論**: 本セッションでは aarch64 実機に到達できないため、実装計画（#748 実装計画 §2）に
従い、**既定ディスパッチ（`NeonKernel`）への接続は行わず変種併設**とし、実装・コンパイル
検証（クロス `cargo check`／`cargo clippy`／`cargo build --release`）・x86_64 側の
リグレッション確認までを実施した。`neon` モジュール自体が `#[cfg(target_arch = "aarch64")]`
限定（`std::arch::aarch64` intrinsics に依存）のため、追加した bit 完全一致テスト
（`compute_b_laneq_matches_compute_bit_exact` 等）は x86_64 開発環境ではコンパイル対象外で
あり実行できない（`qemu-aarch64` も不在。上記ゲート判定参照）。**実測値の捏造・
placeholder 値での完了扱いは行わない**（fail-closed。#552/#559/#561/#488 と同じ前例）。
aarch64 実機での bit 完全一致・A/B スループット計測は後続セッション・Agent
（`bench-runner` 委譲想定）が引き継いで実施する。

## 変更内容（実装済み・コンパイル検証済み）

- `crates/backend-cpu/src/gemm_blis/microkernel/neon.rs`:
  - `compute_b_laneq(ap, bp, c, ldc, kc_len)`: B 側レーン参照 FMA の演算本体。列優先
    `acc[j][h]`（`[[float32x4_t; 2]; NR]`）を保持し、`vfmaq_laneq_f32::<lane>(acc, a_h, b_g)`
    （`acc + a_h * b_g[lane]`）で列 `j = 4g + lane` へ累積する。C タイル入口ロード・出口
    ストアは row-major（`c[i*ldc+j]`）↔ 列優先 acc の転置を伴うため、新規 unsafe 面を
    広げないスカラー gather/scatter（スタック上 `[f32; 4]` 経由）で実装した（`vzip`/`vtrn`
    ベクトル化転置は導入せず、実測で入口/出口コストが支配的と判明した場合の追加最適化候補
    として計画時点で out-of-scope とした）。
  - k ループは `vld1q_f32_x2`（A: 8 行を 1 命令）・`vld1q_f32_x3`（B: 12 列を 1 命令）による
    複数レジスタ同時ロードを用い、`kernel`（既定）と同型の k=4 アンロール＋2 段ソフトウェア
    パイプライン構造（#561）を踏襲した。先読み境界の証明は既定カーネルと同一（
    `p+3 <= k_main-1 < kc_len`）。
  - `kernel_b_laneq_with_ldc`（`Result` 版・`ldc` 一般化。`kernel_with_ldc` と同型）・
    `kernel_b_laneq`（`assert!` 版・`ldc = NR` 密パッキング固定。`kernel` と同型）を公開
    入口として追加した。
  - モジュール冒頭に `## B 側レーン参照 FMA 変種（イシュー #748）` 節を追加し、技法・
    bit 完全一致契約が可換性により保たれる理由・転置コストのトレードオフ・複数レジスタ
    ロードの stable 可用性を記録した。
  - 既存 `kernel`／`kernel_with_ldc`／`kernel_12x8` は一切変更していない（A/B 比較の基準・
    公開 API 非破壊）。
- `crates/backend-cpu/src/gemm_blis/microkernel.rs`: `NeonBLaneqKernel`（`Microkernel`
  トークン。MR=8/NR=12）を追加した。`Neon12x8Kernel` で判明した A/B 公平性問題（ヒープ
  確保の非対称。同トークンのドキュメント参照）を避けるため、`run_with_ldc` は最初から
  `kernel_b_laneq_with_ldc` への直接委譲とした（デフォルト実装のヒープ確保ギャザー/
  スキャッタに頼らない）。既定の `dispatch_region` 経路には接続しない。
- `crates/backend-cpu/src/gemm_blis/mod.rs`:
  - `neon_8x12_and_12x8_match_scalar_forced_bit_exact` に `NeonBLaneqKernel` の
    `ScalarKernel` 強制経路との bit 完全一致検証を追加した（既存の k グリッド
    `[700, 701, 702, 703]` を共用）。
  - `neon_8x12_vs_b_laneq_ab_median_throughput`（`#[ignore]`。5 回計測中央値・ウォーム
    アップ・交互実行。512/1024/2048/4096 平方形状）を `neon_8x12_vs_12x8_ab_median_throughput`
    と同型で追加した。
- `crates/backend-cpu/src/gemm_blis/microkernel/neon.rs`（テストモジュール）:
  `kernel_b_laneq_with_ldc` 単体の手計算 2x2・`ldc` 拡張（ギャップ列非破壊）・
  `ldc < NR` エラーテストに加え、**本イシューの最重要ローカル検証**である
  `compute_b_laneq_matches_compute_bit_exact`（既定 `kernel_with_ldc` との `assert_eq!`
  bit 比較。k%4 剰余 0/1/2/3 網羅の kc_len グリッド）を追加した。

## 設計判断の要点

- **bit 完全一致契約（REQ-2）の維持**: IEEE-754 の fused multiply-add は乗数 2 項について
  可換（`acc + b*a` と `acc + a*b` は bit 同一）であり、各 C 要素は引き続き p ごとに 1 回
  だけ FMA を受け p 昇順の連鎖は不変であるため、オペランド入れ替えは理論上 bit 完全一致を
  保つ。C タイル転置はデータ移動のみで丸め・演算順序に影響しない。この理論的根拠の実機
  検証（`compute_b_laneq_matches_compute_bit_exact`）は下記「未実測」節のとおり本セッション
  では未実施であり、aarch64 実機で実際に確認するまでは理論上の根拠にとどまる。
  なお bit 完全一致契約は「FMA 乗数可換性」だけでなく、`compute_b_laneq` が
  `vld1q_f32_x2`／`vld1q_f32_x3` の複数レジスタロード（戻り値タプルの `.0`／`.1`／`.2` が
  ロード元アドレスの offset `+0`／`+4`／`+8` の f32 レーンへ連続対応するという AArch64
  NEON のレーン順序セマンティクス）にも依存する。この前提はコンパイル可否検証（実測済み・
  上記参照）とは別種であり本セッションでは未検証だが、
  `compute_b_laneq_matches_compute_bit_exact` は前提が誤っていれば必ず失敗する構成（既定
  `kernel_with_ldc` の出力との `assert_eq!` bit 比較）のため、実機実行前に見落としが
  読み手へ伝播するリスクは小さい。
- **既定切り替えの fail-closed 方針**: C タイル転置という劣化要因を新規に抱えるため、
  #559 の `kernel_12x8` と同じ前例に従い、まず変種カーネルとして併設し、既定ディスパッチ
  （`NeonKernel` → `kernel_with_ldc`）は実機で非劣化＋改善が確認できるまで切り替えない。
- **A/B 対称性**: `Neon12x8Kernel::run_with_ldc` がヒープ確保するデフォルト実装に頼って
  いた非対称（#559 で指摘・#748 では設計時点で回避）を踏襲しないよう、`NeonBLaneqKernel`
  は最初から `kernel_b_laneq_with_ldc` への strided 直接委譲とした。
- **複数レジスタロードの導入**: `vld1q_f32_x2`／`vld1q_f32_x3` は stable rustc
  （本環境: rustc 1.96.0）で aarch64-unknown-linux-gnu 向けにコンパイル可能（本セッション
  で `cargo build --release --target aarch64-unknown-linux-gnu` の成功により実測確認済み）。
  A 側 8 行 = x2 ロード 1 命令、B 側 12 列 = x3 ロード 1 命令へ集約した。命令数収支
  （ロード・FMA 数・レジスタ収支）は既定カーネルと同一であり、効果の有無は実機計測でしか
  確定できない（#748 実装計画の事前調査結論を踏襲）。

## 検証済み事項（本セッションで実施）

| 検証 | コマンド | 結果 |
|---|---|---|
| フォーマット | `cargo fmt --all` | 差分なし |
| lint（x86_64） | `cargo clippy --workspace --all-targets --all-features -- -D warnings` | 警告なし |
| NEON クロス型検査（テスト込み） | `cargo clippy -p backend-cpu --all-targets --target aarch64-unknown-linux-gnu -- -D warnings` | 警告なし |
| NEON クロスビルド（release） | `cargo build -p backend-cpu --release --target aarch64-unknown-linux-gnu --lib` | 成功（`vld1q_f32_x2`／`vld1q_f32_x3` を含むコード全体がコンパイル可能であることを実測確認） |
| x86_64 リグレッション | `cargo test -p backend-cpu` | 全 pass |
| workspace テスト | `cargo test --workspace` | 全 pass |

`compute_b_laneq_matches_compute_bit_exact`（本イシューの最重要ローカル検証。既定
`kernel_with_ldc` との `assert_eq!` bit 比較）を含む `neon` モジュール内の新設テストは
`std::arch::aarch64` intrinsics に依存する `#[cfg(target_arch = "aarch64")]` 限定
モジュールに属するため、コンパイル自体が aarch64 ターゲット（実機または `qemu-aarch64`
エミュレーション）を要する。本環境は x86_64 かつ `qemu-aarch64` 不在（上記ゲート判定）の
ため、cfg ゲートの一時解除では回避できず（`std::arch::aarch64` は x86_64 ターゲットの
コンパイラには存在しない）、**実行できていない**。クロス `cargo build --release
--target aarch64-unknown-linux-gnu` の成功はコンパイル可能性のみを保証し、実行時の
正しさ・数値一致は未検証のままである。

**未実施（環境制約）**: レジスタスピルの静的検査（#561 記録で用いた
`llvm-objdump --disassemble-symbols` による逆アセンブル）は、本開発環境に `llvm-objdump`
が存在せず、`objdump`（binutils）も aarch64 アーキテクチャの逆アセンブルに対応していない
（`objdump: can't disassemble for architecture UNKNOWN!` で実測確認）ため実施できなかった。
`llvm-objdump` が利用可能な環境（後続セッション）で実施することを推奨する。

## 未実測（fail-closed・後続セッションへの引き継ぎ事項）

以下は受け入れ基準だが、本セッションでは aarch64 実機に到達できないため未実施:

1. **bit 完全一致（実機）**: 以下いずれも aarch64 実機（M4 Max または Grace CPU）で
   実行し pass を確認する必要がある（`gemm_blis_parity` 全件は既定 `NeonKernel` 経路のみを
   検証するため `NeonBLaneqKernel` には影響しない）。検証対象は「設計判断の要点」節で
   述べた FMA 乗数可換性に加え、複数レジスタロードのレーン順序セマンティクス（前提が
   誤っていれば下記テストは必ず失敗する）の 2 点:
   - `cargo test -p backend-cpu --release --lib -- neon_8x12_and_12x8_match_scalar_forced_bit_exact`
     （`NeonBLaneqKernel` の ScalarKernel 強制経路との bit 完全一致を含む拡張後版）
   - `cargo test -p backend-cpu --lib -- compute_b_laneq_matches_compute_bit_exact
     kernel_b_laneq_matches_hand_computed_subset
     kernel_b_laneq_with_larger_ldc_matches_tight_packing_and_preserves_gap
     kernel_b_laneq_rejects_ldc_smaller_than_nr`（`neon.rs` 内の新設ユニットテスト。
     とりわけ `compute_b_laneq_matches_compute_bit_exact` は本イシューの理論的根拠
     ─FMA 乗数可換性─の実機一次検証であり優先度が高い）
2. **効果ベンチ（スループット比較）**: `#[ignore]` テスト
   `neon_8x12_vs_b_laneq_ab_median_throughput`（`--ignored --nocapture`・5 回計測中央値・
   計測順交互化済み）を `--release` 実行し、512/1024/2048/4096 平方形状の中央値スループット
   を比較する。再現コマンド:

   ```bash
   cargo test -p backend-cpu --release --lib \
     -- --ignored neon_8x12_vs_b_laneq_ab_median_throughput --nocapture
   ```

   良化・悪化いずれであっても実測値をそのまま記録する（**placeholder 値・捏造値での
   達成扱いは行わない**）。**改善確認時のみ** `NeonKernel::run_with_ldc` の委譲先を
   `kernel_b_laneq_with_ldc` へ切り替え（`crates/backend-cpu/src/gemm_blis/microkernel.rs`
   の 1 行変更）、`gemm_blis_parity` を実機で再実行し、既定経路の 5 回中央値を再計測する。
   劣化時は既定切り替えを行わず変種併設のまま計測結果を記録する（安全側。#748 実装計画
   §2 の fail-closed 方針）。
3. **レジスタスピル静的検査**: 上記「検証済み事項」注記のとおり `llvm-objdump` が利用可能な
   環境で `cargo build -p backend-cpu --release --target aarch64-unknown-linux-gnu` の成果物
   を `compute_b_laneq`／`kernel_b_laneq` シンボルについて逆アセンブルし、`[sp` 参照
   （ベクタレジスタのスタック退避）の有無を確認する（#561 と同一手法）。

上記 3 項目は aarch64 実機（および `llvm-objdump` 環境）へアクセス可能な後続セッション・
Agent（`bench-runner` 委譲想定）が引き継いで実施する。

## スコープ外（out-of-scope-tracking 対象候補）

- 既存既定カーネル（A レーン参照版）自体への `vld1q_f32_x2` 適用（本イシューは変種側での
  適用可否確認まで。効果があれば別イシュー提案）
- C タイル転置の `vzip`/`vtrn` ベクトル化（実測で入口/出口コストが支配的と判明した場合の
  追加最適化）
- 実機未達時の実測完遂（本ドキュメントの「未実測」節に従い後続セッションで実施）
