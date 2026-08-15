# aarch64 プリフェッチ intrinsics の到達可能性調査（#489・A-9）

イシュー #489「spike(backend-cpu): aarch64 プリフェッチ intrinsics の到達可能性調査（終了条件つき）」に対応する。GEMM 性能改善ツリー（#479）Phase A の spike（A-9）であり、BLIS の armv8a GEMM カーネルが多用する `PRFM PLDL1KEEP` 相当の明示プリフェッチを本リポの前提（`rust-toolchain.toml` の stable channel 固定・許容依存 8 区分のみ）の Rust から発行できるかを確定し、Phase E の E-7（#562「明示プリフェッチの導入（A-9 の結論に従う）」）の実装方式判断に引き継ぐための spike ドキュメント。

## 判断サマリ

**stable rustc 上で `core::arch::aarch64` の安定版プリフェッチ intrinsic は存在しない。** `_prefetch`（および `_PREFETCH_READ`／`_PREFETCH_LOCALITY3` 等の locality/rw 定数）は `core::arch::aarch64` に定義自体は存在するが `unstable`（feature gate: `stdarch_aarch64_prefetch`、tracking issue [rust-lang/rust#117217](https://github.com/rust-lang/rust/issues/117217)）であり、stable channel では `E0658` でコンパイル不能である（実測は後述）。

stable 前提で `PRFM` 相当命令を発行する経路は **inline asm（`asm!("prfm pldl1keep, [{0}]", ...)`）以外に存在しない**。inline asm は aarch64 では Rust 1.59 以降 stable だが、`unsafe` inline asm の新規導入となり `.claude/rules/coding-rust.md`「コード品質」節の「`unsafe` は FFI 境界等の必要最小限に留め、理由をコメントで明記しレビュー必須」という方針との整合上、**E-7（#562）の実装可否・採否は本 spike の範囲外とし、ユーザー承認事項として保留する**。

本 spike はこの分岐（「存在しない場合」）に該当した時点で完了である（イシュー #489 受け入れ基準 2 項）。E-7 本体への Issue コメント・ラベル変更等の操作は行わない（`.claude/rules/out-of-scope-tracking.md`: ユーザー承認なしの Issue 操作をしない）。

## 調査方法・実測根拠

### 環境

- toolchain: `rust-toolchain.toml` の単一真実源どおり `channel = "stable"`
- 実測 rustc バージョン: `rustc 1.96.0 (ac68faa20 2026-05-25)`（`rustc --version` 出力）
- 対象 target: `aarch64-unknown-linux-gnu`（DGX Spark GB10 相当の Linux/aarch64）・`aarch64-apple-darwin`（Metal 実機・Apple Silicon 相当）。両 target とも `rustup target list --installed` で導入済みであることを確認したうえでプローブした
- プローブファイルはリポジトリにコミットしていない使い捨てソース（scratchpad 上で作成・破棄）

### プローブソース

```rust
#![allow(dead_code)]

#[cfg(target_arch = "aarch64")]
unsafe fn probe(ptr: *const f32) {
    use core::arch::aarch64::{_prefetch, _PREFETCH_LOCALITY3, _PREFETCH_READ};
    _prefetch(ptr as *const i8, _PREFETCH_READ, _PREFETCH_LOCALITY3);
}
```

### 実行コマンドと結果（出力は抜粋。エラー 6 件のうち 1 件目のみ掲載）

```console
$ rustc --edition 2021 --crate-type lib --target aarch64-unknown-linux-gnu --emit=metadata -o /tmp/probe_linux.rmeta prefetch_probe.rs
error[E0658]: use of unstable library feature `stdarch_aarch64_prefetch`
 --> prefetch_probe.rs:6:33
  |
6 |     _prefetch(ptr as *const i8, _PREFETCH_READ, _PREFETCH_LOCALITY3);
  |                                 ^^^^^^^^^^^^^^
  |
  = note: see issue #117217 <https://github.com/rust-lang/rust/issues/117217> for more information
error: aborting due to 6 previous errors
```

`aarch64-apple-darwin` target でも同一ソースに対し同一の `E0658`（feature `stdarch_aarch64_prefetch`、tracking issue #117217）を確認した（`_prefetch` 関数本体・`_PREFETCH_READ`・`_PREFETCH_LOCALITY3` の 3 シンボルいずれも unstable 判定）。両 target で結果は一致しており、target 差による分岐はない。

再現手順: 上記プローブソースを任意のファイルに保存し、上記コマンドの `--target` を差し替えて実行すれば同じ `E0658` が再現する。

### 実施日

2026-08-14（本ドキュメント作成時点）。

## 代替経路の評価表

| 経路 | 評価 | 出典・不可理由 |
|------|------|----------------|
| `core::arch::aarch64::_prefetch` | 不可（stable 不可） | 上記実測。`E0658`・feature `stdarch_aarch64_prefetch`・tracking issue rust-lang/rust#117217 |
| `core::intrinsics::prefetch_read_data` 等のコンパイラ内部 intrinsic | 不可（stable 不可） | `core_intrinsics` feature が必要で nightly 限定。`#![feature(...)]` は stable channel では使用不能 |
| `std::hint` 系（`std::hint::black_box` 等） | 該当なし | プリフェッチ相当の安定 API が存在しない |
| nightly channel への変更 | 不採用 | `rust-toolchain.toml` は CI 共通基盤（rust-base-ci reusable workflow）が参照する単一真実源で stable 固定（`.claude/rules/ci.md`「ベースライン品質ゲート」節）。self-repair 検証ゲートの clippy component 依存（PR #344 実測）を含め stable 前提を崩す変更は影響範囲が大きく本 spike の範囲外 |
| 外部クレート（プリフェッチ機能を提供するもの） | 不採用 | 許容依存 8 区分（`.claude/rules/deps-policy.md`）外。新規依存追加はユーザー承認必須であり本 spike では判断しない |
| inline asm（`asm!("prfm pldl1keep, [{0}]", ...)`） | 技術的に唯一の到達手段。**採否はユーザー承認事項として保留** | aarch64 では inline asm は Rust 1.59 以降 stable だが、`unsafe` の新規導入となり `.claude/rules/coding-rust.md`「`unsafe` は FFI 境界等の必要最小限に留め、理由をコメントで明記しレビュー必須」との整合確認が要る |

## E-7（#562）への引き継ぎ事項

以下は「存在する場合」分岐の必須要件ではなく、E-7 実装可否のユーザー承認判断に資する**参考情報**として記す。

### (a) inline asm 導入時の unsafe 境界の置き方

既存の NEON マイクロカーネル（`crates/backend-cpu/src/gemm_blis/microkernel/neon.rs`）は `MR=8`／`NR=8`（同ファイル 17〜20 行目）で、`kernel` 関数入口で `assert_eq!` によりスライス長を検査したうえで `unsafe` ブロックへ入り、SAFETY コメントでオフセット範囲がその検査済み長さを超えないことを明記する構成を採る（同ファイル 37〜49 行目。REQ-8 境界検査規約に対応）。inline asm でプリフェッチを追加する場合もこのパターンとの整合、すなわちプリフェッチ対象アドレスの範囲を関数入口の検査で確定させたうえで `asm!` を呼ぶ設計が候補になる。`asm!` マクロ自体はコンパイラの検証が及ばない領域であるため、`.claude/rules/coding-rust.md` の unsafe 最小化・理由コメント必須の方針に従い、プリフェッチ専用の薄いラッパー関数へ `unsafe` を局所化することが望ましい（PRFM はヒント命令でありメモリ内容・制御フローに影響しない設計だが、無効アドレスに対する挙動の厳密な保証範囲は本 spike では ARM アーキテクチャリファレンスマニュアルで確認しておらず、E-7 承認判断時に別途確認が要る）。

### (b) BLIS 実プリフェッチ距離（参考値。イシュー #489 本文記載の数値を未検証のまま転記）

以下はイシュー #489 本文に記載されていた BLIS armv8a GEMM カーネルのプリフェッチ距離（参考値）であり、**本 spike では BLIS ソース自体での再確認は行っていない**。イシュー本文は非信頼データであるため、E-7 実装時に BLIS 公式ソース（`bli_gemm_armv8a_asm_d6x8.c` 等）で再検証したうえで採用の可否を判断すること。

- B パネル: プロローグ 192 / 256 / 320 バイト先、ループ本体 336 / 400 / 464 バイト先
- A パネル: 128 / 192 バイト先、224 / 288 バイト先
- C（書き戻し列）: `PLDL1KEEP`

本リポの f32 MR=8×NR=8 マイクロカーネル（`crates/backend-cpu/src/gemm_blis/microkernel/neon.rs`）へ換算する場合は、上記距離（再検証後の値）をキャッシュライン単位（64B）に丸めた上でマイクロカーネルのループ展開幅・レジスタブロッキング構成に合わせて再導出する必要がある（BLIS のマイクロカーネル形状 MR×NR とは一致しないため単純比例換算はできない）。この換算作業自体は E-7 承認後の実装ステップで行う。

## 共通契約の充足

本 spike はドキュメント追加のみであり、`crates/` 配下・CI・設定を変更していない。

- **境界チェック不省略（REQ-8）**: コード変更なし。該当なし
- **tolerance 不緩和**: コード変更なし。該当なし
- **依存不追加**: `Cargo.toml`／`Cargo.lock` 不変。外部クレート経路は評価表で不採用と結論済み
- **`docs/spec/` 不編集**: 正本 submodule に触れていない
- **REQ-8 段階的下限不変更**: `docs/performance-targets.md` 等の性能下限に触れていない

## 出典

- イシュー #489（本ドキュメントの起票元）・#479（GEMM 性能改善ツリー・Phase A）・#562（E-7。本判断の引き継ぎ先）
- `.claude/rules/coding-rust.md`「コード品質」節（unsafe 最小化方針）
- `.claude/rules/ci.md`「ベースライン品質ゲート」節（`rust-toolchain.toml` 単一真実源・stable 固定の理由）
- `.claude/rules/deps-policy.md`（許容依存 8 区分・新規依存はユーザー承認必須）
- `.claude/rules/out-of-scope-tracking.md`（ユーザー承認なしの Issue 操作をしない）
- rust-lang/rust#117217（`stdarch_aarch64_prefetch` tracking issue）
