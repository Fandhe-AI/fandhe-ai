# aarch64 プリフェッチ intrinsics の到達可能性調査（#489・A-9）

イシュー #489「spike(backend-cpu): aarch64 プリフェッチ intrinsics の到達可能性調査（終了条件つき）」に対応する。GEMM 性能改善ツリー（#479）Phase A の spike（A-9）であり、BLIS の armv8a GEMM カーネルが多用する `PRFM PLDL1KEEP` 相当の明示プリフェッチを本リポの前提（`rust-toolchain.toml` の stable channel 固定・許容依存 8 区分のみ）の Rust から発行できるかを確定し、Phase E の E-7（#562「明示プリフェッチの導入（A-9 の結論に従う）」）の実装方式判断に引き継ぐための spike ドキュメント。

## 判断サマリ

**stable rustc 上で `core::arch::aarch64` の安定版プリフェッチ intrinsic は存在しない。** `_prefetch`（および `_PREFETCH_READ`／`_PREFETCH_LOCALITY3` 等の locality/rw 定数）は `core::arch::aarch64` に定義自体は存在するが `unstable`（feature gate: `stdarch_aarch64_prefetch`、tracking issue [rust-lang/rust#117217](https://github.com/rust-lang/rust/issues/117217)）であり、stable channel では `E0658` でコンパイル不能である（実測は後述）。

stable 前提で `PRFM` 相当命令を発行する経路は、関数内 inline asm（`asm!("prfm pldl1keep, [{0}]", ...)`）とモジュールスコープの `core::arch::global_asm!` の 2 経路が技術的に到達可能である（後述の評価表参照）。ただし `global_asm!` はマイクロカーネルのホットループ内で計算済みポインタへ直接発行する用途には適合しない（評価表・下記参照）ため、**本リポのユースケース（ループ内の動的アドレスへのプリフェッチ）における実質的な到達手段は inline asm（`asm!`）である**。inline asm は aarch64 では Rust 1.59 以降 stable だが、`unsafe` inline asm の新規導入となり `.claude/rules/coding-rust.md`「コード品質」節の「`unsafe` は FFI 境界等の必要最小限に留め、理由をコメントで明記しレビュー必須」という方針との整合上、**E-7（#562）の実装可否・採否は本 spike の範囲外とし、ユーザー承認事項として保留する**。

本 spike はこの分岐（「存在しない場合」）に該当した時点で完了である（イシュー #489 受け入れ基準 2 項）。E-7 本体への Issue コメント・ラベル変更等の操作は行わない（`.claude/rules/out-of-scope-tracking.md`: ユーザー承認なしの Issue 操作をしない）。

**2026-08-19 追補**: 上記の「ユーザー承認待ちの保留」は、OSS 一次ソース（BLIS armv8a・matrixmultiply）の照合結果を踏まえ「原則不要」へ格下げした。結論の詳細は末尾「## 2026-08-19 追補: 保留判断の格下げ」節を参照。

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
| inline asm（`asm!("prfm pldl1keep, [{0}]", ...)`） | 到達可能。本ユースケース（ホットループ内の動的アドレスへのプリフェッチ）における実質的な採用候補。**採否はユーザー承認事項として保留** | aarch64 では inline asm は Rust 1.59 以降 stable だが、`unsafe` の新規導入となり `.claude/rules/coding-rust.md`「`unsafe` は FFI 境界等の必要最小限に留め、理由をコメントで明記しレビュー必須」との整合確認が要る |
| モジュールスコープ asm（`core::arch::global_asm!("prfm pldl1keep, [x0]" ...)`） | 到達可能だが本ユースケースには不適合 | `global_asm!` は `asm!` と同じ Rust 1.59 で stable 化された aarch64 対応マクロだが、意味論が異なる: 関数本体の中には書けず、モジュールスコープに独立したアセンブリ項目（フリースタンディングなシンボル・関数）を直接定義するものであり、`in`/`out` オペランドで Rust のローカル変数（レジスタ割付されたポインタ値等）を asm へ渡す機能を持たない。マイクロカーネルのループ内で計算済みの動的アドレスへ `PRFM` を発行するには、`global_asm!` でプリフェッチ専用の `extern "C"` 関数を別途定義し、ループ側からその関数を FFI 境界越しに呼び出す構成が必要になる。これは (1) 呼び出し規約（AAPCS64）に従った関数呼び出しオーバーヘッドをホットパス命令ごとに発生させ、(2) インライン化をコンパイラの裁量から外し、(3) `unsafe extern "C"` の FFI 境界と手動シンボル管理を新規に持ち込む点で、関数内 `asm!` を直接インラインで発行する経路より不利であり、性能目的のプリフェッチ挿入という本ユースケースには適合しない。`unsafe` 統制の観点でも `asm!` 1 箇所への局所化（引き継ぎ事項 (a) 参照）に比べ `unsafe extern "C"` 宣言・呼び出し元双方に `unsafe` 境界が増える分、`.claude/rules/coding-rust.md` の unsafe 最小化方針との整合が悪化する。保守性の観点では、モジュールスコープの生アセンブリ関数はマイクロカーネル本体（Rust コード）と物理的に分離されるため、MR/NR 変更等マイクロカーネル形状の変更時に追随漏れが起きやすい |

## E-7（#562）への引き継ぎ事項

以下は「存在する場合」分岐の必須要件ではなく、E-7 実装可否のユーザー承認判断に資する**参考情報**として記す。

### (a) inline asm 導入時の unsafe 境界の置き方

既存の NEON マイクロカーネル（`crates/backend-cpu/src/gemm_blis/microkernel/neon.rs`）は既定で `MR=8`／`NR=12`（同ファイル 85・87 行目。イシュー #559 で旧実装〈#552〉の `MR=8`／`NR=8` から拡張済み。本ドキュメント下記「E-7（#562）着手判断」節も同カーネル形状で記述する）で、`kernel` 関数入口（114〜117 行目）で `assert_eq!` によりスライス長を検査したうえで `unsafe` ブロック（136 行目〜）へ入り、SAFETY コメント（119〜127 行目）でオフセット範囲がその検査済み長さを超えないことを明記する構成を採る（REQ-8 境界検査規約に対応）。同カーネルは k=4 アンロール（イシュー #561。同ファイル 128〜181 行目）で主ループを 4 p-ステップ単位のチャンクとして処理しており、1 チャンク（k=4 ステップ）あたりの packed パネル読み出し幅はキャッシュライン単位換算の起点として次のとおりである: B が `NR(=12) × 4 ステップ × 4B = 48 f32 = 192B`、A が `MR(=8) × 4 ステップ × 4B = 32 f32 = 128B`（下記「E-7（#562）着手判断」節の距離初期値・再導出で用いる換算値。以下このドキュメント内で「§『(a)』節参照」とする箇所はこの換算値を指す）。inline asm でプリフェッチを追加する場合もこのパターンとの整合、すなわちプリフェッチ対象アドレスの範囲を関数入口の検査で確定させたうえで `asm!` を呼ぶ設計が候補になる。`asm!` マクロ自体はコンパイラの検証が及ばない領域であるため、`.claude/rules/coding-rust.md` の unsafe 最小化・理由コメント必須の方針に従い、プリフェッチ専用の薄いラッパー関数へ `unsafe` を局所化することが望ましい（PRFM はヒント命令でありメモリ内容・制御フローに影響しない設計だが、無効アドレスに対する挙動の厳密な保証範囲は本 spike では ARM アーキテクチャリファレンスマニュアルで確認しておらず、E-7 承認判断時に別途確認が要る）。

### (b) BLIS 実プリフェッチ距離（参考値。イシュー #489 本文記載の数値を未検証のまま転記）

以下はイシュー #489 本文に記載されていた BLIS armv8a GEMM カーネルのプリフェッチ距離（参考値）であり、**本 spike では BLIS ソース自体での再確認は行っていない**。イシュー本文は非信頼データであるため、E-7 実装時に BLIS 公式ソース（`bli_gemm_armv8a_asm_d6x8.c` 等）で再検証したうえで採用の可否を判断すること。

- B パネル: プロローグ 192 / 256 / 320 バイト先、ループ本体 336 / 400 / 464 バイト先
- A パネル: 128 / 192 バイト先、224 / 288 バイト先
- C（書き戻し列）: `PLDL1KEEP`

本リポの f32 既定マイクロカーネル（`MR=8`／`NR=12`。`crates/backend-cpu/src/gemm_blis/microkernel/neon.rs` 85・87 行目。イシュー #559 で旧 `MR=8`／`NR=8` から拡張済み）へ換算する場合は、上記距離（再検証後の値）をキャッシュライン単位（64B）に丸めた上でマイクロカーネルのループ展開幅・レジスタブロッキング構成に合わせて再導出する必要がある（BLIS のマイクロカーネル形状 MR×NR とは一致しないため単純比例換算はできない）。この換算作業自体は E-7 承認後の実装ステップで行う。

## 共通契約の充足

本 spike はドキュメント追加のみであり、`crates/` 配下・CI・設定を変更していない。

- **境界チェック不省略（REQ-8）**: コード変更なし。該当なし
- **tolerance 不緩和**: コード変更なし。該当なし
- **依存不追加**: `Cargo.toml`／`Cargo.lock` 不変。外部クレート経路は評価表で不採用と結論済み
- **`docs/spec/` 不編集**: 正本 submodule に触れていない
- **REQ-8 段階的下限不変更**: `docs/performance-targets.md` 等の性能下限に触れていない

## E-7（#562）着手判断（2026-08-16）

イシュー #562「明示プリフェッチの導入（A-9 の結論に従う）」の着手時点の判断を記録する。#562・親ツリー #479 いずれもコメント 0 件（`gh issue view 562/479 --comments` 実測）であり、リポジトリオーナーによる inline asm（`asm!` による `PRFM` 発行）採用の明示承認記録は存在しない。上記「判断サマリ」節のとおり E-7 の実装可否はユーザー承認事項として保留されており、承認が確認できない自動運転実行下では **inline asm 導入を見送る**。本追記はコード変更を伴わない判断記録であり、`crates/` 配下は変更していない。E-7 の実装系受け入れ基準（マイクロカーネルへのプリフェッチ挿入）は、承認が成立するまで構造的に未達のままとなる。

### BLIS プリフェッチ距離の一次ソース再検証

上記「(b) BLIS 実プリフェッチ距離」節は、#489 本文記載値を未検証のまま転記したものだった。本追記で BLIS 公式リポジトリ（`flame/blis`）のソースを取得し照合した。

- 参照ソース: `kernels/armv8a/3/bli_gemm_armv8a_asm_d6x8.c`（コミット `a49238e6141c96a41aa3c2a4adb0b0663d0b4968` 時点。`gh api repos/flame/blis/contents/...` で取得）
- **f32（`s` 系）カーネルは armv8a ディレクトリに存在しない**: `kernels/armv8a/3/` 配下は `bli_gemm_armv8a_asm_d6x8.c`（f64・MR=6×NR=8）・`bli_gemm_armv8a_asm_d8x6r.c`（f64）と `3/sup/`・`3/1m/` 配下の gemmsup 系（いずれも `d` = f64 プレフィックス）のみで構成され、f32 用の GEMM マイクロカーネルソースは存在しなかった（`gh api repos/flame/blis/contents/kernels/armv8a/3` の一覧で確認）。したがって以下の距離値は f64・MR=6×NR=8 カーネルのものであり、本リポの f32・MR=8×NR=12 マイクロカーネルへ適用する際は要素サイズ（8B→4B）・MR/NR 形状の両方の差を踏まえた再導出が必要（単純比例換算不可、既存記載どおり）
- 照合結果: `bli_gemm_armv8a_asm_d6x8.c` 119〜258 行目の `prfm PLDL1KEEP` オペランドを実測したところ、#489 本文由来の転記値と**完全一致**した
  - B パネル: プロローグ `x1` レジスタ起点で 192 / 256 / 320 バイト先（119・123・129 行目）、ループ本体 336 / 400 / 464 バイト先（219・221・224 行目）
  - A パネル: プロローグ `x0` レジスタ起点で 128 / 192 バイト先（141・145 行目）、ループ本体 224 / 288 バイト先（256・258 行目）
  - C（書き戻し列）: `prfm pldl1keep` を各行アドレスへ個別発行（120〜185 行目。`PLDL2KEEP` によるパネル先読みは 695・696・1047・1048 行目の別経路）
- 結論: 転記値に誤りはなかった。ただし出典が f64・MR=6×NR=8 カーネルである点は #489 本文に明記されていなかったため、本追記で明示した。E-7 承認後の実装では、この f64 カーネルの距離をキャッシュライン（64B）単位で解釈し直し、本リポの f32・MR=8×NR=12・k=4 アンロール構成（B: 48 f32 = 192B/chunk〈k=4 ステップ分。1 ステップあたりは NR=12 f32 = 48B〉、A: 32 f32 = 128B/chunk〈1 ステップあたりは MR=8 f32 = 32B〉。§「(a)」節参照）に合わせて独自に再導出する必要がある

**2026-08-19 訂正**: 上記「f32（`s` 系）カーネルは armv8a ディレクトリに存在しない」は誤りだったことが判明した。イシュー #751 での再照合で `kernels/armv8a/3/bli_gemm_armv8a_asm_d8x6r.c`（`d6x8.c` とは別ファイル）内に f32 行優先カーネル `bli_sgemm_armv8a_asm_12x8r`（MR=12×NR=8）が同居しており、firestorm config（`config/firestorm/bli_cntx_init_firestorm.c`）はこの f32 カーネルを実際に使用することを確認した。詳細・訂正後の結論は末尾「## 2026-08-19 追補: 保留判断の格下げ」節を参照。E-7 が原則不要へ格下げされたため、上記の f64 カーネル距離を f32・MR=8×NR=12 へ再導出する作業自体も当面不要である。

### PRFM の安全性（無効アドレスに対する挙動）

上記「(a)」節が保留していた「無効アドレスに対する `PRFM` の挙動」について、Arm Developer サイト（`support.arm.com` の Architecture Reference Manual 該当ページ）へのアクセスを試みたが、当該ページは JavaScript 描画のためテキスト取得ツールでは本文（Operation／Notes／CONSTRAINED UNPREDICTABLE 記述）を取得できなかった。本追記の時点でも一次ソースでの確認は取れておらず、**未確認のまま残す**（推定で断定しない）。E-7 承認判断時には ARM Architecture Reference Manual（DDI0602 系）の `PRFM` 命令ページを PDF 版等の一次ソースで直接確認することが必要。

### 実装設計案（参考・承認後にそのまま着手できる形で記す）

承認が得られた場合の実装方針は上記「E-7（#562）への引き継ぎ事項」節（(a)・(b)）に集約済みであり、追加すべき要点は以下のとおり:

- **unsafe 局所化**: `#[inline(always)] unsafe fn prefetch_read_l1(ptr: *const f32)` をプリフェッチ専用ラッパーとして新設し、`asm!("prfm pldl1keep, [{0}]", in(reg) ptr, options(readonly, nostack, preserves_flags))` を `asm!` 発行の唯一の箇所とする。SAFETY コメントには「PRFM はヒント命令」という設計意図に加え、無効アドレスに対する挙動が本ドキュメント時点で未確認である旨も明記し、既存 `assert_eq!` によるスライス長検査の範囲内に発行対象アドレスをクランプする既存 SAFETY 証明の枠内に収める（REQ-8 境界検査規約を緩めない）
- **距離初期値**: 上記で再検証した f64・MR=6×NR=8 の距離を単純採用せず、本リポの f32・MR=8×NR=12・k=4 チャンク構成（B: 192B/chunk〈1 ステップあたり 48B〉、A: 128B/chunk〈1 ステップあたり 32B〉）に合わせてキャッシュライン単位で再導出する
- **検証**: aarch64 実機（DGX Spark GB10 / Metal 実機）でのみ距離スイープの効果測定が可能なため、`#[ignore]` テストまたは `gemm_blis_perf` 系の枠組みまで実装したうえで実測自体は実機セッションへ fail-closed 引き継ぎとする（`docs/perf/cpu-gemm-neon-k4-unroll.md` と同運用）。効果が無い・悪化する場合は不採用の判断を `docs/perf` 文書へ記録する
- 委譲区分: 実装は backend-builder 相当、レビューは reviewer に加え unsafe 変更のため security-auditor 並列必須（`.claude/rules/security.md`）

## 2026-08-19 追補: 保留判断の格下げ（原則不要・C 出力直前 PLDL1KEEP のみ将来検討）

イシュー #751 の一環として、GEMM 第 2 次最適化ツリー（#735 → Phase 3 #738）向けの OSS 技法ギャップ分析で BLIS armv8a・matrixmultiply の一次ソースを再照合した。結果、上記「判断サマリ」節の「ユーザー承認待ちの保留」を **「原則不要（HW ストリームプリフェッチャー任せ）」へ格下げ**する。

### 一次ソース照合結果

- **BLIS**（`flame/blis` master `061c2ebef87eda9189e6cdf38af4ea3d4a8efe7b` 時点。
  `kernels/armv8a/3/bli_gemm_armv8a_asm_d8x6r.c`、blob SHA `81d558a9482f57bf673c8aa41a7804703d010067`。
  `gh api repos/flame/blis/contents/...` で取得・実測）:
  - 同ファイルは f32 行優先カーネル `bli_sgemm_armv8a_asm_12x8r`（135 行目〜）と
    f64 カーネル `bli_dgemm_armv8a_asm_8x6r` を同居させている。firestorm config
    （`config/firestorm/bli_cntx_init_firestorm.c` 52〜53 行目・実測）が `BLIS_GEMM_UKR` へ
    両者を束縛しており、Apple M1 系（firestorm）では f32 は 12x8r カーネルが使われる
  - 378 行目のコメント「Stream HW prefetcher is assumed s.t. PRFM instructions for packed
    A&B are omitted.」のとおり、**packed A/B パネルへの k ループ内 PRFM は省略**されている
    （実測: 129〜132 行目・179〜216 行目の `PRFMC_FWD` マクロは C 出力タイルの
    `prfm PLDL1KEEP` のみを発行。281〜289 行目に次パネル先頭へ `prfm PLDL1STRM` を数発
    発行する「to try to activate hardware prefetcher」コメント付きコードがあるが、A/B の
    k ループ内主経路には PRFM が存在しない）
  - **旧記載の訂正**: 本ドキュメント「(b) BLIS 実プリフェッチ距離」節および
    「BLIS プリフェッチ距離の一次ソース再検証」節の「f32（`s` 系）カーネルは armv8a
    ディレクトリに存在しない」は誤りである。`bli_gemm_armv8a_asm_d8x6r.c` 内に
    `bli_sgemm_armv8a_asm_12x8r`（f32・MR=12×NR=8）が同居しており、firestorm config は
    この f32 カーネルを使用する。旧記載が参照した `bli_gemm_armv8a_asm_d6x8.c`
    （f64・MR=6×NR=8）は firestorm では未使用の旧来カーネルである
- **matrixmultiply**（`bluss/matrixmultiply` master `07f968c20e0df41e3fd694e9c47ef5368ec9eee1`。
  `gh api repos/bluss/matrixmultiply/contents/src/sgemm_kernel.rs` の内容に対し
  `grep -ci "prefetch\|prfm"` を実行し `0` を確認。`src/aarch64/`・`src/archparam_kernels/`
  配下含め aarch64 向けカーネル一式に `prefetch`／`prfm` の記述は見当たらない）:
  aarch64 では明示プリフェッチを一切使わない構成である
- **本リポの実測整合**: 2026-08-19 M4 Max 計測で MR=8×NR=12 現行カーネル（プリフェッチなし）
  が matrixmultiply 比 1.3〜2.6 倍（イシュー #735「棚卸しで実施した整理」節記載）であり、
  A/B パネルへのプリフェッチ欠如が律速している兆候はない

### 判断

E-7（#562・closed）の inline asm による packed A/B プリフェッチ導入は **原則不要**とし、
ユーザー承認待ちの保留項目から外す。将来検討に残すのは **C タイル書き戻し直前の
`PLDL1KEEP`（BLIS `PRFMC_FWD` 相当）のみ**であり、着手条件は以下の両方を満たすこと:

1. M4 Max／Grace 実機で C ストア律速の証拠（ストア待ちがボトルネックであること）が計測で
   示される
2. `unsafe asm!` の追加は「実装設計案」節の枠組み（1 箇所局所化・SAFETY コメント）を踏襲し、
   security-auditor 並列レビューを実施する

着手条件が成立した場合は新規 Issue を起票し、ユーザー承認を得たうえで再開する
（`.claude/rules/out-of-scope-tracking.md`）。上記「E-7（#562）への引き継ぎ事項」節・
「実装設計案」節は履歴として保持し、C 出力直前プリフェッチを将来検討する際の参考資料と
位置づける（packed A/B へのプリフェッチ導入という当初想定の用途では既に不採用が確定した
ため、そのままでは適用しない）。

本追補はドキュメント変更のみであり `crates/` 配下・CI・設定は変更していない。REQ-8 下限値・
数値一致許容誤差も変更していない。

## 出典

- イシュー #489（本ドキュメントの起票元）・#479（GEMM 性能改善ツリー・Phase A）・#562（E-7。本判断の引き継ぎ先）
- イシュー #751（本追補の起票元）・#735（GEMM 第 2 次最適化ツリー起点）・#738（Phase 3 CPU）
- flame/blis `kernels/armv8a/3/bli_gemm_armv8a_asm_d8x6r.c`（master `061c2ebef87eda9189e6cdf38af4ea3d4a8efe7b`、blob `81d558a9482f57bf673c8aa41a7804703d010067`）・`config/firestorm/bli_cntx_init_firestorm.c`（同 master）
- bluss/matrixmultiply `src/sgemm_kernel.rs` ほか（master `07f968c20e0df41e3fd694e9c47ef5368ec9eee1`）
- `.claude/rules/coding-rust.md`「コード品質」節（unsafe 最小化方針）
- `.claude/rules/ci.md`「ベースライン品質ゲート」節（`rust-toolchain.toml` 単一真実源・stable 固定の理由）
- `.claude/rules/deps-policy.md`（許容依存 8 区分・新規依存はユーザー承認必須）
- `.claude/rules/out-of-scope-tracking.md`（ユーザー承認なしの Issue 操作をしない）
- rust-lang/rust#117217（`stdarch_aarch64_prefetch` tracking issue）
- flame/blis `kernels/armv8a/3/bli_gemm_armv8a_asm_d6x8.c`（コミット `a49238e6141c96a41aa3c2a4adb0b0663d0b4968`。BLIS プリフェッチ距離の一次ソース再検証根拠）
- [The Rust Reference: Inline assembly](https://doc.rust-lang.org/reference/inline-assembly.html)（`asm!`／`global_asm!` の stable 化・aarch64 対応・意味論の違いの根拠）
