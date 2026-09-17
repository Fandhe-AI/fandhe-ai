# CPU GEMM: Arm SME `fmopa` マイクロカーネルの形状しきい値付き NEON 切替（イシュー #1587）

## 1. 背景・目的

Mac（Apple Silicon）の CPU GEMM は Accelerate（AMX。非公開 ISA）に対して劣後しており、
「完全自作コア」方針のまま追いつける唯一の公開 ISA 拡張が Arm SME
（Scalable Matrix Extension）の `fmopa`（非拡張 FP32 外積累積）である
（`docs/perf/lowlayer-diagnosis-2026-09-12.md` §2・§7「A-5 SME＝Mac 限定の答え」）。
GB10（DGX Spark GB10）の CPU（Grace）は SME 非対応であり、CPU GEMM は既に
比較対象全てに勝っているため、本変更は GB10 側では「非後退の確認」のみが目的となる。

## 2. 一次ソース確認（Arm ARM）

- **FMOPA（非拡張 FP32）**: Arm Architecture Reference Manual（DDI0487・SME 拡張は
  DDI0616 系）は `FMOPA <ZAda>.S, <Pn>/M, <Pm>/M, <Zn>.S, <Zm>.S` を
  「各要素 `ZAda[i][j] = fma(Zn[i], Zm[j], ZAda[i][j])`（単一丸めの
  fused multiply-add・IEEE 754 準拠）」と定義する。乗算オペランド順の
  可換性は有限値に限り成立する（IEEE 754-2008 §5.4.1）。
- **SMSTART/SMSTOP と FPCR**: SMSTART／SMSTOP は Streaming SVE モードへの
  遷移・ZA ストレージの有効化のみを行い、`FPCR`（丸めモード・FZ 等）は
  変更しない（SME はスカラー FP 挙動に影響を与えない設計）。本実装は
  既定の round-to-nearest-even・非 FZ モードを前提とする（実測は §5 R3(b)
  の非正規化数系列で確認済み）。
- **レジスタ制約**: `mova <Zd>.S, <Pg>/M, <ZAda>h.S[<Ws>, #imm]` のスライス
  index レジスタは w12〜w15 に限定される。`[<Xn>, #imm, MUL VL]` の
  イミディエートオフセットは実行時 SVL（Streaming Vector Length）で
  スケールされる。

未確認事項（NaN 選択規則の実装依存性）は既存 `neon::compute_b_laneq`
（#748）と同じ理由でリスクとして残し、R3(b) の非正規化数系列テストを
必須の代替検証としている。

## 3. 実装

### 3.1 検出（fail-closed。`crates/backend-cpu/src/sme_detect.rs`）

- `std::arch::is_aarch64_feature_detected!("sme")` は本リポジトリの stable
  rustc（1.96 系）では `stdarch_aarch64_feature_detection` unstable のため
  使用不可（E0658。計画セッションで実測確認済み）。
- macOS: `sysctl -n hw.optional.arm.FEAT_SME`・`hw.optional.arm.SME_F32F32`
  が両方 `1` のときのみ候補（`crate::thread_limit::read_sysctl` を再利用。
  新規子プロセス起動経路を増やさない）。
- Linux: `/proc/cpuinfo` 最初の `Features` 行が `sme`・`smef32f32` を
  両方含むときのみ候補。
- 上記が真の場合に限り `rdsvl`（読み取り専用命令。`options(nomem, nostack,
  preserves_flags)`）で SVL を実測し、**64 バイト（512 bit）のときのみ**
  `SmeReport::kernel_enabled = true`。
- **キャッシュ範囲（`SmeReport` 全体ではなく OS フラグのみ）**: OS フラグ
  （CPU モデルに紐づく静的性質）のみ `OnceLock` にプロセス全体キャッシュ
  する（`Isa::detect` と同型）。**SVL は `sme_report()` を呼ぶたびに
  `rdsvl` で毎回読み直し**、`SmeKernel::run`／`run_with_ldc` が実際に
  `fmopa` を発行するスレッド自身で毎回呼び直す契約とする（Linux では
  `prctl(PR_SME_SET_VL)` によりスレッドごとに SVL が異なりうるため。
  `rdsvl` はメモリアクセスを伴わない読み取り専用の 1 命令でありキャッシュ
  しなくても計測に有意な影響を与えない）。環境変数による上書き機構は
  設けない（既存方針。OWASP A03）。
- **保留中 lazy ZA save の検査（TPIDR2_EL0）**: 上記に加え、実行スレッド
  自身の `TPIDR2_EL0`（`mrs S3_3_C13_C0_5`。AAPCS64／ACLE の private-ZA
  関数契約）が非ゼロでない（呼び出し元が ZA を dormant のまま呼んで
  いない）ことも毎回確認する。非ゼロの場合は SME 対応・SVL 一致でも
  「実行不可」として扱いフォールバックへ倒す（本節末尾「フォール
  バック契約」参照）。この確認も `rdsvl` と同様キャッシュしない
  （codex-review P0 指摘対応。`crates/backend-cpu/src/gemm_blis/microkernel/sme.rs::has_pending_lazy_za_save`）。

### 3.2 マイクロカーネル（`crates/backend-cpu/src/gemm_blis/microkernel/sme.rs`）

- **MR=16×NR=16・ZA0 単一タイル**（SVL=512 bit＝f32 16 要素）を採用。
  当初案の MR=32×NR=32（ZA0〜ZA3 の 4 タイル）は `MAX_TILE`（256 要素）
  拡大を要し GB10 の NEON 端タイル経路へ副作用が及ぶため、計画のリスク
  §10「フォールバック案」に従い MR=16×NR=16（`MR*NR=256=MAX_TILE` で
  ちょうど収まる）へ縮小した。`MAX_TILE` は不変。
- **bit 完全一致契約の核**: ループ前に C の現在値を ZA0 へプリロードし
  （`ld1w` → `mova za0h.s[wN,#0]` を 16 行ぶん）、`p` を昇順に走査して
  `fmopa` を `kc_len` 回発行した後、ZA0 を C へストアし直す（`mova` →
  `st1w`）。この方式により各 `c[i][j]` は「初期値 = 呼び出し時点の
  `c[i][j]`、p 昇順に 1 回ずつ `fma(a[p][i], b[p][j], acc)`」となり、
  NEON `vfmaq_laneq_f32(acc, b, a, lane)` = `acc + b*a[lane]` と乗算の
  可換性を除いて演算列が完全に同一になる。
- **フォールバック契約（codex-review P0 指摘対応）**: `SmeKernel::run`／
  `run_with_ldc` は `unsafe { compute(...) }`（`smstart`／`fmopa`／`mova`
  で ZA0 に触れる）を呼ぶ前に、実行スレッド自身で (a) SVL=64 バイト一致・
  (b) `TPIDR2_EL0 == 0`（保留中 lazy ZA save が無い）の両方を確認する
  （`current_thread_capable`）。(a)(b) いずれかが不成立の場合は
  `panic!` ではなく `scalar_fallback_kernel`／`scalar_fallback_with_ldc`
  （`compute` と同一の演算列を安全な Rust で再現し、有限値入力で bit
  完全一致する）へ切り替え、`compute` 自体を一切呼ばない（＝ZA に
  一切触れない）。TPIDR2_EL0 の確認を怠ると、呼び出し元（さらに上位の
  private-ZA 関数）が ZA を dormant のまま本関数を呼んだ場合に
  `smstart`／`mova` が呼び出し元の未保存 ZA を破壊しうる
  （`crates/backend-cpu/src/gemm_blis/microkernel/sme.rs::has_pending_lazy_za_save`
  doc 参照）。
  **本契約は `SmeKernel::run`／`run_with_ldc` 経由に限らず、公開 unsafe
  入口 `sme::kernel_unchecked_with_ldc`／`sme::kernel_unchecked` 自身も
  対象とする**（codex-review P0 再指摘対応。`PRRT_kwDOTuUCJc6h0ZMD` の
  追加指摘）。これらは `# Safety` 契約が SME 対応・SVL 一致のみを要求し
  `TPIDR2_EL0` の事前確認を呼び出し元の努力目標にとどめる設計では、
  契約を字面どおり満たした外部からの直接呼び出しでも保留中の lazy ZA
  save を破壊しうる。そのため両関数は自身の内部で
  `has_pending_lazy_za_save()` を検査し、非ゼロ（保留中）であれば
  `compute` を一切呼ばず同じパック形状の `scalar_fallback`（bit 完全
  一致）へ切り替える。呼び出し元は `TPIDR2_EL0` の事前確認を保証する
  必要がなくなり（`# Safety` 契約が「非ゼロ時は自動フォールバックする」
  へ変更）、`SmeKernel::run`／`run_with_ldc` 経由の `current_thread_capable`
  確認とは独立に、この 2 入口自身が fail-closed を担保する二重の防御と
  なる。
- **長さ・境界検査の型付きエラー化（codex-review P1 再指摘対応）**:
  公開 unsafe 入口 `sme::kernel_unchecked`（従来シグネチャ後方互換
  ラッパー）と安全な公開フォールバック入口 `sme::scalar_fallback_kernel`
  は、以前は `ap`／`bp`／`c` の長さ契約違反を `assert!`／`assert_eq!` で
  検出していたため、`SmeKernel::run` を経由せずこれらを直接呼び出す
  外部コードへ panic が漏れ得た（AGENTS.md「本番経路の panic 禁止」）。
  現在は `sme::kernel_unchecked` は `sme::kernel_unchecked_with_ldc`
  （既に `check_panel_lengths`／`check_c_tile_bounds` を経て
  `TileBoundsError` を返す）へ `ldc = NR` で委譲し、
  `sme::scalar_fallback_kernel` も同様に `sme::scalar_fallback_with_ldc`
  へ委譲する形へ変更した。境界検査そのもの（REQ-8 境界検査規約）は
  維持し、検出結果を `Result<(), TileBoundsError>` として返す点のみが
  変わる。トレイト `Microkernel::run`（`()` を返す必須メソッド。#691
  レビューにより非破壊のため `Result` 化不可）から呼ばれる場合は、
  呼び出し元契約（`gemm_blis_region` が常に正しい長さで呼ぶ）により
  `Err` は実際には到達しないため、`SmeKernel::run` はこの `Result` を
  明示的に破棄する（`let _ = ...`）。非実機で実行できる回帰テスト
  （`kernel_unchecked_returns_err_instead_of_panicking_on_ap_length_mismatch`
  等。`crates/backend-cpu/src/gemm_blis/microkernel/sme.rs` 末尾の
  `mod tests`）で、長さ不一致時に panic ではなく `Result::Err` が
  返ることを確認済み（境界検査は SME 命令発行より前に完了するため、
  SME 非対応環境でも安全に呼び出せる）。
- **保留中 lazy ZA save の後始末（テスト専用検証コード。
  `crates/backend-cpu/src/gemm_blis/microkernel/sme.rs::PendingLazySaveGuard`）
  を単一 asm ブロック内へ集約（codex-review P0 再指摘対応。
  「ストリーミングモードを同じ asm ブロック内で解除する」）**: 以前は
  ZA0 読み出し（`smstart sm` を伴う asm）を抜けた後、通常の Rust
  メソッド呼び出し（`finish_pending_save`。別 asm で `msr TPIDR2_EL0,
  xzr` → `smstop`）を挟んでいたため、PSTATE.SM=1 のまま関数呼び出し・
  条件分岐というコンパイラ生成コードを実行する区間が生じ、Arm ACLE の
  asm 制約（各 asm が呼び出し時点の PSTATE.SM を保存する）に違反し
  うる状態だった。現在は `PendingLazySaveGuard::read_za0_and_finish_streaming`
  が「`smstart sm` → ZA0 読み出し → `msr TPIDR2_EL0, xzr` → `smstop`」
  を単一 asm ブロック内で完結させ、asm を抜けた時点で必ず
  PSTATE.SM=0（かつ PSTATE.ZA=0）に戻す。panic 時の `Drop::drop` は
  別経路の `finish_pending_save`（dormant のまま panic した場合専用。
  `smstart` を発行しないため PSTATE.SM の遷移自体を伴わない）を呼び、
  両者は `cleaned_up` フラグで二重発行を防ぐ。あわせて、
  `smstart`/`smstop` を発行する全 asm ブロック（`compute`・
  `enter_dormant_za_with_pending_lazy_save`・
  `read_za0_and_finish_streaming`・`finish_pending_save`）に
  Z0-Z31／P0-P15 に加え FFR のクロバー宣言（`out("v0") _ ...
  out("p15") _, out("ffr") _`）を揃えた（モード切替で不定化される
  レジスタの宣言網羅性。Arm ACLE の asm 制約が要求する事項）。
- `asm!` は `compute` 1 箇所に局所化。SAFETY コメントに以下を明記:
  `smstart`/`smstop` を 1 ブロック内で対にする・v0〜v31／p0〜p15 全列挙・
  w12 明示・`preserves_flags` を付けない（`subs`/`cmp` 使用）・`nomem`/
  `pure` を付けない（`ld1w`/`st1w` 使用）・`options(nostack)`。ポインタ
  演算（`ldc_bytes`）は asm 外の Rust 側で確定し、境界検査
  （`check_panel_lengths`／`check_c_tile_bounds`。REQ-8）を経てから
  `compute` へ渡す。
- `SmeKernel`（`Avx2Kernel` と同型の「検出済みトークンのみ構築可能」
  パターン）: `try_new()` は `crate::sme_detect::sme_report().kernel_enabled`
  が `true` のときのみ `Some` を返す。

### 3.3 ディスパッチ結線（`crates/backend-cpu/src/gemm_blis/mod.rs`）

- 単一 const ゲート `SME_PRODUCTION_ENABLED`（既定 `false`。
  `TWO_D_DYNAMIC_PRODUCTION_ENABLED` と同型のロールバック機構）。
- 純関数 `sme_shape_eligible(m_total, n, k) -> bool`（`SME_MIN_M=256`・
  `SME_MIN_N=256`・`SME_MIN_K=64`。§5.4.1 R4 参照）。
- aarch64 版 `dispatch_two_d_dynamic`（`gemm_blis_parallel_with_transpose`・
  `gemm_blis_bias_act_parallel` の共有本番入口）に
  `if SME_PRODUCTION_ENABLED && sme_shape_eligible(...) && let Some(kernel)
  = SmeKernel::try_new() { … }` を追加し、それ以外は従来どおり
  `NeonKernel`。`SME_PRODUCTION_ENABLED=false` の間は実行 CPU が SME に
  対応していても常に NEON が選ばれ、#1313 以前と bit 完全一致する
  （`sme_production_enabled_is_false_pending_measurement` がドリフト検出）。
- `Isa` enum への variant 追加は行わない（SME 検出は `Isa::detect()` とは
  独立の軸であり、`Isa::Sme` を追加すると crates.io 公開クレート
  `fandhe-ai-backend-cpu` の `pub enum` への semver 破壊になる一方、
  ディスパッチ判定には不要なため）。
- **到達範囲**: `gemm_blis_parallel_nt`／`_tn`（VJP 転置入口・#1213）・
  `gemm_blis_bias_act_parallel`（epilogue 融合）はいずれも
  `dispatch_two_d_dynamic` を経由するため、`SME_PRODUCTION_ENABLED=true`
  かつ形状条件を満たせば自動的に SME へ到達する（grep で確認済み）。

### 3.4 A/B 計測ハーネス（`#[cfg(test)]`）

- `GemmDriverVariant::TwoDDynamicSme`（aarch64 限定。`all_gemm_driver_variants()`
  には含めない — SME 非対応環境で `.expect` panic するため）。
- `sme_vs_neon_ab_shape_sweep`（`#[ignore]`）: NEON／SME を interleave
  方式で比較する round-robin ハーネス（§5.2 の生データ取得元）。

## 4. 数値契約（R3。必須・実機実測済み）

Apple M4 Max 実機（本セッション機。`sysctl hw.optional.arm.FEAT_SME=1`・
`SME_F32F32=1`）で以下を確認した:

| 検証 | 内容 | 結果 |
|------|------|------|
| R3(a) run-to-run | 同一入力を 2 回実行し bit 同一 | PASS（`sme_kernel_is_deterministic_across_runs`） |
| R3(b) 有限値・cross-kernel | SME vs scalar 参照（p 昇順 `f32::mul_add` 連鎖）。kc_len ∈ {0,1,3,4,17,255,256}・端タイル（ldc>NR） | PASS（`sme_kernel_matches_scalar_reference_finite_values`・`sme_kernel_with_ldc_matches_scalar_reference_strided`） |
| R3(b) 非正規化数 | 全要素 1e-40（非正規化数）入力 | PASS（`sme_kernel_matches_scalar_reference_denormal_values`） |
| R3(b) NaN | NaN 混入入力（bit 一致は要求しない） | panic なしを確認（`sme_kernel_nan_input_does_not_panic`） |
| R3(c) 本番入口（`SME_PRODUCTION_ENABLED=true` を一時的に有効化して確認） | `gemm_blis_parallel_matches_naive_bit_exact_across_thread_pools`（m=523・n=600・k=700。5-loop ドライバ全体） | PASS |
| R3(c) 本番全体 | `SME_PRODUCTION_ENABLED=true` で `cargo test -p fandhe-ai-backend-cpu --lib` 全体（421 件） | 全件 PASS（既存契約への回帰なし） |
| R3(c) 5-loop 全体（本番形状帰属） | `gemm_blis_parallel_variant_sme_matches_naive_bit_exact_when_available`（しきい値ちょうど・端タイル・MC/KC/NC 境界跨ぎ形状 × スレッド数 1/2/4） | PASS |

いずれも `cargo test -p fandhe-ai-backend-cpu --lib`（stable rustc 1.96.0・
`aarch64-apple-darwin`）で実機実行済み。`cargo build --lib --target
aarch64-apple-darwin`（codegen を伴う）でアセンブラ検証済み・`cargo check
--target x86_64-unknown-linux-gnu -p fandhe-ai-backend-cpu` で SME モジュールが
非対象アーキテクチャで警告ゼロにコンパイル対象外になることを確認済み。

## 5. 性能実測

**§5.4 で Apple M4 Max の正式実測（R1/R2/R4。イシュー #1978）を追記した。
以下 §5.1〜§5.3 は初回 PR 時点の記述（参考計測のみ）をそのまま残す。**

### 5.1 事前登録規則（issue #1587 コメント。実装着手前に固定）

正式版は GitHub issue #1587 のコメント
（`https://github.com/Fandhe-AI/fandhe-ai/issues/1587#issuecomment-5648848287`）
を正とする。要旨:

- R1（非後退）: framework-compare gemm/train/infer cpu を両実機 5 run。
  到達セルは中央値 `<=1.00` かつ 5/5 run 一貫で ADOPT 候補。
- R2（checksum）: 全セル完全一致。
- R4（しきい値）: マイクロ A/B で「しきい値以上の全格子点で SME≥NEON が
  5/5 run 一貫」を満たす最小の組を採用。
- **本セッションの制約（事前宣言済み）**: R1/R2 の完全な 5-run
  framework-compare campaign（両実機）は所要時間の制約により本セッション
  では実施しない可能性が高く、その場合 `SME_PRODUCTION_ENABLED=false`
  （R5: undetermined → false）で出荷すると明記していた。

### 5.2 実測結果（初回 PR 時点の参考計測。正式実測は §5.4）

**R3（数値契約）は §4 のとおり完全実施・全 PASS。**

**R4（しきい値スイープ）・R1/R2（本番性能・非後退）は、事前宣言どおり
本セッションの時間制約により正式な 5-run 独立プロセス起動プロトコルを
実施できなかった。** 代わりに `sme_vs_neon_ab_shape_sweep`（interleave
round-robin・各候補 10 反復の中央値。同一プロセス内の単発計測 2 回。
Apple M4 Max・共有負荷下）による**参考（非正式）計測**を実施した:

Run 1:

| 形状 (m,n,k) | NEON GFLOP/s | SME GFLOP/s | SME/NEON |
|---|---|---|---|
| (64,64,32) | 4.543 | 7.316 | 1.61 |
| (128,128,64) | 42.836 | 37.645 | 0.88 |
| (256,256,64)（=SME_MIN） | 130.056 | 133.064 | 1.02 |
| (256,256,128) | 180.563 | 236.023 | 1.31 |
| (512,512,256) | 472.737 | 714.239 | 1.51 |
| (1024,1024,512) | 852.909 | 1561.239 | 1.83 |
| (2048,2048,1024) | 885.895 | 2109.469 | 2.38 |

Run 2（同一プロセス再実行）:

| 形状 (m,n,k) | NEON GFLOP/s | SME GFLOP/s | SME/NEON |
|---|---|---|---|
| (64,64,32) | 3.051 | 5.300 | 1.74 |
| (128,128,64) | 32.346 | 32.535 | 1.01 |
| (256,256,64)（=SME_MIN） | 105.573 | 118.426 | 1.12 |
| (256,256,128) | 177.068 | 190.471 | 1.08 |
| (512,512,256) | 476.937 | 774.147 | 1.62 |
| (1024,1024,512) | 888.552 | 1648.634 | 1.86 |
| (2048,2048,1024) | 915.849 | 2096.213 | 2.29 |

2 回とも `(m,n,k) >= (SME_MIN_M, SME_MIN_N, SME_MIN_K) = (256,256,64)`
の全形状で SME が NEON 以上（1.01〜2.38 倍）だった。`(128,128,64)`
（しきい値未満）は run1 で SME が劣後（0.88 倍）し run2 でほぼ同等
（1.01 倍）と非単調で、しきい値未満領域を対象外とする現行設計と整合する。

**この参考計測は事前登録した正式プロトコル（両実機・5 プロセス独立起動・
中央値・checksum 完全一致・非到達セルの非対称規則）を満たしていない
（単発・単一実機・同一プロセス内・GB10 未実施）ため、R1/R2/R4 の正式
採否判定には使わない。** 数値自体は ADOPT 方向を強く示唆するが、事前
宣言した R5「事後緩和なし」の原則に従い、本 PR では

```
SME_PRODUCTION_ENABLED = false
```

のまま出荷する（コード・テスト・機構は維持。§5.3 の再開手順で正式化する）。

### 5.3 再開条件・引き継ぎ（初回 PR 時点の記述）

正式な R1/R2/R4 実測（`docs/perf/logs/cpu-gemm-sme-fmopa-1587/` の
オーケストレーションスクリプト参照）を両実機（Apple M4 Max・DGX Spark
GB10）で完了し、事前登録規則を満たした場合のみ
`SME_PRODUCTION_ENABLED = true` へ切り替える。GB10 は
`sme_report().kernel_enabled == false` になる想定のため（Grace CPU は
SME 非対応）、GB10 側の役割は「本番経路への副作用がないことの確認
（既存 bit 一致テスト群・framework-compare の非後退）」に限られる。

**Apple M4 Max 側の正式実測は §5.4（イシュー #1978）で完了した。**
本節が条件としていた「両実機で完了」は字義どおりには満たしていない
（GB10 側の非後退確認は本セッションの対象外。RULE.txt が
「GB10 は SME 非対応のため本セッションの対象外」と明記して範囲を
Apple M4 Max 限定へ絞ったため）。`SME_PRODUCTION_ENABLED` の本番切替・
しきい値の確定は、GB10 側の要否判断を含め引き続き #1979 のユーザー
承認事項とする。

### 5.4 正式実測（Apple M4 Max・イシュー #1978。#1587 事前登録 R1/R2/R4）

一次ソースは issue #1587 の事前登録コメントで、
`docs/perf/logs/cpu-gemm-sme-fmopa-1587/RULE.txt`（固定日時
2026-09-17T16:32:14Z）はそれを緩めない運用規則のみを足したもの。
要旨は §5.1 と同じ（R4 のしきい値格子スイープ→採用候補確定→R1 の
到達セル・非到達セルの非後退確認→R2 の checksum 確認→総合判定）。
before 腕は main（`a1c50f61`）、after 腕は `SME_PRODUCTION_ENABLED`
のみ `true` へ反転した計測専用 worktree（`docs/perf/logs/
cpu-gemm-sme-fmopa-1587/on-arm.patch`。main 側の同定数は §3.3 のとおり
`false` のまま不変）。

#### 5.4.1 R4（しきい値格子。16 点 × 5 プロセス独立起動。2026-09-17T16:35Z 台）

`sme_vs_neon_ab_r4_grid`（正方 m=n・ratio = SME/NEON の median_gflops 比）。

| min(m,n) | k | run ごとの比 | 中央値 | 5/5 run で SME>=NEON |
|---:|---:|---|---:|---|
| 64 | 32 | 1.395, 1.262, 1.416, 1.426, 1.482 | 1.416 | yes |
| 64 | 64 | 1.364, 1.098, 0.981, 1.253, 0.894 | 1.098 | no |
| 64 | 128 | 1.276, 1.464, 1.132, 1.071, 1.011 | 1.132 | yes |
| 64 | 256 | 1.104, 1.533, 1.292, 1.405, 1.921 | 1.405 | yes |
| 128 | 32 | 1.033, 0.866, 0.935, 0.847, 0.940 | 0.935 | no |
| 128 | 64 | 1.055, 0.857, 0.700, 0.971, 0.956 | 0.956 | no |
| 128 | 128 | 0.982, 0.966, 1.018, 1.043, 1.018 | 1.018 | no |
| 128 | 256 | 1.330, 1.139, 0.959, 1.125, 1.202 | 1.139 | no |
| 256 | 32 | 0.792, 1.142, 1.027, 1.321, 0.910 | 1.027 | no |
| 256 | 64 | 0.988, 0.959, 1.518, 1.161, 1.161 | 1.161 | no |
| 256 | 128 | 1.267, 1.320, 1.826, 1.371, 1.184 | 1.320 | yes |
| 256 | 256 | 1.249, 1.291, 1.068, 1.358, 1.396 | 1.291 | yes |
| 512 | 32 | 1.096, 1.057, 1.427, 1.161, 1.384 | 1.161 | yes |
| 512 | 64 | 1.522, 1.310, 1.385, 1.306, 1.236 | 1.310 | yes |
| 512 | 128 | 1.489, 1.363, 1.642, 1.355, 1.291 | 1.363 | yes |
| 512 | 256 | 1.659, 1.429, 1.218, 1.616, 1.729 | 1.616 | yes |

R4 採用候補（しきい値以上の全格子点が 5/5 run で SME>=NEON となる
極小の組。`aggregate.py` の機械判定）:

- `min(m,n) >= 256` かつ `k >= 128`
- `min(m,n) >= 512` かつ `k >= 32`

**現行定数 `SME_MIN_M/N=256`・`SME_MIN_K=64` が含む格子点 `(256, 64)`
は本集計では 5/5 run 一貫ではない**（run 比 0.988, 0.959, 1.518, 1.161,
1.161・中央値 1.161。5 run 中 2 run が 1.0 未満）。§5.2 の参考計測
（同じ `(256,256,64)` 形状で 1.02／1.12 と 2 回とも SME≥NEON）とは
異なる結果であり、正式な 5 プロセス独立起動・16 格子点スイープでは
現行しきい値の境界点が「5/5 run で SME>=NEON」を満たさないことが
判明した。しきい値自体は本 PR では変更しない（本節は集計のみ・
判定基準の変更は行わない）。

#### 5.4.2 R1（framework-compare 非後退。5 round・起動順反転。2026-09-17T16:36Z 台）

到達セル（gemm cpu。m=n=k の正方形状。単位 us）:

| size/mode | before median | after median | after/before | checksum | 判定 | run 内比（5 run） |
|---|---|---|---|---|---|---|
| 512/fresh | 654.8 us | 483.2 us | 0.7378 | 完全一致 | 非後退 | 0.7352, 0.7514, 0.7507, 0.6922, 0.7476 |
| 512/reuse | 651.7 us | 479.8 us | 0.7363 | 完全一致 | 非後退 | 0.7897, 0.7234, 0.7321, 0.7165, 0.7326 |
| 1024/fresh | 2.935 ms | 1.762 ms | 0.6005 | 完全一致 | 非後退 | 0.6005, 0.6009, 0.6137, 0.5784, 0.5917 |
| 1024/reuse | 3.311 ms | 2.070 ms | 0.6251 | 完全一致 | 非後退 | 0.6209, 0.6027, 0.6521, 0.6743, 0.6161 |
| 2048/fresh | 17.918 ms | 10.258 ms | 0.5725 | 完全一致 | 非後退 | 0.5672, 0.5827, 0.5746, 0.5690, 0.5763 |
| 2048/reuse | 20.533 ms | 11.953 ms | 0.5821 | 完全一致 | 非後退 | 0.6026, 0.5736, 0.5653, 0.5977, 0.5762 |

到達セル 6 個は全て「5 round 中央値 <=1.00（実際は 0.57〜0.74 倍へ
大幅改善）」かつ「run 内比 5/5 とも <=1.00」で RULE.txt の ADOPT 候補
条件を満たす（最大値でも 512/reuse の 0.7897）。gemm cpu の到達形状
（m=n=k=512/1024/2048）はいずれも `min(m,n)=512 以上・k=512 以上` に
相当し、§5.4.1 の R4 採用候補 2 組（`min>=256 かつ k>=128`・
`min>=512 かつ k>=32`）のどちらでも SME 到達の対象に含まれるため、
本 R1 結果は §5.4.1 が特定した `(256, 64)` 境界点の判定不成立とは
独立で（しきい値の選び方に依存しない）成立している。

**正誤（PR #2016 レビュー指摘による再分類）**: RULE.txt は train／infer
size=64 を一括で「SME 非到達セル」と事前宣言したが、これは形状分析を
記録せずに分類した誤りである。after 腕（`on-arm.patch` で
`SME_PRODUCTION_ENABLED=true`）の現行しきい値 `SME_MIN_M/N/K=256/256/64`
で、train の第 1 層 weight 勾配 GEMM が `sme_shape_eligible` を満たし
SME へディスパッチされる（fresh は `matmul_vjp` → `CpuBackendOps::gemm`
→ `gemm_blis_parallel_tn`、reuse は `fill_resident_weight_grad` →
`gemm_fp32_strict_into` → 同じ `gemm_blis_parallel_tn` →
`dispatch_two_d_dynamic` → `sme_shape_eligible(784, 256, 64)`）。
MNIST 規模 train（784→256→10・batch 64）の GEMM を層別に分解すると:

| GEMM | m | n | k | `sme_shape_eligible`（256/256/64） |
|---|---|---|---|---|
| L1 forward `x @ W1` | 64 | 256 | 784 | 否（m<256） |
| L1 d_weight `x^T @ g`（TN） | 784 | 256 | 64 | **可** |
| L1 d_input（非計算・葉） | – | – | – | – |
| L2 forward `h @ W2` | 64 | 10 | 256 | 否（m, n<256） |
| L2 d_weight `h^T @ g`（TN） | 256 | 10 | 64 | 否（n<256） |
| L2 d_input `g @ W2^T`（NT） | 64 | 256 | 10 | 否（m<256・k<64） |
| infer forward（fresh／reuse とも L1・L2） | 64 | 256／10 | 784／256 | 否（m<256） |

したがって **train fresh／reuse は到達セル**（L1 d_weight のみ SME）、
infer fresh／reuse は非到達セルである。RULE.txt 自体は事前宣言の一次
記録として書き換えず、以下の表と §5.4.4 の判定はこの再分類に RULE.txt
の各規則（到達セル: 5 round 中央値 <=1.00 かつ 5/5 round <=1.00 で
ADOPT 候補。非到達セル: 5/5 一貫の後退のみ REJECT 材料）を適用する。
なお L1 d_weight の形状 (784, 256, 64) は k=64 の境界点にあり、§5.4.1 で
`(256, 64)` 格子点が 5/5 run 一貫しなかった事実と機構的に整合する
（仮説。#1979 のしきい値判断への申し送り材料）。

train／infer size=64（単位 us。train は到達セル・infer は非到達セル）:

| task/size/mode | before median | after median | after/before | checksum | run 内比（5 run） | 5/5 一貫の後退 |
|---|---|---|---|---|---|---|
| train 64/fresh | 808.8 us | 732.8 us | 0.9061 | 完全一致 | 0.8660, 0.8747, 0.8929, 0.9473, 0.8983 | いいえ |
| train 64/reuse | 901.6 us | 891.9 us | 0.9892 | 完全一致 | 1.0244, 0.9385, 0.9892, 0.8909, 1.0248 | いいえ |
| infer 64/fresh | 180.7 us | 173.3 us | 0.9592 | 完全一致 | 0.9050, 1.0375, 0.9753, 0.9076, 0.9749 | いいえ |
| infer 64/reuse | 177.7 us | 156.1 us | 0.8787 | 完全一致 | 0.8349, 1.1102, 0.8187, 1.1041, 0.7438 | いいえ |

到達セルとして扱う train 64/fresh は中央値 0.9061 かつ 5/5 round
<=1.00 で ADOPT 候補条件を満たす。train 64/reuse は中央値 0.9892 で
<=1.00 だが round 1・5 が 1.0244／1.0248 と 1.00 を超えるため ADOPT
候補条件（5/5 round <=1.00）を満たさない。一方 5/5 一貫の後退（全 run
>1.00）でもないため REJECT 材料にも該当しない。非到達セルの infer
64/fresh（1.0375）・infer 64/reuse（1.1102, 1.1041）は 1.0 超の run を
含むが 5/5 一貫ではなく REJECT 材料に該当しない（SME 非到達のため
機構上の因果はなくノイズ帯と整合）。

#### 5.4.3 R2（checksum）

R1 の gemm・train・infer 全 10 セルで `compare_gemm_ab.py
--require-checksum-exact` が「完全一致」と判定（§5.4.2 表の
checksum 列）。

#### 5.4.4 総合判定

RULE.txt の総合規則（「R4 に採用候補があり・R1 の到達セルが全て
ADOPT 候補・非到達セルに REJECT 材料がなく・R2 が全セル一致のとき
ADOPT。到達セルに 5/5 一貫の後退が 1 つでもあれば REJECT。それ以外は
undetermined」）に、§5.4.2 の再分類（train fresh／reuse は到達セル）を
当てると、到達セル 8 個のうち train 64/reuse が ADOPT 候補条件を満たさず
（round 1・5 が 1.0244／1.0248）、かつ 5/5 一貫の後退はないため、本
セッションの総合判定は **undetermined** である。本 PR の初版は RULE.txt
の一括分類のまま ADOPT と記録していたが、PR #2016 のレビュー指摘
（train の L1 d_weight GEMM が SME 到達）を受けて再分類した結果であり、
判定規則自体は緩めていない（ADOPT → undetermined の厳格化方向）。
gemm cpu 到達セル 6 個は 0.5725〜0.7378 倍の改善で 5/5 一貫しており、
train 64/fresh も ADOPT 候補条件を満たすため、SME の性能効果自体は
本実測で否定されていない。RULE.txt・#1587 の合意どおり、本判定は
`SME_PRODUCTION_ENABLED` を実際に切り替えるものではない。§5.4.1 で
判明した「現行しきい値の境界点 `(256, 64)` は 5/5 run 一貫ではない」
という事実から、しきい値を `min(m,n)>=256 かつ k>=128` または
`min(m,n)>=512 かつ k>=32` へ変更する案（後者は `SME_MIN_K` を
64→32 に下げる一方 `SME_MIN_M/N` を 256→512 に上げる組）が R4 の
実測から導かれる候補になる。本番切替の可否・しきい値の確定値は
#1979 のユーザー承認事項へ申し送る（§5.3 参照。GB10 側の非後退確認
の要否も含む）。

## 6. セキュリティ考慮（OWASP Top 10）

- **unsafe の統制**: `asm!` は `compute`／`has_pending_lazy_za_save`
  （`sme.rs`）と `rdsvl_bytes`（`sme_detect.rs`）の 3 箇所に局所化。
  各所に SAFETY コメントで契約を明記（§3.2・§3.1）。
- **A03**: 検出は固定引数の子プロセス（`sysctl`。`env_clear()`）と
  `/proc/cpuinfo` 読み取りのみ。環境変数による ISA 上書き機構は設けない。
- **A08**: 数値契約は bit 一致を維持（tolerance 変更なし）。fail-closed
  （検出不能・SVL≠64・OS フラグ欠落は `dispatch_two_d_dynamic` 側の
  ディスパッチ判定〈`sme_shape_eligible`／`SmeKernel::try_new`〉により
  すべて NEON。SVL 不一致・`TPIDR2_EL0` 非ゼロ〈保留中 lazy ZA save〉は
  `SmeKernel::try_new()` 自体は成功した後で `run`／`run_with_ldc` が
  実行スレッド自身で再確認し、`compute` を呼ばずスカラーフォール
  バックへ切り替える。§3.2「フォールバック契約」参照）。
- **REQ-8**: `check_panel_lengths`／`check_c_tile_bounds` を asm 呼び出し
  前に必ず通す（最適化を理由に境界検査を省略しない）。

## 7. スコープ外

- SME2／f16（`SME_F32F32` 以外）・int8 経路、SVL≠512 対応、MC/NC の SME
  向け再チューニング、E コア／P コア親和性制御、`gemm_blis`（直列入口）
  の SME 対応、32×32（4 タイル）版。既存 `out-of-scope-tracking.md`
  規約に従い、承認後に別 issue で追跡する。
