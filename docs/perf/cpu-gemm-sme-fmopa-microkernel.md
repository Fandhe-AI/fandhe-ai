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

- 単一 const ゲート `SME_PRODUCTION_ENABLED`（既定 `false`〈#3231 で一度 `true` にしたが #3232 の GB10 REJECT により `false` へ戻した。§5.14〉。
  `TWO_D_DYNAMIC_PRODUCTION_ENABLED` と同型のロールバック機構）。
- 純関数 `sme_shape_eligible(m_total, n, k) -> bool`（`SME_MIN_M=256`・
  `SME_MIN_N=256`・`SME_MIN_K=64`。§5.4.1 R4 参照）。
- aarch64 版 `dispatch_two_d_dynamic`（`gemm_blis_parallel_with_transpose`・
  `gemm_blis_bias_act_parallel` の共有本番入口）は safe 関数
  `sme_kernel_for_shape(m_total, n, k)`（`SME_PRODUCTION_ENABLED &&
  sme_shape_eligible(...)` のときだけ `SmeKernel::try_new()`）が `Some` の
  ときに SME へ、それ以外は従来どおり `NeonKernel`。SME 非搭載ホストでは
  `try_new()` が `None` となり NEON へ fall back して出力は不変。`false` に
  戻すと実行 CPU が SME に対応していても常に NEON（#1313 以前と bit 完全一致）。
  ドリフト検出は `sme_production_enabled_is_true_with_neon_fallback`
  （#3231 で `..._is_false_pending_measurement` から置換）。
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

**§5.8 に `SME_MIN_K` 候補（64／128／256）の M4 Max 再実測と GB10 非後退再確認
（イシュー #2118）の事前登録規則・実行基盤・記入欄を追加した（実測は未実施）。**

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
は §5.5 で実施済み）。

### 5.5 GB10 側の非後退確認（DGX Spark GB10・イシュー #1978 残。2026-09-18 UTC）

§5.3 が「両実機で完了」の条件として残していた GB10 側を、RULE.txt を
変更せず新規の事前登録規則 `docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/RULE-gb10.txt`
（計測前にコミット）で実施した。Grace CPU は SME 非対応のため、本節の
目的は「`SME_PRODUCTION_ENABLED=true` にした after 腕が非 SME 環境で
fail-closed に NEON へフォールバックし（R0）、ハーネスの checksum が
一致し（R2-GB10）、非後退であること（R1-GB10）」の確認に限られる。
R2-GB10 が比較する checksum は gemm／infer の全要素和・train の最終
loss を小数点以下 6 桁へ丸めた値（`bench-common::Record::to_json_line`）
であり、要素間の相殺や丸めで異なる出力でも一致しうるため、**#1979
受け入れ条件「非 SME 環境で bit 同一のフォールバック」の実証には
ならない**（bit 同一性は全出力の `to_bits()` 比較または生バイト列の
ハッシュ比較が必要。本節は未実施。PR #2048 codex-review 指摘）。Apple M4 Max の総合判定（§5.4.4 undetermined）は本節で
変更しない。

- **腕**: before = main `176f27e7`（`.rev-stamp` は RULE-gb10.txt のみ追加した
  `b127e9f2`）、after = 同一ツリーに `on-arm.patch`（sha256
  `ca493a09…`）を適用。ツリー指紋の差分は `gemm_blis/mod.rs` の 1 ファイル
  のみ・before 指紋は Mac 側 `git ls-files` 一覧の sha256 と全 1422 ファイル
  一致（`gb10/rev-stamp-verification.md`）。
- **専有ゲート**: 外側（load1 < 1.0 かつ gpu_util 0%）は build 前に
  attempt=1 で通過（load1 0.26）。内側（`run_ab_sme_cpu.sh` 固定 8.0）は
  5/5 round 通過（load1 4.43〜7.88）だが、round 1〜2 の load1 は直前の
  bench-fandhe 2 腕ビルド（20 スレッド）の残留を含む（`uptime-*.log` の
  round 2 は 8.34）。
- **R0**: `sme_report()` は before／after とも
  `SmeReport { os_flag: false, svl_bytes: None, kernel_enabled: false }`
  （`/proc/cpuinfo` Features に `sme`／`smef32f32` なし）。よって 10 セル
  すべてが SME 非到達。
- **RT（既存テスト群・after 腕 `cargo test -p fandhe-ai-backend-cpu --release`）**:
  637 pass・1 FAIL・32 ignored。FAIL は
  `gemm_blis::tests::sme_production_enabled_is_false_pending_measurement`
  （`SME_PRODUCTION_ENABLED` が `false` であることそのものを検証する定数
  ドリフトガード。after 腕は定数を反転した計測専用ツリーのため構造的に
  FAIL する）。RULE-gb10.txt は「FAIL があれば名前を記録し後退あり相当
  （要調査）」と定めており、要調査の結果はこのガード 1 件で、数値経路の
  FAIL は 0 件。規則は緩和しない（ガード除外を事前登録しなかったのは
  規則側の不備として記録）。
- **R1-GB10（`run_ab_sme_cpu.sh 1978-gb10`・5 round・起動順反転・checksum
  完全一致必須）**: 中央値比（after/before）と 5 round 内比・5/5 一貫の
  後退判定は次のとおり（`gb10/r1r2/compare-*-1978-cpu-1978-gb10.md`）。

  | セル | 中央値比 | run 内比（5 round） | 5/5 一貫の後退 |
  |---|---|---|---|
  | gemm 512/fresh | 1.0223 | 0.9367, 0.9948, 2.0070, 0.9880, 1.0223 | いいえ（round 3 に before spread > 1.5× の外れ値） |
  | gemm 512/reuse | 1.0176 | 1.1115, 1.0048, 1.1221, 0.9922, 0.9447 | いいえ |
  | gemm 1024/fresh | 0.9967 | 0.8942, 0.9863, 1.0437, 1.0500, 0.8139 | いいえ |
  | **gemm 1024/reuse** | **1.0363** | **1.0554, 1.0195, 1.0246, 1.0326, 1.0363** | **はい** |
  | gemm 2048/fresh | 0.9841 | 1.0234, 1.0247, 0.9332, 0.9371, 0.9686 | いいえ |
  | gemm 2048/reuse | 0.8968 | 0.8968, 0.9870, 0.9937, 1.0000, 0.7655 | いいえ |
  | train 64/fresh | 0.8897 | 0.8815, 0.8342, 0.9486, 0.8526, 0.8897 | いいえ |
  | train 64/reuse | 0.9346 | 0.8021, 0.8925, 1.0509, 1.0783, 0.9586 | いいえ |
  | infer 64/fresh | 1.0132 | 0.9427, 1.0132, 1.1313, 1.0241, 0.9477 | いいえ |
  | infer 64/reuse | 1.0606 | 1.0528, 1.0434, 1.0606, 1.0998, 0.8800 | いいえ |

- **R2-GB10**: 全 10 セル checksum 完全一致（`skipped-*.log` は空・
  `compare-exit` は gemm 3／train 0／infer 3＝いずれも判定結果コード）。
- **総合判定（RULE-gb10.txt の語彙）: 「GB10 非後退確認: 後退あり」**。
  根拠は gemm cpu 1024/reuse の 5/5 round 一貫の後退（1.0195〜1.0554）。
  R0・R2-GB10 は成立しているが、上記のとおり checksum 一致は bit 同一の
  証拠にならないため、#1979 受け入れ条件「非 SME 環境で bit 同一の
  フォールバック」は **checksum 一致を確認・bit 同一性は未確認** として
  #1979 へ申し送る。是正・規則の事後緩和はしない。
- **原因帰属（未検証の仮説として記録・判定には用いない）**: after 腕が
  before 腕と異なるのは `dispatch_two_d_dynamic` の
  `SME_PRODUCTION_ENABLED && sme_shape_eligible(..) && SmeKernel::try_new()`
  評価が short-circuit せずに `sme_report()`（OS フラグは `OnceLock`
  キャッシュ）まで到達する点のみで、1024/reuse だけが 5/5 一貫となる
  機構は本実測から特定できない。同じ後退方向でない他セル（1024/fresh
  0.9967・2048/reuse 0.8968）と round 内の大きな揺れ（512/fresh の 2.007）
  は共有負荷ノイズと整合するが、5/5 一貫という事実は規則どおり後退
  として扱う。分離には SME 検出コールの有無だけを変えた腕（`try_new`
  を呼ばない after′）のプロセス分離計測が必要で、本 issue の範囲外
  （→ §5.7 で実施済み。事前登録規則 (c)「未分離（ノイズ帯）。#1978 残の
  5/5 一貫は再現せず」）。
- **#1979 への申し送り**: `SME_PRODUCTION_ENABLED=true` へ切り替える場合、
  GB10 側は本節の後退 1 セルを前提に再計測（または `sme_report()` の
  結果で `SME_PRODUCTION_ENABLED` 経路をプロセス起動時に一度だけ閉じる
  構成の検討）が必要。加えて「非 SME 環境で bit 同一のフォールバック」は
  本節では checksum 一致までしか確認していないため、before／after 両腕の
  全出力 `to_bits()` 比較（または生バイト列ハッシュ）を GB10 で別途取る
  必要がある（→ §5.6 で実施済み・「bit 同一: 成立」）。

### 5.6 GB10 側の全出力 bit 同一実証（DGX Spark GB10・イシュー #2050。2026-09-19 UTC）

§5.5 が checksum 一致までしか確認していなかった「非 SME 環境で bit 同一の
フォールバック」を、#2049 で追加した `crates/facade/tests/
cpu_sme_gate_bit_dump.rs::dump_cpu_sme_gate_bits`（gemm 512/1024/2048・
train size=64〈reuse・L1 d_weight 784×256×64 が `sme_shape_eligible`
到達形状〉・infer size=64〈非到達対照〉の全出力を `to_bits()` で 1 要素
1 行印字）と `gb10/run_bitdump.sh` で直接検証した。判定規則は実測開始前に
コミットした `gb10/RULE-bitdump.txt`（登録 2026-09-19T01:30:03Z・
コミット `1f36ee11`。実測開始 01:34:08Z）。オーケストレーションは
`gb10/orchestrate_bitdump_gb10.sh`・成果物は `gb10/bitdump/`（dump 本体
`*_bits.txt`／`*_raw.log`〈各約 269 MB〉は収録しない）。

- **腕**: before = main `82160b47` 相当（ブランチ tip `daffdd14`。main との
  差は `gb10/` 配下の docs のみで `git diff --stat main..tip -- crates` は空）
  ・`SME_PRODUCTION_ENABLED=false`。after = 同一ツリーに `on-arm.patch`
  （sha256 `ca493a09…721cff`。`gb10/bitdump/patch_sha256.txt`）を適用し
  `SME_PRODUCTION_ENABLED=true`。ツリー指紋差分は
  `crates/backend-cpu/src/gemm_blis/mod.rs` の 1 件のみ
  （`gb10/bitdump/fp-diff.txt`。テストファイルは両ツリーに存在するため
  `run_bitdump.sh` のコピー手順は発火せず）。
- **専有ゲート**: attempt=1 で load1 0.43・gpu_util 0% の通過（記録のみ。
  `load_gate_outer.log`）。
- **R0（前提）**: 成立。両腕とも
  `SmeReport { os_flag: false, svl_bytes: None, kernel_enabled: false }`
  （`gb10/bitdump/sme_report.txt`）。
- **RB（bit 同一・判定）**: **成立**。before／after とも期待行数 6,726,847
  行に一致（`line_counts.txt`）・全体 sha256 `b8c399e9…a694af` が一致・
  `cmp`／`diff` の差分 0 バイト・0 行（`bitdump_cmp.txt`／
  `bitdump_diff.txt` とも空）。ラベル別（`summary.txt`）も gemm512
  （262,144 行）・gemm1024（1,048,576 行）・gemm2048（4,194,304 行）・
  train（1,221,183 行）・infer（640 行）の 5 ラベルすべて一致。
  `run_bitdump.sh` は exit 0（規則の対応表で「成立」）。
- **RR（再現性・記録）**: after 腕の 2 回目も行数 6,726,847・sha256
  `b8c399e9…a694af` で 1 回目と一致（`rerun_after_sha256.txt`。
  run-to-run 決定的）。
- **総合判定: 「bit 同一: 成立」**。非 SME 環境（GB10）では
  `SME_PRODUCTION_ENABLED` の on/off にかかわらず、SME 到達形状を含む
  全出力（GEMM 出力・3 step 分の loss／勾配／更新後パラメータ・推論出力）
  が bit 完全一致する。§5.5 の #1979 申し送り「bit 同一性は未確認」は
  本節で解消。`SME_PRODUCTION_ENABLED=false` は不変（切替は #1979 の
  ユーザー承認事項）。§5.5 の性能後退 1 セル（gemm cpu 1024/reuse）の
  原因帰属は本節の対象外で #2051（#2052／#2053）へ引き継ぐ。
- 環境: Linux 6.17.0-1031-nvidia aarch64・rustc 1.97.0・nproc 20・
  `RAYON_NUM_THREADS` 未設定（`gb10/bitdump/env_info.txt`。絶対パスは
  `<home>` へマスク・内部ホスト名は含めない）。両腕とも release ビルドは
  約 10 秒（`*_raw.summary.log` の `Finished` 行）・テスト本体は約 0.4 秒。

### 5.7 GB10 側の 1024/reuse 後退の帰属切り分け（DGX Spark GB10・イシュー #2051／#2052／#2053。2026-09-19 UTC）

§5.5 が「未検証の仮説」として残した gemm cpu 1024/reuse の 5/5 一貫
後退（1.0363）の原因を、`SmeKernel::try_new()` 到達の有無だけを変えた
第 3 の腕 after′ を挟む 2 組のプロセス分離 A/B で切り分けた。判定規則は
実測前にコミットした `gb10/RULE-attribution.txt`（#2052・PR #2056）。
成果物は `docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/attribution/`。

- **腕**: before = main `ac3e0639`（`SME_PRODUCTION_ENABLED=false`）、
  after′ = 同ツリー + `gb10/on-arm-prime.patch`（sha256 `024d4fe2…`。
  定数を `true` にしたうえで `dispatch_two_d_dynamic` の
  `SmeKernel::try_new()` を `None::<SmeKernel>` へ置換）、after = 同ツリー
  + `on-arm.patch`（sha256 `ca493a09…`。#1978 残・#2050 と同一）。
  各パッチ腕と before のツリー指紋差分は `gemm_blis/mod.rs` の 1 件のみ
  （`fp-diff-{prime,after}.txt`。オーケストレーションが実測前に assert）。
  before 指紋は Mac 側 `git archive` 展開の sha256 と全 5,598 ファイル一致
  （`attribution/fp-mac.txt`・`rev-stamp-verification.md`）。
- **専有ゲート（記録）**: 外側 load1 < 1.0 かつ gpu_util 0% を開始時
  attempt=1・組 1 attempt=2・組 2 attempt=4 で通過（`load_gate_outer.log`）。
- **R0**: 3 腕とも `SmeReport { os_flag: false, svl_bytes: None,
  kernel_enabled: false }`（`sme_report.txt`）。
- **A/B**（`run_ab_sme_cpu.sh`・5 round・起動順反転・内側 load ゲート
  8.0・両組とも rc=0・skipped 0 行・全セル checksum 完全一致）:

  | 組 | セル | after/before 中央値比 | round 内比（5 round） | 5/5 一貫 |
  |---|---|---|---|---|
  | 組 1 before vs after′ | 512/fresh | 1.0160（before spread > 1.5x） | 1.7297, 0.8976, 1.1470, 0.8557, 1.0154 | いいえ |
  | 〃 | 512/reuse | 1.0035 | 0.9713, 0.9800, 1.0035, 1.0003, 0.9730 | いいえ |
  | 〃 | 1024/fresh | 0.9908 | 0.9262, 0.9691, 0.8996, 1.0062, 1.0226 | いいえ |
  | 〃 | **1024/reuse** | 1.0171 | 0.9907, 1.0035, 0.9570, 1.0251, 1.0292 | いいえ |
  | 〃 | 2048/fresh | 1.0039 | 1.0011, 0.9776, 0.9514, 1.0344, 1.0430 | いいえ |
  | 〃 | 2048/reuse | 0.9695 | 1.0005, 1.1628, 0.9656, 1.0905, 0.9495 | いいえ |
  | 組 2 after′ vs after | 512/fresh | 1.0197 | 0.9392, 1.0270, 1.1572, 0.8938, 1.0197 | いいえ |
  | 〃 | 512/reuse | 1.0724 | 1.1077, 1.1330, 1.1191, 1.1110, 0.9470 | いいえ（4/5） |
  | 〃 | 1024/fresh | 0.9375 | 0.9678, 0.9976, 0.9914, 0.9611, 0.9021 | いいえ |
  | 〃 | **1024/reuse** | 1.0041 | 1.0532, 1.0355, 0.9669, 1.0020, 0.9719 | いいえ |
  | 〃 | 2048/fresh | 0.9430 | 0.9115, 0.9503, 0.9474, 0.9248, 0.9823 | いいえ |
  | 〃 | 2048/reuse | 0.9653 | 0.7856, 0.9539, 0.9918, 0.9653, 1.0518 | いいえ |

  参考（判定に用いない）: train size=64 fresh／reuse は組 1 で
  1.0102／0.9328・組 2 で 0.9698／1.0388、infer size=64 fresh／reuse は
  組 1 で 1.0032／1.0024・組 2 で 1.0638／0.9613。いずれも 5/5 一貫でない。
- **帰属判定（RULE-attribution.txt の 3 分岐。事後緩和なし）**: 組 1 の
  1024/reuse は 5/5 一貫の後退でない（(b) 不成立）・組 2 の 1024/reuse も
  5/5 一貫でない（(a) 不成立）ため、**(c)「未分離（ノイズ帯）。#1978 残の
  5/5 一貫は再現せず」**。対象 12 セル（2 組 × 6）に 5/5 一貫の後退セルは
  1 つもなく、組 2 の 512/reuse（4/5・中央値 1.0724）が最も後退方向に
  寄るが規則上は「証拠なし」。
- **解釈（判定ではない）**: 専有ゲート通過後の計測でも #1978 残の
  5/5 一貫（§5.5・共有負荷なし・attempt=1 通過）が再現しなかったため、
  §5.5 の後退 1 セルは `try_new` 到達コスト・定数切替のコード配置差の
  いずれにも帰属できない。§5.5 の総合判定「後退あり」は規則どおり不変
  （本節は §5.5 を書き換えない）。#1979 で `SME_PRODUCTION_ENABLED=true`
  へ切り替える際に GB10 側で要求される再計測は §5.5 の申し送りのまま
  だが、本節の結果は「同一条件で再測定すれば後退は 5/5 一貫では
  出ない」ことを示す 1 系列の証拠として併記する。§5.5 が列挙した
  `sme_report()` でプロセス起動時に経路を閉じる構成は、本節の証拠からは
  必要性を導けない（対策案の列挙は (a) 成立時のみ・本節では不要）。
  是正・`SME_PRODUCTION_ENABLED` の切替・新規 issue の起票は行わない。
- 環境: Linux 6.17.0-1031-nvidia aarch64・rustc 1.97.0・nproc 20・
  `RAYON_NUM_THREADS` 未設定（`gb10/attribution/env_info.txt`。
  `cpu_model` 行はノード上の `lscpu` 出力形式の相違で空。絶対パスは
  `<home>` へマスク・内部ホスト名は含めない）。`gate_constant.txt` の
  2 行目（`let Some(kernel) = ` の先頭一致）は AVX-512 経路の 993 行目を
  拾っており SME 経路の記録ではない（スクリプトの grep 不備。SME 経路の
  差分は `fp-diff-*.txt` と `patch_apply_*.log` で担保）。

### 5.8 SME しきい値候補の再実測（イシュー #2118。**未実測**・事前登録と実行基盤のみ）

§5.4 では R4 の採用候補が `min(m,n)>=256 かつ k>=128`／`min(m,n)>=512 かつ k>=32` の
2 組となり、現行の境界点 `(256, 64)` は 5/5 run 一貫にならなかった。train size=64 は
L1 d_weight GEMM（TN・784×256×64）が現行 `SME_MIN_K=64` で SME に到達し、その到達セル
train 64/reuse が ADOPT 候補条件を満たさず（round 1・5 が 1.00 超）総合判定は
undetermined だった。GB10 側は §5.5 で gemm 1024/reuse が 5/5 一貫の後退（「後退あり」）と
なったが、§5.7 の帰属切り分けでは再現しなかった（ノイズ帯）。#1979 の再開条件
（`SME_MIN_K` 候補ごとに到達セルが 5/5 round <=1.00 かつ checksum 一致・GB10 で既存経路の
非後退を再確認）を判定できる状態を、本節で実測前に固定する。

**状態**: 実行ホストが x86_64 Linux で M4 Max・GB10 に届かないため、**実測値・推定値は
記載しない**。イシューは open のまま（AC1・AC2 未達）。`SME_PRODUCTION_ENABLED=false`・
`SME_MIN_M/N=256`・`SME_MIN_K=64` は不変。採否と定数切替は #2119 のユーザー承認事項。
`.rs`・`Cargo.toml`・`Cargo.lock`・tolerance・baseline・`docs/spec/` は変更していない。
成果物は `docs/perf/logs/sme-threshold-remeasurement-2118/`（`RULE.txt`〈事前登録・単独コミット〉・
`on-arm-k{64,128,256}.patch`・`orchestrate_m4max.sh`・`gb10/orchestrate_gb10.sh`・
`aggregate.py`・`README.md`）。

#### 5.8.1 事前登録の要旨（正は `RULE.txt`）

- **腕**: before = main HEAD `0b25525f`、after(K) = 同一コミット + 計測専用パッチ
  （K=64 は定数反転のみ・K=128／256 は加えて `SME_MIN_K` を変更）。ツリー差分が
  `gemm_blis/mod.rs` の 1 件のみ・定数行がパッチどおりであることを実測前に assert する。
- **R4**（候補共通の 1 系列）: `sme_vs_neon_ab_r4_grid` 5 プロセス。候補 K の成立条件は
  `min(m,n) ∈ {256, 512}` × `k ∈ {32,64,128,256}, k >= K` の全格子点が 5/5 run で SME>=NEON。
- **R1**: 候補ごとに `run_ab_sme_cpu.sh`（LABEL `2118-m4max-k{K}`）。到達セルは 5 round
  中央値 <=1.00 かつ 5/5 round <=1.00 で ADOPT 候補。非到達セルは 5/5 一貫の後退のみ
  REJECT 材料。
- **R2**: 全 10 セル checksum 完全一致（丸めた集約値で bit 同一の証拠ではない）。
- **総合**: ADOPT 候補 = R4 成立 + 到達セルすべて ADOPT 候補 + 非到達セルに REJECT 材料なし
  + R2 全一致。到達セルに 5/5 一貫の後退があれば REJECT。それ以外 undetermined。
  AC1 の直接判定欄（到達セル 5/5 <=1.00 かつ checksum 一致）は独立に記録する。
- **判定不能の前提**（RULE.txt §13〜§15。判定規則・閾値は不変）: ツリー記録・`run_ab_sme_cpu.sh` と
  成果物収録の終了コード 0・全 10 セルの記録・R4 の登録 sha／実行／ログ（候補共通のため全候補に課す）・
  GB10 の R0／RT 記録のいずれかが不成立なら、FAIL／REJECT／後退あり等を出さず undetermined とする。
- **K=64 は再実測する**: 現行定数のベースラインで、#1979 再開条件 1 が名指しで求める
  対象であり、既存 undetermined の重複実験には当たらない。
- **GB10**: §5.5 と同じ R0・RT・R1-GB10・R2-GB10 を候補ごとに実施。語彙は
  `RULE-gb10.txt` を継承（pass／後退あり／undetermined）。「全セル 5/5 <=1.00」は採らない
  （§5.5 の 512/fresh round 3 = 2.007 の教訓）。RT は after 腕で定数ドリフトガード
  `sme_production_enabled_is_false_pending_measurement` の 1 件のみ想定内 FAIL と事前登録し、
  それ以外の FAIL は「後退あり相当（要調査）」とする（§5.5 で指摘された規則の不備の是正）。

#### 5.8.2 候補別の到達セル表（数値を見る前に固定）

形状根拠は `bench-fandhe/src/main.rs`（`BATCH=64`・`D_IN=784`・`D_HIDDEN=256`・`D_OUT=10`。
Linear(784→256)→ReLU→Linear(256→10)）。`sme_shape_eligible(m_total, n, k)` は
`m_total>=256 かつ n>=256 かつ k>=K`。L1 forward（64×256×784）・L2 各 GEMM は m または n が
足りず全候補で非到達、L1 d_weight（784×256×64）だけが K=64 で到達する。

| セル | K=64 | K=128 | K=256 |
|---|---|---|---|
| gemm 512／1024／2048 × fresh／reuse | 到達 | 到達 | 到達 |
| train size=64 fresh／reuse | 到達 | 非到達 | 非到達 |
| infer size=64 fresh／reuse | 非到達 | 非到達 | 非到達 |

K=128／256 で train が非到達になるのは規則上の帰結で、事後の再分類ではない。

#### 5.8.3 記入欄（すべて未実測。実機セッションが `aggregate.py` の出力を転記する）

M4 Max R4（候補共通）: 未実測。

| K | R4 成立 | R1 到達セル AC1 欄（5/5 <=1.00 かつ checksum 一致） | R2 | 総合判定 | 系列（load ゲート） |
|---:|---|---|---|---|---|
| 64 | 未実測 | 未実測 | 未実測 | 未確定（実測未実施） | 未実測 |
| 128 | 未実測 | 未実測 | 未実測 | 未確定（実測未実施） | 未実測 |
| 256 | 未実測 | 未実測 | 未実測 | 未確定（実測未実施） | 未実測 |

推奨候補（ADOPT 候補になった最小の K）: 未確定。

GB10（外側専有ゲートは記録のみ）:

| K | R0 | RT | 5/5 <=1.00 セル数（参考） | 総合判定 |
|---:|---|---|---:|---|
| 64 | 未実測 | 未実測 | 未実測 | 未確定（実測未実施） |
| 128 | 未実測 | 未実測 | 未実測 | 未確定（実測未実施） |
| 256 | 未実測 | 未実測 | 未実測 | 未確定（実測未実施） |

**verdict: 未確定（実測未実施）**。

#### 5.8.4 実機セッションへの申し送りと #2119 との境界

- 手順は `docs/perf/logs/sme-threshold-remeasurement-2118/README.md`（M4 Max: `orchestrate_m4max.sh all`、
  GB10: `gb10/orchestrate_gb10.sh 64|128|256`、集計: `aggregate.py`）。実行順序は固定（R4 → k64 → k128 → k256）で、
  run の差し替えはしない（再実行は別 LABEL）。
- 結果が出たら本節の記入欄と verdict を更新する（規則本文は変更せず訂正は追記方式）。
- 採否・`SME_MIN_K` の確定・`SME_PRODUCTION_ENABLED` の切替は #2119
  （`docs/cpu-gemm-sme-unsafe-audit.md` の条件付き再承認申請）のユーザー承認事項であり、
  本イシューでは行わない。
- Linux 側で確認済み: パッチ 3 本の適用・差分 1 件・`cargo check --lib --tests` の
  `aarch64-unknown-linux-gnu`／`aarch64-apple-darwin` 通過（コンパイル確認のみ）、
  `aggregate.py --self-test`、スクリプトの `bash -n`／`shellcheck`／`--dry-run`。

### 5.9 Mac mini（Apple M4）での A/B（イシュー #3176。実測済み・総合判定 ADOPT 候補）

§5.4 の M4 Max とは SME ユニットの構成（P クラスタに 1 基）とコア構成（P4＋E6）が異なる Mac mini で、
`SME_PRODUCTION_ENABLED` を反転した framework-compare の CPU 10 セル、R4 格子、MLP 長方形 4 形状を実測した（2026-10-21・計測コミット `1a9031537`）。
全セルで checksum 完全一致。gemm 到達 6 セルの after/before は 0.47〜0.60 倍（M4 Max は 0.57〜0.74 倍）で 5/5。
PyTorch 比（同 run 内）は gemm で 0.49〜0.74 倍から 1.03〜1.41 倍へ。現行しきい値領域（min>=256 かつ k>=64）は R4 で全点成立。
`SME_PRODUCTION_ENABLED` は #3231 で `true` へ切り替えた（§5.10）。ログは `logs/sme-macmini-3176/macmini/`、結論は
[`cpu-gemm-sme-ab-macmini.md`](cpu-gemm-sme-ab-macmini.md)。本番切替は #3230、しきい値の確定は #3231。（訂正: 本番切替は #3231。#3230 は unsafe 監査の更新と再開条件の確認〈`docs/cpu-gemm-sme-unsafe-audit.md` §12〉）

### 5.10 既定 ON とするしきい値の確定（イシュー #3231）

所有者確認（https://github.com/Fandhe-AI/fandhe-ai/issues/3231#issuecomment-6369076139・2026-10-24。`docs/cpu-gemm-sme-unsafe-audit.md` §12.8 の 3 項目）を根拠に、`SME_PRODUCTION_ENABLED` を `true` に切り替え、しきい値を
**`SME_MIN_M=256`・`SME_MIN_N=256`・`SME_MIN_K=64` で確定**した（据え置き）。

- 根拠（判定機は Mac mini〈Apple M4〉。M4 Max の値は Mac mini の値として扱わない）: R4 格子
  （`docs/perf/logs/sme-macmini-3176/macmini/r4_run{1..5}.log`・集計 `aggregate.md`・`cpu-gemm-sme-ab-macmini.md` §2）で、現行領域の全 6 点が 5/5 で SME>=NEON
  （(256,64)=1.404・(256,128)=1.607・(256,256)=1.602・(512,64)=1.641・(512,128)=1.930・(512,256)=1.983）。
  K=32 へ下げて入る (256,32)=1.122 や、M/N<=128 で入る (128,32)・(128,64)・(64,64) は 5/5 を満たさないため下げない。
  MLP 長方形 4 形状は事前登録上「記録のみ」のため判定に使わず、(64,256,784)・(784,64,256) への拡張はしない（m=64 は #3233）。
- 残留事項: M4 Max（§5.4.1）では境界点 (256,64) が 5/5 を満たさなかった（中央値 1.161・2/5 run で 1.0 未満）。
- GB10 の非後退（再開条件 2）は #3232 の後置ゲート。後退が出た場合のロールバックは `SME_PRODUCTION_ENABLED` を `false` に戻す 1 行と、
  `sme_production_enabled_is_true_with_neon_fallback` の反転のみ。
- 同時に A/B ハーネスの NEON 腕を `GemmDriverVariant::TwoDDynamicNeon`（NEON 強制・ゲート迂回）へ切り替えた
  （`TwoDDynamic` は ON 後に SME 搭載機で SME を含むため）。ラベル `NEON(TwoDDynamic)` は集計スクリプト互換で不変。
- bit 同一: `production_gemm_matches_naive_and_neon_forced_bit_exact_across_thread_counts`（NN/TN/NT/bias+ReLU・スレッド {1,2,3,4,8}）と
  プロセス単位の `RAYON_NUM_THREADS` sweep（`logs/sme-default-on-3231/`）。GB10 の全出力 bit 同一は §5.6（旧 sha）を参照。

> 注記（#3232）: 本節の既定 ON は #3232 の GB10 REJECT により **既定 OFF へ戻した**（§5.14。2026-10-24 所有者判断・RULE.txt:55）。本節は当時の記録で書き換えない。

### 5.11 job ごとの SME／NEON 混在分配の着手条件の判定（イシュー #3238。不成立・未着手で close）

`dispatch_two_d_dynamic` は領域ごとに SME か NEON のどちらか 1 種類を選ぶ。これを job ごとに混在させる分配は、
「SME が 1 スレッドで飽和し、残りのコアで回す NEON が上乗せになる」ことが #3175 の実測で確認できた場合だけ着手する条件付きの工夫である。
判定元は `docs/perf/cpu-sme-thread-scaling-macmini.md` §2・§3（Mac mini・5 run 中央値）で、新しい計測は行っていない。

| 観点 | 実測 | 条件との照合 |
|---|---|---|
| 1 スレッドで飽和しているか | (2048,2048,1024) の SME T=1 は 355.5 GFLOPS。第三者ピーク 2008 GFLOPS の 0.177 | 飽和していない |
| SME 自身がスレッドで伸びるか | (2048,…) は S(4)=2.14・S(10)=3.07、(1024,1024,512) は S(4)=2.14・S(10)=2.70。判定欄はどちらも「頭打ちなし」 | 対象セル（N=1024／2048）で SME はスレッド数に応じて伸びる |
| 結論との整合 | 「1 コアで飽和し P コアを増やしても足し算にならない」描像は現行カーネルでは観測されなかった（同 doc §3） | 前提が成立しない |
| T=4→10 の増加量 | (2048,…) は SME 760.9→1092.5（+331.6）、NEON 276.4→474.5（+198.1） | 6 スレッドを SME から NEON に回すと増加幅が小さくなる（算術による推定） |

- (256,256,128)・(512,512,256) は T=4→10 で SME が低下・停滞する（G(10)/G(4)=0.90／0.88）が、対象セルではなく、1 スレッドで飽和していない点も同じため判定は変わらない。
- 結論: **着手条件は不成立**。混在分配のゲート・実装・bit 一致テストは行わず、`dispatch_two_d_dynamic` の分配・ゲート・テストは一切変えていない。
  FMA 契約・tolerance・baseline・REQ-8 の境界検査も不変。
- 実測ログ: `docs/perf/logs/sme-scale-macmini-3175/` の `RULE.txt`（規則）・`aggregate.md`（集計）・`env_info.txt`（機体・ツリー）・
  `thread_t{1,2,4,10}_run{1..5}.log`（生ログ。無編集の `.raw.log` を併置）。
- 計測時点の注意: #3175 は `SME_PRODUCTION_ENABLED=false` の時期に診断テスト `sme_vs_neon_ab_thread_sweep` で SME 腕と NEON 腕を直接比べたもので、
  既定値に依存しない。#3231 で既定 ON にした後もカーネル（MR=NR=16・ZA0 1 タイル・micro-call ごとの `smstart`/`smstop`）は同じなので判定は有効。
- 再評価の条件: 1 スレッドで SME を飽和させる別カーネル（例: m=64 専用・ZA 4 タイル・`smstart` 1 区間の #3233／#3234 系）が本番に入り、
  #3175 と同じ RULE で対象セルの頭打ち（S(4) < 1.25）が観測された場合に限り、新しい issue として再提起する。

### 5.12 SME と NEON の異種並列 A/B の判定（イシュー #3239。前提消滅・未計測で close）

本 A/B は、#3238 の job ごとの SME／NEON 混在分配が既定 OFF のゲート付きで実装済みであることを前提にしていた。
#3238 は §5.11 のとおり着手条件が不成立のまま未実装で close されたため、比べる対象が存在しない。

- 根拠（`crates/backend-cpu/src/gemm_blis/mod.rs`）: `const ...: bool` のゲートは `TWO_D_DYNAMIC_PRODUCTION_ENABLED`（3525 行）と
  `SME_PRODUCTION_ENABLED`（3540 行）の 2 つだけで、混在分配のゲートはない。`crates/`・`scripts/` に混在分配（hetero）の実装も見当たらない。
  このため `on` 腕と `off` 腕は同一コードになり、計測しても差は出ない。

| 受け入れ条件 | 扱い | 理由 |
|---|---|---|
| 5 回計測の中央値と事前登録規則に対する判定 | 該当なし（未計測） | 比べる腕がない |
| checksum の `on`／`off` 完全一致 | 該当なし | 同一コードのため比較の意味がない |
| ADOPT のときだけ既定を反転する PR | 該当なし（反転なし） | 判定が成立しない。既定は現状のまま |
| 相手比の記録・推測の明記・tolerance 不変 | 背景比は引用のみ | イシューの背景比（Mac mini 系列 B で N=1024 が 0.89 倍・N=2048 が 0.69 倍）は再計測していない引用で、この A/B の結果ではない。「数 % 〜 10 %」の見込みは推測のまま検証していない。tolerance・FMA 契約・baseline・REQ-8 の境界検査は不変 |

- 実施しなかったこと: 実機ロック（hw-lock）の取得、Mac mini のセッション、rsync・ビルド・計測、`docs/perf/logs/sme-neon-hetero-ab-3239/` と `RULE.txt` の作成。
- 再提起の条件: §5.11 の再評価の条件に従う。混在分配が新しい issue で実装された後に、改めて A/B の issue を起こす（起票は所有者の判断事項）。

### 5.13 SME 既定 ON の前後 A/B（Mac mini・GB10。イシュー #3232）

#3231 で `SME_PRODUCTION_ENABLED` を `true` にした main そのもの（ON）と、同じコミットで定数 1 行を `false` に戻したもの（OFF。`off-arm.patch`）を
Mac mini（Apple M4）と GB10 で 5 round ずつ測った。規則は `docs/perf/logs/sme-default-on-ab-3232/RULE.txt`（計測前に固定。今回も変更していない）、
集計は `aggregate.py` の機械出力（`aggregate-macmini.md`・`aggregate-gb10.md`。手で直していない）。腕の向きは before = OFF・after = ON、比 = ON / OFF（1.00 以下なら ON が速い）。

| 項目 | 内容 |
|---|---|
| 計測コミット | `9cfadc452fbf6ee3633b8a5c38c016976f02f4a3`（`origin/main`。RULE 登録コミット `42ea08d4d` から `crates/`・`scripts/`・`Cargo.*`・`rust-toolchain.toml` に差分なし。両機とも ON ツリーは bundle の clone で `rev-stamp.txt` と一致・tree_fingerprint は各 `macmini/`・`gb10/` に収録） |
| 計測日時 | 2026-10-23 20:39〜20:47 UTC（Mac mini）・20:39〜20:43 UTC（GB10）。実機ロック取得・解放は各 `hwlock.log` |
| ログ | `docs/perf/logs/sme-default-on-ab-3232/{macmini,gb10}/`（マスク済み。回収後に実ホスト名・`/Users/`・`/home/`・ユーザー名・IP の grep 検査は 0 件） |
| 前提ゲート | 両機とも成立（P-TREE・P-R0・負荷ゲート・`run_ab exit=0`・`collect exit=0`。GB10 は加えて `rt_on exit=0`・`bitdump exit=0 cmp_exit=0`）。Mac mini の `sme_report` は両腕 `kernel_enabled: true`、GB10 は両腕 `false` |

#### 判定（機械出力のまま）

- GB10: **REJECT**。5/5 一貫の後退セルが 2 つある（`gemm 256 fresh`・`gemm 2048 reuse`）。
- Mac mini: **REJECT（GB10 が REJECT のため。RULE の合成規則）**。Mac mini 単独のセルは次のとおり良好で、5/5 一貫の後退はない。
- 総合: **REJECT**。

#### Mac mini（到達 10 セル・非到達 2 セル）の ON/OFF 比（5 round 中央値）

| セル | ON/OFF | 5/5 判定 |
|---|---|---|
| gemm 256 fresh／reuse | 0.744／0.782 | 5/5 で ON が速い |
| gemm 512 fresh／reuse | 0.561／0.582 | 同上 |
| gemm 1024 fresh／reuse | 0.474／0.474 | 同上 |
| gemm 2048 fresh／reuse | 0.474／0.472 | 同上 |
| train 64 fresh／reuse | 0.912／0.876 | 同上 |
| infer 64 fresh／reuse（非到達ガード） | 0.998／0.982 | 4/5 が 1.00 以下。5/5 一貫の後退なし |

#### GB10（全 12 セルが非到達のガード。SME 非搭載）の ON/OFF 比（5 round 中央値）

| セル | ON/OFF | 1.00 以下／超 |
|---|---|---|
| gemm 256 fresh／reuse | **1.014**／1.098 | 0／5（5/5 後退）・2／3 |
| gemm 512 fresh／reuse | 0.958／0.924 | 4／1・3／2 |
| gemm 1024 fresh／reuse | 1.039／1.007 | 2／3・2／3 |
| gemm 2048 fresh／reuse | 1.068／**1.061** | 1／4・0／5（5/5 後退） |
| train 64 fresh／reuse | 0.980／1.057 | 3／2・2／3 |
| infer 64 fresh／reuse | 0.947／0.932 | 3／2・4／1 |

- 5/5 一貫の後退: `gemm 256 fresh`（round 別 1.002・1.014・1.148・1.008・1.172）と `gemm 2048 reuse`（1.042・1.061・1.107・1.075・1.059）。
- 事実として確認できていること: GB10 は `kernel_enabled: false` で SME カーネルに到達せず、ON/OFF は同じ NEON 経路を通る。checksum は全 12 セルで ON/OFF 完全一致。
  bitdump は 6,726,847 行（期待値と一致）・sha256 が ON/OFF で一致（`off_sha256`＝`on_sha256`＝`b8c399e9…94af`）・`cmp` の差分 0。
- 後退の原因は未検証である。同じ経路で bit 同一なのに 5/5 一貫の後退が出ているため、定数の有無によるコード配置・計測ノイズの可能性があるが、**これは推定で、本 run では切り分けていない**。
  RULE は原因の帰属を判定に使わない（規則どおり REJECT と記録する）。切り分けには別の事前登録（#2053 §5.7 の 3 腕方式など）が要る。

#### 相手（PyTorch）比（記録のみ・判定に使わない）

- Mac mini（本 run の実測。PyTorch median / fandhe median の 5 round 中央値。torch 2.14.0・numpy 2.5.4）:

| セル | OFF | ON |
|---|---|---|
| gemm 256 fresh／reuse | 0.675／0.702 | 0.888／0.893 |
| gemm 512 fresh／reuse | 0.733／0.745 | 1.303／1.283 |
| gemm 1024 fresh／reuse | 0.656／0.649 | 1.351／1.358 |
| gemm 2048 fresh／reuse | 0.516／0.518 | 1.081／1.094 |
| train 64 fresh／reuse | 0.198／0.273 | 0.215／0.312 |
| infer 64 fresh／reuse | 0.153／0.149 | 0.153／0.153 |

- 系列 B（`framework-compare-0.11.0-macmini-remeasure`。0.11.0 registry 版）の値は参考列で、**本 run の値ではない（推定・参考）**。
  gemm 256／512 は 1.04、1024 は 0.89、2048 は 0.69、train 64 は 0.20、infer 64 は 0.25。#3231 の到達見込みも推定であり、上の実測と区別する。
- GB10: **未計測**。GB10 には RULE が前提とする torch 2.14.0+cu130 の venv がなく（確認できた torch は別 venv の 2.11.0 のみで、別版のため使っていない）、`TORCH_PY` を渡さずに実行した。
  PyTorch の jsonl は空で、相手比は欠測（RULE §6。記録のみ）。

#### 処置（所有者判断待ち）

- RULE §5 は REJECT・FAIL・GB10 後退ありのとき `SME_PRODUCTION_ENABLED` を戻す別コミットを定める。一方 #3232 の再オープンコメント（後発）は「RULE どおり記録して止め、戻すかは結果を添えて所有者に確認する」と指示している。
  自動運転では安全側に倒して後者に従い、**本 PR は Rust を一切変更していない**。`SME_PRODUCTION_ENABLED` を戻すか、GB10 の後退を別途切り分けるかは所有者の判断事項である。
- Mac mini 単独では 10 セルすべてで ON が速く、後退がないことは事実として記録する。ただし RULE の合成規則により総合判定は REJECT のままである。
- 再計測するなら RULE-reopen の追記と新しいラベルが要る（今回は再実行していない）。tolerance・baseline・FMA 契約・REQ-8 の境界検査は変更していない。

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
- **unsafe 監査記録**: `asm!` の監査結果と本番化の条件付き再承認申請は（#3231 で既定 ON。§12.11）
  `docs/cpu-gemm-sme-unsafe-audit.md`（イシュー #2119）を参照。

## 7. スコープ外

- SME2／f16（`SME_F32F32` 以外）・int8 経路、SVL≠512 対応、MC/NC の SME
  向け再チューニング、E コア／P コア親和性制御、`gemm_blis`（直列入口）
  の SME 対応、32×32（4 タイル）版。既存 `out-of-scope-tracking.md`
  規約に従い、承認後に別 issue で追跡する。

### 5.14 既定 OFF への復帰（イシュー #3232 → #3229）

§5.13 の総合判定 REJECT を受け、事前登録規則 `docs/perf/logs/sme-default-on-ab-3232/RULE.txt:55`
（REJECT／FAIL／GB10 後退ありの場合は `SME_PRODUCTION_ENABLED` を `false` に戻す 1 行とドリフト検出テストの反転を、計測記録とは別コミットで入れる）
どおりに **`SME_PRODUCTION_ENABLED` を `false`（既定 OFF）へ戻した**（2026-10-24 所有者判断）。

- 根拠: GB10 で 5/5 一貫の後退 2 セル（`gemm 256 fresh`・`gemm 2048 reuse`。`aggregate-gb10.md`）。Mac mini は到達 10 セルで ON が 5/5 速かったが、規則の合成で総合 REJECT。
- 変更: 定数 1 行と、ドリフト検出テストの反転（`sme_production_enabled_is_true_with_neon_fallback` → `sme_production_disabled_after_gb10_reject`）のみ。
  しきい値 `SME_MIN_M/N/K=256/256/64`・`sme_kernel_for_shape` の置換・`debug_assert!`・`GemmDriverVariant::TwoDDynamicNeon`・bit 一致テストは維持。tolerance・baseline・RULE.txt は不変。
- OFF では `sme_kernel_for_shape` が常に `None`（NEON）で、出力 bit は #1313 以前と同一。

### 5.15 GB10 後退 2 セルの 3 腕帰属切り分け（イシュー #3613）

§5.13 で GB10（SME 非搭載・両腕が同じ NEON 経路・bitdump bit 同一）でも `gemm 256 fresh`・`gemm 2048 reuse` が 5/5 round で後退した原因は推定のままだった。
§5.7 型の 3 腕 A/B で帰属を切り分ける。**状態: GB10 実測済み（2026-10-27 UTC・計測ツリー `e2660eba0`。規則 §8 の計測前追記を参照）**。数値・分類は `aggregate-gb10.md` の機械出力の転記で、推測値・M4 Max／Linux の代替値は書かない。

- 事前登録規則・パッチ・スクリプト: `docs/perf/logs/sme-gb10-attribution-3613/`（`RULE.txt`・`on-arm.patch`・`on-arm-prime.patch`・`orchestrate_gb10.sh`・`aggregate.py`）、A/B 本体 `scripts/bench/framework-compare/run_ab_sme_attribution_gb10_cpu.sh`。
- 3 腕（登録基準 `3e5d41016`。実際の計測ツリーは squash 後の `e2660eba0`。理由は下記「計測ツリー」。`gemm_blis/mod.rs` のみの差分）: before = HEAD そのまま（定数 false）／prime = 定数 true + `sme_kernel_for_shape` の `try_new()` を `None` へ置換／after = 定数 true のみ。
- **構造上の注意（計測前に記録）**: HEAD で定数を使う本番コードは `sme_kernel_for_shape` だけ（`mod.rs:3589〜3595`）のため、prime は before とほぼ同じコードに畳まれる。組 1（before/prime）は実質 A/A 対照、帰属を見分けるのは組 2（prime/after。`try_new()` 呼び出しの有無）。バイナリ sha256 の差は帰属の根拠にしない（パス埋め込みのため）。
- 分類（RULE §4）: 非再現（ノイズ帯・判定不能）／方法感度（判定不能）／SME 経路の選択ロジックに帰属／再現したが未分離（ノイズ帯）／FAIL／undetermined（前提不成立）。**非再現・判定不能も正式な結果**として記録する（§5.7 は未分離だった）。
- **計測ツリー**: 規則冒頭の「3e5d41016 直系」のツリーでは bench-fandhe の固定版 `=0.11.0` と workspace `0.12.0` が食い違い、A/B 本体の path patch が使われず `cargo tree` 検査で必ず失敗する（推論。`1276ff3863` での実ビルド確認は省略）。
  PR #3634 の squash 後 `e2660eba0` では固定版が `=0.12.0` で解決する。`crates/` の backend-cpu・facade・tensor-core・autodiff は `3e5d41016` との差分なしを確認した。計測前に `RULE.txt` §8 へ追記し、閾値・分類表は変えていない（所有者の事後確認対象）。
  固定の証跡: 本 PR 内の §8 追記コミット（squash merge で SHA は消えるため、正は `gb10/prereg_append.txt` の sha256 と時刻）。`gb10/prereg_append.txt`（追記後 RULE.txt の sha256 と準備時刻 2026-10-27T07:12:30Z）が `gb10/hwlock.log` の acquired（07:13:03Z）より前。
- **結果（機械出力。`docs/perf/logs/sme-gb10-attribution-3613/aggregate-gb10.md` を正とする。ログ: `.../gb10/`）**:
  - 前提ゲート: P-TREE・P-R0・P-BIT・P-RT・P-LOCK・P-GATE・P-RUNAB・P-COLLECT すべて成立。腕: before=`e2660eba0`／prime・after=それぞれ `+on-arm-prime.patch`／`+on-arm.patch`（`gb10/rev-stamp.txt`）。
  - 判定 2 セルの分類: **gemm 256 fresh = 非再現（ノイズ帯・判定不能）／gemm 2048 reuse = 非再現（ノイズ帯・判定不能）**。
  - 判定 2 セルの中央値比（> 1.00 = 後退）: gemm 256 fresh は pair0 0.964・pair1 0.876・pair2 1.072、gemm 2048 reuse は pair0 0.869・pair1 0.944・pair2 0.967。どの組も 5/5 後退ではない。round 別比は aggregate-gb10.md の表。
  - checksum は 3 腕で全セル完全一致。bitdump は 3 腕とも 6726847 行・sha256 一致（`b8c399e9…`。`gb10/bitdump/summary.txt`）。
  - 併記 10 セルのうち 5/5 後退は train 64 fresh の pair0（after/before 中央値 1.062）と pair2（after/prime 1.067）のみ（pair1 は 1.003）。併記セルで判定外のため分類しない。事実のみ記録する。
  - 腕別 bench sha256（参考。パス埋め込みのため帰属の根拠にしない）: before `4f2c2a27…`・prime `4aa35fd0…`・after `35f3f528…`（`gb10/bench_sha256.txt`）。
  - 計測は 07:13:03Z〜07:17:55Z（ロック保持時間）。PyTorch は GB10 に venv がないため未計測（RULE §5）。
- **交絡の注記**: #3635 で bench-fandhe の `--mode reuse` が常駐入力経路へ変わり、`gemm 2048 reuse` は #3232 の旧 Tape 経路と同一条件ではない。今回は非再現のため「後退がハーネス変更で消えた」可能性と「#3232 の後退がノイズだった」可能性を本計測では区別できない。
- 次の手（選択肢。いずれも **所有者判断待ち（承認未取得）**。本 issue では定数の切替・是正・新規 issue 起票を行わない）: (a) 現状維持（既定 OFF）、(b) 新しい事前登録での既定 ON の再 A/B（旧 Tape 経路相当の条件を含めるか）、(c) 計測方式の変更（round 数・ノイズ帯の見積り）。
