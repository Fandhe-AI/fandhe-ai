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
  `SME_MIN_N=256`・`SME_MIN_K=64`。§5 R4 参照）。
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

### 5.2 実測結果

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

### 5.3 再開条件・引き継ぎ

正式な R1/R2/R4 実測（`docs/perf/logs/cpu-gemm-sme-fmopa-1587/` の
オーケストレーションスクリプト参照）を両実機（Apple M4 Max・DGX Spark
GB10）で完了し、事前登録規則を満たした場合のみ
`SME_PRODUCTION_ENABLED = true` へ切り替える。GB10 は
`sme_report().kernel_enabled == false` になる想定のため（Grace CPU は
SME 非対応）、GB10 側の役割は「本番経路への副作用がないことの確認
（既存 bit 一致テスト群・framework-compare の非後退）」に限られる。

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
