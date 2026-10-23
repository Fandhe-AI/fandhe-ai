# SME `unsafe asm!` の監査記録と本番化の条件付き再承認申請（イシュー #2119）

## 1. 状態

> 注記（#3231）: `SME_PRODUCTION_ENABLED` は #3231 で既定 ON に切り替えた。現在の状態は §12.11 を正とする（本節以降 §11 までは当時の記録で書き換えない）。

| 項目 | 内容 |
|------|------|
| 承認状態 | **未承認（申請中）**。本 doc は記録と申請のみであり、SME 本番化を承認したものではない |
| `SME_PRODUCTION_ENABLED` | **`false` のまま（本 PR では変更しない）**。`crates/backend-cpu/src/gemm_blis/mod.rs:3085` の `const SME_PRODUCTION_ENABLED: bool = false;` を確認済み |
| 監査対象コミット | `8bba6b57dd053f23f90b93ca1a982ade30c875ed`（main 先端。以下の行番号はこの sha 基準） |
| 監査日 | 2026-09-30 |
| 監査方法 | 機械的インベントリ（grep）・aarch64 2 ターゲットでの clippy・全 `asm!` ブロックの手読み突合。所見は監査者（本 PR の実装担当）のもので、最終判断はユーザーが行う |
| 変更範囲 | `.md` のみ。`.rs`・`Cargo.toml`・`Cargo.lock`・tolerance・baseline・ガードレール閾値・`docs/spec/` は変更しない |

> 2026-10-23 に現行 main 基準で更新した（イシュー #3230）。更新内容・再開条件の判定・所有者確認欄は §12 を参照。本節以降は 2026-09-30 時点の記録として残す。

## 2. 背景

- #1979（SME 本番結線の承認依頼）は 2026-09-19 のユーザー判断で「SME は既定 OFF のまま・結線しない」となり NOT_PLANNED で close された。その際、再開条件が 3 つ定められた。
  1. しきい値候補（`SME_MIN_K=128` 等）で M4 Max の到達セル全部が 5/5 round で 1.00 以下、かつ checksum 一致の ADOPT になる。
  2. GB10 既存経路の非後退が 5/5 run 一貫で再現する。
  3. `unsafe asm!` の監査を付けたうえで、別 issue で承認を取り直す。
- 条件 1・2 は兄弟イシュー #2118 が担当する。2026-09-30 時点で #2118 は OPEN（未達）。
- 本 doc は条件 3 の成果物（監査記録と再承認申請）である。条件 1・2 の代替にはならない。

## 3. 監査範囲と、イシュー記載との違い

イシューが挙げる `crates/backend-cpu/src/sme.rs` は存在しない。実際の対象は次のとおり。

| ファイル | 内容 |
|----------|------|
| `crates/backend-cpu/src/gemm_blis/microkernel/sme.rs` | `asm!` 5 ブロック（本番 2・`#[cfg(test)]` 3） |
| `crates/backend-cpu/src/sme_detect.rs` | `rdsvl_bytes` の `asm!` 1 ブロック（本番） |
| `crates/backend-cpu/src/gemm_blis/microkernel.rs` | `SmeKernel::try_new`／`current_thread_capable`／`run`／`run_with_ldc` の unsafe 呼び出し |
| `crates/backend-cpu/src/gemm_blis/mod.rs` | `SME_PRODUCTION_ENABLED`・`dispatch_two_d_dynamic`・ドリフト検出テスト |

「unsafe 3 件」の対応は次のとおり。

| イシューの呼称 | 実体 | 区分 |
|----------------|------|------|
| fmopa 手書き asm | `sme.rs::compute`（`smstart`→C プリロード→`fmopa` ループ→ストア→`smstop`） | 本番 |
| state machine | `sme.rs::has_pending_lazy_za_save`（`mrs TPIDR2_EL0`）・`sme_detect.rs::rdsvl_bytes`（SVL 読み取り）・`SmeKernel::current_thread_capable`（実行スレッドでの再確認） | 本番 |
| save-restore | `sme.rs` の `mod tests` 内 `PendingLazySaveGuard` と 3 つの `asm!`、`AlignedSaveBuffer` の `alloc_zeroed`／`dealloc` | テスト専用（本番から到達しない） |

## 4. 機械的インベントリ

再現コマンド: `grep -n "asm!" <ファイル>`、`grep -n "unsafe fn\|unsafe {" <ファイル>`、`grep -rn "asm!" crates/backend-cpu/src`（コメント行を除く）。`sme.rs` の `mod tests` は 431 行から。

| 場所 | 区分 | 用途 |
|------|------|------|
| `sme.rs:127` | 本番 | `mrs {0}, S3_3_C13_C0_5`（TPIDR2_EL0 読み取り。`options(nomem, nostack, preserves_flags)`） |
| `sme.rs:201` | 本番 | `compute` 本体（`options(nostack)`） |
| `sme.rs:911` | テスト | `read_za0_and_finish_streaming` |
| `sme.rs:1005` | テスト | `finish_pending_save`（`msr`＋`smstop`） |
| `sme.rs:1107` | テスト | `enter_dormant_za_with_pending_lazy_save` |
| `sme_detect.rs:160` | 本番 | `rdsvl {0}, #1`（`options(nomem, nostack, preserves_flags)`） |

- クレート内で SME 関連以外の `asm!` 実行コードはない（`lib.rs`・`gemm_prefetch_bandwidth_diag_tests.rs` の該当行はコメントのみ）。
- 本番の unsafe 境界: `unsafe fn` は `has_pending_lazy_za_save`（112）・`compute`（154）・`kernel_unchecked_with_ldc`（299）・`kernel_unchecked`（352）の 4 件、`unsafe` ブロックは `sme.rs` 126・200・311・321・360 の 5 件、`sme_detect.rs` は `rdsvl_bytes`（150）と 159・209 行、`microkernel.rs` の呼び出しは 825・864・883 行。
- 各 `unsafe fn` に `# Safety` 節があり、各 unsafe ブロックの直前に SAFETY コメントがある（手読みで確認）。
- `smstart` を含む全 `asm!` ブロック（本番・テスト）で、`out("v0")`〜`out("v31")`・`out("p0")`〜`out("p15")`・`out("ffr")` を宣言している（手読みで確認）。スライスインデックスを使うブロックは `out("w12")` も宣言している。`smstart`／`smstop` は同一ブロック内で対に閉じている（`compute`、`read_za0_and_finish_streaming`）。
- `options` の整合: `nomem` を付けたブロック（`mrs`／`rdsvl`／`msr`＋`smstop`）はメモリアクセスを含まない。`compute` は `cmp`／`subs` でフラグを書き換えるため `preserves_flags` を付けていない。

## 5. lint 検証

| 実行 | 結果 |
|------|------|
| `cargo clippy -p fandhe-ai-backend-cpu --target aarch64-unknown-linux-gnu --all-targets -- -D warnings` | 成功 |
| `cargo clippy -p fandhe-ai-backend-cpu --target aarch64-apple-darwin --lib -- -D warnings` | 成功 |
| 同 `--target aarch64-apple-darwin --all-targets` | このホストでは失敗。原因は `backend-metal`（dev-dependency）の dead_code／type_complexity 警告 5 件で SME とは無関係（本 PR は未変更） |
| 報告専用: `-W clippy::undocumented_unsafe_blocks -W clippy::missing_safety_doc`（両ターゲット・`--lib`） | 警告 1 件のみ。`sme_detect.rs:159`（SAFETY コメントの書式が `SAFETY（…）:` で、lint が要求する `SAFETY:` 形式ではない。内容自体は存在する） |

- SME 命令を実行するテストはこのホスト（SME 非対応）では動かせないため、再実測していない。既存の証拠を引用する。
  - `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §4: M4 Max 実機で bit 一致・lazy save フォールバックのテストが PASS。
  - 同 §5.6: GB10 でゲート ON/OFF の全出力が bit 同一（GB10 は SME 非対応で `kernel_enabled: false`）。

## 6. 項目別の所見

判定は 適合／条件付き適合／不適合。根拠は監査対象 sha の行番号。

| ID | 観点 | 判定 | 所見・根拠 |
|----|------|------|-----------|
| M1 | 状態の出入り | 適合 | `compute` は `smstart`（SM と ZA を有効化）で始まり `smstop` で閉じる（`sme.rs:201`〜）。SM=1 の間に実行されるのは asm 内の命令のみで、コンパイラ生成コードが SM=1 で走る区間はない。入口条件は `TPIDR2_EL0==0` の確認（M2）で担保する |
| M2 | TOCTOU | 適合 | `kernel_unchecked_with_ldc` が `has_pending_lazy_za_save`（311）の直後に `compute`（321）を同一スレッド・同一関数内で呼ぶ。間に ZA へ触れるコードはない。SVL は `current_thread_capable`（`microkernel.rs`）が実行スレッドで毎回 `rdsvl` を読み直す（`sme_detect.rs::sme_report`）。`SmeKernel` は `Copy` だが再確認するため、Rayon worker の SVL 違いに耐える |
| M3 | 境界 | 適合 | asm の前に `check_panel_lengths`／`check_c_tile_bounds`（`checked_sub`／`checked_mul`／`checked_add`。`microkernel.rs:228`・282）を通す（`sme.rs:299` 以降）。最大読み取りオフセットは ap・bp とも各長さ以内、c は `(MR-1)*ldc+NR <= c.len()`。`ldc_bytes = ldc*4` は `c.len()*4` 以下に抑えられ桁あふれしない。`kc` は X レジスタ（64 bit）で扱う。`ld1w`／`st1w` の要求アライメントは要素サイズ（4 バイト）で、`&[f32]` が満たす。ループ後のポインタ（`cpre` 等）は加算のみで参照しない |
| M4 | エイリアシング | 適合 | `c` は `&mut`、`ap`／`bp` は `&` で、Rust の借用規則上重ならない。asm 内の書き込み先は `c` のみ |
| M5 | クロバー | 適合 | 4 節のとおり V/P/FFR/w12 を全宣言。ポインタ・カウンタは `inout(reg) … => _` で破棄扱い。v8〜v15 の callee-saved 部分はクロバー宣言によりコンパイラが退避する |
| L1 | リーク | 適合（既存記録に依拠） | `compute` は `smstop` で SM・ZA を無効化して終わる。ストリーミングモード終了と ZA 無効化で Z・P・FFR・ZA がゼロ化されるのは Arm ARM の仕様（`docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §2 に一次ソース確認の記録あり。本監査で Arm ARM を再読はしていない）。ZA と SVL はスレッドごとの状態 |
| L2 | テストガードの解放順序 | 適合 | `PendingLazySaveGuard::drop` が `finish_pending_save` で `TPIDR2_EL0` をクリアしてから、フィールド（`_block`／`_save_buffer`）が宣言順に解放される。`cleaned_up` で二重発行を防ぐ。`AlignedSaveBuffer` の `alloc_zeroed`／`dealloc` は同一 `Layout` で対になる（`sme.rs:783`・800）。テスト専用で本番から到達しない |
| L3 | シグナル | 条件付き適合 | Linux: カーネル文書（`Documentation/arch/arm64/sme.rst`。本監査で参照）は「シグナルハンドラは PSTATE.SM=0・PSTATE.ZA=0・TPIDR2_EL0=0 で起動し、ZA は signal frame に保存・復帰時に復元される」と記す。**macOS は一次ソースを確認できていない**（残留リスク。M4 Max 実機テストは PASS しているが、シグナル配送下の挙動は未検証） |
| P1 | panic しない | 適合 | 本番経路（`sme.rs` 431 行より前・`sme_detect.rs`・`microkernel.rs` の SmeKernel）に `unwrap`／`expect`／実行時 `assert!` はない（`const _: () = assert!` はコンパイル時）。`expect` はテスト helper のみ（`enter_dormant_za_with_pending_lazy_save`）。非対応・保留中 save・長さ不正は `scalar_fallback` か `Result` で返す |
| R1 | 到達可能性 | 条件付き適合 | `pub mod gemm_blis`→`pub mod microkernel`→`pub mod sme` により、公開済みクレート `fandhe-ai-backend-cpu` から `pub unsafe fn kernel_unchecked*` に外部から到達できる。関数内で TPIDR2 を自己検査し保留中 save は破壊しないが、SME 対応・SVL=64 の確認は呼び出し元契約（`# Safety`）に委ねられる（非対応 CPU では `mrs` が SIGILL になりうる）。`SmeKernel` は `pub(crate)` の `try_new` でのみ構築できる。facade（唯一のサポート公開面）は SME を公開していない（`crates/facade/src` に該当なし）。内部クレートの公開面は既存方針どおりで、本 PR では変更しない |
| G1 | ゲート | 適合 | `dispatch_two_d_dynamic`（`mod.rs:2981`）は `SME_PRODUCTION_ENABLED && … && SmeKernel::try_new()` の短絡評価で、`false` の間は `try_new`（`sme_report`・`rdsvl`）自体が呼ばれない。ドリフト検出テスト `sme_production_enabled_is_false_pending_measurement`（`mod.rs` 5044 付近）がある |
| D1 | 検出入力 | 適合 | `/proc/cpuinfo` は最初の `Features` 行のみ解釈し、`sme` と `smef32f32` の両方を要求する（`sme_detect.rs::parse_cpuinfo_features`）。macOS は `read_sysctl` の固定引数・`env_clear()`。環境変数による上書き機構はない。読み取り・parse の失敗はすべて非対応（fail-closed） |

## 7. OWASP Top 10 の対応付け

| 項目 | 対応 |
|------|------|
| A03 インジェクション | 検出入力（`/proc/cpuinfo`・`sysctl`）の扱いは D1 のとおり。外部入力を SME 経路の制御に使わない |
| A04／A05 不安全な設計・設定ミス | 本番化は const ゲート（既定 OFF）で、環境変数・実行時設定では切り替えられない。検出失敗は NEON 側に倒れる。本 PR は値を変えない |
| A06 脆弱なコンポーネント | 依存の追加・変更なし。`std::arch::asm!` のみ |
| A08 データ整合性 | bit 一致契約を維持し、tolerance・baseline は変更しない。本番化はユーザー承認を経る |
| A01・A02・A07・A09・A10 | 該当しない（アクセス制御・暗号・認証・ログ・SSRF の要素を持たない演算カーネル） |

## 8. 発見事項

いずれもコード修正を伴うため本 PR では直さない。**未起票・ユーザー承認待ち**（`.claude/rules/out-of-scope-tracking.md` に従い、承認なしに起票しない）。

| # | 重大度 | 内容 |
|---|--------|------|
| a | 低（文書） | `sme.rs::compute` の SAFETY コメントが存在しないパス `docs/cpu-gemm-sme-fmopa-microkernel.md` を参照している（正しくは `docs/perf/…`） |
| b | 低（文書） | 同コメント内（`sme.rs:170`）の「外径」は「外積」の誤記 |
| c | 低（文書） | `sme_detect.rs` の冒頭 doc は `rdsvl` を「ストリーミングモードへ一時的に遷移する」命令と書くが、`rdsvl` は SVL を読むだけでモード遷移を伴わないと監査者は理解している（Arm ARM での再確認を推奨） |
| d | 低（lint） | `sme_detect.rs:159` の SAFETY コメント書式（`SAFETY（…）:`）が `clippy::undocumented_unsafe_blocks` の書式に合わない |
| e | 情報 | `compute` は、呼び出し元が ZA をアクティブに使用中（TPIDR2==0 だが ZA 有効）の状態を検出しない。Rust コードは通常この状態を作らないが、将来インライン asm で ZA を持ち込む呼び出し元が現れると `smstop` がその ZA を破壊しうる。現状の呼び出し元（`dispatch_two_d_dynamic`）では発生しない |

## 9. 総合所見

- 本番の `unsafe`（`compute`・`has_pending_lazy_za_save`・`rdsvl_bytes` と呼び出し 3 箇所）について、境界検査・状態の出入り・クロバー宣言・panic 不在・ゲートの各観点で不適合は見つからなかった。
- 残留リスク: (1) macOS のシグナル配送下の挙動は一次ソース未確認（L3）。(2) L1 の「ゼロ化」は既存 doc の一次ソース確認に依拠し本監査で再確認していない。(3) 実機テストは既存の証拠の引用で、本監査での再実行はない。(4) 発見事項 e。
- 上記は監査者の所見であり、承認ではない。

## 10. 本番化の再承認申請（条件付き）

承認を求める事項は次の 2 点。

1. 本 doc の所見（unsafe の正当性）の受理。
2. SME 本番化。**ただし #2118 が ADOPT（再開条件 1・2 の成立）になることを前提とする**。#2118 が未達の間は何も変更しない。

承認された場合の別 PR の変更内容:

- `SME_PRODUCTION_ENABLED = true` への変更。
- `sme_production_enabled_is_false_pending_measurement` の更新（値のドリフト検出を新しい契約に合わせる）。
- しきい値の確定（perf doc §5.4.4 の候補。`SME_MIN_K=128` 等は #2118 の結果で決める）。
- 切替後の framework-compare gemm cpu の before/after の記録。
- 非 SME 環境（GB10 等）で従来経路と bit 同一のまま動くことの維持確認。

| 承認欄 | 状態 |
|--------|------|
| ユーザー承認 | **未承認（申請中）** |

## 11. 参照

- `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §3・§4・§5.4〜§5.7・§6
- イシュー #1587（SME マイクロカーネル）・#1978（実測）・#1979（ユーザー判断）・#2050（GB10 bit 同一実証）・#2118（再実測）・#2119（本監査）
- Linux `Documentation/arch/arm64/sme.rst`（シグナル配送時の状態・スレッドごとの VL）

## 12. 2026-10-23 更新（イシュー #3230）

§1〜§11 は 2026-09-30 時点の記録として書き換えず、本節に現行 main 基準の更新を追記する。条件 3（security-auditor の監査）の更新と、再開条件 1〜3 の充足判定が目的。**承認ではない**。

### 12.1 状態

| 項目 | 内容 |
|------|------|
| 承認状態 | **未承認**。D-C（#3018 コメント 6111643437・2026-10-11）は「SME カーネルの本番化」を承認したが、個々の `unsafe` ブロックの security-auditor 監査を条件とする条件付き承認である。本節の更新所見の受理は 12.8 のとおり確認依頼中 |
| `SME_PRODUCTION_ENABLED` | **`false` のまま（本 PR では変更しない）**。`crates/backend-cpu/src/gemm_blis/mod.rs:3541` の `const SME_PRODUCTION_ENABLED: bool = false;`（§1 の 3085 から移動） |
| 監査対象コミット | `45347ae0a5f3ca436b8136424a136d660c5a5280`（main 先端。本節の行番号はこの sha 基準。旧: `8bba6b57dd05…`） |
| 監査日 | 2026-10-23 |
| 監査方法 | 旧監査との差分評価・機械的インベントリの再取得・全項目の再突合・lint とテストの再実行。所見は監査者（本 PR の実装担当）のものであり、最終判断はユーザーが行う。独立した security-auditor subagent による監査は自動運転モードの制約により未実施で、PR レビューでの実施を要する |
| 変更範囲 | `.md` のみ。`.rs`・`Cargo.toml`・`Cargo.lock`・tolerance・baseline・`docs/spec/` は変更しない |

### 12.2 旧 sha からの差分評価

- `git diff --stat 8bba6b57..45347ae0a` を SME 関連 4 ファイルへ限ると、差分は `crates/backend-cpu/src/gemm_blis/mod.rs`（+800／-42）のみ。`microkernel/sme.rs`・`sme_detect.rs`・`microkernel.rs` は差分なし。したがって §4・§6 の行番号はこの 3 ファイルについて現行 main でもそのまま有効。
- `mod.rs` の差分（+800／-42）は診断テストだけではない。内訳は次のとおり。
  - `#[ignore]` の診断テスト（#3175・#3176。`sme_vs_neon_ab_thread_sweep`・`sme_vs_neon_ab_mlp_rect`・`microcall_median_ns`・`sme_microcall_fixed_cost_probe`）と、`cfg(test)` 限定の `jobgrid_probe`（#3486）。
  - 本番経路の safe コードの変更。小 m GEMM の job 統合・A の pack 共有（#3521）、常駐重みの pack 済み panel キャッシュ（#3522）に伴い、`run_two_d_job`・`gemm_blis_two_d_dynamic_region`・`dispatch_two_d_dynamic` が `jobs_per_worker`・`share_a`・`prepack` を受け取るようになった。SME 経路もこの region 関数を経由する。
  - 差分の追加行で `unsafe`・`asm!` に一致するのはコメント行 3 件のみで、**新規の `unsafe`・`asm!` は 0 件**。`unwrap`／`expect` の追加はいずれも `cfg(test)` 内（`install_in_pool_for_test` ほか）。追加テストは safe な `SmeKernel::try_new`・`Microkernel::run` だけを呼ぶ。
  - 上記の本番経路の変更が SME の `unsafe` 境界に与える影響: `SmeKernel::run`／`run_with_ldc` は引き続き `check_panel_lengths`／`check_c_tile_bounds` を経由してから asm に入る（`microkernel.rs` 無変更）。C の分割（`split_c_into_jobs`）は safe なスライス分割。したがって unsafe の前提（M3・M4）を変える変更は見つからなかった。ただし region 関数の新しい分配ロジック自体の逐行レビューは本更新の範囲外で、PR レビューでの確認を要する。
- `mod.rs` の行番号の対応表。

| 項目 | 旧 | 新 |
|------|----|----|
| `SME_PRODUCTION_ENABLED` | 3085 | 3541 |
| `SME_MIN_M`／`SME_MIN_N`／`SME_MIN_K` | （未記載） | 3549／3551／3553 |
| `sme_shape_eligible` | （未記載） | 3561 |
| `dispatch_two_d_dynamic`（aarch64 版。ゲート式） | 2981 | 3415（3429） |
| `sme_production_enabled_is_false_pending_measurement` | 5044 付近 | 5612 |

### 12.3 機械的インベントリ（新 sha）

再現コマンド: `grep -n "asm!" <ファイル>`、`grep -n "unsafe fn\|unsafe {" <ファイル>`、`grep -rn "asm!" crates/backend-cpu/src`（コメント行を除く）。実行コードの `asm!` は 6 ブロックのまま（本番 3・テスト 3）で、§4 の表と行番号まで一致する。

| 場所 | 区分 | 用途 |
|------|------|------|
| `sme.rs:127` | 本番 | `mrs`（TPIDR2_EL0 読み取り） |
| `sme.rs:201` | 本番 | `compute` 本体 |
| `sme.rs:911`・`1005`・`1107` | テスト | 順に `read_za0_and_finish_streaming`・`finish_pending_save`・`enter_dormant_za_with_pending_lazy_save` |
| `sme_detect.rs:160` | 本番 | `rdsvl` |

- 本番の `unsafe fn`: `has_pending_lazy_za_save`（`sme.rs:112`）・`compute`（154）・`kernel_unchecked_with_ldc`（299）・`kernel_unchecked`（352）・`rdsvl_bytes`（`sme_detect.rs:150`）。本番の `unsafe` ブロック: `sme.rs` 126・200・311・321・360、`sme_detect.rs` 159・209、`microkernel.rs` の呼び出し 825・864・883。
- `mod tests`（`sme.rs:432` 以降）の `unsafe` はテスト専用で本番から到達しない。

### 12.4 項目別の再突合

| ID | 判定 | 根拠（新 sha） |
|----|------|----------------|
| M1 | 適合（変更なし） | `sme.rs` が無変更。`compute`（201）は `smstart` で始まり `smstop` で閉じる |
| M2 | 適合（変更なし） | `sme.rs:311`→`321` の直列。`microkernel.rs:825` も実行スレッドで `has_pending_lazy_za_save` を確認 |
| M3 | 適合（変更なし） | `check_panel_lengths`／`check_c_tile_bounds`（`microkernel.rs:228`・282）が経路上に残る。`mod.rs` の追加行は `unsafe`・`asm!` を持たず、カーネル呼び出しは `Microkernel::run` 経由（12.2 の注記のとおり region 関数の分配ロジックの逐行レビューは範囲外） |
| M4 | 適合（変更なし） | `sme.rs` 無変更 |
| M5 | 適合（変更なし） | 6 ブロックの `out(...)` 宣言は旧監査から不変（ファイル無変更） |
| L1 | 適合（既存記録に依拠） | Arm ARM の再読は未実施 |
| L2 | 適合 | `sme.rs:783`・800 は無変更。テスト専用 |
| L3 | 条件付き適合（据え置き） | macOS のシグナル配送時の一次ソースは本更新でも未確認。推測で断定しない |
| P1 | 適合 | `microkernel.rs` の `SmeKernel` 経路とあわせ、追加行に `unwrap`／`expect` の本番経路混入なし（診断テストはテスト専用） |
| R1 | 条件付き適合（据え置き） | 公開面は不変。facade は SME を公開していない |
| G1 | 適合 | `dispatch_two_d_dynamic`（`mod.rs:3415`）のゲート式は `SME_PRODUCTION_ENABLED && sme_shape_eligible(..) && let Some(kernel) = SmeKernel::try_new()`（3429〜3431）の短絡評価で、`false` の間は `try_new` が呼ばれない。ドリフト検出テスト（`mod.rs:5612`）が PASS。環境変数による上書き機構はない。追加された診断テストは `#[ignore]` かつ `mod tests`（`#[cfg(test)]`）内で本番経路に入らない |
| D1 | 適合（変更なし） | `sme_detect.rs` 無変更 |

新しい所見は追加なし。

### 12.5 lint とテストの再実行（M4 Max・aarch64-apple-darwin。このホストは SME 対応）

| 実行 | 結果 |
|------|------|
| `cargo clippy -p fandhe-ai-backend-cpu --target aarch64-apple-darwin --all-targets -- -D warnings` | 成功（§5 で失敗していた `backend-metal` の警告は解消している） |
| 同 `--target aarch64-unknown-linux-gnu --all-targets` | 成功 |
| 報告専用 `-W clippy::undocumented_unsafe_blocks -W clippy::missing_safety_doc`（`--lib`） | SME 関連の警告は `sme_detect.rs:159` の 1 件のみ（発見事項 d が残存。ほかに SME 無関係の `neon.rs`・`softmax.rs` に警告あり） |
| `cargo test -p fandhe-ai-backend-cpu --release` | 失敗 0 件。ドリフトガード `sme_production_enabled_is_false_pending_measurement` を含む |
| `cargo test … --release --lib -- --ignored sme_kernel gemm_blis_parallel_variant_sme_matches_naive_bit_exact_when_available --test-threads=1` | 11 件 PASS（scalar 参照との bit 一致・非正規化数・NaN・lazy save フォールバック・決定性）。#[ignore] の計測テストは実行していない |

- §9 の残留リスク (3)（実機テストが引用のみ）は、SME 対応ホストでの再実行で PASS を確認したため解消した。同 (2)（L1 の一次ソース）・(1)（L3）は据え置き。
- Mac mini・GB10 には接続していない（計測しないため実機ロックも不要）。

### 12.6 発見事項 a〜e の現状

| # | 現行 main の位置 | 扱い |
|---|------------------|------|
| a | `sme.rs:172`（存在しないパス `docs/cpu-gemm-sme-fmopa-microkernel.md` を参照） | #3231 の PR で是正済み（§12.11） |
| b | `sme.rs:170`「外径」 | #3231 の PR で是正済み（§12.11） |
| c | `sme_detect.rs:29`・`147`（`rdsvl` を「ストリーミングモードへ遷移」と記述） | #3231 の PR で是正済み（§12.11） |
| d | `sme_detect.rs:152`（SAFETY の書式。unsafe ブロックは 159） | #3231 の PR で是正済み（§12.11） |
| e | `compute` は呼び出し元の ZA 使用中状態を検出しない | 情報。現行の呼び出し元（`dispatch_two_d_dynamic` → `gemm_blis_two_d_dynamic_region`）と新しい診断テストは ZA を持ち込まないため据え置き |

起票候補: a〜d を 1 件の docs/lint issue として #3231 に含めるか別 issue にするか。自動運転モードのため起票していない。

### 12.7 再開条件の判定

| 条件 | 判定 | 根拠 | 保留・補足 |
|------|------|------|-----------|
| 1（到達セル 5/5 で 1.00 以下かつ checksum 一致の ADOPT） | **Mac mini では充足。ただし計測機の読み替えは所有者確認が未取得のため判定保留** | `docs/perf/cpu-gemm-sme-ab-macmini.md`・`docs/perf/logs/sme-macmini-3176/aggregate.md`・`macmini/r1r2/`・`macmini/r4_run{1..5}.log`・`macmini/sme_report.txt`。#3175 の `docs/perf/cpu-sme-thread-scaling-macmini.md` は参考 | 元の条件は M4 Max。M4 Max は §5.4.4 が undetermined で、#2118 の §5.8.3 は全欄未実測。2026-10-11 の所有者指示（計測は Mac mini）は根拠として引用するが、条件 1 の読み替えへの確認の代わりにはしない。M4 Max の値を Mac mini の値として扱わない |
| 2（GB10 既存経路が 5/5 一貫で非後退） | **未充足（#3232 の実測で再確認。2026-10-23 UTC）** | `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.5（後退あり。1024/reuse が 5/5 一貫）・§5.6（bit 同一は成立）・§5.7（未分離。ノイズ帯。§5.5 の判定は覆っていない）・`docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/`・#3232 実測: `docs/perf/logs/sme-default-on-ab-3232/aggregate-gb10.md`（5/5 一貫の後退が `gemm 256 fresh`・`gemm 2048 reuse` の 2 セル。checksum 全一致・bitdump 一致。判定 REJECT）・`docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.13 | GB10 は SME 非対応で `kernel_enabled: false`。ツリー設計では最終確認を #3232（ON の前後比較）が担い、後退があれば #3231 を戻す。この「後置ゲート」扱いも 12.8 の確認対象 |
| 3（unsafe 監査＋別 issue での承認） | **監査は本 PR で更新（自己点検。独立監査は PR レビューで要実施）。承認は D-C の条件付き承認にとどまる** | 12.3〜12.5・#3018 コメント 6111643437 | D-C は本更新前の方針承認のため、更新所見の受理を 12.8 で確認する |

### 12.8 所有者確認欄

| # | 確認事項 | 状態 |
|---|----------|------|
| 1 | 条件 1 の計測機を M4 Max から Mac mini へ読み替えてよいか | 確認済み（2026-10-24 所有者、#3231 コメント） |
| 2 | 条件 2（GB10 非後退）を #3232 の後置ゲートとして扱い、#3231 は条件 2 未充足のまま着手してよいか | 確認済み（2026-10-24 所有者、#3231 コメント） |
| 3 | 12.3〜12.5 の更新所見の受理（独立 security-auditor 監査を含む） | 確認済み（2026-10-24 所有者、#3231 コメント。条件は §12.11） |

確認依頼は PR 本文と #3230 で行う。承認を得ていないものを承認済みとは記載しない。

### 12.9 #3231 のゲート（2026-10-24 に 12.8 が確認済みとなり解除）

**12.8 の 3 項目が「確認済み」になるまで、#3231（しきい値の確定と `SME_PRODUCTION_ENABLED` の既定 ON）は着手しない。** 条件 2 は未充足のため、条件 2 を後置ゲートとして扱う確認が取れない限り止める側に倒す。#3229 の前提作業としての本 issue の結論はこれである。

### 12.10 承認欄の追記

| 承認欄 | 状態 |
|--------|------|
| D-C（SME カーネルの本番化。#3018 コメント 6111643437） | 条件付き承認（unsafe 監査が条件） |
| 本節の更新所見の受理 | 条件つき受理（2026-10-24 所有者。§12.11） |
| §10 の再承認申請 | §10 の記載は 2026-09-30 時点のまま。現在の状態は本節を正とする |

### 12.11 #3231 での既定 ON と受理条件の履行（2026-10-24 追記）

出典: https://github.com/Fandhe-AI/fandhe-ai/issues/3231#issuecomment-6369076139（所有者コメント。12.8 の 3 項目の確認根拠）。

- **是正（本 PR）**: 発見 a（`sme.rs` のパス誤り）・b（「外径」→「外積」）・c（`sme_detect.rs` の `rdsvl` 記述。ストリーミングモードへの遷移を伴わない読み取り専用命令・SME 非実装 CPU では UNDEFINED へ修正）・d（SAFETY コメントの書式を `// SAFETY:` へ）を是正済み。発見 e は情報として据え置く。
  c の一次ソース（Arm ARM）の再参照は本 PR では行っておらず、修正文言は SIGILL 注意の要点（OS 判定の後にのみ実行）を保った記述である。
- **`debug_assert!` 追加**: `microkernel.rs` の `sme::kernel_unchecked` 呼び出しの戻り値を `res` で受け `debug_assert!(res.is_ok())` とした（release では従来どおり破棄。新規 `unsafe`・`asm!` なし）。
- **L3 は残留リスクとして受け入れ**（macOS でストリーミング中にシグナルが配送されたときの XNU の保証が一次ソースで未確認。所有者が受け入れ）。
- **副作用（既定 ON）**: macOS では最初に対象となる GEMM のとき `cached_os_flag`（`OnceLock`）の初期化で `/usr/sbin/sysctl` を子プロセスとして起動する。`FEAT_SME` から `SME_F32F32` への `&&` 短絡のため、SME 搭載機では最大 2 回、非搭載の Mac では 1 回（所有者コメントは「1 回」と記述。実装に即して最大 2 回と記す）。いずれも固定パス・固定引数・`env_clear()`、失敗したら NEON。Linux（GB10）では `/proc/cpuinfo` を 1 回読む。
- **独立 security-auditor 監査の要約（2026-10-24・HEAD `53d70a83b`・読み取り専用）**: SME 3 ファイルは §12 基準 sha から変更なし。`asm!` は 6 ブロック（本番 3・テスト 3）、本番 `unsafe fn` 5・`unsafe` ブロック 10 で §12.3 と一致し全てに SAFETY コメントあり。smstart/smstop は同一 `asm!` 内で対、clobber は全宣言、ポインタ・長さ・ldc は `checked_*` 検査（REQ-8）を通る。検出は OS フラグ → `rdsvl` の順で実行スレッドごとに SVL と TPIDR2 を再確認し、失敗時は NEON かスカラー（GB10 は `rdsvl` に到達しない）。prepack キーに `nr` が含まれ NEON と SME の panel は混ざらない。P0・P1 なし、P2 は 2 件（`debug_assert!` と L3）。結論は条件つきで既定 ON 可。
- 確定記録は `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.10。GB10 の非後退は #3232 の後置ゲート。

| 承認欄 | 状態 |
|--------|------|
| D-C（SME カーネルの本番化。#3018 コメント 6111643437） | 条件付き承認（unsafe 監査が条件）。#3231 で既定 ON |
| 本節の更新所見の受理 | 条件つき受理（2026-10-24 所有者・https://github.com/Fandhe-AI/fandhe-ai/issues/3231#issuecomment-6369076139） |
