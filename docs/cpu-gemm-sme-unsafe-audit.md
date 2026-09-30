# SME `unsafe asm!` の監査記録と本番化の条件付き再承認申請（イシュー #2119）

## 1. 状態

| 項目 | 内容 |
|------|------|
| 承認状態 | **未承認（申請中）**。本 doc は記録と申請のみであり、SME 本番化を承認したものではない |
| `SME_PRODUCTION_ENABLED` | **`false` のまま（本 PR では変更しない）**。`crates/backend-cpu/src/gemm_blis/mod.rs:3085` の `const SME_PRODUCTION_ENABLED: bool = false;` を確認済み |
| 監査対象コミット | `8bba6b57dd053f23f90b93ca1a982ade30c875ed`（main 先端。以下の行番号はこの sha 基準） |
| 監査日 | 2026-09-30 |
| 監査方法 | 機械的インベントリ（grep）・aarch64 2 ターゲットでの clippy・全 `asm!` ブロックの手読み突合。所見は監査者（本 PR の実装担当）のもので、最終判断はユーザーが行う |
| 変更範囲 | `.md` のみ。`.rs`・`Cargo.toml`・`Cargo.lock`・tolerance・baseline・ガードレール閾値・`docs/spec/` は変更しない |

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
