# ライセンス可否表（TASK-1.3）

REQ-1（`docs/spec/04-requirements.md:61` のライセンス要件）・`.claude/rules/deps-policy.md`
に基づき、許容依存 8 区分（直接依存）とその推移的依存の可否を記録する。
`docs/spec/01-brainstorm.md:149-163` の判断軸 a〜e に基づく境界判断も参照する。

## 1. 目的・適合基準

- 適合基準は **MIT OR Apache-2.0 系**とする（`objc2-metal` は Zlib OR Apache-2.0 OR MIT の三重ライセンス。deps-policy.md）
- 旧 issue #2 の教訓により、**feature 除外による回避を推定で記述しない**。有効化しうる feature 組合せごとに `cargo tree` を実測し、個別に適合確認する（本ファイル 3〜4 節）
- MPL-2.0 等コピーレフトの推移的混入は実測で監視する（6 節）

## 2. 直接依存 8 区分の可否表

crates.io の `license` フィールドを `cargo metadata --locked` 経由で実確認した（実測方法は 7 節）。

| 区分 | クレート | 固定バージョン | ライセンス | 可否 | 判断軸（根拠） |
|------|---------|---------------|-----------|------|---------------|
| CUDA | `cudarc` | `=0.19.8`（`driver`/`nvrtc`/`dynamic-loading`/`cuda-13000`/`f16`） | MIT OR Apache-2.0 | 可 | a, d, e（`docs/spec/01-brainstorm.md:181`）。ドライバ FFI の自作は unsafe 面積増のみで差別化にならない |
| Metal | `objc2` | `=0.6.4` | MIT | 可 | a, d, e |
| Metal | `objc2-foundation` | `=0.3.2` | MIT | 可 | a, d, e |
| Metal | `objc2-metal` | `=0.3.2` | Zlib OR Apache-2.0 OR MIT | 可 | a, d, e。三重ライセンスのうち MIT を選択すれば直接依存 8 区分の適合基準（MIT OR Apache-2.0 系）と両立する |
| 相互運用 | `safetensors` | `=0.7.0` | Apache-2.0 | 可 | c, e（`docs/spec/01-brainstorm.md:183`）。ワイヤフォーマット処理のみに使用（テンソルへのマッピングは自作） |
| 相互運用 | `prost` | `=0.14.4` | Apache-2.0 | 可 | c（`docs/spec/01-brainstorm.md:184`）。protobuf デコードのみ。`prost-build`（`protoc` ビルド時依存）は使わない |
| シリアライズ | `serde` | `=1.0.229`（`derive`） | MIT OR Apache-2.0 | 可 | a。構造化データのシリアライズ |
| シリアライズ | `serde_json` | `=1.0.151` | MIT OR Apache-2.0 | 可 | a |
| CPU 並列 | `rayon` | `=1.12.0` | MIT OR Apache-2.0 | 可 | c（`docs/spec/01-brainstorm.md:185`）。PoC-v2-1 で naive/blocked 比 約 6〜8.5 倍改善を実測 |
| 数値型 | `half` | `=2.7.1` | MIT OR Apache-2.0 | 可 | a |
| ベンチ | `criterion` | `=0.8.2` | Apache-2.0 OR MIT | 可 | c。`dev-dependencies` 限定（deps-policy.md） |

すべて MIT OR Apache-2.0 系（`objc2-metal` のみ三重ライセンス）であり、商用配布・改変・再頒布に適合する。

## 3. feature 組合せの実測範囲の定義

「有効化しうる feature 組合せ」を推定で除外しないため、まず組合せの軸そのものを実測で確定する。

- **自作 9 クレートは `[features]` を宣言していない**（cfg ベースのバックエンド切替方針。PoC-v2-5。`.claude/rules/coding-rust.md`）。実測:

  ```
  $ grep -rn '\[features\]' crates/*/Cargo.toml
  （該当なし）
  ```

- **依存側の feature は `[workspace.dependencies]`（`Cargo.toml:36-` 以降）で固定**されており、member（各 crate）側からは変更できない（`default-features = false` を workspace 側で確定させている理由も同ファイルに記載）。よって「有効化しうる組合せ」を member 側で自由に選べる余地はない
- 上記 2 点により、実測すべき組合せ軸は次の 3 つに帰着する（この帰着自体を実測で裏付ける。推定ではない）:
  1. **ターゲット triple 差**: `objc2`・`objc2-foundation`・`objc2-metal` は `cfg(target_os = "macos")` 限定（deps-policy.md）。Linux 系ターゲットでは依存グラフに現れないことを実測する
  2. **dev エッジ有無**: `criterion` は `bench-harness` の `dev-dependencies` 限定。通常ビルドの依存グラフに含まれないことを実測する
  3. **Cargo.lock 全域**: `cargo deny check`・依存禁止検査（deps-forbidden）は Cargo.lock 全体を対象とするため、上記 1・2 の和集合を上限とする全パッケージ集合をライセンス集計の対象とする（5 節）

## 4. 推移的依存の `cargo tree` 実測記録

実測環境・コミット SHA は 7 節を参照。すべて `--locked` を付与し Cargo.lock を書き換えていない。

| # | コマンド | 結果要約 |
|---|---------|---------|
| 1 | `cargo tree --locked --workspace -e normal,build --target x86_64-unknown-linux-gnu` | 全 93 行。パッケージ集合は #2 と完全一致（`diff` で確認）。`objc2` 系 0 件 |
| 2 | `cargo tree --locked --workspace -e normal,build --target aarch64-unknown-linux-gnu` | #1 と `diff` して差分なし（DGX Spark GB10 実機系列も同一構成） |
| 3 | `cargo tree --locked --workspace -e normal,build --target aarch64-apple-darwin` | 全 113 行。`objc2`/`objc2-foundation`/`objc2-metal`/`objc2-core-foundation`/`dispatch2` 系が新規出現（`grep -c objc2` で 12 件）。他は #1 と共通 |
| 4 | `cargo tree --locked --workspace -e normal,build,dev` | #1 に加え `bench-harness` の `[dev-dependencies]` として `criterion v0.8.2` 以下のサブツリー（`alloca`・`clap`・`plotters`・`regex` 等）が追加出現。コピーレフトライセンスの新規混入なし |
| 5 | `cargo metadata --locked --format-version 1` によるライセンス集計 | 5 節参照。Cargo.lock 全域（上記 1〜4 の上限集合）を対象とする |
| 6 | `cargo tree --locked -p cudarc -e normal,build`（イシュー #412。`cuda-13000` feature 込みの現行 `[workspace.dependencies]` 構成） | `cudarc v0.19.8` の直接依存は `half v2.7.1`・`libloading v0.9.0` の 2 件のみ（`cuda-13000` は API バージョン feature でありサブツリーの依存パッケージを追加しない）。推移的依存（`num-traits`・`rand`／`rand_chacha`／`rand_core`・`rand_distr`・`zerocopy`／`zerocopy-derive`・`getrandom`・`libc`・`cfg-if`・`libm`・`autocfg`・`proc-macro2`・`quote`・`syn`・`unicode-ident`・`ppv-lite86` 等）はすべて 5 節のライセンス分布（MIT OR Apache-2.0 系・許諾的ライセンス）の範囲内。コピーレフト（GPL/LGPL/MPL）の新規混入なし |

いずれも実行時に `Cargo.lock` が変更されていないこと（`git status` で無差分）を確認済み。

## 5. `cargo metadata --locked` によるライセンス集計（Cargo.lock 全域）

対象は自作 9 クレートを含む全 113 パッケージ（外部 104・自作 9）。外部 104 パッケージの `source` は
すべて `registry+https://github.com/rust-lang/crates.io-index`（crates.io 限定。git 依存・代替レジストリなし）。

外部 104 パッケージのライセンス分布:

| ライセンス式 | 件数 |
|-------------|------|
| MIT OR Apache-2.0 | 58 |
| MIT | 14 |
| Apache-2.0 OR MIT | 6 |
| Apache-2.0 | 6 |
| MIT/Apache-2.0 | 4 |
| Unlicense OR MIT | 3 |
| Zlib OR Apache-2.0 OR MIT | 3 |
| Unlicense/MIT | 2 |
| Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | 2 |
| BSD-2-Clause OR Apache-2.0 OR MIT | 2 |
| Zlib | 1 |
| ISC | 1 |
| MIT OR Apache-2.0 OR LGPL-2.1-or-later | 1 |
| (MIT OR Apache-2.0) AND Unicode-3.0 | 1 |

**GPL / LGPL / MPL を含むライセンス式の全件**（`license` フィールドの部分一致検索）:

- `r-efi v5.3.0`: `MIT OR Apache-2.0 OR LGPL-2.1-or-later` の 1 件のみ。個別判断は 6 節を参照
- 上記以外に GPL・LGPL・MPL を含む式は **0 件**（コピーレフトの単独ライセンス・AND 結合による強制混入は検出されず）

## 6. 個別判断の記録

- **`r-efi v5.3.0`**（`getrandom` の UEFI ターゲット向け依存。`MIT OR Apache-2.0 OR LGPL-2.1-or-later`）: OR 選択のライセンス式であり、MIT を選択すれば LGPL の条件（動的リンク・ソース開示義務等）を負わない。本リポの対象ターゲット（Linux／macOS／CUDA／Metal）では UEFI 経路が到達しないため実質的な影響もない。MIT 選択により適合と判断する
- **`unicode-ident v1.0.24`**（`(MIT OR Apache-2.0) AND Unicode-3.0`）: AND 結合だが、Unicode-3.0 は Unicode Character Database（データファイル）由来の許諾的ライセンスであり、商用利用・改変・再頒布を妨げない（帰属表示のみ要求）。適合と判断する
- **`libloading v0.9.0`**（ISC）: MIT 相当の短文許諾ライセンス。適合と判断する
- **`foldhash v0.2.0`**（Zlib）: 許諾的ライセンス。適合と判断する
- **`objc2-metal`/`objc2-core-foundation`/`dispatch2`**（Zlib OR Apache-2.0 OR MIT）: 2 節の `objc2-metal` と同様、MIT を選択すれば適合する
- **`memchr`等**（Unlicense OR MIT）: MIT を選択すれば適合する
- **`zerocopy v0.8.56`**（BSD-2-Clause OR Apache-2.0 OR MIT）: MIT を選択すれば適合する
- **`wasip2`等**（Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT）: MIT を選択すれば適合する（LLVM 例外条項の受諾は不要）

## 7. 実測環境の記録

- 実測日: 2026-08-06
- `rustc 1.96.0 (ac68faa20 2026-05-25)` / `cargo 1.96.0 (30a34c682 2026-05-25)`
- 対象 `Cargo.lock` のコミット SHA: `63748a677a0800257c8175b8fb8bbf187669cbe1`（origin/main）
- 実測コマンドはすべて worktree ルート（`Cargo.toml` と同階層）で `--locked` 付きで実行し、実行後 `git status` で `Cargo.lock` に差分がないことを確認した

### 追加実測（イシュー #412・`cuda-13000` feature 整合。4 節 #6）

- 実測日: 2026-08-09
- `rustc 1.96.0 (ac68faa20 2026-05-25)` / `cargo 1.96.0 (30a34c682 2026-05-25)`
- 対象 `Cargo.lock` のコミット SHA: `ef221531ec5710c973a26f52f165386117089a31`（origin/main）
- `cargo tree --locked -p cudarc -e normal,build` 実行後 `git status --porcelain` で `Cargo.lock` に差分がないことを確認した

### 追加実測（イシュー #425・zerocopy 記録乖離の解消。6 節）

- 実測日: 2026-08-09
- `rustc 1.96.0 (ac68faa20 2026-05-25)` / `cargo 1.96.0 (30a34c682 2026-05-25)`
- 対象 `Cargo.lock` のコミット SHA: `65dea84463472db78ab5dfcb7205b69cf43f4c1b`（origin/main）
- `cargo metadata --locked --format-version 1` で `zerocopy`／`zerocopy-derive` の `version`・`license` を抽出し、両パッケージとも `v0.8.56`・`BSD-2-Clause OR Apache-2.0 OR MIT`（6 節記載の `v0.8.55` から**バージョンのみ更新、ライセンス式は不変**）であることを確認した。実行後 `git status --porcelain Cargo.lock` で差分がないことを確認した（依存・バージョン自体は変更していない）

## 8a. 本表（1〜8 節）の直接の走査対象外にある第 9 区分の監査（OSS 直接比較ハーネス。イシュー #755）

`scripts/bench/oss-gemm-compare/`（本体 workspace 外の独立 Cargo プロジェクト）の
`matrixmultiply`・`gemm` crate は、許容依存第 9 区分（ベンチ比較対象）として
2026-08-20 に条件付きユーザー承認済みの依存であり、本表 1〜8 節・`deny.toml`
（ルート）の走査対象（本体 workspace の依存グラフ）には含まれないが、監査対象
外の例外ではなく本節（8a）と 9 節で正式に統制する（詳細は 9 節を参照）。適用範囲の
定義・ユーザー承認条件は `.claude/rules/deps-policy.md`「許容依存 9 区分」表の
第 9 区分の行（PR #772 で先行して整備）を正とし、本節では二重管理しない。allow
リストの実体は本表と二重管理せず `scripts/bench/oss-gemm-compare/deny.toml`
冒頭コメントを参照する。設計判断の詳細は `docs/oss-comparison-harness-decision.md`
（イシュー #755）を参照。

### 8a-1. 実測（イシュー #755 review 指摘対応。`matrixmultiply`・`gemm` の実ライセンス）

- 実測日: 2026-08-20
- 対象 `Cargo.lock`: `scripts/bench/oss-gemm-compare/Cargo.lock`
- `cargo deny --manifest-path scripts/bench/oss-gemm-compare/Cargo.toml --locked check --config scripts/bench/oss-gemm-compare/deny.toml licenses sources` の実行結果: `licenses ok, sources ok`
- `cargo metadata --manifest-path scripts/bench/oss-gemm-compare/Cargo.toml --format-version 1` で直接依存 2 crate のライセンス式を抽出（推定記載ではなく実測値）:

| crate | version | license（`cargo metadata` 実測） |
|-------|---------|-----------------------------------|
| `matrixmultiply` | 0.3.11 | `MIT/Apache-2.0` |
| `gemm` | 0.19.0 | `MIT` |

いずれも deny.toml の `[licenses] allow` リスト（MIT・Apache-2.0 等。本表 2 節と同一方針）の範囲内であることを `cargo deny check licenses` が機械検査済み。推移的依存（`gemm-common`・`gemm-f32`・`pulp`・`dyn-stack` 等）を含む全域監査は同コマンドの `sources` 検査と合わせて CI（`ci.yml` の `deps-forbidden` ジョブ「OSS 直接比較ハーネスのライセンス監査」ステップ）で毎回再実行し、本表への転記のみに依拠しない（cargo-deny の fail-closed 機械検査が一次情報源）。

### 8b. 第 9 区分の適用範囲拡張の監査（フレームワーク横並びベンチ。PR #915）

`scripts/bench/framework-compare/`（本体 workspace 外の独立 Cargo workspace）の
`burn`・`candle-core`・`fandhe-ai`（crates.io 公開版の自社クレート）は、許容依存
第 9 区分（ベンチ比較対象）の適用範囲拡張として 2026-08-28 にユーザー承認済み
（承認記録・設計判断は `docs/framework-compare-harness-decision.md`）。適用範囲の
定義は `.claude/rules/deps-policy.md`「許容依存 9 区分」表を正とし本節では二重管理
しない。allow リストの実体は `scripts/bench/framework-compare/deny.toml` 冒頭
コメントを参照する（比較対象の推移的依存に含まれる MPL-2.0・CC0-1.0・BSL-1.0 を
**本 workspace 限定**で許容。ルート `deny.toml`・本表 2 節の適合基準は変更しない）。

実測（2026-08-28）:

- 対象 `Cargo.lock`: `scripts/bench/framework-compare/Cargo.lock`
- `cargo deny --manifest-path scripts/bench/framework-compare/Cargo.toml --locked check --config scripts/bench/framework-compare/deny.toml advisories bans licenses sources` の実行結果: `advisories ok, bans ok, licenses ok, sources ok`
- `cargo metadata --manifest-path scripts/bench/framework-compare/Cargo.toml --format-version 1 --locked` で直接依存 3 crate のライセンス式を抽出（推定記載ではなく実測値）:

| crate | version | license（`cargo metadata` 実測） |
|-------|---------|-----------------------------------|
| `burn` | 0.21.0 | `MIT OR Apache-2.0` |
| `candle-core` | 0.11.0 | `MIT OR Apache-2.0` |
| `fandhe-ai` | 0.3.0 | `MIT OR Apache-2.0` |

推移的依存を含む全域監査は CI（`ci.yml` の `deps-forbidden` ジョブ
「フレームワーク横並びベンチの依存監査」ステップ）で毎回再実行し、本表への転記
のみに依拠しない（cargo-deny の fail-closed 機械検査が一次情報源）。同 workspace の
`Cargo.lock` は禁止リスト grep の対象外だが、`scripts/check-forbidden-deps.sh
lock-all` の専用契約検査（存在・`[workspace]` 隔離・承認済みピンのドリフト検出）が
fail-closed で適用される。

## 8. 運用

- 依存の追加・更新は本表の更新とセットで行う（**ユーザー承認必須**。REQ-5・deps-policy.md）
- `deny.toml` の `[licenses]` allow リストの変更（新規ライセンス式の許可）も同じ承認フローに従う。allow リストは本表 5 節の実測結果をそのまま運用化したものであり、単独での緩和は行わない
- MPL-2.0 等コピーレフト混入の監視は CI の `deny` ジョブ（`cargo deny --locked check licenses sources`）で継続する
- `deny.toml` の `[licenses]` は `include-dev = true` を明示する。既定値（`false`）のままだと `criterion`（`bench-harness` の `dev-dependencies` 限定）とその推移的依存サブツリーがライセンス監査から漏れ、本表が前提とする「Cargo.lock 全域」（3 節・4 節 #4）の実測スコープと不整合になる（PR #211 Bugbot 指摘）

## 9. 第 9 区分（ベンチ比較対象。OSS 直接比較ハーネス。イシュー #755）

`matrixmultiply`・`gemm` は `.claude/rules/deps-policy.md`「許容依存 9 区分」表の
第 9 区分（ベンチ比較対象）として正式に許容された依存であり、監査対象外の例外
ではなく、`scripts/bench/oss-gemm-compare/`（`[workspace]` を空テーブルで持つ独立
Cargo プロジェクト）限定で正式に統制される依存として扱う。本表 2 節「直接依存
8 区分」は本体 workspace（ルート `Cargo.toml`／`Cargo.lock`）の直接依存のみを
指し、第 9 区分は別枠として本節で扱う。

本表 4〜5 節（`cargo tree`／`cargo metadata` 実測）はルート `Cargo.lock` を対象と
するため、この独立プロジェクトの依存グラフには及ばない。そのため第 9 区分の
ライセンス監査は、本パッケージ専用の `scripts/bench/oss-gemm-compare/deny.toml`
（allow リストは本表 2 節と同一方針）を用い、CI（`ci.yml` の `deps-forbidden`
ジョブ）で `cargo deny --manifest-path scripts/bench/oss-gemm-compare/Cargo.toml
--locked check --config scripts/bench/oss-gemm-compare/deny.toml licenses sources`
を実行することを必須条件とする（`.claude/rules/deps-policy.md` 第 9 区分の行を
参照）。同ハーネスの `Cargo.lock` は依存禁止リスト検査（`scripts/check-forbidden-deps.sh`）
の走査対象にも含める。

**本節時点では `scripts/bench/oss-gemm-compare/` はリポジトリに未追加**であり、
上記の CI 監査ステップ・依存禁止リスト検査の対象化は、第 9 区分を実際に導入する
PR（イシュー #755・PR #770）がマージされる際の必須条件として課す。`matrixmultiply`・
`gemm` の実ライセンス実測値は、PR #770 で記録される
`docs/oss-comparison-harness-decision.md`（イシュー #755）を出典として参照する
（本節では転記しない）。allow リストの実体は本表と二重管理しない
（`scripts/bench/oss-gemm-compare/deny.toml` 冒頭コメント参照）。
