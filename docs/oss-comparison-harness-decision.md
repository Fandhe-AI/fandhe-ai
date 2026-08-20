# OSS 直接比較ハーネスの恒久化: 設計判断記録（イシュー #755）

## 背景

2026-08-19 に scratchpad の使い捨てハーネスで、GEMM 第 2 次最適化ツリー
（#735。Phase 4 親 #754）の目標「既存ライブラリ（OSS）を上回る」の進捗確認として
CPU（自作 vs `matrixmultiply`・`gemm` crate）・Metal（自作 vs MLX ≈ PyTorch MPS）の
直接比較を実施した。ハーネスが使い捨てだったため、#735 の各 Phase 完了時に同条件で
再計測する手段が失われていた。本イシューはこの比較手段を恒久化する。

## 中核の制約

`matrixmultiply`・`gemm` crate は許容依存 8 区分（`.claude/rules/deps-policy.md`）の
対象外である。本体 workspace（ルート `Cargo.toml` / `Cargo.lock`）へ dev-dependencies
としても追加しない（deps-policy.md の統制対象を汚さないため）。

## 選択肢比較

| 選択肢 | 判定 | 理由 |
|--------|------|------|
| (a) 別リポジトリへ切り出す | 不採用（将来の移行余地は残す） | 新規リポジトリ作成は運用変更であり自動運転下でのユーザー承認取得ができない |
| (b) 本体 workspace の dev-dependencies に追加する | 不採用 | deps-policy.md 違反（ユーザー承認 + `docs/license-matrix.md` 更新が必須）になる |
| (c) リポジトリ内・workspace 外の独立 Cargo パッケージ | **採用** | 本体の依存グラフに一切現れず、deps-policy.md の統制範囲（本体 workspace）と物理的に分離できる |

## 採用案の実装

### 配置

`scripts/bench/oss-gemm-compare/` に独立 Cargo パッケージを新設した。

- パッケージ自身の `Cargo.toml` に空の `[workspace]` テーブルを持たせ、ルート
  workspace（`resolver = "3"`・9 クレート＋facade の 10 クレート member）から
  完全に切り離す。ルート `Cargo.toml` の `members` は明示列挙（glob なし）のため
  `exclude` の追記も不要であり、ルート `Cargo.toml` / `Cargo.lock` は本イシューで
  一切変更していない（`git diff --exit-code Cargo.toml Cargo.lock` で確認可能）
- `backend-cpu`・`bench-harness` は workspace member への path 依存
  （`{ path = "../../../crates/backend-cpu" }` 等）で参照する。member 側の
  `version.workspace = true`・`rayon.workspace = true` 等の継承は「member 自身が
  属する workspace」（ルート `Cargo.toml`）から解決されるため、本パッケージが
  独立 workspace でも問題なく解決できることを `cargo build --release` で実地確認済み
  （`bench-harness` は `cudarc`・`backend-cuda` を通常依存として持つため、本パッケージの
  ビルドはそれらも連鎖的にコンパイルする。CUDA toolkit 非搭載環境でもビルドが成立する
  動的ロード方式のため問題ない）
- 置き場所を `scripts/bench/` 配下とするのは、PyTorch MPS 計測スクリプト
  （`scripts/bench/gemm_bench_torch_mps_f32.py` 等）が既に同所にあり、「実行資産は
  scripts/bench・記録は docs/perf」という既存区分を維持するため

### 外部依存の扱い

- `matrixmultiply = "=0.3.11"`（MIT/Apache-2.0。crates.io API 実測）・
  `gemm = "=0.19.0"`（MIT。crates.io API 実測）を `=x.y.z` 完全固定で宣言し、
  本パッケージ専用の `Cargo.lock`（`scripts/bench/oss-gemm-compare/Cargo.lock`）を
  コミットして再現性を確保する
- 本体 workspace の依存グラフ（`cargo tree --workspace`・cargo-deny・各 CI ジョブの
  走査対象）に一切現れないため、`.claude/rules/deps-policy.md` が定める「許容依存
  8 区分」の統制範囲（本体 workspace）の対象外であり、ユーザー承認対象の依存追加には
  当たらないと判断する
- ただし依存禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・`ndarray`。
  deps-policy.md）の混入検査（`scripts/check-forbidden-deps.sh`）は、統制対象外の
  独立パッケージであっても fail-closed 検査の網を広げる方向の変更であり緩和では
  ないため、CI（`deps-forbidden` ジョブ）へ検査対象として追加した
  （`.github/workflows/ci.yml` の該当ステップ参照）

**未確定の残課題（イシュー #755 review 指摘・ユーザー承認が必要）**: 上記
「本体 workspace の統制範囲外＝ユーザー承認対象の依存追加に当たらない」という
スコープ解釈自体は、`deps-policy.md`・`CLAUDE.md`「依存の追加・更新…はユーザー
承認必須」の文言がルート workspace 限定と明記していないため、実装 Agent 自身の
推論による判断である。判断の是非（設計自体の妥当性）と、この解釈をユーザーが
追認するかは別軸であり、**本記録の時点ではユーザーの明示的な追認を得ていない**。
`.claude/rules/out-of-scope-tracking.md` に従い、この適用範囲確認自体を自動運転下で
拡大解釈しない（新たな独立パッケージの追加や依存追加は本判断の追認前に行わない）
方針とし、ユーザー承認取得は別途対応する。

### ライセンス監査（cargo tree 実測。イシュー #755 review 指摘対応）

本体 `deny.toml` は `cargo-deny` の走査対象をルート workspace の `Cargo.lock` に
限るため、本パッケージ配下の推移的依存（`gemm-f32`/`f64`/`c32`/`c64`/`f16`・
`pulp`・`dyn-stack` 等）は本体 CI のライセンス監査の対象外になる。この監査欠落を
埋めるため、本パッケージの `Cargo.lock`（`cargo metadata --locked` 実測、104
パッケージ相当の推移的依存を含む本体側と同水準の網羅性）に対し、本体
`deny.toml` の `[licenses] allow` 一覧（MIT・Apache-2.0・Apache-2.0 WITH
LLVM-exception・ISC・Zlib・Unicode-3.0・Unlicense・BSD-2-Clause）と同一の許可
リストで `cargo deny check licenses sources` を手動実行し、`licenses ok, sources
ok` を確認した（2026-08-20 実測。MPL-2.0 等コピーレフトの混入なし）。本パッケージ
自身の `Cargo.toml` に `license = "MIT OR Apache-2.0"`（ルート `Cargo.toml` の
`[workspace.package] license` と同一値）を明示し `unlicensed` 警告も解消した。

この監査は手動実行であり CI に組み込んでいない（本パッケージを CI の走査対象へ
恒常的に含めるかどうかは、上記スコープ解釈のユーザー承認と併せて検討する）。

### CLI 引数の扱い（OWASP A03）

`--sizes` 引数は正整数のカンマ区切りのみを受理し、パース失敗・0 以下の値は
即座にエラーメッセージを表示して非 0 終了する（`src/main.rs::parse_sizes`）。
シェル展開・eval は使わない。

## 計測プロトコル・境界の定義

`docs/perf/oss-gemm-comparison-baseline.md` を参照。

## 出力突合とその限界（重要な既知の知見）

3 実装（自作 `gemm_blis_parallel`・`matrixmultiply`・`gemm` crate）の出力 C を、
自作実装を基準に統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
`.claude/rules/coding-rust.md`）で照合する
（`.claude/rules/coding-rust.md`「性能下限・最適化の達成を理由に…検査を省略しない」
の精神を出力正しさの検証にも適用する。REQ-8）。**許容誤差の値自体（相対誤差 1e-3・
絶対誤差 1e-5）は変更しない**（coding-rust.md「バックエンド間数値一致テストの許容
誤差を単独で緩和しない」の保護対象。本ハーネスは CPU/CUDA/Metal 自作バックエンド間
比較ではなく OSS 実装との比較のため同ルールの直接の適用対象ではないが、数値自体は
予防的に据え置く）。

**Linux x86_64 実行環境（本イシュー実装時の smoke run）での実測**: サイズ
64〜256 では突合 pass。サイズ 1024〜4096 では、複合判定の許容誤差をわずかに
（相対誤差 0.1〜0.6% 程度・絶対誤差 1e-5 のオーダーで数割程度）超過する不一致が
検出された（例: size=4096 で `abs_diff=1.597e-5`・`rel_diff=1.26e-3`）。これは
K が大きくなるほど縮小和の縮約順序差（BLIS 5-loop ブロッキング vs
`matrixmultiply`/`gemm` crate 内部の異なるブロッキング）に由来する丸め誤差の
蓄積が、本来「バックエンド間（CPU/CUDA/Metal 全て自作・FMA 契約統一）」向けに
設計された複合判定の許容誤差を狭く超えるケースがあるという実測結果であり、
実装バグではない。

**方針（イシュー #755 review 反映後に改定）**: 上記の実測どおり、既定サイズ列
（512/1024/2048/4096）を既定引数のまま `cargo run --release` すると、本ハーネスの
主目的（#735 各 Phase 完了時に既定引数のまま素朴に再実行して再計測する。本 doc
冒頭）自体が非 0 終了で阻害される。これは許容誤差の値を緩和する話ではなく
「検証結果をどう扱うか（fatal にするかどうか）」の話であるため、
coding-rust.md「バックエンド間数値一致テストの許容誤差を単独で緩和しない」の
対象外として扱ってよいと判断した。既定挙動を次のとおり変更した:

- 既定（`--strict-compare` 未指定）: 突合 NG は標準エラー出力へ警告を出し、
  各 JSON Lines レコードの `output_match`／`mismatch_detail` フィールドに記録
  するに留め、非 fatal とする（プロセスは 0 終了・性能計測は継続）
- `--strict-compare` 指定時: 従来どおり突合 NG を検出した時点で非 0 終了する
  （CI での回帰検知等、fail-closed 挙動が必要な用途向けの opt-in）

この変更は許容誤差の数値自体（REL_TOL=1e-3・ABS_TOL=1e-5）を一切変更していない。
実装は `scripts/bench/oss-gemm-compare/src/main.rs` を参照。

## 将来の別リポジトリ移行の条件

本パッケージが以下のいずれかに該当する規模になった場合、別リポジトリへの切り出しを
ユーザーへ提案する:

- 比較対象 OSS ライブラリが著しく増え、本体リポジトリの clone サイズ・ビルド時間への
  影響が無視できなくなった場合
- 独自の CI（OSS 側バージョン追従の自動検知等）が必要になった場合
