# OSS 直接比較ハーネスの恒久化: 設計判断記録（イシュー #755）

## 背景

2026-08-19 に scratchpad の使い捨てハーネスで、GEMM 第 2 次最適化ツリー
（#735。Phase 4 親 #754）の目標「既存ライブラリ（OSS）を上回る」の進捗確認として
CPU（自作 vs `matrixmultiply`・`gemm` crate）・Metal（自作 vs MLX ≈ PyTorch MPS）の
直接比較を実施した。ハーネスが使い捨てだったため、#735 の各 Phase 完了時に同条件で
再計測する手段が失われていた。本イシューはこの比較手段を恒久化する。

## 中核の制約

`matrixmultiply`・`gemm` crate は許容依存第 9 区分（ベンチ比較対象。
`.claude/rules/deps-policy.md`）として条件付きユーザー承認済み（2026-08-20。
詳細は本節末尾「スコープ解釈のユーザー承認」）である。本体 workspace（ルート
`Cargo.toml` / `Cargo.lock`）へ dev-dependencies としても追加しない
（deps-policy.md が定める本体 workspace の統制対象を汚さないため。第 9 区分の
承認条件そのもの）。

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
  走査対象）に一切現れないため、実装当初は `.claude/rules/deps-policy.md` が定める
  「許容依存 8 区分」の統制範囲（本体 workspace）の対象外であり、ユーザー承認対象の
  依存追加には当たらないと実装 Agent 自身の推論で判断していた（後述のとおり、この
  スコープ解釈は 2026-08-20 にユーザーの条件付き承認を経て許容依存第 9 区分
  〈ベンチ比較対象〉として正式化されている。現在の扱いは deps-policy.md 該当節が
  正）
- ただし依存禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・`ndarray`。
  deps-policy.md）の混入検査（`scripts/check-forbidden-deps.sh`）は、統制対象外の
  独立パッケージであっても fail-closed 検査の網を広げる方向の変更であり緩和では
  ないため、CI（`deps-forbidden` ジョブ）へ検査対象として追加した
  （`.github/workflows/ci.yml` の該当ステップ参照）

**スコープ解釈のユーザー承認（2026-08-20 取得済み）**: 上記「本体 workspace の
統制範囲外＝ユーザー承認対象の依存追加に当たらない」というスコープ解釈自体は、
実装 Agent 自身の推論による判断であったため当初はユーザーの明示的な追認を
得ていなかったが、2026-08-20 にユーザーが以下の条件付きで承認した:

1. 本体 workspace（ルート `Cargo.toml` / `Cargo.lock`）への `matrixmultiply`・
   `gemm` crate の混入は引き続き禁止する
2. ライセンス監査（`cargo-deny` 相当）を CI に組み込む（本節「ライセンス監査」
   参照。従来の手動実行から `.github/workflows/ci.yml`
   `deps-forbidden` ジョブへのステップ追加へ切り替えた）

以後、本パッケージへの依存追加・変更は上記条件（本体 workspace 非混入・
CI ライセンス監査の維持）を満たす限りにおいて許容される。

**規約側への反映（codex-review P1 指摘対応。先行 PR #772）**: 当初は本承認を
このハーネス実装 PR（#770）自身の中で規約（`.claude/rules/deps-policy.md`・
`AGENTS.md`・`.github/codex/prompts/review.md`）へ直接反映していたが、codex-review
から「例外化の恩恵を受ける PR 自身が同一 PR 内でレビュー基準を書き換えるのは
enforcement の弱体化であり、独立に審査できる先行 PR へ分離すべき」との P1 指摘
（未解決スレッド `PRRT_kwDOTuUCJc6ar3-Q`・`PRRT_kwDOTuUCJc6ar3-W`）を受けた。
指摘は正当と判断し、規約側の変更は先行 PR #772（`docs/755-oss-deps-policy-exception`
ブランチ）へ分離した。本 PR（#770）は #772 マージ後に確定した基準の下でハーネス
実体（本ファイル・実装・CI ステップ・実測記録）のみを追加する。適用範囲・承認条件の
正本は `.claude/rules/deps-policy.md`「適用範囲（本体 workspace 限定）」節であり、
本節では二重管理しない。

### ライセンス監査（専用 deny.toml + CI 組み込み。イシュー #755 review 指摘対応）

本体 `deny.toml` は `cargo-deny` の走査対象をルート workspace の `Cargo.lock` に
限るため、本パッケージ配下の推移的依存（`gemm-f32`/`f64`/`c32`/`c64`/`f16`・
`pulp`・`dyn-stack` 等）は本体 CI のライセンス監査の対象外になる。この監査欠落を
埋めるため、本パッケージ直下に専用の `scripts/bench/oss-gemm-compare/deny.toml`
を新設した。`[licenses] allow` 一覧は本体 `deny.toml` と同一
（MIT・Apache-2.0・Apache-2.0 WITH LLVM-exception・ISC・Zlib・Unicode-3.0・
Unlicense・BSD-2-Clause）、`[sources]` も crates.io 限定（`unknown-registry` /
`unknown-git` を `deny`）で本体と同一方針とする。本パッケージ自身の `Cargo.toml`
に `license = "MIT OR Apache-2.0"`（ルート `Cargo.toml` の `[workspace.package]
license` と同一値）を明示し `unlicensed` 警告も解消済み。

**CI 組み込み（2026-08-20・ユーザー承認条件 (2) 対応）**: `.github/workflows/ci.yml`
の `deps-forbidden` ジョブに「OSS 直接比較ハーネスのライセンス監査」ステップを
追加し、`cargo deny --manifest-path scripts/bench/oss-gemm-compare/Cargo.toml
--locked check licenses sources` を毎 CI 実行で走らせる（既存ジョブへのステップ
追加でありジョブ追加ではないため ruleset の required contexts 更新は不要。
`.claude/rules/ci.md`）。ローカル実測（`cargo metadata --locked` 実測、104
パッケージ相当の推移的依存を含む本体側と同水準の網羅性）でも `licenses ok,
sources ok` を確認済み（2026-08-20 実測。MPL-2.0 等コピーレフトの混入なし）。

**監査範囲の advisories/bans への拡張（PR #770 review 指摘 P1 対応・2026-08-20
ユーザー承認）**: 上記の初期実装は監査範囲をライセンス（licenses/sources）限定と
し、advisories（RUSTSEC アドバイザリ）・bans（重複バージョン・ワイルドカード）を
本体 `deny.toml`（#353）と非対称に除外していた。PR #770 レビューで
「専用 `deny.toml` に advisories/bans がない」との P1 指摘を受け、監査範囲を
本体 `deny.toml` と同一方針（advisories/bans/licenses/sources の 4 種）へ拡張する
方針をユーザーが承認した。`scripts/bench/oss-gemm-compare/deny.toml` に
`[advisories]`・`[bans]` セクションを追加し、`.github/workflows/ci.yml` の
「OSS 直接比較ハーネスの依存監査」ステップを
`cargo deny check advisories bans licenses sources` へ拡張した。

advisories 実測で `RUSTSEC-2024-0436`（`paste` unmaintained）が検出された。
`paste`（手続き型マクロ crate）は本パッケージの推移的依存 `gemm =0.19.0`
（OSS 比較対象の 3 実装の 1 つ）経由の固定バージョン依存であり、`gemm` 側が
`paste` を安全な代替へ差し替えた新版は本 PR 時点で存在しない（アップグレード先
なし）。同アドバイザリは vulnerability ではなく unmaintained（情報提供型）の
分類であるため、`[advisories] ignore = ["RUSTSEC-2024-0436"]` として理由コメント
付きで ignore 登録することを 2026-08-20 にユーザーが承認した
（`scripts/bench/oss-gemm-compare/deny.toml` の該当コメント参照。解消予定は
「`gemm` が `paste` 依存を解消した新版をリリースした時点で再評価」とし、追跡
Issue は本 PR 時点で未起票〈起票の要否は別途ユーザー確認〉）。

bans 実測（`multiple-versions = "warn"`・`wildcards = "deny"`）で違反 0 件を
確認したのは `allow-wildcard-paths = true` を明示したうえでの結果である。
本パッケージは本体 workspace member（`backend-cpu`・`bench-harness`。いずれも
`publish = false`）へ version 無指定の path 依存を持ち、これらは cargo 上 `*`
扱いになるため、`allow-wildcard-paths` を指定しないと wildcard エラーとして
検出される（ローカル実測で確認済み）。ルート `deny.toml` が同じ理由（workspace
内 path 依存の scope）で同一設定を持つのと同一方針であり、`wildcards = "deny"`
自体の意図（`=x.y.z` 完全固定の未指定バージョン検出）を弱めるものではない。

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

**方針（イシュー #755 review 指摘対応で再改定。P1/P2「既定 fail-open」指摘への対応）**:
当初は上記の実測を理由に既定を非 fatal（`--strict-compare` opt-in で fail-closed）と
していたが、これは性能比較の前提となる正しさの検証を既定で無効化する fail-open な
設計であり、review で P1/P2 として指摘された。既定を次のとおり fail-closed に改めた:

- 既定（唯一の挙動。opt-in フラグは廃止）: 全サイズの計測・JSON Lines 出力を終えた
  うえで、突合 NG を 1 件でも検出していれば非 0 終了する
- 各レコードの `output_match`／`mismatch_detail` フィールドへの記録・標準エラー
  出力への警告は従来どおり行う（非 0 終了と併用。性能値そのものは exit code に
  かかわらず JSON Lines から参照できる）

大きい K（1024〜4096）での既知の限界（実装バグではない縮約順序差由来の丸め誤差
蓄積）を理由に既定挙動を非 fatal へ戻すことはしない。「性能値と併せて既知の限界を
確認する」運用（`--sizes` で対象サイズを絞る、`mismatch_detail` を個別に確認する等）
は利用側に委ねる。この変更は許容誤差の数値自体（REL_TOL=1e-3・ABS_TOL=1e-5）を
一切変更していない。実装は `scripts/bench/oss-gemm-compare/src/main.rs` を参照。

## 将来の別リポジトリ移行の条件

本パッケージが以下のいずれかに該当する規模になった場合、別リポジトリへの切り出しを
ユーザーへ提案する:

- 比較対象 OSS ライブラリが著しく増え、本体リポジトリの clone サイズ・ビルド時間への
  影響が無視できなくなった場合
- 独自の CI（OSS 側バージョン追従の自動検知等）が必要になった場合
