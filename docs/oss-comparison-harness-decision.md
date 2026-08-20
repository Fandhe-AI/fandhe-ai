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

### CLI 引数の扱い（OWASP A03）

`--sizes` 引数は正整数のカンマ区切りのみを受理し、パース失敗・0 以下の値は
即座にエラーメッセージを表示して非 0 終了する（`src/main.rs::parse_sizes`）。
シェル展開・eval は使わない。

## 計測プロトコル・境界の定義

`docs/perf/oss-gemm-comparison-baseline.md` を参照。

## 出力突合とその限界（重要な既知の知見）

3 実装（自作 `gemm_blis_parallel`・`matrixmultiply`・`gemm` crate）の出力 C を、
自作実装を基準に統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
`.claude/rules/coding-rust.md`）で照合し、不一致があれば非 0 終了する
（`.claude/rules/coding-rust.md`「性能下限・最適化の達成を理由に…検査を省略しない」
の精神を出力正しさの検証にも適用する。REQ-8）。

**Linux x86_64 実行環境（本イシュー実装時の smoke run）での実測**: サイズ
64〜256 では突合 pass。サイズ 1024〜4096 では、複合判定の許容誤差をわずかに
（相対誤差 0.1〜0.6% 程度・絶対誤差 1e-5 のオーダーで数割程度）超過する不一致が
検出された（例: size=4096 で `abs_diff=1.597e-5`・`rel_diff=1.26e-3`）。これは
K が大きくなるほど縮小和の縮約順序差（BLIS 5-loop ブロッキング vs
`matrixmultiply`/`gemm` crate 内部の異なるブロッキング）に由来する丸め誤差の
蓄積が、本来「バックエンド間（CPU/CUDA/Metal 全て自作・FMA 契約統一）」向けに
設計された複合判定の許容誤差を狭く超えるケースがあるという実測結果であり、
実装バグではない。

**方針**: `.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差
（tolerance）を単独で緩和しない」に従い、本イシューでは許容誤差を変更しない
（自動運転下ではユーザー承認を取得できないため）。fail-closed のまま維持し、
この既知の挙動（大きい K での OSS 実装間の突合が複合判定をわずかに超えうる）は
`docs/perf/oss-gemm-comparison-baseline.md` に記録した上で、許容誤差の再設計要否
（例: OSS 比較専用の別基準を設けるか）はユーザー承認を得て別イシューで検討する
（`.claude/rules/out-of-scope-tracking.md`）。

## 将来の別リポジトリ移行の条件

本パッケージが以下のいずれかに該当する規模になった場合、別リポジトリへの切り出しを
ユーザーへ提案する:

- 比較対象 OSS ライブラリが著しく増え、本体リポジトリの clone サイズ・ビルド時間への
  影響が無視できなくなった場合
- 独自の CI（OSS 側バージョン追従の自動検知等）が必要になった場合
