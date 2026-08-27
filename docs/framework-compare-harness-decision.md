# フレームワーク横並びベンチ（framework-compare）の設計判断・承認記録（PR #915）

`scripts/bench/framework-compare/` に fandhe-ai / candle / Burn の横並びベンチ
（GEMM・MLP 学習・推論）を恒久化する際の、許容依存第 9 区分（ベンチ比較対象）の
適用範囲拡張に関する設計判断とユーザー承認の記録。
`docs/oss-comparison-harness-decision.md`（イシュー #755。`matrixmultiply`・`gemm` の
第 9 区分導入）と同型の統制を、依存禁止リスト掲載クレートを比較対象として含む
ケースへ拡張する。

## 1. 目的と位置づけ

- 目的: fandhe-ai（crates.io 公開版 `fandhe-ai =0.3.0`）を、既存 ML フレームワーク
  `candle-core =0.11.0`・`burn =0.21.0` と**同一プロトコル**（同一シード・同一入力・
  同一の同期境界・warmup 20 → 計測 20・中央値 + Q1/Q3）で横並び計測する
- 本 workspace はベンチ専用ツール（全クレート `publish = false`・非配布）であり、
  本体ライブラリの実装・公開 API とは無関係。完全自作コア方針（REQ-1 v2）の
  「既存 ML フレームワークへの統合・放棄」には該当しない（比較対象としての
  計測利用のみ。本体クレートは burn / candle のコードも API も一切使わない）
- 実測記録（`results/summary.md`・raw JSONL・run ログ）は再現に必要な生成物一式と
  してディレクトリ配下にコミットする（`docs/perf/` の実測記録群と同趣旨）

## 2. 依存ポリシー上の統制（第 9 区分の適用範囲拡張）

`.claude/rules/deps-policy.md`「許容依存 9 区分」表の「ベンチ比較対象
（フレームワーク横並び）」行を正とする。要点:

- 適用範囲は `scripts/bench/framework-compare/`（独自の `[workspace]` を持つ独立
  Cargo workspace。本体 workspace 外）**限定**。本体 workspace（ルート
  `Cargo.toml`／`Cargo.lock`）への混入は引き続き禁止で、ルート Cargo.lock・
  `cargo tree` に対する `scripts/check-forbidden-deps.sh` が fail-closed に検出する
- 直接依存は `=x.y.z` 完全固定（`burn =0.21.0`・`candle-core =0.11.0`・
  `fandhe-ai =0.3.0`）で、`Cargo.lock` をコミットして再現性を確保する
- 同 workspace の `Cargo.lock` は比較対象という性質上、依存禁止リストのクレート
  （`burn-*`・`candle-*`・`cubecl`・`ndarray`・`tch` 等の推移的混入を含む）を
  **意図的に含む**。このため禁止リスト grep（`check_lock`）は適用せず、代わりに
  `scripts/check-forbidden-deps.sh lock-all` が**専用の fail-closed 契約検査**
  （`check_framework_compare`）を毎回実行する:
  1. `Cargo.lock` の存在（不在はエラー）
  2. `Cargo.toml` の独自 `[workspace]` 宣言（本体 workspace への構造的非混入）
  3. 承認済みピン（burn 0.21.0・candle-core 0.11.0・fandhe-ai 0.3.0）の存在
     （承認外バージョンへのドリフト・比較対象の削除を検出）
- 専用 `scripts/bench/framework-compare/deny.toml` による依存監査
  （advisories / bans / licenses / sources）を CI（`ci.yml` の `deps-forbidden`
  ジョブ）の必須ステップとして実行する（oss-gemm-compare と同一方式）

## 3. ライセンス監査（実測）

実測値・監査コマンドは `docs/license-matrix.md` 8b 節を参照（直接依存 3 crate は
いずれも `MIT OR Apache-2.0`。`cargo deny check advisories bans licenses sources` が
`ok`）。本 workspace 限定で allow リストへ追加した 3 ライセンスと理由:

| ライセンス | 該当クレート（実測） | 判断 |
|-----------|---------------------|------|
| MPL-2.0 | `colored`（burn 経由）・`option-ext`（dirs 経由） | ファイル単位の弱いコピーレフト。ベンチ実行のみ（改変・再配布なし・非配布ツール）のため受容 |
| CC0-1.0 | `hexf-parse`・`tiny-keccak`（wgpu / cubecl 経由） | パブリックドメイン相当 |
| BSL-1.0 | `xxhash-rust`（burn 経由） | Boost Software License（permissive） |

RUSTSEC ignore（`RUSTSEC-2025-0141` bincode unmaintained・`RUSTSEC-2024-0436`
paste unmaintained）はいずれも情報提供型（脆弱性ではない）で、比較対象の固定
バージョンの推移的依存にアップグレード先がないため受容する（理由コメントは
`deny.toml` に記載。比較対象バージョンを更新する再計測キャンペーン時に再評価）。

ルート `deny.toml`・`docs/license-matrix.md` 2 節の本体 workspace 向け適合基準は
一切変更しない。

## 4. ユーザー承認記録

- 2026-08-28: ユーザー（maintainer）が PR #915 の導入・マージを明示的に指示
  （承認の出典は本ドキュメントと PR #915。deps-policy.md 第 9 区分の
  「フレームワーク横並び」行の承認記録欄が本ドキュメントを指す）
- 承認条件（本 PR で充足済み）: 上記 2 節の統制一式（独立 workspace・完全固定・
  専用契約検査・専用 deny.toml の CI 監査）と 3 節のライセンス実測
- 承認外の変更（ピンの更新・allow リストの拡張・検査の緩和・適用範囲の変更）は
  従来どおりユーザー承認必須

## 5. tch-rs を計測対象に含めない判断

libtorch（C++ 配布物）の導入・リンクが必要で、ベンチ環境の再現性・導入コストが
Rust 純正 3 者比較の目的に見合わないため未計測とする（`results/summary.md` に
「未計測」として明記。数値の捏造はしない）。`tch` crate 自体は candle / burn の
推移的依存として Cargo.lock に現れるが、計測バイナリからは使用しない。
