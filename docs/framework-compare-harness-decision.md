# フレームワーク横並びベンチ（framework-compare）の設計判断・承認記録（PR #915）

`scripts/bench/framework-compare/` に fandhe-ai / candle / Burn の横並びベンチ
（GEMM・MLP 学習・推論）を恒久化する際の、許容依存第 9 区分（ベンチ比較対象）の
適用範囲拡張に関する設計判断とユーザー承認の記録。
`docs/oss-comparison-harness-decision.md`（イシュー #755。`matrixmultiply`・`gemm` の
第 9 区分導入）と同型の統制を、依存禁止リスト掲載クレートを比較対象として含む
ケースへ拡張する。

## 1. 目的と位置づけ

- 目的: fandhe-ai（crates.io 公開版 `fandhe-ai =0.6.0`。2026-09-02 に crates.io 公開済み
  〈`docs/crates-io-publishing-order.md` §10 追補〉。v0.6.0 リリースサイクルで `=0.5.0` から更新）を、既存 ML フレームワーク
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
  `fandhe-ai =0.6.0`）で、`Cargo.lock` をコミットして再現性を確保する
- 同 workspace の `Cargo.lock` は比較対象という性質上、依存禁止リストのクレート
  （`burn-*`・`candle-*`・`cubecl`・`ndarray`・`tch` 等の推移的混入を含む）を
  **意図的に含む**。このため禁止リスト grep（`check_lock`）は適用せず、代わりに
  `scripts/check-forbidden-deps.sh lock-all` が**専用の fail-closed 契約検査**
  （`check_framework_compare`）を毎回実行する:
  1. `Cargo.lock` の存在（不在はエラー）
  2. `Cargo.toml` の独自 `[workspace]` 宣言（本体 workspace への構造的非混入）
  3. 承認済みピン（burn 0.21.0・candle-core 0.11.0・fandhe-ai 0.6.0）の存在
     （承認外バージョンへのドリフト・比較対象の削除を検出。加えて各エントリが
     `source = "registry+https://github.com/rust-lang/crates.io-index"` を
     伴うことを要求する＝path/git 依存への差し替えで `source`/`checksum` 行が
     消える・書き換わるケースを fail-closed に検出する。イシュー #982）
  4. 各メンバー crate の `[dependencies]` が承認済み allowlist（比較対象の
     `=x.y.z` 完全固定 + `bench-common` の path 依存のみ）の範囲内であること
     （`tch` を含む allowlist 外の直接依存追加・ドット付きキー宣言・完全固定でない
     バージョン指定に加え、`@=` 付き承認済みエントリが `path`/`git`/`registry`/
     `rev`/`branch`/`tag`/`package` キーで非 registry 取得元へ差し替えられている
     ことを検出。`deny.toml` の `allow-wildcard-paths = true`（`bench-common` 用）
     は path 依存自体を止めないため、この manifest 層検査が承認済み比較対象の
     取得元を守る唯一の防御となる。イシュー #982）
  5. 各 Cargo.toml のセクションヘッダが allowlist の範囲内であること
     （`[dev-dependencies]`・`[build-dependencies]`・`[target.'cfg'.dependencies]`・
     `[dependencies.<crate>]` 等の代替依存宣言経路をセクション単位で遮断）
  6. workspace `members` 宣言が期待値と完全一致すること
  7. ディレクトリ配下の Cargo.toml ファイル集合が契約と一致すること
     （allowlist 未登録の member crate 追加を遮断）
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
- 2026-08-29: ユーザーが #966 ツリーの議論でイシュー #982（`fandhe-ai` 承認ピンを
  crates.io 公開済みの `=0.4.0` へ更新し、非 registry 取得元〈path/git 等〉への
  差し替えを manifest 層・Cargo.lock 層の双方で fail-closed 検出する契約検査強化）
  を承認。ピン更新自体は「承認済みバージョンの更新」で通常どおりユーザー承認が
  必要な変更、検査強化は検出範囲の追加（fail-closed の強化）であり本来承認不要だが、
  同一イシューでまとめて実施したため承認記録も本節にまとめて残す
- 2026-08-31: ユーザーがイシュー #1011（CUDA 都度同期廃止の framework-compare
  実践規模 A/B 計測前提。`docs/perf/cuda-async-sync-removal-framework-compare-ab.md`）
  で `fandhe-ai` 承認ピンを crates.io 公開済みの `=0.5.0`（`release-all.yml` run
  33388884217・tag `v0.5.0` = `a5e465d`）へ更新することを承認
- 2026-09-02: ユーザーが v0.6.0 リリースサイクルの一環として `fandhe-ai` 承認ピンを
  crates.io 公開済みの `=0.6.0`（`release-all.yml` run 33503500987・tag `v0.6.0` =
  `8863b09`）へ更新することを承認

## 5. tch-rs を計測対象に含めない判断

libtorch（C++ 配布物）の導入・リンクが必要で、ベンチ環境の再現性・導入コストが
Rust 純正 3 者比較の目的に見合わないため未計測とする（`results/summary.md` に
「未計測」として明記。数値の捏造はしない）。`tch` crate 自体は candle / burn の
推移的依存として Cargo.lock に現れるが、計測バイナリからは使用しない。

## 6. 結果検証の設計（イシュー #965・#970）

GEMM の結果検証は 2 層防御で構成する。

1. **バイナリ内の縮退ガード**（`bench-common::validate_gemm_checksum`。#965）:
   結果テンソル全要素和（checksum）が 0.0 または非有限なら emit 前に遮断する
   （壊れた計算の実行時間を性能値として記録しない）
2. **要素単位検証**（`bench-common::parity`。#970）: checksum は要素の入れ替わり・
   正負誤差の相殺で偶然一致しうる破損を見逃すため、各反復で結果を FMA 契約の
   参照 GEMM（`GemmReference`。本体 `backend-cpu::parity::matmul_reference_fma`
   と同じ契約の自前実装）と要素単位で突合し、`parity_total`/`parity_fail_count`/
   `parity_max_abs_err`/`parity_max_rel_err` を JSONL に記録する

`summarize.py` はさらに 2 段の突合を行う: (1) checksum のフレームワーク間相互突合
（#965。同一 size の checksum が本体の数値一致契約内で一致するはずという性質を
利用）、(2) 要素単位検証の閾値超過報告（#970。バイナリ側の判定をそのまま集計）。
両者は独立に判定し、該当行は表・データ有効性節で理由を併記する。`--strict` は
両方を対象にする。

**閾値は本体の数値一致契約（`.claude/rules/coding-rust.md`「バックエンド構成」節。
相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）と同値に固定する。変更はユーザー
承認事項**（`.claude/rules/security.md` A08・`crates/backend-cpu/src/parity.rs`
の `RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD` と同じ制約）。ベンチの相互
検証を本体の数値一致契約と異なる基準で判定すると、閾値の意味が本体と乖離し
「ベンチで無効表示だが本体の契約では合格」という混乱を生むため、ベンチ側で
独自に緩めない。

参照実装は f64 累積（真値との差という別の指標になり本体契約と整合しない）でも
結果ダンプ + summarize.py 側突合（N=4096 で 64 MiB/行になりコミット・転送が
非現実的）でもなく、各バイナリが自己完結で計算する自前 GEMM を採用した
（詳細・タイミング実測は README.md「要素単位検証」節を参照）。
