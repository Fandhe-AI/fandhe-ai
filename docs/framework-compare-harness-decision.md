# フレームワーク横並びベンチ（framework-compare）の設計判断・承認記録（PR #915）

`scripts/bench/framework-compare/` に fandhe-ai / candle / Burn の横並びベンチ
（GEMM・MLP 学習・推論）を恒久化する際の、許容依存第 9 区分（ベンチ比較対象）の
適用範囲拡張に関する設計判断とユーザー承認の記録。
`docs/oss-comparison-harness-decision.md`（イシュー #755。`matrixmultiply`・`gemm` の
第 9 区分導入）と同型の統制を、依存禁止リスト掲載クレートを比較対象として含む
ケースへ拡張する。

## 1. 目的と位置づけ

- 目的: fandhe-ai（crates.io 公開版 `fandhe-ai =0.11.0`。2026-10-09 に crates.io 公開済み
  〈`docs/crates-io-publishing-order.md` §10 追補〉。v0.11.0 リリースサイクルで `=0.10.0` から更新）を、既存 ML フレームワーク
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
  `fandhe-ai =0.11.0`）で、`Cargo.lock` をコミットして再現性を確保する
- 同 workspace の `Cargo.lock` は比較対象という性質上、依存禁止リストのクレート
  （`burn-*`・`candle-*`・`cubecl`・`ndarray`・`tch` 等の推移的混入を含む）を
  **意図的に含む**。このため禁止リスト grep（`check_lock`）は適用せず、代わりに
  `scripts/check-forbidden-deps.sh lock-all` が**専用の fail-closed 契約検査**
  （`check_framework_compare`）を毎回実行する:
  1. `Cargo.lock` の存在（不在はエラー）
  2. `Cargo.toml` の独自 `[workspace]` 宣言（本体 workspace への構造的非混入）
  3. 承認済みピン（burn 0.21.0・candle-core 0.11.0・fandhe-ai 0.11.0）の存在
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
2026-10-17 に追加した `RUSTSEC-2026-0367`〜`0369`（`zip` の展開時の脆弱性）は
脆弱性型だが、ハーネスが ZIP を展開する経路を持たないため同じく受容する（§4 の
2026-10-17 の記録）。

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
- 2026-09-06: ユーザーがイシュー #1185（ブロッカーとして名指し）に対する指示
  「#1185 公開してください」により v0.7.0 の crates.io 公開と `fandhe-ai` 承認ピンの
  `=0.7.0` への更新を承認（`release-all.yml` の run ID は
  `docs/crates-io-publishing-order.md` §10 追補を参照）。ピン更新理由は v0.7.0 に
  含まれる本体側の性能改善（#1164 CUDA cp.async 多段パイプライン結線・#1208
  `CudaGemmAuto` f16 経路の mma 優先本番有効化・#1167 Metal 転置タイル variant・
  #1174 CPU GEMM 再チューニング結線・#1211〜#1219 学習・推論固定費改善）を
  crates.io 公開版としてフレームワーク横並びベンチの比較対象に反映するため
- 2026-09-10: ユーザーが #1486 完了報告（v0.8.0 の crates.io 公開のユーザー
  承認 2026-09-10）およびイシュー #1487 の承認節に基づき、`fandhe-ai` 承認
  ピンを crates.io 公開済みの `=0.8.0`（`release-all.yml` run 34417008617・
  tag `v0.8.0` = `81411488`）へ更新することを承認。同時に、借用ビュー
  readout API（`Var::host_view`／`Tensor::host_slice`。#1335・#1336・#1337）
  が旧ピン `=0.7.0` に未収録だったため #1438 で導入していた早期停止ガード
  （`bench_fandhe_pin_guard.sh`）を撤去した。ピン更新理由は v0.8.0 に含まれる
  #1269 完了後の引き継ぎ改善（CPU GEMM 2D 動的分配の本番結線・借用ビュー
  readout の既定経路化・CUDA ホストステージング `Pinned` 既定化・Metal
  split-K 本番結線・resident `GradStaging` 読み出し API・CPU 出力並列ゼロ
  埋めの本番既定確定）を crates.io 公開版としてフレームワーク横並びベンチの
  比較対象に反映するため
- 2026-09-17: ユーザー指示「v0.9.0 を出してから、再計測してください」
  （2026-09-17）に基づき、v0.9.0 の crates.io 公開と `fandhe-ai` 承認ピンの
  `=0.9.0` への更新を承認（`.github/workflows/release-all.yml` run
  35136585936・tag `v0.9.0` = `88c79317`。`verify` → environment
  `crates-io-release` 承認 → `publish` の順に success・6 クレート crates.io
  反映確認済み）。ピン更新理由は v0.8.0 公開（2026-09-09）以降の #1570 ツリー
  （spec REQ-9 Tier 1／2 の機能網羅）・同期境界最適化（#1555／#1559／#1563／
  #1566／#1580／#1688／#1689）・VJP 転置入口（#1213／#1214／#1215）・
  reduction カーネル是正（#1893／#1894）を crates.io 公開版としてフレーム
  ワーク横並びベンチの比較対象に反映するため。`git diff v0.9.0..HEAD --
  crates/` は空（facade 差分なし）
- 2026-10-03: ユーザー指示「最新の crate を release し、html の値を測り直して」
  （2026-10-03）に基づき、v0.10.0 の crates.io 公開と `fandhe-ai` 承認ピンの
  `=0.10.0` への更新を承認（`.github/workflows/release-all.yml` run
  37112020053・tag `v0.10.0` = `e8d91bc6`。7 クレート公開。
  `fandhe-ai-onnx-interop` は初回公開）。ピン更新理由は v0.9.0 公開（2026-09-17）
  以降の #2058 ツリー（対応表穴埋め）・Windows ビルド対応（#2481 ツリー）・
  各バックエンドの opt-in 性能改善候補を crates.io 公開版としてフレームワーク
  横並びベンチの比較対象に反映するため。`Cargo.lock` には自社の
  `fandhe-ai-onnx-interop` と第 3 区分の承認済み依存 `prost`・`prost-derive` が
  推移的依存として加わる（外部依存の新規追加はこれのみ。ライセンスは
  `docs/license-matrix.md` 8b）。`git diff v0.10.0..HEAD -- crates/` は空
  （ピン更新時点）
- 2026-10-10: 所有者の明示指示（2026-10-10「ok」）に基づき、v0.11.0 の crates.io
  公開（2026-10-09・`release-all.yml` run 37933003465・tag `v0.11.0` =
  `6b14fdb4`。7 クレート公開）を受けて `fandhe-ai` 承認ピンを `=0.10.0` から
  `=0.11.0` へ更新。`Cargo.lock` の差分は自社 7 クレートの版・checksum のみで、
  外部依存の追加・更新・削除はない（`cargo update -p fandhe-ai --precise
  0.11.0`）。ピン更新理由は v0.10.0 公開（2026-10-03）以降の facade／各バックエンド
  の変更（`jvp`／`jacfwd` の公開を含む）を crates.io 公開版としてフレームワーク
  横並びベンチの比較対象に反映するため。ライセンス・監査は
  `docs/license-matrix.md` 8b（2026-10-10 再実測）。
- 2026-10-11: 所有者の明示指示（2026-10-11「PR #2959 を対応し、マージして下さい」）に
  基づき、dependabot PR #2959 による推移的依存 `rustls` の `0.23.43` → `0.23.45`
  更新（パッチ版。TLS ハンドシェイク検証の強化を含む）を承認。同じ依存解決で
  `tempfile 3.27.0` の `getrandom` 参照が `0.4.3` から既存エントリの `0.3.4` へ
  切り替わる（`tempfile` の要求範囲内。crate の追加・削除はない）。承認済みピン
  （burn 0.21.0・candle-core 0.11.0・fandhe-ai 0.11.0）・`deny.toml`・契約検査は
  変更しない。ライセンス・監査は `docs/license-matrix.md` 8b（2026-10-11 再実測）。
- 2026-10-14: 所有者の明示承認（2026-10-14。framework-compare の `Cargo.lock` の
  `rustls` を 0.23.46 へ上げることのみ。他の依存は変えない）に基づき、PR #3300 による
  推移的依存 `rustls` の `0.23.45` → `0.23.46` 更新（パッチ版）を承認。更新理由は
  rustls 0.23.46 未満を対象とする RustSec 勧告 RUSTSEC-2026-0346〜0349 の公開
  （0.23.46 で修正）。`Cargo.lock` の差分は `rustls` の版・checksum のみで、crate の
  追加・削除や他の依存の更新はない。承認済みピン（burn 0.21.0・candle-core 0.11.0・
  fandhe-ai 0.11.0）・`deny.toml`・契約検査は変更しない。ライセンス・監査は
  `docs/license-matrix.md` 8b（2026-10-14 再実測）。
- 2026-10-17: 所有者の明示承認（2026-10-17。「ハーネス限定で ignore（推奨）」。出典:
  https://github.com/Fandhe-AI/fandhe-ai/pull/3410#issuecomment-6232331383）に基づき、
  `deny.toml` の `[advisories] ignore` に `zip` 9.0.0 未満の展開（extract）API に関する
  3 勧告を追加した（修正版は >= 9.0.0。勧告 ID は版を問わず効く）。ID ごとの内容は次の
  とおりで、各 ID の理由・到達性の根拠・失効条件はいずれも下記の共通節に従う。
  - `RUSTSEC-2026-0367`: ZIP 展開時に SUID/SGID/sticky ビットを安全な権限へマスクせず
    保持する。理由・到達性の根拠・失効条件は共通節を参照。
  - `RUSTSEC-2026-0368`: `ZipArchive::extract` が展開先ディレクトリの外を指す symlink を
    作成する。理由・到達性の根拠・失効条件は共通節を参照。
  - `RUSTSEC-2026-0369`: `ZipStreamReader::extract`／`ZipArchive::extract` が symlink 越しに
    Unix 権限を適用する。理由・到達性の根拠・失効条件は共通節を参照。

  **理由（共通）**: 解決グラフに入る zip は `zip 8.6.0` だけで、比較対象
  `candle-core =0.11.0` の推移的依存。修正版へは承認済みピンを変えない限り更新できない。
  ベンチ専用・非配布のツリーのため受容する。承認済みピン・`Cargo.lock`・ルート
  `deny.toml` は変更しない。

  **到達性の根拠（共通）**: ハーネスと比較対象は勧告対象の展開 API を呼ばない。リポジトリ
  ルート基準の次の手順で再現できる。事前に
  `cargo fetch --manifest-path scripts/bench/framework-compare/Cargo.toml --locked`
  （またはビルド）で依存ソースを取得しておく。2026-10-17 に A〜E を実行して確認した。
  - 0: `C="$(ls -d "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/candle-core-0.11.0/src)"`
    → 1 件のパスが出ること（出なければ B・C は偽陰性になるため先に失敗させる）。
  - A: `grep -rnE '\bzip::|use zip|ZipArchive|ZipStreamReader|\.extract[_(]|::extract[_(]|^zip\b' scripts/bench/framework-compare/bench-*/ --include='*.rs' --include=Cargo.toml`
    → 出力なし（終了コード 1）。ハーネスは zip を直接依存にも使用にも持たない。
  - B: `grep -rnE '\.extract[_(]|::extract[_(]|ZipStreamReader' "$C"`
    → 出力なし（終了コード 1）。`extract_unwrapped_root_dir(` 等も含め展開 API の呼び出しはない。
  - C: `grep -rnE 'zip::|ZipArchive|ZipWriter' "$C" | grep -vE 'ZipArchive::new|ZipWriter::new|zip::(write::FileOptions|CompressionMethod|result::ZipError)'`
    → 出力なし（パイプ末尾の `grep -v` の終了コード 1 が期待値。0 が成功していることが
    前提）。除外せずに実行すると zip の利用は error.rs・npy.rs・pickle.rs の
    `ZipArchive::new`／`ZipWriter::new` と関連型だけ。
  - D: `cargo tree --manifest-path scripts/bench/framework-compare/Cargo.toml --offline --locked -i zip@7.2.0`
    → `nothing to print`（解決グラフに入らない。burn-candle 由来は無効 feature 側）。
  - E: `cargo tree --manifest-path scripts/bench/framework-compare/Cargo.toml --offline --locked -i zip@0.6.6`
    → `nothing to print`（同。burn-tch 由来は無効 feature 側）。`-i zip@8.6.0` は
    candle-core 0.11.0（bench-candle 経由）だけを示す。

  入力は自前生成のテンソルと safetensors のみで外部の ZIP を展開しない。

  **失効条件**: zip の修正版（>= 9.0.0）が candle-core の要求範囲に入った時点、または
  比較対象（candle-core・burn）のピン更新時に、この 3 件の ignore を削除する。

  **版集合の固定**: `scripts/check-forbidden-deps.sh lock-all`（`check_framework_compare`）が
  `Cargo.lock` 上の `zip` の版集合 {0.6.6, 7.2.0, 8.6.0} の完全一致と全エントリの
  crates.io registry 取得元を fail-closed で固定する（追加・削除・変更のいずれも失敗。
  Codex 指摘への対応として 2026-10-17 に所有者承認のうえ追加。self-test に正負ケースあり）。
  別の版が入った場合は ignore を足さず承認を取り直す。

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

## 7. bench-fandhe の非対称是正の範囲（イシュー #3264・親 #3263）

### 7.0 位置づけ

- 承認の出典は D-C（`docs/compat-api-scope.md` の D-C 表「ベンチ fandhe 側の非対称是正」・#3263）である。本節は是正の**範囲の確定**であり、新しい承認ではない。依存の追加・更新もない
- 適用範囲は `scripts/bench/framework-compare/bench-fandhe` に限る。本イシューはドキュメントだけを変え、コードは変えない
- 行番号は変わりやすいため、関数名と ref（tag `v0.11.0`・`origin/main` の `45347ae0a`）を併記する。ベンチは crates.io の 0.11.0 を使うため、crates 側の出典は `v0.11.0` tag 基準とする

### 7.1 是正の範囲と API の区分

| 項目 | 是正内容 | 必要な API | 担当 | 状態 |
|------|---------|-----------|------|------|
| (1) | GEMM reuse の結果を反復ごとに解放する | 現行 0.11.0 | #3265 | 確定 |
| (2) | loss の読み出し位置を step 末尾へ移す | - | - | **採らない**（7.2） |
| (3) | Metal の readout を借用ビューにする | 現行 0.11.0 | #3265 | 条件つき |
| (4) | 入力（GEMM の A・B、train の x・y、infer の x）を計測窓の外で 1 回だけデバイスに置く | 0.12.0（常駐入力 API。#3250） | #3272 | 確定（0.12.0 待ち） |
| (5) | 学習は 0.12.0 の学習ステップ入口を使う | 0.12.0 | #3272（入口の等価性確認は #3255） | 確定（0.12.0 待ち） |
| (6) | 学習の入力を `var_no_grad` で渡す | 書き方は現行、効果は 0.12.0 | #3265 | 条件つき（#3245 が案 2 の場合） |

**(1) GEMM reuse の結果の解放（0.11.0 でできる）**

- 非対称の事実: `run_gemm_reuse`（`bench-fandhe/src/main.rs`。`origin/main` で `:542`）は同じ `Tape` に葉 `a`・`b` を登録し（`:572-573`）、計測窓の中で `a.matmul(&b)` を繰り返す。結果ノードが tape に溜まり続ける。candle は `c` を反復ごとに捨てる（`bench-candle/src/main.rs`。`elapsed` 確定が `:133`、closure 末尾が `:144`）
- train の reuse は step ごとに `make_tape` で作り直し、infer の reuse は `predict_resident` の内部で tape を生成・破棄する。この非対称があるのは GEMM の reuse だけである
- 手段: facade `Tape::reset(&mut self)`・`Tape::leaf(index)`・`Tape::leaf_count()`（`v0.11.0` の `crates/facade/src/lib.rs` で `:472`・`:485`・`:478`）
- 反復の形: `matmul → readout → checksum／parity → c を drop → tape.reset() → a = tape.leaf(0)、b = tape.leaf(1)`（`None` は型付きエラーにする）。葉プレフィックスが 2 なのは、`a`・`b` を最初の matmul より前に登録するためである。`reset` が `&mut self` のため、既存の `one` closure（`tape`・`a`・`b` を借用）は #3265 で組み直す
- **計測窓との位置関係**: 解放（drop・`reset`・`leaf` の再取得）は `elapsed` 確定の後、つまり計測窓の外に置く。candle の `c` の drop が `elapsed` の後にあることとの対称性による。窓に入る処理（matmul・readout・checksum 計算）は変えない
- 同種の経路: `run_gemm_reuse_device_checksum`（`:732`。`mode=reuse` の JSONL を出す本体計測の代替経路）にも同じ処理を入れる。`run_gemm_reuse_phases`（`:976`。`--phases` 診断）も内訳を本体計測と比べられるよう揃える。揃えられない場合は #3265 の PR 本文に理由を書く
- host arena（#2104。opt-in・既定 OFF）が有効なときは `reset` がバッファを回収する。既定構成の計測条件は変わらない

**(3) Metal の借用ビュー（条件つき）**

- 現状: `readout_uses_borrowed_view("metal", None)`（`:281`）は false で、legacy 経路の全面コピー（`to_vec`）が入る
- 経緯: #1520 で REJECT 済み（`docs/perf/metal-gemm-candle-gate-remeasurement.md` §17）。#3184 の再開分（`docs/perf/optin-ab-macmini.md` の G5）は専有ゲートが成立し、継承集計器の判定は ADOPT だったが、同 doc は「#1520 を覆さず、採否は #3265」としている。出典は `docs/perf/logs/optin-ab-macmini-3184/readout-r2/compare-readout.md`
- 本記録では切り替えを確定させない。継承集計器の規則（`docs/perf/metal-gemm-candle-gate-remeasurement.md` §15.1）で ADOPT の場合に限り、条件を変えた再測定として #3265 が切り替える。過去の判定（#1520）は書き換えない。切り替える場合は既存テスト（`readout_uses_borrowed_view_*`・`readout_var_metal_legacy_branch_*`）も更新する

**(4)(5) 0.12.0 の API が要るもの**

- (4) の非対称: 相手は窓の外で 1 回だけデバイスに置く。fandhe は GEMM で `tape.var(&a_data)`、train で毎 step `tape.var(&x_data)`（`:1214-1215`）、infer で `predict_resident(&store, &x_data)`（`:1650`）を呼ぶ。影響が出るのは CUDA と Metal（CPU には H2D がない）
- 常駐入力 API（#3250）と学習ステップ入口は 0.12.0 の公開を待つ。ピン更新（#3271）の後に #3272 が切り替える。同期を 1 回にするのは入口側の仕事であり、(2) を採らない理由（7.2）とつながる

**(6) 入力の勾配省略（条件つき）**

- PyTorch と burn は入力の勾配を計算しない。`Tape::var_no_grad` は `v0.11.0` に既にある（`:440`・`:1527`）。ただし効果を生む `LinearResident` の VJP での `d_input` 省略（#3246）は 0.12.0 で公開されるまで計測に現れない。区分は「書き方は現行 API、効果は 0.12.0」とする
- #3245 は本記録の時点で未確定である。案 2 にならなければ実施しない。実施する場合も `last_loss` が是正前後で一致することを確認する

### 7.2 loss の読み出し位置を移さない理由

- fandhe の train reuse は `loss.to_tensor().get(&[])` を `backward_device_param_store` の前で呼ぶ（`main.rs` `:1219-1223`）
- 同期はそれより前、`mse_loss` の内部で起きる。経路は autodiff `Var::mse_loss`（`v0.11.0` の `crates/autodiff/src/var.rs:1397`）→ `mse_loss_with`（`:1419`）→ `BackendOps::mse_loss`。Metal は `run_mse_loss_f32_tracked` 内の `check_dispatch_token(ctx.synchronize(), ..)`（`crates/backend-metal/src/mse.rs:227`）、CUDA は `run_mse_loss_f32` 内の `readback(..)`（`crates/backend-cuda/src/mse.rs:201`）
- したがって `loss.to_tensor().get()` は計算済みのホスト値を読むだけである。診断でも `loss_readout` は 0.0 µs だった（`docs/perf/cuda-train-forward-resident-diagnosis.md` §3.1）
- 結論: 移しても速くならないため是正に含めない

### 7.3 旧条件の値の出典と新旧併記の方針

- 出典: `docs/perf/logs/framework-compare-0.11.0-macmini-remeasure/`（Mac mini。`macmini-series-a`・`macmini-series-b`・`aggregate/`）と `docs/perf/logs/framework-compare-0.11.0-remeasure/gb10/`（GB10。`results-dgx-0.11.0-median5.jsonl` ほか）。本記録は個別の数値を転記しない（値は出典ファイルから取る）
- 方針: 是正後の計測（#3266・#3272 以降）は旧条件腕と新条件腕を同一ラウンドで並べて記録する。判定規則（同一 run 内の最速相手比）は変えず、旧条件の判定も併記する。旧条件の連続性のため旧条件の行は消さない

### 7.4 変えないもの

- 相手側のコード（`bench-candle`・`bench-burn`・Python 参照計測）
- 判定規則（同一 run 内の最速相手比）
- checksum の定義と `validate_gemm_checksum`、parity 判定（`bench-common::parity`・`verify_strict`）、tolerance 定数（§6。`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）、parity baseline
- 依存ピン（`fandhe-ai =0.11.0`・`candle-core =0.11.0`・`burn =0.21.0`）、`deny.toml`、契約検査
- `fresh` モード（全タスク）

### 7.5 契約検査への影響と実行記録

- `scripts/check-forbidden-deps.sh` の framework-compare 検査は manifest と `Cargo.lock` だけを見るため、`bench-fandhe/src/main.rs` の変更は影響しない
- 実行記録: `origin/main` の `45347ae0a` 上で `bash scripts/check-forbidden-deps.sh lock-all` は rc=0、出力 17 行すべて `OK:`（本変更は docs のみで manifest・lock を変えないため、変更前後で同一）

### 7.6 後続 issue への引き継ぎ

- #3265: (1)・条件つきの (3)・(6)
- #3266: 新旧併記の計測
- #3272: (4)・(5)（#3271 のピン更新後）

### 7.7 実装記録（#3265）

#3265 で §7.1 の (1)・(3)・(6) を `bench-fandhe/src/main.rs` に実装した。変更はこのファイルと docs に閉じ、相手側コード・`bench-common`・`Cargo.toml`・`Cargo.lock` は変えていない。

- (1) GEMM reuse の結果解放: `run_gemm_reuse` と `measure_gemm_reuse_phases`（`--phases`）は、反復の終わりに `Tape::reset` で結果ノードを解放し、次の反復の先頭で `tape.leaf(0)`・`tape.leaf(1)` から `a`・`b` を再取得する。解放と再取得は計測窓の外に置き、窓の中身と phase の 5 区間は変えない。葉の取得失敗と `leaf_count` の不一致は型付き `MEASURE_ERROR`（panic なし）にする。`run_gemm_reuse_device_checksum`（`--device-checksum`）は変えない。`matmul_checksum` は tape にノードを積まないため、蓄積という非対称が無い
- (3) Metal の readout: 条件は成立した。#3184 G5（専有 Mac mini）の `docs/perf/logs/optin-ab-macmini-3184/readout-r2/compare-readout.md` は総合判定 ADOPT（`.exit`=0）。6 セルとも borrowed/legacy が 0.9592〜0.9773 で 1.00 以下、checksum は完全一致、専有ゲートは load1 2.60 < 4.0 を 2 回連続で満たした。`readout_uses_borrowed_view` の Metal 分岐を除去し、`device` 引数を削除した。`--readout legacy|borrowed` の override は残す。#1520 の REJECT は書き換えない（条件を変えた再測定として扱う。`docs/perf/metal-gemm-candle-gate-remeasurement.md` §20）。この関数は infer の Metal 経路にも使われるため、infer の Metal も借用ビューになる。G5 が測ったのは gemm の 6 セルだけだが、§15.1 の処置は「関数の Metal 分岐を除去する」であり、集約した 1 箇所の切り替えを意図している。checksum は同じ順序の `f64` 逐次和で bit 同一。infer の新旧併記は #3266
- (6) train reuse の入力: 条件は成立した（#3245 は案 2 を採用）。`run_train_reuse` と `measure_train_reuse_phases` の入力 `x` を `tape.var_no_grad` に変えた。`y`（mse target）は #3245 §9 のとおり対象外。fresh は変えない。0.11.0 は d_input を計算してから捨てるので速度は変わらず、効果は 0.12.0（#3246）で出る
- #3247 への影響: `docs/perf/logs/dinput-skip-ab-3247/bench-var-no-grad.patch` は `tape.var(&x_data)` を前提にしており、本変更の後のツリーには適用できない（runner の precheck が fail-closed で止める）。RULE.txt は runner ツリーを `a5fa15fd3` に固定しているため運用は変わらない。`dinput-skip-ab-3247/*` は変更していない

検証（正しさの確認のみ。性能値は報告しない）:

- AC1: 変更前（`origin/main`）と変更後のバイナリで、CPU と Metal の `gemm reuse`（512・1024）、`gemm reuse --phases`、`train reuse`、`infer reuse` を実行し、`checksum`・`parity_total`・`parity_fail_count`・`parity_max_abs_err`・`parity_max_rel_err`・`parity_scaled_abs_*` が完全一致した。phases は 5 行のまま。Metal の gemm 1024 は変更前（legacy）・変更後（borrowed）・変更後 `--readout legacy` の 3 者で checksum が一致した。さらに単体テストで、`var_no_grad` と `var` の最終 loss と全パラメータ、reset 前後の matmul 結果と checksum を bit で比較して一致を固定した
- AC3: `scripts/check-forbidden-deps.sh lock-all` は rc=0、`OK:` 行は 17 行で、変更前と同じ
- fmt・clippy（`-D warnings`・`device-checksum` あり／なし）・`cargo test`（bench-fandhe・bench-common）・Mac の `--ignored` smoke 3 件が通る。あわせて、macOS 限定の `#[ignore]` テスト 4 件が `TempOutFile` のメソッド呼び出しの誤りでコンパイルできなかった既存の不具合を `.path()` に直した（Linux では対象外のため CI では見えなかった）
- 未検証: GB10 の CUDA 実機（`reset` は値を変えず、`var_no_grad` は 0.11.0 では計算してから捨てるだけ、という机上の根拠のみ）
- 承認: 本変更は #3264 の範囲（D-C 承認の bench-fandhe 側の是正）内であり、新しい承認は得ていない
