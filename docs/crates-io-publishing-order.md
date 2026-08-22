# crates.io 公開順序と版数運用（#881）

イシュー #881「クレート間依存への version 付与と公開順序の設計」に対応する。
親ツリー #864（GitHub Pages / crates.io 公開トラッキング）→ Phase 2 親 #866 の
sub-issue。公開対象は `fandhe-ai`（facade）と依存連鎖 5 クレート
（`fandhe-ai-tensor-core`・`fandhe-ai-autodiff`・`fandhe-ai-backend-cpu`・
`fandhe-ai-backend-cuda`・`fandhe-ai-backend-metal`）の計 6 クレート
（命名確定は `docs/crates-io-naming-decision.md`・#878/#879）。

crates.io へ `cargo publish` するには、workspace member 間の `path` 依存が
registry 上で解決可能な `version` 指定を持つ必要がある。本ドキュメントは
その version 併記方針と、依存グラフのトポロジカル順に基づく公開順序・
版数運用（`workspace.version` 一括更新）を確定する。

`cargo publish --dry-run`／`cargo package` の実行検証・rustdoc 警告解消は
別イシュー #883 のスコープであり、本ドキュメントはその前段の方針確定に
留める。

## 1. `[dependencies]`／`[target.*.dependencies]` の公開クレート間依存: `path` + `version` 併記

公開 6 クレート間の通常依存（`[dependencies]`・`[target.'cfg(target_os =
"macos")'.dependencies]`）は、`path` に加えて `version = "=x.y.z"`
（workspace 公開バージョンと完全一致）を併記する。

```toml
fandhe-ai-tensor-core = { path = "../tensor-core", version = "=0.3.0" }
```

理由: `cargo publish` は `[dependencies]` の path 依存に registry 解決可能な
`version` がないと公開パッケージを生成できない（`dev-dependencies` は対象外。
2 節参照）。version なしで path のみの通常依存は非公開クレート
（`bench-harness` 等）への依存にのみ許容される。

**fail-closed 特性**: path 依存の `version` は path 先クレートの実際の
`Cargo.toml` の `version` と `cargo metadata`／`cargo build` 実行時に照合
される。`workspace.version` バンプ時に内部依存側の `version` 更新を漏らすと、
この照合が即座にエラーで検出する（下記 4 節の実測記録を参照）。よって
版数一致を保証する追加の検査機構（CI ジョブ等）は不要と判断する。

## 2. `[dev-dependencies]` の公開クレート間依存: `version` を外す（path のみ）

対象は以下の 5 箇所（#881 実装時点）:

| クレート | dev-dependency | 循環関係 |
|---|---|---|
| `backend-cpu` | `fandhe-ai-backend-cuda` | backend-cpu ⇄ backend-cuda |
| `backend-cpu` | `fandhe-ai-autodiff` | （循環なし。片方向 dev-dep） |
| `backend-cpu`（macOS 限定） | `fandhe-ai-backend-metal` | backend-cpu ⇄ backend-metal（macOS 限定） |
| `backend-cuda` | `fandhe-ai-backend-cpu` | backend-cuda ⇄ backend-cpu |
| `backend-metal` | `fandhe-ai-backend-cpu` | backend-metal ⇄ backend-cpu |

理由:

1. **`cargo package` の自動 strip（実測確認済み）**: version なし・path
   のみの dev-dependency は公開パッケージ生成時に自動的に取り除かれ、
   公開物（crates.io 上の `.crate`）には一切残らない。crates.io の公開
   要件は通常依存（`[dependencies]`）にのみ及ぶため、dev-dependency に
   registry 解決可能な version を持たせる必要はそもそもない。

   **実測記録**（2026-08-22・#881 実装時。コミットしない一時検証。
   `fandhe-ai-tensor-core` は crates.io 未公開のため `cargo package` の
   依存解決を通すべく、`.cargo/config.toml`（一時作成・検証後削除）に
   `[patch.crates-io] fandhe-ai-tensor-core = { path = "crates/tensor-core" }`
   を設定した上で `cargo package -p fandhe-ai-backend-cpu --no-verify
   --allow-dirty` を実行し、生成された `.crate`（`target/package/
   fandhe-ai-backend-cpu-0.3.0.crate`）を展開して同梱の `Cargo.toml`
   （cargo が自動生成する normalize 済みマニフェスト）を確認した。
   結果、`[dev-dependencies]` セクションと `[target.'cfg(target_os =
   "macos")'.dev-dependencies]` セクションはいずれも空で出力され、
   本来の `Cargo.toml` にあった `bench-harness`・`fandhe-ai-backend-cuda`・
   `fandhe-ai-autodiff` の path-only dev-dependency 3 件は完全に除去
   されていた（`[dependencies]` 側の `fandhe-ai-tensor-core`・`half`・
   `rayon` は version 付きでそのまま残存）。これにより、通常の
   `[dev-dependencies]` に加えて macOS 限定 target dev-dependencies
   （`backend-metal` の場合に相当する形）も strip 対象であることを
   確認した。検証用の `.cargo/config.toml`・`target/package/` は
   検証後に削除済み（コミットに含まれない）。
2. **循環依存の解消**: 上表のとおり `backend-cpu ⇄ backend-cuda`・
   `backend-cpu ⇄ backend-metal`（数値一致回帰テストが相互にテスト対象
   クレートを dev-dependency として参照するための構成。REQ-2）は
   dev-dependency 経由の循環である。version 併記のまま初回 publish を
   行うと、まだ registry に存在しない側のクレートへの
   「registry 未存在バージョンへの依存」で publish がブロックされうる。
   strip 方式（1 と同じ）にすることで、公開順序の制約は次節の
   `[dependencies]` のトポロジカル順のみに単純化される。
3. **非公開クレートへの dev-dep と同一規則への統一**: `bench-harness`
   （非公開・`publish = false`）への dev-dependency は元から version
   非併記・path のみである。公開クレート間の dev-dependency もこれと
   同一規則（「dev-dep は version 非併記・strip 対象」）に統一することで、
   「`[dependencies]` は version 併記・`[dev-dependencies]` は非併記」という
   単純な二分規則になり、レビュー・保守コストを下げる。

## 3. 公開順序（トポロジカル順）

`[dependencies]`（通常依存のみ。2 節の strip 方針により dev-dependency 経由の
循環は公開順序の制約から除外される）に基づくトポロジカル順は以下のとおり。

```
① fandhe-ai-tensor-core
       │
       ▼
② fandhe-ai-autodiff / fandhe-ai-backend-cpu / fandhe-ai-backend-cuda / fandhe-ai-backend-metal
   （この 4 つは相互に [dependencies] 依存がなく順不同。全て ① にのみ依存）
       │
       ▼
③ fandhe-ai（facade）
   （①・② の全 5 クレートに依存。facade/Cargo.toml [dependencies] 参照）
```

- ① `fandhe-ai-tensor-core`: 依存連鎖の起点（`crates/tensor-core/Cargo.toml`
  実測確認済み。`[dependencies]` は許容依存の `half`（外部クレート）のみで
  他の公開クレートへの依存を持たず、`[dev-dependencies]` セクション自体が
  存在しない）。最初に publish する。
- ② の 4 クレート: いずれも `fandhe-ai-tensor-core` のみを公開クレート間の
  `[dependencies]` として持つ（`backend-cuda`・`backend-metal` は加えて
  許容依存区分の外部クレート `cudarc`・`objc2` 系を持つが、これらは
  workspace member ではないため公開順序の制約に無関係）。① の publish 完了後
  ならどの順で publish してもよい。
- ③ `fandhe-ai`（facade）: ①・② の全 5 クレートに `[dependencies]` として
  依存する（`crates/facade/Cargo.toml`）ため、最後に publish する。

## 4. 版数運用: `workspace.version` による lockstep

全公開クレートは `version.workspace = true`（各 `Cargo.toml` の
`[package]`）により、ルート `Cargo.toml` の `[workspace.package].version`
と同一バージョンで一括管理する（lockstep）。

**バンプ手順**:

1. ルート `Cargo.toml` の `[workspace.package] version = "x.y.z"` を更新する。
2. 公開 6 クレート間の内部依存 `version = "=x.y.z"`（#881 実装時点で計 9 箇所。
   `grep -n 'version = "=0.3.0"' crates/*/Cargo.toml` で実測確認済み:
   `autodiff`・`backend-cpu`・`backend-cuda`・`backend-metal` 各 1 箇所の
   `fandhe-ai-tensor-core` 依存 + `facade` の 5 箇所〈`fandhe-ai-tensor-core`・
   `fandhe-ai-autodiff`・`fandhe-ai-backend-cpu`・`fandhe-ai-backend-cuda`・
   `fandhe-ai-backend-metal`〉）を同時に新バージョンへ更新する。
3. `cargo build --workspace`（または `cargo metadata`）を実行する。

**fail-closed 動作の実測記録**（2026-08-22・#881 実装時。コミット前に検証し
元に戻した一時変更）: `crates/autodiff/Cargo.toml` の
`fandhe-ai-tensor-core` 依存の `version` のみを `"=0.3.1"`
（`tensor-core` 側の実バージョン `0.3.0` とは不一致）に変更し
`cargo metadata --format-version 1` を実行したところ、以下のエラーで
即座に失敗した。

```
error: failed to select a version for the requirement `fandhe-ai-tensor-core = "=0.3.1"`
candidate versions found which didn't match: 0.3.0
location searched: crates/tensor-core
required by package `fandhe-ai-autodiff v0.3.0 (crates/autodiff)`
```

この実測により、手順 2 の更新漏れ（内部依存の version 更新忘れ）は
`cargo build`／`cargo metadata`／`cargo test` のいずれの実行でも
即座に検出される（fail-closed）ことを確認した。よって version 一致を
保証する専用の追加検査スクリプトは導入しない。

## 5. 内部依存の `[workspace.dependencies]` への集約は行わない

公開クレート間の内部依存（`fandhe-ai-tensor-core` 等）は、各
`Cargo.toml` に直書き（`{ path = "...", version = "=x.y.z" }`）する現状の
形を維持し、`[workspace.dependencies]` への集約は行わない（最小差分方針）。
各依存行には利用理由・呼び出し元コンテキストを明記したコメント
（`.claude/rules/code-comment-style.md` 準拠）が付いており、直書き維持の
ほうが依存 1 件ごとの文脈が追いやすく可読性で勝ると判断する。集約が
将来必要になった場合は `.claude/rules/out-of-scope-tracking.md` の規約に
従い、ユーザー承認を前提とした別イシューで扱う。

## 6. 非公開クレートとの依存が公開を阻害しないことの確認

`onnx-interop`・`guardrail`・`self-repair`・`bench-harness`・`docs-site`
（いずれも `publish.workspace = true` によりルート `[workspace.package]` の
`publish = false` を継承。非公開 5 クレート）への依存（dev-dependency 含む）
は、公開 6 クレート（`tensor-core`・`autodiff`・`backend-cpu`・
`backend-cuda`・`backend-metal`・`facade`）の `Cargo.toml` 全件を実測確認
した時点（#881 実装時）で以下のいずれかに限られる:

- `[dev-dependencies]` への version なし・path のみの依存
  （`backend-cpu`・`backend-cuda`・`backend-metal`・`autodiff`・`facade` の
  `bench-harness` 依存）
- `tensor-core` は `[dev-dependencies]` セクション自体を持たず、非公開
  クレートへの依存は皆無（3 節 ① の実測記録参照）

これらは 2 節と同じ理由（`cargo publish` 時の自動 strip）により公開物に
一切残らず、非公開クレートへの依存が公開クレートの publish を阻害する
経路は存在しない。公開 6 クレートの `[dependencies]`（通常依存）に非公開
クレートが現れないことも合わせて確認済み。

## 7. 公開 6 クレートの `publish = true` 明示（非公開既定の上書き）

ルート `Cargo.toml` の `[workspace.package]` は `publish = false`
（非公開 5 クレート・#881 時点で追加された `docs-site` 向けの既定。1 節
冒頭のリポジトリ構成コメント参照）であり、公開 6 クレートは従来
`publish.workspace = true` でこれをそのまま継承していた。この状態では
本ドキュメントが確定する公開順序どおりに `cargo publish` を実行しても
crates.io への公開が拒否される（`publish = false` は登録禁止指定のため）。
よって公開 6 クレート（`tensor-core`・`autodiff`・`backend-cpu`・
`backend-cuda`・`backend-metal`・`facade`）の各 `Cargo.toml` では
`publish.workspace = true` を `publish = true` に変更し、workspace 既定を
明示的に上書きする（#881 実装時に反映済み）。非公開 5 クレートは
`publish.workspace = true`（`false` 継承）のまま変更しない。

`publish = true` は crates.io への公開を許可する状態にするのみで、実際の
初回 `cargo publish` 実行（トークン発行・`--dry-run` 検証・rustdoc 警告
解消を含む）は別イシュー #883 のスコープであり、本ドキュメント・本 PR は
実行しない。

## 8. 公開前検証手順と実測記録（#883）

イシュー #883「`cargo publish --dry-run` / `cargo package` 検証と rustdoc
警告解消」の実測記録。`cargo` バージョンは 1.96.0（2026-08-22 実測）。

### 8.1 多パッケージ dry-run（正式手順）

cargo 1.90 で安定化した workspace publish 機構により、公開 6 クレートを
1 コマンドで検証できる（3 節のトポロジカル順を内部で解決するため
`-p` の列挙順に依存しない）。

```sh
cargo publish --dry-run \
  -p fandhe-ai-tensor-core \
  -p fandhe-ai-autodiff \
  -p fandhe-ai-backend-cpu \
  -p fandhe-ai-backend-cuda \
  -p fandhe-ai-backend-metal \
  -p fandhe-ai
```

2026-08-22 実測（クリーンな worktree・登録済みトークン不要。`--dry-run` は
crates.io index の参照のみでアップロードは行わない）: 6 件すべてが
Packaging → Verifying（依存クレートを一時レジストリで解決してビルド
検証）→ `warning: aborting upload due to dry run` まで到達し成功した。

### 8.2 単一クレートの dry-run（想定内の失敗。参考情報）

依存順の単一クレート実行は、`fandhe-ai-tensor-core`（依存連鎖の起点。
外部依存 `half` のみ）のみ単独で成功し、後続クレートは実依存先
（`fandhe-ai-tensor-core` 等）が crates.io に未公開である限り失敗する。
これは 8.1 の多パッケージ dry-run が一時レジストリで内部依存を解決する
のに対し、単一クレート dry-run は本物の crates.io index のみを参照する
ためであり、想定内の挙動である（受け入れ条件 a の「未公開依存起因の
失敗は想定内」に対応）。

```sh
$ cargo publish --dry-run -p fandhe-ai-autodiff
...
error: failed to prepare local package for uploading

Caused by:
  no matching package named `fandhe-ai-tensor-core` found
  location searched: crates.io index
  required by package `fandhe-ai-autodiff v0.3.0 (crates/autodiff)`
```

`fandhe-ai-backend-cpu`・`fandhe-ai-backend-cuda`・`fandhe-ai-backend-metal`・
`fandhe-ai` も同様に `fandhe-ai-tensor-core`（および `fandhe-ai-autodiff` 等）
未公開のため同型のエラーで失敗することを確認した。実際の公開時は 3 節の
順序に従い ①→②→③ の順で 1 クレートずつ実行すれば、この失敗は publish 済み
クレートから解消されていく。

### 8.3 パッケージ内容の検証

`cargo package --list -p <クレート>` を公開 6 クレート全件（`fandhe-ai-tensor-core`・
`fandhe-ai-autodiff`・`fandhe-ai-backend-cpu`・`fandhe-ai-backend-cuda`・
`fandhe-ai-backend-metal`・`fandhe-ai`）で個別に実行し、いずれも README.md・
`LICENSE-APACHE`・`LICENSE-MIT`・`src/`・`tests/`（`facade` は `examples/` も）の
同梱を確認した（README.md の同梱は 8.1 の 6 クレート dry-run 成功自体が
間接的に裏付ける。`README.md` 欠落は `cargo package` を fail させるが、
`LICENSE-*` 欠落は fail させず warning にもならないため、この 6 クレート
個別実行が LICENSE 同梱を確認する唯一の手段である）。LICENSE ファイルは
ルート直下の `LICENSE-APACHE`・
`LICENSE-MIT`（Apache-2.0 / MIT デュアルライセンス全文）を各公開クレート
ディレクトリへコピーする方式で同梱した（`license = "MIT OR Apache-2.0"`
のみではクレートディレクトリ外のルート LICENSE ファイルは自動同梱され
ないため。symlink ではなくコピーにしたのは Windows checkout・`cargo
package` での確実な同梱のため）。6 クレートいずれも `include`/`exclude`
キーを持たないため、追加後の同梱漏れリスクはない（実測確認済み）。

`[dev-dependencies]` の strip は 2 節で確認済みの方式が引き続き機能して
おり、`cargo package` が生成する正規化済み `Cargo.toml`（`.crate` 展開）
でも空セクションとして出力されることを本イシューの多パッケージ dry-run
実行時にも再確認した。

**`manifest has no documentation, homepage or repository` 警告について**:
本イシューの実測時点（2026-08-22）で並行 issue #880「publish メタデータ
整備と publish フラグの per-crate 制御」は未マージ（open）であり、6 クレート
すべてでこの警告が引き続き出力される。#880 のスコープ（`repository`・
`documentation`・`keywords`・`categories`・`description` の整備）であり
本 PR では対応しない（実装計画のスコープ境界節参照）。#880 マージ後は
この警告は解消される想定であり、`cargo publish --dry-run` の成否
（Packaging → Verifying → dry-run 中断）には影響しない warning 止まりの
項目である。

### 8.4 docs.rs でのビルド成立性

docs.rs 既定ビルド相当の `cargo doc --no-deps --target
x86_64-unknown-linux-gnu` は全 11 クレート（非公開クレートを含む）で
ビルド成立を確認した（`cudarc` は動的ロード方式のため CUDA toolkit 非搭載
でもビルドでき、`objc2` 系は `cfg(target_os = "macos")` により当該
ターゲットでは自然に除外される）。

`backend-metal` は主要 API（`gemm`・`buffer`・`error` 等）のほぼ全体が
`cfg(target_os = "macos")` 限定のため、docs.rs の既定ターゲット
（`x86_64-unknown-linux-gnu`）上ではほぼ空のドキュメントページになる。
`Cargo.toml` へ `[package.metadata.docs.rs] default-target =
"aarch64-apple-darwin"` を追加すれば docs.rs 上で macOS 限定 API を
表示できる可能性があるが、2026-08-22 時点で docs.rs が
`aarch64-apple-darwin` を実際にサポートするターゲットとして公式に
明記しているかを本イシューの範囲内では確証できなかった（誤った
`default-target` 指定は docs.rs 側のビルド失敗を招き、現状の
「ほぼ空ページ」より悪化しうる）。fail-closed の判断としてこの
`Cargo.toml` 変更は見送り、`docs.rs` 側のサポート対象ターゲット一覧を
実機（docs.rs のドキュメント）で確認したうえで別イシューとして対応する
（out-of-scope 追跡対象。ユーザー承認事項）。

### 8.5 rustdoc 警告の解消

`cargo doc --workspace --no-deps` の rustdoc 警告（cfg による差異のため
ホスト〈aarch64-apple-darwin〉・`--target x86_64-unknown-linux-gnu` の両方で
実測。2026-08-22 時点でホスト 279 件・Linux ターゲット 303 件）はすべて
解消した。大半（343 箇所）は `-->` 付きの位置情報を持つため
`file:line:col` の和集合で機械抽出できたが、`facade`・`self-repair` の
モジュール doc（`//!`）内の 8 件は rustdoc が `note: the link appears in
this line` 形式（`-->` 行なし）で報告したため機械抽出の対象外となり、
`grep` によるテキスト検索で個別に特定し手動修正した（延べ 351 箇所）。
修正方針は 3 種:

1. private item へのリンク・cfg で消えるアイテムへの unresolved link
   （大半）: `` [`foo`] `` → `` `foo` ``（表示テキストは不変のままコード
   スパン化。情報量を落とさない）
2. `fn@`/`mod@` ディスアンビゲータで解決した 3 件（同名の関数とモジュール
   の曖昧参照）
3. bare URL 1 件を `<...>` でエスケープ

`#[allow(rustdoc::...)]` や `--document-private-items` での抑制、リンク先の
public 化は行っていない（`.claude/rules/coding-rust.md`・実装計画のスコープ
境界節）。検証コマンド（CI と同一）:

```sh
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --target x86_64-unknown-linux-gnu
```

いずれも 2026-08-22 実測で exit 0（警告 0 件）。`.github/workflows/ci.yml`
の `build` ジョブへ同等のゲート step を追加した: Linux ホスト分は
`--workspace`、`aarch64-apple-darwin` クロス分は `-p fandhe-ai-backend-metal
-p fandhe-ai-backend-cpu`（cfg 分岐で警告集合が実際に変わる 2 クレート）に
限定する。既存の「cargo build（macOS ターゲット・Metal 有効・lib のみ）」
step が `--workspace` にせず `--lib` 限定にしている理由（guardrail の
`[[bin]]` が macOS クロスリンカ非搭載の Linux runner 上でリンクを要求し
うる懸念）と同じ判断であり、`cargo doc --no-deps` はリンクを行わないため
理論上は安全と考えられるが、既存の縮小方針に倣い実際に警告差分が出る
2 クレートへ限定することで不要なリスクを避けた。`make doc-warnings` で
ローカル再現できるようにした（ci.yml と同一コマンドを共用）。

## 9. release ワークフローによる公開手順（#884）

イシュー #884 で追加した `.github/workflows/release.yml`（workflow_dispatch
起点・`CARGO_REGISTRY_TOKEN` 方式）を使った公開手順。初回公開の実行自体は
イシュー #885 のスコープであり、本節は運用手順の記録に留める。

### 9.1 入力

GitHub の Actions タブから `crates.io release` ワークフローを手動実行
（`workflow_dispatch`）する。入力は 3 つ:

| 入力 | 内容 |
|---|---|
| `crate` | 公開対象クレート（choice 型で公開 6 クレートに固定） |
| `version` | 公開バージョン（対象クレートの `Cargo.toml` の `version` と完全一致必須） |
| `mode` | `dry-run-only`（既定・トークン不要）／`publish`（実公開。environment 承認ゲートを通る。main ブランチからのディスパッチ限定） |

### 9.2 2 段運用（まず dry-run、次に publish）

1. まず `mode: dry-run-only` で実行し、`verify` ジョブ（semver 形式検証・
   `Cargo.toml` バージョン一致検証・crates.io 既公開バージョン検証・
   `cargo package --list`・`cargo publish --dry-run`）が green であることを
   確認する。
2. green を確認したら、同一の `crate`／`version` で `mode: publish` を指定して
   再実行する。`mode: publish` は `refs/heads/main` からのディスパッチのみ
   受け付ける（`verify` ジョブが fail-closed で検査し、main 以外からの
   ディスパッチはここで失敗する）。`verify` ジョブが再度走った後、`publish`
   ジョブが `environment: crates-io-release` の承認ゲートを経て
   `cargo publish` を実行する。`environment` の deployment branch 制限
   （main 限定）＋ required reviewers は GitHub 側の設定であり（本ワークフロー
   自体では代替できない）、`mode: publish` を実運用する前にユーザーが GitHub
   側で設定しておく前提条件である。

### 9.3 公開順序（3 節のトポロジカル順を 1 クレートずつ実行）

release ワークフロー自身はクレート横断の順序保証を持たないため、3 節の
順序に従い手動で ①→②→③ の順に 1 クレートずつ実行する:

```
① fandhe-ai-tensor-core
② fandhe-ai-autodiff / fandhe-ai-backend-cpu / fandhe-ai-backend-cuda /
  fandhe-ai-backend-metal（順不同）
③ fandhe-ai
```

各クレートの `publish` 完了後、次のクレートを実行する前に sparse index への
反映を確認する:

```sh
curl https://index.crates.io/<index-path>
```

`<index-path>` は crates.io のシャーディング規則（1〜2 文字: `<len>/<name>`、
3 文字: `3/<先頭1文字>/<name>`、4 文字以上: `<先頭2文字>/<次2文字>/<name>`）に
従う。公開 6 クレート名はいずれも 9 文字以上のため `<先頭2文字>/<次2文字>/
<name>` 形（例: `fandhe-ai-tensor-core` → `fa/nd/fandhe-ai-tensor-core`）。
出力に `"vers":"<公開したバージョン>"` が含まれていれば反映済みと判断できる
（反映まで数分かかることがある）。反映前に依存先クレートの `publish` を
実行すると 8.2 節と同型の「no matching package named ...」で想定内失敗に
なるため、反映確認を待ってから次を実行する。

### 9.4 途中失敗時の再実行

`verify` ジョブの既公開バージョン検証（sparse index 参照）が、同一
バージョンの再公開を自動的に阻止する（非冪等な `cargo publish` の冪等性を
CI 側で担保する設計。release.yml 冒頭コメント参照）。よって公開順序の
途中でいずれかのクレートが失敗した場合、公開済みクレートは再実行しても
このガードで即座に失敗して停止するため無害であり、**失敗したクレートから
そのまま再実行すればよい**（origin/main の状態を戻す等の後始末は不要）。

### 9.5 §8.2 との関係

単一クレートの `cargo publish --dry-run`（release.yml の `verify` ジョブが
実行するのと同じコマンド）は実 crates.io index のみを参照するため、
依存先クレートが未公開の間は 8.2 節で実測した「no matching package named
`fandhe-ai-tensor-core`」等の失敗になる。これは release ワークフロー上でも
同様に発生する想定内の失敗であり、9.3 節の順序どおり ①→②→③ で 1 クレート
ずつ実行すれば解消されていく。

## 10. 初回公開の前提条件未充足による保留記録（#885）

イシュー #885「初回公開実行と crates.io / docs.rs 反映検証」の実行時（2026-08-23）に
`mode: publish` 実行前の必須ゲート（G0。`cargo publish` は unpublish 不可・yank のみの
不可逆操作であるため設けた事前チェック）を再実測した結果、以下 2 点が未充足であり、
**本イシューでは実公開（`mode: publish` dispatch）を一切実行していない**。

### 10.1 実測結果

| 前提 | 確認コマンド | 実測結果（2026-08-23） |
|------|-------------|------------------------|
| #880（publish メタデータ整備: `repository`/`keywords`/`categories`/`description` 充実）が main に反映済み | `gh issue view 880 --json state` | **未充足**: `"state":"OPEN"`（未マージ。対応 PR #892 も open） |
| GitHub environment `crates-io-release` が required reviewers + deployment branch 制限（main 限定）で設定済み | `gh api repos/Fandhe-AI/rust-ai-library/environments` | **未充足**: `{"total_count":0,"environments":[]}`（environment 自体が未作成） |

上記いずれも `.github/workflows/release.yml` 冒頭コメントが明記する前提条件であり、
未充足のまま `mode: publish` を dispatch すると (a) `manifest has no documentation,
homepage or repository` 等のメタデータ欠落が当該 version に恒久的に残る、
(b) environment 承認ゲートが機能せず誤 dispatch を止められない、の 2 リスクを負う。
よって fail-closed の方針（`.claude/rules/security.md` A08・本ドキュメント冒頭の
「公開は監査可能な CI ワークフロー経由に限定」方針）に従い、実公開を保留した。

### 10.2 dry-run-only の実行有無

8.1 節・8.2 節の時点（#883）で `fandhe-ai-tensor-core` を含む 6 クレート全件の
`cargo publish --dry-run` 実測記録が既に存在し、`no matching package named
fandhe-ai-tensor-core`（依存先未公開時の想定内失敗）等の結果は記録済みである。
release ワークフロー（#884）経由の `mode: dry-run-only` 再実行は追加の実行環境
（GitHub Actions runner）で同じ検証を繰り返すのみで、G0-1（#880）が未マージの間は
同一のメタデータ欠落警告が再現するだけであり新たな知見を追加しない。加えて #880
マージ後は本節の dry-run 結果自体が陳腐化する。よって本イシューでは release.yml の
`workflow_dispatch` を一度も実行していない（`gh workflow run` を叩いていない）。

### 10.3 crates.io 未公開の確認（read-only）

sparse index への到達性のみを read-only な `curl` で確認した（トークン不要・
アップロードなし）。6 クレートすべてで `HTTP 404`（未公開）を確認した。

| クレート | index path | HTTP status |
|---------|------------|--------------|
| `fandhe-ai-tensor-core` | `fa/nd/fandhe-ai-tensor-core` | 404 |
| `fandhe-ai-autodiff` | `fa/nd/fandhe-ai-autodiff` | 404 |
| `fandhe-ai-backend-cpu` | `fa/nd/fandhe-ai-backend-cpu` | 404 |
| `fandhe-ai-backend-cuda` | `fa/nd/fandhe-ai-backend-cuda` | 404 |
| `fandhe-ai-backend-metal` | `fa/nd/fandhe-ai-backend-metal` | 404 |
| `fandhe-ai` | `fa/nd/fandhe-ai` | 404 |

### 10.4 受け入れ条件の充足状況

イシュー #885 の受け入れ条件 (a)（6 クレート公開・docs.rs ビルド成功）・
(b)（新規プロジェクトから `cargo add` して最小例が動作）は、上記のとおり
実公開自体を行っていないため **共に未充足**である。docs.rs ビルド確認・
`cargo add` スモークテストも実行していない（前提の公開が存在しないため実行不能）。

### 10.5 再開に必要な対応（担当・対応リポジトリ外）

- #880（本リポジトリの PR #892）のマージ
- GitHub environment `crates-io-release`（required reviewers + deployment branch
  制限〈main 限定〉）のユーザーによる設定（`.github/workflows/release.yml` 冒頭
  コメント参照。エージェント単独では実施しない）

上記 2 点が満たされた後、9.3 節のトポロジカル順（①→②→③）で、
クレートごとに `-f crate=<crate> -f version=<version>` を明示した
`workflow_dispatch` を実行する（`version` は必須入力で既定値を持たないため、
`-f version=<version>` を省略すると dispatch できない。9.1 節）。dry-run と
publish は同一の `crate`／`version` を渡し、まず `dry-run-only` を green
確認してから `publish`（environment 承認）へ進める（9.2 節）。①のクレート・
`workspace.version`（4 節で確定した公開バージョン）を仮に `0.3.0` とした例:

```sh
gh workflow run release.yml \
  -f crate=fandhe-ai-tensor-core -f version=0.3.0 -f mode=dry-run-only
gh workflow run release.yml \
  -f crate=fandhe-ai-tensor-core -f version=0.3.0 -f mode=publish
```

sparse index 反映確認（9.3 節）後、②→③の各クレートについても
`-f crate=<crate>` のみを対象クレート名に差し替え、`-f version` は
一括バンプ後の同一 `workspace.version` を渡して同様に
`dry-run-only` → `publish` の順で 1 クレートずつ実行する。

## 11. リリース手順まとめ（バージョン更新〜タグ付け。イシュー #886）

4 節（版数バンプ）・9 節（release ワークフロー運用）を通しの手順として
まとめ、公開完了後のタグ付けを新たに確定する。

1. **バージョン更新**: 4 節の手順に従い、ルート `Cargo.toml` の
   `[workspace.package].version` と内部依存 `version = "=x.y.z"`（9 箇所）を
   同時に更新する。`cargo build --workspace`（または `cargo metadata`）で
   更新漏れが fail-closed に検出されることを確認する。
2. **PR → CI green → main マージ**: 通常の Conventional Commits フロー
   （`build(workspace):` 等）で PR を作成し、`ci-complete` を含む必須
   チェックが green であることを確認してから main へマージする。
3. **release ワークフローの実行**: 9.2〜9.3 節の手順（`dry-run-only` →
   `publish`、①→②→③のトポロジカル順で 1 クレートずつ）に従い、main への
   マージ後のコミットを起点に実行する。
4. **公開完了後のタグ付け**: 公開 6 クレート全件の `publish` 完了・sparse
   index への反映確認（9.3 節）後、公開した main コミットへ注釈付きタグ
   `vX.Y.Z`（`workspace.version` と同一の単一タグ。lockstep のためクレート
   別タグは付けない）を付与し push する。

   ```sh
   git tag -a vX.Y.Z -m "release: vX.Y.Z" <公開完了時点の main コミット sha>
   git push origin vX.Y.Z
   ```

   このタグは **記録用（公開済みバージョンと main コミットの対応関係を
   追跡する目的）であり、CI トリガーではない**。release ワークフロー
   （#884）は `workflow_dispatch` のみをトリガーとする方式に確定済みで
   （`.claude/rules/ci.md` release.yml 節）、タグ push によるワークフロー
   起動は行わない。
5. **変更履歴への追記**: 本ドキュメントの「変更履歴」節に、公開したバージョン・
   対応イシュー・付与したタグを追記する。

GitHub Release の作成等、タグ付け以降の追加運用は本ドキュメントのスコープ
外とする（必要になった時点で別イシューとして起票し検討する）。

## 変更履歴

- 2026-08-23（#886）: CLAUDE.md・`.claude/rules/ci.md` への公開構成反映に
  あわせ、4 節（版数バンプ）・9 節（release ワークフロー運用）を通しの
  リリース手順としてまとめ、公開完了後のタグ付け手順（注釈付きタグ
  `vX.Y.Z`・記録用で CI トリガーではない）を 11 節として新規追記した。
- 2026-08-23（#885）: 初回公開実行を試行したが、前提条件ゲート（#880 未マージ・
  environment `crates-io-release` 未設定）が未充足であったため実公開
  （`mode: publish`）を実行せず保留し、実測結果を 10 節として記録した。
- 2026-08-23（#884）: release ワークフロー（`.github/workflows/release.yml`。
  workflow_dispatch + `CARGO_REGISTRY_TOKEN` 方式）による公開手順を新節（§9）
  として追記した。ワークフロー本体の設計・セキュリティ考慮は
  `.github/workflows/release.yml` 冒頭コメントを正とし、本節では運用手順の
  みを記録する。
- 2026-08-22（#881）: 本ドキュメント新規作成。PR #891（#880 系）で先行実施
  済みだった公開クレート間 `[dependencies]` の version 併記（codex-review
  P1 指摘対応）を正式方針として確定し、`[dev-dependencies]` については
  version を外す方針（strip 方式）を新たに確定した。
- 2026-08-22（#881・同日追補）: `[patch.crates-io]` 一時設定による
  `cargo package` 実測で strip 方針（2 節）を裏付け、内部依存 version
  箇所数の誤記（10 → 9）を修正し、`tensor-core` の `Cargo.toml` 実測
  （非公開クレートへの依存なし・依存連鎖の起点であること）を確認した。
- 2026-08-22（#881・PR #893 codex-review P1 対応）: 公開 6 クレートが
  ルート `[workspace.package]` の `publish = false` を `publish.workspace =
  true` 経由でそのまま継承しており、本ドキュメントが確定する公開順序
  どおりに `cargo publish` しても公開禁止で失敗する矛盾を修正した。公開
  6 クレートの `Cargo.toml` へ `publish = true` を明示する変更を反映し、
  7 節として本方針を追記した。
- 2026-08-22（#883）: `cargo publish --dry-run`／`cargo package` の実行検証・
  rustdoc 警告解消（`cargo doc --workspace --no-deps` の全 351 箇所解消。
  機械抽出できた 343 箇所＋モジュール doc 内の位置情報なし警告 8 箇所）・
  LICENSE ファイルの per-crate 同梱・docs.rs ビルド成立性確認の結果を
  8 節として追記した。backend-metal の `[package.metadata.docs.rs]` 追加は
  docs.rs のターゲットサポート未確証のため見送り、別途確認する
  out-of-scope 事項とした。ci.yml の macOS ターゲット分 rustdoc ゲート step
  は既存の「cargo build（macOS ターゲット）」step と同じ理由（guardrail の
  `[[bin]]` によるクロスリンク失敗の懸念）で `--workspace` にせず、cfg 分岐で
  警告集合が実際に変わる `fandhe-ai-backend-metal`・`fandhe-ai-backend-cpu`
  の 2 クレートに限定した。
