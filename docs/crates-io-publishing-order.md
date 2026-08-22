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

1. **`cargo publish` の自動 strip**: version なし・path のみの dev-dependency
   は公開パッケージ生成時に自動的に取り除かれ、公開物（crates.io 上の
   `.crate`）には一切残らない。crates.io の公開要件は通常依存
   （`[dependencies]`）にのみ及ぶため、dev-dependency に registry
   解決可能な version を持たせる必要はそもそもない。
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

- ① `fandhe-ai-tensor-core`: 依存連鎖の起点（他の公開クレートへの
  `[dependencies]` を持たない）。最初に publish する。
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
2. 公開 6 クレート間の内部依存 `version = "=x.y.z"`（#881 実装時点で計 10 箇所:
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

`onnx-interop`・`guardrail`・`self-repair`・`bench-harness`（いずれも
`publish.workspace = true` が現状 `false` を継承）への依存（dev-dependency
含む）は、公開 6 クレートの `Cargo.toml` 全件を実測確認した時点
（#881 実装時）で以下のいずれかに限られる:

- `[dev-dependencies]` への version なし・path のみの依存
  （`backend-cpu`・`backend-cuda`・`backend-metal`・`autodiff`・`facade` の
  `bench-harness` 依存）

これらは 2 節と同じ理由（`cargo publish` 時の自動 strip）により公開物に
一切残らず、非公開クレートへの依存が公開クレートの publish を阻害する
経路は存在しない。公開 6 クレートの `[dependencies]`（通常依存）に非公開
クレートが現れないことも合わせて確認済み。

## 変更履歴

- 2026-08-22（#881）: 本ドキュメント新規作成。PR #891（#880 系）で先行実施
  済みだった公開クレート間 `[dependencies]` の version 併記（codex-review
  P1 指摘対応）を正式方針として確定し、`[dev-dependencies]` については
  version を外す方針（strip 方式）を新たに確定した。
