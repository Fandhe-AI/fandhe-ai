# 否定ガードを正ガードへ反転する手順書（イシュー #3782）

`api_surface.rs` の否定ガード（「この名前は公開しない」）と保留 doctest ガード
（`*HoldDoctestGuard`）を、**承認済みの形だけを許す正ガード**へ反転するときの共通手順。
共通部品は同ディレクトリの `mod.rs`（`api_surface.rs` の `mod surface_guard;` から取り込む）。

## 前提

- 反転してよいのは所有者が承認した形だけ。未承認の形へは反転しない。
- REQ-2 の tolerance 定数・parity baseline・ガードレール閾値・公開面（`crates/facade/src/**`）は
  このガードの反転では変えない。依存も追加しない。
- 先例は行番号がずれるため、関数名で参照する。

## 反転の 2 つの型

| 型 | 先例 | やること |
|----|------|----------|
| 全部撤去型 | #2508 `accumulate_steps` | `src/lib.rs` の `*HoldDoctestGuard` の struct と doc、`*_hold_doctest_globs_all_pub_modules`、`*_hold_doctest_probe_body_matches_fixed_contract`、`*_HOLD_PROBE_BODY` 定数、否定テストを消し、正ガードに置き換える |
| 部分反転型 | #2560／#2561 EMA、#2679／#3060 SWA | 保留ガードは未承認の経路（`FitConfig`／`Sequential` への inherent メソッド等）専用に縮め、`*_HOLD_PROBE_BODY` を同じ PR で更新する。名前は残す |

## 5 手順

### 手順 1: 宣言の固定

`fn_declaration_sites`／`type_declaration_sites` で宣言場所を集め、`declaration_count_violations`
で承認形と多重集合として完全一致させる。0 件（取り下げ・改名）・承認外の場所・重複はすべて違反。
`expected = &[]` は「宣言が残っていない」の確認に使う。

```rust
let sites = surface_guard::fn_declaration_sites(&facade_src_files(), "accumulate_steps");
assert!(surface_guard::declaration_count_violations(
    "accumulate_steps", &sites, &["compat/training.rs"]).is_empty());
```

先例: `facade_declares_fit_config_accumulate_steps_exactly_once`（fn）、SWA の `swa_surface_violations`（型）。

### 手順 2: 再エクスポートの固定

`reexport_violations(files, names, approved)`（= `approved_reexport_surface_violations`）に、
承認した `pub use` 文を `(src 相対パス, 文)` で渡す。rustfmt の折り返しで付く末尾カンマは正規化済み。
独自ラッパー型を許す場合は手順 1 の `type_declaration_sites` と組み合わせ、`names` には内部型の名前だけを渡す。

先例: `facade_exposes_phase4_training_data_only_in_approved_shape`。

### 手順 3: すり替え検出の自己テスト

`assert_detects_each_category(label, checker, base, missing, injected, benign)` で、検出器が
「常に通る」状態になっていないことを固定する。最低限入れる類型:

- 違反になるもの（`injected`）: 別名 `as`、波括弧のグループ、別ファイルからの再エクスポート、重複、
  独自宣言（`struct`／`enum`／`type`…）、内部型の素の再エクスポート、承認外の `fn` 宣言
- 欠落（`missing = true`）: 承認形のファイルを 1 つ抜く
- 違反にならないもの（`benign`）: コメント・文字列内の記述、非公開 `use`、無関係な `pub use`／`fn`

先例: `facade_exposes_phase4_training_data_only_in_approved_shape_detects_each_category`。

### 手順 4: 到達性テスト（ヘルパーなし）

コンパイル時の検査なので汎用ヘルパーは作らず、次のひな形で書く。`fandhe_ai::*` だけを import し
（`fandhe_ai_autodiff` は import しない）、関数ポインタ型の注釈でシグネチャを固定する。
スタブを弾くため振る舞いも最小限確かめる。

```rust
#[test]
fn xxx_is_reachable_via_facade_only() {
    use fandhe_ai::xxx::Type;
    let f: fn(&Type, usize) -> Type = Type::method;
    assert_ne!(f(&base, 2), base);
}
```

先例: `callback_swa_variant_is_constructible_and_destructurable_via_facade_only`。

### 手順 5: doctest の実在検査

`usage_doctest_violations(&[UsageDoctest { label, file, anchor, needles }])`。`anchor` は
`DocAnchor::Module`（`//!`）か `DocAnchor::Item("pub struct X")`（直前の `///`）。実際にコンパイルされる
ブロック（info が空または `rust`）が、`needles` をすべて含むことを要求する。
否定の保証に `compile_fail` は使わない（stable rustdoc はエラーコードを照合しないため）。
実体のコンパイル・実行は `cargo test --doc`。

先例: `swa_usage_doctests_are_present_and_compiled`。

## 保留ガード撤去の副作用チェックリスト

- [ ] `hold_doctest_probe_blocks_reference_every_glob_imported_item` の `MIN_KNOWN_PROBE_BLOCKS`
  は下限。プローブモジュールを消して検出数が下回る場合は、履歴コメントを付けて下げる。`known` 配列が
  すべて反転で消えるなら、残る既知プローブへ差し替える
- [ ] 残す保留ガードがあるとき `*_globs_all_pub_modules` が通ること（新しい `pub mod` を足したら、
  残る全保留ガードの glob 一覧に足す）
- [ ] `crates/autodiff/src/*.rs` の `//!` に残る「否定ガードが固定する」の記述、決定記録 doc の該当節、
  `docs/compat-api-scope.md` §5 の適用記録、`docs/README.md` の索引の文言を更新する

## 検証コマンドと PR 運用

```sh
cargo fmt --all -- --check
cargo clippy -p fandhe-ai --all-targets --all-features -- -D warnings
cargo test -p fandhe-ai --test api_surface
cargo test -p fandhe-ai --doc
```

- 実機の要素がない反転は `Closes #N`、部品だけ・実測前の PR は `Refs #N`。
- 既存テストは消さず、置き換える場合も判定が変わらないことを PR 本文に書く。

## ヘルパーを `tests/common/` に移さない理由

`tests/common/mod.rs` は多数のテストバイナリが `mod common;` で取り込む。あるバイナリで使わない
`pub` 項目は `dead_code` になり、clippy `-D warnings` で落ちる。`surface_guard` は `api_surface`
バイナリ専用なので、そこに閉じる。
