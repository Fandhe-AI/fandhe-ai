# 腕同定の検証（イシュー #1978 GB10 分・2026-09-18 UTC）

| 項目 | 値 |
|---|---|
| before 腕 `.rev-stamp` | `b127e9f25804661e30626d8c45da8aafc72ea836`（main `176f27e7` に `RULE-gb10.txt` のみ追加。crates／scripts は main と同一） |
| after 腕 `.rev-stamp` | `b127e9f2…+on-arm.patch` |
| `on-arm.patch` sha256 | `ca493a098c14269615b285a79c70f43f4b6f74e42d2c8ca178eed2d2be721cff`（`patch -p1 --forward` 成功。`patch_apply.log`） |
| 定数 | before: `SME_PRODUCTION_ENABLED: bool = false` ／ after: `= true`（`gate_constant.txt`。両ツリーとも `mod.rs:3085`） |
| ツリー指紋（`git ls-files crates scripts/bench/framework-compare` 1422 ファイルの sha256） | before 腕（`fp-before.txt`）は Mac 側の同一一覧に対する `shasum -a 256` と全 1422 行一致。after 腕との差分（`fp-diff.txt`）は `crates/backend-cpu/src/gemm_blis/mod.rs` の 1 ファイルのみ |
| bench-fandhe バイナリ sha256 | `r1r2/sha-1978-{before,after}-1978-gb10.txt`（異なる 2 本。`cargo tree` は各腕の `crates/facade` へ path 解決。`r1r2/tree-*.txt`） |
| 指紋の採取時点 | ビルド前（`cargo` が `Cargo.lock` を書き換える前） |
