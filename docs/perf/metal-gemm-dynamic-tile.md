# Metal GEMM 動的タイル選択 計測記録（#188・TASK-1.8f）

イシュー #188「perf(backend-metal): TASK-1.8f 動的タイル選択（行列サイズ別パラメータ化）の実装」の実測記録テンプレート。
受け入れ条件「動的タイル選択（`dispatch_auto`）が simdgroup 版（TASK-1.8c・#40）比で性能向上を示す実測記録」に対応する。

## 状態: 実測未実施（macOS 実機なし。実装環境は Linux）

本実装セッションは Linux worktree（`.claude/worktrees/wf_13f23f1a-da1-182`）で行っており、Metal 実機
（Apple Silicon）が同一セッションで使用できない。本ファイルは計測手順・記録テンプレートのみを整備し、
実測は Apple Silicon 実機で `cargo run -p backend-metal --example gemm_bench --release` を実行した後、
下記テンプレートへ結果を転記する運用とする。実機 CI 整備自体はイシュー #42（TASK-1.8e）のスコープ。

代わりに以下は本実装セッションで検証済み:

- `cargo check` / `cargo clippy --all-targets -- -D warnings` を `--target aarch64-apple-darwin` で実行し、
  `gemm_simdgroup_tiled` の Rust／objc2 側結線（`crate::gemm`・`crate::pipeline`）がコンパイル可能であることを
  クロスターゲットビルドで確認済み（`Makefile` の `build-cross` と同一方式）
- `crate::tile`（`TileConfig`・`select`・`fallback_chain`・`validate`）は GPU 非依存の純粋関数のため、
  Linux 上の `cargo test -p backend-metal` で単体テスト済み（23 tests pass。本 PR に含む）
- `crates/backend-metal/shaders/gemm.metal` の `gemm_simdgroup_tiled` カーネル自体（MSL 構文・
  `simdgroup_load`/`simdgroup_multiply_accumulate`/`simdgroup_store` の呼び出し形）は Metal コンパイラでの
  構文検証を実施できていない（Linux 環境に Metal コンパイラが存在しないため）。**実機での最初の実行が
  構文検証を兼ねる**点に注意

## 計測手順（Apple Silicon 実機）

```sh
git fetch origin
git checkout perf/188-metal-dynamic-tile   # 本イシューの実装ブランチ
cargo run -p backend-metal --example gemm_bench --release
```

出力形式（`examples/gemm_bench.rs` 参照）:

- `size=<N>` 行: 正方形状（256/512/1024/2048/4096）で naive/tiled/simdgroup/dynamic-tile-auto の TFLOPS と
  `auto_over_simdgroup`・`simdgroup_over_naive` 比
- `shape=(<M>x<N>x<K>)` 行: 縦長（4096x512x512）・横長（512x4096x512）で simdgroup と dynamic-tile-auto の比較
  （`crate::tile::select` の tall/wide 分岐の実測対象）
- `size=<N> candidate=<label>` 行: `GemmVariant::SimdgroupTiled` 候補構成（64x64 staged・32x32 staged・
  32x32 direct）を size=2048 固定形状で明示比較（協調ロード有無の実測比較）

数値一致確認（受け入れ条件に必須の前提）:

```sh
cargo test -p backend-metal -- --ignored --nocapture
```

`tests/gemm_dynamic_tile_parity.rs` の全ケース（候補構成別・直接ロード経路・境界形状・`dispatch_auto`・
K ストレスケース）が PASS することを先に確認してから性能値を採用する。

## 実測結果（記入待ち）

### 計測環境

| 項目 | 値 |
|------|-----|
| GPU | （記入: 例 Apple M4 Max） |
| OS | （記入: macOS バージョン） |
| rustc | （記入: `rustc --version`） |
| 計測プロトコル | `bench-harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1） |
| 決定的シード | `0xC0FFEE`（`crates/backend-metal/examples/gemm_bench.rs::SEED`） |

### 正方形状（naive/tiled/simdgroup/dynamic-tile-auto）

| size | naive TFLOPS | tiled TFLOPS | simdgroup TFLOPS | dynamic-tile-auto TFLOPS | auto/simdgroup |
|------|------|------|------|------|------|
| 256  | | | | | |
| 512  | | | | | |
| 1024 | | | | | |
| 2048 | | | | | |
| 4096 | | | | | |

### 非正方形状（縦長・横長）

| shape (MxNxK) | simdgroup TFLOPS | dynamic-tile-auto TFLOPS | auto/simdgroup |
|------|------|------|------|
| 4096x512x512（縦長） | | | |
| 512x4096x512（横長） | | | |

### 候補構成別（size=2048 固定・協調ロード有無比較）

| candidate | BM | BN | BK | WM | WN | staged | TFLOPS |
|-----------|----|----|----|----|----|--------|--------|
| bm64_bn64_bk16_staged | 64 | 64 | 16 | 2 | 2 | true | |
| bm32_bn32_bk16_staged | 32 | 32 | 16 | 2 | 2 | true | |
| bm32_bn32_bk16_direct | 32 | 32 | 16 | 2 | 2 | false | |

## 選択閾値の確定（実測後に記入）

`crates/backend-metal/src/tile.rs` の `select` 関数・`CANDIDATES` は下記の暫定値である
（実測前の初期値。MLX steel の実装傾向を参考にした推定）:

- 微小形状しきい値: `SMALL = 64`（`m/n/k` のいずれかがこれ未満なら単一 simdgroup 8x8）
- 大形状しきい値: `LARGE = 512`（`m`・`n` ともこれ以上なら 64x64 staged）
- 縦長・横長判定: `ASPECT_RATIO = 2`（`m >= n*2` で縦長、`n >= m*2` で横長）

実測結果が上記閾値の境界付近（256〜1024・アスペクト比 1.5〜3 等）で候補間の優劣が閾値と食い違う場合は、
本節に確定理由（実測値の根拠付き）を記載したうえで `tile.rs` の定数・分岐条件を更新する
（`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値を採用」・実測に基づかない推測記述をしない
方針に従う）。

## 未実施・後続作業

- 本ファイルの「実測結果」節は Apple Silicon 実機での `cargo run --release` 実行後に埋める
- 選択閾値の確定後、`crate::tile::select`/`CANDIDATES` のコメント（「暫定値」の記述）を実測確定版へ更新する
- 実機 CI 整備（TASK-1.8e・#42）と関連付けて追跡する（`.claude/rules/out-of-scope-tracking.md`）
