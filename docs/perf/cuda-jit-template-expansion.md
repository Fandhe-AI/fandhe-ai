# CUDA JIT カーネルテンプレート展開（shape/タイル/段数）記録（#516・Phase C-6）

イシュー #516「shape/タイル/段数のテンプレート文字列展開によるカーネルソース生成を実装」の設計要約・PTX/SASS 確認手順・実測記録テンプレート。
親イシュー #503（Phase C: CUDA JIT shape 特化・コンパイルキャッシュ・静的タイル選定）の C-6。C-1（`CudaKernelDescriptor`・#504）・C-8（`derive_pipeline_stages`・#521）を踏まえ、後続の C-7（#519 次元ごとの定数化選択）・C-9（#524/#527 タイル選定）・C-2〜C-5（キャッシュ系）が消費する生成機構を提供する。

## 状態: 未実測・実機実行待ち（DGX Spark GB10 実測は #531・#534 へ引き継ぐ）

本実装セッションの環境には NVRTC（`libnvrtc`）が存在せず（`crates/backend-cuda/src/nvrtc.rs` 冒頭コメント「A03」節・`kernels_mma.rs` 冒頭コメント「検証状態」参照）、特化 render 経路が生成する CUDA ソースは **NVRTC による構文検証を一度も通過していない**。本ファイルは以下 2 つを分けて記録する:

- 通常 CI で機械検証済みの事項（§4）: 文字列レベルのテンプレート展開検査（`#define` 置換・REQ-8 境界チェック needle 残存・fail-closed 拒否・決定性）
- 実機必須・未実測の事項（§5）: NVRTC コンパイル成否・PTX 命令出現数・`cuobjdump -sass` によるコンパイル時展開（定数伝播・ループ unroll）の確認

## 1. 設計要約

### 1.1 置換ベースのテンプレート展開（既存静的ソース = テンプレート本体兼デフォルトインスタンス）

`kernels_mma.rs::MMA_F16`・`kernels_wmma_opt.rs::WMMA_TF32_F32_OPT`／`WMMA_F16_OPT` の静的カーネルソース文字列自体をテンプレート本体として維持し、`render_mma_f16`／`render_wmma_tf32_opt`／`render_wmma_f16_opt`（いずれも `pub(crate)` ではなく `pub`。呼び出し元は `gemm_mma.rs`／`gemm_wmma.rs` 相当の同クレート内経路を想定）が [`CudaKernelDescriptor`](../../crates/backend-cuda/src/nvrtc.rs) 相当の検証済み構成型（`MmaKernelConfig`／`WmmaOptKernelConfig`）から `#define` 行を `format!` で組み立て直す。

実装計画（Plan フェーズ）は新規モジュール `kernel_template.rs` に置換エンジンを切り出す設計を想定していたが、実装は各カーネルファイル（`kernels_mma.rs`・`kernels_wmma_opt.rs`）内に render 関数を直接持つ構成へ収束した。**これは意図的な位置の乖離であり、置換対象のテンプレート本体・タイル定数・コンパイル時 `const _: () = assert!(...)` 契約検査と同一ファイル内に render 関数を置くことで、テンプレート文字列と展開ロジックの乖離（`#define` 行の文言変更に render 側が追従し忘れる回帰）を型・視認性の両面で抑える判断による**。置換対象出現数の検証・境界チェック needle 検査は当初計画どおり保持している（§4）。

### 1.2 shape（M/N/K）焼き込み: マクロ間接化方式（計画のプリプロセッサガード方式からの変更）

実装計画は `#if defined(GEMM_SHAPE_M) m = GEMM_SHAPE_M; ... #endif` という「カーネル本体冒頭で実行時形状引数を上書きする」ガードブロックを想定していたが、実装は次の形へ単純化されている:

- カーネル本体全体を `m`/`n`/`k` の直接参照ではなく `DIM_M`/`DIM_N`/`DIM_K` マクロ経由の参照へ書き換え済み
- `DimSpec::Dynamic` の場合 `#define DIM_M m`（マクロがカーネル引数名へ展開されるだけなので、字面上のトークン列はデフォルト経路と完全に同一になる）
- `DimSpec::Static(value)` の場合 `#define DIM_M <value>`（コンパイル時定数化。境界比較・ループ境界の定数伝播が有効になる）

この方式は計画の「未定義時はプリプロセッサ除去で意味不変」という目標を、実行時分岐（`#if`）を経由せず**マクロ置換の字面一致**で達成する（`DimSpec::Dynamic` は常に `#define DIM_M m` を生成するため、デフォルト構成の展開結果は「`m`/`n`/`k` の生トークンを `DIM_M`/`DIM_N`/`DIM_K` に機械的に置き換えただけ」のソースになり、プリプロセッサ後のトークン列はデフォルト（旧）カーネルと同一になる）。

3 次元のうち一部のみ静的化する組み合わせ（`render_mma_f16_specializes_tile_and_static_dim` テストが検査する `dim_m=Static/dim_n=Dynamic/dim_k=Dynamic` 等）は既に生成 API 側でサポート済み。「どの次元を焼き込むか」の選択ポリシー自体は C-7（#519）のスコープであり、本タスクは全 3 次元の個別焼き込みが可能な生成 API を提供するのみに留める。

### 1.3 生成 API の構造

計画の `GeneratedKernelSource` に相当する `RenderedMmaKernel`／`RenderedWmmaOptKernel` は、生ソース文字列を外部へ返す公開メソッドを持たない（`#[cfg(test)]` 限定の `source()` アクセサのみ）。ソースの受け渡し先は `Self::compile`（NVRTC コンパイル・固定エントリポイントのロード）内部に限定し、コンパイル済み `CudaFunction` は展開元の `cfg`（構成情報）と不可分に束ねた `CompiledMmaKernel`／`CompiledWmmaOptKernel` としてのみ取得できる。これは実装計画 §3.3 が要求した「消費者は crate 内実装に限定」「起動時形状検査を型で強制」という契約を、PR レビュー（PR #643 codex-review 複数ラウンド）を経てより厳格化した結果である（各型のドキュメンテーションコメント参照）。

### 1.4 A03（インジェクション）契約の改定

`nvrtc.rs::compile_ptx` の `src` 引数契約を「`&'static str` 限定」から「静的テンプレート、または検証済み数値・enum パラメータのみから決定的に組み立てた `String` も許容（外部入力文字列の連結は引き続き禁止）」へ改定した（`nvrtc.rs` 冒頭ドキュメンテーションコメント参照）。

## 2. 数値 bit 一致・parity 非後退の論拠

デフォルト構成（`MmaKernelConfig` の全次元 `Dynamic`・タイル値が既存 `MMA_BM`/`MMA_BN`/`MMA_BK`/`MMA_STAGES` と同一）で render した結果は、§1.2 のとおりプリプロセッサ後のトークン列がデフォルト（旧）カーネルソースと完全に一致する。したがって:

- `tests/parity_nonregression.rs` のベースライン fixture・tolerance 定数（§1.2 契約）は無変更（本 PR の `git diff` でも該当ファイルに差分がないことを確認済み）
- FMA 契約（CPU 参照実装の `f32::mul_add`・GPU 側既定 FMA 契約。`.claude/rules/coding-rust.md`）に本タスクは非接触（テンプレート展開はソース文字列の組み立てのみで、演算命令列・アキュムレート順序を変更しない）
- 特化構成（非デフォルト値）はディスパッチ・実行経路へまだ結線されていない（結線は C-7 #519 のスコープ）ため、本 PR 単体では実行時の数値挙動に一切影響しない

## 3. PTX/SASS でのコンパイル時展開確認手順（実機必須・DGX Spark 実行時の手順書）

実機（DGX Spark GB10・sm_121。または NVRTC を持つ任意の CUDA 環境）で以下を実行する。

```bash
# 1. 実機必須テストを実行し、NVRTC コンパイル成否・PTX 命令出現数を確認する
cargo test -p backend-cuda --release -- --ignored --nocapture

# 2. 個別に PTX テキストを確認したい場合（例示。実際のエントリポイントは
#    render_mma_f16 の compile() が使う "gemm_mma_f16" 等）:
#    - CudaKernelDescriptor / MmaKernelConfig で非デフォルト構成を作り
#      render_mma_f16(&cfg) → RenderedMmaKernel::source()（テスト専用）で
#      文字列を取得し、一時ファイルへ書き出した上で nvrtc 経由でコンパイル
#      した Ptx をテキストとしてダンプする（cudarc::nvrtc::Ptx の内部表現）
```

確認観点:

1. **コンパイル成功**: NVRTC が特化ソースを構文エラーなく受理すること
2. **定数伝播・unroll の確認**: PTX テキストで `mma.sync.aligned.m16n8k16` の出現回数が `WARP_TILES_M * WARP_TILES_N * K_STEPS` 以上（`#pragma unroll` によるループ展開が有効であることの間接確認。K_STEPS は `BK / MMA_K`）
3. **デフォルト経路との差分確認**: `DimSpec::Static` を使わないデフォルト構成の PTX と、shape 焼き込みを使った特化構成の PTX を比較し、後者でのみ境界比較・アドレス計算に即値が現れること（`cuobjdump --dump-sass` で SASS レベルの定数畳み込みも確認する）

```bash
# SASS 確認（cubin を経由する場合。cudarc の Ptx→cubin ロード経路や
# 別途 nvrtc から cubin 出力を得る手順は実機側の CUDA toolkit 構成に依存）
cuobjdump -sass <compiled.cubin> | less
```

実測結果は本節に追記する（未実測。DGX Spark 実機タスク #531／#534 のスコープ）。

## 4. 通常 CI で機械検証済みの事項（本 PR 内で完結）

- `cargo fmt --all --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-features`（実機依存 `#[ignore]` は除外）:
  - デフォルト値 config での render 結果が「静的ソース + `#define` 群が値として同一」であること（`render_mma_f16_specializes_tile_and_static_dim` 等の特化系テスト・既存 `mma_tile_constants_match_kernel_source_defines` 相当）
  - 非デフォルト値（タイル・段数・shape 静的化の組み合わせ）での render 結果に指定値の `#define` と `DIM_M`/`DIM_N`/`DIM_K` 焼き込みが正しく現れること
  - render 結果にも REQ-8 境界チェック needle（`gr < DIM_M && gc < DIM_K` 等）が残存すること（`mma_f16_source_retains_req8_boundary_guards` の焼き込み版）
  - 不正 config（dtype 不一致相当・SMEM 超過・割り切り違反・スレッド数超過・ゼロ次元等）が `CudaError::InvalidKernelConfig` で拒否されること
  - **決定性**（同一 config で 2 回 render して byte 一致。本 PR で追加した `render_mma_f16_is_deterministic_for_same_config`・`render_wmma_tf32_opt_is_deterministic_for_same_config`・`render_wmma_f16_opt_is_deterministic_for_same_config`。C-5（#514 ソース断片ハッシュ）・C-2（#506 キャッシュディレクトリ命名）が本 render 出力をキャッシュキー材料にする前提の契約）
  - 起動時形状不一致（`Static` 焼き込み値と実引数の食い違い）が `validate_launch_shape` で fail-closed に拒否されること
- `cargo build --workspace --locked`（CUDA toolkit 非搭載環境でのビルド成立契約の維持）
- `tests/common/parity_baseline`・tolerance 定数の無差分確認（`git diff` で機械確認）

## 5. 未検証・実機実行待ちの事項

- NVRTC によるカーネルソースの構文検証そのもの（デフォルト・特化構成いずれも）
- PTX テキストでの `mma.sync`/`wmma` 系命令の unroll 期待回数確認・`cuobjdump -sass` によるコンパイル時定数畳み込みの確認（§3）
- 生成ソースのディスパッチ・実行経路への結線後の実測（C-7 #519 のスコープ）
- 段数逆算（`derive_pipeline_stages`・C-8 #521）との統合実測

`cargo test -p backend-cuda --release -- --ignored --nocapture` は本実装セッションの環境（NVRTC 非搭載）では実行不能なため未実行のまま。実機実測は #531／#534 へ引き継ぐ。
