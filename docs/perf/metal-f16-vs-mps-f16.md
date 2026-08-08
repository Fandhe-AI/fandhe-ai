# Metal f16 対 PyTorch MPS f16 計測記録（#156・TASK-8.3b）

イシュー #156「test(bench-harness): TASK-8.3b Metal f16 対 PyTorch MPS f16 の実測」の実測記録テンプレート。
REQ-8 性能下限表（`docs/spec/04-requirements.md`）の「Metal f16 対 PyTorch MPS f16」行は本イシュー着手時点で
**唯一の未実測行**であり、「未設定（v2 PoC で未実測のため）。自作カーネルでの f16 実測後に本規則で設定する」と
されていた。本ファイルは v2 で新規に自作した Metal f16 カーネル（`gemm_simdgroup_f16`）の実測手順・記録
テンプレートを整備する。**下限値そのものの確定は #158（TASK-8.3d・人間担当）であり本イシューのスコープ外。**

## 状態: 実測未実施（macOS 実機なし。実装環境は Linux）

本実装セッションは Linux worktree（`.claude/worktrees/wf_e6a16ce7-361-23`）で行っており、Metal 実機
（Apple Silicon）・PyTorch MPS 実行環境が同一セッションで使用できない。本ファイルは計測手順・記録
テンプレートのみを整備し、実測は Apple Silicon 実機で下記手順を実行した後、テンプレートへ結果を転記する
運用とする（`docs/perf/metal-gemm-dynamic-tile.md`〈#188〉・`docs/perf/cuda-gemm-mma-pipeline.md`〈#187〉と
同じ運用方針）。

**明記**: 下記「検証済み事項」に記載する数値一致回帰テスト（`tests/cpu_metal_f16_parity.rs`）・命令実在検査
（`tests/shader_source_evidence.rs`）は、Linux 環境で**型検査（`cargo check --tests --target aarch64-apple-darwin`）
のみ**を通しており、Metal 実機・Metal コンパイラ上で**一度も実行されていない**。パリティ PASS・性能実測の
いずれも本ファイル執筆時点では確認できていない。

代わりに以下は本実装セッションで検証済み:

- `cargo check -p backend-metal --tests --examples --target aarch64-apple-darwin`（`Makefile` の
  `check-cross-metal-tests` と同一方式。`--examples` を追加して `gemm_f16_bench.rs` も対象に含めた）で、
  `gemm_simdgroup_f16` の Rust／objc2 側結線（`crate::gemm::dispatch_f16_unverified`・`crate::half_buffer::MetalHalfBuffer`・
  `crate::pipeline::make_pipeline`）が**型として**コンパイル可能であることを確認済み（クロスターゲット
  ビルドのため実際のリンク・実行は検証できない。`make build-cross` は本 Linux 環境に macOS SDK が無く
  リンクエラーになるため実行できず、`cargo check`（型検査のみ）に留めている点に注意）
- `crate::pad::pad_matrix_f16`/`unpad_matrix_f16`（GPU 非依存の純粋関数）は Linux 上の
  `cargo test -p backend-metal --lib` で単体テスト済み（本 PR に含む。`objc2` 系 FFI に触れないため）
- `crates/backend-metal/src/shaders/gemm.metal` の `gemm_simdgroup_f16` カーネル自体（MSL 構文・
  `simdgroup_load`/`simdgroup_multiply_accumulate`/`simdgroup_store` の呼び出し形）は Metal コンパイラでの
  構文検証を実施できていない（Linux 環境に Metal コンパイラが存在しないため）。**実機での最初の実行が
  構文検証を兼ねる**点に注意（`metal-gemm-dynamic-tile.md` と同じ注意書き）
- `tests/shader_source_evidence.rs`（`objc2` 系 FFI に触れない `include_str!` ベースの文字列検査のみ）は
  Linux で実際に実行し green を確認済み（`gemm_simdgroup_f16_source_uses_simdgroup_half_matrix_instructions`・
  `gemm_simdgroup_f16_source_retains_req8_boundary_guard`）

## 精度契約（実装計画 §3.1 の判断。CUDA 側との差異）

`gemm_simdgroup_f16` は A・B・累算のすべてに `simdgroup_half8x8`（half 型統一）を使う。理由:
`apple-silicon` スキル（`references/msl/data-types.md`・`references/msl/simdgroup-functions.md`）のいずれにも
「A/B が half・累算が float」という混在精度オーバーロードの記載がなく、Linux 実装環境では Metal コンパイラで
実地検証もできないため、未確認のオーバーロードを推定で使わず仕様上確実に成立する単一型テンプレートを選んだ。

これは CUDA 側 WMMA f16（`crates/backend-cuda/src/kernels_wmma.rs::gemm_wmma_f16`。`f32.f16.f16.f32` 累算）とは
**異なる精度契約**である。half 累算は f32 累算より桁落ちしやすく、K が大きいストレスケース（K=4096）では
REQ-2 複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を外れる可能性が高い
（#186〈`docs/perf/cuda-tensor-core-tolerance-evaluation.md`〉で CUDA f16 WMMA も K≥512 で閾値超過が実測されて
おり、half 累算の Metal 側では同様かそれ以上の結果が想定される）。`shaders/gemm.metal` は `MM_T`/`ACC_T` の
typedef を切り出しており、実機で混在精度オーバーロードが確認できた場合は `ACC_T` の型変更（と `c` バッファ・
`MetalGemm::dispatch_f16_unverified` の出力型の追随変更）で切替可能にしてある。

**公開 API 上の扱い（PR #346 codex-review P1-2 指摘への対応）**: 上記の通り精度契約が Metal 実機・
Metal コンパイラで一度も実行検証されていないため、`MetalGemm::dispatch_f16`／`dispatch_f16_prepared` は
`dispatch_f16_unverified`／`dispatch_f16_prepared_unverified` へ改名し `#[doc(hidden)]` を付けた（公開
ドキュメントには載せず、意図的に呼び出す利用者のみが到達できるようにする）。検証（本ファイルの実測結果
記入・#158 での下限確定）が済むまで `dispatch_auto`／`dispatch_backend_auto`（production 経路）へは統合
しない。

## 計測手順（Apple Silicon 実機）

### 1. 数値一致の先行確認（受け入れ条件に必須の前提）

```sh
git fetch origin
git checkout test/156-metal-f16-vs-mps-f16   # 本イシューの実装ブランチ
cargo test -p backend-metal --release -- --ignored --nocapture cpu_metal_f16_parity
```

`tests/cpu_metal_f16_parity.rs` の全ケース（8x8x8 基準・512 基準・非倍数境界形状・K=4096 ストレス・決定性）が
PASS することを確認する。K=4096 ストレスケースが FAIL した場合も緩和せず本ファイルへ FAIL 事実
（`fail_count`・`max_abs_diff`・`max_rel_err` 等）を記録し、#158 へ引き継ぐ。

### 2. Rust 側（Metal f16）実測

```sh
cargo run -p backend-metal --example gemm_f16_bench --release
```

出力形式（`examples/gemm_f16_bench.rs` 参照）: `size=<N> metal_f16_simdgroup_tflops=<値>` 行を
size=512/1024/2048/4096（正方）で出力する。

### 3. PyTorch 側（MPS f16）実測

```sh
python3 -m venv .venv-mps-bench
source .venv-mps-bench/bin/activate
pip install torch
python3 scripts/bench/gemm_bench_torch_mps_f16.py
```

出力形式: `size=<N> pytorch_mps_f16_tflops=<値>` 行を同一形状で出力する。

## 実測結果（記入待ち）

### 計測環境

| 項目 | 値 |
|------|-----|
| GPU | （記入: 例 Apple M4 Max） |
| OS | （記入: macOS バージョン） |
| rustc | （記入: `rustc --version`） |
| torch | （記入: `python3 -c "import torch; print(torch.__version__)"`） |
| 計測プロトコル（Rust 側） | `bench-harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1） |
| 計測プロトコル（PyTorch 側） | warmup 20 回・計測 20 回・`time.perf_counter()` 中央値（`scripts/bench/gemm_bench_torch_mps_f16.py`） |
| 決定的シード | `0xC0FFEE`（両スクリプト共通） |
| 同期境界 | Rust: コマンドバッファ完了待ち／PyTorch: `torch.mps.synchronize()`（ホスト転送を伴わない完了待ち。REQ-8 v2 方針） |

### 数値一致（`cpu_metal_f16_parity.rs`。受け入れ条件の前提）

| ケース | 判定 | 備考（FAIL 時は fail_count・max_abs_diff・max_rel_err を記載） |
|------|------|------|
| f16_parity_baseline_8x8x8 | （記入: PASS/FAIL） | |
| f16_parity_baseline_shape_512 | （記入: PASS/FAIL） | |
| f16_parity_boundary_shapes_non_multiple_of_eight | （記入: PASS/FAIL） | |
| f16_k4096_stress | （記入: PASS/FAIL） | half 累算のため FAIL の可能性が高い（本ファイル「精度契約」節参照） |
| f16_dispatch_is_bit_deterministic_across_runs | （記入: PASS/FAIL） | |

### TFLOPS 比較（Metal f16 対 PyTorch MPS f16）

| size | Metal f16 TFLOPS（simdgroup） | PyTorch MPS f16 TFLOPS | Metal/PyTorch 比 |
|------|------|------|------|
| 512  | | | |
| 1024 | | | |
| 2048 | | | |
| 4096 | | | |

REQ-8 の主指標は 2048/4096（512 は起動オーバーヘッド支配のため参考値。PoC-v2-4 先例）。

### 参考: PoC-v2-4 f32 実測との対比

PoC-v2-4（Apple M4 Max・size=4096）の f32 実測: simdgroup 3.134 TFLOPS 対 PyTorch MPS（f32）13.505 TFLOPS
（比 ≒ 23.2%。`docs/spec/03-poc/poc-v2-4-metal-gemm/README.md`「計測結果」節）。f16 は理論ピーク演算性能が
f32 の約 2 倍（Apple GPU の一般的傾向）であり、Metal・PyTorch 双方が同程度にこの倍率を享受するかどうかが
比較対象になる。half 累算（本カーネルの精度契約）が性能に与える影響（f32 累算より高速か、桁落ち対策の
追加命令で相殺されるか）は実測後に本節へ追記する。

## 未実施・後続作業

- 本ファイルの「実測結果」節は Apple Silicon 実機での `cargo test -- --ignored`・`cargo run --release`・
  PyTorch スクリプト実行後に埋める
- 下限値の確定（REQ-8 性能下限表の当該行の更新）は #158（TASK-8.3d・人間担当）が行う。本ファイルの実測結果を
  入力として使う。**下限確定の判断（実測未実施のため「未設定」を維持する据え置き確定案）は
  `docs/perf/performance-floor-decision.md` を参照**
- `docs/spec/04-requirements.md`（正本 submodule）の更新は本リポでは行わない。仕様変更は spec リポ側で対応する
  （`.claude/rules/out-of-scope-tracking.md`「仕様変更が必要な場合」）
- f16 の自動ディスパッチ規則への統合（`docs/dispatch-rules-design.md`）は本イシューのスコープ外
  （実装計画 §3.4「Metal f16 行は含めない」）。REQ-11 系の後続課題として別途追跡する
- K=4096 ストレスケースが実機で複合判定を外れた場合、#186 と同じ枠組みで許容誤差の再評価が必要かどうかを
  #158 で判断する（本ファイルは事実の記録のみを担い、閾値変更の判断は行わない）。#158 時点では実機結果が
  存在しないため「判断材料なし・実測後に再評価（許容誤差は変更しない）」と記録した
  （`docs/perf/performance-floor-decision.md` §5(b)）
