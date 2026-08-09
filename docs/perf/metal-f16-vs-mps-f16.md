# Metal f16 対 PyTorch MPS f16 計測記録（#156・TASK-8.3b）

イシュー #156「test(bench-harness): TASK-8.3b Metal f16 対 PyTorch MPS f16 の実測」の実測記録テンプレート。
REQ-8 性能下限表（`docs/spec/04-requirements.md`）の「Metal f16 対 PyTorch MPS f16」行は本イシュー着手時点で
**唯一の未実測行**であり、「未設定（v2 PoC で未実測のため）。自作カーネルでの f16 実測後に本規則で設定する」と
されていた。本ファイルは v2 で新規に自作した Metal f16 カーネル（`gemm_simdgroup_f16`）の実測手順・記録
テンプレートを整備する。**下限値そのものの確定は #158（TASK-8.3d・人間担当）であり本イシューのスコープ外。**

## 状態: 数値一致は実機検証済み（イシュー #380）。TFLOPS 実測は未実施

本ファイルは当初 Linux worktree で作成され、Metal 実機・PyTorch MPS 実行環境が同一セッションで使用できな
かったため計測手順・記録テンプレートのみを整備していた。イシュー #380 で Apple Silicon 実機
（M4 Max・macOS 26.6）を用いて `tests/cpu_metal_f16_parity.rs`（数値一致回帰テスト 6 件）・
`tests/shader_source_evidence.rs`（命令実在検査）を実行し、MSL 構文検証（`gemm_simdgroup_f16` を含む
`gemm.metal` 全体のコンパイル）・数値一致（複合判定）の両方を確認済み（詳細は下記「数値一致」節）。
**TFLOPS 実測（Metal f16 対 PyTorch MPS f16 の性能比較）は #380 のスコープ外であり、引き続き未実施**
（下記「計測手順」以降のテンプレートに従い後続イシューで実施する）。

イシュー #380（Apple Silicon 実機・M4 Max・macOS 26.6・`stable-aarch64-apple-darwin`）で以下を実機検証済み:

- `crates/backend-metal/src/shaders/gemm.metal` の `gemm_simdgroup_f16`（`gemm_naive`/`gemm_tiled`/
  `gemm_simdgroup`/`gemm_simdgroup_tiled` を含む `gemm.metal` 全体）が `MetalGemm::new` の
  `newLibraryWithSource` で実機コンパイル成功する（**MSL 構文検証は完了**。当初の懸念「実機での最初の
  実行が構文検証を兼ねる」は成立し、かつ pass した）
- `tests/cpu_metal_f16_parity.rs` 6 件全件が `cargo test -p backend-metal --release -- --ignored --nocapture`
  で PASS（数値一致は下記「数値一致」節を参照）
- `tests/shader_source_evidence.rs`（`include_str!` ベースの文字列検査のみ。`cfg(target_os = "macos")`
  不要で Linux CI でも実行可能）は実機セッションでも green

Linux 実装環境時点で検証済みだった事項（`cargo check --tests --target aarch64-apple-darwin` による型検査
のみ・`crate::pad::pad_matrix_f16`/`unpad_matrix_f16` の単体テスト）は上記の実機実行で上書き・補完された。

## 精度契約（イシュー #380 の実機検証で確定。実装計画 §3.1 の half 統一判断から変更）

`gemm_simdgroup_f16` は当初（実装計画 §3.1・Linux 実装環境時点）A・B・累算のすべてに
`simdgroup_half8x8`（half 型統一）を使っていた。理由: `apple-silicon` スキル
（`references/msl/data-types.md`・`references/msl/simdgroup-functions.md`）のいずれにも「A/B が half・累算が
float」という混在精度オーバーロードの記載がなく、Linux 実装環境では Metal コンパイラで実地検証もできない
ため、未確認のオーバーロードを推定で使わず仕様上確実に成立する単一型テンプレートを選んでいた。

イシュー #380 で Apple Silicon 実機（M4 Max・macOS 26.6）を用い、`MTLDevice.makeLibrary(source:)`
ランタイムコンパイルによる spike を実施し以下が判明した:

1. `simdgroup_multiply_accumulate(simdgroup_float8x8&, simdgroup_half8x8, simdgroup_half8x8,
   simdgroup_float8x8)` は**コンパイル成功**する（A/B=half・アキュムレータ=float の混在オーバーロードは
   実在する）。
2. ただし `simdgroup_store(simdgroup_float8x8, device half*)` は**コンパイル不可**
   （診断: `deduced conflicting types for parameter 'T' ('float' vs. 'half')`）。float アキュムレータを
   half 出力バッファへ直接 store する経路は存在しない。

この 2 点から、`simdgroup_float8x8` → `threadgroup float` へ一旦 store → `threadgroup_barrier` で同期 →
スレッド単位で `(half)` へ変換して `device half*` へ書き戻す 2 段エピローグ（変種 B）を採用し、
`ACC_T` を `simdgroup_float8x8`（f32 累算）へ変更した（`shaders/gemm.metal::gemm_simdgroup_f16` 本体
参照）。変種 B を選ぶ理由: `device float*` へ直接 store する変種 A（C 転送バイト数が 2 倍になり本ファイルの
比較手法・後続 #383 の前提を壊す）ではなく、`dispatch_f16_prepared_unverified` のシグネチャ・
`MetalHalfBuffer` の `c_buf`・C バッファの転送バイト数を変えずに済むため。

この変更により CUDA 側 WMMA f16（`crates/backend-cuda/src/kernels_wmma.rs::gemm_wmma_f16`。
`f32.f16.f16.f32` 累算）と**精度契約が整合した**。half 累算時点（実装計画時点）の実機実測では
K=4096 ストレスケースで 60155/65536 要素（max_rel_err 1.992・max_abs_diff 1.562）が REQ-2 複合判定
（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を外れていたが、f32 累算化後は同一形状で複合判定 green を
実機確認済み（詳細は下記「数値一致」節）。本変更は判定式・閾値・`backend_cpu::parity` を一切変更しない
「累算精度の向上」であり、「許容誤差の緩和」ではない。

**公開 API 上の扱い（PR #346 codex-review P1-2 指摘への対応）**: `MetalGemm::dispatch_f16`／
`dispatch_f16_prepared` は `dispatch_f16_unverified`／`dispatch_f16_prepared_unverified` へ改名し
`#[doc(hidden)]` を付けている（公開ドキュメントには載せず、意図的に呼び出す利用者のみが到達できるように
する）。数値一致は #380 で実機検証済みだが、`dispatch_auto`／`dispatch_backend_auto`（production 経路）への
統合はスコープ外のため `_unverified` suffix・`#[doc(hidden)]` は当面維持する。

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

### 数値一致（`cpu_metal_f16_parity.rs`。受け入れ条件の前提。イシュー #380 実機実測。M4 Max・macOS 26.6）

累算精度契約変更（half 統一 → f32 累算。本ファイル「精度契約」節）の before/after。before は実装計画時点の
half 統一アキュムレータでの実機実測、after は #380 の f32 累算化後の実機実測。

| ケース | before（half 累算） | after（f32 累算） |
|------|------|------|
| f16_parity_baseline_8x8x8（8×8×8） | FAIL: fail_count=7/64・max_rel_err=1.171e-2・max_abs_diff=1.953e-3 | PASS |
| f16_parity_boundary_shapes_non_multiple_of_eight（17×19×23） | FAIL: fail_count=92/323・max_rel_err=4.754e-2・max_abs_diff=7.812e-3 | PASS |
| f16_parity_baseline_shape_512（512³） | FAIL: fail_count=203946/262144・max_rel_err=1.998e0・max_abs_diff=2.500e-1 | PASS |
| f16_k4096_stress（256×256×4096） | FAIL: fail_count=60155/65536・max_rel_err=1.992e0・max_abs_diff=1.562e0 | PASS |
| f16_dispatch_is_bit_deterministic_across_runs | PASS | PASS |
| f16_dispatch_prepared_rejects_undersized_and_misaligned_inputs | PASS | PASS |

6 件全件が after（f32 累算）で PASS。judgement 式・閾値・`backend_cpu::parity` は変更していない
（`git diff -- crates/backend-cpu/src/parity.rs` は空）。

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
