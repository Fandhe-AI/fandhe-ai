# CUDA 最適化後下限（暫定 40%）再実測 記録（#157・TASK-8.3c）

イシュー #157「test(bench-harness): TASK-8.3c CUDA 最適化後下限（暫定 40%）の再実測」の実測記録テンプレート。
受け入れ条件「実測記録と候補下限値」に対応する。

## 目的・受け入れ条件対応

`docs/spec/04-requirements.md`「CUDA f32 対 PyTorch CUDA」「CUDA f16 対 PyTorch f16」行の最適化後下限
**40%** は次の理由で**暫定値**として記録されている:

> tensor core（WMMA/mma）化により、cuBLAS の f16 tensor core 経路の実効値（4096: 97.6 TFLOPS）に対する
> 一般的な手書き GEMM の到達目安 40〜70%（PoC-v2-3「tensor core 化の段階見積もり」節）を適用すると
> 39〜68 TFLOPS 相当となる。この見積もりは PyTorch **f16**（tensor core 経路）を基準としたものであり、
> PyTorch **f32**（4096 実測 17.8 TFLOPS）を基準に換算すると非現実的な外挿になる。よって f32 の最適化後
> 下限は当該見積もりをそのまま流用せず、保守的な値として PyTorch f32 比 **40%** を暫定目標とする。
> **tensor core 実装完了後の実測で本値を再確定すること**

（`docs/spec/04-requirements.md:180-181`）。`docs/spec/05-tasks.md` TASK-8.3 も同じ再確定条件を明記して
いる:「CUDA f32/f16 の最適化後下限（暫定値 40%）は Tensor Core 実装完了後の実測で再確定する（REQ-11 の
TASK-11.1 完了が前提）」（`docs/spec/05-tasks.md:286`）。TASK-11.1 系（#59〜#65・#186・#187）は完了済み
のため、本イシューがその再実測を担う。

TASK-8.3 の担当欄は「共同（計測実行は Claude Code、下限値の最終確定は人間）」（`docs/spec/05-tasks.md:290`）
であり、**本ドキュメントは候補下限値の導出と記録までを行い、最終確定は #158（TASK-8.3d）へ引き継ぐ**。

## 実測バイナリ

`crates/backend-cuda/examples/cuda_floor_bench.rs`（本イシューで新規追加）。

- 計測経路: tiled f32（基準）／WMMA(TF32) opt（f32 最良。opt 不可時は `CudaGemm::run_wmma_tf32` 内部で
  基本版へ自動フォールバック）／WMMA f16 opt／`mma.sync` f16 パイプライン（f16 最良）
- 形状: M=N=K = 512／2048／4096（PoC-v2-3 の PyTorch 参照値と同一形状）
- 計測プロトコル: `bench_harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1）・
  決定的シード `0xC0FFEE`
- 判定対象形状: REQ-8 の規定どおり **M=N=K=2048・4096 の実測比率の最小値**を候補下限値の算出に用いる。
  512 は参考値としてのみ出力する（ディスパッチ・起動オーバーヘッドが支配的で試行間ばらつきが大きいため。
  `docs/spec/04-requirements.md`「判定対象形状」節）
- 丸め規則: `docs/spec/04-requirements.md`「丸め規則の統一」節（実測比率 10% 以上は 5% 刻み切り下げ、
  10% 未満は 1% 刻み切り下げ、条件付き追加ステップなし）を `cuda_floor_bench.rs::floor_round` に純関数
  として実装（単体テストで仕様例〈10.3%→10%・26.6%→25%・1.9%→1%・境界 10%→10%〉と突合済み）。
  `bench-harness` の TASK-8.2 下限判定モジュール（#151〜#153）は本イシュー着手時点で未マージのため
  インライン実装とした。マージ後は #158/#159 で `bench-harness` 側の公開 API へ一本化する
  （out-of-scope-tracking 対応）
- GPU 名が `GB10` を含まない場合は警告行を出力する（PoC-v2-3 参照値と計測機が異なるため比率は参考値。
  REQ-8「いずれも同一ハードウェア上の同一バックエンド比較」）

## 計測手順（DGX Spark GB10 等 CUDA 実機）

```sh
git fetch origin
git checkout test/157-cuda-floor-remeasurement   # 本イシューの実装ブランチ

# 1. 数値一致確認を先に行う（既存 parity テスト群。閾値は緩和しない）
cargo test -p backend-cuda --release -- --ignored

# 2. 再実測バイナリを実行
cargo run -p backend-cuda --example cuda_floor_bench --release
```

出力形式（`crates/backend-cuda/examples/cuda_floor_bench.rs::main` 参照）:

- `WARNING: ...` 行（GPU 名が GB10 系でない場合のみ）: PyTorch 参照値との比較が参考値に留まる旨
- `device: name=... compute_capability=...` 行: 計測環境（下表「計測環境」への転記元）
- `size=<N> tiled_f32_tflops=... wmma_tf32_tflops=... wmma_f16_tflops=... mma_f16_tflops=... f32_best_over_pytorch=... f16_best_over_pytorch=...` 行:
  形状ごとの経路別 TFLOPS・対 PyTorch 比（f32 は WMMA(TF32) opt 優先、f16 は `mma.sync` 優先の最良値）
- `CUDA f32 candidate optimized floor ... = N%` / `CUDA f16 candidate optimized floor ... = N%` 行:
  判定対象形状（2048/4096）の最小比率に丸め規則を適用した候補下限値

### PyTorch 参照値の扱い

第一義は**同一実機での PyTorch 再計測**（`docs/spec/03-poc/poc-v2-3-cuda-gemm/code/` の計測スクリプトを
再実行し、同一プロトコル・同一シードで再取得する）。再計測不可の場合は PoC-v2-3 記録値（同一機 GB10 実測、
`docs/spec/03-poc/poc-v2-3-cuda-gemm/README.md`「計測結果」節）を `cuda_floor_bench.rs` に定数として埋め込んだ
値を使用し、その旨を下表「計測環境」に明記する。

PoC-v2-3 実測値（`torch.matmul`, CUDA, DGX Spark GB10, PyTorch 2.13.0+cu130, 5〜20 回中央値）:

| M=N=K | PyTorch f32 (TFLOPS) | PyTorch f16 (TFLOPS) |
|-------|----------------------|----------------------|
| 512   | 7.8803  | 17.1898 |
| 2048  | 17.4241 | 91.2115 |
| 4096  | 17.7774 | 97.6308 |

## 実測結果（記入待ち）

本セッションの実行環境は Linux（RTX 3060、compute capability 8.6）で、CUDA driver は利用可能だが
**libnvrtc（NVRTC）が非搭載**のため、tiled f32・WMMA(TF32)・WMMA f16・`mma.sync` の全カーネル経路が
初期化失敗し理由表示付きで graceful skip する（`cuda_floor_bench.rs` の環境適応分岐。実行ログは
「動作確認（本セッション実施済み）」節参照）。DGX Spark GB10 等の CUDA+NVRTC 実機はこのセッションから
アクセス不可のため、以下は実機実行後に転記する記入待ちテンプレートとして固定する
（先例: `cuda-tensor-core-measurement.md`・`cuda-gemm-mma-pipeline.md` と同運用）。

### 計測環境

| 項目 | 値 |
|------|-----|
| GPU（`CudaDevice::name()`） | （記入: 例 NVIDIA GB10） |
| compute capability（`CudaDevice::compute_capability()`） | （記入: 例 (12, 1)） |
| driver バージョン | （記入: `nvidia-smi` 出力等） |
| rustc | （記入: `rustc --version`） |
| commit SHA | （記入） |
| 実施日 | （記入） |
| PyTorch 参照値の出典 | （記入: 同一機再計測 or PoC-v2-3 記録値流用のいずれか） |
| 計測プロトコル | `bench_harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1） |
| 決定的シード | `0xC0FFEE`（`cuda_floor_bench.rs::SEED`） |

### 経路×形状 TFLOPS 実測

| M=N=K | tiled f32 | WMMA(TF32) opt | WMMA f16 opt | mma.sync f16 |
|-------|-----------|-----------------|--------------|--------------|
| 512（参考値） | | | | |
| 2048 | | | | |
| 4096 | | | | |

### 対 PyTorch 比

| M=N=K | f32 最良（WMMA(TF32) opt 優先） / PyTorch f32 比 | f16 最良（mma.sync 優先） / PyTorch f16 比 |
|-------|----------------------------------------------------|----------------------------------------------|
| 512（参考値） | | |
| 2048 | | |
| 4096 | | |

### 丸め適用後の候補下限値

| 精度 | 判定対象形状の最小比率（2048/4096） | 丸め規則適用後の候補下限値 | 現行暫定値（40%）との比較 |
|------|--------------------------------------|------------------------------|------------------------------|
| f32  | （記入）% | （記入）% | （記入: 上回る/下回る/一致） |
| f16  | （記入）% | （記入）% | （記入: 上回る/下回る/一致） |

### 暫定 40% との比較所見（記入待ち）

（記入: 候補下限値が暫定 40% を上回った場合／下回った場合それぞれの要因分析。tensor core 化前提の
見積もり〈PoC-v2-3「tensor core 化の段階見積もり」節、40〜70% 到達目安〉との整合も記載する）

## 動作確認（本セッション実施済み）

実機（CUDA+NVRTC）が利用できないため、以下でバイナリ・丸め規則の正しさを確認した:

- `cargo build --workspace --locked` — `cudarc` 動的ロード契約（CUDA toolkit 非搭載環境でもビルド成立する。
  `.claude/rules/coding-rust.md`）を崩していないことを確認済み
- `cargo build -p backend-cuda --example cuda_floor_bench --release` — example のビルド成立
- `cargo fmt --all -- --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo run -p backend-cuda --example cuda_floor_bench`（本環境。NVRTC 非搭載）—
  tiled f32・WMMA(TF32)・WMMA f16・`mma.sync` の全経路が初期化失敗を検出し理由表示付きで graceful skip、
  パニックなしで正常終了することを確認
- `cargo test -p backend-cuda --example cuda_floor_bench` — `floor_round` の単体テスト 3 件
  （仕様例との突合・10% 境界を跨ぐ非減少性・非有限値/負値の防御）が green であることを確認

## 役割分担（二重管理を避ける）

- **#158（TASK-8.3d・人間判断）**: 本ドキュメントの候補下限値を受け取り、下限の最終確定・
  `docs/spec/04-requirements.md` への反映判断を行う。`docs/spec/` の更新自体は spec リポジトリ
  （Fandhe-AI/rust-ai-library-spec）側で対応する（本リポでは編集しない）
- **`docs/spec/v2-amendment-proposal-2026-08-06.md`**（改定提案ドラフトが存在する場合）: 下限＝回帰検知
  ラインとし目標 90% を別レイヤ化する改定案との関係整理は #158 側で行う
- **`docs/performance-targets.md`（TASK-8.4・#159）**: 段階的下限の一覧整備。本ドキュメントは #157
  固有の実測記録に限定し、全バックエンド横断の一覧化は #159 に委ねる
- **丸め規則のモジュール一本化**: `bench-harness` の TASK-8.2 下限判定モジュール（#151〜#153）マージ後、
  `cuda_floor_bench.rs::floor_round` のインライン実装は削除し公開 API へ委譲する（#158/#159 実施時に対応）

## 未実施・後続作業

- 本ファイルの「実測結果」節は DGX Spark GB10 等 CUDA+NVRTC 実機での
  `cargo run -p backend-cuda --example cuda_floor_bench --release` 実行後に埋める
  （実機アクセス確保後の作業）
- 丸め規則の `bench-harness` モジュール一本化（#151〜#153 マージ後、#158/#159 で対応）
- 候補下限値の最終確定・REQ-8 反映判断（#158・人間判断）
