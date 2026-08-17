# Transformer 複合ワークロード ベースライン計測プロトコル（Phase G-4・#589）

親 #582（Phase G「融合 RMSNorm / online softmax / 量子化キャストによる Transformer 複合ワークロード
改善」）の評価軸「Transformer 複合ワークロード」について、ワークロード定義・計測プロトコル・
比較対象 PyTorch 構成・評価方式を確定する文書。イシュー #589（TASK-8.4／Phase G-4）の成果物。

## 1. 位置づけ

- 本文書が対象とする「Transformer 複合ワークロード」は、`docs/performance-targets.md` §2
  「v2 段階的下限表」の GEMM 単体 5 行（CPU／CUDA f32・f16／Metal f32・f16。
  `crates/bench-harness/src/threshold.rs::floor_spec` の合否判定対象）とは**別行・別系列**である。
  同表の当該行は「初期リリース: 下限を設定しない／最適化後: 下限を設定しない」（REQ-8 の
  受け入れ基準・`docs/performance-targets.md:30`）であり、本文書の確定内容もこの位置づけを
  変更しない
- Phase G は本行を実装対象とする最初のフェーズであり、本イシュー（G-4）は実機計測に先立つ
  「定義・プロトコルの確定」を担う準備タスクである。実機計測そのものは G-12（#602・CUDA 実機）／
  G-14（#605・Metal 実機）のスコープ

## 2. ワークロード定義（確定）

単一真実源は `bench_harness::transformer_workload::baseline_spec()`
（`crates/bench-harness/src/transformer_workload.rs`。#589）。

| 項目 | 値 |
|------|-----|
| 層数（`num_layers`） | 1 |
| 隠れ層次元（`d_model`） | 512 |
| ヘッド数（`n_heads`） | 8（`head_dim = d_model / n_heads = 64`） |
| FFN 中間層次元（`d_ff`） | 2048 |
| バッチサイズ（`batch`） | 8 |
| 系列長（`seq_len`） | 128 |
| 精度 | f32 |
| Attention | あり（Multi-Head Self-Attention。スケーリング `1/sqrt(head_dim)`・softmax） |
| Activation | GELU（`0.5*x*(1+erf(x/sqrt(2)))` の erf 合成。ONNX に `Gelu` 単体オペが無く
  `transformer.onnx` フィクスチャも同じ合成方式を採るため。REQ-7 準拠） |
| Norm 配置 | post-norm（`Add(残差) → LayerNormalization` の順。`transformer.onnx` の
  `norm_first=false` と整合。`layer_norm_eps=1e-5`） |

この形状・決定的シード（`SEED = 155_083`）は PoC-8 定義（PoC-5 流用）を踏襲し、
#155（TASK-8.3a）実装当時のテストファイル内ローカル定数と同値である（#589 で
`bench_harness::transformer_workload` モジュールへ単一真実源化。挙動不変）。

## 3. 計測プロトコル（確定）

`docs/performance-targets.md` §4「計測プロトコル」に準拠する。

- warmup 20 回以上・計測 20 回以上の中央値・Q1/Q3 を記録する
  （`bench_harness::protocol::MeasurementConfig` の下限 `MIN_ITERATIONS = 20`）
- 決定的シードは `bench_harness::rng::Xorshift64Star`、値は
  `bench_harness::transformer_workload::SEED`（`155_083`）
- 同期方式は「ホスト転送を伴わない完了待ち」で統一する（バックエンド固有 API に委ねる）
  - CPU: 該当なし（`sync::CpuSync`。`ops.gemm` 等がホスト常駐 `Tensor<f32>` を同期的に
    返す契約のため、ワークロードクロージャの戻り時点で計測対象処理は完了している）
  - CUDA: `stream.synchronize()`
  - Metal: コマンドバッファ完了待ち（`device.poll(PollType::wait_indefinitely())`）
- 出力は `bench_harness::report::BenchReport::to_json`（`schema_version` 付き構造化 JSON）

## 4. 比較対象 PyTorch 構成（確定）

- バージョン: `torch==2.13.0`
- モデル: `nn.TransformerEncoderLayer`（`dropout=0.0`・`activation="gelu"`・`norm_first=False`）
- dtype: `float32`
- デバイス別実行環境:
  - CPU／MPS: Apple M4 Max（macOS arm64 wheel、`torch==2.13.0`）
  - CUDA: DGX Spark GB10（`torch==2.13.0+cu130`。`docs/perf/cuda-floor-remeasurement.md`
    の実測環境と整合させる）
- 実行スクリプト: `scripts/bench-transformer-pytorch.py`
  （分位点定義は `bench_harness::stats::median_q1_q3` と同方式:
  ソート後 `idx = round(p*(n-1))` 番目の要素を採用）

## 5. 評価方式（確定）

- **QEMU 参考値（約 6.1%。`docs/perf/transformer-workload-measurement.md`）は目標の分母にしない。**
  同文書自身が明記する通り、非実機（QEMU 仮想 CPU）・attention 行列積の naive 経路混入という
  2 重の下振れ要因を含む参考値であり、性能改善の基準値としては使わない
- Phase G の評価は G-12（#602・CUDA 実機）／G-14（#605・Metal 実機）で取得する
  **実機ベースラインからの相対改善**（改善前 → 改善後の対 PyTorch 比の変化）で行う。
  CUDA／Metal での複合ワークロード実行経路の実装自体は G-12／G-14 のスコープであり、
  本イシュー（G-4）は現行 CPU 経路（`crates/bench-harness/tests/transformer_workload.rs`）の
  定義確定に閉じる

## 6. REQ-8 下限

- 本行は元来「下限を設定しない」行であり（`docs/performance-targets.md:30`）、Phase G 完了後も
  **下限を設定しない**（REQ-8 の丸め規則〈F-5・人間承認タスク〉の適用対象外）
- `crates/bench-harness/src/threshold.rs::floor_spec` へ本系列を追加しない
  （GEMM 単体 5 行との系列分離。§1 参照）

## 7. ベースライン記入枠（G-12／G-14 が転記）

`docs/perf/cuda-tensor-core-measurement.md`・`docs/perf/transformer-workload-measurement.md`
の確立済み先例形式を踏襲する。CUDA／Metal での複合ワークロード実行経路の実装自体は
G-12／G-14 のスコープであり、下表は実測後の記入枠として本イシューで確定するのみ。

### CPU（Apple M4 Max）

| 段階 | 実装 | median | Q1 | Q3 | 対 PyTorch 比 | commit SHA | 実施日 |
|------|------|------|------|------|------|------|------|
| 改善前 | Rust（自作コア） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） |
| 改善後 | Rust（自作コア。融合 RMSNorm / online softmax / 量子化キャスト適用後） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） |
| 基準 | PyTorch 2.13.0 | （記入） | （記入） | （記入） | 1.00（基準） | （記入） | （記入） |

### CUDA（DGX Spark GB10。G-12・#602 のスコープ）

| 段階 | 実装 | median | Q1 | Q3 | 対 PyTorch 比 | commit SHA | 実施日 |
|------|------|------|------|------|------|------|------|
| 改善前 | Rust（自作コア） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） |
| 改善後 | Rust（自作コア。融合 RMSNorm / online softmax / 量子化キャスト適用後） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） |
| 基準 | PyTorch 2.13.0+cu130 | （記入） | （記入） | （記入） | 1.00（基準） | （記入） | （記入） |

### Metal（Apple M4 Max。G-14・#605 のスコープ）

| 段階 | 実装 | median | Q1 | Q3 | 対 PyTorch 比 | commit SHA | 実施日 |
|------|------|------|------|------|------|------|------|
| 改善前 | Rust（自作コア） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） |
| 改善後 | Rust（自作コア。融合 RMSNorm / online softmax / 量子化キャスト適用後） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） |
| 基準 | PyTorch 2.13.0（MPS） | （記入） | （記入） | （記入） | 1.00（基準） | （記入） | （記入） |

イシュー #605（G-14）で GEMM epilogue 実融合化（elementwise 5 演算・
`gemm_bias_act`）を実装したが、実行環境が Linux のため Metal 実機計測は
未実施のまま（`docs/perf/metal-gemm-epilogue-fusion.md`「実機検証未完・
ブロック中」節）。複合ワークロード実行経路自体（`transformer_workload_metal.rs`）
の実装は本イシューでは着手せず別イシューへ切り出したため、上表の記入は
その別イシュー完了後になる。

## 8. 共通契約の遵守宣言

本イシュー（#589）はワークロード定義・計測プロトコルの確定文書化のみを行い、以下を変更しない。

- カーネル境界チェックの省略なし（`.claude/rules/coding-rust.md` REQ-8 節）
- バックエンド間数値一致テストの許容誤差（tolerance）の緩和なし
- 依存クレートの追加・更新なし（`Cargo.toml`／`Cargo.lock` 差分なし）
- `docs/spec/`（正本 submodule）の編集なし
- REQ-8 下限値・`crates/bench-harness/src/threshold.rs::floor_spec` の変更なし

## 9. 関連文書

- `docs/performance-targets.md`（§2 v2 段階的下限表・§4 計測プロトコル。正本の反映済み文書）
- `docs/perf/transformer-workload-measurement.md`（#155。QEMU 参考実測記録・実機記入待ちテンプレート。
  本文書はその評価方式の確定先）
- `crates/bench-harness/src/transformer_workload.rs`（形状・シードの単一真実源）
- `crates/bench-harness/tests/transformer_workload.rs`（CPU 経路の実測実装。#155）
