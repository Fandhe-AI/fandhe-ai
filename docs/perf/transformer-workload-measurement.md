# Transformer 複合ワークロード実測記録（#155・TASK-8.3a）

イシュー #155「test(bench-harness): TASK-8.3a Transformer 複合ワークロードの実測」（親 #154・TASK-8.3）の実測記録。
受け入れ条件「実測記録（中央値・Q1/Q3）が残されている」に対応する。

REQ-8（`docs/spec/04-requirements.md`）は複合ワークロード系（Transformer 推論等）について
「v2 自作カーネルでは未実測のため下限を設定しない。自作カーネルでの Transformer ブロック実測後、
本要件の丸め規則で下限を設定する」を受け入れ基準として残しており、本ファイルはその実測を提供する。

## 状態: 本実装セッション環境（x86_64 Linux・QEMU 仮想 CPU）で実測完了。実機（Apple M4 Max・DGX Spark GB10）実測は記入待ち

REQ-8 の比較実機は CPU/Metal が Apple M4 Max、CUDA が DGX Spark GB10 であり、本セッションの実行環境
（`model name: QEMU Virtual CPU version 2.5+`・12 vCPU・CUDA toolkit 非搭載）はいずれにも該当しない。
そのため本ファイルの実測値は **下限確定（#158・TASK-8.3d）の正式根拠ではなく、参考実測**の位置づけとする。
実機値は下表の記入待ちテンプレートに転記する運用とし、`docs/perf/cuda-tensor-core-measurement.md` の
確立済み先例形式を踏襲する。

## ワークロード形状（PoC-8 定義。PoC-5 流用）

`d_model=512, n_heads=8, d_ff=2048, batch=8, seq_len=128, num_layers=1`・activation=GELU
（`0.5*x*(1+erf(x/sqrt(2)))` 合成）・post-norm（`Add → LayerNormalization`。`transformer.onnx`
フィクスチャの `norm_first=false` と整合）。

## 実装経路（性能解釈に影響するため明記する）

- **Rust 側（`crates/bench-harness/tests/transformer_workload.rs`）**:
  - QKV／出力射影・FFN 2 層の 2D GEMM は `backend_cpu::CpuBackendOps::gemm`（BLIS 型・rayon 並列の
    最適化済み自作カーネル）を経由する。REQ-8 が求める「自作カーネルでの Transformer ブロック実測」の主対象
  - attention 内のバッチ行列積（`Q @ K^T`・`softmax(...) @ V`）は `onnx_interop::ops::matmul`
    （`numpy.matmul` 準拠のバッチ対応 **naive** 実装）を使用する。ヘッド単位 2D GEMM ループへの分解は
    行っていない。**この naive 経路が計測値に含まれており、最適化済みカーネルのみで構成した場合と比べて
    過小評価方向であることに注意する**（実装計画 §8「attention 行列積の経路」参照）
  - softmax・LayerNormalization・Erf（GELU 合成）・残差 Add は `onnx_interop::ops` の naive 実装をそのまま使用する
- **PyTorch 側（`scripts/bench-transformer-pytorch.py`）**: `nn.TransformerEncoderLayer`
  （`dropout=0.0`・`activation="gelu"`・`norm_first=False`）1 層。内部カーネル（`aten::addmm`・
  `aten::softmax` 等)は PyTorch 標準実装であり、Rust 側 naive 経路との構成非対称性がある

## 計測手順

```sh
git fetch origin
git checkout test/155-transformer-workload-bench   # 本イシューの実装ブランチ

# Rust 側（自作コア。CPU・BLIS 型並列 GEMM 経路）
cargo test -p bench-harness --release --test transformer_workload -- --ignored transformer --nocapture

# PyTorch 側（計測専用の一時 venv。torch==2.13.0 は Cargo 依存に追加しない）
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0"
/path/to/venv/bin/python scripts/bench-transformer-pytorch.py --device cpu
```

出力形式:

- Rust 側: `bench_harness::report::BenchReport::to_json`（`schema_version`・`warmup`／`iters`（20/20 以上）・
  `median_secs`／`q1_secs`／`q3_secs`／`samples_secs`。TASK-8.1 準拠）
- PyTorch 側: `scripts/bench-transformer-pytorch.py` が同形の JSON（`median_secs`／`q1_secs`／`q3_secs`。
  分位点定義は `bench_harness::stats::median_q1_q3` と同じ「ソート後 `idx = round(p*(n-1))` 番目の
  要素を採用」方式で統一）を出力する

## 実測結果（本セッション環境。参考値）

### 計測環境

| 項目 | 値 |
|------|-----|
| CPU | QEMU Virtual CPU version 2.5+（12 vCPU。物理実機ではなく仮想化環境） |
| OS | Linux 7.0.0-28-generic x86_64（Ubuntu ベース） |
| rustc | 1.96.0 (ac68faa20 2026-05-25) |
| PyTorch | 2.13.0+cpu（計測専用 venv。CPU wheel） |
| commit SHA（分岐元 origin/main） | `4a6397cae8f8497e80538d5e39776ae4d69e1467` |
| 実施日 | 2026-08-08 |
| 計測プロトコル | `bench_harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1） |
| 決定的シード | Rust 側: `155_083`（`bench_harness::rng::Xorshift64Star`）／PyTorch 側: `torch.manual_seed(155_083)`（RNG アルゴリズムが異なるため入力ビット列は一致しない。運用上のシード値を揃えたのみ） |

### 実測値（秒。1 回の forward あたり）

| 実装 | バックエンド | median | Q1 | Q3 | 対 PyTorch 比（median） |
|------|------|------|------|------|------|
| Rust（自作コア。GEMM は BLIS 型並列、attention 行列積は naive） | CPU | 0.424358 | 0.419382 | 0.426337 | 1.00（基準） |
| PyTorch 2.13.0+cpu（`nn.TransformerEncoderLayer`） | CPU | 0.026077 | 0.024677 | 0.027008 | 約 16.3 倍高速（Rust は PyTorch の約 6.1% の速度） |

生の `BenchReport` JSON（Rust 側。`samples_secs` は 20 サンプル全件）:

```json
{"schema_version":"1","name":"transformer-block-forward-cpu-blis","backend":"cpu","warmup":20,"iters":20,"median_secs":0.424357532,"q1_secs":0.419382149,"q3_secs":0.426337253,"samples_secs":[0.417835355,0.414654375,0.42448855,0.431536519,0.415673992,0.426337253,0.425320093,0.429413515,0.422293877,0.418985742,0.41489793,0.421280872,0.435151255,0.424053655,0.424878439,0.423393191,0.439532842,0.424357532,0.42766128,0.419382149]}
```

PyTorch 側 JSON:

```json
{"schema_version":"1","name":"transformer-block-forward-pytorch","backend":"cpu","framework":"pytorch","framework_version":"2.13.0+cpu","warmup":20,"iters":20,"median_secs":0.02607715199701488,"q1_secs":0.02467725399765186,"q3_secs":0.027007898985175416,"samples_secs":[0.022971405007410794,0.025220120995072648,0.03186345598078333,0.02658404898829758,0.02607715199701488,0.02525348900235258,0.050119535997509956,0.039101143018342555,0.04466192799736746,0.026961330004269257,0.0221065680088941,0.02333180099958554,0.023094684991519898,0.025641339976573363,0.02393205201951787,0.026193958008661866,0.027007898985175416,0.02496066500316374,0.02467725399765186,0.03574399001081474]}
```

### 解釈上の注意（下限確定〈#158〉への申し送り）

- 対 PyTorch 比（約 6.1%）は **本セッション環境（非実機・仮想 CPU）かつ attention 行列積が naive 経路**
  という 2 重の下振れ要因を含む参考値である。REQ-8 の丸め規則を直接適用する根拠には使わない
  （実装計画 §8 リスク節に明記済み）
- QKV／出力射影・FFN の 2D GEMM 部分（BLIS 型並列カーネル）と attention 部分（naive バッチ matmul・
  softmax）の内訳計測は本イシューのスコープ外。内訳が必要な場合は別途プロファイリング（`perf`・
  `criterion` の関数単位ベンチ等）を検討する（out-of-scope-tracking.md 対象。ユーザー承認を得て
  別 Issue 化する）
- naive バッチ matmul をヘッド単位 2D GEMM（`CpuBackendOps::gemm`）へ置き換えた場合、対 PyTorch 比は
  改善する方向に働くと見込まれる。この置き換え自体は本イシューのスコープ外（実装計画 §3「やらないこと」）

## 実機実測（記入待ち）

`docs/perf/cuda-tensor-core-measurement.md` の先例形式に従い、実機行を記入待ちテンプレートとして固定する。

### CPU / Metal（Apple M4 Max）

```sh
cargo test -p bench-harness --release --test transformer_workload -- --ignored transformer --nocapture
python3 scripts/bench-transformer-pytorch.py --device cpu   # または --device mps
```

| 実装 | バックエンド | median | Q1 | Q3 | 対 PyTorch 比 | commit SHA | 実施日 |
|------|------|------|------|------|------|------|------|
| Rust（自作コア） | CPU（Apple M4 Max） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） |
| PyTorch 2.13.0 | CPU（Apple M4 Max） | （記入） | （記入） | （記入） | 1.00（基準） | （記入） | （記入） |
| Rust（自作コア。Metal 経路実装後） | Metal（Apple M4 Max） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） |
| PyTorch 2.13.0 | MPS（Apple M4 Max） | （記入） | （記入） | （記入） | 1.00（基準） | （記入） | （記入） |

Metal f16 実測は #156 のスコープ（本表は CPU／Metal f32 の記入枠のみ）。

### CUDA（DGX Spark GB10）

```sh
cargo test -p bench-harness --release --test transformer_workload -- --ignored transformer --nocapture   # CUDA 経路実装後
python3 scripts/bench-transformer-pytorch.py --device cuda
```

| 実装 | バックエンド | median | Q1 | Q3 | 対 PyTorch 比 | commit SHA | 実施日 |
|------|------|------|------|------|------|------|------|
| Rust（自作コア。CUDA 経路実装後） | CUDA（DGX Spark GB10） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） |
| PyTorch 2.13.0+cu* | CUDA（DGX Spark GB10） | （記入） | （記入） | （記入） | 1.00（基準） | （記入） | （記入） |

CUDA 最適化後の下限再実測は #157 のスコープ（本表は初期実測の記入枠）。

## 関連イシューとの役割分担（二重管理を避ける）

- **#152／#153（TASK-8.2）**: 丸め規則の実装・段階的下限表の合否判定。本ファイルは実測値の記録のみを担う
- **#158（TASK-8.3d・人間判断）**: 下限値の確定・REQ-8 表への反映。spec（正本）反映は spec リポ側の対応
- **#157**: CUDA 最適化後下限の再実測（上表「CUDA」節を埋める）
- **#156**: Metal f16 実測（上表「CPU / Metal」節の f16 版）

## 未実施・後続作業

- 実機（Apple M4 Max・DGX Spark GB10）での実測実行（上記記入待ちテンプレートを埋める）
- attention バッチ行列積のヘッド単位 2D GEMM 化（naive 経路脱却）による性能改善の検証（本イシューのスコープ外。
  ユーザー承認を得て別 Issue 化する）
- 2D GEMM 部分（BLIS 型並列カーネル）と attention 部分（naive）の内訳計測によるボトルネック特定（同上）

## セキュリティ・カーネル境界検査に関する注記

本イシューはワークロード計測のみでカーネル実装（境界検査を含む）を変更しない。上記の対 PyTorch 比
（naive 経路を含む参考値）を根拠に、カーネル側の手動境界チェック省略を提案しない
（`.claude/rules/coding-rust.md` REQ-8 節・実装計画 §7）。
