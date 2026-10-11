# INT8 量子化の格上げ条件 (c)・(d-2) 測定の申し送り（イシュー #2609）

`docs/int8-quant-grade-up-verification-plan.md` §4.1・§5.1 の案に対応する、実機測定コマンドと実測記録の文書である。(c) の GB10 分と Mac mini（Apple M4）CPU 参考行は #2997 で記入済み（2026-10-11）。Metal 分母は #2998（facade 代替分母の Metal 側は記入済み・PyTorch MPS 基準と対 PyTorch 比は未計測。§2.2）、(d-2) は未実施（#2999 は not planned・spec#83 で範囲外と確定）。**本文書は条件 (c)・(d) の充足を宣言しない**（判断は spec 側）。計測値は記録のみで、閾値・baseline・REQ-8 下限へは反映しない。

## 1. 事前登録規則の案（実測前に固定し、事後に緩めない）

```
R1: 腕ごとに 1 プロセスを 5 回実行し、各 run の median を集めた 5 run の中央値を採用する。Q1/Q3 を併記する。
R2: テストは関数名の完全一致で個別実行する。部分一致フィルタは使わない。
R3: 計測中はビルドを並走させない。
R4: 腕の同定のため .rev-stamp と env_info.txt を回収する。ホスト名は <cuda-node> とする。
R5: #2608 の TOPS（約 244）は合否ゲート・分母に使わない。
R6: 規則は実測後に緩和しない。変更が必要なら人間承認を得る。
```

#2997 では上記を `RULE.txt`（腕・実行順・破棄条件・分位点定義）に具体化し、実測開始前に固定した。集計は `aggregate.py`（`--self-test` 付き）、実行は `orchestrate_cuda.sh`／`orchestrate_mac.sh`（1 呼び出し = 1 プロセス）を使った。`aggregate.py` は新規 run の `<arm>-run<N>.status`（終了コード記録）を必須とし、CUDA 腕は `--require-occupancy` で監視終了記録のない occupancy も拒否する。本記録は status を書かない版のスクリプトで採取したため、再集計には明示オプション `--legacy-no-status` が必要。

## 2. (c) 実機ベースライン

分母の定義は `crates/bench-harness/src/transformer_workload.rs::baseline_spec()` と `docs/perf/transformer-workload-baseline.md` §2〜§5 を正とし、ここでは再定義しない。

### 2.1 CUDA（GB10。接続情報は `docs/real-hardware-verification-env.md` の方式。`<cuda-node>`）

実行コマンド（`transformer-workload-baseline.md` §7 形式。3 テストを個別に実行する）:

```sh
env PATH=...:$PATH CARGO_TARGET_DIR=... \
  cargo test -p bench-harness --release --test transformer_workload \
  -- transformer_block_forward_bench_cuda_prefusion --exact --ignored --nocapture
# 同様に（いずれも --exact を付ける）
#       transformer_block_forward_bench_cuda_fused
#       transformer_block_forward_cuda_fused_parity
```

PyTorch 基準（`torch==2.13.0+cu130`）:

```sh
python3 scripts/bench-transformer-pytorch.py --device cuda
```

実測記録（2026-10-11。`cuda/` に生ログ・JSON・`env_info.txt`・`.rev-stamp`。`aggregate.py` の出力を転記。各 run は 1 プロセス、5 run の median_secs の中央値。腕は run ごとに prefusion → fused → parity → pytorch の順で交互実行）:

| 腕 | run1 | run2 | run3 | run4 | run5 | 中央値 | Q1 | Q3 | 対 PyTorch 比 |
|---|---|---|---|---|---|---|---|---|---|
| prefusion | 141.872 | 142.102 | 115.413 | 145.101 | 118.367 | 141.872 | 118.367 | 142.102 | 219.60 |
| fused | 120.929 | 122.988 | 120.812 | 120.930 | 123.648 | 120.930 | 120.929 | 122.988 | 187.18 |
| pytorch | 0.646 | 0.647 | 0.646 | 0.646 | 0.645 | 0.646 | 0.646 | 0.646 | 1.00（基準） |

単位: ms（各 run の median_secs を 1000 倍。中央値・Q1・Q3 は 5 run の値から算出）。対 PyTorch 比は Rust 腕の中央値 / PyTorch 腕の中央値（1 より大きければ遅い）。

parity（run1..5）: run1=pass, run2=pass, run3=pass, run4=pass, run5=pass

- PyTorch 基準は `torch==2.13.0+cu130`（`scripts/bench-transformer-pytorch.py --device cuda`、warmup/iters 20/20）。
- prefusion は run 間のばらつきが大きい（115〜145 ms。run3・run5 が低い）。他プロセスの混入は検出されておらず（`cuda/gpu-occupancy-*.txt`。run 前の GPU utilization は 0〜2%）、原因は未調査。5 run 中央値（141.872 ms）をそのまま採用し、取り直しは行っていない。
- 対 PyTorch 比が約 190〜220 倍と大きいのは、attention の bmm・LayerNorm・GELU・残差がホストの naive 実装のままで、`CudaBackendOps` が per-op にホスト転送するため（`transformer-workload-baseline.md` §7）。
- 注記: per-op ホスト転送が計測値に含まれる。fused と prefusion の差分には「linear 側の真の融合効果」と「softmax のホスト→GPU 移行効果」が混ざる（`transformer-workload-baseline.md` §7 の非対称性の注記）。
- ノードには常駐サービスの GPU プロセスが 2 件あり、停止していない（run 前の utilization.gpu 0〜2% を確認）。

### 2.2 Metal（Mac mini〈Apple M4〉。案 ii: facade 代替分母）

案 ii（facade `TransformerEncoderLayer` で代替分母）は所有者判断（#2995・2026-10-11）で決定済み。計測は #2998（2026-10-11）。規則は `RULE-metal.txt`（実測前コミットで固定）、生ログ・JSON・`env_info.txt` は `mac-mini-metal/`。案 i（`baseline_spec()` と同一定義の専用実行経路）は所有者判断で量子化の格上げ後に回されており、本節では実装しない。

**本行は `baseline_spec()` 同一定義の分母ではなく代替分母**である。`transformer-workload-baseline.md` §7 の Metal 表（`baseline_spec()` 系列）には転記しない。案 i の実装後に置き換える前提。

定義差分（正は `crates/bench-harness/src/transformer_workload.rs::baseline_spec()` と `crates/facade/tests/transformer_encoder_metal_denominator.rs`）:

| 項目 | `baseline_spec()`（§4.1 の分母） | 本計測（facade 経路） |
|---|---|---|
| d_model / n_heads / d_ff / batch / seq_len / 層数 | 512 / 8 / 2048 / 8 / 128 / 1 | 同じ（`baseline_spec()` から取得） |
| 精度・norm | f32・post-norm・LayerNorm eps 1e-5 | 同じ |
| FFN 活性化 | GELU | **ReLU 固定**（`add_transformer_encoder`。facade は GELU を公開しない） |
| attention | テスト内の手書き合成 | `nn::MultiheadAttention` の `Var` 演算合成 |
| 重み・入力 | `Xorshift64Star` + `SEED` | 入力は同じ生成器・`SEED`。重みは autodiff `nn::init` のシード導出（`SEED` 流用） |
| 実行経路 | backend ops 直呼び | autodiff tape 経由。反復ごとに `tape_for(Device::Metal)` を作り直す（`predict` と同じ運用） |
| ホスト転送 | （CUDA は per-op 転送を含む） | `MetalBackendOps` の各演算はホスト常駐 `Tensor` を返すため per-op 転送を含む |

実行コマンド（`docs/real-hardware-verification-env.md` §7。`orchestrate_mac.sh <arm> <run>` が 1 呼び出し = 1 プロセス。`OUT` は #2998 用に別ディレクトリ）:

```sh
cargo test -p fandhe-ai --release --test compat_sequential_layers_backend_parity \
  -- metal_transformer_encoder_matches_cpu --exact --ignored --nocapture   # 前提ゲート（parity-metal）
BENCH_TRANSFORMER_METAL_REPORT=<json> cargo test -p fandhe-ai --release \
  --test transformer_encoder_metal_denominator \
  -- transformer_encoder_facade_forward_bench_metal_relu --exact --ignored --nocapture
# PyTorch 基準: python3 scripts/bench-transformer-pytorch.py --device mps --output <json>
```

実測（2026-10-11。前提ゲート `metal_transformer_encoder_matches_cpu` は pass。転送元 HEAD `52e86491`。`aggregate.py mac-mini-metal --arms facade-metal` の出力を転記）:

| 腕 | run1 | run2 | run3 | run4 | run5 | 中央値 | Q1 | Q3 | 対 PyTorch 比 |
|---|---|---|---|---|---|---|---|---|---|
| facade-metal（ReLU） | 68.972 | 69.992 | 69.359 | 69.730 | 69.829 | 69.730 | 69.359 | 69.829 | 未計測 |
| pytorch-mps（GELU） | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | - | - | - |

単位: ms（各 run の median_secs を 1000 倍）。run 前 load1 はすべて 2.0 未満（`mac-mini-metal/load-*.txt`）。

- **PyTorch MPS 基準は未計測**。`<metal-node>` の python3 は 3.15.0 で torch の wheel が無く、`pip install torch` が失敗した（`/usr/bin/python3` は 3.9.6）。このため**対 PyTorch 比は算出できず、(c) の分母（比）はまだ埋まっていない**。torch 2.13.0 を導入できる Python（3.10〜3.13 系）を用意したうえで `orchestrate_mac.sh pytorch-mps <run>`（`PYVENV` で venv を指定）を 5 run 実行し、本表の行を埋める（申し送り）。値は推測で埋めない。
- 比を出す際の注記: facade は ReLU 固定、PyTorch 基準は GELU で活性化が非対称（`RULE-metal.txt` M3）。
- 補助計測 `tape_for(Device::Metal)` 単体の median は 1.39〜1.50e-5 秒（5 run の中央値 14 µs。`mac-mini-metal/facade-metal-run*.log` の `arm=facade_tape_for`）。facade 腕の約 69.7 ms に対し tape 生成・デバイス存在検証の固定費は無視できる。
- autodiff 直結 GELU 腕は未実施（`RULE-metal.txt` M2）。
- 本記録は条件 (c) の充足を宣言しない（判断は spec 側）。計測値は閾値・baseline・REQ-8 下限へ反映しない。

### 2.3 CPU（Mac mini〈Apple M4〉。参考行）

(c) の充足要件には数えない。2026-10-11 の Mac mini（Apple M4・10 コア・macOS 27.0.1）の実測（`mac-mini-cpu/`）。M4 Max（開発機）の過去実測とは直接比較しない。

| 腕 | run1 | run2 | run3 | run4 | run5 | 中央値 | Q1 | Q3 | 対 PyTorch 比 |
|---|---|---|---|---|---|---|---|---|---|
| rust-cpu | 254.009 | 253.772 | 275.802 | 253.950 | 253.845 | 253.950 | 253.845 | 254.009 | - |

単位: ms（各 run の median_secs を 1000 倍。中央値・Q1・Q3 は 5 run の値から算出）。対 PyTorch 比は Rust 腕の中央値 / PyTorch 腕の中央値（1 より大きければ遅い）。

- 各 run 前の load1 は 2.0 未満（`mac-mini-cpu/load-*.txt`）。run3 の 275.802 ms は外れ値で、5 run 中央値には影響しない（原因は未調査）。
- **PyTorch CPU 行は未計測（未検証）**: Mac mini の利用可能な Python が 3.9.6（macOS CLT）と 3.15.0（Homebrew）のみで、`torch==2.13.0` の wheel が存在しなかった（3.9 は 2.8.0 まで、3.15 は該当版なし）。版を変えると基準が変わるため、別版では測っていない。Python 3.10〜3.13 の導入は共有ノードの環境変更になるため行っていない。したがって Rust CPU の対 PyTorch 比は存在しない。

## 3. (d-2) 係数確認の実測プロトコル（実施不能。案のみ）

quantize–dequantize 参照実装が必要で、spec の除外事項ゲート（`docs/spec/04-requirements.md:365`）で止まっているため、**現時点では実施できない**。ゲート解除と承認の後に実施する案:

- 入力分布: 一様・正規・外れ値混入・同符号最悪値・amax < 1e-4
- 形状: K ∈ {128, 512, 4096, 16384}。M・N は既存 GEMM ベースライン形状から選ぶ。決定的シード
- 判定: `|C_q − C64| ≤ E(m,n) + u·|C64| + γ_nb`（計画 doc §5.1）。違反 0 件を期待し、1 件でも出れば実装バグとして fail-closed
- 記録: `diff / E` の最大値・分位点、√K 統計、違反件数

| 分布 | K | max(diff/E) | 分位点 | √K 統計 | 違反件数 |
|---|---|---|---|---|---|
| （記入） | （記入） | （記入） | （記入） | （記入） | （記入） |

## 4. 環境記録

- GB10: `cuda/env_info.txt`（ホスト名は `<cuda-node>`）、`cuda/.rev-stamp`
- Mac mini: `mac-mini-cpu/env_info.txt`（ホスト名は `<metal-node>`）、`mac-mini-cpu/.rev-stamp`
- ログ内の絶対パスは `~` に置換済み。実接続情報・ユーザー名は書かない。
- 転送した HEAD は両機とも `53e817e6`。`crates/`・`Cargo.toml`・`Cargo.lock` は無変更で、HEAD との差分は本ディレクトリの計測スクリプトとログのみ。

## 5. 申し送り（人間承認事項）

- 本ログの値は**記録のみ**。閾値・baseline・REQ-8 下限、`docs/perf/transformer-workload-baseline.md` §7 の表への転記は人間承認後に行う（未転記）。
- (c) の充足は宣言しない（判断は spec 側）。Metal 分母は #2998、PyTorch CPU 参考行は Python 3.10〜3.13 を用意できた時点で再計測する。
- prefusion の run 間ばらつきと run3（Mac CPU）の外れ値の原因調査は本 issue の範囲外。
