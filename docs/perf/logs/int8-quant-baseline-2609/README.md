# INT8 量子化の格上げ条件 (c)・(d-2) 測定の申し送り（イシュー #2609）

`docs/int8-quant-grade-up-verification-plan.md` §4.1・§5.1 の案に対応する、実機測定コマンドと実測記録の文書である。(c) の GB10 分と Mac mini（Apple M4）CPU 参考行は #2997 で記入済み（2026-10-11）。Metal 分母は #2998、(d-2) は未実施（#2999 は not planned・spec#83 で範囲外と確定）。**本文書は条件 (c)・(d) の充足を宣言しない**（判断は spec 側）。計測値は記録のみで、閾値・baseline・REQ-8 下限へは反映しない。

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

### 2.2 Metal（Mac mini〈Apple M4〉）

案 ii（facade `TransformerEncoderLayer` で代替分母）に決定済み（#2995・2026-10-11）。計測は #2998 の担当で、本 issue では測らない。以下は #2609 時点の記述。

複合ワークロードの実行経路（`transformer_workload_metal.rs` 相当）が未実装のため、**現時点では測定不能**。案 i（専用経路）か案 ii（facade `TransformerEncoderLayer` 代替）かは承認待ち（計画 doc §8 項 11）。記入欄は `transformer-workload-baseline.md` §7 の Metal 表を使う。

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
