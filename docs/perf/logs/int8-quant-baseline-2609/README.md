# INT8 量子化の格上げ条件 (c)・(d-2) 測定の申し送り（イシュー #2609）

`docs/int8-quant-grade-up-verification-plan.md` §4.1・§5.1 の案に対応する、**実機測定コマンドと記入欄のみ**の文書である。値はすべて未記入で、**本文書は条件 (c)・(d) の充足を宣言しない**（判断は spec 側）。実測は承認後の別 issue で行う。

## 1. 事前登録規則の案（実測前に固定し、事後に緩めない）

```
R1: 腕ごとに 1 プロセスを 5 回実行し、各 run の median を集めた 5 run の中央値を採用する。Q1/Q3 を併記する。
R2: テストは関数名の完全一致で個別実行する。部分一致フィルタは使わない。
R3: 計測中はビルドを並走させない。
R4: 腕の同定のため .rev-stamp と env_info.txt を回収する。ホスト名は <cuda-node> とする。
R5: #2608 の TOPS（約 244）は合否ゲート・分母に使わない。
R6: 規則は実測後に緩和しない。変更が必要なら人間承認を得る。
```

## 2. (c) 実機ベースライン

分母の定義は `crates/bench-harness/src/transformer_workload.rs::baseline_spec()` と `docs/perf/transformer-workload-baseline.md` §2〜§5 を正とし、ここでは再定義しない。

### 2.1 CUDA（GB10。接続情報は `docs/real-hardware-verification-env.md` の方式。`<cuda-node>`）

実行コマンド（`transformer-workload-baseline.md` §7 形式。3 テストを個別に実行する）:

```sh
env PATH=...:$PATH CARGO_TARGET_DIR=... \
  cargo test -p bench-harness --release --test transformer_workload \
  -- transformer_block_forward_bench_cuda_prefusion --ignored --nocapture
# 同様に transformer_block_forward_bench_cuda_fused
#       transformer_block_forward_cuda_fused_parity
```

PyTorch 基準（`torch==2.13.0+cu130`）:

```sh
python3 scripts/bench-transformer-pytorch.py
```

記入欄（単位は `transformer-workload-baseline.md` §3 に従う）:

| 腕 | run1 | run2 | run3 | run4 | run5 | 中央値 | Q1 | Q3 | 対 PyTorch 比 |
|---|---|---|---|---|---|---|---|---|---|
| Rust prefusion | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） |
| Rust fused | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） |
| PyTorch | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） | （記入） | 1.00（基準） |

注記: per-op ホスト転送が計測値に含まれる。

### 2.2 Metal（M4 Max）

複合ワークロードの実行経路（`transformer_workload_metal.rs` 相当）が未実装のため、**現時点では測定不能**。案 i（専用経路）か案 ii（facade `TransformerEncoderLayer` 代替）かは承認待ち（計画 doc §8 項 11）。記入欄は `transformer-workload-baseline.md` §7 の Metal 表を使う。

### 2.3 CPU（M4 Max。参考行）

`docs/perf/transformer-workload-measurement.md` の記入待ちテンプレートを使う。(c) の充足要件には数えない。

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

`env_info.txt` は計測時に作成する（ホスト名は `<cuda-node>` に置換し、実接続情報・ユーザー名・絶対パスは書かない）。
