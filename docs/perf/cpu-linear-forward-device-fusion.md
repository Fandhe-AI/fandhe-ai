# CPU `linear_forward_device` の融合 epilogue 経路化（イシュー #3196）

## 1. 目的

reuse 推論（`Sequential::predict_resident` → `predict_device_chain` → `linear_forward_device`）の
CPU 実装を、非融合 3 段（`gemm_blis_parallel` → bias 行方向加算 → `max(x, 0.0)`）から
融合カーネル `gemm_blis_bias_act_parallel` 経由へ切り替え、fresh の `predict`・
`gemm_resident_rhs_act` と実装経路を一本化する。あわせて `cpu-predict-resident-fixedcost.md` §3 が
指摘した doc の食い違い（「融合は加算順序が変わりうる」）を是正する。

## 2. 変更内容

- `crates/backend-cpu/src/ops.rs` `linear_forward_device`: 融合カーネル呼び出しへ置換。
  未対応 `Activation` は GEMM 前に `BackendError::Unsupported` で拒否（チェーン全体の
  フォールバック判定 `docs/inference-chain-single-sync-design.md` 決定 7 を保つ）。
  `m * n` の `checked_mul` と bias `[n]` 厳密一致検査は従前どおりカーネル前に実施。
- tolerance・FMA 契約・判定規則・事前登録プロトコル（`logs/cpu-predict-resident-fixedcost-2105/RULE.txt`）は不変。

## 3. bit 一致の根拠とテスト結果

epilogue は GEMM 本体が K 全域を累積し終えた後に要素独立で 1 回だけ適用され、GEMM 本体の
累積順序は両入口で同一。`crates/backend-cpu/tests/linear_forward_device_parity.rs` に
`f32::to_bits()` 完全一致のテストを追加し、**切替前の HEAD で pass** を確認してから切り替えた。

- 融合 vs 非融合: MLP 形状 (64,784,256)・(64,256,10)、m=1、m=0、n=0、k=1、MR/NR 端数、KC 境界
  (255/256/257)、n=600、bias 有無 × None/Relu、rayon スレッド 1/2/既定
- NaN／±inf／bias=-0.0 を含む入力
- `linear_forward_device` vs 非融合合成、vs `gemm_resident_rhs_act`
- 実行環境: Linux x86_64（CI 相当）。既存の backend-cpu／autodiff／facade のテストも全 pass。

## 4. 性能見込み（正直な記述）

本番の GEMM は 2D 動的経路で、融合版でも epilogue は GEMM 完了後に C 全体へ 1 回逐次適用される。
旧実装の手書き 2 ループと C の走査回数は同じで、カーネル単体の差はほぼ無いと見込む。
fresh で見えた短縮は rayon 並列の `elementwise::add`／`relu` 3 リージョンとの比較で、
`linear_forward_device` は元々それを使っていない。本変更の主な成果は経路の一本化と doc 契約の是正であり、
「改善なし」も正当な計測結果として扱う。

## 5. 前後計測プロトコル（実機へ申し送り）

- before = merge-base SHA、after = PR head SHA（`.rev-stamp` 等で同定）
- `scripts/bench/framework-compare` で
  `cargo build --release -p bench-fandhe --config 'patch.crates-io.fandhe-ai.path="<tree>/crates/facade"'` →
  `./target/release/bench-fandhe --task infer --device cpu --mode reuse` を独立 5 プロセスで実行
  （batch 64・784→256→ReLU→10、warmup／measure 既定 20/20）。各 run の median の中央値と min–max、
  checksum が before／after で完全一致することを記録
- 記録項目: 機体、版（SHA）、rustc、`RAYON_NUM_THREADS`。GB10 は `load1 < 1.0` の専有ゲート、計測中は cargo を並走させない
- path patch は `Cargo.lock` を書き換えるため計測後に `bench_fandhe_lock_restore.sh` で戻し、lock 差分はコミットしない

## 6. 実測表

| 機体 | 版 | before 中央値 (min–max) | after 中央値 (min–max) | checksum 一致 |
|---|---|---|---|---|
| Mac mini | 未実測 | 未実測 | 未実測 | 未実測 |
| DGX Spark GB10（#3188 再オープン分。infer reuse total） | pre `17c7767b5`／head `4f6716f63` | 189.4 µs（170.3-209.9） | 192.8 µs（172.9-204.2） | 一致 |

- GB10 の行は bench-fandhe（`--task infer --device cpu --size 64 --mode reuse`）の 5 round 中央値。pre／head は #3196 単独の前後ではなく、#3193・#3194 を含む区間である（infer 経路に効く本番変更は #3196 のみ）。reuse total は 189.4 → 192.8 µs で変化がなく、min-max は重なる（head/pre = 1.018）。fresh は 176.4 → 175.8 µs（0.997）。出典: `logs/cpu-train-fixedcost-gb10-3188/reopen/aggregate_bench_prepost.md`
- Mac mini は未実測のまま

参考: 診断ベース（別プロトコル）の GB10 前後は `cpu-predict-resident-fixedcost.md` §6（#3188）、bench の詳細は同 §6.1。

## 7. 限界・申し送り

- GB10 の bench-fandhe 実測は #3188 再オープン分で実施した（§6）。Mac mini は未実施。#3196 の効果は GB10 の infer reuse total では確認できなかった（変化なし）
- スコープ外: `predict_device_chain` の入力 upload コピー削減・出力 arena 確保、GEMM 本体の改善（#3197・#3233）、
  epilogue のパネル単位適用（カーネル契約の変更になるため別 Issue）
