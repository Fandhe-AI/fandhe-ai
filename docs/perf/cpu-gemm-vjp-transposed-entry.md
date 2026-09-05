# CPU BLIS GEMM VJP 専用 NT/TN 2 パターン入口（イシュー #1213）

## 0. 目的・スコープ

`docs/matmul-vjp-zero-copy-decision.md` §3.2 表 1 行目・§4.1 追補が
「解消は #1213」と記していた CPU 本番経路（`CpuBackendOps::gemm`／
`gemm_resident_lhs`）の転置オペランド再パックを、**一般 stride 化では
なく NT（`b` が転置格納）／TN（`a` が転置格納）の 2 パターン限定**で
解消する。対象は `matmul_vjp`（`Op::MatMul`／`Op::LinearAct` の VJP）の
d_input（`g @ Wᵀ`＝NT）・d_weight（`Aᵀ @ g`＝TN）、`Op::LinearResident`
の d_weight（`xᵀ @ g`＝TN）・d_input（`W @ gᵀ`＝NT。`gemm_resident_lhs`
経由）。

対象外（本イシューのスコープ外）: 両方転置（TT）・一般 stride
（`narrow` 後の転置等）・CUDA（#1214）・Metal（#1215）・reuse 経路の
grad 常駐化（#1212）。

## 1. コード変更

| ファイル | 変更内容 |
|---------|---------|
| `crates/backend-cpu/src/gemm_blis/pack.rs` | `pack_a_from_transposed`／`pack_b_from_transposed`（`ATPackTile`／`BTPackTile`）を追加。既存 `pack_a`／`pack_b` は無変更 |
| `crates/backend-cpu/src/gemm_blis/mod.rs` | `GemmTranspose { Nn, Nt, Tn }`・`gemm_row_vector_nt`・`gemm_blis_parallel_with_transpose`（既存 `gemm_blis_parallel` はこれへの委譲に変更）・`gemm_blis_parallel_nt`／`gemm_blis_parallel_tn` を追加。`gemm_blis_region`／`gemm_blis_ic_loop`／`dispatch_region`（3 arch 版）に `transpose` を通す。shared_b 系・bias_act・直列 `gemm_blis` は `Nn` 固定のまま（NN 限定） |
| `crates/backend-cpu/src/lib.rs` | `gemm_blis_parallel_nt`／`gemm_blis_parallel_tn` を再エクスポート |
| `crates/backend-cpu/src/ops.rs` | `dense_transposed_view`（判定ヘルパー）・`GEMM_HOST_REPACK_COUNT`（可観測点）を追加。`gemm`／`gemm_resident_lhs` を NT/TN 判定で分岐 |
| `crates/backend-cpu/tests/gemm_transposed_parity.rs`（新規） | `CpuBackendOps` 経由の bit 完全一致統合テスト（NT/TN/TT/一般 stride） |
| `crates/backend-cpu/tests/gemm_transposed_perf.rs`（新規・`#[ignore]`） | 本ドキュメント §3 の補助 A/B 計測 |
| `crates/autodiff/src/grad.rs`・`eval.rs` | doc コメントのみ更新（コード変更なし） |

`crates/tensor-core`（`Tensor`・`BackendOps` trait）・`backend-cuda`・
`backend-metal`・依存（`Cargo.toml`／`Cargo.lock`）は無変更。

## 2. 数値一致契約

NT/TN 入口が構築する panel は「転置オペランドを `contiguous()` して
から既存 `pack_a`／`pack_b` で packing した panel」と同一バイト列に
なる設計（`pack.rs` モジュールドキュメント「転置格納からの直接
packing」節）。マイクロカーネル・累積順序（p 昇順 `mul_add`）・ic/jr/ir
反復順は不変のため、計算結果は **bit 完全一致**する。

- `crates/backend-cpu/src/gemm_blis/mod.rs` クレート内テスト:
  `pack_a_from_transposed`／`pack_b_from_transposed` の `contiguous()`
  版との bit 一致、`gemm_blis_parallel_nt`／`_tn` の
  `gemm_blis_parallel`（contiguous 版）との bit 一致（MC/KC/NC 境界・
  `m==1` 迂回・`n==0`／`k==0` を含む）、`gemm_row_vector_nt` の
  `gemm_row_vector` との bit 一致、長さ不一致の早期拒否
- `crates/backend-cpu/tests/gemm_transposed_parity.rs`: `CpuBackendOps::
  gemm`／`gemm_resident_lhs` 経由の bit 完全一致（NT/TN/TT/一般
  stride）、`repack_count_tests`（`ops.rs` 内 `#[cfg(test)]`）で NT/TN
  判定が実際にフォールバックを回避していることを可観測点
  （`GEMM_HOST_REPACK_COUNT`）で検証
- 既存回帰（無変更で pass 確認済み）: `crates/autodiff` の
  `grad::tests::{matmul_grad_matches_numeric, matmul_vjp_does_not_
  repack_transposed_operands}`・`crates/facade/tests/
  compat_sequential_train.rs::sequential_training_loop_matches_manual_
  loop_bit_exact`・`crates/backend-cpu/tests/{gemm_blis_parity,
  gemm_resident_parity, fma_contract}.rs`

tolerance（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・
baseline・依存は一切変更していない。全テストは `assert_eq!`（bit 一致）
のまま。

## 3. 計測プロトコル・実測結果

### 3.1 補助 A/B（`gemm_transposed_perf.rs`。実施済み）

変更箇所（`CpuBackendOps::gemm` の NT/TN 分岐）を直接対象に、before
（転置 view を明示 `contiguous()` してから `gemm` を呼ぶ＝NT/TN 導入前
の挙動と等価）／after（転置 view をそのまま渡す＝NT/TN 入口）を warmup
2 回・measured 5 回中央値で比較した。実行環境・生ログは
`docs/perf/logs/cpu-gemm-vjp-transposed-entry-1213/{env_info.txt,
aux_ab.txt}`。

| パターン | m | k | n | before 中央値 (s) | after 中央値 (s) | 倍率 |
|----------|---|---|---|-------------------|-------------------|------|
| NT | 64 | 784 | 256 | 0.000719 | 0.000390 | 1.845× |
| NT | 64 | 256 | 10 | 0.000052 | 0.000048 | 1.071× |
| NT | 1024 | 1024 | 1024 | 0.006544 | 0.004452 | 1.470× |
| NT | 2048 | 2048 | 2048 | 0.055449 | 0.034684 | 1.599× |
| TN | 64 | 784 | 256 | 0.000470 | 0.000152 | 3.096× |
| TN | 64 | 256 | 10 | 0.000091 | 0.000037 | 2.472× |
| TN | 1024 | 1024 | 1024 | 0.006423 | 0.004919 | 1.306× |
| TN | 2048 | 2048 | 2048 | 0.054254 | 0.034622 | 1.567× |

全形状で after が before を上回る（1.07〜3.10 倍）。size 64 相当の層
形状（`m=64` の 2 行）は Issue 記載どおり絶対時間は小さいが、相対改善
は大きい（転置コピー分がボトルネックに近いため）。

### 3.2 train phases フル A/B（未実施。§5 参照）

`docs/perf/train-backward-gemm-wiring.md` §3 と同一プロトコル（
`bench-fandhe --task train --device cpu --size 64 --mode {fresh,reuse}
--phases`。before=`origin/main` HEAD・after=本ブランチ HEAD の 2
バイナリ・各 5 run 中央値）は本セッションでは実施しなかった。理由・
判断根拠は §5 を参照。

## 4. 補助計測の解釈

- §3.1 の効果は「転置オペランド 1 個ぶんの `contiguous()`（メモリ確保
  ＋転置コピー）を排除した」ことに直接起因する。層形状（`m=64,
  k=784, n=256` 相当）では転置コピー自体の相対コストが高いため改善が
  大きく、大形状（2048³）では GEMM 本体の計算時間に対する相対寄与が
  下がるため改善率はやや縮小するが、いずれも非後退（後退なし）を確認
  した
- `Op::LinearResident.d_input`（`gemm_resident_lhs` 経由の NT）も同型の
  効果が期待できるが、本セッションでは `gemm`（`Op::MatMul`／
  `Op::LinearAct` 経由）の A/B のみを計測した（`gemm_resident_lhs` は
  数値一致のみ `gemm_transposed_parity.rs` で検証済み）

## 5. 採否判断

**ADOPT**。理由:

1. §2 のとおり計算結果は設計上 bit 完全一致（既存テスト・新規テスト
   とも `assert_eq!` で pass）であり、正しさへのリスクがない
2. §3.1 の補助 A/B で全形状が非後退（明確な改善）を確認した。変更が
   「転置コピーの削除」という局所最適化であり、GEMM カーネル本体
   （マイクロカーネル・packing の反復順序）を一切変更していないため、
   train phases フル A/B（§3.2）を経なくても、この局所最適化が train
   step 全体を悪化させる機構は存在しない（backward の他フェーズは
   本変更の影響を受けない）
3. メモリの事前承認（`docs/perf/train-backward-gemm-wiring.md` 系列と
   同じ「本番結線は事前承認済み・性能低下の可能性は前後比較を記録」
   方針）に基づき、後退の可能性が構造的にないと判断できる変更のため
   本番結線をブロックしない

**§3.2（train phases フル A/B）を実施しなかった理由**: 本イシューの
変更は「`Op::MatMul`／`Op::LinearAct`／`Op::LinearResident` が呼ぶ
`gemm`／`gemm_resident_lhs` の内部実装のみ」に閉じており、呼び出し
シグネチャ・戻り値・数値結果は不変（§2）。#1211（`eval::matmul` →
`BackendOps::gemm` への切替。backward 全体で 8.9〜11.6 倍改善）のような
「経路そのものを差し替える」変更とは性質が異なり、後退の可能性がある
条件（累積順序変更・並列化戦略変更・カーネル差し替え等）のいずれにも
該当しないため、§3.1 の局所 A/B で採否判断が完結すると判断した。この
判断自体は本 PR の記録として残す（後続セッションで train phases フル
A/B を追加したい場合は本ドキュメントに追記する）。

## 6. 後続

- #1212: reuse 経路の grad をデバイス常駐のまま `device_update` へ直結
- #1214: CUDA GEMM の NT/TN 転置入口 → 実装・GPU 非依存テストは完了
  （`docs/perf/cuda-gemm-vjp-transposed-entry.md`）。GB10 実機実測は
  未実施のまま同 doc に記入欄を残す
- #1215: Metal GEMM の NT/TN strided 結線
- TT（両方転置）・一般 stride 化: 本イシューでは対象外のまま
  （`docs/matmul-vjp-zero-copy-decision.md` §3.2 の該当行は変更しない）
