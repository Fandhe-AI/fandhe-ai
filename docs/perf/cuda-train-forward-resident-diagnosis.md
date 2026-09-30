# CUDA train `forward_resident` 内訳と fresh `param_readout` の診断（イシュー #2116）

DGX Spark GB10 の CUDA reuse 学習で `forward_resident` が 155.9 µs（step_total の 49.1%）と
最大の区間になった件と、cuda fresh の `param_readout` が cpu fresh の約 2.2 倍ある件を、
区間に分解して支配項を特定するための診断基盤の記録。**本書の時点では診断基盤のみ実装済みで、
GB10 の 5 run 実測と仮説判定（§6）は未実施（実機セッションへの申し送り）**。削減の修正実装は
含まない（§7 は起票案の列挙のみ）。本番コード・tolerance・baseline・`Cargo.toml`・
`docs/spec/` は変更していない。

## 1. 状態

| 項目 | 状態 |
|---|---|
| 上位層のフェーズ分解テスト（`crates/facade/tests/cuda_train_forward_resident_diag.rs`） | 実装済み。CPU の bit 一致テストは CI 実行 |
| 下位層のフェーズ分解テスト（`crates/backend-cuda/src/train_forward_resident_diag_tests.rs`） | 実装済み（`#[ignore]`・CUDA 実機必須） |
| 事前登録規則・計測・集計スクリプト（`logs/cuda-train-forward-resident-2116/`） | 実装済み（`RULE.txt` は実測前に固定） |
| GB10 の 5 run 実測・支配項の判定 | **未実施（実機セッションへ申し送り）** |
| 修正実装 | スコープ外（§7 に起票案を列挙） |

作業ホストには RTX 3060 があるが `libnvrtc` が無く（CUDA toolkit 非搭載）、CUDA の
`#[ignore]` テストは実行できなかった。CPU 経路の出力形式・集計スクリプトの動作確認のみ行った。

## 2. 現象

- `train-step-phase-breakdown.md` §17.6.2（#1980・GB10・registry `fandhe-ai =0.9.0`）: cuda reuse の
  `forward_resident` 155.9 µs は backward 149.1 µs を上回る。§17.6.4 は内訳（encode／同期回数）が
  未取得と記録している
- 同表の cuda fresh の更新区間（`param_readout` 87.0 + `host_sgd` 54.4 + `apply_params` 0.3 =
  141.7 µs・27.4%）。`param_readout` は同一ホストの cpu fresh 40.2 µs の約 2.2 倍
- `loss-attribution-matrix.md` の #2116 行: fresh の `param_readout` は scoreboard の判定行
  （reuse）に効かないため、**主対象は `forward_resident`**

## 3. コード事実とサブフェーズ名の対応

### 3.1 `forward_resident` 区間の実体

bench-fandhe `measure_train_reuse_phases` の区間は `Sequential::forward_resident` と
`mse_loss` の合計。呼び出し列は次のとおり。

```
Sequential::forward_resident                (crates/facade/src/compat/sequential.rs)
 ├ ガード 2 種
 ├ DeviceParamStore::register_resident_params  (resident leaf の tape 登録。H2D なし)
 └ forward_from_flat_leaves
     └ 層ごとに DeviceParamStore::linear_forward_with_activation
         └ BackendOps::gemm_resident_rhs_act   (CUDA は override なし → trait 既定の合成)
             ├ CudaBackendOps::gemm_resident_rhs  (ops.rs)
             └ CudaBackendOps::relu               (ホスト往復)
mse_loss → CudaBackendOps::mse_loss            (H2D → カーネル → D2H。target も毎 step 再アップロード)
```

`gemm_resident_rhs` は呼び出しごとに `CudaMemory::new` → `upload(a)`（H2D）→ `alloc_zeroed(c)` →
`cached_gemm` → `launch_tiled_bias_act_f32_resident`（`act_relu = false`）→ `download(c)`
（同期 + D2H）を行う。計測モデル 784→256（ReLU）→10・batch 64 では、1 step の forward に
「L1 GEMM 往復・relu 往復・L2 GEMM 往復・mse 往復」の 4 回以上のホスト往復がある。
`loss_readout` が 0.0 µs なのは forward 区間内で既に同期が済むためと読める
（`train-linear-epilogue-fusion.md` §4 が CUDA の epilogue 未融合を記録）。

`BackendOps::linear_forward_device`（a・w・bias・戻り値がすべて常駐）は推論チェーン
（`predict_device_chain`）の経路で、学習 forward からは呼ばれない。

### 3.2 イシューが名指しするサブフェーズ名の対応

| イシュー上の名称 | 実コード上の区間 |
|---|---|
| param_apply | `register_resident_params`（resident パラメータの leaf 登録）。パラメータ更新そのもの（`device_update`）は forward 区間の外 |
| linear_forward_device（学習経路） | `linear_forward_with_activation` → CUDA の `gemm_resident_rhs_act` 既定合成（`gemm_resident_rhs` + ホスト `relu`） |
| linear_forward_device（what-if） | 推論チェーン用の融合起動相当（`act_relu = true`）。下位層で record-only の参照 arm として計測 |
| alloc | 層ごとの `CudaMemory::new` + `upload`（確保 + H2D）+ `alloc_zeroed(c)`（下位層）。上位層では tape ノード push・Vec 確保を残差として計上 |

### 3.3 fresh の `param_readout`

各 param／grad に `contiguous().as_slice().to_vec()` を行い、3 本の `Vec::with_capacity` を
確保する。`Tensor` はホスト storage だけを持つため（`crates/tensor-core/src/tensor.rs`）、
**この区間に D2H・デバイス同期は構造上ない**（grad は backward 内で既にホストへ実体化済み）。
contiguous なら `contiguous()` は Arc clone、非 contiguous なら要素単位の gather が走る。
1 step で約 203,530 要素（約 814 KB）× 2 をコピーし、w1（約 802 KB）は glibc の既定 mmap
閾値を超えるため毎 step mmap／munmap とページフォールトが起きうる（仮説）。

### 3.4 既存記録との重複回避

#1182（`cuda-gemm-reuse-phase-breakdown.md`）は正方行列 `Var::matmul` の H2D／カーネル／D2H を
分解済み。本件は resident-RHS カーネルの MLP 形状・relu／mse の往復・学習 forward の上位 API
固定費を対象とし、正方 GEMM 分解は再実行しない。#1336／#1437（readout 方式）・#1149（per-call
alloc）・#1585（pinned H2D）・#1353（managed）の A/B も再実行しない。

## 4. 事前登録仮説（`RULE.txt` が正。閾値は事後に変えない）

F = `public/forward_resident` に対する比が 0.40 以上で支配項、0.20 以上 0.40 未満で寄与あり。

各項（H1〜H6）は計測スケジュールの異なる独立した参考指標（H1・H4 は syncsplit、H3 は nosync、H2・H6 は prod 単独、H5 は上位 API）であり、排他的な内訳ではない。F 比の和は 1.0 を超えうるため、合算して F の内訳とは読まない。

| 仮説 | 内容 | 定義 |
|---|---|---|
| H1 | 層境界の upload/download 呼び出し全体（転送のみの分離計測ではなく、呼び出し内のホスト側確保等を含む） | syncsplit の (h2d_upload + d2h_download) を l1・l2 で合算。nosync の d2h_download はカーネル完了待ちを含み H4 と重複するため使わない（syncsplit は kernel_wait を別区間に分離済みで重ならない） |
| H2 | 未融合 relu の往復 | prod/relu total |
| H3 | 呼び出しごとのデバイス確保 | nosync の (mem_new + alloc_c) を l1・l2 で合算 |
| H4 | カーネル実行 | syncsplit の kernel_wait を l1・l2 で合算 |
| H5 | 上位 API の固定費 | decomposed/register + paired/residual（反復単位の残差） |
| H6 | mse の往復（target の H2D 含む） | prod/mse total |

param_readout: P1 host alloc／ページフォールト（(R0−R2)/R0 ≥ 0.50）・P2 非 contiguous の gather・
P3 CUDA 由来の差（R0/R3 > 1.50）・P4 sync 待ち（構造上 0。測定しない）。

## 5. 計測プロトコル

| 層 | テスト | 役割 |
|---|---|---|
| 上位 | `crates/facade/tests/cuda_train_forward_resident_diag.rs` | `public`（bench と同じ呼び出し列）と `decomposed`（register／l1_linear_relu／l2_linear／mse_loss）の 2 arm を同一反復内で実行し、ラウンドごとに順序を反転。`paired/residual` は反復ごとに差を取ってから中央値。fresh の readout は R0（逐語写し）・R1（tensor ごとの分解）・R2（事前タッチ済み再利用 Vec）・R3（CPU 対照） |
| 下位 | `crates/backend-cuda/src/train_forward_resident_diag_tests.rs` | l1・l2・relu・mse を batch 16／64／256／1024 で計測。`prod`（本番）・`nosync`（本番と同じスケジュールの区間分解。fidelity 評価はこれのみ）・`syncsplit`（起動直後に同期を挿入して `kernel_wait` を分離。本番より長くなりうるため H4 の値のみに使う）・`whatif`（融合 relu の参照値・record-only） |

- CI 実行の bit 一致テスト（CPU）: `public` と `decomposed` の pred・loss・最終 params、readout の
  R0／R1／R2 の最終 params が bit 一致することを hard assert する。CPU は
  `gemm_resident_rhs_act` を融合 override しているため、CUDA 固有の既定合成の bit 一致は CI では検証
  できない。`#[ignore]` の CUDA 版（上位）と下位層の hard assert（分解出力 = 本番 `gemm_resident_rhs`
  出力）がそれを担う
- 実機計測は `orchestrate.sh <gb10|rtx3060>` が独立 5 プロセスずつを実行し、`aggregate.py` が
  checksum 検査（fail-closed）・帰属判定・非後退 ratio（record_only）を出力する
- 手順は `logs/cuda-train-forward-resident-2116/README.md`

## 6. 結果

**未実施（GB10 の 5 run 実測は実機セッションへ申し送り）。** 実測後、`aggregate.py` の出力を
ここへ記入する。RTX 3060 等の系列を取る場合は「非 GB10・参考・record_only」と明記し、GB10 の
受け入れ条件を満たしたとは扱わない。

## 7. 削減施策の起票案（列挙のみ。起票はユーザー承認待ち）

1. CUDA `gemm_resident_rhs_act` の override（`act_relu = true` の融合起動で relu の往復を排除。
   `train-linear-epilogue-fusion.md` §4）
2. `mse_loss` の target アップロードのキャッシュ（毎 step 定数の target の再 H2D を避ける）
3. 学習 forward のデバイス常駐チェーン化（`Op::LinearResident` が VJP 用にホスト `y` を保持して
   いるため設計変更が必要）
4. `param_readout` の事前タッチ済み再利用バッファ化（fresh 経路のみ。scoreboard の判定行には効かない）

実測結果次第で優先度が変わる（支配項に対応する施策のみ起票する）。

## 8. 限界

- GB10 未実測。本書の時点で帰属の主張はない
- 診断は HEAD のソースをテストとして実行するため、registry 版 `fandhe-ai =0.9.0` の bench-fandhe
  とはバイナリ・ハーネスが異なる。非後退 ratio は参考値
- 下位層は本番経路の写しで、本番の `with_driver_call`（poison・世代検査）を通らない。fidelity
  （nosync 合計 / prod）で乖離を記録する
- `syncsplit` は同期を挿入するため schedule が本番と異なる
- 上位層と下位層は別プロセス・別 tape の計測で、F と下位層の区間は同一反復のものではない
  （5 run 中央値同士の比）
- 完全な専有ではない（共有環境。gb10 ゲートは load1 と GPU utilization のみ）
