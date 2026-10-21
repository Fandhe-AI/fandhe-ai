# CUDA train `forward_resident` 内訳と fresh `param_readout` の診断（イシュー #2116）

DGX Spark GB10 の CUDA reuse 学習で `forward_resident` が 155.9 µs（step_total の 49.1%）と
最大の区間になった件と、cuda fresh の `param_readout` が cpu fresh の約 2.2 倍ある件を、
区間に分解して支配項を特定するための診断基盤の記録。**診断基盤に加え、GB10 の 5 run 実測と仮説判定（§6）を
#3189 で実施済み（2026-10-21、HEAD `ad1d106e4`）**。削減の修正実装は
含まない（§7 は起票案の列挙のみ）。本番コード・tolerance・baseline・`Cargo.toml`・
`docs/spec/` は変更していない。

## 1. 状態

| 項目 | 状態 |
|---|---|
| 上位層のフェーズ分解テスト（`crates/facade/tests/cuda_train_forward_resident_diag.rs`） | 実装済み。CPU の bit 一致テストは CI 実行 |
| 下位層のフェーズ分解テスト（`crates/backend-cuda/src/train_forward_resident_diag_tests.rs`） | 実装済み（`#[ignore]`・CUDA 実機必須） |
| 事前登録規則・計測・集計スクリプト（`logs/cuda-train-forward-resident-2116/`） | 実装済み（`RULE.txt` は実測前に固定） |
| GB10 の 5 run 実測・支配項の判定 | **実施済み（#3189。§6、生ログ `logs/cuda-train-forward-resident-2116/gb10/`）** |
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

GB10（NVIDIA GB10）、HEAD `ad1d106e4`（#3203 の target デバイス再利用を含む版）、独立 5 プロセス、負荷ゲート 5/5 pass、checksum は run 間・腕間で一致。
生ログと機械出力は `logs/cuda-train-forward-resident-2116/gb10/`（`aggregate.md`。`aggregate.py` exit 0）。以下は RULE.txt の判定文言どおりの転記。

forward_resident（public、5 run 中央値）: batch 16 = 102.1 µs・64 = 131.8 µs・256 = 149.4 µs・1024 = 255.2 µs。

| 項 | batch 16 | batch 64 | batch 256 | batch 1024 |
|---|---|---|---|---|
| H1 層境界の upload/download 呼び出し全体 | 20.4 µs・0.20 寄与あり | 31.4 µs・0.24 寄与あり | 61.0 µs・0.41 支持（支配項） | 154.6 µs・0.61 支持（支配項） |
| H2 未融合 relu の往復 | 14.0 µs・0.14 非支持 | 17.2 µs・0.13 非支持 | 34.6 µs・0.23 寄与あり | 89.5 µs・0.35 寄与あり |
| H3 呼び出しごとのデバイス確保 | 5.5 µs・0.05 非支持 | 5.6 µs・0.04 非支持 | 5.9 µs・0.04 非支持 | 5.9 µs・0.02 非支持 |
| H4 カーネル実行（syncsplit） | 53.2 µs・0.52 支持（支配項） | 69.7 µs・0.53 支持（支配項） | 69.5 µs・0.47 支持（支配項） | 97.4 µs・0.38 寄与あり |
| H5 上位 API 固定費 | 0.6 µs・0.01 非支持 | 0.6 µs・0.00 非支持 | 0.8 µs・0.01 非支持 | 0.2 µs・0.00 非支持 |
| H6 mse の往復 | 13.5 µs・0.13 非支持 | 13.5 µs・0.10 非支持 | 14.8 µs・0.10 非支持 | 19.5 µs・0.08 非支持 |

- 支配項: batch 16・64 は H4、batch 256 は H1 と H4、batch 1024 は H1。各項は計測スケジュールの異なる独立した参考指標で、F 比の和は 1.0 を超えうる（排他的な内訳ではない）。
- fidelity（nosync total / prod total）は l1 = 0.95〜0.98・l2 = 0.89〜0.94 で範囲内。層 l1・l2 の支配区間は d2h_download（batch 1024 の l2 は h2d_upload と d2h_download）。
- what-if（融合 relu）は total / (prod l1 + prod relu) = 0.76（16）・0.79（64）・0.70（256）・0.64（1024）で bits_equal=True（record-only）。
- 非後退 ratio（record_only）= 131.8 µs / 155.9 µs = 0.85。バイナリ（HEAD 対 registry 0.9.0）・ハーネスが異なる。

param_readout（batch 64、5 run 中央値）: R0 mirror 78.5 µs・R1 split 75.1 µs（param_to_vec 34.7・grad_to_vec 36.5）・R2 pretouched 76.7 µs・R3 cpu control 47.4 µs。

- P1 host alloc / page fault: (R0−R2)/R0 = 0.02 → 非支持
- P2 非 contiguous の gather: noncontig_grad_count = 0・grad_contiguous/R0 = 0.00 → 非支持
- P3 CUDA 由来の差: R0/R3 = 1.65 → CUDA 由来の差あり
- P4 sync 待ち: 構造上 0（測定対象外。RULE.txt）

mse（#3203 の判定）: decomposed の mse_loss は 14.2〜23.0 µs（batch 16〜1024）、H6 は F の 0.08〜0.13 で非支持。#3203 の前後 A/B（腕 C）は本件では未計測のため、変更を戻すかどうかの所有者判断の材料は §7.1 では「未計測」のまま。

## 7. 削減施策の起票案（列挙のみ。起票はユーザー承認待ち）

1. CUDA `gemm_resident_rhs_act` の override（`act_relu = true` の融合起動で relu の往復を排除。
   `train-linear-epilogue-fusion.md` §4）
2. `mse_loss` の target アップロードのキャッシュ（毎 step 定数の target の再 H2D を避ける）。**#3203 で実装済み（§7.1。GB10 前後計測は未実施）**
3. 学習 forward のデバイス常駐チェーン化（`Op::LinearResident` が VJP 用にホスト `y` を保持して
   いるため設計変更が必要）
4. `param_readout` の事前タッチ済み再利用バッファ化（fresh 経路のみ。scoreboard の判定行には効かない）

実測結果次第で優先度が変わる（支配項に対応する施策のみ起票する）。

### 7.1 #3203: `CudaMse` の target デバイス再利用（実装済み・GB10 計測は申し送り）

- 実装: `crates/backend-cuda/src/mse.rs`。`CudaMse`（context ごとに 1 つ）が直前の target の
  ホスト内容とデバイスバッファを 1 組保持し、`f32::to_bits` 単位で完全一致したときだけ H2D を
  省く。ミス時は先にキャッシュを無効化し、H2D 成功時のみ登録する。上限 `1 << 20` 要素超と
  Mutex poison は従来経路（一時バッファへ H2D）に縮退する。公開 API・カーネル・`unsafe` は不変で、
  出力は従来とビット同一。
- 効く範囲: forward と backward が同じ target を送る学習 step で 2 回目の H2D を必ず省く。
  step ごとに target が変わる場合も、同長なら確保を省いて上書きするのみで、1 step あたり
  1 回の H2D は残る。`pred` の H2D・readback 同期は対象外（#3249・#3252）。
- 検証: 単体テスト（`mse::tests`）・`tests/mse_parity.rs` のスモークは CI で実行。
  ビット一致・別 context・複数 ordinal は `#[ignore]` の実機テストで、**未検証（GB10 申し送り）**。
- 申し送り（受入基準 4）: GB10 で変更前 commit（`1a7eb9a2`）と変更後を、`logs/cuda-train-forward-resident-2116/README.md`
  の手順で各 5 run・中央値・機体名・版つきで測り、下表へ記入する。

| 区間 | 変更前（中央値） | 変更後（中央値） | 機体・版 |
|------|------------------|------------------|----------|
| `mse` | 未計測 | 未計測 | |
| `prod` | 未計測 | 未計測 | |
| `total`（train CUDA reuse） | 未計測 | 未計測 | |

  mse 区間への寄与がノイズ以下だった場合（#3189 の判定）は、その結果をここへ記録したうえで
  本変更を戻すかどうかを所有者が判断する。

## 8. 限界

- GB10 実測は #3189 で 1 系列（5 run）のみ。帰属は RULE.txt の判定文言に限り、原因の断定はしない。腕 C（#3203 前後）は未計測
- 診断は HEAD のソースをテストとして実行するため、registry 版 `fandhe-ai =0.9.0` の bench-fandhe
  とはバイナリ・ハーネスが異なる。非後退 ratio は参考値
- 下位層は本番経路の写しで、本番の `with_driver_call`（poison・世代検査）を通らない。fidelity
  （nosync 合計 / prod）で乖離を記録する
- `syncsplit` は同期を挿入するため schedule が本番と異なる
- 上位層と下位層は別プロセス・別 tape の計測で、F と下位層の区間は同一反復のものではない
  （5 run 中央値同士の比）
- 完全な専有ではない（共有環境。gb10 ゲートは load1 と GPU utilization のみ）
