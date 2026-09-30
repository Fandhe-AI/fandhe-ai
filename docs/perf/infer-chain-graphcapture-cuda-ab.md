# CUDA 推論チェーンの CUDA Graph capture（opt-in）A/B 記録（イシュー #2115）

- 対象: `DeviceParamStore::predict_device_chain`（#1688）の層カーネル列を CUDA Graph へ
  stream capture し、2 回目以降を graph launch 1 回で再生する opt-in 経路。update 区間 capture
  （#1349・`STEP_GRAPH_MODE`）とは別機構
- スコープ: infer の fresh／reuse のみ。train の update 区間 capture（#1349）は対象外
- **現状の verdict: `undetermined`**（GB10 未実測。本 PR は opt-in 実装・A/B 基盤・事前登録規則まで）
- 事前登録規則の正: [`logs/cuda-infer-chain-graphcapture-2115/RULE.txt`](./logs/cuda-infer-chain-graphcapture-2115/RULE.txt)
  （実測前に固定。事後に緩めない。#1349/#1350 の非後退ゲートとは別に定めた規則）
- 実行手順: [`logs/cuda-infer-chain-graphcapture-2115/README.md`](./logs/cuda-infer-chain-graphcapture-2115/README.md)

## 1. 実装の要点

| 項目 | 内容 | 出典 |
|------|------|------|
| opt-in | 環境変数 `FANDHE_AI_CUDA_GRAPH_INFER`（`1`／`true` 完全一致のみ ON・既定 OFF）。単一ゲート `INFER_GRAPH_DEFAULT_ENABLED = false` | `crates/backend-cuda/src/graph.rs` |
| created stream | opt-in ON で created stream を選ぶ（`step_graph_mode` との OR・ordinal ごとに sticky）。初回デバイス初期化前に設定が必要 | `crates/backend-cuda/src/device.rs` |
| BackendOps 拡張 | `linear_chain_forward_captured`（default `Ok(None)`）。不適用は `Ok(None)`、`Err` は実 driver 失敗のみ（`Unsupported` は tape 経路への全体フォールバックを意味するため不使用） | `crates/tensor-core/src/backend_ops.rs` |
| CUDA 実装 | key 構築（世代・base バッファのアドレス・config_key）→ thread-local キャッシュ（ordinal 分割・上限 8・世代 evict）→ ミス時 warmup（NVRTC＋実 launch 1 回）→ capture → replay。同期点は入力 `upload_into` 1・出力 `download` 1（決定 3） | `graph.rs::run_captured_linear_chain`・`ops.rs::linear_chain_forward_captured` |
| 結線 | `predict_device_chain` が層ループの前に 1 回呼ぶ。`None` なら既存経路（無変更） | `crates/autodiff/src/optim/device_store.rs` |
| TLS 破棄耐性 | thread_local にバッファを持つため `DRIVER_CALL_DEPTH` 補助関数を `try_with` 化 | `crates/backend-cuda/src/context_cache.rs` |
| 診断 | `InferGraphStats { captured, replayed, graph_launches }`（学習側 `StepGraphStats` は不変） | `graph.rs` |
| facade 公開面 | 変更なし（環境変数と backend-cuda 内部 `pub fn` のみ） | — |

承認事項に当たらない根拠: `Cargo.toml`／`Cargo.lock` 不変（依存追加なし）・新規 `unsafe` なし
（cudarc の安全 API と既存 launch 関数の再利用）・tolerance／baseline／ガードレール閾値／
`docs/spec/` 不変・`SME_PRODUCTION_ENABLED` 不変。

### 既定反転（結線）を本 PR に含めない理由

既定 const を反転すると `requires_created_stream` を通じて**全 CUDA 利用者の既定ストリームが
created に変わる**（イベント追跡の無効化を含む）。GB10 で ADOPT が出た後の別 PR とし、影響評価を含める。

## 2. 判定セルと帰属（事前登録の要約。正は RULE.txt）

- セル: infer × {fresh, reuse} × batch {64, 1024, 4096}（`bench-fandhe --infer-batch`。allowlist・既定 64 で既存挙動不変）
- fresh は推論チェーンへ到達しない（`model.forward`）。効き得るのは created stream 化のみ。reuse は stream 化と capture の合算
- ADOPT: 専有ゲート成立・全 6 セル中央値 ratio ≤ 1.00・checksum 完全一致・reuse の 1 セル以上で 5/5 run ratio < 1.00・R0〜R2 合格

## 3. 実測記入欄（GB10。未実施）

| 項目 | 結果 |
|------|------|
| 専有ゲート | 未実施 |
| R0（前提ゲート） | 未実施 |
| R1（capture 正当性） | 未実施（本機 RTX 3060 では NVRTC ヘッダ不足で実行不能。`logs/…/env_info.txt`） |
| R2（bit ダンプ diff） | 未実施 |

| batch/mode | before 中央値 | after 中央値 | after/before | 5 run 一貫 < 1.00 | checksum |
|---|---|---|---|---|---|
| 64/fresh | – | – | – | – | – |
| 64/reuse | – | – | – | – | – |
| 1024/fresh | – | – | – | – | – |
| 1024/reuse | – | – | – | – | – |
| 4096/fresh | – | – | – | – | – |
| 4096/reuse | – | – | – | – | – |

**verdict: undetermined**（RULE.txt の undetermined 条件「R0〜R2 の未実行」「GB10 以外での計測」に該当）

## 4. 実装セッションで確認できたこと

- GPU 不要の単体テスト: opt-in の環境変数 allowlist・既定 OFF・override 往復・カウンタ・TLS 破棄後のキャッシュ操作、
  `linear_chain_forward_captured` 既定 `Ok(None)`、`predict_device_chain` の captured 経路（`Some` は層ループ・upload・download を経由しない／`Err` は poison せず伝播／`None` は既存経路へ）、bench-fandhe `--infer-batch` allowlist、`compare_gemm_ab.py --sizes infer-batches`
- capture 経路の実機挙動（bit 同一・replay・weight 更新反映・再 capture・スレッド終了）は GB10 の R1 で検証する（テストは実装済み）

## 5. スコープ外（後続）

- GB10 での R0〜R2 と 5 run A/B の実測
- ADOPT 時の既定反転 PR
- facade 公開 API（`set_cuda_graph_infer_enabled` 相当）の要否判断
- `cuGraphExecUpdate`（`unsafe`・承認要）による batch 可変対応
- Metal への同等機構
