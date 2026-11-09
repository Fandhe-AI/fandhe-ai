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

## 3. 実測（GB10。イシュー #3190・2026-10-20）

収録: `logs/cuda-infer-chain-graphcapture-2115/gb10/`。補遺規則は `logs/optin-ab-gb10-3190/RULE.txt` 6。
判定は RULE.txt のとおり機械的に当てはめた（事後に緩めていない）。

| 項目 | 結果 |
|------|------|
| 専有ゲート | 成立（A/B 起動前に enforce モードで通過。第 2 系列〈`2115-gb10-3190b`〉のゲートログは作業ミスで消失〈下記〉。計測が走ったこと自体が通過の推定証跡〈不成立なら計測せず終了する設計〉。残る実測証跡は `ab-stdout-2115-gb10-3190b.log` の run 終了時 `load average` 行〈2.59〜3.4 は自身の計測負荷を含む〉）。第 1 系列のゲートは 8 サンプル目で 3 連続 ok |
| R0（前提ゲート） | 合格（`linear_forward_device_real_device` 5 passed・`graph_capture_real_device` 2 passed） |
| R1（capture 正当性） | **(a) 合格**（`INFER_CAPTURE_R1: PASS` が 5 行）。**(b) 不合格**: README の事前登録コマンド（`--test-threads` 指定なし）で `predict_device_chain_cuda_bit_identity` が ON で 4 件失敗（`Backend(DeviceContextCaptureInProgress { ordinal: 0 })`。1 passed・4 failed）。bit 不一致ではなく、別スレッドの capture 中に同一デバイスを使った呼び出しがエラーを返したもの |
| R2（bit ダンプ diff） | 合格（batch 64／1024／4096 の OFF/ON ダンプが行数〈640／10240／40960〉・sha256 とも完全一致） |
| 診断（判定に使わない） | `--test-threads=1` で R1(b) を再実行すると 5 passed（`diag-r1b-single-thread.log`）。失敗は並列実行下の capture 競合に限られる |

| batch/mode | before 中央値 | after 中央値 | after/before | 5 run 一貫 < 1.00 | checksum |
|---|---|---|---|---|---|
| 64/fresh | 155.1 us | 143.4 us | 0.9243 | はい | 完全一致 |
| 64/reuse | 99.0 us | 85.2 us | 0.8610 | はい | 完全一致 |
| 1024/fresh | 298.9 us | 291.8 us | 0.9763 | はい | 完全一致 |
| 1024/reuse | 182.8 us | 169.4 us | 0.9270 | はい | 完全一致 |
| 4096/fresh | 804.3 us | 800.7 us | 0.9955 | いいえ（run 1 が 1.0054） | 完全一致 |
| 4096/reuse | 509.4 us | 497.2 us | 0.9761 | はい | 完全一致 |

**verdict: undetermined**。ADOPT 条件 1〜4（専有ゲート・全 6 セル中央値 ratio <= 1.00・checksum 完全一致・reuse の 5/5 一貫改善）は成立したが、
条件 5（R0〜R2 がすべて合格）が R1(b) の失敗で成立しない。REJECT 条件（中央値 ratio > 1.00・checksum 不一致・R1/R2 の **bit 不一致**）には該当しない
（R1(b) の失敗はエラーであり bit 不一致ではないため、規則の文言どおり REJECT にはしない）。診断の再実行が通ったことを理由に判定を ADOPT へ寄せない。

- 性能は全セルで改善方向（reuse 0.86〜0.98。batch 64 が最大）。fresh は created stream 化（event tracking 無効化）の寄与が小さく出ている。
- **既定の反転の前に直すべき欠陥**: `FANDHE_AI_CUDA_GRAPH_INFER=1` のとき、別スレッドが capture 中の同一デバイスで推論を呼ぶと
  `DeviceContextCaptureInProgress` をエラーとして返す。複数スレッド推論（既存の並列テストと同じ使い方）が壊れるため、待機またはフォールバックが要る。
  これを直したうえでの R1(b) 再判定は、別 label・新規の事前登録で行う（#3205 へ申し送り）。
  **#3205 の再判定（2026-10-23。label `3205-b3-rejudge`）**: `predict_resident` の競合は是正（待ち上限 1 s・上限内で繰り返し）で消えたが、
  R1(b) のテスト自体が旧経路（`init_device_param_store`・`forward_resident`）でも同じエラーに当たり不合格のまま。verdict は undetermined。
  詳細は `infer-chain-fixed-cost-reduction-3205.md` §10。
- 記録上の注意（作業ミスの開示）: 第 1 系列（`2115-gb10-3190`）は、セッションスクリプトの `CARGO_TARGET_DIR` 設定がスクリプトのバイナリ探索を壊し、
  全 run が起動できず計測値ゼロで失敗した（`invalid-attempt1-2115-gb10-3190/`）。結果を一度も見ていない状態の環境起因の失敗で、別 label `2115-gb10-3190b` で再実行した。
  第 2 系列の `compare` 表・JSONL・ゲートログは、成果物の掃除（`git clean`）で作業ツリー上のファイルを消してしまったため、
  実行時の標準出力ログ（`ab-stdout-2115-gb10-3190b.log`。全 run の JSON 行を含む）から JSON 行を無加工で取り出して再構成し（`*.reconstructed.jsonl`）、
  同じ `compare_gemm_ab.py` コマンドで表を再生成した。値の改変・補完はしていない。

## 4. 実装セッションで確認できたこと

- GPU 不要の単体テスト: opt-in の環境変数 allowlist・既定 OFF・override 往復・カウンタ・TLS 破棄後のキャッシュ操作、
  `linear_chain_forward_captured` 既定 `Ok(None)`、`predict_device_chain` の captured 経路（`Some` は層ループ・upload・download を経由しない／`Err` は poison せず伝播／`None` は既存経路へ）、bench-fandhe `--infer-batch` allowlist、`compare_gemm_ab.py --sizes infer-batches`
- capture 経路の実機挙動（bit 同一・replay・weight 更新反映・再 capture・スレッド終了）は GB10 の R1 で検証する（テストは実装済み）

## 5. スコープ外（後続）

- GB10 での R0〜R2 と 5 run A/B の実測
- ADOPT 時の既定反転 PR
- facade 公開 API（`set_cuda_graph_infer_enabled` 相当）の要否判断（#3115 で公開を決定し、#3116 で公開済み。効果の実測は Phase 8）
- `cuGraphExecUpdate`（`unsafe`・承認要）による batch 可変対応
- Metal への同等機構

## 6. 既定化の影響評価（イシュー #4163。部品のみ・未実測）

推論 Graph（`FANDHE_AI_CUDA_GRAPH_INFER`）を既定にした場合の影響を、事前登録した A/B で評価する。**本節の時点（PR-1）は計測部品の追加だけで、GB10 の実測はまだない。** 記録先は `logs/cuda-infer-graph-default-ab-4163/`（手順は同 `README.md`）。`INFER_GRAPH_DEFAULT_ENABLED` の反転は本イシューでは行わない（所有者承認・結線は #4167）。

### 6.1 計画の前提を変える所見（コード確認）

- **0.14.0 の本番経路（スコアボード G-CUDA-INF reuse）は capture に届かない。** bench の `run_infer_reuse` は #3272 以降、常駐入力の `predict_resident_input` → `DeviceParamStore::predict_device_chain_resident` を通る。`predict_device_chain_once` が `linear_chain_forward_captured` を呼ぶのは `ChainInput::Host` のときだけ（`crates/autodiff/src/optim/device_store.rs`）。したがって本番 reuse セルでの既定 ON の効果は created stream 化（event tracking の無効化）だけになる。§3 の 0.861 はホスト入力だった旧 bench の値で、同じ式で本番セルへ当てはめない。
- そのため capture が届くホスト入力経路を窓内で呼ぶ診断腕 `--diag-host-input-reuse`（`bench-fandhe`。`--task infer --mode reuse --device cuda`、既定でない `--out` の明示が必須。出力行に `diag":"host_input_reuse"`）を追加した。常駐入力経路の capture 対応は製品の挙動の変更になるため実装せず、#4167 への所見とする。
- **R1(b) の原因 2 は #3637 で是正されている可能性がある**（チェーン外の操作を待機・再試行させる。結果は予測しない）。現ツリーで事前登録どおり（`--test-threads` 指定なし）に再実行し、合否と失敗箇所を記録する。
- **既定 ON は学習にも効く。** `CudaDevice::new` の stream 種別は `step_graph_mode || infer_graph_stream_requested || train_graph_stream_requested` の OR で、推論の既定 ON は全 CUDA 利用者の既定 stream を created に変える。train の fresh／reuse を非後退セルに含める。

### 6.2 設計

- 腕: off = 3 変数未設定、on = `FANDHE_AI_CUDA_GRAPH_INFER=1`（同一バイナリ。env=1 と const `true` は同じ `resolve_infer_requested` を通り同値）。const 反転ビルド（`default-on.patch`）は正しさの R3（既定 ON の波及テスト）にだけ使う。
- セル（11）: infer fresh／reuse（常駐入力）／reuse-host × batch {64, 1024, 4096}、train fresh／reuse × 64。5 round。
- 前提ゲート: R0（非後退）・R1(a)（capture 実機）・R1(b)（並列推論 5 回）・R2（off／on のダンプ一致）・R3（既定 ON の全実機テスト差分）。判定規則は `RULE.template.txt`（PR-2 で `RULE.txt` として測る前に単独コミット）。

### 6.3 状態

| 項目 | 状態 |
|---|---|
| 計測部品（bench 診断腕・ランナー・ゲート・集計・雛形） | 追加済み（PR-1）。GB10 での動作は未検証 |
| RULE.txt の固定 | 未（PR-2） |
| GB10 実測・判定 | 未（PR-3） |
| #4167 の判断 | 結線見送り（ADOPT 入力なし。§7） |

## 7. #4167 既定経路への結線の判断（見送り）

### 7.1 判断

結線は見送る。`INFER_GRAPH_DEFAULT_ENABLED = false`・`TRAIN_GRAPH_DEFAULT_ENABLED = false` を維持し、コードは変更しない。これは自動運転セッションでの安全側の判断であり、所有者が見送りを承認したものではない。所有者はいつでも見直せる。

### 7.2 理由（入力の棚卸し）

- 推論 Graph の既定化（#4163）: PR #4199 は計測部品（PR-1）のみ。RULE.txt の固定・GB10 実測・`judge-4163.md` がなく、ADOPT 判定が存在しない。旧 A/B（§3。#3205 の再判定を含む）も verdict は `undetermined`。
- 学習 Graph の既定化: `docs/backend-cuda-graph-step-capture-design.md` §8.12 で見送り確定。Phase 3 の A/B 対象にも入っていない。
- その他の CUDA 項目（#4150 ほか）: 診断のみで、ADOPT 候補ではない。
- 既定反転の前提となる所有者承認も得ていない。

Phase 3 由来の CUDA 系 ADOPT は 0 件で、結線する入力がない。

### 7.3 前後比較

結線がないため「変更なし（0.14.0 の既定経路のまま）」と記録する。#4168 の CUDA セル（G-CUDA-INF ほか）は結線による変化なしとして引き継ぐ。

### 7.4 再開条件（すべて満たしたときだけ、新しい結線イシューで行う）

1. #4163 の PR-2（RULE.txt を測る前に単独コミット）と PR-3（GB10〈`<cuda-node>`〉実測・`judge-4163.md`）が完了し、判定が ADOPT であること（R1(b)・R3 を含む）。
2. 所有者が既定反転を承認し、結線イシューを起票すること。
3. 反転 PR で次を行う。
   - const を反転する。
   - `default-off-assumption-tests.txt` の 2 件（`lib::graph::tests::infer_graph_default_is_off`・`graph_capture_real_device_optin_off::opt_in_off_keeps_legacy_stream_and_returns_none`）を既定 ON 前提へ更新する。
   - R3 相当（既定 ON での `#[ignore]` 実機テスト全件の差分）を行う。
   - 推論・学習の両方で前後比較を記録する（既定 ON は `requires_created_stream` を通じて全 CUDA 利用者の stream を created に変える）。

### 7.5 所見（結線で得られる効果の上限）

§6.1 のとおり、本番の G-CUDA-INF reuse セル（常駐入力 `predict_device_chain_resident`）は capture に届かない。ADOPT 後に反転しても本番セルでの効果は created stream 化だけになる。常駐入力経路の capture 対応は製品の挙動の変更で、別の A/B と所有者承認が要る。

### 7.6 注意

#4163 は部品のみの PR #4199 で閉じられている。reopen するかは所有者または main セッションの判断に委ね、本記録では扱わない。
