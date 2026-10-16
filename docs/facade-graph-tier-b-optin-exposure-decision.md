# 区分 B の opt-in setter の公開形と B-1 新規 `unsafe` の承認状況（決定記録）

イシュー #3115（親 #3114・Phase 4 親 #3088）。**コード変更なし**。facade に公開する opt-in setter の形を確定し、後続 #3116 が新しい判断をせずに実装できる状態にする。本 doc は承認を代行しない。

## 0. 結論

| 対象 | 結論 |
|---|---|
| 推論チェーン CUDA Graph capture の setter | 新規 `unsafe` がなく、2026-10-11 の所有者判断 D-A に基づき公開できる形を §3・§4 で確定 |
| B-1（GPU elementwise 融合）の setter 2 対 | 形は §3 で確定。ただし B-1 の新規 `unsafe` 4 か所の承認が記録上不明（§6 R1）のため、**所有者が確認するまで公開を保留**する条件付き |
| 既定値 | すべて OFF のまま（§7） |

## 1. 位置づけ

- 根拠: `docs/graph-compile-scope-spec-proposal.md` §3（承認だけで着手できる区分 B）と、`docs/compat-api-scope.md` §5 末尾の D-A 適用記録（#3020）
- 後続: #3116 が facade に setter を公開しテストを足す。公開形は本 doc を正とする

## 2. 事実（`origin/main` 3eba97bb 時点）

| # | 事実 | 出典 |
|---|---|---|
| E1 | B-1 のゲートは `set_gpu_elementwise_fusion_enabled`／`gpu_elementwise_fusion_enabled`（CUDA `:77`／`:82`、Metal `:67`／`:72`）。`static AtomicBool`。facade に再公開はない | `crates/backend-{cuda,metal}/src/fused_elementwise.rs` |
| E2 | 推論 capture に setter はない。判定は `infer_graph_enabled()`（`graph.rs:697`）= thread-local テスト override → `infer_graph_stream_requested()`（`:681`。環境変数 `FANDHE_AI_CUDA_GRAPH_INFER` を `OnceLock` で 1 回だけ読む。`1`／`true` の完全一致のみ）→ `INFER_GRAPH_DEFAULT_ENABLED=false`（`:658`） | `crates/backend-cuda/src/graph.rs` |
| E3 | `device.rs:379-382` は `step_graph_mode().requires_created_stream() \|\| infer_graph_stream_requested()` で created stream を選ぶ。選択は ordinal ごとに sticky | `crates/backend-cuda/src/device.rs` |
| E4 | `linear_chain_forward_captured` は、ゲート OFF または stream が capture 不可のとき `Ok(None)` を返し既存経路へ黙ってフォールバックする（学習 step capture の fail-closed `Unsupported` とは異なる） | `crates/backend-cuda/src/ops.rs` |
| E5 | 学習 step capture の先例: `set_step_graph_enabled` は「API 明示 > 環境変数 > 既定 OFF」 | `graph.rs` |
| E6 | facade の既存 opt-in 対: `set_cuda_tf32_gemm_enabled`・`set_cuda_graph_step_enabled`・`set_cuda_managed_memory_enabled`・`set_cuda_pinned_h2d_enabled`・`set_metal_split_k_gemm_enabled`（macOS cfg）・`set_cuda_onnx_gpu_execution_enabled`・`set_metal_onnx_gpu_execution_enabled`（macOS cfg）ほか。getter が対で付く | `crates/facade/src/lib.rs`（`:2365-2650`） |
| E7 | facade の `backend-metal` 依存は `cfg(target_os = "macos")` 限定のため、Metal setter は macOS cfg でのみ存在できる | `crates/facade/Cargo.toml` |
| E8 | B-1 の新規 `unsafe` 4 か所: CUDA `crates/backend-cuda/src/elementwise.rs:681`（`launch_nary`）、Metal `crates/backend-metal/src/elementwise.rs:770`・`:776`・`:781`（`encode_nary_dispatch`）。PR #2232 で 2026-09-23 マージ | 各ファイル |
| E9 | 推論 capture の A/B 記録は「新規 `unsafe` なし」と書いている | `docs/perf/infer-chain-graphcapture-cuda-ab.md` |
| E10 | 既定反転の事前登録規則: B-1 は `docs/perf/gpu-elementwise-fusion-b1.md`、推論 capture は `docs/perf/logs/cuda-infer-chain-graphcapture-2115/RULE.txt`。既定反転は #3227 の担当 | 各 doc |

確認コマンド: `grep -n "fn set_gpu_elementwise_fusion_enabled" crates/backend-*/src/fused_elementwise.rs`、`grep -n unsafe crates/backend-cuda/src/elementwise.rs crates/backend-metal/src/elementwise.rs`、`grep -n "^pub fn set_" crates/facade/src/lib.rs`。

## 3. 公開する setter（公開形の正）

| 機構 | facade の setter／getter | 既定 | 対象 | cfg | 委譲先 |
|---|---|---|---|---|---|
| B-1 CUDA | `set_cuda_elementwise_fusion_enabled(bool)`／`cuda_elementwise_fusion_enabled() -> bool` | OFF | CUDA | なし | `fandhe_ai_backend_cuda::fused_elementwise::{set_,}gpu_elementwise_fusion_enabled`（既存・1 行委譲） |
| B-1 Metal | `set_metal_elementwise_fusion_enabled(bool)`／`metal_elementwise_fusion_enabled() -> bool` | OFF | Metal | `#[cfg(target_os = "macos")]`（E7） | `fandhe_ai_backend_metal::fused_elementwise::` 同名（既存・1 行委譲） |
| 推論 capture | `set_cuda_graph_infer_enabled(bool)`／`cuda_graph_infer_enabled() -> bool` | OFF | CUDA | なし | **新規** `fandhe_ai_backend_cuda::graph::set_infer_graph_enabled(bool)` と既存 `infer_graph_enabled()` |

- 命名: facade は `cuda_`／`metal_` 接頭辞のバックエンド別の対で揃える（E6）。推論 capture は学習側 `set_cuda_graph_step_enabled` と環境変数名に揃える。CUDA と Metal を一括で切り替える単一 setter は作らない（ADOPT 判定がバックエンドごとに独立。ONNX GPU 実行の対が先例）
- 型はプロセスワイドの `AtomicBool`。`Device` 単位の切り替えは設けない
- 非対応環境: CUDA の setter は CUDA 非搭載環境でも呼べる（フラグを立てるだけで driver に触れない）。CPU のみの実行では効果がなくパニックもしない。Metal の setter は非 macOS ビルドには存在しない

## 4. 推論 capture setter の追加仕様（#3116 が backend-cuda に足す最小変更。新規 `unsafe`・依存なし）

- 優先順位は「API 明示 > 環境変数 > 既定 const」。`set_step_graph_enabled` と同型の `EXPLICIT` フラグを持つ `AtomicBool` 対で実現する（E5）
- テスト専用 thread-local override は最上位のまま残し公開しない
- `infer_graph_stream_requested()` も API 明示値を反映する。stream 種別は ordinal ごとに sticky なので thread-local override は混ぜない（E3）
- **設定タイミングの契約**: 最初の CUDA デバイス初期化より前に呼ぶ。間に合わない場合、その ordinal の stream は capture 不可のまま残り、`linear_chain_forward_captured` は既存契約どおり `Ok(None)` を返して既存経路で動く（出力は非 capture 経路と同一）。fail-closed への変更は #2115 の契約変更になるため採らない。rustdoc にこの差を明記する
- `InferGraphStats` の facade 版は今回公開しない。テストは `fandhe_ai_backend_cuda::graph::infer_graph_stats` を直接参照する
- 環境変数は従来どおり完全一致 allowlist（`1`／`true`）で解釈し、値をログへエコーしない

## 5. 公開しないもの

- 利用者向けの融合制御 API（融合 op の選択・融合境界の指定・`Tape`／`Sequential` 単位の融合フラグ）
- 汎用 `compile`（`torch.compile`／`tf.function` 相当）の入口、トレース・書き換え・コード生成（spec では対象外のまま。分岐 (i)）
- `override_infer_graph_for_scope`・`InferGraphOverrideGuard`・`InferGraphStats`／`infer_graph_stats`
- `match_elementwise_plan`・`ElementwiseProgram`・融合カーネルキャッシュ上限などの内部
- B-2〜B-6（B-3 の学習 step 部分と B-5 は前提ゲート未充足、B-6 は着手根拠なし、B-2／B-4 は本 issue の範囲外）
- REQ-12 の解釈: 本 setter は TF32 等と同型のプロセスワイドな実行戦略トグルで、どの演算をどこで融合するかを利用者が指定する手段ではない。rustdoc に「利用者が融合を直接制御する API ではない」と書く

## 6. B-1 新規 `unsafe` の承認状況

時系列（出典のまま。言い換えない）:

1. #2085 追補（`docs/autodiff-graph-optimization-scope-decision.md`・`docs/perf/gpu-elementwise-fusion-b1.md`）は、新規 `unsafe` 4 か所を「本 PR のユーザー承認事項 (3)」として列挙しているのみで、取得の記録はない
2. PR #2232 のレビュー投稿は Codex 自動レビューと Cursor Bugbot のみ。Codex は「unsafe の根拠」を確認対象に挙げ P0/P1 なしとしたが、自動レビューであり security-auditor の監査記録でも所有者の承認でもない。#2085 のコメントは 0 件
3. #2615 の 2026-10-07 コメントは分岐 (i) を承認し、「§7 項 3 の区分 B（新規 `unsafe` FFI を含む実装の着手）は承認していません」と明記
4. 2026-10-11 の D-A（`compat-api-scope.md` §5 末尾の適用記録〈#3020〉）の generate 系の行は #3114 を各 issue の決定記録を正とすると述べるのみで、`unsafe` への言及はない。同記録の D-C 注記は新しい `unsafe` は個別に確認を得るとする
5. `docs/` で `launch_nary`／`encode_nary_dispatch` を検索してもヒットは 1 の 2 ファイルのみ。security-auditor の監査記録は見当たらない

**結論: 承認状況は記録上不明。「取得済み」「監査済み」とは書かない。**

所有者へ戻す点:

| 項 | 内容 | 記録上の状況 | 戻し先・影響 |
|---|---|---|---|
| R1 | B-1 の新規 `unsafe` 4 か所（E8）の承認 | 承認記録なし。区分 B の `unsafe` を含む着手は 2026-10-07 時点で未承認。D-A の行に言及なし | 所有者へ確認する。#3116 の B-1 setter 2 対の公開は R1 の回答まで保留 |
| R2 | 同 4 か所の security-auditor 監査記録 | doc・PR #2232 に見当たらない | R1 で求められれば監査記録の作成を別 issue として提案（起票はユーザー承認後） |

- 推論 capture は新規 `unsafe` がなく（E9）R1・R2 の対象外。#3116 の追加範囲（委譲と `AtomicBool`）でも新規 `unsafe` は生じない
- ただし B-1 setter を公開すると既存 4 か所が facade 利用者から到達可能になる。R1 を公開の前提とするのはこのため

## 7. 既定 ON 化

本 issue と #3116 では既定 OFF のまま。反転は事前登録規則（E10）の ADOPT が実機実測で出た後の別判断（#3227）。推論 capture の反転は全 CUDA 利用者の既定 stream を created に変えるため影響評価を伴う。GB10・M4 Max の実測は未実施。

## 8. 関連 doc への追記

- 本 issue: `graph-compile-scope-spec-proposal.md`「#3115 追補」・`autodiff-graph-optimization-scope-decision.md`「#3115 追補」・`compat-api-scope.md` §5 適用記録・`docs/README.md` 索引
- #3116: 上記 3 doc へ実装記録を追記。`crates/facade/tests/api_surface.rs` の正ガード（先例: `set_cuda_onnx_gpu_execution_enabled`）

## 9. 変えないもの

汎用 JIT は対象外のまま。tolerance・baseline・依存・`docs/spec/`・既定値・既存環境変数の意味・facade 公開面は変更しない。

## 10. OWASP 観点

- A01: 承認は記録された事実のみ書き、代行・捏造しない
- A03: 環境変数は完全一致 allowlist（fail-closed）
- A04: 公開面は既定 OFF のプロセスワイド toggle に限り、設定タイミング遅れ時の挙動を明文化
- A06: 依存の追加なし
- A08: B-1 公開は R1 を前提条件とし、監査記録なし（R2）を明記

## 11. 出典

`docs/graph-compile-scope-spec-proposal.md`・`docs/autodiff-graph-optimization-scope-decision.md`・`docs/compat-api-scope.md` §5・`docs/perf/gpu-elementwise-fusion-b1.md`・`docs/perf/infer-chain-graphcapture-cuda-ab.md`・`docs/facade-inference-serving-scope-decision.md`

## 12. #3116 実装記録

`set_cuda_graph_infer_enabled`／`cuda_graph_infer_enabled`（facade）と `fandhe_ai_backend_cuda::graph::set_infer_graph_enabled`（新規。優先順位は API 明示 > 環境変数 > 既定 OFF）を公開した。テストは CPU で常に実行される受入テスト（`cuda_graph_infer_optin.rs`）・`api_surface.rs` の正ガードと、`#[ignore]` の CUDA 実機 parity テスト（`cuda_graph_infer_parity_real_device.rs`。実測は Phase 8・未実行）。**B-1 の CUDA／Metal の対は未公開**（新規 `unsafe` 4 か所の承認が記録上不明のため、R1 への所有者の回答待ち。決定記録 §6）。既定 OFF・tolerance・baseline・依存・`docs/spec` は不変で、新規 `unsafe` はない。既定 ON 化は #3227 の担当。 本文 §0〜§11 は書き換えていない。

## 13. #3385: R1・R2 の状況更新と B-1 実装記録

### 13.1 R1（承認）

所有者が #3114 のコメント（id 6213949713・2026-10-16）で R1（B-1 新規 `unsafe` 4 か所の承認）を承認した。条件は「4 か所を監査し、結果を決定記録に残すこと（R2）」。承認の範囲外: 依存追加、tolerance・baseline・ガードレール閾値の変更、既定 ON 化（#3227）、4 か所以外の新規 `unsafe`。

### 13.2 R2（監査）

- 実施日: 2026-10-16。対象: `backend-cuda/src/elementwise.rs` `launch_nary`（`unsafe` 1 か所）と `backend-metal/src/elementwise.rs` `encode_nary_dispatch`（3 か所）。
- **実施者の限界**: 本実装セッションにはサブエージェントを起動する手段がなく、静的な読解監査を実装担当が行った。独立した security-auditor による再確認はレビュー段階の要対応事項として残す（outOfScope に記載）。
- 確認できた点（P0/P1 指摘なし。ファイルはいずれも `origin/main` cdf66c29 時点）:
  1. CUDA: `unsafe` 直前に全 leaf の長さ一致検証・`validate_elementwise_len`（i32::MAX 以下）・`numel == 0` の早期 return を行い、各 leaf を同長のデバイスバッファへ `clone_htod`、出力も `alloc_uninit_f32(numel)`。グリッドは `div_ceil` で切り上げ、生成カーネルは `if (idx < numel)` で余剰スレッドを弾く（REQ-8）。SAFETY コメントの主張と一致。
  2. Metal: leaf は index 0..n、out は n、numel は n+1 に結線し、生成 MSL の `[[buffer(i)]]` 宣言（`fused_elementwise_source.rs`）と一致。`numel` は `validate_elementwise_len`（u32::MAX 以下）済みで `setBytes` 長は `size_of::<u32>()`。バッファは `dispatch_sync` 完了まで生存。MSL も `if (idx < numel)`。
  3. 引数数の整合: `run_fused_elementwise_allowlist`（CUDA／Metal）が dtype F32 限定・`leaves.len() == program.leaf_count`・各 leaf の shape が `output_shape` と一致・`checked_bytes_for::<f32>`・`contiguous()` を起動前に検証。`FusionPlan::from_ops` が `leaf_index` を `0..leaf_count` に重複なく要求する。
  4. Metal のバッファスロット上限 31: `match_elementwise_plan` が `leaf_count <= MAX_FUSED_SEGMENT_NODES + 1`（= 13）を課すため最大 index は 14 で上限内。この強制はディスパッチ前（ゲート判定後・パイプライン生成前）で行われる。
  5. facade からの到達経路: カーネルソースは allowlist 済みの `FusedOpKind` 列と `leaf_count` のみから生成され、利用者の文字列は NVRTC／MSL に届かない。setter は `AtomicBool` を書くだけで `unsafe` の前提はゲート値に依存しない。ゲートは `run_fused` 呼び出しごとに 1 回読む。CUDA は `with_driver_call` で Graph capture と排他。
  6. 新しい setter は新規 `unsafe`・依存・環境変数読み取りを増やさない。
- P3（記録のみ・本 PR では修正しない）: Metal 側 `encode_nary_dispatch` の SAFETY コメントには、バッファスロット上限（31）が `match_elementwise_plan` で強制されている旨が書かれていない。コメント補強は backend 変更になるため別途。

### 13.3 実装記録

- 公開: `set_cuda_elementwise_fusion_enabled`／`cuda_elementwise_fusion_enabled`、macOS 限定の `set_metal_elementwise_fusion_enabled`／`metal_elementwise_fusion_enabled`（いずれも §3 の名前・1 行委譲・既定 OFF）。
- テスト: CPU で常に実行する `crates/facade/tests/gpu_elementwise_fusion_optin.rs`、`#[ignore]` の実機 parity `gpu_elementwise_fusion_parity_real_device.rs`（実測は Phase 8・未実行）、`api_surface.rs` の正ガード・否定ガード・正のプローブ。
- 着手時に確認した事実: `api_surface.rs` に B-1 の保留ガード・否定プローブは存在せず、反転対象はなかった（正ガードと否定ガードを新規追加）。B-1 には経路カウンタがなく、融合出力は per-op と bit 一致するため ON/OFF の出力比較では経路を区別できない。代替として `run_fused` を直接呼んだ返り値の分類（OFF は `Unsupported`）で facade setter が backend ゲートに効くことを確認した。カウンタは backend 変更を伴うため追加していない。
- `gpu_elementwise_fusion_bench.rs` は facade の公開 setter を使うよう置き換えた（環境変数の解釈・計測は不変）。
- 不変: 既定 OFF・tolerance・baseline・依存・`docs/spec`・新規 `unsafe` なし。既定 ON 化は #3227、実測は Phase 8。本文 §0〜§12 は書き換えていない。

### 13.4 独立監査（security-auditor、2026-10-16）

§13.2 が「残る」とした独立 security-auditor による再確認を実施した（読み取り専用・静的監査。なお、本節は §13.3 実装記録の後に追記したため番号は 13.4 とした）。所有者の条件（監査を通し結果を決定記録に残す。#3114 コメント 6213949713）は本監査で満たされた。

- 対象: CUDA `launch_nary`（`crates/backend-cuda/src/elementwise.rs`）、Metal `encode_nary_dispatch`（`crates/backend-metal/src/elementwise.rs`、`unsafe` 3 か所）、facade の `set_{cuda,metal}_elementwise_fusion_enabled` から `run_fused` への到達経路。
- 判定: **P0・P1 なし**。`unsafe` の前に葉長一致・numel 上限（CUDA `i32::MAX`／Metal `u32::MAX`）・葉数上限 13・dtype F32・葉 shape 一致・`checked_bytes_for` が揃う。生成カーネルは `if (idx < numel)` の手動境界検査（REQ-8）を維持する。空テンソル・非連続・ブロードキャスト（allowlist 外）・入力数超過は `unsafe` に到達せず per-op フォールバックか Err になる。
- P2 4 件:
  1. `launch_nary` の SAFETY コメントに「`func` は `leaves.len()` 個の葉 + out + numel のシグネチャでコンパイルされている」前提がなく、関数自体にも検査がない（呼び出し元は `fused_elementwise.rs` の 1 か所で、現状到達不能）。
  2. `encode_nary_dispatch` に葉バッファ数の上限検査がない（上流で 13 に制限。Metal の buffer index 上限は 31）。
  3. Metal は numel が `u32::MAX - 255` 以上で grid が 2^32 を超え idx が回り込む（重複書き込みのみ・範囲外アクセスなし・既存 elementwise カーネルと共通・実機未確認）。
  4. CUDA `fused_elementwise.rs` の「facade への再公開は本 PR の対象外」コメントが古い。
- 是正（本 PR・挙動不変）: (1) は SAFETY コメントに前提を明記（関数に葉数とプログラムの対応情報がないため `debug_assert` は足せない）。(2) は SAFETY コメントに前提を明記し `debug_assert!(leaf_bufs.len() + 1 < 31)` を追加（release 不変）。(4) は CUDA・Metal 両 `fused_elementwise.rs` のコメントを #3386 で公開済みの記述へ修正。§13.2 の P3（Metal SAFETY コメント補強）も (2) で解消。
- 残置: (3) は既存 elementwise カーネル共通の挙動のため別扱い（本 PR では修正しない）。
- 未確認: Metal 実機動作、`alloc_uninit_pooled` の容量検査が `debug_assert` のみであること、カーネル／パイプラインキャッシュ内部（精読していない）。
