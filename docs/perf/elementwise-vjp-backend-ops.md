# elementwise VJP の BackendOps 経由化（イシュー #1583）

## 背景

#1211 が GEMM 系 VJP（`matmul_vjp`・`Op::LinearResident.d_weight`）を
`eval::matmul`（ホスト scalar 参照実装）から `BackendOps::gemm_fp32_strict`
（forward と同じ CPU BLIS／CUDA／Metal カーネル）へ切り替えた。本イシューは
残る elementwise VJP（`Op::Mul`・`Op::Exp`／`Tanh`／`Sigmoid` の乗算・
`backward.rs::accumulate` の fan-out 勾配合算）についても同じ方針を適用
できるか検証する。

## 設計

- ゲート定数 `crates/autodiff/src/grad.rs::ELEMENTWISE_VJP_VIA_BACKEND_OPS`
  （`#1578` の `MSE_BACKWARD_PARALLEL_MIN_ELEMS` 方式と同型）。
- 切替対象: `Op::Mul`（`da`／`db`）・`Op::Exp`／`Tanh`／`Sigmoid`（
  `g ⊙ factor`）・`backward.rs::accumulate` の fan-out 合算。いずれも
  単一 IEEE 演算（乗算または加算 1 回）で縮約を含まないため、
  `ops.mul`／`ops.add` と `eval::mul`／`eval::add` は run-to-run・
  バックエンド間を問わず bit 同一。
- **対象外**（不変）: `Op::Relu`／`LinearAct`／`LinearResident` のマスク
  演算（`elementwise_mul_mask`。#1577 の stride 対応 host 経路。
  `BackendOps` にマスク演算面がなく追加は公開 trait 拡張のため別途
  ユーザー承認事項）、`Op::Add` の broadcast 縮約（`reduce_bias_grad`／
  `reduce_to_shape`。f64 アキュムレータ統一・Metal 側 `BackendOps::sum`
  未実装のため対象外）。
- フォールバックは `BackendError::Unsupported` の場合のみ `eval` 経路へ。
  他のエラーは `AutodiffError::Backend` として fail-closed に伝播し、
  バックエンド実装が誤った shape を返した場合も fail-closed でエラーに
  する（`vjp_elementwise_mul`／`vjp_elementwise_add` doc 参照）。
- **付随是正**（イシュー #1583 スコープ）: `crates/backend-cpu/src/
  elementwise.rs::binary_elementwise` の non-contiguous 経路（broadcast
  拡張軸・transpose view）を、`autodiff::grad::MaskReadOperand`（#1577）
  と同型の stride 読み（`ElementwiseReadOperand::classify`。`as_slice`
  優先・次に `as_view_slice`）へ変更した。`Tensor::get`（rank・範囲検査
  つきの要素ごとアクセス）を経由しない借用スライス直接インデックスに
  なるが、値は不変（bit 同一）。forward の broadcast add/mul にも影響
  するため `crates/backend-cpu/tests/elementwise.rs` に bit 一致テスト
  （transpose × transpose・broadcast stride-0 × transpose・NaN／-0.0）
  を追加して回帰を防いでいる。

## 事前登録規則

イシュー #1583 のコメントに実装着手前に固定済み:
<https://github.com/Fandhe-AI/fandhe-ai/issues/1583#issuecomment-5646324585>

要旨（詳細は上記コメント本文を正とする）:

- 比較腕: before = `origin/main`（マージ base）を `git archive` した
  非 git ツリー、after = 実装ブランチ worktree（ゲート `true` へ一時
  変更してビルド）。
- マイクロベンチ（`crates/facade/tests/elementwise_vjp_bench.rs`。
  `#[ignore]`）: 対象ケース (A) mul 連続×連続・(B) mul upstream が
  transpose view・(C) mul broadcast・(D) tanh・(E) sigmoid・(F) fan-out
  （`backward.rs::accumulate` 到達）。`numel ∈ {16384, 65536, 1048576}`
  （`PARALLEL_THRESHOLD=32768` を挟む）。各セル 5 プロセス起動・run
  単位で before/after 起動順反転・負荷ゲートなし（record_only）。
- 判定: バックエンドごとに全対象セル `ratio(after/before) <= 1.00`
  かつ勾配 bit 完全一致 → ADOPT。いずれか `ratio > 1.00` → REJECT。
- 出荷規則: 計測できた全バックエンドが ADOPT のときのみ
  `ELEMENTWISE_VJP_VIA_BACKEND_OPS = true`。1 つでも REJECT なら
  `false`（機構は維持し出荷しない）。
- 事後緩和・セル除外・run 追加による再判定は行わない。

## 正しさの検証（性能とは独立）

- `crates/autodiff/src/grad.rs::tests` に `vjp_elementwise_mul_via`／
  `vjp_elementwise_add_via`（ゲート値を引数化したテスト用バリアント）
  の単体テスト 12 件を追加: ゲート false／true の bit 完全一致・
  broadcast・`Unsupported` フォールバック・非 `Unsupported` エラー伝播・
  誤 shape の fail-closed 拒否・ゲート既定値のドリフト検出。
- ゲートを一時的に `true` へ切り替えた状態で `fandhe-ai-autodiff`
  （230 件）・`fandhe-ai`（facade。mnist 収束テスト含む全件）が
  green であることを確認済み（実装時に本エージェントが実行。ドリフト
  検出テスト 1 件のみ意図的に FAIL することも確認）。
- マイクロベンチの勾配ダンプ（`fold_bits`）は M4 Max（cpu／metal）・
  GB10（cuda）の全 18 セル × 5 run で before/after 完全一致
  （`aggregate.py` の `grad_ok` 列。下記ログ参照）。

## 実測結果

### Apple M4 Max（cpu）

`docs/perf/logs/elementwise-vjp-backend-ops-1583/m4max/`（5 run・
record_only・共有負荷 load average 約 7〜9）。

18 セル中 8 セルで `ratio > 1.00`（`mul_contig`／`mul_broadcast`／
`sigmoid`／`tanh`／`fan_out_accumulate` の `numel=65536` が特に顕著
〈2.2〜6.0 倍〉）。`numel=65536` は `PARALLEL_THRESHOLD=32768` 超で
`CpuBackendOps::mul`／`add` が rayon 並列経路へ切り替わる境界であり、
共有負荷下では並列フォーク・ジョインのスレッド同期コストが逐次実装
（`eval::mul`／`eval::add`）を上回ったと見られる（原因は記録するが
判定規則自体は変えない）。勾配 bit-fold は全セル一致。

**判定: REJECT**（cpu）

### Apple M4 Max（metal）

`docs/perf/logs/elementwise-vjp-backend-ops-1583/m4max-metal/`（5 run・
record_only・cpu 計測と同時刻帯・共有負荷）。

18 セル**全て**で `ratio > 1.00`（1.34〜16.8 倍）。事前の見通しどおり、
`ops.mul`／`ops.add` の per-call `broadcast_with` → `contiguous()` →
H2D → カーネル → D2H（`MetalBackendOps::elementwise_binary` 相当）の
往復コストが、孤立した単一 VJP セルではホスト側 1 回乗算
（`eval::mul`）を大きく上回る。勾配 bit-fold は全セル一致（REQ-2
複合判定ではなく単一 IEEE 演算のため bit 完全一致）。

**判定: REJECT**（metal）

### DGX Spark GB10（cuda）

`docs/perf/logs/elementwise-vjp-backend-ops-1583/gb10/`（5 run・
record_only・低負荷 load average 約 0.3〜1.5）。

18 セル中 13 セルで `ratio > 1.00`（1.06〜2.52 倍。`numel=1048576` の
一部セルは `ratio < 1.00` だが、他形状で規則を満たさないため全体判定
には影響しない）。Metal と同じ per-call H2D／D2H オーバーヘッドが
支配的と見られる。勾配 bit-fold は全セル一致。

**判定: REJECT**（cuda）

## verdict

3 バックエンド（cpu／metal／cuda）すべて REJECT。出荷規則により
`ELEMENTWISE_VJP_VIA_BACKEND_OPS = false`（既定値のまま・変更なし）
で確定する。機構（ヘルパー関数・ゲート定数・テスト）は維持する。

## スコープ外・後続への引き継ぎ

- **p1-a3（#1579〜#1581 の推論チェーン単一同期化）との組み合わせ**:
  Issue 本文が明記するとおり、GPU 側の孤立 VJP セルでの後退は
  per-call 同期・転送のオーバーヘッドが支配的であり、推論チェーン全体
  を単一同期にまとめる設計（p1-a3）と組み合わせれば個別呼び出しの
  オーバーヘッドが償却される可能性がある。本イシュー単独では検証しない
  （Issue 本文の記載どおり）。
- **CPU 側の並列閾値**: `numel=65536` 近傍（`PARALLEL_THRESHOLD` 超過
  直後）での rayon 並列経路のオーバーヘッドが REJECT の主因の一つ。
  `PARALLEL_THRESHOLD` の見直し自体は本イシューのスコープ外
  （`.claude/rules/delegation-impl.md` 「実装 Agent にテスト許容誤差を
  緩和させない」と同種の慎重さで、性能パラメータの変更もユーザー承認
  事項として別途起票が必要）。
- **`BackendOps` へのマスク演算面追加**（`Op::Relu` 系を切替対象へ拡大
  する場合の前提）は公開 trait 拡張のため別途ユーザー承認事項。
- **`Op::Add` の broadcast 縮約の backend 化**（Metal 側 `BackendOps::
  sum` 未実装）も対象外のまま。
- **GPU の `DeviceBuffer` 常駐 VJP**（per-call 転送そのものを無くす
  設計）は本イシューの範囲外（#1584 等、別イシューでの検討を想定）。

`crates/backend-cpu/src/elementwise.rs::binary_elementwise` の
stride 読み経路化（付随是正）は性能実測の対象外だが、forward の
broadcast/transpose 経路を `Tensor::get` の rank・範囲検査コストから
解放するため、独立の効果として維持する（ADOPT／REJECT 判定とは無関係
の正しさ改善）。
