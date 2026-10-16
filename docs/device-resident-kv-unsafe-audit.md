# デバイス常駐 KV（K-3）の新規 `unsafe` の監査記録（イシュー #3111）

## 1. 状態

| 項目 | 内容 |
|------|------|
| 承認状態 | 新規 `unsafe` の追加自体は、所有者が #3018 のコメント（<https://github.com/Fandhe-AI/fandhe-ai/issues/3018#issuecomment-6207013704>・2026-10-16）で承認済み（`docs/device-resident-kv-paged-attention-design.md` §9 の項目 3）。**security-auditor による個別監査は承認とは別に必須**で、本 doc の時点では**未実施**（下記） |
| 監査の種類 | 実装担当による**自己点検**（機械的インベントリ・全ブロックの手読み・Linux 上の検証）。独立した security-auditor の監査ではない。PR のレビューで security-auditor の監査を受け、その結果で本 doc を更新する |
| 変更範囲 | 本 doc は記録のみ。依存・tolerance 定数・baseline・ガードレール閾値・`docs/spec/` は変更しない |
| 未検証 | **CUDA カーネルの NVRTC コンパイルと実機実行**（DGX Spark GB10。#3171）、**MSL のコンパイルと Mac 実機実行**（#3172）。実装環境（Linux・RTX 3060・`libnvrtc` なし）では検証できていない。`#[ignore]` テストは用意済みで未実行 |

## 2. 新規 `unsafe` の一覧（機械的インベントリ）

再現コマンド: `grep -n "unsafe" crates/backend-cuda/src/paged_kv.rs crates/backend-metal/src/paged_kv.rs`。

| 場所 | 区分 | 内容 |
|------|------|------|
| `crates/backend-cuda/src/paged_kv.rs` の `launch_append` 内 `unsafe` | CUDA launch | `paged_kv_append_f32` の起動（`launch_builder(..).launch(..)`） |
| `crates/backend-cuda/src/paged_kv.rs` の `launch_attention` 内 `unsafe` | CUDA launch | `paged_attention_f32` の起動 |
| `crates/backend-metal/src/paged_kv.rs` の `encode_append` 内 `unsafe` ×2 | Metal encode | `setBuffer_offset_atIndex`（バッファ結線）・`setBytes_length_atIndex`（スカラー） |
| `crates/backend-metal/src/paged_kv.rs` の `encode_attention` 内 `unsafe` ×2 | Metal encode | 同上 |

- 合計 6 ブロック（CUDA 2・Metal 4）。いずれも FFI 境界で、直前に `// SAFETY:` コメントがある
- 次のクレート・モジュールは新規 `unsafe` を持たず、`#![forbid(unsafe_code)]` を付けた: `crates/autodiff/src/nn/paged_kv_device.rs`・`crates/autodiff/src/nn/paged_kv.rs`・`crates/backend-cpu/src/paged_kv.rs`。`crates/tensor-core/src/paged_kv.rs` と `crates/backend-metal/src/paged_kv_model.rs` も `unsafe` を持たない
- `Cargo.toml`／`Cargo.lock` は変更していない（`cudarc`・`objc2` 系は既存の固定版のまま）

## 3. 観点ごとの点検結果

### 3.1 境界（OOB 読み書き。REQ-8）

| 観点 | 結果 |
|------|------|
| ホスト側の検証 | 3 バックエンドは FFI の前に `tensor-core::validate_paged_kv_append`／`validate_paged_attention` を通す。長さの一致・スロット範囲・重複スロット・使用範囲のブロック ID（番兵 `u32::MAX` は拒否）・`seq_len` の範囲・`num_heads` の割り切り・`scale` の有限性・作業バッファ上限（`1 << 30` 要素）を検査する |
| launch 直前の再検証 | CUDA は `append_launch_params`／`attention_launch_params`、Metal は `paged_kv_model::append_dispatch`／`attention_dispatch` が、バッファ長の整合とカーネル引数型（`i32`／`u32`）への収まりを再検査する。違反は型付きエラー |
| カーネル側の手動境界検査 | append は `idx < total`（CUDA）・`r < rows`・`e < embed_dim`・`slot < pool_rows`。attention はブロックテーブル添字・ブロック ID・`seq_len` の範囲・`b < batch` を検査し、違反時は出力に NaN を書いて抜ける（黙ってゼロにしない）。ベクトル化ロードは使わない。ソース不変条件はテストで固定（CUDA: `kernels_paged_kv::tests`、Metal: `paged_kv_model::tests`。Linux でも実行される） |
| インデックス計算 | 64bit（CUDA `long long`・MSL `ulong`）。ホスト側のサイズ計算は `checked_*` と `isize::MAX` |
| 結論 | 追加の指摘なし。ただしカーネルが実際に NVRTC／MSL でコンパイルでき、実機で境界検査が期待どおり働くかは**未検証**（§1） |

### 3.2 生存期間とエイリアシング

- CUDA: 入出力は `CudaArg`／`CudaArgMut` で、launch の完了までローカルに生存する。`slot_mapping`・`block_tables`・`seq_lens`・作業バッファは `clone_htod`／プール確保のストリーム順序付きデバイスメモリで、カーネルより後に drop される（`cuMemFreeAsync`）。K／V プールと出力は呼び出し元が `&mut` で排他的に借用している
- Metal: encode は待たずに返るので、全バッファを `ctx.encode` の `resources` に渡して `Batch::in_flight` へ retain させる（`sgd.rs::MetalSgd::run` と同じ契約）。作業バッファ・index バッファも `resources` に含めた。`token: None` は `encode` 直後に `synchronize` する
- K／V プール（書き込み）と K／V 入力（読み取り）は別バッファで、重複スロットはホストで拒否するため書き込みは競合しない

### 3.3 データ競合

- CUDA・MSL とも、attention の `s_bad`（共有メモリのフラグ）は複数スレッドが同じ値 `1` を書く。値が一致するので結果は決定的だが、厳密にはデータ競合である。違反時の多層防御にだけ使い、バリアの後で読む。アトミック化は縮約順序の主張（アトミック・warp 集約を使わない）と矛盾するので採らず、残リスクとして記録する
- score 作業バッファは各ブロックが自分の行だけを書き、同ブロック内ではバリアを挟んで読む（CUDA `__syncthreads`・MSL `threadgroup_barrier(mem_device | mem_threadgroup)`）

### 3.4 失敗時の状態（OWASP A08）

- 追記は `BlockTable` を複製して確保し、バックエンドの書き込み成功後にだけ確定する。バックエンドが実行時エラーを返したら `PagedKvCache` を poison する（`Unsupported` は何も書かないので poison しない）
- Metal の遅延失敗は同一ロック区間で `failure_token` に登録される。`attention` は `download` の後に再検査して poison する
- CUDA は ordinal 単位の poison と世代検査に参加する。自明な早期 return（0 行・batch 0）でも `begin_driver_call` で検査する

### 3.5 OWASP 対応

| 項目 | 対応 |
|------|------|
| A03 インジェクション | NVRTC ソースは定数文字列のみ（`format!` を使わない）。利用者の入力を文字列として連結しない。スロット・ブロックテーブル・`seq_lens` は共通検証関数で FFI の前に検査する |
| A04 安全でない設計 | プール・作業バッファ・出力の要素数とバイト数を `checked_mul`・`isize::MAX`・カーネル引数型・grid 上限で検査する。`max_blocks` は必須（0 と無制限は拒否）。確保失敗は型付きエラー |
| A06 脆弱・古いコンポーネント | 依存を追加していない |
| A08 データ整合性 | §3.4 |
| 秘密情報 | 該当なし（ワークフロー・hooks・設定は変更していない） |

## 4. 残リスク・申し送り

| # | 内容 | 担当 |
|---|------|------|
| 1 | NVRTC コンパイル・GB10 実機での CPU 参照との一致（統一複合判定）が未確認 | #3113（網羅）・#3171（実測） |
| 2 | MSL のコンパイルと Mac 実機での一致が未確認 | #3113・#3172 |
| 3 | `s_bad` の同値書き込みの競合（§3.3） | 実機での挙動確認と、必要ならカーネル構造の見直し（security-auditor の判断） |
| 4 | security-auditor による独立監査 | PR レビュー |
| 5 | per-call の作業バッファ確保（Metal は非プール確保）。性能の最適化は縮約順序を変えうるので別判断（baseline 方式の対象になる） | Phase 8 以降 |
