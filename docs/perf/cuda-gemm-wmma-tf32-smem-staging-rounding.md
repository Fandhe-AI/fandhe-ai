# CUDA GEMM: TF32 丸めの smem ステージング時 1 回化（イシュー #800）

## 0. 背景・目的

TF32 WMMA GEMM 経路（`crates/backend-cuda/src/kernels_wmma_opt.rs`）は
`wmma::__float_to_tf32` による明示丸めを、従来は **K ループ（kstep）内・
fragment ロード直後のレジスタ上**で毎回発行していた。

- TF32 opt カーネル（`WMMA_TF32_F32_OPT_BODY`）: fragment 変換ループ
- TF32 opt-staged カーネル（`WMMA_TF32_F32_STAGED_BODY`。本番経路の最優先。
  `gemm.rs::run_wmma_tf32` 3 段選択で cp.async 整列形状が選ぶ）:
  `LDWM_A_FRAG`/`LDWM_B_FRAG` マクロ内

この方式は同一 smem 要素が複数 warp・複数 kstep から重複変換され、
変換命令列が `mma_sync` 発行と同じ命令ストリームで競合する。本イシューは
CUTLASS 同様に **global→smem ステージング段で要素あたり 1 回だけ丸め、
smem に tf32 丸め済み値を置く**構成へ移し、kstep ループ内の変換を消して
MMA 発行帯の命令数を削減することを目的とする。

## 1. 変更内容

### 1.1 `WMMA_TF32_F32_OPT_BODY`

- guarded load の smem 格納 4 箇所（プロローグ A/B・プリフェッチ A/B）を
  `wmma::__float_to_tf32(a[gr * DIM_K + gc])` の形へ変更（ガード式・
  ゼロ充填値 `0.0f` は不変。`0.0f` は tf32 で正確に表現できるため変換
  不要）。
- kstep ループ内の fragment 変換ループ（`for (e...) a_frag[fi].x[e] = ...`）
  を削除。丸め出現回数はプロローグ・プリフェッチの guarded load
  4 箇所のみ（ロックテスト
  `wmma_tf32_opt_source_rounds_tf32_once_at_smem_staging` で固定）。

### 1.2 `WMMA_TF32_F32_STAGED_BODY`

cp.async は生バイトコピーのため格納「中」の変換は不可能なため、**タイル
到着後・fragment ロード前の in-smem 変換パス**として実現した。

- `LOAD_A_STAGE_GROUP`/`LOAD_B_STAGE_GROUP` と同一の idx 算術（tid・
  グループ g の純関数）で smem チャンク（f32 4 要素）を読み・
  `__float_to_tf32` し・書き戻す `CONVERT_A_STAGE_GROUP(stage, g)`/
  `CONVERT_B_STAGE_GROUP(stage, g)` マクロを LOAD マクロ群直後に追加した。
- t ループ先頭の `cp.async.wait_group (STAGES-2)` と既存 `__syncthreads()`
  の**間**に、全グループ g について `CONVERT_*_STAGE_GROUP(compute_stage, g)`
  を呼ぶ変換パスを挿入した。
- `LDWM_A_FRAG`/`LDWM_B_FRAG` マクロから変換 for ループを削除し
  `load_matrix_sync` のみを残した。
- ループ末尾の `#undef` 群へ `CONVERT_*_STAGE_GROUP` の `#undef` を追加。

丸め出現回数は `CONVERT_A_STAGE_GROUP`/`CONVERT_B_STAGE_GROUP` マクロ定義の
各 1 箇所（計 2 箇所）のみ（ロックテスト
`wmma_tf32_staged_source_rounds_tf32_once_at_smem_staging` で固定）。

### 1.3 正しさの論証

1. **自スレッドの読み取り安全性**: 変換パスの chunk 割り当ては
   `LOAD_A_STAGE_GROUP`/`LOAD_B_STAGE_GROUP` と同一の tid 純関数のため、
   各スレッドが変換で読む要素は必ず「自分自身が直前に cp.async で書き
   込んだ要素」である。`cp.async.wait_group` は当該スレッド自身が発行
   した cp.async の完了を保証する（PTX 契約）ため、これのみで安全に
   読める（他スレッド分の smem には一切触れない）。
2. **他スレッドへの公開**: 変換結果を全 warp の `load_matrix_sync` へ
   公開するのは wait_group 単体ではなく、**変換パス呼び出し直後に
   既存のまま保持している `__syncthreads()`** である（変換パスは
   その `__syncthreads()` より前・`cp.async.wait_group` の直後に置く。
   順序を入れ替えると本論証は成立しない）。旧実装でもこの
   `__syncthreads()` は cp.async 到着分の公開を担っていたため、公開対象
   が「生データ」から「変換済みデータ」に変わるのみで、同期構造
   （バリア個数・配置）自体は変更していない。
3. **stage バッファ再利用時の WAR 安全性**: 同一物理 stage バッファは
   `STAGES` イテレーションごとに再利用されるが、直前の利用
   （`load_matrix_sync`/`mma_sync` によるフラグメント読み出し）は
   t ループ末尾の無条件 `__syncthreads()` で必ず完了してから次の
   cp.async 上書き・変換パスが走る。よって「読み出し中の stage への
   書き込み」（write-after-read）は発生しない。二重変換・変換漏れも
   生じない（`__float_to_tf32` は冪等なので万一の重複も数値影響なし）。
4. src_size=0 でゼロ充填された chunk の変換は `0.0f`→`0.0f` で無害
   （ガード分岐不要）。

### 1.4 数値契約上の同値性

`__float_to_tf32` は要素単位の変換であり、「smem 格納時に 1 回」も
「fragment ロード直後」も各要素が `mma_sync` に渡る前に**ちょうど 1 回**
丸められる点は同一。結果は bit 一致が期待され、統一複合判定（相対誤差
1e-3 未満 または 絶対誤差 1e-5 未満）・parity ベースライン
（`docs/perf/cuda-parity-baseline.md`）は無変更のまま通る想定である。
許容誤差・閾値は変更していない。

## 2. ローカル検証（実施済み）

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo test --all-features -p backend-cuda
bash scripts/run-verification-gates.sh all
```

いずれも green（本コミット時点。NVRTC 実コンパイルはローカル環境に CUDA
toolkit・実機がないため未検証。下記 3 節参照）。

追加したソースロックテスト:

- `wmma_tf32_opt_source_rounds_tf32_once_at_smem_staging`
  （`crates/backend-cuda/src/kernels_wmma_opt.rs`）: OPT 版の丸め出現回数
  4（プロローグ A/B・プリフェッチ A/B）・kstep ループ以降に丸めが
  存在しないことをロック。
- `wmma_tf32_staged_source_rounds_tf32_once_at_smem_staging`（同ファイル。
  旧テスト `..._applies_tf32_conversion_to_every_fragment_load` の差し替え）:
  staged 版の丸め出現回数 2（`CONVERT_A_STAGE_GROUP`/
  `CONVERT_B_STAGE_GROUP` マクロ定義）・`load_matrix_sync` 出現回数 2
  （不変）・kstep ループ以降に丸めが存在しないことを、無印ソースと
  swizzle 変種（`wmma_tf32_f32_staged_source_with_swizzle(8)`）の両方で
  ロック（「変種は同一 BODY テンプレートを共有するため自動追従する」
  という設計前提を実測で確認する）。

## 3. 実機検証（未実施・要実機実行）

本作業環境には `docs/real-hardware-verification-env.local.md`
（DGX Spark GB10 実機接続情報）が存在しないため、以下は**未実測**である。
安全側の方針（`docs/real-hardware-verification-env.md`・#742 の先例）に
従い、実測していない数値は記入せず空欄のまま記録する。

- [ ] NVRTC 実コンパイル確認（`wmma_tf32_staged_available()`/
      `wmma_tf32_opt_available()` が true であること）
- [ ] 数値一致・parity 非後退（統一複合判定）:
      `cargo test -p backend-cuda --release -- --ignored`
      （`gemm_wmma_tf32_staged`/`gemm_wmma_tf32_opt`/
      `cpu_cuda_wmma_parity`/`parity_nonregression`/
      `tensor_core_real_device` 等）
- [ ] `cargo test -p backend-cuda --release --features internal-diagnostics
      --test specialized_mma_parity -- --ignored`
- [ ] ベンチ A/B（`cargo run -p backend-cuda --example cuda_floor_bench
      --release`）: main チェックアウトと本変更ブランチそれぞれで
      512/1024/2048/4096 の中央値（5 回計測）を突合

| M=N=K | main（TFLOPS） | 本変更（TFLOPS） | 差分 |
|------:|---------------:|-----------------:|-----:|
| 512   | 未実測 | 未実測 | - |
| 1024  | 未実測 | 未実測 | - |
| 2048  | 未実測 | 未実測 | - |
| 4096  | 未実測 | 未実測 | - |

## 4. 採否判断

**暫定採用（実機未検証のためコミット時点の判断。実機 A/B が劣化を示した
場合は 3 章のフォールバック規定に従い再判断する）**。

- ソースロックテスト・数値契約の同値性論証により、実装は計画どおり完了
  している。
- 実機ベンチ A/B が未実施のため、性能改善効果自体は本ドキュメント時点
  では確認できていない。実機接続可能な環境で 3 章のチェックリストを
  実行し、本ドキュメントの表・チェックボックスを更新すること。
- 劣化が確認された場合: staged 側の変更は取り込まず（LDWM 内変換の
  ままへ戻す）、OPT 側のみ採用または全面不採用とする
  （`backend-metal-*-decision.md` 系の先例に倣い、本ドキュメントへ
  不採用判断を追記する）。

## 5. スコープ外（別イシュー化候補）

- `kernels.rs::WMMA_TF32_F32`（基本版。opt/staged コンパイル失敗時の
  最終フォールバックのみで本番・ベンチ経路に乗らない）の丸め位置は
  変更していない。リスクに対して性能効果が観測不能なため。必要なら
  別イシュー化をユーザーへ提案する
  （`.claude/rules/out-of-scope-tracking.md`。無承認では起票しない）。
- f16 経路（`WMMA_F16_OPT_BODY`・`kernels_mma.rs`）は TF32 変換を持たない
  ため対象外。
- タイル構成・段数・許容誤差・ガードレール閾値は変更していない。
