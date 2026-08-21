# CUDA GEMM: TF32 丸めの smem ステージング時 1 回化（イシュー #800・wmma 側は #851 で revert 済み）

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

## 3. 実機検証（#851 で bisect A/B 実施・回帰確定）

本ドキュメント作成時点（コミット `2c0f9ec`・#816）では実機未検証だった。
その後 GB10 実機での bisect A/B（親 `9bbac56` vs `2c0f9ec`。5 回計測
中央値）で、本番最優先経路（TF32 opt-staged）が 512〜2048 において明確な
性能回帰を示すことが単独確定した。

| M=N=K | `9bbac56`（丸め毎回・TFLOPS） | `2c0f9ec`（丸め 1 回化・TFLOPS） | 差分 |
|------:|------------------------------:|----------------------------------:|-----:|
| 512   | 8.2685 | 6.6000 | 約 −20.2% |
| 1024  | 12.7316 | 10.6521 | 約 −16.3% |
| 2048  | 14.4050 | 11.9808 | 約 −16.8% |
| 4096  | 9.0358 | 8.9452 | 約 −1.0%（ほぼ中立） |

小〜中サイズ（512〜2048）では、staged 経路の独立変換パス（`cp.async` が
生バイトコピーのため、丸めをロード段へ融合できず smem を再読込・再丸め・
再書込する専用パスとして追加されたこと・t ループ先頭での全グループ
走査によるオーバーヘッド）が支配的になり、丸め回数削減による MMA 発行帯
の命令数削減効果を上回って悪化したとみられる。4096 は元々データ再利用
崩壊（`docs/perf/cuda-gemm-bottleneck-diagnosis.md`）が支配的なためほぼ
中立。

- [x] NVRTC 実コンパイル確認（`wmma_tf32_staged_available()`/
      `wmma_tf32_opt_available()` が true であること）— bisect 実行前提
      として確認済み
- [x] 数値一致・parity 非後退（統一複合判定）: bisect 対象コミット双方で
      parity 系テストが green であることを確認したうえで A/B を実施
- [x] ベンチ A/B（`gemm_wmma_tf32_swizzle_bench` の base 系列。上表）

revert 後（本ドキュメント §4 の判断適用後）の実機再計測は
イシュー #851 の PR 本文・コミットに実測ログを記録する
（`9bbac56` の値への復帰・4096 の非劣化を確認する運用。本ドキュメント
自体は不採用判断の記録に主眼を置き、逐次の再計測ログまでは転記しない）。

## 4. 採否判断

**不採用（#851 で revert）**。

- #816（`2c0f9ec`）は実機 A/B 未実施のままマージされ、上記のとおり
  512〜2048 で約 −16〜−20% の性能回帰を招いた。
- `kernels_wmma_opt.rs` を親コミット `9bbac56`（#816 直前）の状態へ
  全面復元し、TF32 opt-staged・TF32 opt 両カーネルの丸め位置を
  「kstep ループ内・fragment ロード直後に毎回丸める」旧方式へ戻した
  （§1.1・§1.2 で説明した変更・ソースロックテストとも revert 対象）。
  OPT 側も実機未検証のまま temporary に残す理由がないため、3 章末尾の
  フォールバック規定（「劣化時は staged 側を戻し、OPT 側のみ採用
  **または全面不採用**」）に従い全面不採用とした。
- 丸め 1 回化そのものの再設計（cp.async のロード段へ丸めを融合する形。
  cp.async が生バイトコピーである以上ロード段融合は構造的に不可能で
  あり、別のステージング方式が必要）は本イシューでは追わない。実機
  A/B で非劣化を確認できる形の再設計が必要になった場合は改めて別
  イシューで扱う。
- `kernels_mma_tf32.rs`（mma 経路。#800 の設計を独自に内蔵）は bisect
  対象外であり、この経路で同型の回帰が確認されたわけではないため
  本設計（丸め 1 回化）を維持する。同経路の変換パスにも同種の
  オーバーヘッドが存在しうる点は所見として記録するに留め、無承認では
  Issue 化しない（`out-of-scope-tracking.md`）。

## 5. スコープ外（別イシュー化候補）

- `kernels.rs::WMMA_TF32_F32`（基本版。opt/staged コンパイル失敗時の
  最終フォールバックのみで本番・ベンチ経路に乗らない）の丸め位置は
  変更していない。リスクに対して性能効果が観測不能なため。必要なら
  別イシュー化をユーザーへ提案する
  （`.claude/rules/out-of-scope-tracking.md`。無承認では起票しない）。
- f16 経路（`WMMA_F16_OPT_BODY`・`kernels_mma.rs`）は TF32 変換を持たない
  ため対象外。
- タイル構成・段数・許容誤差・ガードレール閾値は変更していない。
