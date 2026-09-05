# matmul VJP 転置ゼロコピー化（イシュー #1046）: スコープ判断・実測記入欄

## 1. 背景

親イシュー #1043（Phase 3）は「GEMM 入口の転置フラグ／stride 化を
`autodiff` の matmul VJP（`grad.rs::matmul_vjp`）へ波及させ、ホスト側の
転置コピーをなくす」ことを要求する。前提となる Metal 限定の GEMM 入口
転置フラグ（`layout::classify_2d`・`dispatch_strided_bias_act_prepared`）
は #1040（PR #1076）で導入済みで、同 PR の
`docs/backend-metal-transpose-collapse-design.md` §4 が
「`grad.rs` の `transpose2d` 排除・CPU/CUDA の同型 zero-repack 化」を
本イシューへ引き継いでいた。

## 2. 調査で判明した事実

- `grad.rs::transpose2d` は `Tensor::transpose(0, 1)` を呼ぶのみで、
  既に **zero-copy**（stride 入れ替えのみ）である。転置コピーが実際に
  発生していたのは `eval::matmul`（ホスト参照実装）内部の `dense_vec`
  （`Tensor::contiguous()` 経由でホスト転置コピーを強制する）だった
- `crates/backend-cpu/src/gemm_blis/pack.rs::pack_a`／`pack_b` は
  複数のマイクロカーネル variant（`gemm_blis_region`・
  `gemm_blis_parallel_with_blocks`・`gemm_blis_parallel_2d_with_blocks`
  等。`gemm_blis/mod.rs` 内 6 箇所以上の呼び出し）に直接埋め込まれて
  おり、行優先前提の添字（`k_total`/`n_total` 固定）を stride 一般化
  （`(rs, cs)` 化）するには全 variant・全 microkernel 経路への横断変更
  が必要で、影響範囲が単一のディスパッチ経路に閉じない

## 3. 決定: 本イシューはスコープを縮小する

### 3.1 実施した変更（本イシューの範囲）

1. `crates/autodiff/src/layout.rs`（新規）: `backend-metal/src/layout.rs`
   （#1040）が持つ `MatrixLayout`／`classify_2d`（純粋関数・FFI 非依存）
   を `autodiff` 側にもクレート内非公開（`mod layout;`。`pub` にしない）
   で複製した。初版（PR #1077 codex-review 前）では両クレートの共通
   モジュールとして `tensor-core::layout`（`pub mod layout`）へ集約して
   いたが、`fandhe-ai-tensor-core`（crates.io 公開クレート）の公開面へ
   内部レイアウト型を露出させてしまう問題が codex-review で指摘された
   （`#[doc(hidden)]` は semver 上の公開面を変えないため契約にできない。
   AGENTS.md「内部表現の公開 API への漏出は P1」）。そのため `tensor-core`
   からは `layout` モジュールを削除し、`backend-metal`（`collapse_leading_dims`／
   `required_span`／`TransposePattern` を含む元の実装のまま）・`autodiff`
   （`classify_2d`／`MatrixLayout` のみ。`collapse_leading_dims` 等は
   `autodiff` 側で未使用のため複製しない）それぞれのクレート内非公開
   モジュールへ分類ロジックを複製する構成へ差し戻した（両モジュールは
   `backend-metal::shaders::gemm.metal` の添字計算を共通契約とする双子
   モジュール。変更時は両方に反映する）。`gemm.rs`・`ops.rs`・既存テスト
   の参照パス（`crate::layout::…`）は変更していない
2. `crates/autodiff/src/eval.rs::matmul`: `dense_vec`（`contiguous()`
   強制）をやめ、`layout::classify_2d` + `Tensor::as_view_slice`
   （借用）で行優先／転置 view の両方を直接読み出す `matmul_operand`
   ヘルパーを新設。k ループの反復順・`f32::mul_add` 呼び出しは不変の
   ため、行優先 contiguous 入力（既存の主経路）では **bit 完全一致**を
   維持する
3. 診断カウンタ `eval::MATMUL_HOST_REPACK_COUNT`（`backend-metal::ops::
   RESIDENT_HOST_REPACK_COUNT` と同型の `thread_local` カウンタ）を
   追加し、`grad.rs::matmul_vjp_does_not_repack_transposed_operands`
   （`crates/autodiff/src/grad.rs`）が `matmul_vjp` 実行後もカウンタが
   増えないことを機械検証する

この変更により、`matmul_vjp`（`Op::MatMul`）・`Op::LinearResident` の
`d_weight`（`eval::matmul(xᵀ, g)`）の**ホスト参照実装経路**では、転置
オペランドに対するホスト側転置コピーが発生しなくなる（受け入れ条件 (a)
をこの経路について機械検証済み。(b) は既存の workspace テスト
green で確認済み・許容誤差は変更していない）。

### 3.2 スコープ外とした項目（別イシューでの対応が必要）

| 項目 | 理由 | 引き継ぎ先 |
|------|------|-----------|
| CPU BLIS packing の stride 一般化（`ops.rs::gemm`／`gemm_resident_lhs` 等を stride 対応にし `Op::MatMul` の VJP を `ops.gemm` 経由へ切替） | `pack_a`/`pack_b` が複数マイクロカーネル variant に埋め込まれ、影響範囲が単一ディスパッチ経路に閉じない。中途半端に `Op::MatMul` だけ `ops.gemm` へ繋ぎ替えると、CPU バックエンドでは転置コピーがホスト（`eval::matmul`）から `backend-cpu::gemm` の `contiguous()` へ単に移動するだけで受け入れ条件 (a) を満たさない | 別イシュー提案（ユーザー承認後に起票。`.claude/rules/out-of-scope-tracking.md`） |
| CUDA 本番 GEMM カーネルの lda／転置対応 | 本ラン環境（Linux x86_64・NVRTC 非搭載）では実機検証不能。`backend-cuda::ops.rs::gemm`／`gemm_resident_lhs` は `contiguous()` を維持（コード変更なし） | 別イシュー提案 |
| Metal `BackendOps::gemm` の NT/TN/TT strided 結線・`gemm_resident_rhs_nt` 新設 | 受け入れ条件 (a) は上記 3.1 のホスト参照実装経路で機械検証済みであり、Metal 側の追加結線は Mac 実機での検証が前提。今回のホスト側変更のみで PR の受け入れ条件を満たすため、リスクを増やす追加変更を見送った | 別イシュー提案（`docs/backend-metal-transpose-collapse-design.md` §4 の残項目として引き続き追跡） |
| `Op::LinearResident.d_input`（`ops.gemm_resident_lhs(w_dev, gᵀ) → transpose2d`）のデバイス側直接計算化 | `BackendOps` trait 拡張（`gemm_resident_rhs_nt` 等）を伴う公開面変更のため、実機検証が可能なセッションでの設計・実装が必要 | 別イシュー提案 |

## 4. 数値一致・許容誤差

- ガードレール閾値・テスト許容誤差は変更していない
  （`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの
  許容誤差を単独で緩和しない」）
- `Op::MatMul`／`Op::LinearResident.d_weight` は引き続き
  `eval::matmul`（ホスト参照実装）を通るため、CPU/CUDA/Metal いずれの
  バックエンドを使う学習ループでも既存の数値一致契約に影響しない
  （バックエンド固有の GEMM カーネル自体は変更していない）

### 4.1 追補（イシュー #1211）

上記 3.2 表 1 行目「CPU BLIS packing の stride 一般化」の縣念どおり、
イシュー #1211 で `Op::MatMul`／`Op::LinearAct`（`matmul_vjp` 経由）・
`Op::LinearResident.d_weight` は本番経路で `eval::matmul`（ホスト参照
実装）から `BackendOps::gemm`（CPU は BLIS 並列 GEMM・CUDA/Metal は
デバイス GEMM）へ切り替わった。`transpose2d`（zero-copy view）で作った
転置オペランドはそのまま `ops.gemm` へ渡すため、CPU 本番経路では転置
コピーが本ドキュメントが保証していたホスト側ゼロコピーから
`backend-cpu::ops::gemm` 内の `contiguous()` へ移動する（受け入れ条件
(a) はホスト参照実装〈`NaiveOps`／`TestOps` 経由の compat・テスト経路〉
に限定されたまま。本番 CPU 経路のゼロコピー化は #1213 のスコープ）。

CPU での前後比較実測（`backward`・`step_total` の中央値〈fresh・reuse とも
5 run。詳細は同 doc §4〉。fresh 8.71×・reuse 5.63× の速度改善）は
`docs/perf/train-backward-gemm-wiring.md` を参照。ガードレール閾値・テスト
許容誤差は本追補でも変更していない。

### 4.2 追補（イシュー #1213）

上記 §4.1・§3.2 表 1 行目「CPU BLIS packing の stride 一般化」で残って
いた CPU 本番経路（`backend-cpu::ops::gemm`／`gemm_resident_lhs`）の
転置再パックを、**一般 stride 化ではなく NT（`b` が転置格納）／TN
（`a` が転置格納）の 2 パターン限定**で解消した。判定条件は「転置
オペランドが dense な転置格納（`Tensor::strides() == [1, shape()[0]]`。
`transpose2d` を行優先連続テンソルへ適用した結果と同値）」であり、
`gemm_blis::pack::{pack_a_from_transposed, pack_b_from_transposed}`
（既存 `pack_b`／`pack_a` と役割を入れ替えた実装）が BLIS packing 側で
転置格納から直接 panel を構築する。両方転置（TT）・一般 stride
（`narrow` 後の転置等）は引き続き `contiguous()` フォールバックのまま
（§3.2 表 1 行目の「一般化」自体は不採用のまま）。

数値契約: NT/TN 入口が書く panel の内容は「`contiguous()` してから
既存 `pack_a`／`pack_b` で pack した panel」と同一バイト列になる設計
のため、計算結果は `gemm_blis_parallel`（NN 経路）と **bit 完全一致**
する（`crates/backend-cpu/src/gemm_blis/mod.rs` のクレート内テスト・
`crates/backend-cpu/tests/gemm_transposed_parity.rs` で検証）。tolerance
の新設・変更は行っていない。

CPU 実機実測・採否判断は `docs/perf/cpu-gemm-vjp-transposed-entry.md`
を参照。CUDA（#1214）・Metal（#1215）は引き続き未対応（`docs/matmul-
vjp-zero-copy-decision.md` §3.2 の該当行は変更しない）。

### 4.3 追補（イシュー #1214）

CUDA 本番 GEMM カーネル（§3.2 表 2 行目「CUDA 本番 GEMM カーネルの
lda／転置対応」）の転置再パックを、CPU 版（§4.2）と同じ **NT（`b` が
転置格納）／TN（`a` が転置格納）の 2 パターン限定**で解消した。ただし
CPU の BLIS packing 側吸収方式（`pack_a_from_transposed`／
`pack_b_from_transposed`）とは異なり、CUDA は**GPU 側 smem 転置カーネル
（`kernels_transpose::transpose_smem_source_f32(false)`。パディングのみ
変種。#601 で実装済み・`docs/perf/cuda-gemm-transpose-ab.md` §2 の
「未計測のまま本番導入しない」を経て本イシューが結線イシューとなる）→
既存 NN GEMM カーネル（`select_tiled_f32_kernel` が選ぶ classic／cp.async
パイプライン。#1137）**方式を採る。転置オペランドの元 storage（行優先
連続）をそのまま H2D 転送し、デバイス上で転置してから標準 GEMM カーネル
へ渡すため、既存カーネル・カーネル選択ロジック（`kernel_specs()`）には
一切手を入れない。

判定条件は CPU 版と同一（`rank() == 2 && strides() == [1, shape()[0]]`）
で、`crates/backend-cuda/src/ops.rs` に `dense_transposed_view`（CPU 版の
private 複製）を新設した。適用範囲は `gemm_fp32_strict_impl`（VJP・TF32
opt-in OFF 時の `gemm`・`gemm_bias_act` の `ComposedFallback`）と
`gemm_resident_lhs` の 2 箇所のみ。**TF32 opt-in ON 時の `gemm`
（`run_wmma_tf32`）・`gemm_bias_act` の融合経路・`gemm_resident_rhs` は
対象外**（意図的な限定。転置カーネル起動を挟むと epilogue 融合の利点が
薄れるため）。両方転置（TT）・一般 stride（`narrow` 後の転置等）・転置
カーネル自体が使用不能な環境（`CudaGemm::new` 時のコンパイル失敗。
fail-soft）は従来どおり `contiguous()` フォールバックへ倒す
（`ops.rs::GEMM_HOST_REPACK_COUNT` へ計上）。

数値契約: GEMM カーネルに渡るデバイス上のバイト列は「`contiguous()`
してから upload した場合」と同一（転置カーネルは純データ移動のみで
丸めを追加しない）ため、計算結果は既存 NN 経路と **bit 完全一致**する
契約（`crates/backend-cuda/tests/gemm_transposed_parity.rs` で検証。
CPU 参照実装との REQ-2 複合判定も併せて確認する）。tolerance の新設・
変更は行っていない。

CUDA（GB10）実機実測・採否判断は `docs/perf/cuda-gemm-vjp-transposed-
entry.md` を参照。Metal（#1215）は引き続き未対応。

### 4.4 追補（イシュー #1215）

Metal 本番経路（`backend-metal::ops::MetalBackendOps::gemm`）の転置
再パックを、CPU（§4.2）・CUDA（§4.3）と同じ **NT（`b` が転置格納）／
TN（`a` が転置格納）の 2 パターン限定**で解消した。ただし CPU の
BLIS packing 側吸収方式・CUDA の GPU 側転置カーネル方式のいずれとも
異なり、Metal は **既存の別カーネル入口へ切り替える**方式を採る:
`layout::classify_2d` で分類した `MatrixLayout`（#1040 で導入済み）を
`gemm::MetalGemm::dispatch_strided_bias_act_prepared`（`gemm_resident_
lhs` が確立済みの classic strided カーネル `gemm_tiled_bias_act`）へ
渡し、`MetalMemory::upload_view` で両オペランドを zero-copy アップ
ロードする。NN・TT・分類不能形状は従来どおり `contiguous()` +
`dispatch_auto`（`gemm_simdgroup_tiled`）の bit 同一経路のまま。適用
範囲は `MetalBackendOps::gemm` のみ（`gemm_bias_act` は融合カーネル
経路〈`GemmBiasActRoute::Fused`〉が対象外。`ComposedFallback` 側は
内部で `self.gemm` を呼ぶため NT/TN 結線を自動的に継承する）・
`gemm_resident_rhs`〈#1040 で既に転置対応済み〉・
`gemm_fp32_strict_into`〈既定 `Unsupported` のまま。#1212 引き継ぎ〉は
対象外）。

**数値契約が CPU／CUDA と異なる**: `dispatch_strided_bias_act_prepared`
（classic strided カーネル）は従来の `dispatch_auto`（動的タイル選択）
とは別カーネルであり、アキュムレータの蓄積順序が異なりうるため、
NT/TN 経路は `contiguous()` 経路と bit 一致する保証がない。受け入れ
判定は **REQ-2 統一複合判定**（`gemm_resident_lhs`〈#1040〉と同じ契約。
`crates/backend-metal/tests/gemm_transposed_parity.rs` で検証）とし、
`assert_eq!` によるビット一致は要求しない。NN・TT・分類不能形状の
従来経路自体は不変のため bit 同一のまま。tolerance の新設・変更は
行っていない。

M4 Max 実機実測（補助 A/B・train phases フル A/B）・採否判断（ADOPT）は
`docs/perf/metal-gemm-vjp-transposed-entry.md` を参照。`Op::
LinearResident.d_input` のデバイス直接計算化（§3.2 表 4 行目）は
引き続き未対応（`BackendOps` trait 拡張を伴う公開 API 変更のため別途
ユーザー承認が必要）。

## 5. 実機実測（未実施）

本ランは Linux x86_64（NVRTC 非搭載・Metal 実機なし）のため、以下は
未実施のまま記入欄を残す。

```sh
# CPU 側「1 学習 step 中の転置コピー削減」の効果測定（before/after）
cargo bench -p bench-harness -- <該当ベンチ名>
```

（実測値は未記入。後続セッションでの実行後にこの節を更新する）
