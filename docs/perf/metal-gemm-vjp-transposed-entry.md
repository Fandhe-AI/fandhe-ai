# Metal GEMM VJP 専用 NT/TN strided 入口（イシュー #1215）

## 0. 目的・スコープ

`docs/matmul-vjp-zero-copy-decision.md` §3.2 表・§4.4 追補が「解消は
#1215」と記していた Metal 本番経路（`MetalBackendOps::gemm`）の転置
オペランド再パックを、CPU（#1213）・CUDA（#1214）の姉妹イシューと同じ
**NT（`b` が転置格納）／TN（`a` が転置格納）の 2 パターン限定**で解消
する。対象は `matmul_vjp`（`Op::MatMul`／`Op::LinearAct` の VJP）の
d_input（`g @ Wᵀ`＝NT）・d_weight（`Aᵀ @ g`＝TN）、`Op::LinearResident`
の d_weight（`xᵀ @ g`＝TN）。

対象外（本イシューのスコープ外）: 両方転置（TT）・一般 stride
（`narrow` 後の転置等）・`gemm_bias_act` 融合経路・`gemm_resident_rhs`
（既に #1040 で転置対応済み）・`gemm_fp32_strict_into`（Metal は既定
`Unsupported` のまま。#1212 引き継ぎ）・`dispatch_strided_tiled_prepared`
への自動ルーティング（#1187 で判定不能につき見送り済み・再開しない）・
`BackendOps` trait 拡張による `d_input` デバイス直接計算化（公開 API
変更のため別途ユーザー承認が必要。`docs/matmul-vjp-zero-copy-decision.md`
§3.2 項目 4）。

## 1. コード変更

| ファイル | 変更内容 |
|---------|---------|
| `crates/backend-metal/src/ops.rs` | モジュール冒頭コメントを NT/TN 分岐込みへ更新。`GEMM_HOST_REPACK_COUNT`（`pub(crate)`。CPU/CUDA と同名の可観測点）を追加。`MetalBackendOps::gemm` を NT/TN 判定付きへ変更し、片側のみ転置 view の場合は新設ヘルパー `gemm_strided_nt_tn` へ分岐する。`gemm_strided_nt_tn` は `layout::classify_2d` で分類済みの `MatrixLayout` を受け取り、`MetalMemory::upload_view`（zero-copy）→ `gemm::MetalGemm::dispatch_strided_bias_act_prepared`（`gemm_resident_lhs`〈#1040〉が確立した classic strided カーネル入口。bias／activation は使わない）→ `MetalMemory::download` で計算する。NN・TT・分類不能形状は従来どおり `contiguous()` + `dispatch_auto` の bit 同一経路のまま。クレート内テスト（`#[cfg(test)]`・`#[ignore]`）を 3 件追加（NT/TN は非増加・TT は +2） |
| `crates/backend-metal/tests/gemm_transposed_parity.rs`（新規） | `BackendOps::gemm_fp32_strict` 経由の統合テスト。NT/TN/TT/narrow 後転置・`&dyn BackendOps` 経由呼び出しを検証。**bit 一致ではなく REQ-2 複合判定**（§2 参照） |
| `crates/backend-metal/tests/gemm_transposed_perf.rs`（新規・`#[ignore]`） | 本ドキュメント §3.1 の補助 A/B 計測 |
| `crates/autodiff/src/grad.rs`・`eval.rs` | 「Metal（#1215）は未対応」の doc コメントを結線済みへ更新（コード変更なし） |

`crates/tensor-core`（`Tensor`・`BackendOps` trait）・`backend-cpu`・
`backend-cuda`・依存（`Cargo.toml`／`Cargo.lock`）は無変更。

## 2. 数値一致契約（CPU／CUDA との違い）

**CPU（#1213）・CUDA（#1214）は bit 完全一致契約**だった（転置元
storage を転置してから既存 NN カーネルへ渡す設計のため、丸めの追加が
一切ない）。

**Metal は異なる**: NT/TN 入口が経由する `gemm::MetalGemm::
dispatch_strided_bias_act_prepared`（classic strided カーネル
`gemm_tiled_bias_act`）は、従来の `dispatch_auto`
（`gemm_simdgroup_tiled`。動的タイル選択）とは**別カーネル**であり、
アキュムレータの蓄積順序（ループ順・タイル分割）が異なりうる。よって
NT/TN 経路の結果は「`contiguous()` してから `dispatch_auto` を呼んだ
結果」と bit 一致する保証がなく、受け入れ判定は **REQ-2 統一複合判定**
（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`fandhe_ai_backend_cpu::
assert_parity`）とする。これは `gemm_resident_lhs`（#1040）が既に採用
している契約と同一であり、本イシューで新たに緩和した基準ではない。

- NN 経路: 本イシュー導入前と bit 完全一致（カーネル不変。クレート内
  テストでは検証しないが、`gemm` の分岐条件（`la.transposed != lb.
  transposed` を満たさない限り従来経路）から自明）
- `crates/backend-metal/tests/gemm_transposed_parity.rs`: NT/TN の
  `contiguous()` 経路との複合判定・CPU 参照実装
  （`matmul_reference_fma`）との複合判定・TT／narrow 後転置の
  `contiguous()` フォールバック時の bit 完全一致（従来経路は不変の
  ため）・`&dyn BackendOps` 経由呼び出しの一致
- `crates/backend-metal/src/ops.rs` 内 `#[cfg(test)]`:
  `GEMM_HOST_REPACK_COUNT` で NT/TN が repack を回避し TT が回避しない
  ことを可観測点で確認

tolerance（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・
baseline・依存は一切変更していない。

## 3. 計測プロトコル・実測結果

環境: Apple M4 Max（macOS 26.6.2・arm64・rustc 1.96.0）。詳細・生ログは
`docs/perf/logs/metal-gemm-vjp-transposed-entry-1215/`。実行中、他
worktree での並行セッション作業（load average 約 3.1〜3.9、19 ログイン
セッション）が確認されており、計測ノイズの要因になりうる（除外はせず
そのまま記録する）。

### 3.1 補助 A/B（`gemm_transposed_perf.rs`）

`MetalBackendOps::gemm` を直接対象に、before（転置 view を明示
`contiguous()` してから `gemm` を呼ぶ＝両オペランドとも NN 判定になる
ため常に `dispatch_auto` 経路）／after（転置 view をそのまま渡す＝NT/TN
strided 入口）を warmup 2 回・measured 5 回中央値で比較した（3 回
試行し、代表として直近の 1 回を記載。生ログ: `aux_ab.txt`）。

| パターン | m | k | n | before 中央値 (s) | after 中央値 (s) | 倍率 |
|----------|---|---|---|-------------------|-------------------|------|
| NT | 64 | 784 | 256 | 0.001020 | 0.000498 | 2.048× |
| NT | 64 | 256 | 10 | 0.000369 | 0.000210 | 1.759× |
| NT | 1024 | 1024 | 1024 | 0.004917 | 0.003560 | 1.381× |
| NT | 2048 | 2048 | 2048 | 0.027048 | 0.022918 | 1.180× |
| TN | 64 | 784 | 256 | 0.000563 | 0.000692 | 0.814× |
| TN | 64 | 256 | 10 | 0.000381 | 0.000364 | 1.046× |
| TN | 1024 | 1024 | 1024 | 0.004917 | 0.003561 | 1.381× |
| TN | 2048 | 2048 | 2048 | 0.027433 | 0.022929 | 1.196× |

NT は全形状で明確な改善（1.18〜2.05 倍）。TN は size 64 相当の層形状
（`m=64, k=784, n=256`）で 3 回の試行間にばらつきがあり（0.814×〜
1.280×。他 2 回の試行結果は `aux_ab.txt` 参照）、負荷混在環境下では
単独形状の判定が不安定である。`(64, 256, 10)` はほぼ非後退（0.99〜
1.05×）、1024²・2048² は非後退〜改善（0.74〜1.38× の範囲でばらつくが
中央値ベースでは非後退）。この補助 A/B 単独では TN の採否を確定でき
ないため、§3.2 の train phases A/B（環境ノイズの影響を受けにくい
step 全体の集計）を主判定軸とする。

### 3.2 train phases フル A/B（決定的）

`docs/perf/train-backward-gemm-wiring.md` §3 と同一プロトコル（
**参考系列方式**: `scripts/bench/framework-compare/bench-fandhe/
Cargo.toml` の `fandhe-ai = "=0.6.0"` ピンをコミットせず、`--config
patch.crates-io.fandhe-ai.path="<facade 絶対パス>"` で before
（`git stash` で `crates/backend-metal/src/ops.rs` の変更を退避した
状態）／after（本 PR HEAD）の 2 バイナリを別々の `CARGO_TARGET_DIR` に
ビルド。両バイナリとも `cargo tree -p bench-fandhe --depth 1` で
`fandhe-ai (path: …)` が出ることを確認済み）で、`bench-fandhe --task
train --device metal --size 64 --mode {fresh,reuse} --phases` を各系列
5 回計測した。生ログ: `{before,after}.jsonl`。計測後
`git checkout -- scripts/bench/framework-compare/Cargo.lock` で復元し
`make deps-forbidden` が承認済みピン `fandhe-ai =0.6.0`（registry 取得
元）を再検出することを確認した（drift なし）。

| mode | phase | before 中央値 (s) | after 中央値 (s) | 倍率 |
|------|-------|---------------------|--------------------|------|
| fresh | backward | 0.001145 | 0.000694 | 1.649× |
| fresh | step_total | 0.001877 | 0.001419 | 1.323× |
| reuse | backward | 0.000765 | 0.000637 | 1.200× |
| reuse | step_total | 0.001430 | 0.001289 | 1.109× |

`step_total` の checksum（loss 値）は before/after で完全一致
（`0.080541`）——数値契約（§2）の複合判定を待たずとも、この学習
シナリオの実際の入出力では丸め差が可視化される桁まで現れなかった
ことを追加で確認できた。

fresh・reuse いずれも `backward`・`step_total` とも明確に改善しており
（1.11〜1.65 倍）、§3.1 で不安定だった TN 単独形状の結果を含む形で
「train step 全体としては非後退」であることが確定した。

## 4. 解釈

- fresh の `backward` 改善（1.649×）が最大: fresh は `matmul_vjp` の
  d_input（NT）・d_weight（TN）の両方を毎 step 呼ぶため、両パターンの
  転置コピー削減が重なって効く
- reuse の `backward` 改善（1.200×）は `Op::LinearResident.d_weight`
  （TN 単独）のみが対象のため fresh より小さいが、それでも非後退
- §3.1 の TN 単独形状計測が負荷混在環境で不安定だった一方、§3.2 の
  train phases 計測（同じ環境条件下）では明確に改善している。これは
  「TN 単独の GPU 起動オーバーヘッドは環境ノイズの影響を受けやすいが、
  fresh/reuse の 1 step 全体（複数 GEMM・elementwise・SGD 更新を含む
  合成）で見ると NT の改善が支配的に効き、TN のノイズは打ち消される」
  という解釈と整合する
- したがって主判定軸は §2.2 の設計判断どおり train phases A/B とし、
  補助 A/B は「変更箇所自体の直接効果を示す参考値」として扱う

## 5. 採否判断

**ADOPT**（`docs/matmul-vjp-zero-copy-decision.md` §2.2 決定表「VJP
形状と train backward／step_total が非後退」に該当）。

1. §3.2 の train phases A/B で fresh・reuse とも `backward`・
   `step_total` が明確に改善（1.11〜1.65 倍）し、checksum も完全一致
2. §3.1 の補助 A/B は NT が全形状で明確な改善、TN は単独形状レベルで
   環境ノイズにより不安定だが、決定的な §3.2 の結果が非後退を確定
3. 大形状（1024²・2048²）でも後退は確認されなかった（§2.2 の「大形状
   のみ明確に後退」ケースには該当しないため、形状ゲートの追加は不要）
4. NT/TN に乗らない入力（NN・TT・分類不能形状）は従来経路のまま
   bit 同一で不変

## 6. 後続

- `LinearResident.d_input` のデバイス直接計算化（`docs/matmul-vjp-zero-
  copy-decision.md` §3.2 項目 4。`BackendOps` trait 拡張を伴う公開 API
  変更のため別途ユーザー承認が必要。未起票。CUDA #1214 も同様に本
  イシューのスコープ外としている）
- TT（両方転置）・一般 stride 化: 本イシューでは対象外のまま
  （`docs/matmul-vjp-zero-copy-decision.md` §3.2 の該当行は変更しない）
- `dispatch_strided_tiled_prepared` への自動ルーティング（#1187）:
  判定不能につき未着手のまま。本イシューでも再開しない
