# 勾配の長軸縮約の Metal 実装形の決定記録（イシュー #1566）

> **状態: 確定（2026-09-12）**。Metal の勾配長軸縮約（dw 行方向蓄積・bias 勾配の
> 行方向縮約）は **IEEE 754 binary64 逐次加算の 64bit 整数ソフトウェア
> エミュレーション**として実装し、ホスト `f64` 逐次和（index 順）を 1 回
> `f32` へ downcast した値と**bit 完全一致**する（NaN のみ quiet NaN へ正規化
> しクラス一致で比較）。tolerance・baseline・REQ-2 判定は不変。事前判定可能な
> 入力上限・非有限値の特例契約は不要（bit 一致のため。詳細は下記「2. 確定
> 契約」）。

## 1. 経緯

PR #1666（本ドキュメント）は当初、Metal が `double` 型非対応であることを前提に
f32 のみの Neumaier 改良版 Kahan 補償和 + 2 の冪 scale で「`f64` 相当」を狙う
実装形を採用し、その数値契約を明文化する目的で作成された。しかし `[2^48, 2^24,
1, -2^48, -2^24]` のような相殺列でホスト `f64` 逐次和と一致しない問題が
codex-review で繰り返し指摘され（判定方式の「除外」表現の弱体化・`O` 記法の
計算不能性・参照値の混同・入力上限未定義・非有限値未対応・2 の冪 scale の
underflow・downcast 誤差上界の誤り・二次項係数の未証明性、計 9 件の P1／P2）、
その都度、事前判定可能な Tier A（無条件誤差上界）／Tier B（REQ-2 適用可否の
事前判定述語）の 2 層契約として改訂を重ねた（係数 `C` は当初 `C = 8` の暫定値
→ 理論導出により `C = 3` → downcast 上界の是正で `C = 4` へ変遷）。

並行して別セッション（PR #1659）が、判定契約を緩める方向ではなく**実装側を
ホストと同一の演算列にする**方針へ転換し、Metal カーネルを `f64` 加算そのものを
64bit 整数で忠実に再現するソフトウェアエミュレーション（`bias_f64_widen`／
`bias_f64_add`／`bias_f64_narrow`。ホスト側逐語モデル
`crates/backend-metal/src/soft_f64.rs`）へ置き換えた（commit `cf6c9b51`／
`754b36e9`）。この実装によりホスト参照実装と bit 完全一致する契約が成立した
ため、本ドキュメントが定義していた Tier A/B の数値許容契約は不要になった
（実装記録の正本は `docs/backend-metal-command-batching-design.md` §10.10〜
§10.14）。本ドキュメントはこの結論に合わせて全面的に縮小する。

## 2. 確定契約

- **実装**: `gemm.metal::gemm_bias_grad_reduce_f32` の `m ≥ 2` 経路は IEEE 754
  binary64 の逐次加算（round-to-nearest-even）を 64bit 整数（`ulong`／`long`）
  演算で忠実に再現する（`bias_f64_widen`／`bias_f64_add`／`bias_f64_narrow`／
  `bias_clz64`）
- **契約**: 演算列はホスト参照実装（`crates/autodiff/src/eval.rs::
  reduce_bias_grad_rows`／`crates/backend-metal/src/layout.rs::
  reduce_bias_grad_rows_host`。`acc: f64 = 0.0` から index 順に加算し最後に
  1 回 `acc as f32`）と同一であり、結果は**bit 完全一致**する（NaN のみ
  payload がハードウェア依存のため quiet NaN へ正規化し、クラス一致で比較
  する）
- **ホスト側逐語モデル**: `crates/backend-metal/src/soft_f64.rs`
  （`widen_f32_bits`／`add_f64_bits`／`narrow_f64_bits`）が Rust の `f64` 実
  演算と `to_bits` で一致することをユニットテストで検証する
- **tolerance・baseline・REQ-2 の適用範囲はいずれも不変**（`RELATIVE_
  TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD` は無変更）。weight 勾配
  （`gemm_fp32_strict_into`）と同じ bit 一致契約へ揃う
- 「3〜4. C の確定」節が定義していた入力上限（`n < 2^24` の `InvalidArgument`
  拒否）・非有限値クラス一致規則は、bit 完全一致契約の下では不要（実装が
  ホストと同一演算列であるため、有効範囲・非有限値いずれも自然に一致する）

## 3. 検討したが不採用の案（履歴）

- **Neumaier 改良版 Kahan 補償和 + 2 の冪 scale**（f32 のみで「`f64` 相当」を
  狙う案）: 相殺列でホスト `f64` 逐次和と一致しない問題が解消できなかった
- **Tier A/B の 2 層契約**（無条件誤差上界 `(C + n·ε32)·ε32·Σ|x_i|` ＋ REQ-2
  適用可否の事前判定述語）: 判定方式としては事前判定可能な形へ改善を重ねたが
  （`C = 8 → 3 → 4`）、根本解決（bit 一致するカーネル実装への置換）が可能
  だったため不採用となった
- **入力上限 `n < 2^24` の明示拒否・非有限値クラス一致規則**: Tier A/B 前提の
  特例であり、bit 完全一致契約下では不要

## 4. spec 起票要否

本ドキュメントの縮小に伴い、Tier A/B 判定パターンの spec REQ-2 への一般化提案は
行わない（対象契約が存在しなくなったため）。

## 5. 関連

- PR #1666（本ドキュメント）・PR #1659（イシュー #1566 実装 PR。binary64
  ソフトウェアエミュレーションへの置換元。commit `cf6c9b51`／`754b36e9`）・
  イシュー #1566（親）
- `docs/backend-metal-command-batching-design.md` §10.10〜§10.14（実装記録の
  正本。方針転換の詳細経緯）
- 本ドキュメントが応答した codex-review スレッド（方針変更により全件が
  「実装置換により対象契約が消滅」として解消済み。各スレッドへ個別返信
  済み）: `PRRT_kwDOTuUCJc6hvjYM`・`PRRT_kwDOTuUCJc6hvnzg`・
  `PRRT_kwDOTuUCJc6hvnzi`・`PRRT_kwDOTuUCJc6hvrGx`・`PRRT_kwDOTuUCJc6hvrGy`・
  `PRRT_kwDOTuUCJc6hvug6`・`PRRT_kwDOTuUCJc6hvxnm`・`PRRT_kwDOTuUCJc6hv2ss`・
  `PRRT_kwDOTuUCJc6hv2sw`・`PRRT_kwDOTuUCJc6hv5IK`
- `AGENTS.md`「数値契約の統一（P1）」・`.claude/rules/coding-rust.md`「バック
  エンド構成（REQ-2）」節（本ドキュメントを正本として参照していた側。今回の
  改訂で `docs/backend-metal-command-batching-design.md` §10.10〜§10.14 を
  参照する形へ差し替え済み）
