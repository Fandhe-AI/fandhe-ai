# 勾配の長軸縮約の Metal 実装形の決定記録（イシュー #1566）

> **状態: 契約確定（2026-09-12 ユーザー承認の形式化）。実装は PR #1659
> （イシュー #1566）で導入予定・未マージ**。base（main）時点では本契約を
> 満たす実装（`soft_f64.rs`・`bias_f64_*` 関数群・`gemm_bias_grad_reduce_f32`・
> `reduce_bias_grad_rows` 系）は存在しない。契約: Metal の勾配長軸縮約（dw
> 行方向蓄積・bias 勾配の行方向縮約）は **IEEE 754 binary64 逐次加算の 64bit
> 整数ソフトウェアエミュレーション**として実装し、ホスト `f64` 逐次和（index
> 順）を 1 回 `f32` へ downcast した値と**bit 完全一致**する（NaN のみ quiet
> NaN へ正規化しクラス一致で比較）。tolerance・baseline・REQ-2 判定は不変。
> この契約を満たす実装が導入されれば、事前判定可能な入力上限・非有限値の
> 特例契約は不要になる見込み（bit 一致のため。詳細は下記「2. 確定契約」）。
> **PR #1659 マージ後、本文書へ実装記録節（ファイル・関数名・実測結果）を
> 追記する**。

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
64bit 整数で忠実に再現するソフトウェアエミュレーション（PR #1659 で導入予定の
`bias_f64_widen`／`bias_f64_add`／`bias_f64_narrow`。ホスト側逐語モデルとして
同 PR が導入予定の `crates/backend-metal/src/soft_f64.rs`）へ置き換える作業を
進めている（同 PR 上のコミット `cf6c9b51`／`754b36e9`。**base への未マージ**）。
この実装が導入されれば、ホスト参照実装と bit 完全一致する契約が成立し、本
ドキュメントが定義していた Tier A/B の数値許容契約は**不要となる見込み**である
（実装記録の正本となる予定は `docs/backend-metal-command-batching-design.md`
§10.10〜§10.14 だが、**base〈main〉の同ドキュメントは §10.6 までであり、
これらの節も PR #1659 側で導入予定**。本ドキュメントは PR #1659 のマージを
前提として全面的に縮小したが、マージ完了・実装記録の正本反映は別途確認する）。

## 2. 確定契約

**以下は PR #1659（未マージ）で導入予定の契約であり、base（main）時点では
まだ実装されていない**:

- **実装**: `gemm.metal::gemm_bias_grad_reduce_f32` の `m ≥ 2` 経路を IEEE 754
  binary64 の逐次加算（round-to-nearest-even）を 64bit 整数（`ulong`／`long`）
  演算で忠実に再現する方式にする（PR #1659 で導入予定の `bias_f64_widen`／
  `bias_f64_add`／`bias_f64_narrow`／`bias_clz64`）
- **契約**: 演算列はホスト参照実装（PR #1659 で導入予定の
  `crates/autodiff/src/eval.rs::reduce_bias_grad_rows`／
  `crates/backend-metal/src/layout.rs::reduce_bias_grad_rows_host`。
  `acc: f64 = 0.0` から index 順に加算し最後に 1 回 `acc as f32`）と同一に
  なり、結果は**bit 完全一致**する（NaN のみ payload がハードウェア依存の
  ため quiet NaN へ正規化し、クラス一致で比較する）
- **ホスト側逐語モデル**: PR #1659 で導入予定の
  `crates/backend-metal/src/soft_f64.rs`（`widen_f32_bits`／`add_f64_bits`／
  `narrow_f64_bits`）が Rust の `f64` 実演算と `to_bits` で一致することを
  ユニットテストで検証する
- **tolerance・baseline・REQ-2 の適用範囲はいずれも不変**（`RELATIVE_
  TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD` は無変更）。weight 勾配
  （`gemm_fp32_strict_into`）と同じ bit 一致契約へ揃える
- 「3. 検討したが不採用の案」に記す入力上限（`n < 2^24` の `InvalidArgument`
  拒否）・非有限値クラス一致規則は、上記の bit 完全一致契約が実装されれば
  不要になる見込み（実装がホストと同一演算列であれば、有効範囲・非有限値
  いずれも自然に一致するため）

## 3. 検討したが不採用の案（履歴）

- **Neumaier 改良版 Kahan 補償和 + 2 の冪 scale**（f32 のみで「`f64` 相当」を
  狙う案）: 相殺列でホスト `f64` 逐次和と一致しない問題が解消できなかった
- **Tier A/B の 2 層契約**（無条件誤差上界 `(C + n·ε32)·ε32·Σ|x_i|` ＋ REQ-2
  適用可否の事前判定述語）: 判定方式としては事前判定可能な形へ改善を重ねたが
  （`C = 8 → 3 → 4`）、bit 一致方式の採用により不要となる見込みとなったため
  不採用となった
- **入力上限 `n < 2^24` の明示拒否・非有限値クラス一致規則**: Tier A/B 前提の
  特例であり、bit 完全一致契約が導入されれば不要となる見込み

## 4. spec 起票要否

本ドキュメントの縮小に伴い、Tier A/B 判定パターンの spec REQ-2 への一般化提案は
行わない（対象契約が不要となる見込みのため）。

## 5. 関連

- PR #1666（本ドキュメント）・PR #1659（イシュー #1566 実装 PR。binary64
  ソフトウェアエミュレーションを導入予定。commit `cf6c9b51`／`754b36e9`。
  **未マージ**）・イシュー #1566（親）
- `docs/backend-metal-command-batching-design.md`（**base〈main〉時点は
  §10.6 まで**。§10.10〜§10.14 は PR #1659 側で導入予定の実装記録の正本）
- 本ドキュメントが応答した codex-review スレッド: `PRRT_kwDOTuUCJc6hvjYM`・
  `PRRT_kwDOTuUCJc6hvnzg`・`PRRT_kwDOTuUCJc6hvnzi`・`PRRT_kwDOTuUCJc6hvrGx`・
  `PRRT_kwDOTuUCJc6hvrGy`・`PRRT_kwDOTuUCJc6hvug6`・`PRRT_kwDOTuUCJc6hvxnm`・
  `PRRT_kwDOTuUCJc6hv2ss`・`PRRT_kwDOTuUCJc6hv2sw`・`PRRT_kwDOTuUCJc6hv5IK`・
  `PRRT_kwDOTuUCJc6hv-z1`（本改訂の指摘元。「実装置換により成立した」という
  完了形の記述が base 時点で未検証だった点の是正）
- `AGENTS.md`「数値契約の統一（P1）」・`.claude/rules/coding-rust.md`「バック
  エンド構成（REQ-2）」節（本ドキュメントを正本として参照する側。契約文は
  維持しつつ「実装は PR #1659 で導入（未マージ時点では契約のみ）」を明記）
