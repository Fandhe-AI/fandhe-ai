# amax／amin 勾配分配方式（均等分配 対 先勝ち決定的）の確定（#1718）

イシュー #1718「amax／amin 勾配分配方式（均等分配 対 先勝ち決定的）を確定する」の設計判断記録。親: #1601（Tier 1「縮約」）。前段の再確認: #224（Max VJP 先勝ち挙動の再確認）。本ドキュメントは**設計判断のみ**を確定するものであり、コード・テスト期待値・tolerance／baseline の変更は伴わない（挙動は現状から一切変更しない）。

## 1. 背景・要件

`docs/spec/04-requirements.md` REQ-9 の 2026-09-12 改定は「Tier 1（1.2 節・#1601）」の縮約 API 群に `amax` 相当を含めたうえで、勾配分配方式（先勝ち決定的 対 均等分配）の確定を #1601 配下の issue（本 #1718）へ引き継いだ（`04-requirements.md:234`。`docs/compat-api-scope.md` §2「`amax`/`max` 縮約 API」節）。

`crates/autodiff/src/grad.rs::extremum_first_match_vjp`（旧 `max_vjp`。#1720 で `Op::Max`／`Op::Min` の共有ヘルパーへ改称）は、縮約軸上で forward 記録値 `out_value` と `==` 一致する**最初の 1 箇所のみ**へ上流勾配を置く「先勝ち決定的」方式である。PyTorch の `torch.amax`／`torch.amin` は同値タイの発生時に勾配を**均等分配**するため、この点で挙動が異なる。

この差異は #224（2026-08-09 頃）で「compat 公開面に `amax` 相当 API が存在しないため判断を保留し先勝ちを維持する」と結論され、その後の spec REQ-9 改定・`docs/compat-api-scope.md` §2 の保留事項を経て Tier 1 縮約 issue（#1601 → 本 #1718）へ引き継がれた。#1719（`Var::max_dims`）・#1720（`Var::min`）は本 issue が未確定であることを理由に、先勝ち方式を無変更のまま採用し出荷済みである。両 issue のコード comment・doc 計 4 箇所が「#1718 が均等分配へ確定した場合はヘルパー 1 箇所の差し替えで反映される」という前提を記している。本 issue はこの前提を検証し、確定した結論で参照を更新する。

## 2. 現行実装の事実整理

- **`extremum_first_match_vjp`**（`crates/autodiff/src/grad.rs`）: `out_value` と `==` 一致する最初の要素へ `g` をそのまま置き、他は 0。`Op::Max`（`max_vjp` 経由の薄いラッパー）・`Op::Min` の両方が共有する。
- **`argmax`／`argmin` のタイ契約**（`crates/autodiff/src/eval.rs::arg_extremum`）: `better(v, best_val)` に**狭義**の `>`／`<` を使うため、同値タイでは最初に出現した添字のみが残る（`backend-cpu::reduction` のテスト `argmax_tie_returns_first_index` が同じ契約を検証）。先勝ち VJP はこの「タイは最初の添字」契約と内部的に整合している——`argmax`（値＋添字を返す族）が指す添字と、`Op::Max` の VJP が勾配を置く位置が一致する。
- **`max_dims`／`min` の出荷履歴**: `max_vjp`（先勝ち方式）は crates.io 公開全版（v0.3.0〈2026-08-23 公開〉〜v0.8.0〈2026-09-09 公開〉）を通じて出荷済みの挙動である（`crates/autodiff/src/grad.rs` の `max_vjp` 自体は 2026-08-09 頃から存在）。`Var::max`（単一軸／全軸・複数軸）は compat 公開面に含まれ、既存利用者はこの勾配値を前提にコードを書きうる。
- **PyTorch の `max(dim)`／`min(dim)` 族との対比**: PyTorch でも `torch.max(input, dim)`／`torch.min(input, dim)`（値＋添字を返す形）は均等分配ではなく**返した添字 1 箇所のみ**へ勾配を伝播する。均等分配になるのは添字を返さない `torch.amax`／`torch.amin`（`torch.max(input)` の全縮約形も同様）に限られる。

## 3. 案比較

| 案 | 内容 | 採否 |
|---|---|---|
| A. 先勝ち決定的を全面維持する（`amax` 名称の API も先勝ちのまま実装する） | 現状挙動を変更しない。`amax`／`amin` を将来追加する際も `extremum_first_match_vjp` をそのまま使う | 不採用。REQ-9 改定は Tier 1「縮約」を PyTorch 機能網羅の対象としており、`amax` という名称で PyTorch と異なる勾配分配を返す互換 API は利用者の期待（`torch.amax` からの移植）を裏切る不整合を生む |
| B. `extremum_first_match_vjp` を均等分配へ差し替える（`max`／`min`／`max_dims` すべてを均等分配化する） | grad.rs／tape.rs の comment が示唆する「ヘルパー 1 箇所差し替え」をそのまま実行する | 不採用。理由 3 点: (1) `Var::max` の先勝ち VJP は crates.io 公開全版で出荷済みの挙動であり、勾配値が変わる変更は**公開 API の破壊的変更**（ガードレール条件「公開 API 非破壊」に抵触。`.claude/rules/security.md`）。(2) PyTorch 自身も `max(dim)`／`min(dim)`（値＋添字を返す族）は均等分配ではなく返した添字 1 箇所のみへ伝播する仕様であり、本リポの `argmax`／`argmin`「タイは最初の添字」契約（§2）と先勝ち VJP は内部整合している——均等分配化するとこの整合が崩れる。(3) #1719／#1720 のテスト期待値・checkpoint 再計算前提・`reduce_dims` 併合順の設計記述がすべて変わり、本 issue のスコープ（設計判断の確定）を超える |
| **C. 分離方式**: 既存 `Var::max`／`min`／`max_dims`（`max(dim)`／`min(dim)` 族の意味論）は先勝ち決定的を**確定維持**する。`amax`／`amin` という名称の API（`torch.amax`／`amin` 相当）を新設する場合は**均等分配**を契約とし、実装は既存ヘルパーを差し替えず独立の VJP として別 issue で行う | — | **採用** |

## 4. 採用（案 C）の根拠

1. **先勝ち維持の根拠の精密化**: `grad.rs::max_vjp` の現行 doc comment は「決定性方針との衝突」を均等分配を採らない理由として挙げているが、均等分配（`g / k`。`k` はタイ数）も走査順序に依存しない決定的な演算であり、決定性の有無自体は両案を判別する要因ではない。真の根拠は次の 2 点である: (a) 出荷済み公開 API（`Var::max`／`max_dims`）の勾配値を破壊的に変更しないこと、(b) `max(dim)`／`min(dim)`／`argmax`／`argmin` という「添字を返す」族の意味論として、PyTorch 自身も先勝ち相当（返した添字のみへ伝播）であり内部整合を保つこと。本 issue でこの根拠を明記し、doc comment 側の表現もこの精密化に合わせて更新する。
2. **「1 箇所差し替え」フックは採用しない**: grad.rs／tape.rs／var.rs の既存 comment が示す「#1718 が均等分配へ確定したらヘルパー 1 箇所の差し替えで `Max`／`Min` 両方へ反映される」という前提は、案 B を不採用としたことに伴い**採用しない**。均等分配を要する `amax`／`amin`（PyTorch 名称互換 API）は、`extremum_first_match_vjp` と並置する別ヘルパー（実装時の名称案 `extremum_even_split_vjp`）＋別 `Op`（実装時の名称案 `Op::Amax`／`Op::Amin`。forward 値は `Op::Max`／`Op::Min` と同一で VJP のみ異なる）として、独立した後続 issue で実装する方針とする（実装自体は本 issue のスコープ外。§7 参照）。

## 5. 均等分配の数値契約（後続実装の正として先に固定する）

`amax`／`amin` を実装する際の VJP 契約を、実装より先に本 doc で固定しておく:

- 縮約軸上で forward 記録値 `out_value` と `==`（IEEE 754 比較）一致する要素数 `k` を数え、各一致位置へ `g / (k as f32)` を置く（`f32` 除算 1 回）。勾配の総量は `g` を保存する（`sum(distributed) == g`。丸め誤差を除く）。
- `k == 0`（`out_value` が走査対象のどの要素とも一致しない契約違反。例えば `Var::max` の NaN 伝播 max により `out_value` が NaN になった場合。§6 参照）は全ゼロとする——先勝ち方式（`extremum_first_match_vjp`）の契約違反時フォールバックと同じ安全側の扱い。
- **`.claude/rules/coding-rust.md` の「勾配の長軸縮約は `f64` アキュムレータで統一する」契約は本 VJP には適用対象外**と明記する。同契約が対象とするのは正規化統計の二乗和や dw の行方向蓄積など、複数要素を**加算**で縮約する経路である。均等分配 VJP は「タイ数 `k`（整数カウント。丸めなし）を数え、選択された各位置へ定数 `g/k` を置く」という選択・除算のみの演算であり、複数要素の総和を蓄積する経路を持たないため、丸め誤差蓄積の観点で `f64` アキュムレータ化を要する対象ではない。
- forward は既存 `max`／`min` カーネル（`BackendOps::gemm` 等と同様に CPU／CUDA／Metal の既存カーネルをそのまま再利用。`amax`／`amin` の forward 値は `max`／`min` と bit 同一）を再利用し、VJP のみホスト実装とする（`BackendOps` の新規拡張は不要）。

## 6. 影響なし確認・既知の残差（対象外として記録）

- `extremum_first_match_vjp`・`Op::Max`／`Op::Min`・`reduce_dims` の併合順・checkpoint 再計算（`docs/autodiff-checkpoint-design.md`）・#1719／#1720 のテスト期待値・tolerance／baseline はすべて不変。
- **NaN 伝播の非対称性**（対象外。#1720 の実装計画 §7 で既にスコープ外と整理済みの既知事項）: `Var::max` は NaN 伝播 max（`nan_propagating_max`）を使うため、`out_value` が NaN のとき先勝ち VJP は `==` 不一致により勾配が全ゼロになる。`Var::min` は NaN 非伝播であり非対称。均等分配方式に切り替えても NaN 伝播そのものの非対称性は解消されない（forward の max/min 実装自体を変更しない限り）。
- **`torch.max(input)`（`dim` なし全縮約）との突合は対象外**: PyTorch 側の全縮約 `max`／`min`（添字なし）の実際の勾配分配仕様は本 issue では実測・確認していない。§3 の比較は `torch.amax`／`amin`（均等分配が明確な API）と `torch.max(dim)`／`min(dim)`（添字を返す族）の対比に限定し、全縮約形の扱いは `amax`／`amin` 実装 issue で別途確認する。

## 7. スコープ外・後続提案（起票はしない。ユーザー承認後に `.claude/rules/out-of-scope-tracking.md` の手順で記録する）

- `Var::amax`／`Var::amin`（および複数軸版 `amax_dims`／`amin_dims`）の実装・`Op::Amax`／`Op::Amin`・`extremum_even_split_vjp` の実装（親 #1601 配下への追加 issue として提案）。`Var` 既存再エクスポート経由での到達を想定するため facade 新規公開面は不要と見込まれるが、提案時にその旨を明記する。
- §6 に記載した NaN 伝播の非対称性。
- PyTorch `torch.max(input)`（全縮約）の勾配分配仕様の確認と `Var::max(None)` との突合。
- CUDA／Metal 実機 parity（本 issue はコード変更を伴わないため対象なし）。

## 8. 承認事項（`amax`／`amin` 実装時に確認する事項）

- facade 直下への新規公開面の要否（`docs/compat-api-scope.md` §5「範囲拡張手続き」の対象か、`unique`〈#1734〉と同様に既存 `Var` 再エクスポート経由で足りるかの判断）。
- `Op::Amax`／`Op::Amin` という新規 `Op` variant 追加そのものはガードレール上の破壊的変更に該当しないが（既存 `Op::Max`／`Op::Min` の挙動を変えない加算的変更）、実装 PR のレビューで改めて確認する。
