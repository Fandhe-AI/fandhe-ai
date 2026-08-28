# facade optimizer 公開 API 昇格の設計判断（#932）

イシュー #932「optimizer 公開 API 昇格の設計判断」（親 #923）に対応する。**本文書は設計判断
（ドキュメント成果物）のみを扱い、`facade` への optimizer API 実装そのものは含まない**（実装は
本文書 §8 の提案に基づき後続イシューとして切り出す）。デバイス常駐化（性能面の改善）は兄弟
イシュー #933〜#936 の担当であり、本文書は API 公開面の形（サポート境界・シグネチャ）のみを
扱う。

## 判断サマリ

**昇格を採用と推奨する。** `fandhe_ai::optim` モジュール（案 A）として `Sgd`/`SgdConfig`/
`AdamW`/`AdamWConfig` を再エクスポートする形を軸とし、`clip_grad_norm`/`LrScheduler` の同時
昇格も推奨する。あわせて、サポート境界の変更を伴うため `docs/compat-api-scope.md` §0 の入口
列挙の更新と、正本 spec リポジトリ側での REQ-9 受け入れ基準の改定提案が必要と判断する（§7）。
本文書は判断の記録であり、範囲拡張の実施自体は `docs/compat-api-scope.md` §5 の手続き（spec
改定またはユーザー承認）を経て別イシューで行う。

## §1 背景・実測根拠

現在、サポート対象公開面（`facade` = `fandhe_ai` クレートが唯一のサポートされる公開 API 面。
`docs/compat-api-scope.md` §0）には optimizer API が存在しない。利用者が学習を行う唯一の経路は
「`Sequential::trainable_parameters()` でパラメータを取得 → ホスト側で `param - lr * grad` を
手計算 → `Sequential::apply_parameters()` で書き戻し」という手動 SGD である。この手動経路は
`scripts/bench/framework-compare/bench-fandhe/src/main.rs:206-238`（`run_train`）に実装されて
おり、各 step で全パラメータを `.contiguous().as_slice().ok_or(..)?.to_vec()` によりホストへ
吸い出し、更新後の値を `Tensor::from_slice(&upd, param.shape())?` で再構築している。

フレームワーク横並びベンチ（PR #915・`scripts/bench/framework-compare/results/summary.md`「(b)
MLP 学習」節・計測 2026-08-28・Apple M4 Max）では、この手動経路による MLP 学習（784→256→10、
ReLU、バッチ 64、MSE、SGD lr=0.01）が次のとおり計測されている。

| デバイス | フレームワーク | 中央値（1 step） |
| --- | --- | --- |
| cpu | fandhe-ai（手動 SGD） | 18.185 ms |
| cpu | candle（公式 optimizer） | 797.5 µs |
| cpu | burn（公式 optimizer） | 626.5 µs |
| metal | fandhe-ai（手動 SGD） | 48.845 ms |
| metal | candle（公式 optimizer） | 751.8 µs |
| metal | burn（公式 optimizer） | 1.606 ms |

CPU で 1 桁以上、Metal ではさらに大きな差が生じている。ただし同 summary.md の Metal 行注記
（「(b)/(c) Metal 行のプロトコル注意」節）が明記するとおり、fandhe-ai は毎ステップ新規
`tape_for(Device)` を構築する運用であるのに対し candle/Burn はデバイスを使い回すため、Metal 側
の差にはデバイス/tape 再構築コストが大きく乗っている。**この差の内訳はデバイス常駐化
（#933〜#936 の担当領域）と、手動 SGD 特有のホスト往復コスト（本文書が扱う範囲）の 2 要因が
混在しており、optimizer API 昇格単体が Metal 側の差を全て解消するわけではない。** CPU 側の差
（tape 構築コストがほぼ無視できる環境）は、手動経路のホスト往復（`as_slice().to_vec()` による
全パラメータのコピー・`Tensor::from_slice` による再構築、および毎 step の tape 再生成）の寄与が
相対的に大きいと推測されるが、本文書は「optimizer API 昇格が公開面としてサポート境界の矛盾を
解消する」ことを主目的とし、性能改善の定量的な内訳分析は本文書のスコープ外とする（切り分けが
必要であれば後続イシューで計測する）。

## §2 現状の公開面と矛盾

`facade`（`fandhe_ai` クレート）が唯一のサポートされる公開 API 面であり、内部クレート
（`tensor-core`/`autodiff`/`backend-*`）を直接利用することはサポート対象外である
（`docs/compat-api-scope.md` §0）。しかし `crates/facade/src/compat/sequential.rs` の doc
コメントは、利用者に内部クレートへの接続を明示的に案内している。

- モジュール冒頭（`sequential.rs:14-15`）: 「`fandhe_ai_autodiff::optim::Sgd`・
  `fandhe_ai_autodiff::nn::optim::AdamW` の位置対応契約にそのまま渡せる」
- `trainable_parameters`/`apply_parameters` の doc（`sequential.rs:162,179,203,379`）: いずれも
  `Sgd::step`/`AdamW::step`（内部クレート `fandhe_ai_autodiff` の型）を名指しで案内する

`docs::compat-api-scope.md` §0 は「`autodiff` の... API は Rust の可視性としては `pub`（`autodiff`
クレートのドキュメント上は到達可能）だが、サポート境界上は内部 API」と明記しており、
`fandhe_ai::tape()`/`fandhe_ai::tape_for(Device)` と `fandhe_ai::compat::{array, Sequential}` の
2 つのみが利用者の入口と定める。`sequential.rs` の doc 案内はこの境界と矛盾した状態にある
（利用者が doc コメントの案内どおりに実装しようとすると、サポート対象外の内部クレートへ直接
依存することになる）。

## §3 昇格の是非（判断本体）

昇格を採用と推奨する根拠は以下の 3 点である。

1. **サポート境界矛盾の解消**: 昇格により、利用者が内部クレート（`fandhe_ai_autodiff`）への
   直接依存なしに学習ループを完結できるようになり、`sequential.rs` の doc 案内と
   `compat-api-scope.md` §0 の矛盾が解消する。
2. **REQ-12 非抵触**: `Sgd`/`AdamW` は `Tape`/`Var`/`BackendOps` に依存しない値型
   （`Tensor<f32>` の集合を受け取り `Tensor<f32>` の集合を返す純粋な計算。`crates/autodiff/src/
   optim/mod.rs` 冒頭コメント）であり、`Tape::new_with_ops` のような `BackendOps` 注入経路を
   持たない。`crates/facade/tests/api_surface.rs` の機械検査（(a) `pub use` に `Tape`/
   `BackendOps`/`new_with_ops` を含めない、(b) `pub fn` が `BackendOps` を引数に取らない）に
   抵触しない再エクスポートが可能であり、REQ-12「任意 `BackendOps` 実装を注入できる公開 API を
   設けない」制約を維持できる。
3. **横並びベンチの公平性**: candle/Burn は公式 optimizer API で計測しているのに対し、
   fandhe-ai のみ手動実装で計測しており、フレームワーク間の性能比較の前提が非対称である
   （§1）。公開 optimizer API があれば、この非対称を将来のベンチ更新で解消できる。

## §4 公開面の形の設計

### 4.1 モジュール構成案の比較

- **案 A（推奨）**: `fandhe_ai::optim` として `Sgd`/`SgdConfig`/`AdamW`/`AdamWConfig` を
  再エクスポートする。内部クレートでは `Sgd`/`SgdConfig` が `fandhe_ai_autodiff::optim` に、
  `AdamW`/`AdamWConfig` が `fandhe_ai_autodiff::nn::optim` に置かれ配置が不統一だが
  （`crates/autodiff/src/nn/optim/mod.rs` 冒頭コメント「統合は親 #192 完了時に判断する」）、
  facade 側で 1 モジュールへ吸収することで利用者から見た配置の不統一を隠蔽できる。
  `crates/facade/src/lib.rs` が既に `Var`/`Gradients`/`AutodiffError`/`LinearVars`/`Tensor` を
  値型として再エクスポートしている前例（迂回経路を持たない値型は facade の正式な公開契約とする
  方針。`lib.rs:83-88` のコメント）と同じ扱いで一貫する。
- **案 B（不採用）**: facade 所有の newtype でラップする（`Tape` newtype と同型の設計）。
  `Tape` newtype が必要なのは `Tape::new_with_ops`（`BackendOps` 注入経路）という到達させては
  ならない迂回経路が内部型に存在するためである（`lib.rs:92-101` のコメント）。`Sgd`/`AdamW` に
  はそのような迂回経路が存在しない値型であるため、newtype ラップは過剰な間接化であり複雑性を
  増すだけで REQ-12 上のメリットがない。案 A（素の再エクスポート）を推奨する。

### 4.2 gradient clipping・LR スケジューラの同時昇格

`clip::clip_grad_norm`/`lr_scheduler::LrScheduler`（`crates/autodiff/src/nn/optim/mod.rs`）も
`Gradients`/`Var` に依存しない純関数・純データ構造であり、案 A と同じ理由で昇格可能である。
1 学習ステップの適用順序契約（`backward → clip → optimizer step`。同モジュール冒頭コメント）は
facade 側の doc でも踏襲して明記する必要があり、後続の実装イシューで `fandhe_ai::optim` の
モジュール doc に契約を転記することを推奨する。

### 4.3 シグネチャ不統一の扱い

`Sgd::step(&mut self, params: &[&Tensor<f32>], grads: &[&Tensor<f32>]) -> Result<Vec<Tensor<f32>>,
AutodiffError>`（`crates/autodiff/src/optim/sgd.rs:185-188`）と `AdamW::step(&mut self,
params_and_grads: &[(&Tensor<f32>, &Tensor<f32>)]) -> Result<Vec<Tensor<f32>>, AutodiffError>`
（`crates/autodiff/src/nn/optim/adamw.rs:139-141`）は、2 スライス方式とタプルスライス方式で
引数形が異なる。この不統一は親 #192 コメント（`optim/mod.rs`「共通 `Optimizer` trait の導入は
AdamW 実装時に必要性を判断する」「統合は親 #192 完了時に判断する」）で意図的に先送りされている
未解決事項である。

昇格の判断としては、**シグネチャ統一を昇格の前提条件にしない**ことを推奨する。理由は次のとおり。

- 統一（共通 `Optimizer` trait 導入等）は親 #192 の統合判断を要する横断的な変更であり、本イシュー
  （#932。設計判断のみ）のスコープを越える
- 現状のシグネチャのまま `fandhe_ai::optim::{Sgd, AdamW}` として再エクスポートしても、それぞれ
  独立した具象型としては REQ-12 に抵触しない（§3-2）
- ただし一度公開すると、後から `step` のシグネチャを変える変更は破壊的変更（`!` 付き
  Conventional Commits・`BREAKING CHANGE:` 記載。`.claude/rules/conventional-commits.md`）になる
  ため、**昇格前に親 #192 の統合判断（共通 trait 導入の要否）を先に確定させることが望ましい**。
  これは「統一してから昇格」を必須にする意味ではなく、「昇格後の破壊的変更コストを認識した上で
  昇格順序を判断する」ことを後続イシューの計画時に明記すべき事項として記録する。

## §5 #933〜#936（デバイス常駐化）との整合

デバイス常駐化（#934 が担当する設計）は `Sequential`/`Tape` のライフサイクル変更（毎 step の
`tape_for` 再構築をやめ、デバイス上にパラメータ・tape を常駐させる）を扱う見込みである。昇格する
optimizer API の形（`Sgd::step`/`AdamW::step` が `&[&Tensor<f32>]` の位置対応契約で受け渡しする
構造）はデバイス常駐化の設計を妨げない。`Sequential::trainable_parameters`/`apply_parameters` の
位置対応契約（層追加順・weight → bias。`sequential.rs:160-260`）は、常駐化後もパラメータの
「取得 → 更新後の値で差し替え」という抽象を維持できる限り両立する。

公開タイミングについては、**#934 の設計確定を待たずに先行公開することを推奨する**。理由は、
`Sgd`/`AdamW` の `step` シグネチャ自体は `Tape`/`Var`/`BackendOps` から独立しており（§3-2）、
常駐化がテープ・デバイス側の内部実装を差し替えるものであれば `Sequential` 側のパラメータ取得・
書き戻し API（`trainable_parameters`/`apply_parameters`）とその契約が変わらない限り optimizer
側の公開シグネチャに影響しない可能性が高いためである。ただし常駐化の設計次第で
`trainable_parameters`/`apply_parameters` 自体のシグネチャが変わる場合は、その変更が
optimizer の呼び出し形にも波及しうるため、#934 の設計内容を実装イシュー着手前に確認することを
後続イシューの前提条件として明記する。

## §6 compat-api-scope との整合確認（受け入れ条件 2）

`docs/compat-api-scope.md` §1 が定める compat 層の対象範囲（numpy/Keras 慣習の薄いラッパー。
`compat::array`/`compat::Sequential` の 3 種限定）に optimizer は含まれない。optimizer は
numpy/Keras 慣習のラッパーではなく、**facade 素の公開契約（`tape()`/`tape_for`/再エクスポート
値型と同じ層）への追加**として位置づけるべきであり、compat 層（§1）の対象範囲を変更する話では
ない。

ただし、`docs/compat-api-scope.md` §0 は「`fandhe_ai::tape()`/`fandhe_ai::tape_for(Device)`
（composition root）と `fandhe_ai::compat::{array, Sequential}`（compat 公開面）の 2 つが、
利用者が使うことを想定する唯一の入口である」と入口を列挙している。optimizer 昇格はこの列挙に
第 3 の入口（`fandhe_ai::optim::{Sgd, AdamW}`）を加えることになるため、**§0 の入口列挙の更新を
要する**。これは §1 の対象範囲拡張ではなく §0 のサポート境界記述の更新であるが、§0 自体が
「サポート境界の変更（内部クレートの直接利用をサポート対象に含める等）は正本 spec リポジトリ
側での REQ-9/REQ-12 受け入れ基準の改定を要する（5 節「範囲拡張の手続き」と同じ手続き）」と
定めているため、**§5 と同じ手続き（spec 側改定またはユーザー承認）を経て初めて実施できる**。
本イシューの受け入れ条件は「整合の確認」であり「範囲の拡張」ではないため、本文書では確認の
結論のみを記録し、§0 自体の書き換えは行わない。

## §7 spec 側更新の要否判断（受け入れ条件 3）

**「要」と結論する。** サポート境界（＝利用者が使う入口の列挙）は `docs/compat-api-scope.md` §0
が「正本 spec リポジトリ（`Fandhe-AI/rust-ai-library-spec`）の REQ-9 の 2026-08-08 追記」を
根拠として定めたものである。公開面へ optimizer という新しい入口を追加することは、この受け入れ
基準（サポート境界の明文化）の改定に相当する。

`docs/spec/` は正本 submodule であり本リポジトリでは編集しない
（`.claude/rules/out-of-scope-tracking.md`「仕様変更が必要な場合」）。以下は spec リポジトリ側
への改定提案の文案であり、実施はユーザーへの提案に留める。

> **改定提案（案）**: REQ-9（`docs/spec/04-requirements.md:200-209`）の 2026-08-08 追記
> （サポート境界の明文化）に、「`fandhe_ai::optim`（`Sgd`/`AdamW`/`clip_grad_norm`/
> `LrScheduler` 等の optimizer 群）も facade が唯一のサポートされる公開 API 面に含まれる入口の
> 1 つである」旨を追記する。根拠はイシュー #932（本文書）の判断記録。

## §8 後続イシューの提案（実施はユーザー承認後）

以下を実装イシューとして切り出すことを提案する（本文書では起票しない。ユーザー承認後に
`create-issue`/`create-issue-tree` で起票する）。

1. `crates/facade/src/optim.rs`（新規）: `fandhe_ai_autodiff::optim::{Sgd, SgdConfig}`・
   `fandhe_ai_autodiff::nn::optim::{AdamW, AdamWConfig}`・`clip::clip_grad_norm`・
   `lr_scheduler::LrScheduler` を `fandhe_ai::optim` として再エクスポートし、適用順序契約
   （`backward → clip → step`）を doc に転記する
2. `crates/facade/tests/api_surface.rs` の拡張: 新規追加分についても (a)/(b) の機械検査対象に
   含める（`visit_rs_files` は `src/` 配下を再帰走査するため自動的に含まれるが、`optim.rs` 固有
   の新規公開型に対する検査観点の見直しを実装時に確認する）
3. `docs/compat-api-scope.md` §0 の入口列挙更新（§6 の整合確認の結論を反映。実施は同文書 §5 の
   手続き＝spec 改定またはユーザー承認を経てから）
4. `crates/facade/src/compat/sequential.rs` の doc コメント差し替え: `fandhe_ai_autodiff::optim`/
   `fandhe_ai_autodiff::nn::optim`（内部クレート）への案内を `fandhe_ai::optim`（facade 公開面）
   への案内に置き換える
5. （任意・親 #192 の統合判断待ち）`Sgd::step`/`AdamW::step` シグネチャ統一の要否判断
   （§4-3 参照）

## §9 出典一覧

| 出典 | 内容 |
|------|------|
| `scripts/bench/framework-compare/results/summary.md`「(b) MLP 学習」節 | 手動 SGD 経路の実測（CPU 18.185 ms・candle 797.5 µs・burn 626.5 µs 等。計測 2026-08-28・Apple M4 Max） |
| `scripts/bench/framework-compare/results/summary.md`「(b)/(c) Metal 行のプロトコル注意」節 | Metal 側の差にデバイス/tape 再構築コストが乗っている旨の既存注記 |
| `scripts/bench/framework-compare/bench-fandhe/src/main.rs:206-238` | 手動 SGD 実装（`run_train`。`as_slice().to_vec()`・`Tensor::from_slice` によるホスト往復） |
| `crates/facade/src/lib.rs:83-101` | facade の現行公開面（`compat`・値型再エクスポート・`Tape` newtype とその理由） |
| `crates/facade/tests/api_surface.rs:1-16` | REQ-12 機械検査（`pub use` 禁止語・`pub fn` 引数検査）の対象・走査方式 |
| `crates/autodiff/src/optim/sgd.rs:185-188`・`optim/mod.rs` | `Sgd::step` シグネチャ・`Tape`/`Var` 非依存の設計方針コメント |
| `crates/autodiff/src/nn/optim/adamw.rs:139-141`・`nn/optim/mod.rs` | `AdamW::step` シグネチャ・適用順序契約（`backward → clip → step`）・モジュール配置不統一の経緯 |
| `crates/facade/src/compat/sequential.rs:14-15,160-260,379` | 位置対応契約・内部クレート `fandhe_ai_autodiff::optim`/`nn::optim` への doc 案内（サポート境界との矛盾箇所） |
| `docs/compat-api-scope.md` §0 | サポート境界（facade が唯一の公開面・入口 2 つの列挙・§5 と同じ手続きの適用） |
| `docs/compat-api-scope.md` §1・§5・§6 | compat 層対象範囲（3 種限定）・範囲拡張の手続き・出典一覧の体裁踏襲元 |
| `docs/public-api-design.md:6,13` | compat 層と自作コア素の公開 API の境界記述 |
| `.claude/rules/coding-rust.md`「基盤方針」 | 「互換 API 層は自作コアの上の薄いラッパーに徹する（REQ-9）」 |
| `.claude/rules/conventional-commits.md` | 破壊的変更時の `!`・`BREAKING CHANGE:` 記載規約 |
| `.claude/rules/out-of-scope-tracking.md` | spec 変更が必要な場合の扱い（`docs/spec/` 非編集・spec リポ側提案） |
| イシュー #932（本イシュー） | 本文書の対応イシュー |
| イシュー #923（親） | optimizer 公開 API 昇格系列の親トラッキング |
| イシュー #933〜#936（兄弟） | デバイス常駐化（性能面）の担当領域 |
| イシュー #192（親。#193/#194/#195 の統合判断待ち） | 共通 `Optimizer` trait 導入・SGD/AdamW モジュール配置統一の判断元 |
| イシュー #294・#426 | `Sequential::trainable_parameters`/`apply_parameters` 位置対応契約 |
| PR #915 | フレームワーク横並びベンチ（`scripts/bench/framework-compare/`）導入 |
