# 参照モデル定義（`Mlp`・`LeNet`）の設計判断記録

イシュー #2201（親 #2190）で実装した PyTorch 定番モデル 2 種
（MLP・LeNet）の参照実装に関する配置・保留事項・PyTorch 対応・
事前登録した学習判定式の記録。

## 1. 目的と位置づけ

- 目的は次の 3 点。
  - 利用者が公開 API（`fandhe_ai::compat::Sequential`・`fit`・`evaluate`）
    だけで PyTorch の定番モデルを組めることを、実行できるコードで示す。
  - PyTorch の参照定義と層ごとの重みの対応（キー名・shape・転置の有無）を
    はっきりさせる。
  - MNIST 規模の学習が収束する方向に進むことを、あらかじめ決めた判定式で
    確かめる。
- 親 #2190（torchvision／torchtext 相当の最小セットの提供）のうち、本
  イシューは MLP・LeNet を受け持つ。ResNet・Transformer は #2202 の担当。

## 2. 配置（`crates/facade/examples/models/` を選んだ理由）

- ルートの `Cargo.toml` は virtual workspace（`[package]` を持たない）
  のため、イシュー本文が想定していたルート直下 `examples/models/` は
  ビルド対象にならない。配置先は facade クレート配下
  `crates/facade/examples/models/` とした。
- facade の `examples/` は crates.io パッケージに同梱される
  （`crates/facade/Cargo.toml` の `[[example]]` コメント。
  `cargo package -p fandhe-ai --list` で `examples/models/{mod,mlp,lenet}.rs`
  が含まれることを確認済み）。そのため model ファイルは公開パス
  （`fandhe_ai::…`）だけを使い、`bench_harness`（`publish = false`）や
  内部クレート（`fandhe_ai_autodiff` 等）を import しない。
- 先例として `crates/facade/examples/hf_safetensors_sequential/convert.rs`
  があり、`crates/facade/tests/interop_safetensors_hf_layout.rs` が
  `#[path = "../examples/…/convert.rs"] mod convert;` で取り込んでいる。
  本 issue も同じ方式（`#[path]` によるテストからの直接取り込み）を使う。
- `mlp.rs`・`lenet.rs` はそれぞれ**単独で完結**させ、`super::`／`crate::`
  による相互参照を持たない。各テストが自分に必要なファイルだけを
  `#[path]` で取り込むためで、対応表の型（`MlpParamMap`／`LeNetParamMap`）
  も個別に定義している（多少の重複は許容）。

## 3. 保留事項（facade 公開面拡張・doctest）

### 3.1 facade 公開面拡張（`pub use`）

- イシュー本文は `MLP`／`LeNet` 型を `fandhe_ai::` から `pub use` する
  ことを前提としているが、#2201・親 #2190 のいずれにも所有者の明示承認
  コメントが確認できなかった。`docs/compat-api-scope.md` §5「範囲拡張の
  手続き」の経路 2（本リポジトリのユーザー承認）に該当するため、
  先例（#2311 の `train_step_fn`・#2180 の勾配累積等）と同じく保留し、
  `crates/facade/src/` は一切変更していない。
- 他の保留エントリと異なり、本 issue では「facade 側の保留対象コード」
  自体を書いていない（`Mlp`／`LeNet` は `compat::Sequential::add_*` の
  組み合わせのみで構成した examples 配下の**利用者コード**）。そのため
  `HoldDoctestGuard` 方式の否定ガード（`api_surface.rs` への不在検査）は
  **追加していない**——ガードで守るべき対象がなく、否定ガードだけの検査は
  #2212 のレビューで受け入れられなかった前例と同じ判断軸による
  （`docs/compat-api-scope.md` §5 追記）。
- 承認取得後の移行手順: `crates/facade/src/models/` へ移設 →
  `pub mod models`（または個別 `pub use`）→ 本物の doctest への切り替え
  → `crates/facade/tests/api_surface.rs` の期待値更新。

### 3.2 doctest（AC3 の代替）

- rustdoc は `examples/` 配下を doctest しない（`cargo test --doc` の
  対象外）。そのため AC3「各型の生成と forward を doctest で実行できる」
  は、統合テスト（`example_mlp_mnist.rs`・`example_lenet_mnist.rs`）と
  runnable example（`cargo run -p fandhe-ai --example reference_models`）
  で代替する。3.1 の移行後、`src/models/` へ移す際に本物の doctest を
  付ける。

## 4. イシュー本文との差分

| 項目 | イシュー本文 | 本実装 |
|---|---|---|
| 配置 | ルート `examples/models/` | `crates/facade/examples/models/`（§2） |
| 公開 | `fandhe_ai::{MLP, LeNet}` | facade 公開面拡張は保留（§3.1）。examples 限定 |
| 型名 | `MLP` | `Mlp`（`MLP` は `clippy::upper_case_acronyms` に抵触するため） |
| 内部構造 | `Vec<Linear>` 等のフィールド | `compat::Sequential` を内部に持つラッパー（facade は層を直接公開しないため） |
| doctest | 型定義に直接 doctest | 統合テスト＋runnable example で代替（§3.2） |
| LeNet の fc 層数 | — | Conv2d 2 層 + Dense 2 層版（古典 LeNet-5 の fc 3 層版とは異なる。イシュー受け入れ条件「Conv2d 2 層 + Dense 2 層」に合わせた選択） |
| データ | MNIST | 合成 MNIST 相当データ（実データは同梱せずネットワーク取得も行わない。§6） |

## 5. PyTorch 参照定義と重み対応表

### 5.1 `Mlp`

```text
nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(p),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p),
    nn.Linear(128, 10),
)
```

`Mlp::new(784, &[256, 128], 10, p)` が対応する。fandhe の
`nn::Linear.weight` は `[in_features, out_features]`
（`crates/autodiff/src/nn/linear.rs` 冒頭 doc）で PyTorch
`nn.Linear.weight` の `[out_features, in_features]` とは**転置の関係**。
bias は両者とも `[out_features]` で転置不要。`Sequential` 上の index は
PyTorch `nn.Sequential` と同じ規約（活性化・Dropout も index を消費する）
で、`0`／`3`／`6` に `Linear` が並ぶ。

### 5.2 `LeNet`

```text
conv1 = Conv2d(1, 6, 5)                  # 28 -> 24 (padding 0)
relu -> max_pool2d(2)                    # 24 -> 12
conv2 = Conv2d(6, 16, 5)                 # 12 -> 8
relu -> max_pool2d(2)                    # 8  -> 4
flatten(1)                               # 16*4*4 = 256
fc1 = Linear(256, 120) -> relu
fc2 = Linear(120, num_classes)
```

`nn::Conv2d.weight` は `[out, in/groups, kH, kW]`
（`crates/autodiff/src/nn/conv.rs`）で PyTorch `nn.Conv2d.weight` と
**同一 shape**（転置不要）。`Linear` 系は `Mlp` と同じ転置契約。
`Sequential` 上の index: `0` conv1／`1` relu／`2` maxpool／`3` conv2／
`4` relu／`5` maxpool／`6` flatten／`7` fc1／`8` relu／`9` fc2。

## 6. 合成 MNIST 相当データ（実 MNIST ではない）

- リポジトリに MNIST を同梱せず、ネットワーク取得も行わない（依存・
  ネットワークアクセスを増やさない方針）。既存テスト
  （`compat_sequential_fit_amp_conv_mha.rs` 等）も「MNIST 規模」の
  合成データを使っており、同じ方針を踏襲した。
- `crates/facade/tests/example_models_common/mod.rs` が
  `bench_harness::rng::Xorshift64Star`（決定的シード駆動。
  `.claude/rules/coding-rust.md`「学習系回帰テストには決定的シード
  設定ユーティリティを使う」）でクラスごとの固定プロトタイプ画素パターン
  （0.2 / 0.6 の 2 値）へ `[-0.25, 0.25)` の一様ノイズを乗せて生成する。
  戻り値は行優先の平坦な `Vec<f32>` で、`[N, 784]`（MLP）と
  `[N, 1, 28, 28]`（LeNet）は同じメモリレイアウトの reshape に過ぎない
  ため単一関数に統一した（shape 別に関数を分けると一方のテストでしか
  使われない関数が `dead_code` になるため）。
- ノイズ振幅はプロトタイプの点灯／非点灯の差（0.2 対 0.6）と同程度に
  設定し、1 epoch で完全分離しきらない程度の難度に調整した（§7 の判定式
  が実質的な学習進捗を確認できるようにするため）。

## 7. 事前登録した判定式（AC4）と実測結果

手順（判定式は結果を見る前に固定し、FAIL しても緩めない）:

1. モデルを固定シードで構築する。
2. `compile(Optimizer::Adam(AdamConfig{lr, ..}), Loss::CrossEntropy)`。
3. `before = evaluate(x, y, batch_size)`。
4. `fit(x, y, FitConfig::new(1, batch_size))`（`shuffle` は既定 `false`）。
5. `after = evaluate(x, y, batch_size)`。

判定: `before`・`after`・`History.loss[0]` がすべて有限であり、
`after <= 0.5 * before`。調整してよいのは判定式を確定する前の合成データの
設計・N・batch size・学習率・optimizer だけであり、判定式（係数 0.5 や
比較の向き）自体は変更していない。tolerance・baseline・ガードレール
閾値には一切触れていない。

| モデル | N | batch_size | optimizer | 実測 before → after | 判定 |
|---|---|---|---|---|---|
| `Mlp` | 256 | 32 | Adam(lr=8e-3) | 2.3025 → 0.0842 | pass |
| `LeNet` | 1024 | 16 | Adam(lr=3e-3) | 2.3055 → 0.2436 | pass |

**LeNet の lr 探索メモ**: Adam lr を 0.05〜2.0 の範囲で試したところ、
lr が高いと 1 エポック目の大きな更新で ReLU が広範囲に死んで
（"dying ReLU"）以降のエポックで loss が `ln(10) ≈ 2.3026`（一様分布と
同値）に張り付いたまま動かなくなる現象を確認した（`lr=0.15` で
`history.loss = [4.09, 2.304, 2.304, 2.304, 2.304]` という 5 epoch
診断で確認）。`lr=3e-3` ではこの collapse が起きず単調に減少した
（`history.loss = [1.74, 0.083, 0.016, 0.005, 0.002]`）。最終的に採用した
`lr=3e-3`・1 epoch（64 ステップ）は診断より低い値だが、collapse が
起きない安全域であることを多エポック実行で確認済み。

## 8. GPU parity（該当なし）

本 issue は新しい演算・カーネルを追加していない
（`compat::Sequential::add_*` の組み合わせのみ）。`fit`／`predict` は
CPU（`fandhe_ai::tape()` 既定バックエンド）固定で実行するため、CUDA／
Metal での facade parity 実測・`docs/perf/logs/` への申し送りは不要と
判断した。

## 9. スコープ外

- ResNet・Transformer の参照モデル定義（#2202）。
- 重み初期化方式の詳細検討（#2140）。
- 事前学習済み重みのロード（#2082）。
- facade 公開面拡張・本物の doctest への切り替え（§3。承認後に別途
  対応。起票自体もユーザー承認を得てから行う。
  `.claude/rules/out-of-scope-tracking.md`）。
