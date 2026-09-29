# 参照モデル定義（`Mlp`・`LeNet`・`ResNet`・`Transformer`）の設計判断記録

イシュー #2201（親 #2190）で実装した PyTorch 定番モデル 2 種
（MLP・LeNet）の参照実装に関する配置・保留事項・PyTorch 対応・
事前登録した学習判定式の記録。イシュー #2202（親 #2190）で追加した
ResNet・Transformer は §10 にまとめる。

## 1. 目的と位置づけ

- 目的は次の 3 点。
  - 利用者が公開 API（`fandhe_ai::compat::Sequential`・`fit`・`evaluate`）
    だけで PyTorch の定番モデルを組めることを、実行できるコードで示す。
  - PyTorch の参照定義と層ごとの重みの対応（キー名・shape・転置の有無）を
    はっきりさせる。
  - MNIST 規模の学習が収束する方向に進むことを、あらかじめ決めた判定式で
    確かめる。
- 親 #2190（torchvision／torchtext 相当の最小セットの提供）のうち、本
  イシューは MLP・LeNet を受け持つ。ResNet・Transformer は #2202
  （§10）で追加済み。

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

- 重み初期化方式の詳細検討（#2140）。
- 事前学習済み重みのロード（#2082）。
- facade 公開面拡張・本物の doctest への切り替え（§3・§10.2。承認後に
  別途対応。起票自体もユーザー承認を得てから行う。
  `.claude/rules/out-of-scope-tracking.md`）。
- Vision Transformer・Attention の性能最適化（#2202 スコープ外。§10）。

## 10. #2202（`ResNet`・`Transformer`）

イシュー #2202（親 #2190）で追加した ResNet（CIFAR 版 `6n+2` 構成）・
Transformer（画像の行をトークン化する非 ViT 構成）の設計判断・保留
事項・判定式の記録。§1〜§9 の MLP・LeNet と同じ制約・同じ配置方針
（`crates/facade/examples/models/`）を踏襲する。

### 10.1 配置・取り込み方

- `resnet.rs`・`transformer.rs`・`reference_module.rs`・
  `synthetic_cifar.rs` はいずれも `crates/facade/examples/models/`
  配下に置き、`mlp.rs`／`lenet.rs` と同じ理由で**単独完結**とする
  （`super::reference_module` への参照のみ許容）。
- **取り込み方**: `reference_models.rs`（#2201）が `mod models;`
  経由で `examples/models/mod.rs` を使うのに対し、本イシューの学習
  script は `crates/facade/examples/main.rs`（イシュー本文が名指しした
  ファイル名。`cargo run -p fandhe-ai --example main` で実行）とし、
  `examples/models/mod.rs` は**経由しない**。`main.rs` は
  `#[path = "models/reference_module.rs"] mod reference_module;` の
  ように 4 ファイル（`reference_module`・`resnet`・`synthetic_cifar`・
  `transformer`）を個別に `#[path]` 取り込みする。統合テスト
  （`crates/facade/tests/example_resnet_cifar10.rs`・
  `crates/facade/tests/example_transformer_cifar10.rs`）も同じ 3
  ファイル（`reference_module`・`synthetic_cifar` + 自分のモデル
  ファイル）を同じ `mod` 識別子名で `#[path]` 取り込みする。これは
  `resnet.rs`／`transformer.rs` 内部の `super::reference_module::…`
  参照が、取り込み元（`main.rs` でも各テストでも）に依らず同じ
  相対パスで解決できるようにするための契約であり、`examples/models/
  mod.rs` は #2201 の 2 ファイル（`mlp`・`lenet`）専用のまま変更して
  いない。
- `compat::Sequential` は直列専用の合成 API のため、ResNet の
  residual 加算（`main(x) + shortcut(x)`）・Transformer の位置符号
  加算（`embed(x) + pos_encoding`）はいずれも `compat::Sequential`
  単体では表現できない。両モデルとも複数の `Sequential` 部品
  （`ResNet`: `stem`／`blocks: Vec<ResNetBlock>`／`head`。
  `Transformer`: `embed`／`encoder`／`head` + 非学習の位置符号
  `Tensor`）を保持するラッパー構造体とし、部品間の加算・活性化だけを
  手組みする設計にした。

### 10.2 保留事項（facade 公開面拡張・doctest）

- イシュー #2202 の承認事項節は `ResNetBlock`・`ResNet`・
  `Transformer` の facade 公開面拡張（`pub use`）を挙げているが、
  本 PR の作業時点で所有者の明示承認コメントは確認できなかった。
  §3.1（MLP・LeNet）と同じ理由（`docs/compat-api-scope.md` §5
  「範囲拡張の手続き」経路 2）により保留し、`crates/facade/src/` は
  変更していない。`HoldDoctestGuard` 方式の否定ガードも同じ理由
  （守るべき対象コードが facade 側に無い）で追加していない
  （`docs/compat-api-scope.md` §5 追記）。
- doctest 代替も §3.2 と同じ理由（`examples/` は `cargo test --doc`
  対象外）で、統合テスト 2 本と runnable example（`cargo run -p
  fandhe-ai --example main`）で代替する。

### 10.3 `compat::Sequential` は直列専用（設計制約の明記）

`ResNetBlock::forward`／`ResNet::forward`・`Transformer::forward` が
`Var::add`（residual・位置符号加算）を手組みしているのは、
`compat::Sequential::add_*` が単一の入力を単一の出力へ直列変換する
層しか結線できず、分岐（shortcut）や外部定数との加算を表現する API
を持たないため。学習時も同様に、各部品を `bind(&tape)` した
`SequentialVars`（BatchNorm running stats 更新を伴う train forward）
を使い、部品間の加算・活性化・cross entropy・backward・optimizer 適用
までを `train_step` 内で手組みしている（`compat::Sequential::fit`
一括 API は使えない）。

### 10.4 `cross_entropy_mean`（facade 非公開の `Reduction` を避ける書き方）

`facade` は `Var::cross_entropy_loss` の `Reduction` 引数を
再エクスポートしていない（`docs/compat-api-scope.md`）。公開パス
だけで mean cross-entropy を得るため、`log_softmax` の出力から
`Var::gather` で正解クラスの log-probability のみを選択し、
`1/N` の定数（`tape.var` で tape に載せるだけの非学習対象）を掛けて
総和を取ることで、スカラー乗算 op を使わずに mean 相当を実現した
（`crates/facade/examples/models/reference_module.rs::
cross_entropy_mean`）。この定数への勾配は `Trainable::train_step` が
モデル内部パラメータだけを `trainable_grads` で抽出するため無視される。

当初は正解位置が `1/N`・それ以外が `0` の one-hot 定数を
`log_softmax` の出力と要素積してから総和する実装だったが、
`log_softmax` が非正解クラスに返す `-inf` と one-hot の `0` の積が
`0 * -inf = NaN` になり、正解クラスの loss が有限でも合計が NaN
汚染されうる不具合があった（イシュー #2202 PR #2325 レビュー
指摘）。`gather` で正解クラスの列のみを選択する現行実装は非正解
クラスの値に一切触れないため、この経路の NaN 汚染は起きない。

### 10.5 合成 CIFAR-10 相当データ（実 CIFAR-10 ではない）

- §6（合成 MNIST）と同じ方針で、実 CIFAR-10 は同梱せずネットワーク
  取得も行わない。`crates/facade/examples/models/synthetic_cifar.rs`
  がクラスごとに周波数・向き・チャネルバイアスが異なる正弦波縞
  パターンへ、サンプルごとの巡回平行移動（`±3px`）と一様ノイズ
  （振幅 `±0.1`）を乗せて `[N, 3, 32, 32]` 相当の行優先平坦データを
  生成する。
- `Transformer` 向けに `to_row_tokens` が `[N, 3, 32, 32]`
  （`(n, c, h, w)` 行優先）を `[N, 32, 96]`（`(n, h, c, w)` 順。1 行を
  1 トークン、チャネルを特徴次元へ連結）へ並べ替える。`Var::reshape`
  は非 contiguous な入力を拒否する（`crates/autodiff/src/var.rs`）
  ため、この並べ替えは `Var::permute` ではなくホスト側の `Vec<f32>`
  を tape に入れる前に並べ替える方式にした。

### 10.6 事前登録した判定式（AC4）と実測結果

手順（§7 と同型。判定式は結果を見る前に固定し、FAIL しても緩めない）:

1. モデルを固定シードで構築する。
2. 固定シードの `SplitMix64`（依存追加なしの局所 PRNG。`main.rs` は
   `bench_harness`〈非公開クレート〉を import できないため、
   `bench_harness::rng::Xorshift64Star` の代わりに使う）で train／
   test の合成データを生成する。
3. `Adam` で `EPOCHS` epoch 学習する（`fit_epochs`。shuffle しない
   固定順ミニバッチ）。
4. held-out（test）データで `accuracy` を計算する。

判定（`main.rs::check_ac4`）: 全 epoch の loss が有限・最終 epoch の
loss が初回 epoch の loss を下回る・held-out 精度が **0.50 以上**。
調整してよいのは判定式を確定する前の合成データの設計・N・batch
size・学習率だけであり、epoch 数（10。イシュー受け入れ条件が固定）・
判定式の係数（0.50）自体は変更していない。tolerance・baseline・
ガードレール閾値には一切触れていない。

| モデル | 構成 | N_train/N_test | batch_size | optimizer | epoch 数 | 実測 held-out 精度 | 判定 |
|---|---|---|---|---|---|---|---|
| `ResNet`（`main.rs`） | depth=8, width=8 | 64/32 | 16 | Adam(lr=5e-3) | 10 | 1.0000 | pass |
| `Transformer`（`main.rs`） | embed_dim=32, heads=4, layers=2 | 64/32 | 16 | Adam(lr=5e-3) | 10 | 0.9062 | pass |
| `ResNet`（統合テスト） | depth=8, width=4 | 32/16 | 8 | Adam(lr=8e-3) | 10 | ≥0.50（fail-closed 判定。実測は実行のたびに変動しうるため表には最小構成のみ記載） | pass |
| `Transformer`（統合テスト） | embed_dim=16, heads=2, layers=1 | 32/16 | 8 | Adam(lr=5e-3) | 10 | ≥0.50（同上） | pass |

統合テスト（`crates/facade/tests/example_resnet_cifar10.rs`・
`example_transformer_cifar10.rs`）は debug ビルドの実行時間予算
（CI `rust-ci / cargo test` ジョブの 20 分枠。`.claude/rules/ci.md`）
に収めるため、`main.rs` より小さい構成（`width`／`embed_dim`・
`num_layers`・データ件数を縮小）を使う。調整対象は §7 と同じ「判定式
確定前のデータ設計・N・batch size・学習率」のみで、epoch 数（10）・
判定係数（0.50）は変更していない。実測ではローカル環境で
`example_resnet_cifar10.rs`（6 テスト。10 epoch 学習テストを含む）が
約 18 秒・`example_transformer_cifar10.rs`（6 テスト）が 1 秒未満で
完了しており、CI 予算に十分収まる。

### 10.7 GPU parity（該当なし）

§8 と同じ理由（新しい演算・カーネルを追加していない。既存の
`compat::Sequential::add_*`・`Var::add`／`relu`／`mean`／`reshape` の
組み合わせのみ）で、CUDA／Metal での facade parity 実測・
`docs/perf/logs/` への申し送りは不要と判断した。

### 10.8 `ReferenceModule`（examples 限定 trait）と facade `nn::Module` の関係（事前評価）

イシュー #2338 は「`ReferenceModule` を使った代替が本 doc §10 に
書かれている」ことを前提としていたが、`docs/reference-models-decision.md`
本文には `ReferenceModule` という語は一度も出現しない（grep で確認
済み）。実際に `ReferenceModule` が定義・言及されているのは
`crates/facade/examples/models/reference_module.rs` 冒頭コメントと
`docs/README.md` の索引行のみである。本節はこの前提の食い違いを
記録したうえで、`ReferenceModule` と facade `nn::Module`（§10 承認
事項 1・案 B。`docs/facade-nn-module-exposure-decision.md` §5・§6）
との事前対応関係を整理する。

**(a) 代替の事実と理由**: `reference_module.rs` 冒頭コメントに明記
のとおり、facade の `nn::Module` は #2133 以降も承認待ちで保留中
（`crates/facade/src/lib.rs::NnModuleHoldDoctestGuard`）であり、
examples から内部の `fandhe_ai_autodiff::nn::Module` を impl する
ことも公開パス限定の方針により行わない。`ReferenceModule`／
`Trainable` は、この 2 つの制約の下で PyTorch `nn.Module` に似せた
層積層を示すための **examples 限定**の代替 trait である
（`crates/facade/examples/models/reference_module.rs:1-21`）。

**(b) 案 B との事前対応表**（`docs/facade-nn-module-exposure-decision.md`
§5・§6・§10 承認事項 2）:

| `ReferenceModule`／`Trainable` のメソッド | 案 B の対応 | 一致度 |
|---|---|---|
| `forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError>` | required method `forward`（§6 利用例） | シグネチャが同一（注: 2026-09-29 に案 B の `forward` の第 1 引数は `TapeRef<'t>` で確定したため〈#2394〉、「同一」の評価は #2403 で再確定する） |
| `named_parameters(&self) -> Vec<(String, &Tensor<f32>)>` | defaulted メソッド `named_parameters`（§10 承認事項 2） | 名前・意味論とも同じ |
| `set_training(&mut self, training: bool)` | defaulted メソッド `set_training`（同上） | 名前・意味論とも同じ |
| `is_training(&self) -> bool` | defaulted メソッド `training`（同上） | 意味論は同じだが**名前が異なる**（`is_training` vs `training`） |
| （なし。`set_parameter`／`state_dict`／`load_state_dict` は `ReferenceModule` に未実装） | defaulted メソッド `set_parameter`／`state_dict`／`load_state_dict`（同上） | `ReferenceModule` 側に対応なし |
| `Trainable::train_step(&mut self, x, y, opt: &mut Adam) -> Result<f32, AutodiffError>` | 対応なし（案 B は required／defaulted のいずれにも学習ステップを含まない。§6） | `Trainable` 固有。移行しても examples 側に残る見込み |

**(c) 結論（事前評価のみ・確定は承認後）**: 案 B が §10 で承認されれ
ば、`forward`・`named_parameters`・`set_training` の 3 メソッドは
シグネチャ・命名とも一致するためほぼ機械的に寄せられる見込みである。
`is_training` → `training` の改名と、`Trainable::train_step`（案 B
に対応なしのため examples 側に残る）だけが単純な置き換えでは済まない
差分になる。本節は評価のみであり、`ReferenceModule` から facade
`nn::Module` への実際の移行・置き換えは、案 B の採否が承認されたのち
に別イシューで行う（§9 のスコープ外一覧・`docs/facade-nn-module-
exposure-decision.md` §13.5 の「承認取得後に実施する変更範囲」参照）。
