# デバイス常駐更新の設計（イシュー #934・#933 ツリー第 1 段）

## 0. 位置づけ・スコープ

本ドキュメントは #933（親イシュー「学習ループのパラメータ更新をデバイス常駐化し
ホスト往復を削減する」）の 3 分解のうち **設計のみ**を担当する #934 の成果物である。
実装は #935、数値一致（parity）テスト・ベンチ非後退確認は #936 が担う。本文書は
両イシューがそのまま着手できる具体度（trait シグネチャ・契約・検証基準）で
更新経路（§2）・所有権（§3）・数値一致契約（§4）の 3 点を定める。

コード（`crates/`）・spec（`docs/spec/`）・CI・依存は本イシューでは一切変更しない。

## 1. 背景・実測根拠

フレームワーク横並びベンチ（PR #915・`scripts/bench/framework-compare/results/summary.md`・
計測 2026-08-28）で、MLP 学習 1 ステップ（784→256→10・ReLU・バッチ 64・MSE・SGD lr=0.01）が
以下のとおり fandhe-ai と candle/Burn とで 1 桁以上の差が実測されている。

| デバイス | fandhe-ai | candle | burn | 出典行 |
|---|---|---|---|---|
| CUDA（環境 2・DGX Spark GB10） | 2.418 s | 268.1 µs | — | summary.md:165-166 |
| CPU（環境 1・Apple M4 Max） | 18.185 ms | 797.5 µs | 626.5 µs | summary.md:74-76 |
| Metal（環境 1） | 48.845 ms | 751.8 µs | 1.606 ms | summary.md:77 |

summary.md 自身の注記（191 行目）が指摘するとおり、この差の支配要因は 2 つに
分解できる:

- **(a) 毎ステップの tape（CUDA コンテキスト等）初期化コスト**: fandhe-ai の
  計測プロトコルは「計測ごとに新しい `tape_for(Device::Cuda(0))` を作る」ため、
  N 非依存の約 440〜460 ms（CUDA。summary.md:191）の固定オーバーヘッドが毎回計上
  される。この要因は phase-2 #922 系（#929/#930 完了・#931 設計中）が担当する。
- **(b) パラメータ更新のホスト往復**: 本ドキュメント（#933 ツリー・#934/#935/#936）
  が担当する要因。(a) を除去しても本要因は独立に残る（tape を使い回しても、
  学習ステップごとの param/grad のホスト⇔デバイス転送は別途発生する）。

本ドキュメントは (b) のみを扱う。(a) との合算値（2.418 s 等）を (b) 単独の
定量的な削減目標として引用しない（両要因が未分離のため）。(b) の定量的な
影響度は #935 実装後、#936 のベンチで tape 再利用モード（`--mode reuse`。
`gemm` タスクのみ現状対応）に相当する形の計測を検討する。

## 2. 現状整理

| 項目 | 現状 | 根拠 |
|---|---|---|
| パラメータ保持 | `nn::Linear` がホスト `Tensor<f32>` を所有。毎ステップ `Linear::from_parameters` で新規構築（イミュータブルな丸ごと差し替え運用） | `crates/autodiff/src/nn/linear.rs`・`crates/autodiff/src/optim/sgd.rs:20` コメント |
| 更新計算 | `optim::Sgd::step` がホスト側 `Tensor<f32>` ループで `torch.optim.SGD` の更新則（weight_decay → momentum → nesterov → 減算の式順序）を計算し、更新後 `Vec<Tensor<f32>>` を返す関数型 API | `crates/autodiff/src/optim/sgd.rs:1-23`（アルゴリズム引用）・`:183-200`（`step` シグネチャ・位置対応契約） |
| 書き戻し | `Sequential::apply_parameters`（two-pass: 1 パス目で全層の shape 完全一致検証込み `Linear` 再構築、2 パス目で代入。#426 の置換前 shape 完全一致契約） | `crates/facade/src/compat/sequential.rs:229-` |
| デバイス転送抽象 | `MemoryOps`（`alloc_zeroed`/`upload`/`download`）と `DeviceBuffer<f32>`（`Box<dyn BufferHandle>` 経由の不透明ハンドル・RAII 解放・空テンソル契約・download 同期契約）は `tensor-core::buffer` に実装済みだが、`BackendOps` の supertrait としては未結線（TASK-1.9c 時点で意図的に未結線。`backend_ops.rs` 冒頭コメント） | `crates/tensor-core/src/buffer.rs:1-13`・`crates/tensor-core/src/backend_ops.rs:10-27` |
| カーネル入口 | `BackendOps` はホスト `Tensor<f32>` 入出力契約。CPU 実装は追加転送コストなし、CUDA/Metal 実装は各メソッド内で H2D（`upload` 相当）→ カーネル → D2H（`download` 相当）を完結させる。GEMM 以外の一部カーネル（GPU 側 elementwise/reduction の一部）は `BackendError::Unsupported` fail-safe | `crates/tensor-core/src/backend_ops.rs:28-40`（trait 定義冒頭コメント）・`:83-103`（メソッド一覧） |
| 数値一致 | バックエンド間統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」・FMA 契約（CPU: `f32::mul_add`、GPU: 既定 FMA 契約） | `.claude/rules/coding-rust.md` |
| 隣接イシュー | #931（デバイスハンドル再利用の公開 API 設計。phase-2）・#932（optimizer facade 公開の設計判断）・#935（本設計の実装）・#936（parity・ベンチ非後退確認） | gh issue 参照 |

現行の学習 1 ステップの転送回数（概算・per-step）:

1. forward: 各層で `BackendOps::gemm`/`add`/`relu` 呼び出しのたびに引数テンソルが
   H2D、結果が D2H（CUDA/Metal 実装内部で毎回発生。`Tensor<f32>` 契約のため）
2. backward（`Tape::backward`）: 同様に演算ごとの転送 + 最終的にホスト `Tensor` の
   勾配列を返す
3. `Sgd::step`: ホスト `Tensor<f32>` 上で更新計算（デバイス転送なし。ただし
   入力の `params`/`grads` が既にホストにある前提＝上記 1・2 で download 済み）
4. `Sequential::apply_parameters` → `Linear::from_parameters`: 更新後パラメータを
   新規 `Linear` として保持（次ステップの forward で再度 upload される）

つまり **パラメータそのものは「ステップ開始時に前ステップの結果としてホストへ
download 済み → 今回ステップの forward で再度 upload」という往復を毎ステップ
繰り返す**。これが本ドキュメントが解消対象とする (b) の具体的な構造である。

## 3. 更新経路（design decision 1）

### 3.1 採用方針: `BackendOps` の非破壊拡張 + 段階的常駐化

`tensor-core::BackendOps` に対し、`gemm_bias_act`/`run_fused` 相当の**デフォルト
メソッド追加**（既存メソッドのシグネチャ変更なし。TASK-1.9c の非破壊拡張方針
〈`backend_ops.rs:19-27`〉を踏襲）として、in-place パラメータ更新の入口を追加する。

```rust
/// デバイス常駐パラメータの in-place SGD 更新入口（#934 設計・#935 実装）。
///
/// `param` を `grad`（および `velocity`。momentum 有効時のみ `Some`）から
/// 直接更新し、ホストへの往復を発生させない契約とする。既定実装は
/// fail-safe（§3.2）。CPU/CUDA/Metal 各バックエンドは自身のカーネルで
/// オーバーライドできる（デフォルト実装のままでも正しく動作する）。
fn sgd_step_device(
    &self,
    param: &mut DeviceBuffer<f32>,
    grad: &DeviceBuffer<f32>,
    velocity: Option<&mut DeviceBuffer<f32>>,
    config: &SgdStepConfig,
) -> Result<(), BackendError>;
```

`SgdStepConfig`（`tensor-core` 側に新設。`autodiff::optim::SgdConfig` の
lr・momentum・dampening・weight_decay・nesterov をそのまま保持する変換先の
設定構造体）は §1 のアルゴリズム引用（`sgd.rs:1-23`）と同一の意味を持つ
フィールド名にし、`autodiff` 側の変換関数（`impl From<&SgdConfig> for
SgdStepConfig` 等）は #935 が実装する。`tensor-core` は `autodiff` に依存しない
（既存の依存方向: `autodiff` → `tensor-core`）ため、`SgdStepConfig` は
`tensor-core` 側に独立定義し、`autodiff` 側が変換して渡す構成とする。

### 3.2 デフォルト実装の方針: fail-safe 合成（`Unsupported` 分岐にしない）

既存の未実装カーネル（GPU 側 elementwise/reduction の一部）は `BackendError::
Unsupported` を返す設計だが（`backend_ops.rs:28-33`）、`sgd_step_device` は
**それとは異なり「必ず成立するが遅い」合成をデフォルト実装とする**:

```text
デフォルト実装 = download(param) → ホスト参照実装（CPU と同一式順序・
                 f32::mul_add）で更新 → upload(param) で書き戻し
                 （velocity も同様に往復）
```

採否理由: `Unsupported` + 呼び出し側ホストフォールバックにすると、#935 の
実装順序（CPU → CUDA → Metal）の途中で「まだデバイスカーネルを実装していない
バックエンド」を使うコードパスが `sgd_step_device` を呼べず、`autodiff::optim`
側に「両方の経路（デバイス常駐 API と旧ホスト API）を呼び分ける分岐」を
恒久的に持たせることになる。デフォルト実装が fail-safe に成立していれば、
`autodiff::optim::DeviceParamStore`（§3.3）は常に `sgd_step_device` だけを
呼べばよく、各バックエンドが最適化されたカーネルを持つかどうかは実装詳細に
留められる。デフォルト実装は `MemoryOps` が既に提供する `download`/`upload`
のみで構成でき、`MemoryOps` を `BackendOps` の supertrait にする必要はない
（デフォルト実装の実装側で `MemoryOps: Sized` 経由の呼び出しに閉じる、あるいは
`sgd_step_device` のデフォルト実装を持つ trait 側で `Self: MemoryOps` の
where 句を付ける形を #935 で確定する）。

### 3.3 段階設計: 「param + optimizer 状態のみ常駐、勾配は毎ステップ 1 回 upload」

現行 `Tape::backward` はホスト `Tensor` を返す（forward/backward の完全
デバイス常駐化は `DeviceBuffer` 版 `BackendOps`〈`upload`/`download` を
含む全面移行〉が前提であり、変更範囲が 4h 粒度を大きく超えるため #931・
phase-4（#924）へ接続するスコープ外事項とする。§5(b) 参照）。

そのため第 1 段（#935 スコープ）は次の分担とする:

- **常駐化する**: パラメータ本体（`weight`/`bias`）・momentum バッファ
  （`velocity`）。学習セッション開始時に 1 回 `upload` し、以降は
  `sgd_step_device` の in-place 更新のみで完結させる。
- **常駐化しない（毎ステップ 1 回転送）**: forward/backward の入出力
  （入力バッチ・中間活性・勾配）。`Tape::backward` が返すホスト勾配を
  `sgd_step_device` 呼び出し直前に 1 回だけ `upload` する。

これにより、現行「forward での param upload → backward 後の grad
download → ホスト更新 → 次ステップの param 再 upload」という毎ステップの
param 往復（§2 末尾）が「学習開始時の 1 回の upload + 終了時（または明示
同期時）の 1 回の download」に縮退する。grad の upload は本段階では残る
（forward/backward 自体が `Tensor<f32>` 契約のカーネルを呼ぶ限り、勾配は
一度ホストへ出てくるため）。

### 3.4 クレート責務分担

| クレート | 責務 |
|---|---|
| `tensor-core` | `sgd_step_device` trait メソッド・`SgdStepConfig` 型定義（`BackendOps` の非破壊拡張） |
| `backend-cpu`／`backend-cuda`／`backend-metal` | 各バックエンドのカーネル実装（CPU: 逐次または `rayon` 並列 in-place・CUDA: NVRTC 1 カーネル・Metal: compute shader 1 個）。実装順序は CPU → CUDA → Metal（#935 引き渡し事項。§6） |
| `autodiff::optim` | `DeviceParamStore`（常駐ストア保持型。§3.3・§4 の所有者）・`SgdConfig → SgdStepConfig` 変換・`Tape` の勾配出力を `DeviceParamStore` へ渡す統合ロジック |
| `facade` | 公開面。#932 の optimizer facade 公開設計の結論に従属させる（本ドキュメントは compat 面の意匠を確定しない） |

### 3.5 既存 API との関係

`Sequential::apply_parameters`／`trainable_parameters`（ホスト `Tensor<f32>`
契約）は**公開契約として変更しない**（公開 API 非破壊はガードレール条件。
`.claude/rules/security.md` A08）。デバイス常駐経路（`DeviceParamStore` 経由の
学習ループ）は既存経路に追加する新経路であり、既存経路を置き換えない。
`DeviceParamStore` からホスト `Tensor<f32>` への同期（学習終了時・`predict`/
保存前）は明示 API（例: `DeviceParamStore::sync_to_host() -> Vec<Tensor<f32>>`。
確定シグネチャは #935 で行う）を介して行い、`apply_parameters` へ渡す形で
既存の書き戻し経路と接続する。

## 4. 所有権（design decision 2）

### 4.1 所有者と単一真実源

学習セッション中は `autodiff::optim::DeviceParamStore` が保持する
`DeviceBuffer<f32>` 群を**正**とする。`DeviceParamStore` の寿命は `Tape` とは
独立とする（毎ステップの tape 再生成・再初期化〈#922 系〉をまたいで生存する。
tape 初期化コスト削減〈(a)〉とパラメータ常駐化〈(b)〉が互いに独立した最適化
であることの表れでもある）。

`nn::Linear` が保持するホスト `Tensor<f32>` は、学習セッション中は
「セッション開始時にアップロードした初期値のスナップショット」であり、
`DeviceParamStore` 側の更新を都度ミラーしない（stale になり得る）。誤って
stale なホスト値を「現在のパラメータ」として使う事故を防ぐため、次の契約と
する:

- 学習ループ内で `Linear`（または `Sequential`）のホスト `Tensor` を直接
  読み取る API（`weight()`/`bias()`/`trainable_parameters()`）は、
  `DeviceParamStore` 使用中は「最後に明示同期した時点のスナップショット」
  であることをドキュメンテーションコメントで明記する（#935 で追記）。
- 明示同期 API（§3.5 の `sync_to_host`）を呼んだ直後のみ最新値であることが
  保証される。同期タイミングは「学習終了時」「`predict`/保存直前」の 2 箇所
  を最低限の契約とし、#935 の実装がこれ以外の暗黙同期（例: 勝手なタイミングで
  自動 download する等）を行わないことを fail-closed の前提とする（暗黙
  コピー禁止方針。`docs/public-api-design.md` §2.2.1 と同じ思想）。

### 4.2 fail-closed 検査

`sgd_step_device` 呼び出し前に `DeviceParamStore` が担う検証（`Sgd::step`
既存契約〈`sgd.rs:183-200`〉と同水準を装置常駐版へ引き継ぐ):

- **デバイス一致検査**: `param.device() == grad.device()`（`velocity` がある
  場合は同様に検査）でなければ `BackendError::DeviceMismatch` を返す
  （`buffer.rs` の `DeviceBuffer::device()` を用いる）。
- **位置対応契約**: `params.len() != grads.len()` は `InvalidArgument`。
  momentum 有効時、2 回目以降の呼び出しで件数・各 `DeviceBuffer` の
  `shape()` が前ステップと変化した場合も `InvalidArgument`（`sgd.rs:196-215`
  の既存検証と同じ判定を `DeviceBuffer` 版に移植する）。
- **shape 保存契約**: `param` の `shape()` は呼び出し前後で不変（in-place
  更新であり形状変化は想定しない）。`Sequential::apply_parameters` の
  #426（置換前 shape 完全一致検証）と同水準の検査を、常駐経由の更新でも
  維持する。

### 4.3 アトミック性

`Sequential::apply_parameters` の two-pass 方式（1 パス目で全層の検証込み
再構築、2 パス目で代入。`sequential.rs:229-` 以降のコメント）を in-place
更新にも適用する契約とする:

```text
検証フェーズ: 全パラメータ・全 velocity の shape・device 一致を検査
              （1 件でも不一致なら早期 return、どのバッファも未更新）
更新フェーズ: 検証済みの全件について sgd_step_device を呼ぶ
```

検証フェーズを更新フェーズより前に完全に終わらせることで、途中エラーに
よる「一部パラメータのみ更新済み」の混在状態を防ぐ（#935 の
`DeviceParamStore::step` 実装が満たすべき契約として明記する）。

### 4.4 解放・スレッド境界

- **RAII 一本化**: `DeviceBuffer`/`BufferHandle` の既存方針（`buffer.rs`
  「解放方針」節）をそのまま踏襲する。`DeviceParamStore` に明示 `free()`
  は設けない。
- **Send/Sync**: 現行の `DeviceBuffer`/`BufferHandle` が `Send`/`Sync` を
  過剰に約束しない方針を維持し、`DeviceParamStore` も同様とする（#935 で
  必要になった時点で最小限のみ追加する）。
- **空テンソル契約**: `numel() == 0` のパラメータ（想定は稀だが、テストの
  境界ケース等）は `buffer.rs` の空テンソル契約（FFI を呼ばない空ハンドル）
  をそのまま適用する。

## 5. 数値一致契約（design decision 3）

### 5.1 参照実装と判定基準

CPU ホスト参照実装（既存 `optim::Sgd::step` の式順序・`f32::mul_add`。
`sgd.rs:1-23` のアルゴリズム）を正とし、デバイス常駐更新後のパラメータとの
比較は既存の統一複合判定を**そのまま適用する**:

> 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満（`.claude/rules/coding-rust.md`）

**新しい tolerance を新設・緩和しない**。変更が必要と判明した場合は
ユーザー承認必須（`.claude/rules/coding-rust.md`「テスト・ベンチ」節・
`.claude/rules/security.md`「自己修復ループ固有のガードレール」節と同じ
判断軸を適用する）。

### 5.2 FMA 契約

更新カーネル（CPU/CUDA/Metal のいずれも）は既存の丸め方針統一
（`.claude/rules/coding-rust.md`「バックエンド構成」節）に従う: CPU 参照
実装は `f32::mul_add`、GPU 側は各バックエンドの既定 FMA 契約
（CUDA NVRTC・Metal `simdgroup_multiply_accumulate` 相当）と揃える。

更新式の演算順序は §1 のアルゴリズム引用（weight decay 加算 → momentum
更新 → nesterov 分岐 → 減算）を全バックエンドで同一に固定する。順序が
バックエンド間で異なると、複合判定の閾値内であっても丸め誤差の蓄積パターンが
変わり得るため、#935 実装レビューで式順序の一致を明示確認する。

### 5.3 累積 parity の判定方式

単一ステップの判定（§5.1）に加え、#936 の parity テスト設計への引き渡し
事項として次を定める:

- **多ステップ累積判定**: 決定的シード（既存の決定的シード設定ユーティリティ
  を使用。`.claude/rules/coding-rust.md`「テスト・ベンチ」節）で 100 step
  程度学習を回し、CPU ホスト参照実装で計算した最終パラメータと、デバイス
  常駐経路（CUDA/Metal）で計算した最終パラメータを比較する。判定基準は
  単一ステップと同じ複合判定を最終値に適用する（中間ステップごとの判定は
  必須としない。誤差蓄積を最終値で捕捉する設計）。
- **ベンチ様式**: 5 回計測の中央値を採用する既存方針（`.claude/rules/
  coding-rust.md`「テスト・ベンチ」節）を #936 のベンチ（常駐化前後の
  per-step 時間比較）にも適用する。

### 5.4 境界検査

更新カーネルでも手動境界チェックを省略しない（REQ-8 境界検査規約。
`.claude/rules/coding-rust.md`「カーネル実装の境界検査」節）。性能目標
達成を理由にベクトル化ロード等で境界検査を省略する場合は、シェーダ側で
手動境界チェックを維持したうえで行う（既存 GEMM カーネルと同じ制約）。

## 6. 代替案と採否

| 案 | 内容 | 採否 |
|---|---|---|
| (a) 暗黙常駐キャッシュ方式 | `Tensor` 内部にデバイスキャッシュを持たせ、書き込み時に暗黙で同期する方式（candle の `Var::set` 型） | **不採用**。暗黙コピー禁止方針（`docs/public-api-design.md` §2.2.1）・型分離（ホスト `Tensor` とデバイス `DeviceBuffer` を明確に分ける現行設計）との不整合。stale 値の誤用検知も難しくなる |
| (b) `BackendOps` 全面 `DeviceBuffer` 化を先行 | forward/backward を含む全カーネル入口を `DeviceBuffer` 契約へ一気に移行してから更新経路を設計する | **不採用（本イシュースコープでは）**。変更範囲が 4h 粒度を大きく超え、#931（デバイスハンドル再利用の公開 API 設計）・phase-4（#924）と重複する。接続点のみ §3.3 に記録し、本設計は非破壊拡張 + 段階常駐化（案 c）で先行する |
| (c) 採用案 | §2〜§4 の非破壊拡張（`sgd_step_device` デフォルトメソッド）+ 常駐ストア（`DeviceParamStore`）+ 段階的常駐化（param/velocity のみ常駐、grad は毎ステップ 1 回 upload） | **採用** |

## 7. スコープ境界・受け渡し

### #935（実装）への引き渡し事項

- §3.1 の確定シグネチャ（`sgd_step_device`／`SgdStepConfig`）をそのまま
  `tensor-core::backend_ops` へ追加する。
- 実装順序: CPU → CUDA → Metal（既存 GEMM カーネルの実装順序・PoC 実証の
  蓄積と揃える）。
- フォールバック契約: §3.2 のデフォルト実装（download → ホスト更新 →
  upload）を先に用意し、各バックエンドが最適化カーネルでオーバーライド
  する形で段階的に置き換える（デフォルト実装のままでも正しく動作する
  ことを前提に、性能改善は独立した最適化として進められる）。
- `DeviceParamStore` の具体的な API（`new`/`step`/`sync_to_host` 等の
  シグネチャ）・`autodiff::optim` 内の配置（既存 `optim::Sgd` との関係。
  `Sgd` を置き換えるのではなく、`DeviceParamStore` が内部で §5.1 の判定に
  使う参照実装として `Sgd::step` 相当のロジックを再利用する形を想定）は
  #935 側で確定する。
- 更新カーネル（CUDA NVRTC・Metal compute shader）の `unsafe` 使用は
  FFI 境界に限定し、理由コメント + security-auditor レビュー必須
  （`.claude/rules/security.md`「unsafe」節）。

### #936（parity・ベンチ非後退確認）への引き渡し事項

- §5.3 の累積 parity 判定方式・ベンチ様式をそのまま使用する。
- 常駐化前後（旧経路 `Sgd::step` + `apply_parameters` 経由 vs 新経路
  `DeviceParamStore`）で per-step 時間を比較し、(b) の要因分離が有効に
  機能しているかを確認する（(a) の tape 初期化コストとは独立に、この
  比較単体で改善が見えることが望ましいが、#931 の tape 再利用が未着手の
  間は改善量が (a) に隠れる可能性がある点に留意する）。

### #931／#932 との整合前提

- #931（デバイスハンドル再利用の公開 API 設計）が確定するデバイス/tape
  再利用 API と、本設計の `DeviceParamStore`（tape 非依存の寿命）は独立
  だが、#935 の実装時に「tape 再利用時に `DeviceParamStore` をどう束ねるか」
  の結線点が生じる。本ドキュメントはこの結線の具体設計を確定しない
  （#931 側の結論を待つ）。
- #932（optimizer facade 公開の設計判断）が確定する公開面の意匠に
  `facade` 側の統合を従属させる（§3.4）。

### spec 更新の要否

本設計は REQ-2（数値一致複合判定・FMA 契約）・REQ-8（境界検査）の枠内で
完結し、`docs/spec/` の要件・受け入れ基準の変更を要しない見込みである。
実装中に spec 変更が必要と判明した場合は、`docs/spec/` を直接編集せず
正本リポジトリ（`Fandhe-AI/rust-ai-library-spec`）側での対応をユーザーへ
提案する（`.claude/rules/out-of-scope-tracking.md`「仕様変更が必要な場合」節）。

### スコープ外事項（本ツリー #933 の対象外）

- forward/backward の完全デバイス常駐化（`DeviceBuffer` 版 `BackendOps` への
  全面移行）— §6(b) 参照。接続先: #931・phase-4（#924）。
- f16 経路への拡張。
- optimizer 公開面（`facade` 側の意匠）の変更 — 接続先: #932。
