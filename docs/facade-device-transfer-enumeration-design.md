# `Tensor`／`Var` の `.to(device)` と `Device::available()` 列挙の設計・実装記録（イシュー #1614）

## 0. 承認状況・前提

issue コメント（2026-09-12・ユーザー承認）で、facade 公開面（`fandhe_ai`／`compat`）の
`docs/compat-api-scope.md` §5 手続きに基づく範囲拡張・公開クレート `fandhe-ai-tensor-core`
の `BackendOps` trait 拡張が承認済み。前提 spec 改定（spec PR #69）・#1591（PR #1661）は
マージ済み。`docs/compat-api-scope.md` §1.2 Tier 1 に「デバイス転送と列挙 | #1614」
（行 232）が既に列挙済みのため §5 の再適用は不要（#1750／#1748 と同じ整理）。

承認範囲外（本 issue では行わない）: tolerance 定数・`BASELINES` 等 baseline 値の変更、
依存クレートの追加・更新、`unsafe` の追加（本実装は `unsafe` を一切使わない）。
`docs/spec/` は編集しない。`=0.8.0` 版数ピンは不変。

## 1. 背景・目的

`docs/public-api-design.md` §4.1 は `Device::available()`（実行時に利用可能な
デバイスの列挙）を定めたが、TASK-1.9a（#44）実装時に「`tensor-core` から 3
バックエンドを直接参照できない」ため `enumerate_all(providers)`（依存逆転）を
同等機能として提供するに留め、**集約入口（どの層で結線するか）は未決のまま**残って
いた（`crates/tensor-core/src/device.rs` モジュール冒頭コメント・
`docs/compat-feature-gap.md` §1.8 device 節・§2.13）。

また `torch.Tensor.to(device)` 相当の「テンソル単体の明示転送 API」も存在せず、
`tape_for(device)` によるバックエンド単位の切替のみだった（§2.13 行 2）。

本 issue はこの 2 つの未決事項を、既存アーキテクチャ（下記 2 節の不変条件）を壊さずに
解消する:

- **列挙**: facade（composition root）に 3 バックエンドの `DeviceProvider` を束ねた
  列挙入口 `fandhe_ai::available_devices()` を追加する。
- **転送**: `Var` に「同一デバイスなら恒等」「別デバイス（別 `Tape`）へは値を転送して
  新しい葉として登録」の 2 段構成の API（`Var::to`／`Var::to_tape`）を追加し、
  `Tensor` については設計上デバイス常駐を持たないことを本文書で確定する。

## 2. 現状調査で確定した制約（設計の前提）

| # | 事実 | 出典 |
|---|------|------|
| C1 | `Device` は `tensor-core` の型。facade から inherent メソッド（`Device::available()`）を追加できない（orphan rule）。既存の `fandhe_ai::release_cached_memory(device)`／`memory_pool_stats(device)` は同じ理由で自由関数として公開している | `crates/facade/src/lib.rs` `release_cached_memory` doc・`docs/facade-device-handle-design.md`「案 B のみ採用」 |
| C2 | `tensor-core::device::enumerate_all(&[&dyn DeviceProvider]) -> Vec<DeviceInfo>` が既に存在し、provider の `Err` は捨てて他 provider の結果を返す fail-safe 方針 | `crates/tensor-core/src/device.rs` |
| C3 | `CudaDeviceProvider::enumerate` は `CudaDevice::device_count()` 失敗（driver 不在＝cudarc 動的ロード失敗）で `Ok(vec![])` を返す。`probe` は `context_cache::cached_device(ordinal)` を経由するため、列挙は ordinal ごとに CUDA コンテキストを（初回のみ）初期化する副作用を持つ | `crates/backend-cuda/src/device.rs` |
| C4 | `CpuDeviceProvider::enumerate` は常に `Device::Cpu` 1 件。`MetalDeviceProvider::enumerate` は `probe_all()`（0 件なら空） | `crates/backend-cpu/src/device.rs`・`crates/backend-metal/src/device.rs` |
| C5 | `Tensor<T>` はホスト常駐固定（`Storage { data: Vec<T> }`）でデバイス情報を持たない。`Tape::var` もアップロードせず、転送は演算ごとに各 `BackendOps` 内部で行う。デバイス常駐は `DeviceParamStore`／`Op::ResidentLeaf` のみ | `crates/tensor-core/src/tensor.rs`・`docs/public-api-design.md` §2.1／§4.2 |
| C6 | `Var<'t>` は `&'t Tape` を借用し、`Tape` は 1 つの `Box<dyn BackendOps + Send>`（＝1 デバイス）を所有する。異なる `Tape` 間の二項演算は `AutodiffError::TapeMismatch`。`Tape::backward` はテープ単位で、テープをまたぐ勾配伝播機構は存在しない | `crates/autodiff/src/var.rs`・`crates/autodiff/src/tape.rs` |
| C7 | `Tape::ops()` は `pub(crate)`。facade は `BackendOps`／`MemoryOps`／`DeviceBuffer` を利用者向け公開面へ露出しない（REQ-12・spec `04-requirements.md`） | `crates/facade/tests/api_surface.rs` |
| C8 | `Tape::push_leaf(value, requires_grad)` は葉プレフィックスを固定しない（固定は `push_eager`／`push_lazy`／`push_view` のみ）。非葉ノードの後に push した葉は `Tape::reset` で truncate される（既存 `Tape::var` と同じ挙動） | `crates/autodiff/src/tape.rs` |
| C9 | `BackendOps::device(&self) -> Device` は必須メソッドとして全実装（CPU／CUDA／Metal・`NaiveOps`・テスト用 ops）に存在する | `crates/tensor-core/src/backend_ops.rs` |
| C10 | 同種の直近実装（#1750 cast・#1748 no_grad/detach）は新規 `Op`／`BackendOps`／VJP を追加せず、受け入れ条件テンプレートの「Op／VJP／parity」項目を「非微分境界・恒等」として設計文書で整理して満たしている | `docs/tensor-core-cast-design.md` §2 |

## 3. 設計（確定事項）

### 3.1 デバイス列挙: `fandhe_ai::available_devices() -> Vec<Device>`

facade の自由関数として追加（C1 の precedent。`Device` は `tensor-core` 由来の
外部型のため orphan rule により inherent メソッド形は採れない）。`tensor-core`・
`autodiff`・`backend-*` は変更しない。

実装: `CpuDeviceProvider::new()`・`CudaDeviceProvider::new()`・
`#[cfg(target_os = "macos")] MetalDeviceProvider::new()` を `Vec<&dyn DeviceProvider>`
（cfg により要素数が変わるため cfg ごとに構築を分ける——`mut` にして条件付き `push`
する形は非 macOS ビルドで `unused_mut`〈`-D warnings` で fail〉になるため採らない）
に束ね `enumerate_all` を呼び、`DeviceInfo::device` のみを収集する。

**順序契約**: `Device::Cpu` → `Device::Cuda(0..n)`（ordinal 昇順）→ `Device::Metal`
（macOS のみ）。同一プロセス内で再呼び出ししても順序は決定的。

**失敗契約**: `panic!`／`unwrap()` なし。CUDA driver 不在（cudarc 動的ロード失敗）は
C3 により CUDA 分が空になり、`Device::Cpu` は常に含まれる。個々の provider の `Err`
は C2 により捨てられる（設計どおり。doc comment に明記）。

**副作用の明記**（doc comment）: C3 により、列挙は検出した CUDA ordinal ごとに
`context_cache` のコンテキストを初期化する（2 回目以降はキャッシュヒット。
`tape_for(Device::Cuda(i))` が後で払うコストを前倒しするだけで追加コストではない）。

**一貫性契約**: 返された各 `Device` は `tape_for(d)` が `Ok` になることを意図する
（ただし列挙と `tape_for` の間でデバイス状態が変わる TOCTOU は契約外と明記）。

`DeviceInfo`／`DeviceProvider`／`enumerate_all`／`select_from` は facade から
再エクスポートしない（REQ-12「`Device` 識別子のみ」の最小公開面。`Vec<DeviceInfo>`
版はスコープ外・後続提案）。

`docs/public-api-design.md` §4.1 の `Device::available()` という inherent メソッド形
からの逸脱（orphan rule）は §4「対応」で追記した。

### 3.2 `Var` の転送 API（2 段構成）

`autodiff` クレートに以下を追加した（facade は既存の `Var` 再エクスポート経由で
到達。`fandhe_ai::Tape` newtype には薄い委譲メソッドを追加）。

1. `Tape::device(&self) -> Device`（`pub`）: `self.ops.device()` への委譲。`ops()`
   自体は `pub(crate)` のまま（REQ-12）。facade `Tape::device()` ラッパーを追加。
2. `Var::device(&self) -> Device`（`pub`）: `self.tape.device()`。
3. `Var::to(&self, device: Device) -> Result<Var<'t>, AutodiffError>`（`pub`）:
   - `device == self.device()` のとき `Ok(*self)`（恒等。`Var: Copy`。既存の
     `Var::to_f32` と同型で勾配は通常どおり伝播・tape にノードを追加しない）。
   - それ以外は `Err(AutodiffError::DeviceMismatch { requested: Device, actual: Device })`
     （新規 variant。`#[non_exhaustive]` への非破壊追加・`Display` 実装追加。単位
     variant の `BackendError::DeviceMismatch` を包む案は「要求先・実際のデバイス」の
     情報を失うため不採用）。エラーメッセージで `Var::to_tape`／`Tape::transfer` への
     誘導を書く。
   - 理由: `Var` は所属 `Tape` の 1 デバイスに束縛され（C6）、同一 tape 内でデバイスを
     変える演算は表現不能。PyTorch 風の `x.to(device)` を「整合していれば no-op・
     不整合なら型付きエラーで即失敗」という fail-fast の検査として提供する。
4. `Var::to_tape<'u>(&self, target: &'u Tape) -> Result<Var<'u>, AutodiffError>`（`pub`）:
   別 `Tape`（＝別デバイスでも同一デバイスでもよい）への値の転送。
   - `target.id == self.tape.id`（`TapeId: PartialEq`。`check_same_tape` と同じ比較）
     なら `Ok(*self)`（恒等）。**この同一 tape 判定は `self.tape.nodes.borrow()` より
     前に行う**——後回しにすると、同一 tape の場合に不変借用が生きたまま
     `target.push_leaf`（`borrow_mut`）を呼び `RefCell` の二重借用 panic になる。
     早期 return により二重借用は到達不能になる（`Var::detach` は常に同一 tape へ
     push するためスコープ分離だけで済んでいる点が異なる）。
   - それ以外は `materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()`
     で**転送元デバイスの ops** で実体化（lazy elementwise 連鎖・view はここで確定。
     checkpoint 解放済みで再計算に失敗した poison 値は `Err` で fail-closed。
     `Var::detach` と同じ規則）した `Tensor<f32>` を、`target.push_leaf(value,
     self.requires_grad())` で新しい葉として登録する。`Tensor` は `Arc` 共有のため
     clone は実データコピーなし。
   - 数値契約: 転送はホスト `Tensor<f32>` の値をそのまま（算術なし）渡すため、転送元で
     実体化された値と `to_tape` 後の葉の値は**bit 完全一致**。
   - 勾配契約: 勾配はテープをまたがない（C6。`Tape::var_from`／`detach` と同じ
     「非微分境界」）。転送先では通常の葉として勾配を受け取る（`requires_grad` は
     転送元を引き継ぐ）。転送元 `Var` の勾配経路は影響を受けない。
   - `Tape::reset` 契約: 転送先で既に非葉ノードが積まれている場合、この葉は `reset`
     で破棄される（C8。既存 `Tape::var` と同じ）。doc comment に明記。
5. facade `Tape::transfer(&self, source: &Var<'_>) -> Result<Var<'_>, AutodiffError>`:
   `source.to_tape(&self.0)` への薄い委譲。facade 利用者は `fandhe_ai::Tape` から
   生の `fandhe_ai_autodiff::Tape` を取り出せない（`.0` は `pub(crate)`）ため、この
   入口が facade での cross-device 転送の唯一の経路。`api_surface.rs` の既存検査
   （`pub fn` 行に `BackendOps` を含まない・compat 内で生の autodiff `Tape` を引数に
   取らない）と矛盾しない。

新規 `Op` variant・`BackendOps` メソッド・VJP は追加しない（恒等はノードを持たず、
cross-tape 葉には VJP が定義できない。C10 と同じ整理）。

### 3.3 `Tensor::to(device)` の扱い（設計上の非対応・API 追加なし）

C5 のとおり `Tensor<f32>` はデバイス常駐を持たず、`Tape::var` すら転送を行わない。
`Tensor` の「デバイスへの転送」は「そのデバイスへ結線した `Tape` への登録
（`tape_for(device)?.var(&t)`）」と等価であり、独立した転送 API を設ける意味がない。

`MemoryOps::upload`→`download` の往復で「デバイスを経由した `Tensor`」を返す自由
関数は、値が変わらず GPU メモリと転送時間を浪費し利用者を誤解させるため**設けない**。
`DeviceBuffer`／`MemoryOps` の公開面露出は REQ-12・spec で禁止（C7）。

したがって §2.13 行 2 の `Tensor` 側は「設計上の非対応（等価な経路の明示）」として
本文書・`docs/compat-feature-gap.md` 追補・`Tape::var`／`Var::to_tape` の doc comment
で確定する。

## 4. `docs/public-api-design.md` §4.1 との対応

同文書「TASK-1.9a 実装時の突合結果」の `Device::available()` 項に「集約入口は facade
の自由関数 `available_devices` として #1614 で結線（orphan rule により inherent 形は
採らない）」を追記した。既定デバイス選択ロジック（GPU 自動選択）は本 issue でも実装
しない（同文書 §6 未決事項 2 のまま）。

## 5. テスト

### (A) `crates/autodiff/tests/device_transfer.rs`（`common::naive_ops()` 使用）

- A1: `to(Device::Cpu)` on CPU tape は恒等（tape 長不変）で、`sum(x.to(Cpu) * x)`
  の勾配が `to` を省いた場合と bit 一致することを確認。
- A2: `to(Device::Cuda(0))` on CPU tape は `Err(DeviceMismatch { requested, actual })`。
  tape 長不変。
- A3: `to_tape` 同一 tape は恒等（`leaf_count()` 不変）。
- A4: 2 つの naive tape 間 `to_tape`: 値 bit 一致・転送先 `leaf_count()` が +1・
  転送先で `backward` すると葉が勾配を受け取る・転送元 tape の `backward` は転送の
  影響を受けない（勾配 bit 一致）。
- A4b: `requires_grad=false`（`var_no_grad`）の引き継ぎ確認（転送先でも
  `GradientTrackingDisabled`）。
- A5: lazy elementwise 連鎖（`add`→`mul`）の `to_tape` が転送前に実体化され
  `to_tensor()` と bit 一致。
- A6: `Tape::reset` 契約: 葉プレフィックス確定後に転送した葉は `reset` の対象。
- A7: checkpoint 解放済み・再計算失敗（poison）ノードの `to_tape` が `Err`
  （`crates/autodiff/tests/checkpoint_review_1624.rs::InstrumentedOps` と同型の
  最小フィクスチャ `FailingSecondGemmOps` を本ファイル内に複製）。

計 8 テストすべて green（Linux 実行）。

### (B) `crates/facade/tests/device_enumeration.rs`

`available_devices()[…]` が常に `Device::Cpu` を含む・全 `Device` で
`tape_for(d).is_ok()`・`CudaDeviceProvider::is_available()` と `Device::Cuda(_)` の
有無整合（ordinal 昇順連続も確認）・2 回呼んで同一 `Vec`・macOS 限定で
`MetalDeviceProvider::is_available()` との整合（`tape_construction.rs` の実行環境
適応型パターン）。計 5 テストすべて green（Linux・実機なし環境で実行。CUDA driver
不在のため `Device::Cuda(_)` は列挙されないことを確認済み）。

### (C) `crates/facade/tests/device_transfer_backend_parity.rs`

属性なし＝`fandhe_ai::tape()`（CPU）↔ `fandhe_ai_autodiff::Tape::new()`（naive CPU
参照実装）間の `Tape::transfer`／`Var::to_tape` で値 bit 一致（往復・lazy チェーン
実体化込み）。計 2 テスト green。

`#[ignore]`＝`tape_for(Device::Metal)`（`cfg(target_os = "macos")`）／
`tape_for(Device::Cuda(0))` で GPU 側で計算した `Var`（`matmul`）を CPU tape へ
`Tape::transfer` し、転送バイト自体は GPU 側 `to_tensor()` と bit 一致・CPU 計算値
とは REQ-2 統一複合判定（`fandhe_ai_backend_cpu::assert_parity`）で比較する。
**実機実測は本エージェント実行環境に CUDA／Metal 実機がないため未実施のまま
Mac／GB10 セッションへ申し送る。**

### (api_surface.rs 追加分)

`fandhe_ai::available_devices`／`Tape::device`／`Tape::transfer` の `pub fn` 存在
固定、`DeviceProvider`／`DeviceInfo`／`enumerate_all`／`select_from` の非再
エクスポート固定、`available_devices()` のコンパイル時型検証。既存検査
（`facade_does_not_reexport_tape_or_backend_ops`・`facade_public_functions_do_not_
accept_backend_ops_argument` 等）との非抵触を確認済み。

## 6. スコープ外（起票はユーザー承認後・本エージェントは起票しない）

- `compat::Sequential`／`nn::Module` レベルの `.to(device)`（モデル全体の移送）。
- `Vec<DeviceInfo>` を返す列挙（デバイス名・メモリ容量の公開）。
- `Var` 単位の真のデバイス常駐（`DeviceParamStore` 以外の resident 化）・テープを
  またぐ勾配伝播。
- 既定デバイス自動選択ロジック（§4.1 未決事項の継続）。
- CUDA／Metal 実機での `#[ignore]` テスト実測（Mac／GB10 セッションへ申し送り）。

## 7. 出典

- `docs/public-api-design.md` §4.1・§4.4・§6
- `crates/tensor-core/src/device.rs`
- `crates/autodiff/src/{error,tape,var}.rs`
- `crates/facade/src/lib.rs`
- `docs/facade-device-handle-design.md`
- `docs/compat-api-scope.md` §0・§1.2
- `docs/tensor-core-cast-design.md`（同型の設計判断の precedent）
