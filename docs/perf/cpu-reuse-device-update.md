# CPU reuse 学習 `device_update` の内訳切り分け（イシュー #2106）

DGX Spark GB10 の CPU reuse 学習で `device_update` が 277.5 µs（step_total の
25.7%）を占める件について、内訳を sub-phase に分解して支配項を特定するための
診断基盤の記録。**本書の時点では診断基盤のみ実装済みで、M4 Max／GB10 の 5 run
実測と仮説判定（§6）は未実施（実機セッションへの申し送り）**。本番コード・
tolerance・baseline・`Cargo.toml`・`docs/spec/` は変更していない。

## 1. 背景

- `docs/perf/train-step-phase-breakdown.md` §17.6.2: GB10 の CPU reuse で
  `device_update` 277.5 µs（25.7%）。§17.2: M4 Max は 122.3 µs（GB10 は約 2.3 倍）
- 同じ算術（`p - lr*g`）を fresh 経路でホスト Vec 上に行う `host_sgd` は GB10
  65.4 µs・M4 Max 33.6 µs。`device_update` は両機ともその約 4 倍で、構造上の
  要因は GB10 固有ではなく、GB10 で絶対値がさらに大きいと読める（仮説の出発点）
- `docs/perf/loss-attribution-matrix.md` の G-CPU-TRN 行は、内訳を「未確定」とする

## 2. 計測対象の経路（イシュー本文の記述の訂正）

イシュー本文は `crates/autodiff/src/optim/sgd.rs`（ホスト `Sgd::step`）を挙げるが、
reuse 経路には乗らない。bench-fandhe `measure_train_reuse_phases` の
`PHASE_DEVICE_UPDATE` 区間の実体は次のとおり。

```
Tape::step_device_param_store            (crates/facade/src/lib.rs。薄い委譲)
 → DeviceParamStore::step                (crates/autodiff/src/optim/device_store.rs)
 → CpuBackendOps::sgd_step_device        (crates/backend-cpu/src/ops.rs)
```

MLP 784→256→10・batch 64、total_numel = 200704+256+2560+10 = 203,530 要素。
`step` の実行内容:

| 段 | 内容 | 本診断の区間名 |
|---|---|---|
| prologue | poison・device・config 検査、pending／tape_id／epoch 照合、`Vec<Var>`・`Vec<bool>` 構築、host 経由 slot の `grads.get` と shape 検査 | in-situ の残差（H4） |
| alloc | `flat_grad = Vec::with_capacity(total_numel)`（約 814 KB）を毎 step 無条件に確保。CPU の resident 分岐では未使用 | `alloc` |
| stage | CPU は `gemm_fp32_strict_into_with_bias_reduce_tracked` を override せず bias 勾配が host 経由。bias（256+10 要素）を `upload_into` で staging へコピー | `stage` |
| sgd_kernel | `sgd_step_device` を total_numel 要素に 1 回。添字アクセスの逐次スカラーループ（境界検査付き）で compute と apply（in-place 書き戻し）は融合 | `sgd_kernel` |

momentum 0 のため velocity は確保されず、momentum 系の腕は作らない。

イシュー呼称との対応: alloc → `alloc`（と補助の `stage`）／sgd_compute →
`sgd_kernel`・`sgd_compute_split`／apply_params → **この経路では融合のため独立区間なし**
（非融合の写しの `apply_params_split` を書き戻しトラフィックの参考値とする）。

## 3. 仮説

- **H1 ループ形**: 境界検査付きの添字スカラーループが自動ベクトル化されず、1 要素の
  コストが大きい（`sgd_kernel` からループ外固定費 `sgd_kernel_fixed`〈同じ `sgd_step_device` を 1 要素で呼ぶ〉を引いた値と、境界検査なし zip 形 `sgd_kernel_zip` の差で見る）
- **H2 cache 状態**: backward で rayon ワーカーが staging を書き、forward で params を
  読んだ直後に main スレッドが両者を読み書きするため、コア間（GB10 は 2 クラスタ構成）
  で cache line が移動する（`insitu_direct` と `insitu_pretouch` の差で見る）
- **H3 ホスト側の確保**: `flat_grad` の未使用 capacity 確保・各種 Vec 確保・`grad.clone()`
- **H4 prologue の簿記**: 検査・`resident_filled_slots`・`grads.get`

## 4. 計測設計

`crates/facade/tests/cpu_reuse_device_update_diag.rs`（#2105 の
`cpu_predict_resident_fixedcost_diag.rs` と同型。公開 API のみで分解し、本番コードへ
計装を残さない）。

- **テスト 1（CI 実行）**: 実際に backward した直後の状態から、公開 API で更新前 params
  （`sync_device_param_store_to_host`）と勾配（`param_grads_to_host`）を再構成し、
  融合カーネル・非融合・zip 形の 3 つの写しが本物の `step_device_param_store` と
  **bit 完全一致**することを hard assert する。区間の帰属が本番と乖離した写しへ向かう
  のを防ぐ。あわせて CPU の resident 勾配 slot が weight=`Some`・bias=`None` である
  前提（stage 区間が存在する前提）と、pretouch が状態を変えないことを固定する
- **テスト 2（`#[ignore]`・実機用・record-only）**: `DIAG_JSON` を 1 行 1 区間で出力
  - in-situ: `insitu_direct`（backward 直後の step を計時）／`insitu_pretouch`（未計時で
    staging と params を読んでから step を計時）。2 腕は別 store（同一初期値・同一データ）
    を交互に 1 step ずつ進める
  - standalone: `alloc`／`stage`／`sgd_kernel`／`sgd_compute_split`／`apply_params_split`／
    `sgd_kernel_zip`／`sgd_kernel_fixed`（ループ外固定費）／`sgd_kernel_xthread`（書き込み元の勾配 Tensor を別スレッドで生成してから main が書き込んだ直後の融合カーネル。
    `DeviceBuffer` が `Send` でなく対象バッファへの書き込み自体は main のため、クロスコアの
    cache 移動は再現しない。H2 は in-situ 2 腕の差で見る。判定には使わない）
  - checksum は最終 params の bit ハッシュ（腕間・run 間一致を集計で fail-closed 検査）
- 判定規則は `logs/cpu-reuse-device-update-2106/RULE.txt`（実測前に固定）、集計は
  `aggregate.py`（`--self-test` あり）、実行は `orchestrate.sh`（独立 5 プロセス）

## 5. 判定規則

`RULE.txt` を正とし本節では書き写さない（T の 50% 以上を占める項を「支持」、届かなければ
「未確定」。閾値は事後に変えない）。

## 6. 結果

| ホスト | 状態 |
|---|---|
| GB10（5 run・load1 < 1.0 ゲート） | 未実施（実機セッションへ申し送り） |
| M4 Max（5 run・record_only） | 未実施（実機セッションへ申し送り） |

参考（判定外）: x86_64 Linux（Core i7-13700K・10 コア）の 5 run スモーク
`logs/cpu-reuse-device-update-2106/smoke-x86/`。`sgd_kernel` が約 171 µs（ループ外固定費 sgd_kernel_fixed は約 0.2 µs）に対し境界検査
なしの zip 形は約 29 µs で、この環境では H1（ループ形）が大きく見える。ただし対象機種・
コア構成・アロケータが GB10／M4 Max と異なり、RULE.txt のゲートも通していないため、
仮説判定にも §7 の対処採否にも使わない。

## 7. 対処候補（列挙のみ。起票はユーザー承認が必要）

いずれも bit 一致契約・FMA 契約（`mul_add` は weight_decay／momentum の項だけ）を
保てるか、後続で確認してから判断する。

1. `flat_grad` の `with_capacity` を `!any_resident` 分岐の中へ移す（H3）
2. `sgd_step_device` を zip／chunk 化して境界検査を除き自動ベクトル化させる。または
   rayon 化してしきい値付きの逐次フォールバックにする（H1。#2446 の縮約側機構と同型）
3. CPU で `gemm_fp32_strict_into_with_bias_reduce_tracked` を override し、bias 勾配の
   host 経由をなくす（stage 区間の削除）

## 8. 限界

- 診断は HEAD をビルドして計測するため、registry 0.9.0 のバイナリを使った §17.6.2 とは
  条件が異なる。§17.6.2 の 277.5 µs との乖離は参考値として記録する
- `param_grads_to_host` は staging をコピーして読み出すため、pretouch 自体が確保を伴う。
  pretouch は計時窓の外だが、アロケータの状態は変わりうる
- standalone 区間は step 全体の再現ではなく、prologue 相当の簿記コストは in-situ の残差
  （H4）でしか見えない
- `apply_params_split` の計時には `Tensor` 経由の `upload_into` を使うため、本番の融合
  経路の in-place 書き戻しより余分なコピーを含む参考値である
- 既存の REJECT／undetermined 実験（#1575・#1576・RAYON_NUM_THREADS スイープ・#2446・
  #2104）は再実行しない
