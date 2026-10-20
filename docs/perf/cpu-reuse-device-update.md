# CPU reuse 学習 `device_update` の内訳切り分け（イシュー #2106）

DGX Spark GB10 の CPU reuse 学習で `device_update` が 277.5 µs（step_total の
25.7%）を占める件について、内訳を sub-phase に分解して支配項を特定するための
診断基盤の記録。**GB10 の 5 run 実測と仮説判定は #3188 で実施済み（§11。pre／head の 2 系列）。
M4 Max の実測は未実施**。本番コード・
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
| alloc | `flat_grad = Vec::with_capacity(total_numel)`（約 814 KB）を確保。CPU の resident 分岐では未使用のため、#3194 以降は `any_resident` のとき確保しない（`!any_resident` のときだけ `with_capacity`）。standalone の `alloc` 腕は生の `with_capacity` を測る写しで本番経路ではない | `alloc` |
| stage | CPU は `gemm_fp32_strict_into_with_bias_reduce_tracked` を override せず bias 勾配が host 経由。bias（256+10 要素）を `upload_into` で staging へコピー | `stage` |
| sgd_kernel | `sgd_step_device` を total_numel 要素に 1 回。添字アクセスの逐次スカラーループ（境界検査付き。**#3193 以前の形**。以後は zip 反復）で compute と apply（in-place 書き戻し）は融合 | `sgd_kernel` |

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
| GB10（5 run・load1 < 1.0 ゲート） | 実施（#3188。pre／head の 2 系列。結果は §11） |
| M4 Max（5 run・record_only） | 未実施（実機セッションへ申し送り） |

参考（判定外）: x86_64 Linux（Core i7-13700K・10 コア）の 5 run スモーク
`logs/cpu-reuse-device-update-2106/smoke-x86/`。`sgd_kernel` が約 171 µs（ループ外固定費 sgd_kernel_fixed は約 0.2 µs）に対し境界検査
なしの zip 形は約 29 µs で、この環境では H1（ループ形）が大きく見える。ただし対象機種・
コア構成・アロケータが GB10／M4 Max と異なり、RULE.txt のゲートも通していないため、
仮説判定にも §7 の対処採否にも使わない。

## 7. 対処候補（列挙のみ。起票はユーザー承認が必要）

いずれも bit 一致契約・FMA 契約（`mul_add` は weight_decay／momentum の項だけ）を
保てるか、後続で確認してから判断する。

1. `flat_grad` の `with_capacity` を `!any_resident` 分岐の中へ移す（H3）。**実施（#3194、§9）**
2. `sgd_step_device` を zip／chunk 化して境界検査を除き自動ベクトル化させる（**#3193 で zip 化を実装。rayon 案は未採用**。§10）。または
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

## 9. #3194: resident 分岐の `flat_grad` 確保をやめた結果

`DeviceParamStore::step`／`step_adam_impl`／`step_rmsprop`／`step_adagrad`／`step_lamb` の
5 か所で、`flat_grad` を `any_resident` のときは `Vec::new()`（確保なし）にした。後段で
消費するのは `!any_resident` 分岐だけで、その側の容量・積む順序は不変（数値・公開 API 不変）。

- 確保量テスト `crates/facade/tests/device_param_store_resident_no_flat_grad_alloc.rs`
  （`TrackingAllocator`・MLP 784→256→10・2 step 後の 3 step 目。時間計測ではなく確保の証明）
  - 修正前（main）: SGD の step 確保ピーク 814,532 B（`flat_grad` = 814,120 B）で失敗
  - 修正後: SGD・Adam とも 412 B で成功
- bit 一致: 混在は既存の facade `device_param_store_*_train`・autodiff の
  `resident_weight_grad_merge_matches_host_only_grad_path`、resident のみは新規
  `resident_only_weight_grad_matches_host_only_grad_path`（SGD・mock）、なしは確保式が
  不変のため構造上不変
- GB10 の前後は #3188 で測定済み（§11。ただし #3193 と同じ系列間に入るため分離不能）。
  Mac mini の記録は未実施。測定手段は `cpu_reuse_device_update_diag.rs` の
  テスト 2（`#[ignore]`）と `logs/cpu-reuse-device-update-2106/orchestrate.sh`


## 10. #3193 zip 化（`sgd_step_device`）

- 変更: `crates/backend-cpu/src/ops.rs` の `sgd_step_device` を「検証 → スライス確定 → `sgd_step_slices`（zip 反復）」へ再構成。ハンドル実体の長さ（param・grad・velocity）を書き込み前にすべて検証する（REQ-8 の手動境界検査は維持。不一致は `ElementCountMismatch` で、更新前の状態を保つ）。評価順・FMA 契約（`mul_add` は weight_decay／momentum／nesterov の項のみ、`p - lr*g` は 2 演算）は不変
- 回帰: `crates/backend-cpu/tests/sgd_device_bit_identity.rs` が #3193 以前の添字ループの逐語の写しと bit 一致（NaN はクラス一致）を、momentum・dampening・weight_decay（-0.0 含む）・nesterov・is_first_step・長さ 0〜203,530・特殊値入りで検証する

### 前後比較（device_update。MLP 784→256→10・203,530 要素）

| 機体 | 版 | insitu_direct device_update 中央値 | sgd_kernel 中央値 | 状態 |
|---|---|---|---|---|
| x86_64 Linux（Core i7-13700K・共有環境・record_only） | before = `1a7eb9a2`（origin/main） | 269.1 µs（範囲 192.6-278.8） | 197.3 µs | 参考値（判定外・5 run 独立プロセス） |
| 同上 | after = 本変更 | 42.7 µs（範囲 41.5-43.7） | 34.2 µs | 同上 |
| GB10（#3188。5 run・負荷ゲート全 run pass） | before = `17c7767b5`（#3193・#3194・#3196 の直前） | 279.1 µs（範囲 270.9-281.0） | 107.1 µs | 実測。#3193 と #3194 を含む区間の合算（§11） |
| GB10（同） | after = `4f6716f63`（origin/main） | 169.2 µs（範囲 165.5-172.6） | 27.7 µs | 同上 |
| M4 Max | - | 未計測 | 未計測 | 実機セッションへ申し送り |
| Mac mini | - | 未計測 | 未計測 | 実機セッションへ申し送り |

- ログ: `logs/cpu-reuse-device-update-2106/smoke-x86-3193/{before,after}/`。before は共有環境のため run 間ばらつきが大きい（§6 と同じく RULE.txt のゲートを通していない）。後退は観測されなかった
- 受け入れ条件「前後の device_update（機体・版つき）」のうち実機分は未達。GB10・Mac mini・M4 Max では、各版で `orchestrate.sh` を回して `aggregate.py` で集計する。`bench-fandhe --task train --device cpu --mode reuse --phases` の前後比較は `[patch.crates-io.fandhe-ai]` の path patch で行う（既存 `run_ab_*_cpu.sh` は定数反転専用でコード差分の A/B には流用できず、派生スクリプトが要る。本変更の範囲外）
- x86 baseline（FMA の target feature なし）では `mul_add` が `fmaf` 呼び出しになるため weight_decay・momentum 経路は自動ベクトル化されない（bit は同一）。計測経路（momentum 0・wd 0）は `p - lr*g` のみでベクトル化が効く

## 11. GB10 実測（#3188）

DGX Spark GB10（20 コア・rustc 1.97.0・affinity pin なし・`RAYON_NUM_THREADS` 未設定）で、継承 RULE（§5）を無変更のまま 2 系列について 5 run ずつ実測した。追加規則は `logs/cpu-train-fixedcost-gb10-3188/RULE.txt`（計測前に固定）。診断のみで、コード・既定値・tolerance は無変更。

- 系列: **pre** = `17c7767b588a8c4c52eccccc39ba880519962d53`（#3196・#3194・#3193 の直前）、**head** = `4f6716f638a6997559847480ddf93cf6924a96e3`（#3188 着手時の origin/main）
- ツリー同定: 各 run 出力と `.rev-stamp` が上記 SHA と一致。計測前後のツリー指紋は同一で、Mac 側 `git archive` の指紋とも差分 0 件（`logs/cpu-train-fixedcost-gb10-3188/{pre,head}/tree_verify.txt`）。負荷ゲートは 2 系列とも全 run pass。checksum は run 間・腕間一致。aarch64 での bit 一致テストも pass（`aarch64_ci_test_*.txt`）
- 生ログと集計: `logs/cpu-train-fixedcost-gb10-3188/{pre,head}/device-update-2106/`（`run{1..5}.jsonl`・`aggregate.md`）

| 区間（中央値 µs。min-max） | pre | head |
|---|---|---|
| T = insitu_direct `device_update` | 279.1（270.9-281.0） | 169.2（165.5-172.6） |
| insitu_pretouch `device_update` | 323.9（317.1-325.1） | 172.2（168.1-174.9） |
| standalone `sgd_kernel`（SGD ループ） | 107.1（105.6-121.5） | 27.7（23.9-31.7） |
| standalone `sgd_kernel_zip` | 22.3 | 22.6 |
| standalone `sgd_kernel_fixed` | 0.1 | 0.1 |
| standalone `alloc`（`flat_grad` 相当の `with_capacity`） | 0.9 | 0.9 |
| standalone `stage` | 0.7 | 0.7 |
| H1 ループ形 | 84.7（T 比 0.30） | 5.0（0.03） |
| H2 cache 状態（direct − pretouch） | -44.8 | -3.0 |
| H3 ホスト確保（alloc + stage） | 1.6（0.01） | 1.6（0.01） |
| H4 prologue + 残差 | 215.2（0.77）**支持** | 142.8（0.84）**支持** |
| 補助: apply_params_split / (compute + apply)_split | 0.36 | 0.34 |

継承 RULE の判定は両系列とも「H4 prologue + 残差」に帰属する。

### イシュー指定の 3 区分への対応（head = 現行の姿）

| 区分 | head の値 | T 比 |
|---|---|---|
| SGD ループ（`sgd_kernel`） | 27.7 µs | 0.16 |
| `flat_grad` 確保 | 構造上 0（#3194 で resident 分岐は確保しない）。standalone の `alloc` は 0.9 µs（未タッチの `with_capacity` のため、ページ初回アクセスのコストは含まない） | 0.01 以下 |
| その他（stage・prologue・cache 状態ほか） | 約 141.5 µs（T − sgd_kernel） | 0.84 |

参考（判定ではない）: 本番経路換算の残差 = insitu_pretouch −（sgd_kernel + stage）= 172.2 −（27.7 + 0.7）= 143.8 µs。

### pre → head の変化（参考。#3193・#3194 の GB10 実測効果）

- T は 279.1 → 169.2 µs（-109.9 µs、0.61 倍）。うち `sgd_kernel` の短縮が 79.4 µs（#3193 の zip 化に対応。H1 が 84.7 → 5.0 µs）。T − `sgd_kernel` は 172.0 → 141.5 µs（-30.5 µs）で、#3194 の `flat_grad` 確保除去（約 814 KB）に対応しうるが、T では #3193 と #3194 を分離できない。分離できるのは standalone 区間（`sgd_kernel`）だけで、この値は #3193 の効果を示す
- 0.9.0 の bench（277.5 µs）は pre の T（279.1 µs）とほぼ一致する。0.11.0 の bench の train reuse 1.030 ms は別プロトコルの値で、本診断の T とは窓が異なる
- 現行の `device_update` は T の約 84% が SGD ループ以外に残る。この内訳は `stage`（0.7 µs）と `alloc`（0.9 µs）で説明されず、standalone の区間に現れない固定費（prologue・tape 側の状態）が支配している。**因果の確定は本診断の範囲外**（H4 は残差項のため）

### 限界と申し送り

- MSE backward の区間はどちらの診断も測らない（#3180 の担当）。#3195 へは、head の `device_update` が約 169 µs で、そのうち SGD ループが約 28 µs にとどまることを前提として渡す
- H4 の内訳（prologue と残差の分離）は本診断で分解できていない。分離するには別の計測が要る（本イシューのスコープ外）
- M4 Max・Mac mini は未実施。`bench-fandhe --phases` の前後比較は未実施
- 診断の値は bench の値と計測窓が異なる（`logs/cpu-train-fixedcost-gb10-3188/RULE.txt`）
