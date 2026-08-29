# デバイス常駐更新のベンチ非後退確認（イシュー #936）

## 1. 目的・対応

イシュー #936「デバイス常駐更新の parity テストとベンチ非後退確認」の
受け入れ条件 2（「常駐化前後でベンチが非後退であることを確認する」）に
対応する実測記録。判定方式・比較軸は設計文書
`docs/device-resident-update-design.md` §5.3・§7「#936 への引き渡し事項」を
正とする。

- 計測ハーネス: `crates/facade/tests/device_param_store_bench.rs`
- 実行コマンド:
  ```sh
  cargo test -p fandhe-ai --release --test device_param_store_bench -- --nocapture
  ```
- 計測方式: 5 回計測中央値（`.claude/rules/coding-rust.md`）。**record
  only（hard assert なし）**——`crates/facade/tests/tape_cuda_cache_bench.rs`
  と同じ方針（GPU クロック挙動・環境揺らぎを hard assert に持ち込むと
  flaky 化するため）。tolerance・ガードレール閾値には触れていない。

## 2. 比較軸・計測区間

| 経路 | 内容 |
|------|------|
| 旧経路（legacy） | 毎 step: `Sequential::bind`（`weight`／`bias` を毎回ホストから再アップロード）→ forward → backward → `Sgd::step` → `apply_parameters` |
| 新経路（resident） | 初回のみ `init_device_param_store`（1 回アップロード）。毎 step: `forward_resident`（`register_resident_leaves` が D2H download）→ backward → `step_device_param_store`（grad を 1 パラメータずつ upload → `sgd_step_device`） |

主計測は 1 step 全体（forward + backward + update）。参考として update
フェーズ単体（旧: `Sgd::step` + `apply_parameters`、新:
`step_device_param_store`）も計測し、要因分離の観察に用いる。各バックエンド
とも 1 回の warmup 呼び出し（`tape_for` の初回結線コスト・#931 系タイム
アップ初期化コスト (a) を両経路の本計測から除く）の後、
`STEPS_PER_TRIAL = 20` step の平均 per-step 時間を 1 trial とし、
`TRIALS = 5` 回計測して中央値・Q1・Q3 を求める。

対象モデル: `D_IN=8 → D_HIDDEN=16（ReLU）→ D_OUT=4`、`BATCH=4`（`crates/
facade/tests/device_param_store_train.rs` と同一モデル様式）。

## 3. 実測結果

環境: Apple M4 Max（macOS 26.6.2）・`rustc 1.96.0`・`--release`。

**#955 レビュー指摘（PR #955・`crates/facade/tests/device_param_store_bench.rs`
の計測区間修正）を反映した再計測値**: 旧経路の `update_secs` は従来
`Sgd::step` 直後（`model.apply_parameters(updated)` を含まない）で確定して
いたため、resident 側（`step_device_param_store` 全体を計測）と非対称な
区間の比較になっていた。`t_update` の経過時間加算を `apply_parameters` の
後へ移し、update フェーズを両経路とも「`Sgd::step`（または相当処理）+
パラメータ反映」で揃えたうえで下表を再計測した。

| バックエンド | legacy 中央値 (s/step) | resident 中央値 (s/step) | total_speedup_x | resident_faster |
|---|---|---|---|---|
| CPU | 1.18〜1.22e-4 | 1.27〜1.36e-4 | 0.88〜0.94 | 試行によりどちらも僅差（ノイズレベル） |
| Metal（実機） | 6.3〜7.2e-4 | 1.04〜1.11e-3 | 0.60〜0.65 | **false（新経路が一貫して遅い）** |
| CUDA | 未計測（本ローカル環境に実機なし。`#[ignore]` テスト整備済み） | — | — | — |

update フェーズ単体（参考。1 step 全体のうち更新処理のみ。修正後は旧経路も
`apply_parameters` を含む）:

| バックエンド | legacy update 中央値 (s) | resident update 中央値 (s) | update_speedup_x |
|---|---|---|---|
| CPU | ~2.97〜3.13e-6 | ~1.19〜1.35e-6 | 2.3〜2.5（resident 側が明確に速い） |
| Metal | ~2.88〜2.98e-6 | ~3.94〜4.37e-4 | **0.007（resident 側が約 132〜152 倍遅い）** |

`apply_parameters`（`crates/autodiff/src/compat/sequential.rs`）はホスト側の
`Tensor<f32>` 差し替えのみで GPU ディスパッチを伴わないため、修正前後で
legacy update の絶対値自体はいずれのバックエンドも同オーダー（数 μs 未満）
のまま変化していない。一方 CPU の `update_speedup_x` は「resident 側が
僅差で速い（1.11〜1.30）」から「resident 側が明確に速い（2.3〜2.5）」へ、
Metal の乖離倍率は「約 250 倍」から「約 132〜152 倍」へ、それぞれ計測区間の
対称化により数値が変わった（結論の方向性——CPU は非後退・Metal は resident
側が update フェーズ単体で 2 桁倍遅い——は変わらない）。

CPU は複数回実行してもいずれかが僅かに速い程度でノイズレベル（±1 割
程度）の差に留まり、明確な後退は観測されない。**Metal は複数回実行して
一貫して新経路（resident）が旧経路より遅い**（total で約 1.5〜1.7 倍、
update フェーズ単体では約 132〜152 倍）。

## 4. 原因分析（転送モデルの前提との突合）

PR #954 の #936 への申し送り（設計文書 §3.3）どおり、新経路が削減するのは
「param の毎 step 再アップロード」のみであり、以下は新経路でも毎 step
発生する:

- `register_resident_leaves`（`crates/autodiff/src/optim/device_store.rs`）
  は forward 用に毎 step D2H download を行う
- `DeviceParamStore::step`（同ファイル）は「① 事前検証 → ② 1 パラメータ
  ずつ grad を upload → `sgd_step_device`」の順で、**grad を 1 パラメータ
  ずつ**（本モデルでは weight1・bias1・weight2・bias2 の計 4 バッファ）
  upload してから GPU カーネルを起動する

対象モデルが小さい（`D_HIDDEN=16` 程度）ため、実データの転送量そのものは
どちらの経路でも小さく、支配的なのは **Metal のコマンドバッファ
生成・コミット・同期（`waitUntilCompleted` 相当）のディスパッチ単位あたり
固定オーバーヘッド**だと考えられる。旧経路は `Sgd::step`（ホスト側で
全パラメータをまとめて計算し、`apply_parameters` で置き換えるのみ）が
GPU ディスパッチを一切伴わないのに対し、新経路は 1 step あたり
「forward の D2H download（複数バッファ）+ update の grad upload ×
パラメータ数 + `sgd_step_device` カーネル起動 × パラメータ数」という
複数回の GPU ディスパッチを伴う。小規模モデルではこの固定オーバーヘッドの
回数が実データ転送量削減の効果を上回り、resident 経路が遅くなっている
と考えられる。

この観察は設計文書 §3.3・PR #954 申し送りの前提（「削減されるのは param
再アップロードのみ。step あたり総転送量が旧経路より必ず減るとは限らない」）
と整合する。本イシューはこの前提の検証・記録までがスコープであり、
tolerance・実装（`register_resident_leaves`・`DeviceParamStore::step` の
ディスパッチ粒度等）の変更はスコープ外とする（イシュー #936 実装計画
7 節「スコープ外」）。

## 5. 非後退判定の結論

- **CPU**: 非後退（ノイズレベルの差。明確な後退なし）
- **Metal**: **後退を観測**（実機・複数回実行で再現。total 約 1.5〜1.7 倍
  遅い、update フェーズ単体では約 132〜152 倍遅い）。原因は上記 4 節のとおり
  小規模モデルにおける GPU ディスパッチ回数増加（D2H download 継続 + grad
  upload のパラメータ単位分割）であり、tolerance・実装の変更で対処する
  事項ではないためスコープ外として記録する（4 節参照。改善実装は本
  イシューのスコープ外〈実装計画 7 節〉）
- **CUDA**: 本ローカル環境（Apple Silicon 実機）に実機がないため未計測。
  `crates/facade/tests/device_param_store_bench.rs::legacy_vs_resident_per_step_cuda`
  （`#[ignore]`）を整備済みであり、DGX Spark 等の実機アクセス時に
  `cargo test -p fandhe-ai --release --test device_param_store_bench --
  --ignored --nocapture` で計測可能

**総括**: 常駐化はモデル・バッチサイズによっては現状 Metal で性能後退
となりうることが実測で確認された。この結果は「param 再アップロードの
削減が総転送量削減を保証しない」という設計文書の前提どおりであり、本
イシューの受け入れ条件（ベンチ非後退の**確認**。改善の実装ではない）を
満たすため、原因分析とともにここへ記録する。改善（例: grad upload の
バッチ化・D2H download 頻度の削減）が必要かどうかの判断・後続対応は
別イシューでの検討をユーザーへ提案する（`out-of-scope-tracking.md`）。

## 6. #1022 後の再計測（forward の param D2H 排除）

イシュー #1022 は「削減されるのは param 再アップロードのみ」という
上記 4 節・設計文書 §3.3 の前提そのもの（`register_resident_leaves` が
毎 step weight/bias を download していた点）を変更した——
`DeviceParamStore::register_resident_leaves`／`linear_forward` が
download を行わなくなったため、新経路の GPU ディスパッチ回数は
「grad upload × パラメータ数 + `sgd_step_device` カーネル起動 ×
パラメータ数」（forward 側の download が消えた分だけ旧経路より少ない）
へ変わった。

**計測区間・環境**: `crates/facade/tests/device_param_store_bench.rs`
（4 節と同一ハーネス・同一モデル・`TRIALS=5`・`STEPS_PER_TRIAL=20`）。
本ラン環境は Linux x86_64（RTX 3060・nvcc なし。CUDA 実機計測は本ランの
スコープ外。手順注記参照）。

- **CPU**（`cargo test -p fandhe-ai --release --test device_param_store_bench
  legacy_vs_resident_per_step_cpu -- --nocapture`）:
  `legacy_total_median=64.9µs (q1=56.6µs, q3=77.0µs)` 対
  `resident_total_median=59.8µs (q1=56.7µs, q3=64.7µs)`
  （`total_speedup_x=1.09`。resident が僅かに高速）。update フェーズ単体は
  `legacy=2.76µs` 対 `resident=1.12µs`（`update_speedup_x=2.48`）。
  4 節の CPU 計測（非後退・ノイズレベル差）から明確な後退は生じておらず、
  update フェーズは #1022 と無関係に既存どおり高速だが、total 側も
  #1022 前よりわずかに resident 優位側へ寄っている（forward の download
  排除分。ただし CPU は転送コスト自体が小さいためこの差もノイズレベルに
  近い。4 節の record only 方針を維持し hard assert はしない）。
- **Metal／CUDA**: 本ラン（Linux・Metal 実機なし・CUDA toolkit なし）では
  未計測。Mac / DGX Spark セッションでの実測を後続に委ねる
  （`crates/facade/tests/device_param_store_bench.rs::
  legacy_vs_resident_per_step_cuda`〈`#[ignore]`〉・Metal 実機での
  `cargo test -p fandhe-ai --release --test device_param_store_bench --
  --ignored --nocapture` を参照）。4 節で観測された Metal の後退
  （小規模モデルでの GPU ディスパッチ回数増加が主因）が #1022 の
  forward download 排除でどの程度緩和されるかは実機再計測が必要。
