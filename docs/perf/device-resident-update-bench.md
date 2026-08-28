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

| バックエンド | legacy 中央値 (s/step) | resident 中央値 (s/step) | total_speedup_x | resident_faster |
|---|---|---|---|---|
| CPU | 1.23〜1.31e-4 | 1.21〜1.29e-4 | 0.96〜1.08 | 試行によりどちらも僅差（ノイズレベル） |
| Metal（実機） | 5.8〜6.9e-4 | 9.4〜11.2e-4 | 0.61〜0.64 | **false（新経路が一貫して遅い）** |
| CUDA | 未計測（本ローカル環境に実機なし。`#[ignore]` テスト整備済み） | — | — | — |

update フェーズ単体（参考。1 step 全体のうち更新処理のみ）:

| バックエンド | legacy update 中央値 (s) | resident update 中央値 (s) | update_speedup_x |
|---|---|---|---|
| CPU | ~1.49e-6 | ~1.15e-6 | 1.11〜1.30（僅差で resident 側が速い） |
| Metal | ~1.5e-6 | 4.1〜4.3e-4 | **0.004（resident 側が約 250 倍遅い）** |

CPU は複数回実行してもいずれかが僅かに速い程度でノイズレベル（±1 割
程度）の差に留まり、明確な後退は観測されない。**Metal は複数回実行して
一貫して新経路（resident）が旧経路より遅い**（total で約 1.6〜1.7 倍、
update フェーズ単体では約 250 倍）。

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
- **Metal**: **後退を観測**（実機・複数回実行で再現。total 約 1.6〜1.7 倍
  遅い、update フェーズ単体では約 250 倍遅い）。原因は上記 4 節のとおり
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
