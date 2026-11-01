# NN 層・最適化・ONNX の学習の実機 parity 雛形（イシュー #3778）

**現在の状態: 未実測。** この PR は実機で何も測っていない。共通部品と `RULE.txt` 雛形を用意しただけで、
結果欄はすべて空である。

## 目的

Phase 6〜8 で GPU に触れる機能が増える。NN 層の forward・backward、param groups と `DeviceParamStore`
の常駐更新、GPU tape 上の ONNX の学習である。Phase 10 でこれらを実機（`<cuda-node>`・`<metal-node>`）で
測る。テストの形・判定の呼び方・ログの書式を 1 か所にそろえ、保守と集計を軽くする。

| 項目 | 場所 |
|---|---|
| 比較・記録の共通部品 | `crates/facade/tests/common/train_gpu_parity.rs` |
| 登録表と実機テスト入口 | `crates/facade/tests/train_gpu_parity.rs` |
| 事前登録規則の雛形 | `docs/perf/logs/train-gpu-parity-template/RULE.txt` |
| 実行手順 | `docs/real-hardware-verification-env.md` §4.10 |

## 族と issue の対応

| 族 | 実装 issue（Phase 6〜8） | Phase 10 の測定 issue |
|---|---|---|
| `scaffold`（雛形の自己検査用） | #3778 | なし |
| `nn` | #3814、#3817〜#3827 | #3860（GB10）、#3861（Mac mini） |
| `optim` | #3837、#3838 | #3862（GB10）、#3863（Mac mini） |
| `onnx` | #3842 | #3864（GB10）、#3865（Mac mini） |

実機テストは族ごとに 1 本。名前は `train_parity_<族>_<cuda|metal>_matches_cpu` で固定する（`metal` は macOS のみ）。

## 行の足し方（Phase 6〜8 向け）

1. `run` 関数を書く。`fn(Device) -> Result<Vec<Quantity>, String>`。決まった種を使い、S1（CI で回る小形状）と
   S2（2 のべき乗でない中形状）の 2 行を足す。勾配が自明にゼロにならないよう `require_nonzero: true` を付ける
2. `CASES` に `Case` を 1 行足す。`family` は `FAMILIES` にある族を指定する
3. `Judge` は設計記録に従う。param groups の常駐更新は `docs/autodiff-param-groups-decision.md` の設計どおり
   `Bit` か `Req2`。それ以外は `Req2`
4. ホスト実行の op（ONNX の `Erf`・`Softmax`・`LayerNormalization` など）を含むケースは `Route::H`。
   opt-in ゲートの後ろは `Route::G`（既定 OFF なので H として測る）
5. 参照腕がデバイス常駐対ホスト参照のように CPU tape 以外なら `reference: Some(...)` を指定する
6. まず CI で走る自己検査（`cargo test -p fandhe-ai --test train_gpu_parity`）が CPU 同士で通ることを確かめる

ONNX 族は #3842 の公開形でつなぐ。素の `fandhe_ai_autodiff::Tape` が要る間は
`crates/facade/tests/spatial_layers_backend_parity.rs::raw_tape_for` の組み立てを参照する。それまで `onnx` 族は
0 件で、実行すると undetermined で止まる。

## Phase 10 での手順

1. `RULE.txt` を `docs/perf/logs/<名前>-<issue 番号>/RULE.txt` へ写し、`<...>` を埋める
2. 実機での実行より前にコミットする
3. `--no-run` でビルド、`--ignored --list` で選ばれるテストが 1 件であることを確認
4. `--exact` で 1 本ずつ別プロセスで実行
5. `TRAIN_PARITY` 行を集計し、結果をその README に書く
6. `docs/README.md` の索引に 1 行足し、`docs/compat-matrix-ari-tracker.md` の「実機 parity」列を更新する

## 禁止事項

- 判定式を書き写さない。`compare` を呼ぶ（既存の `device_param_store_backend_parity.rs::assert_close` が式を手で
  書いている点はこの PR の対象外）
- tolerance を変えない。baseline を足さない（足すなら実機の実測値と所有者の承認が要る）
- Linux や M4 Max 開発機の値で GB10／Mac mini の pass にしない
- fail の後に再実行して pass に置き換えない
- 実測値を推測で埋めない

## 結果

未実測。
