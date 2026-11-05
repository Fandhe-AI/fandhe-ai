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

ONNX 族は #3842 の公開形（`OnnxModel::bind_autograd`）でつないだ。行は `onnx_from_sequential_forward_backward_s1`・`_s2`
（route は H。11 op はホスト実行）で、CI の自己検査は `onnx_family_passes_cpu_against_cpu`。結果欄は未実測のまま
（#3864・#3865）。

## nn・optim 族の登録行（#4026）

実機 parity は #3860〜#3863 で未実測（結果欄は未実測のまま）。tolerance・baseline・閾値は変えていない。
全行 `Judge::Req2`（骨格に Linear／Conv の GEMM が入るため。optim も `docs/autodiff-param-groups-decision.md`
§19.5 (3) が GPU 対参照を REQ-2 と定めるため）。

- 量: nn 族は `forward`・`loss`・`grad_p<i>`（スロット順）・`loss_after_step`（ホスト SGD 1 step 後の loss）。
  `Var` グラフ行（bilinear・similarity）は `loss`・`grad_p<i>`・`loss_after_step`。optim 族は N step 後の `param_*`
- 選定規則: 追跡表の「実機 parity」列が #4026 待ちだった 7・9〜15・22 行。8 行（独自 `Module`）は新しい数値経路がなく対象外
- FractionalMaxPool の行は共有 RNG に依存するため各 forward の直前で `manual_seed` を呼ぶ。実機は `--test-threads=1` で走らせる

| 行（`_s1`／`_s2`） | 追跡表 | route | 骨格 |
|---|---|---|---|
| `nn_upsample_scale_factor_train_step` | 7 | K | conv2d → upsample（bilinear）→ linear |
| `nn_shrink_train_step` | 9 | H | conv2d → hardshrink → softshrink → linear |
| `nn_bilinear_train_step` | 10 | H | `Var::bilinear` → MSE |
| `nn_batch_norm3d_train_step` | 11 | K | conv3d → batch_norm3d → relu → linear |
| `nn_similarity_distance_train_step` | 12・13 | H | matmul → cosine_similarity・pairwise_distance |
| `nn_channel_shuffle_train_step` | 14 | H | conv2d → channel_shuffle → conv2d → linear |
| `nn_adaptive_pool3d_train_step` | 15 | H | conv3d → adaptive avg／max pool3d → linear |
| `nn_lp_pool_train_step` | 15 | H | conv2d → lp_pool2d → linear |
| `nn_fractional_max_pool_train_step` | 15 | H | conv2d → fractional_max_pool2d → linear |
| `optim_param_groups_resident_{sgd,adam,adamw}` | 22 | K | param groups 常駐更新 対 ホスト `step_with_groups` |

CI の自己検査は `nn_family_passes_cpu_against_cpu`・`optim_family_passes_cpu_against_cpu`・
`every_declared_family_has_cases`（登録 0 件の族を防ぐ）。

## Phase 10 での手順

1. `RULE.txt` を `docs/perf/logs/<名前>-<issue 番号>/RULE.txt` へ写し、`<...>` を埋める
2. 実機での実行より前にコミットする
3. （#4044）ビルドの前に、全 workspace メンバーの成果物を `cargo clean -p <名前>`（dev プロファイル）で消す。ビルド後、テストバイナリと全メンバーの lib 成果物の更新時刻がロック取得より後であることを確かめ、外れれば undetermined。雛形は `docs/perf/logs/hw-parity-run-template/`、手順は `docs/real-hardware-verification-env.md` §4.11
4. `--no-run` でビルド、`--ignored --list` で選ばれるテストが 1 件であることを確認
5. `--exact` で 1 本ずつ別プロセスで実行
6. `TRAIN_PARITY` 行を集計し、結果をその README に書く
7. `docs/README.md` の索引に 1 行足し、`docs/compat-matrix-ari-tracker.md` の「実機 parity」列を更新する

## 禁止事項

- 判定式を書き写さない。`compare` を呼ぶ（既存の `device_param_store_backend_parity.rs::assert_close` が式を手で
  書いている点はこの PR の対象外）
- tolerance を変えない。baseline を足さない（足すなら実機の実測値と所有者の承認が要る）
- Linux や M4 Max 開発機の値で GB10／Mac mini の pass にしない
- fail の後に再実行して pass に置き換えない
- 実測値を推測で埋めない

## 結果

未実測。
