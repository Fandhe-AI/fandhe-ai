# Metal GEMM threadgroup ID スウィズル（swizzle_log 相当）A/B 計測記録（#540）

イシュー #540「perf(backend-metal): threadgroup ID スウィズル（swizzle_log 相当）を実験的に追加」の A/B 計測手順・記録テンプレート。
`gemm_simdgroup_tiled` の dispatch grid 走査順を、素朴な行優先（`row0 = tgid.y * BM`・`col0 = tgid.x * BN`）から
threadgroup ID スウィズル（固定値 `SWIZZLE_LOG = 2`。`tile` = 4 threadgroup を 1 群として縦方向へ束ねる）へ変更した効果を計測する
（MLX steel `swizzle_log`・DeepGEMM の L2 スウィズルと同種の技法。MLX 自身は classic 経路で `swizzle_log = 0`〈無効〉のまま
据え置いており未実証の技法である点に留意。計画「背景・目的」節）。

## 状態: プロトコル整備済み・実測は Mac 実機セッションで消化

イシュー #746 により、旧 checkout＋一時トグル方式（後述「旧手順（履歴）」節）を、同一コミット上で base（swizzle
off）/head（swizzle on）の 2 `MetalGemm` インスタンスを構築し interleaved に A/B 計測する手順へ差し替えた
（`docs/perf/metal-bench-noise-protocol.md` 参照。ノイズ対策プロトコルの設計・根拠はそちらを正とし本節では
書き写さない）。本ファイルは Linux worktree で作成され、Metal 実機（Apple Silicon）が同一セッションで使用できない
ため計測手順・記録テンプレートのみを整備した状態（`metal-gemm-serpentine-ab.md`〈#536〉と同じ運用）。
`crates/backend-metal/src/tile.rs` の `gemm_simdgroup_tiled_source_uses_tgid_swizzle`（crate 内 unit test。PR #661
codex-review 指摘対応で `tests/shader_source_evidence.rs` から移設）によりスウィズル式の実在は Linux CI（ubuntu-latest）
上で機械検査済みだが、性能効果の実測・採否判断（下記「判断基準」）は Mac 実機セッションでの後続対応が必要。

**本番 dispatch は `crate::tile::SWIZZLE_ENABLED`（既定 `false`）でスウィズルを無効化済み**（PR #661 codex-review 指摘対応:
実機未検証のまま本番経路へ無条件適用しない）。`gemm_simdgroup_tiled` の MSL function constant `SWIZZLE_ENABLED`（index 7。
`TGP_PAD`〈#538・index 6〉との index 重複は `tile.rs` 側の機械検査（`gemm_simdgroup_tiled_source_uses_tgid_swizzle`）で
ロック済み）が `false` の間はシェーダ側も恒等変換（`tid_y = tgid.y`・`tid_x = tgid.x`）で動作し、本番経路（`MetalGemm::new`）
の挙動・性能は変わらない。イシュー #746 で `swizzle_enabled` を `MetalGemm` の instance フィールドへ格上げした
（`crate::pipeline::make_pipeline_with_constants`・`crate::tile::tiled_dispatch_grid_with` の引数化。
`crates/backend-metal/src/gemm.rs`・`pipeline.rs`・`tile.rs` 参照）ため、A/B 計測用の `MetalGemm::new_with_swizzle(ctx, true)`
インスタンスを本番経路（`MetalGemm::new`。常に `SWIZZLE_ENABLED=false` を渡す）とは別に構築でき、`tile.rs` のソースを
一時的に書き換える必要はなくなった。

## 計測手順（Apple Silicon 実機）

`bench_harness::ab::run_ab`（`docs/perf/metal-bench-noise-protocol.md` 参照）で base（swizzle off）/head（swizzle on）
を同一プロセス内・interleaved に計測する。

```sh
git fetch origin
git checkout test/746-metal-bench-noise-protocol   # 本イシューの実装ブランチ（PR マージ後は main で可）

# 実行前のサーマル状態を記録する（非特権コマンド。sudo 必須の powermetrics は不使用）。
pmset -g therm

cargo run -p backend-metal --example gemm_swizzle_ab_bench --release > /tmp/gemm_swizzle_ab_bench.txt

# 実行後のサーマル状態も記録する。
pmset -g therm
```

`gemm_swizzle_ab_bench` はフェーズ 1（安定性セルフチェック。対照カーネルの spread が全サイズで ≤5% 相当か）を
先に実行し、いずれかのサイズが gate を超過した場合はフェーズ 2（swizzle A/B）へ進まず「判定不可」を出力して
終了する（安全側判断。`docs/perf/metal-bench-noise-protocol.md` §「安定性ゲートと不成立時の中断規定」）。gate を
満たさない場合は同ドキュメントの調整手順（`ROUNDS`/`COOLDOWN`/`MIN_WARMUP` を増やす方向のみ）に従い再実行する。

フェーズ 2 の出力は size ∈ {256, 512, 1024, 2048, 4096} ごとに `base_median_tflops`・`head_median_tflops`・
`head_over_base`（head/base 比）・`spread_base`・`spread_head`・base/head 双方の `resolved` タイル構成
（`pipeline_for_tile` のフォールバック発生有無）を含む。受け入れ基準に従い **size ∈ {2048, 4096}** の
`head_over_base` を採否判断の根拠とし、小〜中形状（256〜1024）の悪化有無も併せて記録する（計画「リスクと
安全側の判断」節: grid.x が 4 倍化し `tiles_m` 非倍数時の余剰 threadgroup が増えるため、小形状での軽微な悪化があり得る）。

数値一致確認（採否判断より前に必須。走査順の変更のみでビット単位一致が理論上成立するはずの前提を検証する）:

```sh
cargo test -p backend-metal --release -- --ignored --nocapture
```

`gemm_dynamic_tile_parity`・`cpu_metal_parity`・`gemm_auto_parity` 等が green であること（tolerance は変更しない。coding-rust.md）。

## 判断基準

- size 2048/4096 の両方で base に対し head の中央値 TFLOPS（`head_over_base` > 1.0 相当）が改善していれば「採用」とし、
  本ドキュメントへ実測結果を追記する。採用の場合は `crates/backend-metal/src/tile.rs` の `SWIZZLE_ENABLED` を `true` へ
  変更してコミットし、`MetalGemm::new`（本番経路）の既定挙動でスウィズルを有効化する
  （このドキュメント・`tile.rs` の doc comment 双方を実測結果込みで更新する）
- 改善が確認できなければ**採用しない**と判断し、スウィズルの変更一式（`tile.rs` の `SWIZZLE_LOG`/`SWIZZLE_ENABLED`/
  `swizzled_grid`/`tiled_dispatch_grid`/`tiled_dispatch_grid_with`・`gemm.metal` の tgid 変換・`SWIZZLE_ENABLED` function
  constant・`gemm.rs` の `MetalGemm::swizzle_enabled` フィールド・`new_with_swizzle`・`encode_dispatch_tiled` の
  `swizzle_enabled` 引数・`pipeline.rs` の `make_pipeline_with_constants` の `swizzle_enabled` 引数・`tile.rs` 内の
  `gemm_simdgroup_tiled_source_uses_tgid_swizzle` 等のテスト・`gemm_swizzle_ab_bench.rs` example）を revert PR で撤去し、
  その判断と実測値を本ドキュメントへ記録する（既定 off のパラメータとして残さない。計画「背景・目的」節の受け入れ基準）

## 実測結果

（未計測。実機セッションで本節へ追記する）

## 旧手順（履歴）

イシュー #746 以前は checkout 切替＋一時トグル方式だった。base/head 計測が時間的に分離されサーマルドリフトが
系統誤差として乗る問題があり（#746 イシュー本文の 2026-08-19 実測: 対照カーネルが最大 70% 超変動）、
上記「計測手順」節の interleaved 方式へ置き換えた。参考として残す。

```sh
git fetch origin

# base（変更前。スウィズル導入前の直近コミット）
git checkout <base-sha>
cargo run -p backend-metal --example gemm_bench --release > /tmp/gemm_bench_base.txt

# head（本イシューの実装ブランチ）。SWIZZLE_ENABLED は既定 false のため、
# 計測前に crates/backend-metal/src/tile.rs の SWIZZLE_ENABLED を一時的に
# true へ書き換える（コミットしない。計測後に revert する）。
git checkout perf/540-metal-gemm-tgid-swizzle
# （ここで SWIZZLE_ENABLED を true へ一時変更）
cargo run -p backend-metal --example gemm_bench --release > /tmp/gemm_bench_head.txt
# （計測後: git checkout -- crates/backend-metal/src/tile.rs で revert）
```
