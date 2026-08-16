# Metal GEMM threadgroup ID スウィズル（swizzle_log 相当）A/B 計測記録（#540）

イシュー #540「perf(backend-metal): threadgroup ID スウィズル（swizzle_log 相当）を実験的に追加」の A/B 計測手順・記録テンプレート。
`gemm_simdgroup_tiled` の dispatch grid 走査順を、素朴な行優先（`row0 = tgid.y * BM`・`col0 = tgid.x * BN`）から
threadgroup ID スウィズル（固定値 `SWIZZLE_LOG = 2`。`tile` = 4 threadgroup を 1 群として縦方向へ束ねる）へ変更した効果を計測する
（MLX steel `swizzle_log`・DeepGEMM の L2 スウィズルと同種の技法。MLX 自身は classic 経路で `swizzle_log = 0`〈無効〉のまま
据え置いており未実証の技法である点に留意。計画「背景・目的」節）。

## 状態: 未計測。実機セッションで消化

本ファイルは Linux worktree で作成され、Metal 実機（Apple Silicon）が同一セッションで使用できないため計測手順・
記録テンプレートのみを整備した状態（`metal-gemm-serpentine-ab.md`〈#536〉と同じ運用）。
`crates/backend-metal/src/tile.rs` の `gemm_simdgroup_tiled_source_uses_tgid_swizzle`（crate 内 unit test。PR #661
codex-review 指摘対応で `tests/shader_source_evidence.rs` から移設）によりスウィズル式の実在は Linux CI（ubuntu-latest）
上で機械検査済みだが、性能効果の実測・採否判断（下記「判断基準」）は Mac 実機セッションでの後続対応が必要。

**本番 dispatch は `crate::tile::SWIZZLE_ENABLED`（既定 `false`）でスウィズルを無効化済み**（PR #661 codex-review 指摘対応:
実機未検証のまま本番経路へ無条件適用しない）。`gemm_simdgroup_tiled` の MSL function constant `SWIZZLE_ENABLED`（index 7。
`TGP_PAD`〈#538・index 6〉との index 重複は `tile.rs` 側の機械検査（`gemm_simdgroup_tiled_source_uses_tgid_swizzle`）で
ロック済み）が
`false` の間はシェーダ側も恒等変換（`tid_y = tgid.y`・`tid_x = tgid.x`）で動作し、本ブランチをそのままマージしても挙動・
性能は base（スウィズル導入前）と同一になる。下記計測手順の「head」計測時は、`crates/backend-metal/src/tile.rs` の
`SWIZZLE_ENABLED` を一時的に `true` へ書き換えてから `cargo run` する（**この変更はコミットしない**。計測後に `git checkout --
crates/backend-metal/src/tile.rs` 等で元に戻す）。

## 計測手順（Apple Silicon 実機）

base（変更前）と head（変更後）それぞれについて計測し、5 回計測の中央値 TFLOPS を比較する
（`bench-harness::protocol::run` が中央値計測を実装済み。`coding-rust.md` 準拠。手順自体は
`docs/real-hardware-verification-env.md` の接続・転送手順に従う。実ホスト名はローカル管理外ファイル参照）。

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

出力形式（`examples/gemm_bench.rs` 参照）は `docs/perf/metal-gemm-dynamic-tile.md` と同一（`size=<N>` 行・
`shape=(<M>x<N>x<K>)` 行・`size=<N> candidate=<label>` 行）。受け入れ基準に従い **size ∈ {2048, 4096}** の
`gemm_simdgroup_tiled` 系（`GemmVariant::SimdgroupTiled` 候補構成・`dispatch_auto` 経由でこの経路を通る形状）の
TFLOPS を base/head で突き合わせる。代表的な小〜中形状（例: 128〜512）の悪化有無も併せて記録する（計画「リスクと
安全側の判断」節: grid.x が 4 倍化し `tiles_m` 非倍数時の余剰 threadgroup が増えるため、小形状での軽微な悪化があり得る）。

数値一致確認（採否判断より前に必須。走査順の変更のみでビット単位一致が理論上成立するはずの前提を検証する）:

```sh
cargo test -p backend-metal --release -- --ignored --nocapture
```

`gemm_dynamic_tile_parity`・`cpu_metal_parity`・`gemm_auto_parity` 等が green であること（tolerance は変更しない。coding-rust.md）。

## 判断基準

- size 2048/4096 の両方で base に対し head の中央値 TFLOPS が改善していれば「採用」とし、本ドキュメントへ実測結果を追記する。
  採用の場合は `crates/backend-metal/src/tile.rs` の `SWIZZLE_ENABLED` を `true` へ変更してコミットし、本番 dispatch で
  スウィズルを有効化する（このドキュメント・`tile.rs` の doc comment 双方を実測結果込みで更新する）
- 改善が確認できなければ**採用しない**と判断し、スウィズルの変更一式（`tile.rs` の `SWIZZLE_LOG`/`SWIZZLE_ENABLED`/
  `swizzled_grid`/`tiled_dispatch_grid`・`gemm.metal` の tgid 変換・`SWIZZLE_ENABLED` function constant・`gemm.rs` の
  `encode_dispatch_tiled` 呼び出し切替・`pipeline.rs` の function constant 設定・`tile.rs` 内の
  `gemm_simdgroup_tiled_source_uses_tgid_swizzle` 等のテスト）を revert PR で撤去し、その判断と実測値を本ドキュメントへ記録する
  （既定 off のパラメータとして残さない。計画「背景・目的」節の受け入れ基準）

## 実測結果

（未計測。実機セッションで本節へ追記する）
