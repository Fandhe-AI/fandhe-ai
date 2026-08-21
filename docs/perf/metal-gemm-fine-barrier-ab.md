# Metal GEMM simdgroup 細粒度同期（`simdgroup_barrier(mem_none)`）A/B 計測記録（#809）

イシュー #809「perf(backend-metal): simdgroup_barrier 細粒度同期とバッファプール要否の調査」の A/B 計測手順・記録テンプレート。
MLX steel `mma.h`（`docs/perf/metal-gemm-bottleneck-diagnosis.md`〈#487〉・`docs/backend-metal-mlx-classic-nax-decision.md`
〈#549〉で構成対比済み）が simdgroup フラグメントロード間に用いる `simdgroup_barrier(mem_flags::mem_none)`
（`threadgroup_barrier` より軽量な simdgroup スコープのフェンス）を、`gemm_simdgroup_tiled` の staged 経路 kk ループへ
適用した効果を計測する。

## 状態: プロトコル整備済み・実測は Mac 実機セッションで消化（未消化）

本ファイルは Linux worktree（`.claude/worktrees/wf_6c80a1fd-533-189`）で作成され、Metal 実機（Apple Silicon）が同一
セッションで使用できないため計測手順・記録テンプレートのみを整備した状態（`metal-gemm-tgid-swizzle-ab.md`〈#540〉・
`metal-gemm-serpentine-ab.md`〈#536〉と同じ運用）。`crate::tile::FINE_BARRIER_ENABLED` は引き続き `false` のままであり、
本番挙動（`MetalGemm::new` の既定経路）に変更はない。Mac 実機セッションでの実測消化が必要（実測値の推定・捏造は
行わない）。

`crates/backend-metal/tests/shader_source_evidence.rs` の
`gemm_metal_source_declares_fine_barrier_enabled_function_constant`・
`gemm_simdgroup_tiled_source_gates_fine_barrier_between_fragment_load_and_mma`（Linux CI・ubuntu-latest 上で実行）
により、`FINE_BARRIER_ENABLED` function constant（index 8）の宣言と、B フラグメントロード直後・MMA 発行直前という
挿入位置の契約は機械検査済みだが、性能効果の実測・採否判断（下記「判断基準」）は Mac 実機セッションでの後続対応が
必要。

## 適用箇所と数値契約

`shaders/gemm.metal` の `gemm_simdgroup_tiled`（本番既定経路。`dispatch_variant(GemmVariant::SimdgroupTiled(..))`）の
staged 経路（`USE_TGP_STAGING=true`）kk ループ内、A/B フラグメント一括ロード（`a_frag`/`b_frag`。イシュー #745 で
レジスタ常駐化済み）と MMA 発行（`simdgroup_multiply_accumulate`）の間に、`FINE_BARRIER_ENABLED` でゲートされた
`simdgroup_barrier(mem_flags::mem_none)` を挿入した。

- **direct-load（else 節）経路は適用対象外**: フラグメント再ロード構造が staged 経路と異なるため（実装計画 §2.1）。
- `tile_a`/`tile_b`（threadgroup メモリ）の内容自体は前段の `threadgroup_barrier(mem_flags::mem_threadgroup)`（協調
  ロード直後）で既に確定済みのため、この挿入は演算オペランド列（値・順序）を一切変えない。よって base（barrier
  なし）/head（barrier あり）の出力は理論上ビット単位で一致するはずである（#536・#538 と同じ論法）。
- `examples/gemm_fine_barrier_ab_bench.rs` のフェーズ 0 でこの数値契約を計測前に自己検証する（`assert_eq!` で
  `dispatch_auto` の出力を直接比較。size 256/512/1024）。

## 計測手順（Apple Silicon 実機）

`bench_harness::ab::run_ab`（`docs/perf/metal-bench-noise-protocol.md` 参照）で base（`fine_barrier_enabled=false`）/
head（`fine_barrier_enabled=true`）を同一プロセス内・interleaved に計測する。

```sh
git fetch origin
git checkout perf/809-metal-fine-barrier-buffer-pool   # 本イシューの実装ブランチ（PR マージ後は main で可）

# 実行前のサーマル状態を記録する（非特権コマンド。sudo 必須の powermetrics は不使用）。
pmset -g therm

cargo run -p backend-metal --example gemm_fine_barrier_ab_bench --release > /tmp/gemm_fine_barrier_ab_bench.txt

# 実行後のサーマル状態も記録する。
pmset -g therm
```

`gemm_fine_barrier_ab_bench` は次の順で実行する:

1. **フェーズ 0（bit 一致自己検証）**: base/head の `dispatch_auto` 出力を size 256/512/1024 で直接比較する。不一致の
   場合は `panic` し、フェーズ 1・2 へは進まない（安全側判断）。
2. **フェーズ 1（安定性セルフチェック）**: 本番既定（fine barrier off）の `MetalGemm::new` を使い、対照カーネルの
   spread が全サイズ（256〜4096）で `bench_harness::ab::STABILITY_SPREAD_GATE` 以下か確認する。いずれかのサイズが
   gate を超過した場合はフェーズ 2 へ進まず「判定不可」を出力して終了する（`docs/perf/metal-bench-noise-protocol.md`
   §「安定性ゲートと不成立時の中断規定」）。gate を満たさない場合は同ドキュメントの調整手順（`ROUNDS`/`COOLDOWN`/
   `MIN_WARMUP` を増やす方向のみ）に従い再実行する。
3. **フェーズ 2（prepared 境界 A/B）**: `MetalGemm::new_with_fine_barrier` で構築した base/head を
   `dispatch_tiled_prepared`（アップロード済みバッファ共有・確定 `TileConfig` 直接指定）で計測する。出力は
   size ∈ {256, 512, 1024, 2048, 4096} ごとに `base_median_tflops`・`head_median_tflops`・`head_over_base`
   （head/base 比）・`spread_base`・`spread_head`・base/head 双方の `resolved` タイル構成を含む。

数値一致確認（採否判断より前に必須。フェーズ 0 に加え、既存の parity テストでも再確認する）:

```sh
cargo test -p backend-metal --release -- --ignored --nocapture
```

`gemm_dynamic_tile_parity`・`cpu_metal_parity`・`gemm_auto_parity` 等が green であること（tolerance は変更しない。
coding-rust.md）。

## 判断基準（イシュー #809 実装計画 §3.1）

- **採用**: size 2048/4096 の `head_over_base`（head/base の中央値 TFLOPS 比）に改善があり、かつ他サイズ
  （256〜1024）で劣化中央値 5% 超がない場合、`crates/backend-metal/src/tile.rs` の `FINE_BARRIER_ENABLED` を `true`
  へ変更してコミットし、`MetalGemm::new`（本番経路）の既定挙動で細粒度同期を有効化する（#784 の swizzle 結線と
  同型。このドキュメント・`tile.rs`/`gemm.rs`/`pipeline.rs`/`shaders/gemm.metal` の doc comment を実測結果込みで
  更新する。結線後は既定経路 parity テスト〈`gemm_dynamic_tile_parity` 等〉を再実行して非後退を確認する）
- **不採用**: 改善が確認できない場合、`FINE_BARRIER_ENABLED=false` のまま維持し、本ドキュメントへ実測結果と
  不採用の判断理由を記録する（機構自体は revert せず残す。実機未検証構成を安全に切り替え可能な状態に保つ設計
  判断は `SWIZZLE_ENABLED`〈#540〉と同型）。

判定統計・許容誤差・閾値自体は変更しない（`.claude/rules/security.md`「自己修復ループ固有のガードレール」・
`docs/spec/04-requirements.md` REQ-8 の対象事項のためユーザー承認が必要。イシュー #809 の実装スコープには含まない）。

## 実測記録（未実施）

| size | base median TFLOPS | head median TFLOPS | head/base | spread base | spread head |
|------|--------------------|--------------------|-----------|--------------|--------------|
| 256  | (Mac 実機計測待ち) | | | | |
| 512  | | | | | |
| 1024 | | | | | |
| 2048 | | | | | |
| 4096 | | | | | |

サーマル記録（`pmset -g therm` 実行前後）: (Mac 実機計測待ち)

## 参照

- `crates/backend-metal/src/shaders/gemm.metal`（`FINE_BARRIER_ENABLED` function constant 宣言・staged 経路挿入箇所）
- `crates/backend-metal/src/pipeline.rs::make_pipeline_with_constants`（index 8 特殊化）
- `crates/backend-metal/src/tile.rs::FINE_BARRIER_ENABLED`（本番既定値。コミット状態既定 `false` ロックテスト
  `fine_barrier_enabled_is_false_by_default`）
- `crates/backend-metal/src/gemm.rs::MetalGemm::new_with_fine_barrier`（A/B 計測専用入口）
- `crates/backend-metal/examples/gemm_fine_barrier_ab_bench.rs`（A/B ベンチ本体）
- `crates/backend-metal/tests/shader_source_evidence.rs`（Linux CI 上のシェーダ証跡機械検査）
- `docs/backend-metal-buffer-pool-decision.md`（同イシュー #809 のバッファプール要否調査。本ドキュメントとは
  独立の判断対象）
