# NEON マイクロカーネル vfmaq_laneq_f32 化の実測記録（#552）

イシュー #552「NEON マイクロカーネルの A オペランドを `vfmaq_laneq_f32` 化（broadcast 命令の排除）」の
実測記録。`crates/backend-cpu/src/gemm_blis/microkernel/neon.rs` の k ループを、A の各行値をスカラー
読み出し → `vdupq_n_f32` で明示 broadcast → `vfmaq_f32` する 2 命令方式から、`vld1q_f32` によるベクタ
ロード → レーン選択 FMA（`vfmaq_laneq_f32`。単一命令 `FMLA v.4s, v.4s, v.s[lane]`）方式へ転換した。

**本ドキュメントは REQ-8 の下限値・数値一致許容誤差を一切変更しない**。

## 状態: 実装・クロス型検査・x86 側リグレッションまで完了。aarch64 実機での bit 一致・スループット実測は未実施（環境ゲートで未達）

### 実行環境ゲート判定（本イシュー実装セッション時点）

受け入れ基準の bit 完全一致・スループット向上確認は aarch64 実機（Apple M4 Max・DGX Spark GB10 の
Grace CPU）でのみ有効という前提のもと、実装セッション開始時に以下を判定した:

1. `uname -sm` → `Linux x86_64`（本開発環境。実測）。aarch64 ではないため NEON 経路は実行不能。
2. `docs/real-hardware-verification-env.local.md`（Git 管理外のローカル用ファイル）→ 不存在（実測）。
   実機（M4 Max ローカル / `$CUDA_NODE` の Grace CPU）への接続情報が定義されていないため到達不可。
3. `qemu-aarch64`／`qemu-aarch64-static`／`aarch64-linux-gnu-gcc` の存在確認（`command -v`）→
   いずれも不在（実測）。`/proc/sys/fs/binfmt_misc/` にも aarch64 実行用エントリなし。よって
   `cargo test --target aarch64-unknown-linux-gnu` によるエミュレーション実行も本環境では不可能
   （実行を伴う代替経路がないことを実測で確認済み。クロス `cargo check`／`cargo clippy` による
   コンパイル検証のみが本環境で可能な検証の上限）。

**結論**: 本セッションでは aarch64 実機に到達できないため、実装・コンパイル検証（クロス
`cargo check`）・x86_64 側のリグレッション確認（scalar/AVX2 経路に影響がないこと）までを実施し、
**実測値の捏造・placeholder 値での完了扱いは行わない**（fail-closed。#488
`docs/perf/cpu-gemm-baseline-remeasurement.md` と同じ前例）。実機での bit 一致・スループット計測は
aarch64 実機へアクセス可能な後続セッション・Agent（`bench-runner` 委譲想定）が引き継いで実施する。

## 変更内容（実装済み・コンパイル検証済み）

- `crates/backend-cpu/src/gemm_blis/microkernel/neon.rs`: k ループ内で `ap[p * MR..]` / `ap[p * MR + 4..]`
  を `vld1q_f32` で 1 回ずつロードし（A の 8 行を 2 レジスタへ一括取得。`pack.rs` の p-major・mr 方向
  連続配置 `dst[p * mr + i]` を前提とする配置と整合）、行ごとに `vfmaq_laneq_f32::<LANE>` でレーンを
  直接 FMA へ渡す方式へ書き換えた。`vdupq_n_f32`（DUP）・スカラーロードは排除した。
- 行 i とレーンの対応（a0: 行 0..3 = レーン 0..3、a1: 行 4..7 = レーン 0..3）・行昇順・`[0]` → `[1]`
  順・p 昇順の FMA 連鎖順序は旧実装と同一に保った（bit 完全一致契約の前提）。
- 冒頭 3 つの `assert_eq!`（REQ-8 境界検査）は変更なし。

## 検証済み事項（本セッションで実施）

| 検証 | コマンド | 結果 |
|---|---|---|
| fmt | `cargo fmt --all` | 差分なし |
| clippy（x86_64） | `cargo clippy --workspace --all-targets --all-features -- -D warnings` | 警告なし |
| NEON 型検査 | `cargo check -p backend-cpu --target aarch64-unknown-linux-gnu` | 成功 |
| NEON クロス型検査 | `cargo check -p backend-cpu --target aarch64-apple-darwin` | 成功 |
| NEON clippy 型検査 | `cargo clippy -p backend-cpu --target aarch64-unknown-linux-gnu --all-targets -- -D warnings` | 警告なし |
| リグレッション | `cargo test --workspace` | 全 pass（0 failed。x86_64 では scalar/AVX2 経路のみ実行され NEON 経路は実行対象外） |

## 未実測（fail-closed・後続セッションへの引き継ぎ事項）

以下 2 項目は受け入れ基準だが、本セッションでは aarch64 実機に到達できないため未実施:

1. **bit 完全一致**: `cargo test -p backend-cpu --release --test gemm_blis_parity` を aarch64 実機
   （M4 Max または Grace CPU）で実行する。aarch64 では `Isa::detect` が無条件に NEON を選ぶため、
   このテスト実行自体が新方式 NEON 経路の `gemm_naive` との bit 完全一致検証になる。**不一致の場合は
   実装を revert し、tolerance の変更・テスト側の調整で通すことは行わない**（`.claude/rules/coding-rust.md`
   「バックエンド間数値一致テストの許容誤差を単独で緩和しない」）。
2. **スループット向上**: 変更前（`main` の HEAD）・変更後それぞれで M=N=K=4096 を
   `tests/gemm_blis_perf.rs`（`--ignored`）または `bench_harness::protocol::run` で 5 回計測し中央値を
   採用、変更後 > 変更前を確認する。変更後が上回らない場合も受け入れ基準未達として本ドキュメント・
   PR に実測値を添えて報告する（不採用の判断も含め、実測なしに達成を偽装しない）。

### 再現コマンド（後続セッション向け）

```bash
# bit 完全一致（aarch64 実機）
cargo test -p backend-cpu --release --test gemm_blis_parity

# スループット（aarch64 実機。--ignored ハーネス使用）
cargo test -p backend-cpu --release --test gemm_blis_perf -- --ignored --nocapture
```

実機接続手順は `docs/real-hardware-verification-env.md` §3-4（rsync 転送・除外フィルタ厳守）を参照する。
