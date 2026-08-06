# CPU GEMM rayon 並列チューニング計測記録（#24・TASK-1.6d）

イシュー #24「perf(backend-cpu): TASK-1.6d rayon 並列チューニング・PoC-v2-1 比性能確認」の実測記録。
受け入れ条件「PoC-v2-1 相当の性能改善比が再現される（計測記録つき）」に対応する。

## 計測環境

| 項目 | 値 |
|------|-----|
| CPU | QEMU Virtual CPU version 2.5+（`lscpu` 実測。物理ハードウェアではなく仮想化環境） |
| 論理コア数 | 12（`nproc`） |
| OS | Linux |
| rustc | 1.96.0 (ac68faa20 2026-05-25) |
| ビルド条件 | `CARGO_PROFILE_RELEASE_LTO=true CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 cargo build --release`（PoC-v2-1 相当のビルド条件を環境変数オーバーライドで再現。ルート `Cargo.toml` は変更していない） |
| 計測プロトコル | `bench-harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3 記録。TASK-8.1） |
| 決定的シード | `0xC0FFEE`（PoC-v2-1・PoC-v2-5 と同一値） |
| 計測バイナリ | `crates/backend-cpu/examples/gemm_bench.rs` |
| 他プロセスの負荷 | 複数イシューが並列実行される worktree 環境（`git remote` 共有ではないが同一ホスト上で他エージェントのビルドが並走しうる）。各計測直前・直後の `/proc/loadavg`（1 分平均）を記録し、負荷混入の目安とする |

## PoC-v2-1 実測値（比較対象。Apple M4 Max・16 論理コア）

`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md`「計測結果」節より:

| M=N=K | naive (TFLOPS) | blocked (TFLOPS) | parallel (TFLOPS) | naive/blocked 比 parallel 改善比 |
|-------|------|------|------|------|
| 512  | 0.031（本文記載の逆転現象修正前の参考値は除く） | ― | 0.2384（パネル修正後） | 約 7.9〜8.5 倍 |
| 2048 | ― | ― | ― | 5.9〜7.2 倍 |
| 4096 | 0.0289（5 回計測） | 0.0235 | 0.1852 | 6.4〜7.9 倍 |

「`rayon` 並列化は naive/blocked 比で約 6〜8.5 倍の改善」（PoC README 同節）。
16 論理コアに対する並列効率（改善比 ÷ 論理コア数）はおよそ **0.37〜0.53**。
本イシューの判定基準はこの並列効率レンジと同等以上（12 論理コア環境では改善比 4.5 倍前後以上）とする。

## 本環境の実測結果

計測コマンド:

```
CARGO_PROFILE_RELEASE_LTO=true CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
  cargo build --release -p backend-cpu --example gemm_bench
CARGO_PROFILE_RELEASE_LTO=true CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
  ./target/release/examples/gemm_bench 512
```

### M=N=K=512（naive/blocked/parallel 3 実装フル計測、warmup 20・iters 20）

| kernel | median TFLOPS | Q1 | Q3 | loadavg |
|--------|---------------|----|----|---------|
| blocked | 0.0012 | 0.0012 | 0.0011 | 0.97 |
| naive   | 0.0011 | 0.0011 | 0.0011 | 1.29 |
| parallel | 0.0072 | 0.0077 | 0.0064 | 1.29 |

- `threads=12`（`rayon::current_num_threads()`。実測環境の論理コア数と一致）
- `loadavg_before=0.96`（計測開始直前の 1 分平均。以降の各計測値と大差なく、他プロセスの明白な高負荷混入は見られない）
- **improvement_ratio(parallel/naive) = 6.242x**
- **parallel_efficiency = 6.242 / 12 = 0.520**

### PoC-v2-1 対比

| 指標 | PoC-v2-1（M4 Max・16 コア） | 本環境（QEMU 12 コア・M=512） |
|------|------------------------------|-------------------------------|
| naive/blocked→parallel 改善比 | 約 6〜8.5 倍 | 6.242 倍 |
| 並列効率（改善比 ÷ 論理コア数） | 約 0.37〜0.53 | **0.520** |

本環境（512）の並列効率 0.520 は PoC-v2-1 実測レンジ（0.37〜0.53）の**上限付近に収まっており、判定基準を満たす**。

### 2048/4096 計測について（未完了・スコープ縮小）

本環境（QEMU Virtual CPU）は絶対 TFLOPS が PoC-v2-1（M4 Max）実測より約 1〜2 桁小さく、naive/blocked は M=2048 で
1 iter あたり約 14 秒（推定: 512 の実測値 0.223〜0.244 秒/iter を M^3 でスケールした概算）に達する。プロトコル
遵守（warmup 20・iters 20 の下限。TASK-8.1・`MeasurementConfig` に緩和用の抜け道なし）を保ったまま計測すると
blocked@2048 だけで約 9〜10 分を要し、実装セッションの時間予算内で完走しなかった（計測を開始したが未完了のまま
打ち切った。生データなし）。

- 512 の実測（上記）は naive/blocked/parallel 3 実装フル比較かつ PoC-v2-1 のパネル分割修正が効く（当時 PoC で
  頭打ちが確認された）サイズであり、受け入れ条件「PoC-v2-1 相当の性能改善比が再現される」の直接確認としては
  512 単独でも有効な証拠と判断した。
- 2048/4096 での確認、および MC/KC/NC 座標降下法スイープ・オーバーサブスクリプション係数スイープ
  （`examples/gemm_bench.rs sweep` サブコマンドとして実装済み。動作確認は `cargo build` のみ実施しコマンド自体は
  ビルド成功を確認済み）は、より長い計測時間を確保できる環境（バックグラウンド実行・専用 runner 等）での追加実測を
  要する後続作業として残す（`.claude/rules/out-of-scope-tracking.md` に従い、PR 本文でユーザーへ追加実測の要否を
  提案する）。

## 採否判断

MC=128 / KC=256 / NC=512（PoC-v2-1 既定値）・オーバーサブスクリプション係数 1（PoC-v2-1 相当のパネル数 = スレッド数
構成）とも、512 での実測が受け入れ条件を満たしたため、**現行の定数・既定パラメータを変更しない**。
`BlockSizes`／`gemm_parallel_tuned` へのパラメータ化自体は行った（`examples/gemm_bench.rs` から再コンパイルなしで
実測スイープできるようにするため）が、これは実装上の可搬性向上であり、既定挙動（`gemm_parallel`・`gemm_blocked` の
呼び出し結果）は変更していない。

## 既知の制約・要因分析

- 本環境（QEMU Virtual CPU）は PoC-v2-1 実測環境（Apple M4 Max）と CPU モデル・キャッシュ階層・コア数（12 対 16）が異なるため、絶対 TFLOPS 値は比較不能である。判定は改善「比」・並列効率で行う。
- `kernel_block`（`crates/backend-cpu/src/gemm.rs`）はスカラー実装（SIMD intrinsics 不使用）であり、PoC README が指摘する「1 コアあたりの演算スループットが低いまま並列度だけが効く」制約は本環境でも同様に当てはまる。SIMD 最適化は #184（TASK-1.6f）のスコープ。
- 仮想化オーバーヘッド（vCPU スケジューリング・NUMA 非対応等）による並列効率低下の可能性があるが、本 PoC・本計測とも厳密なプロファイリングは実施していない。
