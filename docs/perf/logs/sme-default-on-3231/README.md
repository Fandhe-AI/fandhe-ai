# SME 既定 ON（#3231）の bit 同一確認ログ

- `off-arm.patch`: `SME_PRODUCTION_ENABLED` を `true` から `false` へ戻す 1 行のパッチ（OFF ツリー作成用。`patch_sha256.txt`）。
  - 現在の main は ON（#4165 で再び `true` にした。`docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.19）。OFF ツリーは、`crates/backend-cpu/src/gemm_blis/mod.rs` の `SME_PRODUCTION_ENABLED` の定数 1 行を `false` にして作る。
    `off-arm.patch` と `docs/perf/logs/sme-default-on-ab-4156/on-arm.patch`（`git apply -R`）は、#4165 で周りのコメントが変わったため現在の main には当たらない（2026-11-13 に `git apply --check` で確認）。
  - 本ログの取得時点（#3231）の main も ON だった。その後 #3232 の REJECT で OFF（§5.14）となり、#4165 で ON に戻った。取得時点と現在の main は同一ではないため、再取得するときはコミットを明記する。
- `bitdump/summary.txt`: ON（`RAYON_NUM_THREADS` 未設定・1・2・4・8）と OFF（未設定）の全出力行数と sha256。6 本一致。
- 手順（M4 Max 等の SME 搭載 aarch64。`--release`）:

```sh
cargo test -p fandhe-ai --release --test cpu_sme_gate_bit_dump -- --ignored --nocapture --exact dump_cpu_sme_gate_bits \
  | grep '^out\[' | shasum -a 256
```

- `RAYON_NUM_THREADS` 未設定時は thread_limit・small_shape_cap が効き、env を設定すると無効になるため、プロセス内スレッド数を振る
  単体テスト（`production_gemm_matches_naive_and_neon_forced_bit_exact_across_thread_counts`）とは別の分岐を通る。
- GB10 実機の全出力 bit 同一は本 PR では未再取得（未検証）。旧 sha の実証は `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.6、
  GB10 の非後退と再取得は #3232（`../cpu-gemm-sme-fmopa-1587/gb10/run_bitdump.sh` を流用）。
