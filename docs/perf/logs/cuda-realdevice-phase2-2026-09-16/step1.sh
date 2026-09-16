#!/usr/bin/env bash
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai
cd ~/work/rust-ai-library-run
L=$HOME/work/cuda-phase2/make-test-ignored-cuda.log
{ echo "cmd: make test-ignored-cuda 相当（cargo test -p fandhe-ai-backend-cuda --release --all-features --no-fail-fast -- --ignored --nocapture --skip sgd_update_segment_captures_then_replays_bit_identically --skip different_config_key_produces_a_different_segment_key ／ cargo test -p fandhe-ai-backend-cuda --release --test graph_capture_real_device -- --ignored --nocapture --test-threads=1）。--no-fail-fast は全バイナリの一覧取得のために付加"; echo "rev: $(cat .rev-stamp)"; echo "start: $(date -u +%FT%TZ)"; uptime; } > $L
cargo test -p fandhe-ai-backend-cuda --release --all-features --no-fail-fast -- --ignored --nocapture --skip sgd_update_segment_captures_then_replays_bit_identically --skip different_config_key_produces_a_different_segment_key >> $L 2>&1; echo "exit1=$?" >> $L
cargo test -p fandhe-ai-backend-cuda --release --test graph_capture_real_device -- --ignored --nocapture --test-threads=1 >> $L 2>&1; echo "exit2=$?" >> $L
echo "end: $(date -u +%FT%TZ)" >> $L; echo "done." >> $L
