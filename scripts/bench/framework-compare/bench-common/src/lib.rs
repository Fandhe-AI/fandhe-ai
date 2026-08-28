//! Shared benchmark utilities: deterministic RNG (xorshift64*), robust
//! statistics (median / Q1 / Q3), JSON-line result output, and CLI parsing.
//!
//! All frameworks under test consume identical input data generated from the
//! same seed and the same generation formula (`Xorshift64Star::fill_vec`).

use std::io::Write;
use std::time::Duration;

/// Typed error for the shared bench utilities. Bench binaries propagate this
/// (as `Box<dyn Error>`) up to `main`, diagnose on stderr, and exit non-zero;
/// no `panic!` / `unwrap` / `expect` on these paths (coding-rust.md).
#[derive(Debug)]
pub enum BenchError {
    /// `stats` was called with an empty sample set.
    EmptySamples,
    /// Result-file I/O failed (open / create-dir / write).
    Io {
        path: String,
        source: std::io::Error,
    },
    /// A CLI flag had an unparsable value.
    InvalidArg { flag: &'static str, value: String },
    /// `--mode` was neither `fresh` nor `reuse` (allowlist validation; see
    /// `parse_cli`). A3 インジェクション対策と同じ思想で、未知の値を
    /// フレームワーク側の分岐へそのまま流さず、ここで fail-fast する。
    InvalidMode { value: String },
}

impl std::fmt::Display for BenchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchError::EmptySamples => write!(f, "MEASURE_ERROR: no samples to aggregate"),
            BenchError::Io { path, source } => {
                write!(
                    f,
                    "MEASURE_ERROR: results file I/O failed ({path}): {source}"
                )
            }
            BenchError::InvalidArg { flag, value } => {
                write!(
                    f,
                    "MEASURE_ERROR: {flag} must be an integer (got '{value}')"
                )
            }
            BenchError::InvalidMode { value } => {
                write!(
                    f,
                    "MEASURE_ERROR: --mode must be 'fresh' or 'reuse' (got '{value}')"
                )
            }
        }
    }
}

impl std::error::Error for BenchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BenchError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// xorshift64* PRNG (Vigna 2016). Deterministic across platforms.
pub struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    pub fn new(seed: u64) -> Self {
        // state must be non-zero
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // take top 24 bits for a uniform float in [0,1)
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }

    /// Fill a Vec<f32> with values in [-0.5, 0.5). Shared generation formula
    /// for every framework so inputs are bit-identical.
    pub fn fill_vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() - 0.5).collect()
    }
}

/// Fixed seeds shared by every bench binary.
pub const SEED_A: u64 = 0xA11CE;
pub const SEED_B: u64 = 0xB0B;
pub const SEED_X: u64 = 0xDA7A_0001;
pub const SEED_Y: u64 = 0xDA7A_0002;
pub const SEED_L1: u64 = 0x1111_1111;
pub const SEED_L2: u64 = 0x2222_2222;

/// Measurement protocol constants (matches fandhe-ai's convention, scaled up:
/// warmup 20, measure 20, report median + Q1/Q3).
pub const WARMUP_ITERS: usize = 20;
pub const MEASURE_ITERS: usize = 20;

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub median_s: f64,
    pub q1_s: f64,
    pub q3_s: f64,
}

/// Median / Q1 / Q3 over a set of durations (linear interpolation quartiles).
/// Errors on an empty sample set instead of panicking.
pub fn stats(durations: &[Duration]) -> Result<Stats, BenchError> {
    if durations.is_empty() {
        return Err(BenchError::EmptySamples);
    }
    let mut secs: Vec<f64> = durations.iter().map(|d| d.as_secs_f64()).collect();
    // total_cmp: Duration::as_secs_f64 never yields NaN, and a total order
    // avoids the partial_cmp unwrap.
    secs.sort_by(f64::total_cmp);
    let q = |p: f64| -> f64 {
        let idx = p * (secs.len() - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        let frac = idx - lo as f64;
        secs[lo] * (1.0 - frac) + secs[hi] * frac
    };
    Ok(Stats {
        median_s: q(0.5),
        q1_s: q(0.25),
        q3_s: q(0.75),
    })
}

/// One benchmark result, serialized as one JSON line.
pub struct Record<'a> {
    pub framework: &'a str,
    pub framework_version: &'a str,
    pub task: &'a str,
    pub device: &'a str,
    pub size: usize,
    pub stats: Stats,
    /// GFLOP/s for GEMM (2N^3 / median); None for other tasks.
    pub gflops: Option<f64>,
    /// throughput: forward passes (batches) per second for the infer task
    /// (each pass processes one batch of 64 inputs; this is batches/s, not
    /// single-sample inferences/s); None if not applicable.
    pub throughput_per_s: Option<f64>,
    /// checksum of the result tensor (proves host materialization + sanity).
    pub checksum: f64,
    pub warmup: usize,
    pub iters: usize,
    /// 計測モード（"fresh" = 毎回新規デバイス/tape・"reuse" = 使い回し）。
    /// 常に emit する。既存コミット済み JSONL（本フィールド追加前）にはこの
    /// キーが無いため、読み手（summarize.py）は欠損を "fresh" として扱う
    /// （イシュー #925。互換維持）。
    pub mode: &'a str,
    /// reuse モードのみ Some: デバイス/tape 構築から初回計測までの
    /// 初期化 1 回分の経過時間（秒）。fresh モードは常に None（emit しない）。
    pub init_s: Option<f64>,
}

impl Record<'_> {
    pub fn to_json_line(&self) -> String {
        let mut s = format!(
            "{{\"framework\":\"{}\",\"version\":\"{}\",\"task\":\"{}\",\"device\":\"{}\",\"size\":{},\"median_s\":{:.9},\"q1_s\":{:.9},\"q3_s\":{:.9}",
            self.framework,
            self.framework_version,
            self.task,
            self.device,
            self.size,
            self.stats.median_s,
            self.stats.q1_s,
            self.stats.q3_s,
        );
        if let Some(g) = self.gflops {
            s.push_str(&format!(",\"gflops\":{g:.3}"));
        }
        if let Some(t) = self.throughput_per_s {
            s.push_str(&format!(",\"throughput_per_s\":{t:.3}"));
        }
        s.push_str(&format!(
            ",\"checksum\":{:.6},\"warmup\":{},\"iters\":{}",
            self.checksum, self.warmup, self.iters
        ));
        s.push_str(&format!(",\"mode\":\"{}\"", self.mode));
        if let Some(init) = self.init_s {
            s.push_str(&format!(",\"init_s\":{init:.9}"));
        }
        s.push('}');
        s
    }

    /// Print to stdout and append to `path` (JSONL). I/O failures are
    /// propagated (typed) instead of panicking, so a bench binary exits with a
    /// diagnostic and the sweep script records the combination as skipped.
    pub fn emit(&self, path: &str) -> Result<(), BenchError> {
        let io_err = |source: std::io::Error| BenchError::Io {
            path: path.to_string(),
            source,
        };
        let line = self.to_json_line();
        println!("{line}");
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).map_err(io_err)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(io_err)?;
        writeln!(f, "{line}").map_err(io_err)?;
        Ok(())
    }
}

/// Parsed CLI: --task gemm|train|infer --device cpu|metal --size N
/// [--mode fresh|reuse] [--out path]
pub struct Cli {
    pub task: String,
    pub device: String,
    pub size: usize,
    pub out: String,
    /// "fresh"（既定。既存プロトコルと同一）または "reuse"（イシュー #925。
    /// デバイス/tape を使い回し、初期化コストとカーネル実行を分離計測する）。
    pub mode: String,
}

/// Parse the CLI arguments. An unparsable `--size` / unknown `--mode` is a
/// typed error (diagnosed at the binary boundary), not a panic. `--mode` は
/// allowlist（fresh/reuse）検証のみで、それ以外の値をフレームワーク側の
/// 分岐へそのまま流さない（security.md A03 相当の方針）。
pub fn parse_cli() -> Result<Cli, BenchError> {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let size = match get("--size") {
        Some(s) => s.parse().map_err(|_| BenchError::InvalidArg {
            flag: "--size",
            value: s,
        })?,
        None => 256,
    };
    let mode = get("--mode").unwrap_or_else(|| "fresh".into());
    if mode != "fresh" && mode != "reuse" {
        return Err(BenchError::InvalidMode { value: mode });
    }
    Ok(Cli {
        task: get("--task").unwrap_or_else(|| "gemm".into()),
        device: get("--device").unwrap_or_else(|| "cpu".into()),
        size,
        out: get("--out").unwrap_or_else(|| "results/raw/results.jsonl".into()),
        mode,
    })
}

pub fn gemm_gflops(n: usize, median_s: f64) -> f64 {
    2.0 * (n as f64).powi(3) / median_s / 1e9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic() {
        let mut a = Xorshift64Star::new(42);
        let mut b = Xorshift64Star::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn stats_quartiles() {
        let d: Vec<Duration> = (1..=5).map(Duration::from_secs).collect();
        let s = stats(&d).expect("non-empty samples");
        assert!((s.median_s - 3.0).abs() < 1e-9);
        assert!((s.q1_s - 2.0).abs() < 1e-9);
        assert!((s.q3_s - 4.0).abs() < 1e-9);
    }

    #[test]
    fn stats_empty_is_typed_error() {
        assert!(matches!(stats(&[]), Err(BenchError::EmptySamples)));
    }

    fn sample_record<'a>(mode: &'a str, init_s: Option<f64>) -> Record<'a> {
        Record {
            framework: "fandhe-ai",
            framework_version: "0.3.0",
            task: "gemm",
            device: "cpu",
            size: 256,
            stats: Stats {
                median_s: 0.001,
                q1_s: 0.0009,
                q3_s: 0.0011,
            },
            gflops: Some(123.456),
            throughput_per_s: None,
            checksum: 1.0,
            warmup: 20,
            iters: 20,
            mode,
            init_s,
        }
    }

    #[test]
    fn json_line_fresh_mode_omits_init_s() {
        let line = sample_record("fresh", None).to_json_line();
        assert!(line.contains("\"mode\":\"fresh\""));
        assert!(!line.contains("init_s"));
    }

    #[test]
    fn json_line_reuse_mode_includes_init_s() {
        let line = sample_record("reuse", Some(0.45)).to_json_line();
        assert!(line.contains("\"mode\":\"reuse\""));
        assert!(line.contains("\"init_s\":0.450000000"));
    }

    #[test]
    fn invalid_mode_is_typed_error() {
        let err = BenchError::InvalidMode {
            value: "bogus".to_string(),
        };
        assert!(
            err.to_string()
                .contains("--mode must be 'fresh' or 'reuse'")
        );
    }
}
