//! イシュー #3149（親 #3143）: DGX Spark GB10 で**同一 GPU（`Device::Cuda(0)`）を共有する 2 OS プロセス**が
//! ホスト経由の ring all-reduce で行う分散学習を実測する実機テストと、CI で配線を先に確かめる CPU 自己検査。
//!
//! 位置づけ: #3148（`distributed_cpu_multiprocess.rs`）は CPU・loopback・複数プロセスの正しさを CI で確認済み。
//! 本テストは同じ構成の CUDA 版で、各プロセスが自分の CUDA コンテキストを持ち、勾配を D2H → ホスト TCP
//! all-reduce → H2D で平均する経路（`docs/facade-distributed-host-collective-decision.md` §15）を通す。
//! 手元の CUDA 実機は GB10（GPU 1 基）だけなので複数 GPU の DDP は検証できず、同一 GPU の 2 プロセスで代替する。
//! 性能（スケーリング効率・格上げ条件 (f)）と NCCL は対象外（正しさの確認だけ）。
//!
//! 比較対象（いずれも REQ-2 統一複合判定。`fandhe_ai_backend_cpu::parity::compare`）:
//! - `single_dev`: 同じ初期値・同じ全 8 行を 1 プロセスで `tape_for(device)` の resident 経路で学習した結果
//! - `single_cpu`: 同じ条件を CPU tape で学習した結果
//!
//! 合否は上記 2 比較の `fail_count == 0`（全 optimizer）と、子の正常終了・期限内終了・loopback 待受。
//! ランク間・run 間の bit 一致は記録だけで合否に使わない（記録 §6: GPU 経路の backward 決定性は未検証。
//! 事前登録は `docs/perf/logs/ddp-host-collective-gb10-3149/RULE.txt`）。fail があっても全行を出力してから
//! 最後に 1 回だけ panic し、tolerance・baseline・本体コード・公開面・依存は変更しない。
//!
//! プロセス構成は #3148 と同形（親 = ランク 0、子 = ランク 1。子は `current_exe()` を `--exact` で再実行し、
//! 結果はファイルで返す）。環境変数の接頭辞は #3148 と混線しないよう `FANDHE_DDP_DEV_TEST_` とする。
//! 子のエントリは `#[ignore]` を付けず、合図の環境変数が無ければ no-op（親が `--ignored` 付きでも子は動く）。
//! 実行手順・記入欄は `docs/perf/logs/ddp-host-collective-gb10-3149/README.md`。

mod common;

use std::collections::BTreeMap;
use std::io::Read;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bench_harness::rng::Xorshift64Star;
use common::temp_dir::TempDirGuard;
use fandhe_ai::compat::Sequential;
use fandhe_ai::distributed::{
    Communicator, DistributedConfig, Rendezvous, step_adam, step_adamw, step_sgd,
};
use fandhe_ai::optim::{AdamConfig, AdamWConfig};
use fandhe_ai::{Device, SgdConfig, Tensor};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};

const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;
const FULL_ROWS: usize = 8;
const STEPS: usize = 5;
/// 同一 GPU の 2 プロセスに固定する。
const WORLD: usize = 2;
/// 各プロセスは最初のステップで NVRTC の JIT を払う。片方がコンパイル中に他方が待って偽の
/// `Timeout` にならないよう、#3148（20 秒）より長くする（値は RULE.txt に事前登録）。
const ESTABLISH_TIMEOUT: Duration = Duration::from_secs(120);
const IO_TIMEOUT: Duration = Duration::from_secs(60);
const OP_TIMEOUT: Duration = Duration::from_secs(180);
const SCENARIO_DEADLINE: Duration = Duration::from_secs(600);

const ENV_CHILD: &str = "FANDHE_DDP_DEV_TEST_CHILD";
const ENV_WORLD: &str = "FANDHE_DDP_DEV_TEST_WORLD_SIZE";
const ENV_RANK: &str = "FANDHE_DDP_DEV_TEST_RANK";
const ENV_ADDR: &str = "FANDHE_DDP_DEV_TEST_RDV_ADDR";
const ENV_SESSION: &str = "FANDHE_DDP_DEV_TEST_SESSION";
const ENV_DEVICE: &str = "FANDHE_DDP_DEV_TEST_DEVICE";
const ENV_OUT: &str = "FANDHE_DDP_DEV_TEST_OUT";

/// 検証対象デバイス（環境変数は allowlist で型付きに parse する）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dev {
    Cpu,
    Cuda0,
}

impl Dev {
    fn parse(s: &str) -> Self {
        match s {
            "cpu" => Dev::Cpu,
            "cuda0" => Dev::Cuda0,
            other => panic!("device: {other}"),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Dev::Cpu => "cpu",
            Dev::Cuda0 => "cuda0",
        }
    }
    fn device(self) -> Device {
        match self {
            Dev::Cpu => Device::Cpu,
            Dev::Cuda0 => Device::Cuda(0),
        }
    }
    fn tape(self) -> fandhe_ai::Tape {
        fandhe_ai::tape_for(self.device()).expect("対象デバイスが利用可能な前提のテスト")
    }
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, 0x1111_1111)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, 0x2222_2222)
        .unwrap()
}

fn full_data() -> (Vec<f32>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(0xC0FFEE);
    (
        rng.fill_vec(FULL_ROWS * D_IN),
        rng.fill_vec(FULL_ROWS * D_OUT),
    )
}

/// ランク `r` が受け持つ連続 `FULL_ROWS / WORLD` 行。
fn shard(r: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let rows = FULL_ROWS / WORLD;
    let (x, y) = full_data();
    (
        x[r * rows * D_IN..(r + 1) * rows * D_IN].to_vec(),
        y[r * rows * D_OUT..(r + 1) * rows * D_OUT].to_vec(),
        rows,
    )
}

#[derive(Clone, Copy)]
enum Opt {
    Sgd,
    Adam,
    AdamW,
}

const OPTS: [(Opt, &str); 3] = [
    (Opt::Sgd, "sgd"),
    (Opt::Adam, "adam"),
    (Opt::AdamW, "adamw"),
];

fn flatten(params: &[Tensor<f32>]) -> Vec<f32> {
    params
        .iter()
        .flat_map(|p| p.contiguous().as_slice().unwrap().to_vec())
        .collect()
}

/// `steps` ステップ学習して最終パラメータを返す。`comm` が `None` なら既存の単一プロセス入口
/// （resident 経路）、`Some` なら分散ステップ。tape はステップごとに作り直す
/// （前例: `device_param_store_backend_parity.rs::train_device_resident`）。
fn train(
    dev: Dev,
    opt: Opt,
    x: &[f32],
    y: &[f32],
    rows: usize,
    steps: usize,
    mut comm: Option<&mut Communicator>,
) -> Vec<f32> {
    let model = build_model();
    let x_t = tensor(x.to_vec(), &[rows, D_IN]);
    let y_t = tensor(y.to_vec(), &[rows, D_OUT]);
    let mut store = model.init_device_param_store(&dev.tape()).unwrap();
    for _ in 0..steps {
        let tape = dev.tape();
        let xv = tape.var(&x_t);
        let yv = tape.var(&y_t);
        let pred = model.forward_resident(&tape, &xv, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &yv).unwrap();
        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        match (opt, comm.as_deref_mut()) {
            (Opt::Sgd, None) => tape
                .step_device_param_store(&mut store, &grads, &SgdConfig::new(0.05))
                .unwrap(),
            (Opt::Adam, None) => tape
                .step_device_param_store_adam(&mut store, &grads, &AdamConfig::default())
                .unwrap(),
            (Opt::AdamW, None) => tape
                .step_device_param_store_adamw(&mut store, &grads, &AdamWConfig::default())
                .unwrap(),
            (Opt::Sgd, Some(c)) => {
                step_sgd(&tape, &mut store, &grads, &SgdConfig::new(0.05), c).unwrap()
            }
            (Opt::Adam, Some(c)) => {
                step_adam(&tape, &mut store, &grads, &AdamConfig::default(), c).unwrap()
            }
            (Opt::AdamW, Some(c)) => {
                step_adamw(&tape, &mut store, &grads, &AdamWConfig::default(), c).unwrap()
            }
        }
    }
    flatten(&dev.tape().sync_device_param_store_to_host(&store).unwrap())
}

fn config(rank: usize, addr: SocketAddr, session: u64) -> DistributedConfig {
    DistributedConfig::builder(WORLD, rank, addr)
        .session_id(session)
        .establish_timeout(ESTABLISH_TIMEOUT)
        .io_timeout(IO_TIMEOUT)
        .op_timeout(OP_TIMEOUT)
        .build()
        .unwrap()
}

/// establish 後・計測ループ前の JIT 前払い。両ランクが同じ処理をするので collective の同期は崩れない。
/// 使い捨て store で集合なしの 1 ステップを回すだけで、計測対象の store には触れない。
fn prewarm(dev: Dev, rank: usize) {
    let (x, y, rows) = shard(rank);
    for (opt, _) in OPTS {
        let _ = train(dev, opt, &x, &y, rows, 1, None);
    }
}

fn train_all(dev: Dev, rank: usize, comm: &mut Communicator) -> BTreeMap<String, Vec<f32>> {
    let (x, y, rows) = shard(rank);
    let mut out = BTreeMap::new();
    for (opt, name) in OPTS {
        out.insert(
            name.to_string(),
            train(dev, opt, &x, &y, rows, STEPS, Some(&mut *comm)),
        );
    }
    out
}

/// シナリオ全体の期限を**実行中に**強制する番犬。開始から `SCENARIO_DEADLINE` 経過しても
/// `Drop`（正常完了・panic）されなければプロセスごと終了する。親の CUDA 呼び出しが停止しても
/// 期限で止まり、子は自身の番犬（同じ期限）で終了する。
struct DeadlineWatchdog {
    done: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl DeadlineWatchdog {
    fn start(label: &'static str) -> Self {
        use std::sync::atomic::{AtomicBool, Ordering};
        let done = std::sync::Arc::new(AtomicBool::new(false));
        let flag = done.clone();
        let started = Instant::now();
        std::thread::spawn(move || {
            while !flag.load(Ordering::Relaxed) {
                if started.elapsed() > SCENARIO_DEADLINE {
                    eprintln!(
                        "DDP_GB10 {label}: シナリオ期限 {SCENARIO_DEADLINE:?} 超過。強制終了"
                    );
                    std::process::exit(124);
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        Self { done }
    }
}

impl Drop for DeadlineWatchdog {
    fn drop(&mut self) {
        self.done.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------- 子プロセス側

struct ChildEnv {
    rank: usize,
    addr: SocketAddr,
    session: u64,
    dev: Dev,
    out: PathBuf,
}

fn env_var(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("環境変数 {name} が未設定"))
}

impl ChildEnv {
    /// 子モードの合図が無ければ `None`（通常の `cargo test` では no-op）。値は使う前に型付きで検証する。
    fn from_env() -> Option<Self> {
        if std::env::var(ENV_CHILD).ok().as_deref() != Some("1") {
            return None;
        }
        let world: usize = env_var(ENV_WORLD).parse().expect("world_size");
        assert_eq!(world, WORLD, "world_size は 2 固定");
        let rank: usize = env_var(ENV_RANK).parse().expect("rank");
        assert_eq!(rank, 1, "rank は 1 固定");
        let addr: SocketAddr = env_var(ENV_ADDR).parse().expect("rdv addr");
        assert!(addr.ip().is_loopback() && addr.port() != 0, "addr: {addr}");
        let session: u64 = env_var(ENV_SESSION).parse().expect("session");
        let dev = Dev::parse(&env_var(ENV_DEVICE));
        let out = PathBuf::from(env_var(ENV_OUT));
        assert!(out.starts_with(std::env::temp_dir()), "out: {out:?}");
        Some(Self {
            rank,
            addr,
            session,
            dev,
            out,
        })
    }
}

fn hex_line(v: &[f32]) -> String {
    v.iter()
        .map(|f| format!("{:08x}", f.to_bits()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// tmp へ書いてから rename し、親が書きかけを読まないようにする。
fn write_atomic(path: &Path, body: &str) {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, body).expect("結果の書き出し");
    std::fs::rename(&tmp, path).expect("結果の rename");
}

/// 子のエントリ。親が `current_exe()` で `--exact child_rank_entry_device` として再実行する。
#[test]
fn child_rank_entry_device() {
    let Some(env) = ChildEnv::from_env() else {
        return;
    };
    let _watchdog = DeadlineWatchdog::start("child");
    let cfg = config(env.rank, env.addr, env.session);
    let mut comm = Rendezvous::bind(&cfg).unwrap().establish().unwrap();
    assert_eq!((comm.rank(), comm.world_size()), (env.rank, WORLD));
    prewarm(env.dev, env.rank);
    let result = train_all(env.dev, env.rank, &mut comm);
    let mut body = String::new();
    for (name, p) in &result {
        body.push_str(&format!("opt={name}\n{}\n", hex_line(p)));
    }
    write_atomic(&env.out, &body);
}

// ---------------------------------------------------------------- 親プロセス側

/// 子プロセスの RAII 保持。親が panic しても孤児を残さない。
struct ChildGuard {
    child: Child,
    stderr: Option<JoinHandle<String>>,
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_child(dev: Dev, addr: SocketAddr, session: u64, out: &Path) -> ChildGuard {
    let exe = std::env::current_exe().expect("current_exe");
    let mut child = Command::new(exe)
        // --nocapture: panic メッセージ（CUDA 初期化・step の unwrap 失敗等）を stdout の失敗レポートでなく
        // stderr へ出させ、下の回収スレッドで保持する（stdout は捨てるため必須）。
        .args([
            "child_rank_entry_device",
            "--exact",
            "--test-threads=1",
            "--nocapture",
        ])
        .env(ENV_CHILD, "1")
        .env(ENV_WORLD, WORLD.to_string())
        .env(ENV_RANK, "1")
        .env(ENV_ADDR, addr.to_string())
        .env(ENV_SESSION, session.to_string())
        .env(ENV_DEVICE, dev.name())
        .env(ENV_OUT, out)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("子プロセスの起動");
    let mut pipe = child.stderr.take().expect("stderr pipe");
    // パイプ詰まりを避けるため別スレッドで回収し、末尾 4 KiB だけ保持する。
    let stderr = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = pipe.read_to_end(&mut buf);
        let tail = buf.len().saturating_sub(4096);
        String::from_utf8_lossy(&buf[tail..]).into_owned()
    });
    ChildGuard {
        child,
        stderr: Some(stderr),
    }
}

/// 子の終了を期限付きでポーリングする。期限超過は kill して panic。
fn wait_child(c: &mut ChildGuard, deadline: Duration) -> (ExitStatus, String) {
    let start = Instant::now();
    loop {
        if let Some(st) = c.child.try_wait().expect("try_wait") {
            let tail = c.stderr.take().map(|h| h.join().unwrap_or_default());
            return (st, tail.unwrap_or_default());
        }
        if start.elapsed() > deadline {
            let _ = c.child.kill();
            panic!("期限 {deadline:?} 内に終了しない子ランク 1（ハング）");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn parse_train_result(path: &Path) -> BTreeMap<String, Vec<f32>> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut lines = text.lines();
    let mut map = BTreeMap::new();
    while let Some(head) = lines.next() {
        let name = head.strip_prefix("opt=").expect("opt= 行");
        let data = lines.next().expect("データ行");
        let v = data
            .split_whitespace()
            .map(|h| f32::from_bits(u32::from_str_radix(h, 16).expect("16 進")))
            .collect();
        map.insert(name.to_string(), v);
    }
    map
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits())
}

type RankResult = BTreeMap<String, Vec<f32>>;

/// 1 回の分散 run。ランク 0（親）と子（ランク 1）の optimizer 名 → 最終パラメータを返す。
/// 待受が loopback であることの証跡行もここで出す。
fn run_dist(dev: Dev, session: u64, scenario_start: Instant) -> (RankResult, RankResult) {
    let dir = TempDirGuard::new("ddp-dev");
    let cfg = config(0, "127.0.0.1:0".parse().unwrap(), session);
    let non_loopback_allowed = cfg.is_non_loopback_allowed();
    assert!(cfg.rendezvous_addr().ip().is_loopback() && !non_loopback_allowed);
    let rdv = Rendezvous::bind(&cfg).unwrap();
    let addr = rdv.local_addr().expect("rank 0 は実アドレスを持つ");
    assert!(addr.ip().is_loopback() && addr.port() != 0, "{addr}");
    println!("DDP_GB10 rdv_addr={addr} loopback=true non_loopback_allowed={non_loopback_allowed}");
    let out = dir.path().join("rank1.out");
    let mut child = spawn_child(dev, addr, session, &out);
    let mut comm = rdv.establish().unwrap();
    assert_eq!((comm.rank(), comm.world_size()), (0, WORLD));
    prewarm(dev, 0);
    let rank0 = train_all(dev, 0, &mut comm);
    drop(comm);
    let (st, tail) = wait_child(
        &mut child,
        SCENARIO_DEADLINE.saturating_sub(scenario_start.elapsed()),
    );
    assert!(st.success(), "rank 1 が異常終了 {st:?}\n{tail}");
    (rank0, parse_train_result(&out))
}

fn report_line(dev: Dev, opt: &str, vs: &str, a: &[f32], b: &[f32]) -> usize {
    let r = fandhe_ai_backend_cpu::parity::compare(a, b).expect("長さ一致");
    println!(
        "DDP_GB10 device={} opt={opt} vs={vs} total={} fail={} max_abs={:e} max_rel={:e} mean_abs={:e}",
        dev.name(),
        r.total,
        r.fail_count,
        r.max_abs_diff,
        r.max_rel_err,
        r.mean_abs_diff
    );
    r.fail_count
}

/// ハーネス本体。fail があっても全行を出力してから最後に 1 回だけ panic する。
fn run_harness(dev: Dev) {
    let started = Instant::now();
    let _watchdog = DeadlineWatchdog::start("parent");
    let (run_a0, run_a1) = run_dist(dev, 0xD157_3149_0000, started);
    let (run_b0, _) = run_dist(dev, 0xD157_3149_1000, started);
    let (x, y) = full_data();
    let mut failures = 0usize;
    for (opt, name) in OPTS {
        let single_dev = train(dev, opt, &x, &y, FULL_ROWS, STEPS, None);
        let single_cpu = train(Dev::Cpu, opt, &x, &y, FULL_ROWS, STEPS, None);
        let dist = &run_a0[name];
        failures += report_line(dev, name, "single_dev", dist, &single_dev);
        failures += report_line(dev, name, "single_cpu", dist, &single_cpu);
        // 参考（判定対象外）: fail の切り分け用に single_dev 対 single_cpu も出す。
        let _ = report_line(
            dev,
            name,
            "ref_single_dev_vs_single_cpu",
            &single_dev,
            &single_cpu,
        );
        println!(
            "DDP_GB10 bits_equal kind=rank0_vs_rank1 opt={name} value={}",
            bits_equal(dist, &run_a1[name])
        );
        println!(
            "DDP_GB10 bits_equal kind=runA_vs_runB opt={name} value={}",
            bits_equal(dist, &run_b0[name])
        );
    }
    let elapsed = started.elapsed();
    println!(
        "DDP_GB10 device={} elapsed_s={:.1}",
        dev.name(),
        elapsed.as_secs_f64()
    );
    assert!(
        elapsed < SCENARIO_DEADLINE,
        "シナリオ全体が期限超過: {elapsed:?}"
    );
    assert_eq!(
        failures, 0,
        "統一複合判定の fail 要素数（上の DDP_GB10 行を参照）"
    );
}

/// CUDA 実機（GB10）の本測定。同一 GPU を 2 プロセスが共有する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/ddp-host-collective-gb10-3149/README.md 参照"]
fn cuda_two_process_ddp_on_shared_gpu_matches_single_cuda_and_cpu() {
    run_harness(Dev::Cuda0);
}

/// CI 用の自己検査。同じハーネスを CPU で回し、環境変数・デバイス引数・出力書式・比較の配線誤りを
/// 実機に持ち込む前に捕まえる（外部通信なし・待受は loopback のみ）。
#[test]
fn harness_self_check_on_cpu_device_two_process() {
    run_harness(Dev::Cpu);
}
