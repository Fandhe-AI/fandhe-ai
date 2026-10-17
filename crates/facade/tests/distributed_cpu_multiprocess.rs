//! イシュー #3148（親 #3143）: 単一ノード・複数 OS プロセス・loopback TCP の CPU 分散学習が、
//! 単一プロセス学習と一致することを確認する受け入れテスト。
//!
//! 検証する契約（`docs/facade-distributed-host-collective-decision.md` §6・§15・§17）:
//! - world_size 2 と 4 で、SGD・Adam・AdamW の分散ステップを数ステップ進めた最終パラメータが、
//!   同じ初期値・同じ全バッチの単一プロセス学習と統一複合判定（相対誤差 1e-3 未満 または
//!   絶対誤差 1e-5 未満。`tensor-core::tolerance::composite_close_f32`）で一致する。
//!   §15 が本イシューへ委ねた Adam／AdamW の数値検証と、§6 の f32 固定順加算の妥当性検証を担う。
//! - 同じ run 内の全ランクの最終パラメータは bit 一致し、別ポート・別 session で繰り返した run の
//!   ランク 0 の最終パラメータも bit 一致する（決定論性）。W が違う run どうしは比較しない。
//! - 1 ランクが途中で落ちても、生存ランクは期限内に型付きエラー（`DistributedError`）で終了し
//!   ハングせず、通信路は poisoned になり、パラメータは失敗直前から変わらない（fail-closed）。
//!
//! プロセス構成: 親テストプロセスがランク 0、子プロセスがランク 1..W-1。親が `127.0.0.1:0` へ bind
//! して OS 割り当ての実ポートを取り、環境変数で子へ渡すためポートの TOCTOU が無く、全 collective が
//! 実プロセス境界をまたぐ。子は同じテストバイナリの `child_rank_entry` を `current_exe()` で再実行し
//! （前例: `backend-cuda` の `mp_store_child`）、結果は stdout ではなくファイルで親へ返す。
//!
//! 実機非依存（CPU・loopback のみ・外部通信なし）のため `#[ignore]` 分離はしない。新しい tolerance・
//! baseline は作らず、本体コード・公開面・依存は変更しない。

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
    Communicator, DistributedConfig, DistributedError, Rendezvous, step_adam, step_adamw, step_sgd,
};
use fandhe_ai::optim::{AdamConfig, AdamWConfig};
use fandhe_ai::{SgdConfig, Tensor};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};

const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;
/// 全バッチの行数。world_size 2・4 の両方で割り切れる（記録 §6: 各ランクの局所バッチ等分が前提）。
const FULL_ROWS: usize = 8;
const STEPS: usize = 5;
/// 障害テストで落ちるランク。ランク 0 の隣（前: 3・次: 1）ではないため、エラーがリングを伝わって届く。
const CRASH_RANK: usize = 2;
/// 障害テストで落ちる子の終了コード（panic の 101 と区別する）。
const CRASH_EXIT_CODE: i32 = 42;
const OP_TIMEOUT: Duration = Duration::from_secs(20);
/// 1 シナリオ（子プロセス群の終了待ち）の全体期限。
const SCENARIO_DEADLINE: Duration = Duration::from_secs(120);

const ENV_CHILD: &str = "FANDHE_DDP_TEST_CHILD";
const ENV_WORLD: &str = "FANDHE_DDP_TEST_WORLD_SIZE";
const ENV_RANK: &str = "FANDHE_DDP_TEST_RANK";
const ENV_ADDR: &str = "FANDHE_DDP_TEST_RDV_ADDR";
const ENV_SESSION: &str = "FANDHE_DDP_TEST_SESSION";
const ENV_MODE: &str = "FANDHE_DDP_TEST_MODE";
const ENV_OUT: &str = "FANDHE_DDP_TEST_OUT";

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

/// ランク `r` が受け持つ連続 `FULL_ROWS / w` 行。
fn shard(r: usize, w: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let rows = FULL_ROWS / w;
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

/// 1 ステップ分（forward → backward → 更新）。`comm` が `None` なら既存の単一プロセス入口。
fn one_step(
    opt: Opt,
    model: &Sequential,
    store: &mut fandhe_ai::DeviceParamStore,
    x_t: &Tensor<f32>,
    y_t: &Tensor<f32>,
    comm: Option<&mut Communicator>,
) -> Result<(), DistributedError> {
    let tape = fandhe_ai::tape();
    let xv = tape.var(x_t);
    let yv = tape.var(y_t);
    let pred = model.forward_resident(&tape, &xv, store).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &yv).unwrap();
    let grads = tape.backward_device_param_store(&loss, store).unwrap();
    match (opt, comm) {
        (Opt::Sgd, None) => tape
            .step_device_param_store(store, &grads, &SgdConfig::new(0.05))
            .unwrap(),
        (Opt::Adam, None) => tape
            .step_device_param_store_adam(store, &grads, &AdamConfig::default())
            .unwrap(),
        (Opt::AdamW, None) => tape
            .step_device_param_store_adamw(store, &grads, &AdamWConfig::default())
            .unwrap(),
        (Opt::Sgd, Some(c)) => step_sgd(&tape, store, &grads, &SgdConfig::new(0.05), c)?,
        (Opt::Adam, Some(c)) => step_adam(&tape, store, &grads, &AdamConfig::default(), c)?,
        (Opt::AdamW, Some(c)) => step_adamw(&tape, store, &grads, &AdamWConfig::default(), c)?,
    }
    Ok(())
}

fn new_store(model: &Sequential) -> fandhe_ai::DeviceParamStore {
    let tape = fandhe_ai::tape();
    model.init_device_param_store(&tape).unwrap()
}

fn train(
    opt: Opt,
    x: &[f32],
    y: &[f32],
    rows: usize,
    mut comm: Option<&mut Communicator>,
) -> Vec<f32> {
    let model = build_model();
    let x_t = tensor(x.to_vec(), &[rows, D_IN]);
    let y_t = tensor(y.to_vec(), &[rows, D_OUT]);
    let mut store = new_store(&model);
    for _ in 0..STEPS {
        one_step(opt, &model, &mut store, &x_t, &y_t, comm.as_deref_mut()).unwrap();
    }
    flatten(
        &fandhe_ai::tape()
            .sync_device_param_store_to_host(&store)
            .unwrap(),
    )
}

fn config(w: usize, rank: usize, addr: SocketAddr, session: u64) -> DistributedConfig {
    DistributedConfig::builder(w, rank, addr)
        .session_id(session)
        .establish_timeout(Duration::from_secs(20))
        .io_timeout(Duration::from_secs(10))
        .op_timeout(OP_TIMEOUT)
        .build()
        .unwrap()
}

fn assert_close(actual: f32, expected: f32, ctx: &str) {
    assert!(
        fandhe_ai_tensor_core::tolerance::composite_close_f32(actual, expected),
        "{ctx}: actual={actual} expected={expected}"
    );
}

fn assert_bits_eq(a: &[f32], b: &[f32], ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: 長さ");
    for (i, (p, q)) in a.iter().zip(b).enumerate() {
        assert_eq!(p.to_bits(), q.to_bits(), "{ctx}: element {i}");
    }
}

/// 失敗系の 1 ランク分の結果。落ちるランクは 2 ステップ目の前に `exit` するので返らない。
struct FailReport {
    err_variant: String,
    poisoned: bool,
    fail_closed: bool,
    elapsed: Duration,
}

/// 障害シナリオ。1 ステップ目は全ランク正常、2 ステップ目の前に `crash` のランクが落ちる。
/// 生存ランクは 2 ステップ目の `Err`・poisoned・パラメータ不変（abandon 後）・経過時間を返す。
fn fail_run(w: usize, rank: usize, comm: &mut Communicator, crash: bool) -> FailReport {
    let model = build_model();
    let (x, y, rows) = shard(rank, w);
    let x_t = tensor(x, &[rows, D_IN]);
    let y_t = tensor(y, &[rows, D_OUT]);
    let mut store = new_store(&model);
    one_step(Opt::Sgd, &model, &mut store, &x_t, &y_t, Some(&mut *comm)).unwrap();
    if crash {
        // ソケットは OS が閉じる。panic（101）と区別できる専用コードで落ちる。
        std::process::exit(CRASH_EXIT_CODE);
    }
    let tape = fandhe_ai::tape();
    let before = flatten(&tape.sync_device_param_store_to_host(&store).unwrap());
    let started = Instant::now();
    let err = one_step(Opt::Sgd, &model, &mut store, &x_t, &y_t, Some(&mut *comm)).unwrap_err();
    let elapsed = started.elapsed();
    let err_variant = match err {
        DistributedError::PeerDisconnected => "PeerDisconnected",
        DistributedError::Timeout { .. } => "Timeout",
        DistributedError::Io(_) => "Io",
        other => panic!("想定外のエラー: {other:?}"),
    }
    .to_string();
    assert!(store.abandon_pending_forward());
    let after = flatten(&tape.sync_device_param_store_to_host(&store).unwrap());
    let fail_closed = before.len() == after.len()
        && before
            .iter()
            .zip(&after)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    FailReport {
        err_variant,
        poisoned: comm.is_poisoned(),
        fail_closed,
        elapsed,
    }
}

// ---------------------------------------------------------------- 子プロセス側

struct ChildEnv {
    world: usize,
    rank: usize,
    addr: SocketAddr,
    session: u64,
    mode: String,
    out: PathBuf,
}

fn env_var(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("環境変数 {name} が未設定"))
}

impl ChildEnv {
    /// 子モードの合図が無ければ `None`（通常の `cargo test` では no-op）。値は型付きで検証する。
    fn from_env() -> Option<Self> {
        if std::env::var(ENV_CHILD).ok().as_deref() != Some("1") {
            return None;
        }
        let world: usize = env_var(ENV_WORLD).parse().expect("world_size");
        assert!((2..=4).contains(&world), "world_size 範囲外: {world}");
        let rank: usize = env_var(ENV_RANK).parse().expect("rank");
        assert!((1..world).contains(&rank), "rank 範囲外: {rank}");
        let addr: SocketAddr = env_var(ENV_ADDR).parse().expect("rdv addr");
        assert!(addr.ip().is_loopback() && addr.port() != 0, "addr: {addr}");
        let session: u64 = env_var(ENV_SESSION).parse().expect("session");
        let mode = env_var(ENV_MODE);
        assert!(
            matches!(mode.as_str(), "train" | "crash" | "survive"),
            "mode: {mode}"
        );
        let out = PathBuf::from(env_var(ENV_OUT));
        assert!(out.starts_with(std::env::temp_dir()), "out: {out:?}");
        Some(Self {
            world,
            rank,
            addr,
            session,
            mode,
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

/// 子のエントリ。親が `current_exe()` で `--exact child_rank_entry` として再実行する。
#[test]
fn child_rank_entry() {
    let Some(env) = ChildEnv::from_env() else {
        return;
    };
    let cfg = config(env.world, env.rank, env.addr, env.session);
    let mut comm = Rendezvous::bind(&cfg).unwrap().establish().unwrap();
    assert_eq!((comm.rank(), comm.world_size()), (env.rank, env.world));
    let mut body = String::new();
    if env.mode == "train" {
        let (x, y, rows) = shard(env.rank, env.world);
        for (opt, name) in OPTS {
            let p = train(opt, &x, &y, rows, Some(&mut comm));
            body.push_str(&format!("opt={name}\n{}\n", hex_line(&p)));
        }
    } else {
        let r = fail_run(env.world, env.rank, &mut comm, env.mode == "crash");
        body.push_str(&format!(
            "err={}\npoisoned={}\nfail_closed={}\nelapsed_ms={}\n",
            r.err_variant,
            r.poisoned,
            r.fail_closed,
            r.elapsed.as_millis()
        ));
    }
    write_atomic(&env.out, &body);
}

// ---------------------------------------------------------------- 親プロセス側

/// 子プロセスの RAII 保持。親が panic しても孤児を残さない。
struct ChildGuard {
    child: Child,
    stderr: Option<JoinHandle<String>>,
    rank: usize,
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_child(
    w: usize,
    rank: usize,
    addr: SocketAddr,
    session: u64,
    mode: &str,
    out: &Path,
) -> ChildGuard {
    let exe = std::env::current_exe().expect("current_exe");
    let mut child = Command::new(exe)
        .args(["child_rank_entry", "--exact", "--test-threads=1"])
        .env(ENV_CHILD, "1")
        .env(ENV_WORLD, w.to_string())
        .env(ENV_RANK, rank.to_string())
        .env(ENV_ADDR, addr.to_string())
        .env(ENV_SESSION, session.to_string())
        .env(ENV_MODE, mode)
        .env(ENV_OUT, out)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("子プロセスの起動");
    let mut pipe = child.stderr.take().expect("stderr pipe");
    // パイプ詰まりによるデッドロックを避けるため別スレッドで回収し、末尾 4 KiB だけ保持する。
    let stderr = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = pipe.read_to_end(&mut buf);
        let tail = buf.len().saturating_sub(4096);
        String::from_utf8_lossy(&buf[tail..]).into_owned()
    });
    ChildGuard {
        child,
        stderr: Some(stderr),
        rank,
    }
}

/// 全子の終了を期限付きでポーリングする。期限超過は残りを kill して fail。
fn wait_all(
    children: &mut [ChildGuard],
    deadline: Duration,
) -> BTreeMap<usize, (ExitStatus, String)> {
    let start = Instant::now();
    let mut done: BTreeMap<usize, (ExitStatus, String)> = BTreeMap::new();
    while done.len() < children.len() {
        for c in children.iter_mut() {
            if done.contains_key(&c.rank) {
                continue;
            }
            if let Some(st) = c.child.try_wait().expect("try_wait") {
                let tail = c.stderr.take().map(|h| h.join().unwrap_or_default());
                done.insert(c.rank, (st, tail.unwrap_or_default()));
            }
        }
        if done.len() < children.len() && start.elapsed() > deadline {
            let pending: Vec<usize> = children
                .iter()
                .filter(|c| !done.contains_key(&c.rank))
                .map(|c| c.rank)
                .collect();
            for c in children.iter_mut() {
                let _ = c.child.kill();
            }
            panic!("期限 {deadline:?} 内に終了しない子ランク: {pending:?}（ハング）");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    done
}

fn out_path(dir: &Path, rank: usize) -> PathBuf {
    dir.join(format!("rank{rank}.out"))
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

fn bind_rank0(w: usize, session: u64) -> (Rendezvous, SocketAddr) {
    let cfg = config(w, 0, "127.0.0.1:0".parse().unwrap(), session);
    assert!(cfg.rendezvous_addr().ip().is_loopback() && !cfg.is_non_loopback_allowed());
    let rdv = Rendezvous::bind(&cfg).unwrap();
    let addr = rdv.local_addr().expect("rank 0 は実アドレスを持つ");
    assert!(addr.ip().is_loopback() && addr.port() != 0, "{addr}");
    (rdv, addr)
}

/// 1 回の正常 run。ランク 0（親）の optimizer 名 → 最終パラメータを返す。
/// 全ランクの bit 一致もここで確認する。
fn run_train(w: usize, session: u64) -> BTreeMap<String, Vec<f32>> {
    let dir = TempDirGuard::new("ddp-train");
    let (rdv, addr) = bind_rank0(w, session);
    let mut children: Vec<ChildGuard> = (1..w)
        .map(|r| spawn_child(w, r, addr, session, "train", &out_path(dir.path(), r)))
        .collect();
    let mut comm = rdv.establish().unwrap();
    assert_eq!((comm.rank(), comm.world_size()), (0, w));
    let (x, y, rows) = shard(0, w);
    let mut rank0 = BTreeMap::new();
    for (opt, name) in OPTS {
        rank0.insert(name.to_string(), train(opt, &x, &y, rows, Some(&mut comm)));
    }
    drop(comm);
    let exits = wait_all(&mut children, SCENARIO_DEADLINE);
    for (r, (st, tail)) in &exits {
        assert!(st.success(), "rank {r} が異常終了 {st:?}\n{tail}");
        let got = parse_train_result(&out_path(dir.path(), *r));
        for (_, name) in OPTS {
            assert_bits_eq(
                &rank0[name],
                &got[name],
                &format!("W={w} {name} rank0 vs rank{r}"),
            );
        }
    }
    rank0
}

#[test]
fn multiprocess_ddp_matches_single_process_for_world_size_2_and_4() {
    let (x, y) = full_data();
    for w in [2usize, 4] {
        let run_a = run_train(w, 0xD157_3148_0000 + w as u64);
        let run_b = run_train(w, 0xD157_3148_1000 + w as u64);
        for (opt, name) in OPTS {
            assert_bits_eq(&run_a[name], &run_b[name], &format!("W={w} {name} run 間"));
            let single = train(opt, &x, &y, FULL_ROWS, None);
            assert_eq!(run_a[name].len(), single.len());
            for (i, (d, s)) in run_a[name].iter().zip(&single).enumerate() {
                assert_close(*d, *s, &format!("W={w} {name} param element {i}"));
            }
        }
    }
}

#[test]
fn multiprocess_ddp_rank_crash_terminates_survivors_with_typed_error() {
    let w = 4;
    let session = 0xD157_3148_F000;
    let dir = TempDirGuard::new("ddp-crash");
    let (rdv, addr) = bind_rank0(w, session);
    let mut children: Vec<ChildGuard> = (1..w)
        .map(|r| {
            let mode = if r == CRASH_RANK { "crash" } else { "survive" };
            spawn_child(w, r, addr, session, mode, &out_path(dir.path(), r))
        })
        .collect();
    let mut comm = rdv.establish().unwrap();
    let started = Instant::now();
    let rep = fail_run(w, 0, &mut comm, false);
    assert!(
        ["PeerDisconnected", "Timeout", "Io"].contains(&rep.err_variant.as_str()),
        "{}",
        rep.err_variant
    );
    assert!(rep.poisoned, "rank 0 の communicator が poisoned でない");
    assert!(rep.fail_closed, "rank 0 のパラメータが失敗後に変化");
    assert!(
        rep.elapsed <= OP_TIMEOUT + Duration::from_secs(10),
        "{:?}",
        rep.elapsed
    );
    drop(comm);

    let exits = wait_all(&mut children, SCENARIO_DEADLINE);
    assert!(started.elapsed() < SCENARIO_DEADLINE);
    for (r, (st, tail)) in &exits {
        if *r == CRASH_RANK {
            assert_eq!(st.code(), Some(CRASH_EXIT_CODE), "rank {r}: {st:?}\n{tail}");
            continue;
        }
        assert!(st.success(), "rank {r} が異常終了 {st:?}\n{tail}");
        let text = std::fs::read_to_string(out_path(dir.path(), *r)).expect("生存ランクの結果");
        let kv: BTreeMap<&str, &str> = text.lines().filter_map(|l| l.split_once('=')).collect();
        assert!(
            ["PeerDisconnected", "Timeout", "Io"].contains(&kv["err"]),
            "rank {r}: {text}"
        );
        assert_eq!(kv["poisoned"], "true", "rank {r}");
        assert_eq!(kv["fail_closed"], "true", "rank {r}");
        let ms: u64 = kv["elapsed_ms"].parse().expect("elapsed_ms");
        assert!(
            Duration::from_millis(ms) <= OP_TIMEOUT + Duration::from_secs(10),
            "rank {r}: {ms} ms"
        );
    }
}
