//! Simulated hosts: independent runtimes driven by one executor.
//!
//! [Context::host] starts a [Process] that its applications see as a runtime of its own:
//!
//! - **Storage**: applications use plain partition names, which the host transparently places in
//!   a namespace of its own (`<host>-<partition>` in the runtime's storage). Crashes, wipes, full
//!   disks, and storage latency of the host apply to exactly that namespace.
//! - **Metrics**: the host registers metrics in a registry of its own, under names relative to
//!   the host, and its context's [crate::Metrics::encode] returns only them. A restarted host
//!   starts with fresh metrics, as a restarted program would.
//! - **Network**: the host has an IP of its own (assigned by name, so a restarted host keeps its
//!   address). Its dials originate from that IP, and binding or dialing a loopback or unspecified
//!   address refers to the host itself.
//! - **Clock**: the host's clock can be offset or drift from the runtime's (see [HostConfig]).
//!
//! Every host is still driven by the runtime's one task queue, so a simulation of many hosts
//! remains deterministic. [Hosts] manages a set of named hosts, starting, crashing, and
//! restarting them by name.

use super::{ClockOffset, Context, Process};
use crate::{Error, storage::validate_partition_name, telemetry::metrics::Registry};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    ops::Range,
    sync::Arc,
    time::Duration,
};

/// The identity a host gives the contexts started from it.
pub(super) struct HostScope {
    /// The context label of the host, which the names of its metrics are relative to.
    label: String,
    /// The prefix of the runtime's names for the host's partitions (one `<host>-` per enclosing
    /// host). Host names cannot contain `-`, so prefixes of different hosts never collide.
    pub(super) namespace: String,
    /// The host's metrics.
    pub(super) registry: Registry,
}

impl HostScope {
    /// `name` (a label of a context of the host) relative to the host's label.
    pub(super) fn relative<'a>(&self, name: &'a str) -> &'a str {
        name.strip_prefix(self.label.as_str())
            .map_or(name, |rest| rest.strip_prefix('_').unwrap_or(rest))
    }

    /// Strip the host's namespace from a partition named in `error`.
    fn error(&self, error: Error) -> Error {
        let local = |partition: String| {
            partition
                .strip_prefix(self.namespace.as_str())
                .map_or_else(|| partition.clone(), str::to_string)
        };
        match error {
            Error::PartitionNameInvalid(p) => Error::PartitionNameInvalid(local(p)),
            Error::PartitionCreationFailed(p) => Error::PartitionCreationFailed(local(p)),
            Error::PartitionMissing(p) => Error::PartitionMissing(local(p)),
            Error::PartitionCorrupt(p) => Error::PartitionCorrupt(local(p)),
            Error::BlobAlreadyOpen(p, n) => Error::BlobAlreadyOpen(local(p), n),
            Error::BlobOpenFailed(p, n, e) => Error::BlobOpenFailed(local(p), n, e),
            Error::BlobMissing(p, n) => Error::BlobMissing(local(p), n),
            Error::BlobResizeFailed(p, n, e) => Error::BlobResizeFailed(local(p), n, e),
            Error::BlobSyncFailed(p, n, e) => Error::BlobSyncFailed(local(p), n, e),
            Error::BlobCorrupt(p, n, r) => Error::BlobCorrupt(local(p), n, r),
            other => other,
        }
    }
}

/// How [Context::host] starts a host.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostConfig {
    ip: Option<IpAddr>,
    storage_latency: Option<Range<Duration>>,
    clock_offset: Option<ClockOffset>,
    clock_drift: i64,
}

impl HostConfig {
    /// A host with an assigned IP, the runtime's storage latency, and the runtime's clock.
    pub fn new() -> Self {
        Self::default()
    }

    /// Give the host `ip` instead of assigning it one.
    pub const fn with_ip(mut self, ip: IpAddr) -> Self {
        self.ip = Some(ip);
        self
    }

    /// Draw the latency of each operation on the host's storage from `range` (see
    /// [Process::set_storage_latency]).
    pub const fn with_storage_latency(mut self, range: Range<Duration>) -> Self {
        self.storage_latency = Some(range);
        self
    }

    /// Start the host's clock `offset` from the runtime's (see [Process::set_clock_offset]).
    pub const fn with_clock_offset(mut self, offset: ClockOffset) -> Self {
        self.clock_offset = Some(offset);
        self
    }

    /// Make the host's clock drift by `ppm` parts per million (see [Process::set_clock_drift]).
    pub const fn with_clock_drift(mut self, ppm: i64) -> Self {
        self.clock_drift = ppm;
        self
    }
}

/// The `n`th address assigned to hosts without a configured IP (from `10.0.0.1`, skipping
/// network and broadcast-looking addresses).
fn assigned_ip(n: u32) -> IpAddr {
    let mut index = 0u32;
    let mut candidate = 0u32;
    loop {
        candidate = candidate.checked_add(1).expect("host addresses exhausted");
        let last = candidate & 0xff;
        if last == 0 || last == 0xff {
            continue;
        }
        if index == n {
            break;
        }
        index += 1;
    }
    assert!(candidate < 1 << 24, "host addresses exhausted");
    IpAddr::V4(Ipv4Addr::from(0x0a00_0000 | candidate))
}

impl Context {
    /// Start a simulated host named `name`: a [Process] that its applications see as a runtime of
    /// its own, with its own storage namespace, metrics, IP, and clock.
    ///
    /// The process owns exactly the host's storage, so [Process::crash], [Process::wipe],
    /// [Process::set_disk_full], and [Process::set_storage_latency] apply to all of it and nothing
    /// else, and a crash or latency injected by [crate::deterministic::FaultConfig] lands on it
    /// when its storage is operated on. Restart a crashed host by starting a host with the same
    /// name from the same context: it opens the same storage and gets the same IP (unless `cfg`
    /// sets another).
    ///
    /// A host started from a host's context is nested in it: its storage namespace is within its
    /// parent's, and crashing the parent crashes it too.
    ///
    /// # Panics
    ///
    /// Panics if `name` is not a valid label (it must start with `[a-zA-Z]` and contain only
    /// `[a-zA-Z0-9_]`), or if `cfg` sets an IP another host already has.
    pub fn host(&self, name: &str, cfg: HostConfig) -> (Self, Process) {
        let context = self.child_named(name);
        let namespace = format!(
            "{}{name}-",
            self.host_scope()
                .map_or("", |host| host.namespace.as_str())
        );
        let scope = Arc::new(HostScope {
            label: context.name.clone(),
            namespace: namespace.clone(),
            registry: Registry::new(),
        });

        // Hosts keep their address across restarts
        let ip = {
            let executor = self.executor();
            let mut ips = executor.host_ips.lock();
            let ip = cfg
                .ip
                .or_else(|| ips.get(&namespace).copied())
                .unwrap_or_else(|| {
                    (0..)
                        .map(assigned_ip)
                        .find(|ip| !ips.values().any(|taken| taken == ip))
                        .expect("host addresses exhausted")
                });
            assert!(
                ips.iter()
                    .all(|(other, taken)| *taken != ip || *other == namespace),
                "host IP {ip} already taken"
            );
            ips.insert(namespace.clone(), ip);
            executor.auditor.event(b"host", |hasher| {
                hasher.update(namespace.as_bytes());
                hasher.update(ip.to_string());
            });
            ip
        };

        let (context, process) = self.start_process(
            context,
            Arc::new(move |partition: &str| partition.starts_with(namespace.as_str())),
            Some(ip),
            Some(scope),
        );
        if let Some(offset) = cfg.clock_offset {
            process.set_clock_offset(offset);
        }
        if cfg.clock_drift != 0 {
            process.set_clock_drift(cfg.clock_drift);
        }
        if let Some(range) = cfg.storage_latency {
            process.set_storage_latency(Some(range));
        }
        (context, process)
    }

    /// The runtime's name for the partition the context's applications name `partition`.
    pub(super) fn host_partition<'a>(&self, partition: &'a str) -> Result<Cow<'a, str>, Error> {
        let Some(host) = self.host_scope() else {
            return Ok(Cow::Borrowed(partition));
        };
        validate_partition_name(partition)?;
        Ok(Cow::Owned(format!("{}{partition}", host.namespace)))
    }

    /// `error` with partitions named as the context's applications name them.
    pub(super) fn host_error(&self, error: Error) -> Error {
        match self.host_scope() {
            Some(host) => host.error(error),
            None => error,
        }
    }

    /// The address the context reaches by binding or dialing `socket`: within a host, a loopback
    /// or unspecified address is the host's own.
    pub(super) fn host_socket(&self, socket: SocketAddr) -> SocketAddr {
        if self.host_scope().is_none() || !(socket.ip().is_loopback() || socket.ip().is_unspecified()) {
            return socket;
        }
        self.source_ip()
            .map_or(socket, |ip| SocketAddr::new(ip, socket.port()))
    }
}

impl Process {
    /// The IP the process dials from, if it has one (every host does; see [Self::set_ip]).
    pub fn ip(&self) -> Option<IpAddr> {
        *self.host.state.ip.lock()
    }

    /// Draw the latency of each operation on the process's partitions from `range` (or, with
    /// `None`, from [crate::deterministic::FaultConfig::latency] again), as a slower or faster
    /// disk would.
    ///
    /// While an operation waits out its latency, other tasks and simulated time keep moving, so a
    /// crash or pause can land in the middle of it. An empty or inverted range is a fixed latency
    /// of its start. A process restarted after a crash starts with the runtime's latency.
    pub fn set_storage_latency(&self, range: Option<Range<Duration>>) {
        self.executor().auditor.event(b"storage_latency", |hasher| {
            if let Some(range) = &range {
                hasher.update(range.start.as_nanos().to_be_bytes());
                hasher.update(range.end.as_nanos().to_be_bytes());
            }
        });
        *self.host.latency.lock() = range;
    }

    /// List the durable blobs of the process's partitions, as
    /// [Context::durable_blobs] does. The partitions of a host are named as its applications
    /// name them.
    pub fn durable_blobs(&self) -> Vec<(String, Vec<u8>, u64)> {
        let blobs = self
            .host
            .storage
            .inner()
            .inner()
            .inner()
            .durable_blobs(&|partition: &str| (self.host.partitions)(partition));
        let Some(scope) = &self.host.state.scope else {
            return blobs;
        };
        blobs
            .into_iter()
            .map(|(partition, name, len)| {
                let local = partition
                    .strip_prefix(scope.namespace.as_str())
                    .map_or_else(|| partition.clone(), str::to_string);
                (local, name, len)
            })
            .collect()
    }
}

/// A set of named hosts started from one context, as a cluster manager would run them.
///
/// Each host remembers the [HostConfig] it was first started with, so [Self::restart] brings it
/// back on the same storage, IP, clock, and storage latency. Hosts are kept in name order, so
/// operations over all of them are deterministic.
pub struct Hosts {
    context: Context,
    configs: BTreeMap<String, HostConfig>,
    running: BTreeMap<String, Process>,
    starts: BTreeMap<String, u64>,
}

impl Hosts {
    /// A manager of hosts started from `context` (none yet).
    pub fn new(context: &Context) -> Self {
        Self {
            context: crate::Supervisor::child(context, "hosts"),
            configs: BTreeMap::new(),
            running: BTreeMap::new(),
            starts: BTreeMap::new(),
        }
    }

    /// Start host `name` with `cfg`, returning its context, from which its tasks are spawned.
    ///
    /// A host started before (and since crashed) restarts with `cfg` instead of its previous
    /// configuration.
    ///
    /// # Panics
    ///
    /// Panics if `name` is running.
    pub fn start(&mut self, name: &str, cfg: HostConfig) -> Context {
        assert!(!self.is_running(name), "host {name} already running");
        let (context, process) = self.context.host(name, cfg.clone());
        self.configs.insert(name.to_string(), cfg);
        self.running.insert(name.to_string(), process);
        *self.starts.entry(name.to_string()).or_default() += 1;
        context
    }

    /// Crash host `name` if it is running, then start it again with its configuration.
    ///
    /// # Panics
    ///
    /// Panics if `name` was never started.
    pub fn restart(&mut self, name: &str) -> Context {
        let cfg = self
            .configs
            .get(name)
            .unwrap_or_else(|| panic!("host {name} never started"))
            .clone();
        self.crash(name);
        self.start(name, cfg)
    }

    /// Crash host `name` (see [Process::crash]). Does nothing if it is not running.
    pub fn crash(&mut self, name: &str) {
        if let Some(process) = self.running.remove(name) {
            process.crash();
        }
    }

    /// Crash host `name` if it is running and erase its storage (see [Process::wipe]).
    ///
    /// # Panics
    ///
    /// Panics if `name` was never started.
    pub fn wipe(&mut self, name: &str) {
        match self.running.remove(name) {
            Some(process) => process.wipe(),
            None => {
                assert!(self.configs.contains_key(name), "host {name} never started");
                let namespace = format!(
                    "{}{name}-",
                    self.context
                        .host_scope()
                        .map_or("", |host| host.namespace.as_str())
                );
                let erased = self
                    .context
                    .storage
                    .inner()
                    .inner()
                    .inner()
                    .erase_partitions(&|partition: &str| partition.starts_with(namespace.as_str()));
                self.context
                    .executor()
                    .auditor
                    .event(b"wipe_host", |hasher| {
                        for partition in &erased {
                            hasher.update(partition.as_bytes());
                        }
                    });
            }
        }
    }

    /// Crash every running host at once, as a power loss across the cluster would. No task of
    /// any host runs between the first crash and the last.
    pub fn crash_all(&mut self) {
        for (_, process) in std::mem::take(&mut self.running) {
            process.crash();
        }
    }

    /// Whether host `name` is running (it was started and has not crashed since, whether through
    /// this manager or from within its own storage operation).
    pub fn is_running(&self, name: &str) -> bool {
        self.running
            .get(name)
            .is_some_and(|process| !process.crashed())
    }

    /// The running host `name`, to pause, skew, or otherwise fault directly.
    pub fn get(&self, name: &str) -> Option<&Process> {
        self.running.get(name).filter(|process| !process.crashed())
    }

    /// The names of the running hosts, in name order.
    pub fn running(&self) -> Vec<String> {
        self.running
            .iter()
            .filter(|(_, process)| !process.crashed())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// How many times host `name` has been started.
    pub fn starts(&self, name: &str) -> u64 {
        self.starts.get(name).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Blob as _, Clock as _, Listener as _, Metrics as _, Network as _, ReadOptions, Runner as _,
        Sink as _, Spawner as _, Storage as _, Stream as _, Supervisor as _, WriteOptions,
        deterministic::{self, FaultConfig},
        telemetry::metrics::raw,
    };
    use commonware_utils::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Write `durable` (synced) then `pending` (unsynced) to `partition`/`blob` from `context`,
    /// returning the open handle (a reopen sees only durable contents).
    async fn write(
        context: &Context,
        partition: &str,
        durable: &[u8],
        pending: &[u8],
    ) -> deterministic::Blob {
        let (blob, _) = context.open(partition, b"blob").await.unwrap();
        blob.write_at(0, durable.to_vec(), WriteOptions::default())
            .await
            .unwrap();
        blob.sync().await.unwrap();
        blob.write_at(0, pending.to_vec(), WriteOptions::default())
            .await
            .unwrap();
        blob
    }

    async fn read(context: &Context, partition: &str) -> Vec<u8> {
        let (blob, len) = context.open(partition, b"blob").await.unwrap();
        let len = usize::try_from(len).unwrap();
        blob.read_at(0, len, ReadOptions::default())
            .await
            .unwrap()
            .coalesce()
            .as_ref()
            .to_vec()
    }

    #[test]
    fn test_hosts_isolate_storage_under_plain_names() {
        deterministic::Runner::default().start(|context| async move {
            let (a, host_a) = context.host("a", HostConfig::new());
            let (b, _host_b) = context.host("b", HostConfig::new());
            drop(write(&a, "journal", b"a-durable", b"a-pending").await);
            let b_blob = write(&b, "journal", b"b-durable", b"b-pending").await;

            // Each host sees only its own partition
            assert_eq!(a.scan("journal").await.unwrap(), vec![b"blob".to_vec()]);
            assert_eq!(context.logical_blob("journal", b"blob"), None);
            assert_eq!(
                context.logical_blob("a-journal", b"blob").as_deref(),
                Some(&b"a-durable"[..])
            );
            assert_eq!(
                host_a.durable_blobs(),
                vec![("journal".to_string(), b"blob".to_vec(), 9)]
            );

            // Errors name partitions as the host's applications do
            assert!(matches!(
                a.scan("missing").await,
                Err(Error::PartitionMissing(partition)) if partition == "missing"
            ));
            assert!(matches!(
                a.open("", b"blob").await,
                Err(Error::PartitionNameInvalid(_))
            ));

            // Crashing one host loses only its unsynced writes
            let a_blob = write(&a, "journal", b"a-durable", b"a-pending").await;
            host_a.crash();
            drop(a_blob);
            let (a, _host_a) = context.host("a", HostConfig::new());
            assert_eq!(read(&a, "journal").await, b"a-durable");
            b_blob.sync().await.unwrap();
            drop(b_blob);
            assert_eq!(read(&b, "journal").await, b"b-pending");
        });
    }

    #[test]
    fn test_nested_process_selects_plain_partition_names() {
        deterministic::Runner::default().start(|context| async move {
            let (host, _process) = context.host("node", HostConfig::new());
            let (component, process) = host.process("marshal", |partition| partition == "blocks");
            let blocks = write(&component, "blocks", b"durable", b"pending").await;
            let votes = write(&component, "votes", b"durable", b"pending").await;
            process.crash();
            drop(blocks);
            assert_eq!(read(&host, "blocks").await, b"durable");
            // Unowned partitions keep their live handles and unsynced writes
            votes.sync().await.unwrap();
            drop(votes);
            assert_eq!(read(&host, "votes").await, b"pending");
        });
    }

    #[test]
    fn test_hosts_have_own_metrics() {
        deterministic::Runner::default().start(|context| async move {
            let (a, host_a) = context.host("a", HostConfig::new());
            let (b, _host_b) = context.host("b", HostConfig::new());
            let a_counter = a.child("engine").register("count", "help", raw::Counter::<u64>::default());
            let b_counter = b.child("engine").register("count", "help", raw::Counter::<u64>::default());
            a_counter.inc();
            assert!(a.encode().contains("engine_count_total 1"));
            assert!(b.encode().contains("engine_count_total 0"));
            assert!(!context.encode().contains("engine_count"));

            // A restarted host starts with fresh metrics
            host_a.crash();
            let (a, _host_a) = context.host("a", HostConfig::new());
            let counter = a.child("engine").register("count", "help", raw::Counter::<u64>::default());
            assert!(a.encode().contains("engine_count_total 0"));
            drop((counter, a_counter, b_counter));
        });
    }

    #[test]
    fn test_hosts_have_own_addresses() {
        deterministic::Runner::default().start(|context| async move {
            let (a, host_a) = context.host("a", HostConfig::new());
            let (b, host_b) = context.host("b", HostConfig::new());
            let a_ip = host_a.ip().unwrap();
            let b_ip = host_b.ip().unwrap();
            assert_ne!(a_ip, b_ip);

            // Both hosts bind the same unspecified address
            let mut a_listener = a.bind("0.0.0.0:3000".parse().unwrap()).await.unwrap();
            let mut b_listener = b.bind("0.0.0.0:3000".parse().unwrap()).await.unwrap();
            assert_eq!(a_listener.local_addr().unwrap().ip(), a_ip);

            // A dial to another host's address arrives from the dialer's address
            let (mut sink, _stream) = a.dial(SocketAddr::new(b_ip, 3000)).await.unwrap();
            let (peer, _, mut stream) = b_listener.accept().await.unwrap();
            assert_eq!(peer.ip(), a_ip);
            sink.send(&b"hello"[..]).await.unwrap();
            assert_eq!(stream.recv(5).await.unwrap().coalesce(), b"hello");

            // A loopback dial reaches the dialer's own host
            let _ = b.dial("127.0.0.1:3000".parse().unwrap()).await.unwrap();
            let (peer, _, _) = b_listener.accept().await.unwrap();
            assert_eq!(peer.ip(), b_ip);
            drop(a_listener.accept());

            // A restarted host keeps its address; a configured one is honored
            host_a.crash();
            let (_, host_a) = context.host("a", HostConfig::new());
            assert_eq!(host_a.ip(), Some(a_ip));
            let ip: IpAddr = "192.168.1.1".parse().unwrap();
            let (_, host_c) = context.host("c", HostConfig::new().with_ip(ip));
            assert_eq!(host_c.ip(), Some(ip));
        });
    }

    #[test]
    #[should_panic(expected = "already taken")]
    fn test_host_ip_must_be_unique() {
        deterministic::Runner::default().start(|context| async move {
            let ip: IpAddr = "192.168.1.1".parse().unwrap();
            let _a = context.host("a", HostConfig::new().with_ip(ip));
            let _b = context.host("b", HostConfig::new().with_ip(ip));
        });
    }

    #[test]
    fn test_assigned_ips_skip_reserved_octets() {
        assert_eq!(assigned_ip(0), "10.0.0.1".parse::<IpAddr>().unwrap());
        assert_eq!(assigned_ip(253), "10.0.0.254".parse::<IpAddr>().unwrap());
        assert_eq!(assigned_ip(254), "10.0.1.1".parse::<IpAddr>().unwrap());
    }

    #[test]
    fn test_host_storage_latency() {
        let cfg = deterministic::Config::default().with_storage_fault_config(
            FaultConfig::default().latency(Duration::from_millis(1)..Duration::from_millis(1)),
        );
        deterministic::Runner::new(cfg).start(|context| async move {
            let slow = HostConfig::new()
                .with_storage_latency(Duration::from_millis(50)..Duration::from_millis(50));
            let (a, _host_a) = context.host("slow", slow);
            let (b, _host_b) = context.host("fast", HostConfig::new());
            for (host, expected) in [(a, 50), (b, 1)] {
                let (blob, _) = host.open("journal", b"blob").await.unwrap();
                let start = host.current();
                blob.sync().await.unwrap();
                assert_eq!(
                    host.current().duration_since(start).unwrap(),
                    Duration::from_millis(expected)
                );
            }
        });
    }

    #[test]
    fn test_host_clock() {
        deterministic::Runner::default().start(|context| async move {
            let cfg = HostConfig::new()
                .with_clock_offset(ClockOffset::Ahead(Duration::from_secs(5)))
                .with_clock_drift(100_000);
            let (host, _process) = context.host("skewed", cfg);
            assert_eq!(
                host.current().duration_since(context.current()).unwrap(),
                Duration::from_secs(5)
            );
            context.sleep(Duration::from_secs(10)).await;
            assert_eq!(
                host.current().duration_since(context.current()).unwrap(),
                Duration::from_secs(6)
            );
        });
    }

    #[test]
    fn test_hosts_manager_restarts_and_power_loss() {
        deterministic::Runner::default().start(|context| async move {
            let mut hosts = Hosts::new(&context);
            let ticks = Arc::new(Mutex::new(BTreeMap::<String, usize>::new()));
            for name in ["n0", "n1", "n2"] {
                let host = hosts.start(name, HostConfig::new());
                drop(write(&host, "state", b"durable", b"pending").await);
                let ticks = ticks.clone();
                host.spawn(move |context| async move {
                    loop {
                        context.sleep(Duration::from_millis(1)).await;
                        *ticks.lock().entry(name.to_string()).or_default() += 1;
                    }
                });
            }
            let ip = hosts.get("n1").unwrap().ip();
            let host = hosts.restart("n1");
            assert_eq!(hosts.starts("n1"), 2);
            assert_eq!(hosts.get("n1").unwrap().ip(), ip);
            assert_eq!(read(&host, "state").await, b"durable");

            // Whole-cluster power loss: nothing runs afterwards
            hosts.crash_all();
            assert!(hosts.running().is_empty());
            let before = ticks.lock().clone();
            context.sleep(Duration::from_millis(10)).await;
            assert_eq!(*ticks.lock(), before);

            // A wiped host restarts empty; the others keep their synced state
            hosts.wipe("n2");
            let host = hosts.restart("n2");
            assert!(matches!(host.scan("state").await, Err(Error::PartitionMissing(_))));
            let host = hosts.restart("n0");
            assert_eq!(read(&host, "state").await, b"durable");
        });
    }

    #[test]
    fn test_crash_in_storage_stops_host() {
        let cfg = deterministic::Config::default()
            .with_storage_fault_config(FaultConfig::default().crash(commonware_utils::probability!(1.0)));
        deterministic::Runner::new(cfg).start(|context| async move {
            let mut hosts = Hosts::new(&context);
            let host = hosts.start("n0", HostConfig::new());
            let progress = Arc::new(AtomicUsize::new(0));
            let counter = progress.clone();
            let handle = host.spawn(move |context| async move {
                let (blob, _) = context.open("state", b"blob").await.unwrap();
                counter.fetch_add(1, Ordering::SeqCst);
                blob.write_at(0, b"x".to_vec(), WriteOptions::default())
                    .await
                    .unwrap();
                counter.fetch_add(1, Ordering::SeqCst);
            });
            assert!(handle.await.is_err());
            assert_eq!(progress.load(Ordering::SeqCst), 1);
            assert!(!hosts.is_running("n0"));
            let _host = hosts.restart("n0");
            assert!(hosts.is_running("n0"));
        });
    }

    fn run_cluster(seed: u64) -> String {
        let cfg = deterministic::Config::default()
            .with_seed(seed)
            .with_storage_fault_config(
                FaultConfig::default().latency(Duration::ZERO..Duration::from_millis(5)),
            );
        deterministic::Runner::new(cfg).start(|context| async move {
            let mut hosts = Hosts::new(&context);
            for round in 0..3u8 {
                for name in ["n0", "n1", "n2"] {
                    let host = if round == 0 {
                        hosts.start(name, HostConfig::new())
                    } else {
                        hosts.restart(name)
                    };
                    host.spawn(move |context| async move {
                        let (blob, len) = context.open("log", b"blob").await.unwrap();
                        for i in 0..10u64 {
                            blob.write_at(len + i, vec![round], WriteOptions::default())
                                .await
                                .unwrap();
                            if i % 3 == 0 {
                                blob.sync().await.unwrap();
                            }
                        }
                    });
                }
                context.sleep(Duration::from_millis(20)).await;
            }
            context.auditor().state()
        })
    }

    #[test]
    fn test_hosts_are_deterministic() {
        assert_eq!(run_cluster(1), run_cluster(1));
        assert_ne!(run_cluster(1), run_cluster(2));
    }
}
