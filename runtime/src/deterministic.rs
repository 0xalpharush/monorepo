//! A deterministic runtime that randomly selects tasks to run based on a seed
//!
//! # Panics
//!
//! Unless configured otherwise, any task panic will lead to a runtime panic.
//!
//! # External Processes
//!
//! When testing an application that interacts with some external process, it can appear to
//! the runtime that progress has stalled because no pending tasks can make progress and/or
//! that futures resolve at variable latency (which in turn triggers non-deterministic execution).
//!
//! To support such applications, the runtime can be built with the `external` feature to both
//! sleep for each [Config::cycle] (opting to wait if all futures are pending) and to constrain
//! the resolution latency of any future (with `pace()`).
//!
//! **Applications that do not interact with external processes (or are able to mock them) should never
//! need to enable this feature. It is commonly used when testing consensus with external execution environments
//! that use their own runtime (but are deterministic over some set of inputs).**
//!
//! # Metrics
//!
//! This runtime enforces metrics are unique and well-formed:
//! - Labels must start with `[a-zA-Z]` and contain only `[a-zA-Z0-9_]`
//! - Re-registering the same metric key reuses the existing metric handle when the type matches
//!
//! # Example
//!
//! ```rust
//! use commonware_runtime::{Spawner, Runner, deterministic, Metrics, Supervisor};
//!
//! let executor =  deterministic::Runner::default();
//! executor.start(|context| async move {
//!     println!("Parent started");
//!     let result = context.child("child").spawn(|_| async move {
//!         println!("Child started");
//!         "hello"
//!     });
//!     println!("Child result: {:?}", result.await);
//!     println!("Parent exited");
//!     println!("Auditor state: {}", context.auditor().state());
//! });
//! ```

use crate::{
    BlobVersion, BufferPool, BufferPoolConfig, Clock, Error, Execution, Handle, IoBufs, ListenerOf,
    METRICS_PREFIX, Name, Panicked, child_label,
    network::{
        audited::Network as AuditedNetwork,
        deterministic::{Network as DeterministicNetwork, Timer, TimerSlot},
        metered::Network as MeteredNetwork,
    },
    prefixed_name,
    storage::{
        audited::{Blob as AuditedBlob, Storage as AuditedStorage},
        faulty::{Blob as FaultyBlob, Crasher, Storage as FaultyStorage},
        memory::{
            Blob as MemBlob, Snapshot as MemStorageSnapshot, Storage as MemStorage,
            open::{Blob as OpenBlob, Opens},
        },
        metered::{Blob as MeteredBlob, Storage as MeteredStorage},
    },
    telemetry::metrics::{
        Counter, CounterFamily, GaugeFamily, Metric, Register, Registered, Registry, add_attribute,
        raw, task::Label, validate_label,
    },
    utils::{
        FactoryGuard, Panicker,
        signal::{Signal, Stopper},
        supervision::Tree,
    },
};
#[cfg(feature = "external")]
use crate::{Blocker, Pacer};
pub use crate::{
    network::deterministic::{
        Delivery as NetworkDelivery, Fate as NetworkFate, Link as NetworkLink,
        Policy as NetworkPolicy, Transmission as NetworkTransmission,
    },
    storage::faulty::{
        Config as FaultConfig, Decision as FaultDecision, Draw as FaultDraw, Op as StorageOp,
        PartialWriteMode, Policy as FaultPolicy, ResizeConfig, SharedRng, WriteConfig,
    },
};
use commonware_codec::Encode;
use commonware_formatting::hex;
use commonware_macros::select;
use commonware_parallel::{Rayon, ThreadPool};
use commonware_utils::{
    Cached, SystemTimeExt,
    sync::{Mutex, RwLock},
    time::SYSTEM_TIME_PRECISION,
};
#[cfg(feature = "external")]
use futures::task::noop_waker;
use futures::{
    Future,
    task::{ArcWake, AtomicWaker, waker},
};
use governor::clock::{Clock as GClock, ReasonablyRealtime};
#[cfg(feature = "external")]
use pin_project::pin_project;
use rand::{CryptoRng, Rng, SeedableRng, TryCryptoRng, TryRng, prelude::SliceRandom, rngs::StdRng};
use rayon::{ThreadPoolBuildError, ThreadPoolBuilder};
use sha2::{Digest as _, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap},
    convert::Infallible,
    mem::{replace, take},
    net::{IpAddr, SocketAddr},
    num::NonZeroUsize,
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    pin::Pin,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering as AtomicOrdering},
    },
    task::{self, Poll},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tracing::trace;

#[derive(Debug)]
struct Metrics {
    iterations: Counter,
    tasks_spawned: CounterFamily<Label>,
    tasks_running: GaugeFamily<Label>,
    task_polls: CounterFamily<Label>,
}

impl Metrics {
    pub fn init(registry: &mut impl Register) -> Self {
        Self {
            iterations: registry.register(
                "iterations",
                "Total number of iterations",
                raw::Counter::default(),
            ),
            tasks_spawned: registry.register(
                "tasks_spawned",
                "Total number of tasks spawned",
                raw::Family::default(),
            ),
            tasks_running: registry.register(
                "tasks_running",
                "Number of tasks currently running",
                raw::Family::default(),
            ),
            task_polls: registry.register(
                "task_polls",
                "Total number of task polls",
                raw::Family::default(),
            ),
        }
    }
}

/// A SHA-256 digest.
type Digest = [u8; 32];

/// Hashes an unambiguous sequence of fields for deterministic runtime auditing.
pub(crate) struct AuditHasher(Sha256);

impl AuditHasher {
    /// Creates an empty audit hasher.
    pub(crate) fn new() -> Self {
        Self(Sha256::new())
    }

    /// Adds a length-prefixed field to the audit.
    pub(crate) fn update(&mut self, value: impl AsRef<[u8]>) {
        let value = value.as_ref();
        self.0.update((value.len() as u64).to_be_bytes());
        self.0.update(value);
    }

    /// Adds the logical contents of `bufs` as one length-prefixed field.
    ///
    /// Physical chunk boundaries are excluded because they are not part of the storage or network
    /// operation being audited.
    pub(crate) fn update_bufs(&mut self, bufs: &IoBufs) {
        self.0.update((bufs.len() as u64).to_be_bytes());
        bufs.for_each_chunk(|chunk| self.0.update(chunk));
    }

    /// Returns the digest of all fields added to the audit.
    pub(crate) fn finalize(self) -> Digest {
        self.0.finalize().into()
    }
}

/// Track the state of the runtime for determinism auditing.
pub struct Auditor {
    digest: Mutex<Digest>,
}

impl Default for Auditor {
    fn default() -> Self {
        Self {
            digest: Digest::default().into(),
        }
    }
}

impl Auditor {
    /// Record that an event happened.
    /// This auditor's hash will be updated with the event's `label` and
    /// whatever other data is passed in the `payload` closure.
    pub(crate) fn event<F>(&self, label: &'static [u8], payload: F)
    where
        F: FnOnce(&mut AuditHasher),
    {
        let mut digest = self.digest.lock();

        let mut hasher = AuditHasher::new();
        hasher.update(digest.as_ref());
        hasher.update(label);
        payload(&mut hasher);

        *digest = hasher.finalize();
    }

    /// Generate a representation of the current state of the runtime.
    ///
    /// This can be used to ensure that logic running on top
    /// of the runtime is interacting deterministically.
    pub fn state(&self) -> String {
        let hash = self.digest.lock();
        hex(hash.as_ref())
    }
}

/// Record a blob's identity unambiguously in an audit event.
fn audit_blob(hasher: &mut AuditHasher, partition: &str, name: &[u8]) {
    hasher.update((partition.len() as u64).to_be_bytes());
    hasher.update(partition.as_bytes());
    hasher.update((name.len() as u64).to_be_bytes());
    hasher.update(name);
}

/// A blob's durable contents captured by [Context::snapshot_blob], restored with
/// [Context::restore_blob] to simulate lost writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobSnapshot {
    partition: String,
    name: Vec<u8>,
    raw: Vec<u8>,
}

impl BlobSnapshot {
    /// The partition of the captured blob.
    pub fn partition(&self) -> &str {
        &self.partition
    }

    /// The name of the captured blob.
    pub fn name(&self) -> &[u8] {
        &self.name
    }
}

mod scheduling;
pub use scheduling::{DelayBounded, Pausing, Pct, RandomWalk};

/// Tasks held back by a [SchedulingPolicy].
#[derive(Default)]
struct Held {
    /// When each held task is released.
    until: BTreeMap<u128, SystemTime>,
    /// Held tasks in release order.
    releases: BTreeSet<(SystemTime, u128)>,
    /// Released tasks to poll once without consulting the policy.
    released: BTreeSet<u128>,
}

/// A dynamic RNG that can safely be sent between threads.
pub type BoxDynRng = Box<dyn CryptoRng + Send + 'static>;

/// One occurrence of a runnable task in a snapshotted polling batch.
pub struct RunnableTask {
    id: u128,
    label: String,
    occurrence: u64,
}

impl RunnableTask {
    /// Stable spawn-order identity within this runtime incarnation.
    pub const fn id(&self) -> u128 {
        self.id
    }

    /// Supervisor-derived task label, including runtime attributes.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Distinguishes repeated wakeups for the same task in this batch.
    pub const fn occurrence(&self) -> u64 {
        self.occurrence
    }
}

/// Controls cooperative polling order and task pauses, not native-thread preemption.
///
/// Each iteration of the event loop polls a batch of runnable tasks. A policy permutes the batch
/// ([Self::order]) and may hold individual tasks back for some virtual time ([Self::hold]), as
/// if their threads were descheduled. A policy cannot add, remove, or modify tasks.
pub trait SchedulingPolicy: Send + 'static {
    /// Reorders the current runnable batch at this virtual time.
    fn order(&mut self, time: SystemTime, ready: &mut [RunnableTask]);

    /// How long to hold `task` back instead of polling it in the current (ordered) batch.
    ///
    /// A held task is not polled until the hold elapses, even if it is woken in the meantime;
    /// it is then polled once without consulting this method. Other tasks and time keep
    /// moving. Returning [Duration::ZERO] (the default) polls the task now.
    fn hold(&mut self, time: SystemTime, task: &RunnableTask) -> Duration {
        let _ = (time, task);
        Duration::ZERO
    }
}

type BoxDynSchedulingPolicy = Box<dyn SchedulingPolicy>;

/// Configuration for the `deterministic` runtime.
pub struct Config {
    /// Random number generator.
    rng: BoxDynRng,

    scheduling_policy: Option<BoxDynSchedulingPolicy>,

    /// The cycle duration determines how much time is advanced after each iteration of the event
    /// loop. This is useful to prevent starvation if some task never yields.
    cycle: Duration,

    /// Time the runtime starts at.
    start_time: SystemTime,

    /// If the runtime is still executing at this point (i.e. a test hasn't stopped), panic.
    timeout: Option<Duration>,

    /// Whether spawned tasks should catch panics instead of propagating them.
    catch_panics: bool,

    /// Configuration for deterministic storage fault injection.
    /// Defaults to no faults being injected.
    storage_fault_cfg: FaultConfig,

    /// Decides each storage fault the configuration enables. Defaults to the shared RNG.
    storage_fault_policy: Option<Arc<dyn FaultPolicy>>,

    /// Decides connection faults and latency. Defaults to lossless, zero-latency connections.
    network_policy: Option<Arc<dyn NetworkPolicy>>,

    /// Buffer pool configuration for network I/O.
    network_buffer_pool_cfg: BufferPoolConfig,

    /// Buffer pool configuration for storage I/O.
    storage_buffer_pool_cfg: BufferPoolConfig,
}

impl Config {
    /// Returns a new [Config] with default values.
    pub fn new() -> Self {
        cfg_if::cfg_if! {
            if #[cfg(miri)] {
                // Reduce max_per_class to avoid slow atomics under Miri
                let network_buffer_pool_cfg = BufferPoolConfig::for_network()
                    .with_max_per_class(commonware_utils::NZU32!(32))
                    .with_thread_cache_disabled();
                let storage_buffer_pool_cfg = BufferPoolConfig::for_storage()
                    .with_max_per_class(commonware_utils::NZU32!(32))
                    .with_thread_cache_disabled();
            } else {
                let network_buffer_pool_cfg =
                    BufferPoolConfig::for_network().with_thread_cache_disabled();
                let storage_buffer_pool_cfg =
                    BufferPoolConfig::for_storage().with_thread_cache_disabled();
            }
        }

        Self {
            rng: Box::new(StdRng::seed_from_u64(42)),
            scheduling_policy: None,
            cycle: Duration::from_millis(1),
            start_time: UNIX_EPOCH,
            timeout: None,
            catch_panics: false,
            storage_fault_cfg: FaultConfig::default(),
            storage_fault_policy: None,
            network_policy: None,
            network_buffer_pool_cfg,
            storage_buffer_pool_cfg,
        }
    }

    // Setters
    /// See [Config]
    pub fn with_seed(self, seed: u64) -> Self {
        let rng: BoxDynRng = Box::new(StdRng::seed_from_u64(seed));
        self.with_rng(rng)
    }

    /// Provide the config with a dynamic RNG directly.
    ///
    /// This can be useful for, e.g. fuzzing, where beyond just having randomness,
    /// you might want to control specific bytes of the RNG. By taking in a dynamic
    /// RNG object, any behavior is possible.
    pub fn with_rng(mut self, rng: impl Into<BoxDynRng>) -> Self {
        self.rng = rng.into();
        self
    }

    /// Replaces seeded batch shuffling with a controlled cooperative policy.
    pub fn with_scheduling_policy(mut self, policy: impl SchedulingPolicy) -> Self {
        self.scheduling_policy = Some(Box::new(policy));
        self
    }

    /// See [Config]
    pub const fn with_cycle(mut self, cycle: Duration) -> Self {
        self.cycle = cycle;
        self
    }
    /// See [Config]
    pub const fn with_start_time(mut self, start_time: SystemTime) -> Self {
        self.start_time = start_time;
        self
    }
    /// See [Config]
    pub const fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }
    /// See [Config]
    pub const fn with_catch_panics(mut self, catch_panics: bool) -> Self {
        self.catch_panics = catch_panics;
        self
    }
    /// See [Config]
    pub fn with_network_buffer_pool_config(mut self, cfg: BufferPoolConfig) -> Self {
        self.network_buffer_pool_cfg = cfg;
        self
    }
    /// See [Config]
    pub fn with_storage_buffer_pool_config(mut self, cfg: BufferPoolConfig) -> Self {
        self.storage_buffer_pool_cfg = cfg;
        self
    }

    /// Configure storage fault injection.
    ///
    /// When set, the runtime will inject deterministic storage errors based on
    /// the provided configuration. Faults are drawn from the shared RNG, ensuring
    /// reproducible failure patterns for a given seed.
    pub const fn with_storage_fault_config(mut self, faults: FaultConfig) -> Self {
        self.storage_fault_cfg = faults;
        self
    }

    /// Decide each storage fault with `policy` instead of the shared RNG.
    ///
    /// The fault configuration still selects which faults can occur; the policy sees every
    /// individual decision with the file and operation it concerns, so it can derive, record,
    /// replay, or override each one independently of all other randomness in the runtime. The
    /// policy survives [Runner::start_and_recover].
    pub fn with_storage_fault_policy(mut self, policy: Arc<dyn FaultPolicy>) -> Self {
        self.storage_fault_policy = Some(policy);
        self
    }

    /// Decide each dial's success and each send's latency or reset with `policy`.
    ///
    /// Without a policy, connections are lossless, ordered, zero-latency pipes. The policy sees
    /// every dial and send with the connection it concerns, so it can derive, record, replay, or
    /// override each decision independently. The policy survives [Runner::start_and_recover].
    pub fn with_network_policy(mut self, policy: Arc<dyn NetworkPolicy>) -> Self {
        self.network_policy = Some(policy);
        self
    }

    // Getters
    /// See [Config]
    pub const fn cycle(&self) -> Duration {
        self.cycle
    }
    /// See [Config]
    pub const fn start_time(&self) -> SystemTime {
        self.start_time
    }
    /// See [Config]
    pub const fn timeout(&self) -> Option<Duration> {
        self.timeout
    }
    /// See [Config]
    pub const fn catch_panics(&self) -> bool {
        self.catch_panics
    }
    /// See [Config]
    pub const fn network_buffer_pool_config(&self) -> &BufferPoolConfig {
        &self.network_buffer_pool_cfg
    }
    /// See [Config]
    pub const fn storage_buffer_pool_config(&self) -> &BufferPoolConfig {
        &self.storage_buffer_pool_cfg
    }

    /// Assert that the configuration is valid.
    pub fn assert(&self) {
        assert!(
            self.cycle != Duration::default() || self.timeout.is_none(),
            "cycle duration must be non-zero when timeout is set",
        );
        assert!(
            self.cycle >= SYSTEM_TIME_PRECISION,
            "cycle duration must be greater than or equal to system time precision"
        );
        assert!(
            self.start_time >= UNIX_EPOCH,
            "start time must be greater than or equal to unix epoch"
        );
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

/// Deterministic runtime that randomly selects tasks to run based on a seed.
pub struct Executor {
    registry: Registry,
    cycle: Duration,
    deadline: Option<SystemTime>,
    metrics: Arc<Metrics>,
    auditor: Arc<Auditor>,
    rng: Arc<Mutex<BoxDynRng>>,
    scheduling_policy: Option<Arc<Mutex<BoxDynSchedulingPolicy>>>,
    network_policy: Option<Arc<dyn NetworkPolicy>>,
    time: Mutex<SystemTime>,
    tasks: Arc<Tasks>,
    sleeping: Mutex<BinaryHeap<Alarm>>,
    /// Woken tasks of paused processes, polled again once their process resumes.
    parked: Mutex<Vec<u128>>,
    /// Live simulated processes, in start order.
    processes: Mutex<Vec<Arc<ProcessHost>>>,
    /// Tasks held back by the scheduling policy, and when each is released.
    held: Mutex<Held>,
    shutdown: Mutex<Stopper>,
    panicker: Panicker,
    dns: Mutex<HashMap<String, Vec<IpAddr>>>,
}

impl Executor {
    fn order_ready(&self, current: SystemTime, queue: &mut Vec<u128>) {
        let Some(policy) = &self.scheduling_policy else {
            if queue.len() > 1 {
                queue.shuffle(&mut *self.rng.lock());
            }
            return;
        };
        let mut occurrences = BTreeMap::<u128, u64>::new();
        let mut ready: Vec<_> = queue
            .iter()
            .filter_map(|id| {
                let task = self.tasks.get(*id)?;
                let occurrence = occurrences.entry(*id).or_default();
                let item = RunnableTask {
                    id: *id,
                    label: task.label.name(),
                    occurrence: *occurrence,
                };
                *occurrence += 1;
                Some(item)
            })
            .collect();
        let mut before: Vec<_> = ready
            .iter()
            .map(|task| (task.id, task.occurrence))
            .collect();
        before.sort_unstable();
        let mut policy = policy.lock();
        policy.order(current, &mut ready);
        let mut after: Vec<_> = ready
            .iter()
            .map(|task| (task.id, task.occurrence))
            .collect();
        after.sort_unstable();
        assert_eq!(
            before, after,
            "scheduling policy changed the runnable batch"
        );

        // Hold back the tasks the policy pauses; a released task is polled without asking
        let mut held = self.held.lock();
        queue.clear();
        for task in ready {
            if held.released.remove(&task.id) {
                queue.push(task.id);
                continue;
            }
            if held.until.contains_key(&task.id) {
                continue;
            }
            let hold = policy.hold(current, &task);
            if hold.is_zero() {
                queue.push(task.id);
                continue;
            }
            let until = current
                .checked_add(hold)
                .expect("overflow when holding task");
            self.auditor.event(b"hold_task", |hasher| {
                hasher.update(task.id.to_be_bytes());
                hasher.update(hold.as_nanos().to_be_bytes());
            });
            held.until.insert(task.id, until);
            held.releases.insert((until, task.id));
        }
    }

    /// Crash `host` unless it already crashed.
    fn crash_process(&self, host: &Arc<ProcessHost>) {
        let live = {
            let mut processes = self.processes.lock();
            processes
                .iter()
                .position(|process| Arc::ptr_eq(process, host))
                .map(|index| processes.remove(index))
        };
        if live.is_some() {
            host.crash(self);
        }
    }

    /// Requeue every parked task; those whose process is still paused park again.
    fn unpark(&self) {
        let parked = take(&mut *self.parked.lock());
        let mut seen = BTreeSet::new();
        for id in parked {
            if seen.insert(id) {
                self.tasks.queue(id);
            }
        }
    }

    /// Advance simulated time by [Config::cycle].
    ///
    /// When built with the `external` feature, sleep for [Config::cycle] to let
    /// external processes make progress.
    fn advance_time(&self) -> SystemTime {
        #[cfg(feature = "external")]
        std::thread::sleep(self.cycle);

        let mut time = self.time.lock();
        *time = time
            .checked_add(self.cycle)
            .expect("executor time overflowed");
        let now = *time;
        trace!(now = now.epoch_millis(), "time advanced");
        now
    }

    /// Ensure the runtime has not reached its configured deadline.
    fn assert_deadline(&self, current: SystemTime) {
        if self.deadline.is_some_and(|deadline| current >= deadline) {
            panic!("runtime timeout");
        }
    }

    /// When idle, jump directly to the next actionable time.
    ///
    /// When built with the `external` feature, never skip ahead (to ensure we poll all pending tasks
    /// every [Config::cycle]).
    fn skip_idle_time(&self, current: SystemTime) -> SystemTime {
        if cfg!(feature = "external") || self.tasks.ready() != 0 {
            return current;
        }

        // The next alarm or held-task release, unless one is already due
        let next_alarm = self.sleeping.lock().peek().map(|alarm| alarm.time);
        let next_release = self
            .held
            .lock()
            .releases
            .first()
            .map(|(release, _)| *release);
        let next = match (next_alarm, next_release) {
            (Some(alarm), Some(release)) => Some(alarm.min(release)),
            (next, None) | (None, next) => next,
        };
        let skip_until = next.filter(|next| *next > current);

        skip_until.map_or(current, |deadline| {
            let mut time = self.time.lock();
            *time = deadline;
            let now = *time;
            trace!(now = now.epoch_millis(), "time skipped");
            now
        })
    }

    /// Requeue every held task whose hold has elapsed.
    fn release_held(&self, current: SystemTime) {
        let mut held = self.held.lock();
        while let Some(&(release, id)) = held.releases.first() {
            if release > current {
                break;
            }
            held.releases.pop_first();
            held.until.remove(&id);
            if self.tasks.get(id).is_some() {
                held.released.insert(id);
                self.tasks.queue(id);
            }
        }
    }

    /// Wake any sleepers whose deadlines have elapsed.
    fn wake_ready_sleepers(&self, current: SystemTime) {
        let mut sleeping = self.sleeping.lock();
        while let Some(next) = sleeping.peek() {
            if next.time <= current {
                let sleeper = sleeping.pop().unwrap();
                sleeper.waker.wake();
            } else {
                break;
            }
        }
    }

    /// Wake sleepers until the runtime can make progress.
    ///
    /// Canceling a polled sleep leaves its alarm registered until its deadline. If that alarm
    /// wakes no task, continue to later deadlines before deciding the runtime has stalled.
    ///
    /// When built with the `external` feature, the passage of time is sufficient to continue.
    fn wake_until_progress(&self, mut current: SystemTime) {
        loop {
            // Move to the next actionable time. Check the runtime deadline before waking sleepers
            // so timeout takes precedence over work scheduled at the deadline.
            current = self.skip_idle_time(current);
            self.assert_deadline(current);
            self.wake_ready_sleepers(current);
            self.release_held(current);

            // Continue once external work or a woken task can make progress. Without either,
            // another alarm or held task is the runtime's only remaining source of progress.
            if cfg!(feature = "external") || self.tasks.ready() != 0 {
                return;
            }
            if self.sleeping.lock().is_empty() && self.held.lock().releases.is_empty() {
                panic!("runtime stalled");
            }
        }
    }
}

/// An artifact that can be used to recover the state of the runtime.
///
/// This is useful when mocking unclean shutdown (while retaining deterministic behavior).
pub struct Checkpoint {
    cycle: Duration,
    deadline: Option<SystemTime>,
    auditor: Arc<Auditor>,
    rng: Arc<Mutex<BoxDynRng>>,
    scheduling_policy: Option<Arc<Mutex<BoxDynSchedulingPolicy>>>,
    time: Mutex<SystemTime>,
    storage: MemStorageSnapshot,
    storage_fault_cfg: FaultConfig,
    storage_fault_policy: Arc<dyn FaultPolicy>,
    network_policy: Option<Arc<dyn NetworkPolicy>>,
    dns: Mutex<HashMap<String, Vec<IpAddr>>>,
    catch_panics: bool,
    network_buffer_pool_cfg: BufferPoolConfig,
    storage_buffer_pool_cfg: BufferPoolConfig,
}

impl Checkpoint {
    /// Get a reference to the [Auditor].
    pub fn auditor(&self) -> Arc<Auditor> {
        self.auditor.clone()
    }
}

#[allow(clippy::large_enum_variant)]
enum State {
    Config(Config),
    Checkpoint(Checkpoint),
}

/// Implementation of [crate::Runner] for the `deterministic` runtime.
pub struct Runner {
    state: State,
}

impl From<Config> for Runner {
    fn from(cfg: Config) -> Self {
        Self::new(cfg)
    }
}

impl From<Checkpoint> for Runner {
    fn from(checkpoint: Checkpoint) -> Self {
        Self {
            state: State::Checkpoint(checkpoint),
        }
    }
}

impl Runner {
    /// Initialize a new `deterministic` runtime with the given seed and cycle duration.
    pub fn new(cfg: Config) -> Self {
        // Ensure config is valid
        cfg.assert();
        Self {
            state: State::Config(cfg),
        }
    }

    /// Initialize a new `deterministic` runtime with the default configuration
    /// and the provided seed.
    pub fn seeded(seed: u64) -> Self {
        Self::new(Config::default().with_seed(seed))
    }

    /// Initialize a new `deterministic` runtime with the default configuration
    /// but exit after the given timeout.
    pub fn timed(timeout: Duration) -> Self {
        let cfg = Config {
            timeout: Some(timeout),
            ..Config::default()
        };
        Self::new(cfg)
    }

    /// Like [crate::Runner::start], but also returns a [Checkpoint] that can be used
    /// to recover the state of the runtime in a subsequent run.
    pub fn start_and_recover<F, Fut>(self, f: F) -> (Fut::Output, Checkpoint)
    where
        F: FnOnce(Context) -> Fut,
        Fut: Future,
    {
        // Setup context and return strong reference to executor
        let (context, executor, panicked) = match self.state {
            State::Config(config) => Context::new(config),
            State::Checkpoint(checkpoint) => Context::recover(checkpoint),
        };

        // Pin root task to the heap
        let storage = context.storage.clone();
        let network_buffer_pool_cfg = context.network_buffer_pool.config().clone();
        let storage_buffer_pool_cfg = context.storage_buffer_pool.config().clone();
        let mut root = Box::pin(panicked.interrupt(f(context)));

        // Register the root task
        Tasks::register_root(&executor.tasks);

        // Process tasks until root task completes or progress stalls.
        // Wrap the loop in catch_unwind to ensure task cleanup runs even if the loop or a task panics.
        let result = catch_unwind(AssertUnwindSafe(|| {
            loop {
                // Ensure we have not exceeded our deadline
                let current = *executor.time.lock();
                executor.assert_deadline(current);

                // Drain all ready tasks
                let mut queue = executor.tasks.drain();

                executor.order_ready(current, &mut queue);

                // Run all snapshotted tasks
                //
                // This approach is more efficient than randomly selecting a task one-at-a-time
                // because it ensures we don't pull the same pending task multiple times in a row (without
                // processing a different task required for other tasks to make progress).
                trace!(
                    iter = executor.metrics.iterations.get(),
                    tasks = queue.len(),
                    "starting loop"
                );
                let mut output = None;
                for id in queue {
                    // Lookup the task (it may have completed already)
                    let Some(task) = executor.tasks.get(id) else {
                        trace!(id, "skipping missing task");
                        continue;
                    };

                    // Hold back the tasks of paused processes until they resume
                    if task
                        .process
                        .as_ref()
                        .is_some_and(|process| process.paused())
                    {
                        trace!(id, "parking task of paused process");
                        executor.parked.lock().push(id);
                        continue;
                    }

                    // Record task for auditing
                    executor.auditor.event(b"process_task", |hasher| {
                        hasher.update(task.id.to_be_bytes());
                        hasher.update(task.label.name().as_bytes());
                    });
                    executor.metrics.task_polls.get_or_create(&task.label).inc();
                    trace!(id, "processing task");

                    // Prepare task for polling
                    let waker = waker(Arc::new(TaskWaker {
                        id,
                        tasks: Arc::downgrade(&executor.tasks),
                    }));
                    let mut cx = task::Context::from_waker(&waker);

                    // Poll the task
                    match &task.mode {
                        Mode::Root => {
                            // Poll the root task
                            if let Poll::Ready(result) = root.as_mut().poll(&mut cx) {
                                trace!(id, "root task is complete");
                                output = Some(result);
                                break;
                            }
                        }
                        Mode::Work(future) => {
                            // Get the future (if it still exists)
                            let mut fut_opt = future.lock();
                            let Some(fut) = fut_opt.as_mut() else {
                                trace!(id, "skipping already complete task");

                                // Remove the future
                                executor.tasks.remove(id);
                                continue;
                            };

                            // Poll the task
                            if fut.as_mut().poll(&mut cx).is_ready() {
                                trace!(id, "task is complete");

                                // Remove the future
                                executor.tasks.remove(id);
                                *fut_opt = None;
                                continue;
                            }
                        }
                    }

                    // Try again later if task is still pending
                    trace!(id, "task is still pending");
                }

                // If the root task has completed, exit as soon as possible
                if let Some(output) = output {
                    break output;
                }

                // Advance time and wake sleepers until the runtime can make progress
                let current = executor.advance_time();
                executor.wake_until_progress(current);

                // Record that we completed another iteration of the event loop.
                executor.metrics.iterations.inc();
            }
        }));

        // Clear remaining tasks from the executor.
        //
        // It is critical that we wait to drop the strong
        // reference to executor until after we have dropped
        // all tasks (as they may attempt to upgrade their weak
        // reference to the executor during drop).
        executor.sleeping.lock().clear(); // included in tasks
        let tasks = executor.tasks.clear();
        for task in tasks {
            let Mode::Work(future) = &task.mode else {
                continue;
            };
            *future.lock() = None;
        }

        // Drop the root task to release any Context references it may still hold.
        // This is necessary when the loop exits early (e.g., timeout) while the
        // root future is still Pending and holds captured variables with Context references.
        drop(root);

        // Release simulated processes and the storage's way back to the executor.
        storage.inner().inner().crasher().lock().take();
        executor.processes.lock().clear();

        // No task can issue or make a write durable after this crash boundary.
        storage
            .inner()
            .inner()
            .crash()
            .expect("retaining successful unsynced writes at crash should succeed");
        let storage_fault_cfg = storage.inner().inner().config().read().clone();
        let storage_fault_policy = storage.inner().inner().policy();
        let storage = storage.inner().inner().inner().take_snapshot();

        // Assert the context doesn't escape the start() function (behavior
        // is undefined in this case)
        assert!(
            Arc::weak_count(&executor) == 0,
            "executor still has weak references"
        );

        // Handle the result — resume the original panic after cleanup if one was caught.
        let output = match result {
            Ok(output) => output,
            Err(payload) => resume_unwind(payload),
        };

        // Extract the executor from the Arc
        let executor = Arc::into_inner(executor).expect("executor still has strong references");

        // Construct a checkpoint that can be used to restart the runtime
        let checkpoint = Checkpoint {
            cycle: executor.cycle,
            deadline: executor.deadline,
            auditor: executor.auditor,
            rng: executor.rng,
            scheduling_policy: executor.scheduling_policy,
            network_policy: executor.network_policy,
            time: executor.time,
            storage,
            storage_fault_cfg,
            storage_fault_policy,
            dns: executor.dns,
            catch_panics: executor.panicker.catch(),
            network_buffer_pool_cfg,
            storage_buffer_pool_cfg,
        };

        (output, checkpoint)
    }
}

impl Default for Runner {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

impl crate::Runner for Runner {
    type Context = Context;

    fn start<F, Fut>(self, f: F) -> Fut::Output
    where
        F: FnOnce(Self::Context) -> Fut,
        Fut: Future,
    {
        let (output, _) = self.start_and_recover(f);
        output
    }
}

/// The mode of a [Task].
enum Mode {
    Root,
    Work(Mutex<Option<Pin<Box<dyn Future<Output = ()> + Send + 'static>>>>),
}

/// A future being executed by the [Executor].
struct Task {
    id: u128,
    label: Label,
    /// The simulated process the task belongs to, if any.
    process: Option<Arc<ProcessState>>,

    mode: Mode,
}

/// A waker for a [Task].
struct TaskWaker {
    id: u128,

    tasks: Weak<Tasks>,
}

impl ArcWake for TaskWaker {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        // Upgrade the weak reference to re-enqueue this task.
        // If upgrade fails, the task queue has been dropped and no action is required.
        //
        // This can happen if some data is passed into the runtime and it drops after the runtime exits.
        if let Some(tasks) = arc_self.tasks.upgrade() {
            tasks.queue(arc_self.id);
        }
    }
}

/// A collection of [Task]s that are being executed by the [Executor].
struct Tasks {
    /// The next task id.
    counter: Mutex<u128>,
    /// Tasks ready to be polled.
    ready: Mutex<Vec<u128>>,
    /// All running tasks.
    running: Mutex<BTreeMap<u128, Arc<Task>>>,
}

impl Tasks {
    /// Create a new task queue.
    const fn new() -> Self {
        Self {
            counter: Mutex::new(0),
            ready: Mutex::new(Vec::new()),
            running: Mutex::new(BTreeMap::new()),
        }
    }

    /// Increment the task counter and return the old value.
    fn increment(&self) -> u128 {
        let mut counter = self.counter.lock();
        let old = *counter;
        *counter = counter.checked_add(1).expect("task counter overflow");
        old
    }

    /// Register the root task.
    ///
    /// If the root task has already been registered, this function will panic.
    fn register_root(arc_self: &Arc<Self>) {
        let id = arc_self.increment();
        let task = Arc::new(Task {
            id,
            label: Label::root(),
            process: None,
            mode: Mode::Root,
        });
        arc_self.register(id, task);
    }

    /// Register a non-root task to be executed.
    fn register_work(
        arc_self: &Arc<Self>,
        label: Label,
        process: Option<Arc<ProcessState>>,
        future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
    ) {
        let id = arc_self.increment();
        let task = Arc::new(Task {
            id,
            label,
            process,
            mode: Mode::Work(Mutex::new(Some(future))),
        });
        arc_self.register(id, task);
    }

    /// Register a new task to be executed.
    fn register(&self, id: u128, task: Arc<Task>) {
        // Track as running until completion
        self.running.lock().insert(id, task);

        // Add to ready
        self.queue(id);
    }

    /// Enqueue an already registered task to be executed.
    fn queue(&self, id: u128) {
        let mut ready = self.ready.lock();
        ready.push(id);
    }

    /// Drain all ready tasks.
    fn drain(&self) -> Vec<u128> {
        let mut queue = self.ready.lock();
        let len = queue.len();
        replace(&mut *queue, Vec::with_capacity(len))
    }

    /// The number of ready tasks.
    fn ready(&self) -> usize {
        self.ready.lock().len()
    }

    /// Lookup a task.
    ///
    /// We must return cloned here because we cannot hold the running lock while polling a task (will
    /// deadlock if [Self::register_work] is called).
    fn get(&self, id: u128) -> Option<Arc<Task>> {
        let running = self.running.lock();
        running.get(&id).cloned()
    }

    /// Remove a task.
    fn remove(&self, id: u128) {
        self.running.lock().remove(&id);
    }

    /// Clear all tasks.
    fn clear(&self) -> Vec<Arc<Task>> {
        // Clear ready
        self.ready.lock().clear();

        // Clear running tasks
        let running: BTreeMap<u128, Arc<Task>> = {
            let mut running = self.running.lock();
            take(&mut *running)
        };
        running.into_values().collect()
    }
}

type Network = MeteredNetwork<AuditedNetwork<DeterministicNetwork>>;
type Storage = MeteredStorage<AuditedStorage<FaultyStorage<MemStorage>>>;

/// A blob handle whose open stays exclusive until its final owner drops or the blob is removed.
pub type Blob = OpenBlob<MeteredBlob<AuditedBlob<FaultyBlob<MemBlob>>>>;

fn build_storage(
    inner: MemStorage,
    policy: Arc<dyn FaultPolicy>,
    faults: FaultConfig,
    auditor: Arc<Auditor>,
    registry: &mut impl Register,
) -> Storage {
    MeteredStorage::new(
        AuditedStorage::new(
            FaultyStorage::with_policy(inner, policy, Arc::new(RwLock::new(faults))),
            auditor,
        ),
        registry,
    )
}

/// Implementation of [crate::Spawner], [crate::Clock],
/// [crate::Network], and [crate::Storage] for the `deterministic`
/// runtime.
pub struct Context {
    name: String,
    attributes: Vec<(String, String)>,
    executor: Weak<Executor>,
    network: Arc<Network>,
    storage: Arc<Storage>,
    opens: Arc<Opens>,
    network_buffer_pool: BufferPool,
    storage_buffer_pool: BufferPool,
    tree: Arc<Tree>,
    process: Option<Arc<ProcessState>>,
    execution: Execution,
}

impl Context {
    fn new(cfg: Config) -> (Self, Arc<Executor>, Panicked) {
        // Create a new registry
        let mut registry = Registry::new();
        let mut runtime_registry = registry.sub_registry(METRICS_PREFIX);

        // Initialize runtime
        let metrics = Arc::new(Metrics::init(&mut runtime_registry));
        let start_time = cfg.start_time;
        let deadline = cfg
            .timeout
            .map(|timeout| start_time.checked_add(timeout).expect("timeout overflowed"));
        let auditor = Arc::new(Auditor::default());

        // Create shared RNG (used by both executor and storage)
        let rng = Arc::new(Mutex::new(cfg.rng));

        // Initialize buffer pools
        let network_buffer_pool = BufferPool::new(
            cfg.network_buffer_pool_cfg.clone(),
            &mut runtime_registry.sub_registry("network_buffer_pool"),
        );
        let storage_buffer_pool = BufferPool::new(
            cfg.storage_buffer_pool_cfg.clone(),
            &mut runtime_registry.sub_registry("storage_buffer_pool"),
        );

        let policy = cfg
            .storage_fault_policy
            .unwrap_or_else(|| Arc::new(SharedRng(rng.clone())));
        let storage = build_storage(
            MemStorage::new(storage_buffer_pool.clone()),
            policy,
            cfg.storage_fault_cfg,
            auditor.clone(),
            &mut runtime_registry,
        );

        // Create network; its timer is installed once the executor exists
        let timer = TimerSlot::default();
        let network = AuditedNetwork::new(
            DeterministicNetwork::with_policy(cfg.network_policy.clone(), timer.clone()),
            auditor.clone(),
        );
        let network = MeteredNetwork::new(network, &mut runtime_registry);

        // Initialize panicker
        let (panicker, panicked) = Panicker::new(cfg.catch_panics);

        let executor = Arc::new(Executor {
            registry,
            cycle: cfg.cycle,
            deadline,
            metrics,
            auditor,
            rng,
            scheduling_policy: cfg
                .scheduling_policy
                .map(|policy| Arc::new(Mutex::new(policy))),
            network_policy: cfg.network_policy,
            time: Mutex::new(start_time),
            tasks: Arc::new(Tasks::new()),
            sleeping: Mutex::new(BinaryHeap::new()),
            parked: Mutex::new(Vec::new()),
            processes: Mutex::new(Vec::new()),
            held: Mutex::new(Held::default()),
            shutdown: Mutex::new(Stopper::default()),
            panicker,
            dns: Mutex::new(HashMap::new()),
        });
        install_timer(&timer, &executor);
        install_crasher(&storage, &executor);

        (
            Self {
                name: String::new(),
                attributes: Vec::new(),
                executor: Arc::downgrade(&executor),
                network: Arc::new(network),
                storage: Arc::new(storage),
                opens: Arc::default(),
                network_buffer_pool,
                storage_buffer_pool,
                tree: Tree::root(),
                process: None,
                execution: Execution::default(),
            },
            executor,
            panicked,
        )
    }

    /// Recover the inner state (deadline, metrics, auditor, rng, storage, etc.) from the current
    /// runtime and use it to initialize a new instance of the runtime. Storage recovery includes
    /// durable state and any unsynchronized mutations retained by the configured crash policy. A
    /// recovered runtime does not inherit pending tasks, network connections, or its shutdown
    /// signaler.
    ///
    /// This is useful for performing a deterministic simulation that spans multiple runtime instantiations,
    /// like simulating unclean shutdown (which involves repeatedly halting the runtime at unexpected intervals).
    ///
    /// It is only permitted to call this method after the runtime has finished (i.e. once `start` returns)
    /// and only permitted to do once (otherwise multiple recovered runtimes will share the same inner state).
    /// If either one of these conditions is violated, this method will panic.
    fn recover(checkpoint: Checkpoint) -> (Self, Arc<Executor>, Panicked) {
        // Rebuild metrics
        let mut registry = Registry::new();
        let mut runtime_registry = registry.sub_registry(METRICS_PREFIX);
        let metrics = Arc::new(Metrics::init(&mut runtime_registry));

        // Copy state
        let timer = TimerSlot::default();
        let network = AuditedNetwork::new(
            DeterministicNetwork::with_policy(checkpoint.network_policy.clone(), timer.clone()),
            checkpoint.auditor.clone(),
        );
        let network = MeteredNetwork::new(network, &mut runtime_registry);

        // Initialize buffer pools
        let network_buffer_pool = BufferPool::new(
            checkpoint.network_buffer_pool_cfg.clone(),
            &mut runtime_registry.sub_registry("network_buffer_pool"),
        );
        let storage_buffer_pool = BufferPool::new(
            checkpoint.storage_buffer_pool_cfg.clone(),
            &mut runtime_registry.sub_registry("storage_buffer_pool"),
        );
        let storage = build_storage(
            MemStorage::from_snapshot(checkpoint.storage, storage_buffer_pool.clone()),
            checkpoint.storage_fault_policy,
            checkpoint.storage_fault_cfg,
            checkpoint.auditor.clone(),
            &mut runtime_registry,
        );

        // Initialize panicker
        let (panicker, panicked) = Panicker::new(checkpoint.catch_panics);

        let executor = Arc::new(Executor {
            // Copied from the checkpoint
            cycle: checkpoint.cycle,
            deadline: checkpoint.deadline,
            auditor: checkpoint.auditor,
            rng: checkpoint.rng,
            scheduling_policy: checkpoint.scheduling_policy,
            network_policy: checkpoint.network_policy,
            time: checkpoint.time,
            dns: checkpoint.dns,

            // New state for the new runtime
            registry,
            metrics,
            tasks: Arc::new(Tasks::new()),
            sleeping: Mutex::new(BinaryHeap::new()),
            parked: Mutex::new(Vec::new()),
            processes: Mutex::new(Vec::new()),
            held: Mutex::new(Held::default()),
            shutdown: Mutex::new(Stopper::default()),
            panicker,
        });
        install_timer(&timer, &executor);
        install_crasher(&storage, &executor);
        (
            Self {
                name: String::new(),
                attributes: Vec::new(),
                executor: Arc::downgrade(&executor),
                network: Arc::new(network),
                storage: Arc::new(storage),
                opens: Arc::default(),
                network_buffer_pool,
                storage_buffer_pool,
                tree: Tree::root(),
                process: None,
                execution: Execution::default(),
            },
            executor,
            panicked,
        )
    }

    /// Upgrade Weak reference to [Executor].
    fn executor(&self) -> Arc<Executor> {
        self.executor.upgrade().expect("executor already dropped")
    }

    /// Signed offset of this context's clock from the runtime's, in nanoseconds.
    fn clock_offset(&self) -> i128 {
        self.process.as_ref().map_or(0, |process| process.offset())
    }

    /// Get a reference to [Metrics].
    fn metrics(&self) -> Arc<Metrics> {
        self.executor().metrics.clone()
    }

    /// Get a reference to the [Auditor].
    pub fn auditor(&self) -> Arc<Auditor> {
        self.executor().auditor.clone()
    }

    /// Compute a [Sha256] digest of all storage contents.
    pub fn storage_audit(&self) -> Digest {
        self.storage.inner().inner().inner().audit()
    }

    /// Return a copy of a blob's durable logical contents without opening it, or `None` when
    /// the blob is missing or its container header does not resolve.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn logical_blob(&self, partition: &str, name: &[u8]) -> Option<Vec<u8>> {
        self.storage
            .inner()
            .inner()
            .inner()
            .logical_blob(partition, name)
    }

    /// Apply `f` to a blob's durable logical contents.
    ///
    /// # Panics
    ///
    /// Panics if the blob is missing, its container header does not resolve, or a handle that
    /// can still publish to it is open.
    fn corrupt_durable<R>(
        &self,
        partition: &str,
        name: &[u8],
        f: impl FnOnce(&mut [u8]) -> R,
    ) -> R {
        let memory = self.storage.inner().inner().inner();
        self.opens.inspect(partition, name, |live| {
            assert!(
                !live || !memory.is_current(partition, name),
                "blob {partition}/{} is open; crash its process before corrupting it",
                hex(name)
            );
            memory
                .update_logical(partition, name, f)
                .unwrap_or_else(|| panic!("blob {partition}/{} has no durable contents", hex(name)))
        })
    }

    /// Flip bit `bit` (bit `bit % 8` of byte `bit / 8`) of a blob's durable logical contents,
    /// as a cosmic ray or media error would.
    ///
    /// Offsets exclude the runtime's blob container header. The corruption persists: the next
    /// open reads it. Unsynchronized writes still awaiting a crash outcome are discarded. Use this
    /// while the owning process is down (after [Process::crash]); handles opened before that
    /// crash may still be alive, but they can no longer publish.
    ///
    /// # Panics
    ///
    /// Panics if the blob is missing, `bit` is outside its logical contents, or a handle that
    /// can still publish to it is open.
    pub fn corrupt_bit(&self, partition: &str, name: &[u8], bit: u64) {
        self.auditor().event(b"corrupt_bit", |hasher| {
            audit_blob(hasher, partition, name);
            hasher.update(bit.to_be_bytes());
        });
        self.corrupt_durable(partition, name, |content| {
            let bits = (content.len() as u64).saturating_mul(8);
            assert!(
                bit < bits,
                "bit {bit} is outside blob {partition}/{} ({bits} bits)",
                hex(name)
            );
            content[(bit / 8) as usize] ^= 1 << (bit % 8);
        });
    }

    /// Overwrite the durable logical contents of `target` starting at `offset` with the durable
    /// bytes `range` of `source`, as a misdirected write would. `source` and `target` may be the
    /// same blob, and the ranges may overlap.
    ///
    /// The target's length is unchanged. Offsets, preconditions, and persistence are as in
    /// [Self::corrupt_bit]; only `target` must have no handle that can still publish.
    ///
    /// # Panics
    ///
    /// Panics if either blob is missing, `range` is outside the source's logical contents, the
    /// copied range does not fit within the target's logical contents, or a handle that can
    /// still publish to `target` is open.
    pub fn misdirect(
        &self,
        source: (&str, &[u8]),
        range: Range<u64>,
        target: (&str, &[u8]),
        offset: u64,
    ) {
        self.auditor().event(b"misdirect", |hasher| {
            audit_blob(hasher, source.0, source.1);
            hasher.update(range.start.to_be_bytes());
            hasher.update(range.end.to_be_bytes());
            audit_blob(hasher, target.0, target.1);
            hasher.update(offset.to_be_bytes());
        });
        let bytes = self
            .storage
            .inner()
            .inner()
            .inner()
            .logical_blob(source.0, source.1)
            .unwrap_or_else(|| {
                panic!(
                    "blob {}/{} has no durable contents",
                    source.0,
                    hex(source.1)
                )
            });
        assert!(
            range.start <= range.end && range.end <= bytes.len() as u64,
            "range {range:?} is outside blob {}/{} ({} bytes)",
            source.0,
            hex(source.1),
            bytes.len()
        );
        let bytes = &bytes[range.start as usize..range.end as usize];
        self.corrupt_durable(target.0, target.1, |content| {
            let end = offset
                .checked_add(bytes.len() as u64)
                .filter(|end| *end <= content.len() as u64)
                .unwrap_or_else(|| {
                    panic!(
                        "{} bytes at offset {offset} exceed blob {}/{} ({} bytes)",
                        bytes.len(),
                        target.0,
                        hex(target.1),
                        content.len()
                    )
                });
            content[offset as usize..end as usize].copy_from_slice(bytes);
        });
    }

    /// Capture a blob's durable contents (including its length and container header), which
    /// [Self::restore_blob] reinstates to simulate lost writes.
    ///
    /// Only synchronized contents are captured; unsynchronized writes of open handles are not.
    ///
    /// # Panics
    ///
    /// Panics if the blob is missing.
    pub fn snapshot_blob(&self, partition: &str, name: &[u8]) -> BlobSnapshot {
        self.auditor().event(b"snapshot_blob", |hasher| {
            audit_blob(hasher, partition, name);
        });
        let raw = self
            .storage
            .inner()
            .inner()
            .inner()
            .raw_blob(partition, name)
            .unwrap_or_else(|| panic!("blob {partition}/{} is missing", hex(name)));
        BlobSnapshot {
            partition: partition.to_string(),
            name: name.to_vec(),
            raw,
        }
    }

    /// Revert a blob's durable contents, including its length, to `snapshot`, as if every write
    /// synchronized since the snapshot was lost. Recreates the blob if it was removed.
    ///
    /// Unsynchronized writes still awaiting a crash outcome are discarded. Use this while the
    /// owning process is down (after [Process::crash]); handles opened before that crash may
    /// still be alive, but they can no longer publish.
    ///
    /// # Panics
    ///
    /// Panics if a handle that can still publish to the blob is open.
    pub fn restore_blob(&self, snapshot: &BlobSnapshot) {
        let BlobSnapshot {
            partition,
            name,
            raw,
        } = snapshot;
        self.auditor().event(b"restore_blob", |hasher| {
            audit_blob(hasher, partition, name);
            hasher.update((raw.len() as u64).to_be_bytes());
            hasher.update(raw);
        });
        let memory = self.storage.inner().inner().inner();
        self.opens.inspect(partition, name, |live| {
            assert!(
                !live || !memory.is_current(partition, name),
                "blob {partition}/{} is open; crash its process before restoring it",
                hex(name)
            );
            memory.set_raw_blob(partition, name, raw.clone());
        });
    }

    /// Access the storage fault configuration.
    ///
    /// Changes to the returned [`FaultConfig`] take effect immediately for
    /// subsequent storage operations. This allows dynamically enabling or
    /// disabling fault injection during a test.
    pub fn storage_fault_config(&self) -> Arc<RwLock<FaultConfig>> {
        self.storage.inner().inner().config()
    }

    /// Start a simulated process that owns every partition `partitions` selects.
    ///
    /// Returns the process's context, from which its tasks are spawned, and a [Process] that
    /// crashes, pauses, or skews the clock of it. Peers started from other contexts keep running
    /// when it crashes, so one process can crash and restart (by opening its partitions again
    /// from a new process) while the rest of the simulation continues. With
    /// [FaultConfig::crash], a write or sync of an owned partition can also crash the process
    /// from within that operation (the latest started live process owning a partition crashes).
    pub fn process(
        &self,
        label: &'static str,
        partitions: impl Fn(&str) -> bool + Send + Sync + 'static,
    ) -> (Self, Process) {
        let mut context = crate::Supervisor::child(self, label);
        let state = Arc::new(ProcessState {
            parent: context.process.take(),
            paused: AtomicBool::new(false),
            offset: Mutex::new(0),
        });
        context.process = Some(Arc::clone(&state));
        let host = Arc::new(ProcessHost {
            state,
            tree: Arc::clone(&context.tree),
            storage: context.storage.clone(),
            partitions: Box::new(partitions),
            crashed: AtomicBool::new(false),
        });
        self.executor().processes.lock().push(Arc::clone(&host));
        let process = Process {
            host,
            executor: context.executor.clone(),
        };
        (context, process)
    }

    /// Register a DNS mapping for a hostname.
    ///
    /// If `addrs` is `None`, the mapping is removed.
    /// If `addrs` is `Some`, the mapping is added or updated.
    pub fn resolver_register(&self, host: impl Into<String>, addrs: Option<Vec<IpAddr>>) {
        // Update the auditor
        let executor = self.executor();
        let host = host.into();
        executor.auditor.event(b"resolver_register", |hasher| {
            hasher.update(host.as_bytes());
            hasher.update(addrs.encode());
        });

        // Update the DNS mapping
        let mut dns = executor.dns.lock();
        match addrs {
            Some(addrs) => {
                dns.insert(host, addrs);
            }
            None => {
                dns.remove(&host);
            }
        }
    }
}

impl crate::Spawner for Context {
    fn dedicated(mut self) -> Self {
        self.execution = Execution::Dedicated;
        self
    }

    fn shared(mut self, blocking: bool) -> Self {
        self.execution = Execution::Shared(blocking);
        self
    }

    fn spawn<F, Fut, T>(mut self, f: F) -> Handle<T>
    where
        F: FnOnce(Self) -> Fut + Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        // Get metrics
        let (label, metric) = spawn_metrics!(self);

        // Track supervision before resetting configuration
        let parent = Arc::clone(&self.tree);
        self.execution = Execution::default();
        let (child, aborted) = Tree::child(&parent);
        if aborted {
            return Handle::closed(metric);
        }
        self.tree = child;

        // Spawn the task (we don't care about Model)
        let guard = FactoryGuard::new(&parent, metric);
        let executor = self.executor();
        let process = self.process.clone();
        let future = f(self);
        let (f, handle) = Handle::init(
            future,
            guard.disarm(),
            executor.panicker.clone(),
            Arc::clone(&parent),
        );
        Tasks::register_work(&executor.tasks, label, process, Box::pin(f));

        handle
    }

    async fn stop(self, value: i32, timeout: Option<Duration>) -> Result<(), Error> {
        let executor = self.executor();
        executor.auditor.event(b"stop", |hasher| {
            hasher.update(value.to_be_bytes());
        });
        let stop_resolved = {
            let mut shutdown = executor.shutdown.lock();
            shutdown.stop(value)
        };

        // Wait for all tasks to complete or the timeout to fire
        let timeout_future = timeout.map_or_else(
            || futures::future::Either::Right(futures::future::pending()),
            |duration| futures::future::Either::Left(self.sleep(duration)),
        );
        select! {
            result = stop_resolved => {
                result.map_err(|_| Error::Closed)?;
                Ok(())
            },
            _ = timeout_future => Err(Error::Timeout),
        }
    }

    fn stopped(&self) -> Signal {
        let executor = self.executor();
        executor.auditor.event(b"stopped", |_| {});

        executor.shutdown.lock().stopped()
    }
}

// Rayon permits one permanent registry registration per OS thread. Cache the pool that
// registered the executor thread so later requests and runners reuse it.
commonware_utils::thread_local_cache!(static THREAD_POOL: ThreadPool);

/// Returns the single-threaded pool the executor thread registered with, created on first use.
///
/// All pool work executes inline on the executor thread, so a larger pool would only
/// add permanently unstarted workers.
fn shared_thread_pool() -> Result<ThreadPool, ThreadPoolBuildError> {
    let pool = Cached::take(
        &THREAD_POOL,
        || {
            ThreadPoolBuilder::new()
                .num_threads(1)
                .use_current_thread()
                .build()
                .map(Arc::new)
        },
        |_| Ok(()),
    )?;
    Ok(Arc::clone(&pool))
}

/// Spawning threads would be nondeterministic, so the pool has no background workers. The
/// executor thread registers itself as its sole member and all work executes inline.
///
/// Rayon's current-thread registration is permanent and per-OS-thread, so only one pool
/// can ever execute work on the executor thread. Every request (including from a later
/// runner on the same thread) returns a strategy on that single-threaded pool with its
/// planning parallelism set independently. This controls adaptive decisions and manual
/// partitioning hints while Rayon executes on the sole registered thread. The returned
/// strategy is therefore tied to the executor thread.
impl crate::Strategizer for Context {
    fn strategy(&self, parallelism: NonZeroUsize) -> Rayon {
        Rayon::with_pool(
            shared_thread_pool().expect("failed to create deterministic Rayon thread pool"),
        )
        .with_parallelism(parallelism)
    }
}

/// A simulated process started by [Context::process].
///
/// The process's tasks are those spawned from its context (and their descendants). Its storage
/// is the set of partitions it was started with.
pub struct Process {
    host: Arc<ProcessHost>,
    executor: Weak<Executor>,
}

/// The tasks and storage of a [Process].
struct ProcessHost {
    state: Arc<ProcessState>,
    tree: Arc<Tree>,
    storage: Arc<Storage>,
    partitions: Box<dyn Fn(&str) -> bool + Send + Sync>,
    crashed: AtomicBool,
}

impl ProcessHost {
    fn crash(&self, executor: &Executor) {
        self.tree.abort();
        self.crashed.store(true, AtomicOrdering::Relaxed);
        executor.auditor.event(b"crash_process", |_| {});

        // Aborted tasks parked by a pause are dropped once polled
        self.state.paused.store(false, AtomicOrdering::Relaxed);
        executor.unpark();
        self.storage
            .inner()
            .inner()
            .crash_partitions(&self.partitions)
            .expect("retaining successful unsynced writes at crash should succeed");
    }
}

impl Process {
    /// Crash the process, as a power loss on its host would.
    ///
    /// Every task of the process is aborted before this returns, so none of them runs again
    /// (unlike [Handle::abort], which aborts descendants once the aborted task is next polled).
    /// Its listeners and connections close as their tasks are dropped. Then each owned
    /// partition's unsynchronized writes are resolved as a crash would resolve them (per the
    /// storage fault configuration and policy), and blobs opened before the crash stop publishing.
    ///
    /// Does nothing if the process already crashed (see [Self::crashed]).
    pub fn crash(self) {
        self.executor().crash_process(&self.host);
    }

    /// Whether the process has crashed, either through [Self::crash] or from within one of its
    /// storage operations (see [FaultConfig::crash]).
    pub fn crashed(&self) -> bool {
        self.host.crashed.load(AtomicOrdering::Relaxed)
    }

    /// Pause the process, as `SIGSTOP` would.
    ///
    /// None of its tasks (nor those of processes started from its context) is polled until
    /// [Self::resume]. Time keeps advancing for everything else: its timers expire and messages
    /// sent to it are buffered, and it observes all of that at once when it resumes.
    pub fn pause(&self) {
        self.executor().auditor.event(b"pause_process", |_| {});
        self.host.state.paused.store(true, AtomicOrdering::Relaxed);
    }

    /// Resume a paused process, as `SIGCONT` would.
    pub fn resume(&self) {
        let executor = self.executor();
        executor.auditor.event(b"resume_process", |_| {});
        self.host.state.paused.store(false, AtomicOrdering::Relaxed);
        executor.unpark();
    }

    /// Set how far the process's clock is from the runtime's, replacing any previous offset.
    ///
    /// The offset shifts what [Clock::current] returns and the deadlines passed to
    /// [Clock::sleep_until] in the process (and in processes started from its context), so
    /// setting it jumps the process's clock and setting it repeatedly strobes it. Durations
    /// (as in [Clock::sleep]) and other processes are unaffected.
    pub fn set_clock_offset(&self, offset: ClockOffset) {
        let nanos = offset.nanos();
        self.executor().auditor.event(b"clock_offset", |hasher| {
            hasher.update(nanos.to_be_bytes());
        });
        *self.host.state.offset.lock() = nanos;
    }

    fn executor(&self) -> Arc<Executor> {
        self.executor.upgrade().expect("executor already dropped")
    }
}

/// How far a [Process]'s clock is from the runtime's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClockOffset {
    /// The process's clock reads later than the runtime's by this much.
    Ahead(Duration),
    /// The process's clock reads earlier than the runtime's by this much.
    Behind(Duration),
}

impl ClockOffset {
    fn nanos(self) -> i128 {
        let (duration, sign) = match self {
            Self::Ahead(duration) => (duration, 1),
            Self::Behind(duration) => (duration, -1),
        };
        i128::try_from(duration.as_nanos()).expect("clock offset overflow") * sign
    }
}

/// Pause and clock state shared by the contexts of a [Process].
struct ProcessState {
    /// The process whose context this process was started from, if any.
    parent: Option<Arc<Self>>,
    paused: AtomicBool,
    /// Signed offset of this process's clock from its parent's, in nanoseconds.
    offset: Mutex<i128>,
}

impl ProcessState {
    /// Whether this process or any process it was started from is paused.
    fn paused(&self) -> bool {
        self.paused.load(AtomicOrdering::Relaxed)
            || self.parent.as_ref().is_some_and(|parent| parent.paused())
    }

    /// Signed offset of this process's clock from the runtime's, in nanoseconds.
    fn offset(&self) -> i128 {
        let parent = self.parent.as_ref().map_or(0, |parent| parent.offset());
        parent
            .checked_add(*self.offset.lock())
            .expect("clock offset overflow")
    }
}

/// Shifts `time` by a signed number of nanoseconds.
fn shift(time: SystemTime, nanos: i128) -> SystemTime {
    let magnitude =
        Duration::from_nanos(u64::try_from(nanos.unsigned_abs()).expect("clock offset overflow"));
    if nanos >= 0 {
        time.checked_add(magnitude)
    } else {
        time.checked_sub(magnitude)
    }
    .expect("clock offset moved time out of range")
}

impl crate::Supervisor for Context {
    fn child(&self, label: &'static str) -> Self {
        let (tree, _) = Tree::child(&self.tree);
        Self {
            name: child_label(&self.name, label),
            attributes: self.attributes.clone(),
            executor: self.executor.clone(),
            network: self.network.clone(),
            storage: self.storage.clone(),
            opens: self.opens.clone(),
            network_buffer_pool: self.network_buffer_pool.clone(),
            storage_buffer_pool: self.storage_buffer_pool.clone(),
            tree,
            process: self.process.clone(),
            execution: Execution::default(),
        }
    }

    fn with_attribute(mut self, key: &'static str, value: impl std::fmt::Display) -> Self {
        // Validate label format (must match [a-zA-Z][a-zA-Z0-9_]*)
        validate_label(key);

        // Add the attribute to the list of attributes
        add_attribute(&mut self.attributes, key, value);
        self
    }

    fn name(&self) -> Name {
        Name {
            label: self.name.clone(),
            attributes: self.attributes.clone(),
        }
    }
}

impl crate::Metrics for Context {
    fn register<N: Into<String>, H: Into<String>, M: Metric>(
        &self,
        name: N,
        help: H,
        metric: M,
    ) -> Registered<M> {
        let name = name.into();
        let help = help.into();
        let executor = self.executor();
        executor.auditor.event(b"register", |hasher| {
            hasher.update(name.as_bytes());
            hasher.update(help.as_bytes());
            for (k, v) in &self.attributes {
                hasher.update(k.as_bytes());
                hasher.update(v.as_bytes());
            }
        });
        let metric = Arc::new(metric);
        executor.registry.register(
            prefixed_name(&self.name, &name),
            help,
            self.attributes.clone(),
            metric,
        )
    }

    fn encode(&self) -> String {
        let executor = self.executor();
        executor.auditor.event(b"encode", |_| {});
        executor.registry.encode()
    }
}

struct Sleeper {
    executor: Weak<Executor>,
    time: SystemTime,
    waker: Option<Arc<AtomicWaker>>,
}

impl Sleeper {
    /// Upgrade Weak reference to [Executor].
    fn executor(&self) -> Arc<Executor> {
        self.executor.upgrade().expect("executor already dropped")
    }
}

struct Alarm {
    time: SystemTime,
    waker: Arc<AtomicWaker>,
}

impl PartialEq for Alarm {
    fn eq(&self, other: &Self) -> bool {
        self.time.eq(&other.time)
    }
}

impl Eq for Alarm {}

impl PartialOrd for Alarm {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Alarm {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse the ordering for min-heap
        other.time.cmp(&self.time)
    }
}

impl Future for Sleeper {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
        let executor = self.executor();
        {
            let current_time = *executor.time.lock();
            if current_time >= self.time {
                return Poll::Ready(());
            }
        }
        if let Some(waker) = &self.waker {
            waker.register(cx.waker());
        } else {
            let waker = Arc::new(AtomicWaker::new());
            waker.register(cx.waker());
            executor.sleeping.lock().push(Alarm {
                time: self.time,
                waker: waker.clone(),
            });
            self.waker = Some(waker);
        }
        Poll::Pending
    }
}

impl Clock for Context {
    fn current(&self) -> SystemTime {
        shift(*self.executor().time.lock(), self.clock_offset())
    }

    fn sleep(&self, duration: Duration) -> impl Future<Output = ()> + Send + 'static + use<> {
        let deadline = self
            .current()
            .checked_add(duration)
            .expect("overflow when setting wake time");
        self.sleep_until(deadline)
    }

    fn sleep_until(
        &self,
        deadline: SystemTime,
    ) -> impl Future<Output = ()> + Send + 'static + use<> {
        Sleeper {
            executor: self.executor.clone(),

            time: shift(deadline, -self.clock_offset()),
            waker: None,
        }
    }
}

/// Times network latency with the executor's simulated clock. Holds only a weak reference, so
/// the network never keeps its executor alive.
struct ExecutorTimer(Weak<Executor>);

impl Timer for ExecutorTimer {
    fn sleep(&self, delay: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        let executor = self.0.upgrade().expect("executor already dropped");
        let time = executor
            .time
            .lock()
            .checked_add(delay)
            .expect("overflow when setting wake time");
        Box::pin(Sleeper {
            executor: self.0.clone(),
            time,
            waker: None,
        })
    }
}

/// Crashes the processes that own partitions from within their storage operations. Holds only a
/// weak reference, so storage never keeps its executor alive.
struct ExecutorCrasher(Weak<Executor>);

impl ExecutorCrasher {
    /// The latest started live process that owns `partition`.
    fn owner(&self, partition: &str) -> Option<(Arc<Executor>, Arc<ProcessHost>)> {
        let executor = self.0.upgrade()?;
        let owner = executor
            .processes
            .lock()
            .iter()
            .rev()
            .find(|host| (host.partitions)(partition))
            .cloned()?;
        Some((executor, owner))
    }
}

impl Crasher for ExecutorCrasher {
    fn owns(&self, partition: &str) -> bool {
        self.owner(partition).is_some()
    }

    fn crash(&self, partition: &str) {
        let (executor, owner) = self
            .owner(partition)
            .expect("crashed partition has no owner");
        executor.auditor.event(b"crash_in_storage", |hasher| {
            hasher.update(partition.as_bytes());
        });
        executor.crash_process(&owner);
    }
}

fn install_crasher(storage: &Storage, executor: &Arc<Executor>) {
    let previous = storage
        .inner()
        .inner()
        .crasher()
        .lock()
        .replace(Arc::new(ExecutorCrasher(Arc::downgrade(executor))));
    assert!(previous.is_none(), "storage crasher installed twice");
}

fn install_timer(slot: &TimerSlot, executor: &Arc<Executor>) {
    assert!(
        slot.set(Arc::new(ExecutorTimer(Arc::downgrade(executor))))
            .is_ok(),
        "network timer installed twice"
    );
}

/// A future that resolves when a given target time is reached.
///
/// If the future is not ready at the target time, the future is blocked until the target time is reached.
#[cfg(feature = "external")]
#[pin_project]
struct Waiter<F: Future> {
    sleeper: Sleeper,
    #[pin]
    future: F,
    ready: Option<F::Output>,
    started: bool,
}

#[cfg(feature = "external")]
impl<F> Future for Waiter<F>
where
    F: Future + Send,
{
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();

        // Poll once with a noop waker so the future can register interest or start work
        // without being able to wake this task before the sampled delay expires. Any ready
        // value is cached and only released after the sleeper's deadline.
        if !*this.started {
            *this.started = true;
            let waker = noop_waker();
            let mut cx_noop = task::Context::from_waker(&waker);
            if let Poll::Ready(value) = this.future.as_mut().poll(&mut cx_noop) {
                *this.ready = Some(value);
            }
        }

        // Only allow the task to progress once the sampled delay has elapsed.
        std::task::ready!(Pin::new(this.sleeper).poll(cx));

        // If the underlying future completed during the noop pre-poll, surface the cached value.
        if let Some(value) = this.ready.take() {
            return Poll::Ready(value);
        }

        // Block the current thread until the future reschedules itself, keeping polling
        // deterministic with respect to executor time.
        let blocker = Blocker::new();
        loop {
            let waker = waker(blocker.clone());
            let mut cx_block = task::Context::from_waker(&waker);
            match this.future.as_mut().poll(&mut cx_block) {
                Poll::Ready(value) => {
                    break Poll::Ready(value);
                }
                Poll::Pending => blocker.wait(),
            }
        }
    }
}

#[cfg(feature = "external")]
impl Pacer for Context {
    fn pace<'a, F, T>(&'a self, latency: Duration, future: F) -> impl Future<Output = T> + Send + 'a
    where
        F: Future<Output = T> + Send + 'a,
        T: Send + 'a,
    {
        // Compute target time
        let target = self
            .executor()
            .time
            .lock()
            .checked_add(latency)
            .expect("overflow when setting wake time");

        Waiter {
            sleeper: Sleeper {
                executor: self.executor.clone(),
                time: target,
                waker: None,
            },
            future,
            ready: None,
            started: false,
        }
    }
}

impl GClock for Context {
    type Instant = SystemTime;

    fn now(&self) -> Self::Instant {
        self.current()
    }
}

impl ReasonablyRealtime for Context {}

impl crate::Network for Context {
    type Listener = ListenerOf<Network>;

    async fn bind(&self, socket: SocketAddr) -> Result<Self::Listener, Error> {
        self.network.bind(socket).await
    }

    async fn dial(
        &self,
        socket: SocketAddr,
    ) -> Result<(crate::SinkOf<Self>, crate::StreamOf<Self>), Error> {
        self.network.dial(socket).await
    }
}

impl crate::Resolver for Context {
    async fn resolve(&self, host: &str) -> Result<Vec<IpAddr>, Error> {
        // Get the record
        let executor = self.executor();
        let dns = executor.dns.lock();
        let result = dns.get(host).cloned();
        drop(dns);

        // Update the auditor
        executor.auditor.event(b"resolve", |hasher| {
            hasher.update(host.as_bytes());
            hasher.update(result.encode());
        });
        result.ok_or_else(|| Error::ResolveFailed(host.to_string()))
    }
}

impl TryRng for Context {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        let executor = self.executor();
        executor.auditor.event(b"rand", |hasher| {
            hasher.update(b"next_u32");
        });
        let result = executor.rng.lock().next_u32();
        Ok(result)
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let executor = self.executor();
        executor.auditor.event(b"rand", |hasher| {
            hasher.update(b"next_u64");
        });
        let result = executor.rng.lock().next_u64();
        Ok(result)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
        let executor = self.executor();
        executor.auditor.event(b"rand", |hasher| {
            hasher.update(b"fill_bytes");
        });
        executor.rng.lock().fill_bytes(dest);
        Ok(())
    }
}

impl TryCryptoRng for Context {}

impl crate::Storage for Context {
    type Blob = Blob;

    async fn open_versioned(
        &self,
        partition: &str,
        name: &[u8],
        versions: std::ops::RangeInclusive<BlobVersion>,
    ) -> Result<(Self::Blob, u64, BlobVersion), Error> {
        let opened = self.opens.open(
            partition,
            name,
            self.storage.open_versioned(partition, name, versions),
        )?;
        let retired = self.storage.inner().inner().admit(partition, name);
        let opened = opened.finish();
        drop(retired);
        Ok(opened)
    }

    async fn remove(&self, partition: &str, name: Option<&[u8]>) -> Result<(), Error> {
        let audited = self.storage.inner();
        let retired = self.opens.remove(
            partition,
            name,
            audited.remove_with(
                partition,
                name,
                audited.inner().remove_retired(partition, name),
            ),
        )?;
        drop(retired);
        Ok(())
    }

    async fn scan(&self, partition: &str) -> Result<Vec<Vec<u8>>, Error> {
        self.storage.scan(partition).await
    }
}

impl crate::BufferPooler for Context {
    fn network_buffer_pool(&self) -> &crate::BufferPool {
        &self.network_buffer_pool
    }

    fn storage_buffer_pool(&self) -> &crate::BufferPool {
        &self.storage_buffer_pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "external")]
    use crate::FutureExt;
    use crate::{
        Blob, Metrics as _, ReadOptions, Resolver, Runner as _, Spawner as _, Storage, Strategizer,
        Supervisor as _, WriteOptions, deterministic, reschedule,
    };
    use bytes::Bytes;
    use commonware_macros::test_traced;
    use commonware_parallel::Strategy;
    #[cfg(feature = "external")]
    use commonware_utils::channel::mpsc;
    use commonware_utils::{
        NZUsize, ScriptedRng,
        channel::{mpsc, oneshot},
        probability,
    };
    #[cfg(feature = "external")]
    use futures::StreamExt;
    #[cfg(not(feature = "external"))]
    use futures::future::pending;
    #[cfg(not(feature = "external"))]
    use futures::stream::StreamExt as _;
    use futures::{FutureExt as _, stream::FuturesUnordered};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct ReversePolicy(Arc<Mutex<usize>>);

    impl SchedulingPolicy for ReversePolicy {
        fn order(&mut self, _: SystemTime, ready: &mut [RunnableTask]) {
            *self.0.lock() += 1;
            ready.sort_by_key(|task| std::cmp::Reverse((task.id(), task.occurrence())));
        }
    }

    #[test]
    fn test_controlled_order_and_policy_recovery() {
        let calls = Arc::new(Mutex::new(0));
        let config = Config::new().with_scheduling_policy(ReversePolicy(calls.clone()));
        let (order, checkpoint) = Runner::new(config).start_and_recover(|context| async move {
            let order = Arc::new(Mutex::new(Vec::new()));
            let first_order = order.clone();
            let first = context.child("first").spawn(move |_| async move {
                first_order.lock().push(1);
            });
            let second_order = order.clone();
            let second = context.child("second").spawn(move |_| async move {
                second_order.lock().push(2);
            });
            first.await.unwrap();
            second.await.unwrap();
            order.lock().clone()
        });
        assert_eq!(order, vec![2, 1]);
        let before = *calls.lock();
        Runner::from(checkpoint).start(|context| async move {
            context.sleep(Duration::from_millis(1)).await;
        });
        assert!(*calls.lock() > before);
    }

    struct DuplicatingPolicy;

    impl SchedulingPolicy for DuplicatingPolicy {
        fn order(&mut self, _: SystemTime, ready: &mut [RunnableTask]) {
            if ready.len() > 1 {
                ready[1].id = ready[0].id;
            }
        }
    }

    #[test]
    #[should_panic(expected = "scheduling policy changed the runnable batch")]
    fn test_policy_cannot_change_ready_set() {
        Runner::new(Config::new().with_scheduling_policy(DuplicatingPolicy)).start(
            |context| async move {
                let first = context.child("first").spawn(|_| async {});
                let second = context.child("second").spawn(|_| async {});
                first.await.unwrap();
                second.await.unwrap();
            },
        );
    }

    #[rstest::rstest]
    #[case::open_named(true, true)]
    #[case::open_partition(true, false)]
    #[case::remove_named(false, true)]
    #[case::remove_partition(false, false)]
    fn test_logical_open_namespace_handoff(#[case] open_first: bool, #[case] named: bool) {
        Runner::default().start(|context| async move {
            // Nonempty durable contents distinguish the unlinked incarnation from its replacement.
            let (seed, _) = context.open("partition", b"blob").await.unwrap();
            seed.write_at(0, b"saved", WriteOptions::SYNC)
                .await
                .unwrap();
            drop(seed);

            // Pause between the namespace operation and its registration update to check that
            // a competing operation cannot observe only half of the transaction.
            let name = named.then_some(b"blob".as_slice());
            let worker_context = context.child("namespace");
            let competing_context = context.child("competing");
            let (entered, entering) = std::sync::mpsc::channel();
            let (release, released) = std::sync::mpsc::channel();
            let (observed, observing) = std::sync::mpsc::channel();
            let operation = move |context: Context, open| async move {
                if open {
                    Some(context.open("partition", b"blob").await.unwrap())
                } else {
                    context.remove("partition", name).await.unwrap();
                    None
                }
            };

            // Both operations run on scoped threads so the coordinator can release a held registry.
            // The contender observes the registry only after the worker reaches its handoff.
            let (worker, competing) = std::thread::scope(move |scope| {
                let worker = scope.spawn(move || {
                    worker_context.opens.pause_namespace(entered, released);
                    operation(worker_context, open_first)
                        .now_or_never()
                        .unwrap()
                });
                entering.recv().unwrap();
                let competing = scope.spawn(move || {
                    competing_context.opens.watch_registry(observed);
                    operation(competing_context, !open_first)
                        .now_or_never()
                        .unwrap()
                });

                // A held registry requires releasing its owner before joining the contender.
                // At an unlocked handoff, the contender completes before the owner continues.
                let locked = observing.recv().unwrap();
                if locked {
                    release.send(()).unwrap();
                }
                let competing = competing.join().unwrap();
                if !locked {
                    release.send(()).unwrap();
                }
                (worker.join().unwrap(), competing)
            });

            // Opening before removal preserves access to the old contents. Opening after
            // removal must return a fresh incarnation.
            let (old, current) = if open_first {
                let (old, len) = worker.unwrap();
                assert_eq!(len, 5);
                assert_eq!(
                    old.read_at(0, 5, ReadOptions::default())
                        .await
                        .unwrap()
                        .coalesce(),
                    b"saved"
                );
                let (current, len) = context.open("partition", b"blob").await.unwrap();
                assert_eq!(len, 0);
                (Some(old), current)
            } else {
                let (current, len) = competing.unwrap();
                assert_eq!(len, 0);
                (None, current)
            };

            // An old handle's cleanup cannot release the replacement's logical open.
            let current = Arc::new(current);
            let retained = current.clone();
            drop(old);
            drop(current);
            assert!(matches!(
                context.open("partition", b"blob").await,
                Err(Error::BlobAlreadyOpen(p, n)) if p == "partition" && n == "626c6f62"
            ));

            // Dropping the replacement's last owner permits reopening its durable contents.
            retained
                .write_at(0, b"new", WriteOptions::SYNC)
                .await
                .unwrap();
            drop(retained);
            let (reopened, len) = context.open("partition", b"blob").await.unwrap();
            assert_eq!(len, 3);
            assert_eq!(
                reopened
                    .read_at(0, 3, ReadOptions::default())
                    .await
                    .unwrap()
                    .coalesce(),
                b"new"
            );
        });
    }

    #[rstest::rstest]
    #[case::retained(false)]
    #[case::synced(true)]
    fn test_logical_open_releases_retained_mutations(#[case] sync: bool) {
        // Keep unsynced writes eligible for crash replay without injecting write failures.
        let cfg = Config::default().with_storage_fault_config(FaultConfig::default().write(
            WriteConfig {
                failure_rate: probability!(0.0),
                retention_rate: probability!(1.0),
                mode: PartialWriteMode::Prefix,
            },
        ));
        let (_, checkpoint) = Runner::new(cfg).start_and_recover(|context| async move {
            // The range sync persists only the middle byte of the overwrite. Reopening must
            // discard the unsynced fragments excluded from that durable snapshot.
            let (blob, _) = context.open("partition", b"blob").await.unwrap();
            let blob = Arc::new(blob);
            blob.write_at(0, b"saved", WriteOptions::SYNC)
                .await
                .unwrap();
            blob.write_at(0, b"stale", WriteOptions::default())
                .await
                .unwrap();
            blob.write_at(2, b"X", WriteOptions::SYNC).await.unwrap();
            let retained = blob.clone();
            drop(blob);
            drop(retained);

            // Retained write fragments must not keep a logical open alive.
            let (reopened, len) = context.open("partition", b"blob").await.unwrap();
            let reopened = Arc::new(reopened);
            assert_eq!(len, 5);
            assert_eq!(
                reopened
                    .read_at(0, 5, ReadOptions::default())
                    .await
                    .unwrap()
                    .coalesce(),
                b"saXed"
            );
            let retained = reopened.clone();
            drop(reopened);

            // Duplicate opens contribute to the runtime audit.
            let before = context.auditor().state();
            assert!(matches!(
                context.open("partition", b"blob").await,
                Err(Error::BlobAlreadyOpen(_, _))
            ));
            assert_ne!(before, context.auditor().state());

            // Mutations on the new open must supersede the admitted snapshot.
            if sync {
                retained
                    .write_at(0, b"fresh", WriteOptions::default())
                    .await
                    .unwrap();
                retained.sync().await.unwrap();
            }
        });

        // Crash replay must not restore fragments excluded by the successful reopen.
        Runner::from(checkpoint).start(|context| async move {
            let (blob, len) = context.open("partition", b"blob").await.unwrap();
            assert_eq!(len, 5);
            assert_eq!(
                blob.read_at(0, 5, ReadOptions::default())
                    .await
                    .unwrap()
                    .coalesce(),
                if sync { b"fresh" } else { b"saXed" }
            );
        });
    }

    #[rstest::rstest]
    #[case::remove("remove")]
    #[case::remove_partition("remove_partition")]
    #[case::admit("admit")]
    fn test_retired_write_owner_releases_other_blob(#[case] operation: &'static str) {
        struct Owner<B> {
            data: Vec<u8>,
            _blob: B,
            released: Arc<std::sync::atomic::AtomicBool>,
        }

        impl<B> AsRef<[u8]> for Owner<B> {
            fn as_ref(&self) -> &[u8] {
                &self.data
            }
        }

        impl<B> Drop for Owner<B> {
            fn drop(&mut self) {
                self.released.store(true, Ordering::SeqCst);
            }
        }

        // Retain successful writes so each namespace operation has payload owners to retire.
        let faults = FaultConfig::default().write(WriteConfig {
            failure_rate: probability!(0.0),
            retention_rate: probability!(1.0),
            mode: PartialWriteMode::Prefix,
        });
        Runner::new(Config::default().with_storage_fault_config(faults)).start(
            |context| async move {
                // The retained payload keeps b open, making its destruction re-enter the registry.
                let (a, _) = context.open("partition", b"a").await.unwrap();
                let (b, _) = context.open("partition", b"b").await.unwrap();
                let b = Arc::new(b);
                let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
                let payload = Bytes::from_owner(Owner {
                    data: b"saved".to_vec(),
                    _blob: b.clone(),
                    released: released.clone(),
                });
                a.write_at(0, payload, WriteOptions::default())
                    .await
                    .unwrap();

                // Give partition removal evidence under both names.
                if operation == "remove_partition" {
                    b.write_at(0, b"second", WriteOptions::default())
                        .await
                        .unwrap();
                }
                drop(b);
                assert!(!released.load(Ordering::SeqCst));

                // Retiring a's payload must release the namespace lock before destroying it,
                // so b can release its open.
                if operation == "remove" {
                    context.remove("partition", Some(b"a")).await.unwrap();
                } else if operation == "remove_partition" {
                    context.remove("partition", None).await.unwrap();
                } else {
                    assert_eq!(operation, "admit");
                    drop(a);
                    drop(context.open("partition", b"a").await.unwrap());
                }

                // Both payload destruction and release of b's open must finish before returning.
                assert!(released.load(Ordering::SeqCst));
                drop(context.open("partition", b"b").await.unwrap());
            },
        );
    }

    #[rstest::rstest]
    #[case::sync("sync")]
    #[case::start_sync("start_sync")]
    #[case::overwrite("overwrite")]
    fn test_sync_retirement_progresses_with_namespace_open(#[case] operation: &'static str) {
        /// Signals retirement before releasing another blob's open.
        struct Owner<B> {
            data: Vec<u8>,
            _blob: B,
            retiring: std::sync::mpsc::Sender<()>,
        }

        impl<B> AsRef<[u8]> for Owner<B> {
            fn as_ref(&self) -> &[u8] {
                &self.data
            }
        }

        impl<B> Drop for Owner<B> {
            fn drop(&mut self) {
                self.retiring.send(()).unwrap();
            }
        }

        // Keep the payload until a durability operation retires its write.
        let faults = FaultConfig::default().write(WriteConfig {
            failure_rate: probability!(0.0),
            retention_rate: probability!(1.0),
            mode: PartialWriteMode::Prefix,
        });
        Runner::new(Config::default().with_storage_fault_config(faults)).start(
            |context| async move {
                // Retiring a's write drops the last owner of b and needs the open registry.
                let (a, _) = context.open("partition", b"a").await.unwrap();
                let (b, _) = context.open("partition", b"b").await.unwrap();
                let b = Arc::new(b);
                let (retiring, retired) = std::sync::mpsc::channel();
                a.write_at(
                    0,
                    Bytes::from_owner(Owner {
                        data: b"saved".to_vec(),
                        _blob: b.clone(),
                        retiring,
                    }),
                    WriteOptions::default(),
                )
                .await
                .unwrap();
                drop(b);

                // Hold the registry while opening c, before admission locks the pending mutations.
                let (entered, entering) = std::sync::mpsc::channel();
                let (release, released) = std::sync::mpsc::channel();
                let namespace = context.child("namespace");
                let opener = std::thread::spawn(move || {
                    namespace.opens.pause_namespace(entered, released);
                    drop(
                        namespace
                            .open("partition", b"c")
                            .now_or_never()
                            .unwrap()
                            .unwrap(),
                    );
                });
                entering.recv().unwrap();

                // Retire a's payload concurrently, forcing b's cleanup to wait for the registry.
                let mutator = std::thread::spawn(move || {
                    if operation == "sync" {
                        a.sync().now_or_never().unwrap().unwrap();
                    } else if operation == "start_sync" {
                        a.start_sync()
                            .now_or_never()
                            .unwrap()
                            .now_or_never()
                            .unwrap()
                            .unwrap();
                    } else {
                        assert_eq!(operation, "overwrite");
                        a.write_at(0, b"fresh", WriteOptions::SYNC)
                            .now_or_never()
                            .unwrap()
                            .unwrap();
                    }
                });

                // The signal precedes b's cleanup. Let c continue so both operations can finish
                // only if retirement has released the pending-mutation lock.
                retired.recv().unwrap();
                release.send(()).unwrap();
                mutator.join().unwrap();
                opener.join().unwrap();
                drop(context.open("partition", b"b").await.unwrap());
            },
        );
    }

    async fn task(i: usize) -> usize {
        for _ in 0..5 {
            reschedule().await;
        }
        i
    }

    fn run_tasks(tasks: usize, runner: deterministic::Runner) -> (String, Vec<usize>) {
        runner.start(|context| async move {
            let mut handles = FuturesUnordered::new();
            for i in 0..=tasks - 1 {
                handles.push(context.child("task").spawn(move |_| task(i)));
            }

            let mut outputs = Vec::new();
            while let Some(result) = handles.next().await {
                outputs.push(result.unwrap());
            }
            assert_eq!(outputs.len(), tasks);
            (context.auditor().state(), outputs)
        })
    }

    fn run_with_seed(seed: u64) -> (String, Vec<usize>) {
        let executor = deterministic::Runner::seeded(seed);
        run_tasks(5, executor)
    }

    fn run_with_metric(name: &'static str, help: &'static str) -> String {
        deterministic::Runner::default().start(|context| async move {
            let _: Registered<raw::Counter> = context.register(name, help, raw::Counter::default());
            context.auditor().state()
        })
    }

    #[test]
    fn test_auditor_separates_metric_fields() {
        let state_a = run_with_metric("a", "bc");
        let state_b = run_with_metric("ab", "c");

        assert_ne!(state_a, state_b);
    }

    #[test]
    fn test_same_seed_same_order() {
        // Generate initial outputs
        let mut outputs = Vec::new();
        for seed in 0..1000 {
            let output = run_with_seed(seed);
            outputs.push(output);
        }

        // Ensure they match
        for seed in 0..1000 {
            let output = run_with_seed(seed);
            assert_eq!(output, outputs[seed as usize]);
        }
    }

    #[test_traced("TRACE")]
    fn test_different_seeds_different_order() {
        let output1 = run_with_seed(12345);
        let output2 = run_with_seed(54321);
        assert_ne!(output1, output2);
    }

    #[test]
    fn test_alarm_min_heap() {
        // Populate heap
        let now = SystemTime::now();
        let alarms = vec![
            Alarm {
                time: now + Duration::new(10, 0),
                waker: Arc::new(AtomicWaker::new()),
            },
            Alarm {
                time: now + Duration::new(5, 0),
                waker: Arc::new(AtomicWaker::new()),
            },
            Alarm {
                time: now + Duration::new(15, 0),
                waker: Arc::new(AtomicWaker::new()),
            },
            Alarm {
                time: now + Duration::new(5, 0),
                waker: Arc::new(AtomicWaker::new()),
            },
        ];
        let mut heap = BinaryHeap::new();
        for alarm in alarms {
            heap.push(alarm);
        }

        // Verify min-heap
        let mut sorted_times = Vec::new();
        while let Some(alarm) = heap.pop() {
            sorted_times.push(alarm.time);
        }
        assert_eq!(
            sorted_times,
            vec![
                now + Duration::new(5, 0),
                now + Duration::new(5, 0),
                now + Duration::new(10, 0),
                now + Duration::new(15, 0),
            ]
        );
    }

    #[test]
    fn test_sleep_refreshes_one_alarm_after_repolling() {
        struct Counter(AtomicUsize);

        impl ArcWake for Counter {
            fn wake_by_ref(this: &Arc<Self>) {
                this.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        Runner::default().start(|context| async move {
            // Count notifications separately for the initial and final pollers.
            let first = Arc::new(Counter(AtomicUsize::new(0)));
            let latest = Arc::new(Counter(AtomicUsize::new(0)));
            let mut sleep = context.sleep(Duration::from_millis(10)).boxed();

            // Changing the poller's waker must reuse the existing alarm.
            for counter in [&first, &latest].into_iter().cycle().take(100) {
                assert!(
                    sleep
                        .poll_unpin(&mut task::Context::from_waker(&waker(counter.clone())))
                        .is_pending()
                );
            }
            assert_eq!(context.executor().sleeping.lock().len(), 1);

            // Once due, the sleep must notify only its latest poller and resolve.
            context.sleep(Duration::from_millis(20)).await;
            assert_eq!(first.0.load(Ordering::Relaxed), 0);
            assert_eq!(latest.0.load(Ordering::Relaxed), 1);
            assert!(futures::poll!(&mut sleep).is_ready());
        });
    }

    #[test]
    fn test_dropped_sleeper_before_live_deadline() {
        let executor = deterministic::Runner::default();
        executor.start(|context| async move {
            let (started_sender, started_receiver) = oneshot::channel();
            let sleeper = context.child("sleeper").spawn(|context| async move {
                let mut sleepers = FuturesUnordered::new();
                sleepers.push(context.sleep(Duration::from_secs(1)));
                started_sender.send(()).unwrap();
                sleepers.next().await;
            });

            // Waiting for the signal ensures the child registered its alarm before being aborted.
            started_receiver.await.unwrap();
            sleeper.abort();

            // The stale child alarm must not prevent a later live alarm from firing.
            context.sleep(Duration::from_secs(2)).await;
        });
    }

    #[cfg(not(feature = "external"))]
    #[test]
    #[should_panic(expected = "runtime timeout")]
    fn test_dropped_sleeper_beyond_timeout() {
        let executor = deterministic::Runner::timed(Duration::from_secs(10));
        executor.start(|context| async move {
            let (started_sender, started_receiver) = oneshot::channel();
            let sleeper = context.child("sleeper").spawn(|context| async move {
                let mut sleep = Box::pin(context.sleep(Duration::from_secs(20)));
                assert!(sleep.as_mut().now_or_never().is_none());
                started_sender.send(()).unwrap();
                sleep.await;
            });

            started_receiver.await.unwrap();
            sleeper.abort();
            pending::<()>().await;
        });
    }

    #[test]
    #[should_panic(expected = "runtime timeout")]
    fn test_timeout() {
        let executor = deterministic::Runner::timed(Duration::from_secs(10));
        executor.start(|context| async move {
            loop {
                context.sleep(Duration::from_secs(1)).await;
            }
        });
    }

    #[test]
    #[should_panic(expected = "cycle duration must be non-zero when timeout is set")]
    fn test_bad_timeout() {
        let cfg = Config {
            timeout: Some(Duration::default()),
            cycle: Duration::default(),
            ..Config::default()
        };
        deterministic::Runner::new(cfg);
    }

    #[test]
    #[should_panic(
        expected = "cycle duration must be greater than or equal to system time precision"
    )]
    fn test_bad_cycle() {
        let cfg = Config {
            cycle: SYSTEM_TIME_PRECISION - Duration::from_nanos(1),
            ..Config::default()
        };
        deterministic::Runner::new(cfg);
    }

    /// Removing a blob frees its name while the removed handle lives.
    #[test]
    fn test_removed_blob_reopens_while_handle_alive() {
        deterministic::Runner::default().start(|context| async move {
            let (old, _) = context.open("partition", b"blob").await.unwrap();
            old.write_at(0, b"old", WriteOptions::default())
                .await
                .unwrap();
            context.remove("partition", None).await.unwrap();
            let (current, len) = context.open("partition", b"blob").await.unwrap();
            assert_eq!(len, 0);

            // Cleanup of the removed handle must not affect writes through its replacement.
            drop(old);
            current
                .write_at(0, b"new", WriteOptions::default())
                .await
                .unwrap();
            current.sync().await.unwrap();
            let read = current
                .read_at(0, 3, ReadOptions::default())
                .await
                .unwrap()
                .coalesce();
            assert_eq!(read.as_ref(), b"new");
        });
    }

    #[test]
    fn test_recover_synced_storage_persists() {
        // Initialize the first runtime
        let executor1 = deterministic::Runner::default();
        let partition = "test_partition";
        let name = b"test_blob";
        let data = b"Hello, world!";

        // Run some tasks, sync storage, and recover the runtime
        let (state, checkpoint) = executor1.start_and_recover(|context| async move {
            let (blob, _) = context.open(partition, name).await.unwrap();
            blob.write_at(0, data, WriteOptions::default())
                .await
                .unwrap();
            blob.sync().await.unwrap();
            context.auditor().state()
        });

        // Verify auditor state is the same
        assert_eq!(state, checkpoint.auditor.state());

        // Check that synced storage persists after recovery
        let executor = Runner::from(checkpoint);
        executor.start(|context| async move {
            let (blob, len) = context.open(partition, name).await.unwrap();
            assert_eq!(len, data.len() as u64);
            let read = blob
                .read_at(0, data.len(), ReadOptions::default())
                .await
                .unwrap();
            assert_eq!(read.coalesce(), data);
        });
    }

    #[test]
    #[should_panic(expected = "goodbye")]
    fn test_recover_panic_handling() {
        // Initialize the first runtime
        let executor1 = deterministic::Runner::default();
        let (_, checkpoint) = executor1.start_and_recover(|_| async move {
            reschedule().await;
        });

        // Ensure that panic setting is preserved
        let executor = Runner::from(checkpoint);
        executor.start(|_| async move {
            panic!("goodbye");
        });
    }

    #[test]
    fn test_recover_unsynced_storage_does_not_persist() {
        // Initialize the first runtime
        let executor = deterministic::Runner::default();
        let partition = "test_partition";
        let name = b"test_blob";
        let data = b"Hello, world!";

        // Run some tasks without syncing storage
        let (_, checkpoint) = executor.start_and_recover(|context| async move {
            let (blob, _) = context.open(partition, name).await.unwrap();
            blob.write_at(0, data, WriteOptions::default())
                .await
                .unwrap();
        });

        // Recover the runtime
        let executor = Runner::from(checkpoint);

        // Check that unsynced storage does not persist after recovery
        executor.start(|context| async move {
            let (_, len) = context.open(partition, name).await.unwrap();
            assert_eq!(len, 0);
        });
    }

    #[test]
    fn test_recover_snapshots_fault_configuration() {
        let (stale_config, checkpoint) =
            deterministic::Runner::default().start_and_recover(|context| async move {
                let config = context.storage_fault_config();
                *config.write() = FaultConfig::default().open(probability!(1.0));
                config
            });
        *stale_config.write() = FaultConfig::default();

        deterministic::Runner::from(checkpoint).start(|context| async move {
            assert!(context.open("fault_config", b"blob").await.is_err());
        });
    }

    #[test]
    fn test_recover_retained_successful_resize() {
        let retained_resize = [u64::MAX, 0];
        let cfg = deterministic::Config::default()
            .with_rng(ScriptedRng::new(retained_resize))
            .with_storage_fault_config(FaultConfig::default().resize(ResizeConfig {
                failure_rate: probability!(0.5),
                partial_rate: probability!(0.0),
            }));
        let (_, checkpoint) =
            deterministic::Runner::new(cfg).start_and_recover(|context| async move {
                let (blob, _) = context.open("crash_resize", b"blob").await.unwrap();
                blob.write_at(0, b"abcdefgh", WriteOptions::SYNC)
                    .await
                    .unwrap();
                blob.resize(3).await.unwrap();
            });

        deterministic::Runner::from(checkpoint).start(|context| async move {
            let (blob, len) = context.open("crash_resize", b"blob").await.unwrap();
            assert_eq!(len, 3);
            assert_eq!(
                blob.read_at(0, 3, ReadOptions::default())
                    .await
                    .unwrap()
                    .coalesce(),
                b"abc"
            );
        });
    }

    #[test]
    fn test_recover_random_crash_writes_is_seeded_and_epoch_scoped() {
        const STABLE_LEN: usize = 32;
        const PENDING_LEN: usize = 256;

        fn run(seed: u64) -> (Vec<u8>, Digest) {
            let cfg = deterministic::Config::default()
                .with_seed(seed)
                .with_storage_fault_config(FaultConfig::default().write(WriteConfig {
                    failure_rate: probability!(0.0),
                    retention_rate: probability!(0.5),
                    mode: PartialWriteMode::Subset,
                }));
            let (_, checkpoint) =
                deterministic::Runner::new(cfg).start_and_recover(|context| async move {
                    let (blob, _) = context.open("crash_epoch", b"blob").await.unwrap();
                    blob.write_at(0, vec![0xA5; STABLE_LEN], WriteOptions::default())
                        .await
                        .unwrap();
                    blob.sync().await.unwrap();
                    blob.write_at(
                        STABLE_LEN as u64,
                        vec![0x5A; PENDING_LEN],
                        WriteOptions::default(),
                    )
                    .await
                    .unwrap();
                });

            deterministic::Runner::from(checkpoint).start(|context| async move {
                let (blob, len) = context.open("crash_epoch", b"blob").await.unwrap();
                let mut bytes = vec![0; STABLE_LEN + PENDING_LEN];
                let len = usize::try_from(len).unwrap();
                let durable = blob
                    .read_at(0, len, ReadOptions::default())
                    .await
                    .unwrap()
                    .coalesce();
                bytes[..durable.len()].copy_from_slice(durable.as_ref());
                (bytes, context.storage_audit())
            })
        }

        let first = run(12345);
        let second = run(12345);
        let different = run(54321);
        assert_eq!(first, second);
        assert_ne!(first.0, different.0);
        assert!(first.0[..STABLE_LEN].iter().all(|&byte| byte == 0xA5));
        assert!(first.0[STABLE_LEN..].contains(&0));
        assert!(first.0[STABLE_LEN..].contains(&0x5A));
    }

    #[test]
    fn test_recover_dns_mappings_persist() {
        // Initialize the first runtime
        let executor = deterministic::Runner::default();
        let host = "example.com";
        let addrs = vec![
            IpAddr::V4(std::net::Ipv4Addr::new(192, 168, 1, 1)),
            IpAddr::V4(std::net::Ipv4Addr::new(192, 168, 1, 2)),
        ];

        // Register DNS mapping and recover the runtime
        let (state, checkpoint) = executor.start_and_recover({
            let addrs = addrs.clone();
            |context| async move {
                context.resolver_register(host, Some(addrs));
                context.auditor().state()
            }
        });

        // Verify auditor state is the same
        assert_eq!(state, checkpoint.auditor.state());

        // Check that DNS mappings persist after recovery
        let executor = Runner::from(checkpoint);
        executor.start(move |context| async move {
            let resolved = context.resolve(host).await.unwrap();
            assert_eq!(resolved, addrs);
        });
    }

    #[test]
    fn test_recover_time_persists() {
        // Initialize the first runtime
        let executor = deterministic::Runner::default();
        let duration_to_sleep = Duration::from_secs(10);

        // Sleep for some time and recover the runtime
        let (time_before_recovery, checkpoint) = executor.start_and_recover(|context| async move {
            context.sleep(duration_to_sleep).await;
            context.current()
        });

        // Check that the time advanced correctly before recovery
        assert_eq!(
            time_before_recovery.duration_since(UNIX_EPOCH).unwrap(),
            duration_to_sleep
        );

        // Check that the time persists after recovery
        let executor2 = Runner::from(checkpoint);
        executor2.start(move |context| async move {
            assert_eq!(context.current(), time_before_recovery);

            // Advance time further
            context.sleep(duration_to_sleep).await;
            assert_eq!(
                context.current().duration_since(UNIX_EPOCH).unwrap(),
                duration_to_sleep * 2
            );
        });
    }

    #[test]
    #[should_panic(expected = "executor still has weak references")]
    fn test_context_return() {
        // Initialize runtime
        let executor = deterministic::Runner::default();

        // Start runtime
        let context = executor.start(|context| async move {
            // Attempt to recover before the runtime has finished
            context
        });

        // Should never get this far
        drop(context);
    }

    #[test]
    fn test_default_time_zero() {
        // Initialize runtime
        let executor = deterministic::Runner::default();
        executor.start(|context| async move {
            // Check that the time is zero
            assert_eq!(
                context.current().duration_since(UNIX_EPOCH).unwrap(),
                Duration::ZERO
            );
        });
    }

    #[test]
    fn test_start_time() {
        // Initialize runtime with default config
        let executor_default = deterministic::Runner::default();
        executor_default.start(|context| async move {
            assert_eq!(context.current(), UNIX_EPOCH);
        });

        // Initialize runtime with custom start time
        let start_time = UNIX_EPOCH + Duration::from_secs(100);
        let cfg = Config::default().with_start_time(start_time);
        let executor = deterministic::Runner::new(cfg);
        executor.start(move |context| async move {
            // Check that the time matches the custom start time
            assert_eq!(context.current(), start_time);
        });
    }

    #[test]
    #[should_panic(expected = "start time must be greater than or equal to unix epoch")]
    fn test_bad_start_time() {
        let cfg = Config::default().with_start_time(UNIX_EPOCH - Duration::from_secs(1));
        deterministic::Runner::new(cfg);
    }

    #[cfg(not(feature = "external"))]
    #[test]
    #[should_panic(expected = "runtime stalled")]
    fn test_stall() {
        // Initialize runtime
        let executor = deterministic::Runner::default();

        // Start runtime
        executor.start(|_| async move {
            pending::<()>().await;
        });
    }

    #[cfg(not(feature = "external"))]
    #[test]
    #[should_panic(expected = "runtime stalled")]
    fn test_external_simulated() {
        // Initialize runtime
        let executor = deterministic::Runner::default();

        // Create a thread that waits for 1 second
        let (tx, rx) = oneshot::channel();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(1));
            tx.send(()).unwrap();
        });

        // Start runtime
        executor.start(|_| async move {
            rx.await.unwrap();
        });
    }

    #[cfg(feature = "external")]
    #[test]
    fn test_paced_future_moves_between_tasks() {
        Runner::timed(Duration::from_secs(1)).start(|context| async move {
            // An immediately ready payload makes the pacing timer the only wakeup source.
            let clock = context.child("clock");
            let latency = Duration::from_millis(10);
            let deadline = context.current() + latency;
            let mut future = async move { async { 7 }.pace(&clock, latency).await }.boxed();

            // Register the source task's waker before transferring the pending future.
            // The destination must receive the timer's eventual notification.
            assert!(futures::poll!(&mut future).is_pending());
            let moved = context.child("moved").spawn(move |_| future);

            // Changing tasks must preserve both the result and the pacing deadline.
            assert_eq!(moved.await.unwrap(), 7);
            assert!(context.current() >= deadline);
        });
    }

    #[cfg(feature = "external")]
    #[test]
    fn test_external_realtime() {
        // Initialize runtime
        let executor = deterministic::Runner::default();

        // Create a thread that waits for 1 second
        let (tx, rx) = oneshot::channel();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(1));
            tx.send(()).unwrap();
        });

        // Start runtime
        executor.start(|_| async move {
            rx.await.unwrap();
        });
    }

    #[cfg(feature = "external")]
    #[test]
    fn test_external_realtime_variable() {
        // Initialize runtime
        let executor = deterministic::Runner::default();

        // Start runtime
        executor.start(|context| async move {
            // Initialize test
            let start_real = SystemTime::now();
            let start_sim = context.current();
            let (first_tx, first_rx) = oneshot::channel();
            let (second_tx, second_rx) = oneshot::channel();
            let (results_tx, mut results_rx) = mpsc::channel(2);

            // Create a thread that waits for 1 second
            let first_wait = Duration::from_secs(1);
            std::thread::spawn(move || {
                std::thread::sleep(first_wait);
                first_tx.send(()).unwrap();
            });

            // Create a thread
            std::thread::spawn(move || {
                std::thread::sleep(Duration::ZERO);
                second_tx.send(()).unwrap();
            });

            // Wait for a delay sampled before the external send occurs
            let first = context.child("sample_before_send").spawn({
                let results_tx = results_tx.clone();
                move |context| async move {
                    first_rx.pace(&context, Duration::ZERO).await.unwrap();
                    let elapsed_real = SystemTime::now().duration_since(start_real).unwrap();
                    assert!(elapsed_real > first_wait);
                    let elapsed_sim = context.current().duration_since(start_sim).unwrap();
                    assert!(elapsed_sim < first_wait);
                    results_tx.send(1).await.unwrap();
                }
            });

            // Wait for a delay sampled after the external send occurs
            let second = context
                .child("sample_after_send")
                .spawn(move |context| async move {
                    second_rx.pace(&context, first_wait).await.unwrap();
                    let elapsed_real = SystemTime::now().duration_since(start_real).unwrap();
                    assert!(elapsed_real >= first_wait);
                    let elapsed_sim = context.current().duration_since(start_sim).unwrap();
                    assert!(elapsed_sim >= first_wait);
                    results_tx.send(2).await.unwrap();
                });

            // Wait for both tasks to complete
            second.await.unwrap();
            first.await.unwrap();

            // Ensure order is correct
            let mut results = Vec::new();
            for _ in 0..2 {
                results.push(results_rx.recv().await.unwrap());
            }
            assert_eq!(results, vec![1, 2]);
        });
    }

    #[cfg(not(feature = "external"))]
    #[test]
    fn test_simulated_skip() {
        // Initialize runtime
        let executor = deterministic::Runner::default();

        // Start runtime
        executor.start(|context| async move {
            context.sleep(Duration::from_secs(1)).await;

            // Check if we skipped
            let metrics = context.encode();
            let iterations = metrics
                .lines()
                .find_map(|line| {
                    line.strip_prefix("runtime_iterations_total ")
                        .and_then(|value| value.trim().parse::<u64>().ok())
                })
                .expect("missing runtime_iterations_total metric");
            assert!(iterations < 10);
        });
    }

    #[cfg(feature = "external")]
    #[test]
    fn test_realtime_no_skip() {
        // Initialize runtime
        let executor = deterministic::Runner::default();

        // Start runtime
        executor.start(|context| async move {
            context.sleep(Duration::from_secs(1)).await;

            // Check if we skipped
            let metrics = context.encode();
            let iterations = metrics
                .lines()
                .find_map(|line| {
                    line.strip_prefix("runtime_iterations_total ")
                        .and_then(|value| value.trim().parse::<u64>().ok())
                })
                .expect("missing runtime_iterations_total metric");
            assert!(iterations > 500);
        });
    }

    #[test]
    #[should_panic(expected = "label must start with [a-zA-Z]")]
    fn test_metrics_label_empty() {
        let executor = deterministic::Runner::default();
        executor.start(|context| async move {
            let _ = context.child("");
        });
    }

    #[test]
    #[should_panic(expected = "label must start with [a-zA-Z]")]
    fn test_metrics_label_invalid_first_char() {
        let executor = deterministic::Runner::default();
        executor.start(|context| async move {
            let _ = context.child("1invalid");
        });
    }

    #[test]
    #[should_panic(expected = "label must only contain [a-zA-Z0-9_]")]
    fn test_metrics_label_invalid_char() {
        let executor = deterministic::Runner::default();
        executor.start(|context| async move {
            let _ = context.child("invalid-label");
        });
    }

    #[test]
    #[should_panic(expected = "using runtime label is not allowed")]
    fn test_metrics_label_reserved_prefix() {
        let executor = deterministic::Runner::default();
        executor.start(|context| async move {
            let _ = context.child(METRICS_PREFIX);
        });
    }

    #[test]
    fn test_metrics_duplicate_attribute_overwrites() {
        let executor = deterministic::Runner::default();
        executor.start(|context| async move {
            let context = context
                .child("test")
                .with_attribute("epoch", "old")
                .with_attribute("epoch", "new");
            assert_eq!(
                context.name().attributes,
                vec![("epoch".to_string(), "new".to_string())]
            );
        });
    }

    #[test]
    fn test_storage_fault_injection_and_recovery() {
        // Phase 1: Run with 100% sync failure rate
        let cfg = deterministic::Config::default().with_storage_fault_config(FaultConfig {
            sync_rate: Some(probability!(1.0)),
            ..Default::default()
        });

        let (result, checkpoint) =
            deterministic::Runner::new(cfg).start_and_recover(|ctx| async move {
                let (blob, _) = ctx.open("test_fault", b"blob").await.unwrap();
                blob.write_at(0, b"data".to_vec(), WriteOptions::default())
                    .await
                    .unwrap();
                blob.sync().await // This should fail due to fault injection
            });

        // Verify sync failed
        assert!(result.is_err());

        // Phase 2: Recover and disable faults explicitly
        deterministic::Runner::from(checkpoint).start(|ctx| async move {
            // Explicitly disable faults for recovery verification
            *ctx.storage_fault_config().write() = FaultConfig::default();

            // Data was not synced, so blob should be empty (unsynced writes are lost)
            let (blob, len) = ctx.open("test_fault", b"blob").await.unwrap();
            assert_eq!(len, 0, "unsynced data should be lost after recovery");

            // Now we can write and sync successfully
            blob.write_at(0, b"recovered".to_vec(), WriteOptions::default())
                .await
                .unwrap();
            blob.sync()
                .await
                .expect("sync should succeed with faults disabled");

            // Verify data persisted
            let read_buf = blob.read_at(0, 9, ReadOptions::default()).await.unwrap();
            assert_eq!(read_buf.coalesce(), b"recovered");
        });
    }

    /// Delays the first send on every link, resets a link's third send, and refuses dials to
    /// one port.
    struct ScriptedNetwork;

    impl NetworkPolicy for ScriptedNetwork {
        fn connects(&self, _: &SocketAddr, listener: &SocketAddr) -> bool {
            listener.port() != 4000
        }

        fn delivers(&self, transmission: &NetworkTransmission<'_>) -> NetworkDelivery {
            match transmission.index {
                0 => NetworkDelivery::after(Duration::from_millis(250)),
                2 => NetworkDelivery::RESET,
                _ => NetworkDelivery::NOW,
            }
        }
    }

    /// Corrupts a link's first send, duplicates its second, and drops its third.
    struct GarbledNetwork;

    impl NetworkPolicy for GarbledNetwork {
        fn delivers(&self, transmission: &NetworkTransmission<'_>) -> NetworkDelivery {
            match transmission.index {
                0 => NetworkDelivery::NOW.corrupted(8 * 2 + 5),
                1 => NetworkDelivery::NOW.duplicated(Duration::from_millis(10)),
                2 => NetworkDelivery::DROP,
                _ => NetworkDelivery::NOW,
            }
        }
    }

    #[test]
    fn test_network_policy_corrupts_duplicates_and_drops() {
        use crate::{Listener as _, Network as _, Sink as _, Stream as _};
        let cfg = deterministic::Config::default().with_network_policy(Arc::new(GarbledNetwork));
        deterministic::Runner::new(cfg).start(|ctx| async move {
            let address = SocketAddr::from(([127, 0, 0, 1], 4003));
            let mut listener = ctx.bind(address).await.unwrap();
            let (mut sink, _stream) = ctx.dial(address).await.unwrap();
            let (_, _, mut stream) = listener.accept().await.unwrap();
            sink.send(b"abcd".to_vec()).await.unwrap();
            let start = ctx.current();
            sink.send(b"xy".to_vec()).await.unwrap();
            assert_eq!(
                ctx.current().duration_since(start).unwrap(),
                Duration::from_millis(10)
            );
            // Bit 5 of byte 2 flipped, then the duplicated send twice.
            assert_eq!(stream.recv(8).await.unwrap().coalesce(), b"abCdxyxy");
            // A stream cannot lose bytes without breaking: the drop resets the connection.
            assert!(matches!(sink.send(b"z".to_vec()).await, Err(Error::Closed)));
            assert!(stream.recv(1).await.is_err());
        });
    }

    #[test]
    fn test_network_policy_delays_resets_and_refuses() {
        use crate::{Listener as _, Network as _, Sink as _, Stream as _};
        let cfg = deterministic::Config::default().with_network_policy(Arc::new(ScriptedNetwork));
        deterministic::Runner::new(cfg).start(|ctx| async move {
            let refused = SocketAddr::from(([127, 0, 0, 1], 4000));
            let _refusing = ctx.bind(refused).await.unwrap();
            assert!(matches!(
                ctx.dial(refused).await,
                Err(Error::ConnectionFailed)
            ));

            let address = SocketAddr::from(([127, 0, 0, 1], 4001));
            let mut listener = ctx.bind(address).await.unwrap();
            let (mut sink, _stream) = ctx.dial(address).await.unwrap();
            let (_, _, mut stream) = listener.accept().await.unwrap();
            let start = ctx.current();
            sink.send(b"first".to_vec()).await.unwrap();
            assert_eq!(
                ctx.current().duration_since(start).unwrap(),
                Duration::from_millis(250)
            );
            sink.send(b"second".to_vec()).await.unwrap();
            assert_eq!(stream.recv(11).await.unwrap().coalesce(), b"firstsecond");
            assert!(matches!(
                sink.send(b"third".to_vec()).await,
                Err(Error::Closed)
            ));
            assert!(matches!(
                sink.send(b"fourth".to_vec()).await,
                Err(Error::Closed)
            ));
            assert!(
                stream.recv(1).await.is_err(),
                "a reset closes the peer's stream"
            );
        });
    }

    #[test]
    fn test_process_crash_stops_tasks_immediately() {
        // A task woken in the same tick as its process's crash must not run, whatever order the
        // runnable batch is polled in.
        for seed in 0..64 {
            deterministic::Runner::seeded(seed).start(|ctx| async move {
                let ran = Arc::new(AtomicUsize::new(0));
                let child_ran = ran.clone();
                let (wake, woken) = oneshot::channel::<()>();
                let (process_ctx, process) = ctx.process("node", |_| false);
                process_ctx.spawn(move |ctx| async move {
                    ctx.child("child").spawn(move |_| async move {
                        let _ = woken.await;
                        child_ran.fetch_add(1, Ordering::SeqCst);
                    });
                    futures::future::pending::<()>().await;
                });
                ctx.sleep(Duration::from_millis(10)).await;
                process.crash();
                let _ = wake.send(());
                ctx.sleep(Duration::from_millis(10)).await;
                assert_eq!(ran.load(Ordering::SeqCst), 0, "seed {seed}");
            });
        }
    }

    #[test]
    fn test_dropped_listener_frees_its_port() {
        use crate::Network as _;
        deterministic::Runner::default().start(|ctx| async move {
            let address = SocketAddr::from(([127, 0, 0, 1], 4002));
            let listener = ctx.bind(address).await.unwrap();
            assert!(matches!(ctx.bind(address).await, Err(Error::BindFailed)));
            drop(listener);
            assert!(matches!(
                ctx.dial(address).await,
                Err(Error::ConnectionFailed)
            ));
            let _rebound = ctx.bind(address).await.unwrap();
            ctx.dial(address).await.unwrap();
        });
    }

    /// Records every storage fault decision and fails exactly the syncs of one partition.
    #[derive(Default)]
    struct FailPartitionSyncs {
        #[allow(clippy::type_complexity)]
        seen: Mutex<Vec<(String, Option<Vec<u8>>, StorageOp, FaultDraw)>>,
    }

    impl FaultPolicy for FailPartitionSyncs {
        fn occurs(&self, decision: &FaultDecision<'_>, _: commonware_utils::Probability) -> bool {
            self.seen.lock().push((
                decision.partition.to_string(),
                decision.name.map(<[u8]>::to_vec),
                decision.op,
                decision.draw,
            ));
            decision.op == StorageOp::Sync && decision.partition == "faulty"
        }

        fn between(&self, _: &FaultDecision<'_>, range: std::ops::Range<u64>) -> u64 {
            range.start
        }
    }

    #[test]
    fn test_storage_fault_policy_decides_each_fault() {
        let policy = Arc::new(FailPartitionSyncs::default());
        let cfg = deterministic::Config::default()
            .with_storage_fault_config(FaultConfig::default().sync(probability!(0.5)))
            .with_storage_fault_policy(policy.clone());
        deterministic::Runner::new(cfg).start(|ctx| async move {
            for (partition, fails) in [("faulty", true), ("healthy", false)] {
                let (blob, _) = ctx.open(partition, b"blob").await.unwrap();
                blob.write_at(0, b"data".to_vec(), WriteOptions::default())
                    .await
                    .unwrap();
                assert_eq!(blob.sync().await.is_err(), fails, "{partition}");
            }
        });
        let seen = policy.seen.lock();
        let syncs: Vec<_> = seen
            .iter()
            .filter(|(_, _, op, _)| *op == StorageOp::Sync)
            .map(|(partition, name, _, draw)| (partition.as_str(), name.as_deref(), *draw))
            .collect();
        assert_eq!(
            syncs,
            [
                ("faulty", Some(&b"blob"[..]), FaultDraw::Fail),
                ("healthy", Some(&b"blob"[..]), FaultDraw::Fail)
            ]
        );
    }

    /// Spawns a task in `ctx` that counts every millisecond tick it observes.
    fn ticker(ctx: Context) -> Arc<AtomicUsize> {
        let ticks = Arc::new(AtomicUsize::new(0));
        let counter = ticks.clone();
        ctx.spawn(move |ctx| async move {
            loop {
                counter.fetch_add(1, Ordering::SeqCst);
                ctx.sleep(Duration::from_millis(1)).await;
            }
        });
        ticks
    }

    #[test]
    fn test_process_pause_holds_tasks_until_resume() {
        for seed in 0..32 {
            deterministic::Runner::seeded(seed).start(|ctx| async move {
                let (process_ctx, process) = ctx.process("node", |_| false);
                let ticks = ticker(process_ctx.child("ticker"));
                let (sender, mut receiver) = mpsc::unbounded_channel::<u64>();
                let received = Arc::new(Mutex::new(Vec::new()));
                let log = received.clone();
                process_ctx.child("receiver").spawn(move |ctx| async move {
                    while let Some(sent) = receiver.recv().await {
                        log.lock().push((sent, ctx.current()));
                    }
                });
                let peer_ticks = ticker(ctx.child("peer"));
                ctx.sleep(Duration::from_millis(10)).await;

                process.pause();
                let paused_at = ctx.current();
                let frozen = ticks.load(Ordering::SeqCst);
                let peer_before = peer_ticks.load(Ordering::SeqCst);
                sender.send(1).unwrap();
                ctx.sleep(Duration::from_millis(50)).await;
                assert_eq!(ticks.load(Ordering::SeqCst), frozen, "paused tasks ran");
                assert!(received.lock().is_empty(), "paused task received a message");
                assert!(
                    peer_ticks.load(Ordering::SeqCst) >= peer_before + 45,
                    "peer stalled"
                );

                process.resume();
                ctx.sleep(Duration::from_millis(5)).await;
                // The buffered message and the overdue timer are both observed after resuming.
                let received = received.lock().clone();
                assert_eq!(received.len(), 1);
                assert!(received[0].1 >= paused_at + Duration::from_millis(50));
                ctx.sleep(Duration::from_millis(10)).await;
                assert!(
                    ticks.load(Ordering::SeqCst) > frozen + 5,
                    "resumed tasks did not run"
                );
            });
        }
    }

    #[test]
    fn test_process_pause_covers_nested_processes() {
        deterministic::Runner::default().start(|ctx| async move {
            let (outer_ctx, outer) = ctx.process("host", |_| false);
            let (inner_ctx, _inner) = outer_ctx.process("node", |_| false);
            let ticks = ticker(inner_ctx);
            ctx.sleep(Duration::from_millis(5)).await;
            outer.pause();
            let frozen = ticks.load(Ordering::SeqCst);
            ctx.sleep(Duration::from_millis(20)).await;
            assert_eq!(ticks.load(Ordering::SeqCst), frozen);
            outer.resume();
            ctx.sleep(Duration::from_millis(5)).await;
            assert!(ticks.load(Ordering::SeqCst) > frozen);
        });
    }

    #[test]
    fn test_process_crash_while_paused_drops_tasks() {
        deterministic::Runner::default().start(|ctx| async move {
            let (process_ctx, process) = ctx.process("node", |_| false);
            let ticks = ticker(process_ctx);
            ctx.sleep(Duration::from_millis(5)).await;
            process.pause();
            let frozen = ticks.load(Ordering::SeqCst);
            ctx.sleep(Duration::from_millis(5)).await;
            process.crash();
            ctx.sleep(Duration::from_millis(20)).await;
            assert_eq!(ticks.load(Ordering::SeqCst), frozen, "crashed task ran");
            // The crashed task was dropped rather than left parked.
            assert_eq!(Arc::strong_count(&ticks), 1);
        });
    }

    #[test]
    fn test_process_clock_offset() {
        deterministic::Runner::default().start(|ctx| async move {
            let (process_ctx, process) = ctx.process("node", |_| false);
            let start = ctx.current();
            assert_eq!(process_ctx.current(), start);

            // Jump ahead: only the process's clock moves.
            process.set_clock_offset(ClockOffset::Ahead(Duration::from_secs(100)));
            assert_eq!(process_ctx.current(), start + Duration::from_secs(100));
            assert_eq!(ctx.current(), start);
            let child = process_ctx.child("child");
            assert_eq!(child.current(), start + Duration::from_secs(100));

            // A deadline on the process's clock fires when its clock reaches it.
            let deadline = process_ctx.current() + Duration::from_millis(10);
            process_ctx.sleep_until(deadline).await;
            assert_eq!(process_ctx.current(), deadline);
            assert_eq!(ctx.current(), start + Duration::from_millis(10));

            // Durations are unaffected by the offset.
            let before = ctx.current();
            process_ctx.sleep(Duration::from_millis(10)).await;
            assert_eq!(ctx.current(), before + Duration::from_millis(10));

            // Jump behind: the process observes its clock going backwards.
            let ahead = process_ctx.current();
            process.set_clock_offset(ClockOffset::Behind(Duration::from_secs(5)));
            assert!(process_ctx.current() < ahead);
            assert_eq!(
                process_ctx.current(),
                ctx.current() - Duration::from_secs(5)
            );

            // Nested processes add their offsets.
            let (inner_ctx, inner) = process_ctx.process("inner", |_| false);
            inner.set_clock_offset(ClockOffset::Ahead(Duration::from_secs(2)));
            assert_eq!(inner_ctx.current(), ctx.current() - Duration::from_secs(3));
        });
    }

    #[test]
    fn test_process_faults_are_deterministic() {
        let run = |seed| {
            deterministic::Runner::seeded(seed).start(|ctx| async move {
                let (process_ctx, process) = ctx.process("node", |_| false);
                let ticks = ticker(process_ctx.child("ticker"));
                let peer = ticker(ctx.child("peer"));
                for round in 0..10u64 {
                    ctx.sleep(Duration::from_millis(3)).await;
                    if round % 2 == 0 {
                        process.pause();
                    } else {
                        process.resume();
                    }
                    let offset = Duration::from_millis(round * 7);
                    process.set_clock_offset(if round % 3 == 0 {
                        ClockOffset::Behind(offset)
                    } else {
                        ClockOffset::Ahead(offset)
                    });
                }
                (
                    ticks.load(Ordering::SeqCst),
                    peer.load(Ordering::SeqCst),
                    ctx.auditor().state(),
                )
            })
        };
        assert_eq!(run(7), run(7));
    }

    /// Crashes the `crash_at`-th crash decision (counting from zero) and keeps the first `keep`
    /// bytes of every write a crash resolves.
    struct CrashAt {
        crash_at: usize,
        keep: usize,
        crashes: Mutex<usize>,
    }

    impl CrashAt {
        fn new(crash_at: usize, keep: usize) -> Arc<Self> {
            Arc::new(Self {
                crash_at,
                keep,
                crashes: Mutex::new(0),
            })
        }
    }

    impl FaultPolicy for CrashAt {
        fn occurs(&self, decision: &FaultDecision<'_>, _: commonware_utils::Probability) -> bool {
            match decision.draw {
                FaultDraw::Crash => {
                    let mut crashes = self.crashes.lock();
                    let crash = *crashes == self.crash_at;
                    *crashes += 1;
                    crash
                }
                FaultDraw::RetainByte { index } => index < self.keep,
                _ => false,
            }
        }

        fn between(&self, _: &FaultDecision<'_>, range: std::ops::Range<u64>) -> u64 {
            range.start
        }
    }

    fn crash_config(policy: Arc<CrashAt>) -> deterministic::Config {
        deterministic::Config::default()
            .with_storage_fault_config(
                FaultConfig::default()
                    .write(WriteConfig {
                        failure_rate: probability!(0.0),
                        retention_rate: probability!(0.5),
                        mode: PartialWriteMode::Prefix,
                    })
                    .crash(probability!(0.5)),
            )
            .with_storage_fault_policy(policy)
    }

    #[test]
    fn test_crash_in_write_tears_it_and_stops_the_process() {
        let policy = CrashAt::new(1, 3);
        deterministic::Runner::new(crash_config(policy.clone())).start(|ctx| async move {
            let (node, process) = ctx.process("node", |partition| partition == "node");
            let (peer, _peer_process) = ctx.process("peer", |partition| partition == "peer");
            let after = Arc::new(AtomicUsize::new(0));
            let reached = after.clone();
            let sibling = ticker(node.child("sibling"));
            let peer_ticks = ticker(peer.child("ticker"));
            let writer = node.child("writer").spawn(move |ctx| async move {
                let (blob, _) = ctx.open("node", b"journal").await.unwrap();
                blob.write_at(0, b"durable".to_vec(), WriteOptions::SYNC)
                    .await
                    .unwrap();
                // Crashes the process from within this synced write.
                blob.write_at(7, b"pending".to_vec(), WriteOptions::SYNC)
                    .await
                    .unwrap();
                reached.fetch_add(1, Ordering::SeqCst);
            });
            assert!(writer.await.is_err(), "the crashed writer completed");
            assert!(process.crashed());
            assert_eq!(after.load(Ordering::SeqCst), 0, "code after the crash ran");
            let frozen = sibling.load(Ordering::SeqCst);
            let peer_before = peer_ticks.load(Ordering::SeqCst);
            ctx.sleep(Duration::from_millis(10)).await;
            assert_eq!(sibling.load(Ordering::SeqCst), frozen, "a crashed task ran");
            assert!(
                peer_ticks.load(Ordering::SeqCst) > peer_before,
                "the peer stopped"
            );

            // The crashed write was resolved as unsynced: only its first 3 bytes survive.
            let (blob, len) = ctx.open("node", b"journal").await.unwrap();
            assert_eq!(len, 10);
            let read = blob.read_at(0, 10, ReadOptions::default()).await.unwrap();
            assert_eq!(read.coalesce(), b"durablepen");

            // Crashing an already crashed process does nothing.
            process.crash();
        });
        assert_eq!(*policy.crashes.lock(), 2);
    }

    #[test]
    fn test_crash_in_sync_loses_unsynced_writes() {
        // Decisions: write, sync, write, then the crashing sync.
        deterministic::Runner::new(crash_config(CrashAt::new(3, 0))).start(|ctx| async move {
            let (node, process) = ctx.process("node", |partition| partition == "node");
            let writer = node.spawn(move |ctx| async move {
                let (blob, _) = ctx.open("node", b"journal").await.unwrap();
                blob.write_at(0, b"durable".to_vec(), WriteOptions::default())
                    .await
                    .unwrap();
                blob.sync().await.unwrap();
                blob.write_at(7, b"pending".to_vec(), WriteOptions::default())
                    .await
                    .unwrap();
                // Crashes the process instead of syncing.
                blob.sync().await.unwrap();
            });
            assert!(writer.await.is_err());
            assert!(process.crashed());
            let (_, len) = ctx.open("node", b"journal").await.unwrap();
            assert_eq!(len, 7);
        });
    }

    #[test]
    fn test_crash_in_storage_stalls_the_process_other_operations() {
        // Decisions: the writes to `a` and `b`, then the crashing sync of `a`.
        deterministic::Runner::new(crash_config(CrashAt::new(2, 0))).start(|ctx| async move {
            let (node, process) = ctx.process("node", |partition| partition == "node");
            let other = Arc::new(Mutex::new(None));
            let observed = other.clone();
            let writer = node.spawn(move |ctx| async move {
                let (a, _) = ctx.open("node", b"a").await.unwrap();
                let (b, _) = ctx.open("node", b"b").await.unwrap();
                a.write_at(0, b"a".to_vec(), WriteOptions::default())
                    .await
                    .unwrap();
                b.write_at(0, b"b".to_vec(), WriteOptions::default())
                    .await
                    .unwrap();
                let b_sync = async {
                    let result = b.sync().await;
                    *observed.lock() = Some(result.is_ok());
                };
                let _ = futures::join!(a.sync(), b_sync);
            });
            assert!(writer.await.is_err());
            assert!(process.crashed());
            // The sibling operation neither succeeded nor failed: it never completed.
            assert_eq!(*other.lock(), None);
        });
    }

    #[test]
    fn test_crash_rate_ignores_unowned_partitions() {
        let policy = CrashAt::new(0, 0);
        deterministic::Runner::new(crash_config(policy.clone())).start(|ctx| async move {
            let (_node, process) = ctx.process("node", |partition| partition == "node");
            let (blob, _) = ctx.open("other", b"journal").await.unwrap();
            blob.write_at(0, b"data".to_vec(), WriteOptions::SYNC)
                .await
                .unwrap();
            blob.sync().await.unwrap();
            assert!(!process.crashed());
        });
        assert_eq!(
            *policy.crashes.lock(),
            0,
            "drew a crash for an unowned partition"
        );
    }

    #[test]
    fn test_crash_in_storage_restarts_from_surviving_contents() {
        // A restarted process owning the same partitions can be crashed from storage again,
        // and its predecessor no longer owns them.
        deterministic::Runner::new(crash_config(CrashAt::new(1, 0))).start(|ctx| async move {
            for incarnation in 0..2u8 {
                let (node, process) = ctx.process("node", |partition| partition == "node");
                let writer = node.spawn(move |ctx| async move {
                    let (blob, len) = ctx.open("node", b"journal").await.unwrap();
                    blob.write_at(len, vec![incarnation], WriteOptions::SYNC)
                        .await
                        .unwrap();
                    blob.write_at(len + 1, vec![incarnation], WriteOptions::SYNC)
                        .await
                        .unwrap();
                });
                let result = writer.await;
                assert_eq!(result.is_err(), incarnation == 0);
                assert_eq!(process.crashed(), incarnation == 0);
            }
            let (_, len) = ctx.open("node", b"journal").await.unwrap();
            assert_eq!(len, 3);
        });
    }

    #[test]
    fn test_process_crash_crashes_only_its_partitions() {
        deterministic::Runner::default().start(|ctx| async move {
            let (_, process) = ctx.process("node0", |partition| partition == "node0");
            for partition in ["node0", "node1"] {
                let (blob, _) = ctx.open(partition, b"blob").await.unwrap();
                blob.write_at(0, b"durable".to_vec(), WriteOptions::default())
                    .await
                    .unwrap();
                blob.sync().await.unwrap();
                blob.write_at(0, b"pending".to_vec(), WriteOptions::default())
                    .await
                    .unwrap();
            }
            let (survivor, _) = ctx.open("node1", b"other").await.unwrap();
            survivor
                .write_at(0, b"live".to_vec(), WriteOptions::default())
                .await
                .unwrap();

            process.crash();

            // The crashed partition keeps only its synced bytes.
            let (blob, len) = ctx.open("node0", b"blob").await.unwrap();
            assert_eq!(len, 7);
            let read = blob.read_at(0, 7, ReadOptions::default()).await.unwrap();
            assert_eq!(read.coalesce(), b"durable");
            // Other partitions keep their live handles and unsynced writes.
            survivor.sync().await.unwrap();
            assert_eq!(
                ctx.logical_blob("node1", b"other").as_deref(),
                Some(&b"live"[..])
            );
        });
    }

    #[test]
    fn test_storage_fault_dynamic_config() {
        let executor = deterministic::Runner::default();
        executor.start(|ctx| async move {
            let (blob, _) = ctx.open("test_dynamic", b"blob").await.unwrap();

            // Initially no faults - sync should succeed
            blob.write_at(0, b"initial".to_vec(), WriteOptions::default())
                .await
                .unwrap();
            blob.sync().await.expect("initial sync should succeed");

            // Enable sync faults dynamically
            let storage_fault_cfg = ctx.storage_fault_config();
            storage_fault_cfg.write().sync_rate = Some(probability!(1.0));

            // Now sync should fail
            blob.write_at(0, b"updated".to_vec(), WriteOptions::default())
                .await
                .unwrap();
            let result = blob.sync().await;
            assert!(result.is_err(), "sync should fail with faults enabled");

            // Disable faults
            storage_fault_cfg.write().sync_rate = Some(probability!(0.0));

            // Sync should succeed again
            blob.sync()
                .await
                .expect("sync should succeed with faults disabled");
        });
    }

    #[test]
    fn test_storage_fault_determinism() {
        // Run the same sequence twice with the same seed
        fn run_with_seed(seed: u64) -> Vec<bool> {
            let cfg = deterministic::Config::default()
                .with_seed(seed)
                .with_storage_fault_config(FaultConfig {
                    open_rate: Some(probability!(0.5)),
                    ..Default::default()
                });

            let runner = deterministic::Runner::new(cfg);
            runner.start(|ctx| async move {
                let mut results = Vec::new();
                for i in 0..20 {
                    let name = format!("blob{i}");
                    let result = ctx.open("test_determinism", name.as_bytes()).await;
                    results.push(result.is_ok());
                }
                results
            })
        }

        let results1 = run_with_seed(12345);
        let results2 = run_with_seed(12345);
        assert_eq!(
            results1, results2,
            "same seed should produce same failure pattern"
        );

        let results3 = run_with_seed(99999);
        assert_ne!(
            results1, results3,
            "different seeds should produce different patterns"
        );
    }

    #[test]
    fn test_storage_fault_determinism_multi_task() {
        // Run the same multi-task sequence twice with the same seed.
        // This tests that task shuffling + fault decisions interleave deterministically.
        fn run_with_seed(seed: u64) -> Vec<u32> {
            let cfg = deterministic::Config::default()
                .with_seed(seed)
                .with_storage_fault_config(FaultConfig {
                    open_rate: Some(probability!(0.5)),
                    write_rate: Some(WriteConfig {
                        failure_rate: probability!(0.3),
                        retention_rate: probability!(0.0),
                        mode: PartialWriteMode::Prefix,
                    }),
                    sync_rate: Some(probability!(0.2)),
                    ..Default::default()
                });

            let runner = deterministic::Runner::new(cfg);
            runner.start(|ctx| async move {
                // Spawn multiple tasks that do storage operations
                let mut handles = Vec::new();
                for i in 0..5 {
                    let ctx = ctx.child("task");
                    handles.push(ctx.spawn(move |ctx| async move {
                        let mut successes = 0u32;
                        for j in 0..4 {
                            let name = format!("task{i}_blob{j}");
                            if let Ok((blob, _)) = ctx.open("partition", name.as_bytes()).await {
                                successes += 1;
                                if blob
                                    .write_at(0, b"data".to_vec(), WriteOptions::default())
                                    .await
                                    .is_ok()
                                {
                                    successes += 1;
                                }
                                if blob.sync().await.is_ok() {
                                    successes += 1;
                                }
                            }
                        }
                        successes
                    }));
                }

                // Collect results from all tasks
                let mut results = Vec::new();
                for handle in handles {
                    results.push(handle.await.unwrap());
                }
                results
            })
        }

        let results1 = run_with_seed(42);
        let results2 = run_with_seed(42);
        assert_eq!(
            results1, results2,
            "same seed should produce same multi-task pattern"
        );

        let results3 = run_with_seed(99999);
        assert_ne!(
            results1, results3,
            "different seeds should produce different patterns"
        );
    }

    #[test]
    fn test_resolver() {
        let executor = deterministic::Runner::default();
        executor.start(|context| async move {
            // Register DNS mappings
            let ip1: IpAddr = "192.168.1.1".parse().unwrap();
            let ip2: IpAddr = "192.168.1.2".parse().unwrap();
            context.resolver_register("example.com", Some(vec![ip1, ip2]));

            // Resolve registered hostname
            let addrs = context.resolve("example.com").await.unwrap();
            assert_eq!(addrs, vec![ip1, ip2]);

            // Resolve unregistered hostname
            let result = context.resolve("unknown.com").await;
            assert!(matches!(result, Err(Error::ResolveFailed(_))));

            // Remove mapping
            context.resolver_register("example.com", None);
            let result = context.resolve("example.com").await;
            assert!(matches!(result, Err(Error::ResolveFailed(_))));
        });
    }

    /// A strategy with parallelism greater than one must behave as configured under the
    /// deterministic runtime even though no worker threads exist.
    #[test]
    fn test_parallel_strategy_spawn_completes() {
        let executor = deterministic::Runner::default();
        executor.start(|context| async move {
            let strategy = context.child("pool").strategy(NZUsize!(2)).manual();
            assert_eq!(strategy.parallelism(), 2);

            let output = strategy
                .spawn(2, |strategy| strategy.map_collect_vec(0..2, |i| i + 1))
                .await;

            assert_eq!(output, vec![1, 2]);
        });
    }

    /// Strategies share the pool registered with the executor thread, but each request must
    /// retain its own planning parallelism and execute work. This covers multiple strategies
    /// within one runner and a later runner on the same thread.
    #[test]
    fn test_strategies_reuse_pool_across_runners() {
        let executor = deterministic::Runner::default();
        executor.start(|context| async move {
            let first = context.child("pool_a").strategy(NZUsize!(1)).manual();
            assert_eq!(first.parallelism(), 1);
            assert_eq!(first.run(2, || "serial", || "parallel"), "serial");
            let output = first
                .spawn(2, |strategy| strategy.map_collect_vec(0..2, |i| i + 1))
                .now_or_never()
                .expect("single-threaded pool should run spawned work inline");
            assert_eq!(output, vec![1, 2]);

            let second = context.child("pool_b").strategy(NZUsize!(3)).manual();
            assert_eq!(second.parallelism(), 3);
            assert_eq!(second.run(2, || "serial", || "parallel"), "parallel");
            let output = second
                .spawn(3, |strategy| strategy.map_collect_vec(0..3, |i| i + 1))
                .now_or_never()
                .expect("single-threaded pool should run spawned work inline");
            assert_eq!(output, vec![1, 2, 3]);
        });

        let executor = deterministic::Runner::default();
        executor.start(|context| async move {
            let third = context.child("pool_c").strategy(NZUsize!(4)).manual();
            assert_eq!(third.parallelism(), 4);
            assert_eq!(third.run(2, || "serial", || "parallel"), "parallel");
            let output = third
                .spawn(4, |strategy| strategy.map_collect_vec(0..4, |i| i + 1))
                .now_or_never()
                .expect("single-threaded pool should run spawned work inline");
            assert_eq!(output, vec![1, 2, 3, 4]);
        });
    }

    /// Tasks may suspend while a pool exists: pools have no worker tasks for the executor
    /// to poll (a polled rayon worker loop would block or abort the runtime), so suspension
    /// must leave the pool usable.
    #[test]
    fn test_pool_survives_suspension() {
        let executor = deterministic::Runner::default();
        executor.start(|context| async move {
            let strategy = context.child("pool").strategy(NZUsize!(2)).manual();
            context.sleep(Duration::from_millis(10)).await;

            let output = strategy
                .spawn(2, |strategy| strategy.map_collect_vec(0..2, |i| i + 1))
                .await;
            assert_eq!(output, vec![1, 2]);

            context.sleep(Duration::from_millis(10)).await;
            let sum = strategy.fold(0..100u64, || 0u64, |acc, i| acc + i, |a, b| a + b);
            assert_eq!(sum, 4950);
        });
    }

    /// Open `name` in `partition`, write `data`, and sync it.
    async fn write_synced(ctx: &Context, partition: &str, name: &[u8], data: &[u8]) {
        let (blob, _) = ctx.open(partition, name).await.unwrap();
        blob.write_at(0, data.to_vec(), WriteOptions::SYNC)
            .await
            .unwrap();
    }

    /// Reopen a blob and read all of its logical contents.
    async fn read_all(ctx: &Context, partition: &str, name: &[u8]) -> Vec<u8> {
        let (blob, len) = ctx.open(partition, name).await.unwrap();
        blob.read_at(0, len as usize, ReadOptions::default())
            .await
            .unwrap()
            .coalesce()
            .as_ref()
            .to_vec()
    }

    #[test]
    fn test_corrupt_bit_persists_across_reopen() {
        deterministic::Runner::default().start(|ctx| async move {
            write_synced(&ctx, "node", b"blob", b"data").await;

            // A crashed process's handle may still be alive, but it can no longer publish.
            let (process_ctx, process) = ctx.process("node", |partition| partition == "node");
            let (opened, opened_rx) = oneshot::channel();
            let task = process_ctx.spawn(move |ctx| async move {
                let (blob, _) = ctx.open("node", b"blob").await.unwrap();
                blob.write_at(0, b"DATA".to_vec(), WriteOptions::default())
                    .await
                    .unwrap();
                let _ = opened.send(());
                futures::future::pending::<()>().await;
                drop(blob);
            });
            opened_rx.await.unwrap();
            process.crash();

            // Bit 10 is bit 2 of byte 1: 'a' (0x61) becomes 'e' (0x65).
            ctx.corrupt_bit("node", b"blob", 10);
            assert_eq!(
                ctx.logical_blob("node", b"blob").as_deref(),
                Some(&b"deta"[..])
            );

            // Reopening waits for the crashed task (and its handle) to be dropped.
            assert!(task.await.is_err());
            assert_eq!(read_all(&ctx, "node", b"blob").await, b"deta");
        });
    }

    #[test]
    #[should_panic(expected = "crash its process before corrupting it")]
    fn test_corrupt_bit_rejects_open_handle() {
        deterministic::Runner::default().start(|ctx| async move {
            let (blob, _) = ctx.open("node", b"blob").await.unwrap();
            blob.write_at(0, b"data".to_vec(), WriteOptions::SYNC)
                .await
                .unwrap();
            ctx.corrupt_bit("node", b"blob", 0);
        });
    }

    #[test]
    #[should_panic(expected = "bit 32 is outside blob node/626c6f62 (32 bits)")]
    fn test_corrupt_bit_rejects_out_of_range_bit() {
        deterministic::Runner::default().start(|ctx| async move {
            write_synced(&ctx, "node", b"blob", b"data").await;
            ctx.corrupt_bit("node", b"blob", 32);
        });
    }

    #[test]
    #[should_panic(expected = "has no durable contents")]
    fn test_corrupt_bit_rejects_missing_blob() {
        deterministic::Runner::default().start(|ctx| async move {
            ctx.corrupt_bit("node", b"missing", 0);
        });
    }

    #[test]
    fn test_misdirect_copies_exactly_the_range() {
        deterministic::Runner::default().start(|ctx| async move {
            write_synced(&ctx, "node", b"a", b"abcdefgh").await;
            write_synced(&ctx, "other", b"b", b"12345678").await;

            ctx.misdirect(("node", b"a"), 2..5, ("other", b"b"), 1);
            assert_eq!(read_all(&ctx, "other", b"b").await, b"1cde5678");
            assert_eq!(read_all(&ctx, "node", b"a").await, b"abcdefgh");

            // Overlapping ranges within one blob copy the original source bytes.
            ctx.misdirect(("node", b"a"), 0..4, ("node", b"a"), 2);
            assert_eq!(read_all(&ctx, "node", b"a").await, b"ababcdgh");

            // An empty range changes nothing.
            ctx.misdirect(("node", b"a"), 3..3, ("other", b"b"), 8);
            assert_eq!(read_all(&ctx, "other", b"b").await, b"1cde5678");
        });
    }

    #[test]
    #[should_panic(expected = "4 bytes at offset 5 exceed blob other/62 (8 bytes)")]
    fn test_misdirect_rejects_target_overflow() {
        deterministic::Runner::default().start(|ctx| async move {
            write_synced(&ctx, "node", b"a", b"abcdefgh").await;
            write_synced(&ctx, "other", b"b", b"12345678").await;
            ctx.misdirect(("node", b"a"), 0..4, ("other", b"b"), 5);
        });
    }

    #[test]
    #[should_panic(expected = "range 6..9 is outside blob node/61 (8 bytes)")]
    fn test_misdirect_rejects_source_overflow() {
        deterministic::Runner::default().start(|ctx| async move {
            write_synced(&ctx, "node", b"a", b"abcdefgh").await;
            ctx.misdirect(("node", b"a"), 6..9, ("node", b"a"), 0);
        });
    }

    #[test]
    fn test_restore_blob_reverts_length_and_contents() {
        deterministic::Runner::default().start(|ctx| async move {
            write_synced(&ctx, "node", b"blob", b"short").await;
            let snapshot = ctx.snapshot_blob("node", b"blob");
            assert_eq!(snapshot.partition(), "node");
            assert_eq!(snapshot.name(), b"blob");

            {
                let (blob, _) = ctx.open("node", b"blob").await.unwrap();
                blob.write_at(0, b"SHORT and longer".to_vec(), WriteOptions::SYNC)
                    .await
                    .unwrap();
            }
            ctx.restore_blob(&snapshot);
            assert_eq!(read_all(&ctx, "node", b"blob").await, b"short");

            // A removed blob is recreated with its snapshotted contents.
            ctx.remove("node", None).await.unwrap();
            ctx.restore_blob(&snapshot);
            assert_eq!(ctx.scan("node").await.unwrap(), vec![b"blob".to_vec()]);
            assert_eq!(read_all(&ctx, "node", b"blob").await, b"short");
        });
    }

    #[test]
    fn test_restore_blob_discards_pending_crash_writes() {
        let faults = FaultConfig::default().write(WriteConfig {
            failure_rate: probability!(0.0),
            retention_rate: probability!(1.0),
            mode: PartialWriteMode::Prefix,
        });
        Runner::new(Config::default().with_storage_fault_config(faults)).start(|ctx| async move {
            write_synced(&ctx, "node", b"blob", b"old").await;
            let snapshot = ctx.snapshot_blob("node", b"blob");
            let (process_ctx, process) = ctx.process("node", |partition| partition == "node");
            {
                let (blob, _) = process_ctx.open("node", b"blob").await.unwrap();
                blob.write_at(0, b"new!".to_vec(), WriteOptions::default())
                    .await
                    .unwrap();
            }

            // The unsynced write would survive the crash, but the restore discards it.
            ctx.restore_blob(&snapshot);
            process.crash();
            assert_eq!(read_all(&ctx, "node", b"blob").await, b"old");
        });
    }

    #[test]
    #[should_panic(expected = "crash its process before restoring it")]
    fn test_restore_blob_rejects_open_handle() {
        deterministic::Runner::default().start(|ctx| async move {
            write_synced(&ctx, "node", b"blob", b"data").await;
            let snapshot = ctx.snapshot_blob("node", b"blob");
            let _blob = ctx.open("node", b"blob").await.unwrap();
            ctx.restore_blob(&snapshot);
        });
    }

    #[test]
    fn test_durable_corruption_is_audited_and_deterministic() {
        fn run(bit: u64) -> (String, Digest) {
            deterministic::Runner::seeded(7).start(|ctx| async move {
                write_synced(&ctx, "node", b"a", b"abcdefgh").await;
                write_synced(&ctx, "node", b"b", b"12345678").await;
                let snapshot = ctx.snapshot_blob("node", b"b");
                ctx.corrupt_bit("node", b"a", bit);
                ctx.misdirect(("node", b"a"), 0..4, ("node", b"b"), 4);
                ctx.restore_blob(&snapshot);
                ctx.corrupt_bit("node", b"b", bit);
                (ctx.auditor().state(), ctx.storage_audit())
            })
        }

        assert_eq!(run(3), run(3));
        let (state, audit) = run(3);
        let (other_state, other_audit) = run(4);
        assert_ne!(state, other_state);
        assert_ne!(audit, other_audit);
    }

    /// Corrupts exactly the reads of one blob, flipping the highest bit drawn.
    #[derive(Default)]
    struct CorruptBlobReads {
        #[allow(clippy::type_complexity)]
        seen: Mutex<Vec<(Option<Vec<u8>>, StorageOp, FaultDraw)>>,
    }

    impl FaultPolicy for CorruptBlobReads {
        fn occurs(&self, decision: &FaultDecision<'_>, _: commonware_utils::Probability) -> bool {
            self.seen.lock().push((
                decision.name.map(<[u8]>::to_vec),
                decision.op,
                decision.draw,
            ));
            decision.draw == FaultDraw::Corrupt && decision.name == Some(b"target")
        }

        fn between(&self, decision: &FaultDecision<'_>, range: std::ops::Range<u64>) -> u64 {
            assert_eq!(decision.draw, FaultDraw::CorruptBit);
            range.end - 1
        }
    }

    #[test]
    fn test_read_corruption_follows_policy() {
        let policy = Arc::new(CorruptBlobReads::default());
        let cfg = deterministic::Config::default()
            .with_storage_fault_config(FaultConfig::default().corrupt_read(probability!(0.5)))
            .with_storage_fault_policy(policy.clone());
        deterministic::Runner::new(cfg).start(|ctx| async move {
            for name in [&b"target"[..], b"healthy"] {
                write_synced(&ctx, "node", name, b"data").await;
            }
            // The highest bit of the final byte flips: 'a' (0x61) becomes 0xe1.
            assert_eq!(read_all(&ctx, "node", b"target").await, b"dat\xe1");
            assert_eq!(read_all(&ctx, "node", b"healthy").await, b"data");
            // Durable contents are unchanged.
            assert_eq!(
                ctx.logical_blob("node", b"target").as_deref(),
                Some(&b"data"[..])
            );
        });
        let seen = policy.seen.lock();
        let reads: Vec<_> = seen
            .iter()
            .filter(|(_, op, _)| *op == StorageOp::Read)
            .map(|(name, _, draw)| (name.as_deref(), *draw))
            .collect();
        assert_eq!(
            reads,
            [
                (Some(&b"target"[..]), FaultDraw::Fail),
                (Some(&b"target"[..]), FaultDraw::Corrupt),
                (Some(&b"healthy"[..]), FaultDraw::Fail),
                (Some(&b"healthy"[..]), FaultDraw::Corrupt),
            ]
        );
    }
}
