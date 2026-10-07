//! Network faults a test harness injects into a simulation while it runs.
//!
//! [Partitions] is a [NetworkPolicy] holding partition and clog state that any task can change
//! mid-run, delegating every other decision to an inner policy (such as [Jitter]). The same type
//! drives the runtime's deterministic sockets ([Partitions::sockets], installed with
//! [super::Config::with_network_policy]) and message-level simulated networks keyed by peer
//! identity ([Partitions::new], installed with `commonware_p2p::simulated::Oracle::set_policy`).
//! [Swizzle] drives a [Partitions] with FoundationDB's swizzle-clog nemesis.
//!
//! Every change is recorded by the [super::Auditor], and every random choice is drawn from a
//! seeded RNG owned by the helper that makes it, so a run is reproducible from its seeds.

use super::{Context, NetworkDelivery, NetworkPolicy, NetworkTransmission, Process, ProcessState};
use crate::Clock;
use commonware_utils::{Probability, sync::Mutex};
use rand::{RngExt as _, SeedableRng, prelude::SliceRandom, rngs::StdRng};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    net::{IpAddr, SocketAddr},
    sync::{Arc, Weak},
    time::{Duration, SystemTime},
};

/// Draws a duration uniformly from `0..=max`.
fn uniform(rng: &mut StdRng, max: Duration) -> Duration {
    let nanos = u64::try_from(max.as_nanos()).unwrap_or(u64::MAX);
    if nanos == 0 {
        return Duration::ZERO;
    }
    Duration::from_nanos(rng.random_range(0..=nanos))
}

/// A seeded network policy that delays every transmission by a base latency plus uniform jitter
/// and loses a fraction of them.
///
/// Loss is a [NetworkDelivery::DROP], which a message-level network carries out by losing the
/// message and a connection-oriented one (the runtime's sockets) by resetting the connection.
/// Each transmission draws its jitter, then (only when loss is nonzero) whether it is lost.
pub struct Jitter {
    rng: Mutex<StdRng>,
    latency: Duration,
    jitter: Duration,
    loss: Probability,
}

impl Jitter {
    /// Delay every transmission by `latency` plus a uniform draw from `0..=jitter`, drawing from
    /// an RNG seeded with `seed`.
    pub fn new(seed: u64, latency: Duration, jitter: Duration) -> Self {
        Self {
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
            latency,
            jitter,
            loss: commonware_utils::probability!(0.0),
        }
    }

    /// Also lose each transmission with probability `loss`.
    pub const fn with_loss(mut self, loss: Probability) -> Self {
        self.loss = loss;
        self
    }
}

impl<A> NetworkPolicy<A> for Jitter {
    fn delivers(&self, _: &NetworkTransmission<'_, A>) -> NetworkDelivery {
        let mut rng = self.rng.lock();
        let after = self
            .latency
            .checked_add(uniform(&mut rng, self.jitter))
            .expect("latency overflow");
        if !self.loss.is_zero() && self.loss.sample(&mut *rng) {
            return NetworkDelivery::DROP;
        }
        NetworkDelivery::after(after)
    }
}

/// The runtime's time, unaffected by any clock offset of `context`'s process.
fn runtime_time(context: &Context) -> SystemTime {
    *context.executor().time.lock()
}

/// Partition and clog state of a [Partitions].
struct State<H> {
    /// Directed links whose traffic is cut, as `(from, to)`.
    cuts: BTreeSet<(H, H)>,
    /// Directed links whose traffic is held until the given time, as `(from, to)`.
    clogs: BTreeMap<(H, H), SystemTime>,
}

/// A network policy whose partitions and clogs a test changes while the simulation runs.
///
/// Faults apply between hosts: each endpoint `A` of a transmission belongs to the host `H` that
/// a mapping assigns it (a peer is its own host; a socket address belongs to its IP). Traffic
/// between endpoints of the same host is never faulted. Each fault is directed:
///
/// - A cut link (see [Self::cut] and [Self::partition]) carries no traffic. A message-level
///   network loses every message sent on it, while the reverse direction keeps working. Sockets
///   cannot carry a one-way partition (a TCP connection needs both directions for its
///   handshake and acknowledgments), so [Self::sockets] refuses dials between the hosts in
///   either direction, and a send on the cut direction of an open connection stalls for
///   [Self::SOCKET_TIMEOUT] (as a retransmission timeout would) and then resets the connection.
///   Sends in the uncut direction are still delivered until then.
/// - A clogged link (see [Self::clog]) holds all traffic sent on it until the clog expires, then
///   releases it in order, after the latency the inner policy decides. On sockets, a send on a
///   clogged link stalls until then. A clog cannot be lifted early: traffic is held until its
///   scheduled release even after [Self::heal], as the bytes already sit in a stalled queue.
///
/// Every other decision (and the latency of every transmission) is the inner policy's, which is
/// consulted for every transmission whether or not a fault applies, so faults do not shift the
/// inner policy's random draws.
///
/// Changes take a [Context] of the simulation, whose auditor records them and whose runtime
/// clock (ignoring any process clock offset) times clogs.
pub struct Partitions<A, H = A> {
    inner: Box<dyn NetworkPolicy<A>>,
    host: fn(&A) -> H,
    cut: NetworkDelivery,
    connections: bool,
    state: Mutex<State<H>>,
}

impl<A: Ord + Clone + Debug + Send> Partitions<A> {
    /// A policy for a message-level network (such as `commonware_p2p::simulated`) on which each
    /// endpoint is its own host. Messages on a cut link are lost.
    pub fn new(inner: impl NetworkPolicy<A> + 'static) -> Self {
        Self::with_hosts(inner, A::clone)
    }
}

impl Partitions<SocketAddr, IpAddr> {
    /// How long a send on a cut socket link stalls before its connection resets.
    pub const SOCKET_TIMEOUT: Duration = Duration::from_secs(10);

    /// A policy for the runtime's deterministic sockets, on which each IP is a host.
    ///
    /// Give each simulated host its own IP: it listens on that IP, and its dials originate from
    /// it once its process is assigned it with [Process::set_ip] (dials from a context outside
    /// any process with an IP originate from `127.0.0.1`).
    pub fn sockets(inner: impl NetworkPolicy + 'static) -> Self {
        let mut partitions = Self::with_hosts(inner, SocketAddr::ip);
        partitions.cut = NetworkDelivery {
            after: Self::SOCKET_TIMEOUT,
            ..NetworkDelivery::RESET
        };
        partitions.connections = true;
        partitions
    }
}

impl<A, H: Ord + Clone + Debug + Send> Partitions<A, H> {
    /// A policy that assigns each endpoint the host `host` maps it to. Transmissions on a cut
    /// link are lost.
    pub fn with_hosts(inner: impl NetworkPolicy<A> + 'static, host: fn(&A) -> H) -> Self {
        Self {
            inner: Box::new(inner),
            host,
            cut: NetworkDelivery::DROP,
            connections: false,
            state: Mutex::new(State {
                cuts: BTreeSet::new(),
                clogs: BTreeMap::new(),
            }),
        }
    }

    /// Carry out every transmission on a cut link as `delivery` (by default, a loss).
    pub const fn with_cut(mut self, delivery: NetworkDelivery) -> Self {
        self.cut = delivery;
        self
    }

    /// Cut every link between hosts of different `groups`, in both directions. Hosts outside
    /// every group, and links already cut, are unaffected.
    pub fn partition<G, I>(&self, context: &Context, groups: G)
    where
        G: IntoIterator<Item = I>,
        I: IntoIterator<Item = H>,
    {
        let groups: Vec<Vec<H>> = groups
            .into_iter()
            .map(|group| group.into_iter().collect())
            .collect();
        context.auditor().event(b"nemesis_partition", |hasher| {
            for group in &groups {
                hasher.update((group.len() as u64).to_be_bytes());
                for host in group {
                    hasher.update(format!("{host:?}"));
                }
            }
        });
        let mut state = self.state.lock();
        for (i, group) in groups.iter().enumerate() {
            for (j, other) in groups.iter().enumerate() {
                if i == j {
                    continue;
                }
                for from in group {
                    for to in other {
                        if from != to {
                            state.cuts.insert((from.clone(), to.clone()));
                        }
                    }
                }
            }
        }
    }

    /// Cut the link from `from` to `to`, leaving the link from `to` to `from` as it is.
    pub fn cut(&self, context: &Context, from: H, to: H) {
        context.auditor().event(b"nemesis_cut", |hasher| {
            hasher.update(format!("{from:?}"));
            hasher.update(format!("{to:?}"));
        });
        if from != to {
            self.state.lock().cuts.insert((from, to));
        }
    }

    /// Hold all traffic from `from` to `to` for `duration` (from now), then release it in order.
    /// A link that is already clogged stays clogged until the later of the two releases.
    pub fn clog(&self, context: &Context, from: H, to: H, duration: Duration) {
        context.auditor().event(b"nemesis_clog", |hasher| {
            hasher.update(format!("{from:?}"));
            hasher.update(format!("{to:?}"));
            hasher.update(duration.as_nanos().to_be_bytes());
        });
        if from == to {
            return;
        }
        let mut state = self.state.lock();
        let now = runtime_time(context);
        let until = now.checked_add(duration).expect("clog overflow");
        let release = state.clogs.entry((from, to)).or_insert(until);
        *release = (*release).max(until);
    }

    /// Clog every link between `host` and each of `peers`, in both directions, for `duration`.
    pub fn clog_host(&self, context: &Context, host: &H, peers: &[H], duration: Duration) {
        for peer in peers {
            if peer != host {
                self.clog(context, host.clone(), peer.clone(), duration);
                self.clog(context, peer.clone(), host.clone(), duration);
            }
        }
    }

    /// Restore the link from `from` to `to`; see [Self::heal] for what a clog leaves behind.
    pub fn heal_link(&self, context: &Context, from: &H, to: &H) {
        context.auditor().event(b"nemesis_heal_link", |hasher| {
            hasher.update(format!("{from:?}"));
            hasher.update(format!("{to:?}"));
        });
        let mut state = self.state.lock();
        let key = (from.clone(), to.clone());
        state.cuts.remove(&key);
        state.clogs.remove(&key);
    }

    /// Restore every link: no further traffic is cut or held. Traffic a clog already holds is
    /// still released at its scheduled time, and connections already reset stay closed.
    pub fn heal(&self, context: &Context) {
        context.auditor().event(b"nemesis_heal", |_| {});
        let mut state = self.state.lock();
        state.cuts.clear();
        state.clogs.clear();
    }

    /// Whether the link from `from` to `to` is cut.
    pub fn is_cut(&self, from: &H, to: &H) -> bool {
        self.state.lock().cuts.contains(&(from.clone(), to.clone()))
    }

    /// When traffic sent now from `from` to `to` would be released, if the link is clogged.
    pub fn clogged_until(&self, context: &Context, from: &H, to: &H) -> Option<SystemTime> {
        let now = runtime_time(context);
        self.state
            .lock()
            .clogs
            .get(&(from.clone(), to.clone()))
            .copied()
            .filter(|until| *until > now)
    }
}

impl<A, H> NetworkPolicy<A> for Partitions<A, H>
where
    H: Ord + Clone + Debug + Send,
{
    fn connects(&self, from: &A, to: &A) -> bool {
        let connects = self.inner.connects(from, to);
        let (from, to) = ((self.host)(from), (self.host)(to));
        if from == to {
            return connects;
        }
        let state = self.state.lock();
        let cut = state.cuts.contains(&(from.clone(), to.clone()))
            || (self.connections && state.cuts.contains(&(to, from)));
        connects && !cut
    }

    fn delivers(&self, transmission: &NetworkTransmission<'_, A>) -> NetworkDelivery {
        let mut delivery = self.inner.delivers(transmission);
        let key = (
            (self.host)(&transmission.link.from),
            (self.host)(&transmission.link.to),
        );
        if key.0 == key.1 {
            return delivery;
        }
        let mut state = self.state.lock();
        if state.cuts.contains(&key) {
            return self.cut;
        }
        let Some(until) = state.clogs.get(&key).copied() else {
            return delivery;
        };
        match until.duration_since(transmission.at) {
            Ok(held) if !held.is_zero() => {
                delivery.after = held.checked_add(delivery.after).expect("clog overflow");
            }
            _ => {
                state.clogs.remove(&key);
            }
        }
        delivery
    }
}

/// One host's clog in a [Swizzle] plan, timed from the start of the swizzle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwizzleStep<H> {
    /// The clogged host: every link between it and the other swizzled hosts is clogged.
    pub host: H,
    /// When the host is clogged.
    pub start: Duration,
    /// When the host's clog expires.
    pub end: Duration,
}

/// FoundationDB's swizzle-clog nemesis: clog a random subset of hosts one at a time, with random
/// gaps, then unclog them in reverse order.
///
/// Clogging hosts gradually (rather than partitioning them at once) and releasing them in reverse
/// produces long-lived, shifting partial partitions in which different hosts see different
/// subsets of each other. Every choice is drawn from an RNG seeded with the swizzle's seed.
#[derive(Clone, Debug)]
pub struct Swizzle {
    seed: u64,
    count: Option<usize>,
    gap: Duration,
}

impl Swizzle {
    /// A swizzle drawing from `seed`, clogging a random number of hosts with gaps of up to one
    /// second.
    pub const fn new(seed: u64) -> Self {
        Self {
            seed,
            count: None,
            gap: Duration::from_secs(1),
        }
    }

    /// Clog `count` hosts (or all of them, if fewer) instead of a random number.
    pub const fn with_count(mut self, count: usize) -> Self {
        self.count = Some(count);
        self
    }

    /// Draw each gap between consecutive clogs and unclogs uniformly from `0..=gap`.
    pub const fn with_gap(mut self, gap: Duration) -> Self {
        self.gap = gap;
        self
    }

    /// The clogs a swizzle over `hosts` makes, in the order they start.
    ///
    /// Picks the hosts, then draws the gap before each clog starts and the gap before each
    /// (later) host's clog expires, so the last host clogged is the first released.
    pub fn plan<H: Clone>(&self, hosts: &[H]) -> Vec<SwizzleStep<H>> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        if hosts.is_empty() {
            return Vec::new();
        }
        let count = self
            .count
            .unwrap_or_else(|| rng.random_range(1..=hosts.len()))
            .min(hosts.len());
        let mut chosen: Vec<usize> = (0..hosts.len()).collect();
        chosen.shuffle(&mut rng);
        chosen.truncate(count);

        let mut at = Duration::ZERO;
        let mut steps: Vec<_> = chosen
            .into_iter()
            .map(|index| {
                at = at
                    .checked_add(uniform(&mut rng, self.gap))
                    .expect("swizzle overflow");
                SwizzleStep {
                    host: hosts[index].clone(),
                    start: at,
                    end: at,
                }
            })
            .collect();
        for step in steps.iter_mut().rev() {
            at = at
                .checked_add(uniform(&mut rng, self.gap))
                .expect("swizzle overflow");
            step.end = at;
        }
        steps
    }

    /// Run the swizzle over `hosts` on `partitions`, returning once every clog has expired.
    ///
    /// Each step clogs every link between its host and every other of `hosts`, in both
    /// directions. Returns the plan it ran (see [Self::plan]).
    pub async fn run<A, H>(
        &self,
        context: &Context,
        partitions: &Partitions<A, H>,
        hosts: &[H],
    ) -> Vec<SwizzleStep<H>>
    where
        H: Ord + Clone + Debug + Send,
    {
        let steps = self.plan(hosts);
        let mut at = Duration::ZERO;
        for step in &steps {
            context.sleep(step.start - at).await;
            at = step.start;
            partitions.clog_host(context, &step.host, hosts, step.end - step.start);
        }
        if let Some(last) = steps.iter().map(|step| step.end).max() {
            context.sleep(last - at).await;
        }
        steps
    }
}

/// The source IP of each [Process] given one, by process.
static SOURCES: Mutex<Vec<(Weak<ProcessState>, IpAddr)>> = Mutex::new(Vec::new());

/// The source IP set for `process` or the nearest process it was started from.
fn source_of(process: &Arc<ProcessState>) -> Option<IpAddr> {
    let sources = SOURCES.lock();
    let mut current = Some(process);
    while let Some(process) = current {
        if let Some((_, ip)) = sources
            .iter()
            .find(|(candidate, _)| ptr_eq(candidate, process))
        {
            return Some(*ip);
        }
        current = process.parent.as_ref();
    }
    None
}

fn ptr_eq(weak: &Weak<ProcessState>, process: &Arc<ProcessState>) -> bool {
    std::ptr::eq(weak.as_ptr(), Arc::as_ptr(process))
}

impl Process {
    /// Make every dial of the process (and of processes started from its context, unless they
    /// have their own) originate from `ip`, as a host's connections originate from its address.
    ///
    /// A dialer is identified by its source address, so this is what lets a socket policy (such
    /// as [Partitions::sockets]) attribute a connection to the host that dialed it. Listeners
    /// are unaffected: bind the process's listeners on `ip` too.
    pub fn set_ip(&self, ip: IpAddr) {
        self.executor().auditor.event(b"process_ip", |hasher| {
            hasher.update(ip.to_string());
        });
        let mut sources = SOURCES.lock();
        sources.retain(|(process, _)| process.strong_count() > 0);
        let state = &self.host.state;
        match sources
            .iter_mut()
            .find(|(process, _)| ptr_eq(process, state))
        {
            Some((_, source)) => *source = ip,
            None => sources.push((Arc::downgrade(state), ip)),
        }
    }
}

impl Context {
    /// The source IP of the context's dials, if its process has one (see [Process::set_ip]).
    pub(super) fn source_ip(&self) -> Option<IpAddr> {
        self.process.as_ref().and_then(source_of)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Listener as _, Network as _, Runner as _, Sink as _, Spawner as _, Stream as _,
        Supervisor as _,
        deterministic::{self, NetworkFate, NetworkLink},
    };
    use std::net::Ipv4Addr;

    fn host(i: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(10, 0, 0, i))
    }

    fn address(i: u8) -> SocketAddr {
        SocketAddr::new(host(i), 3000)
    }

    #[test]
    fn test_jitter_is_seeded() {
        let link = NetworkLink { from: 1u8, to: 2u8 };
        let draw = |seed| {
            let jitter = Jitter::new(seed, Duration::from_millis(10), Duration::from_millis(5))
                .with_loss(Probability::from_f64(0.5).unwrap());
            (0..32)
                .map(|index| {
                    jitter.delivers(&NetworkTransmission {
                        link: &link,
                        channel: None,
                        index,
                        len: 1,
                        at: SystemTime::UNIX_EPOCH,
                    })
                })
                .collect::<Vec<_>>()
        };
        let deliveries = draw(7);
        assert_eq!(deliveries, draw(7));
        assert_ne!(deliveries, draw(8));
        assert!(deliveries.iter().any(|d| d.fate == NetworkFate::Drop));
        for delivery in deliveries.iter().filter(|d| d.fate == NetworkFate::Deliver) {
            assert!(delivery.after >= Duration::from_millis(10));
            assert!(delivery.after <= Duration::from_millis(15));
        }
    }

    #[test]
    fn test_swizzle_plan_unclogs_in_reverse() {
        let hosts: Vec<u32> = (0..8).collect();
        for seed in 0..32 {
            let swizzle = Swizzle::new(seed).with_gap(Duration::from_millis(100));
            let plan = swizzle.plan(&hosts);
            assert_eq!(plan, swizzle.plan(&hosts), "plans are seeded");
            assert!(!plan.is_empty() && plan.len() <= hosts.len());
            let unique: BTreeSet<_> = plan.iter().map(|step| step.host).collect();
            assert_eq!(unique.len(), plan.len(), "each host is clogged once");
            for pair in plan.windows(2) {
                assert!(pair[0].start <= pair[1].start, "clogs start in order");
                assert!(pair[0].end >= pair[1].end, "clogs end in reverse order");
            }
            let last = plan.last().unwrap();
            assert!(last.start <= last.end);
        }
        assert_eq!(Swizzle::new(0).with_count(3).plan(&hosts).len(), 3);
        assert_eq!(Swizzle::new(0).with_count(30).plan(&hosts).len(), 8);
        assert!(Swizzle::new(0).plan::<u32>(&[]).is_empty());
    }

    /// Connects a dialer process on `from` to a listener on `to`, returning the dialer's sink and
    /// the listener's stream (and the listener's sink and the dialer's stream).
    async fn connect(
        context: &deterministic::Context,
        dialer: &deterministic::Context,
        to: SocketAddr,
    ) -> (
        crate::SinkOf<deterministic::Context>,
        crate::StreamOf<deterministic::Context>,
        crate::SinkOf<deterministic::Context>,
        crate::StreamOf<deterministic::Context>,
    ) {
        let mut listener = context.bind(to).await.unwrap();
        let (sink, stream) = dialer.dial(to).await.unwrap();
        let (_, peer_sink, peer_stream) = listener.accept().await.unwrap();
        (sink, peer_stream, peer_sink, stream)
    }

    #[test]
    fn test_sockets_attribute_dials_to_process_ip() {
        deterministic::Runner::default().start(|context| async move {
            let (node, process) = context.process("node", |_| false);
            process.set_ip(host(1));
            let nested = node.child("nested");
            let mut listener = context.bind(address(2)).await.unwrap();
            let _connection = nested.dial(address(2)).await.unwrap();
            let (dialer, _, _) = listener.accept().await.unwrap();
            assert_eq!(dialer.ip(), host(1));
            let _connection = context.dial(address(2)).await.unwrap();
            let (dialer, _, _) = listener.accept().await.unwrap();
            assert_eq!(dialer.ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
        });
    }

    #[test]
    fn test_sockets_one_way_cut_stalls_then_resets() {
        let partitions = Arc::new(Partitions::sockets(NetworkDelivery::NOW));
        let cfg = deterministic::Config::default().with_network_policy(partitions.clone());
        deterministic::Runner::new(cfg).start(|context| async move {
            let (a, a_process) = context.process("a", |_| false);
            a_process.set_ip(host(1));
            let (mut a_sink, mut b_stream, mut b_sink, mut a_stream) =
                connect(&context, &a, address(2)).await;

            // Cut a -> b: b's sends still arrive, a's stall and then reset the connection
            partitions.cut(&context, host(1), host(2));
            assert!(partitions.is_cut(&host(1), &host(2)));
            assert!(!partitions.is_cut(&host(2), &host(1)));
            b_sink.send(b"pong".to_vec()).await.unwrap();
            assert_eq!(a_stream.recv(4).await.unwrap().coalesce(), b"pong");
            let start = context.current();
            assert!(a_sink.send(b"ping".to_vec()).await.is_err());
            assert_eq!(
                context.current().duration_since(start).unwrap(),
                Partitions::SOCKET_TIMEOUT
            );
            assert!(b_stream.recv(1).await.is_err());

            // New connections fail in either direction while either direction is cut
            let _listener_a = context.bind(address(1)).await.unwrap();
            let _listener_b = context.bind(address(2)).await.unwrap();
            let (b, b_process) = context.process("b", |_| false);
            b_process.set_ip(host(2));
            assert!(a.dial(address(2)).await.is_err());
            assert!(b.dial(address(1)).await.is_err());

            // Healing restores connectivity
            partitions.heal(&context);
            let (mut a_sink, mut b_stream, _, _) = connect(&context, &a, address(3)).await;
            a_sink.send(b"ok".to_vec()).await.unwrap();
            assert_eq!(b_stream.recv(2).await.unwrap().coalesce(), b"ok");
        });
    }

    #[test]
    fn test_sockets_partition_and_clog() {
        let partitions = Arc::new(Partitions::sockets(NetworkDelivery::after(
            Duration::from_millis(5),
        )));
        let cfg = deterministic::Config::default().with_network_policy(partitions.clone());
        deterministic::Runner::new(cfg).start(|context| async move {
            let mut nodes = Vec::new();
            for i in 1..=3 {
                let (node, process) = context.process("node", |_| false);
                process.set_ip(host(i));
                nodes.push((node, process));
            }
            let _listener = context.bind(address(3)).await.unwrap();

            // Partition {1} | {2, 3}: 1 cannot reach 3, 2 can
            partitions.partition(&context, [vec![host(1)], vec![host(2), host(3)]]);
            assert!(nodes[0].0.dial(address(3)).await.is_err());
            drop(_listener);
            let (mut sink, mut stream, _, _) = connect(&context, &nodes[1].0, address(3)).await;

            // Clog 2 -> 3 for a second: sends stall until the clog clears, then arrive in order
            partitions.clog(&context, host(2), host(3), Duration::from_secs(1));
            assert!(
                partitions
                    .clogged_until(&context, &host(2), &host(3))
                    .is_some()
            );
            let start = context.current();
            sink.send(b"a".to_vec()).await.unwrap();
            sink.send(b"b".to_vec()).await.unwrap();
            let elapsed = context.current().duration_since(start).unwrap();
            assert_eq!(
                elapsed,
                Duration::from_secs(1) + Duration::from_millis(5) + Duration::from_millis(5)
            );
            assert_eq!(stream.recv(2).await.unwrap().coalesce(), b"ab");
            assert!(
                partitions
                    .clogged_until(&context, &host(2), &host(3))
                    .is_none()
            );
        });
    }

    #[test]
    fn test_swizzle_is_deterministic() {
        let run = |seed: u64| {
            let partitions = Arc::new(Partitions::sockets(Jitter::new(
                seed,
                Duration::from_millis(5),
                Duration::from_millis(5),
            )));
            let cfg = deterministic::Config::default()
                .with_seed(seed)
                .with_network_policy(partitions.clone());
            deterministic::Runner::new(cfg).start(|context| async move {
                let hosts: Vec<IpAddr> = (1..=4).map(host).collect();
                let mut sinks = Vec::new();
                let mut receivers = Vec::new();
                for i in 1..=4u8 {
                    let (node, process) = context.process("node", |_| false);
                    process.set_ip(host(i));
                    let to = address(if i == 4 { 1 } else { i + 1 });
                    let mut listener = context.bind(to).await.unwrap();
                    let (sink, _) = node.dial(to).await.unwrap();
                    let (_, _, stream) = listener.accept().await.unwrap();
                    sinks.push((node, sink));
                    receivers.push(stream);
                }
                // Each host streams to the next while the swizzle runs
                for (node, mut sink) in sinks {
                    node.spawn(move |context| async move {
                        for i in 0u32..200 {
                            if sink.send(i.to_be_bytes().to_vec()).await.is_err() {
                                return;
                            }
                            context.sleep(Duration::from_millis(20)).await;
                        }
                    });
                }
                let swizzle = Swizzle::new(seed).with_count(2);
                let steps = swizzle.run(&context, &partitions, &hosts).await;
                assert_eq!(steps.len(), 2);
                // Every stream still arrives in order once the swizzle releases it
                for mut stream in receivers {
                    for i in 0u32..200 {
                        let bytes = stream.recv(4).await.unwrap().coalesce();
                        assert_eq!(u32::from_be_bytes(bytes.as_ref().try_into().unwrap()), i);
                    }
                }
                context.auditor().state()
            })
        };
        assert_eq!(run(1), run(1));
        assert_ne!(run(1), run(2));
    }
}
