//! Fault campaigns for the authenticated networks over the deterministic runtime's sockets.
//!
//! Every peer runs in its own simulated process with its own IP. A seeded scenario injects
//! faults (crash/restart on the same address with the same key, partitions, swizzle-clog,
//! connection resets, latency spikes beyond the handshake timeout, bandwidth squeezes, process
//! pauses, small clock skews, and peer set changes), then heals everything and checks that every
//! peer reconnects to every other, every message sent is delivered, no reservation or
//! connection leaks, and connected peers are not redialed.

use crate::{
    Receiver as _, Recipients, Sender as _,
    authenticated::channels,
    simulated::sockets::{self, Testbed},
};
use commonware_cryptography::{Signer as _, ed25519};
use commonware_macros::select;
use commonware_runtime::{
    Clock, Metrics, Runner, Spawner, Supervisor as _,
    deterministic::{self, Bandwidth, ClockOffset, LinkConfig, Swizzle, Topology},
    telemetry::metrics::metric_samples,
};
use commonware_utils::{Probability, TestRng, channel::mpsc};
use rand::{RngExt as _, seq::SliceRandom as _};
use std::{
    collections::{BTreeSet, HashMap},
    marker::PhantomData,
    net::IpAddr,
    time::Duration,
};
use tracing::debug;

/// The sending half of a peer's test channel.
pub(crate) type Sender = channels::Sender<ed25519::PublicKey, deterministic::Context>;

/// The receiving half of a peer's test channel.
pub(crate) type Receiver = channels::Receiver<ed25519::PublicKey>;

/// An authenticated network implementation under test.
pub(crate) trait Variant {
    /// The network's oracle.
    type Oracle: Send + 'static;

    /// Create peer `i` (listening on [sockets::address]`(i)` with `signers[i]`, in `context`'s
    /// process), where the first `bootstrappers` peers serve as bootstrappers (if the variant
    /// has them), and start it.
    fn start(
        context: deterministic::Context,
        signers: &[ed25519::PrivateKey],
        i: usize,
        bootstrappers: usize,
    ) -> (Sender, Receiver, Self::Oracle);

    /// Track `members` (by index into `signers`) as peer set `index`.
    fn track(
        oracle: &mut Self::Oracle,
        signers: &[ed25519::PrivateKey],
        index: u64,
        members: &[usize],
    );
}

const MESSAGES: u32 = 5;

/// A message tag that receivers ignore (used to probe connectivity).
const PROBE: u8 = u8::MAX;

type Inbound = (usize, ed25519::PublicKey, Vec<u8>);

struct Running<O> {
    sender: Sender,
    oracle: O,
    process: deterministic::Process,
}

/// The peers of a campaign.
pub(crate) struct Cluster<V: Variant> {
    context: deterministic::Context,
    signers: Vec<ed25519::PrivateKey>,
    peers: Vec<Option<Running<V::Oracle>>>,
    generations: Vec<usize>,
    inbound_tx: mpsc::UnboundedSender<Inbound>,
    inbound_rx: mpsc::UnboundedReceiver<Inbound>,
    bootstrappers: usize,
    /// Peer sets tracked so far, as `(index, members)`.
    sets: Vec<(u64, Vec<usize>)>,
    _variant: PhantomData<V>,
}

impl<V: Variant> Cluster<V> {
    fn new(context: deterministic::Context, n: usize, bootstrappers: usize) -> Self {
        let mut cluster = Self::stopped(context, n, bootstrappers);
        for i in 0..n {
            cluster.start(i);
        }
        cluster
    }

    /// A cluster of `n` peers, none of them started.
    fn stopped(context: deterministic::Context, n: usize, bootstrappers: usize) -> Self {
        let signers = (0..n as u64).map(ed25519::PrivateKey::from_seed).collect();
        let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
        Self {
            context,
            signers,
            peers: (0..n).map(|_| None).collect(),
            generations: vec![0; n],
            inbound_tx,
            inbound_rx,
            bootstrappers,
            sets: vec![(0, (0..n).collect())],
            _variant: PhantomData,
        }
    }

    fn n(&self) -> usize {
        self.signers.len()
    }

    fn running(&self) -> Vec<usize> {
        (0..self.n()).filter(|&i| self.peers[i].is_some()).collect()
    }

    fn keys(&self) -> Vec<ed25519::PublicKey> {
        self.signers.iter().map(|s| s.public_key()).collect()
    }

    /// Start (or restart) peer `i` on its address with its key, tracking the latest peer set.
    fn start(&mut self, i: usize) {
        assert!(self.peers[i].is_none());
        let context = self
            .context
            .child("peer")
            .with_attribute("index", i)
            .with_attribute("generation", self.generations[i]);
        let (context, process) = context.process("host", |_| false);
        process.set_ip(sockets::ip(i));
        let (sender, mut receiver, mut oracle) = V::start(
            context.child("network"),
            &self.signers,
            i,
            self.bootstrappers,
        );
        let (index, members) = self.sets.last().unwrap();
        V::track(&mut oracle, &self.signers, *index, members);

        // Forward everything received (until the peer crashes) to the cluster
        let inbound = self.inbound_tx.clone();
        context.child("forwarder").spawn(move |_| async move {
            while let Ok((from, message)) = receiver.recv().await {
                let _ = inbound.send((i, from, message.as_ref().to_vec()));
            }
        });
        self.peers[i] = Some(Running {
            sender,
            oracle,
            process,
        });
    }

    /// Crash peer `i` (aborting every task it spawned, which frees its address).
    fn crash(&mut self, i: usize) {
        let peer = self.peers[i].take().expect("peer not running");
        peer.process.crash();
        self.generations[i] += 1;
    }

    /// Track `members` as a new peer set at every running peer.
    fn track(&mut self, members: Vec<usize>) {
        let index = self.sets.last().unwrap().0 + 1;
        for peer in self.peers.iter_mut().flatten() {
            V::track(&mut peer.oracle, &self.signers, index, &members);
        }
        self.sets.push((index, members));
    }

    fn metric(&self, name: &str, i: usize) -> Vec<(String, String)> {
        let encoded = self.context.encode();
        let index = format!("index=\"{i}\"");
        let generation = format!("generation=\"{}\"", self.generations[i]);
        metric_samples(&encoded, name)
            .filter(|(labels, _)| labels.contains(&index) && labels.contains(&generation))
            .map(|(l, v)| (l.to_string(), v.to_string()))
            .collect()
    }

    /// Wait until every running peer can send to every other running peer.
    async fn await_connected(&mut self, deadline: Duration) {
        let start = self.context.current();
        let running = self.running();
        let keys = self.keys();
        loop {
            let mut missing = Vec::new();
            for &i in &running {
                let sender = &mut self.peers[i].as_mut().unwrap().sender;
                let sent = sender.send(Recipients::All, vec![PROBE], false);
                for &j in &running {
                    if j != i && !sent.contains(&keys[j]) {
                        missing.push((i, j));
                    }
                }
            }
            if missing.is_empty() {
                return;
            }
            let elapsed = self.context.current().duration_since(start).unwrap();
            if elapsed >= deadline {
                for &(i, j) in &missing {
                    eprintln!("{}", self.describe(i, j));
                }
                panic!("peers not connected after {elapsed:?}: missing (from, to) = {missing:?}");
            }
            self.context.sleep(Duration::from_millis(100)).await;
        }
    }

    /// Every running peer sends `MESSAGES` tagged messages to all others; assert all arrive.
    async fn assert_delivery(&mut self, tag: u8, deadline: Duration) {
        let running = self.running();
        let keys = self.keys();
        let mut expected = BTreeSet::new();
        for &i in &running {
            for seq in 0..MESSAGES {
                let mut message = vec![tag, i as u8];
                message.extend_from_slice(&seq.to_be_bytes());
                let sender = &mut self.peers[i].as_mut().unwrap().sender;
                let sent = sender.send(Recipients::All, message, false);
                for &j in &running {
                    if j == i {
                        continue;
                    }
                    assert!(
                        sent.contains(&keys[j]),
                        "peer {i} not connected to {j} when sending tag {tag} seq {seq}"
                    );
                    expected.insert((j, i, seq));
                }
            }
        }
        let start = self.context.current();
        let index: HashMap<_, _> = keys.iter().cloned().zip(0..).collect();
        while !expected.is_empty() {
            select! {
                inbound = self.inbound_rx.recv() => {
                    let (to, from, message) = inbound.unwrap();
                    if message.first() != Some(&tag) {
                        continue;
                    }
                    let seq = u32::from_be_bytes(message[2..6].try_into().unwrap());
                    assert_eq!(message[1] as usize, index[&from], "sender mismatch");
                    expected.remove(&(to, index[&from], seq));
                },
                _ = self.context.sleep_until(start + deadline) => {
                    panic!("undelivered (to, from, seq) for tag {tag}: {expected:?}");
                },
            }
        }
    }

    /// Assert each running peer holds exactly one reservation and one connection per other
    /// running peer.
    fn assert_no_leaks(&self) {
        let running = self.running();
        for &i in &running {
            let reserved = self.metric("tracker_directory_reserved", i);
            if reserved.is_empty() {
                let encoded = self.context.encode();
                let all: Vec<_> = encoded.lines().filter(|l| l.contains("reserved")).collect();
                panic!("peer {i}: no reservations metric in {all:?}");
            }
            assert_eq!(
                reserved[0].1,
                (running.len() - 1).to_string(),
                "peer {i} reservations: {reserved:?}"
            );
            let connected = self.metric("tracker_directory_connected", i);
            assert_eq!(connected.len(), running.len() - 1, "peer {i}: {connected:?}");
        }
    }

    /// The metrics peer `i` reports about peer `j`.
    fn describe(&self, i: usize, j: usize) -> String {
        let encoded = self.context.encode();
        let index = format!("index=\"{i}\"");
        let generation = format!("generation=\"{}\"", self.generations[i]);
        let key = self.signers[j].public_key().to_string();
        let lines: Vec<_> = encoded
            .lines()
            .filter(|l| l.contains(&index) && l.contains(&generation) && l.contains(&key))
            .collect();
        format!("peer {i} about {j} ({key}): {lines:#?}")
    }

    fn total_attempts(&self) -> u64 {
        let encoded = self.context.encode();
        metric_samples(&encoded, "dialer_attempts_total")
            .map(|(_, v)| v.parse::<u64>().unwrap())
            .sum()
    }
}

/// A fault campaign derived from a seed.
#[derive(Clone, Debug)]
pub(crate) struct Scenario {
    pub n: usize,
    pub bootstrappers: usize,
    pub latency: Duration,
    pub jitter: Duration,
    pub fault_duration: Duration,
    /// Inject only faults that keep every peer's address and key (no peer set changes).
    pub stable_sets: bool,
}

impl Scenario {
    pub(crate) fn random(seed: u64) -> Self {
        let mut rng = TestRng::new(seed);
        Self {
            n: rng.random_range(3..=9),
            bootstrappers: rng.random_range(1..=2),
            // Links slow enough for a handshake to outlast twice the connection cooldown make
            // simultaneous dials livelock (see `test_dst_simultaneous_dials`)
            latency: Duration::from_millis(rng.random_range(0..=80)),
            jitter: Duration::from_millis(rng.random_range(0..=50)),
            fault_duration: Duration::from_secs(rng.random_range(10..=120)),
            stable_sets: rng.random_bool(0.5),
        }
    }
}

/// One fault injected during a campaign.
#[derive(Debug)]
enum Event {
    CrashRestart(Vec<usize>, Duration),
    Partition(Vec<usize>, Duration),
    Swizzle(u64),
    Lossy(f64, Duration),
    Slow(Duration, Duration),
    Squeeze(u64, Duration),
    Pause(usize, Duration),
    Skew(usize, Duration),
    PeerSet(Vec<usize>),
}

/// Run a scenario: inject faults, heal, then check that every peer reconnects, delivery,
/// leaks, and redialing.
pub(crate) fn run<V: Variant>(seed: u64, scenario: Scenario) {
    let link = LinkConfig::new(scenario.latency, scenario.jitter);
    let testbed = Testbed::new(Topology::sockets(seed).with_default(link));
    let cfg = testbed.install(
        deterministic::Config::default()
            .with_seed(seed)
            .with_timeout(Some(Duration::from_secs(3_600))),
    );
    deterministic::Runner::new(cfg).start(|mut context| async move {
        let n = scenario.n;
        let hosts: Vec<IpAddr> = (0..n).map(sockets::ip).collect();
        let mut cluster =
            Cluster::<V>::new(context.child("cluster"), n, scenario.bootstrappers);
        context.sleep(Duration::from_secs(2)).await;

        let start = context.current();
        let mut events = Vec::new();
        while context.current().duration_since(start).unwrap() < scenario.fault_duration {
            let kinds = if scenario.stable_sets { 8 } else { 9 };
            let event = match context.random_range(0..kinds) {
                0 => {
                    let mut victims: Vec<usize> = (0..n).collect();
                    victims.shuffle(&mut context);
                    victims.truncate(context.random_range(1..=n));
                    let downtime = if context.random_bool(0.3) {
                        Duration::ZERO
                    } else {
                        Duration::from_millis(context.random_range(0..5_000))
                    };
                    Event::CrashRestart(victims, downtime)
                }
                1 => {
                    let mut group: Vec<usize> = (0..n).collect();
                    group.shuffle(&mut context);
                    group.truncate(context.random_range(1..n));
                    Event::Partition(group, Duration::from_millis(context.random_range(500..20_000)))
                }
                2 => Event::Swizzle(context.random()),
                3 => Event::Lossy(
                    context.random_range(0.001..0.2),
                    Duration::from_millis(context.random_range(500..10_000)),
                ),
                4 => Event::Slow(
                    Duration::from_millis(context.random_range(1_000..10_000)),
                    Duration::from_millis(context.random_range(1_000..20_000)),
                ),
                5 => Event::Squeeze(
                    context.random_range(500..50_000),
                    Duration::from_millis(context.random_range(1_000..20_000)),
                ),
                6 => Event::Pause(
                    context.random_range(0..n),
                    Duration::from_millis(context.random_range(100..20_000)),
                ),
                7 => Event::Skew(
                    context.random_range(0..n),
                    Duration::from_millis(context.random_range(0..2_000)),
                ),
                _ => {
                    let mut members: Vec<usize> = (0..n).collect();
                    members.shuffle(&mut context);
                    members.truncate(context.random_range(2..=n));
                    members.sort();
                    Event::PeerSet(members)
                }
            };
            debug!(?event, "injecting");
            match &event {
                Event::CrashRestart(victims, downtime) => {
                    for &victim in victims {
                        cluster.crash(victim);
                    }
                    context.sleep(*downtime).await;
                    for &victim in victims {
                        cluster.start(victim);
                    }
                }
                Event::Partition(group, duration) => {
                    let group: Vec<_> = group.iter().map(|&i| hosts[i]).collect();
                    let rest: Vec<_> =
                        hosts.iter().filter(|h| !group.contains(h)).copied().collect();
                    testbed.partitions().partition(&context, [group, rest]);
                    context.sleep(*duration).await;
                    testbed.partitions().heal(&context);
                }
                Event::Swizzle(seed) => {
                    Swizzle::new(*seed)
                        .run(&context, testbed.partitions(), &hosts)
                        .await;
                }
                Event::Lossy(loss, duration) => {
                    let loss = Probability::from_f64(*loss).unwrap();
                    testbed.topology().set_default(&context, link.with_loss(loss));
                    context.sleep(*duration).await;
                    testbed.topology().set_default(&context, link);
                }
                Event::Slow(latency, duration) => {
                    let slow = LinkConfig::new(*latency, scenario.jitter);
                    testbed.topology().set_default(&context, slow);
                    context.sleep(*duration).await;
                    testbed.topology().set_default(&context, link);
                }
                Event::Squeeze(rate, duration) => {
                    for host in &hosts {
                        testbed.topology().set_bandwidth(
                            &context,
                            *host,
                            Some(Bandwidth::symmetric(*rate)),
                        );
                    }
                    context.sleep(*duration).await;
                    for host in &hosts {
                        testbed.topology().set_bandwidth(&context, *host, None);
                    }
                }
                Event::Pause(victim, duration) => {
                    if let Some(peer) = &cluster.peers[*victim] {
                        peer.process.pause();
                        context.sleep(*duration).await;
                        if let Some(peer) = &cluster.peers[*victim] {
                            peer.process.resume();
                        }
                    }
                }
                Event::Skew(victim, skew) => {
                    if let Some(peer) = &cluster.peers[*victim] {
                        let offset = if context.random_bool(0.5) {
                            ClockOffset::Ahead(*skew)
                        } else {
                            ClockOffset::Behind(*skew)
                        };
                        peer.process.set_clock_offset(offset);
                    }
                }
                Event::PeerSet(members) => cluster.track(members.clone()),
            }
            events.push(event);
            let gap = Duration::from_millis(context.random_range(0..2_000));
            context.sleep(gap).await;
        }

        // Heal: every link, every peer running, and a final peer set of everyone
        testbed.partitions().heal(&context);
        testbed.topology().set_default(&context, link);
        for host in &hosts {
            testbed.topology().set_bandwidth(&context, *host, None);
        }
        for i in 0..n {
            if cluster.peers[i].is_none() {
                cluster.start(i);
            }
        }
        if !scenario.stable_sets {
            cluster.track((0..n).collect());
        }
        debug!(seed, events = events.len(), "healed");

        // Let connections broken by the faults fail (a send on a cut link stalls for the socket
        // timeout before resetting, and bytes queued behind a bandwidth squeeze drain): until
        // then, messages sent on them may be lost
        context.sleep(Duration::from_secs(60)).await;
        cluster.await_connected(Duration::from_secs(120)).await;
        cluster.assert_delivery(0, Duration::from_secs(30)).await;

        // Once settled, nothing should leak or keep redialing
        context.sleep(Duration::from_secs(5)).await;
        cluster.assert_no_leaks();
        let attempts = cluster.total_attempts();
        context.sleep(Duration::from_secs(30)).await;
        assert_eq!(attempts, cluster.total_attempts(), "redialing connected peers");
        cluster.assert_no_leaks();
        cluster.assert_delivery(1, Duration::from_secs(30)).await;
    });
}

/// Run every seed in `seeds` (continuing past failures) and panic listing the failing ones.
pub(crate) fn sweep<V: Variant>(seeds: impl IntoIterator<Item = u64>) {
    let mut failed = Vec::new();
    for seed in seeds {
        let scenario = Scenario::random(seed);
        if std::panic::catch_unwind(|| run::<V>(seed, scenario.clone())).is_err() {
            eprintln!("seed {seed} failed: {scenario:?}");
            failed.push(seed);
        }
    }
    assert!(failed.is_empty(), "failing seeds: {failed:?}");
}

/// The seeds of an environment-driven sweep: `DST_START..DST_END` (defaults `0..100`).
pub(crate) fn env_seeds() -> std::ops::Range<u64> {
    let start: u64 = std::env::var("DST_START").map_or(0, |v| v.parse().unwrap());
    let end: u64 = std::env::var("DST_END").map_or(100, |v| v.parse().unwrap());
    start..end
}

/// Start `n` peers at once over links with `latency` (and no jitter) and require them to connect
/// within `deadline`.
pub(crate) fn connect_simultaneously<V: Variant>(seed: u64, n: usize, latency: Duration, deadline: Duration) {
    let testbed = Testbed::new(
        Topology::sockets(seed).with_default(LinkConfig::new(latency, Duration::ZERO)),
    );
    let cfg = testbed.install(
        deterministic::Config::default()
            .with_seed(seed)
            .with_timeout(Some(Duration::from_secs(3_600))),
    );
    deterministic::Runner::new(cfg).start(|context| async move {
        let mut cluster = Cluster::<V>::new(context.child("cluster"), n, n);
        cluster.await_connected(deadline).await;
    });
}

/// How long `n` peers that start at uniformly random times within `spread` (all at once if
/// zero) over links of one-way `latency` take, from the last start, until every peer can send
/// to every other, or `None` if they are not connected by `deadline`.
pub(crate) fn time_to_connect<V: Variant>(
    seed: u64,
    n: usize,
    latency: Duration,
    spread: Duration,
    deadline: Duration,
) -> Option<Duration> {
    let testbed = Testbed::new(
        Topology::sockets(seed).with_default(LinkConfig::new(latency, Duration::ZERO)),
    );
    let cfg = testbed.install(
        deterministic::Config::default()
            .with_seed(seed)
            .with_timeout(Some(deadline * 2 + Duration::from_secs(60))),
    );
    deterministic::Runner::new(cfg).start(|mut context| async move {
        let mut cluster = Cluster::<V>::stopped(context.child("cluster"), n, n);
        let start = context.current();
        let mut offsets: Vec<(Duration, usize)> = (0..n)
            .map(|i| {
                let offset = if spread.is_zero() {
                    Duration::ZERO
                } else {
                    Duration::from_millis(context.random_range(0..spread.as_millis() as u64))
                };
                (offset, i)
            })
            .collect();
        offsets.sort();
        for (offset, i) in offsets {
            context.sleep_until(start + offset).await;
            cluster.start(i);
        }
        let started = context.current();
        let keys = cluster.keys();
        loop {
            let mut connected = true;
            for i in 0..n {
                let sender = &mut cluster.peers[i].as_mut().unwrap().sender;
                let sent = sender.send(Recipients::All, vec![PROBE], false);
                if (0..n).any(|j| j != i && !sent.contains(&keys[j])) {
                    connected = false;
                }
            }
            let elapsed = context.current().duration_since(started).unwrap();
            if connected {
                return Some(elapsed);
            }
            if elapsed >= deadline {
                return None;
            }
            context.sleep(Duration::from_millis(50)).await;
        }
    })
}

/// Summarize [time_to_connect] over `seeds` as `(median, p90, max, unconnected)` (in seconds;
/// unconnected runs count as `deadline` in the percentiles).
pub(crate) fn connect_stats<V: Variant>(
    seeds: std::ops::Range<u64>,
    n: usize,
    latency: Duration,
    spread: Duration,
    deadline: Duration,
) -> (f64, f64, f64, usize) {
    let mut times: Vec<f64> = Vec::new();
    let mut unconnected = 0;
    for seed in seeds {
        match time_to_connect::<V>(seed, n, latency, spread, deadline) {
            Some(t) => times.push(t.as_secs_f64()),
            None => {
                unconnected += 1;
                times.push(deadline.as_secs_f64());
            }
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f64| times[((times.len() - 1) as f64 * q).round() as usize];
    (at(0.5), at(0.9), *times.last().unwrap(), unconnected)
}
