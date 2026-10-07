//! Latency, jitter, loss, and bandwidth between the hosts of a simulated network.
//!
//! [Topology] is a [NetworkPolicy] configured declaratively: a default [LinkConfig], per-link
//! overrides between hosts, and per-host egress and ingress [Bandwidth]. Like
//! [super::Partitions], it drives both the runtime's deterministic sockets
//! ([Topology::sockets], where each IP is a host) and message-level simulated networks keyed by
//! peer identity ([Topology::new]), and it can be changed while the simulation runs. Wrap it in
//! [super::Partitions] (through an [Arc], to keep a handle to both) to also cut and clog links.

use super::{Context, NetworkDelivery, NetworkPolicy, NetworkTransmission};
use commonware_utils::{Probability, sync::Mutex};
use rand::{RngExt as _, SeedableRng, rngs::StdRng};
use std::{
    collections::BTreeMap,
    fmt::Debug,
    net::{IpAddr, SocketAddr},
    time::{Duration, SystemTime},
};

/// The latency, jitter, and loss of one direction of a link.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkConfig {
    /// Fixed delay of every transmission.
    pub latency: Duration,
    /// Extra delay of each transmission, drawn uniformly from `0..=jitter`.
    pub jitter: Duration,
    /// Probability that a transmission is lost (on sockets, a loss resets the connection).
    pub loss: Probability,
}

impl LinkConfig {
    /// A perfect link.
    pub const PERFECT: Self = Self::new(Duration::ZERO, Duration::ZERO);

    /// A lossless link with `latency` and up to `jitter` of extra delay.
    pub const fn new(latency: Duration, jitter: Duration) -> Self {
        Self {
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

impl Default for LinkConfig {
    fn default() -> Self {
        Self::PERFECT
    }
}

/// The egress and ingress capacity of a host, in bytes per second (`None` is unlimited, and a
/// capacity of zero is treated as one byte per second).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bandwidth {
    /// Bytes per second the host can send.
    pub egress: Option<u64>,
    /// Bytes per second the host can receive.
    pub ingress: Option<u64>,
}

impl Bandwidth {
    /// Unlimited capacity.
    pub const UNLIMITED: Self = Self {
        egress: None,
        ingress: None,
    };

    /// `bytes_per_second` of capacity in each direction.
    pub const fn symmetric(bytes_per_second: u64) -> Self {
        Self {
            egress: Some(bytes_per_second),
            ingress: Some(bytes_per_second),
        }
    }
}

/// How long `len` bytes take at `rate` bytes per second.
fn transfer(len: usize, rate: Option<u64>) -> Duration {
    let Some(rate) = rate else {
        return Duration::ZERO;
    };
    let nanos = (len as u128)
        .saturating_mul(1_000_000_000)
        .div_ceil(u128::from(rate.max(1)));
    Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
}

/// Configuration and bandwidth accounting of a [Topology].
struct State<H> {
    default: LinkConfig,
    links: BTreeMap<(H, H), LinkConfig>,
    default_bandwidth: Bandwidth,
    bandwidth: BTreeMap<H, Bandwidth>,
    /// When each host's egress is next idle.
    egress: BTreeMap<H, SystemTime>,
    /// When each host's ingress is next idle.
    ingress: BTreeMap<H, SystemTime>,
}

/// A network policy that delays and loses transmissions according to the links between hosts
/// and the bandwidth of each host.
///
/// Each endpoint `A` belongs to the host `H` that a mapping assigns it (a peer is its own host; a
/// socket address belongs to its IP). Traffic between endpoints of the same host is delivered
/// immediately and never lost. Every other transmission travels the directed link between its
/// hosts, configured with [Self::with_link] (or [Self::set_link] while running), or the default
/// link otherwise, and:
///
/// - Is lost with the link's loss probability. A lost transmission still uses its sender's egress
///   capacity. On sockets, a loss resets the connection.
/// - Waits for its sender's egress to be idle, then takes `len / egress` to leave it. Each host
///   has one egress queue shared by all its links, served in the order transmissions are sent.
/// - Propagates for the link's latency plus a jitter draw.
/// - Waits for its receiver's ingress to be idle, then takes `len / ingress` to enter it (at least
///   until its last byte has propagated). Each host has one ingress queue, served in arrival
///   order.
///
/// A transmission's delivery delay covers all of the above. On sockets, a send completes once
/// queued (a delay is propagation, not blocking), so a host that sends faster than its egress
/// accumulates delay until its connections' windows fill and its sends block.
///
/// Random draws (jitter, then loss when nonzero) come from an RNG seeded at construction, so a
/// run is reproducible from its seed. Every change made while running is recorded by the
/// [super::Auditor].
pub struct Topology<A, H = A> {
    host: fn(&A) -> H,
    rng: Mutex<StdRng>,
    state: Mutex<State<H>>,
}

impl<A: Ord + Clone + Debug + Send> Topology<A> {
    /// A topology for a message-level network on which each endpoint is its own host, drawing
    /// from an RNG seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self::with_hosts(seed, A::clone)
    }
}

impl Topology<SocketAddr, IpAddr> {
    /// A topology for the runtime's deterministic sockets, on which each IP is a host, drawing
    /// from an RNG seeded with `seed`.
    ///
    /// Give each simulated host its own IP: it listens on that IP, and its dials originate from
    /// it once its process is assigned it with [super::Process::set_ip].
    pub fn sockets(seed: u64) -> Self {
        Self::with_hosts(seed, SocketAddr::ip)
    }
}

impl<A, H: Ord + Clone + Debug + Send> Topology<A, H> {
    /// A topology that assigns each endpoint the host `host` maps it to, drawing from an RNG
    /// seeded with `seed`.
    pub fn with_hosts(seed: u64, host: fn(&A) -> H) -> Self {
        Self {
            host,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
            state: Mutex::new(State {
                default: LinkConfig::PERFECT,
                links: BTreeMap::new(),
                default_bandwidth: Bandwidth::UNLIMITED,
                bandwidth: BTreeMap::new(),
                egress: BTreeMap::new(),
                ingress: BTreeMap::new(),
            }),
        }
    }

    /// Use `link` for every link without its own configuration.
    pub fn with_default(self, link: LinkConfig) -> Self {
        self.state.lock().default = link;
        self
    }

    /// Use `link` from `from` to `to` (one direction).
    pub fn with_link(self, from: H, to: H, link: LinkConfig) -> Self {
        self.state.lock().links.insert((from, to), link);
        self
    }

    /// Give every host without its own bandwidth `bandwidth`.
    pub fn with_default_bandwidth(self, bandwidth: Bandwidth) -> Self {
        self.state.lock().default_bandwidth = bandwidth;
        self
    }

    /// Give `host` `bandwidth`.
    pub fn with_bandwidth(self, host: H, bandwidth: Bandwidth) -> Self {
        self.state.lock().bandwidth.insert(host, bandwidth);
        self
    }

    /// Use `link` from `from` to `to` (one direction) from now on, or the default link if `None`.
    pub fn set_link(&self, context: &Context, from: H, to: H, link: Option<LinkConfig>) {
        context.auditor().event(b"topology_link", |hasher| {
            hasher.update(format!("{from:?}->{to:?}={link:?}"));
        });
        let mut state = self.state.lock();
        match link {
            Some(link) => state.links.insert((from, to), link),
            None => state.links.remove(&(from, to)),
        };
    }

    /// Use `link` for every link without its own configuration from now on.
    pub fn set_default(&self, context: &Context, link: LinkConfig) {
        context.auditor().event(b"topology_default", |hasher| {
            hasher.update(format!("{link:?}"));
        });
        self.state.lock().default = link;
    }

    /// Give `host` `bandwidth` from now on, or the default bandwidth if `None`. Transmissions
    /// already queued keep their timing.
    pub fn set_bandwidth(&self, context: &Context, host: H, bandwidth: Option<Bandwidth>) {
        context.auditor().event(b"topology_bandwidth", |hasher| {
            hasher.update(format!("{host:?}={bandwidth:?}"));
        });
        let mut state = self.state.lock();
        match bandwidth {
            Some(bandwidth) => state.bandwidth.insert(host, bandwidth),
            None => state.bandwidth.remove(&host),
        };
    }

    /// The link from `from` to `to`.
    pub fn link(&self, from: &H, to: &H) -> LinkConfig {
        let state = self.state.lock();
        state
            .links
            .get(&(from.clone(), to.clone()))
            .copied()
            .unwrap_or(state.default)
    }

    /// The bandwidth of `host`.
    pub fn bandwidth(&self, host: &H) -> Bandwidth {
        let state = self.state.lock();
        state
            .bandwidth
            .get(host)
            .copied()
            .unwrap_or(state.default_bandwidth)
    }
}

impl<A, H> NetworkPolicy<A> for Topology<A, H>
where
    H: Ord + Clone + Debug + Send,
{
    fn delivers(&self, transmission: &NetworkTransmission<'_, A>) -> NetworkDelivery {
        let from = (self.host)(&transmission.link.from);
        let to = (self.host)(&transmission.link.to);
        if from == to {
            return NetworkDelivery::NOW;
        }
        let at = transmission.at;
        let mut state = self.state.lock();
        let link = state
            .links
            .get(&(from.clone(), to.clone()))
            .copied()
            .unwrap_or(state.default);

        // Draw jitter, then loss
        let (jitter, lost) = {
            let mut rng = self.rng.lock();
            let jitter = if link.jitter.is_zero() {
                Duration::ZERO
            } else {
                let max = u64::try_from(link.jitter.as_nanos()).unwrap_or(u64::MAX);
                Duration::from_nanos(rng.random_range(0..=max))
            };
            let lost = !link.loss.is_zero() && link.loss.sample(&mut *rng);
            (jitter, lost)
        };

        // Leave the sender's egress
        let egress = state
            .bandwidth
            .get(&from)
            .copied()
            .unwrap_or(state.default_bandwidth)
            .egress;
        let sent = state.egress.get(&from).copied().map_or(at, |idle| idle.max(at));
        let left = sent
            .checked_add(transfer(transmission.len, egress))
            .expect("egress overflow");
        if egress.is_some() {
            state.egress.insert(from, left);
        }
        if lost {
            return NetworkDelivery::DROP;
        }

        // Propagate, then enter the receiver's ingress
        let propagation = link.latency.checked_add(jitter).expect("latency overflow");
        let ingress = state
            .bandwidth
            .get(&to)
            .copied()
            .unwrap_or(state.default_bandwidth)
            .ingress;
        let first = sent.checked_add(propagation).expect("latency overflow");
        let last = left.checked_add(propagation).expect("latency overflow");
        let arrived = match ingress {
            None => last,
            Some(_) => {
                let start = state
                    .ingress
                    .get(&to)
                    .copied()
                    .map_or(first, |idle| idle.max(first));
                let arrived = start
                    .checked_add(transfer(transmission.len, ingress))
                    .expect("ingress overflow")
                    .max(last);
                state.ingress.insert(to, arrived);
                arrived
            }
        };
        NetworkDelivery::after(arrived.duration_since(at).unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Clock as _, Listener as _, Network as _, Runner as _, Sink as _, Spawner as _,
        Stream as _,
        deterministic::{self, NetworkFate, NetworkLink, Partitions},
    };
    use std::{net::Ipv4Addr, sync::Arc};

    fn host(i: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(10, 0, 0, i))
    }

    fn address(i: u8) -> SocketAddr {
        SocketAddr::new(host(i), 3000)
    }

    fn transmission(link: &NetworkLink<u8>, len: usize, at: SystemTime) -> NetworkTransmission<'_, u8> {
        NetworkTransmission {
            link,
            channel: None,
            index: 0,
            len,
            at,
        }
    }

    #[test]
    fn test_links_and_defaults() {
        let topology = Topology::new(0)
            .with_default(LinkConfig::new(Duration::from_millis(10), Duration::ZERO))
            .with_link(1u8, 2, LinkConfig::new(Duration::from_millis(50), Duration::ZERO))
            .with_link(
                2,
                3,
                LinkConfig::PERFECT.with_loss(commonware_utils::probability!(1.0)),
            );
        let at = SystemTime::UNIX_EPOCH;
        let delay = |from, to| {
            topology.delivers(&transmission(&NetworkLink { from, to }, 1, at))
        };
        assert_eq!(delay(1, 2).after, Duration::from_millis(50));
        assert_eq!(delay(2, 1).after, Duration::from_millis(10), "links are directed");
        assert_eq!(delay(1, 1), NetworkDelivery::NOW, "same host");
        assert_eq!(delay(2, 3).fate, NetworkFate::Drop);
    }

    #[test]
    fn test_jitter_is_seeded_and_bounded() {
        let draw = |seed| {
            let topology = Topology::new(seed).with_default(LinkConfig::new(
                Duration::from_millis(10),
                Duration::from_millis(5),
            ));
            let link = NetworkLink { from: 1u8, to: 2 };
            (0..32)
                .map(|_| {
                    topology
                        .delivers(&transmission(&link, 1, SystemTime::UNIX_EPOCH))
                        .after
                })
                .collect::<Vec<_>>()
        };
        let delays = draw(3);
        assert_eq!(delays, draw(3));
        assert_ne!(delays, draw(4));
        for delay in delays {
            assert!(delay >= Duration::from_millis(10) && delay <= Duration::from_millis(15));
        }
    }

    #[test]
    fn test_bandwidth_queues_egress_and_ingress() {
        // 1 KB/s egress at host 1, 500 B/s ingress at host 3, 100ms latency
        let topology = Topology::new(0)
            .with_default(LinkConfig::new(Duration::from_millis(100), Duration::ZERO))
            .with_bandwidth(
                1u8,
                Bandwidth {
                    egress: Some(1_000),
                    ingress: None,
                },
            )
            .with_bandwidth(
                3,
                Bandwidth {
                    egress: None,
                    ingress: Some(500),
                },
            );
        let at = SystemTime::UNIX_EPOCH;
        let send = |from, to, len| {
            topology
                .delivers(&transmission(&NetworkLink { from, to }, len, at))
                .after
        };
        // 1000 bytes take a second to leave host 1, then 100ms to propagate
        assert_eq!(send(1, 2, 1_000), Duration::from_millis(1_100));
        // The next transmission (on another link) queues behind the first
        assert_eq!(send(1, 2, 500), Duration::from_millis(1_600));
        // Host 2 is unlimited; host 3's ingress takes two seconds per 1000 bytes
        assert_eq!(send(2, 3, 1_000), Duration::from_millis(2_100));
        assert_eq!(send(4, 3, 1_000), Duration::from_millis(4_100));
        assert_eq!(send(2, 4, 1_000), Duration::from_millis(100));
    }

    #[test]
    fn test_sockets_carry_latency_and_bandwidth() {
        let topology = Arc::new(
            Topology::sockets(0)
                .with_default(LinkConfig::new(Duration::from_millis(50), Duration::ZERO))
                .with_bandwidth(host(1), Bandwidth::symmetric(10_000)),
        );
        let partitions = Arc::new(Partitions::sockets(topology.clone()));
        let cfg = deterministic::Config::default().with_network_policy(partitions.clone());
        deterministic::Runner::new(cfg).start(|context| async move {
            let (node, process) = context.process("node", |_| false);
            process.set_ip(host(1));
            let mut listener = context.bind(address(2)).await.unwrap();
            let (mut sink, _) = node.dial(address(2)).await.unwrap();
            let (_, _, mut stream) = listener.accept().await.unwrap();

            // 10 sends of 1000 bytes leave over a second (sends return at once), each arriving
            // 50ms after its last byte leaves
            let start = context.current();
            for _ in 0..10 {
                sink.send(vec![0u8; 1_000]).await.unwrap();
            }
            assert_eq!(context.current(), start);
            stream.recv(1_000).await.unwrap();
            assert_eq!(
                context.current().duration_since(start).unwrap(),
                Duration::from_millis(150)
            );
            stream.recv(9_000).await.unwrap();
            assert_eq!(
                context.current().duration_since(start).unwrap(),
                Duration::from_millis(1_050)
            );

            // Slow the link while running
            topology.set_link(
                &context,
                host(1),
                host(2),
                Some(LinkConfig::new(Duration::from_secs(1), Duration::ZERO)),
            );
            let start = context.current();
            sink.send(vec![1u8; 10]).await.unwrap();
            stream.recv(10).await.unwrap();
            assert_eq!(
                context.current().duration_since(start).unwrap(),
                Duration::from_millis(1_001)
            );

            // Partitions still apply on top
            partitions.cut(&context, host(1), host(2));
            let sender = node.spawn(move |_| async move { sink.send(vec![2u8]).await });
            assert!(stream.recv(1).await.is_err());
            assert!(sender.await.unwrap().is_err());
        });
    }
}
