//! Run the production authenticated network over the deterministic runtime's sockets.
//!
//! The rest of [crate::simulated] simulates what happens behind the [crate::Sender] and
//! [crate::Receiver] interfaces. This module instead runs [crate::authenticated::lookup] itself
//! (handshakes, encryption, dialing, rate limiting, and peer sets) over the simulated sockets of
//! [commonware_runtime::deterministic], so a test covers the p2p stack a production application
//! uses. A [Testbed] configures those sockets: latency, jitter, loss, and bandwidth between hosts
//! (with a [Topology]) and partitions and clogs (with [Partitions]).
//!
//! Each peer is a simulated host: a [deterministic::Process] with its own IP ([ip]), listening on
//! [address]. Crash a peer with [Peer::process] and restart it by starting a new [Peer] with the
//! same index and key (and storage partitions).
//!
//! # Example
//!
//! ```rust
//! use commonware_cryptography::{ed25519, Signer as _};
//! use commonware_p2p::{AddressableManager as _, Recipients, Receiver as _, Sender as _};
//! use commonware_p2p::simulated::sockets::{self, Peer, Testbed};
//! use commonware_runtime::{deterministic, Quota, Runner as _};
//! use commonware_runtime::deterministic::{LinkConfig, Topology};
//! use commonware_utils::{NZU32, NZUsize};
//! use std::time::Duration;
//!
//! let testbed = Testbed::new(
//!     Topology::sockets(0).with_default(LinkConfig::new(Duration::from_millis(50), Duration::ZERO)),
//! );
//! let cfg = testbed.install(deterministic::Config::default().with_seed(0));
//! deterministic::Runner::new(cfg).start(|context| async move {
//!     let signers: Vec<_> = (0..2).map(ed25519::PrivateKey::from_seed).collect();
//!     let peers = sockets::peers(signers.iter().map(|signer| signer.public_key()));
//!     let mut channels = Vec::new();
//!     for (i, signer) in signers.into_iter().enumerate() {
//!         let config = sockets::config(signer, i, b"example", NZUsize!(2), 1024);
//!         let (mut peer, mut network) = Peer::start(&context, "peer", |_| false, config);
//!         peer.oracle.track(0, peers.clone());
//!         channels.push(network.register(0, Quota::per_second(NZU32!(100))));
//!         network.start();
//!     }
//!     let (mut sender, _) = channels.remove(0);
//!     let (_, mut receiver) = channels.remove(0);
//!     loop {
//!         if !sender.send(Recipients::All, &b"hello"[..], false).is_empty() {
//!             break;
//!         }
//!         commonware_runtime::Clock::sleep(&context, Duration::from_millis(100)).await;
//!     }
//!     let (_, message) = receiver.recv().await.unwrap();
//!     assert_eq!(message.as_ref(), b"hello");
//! });
//! ```

use crate::{
    Address,
    authenticated::lookup::{self, Oracle},
};
use commonware_cryptography::{ChaCha20Poly1305, Signer};
use commonware_runtime::{
    Quota, Supervisor as _,
    deterministic::{self, NetworkPolicy, Partitions, Topology},
};
use commonware_stream::{
    SakeCups,
    cups::{self, Cups},
    sake::{self, Sake},
};
use commonware_utils::{NZU32, ordered::Map};
use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    num::NonZeroUsize,
    sync::Arc,
    time::Duration,
};

/// The port every peer listens on (at its own [ip]).
pub const PORT: u16 = 3000;

/// The IP of peer `index`: `10.a.b.1`, where `a.b` is the index in base 256, so every peer is in
/// its own `/24` subnet (and per-subnet rate limits apply per peer).
///
/// # Panics
///
/// Panics if `index` does not fit in 16 bits.
pub fn ip(index: usize) -> IpAddr {
    let index = u16::try_from(index).expect("peer index exceeds 16 bits");
    let [a, b] = index.to_be_bytes();
    IpAddr::V4(Ipv4Addr::new(10, a, b, 1))
}

/// The address peer `index` listens on.
pub fn address(index: usize) -> SocketAddr {
    SocketAddr::new(ip(index), PORT)
}

/// The peer set of `keys`, each at the [address] of its position.
///
/// # Panics
///
/// Panics if `keys` contains duplicates.
pub fn peers<P: commonware_cryptography::PublicKey>(
    keys: impl IntoIterator<Item = P>,
) -> Map<P, Address> {
    let peers: Vec<(P, Address)> = keys
        .into_iter()
        .enumerate()
        .map(|(i, key)| (key, address(i).into()))
        .collect();
    Map::try_from(peers).expect("duplicate peer keys")
}

/// The handshake of a peer signing with `C`.
pub type Handshake<C> = SakeCups<C, ChaCha20Poly1305>;

/// The authenticated network of a peer signing with `C`.
pub type Network<C> = lookup::Network<deterministic::Context, Handshake<C>>;

/// The sending half of a channel registered with a [Network].
pub type Sender<P> = lookup::Sender<P, deterministic::Context>;

/// The receiving half of a channel registered with a [Network].
pub type Receiver<P> = lookup::Receiver<P>;

/// A configuration for peer `index` signing with `signer`, listening on [address]`(index)`.
///
/// Starts from [lookup::Config::local] and shortens cooldowns, pings, and dial intervals (as
/// tests do), and permits handshake rates high enough that restarts are never throttled for
/// long. Adjust the returned fields to test other settings.
pub fn config<C: Signer>(
    signer: C,
    index: usize,
    namespace: &[u8],
    max_peers_per_set: NonZeroUsize,
    max_message_size: u32,
) -> lookup::Config<Handshake<C>> {
    let handshake = Cups::<_, ChaCha20Poly1305>::new(
        Sake {
            signer,
            synchrony_bound: Duration::from_secs(5),
            max_handshake_age: Duration::from_secs(10),
            version: sake::Version::V1,
        },
        cups::Version::V1,
    );
    let mut config = lookup::Config::local(
        handshake,
        namespace,
        address(index),
        max_peers_per_set,
        max_message_size,
    );
    config.peer_connection_cooldown = Duration::from_millis(250);
    config.allowed_handshake_rate_per_ip = Quota::per_second(NZU32!(128));
    config.allowed_handshake_rate_per_subnet = Quota::per_second(NZU32!(256));
    config.ping_frequency = Duration::from_secs(1);
    config.dial_frequency = Duration::from_millis(200);
    config.block_duration = Duration::from_mins(1);
    config
}

/// The simulated sockets peers communicate over: a [Topology] (latency, jitter, loss, and
/// bandwidth between hosts) inside [Partitions] (cuts and clogs).
///
/// Install it in the runtime's configuration with [Self::install], then change either while
/// the simulation runs.
pub struct Testbed {
    topology: Arc<Topology<SocketAddr, IpAddr>>,
    partitions: Arc<Partitions<SocketAddr, IpAddr>>,
}

impl Testbed {
    /// Simulated sockets shaped by `topology`.
    pub fn new(topology: Topology<SocketAddr, IpAddr>) -> Self {
        let topology = Arc::new(topology);
        let partitions = Arc::new(Partitions::sockets(topology.clone()));
        Self {
            topology,
            partitions,
        }
    }

    /// Install the testbed's policy in `config`.
    pub fn install(&self, config: deterministic::Config) -> deterministic::Config {
        config.with_network_policy(self.policy())
    }

    /// The testbed's policy, for [deterministic::Config::with_network_policy].
    pub fn policy(&self) -> Arc<dyn NetworkPolicy> {
        self.partitions.clone()
    }

    /// The links and bandwidth between hosts.
    pub fn topology(&self) -> &Topology<SocketAddr, IpAddr> {
        &self.topology
    }

    /// The partitions and clogs between hosts.
    pub fn partitions(&self) -> &Partitions<SocketAddr, IpAddr> {
        &self.partitions
    }
}

/// A peer running the authenticated network in its own simulated process.
pub struct Peer<C: Signer> {
    /// The peer's context (in its process): start the peer's application from it.
    pub context: deterministic::Context,
    /// The peer's process: crash it to crash the peer (and its network).
    pub process: deterministic::Process,
    /// The peer's oracle: track peer sets and block peers.
    pub oracle: Oracle<C::PublicKey>,
}

impl<C: Signer> Peer<C> {
    /// Start a process for a peer, labeled `label`, that owns the storage partitions
    /// `partitions` selects, with the IP `config` listens on, and create its network (register
    /// channels with it, then start it).
    pub fn start(
        context: &deterministic::Context,
        label: &'static str,
        partitions: impl Fn(&str) -> bool + Send + Sync + 'static,
        config: lookup::Config<Handshake<C>>,
    ) -> (Self, Network<C>) {
        let (context, process) = context.process(label, partitions);
        process.set_ip(config.listen.ip());
        let (network, oracle) = lookup::Network::new(context.child("network"), config);
        (
            Self {
                context,
                process,
                oracle,
            },
            network,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AddressableManager as _, Receiver as _, Recipients, Sender as _};
    use commonware_cryptography::ed25519;
    use commonware_runtime::{
        Clock as _, Runner as _,
        deterministic::{Bandwidth, LinkConfig},
    };
    use commonware_utils::NZUsize;
    use std::collections::BTreeSet;

    /// A started peer: its index, handle, and channel.
    type Running = (
        usize,
        Peer<ed25519::PrivateKey>,
        Sender<ed25519::PublicKey>,
        Receiver<ed25519::PublicKey>,
    );

    fn signers(n: usize) -> Vec<ed25519::PrivateKey> {
        (0..n as u64).map(ed25519::PrivateKey::from_seed).collect()
    }

    /// Start `n` peers with one channel each, returning their senders and receivers.
    fn start(
        context: &deterministic::Context,
        signers: &[ed25519::PrivateKey],
        indices: impl IntoIterator<Item = usize>,
    ) -> Vec<Running> {
        let peers = super::peers(signers.iter().map(|s| s.public_key()));
        indices
            .into_iter()
            .map(|i| {
                let config = config(
                    signers[i].clone(),
                    i,
                    b"test",
                    NZUsize!(signers.len()),
                    1024 * 1024,
                );
                let (mut peer, mut network) = Peer::start(context, "peer", |_| false, config);
                peer.oracle.track(0, peers.clone());
                let (sender, receiver) = network.register(0, Quota::per_second(NZU32!(10_000)));
                network.start();
                (i, peer, sender, receiver)
            })
            .collect()
    }

    /// Wait until every sender reaches every other peer.
    async fn connected(
        context: &deterministic::Context,
        peers: &mut [Running],
    ) {
        let n = peers.len();
        loop {
            let mut ready = true;
            for (_, _, sender, _) in peers.iter_mut() {
                if sender.send(Recipients::All, &[][..], false).len() + 1 < n {
                    ready = false;
                }
            }
            if ready {
                return;
            }
            context.sleep(Duration::from_millis(100)).await;
        }
    }

    #[test]
    fn test_many_peers_connect_and_exchange() {
        // Each peer dials every other: more connections than one IP's ephemeral port range
        // would allow before ports were allocated per destination.
        let n = 60;
        let testbed = Testbed::new(
            Topology::sockets(7)
                .with_default(LinkConfig::new(
                    Duration::from_millis(40),
                    Duration::from_millis(20),
                ))
                .with_default_bandwidth(Bandwidth::symmetric(10_000_000)),
        );
        let cfg = testbed.install(
            deterministic::Config::default()
                .with_seed(7)
                .with_timeout(Some(Duration::from_secs(600))),
        );
        deterministic::Runner::new(cfg).start(|context| async move {
            let signers = signers(n);
            let mut peers = start(&context, &signers, 0..n);
            connected(&context, &mut peers).await;

            // Everyone sends one message to everyone
            for (i, _, sender, _) in peers.iter_mut() {
                let sent = sender.send(Recipients::All, vec![*i as u8], false);
                assert_eq!(sent.len(), n - 1);
            }
            for (i, _, _, receiver) in peers.iter_mut() {
                let mut from = BTreeSet::new();
                while from.len() < n - 1 {
                    let (_, message) = receiver.recv().await.unwrap();
                    if message.as_ref().is_empty() {
                        continue;
                    }
                    assert_ne!(message.as_ref()[0] as usize, *i);
                    from.insert(message.as_ref()[0]);
                }
            }
        });
    }

    #[test]
    fn test_partition_heal_and_restart() {
        let n = 4;
        let testbed = Testbed::new(
            Topology::sockets(1)
                .with_default(LinkConfig::new(Duration::from_millis(10), Duration::ZERO)),
        );
        let cfg = testbed.install(
            deterministic::Config::default()
                .with_seed(1)
                .with_timeout(Some(Duration::from_secs(600))),
        );
        deterministic::Runner::new(cfg).start(|context| async move {
            let signers = signers(n);
            let mut peers = start(&context, &signers, 0..n);
            connected(&context, &mut peers).await;

            // Isolate peer 0: its sends stop reaching anyone once connections time out
            testbed
                .partitions()
                .partition(&context, [vec![ip(0)], (1..n).map(ip).collect()]);
            context.sleep(Duration::from_secs(30)).await;
            let (_, _, sender, _) = &mut peers[0];
            assert!(sender.send(Recipients::All, &[1u8][..], false).is_empty());

            // Heal, crash peer 1, and restart it on the same address with the same key
            testbed.partitions().heal(&context);
            let (_, peer, _, _) = peers.remove(1);
            peer.process.crash();
            context.sleep(Duration::from_secs(1)).await;
            let restarted = start(&context, &signers, [1]);
            peers.splice(1..1, restarted);
            connected(&context, &mut peers).await;
        });
    }
}
