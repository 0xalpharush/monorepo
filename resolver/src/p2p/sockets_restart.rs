//! A requester that restarts while a peer is still serving its request, over authenticated
//! lookup on simulated sockets.
//!
//! Each peer runs the resolver over the production authenticated network in its own simulated
//! process (with its own IP and key). The requester asks a slow peer for a key, crashes, restarts
//! on the same key and IP, reconnects, and asks for another key. If the slow peer's response to
//! the pre-crash request reaches the restarted requester, it carries a request id the restarted
//! requester reuses for its new request.

use super::{
    Config, Engine,
    mocks::{Consumer, Key},
};
use crate::Resolver as _;
use bytes::Bytes;
use commonware_cryptography::{
    Signer as _,
    ed25519::{PrivateKey, PublicKey},
};
use commonware_macros::{select, test_traced};
use commonware_p2p::{
    AddressableManager as _,
    authenticated::lookup,
    simulated::sockets::{self, Peer, Testbed},
};
use commonware_runtime::{
    Clock as _, Metrics as _, Quota, Runner as _, Spawner as _, Supervisor as _,
    deterministic::{self, LinkConfig, Topology},
};
use commonware_utils::{
    NZU32, NZUsize,
    channel::{fallible::OneshotExt as _, mpsc, oneshot},
};
use std::{collections::BTreeSet, sync::Arc, time::Duration};

/// A producer that answers each key after a delay, reporting each request it receives.
#[derive(Clone)]
struct SlowProducer {
    context: Arc<deterministic::Context>,
    delay: Duration,
    requested: mpsc::UnboundedSender<Key>,
}

impl crate::p2p::Producer for SlowProducer {
    type Key = Key;

    fn produce(&mut self, key: Key) -> oneshot::Receiver<Bytes> {
        let (sender, receiver) = oneshot::channel();
        let _ = self.requested.send(key.clone());
        let delay = self.delay;
        self.context
            .child("produce")
            .spawn(move |context| async move {
                context.sleep(delay).await;
                sender.send_lossy(value(&key));
            });
        receiver
    }
}

/// The restarted requester fetches keys `2..=KEYS` (so several request ids are in flight).
const KEYS: u8 = 5;

const NAMESPACE: &[u8] = b"resolver_restart";

fn value(key: &Key) -> Bytes {
    Bytes::from(format!("value of key {}", key.0))
}

/// What the restarted requester observed.
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    /// The restarted requester blocked the slow peer.
    blocked: bool,
    /// The restarted requester received every key it asked for.
    delivered: bool,
}

/// Start the resolver of peer `index` (of 2) over authenticated lookup, serving `producer`
/// and accepting `expected` values.
fn start<Pro: crate::p2p::Producer<Key = Key>>(
    context: &deterministic::Context,
    signers: &[PrivateKey],
    index: usize,
    recommended: bool,
    producer: impl FnOnce(&deterministic::Context) -> Pro,
    consumer: Consumer<Key, Bytes>,
) -> (super::Mailbox<Key, PublicKey, ()>, Peer<PrivateKey>) {
    let n = NZUsize!(signers.len());
    let mut config = sockets::config(signers[index].clone(), index, NAMESPACE, n, 1024 * 1024);
    if recommended {
        config = lookup::Config::recommended(
            config.handshake,
            NAMESPACE,
            sockets::address(index),
            n,
            1024 * 1024,
        );
        // Simulated hosts have private IPs
        config.allow_private_ips = true;
    }
    config.tracked_peer_sets = NZUsize!(1);
    let (mut peer, mut network) = Peer::start(context, "host", |_| false, config);
    peer.oracle
        .track(0, sockets::peers(signers.iter().map(|s| s.public_key())));
    let connection = network.register(0, Quota::per_second(NZU32!(256)));
    network.start();
    let (engine, mailbox) = Engine::new(
        peer.context.child("resolver"),
        Config {
            peer_provider: peer.oracle.clone(),
            blocker: peer.oracle.clone(),
            consumer,
            producer: producer(&peer.context),
            mailbox_size: NZUsize!(1024),
            me: Some(signers[index].public_key()),
            timeout: Duration::from_secs(5),
            fetch_retry_timeout: Duration::from_millis(500),
            priority_requests: false,
            priority_responses: false,
        },
    );
    engine.start(connection);
    (mailbox, peer)
}

/// Peer 0 asks peer 1 (which serves every key after `serve_delay`) for key 1, crashes
/// `crash_after` after peer 1 receives the request, restarts after `downtime` on the same key
/// and IP, and asks for keys `2..=KEYS`. Links have one-way latency `latency`. With
/// `recommended`, both use [lookup::Config::recommended] timing and peer 0 waits until their
/// connection is older than the connection cooldown before asking for key 1.
fn restart_during_serve(
    seed: u64,
    recommended: bool,
    latency: Duration,
    serve_delay: Duration,
    crash_after: Duration,
    downtime: Duration,
) -> Outcome {
    let link = LinkConfig::new(latency, Duration::ZERO);
    let testbed = Testbed::new(Topology::sockets(seed).with_default(link));
    let cfg = testbed.install(
        deterministic::Config::default()
            .with_seed(seed)
            .with_timeout(Some(Duration::from_secs(600))),
    );
    deterministic::Runner::new(cfg).start(|context| async move {
        let signers: Vec<_> = (0..2u64).map(PrivateKey::from_seed).collect();

        // Peer 1 serves every key slowly
        let (requested_tx, mut requested) = mpsc::unbounded_channel();
        let (_server, _server_peer) = start(
            &context,
            &signers,
            1,
            recommended,
            |context| SlowProducer {
                context: Arc::new(context.child("producer")),
                delay: serve_delay,
                requested: requested_tx,
            },
            Consumer::dummy(),
        );

        // Peer 0 asks for key 1 and crashes once peer 1 is serving it
        let (consumer, _) = Consumer::new();
        let (mut mailbox, requester) = start(
            &context,
            &signers,
            0,
            recommended,
            |_| super::mocks::Producer::<Key, Bytes>::default(),
            consumer,
        );
        if recommended {
            context.sleep(Duration::from_secs(200)).await;
        }
        mailbox.fetch(Key(1));
        assert_eq!(requested.recv().await.unwrap(), Key(1));
        context.sleep(crash_after).await;
        requester.process.crash();
        drop(mailbox);
        context.sleep(downtime).await;

        // The restarted peer 0 asks for key 2
        let (mut consumer, mut delivered) = Consumer::new();
        for key in 1..=KEYS {
            consumer.add_expected(Key(key), value(&Key(key)));
        }
        let (mut mailbox, requester) = start(
            &context,
            &signers,
            0,
            recommended,
            |_| super::mocks::Producer::<Key, Bytes>::default(),
            consumer,
        );
        for key in 2..=KEYS {
            mailbox.fetch(Key(key));
        }

        // Wait for every key (or long enough for every response to have arrived)
        let deadline = context.current() + serve_delay * 3 + Duration::from_secs(30);
        let mut missing: BTreeSet<u8> = (2..=KEYS).collect();
        while !missing.is_empty() {
            select! {
                item = delivered.recv() => {
                    let (key, _) = item.unwrap();
                    missing.remove(&key.0);
                },
                _ = context.sleep_until(deadline) => break,
            }
        }
        let got = missing.is_empty();
        let blocked = requester
            .context
            .encode()
            .lines()
            .any(|line| line.contains("tracker_directory_blocked{") && !line.ends_with(" 0"));
        Outcome {
            blocked,
            delivered: got,
        }
    })
}

/// Peer 1 takes 2s to serve; peer 0 crashes 100ms into the serve and restarts 100ms later
/// (RTT 100ms). Peer 1's late response to the pre-crash request reaches the restarted peer 0
/// on its new connection, answers the new request (same id) with key 1's data, and peer 0
/// blocks the honest peer 1.
#[test_traced("WARN")]
#[ignore = "finding: request ids restart at zero, so stale responses answer new requests"]
fn test_stale_response_after_restart_over_lookup() {
    let outcome = restart_during_serve(
        0,
        false,
        Duration::from_millis(50),
        Duration::from_secs(2),
        Duration::from_millis(100),
        Duration::from_millis(100),
    );
    assert_eq!(
        outcome,
        Outcome {
            blocked: false,
            delivered: true
        }
    );
}

/// As [test_stale_response_after_restart_over_lookup], with [lookup::Config::recommended] timing
/// (60s connection cooldown, 1s dial frequency) over 150ms links: peer 1 takes 3s to serve, and
/// peer 0 crashes 100ms into the serve and restarts 500ms later.
#[test_traced("WARN")]
#[ignore = "finding: request ids restart at zero, so stale responses answer new requests"]
fn test_stale_response_after_restart_over_lookup_recommended() {
    let outcome = restart_during_serve(
        0,
        true,
        Duration::from_millis(150),
        Duration::from_secs(3),
        Duration::from_millis(100),
        Duration::from_millis(500),
    );
    assert_eq!(
        outcome,
        Outcome {
            blocked: false,
            delivered: true
        }
    );
}

/// Window sweep: for each configuration, one-way latency, downtime, and serve delay, how many
/// seeds end with the restarted requester blocking the slow peer. Prints a table.
#[test]
#[ignore]
fn test_stale_response_after_restart_over_lookup_window() {
    for (recommended, latencies) in [(false, [5u64, 50]), (true, [50, 150])] {
        for latency_ms in latencies {
            for downtime_ms in [0u64, 500, 1_000, 3_000] {
                for serve_ms in [250u64, 500, 1_000, 2_000, 3_000, 5_000, 10_000] {
                    let mut blocked = 0;
                    let seeds = 5;
                    for seed in 0..seeds {
                        let outcome = restart_during_serve(
                            seed,
                            recommended,
                            Duration::from_millis(latency_ms),
                            Duration::from_millis(serve_ms),
                            Duration::ZERO,
                            Duration::from_millis(downtime_ms),
                        );
                        blocked += usize::from(outcome.blocked);
                    }
                    eprintln!(
                        "recommended={recommended} latency={latency_ms}ms downtime={downtime_ms}ms serve={serve_ms}ms blocked={blocked}/{seeds}"
                    );
                }
            }
        }
    }
}
