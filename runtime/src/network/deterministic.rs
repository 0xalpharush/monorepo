use crate::{Error, IoBufs, mocks};
use commonware_utils::{channel::mpsc, sync::Mutex};
use std::{
    collections::HashMap,
    future::Future,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    ops::Range,
    pin::Pin,
    sync::{Arc, OnceLock},
    time::Duration,
};

/// Range of ephemeral ports assigned to dialers.
const EPHEMERAL_PORT_RANGE: Range<u16> = 32768..61000;

/// One direction of a link between two endpoints of a simulated network.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Link<A = SocketAddr> {
    /// The sending endpoint.
    pub from: A,
    /// The receiving endpoint.
    pub to: A,
}

/// One transmission a simulated network is about to carry: a send on a connection, or a
/// message between peers.
#[derive(Clone, Copy, Debug)]
pub struct Transmission<'a, A = SocketAddr> {
    /// The link the transmission travels.
    pub link: &'a Link<A>,
    /// The channel the transmission belongs to, for networks that multiplex channels.
    pub channel: Option<u64>,
    /// The transmission's position among those sent on `link` (and `channel`).
    pub index: u64,
    /// The transmission's length in bytes.
    pub len: usize,
}

/// What a simulated network does with one transmission.
///
/// Built from [Self::after], [Self::DROP], or [Self::RESET], and refined with [Self::duplicated]
/// and [Self::corrupted]. Every network carries out every outcome in its own
/// unit (bytes on a connection, messages between peers); an outcome a network cannot express maps
/// to the closest one it can, as each network documents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Delivery {
    /// Whether, and how, the transmission is carried.
    pub fate: Fate,
    /// How long the transmission takes to arrive.
    pub after: Duration,
    /// Deliver a second copy this long after the first arrives (a stale replay).
    pub duplicate: Option<Duration>,
    /// Flip this bit of the transmission (modulo its length in bits) before delivering it.
    pub corrupt: Option<u64>,
}

/// Whether, and how, a simulated network carries a transmission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fate {
    /// The transmission is delivered.
    Deliver,
    /// The transmission is lost.
    Drop,
    /// The transmission's connection is reset before it is sent.
    Reset,
}

impl Delivery {
    /// The transmission is lost.
    pub const DROP: Self = Self::fate(Fate::Drop);

    /// The transmission's connection is reset.
    pub const RESET: Self = Self::fate(Fate::Reset);

    /// Deliver immediately.
    pub const NOW: Self = Self::after(Duration::ZERO);

    const fn fate(fate: Fate) -> Self {
        Self {
            fate,
            after: Duration::ZERO,
            duplicate: None,
            corrupt: None,
        }
    }

    /// Deliver after `after` of simulated time.
    pub const fn after(after: Duration) -> Self {
        Self {
            after,
            ..Self::fate(Fate::Deliver)
        }
    }

    /// Also deliver a second copy `after` the first arrives.
    pub const fn duplicated(mut self, after: Duration) -> Self {
        self.duplicate = Some(after);
        self
    }

    /// Flip `bit` (modulo the transmission's length in bits) before delivering.
    pub const fn corrupted(mut self, bit: u64) -> Self {
        self.corrupt = Some(bit);
        self
    }
}

/// Decides the faults and latency of a simulated network.
///
/// Without a policy, a network carries every transmission as configured. A policy sees every
/// connection attempt and every transmission with the link it concerns, so it can derive, record,
/// replay, or override each decision independently of all other randomness in the runtime. The
/// same policy type drives the runtime's deterministic sockets (`A = SocketAddr`) and message-level
/// simulated networks keyed by peer identity.
pub trait Policy<A = SocketAddr>: Send + Sync {
    /// Whether a connection from `from` to `to` can be established; `false` fails it. Only
    /// networks with connections consult this.
    fn connects(&self, from: &A, to: &A) -> bool {
        let _ = (from, to);
        true
    }

    /// What happens to `transmission`.
    fn delivers(&self, transmission: &Transmission<'_, A>) -> Delivery;
}

/// A source of simulated time, installed by the runtime that owns the [Network].
pub(crate) trait Timer: Send + Sync {
    /// Resolves once `delay` of simulated time has passed.
    fn sleep(&self, delay: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>>;
}

/// Filled once the owning runtime exists; the network is created before its executor.
pub(crate) type TimerSlot = Arc<OnceLock<Arc<dyn Timer>>>;

/// Implementation of [crate::Sink] for a deterministic [Network].
pub struct Sink {
    /// `None` once the connection has been reset.
    inner: Option<mocks::Sink>,
    link: Link,
    sent: u64,
    policy: Option<Arc<dyn Policy>>,
    timer: TimerSlot,
}

impl Sink {
    fn new(
        inner: mocks::Sink,
        link: Link,
        policy: Option<Arc<dyn Policy>>,
        timer: TimerSlot,
    ) -> Self {
        Self {
            inner: Some(inner),
            link,
            sent: 0,
            policy,
            timer,
        }
    }
    async fn sleep(&self, delay: Duration) {
        if delay.is_zero() {
            return;
        }
        let timer = self
            .timer
            .get()
            .expect("a network with latency needs its runtime's timer")
            .clone();
        timer.sleep(delay).await;
    }
}

/// Flip `bit` (modulo the buffer's length in bits) of `bufs`.
fn flip(bufs: IoBufs, bit: u64) -> IoBufs {
    let mut bytes = bufs.coalesce().as_ref().to_vec();
    if bytes.is_empty() {
        return IoBufs::from(bytes);
    }
    let bits = u64::try_from(bytes.len())
        .expect("bounded send")
        .saturating_mul(8);
    let bit = bit % bits;
    let byte = usize::try_from(bit / 8).expect("bit within the send");
    bytes[byte] ^= 1 << (bit % 8);
    IoBufs::from(bytes)
}

/// Carries out [Delivery] outcomes on a byte stream:
///
/// - `after` delays the send, holding back the sends behind it, as on a TCP stream.
/// - [Fate::Drop] and [Fate::Reset] reset the connection before the send: a stream cannot lose
///   bytes without breaking, and the peer's stream fails too.
/// - `corrupt` flips a bit of the sent bytes.
/// - `duplicate` sends the bytes again (after the extra delay), as a retransmission bug would.
impl crate::Sink for Sink {
    async fn send(&mut self, bufs: impl Into<IoBufs> + Send) -> Result<(), Error> {
        let mut bufs = bufs.into();
        if self.inner.is_none() {
            return Err(Error::Closed);
        }
        let mut duplicate = None;
        if let Some(policy) = &self.policy {
            let index = self.sent;
            self.sent = self.sent.checked_add(1).expect("send count overflow");
            let delivery = policy.delivers(&Transmission {
                link: &self.link,
                channel: None,
                index,
                len: bytes::Buf::remaining(&bufs),
            });
            if delivery.fate != Fate::Deliver {
                // Dropping the sink closes the pipe, so the peer's stream fails too.
                self.inner = None;
                return Err(Error::Closed);
            }
            self.sleep(delivery.after).await;
            if let Some(bit) = delivery.corrupt {
                bufs = flip(bufs, bit);
            }
            duplicate = delivery.duplicate.map(|after| (after, bufs.clone()));
        }
        let Some(inner) = self.inner.as_mut() else {
            return Err(Error::Closed);
        };
        inner.send(bufs).await?;
        if let Some((after, bufs)) = duplicate {
            self.sleep(after).await;
            let Some(inner) = self.inner.as_mut() else {
                return Err(Error::Closed);
            };
            inner.send(bufs).await?;
        }
        Ok(())
    }
}

/// Implementation of [crate::Stream] for a deterministic [Network].
pub type Stream = mocks::Stream;

/// Implementation of [crate::Listener] for a deterministic [Network].
pub struct Listener {
    address: SocketAddr,
    listener: mpsc::UnboundedReceiver<(SocketAddr, Sink, mocks::Stream)>,
}

impl crate::Listener for Listener {
    type Sink = Sink;
    type Stream = Stream;

    async fn accept(&mut self) -> Result<(SocketAddr, Self::Sink, Self::Stream), Error> {
        let (socket, sender, receiver) = self.listener.recv().await.ok_or(Error::ReadFailed)?;
        Ok((socket, sender, receiver))
    }

    fn local_addr(&self) -> Result<SocketAddr, std::io::Error> {
        Ok(self.address)
    }
}

type Dialable = mpsc::UnboundedSender<(
    SocketAddr,
    Sink,          // Listener -> Dialer
    mocks::Stream, // Dialer -> Listener
)>;

/// Deterministic implementation of [crate::Network].
///
/// When a dialer connects to a listener, the listener is given a new ephemeral port
/// from the range `32768..61000`. To keep things simple, it is not possible to
/// bind to an ephemeral port. Likewise, if ports are not reused and when exhausted,
/// the runtime will panic.
#[derive(Clone)]
pub struct Network {
    ephemeral: Arc<Mutex<u16>>,
    listeners: Arc<Mutex<HashMap<SocketAddr, Dialable>>>,
    policy: Option<Arc<dyn Policy>>,
    timer: TimerSlot,
}

impl Default for Network {
    fn default() -> Self {
        Self {
            ephemeral: Arc::new(Mutex::new(EPHEMERAL_PORT_RANGE.start)),
            listeners: Arc::new(Mutex::new(HashMap::new())),
            policy: None,
            timer: Arc::default(),
        }
    }
}

impl Network {
    /// A network whose connection faults and latency `policy` decides, timed by `timer`.
    pub(crate) fn with_policy(policy: Option<Arc<dyn Policy>>, timer: TimerSlot) -> Self {
        Self {
            policy,
            timer,
            ..Self::default()
        }
    }
}

impl crate::Network for Network {
    type Listener = Listener;

    async fn bind(&self, socket: SocketAddr) -> Result<Self::Listener, Error> {
        // If the IP is localhost, ensure the port is not in the ephemeral range
        // so that it can be used for binding in the dial method
        if socket.ip() == IpAddr::V4(Ipv4Addr::LOCALHOST)
            && EPHEMERAL_PORT_RANGE.contains(&socket.port())
        {
            return Err(Error::BindFailed);
        }

        // Ensure the port is not bound by a live listener; a dropped listener frees its port,
        // so a restarted process can bind its address again
        let mut listeners = self.listeners.lock();
        if listeners
            .get(&socket)
            .is_some_and(|existing| !existing.is_closed())
        {
            return Err(Error::BindFailed);
        }

        // Bind the socket
        let (sender, receiver) = mpsc::unbounded_channel();
        listeners.insert(socket, sender);
        Ok(Listener {
            address: socket,
            listener: receiver,
        })
    }

    async fn dial(&self, socket: SocketAddr) -> Result<(Sink, Stream), Error> {
        // Assign dialer a port from the ephemeral range
        let dialer = {
            let mut ephemeral = self.ephemeral.lock();
            let dialer = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), *ephemeral);
            *ephemeral = ephemeral
                .checked_add(1)
                .expect("ephemeral port range exhausted");
            dialer
        };

        // Get listener
        let sender = {
            let listeners = self.listeners.lock();
            let sender = listeners.get(&socket).ok_or(Error::ConnectionFailed)?;
            sender.clone()
        };

        if self
            .policy
            .as_ref()
            .is_some_and(|policy| !policy.connects(&dialer, &socket))
        {
            return Err(Error::ConnectionFailed);
        }

        // Construct connection
        let (dialer_sender, dialer_receiver) = mocks::Channel::init();
        let (listener_sender, listener_receiver) = mocks::Channel::init();
        let to_dialer = Link {
            from: socket,
            to: dialer,
        };
        let to_listener = Link {
            from: dialer,
            to: socket,
        };
        sender
            .send((
                dialer,
                Sink::new(
                    dialer_sender,
                    to_dialer,
                    self.policy.clone(),
                    self.timer.clone(),
                ),
                listener_receiver,
            ))
            .map_err(|_| Error::ConnectionFailed)?;
        Ok((
            Sink::new(
                listener_sender,
                to_listener,
                self.policy.clone(),
                self.timer.clone(),
            ),
            dialer_receiver,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Clock, Runner, Spawner,
        network::{deterministic as DeterministicNetwork, tests},
    };
    use commonware_macros::test_group;
    use rstest::rstest;

    #[rstest]
    #[case::tokio(crate::tokio::Runner::default())]
    #[cfg_attr(
        all(target_os = "linux", feature = "iouring"),
        case::iouring(crate::iouring::Runner::default())
    )]
    fn test_trait<R: Runner>(#[case] runner: R)
    where
        R::Context: Spawner + Clock,
    {
        runner.start(|context| async move {
            tests::test_network_trait(context, DeterministicNetwork::Network::default).await;
        });
    }

    #[rstest]
    #[case::tokio(crate::tokio::Runner::default())]
    #[cfg_attr(
        all(target_os = "linux", feature = "iouring"),
        case::iouring(crate::iouring::Runner::default())
    )]
    #[test_group("slow")]
    fn test_stress_trait<R: Runner>(#[case] runner: R)
    where
        R::Context: Spawner + Clock,
    {
        runner.start(|context| async move {
            tests::stress_test_network_trait(context, DeterministicNetwork::Network::default).await;
        });
    }
}
