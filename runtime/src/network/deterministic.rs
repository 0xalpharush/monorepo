use crate::{Error, IoBufs};
use bytes::{Bytes, BytesMut};
use commonware_utils::{channel::mpsc, sync::Mutex};
use std::{
    cell::Cell,
    collections::{BTreeSet, HashMap, VecDeque},
    future::{Future, poll_fn},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    ops::Range,
    pin::Pin,
    sync::{Arc, OnceLock, Weak},
    task::{Context, Poll, Waker},
    time::{Duration, SystemTime},
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
    /// The simulated time at which the transmission is sent.
    pub at: SystemTime,
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

/// A fixed [Delivery] is a policy that decides every transmission the same way (for example,
/// [Delivery::NOW] for a perfect network).
impl<A> Policy<A> for Delivery {
    fn delivers(&self, _: &Transmission<'_, A>) -> Delivery {
        *self
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

/// A shared policy decides as the policy it points to (so a test can keep a handle to a policy
/// it installs, for example to change it while the simulation runs).
impl<A, P: Policy<A> + ?Sized> Policy<A> for Arc<P> {
    fn connects(&self, from: &A, to: &A) -> bool {
        (**self).connects(from, to)
    }

    fn delivers(&self, transmission: &Transmission<'_, A>) -> Delivery {
        (**self).delivers(transmission)
    }
}

/// Resolves once a [Timer]'s delay has passed.
pub(crate) type Sleep = Pin<Box<dyn Future<Output = ()> + Send + Sync>>;

/// A source of simulated time, installed by the runtime that owns the [Network].
pub(crate) trait Timer: Send + Sync {
    /// Resolves once `delay` of simulated time has passed.
    fn sleep(&self, delay: Duration) -> Sleep;

    /// The current simulated time.
    fn now(&self) -> SystemTime;
}

/// Filled once the owning runtime exists; the network is created before its executor.
pub(crate) type TimerSlot = Arc<OnceLock<Arc<dyn Timer>>>;

/// Default number of bytes a connection holds in each direction (in flight or arrived but not yet
/// read) before sends block, as a TCP send buffer and receive window would.
pub const DEFAULT_WINDOW: usize = 4 * 1024 * 1024;

/// Number of bytes a [Stream] moves into its local (peekable) buffer per receive, at least.
const READ_CHUNK: usize = 64 * 1024;

/// Bytes sent on a connection, readable once `arrival` (if any) has passed.
struct Segment {
    arrival: Option<SystemTime>,
    data: Bytes,
}

/// One direction of a deterministic connection.
///
/// Sent bytes are queued with the time they arrive and become readable, in order, once it has
/// passed: a connection carries latency as propagation delay, so a send does not wait for its
/// bytes to arrive. Sends block only while more than `window` bytes are in flight or unread.
struct Wire {
    in_flight: VecDeque<Segment>,
    /// Bytes queued in `in_flight`.
    pending: usize,
    window: usize,
    /// Arrival of the last queued segment (later segments never arrive before it).
    last_arrival: Option<SystemTime>,
    /// Whether the sending half has been dropped (closing the direction gracefully).
    sink_closed: bool,
    /// Whether the connection has been reset (losing everything still queued).
    reset: bool,
    /// Whether the receiving half is still alive.
    stream_alive: bool,
    reader: Option<Waker>,
    writer: Option<Waker>,
}

impl Wire {
    fn new(window: usize) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            in_flight: VecDeque::new(),
            pending: 0,
            window,
            last_arrival: None,
            sink_closed: false,
            reset: false,
            stream_alive: true,
            reader: None,
            writer: None,
        }))
    }

    /// Queue `data` to arrive at `arrival` (or immediately), but never before earlier bytes.
    fn push(&mut self, arrival: Option<SystemTime>, data: Bytes) {
        let arrival = match (self.last_arrival, arrival) {
            (Some(last), Some(arrival)) => Some(last.max(arrival)),
            (last, arrival) => last.or(arrival),
        };
        self.last_arrival = arrival;
        self.pending += data.len();
        self.in_flight.push_back(Segment { arrival, data });
        if let Some(reader) = self.reader.take() {
            reader.wake();
        }
    }

    /// Reset the direction, losing every queued byte.
    fn reset(&mut self) {
        self.reset = true;
        self.in_flight.clear();
        self.pending = 0;
        if let Some(reader) = self.reader.take() {
            reader.wake();
        }
        if let Some(writer) = self.writer.take() {
            writer.wake();
        }
    }
}

/// An ephemeral port held by a dialer until both halves of its connection are dropped.
struct Port {
    ports: Weak<Mutex<Ports>>,
    key: (IpAddr, SocketAddr),
    port: u16,
}

impl Drop for Port {
    fn drop(&mut self) {
        if let Some(ports) = self.ports.upgrade() {
            let mut ports = ports.lock();
            if let Some(pool) = ports.get_mut(&self.key) {
                pool.used.remove(&self.port);
                if pool.used.is_empty() {
                    ports.remove(&self.key);
                }
            }
        }
    }
}

/// Ephemeral ports in use by connections from one source IP to one destination.
struct Pool {
    /// The next port to try (ports are handed out in rotation, so a released port is not reused
    /// immediately).
    next: u16,
    used: BTreeSet<u16>,
}

/// Ephemeral ports in use, by source IP and destination: like TCP, a connection is identified by
/// both endpoints, so a source port only needs to be unique per destination.
type Ports = HashMap<(IpAddr, SocketAddr), Pool>;

/// Implementation of [crate::Sink] for a deterministic [Network].
pub struct Sink {
    wire: Arc<Mutex<Wire>>,
    /// The opposite direction of the connection, reset along with this one.
    reverse: Weak<Mutex<Wire>>,
    link: Link,
    sent: u64,
    policy: Option<Arc<dyn Policy>>,
    timer: TimerSlot,
    /// Set while a send is in progress (a send canceled mid-flight poisons the sink) and once
    /// the sink fails.
    poisoned: bool,
    _port: Option<Arc<Port>>,
}

impl Sink {
    fn timer(&self) -> Arc<dyn Timer> {
        self.timer
            .get()
            .expect("a network with a policy needs its runtime's timer")
            .clone()
    }

    async fn sleep(&self, delay: Duration) {
        if delay.is_zero() {
            return;
        }
        self.timer().sleep(delay).await;
    }

    /// Reset both directions of the connection, as a TCP reset would.
    fn reset(&mut self) {
        self.wire.lock().reset();
        if let Some(reverse) = self.reverse.upgrade() {
            reverse.lock().reset();
        }
    }
}

impl Drop for Sink {
    fn drop(&mut self) {
        let mut wire = self.wire.lock();
        wire.sink_closed = true;
        if let Some(reader) = wire.reader.take() {
            reader.wake();
        }
    }
}

/// Flip `bit` (modulo the buffer's length in bits) of `bytes`.
fn flip(bytes: &mut [u8], bit: u64) {
    if bytes.is_empty() {
        return;
    }
    let bits = u64::try_from(bytes.len())
        .expect("bounded send")
        .saturating_mul(8);
    let bit = bit % bits;
    let byte = usize::try_from(bit / 8).expect("bit within the send");
    bytes[byte] ^= 1 << (bit % 8);
}

/// Carries out [Delivery] outcomes on a byte stream:
///
/// - `after` is propagation delay: the send completes once its bytes are queued, and they become
///   readable `after` later, never before bytes sent earlier on the connection (as on a TCP
///   stream).
/// - [Fate::Drop] and [Fate::Reset] reset the connection, after stalling the send for `after`: a
///   stream cannot lose bytes without breaking. Both directions fail, and bytes still in flight
///   are lost.
/// - `corrupt` flips a bit of the sent bytes.
/// - `duplicate` delivers the bytes again (that long after the first copy), as a retransmission
///   bug would.
///
/// A send blocks while more than the connection's window of bytes is in flight or unread.
impl crate::Sink for Sink {
    async fn send(&mut self, bufs: impl Into<IoBufs> + Send) -> Result<(), Error> {
        if self.poisoned {
            return Err(Error::Closed);
        }
        let bufs = bufs.into();
        self.poisoned = true;
        let mut delivery = Delivery::NOW;
        let mut now = None;
        if let Some(policy) = &self.policy {
            let index = self.sent;
            self.sent = self.sent.checked_add(1).expect("send count overflow");
            let at = self.timer().now();
            now = Some(at);
            delivery = policy.delivers(&Transmission {
                link: &self.link,
                channel: None,
                index,
                len: bytes::Buf::remaining(&bufs),
                at,
            });
        }
        if delivery.fate != Fate::Deliver {
            // The send stalls for `after` (as a TCP retransmission timeout would) before the
            // connection breaks.
            self.sleep(delivery.after).await;
            self.reset();
            return Err(Error::Closed);
        }
        let mut data = bufs.coalesce().as_ref().to_vec();
        if let Some(bit) = delivery.corrupt {
            flip(&mut data, bit);
        }
        let data = Bytes::from(data);
        let at = |delay: Duration| {
            (!delay.is_zero()).then(|| {
                now.expect("delays are only decided by a policy")
                    .checked_add(delay)
                    .expect("arrival overflow")
            })
        };
        {
            let mut wire = self.wire.lock();
            if wire.reset {
                return Err(Error::Closed);
            }
            if !wire.stream_alive {
                return Err(Error::SendFailed);
            }
            let arrival = at(delivery.after);
            if let Some(extra) = delivery.duplicate {
                wire.push(arrival, data.clone());
                let again = delivery.after.checked_add(extra).expect("arrival overflow");
                wire.push(at(again), data);
            } else {
                wire.push(arrival, data);
            }
        }

        // Block while the window is exceeded
        let wire = &self.wire;
        let result = poll_fn(|cx| {
            let mut wire = wire.lock();
            if wire.reset {
                return Poll::Ready(Err(Error::Closed));
            }
            if !wire.stream_alive {
                return Poll::Ready(Err(Error::SendFailed));
            }
            if wire.pending <= wire.window {
                return Poll::Ready(Ok(()));
            }
            wire.writer = Some(cx.waker().clone());
            Poll::Pending
        })
        .await;
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }
}

/// Implementation of [crate::Stream] for a deterministic [Network].
pub struct Stream {
    wire: Arc<Mutex<Wire>>,
    timer: TimerSlot,
    /// Arrived bytes not yet consumed.
    buffer: BytesMut,
    /// Wakes the stream when the next in-flight segment arrives.
    sleep: Option<(SystemTime, Sleep)>,
    /// Set while a receive is in progress (a receive canceled mid-flight poisons the stream)
    /// and once the stream fails.
    poisoned: bool,
    _port: Option<Arc<Port>>,
}

impl Stream {
    fn poll_recv(&mut self, cx: &mut Context<'_>, len: usize) -> Poll<Result<IoBufs, Error>> {
        loop {
            let mut wire = self.wire.lock();

            // Move arrived bytes into the local buffer
            let target = len.max(READ_CHUNK);
            let mut now = None;
            let mut next = None;
            let mut pulled = false;
            while self.buffer.len() < target {
                let Some(front) = wire.in_flight.front_mut() else {
                    break;
                };
                if let Some(arrival) = front.arrival {
                    let current = *now.get_or_insert_with(|| {
                        self.timer
                            .get()
                            .expect("delayed bytes need the runtime's timer")
                            .now()
                    });
                    if arrival > current {
                        next = Some((arrival, current));
                        break;
                    }
                }
                let take = front.data.len().min(target - self.buffer.len());
                self.buffer.extend_from_slice(&front.data.split_to(take));
                if front.data.is_empty() {
                    wire.in_flight.pop_front();
                }
                wire.pending -= take;
                pulled = true;
            }
            if pulled
                && wire.pending <= wire.window
                && let Some(writer) = wire.writer.take()
            {
                writer.wake();
            }

            if wire.reset {
                self.buffer.clear();
                return Poll::Ready(Err(Error::RecvFailed));
            }
            if self.buffer.len() >= len {
                return Poll::Ready(Ok(IoBufs::from(self.buffer.split_to(len).freeze())));
            }
            if wire.sink_closed && wire.in_flight.is_empty() {
                return Poll::Ready(Err(Error::RecvFailed));
            }
            wire.reader = Some(cx.waker().clone());
            drop(wire);

            // Wait for the next segment to arrive (or for a new one to be sent)
            let Some((arrival, current)) = next else {
                return Poll::Pending;
            };
            if self.sleep.as_ref().is_none_or(|(at, _)| *at != arrival) {
                let timer = self.timer.get().expect("delayed bytes need the runtime's timer");
                let delay = arrival.duration_since(current).expect("arrival is later");
                self.sleep = Some((arrival, timer.sleep(delay)));
            }
            let (_, sleep) = self.sleep.as_mut().expect("sleep set above");
            if sleep.as_mut().poll(cx).is_pending() {
                return Poll::Pending;
            }
            self.sleep = None;
        }
    }
}

impl crate::Stream for Stream {
    async fn recv(&mut self, len: usize) -> Result<IoBufs, Error> {
        if self.poisoned {
            return Err(Error::Closed);
        }
        self.poisoned = true;
        let result = poll_fn(|cx| self.poll_recv(cx, len)).await;
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    fn peek(&self, max_len: usize) -> &[u8] {
        let len = max_len.min(self.buffer.len());
        &self.buffer[..len]
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let mut wire = self.wire.lock();
        wire.stream_alive = false;
        wire.in_flight.clear();
        wire.pending = 0;
        if let Some(writer) = wire.writer.take() {
            writer.wake();
        }
    }
}

/// Implementation of [crate::Listener] for a deterministic [Network].
pub struct Listener {
    address: SocketAddr,
    listener: mpsc::UnboundedReceiver<(SocketAddr, Sink, Stream)>,
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

type Dialable = mpsc::UnboundedSender<(SocketAddr, Sink, Stream)>;

std::thread_local! {
    /// The source IP of the dial being polled on this thread, if its dialer has one.
    static DIAL_SOURCE: Cell<Option<IpAddr>> = const { Cell::new(None) };
}

/// Polls `dial` with `source` as the IP its dialer's ephemeral address takes.
pub(crate) fn sourced<F: Future>(source: Option<IpAddr>, dial: F) -> Sourced<F> {
    Sourced {
        source,
        dial: Box::pin(dial),
    }
}

/// A dial polled with its dialer's source IP; see [sourced].
pub(crate) struct Sourced<F> {
    source: Option<IpAddr>,
    dial: Pin<Box<F>>,
}

impl<F: Future> Future for Sourced<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let previous = DIAL_SOURCE.replace(self.source);
        let result = self.dial.as_mut().poll(cx);
        DIAL_SOURCE.set(previous);
        result
    }
}

/// Deterministic implementation of [crate::Network].
///
/// A dialer is given an ephemeral port on its source IP: the IP set for its process (see
/// [crate::deterministic::Process::set_ip]), or `127.0.0.1` without one. Ports are drawn from the
/// range `32768..61000` in rotation, skipping ports bound by a listener on the source IP and ports
/// still used by a connection from the same source IP to the same destination (as in TCP, a
/// connection is identified by both of its endpoints). A port is released once both halves of
/// its connection are dropped. A dial fails if every port is in use for its source and
/// destination. To keep things simple, it is not possible to bind to an ephemeral port on
/// `127.0.0.1`.
///
/// Each direction of a connection holds up to [DEFAULT_WINDOW] bytes in flight or unread before
/// sends block.
#[derive(Clone)]
pub struct Network {
    ports: Arc<Mutex<Ports>>,
    listeners: Arc<Mutex<HashMap<SocketAddr, Dialable>>>,
    policy: Option<Arc<dyn Policy>>,
    timer: TimerSlot,
    window: usize,
}

impl Default for Network {
    fn default() -> Self {
        Self {
            ports: Arc::default(),
            listeners: Arc::new(Mutex::new(HashMap::new())),
            policy: None,
            timer: Arc::default(),
            window: DEFAULT_WINDOW,
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

    /// Hold up to `window` bytes in flight or unread in each direction of a connection before
    /// sends block.
    #[cfg(test)]
    pub(crate) const fn with_window(mut self, window: usize) -> Self {
        self.window = window;
        self
    }

    /// Reserve an ephemeral port for a connection from `source` to `destination`.
    fn reserve(&self, source: IpAddr, destination: SocketAddr) -> Result<Arc<Port>, Error> {
        let listeners = self.listeners.lock();
        let mut ports = self.ports.lock();
        let key = (source, destination);
        let pool = ports.entry(key).or_insert_with(|| Pool {
            next: EPHEMERAL_PORT_RANGE.start,
            used: BTreeSet::new(),
        });
        for _ in EPHEMERAL_PORT_RANGE {
            let port = pool.next;
            pool.next = if port + 1 == EPHEMERAL_PORT_RANGE.end {
                EPHEMERAL_PORT_RANGE.start
            } else {
                port + 1
            };
            if pool.used.contains(&port)
                || listeners
                    .get(&SocketAddr::new(source, port))
                    .is_some_and(|listener| !listener.is_closed())
            {
                continue;
            }
            pool.used.insert(port);
            return Ok(Arc::new(Port {
                ports: Arc::downgrade(&self.ports),
                key,
                port,
            }));
        }
        if pool.used.is_empty() {
            ports.remove(&key);
        }
        Err(Error::ConnectionFailed)
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
        // Get listener
        let sender = {
            let listeners = self.listeners.lock();
            let sender = listeners.get(&socket).ok_or(Error::ConnectionFailed)?;
            sender.clone()
        };

        // Assign dialer a port from the ephemeral range on its source IP
        let source = DIAL_SOURCE.get().unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));
        let port = self.reserve(source, socket)?;
        let dialer = SocketAddr::new(source, port.port);

        if self
            .policy
            .as_ref()
            .is_some_and(|policy| !policy.connects(&dialer, &socket))
        {
            return Err(Error::ConnectionFailed);
        }

        // Construct connection
        let to_listener = Wire::new(self.window);
        let to_dialer = Wire::new(self.window);
        let sink = |wire: &Arc<Mutex<Wire>>, reverse: &Arc<Mutex<Wire>>, link, port| Sink {
            wire: wire.clone(),
            reverse: Arc::downgrade(reverse),
            link,
            sent: 0,
            policy: self.policy.clone(),
            timer: self.timer.clone(),
            poisoned: false,
            _port: port,
        };
        let stream = |wire: &Arc<Mutex<Wire>>, port| Stream {
            wire: wire.clone(),
            timer: self.timer.clone(),
            buffer: BytesMut::new(),
            sleep: None,
            poisoned: false,
            _port: port,
        };
        let listener_sink = sink(
            &to_dialer,
            &to_listener,
            Link {
                from: socket,
                to: dialer,
            },
            None,
        );
        let listener_stream = stream(&to_listener, None);
        sender
            .send((dialer, listener_sink, listener_stream))
            .map_err(|_| Error::ConnectionFailed)?;
        let dialer_sink = sink(
            &to_listener,
            &to_dialer,
            Link {
                from: dialer,
                to: socket,
            },
            Some(port.clone()),
        );
        Ok((dialer_sink, stream(&to_dialer, Some(port))))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Clock, Error, Runner, Spawner,
        network::{deterministic as DeterministicNetwork, tests},
    };
    use std::net::SocketAddr;
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

    #[test]
    fn test_window_blocks_sends() {
        use crate::{Listener as _, Network as _, Sink as _, Stream as _};
        crate::deterministic::Runner::default().start(|_| async move {
            let network = DeterministicNetwork::Network::default().with_window(8);
            let address = SocketAddr::from(([10, 0, 0, 1], 3000));
            let mut listener = network.bind(address).await.unwrap();
            let (mut sink, _stream) = network.dial(address).await.unwrap();
            let (_, _, mut stream) = listener.accept().await.unwrap();

            // A send that fits the window completes; one that exceeds it blocks until read
            sink.send(vec![1u8; 8]).await.unwrap();
            let mut blocked = Box::pin(sink.send(vec![2u8; 4]));
            assert!(futures::poll!(blocked.as_mut()).is_pending());
            assert_eq!(stream.recv(8).await.unwrap().coalesce(), [1u8; 8].as_slice());
            blocked.await.unwrap();
            assert_eq!(stream.recv(4).await.unwrap().coalesce(), [2u8; 4].as_slice());
        });
    }

    #[test]
    fn test_ephemeral_ports_per_destination() {
        use crate::{Listener as _, Network as _};
        crate::deterministic::Runner::default().start(|_| async move {
            let network = DeterministicNetwork::Network::default();
            let range = super::EPHEMERAL_PORT_RANGE;
            let ports = usize::from(range.end - range.start);

            // Many more connections than ports, spread over destinations, all coexist
            let mut listeners = Vec::new();
            for i in 0..4u8 {
                let address = SocketAddr::from(([10, 0, 0, i + 1], 3000));
                listeners.push((address, network.bind(address).await.unwrap()));
            }
            let mut connections = Vec::new();
            for (address, _) in &listeners {
                for _ in 0..ports {
                    connections.push(network.dial(*address).await.unwrap());
                }
            }

            // A destination whose ports are all in use refuses further dials until one is released
            let (address, listener) = &mut listeners[0];
            assert!(matches!(
                network.dial(*address).await,
                Err(Error::ConnectionFailed)
            ));
            let (dialer, _, _) = listener.accept().await.unwrap();
            connections.swap_remove(0);
            let (_sink, _stream) = network.dial(*address).await.unwrap();
            let mut last = None;
            while let Some(Ok((port, _, _))) = futures::FutureExt::now_or_never(listener.accept()) {
                last = Some(port);
            }
            assert_eq!(last.unwrap(), dialer, "the released port is reused");
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
