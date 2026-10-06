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

/// One direction of a simulated connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Link {
    /// The sending end's address.
    pub from: SocketAddr,
    /// The receiving end's address.
    pub to: SocketAddr,
}

/// What a deterministic [Network] does with one send.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delivery {
    /// The bytes reach the peer after this much simulated time. The connection carries sends in
    /// order, so a delay also holds back the sends behind it, as on a TCP stream.
    After(Duration),
    /// The connection is reset before the send, as a peer crash or network failure would.
    Reset,
}

/// Decides the connection faults and latency of a deterministic [Network].
///
/// Connections otherwise behave as lossless, ordered, zero-latency pipes. A policy sees every
/// dial and every send with the connection it concerns, so it can derive, record, replay, or
/// override each decision independently of all other randomness in the runtime.
pub trait Policy: Send + Sync {
    /// Whether a dial from `dialer` reaches the listener bound at `listener`; `false` fails it.
    fn connects(&self, dialer: SocketAddr, listener: SocketAddr) -> bool;

    /// What happens to `link`'s `index`-th send of `len` bytes.
    fn delivers(&self, link: Link, index: u64, len: usize) -> Delivery;
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
}

impl crate::Sink for Sink {
    async fn send(&mut self, bufs: impl Into<IoBufs> + Send) -> Result<(), Error> {
        let bufs = bufs.into();
        if self.inner.is_none() {
            return Err(Error::Closed);
        }
        if let Some(policy) = &self.policy {
            let index = self.sent;
            self.sent = self.sent.checked_add(1).expect("send count overflow");
            match policy.delivers(self.link, index, bytes::Buf::remaining(&bufs)) {
                Delivery::After(delay) if !delay.is_zero() => {
                    let timer = self
                        .timer
                        .get()
                        .expect("a network with latency needs its runtime's timer")
                        .clone();
                    timer.sleep(delay).await;
                }
                Delivery::After(_) => {}
                Delivery::Reset => {
                    // Dropping the sink closes the pipe, so the peer's stream fails too.
                    self.inner = None;
                    return Err(Error::Closed);
                }
            }
        }
        let Some(inner) = self.inner.as_mut() else {
            return Err(Error::Closed);
        };
        inner.send(bufs).await
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
            .is_some_and(|policy| !policy.connects(dialer, socket))
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
