//! Admission control · `MCP_ENGINE.md` §4.
//!
//! Three bounds, in the order a request meets them:
//!
//! ```text
//! request → [bounded queue] --full--> 503 busy
//!              │
//!              │ waited too long ---> 503 timeout
//!              ▼
//!         [semaphore: max N] → work
//! ```
//!
//! ! The queue bound is not a nicety, it is the second half of the OOM
//! guarantee. The semaphore alone bounds *working* requests; without a cap on
//! waiters, an overloaded engine accumulates queued requests until it dies —
//! and it dies of memory, having successfully bounded the thing that was never
//! going to kill it. Both terms must be bounded for `fixed + (N × ceiling)` to
//! be a real ceiling.
//!
//! The queue holds request descriptors (a query string), ✗ working sets, so a
//! full queue is cheap. The wait-timeout exists so a full queue produces honest
//! rejections rather than 20-second tail latencies.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Why a request was turned away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejected {
    /// The queue was already full · shed load immediately.
    QueueFull { depth: usize, capacity: usize },
    /// Waited past the deadline · shed rather than grow the tail.
    WaitTimeout { waited_ms: u64 },
}

impl Rejected {
    /// Message the agent sees.
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            Self::QueueFull { depth, capacity } => format!(
                "engine at capacity ({depth}/{capacity} queued) · retry shortly"
            ),
            Self::WaitTimeout { waited_ms } => {
                format!("engine busy · waited {waited_ms}ms without a slot · retry shortly")
            }
        }
    }

    /// Actionable recovery, per the return-value contract.
    #[must_use]
    pub const fn hint(&self) -> &'static str {
        match self {
            Self::QueueFull { .. } => {
                "the engine is shedding load rather than queueing without bound · \
                 retry with backoff, or raise concurrency.max_concurrent if the host has cores spare"
            }
            Self::WaitTimeout { .. } => {
                "retry with backoff · if this is persistent the engine is under-provisioned \
                 for its offered load"
            }
        }
    }
}

/// A permit that decrements the depth counter on drop.
#[derive(Debug)]
pub struct Guard {
    _permit: OwnedSemaphorePermit,
    depth: Arc<AtomicUsize>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.depth.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The admission controller, shareable across the transport's tasks.
#[derive(Debug, Clone)]
pub struct Gate {
    semaphore: Arc<Semaphore>,
    depth: Arc<AtomicUsize>,
    capacity: usize,
    wait_timeout: Duration,
}

impl Gate {
    #[must_use]
    pub fn new(max_concurrent: usize, queue_capacity: usize, wait_timeout: Duration) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent.max(1))),
            depth: Arc::new(AtomicUsize::new(0)),
            capacity: queue_capacity.max(1),
            wait_timeout,
        }
    }

    /// Build from the engine's configuration.
    #[must_use]
    pub fn from_config(cfg: &vera_core::ConcurrencyConfig) -> Self {
        Self::new(
            cfg.max_concurrent,
            cfg.queue_capacity,
            Duration::from_millis(cfg.wait_timeout_ms),
        )
    }

    /// Requests queued or in flight · surfaced for health checks and tests.
    #[must_use]
    #[allow(dead_code, reason = "observability surface; exercised by tests")]
    pub fn depth(&self) -> usize {
        self.depth.load(Ordering::SeqCst)
    }

    /// Admit a request, or reject it.
    ///
    /// # Errors
    /// [`Rejected::QueueFull`] or [`Rejected::WaitTimeout`].
    pub async fn admit(&self) -> Result<Guard, Rejected> {
        // ! Reserve before waiting. Incrementing after acquiring the semaphore
        // would let unbounded waiters pile up *behind* the check, which is
        // exactly the failure the queue bound exists to prevent.
        let depth = self.depth.fetch_add(1, Ordering::SeqCst) + 1;
        if depth > self.capacity {
            self.depth.fetch_sub(1, Ordering::SeqCst);
            return Err(Rejected::QueueFull {
                depth: depth - 1,
                capacity: self.capacity,
            });
        }

        let started = std::time::Instant::now();
        match tokio::time::timeout(self.wait_timeout, self.semaphore.clone().acquire_owned()).await
        {
            Ok(Ok(permit)) => Ok(Guard {
                _permit: permit,
                depth: Arc::clone(&self.depth),
            }),
            Ok(Err(_closed)) => {
                self.depth.fetch_sub(1, Ordering::SeqCst);
                Err(Rejected::WaitTimeout { waited_ms: 0 })
            }
            Err(_elapsed) => {
                self.depth.fetch_sub(1, Ordering::SeqCst);
                Err(Rejected::WaitTimeout {
                    waited_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_request_under_capacity_is_admitted() {
        let gate = Gate::new(2, 8, Duration::from_millis(50));
        let g = gate.admit().await.unwrap();
        assert_eq!(gate.depth(), 1);
        drop(g);
        assert_eq!(gate.depth(), 0, "depth must fall when the guard drops");
    }

    #[tokio::test]
    async fn concurrency_is_capped_at_the_semaphore() {
        let gate = Gate::new(2, 100, Duration::from_millis(30));
        let _a = gate.admit().await.unwrap();
        let _b = gate.admit().await.unwrap();
        // ! Third request has queue room but no slot · must time out rather
        // than run, or the concurrency ceiling is not a ceiling.
        let c = gate.admit().await;
        assert!(matches!(c, Err(Rejected::WaitTimeout { .. })), "{c:?}");
    }

    #[tokio::test]
    async fn a_full_queue_sheds_load_immediately_rather_than_waiting() {
        // capacity 2: one running, one queued, third rejected outright.
        let gate = Gate::new(1, 2, Duration::from_secs(30));
        let _running = gate.admit().await.unwrap();

        let queued = tokio::spawn({
            let gate = gate.clone();
            async move { gate.admit().await.map(|_| ()) }
        });
        // Let the queued request register its depth.
        tokio::time::sleep(Duration::from_millis(20)).await;

        let started = std::time::Instant::now();
        let rejected = gate.admit().await;
        assert!(
            matches!(rejected, Err(Rejected::QueueFull { .. })),
            "{rejected:?}"
        );
        // ! Immediate, ✗ after the 30s wait timeout. Shedding load is only
        // useful if it is fast.
        assert!(started.elapsed() < Duration::from_secs(1));
        queued.abort();
    }

    #[tokio::test]
    async fn a_released_slot_admits_the_next_request() {
        let gate = Gate::new(1, 8, Duration::from_millis(500));
        let first = gate.admit().await.unwrap();
        let gate2 = gate.clone();
        let waiter = tokio::spawn(async move { gate2.admit().await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(20)).await;
        drop(first);
        assert!(waiter.await.unwrap().is_ok(), "queued request never ran");
    }

    #[tokio::test]
    async fn depth_returns_to_zero_after_a_rejection() {
        // ! A leaked reservation would permanently shrink the queue, so the
        // engine would shed more and more load until it accepted nothing.
        let gate = Gate::new(1, 1, Duration::from_millis(10));
        let _held = gate.admit().await.unwrap();
        for _ in 0..20 {
            let _ = gate.admit().await;
        }
        assert_eq!(gate.depth(), 1, "only the held guard should remain");
    }

    #[test]
    fn rejections_carry_an_actionable_hint() {
        let r = Rejected::QueueFull {
            depth: 64,
            capacity: 64,
        };
        assert!(r.message().contains("64/64"));
        assert!(r.hint().contains("retry"));
    }
}
