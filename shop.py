"""
shop.py - the simulated failing system for HackathonShop.

A tiny online shop: Orders -> Queue -> Payments, plus a Sync job and a
Workflow engine. It produces BOTH metrics and log lines (the challenge asks
for "logs and system metrics"). Everything is deterministic given a seed, and
the whole shop can be deep-copied (shop.clone()) so agents can rehearse fixes
on a copy without touching the real one.

Run `python shop.py` to execute a self-test.
"""
from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import copy
import dataclasses

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
WINDOW = 15                 # metrics rates are computed over the last 15 ticks
BASE_ORDER_RATE = 5.0       # orders per tick at normal traffic
SYNC_CHECK_RECORDS = 50     # integrity is measured on the last 50 synced records

ACTIONS = [
    "restart_payments",
    "scale_queue",
    "throttle_traffic",
    "rollback_sync",
    "kill_stuck_job",
]


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class LogLine:
    tick: int
    level: str       # INFO / WARN / ERROR
    service: str     # orders / queue / payment / sync / workflow / recovery
    message: str

    def __str__(self) -> str:
        return f"[t={self.tick:04d}] {self.level:<5} {self.service:<8} {self.message}"


@dataclass
class SyncRecord:
    order_id: str
    amount: float
    original_amount: float
    synced_at: int
    corrupted: bool = False


@dataclass
class OrderService:
    orders_created: int = 0
    next_id: int = 1


@dataclass
class QueueService:
    capacity: int = 100
    buffer: Deque[Tuple[str, float]] = field(default_factory=deque)
    messages_in: int = 0
    messages_out: int = 0
    messages_dropped: int = 0


@dataclass
class PaymentService:
    throughput: int = 8                     # orders it can process per tick
    payments_recorded: int = 0
    total_revenue: float = 0.0
    lost_total: int = 0                     # silently lost by this service
    ledger: Deque[Tuple[int, str, float]] = field(
        default_factory=lambda: deque(maxlen=500))   # (seq, order_id, amount)
    drop_active: bool = False               # Fault 1 toggle
    drop_rate: float = 0.0
    base_throughput: int = 8                # healthy throughput (a restart restores it)
    leak_active: bool = False               # slow-degradation fault: throughput decays
    leak_every: int = 4                     # ... by 1 every this many ticks
    flap_period: int = 0                    # intermittent fault: on/off every N ticks (0 = off)
    flap_rate: float = 0.0


@dataclass
class SyncJob:
    records: Deque[SyncRecord] = field(default_factory=lambda: deque(maxlen=200))
    records_synced: int = 0
    last_seq: int = 0
    run_interval: int = 10
    corruption_active: bool = False         # Fault 3 toggle
    corruption_rate: float = 0.0


@dataclass
class WorkflowJob:
    id: str
    steps_total: int = 5
    steps_completed: int = 0
    stuck: bool = False
    stuck_since: int = -1


@dataclass
class WorkflowEngine:
    jobs: List[WorkflowJob] = field(default_factory=list)
    jobs_completed: int = 0
    deadlock_active: bool = False           # Fault 4 toggle
    next_job: int = 1


@dataclass
class Shop:
    seed: int = 0
    tick: int = 0
    traffic_multiplier: float = 1.0
    orders: OrderService = field(default_factory=OrderService)
    queue: QueueService = field(default_factory=QueueService)
    payments: PaymentService = field(default_factory=PaymentService)
    sync: SyncJob = field(default_factory=SyncJob)
    workflows: WorkflowEngine = field(default_factory=WorkflowEngine)
    logs: Deque[LogLine] = field(default_factory=lambda: deque(maxlen=300))
    history: Deque[dict] = field(default_factory=lambda: deque(maxlen=WINDOW))
    rng: random.Random = field(default=None)  # type: ignore
    # recovery tools that are currently out of service (used to test adaptation)
    disabled_actions: set = field(default_factory=set)
    # shop size (the default values reproduce the original shop exactly)
    order_rate: float = BASE_ORDER_RATE
    # imperfect sensors (all off by default): what the agent SEES is degraded,
    # the shop itself is unchanged
    sensor_noise: float = 0.0          # random error on readings
    metrics_dropout: float = 0.0       # chance a reading is stale (monitoring hiccup)
    log_dropout: float = 0.0           # chance each log line is lost
    obs_rng: random.Random = field(default=None)  # type: ignore
    last_obs: Optional[object] = None
    # cascading failure: sustained queue pressure starts data corruption
    cascade_active: bool = False
    cascade_count: int = 0

    def __post_init__(self):
        if self.rng is None:
            self.rng = random.Random(self.seed)
        if self.obs_rng is None:
            self.obs_rng = random.Random(self.seed * 7919 + 13)

    def clone(self) -> "Shop":
        """Independent deep copy (including the random generator state)."""
        return copy.deepcopy(self)


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def _log(shop: Shop, st: dict, level: str, service: str, message: str) -> None:
    shop.logs.append(LogLine(shop.tick, level, service, message))
    st["logs"][f"{service}:{level}"] += 1


def _step(shop: Shop) -> None:
    shop.tick += 1
    rng = shop.rng
    st = {"created": 0, "paid": 0, "lost_pay": 0, "q_in": 0, "q_drop": 0,
          "logs": Counter()}

    # 1. Orders arrive and try to enter the queue
    rate = shop.order_rate * shop.traffic_multiplier
    n = int(rate) + (1 if rng.random() < rate - int(rate) else 0)
    q = shop.queue
    for _ in range(n):
        oid = f"o{shop.orders.next_id}"
        shop.orders.next_id += 1
        amount = round(rng.uniform(10, 200), 2)
        shop.orders.orders_created += 1
        st["created"] += 1
        q.messages_in += 1
        st["q_in"] += 1
        if len(q.buffer) >= q.capacity:
            q.messages_dropped += 1
            st["q_drop"] += 1
        else:
            q.buffer.append((oid, amount))
    if st["q_drop"]:
        _log(shop, st, "ERROR", "queue",
             f"buffer full ({len(q.buffer)}/{q.capacity}), dropped {st['q_drop']} messages")

    # 2. Payment service drains the queue
    p = shop.payments
    if p.leak_active and shop.tick % p.leak_every == 0:
        p.throughput = max(2, p.throughput - 1)          # slow degradation
    if p.flap_period > 0:                                # intermittent fault
        p.drop_active = (shop.tick // p.flap_period) % 2 == 0
        p.drop_rate = p.flap_rate
    for _ in range(min(p.throughput, len(q.buffer))):
        oid, amount = q.buffer.popleft()
        q.messages_out += 1
        if p.drop_active and rng.random() < p.drop_rate:
            # SILENT failure: no exception, order just vanishes
            p.lost_total += 1
            st["lost_pay"] += 1
            if rng.random() < 0.5:   # real systems only partially log these
                _log(shop, st, "WARN", "payment",
                     f"order {oid} acked but no ledger write (timeout)")
            continue
        p.payments_recorded += 1
        p.total_revenue += amount
        p.ledger.append((p.payments_recorded, oid, amount))
        st["paid"] += 1

    # 2b. Cascade: sustained queue pressure starts corrupting the sync job
    if shop.cascade_active and not shop.sync.corruption_active:
        shop.cascade_count = shop.cascade_count + 1 if len(q.buffer) > 0.8 * q.capacity else 0
        if shop.cascade_count >= 8:
            shop.sync.corruption_active = True
            shop.sync.corruption_rate = 0.10
            shop.cascade_active = False
            _log(shop, st, "WARN", "sync", "checksum mismatch on record (load-induced)")

    # 3. Sync job copies new ledger entries every run_interval ticks
    sy = shop.sync
    if shop.tick % sy.run_interval == 0:
        for seq, oid, amount in p.ledger:
            if seq <= sy.last_seq:
                continue
            rec = SyncRecord(oid, amount, amount, shop.tick)
            if sy.corruption_active and rng.random() < sy.corruption_rate:
                rec.amount = round(amount * rng.uniform(0.8, 1.2), 2)
                rec.corrupted = True
                _log(shop, st, "WARN", "sync", f"checksum mismatch on record {oid}")
            sy.records.append(rec)
            sy.records_synced += 1
            sy.last_seq = seq

    # 4. Workflow engine
    w = shop.workflows
    if shop.tick % 3 == 0 and len(w.jobs) < 200:
        w.jobs.append(WorkflowJob(id=f"j{w.next_job}"))
        w.next_job += 1
    if w.deadlock_active:
        stuck = [j for j in w.jobs if j.stuck]
        if stuck and shop.tick % 5 == 0:
            d = shop.tick - stuck[0].stuck_since
            _log(shop, st, "ERROR", "workflow",
                 f"job {stuck[0].id} holds lock, no progress for {d} ticks, "
                 f"{len(w.jobs) - 1} jobs blocked")
    else:
        for job in w.jobs:
            if job.steps_completed < job.steps_total:
                job.steps_completed += 1
        done = [j for j in w.jobs if j.steps_completed >= j.steps_total]
        w.jobs_completed += len(done)
        w.jobs = [j for j in w.jobs if j.steps_completed < j.steps_total]

    # 5. Occasional healthy heartbeat so logs are not empty
    if shop.tick % 10 == 0 and not st["logs"]:
        _log(shop, st, "INFO", "orders", f"{st['created']} orders this tick, all services nominal")

    shop.history.append(st)


def run_shop(shop: Shop, ticks: int = 1) -> None:
    """Advance the shop by N ticks."""
    for _ in range(ticks):
        _step(shop)


# ----------------------------------------------------------------------------
# Metrics (what the detector and agents see; they never touch shop internals)
# ----------------------------------------------------------------------------
# Features used by the detector. Rates and log counts only (no cumulative
# counters), so a trained model generalises to unseen fault severities.
FEATURE_NAMES = [
    "orders_per_second", "queue_depth", "queue_saturation", "drop_rate",
    "mismatch_rate", "integrity_score", "jobs_pending", "jobs_stuck",
    "stuck_duration", "log_payment_warn", "log_queue_error",
    "log_sync_warn", "log_workflow_error",
]


@dataclass
class MetricsSnapshot:
    tick: int
    orders_created: int
    orders_per_second: float
    queue_depth: int
    queue_capacity: int
    queue_saturation: float
    messages_in: int
    messages_out: int
    messages_dropped: int
    drop_rate: float              # windowed: queue drops / queue arrivals
    payments_recorded: int
    total_revenue: float
    order_payment_mismatch: int   # cumulative orders silently lost by payments
    mismatch_rate: float          # windowed payment loss rate
    records_synced: int
    integrity_score: float        # share of last 50 synced records that are clean
    jobs_pending: int
    jobs_completed: int
    jobs_stuck: int
    stuck_duration: int
    log_warn: int
    log_error: int
    log_payment_warn: int
    log_queue_error: int
    log_sync_warn: int
    log_workflow_error: int
    health_score: float
    recent_logs: List[str] = field(default_factory=list)

    def to_features(self) -> List[float]:
        return [float(getattr(self, name)) for name in FEATURE_NAMES]

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


def collect_metrics(shop: Shop, observe: bool = True) -> MetricsSnapshot:
    """What the agent sees. observe=False gives the exact, noise-free truth
    (used only by the evaluation, never by an agent)."""
    imperfect = observe and (shop.sensor_noise > 0 or shop.metrics_dropout > 0
                             or shop.log_dropout > 0)
    if imperfect and shop.metrics_dropout > 0 and shop.last_obs is not None \
            and shop.obs_rng.random() < shop.metrics_dropout:
        return dataclasses.replace(shop.last_obs, tick=shop.tick)     # stale reading
    h = list(shop.history)
    n_ticks = max(1, len(h))
    created = sum(x["created"] for x in h)
    paid = sum(x["paid"] for x in h)
    lost = sum(x["lost_pay"] for x in h)
    q_in = sum(x["q_in"] for x in h)
    q_drop = sum(x["q_drop"] for x in h)
    logs: Counter = Counter()
    for x in h:
        logs.update(x["logs"])

    q = shop.queue
    saturation = len(q.buffer) / max(1, q.capacity)
    drop_rate = q_drop / max(1, q_in)
    mismatch_rate = lost / max(1, paid + lost)

    recent = list(shop.sync.records)[-SYNC_CHECK_RECORDS:]
    integrity = 1.0 - (sum(r.corrupted for r in recent) / len(recent)) if recent else 1.0

    jobs = shop.workflows.jobs
    stuck = [j for j in jobs if j.stuck]
    stuck_duration = max((shop.tick - j.stuck_since for j in stuck), default=0)

    if imperfect:
        g, nz = shop.obs_rng.gauss, shop.sensor_noise
        if nz > 0:
            mismatch_rate = max(0.0, mismatch_rate + g(0, nz))
            drop_rate = max(0.0, drop_rate + g(0, nz))
            integrity = min(1.0, max(0.0, integrity + g(0, nz * 0.25)))
            created = max(0.0, created * (1 + g(0, nz * 3)))
        if shop.log_dropout > 0:
            keep = 1.0 - shop.log_dropout
            for k in list(logs):
                logs[k] = sum(1 for _ in range(logs[k]) if shop.obs_rng.random() < keep)

    penalty = (
        min(40.0, mismatch_rate * 200)             # silent order loss
        + min(35.0, drop_rate * 100)               # queue dropping messages
        + max(0.0, saturation - 0.8) * 50          # queue nearly full
        + min(30.0, (1.0 - integrity) * 600)       # corrupted sync data
        + min(30.0, stuck_duration * 3)            # deadlocked workflow
    )
    health = max(0.0, 100.0 - penalty)

    snap = MetricsSnapshot(
        tick=shop.tick,
        orders_created=shop.orders.orders_created,
        orders_per_second=created / n_ticks,
        queue_depth=len(q.buffer),
        queue_capacity=q.capacity,
        queue_saturation=saturation,
        messages_in=q.messages_in,
        messages_out=q.messages_out,
        messages_dropped=q.messages_dropped,
        drop_rate=drop_rate,
        payments_recorded=shop.payments.payments_recorded,
        total_revenue=round(shop.payments.total_revenue, 2),
        order_payment_mismatch=shop.payments.lost_total,
        mismatch_rate=mismatch_rate,
        records_synced=shop.sync.records_synced,
        integrity_score=integrity,
        jobs_pending=len(jobs),
        jobs_completed=shop.workflows.jobs_completed,
        jobs_stuck=len(stuck),
        stuck_duration=stuck_duration,
        log_warn=sum(v for k, v in logs.items() if k.endswith(":WARN")),
        log_error=sum(v for k, v in logs.items() if k.endswith(":ERROR")),
        log_payment_warn=logs["payment:WARN"],
        log_queue_error=logs["queue:ERROR"],
        log_sync_warn=logs["sync:WARN"],
        log_workflow_error=logs["workflow:ERROR"],
        health_score=round(health, 2),
        recent_logs=[str(l) for l in list(shop.logs)[-8:]],
    )
    if observe and (shop.metrics_dropout > 0):
        shop.last_obs = snap
    return snap


def compute_health_score(shop: Shop) -> float:
    return collect_metrics(shop).health_score


# ----------------------------------------------------------------------------
# Fault injectors
# ----------------------------------------------------------------------------
def inject_silent_dropper(shop: Shop, drop_rate: float = 0.2) -> None:
    """Fault 1: payment service silently ignores a fraction of orders."""
    shop.payments.drop_active = True
    shop.payments.drop_rate = drop_rate


def inject_queue_overflow(shop: Shop, traffic_multiplier: float = 3.0) -> None:
    """Fault 2: traffic spike the queue cannot absorb."""
    shop.traffic_multiplier = traffic_multiplier


def inject_data_corruption(shop: Shop, corruption_rate: float = 0.08) -> None:
    """Fault 3: sync job corrupts a fraction of the records it copies."""
    shop.sync.corruption_active = True
    shop.sync.corruption_rate = corruption_rate


def inject_workflow_deadlock(shop: Shop) -> None:
    """Fault 4: one job holds a lock forever and blocks the pipeline."""
    w = shop.workflows
    if not w.jobs:
        w.jobs.append(WorkflowJob(id=f"j{w.next_job}"))
        w.next_job += 1
    target = w.jobs[0]
    target.stuck = True
    target.stuck_since = shop.tick
    w.deadlock_active = True


def inject_compound(shop: Shop, drop_rate: float = 0.15,
                    traffic_multiplier: float = 2.5) -> None:
    """Two faults at once."""
    inject_silent_dropper(shop, drop_rate)
    inject_queue_overflow(shop, traffic_multiplier)


def inject_payment_leak(shop: Shop, leak_every: int = 4) -> None:
    """Fault 5 (new): payment service slowly degrades, like a memory leak."""
    shop.payments.leak_active = True
    shop.payments.leak_every = leak_every


def inject_flapping(shop: Shop, drop_rate: float = 0.30, period: int = 12) -> None:
    """Fault 6 (new): payment loss that switches on and off by itself."""
    shop.payments.flap_period = period
    shop.payments.flap_rate = drop_rate


def inject_cascade(shop: Shop, traffic_multiplier: float = 3.5) -> None:
    """Fault 7 (new): a traffic spike that, if it lasts, starts corrupting data."""
    shop.traffic_multiplier = traffic_multiplier
    shop.cascade_active = True
    shop.cascade_count = 0


# shop sizes for testing on a configuration nothing was tuned on
PROFILES = {
    "default": {},
    "large": {"order_rate": 8.0, "throughput": 13, "capacity": 160},
    "small": {"order_rate": 3.5, "throughput": 6, "capacity": 60},
}


def make_shop(profile: str = "default", seed: int = 0) -> Shop:
    shop = Shop(seed=seed)
    cfg = PROFILES[profile]
    if cfg:
        shop.order_rate = cfg["order_rate"]
        shop.payments.throughput = shop.payments.base_throughput = cfg["throughput"]
        shop.queue.capacity = cfg["capacity"]
    return shop


FAULTS = {
    "silent_dropper": inject_silent_dropper,
    "queue_overflow": inject_queue_overflow,
    "data_corruption": inject_data_corruption,
    "workflow_deadlock": inject_workflow_deadlock,
    "compound": inject_compound,
    "payment_leak": inject_payment_leak,
    "flapping": inject_flapping,
    "cascade": inject_cascade,
}


# ----------------------------------------------------------------------------
# Recovery actions (the tools an agent can use)
# ----------------------------------------------------------------------------
def apply_action(shop: Shop, action: str, strength: float = 1.0) -> str:
    """
    Apply one recovery action. strength in [0,1]: gentle = partial effect,
    aggressive = full effect. Returns a short human-readable description.
    """
    strength = max(0.0, min(1.0, strength))
    st = {"logs": Counter()}
    p, q, sy, w = shop.payments, shop.queue, shop.sync, shop.workflows

    if action in shop.disabled_actions:
        msg = f"{action} is unavailable (tool outage) - no effect"

    elif action == "restart_payments":
        if p.flap_period > 0:
            p.flap_rate *= (1.0 - strength)
            if p.flap_rate < 0.01:
                p.flap_rate, p.flap_period = 0.0, 0
                p.drop_active, p.drop_rate = False, 0.0
        if p.leak_active:
            if strength >= 0.6:
                p.leak_active = False
                p.throughput = max(p.throughput, p.base_throughput)
            else:
                p.throughput = min(p.base_throughput,
                                   p.throughput + int(round(strength * 6)))
        if p.drop_active:
            p.drop_rate *= (1.0 - strength)
            if p.drop_rate < 0.01:
                p.drop_rate, p.drop_active = 0.0, False
        msg = f"restarted payment service (strength {strength:.2f})"

    elif action == "scale_queue":
        q.capacity = min(1000, int(q.capacity * (1 + 2 * strength)))
        p.throughput = min(40, p.throughput + int(2 + 10 * strength))
        msg = f"scaled queue to capacity {q.capacity}, payment throughput {p.throughput}"

    elif action == "throttle_traffic":
        shop.traffic_multiplier = max(1.0, shop.traffic_multiplier * (1 - 0.6 * strength))
        msg = f"throttled traffic to x{shop.traffic_multiplier:.2f}"

    elif action == "rollback_sync":
        for r in sy.records:
            if r.corrupted:
                r.amount, r.corrupted = r.original_amount, False
        if sy.corruption_active:
            sy.corruption_rate *= (1.0 - strength)
            if sy.corruption_rate < 0.005:
                sy.corruption_rate, sy.corruption_active = 0.0, False
        msg = "rolled back sync to last good state"

    elif action == "kill_stuck_job":
        killed = [j.id for j in w.jobs if j.stuck]
        w.jobs = [j for j in w.jobs if not j.stuck]
        w.deadlock_active = False
        msg = f"killed stuck jobs {killed}" if killed else "no stuck job found"

    else:
        msg = f"unknown action '{action}' (no effect)"

    shop.logs.append(LogLine(shop.tick, "INFO", "recovery", msg))
    return msg


# ----------------------------------------------------------------------------
# Self-test: python shop.py
# ----------------------------------------------------------------------------
def _self_test() -> None:
    print("== Healthy shop ==")
    s = Shop(seed=1)
    run_shop(s, 200)
    m = collect_metrics(s)
    print(f"tick={m.tick} health={m.health_score} saturation={m.queue_saturation:.2f} "
          f"jobs_pending={m.jobs_pending} integrity={m.integrity_score:.2f}")
    assert m.health_score >= 95, "healthy shop should score >= 95"

    cases = [
        ("silent_dropper", ["restart_payments"]),
        ("queue_overflow", ["scale_queue"]),
        ("data_corruption", ["rollback_sync"]),
        ("workflow_deadlock", ["kill_stuck_job"]),
        ("compound", ["restart_payments", "scale_queue"]),
    ]
    all_ok = True
    for name, fixes in cases:
        shop = Shop(seed=2)
        run_shop(shop, 50)
        FAULTS[name](shop)
        run_shop(shop, 60)
        sick = collect_metrics(shop)
        for a in fixes:
            apply_action(shop, a, 1.0)
        run_shop(shop, 40)
        fixed = collect_metrics(shop)
        ok = sick.health_score < 75 and fixed.health_score >= 80
        all_ok &= ok
        print(f"{name:<18} sick={sick.health_score:>6.1f}  after fix={fixed.health_score:>6.1f}  "
              f"{'OK' if ok else 'PROBLEM'}")
        print(f"   sample log: {sick.recent_logs[-1] if sick.recent_logs else '(none)'}")

    # Determinism: same seed -> same result; clones are independent
    a, b = Shop(seed=7), Shop(seed=7)
    run_shop(a, 80)
    run_shop(b, 80)
    assert collect_metrics(a).orders_created == collect_metrics(b).orders_created
    c = a.clone()
    run_shop(c, 20)
    assert a.tick == 80 and c.tick == 100, "clone must not affect the original"
    print("determinism + clone independence: OK")
    print("\nALL TESTS PASSED" if all_ok else "\nSOME TESTS FAILED - send me this output")


if __name__ == "__main__":
    _self_test()
