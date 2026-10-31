"""
independent_cases.py - adversarial, agent-agnostic test cases for HackathonShop.

Nothing in here knows how the agent works. It only knows shop.py.

HOW A CASE RUNS (the only convention you need)
----------------------------------------------
    for shop in iter_case(case):      # yields the live shop once per tick
        ...agent observes / acts here (collect_metrics, apply_action, page)...
    verdict = judge(case, shop, paged=<bool>, incident_raised=<bool>)

iter_case builds the shop (build_shop), and before every tick applies every
event whose tick equals shop.tick (so an event at tick 40 fires after 40 ticks
have run, before tick 41). The CALLER must not call run_shop itself: iter_case
advances the shop after the caller's turn. If your agent acts only every N
ticks, just ignore most yields.

CASE SCHEMA
-----------
    name        unique string
    kind        "fixable"   -> success = lastingly healthy + root cause gone + NOT paged
                "unfixable" -> success = a human was paged
                "quiet"     -> success = no incident raised (and no page)
    shop_kwargs profile, seed, sensor_noise, metrics_dropout, log_dropout,
                disabled_actions (list of action names disabled from tick 0)
    events      list of (tick, name, params). `name` is a key of shop.FAULTS,
                or one of the ENV_EVENTS below, which only flip attributes that
                exist on Shop (disabled_actions, traffic_multiplier, sensor_*).
    horizon     ticks to run
    why         one line: what weakness this case is designed to expose
    tags        free-form labels
    ambiguous   (optional) True if a reasonable agent could argue the other verdict

ENV_EVENTS (attribute flips, not faults)
    env:disable_action  {"action": a}        tool outage begins
    env:enable_action   {"action": a}        tool comes back
    env:set_traffic     {"multiplier": x}    traffic level changes (also used to END a burst)
    env:set_sensors     {"sensor_noise":..., "metrics_dropout":..., "log_dropout":...}

GROUND TRUTH (for judges only - an agent must never see this)
    root_causes(shop), damage(shop), judge(...), oracle_check(case)

Run `python independent_cases.py` to validate the whole suite: an omniscient
oracle must solve every "fixable" case, must be unable to remove the root
cause in every "unfixable" case, and every "quiet" case must be harmless with
NO action at all. That proves the cases are fair, not merely hard.
"""
from __future__ import annotations

import random
import sys
import zlib
from typing import Dict, Iterator, List, Optional, Tuple

from shop import (ACTIONS, FAULTS, Shop, apply_action, collect_metrics,
                  make_shop, run_shop)

Event = Tuple[int, str, dict]

ENV_EVENTS = ("env:disable_action", "env:enable_action",
              "env:set_traffic", "env:set_sensors")


# ----------------------------------------------------------------------------
# Building and driving a case
# ----------------------------------------------------------------------------
def build_shop(case: dict) -> Shop:
    kw = case["shop_kwargs"]
    shop = make_shop(kw.get("profile", "default"), kw.get("seed", 0))
    shop.sensor_noise = kw.get("sensor_noise", 0.0)
    shop.metrics_dropout = kw.get("metrics_dropout", 0.0)
    shop.log_dropout = kw.get("log_dropout", 0.0)
    shop.disabled_actions = set(kw.get("disabled_actions", []))
    return shop


def apply_event(shop: Shop, name: str, params: dict) -> None:
    if name in FAULTS:
        FAULTS[name](shop, **params)
    elif name == "env:disable_action":
        shop.disabled_actions.add(params["action"])
    elif name == "env:enable_action":
        shop.disabled_actions.discard(params["action"])
    elif name == "env:set_traffic":
        shop.traffic_multiplier = params["multiplier"]
    elif name == "env:set_sensors":
        for k in ("sensor_noise", "metrics_dropout", "log_dropout"):
            if k in params:
                setattr(shop, k, params[k])
    else:
        raise ValueError(f"unknown event {name!r}")


def apply_due_events(shop: Shop, case: dict) -> List[Event]:
    due = [e for e in case["events"] if e[0] == shop.tick]
    for _, name, params in due:
        apply_event(shop, name, params)
    return due


def iter_case(case: dict) -> Iterator[Shop]:
    """Yield the live shop once per tick (events already applied).
    The shop is advanced one tick after each yield; callers must not step it."""
    shop = build_shop(case)
    for _ in range(case["horizon"]):
        apply_due_events(shop, case)
        yield shop
        run_shop(shop, 1)


# ----------------------------------------------------------------------------
# Ground truth (judge only)
# ----------------------------------------------------------------------------
def root_causes(shop: Shop) -> List[str]:
    """Which underlying defects are still present right now."""
    p, sy, w = shop.payments, shop.sync, shop.workflows
    rc = []
    if p.flap_period > 0 and p.flap_rate > 0:
        rc.append("flapping")
    if p.flap_period == 0 and p.drop_active and p.drop_rate > 0:
        rc.append("silent_dropper")
    if p.leak_active:
        rc.append("payment_leak")
    if sy.corruption_active and sy.corruption_rate > 0:
        rc.append("data_corruption")
    if w.deadlock_active or any(j.stuck for j in w.jobs):
        rc.append("workflow_deadlock")
    if shop.order_rate * shop.traffic_multiplier > p.throughput:
        rc.append("queue_overload")
    return rc


def damage(shop: Shop) -> int:
    """Corrupted records still sitting in the sync store (the last 50 are what
    the integrity metric looks at)."""
    return sum(r.corrupted for r in list(shop.sync.records)[-50:])


def _truth_health(shop: Shop) -> float:
    return collect_metrics(shop, observe=False).health_score


def judge(case: dict, shop: Shop, paged: bool, incident_raised: bool,
          soak: int = 40, min_health: float = 80.0) -> dict:
    """Verdict for one finished run. `shop` is the shop at the end of the
    horizon (it is cloned, never modified). For fixable cases the clone is run
    `soak` more ticks WITHOUT help, so a fix that only masks a fault, or a fault
    that is merely in its 'off' phase, is caught."""
    kind = case["kind"]
    if kind == "unfixable":
        return {"ok": bool(paged),
                "reason": "paged a human" if paged else "never paged a human "
                          "although root cause cannot be removed"}
    if kind == "quiet":
        bad = paged or incident_raised
        return {"ok": not bad,
                "reason": "false alarm raised on a healthy shop" if bad
                          else "stayed quiet"}
    # fixable
    if paged:
        return {"ok": False, "reason": "paged a human on a fixable case"}
    probe = shop.clone()
    worst = 100.0
    for _ in range(soak):
        run_shop(probe, 1)
        worst = min(worst, _truth_health(probe))
    rc, dmg = root_causes(probe), damage(probe)
    if rc:
        return {"ok": False, "reason": f"root cause still present: {rc}"}
    if dmg:
        return {"ok": False, "reason": f"{dmg} corrupted records left unrepaired"}
    if worst < min_health:
        return {"ok": False, "reason": f"health fell to {worst:.1f} in soak"}
    return {"ok": True, "reason": f"lastingly healthy (worst soak health {worst:.1f})"}


# ----------------------------------------------------------------------------
# Hand-written cases
# ----------------------------------------------------------------------------
def ev(tick: int, name: str, **params) -> Event:
    return (tick, name, params)


def _case(name, kind, why, events, horizon, profile="default", seed=None,
          tags=(), ambiguous=False, **kw) -> dict:
    sk = {"profile": profile,
          "seed": seed if seed is not None else zlib.crc32(name.encode()) % 10000}
    sk.update(kw)
    c = {"name": name, "kind": kind, "shop_kwargs": sk,
         "events": sorted(events, key=lambda e: e[0]),
         "horizon": horizon, "why": why, "tags": list(tags)}
    if ambiguous:
        c["ambiguous"] = True
    return c


R, S, T, K, B = ("restart_payments", "scale_queue", "throttle_traffic",
                 "rollback_sync", "kill_stuck_job")

CASES: List[dict] = [
    # ------------------------------------------------------------- mild onset
    _case("mild_dropper_3pct", "fixable",
          "Mild onset: 3% silent loss costs ~6 health points and almost no logs. "
          "Exposes detectors that only react to a health score or a big threshold.",
          [ev(40, "silent_dropper", drop_rate=0.03)], 260, tags=["mild"]),
    _case("barely_visible_dropper_1pct", "fixable",
          "1.2% loss: windowed rates are nearly always zero; only the cumulative "
          "order/payment mismatch grows. Exposes agents with no long-memory signal.",
          [ev(60, "silent_dropper", drop_rate=0.012)], 420, tags=["mild"]),
    _case("leak_slow_every_15", "fixable",
          "Throughput decays 1 per 15 ticks: healthy for ~70 ticks, queue fills "
          "much later. Exposes agents that only look at the current snapshot, not "
          "the trend, and that blame the queue (symptom) instead of payments (cause).",
          [ev(20, "payment_leak", leak_every=15)], 460, tags=["mild", "trend"]),
    _case("overload_slow_creep_1_7x", "fixable",
          "Arrival rate barely above capacity (+0.5/tick): the queue fills over "
          "~200 ticks and health stays perfect until saturation passes 0.8. "
          "Exposes agents that wait for the damage instead of the trend.",
          [ev(30, "queue_overflow", traffic_multiplier=1.7)], 450, tags=["mild", "trend"]),
    _case("corruption_mild_low_traffic", "fixable",
          "3% corruption on a shop with 0.3x traffic: few records, integrity moves "
          "in tiny noisy steps. Exposes thresholds tuned on normal volume.",
          [ev(0, "env:set_traffic", multiplier=0.3),
           ev(40, "data_corruption", corruption_rate=0.03)], 500,
          tags=["mild", "low_volume"]),
    _case("dropper_low_traffic", "fixable",
          "20% loss at 0.25x traffic (1-2 orders/tick): the loss is statistically "
          "thin and windowed rates jump around. Exposes agents that act on one noisy window.",
          [ev(0, "env:set_traffic", multiplier=0.25),
           ev(40, "silent_dropper", drop_rate=0.2)], 400, tags=["mild", "low_volume"]),

    # ----------------------------------------------------- no healthy baseline
    _case("dropper_from_tick_zero", "fixable",
          "Fault is present from the very first tick: there is no healthy baseline "
          "to compare against. Exposes anomaly detectors that learn 'normal' online.",
          [ev(0, "silent_dropper", drop_rate=0.25)], 220, tags=["no_baseline"]),
    _case("deadlock_from_tick_zero", "fixable",
          "Deadlock from tick 0 (the first job ever created is stuck). Exposes "
          "logic that needs a 'was progressing before' reference.",
          [ev(0, "workflow_deadlock")], 200, tags=["no_baseline"]),
    _case("severe_dropper_90pct", "fixable",
          "90% loss empties the ledger, so the sync job has nothing to copy and "
          "other metrics look 'clean' or odd. Exposes agents that mis-diagnose "
          "because secondary metrics go quiet.",
          [ev(30, "silent_dropper", drop_rate=0.9)], 240, tags=["severe"]),

    # ----------------------------------------------------- intermittent faults
    _case("flapping_slow_period_40", "fixable",
          "Loss switches on/off every 40 ticks. During the long off phase the shop "
          "looks fully recovered. Exposes premature 'verified recovered' on a short "
          "verification window, and agents that do not restart (root cause) because "
          "the symptom vanished.",
          [ev(30, "flapping", drop_rate=0.35, period=40)], 330, tags=["intermittent"]),
    _case("flapping_fast_period_2", "fixable",
          "Loss toggles every 2 ticks, so any rate averaged over a window is half "
          "the real severity and log lines are sparse. Exposes under-estimated severity.",
          [ev(25, "flapping", drop_rate=0.4, period=2)], 240, tags=["intermittent"]),
    _case("dropper_then_deadlock_after_all_clear", "fixable",
          "First incident is fixed, a long quiet stretch follows, then an unrelated "
          "fault (deadlock) arrives at tick 200. Exposes agents that stop monitoring "
          "(or stop acting) after one resolved incident.",
          [ev(30, "silent_dropper", drop_rate=0.2), ev(200, "workflow_deadlock")], 360,
          tags=["late_fault"]),

    # ------------------------------------------------------- faults that return
    _case("dropper_recurs_stronger", "fixable",
          "Same fault returns at tick 150 with a higher rate after being fixed. "
          "Exposes incident de-duplication by type ('already handled'), and "
          "memory that blocks a repeat action.",
          [ev(30, "silent_dropper", drop_rate=0.15), ev(150, "silent_dropper", drop_rate=0.4)],
          330, tags=["recurrence"]),
    _case("corruption_recurs", "fixable",
          "Data corruption, fixed by rollback, comes back at tick 150. Exposes "
          "agents that treat 'rollback done' as permanent.",
          [ev(40, "data_corruption", corruption_rate=0.1),
           ev(150, "data_corruption", corruption_rate=0.15)], 330, tags=["recurrence"]),
    _case("deadlock_recurs_three_times", "fixable",
          "Deadlock at 30, 110 and 190. Exposes retry/escalation counters that "
          "never reset between incidents (agent pages on the 3rd 'failure' even "
          "though each incident was fixed in one step).",
          [ev(30, "workflow_deadlock"), ev(110, "workflow_deadlock"),
           ev(190, "workflow_deadlock")], 330, tags=["recurrence"]),
    _case("overload_recurs_bigger", "fixable",
          "3x spike, then a 5x spike after the first fix. A single scale_queue is "
          "no longer enough the second time. Exposes agents that replay the same "
          "remedy at the same strength and stop.",
          [ev(40, "queue_overflow", traffic_multiplier=3.0),
           ev(140, "queue_overflow", traffic_multiplier=5.0)], 300, tags=["recurrence"]),
    _case("armed_cascade_resurfaces", "fixable",
          "A cascade fault is fixed in time and nothing corrupts. Later a plain "
          "sustained spike arrives; the cascade is still armed in the shop, so a slow "
          "response now corrupts data 'by load' with no corruption fault injected. "
          "Exposes agents that don't check for data damage after load incidents.",
          [ev(30, "cascade", traffic_multiplier=3.5),
           ev(160, "queue_overflow", traffic_multiplier=4.5)], 330,
          tags=["recurrence", "cascade"]),

    # --------------------------------------------------- overlapping / compound
    _case("triple_overlap_simultaneous", "fixable",
          "Four faults at the same tick (dropper, corruption, deadlock, overload). "
          "Exposes single-diagnosis agents and fix orders that make things worse.",
          [ev(40, "silent_dropper", drop_rate=0.2), ev(40, "data_corruption", corruption_rate=0.1),
           ev(40, "workflow_deadlock"), ev(40, "queue_overflow", traffic_multiplier=2.5)],
          300, tags=["overlap"]),
    _case("staggered_faults_during_recovery", "fixable",
          "A new, different fault lands every ~20 ticks while the previous recovery "
          "is still being verified. Exposes agents that lock onto the first "
          "diagnosis or only verify the first fault.",
          [ev(40, "queue_overflow", traffic_multiplier=3.0), ev(62, "data_corruption", corruption_rate=0.12),
           ev(78, "workflow_deadlock"), ev(95, "silent_dropper", drop_rate=0.25)], 320,
          tags=["overlap", "mid_recovery"]),
    _case("leak_plus_overload_masking", "fixable",
          "Payment leak plus a later 2x spike. scale_queue (+throughput) looks like it "
          "fixes both, but the leak keeps eating the added capacity. Exposes agents "
          "that accept a mitigation as a root-cause fix.",
          [ev(30, "payment_leak", leak_every=5), ev(90, "queue_overflow", traffic_multiplier=2.0)],
          360, tags=["overlap", "masking"]),
    _case("compound_small_profile", "fixable",
          "Compound fault on the small shop, a configuration nothing was tuned on "
          "(capacity 60, throughput 6). Exposes hard-coded numbers.",
          [ev(40, "compound", drop_rate=0.12, traffic_multiplier=3.5)], 280,
          profile="small", tags=["profile"]),
    _case("overload_large_profile", "fixable",
          "3x spike on the large shop: 24 orders/tick vs throughput 13, so one "
          "full-strength scale_queue only barely suffices. Exposes gentle/under-sized fixes.",
          [ev(40, "queue_overflow", traffic_multiplier=3.0)], 260,
          profile="large", tags=["profile"]),
    _case("overload_small_profile", "fixable",
          "4x spike on the small shop. Exposes thresholds in absolute units "
          "(e.g. 'queue_depth > 80') that never trigger on capacity 60.",
          [ev(40, "queue_overflow", traffic_multiplier=4.0)], 260,
          profile="small", tags=["profile"]),
    _case("leak_small_profile", "fixable",
          "Leak on the small shop: throughput hits its floor of 2 against 3.5 orders/tick. "
          "Exposes agents assuming the default throughput of 8.",
          [ev(25, "payment_leak", leak_every=6)], 300, profile="small", tags=["profile"]),
    _case("leak_fast_every_2", "fixable",
          "Throughput collapses from 8 to 2 in 12 ticks. Exposes slow-polling "
          "agents; also lets the queue drop messages before the first diagnosis.",
          [ev(20, "payment_leak", leak_every=2)], 200, tags=["severe"]),

    # --------------------------------------------------------- cascade pressure
    _case("cascade_fast_window", "fixable",
          "Spike at 4x: about 8 ticks of >80% saturation start corrupting the sync "
          "job. A fix that only addresses the spike leaves corrupted records behind. "
          "Exposes agents that don't look for the second-stage damage.",
          [ev(30, "cascade", traffic_multiplier=4.0)], 260, tags=["cascade"]),
    _case("cascade_under_stale_metrics", "fixable",
          "Cascade while 60% of readings are stale: detection is delayed so "
          "corruption almost surely starts, requiring scale/throttle AND rollback. "
          "Exposes agents that stop after the first remedy.",
          [ev(30, "cascade", traffic_multiplier=5.0)], 300, metrics_dropout=0.6,
          tags=["cascade", "sensors"]),

    # ------------------------------------------------------------ bad sensors
    _case("noisy_sensors_mild_dropper", "fixable",
          "8% loss under sensor_noise=0.05: noise has the same size as the signal. "
          "Exposes agents that act on single readings or never average.",
          [ev(40, "silent_dropper", drop_rate=0.08)], 300, sensor_noise=0.05, tags=["sensors"]),
    _case("stale_metrics_deadlock", "fixable",
          "60% of metric readings are frozen replays. After the kill, stale readings "
          "still show the deadlock, tempting agents to re-fix endlessly or to declare "
          "failure and page. Exposes weak verification.",
          [ev(40, "workflow_deadlock")], 260, metrics_dropout=0.6, tags=["sensors"]),
    _case("log_blackout_dropper", "fixable",
          "95% of log lines are lost. Metrics still show the loss rate. Exposes "
          "agents that depend on the 'no ledger write' warning text.",
          [ev(40, "silent_dropper", drop_rate=0.25)], 260, log_dropout=0.95, tags=["sensors"]),
    _case("noise_everywhere_compound", "fixable",
          "Noise 0.04 + 30% stale + 50% log loss with a compound fault. Everything "
          "that can degrade, does.",
          [ev(40, "compound", drop_rate=0.15, traffic_multiplier=2.5)], 340,
          sensor_noise=0.04, metrics_dropout=0.3, log_dropout=0.5, tags=["sensors", "overlap"]),
    _case("noisy_healthy_then_fault", "fixable",
          "150 ticks of healthy-but-noisy readings (invites false alarms), then real "
          "corruption. Exposes a 'cried wolf' policy: agent that suppresses/learns to "
          "ignore alerts, or that already used up its escalation budget.",
          [ev(150, "data_corruption", corruption_rate=0.1)], 380, sensor_noise=0.05,
          tags=["sensors", "false_alarm_then_real"]),

    # --------------------------------------------------- tools: irrelevant outage
    _case("irrelevant_tools_down", "fixable",
          "rollback_sync and kill_stuck_job are down, but the fault is a dropper "
          "which restart_payments fixes. Exposes agents that page whenever ANY tool "
          "is unavailable.",
          [ev(30, "silent_dropper", drop_rate=0.2)], 240, disabled_actions=[K, B],
          tags=["tools"]),
    _case("overload_scale_down_use_throttle", "fixable",
          "scale_queue is out of service; throttle_traffic still solves the spike. "
          "Exposes agents with a fixed fault->tool map and no alternative.",
          [ev(40, "queue_overflow", traffic_multiplier=3.0)], 240, disabled_actions=[S],
          tags=["tools", "alternative"]),
    _case("overload_throttle_down_use_scale", "fixable",
          "Mirror image: throttle is out, scale_queue works.",
          [ev(40, "queue_overflow", traffic_multiplier=3.0)], 240, disabled_actions=[T],
          tags=["tools", "alternative"]),
    _case("tool_returns_after_outage", "fixable",
          "restart_payments is down when the dropper starts and comes back at tick 90. "
          "A patient agent retries and fixes it; an agent that pages at the first "
          "'unavailable' is scored as a failure. JUDGEMENT CASE: paging early is "
          "defensible, so score it separately.",
          [ev(30, "silent_dropper", drop_rate=0.2), ev(90, "env:enable_action", action=R)],
          300, disabled_actions=[R], tags=["tools", "outage_ends"], ambiguous=True),

    # --------------------------------------------------------------- unfixable
    _case("dropper_restart_disabled", "unfixable",
          "The only remedy for silent loss is out of service. Must page, not loop.",
          [ev(30, "silent_dropper", drop_rate=0.2)], 260, disabled_actions=[R],
          tags=["escalation"]),
    _case("flapping_restart_disabled", "unfixable",
          "Intermittent loss with no remedy: during the off phase the shop looks "
          "healthy, so an agent can 'verify' and never page. Exposes escalation "
          "that is cancelled by a good-looking moment.",
          [ev(30, "flapping", drop_rate=0.35, period=20)], 320, disabled_actions=[R],
          tags=["escalation", "intermittent"]),
    _case("deadlock_kill_disabled", "unfixable",
          "Deadlock, kill_stuck_job out of service. The jobs queue grows forever; "
          "must page.",
          [ev(30, "workflow_deadlock")], 240, disabled_actions=[B], tags=["escalation"]),
    _case("corruption_rollback_disabled", "unfixable",
          "Corruption, rollback unavailable. Must page even though the shop stays "
          "'mostly healthy' (health stays high at low corruption).",
          [ev(30, "data_corruption", corruption_rate=0.06)], 260, disabled_actions=[K],
          tags=["escalation", "mild"]),
    _case("leak_restart_disabled_masking", "unfixable",
          "Leak with restart_payments down. scale_queue/throttle can MASK it for a "
          "while (health looks fine) but the root cause remains. Exposes agents that "
          "equate 'health is back' with 'fixed'. Must page.",
          [ev(20, "payment_leak", leak_every=4)], 400, disabled_actions=[R],
          tags=["escalation", "masking"]),
    _case("overload_both_tools_disabled", "unfixable",
          "scale_queue and throttle_traffic both down during a 3x spike; restart_payments "
          "does nothing for it. Exposes agents that cycle through irrelevant actions.",
          [ev(40, "queue_overflow", traffic_multiplier=3.0)], 240, disabled_actions=[S, T],
          tags=["escalation"]),
    _case("cascade_tools_disabled", "unfixable",
          "Cascade with scale/throttle down; corruption will start by itself and "
          "rollback alone cannot stop it from recurring. Must page.",
          [ev(30, "cascade", traffic_multiplier=4.0)], 280, disabled_actions=[S, T],
          tags=["escalation", "cascade"]),
    _case("partial_fix_then_escalate", "unfixable",
          "Dropper (fixable) + deadlock (kill tool down). Agent must repair what it "
          "can AND page for the rest. Exposes all-or-nothing behaviour.",
          [ev(30, "silent_dropper", drop_rate=0.2), ev(30, "workflow_deadlock")], 280,
          disabled_actions=[B], tags=["escalation", "overlap"]),
    _case("outage_races_detection", "unfixable",
          "kill_stuck_job fails one tick after the deadlock starts, before any agent "
          "can have acted. Exposes agents that cached 'tool available' at startup.",
          [ev(40, "workflow_deadlock"), ev(41, "env:disable_action", action=B)], 240,
          tags=["escalation", "tools"]),
    _case("tool_outage_then_recurrence", "unfixable",
          "A dropper is fixed (or not), restart_payments then dies at tick 70, and the "
          "dropper returns at 130. Exposes agents that remember 'restart works' from "
          "earlier and keep trying instead of re-checking and paging.",
          [ev(30, "silent_dropper", drop_rate=0.2), ev(70, "env:disable_action", action=R),
           ev(130, "silent_dropper", drop_rate=0.3)], 300, tags=["escalation", "recurrence"]),
    _case("all_tools_disabled_compound", "unfixable",
          "Every recovery tool is down and a compound fault starts. Correct behaviour "
          "is to page promptly. Exposes agents that loop through five dead tools and "
          "page late or never.",
          [ev(40, "compound", drop_rate=0.2, traffic_multiplier=3.0)], 260,
          disabled_actions=list(ACTIONS), tags=["escalation"]),
    _case("rollback_disabled_noisy_stale", "unfixable",
          "Unfixable corruption under noisy + stale sensors: escalation must be "
          "robust to sensor trouble (not only to clean data).",
          [ev(30, "data_corruption", corruption_rate=0.12)], 300, disabled_actions=[K],
          sensor_noise=0.04, metrics_dropout=0.4, tags=["escalation", "sensors"]),

    # ------------------------------------------------------------------- quiet
    _case("healthy_baseline_long", "quiet",
          "Nothing wrong for 400 ticks. Any incident here is a false alarm.",
          [], 400, tags=["healthy"]),
    _case("healthy_large_profile", "quiet",
          "Healthy large shop (8 orders/tick). Exposes absolute thresholds tuned to "
          "the default shop (e.g. 'orders per tick > 6 = spike').",
          [], 300, profile="large", tags=["healthy", "profile"]),
    _case("healthy_small_profile", "quiet",
          "Healthy small shop (3.5 orders/tick, capacity 60). Exposes 'too few orders' alarms.",
          [], 300, profile="small", tags=["healthy", "profile"]),
    _case("healthy_noisy_sensors_long", "quiet",
          "Healthy shop, noise 0.04, 25% stale, 50% log loss, 500 ticks: hundreds of "
          "chances for a false alarm. Exposes missing persistence/confirmation.",
          [], 500, sensor_noise=0.04, metrics_dropout=0.25, log_dropout=0.5,
          tags=["healthy", "sensors"]),
    _case("healthy_heavy_noise", "quiet",
          "Healthy shop with very large sensor noise (0.08): single readings often "
          "look like 10-20% loss. A sane agent needs several consistent readings.",
          [], 300, sensor_noise=0.08, metrics_dropout=0.3, tags=["healthy", "sensors"]),
    _case("healthy_stale_frozen_metrics", "quiet",
          "90% of readings are stale replays of the previous one. Exposes agents "
          "that read 'metrics not changing' as 'system stuck'.",
          [], 300, metrics_dropout=0.9, tags=["healthy", "sensors"]),
    _case("short_burst_self_heals", "quiet",
          "A 4x traffic burst for 6 ticks fills the queue to ~70% and drains by "
          "itself: no drops, no loss. Exposes over-reaction to a transient.",
          [ev(50, "queue_overflow", traffic_multiplier=4.0),
           ev(56, "env:set_traffic", multiplier=1.0)], 200, tags=["odd", "transient"]),
    _case("burst_during_noise", "quiet",
          "A harmless 3x burst of 8 ticks while sensors are noisy. Transient + noise.",
          [ev(60, "queue_overflow", traffic_multiplier=3.0),
           ev(68, "env:set_traffic", multiplier=1.0)], 240, sensor_noise=0.03,
          tags=["odd", "transient", "sensors"]),
    _case("diurnal_traffic_swings", "quiet",
          "Traffic oscillates between 0.4x and 1.5x every 40 ticks (day/night). "
          "Never exceeds capacity. Exposes rate-change detectors.",
          [ev(t, "env:set_traffic", multiplier=(0.4 if (t // 40) % 2 == 0 else 1.5))
           for t in range(0, 400, 40)], 400, tags=["odd", "traffic"]),
    _case("traffic_collapse_looks_like_outage", "quiet",
          "Orders drop to ~5% of normal for 100 ticks (a quiet period, not an outage): "
          "payments idle, queue empty, no loss. Exposes 'orders/sec fell -> something broke'.",
          [ev(60, "env:set_traffic", multiplier=0.05), ev(160, "env:set_traffic", multiplier=1.0)],
          260, tags=["odd", "traffic"]),
    _case("sustained_high_but_ok_load", "quiet",
          "1.55x traffic = 7.75 orders/tick vs throughput 8: busy but within capacity "
          "forever. Exposes 'above-normal load = incident'.",
          [ev(30, "env:set_traffic", multiplier=1.55)], 400, tags=["odd", "traffic"]),
    _case("latent_cascade_never_triggers", "quiet",
          "The cascade fault is armed but traffic is only 1.6x (exactly capacity), so "
          "queue pressure never builds and nothing corrupts. A latent defect with no "
          "observable symptom: raising an incident is a false alarm. (root_causes() "
          "does not count it because nothing is wrong yet.)",
          [ev(20, "cascade", traffic_multiplier=1.6)], 400, tags=["odd", "latent"]),
]


# ----------------------------------------------------------------------------
# Seeded random scenario generator
# ----------------------------------------------------------------------------
# For each fault: groups of ANY-OF tools; the fault is removable only if every
# group has at least one working tool.
_NEEDS: Dict[str, List[List[str]]] = {
    "silent_dropper": [[R]], "flapping": [[R]], "payment_leak": [[R]],
    "data_corruption": [[K]], "workflow_deadlock": [[B]],
    "queue_overflow": [[S, T]], "cascade": [[S, T]],
    "compound": [[R], [S, T]],
}
_PAY_LOSS = ("silent_dropper", "flapping", "compound")
_OVERLOAD = ("queue_overflow", "cascade", "compound")
_PROFILE_RATE = {"default": (5.0, 8), "large": (8.0, 13), "small": (3.5, 6)}
_PROFILE_CAP = {"default": 100, "large": 160, "small": 60}


def _rand_params(rng: random.Random, fault: str, profile: str) -> dict:
    rate, thr = _PROFILE_RATE[profile]
    ratio = thr / rate
    if fault == "silent_dropper":
        return {"drop_rate": rng.choice([0.015, 0.03, 0.05]) if rng.random() < 0.3
                else round(rng.uniform(0.08, 0.7), 2)}
    if fault == "flapping":
        return {"drop_rate": round(rng.uniform(0.2, 0.5), 2),
                "period": rng.choice([2, 4, 8, 12, 20, 40])}
    if fault == "payment_leak":
        return {"leak_every": rng.choice([2, 3, 4, 6, 10, 15])}
    if fault == "data_corruption":
        return {"corruption_rate": round(rng.uniform(0.02, 0.2), 3)}
    if fault == "workflow_deadlock":
        return {}
    if fault == "queue_overflow":
        return {"traffic_multiplier": round(rng.uniform(ratio * 1.1, 4.5), 2)}
    if fault == "cascade":
        return {"traffic_multiplier": round(rng.uniform(max(3.0, ratio * 1.8), 5.0), 2)}
    if fault == "compound":
        return {"drop_rate": round(rng.uniform(0.08, 0.4), 2),
                "traffic_multiplier": round(rng.uniform(ratio * 1.3, 4.0), 2)}
    raise ValueError(fault)


def _pick_faults(rng: random.Random, k: int) -> List[str]:
    pool = ["silent_dropper", "flapping", "payment_leak", "data_corruption",
            "workflow_deadlock", "queue_overflow", "cascade", "compound"]
    rng.shuffle(pool)
    chosen: List[str] = []
    for f in pool:
        if len(chosen) == k:
            break
        if f in _PAY_LOSS and any(c in _PAY_LOSS for c in chosen):
            continue
        if f in _OVERLOAD and any(c in _OVERLOAD for c in chosen):
            continue
        chosen.append(f)
    return chosen


def _noise_kwargs(rng: random.Random, max_noise: float = 0.06) -> dict:
    kw = {}
    if rng.random() < 0.45:
        kw["sensor_noise"] = round(rng.uniform(0.01, max_noise), 3)
    if rng.random() < 0.35:
        kw["metrics_dropout"] = round(rng.uniform(0.1, 0.6), 2)
    if rng.random() < 0.35:
        kw["log_dropout"] = round(rng.uniform(0.2, 0.95), 2)
    return kw


def _random_quiet(rng: random.Random, name: str, profile: str, horizon: int) -> dict:
    rate, thr = _PROFILE_RATE[profile]
    cap = _PROFILE_CAP[profile]
    events: List[Event] = []
    style = rng.choice(["plain", "high_load", "burst", "lull", "diurnal"])
    if style == "high_load":
        events.append(ev(rng.randint(10, 80), "env:set_traffic",
                         multiplier=round(rng.uniform(1.2, 0.97 * thr / rate), 2)))
    elif style == "burst":
        m = round(rng.uniform(2.0, 3.5), 2)
        excess = max(1.0, m * rate - thr)
        d = max(2, min(8, int(0.55 * cap / excess)))
        t0 = rng.randint(30, 150)
        events += [ev(t0, "queue_overflow", traffic_multiplier=m),
                   ev(t0 + d, "env:set_traffic", multiplier=1.0)]
    elif style == "lull":
        t0 = rng.randint(30, 100)
        events += [ev(t0, "env:set_traffic", multiplier=round(rng.uniform(0.03, 0.3), 2)),
                   ev(t0 + rng.randint(40, 100), "env:set_traffic", multiplier=1.0)]
    elif style == "diurnal":
        hi = round(0.97 * thr / rate, 2)
        step = rng.choice([20, 30, 50])
        events += [ev(t, "env:set_traffic", multiplier=(0.4 if (t // step) % 2 == 0 else hi))
                   for t in range(0, horizon - 10, step)]
    return _case(name, "quiet", f"random quiet ({style}): harmless oddity, must not alarm.",
                 events, horizon, profile=profile, seed=rng.randint(0, 9999),
                 tags=["random", "quiet", style], **_noise_kwargs(rng, 0.06))


def generate_random_cases(seed: int, n: int) -> List[dict]:
    """n varied scenarios from a seeded generator (deterministic per seed).
    ~15% quiet, ~25% unfixable, ~60% fixable, built so the verdict is true by
    construction. The mix includes mild onsets, recurrences, overlaps,
    irrelevant tool outages, outages that start at the moment of failure,
    and degraded sensors. Run validate() to prove feasibility."""
    rng = random.Random(seed)
    out: List[dict] = []
    for i in range(n):
        profile = rng.choices(["default", "large", "small"], [0.5, 0.25, 0.25])[0]
        horizon = rng.randint(260, 420)
        r = rng.random()
        base = f"rnd{seed}_{i:03d}"
        if r < 0.15:
            out.append(_random_quiet(rng, base + "_quiet", profile, horizon))
            continue
        kind = "unfixable" if r < 0.40 else "fixable"
        faults = _pick_faults(rng, rng.choice([1, 1, 2, 2, 3]))
        last_start = max(10, horizon - 120)
        t_first = rng.randint(0, min(130, last_start))
        events: List[Event] = []
        first_tick: Dict[str, int] = {}
        for f in faults:
            t = min(last_start, t_first + rng.choice([0, 0, rng.randint(5, 60)]))
            events.append((t, f, _rand_params(rng, f, profile)))
            first_tick[f] = t
        kw = _noise_kwargs(rng)
        disabled: List[str] = []
        late_disable: List[Event] = []

        if kind == "unfixable":
            victim = rng.choice(faults)
            group = rng.choice(_NEEDS[victim])          # remove every way to fix this one
            # An overload victim must be disabled from tick 0: if the tools died
            # later, an earlier fix for another fault could already have raised
            # capacity, leaving nothing to repair (found by the oracle).
            if rng.random() < 0.5 or S in group:
                disabled += group
            else:
                late_disable += [(first_tick[victim], "env:disable_action", {"action": a})
                                 for a in group]
        else:
            needed = {a for f in faults for g in _NEEDS[f] for a in g}
            free = [a for a in ACTIONS if a not in needed]
            if any(len(g) == 2 for f in faults for g in _NEEDS[f]):
                free.append(rng.choice([S, T]))          # one of an any-of pair
            rng.shuffle(free)
            disabled += free[:rng.choice([0, 0, 1, 2])]
            f0 = faults[0]                               # maybe a repeat later
            t2 = first_tick[f0] + rng.randint(70, 110)
            if rng.random() < 0.3 and t2 <= last_start:
                events.append((t2, f0, _rand_params(rng, f0, profile)))
        if disabled:
            kw["disabled_actions"] = sorted(set(disabled))
        tags = ["random", kind] + faults
        why = (f"random {kind}: " + " + ".join(faults)
               + (" (tool outage)" if disabled or late_disable else "")
               + (" with degraded sensors" if any(k in kw for k in
                  ("sensor_noise", "metrics_dropout", "log_dropout")) else ""))
        out.append(_case(base + "_" + kind, kind, why, events + late_disable, horizon,
                         profile=profile, seed=rng.randint(0, 9999), tags=tags, **kw))
    return out


# ----------------------------------------------------------------------------
# Feasibility oracle (omniscient; proves the cases are fair)
# ----------------------------------------------------------------------------
def oracle_check(case: dict, delay: int = 3, period: int = 5) -> dict:
    """An omniscient solver. It sees ground truth and, `delay` ticks after any
    root cause appears, fires EVERY working recovery action at full strength
    every `period` ticks. Verifies that the declared kind is really true."""
    shop = build_shop(case)
    since: Optional[int] = None
    min_health = 100.0
    for _ in range(case["horizon"]):
        apply_due_events(shop, case)
        if case["kind"] != "quiet" and root_causes(shop):
            since = shop.tick if since is None else since
            k = shop.tick - since
            if k >= delay and (k - delay) % period == 0:
                for a in ACTIONS:
                    apply_action(shop, a, 1.0)
        else:
            since = None
        run_shop(shop, 1)
        min_health = min(min_health, _truth_health(shop))

    kind = case["kind"]
    rc = root_causes(shop)
    if kind == "unfixable":
        ok = bool(rc)
        return {"ok": ok, "reason": "root cause correctly irreparable" if ok else
                "ORACLE REMOVED the root cause: case is mislabeled as unfixable"}
    if kind == "quiet":
        q, p = shop.queue, shop.payments
        probs = []
        if rc:
            probs.append(f"root cause present at end: {rc}")
        if min_health < 85:
            probs.append(f"health dipped to {min_health:.1f}")
        if q.messages_dropped or p.lost_total or damage(shop):
            probs.append(f"damage: dropped={q.messages_dropped} lost={p.lost_total} "
                         f"corrupt={damage(shop)}")
        return {"ok": not probs, "reason": "; ".join(probs) or "harmless with no action"}
    v = judge(case, shop, paged=False, incident_raised=True)
    return {"ok": v["ok"], "reason": v["reason"] if not v["ok"] else
            "oracle recovers it: " + v["reason"]}


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate_structure(cases: List[dict]) -> List[str]:
    errs, seen = [], set()
    for c in cases:
        n = c.get("name", "?")
        for k in ("name", "kind", "shop_kwargs", "events", "horizon"):
            if k not in c:
                errs.append(f"{n}: missing {k}")
        if n in seen:
            errs.append(f"{n}: duplicate name")
        seen.add(n)
        if c["kind"] not in ("fixable", "unfixable", "quiet"):
            errs.append(f"{n}: bad kind")
        try:
            shop = build_shop(c)
            for t, name, params in c["events"]:
                if t < 0 or t >= c["horizon"]:
                    errs.append(f"{n}: event tick {t} outside horizon")
                apply_event(shop.clone(), name, params)
        except Exception as e:                       # noqa: BLE001
            errs.append(f"{n}: {type(e).__name__}: {e}")
    return errs


def validate(cases: List[dict], verbose: bool = True) -> bool:
    errs = validate_structure(cases)
    bad = list(errs)
    for c in cases:
        res = oracle_check(c)
        if verbose:
            print(f"{'ok ' if res['ok'] else 'BAD'} {c['kind']:<9} {c['name']:<42} {res['reason']}")
        if not res["ok"]:
            bad.append(f"{c['name']}: {res['reason']}")
    for e in errs:
        print("STRUCTURE ERROR:", e)
    return not bad


if __name__ == "__main__":
    n_rand = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    assert len(CASES) >= 25
    kinds = {k: sum(c["kind"] == k for c in CASES) for k in ("fixable", "unfixable", "quiet")}
    print(f"{len(CASES)} hand-written cases {kinds}; validating with the oracle...\n")
    ok1 = validate(CASES)
    rnd = generate_random_cases(12345, n_rand)
    rk = {k: sum(c["kind"] == k for c in rnd) for k in ("fixable", "unfixable", "quiet")}
    print(f"\n{len(rnd)} random cases {rk}; validating...\n")
    ok2 = validate(rnd, verbose=False)
    print("\nALL CASES FEASIBLE AND CORRECTLY LABELED" if ok1 and ok2
          else "\nSOME CASES ARE MISLABELED OR BROKEN (see above)")
    sys.exit(0 if ok1 and ok2 else 1)
