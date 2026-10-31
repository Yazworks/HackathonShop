"""
evaluate.py - the evidence: how good is the recovery agent, measured honestly?

What this file does, in plain words
-----------------------------------
It builds a set of test situations the agent was NEVER tuned on, runs several
agents on every one of them, and measures the same things for each:

  * did the shop get lastingly healthy again, with the root cause really gone?
  * how many ticks were lost to an unhealthy shop (the "customer pain" number)?
  * how long until lasting recovery, and how many actions were used?
  * did the agent page a human when (and only when) it should?
  * did it raise false alarms on a healthy shop?

AGENTS COMPARED
  do_nothing       never acts                                    (floor)
  rule_based       hand-written if/then rules                    (what an engineer would script)
  evolution_only   evolves fixes, but no diagnosis, memory or dry run
  no_detector      full system minus the diagnosis model
  no_dry_run       full system minus the staging-copy rehearsal
  full_system      diagnosis + memory + evolution + dry run + verification

TEST SITUATIONS (all with harsher settings than anything used for tuning,
and shops built from seeds nobody tuned on)
  single_*        one fault, harsher than in training
  combo_*         two faults at once, including pairs never trained on
  second_fault_*  a second, different fault appears while the first is being fixed
  novel_*         a fault type the system has no label for (slow payments)
  unfixable_*     a root cause that keeps coming back: the RIGHT answer is to
                  try, then page a human
  quiet           a healthy shop for 200 ticks: the right answer is to do nothing

A second experiment measures LEARNING: the same kind of fault three times in a
row, with and without the playbook memory.

Run:
    python evaluate.py quick     1 seed per situation   (a few minutes)
    python evaluate.py full      3 seeds per situation  (the numbers to report)

Needs:  python detector.py   (once, to train the diagnosis model)
Writes: results/evaluation.json, results/evaluation.md, results/*.png
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

from detector import load_detector
from evolution import (
    HEALTHY, AgentContext, DoNothingAgent, RuleBasedAgent, _persistent_part,
    _reapply, symptom_severity,
)
from playbooks import STARTER_PATH, PlaybookLibrary
from recovery_agent import RecoveryAgent
from shop import FAULTS, Shop, apply_action, collect_metrics, make_shop, run_shop

RESULTS_DIR = Path("results")
WARMUP = 60                 # healthy ticks before anything breaks
HORIZON = 260               # ticks the agent gets after the first fault
QUIET_HORIZON = 200
FINAL_HEALTHY = 10          # "lastingly healthy" = healthy for the last 10 ticks

ARMS = ["do_nothing", "rule_based", "random_search", "evolution_only", "no_dry_run", "full_system",
        "full_plus", "reasoning_only", "reasoning_hybrid", "reasoning_plus"]


# ----------------------------------------------------------------------------
# Test situations
# ----------------------------------------------------------------------------
def _slow_payments(shop: Shop, throughput: int = 3) -> None:
    """A fault type the system was never built for."""
    shop.payments.throughput = throughput


FAULT_FUNCS: Dict[str, Callable] = dict(FAULTS)
FAULT_FUNCS["payment_slowdown"] = _slow_payments


def _tool_outage(shop: Shop, action: str) -> None:
    """A recovery tool stops working (staging copies mirror production, so
    they have the same outage)."""
    shop.disabled_actions.add(action)


FAULT_FUNCS["tool_outage"] = _tool_outage

# every setting here is harsher than TRAIN_PARAMS in evolution.py
DROP, TRAFFIC, CORRUPT = {"drop_rate": 0.35}, {"traffic_multiplier": 4.0}, {"corruption_rate": 0.12}

NOISY = {"sensor_noise": 0.03, "metrics_dropout": 0.10, "log_dropout": 0.30}
VERY_NOISY = {"sensor_noise": 0.05, "metrics_dropout": 0.20, "log_dropout": 0.50}

CASES: List[Dict] = [
    {"name": "single_silent_dropper", "group": "single", "events": [(0, "silent_dropper", DROP)]},
    {"name": "single_queue_overflow", "group": "single", "events": [(0, "queue_overflow", TRAFFIC)]},
    {"name": "single_data_corruption", "group": "single", "events": [(0, "data_corruption", CORRUPT)]},
    {"name": "single_workflow_deadlock", "group": "single", "events": [(0, "workflow_deadlock", {})]},
    {"name": "combo_dropper_overflow", "group": "combo",
     "events": [(0, "compound", {"drop_rate": 0.25, "traffic_multiplier": 3.5})]},
    {"name": "combo_corruption_deadlock", "group": "combo",
     "events": [(0, "data_corruption", CORRUPT), (0, "workflow_deadlock", {})]},
    {"name": "combo_dropper_corruption", "group": "combo",
     "events": [(0, "silent_dropper", DROP), (0, "data_corruption", CORRUPT)]},
    {"name": "combo_overflow_deadlock", "group": "combo",
     "events": [(0, "queue_overflow", TRAFFIC), (0, "workflow_deadlock", {})]},
    {"name": "second_fault_dropper_then_deadlock", "group": "second_fault",
     "events": [(0, "silent_dropper", DROP), (25, "workflow_deadlock", {})]},
    {"name": "second_fault_overflow_then_corruption", "group": "second_fault",
     "events": [(0, "queue_overflow", TRAFFIC), (25, "data_corruption", CORRUPT)]},
    {"name": "novel_payment_slowdown", "group": "novel",
     "events": [(0, "payment_slowdown", {"throughput": 3})]},
    {"name": "adapt_overflow_scale_queue_down", "group": "adaptation",
     "events": [(0, "tool_outage", {"action": "scale_queue"}), (0, "queue_overflow", TRAFFIC)]},
    {"name": "adapt_combo_scale_queue_down", "group": "adaptation",
     "events": [(0, "tool_outage", {"action": "scale_queue"}),
                (0, "silent_dropper", DROP), (0, "queue_overflow", TRAFFIC)]},
    {"name": "unfixable_all_queue_tools_down", "group": "unfixable", "fixable": False,
     "events": [(0, "tool_outage", {"action": "scale_queue"}),
                (0, "tool_outage", {"action": "throttle_traffic"}),
                (0, "queue_overflow", TRAFFIC)]},
    # ---- robustness: imperfect sensors (readings noisy, sometimes stale, logs lost)
    {"name": "noisy_single_dropper", "group": "noisy_sensors", "noise": NOISY,
     "events": [(0, "silent_dropper", DROP)]},
    {"name": "noisy_single_overflow", "group": "noisy_sensors", "noise": NOISY,
     "events": [(0, "queue_overflow", TRAFFIC)]},
    {"name": "noisy_combo_corruption_deadlock", "group": "noisy_sensors", "noise": NOISY,
     "events": [(0, "data_corruption", CORRUPT), (0, "workflow_deadlock", {})]},
    # ---- robustness: a shop of a different size than anything was tuned on
    {"name": "bigshop_dropper", "group": "untuned_shop", "profile": "large",
     "events": [(0, "silent_dropper", DROP)]},
    {"name": "bigshop_overflow", "group": "untuned_shop", "profile": "large",
     "events": [(0, "queue_overflow", TRAFFIC)]},
    {"name": "smallshop_combo", "group": "untuned_shop", "profile": "small",
     "events": [(0, "data_corruption", CORRUPT), (0, "workflow_deadlock", {})]},
    # ---- robustness: trickier kinds of failure
    {"name": "newfault_payment_leak", "group": "new_faults",
     "events": [(0, "payment_leak", {})]},
    {"name": "newfault_flapping", "group": "new_faults",
     "events": [(0, "flapping", {"drop_rate": 0.30, "period": 12})]},
    {"name": "newfault_cascade", "group": "new_faults",
     "events": [(0, "cascade", {"traffic_multiplier": 3.5})]},
    # ---- everything at once
    {"name": "hardest_noisy_bigshop_cascade", "group": "new_faults", "profile": "large",
     "noise": NOISY, "events": [(0, "cascade", {"traffic_multiplier": 4.0})]},
    {"name": "unfixable_dropper_returns", "group": "unfixable", "fixable": False,
     "events": [(0, "silent_dropper", {"drop_rate": 0.30})], "persist": {"drop_rate": 0.30}},
    {"name": "quiet_healthy_shop", "group": "quiet", "events": [], "horizon": QUIET_HORIZON},
    {"name": "quiet_noisy_sensors", "group": "quiet", "events": [], "horizon": QUIET_HORIZON,
     "noise": NOISY},
    {"name": "quiet_very_noisy_sensors", "group": "quiet", "events": [], "horizon": QUIET_HORIZON,
     "noise": VERY_NOISY},
]


# ----------------------------------------------------------------------------
# A plain policy (do-nothing / rule-based) run live, with the same interface
# as RecoveryAgent so one harness can drive every agent
# ----------------------------------------------------------------------------
class PolicyRunner:
    def __init__(self, shop: Shop, policy, before_tick: Callable[[Shop], None]):
        self.shop, self.policy, self.before_tick = shop, policy, before_tick
        self.health_trace: List[float] = []
        self.severity: List[float] = []
        self.ctx = AgentContext(health_history=self.health_trace, severity_history=self.severity)
        self.paged_tick: Optional[int] = None

    def tick(self) -> None:
        self.before_tick(self.shop)
        run_shop(self.shop, 1)
        m = collect_metrics(self.shop)
        self.health_trace.append(m.health_score)
        self.severity.append(symptom_severity(m))
        if self.paged_tick is not None:
            return                                   # a human is on it; the agent stops
        self.ctx.t = self.shop.tick
        d = self.policy.decide(m, self.ctx)
        if d.kind == "act":
            apply_action(self.shop, d.action, d.strength)
            c = self.ctx
            c.actions_taken += 1
            c.last_action, c.last_action_tick = d.action, c.t
            c.severity_at_action = self.severity[-1]
        elif d.kind == "escalate":
            self.paged_tick = self.shop.tick


def make_runner(arm: str, shop: Shop, det, hook: Callable, seed: int):
    if arm == "do_nothing":
        return PolicyRunner(shop, DoNothingAgent(), hook)
    if arm == "rule_based":
        return PolicyRunner(shop, RuleBasedAgent(), hook)
    # memory starts EMPTY, except for the "starter" arms which begin with the
    # playbooks trained offline by make_starter.py (mild settings, other seeds)
    starter = STARTER_PATH if arm in ("full_plus", "reasoning_plus") else None
    lib = PlaybookLibrary(None, starter_path=starter)
    kw = dict(seed=seed, before_tick=hook)
    if arm == "random_search":                 # same budget as evolution, but NO fitness-guided breeding
        return RecoveryAgent(shop, None, lib, use_memory=False, use_dry_run=False,
                             evolution_kwargs=dict(population_size=400, max_generations=1,
                                                   min_generations=1, early_stop=1e9), **kw)
    if arm == "evolution_only":
        return RecoveryAgent(shop, None, lib, use_memory=False, use_dry_run=False, **kw)
    if arm == "reasoning_only":                # knowledge + reasoning, no search, no memory
        return RecoveryAgent(shop, det, lib, use_reasoning=True, use_evolution=False,
                             use_memory=False, **kw)
    if arm == "reasoning_hybrid":              # reasoning first, evolution as the fallback
        return RecoveryAgent(shop, det, lib, use_reasoning=True, **kw)
    if arm == "reasoning_plus":                # ... and starter playbooks as well
        return RecoveryAgent(shop, det, lib, use_reasoning=True, **kw)
    if arm == "no_dry_run":
        return RecoveryAgent(shop, det, lib, use_dry_run=False, **kw)
    if arm == "full_system":
        return RecoveryAgent(shop, det, lib, **kw)
    if arm == "full_plus":                     # starter playbooks + act immediately
        return RecoveryAgent(shop, det, lib, act_immediately=True, **kw)
    raise ValueError(arm)


# ----------------------------------------------------------------------------
# One episode = one agent facing one situation
# ----------------------------------------------------------------------------
_DET = None


def _init_worker() -> None:
    global _DET
    _DET = load_detector()


def run_episode(task) -> Dict:
    kind = task[0]
    if kind == "learn":
        return run_learning(task)
    _, case_idx, seed, arm = task
    case = CASES[case_idx]
    horizon = case.get("horizon", HORIZON)
    shop = make_shop(case.get("profile", "default"), seed=9000 + 101 * seed + case_idx)
    for k, v in case.get("noise", {}).items():
        setattr(shop, k, v)
    state = {"t0": None, "done": set()}
    truth: Dict[int, float] = {}        # exact health after each tick (never shown to agents)
    persist = case.get("persist")

    def hook(s: Shop) -> None:
        if s.tick > 0:
            truth[s.tick] = collect_metrics(s, observe=False).health_score
        if state["t0"] is None:
            return
        off = s.tick - state["t0"]
        for i, (when, fault, kw) in enumerate(case["events"]):
            if off >= when and i not in state["done"]:
                FAULT_FUNCS[fault](s, **kw)
                state["done"].add(i)
        if persist:
            _reapply(s, persist)

    runner = make_runner(arm, shop, _DET, hook, seed + 1)
    t_start = time.perf_counter()
    while shop.tick < WARMUP:
        runner.tick()
    state["t0"] = shop.tick
    while shop.tick < state["t0"] + horizon:
        runner.tick()
    truth[shop.tick] = collect_metrics(shop, observe=False).health_score
    post = [truth[k] for k in range(state["t0"] + 1, state["t0"] + horizon + 1)]

    unhealthy = sum(1 for h in post if h < HEALTHY)
    final_ok = all(h >= HEALTHY for h in post[-FINAL_HEALTHY:])
    ttr = None
    if final_ok:                                  # start of the final healthy run
        i = len(post)
        while i > 0 and post[i - 1] >= HEALTHY:
            i -= 1
        ttr = i
    residual = bool(_persistent_part(shop)) and case.get("fixable", True)

    if isinstance(runner, RecoveryAgent):
        incs = runner.incidents
        actions = sum(len(i["actions"]) for i in incs)
        paged = any(i["outcome"] == "paged" for i in incs)
        page_ends = [i["end_tick"] for i in incs if i["outcome"] == "paged" and i["end_tick"]]
        ticks_to_page = (min(page_ends) - state["t0"]) if page_ends else None
        false_alarms = sum(1 for i in incs if i["start_tick"] < state["t0"])
        thinking = sum(i["planning_ticks"] for i in incs)
    else:
        actions = runner.ctx.actions_taken
        paged = runner.paged_tick is not None
        ticks_to_page = (runner.paged_tick - state["t0"]) if paged else None
        false_alarms, thinking = 0, 0

    if case["group"] == "quiet":
        success = (not incs_any(runner)) and actions == 0 and unhealthy == 0
    elif not case.get("fixable", True):
        success = paged                           # handed over to a human
    else:
        success = final_ok and not residual and not paged

    return {"case": case["name"], "group": case["group"], "seed": seed, "arm": arm,
            "success": bool(success), "unhealthy_ticks": unhealthy, "ttr": ttr,
            "actions": actions, "paged": paged, "false_alarms": false_alarms,
            "thinking_ticks": thinking, "ticks_to_page": ticks_to_page,
            "sources": ([i["source"] for i in incs] if isinstance(runner, RecoveryAgent) else []), "residual_root_cause": residual,
            "final_healthy": final_ok, "horizon": horizon,
            "seconds": round(time.perf_counter() - t_start, 1)}


def incs_any(runner) -> bool:
    return isinstance(runner, RecoveryAgent) and len(runner.incidents) > 0


# ----------------------------------------------------------------------------
# Learning experiment: the same kind of fault three times in a row
# ----------------------------------------------------------------------------
LEARN_PLANS = {
    "silent_dropper": [{"drop_rate": 0.20}, {"drop_rate": 0.30}, {"drop_rate": 0.35}],
    "queue_overflow": [{"traffic_multiplier": 3.0}, {"traffic_multiplier": 3.5},
                       {"traffic_multiplier": 4.0}],
}


def run_learning(task) -> Dict:
    _, fault, use_memory = task
    shop = Shop(seed=777)
    agent = RecoveryAgent(shop, _DET, PlaybookLibrary(None), use_memory=use_memory, seed=3)
    agent.run(WARMUP)
    for params in LEARN_PLANS[fault]:
        FAULTS[fault](shop, **params)
        agent.run_until_idle()
        agent.run(40)
    rows = [{"occurrence": n + 1, "source": i["source"], "thinking_ticks": i["planning_ticks"],
             "actions": len(i["actions"]), "outcome": i["outcome"],
             "time_to_recover": i["time_to_recover"]}
            for n, i in enumerate(agent.incidents)]
    return {"kind": "learn", "fault": fault, "memory": use_memory, "rows": rows}


# ----------------------------------------------------------------------------
# Aggregation and reporting
# ----------------------------------------------------------------------------
def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(x: Optional[float], nd: int = 1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def aggregate(rows: List[Dict]) -> Dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["case"])].append(r)
    per_case = {}
    for (arm, case), rs in by.items():
        per_case.setdefault(case, {})[arm] = {
            "n": len(rs),
            "success_rate": sum(r["success"] for r in rs) / len(rs),
            "mean_unhealthy_ticks": _mean([r["unhealthy_ticks"] for r in rs]),
            "mean_ttr_successful": _mean([r["ttr"] for r in rs if r["success"]]),
            "mean_actions": _mean([r["actions"] for r in rs]),
            "paged_rate": sum(r["paged"] for r in rs) / len(rs),
            "mean_ticks_to_page": _mean([r["ticks_to_page"] for r in rs]),
            "false_alarms": sum(r["false_alarms"] for r in rs),
        }
    overall = {}
    fix = [r for r in rows if r["group"] not in ("unfixable", "quiet")]
    for arm in ARMS:
        a = [r for r in fix if r["arm"] == arm]
        if not a:
            continue
        overall[arm] = {
            "n": len(a),
            "success_rate": sum(r["success"] for r in a) / len(a),
            "mean_unhealthy_ticks": _mean([r["unhealthy_ticks"] for r in a]),
            "mean_ttr_successful": _mean([r["ttr"] for r in a if r["success"]]),
            "mean_actions": _mean([r["actions"] for r in a]),
        }
    groups = {}
    for g in sorted({r["group"] for r in rows}):
        for arm in ARMS:
            a = [r for r in rows if r["arm"] == arm and r["group"] == g]
            if a:
                groups.setdefault(g, {})[arm] = {
                    "success_rate": sum(r["success"] for r in a) / len(a),
                    "mean_unhealthy_ticks": _mean([r["unhealthy_ticks"] for r in a]),
                    "n": len(a)}
    return {"per_case": per_case, "overall_fixable": overall, "by_group": groups}


def build_markdown(agg: Dict, learn: List[Dict], seeds: int) -> str:
    L: List[str] = []
    L.append("# HackathonShop - evaluation results\n")
    L.append(f"All situations are held out: harsher than tuning settings, shops built from "
             f"fresh seeds, **{seeds} seed(s) per situation**, every agent starts with an "
             f"empty memory. Results come from the simulated shop.\n")
    L.append("## 1. Headline: situations where a fix exists\n")
    L.append("Success = shop lastingly healthy at the end, root cause really gone, human not paged.\n")
    L.append("| Agent | Episodes | Success rate | Mean unhealthy ticks (lower is better) | Mean ticks to lasting recovery | Mean actions |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        o = agg["overall_fixable"].get(arm)
        if o:
            L.append(f"| {arm} | {o['n']} | {o['success_rate'] * 100:.0f}% | "
                     f"{_fmt(o['mean_unhealthy_ticks'])} | {_fmt(o['mean_ttr_successful'])} | "
                     f"{_fmt(o['mean_actions'])} |")
    L.append("\n## 2. By kind of situation (success rate / mean unhealthy ticks)\n")
    L.append("| Situation type | " + " | ".join(ARMS) + " |")
    L.append("|---|" + "---:|" * len(ARMS))
    for g, d in agg["by_group"].items():
        cells = [f"{d[a]['success_rate'] * 100:.0f}% / {_fmt(d[a]['mean_unhealthy_ticks'], 0)}"
                 if a in d else "-" for a in ARMS]
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    L.append("\n## 3. Every situation (success rate / mean unhealthy ticks)\n")
    L.append("| Situation | " + " | ".join(ARMS) + " |")
    L.append("|---|" + "---:|" * len(ARMS))
    for case, d in agg["per_case"].items():
        cells = [f"{d[a]['success_rate'] * 100:.0f}% / {_fmt(d[a]['mean_unhealthy_ticks'], 0)}"
                 if a in d else "-" for a in ARMS]
        L.append(f"| {case} | " + " | ".join(cells) + " |")
    L.append("\nFor `unfixable_*`, success means the agent paged a human (nothing it can do "
             "fixes the root cause). For `quiet_*`, success means no incident was raised.\n")
    L.append("## 4. When nothing can fix it: does the agent hand over to a human, and how fast?\n")
    L.append("Cells: paged? / mean ticks until the page / mean fixes tried first.\n")
    L.append("| Situation | " + " | ".join(ARMS) + " |")
    L.append("|---|" + "---:|" * len(ARMS))
    for case, d in agg["per_case"].items():
        if not case.startswith("unfixable"):
            continue
        cells = [f"{'yes' if d[a]['paged_rate'] >= 0.5 else 'NO'} / {_fmt(d[a]['mean_ticks_to_page'], 0)} / "
                 f"{_fmt(d[a]['mean_actions'])}" if a in d else "-" for a in ARMS]
        L.append(f"| {case} | " + " | ".join(cells) + " |")
    L.append("")
    if learn:
        L.append("## 5. Learning: the same kind of fault three times in a row\n")
        L.append("| Fault | Memory | Occurrence | Fix came from | Thinking ticks | Actions | Time to recover |")
        L.append("|---|---|---:|---|---:|---:|---:|")
        for run in sorted(learn, key=lambda r: (r["fault"], not r["memory"])):
            for r in run["rows"]:
                L.append(f"| {run['fault']} | {'on' if run['memory'] else 'off'} | {r['occurrence']} | "
                         f"{r['source']} | {r['thinking_ticks']} | {r['actions']} | "
                         f"{r['time_to_recover'] if r['time_to_recover'] is not None else '-'} |")
    L.append("\n## How to read this honestly\n")
    L.append("- The shop is a simulation; real systems are messier.")
    L.append("- Faults are clean and have known fixes in the action menu; the novel-fault row "
             "shows what happens when the fault is outside the diagnosis model's labels.")
    L.append("- Evolution uses randomness; seeds are fixed, so re-running reproduces these numbers.")
    return "\n".join(L) + "\n"


def make_charts(agg: Dict, learn: List[Dict]) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"do_nothing": "#9aa0a6", "rule_based": "#e8a33d", "evolution_only": "#7aa6d6", "random_search": "#b0b7c3",
              "no_detector": "#b08fd1", "no_dry_run": "#5cb89a", "full_system": "#1f5fbf",
              "full_plus": "#0b3d91", "reasoning_only": "#d9534f",
              "reasoning_hybrid": "#c9302c", "reasoning_plus": "#8e1b17"}
    out = []
    # chart 1: success rate and unhealthy ticks by agent
    arms = [a for a in ARMS if a in agg["overall_fixable"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(arms, [agg["overall_fixable"][a]["success_rate"] * 100 for a in arms],
              color=[colors[a] for a in arms])
    ax[0].set_ylabel("success rate (%)")
    ax[0].set_title("Held-out situations with a fix: success")
    ax[1].bar(arms, [agg["overall_fixable"][a]["mean_unhealthy_ticks"] for a in arms],
              color=[colors[a] for a in arms])
    ax[1].set_ylabel("unhealthy ticks (lower is better)")
    ax[1].set_title("Time the shop spent unhealthy")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = RESULTS_DIR / "chart_overall.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append(str(p))
    # chart 2: by situation type
    groups = [g for g in agg["by_group"] if g not in ("quiet", "unfixable")]
    # six key agents only: nine bars per group is unreadable on a slide
    key = [a for a in ("rule_based", "evolution_only", "full_system", "reasoning_only",
                       "reasoning_hybrid") if a in arms] or arms
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.8 / len(key)
    for k, a in enumerate(key):
        ax.bar([i + k * w for i in range(len(groups))],
               [agg["by_group"][g].get(a, {}).get("success_rate", 0) * 100 for g in groups],
               w, label=a, color=colors[a])
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(groups))])
    ax.set_xticklabels(groups)
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Success by kind of situation")
    ax.legend(fontsize=9, ncol=len(key), loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = RESULTS_DIR / "chart_by_group.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append(str(p))
    # chart 3: learning curve
    if learn:
        faults = sorted({r["fault"] for r in learn})
        fig, ax = plt.subplots(1, len(faults), figsize=(5.5 * len(faults), 4), squeeze=False)
        for j, f in enumerate(faults):
            for run in [r for r in learn if r["fault"] == f]:
                xs = [r["occurrence"] for r in run["rows"]]
                ys = [r["thinking_ticks"] for r in run["rows"]]
                ax[0][j].plot(xs, ys, marker="o",
                              label="with memory" if run["memory"] else "no memory",
                              color="#1f5fbf" if run["memory"] else "#9aa0a6")
            ax[0][j].set_title(f"{f}: thinking time per occurrence")
            ax[0][j].set_xlabel("occurrence")
            ax[0][j].set_ylabel("ticks spent planning")
            ax[0][j].set_xticks([1, 2, 3])
            ax[0][j].legend()
            ax[0][j].spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = RESULTS_DIR / "chart_learning.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        out.append(str(p))
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(mode: str, arms_filter: Optional[List[str]] = None,
         cases_filter: Optional[List[str]] = None, seeds_override: Optional[int] = None) -> None:
    try:
        load_detector()
    except FileNotFoundError:
        print("Detector model not found. Run  python detector.py  first.")
        sys.exit(1)
    seeds = seeds_override or (1 if mode == "quick" else 3)
    RESULTS_DIR.mkdir(exist_ok=True)
    use_arms = [a for a in ARMS if not arms_filter or a in arms_filter]
    use_cases = [ci for ci, c in enumerate(CASES)
                 if not cases_filter or any(s in c["name"] or s == c["group"] for s in cases_filter)]
    tasks = [("ep", ci, s, arm) for s in range(seeds) for ci in use_cases for arm in use_arms]
    if not arms_filter and not cases_filter:
        tasks += [("learn", f, mem) for f in LEARN_PLANS for mem in (True, False)]
    # slow tasks first so the pool stays busy
    tasks.sort(key=lambda t: 1 if t[-1] in ("do_nothing", "rule_based") else 0)
    workers = max(1, min(os.cpu_count() or 1, 6))
    print(f"Running {len(tasks)} episodes ({seeds} seed(s)) on {workers} worker(s). "
          f"This takes a while; progress below.\n")
    t0 = time.perf_counter()
    rows, learn = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for n, res in enumerate(pool.map(run_episode, tasks, chunksize=1), 1):
            (learn if res.get("kind") == "learn" else rows).append(res)
            if n % 10 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)} done  ({time.perf_counter() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r["case"], r["seed"], ARMS.index(r["arm"])))
    agg = aggregate(rows)
    md = build_markdown(agg, learn, seeds)
    (RESULTS_DIR / "evaluation.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / "evaluation.json").write_text(
        json.dumps({"mode": mode, "seeds": seeds, "aggregate": agg, "learning": learn,
                    "episodes": rows}, indent=2), encoding="utf-8")
    charts = make_charts(agg, learn)
    print("\n" + md)
    print("Saved results/evaluation.md, results/evaluation.json and:", ", ".join(charts))
    print(f"Total time {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    # python evaluate.py [quick|full] [arms=a,b] [cases=x,y]
    opts = {a.split("=")[0]: a.split("=")[1].split(",") for a in sys.argv[2:] if "=" in a}
    main(sys.argv[1] if len(sys.argv) > 1 else "quick", opts.get("arms"), opts.get("cases"),
         int(opts["seeds"][0]) if "seeds" in opts else None)
