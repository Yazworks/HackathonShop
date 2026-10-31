"""
run_independent.py - run every agent on the INDEPENDENT test cases.

What this file does, in plain words
-----------------------------------
independent_cases.py was written by a separate Claude session that never saw
our agent, our knowledge base or our own tests. It only knew the challenge text
and shop.py. This script is the thin bridge: it feeds those cases to our
agents, uses THEIR judge() to decide who passed, and writes a report.

Nothing in the agents is changed or tuned by this script.

Usage
-----
    python run_independent.py dev                  # 60 hand-written + 60 random (seed 12345)
    python run_independent.py final                # sealed set: 100 random cases, seed 20261031
    python run_independent.py dev arms=reasoning_hybrid,rule_based
    python run_independent.py dev limit=6          # only the first 6 cases (smoke test)
    python run_independent.py dev fresh=1          # ignore saved progress and start over

Progress is saved after every episode (results/independent_<set>_raw.jsonl), so
if the laptop sleeps you can run the same command again and it carries on.

PROTOCOL: run "dev" as often as you like while building. Run "final" ONCE,
after the agent is frozen, and report whatever comes out.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import independent_cases as IC
from detector import load_detector
from evaluate import make_runner
from recovery_agent import RecoveryAgent
from shop import collect_metrics, run_shop

DEFAULT_ARMS = ["do_nothing", "rule_based", "evolution_only", "full_system",
                "reasoning_only", "reasoning_hybrid", "reasoning_plus"]
FINAL_SEED, FINAL_N = 20261031, 100          # nobody looks at this set until the freeze
DEV_RANDOM_SEED, DEV_RANDOM_N = 12345, 60
HEALTHY = 80.0
RESULTS = Path("results")

_DET = None


def _init_worker() -> None:
    global _DET
    _DET = load_detector()


def cases_for(which: str) -> List[dict]:
    if which == "final":
        return IC.generate_random_cases(FINAL_SEED, FINAL_N)
    return list(IC.CASES) + IC.generate_random_cases(DEV_RANDOM_SEED, DEV_RANDOM_N)


# ----------------------------------------------------------------------------
# One episode = one agent facing one case
# ----------------------------------------------------------------------------
def run_one(task) -> Dict:
    case, arm = task
    t_start = time.perf_counter()
    shop = IC.build_shop(case)
    events = sorted(case["events"], key=lambda e: e[0])
    done = set()
    truth: Dict[int, float] = {}               # exact health, never shown to the agent

    def hook(s) -> None:                       # same job as iter_case(): apply due events
        if s.tick > 0:
            truth[s.tick] = collect_metrics(s, observe=False).health_score
        for i, (when, name, params) in enumerate(events):
            if when <= s.tick and i not in done:
                IC.apply_event(s, name, params)
                done.add(i)

    runner = make_runner(arm, shop, _DET, hook, 1)
    while shop.tick < case["horizon"]:
        runner.tick()
    truth[shop.tick] = collect_metrics(shop, observe=False).health_score

    if isinstance(runner, RecoveryAgent):
        incs = runner.incidents
        paged = any(i["outcome"] == "paged" for i in incs)
        raised = len(incs) > 0
        actions = sum(len(i["actions"]) for i in incs)
        page_ends = [i["end_tick"] for i in incs if i["outcome"] == "paged" and i["end_tick"]]
        page_tick = min(page_ends) if page_ends else None
        sources = [i["source"] for i in incs]
    else:
        paged = runner.paged_tick is not None
        actions = runner.ctx.actions_taken
        raised = paged or actions > 0          # a plain policy "raises" by acting or paging
        page_tick = runner.paged_tick
        sources = []

    verdict = IC.judge(case, shop, paged=paged, incident_raised=raised)
    fault_ticks = [e[0] for e in events if not e[1].startswith("env:")]
    t_fault = min(fault_ticks) if fault_ticks else 0
    unhealthy = sum(1 for k, h in truth.items() if k > t_fault and h < HEALTHY)
    return {"case": case["name"], "kind": case["kind"], "arm": arm,
            "ok": bool(verdict["ok"]), "reason": verdict["reason"],
            "ambiguous": bool(case.get("ambiguous")), "tags": case.get("tags", []),
            "paged": paged, "raised": raised, "actions": actions, "sources": sources,
            "unhealthy_ticks": unhealthy,
            "ticks_to_page": (page_tick - t_fault) if page_tick is not None else None,
            "seconds": round(time.perf_counter() - t_start, 1)}


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _f(x, nd=1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _wilson(k: int, n: int) -> str:
    """95% interval for a success rate (so a small sample is not over-read)."""
    if n == 0:
        return "-"
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return f"{100 * max(0, c - h):.0f}-{100 * min(1, c + h):.0f}%"


def report(rows: List[Dict], arms: List[str], which: str) -> str:
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["kind"])].append(r)
    arm_rows = defaultdict(list)
    for r in rows:
        if not r["ambiguous"]:
            arm_rows[r["arm"]].append(r)
    n_cases = len({r["case"] for r in rows})
    L = [f"# Independent test results ({which} set)\n",
         f"{n_cases} cases written by a separate Claude session that never saw our agent. "
         "Verdicts come from its own `judge()`. Ambiguous cases are excluded from the totals "
         "and listed separately. Intervals are 95% (Wilson), because the sample is small.\n",
         "## 1. Overall\n",
         "| Agent | Cases | Passed | Pass rate | 95% interval |", "|---|---:|---:|---:|---:|"]
    for a in arms:
        rs = arm_rows[a]
        k = sum(r["ok"] for r in rs)
        L.append(f"| {a} | {len(rs)} | {k} | {100 * k / max(1, len(rs)):.0f}% | {_wilson(k, len(rs))} |")
    L += ["\n## 2. By kind of case (pass rate)\n",
          "| Agent | fixable | unfixable (must page) | quiet (must stay quiet) |", "|---|---:|---:|---:|"]
    for a in arms:
        cells = []
        for kind in ("fixable", "unfixable", "quiet"):
            rs = [r for r in by[(a, kind)] if not r["ambiguous"]]
            cells.append(f"{100 * sum(r['ok'] for r in rs) / len(rs):.0f}% ({sum(r['ok'] for r in rs)}/{len(rs)})"
                         if rs else "-")
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    L += ["\n## 3. Speed\n",
          "Fixable cases that passed: mean unhealthy ticks (lower is better). "
          "Unfixable cases that paged: mean ticks from the first fault to the page.\n",
          "| Agent | Unhealthy ticks (fixable, passed) | Ticks to page (unfixable) |", "|---|---:|---:|"]
    for a in arms:
        fx = [r["unhealthy_ticks"] for r in by[(a, "fixable")] if r["ok"] and not r["ambiguous"]]
        pg = [r["ticks_to_page"] for r in by[(a, "unfixable")] if r["paged"]]
        L.append(f"| {a} | {_f(_mean(fx))} | {_f(_mean(pg))} |")
    tags = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["ambiguous"]:
            continue
        for t in r["tags"]:
            if t not in ("random", "fixable", "unfixable", "quiet"):
                tags[t][r["arm"]].append(r["ok"])
    L += ["\n## 4. Pass rate by tag (what each case was designed to expose)\n",
          "| Tag | n | " + " | ".join(arms) + " |", "|---|---:|" + "---:|" * len(arms)]
    for t in sorted(tags):
        n = max(len(v) for v in tags[t].values())
        if n < 3:
            continue
        L.append(f"| {t} | {n} | " + " | ".join(
            (f"{100 * sum(tags[t][a]) / len(tags[t][a]):.0f}%" if tags[t][a] else "-") for a in arms) + " |")
    # failures of the main agent, with the reason, so nothing is hidden
    main = "reasoning_plus" if "reasoning_plus" in arms else arms[-1]
    L += [f"\n## 5. Every case `{main}` did NOT pass (nothing hidden)\n",
          "| Case | Kind | Why it failed |", "|---|---|---|"]
    bad = sorted((r for r in rows if r["arm"] == main and not r["ok"]), key=lambda r: r["case"])
    for r in bad:
        L.append(f"| {r['case']}{' (ambiguous)' if r['ambiguous'] else ''} | {r['kind']} | {r['reason']} |")
    if not bad:
        L.append("| (none) | | |")
    amb = [r for r in rows if r["ambiguous"]]
    if amb:
        L += ["\n## 6. Ambiguous cases (reasonable agents could disagree)\n",
              "| Case | Agent | Passed | Reason |", "|---|---|---|---|"]
        for r in sorted(amb, key=lambda r: (r["case"], r["arm"])):
            L.append(f"| {r['case']} | {r['arm']} | {'yes' if r['ok'] else 'no'} | {r['reason']} |")
    L += ["\n## How to read this honestly\n",
          "- The cases were written independently of our agent, but they use OUR simulator, so "
          "this shows robustness to someone else's scenarios, not to a real production system.",
          "- If we change the agent after seeing the `dev` results, those cases become development "
          "cases. Only the sealed `final` set is clean evidence, and it is run once.",
          "- Seeds are fixed, so re-running reproduces these numbers."]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------
# Charts (for slides)
# ----------------------------------------------------------------------------
COLORS = {"do_nothing": "#9aa0a6", "rule_based": "#e8a33d", "evolution_only": "#7aa6d6",
          "full_system": "#1f5fbf", "reasoning_only": "#d9534f",
          "reasoning_hybrid": "#c9302c", "reasoning_plus": "#8e1b17"}


def _ci(k: int, n: int):
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def make_charts(rows: List[Dict], arms: List[str], which: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = [r for r in rows if not r["ambiguous"]]
    out = []

    # 1. overall pass rate with 95% intervals
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rates, lo, hi = [], [], []
    for a in arms:
        xs = [r for r in rs if r["arm"] == a]
        k, n = sum(r["ok"] for r in xs), len(xs)
        l, h = _ci(k, n)
        rates.append(100 * k / max(1, n)); lo.append(100 * l); hi.append(100 * h)
    bars = ax.bar(arms, rates, color=[COLORS.get(a, "#888") for a in arms],
                  yerr=[[r - l for r, l in zip(rates, lo)], [h - r for r, h in zip(rates, hi)]],
                  capsize=4, error_kw={"ecolor": "#444", "lw": 1.2})
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, 2, f"{r:.0f}%", ha="center", color="white", fontweight="bold")
    ax.set_ylim(0, 105); ax.set_ylabel("cases passed (%)")
    ax.set_title(f"Independent test cases ({which} set): pass rate, with 95% interval")
    ax.tick_params(axis="x", rotation=25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = RESULTS / f"chart_independent_{which}_overall.png"; fig.savefig(p, dpi=140); plt.close(fig); out.append(str(p))

    # 2. pass rate by kind of case, key agents only
    key = [a for a in ("rule_based", "evolution_only", "reasoning_only", "reasoning_hybrid",
                       "reasoning_plus") if a in arms] or arms
    kinds = [("fixable", "fixable faults"), ("unfixable", "unfixable (must page a human)"),
             ("quiet", "healthy shop (must stay quiet)")]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.8 / len(key)
    for j, a in enumerate(key):
        vals = []
        for kind, _ in kinds:
            xs = [r for r in rs if r["arm"] == a and r["kind"] == kind]
            vals.append(100 * sum(r["ok"] for r in xs) / max(1, len(xs)))
        ax.bar([i + j * w for i in range(len(kinds))], vals, w, label=a, color=COLORS.get(a, "#888"))
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(kinds))])
    ax.set_xticklabels([lbl for _, lbl in kinds])
    ax.set_ylim(0, 108); ax.set_ylabel("cases passed (%)")
    ax.set_title("Pass rate by kind of case")
    ax.legend(fontsize=9, ncol=len(key), loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    p = RESULTS / f"chart_independent_{which}_by_kind.png"; fig.savefig(p, dpi=140); plt.close(fig); out.append(str(p))

    # 3. reliability against speed (fixable cases): top-left is best
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for a in arms:
        if a == "do_nothing":
            continue
        fx = [r for r in rs if r["arm"] == a and r["kind"] == "fixable"]
        okx = [r["unhealthy_ticks"] for r in fx if r["ok"]]
        if not fx or not okx:
            continue
        ax.scatter(sum(okx) / len(okx), 100 * sum(r["ok"] for r in fx) / len(fx), s=130,
                   color=COLORS.get(a, "#888"), zorder=3)
        ax.annotate(a, (sum(okx) / len(okx), 100 * sum(r["ok"] for r in fx) / len(fx)),
                    textcoords="offset points", xytext=(8, -15 if a == "reasoning_plus" else 6), fontsize=9)
    ax.set_xlabel("average time the shop spent unhealthy (ticks, lower is better)")
    ax.set_ylabel("fixable cases passed (%)")
    ax.set_title("Reliability against speed (top left is best)")
    ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    p = RESULTS / f"chart_independent_{which}_tradeoff.png"; fig.savefig(p, dpi=140); plt.close(fig); out.append(str(p))
    return out


# ----------------------------------------------------------------------------
def main(which: str, opts: Dict[str, str]) -> None:
    arms = opts["arms"].split(",") if "arms" in opts else DEFAULT_ARMS
    cases = cases_for(which)
    if "limit" in opts:
        cases = cases[: int(opts["limit"])]
    RESULTS.mkdir(exist_ok=True)
    raw = RESULTS / f"independent_{which}_raw.jsonl"
    if opts.get("fresh") == "1" and raw.exists():
        raw.unlink()
    done: Dict[tuple, Dict] = {}
    if raw.exists():
        for line in raw.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["case"], r["arm"])] = r
    todo = [(c, a) for c in cases for a in arms if (c["name"], a) not in done]
    print(f"{which}: {len(cases)} cases x {len(arms)} agents = {len(cases) * len(arms)} episodes; "
          f"{len(done)} already saved, {len(todo)} to run.")
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(initializer=_init_worker) as ex, open(raw, "a") as fh:
            futs = [ex.submit(run_one, t) for t in todo]
            for n, f in enumerate(as_completed(futs), 1):
                r = f.result()
                done[(r["case"], r["arm"])] = r
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if n % 10 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} done  ({time.time() - t0:.0f}s)", flush=True)
    wanted = {(c["name"], a) for c in cases for a in arms}
    rows = [r for k, r in done.items() if k in wanted]
    md = report(rows, arms, which)
    (RESULTS / f"independent_{which}.md").write_text(md)
    (RESULTS / f"independent_{which}.json").write_text(json.dumps(rows, indent=1))
    print("\n" + md)
    charts = make_charts(rows, arms, which)
    print(f"Saved results/independent_{which}.md and .json and charts: " + ", ".join(charts))
    print(f"Total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    which = args[0] if args and "=" not in args[0] else "dev"
    if which not in ("dev", "final"):
        sys.exit("first argument must be 'dev' or 'final'")
    opts = dict(a.split("=", 1) for a in args if "=" in a)
    main(which, opts)
