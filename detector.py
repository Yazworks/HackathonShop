"""
detector.py - incident diagnosis for the HackathonShop recovery agent.

What this file does, in plain words
-----------------------------------
The detector looks at the shop's numbers (queue depth, order loss rate,
integrity, stuck jobs ...) AND its log lines (counts of WARN/ERROR messages
per service) and answers: "which problems are happening right now?"

Design: instead of one multiple-choice classifier, it has FOUR independent
yes/no checks, one per fault type:

    silent_dropper    - is the payment service silently losing orders?
    queue_overflow    - is the queue overloaded and dropping messages?
    data_corruption   - is the sync job damaging records?
    workflow_deadlock - is a workflow job frozen and blocking others?

Why: real incidents often come in combinations. Four independent checks can
name two or three problems at once, and can recognise combinations that were
never seen during training.

Training data comes from thousands of simulated shops. It is tested on:
  * new random seeds                        (in-distribution)
  * HARSHER fault settings than trained on  (held-out severities)
  * fault COMBINATIONS never seen in training
  * a brand-new kind of fault               (payment slowdown)
It also measures how many ticks detection takes and how often it raises
false alarms on healthy shops, and an ablation: does reading logs help?

Run:  python detector.py          (trains, evaluates, saves model + report)
      python detector.py quick    (smaller and faster, for a quick check)
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support

from shop import (
    FEATURE_NAMES,
    MetricsSnapshot,
    Shop,
    collect_metrics,
    inject_data_corruption,
    inject_queue_overflow,
    inject_silent_dropper,
    inject_workflow_deadlock,
    run_shop,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
LABELS = ["silent_dropper", "queue_overflow", "data_corruption", "workflow_deadlock"]
LOG_FEATURES = ["log_payment_warn", "log_queue_error", "log_sync_warn", "log_workflow_error"]

MODEL_PATH = "models/incident_detector.joblib"
REPORT_DIR = "results"

ACTIVE_AT = 0.5               # a check says "yes" when its probability is >= this
UNSURE_BAND = (0.25, 0.75)    # probabilities in this band count as "unsure"
CONFIRM_TICKS = 3             # an alarm must persist this many checks in a row
SNAP_TICKS = [10, 15, 20, 30, 45, 60]   # ticks after the fault starts when we sample

# Fault severities used for training, and HARSHER ones used only for testing
TRAIN_RANGES = {"drop_rate": (0.08, 0.30), "traffic": (2.0, 3.5), "corruption": (0.05, 0.12)}
HELDOUT_RANGES = {"drop_rate": (0.32, 0.50), "traffic": (3.6, 5.0), "corruption": (0.13, 0.20)}
# Much MILDER faults than trained on: the hard case, symptoms are faint
MILD_RANGES = {"drop_rate": (0.03, 0.07), "traffic": (1.7, 1.95), "corruption": (0.02, 0.04)}
LOG_DROPOUT = 0.3             # during training, hide the logs from this share of samples


def key_of(labels: Sequence[str]) -> str:
    return "+".join(sorted(labels)) if labels else "healthy"


# ----------------------------------------------------------------------------
# Generating simulated shops
# ----------------------------------------------------------------------------
def _inject(shop: Shop, labels: Sequence[str], rng: random.Random, ranges: Dict) -> None:
    if "silent_dropper" in labels:
        inject_silent_dropper(shop, rng.uniform(*ranges["drop_rate"]))
    if "queue_overflow" in labels:
        inject_queue_overflow(shop, rng.uniform(*ranges["traffic"]))
    if "data_corruption" in labels:
        inject_data_corruption(shop, rng.uniform(*ranges["corruption"]))
    if "workflow_deadlock" in labels:
        inject_workflow_deadlock(shop)


def _start_shop(labels, seed, ranges, custom, warmup_range=(25, 100)) -> Tuple[Shop, random.Random]:
    """Build a warmed-up shop and apply the fault(s) (or a harmless busy spell)."""
    rng = random.Random(seed * 7919 + 13)          # separate stream from the shop's own
    shop = Shop(seed=seed)
    run_shop(shop, rng.randint(*warmup_range))
    if custom is not None:
        custom(shop, rng)
    elif labels:
        _inject(shop, labels, rng, ranges)
    elif rng.random() < 0.4:
        shop.traffic_multiplier = rng.uniform(1.0, 1.5)     # busy but healthy
    return shop, rng


def simulate_snapshots(labels: Sequence[str], seed: int, ranges: Dict,
                       since: Sequence[int] = SNAP_TICKS,
                       custom: Optional[Callable] = None) -> List[Tuple[List[float], float]]:
    """Run one shop and record (features, health) at several ages."""
    shop, _ = _start_shop(labels, seed, ranges, custom)
    out, prev = [], 0
    for t in since:
        run_shop(shop, t - prev)
        prev = t
        m = collect_metrics(shop)
        out.append((m.to_features(), m.health_score))
    return out


def make_dataset(mix: List[Tuple[Tuple[str, ...], int]], seed_base: int, ranges: Dict,
                 custom: Optional[Callable] = None):
    X, Y, H, meta = [], [], [], []
    seed = seed_base
    for labels, n in mix:
        for _ in range(n):
            for feats, health in simulate_snapshots(labels, seed, ranges, custom=custom):
                X.append(feats)
                Y.append([1 if l in labels else 0 for l in LABELS])
                H.append(health)
                meta.append(key_of(labels))
            seed += 1
    return np.array(X, float), np.array(Y, int), np.array(H, float), meta


def trace_run(labels: Sequence[str], seed: int, ranges: Dict, ticks: int,
              warmup: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Per-tick features and health, starting at the moment the fault begins."""
    rng = random.Random(seed * 7919 + 13)
    shop = Shop(seed=seed)
    run_shop(shop, warmup)
    if labels:
        _inject(shop, labels, rng, ranges)
    elif rng.random() < 0.3:
        shop.traffic_multiplier = rng.uniform(1.0, 1.5)
    feats, health = [], []
    for _ in range(ticks):
        run_shop(shop, 1)
        m = collect_metrics(shop)
        feats.append(m.to_features())
        health.append(m.health_score)
    return np.array(feats, float), np.array(health, float)


# ----------------------------------------------------------------------------
# The detector
# ----------------------------------------------------------------------------
@dataclass
class Diagnosis:
    causes: List[str]              # e.g. ["silent_dropper"]; empty = looks healthy
    probs: Dict[str, float]        # probability each problem is happening
    confidence: float              # how decisive all four checks were (0.5..1)
    unsure: bool                   # any check sits in the uncertain band
    key: str                       # "healthy" or "a+b" (used to look up saved fixes)


class Detector:
    def __init__(self, heads: Dict[str, GradientBoostingClassifier],
                 feature_names: Sequence[str] = tuple(FEATURE_NAMES)):
        self.heads = heads
        self.feature_names = list(feature_names)
        self._idx = [FEATURE_NAMES.index(n) for n in self.feature_names]

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        """Probability matrix (rows = samples, columns = the four problems)."""
        Xs = X[:, self._idx]
        return np.column_stack([self.heads[l].predict_proba(Xs)[:, 1] for l in LABELS])

    def diagnose_features(self, feats: Sequence[float]) -> Diagnosis:
        p = self.predict_matrix(np.array([feats], float))[0]
        causes = [l for l, v in zip(LABELS, p) if v >= ACTIVE_AT]
        lo, hi = UNSURE_BAND
        return Diagnosis(
            causes=causes,
            probs={l: round(float(v), 3) for l, v in zip(LABELS, p)},
            confidence=float(np.min(np.maximum(p, 1 - p))),
            unsure=bool(np.any((p > lo) & (p < hi))),
            key=key_of(causes),
        )

    def diagnose(self, m: MetricsSnapshot) -> Diagnosis:
        return self.diagnose_features(m.to_features())

    def diagnose_shop(self, shop: Shop) -> Diagnosis:
        return self.diagnose(collect_metrics(shop))

    def top_features(self, label: str, k: int = 3) -> List[Tuple[str, float]]:
        imp = self.heads[label].feature_importances_
        order = np.argsort(imp)[::-1][:k]
        return [(self.feature_names[i], round(float(imp[i]), 3)) for i in order]


class DiagnosisStream:
    """
    Wraps a Detector for live use. An alarm only counts as CONFIRMED when the
    same diagnosis shows up `confirm` checks in a row. This stops one noisy
    reading from triggering a false alarm.
    """
    def __init__(self, detector: Detector, confirm: int = CONFIRM_TICKS):
        self.detector = detector
        self.confirm = confirm
        self._recent: Deque[frozenset] = deque(maxlen=confirm)

    def update(self, m: MetricsSnapshot) -> Diagnosis:
        d = self.detector.diagnose(m)
        self._recent.append(frozenset(d.causes))
        return d

    def confirmed(self) -> Optional[List[str]]:
        if (len(self._recent) == self.confirm and len(set(self._recent)) == 1
                and next(iter(self._recent))):
            return sorted(next(iter(self._recent)))
        return None

    def reset(self) -> None:
        self._recent.clear()


def zero_logs(X: np.ndarray) -> np.ndarray:
    """Simulate a broken logging pipeline: all log counts read as zero."""
    X = X.copy()
    for name in LOG_FEATURES:
        X[:, FEATURE_NAMES.index(name)] = 0.0
    return X


def train_heads(X: np.ndarray, Y: np.ndarray, feature_names: Sequence[str],
                n_estimators: int = 100, max_depth: int = 3,
                log_dropout: float = 0.0) -> Detector:
    """Train the four checks. log_dropout > 0 hides the logs from that share of
    training samples so the detector does not depend on logs alone."""
    if log_dropout > 0:
        hide = np.random.default_rng(0).random(len(X)) < log_dropout
        X = np.where(hide[:, None], zero_logs(X), X)
    idx = [FEATURE_NAMES.index(n) for n in feature_names]
    heads = {}
    for j, label in enumerate(LABELS):
        clf = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                         learning_rate=0.1, random_state=42)
        clf.fit(X[:, idx], Y[:, j])
        heads[label] = clf
    return Detector(heads, feature_names)


def save_detector(det: Detector, path: str = MODEL_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"heads": det.heads, "feature_names": det.feature_names, "labels": LABELS}, path)


def load_detector(path: str = MODEL_PATH) -> Detector:
    blob = joblib.load(path)
    return Detector(blob["heads"], blob["feature_names"])


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def score(det: Detector, X: np.ndarray, Y: np.ndarray, meta: Optional[List[str]] = None) -> Dict:
    pred = (det.predict_matrix(X) >= ACTIVE_AT).astype(int)
    per_head, f1s = {}, []
    for j, l in enumerate(LABELS):
        p, r, f, _ = precision_recall_fscore_support(
            Y[:, j], pred[:, j], average="binary", zero_division=0)
        per_head[l] = {"precision": float(p), "recall": float(r), "f1": float(f),
                       "support": int(Y[:, j].sum())}
        f1s.append(f)
    exact = (pred == Y).all(axis=1)
    out = {"per_head": per_head, "macro_f1": float(np.mean(f1s)),
           "exact_match": float(exact.mean()), "n": int(len(Y))}
    if meta is not None:
        comp = {}
        for name in sorted(set(meta)):
            idx = [i for i, m in enumerate(meta) if m == name]
            comp[name] = {"exact_match": float(exact[idx].mean()), "n": len(idx)}
        out["by_composition"] = comp
    return out


def _sets(det: Detector, X: np.ndarray) -> List[frozenset]:
    P = det.predict_matrix(X)
    return [frozenset(l for l, v in zip(LABELS, row) if v >= ACTIVE_AT) for row in P]


def _first_confirmed(sets: List[frozenset], target: frozenset, n: int) -> Optional[int]:
    for i in range(n - 1, len(sets)):
        if all(s == target for s in sets[i - n + 1: i + 1]):
            return i
    return None


def _any_alarm(sets: List[frozenset], n: int) -> bool:
    for i in range(n - 1, len(sets)):
        window = sets[i - n + 1: i + 1]
        if window[0] and all(s == window[0] for s in window):
            return True
    return False


def delay_eval(det: Detector, per_case: int, seed_base: int) -> Dict:
    """How many ticks after a fault starts does the detector confirm it?"""
    cases = {"silent_dropper": ("silent_dropper",), "queue_overflow": ("queue_overflow",),
             "data_corruption": ("data_corruption",), "workflow_deadlock": ("workflow_deadlock",),
             "silent_dropper+queue_overflow": ("silent_dropper", "queue_overflow")}
    out, seed = {}, seed_base
    for name, labels in cases.items():
        target = frozenset(labels)
        delays, unhealthy = [], []
        for _ in range(per_case):
            X, H = trace_run(labels, seed, TRAIN_RANGES, ticks=90)
            seed += 1
            i = _first_confirmed(_sets(det, X), target, CONFIRM_TICKS)
            delays.append(None if i is None else i + 1)
            below = np.where(H < 80.0)[0]
            unhealthy.append(None if len(below) == 0 else int(below[0]) + 1)
        found = [d for d in delays if d is not None]
        bad = [u for u in unhealthy if u is not None]
        out[name] = {
            "detected_pct": 100.0 * len(found) / len(delays),
            "median_ticks_to_confirm": float(np.median(found)) if found else None,
            "p90_ticks_to_confirm": float(np.percentile(found, 90)) if found else None,
            "median_ticks_until_health_below_80": float(np.median(bad)) if bad else None,
        }
    return out


def false_alarm_eval(det: Detector, n_shops: int, seed_base: int) -> Dict:
    """Run healthy shops for 200 ticks and count false alarms."""
    raw_alarm_ticks, total_ticks, shops_with_alarm = 0, 0, 0
    for i in range(n_shops):
        X, _ = trace_run((), seed_base + i, TRAIN_RANGES, ticks=200, warmup=20)
        sets = _sets(det, X)
        raw_alarm_ticks += sum(1 for s in sets if s)
        total_ticks += len(sets)
        shops_with_alarm += int(_any_alarm(sets, CONFIRM_TICKS))
    return {"shops": n_shops, "ticks": total_ticks,
            "raw_alarm_tick_pct": 100.0 * raw_alarm_ticks / total_ticks,
            "shops_with_confirmed_false_alarm_pct": 100.0 * shops_with_alarm / n_shops}


def novel_fault_eval(det: Detector, n: int, seed_base: int) -> Dict:
    """A fault type the detector was never trained on: payment slowdown."""
    def slow_payments(shop: Shop, rng: random.Random) -> None:
        shop.payments.throughput = rng.randint(2, 4)

    X, _, H, _ = make_dataset([((), n)], seed_base, TRAIN_RANGES, custom=slow_payments)
    keep = np.arange(len(X)) % len(SNAP_TICKS) >= 2          # ages 20+ so symptoms are visible
    X, H = X[keep], H[keep]
    pred = (det.predict_matrix(X) >= ACTIVE_AT)
    return {
        "samples": int(len(X)),
        "unhealthy_pct": float(100.0 * np.mean(H < 80.0)),
        "fired_pct": {l: float(100.0 * pred[:, j].mean()) for j, l in enumerate(LABELS)},
        "unhealthy_but_nothing_recognised_pct":
            float(100.0 * np.mean((H < 80.0) & (~pred.any(axis=1)))),
    }


# ----------------------------------------------------------------------------
# Main: train, evaluate, save, report
# ----------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def main(quick: bool = False) -> None:
    f = 0.3 if quick else 1.0
    n = lambda k: max(5, int(k * f))
    t0 = time.perf_counter()

    singles = [((l,), n(450)) for l in LABELS]
    train_mix = [((), n(700))] + singles + [(("silent_dropper", "queue_overflow"), n(450))]
    test_mix = ([((), n(150))] + [((l,), n(100)) for l in LABELS]
                + [(("silent_dropper", "queue_overflow"), n(100))])
    unseen_combos = [("data_corruption", "workflow_deadlock"), ("silent_dropper", "data_corruption"),
                     ("queue_overflow", "workflow_deadlock"), ("silent_dropper", "workflow_deadlock"),
                     ("queue_overflow", "data_corruption")]

    print("Generating training data from simulated shops ...")
    X_tr, Y_tr, _, _ = make_dataset(train_mix, 0, TRAIN_RANGES)
    print(f"  {len(X_tr)} samples from {sum(k for _, k in train_mix)} shops "
          f"({time.perf_counter() - t0:.0f}s)")

    print("Training the four checks (plus comparison models for the logs study) ...")
    det = train_heads(X_tr, Y_tr, FEATURE_NAMES, log_dropout=LOG_DROPOUT)      # the real one
    det_plain = train_heads(X_tr, Y_tr, FEATURE_NAMES)                         # no log dropout
    det_nolog = train_heads(X_tr, Y_tr, [x for x in FEATURE_NAMES if x not in LOG_FEATURES])
    save_detector(det)
    print(f"  saved {MODEL_PATH} ({time.perf_counter() - t0:.0f}s)")

    print("Generating test data (new seeds, harsher faults, unseen combinations) ...")
    X_id, Y_id, _, m_id = make_dataset(test_mix, 100_000, TRAIN_RANGES)
    X_ho, Y_ho, _, m_ho = make_dataset(test_mix, 200_000, HELDOUT_RANGES)
    combo_mix = [(c, n(80)) for c in unseen_combos]
    X_cb, Y_cb, _, m_cb = make_dataset(combo_mix, 300_000, TRAIN_RANGES)
    X_ml, Y_ml, _, m_ml = make_dataset(test_mix, 700_000, MILD_RANGES)

    res = {
        "in_distribution": score(det, X_id, Y_id, m_id),
        "heldout_severity": score(det, X_ho, Y_ho, m_ho),
        "mild_faults": score(det, X_ml, Y_ml, m_ml),
        "unseen_combinations": score(det, X_cb, Y_cb, m_cb),
        "logs_main_model_logs_available": score(det, X_ho, Y_ho),
        "logs_main_model_logs_missing": score(det, zero_logs(X_ho), Y_ho),
        "logs_plain_model_logs_missing": score(det_plain, zero_logs(X_ho), Y_ho),
        "logs_model_trained_without_logs": score(det_nolog, X_ho, Y_ho),
        "novel_fault_payment_slowdown": novel_fault_eval(det, n(80), 400_000),
        "detection_delay": delay_eval(det, n(40), 500_000),
        "false_alarms": false_alarm_eval(det, n(100), 600_000),
        "top_features": {l: det.top_features(l) for l in LABELS},
        "settings": {"quick": quick, "confirm_ticks": CONFIRM_TICKS, "active_at": ACTIVE_AT,
                     "train_ranges": TRAIN_RANGES, "heldout_ranges": HELDOUT_RANGES},
    }

    # ---- print a readable summary -------------------------------------------
    def table(title: str, r: Dict) -> None:
        print(f"\n{title}  (exact-match {_pct(r['exact_match'])}, macro F1 {r['macro_f1']:.3f}, "
              f"{r['n']} samples)")
        print(f"  {'check':<20} {'precision':>10} {'recall':>8} {'F1':>7}")
        for l in LABELS:
            h = r["per_head"][l]
            print(f"  {l:<20} {h['precision']:>10.3f} {h['recall']:>8.3f} {h['f1']:>7.3f}")

    table("1. NEW SEEDS, same fault settings", res["in_distribution"])
    table("2. HARSHER fault settings than trained on", res["heldout_severity"])
    print("\n3. COMBINATIONS NEVER SEEN IN TRAINING (both problems must be named):")
    for name, v in res["unseen_combinations"]["by_composition"].items():
        print(f"  {name:<40} exact-match {_pct(v['exact_match'])}  ({v['n']} samples)")

    table("2b. MUCH MILDER faults than trained on (the hard case)", res["mild_faults"])

    print("\n4. LOGS STUDY (harsher faults): do logs help, and what if they go missing?")
    for title, k in [("main model, logs available", "logs_main_model_logs_available"),
                     ("main model, logs MISSING at test time", "logs_main_model_logs_missing"),
                     ("plain model (no log dropout), logs MISSING", "logs_plain_model_logs_missing"),
                     ("model trained without logs at all", "logs_model_trained_without_logs")]:
        r = res[k]
        print(f"  {title:<46} exact-match {_pct(r['exact_match'])}  macro F1 {r['macro_f1']:.3f}")

    nv = res["novel_fault_payment_slowdown"]
    print(f"\n5. NEW KIND OF FAULT (payment slowdown, never trained on): "
          f"{nv['unhealthy_pct']:.0f}% of samples unhealthy")
    for l in LABELS:
        print(f"  {l:<20} fired on {nv['fired_pct'][l]:5.1f}% of samples")
    print(f"  unhealthy but NOTHING recognised: {nv['unhealthy_but_nothing_recognised_pct']:.1f}%")

    print(f"\n6. DETECTION SPEED (alarm confirmed after {CONFIRM_TICKS} identical checks in a row):")
    print(f"  {'incident':<32} {'found':>6} {'median ticks':>13} {'p90':>6} {'health<80 at':>13}")
    for name, v in res["detection_delay"].items():
        med = "-" if v["median_ticks_to_confirm"] is None else f"{v['median_ticks_to_confirm']:.0f}"
        p90 = "-" if v["p90_ticks_to_confirm"] is None else f"{v['p90_ticks_to_confirm']:.0f}"
        hb = ("-" if v["median_ticks_until_health_below_80"] is None
              else f"{v['median_ticks_until_health_below_80']:.0f}")
        print(f"  {name:<32} {v['detected_pct']:>5.0f}% {med:>13} {p90:>6} {hb:>13}")

    fa = res["false_alarms"]
    print(f"\n7. FALSE ALARMS on {fa['shops']} healthy shops x 200 ticks: "
          f"{fa['raw_alarm_tick_pct']:.2f}% of single checks raised an alarm; "
          f"{fa['shops_with_confirmed_false_alarm_pct']:.0f}% of shops had a confirmed false alarm")

    print("\nMost important signals per check:")
    for l in LABELS:
        print(f"  {l:<20} " + ", ".join(f"{n_} ({v:.2f})" for n_, v in res["top_features"][l]))

    # ---- save the report -------------------------------------------------------
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    Path(REPORT_DIR, "detector_report.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved {REPORT_DIR}/detector_report.json   Total time {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main(quick=(len(sys.argv) > 1 and sys.argv[1] == "quick"))
