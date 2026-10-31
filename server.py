"""
server.py - the live operations console for HackathonShop.

What this file does, in plain words
-----------------------------------
It runs the shop and the recovery agent in a background thread and streams
everything that happens (health, metrics, the agent's decisions, evolution
progress, saved playbooks) to the web page over a WebSocket. The page can also
send commands back: break the shop in different ways, make a problem
unfixable, let a "human" fix the root cause, change speed, pause, reset.

Start it with:   python main.py      then open  http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from detector import load_detector
from evolution import HELDOUT_PARAMS, TRAIN_PARAMS, _reapply
from playbooks import PLAYBOOK_PATH, STARTER_PATH, PlaybookLibrary
from recovery_agent import RecoveryAgent, format_event
from shop import FAULTS, Shop, apply_action

BASE = Path(__file__).parent
WARMUP_TICKS = 60
DEFAULT_SPEED = 5.0          # simulated ticks per real second

# faults the console can inject (name -> function(shop, severity))
FAULT_LABELS = {
    "silent_dropper": "Silent order dropper",
    "queue_overflow": "Traffic spike (queue overflow)",
    "data_corruption": "Data corruption",
    "workflow_deadlock": "Frozen workflow",
    "combo_corruption_deadlock": "Corruption + frozen workflow",
    "payment_slowdown": "Slow payments (never seen before)",
    "outage_scale_queue": "Tool outage: scale_queue broken",
    "payment_leak": "Payment service slowly degrading (leak)",
    "flapping": "Flapping fault (comes and goes)",
    "cascade": "Cascade: traffic spike that then corrupts data",
}


class ResetSignal(Exception):
    """Raised inside the simulation thread to stop and restart cleanly."""


def _default(o):
    try:
        return float(o)
    except Exception:
        return str(o)


def dumps(obj) -> str:
    return json.dumps(obj, default=_default)


class Session:
    def __init__(self, keep_memory: bool = False):
        self.keep_memory = keep_memory
        self.detector = load_detector()
        self.cmds: "queue.Queue[Dict]" = queue.Queue()
        self.clients: List[asyncio.Queue] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.speed = DEFAULT_SPEED
        self.paused = False
        self.reset_requested = False
        self.forget_requested = False
        self.build(clear_memory=not keep_memory)

    # ----------------------------------------------------------- set-up
    def build(self, clear_memory: bool) -> None:
        path = Path(PLAYBOOK_PATH)
        if clear_memory and path.exists():
            path.unlink()
        self.reset_requested = False
        self.shop = Shop(seed=int(time.time()) % 100000)
        self.library = PlaybookLibrary(PLAYBOOK_PATH, starter_path=STARTER_PATH)
        self.persist: Dict = {}
        self.injected: List[str] = []
        self.phase = "MONITORING"
        self.detail = "Watching the shop."
        self.last_health: Optional[float] = None
        self.agent = RecoveryAgent(self.shop, self.detector, self.library, seed=1,
                                   record_events=False, act_immediately=True, use_reasoning=True,
                                   before_tick=self._before_tick)
        self.agent.run(WARMUP_TICKS)              # quiet warm-up, nothing streamed
        self.agent.emit = self._emit
        self.agent.incidents.clear()

    # ------------------------------------------------------------ commands
    def _before_tick(self, shop: Shop) -> None:
        if self.reset_requested:
            raise ResetSignal()
        while True:
            try:
                c = self.cmds.get_nowait()
            except queue.Empty:
                break
            self._handle(c)
        if self.persist:
            _reapply(shop, self.persist)

    def _handle(self, c: Dict) -> None:
        kind = c.get("cmd")
        if kind == "inject":
            self._inject(c.get("fault", ""), c.get("severity", "normal"), bool(c.get("sticky")))
        elif kind == "fix_root_cause":
            self.persist = {}
            for a in ("restart_payments", "rollback_sync", "kill_stuck_job"):
                apply_action(self.shop, a, 1.0)
            self.injected.clear()
            self.agent.resolve()
            self._publish_text("A human fixed the root cause. Agent resumes monitoring.", "human")
        elif kind == "noise":
            on = bool(c.get("value"))
            self.shop.sensor_noise = 0.03 if on else 0.0
            self.shop.metrics_dropout = 0.10 if on else 0.0
            self.shop.log_dropout = 0.30 if on else 0.0
            self._publish_text("Sensors now " + ("NOISY: readings jitter, some go stale, "
                               "30% of log lines are lost" if on else "clean again"), "fault")
        elif kind == "speed":
            self.speed = max(0.5, min(60.0, float(c.get("value", DEFAULT_SPEED))))
        elif kind == "pause":
            self.paused = bool(c.get("value"))

    def _inject(self, fault: str, severity: str, sticky: bool) -> None:
        params = HELDOUT_PARAMS if severity == "harsh" else TRAIN_PARAMS
        s = self.shop
        if fault in ("silent_dropper", "queue_overflow", "data_corruption", "workflow_deadlock"):
            FAULTS[fault](s, **params[fault])
            if sticky:
                if fault == "silent_dropper":
                    self.persist["drop_rate"] = params[fault]["drop_rate"]
                elif fault == "data_corruption":
                    self.persist["corruption_rate"] = params[fault]["corruption_rate"]
                elif fault == "workflow_deadlock":
                    self.persist["deadlock"] = True
        elif fault == "combo_corruption_deadlock":
            FAULTS["data_corruption"](s, **params["data_corruption"])
            FAULTS["workflow_deadlock"](s)
        elif fault in ("payment_leak", "flapping", "cascade"):
            FAULTS[fault](s)
        elif fault == "payment_slowdown":
            s.payments.throughput = 3
        elif fault == "outage_scale_queue":
            s.disabled_actions.add("scale_queue")
        else:
            return
        label = FAULT_LABELS.get(fault, fault) + (" [root cause keeps returning]" if sticky else "")
        self.injected.append(label)
        self._publish_text(f"FAULT INJECTED: {label} ({severity})", "fault")

    # ------------------------------------------------------------- streaming
    def _publish(self, msg: Dict) -> None:
        if self.loop is None:
            return
        text = dumps(msg)

        def push() -> None:
            for q in self.clients:
                if q.qsize() < 2000:
                    q.put_nowait(text)

        self.loop.call_soon_threadsafe(push)

    def _publish_text(self, text: str, kind: str) -> None:
        self._publish({"type": "event", "tick": self.shop.tick, "kind": kind, "text": text})

    def state_message(self) -> Dict:
        a = self.agent
        return {"type": "state", "tick": self.shop.tick, "agent_state": a.state,
                "phase": self.phase, "detail": self.detail,
                "playbooks": self.library.rows(),
                "incidents": a.incidents[-12:], "injected": self.injected[-8:],
                "sticky": bool(self.persist), "speed": self.speed, "paused": self.paused,
                "outage": sorted(self.shop.disabled_actions),
                "noisy": self.shop.sensor_noise > 0}

    def _emit(self, e: Dict) -> None:
        k, d = e["type"], e["data"]
        if k == "metrics":
            m = d["snapshot"]
            self.last_health = m["health_score"]
            keep = ("tick", "health_score", "queue_depth", "queue_capacity", "queue_saturation",
                    "drop_rate", "mismatch_rate", "integrity_score", "jobs_stuck",
                    "stuck_duration", "orders_per_second", "total_revenue",
                    "order_payment_mismatch", "jobs_pending")
            self._publish({"type": "metrics", "m": {x: m[x] for x in keep},
                           "diagnosis": d["diagnosis"], "beliefs": d.get("beliefs"),
                           "phase": self.phase,
                           "detail": self.detail, "agent_state": self.agent.state})
            if m["tick"] % 5 == 0:                       # keeps tables fresh after incidents close
                self._publish(self.state_message())
            while self.paused and not self.reset_requested:
                time.sleep(0.1)
            time.sleep(1.0 / self.speed)
            return
        if k == "evolution_update":
            self.phase = "PLANNING"
            self.detail = (f"Evolving a fix: generation {d['generation']}, "
                           f"best fitness {d['best']}")
            if self.reset_requested:
                raise ResetSignal()
            self._publish({"type": "evo", "generation": d["generation"],
                           "best": d["best"], "avg": d["avg"], "tick": e["tick"]})
            return
        # phase bookkeeping
        if k in ("incident_declared", "replan"):
            self.phase, self.detail = "PLANNING", "Incident declared. Choosing a fix."
        elif k == "memory_hit":
            self.detail = f"Found a trusted playbook ({d['playbook']}). Rehearsing it first."
        elif k == "plan_ready":
            self.phase, self.detail = "FIXING", "Applying the fix and watching the shop."
        elif k in ("verification", "self_resolved", "auto_cleared"):
            self.phase, self.detail = "MONITORING", "Recovered and verified. Watching the shop."
        elif k == "escalation":
            self.phase, self.detail = "PAGED", "A human has been paged. Waiting for them."
        text = format_event(e)
        kind = {"incident_declared": "incident", "action_applied": "action",
                "verification": "verified", "escalation": "page", "playbook_saved": "memory",
                "memory_hit": "memory", "playbook_failed": "bad", "replan": "bad",
                "dry_run": "dry", "reasoning": "reason"}.get(k, "info")
        if text:
            self._publish({"type": "event", "tick": e["tick"], "kind": kind, "text": text,
                           "event": k})
        self._publish(self.state_message())

    # --------------------------------------------------------- main thread
    def run_forever(self) -> None:
        while True:
            try:
                self.agent.tick()
            except ResetSignal:
                self.build(clear_memory=self.forget_requested)   # the first start wipes memory; a reset only if asked
                self.forget_requested = False
                self._publish({"type": "reset"})
                self._publish(self.state_message())
            except Exception as ex:                      # never let the thread die silently
                self._publish_text(f"internal error: {ex!r}", "bad")
                time.sleep(1.0)


# ----------------------------------------------------------------------------
# Web app
# ----------------------------------------------------------------------------
app = FastAPI(title="HackathonShop Recovery Console")
SESSION: Optional[Session] = None


def start_session(keep_memory: bool = False) -> Session:
    global SESSION
    SESSION = Session(keep_memory=keep_memory)
    return SESSION


@app.on_event("startup")
async def _startup() -> None:
    global SESSION
    if SESSION is None:
        SESSION = Session(keep_memory=False)
    SESSION.loop = asyncio.get_running_loop()
    threading.Thread(target=SESSION.run_forever, daemon=True, name="simulation").start()


@app.get("/")
async def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/config")
async def config():
    return {"faults": FAULT_LABELS}


@app.get("/api/state")
async def state():
    return JSONResponse(json.loads(dumps(SESSION.state_message())))


@app.post("/api/command")
async def command(body: Dict):
    c = body.get("cmd")
    if c == "reset":
        SESSION.forget_requested = bool(body.get("forget"))
        SESSION.reset_requested = True
        SESSION.paused = False
    else:
        SESSION.cmds.put(body)
        if c == "pause":
            SESSION.paused = bool(body.get("value"))       # takes effect immediately
        if c == "speed":
            SESSION.speed = max(0.5, min(60.0, float(body.get("value", DEFAULT_SPEED))))
    return {"ok": True}


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    q: asyncio.Queue = asyncio.Queue()
    SESSION.clients.append(q)
    try:
        await socket.send_text(dumps(SESSION.state_message()))
        await socket.send_text(dumps({"type": "history",
                                      "health": list(SESSION.agent.health_trace)[-150:],
                                      "end_tick": SESSION.shop.tick}))
        while True:
            await socket.send_text(await q.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if q in SESSION.clients:
            SESSION.clients.remove(q)


app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
