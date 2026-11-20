"""Forward simulation of the shared-station line under retrofit options.

Operating rules are the ones recovered from the recorded history and verified
by :func:`planner.history.replay_check`:

* assembly on L1/L2 takes ``units + 5`` minutes and holds one line slot;
* a per-shift transfer delay precedes the first inspection (3/7/11/15/19 min);
* first inspection takes ``min(14, 2*units)`` minutes on QA, or QA2 when the
  product's route allows the alternative;
* a rejected lot of 2 units goes to RW for 18 minutes and back to a 4-minute
  recheck; a partial order may release the passed remainder before that;
* the buffer holds **lots**, not pieces (厚板 20 units/lot, 薄板 30 units/lot).
  When it is full, the finished batch cannot leave the line, so the line is
  blocked - this is the "downstream full" starvation the plant described;
* each option consumes one slot of its resource for its whole installation
  window (the temporary capacity loss during construction);
* WIP and priority orders are dispatched before ordinary work.

Everything is deterministic: a minute loop, fixed dispatch order (committed ->
priority -> EDD -> order id).  No randomness, so two runs of the same inputs
always produce the same numbers and every result can be traced to an order.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .model import Option, Pack, TimingRules, lot_count

RESOURCES = ("L1", "L2", "QA", "QA2", "RW", "BUF")
BUFFER = "BUF"

STAGE_LABEL = {
    "material": "等料",
    "line_queue": "等待装配",
    "assembly": "装配中",
    "buffer_queue": "装配完成但缓冲区无位",
    "buffer": "缓冲区待复验",
    "inspection": "复验中",
    "rw_queue": "等待返工",
    "rework": "返工中",
    "recheck_queue": "等待复检",
    "recheck": "复检中",
    "done_pending": "交付登记中",
    "done": "已交付",
}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Job:
    order: str
    product: str
    line: str
    units: int
    available_min: int
    due_min: int
    priority: bool
    allow_partial: bool
    material_min: int
    committed: bool = False
    stage0: str = "new"          # new | buffer | rw
    rework: bool = False
    # a WIP batch already inspected once (only the recheck remains)
    inspected: bool = False


@dataclass(frozen=True)
class World:
    """One admissible reading of facts the pack leaves open.

    ``rework_orders`` - which future orders will be rejected at inspection.
    The pack records rejections only for the past, so this is never a fact.
    ``material_slip_min`` - extra material delay applied to every planning
    order (the pack gives arrivals only for historical orders).
    ``release_span_shifts`` - compress all planning orders into the first N
    shifts, preserving their order.  The pack schedules them across three
    shifts; the shop says the order mix varies a lot, so this is the lever
    that shows when capacity actually binds.
    ``due_scale`` - multiply each order's promised lead time.  Used only with
    ``release_span_shifts``; makes the compression bite.
    """
    label: str
    rework_orders: tuple[str, ...] = ()
    material_slip_min: int = 0
    release_span_shifts: int | None = None
    due_scale: float = 1.0
    multiplier: int = 1

    def as_dict(self) -> dict:
        return {"label": self.label, "rework_orders": list(self.rework_orders),
                "material_slip_min": self.material_slip_min,
                "release_span_shifts": self.release_span_shifts,
                "due_scale": self.due_scale, "multiplier": self.multiplier}


@dataclass(frozen=True)
class SimConfig:
    options: frozenset[str] = frozenset()
    route_mode: str = "primary"      # primary | allow_alt
    install_loss: int = 1
    transfer_delay: bool = True
    buffer_mode: str = "lots"        # lots | orders
    recheck_wait_min: int = 0        # extra conservatism knob for QA queueing


@dataclass
class OrderResult:
    order: str
    units: int
    due_min: int
    product: str
    line: str
    priority: bool
    committed: bool
    deliveries: list[tuple[int, int]] = field(default_factory=list)
    waits: dict[str, int] = field(default_factory=dict)
    stage_at_end: str = "done"
    rework: bool = False
    block_reason: str | None = None

    @property
    def delivered(self) -> int:
        return sum(u for u, _ in self.deliveries)

    @property
    def on_time_units(self) -> int:
        return sum(u for u, t in self.deliveries if t <= self.due_min)

    @property
    def late_units(self) -> int:
        return self.units - self.on_time_units

    @property
    def late_unit_minutes(self) -> int:
        return sum(u * max(0, t - self.due_min) for u, t in self.deliveries)

    @property
    def final_delivery_min(self) -> int | None:
        return max((t for _, t in self.deliveries), default=None)

    @property
    def on_time_share(self) -> float:
        return self.on_time_units / self.units if self.units else 1.0

    def as_dict(self) -> dict:
        return {
            "order": self.order, "product": self.product, "line": self.line,
            "units": self.units, "due_min": self.due_min,
            "priority": self.priority, "committed": self.committed,
            "delivered_units": self.delivered, "on_time_units": self.on_time_units,
            "late_units": self.late_units,
            "late_unit_minutes": self.late_unit_minutes,
            "final_delivery_min": self.final_delivery_min,
            "stage_at_end": self.stage_at_end,
            "stage_at_end_label": STAGE_LABEL.get(self.stage_at_end, self.stage_at_end),
            "waits_min": dict(self.waits),
            "block_reason": self.block_reason,
            "rework": self.rework,
        }


@dataclass
class SimResult:
    orders: dict[str, OrderResult]
    t0: int
    t1: int
    config: SimConfig
    world: World
    resource_busy: dict[str, int] = field(default_factory=dict)
    resource_capacity: dict[str, int] = field(default_factory=dict)
    resource_capacity_change: dict[str, int] = field(default_factory=dict)
    buffer_peak_lots: int = 0
    buffer_peak_min: int | None = None
    buffer_block_minutes: int = 0
    note: list[str] = field(default_factory=list)

    @property
    def backlog_units(self) -> int:
        return sum(r.units - r.delivered for r in self.orders.values())

    @property
    def delivered_units(self) -> int:
        return sum(r.delivered for r in self.orders.values())

    @property
    def on_time_units(self) -> int:
        return sum(r.on_time_units for r in self.orders.values())

    @property
    def late_units(self) -> int:
        return sum(r.late_units for r in self.orders.values())

    @property
    def late_unit_minutes(self) -> int:
        return sum(r.late_unit_minutes for r in self.orders.values())

    @property
    def committed_breach(self) -> list[str]:
        return sorted(r.order for r in self.orders.values()
                      if r.committed and (r.late_units > 0 or r.delivered < r.units))

    @property
    def late_orders(self) -> list[str]:
        return sorted(r.order for r in self.orders.values() if r.late_units > 0)

    def utilization(self, res: str) -> float:
        cap = self.resource_capacity.get(res, 0)
        return round(self.resource_busy.get(res, 0) / cap, 3) if cap else 0.0

    def summary(self) -> dict:
        return {
            "options": sorted(self.config.options),
            "route_mode": self.config.route_mode,
            "world": self.world.label,
            "horizon": [self.t0, self.t1],
            "delivered_units": self.delivered_units,
            "on_time_units": self.on_time_units,
            "late_units": self.late_units,
            "late_unit_minutes": self.late_unit_minutes,
            "backlog_units": self.backlog_units,
            "buffer_peak_lots": self.buffer_peak_lots,
            "buffer_peak_min": self.buffer_peak_min,
            "buffer_block_minutes": self.buffer_block_minutes,
            "committed_breach": self.committed_breach,
            "utilization": {r: self.utilization(r) for r in RESOURCES if self.resource_capacity.get(r)},
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# Capacity
# --------------------------------------------------------------------------- #
def capacity_timeline(pack: Pack, config: SimConfig, t0: int, t1: int) -> dict[str, list[int]]:
    base = pack.baseline_capacity
    caps = {r: [base.get(r, 0)] * (t1 - t0) for r in RESOURCES}
    for name in config.options:
        opt: Option = pack.options[name]
        if opt.resource not in caps:
            continue
        for i, t in enumerate(range(t0, t1)):
            c = caps[opt.resource][i]
            during = opt.install_start_min <= t < opt.install_end_min
            if during:
                c = max(0, c - config.install_loss)
            if t >= opt.effective_min and not during:
                c += opt.delta
            caps[opt.resource][i] = c
    return caps


def option_cost(pack: Pack, names) -> int:
    return sum(pack.options[n].cost_cents for n in names)


def exclusive_violations(pack: Pack, names) -> list[str]:
    seen: dict[str, str] = {}
    bad = []
    for n in sorted(names):
        g = pack.options[n].exclusive_group
        if g is None:
            continue
        if g in seen:
            bad.append(f"互斥组「{g}」同时选中 {seen[g]} 和 {n}")
        seen[g] = n
    return bad


def feasible_option_sets(pack: Pack) -> list[dict]:
    names = sorted(pack.options)
    out = []
    for r in range(len(names) + 1):
        for combo in itertools.combinations(names, r):
            cost = option_cost(pack, combo)
            reasons = exclusive_violations(pack, combo)
            if cost > pack.budget_cents:
                reasons.append(f"造价 {cost / 100:.0f} 元超过预算 {pack.budget_cents / 100:.0f} 元")
            out.append({"options": frozenset(combo), "cost_cents": cost,
                        "feasible": not reasons, "violations": reasons})
    out.sort(key=lambda s: (not s["feasible"], len(s["options"]), s["cost_cents"], sorted(s["options"])))
    return out


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
class _Board:
    """Per-minute occupancy of every resource, plus lot occupancy of the buffer."""

    def __init__(self, caps: dict[str, list[int]], t0: int, t1: int,
                 shift_minutes: int = 480):
        self.caps = caps
        self.t0 = t0
        self.t1 = t1
        self.used = {r: {} for r in RESOURCES}
        self.buffer = {t: 0 for t in range(t0, t1)}
        self.buffer_peak = 0
        self.buffer_peak_min = None
        self.buffer_block_minutes = 0
        self.shift_minutes = shift_minutes

    def free(self, res: str, t: int, dur: int = 1) -> bool:
        if res not in self.caps or t + dur > self.t1:
            return False
        idx = t - self.t0
        cap = self.caps[res]
        if idx < 0:
            return False
        for x in range(idx, min(len(cap), idx + dur)):
            if cap[x] - self.used[res].get(self.t0 + x, 0) <= 0:
                return False
        return True

    def take(self, res: str, t: int, dur: int) -> None:
        for x in range(t, min(self.t1, t + dur)):
            self.used[res][x] = self.used[res].get(x, 0) + 1

    def hold(self, res: str, t: int) -> None:
        self.used[res][t] = self.used[res].get(t, 0) + 1

    def buffer_free(self, t: int) -> int:
        return self.caps[BUFFER][t - self.t0] - self.buffer[t]

    def buffer_take(self, lots: int, t: int) -> None:
        for x in range(t, self.t1):
            self.buffer[x] += lots
        occ = self.buffer[t]
        if occ > self.buffer_peak:
            self.buffer_peak, self.buffer_peak_min = occ, t

    def buffer_release(self, lots: int, t: int) -> None:
        for x in range(t, self.t1):
            self.buffer[x] -= lots


def cadence_release(asm_end: int, shift_minutes: int, rules: TimingRules,
                    transfer_delay: bool = True) -> int:
    """Earliest minute a finished batch may enter its first inspection.

    The shop inspects in per-shift rounds: the first inspection of a batch
    completed in shift ``k`` starts no earlier than ``3 + 4*(k % 5)`` minutes
    into the *next* available round.  This is a calendar cadence, not a
    capacity queue - the QA unit is idle for most of it.  ``transfer_delay``
    can be switched off to show what the cadence alone costs.
    """
    if not transfer_delay:
        return asm_end
    shift = asm_end // shift_minutes
    wave = rules.wave_base_min + rules.wave_step_min * (shift % rules.wave_period_shifts)
    return asm_end + wave


def _route(pack: Pack, product: str, config: SimConfig) -> list[str]:
    """Inspection resources in preference order.

    ``allow_alt`` mirrors what the shop may do when a product's route permits
    an alternative: the primary station stays first, and the alternative is
    tried only when the primary has no slot at that minute.  Without
    ``allow_alt`` only the primary station is used.
    """
    rule = pack.route_rules[product]
    if config.route_mode == "allow_alt" and rule.alternative:
        return [rule.inspection, rule.alternative]
    return [rule.inspection]


def make_jobs(pack: Pack, world: World, include_wip: bool = True) -> list[Job]:
    jobs: list[Job] = []
    if include_wip:
        for w in pack.wip:
            c = pack.wip_commitments[w["order"]]
            jobs.append(Job(
                order=w["order"], product=w["product"], line="WIP", units=int(w["units"]),
                available_min=pack.horizon[0], due_min=int(c["due_min"]),
                priority=True, allow_partial=bool(c["allow_partial"]),
                material_min=pack.horizon[0],
                committed=True,
                stage0="rw" if w["stage"] == "返工中" else "buffer",
                rework=w["stage"] == "返工中"))
    rework = set(world.rework_orders)
    plan = [dict(o) for o in pack.planning_orders]
    if world.release_span_shifts is not None:
        # Compress the release pattern into N shifts, keeping order.  Due dates
        # keep their original lead time unless due_scale shortens them.
        plan.sort(key=lambda o: (o["available_min"], o["order"]))
        span = world.release_span_shifts
        n = max(1, len(plan) - 1)
        start = pack.horizon[0]
        for idx, o in enumerate(plan):
            lead = int((o["due_min"] - o["available_min"]) * world.due_scale)
            offset = int(round(idx / n * (span - 1))) * pack.shift_minutes
            o["available_min"] = start + offset
            o["due_min"] = start + offset + lead
    for m in range(max(1, world.multiplier)):
        for o in plan:
            avail = int(o["available_min"]) + world.material_slip_min
            # repeated copies share the same release; ids are suffixed so the
            # per-order audit trail still points at a concrete order
            order_id = o["order"] if m == 0 else f"{o['order']}^{m + 1}"
            jobs.append(Job(
                order=order_id, product=o["product"], line=o["line"],
                units=int(o["units"]), available_min=avail, due_min=int(o["due_min"]),
                priority=bool(o["priority"]), allow_partial=bool(o["allow_partial"]),
                material_min=avail, rework=o["order"] in rework))
    return jobs


def simulate(pack: Pack, jobs: list[Job], config: SimConfig, world: World,
             rules: TimingRules = TimingRules(),
             t0: int | None = None, t1: int | None = None) -> SimResult:
    t0 = pack.horizon[0] if t0 is None else t0
    t1 = pack.horizon[1] if t1 is None else t1
    board = _Board(capacity_timeline(pack, config, t0, t1), t0, t1, pack.shift_minutes)
    by_id = {j.order: j for j in jobs}
    res = {j.order: OrderResult(order=j.order, units=j.units, due_min=j.due_min,
                                product=j.product, line=j.line, priority=j.priority,
                                committed=j.committed, rework=j.rework) for j in jobs}
    st: dict[str, dict] = {}
    for j in jobs:
        lots = lot_count(j.units, j.product, pack) if config.buffer_mode == "lots" else 1
        st[j.order] = {
            "stage": "line_queue", "end": None, "ready": j.material_min,
            "release": j.material_min, "lots": lots, "asm_end": None,
            "blocked_line": None, "buf_in": None,
        }
        if j.stage0 == "buffer":
            # already waiting for a first inspection: no cadence delay applies
            st[j.order].update(stage="buffer", ready=t0, release=t0, buf_in=t0)
            board.buffer_take(lots, t0)
        elif j.stage0 == "rw":
            st[j.order].update(stage="rw_queue", ready=t0, release=t0)
        res[j.order].waits = {"line": 0, "buffer": 0, "inspection": 0, "rework": 0, "recheck": 0}

    pending: list[tuple[int, str, int]] = []     # (time, order, units)
    delivered_events: list[tuple[int, str, int]] = []

    def rank(order: str):
        r = res[order]
        return (0 if r.committed else (1 if r.priority else 2), r.due_min, order)

    def release_pending(t: int):
        nonlocal pending
        keep = []
        for tt, oid, units in pending:
            if tt <= t:
                res[oid].deliveries.append((units, tt))
                delivered_events.append((tt, oid, units))
            else:
                keep.append((tt, oid, units))
        pending = keep

    def pick_inspection(job: Job, t: int, dur: int) -> str | None:
        for resname in _route(pack, job.product, config):
            if board.free(resname, t, dur):
                return resname
        return None

    def buffer_in(t: int, order: str) -> bool:
        s = st[order]
        if board.buffer_free(t) >= s["lots"]:
            board.buffer_take(s["lots"], t)
            s["stage"] = "buffer"
            s["ready"] = t
            s["buf_in"] = t
            res[order].waits["buffer"] += max(0, t - s["asm_end"])
            if s["blocked_line"]:
                s["blocked_line"] = None
            return True
        return False

    for t in range(t0, t1):
        release_pending(t)
        # --- finish operations that ended at or before t
        for order, s in st.items():
            if s["end"] is not None and s["end"] <= t:
                stage = s["stage"]
                s["end"] = None
                job = by_id[order]
                if stage == "assembly":
                    s["asm_end"] = t
                    s["stage"] = "buffer_queue"
                    s["ready"] = t
                    # the next inspection round for this shift's completions
                    s["release"] = cadence_release(t, board.shift_minutes, rules,
                                                   config.transfer_delay)
                elif stage == "inspection":
                    if job.rework:
                        if job.allow_partial and job.units > rules.rework_lot_units:
                            pending.append((t + rules.pass_to_ship_min, order,
                                            job.units - rules.rework_lot_units))
                        s["stage"] = "rw_queue"
                        s["ready"] = t + rules.qa_to_rw_gap_min
                    else:
                        pending.append((t + rules.pass_to_ship_min, order, job.units))
                        s["stage"] = "done_pending"
                elif stage == "rework":
                    s["stage"] = "recheck_queue"
                    s["ready"] = t + rules.rw_to_recheck_gap_min
                elif stage == "recheck":
                    # A rework batch is released at the FULL batch size unless
                    # the order already released a partial shipment from the
                    # first inspection; that remainder only comes off the line
                    # once the reworked lot clears.
                    if job.stage0 == "rw":
                        # the whole WIP batch is in rework, not a 2-unit lot
                        units = job.units
                    elif job.allow_partial and job.units > rules.rework_lot_units:
                        units = rules.rework_lot_units
                    else:
                        units = job.units
                    pending.append((t + rules.finalized_ship_min, order, units))
                    s["stage"] = "done_pending"
        for order, s in st.items():
            if s["stage"] == "done_pending":
                s["stage"] = "done"
        # --- buffer admission; a stuck batch keeps holding its line
        for order in sorted([o for o, s in st.items() if s["stage"] == "buffer_queue"], key=rank):
            s = st[order]
            if buffer_in(t, order):
                continue
            job = by_id[order]
            if job.line in board.caps:
                board.hold(job.line, t)
            s["blocked_line"] = job.line
            board.buffer_block_minutes += 1
        # --- inspection starts
        for order in sorted([o for o, s in st.items()
                             if s["stage"] == "buffer" and s["ready"] <= t
                             and t >= s.get("release", t)], key=rank):
            job = by_id[order]
            dur = min(rules.qa_max_min, int(job.units / rules.qa_units_per_min))
            chosen = pick_inspection(job, t, dur)
            if chosen is None:
                continue
            board.take(chosen, t, dur)
            s = st[order]
            board.buffer_release(s["lots"], t)
            res[order].waits["inspection"] += max(0, t - s["buf_in"])
            s.update(stage="inspection", end=t + dur, ready=t)
        # --- rework starts
        for order in sorted([o for o, s in st.items() if s["stage"] == "rw_queue" and s["ready"] <= t], key=rank):
            if not board.free("RW", t, rules.rework_min):
                break
            board.take("RW", t, rules.rework_min)
            s = st[order]
            res[order].waits["rework"] += max(0, t - s["ready"])
            s.update(stage="rework", end=t + rules.rework_min)
        # --- recheck starts
        for order in sorted([o for o, s in st.items() if s["stage"] == "recheck_queue" and s["ready"] <= t], key=rank):
            job = by_id[order]
            dur = rules.recheck_min
            chosen = pick_inspection(job, t, dur)
            if chosen is None:
                continue
            board.take(chosen, t, dur)
            s = st[order]
            res[order].waits["recheck"] += max(0, t - s["ready"])
            s.update(stage="recheck", end=t + dur)
        # --- line starts
        for order in sorted([o for o, s in st.items() if s["stage"] in ("line_queue",)], key=rank):
            job = by_id[order]
            if job.material_min > t:
                s = st[order]
                s["stage"] = "line_queue"
                continue
            if not board.free(job.line, t, job.units + rules.asm_fixed_overhead_min):
                continue
            dur = job.units + rules.asm_fixed_overhead_min
            board.take(job.line, t, dur)
            s = st[order]
            res[order].waits["line"] += max(0, t - job.material_min)
            s.update(stage="assembly", end=t + dur, ready=t)

    # horizon end: everything still scheduled after t1 is reported as backlog
    for tt, oid, units in sorted(pending):
        res[oid].deliveries.append((units, tt))
    for order, s in st.items():
        res[order].stage_at_end = s["stage"]
        if s["stage"] != "done":
            res[order].block_reason = STAGE_LABEL.get(s["stage"], s["stage"])

    busy = {r: sum(board.used[r].values()) for r in RESOURCES}
    cap_total = {r: sum(board.caps[r]) for r in RESOURCES}
    note = []
    for name in sorted(config.options):
        opt = pack.options[name]
        overlap = max(0, min(t1, opt.install_end_min) - max(t0, opt.install_start_min))
        if opt.resource in board.caps and opt.delta and overlap:
            note.append(f"{name}：{opt.resource} 在 {opt.install_start_min}–{opt.install_end_min} "
                        f"施工 {overlap} 分钟，期间少 1 个工位")
    return SimResult(orders=res, t0=t0, t1=t1, config=config, world=world,
                     resource_busy=busy, resource_capacity=cap_total,
                     buffer_peak_lots=board.buffer_peak,
                     buffer_peak_min=board.buffer_peak_min,
                     buffer_block_minutes=board.buffer_block_minutes, note=note)


# --------------------------------------------------------------------------- #
# Convenience wrappers
# --------------------------------------------------------------------------- #
def run(pack: Pack, options=(), route_mode="primary", world: World | None = None,
        rules: TimingRules = TimingRules()) -> SimResult:
    world = world or World(label="基准")
    cfg = SimConfig(options=frozenset(options), route_mode=route_mode)
    return simulate(pack, make_jobs(pack, world), cfg, world, rules)
