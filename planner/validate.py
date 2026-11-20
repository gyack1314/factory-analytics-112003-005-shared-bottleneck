"""Replay historical shifts through the forward simulator.

The point of this module is falsification, not decoration.  It takes the
recorded orders from a window of shifts, feeds them to the *same* simulator
used for the retrofit comparison, and compares three things the plant cares
about:

* delivered units per order and per shift;
* where the backlog sat at the end of each shift;
* per-resource occupied minutes.

Differences are classified:

``overlap``
    The record let two inspections share one QA unit (5 pairs).  The
    simulator serialises them, so these orders are expected to slip.  They
    are the same orders the attribution excludes from exact comparisons.
``missing_start``
    The record omitted the inspection start; the simulator must derive it
    from the cadence.  If the end time still matches, the derivation is
    consistent (reported as a match, with a flag).
``blocked``
    The simulator stalled a batch because the buffer was full or a line was
    occupied.  A real difference here means the model is missing something.
``unexplained``
    None of the above.  Any of these is a modelling failure and the caller
    should treat the replay as not validated.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import Pack, TimingRules
from .scenario import Job, SimConfig, World, simulate


def history_jobs(pack: Pack, shift_ids: list[int] | None = None) -> list[Job]:
    sm = pack.shift_minutes
    jobs = []
    for o in pack.orders:
        shift = o["available_min"] // sm
        if shift_ids is not None and shift not in shift_ids:
            continue
        arrival = next(a["at_min"] for a in pack.arrivals if a["order"] == o["order"])
        jobs.append(Job(
            order=o["order"], product=o["product"], line=o["line"], units=int(o["units"]),
            available_min=int(o["available_min"]), due_min=int(o["due_min"]),
            priority=bool(o["priority"]), allow_partial=bool(o["allow_partial"]),
            material_min=int(arrival),
            rework=any(e["order"] == o["order"] and e["event"] == "返工" for e in pack.history)))
    return jobs


@dataclass
class ReplayDiff:
    order: str
    kind: str          # match | missing_start | overlap | blocked | unexplained
    expected: list[tuple[int, int]]
    actual: list[tuple[int, int]]
    detail: str

    def as_dict(self):
        return {"order": self.order, "kind": self.kind,
                "expected": self.expected, "actual": self.actual, "detail": self.detail}


@dataclass
class ReplayReport:
    shifts: list[int]
    orders: int
    delivered_units_expected: int
    delivered_units_actual: int
    exact_orders: int
    diffs: list[ReplayDiff]
    kind_counts: dict[str, int]
    backlog_expected: dict[str, str]
    backlog_actual: dict[str, str]
    resource_minutes_expected: dict[str, int]
    resource_minutes_actual: dict[str, int]
    t0: int
    t1: int

    @property
    def validated(self) -> bool:
        return not any(d.kind == "unexplained" for d in self.diffs)

    def as_dict(self) -> dict:
        return {
            "shifts": self.shifts,
            "orders": self.orders,
            "delivered_units_expected": self.delivered_units_expected,
            "delivered_units_actual": self.delivered_units_actual,
            "exact_orders": self.exact_orders,
            "kind_counts": self.kind_counts,
            "validated": self.validated,
            "diffs": [d.as_dict() for d in self.diffs],
            "backlog_expected": self.backlog_expected,
            "backlog_actual": self.backlog_actual,
            "resource_minutes_expected": self.resource_minutes_expected,
            "resource_minutes_actual": self.resource_minutes_actual,
            "window": [self.t0, self.t1],
        }


def replay(pack: Pack, shift_ids: list[int], rules: TimingRules = TimingRules()) -> ReplayReport:
    t0 = min(shift_ids) * pack.shift_minutes
    t1 = (max(shift_ids) + 1) * pack.shift_minutes
    jobs = history_jobs(pack, shift_ids)
    ids = {j.order for j in jobs}
    world = World(label="历史重放")
    cfg = SimConfig(route_mode="primary")
    result = simulate(pack, jobs, cfg, world, rules, t0=t0, t1=t1)

    # Every QA interval in the window, including rechecks and intervals whose
    # start was omitted (implied from end - duration).  Any pair that overlaps
    # is a record inconsistency the serial replay cannot honor.
    intervals = []
    for e in pack.history:
        if e["event"] != "复验" or e["order"] not in ids:
            continue
        dur = (min(rules.qa_max_min, int(e["units"] / rules.qa_units_per_min))
               if e["units"] > rules.rework_lot_units else rules.recheck_min)
        start = e["start_min"] if e["start_min"] is not None else e["end_min"] - dur
        intervals.append((start, e["end_min"], e["order"]))
    intervals.sort()
    overlaps = set()
    missing = set()
    for order in ids:
        qa0 = next(e for e in pack.history if e["order"] == order and e["event"] == "复验")
        if qa0["start_min"] is None:
            missing.add(order)
    for i, (s1, e1, o1) in enumerate(intervals):
        for s2, e2, o2 in intervals[i + 1:]:
            if s2 >= e1:
                break
            if o1 != o2 and s1 < e2 and s2 < e1:
                overlaps.add(o1)
                overlaps.add(o2)

    diffs = []
    delivered_expected = delivered_actual = 0
    exact = 0
    for order in sorted(ids):
        expected = sorted((e["units"], e["end_min"]) for e in pack.history
                          if e["order"] == order and e["event"] == "成品交付")
        actual = sorted(result.orders[order].deliveries)
        delivered_expected += sum(u for u, _ in expected)
        delivered_actual += sum(u for u, _ in actual)
        if expected == actual:
            exact += 1
            diffs.append(ReplayDiff(order, "missing_start" if order in missing else "match",
                                    expected, actual,
                                    "start_min 缺失，按节拍反推后交付时刻吻合" if order in missing else ""))
            continue
        if order in overlaps:
            kind = "overlap"
            detail = "记录中与同资源复验重叠；重放按单台 QA 串行，交付推迟"
        elif result.orders[order].stage_at_end not in ("done",):
            kind = "blocked"
            detail = f"期末仍在 {result.orders[order].block_reason}"
        else:
            kind = "unexplained"
            detail = "交付时刻与记录不符，且不属于已知记录缺陷"
        diffs.append(ReplayDiff(order, kind, expected, actual, detail))

    by_shift_expected: dict[str, str] = {}
    by_shift_actual: dict[str, str] = {}
    for sid in shift_ids:
        lo, hi = sid * pack.shift_minutes, (sid + 1) * pack.shift_minutes
        exp_not_done = [o for o in ids
                        if max([e["end_min"] for e in pack.history
                                if e["order"] == o and e["event"] == "成品交付"], default=0) >= hi]
        act_not_done = [o for o in ids if (result.orders[o].final_delivery_min or 10 ** 9) >= hi]
        by_shift_expected[str(sid)] = ",".join(sorted(exp_not_done)) or "-"
        by_shift_actual[str(sid)] = ",".join(sorted(act_not_done)) or "-"

    def minutes(events, res):
        # records with a missing start still consume a known duration
        total = 0
        for e in events:
            if e["resource"] != res:
                continue
            dur = (rules.recheck_min if e["units"] <= rules.rework_lot_units
                   else min(rules.qa_max_min, int(e["units"] / rules.qa_units_per_min)))
            start = e["start_min"] if e["start_min"] is not None else e["end_min"] - dur
            total += e["end_min"] - start
        return total

    hist = [e for e in pack.history if t0 <= (e["start_min"] if e["start_min"] is not None
                                             else e["end_min"]) < t1]
    res_expected = {r: minutes(hist, r) for r in ("L1", "L2", "QA", "RW")}
    res_actual = {r: result.resource_busy.get(r, 0) for r in ("L1", "L2", "QA", "RW")}
    kinds: dict[str, int] = {}
    for d in diffs:
        kinds[d.kind] = kinds.get(d.kind, 0) + 1
    return ReplayReport(
        shifts=list(shift_ids), orders=len(ids),
        delivered_units_expected=delivered_expected, delivered_units_actual=delivered_actual,
        exact_orders=exact, diffs=diffs, kind_counts=kinds,
        backlog_expected=by_shift_expected, backlog_actual=by_shift_actual,
        resource_minutes_expected=res_expected, resource_minutes_actual=res_actual,
        t0=t0, t1=t1)
