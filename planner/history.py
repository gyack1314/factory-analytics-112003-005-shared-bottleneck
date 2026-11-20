"""Replay of the recorded history with explicit handling of missing queue starts.

The history has three kinds of observability gaps that the analysis must not
paper over:

* first inspections with ``start_min = null`` (22 rows).  The end time is
  recorded and the inspection duration is deterministic, so an *implied*
  start can be derived from ``end - duration``.  That is the latest possible
  start; if the inspector idled the true start is earlier, so every wait
  derived this way is an **upper bound** and the wait lies in ``[0, implied]``.
  Both readings are exposed.
* QA intervals that overlap on the same resource (5 pairs).  These are
  record-consistency violations, not capacity we may spend twice.  Each
  overlapping order is scored twice: once with the overlap treated as real
  delay, once with it ignored.  The pair is the observable [lower, upper]
  bound on that order's lateness and those bounds never enter the exact
  comparison path.
* material arrivals later than ``available_min`` (22 orders).  The pack
  cannot say whether this was a supplier miss or a booking artifact; both
  readings are exposed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .model import Pack, TimingRules, shift_wave


@dataclass
class OrderTrace:
    order: str
    product: str
    line: str
    units: int
    available_min: int
    due_min: int
    priority: bool
    allow_partial: bool
    arrival_min: int
    asm_start: int
    asm_end: int
    qa_start: int | None       # None when the record omits it
    qa_start_implied: int      # latest possible start derived from end - duration
    qa_end: int
    qa_wait_min: int           # implied transfer wait (a lower bound if start is null)
    rework: bool
    deliveries: list[tuple[int, int]]   # (units, time)
    qa_duration: int
    rework_min: int = 0
    recheck_min: int = 0
    recheck_wait_min: int = 0
    slots_reused: int = 0

    @property
    def final_delivery_min(self) -> int:
        return max(t for _, t in self.deliveries)

    @property
    def on_time_units(self) -> int:
        return sum(u for u, t in self.deliveries if t <= self.due_min)

    @property
    def short_units(self) -> int:
        return self.units - self.on_time_units

    @property
    def late_unit_minutes(self) -> int:
        return sum(u * max(0, t - self.due_min) for u, t in self.deliveries)

    @property
    def material_delay_min(self) -> int:
        return max(0, self.arrival_min - self.available_min)

    @property
    def qa_start_observed(self) -> bool:
        return self.qa_start is not None

    @property
    def wait_is_upper_bound(self) -> bool:
        return self.qa_start is None


def iter_traces(pack: Pack, rules: TimingRules = TimingRules()) -> Iterator[OrderTrace]:
    arrivals = {a["order"]: a for a in pack.arrivals}
    orders = pack.order_map()
    by: dict[str, list[dict]] = {}
    for e in pack.history:
        by.setdefault(e["order"], []).append(e)
    for order, events in by.items():
        o = orders[order]
        asm = next(e for e in events if e["event"] == "装配")
        qa_events = [e for e in events if e["event"] == "复验"]
        qa0 = qa_events[0]
        dur = min(rules.qa_max_min, int(qa0["units"] / rules.qa_units_per_min))
        start_implied = qa0["end_min"] - dur
        start = qa0["start_min"]
        wait = (start if start is not None else start_implied) - asm["end_min"]
        rw_events = [e for e in events if e["event"] == "返工"]
        recheck_events = qa_events[1:]
        yield OrderTrace(
            order=order, product=o["product"], line=o["line"], units=o["units"],
            available_min=o["available_min"], due_min=o["due_min"],
            priority=bool(o["priority"]), allow_partial=bool(o["allow_partial"]),
            arrival_min=arrivals[order]["at_min"],
            asm_start=asm["start_min"], asm_end=asm["end_min"],
            qa_start=start, qa_start_implied=start_implied,
            qa_end=qa0["end_min"], qa_wait_min=wait,
            rework=any(e["event"] == "返工" for e in events),
            deliveries=sorted((e["units"], e["end_min"]) for e in events if e["event"] == "成品交付"),
            qa_duration=dur,
            rework_min=sum(e["end_min"] - e["start_min"] for e in rw_events
                           if e["start_min"] is not None),
            recheck_min=sum(e["end_min"] - e["start_min"] for e in recheck_events),
            recheck_wait_min=(recheck_events[0]["start_min"] - rw_events[0]["end_min"]
                              if recheck_events and rw_events
                              and recheck_events[0]["start_min"] is not None else 0),
            slots_reused=len(recheck_events))


def qa_overlap_episodes(pack: Pack, rules: TimingRules = TimingRules()) -> list[dict]:
    """Pairs of first-inspection intervals that cannot both have run on one QA unit."""
    rows = [(t.qa_start_implied, t.qa_end, t) for t in iter_traces(pack, rules)]
    rows.sort(key=lambda r: r[0])
    out = []
    for i, (s1, e1, a) in enumerate(rows):
        for s2, e2, b in rows[i + 1:]:
            if s2 >= e1:
                break
            if a.order == b.order:
                continue
            overlap = min(e1, e2) - max(s1, s2)
            if overlap > 0:
                later, other = (a, b) if a.qa_start_implied >= b.qa_start_implied else (b, a)
                out.append({
                    "pair": sorted([a.order, b.order]),
                    "overlap_min": overlap,
                    "later_order": later.order,
                    "earlier_order": other.order,
                    "later_units": later.units,
                    "later_delivered_min": later.final_delivery_min,
                    "later_due_min": later.due_min,
                    "violates_due_if_serialized": later.final_delivery_min + overlap > later.due_min,
                })
    return out


def replay_check(pack: Pack, rules: TimingRules = TimingRules()) -> dict:
    """Validate the recovered operating rules against every recorded order.

    Four independent checks, all from observed timestamps:

    1. assembly starts at ``max(available, arrival)`` and lasts ``units + 5``;
    2. first inspection lasts ``min(14, 2*units)`` and follows assembly by the
       shift wave (``3 + 4*(shift % 5)``);
    3. rework follows the first inspection by 7 minutes, lasts 18, and the
       4-minute recheck follows the rework by 2;
    4. deliveries occur at ``qa_end + 2`` (clean) or, for rework orders,
       ``qa_end + 2`` for the passed remainder and ``recheck_end + 3`` for
       the rework lot.
    """
    asm_fail, wait_fail, rework_fail, delivery_fail = [], [], [], []
    n = 0
    for t in iter_traces(pack, rules):
        n += 1
        if t.asm_start != max(t.available_min, t.arrival_min):
            asm_fail.append({"order": t.order, "start": t.asm_start,
                             "expected": max(t.available_min, t.arrival_min)})
        if t.asm_end - t.asm_start != t.units + rules.asm_fixed_overhead_min:
            asm_fail.append({"order": t.order, "duration": t.asm_end - t.asm_start,
                             "expected": t.units + rules.asm_fixed_overhead_min})
        wave = shift_wave(t.available_min, pack.shift_minutes, rules)
        if t.qa_wait_min != wave:
            wait_fail.append({"order": t.order, "wait": t.qa_wait_min, "expected": wave,
                              "start_observed": t.qa_start_observed})
        if t.qa_end - t.qa_start_implied != t.qa_duration:
            wait_fail.append({"order": t.order, "qa_duration": t.qa_end - t.qa_start_implied,
                              "expected": t.qa_duration})
        if t.rework:
            events = [e for e in pack.history if e["order"] == t.order]
            rw = next(e for e in events if e["event"] == "返工")
            recheck = [e for e in events if e["event"] == "复验"][1]
            if rw["start_min"] - t.qa_end != rules.qa_to_rw_gap_min:
                rework_fail.append({"order": t.order, "stage": "qa->rw",
                                    "gap": rw["start_min"] - t.qa_end})
            if rw["end_min"] - rw["start_min"] != rules.rework_min:
                rework_fail.append({"order": t.order, "stage": "rework",
                                    "duration": rw["end_min"] - rw["start_min"]})
            if recheck["start_min"] - rw["end_min"] != rules.rw_to_recheck_gap_min:
                rework_fail.append({"order": t.order, "stage": "rw->recheck",
                                    "gap": recheck["start_min"] - rw["end_min"]})
        predicted = _predict_deliveries(t, rules)
        if sorted(predicted) != sorted(t.deliveries):
            delivery_fail.append({"order": t.order, "predicted": predicted,
                                  "observed": sorted(t.deliveries)})
    return {
        "orders": n,
        "assembly_failures": asm_fail,
        "queue_failures": wait_fail,
        "rework_failures": rework_fail,
        "delivery_failures": delivery_fail,
        "match": not (asm_fail or wait_fail or rework_fail or delivery_fail),
    }


def _predict_deliveries(t: OrderTrace, rules: TimingRules) -> list[tuple[int, int]]:
    """Delivery events implied by the observed stage timestamps."""
    qa_end = t.qa_end                      # recorded, authoritative
    if not t.rework:
        return [(t.units, qa_end + rules.pass_to_ship_min)]
    rw_end = qa_end + rules.qa_to_rw_gap_min + rules.rework_min
    recheck_end = rw_end + rules.rw_to_recheck_gap_min + rules.recheck_min
    if not t.allow_partial:
        # the order cannot be split: the whole batch waits for the rework lot.
        return [(t.units, recheck_end + rules.finalized_ship_min)]
    # partial release: passed remainder leaves after the first inspection and
    # the rework lot follows the recheck.
    return [(t.units - rules.rework_lot_units, qa_end + rules.pass_to_ship_min),
            (rules.rework_lot_units, recheck_end + rules.finalized_ship_min)]
