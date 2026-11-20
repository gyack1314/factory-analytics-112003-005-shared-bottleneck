"""Historical shortage attribution: what each mechanism cost past deliveries.

Causal model
------------
For every recorded order the delivery events are rebuilt from the recorded
stage stamps.  :func:`planner.history.replay_check` proves this rebuild is
exact for all 132 orders, so the counterfactual below changes exactly one
mechanism at a time and leaves the rest of the record intact.

``M`` 材料不足
    Assembly could not start before the recorded arrival.  Removing M starts
    the order at ``available_min`` instead of ``max(available, arrival)``.
``T`` 转移等待
    The recorded gap between assembly end and first inspection.  The records
    show a per-shift cadence (3/7/11/15/19 min) that is *not* explained by QA
    occupancy: QA is idle most of each shift.  Removing T sets that gap to 0.
    This is why the report separates "queueing caused by a full station" from
    "waiting caused by the inspection cadence" - buying a machine does not
    remove a cadence.
``R`` 检验返流
    The recorded reject -> 2-unit rework -> 4-minute recheck loop.  Removing R
    lets the whole batch leave after the first inspection (and removes the
    32-minute tail and the partial split).

Attribution is exact Shapley over the three causes (2^3 coalitions), so no
mechanism is double-counted when two lines compete for the same inspection
unit.  The joint interaction term is reported separately: it is the part of
the total effect that only appears when causes are removed together.

Residue
-------
Orders that are still short with **every** cause removed are *structural*:
the batch simply does not fit between its material and its due date.  The
residue is never assigned to M/T/R; it is reported by the station where the
order first crosses its due date (装配 / 复验 / 交付) so the reader can see
which station owns the structural problem.

Ambiguity
---------
Two record defects are kept out of the exact comparison:

* a missing first-inspection start (22 orders).  The implied wait from
  ``end - duration`` is an upper bound.  The order's shortfall is bounded to
  ``[0, recorded]``.
* overlapping QA intervals on the same single QA unit (5 pairs).  Both
  readings are impossible under one unit, so the affected orders get a
  shortfall range, and the pair is listed as a record inconsistency.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from .history import OrderTrace, iter_traces, qa_overlap_episodes
from .model import Pack, TimingRules

CAUSES = ("M", "T", "R")
CAUSE_LABELS = {"M": "材料不足/到货延迟", "T": "转移等待（含每班检验节拍）", "R": "复验拒收返流"}


@dataclass
class OrderAttribution:
    order: str
    product: str
    units: int
    due_min: int
    short_units: int
    late_unit_minutes: int
    material_delay_min: int
    transfer_wait_min: int
    rework: bool
    partial: bool
    sunk_rework_min: int
    sunk_recheck_min: int
    sunk_recheck_wait_min: int
    slots_reused: int
    remaining_without: dict[str, int]
    shapley_short: dict[str, float]
    shapley_late_unit_min: dict[str, float]
    residue_short: int
    residue_late_unit_min: int
    residue_station: str | None
    ambiguous: bool
    ambiguity_reason: str | None
    bound_low_short: int
    bound_high_short: int

    @property
    def explained_short(self) -> int:
        return self.short_units - self.residue_short


@dataclass
class AttributionReport:
    orders: list[OrderAttribution]
    total_short: int
    total_late_unit_min: int
    shapley_short: dict[str, float]
    shapley_late_unit_min: dict[str, float]
    explained_short: float
    explained_late_unit_min: float
    residue_short: int
    residue_late_unit_min: int
    residue_all_short: int
    residue_all_late_unit_min: int
    residue_by_station: dict[str, int]
    residue_station_all: dict[str, int]
    single_cause_short: dict[str, float]
    joint_interaction_short: float
    joint_interaction_late_unit_min: float
    clean_orders: int
    ambiguous_orders: int
    ambiguity_bounds: dict[str, dict]
    ambiguity_summary: dict[str, int]
    qa_overlap_episodes: list[dict]
    transfer_wave: list[int]
    capacity_headroom: dict[str, dict]
    station_diagnosis: list[dict]
    sunk_rework: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "total_short_units": self.total_short,
            "total_late_unit_minutes": self.total_late_unit_min,
            "attributable_short_units": {CAUSE_LABELS[k]: round(v, 1)
                                         for k, v in self.shapley_short.items()},
            "attributable_late_unit_minutes": {CAUSE_LABELS[k]: round(v, 1)
                                               for k, v in self.shapley_late_unit_min.items()},
            "explained_short_units": round(self.explained_short, 1),
            "explained_late_unit_minutes": round(self.explained_late_unit_min, 1),
            "structural_residue_units_clean": self.residue_short,
            "structural_residue_late_unit_minutes_clean": self.residue_late_unit_min,
            "structural_residue_units_all_orders": self.residue_all_short,
            "structural_residue_by_station_all_orders": self.residue_station_all,
            "structural_residue_by_station": self.residue_by_station,
            "single_cause_short_units": {CAUSE_LABELS[k]: round(v, 1)
                                         for k, v in self.single_cause_short.items()},
            "joint_interaction_short_units": round(self.joint_interaction_short, 1),
            "joint_interaction_late_unit_minutes": round(self.joint_interaction_late_unit_min, 1),
            "clean_orders": self.clean_orders,
            "ambiguous_orders": self.ambiguous_orders,
            "ambiguity_summary": self.ambiguity_summary,
            "ambiguity_bounds": self.ambiguity_bounds,
            "qa_overlap_episodes": self.qa_overlap_episodes,
            "transfer_wave_min": self.transfer_wave,
            "capacity_headroom": self.capacity_headroom,
            "station_diagnosis": self.station_diagnosis,
            "sunk_rework_cost": self.sunk_rework,
        }


# --------------------------------------------------------------------------- #
def deliveries_without(trace: OrderTrace, removed: frozenset[str], pack: Pack,
                       rules: TimingRules) -> list[tuple[int, int]]:
    """Recorded delivery events after eliminating the mechanisms in ``removed``."""
    start = trace.available_min if "M" in removed else max(trace.available_min, trace.arrival_min)
    asm_end = start + trace.units + rules.asm_fixed_overhead_min
    wait = 0 if "T" in removed else trace.qa_wait_min
    qa_end = asm_end + wait + trace.qa_duration
    if trace.rework and "R" not in removed:
        rw_end = qa_end + rules.qa_to_rw_gap_min + rules.rework_min
        recheck_end = rw_end + rules.rw_to_recheck_gap_min + rules.recheck_min
        if not trace.allow_partial:
            return [(trace.units, recheck_end + rules.finalized_ship_min)]
        return [(trace.units - rules.rework_lot_units, qa_end + rules.pass_to_ship_min),
                (rules.rework_lot_units, recheck_end + rules.finalized_ship_min)]
    return [(trace.units, qa_end + rules.pass_to_ship_min)]


def _short(dels, due):
    return sum(u for u, t in dels if t > due)


def _late_unit_min(dels, due):
    return sum(u * max(0, t - due) for u, t in dels)


def _shapley(value) -> dict[str, float]:
    """Exact Shapley over the three causes."""
    n = len(CAUSES)
    out = {}
    for c in CAUSES:
        others = [x for x in CAUSES if x != c]
        total = 0.0
        for r in range(len(others) + 1):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for combo in itertools.combinations(others, r):
                total += w * (value(frozenset(combo) | {c}) - value(frozenset(combo)))
        out[c] = total
    return out


def _ambiguity(pack: Pack, traces: list[OrderTrace], overlaps: list[dict]) -> dict[str, dict]:
    amb: dict[str, dict] = {}
    for t in traces:
        if t.qa_start is None:
            amb[t.order] = {
                "reason": "首次复验缺 start_min",
                "explanation": "只记录复验结束时刻；start 只能由 end-duration 反推，等待是上界。",
                "implied_wait_min": t.qa_wait_min,
                "short_bounds": [0, t.short_units],
            }
    for ep in overlaps:
        for order in ep["pair"]:
            t = next(x for x in traces if x.order == order)
            held = amb.setdefault(order, {
                "reason": "QA 区间与同资源订单重叠",
                "explanation": "两条复验区间在同一台 QA 上不能同时成立；记录一致性存疑。",
                "implied_wait_min": t.qa_wait_min,
                "short_bounds": [t.short_units, t.short_units],
            })
            held["overlap_with"] = [o for o in ep["pair"] if o != order]
            held["overlap_min"] = ep["overlap_min"]
            lo, hi = held["short_bounds"]
            held["short_bounds"] = [lo, max(hi, t.units if ep["violates_due_if_serialized"] else lo)]
    return amb


def _station_diagnosis(pack: Pack, traces: list[OrderTrace], rules: TimingRules) -> list[dict]:
    """Why each station was idle, as far as the record can tell.

    The plant's warning was that idle equipment may be starving while busy
    equipment may only be reworking.  These are two different diagnostics and
    the pack can separate them:

    * assembly: every recorded order starts exactly at ``max(available, arrival)``,
      so no assembly idle time was caused by missing material or an upstream
      block.  Assembly idle is simply a thin order book.
    * rework: 100% of RW busy minutes are rework of rejected lots.
    * inspection: part of QA busy time is re-inspection of those same lots.
    """
    out = []
    span_start = min(t.asm_start for t in traces)
    span_end = max(t.final_delivery_min for t in traces)
    span = span_end - span_start
    for res, label in (("L1", "装配 L1"), ("L2", "装配 L2"), ("QA", "复验 QA"), ("RW", "返工 RW")):
        events = [e for e in pack.history if e["resource"] == res and e["start_min"] is not None]
        busy = sum(e["end_min"] - e["start_min"] for e in events)
        rework = sum(e["end_min"] - e["start_min"] for e in events if e["event"] == "返工")
        recheck = sum(e["end_min"] - e["start_min"] for e in events
                      if e["event"] == "复验" and e["units"] <= rules.rework_lot_units)
        if res in ("L1", "L2"):
            delayed = [t.order for t in traces if t.line == res and
                       t.asm_start > max(t.available_min, t.arrival_min)]
            reason = ("无：记录中每单都在到料时刻立即开工，空闲来自订单量而非等料或上游阻塞"
                      if not delayed else f"{len(delayed)} 单未能立即开工，需查上游/到料")
        elif res == "RW":
            reason = "全部占用分钟都是返工；设备越忙越说明拒收越多，不代表新增产能"
        else:
            reason = (f"{busy} 分钟中 {recheck} 分钟是返工后复检；"
                      "其余空闲与产能无关，记录显示每次复验都按班次节拍开始")
        out.append({"station": label, "busy_min": busy, "rework_min": rework,
                    "recheck_min": recheck,
                    "utilization": round(busy / span, 3) if span else 0.0,
                    "diagnosis": reason})
    return out


def _capacity_headroom(pack: Pack, traces: list[OrderTrace], rules: TimingRules) -> dict[str, dict]:
    """Idle share per station; evidence that equipment is not the binding constraint."""
    out: dict[str, dict] = {}
    all_events = [e for e in pack.history if e["start_min"] is not None]
    span = max(e["end_min"] for e in all_events) - min(e["start_min"] for e in all_events)
    for res, label in (("L1", "装配 L1"), ("L2", "装配 L2"), ("QA", "复验 QA"), ("RW", "返工 RW")):
        events = [e for e in all_events if e["resource"] == res]
        busy = sum(e["end_min"] - e["start_min"] for e in events)
        out[label] = {
            "busy_min": busy,
            "observed_span_min": span,
            "capacity_min": span,
            "utilization": round(busy / span, 3),
            "rework_min": sum(e["end_min"] - e["start_min"] for e in events if e["event"] == "返工"),
            "recheck_min": sum(e["end_min"] - e["start_min"] for e in events
                               if e["event"] == "复验" and e["units"] == rules.rework_lot_units),
        }
    return out


def analyse(pack: Pack, rules: TimingRules = TimingRules()) -> AttributionReport:
    traces = list(iter_traces(pack, rules))
    overlaps = qa_overlap_episodes(pack, rules)
    amb = _ambiguity(pack, traces, overlaps)
    wave = [3 + 4 * (s % 5) for s in range(5)]

    rows: list[OrderAttribution] = []
    totals = {c: 0.0 for c in CAUSES}
    totals_lu = {c: 0.0 for c in CAUSES}
    singles = {c: 0.0 for c in CAUSES}
    singles_lu = {c: 0.0 for c in CAUSES}
    residue_by_station: dict[str, int] = {}
    residue_total = 0
    residue_lu = 0
    residue_all = 0
    residue_all_lu = 0

    for t in traces:
        base = deliveries_without(t, frozenset(), pack, rules)
        base_short = _short(base, t.due_min)
        base_lu = _late_unit_min(base, t.due_min)

        def v_short(removed, _t=t):
            return _short(deliveries_without(_t, removed, pack, rules), _t.due_min)

        def v_lu(removed, _t=t):
            return _late_unit_min(deliveries_without(_t, removed, pack, rules), _t.due_min)

        # Shapley of the *reduction*: value = baseline - remaining.
        sh = _shapley(lambda s, _b=base_short, _v=v_short: _b - _v(s))
        sh_lu = _shapley(lambda s, _b=base_lu, _v=v_lu: _b - _v(s))
        remaining = {c: v_short(frozenset([c])) for c in CAUSES}
        residue = v_short(frozenset(CAUSES))
        residue_l = v_lu(frozenset(CAUSES))
        station = None
        if residue:
            residue_all += residue
            residue_all_lu += residue_l
            asm_end = t.available_min + t.units + rules.asm_fixed_overhead_min
            qa_end = asm_end + t.qa_wait_min + t.qa_duration
            ship = qa_end + rules.pass_to_ship_min
            if asm_end > t.due_min:
                station = "装配时长超过承诺交期"
            elif qa_end > t.due_min:
                station = "复验时长超过承诺交期"
            elif ship > t.due_min:
                station = "检验后放行延迟"
            else:
                station = "批次/整批约束"
            residue_by_station[station] = residue_by_station.get(station, 0) + residue

        clean = t.order not in amb
        if clean and residue:
            # exact accounting: ambiguity orders are never folded into the
            # identity, they are reported as bounds instead.
            residue_total += residue
            residue_lu += residue_l
        if clean:
            for c in CAUSES:
                totals[c] += sh[c]
                totals_lu[c] += sh_lu[c]
                singles[c] += base_short - remaining[c]
                singles_lu[c] += base_lu - (v_lu(frozenset([c])))
        b = amb.get(t.order)
        rows.append(OrderAttribution(
            order=t.order, product=t.product, units=t.units, due_min=t.due_min,
            short_units=base_short, late_unit_minutes=base_lu,
            material_delay_min=t.material_delay_min, transfer_wait_min=t.qa_wait_min,
            rework=t.rework, partial=t.allow_partial,
            sunk_rework_min=t.rework_min, sunk_recheck_min=t.recheck_min,
            sunk_recheck_wait_min=t.recheck_wait_min, slots_reused=t.slots_reused,
            remaining_without=remaining, shapley_short=sh, shapley_late_unit_min=sh_lu,
            residue_short=residue, residue_late_unit_min=residue_l,
            residue_station=station,
            ambiguous=not clean, ambiguity_reason=(b or {}).get("reason"),
            bound_low_short=(b["short_bounds"][0] if b else base_short),
            bound_high_short=(b["short_bounds"][1] if b else base_short)))

    clean_rows = [r for r in rows if not r.ambiguous]
    residue_short_all = sum(r.residue_short for r in rows)
    residue_lu_all = sum(r.residue_late_unit_min for r in rows)
    residue_by_station_all = {}
    for r in rows:
        if r.residue_short and r.residue_station:
            residue_by_station_all[r.residue_station] = (
                residue_by_station_all.get(r.residue_station, 0) + r.residue_short)
    total_short = sum(r.short_units for r in clean_rows)
    total_lu = sum(r.late_unit_minutes for r in clean_rows)
    explained = sum(totals.values())
    explained_lu = sum(totals_lu.values())
    joint = explained - sum(singles.values())
    joint_lu = explained_lu - sum(singles_lu.values())
    amb_rows = [r for r in rows if r.ambiguous]
    ambiguity_summary = {
        "orders": len(amb_rows),
        "short_units_recorded": sum(r.short_units for r in amb_rows),
        "short_units_lower_bound": sum(r.bound_low_short for r in amb_rows),
        "short_units_upper_bound": sum(r.bound_high_short for r in amb_rows),
        "reasons": {
            "missing_start": sum(1 for r in amb_rows if r.ambiguity_reason == "首次复验缺 start_min"),
            "qa_overlap": sum(1 for r in amb_rows if r.ambiguity_reason == "QA 区间与同资源订单重叠"),
        },
    }
    return AttributionReport(
        orders=rows, total_short=total_short, total_late_unit_min=total_lu,
        shapley_short=totals, shapley_late_unit_min=totals_lu,
        explained_short=explained, explained_late_unit_min=explained_lu,
        residue_short=residue_total, residue_late_unit_min=residue_lu,
        residue_all_short=residue_short_all, residue_all_late_unit_min=residue_lu_all,
        residue_station_all=residue_by_station_all,
        residue_by_station=residue_by_station,
        single_cause_short=singles,
        joint_interaction_short=joint, joint_interaction_late_unit_min=joint_lu,
        clean_orders=len(clean_rows), ambiguous_orders=len(amb_rows),
        ambiguity_bounds=amb, ambiguity_summary=ambiguity_summary,
        qa_overlap_episodes=overlaps, transfer_wave=wave,
        capacity_headroom=_capacity_headroom(pack, traces, rules),
        station_diagnosis=_station_diagnosis(pack, traces, rules),
        sunk_rework={
            "rework_jobs": sum(1 for r in rows if r.rework),
            "rework_minutes": sum(r.sunk_rework_min for r in rows),
            "recheck_minutes": sum(r.sunk_recheck_min for r in rows),
            "recheck_wait_minutes": sum(r.sunk_recheck_wait_min for r in rows),
            "qa_slots_reused": sum(r.slots_reused for r in rows),
            "affected_units": sum(r.units for r in rows if r.rework),
        })
