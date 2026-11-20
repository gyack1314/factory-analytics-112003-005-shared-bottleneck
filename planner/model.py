"""Typed view over the data pack plus the operating rules recovered from history.

Every rule in :func:`calibrate` was inferred from ``history.json`` and is
re-validated by ``tests/test_history_replay.py``.  If the shop changes a rule,
the test fails and the constants below must be re-derived instead of silently
drifting.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import dataio


@dataclass(frozen=True)
class ProductRule:
    product: str
    assembly: tuple[str, ...]
    inspection: str
    alternative: str | None
    max_rework: int
    allowed_lot_units: int


@dataclass(frozen=True)
class Option:
    option: str
    resource: str
    delta: int
    install_start_min: int
    install_end_min: int
    effective_min: int
    cost_cents: int
    exclusive_group: str | None


@dataclass(frozen=True)
class TimingRules:
    """Stage-timing constants recovered from the recorded history."""

    asm_fixed_overhead_min: int = 5          # assembly = units + 5
    qa_units_per_min: float = 0.5            # first inspection = min(14, 2 * units)
    qa_max_min: int = 14
    recheck_min: int = 4                     # re-inspection of the 2-unit rework lot
    rework_lot_units: int = 2
    rework_min: int = 18
    qa_to_rw_gap_min: int = 7
    rw_to_recheck_gap_min: int = 2
    pass_to_ship_min: int = 2
    finalized_ship_min: int = 3
    buffer_lot_min: int = 3                  # baseline buffer holds 3 lots
    wave_base_min: int = 3                   # transfer wait = 3 + 4*(shift % 5)
    wave_step_min: int = 4
    wave_period_shifts: int = 5


@dataclass
class Pack:
    orders: list[dict]
    history: list[dict]
    arrivals: list[dict]
    rosters: list[dict]
    route_rules: dict[str, ProductRule]
    options: dict[str, Option]
    planning_orders: list[dict]
    wip: list[dict]
    wip_commitments: dict[str, dict]
    approvals: list[dict]
    limits: dict

    @property
    def baseline_capacity(self) -> dict[str, int]:
        return dict(self.limits["baseline_capacity"])

    @property
    def horizon(self) -> tuple[int, int]:
        shifts = self.limits["planning_shifts"]
        return shifts[0][0], shifts[-1][1]

    @property
    def shift_minutes(self) -> int:
        return self.limits["shift_minutes"]

    @property
    def budget_cents(self) -> int:
        return self.limits["investment_budget_cents"]

    def order_map(self) -> dict[str, dict]:
        return {o["order"]: o for o in self.orders}

    def commitment_map(self) -> dict[str, dict]:
        """Due/priority view across already-committed history and open WIP."""
        m = {o["order"]: o for o in self.orders}
        m.update({k: dict(v, product=next(w["product"] for w in self.wip if w["order"] == k))
                  for k, v in self.wip_commitments.items()})
        return m


def build_pack(root=None) -> Pack:
    raw = dataio.load_all(root)
    route = {r["product"]: ProductRule(
        product=r["product"], assembly=tuple(r["assembly"]),
        inspection=r["inspection"], alternative=r["alternative"],
        max_rework=int(r["max_rework"]), allowed_lot_units=int(r["allowed_lot_units"]))
        for r in raw["route_rules"]}
    opts = {o["option"]: Option(**o) for o in raw["options"]}
    return Pack(
        orders=raw["orders"], history=raw["history"], arrivals=raw["arrivals"],
        rosters=raw["rosters"], route_rules=route, options=opts,
        planning_orders=raw["planning_orders"], wip=raw["wip"],
        wip_commitments={c["order"]: c for c in raw["wip_commitments"]},
        approvals=raw["approvals"], limits=raw["limits"])


def stage_durations(units: int, rules: TimingRules = TimingRules()) -> dict[str, int]:
    """Nominal duration of each stage for a batch of ``units``."""
    return {
        "assembly": units + rules.asm_fixed_overhead_min,
        "first_inspection": min(rules.qa_max_min, int(units / rules.qa_units_per_min)),
        "rework": rules.rework_min,
        "recheck": rules.recheck_min,
    }


def lot_count(units: int, product: str, pack: Pack) -> int:
    """Buffer occupancy is counted in allowed-lot units, not pieces."""
    per = pack.route_rules[product].allowed_lot_units
    return -(-units // per)


def shift_wave(available_min: int, shift_minutes: int = 480,
               rules: TimingRules = TimingRules()) -> int:
    shift = available_min // shift_minutes
    return rules.wave_base_min + rules.wave_step_min * (shift % rules.wave_period_shifts)


def direct_stage_time(units: int, product: str, available_min: int, shift_minutes: int = 480,
                      rules: TimingRules = TimingRules()) -> dict[str, int]:
    """Clean-path completion times with no material delay, no queue and no rework."""
    asm_end = available_min + units + rules.asm_fixed_overhead_min
    qa_s = asm_end + shift_wave(available_min, shift_minutes, rules)
    qa_e = qa_s + min(rules.qa_max_min, int(units / rules.qa_units_per_min))
    return {"asm_end": asm_end, "qa_start": qa_s, "qa_end": qa_e,
            "ship": qa_e + rules.pass_to_ship_min}
