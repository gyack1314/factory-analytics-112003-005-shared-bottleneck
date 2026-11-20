"""Range analysis: what the pack does *not* support must stay a range.

The pack leaves three things open that decide the retrofit ranking:

* whether (and which) future orders will be rejected at inspection;
* how late material actually arrives relative to the promise date;
* how tightly the order mix compresses into the planning shifts.

Rather than pick one reading and print a false-precision answer, this module
runs every admissible reading and reports, per package, the *range* of
outcomes.  A package whose range does not overlap another's is a robust
ranking; a package whose range overlaps is reported as "资料不支持唯一排序"
together with the readings that flip it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from .compare import compare
from .model import Pack
from .scenario import World


@dataclass
class WorldFamily:
    name: str
    description: str
    worlds: list[World]

    def __len__(self) -> int:
        return len(self.worlds)


def default_family(pack: Pack) -> WorldFamily:
    """The dimensions the pack leaves open, at defensible settings.

    Demand level is included on purpose: the pack's own order book is so much
    smaller than the shift capacity that on-time delivery cannot rank the
    packages.  A reading where the mix is repeated is the only way to find out
    whether an inspection station would ever bind - and the report must say
    which reading produced any claimed benefit.
    """
    plan_ids = [o["order"] for o in pack.planning_orders]
    rework_sets = [(), tuple(plan_ids[:2]), tuple(plan_ids)]
    slips = [0, 300]
    spans = [None, 2]
    multipliers = [1, 12]
    worlds = []
    for rw, slip, span, mult in itertools.product(rework_sets, slips, spans, multipliers):
        label = (f"返流={len(rw)}单;到料+{slip}分钟;"
                 f"{'原下达' if span is None else f'压缩到{span}班'};需求×{mult}")
        worlds.append(World(label=label, rework_orders=rw, material_slip_min=slip,
                            release_span_shifts=span, multiplier=mult))
    return WorldFamily(
        name="default",
        description="返流 {0,2,全部} × 到料推迟 {0,300} × 下达节奏 {原, 压缩2班} × 需求倍数 {1,12}",
        worlds=worlds)


def by_risk_family(pack: Pack) -> WorldFamily:
    """A smaller family focused on the construction-window risk."""
    plan_ids = [o["order"] for o in pack.planning_orders]
    worlds = []
    for rw in [(), tuple(plan_ids[:2])]:
        for slip in [0, 240]:
            for span in [None, 3]:
                worlds.append(World(
                    label=f"返流={len(rw)}单;到料+{slip};"
                          f"{'原下达' if span is None else f'压缩{span}班'}",
                    rework_orders=rw, material_slip_min=slip, release_span_shifts=span))
    return WorldFamily("risk", "返流 {0,2} × 到料 {0,240} × 下达 {原, 压缩3班}", worlds)


@dataclass
class PackageRange:
    options: frozenset[str]
    min_units: int
    max_units: int
    min_lum: int
    max_lum: int
    min_block: int
    max_block: int
    breach_worlds: int
    total_worlds: int
    best_world: str
    worst_world: str

    def as_dict(self) -> dict:
        return {
            "options": sorted(self.options),
            "on_time_units_range": [self.min_units, self.max_units],
            "late_unit_minutes_range": [self.min_lum, self.max_lum],
            "buffer_block_minutes_range": [self.min_block, self.max_block],
            "committed_breach_worlds": f"{self.breach_worlds}/{self.total_worlds}",
            "best_world": self.best_world,
            "worst_world": self.worst_world,
        }


@dataclass
class RobustnessReport:
    family: WorldFamily
    baseline_range: list[int]
    packages: list[PackageRange]
    robust_winner: str | None
    robust_note: str
    overlaps: list[dict]
    best_world_packages: dict[str, int]
    breach_counts: dict[str, str]
    world_count: int

    def as_dict(self) -> dict:
        return {
            "family": self.family.name,
            "worlds": len(self.family),
            "baseline_on_time_range": self.baseline_range,
            "packages": [p.as_dict() for p in self.packages],
            "robust_winner": self.robust_winner,
            "note": self.robust_note,
            "overlapping_packages": self.overlaps,
            "peak_world_on_time_units": self.best_world_packages,
            "committed_breach_worlds": self.breach_counts,
            "world_count": self.world_count,
        }


def analyse(pack: Pack, family: WorldFamily | None = None) -> RobustnessReport:
    family = family or default_family(pack)
    per_pkg: dict[frozenset[str], list[tuple[World, int, int, int, bool]]] = {}
    baseline_units = []
    for w in family.worlds:
        rep = compare(pack, w)
        baseline_units.append(rep.baseline.on_time_units)
        for pkg in rep.packages:
            per_pkg.setdefault(pkg.options, []).append(
                (w, pkg.result.on_time_units, pkg.result.late_unit_minutes,
                 pkg.result.buffer_block_minutes, bool(pkg.committed_breach)))
    packages = []
    for opts, rows in per_pkg.items():
        units = [r[1] for r in rows]
        lums = [r[2] for r in rows]
        blocks = [r[3] for r in rows]
        best = max(rows, key=lambda r: r[1])
        worst = min(rows, key=lambda r: r[1])
        packages.append(PackageRange(
            options=opts, min_units=min(units), max_units=max(units),
            min_lum=min(lums), max_lum=max(lums),
            min_block=min(blocks), max_block=max(blocks),
            breach_worlds=sum(1 for r in rows if r[4]), total_worlds=len(rows),
            best_world=best[0].label, worst_world=worst[0].label))
    # robust winner: a package must beat every other package's *minimum* with its
    # own minimum, i.e. its worst case is still at least as good as their best.
    # We report the weaker but honest condition: no overlap with any other
    # package's range on on-time units.
    by_units = sorted(packages, key=lambda p: (-p.min_units, -p.max_units, p.options))
    overlaps = []
    for a, b in itertools.combinations(packages, 2):
        if max(a.min_units, b.min_units) <= min(a.max_units, b.max_units):
            overlaps.append({"a": sorted(a.options), "b": sorted(b.options),
                             "a_range": [a.min_units, a.max_units],
                             "b_range": [b.min_units, b.max_units]})
    winner = None
    note = ""
    if packages:
        lead = by_units[0]
        contenders = [p for p in packages if p is not lead and p.max_units >= lead.min_units]
        if not contenders:
            winner = "+".join(sorted(lead.options)) or "不改造"
            note = (f"在所有 {len(family.worlds)} 个可行情景中，"
                    f"{winner} 的最差按期件仍不低于其他方案的任何结果，排序稳健。")
        else:
            note = (f"没有任何方案能在全部 {len(family.worlds)} 个可行情景中稳定领先："
                    f"领先方案的最差结果与 {len(contenders)} 个其他方案的最好结果重叠，"
                    "资料不支持唯一排序。请按下方区间和原因决定。")
    peak = {"+".join(sorted(p.options)) or "不改造": p.max_units for p in packages}
    breaches = {"+".join(sorted(p.options)) or "不改造": f"{p.breach_worlds}/{p.total_worlds}"
                for p in packages}
    return RobustnessReport(family=family, baseline_range=[min(baseline_units), max(baseline_units)],
                            packages=by_units, robust_winner=winner, robust_note=note,
                            overlaps=overlaps, best_world_packages=peak,
                            breach_counts=breaches, world_count=len(family.worlds))
