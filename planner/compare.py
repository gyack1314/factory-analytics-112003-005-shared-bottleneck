"""Scenario comparison, phased rollout and audit of every claimed benefit.

Three questions the retrofit owner asked, answered with the same simulator:

1. **Which package wins on today's order book?**  Every admissible option set
   (respecting the budget and the mutually-exclusive inspection group) is run
   on the *same* jobs and the *same* world, so differences are caused only by
   the package.
2. **Which orders does a package actually unblock?**  For each package we diff
   per-order on-time units against the baseline.  The diff is the audit trail:
   a package's benefit is the sum of the orders it moved, never an aggregate
   estimate.
3. **Which benefits only appear in combination?**  Shapley over the *options*
   (not the causes) gives each option's marginal contribution across every
   package it appears in; the interaction term names the orders that are only
   saved by a particular combination.

Phase logic
-----------
Phase 1 runs the full horizon.  Once management confirms a phase-1 package,
phase 2 starts from the *simulated* phase-1 end state: orders that phase 1
would have shipped are removed, orders still in a stage enter phase 2 at that
stage, and phase-2 capacity includes the phase-1 option as an already-installed
asset.  The phase-1 assumptions (world, selected package, decision minute) are
stored with the run so a later reader can see which demand and constraints the
abandoned alternatives were evaluated against.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from .model import Pack, TimingRules
from .scenario import (Job, SimConfig, SimResult, World, feasible_option_sets,
                       make_jobs, simulate)


# --------------------------------------------------------------------------- #
# Package comparison
# --------------------------------------------------------------------------- #
@dataclass
class PackageResult:
    options: frozenset[str]
    cost_cents: int
    baseline: SimResult
    result: SimResult
    unblocked: dict[str, dict]          # order -> before/after
    blocked_by_other: dict[str, str]    # orders that stay late, and why
    gain_on_time_units: int
    gain_late_unit_minutes: int
    new_backlog_units: int
    committed_ok: bool
    note: list[str] = field(default_factory=list)

    # --- risk axes that differ even when on-time delivery does not
    @property
    def committed_breach(self) -> list[str]:
        return self.result.committed_breach

    @property
    def buffer_block_minutes(self) -> int:
        return self.result.buffer_block_minutes

    @property
    def backlog_units(self) -> int:
        return self.result.backlog_units

    @property
    def violates_commitments(self) -> bool:
        """Committed priority orders and WIP must keep their promises.

        The plant treats this as a hard constraint, not a tradeable metric, so
        a package that breaks it is reported as unacceptable even if its
        on-time total looks good.
        """
        return bool(self.result.committed_breach)

    @property
    def construction_loss_min(self) -> int:
        """Minutes of capacity removed by construction inside the horizon."""
        total = 0
        for opt in self.options:
            total += self.result.resource_capacity.get("_install_" + opt, 0)
        return total

    def as_dict(self) -> dict:
        return {
            "options": sorted(self.options),
            "cost_cents": self.cost_cents,
            "cost_yuan": round(self.cost_cents / 100),
            "summary": self.result.summary(),
            "gain_on_time_units": self.gain_on_time_units,
            "gain_late_unit_minutes": self.gain_late_unit_minutes,
            "risk": {
                "violates_commitments": self.violates_commitments,
                "committed_breach": self.committed_breach,
                "buffer_block_minutes": self.buffer_block_minutes,
                "backlog_units": self.backlog_units,
            },
            "unblocked_orders": {k: v for k, v in self.unblocked.items()},
            "still_blocked": self.blocked_by_other,
            "note": self.note,
        }


def _explain_block(pack: Pack, res: SimResult, order: str, world: World) -> str:
    r = res.orders[order]
    if r.stage_at_end != "done":
        return f"期末仍在「{r.block_reason}」"
    waits = r.waits
    worst = max(waits, key=waits.get)
    return f"已交付但超出交期 {r.late_unit_minutes} 件·分钟；最长等待在 {worst}（{waits[worst]} 分钟）"


def _world_jobs(pack: Pack, world: World) -> list[Job]:
    return make_jobs(pack, world)


def evaluate_package(pack: Pack, options, world: World, baseline: SimResult,
                     baseline_jobs: list[Job], route_mode: str = "primary",
                     rules: TimingRules = TimingRules()) -> PackageResult:
    cfg = SimConfig(options=frozenset(options), route_mode=route_mode)
    res = simulate(pack, baseline_jobs, cfg, world, rules)
    unblocked = {}
    blocked = {}
    gain_units = 0
    gain_lum = 0
    for order, before in baseline.orders.items():
        after = res.orders[order]
        d_units = after.on_time_units - before.on_time_units
        d_lum = before.late_unit_minutes - after.late_unit_minutes
        if d_units > 0 or d_lum > 0:
            unblocked[order] = {
                "on_time_units_before": before.on_time_units,
                "on_time_units_after": after.on_time_units,
                "delta_units": d_units,
                "late_unit_minutes_before": before.late_unit_minutes,
                "late_unit_minutes_after": after.late_unit_minutes,
                "due_min": before.due_min,
                "priority": before.priority,
                "committed": before.committed,
                "why": _why_moved(before, after),
            }
            gain_units += d_units
            gain_lum += d_lum
        elif after.late_units > 0:
            reason = _explain_still_blocked(res, order, baseline)
            if after.late_units > before.late_units:
                reason = "本方案使该单更差：" + reason
            blocked[order] = reason
    return PackageResult(
        options=frozenset(options), cost_cents=sum(pack.options[o].cost_cents for o in options),
        baseline=baseline, result=res, unblocked=unblocked, blocked_by_other=blocked,
        gain_on_time_units=gain_units, gain_late_unit_minutes=gain_lum,
        new_backlog_units=res.backlog_units - baseline.backlog_units,
        committed_ok=not res.committed_breach, note=res.note)


def _why_moved(before, after) -> str:
    parts = []
    for key, label in (("line", "装配排队"), ("buffer", "缓冲区等待"),
                       ("inspection", "复验排队"), ("rework", "返工等待"),
                       ("recheck", "复检等待")):
        delta = before.waits.get(key, 0) - after.waits.get(key, 0)
        if delta > 0:
            parts.append(f"{label}少 {delta} 分钟")
    if before.late_units and not after.late_units:
        parts.append("从迟交变为按期")
    if before.stage_at_end != "done" and after.stage_at_end == "done":
        parts.append("从期末积压变为交付")
    if not parts and before.final_delivery_min and after.final_delivery_min:
        earlier = before.final_delivery_min - after.final_delivery_min
        if earlier > 0:
            parts.append(f"末次交付提前 {earlier} 分钟")
        elif earlier < 0:
            parts.append(f"末次交付推迟 {-earlier} 分钟（其他订单占用释放出的工位）")
    return "；".join(parts) or "交付时刻不变，仅迟交分钟改善"


_STAGE_NAMES = {
    "line": "装配排队", "buffer": "缓冲区无位", "inspection": "复验排队",
    "rework": "返工排队", "recheck": "复检排队",
}


def _binding_stage(result: SimResult, order: str) -> tuple[str, int]:
    r = result.orders[order]
    key, value = max(r.waits.items(), key=lambda kv: kv[1])
    return _STAGE_NAMES.get(key, key), value


def _explain_still_blocked(res: SimResult, order: str, baseline: SimResult) -> str:
    """Say which station still holds the order after this package.

    The plant's question is exactly this: after the first bottleneck is
    removed, which station holds the same late pieces?  Naming the binding
    station and its waiting minutes makes the claim auditable.
    """
    after = res.orders[order]
    before = baseline.orders[order]
    if after.stage_at_end != "done":
        return f"期末仍卡在「{after.block_reason}」，{after.waits.get('line', 0)} 分钟后仍未装配"
    if after.late_units <= 0:
        return ""
    stage, minutes = _binding_stage(res, order)
    before_stage, before_minutes = _binding_stage(baseline, order)
    if stage == before_stage:
        return (f"仍迟交 {after.late_units} 件；{stage} 仍是主要等待（{minutes} 分钟），"
                f"本方案没有触及该工段")
    if before_minutes == 0:
        return (f"仍迟交 {after.late_units} 件；原记录中没有等待，"
                f"改动后{stage}成为主要等待（{minutes} 分钟）")
    return (f"仍迟交 {after.late_units} 件；{before_stage}等待由 {before_minutes} 分钟降到 "
            f"{after.waits.get(_reverse(before_stage), 0)} 分钟，"
            f"但订单改由{stage}卡住（{minutes} 分钟）")


def _reverse(label: str) -> str:
    for k, v in _STAGE_NAMES.items():
        if v == label:
            return k
    return label


def route_coverage(pack: Pack) -> list[dict]:
    """Which products may use each inspection resource, and why not.

    This is the "改工艺也要满足品类复验约束" check: buying QA2 does nothing for
    a product whose route has no alternative inspection station.
    """
    out = []
    for opt in pack.options.values():
        if opt.resource not in {r.inspection for r in pack.route_rules.values()} | {
                r.alternative for r in pack.route_rules.values() if r.alternative}:
            continue
        served = [p for p, r in pack.route_rules.items()
                  if r.inspection == opt.resource or r.alternative == opt.resource]
        blocked = [p for p, r in pack.route_rules.items()
                   if r.inspection != opt.resource and r.alternative != opt.resource]
        out.append({
            "option": opt.option, "resource": opt.resource,
            "products_served": served,
            "products_excluded": blocked,
            "reason": ("该品类工艺路线没有替代复验站" if blocked else "全部品类可用"),
        })
    return out


@dataclass
class ComparisonReport:
    world: World
    route_mode: str
    baseline: SimResult
    packages: list[PackageResult]
    infeasible: list[dict]
    option_shapley_units: dict[str, float]
    option_shapley_lum: dict[str, float]
    combo_only_orders: dict[str, list[str]]
    rank_by_units: list[str]
    rank_by_lum: list[str]
    capacity_headroom: dict[str, float]
    peak_utilization: dict[str, float]
    differentiation: str
    route_coverage: list[dict]
    notes: list[str]

    def as_dict(self) -> dict:
        return {
            "world": self.world.as_dict(),
            "route_mode": self.route_mode,
            "baseline": self.baseline.summary(),
            "packages": [p.as_dict() for p in self.packages],
            "infeasible": self.infeasible,
            "option_shapley_on_time_units": {k: round(v, 1)
                                             for k, v in self.option_shapley_units.items()},
            "option_shapley_late_unit_minutes": {k: round(v, 1)
                                                 for k, v in self.option_shapley_lum.items()},
            "benefit_only_in_combination": self.combo_only_orders,
            "rank_by_on_time_units": self.rank_by_units,
            "rank_by_late_unit_minutes": self.rank_by_lum,
            "capacity_headroom": self.capacity_headroom,
            "peak_utilization": self.peak_utilization,
            "route_coverage": self.route_coverage,
            "differentiation": self.differentiation,
            "notes": self.notes,
        }


def _option_shapley(pack: Pack, feasible: set[frozenset[str]],
                    value_units: dict[frozenset[str], int],
                    value_lum: dict[frozenset[str], int]) -> tuple[dict, dict]:
    """Exact Shapley over options, restricted to budget/admissible coalitions.

    A coalition that breaks the budget or the exclusive group has no admissible
    package, so a player that would make a coalition infeasible contributes 0
    there.  The reported shares therefore sum to the best achievable gain over
    all admissible packages, not to an unconstrained upper bound.
    """
    players = sorted(pack.options)
    n = len(players)
    out_u = {p: 0.0 for p in players}
    out_l = {p: 0.0 for p in players}
    for p in players:
        others = [x for x in players if x != p]
        for r in range(len(others) + 1):
            w = 1.0 / (n * math.comb(n - 1, r))
            for combo in itertools.combinations(others, r):
                base = frozenset(combo)
                with_p = frozenset(combo) | {p}
                if base not in feasible or with_p not in feasible:
                    continue
                out_u[p] += w * (value_units[with_p] - value_units[base])
                out_l[p] += w * (value_lum[with_p] - value_lum[base])
    return out_u, out_l


def compare(pack: Pack, world: World | None = None, route_mode: str = "primary",
            rules: TimingRules = TimingRules()) -> ComparisonReport:
    world = world or World(label="基准")
    jobs = _world_jobs(pack, world)
    baseline_cfg = SimConfig(options=frozenset(), route_mode=route_mode)
    baseline = simulate(pack, jobs, baseline_cfg, world, rules)
    feasible = feasible_option_sets(pack)
    values_u: dict[frozenset[str], int] = {frozenset(): baseline.on_time_units}
    values_l: dict[frozenset[str], int] = {frozenset(): baseline.late_unit_minutes}
    packages: list[PackageResult] = []
    infeasible = []
    for entry in feasible:
        opts = entry["options"]
        if not entry["feasible"]:
            infeasible.append({"options": sorted(opts), "cost_cents": entry["cost_cents"],
                               "reasons": entry["violations"]})
            continue
        pkg = evaluate_package(pack, opts, world, baseline, jobs, route_mode, rules)
        packages.append(pkg)
        values_u[opts] = pkg.result.on_time_units
        values_l[opts] = pkg.result.late_unit_minutes
    feasible_sets = {e["options"] for e in feasible if e["feasible"]}
    shap_u, shap_l = _option_shapley(pack, feasible_sets, values_u, values_l)
    # Shapley over budget-feasible coalitions equals the best feasible package
    # minus the baseline only when every subset is feasible; with an
    # infeasible toy coalition the shares still sum to the best package, so
    # record the anchor explicitly instead of asserting it silently.
    best_pkg = max(packages, key=lambda p: p.result.on_time_units) if packages else None
    # orders that only improve when two (or more) options are bought together
    combo: dict[str, list[str]] = {}
    single_best: dict[str, int] = {}
    for pkg in packages:
        if not pkg.violates_commitments:
            for order, info in pkg.unblocked.items():
                single_best[order] = max(single_best.get(order, 0), info["delta_units"])
    for pkg in packages:
        # a package that breaks committed promises is not a candidate winner;
        # its "gains" cannot be offered to management
        if len(pkg.options) < 2 or pkg.violates_commitments:
            continue
        only = [o for o, info in pkg.unblocked.items()
                if info["delta_units"] > single_best.get(o, 0)]
        if only:
            combo["+".join(sorted(pkg.options))] = sorted(only)
    ranked_u = [ "+".join(sorted(p.options)) or "基线"
                 for p in sorted(packages, key=lambda p: (-p.gain_on_time_units, p.cost_cents))]
    ranked_l = [ "+".join(sorted(p.options)) or "基线"
                 for p in sorted(packages, key=lambda p: (-p.gain_late_unit_minutes, p.cost_cents))]
    headroom = {r: baseline.utilization(r) for r in ("L1", "L2", "QA", "RW")}
    # "Differentiation" means a package beats doing nothing; harm from a bad
    # option set (B's construction loss) is a risk signal, not a benefit.
    gain_units = max(p.result.on_time_units for p in packages) - baseline.on_time_units
    spread_units = (max(p.result.on_time_units for p in packages)
                    - min(p.result.on_time_units for p in packages))
    if gain_units <= 0:
        differentiation = "none"
    elif gain_units < baseline.delivered_units * 0.02:
        differentiation = "weak"
    else:
        differentiation = "clear"
    # Headroom is judged on the run actually simulated, not the base run: a
    # compressed release can saturate a station while the baseline average
    # still looks idle.
    busy = {r: 0 for r in ("L1", "L2", "QA", "RW")}
    cap = {r: 0 for r in busy}
    for pkg in packages:
        for r in busy:
            busy[r] = max(busy[r], pkg.result.resource_busy.get(r, 0))
            cap[r] = max(cap[r], pkg.result.resource_capacity.get(r, 0))
    peak = {r: (busy[r] / cap[r] if cap[r] else 0.0) for r in busy}
    notes = []
    if differentiation == "none":
        notes.append("本批订单上没有任何方案提高按期交付量：资料给出的计划订单量相对班次能力很小，"
                     "按期件数无法区分方案（反而可能因施工降容变差）。")
        notes.append(f"各工段峰值占用：{ {k: f'{v:.1%}' for k, v in peak.items()} }。"
                     "这与参数里的交期承诺共同决定了本情景是否真的卡产能。")
        notes.append("要看方案差异，需要用 --span / --due-scale 构造订单组合更紧的情景，"
                     "或直接比较缓冲阻塞、施工期降容、已承诺订单违约这些风险轴。")
        notes.append("若订单量本身足够小，资料即不支持“增机可提高交付量”的结论；"
                     "此时任意精确排序都是过度解读。")
    elif differentiation == "weak":
        notes.append("方案之间的按期件差异很小，排序结果对到料和返流假设敏感，"
                     "建议同时看风险轴而不是只看按期件。")
    worst = min(packages, key=lambda p: p.result.on_time_units)
    if worst.result.on_time_units < baseline.on_time_units:
        notes.append(f"方案 {'+'.join(sorted(worst.options))} 比不改造更差："
                     f"施工期减少 {worst.result.buffer_block_minutes} 分钟的缓冲阻塞并未抵消"
                     "工位停机造成的能力损失；这说明施工窗口本身是风险，不是收益。")
    return ComparisonReport(
        world=world, route_mode=route_mode, baseline=baseline, packages=packages,
        infeasible=infeasible, option_shapley_units=shap_u, option_shapley_lum=shap_l,
        combo_only_orders=combo, rank_by_units=ranked_u, rank_by_lum=ranked_l,
        capacity_headroom=headroom, peak_utilization=peak,
        differentiation=differentiation, notes=notes,
        route_coverage=route_coverage(pack))


# --------------------------------------------------------------------------- #
# Phased rollout
# --------------------------------------------------------------------------- #
@dataclass
class PhaseState:
    """End state a phase hands to the next one."""
    shipped: list[str]
    in_progress: dict[str, dict]
    capacity_assets: list[str]
    world: World
    route_mode: str
    decision_min: int
    committed_due: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "shipped_orders": sorted(self.shipped),
            "in_progress": self.in_progress,
            "installed_options": sorted(self.capacity_assets),
            "world": self.world.as_dict(),
            "route_mode": self.route_mode,
            "decision_min": self.decision_min,
            "committed_due_min": self.committed_due,
        }


def phase_state(pack: Pack, result: SimResult, world: World, route_mode: str,
                options, decision_min: int) -> PhaseState:
    shipped = [o for o, r in result.orders.items() if r.stage_at_end == "done" and r.late_units <= 0]
    in_progress = {}
    for o, r in result.orders.items():
        if o in shipped:
            continue
        if r.stage_at_end in ("done", "done_pending"):
            continue
        # remaining units: delivered partial release is already out
        remaining = r.units - r.delivered
        in_progress[o] = {
            "stage": r.stage_at_end, "remaining_units": remaining,
            "due_min": r.due_min, "product": r.product,
            "priority": r.priority, "committed": r.committed,
        }
    return PhaseState(shipped=shipped, in_progress=in_progress,
                      capacity_assets=sorted(options), world=world,
                      route_mode=route_mode, decision_min=decision_min,
                      committed_due={o: r.due_min for o, r in result.orders.items()})


def _jobs_for_phase2(pack: Pack, phase1_jobs: list[Job], phase1_result: SimResult,
                     state: PhaseState, new_world: World) -> list[Job]:
    """Carry the phase-1 job list forward, minus what phase 1 fully shipped.

    Using the *same* job objects (instead of re-deriving from the pack) keeps
    the order ids and quantities identical across phases, including copies
    created by a demand multiplier.  Any order that phase 1 did not finish is
    re-entered at its last observed stage so the phase-2 start state is the
    phase-1 end state, not a fresh schedule.
    """
    jobs = []
    shipped = set(state.shipped)
    for j in phase1_jobs:
        if j.order in shipped:
            continue
        remaining = phase1_result.orders[j.order].delivered
        units_left = j.units - remaining
        if units_left <= 0:
            continue
        stage = phase1_result.orders[j.order].stage_at_end
        stage0 = ("buffer" if stage in ("buffer", "inspection", "buffer_queue") else
                  "rw" if stage in ("rework", "rw_queue", "recheck", "recheck_queue") else
                  "new")
        jobs.append(Job(
            order=j.order, product=j.product, line=j.line, units=units_left,
            available_min=state.decision_min, due_min=j.due_min,
            priority=j.priority, allow_partial=j.allow_partial,
            material_min=state.decision_min, committed=j.committed,
            stage0=stage0, rework=j.rework))
    return jobs


@dataclass
class Phase2Report:
    start_state: PhaseState
    phase1_options: frozenset[str]
    phase2_options: list[str]
    baseline_phase2: SimResult
    phase2: SimResult
    marginal_units: int
    marginal_lum: int
    note: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "phase1_state": self.start_state.as_dict(),
            "phase1_package": sorted(self.phase1_options),
            "phase2_new_options": sorted(self.phase2_options),
            "phase2_baseline": self.baseline_phase2.summary(),
            "phase2_with_new_options": self.phase2.summary(),
            "marginal_on_time_units": self.marginal_units,
            "marginal_late_unit_minutes": self.marginal_lum,
            "note": self.note,
        }


def phase2(pack: Pack, phase1_options, phase2_new, phase1_result: SimResult,
           world: World, route_mode: str = "primary",
           rules: TimingRules = TimingRules(),
           phase1_jobs: list[Job] | None = None) -> Phase2Report:
    decision = min(
        pack.options[o].effective_min for o in phase2_new) if phase2_new else pack.horizon[0]
    state = phase_state(pack, phase1_result, world, route_mode, phase1_options, decision)
    if phase1_jobs is None:
        phase1_jobs = make_jobs(pack, world)
    jobs = _jobs_for_phase2(pack, phase1_jobs, phase1_result, state, world)
    carried = frozenset(phase1_options)
    base_cfg = SimConfig(options=carried, route_mode=route_mode)
    base = simulate(pack, jobs, base_cfg, world, rules, t0=decision, t1=pack.horizon[1])
    new_cfg = SimConfig(options=carried | frozenset(phase2_new), route_mode=route_mode)
    new = simulate(pack, jobs, new_cfg, world, rules, t0=decision, t1=pack.horizon[1])
    note = [f"二期以 {decision} 分钟为起点：一期预计已交付 {len(state.shipped)} 单，"
            f"{len(state.in_progress)} 单仍在制。",
            "一期购入的工位作为既有资产带入二期容量。"]
    return Phase2Report(
        start_state=state, phase1_options=frozenset(phase1_options),
        phase2_options=list(phase2_new),
        baseline_phase2=base, phase2=new,
        marginal_units=new.on_time_units - base.on_time_units,
        marginal_lum=base.late_unit_minutes - new.late_unit_minutes, note=note)
