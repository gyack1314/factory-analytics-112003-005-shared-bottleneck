"""Human-readable rendering of the decision outputs.

Everything is plain text so it can be pasted into an email or printed; the
same data is available as JSON from ``python3 -m planner.cli --json``.
"""
from __future__ import annotations

from .attribution import AttributionReport, CAUSE_LABELS
from .compare import ComparisonReport, Phase2Report
from .validate import ReplayReport


def _yuan(cents: int) -> str:
    return f"{cents / 100:,.0f} 元"


def _w(text: str) -> int:
    """Display width in terminal cells (CJK characters take two)."""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(text))


def _pad(text, width: int, align: str = "<") -> str:
    text = str(text)
    gap = max(0, width - _w(text))
    if align == ">":
        return " " * gap + text
    if align == "^":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


def _bar(frac: float, width: int = 24) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def render_attribution(a: AttributionReport) -> str:
    lines = []
    lines.append("一、历史交付短缺归因")
    lines.append("=" * 72)
    lines.append(f"可用于精确比较的订单 {a.clean_orders} 单；"
                 f"记录不全、只给区间的订单 {a.ambiguous_orders} 单。")
    lines.append(f"这 {a.clean_orders} 单合计未按期交付 {a.total_short} 件，"
                 f"迟交量 {a.total_late_unit_min:.0f} 件·分钟。")
    lines.append("")
    total_expl = max(1.0, a.explained_short)
    lines.append("原因（精确 Shapley，三因分摊后相加正好等于可解释总量）")
    lines.append("  " + _pad("原因", 26) + _pad("件", 8, ">") + _pad("占比", 8, ">")
                 + "  " + _pad("件·分钟", 11, ">"))
    for c in ("M", "T", "R"):
        units = a.shapley_short[c]
        lum = a.shapley_late_unit_min[c]
        lines.append("  " + _pad(CAUSE_LABELS[c], 26) + _pad(f"{units:.0f}", 8, ">")
                     + _pad(f"{units / total_expl:.0%}", 8, ">") + "  "
                     + _pad(f"{lum:.0f}", 11, ">"))
    for label, units, lum in (("小计（可解释）", a.explained_short, a.explained_late_unit_min),
                              ("结构性缺口", a.residue_short, a.residue_late_unit_min),
                              ("合计", a.total_short, a.total_late_unit_min)):
        lines.append("  " + _pad(label, 26) + _pad(f"{units:.0f}", 8, ">") + _pad("", 8, ">")
                     + "  " + _pad(f"{lum:.0f}", 11, ">"))
    lines.append("")
    lines.append(f"交互项：三因同时消除比分别消除多出 {a.joint_interaction_short:+.0f} 件；"
                 "这正是两条产线争夺同一台复验装置时不能把贡献分别记满的部分。")
    lines.append("")
    if a.residue_by_station:
        lines.append("结构性缺口按最先撞线的工段拆分：")
        for station, units in sorted(a.residue_by_station.items(), key=lambda x: -x[1]):
            lines.append(f"  {station}: {units} 件")
        lines.append("  这些订单即使把材料、转移等待和返流全部消除仍会迟交："
                     "它们的『件数+5 分钟装配 + min(14,2×件数) 复验』本身就超过承诺交期。"
                     "增机不缩短单批工时，加缓冲位也不改变工时，因此这两类改造都救不了它们；"
                     "只能改承诺交期、拆批或改工艺。")
        lines.append("")
    lines.append("单因消除的边际效果（用于对照，不是分摊）：")
    for c in ("M", "T", "R"):
        lines.append(f"  只消除{CAUSE_LABELS[c]}：少迟交 {a.single_cause_short[c]:.0f} 件")
    lines.append("")
    lines.append("各工段历史占用（证明设备不是唯一瓶颈）：")
    for label, info in a.capacity_headroom.items():
        lines.append("  " + _pad(label, 10) + " 占用 "
                     + _pad(f"{info['utilization']:.1%}", 7, ">")
                     + f"  ({info['busy_min']} / {info['observed_span_min']} 分钟)")
    lines.append("")
    lines.append("记录不全的区间（不进入精确比较）：")
    s = a.ambiguity_summary
    lines.append(f"  {s['orders']} 单；记录值 {s['short_units_recorded']} 件，"
                 f"可观测下界 {s['short_units_lower_bound']} 件、"
                 f"上界 {s['short_units_upper_bound']} 件。")
    lines.append(f"  原因：缺复验开始时刻 {s['reasons']['missing_start']} 单；"
                 f"QA 区间重叠 {s['reasons']['qa_overlap']} 单。")
    for ep in a.qa_overlap_episodes:
        lines.append(f"   - {ep['pair'][0]} / {ep['pair'][1]} 复验重叠 {ep['overlap_min']} 分钟；"
                     f"若串行，{ep['later_order']} 将推迟到迟交。")
    lines.append("")
    lines.append(f"每班检验节拍（记录实证）：{a.transfer_wave} 分钟。"
                 "历史中每次复验都按节拍开始，而 QA 只忙了约 16%，两者同时成立，"
                 "说明这是排班节拍而不是产能排队。因此增机不会自动消除这段等待；"
                 "要消除它必须改排班规则。")
    lines.append("")
    lines.append("各工段“闲/忙”的性质（记录可区分）：")
    for d in a.station_diagnosis:
        extra = ""
        if d["rework_min"]:
            extra += f"，其中返工 {d['rework_min']} 分钟"
        if d["recheck_min"]:
            extra += f"，其中返工复检 {d['recheck_min']} 分钟"
        lines.append(f"  {d['station']:<10} 忙 {d['busy_min']:>5} 分钟"
                     f"（{d['utilization']:.1%}）{extra}")
        lines.append(f"      {d['diagnosis']}")
    sr = a.sunk_rework
    lines.append("")
    lines.append("拒收后的沉没成本（已消耗、不能回收）：")
    lines.append(f"  返流订单 {sr['rework_jobs']} 单，涉及 {sr['affected_units']} 件；"
                 f"返工占用 RW {sr['rework_minutes']} 分钟，"
                 f"返工后复检占用 QA {sr['recheck_minutes']} 分钟，"
                 f"复检前等待 {sr['recheck_wait_minutes']} 分钟。")
    lines.append(f"  这些订单在 QA 上共占用 {sr['qa_slots_reused']} 个额外复检时段："
                 "槽位与工时都已真实消耗，后续排程只能在其之后排队。")
    return "\n".join(lines)


def render_comparison(c: ComparisonReport) -> str:
    lines = []
    lines.append("二、同一批订单上的方案对比")
    lines.append("=" * 72)
    w = c.world
    extra = []
    if w.rework_orders:
        extra.append(f"预计返流：{', '.join(w.rework_orders)}")
    if w.material_slip_min:
        extra.append(f"到料额外推迟 {w.material_slip_min} 分钟")
    if w.release_span_shifts is not None:
        extra.append(f"订单压缩到前 {w.release_span_shifts} 个班次下达")
    if w.due_scale != 1.0:
        extra.append(f"交期承诺折扣 ×{w.due_scale:g}")
    lines.append(f"情景读数：{w.label}" + (f"（{'；'.join(extra)}）" if extra else ""))
    b = c.baseline
    lines.append(f"基线：按期 {b.on_time_units} 件，迟交 {b.late_units} 件，"
                 f"期末积压 {b.backlog_units} 件，缓冲阻塞 {b.buffer_block_minutes} 分钟。")
    if c.notes:
        lines.append("")
        for n in c.notes:
            lines.append(f"  注意：{n}")
    lines.append("")
    lines.append(_pad("方案", 10) + _pad("造价", 10, ">") + _pad("按期件", 8, ">")
                 + _pad("增量", 7, ">") + _pad("迟交件", 7, ">")
                 + _pad("迟交件·分", 11, ">") + _pad("积压", 6, ">")
                 + _pad("缓冲阻塞", 9, ">"))
    for pkg in sorted(c.packages, key=lambda p: (-p.gain_on_time_units, p.cost_cents)):
        name = "+".join(sorted(pkg.options)) or "不改造"
        r = pkg.result
        lines.append(_pad(name, 10) + _pad(_yuan(pkg.cost_cents), 10, ">")
                     + _pad(r.on_time_units, 8, ">")
                     + _pad(f"{pkg.gain_on_time_units:+d}", 7, ">")
                     + _pad(r.late_units, 7, ">") + _pad(r.late_unit_minutes, 11, ">")
                     + _pad(r.backlog_units, 6, ">") + _pad(r.buffer_block_minutes, 9, ">"))
    lines.append("")
    if c.infeasible:
        lines.append("资料不允许的方案（只给原因，不给精确解）：")
        for bad in c.infeasible:
            lines.append(f"  {'+'.join(bad['options']) or '空集'}：{'；'.join(bad['reasons'])}")
        lines.append("")
    lines.append("逐单追溯：每个方案的收益都能落到具体订单上")
    for pkg in sorted(c.packages, key=lambda p: (-p.gain_on_time_units, p.cost_cents)):
        if not pkg.options:
            continue
        name = "+".join(sorted(pkg.options))
        if not pkg.unblocked:
            lines.append(f"  {name}：没有订单因此转为按期。")
            continue
        lines.append(f"  {name}（+{pkg.gain_on_time_units} 件按期）：")
        for order, info in sorted(pkg.unblocked.items()):
            lines.append(f"    {order}: 按期 {info['on_time_units_before']}→{info['on_time_units_after']} 件；"
                         f"{info['why']}")
    lines.append("")
    if c.combo_only_orders:
        lines.append("只在组合时才受益的订单：")
        for combo, orders in sorted(c.combo_only_orders.items()):
            lines.append(f"  {combo}: {', '.join(orders)}")
    else:
        lines.append("本批订单中没有“只在组合时才受益”的订单：每个被救回的订单都至少能被一个单方案救回。")
    lines.append("")
    lines.append("仍会卡住的订单（消除一个瓶颈后，另一个工段接手）：")
    seen = set()
    for pkg in c.packages:
        for order, why in pkg.blocked_by_other.items():
            key = (order, why)
            if key in seen:
                continue
            seen.add(key)
    for pkg in c.packages:
        if not pkg.options:
            continue
        if not pkg.blocked_by_other:
            continue
        lines.append(f"  {'+'.join(sorted(pkg.options))}：")
        for order, why in pkg.blocked_by_other.items():
            lines.append(f"    {order}: {why}")
    lines.append("")
    lines.append("方案选项的 Shapley 边际（按可购组合计算）：")
    for opt in sorted(c.option_shapley_units):
        lines.append(f"  {opt}: 按期 {c.option_shapley_units[opt]:+.1f} 件，"
                     f"迟交量 {c.option_shapley_lum[opt]:+.1f} 件·分钟")
    lines.append("")
    lines.append("风险轴（按期件相同时用这些区分）：")
    lines.append("  " + _pad("方案", 10) + _pad("承诺违约", 9, ">")
                 + _pad("缓冲阻塞(分)", 13, ">") + _pad("期末积压(件)", 13, ">") + "  说明")
    for pkg in sorted(c.packages, key=lambda p: (len(p.options), p.cost_cents)):
        name = "+".join(sorted(pkg.options)) or "不改造"
        breach = ",".join(pkg.committed_breach) if pkg.committed_breach else "无"
        verdict = ("不可接受：已承诺订单/半成品被推迟" if pkg.violates_commitments
                   else f"造价 {_yuan(pkg.cost_cents)}")
        lines.append("  " + _pad(name, 10) + _pad(breach, 9, ">")
                     + _pad(pkg.result.buffer_block_minutes, 13, ">")
                     + _pad(pkg.result.backlog_units, 13, ">") + "  " + verdict)
    if any(pkg.violates_commitments for pkg in c.packages):
        lines.append("")
        lines.append("  「承诺违约」是硬约束：已承诺的优先订单与半成品必须继续履约，"
                     "因此即使按期总量更高，这类方案也不能作为推荐方案。")
    lines.append("")
    if c.route_coverage:
        lines.append("品类复验约束（改工艺也不能绕过）：")
        for rc in c.route_coverage:
            excl = "、".join(rc["products_excluded"]) or "无"
            lines.append(f"  {rc['option']} → {rc['resource']}：可用品类 "
                         f"{'、'.join(rc['products_served'])}；不可用 {excl}"
                         f"（{rc['reason']}）")
    lines.append("")
    lines.append(f"按期件排序：{' > '.join(c.rank_by_units)}")
    lines.append(f"迟交量排序：{' > '.join(c.rank_by_lum)}")
    return "\n".join(lines)


def render_phase2(p: Phase2Report) -> str:
    lines = []
    lines.append("四、分期：以一期预计切换状态为起点")
    lines.append("=" * 72)
    lines.extend(f"  {n}" for n in p.note)
    lines.append(f"  起点：已交付 {len(p.start_state.shipped)} 单，"
                 f"在制 {len(p.start_state.in_progress)} 单，"
                 f"既有工位 {', '.join(p.start_state.capacity_assets) or '无'}。")
    b, n = p.baseline_phase2, p.phase2
    lines.append(f"  二期不加新方案：按期 {b.on_time_units} 件，迟交 {b.late_units} 件，"
                 f"积压 {b.backlog_units} 件。")
    lines.append(f"  二期加入 {', '.join(p.phase2_options)}：按期 {n.on_time_units} 件，"
                 f"迟交 {n.late_units} 件，积压 {n.backlog_units} 件。")
    lines.append(f"  二期边际：按期 {p.marginal_units:+d} 件，"
                 f"迟交量 {p.marginal_lum:+d} 件·分钟。")
    if p.phase2.on_time_units == p.baseline_phase2.on_time_units:
        lines.append("  二期新增方案没有改变按期件：在本起点与需求读数下，"
                     "该方案的作用只体现在缓冲阻塞或未来更紧的订单组合上，"
                     "不能据此宣称二期收益。")
    if n.committed_breach:
        lines.append(f"  已承诺订单仍会违约：{', '.join(n.committed_breach)}")
    else:
        lines.append("  已承诺的优先订单与半成品全部按期。")
    return "\n".join(lines)


def render_robustness(r) -> str:
    lines = []
    lines.append("三、资料不支持处的区间（不做唯一解）")
    lines.append("=" * 72)
    lines.append(f"情景族：{r.family.description}（共 {len(r.family)} 个可行情景）")
    lines.append(f"不改造的按期件区间：{r.baseline_range[0]}–{r.baseline_range[1]}")
    lines.append("")
    lines.append(_pad("方案", 10) + _pad("按期件区间", 14, ">")
                 + _pad("迟交量区间", 20, ">") + _pad("缓冲阻塞", 12, ">")
                 + _pad("承诺违约", 10, ">"))
    for p in r.packages:
        name = "+".join(sorted(p.options)) or "不改造"
        lines.append(_pad(name, 10)
                     + _pad(f"{p.min_units}–{p.max_units}", 14, ">")
                     + _pad(f"{p.min_lum}–{p.max_lum}", 20, ">")
                     + _pad(f"{p.min_block}–{p.max_block}", 12, ">")
                     + _pad(f"{p.breach_worlds}/{p.total_worlds}", 10, ">"))
    lines.append("")
    lines.append(r.robust_note)
    if r.robust_winner:
        lines.append(f"稳健领先：{r.robust_winner}")
    else:
        lines.append("原因：各方案的按期件/缓冲阻塞区间互相重叠，任何“最优”选择都取决于哪一种"
                     "需求/到料/返流读数。")
        lines.append("能确定的事实（由全部情景的结果直接得出，不是推断）：")
        broken = sorted(k for k, v in r.breach_counts.items()
                        if v.split("/")[0] != "0" and v.split("/")[1] == r.breach_counts[k].split("/")[1])
        for name in broken:
            lines.append(f"  · {name}：{r.breach_counts[name]} 个情景里都让已承诺订单/半成品迟交。"
                         "施工窗口与承诺交期重叠是硬冲突，不是需求读数问题。")
        if "不改造" in r.best_world_packages:
            base_peak = r.best_world_packages["不改造"]
            better = sorted((k for k, v in r.best_world_packages.items()
                             if k != "不改造" and v > base_peak),
                            key=lambda k: -r.best_world_packages[k])
            if better:
                lines.append(f"  · 相对不改造（最好 {base_peak} 件），这些方案在某些情景下更好："
                             f"{', '.join(better)}。因此改造并非无用，只是收益依赖需求读数。")
            else:
                lines.append("  · 没有任何方案在任何情景下超过不改造的最好结果："
                             "资料不支持“这些改造会提高交付量”的结论。")
    lines.append("")
    lines.append("按各方案下限排序（保守读法）：")
    for p in r.packages:
        name = "+".join(sorted(p.options)) or "不改造"
        lines.append(f"  {name:<10} 最差按期 {p.min_units} 件；最好 {p.max_units} 件")
    return "\n".join(lines)


def render_validation(r: ReplayReport) -> str:
    lines = []
    lines.append("五、用历史班次检验试算")
    lines.append("=" * 72)
    lines.append(f"重放班次 {r.shifts[0]}–{r.shifts[-1]}，共 {r.orders} 单。")
    lines.append(f"  交付量：记录 {r.delivered_units_expected} 件，试算 {r.delivered_units_actual} 件。")
    lines.append(f"  交付时刻完全吻合：{r.exact_orders} / {r.orders} 单。")
    for kind, count in sorted(r.kind_counts.items()):
        lines.append(f"    {kind}: {count} 单")
    lines.append(f"  资源占用（分钟）：")
    lines.append(f"    {'资源':<6}{'记录':>8}{'试算':>8}")
    for res in r.resource_minutes_expected:
        lines.append(f"    {res:<6}{r.resource_minutes_expected[res]:>8}{r.resource_minutes_actual[res]:>8}")
    lines.append("  各班末积压位置（仅列前 6 单，完整清单见 JSON）：")
    lines.append(f"    {'班次':<6}{'记录':<34}{'试算':<34}")
    for sid in (str(s) for s in r.shifts):
        e = r.backlog_expected.get(sid, "-")
        a = r.backlog_actual.get(sid, "-")
        e = e if len(e) <= 32 else e[:29] + "..."
        a = a if len(a) <= 32 else a[:29] + "..."
        lines.append(f"    {sid:<6}{e:<34}{a:<34}")
    if r.backlog_expected == r.backlog_actual:
        lines.append("    各班末积压清单完全一致。")
    if r.validated:
        lines.append("  结论：所有偏差都可归入已知记录缺陷（缺 start_min / QA 区间重叠），"
                     "没有无法解释的偏差；试算可用于方案比较。")
    else:
        lines.append("  结论：存在无法解释的偏差，试算结果不能直接用于方案比较。")
    return "\n".join(lines)
