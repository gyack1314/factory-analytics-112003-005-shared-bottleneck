"""Command-line entry point for the retrofit decision tool.

Examples
--------
    python3 -m planner.cli report
    python3 -m planner.cli compare --world history
    python3 -m planner.cli compare --world history --route allow_alt
    python3 -m planner.cli compare --world stress --slip 360 --rework N1,N4
    python3 -m planner.cli phase2 --phase1 A --phase2 C --world history
    python3 -m planner.cli validate --shifts 0-2,19-21
    python3 -m planner.cli report --json > decision.json

``--world history`` is the default reading: it reuses the 19/132 historical
rejection rate and attributes each future order a rejection by deterministic
spread.  ``--world none`` assumes no rejections; ``--world stress`` assumes
every listed order is rejected.  The pack cannot settle this, so the tool
always prints which reading produced the numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .attribution import analyse
from .compare import compare, phase2
from .model import build_pack
from .report import (render_attribution, render_comparison, render_phase2,
                     render_robustness, render_validation)
from .scenario import World, make_jobs
from . import ledger
from .robustness import analyse as robustness_analyse, default_family, by_risk_family
from .validate import replay

DEFAULT_LEDGER = Path(__file__).resolve().parent.parent / "out" / "decision-ledger.json"


def _parse_orders(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _history_rework(pack) -> tuple[str, ...]:
    """Deterministic spread of the historical rejection rate onto future orders."""
    total = len(pack.orders)
    rejected = sum(1 for o in pack.orders
                   if any(e["order"] == o["order"] and e["event"] == "返工" for e in pack.history))
    if not total or not rejected:
        return ()
    rate = rejected / total
    ids = sorted(o["order"] for o in pack.planning_orders)
    count = max(1, round(rate * len(ids)))
    step = len(ids) / count
    return tuple(ids[int(i * step)] for i in range(count))


def _world(args, pack) -> World:
    kind = args.world
    span = args.span
    scale = args.due_scale
    mult = args.multiplier
    if kind == "none":
        return World(label="无返流假设", material_slip_min=args.slip,
                     release_span_shifts=span, due_scale=scale, multiplier=mult)
    if kind == "stress":
        ids = _parse_orders(args.rework) or tuple(o["order"] for o in pack.planning_orders)
        return World(label="压力假设：全部返流", rework_orders=ids,
                     material_slip_min=args.slip,
                     release_span_shifts=span, due_scale=scale, multiplier=mult)
    ids = _parse_orders(args.rework) or _history_rework(pack)
    return World(label="按历史返流率外推", rework_orders=ids,
                 material_slip_min=args.slip,
                 release_span_shifts=span, due_scale=scale, multiplier=mult)


def _parse_shifts(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _contiguous(shifts: list[int]) -> list[list[int]]:
    blocks: list[list[int]] = []
    for sh in sorted(set(shifts)):
        if blocks and sh == blocks[-1][-1] + 1:
            blocks[-1].append(sh)
        else:
            blocks.append([sh])
    return blocks


def _replay_all(pack, shifts: list[int]):
    reports = [replay(pack, block) for block in _contiguous(shifts)]
    return reports


def _emit(args, text: str, payload) -> None:
    """JSON mode prints machine-readable output only."""
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(text)


def cmd_validate(args, pack) -> None:
    reports = _replay_all(pack, _parse_shifts(args.shifts))
    text = "\n\n".join(render_validation(r) for r in reports)
    _emit(args, text, [r.as_dict() for r in reports])


def cmd_report(args, pack) -> None:
    a = analyse(pack)
    world = _world(args, pack)
    c = compare(pack, world, route_mode=args.route)
    fam = by_risk_family(pack) if args.family == "risk" else default_family(pack)
    rob = robustness_analyse(pack, fam)
    replays = _replay_all(pack, _parse_shifts(args.shifts))
    text = "\n\n".join([
        render_attribution(a), render_comparison(c), render_robustness(rob),
        "\n\n".join(render_validation(r) for r in replays)])
    _emit(args, text, {"attribution": a.as_dict(), "comparison": c.as_dict(),
                       "robustness": rob.as_dict(),
                       "validation": [r.as_dict() for r in replays]})


def cmd_ranges(args, pack) -> None:
    fam = by_risk_family(pack) if args.family == "risk" else default_family(pack)
    rob = robustness_analyse(pack, fam)
    _emit(args, render_robustness(rob), rob.as_dict())


def cmd_compare(args, pack) -> None:
    world = _world(args, pack)
    c = compare(pack, world, route_mode=args.route)
    entry = ledger.record_comparison(c, note=args.note, path=args.ledger)
    text = (render_comparison(c) + "\n\n"
            + f"已记入决策台账 {args.ledger}（摘要 {entry['digest']}）")
    payload = c.as_dict()
    payload["ledger"] = {"path": str(args.ledger), "digest": entry["digest"]}
    _emit(args, text, payload)


def cmd_ledger(args, pack) -> None:
    entries = ledger.load(args.ledger)
    if args.digest:
        entry = ledger.find(entries, args.digest)
        if entry is None:
            sys.exit(f"台账中没有摘要 {args.digest}")
        print(json.dumps(entry, ensure_ascii=False, indent=2))
        return
    if not entries:
        print(f"台账 {args.ledger} 为空。")
        return
    print(f"台账 {args.ledger} 共 {len(entries)} 条记录：")
    for e in entries:
        kind = e["kind"]
        world = (e.get("world") or e.get("phase1_state", {}).get("world", {}))
        head = (f"{e['recorded_at']}  {e['digest']}  {kind}  "
                f"{world.get('label', '')}")
        print(f"  {head}")
        if e.get("note"):
            print(f"      备注：{e['note']}")
        if kind == "comparison":
            best = sorted(e["packages"], key=lambda p: -p["on_time_units"])[:3]
            for b in best:
                print(f"      {'+'.join(b['options']) or '不改造'}: 按期 {b['on_time_units']} 件")
        else:
            print(f"      二期新增 {', '.join(e['phase2_new_options'])}："
                  f"边际按期 {e['marginal_on_time_units']:+d} 件")


def cmd_phase2(args, pack) -> None:
    world = _world(args, pack)
    p1_opts = tuple(x for x in args.phase1.split(",") if x)
    p2_opts = tuple(x for x in args.phase2.split(",") if x)
    c = compare(pack, world, route_mode=args.route)
    chosen = next((p for p in c.packages if p.options == frozenset(p1_opts)), None)
    if chosen is None:
        sys.exit(f"一期方案 {p1_opts} 不在可行集合内")
    jobs = make_jobs(pack, world)
    report = phase2(pack, chosen.options, p2_opts, chosen.result, world,
                    route_mode=args.route, phase1_jobs=jobs)
    entry = ledger.record_phase2(report, note=args.note, path=args.ledger)
    text = (render_phase2(report) + "\n\n"
            + f"已记入决策台账 {args.ledger}（摘要 {entry['digest']}）")
    payload = report.as_dict()
    payload["ledger"] = {"path": str(args.ledger), "digest": entry["digest"]}
    _emit(args, text, payload)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="共享工段产能归因与改造试算")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--world", choices=["history", "none", "stress"], default="history")
    common.add_argument("--slip", type=int, default=0, help="未来订单到料额外推迟分钟")
    common.add_argument("--rework", default="", help="未来返流订单号，逗号分隔")
    common.add_argument("--route", choices=["primary", "allow_alt"], default="primary",
                        help="是否允许薄板走替代检验路线 QA2")
    common.add_argument("--span", type=int, default=None,
                        help="把计划订单压缩到前 N 个班次下达（订单组合集中情景）")
    common.add_argument("--due-scale", type=float, default=1.0,
                        help="压缩交期承诺，1.0 为资料原值")
    common.add_argument("--multiplier", type=int, default=1,
                        help="把计划订单按倍数复制（同一组合重复下达），用于检验产能是否真的卡住")
    common.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    common.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER,
                        help="决策台账文件（只追加，不覆盖）")
    common.add_argument("--note", default="", help="本次评估的备注，写入台账")

    sp = sub.add_parser("report", parents=[common], help="归因 + 方案对比 + 区间 + 历史重放")
    sp.add_argument("--shifts", default="0-2,19-21", help="重放班次，如 0-2,19-21")
    sp.add_argument("--family", choices=["default", "risk"], default="default",
                    help="区间分析的情景族")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("ranges", parents=[common], help="只做区间/稳健性分析")
    sp.add_argument("--family", choices=["default", "risk"], default="default")
    sp.set_defaults(func=cmd_ranges)

    sp = sub.add_parser("compare", parents=[common], help="只做方案对比")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("validate", parents=[common], help="只做历史重放检验")
    sp.add_argument("--shifts", default="0-2,19-21")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("ledger", parents=[common], help="查看已记录的决策")
    sp.add_argument("--digest", default=None, help="只看某一次记录")
    sp.set_defaults(func=cmd_ledger)

    sp = sub.add_parser("phase2", parents=[common], help="一期确认后的二期评估")
    sp.add_argument("--phase1", default="A", help="一期方案，空字符串表示一期不改造")
    sp.add_argument("--phase2", default="C")
    sp.set_defaults(func=cmd_phase2)

    args = ap.parse_args(argv)
    pack = build_pack()
    args.func(args, pack)


if __name__ == "__main__":
    main()
