"""Browse the data pack and the rules recovered from it.

    python3 inspect_inputs.py                # table sizes
    python3 inspect_inputs.py --rules        # timing rules + validation result
    python3 inspect_inputs.py --orders       # one line per order
    python3 inspect_inputs.py --options      # option feasibility
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_inputs(root=ROOT):
    return {p.stem: json.loads(p.read_text(encoding='utf-8'))
            for p in sorted((root / 'data').glob('*.json'))}


def main(argv=None):
    ap = argparse.ArgumentParser(description="数据包浏览")
    ap.add_argument("--rules", action="store_true", help="显示从记录反推的节拍规则")
    ap.add_argument("--orders", action="store_true", help="逐单显示")
    ap.add_argument("--options", action="store_true", help="显示改造方案可行性")
    args = ap.parse_args(argv)

    data = read_inputs()
    for name, value in data.items():
        count = len(value) if isinstance(value, list) else 1
        print(f'{name}：{count} 条')

    if args.rules:
        from planner.model import build_pack
        from planner.history import replay_check, qa_overlap_episodes
        pack = build_pack(ROOT / "data")
        report = replay_check(pack)
        print()
        print("历史重放校验：")
        print(f"  订单 {report['orders']} 单，全部交付事件可复现：{report['match']}")
        for key in ("assembly_failures", "queue_failures", "rework_failures", "delivery_failures"):
            print(f"  {key}: {len(report[key])}")
        print(f"  QA 区间重叠：{len(qa_overlap_episodes(pack))} 对")

    if args.orders:
        from planner.history import iter_traces
        from planner.model import build_pack
        pack = build_pack(ROOT / "data")
        print()
        print(f"{'订单':<8}{'产品':<6}{'件':>4}{'交期':>8}{'到料':>8}"
              f"{'复验等待':>8}{'返流':>4}{'缺开始':>6}")
        for t in iter_traces(pack):
            print(f"{t.order:<8}{t.product:<6}{t.units:>4}{t.due_min:>8}"
                  f"{t.arrival_min:>8}{t.qa_wait_min:>8}"
                  f"{'是' if t.rework else '':>4}{'是' if not t.qa_start_observed else '':>6}")

    if args.options:
        from planner.model import build_pack
        from planner.scenario import feasible_option_sets
        pack = build_pack(ROOT / "data")
        print()
        print(f"{'方案':<8}{'造价(元)':>10}  {'可行':<4}原因")
        for e in feasible_option_sets(pack):
            name = "+".join(sorted(e["options"])) or "不改造"
            print(f"{name:<8}{e['cost_cents'] / 100:>10.0f}  "
                  f"{'是' if e['feasible'] else '否':<4}"
                  f"{'；'.join(e['violations'])}")


if __name__ == '__main__':
    main()
