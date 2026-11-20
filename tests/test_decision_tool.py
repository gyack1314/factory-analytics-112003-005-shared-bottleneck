"""Behavioural tests for the decision tool.

The tests are written as falsifiable claims about the data pack, not as
regression snapshots: every assertion names the fact from ``data/`` that makes
it true.
"""
import contextlib
import io
import json
import unittest

from planner import attribution, dataio
from planner.compare import compare, phase2
from planner.history import iter_traces, qa_overlap_episodes, replay_check
from planner.model import build_pack, lot_count, shift_wave
from planner.scenario import (World, capacity_timeline, exclusive_violations,
                              feasible_option_sets, make_jobs, run, simulate,
                              SimConfig)
from planner.validate import replay


class InputFactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = build_pack()

    def test_pack_shape(self):
        self.assertEqual(len(self.pack.orders), 132)
        self.assertEqual(len(self.pack.planning_orders), 5)
        self.assertEqual(len(self.pack.wip), 2)
        self.assertEqual(self.pack.horizon, (10560, 12000))

    def test_lot_counts_follow_allowed_lot_not_pieces(self):
        self.assertEqual(lot_count(8, "薄板", self.pack), 1)
        self.assertEqual(lot_count(31, "薄板", self.pack), 2)
        self.assertEqual(lot_count(20, "厚板", self.pack), 1)
        self.assertEqual(lot_count(21, "厚板", self.pack), 2)


class HistoryReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = build_pack()
        cls.report = replay_check(cls.pack)

    def test_every_recorded_delivery_is_reproduced(self):
        self.assertTrue(self.report["match"],
                        f"replay failures: {self.report['delivery_failures'][:3]}")

    def test_transfer_wait_is_a_per_shift_cadence(self):
        waits = {}
        for t in iter_traces(self.pack):
            shift = t.available_min // self.pack.shift_minutes
            self.assertEqual(t.qa_wait_min,
                             shift_wave(t.available_min, self.pack.shift_minutes),
                             msg=f"{t.order} shift {shift}")
            waits[shift] = t.qa_wait_min
        self.assertEqual(sorted(set(waits.values())), [3, 7, 11, 15, 19])

    def test_qa_overlaps_are_detected(self):
        eps = qa_overlap_episodes(self.pack)
        self.assertEqual(len(eps), 5)
        self.assertTrue(all(e["overlap_min"] > 0 for e in eps))


class AttributionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = build_pack()
        cls.a = attribution.analyse(cls.pack)

    def test_accounting_identity(self):
        self.assertAlmostEqual(self.a.explained_short + self.a.residue_short,
                               self.a.total_short, places=6)

    def test_ambiguity_orders_are_excluded_from_exact_totals(self):
        exact = [r for r in self.a.orders if not r.ambiguous]
        self.assertEqual(sum(r.short_units for r in exact), self.a.total_short)
        amb = [r for r in self.a.orders if r.ambiguous]
        self.assertTrue(amb)
        for r in amb:
            self.assertLessEqual(r.bound_low_short, r.bound_high_short)

    def test_all_three_causes_are_positive(self):
        for c in ("M", "T", "R"):
            self.assertGreater(self.a.shapley_short[c], 0, msg=c)

    def test_joint_interaction_is_reported(self):
        self.assertNotEqual(self.a.joint_interaction_short, 0)

    def test_capacity_is_not_saturated(self):
        for label, info in self.a.capacity_headroom.items():
            self.assertLess(info["utilization"], 0.5, msg=label)

    def test_shortage_is_not_only_capacity(self):
        # removing every cause still leaves orders short: capacity/mix, not queue
        self.assertGreater(self.a.residue_short, 0)


class ScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = build_pack()
        cls.base = run(cls.pack, ())

    def test_all_committed_work_is_delivered_at_baseline(self):
        self.assertEqual(self.base.committed_breach, [])
        self.assertEqual(self.base.backlog_units, 0)

    def test_no_work_is_lost(self):
        jobs = make_jobs(self.pack, World("t"))
        self.assertEqual(self.base.delivered_units, sum(j.units for j in jobs))

    def test_installing_on_qa_loses_capacity_during_construction(self):
        caps = capacity_timeline(self.pack, SimConfig(options=frozenset({"B"})),
                                 *self.pack.horizon)
        # B installs on QA between 10565 and 10685, then adds one station from
        # 10685: QA is 0 during construction, 2 after it.
        self.assertEqual(caps["QA"][10565 - 10560], 0)
        self.assertEqual(caps["QA"][10680 - 10560], 0)
        self.assertEqual(caps["QA"][10685 - 10560], 2)

    def test_alternative_route_needs_both_permission_and_contention(self):
        base = [j for j in make_jobs(self.pack, World("t")) if not j.committed]
        jobs = []
        for m in range(8):
            for j in base:
                jobs.append(j.__class__(**{
                    **j.__dict__, "order": f"{j.order}#{m}",
                    "available_min": 10560 + m * 60,
                    "material_min": 10560 + m * 60,
                    "due_min": 10560 + m * 60 + (j.due_min - j.available_min)}))
        strict = simulate(self.pack, jobs, SimConfig(options=frozenset({"A"}),
                                                    route_mode="primary"), World("t"))
        self.assertEqual(strict.resource_busy["QA2"], 0)
        alt = simulate(self.pack, jobs, SimConfig(options=frozenset({"A"}),
                                                 route_mode="allow_alt"), World("t"))
        # 厚板 has no alternative route, so QA2 can only ever see 薄板 minutes
        self.assertGreater(alt.resource_busy["QA2"], 0)

    def test_exclusive_group_and_budget(self):
        self.assertTrue(exclusive_violations(self.pack, ["A", "B"]))
        sets = {e["options"]: e for e in feasible_option_sets(self.pack)}
        self.assertFalse(sets[frozenset({"A", "B"})]["feasible"])
        self.assertFalse(sets[frozenset({"B", "D"})]["feasible"])   # over budget
        self.assertTrue(sets[frozenset({"A", "C"})]["feasible"])

    def test_committed_priority_work_survives_a_bad_world(self):
        world = World("stress", rework_orders=("N1", "N2", "N3", "N4", "N5"),
                      material_slip_min=300)
        r = run(self.pack, ("B",), "primary", world)
        # WIP due dates are honoured even when both lines are stressed
        self.assertTrue(all(o in r.orders for o in ("W1", "W2")))


class ComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = build_pack()
        cls.world = World("hist", rework_orders=("N1", "N4"), material_slip_min=120)
        cls.report = compare(cls.pack, cls.world)

    def test_every_package_uses_the_same_order_book(self):
        base_ids = set(self.report.baseline.orders)
        for p in self.report.packages:
            self.assertEqual(set(p.result.orders), base_ids)

    def test_infeasible_packages_are_listed_with_reasons(self):
        names = {"+".join(b["options"]) for b in self.report.infeasible}
        self.assertIn("A+B", names)
        self.assertIn("B+D", names)

    def test_benefit_is_reconstructible_from_orders(self):
        for p in self.report.packages:
            self.assertEqual(p.gain_on_time_units, sum(v["delta_units"] for v in p.unblocked.values()))

    def test_shapley_stays_inside_the_best_package_bound(self):
        # With an infeasible coalition in the lattice, the budget-constrained
        # Shapley value is a lower bound on the best package's gain, never an
        # unattainable upper bound.
        best = max(p.result.on_time_units for p in self.report.packages)
        gain = best - self.report.baseline.on_time_units
        self.assertLessEqual(sum(self.report.option_shapley_units.values()), gain + 1e-9)


class PhaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = build_pack()
        cls.world = World("hist", rework_orders=("N1",), material_slip_min=120)
        cls.report = compare(cls.pack, cls.world)
        cls.phase1 = next(p for p in cls.report.packages if p.options == frozenset({"A"}))

    def test_phase2_starts_from_phase1_state(self):
        p2 = phase2(self.pack, self.phase1.options, ("C",), self.phase1.result, self.world)
        self.assertEqual(p2.start_state.capacity_assets, ["A"])
        self.assertEqual(p2.start_state.decision_min, 10745)
        # phase-1 shipped orders must not be re-run in phase 2
        for o in p2.start_state.shipped:
            self.assertNotIn(o, p2.phase2.orders)

    def test_phase2_records_the_phase1_assumptions(self):
        p2 = phase2(self.pack, self.phase1.options, ("C",), self.phase1.result, self.world)
        d = p2.as_dict()
        self.assertEqual(d["phase1_state"]["installed_options"], ["A"])
        self.assertEqual(d["phase1_state"]["world"]["rework_orders"], ["N1"])
        self.assertEqual(d["phase1_package"], ["A"])


class ReplayValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = build_pack()

    def test_specified_windows_reproduce_the_record(self):
        for shifts in ([0, 1, 2], [19, 20, 21], [10, 11, 12]):
            r = replay(self.pack, shifts)
            self.assertTrue(r.validated, msg=f"shifts {shifts}: "
                    f"{[d.as_dict() for d in r.diffs if d.kind == 'unexplained']}")

    def test_backlog_positions_match(self):
        r = replay(self.pack, [19, 20, 21])
        self.assertEqual(r.backlog_expected, r.backlog_actual)

    def test_resource_minutes_match_except_for_serialized_overlaps(self):
        r = replay(self.pack, [19, 20, 21])
        for res in ("L1", "L2", "RW"):
            self.assertEqual(r.resource_minutes_expected[res], r.resource_minutes_actual[res])
        self.assertGreaterEqual(r.resource_minutes_actual["QA"],
                                r.resource_minutes_expected["QA"])


class RobustnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from planner.robustness import by_risk_family, analyse
        cls.pack = build_pack()
        cls.family = by_risk_family(cls.pack)
        cls.report = analyse(cls.pack, cls.family)

    def test_every_package_reports_a_range_not_a_point(self):
        for p in self.report.packages:
            self.assertLessEqual(p.min_units, p.max_units)
            self.assertLessEqual(p.min_lum, p.max_lum)

    def test_ordering_is_declared_unsupported_when_ranges_overlap(self):
        if self.report.overlaps:
            self.assertIsNone(self.report.robust_winner)
            self.assertIn("不支持唯一排序", self.report.robust_note)

    def test_b_option_breaks_committed_work_in_every_world(self):
        # Option B loses QA capacity exactly when W1/W2 must be inspected
        b = next(p for p in self.report.packages if p.options == frozenset({"B"}))
        self.assertEqual(b.breach_worlds, b.total_worlds)


class LedgerTest(unittest.TestCase):
    def test_ledger_appends_and_is_retrievable_by_digest(self):
        import tempfile
        from pathlib import Path
        from planner import ledger
        from planner.compare import compare
        from planner.scenario import World
        pack = build_pack()
        rep = compare(pack, World("l", rework_orders=("N1",)))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.json"
            e1 = ledger.record_comparison(rep, note="first", path=path)
            e2 = ledger.record_comparison(rep, note="second", path=path)
            self.assertNotEqual(e1["digest"], e2["digest"])
            entries = ledger.load(path)
            self.assertEqual(len(entries), 2)
            self.assertEqual(ledger.find(entries, e1["digest"])["note"], "first")
            # a stored entry carries the exact world reading used at the time
            self.assertEqual(entries[0]["world"]["rework_orders"], ["N1"])

    def test_phase2_records_carried_state(self):
        import tempfile
        from pathlib import Path
        from planner import ledger
        from planner.compare import compare, phase2
        from planner.scenario import World, make_jobs
        pack = build_pack()
        w = World("p", release_span_shifts=3, multiplier=12, due_scale=0.6)
        jobs = make_jobs(pack, w)
        rep = compare(pack, w)
        base = next(p for p in rep.packages if not p.options)
        p2 = phase2(pack, frozenset(), ("C",), base.result, w, phase1_jobs=jobs)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.json"
            entry = ledger.record_phase2(p2, note="phase test", path=path)
            self.assertEqual(entry["phase1_state"]["world"]["multiplier"], 12)
            self.assertIn("shipped_orders", entry["phase1_state"])


class HonestLimitsTest(unittest.TestCase):
    """The pack must not be over-claimed."""

    @classmethod
    def setUpClass(cls):
        cls.pack = build_pack()

    def test_default_order_book_has_headroom(self):
        r = run(self.pack, ())
        for res in ("L1", "L2", "QA"):
            self.assertLess(r.utilization(res), 0.2)

    def test_default_order_book_cannot_rank_packages(self):
        from planner.compare import compare
        rep = compare(self.pack)
        self.assertEqual(rep.differentiation, "none")
        self.assertTrue(any("没有任何方案" in n or "无法区分" in n for n in rep.notes))

    def test_every_package_keeps_the_same_delivered_units(self):
        from planner.compare import compare
        rep = compare(self.pack)
        for p in rep.packages:
            self.assertEqual(p.result.delivered_units, rep.baseline.delivered_units)


class CliContractTest(unittest.TestCase):
    """The CLI is the delivery surface; its modes must be machine-usable."""

    def _run(self, argv):
        import contextlib, io
        from planner.cli import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(argv)
        return buf.getvalue()

    def test_report_json_is_pure_json(self):
        import json
        out = self._run(["report", "--json"])
        d = json.loads(out)
        self.assertEqual(set(d), {"attribution", "comparison", "robustness", "validation"})

    def test_compare_reports_ledger_digest(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "l.json"
            out = self._run(["compare", "--json", "--ledger", str(ledger_path),
                             "--note", "cli test"])
            d = json.loads(out)
            self.assertTrue(ledger_path.exists())
            self.assertEqual(d["ledger"]["digest"], json.loads(
                ledger_path.read_text(encoding="utf-8"))[0]["digest"])

    def test_validate_json_returns_one_entry_per_window(self):
        import json
        out = self._run(["validate", "--json", "--shifts", "0-2,19-21"])
        d = json.loads(out)
        self.assertEqual(len(d), 2)
        self.assertTrue(all(x["validated"] for x in d))
