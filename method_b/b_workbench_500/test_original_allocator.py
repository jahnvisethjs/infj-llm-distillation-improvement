"""Focused checks for scientific selection controls; no models or GPUs."""
import copy
import json
import random
import unittest
from pathlib import Path
from approach_b_selector import AllocationError, allocate, apportion, random_priorities


def fixture():
    return [{"row_id": f"{source}-{index}", "problem_group_id": f"group-{source}-{index}",
             "dataset_id": source, "length_bin": 0, "supervised_tokens": tokens}
            for source in ["a", "b"] for index, tokens in enumerate([3, 17])]


class AllocatorTests(unittest.TestCase):
    def test_whole_row_swap_matches_exact_budget(self):
        rows = fixture()
        priorities = {row["row_id"]: int(row["row_id"].split("-")[1]) for row in rows}
        selected, report = allocate(rows, priorities, {("a", 0): 1, ("b", 0): 1}, 20, "0")
        self.assertEqual(sum(row["supervised_tokens"] for row in selected), 20)
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["dataset_id"] for row in selected}, {"a", "b"})
        self.assertEqual(report["swap_count"], 1)
        self.assertTrue(all(row in rows for row in selected))

    def test_infeasible_search_does_not_widen_tolerance(self):
        rows = fixture()
        with self.assertRaises(AllocationError) as caught:
            allocate(rows, random_priorities(rows, 42), {("a", 0): 1, ("b", 0): 1}, 7, "0")
        report = caught.exception.report
        self.assertEqual(report["allowed_integer_token_range"], [7, 7])
        self.assertFalse(report["tolerance_relaxed"])
        self.assertIsNotNone(report["nearest_attempt"])
        self.assertFalse(report["infeasibility_proven"])

    def test_provably_insufficient_capacity(self):
        rows = [row for row in fixture() if row["supervised_tokens"] == 3]
        with self.assertRaises(AllocationError) as caught:
            allocate(rows, random_priorities(rows, 42), {("a", 0): 1, ("b", 0): 1}, 100, "0.005")
        self.assertTrue(caught.exception.report["infeasibility_proven"])

    def test_quota_capacity_failure_is_explicit(self):
        with self.assertRaises(AllocationError):
            apportion({("a", 0): 3, ("b", 0): 1}, 4, {("a", 0): 2, ("b", 0): 5}, 1729)

    def test_partial_score_coverage_cannot_shrink_pool(self):
        rows = fixture()
        priorities = random_priorities(rows, 42)
        priorities.pop(rows[0]["row_id"])
        with self.assertRaisesRegex(ValueError, "whole common candidate universe"):
            allocate(rows, priorities, {("a", 0): 1, ("b", 0): 1}, 20)

    def test_ties_and_row_reordering_are_deterministic(self):
        rows = fixture()
        priorities = {row["row_id"]: 0 for row in rows}
        first, _ = allocate(rows, priorities, {("a", 0): 1, ("b", 0): 1}, 20, "0")
        second, _ = allocate(list(reversed(rows)), priorities, {("b", 0): 1, ("a", 0): 1}, 20, "0")
        self.assertEqual(first, second)

    def test_student_metadata_does_not_affect_B_priorities(self):
        rows = fixture()
        modified = copy.deepcopy(rows)
        for i, row in enumerate(modified):
            row.update(student_loss=9999-i, mistake=i % 2, embedding_cluster=i)
        self.assertEqual(random_priorities(rows, 42), random_priorities(modified, 42))

    def test_bounded_row_count_adjustment_is_recorded(self):
        rows = [{"row_id": f"{source}-{i}", "problem_group_id": f"g-{source}-{i}",
                 "dataset_id": source, "length_bin": 0, "supervised_tokens": tokens}
                for source in ["a", "b"] for i, tokens in enumerate([4, 5, 6, 9])]
        priorities = {row["row_id"]: int(row["row_id"].split("-")[1]) for row in rows}
        selected, report = allocate(rows, priorities, {("a", 0): 1, ("b", 0): 1}, 18, "0", max_swaps=0)
        self.assertEqual(len(selected), 4)
        self.assertEqual(report["estimated_rows"], 3)
        self.assertEqual(report["count_adjustment_from_estimate"], 1)


if __name__ == "__main__":
    unittest.main()
