from __future__ import annotations

import unittest

import numpy as np

from src.eval.upstream_analysis import (
    compute_anchor_tree_layout,
    compute_residual_diagnostics,
    DEFAULT_ACTIVE_LOSS_WEIGHTS,
    append_anchor_points,
    build_loss_sensitivity_jobs,
    choose_balanced_indices,
    filter_supported_labels,
    project_points,
    subset_loaded_split,
)


class UpstreamAnalysisTests(unittest.TestCase):
    def test_build_loss_sensitivity_jobs_adds_single_baseline_and_perturbations(self) -> None:
        jobs = build_loss_sensitivity_jobs(
            DEFAULT_ACTIVE_LOSS_WEIGHTS,
            loss_names=["lambda_real", "lambda_pair"],
            scales=[0.0, 1.0, 2.0],
        )

        self.assertEqual(jobs[0]["job_name"], "baseline")
        self.assertEqual(jobs[0]["kind"], "baseline")
        self.assertEqual(len(jobs), 5)

        perturb_names = [job["job_name"] for job in jobs[1:]]
        self.assertEqual(
            perturb_names,
            [
                "lambda_real_x0p00",
                "lambda_real_x2p00",
                "lambda_pair_x0p00",
                "lambda_pair_x2p00",
            ],
        )
        self.assertEqual(jobs[1]["weights"]["lambda_real"], 0.0)
        self.assertEqual(jobs[-1]["weights"]["lambda_pair"], 1.5)
        self.assertEqual(jobs[-1]["weights"]["lambda_real"], 1.0)

    def test_choose_balanced_indices_caps_each_label(self) -> None:
        labels = np.asarray(["REAL", "REAL", "FS", "FS", "FS", "FE"], dtype=object)

        indices = choose_balanced_indices(labels, max_per_label=2, seed=7)

        selected = labels[indices]
        self.assertEqual(int(np.sum(selected == "REAL")), 2)
        self.assertEqual(int(np.sum(selected == "FS")), 2)
        self.assertEqual(int(np.sum(selected == "FE")), 1)

    def test_append_anchor_points_combines_samples_and_anchors(self) -> None:
        sample_points = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        sample_labels = np.asarray(["REAL", "FS"], dtype=object)
        anchor_points = np.asarray([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
        anchor_labels = ["REAL", "FS"]

        merged_points, merged_labels, merged_kinds = append_anchor_points(
            sample_points=sample_points,
            sample_labels=sample_labels,
            anchor_points=anchor_points,
            anchor_labels=anchor_labels,
        )

        self.assertEqual(merged_points.shape, (4, 2))
        self.assertEqual(merged_labels.tolist(), ["REAL", "FS", "REAL", "FS"])
        self.assertEqual(merged_kinds.tolist(), ["sample", "sample", "anchor", "anchor"])

    def test_subset_loaded_split_selects_requested_rows(self) -> None:
        cls = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
        labels = np.asarray(["REAL", "FS", "FE"], dtype=object)
        methods = np.asarray(["m0", "m1", "m2"], dtype=object)

        subset = subset_loaded_split(
            cls=cls,
            labels=labels,
            methods=methods,
            indices=np.asarray([2, 0], dtype=np.int64),
        )

        self.assertEqual(subset.cls[:, 0].tolist(), [3.0, 1.0])
        self.assertEqual(subset.labels.tolist(), ["FE", "REAL"])
        self.assertEqual(subset.methods.tolist(), ["m2", "m0"])

    def test_project_points_rejects_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            project_points(np.zeros((4, 3), dtype=np.float32), method="bogus", seed=42)

    def test_project_points_pca_returns_2d_projection(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        coords = project_points(points, method="pca", seed=42)

        self.assertEqual(coords.shape, (4, 2))

    def test_filter_supported_labels_drops_unknown_classes(self) -> None:
        cls = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
        labels = np.asarray(["REAL", "FR", "FS"], dtype=object)
        methods = np.asarray(["m0", "m1", "m2"], dtype=object)

        subset = filter_supported_labels(
            subset_loaded_split(
                cls=cls,
                labels=labels,
                methods=methods,
                indices=np.arange(3, dtype=np.int64),
            ),
            supported_labels=["REAL", "FS"],
        )

        self.assertEqual(subset.labels.tolist(), ["REAL", "FS"])
        self.assertEqual(subset.cls[:, 0].tolist(), [1.0, 3.0])

    def test_compute_residual_diagnostics_uses_true_label_column(self) -> None:
        labels = np.asarray(["REAL", "FS"], dtype=object)
        model_labels = ["EFS", "FS", "REAL"]
        dists = np.asarray(
            [
                [4.0, 3.0, 1.0],
                [5.0, 2.0, 6.0],
            ],
            dtype=np.float32,
        )

        rows = compute_residual_diagnostics(labels=labels, dists=dists, model_labels=model_labels)

        self.assertEqual(rows[0]["label"], "REAL")
        self.assertEqual(rows[0]["true_residual"], 1.0)
        self.assertEqual(rows[0]["nearest_other_residual"], 3.0)
        self.assertEqual(rows[0]["margin"], 2.0)
        self.assertEqual(rows[1]["label"], "FS")
        self.assertEqual(rows[1]["true_residual"], 2.0)

    def test_compute_anchor_tree_layout_places_real_at_origin(self) -> None:
        anchor_points = np.asarray(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 3.0],
            ],
            dtype=np.float32,
        )
        anchor_labels = ["REAL", "FS", "FE"]

        layout = compute_anchor_tree_layout(anchor_points=anchor_points, anchor_labels=anchor_labels, root_label="REAL")

        self.assertEqual(layout["REAL"], (0.0, 0.0))
        self.assertIn("FS", layout)
        self.assertIn("FE", layout)


if __name__ == "__main__":
    unittest.main()
