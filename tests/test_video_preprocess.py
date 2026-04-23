from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from demo.video_preprocess import (
    build_uncertainty_refined_indices,
    process_video_raw,
    sample_frame_indices,
)


class VideoPreprocessTests(unittest.TestCase):
    def test_uncertainty_refinement_adds_points_inside_uncertain_gap(self) -> None:
        refined = build_uncertainty_refined_indices(
            frame_count=21,
            sampled_indices=[0, 10, 20],
            fake_probs=np.array([0.01, 0.80, 0.85], dtype=np.float32),
            target_num_frames=5,
            threshold=0.2,
        )

        self.assertEqual(refined[:3], [0, 2, 3])
        self.assertEqual(refined[-2:], [10, 20])

    def test_uncertainty_refinement_returns_original_when_budget_not_larger(self) -> None:
        refined = build_uncertainty_refined_indices(
            frame_count=6,
            sampled_indices=[0, 3, 5],
            fake_probs=np.array([0.1, 0.9, 0.8], dtype=np.float32),
            target_num_frames=3,
            threshold=0.2,
        )

        self.assertEqual(refined, [0, 3, 5])

    def test_content_change_sampling_falls_back_to_uniform_when_static(self) -> None:
        sampled = sample_frame_indices(
            frame_count=5,
            mode="content_change",
            num_frames=3,
            change_scores=np.zeros(4, dtype=np.float32),
        )

        self.assertEqual(sampled, [0, 2, 4])

    def test_content_change_sampling_biases_towards_later_motion(self) -> None:
        sampled = sample_frame_indices(
            frame_count=6,
            mode="content_change",
            num_frames=3,
            change_scores=np.array([0.0, 0.0, 10.0, 10.0, 10.0], dtype=np.float32),
        )

        self.assertEqual(sampled, [0, 3, 5])

    def test_process_video_raw_saves_sampled_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_path = tmp_path / "toy.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (32, 24),
            )
            for frame_index in range(5):
                frame = np.full((24, 32, 3), frame_index * 20, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            summary = process_video_raw(
                video_path=video_path,
                output_root=tmp_path / "out",
                mode="fixed_num_frames",
                num_frames=3,
            )

            self.assertEqual(summary.frame_count, 5)
            self.assertEqual(summary.sampled_indices, [0, 2, 4])
            self.assertEqual(summary.saved_count, 3)
            self.assertEqual(summary.failed_count, 0)
            self.assertEqual(
                [path.name for path in sorted(summary.frames_dir.glob("*.png"))],
                ["000000.png", "000002.png", "000004.png"],
            )


if __name__ == "__main__":
    unittest.main()
