from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from demo.annotate import (
    FramePrediction,
    build_sparse_predictions,
    crop_to_vertical_canvas,
    draw_prediction_overlay,
    propagate_predictions,
)


class DemoAnnotationTests(unittest.TestCase):
    def test_propagate_predictions_reuses_latest_sample(self) -> None:
        sparse = {
            2: FramePrediction(frame_index=2, fake_prob=0.8123, pred_label="FAKE"),
            5: FramePrediction(frame_index=5, fake_prob=0.1333, pred_label="REAL"),
        }

        dense = propagate_predictions(frame_count=8, sparse_predictions=sparse)

        self.assertEqual(len(dense), 8)
        self.assertIsNotNone(dense[0])
        self.assertEqual(dense[0].pred_label, "FAKE")
        self.assertEqual(dense[1].pred_label, "FAKE")
        self.assertEqual(dense[2].pred_label, "FAKE")
        self.assertEqual(dense[3].pred_label, "FAKE")
        self.assertEqual(dense[4].pred_label, "FAKE")
        self.assertEqual(dense[5].pred_label, "REAL")
        self.assertEqual(dense[6].pred_label, "REAL")
        self.assertEqual(dense[7].pred_label, "REAL")

    def test_build_sparse_predictions_uses_frame_stem(self) -> None:
        img_paths = np.asarray(
            [
                "/tmp/demo/frames/564_1774777598/002.png",
                "/tmp/demo/frames/564_1774777598/005.png",
            ],
            dtype=object,
        )
        probs = np.asarray([0.8123, 0.1333], dtype=np.float32)

        sparse = build_sparse_predictions(img_paths=img_paths, fake_probs=probs, threshold=0.5)

        self.assertEqual(sorted(sparse.keys()), [2, 5])
        self.assertEqual(sparse[2].pred_label, "FAKE")
        self.assertEqual(sparse[5].pred_label, "REAL")

    def test_crop_to_vertical_canvas_trims_height_to_9_16(self) -> None:
        image = np.arange(1280, dtype=np.uint16).reshape(1280, 1, 1).repeat(592, axis=1).repeat(3, axis=2).astype(np.uint8)
        canvas, offset = crop_to_vertical_canvas(image)

        self.assertEqual(canvas.shape, (1052, 592, 3))
        self.assertEqual(offset, (0, 68))
        restored = image[offset[1] : offset[1] + canvas.shape[0], :, :]
        self.assertTrue(np.array_equal(restored, canvas))

    def test_draw_overlay_keeps_shape_and_writes_box_side_labels(self) -> None:
        image = np.zeros((160, 200, 3), dtype=np.uint8)
        prediction = FramePrediction(frame_index=3, fake_prob=0.8123, pred_label="FAKE")

        rendered = draw_prediction_overlay(
            image,
            prediction=prediction,
            face_box=(50, 60, 120, 135),
            font_scale=0.6,
            thickness=2,
        )

        self.assertEqual(rendered.shape, image.shape)
        self.assertGreater(int(rendered.sum()), 0)
        self.assertTrue(np.any(rendered[36:64, 44:120] != 0))
        self.assertTrue(np.any(rendered[132:155, 44:120] != 0))
        self.assertTrue(np.any(rendered[60:135, 50:120] != 0))
        self.assertFalse(np.any(rendered[0:20, 0:200] != 0))

    def test_draw_overlay_without_face_box_leaves_frame_unchanged(self) -> None:
        image = np.zeros((160, 200, 3), dtype=np.uint8)
        prediction = FramePrediction(frame_index=3, fake_prob=0.8123, pred_label="FAKE")

        rendered = draw_prediction_overlay(
            image,
            prediction=prediction,
            face_box=None,
            font_scale=0.6,
            thickness=2,
        )

        self.assertTrue(np.array_equal(rendered, image))


if __name__ == "__main__":
    unittest.main()
