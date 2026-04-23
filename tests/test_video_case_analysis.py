from __future__ import annotations

import unittest

from src.eval.video_case_analysis import build_route_segments, parse_variant_name


class VideoCaseAnalysisTests(unittest.TestCase):
    def test_parse_variant_name_uses_parent_run_dir(self) -> None:
        path = "/tmp/579_1775983303_no_fr_64f_t02_content/rendered/579_1775983303/report.json"
        self.assertEqual(parse_variant_name(path), "579_1775983303_no_fr_64f_t02_content")

    def test_build_route_segments_groups_consecutive_routes(self) -> None:
        rows = [
            {"frame_index": 0, "route_top1": "REAL", "route_meta_prob": 0.01},
            {"frame_index": 5, "route_top1": "REAL", "route_meta_prob": 0.02},
            {"frame_index": 10, "route_top1": "FS", "route_meta_prob": 0.8},
            {"frame_index": 12, "route_top1": "FS", "route_meta_prob": 0.9},
            {"frame_index": 20, "route_top1": "REAL", "route_meta_prob": 0.1},
        ]

        segs = build_route_segments(rows)

        self.assertEqual(len(segs), 3)
        self.assertEqual(segs[0]["route_top1"], "REAL")
        self.assertEqual(segs[0]["start_frame"], 0)
        self.assertEqual(segs[0]["end_frame"], 5)
        self.assertAlmostEqual(segs[1]["mean_prob"], 0.85)
        self.assertEqual(segs[2]["length"], 1)


if __name__ == "__main__":
    unittest.main()
