from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class DemoRuntimeImportTests(unittest.TestCase):
    def test_main_module_imports_with_runtime_helpers(self) -> None:
        main = importlib.import_module("main")

        self.assertTrue(hasattr(main, "load_main_runtime"))
        self.assertTrue(hasattr(main, "run_sample"))


class GradioBackendTests(unittest.TestCase):
    def test_run_demo_for_uploaded_video_returns_expected_paths(self) -> None:
        app = importlib.import_module("demo.gradio_app")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_path = tmp_path / "toy.mp4"
            video_path.write_bytes(b"fake-video")

            fake_result = {
                "samples": {
                    "toy": {
                        "video_path": str(tmp_path / "rendered.mp4"),
                        "report_json": str(tmp_path / "report.json"),
                        "mean_fake_prob": 0.875,
                        "num_rendered_frames": 32,
                    }
                }
            }

            with patch("demo.gradio_app.run_demo_pipeline", return_value=fake_result):
                result = app.run_demo_for_uploaded_video(
                    video_path=str(video_path),
                    output_root=tmp_path / "runs",
                    mode="fixed_num_frames",
                    num_frames=32,
                    threshold_override=None,
                )

        self.assertEqual(result["sample_name"], "toy")
        self.assertIn("summary_markdown", result)
        self.assertTrue(result["summary_markdown"].startswith("## Demo Summary"))
        self.assertTrue(result["video_path"].endswith("rendered.mp4"))
        self.assertTrue(result["report_json_path"].endswith("report.json"))


if __name__ == "__main__":
    unittest.main()
