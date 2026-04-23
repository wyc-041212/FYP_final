from __future__ import annotations

import importlib.util
import types
import unittest
from pathlib import Path

_SPEC = importlib.util.find_spec("scripts.run_upstream_loss_sensitivity")
if _SPEC is not None:
    from scripts.run_upstream_loss_sensitivity import build_train_command
else:
    build_train_command = None


class RunUpstreamLossSensitivityTests(unittest.TestCase):
    @unittest.skipIf(build_train_command is None, "Archived non-within loss-sensitivity script is not part of the active repo.")
    def test_build_train_command_includes_no_fr_flag(self) -> None:
        args = types.SimpleNamespace(
            python_bin="python",
            cache_root=Path("/tmp/cache"),
            train_split="DF40_train",
            test_split="DF40_test_ff",
            ood_split="DF40_test_ood",
            device="cpu",
            epochs=2,
            linear_warmup_epochs=1,
            patience=1,
            batch_size=8,
            eval_batch_size=16,
            lr=1e-3,
            seed=42,
            real_rank=40,
            efs_rank=24,
            alpha=1.0,
            temperature=0.1,
            real_class_multiplier=2.0,
            selection_real_weight=0.5,
            no_fr=True,
        )
        job = {"weights": {"lambda_real": 1.0}, "job_name": "baseline"}

        cmd = build_train_command(args, job, Path("/tmp/out"))

        self.assertIn("--no-fr", cmd)


if __name__ == "__main__":
    unittest.main()
