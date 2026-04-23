from __future__ import annotations

import unittest
from unittest.mock import patch

from src.prepare.huggingface_compat import patched_version


class HuggingFaceCompatTests(unittest.TestCase):
    def test_patched_version_keeps_modern_hub_version(self) -> None:
        with patch("src.prepare.huggingface_compat._ORIG_VERSION", return_value="1.3.5"):
            self.assertEqual(patched_version("huggingface-hub"), "1.3.5")

    def test_patched_version_upgrades_legacy_hub_version(self) -> None:
        with patch("src.prepare.huggingface_compat._ORIG_VERSION", return_value="0.34.0"):
            self.assertEqual(patched_version("huggingface-hub"), "1.3.0")


if __name__ == "__main__":
    unittest.main()
