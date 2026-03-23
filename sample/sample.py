#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import (  # noqa: F401
    SampleBatch,
    load_sample_batches,
    parse_sample_args,
    run_sample,
)
from main import main as project_main


def parse_args(args: list[str] | None = None):
    return parse_sample_args(args)


def main(argv: list[str] | None = None) -> None:
    payload = ["--mode", "sample"]
    payload.extend(sys.argv[1:] if argv is None else argv)
    project_main(payload)


__all__ = ["SampleBatch", "load_sample_batches", "parse_args", "run_sample", "main"]


if __name__ == "__main__":
    main()
