"""Convenience entry point.

Lets the pipeline be run as either::

    python run_pipeline.py

or::

    python -m src.run_pipeline

The real implementation lives in :mod:`src.run_pipeline`; this file only puts the
project root on ``sys.path`` so the ``src`` package resolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_pipeline import main  # noqa: E402  (must follow the sys.path insert)

if __name__ == "__main__":
    main()
