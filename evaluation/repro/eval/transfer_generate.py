#!/usr/bin/env python3
"""Local transfer generation entrypoint, dry-run by default; never score code."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference.adapters.transfer import main


if __name__ == "__main__":
    main()
