#!/usr/bin/env python3
"""Stable data-only aggregation entrypoint; never run generated programs."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repro.eval.portable.evaluate import main as portable_main


def main(argv=None):
    return portable_main(['aggregate', *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    main()
