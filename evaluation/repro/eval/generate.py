#!/usr/bin/env python3
"""Stable generate entrypoint; dry-run unless execution is explicitly requested."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repro.eval.portable.reproduce import main as portable_main


def main(argv=None):
    return portable_main(['generate', *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    main()
