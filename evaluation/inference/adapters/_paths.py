"""Artifact path validation shared by the portable generation adapters."""
from pathlib import Path


def contained(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Artifact path must be contained and relative')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Missing or escaping artifact')
    return path
