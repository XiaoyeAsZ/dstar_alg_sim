"""Minimal vendored image-resizing utilities used by FID."""

from .resizers import Resizer, build_resizer

__all__ = ["Resizer", "build_resizer"]
