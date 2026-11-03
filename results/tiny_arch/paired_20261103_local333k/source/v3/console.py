"""Reuse the tested terminal geometry handling with V3 labels."""
from scripts.v2_console import Progress as BaseProgress


class Progress(BaseProgress):
    label='V3'
