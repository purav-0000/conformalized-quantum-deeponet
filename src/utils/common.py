import numpy as np
import os


def apply_overrides(cfg, overrides):
    """Allow overrides."""
    for kv in overrides or []:
        key, val = kv.split("=")
        # Infer type from config field
        orig_val = getattr(cfg, key)

        # Check if it's a comma-separated list
        if "," in val:
            try:
                # Try to parse as list of floats or ints
                elem_type = type(orig_val[0]) if isinstance(orig_val, list) and orig_val else float
                val = [elem_type(v) for v in val.split(",")]
            except Exception:
                pass  # Fallback
        elif isinstance(orig_val, bool):
            val = val.lower() in ("true", "1")
        elif isinstance(orig_val, float):
            val = float(val)
        elif isinstance(orig_val, int):
            val = int(val)
        elif val.lower() == "null":
            val = None
        setattr(cfg, key, val)
