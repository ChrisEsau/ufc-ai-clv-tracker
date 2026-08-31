#!/usr/bin/env python3
"""Run canonical V5+WHR test, excluding unmatched WHR rows only from standalone WHR metrics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

base_path = Path(__file__).with_name("v5_whr_feature_test.py")
spec = importlib.util.spec_from_file_location("v5_whr_feature_test_base", base_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {base_path}")
test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test)

_orig_metrics = test.metrics

def finite_metrics(y, p):
    yy = np.asarray(y)
    pp = np.asarray(p, dtype=float)
    keep = np.isfinite(pp)
    return _orig_metrics(yy[keep], pp[keep])

test.metrics = finite_metrics
test.main()
