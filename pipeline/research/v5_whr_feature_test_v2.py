#!/usr/bin/env python3
"""Run canonical V5+WHR test while excluding unmatched WHR rows from standalone WHR metrics only."""
import numpy as np
import pipeline.research.v5_whr_feature_test as test

_orig_metrics = test.metrics

def finite_metrics(y, p):
    yy = np.asarray(y)
    pp = np.asarray(p, dtype=float)
    keep = np.isfinite(pp)
    return _orig_metrics(yy[keep], pp[keep])

test.metrics = finite_metrics
test.main()
