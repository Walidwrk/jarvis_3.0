"""Blame router for feature attribution (optional utility)."""
from __future__ import annotations

import numpy as np


def simple_attribution(feature_importances: np.ndarray) -> np.ndarray:
    """Return normalized attribution weights."""
    weights = np.abs(feature_importances)
    weights /= weights.sum() + 1e-8
    return weights
