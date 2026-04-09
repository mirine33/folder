from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


Point = Tuple[float, float]


@dataclass
class LandmarkSet:
    points: Dict[str, Point]

    def get(self, name: str) -> Point:
        if name not in self.points:
            raise KeyError(f"Missing landmark: {name}")
        return self.points[name]


def to_point(x: float, y: float) -> Point:
    return float(x), float(y)


def midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def distance(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def angle_abc(a: Point, b: Point, c: Point) -> float:
    """
    Angle (degrees) at point b formed by a-b-c.
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm < 1e-9 or bc_norm < 1e-9:
        return 0.0
    cos_theta = float(np.dot(ba, bc) / (ba_norm * bc_norm))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def vertical_deviation(a: Point, b: Point) -> float:
    """
    X-axis difference from b to a.
    Positive => a is right of b, negative => a is left of b.
    """
    return float(a[0] - b[0])


def rolling_mean(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def rolling_std(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.std(arr))

