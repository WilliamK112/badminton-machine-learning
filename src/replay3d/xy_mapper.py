from __future__ import annotations

import numpy as np

from .schema import CourtSpec


def build_homography_from_corners(
    image_corners_xy: list[list[float]],
    court: CourtSpec,
) -> np.ndarray:
    """
    Build a projective transform H from image pixels -> court meters.

    image_corners_xy order: [bottom_left, bottom_right, top_right, top_left]
    """
    src = np.asarray(image_corners_xy, dtype=float)
    if src.shape != (4, 2):
        raise ValueError("image_corners_xy must be a 4x2 array")

    dst = np.asarray(
        [
            [0.0, 0.0],
            [court.width_m, 0.0],
            [court.width_m, court.length_m],
            [0.0, court.length_m],
        ],
        dtype=float,
    )

    # DLT solve for H (3x3)
    a_rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src, dst):
        a_rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        a_rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])

    a = np.asarray(a_rows, dtype=float)
    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    h = h / h[2, 2]
    return h


def map_image_point_to_court(point_xy: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    """Map one image-space XY point (pixels) to court-space XY point (meters)."""
    x, y = point_xy
    p = np.asarray([x, y, 1.0], dtype=float)
    q = homography @ p
    if abs(q[2]) < 1e-8:
        raise ValueError("homography mapping produced invalid scale")
    return float(q[0] / q[2]), float(q[1] / q[2])
