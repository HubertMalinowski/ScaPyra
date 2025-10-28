import numpy as np
from numpy.typing import NDArray
from typing import Optional


def find_circle_intersections(
    p1: NDArray[np.floating],
    r1: float,
    p2: NDArray[np.floating],
    r2: float
) -> Optional[np.ndarray]:
    """
    Compute the intersection points of two circles in 2D.

    Parameters
    ----------
    p1 : array-like of shape (2,)
        Center of the first circle [x1, y1].
    r1 : float
        Radius of the first circle.
    p2 : array-like of shape (2,)
        Center of the second circle [x2, y2].
    r2 : float
        Radius of the second circle.

    Returns
    -------
    np.ndarray | None
        Array of shape (2, 2):
        [[xA, yA],
         [xB, yB]]
        or None if there is no valid intersection.
    """
    # Ensure numpy arrays of shape (2,)
    p1 = np.asarray(p1, dtype=float).reshape(-1)
    p2 = np.asarray(p2, dtype=float).reshape(-1)

    if p1.shape[0] != 2 or p2.shape[0] != 2:
        return None

    d = np.linalg.norm(p2 - p1)

    eps = 1e-9
    if d > (r1 + r2) or d < abs(r1 - r2) or d < eps:
        return None

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h_sq = r1**2 - a**2

    # numerical safety for near-tangent case
    if h_sq < 0:
        if h_sq > -1e-9:
            h_sq = 0.0
        else:
            return None

    h = np.sqrt(h_sq)

    # point along the line between centers
    p3 = p1 + a * (p2 - p1) / d

    # perpendicular offset
    offset = h * np.array([-(p2[1] - p1[1]), (p2[0] - p1[0])]) / d

    intersections = np.stack((p3 + offset, p3 - offset))
    return intersections.astype(float)


def select_intersection_point(
    intersections: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """
    Selects one of two intersection points returned by find_circle_intersections().

    Parameters
    ----------
    intersections : np.ndarray | None
        Array of shape (2, 2) containing two intersection points, or None.

    Returns
    -------
    np.ndarray | None
        Selected intersection point as array [x, y], or None if input is invalid.
    """
    if intersections is None or not isinstance(intersections, np.ndarray):
        return None

    if intersections.shape != (2, 2):
        return None

    idx = np.argmin(intersections[:, 0])
    return intersections[idx]

def calculate_angle(
    point: NDArray[np.floating],
    intersection: NDArray[np.floating]
) -> Optional[float]:
    """
    Computes the absolute angle (in degrees) between a reference point and an intersection point.

    Parameters
    ----------
    point : np.ndarray
        Array [x, y] representing the reference point.
    intersection : np.ndarray
        Array [x, y] representing the target point.

    Returns
    -------
    float | None
        Angle in degrees (0-360), or None if input is invalid.
    """
    if (
        point is None
        or intersection is None
        or not isinstance(point, np.ndarray)
        or not isinstance(intersection, np.ndarray)
    ):
        return None

    if point.shape != (2,) or intersection.shape != (2,):
        return None

    dx = intersection[0] - point[0]
    dy = intersection[1] - point[1]

    angle = np.degrees(np.arctan2(dy, dx))
    if np.isnan(angle):
        return None

    # Normalize angle to [0, 360)
    if angle < 0:
        angle += 360.0

    return float(angle)
    