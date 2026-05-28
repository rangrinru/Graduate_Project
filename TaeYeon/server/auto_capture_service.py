import math


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def point_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_eye_aspect_ratio_from_points(eye_pts):
    p1, p2, p3, p4, p5, p6 = eye_pts
    vertical_1 = point_distance(p2, p6)
    vertical_2 = point_distance(p3, p5)
    horizontal = point_distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)
