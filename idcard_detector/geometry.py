import math


KEYPOINT_NAMES = ("K0", "K1", "K2", "K3")
KEYPOINT_SEMANTICS = "idcard_canonical_v1"


class GeometryError(ValueError):
    pass


def bbox_from_keypoints(points):
    _validate_point_shape(points)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def signed_area(points):
    _validate_point_shape(points)
    return 0.5 * sum(
        float(points[index][0]) * float(points[(index + 1) % 4][1])
        - float(points[(index + 1) % 4][0]) * float(points[index][1])
        for index in range(4)
    )


def validate_keypoints(points, width=None, height=None, bbox=None, min_area=16.0):
    _validate_point_shape(points)
    normalized = [[float(point[0]), float(point[1])] for point in points]
    if width is not None and height is not None:
        tolerance = max(float(width), float(height)) * 0.01
        for x, y in normalized:
            if x < -tolerance or y < -tolerance or x > float(width) + tolerance or y > float(height) + tolerance:
                raise GeometryError("keypoint is outside the image tolerance")

    area = signed_area(normalized)
    if area <= min_area:
        if area < 0:
            raise GeometryError("semantic keypoints have reversed winding")
        raise GeometryError("keypoint quadrilateral area is too small")

    cross_products = []
    for index in range(4):
        a = normalized[index]
        b = normalized[(index + 1) % 4]
        c = normalized[(index + 2) % 4]
        cross_products.append(_cross(a, b, c))
    if any(value <= 1e-6 for value in cross_products):
        raise GeometryError("keypoint quadrilateral is not strictly convex")

    edge_lengths = [
        math.hypot(
            normalized[(index + 1) % 4][0] - normalized[index][0],
            normalized[(index + 1) % 4][1] - normalized[index][1],
        )
        for index in range(4)
    ]
    if min(edge_lengths) < 1.0:
        raise GeometryError("keypoint quadrilateral has a degenerate edge")

    diagonal_a = math.hypot(normalized[2][0] - normalized[0][0], normalized[2][1] - normalized[0][1])
    diagonal_b = math.hypot(normalized[3][0] - normalized[1][0], normalized[3][1] - normalized[1][1])
    if min(diagonal_a, diagonal_b) < 2.0:
        raise GeometryError("keypoint quadrilateral has a degenerate diagonal")

    keypoint_bbox = bbox_from_keypoints(normalized)
    if bbox is not None:
        if len(bbox) != 4:
            raise GeometryError("bbox must contain four values")
        box_area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
        keypoint_box_area = (keypoint_bbox[2] - keypoint_bbox[0]) * (keypoint_bbox[3] - keypoint_bbox[1])
        if box_area <= 0 or keypoint_box_area / box_area < 0.5 or keypoint_box_area / box_area > 2.0:
            raise GeometryError("keypoints are inconsistent with bbox")

    return {
        "area": area,
        "bbox": keypoint_bbox,
        "edge_lengths": edge_lengths,
        "diagonal_lengths": [diagonal_a, diagonal_b],
    }


def destination_size(points, target_width=1016, aspect_ratio=1.5857725):
    metrics = validate_keypoints(points)
    top, right, bottom, left = metrics["edge_lengths"]
    observed_width = max(top, bottom)
    observed_height = max(left, right)
    if observed_height <= 0:
        raise GeometryError("observed card height is zero")
    if not 0.4 <= (observed_width / observed_height) / float(aspect_ratio) <= 2.5:
        raise GeometryError("card aspect ratio is outside the configured tolerance")
    width = max(2, int(target_width))
    height = max(2, int(round(width / float(aspect_ratio))))
    return width, height


def _validate_point_shape(points):
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise GeometryError("exactly four semantic keypoints are required")
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise GeometryError("each keypoint must contain x and y")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in point):
            raise GeometryError("keypoint coordinates must be finite numbers")


def _cross(a, b, c):
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(b[1])) - (
        float(b[1]) - float(a[1])
    ) * (float(c[0]) - float(b[0]))
