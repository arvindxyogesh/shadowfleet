def _iou(box_a: dict, box_b: dict) -> float:
    ix1, iy1 = max(box_a["x1"], box_b["x1"]), max(box_a["y1"], box_b["y1"])
    ix2, iy2 = min(box_a["x2"], box_b["x2"]), min(box_a["y2"], box_b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, box_a["x2"] - box_a["x1"]) * max(0.0, box_a["y2"] - box_a["y1"])
    area_b = max(0.0, box_b["x2"] - box_b["x1"]) * max(0.0, box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compute_disagreement(
    prod_detections: list[dict],
    shadow_detections: list[dict],
    iou_threshold: float = 0.5,
) -> float:
    """Fraction (0..1) of detections with no matching same-class, overlapping
    counterpart in the other prediction set. 0 = full agreement, 1 = no
    detection in either set was matched in the other."""
    total = max(len(prod_detections), len(shadow_detections))
    if total == 0:
        return 0.0

    matched_shadow_indices: set[int] = set()
    matched_count = 0
    for prod_det in prod_detections:
        for idx, shadow_det in enumerate(shadow_detections):
            if idx in matched_shadow_indices:
                continue
            if prod_det["class_id"] != shadow_det["class_id"]:
                continue
            if _iou(prod_det["box"], shadow_det["box"]) >= iou_threshold:
                matched_shadow_indices.add(idx)
                matched_count += 1
                break

    return 1.0 - (matched_count / total)
