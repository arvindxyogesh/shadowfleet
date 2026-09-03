import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-max suppression. boxes: (N, 4) xyxy. Returns indices to keep."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        order = order[1:][iou <= iou_threshold]

    return keep


def decode_yolov8_output(
    raw_output: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    meta: dict,
    class_names: list[str],
    max_detections: int = 300,
) -> list[dict]:
    """Decode a standard Ultralytics YOLOv8 ONNX export output.

    raw_output shape: (1, 4 + num_classes, num_boxes) — 4 box coords (cx, cy, w, h
    in the letterboxed input's pixel space) followed by per-class scores.
    """
    predictions = raw_output[0].transpose(1, 0)  # (num_boxes, 4 + num_classes)

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]

    keep_mask = scores >= conf_threshold
    boxes_xywh = boxes_xywh[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]

    if len(boxes_xywh) == 0:
        return []

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

    # Undo letterbox padding/scaling to map back into the original image.
    scale = meta["scale"]
    boxes_xyxy[:, [0, 2]] -= meta["pad_x"]
    boxes_xyxy[:, [1, 3]] -= meta["pad_y"]
    boxes_xyxy /= scale

    orig_w, orig_h = meta["orig_w"], meta["orig_h"]
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

    detections: list[dict] = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = scores[cls_mask]
        for idx in nms(cls_boxes, cls_scores, iou_threshold):
            x1, y1, x2, y2 = cls_boxes[idx]
            cls_int = int(cls)
            detections.append(
                {
                    "box": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                    "score": float(cls_scores[idx]),
                    "class_id": cls_int,
                    "class_name": class_names[cls_int] if cls_int < len(class_names) else str(cls_int),
                }
            )

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections[:max_detections]
