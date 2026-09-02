import numpy as np

from edge_agent.app.postprocessing import decode_yolov8_output, nms, xywh_to_xyxy


def test_xywh_to_xyxy():
    boxes = np.array([[50, 50, 20, 10]], dtype=np.float32)
    xyxy = xywh_to_xyxy(boxes)
    assert np.allclose(xyxy[0], [40, 45, 60, 55])


def test_nms_suppresses_overlapping_boxes():
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],  # heavily overlaps box 0
            [50, 50, 60, 60],  # far away, should be kept independently
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep == [0, 2]


def test_nms_handles_empty_input():
    boxes = np.empty((0, 4), dtype=np.float32)
    scores = np.empty((0,), dtype=np.float32)
    assert nms(boxes, scores, iou_threshold=0.5) == []


def test_decode_yolov8_output_filters_by_confidence():
    num_classes = 3
    raw = np.zeros((1, 4 + num_classes, 2), dtype=np.float32)
    # box 0: center (50,50) size (20,20), class 1 score 0.9 -> kept
    raw[0, :, 0] = [50, 50, 20, 20, 0.1, 0.9, 0.05]
    # box 1: every class below threshold -> filtered out
    raw[0, :, 1] = [30, 30, 10, 10, 0.05, 0.1, 0.05]

    meta = {"scale": 1.0, "pad_x": 0, "pad_y": 0, "orig_w": 100, "orig_h": 100}
    detections = decode_yolov8_output(
        raw,
        conf_threshold=0.25,
        iou_threshold=0.45,
        meta=meta,
        class_names=["a", "b", "c"],
        max_detections=10,
    )

    assert len(detections) == 1
    det = detections[0]
    assert det["class_name"] == "b"
    assert det["score"] > 0.85
    assert det["box"] == {"x1": 40.0, "y1": 40.0, "x2": 60.0, "y2": 60.0}


def test_decode_yolov8_output_undoes_letterbox_padding():
    # A box exactly at the padded canvas's center, with scale=0.5 and 20px padding
    # on each axis, should map back to the pre-letterbox coordinate space.
    raw = np.zeros((1, 4 + 1, 1), dtype=np.float32)
    raw[0, :, 0] = [70, 70, 10, 10, 0.9]  # cx, cy, w, h, class-0 score

    meta = {"scale": 0.5, "pad_x": 20, "pad_y": 20, "orig_w": 200, "orig_h": 200}
    detections = decode_yolov8_output(
        raw, conf_threshold=0.25, iou_threshold=0.45, meta=meta, class_names=["obj"]
    )

    assert len(detections) == 1
    box = detections[0]["box"]
    # (70 - 20) / 0.5 = 100 for the center; half-width (10/2)/0.5 = 10
    assert box == {"x1": 90.0, "y1": 90.0, "x2": 110.0, "y2": 110.0}


def test_decode_yolov8_output_respects_max_detections():
    num_boxes = 5
    raw = np.zeros((1, 4 + 1, num_boxes), dtype=np.float32)
    for i in range(num_boxes):
        cx = 10 + i * 5
        raw[0, :, i] = [cx, 10, 4, 4, 0.9 - i * 0.01]

    meta = {"scale": 1.0, "pad_x": 0, "pad_y": 0, "orig_w": 100, "orig_h": 100}
    detections = decode_yolov8_output(
        raw, conf_threshold=0.25, iou_threshold=0.0, meta=meta, class_names=["obj"], max_detections=2
    )

    assert len(detections) == 2
    assert detections[0]["score"] >= detections[1]["score"]
