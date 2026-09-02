import time

import onnxruntime as ort
from PIL import Image

from .coco_classes import COCO_CLASSES
from .postprocessing import decode_yolov8_output
from .preprocessing import preprocess


class ONNXModel:
    """Wraps an ONNX Runtime session for a YOLOv8-family detection model."""

    def __init__(self, model_path: str, input_size: int = 640, class_names: list[str] | None = None):
        self.model_path = model_path
        self.input_size = input_size
        self.class_names = class_names or COCO_CLASSES
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(
        self,
        image: Image.Image,
        conf_threshold: float,
        iou_threshold: float,
        max_detections: int = 300,
    ) -> tuple[list[dict], float, int, int]:
        start = time.perf_counter()
        input_tensor, meta = preprocess(image, self.input_size)
        raw_output = self.session.run(None, {self.input_name: input_tensor})[0]
        detections = decode_yolov8_output(
            raw_output, conf_threshold, iou_threshold, meta, self.class_names, max_detections
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return detections, latency_ms, meta["orig_w"], meta["orig_h"]
