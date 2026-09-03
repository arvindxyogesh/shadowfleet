"""Export YOLOv8n pretrained weights to ONNX for the edge agent to serve.

This is a one-time, offline step run by a developer with network access — it is
deliberately kept out of edge_agent/requirements.txt since the runtime service
only needs onnxruntime, not the full ultralytics training/export stack.

Usage:
    pip install ultralytics
    python scripts/export_model.py --output models/yolov8n.onnx
"""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolov8n.pt", help="Ultralytics weights name or path")
    parser.add_argument("--output", default="models/yolov8n.onnx", help="Destination .onnx path")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    exported_path = model.export(format="onnx", imgsz=args.imgsz, opset=12)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(exported_path, output_path)
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
