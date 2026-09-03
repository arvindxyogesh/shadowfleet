import numpy as np
from PIL import Image


def letterbox(image: Image.Image, target_size: int) -> tuple[Image.Image, float, int, int]:
    """Resize keeping aspect ratio and pad to a square target_size x target_size."""
    orig_w, orig_h = image.size
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w, new_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def preprocess(image: Image.Image, target_size: int) -> tuple[np.ndarray, dict]:
    """Letterbox, normalize, and reshape an image into a YOLOv8 ONNX input tensor.

    Returns the (1, 3, target_size, target_size) float32 tensor plus the metadata
    needed to map predicted boxes back into the original image's coordinate space.
    """
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    canvas, scale, pad_x, pad_y = letterbox(image, target_size)

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.ascontiguousarray(arr[np.newaxis, ...])

    meta = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "orig_w": orig_w,
        "orig_h": orig_h,
    }
    return arr, meta
