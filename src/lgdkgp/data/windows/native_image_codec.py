"""Live RoboTwin pixels in the frozen dataset's decoded-JPEG coordinates."""
import io
import numpy as np
from PIL import Image


def robotwin_live_training_pixels(rgb):
    """Mirror the source collector's RGB-as-BGR JPEG write and RGB read.

    Do not apply to LIBERO arrays or already decoded dataset images. Dataset
    payloads and trained weights remain unchanged; this is a live-input boundary.
    """
    import cv2
    rgb=np.asarray(rgb)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:
        raise ValueError('Expected a raw native uint8 RGB frame')
    ok,payload=cv2.imencode('.jpg',np.ascontiguousarray(rgb))
    if not ok:raise ValueError('Native collector JPEG encoding failed')
    with Image.open(io.BytesIO(payload.tobytes()))as image:
        return np.asarray(image.convert('RGB')).copy()
