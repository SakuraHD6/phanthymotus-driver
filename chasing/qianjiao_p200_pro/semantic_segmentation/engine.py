"""UWSegFormer inference and image postprocessing."""

from pathlib import Path
import sys
import time


CLASSES = ("BW", "HD", "WR", "RO", "RI", "FV")
CLASS_NAMES = ("water", "diver", "wreck", "robot", "reef", "fish")
# UWSegFormer's SUIM dataset declares its palette in RGB order.
PALETTE_RGB = (
    (0, 0, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 0),
    (255, 0, 255),
    (255, 255, 0),
)
INPUT_WIDTH = 640
INPUT_HEIGHT = 480


class InvalidFrameError(ValueError):
    """A camera frame could not be decoded and can be skipped."""


class UWSegFormerEngine:
    def __init__(self, source_root: str, checkpoint: str, device: str):
        self.source_root = Path(source_root).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = device
        self.model = None
        self._infer = None

    def load(self):
        config_path = (self.source_root / "local_configs" / "uwsegformer" /
                       "segformer_e+PPsegHead1.b0.640x480.suim.160k.py")
        if not config_path.is_file():
            raise FileNotFoundError(f"UWSegFormer config not found: {config_path}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"UWSegFormer checkpoint not found: {self.checkpoint}")

        existing = sys.modules.get("mmseg")
        if existing is not None:
            existing_path = Path(existing.__file__).resolve()
            if self.source_root not in existing_path.parents:
                raise RuntimeError(f"A different mmseg package is already loaded: {existing_path}")
        else:
            sys.path.insert(0, str(self.source_root))

        from mmcv.runner import load_checkpoint
        from mmseg.apis import inference_segmentor, init_segmentor

        # init_segmentor builds the model and disables the ImageNet pretraining
        # path. Load the actual segmentation checkpoint strictly so an
        # incompatible model cannot silently publish arbitrary labels.
        model = init_segmentor(str(config_path), checkpoint=None, device=self.device)
        checkpoint = load_checkpoint(model, str(self.checkpoint), map_location="cpu", strict=True)
        metadata = checkpoint.get("meta") or {}
        checkpoint_classes = metadata.get("CLASSES")
        if checkpoint_classes and tuple(checkpoint_classes) != CLASSES:
            raise ValueError(f"Unexpected checkpoint class order: {checkpoint_classes}")
        self.model = model
        self._infer = inference_segmentor

    def process(self, jpeg: bytes, opacity: float):
        import cv2
        import numpy as np

        if self.model is None:
            raise RuntimeError("UWSegFormer model is not loaded")
        try:
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise InvalidFrameError("Could not decode the input JPEG") from exc
        if frame is None:
            raise InvalidFrameError("Could not decode the input JPEG")

        height, width = frame.shape[:2]
        scale = min(INPUT_WIDTH / width, INPUT_HEIGHT / height)
        scaled_width = max(1, min(INPUT_WIDTH, round(width * scale)))
        scaled_height = max(1, min(INPUT_HEIGHT, round(height * scale)))
        left = (INPUT_WIDTH - scaled_width) // 2
        top = (INPUT_HEIGHT - scaled_height) // 2
        image = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
        image[top:top + scaled_height, left:left + scaled_width] = cv2.resize(
            frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

        started = time.monotonic()
        result = self._infer(self.model, image)
        inference_ms = round((time.monotonic() - started) * 1000, 2)
        mask = np.asarray(result[0])
        if mask.shape != (INPUT_HEIGHT, INPUT_WIDTH):
            raise ValueError(f"Unexpected segmentation shape: {mask.shape}")
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f"Unexpected segmentation type: {mask.dtype}")
        if mask.min() < 0 or mask.max() >= len(CLASSES):
            raise ValueError("Segmentation output contains unknown class IDs")

        cropped = mask[top:top + scaled_height, left:left + scaled_width]
        mask = cv2.resize(cropped.astype(np.uint8), (width, height),
                          interpolation=cv2.INTER_NEAREST)
        palette = np.asarray(PALETTE_RGB, dtype=np.uint8)[:, ::-1]
        colored = palette[mask]
        blended = cv2.addWeighted(frame, 1.0 - opacity, colored, opacity, 0)
        overlay = frame.copy()
        foreground = mask != 0
        overlay[foreground] = blended[foreground]
        encoded, output = cv2.imencode(".jpg", overlay)
        if not encoded:
            raise RuntimeError("Could not encode the segmentation overlay")

        counts = np.bincount(mask.ravel(), minlength=len(CLASSES))
        total = width * height
        summary = {
            "width": width,
            "height": height,
            "inference_ms": inference_ms,
            "classes": [
                {"id": index, "code": code, "name": CLASS_NAMES[index],
                 "pixels": int(counts[index]),
                 "fraction": round(int(counts[index]) / total, 6)}
                for index, code in enumerate(CLASSES)
            ],
        }
        return output.tobytes(), summary
