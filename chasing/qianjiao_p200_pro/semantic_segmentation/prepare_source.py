"""Repair the published UWSegFormer checkout for its SUIM inference path."""

from pathlib import Path
import sys


def prepare(root: Path):
    heads = root / "mmseg" / "models" / "decode_heads" / "__init__.py"
    original = heads.read_text(encoding="utf-8")
    old_import = "from .segformer_head import PPMobileSegHead_segformer2,PPMobileSegHead_segformer1"
    if old_import not in original:
        raise RuntimeError(f"Unexpected UWSegFormer decoder exports: {heads}")

    config = (root / "local_configs" / "uwsegformer" /
              "segformer_e+PPsegHead1.b0.640x480.suim.160k.py")
    text = config.read_text(encoding="utf-8")
    if "../../_base_/" not in text:
        raise RuntimeError(f"Unexpected UWSegFormer config paths: {config}")

    # The selected model uses only these heads. Avoid importing PointHead's
    # compiled MMCV operator, which this inference path does not use.
    heads.write_text(
        "from .segformer_head import MAA_segformer1, SegFormerHead\n"
        "__all__ = ['MAA_segformer1', 'SegFormerHead']\n",
        encoding="utf-8",
    )

    config.write_text(text.replace("../../_base_/", "../_base_/"), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: prepare_source.py UWSegFormer_checkout")
    prepare(Path(sys.argv[1]))
