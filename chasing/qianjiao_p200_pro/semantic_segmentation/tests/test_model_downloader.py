"""Offline checks for selecting and caching the author's SUIM checkpoint."""

import hashlib
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semantic_segmentation import model_downloader


class FakeGdown:
    def __init__(self, files=None, fail=False):
        self.files = files or []
        self.fail = fail
        self.folder_calls = []
        self.download_calls = []

    def download_folder(self, **kwargs):
        self.folder_calls.append(kwargs)
        return self.files

    def download(self, **kwargs):
        self.download_calls.append(kwargs)
        output = Path(kwargs["output"])
        if self.fail:
            output.with_suffix(".part").write_bytes(b"incomplete")
            raise OSError("network failure")
        output.write_bytes(b"fake checkpoint")
        kwargs["progress"](15, 15)
        return str(output)


class ModelDownloaderTests(unittest.TestCase):
    def test_cached_checkpoint_is_used_without_gdown(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manual.pth"
            target.write_bytes(b"existing")
            with patch.dict(sys.modules, {"gdown": None}):
                result = model_downloader.ensure_suim_checkpoint(target)
            self.assertEqual(Path(result), target.resolve())
            self.assertEqual(target.read_bytes(), b"existing")

    def test_wrong_expected_hash_does_not_replace_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manual.pth"
            target.write_bytes(b"existing")
            expected = hashlib.sha256(b"different").hexdigest()
            with patch.dict(sys.modules, {"gdown": None}):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    model_downloader.ensure_suim_checkpoint(target, expected_sha256=expected)
            self.assertEqual(target.read_bytes(), b"existing")

    def test_downloads_only_named_suim_checkpoint(self):
        fake = FakeGdown([
            SimpleNamespace(path="models/uws_DUT.pth", id="dut_id"),
            SimpleNamespace(path="models/uws_SUIM.pth", id="suim_id"),
        ])
        statuses = []
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "suim.pth"
            with patch.object(model_downloader, "MODEL_DIR", Path(directory)):
                with patch.dict(sys.modules, {"gdown": fake}):
                    result = model_downloader.ensure_suim_checkpoint(
                        target, status_cb=statuses.append,
                    )
            self.assertEqual(Path(result), target.resolve())
            self.assertEqual(target.read_bytes(), b"fake checkpoint")
            self.assertTrue(fake.folder_calls[0]["skip_download"])
            self.assertEqual([call["id"] for call in fake.download_calls], ["suim_id"])
            self.assertTrue(any("100%" in text for text in statuses))

    def test_ambiguous_filename_is_not_downloaded(self):
        fake = FakeGdown([
            SimpleNamespace(path="a/uws_SUIM.pth", id="first"),
            SimpleNamespace(path="b/uws_SUIM.pth", id="second"),
        ])
        with self.assertRaisesRegex(ValueError, "多个"):
            model_downloader._find_suim_file(fake, "", lambda message: None, None)
        self.assertEqual(fake.download_calls, [])

    def test_file_link_is_accepted_but_folder_link_is_rejected(self):
        file_url = "https://drive.google.com/file/d/example_id/view?usp=sharing"
        self.assertEqual(model_downloader._file_id_from_url(file_url), "example_id")
        with self.assertRaisesRegex(ValueError, "文件夹"):
            model_downloader._file_id_from_url(
                "https://drive.google.com/drive/folders/example_id"
            )

    def test_failed_download_does_not_publish_a_partial_file(self):
        fake = FakeGdown(fail=True)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "suim.pth"
            with patch.object(model_downloader, "MODEL_DIR", Path(directory)):
                with patch.dict(sys.modules, {"gdown": fake}):
                    with self.assertRaisesRegex(RuntimeError, "下载失败"):
                        model_downloader.ensure_suim_checkpoint(
                            target, file_url="https://drive.google.com/file/d/example_id/view",
                        )
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).glob(".uwsegformer-*")), [])

    def test_wrong_download_hash_is_not_published(self):
        fake = FakeGdown()
        expected = hashlib.sha256(b"different").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "suim.pth"
            with patch.object(model_downloader, "MODEL_DIR", Path(directory)):
                with patch.dict(sys.modules, {"gdown": fake}):
                    with self.assertRaisesRegex(ValueError, "SHA256"):
                        model_downloader.ensure_suim_checkpoint(
                            target, file_url="https://drive.google.com/file/d/example_id/view",
                            expected_sha256=expected,
                        )
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).glob(".uwsegformer-*")), [])

    def test_stop_before_download_prevents_network_calls(self):
        fake = FakeGdown()
        stop = threading.Event()
        stop.set()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "suim.pth"
            with patch.object(model_downloader, "MODEL_DIR", Path(directory)):
                with patch.dict(sys.modules, {"gdown": fake}):
                    with self.assertRaisesRegex(RuntimeError, "已停止"):
                        model_downloader.ensure_suim_checkpoint(target, stop_event=stop)
            self.assertFalse(target.exists())
            self.assertEqual(fake.folder_calls, [])
            self.assertEqual(fake.download_calls, [])


if __name__ == "__main__":
    unittest.main()
