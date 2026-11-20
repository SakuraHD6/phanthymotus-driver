"""Offline Card contract and lifecycle checks, without ROS or model weights."""

import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semantic_segmentation import engine, model_downloader
from semantic_segmentation.plugin import SemanticSegmentationPlugin


class Message:
    pass


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def make_run(plugin, first_frame):
    header = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=2), frame_id="camera")
    return {
        "settings": dict(plugin._settings),
        "node": None,
        "executor": None,
        "spin_thread": None,
        "worker": None,
        "input_topic": "/camera",
        "finished": threading.Event(),
        "latest": (first_frame, header),
        "overlay": Publisher(),
        "summary": Publisher(),
        "image_type": Message,
        "string_type": Message,
        "ros_error": None,
        "cleanup_error": None,
    }


class FeedAfterBadFrame:
    def __init__(self, plugin, run):
        self.plugin = plugin
        self.run = run
        self.stopped = False
        self.sent_good_frame = False

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, timeout=None):
        if self.plugin._bad_frames and not self.sent_good_frame:
            header = SimpleNamespace(stamp=SimpleNamespace(sec=3, nanosec=4), frame_id="camera")
            self.run["latest"] = (b"good", header)
            self.sent_good_frame = True
        elif self.plugin._frames_processed:
            self.set()
        return self.stopped


class PluginContractTests(unittest.TestCase):
    def test_config_action_declares_its_parameters(self):
        tool = SemanticSegmentationPlugin({}, lambda: None).get_tool()
        schema = tool["inputSchema"]
        self.assertIn("config", schema["properties"]["action"]["enum"])
        config_params = schema["x-action-params"]["config"]["params"]
        self.assertEqual(set(config_params), set(tool["configSchema"]["properties"]))
        self.assertTrue(set(config_params).issubset(schema["properties"]))

    def test_info_exposes_loading_progress_as_desc(self):
        plugin = SemanticSegmentationPlugin({}, lambda: None)
        plugin._state = "loading"
        plugin._model_status = "正在下载 SUIM 权重：40%"
        info = plugin.info()
        self.assertEqual(info["state"], "loading")
        self.assertEqual(info["desc"], plugin._model_status)

    def test_decode_error_becomes_recoverable_frame_error(self):
        class FakeCvError(Exception):
            pass

        def fail_decode(*args):
            raise FakeCvError("invalid JPEG")

        fake_cv2 = SimpleNamespace(error=FakeCvError, imdecode=fail_decode, IMREAD_COLOR=1)
        fake_numpy = SimpleNamespace(frombuffer=lambda *args, **kwargs: [], uint8=object())
        processor = engine.UWSegFormerEngine("/source", "/model.pth", "cpu")
        processor.model = object()
        with patch.dict(sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy}):
            with self.assertRaises(engine.InvalidFrameError):
                processor.process(b"broken", 0.5)

    def test_bad_frame_does_not_prevent_a_later_frame(self):
        class FakeEngine:
            def load(self):
                pass

            def process(self, jpeg, opacity):
                if jpeg == b"bad":
                    raise engine.InvalidFrameError("bad JPEG")
                return b"overlay", {"classes": []}

        plugin = SemanticSegmentationPlugin({}, lambda: None)
        run = make_run(plugin, b"bad")
        run["stop"] = FeedAfterBadFrame(plugin, run)
        plugin._run = run
        plugin._state = "loading"
        with patch.object(model_downloader, "ensure_suim_checkpoint"):
            with patch.object(engine, "UWSegFormerEngine", return_value=FakeEngine()):
                plugin._worker(run)
        self.assertEqual(plugin._bad_frames, 1)
        self.assertEqual(plugin._frames_processed, 1)
        self.assertEqual(len(run["overlay"].messages), 1)
        self.assertEqual(len(run["summary"].messages), 1)
        self.assertIsNone(plugin._last_frame_error)

    def test_inference_failure_is_reported_as_error(self):
        class FakeEngine:
            def load(self):
                pass

            def process(self, jpeg, opacity):
                raise RuntimeError("model output invalid")

        plugin = SemanticSegmentationPlugin({}, lambda: None)
        run = make_run(plugin, b"frame")
        run["stop"] = threading.Event()
        plugin._run = run
        plugin._state = "loading"
        with patch.object(model_downloader, "ensure_suim_checkpoint"):
            with patch.object(engine, "UWSegFormerEngine", return_value=FakeEngine()):
                plugin._worker(run)
        self.assertEqual(plugin.info()["state"], "error")
        self.assertIn("model output invalid", plugin.info()["error"])

    def test_ros_node_is_destroyed_after_executor_and_thread_stop(self):
        calls = []
        node = SimpleNamespace(destroy_node=lambda: calls.append("destroy"))
        executor = SimpleNamespace(
            remove_node=lambda value: calls.append("remove"),
            shutdown=lambda: calls.append("shutdown"),
        )
        spin = SimpleNamespace(is_alive=lambda: True, join=lambda: calls.append("join"))
        SemanticSegmentationPlugin._destroy_ros({
            "node": node, "executor": executor, "spin_thread": spin,
        })
        self.assertEqual(calls, ["remove", "shutdown", "join", "destroy"])

    def test_stop_waits_for_a_worker_that_is_still_loading(self):
        joined = []
        plugin = SemanticSegmentationPlugin({}, lambda: None)
        run = make_run(plugin, b"frame")
        run["stop"] = threading.Event()
        run["worker"] = SimpleNamespace(
            is_alive=lambda: True,
            join=lambda timeout: joined.append(timeout),
        )
        plugin._run = run
        plugin._state = "loading"
        result = plugin.stop()
        self.assertTrue(run["stop"].is_set())
        self.assertEqual(joined, [3.0])
        self.assertEqual(result["state"], "stopping")
        self.assertFalse(run["finished"].is_set())

    def test_ros_executor_failure_is_reported(self):
        plugin = SemanticSegmentationPlugin({}, lambda: None)
        run = make_run(plugin, b"frame")
        run["stop"] = threading.Event()
        run["executor"] = SimpleNamespace(spin=lambda: None)
        plugin._run = run
        plugin._state = "loading"
        plugin._spin_ros(run)
        self.assertEqual(plugin.info()["state"], "error")
        self.assertIn("ROS 线程意外退出", plugin.info()["error"])
        self.assertTrue(run["stop"].is_set())

    def test_incomplete_ros_cleanup_prevents_a_new_run(self):
        plugin = SemanticSegmentationPlugin({}, lambda: None)
        run = make_run(plugin, b"frame")
        run["stop"] = threading.Event()
        run["cleanup_error"] = "ROS cleanup failed"
        plugin._run = run
        result = plugin.start({"input_topic": "/camera"})
        self.assertEqual(result["state"], "error")
        self.assertIn("ROS cleanup failed", result["message"])


if __name__ == "__main__":
    unittest.main()
