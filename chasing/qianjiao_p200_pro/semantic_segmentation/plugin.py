"""MCP processor card for a UWSegFormer camera stream."""

import json
import math
import threading
import time


class SemanticSegmentationPlugin:
    NAME = "semantic_segmentation"
    OVERLAY_TOPIC = "/qianjiao_p200_pro/semantic_segmentation/overlay"
    SUMMARY_TOPIC = "/qianjiao_p200_pro/semantic_segmentation/summary"

    def __init__(self, config, node_provider):
        config = config or {}
        self._settings = {
            "source_root": str(config.get("source_root", "/opt/uwsegformer")),
            "checkpoint": str(config.get("checkpoint", "/models/uwsegformer/suim.pth")),
            "device": str(config.get("device", "cpu")),
            "fps": float(config.get("fps", 2.0)),
            "opacity": float(config.get("opacity", 0.5)),
        }
        self._validate(self._settings)
        self._node_provider = node_provider
        # The source is set in the deployment config, not through remote MCP calls.
        self._google_drive_file_url = str(config.get("google_drive_file_url") or "").strip()
        self._checkpoint_sha256 = str(config.get("checkpoint_sha256") or "").strip()
        self._lock = threading.RLock()
        self._lifecycle = threading.Lock()
        self._run = None
        self._state = "idle"
        self._last_error = None
        self._model_status = None
        self._frames_processed = 0
        self._frames_received = 0
        self._bad_frames = 0
        self._last_frame_at = None
        self._last_frame_error = None

    @staticmethod
    def _validate(settings):
        for name in ("source_root", "checkpoint", "device"):
            if not str(settings[name]).strip():
                raise ValueError(f"{name} must not be empty")
        if not math.isfinite(settings["fps"]) or not 0.1 <= settings["fps"] <= 30:
            raise ValueError("fps must be between 0.1 and 30")
        if not math.isfinite(settings["opacity"]) or not 0 <= settings["opacity"] <= 1:
            raise ValueError("opacity must be between 0 and 1")

    def get_tool(self):
        config_properties = {
            "checkpoint": {"type": "string", "title": "SUIM 模型权重路径",
                           "default": self._settings["checkpoint"]},
            "device": {"type": "string", "title": "推理设备",
                       "default": self._settings["device"]},
            "fps": {"type": "number", "minimum": 0.1, "maximum": 30,
                    "title": "推理频率上限", "default": self._settings["fps"]},
            "opacity": {"type": "number", "minimum": 0, "maximum": 1,
                        "title": "分割颜色透明度", "default": self._settings["opacity"]},
        }
        return {
            "name": self.NAME,
            "type": "processor",
            "multiInstance": False,
            "description": "UWSegFormer 水下图像语义分割，输出叠加画面和各类别像素占比。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "stop", "info", "config"]},
                    "input_topic": {"type": "string", "description": "相机 JPEG 输入话题"},
                    **config_properties,
                },
                "required": ["action"],
                "x-action-params": {
                    "start": {"params": ["input_topic"], "description": "开始处理相机画面"},
                    "stop": {"params": [], "description": "停止分割"},
                    "info": {"params": ["input_topic"], "description": "查看运行状态和话题"},
                    "config": {"params": list(config_properties), "description": "更新模型和画面配置"},
                },
            },
            "configSchema": {
                "type": "object",
                "properties": config_properties,
            },
            "topic_in": [{"format": "image/jpeg", "message_type": "sensor_msgs/msg/CompressedImage"}],
            "topic_out": self._outputs(),
        }

    def _outputs(self):
        return [
            {"topic": self.OVERLAY_TOPIC, "format": "image/jpeg"},
            {"topic": self.SUMMARY_TOPIC, "format": "data/json"},
        ]

    def dispatch(self, action, args):
        if action == "info":
            return self.info(args)
        if action == "config":
            return self.configure(args)
        if action == "start":
            return self.start(args)
        if action == "stop":
            return self.stop()
        raise ValueError(f"unsupported semantic segmentation action: {action}")

    def info(self, args=None):
        args = args or {}
        with self._lock:
            run = self._run
            topic = str(args.get("input_topic") or (run["input_topic"] if run else ""))
            topic_in = {"format": "image/jpeg", "message_type": "sensor_msgs/msg/CompressedImage"}
            if topic:
                topic_in["topic"] = topic
            worker = run.get("worker") if run else None
            active = run["settings"] if worker and worker.is_alive() else None
            restart_required = bool(active and any(
                active[name] != self._settings[name]
                for name in ("source_root", "checkpoint", "device")
            ))
            desc = self._model_status
            if self._state == "running" and self._frames_processed == 0:
                desc = "模型已加载，等待有效相机画面"
            frame_age = (None if self._last_frame_at is None else
                         round(max(0.0, time.monotonic() - self._last_frame_at), 2))
            return {
                "state": self._state,
                "desc": desc,
                "topic_in": [topic_in],
                "topic_out": self._outputs(),
                "settings": dict(self._settings),
                "active_model": ({name: active[name] for name in
                                  ("source_root", "checkpoint", "device")} if active else None),
                "restart_required": restart_required,
                "frames_processed": self._frames_processed,
                "frames_received": self._frames_received,
                "bad_frames": self._bad_frames,
                "last_frame_age_s": frame_age,
                "last_frame_error": self._last_frame_error,
                "model_status": self._model_status,
                "error": self._last_error,
            }

    def configure(self, args):
        with self._lock:
            settings = dict(self._settings)
            try:
                for name in ("checkpoint", "device"):
                    if name in args:
                        settings[name] = str(args[name]).strip()
                for name in ("fps", "opacity"):
                    if name in args:
                        settings[name] = float(args[name])
                self._validate(settings)
            except (TypeError, ValueError) as exc:
                return {"state": "error", "message": str(exc)}
            self._settings = settings
        result = self.info()
        result["state"] = "configured"
        return result

    def start(self, args):
        topic = str(args.get("input_topic") or "").strip()
        topics = args.get("input_topics") or []
        if not topic and topics:
            topic = str(topics[0]).strip()
        if not topic:
            return {"state": "error", "message": "请连接一个 image/jpeg 相机话题"}
        if len(topics) > 1:
            return {"state": "error", "message": "语义分割卡片需要一个相机输入"}

        with self._lifecycle:
            with self._lock:
                current = self._run
                if current and current.get("cleanup_error"):
                    return {"state": "error", "message": current["cleanup_error"]}
                current_worker = current.get("worker") if current else None
                if (current_worker and current_worker.is_alive()
                        and not current["stop"].is_set()
                        and current["input_topic"] == topic):
                    return self.info({"input_topic": topic})
            if current_worker and current_worker.is_alive():
                self._stop_locked()
                if current_worker.is_alive():
                    return {"state": "error", "message": "上一轮推理仍在停止，请稍后重试"}
                if current.get("cleanup_error"):
                    return {"state": "error", "message": current["cleanup_error"]}

            parent_node = self._node_provider()
            if parent_node is None:
                with self._lock:
                    self._state = "error"
                    self._last_error = "ROS 图像节点不可用"
                return {"state": "error", "message": self._last_error}
            run = {"node": None, "executor": None, "spin_thread": None, "worker": None,
                   "input_topic": topic, "stop": threading.Event(),
                   "finished": threading.Event(), "latest": None,
                   "subscription": None, "overlay": None, "summary": None,
                   "ros_error": None, "cleanup_error": None}
            worker_started = False
            try:
                from rclpy.executors import SingleThreadedExecutor
                from rclpy.node import Node
                from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
                from sensor_msgs.msg import CompressedImage
                from std_msgs.msg import String

                # Keep this Card's subscriptions out of the device's status node.
                node = Node("qianjiao_p200_pro_segmentation", context=parent_node.context)
                run["node"] = node
                executor = SingleThreadedExecutor(context=parent_node.context)
                run["executor"] = executor
                executor.add_node(node)
                qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST, depth=1,
                                 durability=DurabilityPolicy.VOLATILE)
                run["image_type"] = CompressedImage
                run["string_type"] = String
                with self._lock:
                    run["settings"] = dict(self._settings)

                def on_image(message):
                    with self._lock:
                        if not run["stop"].is_set() and self._run is run:
                            run["latest"] = (bytes(message.data), message.header)
                            self._frames_received += 1
                            self._last_frame_at = time.monotonic()

                run["overlay"] = node.create_publisher(CompressedImage, self.OVERLAY_TOPIC, qos)
                run["summary"] = node.create_publisher(String, self.SUMMARY_TOPIC, qos)
                run["subscription"] = node.create_subscription(CompressedImage, topic, on_image, qos)
                run["spin_thread"] = threading.Thread(target=self._spin_ros, args=(run,),
                                                      name="qianjiao-segmentation-ros", daemon=True)
                run["worker"] = threading.Thread(target=self._worker, args=(run,),
                                                 name="qianjiao-segmentation", daemon=True)
                with self._lock:
                    self._run = run
                    self._state = "loading"
                    self._last_error = None
                    self._model_status = "正在准备模型"
                    self._frames_processed = 0
                    self._frames_received = 0
                    self._bad_frames = 0
                    self._last_frame_at = None
                    self._last_frame_error = None
                run["spin_thread"].start()
                run["worker"].start()
                worker_started = True
                return self.info({"input_topic": topic})
            except Exception as exc:
                run["stop"].set()
                if worker_started:
                    run["worker"].join(timeout=3.0)
                    with self._lock:
                        self._state = "stopping" if run["worker"].is_alive() else "error"
                        self._last_error = run.get("cleanup_error") or str(exc)
                    return {"state": "error", "message": self._last_error}
                cleanup_error = None
                try:
                    self._destroy_ros(run)
                except Exception as cleanup_exc:
                    cleanup_error = f"释放语义分割 ROS 资源失败：{cleanup_exc}"
                with self._lock:
                    run["cleanup_error"] = cleanup_error
                    self._run = run if cleanup_error else None
                    self._state = "error"
                    self._last_error = cleanup_error or str(exc)
                return {"state": "error", "message": self._last_error}

    def _spin_ros(self, run):
        try:
            run["executor"].spin()
            error = "语义分割 ROS 线程意外退出"
        except Exception as exc:
            error = f"语义分割 ROS 线程失败：{exc}"
        with self._lock:
            if self._run is run and not run["stop"].is_set():
                run["ros_error"] = error
                self._state = "error"
                self._last_error = error
                run["stop"].set()

    def _stop_locked(self):
        with self._lock:
            run = self._run
            if run is None:
                self._state = "idle"
                return
            if run.get("cleanup_error"):
                self._state = "error"
                return
            run["stop"].set()
            worker = run.get("worker")
            if not run["finished"].is_set():
                self._state = "stopping"
        if worker and worker.is_alive():
            worker.join(timeout=3.0)
        with self._lock:
            if run["finished"].is_set() and not run.get("cleanup_error"):
                self._state = "idle"

    def stop(self):
        with self._lifecycle:
            self._stop_locked()
            return self.info()

    def _worker(self, run):
        failure = None
        try:
            from .engine import InvalidFrameError, UWSegFormerEngine
            from .model_downloader import ensure_suim_checkpoint

            settings = run["settings"]

            def model_status(message):
                with self._lock:
                    if self._run is run and not run["stop"].is_set():
                        self._model_status = message

            if run["stop"].is_set():
                return
            ensure_suim_checkpoint(
                settings["checkpoint"], file_url=self._google_drive_file_url,
                status_cb=model_status, stop_event=run["stop"],
                expected_sha256=self._checkpoint_sha256,
            )
            if run["stop"].is_set():
                return
            model_status("正在加载 UWSegFormer 模型")
            engine = UWSegFormerEngine(settings["source_root"], settings["checkpoint"],
                                       settings["device"])
            engine.load()
            with self._lock:
                if not run["stop"].is_set():
                    self._state = "running"
                    self._model_status = "模型已加载"
            while not run["stop"].is_set():
                with self._lock:
                    frame = run["latest"]
                    run["latest"] = None
                    fps = self._settings["fps"]
                    opacity = self._settings["opacity"]
                if frame is None:
                    run["stop"].wait(0.05)
                    continue
                started = time.monotonic()
                jpeg, header = frame
                try:
                    overlay, summary = engine.process(jpeg, opacity)
                except InvalidFrameError as exc:
                    with self._lock:
                        self._bad_frames += 1
                        self._last_frame_error = str(exc)
                    run["stop"].wait(max(0.0, 1.0 / fps - (time.monotonic() - started)))
                    continue
                summary["stamp"] = {"sec": int(header.stamp.sec),
                                    "nanosec": int(header.stamp.nanosec),
                                    "frame_id": str(header.frame_id)}
                image_message = run["image_type"]()
                image_message.header = header
                image_message.format = "jpeg"
                image_message.data = overlay
                summary_message = run["string_type"]()
                summary_message.data = json.dumps(summary, ensure_ascii=False)
                with self._lock:
                    if run["stop"].is_set():
                        break
                    run["overlay"].publish(image_message)
                    run["summary"].publish(summary_message)
                    self._frames_processed += 1
                    self._last_frame_error = None
                run["stop"].wait(max(0.0, 1.0 / fps - (time.monotonic() - started)))
        except Exception as exc:
            failure = str(exc)
        finally:
            with self._lock:
                stopped = run["stop"].is_set()
                run["stop"].set()
            cleanup_error = None
            try:
                self._destroy_ros(run)
            except Exception as exc:
                cleanup_error = f"释放语义分割 ROS 资源失败：{exc}"
            with self._lock:
                run["cleanup_error"] = cleanup_error
                if self._run is run:
                    if cleanup_error or run.get("ros_error"):
                        self._state = "error"
                        self._last_error = cleanup_error or run["ros_error"]
                    elif stopped:
                        self._state = "idle"
                        self._model_status = None
                    else:
                        self._state = "error"
                        self._last_error = failure or "分割工作线程意外停止"
                run["finished"].set()

    @staticmethod
    def _destroy_ros(run):
        node = run.get("node")
        executor = run.get("executor")
        spin_thread = run.get("spin_thread")
        if executor is not None:
            if node is not None:
                executor.remove_node(node)
            executor.shutdown()
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join()
        if node is not None:
            node.destroy_node()
