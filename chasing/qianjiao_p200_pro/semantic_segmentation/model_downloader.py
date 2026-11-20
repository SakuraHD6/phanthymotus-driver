"""Fetch the author's SUIM checkpoint when it is missing from the model cache."""

import hashlib
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import parse_qs, urlparse

try:
    import fcntl
except ImportError:  # The deployed container is Linux; development hosts may not be.
    fcntl = None


MODEL_DIR = Path("/models/uwsegformer")
DRIVE_FOLDER_ID = "1-qtLVDZzWg62khOEaPZXmM3I_8noA4WW"
DRIVE_FILENAME = "uws_SUIM.pth"
_FILE_ID = re.compile(r"[A-Za-z0-9_-]+")


def _file_id_from_url(url):
    """Accept a Google Drive file link, never a folder or arbitrary URL."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "drive.google.com":
        raise ValueError("请填写 Google Drive 单文件分享链接")
    match = re.fullmatch(
        r"/file/(?:u/\d+/)?d/([A-Za-z0-9_-]+)(?:/(?:view|edit|preview))?/?",
        parsed.path,
    )
    if match:
        return match.group(1)
    if parsed.path in ("/open", "/uc"):
        ids = parse_qs(parsed.query).get("id", [])
        if len(ids) == 1 and _FILE_ID.fullmatch(ids[0]):
            return ids[0]
    raise ValueError("请填写 Google Drive 单文件分享链接，不要填写文件夹链接")


def _check_cancelled(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("已停止下载 SUIM 权重")


def _lock(lock_file, stop_event):
    if fcntl is None:
        return
    while True:
        _check_cancelled(stop_event)
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if stop_event is None:
                time.sleep(0.2)
            else:
                stop_event.wait(0.2)


def _find_suim_file(gdown, file_url, status, stop_event):
    if file_url:
        return _file_id_from_url(file_url)

    status("正在查找作者的 SUIM 权重")
    _check_cancelled(stop_event)
    try:
        files = gdown.download_folder(
            id=DRIVE_FOLDER_ID,
            skip_download=True,
            quiet=True,
            use_cookies=False,
            timeout=(15, 60),
        )
    except Exception as exc:
        raise RuntimeError(
            "无法读取作者的 Google Drive 目录；请检查网络或访问权限，"
            "也可在 config.yaml 中设置 google_drive_file_url：" + str(exc)
        ) from exc
    _check_cancelled(stop_event)
    matches = [item for item in files if Path(item.path).name == DRIVE_FILENAME]
    if not matches:
        raise FileNotFoundError(
            f"作者的 Google Drive 目录中未找到 {DRIVE_FILENAME}；"
            "请检查共享权限或设置 google_drive_file_url"
        )
    if len(matches) > 1:
        raise ValueError(
            f"作者目录中有多个 {DRIVE_FILENAME}，请设置 google_drive_file_url 指定文件"
        )
    file_id = matches[0].id
    if not isinstance(file_id, str) or not _FILE_ID.fullmatch(file_id):
        raise ValueError("Google Drive 返回了无效的权重文件 ID")
    return file_id


def _verify_checkpoint(path, expected_sha256, stop_event=None):
    if path.stat().st_size == 0:
        raise ValueError(f"模型权重为空：{path}")
    if not expected_sha256:
        return
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            _check_cancelled(stop_event)
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"模型权重 SHA256 不匹配：{path}")


def ensure_suim_checkpoint(checkpoint, file_url="", status_cb=None, stop_event=None,
                           expected_sha256=""):
    """Return a local checkpoint, downloading the author's file only if absent.

    An existing checkpoint may be anywhere. Automatic downloads are restricted
    to the mounted model cache, so a card-configured path cannot write elsewhere.
    """
    target = Path(checkpoint).expanduser().resolve()
    expected_sha256 = str(expected_sha256 or "").strip().lower()
    if expected_sha256 and not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("checkpoint_sha256 必须是 64 位十六进制 SHA256")

    def status(message):
        if status_cb is not None:
            status_cb(message)

    _check_cancelled(stop_event)
    if target.is_file():
        if expected_sha256:
            status("正在校验本地 SUIM 权重")
        _verify_checkpoint(target, expected_sha256, stop_event)
        return str(target)
    if target.exists():
        raise ValueError(f"模型权重路径不是文件：{target}")

    model_dir = MODEL_DIR.resolve()
    if target == model_dir or model_dir not in target.parents:
        raise ValueError(f"自动下载的权重必须保存在 {model_dir} 目录中")

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("缺少 gdown；请使用包含语义分割依赖的驱动镜像") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.lock"
    with lock_path.open("a+b") as lock_file:
        _lock(lock_file, stop_event)
        try:
            _check_cancelled(stop_event)
            if target.is_file():
                if expected_sha256:
                    status("正在校验本地 SUIM 权重")
                _verify_checkpoint(target, expected_sha256, stop_event)
                return str(target)
            if target.exists():
                raise ValueError(f"模型权重路径不是文件：{target}")

            file_id = _find_suim_file(gdown, file_url.strip(), status, stop_event)
            status("正在下载 UWSegFormer SUIM 权重")

            def progress(done, total):
                done_mb = done / (1024 * 1024)
                if total and total > 0:
                    percent = min(100, int(done * 100 / total))
                    total_mb = total / (1024 * 1024)
                    status(f"正在下载 SUIM 权重：{percent}%（{done_mb:.1f}/{total_mb:.1f} MB）")
                else:
                    status(f"正在下载 SUIM 权重：已接收 {done_mb:.1f} MB")

            with tempfile.TemporaryDirectory(prefix=".uwsegformer-", dir=target.parent) as staging:
                temporary = Path(staging) / DRIVE_FILENAME
                try:
                    gdown.download(
                        id=file_id,
                        output=str(temporary),
                        quiet=True,
                        use_cookies=False,
                        timeout=(15, 60),
                        retries=2,
                        progress=progress,
                        cancel=stop_event,
                    )
                except Exception as exc:
                    raise RuntimeError(f"Google Drive 权重下载失败：{exc}") from exc
                _check_cancelled(stop_event)
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise RuntimeError("Google Drive 未返回有效的权重文件")
                if expected_sha256:
                    status("正在校验下载的 SUIM 权重")
                _verify_checkpoint(temporary, expected_sha256, stop_event)
                if target.is_file():
                    _verify_checkpoint(target, expected_sha256, stop_event)
                    return str(target)
                if target.exists():
                    raise ValueError(f"模型权重路径不是文件：{target}")
                os.replace(temporary, target)
            return str(target)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
