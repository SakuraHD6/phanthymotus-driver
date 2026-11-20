# Qianjiao P200 Pro ROV Driver

本驱动按供应商《如何集成 Mini 系列、P 系列 ROV 运动控制功能？》实现 MAVLink v1 UDP 控制：

- ROV 监听 UDP `14550`；`target_ip`/`target_port` 在 `config.yaml` 配置。
- 每秒发送 `HEARTBEAT`，超过 3 秒没有收到 ROV 心跳则拒绝运动。
- `control.unlock`/`lock` 使用 `COMMAND_LONG`，命令 `400`，`param1=0/1`。
- `control.move` 使用 `RC_CHANNELS_OVERRIDE`。输入轴为 `[-1, 1]`，映射到 PWM `1100..1900`，中位 `1500`：
  `heave=chan1`、`pitch=chan2`、`forward=chan3`、`yaw=chan4`、`lateral=chan5`、`roll=chan7`（chan6 保留）。
- `status` 提供连接、心跳年龄、温度、视频代理状态和最近错误。
- `status` 同时监听 UDP `8500` 的状态广播；`loco_state`、`battery`、`imu` 分别提供姿态/深度/定位、电池和 IMU 数据。
- ROS2 状态主题发布到 Agent Core 的 DDS domain `42`，并使用宿主机挂载的 loopback-only `dds-local.xml`；ROV 本体链路仍通过 MAVLink/UDP，不依赖 DDS。
- MAVLink 目标 system/component ID 默认是 `1/1`，收到 ROV 的 `HEARTBEAT` 后会自动切换为心跳来源 ID；也可通过 `target_system`、`target_component` 显式指定初始值。

相机卡片：

- `camera` 提供 RTSP 转 JPEG 的实时视频卡；当前固件已确认的 HTTP 能力不足以稳定支持拍照和补光灯控制。

当前配置内置设备默认账号 `admin/admin`；如设备密码已修改，可通过 `QIANJIAO_CAMERA_USER`、`QIANJIAO_CAMERA_PASSWORD` 环境变量覆盖。部署时仍需设置 `MCP_ADVERTISE_HOST`（板卡可被 Agent Core 访问的地址）。密码不会通过 MCP 返回值暴露。

将 `mock: true` 用于无硬件开发测试。接入真实设备前，先通过 QGroundControl 验证 UDP 链路并确认解锁/失联保护行为。

镜像依赖说明：`pymavlink` 负责 MAVLink v1 控制和心跳，`ffmpeg` 负责 RTSP 到 MJPEG 转码，ROS 运行时包负责发布状态、电池、IMU 和图像主题；这些均为当前卡片的运行时必需依赖。已完成 Python 编译、状态包解析和 mock 控制/持续运动取消冒烟测试；真实设备验证应记录心跳 ID、六轴通道、锁定保护、UDP 8500 和 RTSP 结果。

## 水下语义分割

`semantic_segmentation` 是 processor 卡片。将 `camera` 的 `image/jpeg`
输出连接到该卡片，启动后它发布两路数据：

- `/qianjiao_p200_pro/semantic_segmentation/overlay`，`image/jpeg`，原图与分割颜色的叠加。
- `/qianjiao_p200_pro/semantic_segmentation/summary`，`data/json`，原图尺寸、输入帧时间戳、推理耗时以及各类别的像素数和占比。

模型采用 [UWSegFormer](https://github.com/SAWRJJ/UWSegFormer) 的 SUIM 六类配置：
水体、潜水员、沉船或遗迹、机器人、礁石或无脊椎动物、鱼类或脊椎动物。
卡片保留最新输入帧，处理线程单独加载模型并进行推理。画面按比例缩放和填充到
640×480，分割结果再映射回输入画面尺寸。

启动卡片时，如果本地还没有权重，程序会查找 UWSegFormer 作者的
[Google Drive 目录](https://drive.google.com/drive/folders/1-qtLVDZzWg62khOEaPZXmM3I_8noA4WW)
中的 `uws_SUIM.pth`，只下载该文件到 `/models/uwsegformer/suim.pth`。首次下载需要
部署设备能够访问 Google Drive 和该文件；卡片 `info` 的 `desc` 和 `model_status`
会显示下载状态。
如果目录读取失败或有同名文件，可在 `config.yaml` 的 `google_drive_file_url` 填写该
权重的 Google Drive 单文件分享链接，并确保部署容器读取更新后的配置。默认配置由
镜像构建时复制到容器；也可以通过 `CONFIG_PATH` 指定容器内的其他配置文件。
文件下载阶段支持停止；如果停止时正在扫描 Google Drive 目录，卡片会保持
`stopping`，直到目录扫描返回。

也可将作者的 SUIM 权重预先放在部署主机的
`/opt/embodied/models/uwsegformer/suim.pth`，已有文件会直接使用。模型目录以可写方式
挂载到容器的 `/models/uwsegformer`，以便保存首次下载的权重；自动下载限于此目录。
其他本地权重可以在卡片配置中指定，并需预先放置在对应路径。
`device` 默认为 `cpu`，`fps` 为处理频率上限，`opacity` 控制分割颜色的透明度。
修改权重路径或推理设备后，停止并重新启动卡片以加载新模型。
无法解码的 JPEG 帧会被跳过，推理错误会停止卡片并报告；`info` 包含收到的帧数、
坏帧数和最近一帧的间隔。

驱动镜像构建时获取固定版本的 UWSegFormer 源码，修复其 SUIM 配置引用与导入，
并安装语义分割依赖（包括 `gdown`）。构建镜像时不下载权重。作者未在项目说明中
提供该权重的独立校验值；如果从可信渠道获得 SHA256，可将其填入 `config.yaml` 的
`checkpoint_sha256`，加载已有文件或下载新文件时都会校验。留空时不做哈希校验；
文件名匹配和模型加载也不能证明文件来源或模型效果。若下载、权重、源码或依赖
不可用，卡片的 `info` 会报告错误信息。

UWSegFormer 的部分 SegFormer 源文件声明了 NVIDIA Source Code License。原始
[SegFormer 许可](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)要求保留许可文件，
并将相关代码限定于非商业的研究或评估用途。UWSegFormer 仓库本身未提供完整许可文件，
发布镜像或用于其他用途前需核实模型源码及权重的授权范围。
