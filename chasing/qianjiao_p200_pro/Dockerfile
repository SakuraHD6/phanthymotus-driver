ARG ROS_BASE_IMAGE=bj-warehouse.tencentcloudcr.com/phanthy-motus/ros-base:latest
FROM ${ROS_BASE_IMAGE}
ARG PYPI_MIRROR=https://pypi.org/simple/
ARG UBUNTU_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports
ARG UWSEGFORMER_REV=b61fa450bad58137b3ab8fecd1e7f7cedba4ce57
WORKDIR /work
RUN grep -RIl 'mirrors.tencentyun.com/ubuntu-ports' /etc/apt 2>/dev/null | \
    xargs -r sed -i "s#http://mirrors.tencentyun.com/ubuntu-ports#${UBUNTU_MIRROR}#g" && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-pip git ffmpeg libgl1 libglib2.0-0 ros-humble-rclpy ros-humble-sensor-msgs ros-humble-std-msgs && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -i "${PYPI_MIRROR}" -r requirements.txt
COPY semantic_segmentation /work/semantic_segmentation
RUN pip3 install --no-cache-dir -i "${PYPI_MIRROR}" -r semantic_segmentation/requirements.txt
RUN git init /opt/uwsegformer && \
    git -C /opt/uwsegformer remote add origin https://github.com/SAWRJJ/UWSegFormer.git && \
    git -C /opt/uwsegformer fetch --depth 1 origin "${UWSEGFORMER_REV}" && \
    git -C /opt/uwsegformer checkout --detach FETCH_HEAD && \
    python3 /work/semantic_segmentation/prepare_source.py /opt/uwsegformer && \
    cp /work/semantic_segmentation/NVIDIA_SEGFORMER_LICENSE.txt /opt/uwsegformer/NVIDIA_SEGFORMER_LICENSE.txt
COPY main.py device.py config.yaml driver.yaml ./
COPY common /work/common
COPY deploy/service.yml /deploy/service.yml
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/work
ENV PYTHONPATH=/work:/opt/ros/humble/lib/python3.10/site-packages:/usr/lib/python3/dist-packages:/usr/local/lib/python3.10/dist-packages
ENV ROS_DOMAIN_ID=42
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/opt/phanthy-motus/dds-local.xml
ENV ROS_LOCALHOST_ONLY=0
ENV RCUTILS_COLORIZED_OUTPUT=0
EXPOSE 15739
CMD ["/bin/bash", "-lc", "source /opt/ros/humble/setup.bash && exec /usr/bin/python3 main.py"]
