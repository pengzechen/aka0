#!/bin/bash
set -e

# 设置交叉编译工具链
export CC=riscv64-unknown-linux-musl-gcc
export CXX=riscv64-unknown-linux-musl-g++

# SDK路径: 环境变量 > 自动搜索 > Docker默认 > 本地默认
if [ -z "$TPU_SDK_PATH" ]; then
    TPU_SDK_PATH=$(find /opt /home -maxdepth 4 -name "cviruntime.h" 2>/dev/null | head -1)
    TPU_SDK_PATH=${TPU_SDK_PATH:+$(dirname "$(dirname "$TPU_SDK_PATH")")}
    : ${TPU_SDK_PATH:="/opt/cvitek_tpu_sdk"}
fi
if [ -z "$OPENCV_PATH" ]; then
    OPENCV_PATH=$(find "${TPU_SDK_PATH}" -maxdepth 3 -name "opencv.hpp" 2>/dev/null | head -1)
    OPENCV_PATH=${OPENCV_PATH:+$(dirname "$(dirname "$(dirname "$OPENCV_PATH")")")}
    : ${OPENCV_PATH:="${TPU_SDK_PATH}/opencv"}
fi

echo "Using TPU_SDK_PATH: $TPU_SDK_PATH"
echo "Using OPENCV_PATH:  $OPENCV_PATH"

# 清理之前的构建
rm -rf build_riscv_debug
mkdir -p build_riscv_debug
cd build_riscv_debug

# 配置cmake
LOG_LEVEL=${LOG_LEVEL:-debug}

ESP32_FLAG=""
if [ "${USE_ESP32}" = "1" ]; then
    ESP32_FLAG="-DUSE_ESP32_CAMERA=ON"
    echo "Camera: ESP32-CAM (ioctl mode)"
else
    echo "Camera: VI + VPSS (hardware mode)"
fi

cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_C_FLAGS="-O2 -mcpu=c906fdv -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-O2 -mcpu=c906fdv -mabi=lp64d" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--dynamic-linker=/lib/ld-musl-riscv64v0p7_xthead.so.1" \
    -DCMAKE_CROSSCOMPILING=ON \
    -DTPU_SDK_PATH=${TPU_SDK_PATH} \
    -DOPENCV_PATH=${OPENCV_PATH} \
    -DLOG_LEVEL=${LOG_LEVEL} \
    ${ESP32_FLAG}

make -j$(nproc)

echo ""
echo "=== Build Summary ==="
for bin in tennis test_arm test_motor test_capture_detect test_img_detect camera_test; do
    if [ -f "$bin" ]; then
        echo "  $bin  $(file $bin | sed 's/.*: //')"
    fi
done
