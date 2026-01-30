#!/bin/bash

# 设置交叉编译工具链
export CC=riscv64-unknown-linux-musl-gcc
export CXX=riscv64-unknown-linux-musl-g++

# 设置SDK路径，请根据实际情况修改
TPU_SDK_PATH=${TPU_SDK_PATH:-"/home/ajax/Projects/sg2002/sdk-samples/cvitek_tpu_sdk"}
OPENCV_PATH=${OPENCV_PATH:-"/home/ajax/Projects/sg2002/sdk-samples/cvitek_tpu_sdk/opencv"}

echo "Using TPU_SDK_PATH: $TPU_SDK_PATH"
echo "Using OPENCV_PATH: $OPENCV_PATH"

# 清理之前的构建
rm -rf build_riscv_debug
mkdir -p build_riscv_debug
cd build_riscv_debug

# 配置cmake
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_C_FLAGS="-O2 -mcpu=c906fdv -march=rv64gcv0p7_zfh_xthead -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-O2 -mcpu=c906fdv -march=rv64gcv0p7_zfh_xthead -mabi=lp64d" \
    -DCMAKE_CROSSCOMPILING=ON \
    -DTPU_SDK_PATH=${TPU_SDK_PATH} \
    -DOPENCV_PATH=${OPENCV_PATH}

# 编译
make -j$(nproc)

echo "Debug version compiled successfully!"
echo "Executable: $(pwd)/tennis"