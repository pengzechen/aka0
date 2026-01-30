#!/bin/bash
# Standalone build script for sensor_test in aka0 repository
# 用法: ./build.sh [clean]

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 如果参数是 clean，执行清理
if [ "$1" == "clean" ]; then
    echo "清理 sensor_test..."
    make clean
    echo "清理完成"
    exit 0
fi

# 检查交叉编译工具链
CROSS_COMPILE_PATH="/home/ajax/Projects/sg2002/LicheeRV-Nano-Build/host-tools/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-gcc"
if [ ! -f "$CROSS_COMPILE_PATH" ]; then
    echo "错误: 交叉编译工具链不存在"
    echo "路径: $CROSS_COMPILE_PATH"
    echo "请检查 LicheeRV-Nano-Build 是否已正确安装"
    exit 1
fi

# 检查必要的库文件
echo "检查依赖库..."
REQUIRED_LIBS="libsys.a libcvi_ispd2.a libjson-c.a libvpu.a libsample.a libisp.a libae.a libawb.a libini.a"
for lib in $REQUIRED_LIBS; do
    if [ ! -f "lib/$lib" ]; then
        echo "错误: 缺少库文件 lib/$lib"
        exit 1
    fi
done
echo "✓ 所有依赖库检查通过"

# 检查头文件
echo "检查头文件..."
if [ ! -d "include/isp" ]; then
    echo "错误: 缺少 ISP 头文件目录 include/isp"
    exit 1
fi
if [ ! -d "include/common" ]; then
    echo "错误: 缺少 common 头文件目录 include/common"
    exit 1
fi
echo "✓ 头文件检查通过"

# 执行编译
echo ""
echo "开始编译 sensor_test..."
START_TIME=$(date +%s)

if make -j$(nproc) 2>&1 | tee /tmp/sensor_test_build.log; then
    if [ -f "sensor_test" ]; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo ""
        echo "✓ 编译成功！耗时: ${DURATION}秒"
        echo "可执行文件: $(pwd)/sensor_test"
        ls -lh sensor_test
        
        # 显示文件信息
        echo ""
        echo "文件信息:"
        file sensor_test
    else
        echo ""
        echo "✗ 编译失败: 未生成可执行文件"
        echo "查看完整日志: cat /tmp/sensor_test_build.log"
        exit 1
    fi
else
    echo ""
    echo "✗ 编译失败"
    echo "查看完整日志: cat /tmp/sensor_test_build.log"
    exit 1
fi
