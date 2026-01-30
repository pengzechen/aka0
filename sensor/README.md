# Sensor Test 独立编译环境

本目录包含从 LicheeRV-Nano-Build middleware 移植过来的 sensor_test 独立编译环境。

## 目录结构

```
sensor/
├── sensor_test.c       # 主程序源文件
├── ae_test.c           # AE 测试源文件
├── Makefile            # 独立 Makefile
├── build.sh            # 编译脚本
├── include/            # 头文件目录
│   ├── *.h            # middleware 公共头文件
│   ├── ae_test.h      # 本地头文件
│   ├── linux/         # kernel 头文件
│   ├── isp/           # ISP 相关头文件
│   │   └── sg200x/    # SG200x 芯片 ISP 头文件
│   └── common/        # sample 公共头文件
└── lib/                # 静态库文件目录
    ├── libsys.a
    ├── libcvi_ispd2.a
    ├── libjson-c.a
    ├── libvpu.a
    ├── libsample.a
    ├── libisp.a
    ├── libisp_algo.a
    ├── libae.a
    ├── libawb.a
    ├── libaf.a
    ├── libcvi_bin.a
    ├── libcvi_bin_isp.a
    ├── libini.a
    └── libsns_*.a      # 各种 sensor 驱动库
```

## 依赖说明

### 静态库依赖
所有必需的库已复制到 `lib/` 目录，使用静态链接方式：

- **系统库**: libsys.a
- **ISP 库**: libisp.a, libisp_algo.a, libcvi_ispd2.a, libcvi_bin.a, libcvi_bin_isp.a
- **3A 算法库**: libae.a, libawb.a, libaf.a
- **VPU 库**: libvpu.a
- **Sample 库**: libsample.a
- **工具库**: libjson-c.a, libini.a
- **Sensor 驱动**: libsns_full.a, libsns_gc4653.a, libsns_sc035gs.a, libsns_os04a10.a, libsns_lt6911.a

### 头文件依赖
所有必需的头文件已复制到 `include/` 目录。

### 交叉编译工具链
使用 LicheeRV-Nano-Build 提供的 RISC-V musl 工具链：
```
/home/ajax/Projects/sg2002/LicheeRV-Nano-Build/host-tools/gcc/riscv64-linux-musl-x86_64/bin/
```

## 编译方法

### 快速编译
```bash
./build.sh
```

### 清理
```bash
./build.sh clean
```

### 手动编译
```bash
make
```

## 编译选项

Makefile 中的关键编译选项：

- **CPU 优化**: `-mcpu=c906fdv -march=rv64imafdcv0p7xthead`
- **代码模型**: `-mcmodel=medany`
- **ABI**: `-mabi=lp64d`
- **优化级别**: `-Os` (大小优化)

## 输出文件

编译成功后生成：
- `sensor_test`: 可执行文件 (约 8.2 MB)
- 目标架构: RISC-V 64-bit, 动态链接

## 原始来源

本环境移植自:
```
/home/ajax/Projects/sg2002/LicheeRV-Nano-Build/middleware/v2/sample/sensor_test
```

## 注意事项

1. 确保 LicheeRV-Nano-Build 工具链路径正确
2. 所有库文件使用静态链接，无需在目标设备上安装额外依赖
3. 编译脚本会自动检查依赖库和头文件的完整性
4. 如需更新源文件或添加新功能，需同时更新 `Makefile` 中的 `SRCS` 变量

## 迁移日期

2026-01-30
