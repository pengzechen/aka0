# aka0 追球机器人部署指南

## 硬件平台

- **开发板**: LicheeRV Nano (RISC-V64, Cvitek SG200x SoC)
- **摄像头**: VI 通道 0, 分辨率 2560x1440 → VPSS 硬件缩放至 640x640
- **电机**: PWM sysfs (`/sys/class/pwm/pwmchip4/`), 4通道差速驱动
- **机械臂**: ZP10D 总线舵机 × 3, UART `/dev/ttyS2`, 115200bps

## 1. 交叉编译 (Docker)

### 构建镜像

```bash
cd /Users/stone/starry_for_car
docker build -t aka0-build .
```

### 编译

```bash
docker run --rm -v $(pwd)/aka0:/workspace -w /workspace aka0-build bash -c "\
  mkdir -p build && cd build && \
  cmake .. \
    -DTPU_SDK_PATH=/opt/cvitek_tpu_sdk \
    -DOPENCV_PATH=/opt/opencv \
    -DCMAKE_TOOLCHAIN_FILE=/opt/Xuantie-900-gcc-linux-6.6.36-musl64-x86_64-V3.3.0/Xuantie-900-gcc-linux-6.6.36-musl64-x86_64-V3.3.0/toolchain-riscv64-linux-musl-x86_64-V3.3.0/riscv64-unknown-linux-musl.cmake \
    -DCMAKE_C_FLAGS='-mcpu=c906fdv -mabi=lp64d' \
    -DCMAKE_CXX_FLAGS='-mcpu=c906fdv -mabi=lp64d' && \
  make -j$(nproc)"
```

编译产物在 `build/` 下：
- `tennis` — 主程序（追球+抓取）
- `capture` — 单帧拍照+YOLO检测工具
- `arm_test` — 舵机交互调试工具
- `motor_test` — 电机测试工具

## 2. 部署到板子

### 通过 WiFi SCP

```bash
# 板子连接到热点 ROS 后
scp build/tennis root@192.168.4.1:/root/
scp <model_path>/tennis_cv181x_bf16.cvimodel root@192.168.4.1:/root/
scp scripts/S98tennis.sh root@192.168.4.1:/etc/init.d/S98tennis
scp scripts/wpa_supplicant.conf root@192.168.4.1:/etc/wpa_supplicant.conf
```

### 通过 SD 卡

```bash
bash flash_to_sd.sh
```

## 3. 依赖库

板子上需要以下库（路径 `/usr/bin/lib/`）：

```bash
# 检查是否已有
ls /usr/bin/lib/libcviruntime.so
ls /usr/bin/lib/libopencv_core.so.3.2

# 缺少的话从 SDK 拷贝
scp cvitek_tpu_sdk/lib/libcviruntime.so* root@192.168.4.1:/usr/bin/lib/
scp cvitek_tpu_sdk/lib/libcvikernel.so* root@192.168.4.1:/usr/bin/lib/
scp cvitek_tpu_sdk/opencv/lib/libopencv_*.so* root@192.168.4.1:/usr/bin/lib/
```

## 4. WiFi 开机自连热点

配置文件 `/etc/wpa_supplicant.conf`：

```
ctrl_interface=/var/run/wpa_supplicant
ctrl_interface_group=0
update_config=1

network={
    ssid="ROS"
    psk="shaotonn"
    key_mgmt=WPA-PSK
}
```

启动脚本 `/etc/init.d/S99wifi_connect`：

```bash
# 确保脚本可执行
chmod +x /etc/init.d/S99wifi_connect
```

连接后板子 IP 为 `192.168.4.1`。

## 5. 追球程序开机自启

脚本 `/etc/init.d/S98tennis`：

```bash
chmod +x /etc/init.d/S98tennis
```

手动控制：

```bash
/etc/init.d/S98tennis start     # 启动
/etc/init.d/S98tennis stop      # 停止（正常清理硬件资源）
/etc/init.d/S98tennis restart   # 重启
```

日志输出：`/root/tennis.log`

## 6. 舵机角度（当前校准值）

| 阶段 | servo0 | servo1 | servo2 | 说明 |
|------|--------|--------|--------|------|
| 初始 (ready) | 200 | 200 | 210(开) | 抬起不挡摄像头 |
| 伸下 (grab) | 250 | 185 | 210(开) | 到球的位置 |
| 夹紧 (grab) | 250 | 185 | 150(闭) | 抓住球 |
| 抬起展示 (lift) | 150 | 185 | 150(闭) | 展示球 |
| 回位 | 200 | 200 | 150(闭) | 回到初始高度 |
| 松开 | 200 | 200 | 210(开) | 放球 |

## 7. 追球控制参数

在 `tennis.cpp` 中可调：

| 参数 | 值 | 说明 |
|------|----|------|
| GRAB_AREA | 0.40 | area_ratio 抓取阈值 |
| GRAB_AREA_MAX | 0.55 | 太近后退阈值 |
| CENTER_MARGIN | 35 | 居中判定 ±px |
| CHASE_SPEED | 56 | 前进速度 |
| TURN_SPEED | 18 | 转向速度 |
| K_TURN_PULSE | 5000.0 | 转向脉冲系数 (pulse = K × area) |
| TURN_PULSE_MIN | 75ms | 最小脉冲 |
| TURN_PULSE_MAX | 500ms | 最大脉冲 |
| GRAB_CONFIRM_THRESHOLD | 5 | 确认帧数 |
| GRAB_LEFT_TURN_COUNT | 4 | 抓取前左转补偿次数 |
| BACKWARD_SPEED | 18 | 后退速度 |
| BACKWARD_PULSE_US | 200ms | 后退脉冲时长 |

**PWM 注意**: 数值越大越慢（反转特性）

## 8. 调试工具

### 单帧拍照 + 检测

```bash
./capture tennis_cv181x_bf16.cvimodel /root/images/test.jpg
```

### 舵机交互调试

```bash
./arm_test
# 输入: 角度0 角度1 角度2
# 输入 grab/release/pos/q
```

### 日志开关

`tennis.cpp` 顶部：
```c
#define ENABLE_LOG       0    // 0=关闭所有日志只输出FPS, 1=打开
#define ENABLE_DRAW_BBOX 0    // 画检测框
#define ENABLE_SAVE_IMAGE 0   // 保存图片到 /root/images/
```

## 9. 常见问题

### VPSS 初始化失败 `CVI_VPSS_CreateGrp failed`

上次进程未正常退出，资源未释放。解决：`killall tennis` 等 2 秒再启动，或 `reboot`。

### 串口打不开机械臂不动

检查 `/dev/ttyS2` 是否存在：`ls -l /dev/ttyS2`。自启脚本会等待串口就绪。

### 库找不到 `libcviruntime.so`

```bash
export LD_LIBRARY_PATH=/usr/bin/lib:$LD_LIBRARY_PATH
```

自启脚本已包含此设置。
