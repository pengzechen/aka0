// Author: Zhihang Shao <dio_ro@outlook.com>
// Source: aka0-ref commits 755a885, 64f8dab, 7b7011d, 9c69f3f, d4fbdad
// Description: ZP10D 舵机 UART 控制实现

#include "arm.hpp"
#include "logger.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <cstdio>
#include <algorithm>

// Author: Zhihang Shao <dio_ro@outlook.com>
// Source: aka0-ref commit d4fbdad
// 舵机角度实测校准
// 初始:    0=200 1=200 2=175(开)
// 伸下:    0=240 1=175 2=175(开)
// 夹紧:    0=240 1=175 2=135(闭)
// 展示:    0=165 1=180 2=135(闭)
// 放球:    0=165 1=180 2=175(开)
const float Arm::ID2_ANGLE_OPEN = 175.0f;
const float Arm::ID2_ANGLE_CLOSE = 135.0f;
const float Arm::ANGLE_MAX = 270.0f;

const float Arm::SERVO0_READY = 200.0f;
const float Arm::SERVO1_READY = 200.0f;
const float Arm::SERVO0_GRAB = 240.0f;
const float Arm::SERVO1_GRAB = 175.0f;
const float Arm::SERVO0_LIFT = 165.0f;
const float Arm::SERVO1_LIFT = 180.0f;

Arm::Arm(const std::string& port, int baudrate) : fd_(-1) {
    open_serial(port, baudrate);
}

Arm::~Arm() {
    close_serial();
}

void Arm::open_serial(const std::string& port, int baudrate) {
    fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd_ < 0) {
        LOGE("[ARM] Failed to open serial port %s: %s", port.c_str(), strerror(errno));
        return;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd_, &tty) != 0) {
        LOGE("[ARM] tcgetattr failed: %s", strerror(errno));
        close_serial();
        return;
    }

    speed_t baud;
    switch (baudrate) {
        case 9600:   baud = B9600;   break;
        case 115200: baud = B115200; break;
        default:     baud = B115200; break;
    }
    cfsetispeed(&tty, baud);
    cfsetospeed(&tty, baud);

    // 8N1, no flow control
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CLOCAL | CREAD;

    // Raw mode
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(INLCR | ICRNL | IGNCR);
    tty.c_oflag &= ~OPOST;

    // 100ms timeout
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;

    tcflush(fd_, TCIFLUSH);
    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        LOGE("[ARM] tcsetattr failed: %s", strerror(errno));
        close_serial();
        return;
    }

    LOGI("[ARM] Serial port %s opened at %d baud", port.c_str(), baudrate);
}

void Arm::close_serial() {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

void Arm::send_command(const std::string& cmd) {
    if (fd_ < 0) {
        LOGE("[ARM] Serial port not open, cannot send: %s", cmd.c_str());
        return;
    }
    write(fd_, cmd.c_str(), cmd.size());
    tcdrain(fd_);
}

int Arm::angle_to_pulse(float angle) const {
    int pulse = static_cast<int>(500 + (angle / ANGLE_MAX) * 2000);
    return std::max(PULSE_MIN, std::min(PULSE_MAX, pulse));
}

void Arm::set_angle(int servo_id, float angle, int time_ms) {
    if (angle < 0 || angle > ANGLE_MAX) {
        LOGW("[ARM] Angle %.1f out of range [0, %.0f], clamping", angle, ANGLE_MAX);
        angle = std::max(0.0f, std::min(ANGLE_MAX, angle));
    }
    int pulse = angle_to_pulse(angle);
    char buf[32];
    snprintf(buf, sizeof(buf), "#%03dP%04dT%d!", servo_id, pulse, time_ms);
    send_command(buf);
    LOGD("[ARM] servo %d -> angle %.1f (pulse %d, %dms)", servo_id, angle, pulse, time_ms);
}

void Arm::release_torque(int servo_id) {
    char buf[16];
    snprintf(buf, sizeof(buf), "#%03dPULK", servo_id);
    send_command(buf);
    LOGD("[ARM] servo %d torque released", servo_id);
}

void Arm::restore_torque(int servo_id) {
    char buf[16];
    snprintf(buf, sizeof(buf), "#%03dPULR", servo_id);
    send_command(buf);
    LOGD("[ARM] servo %d torque restored", servo_id);
}

// Grab sequence
// 初始(200,200,175开) → 伸下(240,175,175开) → 夹紧(240,175,135闭)
// → 展示(165,180,135闭) → 放球(165,180,175开) → 回初始(200,200,175开)
void Arm::grab() {
    LOGI("[ARM] Grab sequence start");

    // 1. 伸下去，爪子打开
    set_angle(0, SERVO0_GRAB);
    set_angle(1, SERVO1_GRAB);
    set_angle(2, ID2_ANGLE_OPEN);
    usleep(1500 * 1000);

    // 2. 爪子闭合
    set_angle(2, ID2_ANGLE_CLOSE);
    usleep(1000 * 1000);

    // 3. 抬起展示
    set_angle(0, SERVO0_LIFT);
    set_angle(1, SERVO1_LIFT);
    usleep(1000 * 1000);

    // 4. 松开球
    set_angle(2, ID2_ANGLE_OPEN);
    usleep(800 * 1000);

    // 5. 回到初始
    set_angle(0, SERVO0_READY);
    set_angle(1, SERVO1_READY);
    usleep(800 * 1000);

    LOGI("[ARM] Grab sequence done");
}

void Arm::release_pos() {
    LOGI("[ARM] Moving to release position");
    set_angle(0, SERVO0_READY);
    set_angle(1, SERVO1_READY);
    set_angle(2, ID2_ANGLE_CLOSE);
}

void Arm::release() {
    LOGI("[ARM] Releasing gripper");
    set_angle(0, SERVO0_LIFT);
    set_angle(1, SERVO1_LIFT);
    set_angle(2, ID2_ANGLE_OPEN);
}

void Arm::grab_pos() {
    LOGI("[ARM] Moving to home/ready position");
    set_angle(0, SERVO0_READY);
    set_angle(1, SERVO1_READY);
    set_angle(2, ID2_ANGLE_OPEN);
}

void Arm::show() {
    LOGI("[ARM] Showing ball - lifting up high");
    set_angle(0, SERVO0_LIFT);
    set_angle(1, SERVO1_LIFT);
    set_angle(2, ID2_ANGLE_CLOSE);
}
