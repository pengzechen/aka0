#include "motor.hpp"
#include <fstream>
#include <iostream>
#include <unistd.h>

Motor::Motor() {
    init_pwm(LEFT_WHEEL_BACKWARD);
    init_pwm(LEFT_WHEEL_FORWARD);
    init_pwm(RIGHT_WHEEL_BACKWARD);
    init_pwm(RIGHT_WHEEL_FORWARD);
}

Motor::~Motor() {
    standby();
}

void Motor::init_pwm(int pwm_id) {
    std::ofstream ofs_export(PWM_PATH + "export");
    if (ofs_export.is_open()) {
        ofs_export << pwm_id;
        ofs_export.close();
    }

    std::string pwm_channel_path = PWM_PATH + "pwm" + std::to_string(pwm_id);

    std::ofstream ofs_period(pwm_channel_path + "/period");
    if (ofs_period.is_open()) {
        ofs_period << PERIOD;
        ofs_period.close();
    }
}

void Motor::set_pwm_duty_cycle(int pwm_id, int duty_cycle) {
    std::string duty_cycle_path = PWM_PATH + "pwm" + std::to_string(pwm_id) + "/duty_cycle";
    std::ofstream ofs(duty_cycle_path);
    if (ofs.is_open()) {
        ofs << duty_cycle;
        ofs.close();
    }
}

void Motor::set_pwm_enable(int pwm_id, bool enable) {
    std::string enable_path = PWM_PATH + "pwm" + std::to_string(pwm_id) + "/enable";
    std::ofstream ofs(enable_path);
    if (ofs.is_open()) {
        ofs << (enable ? "1" : "0");
        ofs.close();
    }
}

void Motor::set_speed(int pwm_id, int speed) {
    if (speed < 0)
        speed = 0;
    if (speed > 100)
        speed = 100;
    int duty_cycle = (speed / 100.0) * PERIOD;
    set_pwm_duty_cycle(pwm_id, duty_cycle);
}

void Motor::forward(int speed) {
    set_speed(LEFT_WHEEL_FORWARD, speed);
    set_pwm_enable(LEFT_WHEEL_FORWARD, true);
    set_pwm_enable(LEFT_WHEEL_BACKWARD, false);

    set_speed(RIGHT_WHEEL_FORWARD, speed);
    set_pwm_enable(RIGHT_WHEEL_FORWARD, true);
    set_pwm_enable(RIGHT_WHEEL_BACKWARD, false);
}

void Motor::backward(int speed) {
    set_speed(LEFT_WHEEL_BACKWARD, speed);
    set_pwm_enable(LEFT_WHEEL_BACKWARD, true);
    set_pwm_enable(LEFT_WHEEL_FORWARD, false);

    set_speed(RIGHT_WHEEL_BACKWARD, speed);
    set_pwm_enable(RIGHT_WHEEL_BACKWARD, true);
    set_pwm_enable(RIGHT_WHEEL_FORWARD, false);
}

void Motor::left(int speed) {
    set_speed(RIGHT_WHEEL_FORWARD, speed);
    set_pwm_enable(RIGHT_WHEEL_FORWARD, true);
    set_pwm_enable(RIGHT_WHEEL_BACKWARD, false);

    set_pwm_enable(LEFT_WHEEL_FORWARD, false);
    set_pwm_enable(LEFT_WHEEL_BACKWARD, false);
}

void Motor::right(int speed) {
    set_speed(LEFT_WHEEL_FORWARD, speed);
    set_pwm_enable(LEFT_WHEEL_FORWARD, true);
    set_pwm_enable(LEFT_WHEEL_BACKWARD, false);

    set_pwm_enable(RIGHT_WHEEL_FORWARD, false);
    set_pwm_enable(RIGHT_WHEEL_BACKWARD, false);
}

void Motor::brake() {
    set_pwm_enable(LEFT_WHEEL_FORWARD, true);
    set_pwm_enable(LEFT_WHEEL_BACKWARD, true);
    set_pwm_enable(RIGHT_WHEEL_FORWARD, true);
    set_pwm_enable(RIGHT_WHEEL_BACKWARD, true);
}

void Motor::standby() {
    set_pwm_enable(LEFT_WHEEL_FORWARD, false);
    set_pwm_enable(LEFT_WHEEL_BACKWARD, false);
    set_pwm_enable(RIGHT_WHEEL_FORWARD, false);
    set_pwm_enable(RIGHT_WHEEL_BACKWARD, false);
}
