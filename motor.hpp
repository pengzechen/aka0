#ifndef MOTOR_HPP
#define MOTOR_HPP

#include <string>

class Motor {
  public:
    Motor();
    ~Motor();

    void forward(int speed);
    void backward(int speed);
    void left(int speed);
    void right(int speed);
    void brake();
    void standby();

  private:
    void set_pwm_duty_cycle(int pwm_id, int duty_cycle);
    void set_pwm_enable(int pwm_id, bool enable);
    void set_speed(int pwm_id, int speed);
    void init_pwm(int pwm_id);

    const std::string PWM_PATH = "/sys/class/pwm/pwmchip4/";
    const int PERIOD = 10000; // 10kHz

    // PWM IDs for left and right wheels
    const int LEFT_WHEEL_BACKWARD = 0;
    const int LEFT_WHEEL_FORWARD = 1;
    const int RIGHT_WHEEL_BACKWARD = 2;
    const int RIGHT_WHEEL_FORWARD = 3;
};

#endif // MOTOR_HPP
