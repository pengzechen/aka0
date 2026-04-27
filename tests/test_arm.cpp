// Author: Zhihang Shao <dio_ro@outlook.com>
// Description: 三舵机调试工具 (CLI模式)
//
// Usage:
//   ./test_arm grab              抓取序列
//   ./test_arm release           松开夹爪
//   ./test_arm show              抬起展示
//   ./test_arm pos               回到待抓取位置
//   ./test_arm demo              演示: pos -> grab -> show -> release
//   ./test_arm 240 220 150       设置三个舵机角度 (0~270)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "arm.hpp"

static void usage(const char *prog)
{
    fprintf(stderr,
        "Servo arm test tool\n\n"
        "Usage:\n"
        "  %s <command>        Execute a preset command\n"
        "  %s <a0> <a1> <a2>   Set servo angles (0~270)\n\n"
        "Commands:\n"
        "  grab     Execute grab sequence\n"
        "  release  Release gripper\n"
        "  show     Show ball (lift up)\n"
        "  pos      Move to ready (home) position\n"
        "  demo     Run demo: pos -> grab -> show -> release\n",
        prog, prog);
}

int main(int argc, char **argv)
{
    Arm arm;

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char *cmd = argv[1];

    if (strcmp(cmd, "-h") == 0 || strcmp(cmd, "--help") == 0) {
        usage(argv[0]);
        return 0;
    }

    if (strcmp(cmd, "grab") == 0) {
        printf("Executing grab...\n"); fflush(stdout);
        arm.grab();
        printf("Done.\n");
        return 0;
    }
    if (strcmp(cmd, "release") == 0) {
        printf("Releasing gripper...\n"); fflush(stdout);
        arm.release();
        printf("Done.\n");
        return 0;
    }
    if (strcmp(cmd, "show") == 0) {
        printf("Showing ball...\n"); fflush(stdout);
        arm.show();
        printf("Done.\n");
        return 0;
    }
    if (strcmp(cmd, "pos") == 0) {
        printf("Moving to ready position...\n"); fflush(stdout);
        arm.grab_pos();
        printf("Done.\n");
        return 0;
    }
    if (strcmp(cmd, "demo") == 0) {
        printf("Demo: pos -> grab -> show -> release\n"); fflush(stdout);
        arm.grab_pos();
        usleep(1500 * 1000);
        arm.grab();
        usleep(2000 * 1000);
        arm.show();
        usleep(2000 * 1000);
        arm.release();
        printf("Demo done.\n");
        return 0;
    }

    /* Parse as three angles */
    if (argc == 4) {
        float a0 = atof(argv[1]);
        float a1 = atof(argv[2]);
        float a2 = atof(argv[3]);
        printf("servo 0=%.0f 1=%.0f 2=%.0f\n", a0, a1, a2);
        fflush(stdout);
        arm.set_angle(0, a0);
        arm.set_angle(1, a1);
        arm.set_angle(2, a2);
        return 0;
    }

    fprintf(stderr, "Unknown command: %s\n", cmd);
    usage(argv[0]);
    return 1;
}
