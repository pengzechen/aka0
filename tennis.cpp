// Author: Zhihang Shao <dio_ro@outlook.com>
// Source: aka0-ref commits 64f8dab, 7b7011d, 9c69f3f, d4fbdad
// Description: 完整的追球抓取状态机 - 两状态(chase/grab) + 脉冲式转向 + 动态脉冲 + 太近后退 + Ctrl-C安全退出
// 
// 主要特性：
// - 两状态机: STATUS_CHASE_TENNIS, STATUS_GRAB_TENNIS
// - 使用 area_ratio (bbox面积/图像面积) 判断距离
// - 脉冲式转向防振荡，动态脉冲时间(与area线性相关)
// - 差速原地转向替代单轮转弯
// - 太近(area>0.55)自动后退，避免挤到球
// - 抓取前左转补偿，修正夹爪偏右
// - 完整抓取序列: 伸下→夹紧→抬起→停2秒→松开→复位
// - Ctrl-C 信号处理，安全清理资源

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include "cviruntime.h"
#include "motor.hpp"
#include "arm.hpp"
#include "detect.hpp"

#if USE_ESP32_CAMERA
#include "esp32_capture.hpp"
#else
// VI related headers - must include base types first
#include "vi_capture.hpp"
#endif
#include "logger.hpp"

// 控制宏定义
#define ENABLE_DRAW_BBOX 0    // 是否画框并保存图片
#define ENABLE_SAVE_IMAGE 0   // 是否保存检测结果图片


static const char* tennis_names[] = {"tennis"}; // 单类别网球检测

// 全局指针，用于信号处理中清理资源
#if USE_ESP32_CAMERA
static ESP32Capture* g_capture = nullptr;
#else
static VICapture* g_vi_capture = nullptr;
#endif
static Motor* g_motor = nullptr;
static CVI_MODEL_HANDLE g_model = nullptr;

static void cleanup_and_exit() {
    if (g_motor) g_motor->standby();
#if USE_ESP32_CAMERA
    if (g_capture) g_capture->deinit();
#else
  #if USE_VPSS_RESIZE
    if (g_vi_capture) g_vi_capture->deinitVpssResize();
  #endif
    if (g_vi_capture) g_vi_capture->deinit();
#endif
    if (g_model) CVI_NN_CleanupModel(g_model);
}

static void signal_handler(int sig) {
    cleanup_and_exit();
    exit(0);
}


static void usage(char** argv) {
    LOGI("Usage:");
    LOGI("   %s cvimodel [vi_channel]", argv[0]);
    LOGI("   Example: %s model.cvimodel 0", argv[0]);
    LOGI("   This will capture video from VI channel (default: 0)");
}


// ============ 状态机 ============

enum RobotStatus {
    STATUS_CHASE_TENNIS,     // 追球：area < GRAB_AREA 且球可见
    STATUS_GRAB_TENNIS,      // 抓取：area >= GRAB_AREA，停车抓球
};

static const char* status_name(RobotStatus s) {
    switch (s) {
        case STATUS_CHASE_TENNIS:    return "chase";
        case STATUS_GRAB_TENNIS:     return "grab";
    }
    return "unknown";
}

// 控制参数
static const int FRAME_WIDTH       = 640;

#if USE_ESP32_CAMERA
// ESP32-CAM: FOV wider, ball appears smaller at same distance
static const float GRAB_AREA       = 0.55f;  // Higher threshold: ESP32 FOV wider, need closer
static const int CENTER_MARGIN     = 55;     // Wider margin to reduce oscillation
static const float K_TURN_PULSE    = 3000.0f; // Less aggressive turning
static const int TURN_PULSE_MAX    = 300 * 1000;  // Shorter max pulse
static const float GRAB_AREA_MAX   = 0.70f;  // Back up threshold
#else
// VI camera: narrower FOV
static const float GRAB_AREA       = 0.40f;
static const int CENTER_MARGIN     = 35;
static const float K_TURN_PULSE    = 5000.0f;
static const int TURN_PULSE_MAX    = 500 * 1000;
static const float GRAB_AREA_MAX   = 0.55f;
#endif

static const int CHASE_SPEED       = 56;     // 追球前进速度
static const int TURN_SPEED        = 18;     // 原地转向速度（数值越小越慢）
static const int IDLE_SPEED        = 18;     // 没看到球时的搜索速度
static const int TURN_PULSE_MIN    = 75 * 1000;  // 最小脉冲 75ms（保证电机能动）
static const int GRAB_CONFIRM_THRESHOLD = 5;
static const int BACKWARD_SPEED = 18;      // 后退速度
static const int BACKWARD_PULSE_US = 200 * 1000; // 后退脉冲时间
static const int GRAB_LEFT_TURN_SPEED = 18;       // 抓取前左转补偿速度（数值越小越慢）
static const int GRAB_LEFT_TURN_US    = 150 * 1000; // 抓取前左转补偿时间
static const int GRAB_LEFT_TURN_COUNT = 4;          // 左转补偿次数

struct RobotState {
    RobotStatus status;
    float area_ratio;
    int ball_cx;    // 球中心 x
    int grab_confirm_count;

    RobotState() : status(STATUS_CHASE_TENNIS), area_ratio(0), ball_cx(0), grab_confirm_count(0) {}
};




int main(int argc, char** argv) {
    int ret = 0;
    CVI_MODEL_HANDLE model;



    if (argc < 2 || argc > 3) {
        usage(argv);
        exit(-1);
    }

#if USE_ESP32_CAMERA
    // Initialize ESP32-CAM
    ESP32Capture capture;
    g_capture = &capture;
    if (capture.init("/dev/cvi-camera") != 0) {
        LOGE("Failed to initialize ESP32-CAM");
        exit(-1);
    }
    capture.setResize(640, 640);
    LOGI("ESP32-CAM initialized (%ux%u)", capture.getWidth(), capture.getHeight());
#else
    CVI_U8 vi_channel = 0;
    if (argc == 3) {
        vi_channel = atoi(argv[2]);
    }

    // 清理上次异常退出残留的硬件资源
    {
        VICapture cleanup_vi;
        cleanup_vi.deinit();  // 无操作或释放残留资源
    }

    // Initialize VI system
    VICapture vi_capture;
    g_vi_capture = &vi_capture;
    if (vi_capture.init() != CVI_SUCCESS) {
        exit(-1);
    }

    // Wait for sensor to stabilize
    usleep(500 * 1000);

  #if USE_VPSS_RESIZE
    // Initialize VPSS for hardware resize (2560x1440 -> 640x640)
    LOGI("Initializing VPSS for hardware resize...");
    if (vi_capture.initVpssResize(2560, 1440, 640, 640) != CVI_SUCCESS) {
        LOGE("Failed to initialize VPSS");
        vi_capture.deinit();
        exit(-1);
    }
    LOGI("VPSS initialized successfully");
  #endif
#endif

    // 初始化电机、机械臂和状态机
    Motor motor;
    g_motor = &motor;
    Arm arm;
    RobotState robot;
    arm.grab_pos(); // 机械臂回到待抓取位置
    CVI_TENSOR* input;
    CVI_TENSOR* output;
    CVI_TENSOR* input_tensors;
    CVI_TENSOR* output_tensors;
    int32_t input_num;
    int32_t output_num;
    CVI_SHAPE input_shape;
    CVI_SHAPE* output_shape;
    int32_t height;
    int32_t width;
    // int bbox_len = 5; // 1 class + 4 bbox
    int classes_num = 1;
    float conf_thresh = 0.5;
    float iou_thresh = 0.5;
    ret = CVI_NN_RegisterModel(argv[1], &model);
    g_model = model;
    if (ret != CVI_RC_SUCCESS) {
        exit(1);
    }
    LOGI("CVI_NN_RegisterModel succeeded");

    // get input output tensors
    CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num, &output_tensors, &output_num);

    LOGI("=== Model information ===");
    LOGI("Input number: %d, Output number: %d", input_num, output_num);

    input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);
    assert(input);
    output = output_tensors;
    output_shape = reinterpret_cast<CVI_SHAPE*>(calloc(output_num, sizeof(CVI_SHAPE)));
    for (int i = 0; i < output_num; i++) {
        output_shape[i] = CVI_NN_TensorShape(&output[i]);
        LOGI("Output[%d] shape: [%d, %d, %d, %d]", i, output_shape[i].dim[0], output_shape[i].dim[1],
               output_shape[i].dim[2], output_shape[i].dim[3]);
    }

    // nchw
    input_shape = CVI_NN_TensorShape(input);
    height = input_shape.dim[2];
    width = input_shape.dim[3];
    assert(height % 32 == 0 && width % 32 == 0);

    // 循环处理摄像头帧
    int frame_idx = 0;
    struct timeval start_time, end_time;
    struct timeval t1, t2;
    long total_time_us = 0;
    int frame_count = 0;

    cv::setNumThreads(1);

    // 注册信号处理，确保 Ctrl-C 时清理硬件资源
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    while (true) {
        gettimeofday(&start_time, NULL);

        frame_idx++;
        LOGD("\n[Frame %d]", frame_idx);

        // Get frame from camera
        gettimeofday(&t1, NULL);

#if USE_ESP32_CAMERA
        cv::Mat bgr_image;
        if (capture.getFrameAsBGR(bgr_image) != 0) {
            LOGW("Failed to get frame from ESP32-CAM");
            usleep(100000);
            continue;
        }
#else
        cv::Mat nv21_image;
        if (vi_capture.getFrameAsNV21(vi_channel, nv21_image) != CVI_SUCCESS) {
            LOGW("Failed to get frame from VI channel %d", vi_channel);
            usleep(100000); // 休眠0.1秒
            continue;
        }
#endif

        gettimeofday(&t2, NULL);
        long read_time = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        LOGD("[PERF] Read frame: %.1fms", read_time / 1000.0);

        // Preprocess: fill NN input tensor with planar RGB data
        gettimeofday(&t1, NULL);

        int8_t* ptr = (int8_t*)CVI_NN_TensorPtr(input);
        int channel_size = height * width;

#if USE_ESP32_CAMERA
        // BGR → planar RGB (ESP32-CAM returns BGR via JPEG decode)
        cv::Mat rgb;
        cv::cvtColor(bgr_image, rgb, cv::COLOR_BGR2RGB);

        // Split channels and copy to NCHW tensor
        std::vector<cv::Mat> channels(3);
        cv::split(rgb, channels);
        memcpy(ptr + 0 * channel_size, channels[0].data, channel_size);
        memcpy(ptr + 1 * channel_size, channels[1].data, channel_size);
        memcpy(ptr + 2 * channel_size, channels[2].data, channel_size);
#else
        // Direct NV21 to planar RGB conversion (saves ~56ms total)
        const uint8_t* y_plane = nv21_image.ptr<uint8_t>(0);
        const uint8_t* vu_plane = nv21_image.ptr<uint8_t>(height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int y_idx = y * width + x;
                int uv_idx = (y / 2) * width + (x & ~1);

                int y_val = y_plane[y_idx];
                int u_val = vu_plane[uv_idx];     // V in NV21
                int v_val = vu_plane[uv_idx + 1]; // U in NV21

                int r = y_val + ((351 * (v_val - 128)) >> 8);
                int g = y_val - (((179 * (u_val - 128)) >> 8) + ((86 * (v_val - 128)) >> 8));
                int b = y_val + ((453 * (u_val - 128)) >> 8);

                r = (r < 0) ? 0 : (r > 255) ? 255 : r;
                g = (g < 0) ? 0 : (g > 255) ? 255 : g;
                b = (b < 0) ? 0 : (b > 255) ? 255 : b;

                int dst_idx = y * width + x;
                ptr[0 * channel_size + dst_idx] = r;
                ptr[1 * channel_size + dst_idx] = g;
                ptr[2 * channel_size + dst_idx] = b;
            }
        }
#endif

        gettimeofday(&t2, NULL);
        long preprocess_time = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        LOGD("[PERF] Preprocess: %.1fms", preprocess_time / 1000.0);

        // run inference
        gettimeofday(&t1, NULL);
        CVI_NN_Forward(model, input_tensors, input_num, output_tensors, output_num);
        gettimeofday(&t2, NULL);
        long inference_time = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        LOGD("[PERF] Inference: %.1fms", inference_time / 1000.0);

        // do post proprocess
        gettimeofday(&t1, NULL);
        int det_num = 0;
        std::vector<detection> dets;

        det_num = getDetections(output, height, width, classes_num, output_shape[0], conf_thresh, dets);
        // correct box with origin image size
        NMS(dets, &det_num, iou_thresh);
        correctYoloBoxes(dets, det_num, FRAME_WIDTH, FRAME_WIDTH, height, width);
        gettimeofday(&t2, NULL);
        long postprocess_time = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        LOGD("[PERF] Postprocess: %.1fms", postprocess_time / 1000.0);

        // ============ 状态机控制逻辑 ============
        if (det_num > 0) {
            // 选择最大的球（最近的）
            int best_idx = 0;
            for (int i = 1; i < det_num; i++) {
                if (dets[i].bbox.w * dets[i].bbox.h > dets[best_idx].bbox.w * dets[best_idx].bbox.h)
                    best_idx = i;
            }

            box b = dets[best_idx].bbox;
            float img_area = FRAME_WIDTH * FRAME_WIDTH; // Image is 640x640
            float area_ratio = (b.w * b.h) / img_area;
            int ball_cx = static_cast<int>(b.x);
            int center = FRAME_WIDTH / 2;

            robot.area_ratio = area_ratio;
            robot.ball_cx = ball_cx;

            LOGD("[DETECT] area=%.3f cx=%d conf=%.3f status=%s",
                 area_ratio, ball_cx, dets[best_idx].score, status_name(robot.status));

            int offset = ball_cx - center;
            bool centered = abs(offset) <= CENTER_MARGIN;

            // 动态脉冲时间：area 越大脉冲越长，避免远距离转过头
            int pulse_us = std::max(TURN_PULSE_MIN, std::min(TURN_PULSE_MAX, (int)(K_TURN_PULSE * area_ratio * 1000)));

            if (area_ratio >= GRAB_AREA && centered) {
                // 球足够近且居中 → 确认计数
                robot.grab_confirm_count++;
                LOGD("[GRAB] confirm %d/%d (area=%.3f cx=%d)", robot.grab_confirm_count, GRAB_CONFIRM_THRESHOLD, area_ratio, ball_cx);

                if (area_ratio >= GRAB_AREA_MAX) {
                    // 太近了，微微后退
                    LOGD("[GRAB] Too close (area=%.3f > %.3f), backing up", area_ratio, GRAB_AREA_MAX);
                    motor.backward(BACKWARD_SPEED);
                    usleep(BACKWARD_PULSE_US);
                    motor.standby();
                } else {
                    motor.standby();
                }

                if (robot.grab_confirm_count >= GRAB_CONFIRM_THRESHOLD) {
                    // 太近后退一步再抓
                    if (area_ratio >= GRAB_AREA_MAX) {
                        LOGD("[GRAB] Backing up before grab");
                        motor.backward(BACKWARD_SPEED);
                        usleep(BACKWARD_PULSE_US);
                        motor.standby();
                    }

                    LOGD("[ARM] Confirmed, compensating gripper offset...");

                    // 爪子偏右，连续多次左转补偿
                    for (int i = 0; i < GRAB_LEFT_TURN_COUNT; i++) {
                        LOGD("[GRAB] Left turn compensation %d/%d (%dms)", i + 1, GRAB_LEFT_TURN_COUNT, GRAB_LEFT_TURN_US / 1000);
                        motor.drive(-GRAB_LEFT_TURN_SPEED, GRAB_LEFT_TURN_SPEED);
                        usleep(GRAB_LEFT_TURN_US);
                        motor.standby();
                        usleep(100 * 1000);  // 间隔100ms
                    }

                    LOGD("[ARM] Full grab sequence");

                    // 抓取: 伸下→夹紧→抬起→停2秒
                    arm.grab();
                    usleep(2000 * 1000);

                    // 松开→复位
                    arm.release();
                    usleep(1000 * 1000);
                    arm.grab_pos();
                    usleep(1000 * 1000);
                    robot.grab_confirm_count = 0;
                    robot.status = STATUS_CHASE_TENNIS;
                }
            } else if (area_ratio >= GRAB_AREA && !centered) {
                // 球够近但没居中 → 原地微调（脉冲式），不前进
                robot.grab_confirm_count = 0;
                LOGD("[ALIGN] area=%.3f OK but cx=%d not centered, pulse=%dms", area_ratio, ball_cx, pulse_us / 1000);
                if (offset < 0) {
                    motor.drive(-TURN_SPEED, TURN_SPEED);
                } else {
                    motor.drive(TURN_SPEED, -TURN_SPEED);
                }
                usleep(pulse_us);
                motor.standby();
            } else {
                // 追球
                robot.grab_confirm_count = 0;
                robot.status = STATUS_CHASE_TENNIS;

                if (!centered) {
                    // 球偏左或偏右 → 差速原地转向（脉冲式）
                    if (offset < 0) {
                        LOGD("[MOTOR] TURN LEFT (cx=%d offset=%d pulse=%dms)", ball_cx, offset, pulse_us / 1000);
                        motor.drive(-TURN_SPEED, TURN_SPEED);
                    } else {
                        LOGD("[MOTOR] TURN RIGHT (cx=%d offset=%d pulse=%dms)", ball_cx, offset, pulse_us / 1000);
                        motor.drive(TURN_SPEED, -TURN_SPEED);
                    }
                    usleep(pulse_us);
                    motor.standby();
                } else {
                    // 球基本居中 → 前进
                    LOGD("[MOTOR] FORWARD (cx=%d area=%.3f < %.3f)", ball_cx, area_ratio, GRAB_AREA);
                    motor.forward(CHASE_SPEED);
                }
            }

#if ENABLE_DRAW_BBOX
            for (int i = 0; i < det_num; i++) {
                box b = dets[i].bbox;
                int x1 = std::max(0, (int)(b.x - b.w / 2));
                int y1 = std::max(0, (int)(b.y - b.h / 2));
                int x2 = std::min(cloned.cols - 1, (int)(b.x + b.w / 2));
                int y2 = std::min(cloned.rows - 1, (int)(b.y + b.h / 2));
                cv::rectangle(cloned, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 255), 3, 8, 0);
                float ar = (b.w * b.h) / img_area;
                char content[100];
                sprintf(content, "tennis %.3f area:%.3f", dets[i].score, ar);
                cv::putText(cloned, content, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            }

            // Center line (red)
            cv::line(cloned, cv::Point(center, 0), cv::Point(center, cloned.rows), cv::Scalar(0, 0, 255), 1);
            // Center margin lines (green)
            cv::line(cloned, cv::Point(center - CENTER_MARGIN, 0), cv::Point(center - CENTER_MARGIN, cloned.rows), cv::Scalar(0, 255, 0), 1);
            cv::line(cloned, cv::Point(center + CENTER_MARGIN, 0), cv::Point(center + CENTER_MARGIN, cloned.rows), cv::Scalar(0, 255, 0), 1);

            // Status info
            char info[128];
            sprintf(info, "status=%s area=%.3f cx=%d grab_area=%.2f",
                    status_name(robot.status), area_ratio, ball_cx, GRAB_AREA);
            cv::putText(cloned, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
#endif

        } else {
            LOGD("[DETECT] No ball detected, searching...");
            robot.grab_confirm_count = 0;
            robot.status = STATUS_CHASE_TENNIS;
            motor.drive(IDLE_SPEED, -IDLE_SPEED);  // 没看到球就连续右转搜索
        }
        
#if ENABLE_SAVE_IMAGE
            {
                const char* save_dir = "/root/images";
                mkdir(save_dir, 0755);  // ensure directory exists
                char output_path[256];
                sprintf(output_path, "%s/detected_%d.jpg", save_dir, frame_idx);
                bool ok = cv::imwrite(output_path, cloned);
                LOGD("[SAVE] %s %s", output_path, ok ? "OK" : "FAILED");
            }
#endif
        // 计算帧率
        gettimeofday(&end_time, NULL);
        long frame_time_us = (end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_usec - start_time.tv_usec);
        float fps = 1000000.0f / frame_time_us;

        frame_count++;
        total_time_us += frame_time_us;
        float avg_fps = 1000000.0f * frame_count / total_time_us;

        printf("[FPS] %.2f  avg: %.2f  (%.1fms)\n", fps, avg_fps, frame_time_us / 1000.0f);

        // 持续运行，无帧数限制
        // if (frame_idx >= 200) break;
    } // end while loop

    // Cleanup
    cleanup_and_exit();
    free(output_shape);
    return 0;
}