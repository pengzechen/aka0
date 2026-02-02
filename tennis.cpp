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
#include <inttypes.h>
#include <opencv2/opencv.hpp>
#include "cviruntime.h"
#include "motor.hpp"

// VI related headers - must include base types first
#include <linux/cvi_type.h>
#include <linux/cvi_common.h>
#include <linux/cvi_comm_video.h>
#include "cvi_buffer.h"
#include "cvi_ae_comm.h"
#include "cvi_awb_comm.h"
#include "cvi_comm_isp.h"
#include "cvi_comm_sns.h"
#include "cvi_ae.h"
#include "cvi_awb.h"
#include "cvi_isp.h"
#include "cvi_sns_ctrl.h"
#include "cvi_vpss.h"
#include "sample_comm.h"

// 控制宏定义
#define ENABLE_DEBUG_OUTPUT 0 // 是否启用详细调试输出
#define ENABLE_DRAW_BBOX 1    // 是否画框并保存图片
#define ENABLE_SAVE_IMAGE 0   // 是否保存检测结果图片
#define USE_VPSS_RESIZE 1     // 使用VPSS硬件加速resize

typedef struct {
    float x, y, w, h;
} box;

typedef struct {
    box bbox;
    int cls;
    float score;
    int batch_idx;
} detection;

static const char* tennis_names[] = {"tennis"}; // 单类别网球检测

// VI global variables
static SAMPLE_VI_CONFIG_S g_stViConfig;
static SAMPLE_INI_CFG_S g_stIniCfg;

#if USE_VPSS_RESIZE
// VPSS global variables for hardware resize
static VPSS_GRP g_VpssGrp = 0;
static VPSS_CHN g_VpssChn = 0;
static bool g_bVpssInited = false;
#endif

static void usage(char** argv) {
    printf("Usage:\n");
    printf("   %s cvimodel [vi_channel]\n", argv[0]);
    printf("   Example: %s model.cvimodel 0\n", argv[0]);
    printf("   This will capture video from VI channel (default: 0)\n");
}

template <typename T> int argmax(const T* data, size_t len, size_t stride = 1) {
    int maxIndex = 0;
    for (size_t i = stride; i < len; i += stride) {
        if (data[maxIndex] < data[i]) {
            maxIndex = i;
        }
    }
    return maxIndex;
}

float calIou(box a, box b) {
    float area1 = a.w * a.h;
    float area2 = b.w * b.h;
    float wi = std::min((a.x + a.w / 2), (b.x + b.w / 2)) - std::max((a.x - a.w / 2), (b.x - b.w / 2));
    float hi = std::min((a.y + a.h / 2), (b.y + b.h / 2)) - std::max((a.y - a.h / 2), (b.y - b.h / 2));
    float area_i = std::max(wi, 0.0f) * std::max(hi, 0.0f);
    return area_i / (area1 + area2 - area_i);
}

static void NMS(std::vector<detection>& dets, int* total, float thresh) {
    if (*total) {
        std::sort(dets.begin(), dets.end(), [](detection& a, detection& b) { return b.score < a.score; });
        int new_count = *total;
        for (int i = 0; i < *total; ++i) {
            detection& a = dets[i];
            if (a.score == 0)
                continue;
            for (int j = i + 1; j < *total; ++j) {
                detection& b = dets[j];
                if (dets[i].batch_idx == dets[j].batch_idx && b.score != 0 && dets[i].cls == dets[j].cls &&
                    calIou(a.bbox, b.bbox) > thresh) {
                    b.score = 0;
                    new_count--;
                }
            }
        }
        std::vector<detection>::iterator it = dets.begin();
        while (it != dets.end()) {
            if (it->score == 0) {
                dets.erase(it);
            } else {
                it++;
            }
        }
        *total = new_count;
    }
}

void correctYoloBoxes(std::vector<detection>& dets, int det_num, int image_h, int image_w, int input_height,
                      int input_width) {
    // 计算缩放比例和padding，与Python代码保持一致
    float scale = std::min((float)input_width / image_w, (float)input_height / image_h);
    int new_h = (int)(image_h * scale);
    int new_w = (int)(image_w * scale);
    int pad_top = (input_height - new_h) / 2;
    int pad_left = (input_width - new_w) / 2;

#if ENABLE_DEBUG_OUTPUT
    printf("=== Coordinate correction ===\n");
    printf("Original image: %dx%d, Input size: %dx%d\n", image_w, image_h, input_width, input_height);
    printf("Scale: %.3f, New size: %dx%d, Padding: left=%d, top=%d\n", scale, new_w, new_h, pad_left, pad_top);
#endif

    for (int i = 0; i < det_num; ++i) {
        // YOLOv8输出的是中心点坐标(cx,cy)和宽高(w,h)
        float cx = dets[i].bbox.x;
        float cy = dets[i].bbox.y;
        float w = dets[i].bbox.w;
        float h = dets[i].bbox.h;

        // 转换为左上角和右下角坐标(相对于640x640输入图像)
        float x1 = cx - 0.5f * w;
        float y1 = cy - 0.5f * h;
        float x2 = cx + 0.5f * w;
        float y2 = cy + 0.5f * h;

        // 去除padding并缩放回原图尺寸
        x1 = std::max(0.0f, (x1 - pad_left) / scale);
        y1 = std::max(0.0f, (y1 - pad_top) / scale);
        x2 = std::min((float)image_w, (x2 - pad_left) / scale);
        y2 = std::min((float)image_h, (y2 - pad_top) / scale);

        // 转换回中心点坐标和宽高格式
        dets[i].bbox.x = (x1 + x2) / 2.0f; // 中心点x
        dets[i].bbox.y = (y1 + y2) / 2.0f; // 中心点y
        dets[i].bbox.w = x2 - x1;          // 宽度
        dets[i].bbox.h = y2 - y1;          // 高度

#if ENABLE_DEBUG_OUTPUT
        printf("Det[%d]: input_bbox(%.1f,%.1f,%.1f,%.1f) -> "
               "output_bbox(%.1f,%.1f,%.1f,%.1f)\n",
               i, cx, cy, w, h, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
#endif
    }
}

/**
 * @brief
 * @param output
 * @note scores_shape : [batch , class_num, det_num, 1]
 * @note des_shape: [batch, 1, 4, det_num]
 * @return int
 */
int getDetections(CVI_TENSOR* output, int32_t input_height, int32_t input_width, int classes_num,
                  CVI_SHAPE output_shape, float conf_thresh, std::vector<detection>& dets) {
#if ENABLE_DEBUG_OUTPUT
    // 添加调试信息：打印输出tensor信息
    printf("=== DEBUG: Output tensor information ===\n");
    printf("Output shape: [%d, %d, %d, %d]\n", output_shape.dim[0], output_shape.dim[1], output_shape.dim[2],
           output_shape.dim[3]);
#endif

    // 检查是否有足够的输出tensor
    if (output == nullptr) {
        printf("ERROR: output tensor is null\n");
        return 0;
    }

    float* output_ptr = (float*)CVI_NN_TensorPtr(&output[0]);

    // 检查指针是否有效
    if (output_ptr == nullptr) {
        printf("ERROR: tensor pointer is null\n");
        return 0;
    }

    float stride[3] = {8, 16, 32};
    int count = 0;
    int batch = output_shape.dim[0];
    int channels = output_shape.dim[1];      // 应该是4(bbox) + 1(objectness) + classes_num
    int total_anchors = output_shape.dim[2]; // 27600

#if ENABLE_DEBUG_OUTPUT
    printf("Batch: %d, Channels: %d, Total_anchors: %d\n", batch, channels, total_anchors);
#endif

    // 计算每个stride层的anchor数量
    int anchor_counts[3];
    for (int i = 0; i < 3; i++) {
        int nh = input_height / stride[i];
        int nw = input_width / stride[i];
        anchor_counts[i] = nh * nw;
#if ENABLE_DEBUG_OUTPUT
        printf("Stride[%d]: %f, grid: %dx%d, anchors: %d\n", i, stride[i], nh, nw, anchor_counts[i]);
#endif
    }

    int anchor_offset = 0;
    for (int b = 0; b < batch; b++) {
        anchor_offset = 0;
        for (int stride_idx = 0; stride_idx < 3; stride_idx++) {
            int nh = input_height / stride[stride_idx];
            int nw = input_width / stride[stride_idx];
            int current_anchors = anchor_counts[stride_idx];

            for (int anchor_idx = 0; anchor_idx < current_anchors; anchor_idx++) {
                int total_anchor_idx = anchor_offset + anchor_idx;

                // 获取objectness/confidence (第5个通道)
                float objectness = output_ptr[4 * total_anchors + total_anchor_idx];

#if ENABLE_DEBUG_OUTPUT
                // 添加调试输出：打印前几个anchor的原始数据
                if (total_anchor_idx < 10 || objectness > 0.1) {
                    printf("Anchor[%d]: raw values = [%.6f, %.6f, %.6f, %.6f, %.6f]\n", total_anchor_idx,
                           output_ptr[0 * total_anchors + total_anchor_idx],
                           output_ptr[1 * total_anchors + total_anchor_idx],
                           output_ptr[2 * total_anchors + total_anchor_idx],
                           output_ptr[3 * total_anchors + total_anchor_idx],
                           output_ptr[4 * total_anchors + total_anchor_idx]);
                }
#endif

                if (objectness <= conf_thresh) {
                    continue;
                }

                // 获取bbox坐标 (前4个通道)
                // YOLOv8输出格式：cx, cy, w, h (相对于640x640输入图像的绝对像素坐标)
                float cx = output_ptr[0 * total_anchors + total_anchor_idx];
                float cy = output_ptr[1 * total_anchors + total_anchor_idx];
                float w = output_ptr[2 * total_anchors + total_anchor_idx];
                float h = output_ptr[3 * total_anchors + total_anchor_idx];

                detection det;
                det.score = objectness;
                det.cls = 0; // 单类别网球检测
                det.batch_idx = b;

                // 直接使用模型输出的中心点坐标和宽高，无需grid计算
                det.bbox.x = cx; // 中心点x坐标
                det.bbox.y = cy; // 中心点y坐标
                det.bbox.w = w;  // 宽度
                det.bbox.h = h;  // 高度

#if ENABLE_DEBUG_OUTPUT
                printf("Detection[%d]: conf=%.3f, bbox_center=(%.1f,%.1f), "
                       "size=(%.1f,%.1f)\n",
                       count, objectness, cx, cy, w, h);
#endif

                count++;
                dets.emplace_back(det);
            }

            anchor_offset += current_anchors;
        }
    }
    return count;
}

// 根据球的位置控制小车移动
void controlMotor(Motor& motor, float ball_x, float ball_y, int image_width, int image_height) {
    // 计算球相对于图像中心的位置
    float center_x = image_width / 2.0f;
    float center_y = image_height / 2.0f;
    float offset_x = ball_x - center_x;
    float offset_y = ball_y - center_y;

    // 定义阈值
    float x_threshold = image_width * 0.08f;  // 左右偏移阈值（8%图像宽度）
    float y_threshold = image_height * 0.08f; // 前后偏移阈值（8%图像高度）

    int speed = 50; // 基础速度50%

    // 决策逻辑：优先处理左右偏移，然后处理前后距离
    if (fabs(offset_x) > x_threshold) {
        // 球偏左或偏右
        if (offset_x < 0) {
            printf("[MOTOR] LEFT (ball_x=%.1f, offset=%.1f)\n", ball_x, offset_x);
            motor.left(speed);
        } else {
            printf("[MOTOR] RIGHT (ball_x=%.1f, offset=%.1f)\n", ball_x, offset_x);
            motor.right(speed);
        }
    } else if (offset_y > y_threshold) {
        // 球在图像下方，表示距离近，后退
        printf("[MOTOR] BACKWARD (ball_y=%.1f, offset=%.1f)\n", ball_y, offset_y);
        motor.backward(speed);
    } else if (offset_y < -y_threshold) {
        // 球在图像上方，表示距离远，前进
        printf("[MOTOR] FORWARD (ball_y=%.1f, offset=%.1f)\n", ball_y, offset_y);
        motor.forward(speed);
    } else {
        // 球在中心位置，停止
        printf("[MOTOR] STANDBY (centered)\n");
        motor.standby();
    }
}

// VI initialization function
static int sys_vi_init(void) {
    MMF_VERSION_S stVersion;
    SAMPLE_INI_CFG_S stIniCfg;
    SAMPLE_VI_CONFIG_S stViConfig;

    PIC_SIZE_E enPicSize;
    SIZE_S stSize;
    CVI_S32 s32Ret = CVI_SUCCESS;
    LOG_LEVEL_CONF_S log_conf;

    CVI_SYS_GetVersion(&stVersion);
    SAMPLE_PRT("MMF Version:%s\n", stVersion.version);

    log_conf.enModId = CVI_ID_LOG;
    log_conf.s32Level = CVI_DBG_INFO;
    CVI_LOG_SetLevelConf(&log_conf);

    // Get config from ini if found.
    if (SAMPLE_COMM_VI_ParseIni(&stIniCfg)) {
        SAMPLE_PRT("Parse complete\n");
    }

    // Set sensor number
    CVI_VI_SetDevNum(stIniCfg.devNum);
    /************************************************
   * step1:  Config VI
   ************************************************/
    s32Ret = SAMPLE_COMM_VI_IniToViCfg(&stIniCfg, &stViConfig);
    if (s32Ret != CVI_SUCCESS)
        return s32Ret;

    memcpy(&g_stViConfig, &stViConfig, sizeof(SAMPLE_VI_CONFIG_S));
    memcpy(&g_stIniCfg, &stIniCfg, sizeof(SAMPLE_INI_CFG_S));

    /************************************************
   * step2:  Get input size
   ************************************************/
    s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(stIniCfg.enSnsType[0], &enPicSize);
    if (s32Ret != CVI_SUCCESS) {
        CVI_TRACE_LOG(CVI_DBG_ERR, "SAMPLE_COMM_VI_GetSizeBySensor failed with %#x\n", s32Ret);
        return s32Ret;
    }

    s32Ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSize);
    if (s32Ret != CVI_SUCCESS) {
        CVI_TRACE_LOG(CVI_DBG_ERR, "SAMPLE_COMM_SYS_GetPicSize failed with %#x\n", s32Ret);
        return s32Ret;
    }

    /************************************************
   * step3:  Init modules
   ************************************************/
    s32Ret = SAMPLE_PLAT_SYS_INIT(stSize);
    if (s32Ret != CVI_SUCCESS) {
        CVI_TRACE_LOG(CVI_DBG_ERR, "sys init failed. s32Ret: 0x%x !\n", s32Ret);
        return s32Ret;
    }

    s32Ret = SAMPLE_PLAT_VI_INIT(&stViConfig);
    if (s32Ret != CVI_SUCCESS) {
        CVI_TRACE_LOG(CVI_DBG_ERR, "vi init failed. s32Ret: 0x%x !\n", s32Ret);
        return s32Ret;
    }

    return CVI_SUCCESS;
}

static void sys_vi_deinit(void) {
    SAMPLE_COMM_VI_DestroyIsp(&g_stViConfig);
    SAMPLE_COMM_VI_DestroyVi(&g_stViConfig);
    SAMPLE_COMM_SYS_Exit();
}

#if USE_VPSS_RESIZE
// Initialize VPSS for hardware resize
static int vpss_resize_init(int input_w, int input_h, int output_w, int output_h) {
    CVI_S32 s32Ret;
    VPSS_GRP_ATTR_S stVpssGrpAttr;
    VPSS_CHN_ATTR_S stVpssChnAttr;
    CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
    
    // Set group attribute
    memset(&stVpssGrpAttr, 0, sizeof(VPSS_GRP_ATTR_S));
    stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
    stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
    stVpssGrpAttr.enPixelFormat = PIXEL_FORMAT_NV21;
    stVpssGrpAttr.u32MaxW = input_w;
    stVpssGrpAttr.u32MaxH = input_h;
    stVpssGrpAttr.u8VpssDev = 0;
    
    // Create and start VPSS group
    s32Ret = CVI_VPSS_CreateGrp(g_VpssGrp, &stVpssGrpAttr);
    if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_CreateGrp failed: 0x%x\n", s32Ret);
        return s32Ret;
    }
    
    // Set channel attribute for resize output
    memset(&stVpssChnAttr, 0, sizeof(VPSS_CHN_ATTR_S));
    stVpssChnAttr.u32Width = output_w;
    stVpssChnAttr.u32Height = output_h;
    stVpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
    stVpssChnAttr.enPixelFormat = PIXEL_FORMAT_NV21;
    stVpssChnAttr.stFrameRate.s32SrcFrameRate = -1;
    stVpssChnAttr.stFrameRate.s32DstFrameRate = -1;
    stVpssChnAttr.u32Depth = 1;
    stVpssChnAttr.bMirror = CVI_FALSE;
    stVpssChnAttr.bFlip = CVI_FALSE;
    stVpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
    stVpssChnAttr.stNormalize.bEnable = CVI_FALSE;
    
    s32Ret = CVI_VPSS_SetChnAttr(g_VpssGrp, g_VpssChn, &stVpssChnAttr);
    if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_SetChnAttr failed: 0x%x\n", s32Ret);
        CVI_VPSS_DestroyGrp(g_VpssGrp);
        return s32Ret;
    }
    
    abChnEnable[g_VpssChn] = CVI_TRUE;
    s32Ret = CVI_VPSS_EnableChn(g_VpssGrp, g_VpssChn);
    if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_EnableChn failed: 0x%x\n", s32Ret);
        CVI_VPSS_DestroyGrp(g_VpssGrp);
        return s32Ret;
    }
    
    s32Ret = CVI_VPSS_StartGrp(g_VpssGrp);
    if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_StartGrp failed: 0x%x\n", s32Ret);
        CVI_VPSS_DisableChn(g_VpssGrp, g_VpssChn);
        CVI_VPSS_DestroyGrp(g_VpssGrp);
        return s32Ret;
    }
    
    g_bVpssInited = true;
    printf("VPSS initialized: %dx%d -> %dx%d\n", input_w, input_h, output_w, output_h);
    return CVI_SUCCESS;
}

static void vpss_resize_deinit(void) {
    if (g_bVpssInited) {
        CVI_VPSS_StopGrp(g_VpssGrp);
        CVI_VPSS_DisableChn(g_VpssGrp, g_VpssChn);
        CVI_VPSS_DestroyGrp(g_VpssGrp);
        g_bVpssInited = false;
    }
}
#endif

// Get YUV frame from VI and convert to BGR
static int vi_get_frame_as_bgr(CVI_U8 chn, cv::Mat& bgr_image) {
    VIDEO_FRAME_INFO_S stVideoFrame;
    struct timeval t1, t2;
    long get_frame_us, mmap_us, resize_us, cvt_us, flip_us, munmap_us, release_us;

    gettimeofday(&t1, NULL);
    if (CVI_VI_GetChnFrame(0, chn, &stVideoFrame, 3000) == 0) {
        gettimeofday(&t2, NULL);
        get_frame_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        
#if USE_VPSS_RESIZE
        // Use VPSS hardware to resize
        gettimeofday(&t1, NULL);
        
        // Send frame to VPSS for hardware resize
        CVI_S32 s32Ret = CVI_VPSS_SendFrame(g_VpssGrp, &stVideoFrame, -1);
        if (s32Ret != CVI_SUCCESS) {
            printf("CVI_VPSS_SendFrame failed: 0x%x\n", s32Ret);
            CVI_VI_ReleaseChnFrame(0, chn, &stVideoFrame);
            return CVI_FAILURE;
        }
        
        // Get resized frame from VPSS
        VIDEO_FRAME_INFO_S stResizedFrame;
        s32Ret = CVI_VPSS_GetChnFrame(g_VpssGrp, g_VpssChn, &stResizedFrame, 1000);
        if (s32Ret != CVI_SUCCESS) {
            printf("CVI_VPSS_GetChnFrame failed: 0x%x\n", s32Ret);
            CVI_VI_ReleaseChnFrame(0, chn, &stVideoFrame);
            return CVI_FAILURE;
        }
        
        gettimeofday(&t2, NULL);
        resize_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        mmap_us = 0; // No separate mmap needed with VPSS
        
        // Get resized frame info
        gettimeofday(&t1, NULL);
        int width = stResizedFrame.stVFrame.u32Width;
        int height = stResizedFrame.stVFrame.u32Height;
        int stride_y = stResizedFrame.stVFrame.u32Stride[0];
        int stride_uv = stResizedFrame.stVFrame.u32Stride[1];
        
        // Map resized frame memory - use full stride size
        size_t y_size = stride_y * height;
        size_t uv_size = stride_uv * height / 2;
        size_t image_size = y_size + uv_size;
        
        CVI_VOID* vir_addr = CVI_SYS_Mmap(stResizedFrame.stVFrame.u64PhyAddr[0], image_size);
        CVI_SYS_IonInvalidateCache(stResizedFrame.stVFrame.u64PhyAddr[0], vir_addr, image_size);
        
        // Always copy to contiguous memory for fast cvtColor
        // Even if stride==width, VPSS output may not be cache-friendly
        cv::Mat yuv_continuous(height * 3 / 2, width, CV_8UC1);
        
        if (stride_y == width && stride_uv == width) {
            // Fast memcpy for entire buffer
            memcpy(yuv_continuous.data, vir_addr, width * height * 3 / 2);
        } else {
            // Copy Y plane line by line
            for (int i = 0; i < height; i++) {
                memcpy(yuv_continuous.data + i * width, 
                       (uint8_t*)vir_addr + i * stride_y, width);
            }
            // Copy UV plane line by line
            for (int i = 0; i < height / 2; i++) {
                memcpy(yuv_continuous.data + height * width + i * width,
                       (uint8_t*)vir_addr + y_size + i * stride_uv, width);
            }
        }
        
        cv::cvtColor(yuv_continuous, bgr_image, cv::COLOR_YUV2BGR_NV21);
        
        gettimeofday(&t2, NULL);
        cvt_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        
        printf("[VI-DETAIL] GetFrame: %.1fms, VPSS_Resize: %.1fms, Cvt: %.1fms (stride=%d)\n",
               get_frame_us/1000.0, resize_us/1000.0, cvt_us/1000.0, stride_y);
        
        // Flip image 180 degrees
        gettimeofday(&t1, NULL);
        cv::flip(bgr_image, bgr_image, -1);
        gettimeofday(&t2, NULL);
        flip_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        
        CVI_SYS_Munmap(vir_addr, image_size);
        CVI_VPSS_ReleaseChnFrame(g_VpssGrp, g_VpssChn, &stResizedFrame);
        CVI_VI_ReleaseChnFrame(0, chn, &stVideoFrame);
        
        printf("[VI-DETAIL] Flip: %.1fms (VPSS mode)\n", flip_us/1000.0);
        
        return CVI_SUCCESS;
#else
        // Software resize path (original code)
        size_t image_size = stVideoFrame.stVFrame.u32Length[0] + stVideoFrame.stVFrame.u32Length[1] +
                            stVideoFrame.stVFrame.u32Length[2];
        CVI_VOID* vir_addr;
        CVI_U32 plane_offset;

        int width = stVideoFrame.stVFrame.u32Width;
        int height = stVideoFrame.stVFrame.u32Height;
        
        // Map physical memory to virtual address
        gettimeofday(&t1, NULL);
        vir_addr = CVI_SYS_Mmap(stVideoFrame.stVFrame.u64PhyAddr[0], image_size);
        CVI_SYS_IonInvalidateCache(stVideoFrame.stVFrame.u64PhyAddr[0], vir_addr, image_size);
        gettimeofday(&t2, NULL);
        mmap_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);

        // Map virtual addresses for each plane
        plane_offset = 0;
        for (int i = 0; i < 3; i++) {
            if (stVideoFrame.stVFrame.u32Length[i] != 0) {
                stVideoFrame.stVFrame.pu8VirAddr[i] = (CVI_U8*)vir_addr + plane_offset;
                plane_offset += stVideoFrame.stVFrame.u32Length[i];
            }
        }

        // Check pixel format and convert accordingly
        // Format 19 = PIXEL_FORMAT_NV21 (YVU420 semi-planar)
        // Format 18 = PIXEL_FORMAT_NV12 (YUV420 semi-planar)
        // Optimization: Use INTER_NEAREST for faster resize (quality loss is acceptable for detection)
        cv::Mat yuv_small;
        gettimeofday(&t1, NULL);
        if (stVideoFrame.stVFrame.enPixelFormat == 19) { // NV21
            // NV21: Y plane + VU interleaved plane
            cv::Mat yuv_nv21(height * 3 / 2, width, CV_8UC1, vir_addr);
            // Use INTER_NEAREST - 3x faster than INTER_LINEAR for resize
            cv::resize(yuv_nv21, yuv_small, cv::Size(640, 640 * 3 / 2), 0, 0, cv::INTER_NEAREST);
        } else if (stVideoFrame.stVFrame.enPixelFormat == 18) { // NV12
            cv::Mat yuv_nv12(height * 3 / 2, width, CV_8UC1, vir_addr);
            cv::resize(yuv_nv12, yuv_small, cv::Size(640, 640 * 3 / 2), 0, 0, cv::INTER_NEAREST);
        } else { // Default to I420
            cv::Mat yuv_i420(height * 3 / 2, width, CV_8UC1, vir_addr);
            cv::resize(yuv_i420, yuv_small, cv::Size(640, 640 * 3 / 2), 0, 0, cv::INTER_NEAREST);
        }
        gettimeofday(&t2, NULL);
        resize_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        
        gettimeofday(&t1, NULL);
        if (stVideoFrame.stVFrame.enPixelFormat == 19) { // NV21
            cv::cvtColor(yuv_small, bgr_image, cv::COLOR_YUV2BGR_NV21);
        } else if (stVideoFrame.stVFrame.enPixelFormat == 18) { // NV12
            cv::cvtColor(yuv_small, bgr_image, cv::COLOR_YUV2BGR_NV12);
        } else { // Default to I420
            cv::cvtColor(yuv_small, bgr_image, cv::COLOR_YUV2BGR_I420);
        }
        gettimeofday(&t2, NULL);
        cvt_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        
        printf("[VI-DETAIL] GetFrame: %.1fms, Mmap: %.1fms, Resize: %.1fms, Cvt: %.1fms\n",
               get_frame_us/1000.0, mmap_us/1000.0, resize_us/1000.0, cvt_us/1000.0);
        
        // Flip image 180 degrees (camera is upside down)
        gettimeofday(&t1, NULL);
        cv::flip(bgr_image, bgr_image, -1);
        gettimeofday(&t2, NULL);
        flip_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);

        gettimeofday(&t1, NULL);
        CVI_SYS_Munmap(vir_addr, image_size);
        gettimeofday(&t2, NULL);
        munmap_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);

        gettimeofday(&t1, NULL);
        if (CVI_VI_ReleaseChnFrame(0, chn, &stVideoFrame) != 0) {
            CVI_TRACE_LOG(CVI_DBG_ERR, "CVI_VI_ReleaseChnFrame NG\n");
            return CVI_FAILURE;
        }
        gettimeofday(&t2, NULL);
        release_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        
        printf("[VI-DETAIL] Flip: %.1fms, Munmap: %.1fms, Release: %.1fms\n",
               flip_us/1000.0, munmap_us/1000.0, release_us/1000.0);

        return CVI_SUCCESS;
#endif  // USE_VPSS_RESIZE
    }

    CVI_TRACE_LOG(CVI_DBG_ERR, "CVI_VI_GetChnFrame NG\n");
    return CVI_FAILURE;
}

int main(int argc, char** argv) {
    int ret = 0;
    CVI_MODEL_HANDLE model;

    if (argc < 2 || argc > 3) {
        usage(argv);
        exit(-1);
    }

    CVI_U8 vi_channel = 0; // Default VI channel
    if (argc == 3) {
        vi_channel = atoi(argv[2]);
    }

    printf("Using VI channel: %d\n", vi_channel);

    // Initialize VI system
    printf("Initializing VI system...\n");
    if (sys_vi_init() != CVI_SUCCESS) {
        printf("Failed to initialize VI system\n");
        exit(-1);
    }
    printf("VI system initialized successfully\n");

    // Wait for sensor to stabilize
    usleep(500 * 1000);

#if USE_VPSS_RESIZE
    // Initialize VPSS for hardware resize (2560x1440 -> 640x640)
    printf("Initializing VPSS for hardware resize...\n");
    if (vpss_resize_init(2560, 1440, 640, 640) != CVI_SUCCESS) {
        printf("Failed to initialize VPSS\n");
        sys_vi_deinit();
        exit(-1);
    }
    printf("VPSS initialized successfully\n");
#endif

    // 初始化电机
    Motor motor;
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
    if (ret != CVI_RC_SUCCESS) {
        printf("CVI_NN_RegisterModel failed, err %d\n", ret);
        exit(1);
    }
    printf("CVI_NN_RegisterModel succeeded\n");

    // get input output tensors
    CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num, &output_tensors, &output_num);

    printf("=== DEBUG: Model information ===\n");
    printf("Input number: %d, Output number: %d\n", input_num, output_num);

    input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);
    assert(input);
    output = output_tensors;
    output_shape = reinterpret_cast<CVI_SHAPE*>(calloc(output_num, sizeof(CVI_SHAPE)));
    for (int i = 0; i < output_num; i++) {
        output_shape[i] = CVI_NN_TensorShape(&output[i]);
        printf("Output[%d] shape: [%d, %d, %d, %d]\n", i, output_shape[i].dim[0], output_shape[i].dim[1],
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

    while (true) {
        gettimeofday(&start_time, NULL);

        frame_idx++;
        printf("\n[Frame %d]\n", frame_idx);

        // Get YUV frame from VI and convert to BGR
        gettimeofday(&t1, NULL);
        cv::Mat image;
        if (vi_get_frame_as_bgr(vi_channel, image) != CVI_SUCCESS) {
            printf("Failed to get frame from VI channel %d\n", vi_channel);
            usleep(100000); // 休眠0.1秒
            continue;
        }

        if (!image.data) {
            printf("Empty image data\n");
            usleep(100000);
            continue;
        }

        cv::Mat cloned = image.clone();
        gettimeofday(&t2, NULL);
        long read_time = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);

        // Image is already 640x640 from vi_get_frame_as_bgr, no need to resize again
        gettimeofday(&t1, NULL);
        
        // Convert BGR to RGB
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        // Packed2Planar
        cv::Mat channels[3];
        for (int i = 0; i < 3; i++) {
            channels[i] = cv::Mat(image.rows, image.cols, CV_8SC1);
        }
        cv::split(image, channels);

        // fill data
        int8_t* ptr = (int8_t*)CVI_NN_TensorPtr(input);
        int channel_size = height * width;
        for (int i = 0; i < 3; ++i) {
            memcpy(ptr + i * channel_size, channels[i].data, channel_size);
        }
        gettimeofday(&t2, NULL);
        long preprocess_time = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);

        // run inference
        gettimeofday(&t1, NULL);
        CVI_NN_Forward(model, input_tensors, input_num, output_tensors, output_num);
        gettimeofday(&t2, NULL);
        long inference_time = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        // do post proprocess
        gettimeofday(&t1, NULL);
        int det_num = 0;
        std::vector<detection> dets;

        det_num = getDetections(output, height, width, classes_num, output_shape[0], conf_thresh, dets);
        // correct box with origin image size
        NMS(dets, &det_num, iou_thresh);
        correctYoloBoxes(dets, det_num, cloned.rows, cloned.cols, height, width);
        gettimeofday(&t2, NULL);
        long postprocess_time = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        // 打印检测到的对象信息并控制电机
        if (det_num > 0) {
            // 选择置信度最高的球
            int best_idx = 0;
            float best_score = dets[0].score;
            for (int i = 1; i < det_num; i++) {
                if (dets[i].score > best_score) {
                    best_score = dets[i].score;
                    best_idx = i;
                }
            }

            box b = dets[best_idx].bbox;
            printf("[DETECT] Ball found: pos(%.1f, %.1f), conf=%.3f\n", b.x, b.y, dets[best_idx].score);

            // 根据球的位置控制电机
            controlMotor(motor, b.x, b.y, cloned.cols, cloned.rows);
            
#if ENABLE_DRAW_BBOX
            // draw bbox on image (only when ball detected)
            for (int i = 0; i < det_num; i++) {
                box b = dets[i].bbox;
                // xywh2xyxy
                int x1 = (b.x - b.w / 2);
                int y1 = (b.y - b.h / 2);
                int x2 = (b.x + b.w / 2);
                int y2 = (b.y + b.h / 2);

                // 确保坐标在图像范围内
                x1 = std::max(0, std::min(x1, cloned.cols - 1));
                y1 = std::max(0, std::min(y1, cloned.rows - 1));
                x2 = std::max(0, std::min(x2, cloned.cols - 1));
                y2 = std::max(0, std::min(y2, cloned.rows - 1));

                cv::rectangle(cloned, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 255), 3, 8, 0);
                char content[100];
                sprintf(content, "%s %0.3f", tennis_names[dets[i].cls], dets[i].score);
                cv::putText(cloned, content, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255),
                            2);
            }
#endif


        } else {
            printf("[DETECT] No ball detected\n");
            printf("[MOTOR] STANDBY\n");
            motor.standby();
        }
        
#if ENABLE_SAVE_IMAGE
            // save picture with detection results (only when ball detected)
            char output_path[256];
            sprintf(output_path, "/boot/images/detected_%d.jpg", frame_idx);
            printf("[DEBUG] Saving image: %dx%d, channels: %d\n", cloned.cols, cloned.rows, cloned.channels());
            cv::imwrite(output_path, cloned);
            printf("[SAVE] %s\n", output_path);
#endif
        // 计算帧率
        gettimeofday(&end_time, NULL);
        long frame_time_us = (end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_usec - start_time.tv_usec);
        float fps = 1000000.0f / frame_time_us;

        frame_count++;
        total_time_us += frame_time_us;
        float avg_fps = 1000000.0f * frame_count / total_time_us;

        printf("[FPS] Current: %.2f, Average: %.2f (total: %.2f ms)\n", fps, avg_fps, frame_time_us / 1000.0f);
        printf("[PROFILE] Read: %.2f ms, Preprocess: %.2f ms, Inference: %.2f ms, "
               "Postprocess: %.2f ms\n",
               read_time / 1000.0f, preprocess_time / 1000.0f, inference_time / 1000.0f, postprocess_time / 1000.0f);

        // 每处理完一张图片，休眠一段时间再处理下一张
        // usleep(500000); // 休眠0.5秒
        if (frame_idx >= 200) {
            // 处理200帧后退出
            break;
        }
    } // end while loop

    // Cleanup
    printf("\nCleaning up...\n");
#if USE_VPSS_RESIZE
    vpss_resize_deinit();
#endif
    sys_vi_deinit();
    CVI_NN_CleanupModel(model);
    printf("CVI_NN_CleanupModel succeeded\n");
    free(output_shape);
    return 0;
}