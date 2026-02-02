#ifndef VI_CAPTURE_HPP
#define VI_CAPTURE_HPP

#include <opencv2/opencv.hpp>
#include <linux/cvi_type.h>
#include <linux/cvi_common.h>
#include "cvi_vpss.h"
#include "sample_comm.h"

// 控制宏定义
#define USE_VPSS_RESIZE 1 // 使用VPSS硬件加速resize

/**
 * @brief VI视频捕获类，封装了视频输入(VI)和VPSS硬件缩放功能
 */
class VICapture {
public:
    VICapture();
    ~VICapture();

    /**
     * @brief 初始化VI系统
     * @return CVI_SUCCESS on success, otherwise CVI_FAILURE
     */
    int init();

    /**
     * @brief 反初始化VI系统
     */
    void deinit();

#if USE_VPSS_RESIZE
    /**
     * @brief 初始化VPSS硬件缩放
     * @param input_w 输入图像宽度
     * @param input_h 输入图像高度
     * @param output_w 输出图像宽度
     * @param output_h 输出图像高度
     * @return CVI_SUCCESS on success, otherwise CVI_FAILURE
     */
    int initVpssResize(int input_w, int input_h, int output_w, int output_h);

    /**
     * @brief 反初始化VPSS硬件缩放
     */
    void deinitVpssResize();
#endif

    /**
     * @brief 从VI通道获取帧并转换为BGR格式
     * @param chn VI通道号
     * @param bgr_image 输出的BGR图像
     * @return CVI_SUCCESS on success, otherwise CVI_FAILURE
     */
    int getFrameAsBGR(CVI_U8 chn, cv::Mat& bgr_image);

private:
    SAMPLE_VI_CONFIG_S m_stViConfig;
    SAMPLE_INI_CFG_S m_stIniCfg;

#if USE_VPSS_RESIZE
    VPSS_GRP m_VpssGrp;
    VPSS_CHN m_VpssChn;
    bool m_bVpssInited;
#endif
};

#endif // VI_CAPTURE_HPP
