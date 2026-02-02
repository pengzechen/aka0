#include "vi_capture.hpp"
#include "logger.hpp"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

// VI related headers needed for implementation
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
#include "cvi_sys.h"
#include "cvi_vi.h"
#include "sample_comm.h"

VICapture::VICapture() {
#if USE_VPSS_RESIZE
    m_VpssGrp = 0;
    m_VpssChn = 0;
    m_bVpssInited = false;
#endif
    memset(&m_stViConfig, 0, sizeof(SAMPLE_VI_CONFIG_S));
    memset(&m_stIniCfg, 0, sizeof(SAMPLE_INI_CFG_S));
}

VICapture::~VICapture() {
    deinit();
#if USE_VPSS_RESIZE
    deinitVpssResize();
#endif
}

int VICapture::init() {
    MMF_VERSION_S stVersion;
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
    if (SAMPLE_COMM_VI_ParseIni(&m_stIniCfg)) {
        SAMPLE_PRT("Parse complete\n");
    }

    // Set sensor number
    CVI_VI_SetDevNum(m_stIniCfg.devNum);

    /************************************************
   * step1:  Config VI
   ************************************************/
    s32Ret = SAMPLE_COMM_VI_IniToViCfg(&m_stIniCfg, &m_stViConfig);
    if (s32Ret != CVI_SUCCESS)
        return s32Ret;

    /************************************************
   * step2:  Get input size
   ************************************************/
    s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(m_stIniCfg.enSnsType[0], &enPicSize);
    if (s32Ret != CVI_SUCCESS) {
        LOGE("SAMPLE_COMM_VI_GetSizeBySensor failed with %#x", s32Ret);
        return s32Ret;
    }

    s32Ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSize);
    if (s32Ret != CVI_SUCCESS) {
        LOGE("SAMPLE_COMM_SYS_GetPicSize failed with %#x", s32Ret);
        return s32Ret;
    }

    /************************************************
   * step3:  Init modules
   ************************************************/
    s32Ret = SAMPLE_PLAT_SYS_INIT(stSize);
    if (s32Ret != CVI_SUCCESS) {
        LOGE("sys init failed. s32Ret: 0x%x !", s32Ret);
        return s32Ret;
    }

    s32Ret = SAMPLE_PLAT_VI_INIT(&m_stViConfig);
    if (s32Ret != CVI_SUCCESS) {
        LOGE("vi init failed. s32Ret: 0x%x !", s32Ret);
        return s32Ret;
    }

    return CVI_SUCCESS;
}

void VICapture::deinit() {
    SAMPLE_COMM_VI_DestroyIsp(&m_stViConfig);
    SAMPLE_COMM_VI_DestroyVi(&m_stViConfig);
    SAMPLE_COMM_SYS_Exit();
}

#if USE_VPSS_RESIZE
int VICapture::initVpssResize(int input_w, int input_h, int output_w, int output_h) {
    if (m_bVpssInited)
        return CVI_SUCCESS;

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
    s32Ret = CVI_VPSS_CreateGrp(m_VpssGrp, &stVpssGrpAttr);
    if (s32Ret != CVI_SUCCESS) {
        LOGE("CVI_VPSS_CreateGrp failed: 0x%x", s32Ret);
        return s32Ret;
    }

    // Set channel attribute for resize output
    memset(&stVpssChnAttr, 0, sizeof(VPSS_CHN_ATTR_S));
    stVpssChnAttr.u32Width = output_w;
    stVpssChnAttr.u32Height = output_h;
    stVpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
#if USE_VPSS_BGR
    stVpssChnAttr.enPixelFormat = PIXEL_FORMAT_BGR_888;
#else
    stVpssChnAttr.enPixelFormat = PIXEL_FORMAT_NV21;
#endif
    stVpssChnAttr.stFrameRate.s32SrcFrameRate = -1;
    stVpssChnAttr.stFrameRate.s32DstFrameRate = -1;
    stVpssChnAttr.u32Depth = 1;
    stVpssChnAttr.bMirror = CVI_TRUE;
    stVpssChnAttr.bFlip = CVI_TRUE;
    stVpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
    stVpssChnAttr.stNormalize.bEnable = CVI_FALSE;

    s32Ret = CVI_VPSS_SetChnAttr(m_VpssGrp, m_VpssChn, &stVpssChnAttr);
    if (s32Ret != CVI_SUCCESS) {
        LOGE("CVI_VPSS_SetChnAttr failed: 0x%x (Format: %d)", s32Ret, stVpssChnAttr.enPixelFormat);
        // Fallback to NV21 if BGR is not supported (unlikely on this platform)
        LOGW("VPSS BGR_888 not supported? Trying NV21...");
        stVpssChnAttr.enPixelFormat = PIXEL_FORMAT_NV21;
        s32Ret = CVI_VPSS_SetChnAttr(m_VpssGrp, m_VpssChn, &stVpssChnAttr);
        if (s32Ret != CVI_SUCCESS) {
             LOGE("CVI_VPSS_SetChnAttr fallback failed: 0x%x", s32Ret);
             CVI_VPSS_DestroyGrp(m_VpssGrp);
             return s32Ret;
        }
    }


    abChnEnable[m_VpssChn] = CVI_TRUE;
    s32Ret = CVI_VPSS_EnableChn(m_VpssGrp, m_VpssChn);
    if (s32Ret != CVI_SUCCESS) {
        LOGE("CVI_VPSS_EnableChn failed: 0x%x", s32Ret);
        CVI_VPSS_DestroyGrp(m_VpssGrp);
        return s32Ret;
    }

    s32Ret = CVI_VPSS_StartGrp(m_VpssGrp);
    if (s32Ret != CVI_SUCCESS) {
        LOGE("CVI_VPSS_StartGrp failed: 0x%x", s32Ret);
        CVI_VPSS_DisableChn(m_VpssGrp, m_VpssChn);
        CVI_VPSS_DestroyGrp(m_VpssGrp);
        return s32Ret;
    }

    m_bVpssInited = true;
    LOGI("VPSS initialized: %dx%d -> %dx%d (PixelFormat=%s, Flip/Mirror enabled)", 
         input_w, input_h, output_w, output_h, 
         (stVpssChnAttr.enPixelFormat == PIXEL_FORMAT_BGR_888) ? "BGR_888" : "NV21");
    return CVI_SUCCESS;
}

void VICapture::deinitVpssResize() {
    if (m_bVpssInited) {
        CVI_VPSS_StopGrp(m_VpssGrp);
        CVI_VPSS_DisableChn(m_VpssGrp, m_VpssChn);
        CVI_VPSS_DestroyGrp(m_VpssGrp);
        m_bVpssInited = false;
    }
}
#endif

int VICapture::getFrameAsBGR(CVI_U8 chn, cv::Mat& bgr_image) {
    VIDEO_FRAME_INFO_S stVideoFrame;
    struct timeval t1, t2;
    long get_frame_us, mmap_us, resize_us, cvt_us, flip_us, munmap_us, release_us;

    gettimeofday(&t1, NULL);
    if (CVI_VI_GetChnFrame(0, chn, &stVideoFrame, 3000) == 0) {
        gettimeofday(&t2, NULL);
        get_frame_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);

#if USE_VPSS_RESIZE
        if (!m_bVpssInited) {
             LOGE("VPSS not initialized!");
             CVI_VI_ReleaseChnFrame(0, chn, &stVideoFrame);
             return CVI_FAILURE;
        }

        // Use VPSS hardware to resize
        gettimeofday(&t1, NULL);
        
        // Send frame to VPSS for hardware resize
        CVI_S32 s32Ret = CVI_VPSS_SendFrame(m_VpssGrp, &stVideoFrame, -1);
        if (s32Ret != CVI_SUCCESS) {
            LOGE("CVI_VPSS_SendFrame failed: 0x%x", s32Ret);
            CVI_VI_ReleaseChnFrame(0, chn, &stVideoFrame);
            return CVI_FAILURE;
        }
        
        // Get resized frame from VPSS
        VIDEO_FRAME_INFO_S stResizedFrame;
        s32Ret = CVI_VPSS_GetChnFrame(m_VpssGrp, m_VpssChn, &stResizedFrame, 1000);
        if (s32Ret != CVI_SUCCESS) {
            LOGE("CVI_VPSS_GetChnFrame failed: 0x%x", s32Ret);
        }

        gettimeofday(&t2, NULL);
        resize_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        mmap_us = 0; // No separate mmap needed with VPSS

        // Get resized frame info
        gettimeofday(&t1, NULL);

        int width = stResizedFrame.stVFrame.u32Width;
        int height = stResizedFrame.stVFrame.u32Height;
        int stride_y = stResizedFrame.stVFrame.u32Stride[0];
        
        // VPSS Hardware BGR Output Path (Default)
        if (stResizedFrame.stVFrame.enPixelFormat == PIXEL_FORMAT_BGR_888) {
             size_t image_size = stride_y * height;
             // Use Cached Mmap for faster memory copy (vital for performance)
             CVI_VOID* vir_addr = CVI_SYS_MmapCache(stResizedFrame.stVFrame.u64PhyAddr[0], image_size);
             
             // Invalidate cache to ensure we read fresh data from DRAM
             CVI_SYS_IonInvalidateCache(stResizedFrame.stVFrame.u64PhyAddr[0], vir_addr, image_size);

             if (bgr_image.empty() || bgr_image.rows != height || bgr_image.cols != width || bgr_image.type() != CV_8UC3) {
                 bgr_image.create(height, width, CV_8UC3);
             }

             // Copy from specialized hardware memory to standard BGR Mat
             if (stride_y == width * 3) {
                 memcpy(bgr_image.data, vir_addr, image_size);
             } else {
                 for (int i = 0; i < height; i++) {
                     memcpy(bgr_image.data + i * width * 3, (uint8_t*)vir_addr + i * stride_y, width * 3);
                 }
             }
             
             CVI_SYS_Munmap(vir_addr, image_size);
             
             gettimeofday(&t2, NULL);
             cvt_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
             printf("[VI-DETAIL] GetFrame: %.1fms, VPSS_Resize+CvB: %.1fms, Cvt: %.1fms (VPSSDirect+Cache)\n", 
                    get_frame_us / 1000.0, resize_us / 1000.0, cvt_us / 1000.0);
                    
             CVI_VPSS_ReleaseChnFrame(m_VpssGrp, m_VpssChn, &stResizedFrame);
             CVI_VI_ReleaseChnFrame(0, chn, &stVideoFrame);
             return CVI_SUCCESS;
        }

        // Fallback for NV21 if BGR failed
        int stride_uv = stResizedFrame.stVFrame.u32Stride[1];

        // Map resized frame memory
        size_t y_size = stride_y * height;
        size_t uv_size = stride_uv * height / 2;
        size_t image_size = y_size + uv_size;

        CVI_VOID* vir_addr = CVI_SYS_MmapCache(stResizedFrame.stVFrame.u64PhyAddr[0], image_size);
        CVI_SYS_IonInvalidateCache(stResizedFrame.stVFrame.u64PhyAddr[0], vir_addr, image_size);

        cv::Mat yuv_continuous(height * 3 / 2, width, CV_8UC1);

        if (stride_y == width && stride_uv == width) {
            memcpy(yuv_continuous.data, vir_addr, width * height * 3 / 2);
        } else {
            for (int i = 0; i < height; i++) {
                memcpy(yuv_continuous.data + i * width, (uint8_t*)vir_addr + i * stride_y, width);
            }
            for (int i = 0; i < height / 2; i++) {
                memcpy(yuv_continuous.data + height * width + i * width, (uint8_t*)vir_addr + y_size + i * stride_uv, width);
            }
        }

        cv::cvtColor(yuv_continuous, bgr_image, cv::COLOR_YUV2BGR_NV21);

        gettimeofday(&t2, NULL);
        cvt_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);

        printf("[VI-DETAIL] GetFrame: %.1fms, VPSS_Resize: %.1fms, Cvt: %.1fms (Stride=%d)\n", 
               get_frame_us / 1000.0, resize_us / 1000.0, cvt_us / 1000.0, stride_y);

        CVI_SYS_Munmap(vir_addr, image_size);
        CVI_VPSS_ReleaseChnFrame(m_VpssGrp, m_VpssChn, &stResizedFrame);
        CVI_VI_ReleaseChnFrame(0, chn, &stVideoFrame);

        return CVI_SUCCESS;
#else
        // Software resize path
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
        // Optimization: Use INTER_NEAREST for faster resize
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

        LOGD("[VI-DETAIL] GetFrame: %.1fms, Mmap: %.1fms, Resize: %.1fms, Cvt: %.1fms", get_frame_us / 1000.0,
               mmap_us / 1000.0, resize_us / 1000.0, cvt_us / 1000.0);

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
            LOGE("CVI_VI_ReleaseChnFrame NG");
            return CVI_FAILURE;
        }
        gettimeofday(&t2, NULL);
        release_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);

        LOGD("[VI-DETAIL] Flip: %.1fms, Munmap: %.1fms, Release: %.1fms", flip_us / 1000.0, munmap_us / 1000.0,
               release_us / 1000.0);

        return CVI_SUCCESS;
#endif // USE_VPSS_RESIZE
    }

    LOGE("CVI_VI_GetChnFrame NG");
    return CVI_FAILURE;
}