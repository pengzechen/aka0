#include "esp32_capture.hpp"
#include "logger.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cstring>
#include <cstdio>
#include <errno.h>

ESP32Capture::ESP32Capture()
    : m_fd(-1), m_initialized(false), m_frameBuf(nullptr),
      m_targetW(640), m_targetH(640)
{
    memset(&m_camInfo, 0, sizeof(m_camInfo));
}

ESP32Capture::~ESP32Capture()
{
    deinit();
}

int ESP32Capture::init(const char* device)
{
    if (m_initialized) return 0;

    m_fd = open(device, O_RDWR);
    if (m_fd < 0) {
        LOGE("[ESP32] open %s failed: %s (errno=%d)", device, strerror(errno), errno);
        return -1;
    }

    // Initialize camera
    if (ioctl(m_fd, ESP32_CAM_INIT, 0) < 0) {
        LOGE("[ESP32] ioctl(INIT) failed: %s", strerror(errno));
        close(m_fd);
        m_fd = -1;
        return -1;
    }
    LOGI("[ESP32] camera initialized");

    // Get camera info
    memset(&m_camInfo, 0, sizeof(m_camInfo));
    if (ioctl(m_fd, ESP32_CAM_GET_INFO, &m_camInfo) < 0) {
        LOGE("[ESP32] ioctl(GET_INFO) failed: %s", strerror(errno));
        close(m_fd);
        m_fd = -1;
        return -1;
    }
    LOGI("[ESP32] camera info: %ux%u format=%u connected=%u",
         m_camInfo.width, m_camInfo.height, m_camInfo.format, m_camInfo.connected);

    // Allocate frame buffer
    m_frameBuf = (uint8_t*)malloc(ESP32_FRAME_MAX_SIZE);
    if (!m_frameBuf) {
        LOGE("[ESP32] malloc frame buffer failed");
        close(m_fd);
        m_fd = -1;
        return -1;
    }

    m_initialized = true;
    return 0;
}

void ESP32Capture::deinit()
{
    if (m_frameBuf) {
        free(m_frameBuf);
        m_frameBuf = nullptr;
    }
    if (m_fd >= 0) {
        close(m_fd);
        m_fd = -1;
    }
    m_initialized = false;
}

void ESP32Capture::setResize(int target_w, int target_h)
{
    m_targetW = target_w;
    m_targetH = target_h;
}

int ESP32Capture::getFrameAsBGR(cv::Mat& bgr_image)
{
    if (!m_initialized || m_fd < 0) {
        LOGE("[ESP32] not initialized");
        return -1;
    }

    // Get JPEG frame from kernel driver
    int len = ioctl(m_fd, ESP32_CAM_GET_FRAME, m_frameBuf);
    if (len < 0) {
        LOGE("[ESP32] ioctl(GET_FRAME) failed: %s", strerror(errno));
        return -1;
    }
    if (len == 0) {
        LOGW("[ESP32] empty frame");
        return -1;
    }

    LOGD("[ESP32] got frame: %d bytes", len);

    // Decode JPEG
    cv::Mat raw(1, len, CV_8UC1, m_frameBuf);
    cv::Mat decoded = cv::imdecode(raw, cv::IMREAD_COLOR);
    if (decoded.empty()) {
        LOGE("[ESP32] imdecode failed");
        return -1;
    }

    // Resize to target dimensions
    if (decoded.cols != m_targetW || decoded.rows != m_targetH) {
        cv::resize(decoded, bgr_image, cv::Size(m_targetW, m_targetH));
    } else {
        bgr_image = decoded;
    }

    return 0;
}
