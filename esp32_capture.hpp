#ifndef ESP32_CAPTURE_HPP
#define ESP32_CAPTURE_HPP

#include <stdint.h>
#include <stdbool.h>
#include <opencv2/opencv.hpp>

// ioctl commands matching StarryOS kernel CviCamera (cvi_camera.rs)
#define ESP32_CAM_INIT      1
#define ESP32_CAM_GET_INFO  2
#define ESP32_CAM_GET_FRAME 3

#define ESP32_FRAME_MAX_SIZE (2 * 1024 * 1024)

struct ESP32CameraInfo {
    uint16_t width;
    uint16_t height;
    uint8_t format;
    uint8_t connected;
};

class ESP32Capture {
public:
    ESP32Capture();
    ~ESP32Capture();

    int init(const char* device = "/dev/cvi-camera");
    void deinit();

    // Set target resize dimensions (software resize, no VPSS)
    void setResize(int target_w, int target_h);

    // Get frame as BGR, resized to target dimensions
    int getFrameAsBGR(cv::Mat& bgr_image);

    int getWidth() const { return m_camInfo.width; }
    int getHeight() const { return m_camInfo.height; }
    bool isConnected() const { return m_camInfo.connected != 0; }

private:
    int m_fd;
    bool m_initialized;
    uint8_t* m_frameBuf;
    int m_targetW;
    int m_targetH;
    ESP32CameraInfo m_camInfo;
};

#endif // ESP32_CAPTURE_HPP
