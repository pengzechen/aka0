#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include "cviruntime.h"
#include "motor.hpp"
#include <unistd.h>

// 控制宏定义
#define ENABLE_DEBUG_OUTPUT 0      // 是否启用详细调试输出
#define ENABLE_DRAW_BBOX 0         // 是否画框并保存图片
#define ENABLE_SAVE_IMAGE 0        // 是否保存检测结果图片


typedef struct {
  float x, y, w, h;
} box;

typedef struct {
  box bbox;
  int cls;
  float score;
  int batch_idx;
} detection;

static const char *tennis_names[] = {
    "tennis"};  // 单类别网球检测

static void usage(char **argv) {
  printf("Usage:\n");
  printf("   %s cvimodel num_images\n", argv[0]);
  printf("   Example: %s model.cvimodel 10\n", argv[0]);
  printf("   This will process images 1.jpg to 10.jpg from /data/images/\n");
}

template <typename T>
int argmax(const T *data,
          size_t len,
          size_t stride = 1)
{
	int maxIndex = 0;
	for (size_t i = stride; i < len; i += stride)
	{
		if (data[maxIndex] < data[i])
		{
			maxIndex = i;
		}
	}
	return maxIndex;
}

float calIou(box a, box b)
{
  float area1 = a.w * a.h;
  float area2 = b.w * b.h;
  float wi = std::min((a.x + a.w / 2), (b.x + b.w / 2)) - std::max((a.x - a.w / 2), (b.x - b.w / 2));
  float hi = std::min((a.y + a.h / 2), (b.y + b.h / 2)) - std::max((a.y - a.h / 2), (b.y - b.h / 2));
  float area_i = std::max(wi, 0.0f) * std::max(hi, 0.0f);
  return area_i / (area1 + area2 - area_i);
}

static void NMS(std::vector<detection> &dets, int *total, float thresh)
{
  if (*total){
    std::sort(dets.begin(), dets.end(), [](detection &a, detection &b)
              { return b.score < a.score; });
    int new_count = *total;
    for (int i = 0; i < *total; ++i)
    {
      detection &a = dets[i];
      if (a.score == 0)
        continue;
      for (int j = i + 1; j < *total; ++j)
      {
        detection &b = dets[j];
        if (dets[i].batch_idx == dets[j].batch_idx &&
            b.score != 0 && dets[i].cls == dets[j].cls &&
            calIou(a.bbox, b.bbox) > thresh)
        {
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

void correctYoloBoxes(std::vector<detection> &dets,
                      int det_num,
                      int image_h,
                      int image_w,
                      int input_height,
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
    printf("Scale: %.3f, New size: %dx%d, Padding: left=%d, top=%d\n", 
           scale, new_w, new_h, pad_left, pad_top);
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
        dets[i].bbox.x = (x1 + x2) / 2.0f;  // 中心点x
        dets[i].bbox.y = (y1 + y2) / 2.0f;  // 中心点y
        dets[i].bbox.w = x2 - x1;           // 宽度
        dets[i].bbox.h = y2 - y1;           // 高度
        
#if ENABLE_DEBUG_OUTPUT
        printf("Det[%d]: input_bbox(%.1f,%.1f,%.1f,%.1f) -> output_bbox(%.1f,%.1f,%.1f,%.1f)\n",
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
int getDetections(CVI_TENSOR *output,
                  int32_t input_height,
                  int32_t input_width,
                  int classes_num,
                  CVI_SHAPE output_shape,
                  float conf_thresh,
                  std::vector<detection> &dets) {
#if ENABLE_DEBUG_OUTPUT
    // 添加调试信息：打印输出tensor信息
    printf("=== DEBUG: Output tensor information ===\n");
    printf("Output shape: [%d, %d, %d, %d]\n", 
           output_shape.dim[0], output_shape.dim[1], output_shape.dim[2], output_shape.dim[3]);
#endif
    
    // 检查是否有足够的输出tensor
    if (output == nullptr) {
        printf("ERROR: output tensor is null\n");
        return 0;
    }
    
    float *output_ptr = (float *)CVI_NN_TensorPtr(&output[0]);
    
    // 检查指针是否有效
    if (output_ptr == nullptr) {
        printf("ERROR: tensor pointer is null\n");
        return 0;
    }
    
    float stride[3] = {8, 16, 32};
    int count = 0;
    int batch = output_shape.dim[0];
    int channels = output_shape.dim[1]; // 应该是4(bbox) + 1(objectness) + classes_num
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
                    printf("Anchor[%d]: raw values = [%.6f, %.6f, %.6f, %.6f, %.6f]\n", 
                           total_anchor_idx,
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
                det.bbox.x = cx;   // 中心点x坐标
                det.bbox.y = cy;   // 中心点y坐标
                det.bbox.w = w;    // 宽度
                det.bbox.h = h;    // 高度
                
#if ENABLE_DEBUG_OUTPUT
                printf("Detection[%d]: conf=%.3f, bbox_center=(%.1f,%.1f), size=(%.1f,%.1f)\n", 
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
void controlMotor(Motor &motor, float ball_x, float ball_y, int image_width, int image_height) {
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

int main(int argc, char **argv) {
  int ret = 0;
  CVI_MODEL_HANDLE model;

  if (argc != 3) {
    usage(argv);
    exit(-1);
  }
  
  int num_images = atoi(argv[2]);
  if (num_images <= 0) {
    printf("Error: num_images must be positive\n");
    exit(-1);
  }
  
  printf("Will process %d images from /data/images/\n", num_images);
  
  // 初始化电机
  Motor motor;
  CVI_TENSOR *input;
  CVI_TENSOR *output;
  CVI_TENSOR *input_tensors;
  CVI_TENSOR *output_tensors;
  int32_t input_num;
  int32_t output_num;
  CVI_SHAPE input_shape;
  CVI_SHAPE* output_shape;
  int32_t height;
  int32_t width;
  //int bbox_len = 5; // 1 class + 4 bbox
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
  CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num, &output_tensors,
                               &output_num);

  printf("=== DEBUG: Model information ===\n");
  printf("Input number: %d, Output number: %d\n", input_num, output_num);

  input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);
  assert(input);
  output = output_tensors;
  output_shape = reinterpret_cast<CVI_SHAPE *>(calloc(output_num, sizeof(CVI_SHAPE)));
  for (int i = 0; i < output_num; i++)
  {
    output_shape[i] = CVI_NN_TensorShape(&output[i]);
    printf("Output[%d] shape: [%d, %d, %d, %d]\n", i,
           output_shape[i].dim[0], output_shape[i].dim[1], 
           output_shape[i].dim[2], output_shape[i].dim[3]);
  }

  // nchw
  input_shape = CVI_NN_TensorShape(input);
  height = input_shape.dim[2];
  width = input_shape.dim[3];
  assert(height % 32 == 0 && width %32 == 0);
  
  // 循环处理图片
  int image_idx = 0;
  struct timeval start_time, end_time;
  long total_time_us = 0;
  int frame_count = 0;
  
  while (true) {
    gettimeofday(&start_time, NULL);
    
    // 构建图片路径
    char image_path[256];
    sprintf(image_path, "/data/images/%d.jpg", (image_idx % num_images) + 1);
    image_idx++;
    
    printf("\n[Frame %d] %s\n", image_idx, image_path);
    
    // imread
    cv::Mat image;
    image = cv::imread(image_path);
    if (!image.data) {
      printf("Could not open or find the image: %s\n", image_path);
      printf("Skipping to next image...\n");
      usleep(500000); // 休眠0.5秒
      continue;
    }
    cv::Mat cloned = image.clone();

    // resize & letterbox
    int ih = image.rows;
    int iw = image.cols;
    int oh = height;
    int ow = width;
    double resize_scale = std::min((double)oh / ih, (double)ow / iw);
    int nh = (int)(ih * resize_scale);
    int nw = (int)(iw * resize_scale);
    cv::resize(image, image, cv::Size(nw, nh));
    int top = (oh - nh) / 2;
    int bottom = (oh - nh) - top;
    int left = (ow - nw) / 2;
    int right = (ow - nw) - left;
    cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT,
                      cv::Scalar::all(0));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    //Packed2Planar
    cv::Mat channels[3];
    for (int i = 0; i < 3; i++) {
      channels[i] = cv::Mat(image.rows, image.cols, CV_8SC1);
    }
    cv::split(image, channels);

    // fill data
    int8_t *ptr = (int8_t *)CVI_NN_TensorPtr(input);
    int channel_size = height * width;
    for (int i = 0; i < 3; ++i) {
      memcpy(ptr + i * channel_size, channels[i].data, channel_size);
    }

    // run inference
    CVI_NN_Forward(model, input_tensors, input_num, output_tensors, output_num);
  // do post proprocess
  int det_num = 0;
  std::vector<detection> dets;
  
  det_num = getDetections(output, height, width, classes_num, output_shape[0],  
                          conf_thresh, dets);
    // correct box with origin image size
    NMS(dets, &det_num, iou_thresh);
    correctYoloBoxes(dets, det_num, cloned.rows, cloned.cols, height, width);

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
  } else {
    printf("[DETECT] No ball detected\n");
    printf("[MOTOR] STANDBY\n");
    motor.standby();
  }
  
#if ENABLE_DRAW_BBOX
  // draw bbox on image
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
    
    cv::rectangle(cloned, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 255),
                  3, 8, 0);
    char content[100];
    sprintf(content, "%s %0.3f", tennis_names[dets[i].cls], dets[i].score);
    cv::putText(cloned, content, cv::Point(x1, y1 - 10),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
  }
#endif

#if ENABLE_SAVE_IMAGE
  // save picture with detection results
  char output_path[256];
  sprintf(output_path, "/data/images/detected_%d.jpg", image_idx);
  cv::imwrite(output_path, cloned);
  printf("[SAVE] %s\n", output_path);
#endif
  
  // 计算帧率
  gettimeofday(&end_time, NULL);
  long frame_time_us = (end_time.tv_sec - start_time.tv_sec) * 1000000 + 
                       (end_time.tv_usec - start_time.tv_usec);
  float fps = 1000000.0f / frame_time_us;
  
  frame_count++;
  total_time_us += frame_time_us;
  float avg_fps = 1000000.0f * frame_count / total_time_us;
  
  printf("[FPS] Current: %.2f, Average: %.2f (frame time: %.2f ms)\n", 
         fps, avg_fps, frame_time_us / 1000.0f);
  
    // 每处理完一张图片，休眠一段时间再处理下一张
    // usleep(500000); // 休眠0.5秒
  } // end while loop

  CVI_NN_CleanupModel(model);
  printf("CVI_NN_CleanupModel succeeded\n");
  free(output_shape);
  return 0;
}