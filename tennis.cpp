#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include "cviruntime.h"


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
  printf("   %s cvimodel image.jpg image_detected.jpg\n", argv[0]);
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
    
    printf("=== Coordinate correction ===\n");
    printf("Original image: %dx%d, Input size: %dx%d\n", image_w, image_h, input_width, input_height);
    printf("Scale: %.3f, New size: %dx%d, Padding: left=%d, top=%d\n", 
           scale, new_w, new_h, pad_left, pad_top);

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
        
        printf("Det[%d]: input_bbox(%.1f,%.1f,%.1f,%.1f) -> output_bbox(%.1f,%.1f,%.1f,%.1f)\n",
               i, cx, cy, w, h, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
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
    // 添加调试信息：打印输出tensor信息
    printf("=== DEBUG: Output tensor information ===\n");
    printf("Output shape: [%d, %d, %d, %d]\n", 
           output_shape.dim[0], output_shape.dim[1], output_shape.dim[2], output_shape.dim[3]);
    
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
    
    printf("Batch: %d, Channels: %d, Total_anchors: %d\n", batch, channels, total_anchors);
    
    // 计算每个stride层的anchor数量
    int anchor_counts[3];
    for (int i = 0; i < 3; i++) {
        int nh = input_height / stride[i];
        int nw = input_width / stride[i];
        anchor_counts[i] = nh * nw;
        printf("Stride[%d]: %f, grid: %dx%d, anchors: %d\n", i, stride[i], nh, nw, anchor_counts[i]);
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
                
                printf("Detection[%d]: conf=%.3f, bbox_center=(%.1f,%.1f), size=(%.1f,%.1f)\n", 
                       count, objectness, cx, cy, w, h);
                
                count++;
                dets.emplace_back(det);
            }
            
            anchor_offset += current_anchors;
        }
    }
    return count;
}

int main(int argc, char **argv) {
  int ret = 0;
  CVI_MODEL_HANDLE model;

  if (argc != 4) {
    usage(argv);
    exit(-1);
  }
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
  // imread
  cv::Mat image;
  image = cv::imread(argv[2]);
  if (!image.data) {
    printf("Could not open or find the image\n");
    return -1;
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
  printf("CVI_NN_Forward Succeed...\n");
  // do post proprocess
  int det_num = 0;
  std::vector<detection> dets;
  
  // 不再检查输出tensor数量，因为新版本支持单一输出tensor
  printf("Processing single output tensor format...\n");
  
  det_num = getDetections(output, height, width, classes_num, output_shape[0],  
                          conf_thresh, dets);
  // correct box with origin image size
  NMS(dets, &det_num, iou_thresh);
  correctYoloBoxes(dets, det_num, cloned.rows, cloned.cols, height, width);

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
    
    printf("Drawing box[%d]: (x1=%d, y1=%d, x2=%d, y2=%d) on image size(%dx%d)\n", 
           i, x1, y1, x2, y2, cloned.cols, cloned.rows);
    
    cv::rectangle(cloned, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 255),
                  3, 8, 0);
    char content[100];
    sprintf(content, "%s %0.3f", tennis_names[dets[i].cls], dets[i].score);
    cv::putText(cloned, content, cv::Point(x1, y1 - 10),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
  }
  
  // 打印检测到的对象信息
  printf("=== Final Detection Results ===\n");
  for (int i = 0; i < det_num; i++) {
    box b = dets[i].bbox;
    printf("Object[%d]: %s, score=%.3f, bbox=(%.1f, %.1f, %.1f, %.1f)\n", 
           i, tennis_names[dets[i].cls], dets[i].score, b.x, b.y, b.w, b.h);
  }

  // save or show picture
  cv::imwrite(argv[3], cloned);

  printf("------\n");
  printf("%d objects are detected\n", det_num);
  printf("------\n");

  CVI_NN_CleanupModel(model);
  printf("CVI_NN_CleanupModel succeeded\n");
  free(output_shape);
  return 0;
}