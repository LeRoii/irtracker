#pragma once

#include "basic.h"
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include "opencv2/opencv.hpp"

using namespace std;

class yolo
{
private:
    uint32_t g_modelWidth_;   // The input width required by the model
    uint32_t g_modelHeight_;  // The model requires high input
    const char *g_modelPath_; // Offline model file path
    basic *Yolo_Instant;

public:
    yolo(const char *modelPath, uint32_t modelWidth, uint32_t modelHeight);
    ~yolo();

    Result Init();
    // Result Preprocess(ImgMat frame);
    Result Preprocess(cv::Mat &image);
    Result Inference();
    vector<BBox> Postprocess();
    Result UnInit();

    void NMS(vector<BBox> &boxes, vector<BBox> &result);
    float IOU(const BBox &b1, const BBox &b2);
    static bool SortScore(BBox box1, BBox box2);
    void DrawBoundBoxToImage(vector<BBox> &detectionResults, cv::Mat &image);
};
