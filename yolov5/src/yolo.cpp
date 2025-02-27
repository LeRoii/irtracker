#include "yolo.h"
using namespace std;


namespace {
const static std::vector<std::string> yolov5Label = { "person", "car", "Boat" };

// Inferential output dataset subscript 0 unit is detection box information data
const uint32_t g_OutId = 0;
// The unit with subscript 1 is the number of boxes
const uint32_t g_modelOutputBoxNum = 25200;    //20160   80640
// const uint32_t g_all = 15;    //5 + 80  //xywhc cc...cc
const uint32_t g_all = 6;    //5 + 80  //xywhc cc...cc
const double g_NMSThreshold = 0.45;
const double g_scoreThreshold = 0.25;

 const vector<cv::Scalar> g_colors {
        cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(0, 125, 255)};
}


yolo::yolo(const char* modelPath, uint32_t modelWidth,
                           uint32_t modelHeight)
    :g_modelWidth_(modelWidth), g_modelHeight_(modelHeight)
{
    g_modelPath_ = modelPath;
    
}

yolo::~yolo()
{
}


Result yolo::Init() {

    // pInstant = new basic (g_modelPath_, g_modelWidth_,  g_modelHeight_);
    Yolo_Instant = new basic(g_modelPath_, g_modelWidth_,  g_modelHeight_);

    Result ret = Yolo_Instant->Init_atlas();
    ret = Yolo_Instant->CreateInput();
    ret = Yolo_Instant->CreateOutput();

    return SUCCESS;
}

Result yolo::UnInit() {
    Yolo_Instant->DestroyResource();
}


Result yolo::Preprocess(cv::Mat &image) {
    // Scale the frame image to the desired size of the model
    cv::Mat img640;
    cv::resize(image, img640, cv::Size(g_modelWidth_, g_modelHeight_));
    // if (frame.empty()) {
    //     std::cout<<"Input image failed"<<std::endl;
    //     return FAILED;
    // }

    //  if(frame.rows != 1024 && frame.cols != 1280){
    //     std::cout<<"Input image not 1280*1024 "<<std::endl;
    //     return FAILED;
    // }

    // Copy the data into the cache of the input dataset
    // aclrtMemcpyKind policy = (g_runMode_ == ACL_HOST)?
    //                          ACL_MEMCPY_HOST_TO_DEVICE:ACL_MEMCPY_DEVICE_TO_DEVICE;
    aclError ret = aclrtMemcpy(Yolo_Instant->g_imageDataBuf_, Yolo_Instant->g_imageDataSize_,
                               img640.ptr<uint8_t>(), Yolo_Instant->g_imageDataSize_, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret) {
        std::cout<<"Copy resized image data to device failed."<<std::endl;
        return FAILED;
    }

    return SUCCESS;
}




Result yolo::Inference() {
    
    struct  timeval tstart,tend;
    double timeuse;
    gettimeofday(&tstart,NULL);

    Result ret = Yolo_Instant->inference();

    gettimeofday(&tend,NULL);
    timeuse = 1000000*(tend.tv_sec - tstart.tv_sec) + \
				(tend.tv_usec - tstart.tv_usec);
    std::cout<<"Inference time:  "<< timeuse/1000<<"ms"<<std::endl;

    return SUCCESS;
}




vector<BBox> yolo::Postprocess() {
    struct  timeval tstart,tend,tbuffer;
    double timeuse;
    gettimeofday(&tstart,NULL);

    uint32_t dataSize = 0;
    float* detectData = (float *)Yolo_Instant->GetInferenceOutputItem(dataSize, g_OutId);    

    // if (detectData == nullptr) return FAILED;
    vector <BBox> boxes;

    gettimeofday(&tbuffer,NULL);

    timeuse = 1000000*(tbuffer.tv_sec - tstart.tv_sec) + \
	                      (tbuffer.tv_usec - tstart.tv_usec);
    std::cout<<"Buffer time: " <<timeuse/1000<<std::endl;

    for (uint32_t i = 0; i < g_modelOutputBoxNum; i++) {
        float maxValue = 0;
        float maxIndex = 0;
        float cf = detectData[4 + i * g_all];
        if(cf >= g_scoreThreshold)
        {
            float x = detectData[0 + i * g_all];
            float y = detectData[1 + i * g_all];
            float w = detectData[2 + i * g_all];
            float h = detectData[3 + i * g_all];

            for (uint32_t j = 5; j < g_all; ++j) {
                float value = detectData[i * g_all + j];
                if (value > maxValue) {
                    maxIndex = j - 5;
                    maxValue = value;
                }
            }
            //cout << "--------------- --第" << i << "轮类别筛选结束----------" << endl;
            if (cf * maxValue >= g_scoreThreshold) {
                BBox b;
                b.x = x;
                b.y = y;
                b.w = w;
                b.h = h;
                b.score = cf;
                b.classIndex = maxIndex;
                b.index = i;
                
                boxes.push_back(b);
            

            }
        }
    }

    vector <BBox> result;
    NMS(boxes, result);

    // DrawBoundBoxToImage(result, image);

    gettimeofday(&tend,NULL);
    timeuse = 1000000*(tend.tv_sec - tstart.tv_sec) + \
		           (tend.tv_usec - tstart.tv_usec);
    std::cout<<"Postprocess time: "<<timeuse/1000<<std::endl;
    if (Yolo_Instant->g_runMode_ == ACL_HOST) {
	aclError ret = aclrtFreeHost(detectData);
        //delete[]((uint8_t *)detectData);
        //delete[]((uint8_t*)boxNum);
    }
    std::cout<<result.size()<<std::endl;

    int half = 2;
    float w = float(640) / float(g_modelWidth_);
    float h = float(512) / float(g_modelHeight_);
    for (int i = 0; i < result.size(); ++i) {
        cv::Point p1, p2;
        p1.x = (uint32_t)((result[i].x - result[i].w / half) * w);
        p1.y = (uint32_t)((result[i].y - result[i].h / half) * h);
        p2.x = (uint32_t)((result[i].x + result[i].w / half) * w);
        p2.y = (uint32_t)((result[i].y + result[i].h / half) * h);

        result[i].x = p1.x;
        result[i].y = p1.y;
        result[i].w = p2.x - p1.x;
        result[i].h = p2.y - p1.y;
        // cv::rectangle(image, p1, p2, g_colors[detectionResults[i].classIndex], 2);
        // char name[50];
        // sprintf(name, "%.1d_%.3f", detectionResults[i].classIndex, detectionResults[i].score);
        // // 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        // cv::putText(image, name, p1, cv::FONT_HERSHEY_SIMPLEX, 1, g_colors[detectionResults[i].classIndex], 1);
	}


    return result;
}




void yolo::NMS(vector<BBox> &boxes, vector<BBox> &result)
{
    result.clear();
    sort(boxes.begin(), boxes.end(), SortScore);  //按照置信度从大到小排序

    while (boxes.size() != 0) {
        result.push_back(boxes[0]);
        size_t index = 1;
        while (boxes.size() > index) {
            float iou = IOU(boxes[0], boxes[index]);
            if (iou > g_NMSThreshold) {
                boxes.erase(boxes.begin() + index);
                continue;
            }
            ++index;
            //cout<<index<<endl;
        }
        boxes.erase(boxes.begin());
    }
}

float yolo::IOU(const BBox &b1, const BBox &b2)
{
    float x1 = max(b1.x - b1.w / 2, b2.x - b2.w / 2);
    float y1 = max(b1.y - b1.h / 2, b2.y - b2.h / 2);
    float x2 = min(b1.x + b1.w / 2, b2.x + b2.w / 2);
    float y2 = min(b1.y + b1.h / 2, b2.y + b2.h / 2);
    float w = max(0.0f, x2 - x1 + 1);
    float h = max(0.0f, y2 - y1 + 1);
    float area = w * h;
    return area / (b1.w * b1.h + b2.w * b2.h - area);
}

bool yolo::SortScore(BBox box1, BBox box2)
{
    return box1.score > box2.score;
}

void yolo::DrawBoundBoxToImage(vector<BBox>& detectionResults, cv::Mat& image)
{
    // int half = 2;
    // float w = float(image.cols) / float(g_modelWidth_);
    // float h = float(image.rows) / float(g_modelHeight_);
    // for (int i = 0; i < detectionResults.size(); ++i) {
    //     cv::Point p1, p2;
    //     p1.x = (uint32_t)((detectionResults[i].x - detectionResults[i].w / half) * w);
    //     p1.y = (uint32_t)((detectionResults[i].y - detectionResults[i].h / half) * h);
    //     p2.x = (uint32_t)((detectionResults[i].x + detectionResults[i].w / half) * w);
    //     p2.y = (uint32_t)((detectionResults[i].y + detectionResults[i].h / half) * h);
    //     cv::rectangle(image, p1, p2, g_colors[detectionResults[i].classIndex], 2);
    //     char name[50];
    //     sprintf(name, "%.1d_%.3f", detectionResults[i].classIndex, detectionResults[i].score);
    //     // 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    //     cv::putText(image, name, p1, cv::FONT_HERSHEY_SIMPLEX, 1, g_colors[detectionResults[i].classIndex], 1);
	// }

    // // cv::imshow("output_pic", image);
    // cv::imwrite("output_pic.jpg", image);
    for (int i = 0; i < detectionResults.size(); ++i) {
        cv::rectangle(image, cv::Rect(detectionResults[i].x,detectionResults[i].y,detectionResults[i].w,detectionResults[i].h), g_colors[detectionResults[i].classIndex], 2);
        char name[50];
        sprintf(name, "%.1d_%.3f", detectionResults[i].classIndex, detectionResults[i].score);
        // 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        cv::putText(image, name, cv::Point(detectionResults[i].x,detectionResults[i].y), cv::FONT_HERSHEY_SIMPLEX, 1, g_colors[detectionResults[i].classIndex], 1);
    }
}
