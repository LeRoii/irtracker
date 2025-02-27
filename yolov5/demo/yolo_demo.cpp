// #include "yolo.h"
// #include <chrono>
// #include "tracker.h"
// using namespace cv;
// int main()
// {
//     uint32_t g_modelWidth = 640;
//     uint32_t g_modelHeight = 640;
//     std::string imgname = "/root/code/ascend-c-yolov5/yolov5/IR.jpg";
//     const char* g_modelPath = "/root/code/ascend-c-yolov5/ir_best_no.onnx.om";
//     // const char* g_modelPath = "/root/code/ascend-c-yolov5/640_640_VL_best_3.om";
    

//     yolo * detect = new yolo(g_modelPath, g_modelWidth, g_modelHeight);
//     Result ret = detect->Init();

//     // cv::Mat img = cv::imread(imgname, 1);
//     cv::Mat img = cv::imread(imgname);
//     vector<BBox> output;
//     cv::VideoCapture cap("/root/code/ascend-c-yolov5/yolov5/2024-02-22-16-08-53.mp4");

//     cv::Mat frame;
    
//     cv::resize(img, frame, cv::Size(g_modelWidth, g_modelHeight));


//     realtracker *tracker = new realtracker();

//     bool trackerInit = false;

//     cv::Mat trackret;

//     int val = 0;

//     while(1)
//     {
//         cap >> img;

//         if(img.empty())
//             return 0;

//         trackret = img.clone();
//         cv::resize(img, frame, cv::Size(g_modelWidth, g_modelHeight));
//         ret = detect->Preprocess(frame);
//         auto start = std::chrono::high_resolution_clock::now();
//         for(int i=0;i<1;i++)
//             ret = detect->Inference();
//         auto end = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

//         std::cout << "elapse:" << duration.count()/100 << " ms" << std::endl;

//         output = detect->Postprocess();
//         detect->DrawBoundBoxToImage(output, img);
//         output.clear();

//         if(!trackerInit)
//         {
//             trackerInit = true;
//             tracker->init(cv::Point(630,440), img);
//         }
//         else
//         {
//             cv::Rect rect = tracker->update(trackret);
//             cv::rectangle(trackret, rect, cv::Scalar(255,255,255), 2);

//         }
        

        

//         cv::imshow("1", img);
//         cv::imshow("2", trackret);
//         char c = cv::waitKey(val);
//         if(c == 'g')
//             val = 1;
//         else if(c == 's')
//             val = 0;
//     }
    
//     cv::imwrite("ir.png", img);

//     ret = detect->UnInit();

//     return 0;
// }


#include <yaml-cpp/yaml.h>
#include "spdlog/spdlog.h"
#include "spdlog/stopwatch.h"

#include "yolo.h"
#include <chrono>
#include "tracker.h"

cv::Rect box;//矩形对象
bool drawing_box = false;//记录是否在画矩形对象
bool box_complete = false;
cv::Point userPt;
int GateSize = 32;
int minIdx = -1;

bool contain = false;

bool trackerInited = false;
realtracker *rtracker = nullptr;
cv::Mat trackFrame;
cv::Mat dispFrame, trackRet, detFrame, trackRetByDet;
int trackOn;
cv::Mat frame;

void onmouse(int event, int x, int y, int flag, void*)//鼠标事件回调函数，鼠标点击后执行的内容应在此
{
    // cv::Mat& image = *(cv::Mat*) img;
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN://鼠标左键按下事件
        drawing_box = true;//标志在画框
        box = cv::Rect(x, y, 0, 0);//记录矩形的开始的点
        userPt.x = x;    //center point
        userPt.y = y;
        break;
    case cv::EVENT_MOUSEMOVE://鼠标移动事件
        if (drawing_box) {//如果左键一直按着，则表明在画矩形
            box.width = x - box.x;
            box.height = y - box.y;//更新长宽
        }
        break;
    case cv::EVENT_LBUTTONUP://鼠标左键松开事件
    {
        //不在画矩形
                            //这里好像没作用
        if (box.width < 0) {//排除宽为负的情况，在这里判断是为了优化计算，不用再移动时每次更新都要计算长宽的绝对值
            box.x = box.x + box.width;//更新原点位置，使之始终符合左上角为原点
            box.width = -1 * box.width;//宽度取正
        }
        if (box.height < 0) {//同上
            box.y = box.y + box.height;
            box.height = -1 * box.width;
        }
        // g_nCount++;
        // cv::Mat dst = image(box);
        // std::string str_save_name = std::to_string(g_nCount) + ".jpg";
        // cv::imwrite(str_save_name.c_str(), dst);
        printf("mouse btn up event, x:%d,y:%d,w:%d,h:%d\n", box.x, box.y, box.width, box.height);
        drawing_box = true;
        box_complete = true;
        if (trackOn) {
            if (rtracker) {
                rtracker->reset();
                rtracker->init( userPt, frame );
                cv::rectangle(trackFrame, cv::Rect(userPt.x - 16, userPt.y - 16, 32,32),cv::Scalar( 48,48,255 ), 2, 8 );
                trackerInited = true;
                cv::imshow("trackRet", trackFrame);
            }
        }
        
    }
        
        break;
    default:
        break;
    }
}

cv::Mat resizeImage(const cv::Mat& inputImage, const cv::Size& targetSize) {
    cv::Mat resizedImage;
    cv::resize(inputImage, resizedImage, targetSize);
    return resizedImage;
}

void adjustContrastAndBrightness(const cv::Mat& input, cv::Mat& output, double alpha, int beta) {
    input.convertTo(output, -1, alpha, beta);
}

void applyGammaCorrection(const cv::Mat& input, cv::Mat& output, double gamma) {
    CV_Assert(gamma >= 0);
    cv::Mat lut(1, 256, CV_8UC1);
    uchar* p = lut.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);

    cv::LUT(input, lut, output);
}

void sharpenImage(const cv::Mat& input, cv::Mat& output) {
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
                      0, -1, 0,
                     -1, 5, -1,
                      0, -1, 0);

    cv::filter2D(input, output, input.depth(), kernel);
}

void brightestTo255(cv::Mat& image) {
    double minVal, maxVal;
    cv::minMaxLoc(image, &minVal, &maxVal);

    double scaleFactor = 255.0 / maxVal;

    image.convertTo(image, CV_8U, scaleFactor);
}

double calculateRedChannelMean(const cv::Mat& image) {
    std::vector<cv::Mat> channels;
    cv::split(image, channels); // Split the image into separate channels

    cv::Mat redChannel = channels[2]; // Red channel is at index 2 in BGR order

    int totalPixels = redChannel.rows * redChannel.cols;
    int sum = 0;

    for (int y = 0; y < redChannel.rows; y++) {
        for (int x = 0; x < redChannel.cols; x++) {
            sum += redChannel.at<uchar>(y, x);
        }
    }

    double mean = static_cast<double>(sum) / totalPixels;
    return mean;
}

void contrastStretching(cv::Mat& image) {
    double minVal, maxVal;
    cv::minMaxLoc(image, &minVal, &maxVal);

    // Linear stretching to [0, 255]
    image = (image - minVal) * (255.0 / (maxVal - minVal));
}

void enhanceBrightAreas(cv::Mat& image) {
    cv::Mat imageGray;
    cvtColor(image, imageGray, cv::COLOR_BGR2GRAY); // Convert image to grayscale

    double minVal, maxVal;
    cv::minMaxLoc(imageGray, &minVal, &maxVal);

    // Linear stretching to [0, 255] with bright areas set to 255
    imageGray = (imageGray - minVal) * (255.0 / (maxVal - minVal));

    // Set bright areas to 255
    // cv::threshold(imageGray, imageGray, 255, 255, cv::THRESH_BINARY);

    // Convert back to BGR if needed
    cv::cvtColor(imageGray, image, cv::COLOR_GRAY2BGR);
}

void preproc(cv::Mat frame)
{
    cv::Mat roi = frame(cv::Rect(220, 156, 200, 200));
    cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Mat blurredImage;
    cv::GaussianBlur(roi, blurredImage, cv::Size(3, 3), 0); 

    cv::Mat gradX, gradY;
    cv::Sobel(blurredImage, gradX, CV_16S, 1, 0, 3);
    cv::convertScaleAbs(gradX, gradX);
    cv::Sobel(blurredImage, gradY, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(gradY, gradY);

    addWeighted(gradX, 0.5, gradY, 0.5, 0, edges); 

    // cv::Mat dstImage;
    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); 
    // // cv::morphologyEx(edges, dstImage, cv::MORPH_OPEN, kernel);
    // // cv::dilate(edges, dstImage, kernel);
    // // cv::erode(edges, dstImage, kernel);
    // cv::morphologyEx(edges, dstImage, cv::MORPH_CLOSE, kernel);

    // cv::cvtColor(dstImage, dstImage, cv::COLOR_GRAY2BGR);
    // dstImage.copyTo(frame(cv::Rect(220, 156, 200, 200)));


    // cv::threshold(edges, edges, 20, 255, cv::THRESH_BINARY);
    // cv::Mat enhancedImage;
    // cv::equalizeHist(edges, enhancedImage);
    // cv::cvtColor(enhancedImage, enhancedImage, cv::COLOR_GRAY2BGR);
    
    cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
    edges.copyTo(frame(cv::Rect(220, 156, 200, 200)));

}


int main(int argc, char*argv[])
{
    // std::cout << cv::getBuildInformation() << std::endl; 

    // // cv::Ptr<cv::SURF> surf = cv::SURF::create();
    // cv::Ptr<cv::SIFT> detector = cv::SIFT::create(20);
    // // vector<KeyPoint> keypoints, keypoints2;
    // // detector->detect(src, keypoints);

    // return 0;
    // cv::Mat src_1 = cv::imread("/home/nx/code/robusttracker/v0/build/detrect.png");
    // // cv::Mat src_2 = cv::imread("/home/nx/code/robusttracker/v0/build/patch.png");
    // cv::Mat src_2 = cv::imread("/home/nx/code/robusttracker/v0/build/initdet.png");

    // cv::Size targetSize(std::min(src_1.cols, src_2.cols), std::min(src_1.rows, src_2.rows));
    // src_1 = resizeImage(src_1, targetSize);
    // src_2 = resizeImage(src_2, targetSize);

    // // cv::Scalar re = CalcMSSIM(src_1,src_2);
    // double ssim = calculateSSIM(src_1, src_2);
    // // std::cout<<"the r g b channles similarity is :"<<re<<std::endl;
    // std::cout<<"the r g b channles similarity is :"<<ssim<<std::endl;

    // return 0;


    int waitVAL =  argc > 1 ? 0 : 10;
    // spdlog::stopwatch sw;    
    // spdlog::info("Welcome to spdlog!");
    // spdlog::error("Some error message with arg: {}", 1);
    
    // spdlog::warn("Easy padding in numbers like {:08d}", 12);
    // spdlog::critical("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42);
    // spdlog::info("Support for floats {:03.2f}", 1.23456);
    // spdlog::info("Positional args are {1} {0}..", "too", "supported");
    // spdlog::info("{:<30}", "left aligned");
    
    spdlog::set_level(spdlog::level::debug); // Set global log level to debug
    // spdlog::debug("This message should be displayed..");    
    
    // // change log pattern
    // spdlog::set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [thread %t] %v");
    
    // // Compile time log levels
    // // define SPDLOG_ACTIVE_LEVEL to desired level
    // SPDLOG_TRACE("Some trace message with param {}", 42);
    // SPDLOG_DEBUG("Some debug message");

    
    // spdlog::debug("Elapsed {}", sw);
    // spdlog::debug("Elapsed {:.3}", sw);       

    // return 0;

    YAML::Node config = YAML::LoadFile("/root/yolov5/demo/config.yaml");
    int detOn = config["detection"].as<int>();
    trackOn = config["track"].as<int>();
    std::string engine = config["engine"].as<std::string>();
    std::string videopath = config["videopath"].as<std::string>();
    std::string irEngine = config["irengine"].as<std::string>();

    std::queue<cv::Mat> frameQueue;



    cv::VideoCapture cap(videopath);
    if(!cap.isOpened())
    {
        printf("open failed\n");
        return 0;
    }

    rtracker = new realtracker();


    
    int nFrames = 0;

    float xMin;
    float yMin;
    float width;
    float height;

    cv::namedWindow("trackRet");
    cv::setMouseCallback("trackRet", onmouse);

    // Tracker results
    cv::Rect result;
    cv::Point pt;

    
    // std::vector<TrackingObject> detRet;

    uint8_t trackerStatus[9];
    memset(trackerStatus, 0, 9);

    cv::Mat templt;

    // rtracker->setIrFrame(true);

    rtracker->setGateSize(20);

    uint32_t g_modelWidth = 640;
    uint32_t g_modelHeight = 640;
    std::string imgname = "/root/code/ascend-c-yolov5/yolov5/IR.jpg";
    // const char* g_modelPath = "/root/code/ascend-c-yolov5/ir_best_no.onnx.om";
    // const char* g_modelPath = "/root/yolov5/det.om";
    // const char* g_modelPath = "/root/hms.om";
    const char* g_modelPath = "/root/irbestno.om";

    yolo * detect = new yolo(engine.c_str(), g_modelWidth, g_modelHeight);
    Result ret = detect->Init();

    vector<BBox> output;

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();

    while(1)
    {
        // cap >> frame;
        // if(frame.empty())
        //     break;

        frame = cv::imread("/root/yolov5/vlc-record-2025-01-21-18h21m18s-rtsp___192.168.3.61_555_live-.mp4_20250121_184214.167.jpg");
        // cv::resize(frame, frame, cv::Size(640,512));
        

        trackFrame = frame.clone();
        detFrame = frame.clone();
        dispFrame = frame.clone();
        trackRetByDet = frame.clone();

        printf("=====nframe:%d======\n", nFrames);

        output.clear();
        // cv::resize(frame, detFrame, cv::Size(640,640));
        ret = detect->Preprocess(detFrame);
        detect->Inference();
        output = detect->Postprocess();

        printf("frame channel:%d\n", frame.channels());

        // cv::threshold(frame, frame, 120, 255, cv::THRESH_BINARY);

        

        cv::Mat roi = frame(cv::Rect(220, 156, 200, 200));

        

        // double m = calculateRedChannelMean(roi);

        // printf("frame m:%f\n", m);

        // cv::threshold(roi, roi, m, 255, cv::THRESH_BINARY);

        
        cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);
        // // Set the clip limit and grid size for CLAHE
        // clahe->setClipLimit(6);
        // clahe->setTilesGridSize(cv::Size(32, 32));
        
        // // Apply CLAHE to enhance the image
        // cv::Mat enhancedImage;
        // clahe->apply(roi, roi);

        // cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

        // cv::Mat edges;
        // cv::Mat blurredImage;
        // cv::GaussianBlur(roi, blurredImage, cv::Size(3, 3), 0); // 先对图像进行高斯模糊处理，以减少噪声

        // cv::Mat gradX, gradY;
        // cv::Sobel(blurredImage, gradX, CV_16S, 1, 0);
        // cv::convertScaleAbs(gradX, gradX);
        // cv::Sobel(blurredImage, gradY, CV_16S, 0, 1);
        // cv::convertScaleAbs(gradY, gradY);

        // addWeighted(gradX, 0.5, gradY, 0.5, 0, edges); // 将水平和垂直方向的边缘强度叠加
        // cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
        // edges.copyTo(frame(cv::Rect(220, 156, 200, 200)));

        // preproc(frame);


        // cv::imshow("edges", edges);


        int center_x,center_y;
        if(trackOn) {
            if (trackerInited) {
                // rtracker->update(trackFrame, detRet, trackerStatus);
                // rtracker->update(frame, detRet, trackerStatus);
                auto rect = rtracker->update(frame, output);
                cv::rectangle(frame, rect, cv::Scalar(255,255,255), 2);
            }
            cv::imshow("trackRet", frame);
        }
        nFrames++;

        if(detOn)
        {
            // rtracker->runDetector(detFrame, detRet);
            // cv::imshow("show", trackRetByDet);

            // cv::resize(dispFrame, dispFrame, cv::Size(1280,720));
            detect->DrawBoundBoxToImage(output, dispFrame);
            cv::imshow("final-detRet", dispFrame);
        }


        // cv::resize(dispFrame, dispFrame, cv::Size(1280,720));
        // cv::resize(trackFrame, trackFrame, cv::Size(1280,720));
        // cv::imshow("show", dispFrame);
        
        
        char c = cv::waitKey(waitVAL);
        if(c == 'g')
            waitVAL = 30;
        else if(c == 's')
            waitVAL = 0;

    }

    return 0;
}