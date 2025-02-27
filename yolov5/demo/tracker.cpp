#include "tracker.h"
#include <unistd.h>

#define TRACKER_DEBUG 0

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(-1, 1);


static double calculateSSIM(const cv::Mat& imgg1, const cv::Mat& imgg2)
{
    // printf("realtracker::calculateSSIM start\n");
    // std::cout<<"img1 :"<<img1.size()<<std::endl;
    // std::cout<<"img2 :"<<img2.size()<<std::endl;
    cv::Mat img1, img2;

    cv::Size targetSize(std::min(imgg1.cols, imgg2.cols), std::min(imgg1.rows, imgg2.rows));
    cv::resize(imgg1, img1, targetSize);
    cv::resize(imgg2, img2, targetSize);

    // printf("realtracker::calculateSSIM\n");

    // 分离通道
    std::vector<cv::Mat> channels1, channels2;
    cv::split(img1, channels1);
    cv::split(img2, channels2);

    double ssim = 0.0;

    // 计算每个通道的SSIM
    for (int i = 0; i < 3; ++i) {
        // 转换为double类型
        channels1[i].convertTo(channels1[i], CV_64F);
        channels2[i].convertTo(channels2[i], CV_64F);

        // 计算均值
        double mean1 = cv::mean(channels1[i])[0];
        double mean2 = cv::mean(channels2[i])[0];

        // 计算方差
        cv::Mat var1, var2;
        cv::multiply(channels1[i] - mean1, channels1[i] - mean1, var1);
        cv::multiply(channels2[i] - mean2, channels2[i] - mean2, var2);
        double var1_scalar = cv::mean(var1)[0];
        double var2_scalar = cv::mean(var2)[0];

        // 计算协方差
        cv::Mat covar;
        cv::multiply(channels1[i] - mean1, channels2[i] - mean2, covar);
        double covar_scalar = cv::mean(covar)[0];

        // 计算SSIM
        double c1 = 0.01 * 255 * 0.01 * 255;
        double c2 = 0.03 * 255 * 0.03 * 255;
        double channel_ssim = (2 * mean1 * mean2 + c1) * (2 * covar_scalar + c2) / ((mean1 * mean1 + mean2 * mean2 + c1) * (var1_scalar + var2_scalar + c2));

        // 累加每个通道的SSIM
        ssim += channel_ssim;
    }

    // 求取均值
    ssim /= 3.0;

    return ssim;
}

double calAngle(const cv::Point &obj, const cv::Point &det, const cv::Point &velo)
{
    std::cout<<"velo:"<<velo<<std::endl;
    std::cout<<"obj:"<<obj<<std::endl;
    std::cout<<"det:"<<det<<std::endl;
    cv::Point v1{det.x - obj.x, det.y - obj.y};
    cv::Point v2{velo.x - obj.x, velo.y - obj.y};
    double dot = v1.x * v2.x + v1.y * v2.y;
    double v1Len = sqrt(v1.x * v1.x + v1.y * v1.y);
    double v2Len = sqrt(v2.x * v2.x + v2.y * v2.y);

    printf("dot:%f, v1 len:%f, v2 len:%f\n", dot, v1Len, v2Len);

    return acos(dot / (v1Len*v2Len)) * 180.0 / M_PI;
}

inline static int newCeil(float i)
{
    return i >= 0 ? (int)ceil(i) : -(int)ceil(-i);
}

veloestimater::veloestimater():offset(60)
{
    criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
}

void veloestimater::init(cv::Mat frame)
{
#if TRACKER_DEBUG
    printf("veloestimater::init\n");
#endif
    old_frame = frame(cv::Rect(0,0,offset,offset));
    cv::cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);
    cv::goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
    // m_veloBuf.clear();
}

void veloestimater::calcof(cv::Mat frame)
{
    frame = frame(cv::Rect(0,0,offset,offset));
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, cv::Size(15,15), 2, criteria);
    std::vector<cv::Point2f> good_new;

    for(uint i = 0; i < p0.size(); i++)
    {
        // Select good points
        if(status[i] == 1) {
            good_new.push_back(p1[i]);
            if(m_veloBuf.size() > 20)
                m_veloBuf.pop_front();
            m_veloBuf.emplace_back(std::pair<float,float>{p1[i].x - p0[i].x, p1[i].y - p0[i].y});
        }
    }

    // printf("good_new size:%d\n", good_new.size());

    p0 = good_new;
    old_gray = frame_gray.clone();
    calcVelo();

#if TRACKER_DEBUG
    float sumx, sumy,x,y;
    sumx = sumy = 0.0;
    for(auto& velo:m_veloBuf)
    {
        sumx += velo.first;
        sumy += velo.second;
    }
    x = sumx/m_veloBuf.size();
    y = sumy/m_veloBuf.size();

    // printf("x:%f, y:%f\n",x,y);

    cv::Point p = cv::Point(50,50);
    cv::Point pee;

    pee.x = p.x - x*10;
    pee.y = p.y - y*10;


    cv::line(frame,p, pee, cv::Scalar(200,30,20), 2);
    // cv::imshow("123",frame);
#endif

    if(good_new.size() == 0)
    {
        init(frame);
    }
}

void veloestimater::calcVelo()
{
    float sumx, sumy,x,y;
    sumx = sumy = 0.0;
    for(auto& velo:m_veloBuf)
    {
        sumx += velo.first;
        sumy += velo.second;
        // printf("x:%f, y:%f\n",velo.first,velo.second);
    }
    m_bgvelo.first = sumx/m_veloBuf.size();
    m_bgvelo.second = sumy/m_veloBuf.size();
#if TRACKER_DEBUG
    printf("size:%d, bg x:%f, bg y:%f\n", m_veloBuf.size(), m_bgvelo.first, m_bgvelo.second);
#endif
}

std::pair<float,float> veloestimater::getVelo()
{
    return m_bgvelo;
}

static int Clamp(int& v, int& size, int hi)
{
    int res = 0;
    if (v < 0)
    {
        res = v;
        v = 0;
        return res;
    }
    else if (v + size > hi - 1)
    {
        res = v;
        v = hi - 1 - size;
        if (v < 0)
        {
            size += v;
            v = 0;
        }
        res -= v;
        return res;
    }
    return res;
};

void trackObj::init(const cv::Rect &box, cv::Mat frame)
{
    m_rect = box;
    m_age = 1;
    m_lostCnt = 0;
    m_trace.clear();
    m_trace.emplace_back(cv::Point(box.x+box.width/2, box.y + box.height/2));

    m_lastPos = m_rect;

    m_wdsz.first = m_rect.width;
    m_wdsz.second = m_rect.height;
    m_velo[0] = m_velo[1] = 0.f;

    m_veloBuf.clear();

    rx = m_rect.x;
    ry = m_rect.y;

    std::cout<<" trackObj::init:::"<<m_rect<<std::endl;

}

cv::Point trackObj::center()
{
    return cv::Point(m_rect.x+m_rect.width/2, m_rect.y + m_rect.height/2);
}

inline void trackObj::calcVelo()
{
    // float sumx, sumy;
    // sumx = sumy = 0.0;
    // for(auto& velo:m_veloBuf)
    // {
    //     sumx += velo.first;
    //     sumy += velo.second;
    //     // printf("x:%d, y:%d\n",velo.first,velo.second);
    // }
    // m_velo[0] = sumx/m_veloBuf.size();
    // m_velo[1] = sumy/m_veloBuf.size();

    m_velo[0] = static_cast<float>(this->center().x - m_trace.front().x)/m_trace.size();
    m_velo[1] = static_cast<float>(this->center().y - m_trace.front().y)/m_trace.size();
#if TRACKER_DEBUG
    printf("this->center().x:%d, m_trace.front().x:%d, size:%d\n", this->center().x, m_trace.front().x, m_trace.size());
    printf("trackObj::calcVelo vx:%f, vy:%f\n", m_velo[0], m_velo[1]);
#endif
}

void trackObj::update(cv::Mat img, const cv::Rect &box, double ssim)
{
    printf("trackObj::update:\n");
    m_rect = box;

    rx = m_rect.x;
    ry = m_rect.y;
    // int sizeDif = abs(m_rect.width - m_wdsz.first) + abs(m_rect.height - m_wdsz.second);
    int sizeDif = abs(m_rect.width - m_lastPos.width) + abs(m_rect.height - m_lastPos.height);
#if TRACKER_DEBUG
    std::cout<<"curPos:"<<m_rect<<std::endl;
    std::cout<<"m_lastPos:"<<m_lastPos<<std::endl;
    // printf("w diff:%d, h diff:%d\n", abs(m_rect.width - m_wdsz.first), abs(m_rect.height - m_wdsz.second));
    printf("w diff:%d, h diff:%d\n", abs(m_rect.width - m_lastPos.width), abs(m_rect.height - m_lastPos.height));
    printf("ave w:%d, ave h:%d\n", m_wdsz.first, m_wdsz.second);
#endif
    if(m_trace.size() > 100)
        m_trace.pop_front();
    m_trace.emplace_back(cv::Point(box.x+box.width/2, box.y + box.height/2));
    m_age++;

    // if(m_lostCnt == 0 && sizeDif < 6 && ssim > 0.7)
    // {
    //     if(m_veloBuf.size() > 40)
    //         m_veloBuf.pop_front();
    //     m_veloBuf.emplace_back(std::pair<int,int>{box.x - m_lastPos.x, box.y - m_lastPos.y});

    //     // calcVelo();

    //     m_wdsz.first = 0.5 * m_wdsz.first + 0.5 * m_rect.width;
    //     m_wdsz.second = 0.5 * m_wdsz.second + 0.5 * m_rect.height;
    // }

    if(m_velomod == 0 && m_lostCnt == 0 && sizeDif < 6 && ssim > 0.7)
    {

        // if(m_veloBuf.size() > 40)
        //     m_veloBuf.pop_front();
        // m_veloBuf.emplace_back(std::pair<int,int>{box.x - m_lastPos.x, box.y - m_lastPos.y});

        calcVelo();

    }

#if TRACKER_DEBUG
    if(!m_veloBuf.empty())
        printf("inst velo x:%d, inst velo y:%d\n", box.x - m_lastPos.x, box.y - m_lastPos.y);
    // std::cout<<"patch size:"<<m_patch.size()<<std::endl;
    // cv::imwrite("patch.png", m_patch);

    cv::Point st{m_rect.x+ m_rect.width/2, m_rect.y+m_rect.height/2};
    cv::Point en{st.x+m_velo[0]*10, st.y+m_velo[1]*10};
    cv::line(img,st, en, cv::Scalar(0,30,200), 2);

#endif

    m_lostCnt = 0;
    m_lastPos = m_rect;

    Clamp(m_rect.x, m_rect.width, img.cols);
    Clamp(m_rect.y, m_rect.height, img.rows);
    // if(ssim > 0.7)
    //     m_patch = img(m_rect).clone();


    // m_hist = CalcHist(img, box);
}

void trackObj::updateWithoutDet(std::pair<float,float> bgvelo)
{

    rx  += m_velo[0] * 0.4;
    ry  += m_velo[1] * 0.4;
    m_rect.x  = rx;
    m_rect.y  = ry;

#if TRACKER_DEBUG
    printf("updateWithoutDet vx:%f, vy:%f\n", m_velo[0], m_velo[1]);
#endif

    // if(m_velo[1] < -1.2)
    // {
    //     m_rect.x += (int)round(m_velo[0] + bgvelo.first*0.6);
    //     m_rect.y += (int)round(m_velo[1] + bgvelo.second*0.6);
    // }
    // else
    // {
    //     m_rect.x += (int)round(m_velo[0]);
    //     m_rect.y += (int)round(m_velo[1]);
    // }
    // m_rect.x += (int)round(m_velo[0]);
    // m_rect.y += (int)round(m_velo[1]);
    // m_rect.x += (int)newCeil(m_velo[0] + bgvelo.first);
    // m_rect.y += (int)newCeil(m_velo[1] + bgvelo.second);
    m_lostCnt++;

    // printf("finale velo x:%d, y:%d\n", (int)round(m_velo[0]),(int)round(m_velo[1]));
    // printf("finale velo x:%d, y:%d\n", (int)round(m_velo[0]),(int)round(m_velo[1]));

}

void trackObj::predict()
{
    m_rect.x += (int)round(m_velo[0]);
    m_rect.y += (int)round(m_velo[1]);
}

bool trackObj::isLost()
{
    if(m_trace.size() < 5)
    {
        return m_lostCnt > 1;
    }
    else if(m_trace.size() < 10)
    {
        return m_lostCnt > 3;
    }
    else if(m_trace.size() < 25)
    {
        return m_lostCnt > 5;
    }
    else
    {
        return m_lostCnt > 25;
    }
}

void trackObj::fixVelo(std::pair<float,float> bgvelo)
{
    // m_velo[0] -= bgvelo.first;
    // m_velo[1] -= bgvelo.second;
    // m_velo[0] = -bgvelo.first;
    // m_velo[1] = -bgvelo.second;
    // m_velo[0] = 0;
    // m_velo[1] = 0;

    m_velo[0] = -bgvelo.first;
    m_velo[1] = -bgvelo.second;

    // if(abs(bgvelo.first) < 0.5 && abs(bgvelo.second) < 0.5)
    // {
    //     m_velomod = 0;

    // }
    // else
    // {
    //     m_velomod = 1;
    //     // m_velo[0] = -bgvelo.first;
    //     // m_velo[1] = -bgvelo.second;
    //     m_velo[0] = m_velo[1] = 0;
    // }
// #if TRACKER_DEBUG
//     printf("m_velomod:%d\n", m_velomod);
// #endif
}

realtracker::realtracker():m_serDis(20.f),m_gateS(32),m_serSiThr(0.6)
{
    m_stracker = new itracker();
    m_stracker->setGateSize(m_gateS);
    // m_smlv = false;
    // if(m_gateS == 10)
    // {
    //     m_smlv = true;
    // }

    m_static = 0;

}

realtracker::~realtracker()
{

}

void realtracker::init(const cv::Point &pt, cv::Mat image)
{
    m_stracker->init(pt, image);

    cv::Rect initBox;
    initBox.x = pt.x - m_stracker->m_GateSize/2;
    initBox.y = pt.y - m_stracker->m_GateSize/2;
    initBox.width = initBox.height = m_stracker->m_GateSize;

    m_trackObj.init(initBox, image);

    m_state = EN_TRACKER_FSM::STRACK;

    m_strackerfailedCnt = 0;
    m_ssearchCnt = 1;

    m_finalRect.width = m_finalRect.height = m_gateS;

    m_estimator.init(image);
    m_initdet = false;
    m_strackerslpcnt = 0;
    m_detDisThres = 30.f;
    m_static = 0;
}

void realtracker::reset()
{
    m_stracker->reset();
}

cv::Rect realtracker::update(cv::Mat &frame, std::vector<BBox> &boxs)
{
    static int nframe = 0;
    printf("realtracker nframe:[%d]\n", nframe++);
    fsmUpdate(frame, boxs);

    static cv::Rect lastfinrect;

    try
    {
        m_estimator.calcof(frame);
    }
    catch(cv::Exception& e)
    {
        printf("m_estimator.calcof catch\n");
        m_estimator.init(frame);
        // std::cerr << msg << std::endl;
    }

    
    static int rancnt;

    double dist = cv::norm(lastfinrect.tl() - m_finalRect.tl());
    printf("dist:%f, ms:%d\n", dist, m_static);
    if(m_static > 20 && dist > 10)
    {
        
        if(rancnt++ % 30 == 0)
        {
            lastfinrect.x += dis(gen);
            lastfinrect.y += dis(gen);
        }
        
        m_finalRect = lastfinrect;
        m_trackObj.m_rect = lastfinrect;
        m_stracker->setRoi(m_trackObj.m_rect);
    }
    else
    {
        rancnt = 0;
    }
    lastfinrect = m_finalRect;
    std::cout<<"m_finalRect:"<<m_finalRect<<std::endl;
    return m_finalRect;
    // return m_showRect;

}

bool realtracker::isLost()
{
    return m_state == EN_TRACKER_FSM::LOST;
}

void realtracker::FSM_PROC_SEARCH(cv::Mat &frame)
{
    printf("\nSM_PROC_SEARCH\n");
    static int searchFrameCnt = 0;
    searchFrameCnt++;
    if (searchFrameCnt > 3)
    {
        m_state = EN_TRACKER_FSM::LOST;
        searchFrameCnt = 0;
        m_stracker->reset();
    }
}

int intersectionArea(cv::Rect r1, cv::Rect r2)
{
    cv::Rect intersection = r1 & r2;
    return intersection.area();
}

inline double getDistance(cv::Point point1, cv::Point point2)
{
    return sqrtf(powf((point1.x - point2.x), 2) + powf((point1.y - point2.y), 2));
}


void realtracker::FSM_PROC_STRACK(cv::Mat &frame, std::vector<BBox> &boxs)
{
    printf("\nFSM_PROC_STRACK\n");
    // printf("%d,%d, %d\n", m_stracker->m_oriPatch.cols, m_stracker->m_oriPatch.rows, m_stracker->m_oriPatch.channels());


    // double minVal, maxVal;
    // cv::Point minLoc, maxLoc;
    // cv::Mat seAre = frame(cv::Rect(m_trackObj.m_rect.x - 100, m_trackObj.m_rect.y - 100, 200, 200)).clone();
    // cv::cvtColor(seAre, seAre, cv::COLOR_BGR2GRAY);
    // cv::minMaxLoc(seAre, &minVal, &maxVal, &minLoc, &maxLoc);

    // // 在最亮的点画一个圆圈
    // cv::cvtColor(seAre, seAre, cv::COLOR_GRAY2BGR);
    // cv::circle(seAre, maxLoc, 5, cv::Scalar(0, 0, 255), 2);

    // cv::imshow("f", seAre);
    // cv::waitKey(1);


    static int dethiconf = 0;

    cv::Rect kcfResult;
    // int64_t currTime = Utils::GetTimeStamp();
    m_showRect = kcfResult = m_stracker->update(frame);
    // std::cout << "m_stracker->update time:"<<Utils::GetTimeStamp() - currTime<<std::endl;
#if TRACKER_DEBUG
    std::cout<<"s tracker:"<<kcfResult<<std::endl;
#endif

    // auto tpr = m_stracker->updateTP(frame);

    // rectangle(frame,tpr,cv::Scalar(40,132,156),2);


    auto cmpDist = [this](BBox box1, BBox box2)
    {
        cv::Point center1{box1.x + box1.w / 2, box1.y + box1.h / 2};
        double dist1 = getDistance(m_trackObj.center(), center1);
        cv::Point center2{box2.x + box2.w / 2, box2.y + box2.h / 2};
        double dist2 = getDistance(m_trackObj.center(), center2);
        if (box1.classIndex != 1)
            dist1 = 1000;
        if (box2.classIndex != 1)
            dist2 = 1000;

        return dist1 < dist2;
    };

    double minDist = 1000.f;
    cv::Rect r = kcfResult;

#if TRACKER_DEBUG
    printf("m_serDis:%f\n", m_serDis);
#endif

    // for(auto &box:boxs)
    // {
    //     cv::Point p{box.x+box.w/2, box.y+box.h/2};
    //     cv::Point o{m_trackObj.m_rect.x+m_trackObj.m_rect.width/2, m_trackObj.m_rect.y+m_trackObj.m_rect.height/2};
    //     cv::Point v{o.x + m_trackObj.m_velo[0]*20, o.y + m_trackObj.m_velo[1]*20};
    //     double angle = calAngle(o, p, v);

    //     cv::line(frame,p, o, cv::Scalar(100,30,200), 2);
    //     cv::putText(frame, std::to_string(angle), p, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 1);
    // }


    // if(m_trackObj.m_velomod == 0 && r.width < 20)
    if(m_trackObj.m_velomod == 1)
    {
        minDist = 1000.f;
        // m_detDisThres = 5.0f;
    }

    // if(m_smlv == true)
    // {
    //     m_stracker->isLost() = false;
    //     minDist = 1000.f;
    // }



    if (m_stracker->isLost())
    // if (m_stracker->isLost())
    {
        m_finalRect = m_trackObj.m_rect;
        m_strackerfailedCnt++;
#if 1
        printf("STRACK lost m_strackerfailedCnt:%d\n", m_strackerfailedCnt);
#endif
        if(m_strackerfailedCnt < 6)
        {
            m_trackObj.updateWithoutDet(m_estimator.getVelo());
            std::cout<<m_trackObj.m_rect<<std::endl;
            m_state = EN_TRACKER_FSM::SSEARCH;
        }
        else
            m_state = EN_TRACKER_FSM::SEARCH;
    }
    else
    {
        m_finalRect = kcfResult;
        if(!boxs.empty())
        {
            std::sort(boxs.begin(), boxs.end(), cmpDist);

            cv::Point center{boxs.front().x + boxs.front().w / 2, boxs.front().y + boxs.front().h / 2};
            minDist = getDistance(m_trackObj.center(), center);

            r = cv::Rect(boxs.front().x,boxs.front().y,boxs.front().w, boxs.front().h);
    #if TRACKER_DEBUG
            std::cout<<"d tracker:"<<r<<std::endl;
            printf("minDist:%f\n", minDist);
    #endif

            // if(r.width > 35 && minDist < 60)
            // {
            //     m_serDis = 40.f;
            //     m_serSiThr = 0.7;
            // }

            if(minDist < 6.f)
            {
                m_finalRect = r;
                dethiconf = 0;

            }
            else if(minDist < 25.f)
            {
                dethiconf++;
            }
            else
            {
                dethiconf = 0;
            }

            if(dethiconf == 10)
            {
                m_stracker->reset();
                m_stracker->init(r, frame);
                m_trackObj.init(r, frame);
                dethiconf = 0;
            }


        }

        // m_finalRect.x = kcfResult.x;
        // m_finalRect.y = kcfResult.y;

        m_state = EN_TRACKER_FSM::STRACK;
        // std::cout<<m_finalRect<<std::endl;
        // m_trackObj.update(frame, m_finalRect, 0.5);
        m_trackObj.update(frame, m_finalRect, 0.8);
        m_trackObj.fixVelo(m_estimator.getVelo());
    }
#if TRACKER_DEBUG
    cv::rectangle(frame, r, cv::Scalar(0,255,255), 2);  //yellow
    cv::rectangle(frame, kcfResult, cv::Scalar(255,0,255), 2);  //purple
#endif
    printf("m_trackObj age:%d, lostcnt:%d, trace size:%d, velo x:%f, velo y:%f\n",
           m_trackObj.m_age, m_trackObj.m_lostCnt, m_trackObj.m_trace.size(), m_trackObj.m_velo[0], m_trackObj.m_velo[1]);

    if(abs(m_trackObj.m_velo[0]) < 0.1f && abs(m_trackObj.m_velo[1]) < 0.1f)
    {
        m_static++;
    }
    else
    {
        m_static = 0;
    }

    // std::cout<<"m_showRect:"<<m_showRect<<std::endl;
    // std::cout<<"m_finalRect:"<<m_finalRect<<std::endl;

    return;

}

void realtracker::FSM_PROC_SSEARCH(cv::Mat &frame, std::vector<BBox> &boxs)
{
    printf("\nFSM_PROC_SSEARCH, boxs:%d\n", boxs.size());

    printf("%d,%d, %d\n", m_stracker->m_oriPatch.cols, m_stracker->m_oriPatch.rows, m_stracker->m_oriPatch.channels());

#if TRACKER_DEBUG

    cv::Point st{m_trackObj.m_rect.x+ m_trackObj.m_rect.width/2, m_trackObj.m_rect.y+m_trackObj.m_rect.height/2};
    cv::Point en{st.x+m_trackObj.m_velo[0]*30, st.y+m_trackObj.m_velo[1]*30};
    // cv::line(frame,st, en, cv::Scalar(0,30,200), 2);

#endif

    if (m_ssearchCnt++ < 500)
    {
        m_trackObj.updateWithoutDet(m_estimator.getVelo());
        std::cout << m_trackObj.m_rect << std::endl;
        usleep(8000);
        m_stracker->setRoi(m_trackObj.m_rect);
        m_finalRect = m_trackObj.m_rect;
    }
    else
    {
        // spdlog::warn("ssearchCntThres met exit");
        m_state = EN_TRACKER_FSM::SEARCH;
        m_finalRect = m_trackObj.m_rect;
        return;
    }

    auto cmpDist = [this](BBox box1, BBox box2)
    {
        cv::Point center1{box1.x + box1.w / 2, box1.y + box1.h / 2};
        double dist1 = getDistance(m_trackObj.center(), center1);
        cv::Point center2{box2.x + box2.w / 2, box2.y + box2.h / 2};
        double dist2 = getDistance(m_trackObj.center(), center2);
        if (box1.classIndex != 1)
            dist1 = 1000;
        if (box2.classIndex != 1)
            dist2 = 1000;

        return dist1 < dist2;
    };

    double minDist = 1000.f;
    cv::Rect r = m_trackObj.m_rect;

    auto ssi = 0.f;

    if(!boxs.empty())
    {
        std::sort(boxs.begin(), boxs.end(), cmpDist);

        cv::Point center{boxs.front().x + boxs.front().w / 2, boxs.front().y + boxs.front().h / 2};
        minDist = getDistance(m_trackObj.center(), center);

        r = cv::Rect(boxs.front().x,boxs.front().y,boxs.front().w, boxs.front().h);

        if(r.x < 0)
            r.x = 0;
        if(r.x + r.width > frame.cols)
            r.x = frame.cols - r.width - 1;
        if(r.y < 0)
            r.y = 0;
        if(r.y + r.height > frame.rows)
            r.y = frame.rows - r.height - 1;


        ssi = calculateSSIM(m_stracker->m_oriPatch, frame(r));

    }

#if TRACKER_DEBUG
        printf("minDist:%f, ssi:%f\n", minDist, ssi);
#endif

    if(minDist < m_serDis && minDist > 6.f && ssi > 0)
    // if(minDist < m_serDis && minDist > 6.f)
    {
        //find, to strack
        m_state = EN_TRACKER_FSM::STRACK;
        m_finalRect = r;
        m_stracker->isLost() = false;
        m_stracker->reset();
        m_stracker->init(r, frame);
        m_trackObj.init(r, frame);
        m_ssearchCnt = 1;
        m_initdet = false;
        // if(minDist < m_serDis)
        // {
        //     m_stracker->reset();
        //     m_stracker->init(r, frame);
        // }
        m_strackerfailedCnt--;
        return;
    }


    if(m_ssearchCnt % 10 == 0)
    {
        double sim = 0.f;
        auto rect = m_stracker->find(frame, sim);
        printf("nSSEA sim:%f\n", sim);

        // if(sim > 0.6)
        if(sseFind(sim))
        {
            //find, to strack
            m_state = EN_TRACKER_FSM::STRACK;
            m_finalRect = rect;
            m_stracker->isLost() = false;
            m_stracker->reset();
            m_stracker->init(rect, frame);
            m_trackObj.init(rect, frame);
            m_ssearchCnt = 1;
            m_strackerfailedCnt--;
            return;
        }

#if TRACKER_DEBUG
        cv::rectangle(frame, rect, cv::Scalar(123,30,56), 2);
#endif

        m_finalRect = m_trackObj.m_rect;
        m_state = EN_TRACKER_FSM::SSEARCH;
        m_ssearchCnt++;

    }

    return;

//     if(m_ssearchCnt > 35)
//     {
//         // printf("find failed\n");
//         m_ssearchCnt = 1;
//         m_state = EN_TRACKER_FSM::STRACK;
//         return;
//     }

//     m_trackObj.updateWithoutDet(m_estimator.getVelo());
//     std::cout<<m_trackObj.m_rect<<std::endl;

//     auto cmpDist = [this](BBox box1, BBox box2)
//     {
//         cv::Point center1{box1.x + box1.w / 2, box1.y + box1.h / 2};
//         double dist1 = getDistance(m_trackObj.center(), center1);
//         cv::Point center2{box2.x + box2.w / 2, box2.y + box2.h / 2};
//         double dist2 = getDistance(m_trackObj.center(), center2);
//         if (box1.classIndex != 1)
//             dist1 = 1000;
//         if (box2.classIndex != 1)
//             dist2 = 1000;

//         return dist1 < dist2;
//     };

//     double minDist = 1000.f;
//     cv::Rect r = m_trackObj.m_rect;

//     auto ssi = 0.f;

//     if(!boxs.empty())
//     {
//         std::sort(boxs.begin(), boxs.end(), cmpDist);

//         cv::Point center{boxs.front().x + boxs.front().w / 2, boxs.front().y + boxs.front().h / 2};
//         minDist = getDistance(m_trackObj.center(), center);

//         r = cv::Rect(boxs.front().x,boxs.front().y,boxs.front().w, boxs.front().h);

//         if(r.x < 0)
//             r.x = 0;
//         if(r.x + r.width > frame.cols)
//             r.x = frame.cols - r.width - 1;
//         if(r.y < 0)
//             r.y = 0;
//         if(r.y + r.height > frame.rows)
//             r.y = frame.rows - r.height - 1;


//         ssi = calculateSSIM(m_stracker->m_oriPatch, frame(r));

//     }



//     // std::cout<<frame<<std::endl;
//     // std::cout<<m_trackObj.m_rect<<std::endl;

//     if(m_trackObj.m_rect.x < 0)
//         m_trackObj.m_rect.x = 0;
//     if(m_trackObj.m_rect.x + m_trackObj.m_rect.width > frame.cols)
//         m_trackObj.m_rect.x = frame.cols - m_trackObj.m_rect.width - 1;
//     if(m_trackObj.m_rect.y < 0)
//         m_trackObj.m_rect.y = 0;
//     if(m_trackObj.m_rect.y + m_trackObj.m_rect.height > frame.rows)
//         m_trackObj.m_rect.y = frame.rows - m_trackObj.m_rect.height - 1;

//     m_stracker->setRoi(m_trackObj.m_rect);



//     printf("SSEARCH minDist:%f, m_serDis:%f, ssi:%f\n", minDist, m_serDis, ssi);

//     if(minDist < m_serDis && minDist > 6.f && ssi > 0.3)
//     {
//         //find, to strack
//         m_state = EN_TRACKER_FSM::STRACK;
//         m_finalRect = r;
//         m_stracker->isLost() = false;
//         m_ssearchCnt = 1;
//         m_initdet = false;
//         if(minDist < m_serDis)
//         {
//             m_stracker->reset();
//             m_stracker->init(r, frame);
//         }
//         m_strackerfailedCnt--;
//         return;
//     }

//     if(m_ssearchCnt % 5 == 0)
//     {
//         // printf("ss find\n");
//         double sim = 0.f;
//         auto rect = m_stracker->find(frame, sim);
// #if TRACKER_DEBUG
//         cv::rectangle(frame, rect, cv::Scalar(123,30,56), 2);
// #endif

//         // if(sim < 0.3)
//         // if(sim > m_serSiThr || minDist < m_serDis)
//         if(sim > m_serSiThr)
//         {
//             //find, to strack
//             m_state = EN_TRACKER_FSM::STRACK;
//             // m_finalRect = sim > 0.3 ? rect : r;
//             m_finalRect = rect;
//             m_stracker->isLost() = false;
//             m_ssearchCnt = 1;
//             m_initdet = false;
//             if(minDist < m_serDis && sim < 0.3)
//             {
//                 m_stracker->reset();
//                 m_stracker->init(m_finalRect, frame);
//             }
//             m_strackerfailedCnt--;
// #if TRACKER_DEBUG
//             cv::rectangle(frame, r, cv::Scalar(0,255,255), 2);
// #endif
//             return;
//         }
// #if TRACKER_DEBUG
//             cv::rectangle(frame, r, cv::Scalar(0,255,255), 2);
// #endif
//         m_finalRect = m_trackObj.m_rect;
//         m_state = EN_TRACKER_FSM::SSEARCH;
//         m_ssearchCnt++;
//     }
//     else
//     {
//         // printf("ss pred\n");
//         // m_trackObj.updateWithoutDet(m_estimator.getVelo());
//         // usleep(5000);

//         // // std::cout<<frame<<std::endl;
//         // // std::cout<<m_trackObj.m_rect<<std::endl;

//         // if(m_trackObj.m_rect.x < 0)
//         //     m_trackObj.m_rect.x = 0;
//         // if(m_trackObj.m_rect.x + m_trackObj.m_rect.width > frame.cols)
//         //     m_trackObj.m_rect.x = frame.cols - m_trackObj.m_rect.width - 1;
//         // if(m_trackObj.m_rect.y < 0)
//         //     m_trackObj.m_rect.y = 0;
//         // if(m_trackObj.m_rect.y + m_trackObj.m_rect.height > frame.rows)
//         //     m_trackObj.m_rect.y = frame.rows - m_trackObj.m_rect.height - 1;

//         // m_stracker->setRoi(m_trackObj.m_rect);
//         m_finalRect = m_trackObj.m_rect;

//         m_state = EN_TRACKER_FSM::SSEARCH;
//         m_ssearchCnt++;
//     }


}


void realtracker::fsmUpdate(cv::Mat &frame, std::vector<BBox> &boxs)
{

    switch (m_state)
    {
    case EN_TRACKER_FSM::STRACK:
        FSM_PROC_STRACK(frame, boxs);
        break;
    case EN_TRACKER_FSM::SEARCH:
        FSM_PROC_SEARCH(frame);
        break;
    case EN_TRACKER_FSM::SSEARCH:
        FSM_PROC_SSEARCH(frame, boxs);
        break;
    default:
        break;
    }
}

void realtracker::setGateSize(int s)
{
    m_stracker->setGateSize(s);
    m_gateS = s;
    if(m_gateS == 10)
    {
        m_smlv = true;
    }

}

cv::Point realtracker::centerPt()
{
    return m_stracker->centerPt();
}

//void realtracker::updateGyro(ST_GYRO_INFO data)
//{
//    m_stracker->updateGyroInfo((float*)&data);
//}


bool realtracker::sseFind(float sim)
{
    printf("m_ssearchCnt:%d\n", m_ssearchCnt);
    if(m_trackObj.m_velo[0] == 0 && m_trackObj.m_velo[1] == 0 || m_ssearchCnt > 90)
        return sim > 0.1;
    else if( m_ssearchCnt > 60)
        return sim > 0.15;
    else if( m_ssearchCnt > 40)
        return sim > 0.2;
    else if( m_ssearchCnt > 30)
        return sim > 0.3;
    else
        return sim > 0.5;
}

uint8_t realtracker::getState()
{
    if(m_state == EN_TRACKER_FSM::STRACK)
    {
        return 0x02;
    }
    else if(m_state == EN_TRACKER_FSM::SSEARCH)
    {
        return 0x06;
    }
    else if(m_state == EN_TRACKER_FSM::LOST)
    {
        return 0x03;
    }
    else if(m_state == EN_TRACKER_FSM::INIT)
    {
        return 0x00;
    }
}

void realtracker::updateServoInfo(float x, float y, int overhead)
{
    m_servoX = x;
    m_servoY = y;
    m_overhead = overhead;

    if(m_overhead == 0x01)
    {
        m_state = EN_TRACKER_FSM::SSEARCH;
    }
}