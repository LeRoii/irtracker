#ifndef _TRACKER_H_
#define _TRACKER_H_

#include "itracker.h"
#include "type_api.h"
#include <random>

enum class EN_TRACKER_FSM
{
    LOST = 0,
    INIT = 1,
    STRACK = 2,
    DTRACK = 3,
    SEARCH = 4,
    SSEARCH = 5,

};

class trackObj
{
public:
    trackObj() = default;
    ~trackObj() = default;

    void init(const cv::Rect &box, cv::Mat frame);

    cv::Point center();
    void update(cv::Mat img, const cv::Rect &box, double s);
    void updateWithoutDet(std::pair<float,float> bgvelo);
    bool isLost();
    void predict();
    
    cv::Rect m_rect;
    float m_prob;
    int m_cls;
    int m_age;

    int m_lostCnt;

    std::deque<cv::Point> m_trace;
    cv::Mat m_hist;
    float m_velo[2];

    bool m_strackerLost;
    bool m_dtrackerLost;

    cv::Rect m_strackerRet;
    cv::Rect m_dtrackerRet;
    cv::Mat m_patch;

    cv::Rect m_initRect;

    void fixVelo(std::pair<float,float> bgvelo);
    int m_velomod;

private:
    inline void calcVelo();
    std::deque<std::pair<int,int>> m_veloBuf;
    cv::Rect m_lastPos;
    std::pair<int,int> m_wdsz;

    float rx, ry;
    
};

class veloestimater
{
public:
    veloestimater();
    ~veloestimater() = default;

    void init(cv::Mat frame);
    void calcof(cv::Mat frame);
    void calcVelo();
    std::pair<float,float> getVelo();

private:
    cv::Mat old_frame, old_gray, frame_gray;
    std::vector<cv::Point2f> p0, p1;
    int offset;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::TermCriteria criteria;
    std::deque<std::pair<float,float>> m_veloBuf;
    std::pair<float,float> m_bgvelo;

};



class realtracker
{
public:
    realtracker();
    ~realtracker();
    void init(const cv::Point &pt, cv::Mat image);
    void reset();
    cv::Rect update(cv::Mat &frame, std::vector<BBox> &boxs);
    bool isLost();
    void setGateSize(int s);
    cv::Point centerPt();
    uint8_t getState();
    void updateServoInfo(float x, float y, int overhead);

private:
    bool sseFind(float sim);
    void fsmUpdate(cv::Mat &frame, std::vector<BBox> &boxs);
    void FSM_PROC_STRACK(cv::Mat &frame, std::vector<BBox> &boxs);
    void FSM_PROC_SEARCH(cv::Mat &frame);
    void FSM_PROC_SSEARCH(cv::Mat &frame, std::vector<BBox> &boxs);
    itracker *m_stracker;
    trackObj m_trackObj;

    EN_TRACKER_FSM m_state;
    cv::Rect m_finalRect, m_showRect;
    int m_strackerfailedCnt;
    int m_ssearchCnt;
    veloestimater m_estimator;
    bool m_initdet;
    int m_strackerslpcnt;
    float m_serDis;
    float m_serSiThr;
    int m_gateS;
    bool m_smlv;

    float m_detDisThres;

    float m_servoX;
    float m_servoY;
    int m_overhead;
    int m_static;
};






#endif