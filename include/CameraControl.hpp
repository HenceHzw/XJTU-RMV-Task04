#ifndef CAMERACONTROL_HPP
#define CAMERACONTROL_HPP

#include "MvCameraControl.h"
#include <opencv2/opencv.hpp>

class CameraControl {
public:
    CameraControl();
    ~CameraControl();

    bool Initialize();
    bool StartGrabbing();
    void StopGrabbing();
    bool GetFrame(cv::Mat& frame);

private:
    void* handle;
    MV_FRAME_OUT_INFO_EX stImageInfo;
    unsigned char* pData;
    int imageWidth;
    int imageHeight;
};

#endif // CAMERACONTROL_HPP
