#include "CameraControl.hpp"
#include <iostream>
#include <cstring>

CameraControl::CameraControl() : handle(nullptr), pData(nullptr), imageWidth(1920), imageHeight(1080) {
    memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
}

CameraControl::~CameraControl() {
    StopGrabbing();
    if (handle != nullptr) {
        MV_CC_CloseDevice(handle);
        MV_CC_DestroyHandle(handle);
    }
    if (pData != nullptr) {
        free(pData);
    }
}

bool CameraControl::Initialize() {
    // 枚举设备
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

    int nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
    if (MV_OK != nRet || stDeviceList.nDeviceNum == 0) {
        std::cout << "Failed to enumerate devices or no device found! Error code: " << nRet << std::endl;
        return false;
    }

    // 创建句柄
    nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[0]);
    if (MV_OK != nRet) {
        std::cout << "Failed to create handle! Error code: " << nRet << std::endl;
        return false;
    }

    // 打开设备
    nRet = MV_CC_OpenDevice(handle);
    if (MV_OK != nRet) {
        std::cout << "Failed to open device! Error code: " << nRet << std::endl;
        MV_CC_DestroyHandle(handle);
        handle = nullptr;
        return false;
    }

    // 设置像素格式为 RGB8
    nRet = MV_CC_SetEnumValue(handle, "PixelFormat", PixelType_Gvsp_RGB8_Packed);
    if (MV_OK != nRet) {
        std::cout << "Failed to set pixel format! Error code: " << nRet << std::endl;
        return false;
    }

    // 设置触发模式为连续采集
    nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
    if (MV_OK != nRet) {
        std::cout << "Failed to set trigger mode! Error code: " << nRet << std::endl;
        return false;
    }

    // 分配内存
    pData = (unsigned char*)malloc(imageWidth * imageHeight * 3);
    if (pData == nullptr) {
        std::cout << "Failed to allocate memory for image data!" << std::endl;
        return false;
    }

    return true;
}

bool CameraControl::StartGrabbing() {
    int nRet = MV_CC_StartGrabbing(handle);
    if (MV_OK != nRet) {
        std::cout << "Failed to start grabbing! Error code: " << nRet << std::endl;
        return false;
    }
    return true;
}

void CameraControl::StopGrabbing() {
    if (handle != nullptr) {
        MV_CC_StopGrabbing(handle);
    }
}

bool CameraControl::GetFrame(cv::Mat& frame) {
    int nRet = MV_CC_GetOneFrameTimeout(handle, pData, imageWidth * imageHeight * 3, &stImageInfo, 1000);
    if (MV_OK == nRet) {
        // 将数据转换为 OpenCV Mat 格式
        cv::Mat tempFrame(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, pData);
        cv::cvtColor(tempFrame, frame, cv::COLOR_RGB2BGR);
        return true;
    }
    std::cout << "Failed to capture frame! Error code: " << nRet << std::endl;
    return false;
}
