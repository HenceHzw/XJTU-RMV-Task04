#include "VideoProcessor.hpp"

namespace Video
{
    // 打开视频文件并返回VideoCapture对象
    VideoCapture VideoProcessor::openVideo(const string& videoPath) {
            VideoCapture video;
            video.open(videoPath);
            if (!video.isOpened()) {
                cout << "Error: Cannot open video file." << endl;
                exit(-1);
            }
            return video;
    }

    // 从视频中读取一帧图像
    bool VideoProcessor::getFrame(VideoCapture& video, Mat& frame) {
            video >> frame;
            if (frame.empty()) {
                cout << "Frame is empty. Exiting..." << endl;
                return false;
            }
            return true;
    }
    
    // 图像预处理
    void VideoProcessor::preprocessFrame(const Mat& frame, Mat& dilated) {
            Mat channels[3], binary, Gaussian;
            Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));// 定义结构元素，用于膨胀操作
            split(frame, channels); // 分离图像的通道

            // 二值化处理
            threshold(channels[0], binary, 220, 255, 0);

            // 高斯模糊
            GaussianBlur(binary, Gaussian, Size(5, 5), 0);

            // 膨胀处理
            dilate(Gaussian, dilated, element);
    }
}