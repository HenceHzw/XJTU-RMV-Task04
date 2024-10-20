#ifndef VIDEO_PROCESSOR_HPP
#define VIDEO_PROCESSOR_HPP

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>


using namespace std;
using namespace cv;

namespace Video
{
    //灯条类
    class LightDescriptor
    {	    //在识别以及匹配到灯条的功能中需要用到旋转矩形的长宽偏转角面积中心点坐标等
    public:float width, length, angle, area;
        cv::Point2f center;
    public:
        LightDescriptor() {};
        //让得到的灯条套上一个旋转矩形，以方便之后对角度这个特殊因素作为匹配标准
        LightDescriptor(const cv::RotatedRect& light)
        {
            width = light.size.width;
            length = light.size.height;
            center = light.center;
            angle = light.angle;
            area = light.size.area();
        }
    };




    //装甲板数字识别，调用预训练模型
    class DigitalRecognition {
    private:
        torch::jit::script::Module module;
        torch::Device device;
        const int IMAGE_COLS = 32;
        const int IMAGE_ROWS = 32;
    public:
        explicit DigitalRecognition(bool use_cuda = true,
                                    const std::string &model_path = "/opt/RM_tasks/XJTU-RMV-Task04/model_data/SVHN_model_script.pt") : device(torch::kCPU) {
            if ((use_cuda) && (torch::cuda::is_available())) {
                // std::cout << "CUDA is available! Training on GPU." << std::endl;
                device = torch::kCUDA;
            }
            module = torch::jit::load(model_path, device);
        }

        //装甲板数字识别
        int matToDigital(const cv::Mat &img, RotatedRect &rect, float &max_prob);

        //截取识别到的装甲板区域
        cv::Mat GetArmorImage(const cv::Mat &img, RotatedRect &rect);
        
        // 识别装甲板距离
        double GetDistance(vector<cv::Point2f>& imagePoints);
        
    };
    



    //图像获取，处理，识别以及显示
    class VideoProcessor
    {
    public:
        // 打开视频文件并返回VideoCapture对象
        VideoCapture openVideo(const string& videoPath);
    
        // 从视频中读取一帧图像
        bool getFrame(VideoCapture& video, Mat& frame);

        // 图像预处理
        void preprocessFrame(const Mat& frame, Mat& dilated);

        //初步筛选灯条
        void ScreeningStrip(const Mat& frame, Mat& dilated, vector<LightDescriptor>& lightInfos);

        //二重循环多条件匹配灯条并且绘制矩形轮廓
        void StripMatching(Mat& frame, vector<LightDescriptor>& lightInfos);

    };

}


#endif // VIDEO_PROCESSOR_HPP
