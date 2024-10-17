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
    class VideoProcessor
    {
    public:
        // 打开视频文件并返回VideoCapture对象
        VideoCapture openVideo(const string& videoPath);
    
        // 从视频中读取一帧图像
        bool getFrame(VideoCapture& video, Mat& frame);
        // 图像预处理
        void preprocessFrame(const Mat& frame, Mat& dilated);
    };

    
    //由于在识别中的核心物体以及相关的物理特性是灯条，所以建一个灯条类
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

    class DigitalRecognition {
    private:
        torch::jit::script::Module module;
        torch::Device device;
        const int IMAGE_COLS = 28;
        const int IMAGE_ROWS = 28;
    public:
        /**
        * 默认使用CPU，可通过标志位开启使用GPU
        * @param use_cuda 是否使用GPU
        * @param model_path 模型文件路径
        */
        explicit DigitalRecognition(bool use_cuda = true,
                                    const std::string &model_path = "/opt/RM_tasks/XJTU-RMV-Task04/model_data/model_script_logits.pt") : device(torch::kCPU) {
            if ((use_cuda) && (torch::cuda::is_available())) {
                // std::cout << "CUDA is available! Training on GPU." << std::endl;
                device = torch::kCUDA;
            }
            module = torch::jit::load(model_path, device);
        }

        /**
        * 单张图片分类器
        * @param img 图片，cv::Mat类型
        * @return 分类结果
        */
        int matToDigital(cv::Mat &img) {
            // 正则化
            img.convertTo(img, CV_32FC1, 1.0f / 255.0f);

            // 将 OpenCV 的 Mat 转换为 Tensor, 注意两者的数据格式
            // OpenCV: H*W*C 高度, 宽度, 通道数
            auto input_tensor = torch::from_blob(img.data, {1, IMAGE_COLS, IMAGE_ROWS, 1});

            // Tensor: N*C*H*W 数量, 通道数, 高度, 宽度
            // 数字表示顺序
            input_tensor = input_tensor.permute({0, 3, 1, 2}).to(device);

            // 添加数据
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(input_tensor);

            // 模型计算
            at::Tensor output = module.forward(inputs).toTensor();
            // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/7) << '\n';

            // 输出分类的结果

            // at::Tensor probabilities = torch::nn::functional::softmax(output, /*dim=*/1);  // 对第 1 维（类别维度）进行 softmax
            // auto max_result = probabilities.max(1);  // 获取概率中的最大值
            // torch::Tensor max_values = std::get<0>(max_result);  // 获取最大值
            // float max_value = max_values.item<float>();
            // cout << "max_value (probability): " << max_value << endl;

            int ans = output.argmax(1).item().toInt();
            std::cout << "当前机器人编号: " << ans << std::endl;

            return ans;
        }

        cv::Mat GetArmorImage(cv::Mat &img, RotatedRect &rect)
        {
            Mat armorPlate, grayArmorPlate, binaryArmorPlate, resizedArmorPlate;

            // 找到最小的外接矩形
            cv::Rect boundingRect = rect.boundingRect();

            // 检查外接矩形是否在图像范围内，防止越界
            boundingRect &= cv::Rect(0, 0, img.cols, img.rows);

            // 使用cv::Rect从frame中截取装甲板区域
            armorPlate = img(boundingRect);


            Mat img_hsv;
            cvtColor(armorPlate, img_hsv, COLOR_BGR2HSV);

            // 分离 HSV 通道
            std::vector<Mat> hsv_channels;
            split(img_hsv, hsv_channels);
            Mat brightness = hsv_channels[2]; // V 通道 (亮度)

            // 创建亮度阈值的掩码 (亮度大于 100 为高亮部分, 仅考虑在低曝光的情况下)
            Mat mask;
            threshold(brightness, mask, 100, 255, THRESH_BINARY);

            Mat result = armorPlate.clone();
            result.setTo(Scalar(0, 0, 0), mask);  // 将亮度高的部分设置为黑色

            // namedWindow("ArmorID", WINDOW_FREERATIO);
            // imshow("ArmorID", result);
            // if (waitKey(3000000) == 27)  // 按下 'Esc' 键退出
            // {
               
            // }

            cvtColor(result, grayArmorPlate, COLOR_BGR2GRAY); // 转换为灰度图
            resize(grayArmorPlate, resizedArmorPlate, cv::Size(IMAGE_COLS, IMAGE_ROWS)); // 调整为28x28
            // threshold(resizedArmorPlate, binaryArmorPlate, 128, 255, THRESH_BINARY); // 二值化处理
            return resizedArmorPlate;
        }
        
    };


    
}


#endif // VIDEO_PROCESSOR_HPP
