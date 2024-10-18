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

    //初步筛选灯条
    void VideoProcessor::ScreeningStrip(const Mat& frame, Mat& dilated, vector<LightDescriptor>& lightInfos){
        // 预处理图像
        VideoProcessor::preprocessFrame(frame, dilated);

        //轮廓检测
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(dilated, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

        //筛选灯条
        for (int i = 0; i < contours.size(); i++) {
            // 求轮廓面积
            double area = contourArea(contours[i]);
            // 去除较小轮廓&fitEllipse的限制条件
            if (area < 5 || contours[i].size() <= 1)
                continue;//相当于就是把这段轮廓去除掉


            // 用椭圆拟合区域得到外接矩形（特殊的处理方式：因为灯条是椭圆型的，所以用椭圆去拟合轮廓，再直接获取旋转外接矩形即可）
            //这里用minAreaRect来拟合，效果不好
            RotatedRect Light_Rec = fitEllipse(contours[i]);
 
            // 长宽比和轮廓面积比限制（由于要考虑灯条的远近都被识别到，所以只需要看比例即可）
            // cout<<"contours "<<i<<" | width: "<<Light_Rec.size.width<<" height: "<<Light_Rec.size.height<<endl;
            // putText(frame, std::to_string(i), Light_Rec.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);    
            // cout<<endl<<"------ ("<< i <<")          height : "<<Light_Rec.size.height<<"    &&&&&&&&&&    width : "<<Light_Rec.size.width<<endl<<endl;

            if (Light_Rec.size.height / Light_Rec.size.width < 2.5)
                continue;
            // drawContours(frame, contours, i, Scalar(0, 255, 0), 3); // 绿色轮廓，线宽为2

            lightInfos.push_back(LightDescriptor(Light_Rec));
        }
    }

    // 主程序 ：多条件匹配灯条并且绘制矩形轮廓 + （数字识别）+ （）
    void VideoProcessor::StripMatching(Mat& frame, vector<LightDescriptor>& lightInfos)
    {
        DigitalRecognition DRObject;

        for (size_t i = 0; i < lightInfos.size(); i++) {
            for (size_t j = i + 1; j < lightInfos.size(); j++) {
                Video::LightDescriptor& leftLight = lightInfos[i];
                Video::LightDescriptor& rightLight = lightInfos[j];
                float angleGap_ = abs(leftLight.angle - rightLight.angle);

                // putText(frame, std::to_string(i), leftLight.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0 , 255), 2);   
                // putText(frame, std::to_string(j), rightLight.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);    
                
                if(angleGap_>176)
                {
                    angleGap_=180-angleGap_;
                }
                //由于灯条长度会因为远近而受到影响，所以按照比值去匹配灯条
                float LenGap_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
                float dis = pow(pow((leftLight.center.x - rightLight.center.x), 2) + pow((leftLight.center.y - rightLight.center.y), 2), 0.5);
                float meanLen = (leftLight.length + rightLight.length) / 2;
                float lengap_ratio = abs(leftLight.length - rightLight.length) / meanLen;
                float yGap = abs(leftLight.center.y - rightLight.center.y);
                float yGap_ratio = yGap / meanLen;
                float xGap = abs(leftLight.center.x - rightLight.center.x);
                float xGap_ratio = xGap / meanLen;
                float ratio = dis / meanLen;
                // cout<<"i------------"<<i<<"            j------------"<<j<<endl;                //便于调参 
                // cout<<"angleGap_ : "<<angleGap_<<endl
                //     <<"LenGap_ratio : "<<LenGap_ratio<<endl
                //     <<"lengap_ratio : "<<lengap_ratio<<endl
                //     <<"yGap_ratio : "<<yGap_ratio<<endl
                //     <<"xGap_ratio : "<<xGap_ratio<<endl
                //     <<"ratio : "<<ratio<<endl;

                
                if (angleGap_ > 4.5 ||
                    LenGap_ratio > 0.2 ||
                    lengap_ratio > 0.2 ||
                    yGap_ratio > 1.0 ||
                    xGap_ratio > 2.2 ||
                    xGap_ratio < 0.75 ||
                    ratio > 3 ||
                    ratio < 0.8) {
                    continue;
                }
                // cout<<"i------------"<<i<<"            j------------"<<j<<endl;                //便于调参 
                // cout<<"angleGap_ : "<<angleGap_<<endl
                //     <<"LenGap_ratio : "<<LenGap_ratio<<endl
                //     <<"lengap_ratio : "<<lengap_ratio<<endl
                //     <<"yGap_ratio : "<<yGap_ratio<<endl
                //     <<"xGap_ratio : "<<xGap_ratio<<endl
                //     <<"ratio : "<<ratio<<endl;
                
                //绘制矩形
                Point center = Point((leftLight.center.x + rightLight.center.x) / 2, (leftLight.center.y + rightLight.center.y) / 2);
                float L_Angle=0,R_Angle=0;
                if(leftLight.angle < 90){
                    L_Angle = leftLight.angle + 180;
                }else{
                    L_Angle = leftLight.angle;
                }
                if(rightLight.angle < 90){
                    R_Angle = rightLight.angle + 180;   
                }else{
                    R_Angle = rightLight.angle;
                }
                float meanAngle = (L_Angle + R_Angle) / 2;
                if(meanAngle>=80&&meanAngle<=100){
                    meanAngle+=90;
                }
                RotatedRect rect = RotatedRect(center, Size(dis, meanLen), meanAngle);
                RotatedRect ArmorNumRect = RotatedRect(center, Size(dis, meanLen*(115.0/67.50)), meanAngle);
        
                //*************************************//
                //            识别装甲板数字             //
                //*************************************//
                int armorID = DRObject.matToDigital(frame, ArmorNumRect);
                // cout<<"leftLight.angle : "<<leftLight.angle<<"     rightLight.angle : "<<rightLight.angle<<endl;
                // cout<<"L_Angle : "<<L_Angle<<"     R_Angle : "<<R_Angle<<endl;
                // cout<<"meanAngle : "<<meanAngle<<endl;


                Point2f vertices[4];
                ArmorNumRect.points(vertices);
                std::vector<cv::Point2f> verticesVec(vertices, vertices + 4);
                //*************************************//
                //            识别装甲板距离             //
                //*************************************//
                DRObject.GetDistance(verticesVec);





                
                for (int i = 0; i < 4; i++) {
                    line(frame, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 2.2);

                    // cout<<"vertices["<<i<<"].x : "<<vertices[i].x<<endl<<"vertices["<<i<<"].y : "<<vertices[i].y<<endl<<endl;
                }
            }
        }

    }




    //装甲板数字识别
    int DigitalRecognition::matToDigital(const cv::Mat &frame, RotatedRect &rect) {

        Mat armorPlate = DigitalRecognition::GetArmorImage(frame,rect);
        // 正则化
        armorPlate.convertTo(armorPlate, CV_32FC1, 1.0f / 255.0f);

        // 将 OpenCV 的 Mat 转换为 Tensor, 注意两者的数据格式
        // OpenCV: H*W*C 高度, 宽度, 通道数
        auto input_tensor = torch::from_blob(armorPlate.data, {1, IMAGE_COLS, IMAGE_ROWS, 1});

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
        if(ans <= 5)
        {
            std::cout << "当前机器人编号: " << ans << std::endl;
        }
        

        return ans;
    }

    //截取识别到的装甲板区域
    cv::Mat DigitalRecognition::GetArmorImage(const cv::Mat &frame, RotatedRect &rect)
    {
        Mat img = frame.clone();
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




    double DigitalRecognition::GetDistance(vector<cv::Point2f>& imagePoints)
    {
        double distance = 0;

        // 相机内参矩阵 (fx, fy, cx, cy)
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
                                718.0, 0, 2673.0, 
                                0, 560.0, 2673.0, 
                                0, 0, 1);

        // 相机畸变系数 (可以忽略畸变设置为0，若不考虑畸变)
        cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 
                            0, 0, 0, 0, 0);

        //物体的真实尺寸 
        //小装甲板
        // std::vector<cv::Point3f> objectPoints;
        // objectPoints.push_back(cv::Point3f(67.50F, -28.50F, 0));            // 右上角
        // objectPoints.push_back(cv::Point3f(67.50F, 28.50F, 0));             // 右下角
        // objectPoints.push_back(cv::Point3f(-67.50F, 28.50F, 0));            // 左下角
        // objectPoints.push_back(cv::Point3f(-67.50F, -28.50F, 0));           // 左上角
        //大装甲板
        std::vector<cv::Point3f> objectPoints;
        objectPoints.push_back(cv::Point3f(115.00F, -28.50F, 0));            // 右上角
        objectPoints.push_back(cv::Point3f(115.00F, 28.50F, 0));             // 右下角
        objectPoints.push_back(cv::Point3f(-115.00F, 28.50F, 0));            // 左下角
        objectPoints.push_back(cv::Point3f(-115.00F, -28.50F, 0));           // 左上角





        // 定义旋转向量和位移向量
        cv::Mat rvec, tvec;

        // 使用PnP算法估算物体的旋转和位移向量
        bool success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
        if (success) {
            // cout << "Rotation Vector (rvec): " << rvec << endl;
            // cout << "Translation Vector (tvec): " << tvec << endl;

            // 物体与相机的距离为 tvec 的z轴值，单位为毫米
            distance = tvec.at<double>(2) / 1000.0; // 转换为米
            cout << "Distance to object: " << distance*(1.25/0.215) << " meters" << endl;
        } else {
            cout << "PnP calculation failed!" << endl;
        }
        
        return distance;

    }



    
}