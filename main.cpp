#include "CameraControl.hpp"
#include "VideoProcessor.hpp"



int main() {
    CameraControl camera;
    Video::DigitalRecognition digitalRec;
    Video::VideoProcessor vp;

    //变量定义
    Mat frame, dilatee, armorPlate;
    int armorID=-1;

    // 初始化相机
    camera.Initialize();

    // 开始抓取图像
    camera.StartGrabbing();

    std::cout << "Start grabbing video frames..." << std::endl;

    while (true) {

        //获取每一帧图像
        if (!camera.GetFrame(frame)) {
            continue;
        }
        
        //创建一个灯条类的动态数组
        vector<Video::LightDescriptor> lightInfos;

        // 预处理图像并且筛选灯条
        vp.ScreeningStrip(frame, dilatee, lightInfos);

        //二重循环多条件匹配灯条并且绘制矩形轮廓   +   (识别装甲板数字)
        vp.StripMatching(frame, lightInfos);
        
        // 显示处理后的图像
        cv::imshow("Captured Frame", frame);

        // 按 'Esc' 键退出
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    camera.StopGrabbing();
    std::cout << "Camera closed." << std::endl;

    return 0;
}







//************** */
//  用本地视频测试 
//************** */


// #include "VideoProcessor.hpp"

// int main()
// {
//     Video::DigitalRecognition digitalRec;
//     Video::VideoProcessor vp;
//     VideoCapture video = vp.openVideo("/opt/RM_tasks/XJTU-RMV-Task04/video/blue.mp4");
//     // "/opt/RM_tasks/XJTU-RMV-Task04/video/blue.mp4"
//     // "/opt/RM_tasks/XJTU-RMV-Task04/video/task04.mp4"
    
//     //变量定义
//     Mat frame, dilatee, armorPlate;
//     int armorID=-1;
//     int count = 0;


//     for (;;) {
//         count++;
//         // cout<<endl<<"第 "<<count<<" 帧****************************************************************"<<endl<<endl;
//         if (!vp.getFrame(video, frame)) {
//             break; // 当读取完毕或出错时，退出循环
//         }
//         vector<Video::LightDescriptor> lightInfos;//创建一个灯条类的动态数组
//         vp.ScreeningStrip(frame, dilatee, lightInfos);// 预处理图像并且筛选灯条

//         //二重循环多条件匹配灯条并且绘制矩形轮廓   +   (识别装甲板数字)
//         vp.StripMatching(frame, lightInfos);

//         namedWindow("video", WINDOW_FREERATIO);
//         imshow("video", frame);
//         if (waitKey(3000000) == 27)  // 按下 'Esc' 键退出
//         {
//             break;
//         }
//     }
//     video.release();
//     cv::destroyAllWindows();
//     return 0;
// }