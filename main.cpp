#include "VideoProcessor.hpp"

int main()
{
    Video::DigitalRecognition digitalRec;
    Video::VideoProcessor vp;
    VideoCapture video = vp.openVideo("/opt/RM_tasks/XJTU-RMV-Task04/video/blue.mp4");
    // "/opt/RM_tasks/XJTU-RMV-Task04/video/blue.mp4"
    // "/opt/RM_tasks/XJTU-RMV-Task04/video/task04.mp4"
    
    //变量定义
    Mat frame, dilatee, armorPlate;;
    Rect boundRect;
    RotatedRect box;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Point2f> boxPts(4);
    //--------------------------------------------
    int armorID=-1;
    int count = 0;
    //图像预处理
    for (;;) {
        count++;
        cout<<endl<<"第 "<<count<<" 帧****************************************************************"<<endl<<endl;
        // Rect point_array[20];
        if (!vp.getFrame(video, frame)) {
            break; // 当读取完毕或出错时，退出循环
        }
        // 预处理图像
        vp.preprocessFrame(frame, dilatee);

        findContours(dilatee, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);//轮廓检测
        vector<Video::LightDescriptor> lightInfos;//创建一个灯条类的动态数组
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

            lightInfos.push_back(Video::LightDescriptor(Light_Rec));
        }
        //二重循环多条件匹配灯条
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
                //均长
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

            
                
                

                //匹配不通过的条件
                // cout<<i<<"  leftLight.angle : "<<leftLight.angle<<endl
                //     <<j<<"  rightLight.angle : "<<rightLight.angle<<endl;
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
                // putText(frame, std::to_string(i)+"+"+std::to_string(j), center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);    
                float meanAngle = (leftLight.angle + rightLight.angle) / 2;
                if(meanAngle>88&&meanAngle<=92)
                {
                    meanAngle+=90;
                }
                // cout<<"meanAngle : "<<meanAngle<<endl;
                RotatedRect rect = RotatedRect(center, Size(dis, meanLen), meanAngle);
                RotatedRect ArmorNumRect = RotatedRect(center, Size(dis, meanLen+50), meanAngle);
                //***************这里的size值还可以商榷一下 */



                armorPlate = digitalRec.GetArmorImage(frame,ArmorNumRect); // 截取装甲板区域
                armorID = digitalRec.matToDigital(armorPlate);
                

                Point2f vertices[4];
                // rect.points(vertices);
                ArmorNumRect.points(vertices);

                for (int i = 0; i < 4; i++) {
                    line(frame, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 2.2);
                }
            }
        }
 
        namedWindow("video", WINDOW_FREERATIO);
        imshow("video", frame);
        if (waitKey(3000000) == 27)  // 按下 'Esc' 键退出
        {
            continue;
        }
    }
    video.release();
    cv::destroyAllWindows();
    return 0;
}