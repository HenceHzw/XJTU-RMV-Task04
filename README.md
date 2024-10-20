# XJTU-RMV-Task04

## tree

```
.
├── CMakeLists.txt
├── include
│   ├── CameraControl.hpp            ---相机图像获取
│   ├── CameraParams.h               
│   ├── MvCameraControl.h
│   ├── MvErrorDefine.h
│   ├── MvISPErrorDefine.h
│   ├── MvObsoleteInterfaces.h
│   ├── ObsoleteCamParams.h
│   ├── PixelType.h
│   └── VideoProcessor.hpp           ---对于图像的处理显示，以及数字识别、距离测算
├── main.cpp
├── model_data
│   ├── model_script_logits.pt          ---MNIST --Linear输出
│   ├── model_script_log_softmax.pt     ---MNIST --log_softmax输出
│   └── SVHN_model_script.pt            ---SVHN  --log_softmax输出
├── README.assets
│   └── image-20241013221859257.png
├── README.md
├── src
│   ├── CameraControl.cpp
│   └── VideoProcessor.cpp
└── video                            ---用于测试装甲板识别的视频
    ├── blue.mp4     
    └── task04.mp4

5 directories, 20 files
```



##  遇到的问题与改进

1. 装甲板的两个相匹配的灯条在识别过程中通过angle差异进行了第一步筛选。在识别过程中大多数情况是（leftLight.angle : 175.444 ，rightLight.angle : 178.246）这样的小差异角度，但是有时候会出现（leftLight.angle : 177.711，rightLight.angle : 0.18965）这样的情况。所以对angle_gap的这个特殊情况进行了处理。

2. 对于两个相匹配的灯条的外可能会有第三个灯条的angle的识别值与这两个的差异较小，这个是小概率事件。如果因为这个一两帧的错误对angle_gap的筛选范围进一步缩小，会导致正常识别的装甲板受到影响。也考虑过对于装甲板的灯条角度的gap进行择优选取绘制矩形，但是这在装甲板旋转时，会导致每一帧只能识别一个装甲板灯条的情况。

   更新：angle_gap的筛选范围我放松了一点，但是在其他参数限制方面进行了调整（例如灯条之间距离比例等参数）。在此情况下，已经对于大多数的装甲板识别情况有较为稳定的结果。但是对于一些特殊位置依然会存在错误识别，把不是相匹配，但是平行、大小相似的灯条之间给框选出来。

3. 尝试用MNIST数据集训练的模型，部署到opencv任务中。但是在安装libtorch时，下载错了libtorch的版本，折腾半天。(T-T) 

   不知道这两个的区别是啥意思，但是下载下面的Download here (cxx11 ABI):就行了

   <img src="README.assets/image-20241013221859257.png" alt="image-20241013221859257" style="zoom: 80%;" />

   更新：之前自建的网络直接输出的是经过线性层的结果，并不是概率，所以无法通过概率值起到一个筛选装甲板的作用。于是我在线性层的基础上用log_softmax作为输出，并且用NLLLoss作为对应的损失函数。这样就可以在调用模型的时候输出概率，加以利用了。

   更新：用MNIST数据集训练的结果，虽然对于手写数字的测试集有较高的识别准确率，并且在大多数情况下也确实可以识别出装甲板的数字。但是从输出的概率结果来看，识别的准确率并不是很高。这一点在实际测试的时候一些帧存在识别错误的情况可以反映出来。

   于是我换了个数据集SVHN，量更大管饱，并且更贴近装甲板数字的情况。由于我网络搭建的比较简单，在其测试集上的准确率并没有那么高，大概85%左右。但是我先姑且用着，结果对于装甲板的数字识别的准确率有非常大的提高。用MNIST训出来的简直逊爆了。

   后续考虑把网络结构再改改，更复杂一点，或许效果会更好一点。

4. 对于装甲板数字的识别，我的想法是先截取装甲板数字区域。本来觉得截取这个区域没有必要完全按照识别到的装甲板的角度来截取，只要截取的image里面有数字不就行了。但是后续在换低曝光的视频时，由于截取区域会有灯条包括在内，所以会导致模型识别的时候注意到灯条，而忽略了亮度低不明显的数字。所以这里我有两个想法来解决问题：

   - 把截取图像的灯条颜色过滤掉，反正就是要通过操作降低或者抹去灯条的亮度干扰。
   - 直接根据识别到的装甲板的角度来截取图像，这样就会避免把灯条也截取进所需的装甲板数字的图像

   最后我选择了第一种方法解决，因为简单方便一点，而且效果也还不错

5. 对于第二点所提到的问题，我在装甲板数字识别的基础上对于装甲板的识别进一步加以限制。识别出灯条并且框选范围后，对所框选范围进行数字识别，如果返回的数字识别的结果的概率值较小，则可以认为这个框选范围里面并没有数字，也就是说这不是装甲板，只是错误识别的不匹配灯条。

   加了这一条件限制之后，已经基本上实现真正的稳定识别了。后面考虑用更好的网络模型来识别数字，以给出一个更加精确的概率筛选值。

6. 对于距离识别，用的是solvePnP求解，具体原理还没细看。

