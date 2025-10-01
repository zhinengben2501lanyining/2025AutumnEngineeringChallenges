## Level 1：配置YOLO环境

#### 安装 pytorch

本人的显卡还是相当不错的，一开始准备安装gpu版本的，结果一直装不上，pytorch只能装到121版本的，而且121还用不了，后来经过各方面的研究学习，发现是自己的50系显卡太新了，不能适配。

但是呢，可爱的清华源，装不上高版本的gpu版本的pytorch，给我装了一个cpu版本的，本人也没再更改。

另外，发现一个问题，挂着GitHub的加速器，不能在清华源下东西。

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 -i https://pypi.tuna.tsinghua.edu.cn/simple

#### 安装 Ultralytics YOLO

这个比较简单，就不阐述了

pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

#### 安装 opencv-python

这个也是

pip install opencv-python matplotlib pillow -i https://pypi.tuna.tsinghua.edu.cn/simple

#### 检验安装

pip install opencv-python matplotlib pillow -i https://pypi.tuna.tsinghua.edu.cn/simple



## 第二次尝试

本人尝试用梯子直接在官方源下载，下好了pytorch的版本cu129

其余步骤与下文相同，不过多阐述