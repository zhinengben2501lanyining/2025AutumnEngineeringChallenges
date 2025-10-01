## Level 3：划分数据集，配置并训练模型

划分数据集为本人手动操作，没有借助python。划分好的数据集与level 2中展示的文件一样，因为本人的原始数据集已经删掉了。

运行下方指令开始训练模型

yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16

使用前需要先传送到data.yaml所在目录，前面所提到的images和labels需与该文件同级

epochs=100，训练了一百轮，本人睡了一觉都没好，匆匆终止，模型为28轮训练后的成果

训练好的模型及其效果图、datal.yaml将在本文件的同级文件中展示

## 第二次用gpu训练

这次很快就弄好了

一个午饭空一百次训练训练完成