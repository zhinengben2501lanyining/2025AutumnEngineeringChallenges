## Level 4：使用训练好的模型进行推理

将配置文件保存到data.yaml同级目录后

在虚拟环境中运行inference_main.py即可，需要手动选择model

其余三个python由inference_main.py调度使用，可自动识别source类型

python inference_main.py --model 模型所在位置 --source 需要识别文件的位置   摄像头（source=0）

python 在conda中使用时需要指明--model --source            model和source填写实际路径
