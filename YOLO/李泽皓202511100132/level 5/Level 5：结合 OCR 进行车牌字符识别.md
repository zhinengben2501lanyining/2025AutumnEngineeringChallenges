## Level 5：结合 OCR 进行车牌字符识别

```
pip install paddlepaddle
pip install paddleocr
```

本人在此处重配了八次环境，python改的次数已经记不清了

先是cpu版本的，本人在[安装 - PaddleOCR 文档](https://www.paddleocr.ai/main/version3.x/installation.html)发现里面的ocr主流是gpu运行的，然后尝试换pytorch

然后转向了GPU版本的torch-cu129，python=3.8，大抵是50系刚出还没完全适配的原因，这个非常不稳定，ocr直接就是没法用，paddleocr和easyocr之间换了好几次仍然不行。

采用比较稳定的cpu版的torch，python=3.8，typeerror

-----------------------------------------，python=3.9，typeerror

-----------------------------------------，python=3.10，终于好了

当然识别不出来中文

在python提供车牌号字库，结果给我打问号

最后采用让他挑选的机制，不能打问号，需要在我提供的字库里找出一个最合适的

终于成功了

在使用时需要更改python中的model和image路径，由于本人的图片和python不在同一级文件里，在路径前加了一个r填写的实际路径，审核要验证的话记得改一下
