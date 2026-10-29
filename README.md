## Live Demo

*** Try it here: https://mengyi-an.github.io/image-classifier/**

Click the link to try the model in your browser. You can pick one of the example images from the CIFAR-10 test set, or upload your own picture. The page shows the predicted class and the top-3 probabilities.

The trained model (90.2% test accuracy) is exported to ONNX and runs fully in your browser with ONNX Runtime Web, so no server is needed and your images are never uploaded anywhere.

Note: the model was trained on small 32×32 images of 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). It works best on simple, centered pictures of these classes, and may be wrong on high-resolution photos.



## Model.pt

存的是acc 最高那一轮的权重参数等，通常就把它当作训练好的模型。
这里 model.pt是 no_dropout跑出来的结果

## 文件说明

model.py: 模型结构

train.py: 训练 + 记录到 MLflow

plot.py: 所有的 plot 相关函数

serve.py: API 服务




