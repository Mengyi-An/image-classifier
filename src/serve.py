"""
把模型封装成一个 API 接口
serve.py 把训练好的模型包装成一个 HTTP 接口，别人用浏览器或程序上传一张图片，就能拿到分类结果。先看整体，再逐段拆开
Image classification API.

5 interface totally:
    GET  /health               检查服务是否就绪
    GET  /classes              列出 10 个类别
    GET  /classes/{name}       查某个类别的说明
    POST /predict              对一张图片分类
    POST /predict/batch        一次对多张图片分类
"""

import io
import os
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image # 把字节解码成图片
from torchvision import transforms

from src.model import IMAGE_SIZE, NORM_MEAN, NORM_STD, ImageClassifier


# 取已经训练好的模型，路径/data/model.pt
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 上传文件上限 10MB，批量接口一次最多 20 张。上传内容会整个读进内存，所以必须设上限，否则有人传一个几个 GB 的文件就能把服务撑垮。
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_FILES = 20

model = None
model_metadata = {}

TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE), # 训练数据图片是 32*32，所以这里需要先 resize 一下用户上传的图片大小
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

CLASS_DESCRIPTIONS = {
    "airplane": "Fixed-wing aircraft",
    "automobile": "Car or similar road vehicle",
    "bird": "Any species of bird",
    "cat": "Domestic cat",
    "deer": "Deer or similar deer-like animal",
    "dog": "Domestic dog",
    "frog": "Frog or toad",
    "horse": "Horse or similar equine",
    "ship": "Large watercraft",
    "truck": "Large road vehicle for cargo",
}


def load_model():
    global model, model_metadata
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found. Run src/train.py first.")

    # map_location=DEVICE 保证在 GPU 上训练的模型能加载到只有 CPU 的机器上
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    # train.py writes a bundle (weights + params + seed + metrics). Older
    # checkpoints are a bare state_dict, so both are accepted.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_metadata = {
            "params": checkpoint.get("params", {}),
            "seed": checkpoint.get("seed"),
            "metrics": checkpoint.get("metrics", {}),
        }
    else:
        state_dict = checkpoint
        model_metadata = {}

    # 创建一个空结构的模型，权重是随机的
    m = ImageClassifier()
    m.load_state_dict(state_dict)
    m.eval()
    model = m
    print(f"Model loaded on {DEVICE}. Metadata: {model_metadata or 'none (bare state_dict)'}")


def read_upload(file: UploadFile) -> bytes:
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
    return contents

# io.BytesIO 把字节包装成类似文件的对象，Image.open 才能读。
# .convert("RGB") 统一成三通道，因为有些图片是灰度图或者带透明通道的 PNG，通道数不对会直接报错。
# 解码失败就返回 400，表示客户端传来的东西不对。
def classify_image(image_bytes: bytes, top_k: int, min_confidence: float) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    # topk(k) 返回概率最大的 k 个值和它们的编号
    top_prob, top_idx = probs.topk(top_k)
    top_predictions = [
        {"class": ImageClassifier.CLASSES[i], "probability": round(p, 4)}
        for i, p in zip(top_idx.tolist(), top_prob.tolist()) # tolist() 把张量转成普通 Python 列表
    ]

    # 取最有可能的类别和概率
    top_class = top_predictions[0]["class"]
    top_confidence = top_predictions[0]["probability"]
    confident = top_confidence >= min_confidence

    # 如果第一名的概率不够高，prediction 就返回 "uncertain"
    return {
        "prediction": top_class if confident else "uncertain",
        "confidence": top_confidence,
        "confident": confident,
        "top_k": top_predictions,
    }


# lifespan 管理应用的生命周期。服务启动时先执行 yield 之前的代码，也就是加载模型。
# yield 之后可以写关闭时的清理，这里没有。这样模型只在启动时加载一次，之后所有请求共用同一个，不会每个请求都重新读文件。
# async def 是异步函数，@asynccontextmanager 是装饰器，是 FastAPI 规定的写法，你现在不需要深究，知道它是启动时的钩子就够了。
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

# FastAPI 应用实例
app = FastAPI(
    title="Image Classifier API",
    description="Classifies images into CIFAR-10 categories using a custom CNN.",
    version="1.0.0",
    lifespan=lifespan,
)

# 返回一句欢迎信息，提醒去 /docs
@app.get("/")
def root():
    return {"message": "Image Classifier API is running. Visit /docs for usage."}

# @app.get("/health") 是一个装饰器，意思是把下面这个函数登记为处理 GET /health 请求的函数。
# 有人访问这个地址，FastAPI 就调用这个函数，把返回值转成 JSON 发回去。
# @app.post 同理，对应 POST 请求。上传文件用 POST，查询信息用 GET，这是 HTTP 的惯例。
@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ok", "device": DEVICE}


@app.get("/classes")
def get_classes():
    return {"classes": ImageClassifier.CLASSES}


# 网址里的 {name} 是路径参数，访问 /classes/cat 时，name 就等于 "cat"，自动传给函数。不存在的类别返回 404
@app.get("/classes/{name}")
def get_class_info(name: str):
    if name not in CLASS_DESCRIPTIONS:
        raise HTTPException(status_code=404, detail=f"Class '{name}' not found.")
    return {"class": name, "description": CLASS_DESCRIPTIONS[name]}


# 参数声明就是接口的说明书，FastAPI 会据此自动解析和校验。
# file: UploadFile = File(...) 表示必须上传一个文件，... 在这里表示必填。
# top_k 是网址里的查询参数，比如 /predict?top_k=5。ge=1, le=10 限制取值在 1 到 10 之间，超出范围 FastAPI 会自动返回 422 错误，不用你手写判断。
# min_confidence 同理，限制在 0 到 1。
@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    top_k: int = Query(default=3, ge=1, le=10, description="Number of top predictions to return"),
    min_confidence: float = Query(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence to return a prediction"
    ),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    contents = read_upload(file)
    return classify_image(contents, top_k, min_confidence)


@app.post("/predict/batch")
def predict_batch(
    files: list[UploadFile] = File(...),
    top_k: int = Query(default=3, ge=1, le=10),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=422,
            detail=f"Batch of {len(files)} exceeds the limit of {MAX_BATCH_FILES} files.",
        )

    results = []
    for file in files:
        contents = read_upload(file)
        result = classify_image(contents, top_k, min_confidence)
        results.append({"filename": file.filename, **result})

    return {
        "total": len(results),
        "results": results,
    }
