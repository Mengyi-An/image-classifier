// 浏览器端推理：加载 ONNX 模型，把图片预处理成和训练时一样的张量，再算出每个类别的概率。
// 页面（index.html）只调用这里的 Classifier.load() 和 Classifier.predict()。
const Classifier = (() => {
  let session = null;
  let meta = null;   // model_meta.json：类别名、图片尺寸、归一化用的均值和标准差，由 src/export_onnx.py 生成

  async function load() {
    meta = await (await fetch("model_meta.json")).json();
    session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
    return meta;
  }

  // 必须和训练时的 EVAL_TRANSFORM 完全一致，否则模型不会报错，只会悄悄地预测错：
  //   1. 缩放到 32×32（和 serve.py 的 Resize 一样，直接拉伸，不保持长宽比）
  //   2. 像素值 0–255 → 0–1
  //   3. 每个通道减均值、除以标准差
  //   4. 浏览器读出来的顺序是 高 × 宽 × RGBA，要改成 PyTorch 的 通道 × 高 × 宽
  function preprocess(image) {
    const [height, width] = meta.image_size;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#fff";   // 透明的 PNG 先垫一层白底，不然透明部分会变成黑色
    ctx.fillRect(0, 0, width, height);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, 0, 0, width, height);

    const rgba = ctx.getImageData(0, 0, width, height).data;
    const pixels = width * height;
    const data = new Float32Array(3 * pixels);
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < pixels; i++) {
        data[c * pixels + i] = (rgba[i * 4 + c] / 255 - meta.mean[c]) / meta.std[c];
      }
    }
    return new ort.Tensor("float32", data, [1, 3, height, width]);
  }

  // 模型输出的是 logits，用 softmax 换算成概率。先减去最大值，避免 exp 溢出
  function softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((x) => x / sum);
  }

  // 返回所有类别的概率，从高到低排好序：[{ name: "cat", prob: 0.93 }, ...]
  async function predict(image) {
    const output = await session.run({ input: preprocess(image) });
    const probs = softmax(Array.from(output.logits.data));
    return meta.classes
      .map((name, i) => ({ name, prob: probs[i] }))
      .sort((a, b) => b.prob - a.prob);
  }

  return { load, predict };
})();
