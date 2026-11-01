// In-browser inference: load the ONNX model, preprocess an image exactly as in training,
// and return the probability of each class.
// The page (index.html) only calls Classifier.load() and Classifier.predict().
const Classifier = (() => {
  let session = null;
  let meta = null;   // model_meta.json: class names, image size, normalization mean/std (written by src/export_onnx.py)

  async function load() {
    meta = await (await fetch("model_meta.json")).json();
    session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
    return meta;
  }

  // Must match EVAL_TRANSFORM in training exactly; a mismatch raises no error, the model
  // just silently predicts wrong:
  //   1. Resize to 32x32 (stretched like serve.py's Resize, aspect ratio not kept)
  //   2. Scale pixel values from 0-255 to 0-1
  //   3. Subtract the mean and divide by the std, per channel
  //   4. Reorder from the browser's height x width x RGBA to PyTorch's channel x height x width
  function preprocess(image) {
    const [height, width] = meta.image_size;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#fff";   // white background, otherwise transparent PNG pixels turn black
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

  // Convert the model's logits to probabilities. Subtracting the max first avoids overflow in exp.
  function softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((x) => x / sum);
  }

  // All classes with their probabilities, highest first: [{ name: "cat", prob: 0.93 }, ...]
  async function predict(image) {
    const output = await session.run({ input: preprocess(image) });
    const probs = softmax(Array.from(output.logits.data));
    return meta.classes
      .map((name, i) => ({ name, prob: probs[i] }))
      .sort((a, b) => b.prob - a.prob);
  }

  return { load, predict };
})();
