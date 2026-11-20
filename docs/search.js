// In-browser image search with CLIP (the "Image search" tab).
// Files in search/ are written by the image-search project (src/export_demo.py):
//   search/meta.json    model name, vector size, image mean, text mean, first caption of each image
//   search/embeds.bin   centered CLIP vector of each image, count x dim float32, row by row
//   search/thumbs/i.jpg small copy of image i
// The page (index.html) only calls the functions returned at the bottom.
const ImageSearch = (() => {
  const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js";

  let transformers = null;
  let meta = null;
  let embeds = null;        // Float32Array, count * dim numbers
  let tokenizer = null;
  let textModel = null;
  let processor = null;
  let visionModel = null;

  // Scale a vector to length 1, so the dot product of two vectors is their cosine similarity.
  // The image vectors in embeds.bin are already normalized in Python.
  function normalize(vector) {
    let sum = 0;
    for (let i = 0; i < vector.length; i++) sum += vector[i] * vector[i];
    const length = Math.sqrt(sum);
    const result = new Float32Array(vector.length);
    for (let i = 0; i < vector.length; i++) result[i] = vector[i] / length;
    return result;
  }

  // Centering, the same as center() in the image-search project: subtract the mean
  // (text mean for texts, image mean for images), then normalize again.
  // It keeps a few "hub" photos from showing up in almost every search.
  function center(vector, mean) {
    const shifted = new Float32Array(vector.length);
    for (let i = 0; i < vector.length; i++) shifted[i] = vector[i] - mean[i];
    return normalize(shifted);
  }

  // Download progress of the model files, reported as "text model 42%"
  function progressReporter(label, onProgress) {
    return (event) => {
      if (event.status === "progress" && event.total) {
        onProgress(`Downloading ${label}… ${Math.round((event.loaded / event.total) * 100)}%`);
      }
    };
  }

  // Load the image vectors and the text half of CLIP (~65 MB, cached by the browser after the first visit).
  async function load(onProgress) {
    onProgress("Loading image vectors…");
    meta = await (await fetch("search/meta.json")).json();
    const buffer = await (await fetch("search/embeds.bin")).arrayBuffer();
    embeds = new Float32Array(buffer);

    onProgress("Loading Transformers.js…");
    transformers = await import(TRANSFORMERS_URL);
    transformers.env.allowLocalModels = false;   // always download the model from the Hugging Face Hub

    tokenizer = await transformers.AutoTokenizer.from_pretrained(meta.model);
    textModel = await transformers.CLIPTextModelWithProjection.from_pretrained(meta.model, {
      dtype: "q8",
      progress_callback: progressReporter("text model", onProgress),
    });
    return meta;
  }

  // The image half of CLIP (~90 MB) is only needed for uploads, so it is loaded on the first upload.
  async function loadVision(onProgress) {
    if (visionModel) return;
    processor = await transformers.AutoProcessor.from_pretrained(meta.model);
    visionModel = await transformers.CLIPVisionModelWithProjection.from_pretrained(meta.model, {
      dtype: "q8",
      progress_callback: progressReporter("image model", onProgress),
    });
  }

  async function encodeText(text) {
    const inputs = tokenizer([text], { padding: true, truncation: true });
    const output = await textModel(inputs);
    return center(normalize(output.text_embeds.data), meta.text_mean);
  }

  async function encodeImage(url) {
    const image = await transformers.RawImage.fromURL(url);
    const inputs = await processor(image);
    const output = await visionModel(inputs);
    return center(normalize(output.image_embeds.data), meta.image_mean);
  }

  // Compare a query vector with every image vector and return the k best:
  // [{ index: 12, score: 0.31 }, ...], best first. skipIndex leaves out the query image itself.
  function topMatches(queryVector, k, skipIndex = -1) {
    const dim = meta.dim;
    const results = [];
    for (let index = 0; index < meta.count; index++) {
      if (index === skipIndex) continue;
      let score = 0;
      const offset = index * dim;
      for (let j = 0; j < dim; j++) score += queryVector[j] * embeds[offset + j];
      results.push({ index, score });
    }
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, k);
  }

  async function searchByText(text, k) {
    const vector = await encodeText(text);
    return topMatches(vector, k);
  }

  async function searchByImage(url, k) {
    const vector = await encodeImage(url);
    return topMatches(vector, k);
  }

  // Similar images for an image that is already in the gallery: its vector is in embeds.bin,
  // so no model needs to run.
  function similarTo(index, k) {
    const start = index * meta.dim;
    const vector = embeds.subarray(start, start + meta.dim);
    return topMatches(vector, k, index);
  }

  function thumbUrl(index) {
    return `search/thumbs/${index}.jpg`;
  }

  function caption(index) {
    return meta.captions[index];
  }

  return { load, loadVision, searchByText, searchByImage, similarTo, thumbUrl, caption };
})();
