# CIFAR-10 Image Classifier

A CNN trained from scratch on CIFAR-10 with PyTorch, with comparison experiments and an in-browser demo.

**Live demo: https://mengyi-an.github.io/image-classifier/**

Try one of the example images or upload your own, and the page will show the three most likely classes. Everything runs in your browser with ONNX Runtime Web. The model only knows 10 classes and was trained on tiny 32×32 images, so clear, centered photos work best.

The demo also has an [image search tab](https://mengyi-an.github.io/image-classifier/#search) that uses CLIP to search 5,000 COCO photos by text or by image. That part is built in my [image-search](https://github.com/Mengyi-An/image-search) project, and this repo just hosts its files in `docs/search.js` and `docs/search/`.

## Results

I trained a baseline and two comparison runs, one without data augmentation and one without dropout. Each run trained for 100 epochs with the same seed and was tracked with MLflow. The best epoch was picked on a validation set, and the test set was only used once at the end.

| Run                  | Augmentation | Dropout | Val acc          | Test acc         |
| -------------------- | ------------ | ------- | ---------------- | ---------------- |
| baseline             | yes          | yes     | 89.12%           | 88.56%           |
| no_augment           | no           | yes     | 85.20%           | 84.78%           |
| **no_dropout** | yes          | no      | **91.38%** | **90.21%** |

![Train vs validation curves](figures/comparison.png)

Without data augmentation the model starts to overfit after about 25 epochs, and its test accuracy drops by almost 4 points. Removing dropout gave the best result. With augmentation already in place, dropout seems to regularize this model too much, so I kept the no_dropout model as the final one.

## Project Structure

```
src/
  model.py         CNN architecture
  train.py         training, evaluation and MLflow logging
  plot.py          learning curves of several runs side by side, plus a summary table
  export_demo.py   export to ONNX for the browser demo, with a check against PyTorch
docs/              the browser demo (GitHub Pages)
  search.js, search/   the image search tab, exported by the image-search project
figures/           result plots
```

## Usage

```bash
pip install -r requirements.txt

python -m src.train                  # baseline
python -m src.train --no-augment     # comparison runs
python -m src.train --no-dropout
python -m src.plot                   # figures/comparison.png

python -m src.export_demo
```

CIFAR-10 is downloaded automatically on the first run.

Training for 100 epochs on a CPU is slow. If you don't have a GPU, you can use [Google Colab](https://colab.research.google.com/) with a free T4 (Runtime → Change runtime type). Clone the project into Google Drive so the results stay after the session ends:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive
!git clone https://github.com/Mengyi-An/image-classifier.git
%cd /content/drive/MyDrive/image-classifier
!pip install -q mlflow     # Colab already has PyTorch with GPU support
!python -m src.train
```

## License

MIT
