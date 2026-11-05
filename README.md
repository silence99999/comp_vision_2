# Handwritten digit recognition with DINOv2

Computer Vision Assignment 2, continuing the 0–9 digit dataset from Assignment 1.
The project compares two CNN reruns with a frozen DINOv2 ViT-S/14 encoder and a
trained linear classifier. An augmentation ablation compares the same DINOv2
classifier with and without two affine variants per training image.

## Deliverables

- `Untitled2_Assignment1_and_2.ipynb`: original A1 cells followed by a clearly marked **Assignment 2** section.
- `Assignment2_Report.pdf`: eight-page experimental report.
- `app.py` and `web/index.html`: local web app, drawing/upload → prediction → nearest training examples.
- `digit_pipeline.py`: reproducible training and evaluation.
- `make_figures.py`: figures generated from actual saved outputs.
- `DEMO_GUIDE_RU.md`: 3–5 minute demo recording and defense guide.
- `artifacts/metrics.json`, histories, prediction probabilities, manifest and environment: experiment evidence.

The original A1 TensorFlow test results (90% baseline, 98% improved) are historical
reference values. New CNN results are **PyTorch reruns**, not the same fitted
TensorFlow weights. Kernel initialization, data order and augmentation
implementation differ. The A1 saved split is retained, not recreated.

## Quick start on the author's computer

Open `Run_Digit_Lab.cmd`, keep its terminal open, and open
<http://127.0.0.1:7860> in Chrome. The local delivery includes the model weights
and feature bank. No model API key is used.

## Reproduce on another computer

Python 3.12 is recommended. From the project folder:

```sh
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python digit_pipeline.py --data "/path/to/ass comp vision"
python make_figures.py
python app.py
```

The input folder must contain the existing A1 `split_dataset/train`, `val`, and
`test` directories, each with class subdirectories `0` to `9`. Images must already
be grayscale 28×28. Expected counts: 800/100/100, half own and half downloaded.
Training writes only to `artifacts/`; the source images are not modified. The
first run downloads `facebook/dinov2-small` from Hugging Face. Later app runs load
the saved local backbone. A CPU is sufficient; CPU training/feature extraction
can take several minutes. A compatible CUDA PyTorch installation can use a GPU.

Public source control excludes the original dataset, cached image arrays,
feature bank and large pretrained backbone. Use your A1 dataset to reproduce
these, or use the complete local submission bundle supplied with the report.
Do not substitute MNIST and claim the results correspond to the A1 dataset.

## Notebook in Colab

Upload `Untitled2_Assignment1_and_2.ipynb` to Colab. The A2 boundary has a large
heading. Start with the **Assignment 2** section when the existing A1 split is
already in Drive. Its first cells mount Drive, install dependencies, and write
the project's modules into `/content/assignment2_run`. The training cell uses
`/content/drive/MyDrive/ass comp vision/split_dataset`. It does not require A1's
old in-memory TensorFlow variables. Run A2 cells in order. App display uses
Colab's port window, not a public tunneling service. Download the results ZIP
before disconnecting the runtime.

## Experimental protocol

- Fixed seed 42; exact file paths and pixel SHA-256 hashes recorded in the manifest.
- Fail on exact image duplicates across splits; no pixel-identical duplicates found in the measured run.
- CNN baseline: Conv16 → pool → Dense32 → ten logits; Adam, 15 epochs.
- CNN improved: Conv32 → pool → Conv64 → pool → dropout → Dense64 → dropout → ten logits; affine augmentation, maximum 30 epochs, early stopping patience 5.
- DINOv2: frozen encoder, resize to 224×224, grayscale replicated to RGB, ImageNet normalization, 384-dimensional CLS representation.
- Standardization fitted on original training features only. Linear head trained with full-batch AdamW, lr 0.01, weight decay 0.01, maximum 200 epochs, patience 20.
- Ablation: original 800 training images versus those images plus 1,600 affine variants. Same seed, head initialization, optimizer, maximum updates and validation set. No flipping and no val/test augmentation.
- Choose one of the two DINOv2 heads by minimum validation cross-entropy **before** test evaluation.
- Report all predefined models on the 100-image test set. No test-driven hyperparameter tuning.
- Latency includes tensor conversion, resize/normalization and model execution. It excludes disk reads, upload and retrieval; 30 batch-one measurements after 5 warm-ups.

An augmentation ablation estimates the effect of that whole training treatment;
it does not prove a universal benefit. The small test set and one seed limit
confidence. Original A1 exploration already exposed test performance, so this
is a continuation comparison, not an untouched external benchmark.

## App details

Draw/upload one digit on a plain background, or click a prepared test example.
Photo mode uses Otsu thresholding, largest-contour cropping, square padding and
28×28 resizing. Processed mode preserves a 28×28 bright digit on a dark background.
This distinction prevents applying photo preprocessing twice to dataset images.

The app shows five nearest **training** images in DINOv2 feature space, whether
the selected classifier is CNN or DINOv2. Neighbor similarity is not a causal
explanation of a CNN decision. Scores are uncalibrated softmax outputs. The app
is not an out-of-distribution detector, and letters or multiple digits may be
misclassified confidently.

Uploads are processed in memory and are not saved. The Flask server binds to
127.0.0.1 only. Do not expose the development server to the internet.

## Data provenance and ethics

The A1 dataset has 500 own digit images and 500 downloaded images. The available
materials do not establish the downloaded collection's precise URLs, license,
or writer consent. These must be checked before redistributing the dataset.
Raw images are excluded from Git. No faces or identity labels are used. No
generative model produces training images: the 1,600 extra samples are labeled
affine transformations, representing 66.7% of the augmented 2,400-image training
pool. Validation and test remain real and unchanged.

## References

- Oquab et al., DINOv2: <https://arxiv.org/abs/2304.07193>
- Official implementation: <https://github.com/facebookresearch/dinov2>
- Official converted weights: <https://huggingface.co/facebook/dinov2-small>
- Neurohive background on the DINO family (DINO 2021, not a DINOv2 announcement): <https://neurohive.io/en/novosti/dino-en/>

DINOv2 is a published foundation-model method used under the assignment's
alternative-modern-method option. This project does not claim a new SOTA on
handwritten digits, or that DINOv2 is the newest model available in 2026.


## Interface update / Обновление интерфейса

Image preparation is automatic; there is no manual preparation selector. Opaque grayscale 28×28 PNGs with a dark border preserve their pixels; photos and drawings are cropped and resized. The technical environment/privacy panel has been removed; experiment environment metadata remains in artifacts/environment.json.

`test_images/0` through `test_images/9` contain ten MNIST test PNGs per digit for additional demonstrations. Source indices and hashes are in `test_images/source.json`. They were not added to training or the reported benchmark. Exact pixel duplicates of existing A1 splits were excluded; writer independence is not established. The observed MNIST 3→6 error is an example of imperfect transfer, not part of the original test accuracy.

Для демонстрации: выбери PNG из test_images и нажми Recognize digit. Режим обработки выбирается автоматически. Ошибки на новых изображениях возможны.
