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
- `artifacts/metrics.json`, histories, prediction probabilities, manifest and environment: experiment evidence.

The original A1 TensorFlow test results (90% baseline, 98% improved) are historical
reference values. New CNN results are **PyTorch reruns**, not the same fitted
TensorFlow weights. Kernel initialization, data order and augmentation
implementation differ. The A1 saved split is retained, not recreated.

## Start here: what to download

**To run the trained app without retraining, use `Assignment2_Ready_To_Run.zip`, the complete companion archive intended to accompany the Moodle submission.** Extract it and open the `Assignment2` folder containing `app.py`. The archive is separate from GitHub's **Code > Download ZIP**.

The GitHub repository is the source/evidence distribution. It intentionally excludes the large pretrained backbone, cached training-image features and original dataset. A GitHub clone alone cannot run the full demo. If the companion archive has not been provided with the submission, request it from the author; there is no automatic download link configured in this repository.

The complete archive includes:
- application code and saved CNN/DINOv2 classifier weights;
- `artifacts/dinov2_backbone/model.safetensors` and its configuration;
- `artifacts/feature_bank.npz` for the nearest-neighbor gallery;
- `artifacts/dataset.npz` for the prepared example buttons;
- metrics, selection metadata, figures and the report;
- `test_images/0` through `test_images/9`: ten additional MNIST PNGs per digit;
- `dataset/split_dataset/{train,val,test}/{0,...,9}`: the original A1 split for optional reproduction.

No author's absolute filesystem paths are required. Python itself and installed libraries are not included.

## Run the trained application (Windows PowerShell)

Prerequisites: **64-bit Python 3.12**, internet for installing packages, and the extracted complete archive. A CPU is sufficient; no API key, GPU or training is required. These pinned dependencies were tested on Windows with Python 3.12; Python 3.14 is not supported by this setup.

Open PowerShell in the extracted `Assignment2` folder (the folder containing `app.py`), then run:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe app.py
```

Alternatively, double-click `Run_Digit_Lab.cmd`; it performs the same setup and starts the server. Keep the terminal open and wait for `Running on http://127.0.0.1:7860`, then open <http://127.0.0.1:7860> in a browser. Stop the server with Ctrl+C.

For subsequent launches:

```powershell
.\.venv312\Scripts\python.exe app.py
```

Do not use a bare `py app.py` unless its interpreter has the required packages. Environment activation is not necessary for the explicit commands above. Model weights load from the archive, so after dependencies are installed, recognition does not require a model download.

## macOS / Linux

Use Python 3.12 in the complete archive's `Assignment2` folder:

```sh
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -r requirements.txt
.venv312/bin/python app.py
```

Then open <http://127.0.0.1:7860>. The recorded experiment and application checks were performed on Windows; these other operating systems were not independently tested.

## Test the application

1. Click **Choose file** and upload a PNG from `test_images/0` to `test_images/9`. The folder name is the correct digit.
2. Select DINOv2 or Baseline CNN and click **Recognize digit**.
3. Inspect the predicted digit, ten class scores, inference time and five nearest training images.
4. Select the other model and repeat with the same input.
5. Click **Draw**, draw one large digit, then recognize it.

Preparation is automatic: opaque grayscale 28-by-28 images with a dark border preserve their pixels; other images use thresholding, cropping, padding and resizing. No manual preparation selector is needed. Device/compute/browser information is shown in the footer.

The additional MNIST examples are for demonstration, not the reported A1 test set. Their source indices and hashes are in `test_images/source.json`. They were not added to training. Exact pixel duplicates of the existing A1 splits were excluded, but writer independence is not established. Errors on new handwriting are possible; a high class score is not a guarantee. The nearest-neighbor gallery uses DINOv2 features even when CNN is selected.

To run the supplied integration checks (complete archive required):

```powershell
.\.venv312\Scripts\python.exe test_app.py
```

The checks exercise both classifiers, compare six predictions with saved experiment probabilities, check the gallery and reject invalid/blank inputs. A successful run prints `"status": "passed"`.

## Optional: reproduce training

**Skip this section if you only want to demonstrate the saved application.** Save an untouched copy of the complete archive first: training replaces files in `artifacts/`.

The complete archive includes the exact original split in `dataset/split_dataset`. With the environment installed, run from the project folder:

```powershell
.\.venv312\Scripts\python.exe digit_pipeline.py --data ".\dataset"
.\.venv312\Scripts\python.exe make_figures.py
.\.venv312\Scripts\python.exe app.py
```

If starting from a GitHub clone, first install the environment using the same commands above and obtain the original A1 data from the companion archive/author. Copy its `dataset` directory into the project or replace `.\dataset` with your own data-root path. The root must contain `split_dataset/train`, `split_dataset/val`, and `split_dataset/test`, each containing digit folders `0` through `9`. Expected counts are 800/100/100 grayscale 28-by-28 images, with the original file names and source split preserved. Do not substitute the 100 demonstration MNIST images and claim to reproduce the original benchmark.

Without a local backbone, training downloads `facebook/dinov2-small` from Hugging Face. Feature extraction and training may take several minutes or longer on CPU. The seed is fixed, but exact metrics can vary with software/hardware. Training does not modify the source image files.

## Troubleshooting

- **Python 3.12 not found:** install 64-bit Python 3.12 with the Windows Python launcher, then reopen PowerShell.
- **No module named flask/torch:** install requirements using the exact `.venv312` Python command above, then run with that same interpreter.
- **No matching distribution for torch:** check `.\.venv312\Scripts\python.exe --version`; use Python 3.12, not 3.14.
- **Missing feature_bank.npz, dataset.npz or backbone:** you likely downloaded only the GitHub source ZIP. Obtain the complete companion archive or reproduce training using the A1 data.
- **Address already in use:** another copy may already be running; stop that copy with Ctrl+C, or open the existing app at the shown address.
- **First prediction is slow:** the model loads on first use. Wait for it to finish before recording demo timings.

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


