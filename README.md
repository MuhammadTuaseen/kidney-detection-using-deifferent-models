# Kidney CT Scan Classification and Stone Detection

This project compares deep-learning approaches for analysing kidney CT images. The accompanying Jupyter notebook trains image classifiers for four kidney conditions and includes a separate YOLOv8 workflow for kidney-stone object detection.

> **Research use only.** This project is not a clinical diagnostic tool and must not be used to make medical decisions.

## What is included

The notebook, [`kidney_detection_using_different_models.ipynb`](./kidney_detection_using_different_models.ipynb), contains:

- a custom convolutional neural network (CNN);
- transfer-learning classifiers based on VGG16 and MobileNetV2;
- four-class classification of **Cyst**, **Normal**, **Stone**, and **Tumor** CT images;
- loss/accuracy plots and single-image prediction examples; and
- a separate YOLOv8 training section for annotated kidney-stone images.

## Datasets

Two Kaggle datasets are used in the notebook:

| Task | Dataset |
| --- | --- |
| CT image classification | [`nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone`](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) |
| YOLOv8 object detection | [`safurahajiheidari/kidney-stone-images`](https://www.kaggle.com/datasets/safurahajiheidari/kidney-stone-images) |

Please review each dataset's licence and terms of use before downloading or redistributing it. The datasets and trained model weights are not stored in this repository.

## Requirements

The notebook was written for Google Colab and Python 3. Install the required packages in a virtual environment or Colab runtime:

```bash
pip install tensorflow opencv-python matplotlib numpy pandas kaggle ultralytics squarify seaborn
```

For GPU training, use a TensorFlow build that is compatible with your platform and CUDA setup. A GPU-enabled Colab runtime is recommended for VGG16 and YOLOv8.

## Run the notebook

1. Clone this repository and open `kidney_detection_using_different_models.ipynb` in Jupyter or Google Colab.
2. Create a Kaggle API token from your Kaggle account settings and download `kaggle.json`.
3. In Colab, run the credential-upload cell and select `kaggle.json`. The notebook moves it to `~/.kaggle/kaggle.json` and sets the required permissions.
4. Run the dataset download and extraction cells for the workflow you want to use.
5. Run the model cells in order to train, plot learning curves, and make predictions.

The classification workflow expects the extracted dataset at:

```text
/content/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/
└── CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/
    ├── Cyst/
    ├── Normal/
    ├── Stone/
    └── Tumor/
```

The YOLOv8 section expects the second dataset to provide `train`, `valid`, and `test` image/label directories, plus a `data.yaml` file. Update the paths in the notebook if your extraction layout differs.

## Models

### Classification

All three classifiers use 150 × 150 RGB input images, normalized to the `[0, 1]` range. They are trained with sparse categorical cross-entropy over four output classes.

- **Custom CNN:** two convolution/max-pooling blocks followed by a dense classifier.
- **VGG16:** ImageNet-pretrained VGG16 feature extractor (frozen), followed by a dense classification head with batch normalization and dropout.
- **MobileNetV2:** ImageNet-pretrained MobileNetV2 feature extractor (frozen), followed by the same classification head pattern.

### Object detection

The notebook initializes `yolov8x.pt` pretrained weights and trains with Ultralytics YOLO using `data.yaml`, a learning rate of `0.001`, seed `42`, and `50` epochs. This is a distinct detection task; its results should not be directly compared with the four-class classification metrics.

## Notes and limitations

- The notebook currently uses different `validation_split` values for the training (`0.1`) and validation (`0.2`) datasets. For a reliable experiment, create one fixed train/validation split or use a predefined validation directory before reporting results.
- Training duration, accuracy, and detection quality depend on the dataset version, split, runtime, and random seed. No fixed benchmark results are claimed here.
- The notebook contains Colab-specific paths and shell commands. Adapt `/content/...` paths and the Kaggle credential setup when running locally.

## Repository structure

```text
.
├── kidney_detection_using_different_models.ipynb  # Training and evaluation workflows
└── README.md
```

## Licence

No licence file is currently included. Contact the repository owner before reusing the project code beyond what applicable law permits.
