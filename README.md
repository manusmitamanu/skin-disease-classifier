# Skin Disease Classifier :p

A student deep learning project that classifies dermatoscopic skin images into 7 categories using the HAM10000 dataset.

The project uses a fine-tuned EfficientNetB0 model and includes Grad-CAM heatmaps, a melanoma alert threshold and an out-of-distribution (OOD) check. A Flask web application provides access to the model.

> **Not a medical device.** This is an educational project. It has not been clinically validated and must not be used to make health decisions. If you are concerned about a skin spot, consult a doctor.

This project performs 7-class image classification. It is not a diagnostic or screening system. The melanoma alert is based on a model probability threshold and has not been medically validated.

## What it does

- Accepts close-up images taken using a **dermatoscope** and predicts one of 7 classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv` and `vasc`.
- Returns the probability for each class.
- Shows a **melanoma alert** when the melanoma probability is 0.25 or higher. This threshold was selected using the validation set. It is deliberately low to reduce missed melanomas, but this also causes more false alarms.
- Generates a **Grad-CAM heatmap** showing which image areas supported the predicted class. It does not prove that the model's reasoning is correct.
- Uses an **OOD gate** to reject images that look too different from the training images. This check can make mistakes in both directions.
- Does not store uploaded images.

## Results

The final model was evaluated once on a separate test set containing 1,481 images from 1,121 lesions.

The test set was not used to select the model, melanoma threshold or OOD cutoff.

The 95% confidence intervals below were calculated using 1,000 lesion-level bootstrap samples. These are percentile confidence intervals.

### Overall test performance

| Metric | Score | 95% Confidence Interval |
|---|---:|---:|
| Accuracy | 0.772 | 0.746–0.797 |
| Balanced Accuracy | 0.615 | 0.555–0.674 |
| Macro F1 | 0.587 | 0.525–0.637 |
| Melanoma Recall (threshold 0.25) | 0.612 | 0.527–0.699 |
| Melanoma Precision (threshold 0.25) | 0.331 | 0.262–0.401 |
| `df` Recall | 0.450 | 0.214–0.737 |
| `vasc` Recall | 0.789 | 0.565–0.955 |

The test Weighted F1 score was 0.779. No confidence interval was calculated for this metric.

At the melanoma threshold of 0.25, 305 out of 1,481 test images were flagged.

### Per-class test performance

The following results use the model's highest-probability class as its prediction (argmax).

| Class | Precision | Recall | Test Images | ROC-AUC (One-vs-Rest) |
|---|---:|---:|---:|---:|
| akiec | 0.375 | 0.522 | 46 | 0.953 |
| bcc | 0.473 | 0.620 | 71 | 0.957 |
| bkl | 0.541 | 0.548 | 168 | 0.902 |
| df | 0.500 | 0.450 | 20 | 0.964 |
| mel | 0.468 | 0.491 | 165 | 0.847 |
| nv | 0.933 | 0.885 | 992 | 0.944 |
| vasc | 0.682 | 0.789 | 19 | 0.975 |

The `df` and `vasc` results are less reliable because each class has only around 20 test images. Their scores can change considerably with such small sample sizes.

### Model experiments (validation set)

I tested MobileNetV2 and EfficientNetB0 with different training settings. Each experiment was run once.

| Model | Accuracy | Balanced Accuracy | Macro F1 | Melanoma Recall / Precision |
|---|---:|---:|---:|---:|
| MobileNetV2 frozen, full class weights | 0.651 | 0.597 | 0.469 | 0.541 / 0.335 |
| MobileNetV2 frozen, sqrt class weights | 0.759 | 0.536 | 0.535 | 0.488 / 0.454 |
| MobileNetV2 fine-tuned, sqrt weights | 0.780 | 0.584 | 0.565 | 0.471 / 0.537 |
| MobileNetV2 fine-tuned + stronger augmentation | 0.693 | 0.544 | 0.506 | 0.506 / 0.398 |
| EfficientNetB0 frozen, sqrt weights | 0.751 | 0.587 | 0.588 | 0.441 / 0.436 |
| **EfficientNetB0 fine-tuned, sqrt weights (final)** | **0.774** | **0.631** | **0.610** | **0.459 / 0.448** |

Since each experiment was run only once, differences smaller than about 0.02 in Macro F1 may not be meaningful.

The complete experiment results are saved in `results/experiment_log.csv`.

### Melanoma alert threshold (validation set)

The melanoma threshold determines when the application raises an alert.

A lower threshold flags more images and catches more melanoma cases, but also produces more false alarms.

| Threshold | Images Flagged | Recall | Precision |
|---|---:|---:|---:|
| 0.10 | 40.3% | 0.835 | 0.230 |
| 0.15 | 31.9% | 0.729 | 0.254 |
| **0.25 (used)** | **20.6%** | **0.600** | **0.323** |
| 0.35 | 14.3% | 0.512 | 0.397 |

The validation set contained 170 melanoma images. Because of this sample size, recall has approximately ±0.07 sampling uncertainty. The results also do not show a clear difference between neighbouring thresholds.

The full threshold analysis is saved in `results/val_mel_threshold_sweep.csv`.

Precision depends on how common melanoma is in the dataset. Melanoma makes up about 11% of HAM10000, so precision would be lower in a population where melanoma is less common.

## Data and split

This project uses the **HAM10000 dataset**, which contains:

- 10,015 dermatoscopic images.
- 7 skin lesion classes.
- 7,470 unique lesions.

The original images are not included in this repository.

### How the data was split

The dataset was divided at the **lesion level**, using stratification and random seed 42.

This ensures that images of the same lesion do not appear in more than one split.

| Split | Percentage | Images |
|---|---:|---:|
| Training | 70% | 7,002 |
| Validation | 15% | 1,532 |
| Testing | 15% | 1,481 |

The dataset metadata does not contain patient IDs. Therefore, it is not possible to rule out data leakage between different lesions belonging to the same patient.

### Data leakage analysis

An additional analysis estimated how much leakage could occur if the data were randomly split by image instead of by lesion.

Under a random image-level split:

- About 35.0% of test images could have another image of the same lesion in the training set.
- For melanoma test images, this estimate rises to 61.8%.

These numbers measure possible exposure to data leakage. They do not measure how much leakage would increase the model's F1 score.

The analysis is saved in `results/leakage_exposure_image_level_split.csv`.

## Built with

| Technology | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| TensorFlow / Keras | Deep learning and model training |
| EfficientNetB0 | Final classification model |
| MobileNetV2 | Model comparison experiments |
| NumPy and pandas | Data handling and processing |
| scikit-learn | Machine learning utilities and evaluation metrics |
| OpenCV and Pillow | Image processing |
| Matplotlib and seaborn | Data visualisation in notebooks |
| Flask | Web application backend |
| HTML, CSS and JavaScript | Web application frontend |

## Run the app

The application requires Python 3.12. It can run on a CPU, so a GPU is not required.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
python app/app.py
```

Open the following address in your browser:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

The application requires these three files:

- `models/final_efficientnetb0.keras`
- `models/final_config.json`
- `models/ood_stats.npz`

The app uses Flask's development server and is intended for local use only.

## Reproduce the experiments

Follow these steps to run the experiments again.

### 1. Download the dataset

Download HAM10000 from either:

- The "Skin Cancer MNIST: HAM10000" dataset on Kaggle.
- The official Harvard Dataverse release.

Check the dataset's licence terms before using it.

### 2. Arrange the dataset files

Place the downloaded files under `data/raw/` using this structure:

```text
data/raw/
├── HAM10000_images_part_1/     # JPG images
├── HAM10000_images_part_2/     # JPG images
└── HAM10000_metadata.csv
```

### 3. Keep the original data split

Keep `data/processed/split_v1.csv` unchanged. It contains the fixed lesion-level train, validation and test split used in this project.

### 4. Run the notebooks

Run notebooks in order, from `01` to `08`.

All training was performed on a CPU. The larger experiments can take several hours.

## Project structure

```text
skin-disease-classifier/
│
├── app/                 Flask app (app.py, templates/index.html)
├── src/data.py          Data loading, split loading, tf.data pipeline, augmentation
├── notebooks/           01 EDA ... 07 OOD gate, 08 analysis
│                        (experiment log, bootstrap, leakage, threshold)
├── models/              Final EfficientNetB0 model, config, OOD statistics
├── results/             Evaluation metrics and analysis tables (CSV/JSON)
├── data/processed/      Fixed lesion-level split (split_v1.csv)
├── tools/               Pre-push check and notebook path cleanup scripts
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

## Limitations

This project has several limitations that are important to understand.

- **Class imbalance:** The `nv` class makes up 66.9% of the dataset. This imbalance makes it harder for the model to perform equally well across all classes. Melanoma is the most difficult class in this project. Its recall at the default argmax prediction never exceeded approximately 0.54 in any experiment.
- **Small classes:** The `df` and `vasc` classes have very few images, so their results are less reliable.
- **Missing patient IDs:** The dataset does not provide patient IDs. Its labels also come from different sources, including histopathology, follow-up, expert consensus and confocal microscopy.
- **Image resizing:** All images are resized to 224 × 224 pixels. This changes their original 4:3 aspect ratio.
- **Single training runs:** Each experiment was run once. The confidence intervals measure variation in the test sample, not variation between repeated training runs.
- **Validation set reuse:** The validation set was used to select the model, melanoma threshold and OOD cutoff.
- **Imperfect OOD detection:** The OOD gate uses Mahalanobis distance calculated from pooled features. Its cutoff is the 99th percentile of the validation-set distances. It is not perfect. In a test of 46 external images, it rejected non-skin images and phone photos, but also incorrectly rejected some genuine dermatoscopic images. Only around 5–6 genuine external dermatoscopic images were available, which is not enough to calculate a reliable error rate.
- **Image compatibility:** The classifier is designed for dermatoscopic images. It does not work reliably with ordinary phone photos or internet images.
- **Concern labels:** The application displays "Lower", "Moderate" or "Elevated" concern using a simple rule. These labels are not direct model outputs. A "Lower" label does not mean that an image is safe.
- **Low-confidence flag:** The application marks predictions as low confidence when the highest class probability is below 0.5. This is an untested heuristic.
- **Grad-CAM:** Grad-CAM heatmaps provide visual explanations, but the evidence is qualitative. They cannot prove that the model is reasoning correctly.
- **Dataset coverage:** The project uses only one dataset. It does not include an analysis of skin tone or demographic coverage.
- **Clinical validation:** The model has not been clinically validated and must not be used for diagnosis or medical decisions.

## Data and citation

Please credit the original dataset researchers and check the dataset's licence terms before reusing it. HAM10000 is distributed for non-commercial use.

**HAM10000 dataset reference**

Tschandl, P., Rosendahl, C., Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data 5, 180161.

**EfficientNet architecture reference**

Tan, M., Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.

## License

The project code is licensed under the MIT License. See the `LICENSE` file for the full terms.

The model weights were trained using the HAM10000 dataset, which is distributed for non-commercial use. The dataset's terms may also apply to the trained model weights. Check those terms before any commercial use.

This is not legal advice.