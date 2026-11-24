# Skin Disease Classifier :p

A student project that classifies dermatoscopic skin-lesion images into the 7 HAM10000 classes using a fine-tuned EfficientNetB0, with Grad-CAM heatmaps, a melanoma alert threshold, and an out-of-distribution (OOD) check, served through a small Flask web app.

> **Not a medical device.** This is an educational project. It has not been clinically validated and must not be used to make health decisions. If you are worried about a spot on your skin, see a doctor.

This is a 7-class image classification experiment, not a diagnostic or screening system. The melanoma alert is a threshold on a model probability and has not been medically validated.

## What it does

- Takes a **dermatoscope** close-up image and returns the most likely of 7 classes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`) with class probabilities.
- Raises a **melanoma alert** when the melanoma probability is at least 0.25 (a threshold chosen on the validation set, deliberately low to miss fewer melanomas at the cost of more false alarms).
- Shows a **Grad-CAM** map of where the model looked. This shows what supported the predicted class, not proof that the reasoning is correct.
- Refuses images that look too different from the training images (**OOD gate**). The gate is imperfect in both directions (see Limitations).
- Does not store uploaded images.

## Results

All test numbers come from a single evaluation of the final model on a held-out, lesion-level test split (1,481 images, 1,121 lesions). The test set was not used for model, threshold or cutoff selection. Confidence intervals are 95% percentile intervals from 1,000 lesion-level bootstrap resamples.

| Metric (test) | Estimate | 95% CI |
|---|---|---|
| Accuracy | 0.772 | 0.746 to 0.797 |
| Balanced accuracy | 0.615 | 0.555 to 0.674 |
| Macro F1 | 0.587 | 0.525 to 0.637 |
| Melanoma recall at alert threshold 0.25 | 0.612 | 0.527 to 0.699 |
| Melanoma precision at alert threshold 0.25 | 0.331 | 0.262 to 0.401 |
| `df` recall | 0.450 | 0.214 to 0.737 |
| `vasc` recall | 0.789 | 0.565 to 0.955 |

Weighted F1 on test is 0.779 (no interval computed). At the 0.25 alert threshold, 305 of 1,481 test images were flagged.

Per-class test results (argmax prediction):

| Class | Precision | Recall | Test images | ROC-AUC (one-vs-rest) |
|---|---|---|---|---|
| akiec | 0.375 | 0.522 | 46 | 0.953 |
| bcc | 0.473 | 0.620 | 71 | 0.957 |
| bkl | 0.541 | 0.548 | 168 | 0.902 |
| df | 0.500 | 0.450 | 20 | 0.964 |
| mel | 0.468 | 0.491 | 165 | 0.847 |
| nv | 0.933 | 0.885 | 992 | 0.944 |
| vasc | 0.682 | 0.789 | 19 | 0.975 |

`df` and `vasc` rest on about 20 test images each, so their numbers are very noisy.

### Experiments (validation set, single runs)

| Model | Accuracy | Balanced acc. | Macro F1 | mel recall / precision |
|---|---|---|---|---|
| MobileNetV2 frozen, full class weights | 0.651 | 0.597 | 0.469 | 0.541 / 0.335 |
| MobileNetV2 frozen, sqrt class weights | 0.759 | 0.536 | 0.535 | 0.488 / 0.454 |
| MobileNetV2 fine-tuned, sqrt weights | 0.780 | 0.584 | 0.565 | 0.471 / 0.537 |
| MobileNetV2 fine-tuned + stronger augmentation | 0.693 | 0.544 | 0.506 | 0.506 / 0.398 |
| EfficientNetB0 frozen, sqrt weights | 0.751 | 0.587 | 0.588 | 0.441 / 0.436 |
| **EfficientNetB0 fine-tuned, sqrt weights (final)** | 0.774 | 0.631 | 0.610 | 0.459 / 0.448 |

These are single runs, so differences under about 0.02 macro F1 should be treated as inconclusive. The machine-readable table is `results/experiment_log.csv`.

### Melanoma alert threshold (validation only)

| Threshold | Flagged | Recall | Precision |
|---|---|---|---|
| 0.10 | 40.3% | 0.835 | 0.230 |
| 0.15 | 31.9% | 0.729 | 0.254 |
| **0.25 (used)** | 20.6% | 0.600 | 0.323 |
| 0.35 | 14.3% | 0.512 | 0.397 |

The validation set has 170 melanomas, so recall carries roughly +/-0.07 of sampling noise and neighbouring thresholds are not clearly different. The full sweep is in `results/val_mel_threshold_sweep.csv`. Precision depends on the melanoma prevalence in HAM10000 (about 11%) and would be lower in a population with fewer melanomas.

## Data and split

- Dataset: HAM10000 (10,015 dermatoscopic images, 7 classes, 7,470 unique lesions). Images are **not** included in this repository.
- The split is **lesion-level** and stratified (70/15/15, seed 42), so images of the same lesion never appear in more than one split. Image counts: train 7,002, validation 1,532, test 1,481.
- There is no patient ID in the metadata, so leakage across different lesions of the same patient cannot be ruled out.
- Leakage exposure estimate: under a random **image-level** split, about 35.0% of test images (61.8% of melanoma test images) would have a same-lesion image in the training set. This measures exposure to leakage, **not** the resulting inflation of F1. See `results/leakage_exposure_image_level_split.csv`.

## Built with

Python 3.12, TensorFlow / Keras (EfficientNetB0 final model; MobileNetV2 used in comparison experiments), NumPy, pandas, scikit-learn, OpenCV and Pillow, Matplotlib and seaborn (notebooks), Flask, and plain HTML, CSS and JavaScript for the web page.

## Run the app

Requires Python 3.12 and a CPU is enough (no GPU needed).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app/app.py
```

Then open http://127.0.0.1:5000. The app needs `models/final_efficientnetb0.keras`, `models/final_config.json` and `models/ood_stats.npz`. It uses Flask's development server and is meant for local use only.

## Reproduce the experiments

1. Download HAM10000 (for example the "Skin Cancer MNIST: HAM10000" dataset on Kaggle, or the official Harvard Dataverse release) and check its license terms.
2. Arrange it under `data/raw/` like this:
   ```
   data/raw/
     HAM10000_images_part_1/     (JPG images)
     HAM10000_images_part_2/     (JPG images)
     HAM10000_metadata.csv
   ```
3. Keep `data/processed/split_v1.csv` as provided. It defines the frozen lesion-level split.
4. Run the notebooks in order (`01` to `08`). Training was done on CPU, and the larger runs take hours.

## Project structure

```
app/                 Flask app (app.py, templates/index.html)
src/data.py          data loading, split loading, tf.data pipeline, augmentation
notebooks/           01 EDA ... 07 OOD gate, 08 analysis (log, bootstrap, leakage, threshold)
models/              final EfficientNetB0, config, OOD statistics
results/             metrics and analysis tables (CSV/JSON)
data/processed/      frozen split (split_v1.csv)
tools/               pre-push check and notebook path cleanup scripts
```

## Limitations

- Strong class imbalance (nv is 66.9% of images). Melanoma is the hardest class: recall never exceeded about 0.54 at argmax in any setup.
- `df` and `vasc` have very few examples, so their results are unreliable.
- No patient ID; labels come from mixed sources (histopathology, follow-up, expert consensus, confocal).
- Images are resized to 224x224 and the 4:3 aspect ratio is squashed.
- Results are single training runs; the confidence intervals capture test-sample variation only, not training variability.
- The validation set was reused for model choice, melanoma threshold and OOD cutoff selection.
- The OOD gate (Mahalanobis distance on pooled features, cutoff = validation 99th percentile) is imperfect. In a check on 46 external images it rejected non-skin images and phone photos but also wrongly rejected some real dermatoscope images. Only about 5 to 6 real external dermatoscope images were available, so no reliable error rate exists.
- The classifier fails on phone and internet photos. It was designed only for dermatoscope images.
- The "Lower / Moderate / Elevated concern" label is a simple grouping rule of this project, not a model output, and "Lower" does not mean safe. The "low confidence" flag (top probability below 0.5) is an untested heuristic.
- Grad-CAM evidence is qualitative.
- HAM10000 is a single data source. No analysis of skin tone or demographic coverage was done.
- Not clinically validated.

## Data and citation

Please credit the dataset and check its license terms (it is distributed for non-commercial use) before reusing it:

- Tschandl, P., Rosendahl, C., Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data 5, 180161 (2018).
- Tan, M., Le, Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.

## License

Code: MIT License (see `LICENSE`). The model weights were trained on HAM10000, which is distributed for non-commercial use, so treat the weights as subject to the dataset's terms and check them before any commercial use. This is not legal advice.