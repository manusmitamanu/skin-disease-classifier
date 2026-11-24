import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # fixed order = label encoding
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 224


def load_split(project_root, split_csv="data/processed/split_v1.csv"):
    """Return train/val/test dataframes with absolute paths and integer labels."""
    df = pd.read_csv(os.path.join(project_root, split_csv))
    df["abs_path"] = df["path"].apply(lambda p: os.path.join(project_root, p))
    df["label"] = df["dx"].map(CLASS_TO_IDX)
    return (df[df.split == "train"].reset_index(drop=True),
            df[df.split == "val"].reset_index(drop=True),
            df[df.split == "test"].reset_index(drop=True))


def _load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)           # forces RGB
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))   # full image, no crop (aspect ratio is squashed)
    img = tf.cast(img, tf.float32)                     # stays in 0-255; model-specific preprocessing comes later
    return img, label


def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=25.0)       # on a 0-255 scale
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


def _geometric_aug():
    # Mild geometric augmentation. Negative zoom = zoom IN. Reflect fill avoids black corners.
    return keras.Sequential([
        keras.layers.RandomRotation(0.25, fill_mode="reflect"),            # up to +/-90 degrees
        keras.layers.RandomZoom(height_factor=(-0.2, 0.1), fill_mode="reflect"),
        keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
    ])


def make_dataset(df, batch_size=32, training=False, seed=42, augment="v1"):
    ds = tf.data.Dataset.from_tensor_slices((df["abs_path"].values, df["label"].values.astype("int32")))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)   # flips, brightness, contrast (same as v1)
    ds = ds.batch(batch_size)
    if training and augment == "v2":
        geo = _geometric_aug()
        ds = ds.map(lambda x, y: (tf.clip_by_value(geo(x, training=True), 0.0, 255.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)