import zipfile
import pandas as pd
import tensorflow as tf
from io import BytesIO
import random

def load_datasets_from_zip(zip_path, img_size=(299, 299), batch_size=32, split_ratio=0.8, shuffle_buffer=512):
    """
    Loads and prepares training and validation datasets from a zip file.
    
    Args:
        zip_path (str): Path to the .zip file containing images and manifest.csv
        img_size (tuple): Target image size (height, width)
        batch_size (int): Batch size for training
        split_ratio (float): Ratio of data to use for training (rest is validation)
        shuffle_buffer (int): Buffer size for shuffling dataset

    Returns:
        train_ds, val_ds, label_to_index (dict), index_to_label (dict), train_labels (List[str])
    """

    with zipfile.ZipFile(zip_path, 'r') as archive:
        # Read the manifest
        with archive.open('manifest.csv') as f:
            df = pd.read_csv(f)
            filenames = df['fname'].tolist()
            labels = df['label'].tolist()
            full_path = (df['label'] + '/' + df['fname']).tolist()
        
        # Read image bytes into memory
        image_data = {name: archive.read(f"{name}") for name in full_path}

    # Encode labels
    label_to_index = {label: i for i, label in enumerate(sorted(set(labels)))}
    index_to_label = {i: label for label, i in label_to_index.items()}

    # Shuffle and split
    data = list(zip(filenames, labels))
    random.shuffle(data)
    split_idx = int(split_ratio * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_labels = [label for _, label in train_data]  # New: list of training labels

    def preprocess_image(image_bytes):
        img = tf.image.decode_png(image_bytes, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def make_dataset(data_subset):
        def generator():
            for filename, label in data_subset:
                image_bytes = image_data[label + '/' + filename]
                yield preprocess_image(image_bytes), label_to_index[label]

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(img_size[0], img_size[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )

    train_ds = make_dataset(train_data).shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = make_dataset(val_data).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, label_to_index, index_to_label, train_labels