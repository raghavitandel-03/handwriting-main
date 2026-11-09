import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "iam_10000"
)

IMAGE_DIR = (
    DATASET_DIR
    / "images"
)

LABEL_FILE = (
    DATASET_DIR
    / "labels.csv"
)


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256


# ============================================================
# DATASET CONFIGURATION
# ============================================================

BATCH_SIZE = 16

VALIDATION_SPLIT = 0.20

RANDOM_SEED = 42

SHUFFLE_BUFFER_SIZE = 500


# ============================================================
# VOCABULARY
# ============================================================

characters = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?'-\"#()*:;"
)


# ============================================================
# CHARACTER -> NUMBER
# ============================================================

char_to_num = tf.keras.layers.StringLookup(
    vocabulary=characters,
    mask_token=None
)


# ============================================================
# NUMBER -> CHARACTER
# ============================================================

num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)


# ============================================================
# READ LABELS
# ============================================================

def read_labels():

    print()
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Image directory:")
    print(IMAGE_DIR)

    print()
    print("Label file:")
    print(LABEL_FILE)

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"\nImage folder not found:\n{IMAGE_DIR}"
        )

    if not LABEL_FILE.exists():

        raise FileNotFoundError(
            f"\nLabel file not found:\n{LABEL_FILE}"
        )

    records = []

    skipped_missing = 0
    skipped_unreadable = 0

    with open(
        LABEL_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:

            raise ValueError(
                "labels.csv does not contain a header."
            )

        if "image" in reader.fieldnames:
            image_column = "image"

        elif "filename" in reader.fieldnames:
            image_column = "filename"

        else:
            raise ValueError(
                "Could not find image column."
            )

        if "text" in reader.fieldnames:
            label_column = "text"

        elif "label" in reader.fieldnames:
            label_column = "label"

        else:
            raise ValueError(
                "Could not find text/label column."
            )

        print()
        print(
            "Image column:",
            image_column
        )

        print(
            "Label column:",
            label_column
        )

        for row in reader:

            image_name = (
                row[image_column]
                .strip()
            )

            text = (
                row[label_column]
                .strip()
            )

            if not image_name:
                continue

            if not text:
                continue

            image_path = (
                IMAGE_DIR
                / image_name
            )

            # ------------------------------------------------
            # Missing image
            # ------------------------------------------------

            if not image_path.exists():

                skipped_missing += 1

                print(
                    "Skipping missing image:",
                    image_name
                )

                continue

            # ------------------------------------------------
            # Check image readability
            # ------------------------------------------------

            try:

                with Image.open(
                    image_path
                ) as test_image:

                    test_image.load()

                    if (
                        test_image.width <= 0
                        or
                        test_image.height <= 0
                    ):

                        raise ValueError(
                            "Invalid image dimensions."
                        )

            except Exception as error:

                skipped_unreadable += 1

                print(
                    "Skipping unreadable image:",
                    image_name
                )

                print(
                    "Reason:",
                    error
                )

                continue

            records.append(
                {
                    "image": str(image_path),
                    "text": text
                }
            )

    print()
    print(
        "Total valid records:",
        len(records)
    )

    print(
        "Skipped missing images:",
        skipped_missing
    )

    print(
        "Skipped unreadable images:",
        skipped_unreadable
    )

    if not records:

        raise ValueError(
            "No valid dataset records found."
        )

    return records


# ============================================================
# VOCABULARY CHECK
# ============================================================

def check_vocabulary(records):

    print()
    print("=" * 60)
    print("VOCABULARY CHECK")
    print("=" * 60)

    vocabulary_set = set(
        characters
    )

    unknown_characters = set()

    maximum_length = 0

    for record in records:

        text = record["text"]

        maximum_length = max(
            maximum_length,
            len(text)
        )

        for character in text:

            if character not in vocabulary_set:

                unknown_characters.add(
                    character
                )

    if unknown_characters:

        print(
            "Unknown characters:"
        )

        print(
            sorted(
                unknown_characters
            )
        )

        raise ValueError(
            "Dataset contains characters "
            "outside vocabulary."
        )

    print()
    print(
        "SUCCESS: All dataset characters "
        "are in the vocabulary."
    )

    print()
    print(
        "Maximum label length:",
        maximum_length
    )

    return maximum_length


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_and_preprocess_image(
    image_path
):

    try:

        image = Image.open(
            image_path
        ).convert("L")

    except Exception as error:

        raise RuntimeError(
            f"Could not read image:\n"
            f"{image_path}\n"
            f"{error}"
        )

    image = image.resize(
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        ),
        Image.Resampling.LANCZOS
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=-1
    )

    return image


# ============================================================
# ENCODE LABEL
# ============================================================

def encode_label(
    text,
    maximum_length
):

    characters_tensor = (
        tf.strings.unicode_split(
            text,
            "UTF-8"
        )
    )

    label = char_to_num(
        characters_tensor
    )

    label = label.numpy()

    padded = np.zeros(
        maximum_length,
        dtype=np.int32
    )

    length = min(
        len(label),
        maximum_length
    )

    padded[:length] = (
        label[:length]
    )

    return padded, length


# ============================================================
# CREATE ARRAYS
# ============================================================

def create_arrays(
    records,
    maximum_length
):

    images = []
    labels = []
    label_lengths = []

    for record in records:

        image = load_and_preprocess_image(
            record["image"]
        )

        label, length = encode_label(
            record["text"],
            maximum_length
        )

        images.append(image)
        labels.append(label)
        label_lengths.append(length)

    images = np.asarray(
        images,
        dtype=np.float32
    )

    labels = np.asarray(
        labels,
        dtype=np.int32
    )

    label_lengths = np.asarray(
        label_lengths,
        dtype=np.int64
    )

    return (
        images,
        labels,
        label_lengths
    )


# ============================================================
# CREATE TF DATASET
# ============================================================

def create_tf_dataset(
    images,
    labels,
    label_lengths,
    training=False
):

    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "image": images,
            "label": labels,
            "label_length": label_lengths
        }
    )

    if training:

        dataset = dataset.shuffle(
            SHUFFLE_BUFFER_SIZE,
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# PREPARE DATASETS
# ============================================================

def prepare_datasets():

    records = read_labels()

    maximum_label_length = (
        check_vocabulary(
            records
        )
    )

    # --------------------------------------------------------
    # Reproducible split
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    indices = np.arange(
        len(records)
    )

    rng.shuffle(
        indices
    )

    records = [
        records[i]
        for i in indices
    ]

    # --------------------------------------------------------
    # Validation split
    # --------------------------------------------------------

    validation_count = int(
        len(records)
        * VALIDATION_SPLIT
    )

    validation_records = (
        records[
            :validation_count
        ]
    )

    training_records = (
        records[
            validation_count:
        ]
    )

    print()
    print(
        "Training samples:",
        len(training_records)
    )

    print(
        "Validation samples:",
        len(validation_records)
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print(
        "Preparing training data..."
    )

    (
        train_images,
        train_labels,
        train_lengths
    ) = create_arrays(
        training_records,
        maximum_label_length
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print(
        "Preparing validation data..."
    )

    (
        val_images,
        val_labels,
        val_lengths
    ) = create_arrays(
        validation_records,
        maximum_label_length
    )

    # --------------------------------------------------------
    # TF datasets
    # --------------------------------------------------------

    train_dataset = create_tf_dataset(
        train_images,
        train_labels,
        train_lengths,
        training=True
    )

    validation_dataset = create_tf_dataset(
        val_images,
        val_labels,
        val_lengths,
        training=False
    )

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    print()
    print(
        "===== DATASET CHECK ====="
    )

    sample = next(
        iter(train_dataset)
    )

    print(
        "Image shape:",
        sample["image"].shape
    )

    print(
        "Label shape:",
        sample["label"].shape
    )

    print(
        "Label length shape:",
        sample["label_length"].shape
    )

    print(
        "Image dtype:",
        sample["image"].dtype
    )

    print(
        "Label dtype:",
        sample["label"].dtype
    )

    print(
        "Label length dtype:",
        sample["label_length"].dtype
    )

    print(
        "===== DATASET READY ====="
    )

    return (
        train_dataset,
        validation_dataset,
        maximum_label_length,
        len(training_records),
        len(validation_records)
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    (
        train_dataset,
        validation_dataset,
        maximum_label_length,
        training_count,
        validation_count
    ) = prepare_datasets()

    print()
    print(
        "Dataset loaded successfully."
    )

    print(
        "Maximum label length:",
        maximum_label_length
    )

    print(
        "Training samples:",
        training_count
    )

    print(
        "Validation samples:",
        validation_count
    )