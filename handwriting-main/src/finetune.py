import os
import csv
import shutil
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINETUNE_DIR = PROJECT_ROOT / "finetune"
IMAGE_DIR = FINETUNE_DIR / "my_handwriting"
CSV_FILE = FINETUNE_DIR / "my_labels.csv"

MODEL_DIR = PROJECT_ROOT / "models"

ORIGINAL_WEIGHTS = MODEL_DIR / "crnn_best.weights.h5"

BACKUP_WEIGHTS = MODEL_DIR / "crnn_before_finetune.weights.h5"

STAGE1_WEIGHTS = MODEL_DIR / "crnn_finetune_stage1_best.weights.h5"

FINAL_BEST_WEIGHTS = MODEL_DIR / "crnn_finetuned_best.weights.h5"

FINAL_WEIGHTS = MODEL_DIR / "crnn_finetuned_final.weights.h5"


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256

SEQUENCE_LENGTH = 32

NUM_CLASSES = 77

BATCH_SIZE = 8

VALIDATION_SPLIT = 0.20

RANDOM_SEED = 42


# ============================================================
# 3. SAME VOCABULARY AS ORIGINAL DATASET
# ============================================================

characters = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?'-\"#()*:;"
)

char_to_num = layers.StringLookup(
    vocabulary=characters,
    mask_token=None
)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# 5. CHECK FILES
# ============================================================

def check_required_files():

    print("\n" + "=" * 60)
    print("CHECKING PROJECT FILES")
    print("=" * 60)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"\nImage folder not found:\n{IMAGE_DIR}"
        )

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"\nCSV file not found:\n{CSV_FILE}"
        )

    if not ORIGINAL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"\nOriginal model weights not found:\n"
            f"{ORIGINAL_WEIGHTS}"
        )

    MODEL_DIR.mkdir(exist_ok=True)

    print("Image folder :", IMAGE_DIR)
    print("CSV file     :", CSV_FILE)
    print("Model        :", ORIGINAL_WEIGHTS)

    print("\nAll required files found.")


# ============================================================
# 6. READ CSV
# ============================================================

def read_csv():

    print("\n" + "=" * 60)
    print("READING FINE-TUNING CSV")
    print("=" * 60)

    samples = []

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")

        print("CSV columns:", reader.fieldnames)

        if "image" not in reader.fieldnames:
            raise ValueError(
                "CSV must contain an 'image' column."
            )

        if "text" not in reader.fieldnames:
            raise ValueError(
                "CSV must contain a 'text' column."
            )

        for row_number, row in enumerate(reader, start=2):

            image_name = row["image"].strip()
            text = row["text"]

            if text is None:
                text = ""

            text = text.strip()

            if not image_name:
                print(
                    f"WARNING: Empty image name at row {row_number}"
                )
                continue

            if not text:
                print(
                    f"WARNING: Empty text at row {row_number}: "
                    f"{image_name}"
                )
                continue

            image_path = IMAGE_DIR / image_name

            if not image_path.exists():
                print(
                    f"WARNING: Image not found at row {row_number}:"
                )
                print(f"         {image_name}")
                continue

            samples.append(
                {
                    "image": image_path,
                    "text": text
                }
            )

    print("\nValid CSV/image pairs:", len(samples))

    if len(samples) < 5:
        raise ValueError(
            "\nToo few valid fine-tuning samples."
        )

    return samples


# ============================================================
# 7. CHECK VOCABULARY + LABEL LENGTH
# ============================================================

def validate_labels(samples):

    print("\n" + "=" * 60)
    print("VALIDATING LABELS")
    print("=" * 60)

    vocabulary_set = set(characters)

    invalid_samples = []

    too_long_samples = []

    maximum_length = 0

    for sample in samples:

        text = sample["text"]

        maximum_length = max(
            maximum_length,
            len(text)
        )

        invalid_chars = [
            c for c in text
            if c not in vocabulary_set
        ]

        if invalid_chars:

            invalid_samples.append(
                (
                    sample["image"].name,
                    text,
                    sorted(set(invalid_chars))
                )
            )

        # Original model has only 32 CTC time steps.
        # Keep a safety margin.
        if len(text) >= SEQUENCE_LENGTH:

            too_long_samples.append(
                (
                    sample["image"].name,
                    text,
                    len(text)
                )
            )

    print("Maximum label length:", maximum_length)

    if invalid_samples:

        print("\nINVALID CHARACTERS FOUND:")

        for name, text, chars in invalid_samples[:20]:

            print(
                f"\nImage: {name}"
            )
            print(
                f"Text : {text}"
            )
            print(
                f"Chars: {chars}"
            )

        raise ValueError(
            "\nSome labels contain characters that are not "
            "present in the original CRNN vocabulary."
        )

    print("Vocabulary check: PASSED")

    if too_long_samples:

        print("\nWARNING:")
        print(
            f"{len(too_long_samples)} samples have "
            f"{SEQUENCE_LENGTH} or more characters."
        )

        print(
            "The current CRNN has only 32 CTC time steps."
        )

        print("\nExamples:")

        for name, text, length in too_long_samples[:10]:

            print(
                f"{name} -> {length} characters -> {text}"
            )

        print(
            "\nThese samples will NOT be used for fine-tuning."
        )

        # Remove samples that cannot fit the current model.
        samples = [
            s for s in samples
            if len(s["text"]) < SEQUENCE_LENGTH
        ]

    print(
        "\nSamples remaining after label validation:",
        len(samples)
    )

    if len(samples) < 5:

        raise ValueError(
            "\nToo few usable samples after label-length filtering."
        )

    return samples


# ============================================================
# 8. LOAD IMAGE
# ============================================================

def load_image(path):

    image = tf.io.read_file(str(path))

    image = tf.io.decode_image(
        image,
        channels=1,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [IMAGE_HEIGHT, IMAGE_WIDTH]
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = image / 255.0

    return image


# ============================================================
# 9. ENCODE LABEL
# ============================================================

def encode_label(text, max_length):

    characters_tensor = tf.strings.unicode_split(
        text,
        "UTF-8"
    )

    label = char_to_num(
        characters_tensor
    )

    label_length = tf.shape(label)[0]

    padding = max_length - label_length

    label = tf.pad(
        label,
        [[0, padding]]
    )

    label = tf.cast(
        label,
        tf.int32
    )

    return label, tf.cast(
        label_length,
        tf.int32
    )


# ============================================================
# 10. CREATE DATASET
# ============================================================

def create_dataset(samples, shuffle=False):

    max_length = max(
        len(s["text"])
        for s in samples
    )

    images = []
    labels = []
    label_lengths = []

    print(
        "\nPreparing",
        len(samples),
        "samples..."
    )

    for sample in samples:

        image = load_image(
            sample["image"]
        )

        label, label_length = encode_label(
            sample["text"],
            max_length
        )

        images.append(image.numpy())
        labels.append(label.numpy())
        label_lengths.append(
            label_length.numpy()
        )

    images = np.array(
        images,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.int32
    )

    label_lengths = np.array(
        label_lengths,
        dtype=np.int32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "image": images,
            "label": labels,
            "label_length": label_lengths
        }
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=len(samples),
            seed=RANDOM_SEED
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# 11. BUILD EXACT ORIGINAL CRNN
# ============================================================

def build_crnn():

    inputs = layers.Input(
        shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            1
        ),
        name="image"
    )

    # -------------------------
    # CNN BLOCK 1
    # -------------------------

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv1"
    )(inputs)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool1"
    )(x)

    # -------------------------
    # CNN BLOCK 2
    # -------------------------

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv2"
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool2"
    )(x)

    # -------------------------
    # CNN BLOCK 3
    # -------------------------

    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv3"
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool3"
    )(x)

    # -------------------------
    # CONVERT IMAGE TO SEQUENCE
    # -------------------------

    x = layers.Permute(
        (2, 1, 3),
        name="permute"
    )(x)

    x = layers.Reshape(
        (
            SEQUENCE_LENGTH,
            2048
        ),
        name="reshape"
    )(x)

    x = layers.Dense(
        256,
        activation="relu",
        name="dense"
    )(x)

    # -------------------------
    # BiLSTM 1
    # -------------------------

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True
        ),
        name="bidirectional"
    )(x)

    # -------------------------
    # BiLSTM 2
    # -------------------------

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True
        ),
        name="bidirectional_1"
    )(x)

    # -------------------------
    # OUTPUT
    # -------------------------

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="character_probabilities"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CRNN"
    )

    return model


# ============================================================
# 12. CTC LOSS
# ============================================================

def ctc_loss(
    labels,
    predictions,
    label_lengths
):

    batch_size = tf.shape(
        predictions
    )[0]

    input_length = tf.shape(
        predictions
    )[1]

    input_lengths = tf.ones(
        shape=(batch_size, 1),
        dtype=tf.int32
    ) * input_length

    label_lengths = tf.expand_dims(
        label_lengths,
        axis=1
    )

    loss = tf.keras.backend.ctc_batch_cost(
        labels,
        predictions,
        input_lengths,
        label_lengths
    )

    return tf.reduce_mean(loss)


# ============================================================
# 13. CUSTOM CTC MODEL
# ============================================================

class CRNNModel(keras.Model):

    def __init__(self, base_model):

        super().__init__()

        self.base_model = base_model

        self.loss_tracker = keras.metrics.Mean(
            name="loss"
        )

    @property
    def metrics(self):

        return [
            self.loss_tracker
        ]

    def train_step(self, data):

        images = data["image"]
        labels = data["label"]
        label_lengths = data["label_length"]

        with tf.GradientTape() as tape:

            predictions = self.base_model(
                images,
                training=True
            )

            loss = ctc_loss(
                labels,
                predictions,
                label_lengths
            )

        gradients = tape.gradient(
            loss,
            self.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.trainable_variables
            )
        )

        self.loss_tracker.update_state(
            loss
        )

        return {
            "loss": self.loss_tracker.result()
        }

    def test_step(self, data):

        images = data["image"]
        labels = data["label"]
        label_lengths = data["label_length"]

        predictions = self.base_model(
            images,
            training=False
        )

        loss = ctc_loss(
            labels,
            predictions,
            label_lengths
        )

        self.loss_tracker.update_state(
            loss
        )

        return {
            "loss": self.loss_tracker.result()
        }


# ============================================================
# 14. CALLBACK TO SAVE BEST MODEL
# ============================================================

class BestWeightsCallback(
    keras.callbacks.Callback
):

    def __init__(
        self,
        filepath
    ):

        super().__init__()

        self.filepath = filepath
        self.best = np.inf

    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        logs = logs or {}

        val_loss = logs.get(
            "val_loss"
        )

        if val_loss is None:
            return

        print(
            f"\nValidation loss: {val_loss:.6f}"
        )

        if val_loss < self.best:

            self.best = val_loss

            self.model.base_model.save_weights(
                self.filepath
            )

            print(
                f"Saved best weights -> {self.filepath}"
            )


# ============================================================
# 15. SPLIT DATA
# ============================================================

def split_samples(samples):

    random.shuffle(samples)

    validation_count = max(
        1,
        int(
            len(samples)
            * VALIDATION_SPLIT
        )
    )

    validation_samples = samples[
        :validation_count
    ]

    training_samples = samples[
        validation_count:
    ]

    print("\n" + "=" * 60)
    print("DATASET SPLIT")
    print("=" * 60)

    print(
        "Training samples  :",
        len(training_samples)
    )

    print(
        "Validation samples:",
        len(validation_samples)
    )

    return (
        training_samples,
        validation_samples
    )


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("HANDWRITING CRNN FINE-TUNING")
    print("=" * 60)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    check_required_files()

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    samples = read_csv()

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    samples = validate_labels(
        samples
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    training_samples, validation_samples = split_samples(
        samples
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = create_dataset(
        training_samples,
        shuffle=True
    )

    validation_dataset = create_dataset(
        validation_samples,
        shuffle=False
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BUILDING ORIGINAL CRNN")
    print("=" * 60)

    base_model = build_crnn()

    dummy_input = tf.zeros(
        (
            1,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            1
        )
    )

    dummy_output = base_model(
        dummy_input,
        training=False
    )

    print(
        "Model output shape:",
        dummy_output.shape
    )

    expected_shape = (
        1,
        SEQUENCE_LENGTH,
        NUM_CLASSES
    )

    if tuple(dummy_output.shape) != expected_shape:

        raise ValueError(
            f"\nWRONG MODEL OUTPUT!\n"
            f"Expected: {expected_shape}\n"
            f"Got     : {tuple(dummy_output.shape)}"
        )

    print(
        "Output class check: PASSED"
    )

    # --------------------------------------------------------
    # Load original model
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LOADING ORIGINAL MODEL")
    print("=" * 60)

    base_model.load_weights(
        ORIGINAL_WEIGHTS
    )

    print(
        "Original weights loaded successfully."
    )

    # --------------------------------------------------------
    # BACKUP ORIGINAL WEIGHTS
    # --------------------------------------------------------

    if not BACKUP_WEIGHTS.exists():

        shutil.copy2(
            ORIGINAL_WEIGHTS,
            BACKUP_WEIGHTS
        )

        print(
            "Backup created:"
        )

        print(
            BACKUP_WEIGHTS
        )

    else:

        print(
            "Backup already exists:"
        )

        print(
            BACKUP_WEIGHTS
        )

    # ========================================================
    # STAGE 1
    # CNN FROZEN
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STAGE 1 - FREEZE CNN")
    print("=" * 60)

    for layer in base_model.layers:

        if (
            layer.name.startswith("conv")
            or
            layer.name.startswith("pool")
        ):

            layer.trainable = False

        else:

            layer.trainable = True

    print("\nLayer trainability:")

    for layer in base_model.layers:

        print(
            f"{layer.name:30s} "
            f"{layer.trainable}"
        )

    model = CRNNModel(
        base_model
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-5
        )
    )

    stage1_callback = BestWeightsCallback(
        STAGE1_WEIGHTS
    )

    reduce_lr_stage1 = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    print("\nStarting Stage 1 training...")

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10,
        callbacks=[
            stage1_callback,
            reduce_lr_stage1
        ]
    )

    # --------------------------------------------------------
    # Load best Stage 1 weights
    # --------------------------------------------------------

    if STAGE1_WEIGHTS.exists():

        base_model.load_weights(
            STAGE1_WEIGHTS
        )

        print(
            "\nBest Stage 1 weights loaded."
        )

    # ========================================================
    # STAGE 2
    # UNFREEZE EVERYTHING
    # ========================================================

    print("\n")
    print("=" * 60)
    print("STAGE 2 - UNFREEZE ALL LAYERS")
    print("=" * 60)

    for layer in base_model.layers:

        layer.trainable = True

    model = CRNNModel(
        base_model
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=3e-6
        )
    )

    stage2_callback = BestWeightsCallback(
        FINAL_BEST_WEIGHTS
    )

    reduce_lr_stage2 = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    )

    print("\nStarting Stage 2 training...")

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=15,
        callbacks=[
            stage2_callback,
            reduce_lr_stage2
        ]
    )

    # --------------------------------------------------------
    # Save final weights
    # --------------------------------------------------------

    base_model.save_weights(
        FINAL_WEIGHTS
    )

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)

    print(
        "\nOriginal weights:"
    )
    print(
        ORIGINAL_WEIGHTS
    )

    print(
        "\nOriginal backup:"
    )
    print(
        BACKUP_WEIGHTS
    )

    print(
        "\nStage 1 best:"
    )
    print(
        STAGE1_WEIGHTS
    )

    print(
        "\nFine-tuned best:"
    )
    print(
        FINAL_BEST_WEIGHTS
    )

    print(
        "\nFine-tuned final:"
    )
    print(
        FINAL_WEIGHTS
    )

    print("\nTraining finished successfully.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()