import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dataset


# ============================================================
# 30-EPOCH MODEL
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "crnn_best.weights.h5"
)


IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256

SEQUENCE_LENGTH = 32

# IMPORTANT:
# The 30-epoch checkpoint has 76 output classes.
NUM_CLASSES = 76


# ============================================================
# BUILD 30-EPOCH CRNN
# ============================================================

def build_model():

    input_img = layers.Input(
        shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            1
        ),
        name="image"
    )

    # --------------------------------------------------------
    # CNN 1
    # --------------------------------------------------------

    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        name="conv1"
    )(input_img)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool1"
    )(x)

    # --------------------------------------------------------
    # CNN 2
    # --------------------------------------------------------

    x = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        name="conv2"
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool2"
    )(x)

    # --------------------------------------------------------
    # CNN 3
    # --------------------------------------------------------

    x = layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same",
        name="conv3"
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool3"
    )(x)

    # --------------------------------------------------------
    # CNN -> SEQUENCE
    # --------------------------------------------------------

    x = layers.Permute(
        (2, 1, 3),
        name="permute"
    )(x)

    x = layers.Reshape(
        (SEQUENCE_LENGTH, 2048),
        name="reshape"
    )(x)

    # --------------------------------------------------------
    # DENSE
    # --------------------------------------------------------

    x = layers.Dense(
        256,
        activation="relu",
        name="dense"
    )(x)

    # --------------------------------------------------------
    # BiLSTM 1
    # --------------------------------------------------------

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True
        ),
        name="bidirectional"
    )(x)

    # --------------------------------------------------------
    # BiLSTM 2
    # --------------------------------------------------------

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True
        ),
        name="bidirectional_1"
    )(x)

    # --------------------------------------------------------
    # 76-CLASS OUTPUT
    # --------------------------------------------------------

    output = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="character_probabilities"
    )(x)

    return keras.Model(
        inputs=input_img,
        outputs=output,
        name="CRNN_30_EPOCH"
    )


# ============================================================
# DECODE PREDICTION
# ============================================================

def decode_prediction(prediction):

    input_length = (
        np.ones(
            prediction.shape[0],
            dtype=np.int32
        )
        * prediction.shape[1]
    )

    decoded, _ = keras.backend.ctc_decode(
        prediction,
        input_length=input_length,
        greedy=True
    )

    decoded = decoded[0].numpy()

    vocabulary = (
        dataset.num_to_char
        .get_vocabulary()
    )

    results = []

    for sequence in decoded:

        text = ""

        for index in sequence:

            index = int(index)

            if index == -1:
                continue

            if index >= len(vocabulary):
                continue

            try:

                character = dataset.num_to_char(
                    index
                ).numpy()

                if isinstance(
                    character,
                    bytes
                ):
                    character = character.decode(
                        "utf-8"
                    )

                text += character

            except Exception:
                continue

        results.append(text)

    return results


# ============================================================
# DECODE TRUE LABEL
# ============================================================

def decode_label(
    label,
    label_length
):

    text = ""

    length = int(
        label_length
    )

    for index in label[:length]:

        try:

            character = dataset.num_to_char(
                int(index)
            ).numpy()

            if isinstance(
                character,
                bytes
            ):
                character = character.decode(
                    "utf-8"
                )

            text += character

        except Exception:
            continue

    return text


# ============================================================
# LEVENSHTEIN DISTANCE
# ============================================================

def levenshtein(
    reference,
    hypothesis
):

    rows = len(reference) + 1
    cols = len(hypothesis) + 1

    matrix = np.zeros(
        (rows, cols),
        dtype=np.int32
    )

    for i in range(rows):
        matrix[i, 0] = i

    for j in range(cols):
        matrix[0, j] = j

    for i in range(1, rows):

        for j in range(1, cols):

            if (
                reference[i - 1]
                == hypothesis[j - 1]
            ):
                cost = 0
            else:
                cost = 1

            matrix[i, j] = min(
                matrix[i - 1, j] + 1,
                matrix[i, j - 1] + 1,
                matrix[i - 1, j - 1] + cost
            )

    return int(
        matrix[-1, -1]
    )


# ============================================================
# CHARACTER MATCHES
# ============================================================

def character_matches(
    reference,
    hypothesis
):

    rows = len(reference) + 1
    cols = len(hypothesis) + 1

    matrix = np.zeros(
        (rows, cols),
        dtype=np.int32
    )

    for i in range(1, rows):

        for j in range(1, cols):

            if (
                reference[i - 1]
                == hypothesis[j - 1]
            ):

                matrix[i, j] = (
                    matrix[i - 1, j - 1]
                    + 1
                )

            else:

                matrix[i, j] = max(
                    matrix[i - 1, j],
                    matrix[i, j - 1]
                )

    return int(
        matrix[-1, -1]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("30-EPOCH MODEL EVALUATION")
    print("=" * 70)

    print()
    print("Model:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        maximum_label_length,
        training_count,
        validation_count
    ) = dataset.prepare_datasets()

    print()
    print("Dataset ready.")

    print(
        "Training samples:",
        training_count
    )

    print(
        "Validation samples:",
        validation_count
    )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    print()
    print("Building 76-class CRNN...")

    model = build_model()

    dummy = tf.zeros(
        (
            1,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            1
        ),
        dtype=tf.float32
    )

    output = model(
        dummy,
        training=False
    )

    print()
    print(
        "Model output shape:",
        output.shape
    )

    if output.shape[-1] != 76:

        raise ValueError(
            f"Expected 76 classes, "
            f"got {output.shape[-1]}"
        )

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\n30-epoch model not found:\n"
            f"{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    print()
    print("Loading 30-epoch weights...")

    model.load_weights(
        MODEL_PATH
    )

    print()
    print(
        "30-EPOCH CRNN WEIGHTS "
        "LOADED SUCCESSFULLY."
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total = 0
    correct_words = 0

    total_chars = 0
    matched_chars = 0
    total_edits = 0

    examples = 0

    print()
    print("=" * 70)
    print("EVALUATING VALIDATION DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    for batch in validation_dataset:

        images = batch["image"]

        labels = batch["label"]

        lengths = batch["label_length"]

        predictions = model.predict(
            images,
            verbose=0
        )

        predicted = decode_prediction(
            predictions
        )

        batch_size = images.shape[0]

        for i in range(batch_size):

            actual = decode_label(
                labels[i],
                lengths[i]
            )

            prediction = predicted[i]

            total += 1

            if actual == prediction:
                correct_words += 1

            total_chars += len(actual)

            matched_chars += (
                character_matches(
                    actual,
                    prediction
                )
            )

            total_edits += (
                levenshtein(
                    actual,
                    prediction
                )
            )

            # ------------------------------------------------
            # First 20 examples
            # ------------------------------------------------

            if examples < 20:

                print()
                print(
                    f"Example {examples + 1}"
                )

                print(
                    "Actual:    ",
                    repr(actual)
                )

                print(
                    "Predicted: ",
                    repr(prediction)
                )

                examples += 1

        if total % 320 == 0:

            print()
            print(
                f"Processed "
                f"{total} "
                f"validation samples..."
            )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    word_accuracy = (
        correct_words
        / total
        * 100
    )

    character_accuracy = (
        matched_chars
        / total_chars
        * 100
    )

    cer = (
        total_edits
        / total_chars
        * 100
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)

    print()
    print("Model: 30 epoch")

    print(
        "Validation samples:",
        total
    )

    print(
        "Correct words:",
        correct_words
    )

    print(
        f"Word Accuracy: "
        f"{word_accuracy:.2f}%"
    )

    print(
        f"Character Accuracy: "
        f"{character_accuracy:.2f}%"
    )

    print(
        f"Character Error Rate (CER): "
        f"{cer:.2f}%"
    )

    print()
    print("=" * 70)
    print("30-EPOCH EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()