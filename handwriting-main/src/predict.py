import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "crnn_best.weights.h5"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256

SEQUENCE_LENGTH = 32

# IMPORTANT:
# This is the 30-EPOCH model.
# It has 76 output classes.
NUM_CLASSES = 76


# ============================================================
# DEBUG OUTPUT
# ============================================================

DEBUG_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "test"
)

DEBUG_IMAGE_PATH = os.path.join(
    DEBUG_DIR,
    "preprocessed_debug.png"
)


# ============================================================
# VOCABULARY
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
# BUILD 30-EPOCH CRNN
# ============================================================

def build_prediction_model():

    input_img = layers.Input(
        shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            1
        ),
        name="image",
        dtype="float32"
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
        (
            SEQUENCE_LENGTH,
            2048
        ),
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
    # OUTPUT
    # --------------------------------------------------------

    output = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="character_probabilities"
    )(x)

    model = keras.Model(
        inputs=input_img,
        outputs=output,
        name="CRNN_30_EPOCH"
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING 30-EPOCH TRAINED CRNN")
print("=" * 70)

print()
print("Model path:")
print(MODEL_PATH)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\n30-epoch model not found:\n"
        f"{MODEL_PATH}"
    )


model = build_prediction_model()


# ------------------------------------------------------------
# Build model before loading weights
# ------------------------------------------------------------

dummy_input = tf.zeros(
    (
        1,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        1
    ),
    dtype=tf.float32
)

dummy_output = model(
    dummy_input,
    training=False
)


print()
print("Model output shape:")
print(dummy_output.shape)


if dummy_output.shape[-1] != NUM_CLASSES:

    raise ValueError(
        "\nWrong model architecture.\n"
        f"Expected {NUM_CLASSES} classes.\n"
        f"Received {dummy_output.shape[-1]}."
    )


# ------------------------------------------------------------
# Load 30-epoch weights
# ------------------------------------------------------------

model.load_weights(
    MODEL_PATH
)


print()
print(
    "30-EPOCH CRNN WEIGHTS "
    "LOADED SUCCESSFULLY."
)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"\nImage not found:\n"
            f"{image_path}"
        )


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    original = Image.open(
        image_path
    ).convert("L")


    gray = np.asarray(
        original,
        dtype=np.float32
    )


    print()
    print("Original image size:")
    print(original.size)


    print()
    print("Original intensity:")

    print(
        "Min:",
        float(gray.min())
    )

    print(
        "Max:",
        float(gray.max())
    )

    print(
        "Median:",
        float(np.median(gray))
    )


    # ========================================================
    # DETERMINE BACKGROUND
    # ========================================================

    median = float(
        np.median(gray)
    )


    # For your photograph the background is approximately
    # 115 and handwriting is considerably darker.
    #
    # We use a threshold below the background.

    threshold = median - 35.0


    # Keep threshold within safe limits.

    threshold = max(
        50.0,
        min(
            threshold,
            90.0
        )
    )


    print()
    print(
        "Handwriting threshold:",
        round(
            threshold,
            2
        )
    )


    # ========================================================
    # FIND DARK HANDWRITING
    # ========================================================

    mask = (
        gray < threshold
    )


    # ========================================================
    # ROW / COLUMN ANALYSIS
    # ========================================================

    row_counts = np.sum(
        mask,
        axis=1
    )

    col_counts = np.sum(
        mask,
        axis=0
    )


    valid_rows = np.where(
        row_counts >= 3
    )[0]


    valid_cols = np.where(
        col_counts >= 3
    )[0]


    if (
        len(valid_rows) == 0
        or
        len(valid_cols) == 0
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            "Could not detect handwriting."
        )

        print(
            "Using complete image."
        )

        crop = gray

    else:

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x_min = int(
            valid_cols.min()
        )

        x_max = int(
            valid_cols.max()
        )

        y_min = int(
            valid_rows.min()
        )

        y_max = int(
            valid_rows.max()
        )


        print()
        print(
            "Detected handwriting box:"
        )

        print(
            f"x={x_min}:{x_max}, "
            f"y={y_min}:{y_max}"
        )


        detected_width = (
            x_max - x_min + 1
        )

        detected_height = (
            y_max - y_min + 1
        )


        print()
        print(
            "Detected handwriting size:"
        )

        print(
            f"{detected_width} x "
            f"{detected_height}"
        )


        # ====================================================
        # SAFETY CHECK
        # ====================================================

        image_area = (
            gray.shape[0]
            * gray.shape[1]
        )

        detected_area = (
            detected_width
            * detected_height
        )

        area_ratio = (
            detected_area
            / image_area
        )


        print()
        print(
            "Detected area ratio:",
            round(
                area_ratio,
                4
            )
        )


        # If detection is absurdly large, use the central
        # region instead of feeding the entire photograph.

        if (
            area_ratio > 0.70
            or
            detected_width
            > gray.shape[1] * 0.95
            or
            detected_height
            > gray.shape[0] * 0.95
        ):

            print()
            print(
                "WARNING:"
            )

            print(
                "Detected region is too large."
            )

            print(
                "Using central region."
            )


            h, w = gray.shape


            cx1 = int(
                w * 0.10
            )

            cx2 = int(
                w * 0.90
            )

            cy1 = int(
                h * 0.10
            )

            cy2 = int(
                h * 0.90
            )


            center_gray = gray[
                cy1:cy2,
                cx1:cx2
            ]


            center_mask = (
                center_gray < threshold
            )


            center_rows = np.where(
                np.sum(
                    center_mask,
                    axis=1
                ) >= 3
            )[0]


            center_cols = np.where(
                np.sum(
                    center_mask,
                    axis=0
                ) >= 3
            )[0]


            if (
                len(center_rows) > 0
                and
                len(center_cols) > 0
            ):

                x_min = (
                    int(center_cols.min())
                    + cx1
                )

                x_max = (
                    int(center_cols.max())
                    + cx1
                )

                y_min = (
                    int(center_rows.min())
                    + cy1
                )

                y_max = (
                    int(center_rows.max())
                    + cy1
                )


                print()
                print(
                    "Refined handwriting box:"
                )

                print(
                    f"x={x_min}:{x_max}, "
                    f"y={y_min}:{y_max}"
                )


        # ====================================================
        # ADD PADDING
        # ====================================================

        width = (
            x_max - x_min + 1
        )

        height = (
            y_max - y_min + 1
        )


        pad_x = max(
            8,
            int(
                width * 0.08
            )
        )

        pad_y = max(
            8,
            int(
                height * 0.12
            )
        )


        x_min = max(
            0,
            x_min - pad_x
        )

        x_max = min(
            gray.shape[1] - 1,
            x_max + pad_x
        )

        y_min = max(
            0,
            y_min - pad_y
        )

        y_max = min(
            gray.shape[0] - 1,
            y_max + pad_y
        )


        crop = gray[
            y_min:y_max + 1,
            x_min:x_max + 1
        ]


        print()
        print(
            "Crop after padding:"
        )

        print(
            f"{crop.shape[1]} x "
            f"{crop.shape[0]}"
        )


    # ========================================================
    # CONVERT PHOTOGRAPH TO CLEAN BLACK/WHITE
    # ========================================================

    crop_mask = (
        crop < threshold
    )


    clean = np.full(
        crop.shape,
        255,
        dtype=np.uint8
    )


    # Keep dark handwriting.

    clean[crop_mask] = np.clip(
        crop[crop_mask],
        0,
        180
    ).astype(
        np.uint8
    )


    # ========================================================
    # SECOND CLEAN HANDWRITING BOX
    # ========================================================

    clean_mask = (
        clean < 180
    )


    ys, xs = np.where(
        clean_mask
    )


    if (
        len(xs) > 20
        and
        len(ys) > 20
    ):

        cx1 = int(
            xs.min()
        )

        cx2 = int(
            xs.max()
        )

        cy1 = int(
            ys.min()
        )

        cy2 = int(
            ys.max()
        )


        # Add small padding.

        px = max(
            5,
            int(
                (cx2 - cx1 + 1)
                * 0.08
            )
        )

        py = max(
            5,
            int(
                (cy2 - cy1 + 1)
                * 0.12
            )
        )


        cx1 = max(
            0,
            cx1 - px
        )

        cx2 = min(
            clean.shape[1] - 1,
            cx2 + px
        )

        cy1 = max(
            0,
            cy1 - py
        )

        cy2 = min(
            clean.shape[0] - 1,
            cy2 + py
        )


        clean = clean[
            cy1:cy2 + 1,
            cx1:cx2 + 1
        ]


        print()
        print(
            "Clean handwriting size:"
        )

        print(
            f"{clean.shape[1]} x "
            f"{clean.shape[0]}"
        )


    # ========================================================
    # SAVE DEBUG IMAGE
    # ========================================================

    os.makedirs(
        DEBUG_DIR,
        exist_ok=True
    )


    debug_image = Image.fromarray(
        clean,
        mode="L"
    )


    debug_image.save(
        DEBUG_IMAGE_PATH
    )


    print()
    print(
        "Debug preprocessed image saved:"
    )

    print(
        DEBUG_IMAGE_PATH
    )


    # ========================================================
    # RESIZE TO TRAINING SIZE
    # ========================================================

    clean_image = Image.fromarray(
        clean,
        mode="L"
    )


    clean_image = clean_image.resize(
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        ),
        Image.Resampling.LANCZOS
    )


    print()
    print(
        "Final handwriting image:"
    )

    print(
        f"{IMAGE_WIDTH} x "
        f"{IMAGE_HEIGHT}"
    )


    # ========================================================
    # NORMALIZE
    # ========================================================

    image_array = np.asarray(
        clean_image,
        dtype=np.float32
    )


    image_array /= 255.0


    # ========================================================
    # CHANNEL
    # ========================================================

    image_array = np.expand_dims(
        image_array,
        axis=-1
    )


    # ========================================================
    # BATCH
    # ========================================================

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    print()
    print(
        "Final model input:"
    )

    print(
        image_array.shape
    )


    return image_array


# ============================================================
# CTC DECODING
# ============================================================

def decode_prediction(
    prediction
):

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
        num_to_char
        .get_vocabulary()
    )


    results = []


    for sequence in decoded:

        text = ""


        for index in sequence:

            index = int(
                index
            )


            # CTC blank/padding

            if index == -1:
                continue


            # Safety check

            if index >= len(
                vocabulary
            ):
                continue


            try:

                character = (
                    num_to_char(
                        index
                    ).numpy()
                )


                if isinstance(
                    character,
                    bytes
                ):

                    character = (
                        character.decode(
                            "utf-8"
                        )
                    )


                text += character


            except Exception:

                continue


        results.append(
            text
        )


    return results


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(
    image_path
):

    print()
    print("=" * 70)
    print("PREDICTING")
    print("=" * 70)


    print()
    print(
        "Image:"
    )

    print(
        image_path
    )


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image = preprocess_image(
        image_path
    )


    print()
    print(
        "Input shape:"
    )

    print(
        image.shape
    )


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    prediction = model.predict(
        image,
        verbose=0
    )


    print()
    print(
        "Prediction shape:"
    )

    print(
        prediction.shape
    )


    # --------------------------------------------------------
    # CTC decode
    # --------------------------------------------------------

    decoded = decode_prediction(
        prediction
    )


    print()
    print("=" * 70)
    print("RECOGNIZED TEXT")
    print("=" * 70)


    print()


    if decoded:

        print(
            repr(
                decoded[0]
            )
        )

    else:

        print(
            "''"
        )


    return (
        decoded[0]
        if decoded
        else ""
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )

        print(
            "python src\\predict.py "
            "<image_path>"
        )

        print()
        print(
            "Example:"
        )

        print(
            "python src\\predict.py "
            "data\\test\\my_handwriting.png"
        )

        sys.exit(1)


    image_path = sys.argv[1]


    predict_image(
        image_path
    )