import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps


# ============================================================
# CONFIG
# ============================================================

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256
SEQUENCE_LENGTH = 32
NUM_CLASSES = 76

CHARACTERS = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?'-\"#()*:;"
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

WEIGHTS = os.path.join(
    PROJECT_ROOT,
    "models",
    "finetuned_best.weights.h5"
)


# ============================================================
# MODEL
# ============================================================

def build_model():

    inputs = layers.Input(
        shape=(64, 256, 1),
        name="image"
    )

    x = layers.Conv2D(
        64, 3,
        activation="relu",
        padding="same",
        name="conv1"
    )(inputs)

    x = layers.MaxPooling2D(
        2,
        name="pool1"
    )(x)

    x = layers.Conv2D(
        128, 3,
        activation="relu",
        padding="same",
        name="conv2"
    )(x)

    x = layers.MaxPooling2D(
        2,
        name="pool2"
    )(x)

    x = layers.Conv2D(
        256, 3,
        activation="relu",
        padding="same",
        name="conv3"
    )(x)

    x = layers.MaxPooling2D(
        2,
        name="pool3"
    )(x)

    x = layers.Permute(
        (2, 1, 3),
        name="permute"
    )(x)

    x = layers.Reshape(
        (32, -1),
        name="reshape"
    )(x)

    x = layers.Dense(
        256,
        activation="relu",
        name="dense"
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True
        ),
        name="bidirectional"
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True
        ),
        name="bidirectional_1"
    )(x)

    outputs = layers.Dense(
        76,
        activation="softmax",
        name="character_probabilities"
    )(x)

    return keras.Model(
        inputs,
        outputs
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("L")

    print(
        "Original image size:",
        image.size
    )

    # Invert if necessary so handwriting is dark
    # on a light background.
    arr = np.array(image)

    if arr.mean() < 127:
        image = ImageOps.invert(image)

    # Resize while preserving aspect ratio
    width, height = image.size

    scale = min(
        IMAGE_WIDTH / width,
        IMAGE_HEIGHT / height
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # White canvas
    canvas = Image.new(
        "L",
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        255
    )

    # Center image
    x = (IMAGE_WIDTH - new_width) // 2
    y = (IMAGE_HEIGHT - new_height) // 2

    canvas.paste(
        image,
        (x, y)
    )

    image = np.array(
        canvas,
        dtype=np.float32
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=-1
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# CTC DECODING
# ============================================================

def decode_prediction(prediction):

    decoded, _ = tf.keras.backend.ctc_decode(
        prediction,
        input_length=np.array([32]),
        greedy=False,
        beam_width=10,
        top_paths=3
    )

    results = []

    for path in decoded:

        ids = path.numpy()[0]

        text = ""

        for index in ids:

            if index == -1:
                continue

            if 0 <= index < len(CHARACTERS):
                text += CHARACTERS[index]

        results.append(text)

    return results


# ============================================================
# MAIN
# ============================================================

if len(sys.argv) < 2:

    print(
        "\nUsage:"
    )

    print(
        'python src\\predict_finetuned.py "image_path"'
    )

    sys.exit(1)


image_path = sys.argv[1]


if not os.path.exists(image_path):

    print(
        "Image not found:"
    )

    print(image_path)

    sys.exit(1)


if not os.path.exists(WEIGHTS):

    print(
        "Fine-tuned weights not found:"
    )

    print(WEIGHTS)

    sys.exit(1)


print(
    "\nLoading fine-tuned model..."
)

model = build_model()

model.load_weights(
    WEIGHTS
)

print(
    "Model loaded successfully."
)


print(
    "\nPreprocessing image..."
)

image = preprocess_image(
    image_path
)

print(
    "Input shape:",
    image.shape
)


print(
    "\nRunning OCR..."
)

prediction = model.predict(
    image,
    verbose=0
)

print(
    "Prediction shape:",
    prediction.shape
)


results = decode_prediction(
    prediction
)


print(
    "\n" + "=" * 60
)

print(
    "TOP PREDICTIONS"
)

print(
    "=" * 60
)

for i, text in enumerate(results, 1):

    print(
        f"{i}. {repr(text)}"
    )

print(
    "=" * 60
)