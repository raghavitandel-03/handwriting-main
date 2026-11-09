import os
import sys

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    PROJECT_DIR,
    "src"
)

MODELS_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

if SRC_DIR not in sys.path:
    sys.path.insert(
        0,
        SRC_DIR
    )


# ============================================================
# DATASET
# ============================================================

from dataset import prepare_datasets


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 30

BATCH_SIZE = 16

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256

SEQUENCE_LENGTH = 32

# 76 actual characters + 1 special CTC/lookup class
NUM_CLASSES = 77

LEARNING_RATE = 0.001

TRAINING_SAMPLES = 8000
VALIDATION_SAMPLES = 1999

STEPS_PER_EPOCH = (
    TRAINING_SAMPLES // BATCH_SIZE
)

VALIDATION_STEPS = (
    VALIDATION_SAMPLES // BATCH_SIZE
)


# ============================================================
# MODEL FILES
# ============================================================

BEST_MODEL = os.path.join(
    MODELS_DIR,
    "crnn_best.weights.h5"
)

FINAL_MODEL = os.path.join(
    MODELS_DIR,
    "crnn_final.weights.h5"
)

HISTORY_FILE = os.path.join(
    MODELS_DIR,
    "training_history.csv"
)


# ============================================================
# GPU
# ============================================================

print("=" * 70)
print("GPU CHECK")
print("=" * 70)

gpus = tf.config.list_physical_devices(
    "GPU"
)

print(
    "Available GPUs:",
    gpus
)

if gpus:

    print(
        "GPU detected successfully."
    )

else:

    print(
        "WARNING: GPU NOT DETECTED."
    )


# ============================================================
# BUILD CRNN
# ============================================================

def build_crnn():

    print()
    print("=" * 70)
    print("BUILDING CRNN MODEL")
    print("=" * 70)

    inputs = layers.Input(
        shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            1
        ),
        name="image"
    )


    # --------------------------------------------------------
    # CNN BLOCK 1
    # --------------------------------------------------------

    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        name="conv1"
    )(inputs)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool1"
    )(x)


    # --------------------------------------------------------
    # CNN BLOCK 2
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
    # CNN BLOCK 3
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
    # CNN → SEQUENCE
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

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="character_probabilities"
    )(x)


    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CRNN"
    )


# ============================================================
# CTC LOSS
# ============================================================

def ctc_loss(
    labels,
    predictions,
    label_lengths
):

    batch_size = tf.shape(
        predictions
    )[0]

    input_lengths = tf.fill(
        [batch_size, 1],
        tf.shape(
            predictions
        )[1]
    )

    labels = tf.cast(
        labels,
        tf.int32
    )

    label_lengths = tf.cast(
        label_lengths,
        tf.int32
    )

    loss = tf.keras.backend.ctc_batch_cost(
        labels,
        predictions,
        input_lengths,
        label_lengths
    )

    return tf.reduce_mean(
        loss
    )


# ============================================================
# CUSTOM MODEL
# ============================================================

class CRNNModel(
    keras.Model
):

    def __init__(
        self,
        base_model
    ):

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


    def call(
        self,
        inputs,
        training=False
    ):

        return self.base_model(
            inputs,
            training=training
        )


    def train_step(
        self,
        data
    ):

        images = data["image"]

        labels = data["label"]

        label_lengths = data[
            "label_length"
        ]


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
            self.base_model.trainable_variables
        )


        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.base_model.trainable_variables
            )
        )


        self.loss_tracker.update_state(
            loss
        )


        return {
            "loss":
            self.loss_tracker.result()
        }


    def test_step(
        self,
        data
    ):

        images = data["image"]

        labels = data["label"]

        label_lengths = data[
            "label_length"
        ]


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
            "loss":
            self.loss_tracker.result()
        }


# ============================================================
# BEST MODEL CALLBACK
# ============================================================

class BestModelCallback(
    keras.callbacks.Callback
):

    def __init__(
        self,
        model_to_save,
        filepath
    ):

        super().__init__()

        self.model_to_save = (
            model_to_save
        )

        self.filepath = filepath

        self.best = float(
            "inf"
        )


    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        if logs is None:

            return


        current = logs.get(
            "val_loss"
        )


        if current is None:

            return


        if current < self.best:

            previous = self.best

            self.best = current


            self.model_to_save.save_weights(
                self.filepath
            )


            if previous == float(
                "inf"
            ):

                print(
                    f"\nEpoch {epoch + 1}: "
                    f"val_loss improved from "
                    f"None to {current:.5f}, "
                    f"saving model."
                )

            else:

                print(
                    f"\nEpoch {epoch + 1}: "
                    f"val_loss improved from "
                    f"{previous:.5f} to "
                    f"{current:.5f}, "
                    f"saving model."
                )

        else:

            print(
                f"\nEpoch {epoch + 1}: "
                f"val_loss did not improve "
                f"from {self.best:.5f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)


    (
        train_dataset,
        validation_dataset,
        max_label_length
    ) = prepare_datasets()


    print()
    print(
        "Dataset loaded successfully."
    )

    print(
        "Maximum label length:",
        max_label_length
    )

    print(
        "Training samples:",
        TRAINING_SAMPLES
    )

    print(
        "Validation samples:",
        VALIDATION_SAMPLES
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Steps per epoch:",
        STEPS_PER_EPOCH
    )

    print(
        "Validation steps:",
        VALIDATION_STEPS
    )


    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    sample_batch = next(
        iter(train_dataset)
    )


    print()
    print(
        "Dataset batch check:"
    )

    print(
        "Image:",
        sample_batch["image"].shape
    )

    print(
        "Label:",
        sample_batch["label"].shape
    )

    print(
        "Label length:",
        sample_batch["label_length"].shape
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    base_model = build_crnn()

    base_model.summary()


    # --------------------------------------------------------
    # CUSTOM MODEL
    # --------------------------------------------------------

    model = CRNNModel(
        base_model
    )


    # Build before callbacks
    model(
        sample_batch["image"],
        training=False
    )


    print(
        "\nCRNN wrapper model built successfully."
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer
    )


    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    best_model_callback = (
        BestModelCallback(
            base_model,
            BEST_MODEL
        )
    )


    reduce_lr = (
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            mode="min",
            verbose=1
        )
    )


    csv_logger = (
        keras.callbacks.CSVLogger(
            HISTORY_FILE,
            append=False
        )
    )


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING CRNN TRAINING")
    print("=" * 70)

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Steps per epoch:",
        STEPS_PER_EPOCH
    )

    print(
        "Validation steps:",
        VALIDATION_STEPS
    )

    print(
        "Output classes:",
        NUM_CLASSES
    )

    print(
        "EarlyStopping: DISABLED"
    )

    print(
        "Training will run for exactly 30 epochs."
    )


    history = model.fit(

        train_dataset,

        epochs=EPOCHS,

        steps_per_epoch=STEPS_PER_EPOCH,

        validation_data=validation_dataset,

        validation_steps=VALIDATION_STEPS,

        callbacks=[
            best_model_callback,
            reduce_lr,
            csv_logger
        ],

        verbose=1
    )


    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    base_model.save_weights(
        FINAL_MODEL
    )


    print(
        "\nFinal model saved:"
    )

    print(
        FINAL_MODEL
    )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history.history
    )

    history_df.insert(
        0,
        "epoch",
        range(
            1,
            len(history_df) + 1
        )
    )

    history_df.to_csv(
        HISTORY_FILE,
        index=False
    )


    # --------------------------------------------------------
    # BEST EPOCH
    # --------------------------------------------------------

    best_epoch_index = (
        history_df[
            "val_loss"
        ].idxmin()
    )

    best_epoch = int(
        history_df.loc[
            best_epoch_index,
            "epoch"
        ]
    )

    best_val_loss = float(
        history_df.loc[
            best_epoch_index,
            "val_loss"
        ]
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("30 EPOCH TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Completed epochs:",
        len(history_df)
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best validation loss:",
        best_val_loss
    )

    print()
    print(
        "Best model:"
    )

    print(
        BEST_MODEL
    )

    print()
    print(
        "Final model:"
    )

    print(
        FINAL_MODEL
    )

    print()
    print(
        "Training history:"
    )

    print(
        HISTORY_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()