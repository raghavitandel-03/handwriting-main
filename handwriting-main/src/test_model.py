import os
import csv
import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Import the already-correct prediction model
from predict import (
    model,
    preprocess_image,
    decode_prediction
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "iam_10000"
)

IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "images"
)

LABEL_FILE = os.path.join(
    DATASET_DIR,
    "labels.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

VALIDATION_SPLIT = 0.20

# Number of images processed at once.
# CPU prediction is safer with a small batch.
BATCH_SIZE = 16


# ============================================================
# LOAD LABELS
# ============================================================

def load_records():

    print()
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    print()
    print("Image directory:")
    print(IMAGE_DIR)

    print()
    print("Label file:")
    print(LABEL_FILE)

    if not os.path.exists(IMAGE_DIR):

        raise FileNotFoundError(
            f"\nImage directory not found:\n{IMAGE_DIR}"
        )

    if not os.path.exists(LABEL_FILE):

        raise FileNotFoundError(
            f"\nLabel file not found:\n{LABEL_FILE}"
        )


    records = []


    with open(
        LABEL_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        print()
        print("CSV columns:")
        print(reader.fieldnames)


        # ----------------------------------------------------
        # Find image column
        # ----------------------------------------------------

        image_column = None

        for column in reader.fieldnames:

            if column.lower() == "image":

                image_column = column
                break


        # ----------------------------------------------------
        # Find text column
        # ----------------------------------------------------

        text_column = None

        for column in reader.fieldnames:

            if column.lower() in [
                "text",
                "label",
                "word"
            ]:

                text_column = column
                break


        if image_column is None:

            raise ValueError(
                "Could not find image column in labels.csv"
            )


        if text_column is None:

            raise ValueError(
                "Could not find text/label column in labels.csv"
            )


        print()
        print(
            "Image column:",
            image_column
        )

        print(
            "Text column:",
            text_column
        )


        # ----------------------------------------------------
        # Read records
        # ----------------------------------------------------

        for row in reader:

            image_name = (
                row[image_column]
                .strip()
            )

            text = (
                row[text_column]
            )

            if text is None:

                continue


            text = text.strip()


            if image_name == "":

                continue


            image_path = os.path.join(
                IMAGE_DIR,
                image_name
            )


            if not os.path.exists(
                image_path
            ):

                continue


            records.append(
                {
                    "image": image_path,
                    "text": text
                }
            )


    print()
    print(
        "Total valid records:",
        len(records)
    )


    return records


# ============================================================
# LEVENSHTEIN DISTANCE
# ============================================================

def levenshtein_distance(
    reference,
    hypothesis
):

    rows = len(reference) + 1
    cols = len(hypothesis) + 1


    previous = list(
        range(cols)
    )


    for i in range(
        1,
        rows
    ):

        current = [
            i
        ] + [
            0
        ] * (
            cols - 1
        )


        for j in range(
            1,
            cols
        ):

            insertion = (
                current[j - 1] + 1
            )

            deletion = (
                previous[j] + 1
            )

            substitution = (
                previous[j - 1]
                + (
                    0
                    if reference[i - 1]
                    == hypothesis[j - 1]
                    else 1
                )
            )


            current[j] = min(
                insertion,
                deletion,
                substitution
            )


        previous = current


    return previous[-1]


# ============================================================
# CHARACTER ACCURACY
# ============================================================

def calculate_character_accuracy(
    actual,
    predicted
):

    total_characters = 0
    total_errors = 0


    for reference, hypothesis in zip(
        actual,
        predicted
    ):

        distance = levenshtein_distance(
            reference,
            hypothesis
        )

        total_errors += distance
        total_characters += len(
            reference
        )


    if total_characters == 0:

        return 0.0


    accuracy = (
        1.0
        - (
            total_errors
            / total_characters
        )
    )


    return max(
        0.0,
        accuracy
    )


# ============================================================
# CHARACTER ERROR RATE
# ============================================================

def calculate_cer(
    actual,
    predicted
):

    total_errors = 0
    total_characters = 0


    for reference, hypothesis in zip(
        actual,
        predicted
    ):

        total_errors += (
            levenshtein_distance(
                reference,
                hypothesis
            )
        )

        total_characters += len(
            reference
        )


    if total_characters == 0:

        return 0.0


    return (
        total_errors
        / total_characters
    )


# ============================================================
# PREPARE VALIDATION SET
# ============================================================

def prepare_validation_set(
    records
):

    image_paths = [
        record["image"]
        for record in records
    ]

    labels = [
        record["text"]
        for record in records
    ]


    (
        train_images,
        validation_images,
        train_labels,
        validation_labels
    ) = train_test_split(
        image_paths,
        labels,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED
    )


    print()
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print()
    print(
        "Training samples:",
        len(train_images)
    )

    print(
        "Validation samples:",
        len(validation_images)
    )


    return (
        validation_images,
        validation_labels
    )


# ============================================================
# PREDICT VALIDATION DATASET
# ============================================================

def evaluate_validation(
    image_paths,
    actual_labels
):

    print()
    print("=" * 70)
    print("EVALUATING VALIDATION SET")
    print("=" * 70)

    print()
    print(
        "Total validation images:",
        len(image_paths)
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )


    predicted_labels = []


    total = len(
        image_paths
    )


    # --------------------------------------------------------
    # Process batches
    # --------------------------------------------------------

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )


        batch_paths = image_paths[
            start:end
        ]


        # ----------------------------------------------------
        # Load images
        # ----------------------------------------------------

        batch_images = []


        for image_path in batch_paths:

            image = preprocess_image(
                image_path
            )

            batch_images.append(
                image[0]
            )


        batch_images = np.stack(
            batch_images,
            axis=0
        )


        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        prediction = model.predict(
            batch_images,
            verbose=0
        )


        # ----------------------------------------------------
        # CTC decode
        # ----------------------------------------------------

        decoded = decode_prediction(
            prediction
        )


        predicted_labels.extend(
            decoded
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        processed = end


        if (
            processed % 160 == 0
            or processed == total
        ):

            print(
                f"Processed: "
                f"{processed}/{total}"
            )


    return predicted_labels


# ============================================================
# PRINT EXAMPLES
# ============================================================

def print_examples(
    actual,
    predicted,
    number=30
):

    print()
    print("=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)


    limit = min(
        number,
        len(actual)
    )


    for i in range(
        limit
    ):

        print()
        print(
            f"Example {i + 1}"
        )

        print(
            "Actual:    ",
            repr(actual[i])
        )

        print(
            "Predicted: ",
            repr(predicted[i])
        )


# ============================================================
# FINAL RESULTS
# ============================================================

def print_results(
    actual,
    predicted
):

    total = len(
        actual
    )


    correct_words = 0


    for reference, hypothesis in zip(
        actual,
        predicted
    ):

        if reference == hypothesis:

            correct_words += 1


    word_accuracy = (
        correct_words
        / total
        if total > 0
        else 0.0
    )


    character_accuracy = (
        calculate_character_accuracy(
            actual,
            predicted
        )
    )


    cer = (
        calculate_cer(
            actual,
            predicted
        )
    )


    print()
    print("=" * 70)
    print("FINAL LOCAL EVALUATION RESULTS")
    print("=" * 70)


    print()
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
        f"{word_accuracy * 100:.2f}%"
    )

    print(
        f"Character Accuracy: "
        f"{character_accuracy * 100:.2f}%"
    )

    print(
        f"Character Error Rate (CER): "
        f"{cer * 100:.2f}%"
    )


    print()
    print(
        "MODEL USED:"
    )

    print(
        os.path.join(
            PROJECT_ROOT,
            "models",
            "crnn_best.weights.h5"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("LOCAL CRNN MODEL TEST")
    print("=" * 70)


    # --------------------------------------------------------
    # Load records
    # --------------------------------------------------------

    records = load_records()


    if len(records) == 0:

        raise ValueError(
            "No valid dataset records found."
        )


    # --------------------------------------------------------
    # Validation split
    # --------------------------------------------------------

    (
        validation_images,
        validation_labels
    ) = prepare_validation_set(
        records
    )


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    predicted_labels = (
        evaluate_validation(
            validation_images,
            validation_labels
        )
    )


    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    print_examples(
        validation_labels,
        predicted_labels,
        number=30
    )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print_results(
        validation_labels,
        predicted_labels
    )


    print()
    print("=" * 70)
    print("LOCAL EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()