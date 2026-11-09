import os
import csv
import shutil

# Paths
IMAGE_ROOT = "data/IAM/iam_words/words"
LABEL_FILE = "data/IAM/iam_words/words.txt"

# Output paths
OUTPUT_DIR = "data/iam_10000"
OUTPUT_IMAGES = os.path.join(OUTPUT_DIR, "images")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "labels.csv")

# Number of images required
MAX_IMAGES = 10000


def get_image_path(word_id):
    """
    Convert IAM word ID into its actual image path.

    Example:
    a01-000u-00-00
    ->
    words/a01/a01-000u/a01-000u-00-00.png
    """

    parts = word_id.split("-")

    if len(parts) < 3:
        return None

    folder1 = parts[0]
    folder2 = f"{parts[0]}-{parts[1]}"

    return os.path.join(
        IMAGE_ROOT,
        folder1,
        folder2,
        word_id + ".png"
    )


def main():

    # Create output folder
    os.makedirs(OUTPUT_IMAGES, exist_ok=True)

    selected_data = []

    print("Reading IAM annotations...")
    print("Searching for valid image-label pairs...\n")

    with open(LABEL_FILE, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Need enough fields
            if len(parts) < 9:
                continue

            word_id = parts[0]
            status = parts[1]

            # Only use correctly segmented words
            if status != "ok":
                continue

            # Text starts after:
            # ID status graylevel components x y width height tag
            text = " ".join(parts[8:])

            # Skip empty or invalid text
            if not text:
                continue

            image_path = get_image_path(word_id)

            # Check image exists
            if image_path and os.path.exists(image_path):

                selected_data.append(
                    (word_id, image_path, text)
                )

            # Stop after 10,000 valid pairs
            if len(selected_data) == MAX_IMAGES:
                break

    print(f"Valid image-label pairs found: {len(selected_data)}")

    # Check whether exactly 10,000 were found
    if len(selected_data) < MAX_IMAGES:
        print("ERROR: Could not find 10,000 valid images.")
        return

    print("\nCopying 10,000 images...")

    # Create CSV
    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        # CSV header
        writer.writerow(["image", "text"])

        for index, (word_id, image_path, text) in enumerate(selected_data, start=1):

            # New simple filename
            new_filename = f"image_{index:05d}.png"

            destination = os.path.join(
                OUTPUT_IMAGES,
                new_filename
            )

            # Copy image
            shutil.copy2(image_path, destination)

            # Save filename and label
            writer.writerow([new_filename, text])

            # Progress
            if index % 1000 == 0:
                print(f"{index} images copied...")

    print("\nDataset preparation completed successfully!")
    print(f"Images folder: {OUTPUT_IMAGES}")
    print(f"Labels file: {OUTPUT_CSV}")
    print(f"Total images: {len(selected_data)}")


if __name__ == "__main__":
    main()