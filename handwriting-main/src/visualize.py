import os
import sys
import matplotlib.pyplot as plt

# Allow importing preprocess.py from the same src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_image


def main():

    IMAGE_FOLDER = "data/iam_10000/images"

    # Select a few sample images
    image_files = [
        "image_00001.png",
        "image_00002.png",
        "image_00003.png",
        "image_00004.png",
        "image_00005.png"
    ]

    # Display each processed image
    fig, axes = plt.subplots(len(image_files), 1, figsize=(10, 6))

    for ax, filename in zip(axes, image_files):

        image_path = os.path.join(IMAGE_FOLDER, filename)

        # Read and preprocess image
        processed_image = preprocess_image(image_path)

        # Display image
        ax.imshow(
            processed_image,
            cmap="gray",
            aspect="auto"
        )

        ax.set_title(f"{filename} | Shape: {processed_image.shape}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()