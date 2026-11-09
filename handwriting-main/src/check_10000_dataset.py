import os
import csv

IMAGE_FOLDER = "data/iam_10000/images"
LABEL_FILE = "data/iam_10000/labels.csv"

# Count images
images = [
    file for file in os.listdir(IMAGE_FOLDER)
    if file.lower().endswith(".png")
]

# Count labels
with open(LABEL_FILE, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    rows = list(reader)

# Exclude header row
label_count = len(rows) - 1

print("DATASET VERIFICATION\n")
print("Number of images:", len(images))
print("Number of labels:", label_count)

print("\nFirst 5 labels:")
for row in rows[1:6]:
    print(row)

print("\n--- RESULT ---")

if len(images) == 10000 and label_count == 10000:
    print("SUCCESS: Dataset contains exactly 10,000 images and 10,000 labels!")
else:
    print("ERROR: Image and label counts do not match 10,000.")