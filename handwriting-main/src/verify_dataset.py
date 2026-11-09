import os

IMAGE_ROOT = "data/IAM/iam_words/words"
LABEL_FILE = "data/IAM/iam_words/words.txt"

print("Checking IAM dataset...\n")

print("Image folder exists:", os.path.exists(IMAGE_ROOT))
print("Label file exists:", os.path.exists(LABEL_FILE))

if os.path.exists(IMAGE_ROOT):
    folders = os.listdir(IMAGE_ROOT)
    print("Number of top-level folders:", len(folders))
    print("First 10 folders:", folders[:10])

if os.path.exists(LABEL_FILE):
    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.startswith("#")
    ]

    print("Number of annotation entries:", len(data_lines))
    print("First valid entry:", data_lines[0])