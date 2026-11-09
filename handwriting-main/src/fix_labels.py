import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "finetune" / "my_labels.csv"
OUTPUT_FILE = PROJECT_ROOT / "finetune" / "my_labels_fixed.csv"

rows = []

with open(INPUT_FILE, "r", encoding="utf-8-sig") as f:

    for line in f:
        line = line.strip()

        if not line:
            continue

        # Skip header if present
        if line.lower().startswith("image"):
            continue

        # Remove outer quotes
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]

        # Split filename and text
        parts = line.split(",,,,,", 1)

        if len(parts) != 2:
            print("SKIPPED:", line)
            continue

        image = parts[0]
        text = parts[1]

        # Remove trailing commas
        while text.endswith(","):
            text = text[:-1]

        # Convert doubled quotes to normal quotes
        text = text.replace('""', '"')

        image = image.strip()
        text = text.strip()

        rows.append([image, text])


with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow(["image", "text"])

    writer.writerows(rows)


print()
print("===================================")
print("CSV FIXED SUCCESSFULLY")
print("===================================")
print("Total rows:", len(rows))
print("Output file:")
print(OUTPUT_FILE)
print()

print("First 5 entries:")

for row in rows[:5]:
    print(row)