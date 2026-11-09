from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import sys
import tempfile

from spellchecker import SpellChecker

# Allow importing files from src
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)

from predict import predict_image


app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE = 10 * 1024 * 1024
SPELL_CHECKER = SpellChecker(distance=2)
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")


def levenshtein_distance(first, second):

    if len(first) < len(second):
        first, second = second, first

    previous_row = list(range(len(second) + 1))

    for row_index, first_character in enumerate(first, start=1):
        current_row = [row_index]

        for column_index, second_character in enumerate(second, start=1):
            insertion = current_row[column_index - 1] + 1
            deletion = previous_row[column_index] + 1
            substitution = previous_row[column_index - 1] + (
                first_character != second_character
            )
            current_row.append(min(insertion, deletion, substitution))

        previous_row = current_row

    return previous_row[-1]


def refine_prediction(text):

    def correct_word(match):
        word = match.group(0)
        normalized_word = word.lower()

        if "'" in word or "’" in word or SPELL_CHECKER.known([normalized_word]):
            return word

        candidates = SPELL_CHECKER.candidates(normalized_word)

        if not candidates:
            return word

        best_candidate = min(
            candidates,
            key=lambda candidate: (
                levenshtein_distance(normalized_word, candidate),
                -SPELL_CHECKER.word_usage_frequency(candidate),
                candidate,
            ),
        )

        if levenshtein_distance(normalized_word, best_candidate) > 2:
            return word

        if word.isupper():
            return best_candidate.upper()

        if word.istitle():
            return best_candidate.capitalize()

        if word.islower():
            return best_candidate

        return word

    return WORD_PATTERN.sub(correct_word, text)


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Handwriting Digitizer API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    temp_path = None

    try:

        file_key = None
        if "image" in request.files:
            file_key = "image"
        elif "file" in request.files:
            file_key = "file"

        if file_key is None or file_key not in request.files:
            return jsonify({
                "success": False,
                "error": "No image uploaded"
            }), 400

        image = request.files[file_key]

        if image.filename == "":
            return jsonify({
                "success": False,
                "error": "No image selected"
            }), 400

        if image.content_length and image.content_length > MAX_FILE_SIZE:
            return jsonify({
                "success": False,
                "error": "Image is too large. Maximum size is 10MB."
            }), 413

        suffix = os.path.splitext(
            os.path.basename(image.filename)
        )[1].lower()

        if suffix not in ALLOWED_EXTENSIONS:
            return jsonify({
                "success": False,
                "error": "Unsupported image type. Please upload JPG or PNG."
            }), 400

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix or ".png"
        ) as temp:
            image.save(temp.name)
            temp_path = temp.name

        raw_prediction = predict_image(temp_path)
        refined_prediction = refine_prediction(raw_prediction)

        return jsonify({
            "success": True,
            "text": refined_prediction,
            "raw_prediction": raw_prediction,
            "refined_prediction": refined_prediction,
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )