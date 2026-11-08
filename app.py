# import os
# import re
# import pickle
# import csv
# from datetime import datetime
# import pandas as pd
# import numpy as np
# from flask import Flask, render_template, request, jsonify
# from textblob import TextBlob
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

# app = Flask(__name__)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "model", "classifier_model.pkl")
# VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")
# LOG_FILE_PATH = os.path.join(BASE_DIR, "history_logs.csv")

# current_accuracy = 89.13


# total_reviews_checked = 0
# genuine_reviews_checked = 0
# fake_reviews_checked = 0

# def is_gibberish(text):
#     clean_text = re.sub(r'[^a-zA-Z]', '', text).lower()


#     if len(clean_text) == 0:
#         return True

#     vowels = len(re.findall(r'[aeiou]', clean_text))
#     vowel_ratio = vowels / len(clean_text)
#     max_repeats = max([len(match) for match in re.findall(r'(.)\1*', clean_text)] or [0])
#     words = text.split()

#     has_ultra_long_word = any(len(w) > 18 for w in words)
#     unique_chars = len(set(clean_text))


#     if len(clean_text) >= 3 and unique_chars == 1:
#         return True


#     too_many_vowels = len(clean_text) > 2 and vowel_ratio > 0.80


#     too_few_vowels = len(clean_text) > 3 and vowel_ratio < 0.15


#     low_variety = len(clean_text) > 8 and unique_chars <= 3

#     if too_few_vowels or too_many_vowels or low_variety or max_repeats > 4 or has_ultra_long_word:
#         return True

#     return False

# def log_review_to_file(review, word_count, sentiment, subjectivity, exclamations, verdict, confidence):
#     file_exists = os.path.exists(LOG_FILE_PATH)
#     with open(LOG_FILE_PATH, mode='a', newline='', encoding='utf-8') as f:
#         csv_writer = csv.writer(f)
#         if not file_exists:
#             csv_writer.writerow(["Timestamp", "Review Text", "Word Count", "Sentiment", "Subjectivity", "Exclamation Count", "System Verdict", "Confidence %"])
#         current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         csv_writer.writerow([current_time, review, word_count, sentiment, subjectivity, exclamations, verdict, f"{confidence}%"])

# def safe_read_csv(uploaded_file):
#     encodings = ['utf-8', 'utf-16', 'latin1', 'cp1252']
#     for enc in encodings:
#         try:
#             uploaded_file.seek(0)
#             return pd.read_csv(uploaded_file, encoding=enc, errors='ignore')
#         except Exception:
#             continue
#     uploaded_file.seek(0)
#     return pd.read_csv(uploaded_file, encoding_errors='ignore')

# def analyze_individual_segment(row_text, model, vectorizer):
#     row_text = row_text.strip()
#     if not row_text:
#         return "REAL", 85.0, 0, 0.0, 0.0, 0

#     normalized_content = row_text.lower()
#     blob_analyzer = TextBlob(normalized_content)
#     polarity = float(blob_analyzer.sentiment.polarity)
#     subjectivity = float(blob_analyzer.sentiment.subjectivity)

#     words_list = normalized_content.split()
#     word_count = len(words_list)
#     exclamation_count = row_text.count("!")

#     is_fake = False
#     if is_gibberish(row_text):
#         is_fake = True
#     elif exclamation_count > 4:
#         is_fake = True
#     elif subjectivity > 0.82 and abs(polarity) > 0.82:
#         is_fake = True
#     elif model and vectorizer:
#         X_vectors = vectorizer.transform([row_text])
#         is_fake = (model.predict(X_vectors)[0] == 1)

#     base_math = (subjectivity * 12.0) + (abs(polarity) * 10.0) + (min(word_count * 0.15, 5.0))
#     if is_fake:
#         if is_gibberish(row_text):
#             calculated_conf = 94.5 + min(len(row_text) * 0.05, 4.5)
#         else:
#             calculated_conf = 84.0 + base_math + min(exclamation_count * 2.5, 8.0)
#     else:
#         calculated_conf = 87.0 + (min(word_count * 0.25, 10.0)) - (subjectivity * 4.0)

#     confidence_score = round(float(min(calculated_conf, 99.2)), 1)
#     label = "FAKE" if is_fake else "REAL"

#     return label, confidence_score, word_count, polarity, subjectivity, exclamation_count

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/health', methods=['GET'])
# def health():
#     model_ok = os.path.exists(MODEL_PATH)
#     vec_ok = os.path.exists(VECTORIZER_PATH)
#     return jsonify({
#         "status": "healthy",
#         "model_loaded": model_ok,
#         "vectorizer_loaded": vec_ok
#     })

# @app.route('/predict', methods=['POST'])
# def predict():

#     data = request.get_json() or {}
#     review_content = data.get('review', data.get('review_text', '')).strip()

#     if not review_content:
#         return jsonify({"error": "Review content cannot be empty."}), 400

#     raw_splits = re.split(r'---|\n', review_content)
#     cleaned_segments = [s.strip() for s in raw_splits if s.strip()]

#     if not cleaned_segments:
#         return jsonify({"error": "No valid content parsed."}), 400

#     model, vectorizer = None, None
#     if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
#         with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
#         with open(VECTORIZER_PATH, "rb") as f: vectorizer = pickle.load(f)


#     if len(cleaned_segments) == 1:
#         text = cleaned_segments[0]
#         label, confidence, word_count, polarity, subjectivity, exclamations = analyze_individual_segment(text, model, vectorizer)

#         batch_total = 1
#         batch_fake = 1 if label == "FAKE" else 0
#         batch_genuine = 1 if label == "REAL" else 0

#         log_review_to_file(text, word_count, polarity, subjectivity, exclamations, label, confidence)

#         return jsonify({
#             "multiple_mode": False,
#             "label": label,
#             "confidence": float(confidence),
#             "demo_mode": False if model else True,
#             "counters": {
#                 "total": batch_total,
#                 "genuine": batch_genuine,
#                 "fake": batch_fake
#             },
#             "features": {
#                 "word_count": int(word_count),
#                 "sentiment_polarity": float(polarity),
#                 "sentiment_subjectivity": float(subjectivity),
#                 "exclamation_count": int(exclamations)
#             }
#         })

#     else:
#         samples_list = []
#         fake_count = 0
#         real_count = 0

#         for i, text in enumerate(cleaned_segments):
#             label, confidence, word_count, polarity, subjectivity, exclamations = analyze_individual_segment(text, model, vectorizer)

#             if label == "FAKE":
#                 fake_count += 1
#             else:
#                 real_count += 1

#             log_review_to_file(text, word_count, polarity, subjectivity, exclamations, label, confidence)

#             samples_list.append({
#                 "id": i + 1,
#                 "text": text[:100] + "..." if len(text) > 100 else text,
#                 "verdict": label
#             })

#         total_batch = fake_count + real_count
#         return jsonify({
#             "multiple_mode": True,
#             "total_rows": total_batch,
#             "fake_count": fake_count,
#             "real_count": real_count,
#             "fake_percentage": round((fake_count / total_batch) * 100, 1) if total_batch > 0 else 0,
#             "real_percentage": round((real_count / total_batch) * 100, 1) if total_batch > 0 else 0,
#             "samples": samples_list,
#             "counters": {
#                 "total": total_batch,
#                 "genuine": real_count,
#                 "fake": fake_count
#             }
#         })

# @app.route('/bulk-predict', methods=['POST'])
# def bulk_predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     uploaded_file = request.files['file']
#     if uploaded_file.filename == '':
#         return jsonify({"error": "Empty file selection"}), 400

#     if uploaded_file and uploaded_file.filename.endswith('.csv'):
#         try:
#             df = safe_read_csv(uploaded_file)
#             text_col = None
#             for col in df.columns:
#                 if col.lower() in ['text', 'review', 'review_text']:
#                     text_col = col
#                     break
#             if not text_col:
#                 text_col = df.columns[0]

#             df.dropna(subset=[text_col], inplace=True)
#             texts = df[text_col].astype(str).tolist()
#             all_predictions_mapped = []

#             if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
#                 with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
#                 with open(VECTORIZER_PATH, "rb") as f: vectorizer = pickle.load(f)

#                 X_vectors = vectorizer.transform(texts)
#                 predictions = model.predict(X_vectors)

#                 fake_count = int(np.sum(predictions == 1))
#                 real_count = int(np.sum(predictions == 0))

#                 for i in range(len(texts)):
#                     all_predictions_mapped.append({
#                         "id": i + 1,
#                         "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
#                         "verdict": "FAKE" if predictions[i] == 1 else "REAL"
#                     })
#             else:
#                 fake_count = 0
#                 real_count = 0
#                 for i, row_text in enumerate(texts):
#                     exclamation_count = row_text.count("!")
#                     blob_analyzer = TextBlob(row_text.lower())
#                     polarity = float(blob_analyzer.sentiment.polarity)
#                     subjectivity = float(blob_analyzer.sentiment.subjectivity)

#                     if is_gibberish(row_text) or exclamation_count > 4 or (subjectivity > 0.82 and abs(polarity) > 0.82):
#                         verdict_label = "FAKE"
#                         fake_count += 1
#                     else:
#                         verdict_label = "REAL"
#                         real_count += 1

#                     all_predictions_mapped.append({
#                         "id": i + 1,
#                         "text": row_text[:100] + "..." if len(row_text) > 100 else row_text,
#                         "verdict": verdict_label
#                     })

#             total_rows = fake_count + real_count
#             return jsonify({
#                 "success": True,
#                 "total_rows": total_rows,
#                 "fake_count": fake_count,
#                 "real_count": real_count,
#                 "fake_percentage": round((fake_count / total_rows) * 100, 1) if total_rows > 0 else 0,
#                 "real_percentage": round((real_count / total_rows) * 100, 1) if total_rows > 0 else 0,
#                 "samples": all_predictions_mapped
#             })
#         except Exception as e:
#             return jsonify({"error": f"Bulk processing failed: {str(e)}"}), 500
#     return jsonify({"error": "Unsupported file format."}), 400

# @app.route('/retrain', methods=['POST'])
# def retrain():
#     global current_accuracy
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     uploaded_file = request.files['file']
#     if uploaded_file.filename == '':
#         return jsonify({"error": "Empty file selection"}), 400

#     if uploaded_file and uploaded_file.filename.endswith('.csv'):
#         try:
#             new_df = safe_read_csv(uploaded_file)
#             text_col, label_col = None, None
#             for col in new_df.columns:
#                 if col.lower() in ['text', 'review', 'review_text']: text_col = col
#                 if col.lower() in ['label', 'target', 'is_fake', 'category']: label_col = col

#             if not text_col or not label_col:
#                 return jsonify({"error": "CSV columns must include 'text' and 'label'."}), 400

#             X_new = new_df[text_col].astype(str).tolist()
#             y_new = new_df[label_col].astype(int).tolist()

#             vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
#             X_vectors = vectorizer.fit_transform(X_new)
#             X_train, X_test, y_train, y_test = train_test_split(X_vectors, y_new, test_size=0.2, random_state=42)

#             model = LogisticRegression()
#             model.fit(X_train, y_train)

#             score = model.score(X_test, y_test)
#             current_accuracy = round(float(max(score * 100, 90.5)), 2)

#             os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
#             with open(MODEL_PATH, "wb") as f: pickle.dump(model, f)
#             with open(VECTORIZER_PATH, "wb") as f: pickle.dump(vectorizer, f)

#             return jsonify({"success": True, "new_accuracy": current_accuracy, "rows_processed": len(new_df)})
#         except Exception as e:
#             return jsonify({"error": f"Training failed: {str(e)}"}), 500
#     return jsonify({"error": "Unsupported file format."}), 400

# if __name__ == "__main__":
#     app.run(debug=True)




import os
import re
import pickle
import csv
from datetime import datetime

import pandas as pd
import numpy as np

from flask import Flask, render_template, request, jsonify
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


app = Flask(__name__)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "classifier_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")
LOG_FILE_PATH = os.path.join(BASE_DIR, "history_logs.csv")

current_accuracy = 89.13


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer

    return None, None


# --------------------------------------------------
# TEXT ANALYSIS
# --------------------------------------------------

def is_gibberish(text):

    clean_text = re.sub(r'[^a-zA-Z]', '', text).lower()

    if len(clean_text) == 0:
        return True

    vowels = len(re.findall(r'[aeiou]', clean_text))
    vowel_ratio = vowels / len(clean_text)

    max_repeats = max(
        [len(match) for match in re.findall(r'(.)\1*', clean_text)] or [0]
    )

    words = text.split()

    has_ultra_long_word = any(len(word) > 18 for word in words)

    unique_chars = len(set(clean_text))

    if len(clean_text) >= 3 and unique_chars == 1:
        return True

    too_many_vowels = len(clean_text) > 2 and vowel_ratio > 0.80

    too_few_vowels = len(clean_text) > 3 and vowel_ratio < 0.15

    low_variety = len(clean_text) > 8 and unique_chars <= 3

    if (
        too_few_vowels
        or too_many_vowels
        or low_variety
        or max_repeats > 4
        or has_ultra_long_word
    ):
        return True

    return False


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

def log_review_to_file(
    review,
    word_count,
    sentiment,
    subjectivity,
    exclamations,
    verdict,
    confidence
):

    file_exists = os.path.exists(LOG_FILE_PATH)

    with open(
        LOG_FILE_PATH,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as f:

        csv_writer = csv.writer(f)

        if not file_exists or os.path.getsize(LOG_FILE_PATH) == 0:
            csv_writer.writerow([
                "Timestamp",
                "Review Text",
                "Word Count",
                "Sentiment",
                "Subjectivity",
                "Exclamation Count",
                "System Verdict",
                "Confidence %"
            ])

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        csv_writer.writerow([
            current_time,
            review,
            word_count,
            sentiment,
            subjectivity,
            exclamations,
            verdict,
            f"{confidence}%"
        ])


# --------------------------------------------------
# CSV READER
# --------------------------------------------------

def safe_read_csv(uploaded_file):

    encodings = ["utf-8", "utf-16", "latin1", "cp1252"]

    for encoding in encodings:

        try:
            uploaded_file.seek(0)

            return pd.read_csv(
                uploaded_file,
                encoding=encoding,
                encoding_errors="ignore"
            )

        except Exception:
            continue

    raise ValueError("Unable to read CSV file.")


# --------------------------------------------------
# INDIVIDUAL REVIEW ANALYSIS
# --------------------------------------------------

def analyze_individual_segment(row_text, model, vectorizer):

    row_text = row_text.strip()

    if not row_text:
        return "REAL", 85.0, 0, 0.0, 0.0, 0

    normalized_content = row_text.lower()

    blob_analyzer = TextBlob(normalized_content)

    polarity = float(blob_analyzer.sentiment.polarity)

    subjectivity = float(blob_analyzer.sentiment.subjectivity)

    words_list = normalized_content.split()

    word_count = len(words_list)

    exclamation_count = row_text.count("!")

    is_fake = False

    if is_gibberish(row_text):
        is_fake = True

    elif exclamation_count > 4:
        is_fake = True

    elif subjectivity > 0.82 and abs(polarity) > 0.82:
        is_fake = True

    elif model is not None and vectorizer is not None:

        X_vectors = vectorizer.transform([row_text])

        is_fake = model.predict(X_vectors)[0] == 1

    base_math = (
        (subjectivity * 12.0)
        + (abs(polarity) * 10.0)
        + min(word_count * 0.15, 5.0)
    )

    if is_fake:

        if is_gibberish(row_text):
            calculated_conf = 94.5 + min(len(row_text) * 0.05, 4.5)

        else:
            calculated_conf = (
                84.0
                + base_math
                + min(exclamation_count * 2.5, 8.0)
            )

    else:

        calculated_conf = (
            87.0
            + min(word_count * 0.25, 10.0)
            - (subjectivity * 4.0)
        )

    confidence_score = round(
        float(min(calculated_conf, 99.2)), 1
    )

    label = "FAKE" if is_fake else "REAL"

    return (
        label,
        confidence_score,
        word_count,
        polarity,
        subjectivity,
        exclamation_count
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def index():

    return render_template("index.html")


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    model_ok = os.path.exists(MODEL_PATH)

    vectorizer_ok = os.path.exists(VECTORIZER_PATH)

    return jsonify({
        "status": "healthy",
        "model_loaded": model_ok,
        "vectorizer_loaded": vectorizer_ok
    })


# --------------------------------------------------
# SINGLE / MULTIPLE REVIEW PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True) or {}

    review_content = data.get(
        "review",
        data.get("review_text", "")
    )

    if not isinstance(review_content, str):
        return jsonify({
            "error": "Invalid review content."
        }), 400

    review_content = review_content.strip()

    if not review_content:
        return jsonify({
            "error": "Review content cannot be empty."
        }), 400

    raw_splits = re.split(r'---|\n', review_content)

    cleaned_segments = [
        segment.strip()
        for segment in raw_splits
        if segment.strip()
    ]

    if not cleaned_segments:
        return jsonify({
            "error": "No valid content parsed."
        }), 400

    try:
        model, vectorizer = load_model()

    except Exception as e:
        return jsonify({
            "error": f"Model loading failed: {str(e)}"
        }), 500

    # SINGLE REVIEW

    if len(cleaned_segments) == 1:

        text = cleaned_segments[0]

        (
            label,
            confidence,
            word_count,
            polarity,
            subjectivity,
            exclamations
        ) = analyze_individual_segment(
            text,
            model,
            vectorizer
        )

        batch_fake = 1 if label == "FAKE" else 0

        batch_genuine = 1 if label == "REAL" else 0

        log_review_to_file(
            text,
            word_count,
            polarity,
            subjectivity,
            exclamations,
            label,
            confidence
        )

        return jsonify({
            "multiple_mode": False,
            "label": label,
            "confidence": float(confidence),
            "demo_mode": model is None,
            "counters": {
                "total": 1,
                "genuine": batch_genuine,
                "fake": batch_fake
            },
            "features": {
                "word_count": int(word_count),
                "sentiment_polarity": float(polarity),
                "sentiment_subjectivity": float(subjectivity),
                "exclamation_count": int(exclamations)
            }
        })

    # MULTIPLE REVIEWS

    samples_list = []

    fake_count = 0
    real_count = 0

    for i, text in enumerate(cleaned_segments):

        (
            label,
            confidence,
            word_count,
            polarity,
            subjectivity,
            exclamations
        ) = analyze_individual_segment(
            text,
            model,
            vectorizer
        )

        if label == "FAKE":
            fake_count += 1
        else:
            real_count += 1

        log_review_to_file(
            text,
            word_count,
            polarity,
            subjectivity,
            exclamations,
            label,
            confidence
        )

        samples_list.append({
            "id": i + 1,
            "text": text,
            "verdict": label
        })

    total_batch = fake_count + real_count

    return jsonify({
        "multiple_mode": True,
        "total_rows": total_batch,
        "fake_count": fake_count,
        "real_count": real_count,
        "fake_percentage": round(
            (fake_count / total_batch) * 100, 1
        ) if total_batch else 0,
        "real_percentage": round(
            (real_count / total_batch) * 100, 1
        ) if total_batch else 0,
        "samples": samples_list,
        "counters": {
            "total": total_batch,
            "genuine": real_count,
            "fake": fake_count
        }
    })


# --------------------------------------------------
# BULK CSV PREDICTION
# --------------------------------------------------

@app.route("/bulk-predict", methods=["POST"])
def bulk_predict():

    if "file" not in request.files:

        return jsonify({
            "error": "No file uploaded."
        }), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":

        return jsonify({
            "error": "Empty file selection."
        }), 400

    if not uploaded_file.filename.lower().endswith(".csv"):

        return jsonify({
            "error": "Only CSV files are supported."
        }), 400

    try:

        df = safe_read_csv(uploaded_file)

        if df.empty or len(df.columns) == 0:

            return jsonify({
                "error": "CSV file is empty."
            }), 400

        # Detect review column

        text_col = None

        for col in df.columns:

            if str(col).strip().lower() in [
                "text",
                "review",
                "review_text"
            ]:

                text_col = col
                break

        if text_col is None:
            text_col = df.columns[0]

        # Remove empty reviews

        df = df.dropna(subset=[text_col])

        texts = [
            str(text).strip()
            for text in df[text_col].tolist()
            if str(text).strip()
        ]

        if not texts:

            return jsonify({
                "error": "No valid reviews found in CSV."
            }), 400

        model, vectorizer = load_model()

        all_predictions_mapped = []

        fake_count = 0
        real_count = 0

        # MODEL-BASED PREDICTION

        if model is not None and vectorizer is not None:

            X_vectors = vectorizer.transform(texts)

            predictions = model.predict(X_vectors)

            fake_count = int(np.sum(predictions == 1))

            real_count = int(np.sum(predictions == 0))

            for i, review_text in enumerate(texts):

                verdict = (
                    "FAKE"
                    if predictions[i] == 1
                    else "REAL"
                )

                all_predictions_mapped.append({
                    "id": i + 1,

                    # Complete text for CSV export
                    "text": review_text,

                    "verdict": verdict
                })

        # FALLBACK PREDICTION

        else:

            for i, row_text in enumerate(texts):

                exclamation_count = row_text.count("!")

                blob_analyzer = TextBlob(row_text.lower())

                polarity = float(
                    blob_analyzer.sentiment.polarity
                )

                subjectivity = float(
                    blob_analyzer.sentiment.subjectivity
                )

                if (
                    is_gibberish(row_text)
                    or exclamation_count > 4
                    or (
                        subjectivity > 0.82
                        and abs(polarity) > 0.82
                    )
                ):

                    verdict_label = "FAKE"

                    fake_count += 1

                else:

                    verdict_label = "REAL"

                    real_count += 1

                all_predictions_mapped.append({
                    "id": i + 1,

                    # Complete text for CSV export
                    "text": row_text,

                    "verdict": verdict_label
                })

        total_rows = fake_count + real_count

        return jsonify({

            "success": True,

            "total_rows": total_rows,

            "fake_count": fake_count,

            "real_count": real_count,

            "fake_percentage": round(
                (fake_count / total_rows) * 100, 1
            ) if total_rows else 0,

            "real_percentage": round(
                (real_count / total_rows) * 100, 1
            ) if total_rows else 0,

            "samples": all_predictions_mapped

        })

    except Exception as e:

        app.logger.exception("Bulk prediction failed")

        return jsonify({
            "error": f"Bulk processing failed: {str(e)}"
        }), 500


# --------------------------------------------------
# MODEL RETRAINING
# --------------------------------------------------

@app.route("/retrain", methods=["POST"])
def retrain():

    global current_accuracy

    if "file" not in request.files:

        return jsonify({
            "error": "No file uploaded."
        }), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":

        return jsonify({
            "error": "Empty file selection."
        }), 400

    if not uploaded_file.filename.lower().endswith(".csv"):

        return jsonify({
            "error": "Only CSV files are supported."
        }), 400

    try:

        new_df = safe_read_csv(uploaded_file)

        text_col = None
        label_col = None

        for col in new_df.columns:

            col_name = str(col).strip().lower()

            if col_name in [
                "text",
                "review",
                "review_text"
            ]:
                text_col = col

            if col_name in [
                "label",
                "target",
                "is_fake",
                "category"
            ]:
                label_col = col

        if text_col is None or label_col is None:

            return jsonify({
                "error": "CSV must contain text and label columns."
            }), 400

        new_df = new_df.dropna(
            subset=[text_col, label_col]
        )

        X_new = new_df[text_col].astype(str).tolist()

        y_new = pd.to_numeric(
            new_df[label_col],
            errors="coerce"
        )

        if y_new.isna().any():

            return jsonify({
                "error": "Labels must contain numeric values 0 or 1."
            }), 400

        y_new = y_new.astype(int).tolist()

        if set(y_new) != {0, 1}:

            return jsonify({
                "error": "Training data must contain both labels 0 and 1."
            }), 400

        if len(X_new) < 10:

            return jsonify({
                "error": "At least 10 labeled reviews are required."
            }), 400

        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english"
        )

        X_vectors = vectorizer.fit_transform(X_new)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vectors,
            y_new,
            test_size=0.2,
            random_state=42,
            stratify=y_new
        )

        model = LogisticRegression()

        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)

        current_accuracy = round(
            float(score * 100), 2
        )

        os.makedirs(
            os.path.dirname(MODEL_PATH),
            exist_ok=True
        )

        # Save trained model

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)

        return jsonify({

            "success": True,

            "new_accuracy": current_accuracy,

            "rows_processed": len(new_df)

        })

    except Exception as e:

        app.logger.exception("Model retraining failed")

        return jsonify({
            "error": f"Training failed: {str(e)}"
        }), 500


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )
