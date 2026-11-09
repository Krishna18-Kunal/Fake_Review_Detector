# Fake Review Detector

An AI/ML-based web application that detects potentially fake and genuine product reviews using Natural Language Processing (NLP) and Machine Learning.

## Features

- **Single Review Detection:** Analyze one review at a time.
- **Multiple Review Analysis:** Analyze multiple reviews together.
- **Bulk CSV Analysis:** Upload a CSV file containing multiple reviews.
- **CSV Export:** Download prediction results as a CSV file.
- **Prediction Summary:** View genuine and deceptive review counts.
- **Prediction History:** Maintain local prediction logs.
- **Model Health Check:** Check whether the ML model and TF-IDF vectorizer are loaded.
- **Model Retraining:** Supports model retraining through the application.

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TextBlob
- TF-IDF
- HTML, CSS and JavaScript

## Project Structure

```text
Fake_Review_Detector_Final/
¦
+-- app.py
+-- requirements.txt
+-- README.md
+-- .gitignore
¦
+-- model/
¦   +-- classifier_model.pkl
¦   +-- tfidf_vectorizer.pkl
¦
+-- static/
¦   +-- css/
¦       +-- style.css
¦
+-- templates/
    +-- index.html
```

## Installation and Setup

### 1. Download or Clone the Repository

Download the ZIP file and extract it, or clone the repository:

```bash
git clone https://github.com/Krishna18-Kunal/Fake_Review_Detector.git
```

### 2. Open the Project Folder

Open the extracted project folder in VS Code or a terminal.

### 3. Create and Activate Virtual Environment (Optional)

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

### 6. Open in Browser

Visit:

```text
http://127.0.0.1:5000
```

## Model Files

The application uses:

- `classifier_model.pkl` – Trained machine learning classifier.
- `tfidf_vectorizer.pkl` – TF-IDF text vectorizer.

Both files are stored inside the `model/` directory.

## Important Note

Predictions indicate whether a review is potentially fake or genuine based on the trained model. A prediction is not proof that a review is actually fake or genuine.

Model performance should be evaluated using a separate labeled test dataset.

## Author

**Krishna Kunal**

GitHub: https://github.com/Krishna18-Kunal

## Project Type

Academic Machine Learning and Natural Language Processing Project.




.
