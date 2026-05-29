# FakeShield — Fake Review Detection Web App
### Capstone Project · Student 4 · Web Application Module

---

## 📁 Project Structure

```
fake_review_detector/
│
├── app.py                    ← Flask backend (main file)
├── requirements.txt          ← Python dependencies
├── README.md
│
├── model/                    ← PUT TEAMMATE'S FILES HERE
│   ├── svm_model.pkl         ← From Student 2
│   └── tfidf_vectorizer.pkl  ← From Student 1
│
├── templates/
│   └── index.html            ← Main UI page
│
└── static/
    ├── css/style.css         ← Styling
    └── js/app.js             ← Frontend logic
```

---

## ⚙️ Setup Instructions

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

### Step 2 — Add model files (from teammates)
Place these files inside the `model/` folder:
- `svm_model.pkl` → from **Student 2** (trained SVM model)
- `tfidf_vectorizer.pkl` → from **Student 1** (fitted TF-IDF vectorizer)

> ⚠️ Without these files, the app runs in **Demo Mode** using rule-based logic.

### Step 3 — Run the app
```bash
python app.py
```
Open your browser at: **http://localhost:5000**

---

## 🔗 How It Connects to Other Parts

| Module | File | What it does |
|--------|------|--------------|
| Student 1 (Features) | `tfidf_vectorizer.pkl` | Transforms review text into TF-IDF vector |
| Student 2 (Model) | `svm_model.pkl` | Predicts FAKE or REAL label |
| Student 3 (Evaluation) | — | Use `/health` endpoint to verify model is loaded |
| Student 5 (Advanced) | `app.py` → add new route | Add `/predict-url` route for link-based checking |

---

## 🌐 API Endpoints

### POST `/predict`
Send a review text, get prediction back.

**Request:**
```json
{ "review": "This product is absolutely amazing!!" }
```

**Response:**
```json
{
  "label": "FAKE",
  "confidence": 87.3,
  "demo_mode": false,
  "features": {
    "sentiment_polarity": 0.8,
    "sentiment_subjectivity": 0.75,
    "word_count": 7,
    "exclamation_count": 2,
    "upper_ratio": 0.02
  }
}
```

### GET `/health`
Check if model files are loaded.
```json
{ "status": "ok", "model_loaded": true, "vectorizer_loaded": true }
```

---

## 🤝 For Student 2 — Saving the Model

After training, save the model like this:
```python
import pickle

# Save the SVM model
with open("model/svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

# Save the TF-IDF vectorizer (Student 1 does this)
with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
```

---

## 🤝 For Student 5 — Adding Advanced Features

To add a URL/link-based review checker, add this route to `app.py`:
```python
@app.route("/predict-url", methods=["POST"])
def predict_url():
    data = request.get_json()
    url = data.get("url", "")
    # scrape reviews from url, then call predict_review() on each
    ...
```

---

## 🎨 Features of the Web App
- ✅ Clean, dark-themed professional UI
- ✅ Real-time character & word count
- ✅ Sample review buttons for demo
- ✅ Confidence score with animated ring
- ✅ Feature breakdown (sentiment, word count, etc.)
- ✅ Last 5 checks history
- ✅ Demo mode (works without model file)
- ✅ Live model badge when model is loaded
- ✅ Mobile responsive


Hello Brother I am Nilesh Kumawat :-
i have done my work on google colab and did some modification ,you can try this now 
 1. open the folder with terminal 
 2. install flask pandas scikit-learn nltk textblob using pip
 3. run python app.py
 4. you can check now the review is fake or genuine .
 5. I will add some more feature within 3 to 4 days.
 6. You can try this and add some more feature

 bhai mene isko modify kiya app.py ko ab agle member ki baari 