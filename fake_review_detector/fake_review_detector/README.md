# FakeShield — Fake Review Detection Web App

FakeShield is a small Flask web application that detects likely fake product/service reviews using a combination of sentiment heuristics and an optional SVM + TF-IDF model. The app includes a clean frontend for quick, interactive checks and API endpoints for programmatic use.

---

## Project layout

```
fake_review_detector/
├── app.py                    # Flask backend (main file)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── model/                    # Put trained model files here (see below)
├── templates/                # Jinja2 HTML templates
│   └── index.html            # Main UI page
└── static/                   # CSS, JS, images
  ├── css/style.css
  └── js/app.js
```

---

## Quick start (development)

1. Create and activate a Python virtual environment (Windows PowerShell):

```powershell
cd fake_review_detector
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
python -m textblob.download_corpora
```

3. Run the app:

```powershell
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

---

## Model files (required for full accuracy)

Place the trained model artifacts in the `model/` folder:

- `classifier_model.pkl` (SVM or other classifier)
- `tfidf_vectorizer.pkl` (TF-IDF vectorizer used during training)

If these files are missing the app falls back to a demo rule-based analyzer (still useful for testing UI and integration).

---

## API

POST `/predict`
- Description: analyze a single review string and return prediction + features.
- Request JSON:

```json
{ "review": "This product is amazing! Best purchase ever!" }
```

- Example (PowerShell):

```powershell
Invoke-RestMethod -Method POST -Uri http://127.0.0.1:5000/predict -ContentType application/json -Body '{"review":"This product is amazing!"}' | ConvertTo-Json
```

GET `/health`
- Description: returns whether model and vectorizer files are present on disk.

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

Responses are JSON objects with keys such as `label`, `confidence`, `demo_mode`, and a `features` block.

---

## Frontend

The UI is at `templates/index.html` and static assets in `static/`. The frontend uses fetch to call `/predict` and `/health`. Sample reviews and history are implemented in `static/js/app.js`.

---

## Recommended next steps

- Move trained `*.pkl` model files to a release asset or enable Git LFS for `model/*.pkl` to avoid bloating the repo.
- Add a small CI workflow (GitHub Actions) to run linting and basic tests on push.
- Containerize with Docker for reproducible deployment.

---

## Docker (optional)

Example Dockerfile snippet (simple):

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 5000
CMD ["python", "app.py"]
```

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Open a PR with a clear description

---

## License

This project is provided as-is for educational purposes. Add your preferred license file if needed.

---

If you want, I can also:

- Add a `.github/workflows/python-app.yml` CI workflow.
- Convert the `model/` files to use Git LFS and rewrite history if you prefer.
