# Absenteeism Prediction App – Setup Guide (Windows)

This guide helps you run the project locally from a ZIP download. It includes training the model, starting the API, and viewing the frontend.

## Prerequisites
- Python 3.12 (recommended) installed and added to PATH
- Windows PowerShell

## 1) Unzip and open a terminal
- Extract the ZIP to a simple path, e.g. `C:\\my_chat_app`.
- Open PowerShell and `cd` into the folder:

```powershell
cd "C:\\my_chat_app"
```

## 2) Create and activate a virtual environment
```powershell
python -m venv backend\\venv
backend\\venv\\Scripts\\Activate.ps1
```

If execution policy blocks activation, run PowerShell as Administrator and:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# Then try activation again
backend\\venv\\Scripts\\Activate.ps1
```

## 3) Install backend requirements
```powershell
pip install -r backend\\requirements.txt
```

## 4) Train the model (downloads dataset ZIP and reads the CSV within)
This generates the model and metadata files used at inference.

```powershell
python backend\\train_model.py
```

Artifacts created in `backend/`:
- `absenteeism_model.pkl` – trained model
- `model_info.json` – performance and fairness summary
- `feature_columns.json` – feature schema
- `feature_means.json` – per-feature means for imputing missing inputs

## 5) Start the API server
```powershell
python backend\\main.py
```
The API runs at `http://127.0.0.1:8000`.

Endpoints:
- `GET /info` – model metrics, limitations, fairness overview, schema
- `POST /predict` – JSON body with optional features; missing ones are imputed

Example request body:
```json
{
  "Age": 35,
  "Service time": 120,
  "Distance from Residence to Work": 10
}
```

## 6) Open the frontend
Open the static file in a browser (Chrome/Edge is fine):
- Double-click `frontend/index.html` OR
- Use PowerShell:

```powershell
start .\\frontend\\index.html
```

Notes:
- The frontend calls the API at `http://127.0.0.1:8000`. Keep the API running.
- Predictions are clipped to non-negative values.
- You can leave any inputs blank; the backend fills them with training means.

## Design and UX highlights
- Guided mode (tips + step-by-step tour)
- Tutorial modal with responsible-use guidance
- Fairness overview (baseline) and mitigation status
- Model card quick download (text format)

## Troubleshooting
- Module not found / missing packages:
  - Ensure the venv is active and re-run `pip install -r backend\\requirements.txt`.
- API not reachable from frontend:
  - Confirm `backend/main.py` is running and listening on `127.0.0.1:8000`.
  - Refresh the browser after API starts.
- ZIP reading error:
  - The training script already reads only the CSV inside the ZIP. Check network connectivity and re-run `train_model.py`.
- Negative predictions:
  - Backend clips predictions to 0; refresh the page if you still see old values.

## Re-training later
If you want to refresh the model with the latest dataset (same source), re-run:
```powershell
python backend\\train_model.py
```
Then restart the API:
```powershell
python backend\\main.py
```

## Project structure
```
my_chat_app/
  backend/
    main.py            # FastAPI app
    my_model.py        # Prediction + info helpers
    train_model.py     # Training + artifacts export
    requirements.txt   # Backend dependencies
    venv/              # (Your local virtual environment)
  frontend/
    index.html         # Simple static UI
  README/
    README.md          # This guide
```

## License & Data Source
- Dataset: UCI Machine Learning Repository – Absenteeism at Work
- This project is for educational use; assess suitability before production use.


