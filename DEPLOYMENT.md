# Deployment Guide — Streamlit Community Cloud

This app is deployed for free on Streamlit Community Cloud, which runs it straight from a public GitHub repo.

## Prerequisites (one-time)
- A GitHub account (username: PavanTeja-hash)
- A free account at https://share.streamlit.io (sign in with GitHub)

## Step 1 — Push the project to GitHub
Run these in the project folder. The repo `Customer-Loan-Risk-Prediction-System` already exists on GitHub from earlier.

```bash
cd "C:\Users\maddi\OneDrive\Desktop\Customer-Loan-Risk-Prediction-System"

git add .
git commit -m "Add upgrades (XGBoost, tuning, explainability) and Streamlit app for deployment"
git push origin main
```

If `git push` complains the remote has commits you don't have locally, run `git pull --rebase origin main` first, then push again.

## Step 2 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **"Create app"** → **"Deploy a public app from GitHub"**.
3. Fill in:
   - **Repository:** `PavanTeja-hash/Customer-Loan-Risk-Prediction-System`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. First build takes a few minutes (it installs everything in `requirements.txt`).
5. You'll get a public URL like `https://customer-loan-risk-prediction-system.streamlit.app` — put this on your resume.

## How it works (for your understanding)
- Streamlit Cloud reads `requirements.txt`, installs those exact pinned versions, then runs `app.py`.
- `app.py` loads `models/best_model.joblib` (the tuned XGBoost pipeline, committed to the repo, ~1.8MB).
- Versions are pinned in `requirements.txt` so the saved model loads without version-mismatch errors.

## If you retrain the model
If you change the model and regenerate `models/best_model.joblib` (via `python src/train_final_model.py`), commit and push it again — Streamlit Cloud auto-redeploys on every push to `main`.

## Updating the app later
Any `git push` to `main` triggers an automatic redeploy. No extra steps.
```
