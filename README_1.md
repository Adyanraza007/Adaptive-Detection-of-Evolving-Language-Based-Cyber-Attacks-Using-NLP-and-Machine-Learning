# Adaptive Detection and Analysis of Contextual Social Engineering Messages Using Machine Learning

B.Tech CSE Academic Project | 6th Semester | 3-Member Team

---

## Project Overview

An ML + NLP system that detects deceptive/social engineering messages
(phishing emails, scam SMS, manipulation texts) and demonstrates
adaptability to evolving attack patterns through a structured
time-based retraining experiment.

---

## Folder Structure

```
social_engineering_detector/
│
├── data/
│   ├── raw/                        # Original unmodified datasets
│   │   ├── spam.csv
│   │   ├── email.csv
│   │   ├── emails.csv
│   │   ├── financial_scams.csv
│   │   └── email_dataset_500k.csv
│   │
│   ├── processed/                  # Cleaned, merged dataset
│   │   └── final_clean_dataset.csv
│   │
│   └── splits/                     # T1, T2a, T2b splits for experiment
│       ├── T1_train.csv
│       ├── T2a_drift.csv
│       └── T2b_test.csv
│
├── src/                            # Core source code
│   ├── preprocess.py               # Data loading, cleaning, merging
│   ├── feature_engineering.py      # TF-IDF + psychological features  [NEXT]
│   └── train_model.py              # Model training and evaluation
│
├── experiments/                    # Adaptability experiment
│   └── adaptive_experiment.py      # T1 → T2a → retrain → T2b
│
├── models/                         # Saved model artifacts
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── app/                            # Streamlit inference interface  [LATER]
│   └── app.py
│
├── outputs/
│   └── reports/                    # Evaluation results, plots
|   └──screenshots                  # Screenshots of outputs
│
├── notebooks/                      # Jupyter notebooks for exploration
│
|── requirements.txt                # requirements of this project
|
└── README.md
```

---

## Pipeline Stages

| Stage | File | Status |
|-------|------|--------|
| 1. Data preprocessing | src/preprocess.py | Done (fixed) |
| 2. Feature engineering | src/feature_engineering.py | Next |
| 3. Model training | src/train_model.py | Done (fixed) |
| 4. Adaptability experiment | experiments/adaptive_experiment.py | Done (fixed) |
| 5. Streamlit app | app/app.py | Pending |

---

## Key Fixes Applied

- Financial scams dataset: ham rows now correctly labeled as `normal`
- Vectorizer: upgraded from unigrams `(1,1)` to bigrams `(1,2)`
- Adaptive experiment: removed data leakage in vectorizer re-fitting
- Large dataset sampling: `random_state=42` added for reproducibility
- Class imbalance: to be addressed in feature_engineering.py

---

## How to Run

```bash
# Step 1: Preprocess all raw datasets
python src/preprocess.py

# Step 2: Train models (after feature engineering)
python src/train_model.py

# Step 3: Run adaptability experiment
python experiments/adaptive_experiment.py

# Step 4: Launch inference app
streamlit run app/app.py
```

---

## Datasets Used

| Dataset | Source | Size |
|---------|--------|------|
| spam.csv | SMS Spam Collection (UCI) | ~5500 |
| email.csv | Email spam dataset | ~5500 |
| emails.csv | Enron email corpus | ~5700 |
| financial_scams.csv | Financial fraud messages | ~523 |
| email_dataset_500k.csv | Large synthetic email dataset | 500,000 (sampled 20k) |
