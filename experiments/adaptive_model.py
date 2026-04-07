import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("final_clean_dataset.csv")

df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str)

df['label'] = df['label'].replace({
    'ham': 'normal',
    'spam': 'malicious'
})

df = df[df['label'].isin(['normal','malicious'])]

# ==============================
# 2. Simulate OLD vs NEW data
# ==============================

# First 70% = OLD data
old_data = df.iloc[:int(0.7*len(df))]

# Last 30% = NEW data (simulates evolving attacks)
new_data = df.iloc[int(0.7*len(df)):]

# ==============================
# 3. Train on OLD Data
# ==============================

vectorizer = TfidfVectorizer(max_features=5000)

X_old = vectorizer.fit_transform(old_data['text'])
y_old = old_data['label']

model = LinearSVC()
model.fit(X_old, y_old)

# ==============================
# 4. Test on NEW Data
# ==============================

X_new = vectorizer.transform(new_data['text'])
y_new = new_data['label']

pred_old = model.predict(X_new)

accuracy_old = accuracy_score(y_new, pred_old)

print("Accuracy on NEW data (before adaptation):", accuracy_old)

# ==============================
# 5. Detect Concept Drift
# ==============================

threshold = 0.90

if accuracy_old < threshold:
    print("\n⚠️ Concept Drift Detected! Model needs adaptation.")

    # ==============================
    # 6. Retrain Model (Adaptation)
    # ==============================

    combined_text = pd.concat([old_data['text'], new_data['text']])
    combined_label = pd.concat([old_data['label'], new_data['label']])

    X_combined = vectorizer.fit_transform(combined_text)

    model.fit(X_combined, combined_label)

    # Test again
    X_new = vectorizer.transform(new_data['text'])
    pred_new = model.predict(X_new)

    accuracy_new = accuracy_score(y_new, pred_new)

    print("Accuracy after adaptation:", accuracy_new)

else:
    print("\n✅ No significant drift detected. Model is stable.")
    