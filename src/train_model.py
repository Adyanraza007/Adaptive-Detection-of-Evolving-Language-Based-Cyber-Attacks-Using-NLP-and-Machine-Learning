# ==============================
# 1. Import Libraries
# ==============================
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 2. Load Clean Dataset
# ==============================

df = pd.read_csv("final_clean_dataset.csv")

# Fix missing text values
df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str)

print("Dataset shape:", df.shape)

df = pd.read_csv("final_clean_dataset.csv")

# Remove rows where text is missing
df = df.dropna(subset=['text'])

# Convert text column to string (safety step)
df['text'] = df['text'].astype(str)

print("Dataset shape after cleaning:", df.shape)
# ==============================
# 3. Fix Labels (if needed)
# ==============================

df['label'] = df['label'].replace({
    'ham': 'normal',
    'spam': 'malicious'
})

df = df[df['label'].isin(['normal','malicious'])]
df = df.dropna()
# ==============================
# 4. Prepare Features and Labels
# ==============================

X = df['text']
y = df['label']


# ==============================
# 5. Convert Text → TF-IDF
# ==============================

vectorizer = TfidfVectorizer(max_features=5000)

X_tfidf = vectorizer.fit_transform(X)


# ==============================
# 6. Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==============================
# 7. Model 1: Naive Bayes
# ==============================

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("\nNaive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))


# ==============================
# 8. Model 2: Support Vector Machine
# ==============================

svm_model = LinearSVC()

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

print("\nSVM Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))


# ==============================
# 9. Model 3: Random Forest
# ==============================

rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# ==============================
# 10. Accuracy Comparison
# ==============================

print("\nModel Comparison")

print("Naive Bayes:", accuracy_score(y_test, nb_pred))
print("SVM:", accuracy_score(y_test, svm_pred))
print("Random Forest:", accuracy_score(y_test, rf_pred))

# Save model
pickle.dump(svm_model, open("model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("Model saved successfully!")
def predict_message(msg):
    msg = [msg]
    msg_vector = vectorizer.transform(msg)
    prediction = svm_model.predict(msg_vector)
    return prediction[0]

# Example
print(predict_message("Congratulations! You won a free iPhone"))
print(predict_message("Hey bro are we meeting today"))