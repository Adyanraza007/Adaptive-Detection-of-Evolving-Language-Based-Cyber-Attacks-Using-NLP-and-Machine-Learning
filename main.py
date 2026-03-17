# ==============================
# 1. Import Required Libraries
# ==============================

import pandas as pd
import re

# ==============================
# 2. Function to Clean Text
# ==============================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-zA-Z ]", " ", text)    # remove special characters
    text = re.sub(r"\s+", " ", text)           # remove extra spaces
    return text.strip()

# ==============================
# 3. Load Datasets
# ==============================

spam_df = pd.read_csv("spam.csv", encoding="latin1")
email_df = pd.read_csv("email.csv")
emails_df = pd.read_csv("emails.csv")
scam_df = pd.read_csv("Financial scams detection dataset.csv")
large_email_df = pd.read_csv("email_dataset_500k.csv")

# ==============================
# 4. Clean spam.csv
# ==============================

spam_df = spam_df[['v1','v2']]
spam_df.columns = ['label','text']

spam_df['label'] = spam_df['label'].replace({
    'ham':'normal',
    'spam':'malicious'
})

spam_df['text'] = spam_df['text'].apply(clean_text)

# ==============================
# 5. Clean email.csv
# ==============================

email_df = email_df[['Category','Message']]
email_df.columns = ['label','text']

email_df['label'] = email_df['label'].replace({
    'ham':'normal',
    'spam':'malicious'
})

email_df['text'] = email_df['text'].apply(clean_text)

# ==============================
# 6. Clean emails.csv
# ==============================

emails_df = emails_df[['text','spam']]
emails_df.columns = ['text','label']

emails_df['label'] = emails_df['label'].replace({
    0:'normal',
    1:'malicious'
})

emails_df['text'] = emails_df['text'].apply(clean_text)

# ==============================
# 7. Clean Financial Scam Dataset
# ==============================

scam_df = scam_df.iloc[:, :2]   # take first two columns
scam_df.columns = ['text','label']

scam_df['label'] = 'malicious'

scam_df['text'] = scam_df['text'].apply(clean_text)

# ==============================
# 8. Use Sample from Large Dataset
# ==============================

large_email_df = large_email_df.sample(20000)

large_email_df = large_email_df.iloc[:, :2]
large_email_df.columns = ['text','label']

large_email_df['label'] = large_email_df['label'].replace({
    0:'normal',
    1:'malicious'
})

large_email_df['text'] = large_email_df['text'].apply(clean_text)

# ==============================
# 9. Combine All Datasets
# ==============================

final_df = pd.concat([
    spam_df[['text','label']],
    email_df[['text','label']],
    emails_df[['text','label']],
    scam_df[['text','label']],
    large_email_df[['text','label']]
])

# ==============================
# 10. Remove Duplicates
# ==============================

final_df = final_df.drop_duplicates()

# ==============================
# 11. Shuffle Dataset
# ==============================

final_df = final_df.sample(frac=1).reset_index(drop=True)

# ==============================
# 12. Check Dataset Summary
# ==============================

print("Final Dataset Shape:", final_df.shape)
print(final_df['label'].value_counts())

# ==============================
# 13. Save Clean Dataset
# ==============================

# Fix label inconsistencies

final_df['label'] = final_df['label'].replace({
    'ham': 'normal',
    'spam': 'malicious'
})

# Remove corrupted rows
final_df = final_df[final_df['label'].isin(['normal','malicious'])]

# Check again
print(final_df['label'].value_counts())

final_df.to_csv("final_clean_dataset.csv", index=False)

print("Dataset cleaning and merging completed successfully!")
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(final_df['text'])
y = final_df['label']