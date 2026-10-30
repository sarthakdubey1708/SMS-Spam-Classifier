
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

project_dir = Path(__file__).parent

data = pd.read_csv(
    project_dir / "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

print(data.head())
print("\nDataset size:", data.shape)
print("\nMessages by category:")
print(data["label"].value_counts())

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate messages:", data.duplicated().sum())
print("\nBefore cleaning:", len(data))

data = data.drop_duplicates().reset_index(drop=True)

print("After cleaning:", len(data))

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_dev_train, X_val, y_dev_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("\nDevelopment training:", len(X_dev_train))
print("Validation messages:", len(X_val))
print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

model = MultinomialNB(alpha=0.1)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

print("\nConfusion Matrix:")
print(cm)

message = ["Hey, are we meeting tomorrow at college?"]
message_tfidf = vectorizer.transform(message)
prediction = model.predict(message_tfidf)

print("\nCustom Message Prediction:")
print(prediction[0])

validation_vectorizer = TfidfVectorizer()

X_dev_tfidf = validation_vectorizer.fit_transform(X_dev_train)
X_val_tfidf = validation_vectorizer.transform(X_val)

validation_model = MultinomialNB(alpha=0.1)
validation_model.fit(X_dev_tfidf, y_dev_train)

val_pred = validation_model.predict(X_val_tfidf)

print("\nValidation Results:")
print(classification_report(y_val, val_pred))

joblib.dump(
    {
        "model": model,
        "vectorizer": vectorizer
    },
    project_dir / "spam_classifier.joblib"
)

print("\nModel saved successfully!")
