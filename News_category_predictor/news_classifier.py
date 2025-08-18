import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load Dataset
try:
    df = pd.read_csv("news_raw.csv", quotechar='"', on_bad_lines="skip")
    print("📂 Dataset loaded successfully.")
except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

print("Total samples:", len(df))
print("Columns:", df.columns)

# Ensure correct column names
if "text" not in df.columns or "category" not in df.columns:
    raise ValueError("Dataset must have 'text' and 'category' columns.")

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
)

# 3. Pipeline (TF-IDF + Naive Bayes Classifier)
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),       # unigrams + bigrams
    max_features=5000,        # limit vocab size
    sublinear_tf=True         # better scaling
)

model = make_pipeline(vectorizer, MultinomialNB())

# 4. Train the model
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Accuracy:", round(accuracy * 100, 2), "%")

print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔎 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example prediction with confidence
sample_text = "প্রবাস আয় বৃদ্ধি"
probs = model.predict_proba([sample_text])[0]
predicted_class = model.classes_[probs.argmax()]
confidence = round(probs.max() * 100, 2)

print("\n📰 Example prediction:")
print("Text:", sample_text)
print("Predicted Category:", predicted_class, f"({confidence}%)")

# 6. Save model and accuracy
joblib.dump(model, "news_classifier_model.pkl")
joblib.dump({"accuracy": accuracy}, "model_info.pkl")

print("\n💾 Model saved as news_classifier_model.pkl")
print("💾 Accuracy saved in model_info.pkl")
