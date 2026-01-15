import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import pickle
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from preprocess import clean_text


# --------------------------------
# Step 1: Load Dataset (KaggleHub)
# --------------------------------
dataset_id = "shamimhasan8/ai-vs-human-text-dataset"
dataset_path = kagglehub.dataset_download(dataset_id)

# Auto-detect CSV file
csv_file = None
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".csv"):
            csv_file = file
            break
    if csv_file:
        break

if csv_file is None:
    raise FileNotFoundError("No CSV file found in the dataset!")

print("Using CSV file:", csv_file)

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    dataset_id,
    csv_file,
)

print("Dataset shape:", df.shape)

# --------------------------------
# Step 2: Label Encoding
# --------------------------------
label_mapping = {
    "Human-written": 0,
    "AI-generated": 1
}

df["label_encoded"] = df["label"].map(label_mapping)
df = df.dropna(subset=["label_encoded"])

# --------------------------------
# Step 3: Text Preprocessing
# --------------------------------
df["clean_text"] = df["text"].apply(clean_text)

# --------------------------------
# Step 4: Improved TF-IDF Vectorizer
# --------------------------------
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["label_encoded"]

# --------------------------------
# Step 5: Train-Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------
# Step 6: Train Logistic Regression
# --------------------------------
model = LogisticRegression(
    max_iter=4000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# --------------------------------
# Step 7: Evaluation
# --------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Human-written", "AI-generated"]
))

# --------------------------------
# Step 8: Confusion Matrix (Save Image)
# --------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Human-written", "AI-generated"],
    yticklabels=["Human-written", "AI-generated"]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("Confusion matrix saved as confusion_matrix.png")

# --------------------------------
# Step 9: ROC Curve (Save Image)
# --------------------------------
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print("ROC curve saved as roc_curve.png")

# --------------------------------
# Step 10: Save Model & Vectorizer
# --------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ Model, vectorizer, and plots saved successfully!")
