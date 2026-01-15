import pickle
from preprocess import clean_text

# Load trained artifacts
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_text(text):
    clean = clean_text(text)
    vector = vectorizer.transform([clean])
    pred = model.predict(vector)[0]
    return "AI-generated" if pred == 1 else "Human-written"


if __name__ == "__main__":
    sample_text = input('Enter the Text here: ')
    print("Prediction:", predict_text(sample_text))

