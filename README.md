# AI vs Human Text Detection

## 📌 Overview
AI vs Human Text Detection is a machine learning project that classifies text as **AI-generated** or **human-written** using Natural Language Processing (NLP) techniques. The project focuses on building an efficient, interpretable, and reproducible text classification pipeline suitable for academic use.

---

## 🎯 Problem Statement
With the rapid growth of AI-generated content, it has become important to distinguish between machine-generated and human-written text. This project addresses this problem using traditional machine learning techniques rather than large transformer models, ensuring stability and explainability.

---

## 🧠 Approach
The project follows a standard NLP pipeline:
1. Text preprocessing (cleaning, normalization, stopword removal)
2. Feature extraction using **TF-IDF**
3. Classification using **Logistic Regression**
4. Model evaluation using multiple metrics

---

## 🗂 Dataset
- **Source:** Kaggle  
- **Dataset Name:** AI vs Human Text Dataset  
- **Classes:**
  - `AI-generated`
  - `Human-written`
- The dataset is balanced and suitable for binary text classification tasks.

---

## 🛠 Technologies Used
- Python
- Scikit-learn
- Pandas
- NLTK
- Matplotlib & Seaborn
- KaggleHub

---

## 📊 Model Evaluation
The trained model is evaluated using:
- **Accuracy**
- **Confusion Matrix**
- **ROC Curve & AUC Score**

Evaluation plots are saved as image files:
- `confusion_matrix.png`
- `roc_curve.png`

---

## 📁 Project Structure
AI_vs_human_text_dl/
│
├── src/
│ ├── preprocess.py
│ ├── train.py
│ └── predict.py
│
├── model.pkl
├── vectorizer.pkl
├── confusion_matrix.png
├── roc_curve.png
├── README.md
└── LICENSE


---

## ▶️ How to Run

1️⃣ Install Dependencies
```bash
pip install scikit-learn pandas nltk matplotlib seaborn kagglehub

2️⃣ Train the Model

From the project root directory:

python -m src.train


This will generate:

Trained model (model.pkl)

TF-IDF vectorizer (vectorizer.pkl)

Evaluation plots

3️⃣ Predict on New Text
python -m src.predict


Enter any text when prompted to see the prediction.

🧪 Example Prediction
Enter the Text here:
I remember struggling with programming during my first semester.

Prediction: Human-written

📄 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute this project with proper attribution.


👤 Author
Debdyuti Chakraborty
Developed as an academic project for learning and experimentation in NLP and Machine Learning.


