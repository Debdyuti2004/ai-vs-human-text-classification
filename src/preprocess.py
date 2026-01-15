import re
import nltk

# Do NOT use English stopwords anymore
# They remove meaningful Hindi tokens

def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Keep:
    # - English letters (a-z)
    # - Hindi characters (Unicode range)
    # - Spaces
    text = re.sub(r"[^a-z\u0900-\u097f\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

