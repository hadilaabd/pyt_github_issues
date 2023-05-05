import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load data from the csv file
data = pd.read_csv("pytorch_issues.csv", on_bad_lines='skip')

# Remove rows with missing fields
data = data.dropna()

# Define function to clean text data
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove digits
    text = re.sub(r"\d+", "", text)
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    # Remove words with less than 3 characters
    tokens = [token for token in tokens if len(token) > 2]
    # Join tokens back into a string
    text = " ".join(tokens)
    return text

# Apply clean_text function to the "body" column of the data
data["clean_text"] = data["Body"].apply(clean_text)

# Save cleaned data to a new csv file
data.to_csv("cleaned_data.csv", index=False)