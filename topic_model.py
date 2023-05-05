import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import matplotlib.pyplot as plt

nltk.download('wordnet')

# Load the cleaned data
df = pd.read_csv('cleaned_data.csv')

# Define a list of stopwords to be removed
stop_words = set(stopwords.words('english'))
stop_words.update(['and', 'to', 'in', 'of', 'with', 'at', 'on', 'for', 'a', 'an', 'the'])

# Define a function to tokenize and preprocess the text
def tokenize(text):
    months = ['january', 'jan', 'february', 'feb', 'march', 'mar', 'april', 'apr', 'may', 'june', 'jun', 'july', 'jul', 'august', 'aug', 'september', 'sep', 'october', 'oct', 'november', 'nov', 'december', 'dec']

    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Remove words less than 3 characters
    tokens = [token for token in tokens if len(token) > 2]
    
    # Remove numbers
    tokens = [token for token in tokens if not token.isnumeric() and token not in months]
    
    # Remove verb to be variations and other forms of comparison
    tokens = [token for token in tokens if not re.match(r'^be.*|^compare.*', token)]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

if __name__ == '__main__':
    # Tokenize and preprocess the text
    df['tokens'] = df['Body'].apply(tokenize)

    # Create a dictionary from the tokens
    dictionary = corpora.Dictionary(df['tokens'])

    # Convert the tokens to a bag of words corpus
    corpus = [dictionary.doc2bow(tokens) for tokens in df['tokens']]

    # Train an LDA model on the corpus
    lda_model = models.LdaMulticore(corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

    # Visualize the topics
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_topics.html')

    # Create a bar chart for the top 10 keywords in each topic
    topics = lda_model.show_topics(num_topics=10, num_words=10, formatted=False)
    for i, topic in enumerate(topics):
        keywords = [word[0] for word in topic[1]]
        weights = [word[1] for word in topic[1]]
        plt.bar(keywords, weights)
        plt.title('Topic ' + str(i+1))
        plt.xlabel('Keywords')
        plt.ylabel('Weights')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('topic_' + str(i+1) + '_chart.png')
        plt.clf()