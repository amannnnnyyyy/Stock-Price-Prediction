import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import LdaModel
import re
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocess the text
def preprocess_text(text):
  
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stem or lemmatize words (optional)
    # stemmer = PorterStemmer()
    # words = [stemmer.stem(word) for word in words]

    # Join the words back into a string
    preprocessed_text = ' '.join(words)

    return preprocessed_text


# Create a TF-IDF matrix
def tdf_matrix(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['headline'])
    return X, vectorizer

# Perform topic modeling
def topic_model(X):
    num_topics = 5  # Adjust the number of topics as needed
    lda_model = LdaModel(X, num_topics=num_topics)
    return lda_model

# Print the topics
def print_topics(lda_model):
    for topic in lda_model.print_topics(num_words=10):
        print(topic)

# Extract keywords based on TF-IDF scores
def extract_keywords(X, vectorizer):
    keywords = vectorizer.get_feature_names_out()
    tfidf_scores = X.max(axis=0)
    keyword_scores = [(keywords[i], tfidf_scores[i]) for i in range(len(keywords))]
    keyword_scores = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
    print(keyword_scores)