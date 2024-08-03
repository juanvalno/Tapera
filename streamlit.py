import streamlit as st
import pandas as pd
import re
import nltk
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Load and preprocess data
data = pd.read_csv('data/modeling.csv')

# Initialize vectorizer and fit
tfidf = TfidfVectorizer(max_features=8000)
tfidf.fit(data['clean_teks'])
X_tfidf = tfidf.transform(data['clean_teks'])
y = data['sentiment']

# Apply SMOTE to balance the data
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_tfidf, y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Initialize and train SVC model
svm_model = SVC(C=4.7450412719997725, 
                class_weight=None, 
                coef0=2.0, 
                degree=2, 
                gamma=10, 
                kernel='poly', 
                max_iter=4000, 
                shrinking=True, 
                tol=0.0001)
svm_model.fit(X_train, y_train)

# Initialize and train Multinomial Naive Bayes model
nb_model = MultinomialNB(alpha=0.15808361216819947)
nb_model.fit(X_train, y_train)

# Define text preprocessing functions
def casefolding(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

try:
    key_norm = pd.read_csv('data/key_norm.csv', encoding='latin1')
except UnicodeDecodeError:
    key_norm = pd.read_csv('data/key_norm.csv', encoding='ISO-8859-1')

def text_normalize(text):
    words = text.split()
    normalized_words = []
    for word in words:
        if (key_norm['singkat'] == word).any():
            normalized_word = key_norm[key_norm['singkat'] == word]['hasil'].values[0]
            normalized_words.append(normalized_word)
        else:
            normalized_words.append(word)
    return ' '.join(normalized_words)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stopwords_ind = stopwords.words('indonesian')
more_stopword = ['yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang',
                 'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                 'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt', 
                 '&amp', 'yah', 'zy_zy', 'mh']
stopwords_ind.extend(more_stopword)

def remove_stop_word(text):
    words = text.split()
    clean_words = [word for word in words if word not in stopwords_ind]
    return ' '.join(clean_words)

def stemming(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def text_preprocessing_process(text):
    text = casefolding(text)
    text = text_normalize(text)
    text = remove_stop_word(text)
    text = stemming(text)
    return text

# Function to classify text using the specified model
def classify_text(text, vectorizer, model):
    preprocessed_text = text_preprocessing_process(text)
    text_tfidf = vectorizer.transform([preprocessed_text])
    prediction = model.predict(text_tfidf)[0]
    return preprocessed_text, prediction

# Streamlit app layout
st.title("Text Sentiment TAPERA 2024")
st.markdown("Nama : Reyhan")
st.markdown("NIM  : A11.2020.1---")

user_input = st.text_area("Enter text for classification:")

if st.button("Tampilkan Text setelah Preprocessing"):
    preprocessed_text = text_preprocessing_process(user_input.strip())
    st.write("**Text setelah Preprocessing:**", preprocessed_text)

if st.button("Prediksi dengan SVM"):
    preprocessed_text, svm_result = classify_text(user_input.strip(), tfidf, svm_model)
    st.write("**SVC Prediksi:**", svm_result)

if st.button("Prediksi dengan Naive Bayes"):
    preprocessed_text, nb_result = classify_text(user_input.strip(), tfidf, nb_model)
    st.write("**Naive Bayes Prediksi:**", nb_result)
