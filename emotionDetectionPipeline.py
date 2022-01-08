# Packages
import pandas as pd
import numpy as np
import string
# Cleaning text
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
# Save
import joblib

#data
df = pd.read_csv('data/emotionsData_raw.csv')


def clean(text, stemm = False):
    words = word_tokenize(text)
    stopw = set(stopwords.words('english'))
    tokens = [token for token in words if token not in stopw]
    tokens = [token for token in tokens
                                if 'http' not in token
                                and not token.startswith('@')
                                and not token.startswith('#')
                                and not token.startswith('&')
                                and not token.startswith('http')
                                and not token.startswith('www')
                                and token != 'RT'
                            ]
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word) for word in tokens]
    
    if stemm:
        ps = PorterStemmer()
        tokens = [ps.stem(token) for token in tokens]
    
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word) for word in tokens]
    tokens = [w.lower() for w in tokens if w.lower() not in string.punctuation]
    tokens = [' '.join(token for token in tokens if not token.isdigit()) ]
    
    
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text 


df['Clean_Text'] = df['Text'].apply(clean)


# Features
content = df['Clean_Text']
labels = df['Emotion']

# Split Data
text_train, text_test, label_train, label_test = train_test_split(content, labels,test_size=0.3)

# Model Build
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                        # ('clf-svm',
                         # SGDClassifier(loss='modified_huber',penalty='l2',alpha=1e-3,max_iter=1000,tol=1e-6,random_state=42),
                         ('lr', LogisticRegression(max_iter=1000, C=3, random_state = 42, penalty='l2', solver='newton-cg')
                         ),])



_ = text_clf.fit(text_train, label_train)

# Evaluation
text_clf.score(text_test, label_test)
test = 'This shirt was great!!'
print(text_clf.predict([test]))
print(text_clf.predict_proba([test]))
print(text_clf.classes_)
predicted = text_clf.predict(text_test)
print(np.mean(predicted == label_test))

# Save Model
pipeline_file = open("App/models/emotion_classifier_final", "wb")
joblib.dump(text_clf,pipeline_file)
pipeline_file.close()