from nltk.corpus import stopwords
import re
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
import time
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('wordnet')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
tokenizer = TweetTokenizer()

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\S+', '', text)
    # Tokenize text using TweetTokenizer
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos=wordnet.VERB) for token in tokens]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    # Join tokens back into a string
    text = ' '.join(tokens)
    return text

data = pd.read_csv('sentiment_data.csv')

labels = data['label'].replace({0: 0, 4: 1}) # convert 4 to 1 for positive label

data['content'] = data['content'].apply(preprocess_text)

data.dropna(subset=['content'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data["content"], labels, test_size=0.2, random_state=61)
print(X_train.shape)


tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
vocabulary = tfidf_vectorizer.vocabulary_

unique_word_count = len(vocabulary)

print("Unique word count:", unique_word_count)


lr_model = LogisticRegression( max_iter=1000,solver="liblinear")

start_time = time.time()
lr_model.fit(X_train_tfidf, y_train)
elapsed_time = time.time() - start_time
print(elapsed_time)

accuracy = lr_model.score(X_test_tfidf, y_test)

print(f"Accuracy: {accuracy}")


joblib.dump(lr_model, "lr_modelTFIDF.joblib")


y_pred = lr_model.predict(X_test_tfidf)
print("Classification report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


fpr, tpr, _ = roc_curve(y_test, lr_model.predict_proba(X_test_tfidf)[:,1])
roc_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test_tfidf)[:,1])
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()




## TROLL + SENT MODEL:

data = pd.read_csv('merged_dataset.csv')
data['content'] = data['content'].apply(preprocess_text)

data.dropna(subset=['content'], inplace=True)

X_train=data["content"]

X = tfidf_vectorizer.fit_transform(X_train)


predicted_sentiments = lr_model.predict(X)

data["sentiment"] = predicted_sentiments

data.to_csv("troll_sentiments_LR_TFIDF.csv", index=False)


def merge(text):
    sentiment_word = 'Negative' if text['sentiment'] == 0 else 'Positive'
    return text['content'] + ' ' + sentiment_word


data = pd.read_csv("troll_sentiments_LR_TFIDF.csv")




data = data.dropna(subset=['content'])
data = data[data['content'] != '']


testo=data['content']

data['content'] = data.apply(merge, axis=1)
data['content'] = data['content'].apply(preprocess_text)

data = data.dropna(subset=['content'])
data = data[data['content'] != '']

data = data.drop('sentiment', axis=1)


data = data.replace(r'^\s*$',np.nan , regex=True)
data = data.dropna()
tfidf_vectorizer =  TfidfVectorizer(ngram_range=(1,3))
X = tfidf_vectorizer.fit_transform(data['content'])

vocabulary = tfidf_vectorizer.vocabulary_

unique_word_count = len(vocabulary)

print("Unique word count:", unique_word_count)

X_df = pd.DataFrame.sparse.from_spmatrix(X)

y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

X_train.columns = X_train.columns.astype('U')
X_test.columns = X_test.columns.astype('U')
model = LogisticRegression( max_iter=1000,solver="liblinear")

start_time = time.time()
model.fit(X_train, y_train)
elapsed_time = time.time() - start_time
print(elapsed_time)


joblib.dump(model, "FINAL_lr_modelTFIDF.joblib")

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred))

sns.set(font_scale=1.4)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot the ROC curve
y_pred_proba = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc=4)
plt.show()