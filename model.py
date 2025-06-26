import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,accuracy_score,precision_score,f1_score,confusion_matrix,recall_score
import nltk #natural language toolkit
import re #regular expression
from nltk.corpus import stopwords
import joblib

#download stopwords
nltk.download("stopwords")
stop_words=set(stopwords.words("english"))

df=pd.read_csv("IMDB Dataset.csv")

#mapping the sentiment to some numerical value
df["sentiment"]=df["sentiment"].map({
    "positive":1,
    "negative":0
})

#clean text
def clean_text(text):
  text=re.sub(r"[^a-zA-Z]"," ",text).lower()
  tokens=text.split()
  tokens=[word for word in tokens if word not in stop_words]
  return " ".join(tokens)

#apply the clean_text function on review
df["cleaned_review"]=df["review"].apply(clean_text)
#loop through entire series of column df["review"] and each single row will be passed as an argument to my function

#frequency of words by feature extraction
vectorizer=CountVectorizer(max_features=5000)
X=vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]

#Divide the dataset into train-test-split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)

#Train the model
model = MultinomialNB()
model.fit(X_train,y_train)

#Make the prediction
y_pred = model.predict(X_test)

#Calculate the performance metrics
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_pred,y_test)
f1 = f1_score(y_test,y_pred)
cm = confusion_matrix(y_pred,y_test)
cr = classification_report(y_pred,y_test)

print("The accuracy is: ",accuracy)
print("the precision is:", precision)
print("the recall is:",recall)
print("the f1 score:", f1)
print("the confusion matrix:", cm)
print("the classification report:\n", cr)

#save the model and vectorizer
joblib.dump(model,"sentimental_model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

print("Model and Vectorizer has been saved")