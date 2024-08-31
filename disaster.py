import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import streamlit as st

#pip install streamlit 

df = pd.read_csv('disaster_tweets_data(DS).csv')
st.subheader('Disaster Tweets')
st.dataframe(df.head())




import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import string




sw = stopwords.words('english')
print(sw)


t = df['tweets'].astype(str)
lm = WordNetLemmatizer()
for i in df['tweets'].iloc[:5]:
  print(i)

nav = st.sidebar.radio("Select Countplot Feature",["tweets","target"])

f1 = plt.figure(figsize=(3,3))
sns.countplot(y ='target',data=df)
st.pyplot(f1)


df1 = df[['tweets','target']].copy()


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df1['tweets']=lb.fit_transform(df1['tweets'])
df1['target']=lb.fit_transform(df1['target'])

st.subheader('Data After LabelEncoding')
st.dataframe(df1.head(5))

x = df1.iloc[:,:-1]
y = df1.iloc[:,-1]
st.write(x.shape,y.shape)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'LogisticRegression', 'Naive Bayes')
)

def select_param(clf_name):
    params = {}
    if clf_name == 'KNN':
        K = st.sidebar.slider('K',1,15)
        params['K']=K
    elif clf_name == 'LogisticRegression':
        L = st.sidebar.slider('L',1,15)
        params['L']=L
    elif clf_name == 'Naive Bayes':
        N = st.sidebar.slider('N',1,15)
        params['N']=N
    else:
        None
    return params

params = select_param(classifier_name)

def get_classifier(clf_name,params):
    clf = None
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'LogisticRegression':
        clf = LogisticRegression(max_iter=params['L'])
    elif clf_name == 'Naive Bayes':
        clf = LogisticRegression(max_iter=params['N'])
    else:
        None
    return clf

model = get_classifier(classifier_name,params)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
st.write(cm)
st.write(f'Classification Report\n',classification_report(y_test,y_pred))
