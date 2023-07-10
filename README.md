# SPAM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.warn('Ignore')
# plt.style.use('dark_background'
sns.set_style("dark")
print('Done')
df = pd.read_csv('/content/spam.csv', encoding='latin-1')
df.head()
df.shape
df.info()
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True,axis=1)
df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
df.columns
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target']=encoder.fit_transform(df['target'])
df.isnull().sum()
df.duplicated().sum
df=df.drop_duplicates(keep='first') # deletes the first occurrence.
df.shape
df['target'].value_counts()
plt.pie(df['target'].value_counts(), labels=['ham','spam'], autopct='%0.2f',colors = ['#ff9999','#66b3ff'])
plt.show()# Natural Language Toolkit
import nltk
df['num_chars']=df['text'].apply(len)
df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_sentence']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
df.head()
df[['num_chars','num_words','num_sentence']].describe()
df[df['target']==0][['num_chars','num_words','num_sentence']].describe()
df[df['target']==1][['num_chars','num_words','num_sentence']].describe()
plt.figure(figsize=(14,5))
sns.histplot(data=df,x='num_chars',hue="target",palette="inferno",kde=True);
plt.figure(figsize=(14,5))
sns.histplot(data = df,hue='target',x='num_words',palette="inferno", kde=True);
plt.figure(figsize=(14,5))
sns.pairplot(df,hue='target',palette='inferno');
sns.heatmap(df.corr(),annot=True);
nltk.download('punkt')
