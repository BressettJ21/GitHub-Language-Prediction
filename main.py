import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection
import sklearn.preprocessing as preproc
from sklearn.feature_extraction import text

def clean_text(text, remove_stopwords=True):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''

    # Convert words to lower case
    text = text.lower()

    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    # remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        stops.add('r')
        stops.add('python')
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    # Tokenize each word
    text = nltk.WordPunctTokenizer().tokenize(text)

    return text

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('//Users/shravankaul/Desktop/UWM/Stat 601/Project 2/Project2Data/dat.csv', header=0,encoding= 'unicode_escape')
py_lib=pd.read_csv('//Users/shravankaul/Desktop/test/Py_lib.csv')
r_lib=pd.read_csv('//Users/shravankaul/Desktop/test/r_packages.csv',header=None)


data = data.dropna()
data = data[data['MajLang']!='Jupyter Notebook']
data = data[data['MajLang']!='Julia']
data['MajLang'] = np.where(data['MajLang'] == 'R', 1, data['MajLang'])
data['MajLang'] = np.where(data['MajLang'] != 1, 0, data['MajLang'])
print(data['MajLang'].unique())
sns.countplot(x='MajLang', data=data, palette='hls')
plt.show()
count_not_R = len(data[data['MajLang'] == 0])
count_R = len(data[data['MajLang'] == 1])
pct_of_not_R = count_not_R / (count_not_R + count_R)
print("percentage of Python", pct_of_not_R * 100)
pct_of_R = count_R / (count_not_R + count_R)
print("percentage of R", pct_of_R * 100)

tags_c=data['tags']
data['tags']=data['tags'].str.lower()
l1=r_lib[0]

l1=list(l1.str.lower())
l2=list(py_lib['Python'].str.lower())

#Cleaning tags
data['tags'] = data['tags'].str.replace(',','')
data['tags'] = data['tags'].str.replace('[','')
data['tags'] = data['tags'].str.replace(']','')
data['tags'] = data['tags'].str.replace("'","")
data['tags']=data['tags'].str.split()
data['tags'] =[['package' if word in l1 else word for word in i] for i in data['tags']]
data['tags'] =[['lib' if word in l2 else word for word in i] for i in data['tags']]
data['Descr'] = data['Descr'].str.lower()
data['Descr'] = data['Descr'].str.replace("data science","")
data['Descr'] = data['Descr'].str.replace("data-science","")

data['Descr']=data['Descr'].str.encode('ascii', 'ignore').str.decode('ascii')

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
df=pd.DataFrame(data=data)

df['Lang-Descr']= None
Lang_Descr=df['Lang-Descr']
df.drop(labels=['Lang-Descr'], axis=1,inplace = True)
df.insert(6, 'Lang-Descr', Lang_Descr)
df=df.reset_index()

import pycld2 as cld2
for i in range(len(df['Descr'])) :
    _, _, _, detected_language = cld2.detect(df['Descr'][i],  returnVectors=True)
    if len(detected_language)==0:
        df['Lang-Descr'][i]=0
    else:
        df['Lang-Descr'][i]= detected_language[0][2]

df=df[df['Lang-Descr'] != 0]

mlb = MultiLabelBinarizer(sparse_output=True)

temp = pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(df.pop('tags')),
               index=df.index,
                columns=mlb.classes_)


temp=temp.drop(temp.columns[temp.apply(lambda col: col.sum() < 5 )],axis=1)
df = df.join(temp)


df['tags_c']=tags_c


df.drop(labels=['tags_c'], axis=1,inplace = True)
df.insert(4, 'tags_c', tags_c)
df=df[df['Lang-Descr'] == 'ENGLISH']
df['Text_Cleaned'] = list(map(clean_text, df['Descr']))


def lemmatized_words(text):
    lemm = nltk.stem.WordNetLemmatizer()
    df['lemmatized_text'] = list(map(lambda word:
                                     list(map(lemm.lemmatize, word)),
                                     df.Text_Cleaned))


lemmatized_words(df.Text_Cleaned)

Text_Cleaned=df['Text_Cleaned']
df.drop(labels=['Text_Cleaned'], axis=1,inplace = True)
df.insert(7, 'Text_Cleaned', Text_Cleaned)

#Term Frequency (BOW)
dfR=df[df['MajLang'] == 1]
dfNR=df[df['MajLang'] == 0]
dfR.reset_index()
dfNR.reset_index()
bow_converter = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
x = bow_converter.fit_transform(dfR['lemmatized_text'])
words = bow_converter.get_feature_names()

dftfR=pd.DataFrame({'count' : [x[:,i].sum() for i in range(len(words))]})
dftfR['words']=words

dftfR=dftfR.sort_values('count',ascending=False)



x = bow_converter.fit_transform(dfNR['lemmatized_text'])
words = bow_converter.get_feature_names()

dftfNR=pd.DataFrame({'count' : [x[:,i].sum() for i in range(len(words))]})
dftfNR['words']=words

dftfNR=dftfNR.sort_values('count',ascending=False)


#df.to_csv('//Users/shravankaul/Desktop/test/sk3.csv')
#top 50 words R & Pythom
dftfR_list=dftfR['words'][0:50]

dftfNR_list=dftfNR['words'][0:50]

dftf_list = dftfR_list
dftf_list = dftf_list.append(dftfNR_list)

df.reset_index()

df['lemmatized_text'] = [[word+'_d' for word in i if word in list(dftf_list)] for i in df['lemmatized_text']]


# MLB for cleaned tokenized Description
df = df.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(df.pop('lemmatized_text')),
                index=df.index,
                columns=mlb.classes_
                )
)

df = df.reset_index()
# Creating final data for modelling
non_features=['level_0', 'index', 'ï»¿S.no', 'URLCODE', 'Name', 'tags_c', 'Descr', 'Text_Cleaned', 'Lang-Descr', 'License', 'LastUpdated']
non_features_t=['level_0', 'index', 'ï»¿S.no', 'URLCODE', 'Name', 'tags_c', 'Descr', 'Text_Cleaned', 'Lang-Descr', 'License', 'LastUpdated','Stars','ai', 'algorithms', 'analysis', 'analytics', 'automl', 'aws', 'bioinformatics', 'book', 'bookdown', 'color', 'color-science', 'color-space', 'color-spaces', 'colorspaces', 'colour', 'colour-science', 'colour-space', 'colour-spaces', 'colourspace', 'colourspaces', 'computer-science', 'cookiecutter', 'cookiecutter-data-science', 'cookiecutter-template', 'data', 'data-analysis', 'data-science', 'data-science-bowl-2018', 'data-visualization', 'database', 'datascience', 'dataset', 'datasets', 'deep-learning', 'django', 'dna', 'docker', 'education', 'genomics', 'hacktoberfest', 'hyperparameter-tuning', 'instance-segmentation', 'iris', 'kaggle', 'keras', 'lib', 'machine-learning', 'machinelearning', 'medical-imaging', 'ml', 'mlops', 'netcdf', 'ngs', 'nlp', 'open-source', 'package', 'pipeline', 'productivity', 'python', 'python3', 'r', 'r-package', 'reproducible-research', 'rmarkdown', 'rstats', 'science', 'scikit-learn', 'segmentation', 'sequencing', 'spark', 'stacking', 'teaching', 'tensorflow', 'unet', 'visualization', 'workflow', 'xgboost']
non_features_d=['level_0', 'index', 'ï»¿S.no', 'URLCODE', 'Name', 'tags_c', 'Descr', 'Text_Cleaned', 'Lang-Descr', 'License', 'LastUpdated','Stars','algorithm_d', 'analysis_d', 'analytics_d', 'application_d', 'assembly_d', 'assignment_d', 'big_d', 'bowl_d', 'build_d', 'building_d', 'capstone_d', 'class_d', 'classification_d', 'code_d', 'collection_d', 'competition_d', 'course_d', 'coursera_d', 'data_d', 'deep_d', 'documentation_d', 'dojo_d', 'exercise_d', 'file_d', 'framework_d', 'function_d', 'ga_d', 'getting_d', 'git_d', 'github_d', 'high_d', 'hopkins_d', 'includes_d', 'intro_d', 'introduction_d', 'jeffrey_d', 'john_d', 'kaggle_d', 'learn_d', 'learning_d', 'library_d', 'life_d', 'list_d', 'machine_d', 'material_d', 'model_d', 'package_d', 'pipeline_d', 'place_d', 'platform_d', 'prediction_d', 'programming_d', 'project_d', 'public_d', 'repo_d', 'repository_d', 'reproducible_d', 'resource_d', 'science_d', 'script_d', 'series_d', 'set_d', 'social_d', 'solution_d', 'source_d', 'specialization_d', 'stanton_d', 'statistical_d', 'template_d', 'tool_d', 'tutorial_d', 'university_d', 'used_d', 'using_d', 'video_d', 'visualization_d', 'web_d', 'work_d', 'youtube_d']
data_vars=df.columns.values.tolist()
print(data_vars)
tokeep=[i for i in data_vars if i not in non_features]
tokeep_t=[i for i in data_vars if i not in non_features_t]
tokeep_d=[i for i in data_vars if i not in non_features_d]
df_allm = df[tokeep]
df_desc = df[tokeep_t]
df_tags=df[tokeep_d]


#df.to_csv('//Users/shravankaul/Desktop/test/sk4.csv')

# All stars, tags and descriptions model
X = df_allm.loc[:, df_allm.columns != 'MajLang']
y=df_allm['MajLang']
y=y.astype('int')
X=X.astype('int')
import statsmodels.api as sm

from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
LogisticRegression(random_state=0)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on all features-test set: {:.4f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic -all ')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
# Descriptions model
X = df_desc.loc[:, df_desc.columns != 'MajLang']
y=df_desc['MajLang']
y=y.astype('int')
X=X.astype('int')
import statsmodels.api as sm

from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
LogisticRegression(random_state=0)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on descriptions test set: {:.4f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Descr')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

X = df_tags.loc[:, df_tags.columns != 'MajLang']
y=df_tags['MajLang']
y=y.astype('int')
X=X.astype('int')
import statsmodels.api as sm

from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
LogisticRegression(random_state=0)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on tags test set: {:.4f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - tags')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
