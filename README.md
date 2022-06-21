# Heart-Attack-Prediction
 Trained and predicting the chance of getting heart attack.


![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/yourGitHubName/yourRepo/yourApp/)

## Description
- The libraries used
```
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os
```

- Functions
```
#%% Functions
# Plot categorical data
def plot_cat(df,categorical_columns):
    
    '''
    This function is to generate plots for categorical columns
    Parameters
    ----------
    df : TYPE
      plotting the categorical data using countplot.
    Returns
    -------
    None.
    
    '''
    for cat in categorical_columns:
        plt.figure()
        sns.countplot(df[cat])
        plt.show()

# Plot continuous data
def plot_con(df,continuous_columns):
    
    '''
    This function is to generate plots for continuous columns
    Parameters
    ----------
    df : TYPE
      plotting the continuous data using distplot.
    Returns
    -------
    None.
  
    '''
    for con in continuous_columns:
        plt.figure()
        sns.distplot(df[con])
        plt.show()


# To compute Cramer's V

def cramers_corrected_stat(confussion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confussion_matrix)[0]
    n = confussion_matrix.sum()
    phi2 = chi2/n
    r,k = confussion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
```

- Statics
```
#%% Statics
HEART_DATASET_PATH = os.path.join(os.getcwd(),'heart.csv')
BEST_MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')
BEST_PIPE_PATH = os.path.join(os.getcwd(),'best_pipeline.pkl')
```


- Data Loading
```
#EDA
#%% Load Data
df = pd.read_csv(HEART_DATASET_PATH)
```

- Data Inspection/Visualization
```
#%% Data Inspection/Visulaization
df.head(10)
df.tail(10)
df.info()   #no NaN value in the dataset
temp = df.describe().T
df.columns

plt.figure(figsize=(12,10))
df.boxplot() 
#there is an noticeable outliers in the 'trtbps','chol' and 'thalachh' column
plt.show()

df.duplicated().sum()   #1 duplicated data have been detected
df[df.duplicated()]   #to extract the duplicated data

df.columns

categorical_column = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 
                      'thall', 'output']

continuous_column = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']

plot_cat(df,categorical_column)
plot_con(df,continuous_column)
```

- Data Cleaning
```
#%% Data Cleaning
df.isna().sum()

df = df.drop_duplicates()
df.duplicated().sum() #to ensure that all duplicated have been removed

#From df.info() or df.isna().sum() shows no NaN
#Therefore, no cleaning is needed except for duplicated
```
- Features Selection
```
#%% Features Selection
#(Categorical vs Categorical)
#Cramer's V

for cat in categorical_column:
    print(cat)
    confussion_mat = pd.crosstab(df[cat],df['output']).to_numpy()
    print(cramers_corrected_stat(confussion_mat))

# Note that 'cp' and 'thall' have the highest correlation(=>0.5) among the others
# which are 0.5090 and 0.5207 respectively

#(Continuous vs Categorical)
#Logistic Regression

for con in continuous_column:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[con],axis=-1),df['output'])
    print(con + '=' + str(lr.score(np.expand_dims(df[con],axis=-1),
                                   df['output'])))

# Here, 'age','trtbps','chol','thalachh', and 'oldpeak' have achieved more than
# 50% accuracy which are 62%, 58%, 53%, 70% and 69% respectively.
# Choose all of them as they achieved more than 50%
```
- Data Preprocessing
```
#%% Data Preprocessing
#From the feature selection above, these features is selected as 
#they have more than 0.5 correlation and 50% accuracy

X = df.loc[:,['age','cp','trtbps','chol','thalachh','oldpeak','thall']]
y = df.loc[:,['output']]

X_train ,X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=123)
```
- Pipeline
```
#%% Pipeline
#LR
step_mms_lr = Pipeline([('MMSScaler', MinMaxScaler()),
                        ('logisticRegression',LogisticRegression())])

step_ss_lr = Pipeline([('StandardScaler', StandardScaler()),
                       ('logisticRegression',LogisticRegression())])

#RF
step_mms_rf = Pipeline([('MMSScaler', MinMaxScaler()),
                        ('RandomForestClassifier',RandomForestClassifier())])

step_ss_rf = Pipeline([('StandardScaler', StandardScaler()),
                       ('RandomForestClassifier',RandomForestClassifier())])

#tree
step_mms_tree = Pipeline([('MMSScaler', MinMaxScaler()),
                          ('Decision Tree Classifier',
                           DecisionTreeClassifier())])

step_ss_tree = Pipeline([('StandardScaler', StandardScaler()),
                         ('Decision Tree Classifier',
                          DecisionTreeClassifier())])

#knn
step_mms_knn = Pipeline([('MMSScaler', MinMaxScaler()),
                         ('KNN',KNeighborsClassifier())])

step_ss_knn = Pipeline([('StandardScaler', StandardScaler()),
                        ('KNN',KNeighborsClassifier())])

#SVC 
step_mms_svc = Pipeline([('MinMaxScaler', MinMaxScaler()),
                       ('SVClassifier', SVC())])

step_ss_svc = Pipeline([('StandardScaler', StandardScaler()), 
                       ('SVClassifier', SVC())])

#%%
#Pipelines
pipelines = [step_mms_lr,step_ss_lr,
             step_mms_rf,step_ss_rf,
             step_mms_tree,step_ss_tree,
             step_mms_knn,step_ss_knn,
             step_mms_svc,step_ss_svc]

for pipe in pipelines:
    pipe.fit(X_train,y_train)

pipe_dict = {0:'MMS+Logistic',
             1:'SS+Logistic',
             2:'MMS+RForest',
             3:'SS+RForest',
             4:'MMS+DTree',
             5:'SS+DTree',
             6:'MMS+KNN',
             7:'SS+KNN',
             8:'MMS+SVC',
             9:'SS+SVC'}

best_accuracy = 0

# model evaluation
for i, model in enumerate(pipelines):
    print(model.score(X_test,y_test))
    if model.score(X_test,y_test) > best_accuracy:
        best_accuracy = model.score(X_test,y_test)
        best_pipeline = model
        best_scaler = pipe_dict[i]

print('The best scaling approach for this Dataset will be {} with \
accuracy of {}'.format(best_scaler,best_accuracy))
```
- Fine tune the model
```
#%%  This is to fine tune the model
#From the pipeline above, it is deduced that the best pipeline with MMS+Logistic
#have highest accuracy (0.7692) when tested against the test dataset

step_lr = [('MinMaxScaler', MinMaxScaler()),
           ('LogisticRegression', LogisticRegression())]

pipeline_lr = Pipeline(step_lr)

grid_param = [{'LogisticRegression':[LogisticRegression()],
               'LogisticRegression__dual':[True,False],
               'LogisticRegression__C':[0.1,1.0,10.0],
               'LogisticRegression__max_iter':[10,100,1000]}]

gridsearch = GridSearchCV(pipeline_lr,grid_param,cv=5,verbose=1,n_jobs=-1)
best_model = gridsearch.fit(X_train,y_train)

#%% Retrain your model the selected parameters
step_lr = [('MinMaxScaler', MinMaxScaler()),
            ('LogisticRegression', LogisticRegression())]

pipeline_lr = Pipeline(step_lr)
pipeline_lr.fit(X_train,y_train)

with open(BEST_MODEL_PATH,'wb') as file:
    pickle.dump(pipeline_lr,file)

#%%
print(best_model.score(X_test,y_test))
print(best_model.best_index_)
print(best_model.best_params_)

#Save the pickle file

with open(BEST_PIPE_PATH,'wb') as file:
    pickle.dump(best_model,file)
```

- Model Analysis
```#%% Model Analysis/evaluation

y_true = y_test
y_pred = best_model.predict(X_test)

cr = classification_report(y_true,y_pred)
cm = confusion_matrix(y_true,y_pred)
acc_score = accuracy_score(y_true,y_pred)

print(cr)
print(cm)
print(acc_score)

disp = ConfusionMatrixDisplay(confusion_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()
```

**You will find a template file and a simple example in the model and trainer folder that shows you how to try your first model simply.**

## To include url link

![markdown_badges]('[https://rahuldkjain.github.io/gh-profile-readme-generator/](https://github.com/Ileriayo/markdown-badges)')
[url_to_cheat_sheet](https://rahuldkjain.github.io/gh-profile-readme-generator/)


## Results
![model developed apps](static/heart_app.png)

## Discussion
From this model, around 78% accuracy can be achieved during training. where MinMaxScaler and LogisticRegression have been choosed through pipeline to be the best scaler and training model. Although the accuracy only around 78%, it can be improved by adding more data to it.

## Credits
