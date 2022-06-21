
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

# Heart-Attack-Prediction :bangbang:
Trained and predicting the chance of getting heart attack.
This project aims to implement a robust machine learning model that can efficiently predict the risk of a heart attack of a human in the future, based on the conditions or symptoms that he/she posses.

## Description
A heart attack (myocardial infarction or MI) is a serious medical emergency in which the supply of blood to the heart is suddenly blocked, usually by a blood clot. A heart attack is a medical emergency. Thus, a precaution step must be taken to avoid any unwanted event. In this model our objective is to predict the possibility that a person might have a chance of getting a heart attack.

## Aproach
- Gathering the data:
Data preparation is the primary step for any machine learning problem. By using a dataset from Kaggle for this problem. This dataset consists of one CSV files that will be use for training. There is a total of 14 columns in the dataset out of which 13 columns represent the features and the last column is the output of the heart attack chances.

- Data Inspection/ Visualization:
Here is where the data visualization is carried on. The mean, median, standard deviation, outliers, graph and etc are all presented in this section for the better understanding baout our dataset.

- Cleaning the Data: 
Cleaning is the most important step in a machine learning project. The quality of our data determines the quality of our machine learning model. So it is always necessary to clean the data before feeding it to the model for training. In our dataset all the columns are numerical, the target column and thse same goes to our target column.

- Feature Selection:
Feature Selection is the method of reducing the input variable to the model by using only relevant data and getting rid of noise in data. It is the process of automatically choosing relevant features for the machine learning model based on the type of problem that need to solve. Since the target column in this project is the categorical type, Cramer's V is been used to solve the categorical vs categorical data type while logistic regression is for continuous vs categorical.

- Data Preprocessing:
In this section, the technique of preparing the raw data to make it suitable for a building and training Machine Learning models is been done. The pipeline also is been introduced to find the best scalling and model training for the project.

- Tuning the model:
Tuning is the process of maximizing a model’s performance without overfitting or creating too high of a variance. In machine learning, this is accomplished by selecting appropriate “hyperparameters”. Here, GridSearchCV is been used where it helps to loop through predefined hyperparameters and fit the estimator (model) on the training set. 

- Model Analysis/Evaluation:
Model evaluation is the process of analysing a machine learning model's performance, as well as its strengths and limitations, using various evaluation criteria. Model evaluation is critical for determining a model's efficacy during the early stages of research, as well as for model monitoring. Hence, confusion matrix,classification report and accuracy is been presented as a part of the model evaluation.

## Results :pencil:
The best pipeline to be use in this model is Logistic Regression and MinMax Scaler.

![confusionmatrix](static/confusion_matrix.png)

![accuracy](static/classification_report.PNG)


## Discussion :books:
From this model, around 78% accuracy can be achieved during training. Although the accuracy only around 78%, it can be improved by adding more data to it.

## Application :video_game:
From the model trained before, an apps is been created to predict the possibility of he/she to have heart attack by using streamlit as a medium.

![app1](static/heart_app1.PNG)

![app2](static/heart_app2.PNG)


## Credits :open_file_folder:
This project is made possible by the data provided from kaggle:
[kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)


