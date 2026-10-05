import pandas as pd
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('Data/Titanic-Dataset.csv')

## Features:
features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

x=df[features]
y=df['Survived']

# numerical features
numerical=['Age','SibSp','Parch','Fare' ]
# categorical features
categorical=['Sex','Embarked','Pclass']

## Preprocessing

numeric_transformer=Pipeline(steps=[('imputer',SimpleImputer(strategy='median'))])
                             
categorical_transformer= Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
    ])

preprocessor=ColumnTransformer(
    transformers=[
        ('num',numeric_transformer,numerical),
        ('cat',categorical_transformer,categorical)
    ])

# create Model:
model=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Complete Ml Pipline:
Pipeline=Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('model',model)
    ]
)

# train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

Pipeline.fit(x_train,y_train)
y_pred=Pipeline.predict(x_test)

# Accuracy
pipeline_score=Pipeline.score(x_test,y_test)
print('score:-',pipeline_score)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print('Accuracy:-',accuracy)

# model pickle :- save the model in pickel file
with open('model.pkl','wb') as f:
    joblib.dump(Pipeline,f)
    
print('model saved')
