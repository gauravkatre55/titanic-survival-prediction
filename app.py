import streamlit as st
import pandas as pd
import joblib

## load trained model
with open('model.pkl','rb') as f:
    model=joblib.load(f)
    
# frontend page

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
)

st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival:")

#user input fields:
Pclass=st.selectbox(
    "Passenger Class",
    options=[1, 2, 3]
)

Sex=st.selectbox(
    'Gender',
    options=['male','female']
)

Age=st.number_input(
    "Age",
    min_value=5.0,
    max_value=90.0
)

SibSp=st.number_input(
    "Number of Siblings",
    min_value=0.0,
    max_value=5.0
)
Parch=st.number_input(
    "Number of Parents/Children",
    min_value=0.0,
    max_value=5.0
)

Fare=st.number_input(
    'Ticket Fare',
    min_value=0.0,
    max_value=40.0
)

Embarked=st.selectbox(
    'Embarked',
    options=['C','Q','S']
)


## Prediction button:

if st.button("Predict"):
    st.write("Predicting survival...")
    
    # create input data for prediction
    input_data=pd.DataFrame([{
        'Pclass':Pclass,
        'Sex':Sex,
        'Age':Age,
        'SibSp':SibSp,
        'Parch':Parch,
        'Fare':Fare,
        'Embarked':Embarked
        }])

    # make prediction
    prediction=model.predict(input_data)

    # display prediction result
    if prediction[0]==1:
        st.success("Survived")
    else:
         st.error("Not Survived")
         
        