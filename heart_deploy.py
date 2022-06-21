# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:42:06 2022

@author: _K
"""

import streamlit as st
import numpy as np
import pickle
import os


MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')
with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)


#%%To test
# X_new = [65,3,142,220,158,2.3,1] #1

# ['age','cp','trtbps','chol','thalachh','oldpeak','thall']
# outcome = model.predict(np.expand_dims(np.array(X_new),axis=0))
# print(outcome)
#%%

# Form creation using streamlit
st.header('Heart Attack Prediction App :pushpin:')

#background
st.markdown(f""" <style>.stApp {{
             background:url('https://www.teahub.io/photos/full/19-197024_\
                            backgrounds-medical-hd-best-free-medicine-\
                                wallpapers-medicine.jpg');
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True)
    
    
with st.form("patients info"):
    
    st.write("According to World Health Organisation (WHO), every year around\
             17.9 million deaths are due to cardiovascular diseases (CVDs)\
             predisposing CVD becoming the leading cause of death\
             globally. CVDs are a group of disorders of the heart and\
             blood vessels, if left untreated it may cause heart\
             attack. Heart attack occurs due to the presence of obstruction of\
             blood flow into the heart.The presence of blockage may be due to \
             the accumulation of fat, cholesterol, and other substances.\
             Despite treatment has improved over the years and most CVD’s\
             pathophysiology have been elucidated, heart attack can still be\
             fatal.")
             
    st.subheader("This app is to predict the chance of getting heart attack.")
    age = st.number_input('age')
    cp = int(st.radio("Select your chest pain type.\
                      (0 : typical angina, \
                      1 : atypical angina, \
                      2 : non-anginal pain, \
                      3 : asymptomatic)",(0,1,2,3)))
    trtbps = st.number_input('trtbps')
    chol = st.number_input('chol')
    thalachh = st.number_input('thalachh')
    oldpeak = st.number_input('oldpeak')
    thall = int(st.radio("Select your thall rate.\
                         (0 : null,\
                         1 : fixed defect,\
                         2 : normal,\
                         3 : reversable defect)",(0,1,2,3)))

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
      X_new = [age,cp,trtbps,chol,thalachh,oldpeak,thall]

      outcome = model.predict(np.expand_dims(np.array(X_new),axis=0))
      outcome_dict = {0:'Less chance of getting heart attack',
                      1:'Higher chance of getting heart attack'}

      st.write(outcome_dict[outcome[0]])

      if outcome == 1:
          st.error('Please take care of your health :sweat:')
          st.write('You have more chance of getting heart attack')

      else:
          st.balloons()
          st.write('Congratulations, it seems you are healthy :thumbsup:')

# For the test data, there is 8 out of 10 the test correctly,
# therefore our model have around 80% accuracy
