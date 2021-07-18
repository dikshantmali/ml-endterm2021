
# -*- coding: utf-8 -*-
"""

@author: Dikshant Mali
"""

import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('randomforest.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('Classification Dataset3.csv')
# Extracting independent variable:
dataset['sd'] = dataset['sd'].fillna(dataset['sd'].mean())
dataset['median'] = dataset['median'].fillna(dataset['median'].mean())
dataset['IQR'] = dataset['IQR'].fillna(dataset['IQR'].mean())
dataset['skew'] = dataset['skew'].fillna(dataset['skew'].mean())
dataset['kurt'] = dataset['kurt'].fillna(dataset['kurt'].mean())
dataset['mode'] = dataset['mode'].fillna(dataset['mode'].mean())
dataset['centroid'] = dataset['centroid'].fillna(dataset['centroid'].mean())






# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_data = LabelEncoder()
dataset.label = labelencoder_data.fit_transform(dataset.label)


X = dataset.iloc[:, 0 : 9]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


def predict_note_authentication(meanfreq,sd,median,iqr,skew,kurt,mode,centroid,dfrange):
  output= model.predict(sc.transform([[meanfreq,sd,median,iqr,skew,kurt,mode,centroid,dfrange]]))
  print("The ", output)
  if output==[0]:
    prediction="person is Male"
  else:
    prediction="person is Female"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:#3EB489" >
   <div class="clearfix">           
   <div class="col-lg-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">RTU EndTerm Practical(Dikshant Mali - PIET18CS049)</p></center> 
   <center><p style="font-size:30px;color:black;margin-top:10px;">Department of Computer Engineering PIET,Jaipur</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;"Machine Learning Lab Endterm </p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Random Forest Classification")
    meanfreq = st.number_input("Enter The meanfreq value",0,1)
    sd = st.number_input("Enter The sd value",0,1)
    median = st.number_input("Enter The median value",0,1)
    iqr = st.number_input("Enter The iqr value",0,1)
    skew = st.number_input("Enter The skew value",0,15)
    kurt = st.number_input('Enter the kurt value ',0,600)
    mode = st.number_input('Enter the mode value',0,1)
    centroid = st.number_input("Enter the centroid value",0,1)
    dfrange = st.number_input("Enter the dfrange value",0,5)
    if st.button("Predict"):
      result=predict_note_authentication(meanfreq,sd,median,iqr,skew,kurt,mode,centroid,dfrange)
      st.success('{} '.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Dikshant Mali")
      st.subheader("Student , Poornima Institute Of Engineering And Technology")

if __name__=='__main__':
  main()
