import joblib

def test(Message_1, Message_2, Message_3, Message_4, Message_5, Message_6, Message_7, Message_8, Message_9, Message_10, Message_11, Message_12, Message_13, Message_14, Message_15, Message_16, Message_17, Message_18, Message_19, Message_20, Message_21, Message_22, Message_23, Message_24, Message_25):
  model = joblib.load('RandomForestBreastCancer.pkl')
  prediction = model.predict([[Message_1, Message_2, Message_3, Message_4, Message_5, Message_6, Message_7, Message_8, Message_9, Message_10, Message_11, Message_12, Message_13, Message_14, Message_15, Message_16, Message_17, Message_18, Message_19, Message_20, Message_21, Message_22, Message_23, Message_24, Message_25]])
  return prediction

import streamlit as st
import pandas as pd
from util import Util
import time, random

st.set_page_config(
        page_title="Breast Cancer Prediction",
)

util = Util(file_path='data.csv')
st.header("Breast Cancer Detection & Prediction Application")

# Create a text element and let the reader know the data is loading.
data_load_state = st.info('Loading data...')
# Load rows of data 
patient_data = util.get_data()
X_train, X_test, y_train, y_test = util.split_data(patient_data)

#train model
data_load_state.info("Training the model..")
model = util.build_model(X_train, y_train)

# Notify the reader that the data was successfully loaded.
data_load_state.info('Application is ready for predictions.')


## FORM for Prediction
st.subheader("Fill in the features computed from the digitized image of fine needle aspirate (FNA) of a digital mass")

with st.sidebar:
    st.subheader("Try other values")

    randomize = st.button("Generate test patient values")

    if randomize:
        data_list = util.sample_data(patient_data)
        idx = random.randint(0, len(data_list))
        input_vals = data_list[idx]
        st.json(input_vals)

util.form_functions(model)

st.markdown(util.page_footer(),unsafe_allow_html=True)