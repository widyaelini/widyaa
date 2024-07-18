import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
from sklearn.ensemble import RandomForestClassifier
import joblib

class Util:
    def __init__(self, file_path = 'data.csv'):
        self.features = [
            'concave points_worst', 'perimeter_worst', 'concave points_mean', 'radius_worst', 'perimeter_mean', 'area_worst',
            'radius_mean', 'area_mean', 'concavity_mean', 'concavity_worst']
 
        self.target_col = 'diagnosis'
        self.file_path = file_path
        
    
    def preprocess(self, df):
        # Rename columns for diagnosis classes
        df.rename(columns={'Dataset': 'diagnosis'}, inplace=True)
        df[self.target_col] = (df[self.target_col] == 'M').astype(int)
        return df       
    

    def get_data(self):
        
        df = pd.read_csv(self.file_path)
       
        # preprocess data
        df = self.preprocess(df)
        return df
 
    def split_data(self, df):
        X = df[self.features]
        y = df[self.target_col]

        # Scaling the feature columns
        # scaler = StandardScaler().fit(X)
        # X_scaled = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler() #create an instance of standard scaler
        scaler.fit(X_train) # fit it to the training data

        scaler.transform(X_train) #transform training data
        scaler.transform(X_test) #transform validation data

        return X_train, X_test, y_train, y_test
 
    def build_model(self, X, y):
        model = LogisticRegression()
        
        print("Fitting the model")
        model.fit(X, y)
                
        return model
    
    def compute_accuracy(self, model, X_test, y_test):
        # model = joblib.load('RandomForestBreastCancerUpdated.pkl')
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)*100
    
    def predict(model, X):
        # model = joblib.load('RandomForestBreastCancerUpdated.pkl')
        prediction = model.predict(X)
        return prediction

    def input_data_fields(self, overwrite_vals=None):

        default_vals = {'concave points_worst': 0.175, 
                        'perimeter_worst': 65.50, 
                        'concave points_mean': 0.04375, 
                        'radius_worst': 10.310, 
                        'perimeter_mean': 58.79, 
                        'area_worst': 324.7,
                        'radius_mean': 9.029, 
                        'area_mean': 250.5, 
                        'concavity_mean': 0.31300, 
                        'concavity_worst': 1.25200}

        col1, col2 = st.columns(2)
        concave_points_worst = col1.number_input("Concave Points Worst", 
                            min_value=None,
                            format= "%.2f",
                            value=default_vals['concave points_worst'],)
        
        perimeter_worst = col2.number_input("Perimeter Worst", 
                            min_value=None,
                     
                            format= "%.2f",
                            value=default_vals['perimeter_worst'],)

        concave_points_mean = col1.number_input("Concave Points Mean", 
                            min_value=None,
                            
                            format= "%.2f",
                            value=default_vals['concave points_mean'],)
        
        radius_worst = col2.number_input("Radius Worst", 
                            min_value=None,
                            
                            format= "%.2f",
                            value=default_vals['radius_worst'],)
        
        perimeter_mean = col1.number_input("Perimeter Mean", 
                            min_value=None,
                           format= "%.2f",
                            value=default_vals['perimeter_mean'],)
        
        area_worst = col2.number_input("Area Worst", 
                            min_value=None,
                            format= "%.2f",
                            value=default_vals['area_worst'],)
        
        radius_mean = col1.number_input("Radius Mean", 
                            min_value=None,
                           format= "%.2f",
                            value=default_vals['radius_mean'],)
        
        area_mean = col2.number_input("Area Mean", 
                            min_value=None,
                           format= "%.2f",
                            value=default_vals['area_mean'],)
        
        concavity_mean = col1.number_input("Concavity Mean", 
                            min_value=None,
                           format= "%.2f",
                            value=default_vals['concavity_mean'],)
        
        concavity_worst = col2.number_input("Concavity Worst", 
                            min_value=None,
                          format= "%.2f",
                            value=default_vals['concavity_worst'],)

        return {'concave points_worst': concave_points_worst, 
                        'perimeter_worst': perimeter_worst, 
                        'concave points_mean': concave_points_mean, 
                        'radius_worst': radius_worst, 
                        'perimeter_mean': perimeter_mean, 
                        'area_worst': area_worst,
                        'radius_mean': radius_mean, 
                        'area_mean': area_mean, 
                        'concavity_mean': concavity_mean, 
                        'concavity_worst': concavity_worst}
        
    def form_functions(self, model):
        with st.form("my_form"): 
            get_values = self.input_data_fields()
            
            submitted = st.form_submit_button("Submit", type="primary")
            if submitted:
                data_values = pd.DataFrame([get_values])
                
                # Get predictions
                with st.spinner('Making prediction...'):
                    time.sleep(3)
                # model = joblib.load('RandomForestBreastCancerUpdated.pkl')
                prediction = model.predict(data_values)
                print("Prediction: ", prediction[0])

                prediction_msg = "The supplied values suggest that the patient does not have breast cancer." if prediction == 0 else "The supplied values suggest that the patient has breast cancer. It is suggested to provide critical emphasis on diagnosing further symptoms of the patient. "
        
                st.subheader("Diagnosis:")

                if prediction == 0:
                    print("Success")
                    st.success(prediction_msg)

                else:
                    st.error(prediction_msg)
    
    def sample_data(self, df):

        test_data = df[['concave points_worst', 'perimeter_worst', 'concave points_mean', 'radius_worst', 'perimeter_mean', 'area_worst', 'radius_mean', 'area_mean', 'concavity_mean', 'concavity_worst']].to_dict(orient='records')

        return test_data

    def page_footer(self):
        # footer="""<style>
        #         a:link , a:visited{
        #         color: blue;
        #         background-color: transparent;
        #         text-decoration: underline;
        #         }

        #         a:hover,  a:active {
        #         color: red;
        #         background-color: transparent;
        #         text-decoration: underline;
        #         }

        #         .footer {
        #         position: fixed;
        #         left: 0;
        #         bottom: 0;
        #         width: 100%;
        #         background-color: white;
        #         color: black;
        #         text-align: center;
        #         }
        #         </style><div class="footer"><p>Developed by <a style='display: block; text-align: center;' target="_blank">Gitesh Kambli, Amit Maity, Chirag Maniyath, Rishi More for Honors in AIML Mini Project</a></p></div>
        #         """

        footer = """
        <style>
            a:link , a:visited{
            color: {text_color};
            background-color: transparent;
            text-decoration: underline;
            }

            a:hover,  a:active {
            color: {text_color};
            background-color: transparent;
            text-decoration: underline;
            }

            .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: {background_color};
            color: {text_color};
            text-align: center;
            }

            @media (prefers-color-scheme: dark) {
                .footer {
                    background-color: rgb(14, 17, 23);
                    color: white;
                }
            }
        </style>
        <div class="footer"><p>Developed by <a style='display: block; text-align: center;' target="_blank">Rishi More, Gitesh Kambli, Amit Maity, Chirag Maniyath</a></p></div>
        """
        return footer
        
