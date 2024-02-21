import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import base64
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from yahoofinancials import YahooFinancials
from keras.models import load_model
from datetime import date

import base64
st.markdown("<h1 style='text-align: center; color: #007FFF; '>Stock Price Prediction</h1>", unsafe_allow_html=True)
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


# st.title("Stock Price Prediction ")

selectbox = st.sidebar.radio(
    ' ',
    ["About Us", "Predict", "Contact Us"]
)

def about_us():
    # st.markdown("<h3 style='text-align: center; color: blue;'>About US </h3>", unsafe_allow_html=True)
    components.html(
        """

        <script src="https://kit.fontawesome.com/e40efcfb88.js" crossorigin="anonymous"></script>

        <div class="section" style="color:White; justify-content:center; font-family: "Lucida Console", "Courier New", monospace; font-size:35px;">
          <div class="container">
            <div class="content-section">
              <div class="title">
                <h1 >About Us</h1>
              </div>
                <div class="content" style="font-size:15px;">
                    <h3>
                        Unleash Your Career Potential with Stock Price Prediction
                    </h3>
                    <p>
                       Leveraging our cutting-edge technology and advanced data analysis, 
                       we provide predictive analytics, 
                       keeping you at the forefront of the industry. 
                       Stay one step ahead, make informed decisions,
                         and thrive in the dynamic world of finance.
                    </p>

                </div>
                <div class="social">
                    <a href="https://www.facebook.com/sai.nilapwar.5" target="_blank" rel="noopener noreferrer"><i class="fab fa-facebook-f" style="text-decoration: none; color: white; padding:10px; margin-left:10px; height:15px;  background-color:blue; border-radius:5px;"></i></a>
                    <a href="https://twitter.com/?lang=en-in" target="_blank" rel="noopener noreferrer"><i class="fab fa-twitter"  style="text-decoration: none; color: white; padding:10px; margin-left:10px; height:15px;  background-color:blue; border-radius:5px;"></i></a>
                    <a href="https://instagram.com/rocking__rohan?igshid=MzRIODBiNWFIZA=:" target="_blank" rel="noopener noreferrer"><i class="fab fa-instagram" style="text-decoration: none; color: #e44b8d; padding:10px; margin-left:10px; height:15px;  background-color:white; border-radius:5px;"></i></a>
                </div>
              </div>
              <div class="image-section>
                 <img src="pxfuel.jpg">
              </div>
            </div>
            
          </div>
        </div>
        """,
        height=800
    )

def predict():
    
    last_date = date.today()
    st.markdown("<h3 style='color: white, font-size:5px;'> Enter Company Name </h3>",unsafe_allow_html=True)
    company_name=st.text_input(" " )
    if(company_name):
        ds = yf.download(company_name, start='2020-01-01', end=last_date, progress=False,)

        #Describing Data
        st.subheader('Data From 2020 - 2023')
        st.write(ds.describe())

        #visualization
        st.subheader('Closing Price vs Time Chart')
        fig = plt.figure(figsize =(12,6))
        ax=plt.axes()
        ax.set_facecolor('black')
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.plot (ds.Close)
        st.pyplot(fig)


        st.subheader('Closing Price vs Time Chart for 100MA')
        
        ma100= ds.Close.rolling(100).mean()
        fig = plt.figure(figsize =(12,6))
        ax=plt.axes()
        ax.set_facecolor('black')
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.plot(ma100)
        plt.plot (ds.Close)
        
        st.pyplot(fig)


        st.subheader('Closing Price vs Time Chart for 100MA & 200MA')
        ma100= ds.Close.rolling(100).mean()
        ma200= ds.Close.rolling(200).mean()
        fig = plt.figure(figsize =(12,6))
        ax=plt.axes()
        ax.set_facecolor('black')
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.plot(ma100,'r')
        plt.plot(ma200, 'g')
        plt.plot (ds.Close, 'b')
        
        st.pyplot(fig)

        # Splitimg Data Into Traning And Testing
        data_training = pd.DataFrame(ds['Close'][0:int(len(ds)*0.70)])
        data_testing = pd.DataFrame(ds['Close'][int(len(ds)*0.70):])

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler (feature_range=(0,1))

        data_training_array = scaler.fit_transform(data_training)


        #Load Model
        model = load_model('keras_model.h5')

        #Testing Part
        past_100_days = data_training.tail(100)
        # final_ds = past_100_days.append(data_testing, ignore_index=True)



        past_100 = past_100_days.values.tolist()
        final_ds_1= data_testing.values.tolist()
        for i in final_ds_1:
          past_100.append(i)

        final_ds=pd.DataFrame(past_100)

        final_ds.columns = ['close']




        input_data = scaler.fit_transform(final_ds)


        x_test = []
        y_test = []
        for i in range(100,input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i,0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)
        scaler = scaler.scale_
        scale_factor = 1/scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        #Final Graph
        st.subheader('Prediction vs Original')
        fig2= plt.figure(figsize=(12,6))
        ax=plt.axes()
        ax.set_facecolor('black')
        plt.plot(y_test,'b',label = 'Original price')
        plt.plot(y_predicted, 'r', label = 'Predicted price')
        plt.ylabel('Time')
        plt.xlabel('Price')
        plt.legend()
        st.pyplot(fig2)

def contact_us():
    components.html(
        """
        <style>

              input[type=text], select, textarea {
              width: 100%; /* Full width */
              padding: 12px; /* Some padding */ 
              border: 1px solid #ccc; /* Gray border */
              border-radius: 4px; /* Rounded borders */
              box-sizing: border-box; /* Make sure that padding and width stays in place */
              margin-top: 6px; /* Add a top margin */
              margin-bottom: 16px; /* Bottom margin */
              resize: vertical /* Allow the user to vertically resize the textarea (not horizontally) */
            }

            /* Style the submit button with a specific background color etc */
            input[type=submit] {
              background-color: blue;
              color: white;
              padding: 12px 20px;
              border: none;
              border-radius: 4px;
              cursor: pointer;
            }

            /* When moving the mouse over the submit button, add a darker green color */
            input[type=submit]:hover {
              background-color: #45a049;
            }

            /* Add a background color and some padding around the form */
            .container {
              border-radius: 5px;
              background-color: #f2f2f2;
              padding: 20px;
            }
        </style>

        <div class="container">
            <form action="action_page.php">

              <label for="fname">First Name</label>
              <input type="text" id="fname" name="firstname" placeholder="Your name..">

              <label for="lname">Last Name</label>
              <input type="text" id="lname" name="lastname" placeholder="Your last name..">

              <label for="country">Country </label>
              <select id="country" name="country" placeholder="select your country">
                <option value="India">India</option>
                <option value="australia">Australia</option>
                <option value="canada">Canada</option>
                <option value="usa">USA</option>
              </select>

              <label for="subject">Subject</label>
              <textarea id="subject" name="subject" placeholder="Write something.." style="height:200px"></textarea>

              <input type="submit" value="Submit">

            </form>
        </div>
        """,
        height=700
    )
if(  selectbox == 'About Us'):
    about_us()
elif( selectbox == 'Predict'):
    predict()
elif(selectbox == 'Contact Us'):
    contact_us()