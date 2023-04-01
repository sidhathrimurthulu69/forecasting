# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:26:23 2023

@author: sidda
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64
from datetime import datetime, timedelta
st.title('Time Series Forecasting')

"""
This data app uses Facebook's open-source Prophet library to automatically generate future forecast values from an imported dataset.
You'll be able to import your data from a CSV file, visualize trends and features, analyze forecast performance, and finally download the created forecast 😵 
**In beta mode**
Created by : sidha 
Code available here: github/sidhathrimurthulu@69
"""

"""
### Step 1: Import Data

"""
df = pd.read_csv("C:/Users/sidda/Desktop/project 97/TMT-JUNE-2020_2 (1) - Copy.csv") 
df=df.drop(['Time', 'Location', 'Sales in Rs/T',
       'Price/ kg', 'Climate', 'Customer ID', 'Diameter', 'Length', 'Grade',
       'Current stock', 'Re-order', 'Lead time', 'Production time',
       'Units Produced ', 'Production cost'], axis=1)

df.columns = ['ds','y']

df['y']= pd.DataFrame(df['y'].interpolate(method='linear'))
df.to_csv('tonne.csv', 
                  index = None)






df = st.file_uploader(r'C:\Users\sidda\tonne.csv')

st.info(
            f"""
                👆 Upload a .csv file first. Sample to try: [tonne.csv](https://raw.githubusercontent.com/zachrenwick/streamlit_forecasting_app/master/example_data/example_wp_log_peyton_manning.csv)
                """
        )

if df is not None:
    data = pd.read_csv(df)
    data.columns = ['ds','y']
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce') 
    data['y']= pd.DataFrame(data['y'].interpolate(method='linear'))
    st.write(data)
    
    max_date = data['ds'].max()
    st.write(max_date)

"""
### Step 2: Select Forecast Horizon
Keep in mind that forecasts become less accurate with larger forecast horizons.
"""

periods_input = st.number_input('how many days u will forecast',
min_value = 1, max_value = 365)

if df is not None:
    m = Prophet()
    m.fit(data)

"""
### Step 3: Visualize Forecast Data
The below visual shows future predicted values. "yhat" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
"""
if df is not None:
    future = m.make_future_dataframe(periods=periods_input)
    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)
    
    """
    The next visual shows the actual (black dots) and predicted (blue line) values over time.
    """
    fig1 = m.plot(forecast)
    datenow = datetime(2023, 2, 28)
    dateend = datenow + timedelta(days=365)
    datestart = datenow
    plt.xlim([datestart, dateend])
    st.write(fig1)

    """
    The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
    """
    fig2 = m.plot_components(forecast)
    st.write(fig2)
   
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)