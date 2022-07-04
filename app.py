import streamlit as st
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import OneHotEncoder
# st.set_page_config(
#     page_title="Flight price predictor",
    
#     layout="wide",
#     initial_sidebar_state="expanded")




st.write("""
# Flight Price Predictor
""")

#Inputs
l1=['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai']
l3=['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
       'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia',
       'Vistara Premium economy', 'Jet Airways Business',
       'Multiple carriers Premium economy', 'Trujet']

def journey(): 
    source=st.selectbox('Source',l1)
    st.write('Your source is:',source)
    l2=[] 
    for i in l1:
        if i==source:
            pass
        else:
                l2.append(i)
    destination=st.selectbox('Destination',l2)
    st.write('Your destination is:',destination)
    
    return source, destination

col1, col2 = st.columns([6, 4])

with col1:
    date = st.date_input("Date of the Journey",
        datetime.date(2019, 7, 6))
    st.write('Date of the Journey:', date)
    source,destination=journey()


with col2:
    time_depart = st.time_input('Departure Time', datetime.time())
    st.write('Departure at:',time_depart)

    time_arr = st.time_input('Arrival Time', datetime.time())
    st.write('Arrival at:',time_arr )

    airline=st.selectbox('Select your airlines',l3)

    stops=st.selectbox('No of Stops',[x for x in range(0,5)])

#Duration
# st.subheader("Duration")
arr_delta= datetime.timedelta(hours=time_arr.hour, minutes=time_arr.minute, seconds=time_arr.second)
depart_delta= datetime.timedelta(hours=time_depart.hour, minutes=time_depart.minute, seconds=time_depart.second)
Duration = arr_delta-depart_delta
Duration_min  = (Duration.total_seconds() / 60)
# st.write("The duration of your flight(in min):",Duration_min)


#Dataframe
def model(date,time_depart,time_arr,stops,source,destination,airline):
  df=pd.DataFrame()
  df['Date_of_Journey']=[date]
  df['Departure_time']=[str(time_depart)]
  df['Arrival_time']=[str(time_arr)]
  df['Total_Stops']=[stops]
  df['Source']=[source]
  df['Destination']=[destination]
  df['Airline']=[airline]
  return df

df1=model(date,time_depart,time_arr,stops,source,destination,airline)


#Pre-processing
def predict(df1):
  cols=['Destination','Source','Airline']
  onehotencoder = OneHotEncoder()
  ohe=OneHotEncoder()
  transformed_data = onehotencoder.fit_transform(df1[cols])  #values extracted
  dfnew=pd.read_excel("Flight_price.xlsx")
  transformed_data1 = ohe.fit_transform(dfnew[cols])    #column names extracted
  df1[ohe.get_feature_names_out()]=0            
  df1[onehotencoder.get_feature_names_out()]=transformed_data.toarray().astype(int)
  for col in cols:
    df1=df1.drop(columns=col,axis=0)

  df1['Arrival_hr']=pd.to_datetime(df1['Arrival_time']).dt.hour
  df1['Arrival_min']=pd.to_datetime(df1['Arrival_time']).dt.minute
  df1['Dept_hr']=pd.to_datetime(df1['Departure_time']).dt.hour
  df1['Dept_min']=pd.to_datetime(df1['Departure_time']).dt.minute
  df1['Journey_day']=date.day
  df1['Journey_month']=date.month
  df1.drop(['Date_of_Journey'],axis=1,inplace=True)
  df1.drop(['Departure_time'],axis=1,inplace=True)
  df1.drop(['Arrival_time'],axis=1,inplace=True)
  
  df1['Duration(in min)'] = [Duration_min]

  X = df1.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dept_hr',
       'Dept_min', 'Arrival_hr', 'Arrival_min', 'Duration(in min)',
       'Destination_Banglore', 'Destination_Cochin', 'Destination_Delhi',
       'Destination_Hyderabad', 'Destination_Kolkata', 'Destination_New Delhi',
       'Source_Banglore', 'Source_Chennai', 'Source_Delhi', 'Source_Kolkata',
       'Source_Mumbai', 'Airline_Air Asia', 'Airline_Air India',
       'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
       'Airline_Jet Airways Business', 'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy']]
  fpp_model=pickle.load(open("model.pkl",'rb'))
  y_pred1= fpp_model.predict(X)

  return y_pred1
value=predict(df1)
#Building the model
# st.subheader("Price")
# st.write('The ticket cost:',value[0])


col1, col2 = st.columns([6, 4])
with col1:
    st.subheader("Duration")    
    st.write("The duration of your flight(in min):",Duration_min)
with col2:
    st.subheader("Price")
    st.write('The ticket cost:',value[0])

